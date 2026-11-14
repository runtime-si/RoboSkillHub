---
name: scheduling-tasks-across-two-arms-for-minimum-makespan
description: Assigns a list of tasks to two arms and times them for the
  shortest total time (makespan) - each task with the arms that reach it, a
  duration per arm, predecessors, an optional same-arm link (a pick and its
  place) and a workspace region - so that the two arms are never working in
  regions closer than two YAM hand envelopes (tools/yam-spheres) at the same
  time. Exact MILP on scipy's HiGHS for up to 30 tasks, list scheduling
  otherwise, the better of the two when the MILP hits its time limit; per-arm
  queues and paired steps for running-two-skills-concurrently-on-two-arms.
  Use when a bimanual job (moving many pens between holders, sorting parts to
  bins, a build) should finish as fast as both arms allow.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [bimanual, scheduling, makespan, milp, task-assignment, cpu]
  hub:
    stage: plan
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [pick-place, sorting, bimanual]
    provenance:
      authors: [runtime-si]
gap:
  allowed_tools: []
  required_inputs: {}
  produces_outputs:
    makespan: float
    lower_bound: float
    method: string
    optimal: bool
    solve_s: float
  exit_conditions:
    scheduled: Every task has an arm and a start time; makespan is the last end; optimal says whether the MILP proved it (else the time-limited MILP's or list scheduling's better schedule).
    infeasible: A task no arm can do, a same-arm link no shared arm can do, cyclic precedence, or the MILP proved no schedule exists.
    timeout: The MILP found no schedule within time_limit; the greedy schedule is returned.
  canonical_scripts:
    - schedule_tasks: scripts/schedule_tasks.py
  examples:
    - title: A pick and its place on one arm
      path: examples/canonical_subgraph.json
---

# scheduling-tasks-across-two-arms-for-minimum-makespan

Generalises assigning-bricks-to-two-arms (APEX-MR's assignment MILP for
LEGO) to any tasks and adds timing.

## Inputs

`tasks`: each `{id, duration (seconds, a number or {arm: seconds}), arms
(the arms that reach it, default all), after: [ids], same_arm_as: id,
region}`. A region is `{min, max}` (a box), `{points, radius}` (the task's
key positions), a Vec3, or a name (same name = same region). `arms` ([0, 1]),
`clearance` (default twice the YAM hand envelope, 2 x 0.148 m from
tools/yam-spheres' link-6 spheres), `method` (`auto`, `milp`, `greedy`),
`time_limit` (20 s), `max_exact` (30).

## Model

Binary x[i, arm], start S[i], makespan C. Each task on one arm; C >= S + D;
predecessors finish first; same-arm links share the arm; for every pair an
order binary - on the same arm they never overlap, and if their regions are
closer than `clearance` they never overlap on any arms (time windows).
Minimise C (+1e-4 x sum of starts). Greedy: tasks in topological order,
longest first, each on the arm that ends it soonest, pushed past
conflicting windows.

## Outputs

`schedule` ([{id, arm, start, end}] by start), `by_arm`, `concurrent_steps`
([{arm_0: id, arm_1: id}] - each arm's next task, for
running-two-skills-concurrently-on-two-arms, whose clearance arbiter still
guards the actual motion), `makespan`, `lower_bound` (sum of each task's
best duration / 2), `method`, `optimal`, `solve_s`.

## Checks

FakeContext unit tests (`tests/bundles/test_scheduling_tasks_across_two_arms_for_minimum_makespan.py`).
Benchmarks (`python -m tests.bundles._schedule_eval`, CPU, time limit 10 s):
durations 3 s + 6 s per metre from the arm's base (bases at y = +-0.305 m),
reach 0.7 m; pens = 12 pick tasks at holder A and 12 place tasks at holder
B, each place after its pick on the same arm; sorting = objects at random
table spots to a bin on the left or right.

| job | tasks | this skill (method, solve) | greedy only | one arm | alternate arms | lower bound |
|---|---|---|---|---|---|---|
| 12 pens, holders 0.4 m apart | 24 | **78.5** (greedy beat the time-limited MILP, 10.1 s) | 78.5 | 146.0 | 79.1 | 64.4 |
| 12 pens, holders 0.2 m apart | 24 | 143.2 (MILP, 10.0 s) | 143.2 | 143.2 | 143.2 | 67.1 |
| sorting 10, seed 0 | 10 | **56.8** (MILP) | 62.8 | n/a | 56.8 | 31.4 |
| sorting 10, seed 1 | 10 | **49.8** (MILP) | 49.8 | n/a | 60.7 | 30.3 |
| sorting 10, seed 2 | 10 | **56.6** (MILP) | 62.1 | n/a | 62.1 | 31.0 |
| sorting 24 | 24 | **116.0** (MILP) | 121.5 | n/a | 127.0 | 71.7 |

Two arms halve the pens job against one arm (78.5 against 146 s) when the
holders are far enough apart for one arm to pick while the other places; at
0.2 m every pick conflicts with every place and no schedule beats one arm.
"One arm" is n/a for sorting because no single arm reaches both bins. The
MILP never proved optimality within 10 s (big-M), so `optimal` is false in
every row; it beat greedy by up to 10 % on sorting.

## Limits

- Regions are boxes or points: the conflict test is distance between them
  against two hand envelopes, not the arms' swept volumes (the forearms can
  meet outside the regions); run-time clearance is
  running-two-skills-concurrently-on-two-arms' job.
- Durations are given, not measured; the benchmark's are a model.
- Big-M MILP: proofs are slow past ~10 tasks; the time limit and greedy keep
  it bounded.

## Provenance

The task-assignment idea is APEX-MR's (via assigning-bricks-to-two-arms,
MIT); this formulation and code are new. The hand envelope comes from
tools/yam-spheres.
