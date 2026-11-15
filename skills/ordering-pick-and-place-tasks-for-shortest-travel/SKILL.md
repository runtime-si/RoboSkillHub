---
name: ordering-pick-and-place-tasks-for-shortest-travel
description: Orders pick-and-place pairs for the least total travel time
  (start -> pick_1 -> place_1 -> pick_2 -> ...) - exactly (Held-Karp dynamic
  programming) up to 12 tasks per arm, nearest-neighbour then 2-opt / Or-opt
  above - with precedence (a slot must be emptied before a part is placed in
  it), per arm when given scheduling-tasks-across-two-arms-for-minimum-makespan's
  by_arm queues. Travel is a lift-move-lower Cartesian estimate. Use after
  deciding which arm does which task, to cut the empty moves between tasks
  (sorting, clearing a table, refilling a rack).
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [planning, sequencing, tsp, pick-place, travel-time, cpu]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting, bimanual]
    provenance:
      authors: [runtime-si]
gap:
  allowed_tools: []
  required_inputs: {}
  produces_outputs:
    total_travel_s: float
    total_given_s: float
    method: string
    solve_s: float
  exit_conditions:
    ordered: Every queue was ordered; order lists each arm's task ids, travel_s the estimate (given_travel_s for the order as given, -1 if it broke precedence).
    infeasible: Precedence within a queue is cyclic.
  canonical_scripts:
    - order_tasks: scripts/order_tasks.py
  examples:
    - title: Order three pen moves, one after another's slot is free
      path: examples/canonical_subgraph.json
---

# ordering-pick-and-place-tasks-for-shortest-travel

## What it does

`tasks`: `[{id, pick, place, after: [ids]}]` (positions as Vec3 or poses).
`by_arm` (from scheduling-tasks-across-two-arms-for-minimum-makespan): each
arm's queue is reordered independently - precedence inside a queue is kept,
across queues it is the scheduler's (re-run the scheduler with the new
orders if cross-arm timing matters). `start`: `{arm: position}` of each
hand (else its first pick). Travel between two points: horizontal distance
plus rising to `lift` (0.10 m) over the higher one and descending, at
`speed` (0.25 m/s). Exact (Held-Karp over subsets, precedence as a subset
mask) up to `exact_max` (12) tasks per queue; above, nearest-neighbour then
2-opt segment reversals and Or-opt moves of 1-3 tasks until no gain (2 s
budget). Outputs `order`, `travel_s`, `given_travel_s`, totals, `method`,
`solve_s`.

## Checks

FakeContext unit tests (`tests/bundles/test_ordering_pick_and_place_tasks_for_shortest_travel.py`):
exact against brute force on three random 7-task sets, precedence kept and
optimal, the heuristic on 25 tasks, two-arm queues, cycles, bad inputs.

**Benchmarks** (`python -m tests.bundles._order_eval`, CPU, one arm starting
at (0, 0, 0.2)); estimated travel seconds:

| job | tasks | given order | nearest neighbour | this skill | method | solve |
|---|---|---|---|---|---|---|
| 12 pens A -> B, slot k to slot k | 12 | 65.4 | 65.0 | **64.9** (1 % less) | held-karp | 0.04 s |
| 12 pens, random target slots | 12 | 66.0 | 65.3 | **65.1** (1 %) | held-karp | 0.03 s |
| 12 pens, random slots, seed 1 | 12 | 65.9 | 65.5 | **65.0** (1 %) | held-karp | 0.03 s |
| sorting 10 to two bins | 10 | 45.7 | 42.8 | **42.5** (7 %) | held-karp | 0.01 s |
| sorting 12 | 12 | 57.9 | 54.4 | **54.0** (7 %) | held-karp | 0.03 s |
| sorting 30 | 30 | 139.1 | 123.1 | **122.7** (12 %) | heuristic | 0.38 s |

Pens between two holders gain ~1 %: every task carries a pen the full A->B
distance, which no order changes; only the empty returns (B -> next pick,
4-12 cm apart in a holder) are reordered. Sorting from scattered spots gains
7-12 % over the given order, and ~0.5 % over nearest neighbour. Not
measured in the twin.

## Limits

- Travel is a Cartesian estimate at constant speed, not the arm's joint-space
  time; it ranks orders, it does not predict seconds on the robot.
- Per-arm only: the other arm's timing and clearance are not considered.

## Provenance

New code (Held-Karp, 2-opt, Or-opt are textbook).
