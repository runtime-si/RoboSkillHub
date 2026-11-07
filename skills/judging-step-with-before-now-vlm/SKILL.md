---
name: judging-step-with-before-now-vlm
description: Judges the current subgoal of a task from camera frames taken
  before the step and now - sends the BEFORE and NOW frames of each camera,
  the task, the subgoal and the subgoals after it to a VLM that returns a
  status (complete / in_progress / failure) and an action (next / continue /
  replan / grasp_tool / place_tool), and routes on the action. Use when a
  graph must decide after (or during) a step whether to advance, keep going,
  replan, or recover a wrong object or a wrong destination, with only cameras,
  after capturing a robot.get_observation before the step.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [judge, success-detection, failure-detection, vlm, before-after]
  hub:
    stage: judge
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting]
    provenance:
      authors: [runtime-si]
      code: https://github.com/NVlabs/VoLoAgent/blob/b4e623079ca8498a16bcd5016920d71f76c44d30/vlm_orchestrator/failure_handlers/vlm.py
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2606.07723
gap:
  allowed_tools:
    - robot.get_observation
    - vlm.query
  required_inputs:
    before: Observation
    task: string
    subgoal: string
  produces_outputs:
    status: string
    action: string
    reason: string
    grasp_target: string
    place_destination: string
    place_held_object: string
  exit_conditions:
    complete: The VLM chose "next" - clear visual evidence in NOW that the subgoal is done (its object at its destination).
    in_progress: The VLM chose "continue" - not done yet but still achievable, including not started; also any action outside recovery_mode's vocabulary.
    failed: The VLM chose "replan" - dropped, stuck, no visible progress, or the plan no longer fits the NOW scene; `reason` says which.
    wrong_object: The VLM chose "grasp_tool" - the hand holds or is closing on an object other than the subgoal's; `grasp_target` names the right one.
    wrong_destination: The VLM chose "place_tool" - the held object is heading to or was released at another destination; `place_destination` is a freeform phrase for the right one.
    no_verdict: No camera has RGB in both BEFORE and NOW, vlm.query raised, or its reply had no JSON; nothing was judged (VoLoAgent continued in this case).
  canonical_scripts:
    - judge_step: scripts/judge_step.py
  examples:
    - title: Judge a subgoal against an observation captured before the step
      path: examples/canonical_subgraph.json
---

# judging-step-with-before-now-vlm

A step judge: did the subgoal happen, is it still happening, or did it fail?

## How a graph uses it

```text
before (tool: robot.get_observation) -> <the step: skills / policy> -> judge (this skill)
```

Capture `before` with a `robot.get_observation` tool node just before the
step, run the step, then route on this skill's exit. `now` defaults to a
fresh `robot.get_observation`; pass it to judge a frame you already have.
Inside a running policy it can also be called periodically (VoLoAgent checked
every 80 sim steps) with `before` held from the start of the subgoal.

## What it sends

One `vlm.query` with VoLoAgent's monitoring prompt followed by its task text:

```text
Overall task: "<task>"
Current subgoal (<i>/<n>): "<subgoal>"
Remaining subgoals after this:   (or "This is the LAST subgoal.")
  <i+1>. ...
Assess the current subgoal status and decide the action.
```

and the frames in VoLoAgent's order, BEFORE then NOW, external cameras before
wrist ones, each listed by number and camera name in the prompt. `cameras`
picks frames by name; by default every camera with RGB in both observations.

The prompt keeps VoLoAgent's rules: "complete" only on clear visual evidence,
not-started counts as in progress, "when in doubt, in_progress / continue",
do not replan just because the scene changed, and the wrong-object
(`grasp_tool`) vs wrong-destination (`place_tool`) rules with their phrasing
rules for `place_destination`.

## Exits

The route follows the action, as VoLoAgent acted on the action token; the
status is reported beside it (`status` can disagree, e.g. failure + continue).

| action | exit |
|---|---|
| next | `complete` |
| continue | `in_progress` |
| replan | `failed` |
| grasp_tool | `wrong_object` (`grasp_target`) |
| place_tool | `wrong_destination` (`place_destination`, `place_held_object`) |
| - | `no_verdict` |

VoLoAgent's taxonomy does not split dropped / stuck / no progress into
separate tokens; they all arrive as `replan` with the cause in `reason`, so
they share the `failed` exit here.

## Inputs

- `before`: an observation (cameras with `name` and `rgb`, as a list or a
  mapping), captured before the step.
- `task`, `subgoal`: the overall instruction and the current subgoal.
- `remaining_subgoals` ([]), `subgoal_index` (1), `subgoal_count` (index +
  remaining): the plan around the subgoal, for the task text.
- `now` (fresh `robot.get_observation`), `cameras` (all shared).
- `recovery_mode` (`replan_tools`): VoLoAgent's action vocabularies,
  `replan`, `replan_grasp`, `grasp`, `place`, `replan_place`, `tools`,
  `replan_tools`. Upstream defaulted to `replan`; the default here offers every
  action so every exit is reachable. Restrict it when the graph has no
  recovery for an exit: an action outside the vocabulary becomes `continue`.
- `stack_mode` (false): adds VoLoAgent's `place_stack` block and passes
  `place_stack` through.
- `model`: per-call model for `vlm.query`; otherwise the tool's configured one.

## Live check

`run()` through the hub's `vlm.query` (OpenAI-compatible provider pointed at
Gemini's endpoint), default `recovery_mode`, one overhead frame per side.
Six pairs from a YAM station twin's calibrated overhead camera (640 x 480)
around a `pushing-objects` push of a 30 mm cube (it moved 84 mm along the
push, 7 mm sideways), and two from the real station's overhead camera:

| pair (BEFORE -> NOW) | subgoal | expected | gemini-3-flash-preview (2 runs) | gemini-robotics-er-2-preview |
|---|---|---|---|---|
| sim: start -> start, arm idle | push the cube 8 cm away from the arm | in_progress | in_progress, in_progress | in_progress |
| sim: start -> hand 12 cm over the cube | same | in_progress | in_progress, in_progress | in_progress |
| sim: start -> pushed, hand still 8 cm over the cube | same | complete | in_progress, in_progress | in_progress |
| sim: start -> pushed, hand moved clear | same | complete | complete, no_verdict | complete |
| sim: start -> pushed, hand clear | put the cube in the blue box | not complete | in_progress, in_progress | in_progress |
| sim: start -> pushed, hand clear | push the cube right, toward the box | not complete | in_progress, in_progress | in_progress |
| real: cube beside the box -> same, box nudged | put the cube into the grey box | not complete | in_progress, in_progress | in_progress |
| real: cable scene -> same scene, minutes later | pick up the cable's free end | in_progress | in_progress, in_progress | in_progress |

7/8 and 6/8 for flash, 7/8 for ER-2, 2-4 s per judgment. What to know:

- **Judge after the hand has moved clear.** With the hand still over the
  pushed cube both models said "in_progress" (3/3 more flash repeats too):
  the hand hides the cube and looks mid-push, and the prompt only allows
  "complete" on clear evidence. The same push judged with the hand moved
  away was "complete" every time it parsed (4/4 flash, 1/1 ER-2).
- **No false "complete"** in 24 judgments, including the two pairs where the
  cube moved but not to the subgoal's destination. Neither model chose
  `failed` / `wrong_destination` for the wrong-direction push: the cube was
  still pushable, and VoLoAgent's rules say in_progress when in doubt. The
  stated reasons there were wrong ("pushing it toward the blue box"), so read
  `in_progress` as "not done", not as "on track".
- One flash reply (6.2 s, the other replies took 2-4 s) did not parse and
  routed `no_verdict`; three repeats of that pair parsed. The cause was not
  captured; a likely one is Gemini 3's thinking tokens counting against
  `vlm.query`'s 1024-token cap.
- The failure exits (`failed`, `wrong_object`, `wrong_destination`) were not
  reached live: the twin has one graspable object and nothing to drop it
  from; they are covered by the unit tests only.

## Changes from VoLoAgent

Adapted from `vlm_orchestrator/failure_handlers/vlm.py` (prompt, parser) and
the BEFORE / NOW labelling of `strategies/subgoal_base.py`, Apache-2.0:

- One judgment per call; the graph decides when to judge (upstream: a
  handler gated every 80 steps).
- BEFORE is the observation before the step, not the episode start; the
  prompt's first bullet says so.
- The hub's `vlm.query` has no system prompt and puts images after the text,
  so the system prompt and the task text are one prompt, and the images are
  listed by number with their BEFORE / NOW label instead of interleaved.
- The mirrored-front-camera block is dropped (no such camera on a YAM).
- Actions become graph exits instead of tool activations; a failed call or an
  unparseable reply routes `no_verdict` instead of silently continuing (wire
  it to the `in_progress` edge for upstream behaviour).
