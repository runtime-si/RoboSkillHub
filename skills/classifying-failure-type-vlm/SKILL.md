---
name: classifying-failure-type-vlm
description: Classifies why a manipulation step failed from camera frames
  taken before, during (optional, a short video as frames) and after it - asks
  a VLM RoboFAC's failure questions (detection, its six failure types,
  explanation, high- and low-level correction) together with a routing class
  and exits on the class - missed_grasp, slipped, wrong_object, misaligned,
  collision_or_stuck, incomplete, or no_failure - with the VLM's explanation
  and correction. Use when a graph knows (or suspects) a step went wrong and
  must pick the recovery that fits the cause, with only cameras, after
  capturing a robot.get_observation before the step.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [judge, failure-detection, failure-classification, vlm, before-after, recovery-routing]
  hub:
    stage: judge
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, failure-detection]
    provenance:
      authors: [runtime-si]
      code: https://github.com/MINT-SJTU/RoboFAC/blob/638eb5320d409e4072710a0db26f1c0d68cef3de/evaluation/eval.py
      upstream_license: MIT
      paper: https://arxiv.org/abs/2505.12224
gap:
  allowed_tools:
    - robot.get_observation
    - vlm.query
  required_inputs:
    before: Observation
    task: string
    step: string
  produces_outputs:
    failed: bool
    failure_class: string
    robofac_type: string
    misalignment: string
    explanation: string
    correction: string
    low_level_correction: string
  exit_conditions:
    no_failure: The VLM said the step succeeded (AFTER shows its intended result); the class is ignored.
    missed_grasp: The gripper closed but the object is not in it - closed beside it or on air, or pushed it away.
    slipped: The object was held at some point and is no longer in the hand - slipped or fell while lifting or carrying.
    wrong_object: The robot grasped, moved or holds an object other than the one the task names.
    misaligned: The right object, the step carried out, but the object or hand ended offset or turned from its target (placed beside the box, tilted); `misalignment` is translation / rotation when RoboFAC's type says which.
    collision_or_stuck: The hand or the object ran into something (a wall, a fixture, another object, the table) and stopped or jammed.
    incomplete: The step stopped short or part of it was never done, with no collision and no wrong object.
    no_verdict: No camera has RGB in both BEFORE and AFTER, vlm.query raised, the reply had no JSON, no success verdict, or a failure with neither a class nor a RoboFAC type that maps to one.
  canonical_scripts:
    - classify_failure_vlm: scripts/classify_failure_vlm.py
  examples:
    - title: Classify a failed step against an observation captured before it
      path: examples/canonical_subgraph.json
---

# classifying-failure-type-vlm

A failure judge: the step went wrong - how, and so which recovery?

## How a graph uses it

```text
before (tool: robot.get_observation) -> <the step> -> [check] -> classify (this skill)
   no_failure -> next step          missed_grasp -> re-perceive, regrasp
   slipped -> re-perceive, pick again   wrong_object -> release, re-perceive the named object
   misaligned -> re-place with low_level_correction   collision_or_stuck -> back off, replan
   incomplete -> resume / repeat the step              no_verdict -> a cheaper check or a human
```

Capture `before` with a `robot.get_observation` tool node just before the
step; `after` defaults to a fresh observation. `during` takes observations
captured while the step ran (a short clip as frames, e.g. one per segment of
a move); at most `max_during` (4) evenly spaced ones are sent. Each exit is
meant for its own recovery node; wire classes the graph cannot recover from
to a shared node.

## How it fits with the other two failure skills

| skill | stage | sees | answers | cost |
|---|---|---|---|---|
| `classifying-failure-from-proprioception` | verify | gripper opening, tool pose, joints | ok / stall / never_gripped / fumble / drop / stuck_mid_air | free, no camera |
| `judging-step-with-before-now-vlm` | judge | BEFORE / NOW frames | complete / in_progress / failed (+ wrong object / destination) | one VLM call |
| this skill | judge | BEFORE / DURING / AFTER frames | which failure, why, how to correct | one VLM call |

Run the proprioception check after every grasp, lift or move: it is free and
catches what the hand itself feels (an empty close, an opening in the air, a
stall). Use the before/now judge to decide whether a step is done, still
running or failed. Call this skill when either says the step failed (or when
a step's own exit reports a failure) and the graph needs to know *why*: it is
the only one that sees which object was taken, where the object went and
what the hand ran into. Proprioception can cheaply confirm a vision verdict
(`never_gripped` vs `missed_grasp`, `drop` vs `slipped`, `stall` vs
`collision_or_stuck`); this skill can name the cause the arm cannot feel
(`wrong_object`, `misaligned`).

## What it sends

One `vlm.query`: the task and the step, the image legend (per camera, BEFORE,
DURING i/k, AFTER; external cameras before wrist ones), then RoboFAC's
questions as its `eval.py` sends them to models other than RoboFAC-7B:

1. failure detection - "In the video, the robotic arm executes a task, can
   you determine whether it was successful?" (judged for the step);
2. failure identification - RoboFAC's options verbatim: Orientation
   deviation, Step omission, Wrong target object, Timing error, Grasping
   error, Position deviation;
3. the routing class, each described with the RoboFAC types it refines;
4. failure explanation, 5. high-level correction, 6. low-level correction
   (direction relative to the arm and how far), with RoboFAC's answer-length
   suffixes.

The reply is JSON: `success`, `robofac_type`, `failure_class`,
`explanation`, `correction`, `low_level_correction`.

## Exits and the taxonomy

RoboFAC's six types are causes in a planner's terms; the exits are what a
recovery needs to know, closer to the AHA failure modes (which are reference
only: AHA's code and data are under NVIDIA's non-commercial licence; nothing
of it is copied). `robofac_type` is returned beside the class. When the class
is missing or unknown the route falls back on RoboFAC's type where it is
unambiguous: Wrong target object -> wrong_object, Position / Orientation
deviation -> misaligned, Step omission -> incomplete. Grasping error and
Timing error split between missed_grasp and slipped, so they route
no_verdict without a class.

## Inputs

- `before`: an observation (cameras with `name` and `rgb`, a list or a
  mapping), captured before the step.
- `task`, `step`: the overall instruction and the step that was executed.
- `after` (fresh `robot.get_observation`), `during` ([]), `max_during` (4),
  `cameras` (all shared by BEFORE and AFTER).
- `crop` (none): a pixel box `[x1, y1, x2, y2]` cut from every frame, or a
  mapping camera name -> box. Pass the workspace's box for a distant overhead
  camera (see the live check).
- `model`: per-call model for `vlm.query`; otherwise the tool's configured one.

## Outputs

`route` (the exit), `failed`, `failure_class`, `robofac_type` (one of
RoboFAC's six options or empty), `misalignment` (translation / rotation /
empty), `explanation`, `correction`, `low_level_correction`, `observations`
(the VLM's one line per image of where the hand and the objects are, asked
first so it looks before it answers), `cameras`, `frames` (images sent),
`raw`.

## Live check

`run()` through the hub's `vlm.query` (OpenAI-compatible provider pointed at
Gemini's endpoint) on 8 known-answer episodes staged in a YAM station twin
(left arm, 30 mm dark cube, open 100 x 100 x 40 mm blue box, a second orange
cube for the wrong-object case), the twin's calibrated overhead camera
(640 x 480, the cube about 15 px across). Truth is what the cube did
(`sim.get_object_obb`):

| episode | step | what happened | truth |
|---|---|---|---|
| grasp_ok | grasp the cube, lift 10 cm | lifted 91 mm | no_failure |
| place_ok | carry the held cube over the box, release inside | on the box floor, 0.5 mm off its centre | no_failure |
| place_beside | same | released 90 mm off the box centre, on the table | misaligned |
| miss | grasp the cube, lift 10 cm | hand 40 mm beside it across the jaws; cube pushed 13 mm, not lifted | missed_grasp |
| drop | grasp the cube, lift 10 cm | lifted, the hand opened at the top; the cube fell back onto its spot | slipped |
| drop_carry | pick up and carry it toward the box | the hand opened 5 cm into the carry; cube on the table 51 mm from its start | slipped |
| wrong | grasp the dark cube, lift 10 cm | grasped and lifted the orange cube 92 mm | wrong_object |
| stall | push the cube 10 cm toward the box, closed hand | cube jammed against the box wall after 22 mm, the move timed out | collision_or_stuck |

`strip` sends BEFORE, 1-3 DURING frames and AFTER (3-5 images); `pair` only
BEFORE and AFTER. `crop` is a fixed 420 x 270 workspace box of the overhead
image ([40, 150, 460, 420]); `full` is the whole frame. One run per cell:

| model, frames | crop | full frame |
|---|---|---|
| gemini-robotics-er-2-preview, strip | **6/8** | 3/8 |
| gemini-robotics-er-2-preview, pair | 5/8 | 3/8 |
| gemini-3-flash-preview, strip | 4/8 | 4/8 |
| gemini-3-flash-preview, pair | **6/8** | 5/8 |

Confusion over the four cropped runs (32 verdicts):

| truth \ predicted | no_failure | missed_grasp | slipped | wrong_object | misaligned | collision_or_stuck |
|---|---|---|---|---|---|---|
| no_failure (8) | **7** | 1 | 0 | 0 | 0 | 0 |
| missed_grasp (4) | 0 | **4** | 0 | 0 | 0 | 0 |
| slipped (8) | 0 | 7 | **1** | 0 | 0 | 0 |
| wrong_object (4) | 0 | 0 | 0 | **4** | 0 | 0 |
| misaligned (4) | 0 | 0 | 0 | 0 | **4** | 0 |
| collision_or_stuck (4) | 2 | 0 | 0 | 0 | 1 | **1** |

Latency per call: median 4.9 s (ER-2 pair), 6.9 s (ER-2 strip), 6.0 s (flash
pair), 5.8 s (flash strip); 2.6-21.7 s overall, the slow ones flash replies
with long thinking. What to know:

- **Wrong object, missed grasp and misplacement are reliable**: 16/16
  wrong_object, 16/16 missed_grasp and 16/16 successful placements over all
  eight configurations, cropped or not; misaligned 4/4 cropped (uncropped,
  ER-2 read the hand hovering over the released cube as still holding it and
  said incomplete).
- **Slipped is not separable from missed_grasp from this camera.** In 15 of
  16 verdicts the drop was called a miss; the per-image `observations` we
  read describe the frame with the cube lifted inside the closed fingers as
  "the cube remains on the table". The one hit was ER-2
  with the cropped strip on drop_carry, where the cube landed away from its
  start. A wrist camera or the proprioception skill's jaw signal (`drop` vs
  `never_gripped`) settles it; route `missed_grasp` through that check when
  the difference matters.
- **Collision / stuck** came out only from flash with BEFORE / AFTER (2/2);
  ER-2 called the jammed push a success both ways (the cube did move toward
  the box) and flash with the strip once said misaligned and once no_failure.
- **Crop the overhead camera to the workspace**: it raised ER-2 from 3/8 to
  5-6/8 and flash pair from 5/8 to 6/8; the cube fills more of the image.
- **DURING frames helped ER-2 with the crop (5 -> 6/8) and hurt flash (6 ->
  4/8)**: with the strip flash read the successful grasp as a miss and the
  jammed push as misaligned.
- 4 of the 64 final replies (all flash, the 20 s ones) hit `vlm.query`'s
  4096-token cap (thinking counts) and were cut mid-JSON; the parser keeps
  the complete fields of a cut reply, so all four still routed (the
  explanation is lost when the cut falls inside it).
- Not staged: incomplete, and the two other episodes' variants (the shared
  GPU ran each episode for 4-60 minutes); incomplete is covered by unit tests
  only.

## Changes from RoboFAC

Adapted from `evaluation/eval.py` and `evaluation/eval_utils.py` (MIT) and
the question texts of RoboFAC's test QA (`MINT-SJTU/RoboFAC-dataset`,
`test_qa_sim`, MIT):

- RoboFAC asks each of its questions in a separate request about a whole
  video, one key frame per second resized to 512 x 512, with no task text
  (task identification is one of its questions). Here one request asks
  detection, identification, explanation and both corrections about the
  frames of one step, with the task and the step given, and asks for JSON.
- A routing class is asked beside RoboFAC's types and becomes the exit;
  RoboFAC's types are free text scored by substring match.
- No RoboFAC-7B: any model behind the hub's `vlm` tool (the RoboFAC-7B
  weights' licence was not checked and they are not used).
- The reply starts with one observation line per image, so the model locates
  the hand and the objects before it answers. Added during the live check;
  it did not measurably change the uncropped scores (before it, on the 7
  episodes staged then: ER-2 strip 4/7, flash pair 5/7), but it shows what
  the model saw, which is how the slipped / missed_grasp confusion was found.
- An optional crop of every frame (a distant overhead camera).
- A failed call, an unparseable reply or a reply without a verdict routes
  `no_verdict` (RoboFAC's client retried forever); a reply cut off mid-JSON
  keeps its complete fields.
