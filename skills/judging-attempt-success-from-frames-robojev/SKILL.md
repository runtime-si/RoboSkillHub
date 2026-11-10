---
name: judging-attempt-success-from-frames-robojev
description: Decides whether a step succeeded from cameras alone with one
  calibrated number - the task sentence, the attempt's frames from a main
  camera (a recorded clip, or the observation captured before the step and
  the one now) and the final frame of the other cameras go to RoboJev-27B
  (tools/robojev), which returns P(done); at 0.7 or more the route is
  success, at 0.3 or less failure, between uncertain. Use when a graph must
  verify a whole step (a cube in a box, a brick placed, a piece seated)
  before moving on, without a task-specific check, or to grade recorded
  attempts.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [judge, success-detection, verification, vlm, calibrated]
  hub:
    stage: judge
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, assembly, sorting]
    provenance:
      authors: [runtime-si]
      code: https://github.com/josefchen/robojev/blob/79a5e2f/robojev/robojev.py
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2610.33718
gap:
  allowed_tools:
    - robot.get_observation
    - robojev.judge
  required_inputs:
    task: string
  produces_outputs:
    p_done: float
    main_camera: string
    n_frames: int
  exit_conditions:
    success: P(done) >= success_at (0.7).
    failure: P(done) <= failure_at (0.3).
    uncertain: P(done) between failure_at and success_at - RoboJev's calibration was fitted on SO-101 runs; treat as unknown (look again, ask a second judge, or a human).
    no_frames: Fewer than two frames of the main camera (no `frames`, and no `before` with that camera), or no camera with RGB.
  canonical_scripts:
    - judge_attempt: scripts/judge_attempt.py
  examples:
    - title: Judge a step from the observation captured before it and the cameras now
      path: examples/canonical_subgraph.json
---

# judging-attempt-success-from-frames-robojev

The end of a step is where a graph either moves on or recovers. RoboJev is a
27B VLM trained for exactly that question, with a calibration that makes
its answer a probability; this skill feeds it what a graph has.

## How it composes

- **Before it**: the step. Capture `robot.get_observation` before the step
  as `before` (judging-step-with-before-now-vlm's convention), or record the
  attempt yourself and pass `frames` (the main camera's RGB frames, start to
  end - ten are taken evenly; more in between help, RoboJev reads ten).
- **After it**: route on `success` / `failure`; `uncertain` is for a second
  look (another judge, a task-specific check such as
  verifying-placement-into-container, or a human). `p_done` for logging or
  as a reward.

## What it does

1. **Main camera**: `camera`, else the first non-wrist camera with RGB in the
   observation now (the overhead one on the YAM station), else a wrist one.
2. **Clip**: `frames` (images or CameraFrames, start to end), else that
   camera's frame in `before` and now - a two-frame clip, which tools/robojev
   pads to one start and nine ends (RoboJev's client would send five and
   five, which read as "not done" on every successful attempt; Checks).
3. **Extra views**: the last frame of every other camera now (`views` names
   them; `use_views: false` none). RoboJev was trained with the final views
   of other cameras appended.
4. `robojev.judge` and the thresholds.

## Inputs

`task` (required: the instruction as one sentence, e.g. "Put the cube in the
box."), `frames`, `before`, `camera`, `views`, `use_views` (true),
`success_at` (0.7), `failure_at` (0.3).

## Outputs

`p_done`, `main_camera`, `views`, `n_frames`, `detail`.

## Checks

CPU unit tests (`tests/bundles/test_judging_attempt_success_from_frames_robojev.py`,
17 with the tool's; the model faked): the prompt, frame picking, the
calibrated read-out, the vLLM path, and the skill's frame sources, camera
choice, views, routes and bad inputs.

**Twin attempts** (YAM station twin, left arm, GCP RTX PRO 6000; harness not
in the hub): 24 scripted "put the cube in the box" attempts, 12 waypoint
frames each from the calibrated overhead camera and the wrist camera; 10
succeed, 14 fail on purpose - grasp 5 cm off (3), released beside the box (3),
dropped on the way (3), released on the rim (3), never closed the hand (2). The
label is the simulator's (cube resting inside the box). Scored with
`python -m tests.bundles._robojev_twin_eval DIR` (real RoboJev-27B, 0.7 s an
answer). Baseline: the hub's judging-step-with-before-now-vlm with Gemini
Robotics-ER 2 as `vlm.query` (BEFORE = first frames, NOW = last).

| input | RoboJev: right at 0.5 | RoboJev P(done), successes / failures | RoboJev routes (0.7 / 0.3) | judge_step + Gemini-ER |
|---|---|---|---|---|
| all 12 overhead frames + wrist end | 24 / 24 | 0.991-1.000 / <= 0.015 | 24 right, 0 uncertain | - |
| start + end overhead + wrist end | 24 / 24 | 0.973-0.98 / <= 0.013 | 24 right, 0 uncertain | 24 / 24 complete vs not; 13 of 14 failures `in_progress`, 1 `failed` |
| all frames, no wrist view | 24 / 24 | >= 0.980 / <= 0.011 | 24 right | - |
| successful attempts cut at the lift (not done) | 9 / 10 | max 0.73 | 7 failure, 2 uncertain, 1 success | 10 / 10 `in_progress` |

The start / end row uses this bundle's padding (one start, nine ends); with
RoboJev's own even spread (five and five) every success read as not done
(P 0.007-0.012, 14 / 24 right). The cost: a short clip cut before the
outcome is padded with its last frame too, which made one lift-only clip look
done (P 0.73); with the even spread all ten were <= 0.025. Record more than
two frames where it matters. On this (easy) task Gemini through judge_step
is as accurate on done-vs-not, but calls nearly every failure `in_progress`
(continue) rather than `failed`; RoboJev gives a calibrated number locally in
0.7 s with no API.

## Limits

- **Twin, not the real station.** The attempts are the YAM station twin's
  renders (its calibrated overhead camera and a wrist D405 view), one task.
  No labelled real YAM attempts exist on disk.
- **Calibration**: fitted on SO-101; refit `success_at` / `failure_at` on ~50
  labelled YAM steps before trusting the probability as such.
- A GPU with ~56 GB for tools/robojev; 0.7 s an answer here.
- The answer is about the end state the frames show: a clip that stops
  before the outcome reads as not done.

## Provenance

The protocol (frames, views, calibrated read-out) is RoboJev's, run by
tools/robojev, which adapts its client (github.com/josefchen/robojev,
Apache-2.0; see that bundle and NOTICE). This script's frame sources, camera
choice and the three-way decision are new.
