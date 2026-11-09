---
name: tracking-gripper-opening-stall
description: Drives the jaws to a target gap (or fully closed) and watches the
  measured gap over a sliding window, reporting whether it reached the target,
  stopped wide of it while closing (something is between the fingers - with
  its width in millimetres, checked against the expected part width), jammed
  (an opening blocked, or no travel at all), or is still moving. Routes
  reached, stalled_on_object, unexpected_width, stalled_jammed, still_moving,
  refused or unknown. Use when a graph closes on a part and must know at once
  whether it closed on something of the right width (a brick across its 15.8 mm
  side, not its 31.8 mm one; one domino, not two), or must confirm a partial
  opening before reaching between tightly packed parts; reads only the
  gripper.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [gripper, stall, grasp-quality, failure-detection, width, proprioception]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, lego, domino, jigsaw, sorting]
    provenance:
      authors: [runtime-si]
      code: https://github.com/Open-RUA/uni-robot-agent/blob/a98934c6a1a16317821fed41a3b816b1d300b427/uni_robot_agent/tools/gripper/opening.py
      upstream_license: MIT
gap:
  requires: {connector: [robot.describe_gripper, robot.set_grip, robot.wait_steps]}
  allowed_tools:
    - robot.describe_gripper
    - robot.get_gripper
    - robot.set_grip
    - robot.close_gripper
    - robot.open_gripper
    - robot.wait_steps
  required_inputs: {}
  # Optional inputs: target_width_m (float) or close (bool), command (default true;
  # false only watches), expected_width_m, width_tolerance_m (0.004), tolerance_m
  # (0.002), progress_m (0.0005), window (8), max_samples (100), poll_steps (50),
  # poll_s (0.1), arm_id.
  produces_outputs:
    target_m: float
    start_m: float
    actual_m: float
    error_m: float
    object_width_m: float
    width_error_m: float
    samples: int
    reason: string
  exit_conditions:
    reached: The gap came within tolerance_m (2 mm) of the target. Closing fully and reaching it means the jaws closed on nothing.
    stalled_on_object: While closing, the gap stopped gaining (under 0.5 mm over 8 samples) wide of the target after moving at least 1 mm; object_width_m is the gap there, within width_tolerance_m of expected_width_m if one was given.
    unexpected_width: As stalled_on_object, but the gap is more than width_tolerance_m from expected_width_m (wrong part, two parts, an edge or corner grasp).
    stalled_jammed: The gap stopped short while opening, settled past the target, or never moved 1 mm from where it started.
    still_moving: max_samples readings without reaching or stalling.
    refused: The gripper command raised or reported failure.
    unknown: The gripper could not be read (no position); nothing was judged.
  canonical_scripts:
    - track_opening: scripts/track_opening.py
  examples:
    - title: Close on a part and check its width
      path: examples/canonical_subgraph.json
---

# tracking-gripper-opening-stall

Did the jaws get where they were sent, and if not, why not?

```text
approach -> track (this skill, close=True, expected_width_m) -> stalled_on_object: lift
                                                            -> reached: closed on nothing, re-perceive
                                                            -> unexpected_width: wrong part / edge grasp, open and retry
```

## What it does

1. Reads the hand (`robot.describe_gripper`: span and width fit) and the
   starting gap (`robot.get_gripper`; `width_m` if the connector reports it,
   else the read fraction, 1.0 = open, through the width fit, as
   verifying-grasps reads it).
2. Commands the target: `robot.close_gripper` for `close` or a target at the
   stop, `robot.open_gripper` for a target at the span, else
   `robot.set_grip(width_m=...)`. The target is clipped to the hand's range
   (upstream clipped to its calibrated range). `command=false` only watches.
3. Polls the gap (`robot.wait_steps(poll_steps)` between samples, or
   `poll_s` of wall clock where the connector cannot step) and runs
   upstream's `OpeningTracker` on it: **reached** within 2 mm of the target;
   **stalled** when the last 8 samples are full, not reached, and the oldest
   error minus the smallest later one is at most 0.5 mm.
4. Splits a stall: closing and stopped wide of the target after at least 1 mm
   of travel is **stalled_on_object** (the gap there is the object's width
   across the pads), or **unexpected_width** when it is more than 4 mm from
   `expected_width_m`; an opening that stopped short, a close that settled
   past the target, or no travel at all is **stalled_jammed**.

## Sim check

YAM station twin (left arm, linear_4310 jaws, 94 mm span), a 25 mm cube,
on the GCP pool. Per trial: park, reset the cube, bring the open hand down to
the cube's mid-height (on it, or 40 / 60 mm across the fingers), run this
skill with `close=True, expected_width_m=0.025` (0.040 for the wrong-width
trial), lift 8 cm, move 12 cm aside; truth from the cube's simulated pose.

| trial | route | gap | truth |
|---|---|---|---|
| pick (run 1, run 2) | stalled_on_object x2 | 26.7 / 26.5 mm (width error +1.7 / +1.5 mm) | cube lifted 79.5 / 79.6 mm |
| second pick (runs 1, 2) | stalled_on_object x2 | 26.7 / 26.5 mm | lifted 79.5 / 79.6 mm |
| pick told to expect 40 mm (runs 1, 2) | unexpected_width x2 | 26.7 / 26.5 mm (error -13.3 / -13.5 mm) | lifted 79.5 / 79.6 mm |
| aim 60 mm across the fingers (run 2, both sides) | reached x2 | 0.0 mm | cube not touched (lifted 0.0 mm) |
| aim 40 mm across, one side (run 1) | reached | 0.0 mm | not touched (the descent stopped 24 mm high on the cube) |
| aim 40 mm across, other side (run 1) | stalled_on_object | 26.3 mm | **the cube was grasped anyway** (lifted 91 mm): the wide pads caught it |

10/10 trials route as the cube's truth says; the stall is declared after the
first 8 samples (the twin's close_gripper returns settled). The 25 mm cube
reads 26.5-26.7 mm: the gap comes from the read fraction times the 94 mm span
(`width_fit.source: span`), so expect a ~1.5 mm bias on this hand, inside the
4 mm `width_tolerance_m`.

Open-air targets (hand clear of everything, `robot.set_grip` for partial
gaps): 80, 50, 20, 0 (close), 94 (open), 60 mm all routed `reached`, within
0.0-1.9 mm, after 1-2 samples.

Creeping jaws (a harness error of mine drove the hand into the table, so the
fingers dragged on it while moving, ~1-7 mm/s): with 10 sim steps per sample
(a 0.14 s window) an opening that crept 0.2 mm across the window was called
`stalled_jammed` ("did not move"); with the default 50 steps (0.7 s window) the
same creeping moves (94 -> 60 mm in 10 s toward a 50 mm target) routed
`still_moving` after 100 samples, which is what they were. A hand pressed into
the table with the fingers pinned shut routed `stalled_jammed` for every
opening target (0.0 mm travel), which is also what happened.

The jammed-while-opening case on a free hand (fingers blocked by a neighbour)
was not staged; it is covered by the unit tests only.

## Relation to other skills

- verifying-grasps (`verify_grip`): one reading after a close, held vs empty
  against the mechanical stop. This skill commands the gap, watches it over a
  window, and adds the width check, partial targets and the jammed case.
- classifying-failure-from-proprioception, detecting-contact-from-joint-effort:
  arm-side signals; this one reads only the jaws.

## Changes from uni-robot-agent

Adapted from `uni_robot_agent/tools/gripper/opening.py` (MIT, licence in
`LICENSE.uni-robot-agent`): `OpeningTracker`'s tolerance (2 mm), progress
(0.5 mm) and window (8) and its reached / stalled / unknown logic are
upstream's.

- The skill commands the gap and polls `robot.get_gripper` itself (upstream
  was fed arm metadata by its environment loop); the metric gap comes from
  `robot.describe_gripper`'s width fit instead of upstream's
  `affine_pad_gap_v1` calibration record.
- A stall is split into stalled_on_object / unexpected_width / stalled_jammed
  by direction, travel and the expected width; upstream reported only
  "stalled".
- Samples are spaced in time (50 sim steps, 0.1 s on a real station) so the
  8-sample window spans ~0.7 s: with 10 steps per sample, jaws creeping
  against friction read as "did not move" (sim, above).
- A window that never settles routes still_moving (upstream stayed pending).
