---
name: aligning-a-held-tip-by-vlm-relative-nudges
description: Puts the tip of a held part - a pen, a peg - over a hole by
  asking a VLM only which way to move, never where things are - each step
  draws +x / +y arrows (the base axes' image directions at the tip) on the
  wrist close-up(s), asks "+x, -x, +y, -y or centred?", moves the hand a step
  that starts at 4 mm and halves on every reversal along that axis, and stops
  on "centred" twice or both steps at 0.5 mm. Use before lowering a held part
  into a hole when no depth-based aligner reads the hole, as a relative check
  that needs no calibration of the model's absolute pointing.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [insertion, visual-servoing, vlm, wrist-camera, pen, peg-in-hole]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [insertion, pick-place]
    provenance:
      authors: [runtime-si]
      upstream_license: original (runtime-si)
gap:
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose
    - vlm.query
  required_inputs:
    tip_offset_m: float
  produces_outputs:
    steps: int
    calls: int
    moved_x: float
    moved_y: float
  exit_conditions:
    centred: The model answered centred twice in a row, or both axes' steps halved down to min_step_m (0.5 mm).
    not_converging: No centred verdict within max_steps (16) model calls.
    no_verdict: No wrist frame with rgb, intrinsics and pose, or two answers in a row without a direction (unsure, refusal, prose).
  canonical_scripts:
    - nudge_tip: scripts/nudge_tip.py
  examples:
    - title: Nudge a held pen over its hole
      path: examples/canonical_subgraph.json
---

# aligning-a-held-tip-by-vlm-relative-nudges

## How

1. Tip = tool position + approach axis x `tip_offset_m`. For each named
   camera (default: the first wrist camera; pass the left wrist too when it
   is posed beside the hole), the 2x2 Jacobian of the tip's pixel w.r.t. a
   base (x, y) move gives where +x and +y point in that image; the arrows are
   drawn from the tip's pixel.
2. `ask_tool` (default `vlm.query`, `prompt` + `images`) gets one word:
   +x, -x, +y, -y, centred or unsure.
3. Per axis, the step starts at `start_step_m` (4 mm) and halves when the
   answer reverses the last direction on that axis. The hand moves by
   `robot.go_to_pose` at the same height and rotation.

## Evaluation (2026-11-11, see the PR)

Synthetic wrist close-ups (16.5 mm peg over an 18.2 mm square hole in a 3x3
lattice, D405 intrinsics, camera moving with the hand), random 2-10 mm
start offsets, Gemini 3.x flash: see the PR body for the numbers. On real
probe-insert-1 frames the model's answers are not checked against truth.

## Limits

- The model judges a 0.5-1 mm offset of a tip it sees from above beside a
  hole it partly hides; that is where the answers become noise.
- One model call per step (5-15 s each with Gemini flash).
