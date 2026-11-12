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

`backend="azure"` asks Azure OpenAI's v1 endpoint through the Responses API
(`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`):
`vlm.query` pointed at that endpoint is refused with 400 (its chat payload
sets temperature and max_tokens, which the reasoning deployment rejects).

## Evaluation (2026-11-11)

Synthetic wrist close-ups (16.5 mm peg over an 18.2 mm square hole in a 3x3
lattice, D405 intrinsics, camera moving with the hand, CPU ray cast), the
same 10 random 2-10 mm start offsets (median 7 mm):

| model | final offset median / mean / max | < 1 mm | centred | calls | wall per case |
|---|---|---|---|---|---|
| Azure `forskillhub` (GPT, reasoning) | 0.99 / 3.23 / 24.1 mm | 5/10 | 9/10 | 8.0 | 77 s |
| Gemini 3.5 flash | 1.67 / 2.00 / 4.6 mm | 2/10 | 2/10 | 14.6 | 243 s |
| Gemini 3.8 flash | 2.38 / 2.43 / 3.4 mm | 0/10 | 0/10 | 16.0 | 563 s |
| geometric #230 (depth) | 0/10 aligned, all hole_not_seen | | | | |

Azure used about 4.1k input and 1.9k output (1.85k reasoning) tokens per
case. Of its 9 `centred` exits, 4 were under the 0.85 mm radial play. One
run kept answering -x and walked 24 mm off; `max_travel` (15 mm) now stops
that as `not_converging`.

On 13 real probe-insert-1 wrist frames, the answer's sign agreed with the
pens task's logged in_hole_check delta on 11/13 (Azure) and 4/12 (Gemini
3.8 flash, 1 unsure). That logged delta is itself in doubt, so this is not
a truth check.

## Limits

- The model judges a 0.5-1 mm offset of a tip it sees from above beside a
  hole it partly hides; that is where the answers become noise.
- One model call per step (about 10 s each for Azure and Gemini flash).
- "centred" is said too early: half of Azure's centred exits were still 1-1.8 mm off.
