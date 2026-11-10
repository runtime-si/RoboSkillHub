---
name: correcting-in-plane-offset-by-discrete-moves
description: Final x, y and yaw alignment of a held part (or the hand) over its
  target from the wrist camera, by a small classifier that looks at one
  cropped wrist frame and says which way to nudge - +/-x, +/-y, +/-yaw or stay
  per axis (27 classes); the hand steps 1 mm / 0.5 deg and repeats until three
  "aligned" in a row. The classifier is trained per target from images the arm
  labels itself (collect_offsets jogs from a taught aligned pose to random
  offsets and records each offset). Use when a part must sit over a slot, stud
  grid or rack within a millimetre and the target has no reference image or
  keypoints to servo on (textureless bricks, domino backs, slots).
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [alignment, visual-servo, learned, wrist-camera, insertion]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [insertion, lego, domino, jigsaw]
    provenance:
      authors: [Hadi Beyzaee Juneghani]
      code: https://github.com/cooper-group-uol-robotics/vision-guided-rack-insertion-ot2/blob/3bfe3fe87b64/reference_implementation/ot2_rack_insertion/cnn_model_misalignment_prediction.py
      upstream_license: MIT
gap:
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - move-classifier.predict
  required_inputs:
    model_path: string
    camera: string
  produces_outputs:
    decisions: int
    turned_deg: float
    last_label: string
    reason: string
  exit_conditions:
    aligned: The classifier said "000" (no move on any axis) streak (3) times in a row; the hand stays there for the next skill to lower / place.
    max_steps: max_decisions (80) decisions without three aligned in a row (dithering, or a target the model was not trained on).
    diverging: The hand drifted more than max_drift (25 mm / 10 deg) from where the loop started, or a step was refused.
    no_frame: The camera gave no RGB or move-classifier.predict raised; nothing was moved by this decision.
  canonical_scripts:
    - align: scripts/align.py
    - collect_offsets: scripts/collect_offsets.py
  examples:
    - title: Align over the target with a trained model
      path: examples/canonical_subgraph.json
---

# correcting-in-plane-offset-by-discrete-moves

This is the closed loop of vision-guided rack insertion (Liverpool, Opentrons
OT-2, 90.3 % insertion success upstream). It brings a held part, or the hand
itself, over its target using the wrist camera.

Upstream used ResNet18 with 27 classes. The classifier is
`tools/move-classifier`; it is trained per target on images the arm labels
itself, so no hand labelling is needed.

## Two scripts

1. **`collect_offsets`** collects training data. Start at the taught aligned
   pose. It jogs `samples` times to random offsets: +-12 mm in x and y,
   +-5 deg in yaw (upstream's ranges). Yaw is turned with the last joint.
   At each offset it saves the cropped wrist frame. The label is the hand's
   offset **read back** after the move (not the commanded one), written to
   `index.jsonl`. At the end it returns to the taught pose. Exits:
   `collected`, `move_refused`, `no_frame`.
2. **`align`** runs the loop. Each decision:
   - calls `move-classifier.predict` on the crop;
   - steps `step_m` (1 mm, upstream) in x / y and `step_yaw_deg` (0.5 deg)
     about +z, by the sign of each axis.

   It ends when three predictions in a row say "000" (upstream's rule).

Then train once: `move-classifier.train(dataset_dir, model_path)`.

Labels are world-frame signs per axis ("+x", "-y", "0" ...). Upstream's two
loops used "Move Up / Move Left" strings whose direction differed between
the loops, so world-frame signs avoid that trap.

`tolerance` is passed to the move tool only when set. **A servo connector's
default 4 mm arrival ball turns every 1 mm step into a no-op that reports
"reached".** That happened in the twin (80 decisions, no motion), so pass
e.g. `tolerance=0.0003`.

## Measured (YAM station twin, wrist D405 render)

**Setup:**
- Target: the station's open box, seen by the left wrist D405 rendered at
  the real station's calibrated mount. The aligned pose is the hand
  top-down, 10 cm over the box centre.
- Data: `collect_offsets` gathered 600 frames on two hosts (0 jogs refused).
  The read-back offset differed from the command by a median 1.5 mm in x
  (the default go_to_pose arrival ball), which is why labels are read back.
- Training: `move-classifier.train` for 20 epochs on 510 frames, 27 s on
  an RTX PRO 6000. On 90 held-out frames, all three axes were right 0.73 of
  the time; per axis, x 0.91, y 0.92, yaw 0.90.
- Trials: 7 closed-loop runs from random offsets (2.9-13.0 mm, up to 4.0
  deg), `tolerance` 0.3 mm, 60-decision cap. Errors are the hand's
  read-back against the taught pose.

| | trials | start error | end error | yaw at end |
| --- | --- | --- | --- | --- |
| no correction (open loop) | 7 | 2.9-13.0 mm (mean 8.1) | = start | up to 4.0 deg |
| this skill | 7 | same | **0.20-1.73 mm (median 0.52)** | **0.02-0.55 deg** |

**Exits:** 1 of 7 ended `aligned` (14 decisions, 0.26 mm / 0.54 deg). The
other 6 ended `max_steps` while already within 0.2-1.7 mm.

**Why the stop rule rarely fires:** only 7 of the 600 random jogs fell
inside the "000" dead-band (0.5 mm and 0.2 deg on all three axes at once).
The classifier therefore seldom says "000" three times in a row, and it
dithers one step about the goal. Collect extra samples near the aligned
pose (for example `max_offset_m=0.001`, `max_yaw_deg=0.3` for a quarter of
the set) before relying on the `aligned` exit. Until then, treat
`max_steps` with a small `moved_m` change as aligned.

**Not measured:**
- a held part;
- a LEGO, domino or jigsaw target;
- the real arm;
- upstream's LLaVA variant.

## Licence

Adapted from vision-guided-rack-insertion-ot2 (MIT, Copyright (c) 2026
Hadi Beyzaee Juneghani). See the script headers and NOTICE.
