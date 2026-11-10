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

Draft; measured numbers below.
