---
name: move-classifier
description: Trains and runs the 27-class "which way to nudge" classifier of
  vision-guided rack insertion (ResNet18, ImageNet weights, MIT) - from one
  cropped wrist image, the sign of the x, y (world) and yaw correction that
  brings a held part over its target. Training data is self-labelled by
  skills/correcting-in-plane-offset-by-discrete-moves (collect_offsets). Use
  when a per-target learned alignment cue is needed and a GPU is available for
  training (inference runs on CPU too).
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, alignment, learned, classifier, gpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    provenance:
      authors: [Hadi Beyzaee Juneghani]
      code: https://github.com/cooper-group-uol-robotics/vision-guided-rack-insertion-ot2/blob/3bfe3fe87b64/reference_implementation/vlm_cnn_train/train_cnn.py
      upstream_license: MIT
gap:
  requires: {gpu: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "move-classifier"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - move-classifier.train: Train the classifier on a folder of self-labelled crops (index.jsonl of measured offsets).
    - move-classifier.predict: Which way to nudge (x, y, yaw signs, world frame) from one cropped wrist image.
---

# move-classifier

The 27-class nudge classifier of vision-guided rack insertion
(`reference_implementation/vlm_cnn_train/train_cnn.py`, MIT), kept as
upstream wrote it:
- ResNet18 with ImageNet weights, a 1024-unit head and 27 outputs;
- 336 px input;
- ColorJitter / grayscale / blur augmentation;
- Adam 1e-4, StepLR (5, 0.1), 20 epochs, keeping the best validation loss.

Labels use upstream's dead-bands (0.5 mm, 0.2 deg). Changed: they are
world-frame signs ("+", "-", "0" for x, y and yaw about +z) instead of the
"Move Up / Rotate Clockwise" strings.

| tool | returns |
| --- | --- |
| `move-classifier.train(dataset_dir, model_path, epochs=20, ...)` | `{model_path, n_train, n_val, val_accuracy, val_axis_accuracy, class_counts, epochs, seconds}` |
| `move-classifier.predict(image, model_path)` | `{label, x, y, yaw, confidence, aligned}` |

`dataset_dir/index.jsonl` rows are `{image, dx, dy, dyaw_deg}`: the hand's
measured offset from the aligned pose (skills/correcting-in-plane-offset-by-discrete-moves
`collect_offsets` writes them).

**Measured in the twin:** 600 frames, 20 epochs in 27 s on an RTX PRO 6000.
On held-out frames, all three axes were right 0.73 of the time (x 0.91, y
0.92, yaw 0.90). The closed-loop numbers are in the skill.

No weights are shipped: a model is trained per target. Upstream did not
publish theirs.
