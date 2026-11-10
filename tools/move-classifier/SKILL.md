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

Draft.
