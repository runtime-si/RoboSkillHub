---
name: training-a-part-detector-from-auto-labels
description: Gets a fast station-specific detector for a fixed set of parts
  (bricks, dominoes, jigsaw pieces) without hand labelling. An
  open-vocabulary teacher (mm-grounding-dino by default, through
  sahi.detect_sliced for 10-30 px parts) boxes a folder of station frames
  from a caption ontology ("lego brick" -> brick, "domino" -> domino). The
  boxes become a COCO dataset with a train / valid split, the labels are
  checked (per-class counts, empty frames), and a small RF-DETR student is
  fine-tuned on them with tools/rfdetr. Routes detector_trained,
  too_few_labels, label_quality_low or training_failed. Use when a closed
  loop needs a detector for a fixed set of parts and the open-vocabulary
  detectors are too slow (0.3-2 s a frame) - once per station and part set,
  again when new parts arrive; the student runs in 7 ms.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, detection, training, auto-labelling, distillation, lego, domino, jigsaw]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting]
    tuned_on: [yam-station-orbbec-overhead, yam-left-wrist-d405]
    provenance:
      authors: [runtime-si]
      code: https://github.com/autodistill/autodistill/tree/838e1c39f9fd
      upstream_license: Apache-2.0
      paper: "autodistill (Roboflow): CaptionOntology, base_model.label, target_model.train; reimplemented, no code copied"
gap:
  allowed_tools:
    - sahi.detect_sliced
    - mm-grounding-dino.detect
    - rfdetr.train
  required_inputs:
    frames_dir: string
    ontology: string
    out_dir: string
  # Optional: teacher (a detector tool name, default mm-grounding-dino.detect), slice_px (320; 0 = whole frame),
  # min_score (0.25), min_labels (50 per class), max_empty (0.5), valid_fraction (0.15), size (nano | small |
  # medium | large), epochs (40), resolution (640), seed, image_paths (a list instead of frames_dir).
  # ontology: a JSON object string {"prompt": "class", ...} (or a dict when wired from a script).
  produces_outputs:
    model_path: string
    n_images: int
    n_empty: int
    median_score: float
    dataset_dir: string
  exit_conditions:
    detector_trained: The teacher's labels passed the checks, the dataset was written to out_dir/dataset and rfdetr.train wrote the model directory model_path (use it with rfdetr.detect).
    too_few_labels: Some class has fewer than min_labels (50) teacher boxes over all frames; counts says which; add frames that show it (or lower min_score).
    label_quality_low: The teacher found nothing in more than max_empty (50 %) of the frames - wrong prompts, wrong scale or frames without the parts; nothing was trained.
    training_failed: rfdetr.train raised (no GPU, out of memory, a broken dataset) - the on_error route; the dataset stays in out_dir/dataset.
  canonical_scripts:
    - train_from_auto_labels: scripts/train_from_auto_labels.py
  examples:
    - title: Distil a brick / domino detector from a folder of station frames
      path: examples/canonical_subgraph.json
---

# training-a-part-detector-from-auto-labels

autodistill's recipe (an ontology of captions, a big model labels, a small
model trains) reimplemented over the hub's tools, with two changes:
- the target model is RF-DETR (Apache-2.0), not autodistill's YOLOv8 /
  ultralytics target (AGPL);
- the teacher runs through sahi, because the parts are 10-30 px in the
  overhead camera.

```python
r = run(
    ctx,
    frames_dir="/data/station_frames",
    ontology='{"lego brick": "brick", "domino": "domino"}',
    out_dir="/data/parts_detector",
)
if r["route"] == "detector_trained":
    dets = ctx.tool("rfdetr.detect", image=rgb, model_path=r["model_path"], threshold=0.3)[
        "detections"
    ]
```

1. Every frame (`.png .jpg .jpeg .bmp`) goes to
   `sahi.detect_sliced(prompts=ontology keys, backend=teacher, slice_px, min_score)`.
   Boxes of a known prompt take that prompt's class; boxes under 2 px are
   dropped.
2. Checks: more than `max_empty` of the frames empty -> `label_quality_low`;
   any class under `min_labels` -> `too_few_labels`.
3. COCO dataset in `out_dir/dataset`: train / valid (the valid split is also
   written as test). Category ids are 0..K-1 in class-name order; the
   images are copied.
4. `rfdetr.train(dataset_dir, out_dir/model, size, epochs, resolution)`.

Pick the teacher and its settings from tools/README.md for the parts and
view. Look at a sample of the labels before trusting the student: it learns
the teacher's misses and false positives (see below).

## Measured

2026-11-09 on a GCP RTX PRO 6000 (rshgpu5).

**Training data.** 480 frames: 320 overhead and 160 wrist renders of LEGO /
DUPLO / dominoes. They are composited on real station frames through the
station's calibrated cameras, from tools/README.md's generator with seed
100, so they are disjoint from the eval set (seed 7). Teacher:
mm-grounding-dino large (transformers 4.57.1) through sahi at 320 px,
`min_score` 0.25. Ontology: `lego brick` -> brick (LEGO and DUPLO),
`domino` -> domino.

**The skill end to end** (rfdetr.train served from its own environment):
- route `detector_trained`;
- 1099 brick and 697 domino labels, 22 of 480 frames empty, median teacher
  score 0.39;
- 3360 teacher calls; labelling took about 28 min with the GPU shared by
  another training; that run trained 2 epochs.
- The same dataset was then trained for 40 epochs through the same
  `rfdetr.train` (Nano, 640 px): 9.6 min.

**Eval** on tools/README.md's set, which the student never saw. Found =
share of parts with a box of the right class at IoU >= 0.5; FP per frame.

| detector | overhead, whole frame: LEGO / DUPLO / domino, FP | overhead, sahi 320 | wrist: LEGO / DUPLO / domino, FP | close domino renders (another renderer) | time a frame |
| --- | --- | --- | --- | --- | --- |
| **student from auto-labels**, 0.3 | **0.39 / 0.66 / 0.34**, 3.0 | 0.45 / 0.77 / 0.28, 4.4 | **0.85 / 1.00 / 0.62**, 1.8 | 0.37, 5.9 | **7 ms** (45 ms sliced) |
| teacher (mm-grounding-dino large), 0.2 | 0.37 / 0.52 / 0.30, 3.0 | 0.58 / 0.82 / 0.50, 5.7 | 0.62 / 0.86 / 0.65, 0.9 | 0.65, 2.0 | 331 ms (1.9 s sliced) |
| student from the true boxes (upper bound), 0.3 | 1.00 / 1.00 / 0.98, 0.1 | 1.00 / 1.00 / 0.98, 0.5 | 0.97 / 1.00 / 1.00, 0.1 | 0.79, 0.4 | 7 ms |

- **On whole frames the student matches or beats its teacher at about 1/50
  of the time**: LEGO +0.02, DUPLO +0.14, domino +0.04 overhead at the
  same false-positive rate, and wrist LEGO +0.23. It learned the parts
  from many frames rather than copying the teacher frame by frame. The
  teacher keeps the lead when it is sliced (it then costs 1.9 s a frame).
- **The labels are the limit.** The same student trained on the true boxes
  finds nearly everything on the same renderer. Correcting a few hundred
  teacher boxes by hand is the cheap next step.
- **Out of domain it falls behind.** On the close domino renders of another
  renderer (RoboSimStudio lines and grids) it found 0.37 against the
  teacher's 0.65. Train on frames from the camera and parts it will see.
- Renders, not photographs: rerun on real station frames when there are
  some.

Reproduce: `tests/bundles/_target_parts_gen_scenes.py` (seed 100, 320 / 160)
for the frames, this skill with a ctx serving sahi, mm-grounding-dino and
rfdetr, then `tests/bundles/_rfdetr_student_eval.py model_dir scenes_dir 0.3`.
