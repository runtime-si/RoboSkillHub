---
name: rfdetr
description: RF-DETR (Roboflow, Apache-2.0) train-and-run tools for a small
  station-specific detector -- fine-tune Nano / Small / Medium / Large from
  its COCO checkpoint on a COCO-format folder (e.g. one written by
  training-a-part-detector-from-auto-labels), then detect at 7 ms a
  640 x 480 frame with labelled boxes in grounding-dino's shape. Use when an
  open-vocabulary detector is too slow or too noisy for a fixed set of parts
  (bricks, dominoes, pieces) seen by the station's own cameras.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, detection, training, real-time, gpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting]
    weights:
      uri: https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth
      license: Apache-2.0 (rfdetr 1.12.0's Nano COCO checkpoint, md5 fb6504cce7fbdc783f7a46991f07639f, downloaded by the package; Small / Medium / Large likewise; the PML-1.0 sizes of rfdetr[plus] are not used)
    provenance:
      authors: [Roboflow]
      code: https://github.com/roboflow/rf-detr/tree/1.12.0
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2511.09554
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "rfdetr"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - rfdetr.train: Fine-tune RF-DETR (Apache-2.0 sizes) on a COCO-format dataset folder; returns the model directory.
    - rfdetr.detect: Run a trained RF-DETR model on an image (labelled boxes + scores, grounding-dino's shape).
---

# rfdetr

The `rfdetr` package (1.12.0, Apache-2.0) behind two tools. This bundle copies none of its code;
it calls `RFDETRNano(...).train(...)` / `.predict(...)` and adapts inputs and outputs.

| tool | returns |
| --- | --- |
| `rfdetr.train(dataset_dir, out_dir, size="nano", epochs=40, batch_size=8, resolution=640, lr=1e-4)` | `{model_path, classes, n_train, n_valid, epochs, seconds}` |
| `rfdetr.detect(image, model_path, threshold=0.5, query=None)` | `{detections: [{box, label, score}]}` best-first; `query` (`"brick. domino"`) keeps those classes, so it works as a sahi backend |

- `dataset_dir`: `train/` and `valid/` with images and `_annotations.coco.json`; category ids
  0..K-1 in name order (what the skill writes); `label` is the category name.
- `resolution` 640 (a multiple of 32) keeps 10-20 px parts of a 640 x 480 frame at their size;
  the Nano default, 384, shrinks them.
- `model_path` is a directory: `checkpoint_best_total.pth`, `classes.json`, `train.json`.
  `detect` loads it once and caches it.
- Only the Apache-2.0 sizes are accepted (`nano`, `small`, `medium`, `large`).

## Measured

2026-11-09, GCP RTX PRO 6000 (rshgpu5), rfdetr 1.12.0, torch 2.10. Nano at 640 px, 40 epochs,
408 training and 72 validation frames (480 renders of LEGO / DUPLO / dominoes in the station's
cameras, tools/README.md): **9.6-11.5 min of training**, **7 ms a 640 x 480 frame** to detect
(median over 156 frames; 45 ms sliced at 320 px), 0.19 GB of GPU memory. The detection numbers are in skills/training-a-part-detector-from-auto-labels.

## Install

```bash
uv sync --project tools/rfdetr   # rfdetr[train]==1.12.0, torch, torchvision
```

The COCO checkpoint downloads (the package's own download) on the first `train`.
`GAP_RFDETR_DEVICE` (default `cuda`).
