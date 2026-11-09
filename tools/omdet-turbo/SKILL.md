---
name: omdet-turbo
description: OmDet-Turbo fast open-vocabulary detection (omlab/omdet-turbo-swin-tiny-hf,
  Apache-2.0) -- text prompts to labelled boxes in grounding-dino's shape at
  ~0.02 s a 640 x 480 frame and 0.6 GB of GPU memory, about 10x faster than
  owlv2 and grounding-dino. Use for video-rate re-detection of table-scale
  objects (an open box, a tray, a bin, a board, a held part) in closed loops or
  as a cheap sahi backend; on 15-30 px parts at ~1 m (cubes, plugs, bricks)
  it finds fewer than owlv2, so use owlv2 there.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, detection, open-vocabulary, real-time, gpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting]
    tuned_on: [yam-station-orbbec-overhead]
    weights:
      uri: hf://omlab/omdet-turbo-swin-tiny-hf@7fe93cecfb770c4d76cf71163956221249cab566
      sha256: 6b270f1fd5184d0f438276a908a60692ce7ddbf47432b68c5c90a5cdd849875c
      license: Apache-2.0 (model card)
    provenance:
      authors: [Tiancheng Zhao, Peng Liu, Xuan He, Lu Zhang, Kyusong Lee]
      code: https://github.com/huggingface/transformers/tree/b39d97c836e8d9f0ee6035a87c7a3ce5eaea43a8/src/transformers/models/omdet_turbo
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2403.06892
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "omdet-turbo"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - omdet-turbo.detect: Fast open-vocabulary detection from text prompts (labelled boxes + scores, grounding-dino's shape).
---

# omdet-turbo

[OmDet-Turbo](https://arxiv.org/abs/2403.06892) (Om AI Lab; Swin-T backbone,
CLIP text encoder, efficient fusion head) through transformers'
`OmDetTurboForObjectDetection`, weights `omlab/omdet-turbo-swin-tiny-hf`
pinned to a commit (model card Apache-2.0; original code
[om-ai-lab/OmDet](https://github.com/om-ai-lab/OmDet), Apache-2.0). Image in:
RGB uint8 `[H, W, 3]`; boxes out in frame pixels, best-first.

```python
cam = ctx.tool("robot.get_observation")["cameras"][0]
# grounding-dino's shape, so a skill can swap backends
dets = ctx.tool("omdet-turbo.detect", image=cam["rgb"], prompts=["open box", "tray"])["detections"]
# or as a sahi backend (one call per slice; ~0.1 s for a table ROI at 256 px)
r = ctx.tool("sahi.detect_sliced", image=cam["rgb"], prompts=["cube"],
             backend="omdet-turbo.detect", slice_px=256, roi=table_box, min_score=0.2)
```

## Tool

| tool | returns |
| --- | --- |
| `omdet-turbo.detect(image, prompts=None, threshold=0.2, query=None, nms_iou=0.5, task=None, max_detections=0)` | `{detections: [{box, label, score}]}` best-first, `label` = the prompt |

- `prompts`: a list of phrases, or one string `"cube. plug."` (grounding-dino's
  form) or `"cube, plug"`; `query` takes the same string (the name sahi
  passes).
- `threshold` floors the class sigmoid (upstream default 0.3; 0.2 here, see
  below); `nms_iou` suppresses lower-scored boxes of the same label
  (upstream 0.5); `max_detections` > 0 keeps the best N after NMS.
- `task`: the sentence the fusion head is conditioned on; default upstream's
  `"Detect a, b."`. The class-prompt embeddings are cached in the model, so
  repeating the same prompts costs only the image pass.

The post-process is transformers' `post_process_grounded_object_detection`
redone in numpy (top-k (query, class) pairs, floor, class-wise NMS, clip):
on the 7 station frames x 4 floors it returned the same 557 boxes, labels and
order as the transformers call, boxes identical, scores within 5e-8.

## Install

```bash
uv sync --project tools/omdet-turbo   # torch, torchvision, transformers, timm
```

Weights (0.5 GB) download from Hugging Face on first call or `gap skills check
--download`. Env knobs: `GAP_OMDET_MODEL` (the pinned repo or a local
directory), `GAP_OMDET_DEVICE` (default `cuda`; CPU works, slowly).

## Measured

2026-11-09 on a GCP RTX PRO 6000 (96 GB) **shared with six running
simulations (GPU at 99 %)**, transformers 5.19.0, torch 2.10.0, on the 7 real
YAM overhead frames (Orbbec, 640 x 480) and hand labels of the sahi / owlv2
checks: 12 small parts (4 cubes 30 mm, 16-36 px, three of them at ~1 m; a
pen; 4 black rings on the black board; 3 cable plugs 14-17 x 8-12 px) and 4
open boxes (50-85 px). Every backend called in-process through
`sahi.detect_sliced` (`full` = one call on the frame, `ROI` = one call on the
table strip, `s256` = 256 px slices plus the whole), raw answers replayed per
score floor; a hit is IoU >= 0.3 with a label of the same name, every other
box a false positive (FP). Prompts `cube, box, pen` / `ring, plug`.
Reproduce with `tests/bundles/_open_vocab_station_eval.py` and
`_owlv2_station_score.py text`.

| backend, config | floor | small parts | cubes at ~1 m | open boxes | FP (7 frames) | s / frame |
| --- | --- | --- | --- | --- | --- | --- |
| owlv2 base, full (hub's best for small parts) | 0.2 | **7 / 12** | **3 / 3** | 0 / 4 | 18 | 0.19 |
| owlv2 base, ROI, s256 | 0.15 | **8 / 12** | 3 / 3 | 0 / 4 | 25 | 0.94 |
| grounding-dino-base, ROI | 0.2 | 7 / 12 | 3 / 3 | 2 / 4 * | 18 * | 0.20 |
| omdet-turbo, full | 0.2 | 1 / 12 | 0 / 3 | **4 / 4** | 16 | **0.022** |
| omdet-turbo, ROI | 0.25 | 2 / 12 | 0 / 3 | **4 / 4** | 7 | **0.020** |
| **omdet-turbo, ROI** | **0.3** | 1 / 12 | 0 / 3 | **4 / 4** | **2** | **0.020** |
| omdet-turbo, ROI | 0.1 | 4 / 12 | 0 / 3 | 4 / 4 | 54 | 0.020 |
| omdet-turbo, ROI, s256 | 0.1 | 5 / 12 | 2 / 3 | 4 / 4 | 108 | 0.11 |
| omdet-turbo, ROI, s256 | 0.2 | 3 / 12 | 1 / 3 | 4 / 4 | 49 | 0.11 |

\* grounding-dino rows from the owlv2 PR (#101), same frames and protocol.

- **Small parts: owlv2 stays the tool.** OmDet-Turbo finds 1-2 of 12 at a
  usable floor and none of the three cubes at ~1 m; lowering the floor to 0.1
  or slicing at 256 px recovers 4-5 / 12 at 54-108 FPs. Its input is resized
  to 640 x 640, so a 16 px cube stays ~16-21 px.
- **Open boxes: 4 / 4 at 2 FPs** on the table ROI at 0.3 (owlv2 base calls
  them "cube": 0 / 4; grounding-dino 2-3 / 4). Larger table objects are what
  it is good at.
- **Latency**: 0.020-0.022 s a frame end to end (CPU preprocessing included)
  on the busy GPU, against 0.19-0.22 s for owlv2 base and 0.23-0.37 s for
  grounding-dino-base in the same run (owlv2 measured 0.06-0.07 s on an idle
  4090 D in #101, so the gap on an idle GPU is ~3x). Peak torch memory 0.58
  GB allocated, 0.69 GB reserved; first call (weights cached) 9.9 s.

Use: `threshold=0.3` on a table ROI for boxes / trays / bins / boards and
other table-scale objects when the loop needs a detector per frame; owlv2
(or owlv2 under sahi) for parts under ~40 px. No LEGO bricks, dominoes or
jigsaw pieces were on the station when these frames were taken.

## Gotchas

- Scores are per (query, class) sigmoids: one query can return the same box
  under two prompts (an open box as "box" and "cube"); NMS is per label, so
  dedupe across labels downstream if that matters.
- Prompts are encoded by CLIP's text tower, so short nouns work best;
  `task` changes the conditioning sentence, not the classes.
- The model is a lazy module-level singleton; the first call pays the load.
