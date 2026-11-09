---
name: mm-grounding-dino
description: MM-Grounding-DINO open-vocabulary detection (OpenMMLab's open
  retraining of Grounding DINO, Apache-2.0 weights) with grounding-dino's
  call and answer -- text phrases to labelled boxes. On the station's
  overhead frames the large checkpoint was the most precise detector the hub
  has measured, finding every 30 mm cube at ~1 m and the open boxes that
  owlv2 misses, at 2-9 false positives over 7 frames instead of 18-19. It
  finds fewer plugs than owlv2. Use when a workflow needs table parts and
  containers named by text with few false positives, as a drop-in for
  grounding-dino.detect or as a sahi backend. Needs transformers 4.x.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, detection, open-vocabulary, small-objects, gpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting]
    tuned_on: [yam-station-orbbec-overhead]
    weights:
      uri: hf://openmmlab-community/mm_grounding_dino_large_all@0445d0a0ae5f0fa6be689cf7243d4d0ef4ba7de0
      sha256: 42d97534df2302b4a3f41bb8c128a79d21d1526b7dfc79766da1bc68914effbf
      license: Apache-2.0 (model card); mm_grounding_dino_base_all@b32293f5 and mm_grounding_dino_tiny_o365v1_goldg_v3det@ef2053aa (Apache-2.0) with GAP_MMGDINO_MODEL
    provenance:
      authors: [Xiangyu Zhao, Yicheng Chen, Shilin Xu, Xiangtai Li, Xinjiang Wang, Yining Li, Haian Huang]
      code: https://github.com/huggingface/transformers/tree/v4.57.1/src/transformers/models/mm_grounding_dino
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2401.02361
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "mm-grounding-dino"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - mm-grounding-dino.detect: Open-vocabulary detection with MM-Grounding-DINO (labelled boxes + scores, grounding-dino's shape).
---

# mm-grounding-dino

[MM-Grounding-DINO](https://arxiv.org/abs/2401.02361) (OpenMMLab; Grounding
DINO's architecture retrained on open data, mmdetection Apache-2.0) through
transformers' `MMGroundingDinoForObjectDetection`. The weights are
`openmmlab-community/mm_grounding_dino_large_all` (default), `..._base_all`
or `..._tiny_o365v1_goldg_v3det`, pinned to commits; the model cards are
Apache-2.0. The call and the answer are `grounding-dino.detect`'s, so a
skill can swap the two by tool name.

```python
cam = ctx.tool("robot.get_observation")["cameras"][0]
dets = ctx.tool("mm-grounding-dino.detect", image=cam["rgb"], query="cube. open box.")["detections"]
# on the table strip and sliced: 8 / 12 small parts, 4 / 4 boxes, 8 FPs over 7 frames
r = ctx.tool(
    "sahi.detect_sliced",
    image=cam["rgb"],
    prompts=["cube", "open box"],
    backend="mm-grounding-dino.detect",
    roi=table_box,
    slice_px=320,
    min_score=0.3,
)
```

## Tool

| tool | returns |
| --- | --- |
| `mm-grounding-dino.detect(image, query=None, box_threshold=0.3, text_threshold=0.2, prompts=None, max_detections=0)` | `{detections: [{box, label, score}]}` best-first |

- `query`: period-separated phrases (`"cube. open box."`; a missing final
  period is added), or `prompts` as a list.
- `label` is the matched text span, as in grounding-dino (sahi snaps it to the
  prompt); "" when no token cleared `text_threshold`.
- Zero or negative thresholds fall back to the defaults; `max_detections` > 0
  keeps the best N.

## Install

```bash
uv sync --project tools/mm-grounding-dino   # torch, torchvision, transformers>=4.52,<5
```

**transformers 5.x is refused.** On 5.19.0 the checkpoints load with no
warning but decode wrong. Tiny and large return collapsed boxes (1-2 px
wide, full height). Base returns plausible boxes but finds 2 / 12 small
parts on the whole frame instead of 6-8 / 12. The same checkpoints and frames
on 4.57.1 give the numbers below, so the bundle pins `<5` and checks at load.

Weights download from Hugging Face on first call or `gap skills check
--download` (large 1.4 GB, base 0.9 GB, tiny 0.7 GB). Env knobs:
`GAP_MMGDINO_MODEL` (one of the pinned repos or a local directory),
`GAP_MMGDINO_DEVICE` (default `cuda`).

## Measured

These numbers come from 2026-11-09 on a GCP RTX PRO 6000 (96 GB) **shared
with six running simulations (GPU at 99 %)**, with transformers 4.57.1 and
torch 2.10.0. The data is the 7 real YAM overhead frames (Orbbec, 640 x 480)
and the hand labels of the sahi / owlv2 checks: 12 small parts and 4 open
boxes.

The small parts are:
- 4 cubes of 30 mm (16-36 px), three of them at ~1 m;
- a pen;
- 3 cable plugs (14-17 x 8-12 px);
- 4 black rings on the black board.

Every backend was called through `sahi.detect_sliced` (`full` = the whole
frame, `ROI` = the table strip, `s320` = 320 px slices plus the whole), with
the raw answers replayed per score floor. A hit is IoU >= 0.3 with a label
of the same name. The mm-grounding-dino rows go through this tool; reproduce
them with `tests/bundles/_mm_grounding_dino_station_eval.py` and
`_owlv2_station_score.py text`.

| backend, config | floor | small parts | cubes at ~1 m | open boxes | FP (7 frames) | warm call |
| --- | --- | --- | --- | --- | --- | --- |
| owlv2 base, full (hub's best for small parts) * | 0.2 | 7 / 12 | 3 / 3 | 0 / 4 | 18 | 0.22 s |
| grounding-dino-base, ROI * | 0.2 | 7 / 12 | 3 / 3 | 2 / 4 | 19 | 0.37-0.51 s |
| grounding-dino-base, ROI, transformers 4.57.1 | 0.2 | 8 / 12 | 3 / 3 | 1 / 4 | 25 | 1.1 s |
| **MM-GDINO large, ROI** | **0.3** | 6 / 12 | **3 / 3** | 3 / 4 | **2** | 0.73 s |
| MM-GDINO large, ROI | 0.25 | 7 / 12 | 3 / 3 | **4 / 4** | 9 | 0.73 s |
| **MM-GDINO large, ROI, s320** | **0.3** | **8 / 12** | 3 / 3 | **4 / 4** | 8 | 3.2 s |
| MM-GDINO large, full | 0.3 | 5 / 12 | 3 / 3 | 3 / 4 | 14 | 0.73 s |
| MM-GDINO base, ROI | 0.25 | 6 / 12 | 3 / 3 | 4 / 4 | 14 | 1.4 s ** |
| MM-GDINO base, full | 0.3 | 6 / 12 | 3 / 3 | 1 / 4 | 15 | 1.4 s ** |
| MM-GDINO tiny, ROI | 0.2 | 4 / 12 | 1 / 3 | 4 / 4 | 15 | 0.5 s |

\* transformers 5.19.0 (the hub's other bundles), same frames; these
reproduce the owlv2 PR's (#101) numbers. \*\* Base measured slower than large
in this run, so read the latencies as contention, not model cost.

- **Precision is where it wins.** On the table ROI at 0.3, large has 2 false
  positives over 7 frames (owlv2 18, grounding-dino 19), with every cube and
  3 / 4 boxes. Sliced at 320 px it reaches 8 / 12 small parts and 4 / 4
  boxes at 8 FPs, which is the best row of any backend measured on these
  frames.
- **Per part (large, ROI, 0.3):** cubes 4 / 4, pen 1 / 1, plugs 1 / 3,
  rings 0 / 4. owlv2 base finds 3 / 3 plugs, so keep owlv2 for plugs and
  similar small, low-contrast connectors. No backend found the black rings
  on the black board.
- **Cost:** large uses 7.4 GB allocated (14.8 GB reserved) and loads in
  6-11 s with the weights cached; base uses 2.2 GB, tiny 1.9 GB. On this
  saturated GPU a whole-frame call took 0.7 s (large), against 0.22 s for
  owlv2 base and 0.02 s for omdet-turbo.

Use large at 0.3 on a table ROI (or sliced at 320 px) when false positives
cost a grasp. Use base when 7 GB is too much. owlv2 is still the fast
default, and the tool for plugs. No LEGO bricks, dominoes or jigsaw pieces
were on the station when these frames were taken.

## Gotchas

- Run it in its own environment (transformers 4.x). Loading it next to a
  transformers-5 bundle in one process gets refused.
- Separate phrases with periods yourself; labels are text spans, so group
  by them or let sahi snap them.
- The model is a lazy module-level singleton; the first call pays the load.
