---
name: grounding-dino
description: Grounding DINO zero-shot object detection — natural-language
  queries to labeled 2D bounding boxes with confidence scores. Use when a
  workflow needs to locate named objects in an RGB image before segmenting
  or grasping them.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, detection, gpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    provenance:
      code: https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/grounding-dino
      upstream_license: MIT
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "grounding-dino"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - grounding-dino.detect: Zero-shot detection from a text prompt (labeled boxes + scores).
---

# grounding-dino

The Grounding DINO servicer (`IDEA-Research/grounding-dino-base` via
transformers) as one in-process tool. Image in: RGB uint8 `[H, W, 3]` numpy
array; out: `{detections: [{box, label, score}, ...]}`.

## When to use

- Locating a named object in a camera frame: `grounding-dino.detect(rgb,
  "cream cheese box.")`, pick the best box (highest score, or closest to a
  pointing-model pixel), then `sam3.segment_box` for a pixel-accurate mask.
- Empty `detections` means nothing cleared the thresholds — treat as
  not-found, don't retry blindly with the same prompt.

## Install

```bash
uv sync --extra grounding-dino   # torch + transformers
# (pip: pip install -e ".[grounding-dino]")
```

Weights download from HuggingFace on first call. Env knobs:
`GAP_DINO_DEVICE` (default `cuda`; CPU works but is slow) and
`GAP_DINO_MODEL` (default `IDEA-Research/grounding-dino-base`).

## Gotchas (carried over from the servicer)

- **Period-separated phrases**: GDINO's text encoder expects each object
  phrase terminated with `.` (`"red cube. green cube."`). A missing final
  period is appended automatically, but separate multiple objects yourself.
- Default thresholds are deliberately low (0.20/0.20) for recall on
  household objects; raise them when false positives leak through. Zero or
  negative thresholds fall back to the defaults (proto-default semantics).
- `label` strings are the matched text spans, not your full query — when
  querying multiple phrases, group detections by label.
- The model is a lazy module-level singleton; the first call pays the
  weights-load latency, subsequent calls don't.

## transformers 5.x: the decoder box head is reloaded and checked

transformers 5.x (5.12.1, the bundle's lock, and 5.19.0 checked) loads
`IDEA-Research/grounding-dino-base` without a warning but leaves the
decoder's shared box head (`decoder_bbox_embed_share`) at its random
initialisation. The checkpoint's `model.decoder.bbox_embed.0.*` is not
loaded; the last layer starts at zero, so the six decoder layers stop
refining the boxes. The same release ties every layer of an unshared head
to layer 0 (MM-Grounding-DINO; see tools/mm-grounding-dino, which pins
transformers 4.x).

The bundle loads the default model at a pinned commit and checks every
tensor against the checkpoint file after `from_pretrained`. When anything
differs, it reloads the decoder box heads from the checkpoint (one shared
module, or one untied module per layer) and checks again. If tensors still
differ, it refuses with `PerceptionFailed` rather than serve a mis-loaded
model. A model without `model.safetensors` is not checked (warning).

Measured 2026-11-09 on a GCP RTX PRO 6000 (rshgpu5/6). The data is the 7
station frames of tools/sahi and tools/owlv2, and the target-part set of
the detector guide (tools/README.md): 96 composited LEGO / DUPLO / domino
renders, 24 domino renders and 40 PuzzleMap jigsaw photos.

| transformers | matched-box IoU, overhead / wrist / close / jigsaw | hits at IoU >= 0.5, overhead LEGO / DUPLO / domino (0.2) | station small parts, frame / ROI (0.2) |
| --- | --- | --- | --- |
| 5.19.0 as loaded | 0.867 / 0.934 / 0.945 / 0.901 | 0.32 / 0.67 / 0.46 | 4 / 12, 23 FP; 7 / 12, 19 FP |
| **5.19.0 repaired** | **0.885 / 0.954 / 0.952 / 0.912** | 0.31 / 0.67 / 0.47 | 4 / 12, 22 FP; 7 / 12, 20 FP |
| 4.57.1 | 0.888 / 0.957 / 0.958 / 0.904 | 0.25 / 0.75 / 0.50 | 6 / 12, 28 FP; 8 / 12, 25 FP |

- The repair brings box tightness back to 4.57.1's. Recall is about the
  same with or without it, because the encoder's proposals carry most of
  the localisation.
- The remaining 5.x vs 4.x differences come from transformers 5's
  refactored Swin backbone (renamed modules, `relative_position_index` no
  longer a checkpoint buffer). The encoder output moves by ~1 % (mean
  |x| 0.0505 vs 0.0508), which reorders the top queries. They are mixed in
  sign: 4.57.1 finds 2 more small parts on the station frames with 5 more
  FPs, and the target set is within noise. Not a degradation to pin
  against.
- The image processor's fast / slow path makes no difference (identical
  scores both ways on both releases). `attn_implementation="eager"` on
  5.19.0 is broken (top scores 0.09-0.12); leave the default.

