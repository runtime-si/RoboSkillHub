---
name: musc
description: MuSc zero-shot mutual anomaly scoring (MuSc / MuSc-V2, MIT)
  on DINOv2 (facebook/dinov2-base, Apache-2.0) - given N >= 3 images of
  things that should look the same (crops of parts in one orientation and
  scale), each image's patches are scored by their distance to the nearest
  patch of each other image, so the one that differs scores high, with no
  reference image and no training. Per-image scores, the re-scored ones
  (RsCon) and patch maps. Use when a workflow must find the odd part among
  many copies (a face-up tile among face-down ones, a fallen domino, a
  wrong brick) in one frame; skills/flagging-odd-one-out-parts-by-mutual-scoring
  cuts and normalises the crops.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, anomaly, zero-shot, dinov2, verification, gpu]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [domino, lego, jigsaw]
    weights:
      uri: hf://facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415
      sha256: c71d261eb4e7d8edbea0003bb7ce4d5536fc3c4da4527b4c4cb635776fa47925
      license: Apache-2.0 (model card); facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056 (Apache-2.0, manifest sha256 2ce1c84c61a3da31481b3678779d7c67fa3449d08f77605dfaf039c08cc83227) with GAP_MUSC_MODEL
    provenance:
      authors: [Xurui Li, Ziming Huang, Feng Xue, Yu Zhou]
      code: https://github.com/HUST-SLOW/MuSc-V2/tree/b5e911dbc089/models/modules
      upstream_license: MIT
      paper: https://arxiv.org/abs/2401.16753
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "musc"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - musc.score: Mutual anomaly scores of N >= 3 nominally identical images (per-image scores, RsCon re-scores, patch maps).
---

# musc

[MuSc](https://arxiv.org/abs/2401.16753) (ICLR 2024) and
[MuSc-V2](https://github.com/HUST-SLOW/MuSc-V2) score unlabelled images of
one kind against each other: normal patches find close matches in the other
images, an anomalous patch does not. This bundle is the 2D branch, with
DINOv2 ViT-B/14 through transformers.

```python
crops = [...]  # RGB uint8, one per part, same orientation and scale
r = ctx.tool("musc.score", images=crops, masks=crop_masks)
odd = int(np.argmax(r["scores"]))
```

## Tool

`musc.score(images, masks=None, image_size=448, topmin=0.3, r_list=None, layers=None)`
returns `{scores, rescored, patch_maps, grid, layers, model}`:

- `images`: N >= 3 RGB uint8 `[H, W, 3]` (any size; resized to
  `image_size` square, which is cut to a multiple of 14).
- `masks`: one per image (any size, > 0 = the part). A patch takes part when
  half of it is inside: outside patches neither get a score nor serve as
  references, so the background does not count.
- `scores[i]`: the maximum of image i's patch map (MuSc's image score). Unit
  DINOv2 features, so 0-2; compare within one call only.
- `rescored[i]`: RsCon - scores min-max normalised, then averaged over each
  image's 1 and 7 most similar images by the feature of its worst patch. It
  lowers a lone odd image's score towards its neighbours': MuSc uses it for
  test sets where defects of a kind recur; for one odd part among copies use
  `scores`.
- `patch_maps[i]`: `[grid, grid]` float32 (32 x 32 at 448), 0 outside the mask.

## How it scores (MuSc-V2, adapted)

1. Tokens of blocks 4, 8 and 12 (MuSc-V2's `feature_layers: [3, 7, 11]`),
   with the encoder's final layer norm (as DINO's `get_intermediate_layers`).
2. **SNAMD**: each patch becomes the similarity-weighted mean
   (exp(-distance)) of its 1 x 1, 3 x 3 and 5 x 5 neighbourhoods (zero
   padded), averaged; unit length.
3. **Mutual scoring**: for image i, each patch's distance to its nearest
   patch in each other image j; the patch score is the mean of the smallest
   30 % of those over j (at least one); averaged over the layers.
4. `scores` = max of the map; `rescored` = RsCon.

All of steps 2-4 are numpy (`score_tokens`); the model only gives tokens.

## Install

```bash
uv sync --project tools/musc      # torch + transformers
```

Weights (`facebook/dinov2-base`, 2 x 350 MB: `.bin` and `.safetensors`)
download from Hugging Face on first call or `gap skills check --download`.
Env knobs: `GAP_MUSC_MODEL` (`facebook/dinov2-small` or a local directory),
`GAP_MUSC_DEVICE` (default `cuda`; CPU works, slowly).

## Measured

2026-11-09 on a GCP RTX PRO 6000 (96 GB), transformers 5.19.0, torch 2.10.0,
through skills/flagging-odd-one-out-parts-by-mutual-scoring on rendered
domino scenes of 10 tiles (that skill's Checks): 10 crops of 448 px in 1.6 s
a call (not profiled; the numpy scoring of 1024 patches against 9 x 1024 is
most of it), the first call 4 s more for the model load. In the 32 scenes
with one odd tile where appearance was scored, the odd tile scored highest in
30: a face-up tile among face-down ones 2.1-2.8x the median, a face-down one
among face-up ones 1.2-2.1x, a standing one 1.3-1.6x, a different pip value
1.1-1.9x; in the 16 identical controls the highest score was 1.1-1.4x the
median. On the first 4 seeds 448 px put the odd tile first in 15 of 16 against
13 at 224 and lowered the controls' highest from 1.72x to 1.37x; layers 8 and
12 alone were worse (11-13 of 16); the SNAMD neighbourhoods (r 1, 3, 5 against
1 alone) changed nothing.

## Limits

- **2D only.** MuSc-V2's 3D branch (Point-MAE tokens on point clouds) is not
  ported; the skill scores heights itself.
- **Absence is weak.** A patch is odd when nothing like it is elsewhere; a
  part *missing* something the others have (a blank tile among pipped ones)
  only scores through context (1.2-2.1x the median in the renders).
- **Small differences are near the noise.** A different pip count on one of
  ten identical tiles scored 1.1-1.9x the median, the identical controls up to
  1.4x.
- Scores are relative to the set: with all parts different (a mixed set of
  dominoes) every part is "odd".
- DINOv2 in place of MuSc-V2's DINO ViT-B/8: not compared on MVTec.

## Provenance

`_aggregate`, `_mutual_scores` and `_rescore` adapt
`models/modules/_SNAMD.py` (`SNAMD._embed_2d`), `models/modules/_MSM.py`
(`compute_scores_single`, `MSM_single`) and `models/modules/_RsCon.py`
(`RsCon`, `WMO`) of MuSc-V2 at b5e911dbc089 (MIT, Copyright (c) 2025 HUST
SLOW; LICENSE.musc), in numpy; the header of `tools.py` lists the changes.
The PatchMaker MuSc-V2 copies from patchcore-inspection is not used (the
neighbourhoods are cut with array slicing).
