---
name: dinov2
description: DINOv2 (facebook/dinov2-base or -small, Apache-2.0) embeddings
  of regions of one RGB frame - boxes or masks - as unit vectors (the patch
  tokens pooled under the mask, and the CLS token of the masked crop). Two
  regions' dot product says how alike they look, across frames and cameras.
  Use when a workflow must recognise a part by comparing it with a few
  labelled examples (which brick, which tile side), or match parts across
  views; skills/segmenting-small-parts-from-a-few-reference-photos keeps a
  memory bank of them.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, embedding, few-shot, recognition, dinov2, gpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [domino, lego, jigsaw]
    weights:
      uri: hf://facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415
      sha256: c71d261eb4e7d8edbea0003bb7ce4d5536fc3c4da4527b4c4cb635776fa47925
      license: Apache-2.0 (model card); facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056 (Apache-2.0, manifest sha256 2ce1c84c61a3da31481b3678779d7c67fa3449d08f77605dfaf039c08cc83227) with GAP_DINOV2_MODEL
    provenance:
      authors: [Maxime Oquab, Timothee Darcet, Theo Moutakanni, et al.]
      code: https://huggingface.co/facebook/dinov2-base/tree/f9e44c814b77203eaa57a6bdbbd535f21ede1415
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2304.07193
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "dinov2"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - dinov2.embed_regions: DINOv2 unit embeddings of boxes or masks of one RGB frame ([mask-pooled patch tokens | CLS]).
---

# dinov2

[DINOv2](https://arxiv.org/abs/2304.07193) features through transformers,
one vector per region of a frame.

```python
r = ctx.tool("dinov2.embed_regions", image=rgb, masks=part_masks)
sim = r["embeddings"] @ bank.T  # cosine, -1..1
```

## Tool

`dinov2.embed_regions(image, boxes=None, masks=None, use_mask=True, margin=0.15, image_size=224)`
returns `{embeddings, boxes, pooled_patches, model}`:

- `image`: RGB uint8 `[H, W, 3]`; `boxes` `[x1, y1, x2, y2]` px (or
  `{x1, y1, x2, y2}`) or `masks` `[H, W]` (> 0 = the region), one of them.
- Each region is cut as a square around its box with `margin` x its long
  side of context on each side (at least 24 px, shifted to stay inside the
  frame); with masks and `use_mask` the pixels outside the mask are filled
  with the ImageNet mean colour. The crop is resized to `image_size`
  (multiple of 14; 16 x 16 patches at 224).
- `embeddings[i]`: `[2C]` float32 (C = 768 base, 384 small), unit length:
  the mean of the final-layer patch tokens at least half inside the mask
  (the patch nearest to it when none is; the whole crop for a box), unit,
  then the CLS token, unit. A dot product of two is the mean of their pooled
  and CLS cosines. Compare vectors of the same `model` only.
- `boxes[i]`: the crop cut from the frame; `pooled_patches[i]`: patches
  pooled.

## Install

```bash
uv sync --project tools/dinov2      # torch + transformers
```

Weights (`facebook/dinov2-base`, 350 MB safetensors) download from Hugging
Face on first call or `gap skills check --download`. Env knobs:
`GAP_DINOV2_MODEL` (`facebook/dinov2-small` or a local directory),
`GAP_DINOV2_DEVICE` (default `cuda`; CPU works, slowly).

## Measured

See skills/segmenting-small-parts-from-a-few-reference-photos (Checks):
the recognition accuracy of these vectors on 482 held-out rendered LEGO,
DUPLO and domino parts against OWLv2 exemplar embeddings.

## Limits

- One forward pass per region (batched 64 at a time): a frame with 40
  parts is 40 crops.
- Parts under ~15 px are upsampled blur; cut them from the wrist camera or
  a SAHI tile (see the skill's Limits).
- The mask fill makes the outline part of the feature: a mask that cuts a
  part in half embeds as a different shape.
