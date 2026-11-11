---
name: dam4sam
description: DAM4SAM - SAM 2.1 mask tracking with a distractor-aware memory
  (CVPR 2025, Apache-2.0; SAM 2.1 base-plus weights, Apache-2.0). Besides
  SAM 2's recent frames it keeps anchor frames, taken when the tracked mask
  is confident while SAM's alternative masks point at something else (a
  look-alike beside the target), so the target is told from its twins when
  they touch or cross. Seeded with a box or a mask; live sessions
  (tracker_init / tracker_update / tracker_close) or a recorded clip
  (track). Use when one part must be followed among identical ones -
  a domino in a row, a brick among same-colour bricks, a pen among pens -
  where tools/sam2 can jump to a twin.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, tracking, video-segmentation, sam2, distractors, gpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [domino, lego, jigsaw, pick-place]
    weights:
      uri: hf://facebook/sam2.1-hiera-base-plus@b7320756a13354e7530a63935656d35b2f91a290
      sha256: a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5
      license: Apache-2.0 (model card; Meta's original sam2.1_hiera_base_plus.pt)
    provenance:
      authors: [Jovana Videnovic, Alan Lukezic, Matej Kristan]
      code: https://github.com/jovanavidenovic/DAM4SAM/tree/aad389b85c22
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2411.17576
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "dam4sam"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - dam4sam.tracker_init: Open a session on a frame from a box or a mask.
    - dam4sam.tracker_update: Advance a session by one frame; mask, box, presence, anchor count.
    - dam4sam.tracker_close: Free a session (idempotent).
    - dam4sam.track: Track one part through a recorded clip from a box or mask on frame 0.
---

# dam4sam

[DAM4SAM](https://arxiv.org/abs/2411.17576) replaces SAM 2.1's memory
policy: recent-appearance memory plus a distractor-resolving memory of
anchor frames. Same call shape as `sam2.tracker_*`.

```python
s = ctx.tool("dam4sam.tracker_init", image=rgb, box=det["box"])
while running:
    r = ctx.tool("dam4sam.tracker_update", tracker_id=s["tracker_id"], image=next_rgb)
    if r["object_present"]:
        ...  # r["mask"], r["box"]
ctx.tool("dam4sam.tracker_close", tracker_id=s["tracker_id"])
```

## Tools

| tool | returns |
| --- | --- |
| `dam4sam.tracker_init(image, box=None, mask=None, object_name="")` | `{tracker_id, initial_mask, initial_box, object_present}`; `tracker_id` "" when the first mask is empty |
| `dam4sam.tracker_update(tracker_id, image)` | `{mask, box, object_present, area_px, frame_index, anchors}` |
| `dam4sam.tracker_close(tracker_id)` | `{closed}` |
| `dam4sam.track(frames, box=None, mask=None, return_masks=False)` | per frame `boxes` (None when absent), `present`, `anchors`, optional `masks` |

Boxes are `[x1, y1, x2, y2]` frame pixels; masks uint8 0 / 255. At most 4
sessions (least recently used closed), closed after 10 min idle. Unlike
`sam2` there is no occlusion score: `object_present` is "the mask is not
empty", and memory grows with the clip (upstream keeps every frame's
outputs).

## Install

```bash
uv sync --project tools/dam4sam
```

The SAM 2 fork (`sam2` package with `add_to_drm`) installs from the pinned
DAM4SAM commit; it conflicts with Meta's `sam2` package, hence its own
environment. Weights: `sam2.1_hiera_base_plus.pt` (309 MB) from the pinned
Hugging Face repo on first call or `gap skills check --download`.
`GAP_DAM4SAM_DEVICE` (default `cuda`).

## Measured

Not yet: the first live run (tests/bundles/_dam4sam_eval.py on rshgpu2) stopped at a CUDA out-of-memory error in `tracker_init`; numbers to follow.

## Provenance

`_Tracker` adapts `DAM4SAMTracker` of dam4sam_tracker.py at aad389b85c22
(Apache-2.0, Copyright the DAM4SAM authors; LICENSE.dam4sam); the header of
`tools.py` lists the changes. The modified SAM 2 video predictor is the
fork's `sam2` package, a dependency, not copied.
