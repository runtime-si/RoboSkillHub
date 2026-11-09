---
name: sam2
description: SAM 2.1 promptable video mask tracking (facebook/sam2.1-hiera-*,
  Apache-2.0; EdgeTAM as the fast option) -- seeded with a box, points or a
  mask on one frame, follows one part through a live camera stream or a
  recorded clip, returning its mask, box and an occlusion-head presence call
  per frame, and picks it up again after an arm or hand has covered or moved
  it. Live sessions hold constant GPU memory. Use when a closed loop or a
  verify stage must keep the identity of one part (a cube, a box, a plug)
  while the arm or a hand passes over or moves it; seed it from a detector
  (owlv2, grounding-dino) or a sam3 mask. Not open-vocabulary.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, tracking, video-segmentation, sam2, edgetam, gpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting, insertion]
    tuned_on: [yam-station-orbbec-overhead, yam-left-wrist-d405]
    weights:
      uri: hf://facebook/sam2.1-hiera-base-plus@b7320756a13354e7530a63935656d35b2f91a290
      sha256: 0b0e295a0327dd8abba4d777c27f580ae570b4e08c3d658dbb3495299dc83e3d
      license: Apache-2.0 (model card); facebook/sam2.1-hiera-tiny / -small / -large and yonigozlan/EdgeTAM-hf (Apache-2.0) at pinned commits with GAP_SAM2_MODEL
    provenance:
      authors: [Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, et al. (SAM 2); Chong Zhou, Chenchen Zhu, Yunyang Xiong, et al. (EdgeTAM)]
      code: https://github.com/huggingface/transformers/tree/b39d97c836e8d9f0ee6035a87c7a3ce5eaea43a8/src/transformers/models/sam2_video
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2408.00714
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "sam2"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - sam2.tracker_init: Open a mask-tracking session on a live frame from a box, points or a mask.
    - sam2.tracker_update: Advance a session by one frame; mask, box, confidence, presence.
    - sam2.tracker_close: Free a session (idempotent).
    - sam2.track: Track one object through a recorded clip from a prompt on any frame.
---

# sam2

SAM 2.1's video predictor ([SAM 2](https://arxiv.org/abs/2408.00714), Meta,
Apache-2.0) and [EdgeTAM](https://arxiv.org/abs/2501.07256) (same design,
2D spatial perceiver on the memory; Apache-2.0) through transformers'
`Sam2VideoModel` / `EdgeTamVideoModel`, weights pinned to commits. Frames in:
RGB uint8 `[H, W, 3]` of one camera; masks out uint8 `[H, W]` 0 / 255, boxes
in frame pixels. The same session semantics as `sam3.tracker_*`, without
text prompts: find the part first, then hand its box or mask over.

```python
cam = ctx.tool("robot.get_observation")["cameras"][0]
det = ctx.tool("owlv2.detect", image=cam["rgb"], prompts=["cube"])["detections"][0]
s = ctx.tool("sam2.tracker_init", image=cam["rgb"], box=det["box"], object_name="cube")
while running:
    cam = ctx.tool("robot.get_observation")["cameras"][0]
    r = ctx.tool("sam2.tracker_update", tracker_id=s["tracker_id"], image=cam["rgb"])
    if r["object_present"]:
        ...  # r["mask"], r["box"]
ctx.tool("sam2.tracker_close", tracker_id=s["tracker_id"])
# a recorded clip (verify stage): prompt on any frame, forward then backward
t = ctx.tool("sam2.track", frames=clip, box=box_on_frame_k, prompt_frame=k)
```

## Tools

| tool | returns |
| --- | --- |
| `sam2.tracker_init(image, box=None, pixel_x=None, pixel_y=None, points=None, point_labels=None, mask=None, object_name="")` | `{tracker_id, initial_mask, initial_box, score, object_present}`; `tracker_id` "" when the first mask is empty |
| `sam2.tracker_update(tracker_id, image)` | `{mask, box, confidence, object_present, area_px, frame_index}`; mask / box None when absent |
| `sam2.tracker_close(tracker_id)` | `{closed}` (idempotent) |
| `sam2.track(frames, box=None, points=None, point_labels=None, mask=None, prompt_frame=0, return_masks=False)` | per frame `boxes` (None when absent), `present`, `confidence`, `area_px`, optional `masks` |

- Prompts: a `box` (optionally refined by `points` with `point_labels` 1 =
  object, 0 = background), one pixel (`pixel_x`, `pixel_y`), `points`, or a
  `mask`.
- `object_present` / `confidence`: SAM 2's occlusion head (logit > 0, the
  gate SAM 2 applies to its own masks) and its sigmoid; the memory keeps
  running while the part is hidden, so it comes back on its own.
- Sessions: at most 8 open (the least recently used is evicted), closed after
  10 min idle. After every frame a live session is pruned to what the memory
  can still read (7 mask memories, 16 object pointers, the prompt frame).

## Install

```bash
uv sync --project tools/sam2   # torch, torchvision, transformers>=4.57
```

Weights download from Hugging Face on first call or `gap skills check
--download` (the base-plus snapshot is 0.6 GB with Meta's original `.pt` beside the
safetensors; EdgeTAM 0.06 GB). Env knobs: `GAP_SAM2_MODEL`
(one of the pinned repos -- `facebook/sam2.1-hiera-{tiny,small,base-plus,large}`,
`yonigozlan/EdgeTAM-hf` -- or a local directory), `GAP_SAM2_DEVICE` (default
`cuda`), `GAP_SAM2_DTYPE` (`bfloat16` on CUDA by default, `float32`).

## Measured

2026-11-09 on a GCP RTX PRO 6000 (96 GB) **shared with six running
simulations (GPU at 99 %)**, transformers 5.19.0, torch 2.10.0, bfloat16.
Two real YAM clips at 15 fps, every frame through `tracker_init` /
`tracker_update` (the live path), the object seeded with its hand-labelled
box on frame 0, scored on hand-labelled keyframes:

- **top148** (overhead Orbbec, 899 frames, 60 s): a pick-and-place. The arm
  grasps a 30 mm cube (~22 px), carries it over the open box (the arm covers
  the box for frames ~420-600), drops it in; then a hand takes the cube out,
  puts it back near its start and pushes the box 55 px to the left. **box**:
  13 keyframes (hit at IoU >= 0.5) + 3 arm-occluded ones (kept at IoU >=
  0.3). **cube**: 8 keyframes where it is visible and not in the gripper
  (before the grasp, in the box, back at its start; hit at IoU >= 0.3).
- **wrist574** (left wrist D405, 450 frames, 30 s): the hand descends on a
  cable; the white USB-C plug (20 x 70 px to 90 x 90 px) moves across the
  frame, with one fast jump (frames 60-90). **plug** (housing and tip): 14
  keyframes, hit at IoU >= 0.3.

Labels, clip sources and the script: `tests/bundles/data/sam2_station_clips_labels.json`,
`tests/bundles/_sam2_station_eval.py`.

| model | box (13) | box under the arm (3) | cube (8) | plug (14) | update, median (p90) | peak GPU |
| --- | --- | --- | --- | --- | --- | --- |
| EdgeTAM | 13, IoU 0.97 | 3 | **6**, IoU 0.71 | 2, IoU 0.23 | **34-41 ms (45)** | **0.25 GB** |
| SAM 2.1 tiny | 13, 0.96 | 3 | 2, 0.24 | 14, 0.77 | 74-79 ms (89) | 0.45 GB |
| SAM 2.1 small | 13, 0.96 | 3 | 2, 0.26 | 14, 0.79 | 66-81 ms (90) | 0.46 GB |
| **SAM 2.1 base-plus (default)** | **13, 0.96** | **3** | **6, 0.67** | **14, 0.75** | 51-86 ms (98) | 0.52 GB |
| SAM 2.1 large | 13, 0.97 | 3 | 6, 0.66 | 14, 0.72 | 101-157 ms (176) | 0.80 GB |

- **Every model keeps the box** through the arm passing over it (IoU
  0.55-0.95 while covered) and follows it when the hand pushes it 55 px
  (0.88-0.95 after).
- **The cube is where they differ.** After the carry, tiny and small hand
  the cube's identity to the black gripper finger next to its start and stay
  there (IoU 0.00-0.10 for the rest of the clip). Base-plus, large and
  EdgeTAM report it absent (base-plus, large) or merged with the box
  (EdgeTAM) while it lies in the box: the 2 misses. All three find it again
  when the hand puts it back (IoU 0.88-0.92).
- **The plug: EdgeTAM spreads onto the cable.** After the fast camera jump
  its mask takes the plug plus 20 cm of attached cable (box 130 x 330 px;
  the plug is inside it, IoU 0.12-0.23). Every SAM 2.1 size keeps the plug
  alone (IoU 0.44-0.90).
- **Memory is constant.** With the pruning, allocated memory stays at 70 MB
  (EdgeTAM) / 113-119 MB (SAM 2.1) from frame 100 to frame 899. Without it
  (transformers' streaming session keeps every frame and output), base-plus
  grew to 0.94 GB at frame 100 and 3.99 GB at frame 449 (8.7 MB a frame;
  EdgeTAM 2.87 GB), with the same plug result (14 / 14, IoU 0.75): an
  hour-long live session would not fit without it.
- Load 9-21 s (weights cached); `tracker_init` 26-136 ms (the first in a
  process 0.7-1.1 s).

Not compared: `sam3.tracker_*`, the hub's text-prompted tracker. Its
checkpoint is gated on Hugging Face and no GPU host had a token. Pending.

Use: base-plus (the default) when the part can be confused with something
next to it or attached to it. Use EdgeTAM (`GAP_SAM2_MODEL=yonigozlan/EdgeTAM-hf`)
for a rigid part well apart from others when the loop needs < 50 ms a frame
or the GPU is small. Tiny / small gave no speed gain over base-plus here and
lost the cube.

## Gotchas

- One object per session; open one session per part (up to 8).
- Prompt from the same camera's frame; the session refuses a frame of
  another size.
- `object_present` False is the model's own call. A part lying inside a
  container of the same colour can read as absent, as the cube did, or
  merge with the container (EdgeTAM). Re-seed from a detector if that
  matters.
- A wire, cable or hose attached to the part can be absorbed into the mask
  after a fast camera move (EdgeTAM). Check `area_px` against the first
  frame's.
- `sam2.track` holds the whole clip on the device (about 6 MB a frame at
  1024 px in bfloat16): split long recordings.
