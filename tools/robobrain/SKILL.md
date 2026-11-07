---
name: robobrain
description: BAAI RoboBrain 2.5-4B run locally on a GPU -- an ordered keypoint
  trajectory for a manipulation from one frame and an instruction (pixels, and
  camera / world points with the camera), pixel points for a phrase in
  gemini-er.point's shape, and task progress 0..1 against the task's start
  frame. Use when a propose stage wants a candidate path ("move the cable's
  free end through the ring") or a judge stage a progress score, without an API.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, planning, pointing, trajectory, progress, gpu]
  hub:
    stage: propose
    embodiments: [any]
    version: 0.1.0
    weights:
      uri: hf://BAAI/RoboBrain2.5-4B@ec4b3bf5e969b48ad17df432b479c10402a0079b
      sha256: 5c2d4d439003eca3d5b0a8750b2ff35cb9edf7e185b522f7a54f1fc4a427082f
      license: Apache-2.0
    provenance:
      authors: [BAAI]
      code: https://github.com/FlagOpen/RoboBrain2.5/blob/af98c932aac9ff715d70da177088d7bb95573ff7/inference.py
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2601.14352
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "robobrain"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - robobrain.trace: Ordered keypoint trajectory for an instruction from one frame (visual-plan schema; pixels, camera/world points with intrinsics).
    - robobrain.point: Pixel points for what a phrase names; drop-in for gemini-er.point.
    - robobrain.progress: Task progress 0..1 of the current frame against the task's start frame.
---

# robobrain

BAAI's [RoboBrain 2.5-4B](https://huggingface.co/BAAI/RoboBrain2.5-4B)
(Qwen3-VL-4B fine-tuned for embodied spatial and temporal reasoning; Apache-2.0,
ungated) as three in-process tools. Image in: RGB uint8 `[H, W, 3]`.

| tool | returns | same shape as |
| --- | --- | --- |
| `robobrain.trace(image, instruction, start_point=None, depth=None, intrinsics=None, camera_pose=None, mode="end_effector")` | an ordered keypoint trajectory: pixels, camera and world points | `molmo-motion.forecast` (the visual-plan schema) |
| `robobrain.point(image, query, max_points=10)` | `{pixel_x, pixel_y, found, points, raw}` | `gemini-er.point`, `locate-anything.point` |
| `robobrain.progress(image, task, start_image, goal_image=None, wrist_images=None, start_wrist_images=None)` | `{progress, score, found, raw, timing}` | - |

```python
plan = ctx.tool(
    "robobrain.trace",
    image=frame["rgb"],
    instruction="move the white cable's free end through the left black ring",
    start_point=[tip_u, tip_v],  # gemini-er.point / robobrain.point on the free end
    depth=frame["depth"],  # only the start's depth is read
    intrinsics=frame["intrinsics"],
    camera_pose=frame["pose"],
    mode="object",
)
waypoints = plan["tracks"][0]["points_world"]  # [F][x, y, z] m, ordered, untimed
```

## Model output and how it is converted

The prompts are upstream's, verbatim (see the module header and NOTICE):

- `point`: RoboBrain 2.5 `inference.py`'s pointing prompt. The answer is
  `[(x, y), ...]` with x, y normalized to 0-1000 across the image's width and
  height; pixels are `x / 1000 * W`, `y / 1000 * H`, clamped to the frame.
- `trace`, `mode="end_effector"`: `inference.py`'s trajectory prompt, "3D
  end-effector-centric waypoints"; `mode="object"`: RoboTracer's "3D
  object-centric visual trace" prompt (RoboTracer's spatial-trace training went
  into RoboBrain 2.5; its evaluation is the source of the depth convention).
  The answer is `[(x, y, d), ...]`: x, y as above, d the depth in metres
  along the optical axis (camera Z, as RoboTracer's evaluation back-projects
  it). With `intrinsics` each keypoint is lifted with its own d; with
  `camera_pose` (camera-to-world) it is also given in the world frame.
- `progress`: Robo-Dopamine's progress prompt (RoboBrain 2.5's progress
  model) in its forward mode: references = the start frame and the goal image
  (default upstream's blank grey placeholder), BEFORE = the start frame, AFTER
  = the current frame, each as front + left wrist + right wrist (the front
  frame repeated when no wrist views are given, as upstream's single-view
  usage does), every image scaled to Robo-Dopamine's 76,800-pixel budget. The
  answer `<score>+NN%</score>` is `score = NN / 100` and, in forward mode, the
  progress; `progress` is it clipped to 0..1.

Decoding is greedy (`do_sample=False`, upstream's 768-token budget; the
checkpoint's generation config samples at 0.7); repeated calls agreed exactly.
Images are passed to the processor directly instead of through
`qwen_vl_utils` (whose extra 28-px resize does not matter to normalized
coordinates).

**Start point.** The model has no start-point input. `start_point` is written
as the beginning of the model's answer, `[(x0, y0, `, so it continues the trace
from that pixel; with `depth` the start's measured depth is written as well,
`[(x0, y0, d0), (`, which anchors the metric depth of the rest (see Measured).
The returned `query_pixel` is the given pixel, not its 0-1000 rounding.

## Output: the visual-plan schema

`robobrain.trace` returns the structure `molmo-motion.forecast` documents, so a
skill switches trajectory planner by tool name (a future `vflash` tool should
return it too):

| field | robobrain.trace |
| --- | --- |
| `found` | at least one keypoint after the start |
| `tracks` | one track: the trace |
| `tracks[0].query_pixel` | the start: `start_point` if given, else the model's first keypoint |
| `tracks[0].pixels` | `[F][u, v]` the keypoints after the start, in order (may leave the image) |
| `tracks[0].start_camera` / `points_camera` | metres, OpenCV camera frame, from the model's depths; `[]` without `intrinsics` |
| `tracks[0].start_world` / `points_world` | metres, world frame via `camera_pose`; `[]` without it |
| `tracks[0].predicted` | `[F]` all true |
| `horizon` | F (the model chooses it; 3-6 on the station frames) |
| `dt_s` | `null`: keypoints are ordered, not timed |
| `frame` | `world`, `camera` or `pixel` |
| `backend` / `model` | `robobrain` / `BAAI/RoboBrain2.5-4B@<commit>` |
| `timing`, `raw` | as molmo-motion (`raw` includes any start written for the model) |

Extra fields: `tracks[0].start_depth_m` / `depths_m` (the model's d per
keypoint), `mode`, `depth_source` (`model`, `model, start from depth map`, or
`none` when the answer had no depths) and `measured_start_depth_m` (the depth
map at the start pixel).

## Install

```bash
gap skills install robobrain          # uv sync --project tools/robobrain
gap skills check --download          # optional: prefetch the 9.65 GB checkpoint
```

torch 2.8 / torchvision 0.23 / transformers 4.57.1 (the checkpoint is a
`Qwen3VLForConditionalGeneration`; upstream pins 4.57.0) and accelerate.
Weights: one 9.65 GB bf16 safetensors file from the pinned Hub commit; the
`sha256` above is the hub manifest digest of the whole snapshot, recomputed
from the download on Brev.

| Env | Meaning | Default |
| --- | --- | --- |
| `GAP_ROBOBRAIN_DEVICE` | torch device | `cuda` |
| `GAP_ROBOBRAIN_MODEL` | a local model directory instead of the pinned snapshot (e.g. RoboBrain2.5-8B-NV, not checked) | pinned Hub commit |

## Measured (2026-11-07, Brev L40S, real YAM station frames)

The server was started the way gap starts it (`uv run --project tools/robobrain
-- python -m gap_core.rpc.server --bundle robobrain`) on an L40S shared with
8-10 simulator processes (GPU at 100 % utilization). Frames: the cable board
(`threading/cable_frame_20261103-1421.npz`: RGB, depth, K) with the calibrated
camera pose `real/top_camera_pose.before-20261105-023658.json`, and the cube +
open box frame `g_sota_frames_20261103-0511/frame_02` (same camera; no depth:
for the anchored runs a depth map of the cube-top plane was computed from the
calibrated pose, which matches the measured depth of the cable frame to 3 mm).

**Traces (19 calls: 13 in a first pass, 6 anchored in a second).** Drawn on the frames and judged by eye:

| frame / instruction | traces | start | end, in the image | 3D |
| --- | --- | --- | --- | --- |
| cube: "push the small cube into the open box", "pick up the cube and place it in the box" | 12 (both modes, with / without start, with / without depth anchor) | `end_effector` without a start begins at the left gripper and passes over the cube; `object` without a start begins on the cube (2/2) | inside the box, 12/12 | a lift-and-arc for both "push" and "pick": anchored (4 traces), the path rises from the cube top to 0.22-0.33 m above the table and ends 0.19-0.27 m above it, over the box, never descending into it |
| cable: "move the white cable's free end through the black ring on the board" | 4 (+1 repeat) | without a start: the right gripper (`end_effector`) or the plug end of the cable (`object`); with `start_point` on the free end: there | at a ring 1/4 (`end_effector`, no start: the right gripper's path ends on the lower right ring); the others end on the board 20-35 px from a ring | anchored on the measured start depth (2 traces): the path rises from the table to 0.18-0.19 m and ends 0.15 m above it; no pass through a ring |
| cable: "...thread ... through the black ring on the board" / "insert the free end of the white cable into the left black ring" | 2 (`object`, start + depth) | the free end | "left black ring": on the left ring's top edge; "thread": 20 px above it | end 0.28 / 0.14 m above the table |

- **Usable from one still frame as a coarse plan in the image, not in 3D.**
  Unlike MolmoMotion (which forecast no motion from a still frame), every call
  returned a purposeful 4-7 keypoint path: from the object (or the gripper) up
  and over to the named target, and the endpoint pixel lands on the target
  when the target is unambiguous (the box, 12/12; "the left black ring").
  The 3D part is a generic lift-arc: "push" is not distinguished from "pick"
  (both lift 20-30 cm), the path ends hovering 14-28 cm above the target, and
  nothing describes going through the ring (no approach axis, no pass). Use
  the endpoint pixel (and the side the path comes from) as a target proposal
  and let the skill derive heights, approach and insertion from geometry.
- **The model's own depth is 25-45 % short on this camera** (0.50-0.69 m at
  the start where the depth map reads 0.89-0.96 m), putting the start of an
  unanchored trace 0.26-0.41 m above the table. Writing the measured start
  depth into the answer (`start_point` + `depth`) fixes the scale: the model
  then continues with consistent depths (the path starts on the table and
  stays above it). Rescaling
  an unanchored trace by measured/model at its start does not work (the
  `end_effector` trace then dips 12 cm under the table), so the tool does not.
- Same request, same answer (greedy; the repeated trace was identical).

**Points** on the locate-anything evaluation's 8 pointing queries (same
frames, same ground truth regions, same scoring):

| | robobrain | gemini-er | locate-anything (batch) |
| --- | --- | --- | --- |
| cube / inside of the open box (4 queries, 2 frames) | 4 | 4 | 4 |
| free end of the (thin) white cable (2) | 0 | 2 | 2 |
| centre of each black ring (2, all 4 rings) | 0.5 | 1 | 0 |
| **total, 8** | **4.5** | **7** | 6 |

The model's pointing prompt asks for one point and it gave one each time.
On the cable frame with this task's phrasing it was right 3/3: "the white
cable's free end" (on the free end), "the black ring on the board" and "the
black ring" (each on one ring). "The free end of the thin white cable" landed
beside the cable's right-hand loop, away from either end: phrasing matters.

**Progress (21 calls).** On Robo-Dopamine's own demo clips (5 evenly spaced
frames, forward mode, single view): "organize the table" 0.03 / 0.51 / 0.70 /
0.99 (0.03 / 0.51 / 0.73 / 0.99 with its goal image), "push the green block
into the green bowl" 0.25 / 0.75 / 0.80 / 0.75, the failed "put the marker in
the cup" 0.05 / 0.17 / 0.11 / 0.15: sensible. On the station's cube frame it
was not: the same frame as start and now gave 0 (right), but a synthetic
"cube moved into the box" edit 0.10, the reverse edit +0.10 (should be
negative), and the cable board as "now" against the cube start 1.0. Treat
`progress` on this station as unvalidated until checked on a real episode.

**Latency** (warm, shared GPU): `point` median 2.5 s (2.3-2.7 s; gemini-er
2.3 s, locate-anything 1.8-15 s in their runs), `trace` median 12 s (8-21 s,
one decoded token per digit: ~40-60 tokens a keypoint), `progress` 3.6 s
(3.1-4.6 s, 8 images). Load plus the first call: 15-18 s from a warm disk
cache; the 9.65 GB download took ~45 s. **GPU memory:** 9.8 GB for the server
process (`nvidia-smi`), steady across all calls.

## Gotchas

- `trace` is a plan proposal from pixels, not a collision-checked path: check
  reach and collisions (`curobo`) before driving an arm along it, and expect
  hover heights, not contact, at the end.
- Without `intrinsics` only pixels come back; the model's depths are still in
  `depths_m`. Give `start_point` and `depth` whenever world points matter.
- `end_effector` traces start at a gripper the model picks (the left or right
  one); pass `start_point` (the object, or the gripper's pixel) to fix it.
- `point` returns one point for "each" queries; ask once per object.
- `progress` compares against `start_image`; the same frame as start and now
  is 0 by construction. A negative `score` (regression) is clipped to 0 in
  `progress`.
