---
name: grconvnet
description: Proposes ranked top-down parallel-jaw grasps from one RGB-D
  frame with GR-ConvNet (skumra/robotic-grasping), on CPU (75 ms a call on 8
  desktop cores) or a GPU -- pixel centre, jaw angle, opening in px and
  metres and a quality, and with a camera pose the world grasp point, the
  heading the jaws close along (for robot.grasp_frame) and an oriented box
  measured from the depth along the jaw line (for OBB-driven grasp skills).
  Use when a pick needs a learned propose rung beside
  proposing-short-axis-grasps because the object's box does not say where to
  grasp it -- a container wider than the hand (grasp its wall), a curved
  cable, an irregular part -- with a mask from a perception skill to focus it
  on one target.
license: BSD-3-Clause
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, grasping, propose, rgbd, cpu]
  hub:
    stage: propose
    embodiments: [any]
    tasks: [pick-place]
    version: 0.1.0
    weights:
      uri: https://raw.githubusercontent.com/skumra/robotic-grasping/183c6f68c44c1c7ff0f07707e2db6fcfd6840d2d/trained-models/jacquard-d-grconvnet3-drop0-ch32/epoch_50_iou_0.94
      sha256: 67d8d2a948bd24f68547bbfd851f6ae2b229f07cf7ed1b75ed8c6730c6626bc6
      license: BSD-3-Clause (trained on Jacquard, academic non-commercial EULA; see Licence)
    provenance:
      authors: [Sulabh Kumra, Shirin Joshi, Ferat Sahin]
      code: https://github.com/skumra/robotic-grasping/tree/183c6f68c44c1c7ff0f07707e2db6fcfd6840d2d/inference
      upstream_license: BSD-3-Clause
      paper: https://arxiv.org/abs/1909.04810
gap:
  requires: {weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "grconvnet"]
    protocol: stdio-msgpack
  tools:
    - grconvnet.propose: Ranked top-down parallel-jaw grasps from an RGB-D frame; pixel centre, angle, width (px, m), quality, and world grasp point / closing heading / box with a camera pose.
---

# grconvnet

[GR-ConvNet](https://github.com/skumra/robotic-grasping) (Kumra et al., IROS
2020; a 1.9 M-parameter generative residual network, code and checkpoints
BSD-3-Clause) as one tool. The network maps an RGB-D (or depth) crop to
per-pixel grasp quality, jaw angle and opening; the tool picks the peaks and
turns them into grasps for a top-down parallel gripper such as the YAM's
linear_4310. In: RGB uint8 `[H, W, 3]`, depth float32 `[H, W]` metres
registered to it, `K`; optionally a target `mask` and the camera-to-world
`camera_pose`. Out: up to `n` grasps, best first.

| call | gives |
| --- | --- |
| `grconvnet.propose(rgb, depth_m, K, mask=m)` | pixel grasps on the target: centre, angle, opening (px, m at its depth), quality |
| `... camera_pose=pose` | also world `position`, `close_heading_deg`, `obb`, support / top heights |
| `... checkpoint="cornell-rgbd"` | the RGB-D Cornell network: thin parts seen in colour (a flat cable from the overhead camera) |
| `... gripper_width_max=w` | drop openings over `w` m (default 0.094, the linear_4310's span) |

```python
cam = ctx.tool("robot.get_observation")["cameras"][0]
mask = Ref("perceive.mug_mask")  # perceiving-objects' <name>_mask
g = ctx.tool(
    "grconvnet.propose",
    rgb=cam["rgb"],
    depth_m=cam["depth"],
    K=cam["intrinsics"],
    mask=mask,
    camera_pose=cam["pose"],
)["grasps"]
frame = ctx.tool("robot.grasp_frame", close_heading_deg=g[0]["close_heading_deg"])
# candidates for selecting-reachable-grasp: [gi["position"] for gi in g] with frame["rotation"],
# or g[0]["obb"] as object_obb / target_obb for an OBB-driven grasp skill (see Output)
```

## Output

`{grasps, n_peaks, n_too_wide, best_quality, crop, checkpoint, model, device,
timing}`; each grasp:

- `center_px` `[u, v]`, `angle_rad` / `angle_deg`: the jaws close along
  `(cos a, -sin a)` in the image (anticlockwise from +u as seen, upstream's
  convention), in `[-90, 90)`. `jaw_px` are the two ends of the opening.
- `width_px`, `width_m`: the network's opening (px of the frame; metres
  across the jaws, on the horizontal world plane at the grasp's top when a
  pose is given, else at its depth). Wider than `gripper_width_max`: dropped
  and counted in `n_too_wide`.
- `quality` 0..1 (smoothed peak height; not a calibrated success rate),
  `depth_m` at the centre and whether it was measured, `in_mask`.
- With `camera_pose`: `top_z` (surface at the centre), `support_z` (given, or
  the low quartile of measured heights in a window around the grasp, off the
  mask, within 0.15 m below its top), `object_height_m`, `position` (the
  centre's xy at mid-height -- the tool-centre target, as
  proposing-short-axis-grasps places its stations), `close_heading_deg`
  (world, `[-90, 90)`; pass it to `robot.grasp_frame(close_heading_deg=...)`),
  `object_width_m` / `object_length_m` (the object across the jaws and along
  their faces, walked out from the centre along the jaw line and its normal
  in the depth until the height drops below max(4 mm, a third of the
  centre's); the network's opening when the object is under 4 mm), and `obb`
  (`OrientedBoundingBox`, half-extents, local x = the closing axis).
  `closing_is_short_axis` says whether x is also the box's shorter
  horizontal side: skills that close across a box's short side
  (grasping-flat-objects, proposing-short-axis-grasps) follow this grasp
  only when it is true; otherwise drive the hand with `close_heading_deg`.

## How it runs

- The network is copied into `tools.py` (layers unchanged so the released
  state dicts load). The checkpoint is fetched once from the repository at
  commit `183c6f68` (raw.githubusercontent.com, 7.6 MB) and checked against
  its SHA-256 (`CHECKPOINTS` in tools.py). Upstream saved whole pickled
  modules, so they cannot load with `weights_only`; the unpickler maps
  upstream's two classes onto the copies and refuses any other
  `inference.models` class.
- Crop: a square centred on the mask (or `roi`, else the frame centre) of
  side `input size / zoom` (300 / (4/3) = 225 px by default), grown so the
  mask fills at most 90 % of it, resampled to the network's input (300 px
  Jacquard, 224 px Cornell). Depth holes are filled from the nearest measured
  pixel inside the crop. Normalisation as upstream (depth in metres minus its
  mean, clipped to [-1, 1], first; RGB / 255 minus its mean).
- Post-processing as upstream (quality sigma 2, width sigma 1 px, width x 150,
  peaks 20 px apart above 0.2 away from the border), except that cos 2a and
  sin 2a are smoothed before the angle is taken (smoothing the angle tears
  where it wraps). Peaks off the frame (padding) or off the dilated mask are
  dropped.
- CPU by default with `min(8, cores)` threads; the model loads on first call
  (0.5-0.6 s from cache); importing the module does not import torch.

## Install

```bash
gap skills install grconvnet          # uv sync --project tools/grconvnet
gap skills check --download           # optional: prefetch the three checkpoints (23 MB)
```

| Env | Meaning | Default |
| --- | --- | --- |
| `GAP_GRCONVNET_DEVICE` | torch device | `cpu` |
| `GAP_GRCONVNET_THREADS` | CPU threads | `min(8, cores)` |
| `GAP_GRCONVNET_CHECKPOINT` | default checkpoint key | `jacquard-d` |
| `GAP_GRCONVNET_CHECKPOINT_PATH` | a local file for that key (digest not checked) | - |
| `GAP_GRCONVNET_CACHE` | download directory | `~/.cache/roboskillhub/grconvnet` |

## Measured (2026-11-08, real YAM station frames)

Four overhead Orbbec frames (640 x 480, depth registered to colour) with the
top-camera calibration in force when each was taken: the 2026-10-31 snapshot
(30 mm black cube and an open 100 x 100 x 40 mm black box on white cloth,
table at ~0.57 m; cube-touch calibration, 12.8 mm RMS) and three 2026-11-03
threading frames (11:44, 13:58, 14:21; a flat white ~4 mm cable laid three
ways on wood, table at ~1.0 m; ChArUco calibration, 5.4 mm RMS). Targets: the
cube, the box and two cable stretches per frame, masked by colour thresholds
(standing in for sam3). Each top-3 grasp was judged in the world frame against
geometry taken from the frame's own depth (`tests/bundles/_grconvnet_criteria.py`):
cube -- within 8 mm of its top face's centre, closing within 15 deg of a face
normal, opening 32-94 mm; box -- within 10 mm of a wall's line, closing within
20 deg of its normal, opening 8-94 mm (the box is wider than the hand);
cable -- within 6 mm of it, closing within 25 deg of its local normal,
opening 8-94 mm. Scripts: `tests/bundles/_grconvnet_station_eval.py`,
`_grconvnet_short_axis_eval.py`. Pass of the top grasp (1), of one of the top
3 (3), fail (x), no proposal (-), at zoom 4/3:

| proposer | cube | box | cable 14:21 a | b | 13:58 a | b | 11:44 a | b | top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **jacquard-d** (default) | 1 | 1 | x | 1 | x | x | x | - | 3/8 |
| cornell-rgbd | x | x | 1 | - | 1 | x | 1 | - | 3/8 |
| jacquard-rgbd | x | 1 | x | 1 | x | x | x | - | 2/8 |
| proposing-short-axis-grasps (mask -> OBB) | 1 | declined | x | 1 | 3 | 3 | x | x | 2/8 |

- **Solid parts: depth-only Jacquard.** On the cube it closed 4 mm off the top
  face's centre, 3 deg off a face normal, 52 mm opening (the cube is 30 mm);
  the support it estimated sat at z = -0.033 m against the calibration's
  table at -0.032 m, and the cube's height read 28.9 mm. On the open box it
  put the jaws across a wall (8 mm off its line, 16 deg, 14 mm opening) --
  proposing-short-axis-grasps declines the box (103 mm short axis > 94 mm
  hand) and has nothing else to offer.
- **Flat cable from 1 m: only RGB-D Cornell, and only sometimes.** The cable
  is ~2 px wide and ~3 mm tall there (the table's depth noise is 3.6 mm), so
  the depth-only network does not see it; Cornell found a grasp across the
  long curved stretch in all three frames (4-5 mm off, 4-8 deg) where
  short-axis put its stations on the chord between the bends, 10-36 mm off
  the cable. Short stretches suit short-axis. Neither is good enough to
  replace grasping-a-cable-lying-flat; use the wrist camera close up.
- **Crop size matters as much as the checkpoint.** Of zoom 0.67, 1, 4/3 and
  mask-filling crops (mask 30 % / 50 % of the crop), 4/3 was best or tied on
  top-1 for every checkpoint (depth-only Jacquard at 50 % fill tied it, 4/8
  in the top 3); with upstream's native-resolution crop (zoom 1) Cornell
  offered nothing that fit the hand on the cube and nothing at all on four
  cable stretches.
- Openings are generous (52-93 mm on the 30 mm cube): read `object_width_m`
  for the part itself.
- GPU and CPU gave the same grasps (pixels identical, quality within 2e-4).

**Speed** (default checkpoint, cube frame with mask and pose, warm, median):

| where | threads | per call | network alone |
| --- | --- | --- | --- |
| Core Ultra 7 265KF (20 cores), over gap's stdio RPC as gap serves it | 8 (default) | **75 ms** (p90 78) | ~60 ms |
| same | 4 / 1 / 20 | 91 / 311 / 685 ms | - |
| Xeon Platinum 8352S (AutoDL, shared), in-process | 8 / 16 / 4 / 1 | 123 / 115 / 192 / 467 ms | 87 / 79 / 157 / 432 ms |
| RTX 4090 D (AutoDL, same Xeon), in-process | - | 38 ms | **2.8 ms** |

Pre- and post-processing (numpy / scipy) take 7 + 5.5 ms on the desktop, 21 +
14 ms on the Xeon's slower cores, so on a GPU they are most of the call. All
20 threads of the hybrid desktop CPU are 9x slower than 8 (oversubscribed),
hence the default cap. First call: 0.8 s including 0.6 s to load.

## Licence

Code and checkpoints are BSD-3-Clause (the repository's LICENSE,
`LICENSE.robotic-grasping` here). The default checkpoint was trained on the
Jacquard dataset, whose EULA allows academic, non-commercial use only
(annotations CC BY-NC-SA 3.0); the Cornell checkpoint on the Cornell grasping
dataset, which states no licence. Check both before commercial use of the
weights.

## Gotchas

- From the overhead camera (tilted ~30 deg) the mask holds a box's front face
  too; a centre there reads a lower `top_z` and a smaller height. Gate on
  `depth_measured` and on `object_height_m` against what the part should be.
- `quality` ranks within one call; it is not comparable across checkpoints
  or crops, and a 0.9 peak on a cable stretch can still be across it wrongly.
- Grasps are top-down only; `close_heading_deg` is inert under
  `robot.go_to_pose` (wrist roll free): plan with orientation locked, as the
  short-axis skill's rules say.
- Trained on table-top objects about 5-25 cm across seen from ~0.5-1 m: parts
  much smaller than the 30 mm cube at 0.57 m need the wrist camera (not
  checked here) or a larger `zoom`.
