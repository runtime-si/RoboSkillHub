---
name: selecting-free-placement-spots
description: Picks world-frame spots on a named surface where a held object can
  be set down - asks a pointing model for points in the vacant area (RoboPoint's
  question to gemini-er.point) or a VLM for free tiles of a grid drawn over the
  image (MOKA), back-projects each through calibrated depth, keeps those whose
  footprint is level, flat, on the surface and clear of anything standing on it,
  and ranks them by free room. Use when a place step needs a target and the task
  says only where in general ("put it somewhere free on the table", "on the
  board", "on the empty part of the tray"), from one overhead RGB-D frame.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [placement, free-space, vlm, pointing, depth, select]
  hub:
    stage: select
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place]
    tuned_on: [yam-station-overhead-rgbd]
    provenance:
      authors: [runtime-si]
      code: https://github.com/moka-manipulation/moka/blob/f6b642514e5eb5f11f45d9da22af7b4b97514f6d/moka/planners/visual_prompt_utils.py
      upstream_license: MIT
      paper: https://arxiv.org/abs/2403.03174
gap:
  allowed_tools:
    - gemini-er.point
    - vlm.query
    - geometry.pixel_to_world_point
    - geometry.mask_to_world_points
  required_inputs:
    observation: Observation
    region: string
  produces_outputs:
    place_position: Vec3
    spot_count: int
    footprint_radius: float
  exit_conditions:
    found: At least one model-proposed spot passed the depth check; place_position is the best, place_positions all of them, ranked.
    none_free: The model named no spot or tile, or every candidate failed the depth check (tilted, occupied, edge, rough, no_depth); `detail` says which.
  canonical_scripts:
    - select_free_spots: scripts/select_free_spots.py
  examples:
    - title: Free spots on a named surface from the overhead camera
      path: examples/canonical_subgraph.json
---

# selecting-free-placement-spots

A model proposes, depth disposes. The model knows what a table, a board and
"free" look like; the depth knows whether the footprint is actually level and
clear. Neither is enough alone: the model happily points at a spot 2 cm from a
gripper finger, and depth cannot tell a bare board from one with a cable lying
flat on it.

## Proposers (`method`)

- **`point`** (default): RoboPoint's free-space question - "several spots
  within the vacant area on {region} where an object about {size} cm across
  can be set down without touching anything" - sent to `gemini-er.point`
  (Gemini Robotics-ER) in place of a RoboPoint checkpoint. One call, ~5 s.
- **`grid`**: MOKA's visual prompt. The image gets a `grid_cols` x `grid_rows`
  grid, columns `a, b, ...` left to right, rows `1, 2, ...` bottom to top, each
  label at its tile's centre; `vlm.query` is asked for a JSON list of tiles
  whose centre shows the bare surface of `region`, best first. Each chosen tile
  is sampled at its centre and on a 3 x 3 lattice (MOKA drew one uniform
  sample) and the tile's best passing sample is kept. Coarser than pointing
  (one spot per 80 px tile) and slower with a thinking model.

## The depth check (every candidate pixel)

1. `geometry.pixel_to_world_point` gives the spot; no depth there: `no_depth`.
2. `geometry.mask_to_world_points` back-projects a pixel box round it large
   enough for twice the footprint; the footprint is the points within
   `footprint_radius` of the spot in world x-y (so perspective and tilt of the
   camera do not matter).
3. The support plane is fitted (least squares, trimmed twice at 3 robust
   sigmas) to the points within 8 mm of the footprint core's median height,
   out to twice the footprint. Then, in order:
   - `tilted`: the plane leans more than `max_tilt_deg`;
   - `occupied`: a 10 mm cell of the footprint is mostly more than
     `obstacle_height` over the plane;
   - `edge`: a 20 mm cell is mostly more than 10 mm under it, or an eighth
     of the footprint's outer half has almost no points (past a table's far
     edge the depth lands on the floor, a metre away in x-y; behind a tall
     object it is occluded);
   - `rough`: the trimmed residual RMS exceeds `flat_rms`.
4. A passing spot's `free_radius` grows in four rings out to twice the
   footprint while each ring stays clear by the same tests.

Spots are ranked by `free_radius`, then the model's order; a spot within one
footprint radius of a better one is dropped as `duplicate`.

## Inputs

- `observation`: the camera is `camera_name`, or else the first one whose name
  does not say wrist / hand. Needs `rgb`, metric `depth`, `intrinsics` and the
  camera-to-world `pose`; world +z must be up.
- `region`: the surface in words ("the table", "the black breadboard").
- `object_obb` (optional): the held object's box; its footprint is the circle
  round its corners seen from above. Else `footprint_radius` (0.025 m).
- `clearance` (0.01 m) is added to either.
- `method` (`point`), `max_spots` (8), `grid_cols` / `grid_rows` (8 x 6),
  `max_tilt_deg` (4), `flat_rms` (0.006 m), `obstacle_height` (0.012 m).

## Outputs

`place_position` (best, world frame, z on the fitted support - add the
object's half height for its centre), `place_positions` (all, ranked; a list
of Vec3), `spots` (per spot: pixel, position, `free_radius`, `tilt_deg`,
`rms_m`, `source`, `model_rank`), `rejected` (pixel, `reason`, `source`),
`spot_count`, `footprint_radius` (with clearance) and `detail`.

## What it cannot see

- Anything lower than `obstacle_height` (12 mm): a cable lying flat (4 mm on
  the station) passes the depth check. Only the model guards it.
- Thin wire fixtures (the station's loops) give few depth returns; a
  footprint next to one passes unless enough of it shows in depth.
- An overhang of under ~1 cm past a 16 mm step: the depth blurs the
  breadboard's edge over ~10 mm.
- It does not check reach; follow with a reach or IK check before moving.

## Measured on the YAM station (overhead camera, 2026-11-03 frame)

Real frame `cable_frame_20261103-1421.npz` (640 x 480, depth in metres) and the
camera pose calibrated then (ChArUco, RMS 5.4 mm). Models:
`gemini-robotics-er-2-preview` for `point`, `gemini-3.8-flash` for `grid`.
Every returned spot was drawn with its footprint circle projected into the
image and looked at.

| run | region | returned | on genuinely free surface |
|---|---|---|---|
| point x 3 | the table | 5, 4, 5 | 14 / 14 |
| point x 3 | the black breadboard | 3, 1, 1 | 5 / 5 |
| grid | the table | 5 | 5 / 5 |
| grid | the black breadboard | 1 | 1 / 1 |
| point, 160 x 120 mm object (110 mm footprint) | the table | 1 of 4 proposed | 1 / 1 |
| point, same object | the black breadboard | none_free (model named nothing) | - |

With the large object the depth check rejected 3 of the model's 4 table
points as `occupied` (a gripper finger or the board inside the 110 mm
footprint) and kept the one 114 mm from anything standing. With the 35 mm
footprint the model's points never needed rejecting on this frame; one
earlier `point` run on the breadboard returned no point at all (`none_free`).
The model returns only 1-3 spots on the breadboard, between its wire
fixtures. `point` took 4-7 s a region; `grid` with gemini-3.8-flash 18-77 s.

Depth-check alone, on a 16 px lattice over the same frame (35 mm footprint):
15 of 25 footprints inside the breadboard passed (the other 10 are within a
few centimetres of its upright loops: 5 `occupied`, 5 `edge` from the depth
shadow beside them), 2 of 20 footprints overhanging its edge by more than
8 mm passed (12 `tilted`, 4 `occupied`, 2 `edge`), and the table read as
free except round the grippers, at its far edge and next to the board.

## The constants, and why

- **`FLAT_RMS = 6 mm`, `OBSTACLE_HEIGHT = 12 mm`**: the overhead camera's
  depth about a plane fitted to 30 mm patches of the bare table and board:
  2nd-98th percentile residuals -8.5..+8.3 mm, RMS 3-4 mm.
- **`MAX_TILT_DEG = 4`**: plane fits over twice the footprint gave median
  1.3 deg (90th percentile 3.4) on bare table, at most 2.8 on the board
  interior, and 2.7-5.9 deg on footprints straddling the board's edge.
- **`DROP_DEPTH = 10 mm`**, **20 mm drop cells**: the breadboard stands 16 mm
  over the table; its 6 mm holes read deep and fill 10 mm cells.
- **`SUPPORT_BAND = 8 mm`**: half that step, so a footprint whose core is on
  the board fits the board.
- **8 x 6 grid**: MOKA used 5 x 5 on cropped views; on a full 640 x 480
  frame 8 x 6 gives 80 px tiles, about 8 cm at 1 m.
- **`FOOTPRINT_RADIUS = 25 mm`, `CLEARANCE = 10 mm`**: a 50 mm part with a
  centimetre round it when no box is given.

## Provenance

Adapted from MOKA (MIT, Kuan Fang, Fangchen Liu):
`moka/planners/visual_prompt_utils.py` (`annotate_grid`,
`sample_waypoint_in_tile`, `parse_json_string`) and the tile convention of
`prompts/visual_prompt_planner/select_motion.txt`. The free-space question is
RoboPoint's (Apache-2.0, `robopoint/serve/gradio_web_server.py`, Wentao Yuan
et al., https://arxiv.org/abs/2406.10721), asked of `gemini-er.point`. The
grid asks for free tiles of one region instead of a motion; the tile is
sampled on a fixed lattice; the depth check, the world-frame output and the
ranking are new.
