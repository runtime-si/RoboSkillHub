---
name: measuring-a-held-piece-from-the-wrist-camera
description: Measures where a flat held part (a jigsaw piece) sits in the
  gripper - x, y, yaw and its bottom's height relative to the tool centre, with
  uncertainty - from one wrist depth frame, by fitting the part's known top-down
  shape (its mask from before the pick) to what the camera sees past the
  blades - depth points at the part's top height say "part here", points on
  the surface below traced back to that height say "no part here" - with a
  coarse-to-fine search round the expected grip. Use when a pick may have
  slipped or turned the part and a place needs its true offset and yaw (e.g.
  seating-a-piece-by-wiggle-search's held_offset and yaw_error_deg), after
  the pick and again after turning it; no motion, about 1.5 s on CPU.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [jigsaw, puzzle, in-hand-pose, wrist-camera, depth, template-matching]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw, pick-place]
    provenance:
      authors: [runtime-si]
      paper: "W. Hess, D. Kohler, H. Rapp, D. Andor, Real-Time Loop Closure in 2D LIDAR SLAM, ICRA 2016 (scan matching against occupied and free evidence; idea only)"
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
  required_inputs:
    piece_mask: Mask
    metres_per_pixel: float
    grip_width_m: float
  # Optional inputs (wired from a script node where gap has no type for them): prior
  # ({x, y, yaw_deg}, the piece frame expected in the world; required in practice), camera
  # (CameraFrame), camera_name, image_yaw_deg, thickness, search_xy_m, search_yaw_deg, arm_id.
  produces_outputs:
    n_piece_points: int
    score: float
  exit_conditions:
    measured: The fit is sharp (spread of near-best poses under 1.5 mm and 4 deg); tcp_piece {x, y, z, yaw_deg} gives the body centre (z its bottom) and the piece frame's yaw in the tool frame, world_pose the same in the world, sigma the spread.
    ambiguous: A fit was found but too little of the piece is visible to pin it down (spread over 1.5 mm or 4 deg); the pose is returned with its spread.
    not_seen: No wrist camera with depth, an empty mask, or under 60 depth points at a piece-top height under the jaws (nothing held, or out of view).
  canonical_scripts:
    - measure_held: scripts/measure_held.py
  examples:
    - title: Measure the held piece after the pick
      path: examples/canonical_subgraph.json
---

# measuring-a-held-piece-from-the-wrist-camera

A pick puts the piece in the jaws a little off and a little turned (0.4-3.7
degrees in the YAM station twin), and a turn of the wrist adds more. A slot
takes a piece only within a few tenths of a millimetre and a degree, so the
place has to know where the piece really is. The wrist camera sees the held
piece between and beyond the blades; most of it is hidden, but its shape is
known from before the pick, so the visible part is enough to place the whole.

## How

1. **Points.** The wrist depth frame (every second pixel) is lifted to the
   world and to the tool frame. Points in the blade tips' footprint (15.5 x
   3.9 mm either side of the grip width) are dropped.
2. **The piece's top** is the densest 0.5 mm height bin from 3 to 22 mm under
   the tool centre (a held piece hangs its top 8-12 mm under it; the surface
   below is further). Points within 1 mm of it are the piece; points 4 mm or
   more below it are seen past it and are traced back along their rays to the
   piece's top height: free space there.
3. **Fit.** The template is the piece mask (dilated 0.5 mm), its frame on
   the body centre (the mask opened to remove the knobs, minimum-area
   rectangle). A pose scores the piece points inside it minus half the free
   points inside it. The grid search covers +-8 mm and +-20 degrees round the
   prior in 1 mm / 2 degree steps, then +-1.5 mm and +-3 degrees in 0.25 mm /
   0.5 degree steps.
4. **Uncertainty.** The standard deviation of the poses scoring at least 97 %
   of the best; over 1.5 mm or 4 degrees is `ambiguous`.

## Inputs

- `piece_mask`, `metres_per_pixel`: the piece's top-down mask before the
  pick (e.g. from scanning-the-board-into-a-wrist-mosaic), its +u at
  `image_yaw_deg` in the world (0 for that mosaic).
- `prior`: where the piece frame is expected in the world now, {x, y,
  yaw_deg} (after a pick: where it was, yaw 0 for the mosaic's mask; after a
  turn: that turned about the tool centre).
- `grip_width_m`: the jaws' grip width (for the blade footprint).
- `camera` (a CameraFrame) or `camera_name`; `thickness` (2 mm).

## Outputs

`tcp_piece` {x, y, z, yaw_deg} (tool frame; z the piece's bottom under the
tool centre), `world_pose`, `sigma` {x, y, yaw_deg}, `n_piece_points`,
`n_free_points`, `score`, `detail`.

## Measured

**YAM station twin** (jigsaw harness: a 21 x 24 mm, 2 mm piece held by its
harness grip block, wrist D405 at the real station's calibrated pose; the
template is the piece's mesh outline or its mosaic mask; truth is the piece
body's pose):

| case | x / y / z error | yaw error | sigma reported |
|---|---|---|---|
| one held frame (offline), priors 0-6 mm and 0-15 deg off | +0.16 / -0.52 / -0.09 mm (every prior) | -1.5 deg | 0.26 / 0.24 mm, 1.3-1.5 deg |
| end-to-end chain, after the pick (mosaic mask), trial 0 | -0.17 / -0.48 / -0.09 mm | -0.18 deg | |
| same, trial 1 | +1.59 / +0.93 / -0.11 mm | -1.1 deg | |
| same, trial 1 after a 136 deg turn | -0.45 / -2.13 / +0.33 mm | -1.1 deg | |

169 piece points (every second pixel) are visible past the blades; a fit
takes 1.2-1.5 s. Part of the 1-2 mm errors in the chain is the mosaic mask's
own error (0.5 mm per pixel) and the body centre's definition (0.3 mm from the
cell centre on this piece). More chain trials are in the pull request.

## Limits

- **The shape must be known**: a mask of the same piece from before the pick.
- **Flat parts held top-down** only; the piece's top must be level within a
  few degrees (a tilted piece spreads its top over several height bins).
- **The prior must be within the search window** (8 mm, 20 degrees).
- Blade footprint and slab are the YAM linear_4310's (station model); other
  grippers need their numbers.

## The constants, and why

`SLAB_M` 3-22 mm (blade tips 10 mm under the tool centre; a 2 mm piece held
by its edge; the mat 30 mm under the tool 3 cm up), `TOP_TOL_M` 1 mm,
`FREE_BELOW_M` 4 mm, `BLADE_HALF_WIDTH_M` 8.5 mm, `BLADE_THICKNESS_M` 4.5 mm,
`SEARCH_XY_M` 8 mm, `SEARCH_YAW_DEG` 20, `GOOD_FRACTION` 0.97, `MIN_POINTS`
60, `MAX_SPREAD_M` 1.5 mm, `MAX_SPREAD_DEG` 4.
