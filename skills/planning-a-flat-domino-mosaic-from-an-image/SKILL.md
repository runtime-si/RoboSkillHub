---
name: planning-a-flat-domino-mosaic-from-an-image
description: Plans a domino mosaic of a picture, every tile lying flat with its
  pips up - DominoArtSolver's integer program chooses, for a number of complete
  double-six sets, which tile covers which two neighbouring grid cells and which
  way round, so the pips' darkness reproduces the image (all tiles used once per
  set; a smaller grid picks from spare tiles). Returns an ordered placement list
  in world (grid origin, cell pitch and yaw given; 48 x 24 mm tiles by default)
  with each tile's value, which end carries which value, its pose, its arm, the
  copy of the set it comes from and the free sides for the fingers; the order
  places the tiles farthest from each arm first and never has an arm reach across
  a placed tile (one or two arms). Also a preview render, per-set usage and the
  image error. CPU only, no motion. Use before laying a domino mosaic with the
  arm, to know which tile to fetch from the supply (reading-domino-pips-and-orientation
  identifies it there), where to put it and which way it faces.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [domino, mosaic, image, integer-programming, milp, pulp, cbc, highs, build-order, plan, cpu]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/ArnoldF/DominoArtSolver/blob/3f9e260ea97a6ad659784fc6931bbb64d9196eba/src/domino_model.py
      upstream_license: MIT
      paper: https://neos-guide.org/case-studies/cs/domino-artwork/
gap:
  requires: {}
  allowed_tools: []
  required_inputs:
    # OPTIONAL extra inputs (gap has no list, array or union type; wire them from a
    # script node in the same subgraph): `image` (H x W grey or RGB array, or a
    # CameraFrame), `max_pips`, `rows` / `cols`, `grid_origin` (point or Se3Pose),
    # `anchor`, `grid_yaw_deg`, `pitch_m`, `tile_size`, `pip_contrast`, `arms`,
    # `solver`, `time_limit_s`, `accept_suboptimal`, `cost`, `preview_path`,
    # `preview_cell_px`, `finger_thickness`, `finger_width`, `finger_margin`.
    image_path: str
    sets: int
  produces_outputs:
    placements_json: str
    pose_sequence: PoseSequence
    first_pose: Se3Pose
    n_tiles: int
    rows: int
    cols: int
    sets: int
    max_pips: int
    objective: float
    cell_rmse: float
    floor_rmse: float
    image_rmse: float
    correlation: float
    solver: str
    solver_status: str
    solve_time_s: float
    reach_over: int
    steps_without_finger_room: int
    preview_path: str
  exit_conditions:
    planned: Every cell is covered once and every tile used `sets` times (or at most `sets` times on a given smaller grid); `placements` (in placing order), `pose_sequence`, `by_arm`, `per_set`, `supply`, `values`, `preview` and the error measures are filled. `problems` lists what a robot should know - poses in the grid's own frame (no `grid_origin`), a grid that stretches the image, steps without finger room at this pitch, a plan found by the time limit but not proven optimal, steps that reach across a placed tile.
    not_enough_sets: "`sets` is under 1, or the given `rows` x `cols` grid needs more tiles than the sets hold (`detail` says how many sets it needs)."
    solver_timeout: The solver found no tiling within `time_limit_s`, or (with `accept_suboptimal` false) found one but could not prove it optimal.
    invalid_image: No image, an unreadable file, an array that is not H x W (x 3), an image smaller than the grid in pixels, or no contrast between the grid's fields.
    out_of_reach: Some cells are outside every arm's `reach_m` (or inside its `min_reach_m`); `out_of_reach` lists them as [row, col]. Checked before solving, so nothing is planned.
    solver_unavailable: Neither PuLP with a CBC binary nor SciPy's `milp` (HiGHS) could be loaded.
    solver_failed: The solver reported the model infeasible or returned values that are not a tiling (not seen; the model is always feasible).
    invalid_input: A grid with an odd cell count or only one of rows / cols, a pitch smaller than the tile, an unknown option, a bad origin or arm (also the route of a raised error).
  canonical_scripts:
    - plan_mosaic: scripts/plan_mosaic.py
  examples:
    - title: Plan a mosaic of a picture from complete double-six sets, one arm
      path: examples/canonical_subgraph.json
---

# planning-a-flat-domino-mosaic-from-an-image

Robert Bosch's domino art as an integer program, ported from DominoArtSolver,
and turned into something an arm can build: which tile to fetch, where it goes
on the table, which end points where, and in what order.

## What it computes

1. **The image** (`image` array or CameraFrame, or `image_path` read in grey as
   upstream does). RGB arrays are taken as gap's RGB.
2. **The grid.** A double-six set has 28 tiles, so `sets` complete sets cover
   56 x `sets` cells. Upstream's rule picks rows x cols = that count with
   cols / rows closest to the image's aspect (any divisor pair; a count with few
   divisors stretches the image, noted in `problems`). Or give `rows` and `cols`
   (even cell count, at most 56 x `sets`): a smaller grid than the sets uses each
   tile at most `sets` times, so the solver picks the tiles it wants.
3. **The target.** Each field's mean brightness (upstream's fields: floor-sized,
   the remainder pixels unused) scaled to the pip range: `(N + 1) (b - min) / range
   - 0.5`, N = `max_pips` (6; upstream 9). With `pip_contrast` dark_on_light
   (default: white tiles, black pips) more pips mean darker, so the target is N
   minus that; light_on_dark is upstream's black tiles with white pips.
4. **The integer program** (DominoModel). Binary x[tile, placement] for every
   tile of one set and every horizontal or vertical pair of neighbouring cells;
   every cell covered exactly once; every tile used `sets` times; minimum total
   cost, a placement's cost being the squared error of the tile's two values
   against the two cells' targets in the better end order, truncated to an
   integer as upstream does (`cost: squared` keeps the fraction). Solved by CBC
   through PuLP when PuLP is installed (upstream's solver), else by HiGHS through
   SciPy (`scipy.optimize.milp`), with `time_limit_s`. The returned values are
   checked to be a tiling before use.
5. **Which way round.** The low value goes on the cell where it fits better
   (untruncated error, ties: low value first). End a is always the low end:
   `value_ab = value = [low, high]`.
6. **World poses.** `grid_origin` is the anchor cell's centre on the table top
   (`anchor` top_left: cell (0, 0), the image's top-left corner; `center`: the
   grid's centre). Columns run along the grid's +x, rows along its -y, so seen
   from above with the grid's +y up the picture is upright; the grid's yaw comes
   from an Se3Pose origin's rotation or `grid_yaw_deg`. Cells are `pitch_m` apart
   (default tile width + 2 mm = 26 mm); a tile spans two cells. Per tile:
   `position` (its centre, z = the origin's z, the table top), `top_z`, `yaw_rad`
   of `axis_world` (end a to end b, the convention of
   reading-domino-pips-and-orientation), `ends_world`, `half_centers_world` and
   `pose` (an Se3Pose whose x axis is `axis_world`).
7. **The placing order.** Each tile goes to the nearest arm that reaches it
   (`arms`: `{arm_id, base, reach_m?, min_reach_m?}` in world; default one arm
   at the world origin, or 0.3 m beyond the image's bottom edge without a
   `grid_origin`). A tile must be placed before any tile lying on the straight
   line from its arm's base to it (else that arm would reach across a placed
   tile); among the tiles free to go, the one farthest from its arm goes first.
   Each step's line is checked again against the tiles placed before it
   (`reaches_over`, `reach_over`; 0 in every test and sweep). `by_arm` lists the
   steps of each arm. Two arms working at the same time are not checked against
   each other: gate concurrent steps with checking-dual-arm-clearance.
8. **Finger room.** At each step, which grasp axes have both finger spots free
   of placed tiles: `across` (fingers on the long sides) and `along` (on the
   ends), each spot a 13 mm deep (10 mm fingertip + 3 mm margin), 26 mm wide
   region (planning-brick-assembly-order's linear_4310 estimates). At the default
   26 mm pitch (2 mm gaps) most later steps have none (`steps_without_finger_room`,
   in `problems`): release the tile from just above, or slide it in, or use a
   pitch of 37 mm or more (every step then has `across` free).
9. **Supply.** `set_index`: the k-th copy of a tile in placing order comes from
   set k; `per_set` (used count and unused tiles per set) and `supply` (tiles
   needed per value, "a-b"). A double's two ends are the same
   (`is_double`: its yaw only matters mod 180 deg).
10. **Error and preview.** `cell_rmse` (pips, placed values against the targets),
    `floor_rmse` (what rounding each cell on its own would reach: the floor for an
    unlimited supply), `image_rmse` (0..1 grey, the mosaic's shading - pips / N,
    inverted for dark pips - against the image's pixels on the same scale),
    `correlation` (placed values with targets), `objective`. `preview` renders the
    tiles with their pips (RGB, `preview_cell_px` per cell), `target_preview` the
    per-cell target; `preview_path` writes the two side by side.

## Feeding a pick-and-place and the pip reader

For each step, read supply tiles with reading-domino-pips-and-orientation until
one has `value` = the step's `value` (sorted pair). Its `value_ab` says which
end is which: if its end a holds the step's low value, turn its `axis_world`
onto the step's `axis_world`; else onto the opposite (`yaw_rad` + 180 deg).
Place it top-down at the step's `pose` (the tile's bottom centre on the table;
add the tile thickness for the hand's release height). `pose_sequence` holds
the same poses in order for an executor that takes a PoseSequence.

## Inputs

`sets` (int), `image_path` or `image`; optional `max_pips` (6), `rows`, `cols`,
`grid_origin`, `anchor` (top_left | center), `grid_yaw_deg`, `pitch_m` (0.026),
`tile_size` ([0.048, 0.024, 0.0075] m), `pip_contrast` (dark_on_light |
light_on_dark), `arms`, `solver` (auto | cbc | highs), `time_limit_s` (120),
`accept_suboptimal` (true), `cost` (upstream | squared), `preview_path`,
`preview_cell_px` (24), `finger_thickness` / `finger_width` / `finger_margin`
(0.010 / 0.020 / 0.003 m).

## Outputs

`placements` (per step: `step`, `arm_id`, `value`, `value_ab`, `is_double`,
`set_index`, `cells` ([row, col] of end a, end b), `orientation` (row | column),
`position`, `yaw_rad`, `yaw_deg`, `axis_world`, `ends_world`,
`half_centers_world`, `pose`, `top_z`, `distance_to_arm`, `finger_room`,
`reaches_over`), `placements_json`, `pose_sequence`, `first_pose`, `by_arm`,
`rows`, `cols`, `n_tiles`, `sets`, `max_pips`, `per_set`, `supply`, `grid`
(pitch, origin, yaw, frame, extent, aspects, whether every tile is used),
`target`, `values`, `preview`, `target_preview`, `preview_path`, the error
measures, `solver`, `solver_status` (optimal | feasible), `solve_time_s`,
`reach_over`, `steps_without_finger_room`, `out_of_reach`, `problems`, `detail`.
The lists, dicts and arrays are not in `produces_outputs`, which takes only gap
schema types.

## Solvers and licences

- **PuLP** (MIT) builds the model as upstream does. PuLP 2.x / 3.x ship a CBC
  2.10.3 binary (`PULP_CBC_CMD`; its `coin-license.txt` is the Eclipse Public
  License 1.0); PuLP 4.x moved CBC to the `cbcbox` package (`pulp[cbc]`, EPL-2.0,
  `COIN_CMD`) and changed its API (`LpProblem.add_variable_dicts`, `LpSolveStats`).
  The skill works with both and was run with PuLP 2.9.0 and 4.2.0 + cbcbox 2.929.
  CBC runs as a separate program; EPL is a weak copyleft licence that allows that
  use - nothing of CBC is copied here.
- **HiGHS** (MIT) through SciPy (BSD-3-Clause) `scipy.optimize.milp`, the
  fallback: SciPy is already in the hub's environment, PuLP is not (install
  `pulp` or `pulp[cbc]` to use CBC). Same model, same optimum (tested).

## Measured

CPU only (Intel Core Ultra 7 265KF, 20 threads), `tests/bundles/_domino_mosaic_eval.py`.
Synthetic 400 x 480 px pictures (a Gaussian blob, a 70/30 diagonal gradient,
concentric rings, a dark letter R on white; 2 grey levels of noise) and
upstream's `Einstein.jpg` (not committed). Double-six, dark pips on white
tiles, upstream cost. Cell RMSE in pips (0-6) against the target; image RMSE
in grey (0-1) of the mosaic's shading against the picture's pixels; solve time
of CBC (PuLP 4.2.0 + cbcbox 2.929) / HiGHS (SciPy 1.15.3).

Every tile of every set used (upstream's rule):

| picture | sets | grid | solve s CBC / HiGHS | cell RMSE | image RMSE |
|---|---|---|---|---|---|
| blob | 1 / 2 / 4 / 8 / 16 / 24 | 8x7 ... 42x32 | 0.09 / 0.03 ... 16.9 / 10.3 | 1.71 / 1.74 / 1.84 / 1.89 / 1.93 / 1.86 | 0.268 / 0.269 / 0.281 / 0.289 / 0.294 / 0.285 |
| gradient | 1 / 2 / 4 / 8 / 16 | 8x7 ... 32x28 | 0.12 / 0.04 ... 8.2 / 7.7 | 1.10 / 1.14 / 1.25 / 1.26 / 1.30 | 0.189 / 0.194 / 0.210 / 0.212 / 0.218 |
| rings | 1 / 2 / 4 / 8 / 16 | 8x7 ... 32x28 | 0.21 / 0.04 ... 5.4 / 53.3 | 0.69 / 0.46 / 0.58 / 0.55 / 0.78 | 0.541 / 0.393 / 0.310 / 0.219 / 0.177 |
| letter | 1 / 2 / 4 / 8 / 16 | 8x7 ... 32x28 | 0.09 / 0.03 ... 3.7 / 1.8 | 3.15 / 3.27 / 3.32 / 3.38 / 3.43 | 0.517 / 0.516 / 0.514 / 0.516 / 0.516 |
| Einstein | 1 / 2 / 4 / 8 / 16 / 24 | 8x7 ... 42x32 | 0.11 / 0.07 ... 6.9 / 12.4 | 0.97 / 0.93 / 1.06 / 1.06 / 1.15 / 1.19 | 0.264 / 0.228 / 0.227 / 0.215 / 0.218 / 0.219 |

Grids by sets: 8x7, 14x8, 16x14, 28x16, 32x28, 42x32 (the 2- and 8-set grids
stretch a 5:6 picture to 0.57). Solve times by sets, CBC: 0.09-0.21, 0.13-0.58,
0.44-1.33, 1.17-2.38, 3.7-8.2, 6.9-16.9 s; HiGHS: 0.03-0.07, 0.04-0.49,
0.21-2.28, 0.59-4.43, 1.8-53.3, 10.3-12.4 s. Building the model and the rest
of the plan add 0.02-1.3 s.

More sets help only where detail matters (the rings: image RMSE 0.54 -> 0.18);
on smooth or two-tone pictures the error stays flat or grows, because every
tile must be used and the sets' values (0-6 equally often) do not match the
picture's histogram - the letter needs mostly blanks and sixes and stays at
~3.3 pips. Rounding each cell alone would reach 0.27-0.48 pips (`floor_rmse`).

With spare tiles - a grid of half the sets' tiles (`rows` / `cols`), so each
tile may be left out - the same pictures (CBC, 0.06-1.5 s):

| picture | sets (grid) | cell RMSE | image RMSE |
|---|---|---|---|
| blob | 2 (8x7) / 4 (14x8) / 8 (16x14) / 16 (28x16) | 0.80 / 0.87 / 0.93 / 0.96 | 0.128 / 0.127 / 0.134 / 0.135 |
| gradient | same | 0.49 / 0.52 / 0.53 / 0.52 | 0.092 / 0.094 / 0.110 / 0.107 |
| rings | same | 0.47 / 0.47 / 0.47 / 0.48 | 0.538 / 0.400 / 0.323 / 0.228 |
| letter | same | 1.88 / 2.00 / 2.11 / 2.20 | 0.332 / 0.329 / 0.332 / 0.332 |
| Einstein | same | 0.51 / 0.49 / 0.49 / 0.47 | 0.230 / 0.189 / 0.172 / 0.152 |

So Einstein on a 28 x 16 grid: 0.215 image RMSE from 8 sets used whole, 0.152
from 16 sets with half left in the box.

Other checks:

- **Same optimum as upstream's own code**: DominoArtSolver's `src/` run
  unchanged (PuLP 2.9.0, its bundled CBC 2.10.3) and this port (`max_pips` 9,
  `light_on_dark`, `solver: cbc`) on upstream's three pictures, 1 and 2
  double-nine sets: the same grids and objectives (Einstein 165 / 393,
  MonaLisa 549 / 1276, Mountains 311 / 568).
- **Upstream's README case** (Einstein, 18 double-nine sets, 1980 tiles,
  55 x 36 cells as upstream's): CBC 2.10.3 optimal in 82.5 s (84.3 s in all),
  cell RMSE 1.70 pips (0-9).
- **PuLP 2.9.0 vs 4.2.0**: both optimal on every case tried (1-24 sets), CBC
  2.10.3 0.22-16.2 s, cbcbox 0.09-16.9 s. CBC and HiGHS reach the same objective
  (tested).
- **`cost: squared`** (untruncated) lowers the cell RMSE by 0.01-0.09 pips
  (rings, 2 sets: 0.46 -> 0.37) at similar solve times; the default stays
  upstream's.
- **Order**: no step reaches across a placed tile in any run (one arm at the
  origin; tests also with one arm beside the grid and two arms either side).
- **Finger room at the default 26 mm pitch**: 46-90 % of steps have no grasp
  axis with 13 mm free on both sides; at 37 mm every step has `across` (tested).
- The solver brute-force check: on a 3 x 4 grid with one double-two set (11
  tilings x 720 assignments) HiGHS finds the exhaustive optimum, both costs.

Not measured: placing a mosaic with an arm, in simulation or on the YAM (this
skill has no motion; no sim check is needed for a plan stage).

## The constants, and why

- **`TILE_SIZE` 48 x 24 x 7.5 mm**: the hub's other domino skills' tile.
- **`GAP` 2 mm**: a choice, so tiles do not touch; see finger room for what a
  hand needs.
- **`FINGER_THICKNESS` 10 mm, `FINGER_WIDTH` 20 mm, `FINGER_MARGIN` 3 mm**:
  singulating-by-pushing's linear_4310 estimates, as planning-brick-assembly-order.
- **`CORRIDOR_SHRINK` 1 mm**: a line running along the gap between two tiles
  does not count as crossing them.
- **`DEFAULT_TIME_LIMIT` 120 s**: twice the slowest double-six solve in
  Measured (HiGHS, 53 s at 16 sets; CBC at most 17 s up to 24 sets).
- **`max_pips` 6, pip contrast dark_on_light**: the common double-six set
  (upstream: double-nine, white pips on black tiles; both are options).
- Pip layouts of the preview: the usual 0.25 / 0.5 / 0.75 positions.

## Preconditions and boundaries

- No robot motion, no tools, no simulation: a plan stage on CPU. The placing
  itself (a flat tile laid top-down at a pose and yaw) is a pick-and-place
  skill's job; this skill says where the fingers have room.
- The arms are points on the table plane for the ordering and the reach check;
  their links are not modelled. "Reach across" means the straight line from the
  arm's base to the tile, in plan view.
- Upstream uses every tile of every set, so an image whose brightness histogram
  is far from the sets' (every value 0-6 equally often) is matched poorly: a
  two-tone letter stays at ~3.3 pips cell RMSE however many sets (Measured).
  A grid smaller than the sets (`rows`, `cols`) lets the solver leave tiles out
  and roughly halves the error.
- The image is stretched to the grid's aspect when the cell count has no
  divisor pair near it (8 double-six sets on a 5:6 picture give 28 x 16 cells,
  aspect 0.57); give `rows` / `cols` to avoid it.

## Provenance

- `scripts/mosaic_model.py` is adapted from DominoArtSolver
  (ArnoldF/DominoArtSolver, commit 3f9e260ea97a6ad659784fc6931bbb64d9196eba,
  MIT, Copyright (c) 2025 Florian Arnold Thurm, `LICENSE.dominoartsolver`):
  `src/domino_model.py` DominoModel, `src/preprocessing.py`,
  `src/image_processing.py` and the end-order rule of `src/plotting.py`. The
  file's header lists the changes: a pip range parameter (double-six default),
  dark-pip targets, a constant image refused, at-most usage on a smaller grid,
  per-cell constraint lists, both PuLP APIs with a time limit and a HiGHS
  fallback, the solution checked, the end order on the untruncated error.
- The model is Robert Bosch's domino artwork formulation
  (https://neos-guide.org/case-studies/cs/domino-artwork/, http://www.dominoartwork.com/).
- `scripts/plan_mosaic.py` (world poses, order, finger room, preview, errors)
  is new. Farthest-from-the-arm first is the rule planning-a-domino-layout uses
  (DominoRobot's FieldPlanner rule, GPL-3.0: the rule only).
