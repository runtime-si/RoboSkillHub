---
name: verifying-build-against-model
description: Compares a brick build with its target model up to the current
  step - LTRON's assembly matching (the best assignment of observed to target
  bricks by shape, colour and pose within a tolerance, F1 score and edit
  distance) - and reports the bricks that are missing, extra, misplaced (same
  brick elsewhere), the wrong colour, unseated (top a stud height high, 1.7 mm
  LEGO / 4.5 mm DUPLO, or at one end) or ahead of the instructions, with an ordered
  repair list. The observed bricks come either as stud-grid cells from upstream
  perception, or from a depth frame and the plate pose - every pixel's ray is
  cast against the target's expected height map and the measured height
  compared per stud cell; colours by tools/colour when the frame has RGB. Use
  after each placement or at the end of a build, to decide whether to go on,
  re-press, remove, move or place a brick.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [lego, duplo, assembly, verification, depth, matching, cpu]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      code: https://github.com/aaronwalsman/ltron/blob/253be3e55cef068c1842fd9a7d2e2b3c22d19d7b/ltron/matching.py
      upstream_license: MIT
      paper: arXiv:2207.13738
gap:
  requires: {}
  allowed_tools:
    - robot.get_observation
    - colour.classify
  required_inputs:
    target: string
  produces_outputs:
    score: float
    edit_distance: int
    n_target: int
    n_matched: int
    n_missing: int
    n_extra: int
    n_misplaced: int
    n_wrong_colour: int
    n_unseated: int
    n_unverified: int
    report_json: string
  exit_conditions:
    matches: Every target brick up to the step is matched (seated where its seating could be measured), nothing extra; bricks of later steps already in place are listed in `ahead`, bricks whose seating is within the depth noise in `seat_unresolved` (a note, unless require_seating).
    misplaced: A target brick is missing and a brick of the same shape and colour that matches no target brick is elsewhere (shifted, turned or a layer off); `misplaced` pairs them with the shift.
    wrong_colour: A brick of the right shape sits at a target brick's place but its colour is another; `wrong_colour` lists expected and seen.
    extra_bricks: A brick (or depth blob) matches no target brick of this or a later step and pairs with no missing brick; `extra` lists it.
    unseated: A matched brick's top reads more than half a stud over its seated height (`how` raised), or its highest stud cell does (`how` one end - seated at one end, on the studs at the other), significant at 3 (brick) / 4 (cell) standard errors of the depth noise; `unseated` lists the heights.
    missing_bricks: Some target bricks up to the step are not there, and nothing above explains them; `missing` lists them.
    unverifiable: No camera frame / depth, no depth point on the plate, or target bricks seen too little to judge (`unverified`); with require_seating, also a seating within the noise.
  canonical_scripts:
    - verify_build: scripts/verify_build.py
  examples:
    - title: Verify a LEGO build against its parsed steps from the wrist camera
      path: examples/canonical_subgraph.json
---

# verifying-build-against-model

A build step is done when the bricks are where the model says, in the right
colour, pressed down. This skill compares what is there with the target up to
the current step and names every difference with the repair it calls for. Pure
CPU: 1-4 ms on a 100-brick list, ~0.15 s on a 640x480 depth frame (Intel Core
Ultra 7 265KF; the offset search included).

## How it composes

- **Before it**: the target from parsing-ldraw-build-instructions (`steps` /
  `steps_json`, pass `upto_step`) or planning-brick-assembly-order
  (`sequence` / `sequence_json`, pass `upto_seq`), or a hand-written brick
  list. The observation either from perception (cells, e.g. each brick's
  computing-brick-pose-from-stud-grid inverse output with its `residual`, or
  pose6d poses as `brick_pose` with `plate_pose`) or a depth frame with the
  baseplate's pose (`camera`, or `camera_name` for `robot.get_observation`).
- **After it**, by route: `misplaced` / `extra_bricks` / `wrong_colour` -
  pick the brick off (`repairs` gives cells, top layer first); `unseated` -
  pressing-a-brick-onto-studs again at the brick's seat
  (computing-brick-pose-from-stud-grid with its cell); `missing_bricks` - place
  the brick (the planner's next entry); `matches` - next step;
  `unverifiable` - look again from the wrist (refining-target-from-wrist-
  closeup) or from another side.

## Conventions

The hub's stud grid (computing-brick-pose-from-stud-grid,
parsing-ldraw-build-instructions, planning-brick-assembly-order,
checking-design-connectivity-and-collisions): a brick is `{cell: {x, y, layer,
rotation}, size: "AxB", colour}`, (x, y) its lowest-index corner cell, layer 1
on the plate, A studs along the plate's x at rotation 0 and along y at 90 /
270; also `cell: [x, y, layer]`, flat `x / y / layer`, ICL's `ori`, `brick_size`,
plates and tiles with `height_units` / `z_units` (parser's), `step`, `seq`,
`id`. "4x2" at 90 is "2x4" at 0; a half turn of a rectangle and any quarter
turn of a square brick are the same brick. Colours are tools/colour's names
(LDraw); an unknown colour matches any. The plate frame is
computing-brick-pose-from-stud-grid's: origin at the centre of the plate's top
(stud bases), x / y along the stud rows (`plate_studs: [nx, ny]`),
`plate_thickness` when the pose is the underside, `grid_bias` as there.

## The matching (LTRON)

LTRON (Walsman et al., ECCV 2022) scores a reconstructed LEGO assembly against
the true one: `match_assemblies` finds the rigid offset that brings the most
bricks into agreement - anchor pairs of the same shape and colour, rarest shape
first, each candidate offset validated by a k-d tree radius search plus shape,
colour and pose-under-symmetry checks, pairs already explained skipped, the
search restarted whenever a better offset turns up - and `score_assemblies`
reports F1 = tp / (tp + (fp + fn) / 2), `edit_distance` the number of rigid
moves (one per further offset group) plus unmatched bricks. Ported as is, with:

- grid poses: centre in pitches, bottom in layers, yaw about the plate normal;
  the radius is an ellipsoid of `lateral_tolerance` (0.3 pitch) and
  `layer_tolerance` (0.45 brick); symmetry is a rectangle's 180 deg and a
  square's 90 deg within `yaw_tolerance_deg` (10);
- one-to-one pairs (LTRON's `validate_matches` let two observed bricks claim
  one target brick) by `scipy.optimize.linear_sum_assignment`, nearest first;
- the identity offset first, and `align: none` (default) stops there: on a
  calibrated plate a shifted build is wrong. The best offset is still searched
  and named in `detail` ("would match with the whole build moved by ...").
  `align: plate` (quarter turns and whole-stud shifts, same layer) or `free`
  makes the offset the verdict (a free-standing build, a model whose origin is
  arbitrary); `build_offset` reports it.

Then the unmatched bricks, in this order: a later step's brick in its place is
`ahead` (not an error); same place and shape, other colour, `wrong_colour`;
same shape and colour elsewhere (shifted, turned, a layer off), `misplaced`
with the shift (also a depth blob that fits inside the brick's footprint and
covers half of it, or the rectangles of one raised region together - a moved
brick partly hidden); the rest `missing` and `extra`. A matched brick with a
height reading is `unseated` when it reads over half a stud high
(`seat_tolerance`, 0.85 mm LEGO / 2.25 mm DUPLO) at 3 standard errors, or its
highest cell does at 4.

## The depth path

Without `observed`, a depth frame (`depth` in metres, or integers in mm, with
`intrinsics` and the camera-to-world `pose`, OpenCV axes) and the plate pose:

1. **Expected height map**: per stud cell the top of the highest target brick
   (0 = the plate), studs on top unless a tile.
2. **Ray cast**: every pixel's ray against that map (column tops and stud
   discs, in descending height levels); a ray that would meet a wall, or land
   within 0.5 mm of a cell edge or a stud rim, is dropped. The residual is the
   measured point's height minus the expected surface's along the same ray, and
   the cell is the expected hit's - not the noisy point's: assigning cells by
   the measured point biased brick heights by ~0.4 mm overhead (noise moves
   points across the edges selectively). Points more than half a brick off the
   expected surface (a brick missing or added) are placed by their own position
   instead, at least a quarter pitch inside a cell (walls).
3. **Level**: a plane (offset and tilt) fitted to the empty plate cells'
   medians is subtracted - a plate pose a millimetre off in height or a few
   tenths of a degree in tilt (`level_plate`, `level_correction_mm`).
4. **Explain**: a target brick with at least a fifth of its seen top cells half
   a brick low is absent; the map is recomputed without it, until stable.
   A present brick's height is the median over its top cells, with its
   standard error from the points' scatter (never under the frame's pooled
   per-point scatter, `depth_sigma_mm`); its highest cell likewise. A brick
   covered by others is `inferred`; one with under half its top seen is
   `unverified`.
5. **Extra**: cells reading over (stud + brick) / 2 above the explained map
   are grouped by connectivity and layer and cut into rectangles: extra
   bricks with footprint and top layer (`stack_layers` when more than one).
6. **Colour** (`check_colour`, the frame has `rgb`): each seen brick's top
   pixels go to `colour.classify` (`palette`, `white_ref`); an `ambiguous`
   answer keeps both names. The bricks then go through the matching above.

A cell needs 3 points and a fifth of the frame's median cell (`min_cell_points`);
the overhead camera gives ~8-10 per LEGO cell, the wrist D405 hundreds.

## Inputs

- `target` (required): list / JSON text / the parser's or planner's output.
- `observed` (bricks) or the depth path: `camera` or `camera_name`,
  `plate_pose`, `plate_studs`, `depth_scale`.
- `upto_step`, `upto_seq`; `system` (`lego` default, `duplo`), `stud_pitch`,
  `brick_height`, `stud_height`, `stud_diameter`, `plate_thickness`,
  `grid_bias`.
- `align` (`none` | `plate` | `free`), `check_colour`, `palette`, `white_ref`,
  `lateral_tolerance`, `layer_tolerance`, `yaw_tolerance_deg`,
  `seat_tolerance`, `level_plate`, `min_cell_points`, `min_visible_fraction`,
  `require_seating` (false: seating within the noise is a note; true: it
  routes `unverifiable`).

## Outputs

`route`, `detail`, `mode` (`bricks` / `depth`), `score` (LTRON F1),
`precision`, `recall`, `edit_distance`, counts (`n_target`, `n_observed`,
`n_matched`, `n_missing`, `n_extra`, `n_misplaced`, `n_wrong_colour`,
`n_unseated`, `n_unverified`, `n_seat_unresolved`), the lists `matched`,
`missing`, `extra`, `misplaced` (with `observed`, `shift_studs`,
`shift_layers`, `yaw_off_deg`), `wrong_colour`, `unseated` (`how`,
`height_residual_mm`, `height_se_mm`, `highest_cell_mm`), `ahead`,
`unverified`, `seat_unresolved`, `repairs` (remove top-down, move, swap, press,
place bottom-up), `build_offset`, `cells` (depth: the cells off by more than
the seat tolerance, expected and measured tops), `level_correction_mm`,
`depth_sigma_mm`, `report_json` (all but `cells`, for graphs that pass text).

## Measured

CPU only (round-5 rule: no GPU host; nothing moves, so no simulation check is
needed).

**Unit tests** (`tests/bundles/test_verifying_build_against_model.py`, 57):
every route on brick lists, LTRON's F1, edit distance of a moved sub-assembly
(1), a whole build shifted (named in `detail`; `align: plate` matches it) and
turned a quarter (found by the offset search), one-to-one pairing, symmetric
and ICL / parser / planner / pose-skill / metric-pose input forms, ahead,
`upto_step` / `upto_seq`; and on rendered depth: every error type from the
wrist for LEGO and DUPLO, colour through tools/colour, overhead LEGO, camera
from the observation, no camera / plate out of view, a plate pose 1.5 mm low
levelled (and not, with `level_plate: false`), uint16 millimetres, a failing
colour tool.

**Synthetic depth** (`tests/bundles/_build_verify_eval.py`): random 8-brick builds (1x1-2x6, layers 1-3) on a 12 x 12 plate, one error
injected per trial, ray-cast depth with the station noise model pose6d uses
(overhead Orbbec at ~0.9 m, oblique: 2.9 mm at 1 m growing with depth^2, 1 mm
steps; wrist D405 at 0.16-0.24 m, 5-15 deg off vertical: 0.5 mm at 0.15 m,
0.1 mm steps), RGB with LDraw colours for the colour trials. Detected = the
route names the error and the report names the right brick; "clean" = the
build is right and the route is `matches`. Seeds fixed; 40 trials per cell with
the exact plate pose:

| error | LEGO overhead | LEGO wrist | DUPLO overhead | DUPLO wrist |
|---|---|---|---|---|
| clean (right = matches) | 40/40 | 40/40 | 40/40 | 39/40 (1 unverifiable) |
| missing | 39/40 | 40/40 | 40/40 | 40/40 |
| extra | 37/40 | 40/40 | 39/40 | 40/40 |
| misplaced | 31/40 (8 read as extra) | 40/40 | 39/40 | 40/40 |
| unseated (on the stud tops) | 19/40 (21 seat_unresolved) | 40/40 | 40/40 | 39/40 (1 unverifiable) |
| one end on the studs | 12/40 (19 of 28 misses seat_unresolved) | 40/40 | 39/40 | 40/40 |
| wrong colour | 30/40 | 40/40 | 40/40 | 40/40 |

No clean build was flagged with an error on any camera (the one DUPLO wrist
`unverifiable` had a brick seen too little). On overhead LEGO 37 of 40 clean
builds carried a `seat_unresolved` note: the camera cannot tell seated from
resting on the studs for most bricks, and says so.

With the plate pose given 1 mm off (random direction) and tilted 0.3 deg, 20
trials per cell: LEGO wrist 139/140 (one tilt missed), DUPLO overhead and
wrist 140/140 each, LEGO overhead 91/140 (clean 20/20; missing 15/20 - 4
routed `unseated` first, a neighbour's edge cell reading high; extra 15/20;
misplaced 14/20; unseated 9/20; one end 3/20; colour 15/20). The level plane
took up the height and tilt; the lateral millimetre is what costs overhead
LEGO (a quarter of its 2 mm pixels' cell).

Before the ray cast assigned cells (cells from the noisy points), overhead LEGO
bricks read 0.4 mm low on average and clean frames had an extra
brick (1 to 4 of 12, depending on the margins); points far off the expected surface placed by the expected hit made a
raised brick's neighbours read as extras; the tilt was first a plane fit,
which stud rims shifted by the tilt (~1 mm) defeated - all three were
replaced (see the depth path) and the table is from the final code.

## Limits

- **Overhead LEGO seating is below the noise.** At ~0.9 m the Orbbec's
  ~2.4 mm per-point noise and 1 mm steps give ~0.5 mm per brick and ~1 mm per
  cell against a 1.7 mm stud: about half the unseated LEGO bricks are caught
  and the rest are `seat_unresolved` (named, not passed silently; with
  `require_seating` the route is `unverifiable`). Verify LEGO seating from the
  wrist. DUPLO's 4.5 mm studs are fine from overhead.
- Overhead LEGO also misses some small extras and misplacements (a 1x1 is
  ~4 x 4 px) and ~1 in 4 wrong colours (tools/colour on ~10-20 px).
- Extras thinner than half a brick (a plate, a tile) are not cut out; a plate
  under a brick reads as that brick raised (`unseated`).
- Two adjacent extra bricks of one height and colour are one blob, cut into
  rectangles largest first; their split may differ from the real bricks.
- The depth path trusts the plate pose to about a millimetre laterally (the
  level plane takes up height and tilt); a stud's 8 mm pitch error is a
  different build. Synthetic frames only: no LEGO / DUPLO build was on disk.
- One camera view per call: what is hidden from it is `inferred` (covered by a
  seen brick) or `unverified`. Merge views by running the skill per view.

## The constants, and why

- **`LATERAL_TOLERANCE` 0.3 pitch, `LAYER_TOLERANCE` 0.45 brick,
  `YAW_TOLERANCE_DEG` 10**: the pose skill calls a pose off its cell past a
  quarter pitch or 10 deg; under half a brick keeps layers apart while an
  unseated brick (0.18 / 0.23 brick) plus overhead noise still matches. LTRON's
  1 LDU / 0.08 rad are for exact simulator poses.
- **`SEAT_TOLERANCE_FRACTION` 0.5, `SEAT_SIGMAS` 3, `CELL_SIGMAS` 4**: half a
  stud separates seated from resting on the stud tops; 4 for the highest of a
  few dozen cells per frame keeps clean frames clean (3 flagged ~5 %).
- **`EDGE_INSET`, `STUD_MARGIN` 0.5 mm; `MEASURED_INSET` 0.25 pitch**: mixed
  pixels and walls; a 1 mm plate error put a raised brick's wall into the next
  cell from the wrist until points placed by their own position kept a quarter
  pitch from the edges.
- **`MIN_CELL_POINTS` 3, `MIN_CELL_SHARE` 0.2, `MIN_VISIBLE_FRACTION` 0.5,
  `LOW_CELL_FRACTION` 0.2, `LEVEL_MIN_CELLS` 6**: a grazing-angle cell past a
  tall neighbour has a few points that read its wall; one low cell of five is a
  moved brick, not noise (the per-cell noise is ~1 mm overhead against a half
  brick of 4.8 mm).
- **`PRESETS`**: the pose and press skills' LEGO / DUPLO dimensions.

## Provenance

The matching is adapted from LTRON (Aaron Walsman, MIT, commit 253be3e):
`ltron/matching.py` (`match_assemblies`, `validate_matches`, `match_lookup`),
`ltron/score.py` (`f1`, `score_assemblies`, `edit_distance`) and
`ltron/geometry/symmetry.py` (`pose_match_under_symmetries`); the licence is
`LICENSE.ltron` beside this file (it also lists the CC BY 2.0 LDraw parts LTRON
uses; none are bundled here). Changed: grid bricks instead of LDraw
assemblies, an ellipsoidal radius, analytic yaw symmetry instead of the rendered
symmetry table, the identity offset first and by default only, offsets about
the plate normal, one-to-one pairing, colour wildcards. Added: the split into
ahead / wrong colour / misplaced / missing / extra, unseated, repairs, and the
whole depth path (LTRON compares assemblies, not sensor data). The script's
header lists the changes. LEGO and DUPLO are trademarks of the LEGO Group,
which does not sponsor or endorse this.
