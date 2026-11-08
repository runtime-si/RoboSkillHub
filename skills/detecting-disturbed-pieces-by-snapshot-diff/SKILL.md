---
name: detecting-disturbed-pieces-by-snapshot-diff
description: Checks whether anything already placed was knocked since the last
  verified step - registers a new RGB(-D) snapshot onto the reference taken
  after that step (ORB + RANSAC homography refined by masked ECC on the static
  background; the camera may have moved a little), divides out lighting drift,
  maps colour/SSIM and depth change, and re-finds every known item (jigsaw
  pieces, placed bricks, standing dominoes - given as masks, pixel polygons,
  world boxes or grid cells) by a local rigid registration, reporting which
  moved, by how many mm and degrees, and how much their depth changed; changes
  no item explains become occluders (a hand or the arm, masked from its pose)
  or unexpected objects. Routes unchanged, disturbed, unexpected_object,
  occluded or unregistrable. Use when a place, press or push has just
  happened near items already placed and the next step will build on them;
  camera-only, no motion.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [verification, change-detection, ssim, ecc, registration, jigsaw, lego, domino, rgb-d]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw, pick-place, sorting]
    provenance:
      authors: [runtime-si]
      code: https://github.com/scikit-image/scikit-image/blob/e098ebbb684788df60b42e3bde112c014d0d9517/src/skimage/metrics/_structural_similarity.py
      upstream_license: BSD-3-Clause
      paper: "Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, Image Quality Assessment: From Error Visibility to Structural Similarity, IEEE TIP 2004; G. D. Evangelidis, E. Z. Psarakis, Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization, IEEE TPAMI 2008"
gap:
  allowed_tools:
    - robot.get_observation
  required_inputs:
    reference: Observation
  # Optional inputs (gap has no list or open-dict type here, so they are wired from a script
  # node in the same graph): items (list of {id, mask | polygon_px | box | cell, expected?}),
  # current (Observation or CameraFrame; default a fresh robot.get_observation), camera_name,
  # grid ({cols, rows, corners_px} or {origin, cols, rows, cell_size, item_height}), region
  # (mask, {polygon_px} or world OrientedBoundingBox), ignore_mask, reference_ignore_mask,
  # expected_ids, mask_arms, arm_base_positions, plane_z, mm_per_px and the thresholds.
  produces_outputs:
    n_disturbed: int
    n_unexpected: int
    max_shift_mm: float
    camera_shift_px: float
    changed_fraction: float
  exit_conditions:
    disturbed: At least one item not listed as expected moved (translated at least max(1 mm, 0.6 px at its scale), turned at least 2 degrees and 0.8 px at its rim, or changed depth by 5 mm, each standing out of the noise for its texture) or is lost (not found as it looked within 25 mm); `disturbed` lists them and `items` says by how much.
    unexpected_object: No item moved, but a change inside the region that no item explains is at least 80 mm^2 (30 px without a scale) and does not enter from the frame edge or the arm; `unexpected` locates it (kind appeared / vacated with depth).
    occluded: Nothing moved and nothing unexpected, but at least one item is less than 60 % visible (under the arm, a hand, an ignore mask, or outside the registered view); it could not be checked.
    unchanged: Every judged item is where it was and nothing unexplained changed in the region.
    unregistrable: The background does not correlate (median tile correlation under 0.8 after registration) or the camera moved more than 8 % of the frame diagonal; the views cannot be compared pixel by pixel - re-take the reference.
  canonical_scripts:
    - diff_snapshots: scripts/diff_snapshots.py
  examples:
    - title: Check the placed items after a place, against the snapshot of the last verified step
      path: examples/canonical_subgraph.json
---

# detecting-disturbed-pieces-by-snapshot-diff

A place or a press touches only one item, but its fingers, the arm or a
dragged cable can nudge the ones already down: a jigsaw section slides, a brick
next to the pressed one lifts, a domino in the line tips. A before/after
comparison of the same camera catches this right after the step that caused it,
while the fix is still a small push. Both frames share the camera and (nearly)
the lighting, so a pixel diff is far more sensitive than re-registering the
whole assembly against its plan.

## How

1. **Frames.** `reference` (the snapshot after the last verified step) and
   `current` (default: a fresh `robot.get_observation`) are Observations
   (camera `camera_name`), CameraFrames or bare RGB arrays. Frames wider than
   960 px are worked on shrunk; outputs are in input pixels.
2. **Register** the current frame onto the reference on the static background
   (frame minus the items, the ignore masks and the arm): ORB + RANSAC
   homography, the homography the two camera poses imply for the support plane
   (a wrist camera), or identity - whichever correlates best - refined by
   masked homography ECC (OpenCV `findTransformECCWithMask`) coarse to fine on
   locally normalised grey (lighting gradients drop out). A second ECC pass
   uses only the background within 60 px of the work region and none of the
   pixels that changed. The result is accepted when the median correlation over
   6 x 8 tiles is at least 0.8 (a hand over a third of the view does not sink
   it, another scene does) and the camera moved at most 8 % of the diagonal.
3. **Change map.** Depth (when both frames have it): the warped difference
   less a drift plane fitted robustly on the background (started from the
   median), changed where over max(6 mm, 1.2 % of range). Appearance: the
   current colour is divided by a per-channel gain - a robust quadratic in
   the pixel position (lamp gain, tint, gradient) times a smooth local residual
   clipped to +-20 % (soft shadows), fitted where depth did not change - then
   scikit-image's SSIM map (Gaussian 11-tap window, sigma 1.5) and the colour
   difference mark a pixel changed when 1 - SSIM > 0.35 and colour moved by 14,
   or colour moved by 40 alone.
4. **Occluders.** A changed blob touching the frame edge or the arm's mask
   (and, with depth, at least 25 mm nearer than the reference) is an occluder:
   its pixels are hidden, not judged.
5. **Items.** Each item's mask (dilated 3 px for its outline, other items cut
   out) is searched for within 25 mm in the registered current frame:
   masked NCC over translations (and +-8 / 16 degree turns when the plain
   template scores under 0.9), the three best peaks refined by Euclidean ECC,
   the best refined match winning after a small penalty for distance and for
   landing on another item's footprint (two identical dominoes 20 mm apart).
   The warp's translation and turn are mapped onto the plane through the item's
   own height (from depth, its world box, or `plane_z` with the camera pose):
   `shift_mm`, `translation_world`, `rotation_deg` (counter-clockwise seen
   from the camera, i.e. about world +z for a camera looking down), plus the
   median `depth_change_mm` over the item. Hidden pixels read as unchanged so
   they cannot pull the search.
6. **Verdict per item.** `occluded` if under 60 % visible. `moved` if it
   translated at least max(`min_shift_mm` 1 mm, 0.6 px at its scale), or
   turned at least 2 degrees and 0.8 px at its rim, or its depth changed 5 mm -
   but a translation or turn counts only if the template has texture in every
   direction (mean squared gradient along the weakest at least 2) and the move
   stands out of the noise (Gauss-Newton information p'J'Jp / 2 sigma^2 >= 100,
   sigma the frames' grey noise). `lost` if its best match correlates under 0.55
   and over 25 % of it changed (or its depth changed): toppled, flipped,
   removed or pushed past the search radius.
7. **Blobs.** Changed pixels inside `region` that neither an item's old nor its
   new footprint explains, at least 80 mm^2 (or 30 px without a scale), are
   `unexpected`: area, centroid, world position, kind (`appeared` - nearer,
   `vacated` - farther, `appearance` - no depth), and the grid cell for a
   pixel grid.

## Inputs

- `reference`: the snapshot after the last verified step (Observation,
  CameraFrame or RGB array). Depth, intrinsics and pose are optional but give
  millimetres and world vectors.
- `items`: `[{"id", "mask" | "polygon_px" | "box" | "cell", "expected"?}]` in the
  reference frame. `mask` (the reference's size, e.g. from sam3 or the planner's
  placed footprint), `polygon_px`, `box` (a world OrientedBoundingBox: needs the
  reference camera's intrinsics and pose), or `cell` `[col, row]` of `grid`.
- `grid`: `{"cols", "rows", "corners_px"}` (the board's four corners in the
  reference image, top-left first, clockwise) or `{"origin": Se3Pose of the
  corner of cell (0, 0), x along columns, y along rows, z up; "cols", "rows",
  "cell_size": m or [w, h], "item_height": m}`.
- `region`: where an unexpected object counts - a mask, `{"polygon_px"}` or a
  world box. Default: the grid, else the items' hull widened by 24 px, else the
  whole frame. Pass the work area: a person moving in the background is
  outside it.
- `ignore_mask` (current frame) / `reference_ignore_mask`: left out entirely
  (the slot just filled, the tool, a cable you expect to move).
  `expected_ids` (or `"expected": true` on an item): measured but not counted
  - the item the step itself placed.
- `mask_arms` (true) masks each arm in an Observation from its `ee_pose`: balls
  of 5 cm radius from 3 cm past the tool centre back 15 cm along the approach
  axis (local z), and on to `arm_base_positions[i]` (world) when given.
- `plane_z` (world height of the support) for RGB-only frames with a pose;
  `mm_per_px` for frames with neither depth nor pose (shifts are in px only
  otherwise).
- Thresholds: `search_radius_mm` (25), `min_shift_mm` (1.0), `min_shift_px`
  (1, no scale), `min_rotation_deg` (2), `min_depth_change_m` (0.005),
  `min_blob_mm2` (80), `min_blob_px` (30), `max_camera_shift_fraction`
  (0.08), `min_rho` (0.8).

## Outputs

`route`; `items` (per item: `status` unchanged / moved / lost / occluded,
`expected`, `shift_mm`, `translation_px`, `translation_world`, `rotation_deg`,
`depth_change_mm`, `changed_fraction`, `visible_fraction`, `match`,
`significance`, `texture`, `centroid_px`); `disturbed` and `occluded` (ids);
`unexpected` and `occluders` (blobs); `n_disturbed`, `n_unexpected`,
`max_shift_mm`; `camera_shift_px` and `registration` (method, rho, ORB
inliers, homography reference px to current px); `changed_fraction` of the
region; `detail`. Hand `disturbed` to a re-localisation or nudge-back step
(the translation and turn are the correction to undo).

## Measured

All CPU, no GPU and no simulator (camera-only skill). 0.2-0.5 s per call at
640 x 480 with 8-30 items on one core-set of a 20-core machine.

**Synthetic boards** (`tests/bundles/_snapshot_diff_eval.py --trials 20
--first-seed 1000`; thresholds were tuned on seeds 1-6, these are fresh
seeds). Ray-cast RGB-D renders, 3 x 3 supersampled, of jigsaw pieces (25 mm,
a partial assembly of 15), 2x4 / 2x2 bricks on a baseplate (8) or standing
dominoes in a line (8, 20 mm apart), seen by the overhead camera (station K,
1 m, 31 degrees tilt: 0.46 px per mm) or a wrist camera (25 cm straight down:
1.6 px per mm). Every current frame gets its own lighting drift (gain
0.85-1.15, tint +-4 %, gradient +-15 %, a soft shadow half the time) and an
unknown camera bump (0-0.3 degrees, 0-2 mm); the wrist camera reports a pose
off by up to 1 mm / 0.15 degrees. 20 boards per family and camera; one random
item disturbed per pair; items given as reference masks.

| family / camera | no-change false alarms | shift 1 / 2 / 3 / 5 / 7 / 10 mm detected | median shift error | turn 2 / 5 / 10 / 15 deg detected |
|---|---|---|---|---|
| jigsaw / overhead | 0/20 | 2 / 20 / 20 / 20 / 20 / 20 of 20 | 0.10-0.28 mm | 0 / 1 / 16 / 20 |
| jigsaw / wrist | 0/20 | 10 / 20 / 20 / 20 / 20 / 20 | 0.05-0.08 mm | 5 / 20 / 20 / 20 |
| lego / overhead | 0/20 | 2 / 20 / 20 / 20 / 20 / 20 | 0.21-0.46 mm | 0 / 0 / 15 / 19 |
| lego / wrist | 0/20 | 12 / 20 / 20 / 20 / 20 / 20 | 0.05-0.11 mm | 1 / 20 / 20 / 20 |
| domino / overhead | 0/20 | 1 / 20 / 20 / 20 / 20 / 20 | 0.14-0.37 mm | 0 / 0 / 0 / 20 (angle off by 9 deg) |
| domino / wrist | 0/20 | 7 / 20 / 20 / 20 / 20 / 20 | 0.07-0.35 mm | 0 / 20 / 20 / 20 |

- Unmoved items read a median 0.09-0.14 px (0.2-0.34 mm) overhead and
  0.05-0.23 px at the wrist; 99th percentile 0.32-0.57 px / 0.19-0.83 px.
- Other items wrongly flagged in the 200 change pairs per family and camera:
  jigsaw overhead 36+4, jigsaw wrist 26, domino wrist 12, domino overhead 9,
  lego 0-1 - almost all the moved item's direct neighbours (a slid jigsaw
  piece overlaps its neighbour's footprint; a domino pushed against the next).
- Removed items: 120/120 `disturbed` (target `lost` or `moved`); toppled
  dominoes: 40/40.
- An object (15 x 15 x 10 mm) set down at least 12 mm clear of the items:
  `unexpected_object` 119/120 (one domino-overhead case hidden behind the tall
  dominoes).
- A hand (forearm box 8-15 cm above the board, from outside the view): never
  `unexpected_object`; overhead 59 of 60 `occluded` or `unchanged`, 1
  `disturbed`; wrist 20 of 60 `unregistrable` (the hand fills most of a 25 cm
  view), the rest `occluded` / `unchanged`.
- The robot's arm hovering 9 cm over the board, masked from its `ee_pose` and
  base: 60/60 `occluded`, no item flagged.

**Station frames** (overhead Orbbec, RGB only, 5 sessions x 11 frames of
`g_sota_frames_*`, the scene static; not committed). Items: a 13 x 7 world grid
of 50 mm cells on the table (`grid.origin`, `plane_z`), region the work area.

- With no items (`items` empty, region only), 20 consecutive pairs: all
  `unchanged`, no changed pixel left in the region.
- 55 no-change pairs (10 consecutive + first-to-last per session): 54
  `unchanged`, 1 `disturbed` (a cell of white tablecloth read 2.7 mm / 8
  degrees). Background correlation 0.994-0.999; camera shift 0.02-0.3 px.
  Before the texture and significance gates 14 of 55 were false alarms.
- Semi-synthetic: six 25 mm textured pieces composited on the table plane of
  25 real frame pairs (real sensor noise, real background), one moved per pair:
  no-change 0/25 false alarms; shifts 1 / 2 / 3 / 5 / 7 / 10 mm detected 0 /
  24 / 25 / 25 / 25 / 25 of 25 (median error 0.07-0.15 mm); turns 5 / 10 / 15
  degrees 0 / 17 / 22 of 25; no other piece flagged.
- Real change pairs: a sheet laid on the table and objects moved between two
  sessions (RGB) -> `disturbed`, the sheet's cells and 15 occluder blobs; the
  threading setup at 11:44 against 14:21 with depth (cable re-routed, posts
  moved, very different window light) -> `disturbed` (24 cells along the
  cable and the light streaks) plus 2 `vacated` blobs; the 11:44 frame against
  itself -> `unchanged`.

## Limits

- **The overhead camera is coarse.** At 1 m the Orbbec gives 0.46 px per mm:
  a 1 mm bump is under half a pixel and is not reported (by design: 0.6 px is
  the noise floor), a 25 mm piece is 11 px and must turn about 8 degrees to
  count. Use the wrist camera (2 px per mm at 25 cm) to check fine seating.
- **Standing dominoes seen obliquely from 1 m** are 4 px wide: a push is
  caught, but a turn is not, and a domino pushed against its neighbour can be
  matched to the neighbour's place (shift read 20 mm instead of 10) and the
  neighbour flagged too.
- **Shadows and big uniform changes.** A soft shadow (up to +-20 % local gain)
  is absorbed; a deeper one, or the arm's shadow, becomes changed pixels -
  take both snapshots from the same arm park pose. Without depth a flat hand or
  sheet over a flat table is only seen by its colour.
- **Textureless items** (white cloth, plain wood across the grain, a white
  domino face without its outline) cannot have their move measured; only the
  change map, depth and match judge them.
- **Parallax.** The registration is a plane homography: a camera that moved
  centimetres (not millimetres) sees tall items shift against the table. Keep
  the camera fixed or re-take the reference.
- **A wrist view mostly covered by a hand** is `unregistrable`, not `occluded`.
- **No simulator check** (camera-only skill); validated on synthetic renders
  and recorded station frames (see Measured). A live check on the robot is
  pending.

## The constants, and why

Each constant in `scripts/diff_snapshots.py` carries its source and
measurement. The decisive ones: `MIN_RHO` 0.8 (station pairs of one scene
0.993-0.999, synthetic 0.90-0.995, hand-covered wrist views 0.05-0.78);
`HIGHPASS_SIGMA` 8 / `HIGHPASS_EPS` 10 (lighting drift moved the registration
5 px without it); `GAIN_LOCAL` 0.2; `T_DSSIM` 0.35 / `T_COLOUR_MIN` 14 /
`T_COLOUR` 40; `DEPTH_CHANGE_M` 6 mm + 1.2 %; `MIN_SHIFT_MM` 1.0 and
`MIN_SHIFT_NOISE_PX` 0.6 (unmoved items read a median 0.1 px, 99th percentile
0.35-0.47 px overhead); `MIN_ROTATION_ARC_PX` 0.8 (11 px pieces read +-5
degrees unchanged); `MIN_TEXTURE` 2 and `MIN_SIGNIFICANCE` 100 (station table
cells read up to 28 mm on a white tablecloth and 2.4 mm across wood grain
without them); `DISTANCE_PENALTY` 0.08 and `OVERLAP_PENALTY` 0.15 (identical
neighbours); `OCCLUDER_NEARER_M` 25 mm.

## Method

The candidate (jigsaw survey, "3 disturbances") was a masked SSIM diff after
ECC alignment, mapped to grid cells. That is kept - the SSIM map is
scikit-image's `structural_similarity(gaussian_weights=True, full=True)`
adapted to OpenCV filters (BSD-3-Clause, header and `LICENSE.scikit-image`),
and alignment is OpenCV's ECC (called, not copied) - and generalised: depth
change, lighting normalisation, occluders and the arm mask, and a per-item
local registration that says how far each item moved instead of only which
cells changed.
