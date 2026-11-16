---
name: finding-boxes-of-known-size-from-overhead-depth
description: Finds an open box, bin or tray whose outer size and rim height are
  already known in one overhead depth frame - takes the depth points in a band
  round the rim height, and slides the box's own outline (every 2 degrees of
  turn) over them in the plan view to where most of the outline is covered, so
  a rim seen in pieces, a box touching another, or a box beside something as
  tall (an arm resting on the table) is still placed where a blob or
  minimum-area-rectangle fit merges or mistakes it. Returns the box as an
  upright OrientedBoundingBox of the stated size, the share of the outline
  seen, and a flag when a nearly square box's turn may be a quarter out. Use
  when a container of measured size must be located from a fixed overhead
  RGB-D camera before picking from it or placing into it, with its size and
  rim height in hand and the hand out of the camera's view; call it once per
  box, passing the boxes already found as exclude_obbs. CPU only.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, container, box, bin, depth, overhead, known-size, cpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tuned_on: [yam_i2rt]
    tasks: [pick-place, sorting]
    provenance:
      authors: [RoboSkillHub contributors]
    notes: Position accuracy was never measured against a ground truth; on the one
      station it ran on, drops aimed from it alone missed the box (see
      locating-a-container-from-the-wrist-camera).
gap:
  allowed_tools:
    - robot.get_observation
  required_inputs:
    size_long: float
    size_short: float
    rim_z: float
  # Optional inputs: camera (string), support_z, band_below, band_above, min_coverage (float),
  # stride (int). Wired from a script node (gap has no type for them): camera_frame
  # (CameraFrame), exclude_obbs (list of OrientedBoundingBox), roi ({x_min, x_max, y_min, y_max}).
  produces_outputs:
    container_obb: OrientedBoundingBox
    center: Vec3
    yaw_deg: float
    coverage: float
    rim_points: int
    yaw_ambiguous: bool
  exit_conditions:
    found: A rectangle of the stated size lies along the rim points with at least min_coverage (0.35) of its outline on them; container_obb is that rectangle, upright, from support_z to rim_z, local x its long side. What the overhead camera sees, in its calibration's frame - not where a hand will find it.
    not_found: No posed depth camera, fewer than 150 depth points in the rim's height band (inside roi, outside exclude_obbs), or the best placement covers less than min_coverage of the outline; `detail` says which, `coverage` and `rim_points` what was seen.
  canonical_scripts:
    - find_known_box: scripts/find_known_box.py
  examples:
    - title: Find a box of known size in the overhead view
      path: examples/canonical_subgraph.json
---

# finding-boxes-of-known-size-from-overhead-depth

Where is the box, when its size is already known? A fit that measures the box
from what it sees (connected rim blobs, a minimum-area rectangle) goes wrong
exactly when a station is busy: two boxes pushed together read as one, an arm
lying beside the box is as tall as its rim, and a wall end-on to the camera
leaves a gap in the rim. This skill does not measure the box. It places it.

## How

1. **Rim band.** The camera's depth is lifted to world points with the
   camera's pose; those from `band_below` (3 cm) under to `band_above` (2 cm)
   over `rim_z` are kept, inside `roi` and outside every `exclude_obbs`
   footprint grown by 2 cm.
2. **Outline slide.** The points are binned on a 4 mm plan-view grid, grown
   one cell. For each turn of the box (2 degree steps over 180), the
   rectangle's outline, two cells thick, is correlated with the grid; the
   value at a cell is the share of the outline covered with the box centred
   there. The best cell and turn win.
3. **Answer.** `coverage` at or over `min_coverage` is `found`. The box
   returned has the stated size (it is not re-measured), the fitted centre and
   turn, and stands from `support_z` to `rim_z`.

## Inputs and frames

Everything is in the world frame of the camera's `pose`. `size_long` and
`size_short` are the **outer** rim. `rim_z` and `support_z` are world heights;
on a station whose world origin is not the table top, pass `support_z`.

## Gotchas

- **Nearly square boxes.** When the sides differ by under 20 %, the outline
  fits almost as well a quarter turn round; `yaw_ambiguous` is then true. On
  the station this came from, a 238 x 196 mm box read -38 and -54 degrees in
  two frames of one afternoon, and nobody established which was right.
- **Something of the same outline.** The score is coverage only. Two boxes of
  the same size are not told apart; use `roi` or `exclude_obbs`.
- **The hand over the box** hides the rim and adds points in the band. Move it
  out of the camera's view first.
- **Overhead-to-arm calibration.** `found` is where the overhead camera puts
  the box. On the source station the wrist camera of the arm that had to reach
  it put the same box 2 to 7 cm away (arm pose error, not this fit, as far as
  could be told); four parts in a row were let go beside the box when the
  overhead position was used alone. Follow with
  `locating-a-container-from-the-wrist-camera` before anything is dropped.
- A caller should still sanity-check the result against the last good one. The
  source task kept a fallback to the last fitted box for frames where the fit
  failed.

## Constants, and what they were measured on

All from one YAM station (two i2rt YAM arms, one overhead depth camera over a
kitchen counter, the median of 3 frames), two cardboard boxes, 449 x 262 mm
with a 154 mm rim and 238 x 196 mm with a 66 mm rim, in that task's real
runs of 2026-11-15, in two layouts (the boxes as first set out, and swapped).

- `CELL = 4 mm`, `YAW_STEP_DEG = 2`: chosen, not swept.
- `BAND_BELOW / BAND_ABOVE = 3 cm / 2 cm`: used for the tall box; the low box
  used 2 cm / 3 cm.
- `MIN_COVERAGE = 0.35`, `MIN_POINTS = 150`: chosen, not swept.
- `NEAR_SQUARE = 0.20`: from the one box above.

## Evidence

What prompted it, on that station: a blob fit read two boxes pushed together
as one 48 x 45 cm box and a 12 x 7 cm one (2026-11-14), and took an arm
resting on the counter for the tall box (2026-11-15). The size-given fit
replaced it for that task's remaining real runs. That is the whole of the
evidence: a fallback to the last good fit stayed in place and how often it
fired was not counted, and no ground-truth position was measured. Nothing
here was run in a simulator suite. Maturity is `community`.

## Required end states

- `found`: `container_obb`, `center` (on the rim plane), `yaw_deg`, `coverage`.
- `not_found`: nothing placed; `detail` gives the reason.
