---
name: classifying-brick-resting-face-from-depth
description: Tells from depth whether a loose LEGO or DUPLO brick of known
  size lies studs up, upside down (resting on its stud tops, hollow underside
  up) or on a side wall, and which of its axes is vertical, with its yaw, its
  stud direction and its body centre. The brick is the connected cluster of
  depth points over the table nearest a rough position. Every resting face of
  the brick is scored on its height over the table and its footprint (the
  height-versus-dimensions match of a UR5 LEGO pick-and-place pipeline). Up
  is told from down by the hollow underside (most of the inner footprint far
  under the rim) instead of the source's YOLO model. Moves the wrist camera
  over the brick first, or reads the current frame of any posed depth camera.
  Use before picking a loose brick, to grasp it directly when it is studs up
  or to route it to a flip / regrasp otherwise, and to pick the grasp axis
  and yaw.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [lego, duplo, bricks, orientation, depth, wrist-camera, d405, opencv, cpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego, reorientation]
    provenance:
      authors: [runtime-si]
      code: https://github.com/pietrolechthaler/UR5-Pick-and-Place-Simulation/blob/596c496867c11ba63f47127833e0f49b7480ffd3/catkin_ws/src/vision/scripts/lego-vision.py
      upstream_license: MIT
gap:
  allowed_tools:
    - robot.get_observation
    - robot.go_to_pose
    - robot.solve_ik
  required_inputs:
    brick_position: Vec3
  produces_outputs:
    vertical_axis: str
    up_vector: Vec3
    stud_direction_known: bool
    yaw_deg: float
    centre: Vec3
    height: float
    footprint_long: float
    footprint_short: float
    table_height: float
    deep_share: float
    samples: int
  exit_conditions:
    studs_up: The brick's z axis is vertical (its height and footprint fit that face best by a margin) and at most 10 % of its inner footprint lies far under its top - a face with studs on it; up_vector (0, 0, 1).
    upside_down: The z axis is vertical and at least 25 % of the inner footprint lies far under the rim - the hollow underside, the brick resting on its stud tops; up_vector (0, 0, -1).
    on_side: A side wall is down - the brick's x or y axis is vertical (vertical_axis); up_vector is the horizontal stud direction when the silhouette resolves which side carries the studs (stud_direction_known), else its axis only.
    unknown: No resting face of the given brick fits the reading (wrong size, two bricks touching, a part that is not that brick), two faces fit about equally, the up / down share is between 10 % and 25 % or the part is too thin to tell (a plate lying flat), or the depth could not be read (no posed camera, too few points around the brick); detail says which.
    no_brick: Fewer than 40 depth points stand over the table near brick_position.
    unreachable: view 'move' and every view of the brick was out of reach or refused.
  canonical_scripts:
    - classify_face: scripts/classify_face.py
  examples:
    - title: Classify a loose DUPLO 2x4 before picking it
      path: examples/canonical_subgraph.json
---

# classifying-brick-resting-face-from-depth

A brick tipped out of a bin lies one of four ways: studs up, upside down,
or on a long or short side wall. Only studs up can be gripped and pressed
straight away. This skill reads which way it lies from depth, using the
brick's known size.

## How it composes

- **Before it**: a rough brick position from detection (detecting-bricks
  by colour, sam3, owlv2, or a sort-landing ROI). Only its xy is used; its
  z only helps aim the view.
- **After it**: `studs_up` goes to a top-down grasp across the short side
  (`yaw_deg` gives the brick's x axis). `on_side` and `upside_down` go to a
  reorientation (flipping-part-upright-by-regrasp,
  reorienting-a-grasped-part-by-pivoting-on-the-table) or a side grasp, with
  `up_vector` telling which way the studs point. `unknown` re-perceives
  (a closer view, singulating-by-pushing if two bricks touch).

## How it reads

1. **View** (`view: move`): aim the wrist camera straight down at the
   brick's highest possible top from 0.20 m (then 0.25 m, then with the
   camera's current rotation), with an IK preview and a 5 cm clearance (the
   views of pressing-a-brick-onto-studs). `view: current` reads the frame as
   it is, from any named camera with a pose.
2. **Points**: the median of `frames` (3) depth frames, back-projected to
   world points.
3. **Table**: the lowest well-supported level of the points in a 3 cm ring
   around the search disc (or `table_height` if given). The source took the
   frame's farthest depth instead.
4. **Brick**: points more than 0.35 of the brick's lowest possible height
   over the table, within its half-diagonal + 2 cm of `brick_position`. They
   are rasterised at 1 mm, closed, split into connected clusters, and the
   cluster nearest the position is kept. An upside-down brick splits into
   its walls and its tubes; the cluster whose hull encloses the nearest one
   is taken, with everything inside it. Lone cells (flying pixels) are
   dropped.
5. **Faces**: the brick's height over the table (98th percentile) and its
   silhouette (`cv2.minAreaRect`, the source's call) are compared with every
   resting face. z vertical: height = brick + stud height, silhouette = the
   footprint. x (or y) vertical: height = that side's length, silhouette =
   the other side by brick + stud height (the studs stick out sideways).
   Score = the larger of |height error| / 0.3 pitch and the worst side error
   / 0.4 pitch; the best must be under 1 and beat the next face by 1.5x. The
   source's getDepthAxis matched the height to whole stud units the same
   way.
6. **Up or down** (z vertical): the share of all depth points in the inner
   footprint (inset 0.2 pitch) that lie more than max(0.45 brick heights,
   1.5 stud heights) under the top. A studs-up brick shows its face one stud
   height under its stud tops (share near 0). An upside-down one shows its
   hollow underside (share 0.7-0.85 in the checks below; the tubes are the
   rest).
7. **Pose**: yaw of the brick's x axis (mod 180 deg, mod 90 for a square
   brick) for z vertical, or of the horizontal stud axis on a side. On a
   side, the stud direction is the side where the silhouette reaches past
   the top wall's extent (the source compared the silhouette's and the top
   surface's centres). `centre` is the body centre for that face.

## Inputs

- `brick_position` (required): rough world position of the brick (xy within
  about a centimetre; z the table's height if known).
- `brick_studs` ([2, 2]: studs along the brick's x and y), `system`
  (`duplo` / `lego`), `stud_pitch`, `stud_height`, `brick_height` (a LEGO
  plate: 0.0032; see Limits).
- `table_height` (measured when omitted), `view` (`move` | `current`),
  `camera`, `frames` (3), `check_reach` (true), `arm_id` (passed only when
  nonzero).

## Outputs

`route`, `detail`, `vertical_axis` (`z`, `x`, `y`), `up_vector` (world,
the brick's +z, i.e. its stud direction), `stud_direction_known`, `yaw_deg`,
`centre` (world body centre), `height` (top over the table, m),
`footprint_long` / `footprint_short` (silhouette sides, m), `table_height`,
`deep_share`, `samples`; and `scores` (per face, lower is better, under 1
fits), a dict returned but not in `produces_outputs` (gap schema types only).

## Checked

**Simulation check: pending (no GPU host).** None of the AutoDL hosts
(autodl5, autodl6, autodl14) answered ssh during this port. The skill only
moves the camera to a view; what still needs a twin run is a rendered
station D405 frame of real brick meshes.

**Synthetic depth** (`tests/bundles/_face_eval.py`, not a simulator):
hollow bricks are ray-cast (walls, top, underside tubes or pins, studs; nominal
dimensions, not CAD) on a table, seen by a 640 x 480 D405 (fx 394 px) from
0.2 m over the brick's highest possible top. The noise is D405-like
(z^2-scaled, correlated, flying pixels, dropouts, 0.1 mm quantisation),
`brick_position` is off by (4, -3) mm, and the full skill runs through a
fake arm (view move, 3 frames). Each row has 5 yaws (0, 17, 33, 62 and 105
deg):

| brick | face | routes | vertical axis right | stud side named right / named | yaw error max deg |
|---|---|---|---|---|---|
| DUPLO 2x2 | studs up / upside down / side | 5 / 5 / 5 right | 15/15 | 5/5 | 1.3 |
| DUPLO 2x4 | studs up / upside down / side x / side y | 5 / 5 / 5 / 5 right | 20/20 | 10/10 | 0.6 |
| LEGO 2x4 | studs up / upside down / side x / side y | 5 / 5 / 5 / 5 right | 20/20 | 4/4 | 3.2 |
| LEGO 2x2 | studs up / upside down / side | 5 / 5 / 5 right | 15/15 | 4/4 | 0.8 |
| LEGO 1x4 | studs up / upside down / side x / side y | 5 / 5 / 5 / 5 right | 20/20 | 2/2 | 4.4 |
| LEGO 1x2 | studs up / upside down / side x / side y | 5 / 5 / 5 / 5 right | 20/20 | 3/3 | 1.5 |

110/110 faces and vertical axes right. On a side, the stud direction was
named in all 15 DUPLO cases (all right). For LEGO it was named in 14 of 40
(all right); otherwise it is reported as unresolved: the 1.7 mm stud
overhang is a few pixels at 0.2 m. The renderer is ideal geometry: no
specular or dark plastic, no stereo shadowing, no chamfers.

## Limits

- **Plates** (LEGO 3.2 mm): upside down and studs up both reach 4.9 mm,
  and the 2.9 mm hollow leaves no cut that clears the 1.7 mm studs and stays
  inside it with a 1 mm margin (`UPDOWN_MARGIN`). Before that guard, a
  studs-up plate read 25 % deep (on the threshold) and an upside-down one
  55 %. Now the skill routes `unknown` for a flat plate and says why; a plate
  on its side is still read (`on_side`, x vertical in the check).
- **Touching bricks** merge into one cluster: the footprint fits no face,
  so the route is `unknown`. Singulate first.
- **Size must be given**: the skill confirms a face of the stated brick; it
  does not identify the brick (a 2x4 on its side and a 2x2 studs up are
  different readings, and a wrong size usually ends in `unknown`).
- **Stud direction on a side** for LEGO is often unresolved (see Checked);
  a closer view (0.12-0.15 m) would help.
- Not run on a real YAM; not run in the station twin (see Checked).

## The constants, and why

- **`HEIGHT_TOLERANCE_FRACTION = 0.3`**, **`FOOTPRINT_TOLERANCE_FRACTION =
  0.4`**: the source accepted the nearest whole stud unit (half a unit). The
  closest faces here are LEGO 1xN on a side (8 mm) against upside down
  (9.6 mm rim + 1.7 mm studs), 3.3 mm apart, more than the 2.4 mm tolerance.
- **`MARGIN = 1.5`**, **`UPSIDE_DOWN_SHARE = 0.25`**, **`STUDS_UP_SHARE =
  0.10`**, **`DEEP_FRACTION = 0.45`** (the middle of a cavity 0.9 brick
  heights deep: **`TOP_THICKNESS_FRACTION = 0.1`**), **`UPDOWN_MARGIN = 1 mm`**, **`INNER_FRACTION =
  0.2`**, **`STUD_SIDE_FRACTION = 0.5`**, **`OBJECT_FRACTION = 0.35`**: set
  on the synthetic depth; the measured shares were 0.00-0.04 (studs up or
  side) and 0.70-0.85 (upside down).
- **`VIEW_RANGES`**, **`MIN_VIEW_CLEARANCE`**: pressing-a-brick-onto-studs /
  refining-target-from-wrist-closeup.

## Provenance

Adapted from lego-vision.py of UR5-Pick-and-Place-Simulation (MIT; Pietro
Lechthaler et al.): the height-versus-dimensions axis match (getDepthAxis),
the top-surface and silhouette `minAreaRect`, and the stud side from the
silhouette and top-surface offset. Its two YOLOv5 models (AGPL, trained in
Gazebo) are not used; up versus down is a geometric test here. The script's
header lists the changes; licence in LICENSE.ur5pickplace.
