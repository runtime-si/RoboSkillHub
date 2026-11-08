---
name: calibrating-baseplate-stud-grid
description: Estimates a LEGO or DUPLO baseplate's pose (the plate frame
  computing-brick-pose-from-stud-grid takes, and APEX-MR's x, y, z, roll,
  pitch, yaw, width, height) in two steps - a coarse pose from ArUco /
  AprilTag markers at known places on or beside the plate (OpenCV aruco +
  solvePnP), a given pose, or a plate mask and its depth plane; then a
  sub-millimetre refinement from a wrist RGB-D close-up, where the stud tops
  are found in depth and as circles in the plate-rectified image and an 8 mm
  (16 mm DUPLO) lattice is fitted by RANSAC (yaw + offset, the coarse pose
  picking the period) - with the residual and the number of studs used. A
  second script reads a brick's own studs - footprint (A x B), top pose and
  stud pixels from a close-up and a mask. Use when bricks will be placed or
  picked by grid cell and the plate's pose must be known to well under a
  pitch, or a brick's size and pose must be read from its studs.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [lego, duplo, assembly, calibration, fiducials, aruco, apriltag, wrist-camera, d405, opencv, cpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.1
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      code: https://github.com/intelligent-control-lab/APEX-MR/blob/6b5a17f1133c3f739caf50fa0fff6c478ef73b7a/config/lego_tasks/plate_calibration.json
      upstream_license: MIT (APEX-MR plate format); Apache-2.0 (OpenCV aruco, solvePnP, HoughCircles)
gap:
  requires: {connector: [robot.grasp_frame]}
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.solve_ik
    - robot.grasp_frame
    - sam3.segment_text
  required_inputs: {}
  produces_outputs:
    plate_pose: Se3Pose
    coarse_plate_pose: Se3Pose
    coarse_source: string
    lattice_status: string
    residual_mm: float
    max_residual_mm: float
    studs_used: int
    studs_detected: int
    snap_margin: float
    marker_reprojection_px: float
    yaw_ambiguity_deg: float
    system: string
    stud_pitch: float
    camera_name: string
  exit_conditions:
    calibrated: A coarse pose was found, a wrist close-up was taken, at least min_studs (12) stud tops lie on the lattice with an RMS residual under max_residual (1/8 pitch - 1.0 mm LEGO, 2.0 mm DUPLO), the yaw correction is under 10 deg and the coarse pose is within snap_tolerance (0.35 pitch) of the nearest stud row along both axes; plate_pose is the refined pose.
    coarse_only: A coarse pose was found but no close-up was used - refine off, no wrist camera with a pose (and depth), or no view pose reached (IK preview, refused or short moves); plate_pose is the coarse pose and lattice_status says which.
    no_lattice: A close-up was taken but the lattice is not trusted - no plate plane near the coarse one (no_plane), too few studs (too_few_studs), residual over max_residual (high_residual), yaw correction over 10 deg (yaw_off) or the coarse pose about half a pitch off (ambiguous); plate_pose stays the coarse pose, the fit's numbers are filled in.
    no_markers: No coarse pose - none of the listed markers seen, no coarse_plate_pose, no usable plate mask; plate_pose is empty.
  canonical_scripts:
    - calibrate_plate: scripts/calibrate_plate.py
    - read_studs: scripts/read_studs.py
  examples:
    - title: Calibrate a 32 x 32 LEGO baseplate from two ArUco markers and a wrist close-up
      path: examples/canonical_subgraph.json
    - title: Read a brick's footprint and top pose from its studs
      path: examples/reading_a_brick_subgraph.json
---

# calibrating-baseplate-stud-grid

Every grid-to-metric conversion (computing-brick-pose-from-stud-grid, then
pressing-a-brick-onto-studs) trusts the plate's pose; a LEGO press-fit needs
it to a fraction of a millimetre. Markers seen from 0.9 m give the plate to a
few millimetres; the studs, seen by the wrist D405 from 0.2 m, give it to a
tenth of one. The lattice repeats every pitch, so the coarse pose is what says
which stud is which.

## How it composes

- **Before it**: nothing but the plate on the table and, for the coarse step,
  either markers at known places in the plate frame (printed beside the plate
  or stuck on it), a rough `coarse_plate_pose` (a previous calibration, a
  hand measurement), or a mask of the plate (`plate_mask`, e.g. from sam3, or
  `plate_prompt` and the skill calls `sam3.segment_text` on the overhead
  frame).
- **After it**: `plate_pose` straight into computing-brick-pose-from-stud-grid's
  `plate_pose` (with `plate_studs`, `system` and `plate_thickness: 0`: the
  pose is the plate's top surface at the stud bases). `apex_plate` is the same
  pose as an APEX-MR `plate_calibration.json` entry.
- **The hand** is left at the last view pose; the next skill moves from there.

## Conventions

The plate frame is computing-brick-pose-from-stud-grid's (ICL's): origin at
the centre of the plate's top surface (the stud bases), x along its first stud
count, z up out of the studs, so stud (i, j) sits at
`((i + 0.5 - nx / 2) * pitch, (j + 0.5 - ny / 2) * pitch, 0)`. A marker is
`{id, size, position: [x, y, z], yaw_deg}` in that frame (face up, OpenCV's
marker axes: x right, y up the printed image, z out of its face; `size` the
black square's side) or `{id, size, pose}`; a marker on the table beside a
plate 3.2 mm thick has z = -0.0032. `apex_plate` has roll, pitch, yaw with
R = Rz(yaw) Ry(pitch) Rx(roll), `deg` the yaw in degrees, `width` / `height`
the stud counts along x / y (ICL's `plate_w` / `plate_h`).

## What it does

1. **Coarse pose**, the first source that works:
   - *markers*: `cv2.aruco.ArucoDetector` (sub-pixel corners) in every camera
     (or `coarse_camera`); the one seeing the most listed markers wins; all
     their corners go into one `solvePnP` (IPPE when coplanar, SQPnP if not),
     refined by Levenberg-Marquardt, and are lifted with the camera's pose.
     Any OpenCV dictionary (`DICT_4X4_50`, `DICT_APRILTAG_36h11`, ...).
   - *coarse_plate_pose* as given.
   - *plate mask*: the masked depth of the overhead camera (or
     `coarse_camera`), a RANSAC plane (4 mm band), and the min-area rectangle
     of its points: the centre, and the yaw with the sides matched to the
     stud counts. A square plate's yaw is only known modulo 90 deg, an oblong
     one's modulo 180 (`yaw_ambiguity_deg`); the turn nearest `yaw_hint_deg`
     is taken. Use markers when cell (0, 0) must be a particular corner.
2. **Close-up**: with `look` (default) the wrist camera is placed on the plate
   normal through each of `view_points` (plate-frame x, y; default the
   centre) at `camera_range` (0.20 m; then 0.25, 0.15, 0.30), looking straight
   down: camera-to-tool from the observation, the current and
   `robot.grasp_frame`'s rotations turned to look down, an IK preview, a rise
   in place, `robot.go_to_pose`, arrival read back (15 mm) and a settle wait.
   Without `look`, the wrist frame where the arm already is.
3. **Plate plane**: close-up depth within 35 mm of the coarse plane and over
   the plate, RANSAC (band 0.35 x stud height); when a plane lower by over
   60 % of a brick holds a quarter of the points, it is taken instead (brick
   tops covering the view are not the plate). A marker pose's origin is then
   slid along the marker camera's ray onto this plane: PnP on small, far
   markers errs mostly in range, which an oblique overhead camera turns into
   a sideways error too (on the YAM twin 5-24 mm low and 2-10 mm sideways,
   along the ray).
4. **Studs**, in the plane frame: depth points splatted into a height map at
   pitch / 32; blobs at the stud top's height, of a stud's area and shape;
   HoughCircles at the stud's metric radius on the RGB rectified onto the
   plane at stud-top height (CLAHE, blur), kept where the height map shows a
   stud top. A pair of both becomes the depth one; every candidate is then
   re-centred on the centroid of its raw top-disc depth points.
5. **Lattice**: RANSAC over neighbour pairs (one pitch apart: the yaw; one
   point: the offset) on the lattice the plate's stud-count parity gives, then
   rigid Procrustes over the assigned nodes (one candidate per node). Of the
   offsets one pitch apart, the one with the smallest correction at the
   studs' centroid is taken; its size in pitches is `snap_margin`.
6. **Refined pose** = coarse * plane frame * lattice (yaw, offset); `correction`
   is it in the coarse frame.

## The second script: read_studs

`scripts/read_studs.py` reads one brick's top from a close-up: `mask` (the
brick, e.g. sam3.segment_box) and `frame` (a CameraFrame) or `camera` (a
name in robot.get_observation; default the wrist camera). The top face is the
highest well-supported plane in the mask; the same stud detection and a free
lattice fit (no coarse pose) give `brick_size` `[a, b]` (a <= b, along
`top_pose`'s x and y), `top_pose` (the top centre at the stud bases, z up -
computing-brick-pose-from-stud-grid's `observed_reference: top`),
`lattice_pose_camera`, `stud_px`, `stud_points`, `stud_indices`,
`residual_mm`, `yaw_ambiguity_deg` (180, or 90 for a square brick). A ray
probe 1.5 mm past each side of the footprint says whether the depth falls
there (`edge`), the top goes on (`continues`), something is in front
(`occluded`) or nothing is seen (`unseen`). Exits: `lattice_found` (every
stud of the A x B rectangle seen and each side an edge), `occluded` (studs
missing inside it, or a side not an edge: the footprint is a lower bound;
`open_sides` lists them), `too_few_studs` (no flat top, no stud, a residual
over 1/8 pitch). A single stud takes its yaw from the mask's outline. Without
a mask it reads the dominant plane's studs (free studs of a plate) and skips
the side test.

## Inputs (calibrate_plate)

- `plate_studs` `[nx, ny]` (required), `system` (`lego` default, or `duplo`);
  `stud_pitch`, `stud_height`, `stud_diameter` override the preset.
- Coarse: `markers`, `marker_dictionary` (`DICT_4X4_50`), `coarse_plate_pose`,
  `plate_mask`, `plate_prompt`, `coarse_camera`, `yaw_hint_deg` (0).
- Close-up: `refine` (true), `look` (true), `view_points`, `camera_range`,
  `wrist_camera`, `detector` (`both`, `depth`, `rgb`), `check_reach` (true),
  `reach_tolerance` (0.015 m); `arm_id`, `tolerance`, `max_steps` passed on
  only when set.
- Gates: `min_studs` (12), `max_residual` (1/8 pitch), `snap_tolerance`
  (0.35), `max_yaw_correction_deg` (10), `plane_window` (0.035 m); `seed`.

## Outputs (calibrate_plate)

`route`, `detail`, `plate_pose`, `coarse_plate_pose`, `coarse_source`
(`markers`, `input`, `mask_plane`), `markers_used`, `marker_reprojection_px`,
`yaw_ambiguity_deg`, `lattice_status` (`ok`, `not_requested`,
`no_wrist_camera`, `unreachable`, `no_plane`, `too_few_studs`,
`high_residual`, `yaw_off`, `ambiguous`), `correction` (x, y, z, yaw_deg,
tilt_deg, lateral of the refined pose in the coarse frame, at_view_x / y,
the correction where the studs were seen, and ray_slide, how far a marker
pose was slid along its camera's ray [m]), `residual_mm`, `max_residual_mm`,
`studs_used`, `studs_detected`, `snap_margin`, `plate_studs`, `system`,
`stud_pitch`, `apex_plate`, `views` (view point, range, view pose),
`camera_name`.

## Checks

CPU only (no GPU, no simulator). Unit tests
(`tests/bundles/test_calibrating_baseplate_stud_grid.py`, 43) on ray-cast
RGB-D renders (`tests/bundles/_plate_render.py`: plate, stud cylinders with
bevelled tops, seated bricks with studs, markers printed on the table,
Lambertian RGB, D405 depth noise sigma = z^2 * 0.08 / (f * 18 mm) partly
spatially correlated, flying pixels at depth edges, 0.5 % dropouts, 0.1 mm
steps): the lattice maths; each coarse source; every exit and lattice_status;
a plate covered by bricks; an odd 15 x 9 plate; the look-down move with a fake
YAM-like arm (camera 86 mm behind and 75 mm above the tool, 45 deg forward;
the view checked on the plate normal at 0.20 m), IK refusals, routed and
raised refusals, a 3 cm short move, two view points pooled, connector knobs;
read_studs on 1x1 to 2x6 LEGO and DUPLO bricks, a brick cut by the image
border, a brick half covered by another, a bare table, unposed frames; a
marker pose 1.4 % too far along the overhead ray (the twin's error, below)
slid back onto the close-up plane.

**Simulation check** (YAM station twin, left arm, run on an RTX 4090 D; the
harness is not in the hub). The twin has no LEGO, so the harness adds a
DUPLO 8 x 8 and a LEGO 16 x 16 baseplate (solid stud cylinders, 128 mm
square) as free bodies, each with two 50 mm DICT_4X4_50 markers printed as
textures on flanges beside the studs, seen by the station's calibrated
overhead camera (640 x 480, ~1 m, ~60 deg down; 22-26 px markers) and the
real left D405's calibration on the wrist. Each trial puts the plate at a
known pose, runs the skill closed loop with the default inputs
(`camera_range` 0.20, `look`), and compares `plate_pose` with the plate's
true pose. The wrist view was reached first time in every trial (hand
~60-86 mm over the plate), a few studs hidden by the open fingers.

| run | plates | routes | `plate_pose` lateral (mm) | z (mm) | yaw (deg) | tilt (deg) | studs used | RMS (mm) |
|---|---|---|---|---|---|---|---|---|
| 0.1.1, markers | DUPLO, 5 poses (yaw -25..35 deg) | calibrated 5 | 0.04-0.10 | 0.014-0.019 | <= 0.04 | <= 0.012 | 54-57 of 64 | 0.11-0.30 |
| 0.1.1, markers | LEGO, 5 poses | calibrated 5 | 0.03-0.09 | 0.004-0.005 | <= 0.006 | <= 0.004 | 221-240 of 256 | 0.10-0.15 |
| 0.1.1, given coarse | DUPLO 5 mm / 3 deg off; LEGO 2.5 mm / 2 deg off | calibrated 2 | 0.10; 0.08 | 0.02; 0.006 | 0.009; 0.000 | 0.015; 0.005 | 53; 224 | 0.09; 0.10 |
| 0.1.0 (before), markers | DUPLO and LEGO, 4 poses each | calibrated 4, no_lattice 4 | 0.04-0.10 when calibrated | | | | | |

The marker poses themselves were off 2-10 mm sideways, 5-24 mm low, up to
6.4 deg in tilt and 1.2 deg in yaw (0.33-0.72 px RMS), almost all of it
along the camera's ray: the rendered corners lie ~0.4 px inside the true
ones, so PnP puts the plate ~1.5 % too far. Version 0.1.0 used the marker
pose as it was and failed half the marker trials: `ambiguous` twice (4.9 mm
on LEGO, 6.3 mm on DUPLO: past the snap tolerance), `no_plane` (24 mm low,
outside the 15 mm window) and `high_residual` (19 mm low: the window cut
the plate to 16 studs). This version slides the marker pose along the ray
onto the close-up plane (5-27 mm slides), which left it 0.01-0.09 pitch
from the nearest stud row in every trial, and widens the window to 35 mm.
No trial in either version returned a wrong stud row as `calibrated`.

**Accuracy** (`python -m tests.bundles._plate_calib_eval --trials 20
--studs 40 --real`, seed 0, 809 s): a plate at a random yaw and up to 2 deg
tilt, 0-14 random bricks, two 40 mm ArUco markers beside it seen by an
overhead camera at 0.85-0.95 m; the wrist frame from 0.18-0.25 m, up to
25 deg off vertical, over a random point of the plate. Errors of
`plate_pose` against the truth, median / p90 / max:

| system | step | routes (of 20) | lateral (mm) | z (mm) | yaw (deg) | tilt (deg) | studs used | RMS (mm) |
|---|---|---|---|---|---|---|---|---|
| LEGO 32x32 | markers, coarse | coarse_only 20 | 1.14 / 2.35 / 3.16 | 1.07 / 1.96 / 2.67 | 0.45 / 0.92 / 1.18 | 3.4 / 4.6 / 6.7 | - | - |
| LEGO 32x32 | markers + lattice | calibrated 19, no_lattice 1 | 0.09 / 0.17 / (3.16) | 0.07 / 0.19 / (0.66) | 0.01 / 0.02 / (1.18) | 0.07 / 0.17 / (6.7) | 838 | 0.42 |
| LEGO 32x32 | given coarse (0-2.5 mm, 0-2 deg off) + lattice | calibrated 20 | 0.10 / 0.16 / 0.25 | 0.13 / 0.28 / 0.43 | 0.01 / 0.01 / 0.03 | 0.06 / 0.12 / 0.16 | 819 | 0.41 |
| LEGO 32x32 | same, wrist pose off 0.5 mm / 0.2 deg | calibrated 20 | 0.62 / 1.06 / 1.12 | 0.25 / 0.42 / 0.64 | 0.09 / 0.18 / 0.19 | 0.16 / 0.21 / 0.27 | 832 | 0.43 |
| DUPLO 16x16 | markers, coarse | coarse_only 20 | 1.48 / 2.58 / 3.09 | 1.24 / 1.86 / 2.08 | 0.51 / 0.95 / 1.20 | 4.1 / 6.0 / 6.3 | - | - |
| DUPLO 16x16 | markers + lattice | calibrated 20 | 0.07 / 0.14 / 0.15 | 0.05 / 0.06 / 0.10 | 0.02 / 0.04 / 0.07 | 0.02 / 0.03 / 0.05 | 187 | 0.60 |
| DUPLO 16x16 | given coarse + lattice | calibrated 20 | 0.06 / 0.15 / 0.16 | 0.05 / 0.06 / 0.07 | 0.01 / 0.04 / 0.06 | 0.02 / 0.02 / 0.04 | 186 | 0.62 |
| DUPLO 16x16 | same, wrist pose off 0.5 mm / 0.2 deg | calibrated 20 | 0.56 / 0.99 / 1.12 | 0.32 / 0.45 / 0.64 | 0.11 / 0.18 / 0.19 | 0.18 / 0.19 / 0.20 | 186 | 0.62 |

The one LEGO `no_lattice` is `ambiguous`: its marker pose was 3.16 mm off,
over the 2.8 mm snap tolerance, and in these renders the error is not along
the camera's ray (the PnP range is good to ~1 mm), so sliding onto the plane
does not save it; its row keeps the coarse pose (the bracketed maxima). No trial returned a wrong stud row as `calibrated`. The hand-eye row
is the point of the last limit below: a 0.2 deg camera rotation at 0.2 m
moves the plate ~0.7 mm and the fit's RMS does not change. A single 50 mm
AprilTag at 0.85 m (unit test) gives ~1 cm and ~4 deg: use two markers or
more, far apart.

**read_studs** (40 random bricks 1x1 to 2x6, random yaw, 0.15-0.20 m, up to
25 deg off vertical): 40/40 `lattice_found` with the right size; top centre
0.09 / 0.21 / 0.50 mm, yaw 0.18 / 0.59 / 4.33 deg (the 4 deg is a 1x1, whose
yaw comes from the mask).

**Real frames**: none on disk shows LEGO or DUPLO. The only lattice-like
pattern is the black perforated board in
`~/yam_gap/threading/cable_frame_20261103-1421.npz` (D405, 0.92 m away, holes
~3 px). read_studs with `detector: rgb`, a hand-drawn mask and the depth limit
lifted to 2 m for the check: 70 holes on a 7 x 11 lattice at 25.0 mm pitch,
1.18 mm RMS (~0.6 px); at 25.4 mm (1 inch) 2.52 mm, at 12.5 / 20 / 50 mm no
lattice. So the board is a 25 mm metric breadboard and the fit tells 0.4 mm
of pitch apart on real pixels; no ground-truth pose exists, so no pose error
is reported. Route `occluded` (7 holes under fixtures). The same board at the
bottom edge of `real/snapshots/20261103-114411-threading-setup/wrist_left.npz`
(0.33 m, ~60 deg grazing, 43 % depth fill) was not read: the plane fit took
another surface, 0 holes. No real frame was used for calibrate_plate.

## Limits

- **Checked on a twin, not on a real arm.** The YAM twin's run (Checks)
  covers the look-down move and the whole loop, but its plates are ideal:
  solid, flat-topped studs with no logo or wear (real DUPLO baseplate studs
  are hollow), perfect depth (no D405 noise, no flying pixels), the wrist
  camera's pose exact, the plate flat on the table, no bricks on it, 50 mm
  markers in plain view. The hand-eye and depth-noise limits below are
  measured on renders only. The view was never refused, so the fallback
  ranges and second rotation did not run in the twin.
- Marker poses can be off by over a pitch along the overhead camera's ray
  (24 mm low on the twin): the slide onto the close-up plane removes that
  part; what remains across the ray (yaw, in-plane offset) must still be
  under the snap tolerance. A camera looking at the plate within ~12 deg of
  its plane gets no slide.
- The world pose is only as good as the wrist camera's pose (hand-eye
  calibration and arm kinematics): their error adds one to one to
  `plate_pose`. The fit's residual does not see it.
- The coarse pose must be within about a third of a pitch at the studs seen
  (2.8 mm LEGO, 5.6 mm DUPLO), after a marker pose's slide. Two 40 mm
  markers at 0.9 m did that 19 times in 20 for LEGO in the renders (the 20th
  was `ambiguous`, not wrong), two 50 mm markers at ~1 m 10 times in 10 on
  the twin; one marker does not.
  A plate mask did it in the unit test's render, but a real overhead mask
  (5.4 mm RMS on the station) will often not for LEGO.
- Not measured on real LEGO / DUPLO: no such frames exist yet; the RGB circle
  detector on real stud tops (logos, specular highlights) is untested, the
  depth detector carries the renders' numbers.
- Lattice ambiguity is resolved only by the coarse pose: a plate edge in the
  close-up is not used to index the rows.
- The depth detector needs the D405 inside 0.6 m (`MAX_DEPTH`); the frames are not
  undistorted (the station's left D405 has k1 = -0.054, under a pixel near the
  centre).
- `read_studs`: a top more than 30 deg from level (from facing the camera,
  for an unposed frame) is not found; a 1 x 1 brick's yaw comes from its mask.

## The constants, and why

- **Presets**: the hub's LEGO / DUPLO values (computing-brick-pose-from-stud-grid,
  pressing-a-brick-onto-studs).
- **`CAMERA_RANGE = 0.20`**, fallbacks, `MIN_TOOL_CLEARANCE`, `REACH_TOLERANCE`,
  settling: refining-target-from-wrist-closeup's, measured on the YAM twin.
- **`RECT_PX_PER_PITCH = 32`**: a LEGO stud's radius is 9.6 rectified pixels.
- **`TOP_BAND = 0.6 mm`**: ~1.5 sigma of a D405 at 0.2 m (z^2 * 0.08 /
  (f * 18 mm)); the band keeps only the visible side wall's top 0.6 mm.
- **Gates**: `MIN_STUDS = 12`, `RESIDUAL_FRACTION = 1/8`,
  `SNAP_TOLERANCE = 0.35`, `MAX_YAW_CORRECTION_DEG = 10`: see the exits; the
  renders' residuals are 0.41 mm (LEGO) and 0.61 mm (DUPLO), well under the
  gates.
- **Marker PnP**: SQPnP and each marker's two IPPE_SQUARE poses, LM-refined,
  the most level of those within 2x the best RMS. OpenCV 5's general IPPE gave
  flipped poses (22 px RMS) for corners off the z = 0 plane; a depth plane in
  the marker quads (tried) tilted the plate 8 deg at 0.9 m, so it is not used.
- **`PLANE_WINDOW = 35 mm`** over the coarse z: the twin's marker poses were
  5-24 mm low (the earlier 15 mm lost two plates of eight), plus tilt;
  `STEP_PLANE_SHARE = 0.25`. `MIN_RAY_PLANE_COS = 0.2`: no slide along a ray
  within ~12 deg of the plate plane.

## Provenance

The output format is APEX-MR's `config/lego_tasks/plate_calibration.json`
(MIT; ICL's hand-measured plate entries: x, y, z, roll, pitch, yaw, deg,
width, height); APEX-MR has no calibration code, so the estimation is new.
Markers: OpenCV's aruco module and solvePnP; studs: OpenCV's HoughCircles and
connected components (Apache-2.0, through cv2). The stud-grid idea (CLAHE,
multi-scale Hough circles, a regular grid fitted to the centres) follows
BrickScan's `ml/preprocessing/stud_detector.py`
(github.com/edrickchang13/BrickScan), which has no licence: reference only,
none of its code is used, and its 1D histogram clustering in image axes is
replaced by a known-pitch metric lattice fitted by RANSAC on the plate plane.
The camera aiming is adapted from refining-target-from-wrist-closeup (RoboRSI,
Apache-2.0).
