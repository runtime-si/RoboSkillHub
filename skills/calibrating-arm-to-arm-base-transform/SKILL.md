---
name: calibrating-arm-to-arm-base-transform
description: Measures where one arm's base sits in the other arm's base frame
  - each arm in turn visits a list of joint configurations, its wrist camera
  poses a ChArUco board (or ArUco marker) fixed in the cell, or one fixed
  camera poses a marker each gripper carries; each arm's views are solved as
  robot-world / hand-eye (AX = ZB, Shah's closed form plus least squares) and
  the two "board in my base" results compose into base0 -> base1, with per-view
  residuals and a leave-one-view-out spread. Use when a two-arm task
  (handover, holding a part while the other arm works, one shared grid for
  both arms) needs the arms' relative placement measured instead of the
  nominal layout, or to check it after a base was moved.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: calibration
  tags: [calibration, bimanual, hand-eye, charuco, aruco]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [bimanual, pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/calibration_handeye.cpp
      upstream_license: Apache-2.0
      paper: https://doi.org/10.1115/1.4024473
gap:
  allowed_tools:
    - robot.move_to_joints
    - robot.get_ee_pose
    - robot.get_observation
  required_inputs: {}
  # Optional inputs (gap has no list / open-dict type here; wire them from a script node or
  # write them in the node): joint_configs ({"0": [[q...], ...], "1": [...]}), views (per arm
  # [{T_base_tool, T_cam_target | T_cam_board}] to solve without moving), arm_ids ([0, 1]),
  # cameras ({"0": "left_wrist", "1": "right_wrist"}, or one name for eye_to_hand), mode
  # (eye_in_hand | eye_to_hand), target ({type: charuco, squares, square_m, marker_m,
  # dictionary} or {type: aruco, dictionary, marker_m, id}), assume_level, min_views,
  # max_rms_mm, max_rms_deg, max_jackknife_mm.
  produces_outputs:
    yaw_deg: float
    tilt_deg: float
    jackknife_mm: float
    jackknife_deg: float
    reason: string
  exit_conditions:
    calibrated: Both arms posed the target in at least min_views (6) views, each arm's views agree on it within max_rms_mm (6 mm) / max_rms_deg (1.5 deg) RMS, and leaving any one view out moves base0 -> base1 by at most max_jackknife_mm (5 mm); T_base0_base1 is arm 1's base in arm 0's base frame.
    residual_too_high: Solved, but an arm's views disagree (RMS over the limits) or one view decides the result (leave-one-out over 5 mm); T_base0_base1 and the numbers are filled in - add views from more varied angles, or check the kinematics / camera intrinsics.
    tag_not_seen: An arm posed the target in fewer than min_views views (moves refused, no frame, target not found); nothing is solved and `reason` lists the misses.
  canonical_scripts:
    - calibrate_arms: scripts/calibrate_arms.py
  examples:
    - title: The YAM station's tour of the board between the arms (real joint configurations)
      path: examples/canonical_subgraph.json
---

# calibrating-arm-to-arm-base-transform

Two arms that hand over parts, or hold a plate while the other presses, need
to know where the other arm's base is. The usual source is the nominal
layout ("0.62 m to the right, parallel"). This skill measures the placement
with the arms' own cameras instead.

## What it does

1. **Tour.** For each arm (`arm_ids`, default 0 then 1), it visits each
   joint configuration in `joint_configs[arm]` with `robot.move_to_joints`
   (`arm_id` is passed only when nonzero). A refused move is checked both
   ways, as an exception or as `outcome` / `reached`; it is skipped and
   reported in `reason`. At each pose it reads `robot.get_ee_pose` and the
   arm's camera frame from `robot.get_observation`, which needs `rgb` and
   `intrinsics`, and `distortion` if your connector provides OpenCV
   coefficients.
2. **Target pose.** A ChArUco board, by default the station's printed 6 x 8
   board of 30 mm squares with 22 mm DICT_5X5_100 markers. Its corners are
   found on a 2x upscaled grey copy with the station's detector settings,
   and the pose is solved by IPPE PnP plus LM refinement. A single ArUco
   marker works too (`target={"type": "aruco", ...}`, IPPE_SQUARE). The
   frames follow OpenCV: the board's x along its rows and y down, with z
   into the board; a marker's x right and y up, with z out of it.
3. **Solve per arm.** Each arm solves `A_i X = Z B_i`. The closed form is
   Shah's Kronecker method (the one OpenCV's `calibrateRobotWorldHandEye`
   offers; OpenCV 5's Python build no longer exposes it). Least squares
   then refines it, weighing 1 rad of rotation like 0.1 m.
   - **eye_in_hand:** each wrist camera sees the board fixed in the cell. A
     = tool pose, B = inverse board pose, X = the wrist camera's hand-eye,
     Z = the board in that base.
   - **eye_to_hand:** one fixed camera sees a marker that each gripper
     carries in turn. X = the marker on the tool, Z = the camera in that
     base.
4. **Compose.** `T_base0_base1 = Z0 @ inv(Z1)`. `assume_level` reduces the
   rotation to a turn about z, for two bases bolted to one flat table.
5. **Judge.** Each arm's RMS of where each view alone puts Z, and the
   largest change from dropping any single view (`jackknife_mm` / `_deg`).

You can also pass recorded `views` (per arm `[{T_base_tool, T_cam_target}]`;
the station's `T_cam_board` key is accepted). The skill then solves without
moving the arms, for example to re-solve a stored tour or to check a base
after it was moved.

## Outputs

- `T_base0_base1` (4x4), `translation_m`, `rpy_deg`, `yaw_deg`, `tilt_deg`.
- `jackknife_mm` / `jackknife_deg`.
- `arms`: per arm, the view count, `rms_mm` / `rms_deg`, `per_view_mm`, the
  solved hand-eye (`T_tool_cam`, or `T_tool_target` for eye_to_hand), and Z.
- `reason`.

**Store the result next to the station calibration.** The skill does not
write files.

## Measured: the YAM station's own tours (real, 2026-11-05)

The station recorded wrist-camera tours of one ChArUco board lying between
the two arms (`real/calibration/yim-system/validation_middle*`):
- the left arm at 13:38, 9 views;
- the right arm at 13:41, 7 views;
- the right arm again at 14:14, 10 views, after its base was re-set. The
  station found the base had turned only 0.04 deg; the board did not move.

Each view has the arm server's tool pose and the board pose from the
station's own detector on the raw frame. Solved with this skill's
`run(views=...)`:

| tours | route | base_R in base_L [m] | yaw | jackknife | RMS left / right |
| --- | --- | --- | --- | --- | --- |
| left 13:38 + right 13:41 (7 views) | residual_too_high | (0.6274, 0.0196, -0.0038) | -0.49 deg | 8.8 mm, 0.58 deg | 3.1 / 2.6 mm, 0.57 / 0.54 deg |
| left 13:38 + right 14:14 (10 views) | calibrated | (0.6292, 0.0187, 0.0034) | 1.93 deg | 3.3 mm, 0.29 deg | 3.1 / 3.0 mm, 0.57 / 0.47 deg |

What to know:

- **The 7-view right tour was flagged.** Leaving out one of its views moved
  the answer by 8.8 mm. The two results differ by 7.6 mm and 2.5 deg of
  yaw, although the station measured the base turning by 0.04 deg. Trust
  the 10-view tour more, and tour each arm with at least 9 or 10 varied
  views.
- **Accuracy is about 1 cm, not 1 mm.**
  - The left solve puts the board 4.8 mm and 1.5 deg away from the overhead
    camera's reading of it. That camera was calibrated with the left wrist
    and joint-offset kinematics. Its 1.5 deg difference, over the
    ~0.4 m lever, becomes 16 mm at the right base.
  - The station's archived estimates of the same transform
    (`right_base_from_board*.json`) range over 47 mm in x and y. This
    result lies 6 mm from the one it adopted at 12:35, (0.6234, 0.0161).
  - It is 21 mm from the nominal layout (0.62, 0, 0), mostly in y. A
    likely cause: the right arm's joint-1 zero is off by -2.4 deg (the
    station's fit), which shifts 0.45 m of reach by ~19 mm. The solve folds
    that error into the base transform, which is what the arm server's
    uncorrected poses need.
- **No touch check was run.** For handovers, verify with a tip-to-tip
  touch at 2-3 points after calibrating, and refine the station's joint
  zeros first if you need millimetres.

**Detector check:**
- Four tilted board poses rendered at known positions through a 600 px pinhole
  were recovered within 1 mm and 0.2 deg (unit tests). A head-on view had
  2.3 mm of depth error (planar PnP ambiguity), so tilt the views.
- The stored station PNGs carry drawn corner annotations, so the detector
  could not be re-run on them.

**Simulation check:** not run. The YAM twin has no fiducial asset to put in
front of the cameras. The motion path (tour, refused moves, `arm_id`
passing, per-arm camera) is covered by the FakeContext tests, which render
the board for each pose.

## Licence

Written for the hub (Apache-2.0) from Shah (2013) and the OpenCV method it
names; no OpenCV source is copied. The detector settings are those of the
station's `real/calibrate_board.py`. Bi-JCR (markerless dual-arm
calibration) has no licence and was not used.
