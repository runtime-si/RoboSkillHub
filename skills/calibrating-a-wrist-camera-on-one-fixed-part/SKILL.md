---
name: calibrating-a-wrist-camera-on-one-fixed-part
description: Calibrates a wrist RGB-D camera's pose in the hand (hand-eye)
  with no board and no second camera - the hand looks at one flat part of
  known size lying still on a level support from about twenty poses (turned
  about the vertical, raised, shifted, tilted), each look giving the middle of
  the part's top face and the support's plane in the camera's frame by depth,
  and one camera-in-tool pose is fitted so that through every look's tool pose
  the part is one point and the support one level plane; reports how far the
  looks disagreed before and after and a leave-one-look-out error. Use when
  things a hand's camera sees are not where the hand then goes - a part found
  in different places from different looks, a support read a centimetre off
  the arm's own contact with it - or after a camera mount was touched; needs
  the part's box to a couple of centimetres and the hand starting above it
  with the part in the wrist camera's view.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: calibration
  tags: [calibration, hand-eye, wrist-camera, eye-in-hand, depth, cpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tuned_on: [yam_i2rt]
    tasks: [pick-place, dominoes, bimanual]
    provenance:
      authors: [runtime-si]
      code: yam_dominos real/calibrate_wrist.py, 2026-11-19
    notes: Draft from yam_dominos. The method ran on the real right YAM (26.4 mm to 3.3 mm); this script has not been run through a hub connector.
gap:
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.solve_ik
    - robot.go_to_pose
  required_inputs:
    target_obb: OrientedBoundingBox
  # Optional inputs (wire them from a script node or write them in the node): views (looks already
  # taken, [{T_base_tool, point_cam, plane_n, plane_d, camera_in_tool?}], to fit without moving), poses
  # (tool poses to look from instead of the built-in tour), camera, arm_id, support_z, min_views,
  # max_point_mm, max_left_out_mm, return_to_start.
  produces_outputs:
    camera_in_tool: Se3Pose
    point_spread_mm: float
    point_spread_before_mm: float
    left_out_mm: float
    moved_mm: float
    moved_deg: float
    reason: string
  exit_conditions:
    calibrated: At least min_views (8) looks made the part out, and with the fitted camera_in_tool the part's middle is one point within max_point_mm (2.5 mm rms) over all looks, each look predicted by the others within max_left_out_mm (4 mm rms); the hand is back at its start pose.
    inconsistent: Fitted, but the looks do not agree on one camera pose within those limits (the arm's own error from pose to pose, or poor looks); camera_in_tool and the numbers are filled in, and point_spread_before_mm says whether it is still better than what was.
    too_few_views: Fewer than min_views looks made the part out (poses out of reach, moves refused, the part cut by the picture's edge or hidden); nothing is fitted and reason lists each miss.
  canonical_scripts:
    - calibrate_wrist_camera: scripts/calibrate_wrist_camera.py
  examples:
    - title: Calibrate the wrist camera on one tile lying under the hand
      path: examples/canonical_subgraph.json
---

# calibrating-a-wrist-camera-on-one-fixed-part

What a manipulation policy needs of a wrist camera's calibration is that a
thing seen is where the hand then goes, from any pose: that the camera agrees
with the arm's own joints. So nothing else is used here. One part lies still;
the hand looks at it from many poses; the camera's pose in the hand is the
one for which all those looks put the part in one place.

## How it composes

- **Before it**: any perception skill's box for one flat part lying alone
  (7 cm clear round it), and the hand above it with the part in the wrist
  view (`refining-target-from-wrist-closeup` leaves it so).
- **After it**: `camera_in_tool` (camera to tool, OpenCV camera axes) goes
  into the station's calibration; this skill does not write it anywhere.
- **Not `calibrating-arm-to-arm-base-transform`**: that poses a printed board
  and solves where the arms' bases are. This needs no board and solves one
  camera's mount. A camera calibrated here against its own arm says nothing
  about where that arm's base is: in yam_dominos the right arm's base turned
  out to be recorded 7 cm off, which showed only once its camera was right.

## What it does

1. Reads the tool pose and builds the tour (`view_poses`): the start pose
   turned about the vertical through the part by 0, +-45 and 90 deg, at each
   heading from two heights and two sideways shifts, plus four looks tilted
   15-25 deg about a level axis through the part. Poses of the hand, so a
   wrong calibration cannot turn them into poses the arm does not have.
2. Each pose is IK-previewed, moved to, and a settled observation taken. The
   camera-in-tool pose in use is read from the first observation (the frame's
   pose against the tool's).
3. In each frame (`wrist_depth.flat_parts`) the support is fitted as a plane
   and the part found by depth; the look keeps the tool pose, the middle of
   the part's top face and the plane, both in the camera's frame.
4. `fit`: least squares (soft-L1) over the six numbers of camera-in-tool so
   that the part's middle is one point, the support's normal is vertical and
   its height one height, over all looks.
5. Reports the spread before and after, the worst look, each look left out
   in turn, and how far the calibration moved.

What fixes what: turning about the vertical fixes the camera's place across
the hand; moving without turning fixes its turn; only the tilted looks fix
its place along its line of sight, and weakly. Check the support's height
against a touch (`measuring-a-held-tip-offset-by-contact-descent`) before
trusting a height to a few millimetres.

## Required end states

As in the frontmatter. Extra fields: `T_tool_camera` (4 x 4), `views`,
`point_worst_mm`.

## The constants, and why

Documented where defined in `scripts/calibrate_wrist_camera.py`. The exit
limits (`MAX_POINT_MM` 2.5, `MAX_LEFT_OUT_MM` 4.0) were set before the first
real fit and that fit missed both narrowly (3.3 and 4.2 mm) while being far
better than what was; they are guesses, and `inconsistent` carries the result
for that reason.

## Measured

- **Simulator** (the YAM station's twin, real-sensor depth noise), right arm
  handed a calibration 15 mm and 2.7 deg wrong: 20 looks; the part's middle
  from 6.4 mm rms to 0.21 mm; a look left out 0.24 mm; the fit 1.1 mm and
  0.03 deg from the simulator's own.
- **Real right YAM** (D405, one tile 49 x 24.5 x 7.4 mm), 2026-11-19, 18
  looks: the part's middle from 26.4 mm rms (worst 41 mm) to 3.3 mm (worst
  6.5 mm); a look left out 4.2 mm; the calibration moved 22 mm and 5.0 deg.
  The upright looks agreed to 1-3 mm; the tilted looks were the outliers
  (4-5 mm), and the support's normal still varied with the hand's heading by
  up to 4 deg: what is left is the arm's, not the camera's.

Not measured: this script through a hub connector (the numbers are the
task's own implementation of the same tour and fit, which built its tilted
looks from the task's look poses rather than `view_poses`); a pick made with
the new calibration on the real arm; any robot but the YAM.

## Provenance

From yam_dominos' `real/calibrate_wrist.py`: the tour built from the start
pose by turns about the part instead of the task's pose helpers, robot access
through connector tools, routed exits.
