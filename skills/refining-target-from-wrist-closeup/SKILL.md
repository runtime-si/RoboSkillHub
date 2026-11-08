---
name: refining-target-from-wrist-closeup
description: Refines a coarse world-frame target (an overhead perception's OBB
  or point) by aiming the wrist RGB-D camera at it from 0.15-0.3 m - the hand
  is placed so the camera's optical axis passes through the target, looking
  straight down when the arm reaches that - then re-pointing the object in the
  close-up frame with a pointing model (gemini-er.point or molmo.point_prompt)
  and back-projecting the pointed pixel with the wrist depth and the camera's
  pose to a refined world point and a shifted OBB. Use when an overhead
  detection is too coarse for the next step (a small part, a grasp that
  missed by 1-3 cm, a target the overhead camera sees at a few pixels) and the
  arm carrying the wrist camera is free.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, active-perception, wrist-camera, eye-in-hand, pointing, d405]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, insertion, sorting]
    provenance:
      authors: [runtime-si]
      code: https://github.com/nssmd/RoboRSI/tree/main/roborsi/embodied/skills/base/find_object_via_wrist/robotwin
      upstream_license: Apache-2.0
gap:
  requires: {connector: [robot.grasp_frame]}
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.solve_ik
    - robot.grasp_frame
    - gemini-er.point
    - molmo.point_prompt
  required_inputs:
    object_name: str
  produces_outputs:
    refined_point: Vec3
    refined_obb: OrientedBoundingBox
    coarse_point: Vec3
    drift: float
    depth: float
    camera_range: float
    view_pose: Se3Pose
  exit_conditions:
    refined: The wrist camera saw the coarse target from camera_range (0.15-0.30 m), the pointer found the object, the wrist depth round the pointed pixel was valid, and the back-projected point lies within the object's half-diagonal + 30 mm of the coarse estimate in the plane; refined_point and refined_obb carry it. The hand stays at the view pose.
    not_seen: There is no wrist camera in the observation, or the pointer found nothing (or failed), or fewer than 3 depth readings in 7-50 cm lay round the pointed pixel, or the pointed point lies further from the coarse estimate than allowed (another object); `detail` says which. The hand stays at the view pose if it moved.
    unreachable: No view pose passed the IK preview and the clearance floor, or three view moves were refused, ended more than reach_tolerance off, or left the target outside the wrist image; `detail` lists each attempt.
  canonical_scripts:
    - refine_from_wrist: scripts/refine_from_wrist.py
  examples:
    - title: Refine an overhead OBB with a wrist close-up
      path: examples/canonical_subgraph.json
---

# refining-target-from-wrist-closeup

Two-step localisation. An overhead camera puts a small object within 1-3 cm;
the wrist camera, brought 0.2 m from it, sees the same object at about 2 px
per millimetre and its depth to well under a millimetre. This skill moves the
camera there, points the object again in that frame and returns the refined
world point.

## How it composes

- **Before it**: a perception skill gives the coarse target -
  `perceiving-objects` (its OBB), `perceiving-objects-oneshot`, or any point.
  This skill does not detect anything from the overhead camera itself
  (RoboRSI's version ran its own head-camera pointing step first).
- **After it**: `refined_obb` is the input box moved in the plane by the
  correction, so a grasp skill or `approaching-above-a-target` takes it in
  place of the coarse box.
- **Not `approaching-above-a-target`**: that skill parks the *tool centre*
  above a box. On a YAM the D405 sits 86 mm behind and 75 mm above the tool
  centre, looking 45 deg forward of the fingers, so a hand parked above the
  target sees it at the edge of the image (about 28 deg off the axis at a
  0.2 m hover, against a 31 deg half field of view). This skill places the
  *camera*.
- **Not `perceiving-objects`' wrist gate**: that uses a wrist frame wherever
  the arm already is; this one moves the camera first.

## What it does

1. Reads one observation; picks the wrist camera (`camera`, else the
   `eye_in_hand` / `wrist` camera riding nearest the tool) and takes the
   camera-to-tool transform from that frame's pose and the arm's tool pose.
   No calibration file is needed: the connector already poses the frame.
2. The coarse target is the top centre of `target_obb` (or `target_point`).
3. Candidate hand rotations: the current one and `robot.grasp_frame`'s two
   top-down ones, each first turned so the camera's optical axis points
   straight down (the best depth, no parallax), then as is (an oblique
   45 deg view on a YAM). A given `rotation` is the only one tried.
4. For each rotation and range (`camera_range`, then 0.20, 0.15, 0.25,
   0.30 m), the hand pose that puts the target on the optical axis at that
   range; skipped when the tool centre would come within 50 mm of the
   target's top or `robot.solve_ik` refuses it.
5. Rises in place to the view height, then `robot.go_to_pose`; judges the
   arrival from a fresh pose (15 mm), waits until two observations agree on
   the tool within 0.5 mm / 0.1 deg, and checks the coarse target projects
   inside the wrist image (8 % margin). A miss tries the next candidate, at
   most three moves.
6. Points `query` (default "the centre of the top face of the
   <object_name>") with `pointer`; of several points the one nearest the
   coarse target's projection wins.
7. Depth: median of the readings in 7-50 cm in a 7 x 7 window; back-projects
   with the frame's intrinsics and pose.
8. Refuses a point further than the object's half-diagonal + 30 mm (or
   `max_drift`) from the coarse one in the plane as another object.

## Inputs

- `object_name` (required): what the pointer looks for.
- `target_obb` or `target_point`: the coarse estimate, world frame.
- `location` (default "the centre of the top face"), `query` (overrides the
  built query), `pointer` (default `gemini-er.point`; `molmo.point_prompt`
  works the same).
- `camera_range` (0.20 m), `camera`, `rotation`, `max_drift`,
  `check_reach` (true), `reach_tolerance` (0.015 m).
- `arm_id`, `tolerance`, `max_steps`: passed on only when set (`arm_id`
  when nonzero).

## Required end states

- `refined`: `refined_point` is the back-projected pixel (a surface point,
  so its z is the top the camera saw); `refined_obb` is `target_obb` with its
  centre moved by the correction in x and y only (its z, size and rotation
  are kept). It is meaningful when `location` names the object's centre.
  `drift` is the planar distance from the coarse point.
- `not_seen`: as above; `refined_point` holds the rejected point when one
  was back-projected.
- `unreachable`: nothing was refined; `refined_point` is the coarse point.

The skill does not move the hand back: the next skill moves from the view
pose (returning-to-a-recorded-pose can, if the graph needs the old pose).

## The constants, and why

- **`CAMERA_RANGE = 0.20` m** is RoboRSI's `hover_height_m` default. Its
  fallbacks stepped down to 0.10 m for an 18 cm finger offset; a D405 reads
  nothing under ~7 cm (`MIN_DEPTH`) and is best inside 50 cm (`MAX_DEPTH`),
  so the fallbacks here stay in 0.15-0.30 m.
- **`MIN_TOOL_CLEARANCE = 0.05` m**: open linear_4310 fingertips reach about
  10 mm below the tool centre; this keeps them and the gripper body clear.
  It rules out the 0.15 m look-down view on a YAM (the hand tilted 45.5 deg,
  the tool centre would be 36 mm over the target's top; 86 mm at 0.20 m).
- **`REACH_TOLERANCE = 0.015` m**: above a real YAM station's 12 mm per-move
  bound. The view need not be exact: the frame is back-projected with the
  camera pose the arm actually reached.
- **Settling (`SETTLE_DISTANCE = 0.5` mm, `SETTLE_ANGLE_DEG = 0.1`)**: a real
  station reads the arm, then the camera frame; a hand still moving between
  the two poses the frame wrong. At 0.2 m on a D405 each bound is about 1 px.
- **`DRIFT_MARGIN = 0.03` m** over the target's half-diagonal: an overhead
  estimate on a YAM station is off by 1-2 cm; a neighbour object further
  than that is not the target.
- **The pointed pixel is not undistorted**: the station's left D405 has
  k1 = -0.054, which moves a pixel near the image centre by well under a
  pixel, and the view keeps the target near the centre.

## Measured

YAM station twin in RoboSimStudio (left arm, 30 mm cube), with a wrist camera
placed by the real station's calibrated left-D405 camera-to-tool transform at
its 640 x 480 intrinsics, path-traced RGB-D. The coarse box was the true box
shifted by 0, 25, 27 and 29 mm in the plane. Each trial reached the look-down
view at 0.20 m on the first move (camera axis within 4.5 deg of straight
down); the camera-to-tool transform read from the observation matched the
real calibration to 0.03 mm.

| pointer | coarse error (mm) | refined error in the plane (mm) | z error (mm) |
|---|---|---|---|
| projected truth (geometry floor) | 0 / 25.0 / 26.9 / 29.2 | 1.05 / 0.20 / 0.06 / 1.07 | <= 0.03 |
| gemini-er.point, live gemini-robotics-er-2-preview | 0 / 25.0 / 26.9 / 29.2 | 0.59 / 0.70 / 1.04 / 0.56 | <= 0.02 |

8/8 `refined`. The ~1 mm floor trials are the harness's: the true top is
projected with the camera pose read a moment after the frame, and the
twin's arm still drifts by a pixel or two. An earlier run without the
settle wait in step 5 gave 1.73 / 0.24 / 0.07 / 1.61 mm (truth) and
0.16 / 0.69 / 0.55 / 1.86 mm (gemini), also 8/8. On a real D405 frame from the
station's folded rest pose, live gemini-er.point put "the centre of the top
face of the grey cube" on the cube's top face 3/3 times (2-5 s a call) and
returned no point 3/3 times for a red cube that was not there (the
`not_seen` path). Not measured: a closed loop on the real arm, molmo.

## Provenance

Adapted from RoboRSI's `find_object_via_wrist/robotwin` policy (Apache-2.0):
the coarse estimate as an input instead of a head-camera step; the camera
aimed rather than the tool, with the extrinsic read from the observation;
IK-previewed candidate views judged from a fresh pose and frame; hub pointing
tools; a nearest-to-coarse point choice; a windowed depth median in the
D405's range; a drift bound; routed exits instead of `ok=False`.
