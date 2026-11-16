---
name: locating-a-container-from-the-wrist-camera
description: Locates an open box, bin or tray of known size with the hand's own
  wrist depth camera before anything is placed into it - moves the empty hand
  to a view short of the container (as it will be turned when it lets go, or
  pointing down from over the container's near half), slides the container's
  outline over the rim points that camera sees near the expected place, and
  returns the container in that camera's frame plus the shift from where
  another camera had it. Because the wrist camera rides on the arm, the arm's
  own pose error cancels - a drop aimed at the located box lands where the hand
  sees it, not where an overhead calibration says it is. Use when a held part
  will be released into a container whose box came from an overhead or other
  fixed camera and the arm is only accurate to a few centimetres, with the
  hand still empty (a held part hides the view) and the container's
  OrientedBoundingBox in hand; add the returned shift to every drop point.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, container, box, bin, wrist-camera, depth, placing, calibration, cpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tuned_on: [yam_i2rt]
    tasks: [pick-place, sorting]
    provenance:
      authors: [RoboSkillHub contributors]
    notes: Measured on one YAM station in eight real runs; no simulator suite, no
      second robot. The view search asks robot.solve_ik at most max_ik_queries times.
gap:
  requires: {connector: [robot.grasp_frame]}
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.solve_ik
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.grasp_frame
  required_inputs:
    container_obb: OrientedBoundingBox
  # Optional inputs: camera (string), move (bool), clear_z, max_shift, min_coverage, min_range
  # (float), max_ik_queries, stride, arm_id (int). Wired from a script node (gap has no type for
  # them): other_obbs (list of OrientedBoundingBox), rotations (list of Quaternion), view_poses
  # (list of Se3Pose), camera_frame (CameraFrame).
  produces_outputs:
    located_obb: OrientedBoundingBox
    aim_point: Vec3
    shift: Vec3
    shift_m: float
    yaw_deg: float
    coverage: float
    views_tried: int
    ik_queries: int
    view_pose: Se3Pose
  exit_conditions:
    located: From a view the hand reached, at least min_coverage (0.30) of the container's outline lay on rim points the wrist camera saw, within max_shift (10 cm) of the expected place; located_obb is the container in the wrist camera's frame, shift what to add to anything aimed from container_obb, aim_point the located centre on the rim plane. The hand stays at the view, empty.
    not_seen: At least one view was looked from (or the given frame was read) and none showed enough of the rim near the expected place, or the only fit lay further than max_shift from it, or the observation has no posed wrist depth camera; `detail` says which. The hand stays where it last looked. Placing may go on uncorrected only at the caller's risk - see Evidence.
    unreachable: No view pose passed robot.solve_ik within max_ik_queries previews (nothing moved), or every view move was refused; `detail` says which.
  canonical_scripts:
    - locate_container: scripts/locate_container.py
  examples:
    - title: Locate the target box from the wrist before the first pick
      path: examples/canonical_subgraph.json
---

# locating-a-container-from-the-wrist-camera

An overhead camera says where the box is in the world. The arm then goes to
where its joints say that is. On a real station those two differ by
centimetres, and not by the same amount everywhere in the workspace. The
wrist camera is bolted to the hand, so whatever it sees is placed with the
same error the hand will have when it moves there: locate the box with it,
aim at that, and the error drops out.

## How

1. **Views** (skipped with `camera_frame`, `move=False` or `view_poses`).
   The views stand on the side the hand starts from, short of the container's
   centre: with each rotation in `rotations` (the hand as it will be turned
   when it lets go) at 14, 20 and 8 cm short, 33 cm down to 12 cm over the
   rim in 3 cm steps; then with the hand pointing down
   (`robot.grasp_frame`) at 5, 9 and 0 cm short, 23 cm down to 8 cm over the
   rim. Per standoff the first rotation and height that passes
   `robot.solve_ik` is kept.
2. **Go and look.** A hand under `clear_z` (default 11 cm over the rim) rises
   straight up first. At the view the wrist frame is read; depth nearer than
   `min_range` (16 cm: the hand's own fingers) is dropped.
3. **Fit.** Points within 30 cm of the expected centre, from 22 mm under to
   12 mm over the rim height and outside `other_obbs` (grown 12 mm), are
   binned at 4 mm; the container's outline is slid over them at turns within
   12 degrees of the expected one (the fit of
   `finding-boxes-of-known-size-from-overhead-depth`). At least 30 % of the
   outline covered, and a centre within `max_shift` of the expected one, is
   `located`; otherwise the next view is tried.

The picture is placed by where the hand **is** (the camera's pose in the
observation), so a view move that ends short still gives a valid look.

## Using the result

`shift` is `located - expected` in the plane. Add it to every point planned
from `container_obb` (drop points, the test "is the hand over the box"), or
plan from `located_obb` directly. Take the look near where the hand will let
go and turned as it will be then: the arm's error changes across the
workspace, so a look from far away corrects less.

Run it **before the first pick**. A held part hangs in front of the lens.

## Search cost

Each candidate view costs one `robot.solve_ik`. The search is capped at
`max_ik_queries` (30) and says so in `detail` when it stops there. The cap is
a guard, not a measurement: on the source station, a pose search that sent
well over a hundred plan requests in a row to the arm's motion server held up
its control loop until the station's supervisor stopped both arms. If your
connector's `solve_ik` is local and cheap, raise it; if it reaches a motion
server, search with `arm-clearance.reachable` instead and pass the result as
`view_poses`.

## Constants, and what they were measured on

One YAM station (i2rt YAM arm with the linear_4310 hand and a wrist D405,
an overhead depth camera), a 238 x 196 mm cardboard box with a 66 mm rim
beside a 154 mm tall one, 2026-11-15.

- `MIN_COVERAGE = 0.30`, `BAND_BELOW / BAND_ABOVE = 22 / 12 mm`,
  `SEARCH_RADIUS = 0.30 m`, `YAW_WINDOW_DEG = 12`, `EXCLUDE_MARGIN = 12 mm`:
  chosen on that box, not swept.
- `MAX_SHIFT = 0.10 m`: a guard over the shifts seen (below).
- `MIN_RANGE = 0.16 m`: that hand's fingertips in that camera.
- Standoffs and heights: the station's values. The leaning views were found
  unreachable when the two boxes were swapped; the hand-down views were added
  for that layout and worked in the next run.
- `CLEAR_OVER_RIM = 0.11 m`, `MAX_IK_QUERIES = 30`: chosen.

## Evidence

Real robot only, one station, 2026-11-15: seven runs placed with the box
located this way, and one more had the look in its graph but found no view.

- The wrist camera put the box 27 mm (19, 19), 72 mm (48, 53) and 78 mm
  (67, 39) from the overhead camera's place for it in the three runs whose
  notes record the shift.
- In the run without a view (the two boxes swapped, before the hand-down
  views existed) the drops used the overhead position
  uncorrected, and all four parts went down on the counter just outside the
  box's near wall.
- Four of the seven located runs put all their parts into the box, all four
  with the boxes as first set out. With the boxes swapped the look located
  the box in the later runs (two of the shifts above are from them) and the
  runs' own logs count one and two parts placed in two of them, but none
  finished, for reasons outside this skill (no carry pose clear of a cabinet,
  a motion-server stop, a refused release).
- One earlier form of the fit, a rectangle put round whatever rim was seen,
  placed a box seen at a slant 3.7 cm off; the slid outline replaced it.

What this does not show: the error left after the correction was never
measured (only that corrected drops went in), nothing was run in a simulator
suite, and no other arm, hand, camera or container was tried. Maturity is
`community`.

## Required end states

- `located`: `located_obb`, `shift`, `aim_point`; the hand at `view_pose`.
- `not_seen`: no correction; `coverage` and `detail` say what was seen.
- `unreachable`: no view; nothing learned.
