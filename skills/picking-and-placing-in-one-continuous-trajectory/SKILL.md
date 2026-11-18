---
name: picking-and-placing-in-one-continuous-trajectory
description: Picks a box-shaped object top-down and places it at a point as ONE timed joint trajectory
  (pre-grasp, grasp, lift, carry, pre-place, place, retreat, home) with straight segments only near contact,
  blended corners, a 2.5 cm hover and stops only to close and release the jaws, timed for the shortest
  duration under joint velocity, acceleration and jerk limits and run as a single robot.execute_trajectory
  with the gripper close/open timed inside it; also returns the joint-stream tracks for the real server.
  Use when pick-and-place cycle time matters and the scene around the object is clear.
license: Apache-2.0
compatibility: requires gap>=0.1; robot.execute_trajectory with waypoint_dt_s, gripper_close_s, gripper_release_s
metadata:
  category: motion
  tags: [motion, pick-place, time-optimal, trajectory]
  hub:
    stage: execute
    embodiments: [yam]
    version: 0.1.0
    tasks: [pick-place]
    provenance:
      authors: [runtime-si]
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.grasp_frame
    - robot.solve_ik
    - robot.open_gripper
    - robot.execute_trajectory
    - robot.get_gripper
    - robot.go_to_pose
  required_inputs:
    object_obb: OrientedBoundingBox
    place: Point3
  produces_outputs:
    duration_s: float
    max_tracking_error_rad: float
    tracks: "stream_joints form: {hz, mode, chunks: [{arm: [[j1..j6 hardware, grip_open], ...]}], max_row_jump_rad, row_jump_ok}"
    waypoints: "12 timed joint waypoints with gripper_open, for move connectors"
  exit_conditions:
    placed: The trajectory ran and the hand ended within reach_tolerance_m (10 mm) of where it started.
    grasp_missed: Only with verify_grasp=true (two calls split at the closing stop) - the jaws closed on nothing.
    not_reached: The hand ended further than reach_tolerance_m from its start (blocked or lagging).
    planning_failed: No grasp rotation reaches every via-point by IK without a joint flip on the straight segments.
  canonical_scripts:
    - pick_place_traj: scripts/pick_place_traj.py
---

# picking-and-placing-in-one-continuous-trajectory

The stepwise pick-and-place makes ~8 separate moves, each accelerating from and stopping at rest, with a
10 cm hover. This skill makes the same motion one trajectory.

## Path

- Via-points by IK (grasp rotation from `robot.grasp_frame`, flipped if the first does not reach):
  home, 5 points on the vertical line from 2.5 cm above the grasp down to it, the same line up, a carry apex
  6 cm above the higher of grasp and place halfway between them, the line down to the place, the line up.
- Straight-line points are kept exactly; joint-space corners (home, apex, the line ends) are blended by
  Gaussian smoothing in joint space (radius 0.08 rad), so the arm passes them without stopping.
- Three moving pieces, each at rest at its ends: home -> grasp, grasp -> place, place -> home, with a
  0.35 s stop to close and a 0.2 s stop to release in between - the only stops.

## Timing

Each piece is timed by a forward/backward pass on the path parameter under per-joint |v| and |a|, then
slowed uniformly until |jerk| (third difference of the 30 Hz rows) holds; lower velocity caps are also
tried and the quickest kept (same code as planning-time-optimal-dual-arm-motions-with-curobo). Defaults:
v 1.5/1.5/1.5/4.5/4.5/4.5 rad/s, a 15/15/15/40/40/40 rad/s^2, jerk 10 x a.

Ruckig was considered: the MIT community version (0.19) has no intermediate waypoints (Pro only), so it
cannot time a blended multi-via-point path in one go.

## Execution

One `robot.execute_trajectory(trajectory, waypoint_dt_s=0.01, gripper_close_s=..., gripper_release_s=...)`.
`gripper_close_s` is new (robosimstudio dual-servo copy, beside `gripper_release_s`). With one call there
is no mid-trajectory grasp check; `verify_grasp=true` splits at the closing stop to read the gripper.

`tracks` hold the same motion at 30 Hz in hardware angles (joint 1 + pi/2) with the gripper column, in
`stream_joints` chunks of <= 64 rows; `row_jump_ok` checks the server's 0.35 rad per-row limit.
