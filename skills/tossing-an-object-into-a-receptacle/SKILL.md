---
name: tossing-an-object-into-a-receptacle
description: Put a held object into a receptacle by dropping it from above the opening when the arm reaches it, or by tossing it when the opening is beyond reach - an analytic ballistic plan picks the launch angle and release speed, a straight IK run-up through the release point delivers it, and the joint speed that plan needs is checked against the arm's limit before anything moves (refusing with the speed it would need). Use when a bin or box sits past the arm's reach, or when dropping into it is faster than placing.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [toss, throw, drop, ballistic, receptacle, pick-place]
  hub:
    stage: execute
    embodiments: [any]
    tasks: [pick-place]
    version: 0.1.0
    provenance:
      code: https://github.com/jimmyyhwu/tidybot/blob/c060278de506699b46ff7361357109cf9dcd5654/robot/kinova.py
      upstream_license: MIT
gap:
  requires: {connector: [robot.solve_ik, robot.describe_gripper]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.solve_ik
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - robot.execute_trajectory
    - robot.open_gripper
    - robot.get_gripper
    - robot.describe_gripper
  required_inputs:
    opening_center: Vec3
    opening_radius_m: float
  produces_outputs:
    mode: string
    launch_deg: float
    release_speed_mps: float
    required_joint_speed: float
    detail: string
  exit_conditions:
    landed: The locate tool found the object within opening_radius_m of the opening's centre and below its rim.
    missed: The object was found elsewhere, or not found, after the release.
    not_feasible: No plan within the limits - the opening is out of reach for a drop and the toss needs more joint speed than joint_speed_limit (detail says how much), or no IK through the release point or run-up. Nothing moved.
    dropped_early: The jaws were on the closed stop before the release - the object was already gone.
    released: The object was let go but no locate_tool was given to read where it landed.
  canonical_scripts:
    - toss_object: scripts/toss_object.py
  streaming: false
---

# tossing-an-object-into-a-receptacle

TidyBot (Wu et al. 2023) puts objects into far bins with a fixed "toss" primitive on its
Kinova Gen3: a pre-swing pose, then a joint-space swing at up to 140 deg/s with the
gripper released 600 ms in (`robot/kinova.py`, `KinovaArm.toss`), and `place` when
the receptacle is close (`server/demo.py`). Here the fixed swing becomes a plan for any
6-DoF arm.

## Plan

1. **Drop first.** If the arm has IK `DROP_CLEARANCE` (8 cm) over the opening, go there
   and open (`mode: drop`), unless `toss_only`.
2. **Release point.** `release_advance_m` (10 cm) from the hand toward the opening, at
   the hand's height; d is the horizontal distance to the opening's centre, h the height
   over it.
3. **Launch.** For each angle in 20-60 deg the speed that lands on the centre is
   `v^2 = g d^2 / (2 cos^2(theta) (d tan(theta) + h))`. The joint speed it needs is the
   finite difference of IK over the last 20 ms of the run-up; the angle that needs the
   least wins.
4. **Refuse or go.** If that joint speed is over `joint_speed_limit`, route
   `not_feasible` with the numbers. Otherwise the run-up is a straight line at constant
   acceleration `accel` ending at the release point with speed v, streamed as 20 ms
   waypoints with `robot.execute_trajectory`, then `robot.open_gripper`.
5. **Read the landing** with `locate_tool` (`position`, `center` or `obb.center`).

## YAM speed limits

| source | joint speed |
| --- | --- |
| real station motion server default (`--stream-max-vel`, 0.25 x `--speed`) | 0.25 rad/s |
| real station fastest (`--speed 6`) | 1.5 rad/s |
| YAM URDF motor limit | 10 rad/s |

`joint_speed_limit` defaults to 0.25 rad/s, so on the station as it ships every toss
beyond reach routes `not_feasible`, and `detail` names the speed it would need. Raise it
only for a station started with a matching `--speed`.

Release speeds the plan asks for (h = 0.2 m over the rim, best angle):

| d past release | v |
| --- | --- |
| 0.2 m | 0.9 m/s |
| 0.3 m | 1.3 m/s |
| 0.4 m | 1.6 m/s |

With about 0.5 m from the shoulder to the hand, these speeds work out to roughly
2-3 rad/s. That is above even the station's 1.5 rad/s.

## Limits

- **Release timing in streamed connectors.** The gripper opens after
  `robot.execute_trajectory` returns. With `tolerance=0.5` the stream does not settle
  first, but the arm is already decelerating toward its last waypoint while the jaws
  open, so the actual release speed is lower than planned. A connector with a timed
  gripper event inside the trajectory (as TidyBot's `gripper_release_ms`) would remove
  that.
- No spin or drag model; the object is a point.
- Not yet measured in the twin (see the PR).
