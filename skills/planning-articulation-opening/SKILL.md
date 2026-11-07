---
name: planning-articulation-opening
description: Plans the hand's path that opens a drawer or a door it already
  holds by the handle - a straight pull for a prismatic joint, or an arc about
  the hinge that turns the hand with the door so its grip on the handle never
  changes - and checks every waypoint with the arm's IK, chained from the
  previous solution, returning the reachable poses and joint trajectory.
  Use when a handle is grasped (or about to be) and its articulation is known
  (pull direction and distance, or hinge point, axis and opening angle), before
  an execution skill moves the arm along the path.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [articulated, drawer, door, cabinet, handle, trajectory, ik]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [articulated]
    provenance:
      authors: [runtime-si]
      code: https://github.com/mosart-system/code/blob/main/modules/motion_planning.py
      upstream_license: MIT
      paper: arXiv:2402.17767
gap:
  allowed_tools:
    - robot.solve_ik
  required_inputs:
    handle_pose: Se3Pose
    joint_type: string
  produces_outputs:
    waypoints: PoseSequence
    all_poses: PoseSequence
    joint_trajectory: Trajectory
    reachable_fraction: float
    opening: float
    planned_opening: float
    radius: float
  exit_conditions:
    planned: Every waypoint from the grasp to the full opening solved (or, with min_fraction under 1, the reachable prefix covers at least that share of the steps); `waypoints` and `joint_trajectory` hold the executable prefix.
    unreachable: The grasp pose itself did not solve, or the reachable prefix falls short of min_fraction; `waypoints` holds whatever prefix did solve (possibly none), `reachable` marks every waypoint, and `detail` names the first failure.
  canonical_scripts:
    - plan_articulation_opening: scripts/plan_articulation_opening.py
  examples:
    - title: Plan a drawer pull from a grasped handle
      path: examples/canonical_subgraph.json
---

# planning-articulation-opening

A hand that holds a handle cannot go where it likes: the drawer runs on its
rails and the door on its hinge, and the hand must follow. This skill writes
that path down as waypoints and asks the arm's IK, one by one, whether it can
follow it. It plans; it does not move.

## The path

Waypoint 0 is the grasp itself (`handle_pose`), as in MOSART.

- **Prismatic** (drawer): `handle_pose` translated along `pull_direction` in
  equal steps to `distance`, the rotation held.
- **Revolute** (door, lid): `handle_pose` rigidly rotated about the hinge line
  (`hinge_point`, `hinge_axis`) in equal steps to `angle_deg`. The position
  stays on the circle of the handle's distance from the line (`radius`), at
  the same height along the axis, and the hand's rotation turns by the same
  angle, so the hand is fixed in the door's frame: MOSART's quarter circle
  with its matching yaw sweep, for any axis and any angle. The door turns
  right-handed about `hinge_axis` unless `pull_direction` is given, in which
  case the axis sign is chosen so the handle starts moving along it (pass the
  direction the door swings toward you and the sign takes care of itself).

Steps are MOSART's sizes, 35 mm or 10 deg (`num_waypoints` overrides): a
15 cm pull has 5 steps, a 60 deg door 6.

## The IK chain

Each waypoint goes to `robot.solve_ik`. A waypoint fails when the call raises
(a real station raises on an unreachable pose), replies `success: false` or
`within_servo_tolerance: false`, or lands more than 20 mm off. As in MOSART:

- an unreachable grasp fails the whole plan at once;
- otherwise every waypoint is tried even after a failure, chaining from the
  last solution that worked, so `reachable` shows the whole picture (a graph
  can open partway, regrasp, and continue).

MOSART warm-starts each solve from the previous solution. Neither the YAM
station's nor its twin's `robot.solve_ik` takes a seed, so seeding is opt-in:
`seed_argument="seed_joints"` (or whatever the connector calls it) passes the
previous joints, and a connector that refuses the keyword is asked again
without it and not seeded afterwards (`seeded` says which happened). What
keeps an unseeded chain honest instead is a **branch-jump gate**: a solution
more than `max_joint_step` (0.8 rad) from the previous one on any joint fails
its waypoint, since a hand on a handle cannot swing through a wrist flip.

The executable plan (`waypoints`, `joint_trajectory`) is the reachable prefix:
the hand cannot skip a waypoint it cannot reach while holding a handle.
MOSART returned every solved waypoint, gaps included.

## Inputs

- `handle_pose`: the hand's pose on the handle, world frame.
- `joint_type`: `"prismatic"` or `"revolute"`.
- `pull_direction` (drawer: required; door: optional sign hint), `distance`
  (0.35 m, MOSART's), `hinge_point`, `hinge_axis`, `angle_deg` (90, MOSART's).
- `num_waypoints`, `min_fraction` (1.0), `seed_argument` (""),
  `max_joint_step` (0.8 rad), `arm_id` (0).

## Outputs

- `waypoints`: a `PoseSequence` of the reachable prefix (`planned_linear`,
  `cartesian: true`), ready for an executor that steps through poses.
- `all_poses`: the full plan as a `PoseSequence`, reachable or not.
- `joint_trajectory`: `{waypoints: [{positions}]}` for the prefix, from the
  IK replies (empty if the connector returns no joints).
- `reachable` (a list of booleans, one per waypoint; returned but not
  declared, as gap has no list type), `reachable_fraction` (share of the steps after the grasp that
  solved, MOSART's `success_idxs`), `opening` (m or rad reached by the
  prefix), `planned_opening`, `radius` (0 for a drawer), `seeded`, `detail`.

## Required end states

- `planned`: execute `waypoints` in order, slowly, with the gripper closed on
  the handle.
- `unreachable`: reposition (the base, or a better grasp) and plan again, or
  execute the prefix and regrasp.

## What this does not know

The plan is the hand's kinematics only. It checks no collisions, assumes the
articulation estimate is right, and cannot tell the drawer's travel from the
asked `distance`. On a real YAM, without a force sensor, a stuck drawer shows
up during execution as tracking lag or joint effort, not here.

## The constants, and why

- **`PRISMATIC_STEP = 35 mm`, `REVOLUTE_STEP_DEG = 10`**: MOSART's 35 cm pull
  in 10 steps and quarter turn in 9.
- **`DEFAULT_DISTANCE = 0.35 m`, `DEFAULT_ANGLE_DEG = 90`**: MOSART's, set for
  full-size kitchen cabinets and a mobile Stretch. A fixed YAM reaches far
  less; pass the distance and angle you need.
- **`POSITION_TOLERANCE = 20 mm`**: the hub's reachability gate
  (selecting-reachable-grasp).
- **`MAX_JOINT_STEP = 0.8 rad`**: a 35 mm or 10 deg step moves no joint more
  than a few tenths of a radian; a wrist branch flip is about pi.
- **`MIN_RADIUS = 10 mm`**: a handle on the hinge line has no arc to follow.

## Checked in simulation

RoboSimStudio, a YAM station twin, left arm, empty hand. No articulated
object exists in the scene: the handle is a point 10 cm above the 30 mm cube
(tool centre at (0.611, 0.276, 0.867)), chosen as the first anchor whose whole
path solved. A front-on grasp (approach +x, horizontal) did not solve there;
both plans used the hand approaching +x tilted 45 deg down. Pull and door
swing toward the arm's base (-x); the hinge is vertical, 12 cm to the
handle's +y side. The skill planned both (`planned`, every waypoint
reachable); each waypoint was then commanded with `robot.go_to_pose` and the
hand read back with `robot.get_ee_pose`:

| plan | waypoints | position error per waypoint (mm) | approach-axis error, max | full rotation error per waypoint (deg) | outcome |
|---|---|---|---|---|---|
| drawer, 15 cm | 6 | 2.9, 3.2, 3.4, 3.4, 2.3, 2.7 | 1.2 deg | 5.4, 6.3, 7.4, 8.9, 11.1, 14.8 | pulled 147.5 mm |
| door, 60 deg, r = 120 mm | 7 | 2.0, 2.0, 4.0, 2.2, 3.1, 2.0, 2.1 | 1.3 deg | 5.5, 5.3, 17.9, 34.2, 52.9, 74.2, 96.0 | 59.1 deg, radius within 1.8 mm |
| drawer, roll finished | 6 | 3.2, 2.5, 3.0, 2.1, 2.1, 1.9 | 1.1 deg | 1.1, 1.2, 2.1, 0.6, 0.6, 0.3 | pulled 148.3 mm |
| door, roll finished | 7 | 1.6, 1.6, 2.0, 2.3, 2.2, 1.8, 1.7 | 1.5 deg | 0.1, 0.3, 1.6, 0.7, 0.9, 0.4, 0.5 | 59.2 deg, radius within 1.3 mm |

The position path is tracked within 4 mm and the approach axis within
1.5 deg. The rotation error of the first two rows is almost all wrist roll
about the approach axis: the twin's IK constrains only the approach axis
(hub porting lesson 10), so `go_to_pose` reports `reached` with the roll
unturned, and on the door it falls a full 96 deg behind by the end - a hand
on a real handle would be wrenched. The same twin's `robot.solve_ik` also
ignores roll, so this skill's IK check cannot see it either. Finishing the
roll on the last joint after each waypoint (as stacking-objects does: read
the roll back, turn `robot.move_to_joints`' last joint, re-centre) brings
every waypoint within 2.1 deg, which is the hand fixed to the handle the plan
asks for. **An executor running these waypoints on a connector whose IK
leaves the roll free must finish the roll itself**; a full-pose arm (the real
YAM station solves the whole quaternion) arrives turned.

## Provenance

Adapted from MOSART (Gupta et al., "Opening Articulated Structures in the Real
World", arXiv:2402.17767), `modules/motion_planning.py`
(`get_motion_plan` and its SeqIK loop), MIT. Changed: world-frame handle pose
and articulation instead of a fixed handle frame and a left / right hinge
class; the arc generalised to any hinge axis and angle; step sizes instead of
fixed counts; the Stretch whole-body Pinocchio IK replaced by
`robot.solve_ik`, seeded only where the connector takes a seed, with a
branch-jump gate; the executable plan cut at the first failure.
