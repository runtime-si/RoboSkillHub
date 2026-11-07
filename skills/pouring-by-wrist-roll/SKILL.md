---
name: pouring-by-wrist-roll
description: Pours from a held container into a target container by standing
  the hand above the target's box at a set height and rolling only the last
  joint (the wrist roll) to the pour angle in small speed-bounded steps,
  holding, and rolling back, checking every step from fresh joint readings.
  Use when a cup, bottle, scoop or bin is already in the hand and its contents
  must go into a bowl, box or other receiver whose box a perception skill
  found, and the grasp left the wrist-roll axis near horizontal (the roll then
  tips the container rather than spinning it).
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [pour, wrist-roll, held-object, household]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [pouring]
    provenance:
      authors: [runtime-si]
      code: https://github.com/OpenMOSS/VLABench/blob/main/VLABench/utils/skill_lib.py
      upstream_license: MIT
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
  required_inputs:
    target_obb: OrientedBoundingBox
  produces_outputs:
    roll_reached: float
    hand_drift: float
    final_joint_error: float
    roll_axis_elevation_deg: float
    rolled_with: str
  exit_conditions:
    poured: The hand reached the pour pose, the last joint followed every step to the full pour angle within track_tolerance with the tool centre within drift_tolerance of the pour pose, held hold_s, and rolled back with every joint within 0.08 rad of its start. This is what the hand did; re-perceive the receiver to know what landed in it.
    unreachable: The move to the pour pose above the target was refused or ended more than reach_tolerance from it; nothing was rolled.
    joint_limit: The asked roll does not fit between the last joint's reading and its limits (less a 0.05 rad margin) in the chosen direction; the hand stands at the pour pose, unrolled.
    roll_refused: A roll step was refused, the last joint lagged its step's target by more than track_tolerance, the tool centre drifted past drift_tolerance, or the roll back failed or ended off; the skill rolled back as far as it could, and `detail` says which.
  canonical_scripts:
    - pour_wrist_roll: scripts/pour_wrist_roll.py
  examples:
    - title: Pour a held container into a perceived receiver
      path: examples/canonical_subgraph.json
---

# pouring-by-wrist-roll

Tip a held container over a receiver by turning only the wrist roll.

## The motion

1. Stand the hand at the pour pose: `pour_height` over the top of
   `target_obb`, over its centre (plus `offset`, to put a spout rather than
   the tool centre over the receiver). The hand keeps its own rotation unless
   `rotation` is given, and rises straight up first when it is below the pour
   height, so the held container is not dragged across a rim.
2. Pick the roll direction (`roll_direction`, or the side of the last joint
   with more room to its limits) and refuse a roll that does not fit.
3. Roll the last joint to `pour_angle_deg` in steps of at most `step_rad`, each
   step taking at least `step_rad / max_speed`, with every other joint
   commanded where it stood. After each step read the joints and the tool
   centre back: the last joint must be within `track_tolerance` of the step's
   target and the tool centre within `drift_tolerance` of the pour pose.
4. Hold `hold_s`, then roll back the same way, and check every joint is within
   0.08 rad of where the roll started.

A refused step, a lag or a drift stops the roll and rolls back from there.

## Inputs

- `target_obb`: the receiver's oriented box, world frame (half-extents).
- `pour_angle_deg` (120): the roll, in degrees, unsigned.
- `pour_height` (0.10 m): tool centre over the box's top.
- `offset` (none): x/y shift of the pour pose from the box's centre.
- `rotation` (none): the hand's rotation at the pour pose; its own otherwise.
- `roll_direction` (0): +1 or -1 to force a direction, 0 for the side with
  more room.
- `roll_limits` (`[-2.0944, 2.0944]`): the last joint's range. The default is
  the i2rt YAM's joint 6 (+-120 deg), narrower than most arms' last joint.
- `roll_with` (`auto`): `joints` rolls with `robot.move_to_joints`; `pose`
  rolls the tool about its own axis with `robot.go_to_pose` at a fixed tool
  centre; `auto` uses joints and falls back to the pose roll when the first
  joint move is refused with the arm unmoved.
- `step_rad` (pi/40), `max_speed` (0.5 rad/s), `hold_s` (1.0 s),
  `track_tolerance` (0.075 rad), `drift_tolerance` (0.02 m),
  `reach_tolerance` (0.02 m): see the constants.
- `arm_id` (0), `tolerance`, `max_steps`: passed to the connector only when
  set (gap's built-in connector's convergence knobs).

## Outputs

`roll_trajectory` and `grip_trajectory` are returned but not listed under
`produces_outputs`: the type registry names no bare list.

- `roll_trajectory`: the last joint's reading [rad] at the start, after every
  step out, at the hold, and after every step back.
- `grip_trajectory`: the gripper reading (`gripper_fraction`) beside each of
  those, -1 where the connector gives none. A held solid leaving the hand
  shows here as the fingers closing on nothing; the skill reports it and does
  not judge it.
- `roll_reached`: the largest measured roll from the start [rad].
- `hand_drift`: the tool centre's largest distance from the pour pose over the
  roll [m].
- `final_joint_error`: the worst joint's distance from its start after the
  roll back [rad], -1 when not read.
- `roll_axis_elevation_deg`: the roll axis (the tool's z) from horizontal at
  the pour pose. Near 0 the roll tips the container; near 90 it spins it, and
  `poured` then says so in `detail`.
- `rolled_with`: `joints` or `pose`.

## Required end states

- `poured`: as the exit says. It reports the hand, not the contents: whether
  anything left the container, and where it went, needs a fresh perception
  of the receiver (verifying-placement, a perception skill).
- `unreachable`, `joint_limit`, `roll_refused`: as the exits say.

## Preconditions

- The container is held. The skill neither opens nor closes the gripper.
- The grasp decides the roll axis. Only a roll axis near horizontal pours: a
  top-down grasp of a cup turns it about the vertical. Grasp the container
  from the side, or pass a `rotation` whose tool z is near horizontal.
- The grasp decides the room: pre-position the wrist so the last joint has
  `pour_angle_deg` of room on one side (on a YAM, a joint 6 near 0 leaves
  exactly 120 deg each way, which the 0.05 rad margin refuses).

## The constants, and why

- **`STEP_RAD = pi/40`**: upstream's `target_q_velocity`, the increment
  `SkillLib.pour` and `SkillLib.flip` step the last joint by.
- **`MAX_SPEED = 0.5` rad/s**: each step takes at least `STEP_RAD / MAX_SPEED`
  (0.16 s). A real YAM station's supervisor commands joints at 0.25 rad/s and
  trips its guard at 0.75 rad/s measured; 0.5 rad/s stays under the guard.
- **`TRACK_TOLERANCE = 0.075` rad**: a real YAM station's single-arm settle
  tolerance. Upstream used 0.01 rad, in a simulator that steps until it
  converges.
- **`RETURN_TOLERANCE = 0.08` rad**: as returning-to-a-recorded-pose, just
  above that settle tolerance.
- **`POUR_ANGLE_DEG = 120`**: upstream's default is pi; a YAM's joint 6 spans
  +-120 deg, so a half turn only fits from a start past 60 deg the other way.
- **`POUR_HEIGHT = 0.10` m**: in the twin below it held the cube's bottom
  about 90 mm over the box's rim.
- **`DRIFT_TOLERANCE = 0.02` m**: the twin's tool centre moved under 1 mm over
  a 120 deg roll, so 20 mm only catches a roll that is not a wrist roll (an
  IK that moved the other joints) or a hand pushed off by a collision.
- **`REACH_TOLERANCE = 0.02` m**: as pushing-objects, loose enough for a servo's
  4 mm and a real YAM's 12 mm per move.
- **`HOLD_S = 1.0` s, `LIMIT_MARGIN = 0.05` rad**: not tuned; a viscous pour
  wants a longer hold.

## Simulation check

In a YAM station twin (left arm, linear_4310 hand), the 30 mm cube as a
stand-in container: a scripted top-down grasp, carried 15 cm up and over the
open 100 x 100 x 40 mm box, the target the box's floor region as an OBB, the
skill at its defaults with a 120 deg pour. The twin's servo leaves the wrist
roll free (all eight grasp headings tried landed joint 6 at 0.001 rad), and
joint 6 near 0 leaves exactly 120 deg each way, which the limit margin
refuses; so the test graph turned joint 6 a quarter turn (to -1.570 rad) by
joint command before descending, which keeps the jaws square to the cube.

| run | roll axis from horizontal | route | joint 6 reached | hand drift | final joint error | the cube |
|---|---|---|---|---|---|---|
| top-down hand, as grasped | 86 deg | `poured` (detail: spun) | -1.570 -> +0.524 -> -1.570 rad (120.0 deg) | 0.7 mm | 0.0002 rad | stayed in the hand, 7 mm from the tool centre, 145 mm over the box floor |
| hand turned to a horizontal tool axis over the box | 0.1 deg | `poured` | -1.570 -> +0.525 -> -1.570 rad (120.0 deg) | 0.6 mm | 0.00002 rad | fell out between 49 and 53 deg of roll (the grip reading dropped from 0.33 to 0.05) and landed in the box |
| as above, `roll_with: pose` | 0.1 deg | `roll_refused` | 0.0001 rad | 1.9 mm | 0.010 rad | stayed in the hand |

- Each step moved joint 6 by 0.0776 rad (2.0944 / 27) on the reading; every
  other joint was commanded where it stood.
- A top-down hand only spins the container: the cube turned 120 deg about the
  vertical and stayed held. The skill still routes `poured` (the hand did what
  was asked) and says so in `detail`.
- With the tool axis horizontal the 30 mm cube, held by friction between
  the pads, slid out at about 50 deg of roll. It had already shifted 23 mm off
  the tool centre when the hand turned horizontal, before the roll began. A
  held solid falling out of the hand is what a pour of a solid does, and the
  grip reading (`grip_trajectory`) is where the skill sees it.
- The pose roll did not turn joint 6 in the twin because its servo leaves the
  roll free; the skill caught it on the first step, rolled back and routed
  `roll_refused`. A real station's IK keeps the full rotation (below), but
  this path has not been run on one.
- Speed: the 0.5 rad/s bound is enforced on the wall clock (each step waits
  out `step_rad / max_speed`), which bounds a real arm. In the twin each step
  is the simulator's own converge-and-settle move; its peak joint speed was
  not measured.

## On a real YAM station

The station's bridge refuses `robot.move_to_joints` (it takes poses only), so
`roll_with: auto` falls back to the pose roll: the tool turned about its own
axis at a fixed tool centre, which the station's bounded IK turns into the
same joint-6 roll when the tool centre lies on that axis. The skill learns the
sign from the first step's joint reading. The station paces each step at its
own 0.25 rad/s.

## Provenance

Adapted from VLABench's `SkillLib.pour` and `SkillLib.flip` (MIT): the same
stepped roll of only the last joint by pi/40 per step with a per-step tracking
tolerance. Added: the pour pose above a world-frame target box, connector
tools instead of the simulator's `env.step`, a speed bound, the direction
chosen against the joint limits, the hold and the roll back, the drift check,
the pose roll for connectors without joint moves, and routed refusals.
