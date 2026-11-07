---
name: centering-a-part-with-the-gripper
description: Centres a loose part between the jaws without perceiving it - closes
  on it, opens, turns the wrist a quarter turn about the hand's approach axis,
  closes again (each close squares the part up along one axis), opens and turns
  back; fails when the jaw gap after a close says nothing is between the pads.
  Use when the hand already stands around a small part whose estimated position
  is a few millimetres off (before a pick that must be centred, an insertion of
  a plug or peg, or threading a ring held in the fingers), the part is free to
  slide on its support, and the hand is open or empty.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [centering, alignment, gripper, recovery, nonprehensile]
  hub:
    stage: recover
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, insertion]
    provenance:
      authors: [runtime-si]
      code: https://github.com/o2ac/o2ac-ur/blob/main/catkin_ws/src/o2ac_routines/src/o2ac_routines/common.py
      upstream_license: MIT
gap:
  requires: {connector: [robot.describe_gripper]}
  allowed_tools:
    - robot.close_gripper
    - robot.open_gripper
    - robot.get_gripper
    - robot.get_ee_pose
    - robot.go_to_pose_cartesian
    - robot.get_observation
    - robot.move_to_joints
    - robot.describe_gripper
  required_inputs: {}
  produces_outputs:
    first_width: float
    second_width: float
    turned_deg: float
    turned_by: string
    returned: bool
  exit_conditions:
    centered: Both closes ended on something (the jaw gap above the closed stop by more than min_held_width, or unread), the wrist turn between them arrived, and the hand opened again (kept closed with release=false); `returned` says whether it turned back.
    nothing_caught: A close ended within min_held_width of the closed stop, so nothing was between the pads; the hand is open and, after the turn, turned back when return_to_start.
    unreachable: Neither quarter turn arrived, by pose or by the last joint (refused, or the read-back rotation more than rotation_tolerance_deg or the tool centre more than position_tolerance off); the hand is open and was sent back to its start rotation.
  canonical_scripts:
    - center_with_gripper: scripts/center_with_gripper.py
  examples:
    - title: Centre the part under the hand
      path: examples/canonical_subgraph.json
---

# centering-a-part-with-the-gripper

Square a small loose part up between the jaws by touch, with no camera. A
parallel gripper closing on a part pushes it toward the jaws' midline along the
closing axis; turned a quarter turn and closed again it does the same along the
other axis, so after two closes the part sits under the tool centre in both
horizontal directions.

## The motion

1. Close (`robot.close_gripper`). If the measured jaw gap is within
   `min_held_width` of the closed stop, open and exit `nothing_caught`.
2. Open (`robot.open_gripper`).
3. Turn the wrist `turn_deg` (90) about the hand's approach axis, in place,
   with `robot.go_to_pose_cartesian`; judge the turn from a fresh
   `robot.get_ee_pose`. If it does not arrive, try the opposite turn (o2ac
   does the same). If neither does, turn the last joint by the same amounts
   with `robot.move_to_joints` from the joints read at the start
   (`joint_fallback`); either quarter turn counts. If nothing arrives, exit
   `unreachable`.
4. Close again, with the same empty check (open, turn back, `nothing_caught`).
5. Open (skip with `release=false` to keep the part held), and turn back to
   the start rotation (`return_to_start`, default true).

The jaw gap is `robot.close_gripper`'s returned `position` (or
`robot.get_gripper`), 1.0 = open, converted to metres through
`robot.describe_gripper`'s `width_fit` (or `span_m`), as verifying-grasps does.
A hand that cannot be read is treated as holding something: absent evidence is
not a failed centring.

## Inputs (all optional)

- `check_held` (true): the empty-close test after each close.
- `min_held_width` (0.003 m): a close ending within this of the closed stop
  held nothing. Lower it for parts thinner than about 4 mm.
- `turn_deg` (90), `clockwise` (false): the turn tried first.
- `return_to_start` (true), `release` (true), `skip_first_close` (false):
  o2ac's `move_back_to_initial_position` and `skip_first_grasp`; `release`
  is new.
- `joint_fallback` (true): turn the last joint when a pose turn does not
  arrive. An IK that leaves wrist roll free (RoboSimStudio's default on the
  YAM) reaches the turned pose without turning; o2ac's own docstring turns
  "the robot's last joint". A connector that refuses joint moves (the real
  YAM station bridge) just routes the refusal.
- `rotation_tolerance_deg` (10), `position_tolerance` (0.015 m): arrival
  bounds of the turn.
- `arm_id` (0), passed to the connector only when nonzero.

## Preconditions

- The hand already stands around the part: top-down over it, tool centre at
  the part's height, the part inside the open jaws in both directions. The
  skill does not move the hand down or sideways.
- The part slides on its support and is narrower than the open hand. Neighbours
  within the open jaws' sweep (the fingers turn through a quarter circle) get
  pushed too.

## Outputs

`first_width`, `second_width` (m, -1 when not measured), `turned_deg` (the
turn that arrived, signed, as read back), `turned_by` (`pose` or
`last_joint`), `returned`. The part's position is not measured:
the exit says the hand closed on something twice, not where the part ended.
Re-perceive it if that matters.

## The constants, and why

- **`TURN_DEG = 90`**: o2ac's `tau / 4`; two closes square the part along
  two perpendicular axes.
- **`MIN_HELD_WIDTH = 0.003` m**: o2ac's `simple_gripper_check` fails under
  1 mm (6 mm on its imprecise hand); 3 mm is the hub's verifying-grasps
  margin, set from an empty close measured at 1.8 mm.
- **`ROTATION_TOLERANCE_DEG = 10`, `POSITION_TOLERANCE = 0.015` m**: no
  tighter than the connectors' own arrival bounds (a servo's 3 deg / 4 mm, a
  real YAM station's 0.04 rad / 12 mm per move).

## Measured in a YAM station twin

RoboSimStudio's YAM station twin (left arm, linear_4310 hand, a 30 mm cube
on the table), one rollout, default inputs. A setup script stood the open hand
top-down with `robot.grasp_frame`'s rotation, tool centre about 19.7 mm over
the cube's bottom (the descent to 18 mm stopped 1.7 mm short), offset from
the cube's centre; `sim.get_object_obb` measured the cube against
`robot.get_ee_pose` before and after:

| asked offset (mm) | cube off the tool centre before | after | jaw gap at the two closes |
|---|---|---|---|
| +8, 0 | 7.8 mm | 0.22 mm | 31.7 / 31.4 mm |
| 0, +8 | 7.8 mm | 0.09 mm | 31.4 / 31.5 mm |
| +6, -6 | 8.2 mm | 0.17 mm | 31.4 / 31.4 mm |
| -5, +7 | 8.6 mm | 0.39 mm | 31.4 / 31.4 mm |
| +7, +7 | 9.6 mm | 1.11 mm | 31.4 / 31.4 mm |
| -10, 0 | 10.1 mm | 0.72 mm | 31.4 / 31.7 mm (first / second) |

All six exited `centered` and turned back. The cube never tipped (tilt
0.0 deg) and turned at most 6 deg about the vertical. Over empty table (the
hand 120 mm from the cube) the first close read 0.0 mm and the skill exited
`nothing_caught` without turning; the cube did not move.

The twin's IK leaves wrist roll free, so every pose turn "arrived" without
turning the hand (read back 90 deg off): the turns ran by the last joint, and
the turn back likewise. A first rollout without the joint fallback routed
all six `unreachable`; its first close alone had already centred the cube
along the jaws' axis (0, +8 mm: 7.8 to 0.43 mm) and left it 7.8 mm off across
them, which is what the second close is for. On a real YAM station the pose
turn is expected to arrive (its IK solves the full rotation) and
`robot.move_to_joints` is refused; that path has not been run on hardware.

## Provenance

Adapted from o2ac-ur's `center_with_gripper` and `simple_gripper_check`
(`catkin_ws/src/o2ac_routines/src/o2ac_routines/common.py`, MIT, Team O2AC,
World Robot Summit 2020 assembly). Changed: robot.* connector tools instead of
MoveIt and the Robotiq driver; the hand opens fully between closes instead of
to a commanded `opening_width`, because the YAM station offers only open and
close; the empty test converts the jaw gap through `robot.describe_gripper`;
the turn is about the hand's approach axis from its current rotation, judged
from a fresh pose reading; refused moves route instead of raising; `release`
is new. o2ac's gripper force and speed arguments have no counterpart here.
