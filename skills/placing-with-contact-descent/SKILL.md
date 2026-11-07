---
name: placing-with-contact-descent
description: Places the object the hand holds by approaching a fixed height
  above its place point, coming straight down until the arm stops on contact,
  and releasing there before retreating straight up - Ravens' pick-place
  discipline, with the arm's stall detection standing in for a contact sensor.
  Contact well above the expected height (stacked on the wrong thing, a rim)
  or no contact at all is routed and the object kept. Use when a held object
  must be set down on a support or into an open container whose height is
  known only to a centimetre or so (a camera's estimate of a table, a box
  floor, the top of a stack), with its box measured after the grasp.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [place, contact, release, stacking]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/google-research/ravens/blob/master/ravens/tasks/primitives.py
      upstream_license: Apache-2.0
      paper: arXiv:2010.14406
gap:
  requires: {connector: [robot.move_cartesian_until_contact]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_cartesian_until_contact
    - robot.solve_ik
    - robot.open_gripper
  required_inputs:
    place_position: Vec3
  produces_outputs:
    released: bool
    expected_contact_z: float
    contact_z: float
    contact_offset: float
    placed_position: Vec3
  exit_conditions:
    placed: The descent stopped on contact inside the contact window (or, with table_z, ended at the tool-centre floor with the object's bottom at or under the table), the hand opened and retreated straight up; placed_position is where the hand left the object's bottom centre.
    contact_too_early: The descent stopped more than contact_tolerance (12 mm) above the expected contact; the hand did not release and lifted the object back to the approach height.
    no_contact: The descent reached its bottom without stopping (no support there, or the object slid up the jaws); the hand did not release and lifted back.
    blocked: The expected contact is out of reach (IK preview), the approach or a descent move was refused, or the retreat after the release failed (`released` says which side of the release); `detail` says which.
  canonical_scripts:
    - place_with_contact_descent: scripts/place_with_contact_descent.py
  examples:
    - title: Place a held object at a point on a support
      path: examples/canonical_subgraph.json
---

# placing-with-contact-descent

Set a held object down without knowing exactly how high the support is: come
down until it touches, then let go.

## The motion

1. Work out where the hand must be. With `held_obb` (the held object's box,
   world frame, measured after the grasp) the object's bottom centre goes to
   `place_position`: the hand goes there shifted by the object's offset in the
   hand, and contact is expected with the tool centre the object's hang (tool
   centre minus the box's lowest corner) over the support. Without it,
   `place_position` is the tool centre at contact (Ravens' place pose).
2. Preview the expected contact pose with `robot.solve_ik` (`check_reach`): a
   stall at the edge of reach reads exactly like contact.
3. Rise straight up if the hand is under the approach height, then go to
   `approach_height` over the expected contact, keeping the hand's rotation.
4. `robot.move_cartesian_until_contact` to the top of the contact window,
   `contact_tolerance` over the expected contact. A stop on the way is
   `contact_too_early`.
5. `robot.move_cartesian_until_contact` on to `overtravel` under the expected
   contact, never under `min_tool_height` over the table (`table_z`) or, without
   it, over the support.
6. A stop on the way is contact: open and retreat `retreat_height` straight up.
   A move that finishes met nothing (`no_contact`), unless `table_z` is given
   and the object's bottom would now be at or under the table, which means it
   rests on something (a hand at the floor pressing the object down).

## Inputs

- `place_position`: where the object's bottom centre goes (with `held_obb`),
  z = the support's height there; or the tool centre at contact (without).
- `held_obb` (optional): the held object's oriented box, half extents.
- `table_z` (optional, recommended): the table plane's height, a station
  calibration. It sets the tool-centre floor and lets a hand that finished at
  that floor count as placed.
- `support_z` (optional): the support's height, for the floor when `table_z`
  is not given; defaults to `place_position.z` with `held_obb`.
- `approach_height` (0.08 m), `overtravel` (0.02 m), `contact_tolerance`
  (0.012 m), `min_tool_height` (0.015 m), `retreat_height` (= approach
  height), `reach_tolerance` (0.015 m): see the constants.
- `check_reach` (true), `arm_id` (0; passed only when nonzero).

## Required end states

- `placed`: released and retreated. This is what the hand did;
  `placed_position` is the object's bottom centre as the hand left it,
  assuming it did not move in the jaws (it may have slid up them, see below).
  Re-perceive it (verifying-placement) before relying on it.
- `contact_too_early`, `no_contact`: still holding, lifted back. Re-perceive
  the support or the object and try again, or recover.
- `blocked`: nothing placed unless `released` is true (then the object is down
  but the hand could not leave it cleanly).

## What changed from Ravens, and why

Ravens' `PickPlace` steps a suction cup down 1 mm per `movep` from 0.32 m
over the place pose until the cup's contact sensor fires, releases, and
returns to 0.32 m. The YAM has no contact sensor, and a real station's moves
settle within 12 mm, so 1 mm steps cannot tell a blocked step from a finished
one. Two contact moves (the arm's own stall detection: tracking progress, no
force sensor) do the descent, and the measured stop is judged against the
height contact was expected at. Ravens' descent is unbounded; here it is
bounded below and floored over the table, and a stop far above the expected
height is routed instead of released.

Ravens' 1 mm steps also keep the cup from pressing far past the contact. A
servo aimed far below the contact presses until a friction-held object slides
up the jaws: in a YAM station twin a 30 mm cube slid 3-4 mm when the move was
aimed 8 mm under the contact and 11-12 mm when aimed 16-28 mm under it, before
the arm stalled. That is why the descent first stops at the top of the
contact window (an early contact is read before the hand has pressed far), why
no lower tolerance is applied to a stop (it reads low by the slide), and why a
finished move at the floor needs `table_z` to count: the slide ends about where
the servo calls a move arrived, so near the floor a pressed object and a free
hand look the same.

## The constants, and why

- **`APPROACH_HEIGHT = 0.08` m**: Ravens' 0.32 m suits a UR5; 0.08 m clears a
  40 mm box wall with a 30 mm part hanging under the hand.
- **`CONTACT_TOLERANCE = 0.012` m**: about twice a calibrated YAM station's
  camera error (RMS 5.4 mm), the error of a perceived support height.
- **`OVERTRAVEL = 0.02` m**: deeper than the 12 mm slide, so a hand that
  reaches an unfloored bottom met nothing.
- **`MIN_TOOL_HEIGHT = 0.015` m**: a real YAM station refuses a tool centre
  under 14 mm over its calibrated table. An object hanging little more than
  this under the hand leaves almost no window over the floor; grasp it higher
  or give `table_z`.
- **`REACH_TOLERANCE = 0.015` m**: no tighter than a real station's 12 mm per
  move.

## Simulation check

A YAM station twin (left arm, 30 mm cube, open 100 x 100 x 40 mm box, MuJoCo
at 500 Hz, simulator ground truth only to set up and to measure). A scripted
top-down grasp closed with the tool centre 5 mm over the cube's centre (the
cube settled 7.1 mm under it, a 22 mm hang; 12 mm and 27 mm with the grasp
10 mm over), lifted it 90 mm, and the cube's box after the lift stood in for
a perception skill's `held_obb`. The skill then placed it with `table_z` set
to the table plane; afterwards the cube was measured:

| case | route | cube vs place point | cube on the support | tilt |
|---|---|---|---|---|
| table, 6 cm +x / -x / +y / -y | placed x4 | 0.33 / 0.04 / 0.37 / 0.39 mm | yes (0.00 mm) | < 0.001 deg |
| table +x / +y, grasped 10 mm higher | placed x2 | 0.57 / 0.30 mm | yes | < 0.001 deg |
| box, region centre | placed | 1.2 mm | on the floor | < 0.001 deg |
| box, 25 mm toward the arm's corner | placed | 7.6 mm | on the floor | < 0.001 deg |
| table, support stated 8 mm high / low | placed x2 | 1.5 / 0.4 mm | yes | < 0.001 deg |
| table, support stated 30 mm high | no_contact | kept in the hand | - | 1.7 deg in the jaws |
| table, support stated 20 mm low | contact_too_early (+17 mm) | kept | - | 0.1 deg |
| box, place point on its far wall | contact_too_early (+26 mm) | kept | - | 1.1 deg |

- The stop read 2.4-5.5 mm under the expected contact with the 22 mm hang and
  9-10 mm under it with the 27 mm hang: the cube slid up the jaws (see above).
  Seven of the ten placements stopped on contact; three ended at the floor
  with the cube's nominal bottom under the table. Without `table_z` (an
  earlier run of the same cases) one of the four 6 cm table placements (+y)
  and both 27 mm-hang ones came back `no_contact` with the cube resting on the
  table: the safe failure, but a failure.
- In the box corner case the jaws, wider than the cube along their closing
  axis, met the box's inner wall and pushed the hand 7 mm sideways; the cube
  still landed upright on the floor, and `placed_position` (from where the
  hand stopped) was 0.8 mm from it. Over the box's centre the jaws cleared.
- `placed_position` was 0.6-2.8 mm from the cube's measured centre (xy).
- The real-station path (a raised "did not reach its target" on contact) is
  covered only by unit tests; nothing here ran on a real YAM.

## Provenance

Adapted from Ravens' `PickPlace` primitive (Zeng et al., Transporter
Networks, arXiv:2010.14406; Apache-2.0): the placing half, with the changes
above.
