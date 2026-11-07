---
name: stacking-objects
description: Stacks the object the hand holds centred on top of a target
  object - turns the hand about the vertical so the held box's faces line up
  with the target's, approaches above the target's top, comes straight down
  until the arm stops on contact, releases and retreats straight up
  (ManiSkill's stack-cube solution and RoboTwin's block stacking, with
  placing-with-contact-descent's descent in place of a move to a known
  height). A stop far from the target's top, a descent that meets nothing or
  a stop that pushed the hand sideways is routed and the object kept. Use
  when a held block or box must go on top of another one whose oriented box
  a camera has measured (towers, pyramids, stacking boxes).
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [stack, place, contact, release, align]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/haosulab/ManiSkill/blob/main/mani_skill/examples/motionplanning/panda/solutions/stack_cube.py
      upstream_license: Apache-2.0
gap:
  requires: {connector: [robot.move_cartesian_until_contact]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_cartesian_until_contact
    - robot.move_to_joints
    - robot.solve_ik
    - robot.open_gripper
  required_inputs:
    target_obb: OrientedBoundingBox
    held_obb: OrientedBoundingBox
  produces_outputs:
    released: bool
    yaw_turn_deg: float
    yaw_error_deg: float
    expected_contact_z: float
    contact_z: float
    contact_offset: float
    stacked_position: Vec3
  exit_conditions:
    stacked: The descent over the target stopped on contact inside the contact window and within 8 mm of the planned point, the hand opened and retreated straight up; stacked_position is where the hand left the held object's bottom centre. What the hand did, not a check that the stack stands.
    misaligned: The descent did not meet the target's top where expected - it stopped more than contact_tolerance (12 mm) above it, reached its bottom without stopping (the held object missed the top, or slid up the jaws), or stopped pushed sideways more than lateral_tolerance; the hand did not release and lifted the object back to the approach height.
    blocked: The target's top is tilted more than 10 deg, no aligned turn can reach the contact pose (IK preview), the rise, the approach, the re-centring after a wrist turn or a descent move was refused, or the retreat after the release failed (`released` says which side of the release); `detail` says which.
  canonical_scripts:
    - stack_on_object: scripts/stack_on_object.py
  examples:
    - title: Stack a held block on a target block
      path: examples/canonical_subgraph.json
---

# stacking-objects

Put the block in the hand on top of another one, faces lined up, without
knowing the stack's height better than the camera does.

## The motion

1. Read both boxes from above (`target_obb`, `held_obb`, world frame, half
   extents, measured after the grasp): each one's top and bottom (highest and
   lowest corner), tilt (world up against the nearest face axis) and yaw (the
   heading of a horizontal face axis). A target tilted more than
   `max_target_tilt_deg` is `blocked`.
2. The goal: the held box's bottom centre on the target's top centre. The hand
   goes there shifted by the held object's offset in the hand, measured now;
   contact is expected with the tool centre the object's hang over the
   target's top.
3. The turn: the hand turns about the vertical through the tool centre by the
   smallest angle that lines the held box's faces up with the target's
   (`yaw_period_deg`, 90 for two boxes), the held offset turning with it.
   ManiSkill's dry-run search of its grasp over 0 and +-90 deg becomes a
   `robot.solve_ik` preview of the contact pose at that turn and one period
   either side, smallest first; the first reachable one is used.
4. Rise straight up if the hand is under the approach height, keeping its
   rotation, then go to `approach_height` over the expected contact, turning.
5. Read the turn back from `robot.get_ee_pose`. An arm that holds the whole
   pose arrives turned; one whose solver keeps only the approach axis (a free
   wrist roll, as in the YAM station twin) does not, and the rest, when over
   `yaw_tolerance_deg`, is turned on the last joint (`robot.move_to_joints`,
   within `roll_limits`, or the equivalent turn one period the other way).
   The first turn learns which way the joint turns the hand; the hand is
   re-centred after each (the tool centre may sit off that joint's axis);
   up to three turns. A wrist that cannot turn (no joint reading, a refused
   move, no room) does not stop the stack: `yaw_error_deg` and `detail` say
   how far off the faces are.
6. placing-with-contact-descent's descent: `robot.move_cartesian_until_contact`
   to `contact_tolerance` over the expected contact (a stop there is too
   early), then on to `overtravel` under it, never under `min_clearance` over
   the target's top (the fingertips) or `min_tool_height` over `table_z`.
7. A stop on the way within `lateral_tolerance` of the planned xy is the
   target's top: open, retreat `retreat_height` straight up.

## Inputs

- `target_obb`: the box of the object to stack on.
- `held_obb`: the held object's box, measured after the grasp
  (registering-held-objects, or the camera on the lifted object).
- `table_z` (optional): the table plane's height, a station calibration; it
  only adds the table floor.
- `align_yaw` (true), `yaw_period_deg` (90): turn to the target's faces; 180
  lines up the boxes' x axes, 360 a full heading.
- `yaw_tolerance_deg` (3), `roll_limits` (the YAM's joint 6, +-2.0944 rad):
  the wrist turn's read-back bound and the last joint's range.
- `approach_height` (0.08 m), `overtravel` (0.02 m), `contact_tolerance`
  (0.012 m), `min_clearance` (0.012 m), `min_tool_height` (0.015 m),
  `lateral_tolerance` (0.008 m), `max_target_tilt_deg` (10), `retreat_height`
  (= approach height), `reach_tolerance` (0.015 m): see the constants.
- `check_reach` (true), `arm_id` (0; passed only when nonzero).

## Required end states

- `stacked`: released on the target's top and retreated. This is what the
  hand did; re-perceive the stack (verifying-placement) before building on
  it.
- `misaligned`: still holding, lifted back. Re-perceive the target and the
  held object and try again, or set the object down elsewhere.
- `blocked`: nothing stacked unless `released` is true (then the object is
  down but the hand could not leave it cleanly).

## What changed from the sources, and why

ManiSkill's `stack_cube.py` reads both cubes' simulator poses, grasps cube A
top-down with a yaw search for a reachable grasp, lifts 0.1 m, and moves the
hand by the offset that puts cube A one cube height over cube B, then opens.
It keeps the grasp's yaw, so the cubes' faces line up only by chance.
RoboTwin's `stack_blocks_three.py` / `place_actor` turns the hand so the
held block's frame matches the target's (`get_place_pose`'s `start2target`),
approaches 0.05 m over it along its up axis and moves down to it.

Here the poses are perceived boxes (no simulator ground truth), RoboTwin's
alignment is folded to a box's 90 deg face symmetry so the hand turns at most
45 deg when it can, and ManiSkill's grasp-yaw search is a reach preview over
the equivalent turns. Both sources move to a computed height; a real stack's
height is known only to the camera's error, so the last centimetres are the
hub's placing-with-contact-descent descent: the arm's own stall detection
(tracking progress, the YAM has no force sensor), judged by the hand's
measured stop. A stop too high, no stop, or a sideways push is routed and the
object kept. Neither source checks the turn it asked for; here it is read
back and finished on the last joint, because the YAM station twin's solver
leaves the wrist roll free (it holds only the approach axis) and its servo
reported a turned approach "reached" with the hand 37 deg from the asked
heading. The descent is floored over the target's top so the fingertips
stay off it, which means a held object that slid up the jaws can end the
descent at the floor and read as `misaligned` (the safe failure).

## The constants, and why

- **`APPROACH_HEIGHT = 0.08` m**: placing-with-contact-descent's; ManiSkill
  lifts 0.1 m, RoboTwin approaches 0.05 m.
- **`CONTACT_TOLERANCE = 0.012` m**: about twice a calibrated YAM station's
  camera error (RMS 5.4 mm).
- **`OVERTRAVEL = 0.02` m**: deeper than the 12 mm a friction-held cube slides
  up the jaws (placing-with-contact-descent's measurement).
- **`MIN_CLEARANCE = 0.012` m**: closed linear_4310 fingertips reach about
  10 mm under the tool centre.
- **`MIN_TOOL_HEIGHT = 0.015` m**: a real YAM station refuses a tool centre
  under 14 mm over its calibrated table.
- **`LATERAL_TOLERANCE = 0.008` m**: a 30 mm block's centre stays 7 mm inside
  the target's edge.
- **`MAX_TARGET_TILT_DEG = 10`**: a choice, not tuned.
- **`YAW_TOLERANCE_DEG = 3`**: a servo's own 3 deg arrival bound (a real YAM
  station's is 0.04 rad, 2.3 deg).
- **`ROLL_LIMITS = (-2.0944, 2.0944)` rad, `LIMIT_MARGIN = 0.05` rad**: the
  i2rt YAM's joint 6 (its MJCF), as in pouring-by-wrist-roll.

## Simulation check

A YAM station twin (left arm, MuJoCo at 500 Hz, control at 50 Hz; simulator
ground truth only to set up and to measure). The harness's 30 mm cube plus a
second 30 mm cube (the target) registered 10 cm from it, turned 0 / 20 / 45
deg from the first cube's yaw, at five spots (nominal, +-1 cm in x and y) per
angle. A scripted top-down grasp (tool centre 5 mm over the cube's centre,
lift 90 mm) from `sim.get_object_obb`; the lifted cube's box stood in for
`held_obb` and the target's for `target_obb`, `table_z` the table plane. After
the skill the scene ran 0.2 s, was measured, ran 2 s more and was measured
again ("stood": still on top within 15 mm and 3 mm of height, under 5 deg of
tilt, moved under 2 mm in the 2 s).

| target turned | route | stood 2 s | top cube offset (xy) | faces off | tilt | hand turn |
|---|---|---|---|---|---|---|
| 0 deg | stacked x5 | 5/5 | 0.4-1.4 mm | <= 0.15 deg | <= 0.003 deg | +6.0 deg |
| 20 deg | stacked x5 | 5/5 | 0.6-1.5 mm | <= 0.10 deg | <= 0.004 deg | +26.0 deg |
| 45 deg | stacked x5 | 5/5 | 0.4-1.5 mm | <= 0.33 deg | <= 0.004 deg | -39.0 deg |

- The grasp left the held cube 6 deg off its starting yaw, hence the +6 deg
  turn at 0 deg. 45 deg is the ambiguous case; the skill took the 39 deg turn
  (the nearest face) and lined the faces up as well as at 0 and 20 deg.
- The approach alone left the hand about 11 deg from its starting heading in
  every case, whatever turn was asked (the free wrist roll); one last-joint
  turn of 17-37 deg finished it, within 0.1 deg of the asked turn.
- Every descent stopped on contact 2.9-3.4 mm under the expected height (the
  cube slid a little up the jaws), 3.0-3.3 mm sideways of the planned point;
  the top cube sat on the target (bottom-to-top gap -0.01 mm) and did not move
  in the 2 s. `stacked_position` was 1.3-2.3 mm from the cube's measured
  centre. The target moved 0.5 mm.
- The `misaligned` exit, target turned 20 deg, its box handed to the skill
  wrong: 20 mm lower than it is -> stopped 17 mm above the stated top, kept
  and lifted back; 20 mm higher -> the descent finished 9 mm under the
  expected contact without a stop, kept; 25 mm off in x (the held cube
  overlapping the target's top by 5 mm) -> finished without a stop, kept, but
  the cube had caught the target's edge and was tilted 15 deg in the jaws
  when the hand lifted back.
- The real-station path (a raised "did not reach its target" on contact, a
  full-pose arm that arrives turned) is covered only by unit tests; nothing
  here ran on a real YAM.

## Provenance

Adapted from ManiSkill's `stack_cube.py` motion-planning solution
(Apache-2.0) and RoboTwin's `stack_blocks_three.py` / `place_actor` (MIT,
Copyright (c) 2025 Tianxing Chen, `LICENSE.robotwin`), with the descent of
the hub's placing-with-contact-descent (adapted from Ravens, Apache-2.0).
