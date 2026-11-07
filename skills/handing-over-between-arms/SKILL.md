---
name: handing-over-between-arms
description: Hands the object one arm holds to the other arm - the donor
  carries it to a handover point both arms reach (keeping its rotation), the
  receiver pre-poses back along a horizontal approach, moves in on the side of
  the object away from the donor's hand and closes; only when the receiver's
  jaw opening says it holds something does the donor open, and the hold is
  rechecked before and after the donor backs out. Use when a bimanual station
  must pass a held object between its arms (regrasp from the other side,
  reach a place only the other arm reaches, free the first arm), with the
  held object's box from a perception skill, both arms addressable by arm_id
  and the receiver empty.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [bimanual, handover, regrasp, transfer]
  hub:
    stage: execute
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [pick-place, bimanual]
    provenance:
      authors: [runtime-si]
      code: https://github.com/robotwin-Platform/RoboTwin/blob/main/envs/handover_block.py
      upstream_license: MIT
gap:
  requires: {connector: [robot.grasp_frame]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_gripper
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.open_gripper
    - robot.close_gripper
    - robot.grasp_frame
    - robot.get_observation
    - robot.move_to_joints
  required_inputs:
    object_obb: OrientedBoundingBox
    handover_position: Vec3
  produces_outputs:
    receiver_grip: float
    receiver_roll_error_deg: float
    donor_handover_pose: Se3Pose
    receiver_grasp_pose: Se3Pose
    object_position: Vec3
  exit_conditions:
    transferred: The receiver closed on something (jaw opening above the stop), the donor opened, the receiver still held after the donor backed out retreat_distance and after it drew back receiver_withdraw; object_position is where the receiver's draw-back should have left the object's centre.
    receiver_missed: The receiver could not reach its pre-grasp pose with either wrist solution, stopped short moving in, or closed to the stop; the receiver opened and backed off, the donor never opened and still holds the object at the handover pose.
    dropped: The donor's jaws were at the stop at the start (nothing held), or the receiver's jaws ran to the stop after the donor opened, while it backed out, or as the receiver drew back; `detail` says when.
    donor_stuck: The donor could not reach the handover pose (it still holds the object), its jaws did not open (both hands hold it), or it opened but could not back out; also routed when donor_arm equals receiver_arm or the connector reports no separate receiver arm.
  canonical_scripts:
    - hand_over: scripts/hand_over.py
  examples:
    - title: Hand a held object from arm 0 to arm 1
      path: examples/canonical_subgraph.json
---

# handing-over-between-arms

Pass a held object from the donor arm to the receiver arm without putting it
down. Every move names its arm (`arm_id`), one arm moves at a time.

## The motion

1. Refuse if the two arm ids are equal, or if the receiver's pose reads as the
   donor's (within 1 mm) or as no pose at all: a connector that clamps or
   ignores `arm_id` would move the donor when told to move the receiver.
2. Read the donor's jaw opening: at the stop means it holds nothing
   (`dropped`).
3. **Donor to the handover pose.** The object rides at a fixed offset from the
   donor's tool centre (from `object_obb` and the donor's pose, read together),
   so the donor goes, with its own rotation, to where that offset puts the
   object's centre on `handover_position`.
4. **Receiver geometry.** The approach is `receiver_approach`, or horizontal
   from the receiver's tool centre toward the handover point (it comes in from
   its own side). The *free side* is the direction across that approach away
   from the donor's tool centre (below the object for a top-down donor). The
   grasp point is `free_offset_fraction` of the object's half-extent that way
   from its centre, `insert_depth` along the approach; the pre-grasp is
   `pre_grasp` back along the approach.
5. **Receiver opens, goes to the pre-grasp** (`robot.grasp_frame` with that
   approach, jaws closing horizontally across it unless `close_heading_deg`
   is given; both wrist solutions). Its jaw roll about the approach is read
   back; more than 3 deg off (a solver that keeps only the approach axis
   leaves the roll free), the rest is turned on its last joint
   (`robot.move_to_joints`, up to three read-back turns, the first learning
   the joint's sign) and the hand re-centred on the pre-grasp, as
   stacking-objects does. Without a joint reading or `move_to_joints` it
   carries on and says so in `detail`; `receiver_roll_error_deg` is what is
   left. Then it moves straight in and closes.
6. **Hold check before the donor lets go.** The receiver's jaw opening
   (`robot.get_gripper`, 1.0 = open) must be above `empty_fraction`; otherwise
   the receiver opens and backs off to the pre-grasp (`receiver_missed`) and the
   donor keeps the object.
7. **Donor opens**; its jaws must read open. The receiver's hold is read again.
8. **Donor backs out** `retreat_distance` the way its hand came in (object
   centre to tool centre; straight up when it gripped at the centre, or
   `donor_retreat`). The receiver's hold is read again.
9. **Receiver draws back** `receiver_withdraw` along its approach with the
   object, and its hold is read a last time.

## Inputs

- `object_obb`: the held object's box, world frame (half-extents), measured
  while the donor holds it (e.g. registering-held-objects); the donor must not
  move between that measurement and this skill.
- `handover_position`: where the object's centre should be handed over, world.
- `donor_arm` (0), `receiver_arm` (1): the connector's arm ids.
- `receiver_approach` (none), `close_heading_deg` (none), `donor_retreat`
  (none): overrides for the geometry above.
- `pre_grasp` (0.10 m), `free_offset_fraction` (1.0), `insert_depth` (0 m),
  `retreat_distance` (0.10 m), `receiver_withdraw` (0.05 m),
  `reach_tolerance` (0.02 m), `empty_fraction` (0.04): see the constants.

## Required end states

- `transferred`: the receiver holds something by its jaw opening at every
  check and both arms finished their moves. The jaw opening cannot tell the
  object from the donor's finger or a corner caught at the jaw tips;
  re-perceive (verifying-grasps' `verify_grasp`) before relying on the hold.
- `receiver_missed`: the donor still holds the object at the handover pose;
  the receiver is open at its pre-grasp (or wherever its refused move left it).
- `dropped`: the object is not in the receiver's hand and the donor has
  opened (or never held it).
- `donor_stuck`: see `detail`; the object may be in the donor's hand, in both,
  or in the receiver's with the donor's open hand around it.

## The constants, and why

- **`PRE_GRASP = 0.10` m**: ACT's PickAndTransferPolicy (receiver 0.10 m back
  from the meet point) and RoboTwin handover_mic's `pre_grasp_dis`;
  handover_block uses 0.07.
- **`FREE_OFFSET_FRACTION = 1.0`**: ACT closes the receiver 0.02 m below the
  meet point on a 0.04 m cube, a whole half-extent. Measured in the YAM twin
  (30 mm cube, donor's tool centre 8 mm above its centre), the two hands'
  fingertips cleared each other by 0.1-0.2 mm at 0.5 (touching), 0.5-2.1 mm
  at 0.8 and 3.1-5.5 mm at 1.0, with the cube held every time. On a small
  object the hands share the same faces; give a larger object, or grip it
  off-centre with the donor, before trusting this clearance.
- **`INSERT_DEPTH = 0`**: RoboTwin's `grasp_dis = 0`.
- **`RETREAT_DISTANCE = 0.10` m**: RoboTwin handover_block's donor lift and
  ACT's giver withdrawal. **`RECEIVER_WITHDRAW = 0.05` m**: RoboTwin
  handover_mic's receiver move.
- **`EMPTY_FRACTION = 0.04`**: 3.8 mm on the YAM's 94 mm span: the stop plus
  verifying-grasps' 3 mm margin.
- **`REACH_TOLERANCE = 0.02` m**: arrival bound read back from each arm, no
  tighter than the servo's 4 mm or a real YAM station's 12 mm.

## Connectors and arm_id

- RoboSimStudio: every robot verb used here takes `arm_id`, counted in role
  order (0 = work arm, 1 = support arm); an id past the last arm is clamped to
  it, which is why step 1 compares the two arms' poses.
- The real YAM station bridge (`yam_real.py`, 2026-11-06) accepts `arm_id` on
  its verbs and drives the left arm whatever it says, and reads no second arm's
  pose: this skill refuses there (`donor_stuck`, "no separate arm") rather
  than moving the donor twice. It needs a dual-arm bridge to run on the real
  station.

## Simulation check

RoboSimStudio YAM station twin, both YAM arms (bases 0.61 m apart; arm 0 =
left = donor, arm 1 = right = receiver, moved by the harness to the mirror
of the left's working pose). The harness picks the 30 mm cube with the left
arm (scripted top-down grasp from `sim.get_object_obb`, tool centre 8 mm
above the cube's centre, jaws along world x), then runs this skill with
default parameters; it measures the cube with `sim.get_object_obb` after a
120-step wait. The twin filters the two arms' collisions against each other
(`sim.clearance` answers infinity, and the physics would let them pass
through each other), so the arm clearance is computed by the harness from
the arms' collision meshes every 20 ms during the skill (a vertex measure:
an overlap reads as about 0, not negative). Handover points are relative to
the bench top's centre:

| point (x, y, z) m | route | right holds after the left retreats | drop | min arm clearance | notes |
|---|---|---|---|---|---|
| (-0.10, 0.00, 0.15) | transferred | yes, cube 21 mm from its tool centre, jaws 0.33 | none | 5.5 mm (fingertip to fingertip) | donor's 0.10 m back-out straight up ended within tolerance (servo timeout) |
| (-0.05, 0.03, 0.20) | transferred | yes, cube 16 mm from its tool centre, jaws 0.46 | none | 0.1 mm (fingertips touching) | straight up refused, donor backed out sideways; jaws 0.46 suggest the cube sits turned in the receiver's grip |
| (-0.15, -0.03, 0.12) | receiver_missed | no, the left still holds it | none | 43 mm | receiver's pre-grasp out of reach (30 mm off) with both wrist solutions |
| (-0.05, -0.04, 0.14) | transferred | yes, cube 21 mm from its tool centre, jaws 0.33 | none | 3.3 mm (fingertip to fingertip) | one last-joint wrist turn on the receiver; roll error after it 0.02 deg |

Earlier runs at `free_offset_fraction` 0.5 (then the default) also
transferred at the three reachable points, but with the fingertips 0.1-0.2 mm
apart, which on the real arms would be the receiver closing on the donor's
fingers. `sim.check_grasp` read false for both hands throughout (also for the
left's plain pick), so "holds" above means the cube stayed within 25 mm of
the receiver's tool centre and did not move during the wait.

## Provenance

Adapted from RoboTwin's `handover_block` / `handover_mic` tasks
(`envs/handover_block.py`, `envs/handover_mic.py`) and ACT's scripted
`PickAndTransferPolicy` (https://github.com/tonyzhaozh/act/blob/main/scripted_policy.py),
both MIT: the meet pose, the receiver's
pre-grasp stand-off and off-centre close, the donor's open-then-back-out and
the receiver's draw-back are theirs. Changed: a world-frame box and the hands'
own poses in place of simulator object poses and contact points; one arm at a
time through robot.* tools with explicit `arm_id` instead of two interpolated
mocap trajectories; the receiver's hold is read from its jaw opening before
the donor opens and after each later move (RoboTwin reads the grippers only at
the end); refused moves are routed.
