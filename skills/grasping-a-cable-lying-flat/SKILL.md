---
name: grasping-a-cable-lying-flat
description: Grasps a thin cable lying flat on a table - top-down over a point
  on its centreline with the jaws closing across its local tangent, the
  fingertips brought down until they meet the table (the arm's stall, no force
  sensor), optionally narrowed first and held pressed on the table while they
  shut, then a test lift that checks the cable came up with the hand. Use when
  a cable, wire or cord a few millimetres thick lies on its support (from
  curve.fit_centerline / curve.endpoints, or a given point and direction) and
  must be picked up, e.g. to route or plug it.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [cable, deformable, grasp, contact, thin]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [cable-routing]
    provenance:
      authors: [runtime-si]
      code: https://github.com/newton-physics/newton/blob/main/newton/examples/multiphysics/example_franka_cable_ik_pick_place.py
      upstream_license: Apache-2.0
gap:
  requires: {connector: [robot.grasp_frame, robot.move_cartesian_until_contact, robot.set_grip, robot.describe_gripper, robot.describe_workspace]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_cartesian_until_contact
    - robot.move_to_joints
    - robot.solve_ik
    - robot.grasp_frame
    - robot.set_grip
    - robot.open_gripper
    - robot.close_gripper
    - robot.get_gripper
    - robot.describe_gripper
    - robot.describe_workspace
    - sam3.segment_text
    - geometry.mask_to_world_points
  required_inputs: {}
  produces_outputs:
    grasp_point: Vec3
    close_heading_deg: float
    planned_tool_z: float
    contact_z: float
    fingertips_over_table: float
    touched_table: bool
    pressed: bool
    pre_closed: bool
    jaw_width: float
    cable_rise: float
    lift_checked: bool
  exit_conditions:
    grasped: The jaws closed short of their closed stop and, with a cable_description, the cable beside the fingers was seen to rise at least min_rise (7 mm) with the test lift (lift_checked true); without one it is judged by the jaw gap alone (lift_checked false), which on a real YAM does not tell a cable caught on the fingertips from a held one. The hand is left lifted, closed.
    missed: The jaws closed to their stop, or the cable beside the fingers did not rise with the hand; the hand opened and stands over the cable.
    unreachable: The cable does not lie flat at the grasp point, the depth the fingertips need is under the tool-centre floor (min_tool_height), no wrist solution reaches the grasp pose (IK preview), or a rise, the approach, the wrist turn or the descent was refused; `detail` says which.
  canonical_scripts:
    - grasp_flat_cable: scripts/grasp_flat_cable.py
  examples:
    - title: Grasp a fitted cable at the middle of its length
      path: examples/canonical_subgraph.json
---

# grasping-a-cable-lying-flat

Pick up a cable a few millimetres thick that lies flat on the table.

## The motion

1. **Where.** From a `centerline` (`curve.fit_centerline` or
   `curve.track_centerline`), at `grasp_s` of its length, or `back_from_end`
   along it from an `end` (one of `curve.endpoints`' ends); or a given
   `grasp_point` and `tangent`. The cable's horizontal way is taken over 1 cm
   either side. A cable lying flat has its centre a radius over the table, and
   that height is used rather than the camera's (the depth of a 3.5 mm cable
   is noisier than its thickness). A centreline more than 12 mm higher there
   does not lie flat: `unreachable`.
2. **How deep.** The fingertips are aimed `pinch_below` (4 mm) under the
   cable's centre (a cable under 3.4 mm, 1.5 mm further), never nearer the
   table than 0.8 mm (0.4 mm), and the tool centre the fingertips' reach over
   that (`robot.describe_gripper`'s `finger.reach_m`). A tool centre under
   `min_tool_height` over the table is `unreachable` rather than tried higher,
   since there the fingertips close over the cable.
3. **Across the cable.** `robot.grasp_frame` with the jaws closing square to
   the tangent, both wrist solutions, previewed with `robot.solve_ik` at the
   grasp pose (a stall at the edge of reach reads like contact).
4. **Approach.** Straight up first if the hand is low, open, over the grasp
   point `approach_height` (6 cm) over the planned depth. The jaws' closing
   direction is read back; an arm whose solver leaves the wrist roll free
   arrives unturned, and the rest is turned on the last joint
   (`robot.move_to_joints`) and the hand re-centred.
5. **Pre-close** (`pre_close_width`, 10 mm; 0 off): 12 mm over the planned
   depth the fingers are narrowed to just wider than the cable.
6. **Down to the table.** `robot.move_cartesian_until_contact` aimed
   `overtravel` (8 mm, twice a servo's 4 mm arrival bound) under the planned
   depth, never under the floor; the arm stops where the fingertips stand on
   the table. A stop more than 6 mm over the planned depth (a fingertip on the
   cable, a higher table) is noted, and so is a stop over an aim that does not
   reach the table (the floor): neither counts as `touched_table`.
7. **Close** (`press`, 3 mm; 0 off): the hand is commanded `press` under where
   it stands, so the fingertips stay on the table while they shut; otherwise it
   is commanded where it stands, so they close resting on it. A close that
   stops wider than twice the cable plus 4 mm is sent once more.
8. **Lift-check.** Up `lift_height` (4 cm). Jaws at their closed stop:
   `missed`. With `cable_description`, `sam3.segment_text` on the camera named
   `camera` (default the wrist one) and `geometry.mask_to_world_points`: the
   median height of the cable's points 8-25 mm from the tool centre, in plan,
   must stand `min_rise` (7 mm) over the cable's height before the grasp.

## Inputs

- `centerline` (Centerline or a points list) with `grasp_s` (0.5), or with
  `end` (a CurveEnd) and `back_from_end` (0.03 m); or `grasp_point` and
  `tangent` (Vec3s).
- `table_z` (recommended; else `robot.describe_workspace`'s `surface_z`, else
  the cable's height less its radius), `cable_radius` (else the centreline's
  `radius_m`, else 1.75 mm).
- `pinch_below`, `press`, `pre_close_width`, `approach_height`, `overtravel`,
  `min_tool_height` (0.014 m), `lift_height`, `min_rise`, `reach_tolerance`:
  see the constants.
- `cable_description` ("" skips the camera check), `camera` ("wrist").
- `check_reach` (true), `roll_limits` (the YAM's joint 6), `arm_id` (0;
  passed only when nonzero).

## Required end states

- `grasped`: the hand stands `lift_height` over where it closed, closed on
  the cable. What the camera saw after the lift when `lift_checked`; what the
  jaws read otherwise. Re-perceive before relying on where the cable is in the
  hand.
- `missed`: the hand is open over the cable. The cable may have moved: fit
  its centreline again before another try.
- `unreachable`: nothing grasped; the hand is open (if it moved at all) and
  stands at the approach height.

## What changed from Newton, and why

Newton's example (`_build_keyframes`) drives a Franka through fixed
keyframes: 0.20 m over the cable's centre, down to put the tool centre at the
cable's centre height, close, lift. It knows the cable's pose from the
simulator, its rod is 10 mm thick, and its gripper meets the cable square.
Here the point and tangent are perceived, the jaws close across the tangent
(the example's cable is straight along x), and the descent is not to a fixed
height: a real YAM station's camera does not know the table to the
millimetre, and its tapered fingertips must reach under a 3.5 mm cable's
middle. So the descent stops on the table by the arm's stall (no force
sensor), aimed a little under it. Newton's separate capture and hold widths
(its comments describe closing to capture and then holding just under the
cable's radius so the fingers do not ratchet through a lagged contact) are
one close here; its current constants set both to 0.

The pre-close and the press come from a real YAM threading station
(`PRE_CLOSE`, `CLOSE_PRESS`): shut from half open with the tips on the
table, the tapered fingertips wedged the flat cable up and over them and
closed empty, and the press keeps the fingertips from lifting off the table
while they close. The lift-check is that station's too: the jaw gap after a
close read the same for a 3.5 mm cable caught on the fingertips as for
nothing, so the cable beside the fingers is looked at after a 4 cm lift.

## The real YAM's floor, and why this can route `unreachable` by default

A real YAM station refuses a tool centre under 14 mm over its table
(`MIN_TOOL_HEIGHT`). The linear_4310's closed fingertips reach 10.2 mm under
the tool centre, so at that floor they stand 3.8 mm over the table: over a
3.5 mm cable lying flat, below anything they can pinch. That is the station's
known failure with a flat charging cable. With the default floor this skill
says so (`unreachable`, the depth needed in `detail`) instead of closing over
the cable; on a station whose motion server allows a lower tool centre (the
threading station was restarted with a 5 mm floor for this), pass
`min_tool_height` to match.

## Simulation check

A YAM station twin (left arm, linear_4310 fingertips reaching 10.2 mm, the
jaws' gains 6000 / 120) with the threading twin's 3.5 mm charging cable: a
rod of 7.2 mm capsules on VBD beside the MuJoCo arm, 2000 kg/m^3, bending
0.08 / 0.1, the pads at mu 0.5 on it (the value that made a flat cable taken
3.5 mm over the table fail to lift there, as on the real arm), laid straight
on the tablecloth 0.27 m in front of the left arm. The simulator's own
capsule centres stood in for `curve.fit_centerline` and measured the outcome
(capsule heights before, after the 4 cm test lift, and after 2 s held up).
`min_tool_height` was 5 mm (the threading station's lowered floor).

| case | route | fingertips over the table at the stop | jaws | cable at the grasp point (before -> lifted -> 2 s) | capsules > 5 mm up |
|---|---|---|---|---|---|
| s = 0.3, press 3 mm | grasped | 0.3 mm | 5.4 mm | 1.75 -> 40.2 -> 40.2 mm | 17 of 37 |
| s = 0.5, press 3 mm | grasped | 0.3 mm | 5.4 mm | 1.75 -> 40.6 -> 40.7 mm | 17 |
| s = 0.7, press 3 mm | grasped | 0.3 mm | 5.5 mm | 1.75 -> 41.0 -> 41.0 mm | 17 |
| s = 0.3, no press | grasped | 0.3 mm | 5.4 mm | 1.75 -> 40.2 -> 40.2 mm | 17 |
| s = 0.5, no press | grasped | 0.3 mm | 5.4 mm | 1.75 -> 40.8 -> 41.0 mm | 17 |
| s = 0.7, no press | grasped | 0.3 mm | 5.5 mm | 1.75 -> 41.1 -> 41.2 mm | 17 |
| s = 0.5, no press, no pre-close | grasped | -0.2 mm | 5.4 mm | 1.75 -> 40.9 -> 41.0 mm | 17 |
| cable at 45 deg, s = 0.5, press | grasped | 0.3 mm | 5.5 mm | 1.75 -> 40.0 -> 40.1 mm | 17 |
| aimed at a 14 mm floor (no press) | missed | 6.4 mm | 0.0 mm | 1.75 -> 1.75 -> 1.75 mm | 0 |

- 8 of 8 grasps that reached the table lifted the cable and held it for 2 s
  (no slip: under 0.2 mm in the hold); `grasped` was right each time, and
  `missed` was right for the one aimed at the floor.
- Each time the twin's solver left the wrist roll free: the jaws arrived
  54-89 deg off square to the cable, and one last-joint turn brought them
  within 0.3 deg.
- Aimed at the real station's floor (tool centre 13.95 mm over the table,
  `overtravel` 0) the arm stood 2.6 mm over its aim in the air, the fingertips
  6.4 mm over the table, and the jaws shut over the cable: the real station's
  known failure. With the default 14 mm floor the skill routes `unreachable`
  before moving.
- An earlier round with `overtravel` 4 mm gave the same lifts; there the
  contact move called itself arrived 3.1 mm short of its aim (the servo's 4 mm
  bound) with the fingertips 0.3 mm over the table, which is why the default
  is now 8 mm.

**The twin makes the flat pinch easier than the real fingertips do.** Every
grasp whose fingertips reached the table lifted the cable, with or without
the press and the pre-close, so the twin cannot tell those options apart;
on the real station, shutting from half open with the tips on the table
wedged the cable up and over the tapered fingertips and closed empty. The
twin's arm stops where it is aimed and its stall detection reads the table
cleanly, while the real arm stands 1.7-3.2 mm over its command near the table
and a real station's move is judged to 12 mm. The twin's jaws stopped at
5.4-5.5 mm on a 3.5 mm cable, so its jaw reading is not the real one either.
What the twin does show is the geometry: fingertips at the table pinch the
cable, fingertips a few millimetres over it close over it. The camera
lift-check did not run (this twin's observation has no cameras); it is
covered by unit tests only. Nothing here ran on a real YAM.

## Provenance

Adapted from Newton's `example_franka_cable_ik_pick_place.py`
(newton-physics/newton, Apache-2.0, Copyright (c) 2026 The Newton
Developers), with the changes above; the pre-close, press and lift-check
values are a real YAM threading station's.
