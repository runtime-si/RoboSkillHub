---
name: lifting-with-both-arms
description: Lifts an elongated object (a bar, a tray, a pot by its handles)
  with both arms - one near-top-down grasp near each end of the object's long
  axis (or at two given handle points), each arm taking the end nearer its
  hand, its wrist leaned 15 deg away from the other hand so the two wrists
  keep clear, then both hands rise together in small interleaved steps that keep their
  separation and height difference bounded, with both jaw openings and the
  separation checked after every step. Use when an object is too long, heavy
  or floppy for one hand and a bimanual station has both arms addressable by
  arm_id, the object's box comes from a perception skill and both hands are
  empty.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [bimanual, lift, dual-arm, long-object]
  hub:
    stage: execute
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [pick-place, bimanual]
    provenance:
      authors: [runtime-si]
      code: https://github.com/robotwin-Platform/RoboTwin/blob/main/envs/lift_pot.py
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
  produces_outputs:
    rise_m: float
    hand_tilt_deg: float
    separation_change_m: float
    first_grip: float
    second_grip: float
    first_grasp_pose: Se3Pose
    second_grasp_pose: Se3Pose
    object_obb: OrientedBoundingBox
  exit_conditions:
    lifted: Both hands rose lift_height in steps, every step arrived within level_tolerance of its height, both jaw openings stayed above the stop and the hands' separation within separation_tolerance of its value at the grasp; object_obb is the input box raised by the hands' mean rise (the hands' motion, not a re-perception).
    slipped: A jaw closed to the stop at its grasp point (both hands then opened and backed off to their hovers), or during the lift a jaw ran to the stop or the hands' separation drifted past separation_tolerance; mid-lift the hands stay where they stopped, closed, and `detail` says which arm and at what height.
    unreachable: No hover, wrist solution or descent for one arm (hands opened and backed off), or an arm could not follow a lift step (the other was brought back level with it, both still closed on the object); also routed when the two arm ids are equal, the connector reports no separate second arm, or the box's long axis is not horizontal.
  canonical_scripts:
    - lift_with_both_arms: scripts/lift_with_both_arms.py
  examples:
    - title: Lift a bar 10 cm with arms 0 and 1
      path: examples/canonical_subgraph.json
---

# lifting-with-both-arms

Two hands, one near each end of a long object, lift it level. Every robot
call names its arm (`arm_id`); the arms move one at a time, in steps small
enough that the object never tilts by more than one step over the grasp
separation.

## The motion

1. Refuse (`unreachable`) if the two arm ids are equal, or if the second arm's
   pose reads as the first's (within 1 mm) or as no pose at all: a connector
   that clamps or ignores `arm_id` would move one arm twice.
2. **Grasp points.** With `handles` (two world points), those. Otherwise the
   box's longest horizontal axis gives two points `end_inset` in from its ends
   (at least a quarter of the half-length from the centre), at a tool-centre
   height `grasp_depth` under the top face (never under the centre). The jaws
   close across the long axis (or across the handle-to-handle line), unless
   `close_heading_deg` is given. Each arm takes the end nearer its current tool
   centre (the pairing with the smaller total distance).
3. **Approach direction.** Straight down, leaned `outward_tilt_deg` (15 deg)
   so the wrist sits away from the other hand: the approach points down and
   toward the other grasp point. `outward_tilt_deg=0` is a plain top-down
   grasp (see the simulation check for why that is not the default).
4. **Each arm in turn** opens, goes to its hover `pre_grasp` back along its
   approach (`robot.grasp_frame` with that approach, both wrist solutions),
   and has its jaw roll about the approach read back; more than 3 deg off (a solver that keeps only the approach axis
   leaves the wrist roll free), the rest is turned on its last joint
   (`robot.move_to_joints`, up to three read-back turns) and the hand
   re-centred, as stacking-objects does.
5. **Both descend** (straight cartesian moves along their approaches) and **both close**. A jaw
   opening at the stop means that hand closed on nothing: both open and back
   off (`slipped`).
6. **Lift** `lift_height` in steps of `lift_step`: for each step one arm rises
   to the step height, then the other; the arm that leads alternates. Each
   move is read back (within `reach_tolerance`, and within `level_tolerance` of
   its height). After each pair both jaw openings must stay above
   `empty_fraction` and the distance between the tool centres within
   `separation_tolerance` of its value at the grasp (`slipped` otherwise).
   An arm that cannot follow ends the lift (`unreachable`) with the other arm
   brought back down level with it.

## Inputs

- `object_obb`: the object's box, world frame (half-extents), lying with its
  long axis horizontal.
- `handles` (none): two world points to grasp instead of the box's ends.
- `first_arm` (0), `second_arm` (1): the connector's arm ids.
- `close_heading_deg` (none): the jaws' closing heading, world yaw [deg].
- `outward_tilt_deg` (15): each wrist's lean away from the other hand [deg].
- `lift_height` (0.10 m), `lift_step` (0.01 m), `pre_grasp` (0.08 m),
  `end_inset` (0.03 m), `grasp_depth` (0.02 m), `reach_tolerance` (0.02 m),
  `level_tolerance` (0.008 m), `separation_tolerance` (0.015 m),
  `empty_fraction` (0.04): see the constants.

## Outputs

`rise_m` (the hands' mean rise from the grasp), `hand_tilt_deg` (the slope of
the line between the tool centres, from their rises), `separation_change_m`
(the largest drift of their distance), `first_grip` / `second_grip` (jaw
openings, 1.0 = open), the two grasp poses, and `object_obb` raised by
`rise_m`. All of it is what the hands did; the jaw opening cannot tell a firm
hold from a corner caught at the jaw tips, so re-perceive the object before
relying on it.

## The constants, and why

- **`LIFT_HEIGHT = 0.10` m**: RoboTwin lifts the pot to z 0.88 m, about 0.1 m
  over its resting height, and scores it lifted above 0.82 m.
- **`OUTWARD_TILT_DEG = 15`**: measured in the YAM twin (below): straight
  top-down, the two arms' wrist links came within 0.1-1.6 mm of each other on
  a 0.14 m grasp separation, whatever the separation (0.14 or 0.17 m), the
  bar's distance from the bases or the jaw heading; leaned 15 deg, 63 mm with
  the bar still lifted level; at 25-35 deg over 100 mm, but the leaning pads
  let the bar pivot 4-8 deg and sag 6-15 mm in the jaws.
- **`LIFT_STEP = 0.01` m**: the hands differ in height by at most one step,
  3.6 deg over a 0.16 m separation.
- **`PRE_GRASP = 0.08` m**: RoboTwin's `pre_grasp_dis` is 0.035 m behind a
  planned approach; with a joint-space move to the hover the open jaws need
  more room over the object, and the last part is a straight cartesian descent.
- **`END_INSET = 0.03` m**: the pads fully on the object near its end.
- **`GRASP_DEPTH = 0.02` m**: the YAM's closed fingertips reach ~10 mm under
  the tool centre; on a 40 mm bar they stop ~10 mm over the table.
- **`SEPARATION_TOLERANCE = 0.015` m**: two servos' 4 mm arrival balls plus a
  real YAM's sag, under RoboTwin's 0.03 m bound on each tool centre's distance
  from its contact point.
- **`LEVEL_TOLERANCE = 0.008` m**: under one step, so a stalled arm is not read
  as following.
- **`EMPTY_FRACTION = 0.04`**: 3.8 mm on the YAM's 94 mm span: the stop plus
  verifying-grasps' 3 mm margin.
- **`REACH_TOLERANCE = 0.02` m**: no tighter than the servo's 4 mm or a real
  YAM station's 12 mm.

## Connectors and arm_id

- RoboSimStudio: every robot verb used here takes `arm_id`, counted in role
  order (0 = work arm, 1 = support arm); an id past the last arm is clamped to
  it, which is why step 1 compares the two arms' poses.
- The real YAM station bridge (2026-11-06) drives the left arm whatever
  `arm_id` says and reads no second arm's pose: this skill refuses there
  (`unreachable`, "no separate arm"). It needs a dual-arm bridge.

## Simulation check

RoboSimStudio YAM station twin, both YAM arms (bases 0.61 m apart along y;
arm 0 = left, arm 1 = right, moved by the harness to the mirror of the left's
working pose). The harness adds a 200 x 40 x 40 mm, 50 g bar (friction 0.8)
lying on the table between the arm bases (bench frame x -0.12 m, y 0, long
axis along y, base to base), takes its box from `sim.get_object_obb` in place
of perception, runs this skill with default parameters (lift 0.10 m) and
measures the bar after a 120-step wait. The twin filters the two arms'
collisions against each other (`sim.clearance` answers infinity and the
physics lets them pass through each other), so the arm clearance is computed
by the harness from the arms' collision meshes every 20 ms during the skill
(a vertex measure: an overlap reads as about 0, not negative).
`sim.check_grasp` read false throughout, so "slipped" below is the harness's
own judgement: the bar's centre moved more than 10 mm relative to the
midpoint of the two tool centres between the grasp and the end.

Defaults (`outward_tilt_deg` 15):

| run | bar yaw | route | bar rise | bar tilt (long axis) | bar moved in the hands | slipped | min arm clearance |
|---|---|---|---|---|---|---|---|
| 1 | 0 deg | lifted | 99.4 mm | 1.06 deg | 0.9 mm | no | 63.0 mm (wrist link4 to link4) |
| 2 | 0 deg | lifted | 99.3 mm | 1.06 deg | 1.0 mm | no | 63.1 mm |
| 3 | 0 deg | lifted | 99.2 mm | 1.07 deg | 1.6 mm | no | 63.0 mm |
| 4 | 20 deg | lifted | 99.0 mm | 0.82 deg | 1.5 mm | no | 88.0 mm |

In all four the hands rose 99.4-99.7 mm (tool centres read back), their
separation drifted at most 1.2 mm, the hands' slope was under 1 deg, both
jaws read 0.45-0.46 (the bar's 40 mm on the 94 mm span) and the bar did not
move during the wait.

Other runs (same bar):

| setting | route | bar rise | bar tilt | moved in the hands | min arm clearance |
|---|---|---|---|---|---|
| top-down (`outward_tilt_deg` 0), yaw 0, three runs | lifted | 99.3 mm | 0.20-0.22 deg | 0.5-0.6 mm | 0.2-1.3 mm (link4 to link4: touching) |
| top-down, yaw 20 deg | lifted | 99.6 mm | 0.04 deg | 0.1 mm | 6.9 mm |
| top-down, grasps 15 mm from the ends | lifted | 99.4 mm | 0.17 deg | 0.5 mm | 0.1 mm |
| top-down, bar 50 / 100 mm nearer the bases | lifted | 99.7 mm | 0.03-0.19 deg | 0.2-1.0 mm | 0.1-0.2 mm |
| tilt 25 deg | lifted | 92.0 mm | 5.76 deg | 8.2 mm | 104.8 mm |
| tilt 25 deg, yaw 20 deg | lifted | 85.5 mm | 7.79 deg | 14.8 mm (slipped) | 110.5 mm |
| tilt 35 deg | lifted | 93.4 mm | 0.41 deg (4.0 deg turned) | 5.9 mm | 102.8 mm |
| tilt 15 deg, `lift_height` 0.40 m | unreachable at step 27/40 (arm 0 timed out 10 mm short of 270 mm) | 238 mm | 0.51 deg | 19.9 mm (slipped) | 63.0 mm |

What the checks do not see: the skill's own checks are the jaw openings and
the tool centres. A bar that pivots or slides in the jaws while the hands
stay where they were sent (tilt 25 deg with the bar turned; the 0.40 m lift
near the edge of reach, where the servo timed out on the last steps but
within the 8 mm level bound) still reads as held at constant separation;
re-perceive the object after the lift when its pose matters.

## Provenance

Adapted from RoboTwin's `lift_pot` task (`envs/lift_pot.py`, `play_once` and
`check_success`, MIT), with handle points as cap-x's two-arm lift takes them
(`capx/integrations/franka/two_arm_lift.py`, MIT): two contact points, one per
arm, a short pre-grasp stand-off, a shared vertical displacement, and success
judged from the tool centres staying at their contact points. Changed: the
pot's simulator contact points replaced by points near the ends of a
world-frame box (or given handles), each assigned to the nearer arm; RoboTwin's
half-closed pre-shape dropped; each wrist leaned 15 deg away from the other
hand (straight top-down the twin's two YAM wrists touched); one synchronous `move_by_displacement` replaced
by small interleaved steps through robot.* tools with an explicit `arm_id`;
both jaw openings and the hands' separation checked after every step
(RoboTwin checks at the end only, from simulator contact points); refused
moves routed, and a stalled arm's partner brought back level.
