---
name: grasping-flat-objects
description: Grasps an object lying flat on a surface that is too thin for a
  normal top-down pinch at its box centre (a card, a phone, a thin plate or
  lid, a coin if wide enough) - turns the jaws across its short side, brings
  the fingertips down to just above the support (as low as the tool-centre
  floor allows), closes, checks the jaw gap reads about the slab's width,
  lifts and checks again; retries once across the long side. An optional
  tilt brings one fingertip in along the support at the slab's edge (it did
  not hold in a station twin). Use when a flat object must be picked from a
  table and a grasp at its box centre would close on air above it, with its
  oriented box from a perception skill and the hand empty.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [grasp, flat, thin, slab, card, plate, pinch]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/nssmd/RoboRSI/tree/main/roborsi/embodied/skills/base/grasp_flat/robotwin
      upstream_license: Apache-2.0
gap:
  requires: {connector: [robot.grasp_frame, robot.describe_gripper, robot.set_grip]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.get_gripper
    - robot.describe_gripper
    - robot.grasp_frame
    - robot.solve_ik
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - robot.open_gripper
    - robot.close_gripper
    - robot.set_grip
  required_inputs:
    object_obb: OrientedBoundingBox
  produces_outputs:
    axis: string
    close_heading_deg: float
    tilt_deg: float
    grasp_position: Vec3
    tip_height: float
    object_width: float
    thickness: float
    jaw_error_deg: float
    wrist_turn_deg: float
    grip_width: float
    held_width: float
    lifted: float
    attempts: int
  exit_conditions:
    grasped: The jaws closed across the slab, the gap within narrow_margin under and full_margin over its width across the jaws (and above empty_margin), and still did after the lift; the hand holds it. `lifted` says how far the hand rose (short when the object is heavy or caught; `detail` says so).
    missed: At least one axis was tried down to the grasp height but every close read empty, too wide or too narrow (not across the slab), the slab slipped out during the lift, a tilted grasp's closing steps were refused, or the descent stopped on the slab's top; the hand is open.
    unreachable: No axis could be tried - the slab is too thin for the fingertips at this tilt (they cannot come below its top), wider than the open hand both ways, every grasp pose failed the IK preview, the hover was refused, or the jaws could not be turned; nothing was closed.
  canonical_scripts:
    - grasp_flat: scripts/grasp_flat.py
  examples:
    - title: Grasp a perceived flat object
      path: examples/canonical_subgraph.json
---

# grasping-flat-objects

Pick a slab lying flat on a table by pinching its body just above the support,
instead of at its box centre.

## The motion

1. Take the slab's frame from its oriented box: centre, top, and its two
   horizontal axes, the shorter first. The support height is `support_z` when
   given, else the 5th height percentile of `object_cloud` (RoboRSI), else the
   box's bottom.
2. Work out the heights over the support. Level (the default): the fingertips
   meet the slab `grasp_fraction` of its thickness up, no lower than
   `min_tip_clearance`; the tool centre may not go under 15 mm (a real YAM
   station refuses lower) and the fingertips reach 10.2 mm below it, so the
   fingertips are at 4.8 mm at the lowest. If they would end less than 1.5 mm
   below the slab's top the axis is refused as too thin, rather than closing on
   air (an 8 mm slab is pinched at 4.8 mm; a 5 mm one is refused level).
3. Preview the grasp pose with `robot.solve_ik` for both wrist solutions; turn
   the jaws with `robot.grasp_frame` (close heading along the axis). Open the
   hand fully (RoboRSI), hover `hover_height` above (straight up first if the
   hand is lower), read the jaws back and finish a free wrist roll on the last
   joint.
4. Come straight down. A descent that stops more than `descent_tolerance` high
   is a miss on that axis (a fingertip on the slab's top).
5. Close and read the gap (`robot.close_gripper`'s `position`, 1.0 = open,
   through `robot.describe_gripper`'s `width_fit`). Within `empty_margin` of
   the closed stop is empty; more than `full_margin` over the slab's width
   across the jaws is something else; more than `narrow_margin` under it means
   a fingertip went over the top or past a corner. Any of these opens the hand,
   rises, and tries the long axis (RoboRSI's single retry).
6. Lift `lift_height` and read the gap again; a slab that slipped out reads
   empty and is a miss.

### The optional tilt (not in RoboRSI; did not hold in the twin)

With `tilt_deg` (or `auto_tilt`) the hand leans about the normal of the plane
its jaws close in, so one fingertip is lower than the other. The jaws open
only to the slab's width plus 12 mm each side (`robot.set_grip`), the hand
comes down with the lower open fingertip 1 mm over the support, then narrows
the jaws to the slab's edges in four steps, following each down so that
fingertip slides along the support into the slab's bottom edge, and closes.
`auto_tilt` takes the tilt whose close line runs from the slab's bottom edge to
1.5 mm under its far top edge, or the steeper tilt the tool-centre floor needs
to bring the lower fingertip to 1 mm (11.3 deg for the 8 mm slab, 17.4 deg for
the 15 mm one, 40 mm across). The tilt lets a thinner slab pass the too-thin
check (a 5 mm slab at 25 deg). But in the station twin every tilted close
pushed the slab 8-41 mm aside and ended narrower than it (below), so leave it
off unless your hand's fingertips are compliant enough to wedge under an edge.

## Inputs

- `object_obb`: the slab's oriented box, world frame, half-extents.
- `object_cloud` (optional): its points, world frame (`{"points": N x 3}` or a
  list of xyz), for the support height.
- `support_z` (optional): the support's height, overriding both.
- `tilt_deg` (0, at most 35), `auto_tilt` (false), `try_long_axis` (true).
- `grasp_fraction` (0.45), `min_tip_clearance` (0.004 m), `hover_height`
  (0.06 m), `lift_height` (0.05 m, 0 skips the lift), `empty_margin`
  (0.003 m), `full_margin` (0.008 m), `narrow_margin` (0.006 m),
  `yaw_tolerance_deg` (5), `roll_limits` (the last joint's range, default the
  YAM's +-120 deg), `reach_tolerance` (0.015 m), `descent_tolerance`
  (0.006 m), `check_reach` (true), `arm_id`.

## Outputs

`axis` (`short` or `long`, the one tried last), `close_heading_deg`,
`tilt_deg`, `grasp_position` (the tool-centre target at the close),
`tip_height` (the lower fingertip's planned height over the support where it
meets the slab), `object_width` (the slab across the jaws), `thickness`,
`jaw_error_deg`, `wrist_turn_deg`, `grip_width` (gap after the close, m; -1
unread), `held_width` (after the lift), `lifted` (m), `attempts` (axes tried
down to the hover). The exit is what the hand measured, not where the slab is:
re-perceive it before relying on its pose in the hand.

## Preconditions

- The hand is empty (it is opened at the start).
- The slab lies flat and is free to be pinched: nothing within half the open
  span (47 mm on a YAM) of its centre along the jaws.
- One of its sides fits the open hand (94 mm on a YAM, less 2 x the finger
  clearance) and it is thicker than about 6.3 mm (the 4.8 mm fingertip floor +
  1.5 mm); thinner slabs exit `unreachable` (slide them to an edge first).

## The constants, and why

- **`GRASP_FRACTION = 0.45`, `MIN_TIP_CLEARANCE = 0.004` m** are RoboRSI's
  `_GRASP_FRAC` and `_MIN_ABOVE_TABLE_M`.
- **`MIN_TOOL_HEIGHT = 0.015` m, `TIP_REACH = 0.0102` m**: a real YAM station
  refuses a tool centre under 14 mm over its table, and the linear_4310's
  fingertips reach 10.2 mm below the tool centre. RoboRSI lowered its own
  descend floor to the table instead, which a YAM station will not do.
- **`MIN_BITE = 0.0015` m**: a choice; less pad than that on the slab's side
  pinches its top edge, if anything.
- **`HOVER_HEIGHT = 0.06` m** (RoboRSI 0.09) and **`LIFT_HEIGHT = 0.05` m**
  (RoboRSI 0.06): kept inside a YAM's reach.
- **`EMPTY_MARGIN = 0.003` m, `FULL_MARGIN = 0.008` m, `NARROW_MARGIN =
  0.006` m**: level pinches in the twin read 0.1 mm under to 1.3 mm over the
  40 mm side; tilted closes that missed read 5-26 mm.
- **`DESCENT_TOLERANCE = 0.006` m**: over the servo's own 4 mm arrival bound;
  the twin's servo reported `reached` 2.5 mm high over the 8 mm slab (a first
  version that held the stop under 60 % of the bite read that as a fingertip
  on the slab and missed all three 8 mm grasps). A fingertip on a slab whose
  top is less than this over the fingertips is therefore not told apart from
  arrival by the pose; the gap check after the close catches it.
- **`TILT_TIP_CLEARANCE = 0.001` m, `FOLLOW_STEPS = 4`, `OPEN_CLEARANCE =
  0.012` m, `MAX_TILT_DEG = 35`**: choices for the tilt.
- **`REACH_TOLERANCE = 0.015` m, `YAW_TOLERANCE_DEG = 5`**: no tighter than
  the connectors' own arrival bounds (servo 4 mm / 3 deg; a real YAM station
  12 mm / 0.04 rad per move).

## Measured in a YAM station twin

RoboSimStudio's YAM station twin, left arm, linear_4310 hand (94 mm span,
fingertips 10.5 mm past the tool centre by the hand's own reading), with two
20 g box slabs (friction 0.8) added beside the cube: 60 x 40 x 8 mm turned
15 deg and 60 x 40 x 15 mm turned -10 deg. Before each grasp a script read the
slab's oriented box from the simulator (standing in for perception) and
passed it in; after it, the slab's own pose was read while the hand still held
it, then the hand set it down, opened and rose. An empty close read 0.0 mm.
Default inputs (level pinch, 50 mm lift):

| slab | tries | exit | jaw gap at close | after the lift | slab rose | slab tilt in the hand |
|---|---|---|---|---|---|---|
| 8 mm | 3 (+1 recheck) | grasped 4/4 | 41.2-41.3 mm | 39.9 mm | 50.8-50.9 mm | 0.4-0.8 deg |
| 15 mm | 3 | grasped 3/3 | 40.7-41.3 mm | 40.4-40.5 mm | 50.6-51.5 mm | 0.2-0.3 deg |

Every grasp closed across the short side on the first axis, the fingertips
planned 4.8 mm (8 mm slab) and 6.75 mm (15 mm slab) over the table, and each
slab came back down within 1.6 mm of where it started. The jaws read 1 mm over
the 40 mm side at the close and settled 0.1 mm under to 0.5 mm over it in the
lift. The twin's IK leaves the wrist roll free; the last-joint turn squared
the jaws to within 0.6 deg (up to 32 deg turned).

With `auto_tilt` the hand did not hold either slab:

| slab | tilt | lift | exit | jaw gap at close | slab pushed |
|---|---|---|---|---|---|
| 15 mm | 17.4 deg | none | missed (25.5 mm, not across it) | 25.5-25.7 mm | 7.8-8.8 mm |
| 8 mm | 11.3 deg | none | missed (4.9 mm) | 4.9-5.1 mm | 17.3-17.5 mm |
| 8 mm, 2 runs | 11.3 / 7.4 deg, both axes | 50 mm | missed (slipped out) | 0.0-12.5 mm | 33-41 mm |
| 15 mm, first run | 17.4 / 11.8 deg, both axes | 50 mm | missed (slipped out) | 0.0-34.0 mm | 27 mm |

The lower fingertip pushed the slab aside along the table rather than
wedging under its edge, and the jaws closed past it. Before `narrow_margin`
was added the two no-lift runs exited `grasped` (the gap was neither empty nor
too wide); with it they exit `missed` (the recheck run above). The first
run also held the descent check under 60 % of the bite, which only changed the
level 8 mm grasps (see `DESCENT_TOLERANCE`).

## Provenance

Adapted from RoboRSI's `grasp_flat/robotwin` policy (Apache-2.0), which its
authors marked EXPERIMENTAL and unvalidated. Kept: the oriented-box footprint,
the support from the cloud's 5th height percentile, the fingertips at 0.45 of
the thickness (at least 4 mm over the support), the wide-open level pinch across
the short side, one retry across the long side, the lift check. Changed: a
world-frame oriented box in, not a Grounded-SAM mask read inside the simulator;
the grasp height is the tool centre's, floored at 15 mm over the support, and
an axis whose fingertips would stay above the slab is refused (RoboRSI lowered
its descend floor to the table); each axis is IK-previewed and the jaws turned
with `robot.grasp_frame`, finished on the last joint when the solver leaves the
roll free; RoboRSI's `is_holding` (simulator contact) and
`verify_holding_visual` (a VLM) are replaced by the jaw gap against the slab's
width after the close and after the lift; refused moves route instead of
raising. Added: the optional tilt with its stepped close (not in RoboRSI).
