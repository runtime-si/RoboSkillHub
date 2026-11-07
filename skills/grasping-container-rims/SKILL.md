---
name: grasping-container-rims
description: Grasps a thin-walled open container (box, bin, tray, cup, bowl) by
  its rim - picks a wall midpoint the arm can reach, turns the open jaws across
  the wall (one finger inside the mouth, one outside), hovers above, comes
  straight down until the pads straddle the wall below the rim top, closes,
  checks the jaw gap reads about the wall's thickness (not empty, not wider),
  and lifts a little; tries the other walls when one misses. Use when an open
  container must be picked or held and a body grasp would close on the empty
  inside (or the container is wider than the open hand), with its oriented box
  from a perception skill and the hand empty.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [grasp, rim, container, thin-wall, bin, cup, bowl]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/nssmd/RoboRSI/tree/main/roborsi/embodied/skills/base/grasp_rim/robotwin
      upstream_license: Apache-2.0
gap:
  requires: {connector: [robot.grasp_frame, robot.describe_gripper]}
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
  required_inputs:
    container_obb: OrientedBoundingBox
  produces_outputs:
    side: string
    grasp_position: Vec3
    close_heading_deg: float
    jaw_error_deg: float
    wrist_turn_deg: float
    grip_width: float
    held_width: float
    wall_thickness: float
    lifted: float
    attempts: int
  exit_conditions:
    grasped: The jaws closed across a wall with the gap above empty_margin and at most wall_thickness + full_margin (max_wall when the thickness is unknown), and still did after the lift; the hand holds the rim. `lifted` says how far the hand rose (short when the container is heavy or fixed; `detail` says so).
    missed: At least one wall was tried down to the grasp height but every close read empty or too wide, the wall slipped out during the lift, or the descent stopped on the rim; the hand is open.
    unreachable: No wall could be tried - the container is too shallow or too narrow for the open hand, every grasp pose failed the IK preview, the hover was refused, or the jaws could not be turned across the wall; nothing was closed.
  canonical_scripts:
    - grasp_rim: scripts/grasp_rim.py
  examples:
    - title: Grasp a perceived open box by its rim
      path: examples/canonical_subgraph.json
---

# grasping-container-rims

Pick an open container by one of its walls instead of its hollow body.

## The motion

1. Open the hand. Take the container's rim frame from its oriented box: the
   rim top (the box's highest corner), its bottom, its centre, and its two
   horizontal axes. With `container_cloud` (world points of the container, e.g.
   from geometry's mask-to-points), the rim top, the opening's centre and its
   half-extents come from the cloud's top 20 mm band instead (RoboRSI's
   `container_opening`).
2. The candidate rim points are the four wall midpoints, half a wall in from
   the outer face. They are tried nearest to the hand first (`side` names one:
   `+x`, `-x`, `+y`, `-y`, the box's own axes), each previewed with
   `robot.solve_ik` at the grasp pose for both wrist solutions. A point whose
   open inner jaw would reach the opposite wall is skipped.
3. Turn the jaws across the wall: `robot.grasp_frame` with the close heading
   along the wall's outward normal, top-down. Go to a hover `hover_height`
   above the grasp pose (straight up first if the hand is lower). Read the
   rotation back; if the jaws are more than `yaw_tolerance_deg` off the wall's
   normal (an arm whose solver leaves the wrist roll free reaches the pose
   unturned), turn the rest on the last joint (`robot.move_to_joints`) and
   re-centre over the hover.
4. Come straight down until the tool centre is `grip_depth` below the rim top
   (floored so the fingertips stay `3 mm` over the container's bottom and the
   tool centre 15 mm over it). A descent that stops more than
   `descent_tolerance` high (a finger on the rim) is a miss on that wall.
5. Close and read the jaw gap: `robot.close_gripper`'s `position` (or
   `robot.get_gripper`), 1.0 = open, through `robot.describe_gripper`'s
   `width_fit`. Within `empty_margin` of the closed stop is a close on nothing;
   above `wall_thickness + full_margin` (or `max_wall`) is something else.
   Either opens the hand, rises to the hover, and tries the next wall.
6. Lift `lift_height` and read the gap again: a wall that slipped out reads
   empty and is a miss. A lift that rises short with the gap still on the wall
   (a heavy or fixed container) is still `grasped`, said in `detail`.

## Inputs

- `container_obb`: the container's oriented box, world frame, half-extents.
- `container_cloud` (optional): its points, world frame (`{"points": N x 3}`
  or a list of xyz).
- `wall_thickness` (unknown): when given, the rim point sits half of it in from
  the outer face and the gap must read at most it + `full_margin`.
- `side` (`auto`), `try_other_sides` (true).
- `grip_depth` (0.012 m), `hover_height` (0.06 m), `lift_height` (0.03 m,
  0 skips the lift), `empty_margin` (0.0008 m), `full_margin` (0.006 m),
  `max_wall` (0.012 m), `yaw_tolerance_deg` (5), `roll_limits`
  (the last joint's range, default the YAM's +-120 deg), `reach_tolerance`
  (0.015 m), `descent_tolerance` (0.008 m), `check_reach` (true), `arm_id`.

## Outputs

`side` (the wall tried last), `grasp_position` (its tool-centre target),
`close_heading_deg`, `jaw_error_deg` (the jaws' read-back heading off the
wall's normal), `wrist_turn_deg` (turned on the last joint), `grip_width` (gap after the close, m; -1 unread),
`held_width` (after the lift), `wall_thickness` (as given, -1 unknown),
`lifted` (m), `attempts` (walls tried down to the hover). The exit is what the
hand measured, not where the container is: re-perceive it before relying on
having moved it.

## Preconditions

- The hand is empty (it is opened at the start).
- The container stands upright and its walls are thinner than the open hand
  minus twice a finger; the space outside the chosen wall is free for the
  outer finger (half the open span, 47 mm on a YAM).

## The constants, and why

- **`GRIP_DEPTH = 0.012` m** is RoboRSI's `_BELOW_RIM_M`. **`HOVER_HEIGHT`**
  0.06 m (RoboRSI 0.10) and **`LIFT_HEIGHT`** 0.03 m (RoboRSI 0.07) are
  shorter so the hover and lift stay inside a YAM's reach over a table box.
- **`EMPTY_MARGIN = 0.0008` m**: a wall can be thinner than the hub's usual
  3 mm empty margin. In the YAM station twin an empty close read 0.0 mm; see
  the measurements below for the wall. A real YAM station's empty close has
  read 1.8 mm (verifying-grasps), so on hardware raise `empty_margin` to about
  2 mm and expect walls under ~3 mm to be indistinguishable from empty by the
  gap alone.
- **`FULL_MARGIN = 0.006` m, `MAX_WALL = 0.012` m**: a choice; a close on a
  corner or a handle reads far wider.
- **`DESCENT_TOLERANCE = 0.008` m**: a fingertip on the rim top stops the tool
  centre 22 mm high (grip depth plus the 10 mm fingertip reach).
- **`YAW_TOLERANCE_DEG = 5`, `REACH_TOLERANCE = 0.015` m**: no tighter than
  the connectors' own arrival bounds (servo 3 deg / 4 mm, a real YAM station
  0.04 rad / 12 mm per move).
- **`RIM_BAND = 0.02` m** and the 90th percentiles of the cloud refinement are
  RoboRSI's `container_opening`.

## Measured in a YAM station twin

RoboSimStudio's YAM station twin, left arm, linear_4310 hand (94 mm span,
fingertips 10.2 mm past the tool centre), the open box 100 x 100 x 40 mm with
2.3 mm walls, turned -94.4 deg, welded to the table. A setup script built the
box's oriented box and wall thickness from `sim.get_region_obb('in_box')`
(the inside) and the walls' own colliders (`sim.get_collision_world`: 2.3 mm,
the yaw, the rim top); the skill ran once per wall with `side` set and
`try_other_sides=false`. An empty close in the air read 0.0 mm.

With `lift_height=0` (the box cannot be lifted):

| wall | exit | gap after the close | wall | jaws off the wall's normal | last-joint turn |
|---|---|---|---|---|---|
| +x | grasped | 4.54 mm | 2.3 mm | -0.01 deg | +37.5 deg |
| -x | grasped | 4.56 mm | 2.3 mm | +0.03 deg | -10.1 deg |
| +y | grasped | 4.36 mm | 2.3 mm | 0.00 deg | -87.8 deg |
| -y | grasped | 4.01 mm | 2.3 mm | -0.02 deg | +6.7 deg |

Every wall was reachable, the tool centre stopped within 0.6 mm of its target,
and a second reading just before the hand opened agreed within 0.04 mm (the
closure held). The gap reads 1.7-2.3 mm over the wall: it is the blade tips'
separation from the finger travel, not the pads' at the contact, which is why
`full_margin` is 6 mm. The twin's IK leaves the wrist roll free, so every
hover arrived with the jaws turned wrong by 7-88 deg and the last-joint turn
did the rest. A negative run with the box moved 70 mm out past its -x wall
closed on air (0.0 mm) and exited `missed`.

With the default `lift_height` (30 mm) on the welded box, the arm rose 28 mm
every time and the pads slid off the wall's top: the gap read 0.0-0.7 mm
after the lift and all four walls exited `missed`. The twin's servo pulls the
fingers off a fixed wall rather than stalling, so a held lift has not been
measured; on a free container the lift is the test that the grip carries it.

## Provenance

Adapted from RoboRSI's `grasp_rim/robotwin` policy and its
`_perception.container_opening` (Apache-2.0). RoboRSI was itself unvalidated
("EXPERIMENTAL"). Changed: a world-frame oriented box (optionally refined by a
cloud's rim band along the box's axes) instead of a Grounded-SAM mask read
inside the simulator; all four wall midpoints, nearest the hand first and
IK-previewed, instead of the arm's x-side and its opposite; jaws turned with
`robot.grasp_frame` and finished on the last joint when the solver leaves the
roll free; the grip depth is the tool centre's and floored over the bottom;
RoboRSI's `is_holding` (simulator contact) and `verify_holding_visual` (a VLM)
are replaced by the jaw gap measured against the wall thickness after the
close and after the lift; refused moves route instead of raising.
