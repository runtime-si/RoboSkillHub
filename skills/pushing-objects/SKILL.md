---
name: pushing-objects
description: Pushes a loose object across its support toward a goal point with a
  closed hand, never grasping it - hovers behind the object, comes straight
  down to a contact height on its side, pushes along a straight line, lifts
  away and opens. Use when an object must be slid rather than lifted (out of a
  corner, away from a wall or a neighbour so it can be grasped, too flat or
  wide for the gripper, or onto a mark), with its OBB from a perception skill
  and the hand empty.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [push, nonprehensile, slide, recovery]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/nssmd/RoboRSI/tree/main/roborsi/embodied/skills/base/push_object/libero
      upstream_license: Apache-2.0
gap:
  requires: {connector: [robot.grasp_frame]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.close_gripper
    - robot.open_gripper
    - robot.grasp_frame
  required_inputs:
    target_obb: OrientedBoundingBox
    goal_position: Vec3
  produces_outputs:
    push_distance: float
    achieved: float
    expected_center: Vec3
  exit_conditions:
    pushed: The hand travelled past the contact point toward the goal by at least min(80 mm, 0.75 x the asked push), lifted away and opened; the object should now sit near expected_center.
    not_pushed: The goal was under 30 mm from the object's centre, or the push ran but the hand travelled short (blocked, or the move refused part way), or the retract failed; `detail` says which.
    unreachable: The hover above and behind the object was refused or ended off with every rotation tried, or the descent stopped more than reach_tolerance from the contact height; nothing was pushed.
  canonical_scripts:
    - push_object: scripts/push_object.py
  examples:
    - title: Push a perceived object toward a goal point
      path: examples/canonical_subgraph.json
---

# pushing-objects

Slide an object instead of lifting it.

## The motion

1. Close the hand (it pushes with its closed fingers).
2. Hover `hover_clearance` above a point `standoff` behind the object's back
   face, along the push direction. The hand goes top-down with its closed
   fingers side by side across the push (`robot.grasp_frame`, both wrist
   solutions), a broad face that keeps the object on the line; if neither
   reaches, it keeps its own rotation. `keep_rotation=true` tries its own
   rotation first.
3. Come straight down to the contact height, still behind the object. A
   descent that stops short but within `reach_tolerance` (the closed
   fingertips met the support first) is accepted, and the push runs at the
   height the hand reached.
4. Push along one straight line through the contact point to `push_distance`
   past it.
5. Lift `hover_clearance` and open.

The back face comes from the OBB's corners projected on the push direction,
so a box pushed corner-first is met at its corner.

## Inputs

- `target_obb`: the object's oriented box, world frame (half-extents).
- `goal_position`: where its centre should go; only x and y are used. The push
  is `min(distance, max_distance)` along the line from the centre to it.
- `max_distance` (0.20 m), `standoff` (0.06 m), `hover_clearance` (0.08 m),
  `contact_fraction` (0.5), `min_contact_height` (0.015 m),
  `reach_tolerance` (0.02 m): see the constants.
- `keep_rotation` (false): try the hand's own rotation before the top-down
  ones (RoboRSI's order, for a policy that must not see the wrist turn).

## Required end states

- `pushed`: the hand's measured travel past the contact point, along the push,
  is at least `min(0.08, 0.75 x push_distance)`, and it lifted away.
  `achieved` is that travel and `expected_center` the centre moved by it.
  This is the hand's travel, not the object's: re-perceive the object
  (a perception skill, or verifying-placement) before relying on where it went.
- `not_pushed`: nothing to do (goal too close), or the push came up short or
  the retract failed. The hand is open either way.
- `unreachable`: the hover failed, or the descent stopped further than
  `reach_tolerance` off; the hand is closed and stands wherever the failed
  move left it.

## The constants, and why

- **`MAX_DISTANCE = 0.20` m, `STANDOFF = 0.06` m, `HOVER_CLEARANCE = 0.08` m,
  `EDGE_PADDING = 3 mm`, the 0.09 m hover backoff cap and the
  `min(80 mm, 0.75 x push)` success rule** are RoboRSI's, tuned on its LIBERO
  push tasks.
- **Fingers across the push first** (RoboRSI keeps the hand's rotation first).
  Measured in a YAM station twin, a 30 mm cube pushed 80 mm:

  | push | hand's own rotation | fingers across (default) |
  |---|---|---|
  | +x | 85 mm along, 0.07 mm sideways | 85 mm along, 0.5 mm sideways, 5 mm from goal |
  | -y | 86 mm along, 14.5 mm sideways | 78 mm along, 11.0 mm sideways, 11 mm from goal |
  | +y | not run | 77 mm along, 4.0 mm sideways, 5 mm from goal |

  The cube never tipped (height change under 0.01 mm). Sideways drift of
  this size is why the exit reports the hand's travel and the object must be
  re-perceived.
- **`CONTACT_FRACTION = 0.5`**: push through the middle of the object's height;
  lower tips it less but drags the fingers on the support.
- **`MIN_CONTACT_HEIGHT = 0.015` m**: the tool centre's floor over the support.
  A real YAM station refuses a tool centre under 14 mm over its calibrated
  table, so a short object is pushed at 15 mm rather than at its middle.
- **`REACH_TOLERANCE = 0.02` m**: arrival bound for the hover and the lowered
  pose, inside RoboRSI's 4-5 cm and loose enough for a servo's 4 mm ball plus
  a real arm's sag. In a YAM station twin, with the hand's own rotation, a
  descent to a tool centre 15 mm over the table stopped 5.6 mm short (the
  closed fingertips reach about 10 mm below the tool centre); the push then
  ran from there and the cube still moved 86 mm.

## Provenance

Adapted from RoboRSI's `push_object/libero` policy (Apache-2.0): world-frame
OBB and goal in place of image pixels, robot.* connector tools, a vertical
descent behind the object, a contact-height floor, fingers across the push
first, and move refusals routed rather than raised.
