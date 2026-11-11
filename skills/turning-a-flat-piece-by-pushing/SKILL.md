---
name: turning-a-flat-piece-by-pushing
description: Turns a flat part lying on the table (a jigsaw piece) to a target
  yaw without gripping it - the closed fingertips push one of its body corners
  tangentially in steps of at most 10 degrees, and after every push the wrist
  camera looks straight down, takes everything 1 mm over the table as the
  piece and fits the piece's known mask over rotations to measure its yaw;
  stops within 5 degrees or after 12 pushes. Use when a piece must be picked
  already at its slot's yaw because a held turn loses it (an edge grip of a
  2 mm piece does not hold against turning), before the pick.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [jigsaw, puzzle, pushing, reorientation, closed-loop, wrist-camera]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw, reorientation]
    provenance:
      authors: [runtime-si]
      paper: "M. T. Mason, Mechanics of Robotic Manipulation, MIT Press 2001, ch. 10 (pushing; idea only)"
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.close_gripper
  required_inputs:
    piece_mask: Mask
    metres_per_pixel: float
    position: Vec3
    target_yaw_deg: float
  produces_outputs:
    yaw_deg: float
    error_deg: float
    pushes: int
  exit_conditions:
    turned: The measured yaw is within 5 deg of the target; position is the piece's measured centre.
    not_turned: Still more than 5 deg off after 12 pushes; yaw_deg and history say where it is.
    not_seen: No wrist camera with depth, an empty mask, or the piece is not under the camera.
    unreachable: A push move was refused.
  canonical_scripts:
    - push_turn: scripts/push_turn.py
  examples:
    - title: Turn a piece to its slot's yaw before picking it
      path: examples/canonical_subgraph.json
---

# turning-a-flat-piece-by-pushing

In the YAM station twin a 2 mm jigsaw piece pinched by its two edges is
flung out of the jaws by a held wrist turn (0 / 14), and slips in them when
twisted on the table (2 / 16 within 10 degrees; docs/jigsaw-e2e). This skill
turns the piece where it lies, by pushing, and closes the loop on the wrist
camera.

## How

1. **Look.** The wrist camera is put straight above the piece at 15 cm.
   Its depth is lifted to the world; within 35 mm of the piece the table is
   the commonest height, and points 1-6 mm over it are the piece.
2. **Yaw.** The piece's mask (any top-down mask of it, its own frame at yaw
   0) is turned in 2 degree steps (a full turn the first time, then +-30
   degrees round the expected yaw) and 0.25 degree steps round the best;
   the turn whose overlap with the footprint is largest is the yaw.
3. **Push.** Within 5 degrees of the target: done. Else, with the jaws
   closed, the blade tips 0.5 mm over the table start 6 mm outside a body
   corner and move tangentially (perpendicular to the corner's radius) by
   the radius times 0.8 of the remaining turn, at most 10 degrees' worth.
   Then look again.

## Measured

See the pull request (YAM twin, 8 trials).

## Limits

- One corner is always pushed; a piece near the arm's reach limit may need
  another corner (not chosen here).
- The piece also slides when pushed; the measured centre is returned.
- The yaw has the mask's symmetry: a nearly symmetric piece may be read
  90 or 180 degrees off.
