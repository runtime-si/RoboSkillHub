---
name: approaching-above-a-target
description: Moves the hand to hover a set clearance above a target's oriented
  box, rising in place first and keeping the hand's current rotation, then
  checks from a fresh pose reading that it arrived. Use when a learned policy
  is about to take over and should start near the object from the rotation it
  was trained with, or whenever a graph needs the hand parked above an object
  before a step of its own; needs the target's OBB from a perception skill.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: motion
  tags: [motion, approach, handoff, policy]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/graph-robots/graph-as-policy/tree/main/examples/steered_policy/graph_loop/scripts/approach
      upstream_license: Apache-2.0
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose
  required_inputs:
    target_obb: OrientedBoundingBox
  produces_outputs:
    hover_pose: Se3Pose
    position_error: float
    rotation_error_deg: float
  exit_conditions:
    above: The tool centre stands within reach_tolerance (default 10 mm) of the point clearance (default 0.12 m) above the target's top, its rotation within rot_tolerance_deg (default 4 deg) of the one asked for.
    unreachable: A move raised or reported that it did not arrive, or the hand ended further off in position or rotation; `detail` says which.
  canonical_scripts:
    - approach_above_target: scripts/approach_above_target.py
  examples:
    - title: Hover above a perceived target before a policy takes over
      path: examples/canonical_subgraph.json
---

# approaching-above-a-target

Park the hand above an object: rise in place to the hover height, move across
to above the target's centre, and confirm the arrival from the arm's own pose.

## Why keep the rotation

A learned policy's proprioception at hand-off should look like the start of
its training episodes. Turning the wrist to a top-down quaternion first (what
the grasp skills' own `approach_above` scripts do) pushes that state out of
distribution. This skill keeps the hand's current rotation unless `rotation`
is given.

## Inputs

- `target_obb`: the target's oriented box in the world frame (`extent` holds
  half-extents, so the top is `center.z + extent.z`).
- `clearance` (default 0.12 m): tool-centre height over the box's top.
- `min_z`: an optional floor on the hover height, in the world frame.
- `rotation`: a quaternion to hover with instead of the current one.
- `arm_id`, `tolerance`, `max_steps`: passed on only when set (`arm_id` when
  nonzero). `tolerance` and `max_steps` are gap's built-in connector's
  convergence knobs; a real station's motion server may not take them.
- `reach_tolerance` (default 10 mm), `rot_tolerance_deg` (default 4 deg): the
  arrival bounds.

## Required end states

- `above`: a fresh `robot.get_ee_pose` lies within `reach_tolerance` of
  `hover_pose` and within `rot_tolerance_deg` of its rotation.
  `position_error` and `rotation_error_deg` report both.
- `unreachable`: a `robot.go_to_pose` raised, or replied that it did not
  arrive (RoboSimStudio's servo routes an unreachable pose as
  `{"outcome": "unreachable"}` instead of raising), or the hand ended further
  off. Both errors are -1 when a move failed.

## The constants, and why

- **`CLEARANCE = 0.12` m**: graph-as-policy's steered-policy hand-off value
  (`obb_clearance`), clear of open fingers above a tabletop object.
- **`REACH_TOLERANCE = 0.01` m**: inside a real YAM station's 12 mm bound,
  past which its motion server fails a move.
- **`ROT_TOLERANCE_DEG = 4.0`**: above RoboSimStudio's IK servo arrival ball
  (3 deg) and a real YAM station's per-move bound (0.04 rad, 2.3 deg), so a
  move the connector accepted is not failed here. Measured on a YAM station
  twin in RoboSimStudio: the hover ended 1.4 mm and 2.15 deg off, and passing
  `tolerance` did not change the rotation (that servo applies it to position
  only).
