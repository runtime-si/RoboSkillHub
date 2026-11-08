---
name: following-a-visual-plan
description: Follows a visual plan from robobrain.trace or molmo-motion.forecast
  (the shared visual-plan schema) the way those models can be trusted - the
  plan's endpoint pixel becomes a target lifted with the measured depth and the
  calibrated camera (not the model's depth), checked (on the table, in reach by
  IK preview, optionally against a pointing model for the target phrase), and
  geometry decides the motion - rise, travel at a height that clears the
  measured scene (optionally through the plan's path), hover above the target,
  descend to the caller's goal height or a clearance over the object's measured
  top. Optionally asks the planner once more from a fresh frame when a check
  fails. Use when a model proposes where something should go ("place the cube
  in the box", "move the cable end to the ring") and the hand should be brought
  there, above or into the target, before a placing, inserting or releasing step.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: motion
  tags: [motion, planning, visual-plan, vlm, trajectory, approach]
  hub:
    stage: propose
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, cable-routing, insertion]
    provenance:
      authors: [runtime-si]
      paper: https://arxiv.org/abs/2601.14352
gap:
  requires: {connector: [robot.grasp_frame, robot.describe_workspace]}
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.solve_ik
    - robot.grasp_frame
    - robot.describe_workspace
    - robobrain.trace
    - molmo-motion.forecast
    - gemini-er.point
    - robobrain.point
    - locate-anything.point
  required_inputs:
    instruction: str
  produces_outputs:
    target_point: Vec3
    goal_pose: Se3Pose
    hover_pose: Se3Pose
    approach_direction: Vec3
    target_obb: OrientedBoundingBox
    final_position: Vec3
    position_error: float
  exit_conditions:
    proposed: (propose only) The plan's endpoint lifted to a target on the table, in reach (IK preview of the hover and the goal), and, with target_phrase, within cross_check_tolerance of the pointing model's point; `proposal` holds the goal, hover, travel height and waypoints. Nothing moved.
    no_plan: (propose only) No plan was given and no instruction to ask for one, the planner raised or returned no track, or the track's endpoint is within 12 px of its start (no motion proposed - what MolmoMotion answers for a still scene); with replan, both plans. Nothing moved.
    plan_rejected: (propose only) The endpoint is outside the image or has no depth, is off the table (its surface under the table plane, past workspace_bounds) or implausibly high over it, the route needs a travel height over the ceiling, no rotation reaches the hover and the goal (IK preview), or the cross-check found no target or disagrees; `detail` says which. With replan, both plans failed. Nothing moved.
    arrived: (execute only) Every leg was accepted and a fresh pose reading puts the tool centre within reach_tolerance (15 mm) of the goal above the target. What the hand did; re-perceive before relying on what is under it.
    unreachable: (execute only) A leg (rise, waypoint, hover, descend) raised or reported it did not arrive (a servo timeout counts only when the hand is further than reach_tolerance from the leg's end), or the hand ended further than reach_tolerance from the goal; `detail` names the leg.
  canonical_scripts:
    - propose_target_from_plan: scripts/propose_target_from_plan.py
    - follow_proposal: scripts/follow_proposal.py
  examples:
    - title: Ask RoboBrain for a plan and bring the hand above its target
      path: examples/canonical_subgraph.json
---

# following-a-visual-plan

A trajectory model's plan is good in the image and poor in 3D. On real YAM
station frames RoboBrain 2.5's traces ended on the right target 12/12 times
(cube into the box), but their heights were a generic lift-and-arc ending
14-28 cm over the target, and the model's own depth read 25-45 % short
(tools/robobrain, Measured). MolmoMotion forecasts only motion already under
way: from a still frame it returned none. So this skill keeps only the image
part of a plan and lets measured geometry do the rest.

Two scripts with a clean hand-off:

- **`propose_target_from_plan`** (stage propose): `plan` or `instruction`
  (+ `observation`) -> `proposal`, `target_point`, `goal_pose`, `hover_pose`,
  `waypoints`, `approach_direction`, `target_obb`. Routes `proposed` /
  `plan_rejected` / `no_plan`. Moves nothing.
- **`follow_proposal`** (stage execute): `proposal` -> routes `arrived` /
  `unreachable`.

```text
propose_target_from_plan
    proposed      -> [previewing-target-overlay: goal_pose + intent] -> follow_proposal
    plan_rejected -> re-perceive / another planner / ask a human
    no_plan       -> start the motion another way (a policy, executing-keypoint-motion)
follow_proposal
    arrived       -> placing-with-contact-descent (place_position = target_point) / open / insert
    unreachable   -> recover
```

`target_obb` (a column from the table to the measured top at the target) is
in approaching-above-a-target's input shape, so a graph that wants that
skill's hover (rise in place, keep the rotation, 12 cm over the top) can use
it in place of `follow_proposal`.

## The plan

The visual-plan schema (`tools/molmo-motion/SKILL.md`): `found`, `tracks[i]`
with `query_pixel` (the start) and `pixels` (`[F][u, v]`, the keypoints after
it, in order), plus 3D fields this skill ignores. Pass a plan computed
elsewhere as `plan`, or let the skill ask `planner` (default
`robobrain.trace`; `molmo-motion.forecast` or any tool returning the schema)
with the camera's RGB, depth, intrinsics and pose, the `instruction`, and
`start_pixel` as `start_point` (`query_points` for a `*.forecast` tool, which
requires it). `planner_args` adds or overrides arguments (`mode="object"`).

## Proposing

1. **Endpoint**: the track's last pixel. Within `MIN_PLAN_PX` (12 px) of the
   start is `no_plan`; outside the image is `plan_rejected`.
2. **Target**: the endpoint lifted with the median measured depth of a 7 x 7
   window and the camera's intrinsics and camera-to-world pose (the surface
   the plan points at: a box's floor, the board under a ring).
3. **On the table**: the table plane is `table_z`, else
   `robot.describe_workspace`'s `surface_z`, else the depth image's fullest
   4 mm height bin. A target more than `on_table_tolerance` (2 cm) under it
   (the floor past the table's edge) or `max_target_height` (0.30 m) over it,
   or outside `workspace_bounds` (`{"x": [lo, hi], "y": [lo, hi]}`), is
   rejected.
4. **Object top**: the 5th-highest measured point within `top_radius` (6 cm)
   of the target, the hand itself excluded (a box's rim, not its floor).
5. **Goal height**: `goal_height` (world z of the tool centre) when given,
   else the top + `goal_clearance` (3 cm); never under 15 mm over the table.
6. **Approach direction**: the plan's last segment in the image, carried to
   the target's height: the side the plan comes from (`approach_direction`,
   a horizontal unit vector). With `approach_offset` > 0 the route passes
   that far back along it before the hover.
7. **Travel height**: at least `travel_height` (12 cm) over the table and
   `travel_clearance` (8 cm) over the object's top, then raised until it
   clears by `travel_clearance` everything the depth image shows in a 4 cm
   corridor under every travel leg (hand -> waypoints -> target), the hand
   excluded. Over `max_travel_height` (0.35 m) is rejected.
8. **Waypoints** (`follow_path`, default off): every path pixel but the
   last, carried along its camera ray to the travel height (where a point at
   that height would be seen at that pixel), closer than 3 cm to the previous
   or to the target dropped, unreachable ones (IK preview) dropped.
9. **Reach**: `robot.solve_ik` previews the hover and the goal with the
   hand's current rotation, then `robot.grasp_frame`'s top-down rotation and
   its flip (`allow_top_down`); the first that reaches both is used. None is
   `plan_rejected`. A connector without IK preview skips this.
10. **Cross-check** (`target_phrase`): `cross_check_tool` (`gemini-er.point`;
    `locate-anything.point`, `robobrain.point` take the same call) points at
    the phrase on the same frame; its nearest point, lifted the same way,
    must be within `cross_check_tolerance` (5 cm) of the target. No point is
    `plan_rejected`; a tool that raised is noted in `detail` and does not
    block.
11. **Replan** (`replan`, default off; needs `instruction`): when the first
    plan routes `no_plan` or `plan_rejected`, take a fresh
    `robot.get_observation` and ask the planner once more; `detail` lists
    both. RoboBrain decodes greedily, so the same frame gives the same plan:
    a replan helps after the scene or the hand moved, or when the first plan
    was computed earlier and passed in.

## Following

Rise in place to the travel height (`robot.go_to_pose`, as
approaching-above-a-target does), then straight-line legs
(`robot.go_to_pose_cartesian`) through the waypoints to the hover above the
target and straight down to the goal, all with the chosen rotation. Every
refused move (raised, or `reached: false` / an `outcome` other than
`reached`) routes `unreachable` with its leg, except a servo `timeout` that
left the hand within `reach_tolerance` of the leg's end (read back), which
continues and is named in `detail` (the YAM twin's servo timed out 3.8 mm
over the goal on some descents); arrival is judged from a fresh
`robot.get_ee_pose` against `reach_tolerance` (15 mm, no tighter than a real
station's 12 mm per move). `tolerance` / `max_steps` go to the free move only
when set; `arm_id` only when nonzero.

## The constants, and why

- **`MIN_PLAN_PX = 12`**: MolmoMotion moved a still cube or cable tip 0-3.5 mm
  (~0-2 px at 0.9 m); RoboBrain's plans moved 50-150 px.
- **`ON_TABLE_TOLERANCE = 0.02`, `MAX_TARGET_HEIGHT = 0.30`** m: a few times
  the calibrated camera's 5.4 mm RMS; anything higher is an arm or a held
  object, not a place.
- **`TOP_RADIUS = 0.06`** m: reaches a 100 mm box's walls from its centre.
  **`TOP_RANK = 5`**: a 2.3 mm rim is a few per cent of that disc, so a
  percentile misses it; the 5th-highest of every 2nd pixel ignores single
  noisy edge pixels.
- **`GOAL_CLEARANCE = 0.03`** m: a 30 mm part under the hand clears the rim.
- **`TRAVEL_HEIGHT = 0.12`** m: approaching-above-a-target's and
  executing-keypoint-motion's hover clearance. **`TRAVEL_CLEARANCE = 0.08`** m:
  placing-with-contact-descent's approach height. **`MAX_TRAVEL_HEIGHT =
  0.35`** m: RoboBrain's own arcs rose to 0.22-0.33 m.
- **`CORRIDOR_RADIUS = 0.04`** m: half an open linear_4310's 94 mm span.
  **`HAND_RADIUS = 0.08`** m (from 6 cm under the tool centre up): the hand
  and fingers in the depth image.
- **`CROSS_CHECK_TOLERANCE = 0.05`** m: half the box; RoboBrain's cable
  endpoints missed a ring by 20-35 px (~4-7 cm at 0.9 m), which this rejects.
- **`MIN_TOOL_HEIGHT = 0.015`** m: a real YAM station refuses a tool centre
  under 14 mm over its table.
- **`REACH_TOLERANCE = 0.015`** m: placing-with-contact-descent's.

## Measured

MEASURED_PLACEHOLDER

## Provenance

Written for the hub (no upstream code). The design follows the live checks of
the hub's `robobrain` and `molmo-motion` tools on real YAM station frames
(2026-11-07): RoboBrain 2.5 (BAAI, arXiv:2601.14352, Apache-2.0 weights)
places the endpoint right in the image but not in depth or height;
MolmoMotion continues motion and does not invent it.
