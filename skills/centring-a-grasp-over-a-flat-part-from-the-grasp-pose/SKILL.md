---
name: centring-a-grasp-over-a-flat-part-from-the-grasp-pose
description: Corrects a planned grasp on a flat part of known size (a domino,
  a tile, a plate) by looking again with the wrist RGB-D camera from the
  grasp's own pose, lifted a few centimetres - the hand already tilted and
  turned as it will close - finding the part by depth alone (what stands a
  part's thickness over a plane fitted to the support in that same frame),
  shifting the grasp onto the part's middle and turning it square to the
  part's long side, and looking once more. No model, no colour. Use when a
  grasp planned from another view (overhead, or a wrist look from a different
  arm pose) comes down a few millimetres to a centimetre and a half to one
  side, so that one jaw meets the part first and pushes it; needs the part's
  box and the planned grasp pose, and the arm free to stand over the part.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [grasping, wrist-camera, eye-in-hand, depth, cpu, dominoes, flat-parts]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tuned_on: [yam_i2rt]
    tasks: [pick-place, dominoes, assembly]
    provenance:
      authors: [runtime-si]
      code: yam_dominos skills/skills.py (_centre_lean, _centre_flat) and skills/perception.py (lying_depth), 2026-11
    notes: Draft from yam_dominos. Measured there on the real left YAM; this script has not been run through a hub connector.
gap:
  allowed_tools:
    - robot.get_observation
    - robot.solve_ik
    - robot.go_to_pose
  required_inputs:
    target_obb: OrientedBoundingBox
    grasp_pose: Se3Pose
  produces_outputs:
    grasp_pose_centred: Se3Pose
    pregrasp_pose: Se3Pose
    offset: Vec3
    turn_deg: float
    residual_mm: float
    detail: string
  exit_conditions:
    centred: The last look from the pregrasp pose has the part's middle within tolerance (1.5 mm) of where the grasp is aimed and its long side within 2.5 deg of the way the grasp was planned for, or within accept (3 mm) after the last correction; the hand stands at pregrasp_pose, standoff (5 cm) above grasp_pose_centred.
    off_centre: After max_looks (3) looks the part's middle is still more than accept from the grasp; the hand stands at the last pregrasp pose and grasp_pose_centred carries the corrections made.
    not_seen: No wrist camera with depth, the support not in view, or no blob of the part's size within 3 cm of where the grasp is aimed (cut by the picture's edge, hidden by the fingers, another size); the poses are the plan's, plus any correction already made.
    unreachable: The pregrasp pose, or a corrected one, failed the IK preview or the move; nothing further was tried.
  canonical_scripts:
    - centre_grasp: scripts/centre_grasp.py
  examples:
    - title: Centre a planned top-down grasp on a lying tile before closing
      path: examples/canonical_subgraph.json
---

# centring-a-grasp-over-a-flat-part-from-the-grasp-pose

A grasp is usually planned from a look taken somewhere else: overhead, or the
wrist camera at another pose of the arm. An arm is not where its joints say by
the same amount at every pose, so the hand comes down a little to one side.
On a flat part taken across its width that is enough for one jaw to meet the
part first and push it over the support before the other closes. This skill
looks again from where the hand will come down, so the arm's error is the
same for the look and for the grasp.

## How it composes

- **Before it**: a perception skill's box for the part and a planned grasp
  pose (`grasping-flat-objects`' or a task's own), jaws already open.
- **After it**: lower straight to `grasp_pose_centred` and close. The hand is
  left at `pregrasp_pose`.
- **Not `grasping-by-wrist-camera-servoing`**: that skill segments the part
  with a model in every frame, steps the hand under a camera looking down and
  closes itself. This one calls no model, works with the hand in whatever
  orientation the grasp has (yam_dominos took tiles with the hand leaning
  15-25 deg past upright), squares the grasp to the part, and leaves the
  descent and the close to the caller.
- **Not `refining-target-from-wrist-closeup`**: that aims the camera at the
  target and returns a better point; it does not look from the grasp's pose,
  which is what cancels the arm's own error.

## What it does

1. Moves to `grasp_pose` lifted by `standoff` (IK preview first).
2. Takes a settled observation and the wrist camera's depth; fits the support
   as a plane in the camera's frame and takes what stands 0.45-1 part
   thickness over it; boxes each blob's outline in the world. A blob of the
   part's length and width within 3 cm of where the grasp is aimed is the part.
3. The part's middle against the aim, and its long side against the box's:
   under 1.5 mm and 2.5 deg ends it.
4. Otherwise the grasp is shifted by the difference and turned about the
   vertical through the part's middle (turns of 2.5-12 deg; a turned pose the
   arm does not have falls back to the shift alone), the hand moves to the new
   pregrasp pose, and it looks again. Three looks at most.

## Inputs

- `target_obb` (required): the part as planned; its extent gives the size
  (sorted: length, width, thickness), its orientation the long side's yaw.
- `grasp_pose` (required): the tool pose the jaws close at.
- `standoff` (0.05 m), `tolerance` (0.0015 m), `accept` (0.003 m),
  `max_turn_deg` (12), `max_looks` (3), `support_z` (0.0, the support's world
  height), `camera`, `arm_id`, `check_reach` (true).

## Required end states

- `centred`, `off_centre`, `not_seen`, `unreachable`: as in the frontmatter.
  `offset` is the total shift in x and y; `turn_deg` the total turn;
  `residual_mm` the last look's distance (-1 when nothing was seen).

## The constants, and why

Each is documented where it is defined in `scripts/centre_grasp.py` and
`scripts/wrist_depth.py`. The ones that matter on another robot: `STANDOFF`
(the part must be whole in the picture past the open fingers), `TOLERANCE`
(set it against the station's own move tolerance: a correction smaller than
what a move is allowed to miss by is not made), and the size gate in
`wrist_depth.flat_parts` (0.8-1.25 of the length, 0.8-1.4 of the width).

## Measured

yam_dominos, real left YAM (D405, linear_4310, tiles 49 x 24.5 x 7.4 mm on a
glossy stone counter), run `graph-20261120-103306`, 11 leaning picks:

| | distance of the hand from the part's middle |
|---|---|
| first look from the grasp's pose | 6-16 mm (and up to 8.8 deg) |
| after one correction | 0.7-2.9 mm |
| after two (8 of the 11 picks) | 0.4-1.8 mm |

The part was made out on every first look. In the simulator (the station's
twin) a grasp put 7 mm and 6 deg off on purpose came to 0.1 mm and 0 deg in
one correction, four picks of four.

Not measured: this script through a hub connector (the numbers are the task's
own implementation of the same steps); parts that are not flat boxes; a
support that is not level; parts touching each other (two touching tiles
read as one blob and are rejected by the size gate).

## Provenance

From yam_dominos' `_centre_lean` and `_centre_flat` and its depth finder
`lying_depth`: robot access through connector tools, the grasp's planned yaw
taken from the box, routed exits.
