---
name: executing-keypoint-motion
description: Plans and executes a one-step manipulation from a language
  instruction with MOKA's mark-based visual prompting - candidate keypoints on
  the named objects and a lettered waypoint grid are drawn on the overhead
  image, a VLM picks the grasp point, the function point, the target point
  and the pre/post-contact waypoint tiles and heights, depth makes them world
  points, and the arm grasps the object at the grasp point (or closes its
  hand) and carries the function point through pre-contact, target and
  post-contact. Use when a short instruction such as "push the cube to the
  right", "press the button", "sweep the cable aside with the stick" or "put
  the plug on the board" must become one motion without a task-specific
  skill, with a calibrated overhead RGB-D camera and a hand that is empty.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [vlm, visual-prompting, keypoints, tool-use, push, press, open-vocabulary]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [tool-use, pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/moka-manipulation/moka/blob/main/moka/policies/franka_policy.py
      upstream_license: MIT
      paper: https://arxiv.org/abs/2403.03174
gap:
  requires: {connector: [robot.grasp_frame]}
  allowed_tools:
    - vlm.query
    - gemini-er.point
    - sam3.segment_text
    - robot.get_ee_pose
    - robot.get_gripper
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.open_gripper
    - robot.close_gripper
    - robot.grasp_frame
  required_inputs:
    observation: Observation
    instruction: str
  produces_outputs:
    final_function_point: Vec3
    hand_offset: Vec3
    held: bool
  exit_conditions:
    planned: (plan only) The VLM's reply named offered marks and tiles, every chosen keypoint has depth, and `plan` holds the world points, heights, mode and hand heading; execute follows.
    no_plan: (plan only) The instruction names no object, an object was not found (sam3 / gemini-er returned nothing), no candidate had depth, or every VLM reply (up to `attempts`) was unparsable or picked marks that were not offered; `detail` says which. Nothing moved.
    done: (execute only) Every phase reached its point within reach_tolerance (read back from the arm), including the grasp (jaws above empty_fraction) when there is one; the hand then released and lifted. This is what the hand did, not what the object did - re-perceive before relying on it.
    grasp_failed: (execute only) The jaws closed to empty_fraction or less at the grasp keypoint; the hand opened and rose back to the safe height.
    unreachable: (execute only) A free move (the first lift, a hover above the grasp or the pre-contact point) or the descent to the grasp was refused or ended off with every rotation tried (nothing touched), or a contact move (pre-contact, target, post-contact) stopped short; `detail` names the phase. After a contact-phase shortfall the hand still released and lifted.
  canonical_scripts:
    - plan_keypoint_motion: scripts/plan_keypoint_motion.py
    - execute_keypoint_motion: scripts/execute_keypoint_motion.py
  examples:
    - title: Plan from the overhead camera and execute
      path: examples/canonical_subgraph.json
---

# executing-keypoint-motion

MOKA (Liu, Fang et al., 2024) turns an instruction into one motion with a
marked image: points on the objects, a grid on the free space, and a VLM
that answers with labels. Two scripts with clean entry and exit:

- **`plan_keypoint_motion`** (stage plan): `observation`, `instruction`
  -> `plan`, `marked_image` (what the VLM saw), `motion_image` (the chosen
  points on the frame). Routes `planned` / `no_plan`. Moves nothing.
- **`execute_keypoint_motion`** (stage execute): `plan` (the dict, or its
  JSON string) -> routes `done` / `grasp_failed` / `unreachable`.

A graph can put a check between them (show `motion_image` to a judge,
previewing-target-overlay, an IK preview) or re-plan on `no_plan`.

## Planning

1. **The task** (MOKA's `propose_subtasks`, one step). When neither
   `object_grasped` nor `object_unattached` is given, the VLM names them and
   the motion direction from the instruction and the frame. Pass them (an
   empty string for "none") to skip the call.
2. **Candidate keypoints** per object (MOKA's
   `get_keypoints_from_segmentation`): the mask's centre first, then
   farthest-point samples of its outline, `num_candidates` (6) in all, sorted
   by image row. The mask is `grasped_mask` / `unattached_mask` when a
   perception skill gives one, else `sam3.segment_text` on the name; when
   sam3 is not served (`candidate_source` auto) the points come from
   `gemini-er.point` instead.
3. **The marked image**: a square crop around the keypoints, at least
   `grid_size` x `tile_size` (5 x 5 cm) across and `crop_margin` (8 cm)
   beyond them, resized to 768 px, with red `P1..` on the object to grasp,
   blue `Q1..` on the object acted on, and a `grid_size` grid lettered
   a.. left to right and numbered 1.. bottom to top (MOKA's layout).
4. **The question** is MOKA's `select_motion` prompt (verbatim apart from
   the grid letters, a note that the closed gripper is the contact when
   nothing is grasped, and the tile size in cm). The reply's JSON is parsed
   (code fences and `//` comments tolerated), the labels checked against the
   marks offered; a bad reply is asked again up to `attempts` (3) times.
5. **World points**. A keypoint's depth is the 30th percentile of a 7 x 7
   window (inside the mask when there is one), so a point on an edge reads
   the object. The support height is the fullest 4 mm height bin of the
   crop between 15 cm under and 5 mm over the lowest keypoint (the floor
   past the table edge is ignored). Contact height:
   - side approach (`pre_contact_height` same): the target's mid height over
     the support (`contact_fraction` 0.5);
   - from above (`above`): the target's surface + `z_offset` (1 cm);
   - pick and place (a grasp, no function point): the target surface plus
     the grasped point's height over the support plus `z_offset`, so the
     object's bottom lands just over the target.
   A waypoint tile's centre is cut by the contact-height plane along its
   camera ray; `above` puts it at the safe height (`hover_clearance` 12 cm
   over the support, and at least 5 cm over the highest keypoint).
6. **Hand heading**: closing across the grasped object's principal axis
   near the grasp point; with nothing grasped, closing across the push
   (pre-contact to target), as pushing-objects does.

`plan` holds `mode` (push, press, tool_push, tool_press, pick_and_place,
move_after_grasp), `grasp`, `function`, `target`, `pre_contact`,
`post_contact` (Vec3 or null), `support_z`, `safe_z`, `contact_z`,
`grasp_heading_deg`, `push_heading_deg`, the labels and pixels, the crop and
the VLM's reply.

## Executing (MOKA's phases)

1. Rise to the safe height where the hand stands (`lift_open`).
2. With a grasp: open, hover over the grasp point (top-down via
   `robot.grasp_frame`, both wrist solutions, then the hand's own rotation),
   come straight down to `grasp_depth` (1 cm) under the keypoint, close, read
   the jaws (`robot.get_gripper`; at or under `empty_fraction` 0.04 is
   `grasp_failed`), lift. The hand-to-function offset is measured there.
   Without a grasp: close the hand; the closed fingers are the function
   point.
3. The function point (hand + offset) goes: over the pre-contact point at
   the safe height (free move), down/along to pre-contact, to the target,
   to post-contact (straight-line moves). Pick and place goes over the
   target and down to it instead (MOKA's `reach_pre_target` / `reach_target`).
4. Release: holding, open then rise `release_lift` (3 cm); empty, rise then
   open. Then back to the safe height.

Every move is read back (`robot.get_ee_pose`) against `reach_tolerance`
(2 cm) and refusals are routed (raised or `reached: false`). The tool
centre never goes under `min_tool_height` (15 mm) over the support.

## The constants, and why

- `NUM_CANDIDATES = 6`, `GRID_SIZE = 5`, `Z_OFFSET = 0.01`,
  `RELEASE_LIFT = 0.03`, `ATTEMPTS`: MOKA's config (`num_candidate_keypoints`,
  `waypoint_grid_size`, `grasp_z_offset`, `release_z_offset`; it retried until
  a reply parsed).
- `TILE_SIZE = 0.05` m, `CROP_MARGIN = 0.08` m: MOKA cropped its table by hand;
  here the crop is sized in metres so one tile is ~5 cm, the step a table-top
  push of a few cm needs. A crop that must hold far-apart objects grows and
  its tiles with it (the plan reports `tile_m`).
- `HOVER_CLEARANCE = 0.12` m: MOKA's safe_z was 13 cm over its table.
- `CONTACT_FRACTION = 0.5`: pushing-objects' contact height. MOKA aimed at the
  target's top + 1 cm, which a horizontal push passes over.
- `MIN_TOOL_HEIGHT = 0.015` m: a real YAM station refuses a tool centre under
  14 mm over its calibrated table. A 3.5 mm cable on the table cannot be
  pinched from that height (the fingertips end ~5 mm over the table).
- `GRASP_DEPTH = 0.01` m: tool centre 1 cm under the grasped surface, the
  closed fingertips (~10 mm below it) then close 2 cm under the top.
- `EMPTY_FRACTION = 0.04`, `REACH_TOLERANCE = 0.02` m: handing-over-between-arms
  and pushing-objects.

## Not carried over from MOKA

- **`target_angle`** is asked and returned but not executed: the hand keeps
  its grasp rotation through the motion, so MOKA's grasp-to-current
  transform is the measured offset. MOKA's Franka policy also overrode it
  with a fixed yaw. A YAM station twin leaves wrist roll free anyway.
- **The grasp sampler** (a learned 2D grasp proposal near the keypoint) is
  replaced by the principal-axis heading; the jaws are read back instead of
  timing the grip out.
- **Multi-step plans** (MOKA's subtask list): one step per call; a graph
  chains calls.

## Measured

**Live planning on a real station frame** (overhead Orbbec 640 x 480 RGB-D,
2026-11-03, calibrated camera pose; no cube or box in that frame). Candidates
from `gemini-er.point` (gemini-robotics-er-2-preview; sam3 not served),
selection through the hub's `vlm.query` with two models; the chosen points
were drawn on the frame and judged by eye. The support height came out at
-20.9 mm against the calibrated table's -21.0 mm.

| instruction | ER-2 as the selector | gemini-3-flash as the selector |
|---|---|---|
| move the cable end next to the ring | grasp + function on the cable's free end, target on the left ring: right points; tiles over-then-onto the ring ('above' both), not quite "next to" | grasp = function on the free end, target on the ring, a side path across it: right points, plausible path |
| pick up the cable plug and put it on the pegboard | grasp on the cable's left free end (not the plug end by the board), target the board's centre: half right | no_plan: gemini-er.point returned no "cable plug" on that call |
| push the white cable loop to the right | target on the cable left of the board, pre tile left, post tile right: right direction, but the post waypoint is on the board at 6 mm, so the hand would hit the board's edge | target on the cable, pre lower-left, post right: right direction |
| push the cube into the box | no_plan (no cube found): correct | no_plan: correct |

A plan took 15-30 s with ER-2 and up to 66 s with gemini-3-flash.

**Twin execution** (YAM station twin, overhead camera render, one run):
"push the cube 5 cm to the right", planned live on the render with ER-2
(task parse: object_unattached cube, left to right; target Q4 on the cube's
lower-left corner, tiles b3 -> d3, both 'same', one tile = 5.0 cm), then
executed: `done`, all four motion phases within 2.6 mm, contact at the
15 mm floor (the 30 mm cube's mid height). The cube moved 74 mm: 73.9 mm
image-right, 9.6 mm image-up, no tipping (height change -0.005 mm). It
overshot the asked 5 cm because the post-contact tile's centre lay 64 mm
past the target point: MOKA's waypoints are tiles, so a push length is good
to about half a tile plus the finger width.

## Provenance

Adapted from MOKA (github.com/moka-manipulation/moka, MIT, Copyright (c)
2024 Kuan Fang, Fangchen Liu): `moka/planners/visual_prompt_utils.py`,
`moka/vision/keypoint.py`, `moka/planners/planner.py`,
`moka/policies/franka_policy.py` and the `visual_prompt_planner` prompts.
GPT-4V replaced by the hub's `vlm.query`, Grounding DINO + SAM by a given
mask / `sam3.segment_text` / `gemini-er.point`, Franka delta actions by
`robot.*` connector moves; see the scripts' headers for every change.
