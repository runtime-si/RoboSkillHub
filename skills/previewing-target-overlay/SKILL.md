---
name: previewing-target-overlay
description: Checks a target gripper pose before the arm moves - projects the
  pose into every camera that sees it (tool centre, open jaw line with its
  fingers at the hand's span, approach axis), draws it as MAESTRO's image
  editor does, dashes what lies behind the depth image's surface, and asks the
  VLM a yes/no question - is this target right for the stated intent (e.g.
  "grasp the cube from above")? Use when a perceived or computed grasp, place
  or approach pose should be sanity-checked against the scene (wrong object,
  wrong side, mid-air, inside the table) before committing a move, with only
  cameras and the hand's own description.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [verify, preview, overlay, projection, vlm, grasp-check, safety]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, cable-routing, insertion]
    provenance:
      authors: [runtime-si]
      code: https://github.com/penn-pal-lab/maestro/blob/e60d326233a238ddc496bf7a08acb7a278e9e66d/maestro/api.py
      upstream_license: MIT
gap:
  requires: {connector: [robot.describe_gripper, robot.grasp_frame]}
  allowed_tools:
    - robot.get_observation
    - robot.describe_gripper
    - robot.grasp_frame
    - vlm.query_yes_no
  required_inputs:
    target_pose: Se3Pose
    intent: string
  produces_outputs:
    approved: bool
    answer: string
    reason: string
  exit_conditions:
    approved: The VLM answered yes - on the overlay the target fits the intent (right object or part, right side, right height); `answer` is its one-line justification.
    rejected: The VLM answered no; `answer` says why (wrong object, approach, height ...). Nothing moved.
    no_view: No camera with RGB, intrinsics and pose has the tool centre inside its image (or in front of it), or vlm.query_yes_no raised; nothing was judged and `reason` says which.
  canonical_scripts:
    - preview_target: scripts/preview_target.py
  examples:
    - title: Check a grasp pose against its intent before moving
      path: examples/canonical_subgraph.json
---

# previewing-target-overlay

Look before you move: draw where the hand is about to go and ask whether that
is where it should go.

## How a graph uses it

```text
<grasp / place pose from a perception or planning skill> -> preview (this skill)
    approved -> <the move>
    rejected -> <re-perceive / another candidate>
    no_view  -> <move the camera / skip the check>
```

Nothing moves. `target_pose` is a tool-centre pose in the world frame, the
same pose the graph would hand to `robot.go_to_pose`; `intent` is one plain
sentence ("grasp the small grey cube from above", "lower the held cube into
the open blue box").

## What it draws

Every camera of `observation` (default: a fresh `robot.get_observation`)
with RGB, intrinsics and a camera-to-world pose, where the tool centre lands
in front of the camera and inside its image, gets the overlay; the views go
side by side into one image, wrist cameras first, padded to the tallest
(MAESTRO's collage). On each view:

- **Tool centre**: MAESTRO's labelled point (red, white rim, white "target"
  on a blue box), drawn as a ring so the object under it stays visible;
  radius MAESTRO's `max(3, 1 cm in pixels + 1)`.
- **Jaws**: a green line through the tool centre, `robot.describe_gripper`'s
  open `span_m` long (or `jaw_width`), along the close axis, with a green
  finger at each end from 45 mm behind the jaw line to the fingertip
  (`finger.reach_m` past the tool centre).
- **Approach**: a blue line 0.10 m long (MAESTRO's axis length) ending at the
  tool centre, where the hand comes from.
- With `geometry_cues` (default on): pieces that lie more than 12 mm behind
  the depth image's surface at their pixel are dashed (inside or under
  something), and the question gets two sentences in words: the approach's
  tilt from straight down ("from above", "from the side", "from below") and
  the tool centre's distance in front of / behind the surface each camera sees
  at its pixel.

The axes come from the target's rotation and the hand's own frame, read off
`robot.grasp_frame` (its rotation for a top-down grasp against the approach
and close axis it reports), so the drawing follows each connector's tool
frame. `robot.describe_gripper`'s `approach_axis` / `close_axis` are not used:
on both the real YAM station and its twin they are not the axes of the
tool-centre frame (they point the approach up for a top-down target). If the
graph has world axes already (from its own `robot.grasp_frame` call), pass
`approach` and `close_axis`.

## The question

One `vlm.query_yes_no` with the overlay and a prompt that explains the
drawing (the robot is not there yet; ring = tool centre; green = open jaws
that close along the line; blue = approach), the cues above, the intent, and:
"Is this target pose right for the intended action: on the right object or
part, approaching from the right side, at the right height (where the jaws can
close on it, not hovering in the air above it, not inside the table or inside
another object)?" Yes routes `approved`, no `rejected`.

## Inputs

- `target_pose` (Se3Pose), `intent` (string).
- `observation` (fresh `robot.get_observation`), `cameras` (all, by name).
- `jaw_width` (describe_gripper's `span_m`; 94 mm, the linear_4310's, when the
  connector states none).
- `approach`, `close_axis`: world vectors overriding the axes from the rotation.
- `geometry_cues` (true): dashes and the two sentences; false draws as MAESTRO
  did, image only.
- `arm_id`, `model` (per-call `vlm.query_yes_no` model).

Outputs besides the route: `approved`, `answer` (the VLM's reply), `reason`,
`views`, `pixels` (tool centre per view), `surface_gap_m` (per view: + in
front of the seen surface, - behind it), `overlay` (the image sent) and
`prompt`. A graph that wants a hard floor can also read `surface_gap_m`
itself: a grasp's tool centre sits ~1-2 cm behind an object's top surface,
the inside-the-table targets below read -4.5 cm.

## Live check

`run()` through the hub's `vlm.query_yes_no` (OpenAI-compatible provider
pointed at Gemini's endpoint), temperature 0, on:

- the real station's overhead frame (`cable_frame_20261103-1421`, white
  cable on a black pegboard, Orbbec 640 x 480) with the calibrated camera pose
  (`top_camera_pose`, ChArUco, 5.4 mm RMS) and the station's
  `describe_gripper` / `grasp_frame`;
- two renders of the YAM station twin's calibrated overhead camera (30 mm
  cube beside an open box; arm parked, and arm hovering 15 cm over the cube)
  with the twin's own `describe_gripper` / `grasp_frame`.

Targets were set from the depth image (real) or the cube's OBB (twin): 8 good
(cable end / mid / right from above, the board's front edge, the cube from
above at two jaw headings, the same with the arm in view, a place into the
box) and 9 bad (wrong object: board centre, box, empty table; wrong side:
approach from below, from the side; mid-air: 20 / 15 cm over the target;
inside the table: 4.5 / 4 cm under its surface).

A = approved, R = rejected; "on" ran twice (identical both times).

| case | expected | ER-2, cues off | ER-2, cues on | flash, cues off | flash, cues on |
|---|---|---|---|---|---|
| real: cable end from above | A | A | A A | A | A A |
| real: cable mid | A | A | A A | A | A A |
| real: cable right | A | A | A A | A | A A |
| real: board front edge | A | A | A A | A | A A |
| real: wrong object (board centre) | R | R | R R | R | R R |
| real: wrong side (from below) | R | **A** | R R | **A** | R R |
| real: mid-air (20 cm up) | R | R | R R | R | R R |
| real: inside table (4.5 cm) | R | **A** | **A A** | **A** | R R |
| twin: cube from above | A | A | A A | A | A A |
| twin: cube from above, jaws turned 90 deg | A | A | A A | A | A A |
| twin: cube from above, arm in view | A | A | A A | A | A A |
| twin: place into the box | A | A | A A | A | A A |
| twin: wrong object (box) | R | R | R R | R | R R |
| twin: wrong side (horizontal) | R | **A** | R R | **A** | R R |
| twin: mid-air (15 cm up) | R | R | R R | R | R R |
| twin: inside table (4 cm) | R | R | R R | **A** | **A A** |
| twin: empty table 12 cm off | R | R | R R | R | R R |

Confusion per run (good targets 8, bad 9):

| model | cues | good approved | good rejected | bad approved | bad rejected |
|---|---|---|---|---|---|
| gemini-robotics-er-2-preview | off (MAESTRO's drawing only) | 8 | 0 | 3 | 6 |
| gemini-robotics-er-2-preview | on | 8 | 0 | 1 | 8 |
| gemini-3-flash-preview | off | 8 | 0 | 4 | 5 |
| gemini-3-flash-preview | on | 8 | 0 | 1 | 8 |

2-6 s per call (median 2.6 s). What to know:

- **No good target was rejected** in any run (48 judgments above, 32 more
  with an earlier prompt). The errors are all false
  approvals, so read `approved` as "nothing visibly wrong", not as proof.
- **An overhead camera cannot show the approach direction or the height.**
  With MAESTRO's drawing alone both models approved the from-below and the
  horizontal approaches every time: seen from above, a pose's approach axis
  is a short stub. Stating the tilt in words fixed both (4/4 per model).
- **Mid-air was rejected for the wrong reason.** The 15-20 cm high targets
  project well off the object (perspective), and the models said "next to
  the cube" / "on the railing", not "too high". A mid-air target straight
  down the camera ray would project onto the object; with the cues the flash
  model quoted the height ("hovering 17.6 cm above").
- **Inside the table stays the weak case.** With the cues each model still
  approved one of the two (ER-2 the real one, flash the twin one), the same
  way on both runs. The ER-2 rejection of the twin case without cues gave an
  offset as its reason, not the depth. If it matters, gate on
  `surface_gap_m` in the graph as well.
- One ER-2 reply in an earlier prompt version started "YES or NO? Let's
  think..." and `vlm.query_yes_no`'s first-word parse read it as yes; the
  tool's coercion, not this skill, decides such replies.
- Only overhead views were tested: the twin has no wrist camera and the real
  frame has none recorded. A wrist view would show the approach and height
  that the overhead view hides; the collage takes it when the observation
  has one.

## Changes from MAESTRO

Adapted from `maestro/api.py`'s `ImageEditApi` (`get_raw_images`,
`world_to_image`, `_draw_point_with_label`, `_draw_pose`, `feedback`), MIT:

- Camera poses are gap's camera-to-world `Se3Pose` (MAESTRO: a 6-D
  position + Euler extrinsic); every camera that sees the target is used
  (MAESTRO: wrist + one side camera).
- The pose is drawn as the gripper (tool centre, jaws at the hand's span,
  fingers, approach) instead of `cv2.drawFrameAxes`' RGB triad, which a VLM
  has to decode; the labelled point is a ring.
- Depth-aware dashes and the two geometry sentences are additions (MAESTRO
  drew without depth; its planner had the pose numbers in its own code).
- The collage goes to one `vlm.query_yes_no` with the intent and routes on the
  answer (MAESTRO: the image went back into the planning LLM's next turn,
  which decided what to do).
