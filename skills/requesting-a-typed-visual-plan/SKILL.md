---
name: requesting-a-typed-visual-plan
description: Asks a ladder of planners (vlm.query in text, robobrain.trace or
  any tool returning the visual-plan schema) for a visual plan in a shape
  declared before asking - which arms, how many waypoints each (10), every
  path starting at its arm's tool centre in the image, both arms of a bimanual
  plan in one request - and hands on only an answer that has that shape and
  agrees with what is measured (in the image, starting at the hand, depth
  under the endpoint, the two arms' distance along the plan). A refused answer
  is named and the next planner is asked; two planners can be required to
  agree; an earlier plan is reused only while it is young and the hands and
  the scene have not moved. Nothing moves. Use when a step needs a model's
  proposal of where one or two hands, or a point one of them carries, should
  go ("pick up the blue pen", "left hand to the cable end, right hand to the
  plug") as a checked plan for following-a-visual-plan.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [planning, visual-plan, vlm, trajectory, bimanual, validation]
  hub:
    stage: propose
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, cable-routing, bimanual]
    provenance:
      authors: [runtime-si]
      paper: vflash-2026
gap:
  requires: {connector: [robot.describe_workspace]}
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.describe_workspace
    - vlm.query
    - robobrain.trace
    - molmo-motion.forecast
  required_inputs:
    instruction: str
  produces_outputs:
    subtask: str
    planner: str
    planner_calls: int
    reused: bool
    metric: str
    min_arm_separation_m: float
    separation_at_fraction: float
  exit_conditions:
    planned: A planner's answer (or the earlier plan passed in, still valid) has the declared arms and waypoint count, every waypoint in the image, every path starting within start_tolerance of its arm's tool centre (or the given start pixel), a valid depth under every moving arm's endpoint and, for two arms, at least min_hand_gap between the tool centres along the plan; with agree=2, a second planner's endpoints within agree_tolerance. `plan` holds it in the visual-plan schema, one track per declared arm in the declared order. Nothing moved.
    plan_rejected: At least one planner answered and no answer passed - not the declared JSON shape, another waypoint count, a waypoint off the image, a start away from the hand, no depth under an endpoint, no metric part that agrees with the depth when require_metric, two arms too close, or fewer than `agree` planners agreeing; `attempts` names each planner's reason. Nothing moved.
    no_plan: No instruction, no usable camera, a declared arm's tool centre is not in the image, every planner raised or returned no track, or the only answers proposed no motion (every path ends within 12 px of its start). Nothing moved.
  canonical_scripts:
    - request_typed_plan: scripts/request_typed_plan.py
  examples:
    - title: Ask for a ten-waypoint plan for both arms and keep it only if it passes
      path: examples/canonical_subgraph.json
---

# requesting-a-typed-visual-plan

A planner's answer is text until something checks it. On real YAM station
frames two capable VLMs answered this skill's request in the declared shape
on all 78 requests and ended a median 1.5 px from the target; a thinking
model ran out of reply tokens before closing its JSON on 4 of 27; a small
model answered in perfect shape and ended 104-223 px from its target on four
of five tasks; `robobrain.trace` ended 7-20 px from the target on 8 of its
12 traces and 42-118 px from it on the other 4 (Measured). So the request
fixes the answer's shape before anything is asked, and an answer is taken
only when it has that shape and agrees with what the camera and the arms
measure.

The idea is VFlash's (arXiv:2611.08449, `papers/vflash-2026`): a tool call
declares the trajectory primitive (how many waypoints, one arm or two) before
any coordinate is predicted, a structurally valid answer is accepted as it
is, an invalid one is produced again by a second decoder, and the plan is
renewed on a period. VFlash's model is not part of this skill (its
checkpoints are not released): the planners are the hub's own tools, asked
under that protocol.

One script, stage propose:

```text
request_typed_plan
    planned       -> following-a-visual-plan (plan = plan; track = 0 | 1, arm_id of that track)
    plan_rejected -> re-perceive / another ladder / ask a human
    no_plan       -> start the motion another way
```

It returns `plan` (The plan, below; `{}` unless planned), `subtask`,
`planner`, `planner_calls`, `attempts` (one line per planner asked and for an
earlier plan passed in: what it answered or why it was refused), `reused`,
`metric` (`anchored`, `none` or `dropped: <why>`), `min_arm_separation_m`
and `separation_at_fraction` (-1 with one arm), `start_pixels` and
`endpoint_pixels` (one `[u, v]` per declared arm) and `detail`.

## The declaration

- `instruction`: one step ("pick up the blue pen", "move the gripper over
  the open box"). For both arms, say which does what, or give
  `arm_instructions` (one per arm; a text planner gets them as each ring's
  part of the task, a schema planner as that arm's instruction).
- `arms` (default `[0]`): one or two arm ids. The plan has one track per
  declared arm, in this order.
- `waypoints` (default 10): the number of waypoints per arm after the start.
- `start_pixels` (optional, one `[u, v]` per arm): plan from this pixel
  instead of the arm's tool centre - a point the arm carries (a cable's free
  end, a held part's tip). `null` for an arm keeps its tool centre.
- `planners` (default `["vlm.query", "robobrain.trace"]`): the ladder, in
  order of preference. An entry is a tool name or `{"tool": ..., "args":
  {...}, "name": ...}`; the same tool with other `args` (another `provider` /
  `model`) is another planner. `attempts` and `planner` call an entry by its
  `name`, by default the tool and its plain arguments
  (`vlm.query(provider=azure)`).
- `observation` (default: `robot.get_observation`) with one calibrated RGB-D
  camera (`rgb`, `depth`, `intrinsics`, camera-to-world `pose`); `camera_name`
  picks it, else the first named top / overhead, else the first usable.

Each arm's start is its tool centre (`robot.get_ee_pose`) projected into that
camera. A tool centre the camera does not see is `no_plan`.

## Asking

**A text planner** (a tool named `*.query`) is asked once for all declared
arms. It gets the frame with a ring drawn on each start (red for the first
arm, blue for the second, white-edged, the point itself left visible) and a
request that states the image size, what each ring is on (a gripper's tool
centre, or a point that arm carries), the task, and the answer's shape: one
JSON object with `subtask` and, per ring colour, a list of exactly
`waypoints + 1` `[x, y]` pairs in 0..1000 across the image - the ring's
centre as the planner sees it, then the waypoints, evenly spaced. A ring that
should not move repeats its centre. For a ring on a gripper the path is the
gripper's motion in the next step of the task, ending on what that step
reaches; for a carried point, where the task takes that point.

The ring's centre in the answer is the check on the planner's coordinates: a
planner that cannot say where the ring is (a swapped axis, pixels for
thousandths, the other gripper) is not believed about the rest.

**A schema planner** (any other tool: `robobrain.trace`, a `*.forecast` tool)
is asked once per arm with the camera's RGB, depth, intrinsics and pose, that
arm's instruction and its start (`start_point`; `query_points` for a
`*.forecast` tool), plus the entry's `args` (`mode: object` for a carried
point). Its first track is that arm's path. A path with another number of
keypoints is resampled to the declared count, evenly by arc length in the
image, the endpoint kept (`resample_paths`, default on; off, another count is
refused). RoboBrain's traces had 2-6 keypoints after the start.

A multi-step instruction is answered differently by the two kinds: a text
planner plans its next step and says which in `subtask` ("pick up the cube"
for "pick up the cube and place it in the box"), `robobrain.trace` draws the
whole manipulation and ends at the place. Give one step.

## The checks

In this order; the first that fails is the reason in `attempts`.

1. **Shape** (text): one JSON object, a list for every declared colour and
   none undeclared, exactly `waypoints + 1` pairs each, every pair two finite
   numbers in 0..1000. (Schema: `found`, a track with `pixels`, the declared
   count unless resampled.)
2. **In the image**: every waypoint.
3. **Start**: the path's own start within `start_tolerance` (5 % of the
   image's width, 32 px at 640) of the arm's start pixel.
4. **Motion**: an arm whose path ends within 12 px of its start is not
   moving (`active: false` in its track). No moving arm at all is `no_plan`.
5. **Depth under the endpoint** of every moving arm: a valid median in a
   7 x 7 window, which is what following-a-visual-plan lifts.
6. **The metric part** (schema planners that give `points_camera`): kept
   only when every point projects within 3 % of the image's width of its
   pixel, the start's depth is within `metric_tolerance` (5 cm) of the
   measured one (the hand's own depth, or the depth at a given start pixel),
   the end's depth is within it of the depth measured under the endpoint, and
   no waypoint lies more than it behind a measured surface. Then the tracks
   carry `points_camera` and `points_world` and the plan's `frame` is
   `world`. Otherwise they are dropped for every arm, `metric` names each
   arm whose metric part disagrees and every way it does, and the plan is
   `pixel` (`require_metric` makes that a refusal). Every RoboBrain trace so
   far is dropped here. Its path ends in the air, 9-29 cm in front of the
   surface under its endpoint (16 of 16 asked live by this script). And its
   start is the depth image's reading at the start pixel, which under a
   hovering hand is the table between the fingers or the gripper's back, not
   the tool centre: a path asked from a hand is compared with the hand.
7. **Two arms** (below).

## Two arms

Each arm's route on the table is rebuilt the way following-a-visual-plan
flies it: from the tool centre, through the path's pixels carried along their
camera rays to `travel_height` (12 cm) over the table, to the endpoint on the
surface measured under it. The table is `table_z`, else
`robot.describe_workspace`'s `surface_z`, else the depth image's fullest 4 mm
height bin. The smallest distance between the two routes is
`min_arm_separation_m`, at `separation_at_fraction` of the plan, under
`order`:

- `together` (default): both arms pass their waypoints in step.
- `a_then_b` / `b_then_a`: one arm flies its whole route while the other
  waits at its start, then the other flies while the first rests at its end.

Under `min_hand_gap` (0.15 m) is refused, except a plan that never brings
the hands closer than they already are. This is a distance between two tool
centres at travel height, not a collision check of the arms: run
checking-dual-arm-clearance on the poses before moving two arms through
each other's side.

## Agreement

`agree: 2` asks down the ladder until two valid answers move the same arms
and end within `agree_tolerance` (4 % of the image's width, 26 px at 640) of
each other, and hands out the one earlier in the ladder (`confirmed_by` names
the other). The shape and the depth cannot tell a wrong endpoint from a right
one; a second planner can. It costs a second call on every request.

It is as good as its second planner. Two capable text planners agreed on 115
of 117 pairs of answers. A capable text planner with `robobrain.trace` after
it (the default ladder; recorded answers, paired offline) was planned on 30
of 54 one-arm requests: all 30 on the five tasks where the trace ended 7-14
px from the target, none of the 24 on the other four, where the text
planner's endpoint was within 26 px of its target on 21 (Measured). For
`agree`, list two text planners first.

## An earlier plan

Pass the last `plan` back in. It is handed out again without a planner call
(`reused: true`) when it was made for the same instruction, arms, waypoint
count and camera, is at most `max_age_s` (30 s; 0: no bound) old, every arm's
tool centre is within 2 cm of where it was, and the depth under every moving
arm's endpoint is within 2 cm of what it was. Otherwise `attempts` says what
changed and the ladder is asked. A graph that loops over one step this way
asks again only after the hand or the scene moved, or the period passed.

## The plan

The visual-plan schema (`tools/molmo-motion/SKILL.md`) with what this skill
adds: `found`, `horizon` (the waypoint count), `dt_s: null` (a path, not a
schedule), `frame`, `backend`, `planner`, `confirmed_by`, `subtask`,
`instruction`, `typed` (`arms`, `waypoints`, `require_metric`), `metric`,
`camera`, `image_size`, `planned_at`, the two separation numbers, and per
arm `tracks[i]`: `arm_id`, `query_pixel` (the measured start), `pixels`
(`[waypoints][u, v]`), `predicted`, `active`, `start_error_px`, `hand_world`,
`endpoint_depth_m`, and `points_camera` / `points_world` when the metric part
was kept. It is plain JSON.

following-a-visual-plan takes it as `plan` with `track` (the arm's index in
`arms`) and that arm's `arm_id`; its own checks (on the table, in reach, the
pointing cross-check) still apply, and `follow_path` flies the waypoints.

## What it does not do

- It does not judge whether the plan is the right one for the task: an
  in-shape path to the wrong object passes every check but `agree`
  (`robobrain.trace`, asked to reach the cube around the box, ended in the
  box and was planned).
- A text planner's path is a line in the image. "Over the holder" was drawn
  to the holder's pixel by one model and 7-22 px up the image by another (as
  a hand hovering over it would be seen); the follower lifts whichever pixel
  with the depth under it.
- It does not move, and it does not replan while something moves.

## The constants, and why

- **`WAYPOINTS = 10`**: VFlash's robot plans (ten per gripper).
- **`START_TOLERANCE = 0.05`** of the width: the two capable planners put
  the ring's centre 0-2 px from it (102 paths), the answers that mistook the
  ring 74-86 px. `robobrain.trace` asked without a start began 26, 37 and
  365 px from the left hand's ring (the last at the other gripper).
- **`MIN_PLAN_PX = 12`**, **`DEPTH_WINDOW = 3`**, **`TRAVEL_HEIGHT = 0.12`**:
  following-a-visual-plan's.
- **`MIN_HAND_GAP = 0.15`** m: picking-and-placing-two-objects' bound for
  two open YAM hands.
- **`METRIC_TOLERANCE = 0.05`** m: VFlash reports 3.6-6.5 cm mean waypoint
  error on its 3D robot benchmarks; RoboBrain's 9-35 cm is refused.
  **`REPROJECT_TOLERANCE = 0.03`** of the width: a 1 cm metric grid alone
  moves a projection 4-8 px.
- **`AGREE_TOLERANCE = 0.04`** of the width (about 5 cm on a table 0.9 m
  away): two capable planners' answers ended within it on 115 of 117 pairs.
- **`HAND_MOVED = 0.02`**, **`SCENE_CHANGED = 0.02`** m: over a real
  station's 12 mm per-move bound and a few times its camera's 5.4 mm RMS.
- **`MAX_AGE_S = 30`**: VFlash renews every 60 control steps of a running
  policy; here a text planner's call took 3-15 s.

## Measured

**Live planners on real frames (2026-11-13).** 141 requests through this
script and the hub's `vlm.query` (10 waypoints per arm, default arguments,
each task 3 times) on three real YAM station frames from the overhead camera
with its calibration: the cube and open box (g_sota 20261103-0511, depth
ray-cast from the twin fitted to it), the cable board (threading
20261103-1421, real depth) and the pens station (20261109-2047, real depth).
13 tasks: one arm to an object (a cube, a pen, a cable's end, over a box, a
tray, a pen holder), the next step of a two-step task, a path around the
box, a carried cable end to a ring, and four two-arm tasks (two targets; one
arm told to stay); the pens frame's four went to the first two planners
only. The arm poses of the captures were not kept: each ring was
put on a pixel read off the frame between the fingertips. Endpoint error is
the distance to a hand-read target pixel, over the paths that should move.

| planner | in shape | planned | endpoint error [px] | seconds per request |
| --- | --- | --- | --- | --- |
| Azure OpenAI reasoning deployment | 39/39 | 39/39 | median 1.5, max 35; 43/48 within 10 | median 7.7 (5.5-15.4) |
| gemini-robotics-er-2-preview | 39/39 | 39/39 | median 1.1, max 15; 46/48 within 10 | median 5.7 (3.2-9.1) |
| gemini-3-flash-preview | 23/27 | 21/27 | planned: median 1.2, max 5 | median 12.5 (4.3-16.5) |
| gemini-3-flash-preview, 16 k reply tokens, two-arm tasks | 9/9 | 9/9 | median 1.2, max 5 | median 12.2 (4.2-53.6) |
| gemini-3.1-flash-lite-preview | 27/27 | 15/27 | planned: median 104, max 223; 3/21 within 26 | median 1.9 (1.3-2.5) |

- **What was refused.** gemini-3-flash: 4 two-arm answers cut off before the
  JSON closed (`GAP_VLM_MAX_TOKENS` 4096; none at 16 k) and 2 carried-point
  answers that started 74 px from the given start. flash-lite: 12 answers
  that started 76-86 px from the ring.
- **What was not caught.** flash-lite's other 15 answers were in shape,
  started within 13 px of the ring and ended 15-223 px from the target. Against Azure's or
  ER-2's answer to the same request they are 103-225 px apart on four of the
  five tasks (15 px on the fifth), which `agree: 2` refuses; Azure's and
  ER-2's own answers agree within 26 px on 115 of 117 pairs.
- **The largest errors of the capable two.** "Through the leftmost ring"
  with the cable's end: Azure's path passes the ring and ends 27-35 px
  beyond it, ER-2's ends 8-15 px from it. "Over the pen holder": Azure 7-22
  px up the image from the holder, ER-2 1 px.
- **The ring**: its centre in the answers of those two was 0-2 px from where
  it was drawn (median 0.5 px, 102 paths). An arm told to stay moved under
  0.5 px in theirs, 13 px in flash-lite's.
- **Spacing**: the longest step of a path of those two was 1.0-1.3 times its
  median step.
- **Around the box**: asked to reach the cube "around the open box and not
  over it", the paths of Azure, ER-2 and gemini-3-flash (9/9) stay 35-70 px
  from the box's centre and none crosses its rim in the image (48 px
  across); the straight line passes 17 px from the centre, through the box.
- **Two arms**: 24/24 planned for Azure and ER-2. The smallest separations
  were 0.194-0.198 m, on the pens frame (left hand to the blue pen, right to
  the pink pen 0.198 m from it).

**`robobrain.trace` live (2026-11-14).** The hub's tool (RoboBrain2.5-4B) in
this script's process on an idle RTX PRO 6000, the only planner of the
ladder, on the same frames, rings and targets: 13 requests, 16 traces (12
distinct: a two-arm request repeats two one-arm questions and got the same
replies). The tasks are the text planners' without the arm told to stay;
the carried cable end was asked in both of the tool's modes, and the
two-step task's target is the box here.

- **13/13 planned, every plan in pixels.** Each trace starts on the given
  pixel, has 2-6 keypoints after it and was resampled to 10. A call took
  0.5-1.4 s once the model was loaded (30 s for the first).
- **Endpoints** (the 12 distinct traces): 8 ended 7-20 px from the target
  (the cube, over the box, the plug, two pens, the tray, the holder, and the
  two-step task in the box); 4 ended 42-118 px from it and were planned all
  the same: the thin cable's free end (69 px, a zigzag beside the cable), the
  ring for the cable's end (42 px in `object` mode, 49 px in `end_effector`),
  and the cube "around the open box", a path that ends in the box (118 px).
- **The metric part was dropped 16/16.** Every trace ends 9-29 cm in front of
  the surface measured under its endpoint (median 20 cm): none passes
  wherever its start is. For the 14 asked from a hand the start is named
  too: the trace's start depth is the depth image's at the start pixel
  (within 5 mm, 16/16), under these hands the table between the fingers (the
  gripper itself for one). The 76-179 mm between that and the hand follow
  from where the hands were assumed to be (8 and 19 cm over the table), not
  from a measurement. The path to the cable's end also has a waypoint 19 cm
  behind what the depth image shows there, the gripper's own finger. (The
  run printed one reason a request; these are the recorded replies through
  the script as it is now.)
- **Two arms**: 3/3 planned, the smallest separations 0.195-0.348 m.
- **As the second planner of `agree: 2`** (offline: each recorded Azure and
  ER-2 answer paired with the live trace for the same task, through this
  script): 30 of 54 one-arm requests planned - 30/30 on the cube, the box,
  the pen, the tray and the holder, where the trace ended 7-14 px from the
  target; 0/24 on the cable's end, the ring, the path around the box and the
  two-step task (the text planner's path ends on the cube, the trace's in
  the box). The text planner's endpoint was within 26 px of its target on 21
  of those 24; the other 3 are Azure's past the ring. Two arms: 12 of 18 (the
  cable frame's 6 refused on the left arm's trace).

**Unit tests** (FakeContext, the scenes of following-a-visual-plan's tests).
16 of the recorded text answers (`tests/bundles/data/typed_plan_answers_station.json`)
are replayed and route as they did live; a recorded two-arm plan goes through
following-a-visual-plan's propose step for each arm (the cube's top, over the
box). The five live `robobrain.trace` requests on the cube frame are replayed
too (the tests render the depth image that run had): the routes, the starts'
distances from the hands and the arms' separation are the live ones, and
each trace's end is named, 132-272 mm in front of the surface; with the hand
lowered to the table under its ring only the end is; under `agree: 2` after
a recorded Azure answer the trace confirms the cube and the box and refuses
the path around the box. Every earlier recorded `robobrain.trace` answer
asked from a given start (12; 4 without the depth image, whose start is
24-35 cm off the measured depth) is planned after resampling with its metric
part dropped, and refused under `require_metric`. The metric checks, the
two-arm orders, reuse and the ladder are tested on built answers.

Not run: `molmo-motion.forecast` (built answers only), a planner that returns
a metric part that agrees with the depth (built answers only), `agree: 2`
live across two kinds of planner (recorded answers paired offline), a plan
followed by a simulated or real arm, a second camera view, and the requests
on frames where a hand holds something. No request had an arm's pose read
from the arm: the captures did not keep them.

## Provenance

Written for the hub (no upstream code). The protocol follows VFlash (NVIDIA,
arXiv:2611.08449; code Apache-2.0 at github.com/NVlabs/Eagle, checkpoints
not released as of 2026-11-13) and the checks follow what the hub measured of
its own planners (`tools/robobrain`, `following-a-visual-plan`).
