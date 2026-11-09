---
name: routing-parts-to-sort-bins-by-id-and-colour
description: Sorts parts dropped one at a time on a staging pad into bins -
  watches the pad against an empty reference frame (grey and depth change,
  largest blob in percent of the zone), waits until a part has landed and the
  zone stops moving, rejects a capture while two parts or a part on the zone's
  edge are in it, classifies the part's colour (colour.classify), measures its
  footprint from depth and optionally its part id, chooses the bin by a rule
  list (colour, then part id, then category, then an "unknown" bin) with
  per-bin capacities and an overflow bin, and outputs the bin's drop position;
  the execute step picks the part (grasping-flat-objects) and drops it into
  the bin (placing-into-container-obb's drop points over a taught bin, or its
  perceived place for a bin given by description). Use when parts arrive on a
  known pad one by one (a feeder, the other arm, a person) and must go to bins
  by colour or size - LEGO bricks before a build, dominoes, jigsaw pieces into
  edge / colour trays.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [sorting, routing, bins, colour, change-detection, lego, domino, jigsaw, pick-place]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [sorting, pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/mbric/lego-sorter-public/blob/525683d3da4ec03017d9355afff8df2f5218b179/src/orchestrator/sort_cycle.py
      upstream_license: MIT
gap:
  requires: {connector: [robot.grasp_frame, robot.describe_gripper, robot.set_grip]}
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
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
    - colour.classify
    - pose6d.register_mask
    - sam3.segment_text
    - geometry.mask_to_world_points
    - geometry.filter_noise
    - geometry.compute_obb
  required_inputs:
    reference: Observation
  # Optional inputs (lists and open dicts are wired from a script node in the same graph): bins
  # (list of {name, accepts?, drop_position, opening_half?, yaw_deg?, rim_z?, capacity?, overflow?,
  # description?, pixel?}), staging_zone ({center: Vec3, half: [hx, hy], yaw_deg?}, world) or
  # staging_roi ({top, bottom, left, right} frame fractions), rules, size_classes, bin_counts,
  # camera_name, palette, white_ref, accept_ambiguous, part_id, categories, use_brickognize,
  # part_models ({name: pose6d mesh spec, e.g. {"box": [l, w, h]}, {"cube": s}}), wait,
  # max_wait_s, poll_s, settle_s, occupied_threshold, occupied_frames, motion_threshold; for the
  # execute step support_z, use_perception, lift_height, approach_clearance, release_over_rim,
  # retreat_height, footprint_margin, grasp_inputs, place_inputs, check_reach, spread_drops, arm_id.
  produces_outputs:
    bin_name: string
    drop_position: Vec3
    part_obb: OrientedBoundingBox
    part_colour: string
    category: string
    part_id: string
    label: string
    long_mm: float
    short_mm: float
    height_mm: float
    occupancy: float
    pose6d_model: string
    release_position: Vec3
  exit_conditions:
    sorted: A part was captured and routed, the hand picked it (grasping-flat-objects read its jaw gap across it after the lift) and opened over the chosen bin with the part's bottom release_over_rim over the rim (taught bin), or placing-into-container-obb placed it (described bin); the bin's count went up by one. What the hand did, not a sighting of the part in the bin.
    unknown_part: The part was captured but no rule matched and no bin accepts its colour, part id, category or "unknown" (or a rule names a bin not in bins); nothing moved.
    bin_full: The part's bin holds capacity parts (by bin_counts) and no overflow bin has room; nothing moved.
    staging_empty: No part landed (largest changed blob under occupied_threshold % of the zone for occupied_frames frames) within max_wait_s, the change was an empty-crop ghost, or there is no reference / camera frame or the zone projects outside it; nothing moved.
    staging_not_clear: Something landed but the capture was rejected 4 times (two parts in the zone), or it never settled (zone motion over motion_threshold) within max_wait_s; nothing moved - clear or singulate the pad.
    grasp_failed: The pick did not hold the part (every close read empty, too wide or too narrow, it slipped in the lift, or no grasp could be tried), there was no part box (no depth), or a perceived place found the hand empty; the hand is open, the part is on the pad or near it.
    drop_failed: The part was held but no drop point over the bin was reached (IK preview, a refused or short move), it does not fit the bin's opening, the jaws did not open, or a perceived place routed no_room / unreachable / container_not_found / not_released; still holding unless not_released.
  canonical_scripts:
    - plan_sort_route: scripts/plan_sort_route.py
    - sort_into_bin: scripts/sort_into_bin.py
  examples:
    - title: Wait for a part on the staging pad, route it by colour and drop it into its bin
      path: examples/canonical_subgraph.json
---

# routing-parts-to-sort-bins-by-id-and-colour

One cycle of a sorter: a part arrives on a staging pad, the camera sees it
land, the part is named (colour, size, optionally its part number), a rule
picks its bin, the arm moves it there. Run it in a loop with `bin_counts`
fed back from one cycle to the next.

## plan_sort_route (stage plan)

lego-sorter's conveyor cycle with the belt replaced by waiting:

1. **Zone.** `staging_zone` (world centre, half extents, yaw - projected
   through the camera's intrinsics and pose to a polygon) or `staging_roi`
   (fractions of the frame, lego-sorter's `LandingRoi`; default 0.40-0.92 x
   0.15-0.85). The comparison window is the zone grown by 25 % a side, shrunk
   until its long side is at most 160 px.
2. **Occupancy.** Each frame (`robot.get_observation`, camera `camera_name`)
   is turned grey, gain-matched to the `reference` (the empty pad, taken with
   the arm out of the view) by the ratio of the zone's medians (lego-sorter:
   means; a 2 cm part fills 5-15 % of a 10 cm zone), and a pixel is changed when it moved
   16 grey levels - or when both frames have depth and it came 8 mm nearer.
   Occupancy is the largest 4-connected changed blob inside the zone in percent
   of the zone (blobs under 8 px are noise).
3. **Landed.** Occupancy at least `occupied_threshold` (0.8 %) for
   `occupied_frames` (2) frames in a row. A drop back under it while settling is
   a false trigger (a hand, a shadow) and waiting resumes.
4. **Settled.** The zone's mean absolute grey difference between consecutive
   frames under `motion_threshold` (8) for `settle_s` (0.5 s). This is
   lego-sorter's coarse check: averaged over the zone, a 2 cm part sliding
   under about its own width per frame reads as still (a 25 mm part moving
   8 mm scored about 7 on a 10 cm zone in the tests); it is `settle_s` of
   stillness that guards a rolling part, so raise it for round or rolling
   parts.
5. **Capture gates.** A second blob at least 0.35 x the largest and 12 px, more
   than 2 px from it, is another part: re-settle, and after 4 rejects
   `staging_not_clear`. A blob touching the zone's edge is still arriving:
   re-settle; after 4 such rejects the blob is taken whole, past the edge. A
   padded crop with no edge energy (lego-sorter's empty-crop gate) is an
   exposure ghost: back to waiting. Nothing within `max_wait_s` (20 s):
   `staging_empty` (or `staging_not_clear` if something was there and never
   settled). `wait: false` checks the current frame once.
6. **The part.** Its mask is the full-resolution change (largest RGB channel
   difference or depth rise) in the blob's box padded 25 %, the component under
   the blob, holes filled. `colour.classify` on that mask (`palette`, default
   `lego_basic`; `white_ref` passed through): `classified` gives the colour,
   `ambiguous` only with `accept_ambiguous`, otherwise unknown. With depth and a
   camera pose, its depth points over the support (the reference's median
   height in the zone) by 4 mm, or all of them for a flat part, give an upright
   box: `cv2.minAreaRect` in xy, the top at the 95th height percentile -
   `part_obb`, `long_mm`, `short_mm`, `height_mm`, `support_z`.
   With `part_models` (`{name: mesh}` in pose6d's format: `{"box": [l, w, h]}`,
   `{"cube": s}`, `{"cylinder": [r, h]}`), each model is fitted to the part's
   mask with `pose6d.register_mask` (the support as its support plane); the
   best `registered` fit by fitness replaces `part_obb` with the model's true
   size and centre as an upright box (the model axis nearest vertical is the
   height) and names the part id (`pose6d_model`) unless `part_id` was given.
   No fit, or no pose6d server: the depth box stays, the detail says why.
   The measured `long_mm` / `short_mm` / `height_mm` stay the depth box's.
7. **Id and category (optional).** `part_id` if the caller knows it (e.g. from
   a recognition skill) or the pose6d model's name, `categories` maps ids to
   categories;
   `use_brickognize` (off by default) uploads the padded crop to the
   Brickognize parts API as lego-sorter does and takes the top item at score
   0.5; `size_classes` (`[{name, max_long_mm, min_short_mm, ...}]`, first match)
   names a size category from the footprint.
8. **Bin.** `rules` (`[{bin, colour?, category?, part_id?, min_/max_long_mm?,
   min_/max_short_mm?, min_/max_height_mm?}]`, first rule whose given keys all
   hold) or, without rules or when none matches, lego-sorter's `virtual_bin`
   priority over the bins' `accepts` labels (default the bin's name): the
   colour, then the part id, then the category, then a bin that accepts
   `"unknown"`. Labels are compared as lego-sorter's slugs (lower case, spaces
   to `_`). A bin whose `bin_counts` entry reached its `capacity` sends the part
   to the first `overflow` bin with room, else `bin_full`.
9. **Output.** `sort_bin` (the bin's dict), `bin_name`, `drop_position` (the
   bin's opening centre at rim height), `label` and `matched_by`, the part's
   measurements, `bin_counts` unchanged.

## sort_into_bin (stage execute)

1. **Pick.** grasping-flat-objects' script (installed beside this skill) on
   `part_obb` with `support_z` and a 50 mm lift (`grasp_inputs` passes anything
   else): jaws across the short side, fingertips 0.45 of the height up, the jaw
   gap checked after the close and after the lift, one retry across the long
   side. Anything but `grasped` is `grasp_failed`.
2. **Held box.** `part_obb` moved by the hand's travel from the grasp pose.
3. **Drop.** A bin with a `description` (and `use_perception`):
   placing-into-container-obb's whole script (sam3 finds it, `pixel` picks the
   instance), its exits mapped (`placed` -> `sorted`, `not_holding` ->
   `grasp_failed`, the rest `drop_failed`). A taught bin (`drop_position` at
   the rim, `opening_half`, `yaw_deg`): placing-into-container-obb's
   `footprint_reach` and `drop_candidates` (the opening centre, then half-way to
   each inner edge, each kept `footprint_margin` from the walls; `drop_failed`
   when it fits nowhere). With `spread_drops` (default) the bin's n-th part
   (by `bin_counts`) starts at the n-th of these points, so parts land beside
   each other rather than on a pile in the centre. The hand rises to carry the part's bottom
   `approach_clearance` over the rim, goes over the drop point (IK preview
   first), down to the part's bottom `release_over_rim` over the rim, opens
   (the jaws must read open) and retreats `retreat_height`. The hand keeps its
   rotation. `sorted` increments `bin_counts[bin_name]`.

## Inputs

- `reference`: the empty pad (Observation or CameraFrame, same camera), taken
  with the arm parked out of the zone. Take it again after the light changes;
  lego-sorter's slow re-baselining is not ported.
- `bins`: `[{"name", "accepts": [labels], "drop_position": Vec3 (the opening's
  centre at rim height), "opening_half": [hx, hy] (default 35 mm), "yaw_deg",
  "rim_z", "capacity", "overflow": bool, "description" (sam3 phrase),
  "pixel"}]`.
- The rest as listed in the header; constants below.

## Required end states

- Before `plan_sort_route`: the arm is parked out of the staging zone's view
  (it is part of every frame's difference otherwise), and the pad was empty
  when `reference` was taken.
- `sorted`: hand open, `retreat_height` over the release point, over the bin.
- `grasp_failed`: hand open, near the pad. `drop_failed`: holding (except
  `not_released`), over the bin or where the carry stopped.
- Plan exits other than `routed`: nothing moved.

## The constants, and why

From lego-sorter's `sort_cycle.py`: `PIXEL_DIFF_THR = 16`,
`OCCUPIED_THRESHOLD = 0.8` %, `OCCUPIED_FRAMES = 2`, `SETTLE_S = 0.5`,
`MOTION_THRESHOLD = 8`, `MIN_BLOB_PIXELS = 8`, `MULTI_BLOB_MIN_FRAC = 0.35`,
`MULTI_BLOB_MIN_PIXELS = 12`, `MULTI_BLOB_NEAR_GAP_PX = 2`,
`BLOB_EDGE_MARGIN_PX = 1`, `CAPTURE_GATE_MAX_REJECTS = 4`, the empty-crop
thresholds (2.0 / 10 / 1.5 % at 8), `CROP_PAD_FRAC = 0.25`, the exposure gain
0.55-1.85. Changed or added:

- **`COMPARE_MAX_SIDE = 160`** of the zone's window rather than the frame at
  160 x 120: a 2 cm part on a YAM station's overhead Orbbec is 10-20 px across;
  the whole frame at 160 x 120 makes it 2-5 px, under `MIN_BLOB_PIXELS`.
- **`DEPTH_RISE = 0.008` m**: 2-3x the camera's depth noise at 0.9 m; a part
  the grey of the table still triggers. Thinner parts (jigsaw pieces) rely on
  grey alone.
- **`MIN_PART_HEIGHT = 0.004` m**, **`TOP_PERCENTILE = 95`**: the footprint
  from points clearly over the support; the top robust to rim pixels.
- **`MAX_WAIT_S = 20`**, **`POLL_S = 0.1`**: choices.
- **`BRICKOGNIZE_MIN_SCORE = 0.5`**: a choice; the service's terms and rate
  limits were not checked, so it is off by default.
- Execute: `LIFT_HEIGHT = 0.05` m (grasping-flat-objects'),
  `FOOTPRINT_MARGIN = 0.008` m (placing-into-container-obb's),
  `APPROACH_CLEARANCE = 0.04` m, `RELEASE_OVER_RIM = 0.015` m (the part drops
  in, so the bin's contents never meet the hand; over a real station's 12 mm
  move tolerance), `RETREAT_HEIGHT = 0.06` m, `REACH_TOLERANCE = 0.015` m,
  `OPENING_HALF = 0.035` m.

## Not the same as

- **perceiving-sorting-pairs**: finds a labelled four-compartment destination
  and one source object at a time from RGB-D with a VLM; no landing trigger,
  no capacities. This skill waits for parts arriving on a fixed pad and routes
  by measured colour and size with a rule table.
- **placing-into-container-obb** / **grasping-flat-objects**: the pick and the
  drop this skill reuses; they know nothing of bins, rules or counts.
- **detecting-disturbed-pieces-by-snapshot-diff**: a registered before/after
  diff that says which known items moved; this one only asks "has one new part
  landed in the zone and stopped".

## Simulation check

A YAM station twin (left arm, overhead RGB-D camera 640 x 480) set up as a
sorting cell: a 100 x 100 mm staging zone on the tablecloth, three open bins
(100 x 100 x 40 mm, 3 mm walls): `red` and `blue` (capacity 4) and `other`
(accepts `unknown`, overflow). Twelve blocks, 20-30 mm (cubes and 30 x 20 x
20 / 30 x 30 x 25 bars) in LDraw red, blue, yellow and green, are dropped one
at a time from 25 mm over the zone, +-12 mm and +-45 deg apart (the harness
plays the feeder); each cycle is `plan_sort_route` -> `sort_into_bin` with
`bin_counts` carried over, then the simulator's box of every part is read.
5 red (the 5th must overflow), 4 blue, 2 yellow and 1 green (no bin: `other`).
`colour.classify` with `lego_basic`, no `part_models`, no Brickognize.

Run 1 (AutoDL RTX 4090, 2026-11-09, drops always at the bin centre - before
`spread_drops`):

- **Plan: 12 / 12** routed; colour right 12 / 12 (CIEDE2000 4.4-12.2; red
  11-12, the others under 10); bin right 12 / 12 including the 5th red part
  overflowing to `other` (`matched_by: colour, overflow`); footprint within
  0.7 mm of the true size (long, short, height), occupancy 7.6-17.6 %.
- **Execute: 12 / 12 grasped and released over the right bin** (`sorted`,
  the part's bottom 16-22 mm over the rim), but by the simulator only
  **7 / 12 ended inside the right bin** at the final tally (centre under
  rim + 10 mm; one of the 7 was still on the pile at its own cycle's check
  and slid in later). Every release was at the bin's centre, so later parts
  landed on earlier ones: one 20 mm cube bounced over the far wall onto the
  table, and four stayed on the pile at or above the rim (two of them, a
  30 x 30 x 25 bar and a 25 mm cube, over 60 mm up on 3-4 earlier parts in
  a 40 mm bin). This is what `spread_drops` (the n-th part to the
  n-th drop point) is for; run 2 below measures it.
- A cycle took ~7.5 min of wall time (the twin renders and steps slowly);
  `max_wait_s` is wall-clock, so the 20 s default held.

A second host (autodl14) whose renderer could not start its denoiser
(`Failed to create NGX context`) gave noisy frames: occupancy 74-80 % on
an empty-but-noisy zone and `staging_not_clear` (never settled) on the first
two parts. The skill's change detection assumes a camera whose frame-to-frame
noise is under `PIXEL_DIFF_THR`; that run was stopped and is not counted.

Run 2 (same host, same 12 drops, with `spread_drops`): plan 12 / 12
routed, colour 12 / 12, bin 12 / 12 (overflow included), footprint within
0.7 mm; execute **12 / 12 grasped, 12 / 12 `sorted`, and 12 / 12 inside the
right bin by the simulator**, at every cycle's own check and at the final
tally (red 4, blue 4, other 4: two yellow, one green, the overflowed red).
The releases walked the drop points centre, -u, +u, -v per bin. One run of
12, one layout: not a rate, and no failure exit was exercised in the sim
(the unit tests cover them).

## Provenance

Adapted from lego-sorter-public by Matt Brickner (MIT,
`LICENSE.lego-sorter` beside this file): `src/orchestrator/sort_cycle.py`
(the landing zone, the exposure-matched grey difference, the largest-blob
occupancy, the landed / settled / capture-gate logic), `src/orchestrator/routing.py`
(`virtual_bin`: colour > category > unknown) and
`src/demo_brickognize/brickognize_client.py` (the optional upload). Changed:
a staging pad instead of a conveyor (no belt control, no WaitClear - the arm
removes the part), the zone in world metres, a zone-sized comparison window,
depth as a second change cue, colour by `colour.classify` (CIEDE2000 against
a palette) instead of HSV hue rules, a depth footprint, a rule table with
capacities and an overflow bin, and the execute step (lego-sorter leaves the
move to an industrial arm's taught dump poses). The source's lighting EMA,
feeder-fringe and webcam auto-exposure heuristics are not ported. LEGO is a
trademark of the LEGO Group, which does not sponsor or endorse this.
