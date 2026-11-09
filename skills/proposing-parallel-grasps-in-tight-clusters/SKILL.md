---
name: proposing-parallel-grasps-in-tight-clusters
description: Proposes top-down parallel-jaw grasps on small flat parts packed
  close together (jigsaw pieces, LEGO bricks, dominoes lying flat) from one
  posed depth frame (wrist D405 or overhead) and optional part masks - builds
  a world heightmap, slides the two fingertip footprints over centres, 8 jaw
  headings plus the part's axes and growing openings (arc-robot-vision's
  finger fit), keeps grasps whose fingertips reach half the part's
  height into free gaps at the opening the hand comes down with, with no other
  part between the jaws, the finger body clear of tall neighbours and the
  plates on an edge rather than a corner, and ranks them by finger clearance,
  centring and coverage. Returns world grasp points, the closing heading and
  jaw opening for robot.grasp_frame / robot.set_grip, and the fraction of
  parts with a collision-free grasp. Use when a part to pick has neighbours
  within a few fingertip widths, before a grasp skill, to find a grasp that
  fits or learn that the cluster must be spread first; CPU only.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: grasping
  tags: [grasping, propose, select, clutter, depth, jigsaw, lego, domino, cpu]
  hub:
    stage: select
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw, pick-place, sorting]
    provenance:
      authors: [runtime-si]
      code: https://github.com/andyzeng/arc-robot-vision/blob/728139e9ad0e003106c71a7c0d0a2fc9d8430644/parallel-jaw-grasping/baseline/predict.m
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/1710.01330
gap:
  requires: {connector: [robot.describe_gripper, robot.grasp_frame]}
  allowed_tools:
    - robot.get_observation
    - robot.describe_gripper
    - robot.grasp_frame
  required_inputs: {}
  # Optional inputs: mask (Mask, the target), camera (string), support_z (float), finger_thickness,
  # finger_width, tip_thickness, max_width, voxel, min_part_height, min_insertion (float, m),
  # rotations, per_piece, max_pieces, arm_id (int), unknown_is_obstacle, frame (bool). Wired from
  # a script node in the same graph (gap has no type for them): piece_masks (list of Mask),
  # camera_frame (CameraFrame), ignore_mask (Mask), roi ({x_min, x_max, y_min, y_max}).
  produces_outputs:
    grasp_pose: Se3Pose
    position: Vec3
    close_heading_deg: float
    jaw_width_m: float
    object_width_m: float
    tip_z: float
    tool_z: float
    clearance_m: float
    score: float
    n_pieces: int
    n_graspable: int
    fraction_graspable: float
    support_z: float
  exit_conditions:
    proposed: At least one grasp fits (the target's, or any part's without a target mask); `grasps` are best first, and the best one's position, close_heading_deg, jaw_width_m, tip_z / tool_z and grasp_pose (robot.grasp_frame's rotation at the tool centre) are flattened. Nothing moved.
    too_crowded: Parts were seen but no fingertip footprint fits beside any of them down to half their height, or every fit has another part between the jaws; `pieces` says why per part. Spread or singulate them (singulating-by-pushing), then call again.
    nothing_graspable: No part stands out of the support (too thin for the depth noise, or none in view), every part is wider than the hand or offers the plates only corners, the target mask covers no depth, or the frame has too few depth pixels; `reason` says which.
  canonical_scripts:
    - propose_cluster_grasps: scripts/propose_cluster_grasps.py
  examples:
    - title: Propose a grasp on a masked part in the wrist view
      path: examples/canonical_subgraph.json
---

# proposing-parallel-grasps-in-tight-clusters

Where can two fingertips go down beside a part that has neighbours on every
side? This skill answers from depth alone: it is the finger-fit search of the
Princeton/MIT Amazon Robotics Challenge 2017 baseline (arc-robot-vision's
`parallel-jaw-grasping/baseline/predict.m`), ported to numpy and re-tuned from
2-10 cm items in a bin to 2-12 mm parts on a table.

## How

1. **Heightmap.** The camera's depth (`camera_frame`, else `camera` from
   `robot.get_observation`, else its wrist camera) is lifted to world points
   with the camera pose and binned at the camera's footprint per pixel
   (0.5-2 mm cells; upstream 2 mm), the highest point per cell. The support
   is a plane fitted to the lowest well-populated height band (so a camera
   calibration tilt does not read as height) and the depth noise sigma comes
   from the spread below it; a cell stands over the support at
   max(1.2 mm, 3 sigma). Cells without depth are unknown, and count as
   obstacles under a finger unless `unknown_is_obstacle=False`.
2. **Parts.** From masks: `mask` is the target (only it gets grasps), every
   `piece_masks` entry a neighbour; unmasked standing cells touching exactly
   one masked part join it (depth smear at edges), others stay anonymous
   neighbours. Without masks: connected standing blobs (touching parts merge;
   see Gotchas).
3. **Search** (upstream's loop). For each part: up to 9 centres on its top
   nearest its centroid, 8 jaw headings 22.5 degrees apart plus the part's
   minimum-area-rectangle axes, and every opening in heightmap cells. Each
   fingertip is a `tip_thickness` (along the jaw) x `finger_width` (across)
   rectangle just outside the opening. A width is valid, as upstream, when
   the 90th percentile of the heights under both fingertips is at least the
   insertion depth below the part's median top; here the insertion is half
   the part's height (at least 1 mm and 2 sigma, at most upstream's 2 cm).
   New checks on top:
   - the fingers must also fit at the opening the hand comes down with
     (the contact opening + `OPEN_MARGIN` 2 mm per side);
   - no other part (over 4 mm^2 of it) between the open jaws;
   - **approach clearance**: under the finger body (`finger_thickness` x
     `finger_width`, grown 1 mm), the 90th percentile height stays under the
     fingertip plus its reach (10.2 mm), so a tall neighbour cannot meet the
     body while a low one may pass under it;
   - **edge contact**: on each side the part's outermost 1 mm (at least two
     cells) must span 4 mm across the jaw - an edge within 14 degrees of the
     plate or a jigsaw knob's round head, not a corner.
   The smallest valid width per centre and heading is kept, as upstream.
4. **Rank.** score = 0.35 x min(clearance / 3 mm, 1) + 0.35 x (1 - centring /
   max(3 mm, a quarter of the part's narrow side)) + 0.15 x coverage + 0.15 x
   (1 - opening / widest opening).
   Clearance is the gap from the open fingers to the nearest obstacle within
   6 mm (other parts, unknown cells); centring is the jaw centre's offset from
   the part's centroid across the jaws plus the part's off-centre along them;
   coverage is upstream's confidence (share of cells between the fingers over
   the finger floor). Up to `per_piece` (3) distinct grasps per part.
5. **Pose.** `position` is the jaw centre at the part's mid-height (the hub's
   proposer convention, as proposing-short-axis-grasps and grconvnet);
   `tip_z` the fingertips' height (0.5 mm over the floor under them) and
   `tool_z = tip_z + reach`. With `frame=True` the best grasp's heading goes
   to `robot.grasp_frame`, and `grasp_pose` is that rotation at
   (x, y, tool_z).

## Outputs

`grasps` (best first): `piece`, `position`, `close_heading_deg` (world, the
direction the jaws close along, [-90, 90)), `jaw_width_m` (open the hand this
far, `robot.set_grip`), `contact_width_m`, `object_width_m` (the part across
the jaws), `part_height_m`, `support_z`, `tip_z`, `tool_z`, `clearance_m`,
`centring_m`, `coverage`, `score`, `closing_is_short_axis`, `obb` (local x =
closing axis, half extents: for OBB-driven grasp skills) and `center_px`.
`pieces`: per part `status` (graspable, crowded, neighbour_between,
too_wide, poor_contact, too_flat, not_seen), `fails` counts, `centroid`,
`height_m`. Plus `n_pieces`, `n_graspable`, `fraction_graspable`,
`support_z`, `noise_m`, `voxel_m`.

## When to use which proposer

| situation | use |
| --- | --- |
| one part, neighbours far, its box from perception | **proposing-short-axis-grasps**: closes across the box's narrow side; blind to neighbours |
| irregular single object, container wall, curved cable; RGB-D, a learned prior | **grconvnet**: a network's grasp map; does not check the fingers against neighbours and opens generously (52-93 mm on a 30 mm cube) |
| small flat parts with neighbours within ~2 fingertip widths (a jigsaw tray, a brick pile, a domino row) | **this skill**: checks both fingertips, the opening and the body against the neighbours in the depth, and says `too_crowded` when nothing fits |
| `too_crowded` | **singulating-by-pushing** (or spread the cluster), then this skill again |
| a grasp found, part 2-4 mm thick | **grasping-flat-objects** with `obb` (see Gotchas: the tool-centre floor) |

Short-axis and grconvnet propose where the part is easy to hold; this skill
proposes where the fingers fit. On a lone part all three agree on the
heading (this skill gives the short axis for a domino or brick); among
neighbours only this one knows that the short-axis heading lands a finger
on the next brick.

## Constants, and why

- **`FINGER_THICKNESS = 10 mm`, `FINGER_WIDTH = 20 mm`**: estimates of the
  tapered linear_4310 tips (singulating-by-pushing uses the same);
  describe_gripper reports `pad_thickness_m: 0`, i.e. unmeasured. They decide
  everything below; measure the tips and pass `finger_thickness`,
  `finger_width` and, for the tapered bottom, `tip_thickness`. Upstream:
  36 x 60 mm.
- **`ROTATIONS = 8`, `FINGER_PERCENTILE = 90`, smallest valid width,
  coverage**: upstream's.
- **`GRIP_FRACTION = 0.5`, `MIN_INSERTION = 1 mm`, `MAX_INSERTION = 2 cm`**:
  upstream demands 2 cm below the top of 2-10 cm items; a 2 mm piece can
  only be gripped by its lower half, and 1 mm is about twice a D405's noise
  at 0.15-0.2 m.
- **`OPEN_MARGIN = 2 mm`**: the hand descends opened 4 mm wider than the
  contact width; checked like the contact width.
- **`MIN_PART_HEIGHT = 1.2 mm`, `NOISE_STANDS = 3`, `NOISE_INSERTION = 2`**:
  a 2 mm piece clears 1.2 mm; noisier depth raises both.
- **`CONTACT_DEPTH = 1 mm`, `CONTACT_RATIO = 4`**: edge within 14 degrees,
  inside a 0.3-0.5 friction cone; a knob head of radius >= 2.6 mm passes.
- **`BETWEEN_AREA = 4 mm^2`, `APPROACH_MARGIN = 1 mm`, `CLEARANCE_REF = 3 mm`
  (singulating-by-pushing's FIT_MARGIN), `CLEARANCE_WINDOW = 6 mm`,
  `WEIGHTS = (0.35, 0.35, 0.15, 0.15)`, `MAX_CENTRES = 9`**: chosen here, not tuned on
  a robot.
- **`TIP_REACH = 10.2 mm`, `MAX_GRASP_WIDTH = 86 mm`**: a real YAM station's
  describe_gripper (used only when it states none).

## Measured

CPU only, nothing executed on an arm. Script: `tests/bundles/_cluster_grasp_eval.py`
(writes the numbers and overlays).

**Packing gap sweep (synthetic).** Parts ray-cast into a top-down D405-like
depth frame from 0.18 m (390 px focal, 0.46 mm per pixel, cells 0.5 mm):
4 x 4 jigsaw pieces (20 mm cells, 2 mm thick, `tests/bundles/jigsaw_cut.py`
classic and wavy cuts), 3 x 3 LEGO 2 x 4 bricks (15.8 x 31.8 x 9.6 mm), 4 x 2
dominoes lying flat (24 x 48 x 7.5 mm), pulled apart by the same gap between
every neighbour. Parts with at least one grasp, all masks given, default
10 x 20 mm fingertip:

| gap (mm) | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jigsaw classic | 0/16 | 0 | 0 | 0 | 0 | 0 | 9 | 14 | 16 |
| jigsaw wavy | 0/16 | 0 | 0 | 0 | 0 | 1 | 4 | 14 | 16 |
| LEGO 2 x 4 | 0/9 | 0 | 0 | 0 | 0 | 3 | 9 | 9 | 9 |
| domino | 0/8 | 0 | 0 | 0 | 0 | 7 | 8 | 8 | 8 |

The threshold is the fingertip plus the 2 mm descent margin each side: a
10 mm tip needs ~12 mm gaps (10 mm passes on the 10 % percentile slack). The
true tip is tapered and unmeasured, so the same sweep with a thinner tip
bottom (`tip_thickness`, the 10 mm body still checked for tall neighbours):

| gap (mm) | 0-4 | 6 | 8 | 10 | 12 | 14 |
| --- | --- | --- | --- | --- | --- | --- |
| jigsaw classic, 6 / 4 mm tip | 0 / 0 | 0 / 0 | 0 / 4 | 8 / 12 | 14 / 16 | 16 / 16 |
| jigsaw wavy, 6 / 4 mm tip | 0 / 0 | 0 / 0 | 1 / 1 | 4 / 10 | 14 / 16 | 16 / 16 |
| LEGO 2 x 4, 6 / 4 mm tip | 0 / 0 | 0 / 7 | 9 / 9 | 9 / 9 | 9 / 9 | 9 / 9 |
| domino, 6 / 4 mm tip | 0 / 0 | 0 / 8 | 8 / 8 | 8 / 8 | 8 / 8 | 8 / 8 |

Jigsaw pieces need more than the tip: a finger must clear the neighbours'
knobs, which reach a quarter of a cell (~5 mm) into the gap. With depth noise
(sigma 0.3 mm) and half the pixels at depth edges dropped, the 10 mm counts
were within 1-4 parts of the clean ones (jigsaw classic 12 mm: 10/16; wavy
12 mm: 8/16; dominoes 10 mm: 3/8). Without masks the graspable counts matched, but
the parts found did not: at gap 0 each cluster is one blob, and the 80 mm
jigsaw block and the 47 mm-wide brick block "fit" the hand (1 of 1 found
part; the 96 mm domino block does not); jigsaw pieces 2-4 mm apart still
merge into one blob. Jigsaw caveat: below about 5 mm the pulled-apart
outlines overlap at the knobs (a knob's head is wider than its blank's
neck, so a real cut cannot open that way while flat); read those columns
as packed. Packed clusters
(gap 0-8 mm with the 10 mm tip) route `too_crowded`. 0.3-0.7 s per frame
for 8-16 parts (1.7 s with the noisy jigsaw), one core of a Core Ultra 7
265KF.

**Real station frames** (no packed small parts were recorded with depth, so
these check the geometry and the plumbing, not the cluster case):

- *Overhead Orbbec, 2026-10-31* (0.57 m, 1.2 mm cells, a 30 mm cube 75 mm
  from an open 100 mm box, masks by colour threshold): the cube `proposed`,
  support plane fitted at z = -0.032 m (the cube-touch calibration's table
  is -0.032 m), depth noise sigma 1.2 mm, cube height read 26.6 mm (true 30),
  closing 16 deg (its rectangle axis), 6 mm clear of everything, but its
  width across the jaws read 41.6 mm: depth smear at the cube's edges and the
  side face seen by the tilted camera widen it (grconvnet's measurement
  read 28.9 mm height on the same frame). The box (wider than the hand)
  `proposed` across its near wall (21 mm wall + smear, 44 mm opening), as
  grconvnet did; proposing-short-axis-grasps declines it. Without masks in a
  0.5 x 0.4 m roi: 13 height blobs, the cube best.
- *Wrist D405, 2026-11-04* (0.1-0.4 m over a white USB-C cable and plug on
  wood, the own fingertips masked out): with the cable masked, one frame
  `proposed` a grasp across the cable (3.0 mm high, 6.7 mm read wide, 15.5 mm
  opening); in the other the cable reads 1-2 mm high, under the noise floor
  (`nothing_graspable`, too_flat). Without masks the far field shows 14-30
  depth bumps of 2-5 mm, 1-7 of which "fit": use masks or an `roi` on real
  wrist frames.

## Gotchas

- **The tool-centre floor.** A real YAM station refuses a tool centre under
  14 mm over its table; with the 10.2 mm reach the fingertips then stop
  3.8 mm up, above a 2 mm jigsaw piece. `tip_z` / `tool_z` say where the
  fingers must go; grasping-flat-objects knows that floor (and refuses
  slabs it cannot reach). Do not send `tool_z` to a move unchecked.
- **The wrist roll is free under `robot.go_to_pose`.** As for every
  top-down proposer: plan with the orientation locked, or finish the turn
  with the last joint (stacking-objects).
- **Open with `robot.set_grip` to `jaw_width_m`,** not `open_gripper`:
  the full 94 mm opening lands the fingers on the neighbours this skill
  just avoided.
- **Without masks, touching parts are one blob.** A packed 4 x 4 jigsaw
  reads as one 80 mm "part" and may even get a grasp; pass piece masks
  (sam3) for anything that touches. On real wrist frames the far field
  (beyond ~0.3 m, where D405 noise grows) also shows 3-5 mm depth bumps that
  read as parts: give a mask or an `roi`.
- **The overhead camera is too coarse for jigsaw pieces.** At 0.57-1 m a
  pixel is 1.2-2.2 mm and the depth noise 1.2 mm or more: a 2 mm piece does
  not stand out (`nothing_graspable`, too_flat). Use the wrist D405 from
  about 0.15-0.2 m.
- **The ranking is not a success rate.** Weights and thresholds are
  geometry, not learned or tuned on grasps; nothing here was executed on
  the arm.

## Provenance

The search is adapted from arc-robot-vision (Apache-2.0, Andy Zeng et al.,
"Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance
Grasping and Cross-Domain Image Matching", ICRA 2018),
`parallel-jaw-grasping/baseline/predict.m` at commit `728139e9`: centres x 8
headings x growing widths, finger footprints, the 90th-percentile finger
floor below the part's median top, the smallest valid width, the coverage
confidence. Its flush (bin-wall) grasps are not ported. The world heightmap,
masks, support plane, noise-scaled thresholds, descent-opening, between-jaws,
finger-body and edge-contact checks, the clearance / centring ranking and the
robot.grasp_frame pose are new; the file's header lists the changes.
