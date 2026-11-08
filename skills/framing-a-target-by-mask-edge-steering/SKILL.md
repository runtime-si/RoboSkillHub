---
name: framing-a-target-by-mask-edge-steering
description: Moves the wrist camera until a named target lies wholly inside its
  image with a margin - segments the target in the wrist frame
  (sam3.segment_text), reads which image edges its mask comes within the margin
  of and by how much, and steps the camera away from those edges in its image
  plane (or backs it off along its optical axis when opposite edges are clipped
  or an edge persists, or approaches when the target is too small), re-segmenting
  after each step, until the mask is inside every edge. Use when a part, ring,
  fixture or cable end is only partly in the wrist camera's view (after a coarse
  approach, a grasp, or a refined close-up that left it at the image border) and
  the next step needs all of it in one wrist frame, with no world estimate
  required.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, active-perception, wrist-camera, eye-in-hand, visual-servoing, segmentation, d405]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, insertion, cable-routing, sorting]
    provenance:
      authors: [runtime-si]
      code: https://github.com/AMMistry18/aic/blob/fd27bbe875dbd656e381f72ae2b44efdfb243924/flowstate/aic_perception/aic_perception/board_visibility.py
      upstream_license: Apache-2.0
gap:
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose
    - sam3.segment_text
  required_inputs:
    object_name: str
  produces_outputs:
    target_point: Vec3
    view_pose: Se3Pose
    area_frac: float
    depth: float
    moves: int
    travel: float
  exit_conditions:
    framed: A fresh wrist frame's mask of the target is inside every image edge by the margin (max(margin_px, 0.1 x its larger side)) and covers at least min_area_frac of the image - or covers less, but the camera is already min_range from it. The hand stays at that view; target_point is the per-axis median of the masked pixels back-projected with the wrist depth (a rough point on the visible surface).
    not_seen: There is no wrist camera, the segmenter failed or found no mask scoring min_score in the first frame, or the target was lost after a step and not found again back at the last view that saw it (the hand is left there); `detail` says which.
    step_limit: max_moves steps were taken and the mask still comes within the margin of an edge; `edges` names them. The hand stays at the last view.
    unreachable: A step was refused (raised or routed) or ended more than reach_tolerance off at full and at half length, or the next step would bring the tool centre within 50 mm of the target or back the camera past max_range; `detail` says which.
  canonical_scripts:
    - frame_target: scripts/frame_target.py
  examples:
    - title: Frame a named target in the wrist camera
      path: examples/canonical_subgraph.json
---

# framing-a-target-by-mask-edge-steering

A cheap image-space servo: no world estimate, no depth model, only a mask and
which borders it touches. Each step the wrist camera is translated away from
the clipped edges (an object clipped on the left asks the camera to move to its
left), by a step that shrinks as the clearance deficit does, and the target is
segmented again in the new frame.

## How it composes

- **Before it**: anything that brings the target roughly into the wrist view -
  `approaching-above-a-target`, a grasp, or `refining-target-from-wrist-closeup`
  (which aims the camera's optical axis at a *world point*; this skill corrects
  from what the camera *sees*, so it fixes a view that point left clipped).
- **After it**: the hand is left at the framing view, so the next node's
  `robot.get_observation` wrist frame holds the whole target (e.g. both the ring
  and the cable tip before `threading-a-cable-tip-through-a-ring`, or a part for
  `perceiving-objects` / `localizing-a-piece-in-a-reference-image`).
  `target_point` is a rough world point of the target for a later skill.
- **Shares helpers with `refining-target-from-wrist-closeup`**: the wrist
  camera choice (the named camera, else the `eye_in_hand` / `wrist` camera
  nearest the tool), its pose and intrinsics from the observation, the settle
  wait, the move-refusal test, the D405's 7-50 cm depth range, the 50 mm tool
  clearance and the 15 mm arrival bound are loaded from that skill's script, so
  it must be installed beside this one (it is in this hub).

## On a YAM

The D405 sits ~86 mm behind and ~75 mm above the tool centre and looks ~45 deg
forward of the fingers. This skill needs none of that as a constant: the camera's
image-right, image-down and optical axes in the world are the columns of the
frame's pose in each fresh observation, so the steps are right for any rotation
the hand holds (look-down or oblique) and for a wrist the twin leaves turned. The
hand's rotation is kept; only its position moves.

In the D405's wrist frame the open fingertips occupy the lower left and right
corners. `ignore_bottom_frac` (default 0) treats a bottom band as the image's
bottom edge, standing in for the source's calibrated gripper mask; set it
(~0.3 for open linear_4310 fingers) when a target hiding behind a finger must
count as clipped.

## What it does

1. Settled observation (two tool readings 0.5 mm / 0.1 deg apart); picks the
   wrist camera.
2. `sam3.segment_text(image, query or object_name, max_results=3)`; masks under
   `min_score` (0.3) are dropped. The best-scoring mask is the target in the
   first frame, or the one nearest `target_point`'s projection when given; after
   that, the one nearest the previous centroid.
3. Islands under 50 px, or under 5 % of the largest, are dropped; the mask's
   bounding box gives the clearance
   to each edge. The margin is `max(margin_px, 0.1 x the box's larger side)`;
   an edge within it is *clipped*. Fully inside and at least `min_area_frac`
   of the image: `framed`.
4. Action (the source's `adaptive_action`): opposite edges clipped, or the same
   edges clipped three views running with no quality gain or a growing area:
   **back off** one step along the optical axis. Inside but too small:
   **approach** along the axis to where the area reaches `min_area_frac` by
   the 1/range^2 law, from the mask's median wrist depth (+5 mm), at most one
   step and never nearer than `min_range`.
   Otherwise **translate** in the image plane along the clearance deficits
   (right minus left, bottom minus top), by step x the worst deficit over the
   margin (0.25-1).
5. The base step (30 mm) halves when a translation reverses direction (an
   overshoot). The tool centre is kept 50 mm over the target's back-projected
   height. A refused step is retried once at half length.
6. A step that loses the target sends the hand back to the last view that saw it,
   with the base step halved; lost again: `not_seen`.

## Inputs

- `object_name` (required), `query` (overrides it as the segmentation phrase),
  `segmenter` (default `sam3.segment_text`; any tool with the same contract).
- `target_point`: optional world point that picks among several masks.
- `camera`, `margin_px` (15), `context_pad_frac` (0.10), `min_area_frac`
  (0.003), `ignore_bottom_frac` (0), `min_score` (0.3).
- `step` (0.03 m), `max_moves` (10), `min_range` (0.15 m), `max_range`
  (0.50 m), `reach_tolerance` (0.015 m).
- `arm_id`, `tolerance`, `max_steps`: passed on only when set (`arm_id` when
  nonzero).

## Outputs

`edges` (clipped edges in the last frame), `clearance_px` (left / right / top /
bottom), `area_frac`, `center_error` (centroid off the image centre, -1..1),
`pixel` (centroid), `depth` (median wrist depth over the mask, 0 if none),
`target_point` (median of the masked pixels back-projected: in the sim it was
5-15 mm from a 30 mm cube's top centre in the plane, since the side faces seen
near the border count too), `moves`, `travel` (hand's net displacement), `view_pose`,
`camera_name`, `trace` (one line per view), `detail`.

## The constants, and why

- **`MARGIN_PX = 15`, `CONTEXT_PAD_FRAC = 0.10`**: analyze_board's defaults.
- **`STEP = 0.03` m**: move_to_board_skill's `CENTER_STEP_M`; at 0.2 m on a
  D405 (fx ~390 px at 640 x 480) it shifts the image ~58 px.
- **`MIN_AREA_FRAC = 0.003`**: the source's 0.02 was for a whole board; 0.003 is
  ~30 x 30 px at 640 x 480 (a 30 mm cube at ~0.4 m).
- **`MIN_BLOB_PX = 50`**: the source's minimum fiducial blob.
- **`ISLAND_FRAC = 0.05`**: on a real D405 frame SAM3's mask of a plate seen
  through railing wires came in five pieces, the wire-cut parts 12-100 % of the
  largest; a 112 px glint 20 px from the left edge (2.5 %) read as a clipped
  edge until this rule dropped it.
- **`APPROACH_SLACK = 0.005` m**: so the next view lands past `min_area_frac`.
- **`SCORE_FLOOR = 0.3`**: the sam3 bundle's advice, as in
  executing-keypoint-motion.
- **`MIN_RANGE = 0.15`, `MAX_RANGE = 0.50` m**: refining-target-from-wrist-closeup's
  nearest view and the D405's ideal depth range.
- **`MAX_MOVES = 10`**: the source has no count limit; a node needs one.

## Measured

**Closed loop, YAM station twin in RoboSimStudio** (left arm, 30 mm grey-blue
cube on a grey cloth, a wrist camera placed by the real station's calibrated
left-D405 camera-to-tool transform at its 640 x 480 intrinsics, path-traced
RGB-D). Each start first aimed the camera straight down at the cube from
0.22 m (refining-target-from-wrist-closeup with the true point), then
translated the hand until the cube's top projected at the given offset, so
the cube hung 1-2 corners outside the frame. Two segmenters: the cube's true
box projected and filled (the geometry floor) and the hub's
`sam3.segment_text` (`tools/sam3`, real SAM3) on the rendered frame, query
"grey cube". Truth check: the cube's 8 projected corners in the final frame.

| start (cube top off centre, px) | segmenter | route | moves | hand travel (mm) | truth corners inside the border by (px) start -> end |
|---|---|---|---|---|---|
| right (+296, +3) | truth mask | framed | 1 | 30 | -0.9 -> 29.5 |
| right (+296, +3) | sam3 | framed | 1 | 30 | -9.3 -> 16.3 |
| top (0, -225) | truth mask | framed | 1 | 29 | -24.7 -> 31.1 |
| top (0, -225) | sam3 | framed | 1 | 29 | -25.0 -> 31.0 |
| bottom (+3, +224) | truth mask | framed | 1 | 29 | -20.9 -> 32.0 |
| bottom (+3, +224) | sam3 | framed | 1 | 29 | -21.0 -> 32.2 |
| top-right (+276, -206) | truth mask | framed | 2 | 51 | -6.3 -> 19.0 |
| top-right (+276, -206) | sam3 | framed | 2 | 48 | -5.7 -> 15.7 |
| top-left (-280, -199) | truth mask | framed | 5 (4 translations, 1 back-off) | 118 | -21.6 -> 51.5 |
| top-left (-280, -199) | sam3 | framed | 5 (4 translations, 1 back-off) | 116 | -21.9 -> 51.3 |
| centred, 0.40 m, `min_area_frac` 0.006 | truth mask | framed | 3 approaches | 71 | area 0.0044 -> 0.0062, range 0.40 -> 0.33 m |
| centred, 0.40 m, `min_area_frac` 0.006 | sam3 | framed | 3 approaches | 73 | area 0.0041 -> 0.0061, range 0.40 -> 0.33 m |

12/12 `framed`, the true box inside the 15 px margin at the end of every
trial. SAM3 scored the cube 0.39-0.91. The top-left start kept its top edge
clipped for four translations (the twin turns the free wrist roll as the hand
moves, so the camera axes change under the steps) until the persistent-edge
rule backed the camera off. In earlier runs of the same harness, starts on the
left and in the bottom corners put the cube behind the open fingers: SAM3
returned nothing (`not_seen`, 3/3), while the occlusion-blind truth mask
"framed" a cube the camera could not see, and its wrist depth (the finger's,
8-11 cm) put `target_point` 160 mm off. `ignore_bottom_frac` does not help
there; a target hidden by a finger is not seen at all.

**Perception on real wrist D405 frames** (the station's left D405, 640 x 480;
live SAM3 via the hub's `tools/sam3`; hand-labelled boxes; views cropped to clip
the target at chosen edges):

| view | query | expected clipped edges | result |
|---|---|---|---|
| whole frame, cube at the bottom | grey cube | bottom | bottom (score 0.89): step image-down |
| cropped at the cube's middle from the left | grey cube | left, bottom | left, bottom (0.70) |
| cropped at the cube's middle from the right | grey cube | right, bottom | not seen: best score 0.297, under the 0.3 floor |
| 40 px strip through the cube | grey cube | top, bottom | not seen (0.09) |
| whole frame, plate behind railing wires | silver perforated metal plate | none | none (0.70): `framed` |
| cropped through the plate from the left | silver perforated metal plate | left | left (0.73) |
| cropped through the plate from the top | silver perforated metal plate | top | wrong object: the black plate on the table scored 0.79 |
| whole frame, black plate on the table | black perforated plate | bottom | bottom (0.73) |

5/8 views right (4/8 before the island rule, when a glint read as a left
edge on the whole-plate view). When the phrase fits another object better than
the clipped target, SAM3 takes that object: give `target_point` (the nearest
mask wins) or a phrase only the target fits.

Not measured: a closed loop on the real arm.

## Provenance

Adapted from AMMistry18/aic's `board_visibility.py` (`analyze_board`,
`view_quality`, `search_progress`, `adaptive_action`, `decide_direction`,
`world_delta`, `optical_axes_in_base`) and `move_to_board_skill.py` (the
observe / step / re-observe loop), Apache-2.0. Changed: a sam3 mask of a named
target instead of an Otsu-thresholded dark board anchored by a magenta fiducial
(the board's rectangularity, long-axis and IVM-readiness tests and its
`reorient` mode are dropped); one wrist camera instead of ranking several
(`combine_cameras`); the camera axes from the observation instead of TF; a bottom
band for the gripper mask; halving on overshoot and refusal; going back when the
target is lost; depth-range and tool-clearance limits on approach and back-off;
a move cap and routed exits.
