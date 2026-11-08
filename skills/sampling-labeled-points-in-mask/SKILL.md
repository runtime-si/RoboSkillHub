---
name: sampling-labeled-points-in-mask
description: Picks the point on (or in) an object that a language instruction
  asks for - lays an evenly spaced grid of numbered markers inside the
  object's mask (the largest stride that leaves 35-45 inside, plus markers
  along its outline), draws them on an upscaled crop, asks a VLM which
  number(s) fit, and back-projects the chosen marker through depth to a
  world point. Use when a grasp, contact or placement point is a named part
  or region of one object ("where to grasp the mug handle", "the plug end of
  the cable", "a free spot on the tray") and a perception step already has,
  or sam3 can make, the object's mask in a calibrated RGB-D frame.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [vlm, visual-prompting, affordance, part, mask, select]
  hub:
    stage: select
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, cable-routing]
    tuned_on: [yam-station-overhead-rgbd]
    provenance:
      authors: [runtime-si]
      code: https://github.com/penn-pal-lab/maestro/blob/e60d326233a238ddc496bf7a08acb7a278e9e66d/maestro/api.py
      upstream_license: MIT
gap:
  allowed_tools:
    - vlm.query
    - sam3.segment_text
    - geometry.mask_to_world_points
  required_inputs:
    observation: Observation
    instruction: str
  produces_outputs:
    point: Vec3
    marker_count: int
  exit_conditions:
    selected: The VLM named offered marker(s); `point` is the first one's world point (near-side depth inside the mask), `points` / `selected` all of them in the VLM's order with index, label and pixel.
    none_suitable: The markers were drawn but the VLM answered [] (no marker fits), its replies (up to `attempts`) held no usable list or named numbers not offered, or no marker had depth; `detail` says which. `marked_image` shows what it was offered.
    not_seen: No mask - the given one is empty or under 20 px, or sam3.segment_text found no instance of `object_name` scoring 0.3 or more.
  canonical_scripts:
    - sample_labeled_points: scripts/sample_labeled_points.py
  examples:
    - title: Segment an object by name and pick the point an instruction asks for
      path: examples/canonical_subgraph.json
---

# sampling-labeled-points-in-mask

MAESTRO's `locate_object_part`: instead of asking a VLM for pixel
coordinates, offer it numbers. A grid of numbered discs covers the object,
the VLM answers "point 17", and the depth under marker 17 is the answer. The
VLM only has to recognise parts, not regress coordinates, and its answer is
always on the object.

## Steps

1. **The mask**: `mask` (a perception skill's `Mask`, e.g. perceiving-objects
   or perceiving-object-parts) or else `sam3.segment_text` on `object_name`,
   best instance scoring >= 0.3.
2. **Markers** (`generate_best_grid_from_mask`): a square grid inside the
   mask's box, inset 5 px, at the largest stride that leaves `target_min`..
   `target_max` (35..45) points inside the mask; if no stride does, the one
   whose count is nearest. Plus `boundary_count` (20) points spaced evenly
   along the largest outline. Outline markers are numbered first, then the
   grid, as upstream; a marker whose disc would cover an earlier one is
   dropped, so the outline (edges, corners, ends) keeps its markers and a
   few grid points next to it go (the board on the station: 20 outline + 27
   of 37 grid points kept).
   A marker whose 7 x 7 window (inside the mask) has no valid depth is not
   drawn and not numbered.
3. **The image** (`_sample_points_from_mask`): the mask's box plus 10 px,
   upscaled by the largest integer that keeps it inside the frame size, red
   discs (radius 17 px, smaller when markers are closer) with white numbers.
4. **The question** (`vlm.select_points`): MAESTRO's selection prompt,
   verbatim, plus "If no numbered point fits the instruction, output []", the
   offered numbers and the instruction, through `vlm.query` (`vlm_model` per
   call when set). The reply's outermost `[...]` is parsed (dicts with
   `index` or bare numbers; `//` comments tolerated). An unparsable reply or
   a number that was not offered is asked again, `attempts` (2) in all.
5. **World points**: for each chosen marker, the window's valid mask pixels
   at or under the 30th depth percentile (the near side: a marker on the
   object's outline reads the object, not the table behind it) go through
   `geometry.mask_to_world_points`; their median is the point.

## Inputs

- `observation`: the camera is `camera_name`, else the first whose name does
  not say wrist / hand. Needs `rgb`, metric `depth`, `intrinsics`, `pose`
  (camera to world).
- `instruction`: what to pick, in words ("where to grasp the mug's handle").
- `object_name` or `mask`: which object the markers go on.
- `target_min` / `target_max` (35 / 45), `boundary_count` (20),
  `crop_margin` (10 px), `max_points` (0 = all the VLM names), `vlm_tool`,
  `vlm_model`, `attempts` (2).

## Outputs

`point` (Vec3, world), `points` (all chosen, VLM order), `pixel` ([u, v] of
the first), `selected` (index, label, pixel, position each), `marker_count`,
`marked_image` (the crop the VLM saw), `selection_image` (the frame with the
mask outline and chosen markers), `reply`, `detail`.

## Which skill when

- **executing-keypoint-motion** (MOKA) marks 6 points per object (centre +
  farthest outline points) together with a waypoint grid, and asks for a whole
  motion. Use it for a one-step push / press / pick-place from an instruction.
  Use this skill when one point on one object must be chosen finely (a handle,
  an end, a corner): 40-60 markers instead of 6, on an upscaled crop.
- **selecting-free-placement-spots** asks for free tiles or points on a
  surface and then checks each footprint in depth (level, flat, clear). This
  skill does no such check: with the tray's mask and "a free spot on the
  tray" it returns where the VLM looked free. Feed the point to a depth check,
  or use that skill, when a placement must be verified.
- **perceiving-object-parts** segments a named part (crop + DINO + SAM3) and
  returns its whole OBB / cloud. This skill returns one point on it; a part
  that sam3 cannot segment alone ("the left end of the handle") can still be
  pointed at with the parent's mask.

## Limits

- Markers are only on the mask: a point just outside an object (beside a
  cable, under a rim) cannot be chosen. A wrong mask gives wrong options.
- Thin masks (a cable 3-5 px wide on a 640 x 480 frame) hold few interior
  points at the 5 px inset; the outline markers carry most of the choice.
- The point is the visible surface. A grasp centre for a thick part is
  below it; add the depth offset downstream.
- No reach or collision check.

## Measured on the YAM station

Real overhead frames `g_sota_frames_20261103-1421/frame_01..06.png` (640 x
480, one view, different sensor noise) with the depth of
`cable_frame_20261103-1421.npz` and the camera pose calibrated then; real
SAM3 (`sam3.segment_text`, the hub's tool module) and the hub's `vlm.query`
against the Gemini API; `geometry.mask_to_world_points`. "Right part" is the
chosen marker inside a hand-labelled region (15 px round a corner / plug /
fingertip, the loop's box, the slot's box, the board away from loops and
cable, the cable below the board).

| case (object; instruction) | ER-2, default, 6 frames | ER-2, 25-35 + 10, 3 frames | gemini-3.8-flash, 3 frames |
|---|---|---|---|
| pegboard; its top-left corner | 6/6 | 3/3 | 3/3 |
| pegboard; a free spot away from the loops and cable | 6/6 | 3/3 | 3/3 |
| pegboard; the wire loop on its left | 5/6 (1 none) | 3/3 | 2/3 (1 none) |
| cable; the plug at its end | 5/6 (1 other end) | 2/3 (1 other end) | 3/3 |
| cable; its free end on the table | 6/6 | 3/3 | 3/3 |
| cable; a point lying flat below the board | 6/6 | 3/3 | 3/3 |
| left gripper (mask given); fingertip on the right | 6/6 | 3/3 | 3/3 |
| metal plate (mask given); the rectangular slot | 3/6 (3 none) | 3/3 | 3/3 |
| **8 positive cases** | **43/48** (1 wrong, 4 none) | **23/24** | **23/24** |
| cable; "the red button" (absent) -> none_suitable | 6/6 | 3/3 | 3/3 |
| sam3 gave the board for "grey perforated metal plate"; the slot -> none_suitable | 6/6 | 3/3 | 3/3 |

- The two wrong picks both chose the cable's other end for "the plug"; a
  marker lay within 4 px of the plug every time. The plate's slot is a hole
  in its mask, so only markers on its rim could be chosen; ER-2 took one 3
  times and said none 3 times.
- sam3 scored both grippers 0.00 for "robot gripper" (its own path would
  return `not_seen`); the harness passed the left instance as `mask`.
  "grey perforated metal plate" segmented the black pegboard (0.89): the VLM
  then refused every time rather than pick a hole on the wrong object.
- 35-47 markers per object at the default range (cable 20-37: thin masks
  catch fewer grid nodes); the sparser upstream default (29-33) did as well
  on these cases.
- `vlm.query` took a median 2-4 s with ER-2 (plate 7-8 s) and 3-33 s with
  gemini-3.8-flash. Every reply parsed on the first try (126 calls).
- World points: the board corner read 14 mm over the cable on the table
  (the board is 16 mm), repeated picks of one marker agree within 3 mm.

## The constants, and why

- `TARGET_RANGE = (35, 45)`, `BOUNDARY_SAMPLES = 20`: MAESTRO's
  `locate_object_part` call (`generate_best_grid_from_mask(mask, (35, 45), 20)`).
- `GRID_INSET = 5` px, `CROP_MARGIN = 10` px, `MARKER_RADIUS = 17` px:
  upstream values.
- `DEPTH_WINDOW = 3`, `DEPTH_PERCENTILE = 30`: executing-keypoint-motion's
  point reading, so the two skills read a marked point the same way.
- `SAM_SCORE_FLOOR = 0.3`: the sam3 bundle's advice.
- `MIN_MARKER_RADIUS = 8` px: the smallest disc a two-digit label fits.

## Provenance

Adapted from MAESTRO (github.com/penn-pal-lab/maestro, MIT, Copyright (c)
2026 The MAESTRO Authors), `maestro/api.py` (`generate_best_grid_from_mask`,
`_sample_points_from_mask`, `locate_object_part`) and `maestro/vlm.py`
(`SELECTION_PROMPT`, `select_points`). GroundedSAM on the wrist camera is
replaced by a given mask or `sam3.segment_text`, the wrist point cloud by
`geometry.mask_to_world_points`, the OpenAI / Gemini client by `vlm.query`;
the script's header lists every change. License: `LICENSE.maestro`.
