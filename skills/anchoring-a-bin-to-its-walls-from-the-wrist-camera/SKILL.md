---
name: anchoring-a-bin-to-its-walls-from-the-wrist-camera
description: Re-anchors a tall open bin (a parts box picked from) to where the
  wrist depth camera sees its walls - in each look taken from over the bin,
  every wall in view is found as the densest thin sheet of wall-coloured depth
  points across it, the offsets of the walls read give the bin's shift from
  where a fixed camera had it, and the bin is moved by the median of the looks
  so far, leaving out a look that reads more than 3 cm from that median. Grasps
  found and made in the anchored bin then share the arm's own pose error
  instead of the overhead calibration's. Use when grasps are proposed from
  wrist-camera looks into a bin whose box came from an overhead camera, the arm
  is only accurate to a few centimetres, and the walls show in the wrist view;
  call it on every look with the same overhead box and the shifts it returned
  last time. CPU only; nothing moves.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, bin, box, container, wrist-camera, depth, calibration, bin-picking, cpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tuned_on: [yam_i2rt]
    tasks: [pick-place, sorting, bin-picking]
    provenance:
      authors: [RoboSkillHub contributors]
    notes: Its built-in wall colour rule is for brown cardboard; pass wall_mask or
      wall_colour "any" for other bins. One station, one box; point-count thresholds
      scale with camera resolution.
gap:
  allowed_tools:
    - robot.get_observation
  required_inputs:
    bin_obb: OrientedBoundingBox
  # Optional inputs: camera, wall_colour (string), wall_thickness, max_shift, outlier, min_range
  # (float), stride (int), wall_mask (Mask). Wired from a script node (gap has no type for
  # them): past_shifts (list of [dx, dy]), camera_frame (CameraFrame).
  produces_outputs:
    anchored_obb: OrientedBoundingBox
    shift: Vec3
    used_shift: Vec3
    looks: int
  exit_conditions:
    anchored: At least one wall was read in this look, and the reading lies within max_shift (10 cm) of bin_obb and, once two readings are held, within outlier (3 cm) of their median; it is added to `shifts`, and anchored_obb is bin_obb moved by the median of all of them.
    outlier: A wall was read but the reading was not counted (too far from bin_obb, or from the median of the looks so far); `shift` carries it, `shifts` is unchanged, and anchored_obb is bin_obb moved by the median of the earlier looks (bin_obb itself when there are none).
    walls_not_seen: No wall could be read - no posed wrist depth camera, fewer than 800 wall-coloured points in the wall's height band, or no wall with a dense enough sheet; anchored_obb as for outlier.
  canonical_scripts:
    - anchor_bin: scripts/anchor_bin.py
  examples:
    - title: Anchor the parts bin on each look
      path: examples/canonical_subgraph.json
---

# anchoring-a-bin-to-its-walls-from-the-wrist-camera

The arm is a few centimetres from where its joints say, so its wrist camera
has the bin a few centimetres from where the overhead camera has it. Grasps
are found in the wrist picture and made by the same arm; if the bin's walls
(which bound the search, and which the fingers must not hit) are taken from
the overhead box, they are in the wrong place by that error. This skill moves
the box to the walls the wrist camera sees.

## How

1. **Wall points.** Wrist depth pixels (every second one, none nearer than
   `min_range`) that are wall: by `wall_mask`, by the cardboard colour rule
   (red over blue by more than 22, red over 70), or all of them with
   `wall_colour="any"`. Lifted to the world with the camera's pose and kept
   from 5 cm over the bin's support to 3 cm over its rim.
2. **Each wall.** In the bin's frame, the points within 9 cm of where a wall
   is expected and at least 3 cm from its corners are histogrammed across the
   wall in 4 mm steps. With over 600 such points and over 60 in the fullest
   step, that step is the wall's sheet; its offset from the wall's expected
   middle is that wall's reading.
3. **This look.** The mean of the end walls read gives the shift along the
   bin's long axis, the side walls along the short one. An axis with no wall
   read contributes 0, so a look that sees one side wall only corrects one
   direction.
4. **Over looks.** A reading is counted when it is under `max_shift` from
   `bin_obb` and, once two are held, under `outlier` from their median.
   `anchored_obb` is `bin_obb` moved by the median of the counted readings.

## Calling it

- `bin_obb` is the box from the fixed camera **every time**, not the anchored
  one from the last call: the readings are shifts from that box.
- Keep `shifts` in the graph state and pass it back as `past_shifts`.
- If no grasp is found in `anchored_obb`, the source task tried once more with
  `bin_obb` as given. That fallback is the caller's.
- One axis only is corrected when one pair of walls is out of view;
  `walls_read` says which were read.

## Gotchas

- **Colour.** The built-in rule is brown cardboard against grey and black
  parts. A grey tote full of grey parts needs `wall_mask`. With
  `wall_colour="any"`, a pile leaning on a wall is part of the sheet search
  and can pull it inwards.
- **Resolution.** The shift is quantised to the 4 mm histogram step.
- **Thresholds 800 / 600 / 60** are point counts at one camera, one stride
  and one viewing distance. A lower-resolution camera will read no wall until
  they are lowered; they are module constants, not inputs, to keep that a
  deliberate change.
- **Low bins.** Walls under about 8 cm leave little between 5 cm over the
  support and the rim. Use `locating-a-container-from-the-wrist-camera`
  (the rim outline) for those.

## Constants, and what they were measured on

One YAM station: an i2rt YAM arm with a wrist D405 looking into a
449 x 262 mm corrugated cardboard box, 154 mm tall, from about 0.3 m over the
counter, hand leaning 12 degrees; 2026-11-15.

- `SPAN = 9 cm`, `BIN = 4 mm`, `WALL_FROM = 5 cm`, `END_CLEAR = 3 cm`,
  `MIN_POINTS / MIN_WALL_POINTS / MIN_PEAK = 800 / 600 / 60`: chosen on that
  box and camera, not swept.
- `WALL_HALF_THICKNESS = 2 mm`: that cardboard.
- `MAX_SHIFT = 10 cm`: a guard.
- `OUTLIER = 3 cm`: added after one look read the walls 7 cm out, and the
  pile was then not seen at all in the box as moved.

## Evidence

Real robot, one station, 2026-11-15; the task ended that day with the
anchoring on by default. What is on record is the failure that shaped it (the
7 cm reading above) and that picking from the anchored box held about one
grasp in two (see `proposing-open-hand-grasps-in-a-dense-pile`). No run
compared anchored with un-anchored picking, and the shift was never checked
against a ground truth, so the size of its benefit is not known. Nothing was
run in a simulator suite. Maturity is `community`.

## Required end states

- `anchored`: `anchored_obb`, `shift`, `used_shift`, `shifts` (one longer).
- `outlier`: `shift` read but not counted.
- `walls_not_seen`: nothing read.
