---
name: resolving-reconstruction-doubts-with-wrist-views
description: Settles what a scene reconstruction from a fixed head or overhead
  camera left in doubt (a size the far camera cannot resolve, a thin or dark
  object that vanished from its depth, a height, a hidden face, a shape
  profile, how an object leans, where it really stands) by carrying the arm's
  wrist RGB-D camera to each doubted point, looking from several tilts and
  ranges without touching anything, and measuring the object there - top
  height over the surface, diameter by depth band under the top, lean, a
  footprint rectangle, position - each value with the views it came from,
  their agreement, and the saved frames, ready to write into the simulator
  twin. Use when a reconstruction (Agent A) has doubts the head camera cannot
  settle and a list of 3D points to check is in hand, before trusting the
  twin for search or policy tuning, and whenever a real run fails in a way the
  twin cannot show; the arm carrying the wrist camera must be free.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, active-perception, wrist-camera, eye-in-hand, real2sim, twin, measurement, d405, cpu]
  hub:
    stage: perceive
    embodiments: [any]
    tuned_on: [yam_i2rt]
    version: 0.1.0
    tasks: [real2sim, pick-place, insertion, sorting]
    notes: >-
      Generalized from the YAM pen station's pen_shape_probe.py and the analysis of its
      capture pen_shape_20261109-001140 (not published); its constants are that run's.
    provenance:
      authors: [runtime-si]
gap:
  requires: {connector: [robot.describe_gripper, robot.describe_workspace]}
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.solve_ik
    - robot.describe_workspace
    - robot.describe_gripper
    - sam3.segment_text
  required_inputs: {}
  # Inputs (not listed: gap has no list or open-dict type here, so they are wired from a
  # script node in the same graph): doubts (list of {point, what, id?, radius?, colour?,
  # object_name?, top?, bands_mm?, base_point?, lean_depth?, max_height?, prior?, write?}),
  # arm_ids, arm_id, camera, surface_z, tilts_deg, distances_m, azimuths_deg,
  # max_views_per_doubt, segmenter, out_dir, twin_path, keep_images, check_reach,
  # reach_tolerance, return_to_start. Outputs also: doubts (per-doubt status, quantities,
  # views, failures), frames (camera_pose, tool_pose, intrinsics, file, rgb/depth when
  # kept), twin_updates (value, prior, write_to, source {views, frames, measured_at}).
  produces_outputs:
    n_doubts: int
    n_measured: int
    n_views: int
  exit_conditions:
    measured: Every doubt got at least one view, and every asked quantity was read from the fused points with no quantity on which fewer than half of the views that read it alone agree with the fused value (top height within 5 mm, band diameter within 4 mm, footprint sides within 5 mm, top centre within 5 mm, lean within 3 deg); each value names its views and frames, and twin_path (if given) now holds it under measured.<doubt>. The hands are back at their start poses (unless return_to_start is false).
    inconsistent: Some doubt was seen, but for at least one quantity fewer than half of the views that read it on their own agree with the fused value (a neighbour inside the crop, a selection that catches two objects, an object that moved between views); that doubt's values are returned with every view's reading and are not written to the twin.
    no_view: Some doubt's view poses were reached and framed the target, but its points were too few (under 20 within its radius), or an asked quantity no view could read (no band seen round, no surface height for a height, fewer than 4 lean slices); `failures` and `points_per_view` say which.
    not_reachable: For some doubt no view pose on any given arm cleared the surface floor (tool 12 cm over it) and the standoff (finger reach + 30 mm from every doubted point), passed the IK preview, was reached within 15 mm and showed the target in the image; `failures` lists every attempt. Nothing was measured for it.
  canonical_scripts:
    - resolve_doubts: scripts/resolve_doubts.py
  examples:
    - title: Look at three doubted pen caps and the holder under them, write the measurements into the twin
      path: examples/canonical_subgraph.json
---

# resolving-reconstruction-doubts-with-wrist-views

A reconstruction from a fixed camera a metre away has blind spots that no
amount of fitting removes: a 2 mm difference in a cap's width is under a
pixel, a black tray's top is a hole in its depth, a pen's lean is a few
pixels along the line of sight, and the faces turned away were never seen.
The robot's own wrist camera, carried 15-25 cm from the place in question,
resolves 1 mm as 2-3 px and sees round the object. This skill turns each
doubt into a short tour of looks (no contact) and a measurement written back
into the twin with its source, which is the fastest way to know the real
state and the most direct way to correct the simulator.

Its first instance found, in one two-minute look, why a week of real grasps
had failed: the twin drew a pen as a 16 mm cylinder standing upright in a
41 mm tray, while the real caps are 18.5 mm with an 11.5 mm knob, the tray is
~25 mm and the pens lean 3-7 deg outward, so neighbouring caps touch 3-10 mm
under their tops, exactly where the fingers were sent.

## How it composes

- **Before it**: the reconstruction (any perception that builds the twin)
  emits doubts: a 3D point in the robot base frame plus what is doubted.
  The point is where to look, not a pose to trust: a `position` doubt crops
  6 cm round it.
- **After it**: `twin_updates` (or the twin JSON at `twin_path`) feed the
  twin rebuild; a graph routes `inconsistent` to another look (more
  azimuths, a tighter `radius`, a `colour` or `object_name` selection),
  `not_reachable` to the other arm (`arm_ids`) or a human, and `no_view` to
  a different selection or a closer range.
- **Not `refining-target-from-wrist-closeup`**: that one moves one coarse
  point by pointing; this one reads sizes, heights and shapes from several
  views and reports their agreement.
- **Not a verifier**: it never touches. Contact probing belongs to
  `verifying-insertion-by-probe-push` and its kin.

## Inputs

`doubts`: a list of dicts, robot base frame, metres:

- `point` (required): the doubted place (Vec3 dict or [x, y, z]); for an
  upright object, a point a little under its top.
- `what`: one or several of `height` (top z and height over the surface),
  `profile` (diameter per depth band under the top), `width` (the
  footprint's short and long side), `footprint` (rectangle centre, sides,
  yaw), `lean` (deg off vertical and toward which azimuth), `position` (the
  top's centre and its offset from `point`). Default `height`.
- `id`, `radius` (crop round the vertical axis through `point`; 30 mm, 60 mm
  for `position`; touching neighbours need less, the pen caps 12 mm),
  `colour` (HSV ranges on OpenCV's scale, `{"h": [45, 85], "s": [120, 255],
  "v": [90, 255]}`; a hue range lo > hi wraps through red; `{"v": [0, 60]}`
  selects dark things), `object_name` (masked with `segmenter`, default
  `sam3.segment_text`; the mask holding or nearest the target pixel),
  `top` (`percentile`, or `edge` for a body with sparser things above it, a
  holder full of pens), `bands_mm` (default the pen bands 0-3, 3-6, 6-10,
  10-16, 16-24), `base_point` (a known foot: lean is then the line from it
  to the top's centre), `lean_depth`, `max_height` over the surface (keeps
  the gripper's own fingers out of a dark selection), `prior` (the twin's
  values, echoed in the updates), `write` (quantity -> dotted twin key, e.g.
  `{"height_over_surface": "black_holder.size_measured_height_m"}`).

Also: `arm_ids` (tried in order per doubt until one gets views; default
`[arm_id]`), `camera` (else the `wrist` / `eye_in_hand` camera riding
nearest the tool), `surface_z` (else `robot.describe_workspace`),
`tilts_deg` (30, 60, 75, 85), `distances_m` (0.15, 0.20, 0.25),
`azimuths_deg` (0: the arm's own side; add 60, -60 to see round),
`max_views_per_doubt`, `out_dir` (frames as npz with `rgb`, `depth_m`, `K`,
`T_base_camera`, the probe's format, plus `measurements.json`), `twin_path`,
`keep_images` (default: when there is no `out_dir`), `check_reach`,
`reach_tolerance`, `return_to_start` (true).

## What it does

1. Reads the observation once: per arm, the wrist camera and its
   camera-to-tool transform from that frame's pose (no calibration file);
   the surface height from `robot.describe_workspace`, the fingers' reach
   from `robot.describe_gripper`.
2. Per doubt and arm, per tilt and azimuth, the nearest range whose look
   pose (the camera's optical axis through the point; the hand keeps its
   start roll, turned by the smallest rotation that aims the camera) keeps
   the tool 12 cm over the surface and finger reach + 30 mm from every
   doubted point, and passes `robot.solve_ik`.
3. Travels there by rising in place to 10 cm over the highest doubted point,
   crossing at that height, then going down onto the view; waits until two
   observations agree on the tool (0.5 mm, 0.1 deg); keeps the frame if the
   hand stands within 15 mm of the view and the point projects inside the
   image.
4. Selects the object's points per frame: segmenter mask and/or colour,
   above the surface (+4 mm), under `max_height`, within `radius` of the
   axis.
5. Fuses all views (the axis re-centred on the fused points) and measures
   (`scripts/doubt_geometry.py`, pure numpy):
   - **top**: 99th percentile of heights, or the `edge` (upper edge of the
     highest 2 mm bin holding half the fullest bin's points).
   - **profile**: per band under the top, its centre and the 90th
     percentile radius, doubled; a band counts only with 20 points filling
     80 % of the 36 sectors round its centre.
   - **lean**: from `base_point` to the top's centre, else a line fit
     through 4 mm slices that are seen round (each centred on its trimmed
     extent, not its median).
   - **footprint**: smallest-area rectangle of the hull, its sides taken
     between the 1st and 99th percentiles along its axes.
   - **position**: the centre of the top band.
6. Checks each quantity view by view against the fused value: a view reads
   it on its own only if it sees enough (the top band round for a top, the
   band round for a diameter, the object round for a footprint); with two or
   more such views, fewer than half agreeing makes the doubt
   `inconsistent`. With fewer the value says `checked: false`.
7. Saves frames and `measurements.json`, writes measured, consistent values
   to `twin_path`, and takes each moved hand back to its start pose by the
   same transit.

## Required end states

- `measured`: every doubt `measured`; values, views and frames in `doubts`
  and `twin_updates`.
- `inconsistent`, `no_view`, `not_reachable`: the first of these, in that
  order, that any doubt has. The other doubts are still measured and
  returned (`n_measured`), and their values written when consistent; a graph
  can use them and re-look only the failed ones.

## The constants, and how they were measured

All on the YAM station (right arm, its D405 at 640 x 480,
fx ~395 px) in the pen-shape capture `pen_shape_20261109-001140`: 11 views of
a 3 x 4 holder of EXPO pens, three targets, tilts 30/60/75/85 deg at
0.15-0.25 m.

- **Tilts 30/60/75/85 deg and ranges 0.15-0.25 m**: the probe's. The views
  that read the cap's top round came from 60-75 deg, the bands under it from
  60-85, the top over a tray from 30-75.
- **Tool 12 cm over the surface**: the probe's look-pose floor
  (`steps.look_at`). **Standoff = finger reach + 30 mm**: the probe's
  closest look put the tool centre 41 mm from its target, fingers reaching
  10 mm, nothing touched.
- **Top at the 99th percentile**: green cap, fused, 89.7 mm (base z); the
  95th read 88.4 (into the knob's rounded edge), the maximum 91.0 (a few
  high pixels of the noisiest view).
- **Edge top (half the fullest 2 mm bin)**: the black holder's dark points
  are dense to 24-26 mm over the table, then a tail of dark pens to 60 mm;
  the edge reads 28 mm fused and 24-29 mm per view at 30-75 deg, the 99th
  percentile 59 mm.
- **Diameter = 2 x p90 radius, bands of 20+ points over 80 % of sectors**:
  green cap, 6-10 mm band, 18.1 mm fused (the journal: 18-19 mm); the 22 per-view
  bands under it with less than 80 % coverage scattered over 9.5-20.4 mm.
- **Tolerances** (top 5 mm, diameter 4 mm, footprint 5 mm, centre 5 mm,
  lean 3 deg) and **half the views must agree**: eight of the ten views that saw
  the green cap's top round read it within 3.7 mm of each other (the two
  85 deg ones 5 and 16 mm low: they see the side); its 6-10 mm
  band read 15.7-18.3 mm in the three views that saw it round.
- **Settling (0.5 mm, 0.1 deg)**: the real YAM server answers a move with
  the tool 3-11 mm off and closes with a 0.26 s time constant
  (`settle_probe`, 2026-11-08); at 0.2 m each bound is ~1 px.
- **Transit 10 cm over the highest point**: not measured; a margin that
  keeps the open fingers 9 cm over anything doubted while the arm crosses
  (the probe planned straight moves instead).

## Measured: the pen-shape capture, replayed offline

`doubt_geometry` on the saved frames, table at z = -0.033 m:

| doubt | selection | result |
| --- | --- | --- |
| green cap, `profile` + `height` + `position` | hue 45-85, radius 12 mm | `measured`; 122.7 mm over the table (9 of 10 views within 5 mm); bands 0-3: 10.2 mm (knob), 3-6: 13.7 (the knob's foot), 6-10: 18.1, 10-16: 17.9 (cap), 16-24: 15.9 mm (barrel); top centre within 0.6-4.3 mm of each of the 8 views that see it round |
| pink cap, same | hue 160-178 | `inconsistent`: its 0-3 mm band reads 19.1 mm fused but 8-11 mm in each of the three views that see it round; the hue range also takes the red and purple neighbours touching it |
| orange cap, same | hue 8-20 | `measured` but wrong (cap bands 12-15 mm, top unchecked): the hue range catches only part of the cap. A selection that misses half an object is not caught by agreement; check `points_per_view` and give a mask (`object_name`) |
| black holder, `height` | V < 60, `edge`, under 60 mm | 28.0 mm over the table (views 24-29 mm, 6 of 8 agree); the twin had 41 mm, the journal recorded ~23-25 mm from the same frames |

The journal (`yam_gap/pens/logs/journal.md`, "A on the arm: what a pen is")
recorded caps 18-19 mm with an 11-12 mm knob, barrels 16 mm, the holder
~23-25 mm, pens leaning 3-7 deg outward (from the caps 3-12 mm off their
sockets over ~95 mm), and that neighbouring caps touch where the fingers were
sent (`runs/real-v1/diagnosis.md`: picks stopped 8-21 mm above the cap on 7
of 9 tries). Lean without a `base_point` is weak on a short cap: the axis
fit over the caps' 30 mm read 6-12 deg, unchecked by single views; give the
socket as `base_point` when it is known.

## Limits

- One arm per doubt view set; two arms are not fused into one doubt.
- The look poses are not collision-checked beyond the IK preview, the
  surface floor and the standoff; the arm's links are the connector's
  planner's business.
- Footprints of objects wider than one view are fused but rarely checked
  (few views see them round): `checked: false`.
- Not run end to end on the arm yet; the pen probe that it generalizes was.
