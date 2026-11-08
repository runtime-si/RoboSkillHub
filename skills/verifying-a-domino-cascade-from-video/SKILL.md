---
name: verifying-a-domino-cascade-from-video
description: Decides from a short RGB(-D) clip of the overhead or a wrist camera
  which tiles of a planned domino line fell and when, how fast the wave ran and
  where it stopped. Per planned slot it measures a fall angle each frame (depth -
  a rotated-box height profile fitted along the tile's fall corridor; RGB only -
  how much of the gap in front of the tile its body covers; optionally a tracked
  point (track) or tracked mask (sam3) fitted to the rotated box's image), drops
  frames where a hand or the arm hides the tile, confirms a fall after 3 frames
  past the fallen threshold and dates it back to when the tile left the upright
  band (predicators' cascade certificate), then routes all_fell, stopped_at
  (with the first standing tile), not_triggered, fell_before_trigger, irregular
  or unverifiable. No motion. Use after triggering a domino line or curve to
  check the chain, before re-triggering from where it stopped or rebuilding.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [domino, cascade, video, verification, onset, rgb-d, occlusion]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/BasisResearch/predicators/blob/afb6f2918091f5ca372a75b9a335aea317a8a209/predicators/envs/pybullet_domino/cascade_certificate.py
      upstream_license: MIT
gap:
  allowed_tools:
    - robot.get_observation
    - track.points
    - sam3.tracker_init
    - sam3.tracker_update
    - sam3.tracker_close
  required_inputs:
    # OPTIONAL extra inputs (gap has no list type, so they can only be wired from
    # a script node in the same subgraph): `slots` (planning-a-domino-layout's
    # `slots` or `slot_poses`, path order, slot 0 the trigger tile) instead of
    # first_slot + pitch + n_tiles (a straight line); `frames` (the clip: a list
    # of CameraFrames with `timestamp`, or a dict of stacked `rgb` / `depth` with
    # `intrinsics`, `pose`, `timestamps`); `tiles` (reading-domino-pips-and-orientation's
    # tiles: their centres replace the slots' within 15 mm); `fps`, `trigger_frame`
    # / `trigger_time`, `signal`, `tracker`, `tile_size`, `table_z`, `tile_colour`,
    # `min_visibility`, `upright_deg`, `fallen_deg`.
    first_slot: Se3Pose
    pitch: float
    n_tiles: int
  produces_outputs:
    n_tiles: int
    n_fell: int
    stop_index: int
    last_fallen_index: int
    stop_point: Vec3
    wave_speed: float
    wave_rate: float
    first_onset_time: float
    last_onset_time: float
    signal: str
  exit_conditions:
    all_fell: Every tile fell, one after another from slot 0 (onsets never run backwards by more than a frame plus their uncertainty) and none before the trigger; `wave_speed` (m/s along the path) and `wave_rate` (tiles/s) are robust fits over the onsets, `intervals` the successive propagation intervals.
    stopped_at: Tiles 0 to k-1 fell and tile k was seen standing after the wave reached it, with every tile after it standing or unseen; `stop_index` is k, `stop_point` its centre (where to re-trigger or fix the gap).
    not_triggered: No tile fell and the first tile was seen standing.
    fell_before_trigger: A tile started falling before `trigger_frame` / `trigger_time` (the frame the push began) - the line was knocked, not triggered.
    irregular: The falls are not one wave from slot 0 - a tile fell while the first stood, a tile fell beyond a standing one, or a later tile started falling clearly before an earlier one.
    unverifiable: No clip, no slots, a tile already down in its first frames, too few usable frames to tell whether the first tile or the tile where the wave stopped fell (a hand over it, out of view), or a raised error.
  canonical_scripts:
    - verify_cascade: scripts/verify_cascade.py
  examples:
    - title: Check a straight domino line after the trigger push, from the overhead camera
      path: examples/canonical_subgraph.json
---

# verifying-a-domino-cascade-from-video

The arm pushed the first tile; did the line go down, all of it, and in order?
This skill watches a clip and answers per tile: fell or standing, the frame
the fall began, how fast the wave ran, and where it stopped. It is the onset
rule of predicators' cascade certificate (`_topple_onset`) and its design
note's rules for perceived angles, put on angles measured from a camera at
the slots planning-a-domino-layout planned.

## Inputs

- The clip, `frames`: CameraFrames (`rgb`, `depth` if any, `intrinsics`,
  `pose` camera-to-world, `timestamp` in s or ns), or one dict of stacked
  `rgb` / `depth` arrays with `intrinsics`, `pose` (or `poses`) and
  `timestamps`. Without timestamps `fps` (else 30 fps, with a problem noted).
  Record it at 60 fps or more (below). `record_frames` > 0 instead polls
  `robot.get_observation` for `camera_name`: fine for fell / stood, too slow
  for onsets.
- The line: `slots` (planning-a-domino-layout's `slots` or `slot_poses`, in
  path order; slot 0 is the trigger tile; each pose's x axis is the way it
  topples) or, for a straight line, the typed `first_slot` + `pitch` +
  `n_tiles`. `tiles` from reading-domino-pips-and-orientation move a slot to
  the tile seen within 15 mm of it. `tile_size` (`{height, width, thickness}`
  or (h, w, t); 48 x 24 x 7.5 mm), `table_z` (else the slots' z).
- `trigger_frame` or `trigger_time`: when the push began, for
  `fell_before_trigger`.
- `signal`: `auto` (depth when every frame has it, else rgb), `depth`, `rgb`.
  `tracker`: `none` (default), `track` or `sam3` (below).
- `min_visibility` (0.5), `upright_deg` (15), `fallen_deg` (45), `min_frames`
  (3), `tile_colour` (rgb; else sampled from the tiles' tops).

## How it decides

1. **Each tile's fall frame.** The pivot is the slot's front bottom edge; x
   runs along the facing, z up from the table. The corridor a falling tile
   sweeps runs from its back face to 2 mm short of the next tile's back face
   (the last tile: one tile height). The face gap there gives the **contact
   angle**, where the top meets the next tile: asin(gap / h), 38 deg at the
   auto pitch. A tile the next one stops rests there.
2. **A fall angle per frame.**
   - *Depth* (default): the frame's valid depth pixels round the line are
     lifted to the world (geometry.mask_to_world_points' pinhole model and
     0.015-20 m bounds, in numpy: one tool call per frame would ship every
     depth image). Within the tile's width less 3 mm, the corridor is cut
     into 2 mm bins; each bin's height (its 90th percentile, less that
     percentile's excess over the median on the table in the first frames:
     the depth noise) is fitted against the height profile of the tile
     rotated 0-90 deg about its pivot (0.5 deg steps).
     Height above the model costs 0.3 of height below it (truncated at
     15 mm): in a fallen row the tile before lies on this one's back. The
     tile's position (along and across) is corrected from its top in the
     first frames (up to 8 mm), and the gaps and contact angles recomputed
     from where the tiles stand. A bin whose median is over the tile's top
     by 12 mm is an occluder.
   - *RGB only*: pixels are tile (any shade of the tile's colour), table
     (within 4 median deviations of its colour) or an occluder (neither). The
     angle is asin(c * L / h), c the share of the gap strip in front of the
     tile (L long, 0.6 of its width) newly covered by tile-coloured pixels,
     against each pixel's first frame with no occluder over it. It saturates
     at the contact angle; the last tile reads up to 90 deg.
   - *`tracker: track`* (TAPNext++, GPU): each tile's top centre is tracked
     from frame 0; the angle is where the rotated top's image path passes
     nearest the tracked pixel; `visible` False (or the pixel more than
     4 px + 10 % of the tile's height off the path) drops the frame.
   - *`tracker: sam3`* (GPU): a sam3 tracker per tile, seeded with the
     tile's box in frame 0; the angle is the one whose projected rotated box
     best overlaps the mask (IoU, 3 deg then 0.5 deg); a held-out mask
     (confidence 0) drops the frame.
   A tracker that raises (no GPU host, tool not installed) is noted in
   `tracker_fallback` and the CPU signal is used.
3. **Visibility gate.** Visibility is the usable depth bins (rgb: the share
   of the tile's corridor not occluded; sam3: mask area over the fitted box's
   projected area) over its 90th percentile across the clip. Frames under
   `min_visibility`, or whose profile fit misses by more than 6 mm (or 3x
   the tile's first-frames fit) are **missing data**.
4. **Each tile against its own first frames**: the median of its first 3
   usable angles is subtracted (unless it is over the upright band: then it
   leans, and is noted). Already past the fallen threshold in its first
   frames: `down_at_start`.
5. **Onset (upstream's rule).** A fall needs 3 consecutive usable frames at
   or past the fallen threshold; missing frames neither extend nor break the
   run; a run cut by the end of the clip counts from 2 frames, unconfirmed.
   Its onset is the frame after the last usable frame inside the upright
   band, with `onset_uncertainty_frames` the hidden frames after it, and
   `onset_time` the sub-frame time the angle crossed the upright band
   (linear between those two frames). Thresholds per tile: fallen =
   min(`fallen_deg`, contact - 8 deg) (rgb: the angle at which 0.75 of the gap
   is covered), upright = min(`upright_deg`, half the fallen or contact
   angle).
6. **States.** `fell`; `standing` when seen in at least 3 usable frames after
   the latest onset before it and last below the fallen threshold;
   `unknown` otherwise; an unknown tile before a fallen one fell (`inferred`).
7. **The line.** Wave speed and rate: Theil-Sen slopes of onset time over
   arc length and over index (timed, non-inferred tiles). Intervals:
   successive onset differences; under 2 frames, a problem is noted (the
   order is not resolvable). Then the exits, in this order: a tile
   `down_at_start` -> unverifiable; an onset before the trigger (by more
   than a frame plus its uncertainty) -> fell_before_trigger; none fell ->
   not_triggered (first tile standing) or unverifiable; the first tile did
   not fall -> irregular; onsets running backwards -> irregular; all fell ->
   all_fell; else k = first non-fallen tile: unknown -> unverifiable, a tile
   after it fell -> irregular, else stopped_at.

## Outputs

`route`, `detail`, `n_tiles`, `n_fell`, `stop_index` (-1 unless the wave
stopped), `last_fallen_index`, `stop_point` (the stop tile's centre),
`wave_speed` (m/s along the path), `wave_rate` (tiles/s), `first_onset_time`,
`last_onset_time` (s from the first frame), `signal` (`depth`, `rgb`,
`track`, `sam3`), `tracker_fallback`, `tiles` (per tile: `state`,
`onset_frame`, `onset_time`, `onset_uncertainty_frames`, `fall_frame`,
`confirmed`, `inferred`, `start_angle`, `max_angle`, `final_angle`,
`usable_frames`, its `upright_deg` / `fallen_deg` / `contact_deg`,
`position`, `face_gap`), `intervals`, `angles` and `visibility` (per tile
per frame), `times`, `problems`.

## The constants, and why

- **`TOPPLE_MIN_FRAMES = 3`**: upstream `_TOPPLE_MIN_STEPS`.
- **`FALLEN_DEG = 45`**, **`UPRIGHT_DEG = 15`**: the design note confirms at
  45-60 deg, far above the 4 -> 13 deg drift and the 29 deg occlusion
  artefact it measured, and its markerless check used 15 deg upright.
  Upstream's simulator uses 10 / 5 deg, inside a perceived angle's noise.
- **`CONTACT_MARGIN_DEG = 8`**: a tile stopped by a standing neighbour rests
  at the contact angle (38 deg at the auto pitch, under 45), so the confirm
  threshold sits under it; 5 mm of placement error moves it 7.6 deg, and a
  5 deg margin missed 2 of 12 stopped chains' last tile with 1.5 mm
  placement noise.
- **`RGB_CONFIRM_COVER = 0.75`**: in rgb the measure is the gap's covered
  share; a stopped tile with its neighbour 5 mm closer still covers 0.89.
- **`MIN_VISIBILITY = 0.5`**: the note's 40-70 % band (80 % deleted a real
  topple); `VIS_REFERENCE_PCT = 90` instead of the note's median, so a hand
  hovering through most of a clip does not set the reference.
- **`END_RUN_MIN_FRAMES = 2`**: upstream counts a run cut by the end at any
  length; one noisy last frame did emit a fall with 4 mm depth noise.
- **`OVER_WEIGHT = 0.3`, `FIT_CAP = 15 mm`, `MAX_FIT_COST = 6 mm`**: with
  a symmetric cost fallen tiles in a row fitted at 6.9 mm (the tile before
  lies on them) and were dropped; 2 mm depth noise fits at 1-2 mm.
- **`BIN_TOP_PCT = 90`, less the noise excess**: a wrist camera from the side
  sees the table under a leaning tile beside the tile itself, and a per-bin
  median read 0 deg for tiles at 23-48 deg there; a plain 90th percentile
  read a table with 4 mm depth noise as a flat tile. The excess is pooled
  over all tiles, so a hand over one tile in the first frames does not set
  it.
- **`BIN = 2 mm`**: the overhead camera's pixel at the table. **`LATERAL_MARGIN
  = 3 mm`, `GAP_MARGIN = 2 mm`, `MAX_SHIFT = 8 mm`, `OCCLUDER_ABOVE = 12 mm`**:
  choices; a placed tile is within a few mm of its slot.
- **`RGB_STRIP_WIDTH = 0.6`**: at 0.8 the last tile of a line placed 6 mm and
  6.6 deg off its slot covered too little of its strip once flat.
- **`RGB_SHADE = (0.4, 1.25)`, `RGB_SPREAD = 4`, `RGB_MIN_CLASS_DIST = 18`**:
  a falling tile's faces turn from 1.0 to 0.55 of the lit top in the renders;
  a dark grey hand is ~0.2 of a white tile.
- **`ORDER_TOLERANCE_FRAMES = 1`, `MIN_INTERVAL_FRAMES = 2`**: onsets are 2-6
  frames apart at 30 fps on a real four-tile cascade (the note); at 15 fps
  the last two were 1 frame apart and the order unresolvable.
- **`TRACK_MAX_RESIDUAL_PX = 4`**: the track tool's tracks stayed within 4 px.

## Measured

All CPU, on rendered clips (`tests/bundles/_cascade_render.py`; the sweep is
`python -m tests.bundles._cascade_eval --seeds 4 [--extra]`). A line of 10
tiles (48 x 24 x 7.5 mm) at a 37 mm pitch, placed with 1.5 mm and 2 deg of
noise off the planned slots; the wave runs at 0.8 m/s (the planner's
StackedLine rate), each tile accelerating to its contact angle as the next
starts, then leaning on it; a tile that is hit but stands stops the wave; the
last tile falls flat. Cameras: an overhead 640 x 480 (fx 461) 0.9 m up, a
wrist D405-like 640 x 480 (fx 390) ~0.35 m away looking from the side and
above, and the **station's real overhead camera** (its ChArUco-calibrated
pose and intrinsics, ~31 deg tilted). Depth noise 2 mm at 0.9 m growing with
range squared, quantised to 1 mm, 1 % holes; rgb noise 4 grey. Hands: none;
*pusher*, a 70 x 90 x 40 mm box hovering over the first tiles 55-95 mm up
until 40 ms before the push, then pulling back (the design note's Mechanism
A); *sweep*, the arm passing over tiles 3-5 mid-cascade (120 ms). Each row:
4 seeds x (all fall, stopped at a random tile 2-8, not triggered) = 12 clips.

| signal / camera | fps | hands | routes right | tiles right | stop index error | onset - true start (median; p90 spread) | wave speed error (median / max) |
|---|---|---|---|---|---|---|---|
| depth / overhead | 30, 60, 120 | none, pusher, sweep | 108/108 | 1080/1080 | 0 | +24-27 ms; 2-14 ms | 0.1-1.4 % / 3.2 % |
| depth / wrist | 30, 60, 120 | none, pusher, sweep | 108/108 | 1080/1080 | 0 | +22-26 ms; 4-32 ms | 0.2-1.9 % / 5.0 % |
| rgb / overhead | 30, 60, 120 | none, pusher, sweep | 108/108 | 1080/1080 | 0 | +23-28 ms; 7-19 ms | 2.2-3.3 % / 4.0 % |

At 60 fps, overhead camera, no hands:

| case | depth: routes / tiles | rgb: routes / tiles |
|---|---|---|
| pitch 24 mm (contact 20 deg), 0.95 m/s | 12/12, 120/120 | 12/12, 120/120 |
| pitch 48 mm (1.0 h), 0.68 m/s | 12/12, 120/120 | 12/12, 120/120 |
| placement error 4 mm, yaw 4 deg | 12/12, 120/120 | **11/12, 119/120** (an all-fell line read stopped at its last tile: placed 6.9 mm back, 6 mm aside and 6.6 deg off, it covered too little of its strip) |
| depth noise 4 mm, 5 % holes | 12/12, 120/120 | 12/12, 120/120 |
| arc, radius 0.25 m (slot list) | 12/12, 120/120 | 12/12, 120/120 |
| 20 tiles | 12/12, 240/240 | 12/12, 240/240 |
| **station camera**, its measured 3.5 mm depth noise and 11 % holes | 12/12, 120/120 | 12/12, 120/120 |

In all: 491 of 492 clips routed right, 5159 of 5160 tiles right, stop index
never off when the route was right. The onset lags the true start of the
fall by a constant ~25 ms (the time the synthetic tile takes to reach the
15 deg upright band); against the frame where the true angle first leaves
that band it is 0 frames off (median) and -1 to +2 frames with the tile in
view; hidden frames widen it (to -8 frames at 120 fps under the sweep, with
`onset_uncertainty_frames` saying so). The design note's hand-built checks
are unit tests: a 4 -> 13 deg drift and a 29 deg one-frame spike 15 frames
before the fall give no onset, gate-dropped frames mid-fall still give the
right onset. A 58-frame clip of 10 tiles takes 0.4 s (overhead depth), 1.1 s (wrist depth) and 0.03-0.2 s (rgb) on a CPU; the
renders, not the skill, dominate the sweep's time.

Each fix above came from a failed sweep row: a symmetric fit dropped fallen
tiles in a row (the tile before lies on them); a per-bin median read 0 deg
from the wrist camera; a per-bin 90th percentile read noisy table as a flat
tile and a single noisy last frame as a fall; a 5 deg contact margin missed
stopped chains' last tile; planned instead of measured gaps put the next
tile into the corridor.

## Preconditions and boundaries

- **Frame rate.** At 0.8 m/s and a 37 mm pitch a new tile starts every
  46 ms: 1.4 frames at 30 fps. States and the stop index do not care; the
  per-tile order and intervals do. Record at 60 fps or more and never
  decimate (the design note).
- **The camera sees the line**: every slot's corridor in view; the overhead
  camera or a wrist D405 parked 0.3-0.5 m away. A moving wrist camera works
  in depth mode (each frame's pose is used); the rgb mode compares pixels
  with the first frames and needs the camera still.
- **RGB only needs tile, table and hand colours apart**: white or ivory
  tiles on a darker, coloured table and a dark grey gripper are fine; a
  table of the tiles' hue, or a white glove, is not (a problem is noted when
  the table looks like a tile). Shadows are not rendered in the synthetic
  clips. It measures at the planned slots (pass `tiles` to correct them), so
  large placement errors cost it more than the depth mode.
- **Dense lines**: at 24 mm pitch the contact angle is 20 deg and the
  thresholds shrink with it (upright 10, fallen 15 deg): depth noise matters
  more there.
- **The trackers are not measured on a GPU here.** This round ran on CPU only;
  the `track` and `sam3` paths are tested with fakes driven by the renders'
  ground truth (plumbing, angle fits, fallback), not with the models.
  **Live check pending: needs a GPU host.**
- **No real cascade clip was available**: the numbers are from rendered
  clips (box tiles, no bounce, sliding or shadows). The station camera case
  uses the real overhead camera's calibration and its measured depth noise.
- No motion: a recover step (re-trigger from `stop_point`, rebuild) is the
  graph's.

## Provenance

`scripts/verify_cascade.py` is adapted from predicators
(BasisResearch/predicators at afb6f291, MIT, Copyright (c) 2022 Rohan
Chitnis and Tom Silver, `LICENSE.predicators`):
`predicators/envs/pybullet_domino/cascade_certificate.py` (`_topple_onset`,
`_TOPPLE_MIN_STEPS`), the upright / fallen bands of
`components/domino_component.py`, and the onset rules of
`docs/envs/domino/continuous-perception.md` sections 3.3-3.4 (visibility
gate as missing data, confirm then backdate, each tile against its own first
frames, propagation intervals, no jump or speed gate, at least 60 fps).
Changed: the simulator's roll becomes a measured angle (above); `is_held`
becomes the visibility gate; the carry floor of the backward search is
dropped; a run cut by the clip's end needs 2 frames; the confirm threshold is
capped under the contact angle. Not ported: the certificate's staging, Push
label and counterfactual probe rules (they need the simulator) and the
markerless box-fit pipeline (a private submodule, BabyRobotPredicator).
