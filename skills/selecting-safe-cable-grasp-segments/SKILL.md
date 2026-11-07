---
name: selecting-safe-cable-grasp-segments
description: Ranks top-down grasp points along a cable's fitted centreline that
  keep the fingers off its ends (a larger margin at the plug end), off hidden
  stretches and crossings, off tight bends, and clear of obstacle boxes, other
  cables and the cable's own other stretches, then checks each with
  robot.grasp_frame (jaws across the local tangent) and robot.solve_ik and
  returns the reachable ones as grasp poses, best first. Use when a cable must
  be picked up somewhere along its length (to pull it, lay it, hand it over or
  lift it clear) and the centreline comes from curve.fit_centerline /
  curve.track_centerline, optionally with curve.endpoints naming the plug end.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [cable, deformable, grasp, select, ik, cpu]
  hub:
    stage: select
    embodiments: [any]
    version: 0.1.0
    tasks: [cable-routing]
    tuned_on: [yam-station-overhead-rgbd]
    provenance:
      authors: [runtime-si]
      code: https://github.com/HAORAN808/dual-arm-cable-manipulation/blob/11b4201032074792fa04979eec918b3fbc536c27/perception/deliverable_perception_min/grasp_detection/core/safe_segments.py
      upstream_license: MIT
gap:
  requires: {connector: [robot.grasp_frame]}
  allowed_tools:
    - robot.grasp_frame
    - robot.solve_ik
  required_inputs:
    centerline: Centerline
  produces_outputs:
    pose: Se3Pose
    candidate_count: int
    checked: int
    detail: string
  exit_conditions:
    found: At least one safe window solved IK; `pose` is the best grasp, `poses` all of them (at most max_poses, at least min_spacing_m apart along the cable), `grasps` their metrics.
    none_safe: No fingertip-wide stretch of the cable passed the geometry (ends, hidden, crossings, bends, clearance), or the centreline was missing or unordered; `rejected` gives the share of the cable each rule ruled out.
    none_reachable: Safe stretches exist but robot.solve_ik reached none of the best max_checks with either wrist solution; re-perceive after moving the cable or use the other arm.
  canonical_scripts:
    - select_cable_grasps: scripts/select_cable_grasps.py
  examples:
    - title: Rank safe grasps along a fitted cable
      path: examples/canonical_subgraph.json
---

# selecting-safe-cable-grasp-segments

Where along a cable may the hand take it? Not at the end (the plug's housing,
and a short end flops out), not where the cable is hidden or crossed (the
fingers land on the other cable, or on what hides it), not on a tight bend
(the tangent the jaws close across is ill-defined and the cable springs), not
where the open fingers would hit a board, a ring post or another stretch of the
same cable, and not where the arm cannot go. This skill turns those rules into
a ranked list of top-down grasp poses.

## The pipeline

1. **Resample** the centreline every 5 mm of arc length (world frame, metres).
2. **End margins** (upstream `apply_endpoint_margin_by_arc_length`):
   `end_margin_m` at a bare end, `tip_margin_m` at the end `endpoints["tip"]`
   names (the plug). A cable shorter than both margins has no candidates.
3. **Hidden stretches** split the cable (upstream
   `split_centerline_by_occlusion`): a tracked node's `visibility` under 0.5,
   `occluded_indices`, or a stretch reading more than `max_rise_m` over the
   cable's support level (its 10th-percentile height). `end_margin_m` is kept
   free either side of each (upstream `apply_inner_margin`).
4. **Crossings** (`crossing_indices`) forbid `end_margin_m` either side
   (upstream `build_crossing_forbidden_intervals`, same radius as its end
   margin).
5. **Bends**: the turning angle of the xy chords over ±20 mm, per metre; above
   `max_curvature` is forbidden. *New.*
6. **Clearance**: the open fingers' footprint, 20 mm along the cable by
   `open_width_m` across it, must keep `min_clearance_m` from every obstacle
   box whose top stands above the cable there (a box the cable lies on is
   ignored), from `other_cables`, and from stretches of the same cable that
   fold back toward the grasp (more than half the opening plus a fingertip
   away along the cable, and at least 20 mm nearer in a straight line than
   along it, so the cable's own continuation does not count). *New.*
7. **Sliding window** (upstream `generate_candidate_grasp_segments`): a
   centre is a candidate when the whole 20 mm fingertip window is legal.
8. **Rank** by the mean of four terms in [0, 1]: distance past the end margin
   (saturating at 10 cm), straightness (1 - curvature / max), clearance past
   the minimum (saturating at 5 cm), and the length of the visible piece it
   is on relative to the longest (upstream keeps only the longest piece).
9. **Reach**: walking the ranking, skipping centres nearer than
   `min_spacing_m` along the cable to one already taken, ask
   `robot.grasp_frame(close_heading_deg = tangent + 90)` for a top-down
   rotation with the jaws closing across the cable, and `robot.solve_ik` at
   the point, trying `rotation` then `rotation_flipped`. A raise, `success:
   false`, `within_servo_tolerance: false` or a position error over 20 mm is
   a miss. Stops at `max_poses` found or `max_checks` tried.

## Inputs

- `centerline` (required): a `Centerline` (or its `points`, or that as JSON
  text, as perceiving-deformable-linear-objects binds `rod`).
- `endpoints`: curve.endpoints' answer for the same centreline. Only `tip` is
  read: `ends[0]` is the `points[0]` end.
- `obstacles`: `OrientedBoundingBox`es (board, ring posts, fixtures).
- `other_cables`: other centrelines, same forms as `centerline`.
- `crossing_indices`, `occluded_indices`: node indices into `centerline`.
- `arm_id` (passed to the robot tools only when nonzero), `check_reach`
  (true), `max_poses` (5), `min_spacing_m` (0.03), `max_checks` (40), and the
  constants below.

## Outputs

`pose`, `poses` (best first), `grasps` (each: `pose`, `point` on the cable
axis, xy `tangent`, `close_heading_deg`, `rotation_used`, `arc_m`, `node`,
`end_distance_m`, `tip_distance_m`, `curvature` [1/m, worst in the window],
`clearance_m` (`null` = nothing to clear), `segment_m`, `score`),
`candidate_count` (safe windows), `checked` (IK candidates tried), `rejected`
(share of the cable each rule ruled out), `detail`.

The pose's height is the fitted cable axis plus `grasp_height_m`. On the
station's overhead frame the fitted z scatters about ±5 mm around the true
table-level cable, so the executor must floor it at its support (a real YAM
station also refuses a tool centre under 14 mm over its table): this skill
selects where and at what yaw, not how deep.

## The constants, and why

- **`STEP_M = 0.005`, window 2 x `FINGER_HALF_M = 0.020` m**: upstream slides
  a 16 px window by 4 px; the window is the YAM station's fingertip width
  (its estimate), the step a quarter of it.
- **`END_MARGIN_M = 0.05`**: upstream uses one value (8 px) for the ends, the
  margin beside a hidden stretch and the crossing radius, and so does this.
  50 mm: the fitted end lies 7-9 mm off the true one on the station's frame
  (curve.fit_centerline at smooth 1e-5, issue #10), plus a 20 mm fingertip and
  enough cable that it does not slip out.
- **`TIP_MARGIN_M = 0.08`**: the station cable's plug housing is about 33 mm
  on the overhead frame, plus the end's error and a fingertip.
- **`MAX_RISE_M = 0.03`**: on the station's frame the cable runs under the
  parked left gripper and the fit read 19 cm there; a cable that is really
  lifted that much is not a top-down pinch either.
- **`MAX_CURVATURE = 25` 1/m** (bend radius 40 mm), estimate; read over
  ±20 mm chords so depth jitter is not a bend (straight stretches of the
  station's cable read 0.3-6 1/m).
- **`OPEN_WIDTH_M = 0.047`**: the station opens its 94 mm linear_4310
  half-way for a cable. **`MIN_CLEARANCE_M = 0.008`**: its fingertip
  clearance from the board's edge when taking a cable.
- **`GRASP_HEIGHT_M = 0.0062`**: closed fingertips 10.2 mm past the tool
  centre, pinching 4 mm under the cable's centre (station values).

## Checked on a real frame (no arm motion)

The station's overhead frame `cable_frame_20261103-1421` (640x480 RGB-D):
curve.fit_centerline (smooth 1e-5) on a mask drawn from the station's own
cable trace, curve.endpoints with gemini-er.point's tip, the breadboard as an
OBB from the largest dark blob back-projected through depth (304 x 212 mm,
up to the ring posts), and `robot.solve_ik` stubbed by the station's
workspace box (x -0.15..0.35, y 0.10..0.55 m, left base frame).

- `found`: 5 grasps from 92 safe windows, 32 candidates and 59 IK calls
  (the ones right of x = 0.35 m failed the reach stub). Ruled out by share of
  the fitted 1.36 m: ends 10%, hidden 35% (the stretch under the parked
  gripper, read up to 19 cm high), beside hidden 8%, bends 6%, clearance 29%
  (near the board's two front corners).
- Four of the five sit on the cable in the image, on the run in front of the
  board: curvature 0.6-10.8 1/m (the 10.8 one at the start of a gentle
  turn), the open fingers' footprint 25-47 mm clear of the board, 282-382 mm
  (xy) from the nearer cable end, 284-335 mm from the visible plug. None is
  on the plug end, a tight bend, the board or its rings.
- The fifth (4th ranked, beside the gripper) lies on bare table about 20 px
  (~35 mm) from the real cable: the station's trace, hence the mask and the
  fit, joins its pieces wrongly there. It sits 0.2 px from the trace, so this
  is the input's error, which this camera-free skill cannot see; a
  re-perception or a verify step after the grasp catches it.
- Not measured: a grasp on the arm; the fit's arc length (1.36 m against the
  trace's 0.936, issue #10's depth jitter) stretches every arc-length margin
  by up to 45% on this frame; the trace, hence the fit, ends 4-5 cm short of
  the plug on the board's back edge, and gemini-er named the other end the
  tip, so the plug end got the bare 50 mm margin (it lies outside the reach
  box regardless).

## Provenance

Adapted from dual-arm-cable-manipulation (MIT, Copyright (c) 2026 Haoran Han,
Yuxin Fu, Boxian Yao; license in `LICENSE.dual-arm-cable`):
`grasp_detection/core/safe_segments.py` (end margin by arc length, split at
occluded points, drop short pieces, prefer the longest) and
`core/dual_grasp_selector.py` (inner margin, crossing forbidden intervals,
subtraction, sliding grasp windows). Changed: world-frame metres instead of
pixels; occlusion from tracker visibility, given indices or a raised stretch
instead of upstream's RGB/mask crossing reasoning (crossings are an input
here); a plug-end margin; bend and clearance gates and a score are new; every
window is ranked and IK-checked as a top-down grasp instead of picking the two
farthest-apart windows for two arms (run it once per arm for a two-hand hold).
