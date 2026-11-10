---
name: correcting-brick-offset-from-stud-circles
description: Closes the last millimetre over a LEGO or DUPLO brick before a
  pick or a press. The hand peeks at a fixed height over the expected stud
  pair; the wrist RGB-D close-up gives the brick top's plane and a circle at
  the known stud radius for each stud, fitted only to the rim that is seen
  (occlusion-aware, as Eye-in-Finger forces its circle tangent to the visible
  mask edges); the two studs nearest the expected pair (with every other stud
  on their lattice) give the pair's midpoint and line, compared with the
  tool axis (or a calibrated tool-centre pixel); the hand steps by the
  difference in x, y and yaw and repeats until under 0.2 mm and 1 deg. Use
  when a brick's pose is known to a few millimetres (an overhead estimate,
  read_studs, computing-brick-pose-from-stud-grid) and a grasp or a press-fit
  needs it to a few tenths, on an arm whose moves land a millimetre off.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [lego, duplo, assembly, visual-servoing, wrist-camera, d405, studs, closed-loop, cpu]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      paper: arXiv:2503.06848
      code: https://github.com/kevinpoi/legoscope/blob/2abdc205ef36/detect_offset.py
      upstream_license: none (no LICENSE file) - reference only, reimplemented from the paper; no code or weights used
gap:
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose
  required_inputs:
    pair_pose: Se3Pose
  produces_outputs:
    pair_pose: Se3Pose
    tool_pose: Se3Pose
    yaw_offset_deg: float
    iterations: int
    moves: int
    pair_range_m: float
    tool_axis_source: string
    camera_name: string
  exit_conditions:
    aligned: The tool axis (or the tool_center_px ray) was measured within tolerance_mm (0.2) of the stud pair's midpoint and within tolerance_deg (1.0) of the wanted yaw (and, with tool_center_range, the camera's range within tolerance_mm of it); the hand is at the peek height over the pair and pair_pose is the last measurement.
    ambiguous_stud_pair: The nearest stud pair is over 0.4 pitch from the expected one, or two pairs are almost as near - the expected pose is about half a stud off, so which pair is meant cannot be told; nothing more is moved.
    lost_studs: No posed wrist RGB-D frame, no brick-top plane within half a brick of the expected pair, or under two stud circles / no pair one pitch apart along the expected line.
    max_iterations: max_iterations measurements were taken and the last was still over tolerance (with max_iterations 1 the skill only measures).
    unreachable: The rise, the peek pose or a correction move was refused (routed or raised); detail says which.
  canonical_scripts:
    - correct_offset: scripts/correct_offset.py
  examples:
    - title: Centre the hand over a brick's stud pair before pressing or picking it
      path: examples/canonical_subgraph.json
---

# correcting-brick-offset-from-stud-circles

A LEGO stud is 4.8 mm across on an 8 mm pitch; a press-fit or a two-finger
grip between studs wants the hand within a few tenths of a millimetre, while
an overhead camera and an arm that lands a millimetre off give a few
millimetres. Eye-in-Finger's answer is to stop looking at the world and look
at the brick from the hand: peek at a fixed height, find the two studs under
the tool, and step until their midpoint is on the tool axis. The error that
remains is the detector's, not the arm's.

## How it composes

- **Before it**: an estimate of the brick, as `pair_pose`: the midpoint of
  the two studs the tool should end over, on the brick's top surface (the
  stud bases), x along the line through the two studs, z up. From
  calibrating-baseplate-stud-grid's `read_studs` (`top_pose` plus half a
  pitch along the right axis), computing-brick-pose-from-stud-grid (a
  placed brick's cell), or an overhead detection. It must be within 0.4 pitch
  (3.2 mm LEGO, 6.4 mm DUPLO) of the true pair, counting the arm's reported
  pose error; else the exit is `ambiguous_stud_pair`.
- **After it**: the hand is at `peek_height` (30 mm) over the stud tops,
  axis on the pair: descend for a pick (pressing a brick's neighbour, picking
  it), or use `pair_pose` for pressing-a-brick-onto-studs' seat.
- **The yaw** the hand ends at, relative to the pair line, is
  `yaw_in_pair_deg`; by default the multiple of 90 deg nearest to where the
  hand is (jaws along or across the pair).

## What it does

1. **Peek** (`move_to_peek`, default true): the tool to `peek_height` over the
   expected pair's stud tops at the wanted yaw, rising in place first when it
   is lower.
2. **Measure** (each round, after the hand settles): the wrist D405's depth
   within 2 pitches of the expected pair and half a brick of its height; a
   RANSAC plane (0.6 mm band; when the stud tops win, the plane a stud below
   them); a height map on it at 1/16 pitch. Cells over half a stud height are
   stud tops. Each stud-sized blob's **rim** is where a top cell meets a seen
   non-top cell; where it meets an unseen cell (the far side of a stud from
   an oblique camera, a finger, a dropout) the rim says nothing. A circle at
   the known stud radius is fitted to the seen rim (Gauss-Newton, Huber); a
   stud is kept with 8 rim cells over 100 deg of arc.
3. **The pair**: pairs of circles one pitch apart (+-15 %) within 30 deg of the
   expected line; the one whose midpoint is nearest the expected one, with
   the half-pitch rule (nearest over 0.4 pitch, or the second within 0.2
   pitch of it, is ambiguous). Every circle on that pair's lattice (within
   0.2 pitch of a node) then refines the midpoint and the yaw by a rigid 2D
   fit: on a 2 x 4 top, 8 studs instead of 2.
4. **Compare**: the tool axis is the reported tool centre dropped vertically
   onto the stud-top plane, or with `tool_center_px` the ray through that
   pixel. The offset is pair midpoint minus axis point, in the pair's level
   frame (`offset_mm`, `[along, across]`); the yaw offset is folded to
   +-90 deg (the pair is symmetric).
5. **Step**: under tolerance, `aligned`. Otherwise the hand turns about its
   axis by the yaw offset (at most 10 deg) and moves by the lateral offset
   (times `gain`, at most half a pitch) to the peek height over the measured
   pair; the measured pair becomes the next round's expectation, so the
   studs are tracked, not re-identified from scratch.

## Inputs

- `pair_pose` (Se3Pose, required): see above. Its x must not be vertical.
- `system`: `lego` (default) or `duplo`; `stud_pitch`, `stud_height`,
  `stud_diameter` override the preset [m].
- `peek_height` (0.03 m): tool centre over the stud tops.
- `yaw_in_pair_deg`: the hand's x relative to the pair line; default the
  nearest multiple of 90 deg.
- `tool_center_px` `[u, v]` and `tool_center_range` [m]: the image point the
  tool axis passes through at the peek height (Eye-in-Finger's
  `TOOL_CENTER`) and the camera's range to the stud tops along that pixel's
  ray. Measure them once: put the hand truly over a pair at the peek height
  (seated by hand, or after a verified press) and run the skill with
  `max_iterations: 1, move_to_peek: false`; its `pair_px` and `pair_range_m`
  are the two. The pixel replaces the hand-eye transform in the comparison;
  the range holds the camera at the calibrated height by its own depth (the
  ray is ~40 deg off the tool axis, so a height error is nearly the same
  lateral error, and a hand-eye tilt of 0.2 deg puts the plane ~0.5 mm off
  in height). Without the range the hand's height comes from the plane in
  the reported frame.
- `tolerance_mm` (0.2), `tolerance_deg` (1.0), `max_iterations` (6), `gain` (1.0).
- `camera` (wrist camera name; default the wrist camera nearest the tool),
  `arm_id`, and the connector knobs `tolerance`, `max_steps` (passed to
  `robot.go_to_pose` only when set), `seed`.

## Outputs

`pair_pose` (the last measured pair, world; the input when nothing was
measured), `tool_pose` (read back at the end), `offset_mm` and
`yaw_offset_deg` (the last measurement, before any step), `iterations`
(measurements), `moves` (including the peek and rise), `history` (per round:
offset, yaw offset, how far the pair was from its expectation in pitches,
studs found), `stud_centres` (the pair's two stud-top centres, world),
`pair_px` (the pair midpoint's pixel in the last frame), `pair_range_m` (the
camera's distance to it), `rim_arc_deg` (the
seen rim arc of each), `tool_axis_source` (`ee_pose` or `tool_center_px`),
`camera_name`, `system`, `stud_pitch`.

## Checks

CPU only. Unit tests (`tests/bundles/test_correcting_brick_offset_from_stud_circles.py`)
on ray-cast RGB-D renders (`tests/bundles/_plate_render.py`, D405 noise,
flying pixels, dropouts) with a fake YAM arm (`tests/bundles/_brick_offset_sim.py`:
the wrist D405 86 mm behind and 75 mm above the tool, 45 deg forward; the
frame rendered from where the hand really is): the measurement at three
yaws, a stud half hidden by a hanging block, DUPLO, the half-pitch
ambiguity; the loop with an ideal arm, with landing and reporting errors,
with a hand-eye error with and without `tool_center_px`, a 90 deg
`yaw_in_pair`, the rise, measure-only, every exit, routed and raised
refusals, the half-pitch step clip and the connector knobs.

**Closed-loop accuracy against a one-shot read** (`python -m
tests.bundles._brick_offset_eval --trials 20`, seed 0): a 2 x 4
brick on a plate at a random yaw, the hand top-down 30 mm over it but up to
3 mm and 4 deg off, `pair_pose` up to 1.5 mm per axis and 3 deg off. Three
fake arms: *ideal* (moves land where commanded), *YAM-like* (each move lands
10 % long plus 0.1 mm / 0.1 deg noise, and the arm reports its pose 1.1 mm
off - a bias common to the tool and the camera) and *hand-eye* (YAM-like, and
the wrist camera really 0.6 mm / 0.3 deg off its nominal mount). The
baseline is the hub's current best: calibrating-baseplate-stud-grid's
`read_studs` on the peek frame (with the brick's mask), the pair taken from
its `top_pose`, one move. True lateral error of the tool axis from the pair
midpoint after the method, median / p90 / max (mm), and moves:

| arm | system | one-shot `read_studs` | one-shot, this detector | closed loop | closed loop + `tool_center_px` |
|---|---|---|---|---|---|
| ideal | LEGO, 20 | 0.10 / 0.12 / 0.14 | 0.04 / 0.07 / 0.11 | 0.04 / 0.07 / 0.11 (1.0 moves) | 0.05 / 0.09 / 0.14 (1.0) |
| YAM-like | LEGO, 20 | 0.27 / 0.46 / 0.63 | 0.27 / 0.36 / 0.66 | 0.12 / 0.20 / 0.23 (1.9) | 0.11 / 0.18 / 0.30 (1.8) |
| hand-eye | LEGO, 20 | 0.89 / 1.12 / 1.28 | 0.80 / 1.00 / 1.19 | 0.71 / 0.80 / 0.86 (2.0) | **0.12 / 0.19 / 0.19** (2.1) |
| YAM-like | DUPLO, 10 | 0.47 / 0.67 / 0.69 | 0.40 / 0.69 / 0.71 | 0.21 / 0.26 / 0.27 (2.4) | 0.12 / 0.23 / 0.24 (2.3) |
| hand-eye | DUPLO, 10 | 0.81 / 1.14 / 1.29 | 0.82 / 1.13 / 1.19 | 0.66 / 0.71 / 0.73 (2.4) | **0.13 / 0.19 / 0.26** (2.1) |

Every closed-loop run ended `aligned`; yaw ended within 0.15-0.20 deg
(median), 0.76 deg at worst. What the loop buys is the arm's landing error
(YAM-like rows: 0.27 to 0.12 mm); what `tool_center_px` (with its range)
buys is the hand-eye error, which no world-frame method sees (hand-eye rows:
0.7-0.9 to 0.12 mm). On the ideal arm the detector alone is 2.5x tighter than
`read_studs`'s top pose (0.04 against 0.10 mm), as it fits the stud rims
rather than a lattice through all studs; its single-frame scatter is 0.025 mm
for the midpoint and 0.18 deg for the yaw (10 views, sd), with a 0.05 mm bias
away from the camera. DUPLO's ee_pose loop stops at 0.21 mm: its measured
pair has a ~0.2 mm bias the loop cannot see, which the pixel calibration
absorbs. 1446 s on 12 CPU threads, no GPU.

## Limits

- **Not run on a real arm or real LEGO.** No frame on disk shows LEGO or
  DUPLO, and the YAM station twin has none (calibrating-baseplate-stud-grid
  added plates to it; no brick press was in reach of this port's time). The
  numbers are renders: ideal cylinders, no logo on the studs, no specular
  ABS (a real D405 on glossy ABS loses depth on highlights), no fingers
  (one test hangs a block in the view instead).
- **Depth-only studs.** The paper segments studs in RGB (a YOLO model whose
  weights are AGPL and unlicensed - not used). A LEGO stud is 1.7 mm high;
  at the 0.14 m peek range a D405's depth noise is ~0.2 mm, so the stud
  stands ~8 sigma out on the renders. Real frames may need the RGB circles
  calibrating-baseplate-stud-grid also uses.
- **The hand-eye error stays without `tool_center_px`.** With the axis from
  the reported tool pose, the loop converges to wherever the camera mount
  error puts the pair (see the eval's hand-eye rows); the pixel and its range
  take it out. An error common to the tool and the camera (a kinematic bias)
  cancels either way. The pixel is only as good as its calibration pose.
- **The prior must be inside 0.4 pitch.** Past it the skill refuses
  (`ambiguous_stud_pair`) rather than lock onto the neighbouring pair, as the
  paper's loop would.
- **A level top.** The plane may tilt (a brick on a plate); the tool axis is
  dropped vertically and the hand is not tilted to match.
- The tool should be roughly top-down; `peek_height` must keep the studs
  inside the wrist camera's view and minimum range (the D405's ~7 cm).

## The constants, and why

- **Presets**: the hub's LEGO / DUPLO values (8 / 16 mm pitch, 1.7 / 4.5 mm
  stud height, 4.8 / 9.4 mm diameter).
- **`PEEK_HEIGHT = 0.03`**: the YAM D405 sees the studs under the tool ~6 deg
  off axis at ~0.14 m; Eye-in-Finger peeks at 17-30 mm.
- **`TOLERANCE = 0.2 mm`, `TOLERANCE_DEG = 1`**: one render frame's midpoint
  scatters 0.025 mm, its yaw 0.18 deg (sd, 10 views, 8 studs); the paper's
  loop ends at ~0.3-0.4 mm.
- **`CELL_PITCHES = 1/16`, `TOP_FRACTION = 0.5`**: a cell cut by the rim
  averages the two heights in proportion to its area, so half a stud height
  is the rim; on 10 renders 0.6 left a 0.2 mm bias away from the camera and
  1/12 or 1/10 cells 0.07-0.13 mm.
- **`MAX_PRIOR_ERROR = 0.4`, `MIN_PAIR_MARGIN = 0.2`** (pitches): the
  paper's within-half-a-stud precondition, with a margin for noise.
- **`MAX_STEP_PITCHES = 0.5`, `MAX_TURN_DEG = 10`**: a larger step is not a
  last-millimetre correction and would carry the hand past the next pair.
- **`MIN_RIM_POINTS = 8`, `MIN_RIM_ARC_DEG = 100`**: a known-radius circle is
  fixed by a quarter of its rim; the camera always sees at least its near
  half.
- **`PLANE_BAND = 0.6 mm`**: ~2.5 sigma of a D405 at 0.14 m, under a stud's
  1.7 mm.

## Provenance

Eye-in-Finger (arXiv:2503.06848, Sec. III-B, Algorithm 1) and its code at
github.com/kevinpoi/legoscope (`detect_offset.py`), which has no licence:
read for the algorithm only, nothing copied, its YOLO stud weights not used.
Kept: the fixed peek height, the two studs nearest the tool, a stud circle
of known size fitted only to the visible edges, the midpoint and the line
through the two compared with the tool centre, a calibrated tool-centre pixel,
the iterate-until-under-tolerance loop and its half-a-stud precondition. New:
studs from depth on the fitted top plane (no learned model), metric offsets,
the tracked prior and the ambiguity exit, the lattice refinement over all
studs of the top, the 180 deg yaw fold, the step limits. The tilt check from
the paper's ring-light reflection is not ported (the D405 has no ring light;
the depth plane gives the tilt).
