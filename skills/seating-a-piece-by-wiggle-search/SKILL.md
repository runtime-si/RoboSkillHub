---
name: seating-a-piece-by-wiggle-search
description: Seats a held jigsaw piece in its slot among already-placed
  neighbours when the arm is less accurate than the slot's clearance - lowers
  the piece over the slot until it stops (tracking lag, or joint effort where
  published), reads from the stop height whether it dropped onto the mat or
  rests on the neighbours (a piece thickness higher), and if it rests, lifts
  and re-lowers it over small x / y / yaw offsets taken most likely first,
  finishing with a scoot-and-wiggle at the best spot (Mark Rober's puzzle
  robot's routine, reimplemented without its spring and encoder). Releases a
  dropped piece and reads its top with the wrist depth camera to confirm it is
  flush. Options - tilted tab-first, or set down short and pushed in. Use when a jigsaw piece (or any thin flat part with a matching
  cut-out) is held over its slot with the slot pose known to a millimetre or
  two and a few degrees, the neighbours fixed around it, and fingers clear of
  the neighbours.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [jigsaw, puzzle, insertion, search, contact, wiggle, wrist-camera]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw, insertion]
    provenance:
      authors: [runtime-si]
      code: https://github.com/markroberyoutube/puzzle_bot/blob/main/src/application.py
      upstream_license: none (reference only; reimplemented, no code copied)
      paper: https://arxiv.org/abs/2002.02580
gap:
  requires: {connector: [robot.describe_gripper]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.get_gripper
    - robot.describe_gripper
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - robot.solve_ik
    - robot.open_gripper
    - robot.close_gripper
  required_inputs:
    slot_position: Vec3
  produces_outputs:
    released: bool
    probes: int
    moves: int
    elapsed_s: float
    stop_height: float
    best_height: float
    resting_height: float
    offset_found: Vec3
    yaw_found_deg: float
    seated_position: Vec3
    flush_error: float
    flush_spread: float
    samples: int
    verified_by: str
  exit_conditions:
    seated: A probe stopped with the piece's bottom within a quarter of its thickness of the mat (it dropped), the jaws opened, and the wrist depth reads the piece's top within a quarter of the thickness of flush with a spread under the same.
    not_seated: No probe or scoot dropped the piece (kept and lifted back, released false); or, released, the wrist depth reads its top high or tilted (resting on a neighbour).
    jammed: The piece went partly in (best stop between a quarter and 0.6 of its thickness over the mat) and the scoot did not finish it (kept, lifted back); or the hand could not lift the piece back out.
    piece_lost: The jaws read at their stop before anything moved, a descent went a whole thickness under the mat without meeting anything, or after the release the wrist depth finds no piece top at the slot.
    unverified: Released after a drop, but the flush check could not run (verify false, no wrist camera, no view reached).
    unreachable: The seated hand pose fails the IK preview, the hover or a turn was not reached, or a move was refused.
  canonical_scripts:
    - seat_piece: scripts/seat_piece.py
  examples:
    - title: Seat a held piece at a slot a perception skill located
      path: examples/canonical_subgraph.json
---

# seating-a-piece-by-wiggle-search

A jigsaw piece goes into its slot among placed neighbours only within about
0.1-0.4 mm and a degree (Mark Rober's write-up measured 0.13-0.38 mm); a YAM
puts its hand within a few millimetres. A rigid piece lowered off by more than
the clearance lands on the neighbours, one thickness (about 2 mm) up; one
lowered on the slot drops to the mat. That height step is the only signal, so
this skill tries positions until it sees it, the way Mark Rober's puzzle robot
does with a spring-mounted head and a linear encoder.

## The motion

1. **Check and approach.** Read the jaws (at their stop: `piece_lost`). Preview
   the seated hand pose with `robot.solve_ik`. Rise if low, turn the piece by
   `yaw_error_deg`, go to the hover (`approach_height` over the neighbours).
2. **First descent.** Fast (`robot.go_to_pose_cartesian`) to `contact_window`
   over the neighbours' tops, then down in 0.4 mm steps. After each step the
   hand is allowed to settle and its lag behind the command is compared with
   free motion (and joint efforts with their free-motion windows where the
   observation carries them); it stops at the first contact (`press_depth` 0;
   a light press can be asked for, but on the station twin a 0.6 mm press
   pushed the piece 0.6-0.8 mm up in the jaws and every later height read
   that much high). Positions use the wrist yaw read back after each move, and
   a yaw that crept off on the way (the twin leaves the roll free) is turned
   back before the descent: with the piece 9 mm from the tool centre, 3 deg of
   creep put it 0.5 mm off.
   The stop gives the piece's bottom over the mat: under a quarter of the
   thickness is a drop; a whole descent through a thickness under the mat
   without contact means the piece is not in the hand.
3. **Probe search** (puzzle_bot's pick-up-and-wiggle). Lift the piece
   `LIFT_CLEARANCE` over the highest stop seen, move it to the next
   (dx, dy, dyaw), and feel down again. Offsets come from a 0.4 mm hexagonal
   grid inside `search_radius` crossed with 1 deg yaw steps inside
   `yaw_range_deg`, ordered by (r / `sigma_xy`)^2 + (yaw / `sigma_yaw_deg`)^2:
   the most likely first. Each probe's hand is re-commanded until it reads back
   within 0.15 mm of its point. The best (lowest) stop is remembered. The
   search ends on a drop, a stop under 0.6 of the thickness (partly in), or
   `max_probes` / `max_time_s`.
4. **Scoot and wiggle** (puzzle_bot's second phase). Back at the best probe,
   with the piece resting where it stopped: out and back along x, along y,
   then a yaw out and back, felt after each, widening over three rounds.
5. **Release and check.** Stop pressing (command the hand where it stands),
   open, retreat straight up, and view the piece's centre straight down with
   the wrist camera at 15-25 cm. The median height of 19 points within
   `top_radius` of the piece's centre against the mat plus a thickness decides
   `seated` / `not_seated` (high or tilted) / `piece_lost` (nothing there).
   A piece that never dropped is lifted back still held (`not_seated` /
   `jammed`) unless `release_when_not_seated`.

Yaw turns use the last joint (`robot.move_to_joints`), read back, the gain
learnt on the first turn: a YAM station twin's IK leaves wrist roll free, so a
rotation command can report reached without turning (PORTING lesson 10). A
connector without joint readings gets a rotation command instead. The
piece's offset from the tool centre turns with the hand, so every probe puts
the piece, not the hand, over its point.

## Inputs

- `slot_position` (required): where the piece's reference point sits when
  seated, z the mat surface under it (the piece's bottom), world frame. From
  localizing-a-piece-in-a-reference-image or matching-a-piece-to-placed-neighbours
  via the board's pose; refine z from the wrist depth of the neighbours' tops
  minus `thickness`.
- `held_offset`: the piece's reference point minus the tool centre, in the
  world frame at the hand's current rotation; z is the piece's bottom minus the
  tool centre's z. Default (0, 0, -10.2 mm): centred, bottom level with closed
  linear_4310 fingertips.
- `yaw_error_deg` (0): how far the held piece must turn about the vertical to
  match the slot, from perception.
- `thickness` (2 mm), `sigma_xy` (1 mm), `sigma_yaw_deg` (2 deg),
  `search_radius` (2.5 mm), `pitch` (0.4 mm), `yaw_range_deg` (4),
  `yaw_step_deg` (1), `max_probes` (40), `max_time_s` (0: none), `scoot`
  (true), `press_depth` (0), `lag_threshold` (half a step), `approach_height`
  (15 mm), `contact_window` (4 mm), `release_when_not_seated` (false), `hold_s`,
  `verify` (true), `camera`, `top_radius` (5 mm), `roll_limits`, `check_reach`,
  `reach_tolerance`, `arm_id` (passed only when nonzero).
- `push_down` (false), `push_down_offset`: after the release, close the jaws
  and push the piece's top down with the fingertips (until the command leads
  the hand by 0.5 mm) at its centre plus the offset, puzzle_bot's last step.
  Off by default: in the twin check it knocked a just-dropped piece back onto
  its neighbours in the one trial it ran.
- `approach`: `wiggle` (default), `tilt_first` or `offset_push` (below), with
  `seam_direction`, `tilt_deg` (10), `tilt_back` (3 mm), `piece_radius`
  (10 mm), `push_offset` (4 mm), `finger_reach` (10.2 mm).

## The constants, and why

- **`SEATED_FRACTION = 0.25`, `NEARLY_FRACTION = 0.6`** of the thickness:
  puzzle_bot calls a piece fully set within 10% of its thickness and nearly set
  within 25% (first 60%) on its encoder. An arm's stop is coarser than an
  encoder; a quarter (0.5 mm) is still a quarter of the 2 mm a resting piece
  stands up.
- **`FEEL_STEP = 0.4 mm`, `LAG_FRACTION = 0.5`, `PRESS_DEPTH = 0`**: a
  fifth of the thickness per step; pressing-a-brick-onto-studs' lag fraction,
  tuned on the YAM station twin (servo within 0.2-0.3 mm); no press past the
  contact (measured: a 0.6 mm press slid the piece up in the jaws).
- **`PITCH = 0.4 mm`, `SEARCH_RADIUS = 2.5 mm`**: every point within 0.23 mm
  of a probe, about the clearance; puzzle_bot's grid steps 0.25 mm over
  +-2.5 mm.
- **`YAW_STEP_DEG = 1`, `YAW_RANGE_DEG = 4`**: a 20 mm piece's tab tips sit
  about 14 mm from its centre; 1 deg moves them 0.24 mm.
- **`SIGMA_XY = 1 mm`, `SIGMA_YAW_DEG = 2`**: the order of the search, not a
  cut-off. A wrist close-up places a target to about a millimetre; picture
  localisation searches angles in steps of a few degrees.
- **`MAX_PROBES = 40`**: puzzle_bot's spiral is 15 rings of 8; 40 probes cover
  about 1 mm at the given yaw plus the nearest yaw steps.
- **`SCOOT = 0.3 mm` (+0.2 per round, 3 rounds), `SCOOT_YAW_DEG = 0.75`**:
  puzzle_bot's scoot starts at 50 counts and widens by 25.

## The other two approaches (`approach`)

Both need `seam_direction`: the horizontal direction from the piece toward the
placed neighbours it goes against (world frame).

- **`tilt_first`** (tilted tab-first, re-expressed from Watson, Miller and
  Correll's InsertTilt, arXiv 2002.02580, paper only). Tilt the piece
  `tilt_deg` (10) leading side low, lower it `tilt_back` (3 mm) short of the
  slot until its edge touches (no press), slide it along `seam_direction` at
  that height until the hand lags (it met the neighbours) or `tilt_back +
  search_radius`, level it about its reference point and feel down. That stop
  is the first probe; the probe search and the scoot continue around where the
  slide ended, so an error along the seam is taken out by the slide and one
  across it by the search. Needs a grip that holds the tilt (a fingertip grip
  can let the piece turn back level in the jaws, as pressing-a-brick-onto-studs'
  twin check saw) and fingers clear of the leading seam.
- **`offset_push`** (placing offset then pushing into the seat). Set the piece
  down flat `push_offset` (4 mm) short of the slot where the mat is clear (a
  stop over a quarter thickness up: `not_seated`, kept), let go, retreat, close
  the jaws, lower the closed fingertips 1 mm over the mat 3 mm behind the
  piece's trailing edge (`piece_radius` from its reference), push along
  `seam_direction` in fine steps to 0.5 mm past the slot, stop on lag (a stop
  more than 1 mm short is `jammed`), then press the piece's centre with the
  fingertips (`finger_reach` under the tool centre) and read its top with the
  wrist depth. The only one of the three that works when no grip clears the
  neighbours, and only where the slot is open toward the push: a rigid tab
  cannot slide sideways into a blank, so it suits a corner or edge slot, not a
  slot with neighbours on opposite sides. Detecting a pushed neighbour needs
  detecting-disturbed-pieces-by-snapshot-diff afterwards.

The default (`wiggle`) assumes the fingers are clear of the neighbours (the
piece is pinched across sides that face empty cells, see
selecting-a-grip-clear-of-placed-neighbours).

## Simulation check

Pending: see the PR.

## Method

Mark Rober's puzzle robot (github.com/markroberyoutube/puzzle_bot,
src/application.py) has no licence, so nothing was copied; its routine was
re-expressed from what the code does: a table-height map plus the piece
thickness gives fully / nearly placed thresholds on a spring-mounted head's
linear encoder; a not-nearly-placed piece is picked up and set down over a
+-0.1 in grid at 0.01 in, or an 8-point ring spiral widening 25 counts per
ring for 15 rings, keeping the best height; then it scoots at a slight-press
height in x then y until fully set. Here the encoder is the arm's stop height
read through tracking lag, the spring is a press bounded by the command's
lead, the grid / spiral is one probe order over x, y and yaw by likelihood,
and the flush check after the release is the wrist depth camera's.
