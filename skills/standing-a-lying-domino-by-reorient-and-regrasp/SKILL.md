---
name: standing-a-lying-domino-by-reorient-and-regrasp
description: Stands a flat-lying domino upright so a top-down pick can take it -
  pinches the lying tile across its width near one end, lifts it, turns it
  about the vertical if the arm needs another facing, pitches the hand 90 deg
  about its closing axis so the tile's long axis stands vertical, reads the
  wrist roll back and finishes it on the last joint, sets the tile on its end
  at a staging spot (by default where it lay), opens a few millimetres, backs
  off and rises; then checks from depth that a tile stands there. Every stage
  is previewed with the IK solver before anything moves, and the facing is
  chosen from those that pass. Use when a domino lies flat (from
  reading-domino-pips-and-orientation or a 6D pose) and must stand before
  placing-an-upright-domino-beside-neighbours picks it into a row.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [domino, reorient, regrasp, upright, pitch, depth-check]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [domino, reorientation]
    provenance:
      authors: [runtime-si]
      code: https://github.com/Zhu-Chenyu/Franka-Domino-Project/blob/main/place_dominoes/place_dominoes/place_dominoes.py
      upstream_license: none (reference only; reimplemented from the behaviour, no code copied)
gap:
  requires: {connector: [robot.grasp_frame, robot.describe_gripper, robot.set_grip]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
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
  required_inputs:
    table_z: float
    tile_center: Vec3
    long_axis: Vec3
  produces_outputs:
    picked: bool
    released: bool
    grasp_height: float
    grip_width: float
    facing_deg: float
    yaw_turn_deg: float
    pitch_turn_deg: float
    roll_turn_deg: float
    tilt_at_release_deg: float
    release_position: Vec3
    standing_tile: Se3Pose
    verified: bool
    tile_top: float
    standing_offset: float
  exit_conditions:
    standing: Released at the staging spot and retreated, and the depth check sees a tile standing there (its top within 10 mm of the tile's height and within 8 mm of the spot); with verify=false, what the hand did only. `standing_tile` is the pose to hand to placing-an-upright-domino-beside-neighbours.
    fell: Released, but the depth check sees no tile standing at the spot - its top is low (lying, on its side, leaning hard), more than 8 mm off the spot, or nothing is there.
    unverified: Released and retreated, but no camera saw the spot whole and unoccluded.
    ungraspable: Nothing was lifted - the descent to the tile stopped high (a fingertip on it, or it is not where it was seen), the jaws closed on nothing or not across its width, or the grasp height leaves the fingertips over the tile.
    dropped: The tile was picked, then the jaws read empty or not across it - after the lift, after the pitch or over the staging spot.
    unreachable: No facing passed the IK preview for the grasp, the pitch and the release; a move was refused; the jaws or the held tile stayed turned off their target (no room on the last joint); the descent to the spot stopped high or the held tile leaned past tilt_tolerance_deg there (lifted back); or the retreat was refused after the release. `detail` says whether the tile is still held.
  canonical_scripts:
    - stand_domino: scripts/stand_domino.py
  examples:
    - title: Stand a lying domino where it lies
      path: examples/canonical_subgraph.json
---

# standing-a-lying-domino-by-reorient-and-regrasp

A domino lying flat cannot go straight into a row: a top-down pinch across a
lying tile puts the fingers where the neighbours stand. This skill stands the
tile on its end at a free staging spot first, so that
placing-an-upright-domino-beside-neighbours can pick it again from above with
the fingers on its broad faces (the regrasp is that skill's job; chain the two
on `standing`).

## The motion

1. Geometry first. The tile is a world-frame centre and long axis (from
   reading-domino-pips-and-orientation, a 6D pose or any detector); the axis
   must have a horizontal direction. The jaws close across the tile's
   **width** (24 mm), `grip_offset` (14 mm) along the long axis from its
   centre toward the end that will be on top, the tool centre `grasp_height`
   (11 mm) over the table. A grasp height that leaves the fingertips less than
   `MIN_BITE` (2 mm) of the 7.5 mm edge, or a tile wider than the open hand, is
   `ungraspable` before anything moves.
2. The plan, previewed with `robot.solve_ik` before anything moves. The
   *facing* is the heading of the pitched approach; once the tile stands its
   broad faces look along it (`standing_tile`'s x axis). Facings every
   `facing_step_deg` (10 deg) are tried, the ones nearest the tile's own axis
   first, for both wrist solutions of the grasp (`robot.grasp_frame`'s
   `rotation`, then `rotation_flipped`). A facing passes when the top-down
   grasp, the pitched hand at the release and every pitch step over the
   staging spot (`pitch_step_deg`, 15 deg) solve within the servo's
   tolerance. None passing is `unreachable` with nothing moved; `detail` lists
   what failed per facing.
3. The pick: open to the tile's width plus `grip_clearance` (4 mm) a side
   (`robot.set_grip`), rise to `lift_height`, go over the grasp point, read the
   jaw heading back and finish it on the last joint (`robot.move_to_joints`)
   when the arm left the roll free, come straight down, close, read the jaw
   gap (empty, 6 mm under or 8 mm over the tile's width is `ungraspable`, the
   jaws reopen and lift). A descent that ends 6 mm or more over its target
   stopped on the tile (`ungraspable`). The measured height at the close is
   the grasp height from here on. Lift straight up and read the jaws again
   (`dropped`).
4. The facing is fixed now: the first of the passing ones whose turn about
   the vertical fits the last joint's remaining travel from where it is.
5. Over the staging spot, turn the held tile about the vertical on the last
   joint until the end that goes up points along the facing, then pitch the
   hand 90 deg about the level axis through the tool centre, in
   `pitch_step_deg` steps, reading the jaws after each (`dropped`). One 90 deg
   pose command stalled in the station twin, and a single joint jump to the
   pitched pose flung the tile out (see the constants).
6. Read the roll back (the tile's up end about the approach) and finish it on
   the last joint; then the tile's lean from vertical, read from the hand's
   rotation, must be under `tilt_tolerance_deg` (6 deg; a 48 x 7.5 mm tile
   tips past its edge at 8.9 deg). Straight down until the tile's bottom is
   `release_gap` (1.5 mm) over the table; a descent that ends 6 mm high, or a
   lean past the tolerance there, lifts back (`unreachable`, still holding).
7. Open to the release width, back off `retreat` (30 mm) along the approach,
   rise.
8. The check (`verify`, default on): park (`park_pose`, default the start
   pose at least at lift height), read `robot.get_observation` and look at
   the staging spot in depth: the highest points inside the standing tile's
   footprint widened by 10 mm are its top. It stands when that top is within
   10 mm of the tile's height and within 8 mm of the spot (`standing`); a low
   top, a top off the spot or nothing raised is `fell`. A camera whose window
   is cut by the image edge or a quarter covered by something higher (the
   hand) does not judge; the overhead (`overhead`, `top`) is tried first, or
   only `camera_name`. No camera judging is `unverified`.

## Inputs

- `tile_center` (Vec3), `long_axis` (Vec3): the lying tile, world frame.
- `table_z` (float): the table plane, a station calibration.
- `staging` (Vec3, default the tile's own centre): where it should stand.
- `tile_size` (`{height, width, thickness}`, default 48 x 24 x 7.5 mm).
- `grasp_height` (0.011 m), `grip_offset` (0.014 m), `lift_height` (0.10 m),
  `release_gap` (0.0015 m), `grip_clearance` (0.004 m), `retreat` (0.03 m),
  `min_tool_height` (0.010 m).
- `facing_step_deg` (10), `pitch_step_deg` (15), `joint_limits` (default
  `robot.describe_arm`'s, else the YAM's), `roll_tolerance_deg` (4),
  `tilt_tolerance_deg` (6), `reach_tolerance` (0.015 m), `descent_tolerance`
  (0.006 m), `check_reach` (true).
- `verify` (true), `camera_name` (""), `park_pose` (out of the camera's view
  of the spot), `arm_id` (0; passed only when nonzero).

## Outputs

`picked`, `released`, `grasp_height` (measured at the close), `grip_width`
(jaw gap at the close, m; -1 unread), `facing_deg`, `yaw_turn_deg`,
`pitch_turn_deg` (the approach's measured turn), `roll_turn_deg` (last-joint
roll finish after the pitch), `tilt_at_release_deg` (from the hand's
rotation), `release_position`, `standing_tile` (centre at half the tile's
height, rotation whose x axis is the facing, `facing_deg`: what
placing-an-upright-domino-beside-neighbours takes as `tile`), `verified`,
`tile_top`, `standing_offset` (from depth, m; -1 unmeasured).

## Preconditions and boundaries

- **Unproven.** In the YAM station twin it stood no tile: the 90 deg pitch
  of a held tile over the table jumps the wrist joints, and the skill now
  refuses those paths before moving (Simulation check, below). Nothing here
  ran on a real YAM.
- **A real YAM station refuses a tool centre under 14 mm over its table.**
  The default grasp (11 mm, fingertips 0.8 mm over the table) needs that floor
  lowered; at 14 mm the fingertips hold only 3.7 mm of a 7.5 mm tile's edge
  (`MIN_BITE` allows it). Thin tiles are the reason this skill exists, and the
  reason it is fragile.
- The tile's lean at the release is read from the hand, assuming the tile
  turned with it. A tile that pivots in the pinch (the station twin's default
  contact, below) is not seen until the depth check.
- The staging spot must be clear of other tiles (not checked) and reachable
  by a pitched hand: on a YAM the release with the hand horizontal 40 mm over the table
  solves only in narrow heading bands (a 30 deg probe grid found one or two
  facing pairs per spot, none along the arm's radius), which is why the
  facing is searched.
- Only what the hand did is certain without the check.

## What changed from the source, and why

Franka-Domino-Project's `normalize_domino` (a real Franka with a wrist
RealSense, ROS 2 / MoveIt) picks a lying 48 x 24 x 7.5 mm tile top-down,
turns the wrist so the long axis is vertical, sets it down by a force-limited
descent at a staging spot 27 mm up, releases, and lets the caller re-grasp it
from above. That repository has **no licence**, so no code was taken: this is
written from the described behaviour. Here:

- the inputs are a world-frame centre and long axis, the outputs a standing
  pose in the convention of placing-an-upright-domino-beside-neighbours;
- the facing is chosen by an IK preview of the whole path (grasp, every pitch
  step, release) before anything moves, then by the last joint's remaining
  travel after the pick - the YAM reaches a pitched hand near the table only
  in narrow bands;
- the quarter turn is commanded in 15 deg pose steps about the level axis
  through the tool centre, and the wrist roll is read back and finished on the
  last joint (the station twin's solver leaves the roll free, PORTING lesson
  10);
- the force-limited descent is a measured height (`descent_tolerance`) and
  a lean read from the hand before and at the release;
- whether the tile stands is judged from depth, not assumed.

## The constants, and why

- **`TILE_SIZE` 48 x 24 x 7.5 mm**: Franka-Domino-Project's tiles.
- **`GRASP_HEIGHT = 0.011` m, `MIN_BITE = 0.002` m, `TIP_REACH = 0.0102` m,
  `MIN_TOOL_HEIGHT = 0.010` m**: in the station twin the hand arrived 2.2-2.7
  mm over the command (13.2-13.7 mm), leaving the pads 4.0-4.5 mm of the
  edge.
- **`GRIP_OFFSET = 0.014` m**: gripped at the centre, the pitched hand
  stalled 37.7 mm over the table when asked for 25.5 mm; 14 mm toward the top
  puts the tool centre 39.5 mm up at the release.
- **`LIFT_HEIGHT = 0.10` m**: the held tile swings half its height plus
  `GRIP_OFFSET` under the tool centre.
- **`FACING_STEP_DEG = 10`, `PITCH_STEP_DEG = 15`**: the release bands and the
  stalled / flung single-move pitches above.
- **`ROLL_TOLERANCE_DEG = 4`, `TILT_TOLERANCE_DEG = 6`, `TURN_TRIES = 3`,
  `CENTRE_TOLERANCE = 0.004` m**: a servo's 3 deg / 4 mm bounds plus a margin;
  the tile tips at 8.9 deg.
- **`RELEASE_GAP = 0.0015` m, `GRIP_CLEARANCE = 0.004` m, `RETREAT = 0.03` m**:
  under the servo's arrival bound; describe_gripper's clearance; past the
  fingertips' 10.2 mm.
- **`EMPTY_MARGIN`, `NARROW_MARGIN`, `FULL_MARGIN`, `DESCENT_TOLERANCE`,
  `REACH_TOLERANCE`**: grasping-flat-objects' (3 / 6 / 8 / 6 / 15 mm).
- **`HEIGHT_TOLERANCE = 0.010` m, `POSITION_TOLERANCE = 0.008` m,
  `SEARCH_MARGIN`, `OCCLUDER_RISE`, `OCCLUDED_SHARE`, `TOP_BAND`,
  `MIN_POINTS`**: the depth check of placing-an-upright-domino-beside-neighbours
  (a 10 deg lean moves a 48 mm top 8.3 mm; the station camera's RMS is 5.4 mm).

## Simulation check

**No tile was stood up in simulation.** A YAM station twin (RoboSimStudio,
left arm, linear_4310 hand, MuJoCo at 500 Hz, control at 50 Hz; simulator
ground truth only to set up and to measure): six standard dominoes (48 x 24 x
7.5 mm boxes, 10 g, friction 0.5) lying flat 0.52-0.66 m in front of the base
at yaws 0-150 deg, each handed to the skill as its simulator box (standing in
for perception) with the staging spot where it lay, or at x = 0.70 m (the
band where a probe of the pitched hand found the most headings). Standing
meant under 10 deg of tilt with its centre at half the tile's height.

| run | setup | stood | routes |
|---|---|---|---|
| 1 | an earlier draft of the script | 0/6 | 3 unreachable before or in the pitch, 2 descent to the spot stopped 12-61 mm high (one tile held at 10 deg lean, 1.9 mm off the spot), 1 dropped after the pitch |
| 2 | facing by IK preview (grasp, over the spot, release), pitch in 15 deg steps | 0/6 | 3 dropped in the pitch (steps 4-5 of 6), 2 unreachable (a pitch step timed out 24-28 mm short), 1 unreachable (no last-joint room) |
| 3 | as 2 with every pitch step previewed, grasp at 9.5 mm (5.4 mm of edge) | 0/2 | 2 dropped in the pitch (stopped after two) |
| 4 | as 3 at 11 mm, pad torsional friction 0.005 m | 0/3 | 3 dropped at step 4 of 6 (stopped after three) |
| 5 | shipped: joint-jump check, torsional, own spot | 0/6 | 6 unreachable in the preview, nothing moved |
| 6 | as 5, staging at x = 0.70 m | 0/6 | 6 unreachable in the preview, nothing moved |
| 7 | as 6, joint-jump limit 0.9 rad | 0/6 | 5 dropped in the pitch (steps 3-5), 1 unreachable holding |

- **Why the tile drops.** The trace of run 4 (tile 1, facing 30 deg): the IK
  solutions of the pitch steps over the spot moved the joints 0.2-0.5 rad a
  step until the last one, where the wrist pitch went from 0.30 rad to its
  1.5708 limit. The jaws read the tile's width after step 3 and closed empty
  after step 4, which took the servo 46 control steps (the others took 16).
  In runs 5 and 6, every facing that reached the release jumped a joint by
  0.60-3.14 rad somewhere in the pitch, so `PITCH_JOINT_JUMP` (0.6 rad)
  turned all twelve attempts into refusals before anything moved. With 0.9
  rad (run 7), five of the six facings it let through lost the tile.
- **The twin's pinch is a free hinge.** By default its contacts have no
  torsional friction (MuJoCo condim 3). A diagnostic pitch of a pinched tile
  turned the hand 90 deg while the tile stayed within 1.7 deg of flat,
  turning between the pads. Runs 4-7 gave the tiles condim 4 with 0.005 m
  torsional friction (MuJoCo's default), closer to rubber pads. The drops
  continued.
- Neighbours: in run 2 two lying tiles were pushed 8.9 and 13.2 mm (by the
  hand or a dropped tile). In run 7 the fourth tile started with the third
  still in the hand: the harness does not put down a tile the skill routes
  `unreachable` still holding.
- Not exercised in the twin: `standing`, `fell` and `unverified` from a real
  release, the depth check on a standing tile, and the regrasp by
  placing-an-upright-domino-beside-neighbours. These are covered by the unit
  tests only.
- What would help, untested: a staging spot chosen where the pitched release
  keeps the wrist off its limit, a pitch driven in joint space along a
  continuous IK chain, or the source's fallbacks: tipping the tile up against
  an edge, or handing it to the other arm (handing-over-between-arms).

## Provenance

Reimplemented from the behaviour of `normalize_domino` in
[Zhu-Chenyu/Franka-Domino-Project](https://github.com/Zhu-Chenyu/Franka-Domino-Project)
(`place_dominoes/place_dominoes/place_dominoes.py`). That repository states no
licence; none of its code is used here.
