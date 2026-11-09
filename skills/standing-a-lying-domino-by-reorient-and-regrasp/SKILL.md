---
name: standing-a-lying-domino-by-reorient-and-regrasp
description: Stands a flat-lying domino upright so a top-down pick can take
  it. Default strategy (handover, two arms) - this arm pinches the lying tile
  across its width, carries it flat to a point in the air between the arms,
  the helper arm pinches its other end across the thickness with a level
  approach and rolls a quarter turn on its last joint so the tile hangs
  upright, this arm takes the top end top-down across the broad faces, the
  helper lets go, and this arm sets the tile on its end at the staging spot;
  a depth check then confirms a tile stands there. Single-arm strategies
  (table_pivot, pose_pitch, hinge_tip) are kept but stood no tile in a YAM
  station twin. Use when a domino lies flat (from
  reading-domino-pips-and-orientation or a 6D pose) and must stand before
  placing-an-upright-domino-beside-neighbours picks it into a row.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [domino, reorient, regrasp, upright, handover, bimanual, depth-check]
  hub:
    stage: execute
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [domino, reorientation]
    provenance:
      authors: [runtime-si]
      code: https://github.com/Zhu-Chenyu/Franka-Domino-Project/blob/main/place_dominoes/place_dominoes/place_dominoes.py
      upstream_license: none (reference only; reimplemented from the behaviour, no code copied)
gap:
  requires: {connector: [robot.grasp_frame, robot.describe_gripper, robot.set_grip, robot.describe_arm]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.get_gripper
    - robot.describe_gripper
    - robot.describe_arm
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
    strategy: str
    pitch_joint: int
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
    standing: Released and retreated, and the depth check sees a tile's top at standing height (within 10 mm of the tile's height) at or near the spot; with verify=false, what the hands did only. `standing_tile` is the pose to hand to placing-an-upright-domino-beside-neighbours; when the top was seen more than 8 mm from the spot, `standing_tile` is moved to where it was seen and `detail` says so.
    fell: Released, but the depth check sees no tile's top at standing height in the window around the spot (lying, on its side, or gone).
    unverified: Released and retreated, but no camera saw the spot whole and unoccluded.
    ungraspable: Nothing was lifted - the descent to the tile stopped high (a fingertip on it, or it is not where it was seen), the jaws closed on nothing or not across its width, or the grasp height leaves the fingertips over the tile.
    dropped: The tile was picked, then the jaws read empty or not across it (after the lift, during a pitch, at the handover point, after the helper took it).
    unreachable: A preview, move or turn was refused before the release - no handover point both arms reach, the helper's jaws not across the thickness, no room on a last joint, a chained IK step that would jump a joint by more than pitch_joint_jump, no spot where a single-arm strategy reaches the upright hand - or the retreat after the release. `detail` says which hand still holds the tile.
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
the fingers on its broad faces (that skill's job; chain the two on
`standing`).

## Why two arms

Franka-Domino-Project stands the tile with one arm: pinch it across its width,
pitch the wrist 90 deg about the closing axis, set it on its end. On a YAM
that fails twice over (Simulation check, below):

- **The wrist.** Turning a held tile a quarter turn about the jaws near the
  table needs the wrist pitch joint (joint 4, +/-1.5708 rad) to cover its
  whole range: at a spot where the top-down set-down solves the upright hand
  runs into +1.53 rad, and where the upright hand solves the top-down
  set-down runs into -1.54 rad.
- **The pinch.** Across the 7.5 mm edge the linear_4310 pads are a stiff
  hinge, not a clamp: in a YAM station twin the held tile turned in them under
  its own weight (from 45 to 64 deg while the hand held 45 deg), was turned
  back flat by the table when its edge touched it (64 to 22 deg in two
  steps), and slid out between 45 and 80 deg of pitch in the air.

The handover needs neither: the turn is a roll on the helper's last joint
(+/-2.09 rad) in the air, and both hands that touch the table come from
above.

## The motion (strategy `handover`, the default)

1. **Where.** A test turn of 0.1 rad on each arm's first joint finds its
   base. Handover points are tried between the two bases
   (`HANDOVER_ACROSS`, fractions from this arm's base) and in front of them
   (`HANDOVER_FORWARD`), the tile's centre `handover_height` (0.12 m) over the
   table; the first where `robot.solve_ik` solves both this arm's top-down
   hand (jaws along the helper's approach) and the helper's level hand
   (approach away from its base) is used. None: `unreachable`, nothing
   picked. Or give `handover_point`.
2. **Helper first.** Before the pick, the helper opens to the thickness plus
   4 mm a side and goes (`robot.go_to_pose`) to wait 60 mm short of the
   handover point along its approach; it reads its jaw axis back and squares
   it across a flat tile's thickness on its last joint (the station twin's
   solver leaves the roll free; PORTING lesson 10), choosing, of the two
   squarings a half turn apart, the one that leaves its last joint the most
   room for a quarter roll. Which way the roll fits decides which end of the
   tile must point which way at the handover. (Done first because this arm's
   pinch sags: the tile turned 7-25 deg in it while it waited for the
   helper.)
3. **Pick.** This arm opens to the tile's width plus `grip_clearance` (4 mm)
   a side, squares its jaws on its last joint (the way round that leaves room
   for the turn after the pick), comes straight down to `grasp_height`
   (11 mm), closes across the 24 mm width `handover_grip_offset` (17 mm) from
   the centre, reads the gap (empty, 6 mm under or 8 mm over the width is
   `ungraspable`), lifts, and turns the held tile about the vertical on its
   last joint in `pitch_step_deg` (15 deg) steps onto its handover heading
   (the other heading when its joint has no room and the helper can roll
   either way).
4. **Carry, helper in.** It carries the tile to the handover point in
   chained IK steps and squares it onto its heading again (the steps leave
   the roll free: once it had drifted 35 deg, and the helper's roll then
   stood the tile at 55 deg). The helper moves in by chained IK until its
   tool centre is `helper_bite` (10 mm) past the tile's near edge,
   `helper_offset` (4 mm) from the centre toward the far end, and closes. A
   gap not across the 7.5 mm thickness is `unreachable` (the helper backs
   off).
5. **Hand over.** This arm opens and rises 60 mm. The helper's jaws are read
   (`dropped`).
6. **Roll.** The helper rolls a quarter turn about its approach on its last
   joint, in steps, so its own end goes up and the tile hangs under its pads;
   the residual roll is read back and corrected. A lean of the broad faces
   from vertical over `tilt_tolerance_deg` (6 deg) is `unreachable`.
7. **Take the top.** This arm comes top-down over the tile's top end with
   its jaws across the thickness (`robot.grasp_frame`, squared on the last
   joint), straight down until its fingertips are `top_bite` (8 mm) below the
   top, and closes; a gap not across the thickness is `unreachable`. The
   helper opens, backs off along its approach and returns to where it
   started.
8. **Set down.** This arm carries the upright tile over the staging spot
   (`staging`, default where the tile lay), straight down until its bottom is
   `release_gap` (1.5 mm) over the table, opens to the thickness plus 4 mm a
   side, rises 60 mm.
9. **Check** (`verify`, default on): park (`park_pose`), read
   `robot.get_observation` and look at the spot in depth; the highest points
   inside the standing tile's footprint widened by 10 mm are its top. It
   stands when that top is within 10 mm of the tile's height (a tile at rest
   that tall leans under its 8.9 deg tipping angle); a top more than 8 mm from
   the spot moves `standing_tile` to where it was seen. A low top or nothing
   raised is `fell`. A camera whose window is cut by the image edge or a
   quarter covered by something higher does not judge; the overhead
   (`overhead`, `top`) is tried first, or only `camera_name`.

## The single-arm strategies (kept, unproven)

- **`table_pivot`**: test-turns the wrist joints with the hand empty to find
  the one whose axis is level and whose quarter turn fits its limits (joint 4
  on a YAM), rehearses the upright hand with the empty hand turned on that
  joint and previews candidate spots from there, picks, turns the tile into
  that joint's plane, optionally pitches `air_pitch_deg` in the air on that
  joint alone, sets the tile on the table and turns it upright about its
  lower edge in `pivot_step_deg` (5 deg) chained IK steps; a failed turn is
  backed out the way it came and the next spot tried.
- **`pose_pitch`**: the earlier version - the quarter turn in the air in pose
  steps about the tool centre over the staging spot, the facing chosen by an
  IK preview of the grasp, every pitch step and the release.
- **`hinge_tip`**: a top-down pinch 8 mm from one end used as a pin; the hand
  keeps its rotation and moves on an arc about the far end's bottom edge, so
  the tile should turn in the pads onto that end.

In all of them every chained IK step is refused before it moves when its
solution would move a joint but the last by more than `pitch_joint_jump`
(0.6 rad; the jump that threw tiles out of the pose pitch).

## Inputs

- `tile_center` (Vec3), `long_axis` (Vec3): the lying tile, world frame.
- `table_z` (float): the table plane, a station calibration.
- `staging` (Vec3, default where the tile lay): where it should stand.
- `strategy` (`handover`; `table_pivot`, `pose_pitch`, `hinge_tip`).
- `helper_arm_id` (1), `handover_point` (default searched),
  `handover_height` (0.12 m), `handover_grip_offset` (0.017 m),
  `helper_offset` (0.004 m), `helper_bite` (0.010 m), `top_bite` (0.008 m).
- `tile_size` (`{height, width, thickness}`, default 48 x 24 x 7.5 mm),
  `grasp_height` (0.011 m), `grip_offset` (0.014 m), `lift_height` (0.10 m),
  `release_gap` (0.0015 m), `grip_clearance` (0.004 m), `retreat` (0.03 m),
  `min_tool_height` (0.010 m), `pitch_step_deg` (15), `pitch_joint_jump`
  (0.6 rad), `joint_limits` (default `robot.describe_arm`'s, else the YAM's),
  `roll_tolerance_deg` (4), `tilt_tolerance_deg` (6), `reach_tolerance`
  (0.015 m), `descent_tolerance` (0.006 m), `check_reach` (true).
- Single-arm strategies: `air_pitch_deg` (0), `pivot_step_deg` (5), `press`
  (0), `probe_turn` (0.15 rad), `stage_step` (0.03 m), `stage_reach`
  (0.30 m), `stage_attempts` (3), `hinge_offset` (0.016 m), `hinge_press`
  (0.002 m), `facing_step_deg` (10).
- `verify` (true), `camera_name` (""), `park_pose` (out of the camera's view
  of the spot), `arm_id` (0; passed only when nonzero).

## Outputs

`strategy`, `pitch_joint` (table_pivot's learned joint, 1-based; -1 else),
`picked`, `released`, `grasp_height` (measured at the close), `grip_width`
(jaw gap at the close, m; -1 unread), `facing_deg` (the standing tile's broad
faces' normal), `yaw_turn_deg`, `pitch_turn_deg` (handover: the helper's
roll), `roll_turn_deg`, `tilt_at_release_deg` (handover: the tile's lean in
the helper's hand as this arm estimates it), `release_position`,
`standing_tile` (centre at half the tile's height, rotation whose x axis is
the facing, `facing_deg`: what placing-an-upright-domino-beside-neighbours
takes as `tile`), `verified`, `tile_top`, `standing_offset` (from depth, m;
-1 unmeasured).

## Preconditions and boundaries

- **Two arms that can both reach a point between them** (the helper's id is
  `helper_arm_id`); the helper must be free and away from the tiles. A
  single-arm robot has only the single-arm strategies, which stood no tile in
  the YAM twin.
- **A real YAM station refuses a tool centre under 14 mm over its table.**
  The default grasp (11 mm, fingertips 0.8 mm over the table) needs that floor
  lowered; at 14 mm the fingertips hold only 3.7 mm of a 7.5 mm tile's edge.
- The tile's lean in the helper's hand within its broad faces is carried over
  from this arm's pinch, which the tile turns in; it is not checked (the
  hanging tile straightens under its own weight, and this arm's top grasp
  takes any point along the width). Only the depth check sees the result.
- The staging spot must be clear of other tiles (not checked). The helper's
  path in front of the bases must be clear too.
- Nothing here ran on a real YAM.

## What changed from the source, and why

Franka-Domino-Project's `normalize_domino` (a real Franka with a wrist
RealSense, ROS 2 / MoveIt) picks a lying 48 x 24 x 7.5 mm tile top-down,
turns the wrist so the long axis is vertical, sets it down by a force-limited
descent at a staging spot, releases, and lets the caller re-grasp it from
above. That repository has **no licence**, so no code was taken: this is
written from the described behaviour. Here:

- the inputs are a world-frame centre and long axis, the outputs a standing
  pose in the convention of placing-an-upright-domino-beside-neighbours;
- the wrist pitch is replaced by a two-arm handover with a roll (default),
  because on a YAM the pitch near the table does not fit the wrist joint and
  the edge pinch does not hold the tile through it; the single-arm versions
  stay as strategies;
- the force-limited descent is a measured height (`descent_tolerance`);
- whether the tile stands is judged from depth, not assumed.

## The constants, and why

- **`TILE_SIZE` 48 x 24 x 7.5 mm**: Franka-Domino-Project's tiles.
- **`GRASP_HEIGHT = 0.011` m, `MIN_BITE`, `TIP_REACH = 0.0102` m,
  `MIN_TOOL_HEIGHT`**: in the station twin the hand arrived 2.2-2.7 mm over the
  command, leaving the pads 4.0-4.5 mm of the edge.
- **`HANDOVER_HEIGHT = 0.12` m, `HANDOVER_ACROSS`, `HANDOVER_FORWARD`**: a
  top-down YAM hand 0.15 m over the table ran into its wrist limit at the
  midpoint between the hands (0.42 m from its base) in the twin; the grid is
  a choice, and a point is used only when both arms' IK solves there.
- **`HANDOVER_GRIP_OFFSET = 0.017` m, `HELPER_OFFSET = 0.004` m, `TOP_BITE =
  0.008` m, `HELPER_BITE = 0.010` m**: keep the two hands' pads apart on a
  48 mm tile at the handover and at the top grasp. With the helper 12 mm from
  its end and that end down after the roll, one tile in four toppled to 53 deg
  in the helper's pads (centre of mass above them); with its end up the tile
  hangs under them.
- **`PITCH_JOINT_JUMP = 0.6` rad**: the pose pitch's last step jumped the
  wrist pitch 1.27 rad onto its limit and lost the tile.
- **`AIR_PITCH_DEG = 0`, `PRESS = 0`, `SETDOWN_GAP`, `STAGE_*`,
  `PITCH_ROOM`, `PROBE_TURN`, `LEVEL_TOLERANCE`, `JOINT_SETTLE`, `IK_STEP`,
  `TURN_STEP_DEG`**: table_pivot's; the measurements are in the docstrings.
- **`ROLL_TOLERANCE_DEG = 4`, `TILT_TOLERANCE_DEG = 6`, `TURN_TRIES = 3`,
  `CENTRE_TOLERANCE = 0.004` m**: a servo's 3 deg / 4 mm bounds plus a margin;
  the tile tips at 8.9 deg.
- **`EMPTY_MARGIN`, `NARROW_MARGIN`, `FULL_MARGIN`, `DESCENT_TOLERANCE`,
  `REACH_TOLERANCE`**: grasping-flat-objects' (3 / 6 / 8 / 6 / 15 mm).
- **`HEIGHT_TOLERANCE = 0.010` m, `POSITION_TOLERANCE = 0.008` m,
  `SEARCH_MARGIN`, `OCCLUDER_RISE`, `OCCLUDED_SHARE`, `TOP_BAND`,
  `MIN_POINTS`**: the depth check of placing-an-upright-domino-beside-neighbours.

## Simulation check

A YAM station twin (RoboSimStudio, both arms, linear_4310 hands, MuJoCo at
500 Hz, control at 50 Hz; simulator ground truth only to set up and to
measure): standard dominoes (48 x 24 x 7.5 mm boxes, 10 g, friction 0.5, the
scene's default contacts) lying flat 0.27-0.34 m in front of the left arm's
base, spread over 70 deg around it, at yaws 0-160 deg, each handed to the skill
as its simulator box (standing in for perception), staging where it lay. A
tile counts as standing when its long axis is within 10 deg of vertical, its
centre at half its height (+/-10 mm) and within 15 mm of the spot.

**Handover (shipped): 8 of 12 tiles stand.** Two runs of six tiles
(positions and yaws in the two sets differ), the shipped script:

| set | stood (truth) | skill's routes | the others |
|---|---|---|---|
| A: yaws 0, 45, 90, 135, 20, 110 | 4/6 | 3 standing, 1 unverified (stood), 2 unreachable | the helper closed on nothing (the tile fell out of this arm's pinch as it came in); this arm's last joint had no room to turn the held tile onto its handover heading (still holding) |
| B: yaws 160, 70, 10, 100, 150, 60 | 4/6 | 2 standing, 2 unverified (both stood), 2 unreachable | the same two causes, once each |

- Every tile the hands released stood: true tilt 0.0 deg, centre 0.05-0.8 mm
  from the spot. No route said `standing` for a tile that did not stand.
- `unverified` (3): the overhead camera's window over the spot was a
  quarter covered by something higher (the park pose is the harness's).
- Before the last two changes (re-squaring at the pick for room, and the
  helper squared before the pick) the same 12 tiles gave 7/12; before the
  helper waited squared at the handover point, two of three far tiles were
  knocked out of this arm's pinch by the helper.
- One run of the harness's 6-tile set takes 22-25 min of simulation
  (~3.5 min a tile).

**Single-arm strategies: 0 stood.**

| run | strategy, setup | stood | what happened |
|---|---|---|---|
| 1-7 (earlier) | pose_pitch, several variants | 0 of ~39 | the pitch steps jumped the wrist pitch onto its limit; refused by the joint-jump guard or the tile dropped (see git history) |
| diagnostic | pitch joint (4) turned alone in the air, 10 steps | - | the tile followed the hand to 75 deg of elevation, then slid out of the pads between 70 and 80 deg of hand pitch; the pitched hand then descended to 40 mm over the table at 0.55 m from the base with joint 4 at 1.496 rad |
| j1 | table_pivot, 60 deg in the air, 1 mm press | 0/4 | 2 lost turning up on the table, 1 dropped in the air at 45 deg, 1 no spot (joint 4 at -1.56) |
| t1 | table_pivot, whole turn on the table, spots previewed from the top-down arm | 0/1 | every preview failed (false negatives); lost at 50 deg on the table |
| t2 | + the empty rehearsal, no press | 0/1 | at the rehearsed spots the top-down set-down ran joint 4 to -1.54 |
| t3 | + 45 deg in the air | 0/1 | the tile turned in the pinch from 45 to 64 deg while the hand held 45, the table turned it back to 22 deg, it fell out; the last upright step needed joint 4 at +1.53 |
| h1 | hinge_tip | 0/2 | the tile lifted with the hand, turning 0.6 deg a step; its foot left the table and it was set down lying |

The 7 earlier pose_pitch runs used the same scene with six tiles 0.52-0.66
m from the base; runs 4-7 gave the tiles condim 4 and 0.005 m torsional
friction through a harness-only patch. All runs since use the scene's default
contacts.

Not exercised in the twin: the regrasp by
placing-an-upright-domino-beside-neighbours from a stood tile, and
`table_pivot` / `hinge_tip` reaching `standing`. Nothing ran on a real YAM.

## Provenance

Reimplemented from the behaviour of `normalize_domino` in
[Zhu-Chenyu/Franka-Domino-Project](https://github.com/Zhu-Chenyu/Franka-Domino-Project)
(`place_dominoes/place_dominoes/place_dominoes.py`). That repository states no
licence; none of its code is used here.
