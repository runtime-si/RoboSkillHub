---
name: placing-an-upright-domino-beside-neighbours
description: Picks a standing domino top-down and sets it upright at a slot in
  a row without touching the tiles already standing there - the jaws close
  across the tile's width so the fingers stay beside the row, open only a few
  millimetres wider than the tile, the hand crosses above every standing tile,
  comes in along the row from its free end, descends vertically, opens just
  enough and retreats straight up; the jaw heading is read back and finished
  on the last joint. Then checks from depth that the tile stands at the slot
  and that every neighbour still stands. Use when building a domino line or
  curve one tile at a time, with the slot poses planned and the tile to pick
  standing upright.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [domino, place, upright, row, release, depth-check]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/BasisResearch/predicators/blob/afb6f2918091f5ca372a75b9a335aea317a8a209/predicators/ground_truth_models/domino/options.py
      upstream_license: MIT
      paper: https://arxiv.org/abs/2509.26255
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
    # OPTIONAL extra inputs (not listed: gap has no list type, so they can only
    # be wired from a script node inside the same subgraph): `slots` (a list of
    # slot poses) with `slot_index` instead of `slot`, and `neighbours` (the
    # standing tiles' poses; default the slots before slot_index).
    slot: Se3Pose
    tile: Se3Pose
    table_z: float
  produces_outputs:
    released: bool
    picked: bool
    grasp_height: float
    grip_width: float
    held_width: float
    jaw_error_deg: float
    wrist_turn_deg: float
    release_position: Vec3
    verified: bool
    tile_top: float
    placement_offset: float
    neighbours_standing: int
  exit_conditions:
    placed: Released at the slot and retreated, and the depth check sees a tile standing there (its top within 10 mm of the tile's height and within 8 mm of the slot) with every seen neighbour standing; with verify=false, what the hand did only.
    knocked_neighbour: Released, but a neighbour that stood before the place (or was not seen then) no longer stands at its pose; `neighbours_knocked` lists which, `detail` says whether the placed tile stands.
    fell: Released, the neighbours stand, but the placed tile does not stand at the slot - its top is low (lying, leaning hard), more than 8 mm off the slot (leaning, misplaced), or nothing is there.
    unverified: Released and retreated, but no camera saw the slot whole and unoccluded (the hand in the way, no depth, outside every image).
    unreachable: Nothing was released - the slot is so close to a neighbour that an open finger would touch it, the grasp height is outside what the fingertips can do, a grasp or release pose failed the IK preview, a move was refused, the jaws stayed turned off the tile, or the descent to the slot stopped high (lifted back); `detail` says whether the tile is still held. Also a refused retreat after the release (`released` true).
    not_holding: The grasp closed on nothing or not across the tile, the descent to the tile stopped on it, or the jaws read empty over the slot (dropped in transit); nothing was placed.
  canonical_scripts:
    - place_domino: scripts/place_domino.py
  examples:
    - title: Pick a standing domino and place it at the next slot of a row
      path: examples/canonical_subgraph.json
---

# placing-an-upright-domino-beside-neighbours

Build a domino row one tile at a time: pick a standing tile, stand it at the
next slot, and do not touch the ones already up.

## The motion

1. Geometry first. The slot and every neighbour are `(x, y, facing)`: an
   Se3Pose whose x axis is the facing (the broad faces' normal, the way the
   tile topples, the way the row runs), or a `position` with `facing_deg`.
   The jaws close across the tile's **width** (close heading = facing + 90),
   so the two fingers stand beside the row and never in the gap between two
   tiles. With the jaws open to the tile's width plus `grip_clearance` a side
   (32 mm for a 24 mm tile) and a 30 mm blade each, a neighbour whose
   footprint comes within 4 mm of that envelope is `unreachable` before
   anything moves. A tile at one tile-height pitch along the row is 30 mm
   clear; a tile in a parallel row 30 mm to the side is not.
2. Neighbours as they stand now, from depth (below), so a tile that was down
   before is not blamed on this placement.
3. The pick (predicators' pick skill: MoveAbove, PreOpen, MoveToGrasp, Grasp,
   LiftSlightly): preview the grasp pose at `grasp_height` over the tile's
   bottom for both wrist solutions (`robot.solve_ik`, the nearer heading
   first), open the jaws only to the tile's width plus 4 mm a side
   (`robot.set_grip`), rise to the transit height, go over the tile, read the
   jaw heading back and finish it on the last joint (`robot.move_to_joints`)
   when the arm left the wrist roll free, come straight down, close, and read
   the jaw gap: empty, narrower than the tile less 6 mm or wider than it plus
   8 mm is `not_holding` (the jaws reopen to the pinch width and lift). The
   hand's measured height at the close is the tile's hang under the tool
   centre for the rest. Lift straight up.
4. The transit height keeps the held tile's bottom (or the open fingertips)
   `transit_clearance` (20 mm) over every standing tile's top.
5. The place (predicators' place skill with a move-above and the partial
   release): go to a lead-in point `lead_in` (30 mm) back along the row from
   the slot, on the side away from the nearest neighbour (the row's free end;
   on an empty row, the hand's side), turn the jaws to the slot's facing and
   read them back as before, then move along the row to over the slot. Read
   the jaws: empty is `not_holding` (dropped in transit). Straight down until
   the tile's bottom is `release_gap` (1.5 mm) over the table; a descent that
   ends more than 6 mm high stopped on something and lifts back
   (`unreachable`, still holding). Open the jaws only to the tile's width plus
   `release_clearance` (4 mm) a side and rise straight up holding that width.
6. The check (`verify`, default on): move the hand to `park_pose` (default:
   where it started, at least at transit height), read `robot.get_observation`
   and judge the slot and each neighbour: in the tile's footprint widened by
   6 mm (and, for the placed tile, up to 20 mm along the row to follow a
   lean, short of the next tile), the highest depth points are its top. It
   stands when that top is within 10 mm of the tile's height and its centre
   within 8 mm of the slot. A camera whose window over the footprint is cut by
   the image edge, or is a quarter or more something higher than a tile (the
   hand), does not judge; the overhead (`overhead`, `top`) is tried first,
   then the rest, or only `camera_name`.

## Inputs

- `slot` (Se3Pose, its x axis the facing) or `slots` + `slot_index`.
- `tile` (Se3Pose): the standing tile to pick. `None` places a tile already
  in the hand, gripped `grasp_height` over its bottom.
- `table_z` (float): the table plane, a station calibration.
- `neighbours`: the standing tiles' poses (default the slots before
  `slot_index`, or none with `slot`).
- `tile_size` (`{height, width, thickness}`, default 48 x 24 x 7.5 mm).
- `grasp_height` (0.030 m; sample it in `GRASP_BAND`, 0.015-0.046 m),
  `release_gap` (0.0015 m), `grip_clearance` / `release_clearance` (0.004 m),
  `transit_clearance` (0.02 m), `lead_in` (0.03 m).
- `verify` (true), `camera_name` (""), `park_pose` (a pose out of the
  camera's view of the row; default the start pose).
- `yaw_tolerance_deg` (4), `roll_limits` (the YAM's joint 6, +-2.0944 rad),
  `reach_tolerance` (0.015 m), `descent_tolerance` (0.006 m), `check_reach`
  (true), `arm_id` (0; passed only when nonzero).

## Outputs

`released`, `picked`, `grasp_height` (the measured hang), `grip_width` (jaw
gap at the close, m; -1 unread), `held_width` (over the slot),
`jaw_error_deg`, `wrist_turn_deg`, `release_position` (where the hand left
the tile's bottom centre), `verified`, `tile_top` and `placement_offset`
(from depth, m; -1 unmeasured), `neighbours_standing`, `neighbours_knocked`
(indices into `neighbours`).

## Preconditions and boundaries

- The tile to pick stands upright and alone enough for the open fingers
  (the pick site is not checked against its own neighbours).
- `park_pose` must leave the camera's rays to the row clear. In the station
  twin the start pose stood exactly between the overhead camera and the far
  end of the row and the check came back `unverified` (below).
- A domino line only propagates when the pitch is under the tile's height;
  this skill places at whatever slots it is given (see the push check below).
- Only what the hand did is certain without the check; with it, the verdict
  is the depth image's, no better than the camera's calibration.

## What changed from the source, and why

predicators' domino options (`options.py`) build a Pick and a Place from its
skill factories and plan both with BiRRT in PyBullet against the simulator's
state: Pick moves above the domino at transport height, pre-opens to the
object's width plus a slack, descends to a sampled `grasp_z_offset` over the
domino's origin (aborting when the target moves 4 mm), closes and lifts 1 cm;
Place moves above the release point, descends straight, opens a step until
the simulator drops the grasp constraint, opens 8 mm more, retreats straight
up holding that width and only opens fully at transport height. Its docstring
records the Panda's band: fingers collide below 0.045 m and close over the
domino above 0.080 m (on a 0.15 m tile). Here:

- the simulator state is perception: the tile and slots are world-frame poses
  and the Upright / Toppled predicates (roll under 5 / over 10 deg) are a
  depth check of the slot and of every neighbour;
- BiRRT's held-bystander clearance (15 mm, "a graze topples a standing
  domino") is a transit height over every tile plus a lead-in along the row
  from its free end and a vertical descent;
- the jaws close across the tile's width, so the fingers never enter the gap
  between tiles (upstream's yaw came from the plan);
- the grasp height is a tool-centre height over the tile's bottom, its band
  re-derived for the linear_4310 (below);
- the release "until is_held flips" is a jaw width of the tile plus 4 mm a
  side, held through the retreat; holding is read from the jaw gap;
- the jaw heading is read back and finished on the last joint, because the
  YAM station twin's solver leaves the wrist roll free (it turned the hand
  70 deg on the first pick of every run).

## The constants, and why

- **`TILE_SIZE` 48 x 24 x 7.5 mm**: a standard domino (Franka-Domino-Project,
  zmatt.net's physics tests).
- **`GRASP_HEIGHT = 0.030` m, `GRASP_BAND = (0.015, 0.046)`, `MIN_BITE =
  0.010` m**: the sweep below; the floor is a real YAM station's (it refuses a
  tool centre under 14 mm over its table).
- **`TIP_REACH = 0.0102` m**: linear_4310 fingertips under the tool centre,
  when `robot.describe_gripper` does not say.
- **`GRIP_CLEARANCE = RELEASE_CLEARANCE = 0.004` m**: upstream's 8 mm
  `_RELEASE_CLEAR_SLACK`, half a side.
- **`RELEASE_GAP = 0.0015` m**: under a servo's 4 mm arrival bound.
- **`TRANSIT_CLEARANCE = 0.02` m**: upstream plans 15 mm.
- **`LEAD_IN = 0.03` m, `FINGER_HALF_DEPTH = 0.012` m (not measured),
  `FINGER_BLADE = 0.030` m (the twin's trimmed blade), `NEIGHBOUR_CLEARANCE =
  0.004` m**: choices for the approach and the crowding check.
- **`EMPTY_MARGIN`, `NARROW_MARGIN`, `FULL_MARGIN`, `DESCENT_TOLERANCE`,
  `REACH_TOLERANCE`**: grasping-flat-objects' (3 / 6 / 8 / 6 / 15 mm).
- **`YAW_TOLERANCE_DEG = 4`, `ROLL_LIMITS`, `LIMIT_MARGIN`**: a servo's 3 deg
  plus one; the YAM's joint 6.
- **`HEIGHT_TOLERANCE = 0.010` m, `POSITION_TOLERANCE = 0.008` m**: a tile
  leaning 10 deg (upstream's fallen threshold) moves its top 8.3 mm; a lying
  one reads 7.5 or 24 mm; the station camera's RMS is 5.4 mm.
- **`SEARCH_MARGIN = 0.006`, `LEAN_SEARCH = 0.020`, `OCCLUDER_RISE = 0.015`,
  `OCCLUDED_SHARE = 0.25`, `TOP_BAND = 0.006`, `MIN_POINTS = 4`** (m): the
  depth check's choices; a 24 x 7.5 mm top is about 12 x 4 px from the
  overhead at 0.9 m.

## Simulation check

A YAM station twin (RoboSimStudio, left arm, linear_4310 hand, MuJoCo at
500 Hz, control at 50 Hz; simulator ground truth only to set up and to
measure). Standard dominoes (48 x 24 x 7.5 mm boxes, 10 g, friction 0.5,
predicators' real-table fit) stood in a staging row 50 mm apart; a line of
slots facing +y ran beside it. For tile k a script handed the skill tile k's
simulator box as its pose (standing in for perception), slot k, and slots
0..k-1 as the neighbours; the skill picked, placed and checked from the
twin's calibrated overhead Orbbec (640 x 480, about 30 deg from vertical) and
a wrist camera, with `park_pose` beside the arm's base. After each placement
the scene ran 0.5 s and every tile was measured (standing: under 10 deg of
tilt and within 15 mm of its slot). Then the closed hand pushed the first tile
along the facing, 35 mm over the table, and the scene ran 3 s (toppled: over
45 deg).

| line | route | placement error | tilt | depth check: top / offset | neighbours moved | push on the first tile |
|---|---|---|---|---|---|---|
| 6 tiles, pitch 1.0 h (48 mm) | placed 6/6 | 0.12-0.53 mm | 0.0 deg | 48 mm / 1.0-2.0 mm (5.9 mm once, see below) | none | 6/6 toppled |
| 8 tiles, pitch 1.0 h | placed 8/8 | 0.14-1.93 mm | 0.0 deg | 48 mm / 0.5-1.9 mm | none | 8/8 toppled |
| 6 tiles, pitch 1.5 h (72 mm) | placed 6/6 | 0.29-0.83 mm | 0.0 deg | 48 mm / 1.0-2.5 mm | none | 1/6 toppled (the first only) |

- No placement touched a neighbour: no standing tile moved or tilted
  measurably (under 0.01 mm and 0.01 deg) in any run - 44 placements in all,
  the runs below included.
- The spacing check: at one tile height the whole line went down (a falling
  tile's top just reaches the next one's foot; they came to rest at 81-90 deg),
  at 1.5 tile heights the first tile fell flat and touched nothing. A row
  meant to cascade needs a pitch under the tile's height; this skill places at
  whatever pitch it is given.
- The free wrist roll: every run's first pick needed a -70 deg last-joint turn
  after the hover reported `reached`, later picks and slots -4 to -10 deg; the
  jaws then ended within 2.2 deg of the tile's width axis.
- The hand arrived 2.2 mm over every commanded grasp height (the servo), so
  `grasp_height` (the measured hang) read 32.2-32.4 mm for the default 30 mm.
- Grasp height sweep (6 tiles, pitch 1.5 h, one height each): commanded 15,
  22, 30, 38 and 46 mm (fingertips 7 to 38 mm over the table, 9.9 mm or more
  of bite) all placed, 0.35-1.26 mm off the slot. Commanded 53 mm (the hand at
  55.3 mm, 2.9 mm of bite) closed reading the tile's width, lost it in transit
  and routed `not_holding` over the slot (the tile lay 86 mm away, nothing
  else moved): hence `MIN_BITE` 10 mm and `GRASP_BAND` 15-46 mm.
- Release faults: with `release_gap` 30 mm every tile still landed standing
  (0.9-2.1 mm off, `placed` 6/6). With 80 mm every tile fell on its long side
  (top 24 mm, 25 mm off); the skill routed `fell` 6/6, and the tiles already
  down did not count as knocked (`neighbours standing 0/k`). The depth
  verdict agreed with the measured state in all 41 released placements it
  judged (the other two were `unverified`, below).
- Not exercised in the twin: `knocked_neighbour` (no placement knocked a
  standing tile; the 80 mm drops fell sideways, across the row) and the
  descent stopping high - both covered only by the unit tests.
- The park pose matters. With the default (the start pose) the hand stood
  between the overhead camera and the far end of the row: in a 1.5 h line
  slots 4 and 5 came back `unverified`, and in a 1.0 h line slot 5 read 5.9 mm
  off (true 0.5 mm) through a partly covered window. A wrist camera parked
  away from the row read tiles 7-12 mm off their true spot, which is why the
  overhead is tried first.
- Nothing here ran on a real YAM; the real-station paths (a raised refusal, a
  full-pose arm that arrives turned) are unit-tested only.

## Provenance

Adapted from predicators (BasisResearch/predicators, MIT, Copyright (c) 2022
Rohan Chitnis and Tom Silver, `LICENSE.predicators`): `ground_truth_models/domino/options.py`
(`_pick_param_defs`, `_create_sf_pick`, `_create_sf_place`),
`ground_truth_models/skill_factories/pick.py` and `place.py`, and the
upright / fallen thresholds of `envs/pybullet_domino/components/domino_component.py`.
predicators is the codebase of the ExoPredicator paper (Liang et al., ICLR
2026, arXiv 2509.26255), whose README lists it.
