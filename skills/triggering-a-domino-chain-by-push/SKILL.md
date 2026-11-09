---
name: triggering-a-domino-chain-by-push
description: Topples the first domino of a standing line so the chain runs -
  closes the hand with the blades across the tile, comes down behind the tile
  (against its facing) to a contact height on its upper half, pushes slowly
  along the facing a short computed stroke that carries the tile past its
  tipping angle without reaching the next tile, backs off along the stroke and
  rises, then checks from depth that the first slot no longer stands. Use
  after a domino line or curve is built (planning-a-domino-layout's slot 0 is
  the trigger tile), to start the cascade; hand the clip to
  verifying-a-domino-cascade-from-video to judge the whole chain.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [domino, push, trigger, topple, cascade, depth-check]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/BasisResearch/predicators/blob/afb6f2918091f5ca372a75b9a335aea317a8a209/predicators/ground_truth_models/skill_factories/push.py
      upstream_license: MIT
      paper: https://arxiv.org/abs/2509.26255
gap:
  requires: {connector: [robot.grasp_frame, robot.describe_gripper]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.describe_gripper
    - robot.grasp_frame
    - robot.solve_ik
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.close_gripper
  required_inputs:
    # OPTIONAL extra inputs (gap has no list type, so `slots` can only be wired
    # from a script node in the same subgraph): `slots` (planning-a-domino-layout's
    # `slots` / `slot_poses`) with `slot_index` instead of `slot`, which also
    # gives the next slot and the one behind; `next_slot` (Se3Pose).
    slot: Se3Pose
    table_z: float
  produces_outputs:
    pushed: bool
    contact_height: float
    tip_travel: float
    push_travel: float
    stroke_length: float
    stroke_time: float
    stroke_speed: float
    stroke_steps: int
    trigger_time: float
    push_direction: Vec3
    verified: bool
    first_standing: bool
    next_standing: bool
  exit_conditions:
    triggered: The stroke ran, the hand backed off, rose and parked, and the depth check sees no tile standing at the slot (`next_standing` says whether the next tile still stands); with verify=false, the stroke carried the blade front at least the tipping travel past the tile's back face (what the hand did only).
    no_contact: The tile still stands at the slot after the stroke - the stroke ended inside the servo's arrival tolerance but short of the tipping travel and the tile rocked back (`pushed` false), or the hand passed over or beside it, or it is not where the slot says (`pushed` true) - or nothing stood at the slot before the push (nothing moved then).
    blocked: The descent behind the tile stopped high (something stands where the hand comes down; lifted back), or the stroke was held - it timed out or ended more than 5 mm short of its end - before the blade reached the tipping travel (a tile that does not give, or the arm at a limit); the hand backed off and rose.
    unverified: The stroke ran, but no camera saw the slot whole and unoccluded after the push; `pushed` says whether the blade reached the tipping travel.
    unreachable: Nothing was pushed - the contact height is outside its band, the stroke would reach the next tile (the line is too tight to push the first tile alone) or the hand would come down on the tile behind, the stroke's start or end failed the IK preview for both wrist solutions, or a move before the stroke was refused.
  canonical_scripts:
    - trigger_domino: scripts/trigger_domino.py
  examples:
    - title: Push the first tile of a planned domino line
      path: examples/canonical_subgraph.json
---

# triggering-a-domino-chain-by-push

Start a domino cascade: push the first tile of a standing line over, gently,
from behind, and get the hand out of the way.

## The motion

1. Geometry first. The slot is `(x, y, facing)`: an Se3Pose whose x axis is
   the facing (the way the tile topples and the line runs; planning-a-domino-layout's
   `first_slot` / `trigger.pose`), or a `position` with `facing_deg`. With
   `slots` + `slot_index` the next slot and the one behind come from the list.
2. The contact height: the closed fingertips at `contact_fraction` (0.7) of
   the tile's height over the table, on its upper half. The fingertips are the
   hand's lowest point and a tile that tips away from a vertical finger face
   keeps touching it there, so that is where it is pushed.
3. The stroke (predicators' push skill: CloseFingers, above and behind,
   descend, push along the facing). The tile tips once its centre of mass
   passes over its front foot, at `atan(thickness / height)` (8.9 deg for a
   48 x 7.5 mm tile). At the contact height `c` the back face has then moved
   `tip_travel = t (1 - cos a) + c sin a` (5.3 mm at 0.7 h). The blade front
   starts `approach_clearance` (8 mm) behind the back face and ends
   `tip_travel + overshoot` (8 mm) past it, so the stroke is about 21 mm.
   When the next slot is known and the stroke would come within 6 mm of its
   tile's back face, nothing moves (`unreachable`: too tight to push the first
   tile alone). A tile behind the trigger tile closer than the hand needs to
   come down is `unreachable` too.
4. The hand: jaws closing across the tile's width (`close_heading_offset_deg`
   90), so the two closed blades span the tile; `robot.grasp_frame` gives both
   wrist solutions and `robot.solve_ik` previews the stroke's start and end
   for each. The jaws close, the hand rises to `transit_clearance` (20 mm)
   over the tile's top if it is lower, goes over the stroke's start and comes
   straight down. A descent that ends more than 6 mm high stopped on
   something (`blocked`, lifted back).
5. The push: one straight `robot.go_to_pose_cartesian` along the facing with
   `speed_scale` (0.25, passed only when under 1; a connector that does not
   take it gets the plain move). The hand is read back: a stroke that timed
   out or ended more than 5 mm short before the blade reached the tipping
   travel was held (`blocked`). One that ended inside the arrival tolerance
   but short of the tipping travel goes on to the check (the tile rocks back:
   `no_contact`).
6. The retreat: `retreat` (30 mm) back along the stroke at the same height,
   then straight up. The tile falls away from the hand, so backing off first
   never meets it; upstream goes home from the stroke's end.
7. The check (`verify`, default on): park at `park_pose` (default: where the
   hand started, at least at transit height), read `robot.get_observation`,
   and look at the slot's footprint (widened by 6 mm) in depth, the overhead
   first: a tile stands when its top is within 10 mm of its height and 8 mm of
   the slot. Down is `triggered`, standing is `no_contact`, not seen is
   `unverified`. The same check before the push turns an empty slot into
   `no_contact` without moving.

The whole chain is not judged here; record the overhead clip from
`trigger_time` and hand it to verifying-a-domino-cascade-from-video (its
`trigger_time`, slots and tile size are this skill's).

## Inputs

- `slot` (Se3Pose, x axis the facing) or `slots` + `slot_index` (0);
  `next_slot` (default `slots[slot_index + 1]`).
- `table_z` (float): the table plane, a station calibration.
- `tile_size` (`{height, width, thickness}`, default 48 x 24 x 7.5 mm).
- `contact_fraction` (0.7, accepted 0.35-0.9), `overshoot` (0.008 m),
  `speed_scale` (0.25), `approach_clearance` (0.008 m), `retreat` (0.03 m),
  `transit_clearance` (0.02 m), `close_heading_offset_deg` (90),
  `finger_half_depth` (0.012 m).
- `verify` (true), `camera_name` (""), `park_pose` (default the start pose).
- `reach_tolerance` (0.015 m), `check_reach` (true), `arm_id` (0; passed only
  when nonzero).

## Outputs

`pushed` (the blade reached the tipping travel), `contact_height` (the
fingertips' measured height over the table at the stroke, m), `tip_travel`
and `push_travel` (the blade front past the back face, measured, m),
`stroke_length`, `stroke_time` (wall clock, s), `stroke_speed` (m/s, wall
clock), `stroke_steps` (control steps the connector reported for the stroke,
-1 when it does not), `trigger_time` (wall clock at the stroke's start, for the
video check), `push_direction`, `verified`, `first_standing`, `next_standing`.

## Preconditions and boundaries

- The hand holds nothing (it closes before the push).
- Nothing but the tile is within about 30 mm across the stroke on either side
  of it (the closed blades are 30 mm each along the closing axis).
- What tips is decided by geometry and checked by depth. The arm cannot feel
  a 10 g tile: tipping a 48 x 7.5 mm tile pushed 34 mm up takes about
  `m g t / 2 / c` = 11 mN, four orders of magnitude under what joint effort
  or tracking lag resolve on a YAM (detecting-contact-from-joint-effort's 0.5
  N m effort and 3 mm lag thresholds). Proprioception here only separates a
  held stroke (`blocked`) from a free one.
- A line only cascades at a pitch under the tile's height (planning-a-domino-layout).
- Without `verify` the route is what the hand did; a tile not where the slot
  says reads `triggered`.

## What changed from the source, and why

predicators' `create_push_skill` (used by its domino options as `Push`)
closes the fingers, moves above and behind the domino along its facing
(`approach_distance`, upstream 3 tile depths from its centre) at transport
height (1.5 tile heights), descends to `contact_z_offset` over the tile's
centre (0.55 tile heights), steps IK straight to the tile's pose (optionally
past it and at a speed clamp), retreats home and opens the fingers, all
against PyBullet's state. Here:

- the simulator pose is a planned slot (world frame, x axis the facing);
- the contact height is the fingertips' over the table, as a fraction of the
  tile's height, and the stroke ends a computed distance past the back face
  (the tipping travel plus an overshoot) instead of at the tile's centre;
  upstream's end point would carry a 24 mm blade 12 mm past the centre and
  the next tile is then 26 mm away at a 37 mm pitch;
- the approach distance is a clearance behind the back face measured from the
  blade front (upstream's 22.5 mm from the centre would put a 24 mm blade
  over the tile);
- the speed clamp is the connector's `speed_scale`;
- BiRRT's collision-free transit becomes a rise, a move over the start and a
  vertical descent with IK previews; the stroke stays a straight move, as
  upstream insists (a planner routed around the tile and struck it backwards);
- the retreat backs off along the stroke and rises instead of going home
  across the falling chain;
- "toppled" is a depth check of the slot, and the routes are named exits.

## The constants, and why

- **`CONTACT_FRACTION = 0.7`**: upper half (the task); 0.35, 0.45, 0.7, 0.85
  and 0.9 all toppled the tile in the twin (below). **`CONTACT_BAND =
  (0.35, 0.9)`**, **`MIN_OVERLAP = 0.004` m**: low enough to stay off the
  table and above sliding (a tile on a table of friction mu tips when pushed
  higher than t / (2 mu), 7.5 mm at 0.5); high enough that a 4 mm height error
  still meets the tile.
- **`OVERSHOOT = 0.008` m**: the twin's strokes ended 1.4-2.8 mm short of
  their target in free air; with no overshoot the blade reached 2.9-3.0 mm of
  the 5.3 mm and the tile rocked back; 4 mm (blade 7.0 mm) toppled it.
- **`STROKE_TOLERANCE = 0.005` m**: a servo's 4 mm arrival ball plus 1 mm.
- **`APPROACH_CLEARANCE = 0.008` m**, **`NEXT_CLEARANCE = 0.006` m**,
  **`RETREAT = 0.03` m**, **`TRANSIT_CLEARANCE = 0.02` m** (upstream's
  transport height is 24 mm over the tile): choices for clearance.
- **`FINGER_HALF_DEPTH = 0.012` m**: placing-an-upright-domino-beside-neighbours'
  (not measured); the overshoot covers a few mm of error in it.
- **`SPEED_SCALE = 0.25`**: a slow stroke; in the twin a 21 mm stroke ran at
  0.027 m/s at 0.1, about 0.05 m/s at 0.25 and 0.067 m/s at 0.5 and 1.0 (the
  stroke is too short to reach the servo's cruise speed).
- **`TIP_REACH = 0.0102` m**, **`MIN_TOOL_HEIGHT = 0.015` m**: the
  linear_4310's fingertips under the tool centre; a real YAM station refuses
  a tool centre under 14 mm over its table.
- The depth judge's constants (10 mm / 8 mm / 6 mm / 15 mm / a quarter) are
  placing-an-upright-domino-beside-neighbours'.

## Simulation check

A YAM station twin (RoboSimStudio, left arm, linear_4310 hand, MuJoCo at
500 Hz, control at 50 Hz; simulator ground truth only to set up and to
measure). Three parallel lines 80 mm apart of standard dominoes (48 x 24 x
7.5 mm boxes, 10 g, friction 0.5) stood facing +y at a 37 mm pitch (30 mm in
the last run); each line was triggered once with its own contact height,
speed and overshoot, the skill given line k's first and second slots from
the layout, `park_pose` beside the arm's base and the calibrated overhead
Orbbec (640 x 480) for the check. After each skill run the scene ran 3 s and
every tile's tilt was read (fell: over 45 deg). Then the skill was asked to
push an empty slot.

| line | contact_fraction | speed_scale | overshoot | fingertips (measured) | blade past back face / tipping travel | stroke | route | tiles fell |
|---|---|---|---|---|---|---|---|---|
| 8 tiles, 37 mm | 0.45 | 0.25 | 8 mm | 23.4 mm | 8.3 / 3.4 mm | 19.4 mm in 0.38 s (0.051 m/s) | triggered | 8/8 |
| 8 tiles, 37 mm | 0.70 | 0.25 | 8 mm | 35.4 mm | 11.1 / 5.3 mm | 21.3 mm in 0.44 s (0.048 m/s) | triggered | 8/8 |
| 8 tiles, 37 mm | 0.85 | 0.25 | 8 mm | 43.1 mm | 11.9 / 6.4 mm | 22.4 mm in 0.44 s (0.051 m/s) | triggered | 8/8 |
| 10 tiles, 37 mm | 0.70 | 0.10 | 8 mm | 35.7 mm | 11.0 / 5.3 mm | 21.3 mm in 0.80 s (0.027 m/s) | triggered | 10/10 |
| 10 tiles, 37 mm | 0.70 | 0.50 | 8 mm | 35.4 mm | 10.5 / 5.3 mm | 21.3 mm in 0.32 s (0.067 m/s) | triggered | 10/10 |
| 10 tiles, 37 mm | 0.70 | 1.00 | 8 mm | 35.2 mm | 11.7 / 5.3 mm | 21.3 mm in 0.32 s (0.067 m/s) | triggered | 10/10 |
| 10 tiles, 37 mm | 0.35 | 0.25 | 8 mm | 19.0 mm | 7.8 / 2.7 mm | 18.7 mm in 0.38 s (0.049 m/s) | triggered | 10/10 |
| 6 tiles, 37 mm | 0.90 | 0.25 | 8 mm | 45.5 mm | 11.9 / 6.8 mm | 22.8 mm in 0.44 s (0.052 m/s) | triggered | 6/6 |
| 6 tiles, 37 mm | 0.70 | 0.25 | 4 mm | 35.2 mm | 7.0 / 5.3 mm | 17.3 mm in 0.38 s (0.046 m/s) | triggered | 6/6 |
| 6 tiles, 37 mm | 0.70 | 0.25 | 0 mm | 35.4 mm | 2.9 / 5.3 mm | 13.3 mm in 0.32 s (0.042 m/s) | no_contact (rocked back) | 0/6 |
| 8 tiles, 30 mm | 0.70 | 0.25 | 8 mm | 35.7 mm | 11.0 / 5.3 mm | 21.3 mm in 0.44 s (0.048 m/s) | triggered | 8/8 |
| 8 tiles, 30 mm | 0.45 | 0.10 | 8 mm | 23.2 mm | 9.2 / 3.4 mm | 19.4 mm in 0.74 s (0.026 m/s) | triggered | 8/8 |
| 8 tiles, 30 mm | 0.85 | 1.00 | 8 mm | 43.1 mm | 12.7 / 6.4 mm | 22.4 mm in 0.32 s (0.070 m/s) | triggered | 8/8 |

- Every push that carried the blade past the tipping travel toppled the
  first tile and the whole line fell (12 lines, 100 tiles; at rest 78-79 deg
  at 37 mm and 75-76 deg at 30 mm, the last one flat); every one that did not left the tile standing (tilt 0.0 deg after
  3 s). The tipping-travel model drew the line in the right place: 2.9 mm
  stood, 7.0 mm fell, against 5.3 mm.
- The depth verdict agreed with the measured state every time it judged
  (the fallen slot read a 9 mm top 5-6 mm off the slot; the standing one
  48 mm), and `next_standing` was right in every run.
- The servo arrived 1.6-2.0 mm over every commanded contact height and ended
  its strokes 1.4-2.8 mm short of their end: hence the overshoot.
- No push disturbed a neighbouring line (their tilts stayed 0.0 deg until
  their own push), and the hand never reached the second tile (at a 37 mm
  pitch the stroke ends about 24 mm short of its back face).
- The empty slot: `no_contact` before anything moved in four of five runs. In one
  run the hand, left over a `blocked` line by an earlier version of the
  skill, hid the slot from the overhead camera, and the push there (at the
  edge of reach) came back `unreachable` from the IK preview.
- Not exercised in the twin: `blocked` from a tile that does not give (a 10 g
  tile always gave), and `unverified`; both are unit-tested. An earlier
  version routed the 0 mm overshoot run `blocked` (it read a stroke ending
  2.4 mm short inside the servo's ball as held); it now goes to the depth
  check.
- Nothing here ran on a real YAM.

## Provenance

Adapted from predicators (BasisResearch/predicators, MIT, Copyright (c) 2022
Rohan Chitnis and Tom Silver, `LICENSE.predicators`):
`ground_truth_models/skill_factories/push.py` (`create_push_skill`) and
`ground_truth_models/domino/options.py` (`_create_sf_push` and its transport
height, approach and contact offsets). The depth check is
placing-an-upright-domino-beside-neighbours'. predicators is the codebase of
the ExoPredicator paper (Liang et al., ICLR 2026, arXiv 2509.26255).
