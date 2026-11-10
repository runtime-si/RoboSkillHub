---
name: picking-upright-dominoes-from-a-loading-rack
description: Takes the next domino out of a slotted loading rack, where tiles stand one
  per slot at a known pitch, instead of standing up each tile from lying. It reads every
  slot from the overhead depth as full, empty (only the walls' tops) or jammed (a top that
  is neither) and chooses the full slot nearest the hand. It picks top-down with the jaws
  across the tile's width at a height that keeps the fingertips over the rack's walls,
  lifts straight up until the tile clears them, and reads the jaw gap; a lift held short
  with the tile in the jaws is a jammed tile. Returns the held tile's grasp height for
  placing-an-upright-domino-beside-neighbours (tile None). Use when a domino build is
  fed from a rack filled by hand or by the other arm, as its supply step.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [domino, pick, rack, fixture, supply, upright, depth-check]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/BasisResearch/predicators/blob/afb6f2918091f5ca372a75b9a335aea317a8a209/predicators/ground_truth_models/skill_factories/pick.py
      upstream_license: MIT (the pick, via placing-an-upright-domino-beside-neighbours); the rack is reimplemented from a project description with no code
      paper: https://martinpeticco.com/projects/5_domino_robot/
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
    - robot.close_gripper
    - robot.set_grip
  required_inputs:
    # OPTIONAL extra inputs (not listed: gap has no list or open-dict type here, so they
    # are wired from a script node in the same subgraph): `rack` ({origin [x, y, floor z],
    # facing_deg, pitch, n_slots, row_deg, wall_height} or {slots: [poses at the tiles'
    # bottom centres], wall_height}), `slot_index` (-1: choose), `occupied` (bookkeeping
    # for slots no camera sees), `order` (nearest | index | reverse), `tile_size`,
    # `wall_height`, `grasp_height`, `grip_clearance`, `transit_clearance`,
    # `lift_clearance`, `read_slots`, `camera_name`, `yaw_tolerance_deg`, `roll_limits`,
    # `reach_tolerance`, `descent_tolerance`, `check_reach`, `arm_id`.
    table_z: float
  produces_outputs:
    slot_index: int
    picked: bool
    grasp_height: float
    grip_width: float
    jaw_error_deg: float
    wrist_turn_deg: float
    tile: Se3Pose
    full_count: int
    remaining: int
    lift_shortfall: float
  exit_conditions:
    picked: A tile was taken from `slot_index`, lifted straight out and carried to transit height with the jaws reading its width; `grasp_height` is the measured tool-centre height over its bottom (pass it to placing-an-upright-domino-beside-neighbours with tile None), `tile` where it stood, `slot_states` every slot as read, `remaining` the full slots left.
    slot_empty: The requested `slot_index` is not full (empty or jammed); nothing moved.
    rack_empty: No slot reads full; `detail` lists the jammed and unseen ones (clear a jam by hand or with another skill).
    grasp_failed: The descent into the slot stopped high (a fingertip on the tile's top or a wall), the jaws closed on nothing or not across the tile, or the tile slipped out during the lift or the rise; the hand is open over the rack.
    stuck: The straight lift stopped more than 6 mm short with the tile still in the jaws (it is jammed in its slot); the jaws reopened and the hand rose.
    hand_full: The jaws already read a tile's width before anything moved (a tile from an earlier step still in the hand); nothing moved - place it or set it down first.
    not_seen: No full slot was seen and some slots were not (no camera saw them whole and unoccluded, also after `look_pose`), and no `occupied` list covers them.
    unreachable: Nothing was picked - every full slot has a standing neighbour where an open finger goes (a rack pitch under 19.8 mm for a 7.5 mm tile), no grasp height clears the walls and keeps 10 mm of tile above the fingertips, the grasp failed the IK preview for both wrist solutions, a move was refused, or the jaws stayed turned off the tile; also a refused rise after the pick (still holding).
    invalid_input: A rack without slots, a pitch or slot count that is not positive, a wall as high as the tile, an unknown order, an `occupied` list of the wrong length or a `slot_index` past the rack.
  canonical_scripts:
    - pick_from_rack: scripts/pick_from_rack.py
  examples:
    - title: Take the next domino out of a loading rack
      path: examples/canonical_subgraph.json
---

# picking-upright-dominoes-from-a-loading-rack

Standing a lying tile up is the slowest and least reliable step of a domino build
(standing-a-lying-domino-by-reorient-and-regrasp). A slotted rack filled by hand, or by
the other arm, removes that step: every tile already stands, at a known pitch, and the
arm only has to lift it out.

## The motion

1. **Rack.** Either `origin` (slot 0's centre at the rack's floor, where the tiles'
   bottoms are), `facing_deg` (the tiles' broad-face normal), `pitch`, `n_slots` and
   `row_deg` (the way the slots advance; default the facing), or `slots` (a pose per
   slot). `wall_height` is how far the rack's walls rise over its floor (20 mm by
   default). Calibrate the origin once, for example with a fiducial on the rack.
2. **The hand must be empty.** A jaw gap of a tile's width (its width less 6 mm up to
   its width plus 5 mm) is a tile still in the hand, and routes `hand_full` without
   moving. Lesson 5: never open on something held.
3. **Slots from depth.** This uses the placing skill's judge on the overhead camera:
   the highest points in each slot's footprint (widened 6 mm). A tile's height within
   10 mm and its top within 8 mm of the slot is **full**. Only the walls' tops (at
   most 5 mm over them) is **empty**. Anything else is **jammed**: a tile leaning,
   lying across the slot, or half out. A slot no camera sees whole and unoccluded is
   unseen. With `look_pose` the hand goes there (out of the camera's view) and the
   rack is read again. `occupied` fills in what stays unseen.
4. **Choice.** `slot_index`, or the full slot nearest the hand (`order`: `nearest`,
   `index`, `reverse`). A slot is skipped when another standing tile in the rack lies
   where an open finger goes (the placing skill's `finger_conflicts`: jaws across
   the width, 24 mm along the row). That makes the rack's pitch 19.8 mm or more for a
   7.5 mm tile, or every slot is `unreachable`.
5. **Grasp height.** The fingertips must close at least 2 mm over the walls and leave
   10 mm of tile above them, inside the placing skill's measured band (15-46 mm
   tool-centre height over the tile's bottom). Walls of 20 mm give 32.2-46 mm and
   25 mm walls give 37.2-46 mm; above 33.8 mm no grasp exists. The default is the
   lowest height in the band, or 30 mm if that is higher.
6. **Pick** (predicators' pick through the placing skill). It previews both wrist
   solutions with `robot.solve_ik` and pre-opens to the tile's width plus 4 mm a
   side. It rises to transit (20 mm over every tile top in the rack), goes over the
   slot and reads the jaw heading back, finishing the turn on the last joint. Then it
   comes straight down and closes. A descent that ends 6 mm high, or a jaw gap that
   is empty, narrower or wider than the tile, is `grasp_failed`.
7. **Lift straight out** until the tile's bottom is 10 mm over the walls, then read
   back. A lift that ended more than 6 mm short with the tile in the jaws is `stuck`:
   the jaws reopen and the hand rises. A lift with empty jaws is `grasp_failed`.
   Then it rises to transit and reads the jaws once more.
8. Hand the tile to placing-an-upright-domino-beside-neighbours with `tile: None`
   and this skill's `grasp_height`, the measured tool-centre height over the tile's
   bottom.

## Measured

**Station twin (GPU, rshgpu2)** with harness `~/sim-wt/picking-upright-dominoes-from-a-loading-rack`.
The left arm worked a rack of fixed blocks, 9.5 mm slots and 40 mm deep, of 48 x 24 x
7.5 mm tiles at 10 g with friction 0.5. Each round ran this skill and then the merged
placing skill (`tile: None`, this `grasp_height`) to the next slot of a line, with
ground truth from `sim.get_object_obb` after each round. `look_pose` and the placing
skill's `park_pose` were the arm's start pose.

| rack | fill | rounds | picked / full | placed standing | last round | slot reads = truth |
|---|---|---|---|---|---|---|
| A: 24 mm pitch, 20 mm walls, 6 slots | full, full, empty, full, a tile lying across slot 4, full | 5 | 4 / 4 | 4 / 4 (1.0-2.7 mm off the slot) | `rack_empty`, "jammed: [4]" | 30 / 30 |
| B: 20 mm pitch, 25 mm walls, 6 slots | all full | 7 | 6 / 6 | 6 / 6 (1.4-4.2 mm off) | `rack_empty` | 42 / 42 |
| C: 18 mm pitch, 20 mm walls, 5 slots | all full | 2 | 0 | - | `unreachable` (open finger on a neighbour), nothing moved | 10 / 10 |

- **Re-run with the fixed grip** (SIM_GRIP, `mujoco_500hz_grip@v1`, impratio 100; rshgpu4,
  2026-11-10) gave the same outcomes:
  - A: 4 / 4 picked and stood (placement 0.6-3.3 mm off), last round `rack_empty`
    "jammed: [4]";
  - B: 6 / 6 picked and stood (1.3-4.9 mm off), then `rack_empty`;
  - C: `unreachable`, nothing moved;
  - slot reads 82 / 82; jaws held 27.3-27.8 mm; the lift ended 1.7-2.2 mm short of its
    command, inside the 6 mm `stuck` bound.
- The order was nearest first: A took slots 5, 3, 1, 0; B took 5 down to 0. No
  tile left in the rack moved (every later read matched the truth).
- Grasp height: commanded 32.2 mm (A) and 37.2 mm (B); the hand arrived 2.2-2.4 mm
  high (34.4-34.6 and 39.4-39.6 mm), as the placing skill measured. The jaws held
  26.8-27.0 mm on the 24 mm tile. The lift ended 1.8-2.3 mm short of its command,
  inside the 6 mm `stuck` bound. The jaw heading was within 3.4 deg after the
  read-back; the first pick of each run turned the wrist 81-85 deg on the last joint
  (the twin's solver leaves the roll free).
- Wall clock was 3.5-6.8 s a pick in the twin, including its simulation.
- A first run found two bugs, both fixed and covered by tests. (1) The placing
  skill's failed place left a tile in the hand, and the next pick opened the jaws
  over the rack and dropped it; this is now `hand_full`. (2) Slots hidden by the
  parked arm made "no full slot seen" read as `rack_empty`; this is now `not_seen`,
  plus `look_pose`.

Not run in the twin: `stuck` and `grasp_failed` (no jam or slip happened there). They
are covered by the unit tests with a fake arm. On a real rack, a tight slot's friction
is what `stuck` is for.

## The constants, and why

- **`WALL_HEIGHT` 20 mm**: holds 42 % of a 48 mm tile. The candidate's 25 mm also
  worked (B). The ceiling is 33.8 mm for the linear_4310 (`grasp_band`).
- **`GRASP_BAND`, `MIN_BITE`, `TIP_REACH`, `GRIP_CLEARANCE`, the jaw-gap margins,
  `TRANSIT_CLEARANCE`, `DESCENT_TOLERANCE`, `REACH_TOLERANCE`, the depth judge's
  tolerances**: the placing skill's, with its measurements.
- **`WALL_CLEARANCE` 2 mm**: fingertips over the walls, under a servo's 4 mm
  arrival bound. The twin hand arrived 2.2-2.4 mm high, adding to it.
- **`LIFT_CLEARANCE` 10 mm**: the tile's bottom over the walls before any sideways
  move, more than a 7.5 mm tile can tilt into at that height.
- **`EMPTY_TOP` 5 mm**: a slot reading at most the walls' tops plus 5 mm is empty. A
  tile lying across a 20 mm wall reads 27.5 mm (jammed).
- **`HELD_MARGIN` 5 mm**: a held 24 mm tile read 25.4-27.0 mm in the twin; the
  placing skill's release leaves 32 mm.

## Not done

- No real rack was built or calibrated; the rack origin is an input.
- The rack is modelled as blocks between slots. Side walls along the tiles' width
  would meet the fingers below their top and are not checked (keep the rack open on
  the width sides, or the walls under the fingertips).
- Refilling the rack (by hand or by the other arm, holding-steady-while-other-arm-acts)
  is not part of this skill.

## Provenance

The pick, the jaw read-back, the jaw-gap tests and the depth judge are
placing-an-upright-domino-beside-neighbours', which adapts predicators (MIT, commit
afb6f29: `skill_factories/pick.py`, `domino_component.py`). Its licence is
`LICENSE.predicators` beside this file. The rack is reimplemented from the
description of the Caltech ME134 Team Domino dispenser (martinpeticco.com, no code
published). New: the slot states from depth, the choice, the grasp band over the
walls, the straight lift judged from the hand's height, and the hand-empty
precondition.
