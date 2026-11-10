---
name: refilling-a-domino-loading-rack
description: Puts a standing domino back into an empty slot of a slotted loading rack, so
  picking-upright-dominoes-from-a-loading-rack always has tiles to take. It reads every
  slot from the overhead depth (full, empty, jammed) and chooses an empty one whose
  neighbours leave room for the open fingers. It picks the tile top-down across its width
  (or takes the tile already in the hand) at a height that keeps the fingertips over the
  rack's walls, and squares the jaws to the slot. It lowers the tile from just over the
  walls into the slot, trying small offsets along the row when the descent stops on a
  wall, releases it, and reads the slot again. Use when one arm, or a person, keeps the
  rack filled while the other arm builds.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [domino, place, rack, fixture, supply, insertion, depth-check]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/BasisResearch/predicators/blob/afb6f2918091f5ca372a75b9a335aea317a8a209/predicators/ground_truth_models/skill_factories/place.py
      upstream_license: MIT (the pick and release, via placing-an-upright-domino-beside-neighbours and picking-upright-dominoes-from-a-loading-rack)
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
    # OPTIONAL extra inputs (wired from a script node in the same subgraph): `rack` (as
    # picking-upright-dominoes-from-a-loading-rack takes it), `tile` (Se3Pose of a standing
    # tile to pick, x axis its facing, or None for the tile in the hand), `grasp_height`
    # (the held tile's hang with tile None, else the commanded grasp), `slot_index` (-1:
    # choose), `order` (index | reverse | nearest), `tile_size`, `wall_height`, `offsets`,
    # `grip_clearance`, `release_clearance`, `transit_clearance`, `read_slots`, `verify`,
    # `camera_name`, `look_pose`, `yaw_tolerance_deg`, `roll_limits`, `reach_tolerance`,
    # `descent_tolerance`, `check_reach`, `arm_id`, `seat_press`, `held_speed_scale`.
    table_z: float
  produces_outputs:
    slot_index: int
    picked: bool
    released: bool
    grasp_height: float
    grip_width: float
    jaw_error_deg: float
    wrist_turn_deg: float
    insert_offset_mm: float
    align_error_mm: float
    tile_tilt_deg: float
    attempts: int
    slot_state_after: str
    empty_count: int
  exit_conditions:
    refilled: The tile went into slot `slot_index` (at `insert_offset_mm` along the row, after `attempts` descents), was released and the hand rose, and the slot reads full from depth (with verify false, what the hand did only).
    rack_full: No slot reads empty (or the requested `slot_index` is not empty); `detail` lists jammed slots; nothing moved.
    jammed: Every offset's descent into the slot stopped more than 3 mm high on a wall; the hand lifted back to transit, still holding the tile.
    fell: Released, but the slot reads empty or jammed after (the tile tipped out, or lies across the slot).
    unverified: Released, but no camera saw the slot whole after the release (give look_pose).
    not_holding: The pick closed on nothing or not across the tile, its descent stopped on the tile, the tile was dropped in transit, or (tile None) the hand holds nothing.
    hand_full: A tile to pick was given but the jaws already read a tile's width; nothing moved.
    not_seen: No camera saw the rack, or no empty slot among those seen while some were not.
    unreachable: Nothing was put in - every empty slot has a standing neighbour where an open finger goes, no grasp height clears the walls, an IK preview or a move failed, or the jaws stayed turned; `detail` says whether the tile is still held.
    invalid_input: A rack without slots, a wall as high as the tile, an unknown order, a slot_index past the rack, or no tile and no grasp_height (also the route of a raised error).
  canonical_scripts:
    - refill_rack: scripts/refill_rack.py
  examples:
    - title: Put a standing domino into the next empty rack slot
      path: examples/canonical_subgraph.json
---

# refilling-a-domino-loading-rack

The supply side of the rack: picking-upright-dominoes-from-a-loading-rack takes tiles
out, this puts them back in, so one arm can keep the rack filled while the other builds.

## The motion

1. **Slots** from the overhead depth with the rack picker's judge (full, empty, jammed).
   `look_pose` lets it read again out of the camera's view. The empty slot is chosen
   by `order`, skipping one whose standing neighbours sit where an open finger goes.
2. **The hand.** Picking a tile needs an empty hand (`hand_full` otherwise). Placing a
   held tile (`tile` None, its hang in `grasp_height`) needs a tile's width in the jaws.
3. **Pick** (the placing skill's): jaws across the tile's width. The grasp height is in
   the rack's band, so the fingertips stay at least 2 mm over the walls once the tile
   is seated. A tile held lower than that is released from just high enough instead.
   The tile's place in the hand is recorded from the grasp's read-back.
4. **Turn and align.** Over the slot the jaws are squared to it on the last joint, at
   most 15 deg a move while holding. The held tile (its offset in the grip re-expressed
   with the hand's rotation, and any tilt) is brought within 0.5 mm of the slot along
   the row, with up to 3 corrective moves read back.
5. **Into the slot.** It comes down from 3 mm over the walls to 1 mm into the floor
   (`seat_press`). A
   descent that ends more than 4 mm higher than the hand's own arrival offset (read on
   the free move just before) stopped on a wall. Offsets of +-1 and +-2 mm along the
   row are then tried; if none goes in, `jammed`, lifted back, still holding.
6. **Release** to the tile's width plus 4 mm a side, straight up.
7. **Check.** The slot is read again. It counts as refilled only when its top reads
   within 4 mm of a tile's height; a tile left on a wall's top reads 34-38 mm and is
   `fell`.

## Measured

**Station twin (GPU, rshgpu2/6; harness `~/sim-wt/refilling-a-domino-loading-rack`), with
the fixed grip (`SIM_GRIP`, `mujoco_500hz_grip@v1`).**
- **Setup:** a 6-slot rack (24 mm pitch, 20 mm walls) holding tiles in slots 0 and 3,
  and five standing tiles at a staging row facing 0 deg. The rack faces 90 deg, so
  every refill turns the wrist 90 deg. Five rounds.
- **Truth:** each tile's box; a slot is full when a tile stands upright in it within
  4 mm along the row and 10 mm across.

| rack slot opening | seated / tried | routes that match truth | then |
|---|---|---|---|
| 11.5 mm (2 mm a side) | 2 / 3 (slots 4, 5) | 5 / 5 | `rack_full`, "jammed: [1, 2]" |
| 9.5 mm (1 mm a side) | 2 / 3 (slots 4, 5) | 5 / 5 | `rack_full`, "jammed: [1, 2]" |

- **Seated tiles** sat 0.0-0.8 mm along and 0.2-0.4 mm across from the slot centre,
  upright. The held tile was aligned within 0.01-0.25 mm along the row before each
  descent.
- **The one failure, the first refill (slot 1) in both racks:** the tile ended flat
  forward across slot 2, 20 mm along. It is reported `fell`, and slots 1 and 2 then
  read jammed, so the rack reads full: the routes are right, but two slots are lost.
- **Fix: slow moves while holding** (`held_speed_scale`, default 0.25: the lift and the
  move over the slot as straight moves at that speed). Twin with the fixed grip, the
  same 5-round scene:

  | slot opening | held speed | seated / tried | slot 1 |
  |---|---|---|---|
  | 11.5 mm | 1.0 (before) | 2 / 3 | fell |
  | 11.5 mm | 0.25 (default) | **4 / 4** | seated |
  | 11.5 mm | 0.1 | **4 / 4** | seated |
  | 9.5 mm | 0.25 | 2 / 3 | fell |

  Seated tiles sat 0.0-0.6 mm along the row. The 9.5 mm slot (1 mm a side) still loses
  the first refill; re-measuring the tile's bottom with the wrist camera before the
  descent is not done.
- **Root cause of the slot-1 failure (measured, round 3).** The harness probes the held
  tile's true pose with `sim.get_object_obb` at each phase.
  - **Before and after the close** the tile sat centred in the jaws: 0.25 / 0.11 mm
    from the tool centre.
  - **Lift:** by the end of the vertical lift its bottom had moved **4.6 mm** from the
    hand, along the pad faces (perpendicular to the jaw axis). Over the slot that
    offset was back to 2.2 mm.
  - **Not a twist.** The tile tilted at most 2.4 deg in all (about 1.7 deg about the
    jaw axis, worth 0.6 mm at its bottom); it is translational slip under the moves'
    acceleration. SIM_GRIP stops steady creep, not this.
  - **Consequence:** the skill aims with the offset read back at the grasp, which is
    stale by then. The tile's bottom came down 2.1-2.8 mm off along the row, more than
    an 11.5 mm slot's 2 mm a side. It landed on the wall's top, slid 16 mm up between
    the pads while the hand still reached its target, and dropped flat across the next
    slot on release.
  - In the refills that seated, the drift was smaller (bottom 0.3-0.5 mm off at the
    descent).
  - Centring the open jaws on the tile before the close (read-back, to within 0.5 mm)
    did not help: the slip happens after the close (still 4.3-4.6 mm after the lift,
    slot 1 failed in both racks).
  - What would fix it is not tested: slower lift and transit moves; re-measuring the
    tile in the hand before the descent (a wrist view of its bottom); chamfered slot
    tops. A real linear_4310 squeezes far harder (SIM_GRIP.md: about 137 N against the
    twin's 19 N), so the drift may be a twin artefact.
- **Older note.** On the stock grip (before SIM_GRIP) the same scene seated 3/4
  (11.5 mm) and 2/4 (9.5 mm). There every failure was a tile on a wall's top leaning
  back, and it followed the staging spot (the same slot seated from one spot and fell
  from another), not the hand's tilt (0.1 deg), the tile's tilt in the hand (0.13 deg)
  or alignment (0.25 mm).
- **Pressing the tile 1 mm into the floor before the release** (`SEAT_PRESS`): with the
  fixed grip, the 9.5 mm rack went from 1/3 to 2/3 seated, and the 11.5 mm rack stayed
  2/3.

**Bugs found in the twin, fixed and tested:**
1. A single 90 deg turn of the last joint while holding left the tile where it was; the
   jaws ended across its thickness (9.3 mm), 4 of 5 rounds. Now 15 deg steps (0 of 9
   since).
2. The hand arrives 2.2-3.1 mm over every command, so "stopped 3 mm high" read every
   seated tile as jammed, and the offset retries wedged it out. Now judged against the
   hand's own arrival offset.
3. A tile that meets a wall slides up between the pads while the hand reaches its
   target, so the hand's height cannot see it. The tile is now aligned before the
   descent.
4. The check after release read a tile leaning on a wall (top 38 mm) as full. Now
   within 4 mm of a tile's height.

## The constants, and why

- **`HELD_TURN_STEP_DEG` 15**: see bug 1.
- **`INSERT_TOLERANCE` 4 mm**: a tile on a 20 mm wall stops 18.5 mm high; the hand's own
  offset is taken out first (bug 2).
- **`ALIGN_TOLERANCE` 0.5 mm, `ALIGN_TRIES` 3**: half of a 9.5 mm slot's 1 mm a side.
- **`OFFSETS` 0, +-1, +-2 mm**: the twin hand landed 1-4 mm off before alignment.
- **`SEATED_TOP` 4 mm**: bug 4.
- **`SEAT_PRESS` 1 mm**: the fixed-grip twin lifts parts tilted 4-6.5 deg about the jaw
  axis; pressed on the floor they square up (above).
- **`ENTRY_CLEARANCE` 3 mm, `RELEASE_CLEARANCE` 4 mm**: the placing skill's release.
- The rest are the rack picker's and the placing skill's.

## Not done

- One refill in three fails in the twin, for a reason not found (above). A rack
  with chamfered slot tops would guide a tile in; this skill does not need one, but the
  twin results argue for it.
- No real rack. The hand's arrival offset and the pinch slip are the twin's; a real
  YAM's will differ.
- Clearing a jammed slot (a tile left on the walls) is not part of this skill.

## Provenance

The rack geometry, slot judge, jaw read-back and pick come from
picking-upright-dominoes-from-a-loading-rack and
placing-an-upright-domino-beside-neighbours. They adapt predicators (MIT, commit
afb6f29: `skill_factories/pick.py`, `place.py`); the licence is `LICENSE.predicators`
beside this file. New here:
- the empty slot chosen from depth;
- holding over the walls;
- turning a held tile in steps;
- aligning the held tile from its place in the grip;
- the descent judged against the hand's own offset, with offset retries;
- the seated check.
