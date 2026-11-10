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
    # `descent_tolerance`, `check_reach`, `arm_id`.
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
5. **Into the slot.** It comes down from 3 mm over the walls to 1.5 mm over the floor. A
   descent that ends more than 4 mm higher than the hand's own arrival offset (read on
   the free move just before) stopped on a wall. Offsets of +-1 and +-2 mm along the
   row are then tried; if none goes in, `jammed`, lifted back, still holding.
6. **Release** to the tile's width plus 4 mm a side, straight up.
7. **Check.** The slot is read again. It counts as refilled only when its top reads
   within 4 mm of a tile's height; a tile left on a wall's top reads 34-38 mm and is
   `fell`.

## Measured

**Station twin (GPU, rshgpu2/6; harness `~/sim-wt/refilling-a-domino-loading-rack`).**
- **Setup:** a 6-slot rack (24 mm pitch, 20 mm walls) holding tiles in slots 0 and 3.
  Five standing tiles at a staging row facing 0 deg (the rack faces 90 deg, so every
  refill turns the wrist 90 deg). Five rounds: four empty slots, then the rack is full.
- **Truth:** each tile's box, with a slot counting as full when a tile stands upright
  within 4 mm along the row and 10 mm across.

| rack slot opening | seated / tried | routes that match truth | last round |
|---|---|---|---|
| 11.5 mm (2 mm a side) | 3 / 4 (slots 1, 2, 4) | 5 / 5 | `rack_full`, "jammed: [5]" |
| 9.5 mm (1 mm a side) | 2 / 4 (slots 1, 2) | 5 / 5 | `rack_full` (the failed tiles lie on the walls) |

- **Seated tiles** sat 0.1-1.2 mm along and 0.1-1.1 mm across from the slot centre,
  upright. The held tile was aligned to within 0.02-0.27 mm along the row before every
  descent.
- **Every failure ended the same way:** the tile on a wall's top, leaning back 28-52
  deg, reported `fell`. In the 11.5 mm rack the fallen tile also tipped the neighbour in
  slot 4 (11.8 deg).
- **Not explained.** The failure follows the staging spot, not the slot. Slot 5 alone
  refilled from the first staging spot (y 0.14) seated; from the fourth (y 0.29) it fell.
  At the slot the hand was level (0.1 deg), the tile tilted 0.13 deg in the hand, and it
  was aligned to 0.25 mm. Torsional friction on the pads (MuJoCo condim 4, 0.005) changed
  nothing. The one difference recorded is the direction of the 90 deg wrist turn (+95
  seated, -85 fell), but in the 4-round runs a -68 deg turn seated.

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
- **`ENTRY_CLEARANCE` 3 mm, `RELEASE_GAP` 1.5 mm, `RELEASE_CLEARANCE` 4 mm**: the placing
  skill's release.
- The rest are the rack picker's and the placing skill's.

## Not done

- A third to half of refills fail in the twin, for a reason not found (above). A rack
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
