---
name: rebuilding-a-fallen-domino-segment
description: Plans the rebuild of a fallen segment in a domino line, one tile at a time.
  From auditing-a-domino-line-for-fallen-or-tilted-tiles' per-slot states and leans it
  finds the first run of fallen or leaning tiles and its fall direction, and picks its
  downstream end (nothing lies on it, and once it is taken the next one drops flat). It
  returns that tile's lying centre and long axis for standing-a-lying-domino-by-reorient-and-regrasp
  and its slot for placing-an-upright-domino-beside-neighbours. Run it again after each
  tile (audit, plan, stand, place) until it routes nothing_fallen. No motion. Use when a
  line fell part-way (a premature or stopped cascade, a bump) and must be rebuilt.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [domino, recovery, rebuild, fallen, plan]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/Zhu-Chenyu/Franka-Domino-Project
      upstream_license: none (reference only; reimplemented as a composition of hub skills, no code copied)
gap:
  allowed_tools: []
  required_inputs:
    # OPTIONAL extra inputs (wired from a script node in the same subgraph): `slots` (the
    # line's planned slots), `states`, `lean_deg`, `shift_mm` (the audit's outputs),
    # `tile_size`, `flat_deg`.
    table_z: float
  produces_outputs:
    tile_index: int
    tile_center: Vec3
    long_axis: Vec3
    slot: Se3Pose
    direction: str
    remaining: int
  exit_conditions:
    rebuild_next: The downstream end of the first fallen segment lies flat (80 deg or more); `tile_center` / `long_axis` are where it lies (for the stand-up skill), `slot` where it goes back (for the placing skill), `segment` the run, `remaining` the fallen and leaning tiles left including it.
    nothing_fallen: No tile is fallen or leaning; `detail` lists missing slots (place new tiles there).
    not_flat: The downstream tile still leans (under 80 deg) on something; take that away or lay it down first.
    invalid_input: No slots, or slots, states and leans of different lengths (also the route of a raised error).
  canonical_scripts:
    - plan_rebuild: scripts/plan_rebuild.py
  examples:
    - title: Choose the next fallen tile to rebuild
      path: examples/canonical_subgraph.json
---

# rebuilding-a-fallen-domino-segment

The planning step of rebuilding a line that fell part-way. The loop is: audit, this
skill, stand the tile up, place it at its slot, audit again, until `nothing_fallen`.

## What it does

1. **Segment:** the first run of `fallen` / `leaning` states in line order. Its direction
   comes from the majority sign of the leans (forward, the usual cascade, or backward).
2. **The downstream end** is the last tile of a forward run (the first of a backward
   one). Nothing lies on it, and once it is lifted away the tile that leaned on it drops
   flat.
3. **Lying pose.** A tile fallen flat about its front bottom edge has its centre h/2 +
   d/2 beyond the slot centre (plus the audit's shift), d/2 over the table, with its long
   axis along the facing. This is what standing-a-lying-domino-by-reorient-and-regrasp
   takes.
4. **Body across an empty slot:** when the audit reads that tile short of flat and puts a
   flat body on the empty slot after it, the body is taken as this tile. The audit's
   window ends at that slot; in the station twin it read the tile at 67 deg or unclear,
   and the body at 90 deg on the empty slot.

## Measured

**Station twin (fixed grip; GPU rshgpu2/3; harness `~/sim-wt/rebuilding-a-fallen-domino-segment`).**
- **Setup:** a 30 mm line of 8 slots with slot 5 empty. One tile started 11 deg forward,
  so a real stopped cascade left a fallen segment: 2-4 (three tiles, R3) or 3-4 (two
  tiles, R2).
- **Graph:** audit, then plan, then stand-up (handover), then place at the slot, then
  truth from the tiles' boxes.
- **Plan on the real audit:**
  - R3: segment [2, 4] forward, tile 4 lying flat, chosen correctly.
  - R2: segment [3, 4] forward, tile 4, chosen correctly.
  - This needed the empty-slot rule; without it both routed `not_flat`.
- **Stand-up, R3:** the merged stand-up skill picked tile 4, then stopped `unreachable`
  at the handover ("the helper closed to 0.0 mm, not across the 7.5 mm tile"). No tile
  was rebuilt.
- **Stand-up, R2:** result not in at the time of writing.

## Not done

- No tile was rebuilt end to end in the twin yet: the stand-up skill's handover failed
  on the first tile (R3).
- The rack swap (take an upright tile from picking-upright-dominoes-from-a-loading-rack
  and clear the lying one aside) is not wired.
- A downstream tile still leaning on something routes `not_flat`; laying it down is not
  done.
