---
name: auditing-a-domino-line-for-fallen-or-tilted-tiles
description: Checks a standing domino line or curve from one overhead depth frame, with
  no reference image, against its planned slots. For every slot it says whether the tile
  stands upright, is displaced (off its slot or turned), tilted, leaning on its neighbour,
  fallen or missing. It fits each tile's lean angle (forward or backward) jointly with its
  shift along the row, to the depth profile along its fall corridor (the cascade-video
  skill's rotated-box model), and its turn from its top face. A fallen or leaning
  neighbour's body is taken out of a tile's window, and neighbours are checked against
  each other physically. It also lists the runs of fallen tiles and the gaps a wave cannot
  cross. Routes line_ok, fix_needed or fallen. No motion. Use before triggering a line,
  after building or bumping it, or after a stopped cascade, to know which tiles to re-place.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [domino, verify, depth, fallen, tilted, missing, audit, single-frame]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/BasisResearch/predicators/blob/afb6f2918091f5ca372a75b9a335aea317a8a209/predicators/envs/pybullet_domino/components/domino_component.py
      upstream_license: MIT
gap:
  allowed_tools:
    - robot.get_observation
  required_inputs:
    # OPTIONAL extra inputs (gap has no list type, so they are wired from a script node in
    # the same subgraph): `slots` (planning-a-domino-layout's `slots` / `slot_poses`, Se3Pose
    # with the x axis the facing, {x, y, facing_deg}, or [x, y, facing_deg]; path order),
    # `frame` (one CameraFrame with depth, intrinsics, pose; default the observation's
    # cameras, the overhead first), `camera_name`, `tile_size`, `upright_deg`, `fallen_deg`,
    # `displaced`, `yaw_tolerance_deg`.
    table_z: float
  produces_outputs:
    n_tiles: int
    n_upright: int
    first_fault: int
    camera: str
  exit_conditions:
    line_ok: Every tile stands upright on its slot (lean under 5 deg, within 4 mm and 10 deg of it) and no gap between neighbours is wider than a wave crosses; trigger the line.
    fix_needed: No tile has fallen or is missing, but some are displaced, tilted, leaning on a neighbour, or a gap between two standing tiles is over the critical gap (`wide_gaps`); `faults` lists them in line order with lean, shift and direction.
    fallen: At least one tile has fallen (lean 60 deg or more, or past where its top meets the neighbour) or is missing; `fallen_runs` gives the runs of fallen and leaning tiles (a stopped or premature cascade), `faults` every faulty tile.
    not_seen: No camera frame with depth saw the slots (none in the observation, out of the image, or a hand over them), or some slots stayed unjudged while the rest were fine; `states` says `unseen` / `unclear` for those.
    invalid_input: No slots, a slot without a facing, or a tile size that is not height > width > thickness (also the route of a raised error).
  canonical_scripts:
    - audit_line: scripts/audit_line.py
  examples:
    - title: Audit a built domino line before triggering it
      path: examples/canonical_subgraph.json
---

# auditing-a-domino-line-for-fallen-or-tilted-tiles

The question before a trigger push, or after a bump: does every tile still stand where
it was put? One overhead depth frame and the plan answer it, with no "before" image.

## What it does

1. **Slots.** The planned slots, a tile's bottom centre and facing (the way it topples).
2. **Window per tile.** The depth points from its back neighbour's front face to its
   front neighbour's back face (2 mm short of each), 8 mm wider than the tile. A camera
   whose window is a quarter or more higher than a tile (a hand) does not judge it.
3. **Turn.** The long axis of the top face's points gives the tile's turn from its
   slot; the fit then runs in the tile's own frame.
4. **Lean and shift.** This is the cascade-video skill's model: the tile's height
   profile along the row, in 2 mm bins (90th percentile), against a rotated box. The fit
   runs jointly over the lean (0-90 deg in 0.5 deg steps, about the front bottom edge
   forward or the back one backward) and the base's shift along the row (±10 mm, 1 mm
   steps). The model counts as present over a whole bin, so a face seen at an angle on
   a bin edge counts. Over-height costs 0.3 of under-height, because a neighbour may lie
   on the tile. A backward lean is taken only when it fits 0.5 mm better than the
   forward one, since cascades topple forward.
5. **Neighbours out.** A neighbour fitted leaning has its box (its fitted lean ±2 deg,
   grown max(2 mm, 3σ)) taken out of the tile's window. A fallen neighbour (45 deg or
   more) has the whole sector from its fitted angle down to flat taken out. Each tile is
   refitted, twice, in line order.
6. **Noise.** The depth noise σ is the MAD of the table points in the window. Something
   is "raised" over max(4 mm, 3σ); bins are lowered by 1.28σ (the noise's share of a
   90th percentile).
7. **States.**
   - `missing`: fewer than 6 raised points.
   - `upright`: under 5 deg; `displaced` if the base is 4 mm or more off the slot along
     or across the row, or turned more than 10 deg.
   - `fallen`: 60 deg or more, or more than 4 deg past the angle at which its top meets
     the neighbour it leans toward.
   - `leaning`: within 8 deg under that contact angle.
   - `tilted`: anything else.
   - `unclear`: a fit residual over 6 mm.
   - Consistency:
     - A tile "fallen" past an *upright* neighbour cannot be its own (it would rest on
       that neighbour), so the body is the tile behind it and the slot is `missing`.
     - A tile at 45 deg or more next to a fallen tile is fallen.
     - A slot between two fallen tiles whose points are all explained by them, or whose
       fit is unclear, is fallen too: inside a pile its body cannot be told from theirs.
8. **Gaps.** Between two standing tiles, a face gap over 0.996 h (47.8 mm, the planner's
   critical gap) is listed in `wide_gaps`.

## Measured

**Station twin (GPU, rshgpu5; harness `~/sim-wt/auditing-a-domino-line-for-fallen-or-tilted-tiles`).**
Tiles of 48 x 24 x 7.5 mm were placed with faults, settled 2 s, and audited from the
station's calibrated overhead camera (one rendered depth frame). Truth comes from each
tile's box (`sim.get_object_obb`) with the same thresholds.

| scene | line | truth | states right |
|---|---|---|---|
| S1 | 37 mm x 10: a tile set leaning on its neighbour pushed it over, and a cascade ran 2-7; slot 8 empty | 2 upright, 6 fallen, 1 missing, 1 upright | 9 / 10 |
| S2 | 30 mm x 10: tile 0 started past its tipping angle and slot 5 empty, so the wave stopped there; tile 7 5 mm sideways | 5 fallen, 1 missing, 2 upright, 1 displaced, 1 upright | 10 / 10 |
| S3 | 45 mm x 8 control | 8 upright | 8 / 8 |
| S3 | arc r 150 mm x 7: a tile set leaning back pushed a backward cascade over 0-3 | 4 fallen, 3 upright | 7 / 7 |

- **34 / 35 states right; routes right on all 4 lines.** The miss is S1's empty slot 8.
  Tile 7 lies flat across it and was itself read leaning the wrong way (-76 deg, truth
  +90), so the consistency rule could not give slot 8 to it.
- Upright tiles read 0-2 deg of lean (the camera's oblique view), every one under the 5
  deg band. The 5 mm sideways tile read 6.1 mm. Turns of upright tiles read within 3.4
  deg of 0.
- **The direction of a fallen tile inside a run is not reliable:** a pile of tiles lying
  on each other fits both ways (4 of 13 read backward). The state `fallen` is.
- 0.3-1.0 s a line of 7-10 tiles in the twin.
- The twin could not hold a tile leaning on a standing neighbour or tilted alone: every
  one placed so pushed its neighbour over (which made S1 and S3 real partial cascades).
  The `leaning` and `tilted` states are checked on synthetic frames only.

**Random synthetic lines** (`tests/bundles/_domino_audit_eval.py`): 40 lines of 8 tiles
at 30 / 37 / 45 mm. Each tile is upright, displaced 5-8 mm or turned 12-20 deg,
tilted 8-20 deg either way, leaning on its neighbour, missing, or (the last) flat.

| depth noise | tiles right | lines all right | lean error median / max |
|---|---|---|---|
| 0 | 309 / 320 | 31 / 40 | 0.6 / 3.5 deg |
| 1 mm | 303 / 320 | 25 / 40 | 1.1 / 3.3 deg |
| 2 mm | 289 / 320 | 16 / 40 | 1.1 / 3.9 deg |

- Every fallen and leaning tile was right at every noise level.
- The errors:
  - upright read tilted: 2 / 8 / 17 of 170;
  - missing read displaced or upright: 3-6 of 19;
  - a 20 deg tilt at a 30 mm pitch reads `leaning`: its contact angle there is 28 deg,
    so it is inside the 8 deg band (3 of 23).

**Synthetic frames** (`tests/bundles/_domino_depth_scene.py`: an overhead camera over
tiles splatted into a 1 mm heightmap, 37 mm line). One line held a 6 mm displaced tile,
a tile leaning on its neighbour (37.9 deg), an upright tile behind it, a 12 deg tilt, a
15 deg turn, an empty slot and a tile flat at the end.
- At 0 and 1 mm depth noise, all 8 states are right: lean within 1.1 deg, shift
  within 1 mm, turn within 0.4 deg.
- At 2 mm noise, the upright tile behind the leaner reads tilted (+13 deg).

## The constants, and why

- **`UPRIGHT_DEG` 5**: predicators' upright band; 4.2 mm of top movement, two bins.
- **`FALLEN_DEG` 60, `CONTACT_BAND` (8, 4) deg**: tiles fallen on each other rest at
  75-80 deg in the twin. A tile within 8 deg under its contact angle rests on its
  neighbour (the cascade-video skill's margin).
- **`DISPLACED` 4 mm, `YAW_TOLERANCE_DEG` 10**: the placing skill places within 2 mm in
  the twin; its own check allows 8 mm.
- **`SHIFT_RANGE` 10 mm**: a few mm of placing error plus a lean's base slide.
- **`BACKWARD_MARGIN` 0.5 mm, `SECTOR_FROM_DEG` 45**: see Measured (fallen runs).
- **`LEAN_SLACK_DEG` 2**: a leaner's fit is good to about a degree. The sliver one
  degree left made the tile it leans on read tilted at 1 mm noise.
- **`CRITICAL_GAP_H` 0.996**: planning-a-domino-layout's critical gap.
- The profile constants (bin, margins, over-weight 0.3, cap 15 mm, residual 6 mm) are
  the cascade-video skill's.

## Not done

- No real camera frame was audited. The twin's depth is rendered without sensor noise;
  the synthetic tests add Gaussian noise only.
- Fall direction inside a fallen run (see above).
- Without slots (a line found in the frame alone) is not done: the plan is the only
  reference.

## Provenance

The rotated-box profile model and fit are verifying-a-domino-cascade-from-video's
(`Tile`, `_profiles`, `_profile_obs`, `_fit_profile`), which adapts predicators (MIT,
commit afb6f29) with the upright / toppled bands of
`envs/pybullet_domino/components/domino_component.py`. The licence is
`LICENSE.predicators` beside this file. New here:
- the joint lean and shift fit with a backward lean;
- per-tile states and the neighbours' bodies taken out;
- the physical consistency rules;
- the gaps check.
