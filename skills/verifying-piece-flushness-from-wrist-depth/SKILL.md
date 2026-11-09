---
name: verifying-piece-flushness-from-wrist-depth
description: Checks from one close wrist depth frame (D405 at 8-16 cm) whether a
  just-placed jigsaw piece lies flush - fits a RANSAC plane (Open3D's
  segment_plane, reimplemented) to the placed neighbours' tops, or to a ring
  round the piece holding the table and neighbours as two parallel levels,
  then measures the piece's own points against it - median height (raised or
  sunk), a plane through the piece's top (tilted, which side is high) and
  height profiles across the seam facing each neighbour (open seam where the
  table shows between two tops). Routes flush, raised, tilted, gap or
  unverifiable. Use when a piece has just been placed or pressed, before releasing
  the region to the next placement; camera-only, about 30 ms on CPU.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [jigsaw, puzzle, depth, verification, ransac, plane-fit, wrist-camera]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw]
    provenance:
      authors: [runtime-si]
      code: https://github.com/isl-org/Open3D/blob/main/cpp/open3d/geometry/PointCloudSegmentation.cpp
      upstream_license: MIT
gap:
  allowed_tools: []
  required_inputs:
    camera: CameraFrame
  # Optional inputs (wired from a script node in the same graph where gap has no type for
  # them): piece_outline ([[x, y], ...] world metres, e.g. the planned slot) or piece_mask
  # (image mask), neighbour_outlines (list of world polygons), table_z (world z of the
  # table top), piece_thickness (m, 0.002), raised_m (0.0005), tilt_rise_m (0.001).
  produces_outputs:
    level_m: float
    tilt_rise_m: float
    tilt_deg: float
    seam_gap_m: float
  exit_conditions:
    flush: The piece's median height is within 0.5 mm above / 0.6 mm below its expected top, its top rises under 1 mm across it, and no open seam of 3 mm or more faces a neighbour.
    tilted: The plane through the piece's top rises 1 mm or more across the piece (one side or a tab is up); `high_side` is the world x, y direction it rises toward.
    raised: Not tilted, but the piece's median height is 0.5 mm or more above its expected top (riding on something).
    gap: Not tilted or raised, but the piece sits 0.6 mm or more below its neighbours (`gap_kind` low) or the table shows between it and a neighbour over 3 mm or more of the outline (`gap_kind` seam, `seam_gap_points` in world x, y).
    unverifiable: Too little depth on the piece (under 40 % valid, or under 150 points) or round it, no depth or intrinsics, no piece given, or most of the slot shows the surface under the piece.
  canonical_scripts:
    - verify_flush: scripts/verify_flush.py
  examples:
    - title: Check a just-placed piece against its placed neighbours
      path: examples/canonical_subgraph.json
---

# verifying-piece-flushness-from-wrist-depth

A jigsaw piece put down on a tab, half over its neighbour or a millimetre short
of its slot looks placed from above in colour. In depth it is not: it stands
proud, slopes, or leaves the table showing along a seam. The wrist D405 at
10-15 cm resolves the 2 mm piece thickness, so one frame says whether to move on
or press, nudge or re-seat.

## How

1. **Points.** The depth frame is lifted to world points with the intrinsics
   and the camera pose. The piece is `piece_outline` (world x, y, projected at
   the expected top height) or `piece_mask`; it is eroded by 1.2 mm, the edge
   the D405 smooths.
2. **Reference.** With `neighbour_outlines` the reference is a RANSAC plane
   (Open3D's `segment_plane`: random 3-point planes, the most points within
   0.8 mm, least-squares refit; planes over 10 degrees off horizontal are not
   tried) through the neighbours' tops, kept 1.2 mm off the piece; the piece
   should sit level with it. Without them it is a ring 2-12 mm round the
   piece, fitted as **two parallel levels a thickness apart** (a candidate
   plane collects points near itself and near its copies a thickness above and
   below), because the ring usually holds both table and neighbours. If both
   levels hold 15 % of the ring, the upper (neighbour tops) is the reference;
   if only one, `table_z` says whether it is the table (the piece should then
   be a thickness above it), or, without `table_z`, it is taken for the table
   when the piece sits more than 0.75 thickness above it.
3. **Piece.** Heights of the piece's points over the expected top. Points a
   thickness below (the table inside the slot of a piece that stopped short)
   are left out and counted (`uncovered_fraction`). `level_m` is the median;
   a plane fitted through the piece's points gives `tilt_rise_m` (highest
   minus lowest point of that plane inside the piece), `tilt_deg` and
   `high_side`.
4. **Seam.** Along the stretch of outline that faces a neighbour, the height
   is sampled on a line across the outline, 2.5 mm either way; a stretch lower
   than the reference by half a thickness **with surface at the reference level
   on both sides** is an open seam (the bare corner of an empty diagonal cell is
   low on one side only). `seam_gap_m` is the longest run along the outline.
5. **Verdict**, in this order: tilted (rise >= 1 mm), raised (level >=
   0.5 mm), gap low (level <= -0.6 mm), gap seam (run >= 3 mm), else flush.

## Inputs

- `camera`: a fresh wrist frame above the piece: `depth` (m), `intrinsics`,
  `pose` (camera to world, Se3Pose or 4x4). World z must be the table's normal
  to within 10 degrees.
- `piece_outline` (world x, y polygon, usually the planned slot from
  `matching-a-piece-to-placed-neighbours` or the grid) or `piece_mask`.
- `neighbour_outlines`: the placed neighbours' outlines (world x, y). Needed
  for the seam test; makes the reference unambiguous.
- `table_z`: the table top's world z, for a ring reference.
- `piece_thickness` (2 mm, assumed - measure the real puzzle), `raised_m`
  (0.5 mm), `tilt_rise_m` (1 mm).

## Outputs

`level_m`, `tilt_rise_m`, `tilt_deg`, `high_side`, `residual_p90_m`,
`uncovered_fraction`, `seam_gap_m`, `seam_gap_points`, `gap_kind`, `reference`
(neighbours / table), `valid_fraction`, `n_piece_points`,
`n_reference_points`, `detail`.

## Measured

**Real sensor noise.** 648 wrist D405 frames from a real YAM station
(640 x 480, fx 394, the station's wood table at 8-16 cm). Per-pixel plane-fit
noise is 0.39-0.44 mm std (60 x 60 px patches). A flat piece-sized disc (22.5 mm)
measured against the ring round it, on 113 real windows, reads a level of
0.06 mm median, 0.19 mm 90th percentile, 0.37 mm 99th, and a **rise of 0.30 mm
median, 0.77 mm 90th, 1.16 mm 99th**. This is the sensor's low-frequency error
plus the table's own relief; a quadratic reference did not reduce it. This is
why the tilt gate is 1 mm, not the 0.6 mm first planned.

**Rendered pieces with that real noise.** `tests/bundles/_flush_eval.py`:

- The scene is a 3 x 3 patch of a generated cut (classic or random-cut, 22.5 mm cells, 2 mm thick) on a table, with 1-4 neighbours placed at random.
- The camera sits at the window's depth, tilted 0-15 degrees, and the depth is ray-cast with parallax and blurred 1 px.
- The depth is a mosaic of real noise windows (256 px, at least 95 % valid, plane removed in inverse depth), one centred on the piece, holes kept.

30 trials per case; each cell is right / 30:

| case | neighbour outlines | ring + table_z | ring only |
|---|---|---|---|
| flush | 24 | 24 | 26 |
| raised 0.3 / 0.5 / 0.8 / 1 / 2 mm | 1 / 15 / 23 / 23 / 27 | 3 / 22 / 27 / 28 / 26 | 1 / 20 / 26 / 27 / 26 |
| tilted (one side up) 0.3 / 0.6 / 1 / 2 mm | 4 / 3 / 19 / 30 | 5 / 3 / 17 / 30 | 5 / 3 / 14 / 30 |
| open seam (piece short of a neighbour) 0.5 / 1 / 1.5 / 2 mm | 12 / 29 / 27 / 25 | 0 (no seam test) | 0 |
| sunk 1 mm (neighbours raised) | 30 | 9 | 9 |

- **Flush, with neighbour outlines:** 24 of 30 pass; the misses are 4 tilted, 1 raised and 1 gap. These are the false alarms of real relief (the flush rise measured 0.52 mm median, 1.55 mm max).
- **Tilted 1 mm:** a 1 mm tilt has a 0.5 mm mean rise, so 7 of the 11 not called tilted were called raised. Counting any not-flush route: 27 / 30 for tilted 1 mm, 28 / 30 for raised 0.8 mm, 29 / 30 for raised 1 mm.
- **Breadboard windows** (40-75 % valid; 6 mm holes and edges, not just dropout), 15 trials: 7 / 15 flush pieces passed. This surface is too uneven to use as the reference.
- **Time:** median 27 ms, max 0.10 s.

## Limits

- **Synthetic geometry, real noise.** The piece, its defects and the seams are
  rendered; only the sensor noise and holes are real. Glossy print, printed
  texture and real cut edges (bevel, fibres) are not in it; measure on the
  real puzzle before trusting the gates.
- **Flush false alarms are about 1 in 5.** That is the price of catching a
  raised piece from 0.5-0.8 mm. A second frame from another height, or one
  more press before re-checking, is cheaper than missing a raised piece.
  Under 0.5 mm raised or 1 mm tilted is below what the D405 resolves here.
- **The seam test needs neighbour outlines** and a gap of about 1 mm; 0.5 mm
  was found 12 / 30.
- **Ring mode without `table_z`** cannot tell a piece riding up by most of a
  thickness on a ring of neighbours only from a flat piece on the table; give
  `neighbour_outlines` or `table_z`.
- **The piece thickness is assumed** (2 mm); measure the target puzzle.
- **The pose must be good**: the outline is projected with the camera pose;
  a few mm of hand-eye error moves the regions (the 1.2 mm margins absorb
  about 1 mm).

## The constants, and why

Each constant in `scripts/verify_flush.py` carries its source: `RAISED_M`
0.5 mm and `SUNK_M` 0.6 mm (real level noise 0.37 mm at the 99th percentile),
`TILT_RISE_M` 1 mm (real rise 1.16 mm at the 99th percentile), `EDGE_MARGIN_M`
1.2 mm (D405 edge smoothing about 3 px), `RING_M` 2-12 mm, `SEAM_BAND_M`
2.5 mm, `SEAM_MIN_LENGTH_M` 3 mm, `RANSAC_THRESHOLD_M` 0.8 mm (2 sigma),
`MAX_PLANE_TILT_DEG` 10, `MIN_VALID_FRACTION` 0.4, `MIN_LEVEL_FRACTION` 0.15,
`TABLE_RULE` 0.75.
