---
name: reading-domino-tiles-in-clutter
description: Reads every domino in a cluttered top-down photo - touching,
  rotated, scattered tiles - as (left, right) halves with each tile's centre
  and angle, by finding each tile's black centre bar (which separates
  touching tiles without a learned model) and counting the coloured pips in
  each half (pipcounter's HSV mask and contour filter, scaled to the tile).
  Use when a domino workflow must know which tiles lie on the table (a
  draw pile, a hand laid out, a played line) from one wrist or overhead frame
  where tiles touch, before choosing or matching a tile.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [domino, pips, perception, clutter, opencv, cpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/jjmlovesgit/pipcounter/blob/79cad4071f4c/pipperjones.py
      upstream_license: MIT
gap:
  allowed_tools:
    - robot.get_observation
  required_inputs: {}
  produces_outputs:
    n_tiles: int
    total_pips: int
    camera_name: string
  exit_conditions:
    tiles_read: At least one tile bar was found, every half reads 0-9, and every pip-sized coloured blob lies on a tile.
    ambiguous: Tiles were read but a half reads over 9 or pip-sized blobs lie on no tile (a tile whose bar was missed, a face-down or occluded tile, or stray colour); tiles still lists what was read.
    no_tiles: No RGB frame, or no tile bar found.
  canonical_scripts:
    - read_tiles: scripts/read_tiles.py
  examples:
    - title: Read the tiles under the wrist camera
      path: examples/canonical_subgraph.json
---

# reading-domino-tiles-in-clutter

reading-domino-pips-and-orientation reads one singulated tile. On a table
the tiles touch; this skill separates them by the one feature every tile
has exactly once - its centre bar - and reads both halves around it.

## What it does

1. The frame (or `frame`, or `camera` in `robot.get_observation`; `mask`
   limits the read) is resized to 1200 px.
2. **Bars**: dark, low-saturation components at least 5x as long as thick,
   flanked by white on 70 % of their outline; the most common bar length
   (bars within 15 % of each other) is the tile scale and other lengths are
   dropped. A tile is then 2W long and W wide, W = bar / 0.8, across the
   bar.
3. **Pips** (pipcounter): saturated pixels (S > 80, V > 60), external
   contours; a contour of a pip's area (radius 0.05-0.16 W) inside a tile's
   rectangle counts for the half it lies in, nearest tile first.
4. Each tile: `left` / `right` (the halves along the tile's axis), `value`
   (sorted), `center_px`, `angle_deg` (long axis), `width_px`.

## Checks

**Real photos**: the 7 photos pipcounter ships (phone, double-nine sets,
2-8 tiles, rotated, scattered or touching in a grid on a white board or dark
cloth), named by their total pip count:

| photo | tiles | truth | this skill | pipcounter (its defaults) |
|---|---|---|---|---|
| 23 | 2 | 23 | 23 (8-8, 0-7) | 23 |
| 28 | 3 | 28 | **19** (4-7, 2-6; the 0-9 tile missed) | 28 |
| 35 | 3 | 35 | 35 | 35 |
| 36w (2 x 2 touching) | 4 | 36 | 36 | 36 |
| 46 | 4 | 46 | 46 | 46 |
| 64 | 6 | 64 | 64 | 64 |
| 86 (2 x 4 touching) | 8 | 86 | 86 | 86 |

Totals right on 6 of 7 (pipcounter: 7 of 7 - these are its own demo photos,
and it gives only a total, no tiles). In 28 the 0-9 tile's bar is 22 %
shorter than the others (camera tilt) and fell outside the 15 % window;
widening it to 25 % let the dark seam between the rows of 36w in as a fifth
tile. The per-tile values were not labelled independently; every half read
0-9 and the halves sum to the totals.

Unit tests (`tests/bundles/test_reading_domino_tiles_in_clutter.py`) on the
hub's rendered tiles: four rotated tiles, three touching, the observation
path, no tiles, a mask.

## Limits

- Coloured pips only (most commercial sets); white-on-black or black pips on
  white need blob detection instead of the saturation mask.
- Face-down tiles have no bar and are not reported (only as `ambiguous` when
  pip-sized colour lies on no tile).
- A tilted camera makes bar lengths differ by more than 15 % (photo 28).
- Needs ~15 px pips: the wrist D405 at 15-20 cm, not the overhead camera.

## Provenance

pipcounter (github.com/jjmlovesgit/pipcounter, `pipperjones.py`, MIT,
LICENSE.pipcounter): the HSV saturation mask, external contours and an area
window for a pip, adapted (scaled per tile, counted per half). The bar-based
tile finding is new.
