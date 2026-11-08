---
name: reading-domino-pips-and-orientation
description: Reads each domino's two pip values, which end is which, its long axis
  and whether it lies flat, stands or has fallen, from above. Per tile mask
  (sam3.segment_text, or upstream masks) it finds the centre divider (an elongated dark MSER region,
  as Domino-Arm did), scores the pips, splits them into the halves and decodes
  each against the seven double-six layouts, and lifts the tile
  with the depth (geometry.mask_to_world_points) to its world centre, ends, long
  axis and resting state (flat / upright / fallen, from its height over the table).
  Returns (a, b) with a at end a, end and half-centre pixels and world points,
  pip scores and layout margins. Pips are ~2 px from the
  station's overhead camera, so a tile under 30 px wide routes needs_closeup with
  its world centre - read it from a wrist D405 close-up
  (refining-target-from-wrist-closeup / framing-a-target-by-mask-edge-steering
  first). Use when a domino task needs a tile's value, ends, pose or whether it
  still stands (game play, sorting, a toppled chain).
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, domino, pips, counting, blob-detection, mser, orientation, d405]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/SponTanious/Domino-Arm/blob/master/Functions/Domino_Detection.m
      upstream_license: MIT
gap:
  allowed_tools:
    - robot.get_observation
    - sam3.segment_text
    - geometry.mask_to_world_points
  required_inputs: {}
  produces_outputs:
    state: str
    center_world: Vec3
    axis_world: Vec3
    camera_name: str
  exit_conditions:
    read: Every judged tile (the one nearest target_point if given, else all) lies flat face up, its mask is one 2:1 tile, its divider was found and both halves decoded with a layout margin of at least 0.3 (glare-covered positions left out) and no strong pip left unexplained; value / value_ab / ends carry the (first) tile.
    ambiguous: Tiles were segmented but at least one judged tile was not read, its status says why - `not_flat` (standing or fallen; state, axis and ends are still given), `face_down` (no divider and no pips), or `ambiguous` (a small layout margin, glare over a deciding position, an unexplained pip, no divider, or a mask that is not one tile).
    needs_closeup: A judged flat tile is narrower than min_tile_width_px (30 px) in this frame and none failed otherwise; `closeup_targets` lists their world centres for refining-target-from-wrist-closeup, then run this skill again on the wrist frame.
    not_seen: No camera frame, the segmenter failed or returned no mask scoring min_score and 40 px.
  canonical_scripts:
    - read_dominoes: scripts/read_dominoes.py
  examples:
    - title: Read the dominoes in the wrist camera's view
      path: examples/canonical_subgraph.json
---

# reading-domino-pips-and-orientation

Domino-Arm's idea, kept: the divider line in the middle of a domino is the
most reliable feature on it, so find that first, then count the pips in a
rectangle on each side of it. This port does it per segmented tile in a
canonical crop, decodes the counts against the real pip layouts, and puts the
tile in the world.

## Camera: use the wrist D405, not the overhead

A pip is ~4 mm on a ~24 mm tile. The station's overhead Orbbec (640 x 480,
fx 461, ~1.0 m to the table) sees the tile ~11 px wide and a pip ~1.8 px:
enough to find the tile and its state, not to read it. The skill measures the
tile's width in the frame and, under `min_tile_width_px` (30 px, pips ~5 px),
routes `needs_closeup` with the tile's world centre in `closeup_targets`. A
wrist D405 at 0.20 m (fx ~390 at 640 x 480) sees the tile ~47 px wide.

The graph:

1. `reading-domino-pips-and-orientation` on the overhead frame
   (`camera_name` = the overhead camera): states, axes and ends of every tile,
   `needs_closeup` for the flat ones.
2. `refining-target-from-wrist-closeup` with `target_point` = a
   `closeup_targets` entry and `object_name: "domino"` (look-down view at
   0.15-0.25 m); add `framing-a-target-by-mask-edge-steering` when the tile is
   clipped at the image edge.
3. This skill again with `target_point` set to the same point (the default
   camera choice is the wrist camera riding nearest the tool): `read`.

This skill does not move the arm.

## What it does

1. Camera frame: `camera`, else from `robot.get_observation` the named
   camera, else the `wrist` / `eye_in_hand` camera nearest the tool, else the
   first one.
2. Masks: `masks` from upstream, else `sam3.segment_text(query="domino")`;
   masks under `min_score` (0.3) or 40 px dropped, the largest component kept,
   overlapping duplicates dropped (best first), at most `max_tiles`.
3. Table height: `table_z`, else the median z of a ring a third of a tile
   wide round all the masks, lifted with `geometry.mask_to_world_points`.
4. Per tile: lift the mask; height = 95th-percentile z minus the table;
   footprint from the 2-98 % extents along its principal axes. State:
   height under (T + W) / 2 is `flat`, over (W + L) / 2 is `upright`
   (standing on an end, as in a chain), else `fallen` (on a long edge or
   leaning). Without depth the state is `unknown` and the pips are read
   anyway.
5. Not flat: status `not_flat`; upright tiles get a vertical axis from the
   table to the top, fallen ones the axis between the two halves' world
   medians. Narrower than `min_tile_width_px`: `too_small`.
6. Flat: the mask's minimum-area rectangle is warped to a 128 x 64 crop
   (end a, the image-left end, at x = 0), so every pixel constant below is a
   fraction of the tile. The image is read in its darkest channel (coloured
   pips - yellow, light blue - are dark there; in luminance a yellow pip is
   within 0.2 of a white tile). Light pips on a dark tile are inverted
   (median under 110, or `pip_polarity`).
7. Divider (the source's test): an MSER region at least 4x longer than wide,
   0.3-1.1 tile widths long, within 20 deg of square to the long axis, its
   centre within 15 % of the length of the middle and darker than the tile.
   Fallback: the column within 7 % of the middle that stays dark down 80 % of
   the middle rows. Neither: the geometric middle, and the tile is not
   `read`. The divider's pixels (dilated by a pip radius) are taken out of
   the pip evidence.
8. Pip candidates: OpenCV `SimpleBlobDetector` (dark, circularity 0.6,
   convexity 0.8, inertia 0.4, radius 0.55-1.6 x the nominal 0.085 W),
   `HoughCircles` (the source used `imfindcircles`) and round MSER regions
   (the source's second branch), merged within 1.2 radii; each scored by its
   Weber contrast against the surrounding ring (capped at the tile's level).
9. Halves (the source's rectangles): from 5 % of the length past the divider
   to the tile's end. Each half's nine layout positions (0.25 / 0.5 / 0.75)
   get the strongest detection whose nearest position they are (within 0.16
   half-widths), else the best contrast probed within 0.08 of the position;
   positions take pips greedily so one pip never counts twice.
10. Decoding: every double-six layout (and the quarter turn of 2, 3 and 6)
    scores sum(score - 0.25) over its positions plus sum(0.25 - score) over
    the others; the best value and its margin over the best other value.
    Positions under saturated glare (>= 250 over 30 % of the pip disc; on a
    light tile only those read empty) are left out as missing data.
11. Tile status: `read` when the mask is one tile (1.5-2.7 times longer than
    wide and filling >= 85 % of its rectangle), the divider was found, both
    margins are >= 0.3 and no spot scoring >= 0.5 is left out of the layout
    (on or off the grid); `face_down` when there is neither divider nor pip;
    else `ambiguous` with the reason in `detail`.
12. World: the two half masks are lifted; the axis runs from half a's median
    to half b's, the ends are the tile cloud's 1-99 % extent along it.

## Inputs

- `camera` (a CameraFrame: `rgb`, `depth`, `intrinsics`, `pose`) or
  `camera_name` / `arm_id` for the observation's camera.
- `masks` (+ `mask_scores`) from upstream, else `query` (default "domino";
  SAM3 scored "domino tile" under 0.15 on every photo below) for `segmenter`
  (default `sam3.segment_text`).
- `target_point`: judge only the tile nearest it (in the plane).
- `table_z`, `tile_size` (L, W, T; default 0.048, 0.024, 0.008 m),
  `pip_polarity` (`auto` | `dark` | `light`), `min_score` (0.3),
  `min_tile_width_px` (30), `max_tiles` (8).

## Outputs

`tiles`: per tile `index`, `status` (`read` | `ambiguous` | `face_down` |
`not_flat` | `too_small`), `state`, `value` (sorted pair), `value_ab` (a at
`ends_px[0]` / `ends_world[0]`), `ends_px`, `half_centers_px`, `pips_px` and
`pip_scores` (per half, the positions the decoded layout uses),
`pip_counts_raw` (the detections over 0.25 per half - the source's count),
`layout_margin`, `divider_found` / `divider_score`, `glare_positions`,
`stray_pips`, `center_world`, `axis_world` (unit, end a to end b),
`ends_world`, `half_centers_world`, `length`, `height`, `footprint`,
`width_px`, `fill`, `mask_score`, `detail`. The lists (`value`, `value_ab`,
`ends_world`, `closeup_targets`, `tiles`) are not in `produces_outputs`, which
takes only gap schema types. The top level repeats the judged tile's
`value`, `value_ab`, `state`, `center_world`, `axis_world`, `ends_world`
(`tile` is it whole), plus `closeup_targets`, `camera_name`, `detail`.

## The constants, and why

- **`TILE_SIZE` 48 x 24 x 8 mm**: a common double-six size (US sets ~51 x 25
  x 9.5). Only the state thresholds use it, midway between the three resting
  heights (16 mm and 36 mm), so a set 20 % off still classifies.
- **`MIN_TILE_WIDTH_PX = 30`**: see Measured; pips are then ~5 px.
- **`DIVIDER_ELONGATION = 4`, `DIVIDER_GAP_FRAC = 0.05`**: Domino_Detection.m
  (`axe(1) > 4*axe(2)`; rectangles from 0.05 x the length past the divider).
  The gap also keeps a centre spinner rivet out of both halves.
- **`MSER_MAX_VARIATION = 1.0`** (source 0.25): a 2-4 px line in the crop
  grows by its whole length per grey step; at 0.25 / 0.5 MSER missed the
  divider in up to a third of blurred / noisy renders, at 1.0 in none.
  OpenCV's MSER also returns nothing for a hard-edged noise-free region, so
  the crop is blurred by 0.7 px first.
- **`DIVIDER_CENTRE_BAND = 0.15`, `PROFILE_CENTRE_BAND = 0.07`**: the source
  took the divider anywhere; a photo at ~40 deg put it 0.1 of the length off
  the mask's middle. The fallback column stays within 0.07: a six's inner pip
  column is a dark streak 0.125 off the middle, and at 0.15 the fallback took
  it for the divider and misread sixes.
- **The divider is required for `read`**: splitting at the mask's middle
  misread a tilted 6-4 as 6-6 on a photo.
- **`MIN_FILL = 0.85`**: single tiles on the near-top-down photos filled
  0.88-0.97 of their rectangle; a sam3 mask joining halves of two touching
  tiles (read 2-4 for a 2-2) and obliquely seen tiles filled less.
- **`STRAY_SCORE = 0.5`**: pips on the photos scored 0.85-0.97; an obliquely
  seen tile's shaded rim scored 0.28 next to a layout position.
- **`MIN_PIP_SCORE = 0.25`, `MIN_MARGIN = 0.3`**: a clear pip scores ~0.8-0.9
  (black on white or ivory, white on black). A position with no trace at all
  shifts a layout by only 0.25, so a margin under 0.3 cannot exclude a pip
  that left no trace; one clear pip separates neighbouring values by ~1.1.
- **`LAYOUT_TOLERANCE = 0.16`, nearest position only**: sets that place pips
  at 0.20 / 0.80 or 0.30 / 0.70 are 0.05 off the grid; at 0.12 those sets went
  `ambiguous` 10 % of the time, at 0.16 without the nearest-position rule one
  jittered pip answered for two positions and fours read as sixes.
- **`PIP_DIAMETER_FRAC = 0.17`**: 4 mm on 24 mm; 0.75x and 1.3x pips read
  28/28 in the sweep.

## Measured

**Synthetic frames** (`tests/bundles/_domino_render.py`: a look-down RGB-D
camera over tiles drawn 4x supersampled with rounded corners, a shaded rim, a
divider groove and pips at the layout positions with 0.02 W random offsets;
masks are the drawn tiles; lifting by the real `geometry.mask_to_world_points`).
Each condition: all 28 values x 6 angles (0-211 deg), a tile 48 px wide (a
D405 at ~0.2 m).

| condition | read right | wrong | not read |
|---|---|---|---|
| clean; blur 1.5 px; noise 10 / 20 grey; blur 1 + noise 6; black tiles with white pips; ivory tiles; coloured pips (with and without blur 1 + noise 6); spinner rivet; pips 0.75x / 1.3x; pips at 0.20 / 0.80; mask eroded 2 px | 168 each | 0 | 0 |
| pips at 0.30 / 0.70 | 166 | 0 | 2 |
| mask dilated 3 px | 167 | 0 | 1 |
| blur 2.5 px | 151 | 0 | 17 (divider not seen) |
| glare spot (+80 / +120 / +200 grey, sigma 0.2 W) | 144 / 132 / 88 | 0 | 24 / 36 / 80 |
| blur 1 + noise 6 + glare +80 | 145 | 0 | 23 |
| pip offsets 0.05 W (1.2 mm) | 154 | 2 (4-6 and 5-6 as 6-6) | 12 |

Over all 22 conditions: 3499 / 3696 read right, 2 wrong, the rest `ambiguous`.
Per value (132 trials each): 0-0 119, 0-1 128, 0-2 112, 0-3 126, 0-4 118,
0-5 129, 0-6 128, 1-1 130, 1-2 126, 1-3 128, 1-4 126, 1-5 130, 1-6 125,
2-2 125, 2-3 125, 2-4 117, 2-5 118, 2-6 122, 3-3 129, 3-4 120, 3-5 130,
3-6 127, 4-4 121, 4-5 129, 4-6 127 (1 wrong), 5-5 130, 5-6 127 (1 wrong),
6-6 127. Every right read had `value_ab` in the order of its ends; the world
axis was within 1.6 deg. Tile width (blur 1 + noise 6): 28 px 168 / 0 wrong,
24 px 157 / 0, 20 px 149 / 0, 16 px 17 / 4 wrong - hence the 30 px gate. The
unit tests check heights (flat 8 mm, upright 48 mm, fallen 24 mm, each within
2-3 mm), footprint, centre (2 mm) and ends (4 mm).

**Real photos** (Wikimedia Commons, 1280 px wide; masks from the hub's
`tools/sam3` `segment_text("domino")` on the Brev L40S; photos have no depth,
so state is `unknown`; truth is my reading of each tile at up to 2560 px):

| photo (licence) | set, view | tiles | read: right / wrong | ambiguous | face_down |
|---|---|---|---|---|---|
| Dominoes - Muggins - IMG 7709 (CC BY-SA 4.0) | double-six, black pips, spinners; ~25 deg tilt; 97-107 px wide | 7 | 7 / 0 | 0 | 0 |
| Dominoes - Sniff - IMG 7711 (CC BY-SA 4.0) | the same set | 9 | 9 / 0 | 0 | 0 |
| Muggins.jpg (CC BY 3.0) | double-six, coloured pips; ~45 deg oblique, tiles touching; 43-76 px | 28 | 2 / 0 | 22 | 4 (the face-down pile) |
| Chicken Foot domino game 2 (CC BY 2.0) | double-nine, oblique | 19 | 1 / 0 | 18 | 0 |
| Mexican Train (CC BY 3.0) | double-twelve, oblique | 40 (cap) | 1 / 0 | 38 | 1 (the pile) |
| Domino piece 6 and 2 (CC BY-SA 2.0) | black tile with embossed and white pips | 1 | 0 | 1 (best 2-6, margin 0.16) | 0 |
| Domino Piece isolated on White (CC BY-SA 4.0) | a 5-1 seen from the side | 1 | 0 | 1 | 0 |

20 reads, 20 right. On the oblique photos most tiles fail the mask-fill test
(a perspective trapezoid, or touching tiles merged), lose the divider, or
show pips of a bigger set; they route `ambiguous` rather than wrong. Earlier
versions of the script read wrong on these photos, and each became a rule
above: luminance missed yellow and light-blue pips (a yellow 6 read as 0
twice, a 2-6 as 2-2), splitting at the mask's middle read a tilted 6-4 as
6-6, a mask joining two tiles read 2-4 for a 2-2, and a side face in the
mask shifted a 4-0 to 6-0. 34-45 ms per tile on a CPU at 1280 px (0.3 s for
a tile 525 px wide).

Not measured: real wrist D405 frames of dominoes (none exist at the station
yet), the depth and state path on real depth (synthetic depth only), a VLM
cross-check of the counts.

## Provenance

Adapted from SponTanious/Domino-Arm (MIT, 2016, MATLAB on a Kinect v2):
`Functions/Domino_Detection.m` (divider by an elongated MSER region, pips by
`imfindcircles` plus round MSER regions, duplicates merged, counting in the
two rectangles beside the divider) and `Class/Domino.m` (the sorted value
pair, the tile's position between the rectangles). The pip blob detector is
OpenCV's `SimpleBlobDetector` (Apache-2.0, called, not copied). Not copied:
HookJordan/DominoesDetection (GPL-3.0). Changed: per-mask canonical
rectification instead of fixed pixel sizes for one camera height; contrast
scoring, position probes and layout decoding with a margin instead of a raw
count; glare and off-layout checks; dark tiles; world lifting, state and
axis from depth; routed exits. The arm control, `get_domino_location.m`'s
goal-slot table and the Kinect calibration are not ported. The license is
`LICENSE.domino-arm` beside this file.
