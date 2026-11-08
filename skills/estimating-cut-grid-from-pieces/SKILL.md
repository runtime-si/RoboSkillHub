---
name: estimating-cut-grid-from-pieces
description: Estimates a jigsaw's real cut grid (cols x rows) and cell size from
  the piece count printed on the box, the board's size or the box picture's
  aspect, and the corner-to-corner side lengths of a few measured pieces (top-
  down masks with depth scale, corners found as in
  matching-a-piece-to-placed-neighbours). Every grid with c x r within 10 % of
  the count is scored against the pieces' sides (with the side pairs mixed both
  ways unless each piece's turn is known), under one shared scale factor, a
  prior on how manufacturers round the count and, if given, a full count of
  edge pieces (2(c + r) - 4). Returns the grid, its posterior confidence and
  the runner-up. Use before localizing-a-piece-in-a-reference-image or
  matching-a-piece-to-placed-neighbours, which both take the grid as input,
  when the real grid has not been counted; CPU only, under 0.1 s plus a few ms
  per piece.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [jigsaw, puzzle, grid, measurement, bayesian, wrist-camera]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw]
    provenance:
      authors: [runtime-si]
gap:
  allowed_tools:
    - robot.get_observation
    - sam3.segment_text
  required_inputs:
    piece_count: int
  # Optional inputs (wired from a script node in the same graph where gap has no type for
  # them): board_width, board_height (float, m), board_aspect (float), reference (picture
  # path or array), pieces (list of {id, side_lengths | mask, ...}), camera (CameraFrame) with
  # piece_masks (list of Mask), measure_view (bool), camera_name, metres_per_pixel,
  # edge_piece_count (int), min_pieces, min_confidence, top_k.
  produces_outputs:
    grid_cols: int
    grid_rows: int
    cell_width: float
    cell_height: float
    confidence: float
    runner_up_cols: int
    runner_up_rows: int
  exit_conditions:
    estimated: At least min_pieces (6) pieces were measured, the best grid's posterior probability is at least min_confidence (0.9) and a given edge-piece count agrees with it (within 3 sigma). grid_cols, grid_rows and the cell size are the answer.
    ambiguous: Pieces were measured but the best grid is under min_confidence, or a given edge-piece count disagrees with it; grid_cols / grid_rows are the best guess and `candidates` ranks the alternatives. Give each piece's turn (angle_deg), measure more pieces, or count the edge pieces.
    insufficient_pieces: Fewer than min_pieces usable pieces (the prior-only best guess is still returned), no board size / aspect / reference, a piece count under 4, no camera for measure_view, or the corner skill is not installed; `detail` says which.
  canonical_scripts:
    - estimate_grid: scripts/estimate_grid.py
  examples:
    - title: Estimate the grid from the pieces in the current wrist view
      path: examples/canonical_subgraph.json
---

# estimating-cut-grid-from-pieces

The box says "550 pieces" and "18 x 24 in"; it does not say the cut is 20 x 27
(540 pieces), and both jigsaw skills in this hub need exactly that. Every grid
near the count predicts a cell size; a few measured pieces say which one the
die really cut.

## How

1. **Candidates.** Every (c, r) with c x r within `MAX_COUNT_DEVIATION`
   (10 %) of the count and a predicted cell (board_width / c x board_height /
   r) no longer than 1.6 times its width.
2. **Pieces.** Each piece gives its four corner-to-corner side lengths in
   metres: from a top-down mask, the corners come from
   matching-a-piece-to-placed-neighbours' Zolver corner search (loaded from
   the skill installed beside this one), plus 1 px per side for the corner
   search's own inward bias, times the mask's metres per pixel (median depth
   under the mask over the focal length). Opposite sides are averaged into two
   pair lengths. Corner-to-corner chords ignore knobs and blanks, so edge,
   corner and interior pieces all count.
3. **Score.** Per grid, every piece's two pairs are compared with the cell in
   log space: Gaussian with a per-pair sigma estimated from how much each
   piece's opposite sides differ, mixed 92 / 8 with a broad outlier term (a
   wrong corner split). A loose piece does not show which pair is its width,
   so both assignments are mixed half and half, unless the piece carries its
   turn (`angle_deg` from localizing-a-piece-in-a-reference-image, or
   `width_pair`). All pieces share one scale factor (depth and mask-edge bias
   shrink every side alike), profiled over +-25 % with a log-normal prior of
   8 %. Priors: (c x r - count) / count ~ N(0, 3 %), log cell aspect
   ~ N(0, 0.15), and a full `edge_piece_count` ~ N(2(c + r) - 4, max(1, 2 %)).
4. **Verdict.** Posterior probabilities over the candidates; the best, its
   probability (`confidence`) and the runner-up are returned.

**Without a board size** (only `board_aspect` or the box picture, cropped to the
board) the board is sized so its cells match the pieces' median cell, and only
the cells' shape and the count decide.

**Orientation is the lever.** Real cuts move every grid corner by several
percent of a cell, so one piece's sides scatter 4-6 % around the cell. For the
target, 20 x 27 cells are 1.3 % off square and 21 x 27 cells 3.6 % the other
way: with unknown orientation the long and short pairs are folded together
and the two look alike for a few hundred pieces; with each piece's turn known,
24-48 pieces separate them. The absolute size cannot (an unknown 1-5 % mask
and depth bias). The practical loop: run this skill, localize a dozen textured
pieces with its best guess (the angle survives a wrong provisional grid, see
below), feed the same pieces back with their `angle_deg`, and run it again.

## Inputs

- `piece_count`: the count printed on the box.
- `board_width`, `board_height` (m, the assembled puzzle's size from the box),
  or `board_aspect` (width / height), or `reference` (the box picture cropped
  to the board, path or RGB array; its aspect is used). Columns run along the
  board's width.
- Pieces, any mix of:
  - `pieces`: `[{"id", "side_lengths": [4 lengths in m, in order round the
    piece], "width_pair"?: 0 | 1}]` (pair 0 = sides 0 and 2 run along the
    board's x), or `[{"id", "mask", "metres_per_pixel"?, "angle_deg"?}]`, or
    this skill's own `pieces` output fed back with `angle_deg` added;
  - `piece_masks` (top-down masks) on `camera` (a CameraFrame with depth and
    intrinsics: each mask's scale is its median depth over the focal length;
    depth in metres or millimetres);
  - `measure_view: true`: one `robot.get_observation`, the wrist camera (or
    `camera_name`), `sam3.segment_text` "jigsaw puzzle piece" (score >= 0.5);
    masks touching the image border, or with an area over 1.8 times or under
    1 / 1.8 of the view's median, are dropped.
  `metres_per_pixel` is the fallback scale for masks without one. Keep the
  camera square to the table (no perspective correction).
- `edge_piece_count`: a full count of the pieces with one or two straight
  sides, if one has been made (the 4 corners included).
- `min_pieces` (6), `min_confidence` (0.9), `top_k` (5).

## Outputs

`grid_cols`, `grid_rows`, `cell_width`, `cell_height` (m: the board divided by
the grid; without a board size, the measured cells), `confidence`,
`runner_up_cols`, `runner_up_rows`, `runner_up_confidence`, `grid_count`
(c x r), `expected_edge_pieces`, `pieces_used`, `scale_ratio` (measured over
predicted cell size; 0.95-0.98 is the usual mask shrink), `side_sigma`,
`residual` (RMS log misfit of the best grid), `edge_check` (`agrees`,
`disagrees` or `not_given`), `candidates` (up to `top_k`: cols, rows, count,
cell size, probability, expected edge pieces), `pieces` (per piece: side
lengths, width pair, class, corners in pixels, scale, or the error that
stopped it) and `detail`.

## Measured

**The real target.** The official flat image of the 550-piece target
(800 x 1066 px) prints its cut lines. The spectrum of its seam-ridge
projections (Sato ridge filter, no grid assumed) peaks at 20.0 periods across
and 27.08 down; the user's 768 x 1024 copy gives 19.98 and 27.06. **The target
is cut 20 x 27 = 540**, cells 22.9 x 22.6 mm, not the 22 x 25 the two jigsaw
skills' tests assume. No photographed piece of it exists yet, so the skill
itself was not run on real pieces.

**Synthetic cuts** (`tests/bundles/jigsaw_cut.py`, classic and wavy styles,
half each): 1120 puzzles, 160 per printed count of 100, 200, 300, 500, 550,
750 and 1000. Each true grid is drawn with c x r within 4 % of the count, a
board aspect of 0.6-1.67 and a cell ratio of 0.9-1.12 (a quarter of the 550
puzzles are the target's 20 x 27 on 45.72 x 60.96 cm). The skill gets the
printed count and the board size. Pieces are random (edge pieces in their
true proportion), rendered as masks at 58-70 px per cell, any angle; half the
puzzles with noise: outline jitter 1-2 px, blur 0.8-1.5 px, threshold
0.35-0.65, 1 px erosion half the time, per-piece scale error 1 % and a
common depth bias of up to +-2 %. Every one of the 53 760 masks was split
into 4 sides; piece class right 99.9 %.

Top-1, with unknown orientation / with each piece's turn known:

| count | no pieces | 6 | 12 | 24 | 48 pieces |
|---|---|---|---|---|---|
| 100 | 54 % | 83 / 98 % | 87 / 99 % | 93 / 100 % | 93 / 100 % |
| 200 | 51 % | 73 / 96 % | 77 / 99 % | 87 / 100 % | 86 / 100 % |
| 300 | 30 % | 62 / 90 % | 66 / 94 % | 77 / 98 % | 83 / 99 % |
| 500 | 23 % | 44 / 73 % | 54 / 86 % | 68 / 95 % | 72 / 96 % |
| 550 | 14 % | 47 / 73 % | 56 / 81 % | 61 / 91 % | 68 / 98 % |
| 750 | 16 % | 33 / 56 % | 41 / 69 % | 52 / 79 % | 57 / 86 % |
| 1000 | 14 % | 23 / 48 % | 29 / 63 % | 41 / 78 % | 46 / 81 % |
| all | 29 % | 52 / 76 % | 59 / 85 % | 68 / 92 % | 72 / 94 % |

The right grid is in the top two in 91 % (unknown orientation) and 99 % (known
turn) of puzzles at 48 pieces. `estimated` answers: with unknown orientation
20-28 % of puzzles at 24-48 pieces, 99.4-99.5 % of them right; with the turn
known 60-69 % at 24-48 pieces, 99.7-99.9 % right (at 6 pieces 32 %, 98.6 %
right). The rest are `ambiguous`. Noise costs little (known turn, 48 pieces:
clean 95.7 %, noisy 92.7 %); the wavy cut costs more than the classic (90.9
against 97.5 %; unknown orientation 65 against 79 %).

- **The target's grid** (20 x 27 on its board, 40 puzzles): no pieces, never
  first (21 x 26 and 20 x 27 tie at 0.11), always in the top two; with the
  turn known, 75 / 78 / 95 / 100 % top-1 at 6 / 12 / 24 / 48 pieces, 92 %
  `estimated` at 48, all right; unknown orientation stays at 70-85 %
  (classic 100 %, wavy 50-90 %) and is never `estimated`.
- **A noisy edge count** (the true count +-2): unknown orientation, 24
  pieces, 68 to 72 %; no pieces, 29 to 36 %. Exact count, 24 pieces: 79 %.
  With the turn known, 24 pieces: 91.5 to 94.6 %.
- **Aspect only** (no board size): unknown orientation, 67 / 72 % at 24 /
  48 pieces; known turn 91 / 93 %; `estimated` right 99.3-99.9 %.
- **The localizer's angle with a wrong grid.** 64 synthetic pieces of a
  20 x 27 picture (`tests/bundles/_jigsaw_pictures.py`), located by
  localizing-a-piece-in-a-reference-image told 20 x 27, 21 x 27, 20 x 28 or
  19 x 29: the angle was within 10 degrees for 97-98 % of pieces in every
  case, so a provisional grid gives the turns this skill needs.
- **The traced real cut.** The target's 540 pieces traced from its printed
  seams (the trace started from a 20 x 27 lattice, so this is not
  independent, and about 17 % of its sides miss the seam), rendered as noisy
  masks, 40 draws: known turn 28 / 40 / 55 / 65 % top-1 at 6 / 12 / 24 / 48
  pieces, unknown orientation 10-35 %; never `estimated`. The traced pieces
  measure 1.001 wide over high where the seams give 1.016 and 4.5 % small,
  which favours 21 x 27 and 20 x 28; real masks need checking for the same.

Time on one CPU core: measuring a mask median 3 ms, 95th percentile 55 ms;
scoring 48 pieces 6 ms (100 pieces) to 48 ms (1000 pieces).

## Limits

- **No real piece photo yet.** All piece numbers are synthetic or from the
  trace; the real grid was measured from the printed seams, not by the skill.
- **Orientation matters most.** Loose pieces of near-square cells leave the
  neighbouring grids `ambiguous`; give `angle_deg` (or `width_pair`) for at
  least 12-24 pieces, or count the edge pieces as well. Loose pieces of
  clearly oblong cells have a twin: the grid whose cell is the same cell
  turned 90 degrees (10 x 20 and 15 x 14 on a 30 x 41 cm board) fits them
  equally, and only the count prior separates the two.
- **Absolute size is weak.** A shared scale factor absorbs mask and depth
  bias, so grids differing only by one column in a large puzzle (1000: 2.6 %
  in one dimension) need many oriented pieces.
- **A top-down camera.** Masks are not rectified; a tilted view foreshortens
  one pair.
- **Edge count is a cross-check, not a decider.** Near-square boards make
  2(c + r) - 4 nearly the same for every grid near the count (20 x 27,
  21 x 26 and 22 x 25 all give 90).
- **Needs matching-a-piece-to-placed-neighbours** installed beside it (or
  under `GAP_SKILLS_PATH`) to measure masks; pre-measured `side_lengths` do
  not.
- No robot or simulator check: the skill does not move the arm.

## The constants, and why

Each constant in `scripts/estimate_grid.py` carries its source. In short:
`MAX_COUNT_DEVIATION = 0.10` and `COUNT_SIGMA = 0.03` from known rounding
(target 540 for 550, -1.8 %); `CHORD_BIAS_PX = 1.0` from clean synthetic
masks (sides 0.8-1.3 px short); `SIDE_SIGMA_RANGE = (0.03, 0.15)`,
`OUTLIER_FRACTION = 0.08` (1 in 11 wavy pieces gets wrong corners in the
corner skill's own measurement); `SCALE_SIGMA = 0.08` (0.05 pulled the traced
cut to finer grids; 0.03-0.15 moved synthetic top-1 by under 1.5 points);
`CELL_ASPECT_SIGMA = 0.15`, `MAX_CELL_ASPECT = 1.6`;
`EDGE_COUNT_SIGMA = (1, 2 %)`; `MIN_PIECES = 6`, `MIN_CONFIDENCE = 0.9` from
the precision of `estimated` above. Sweeps (12-48 pieces, both orientation
modes) of the count prior (0.02-0.045), the scale prior, the outlier share
(0.03-0.15) and the aspect prior (0.10) moved top-1 by under 2.5 points; a
sigma floor of 0.045 instead of 0.03 cost 12 points without orientation, and
0.02 gained under 1.

## Method

Written for this hub: a likelihood over candidate grids with a profiled
common scale and an orientation mixture. No code was copied; the corner
search is matching-a-piece-to-placed-neighbours' (adapted from Zolver, MIT),
called through its script.
