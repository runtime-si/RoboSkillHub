---
name: registering-assembled-board-to-reference
description: Aligns a top-down view of the part-built jigsaw (overhead frame,
  rectified view or wrist mosaic) to the box picture and judges every placed
  cell - an initial homography from the board's corners (or SIFT + RANSAC) is
  refined by masked ECC coarse-to-fine (translation, affine, then homography)
  on the placed cells only, dropping cells that do not match and fitting
  again; then each placed cell is compared with its own cell of the picture
  by NCC over a small shift and over the turns its shape allows, and judged
  match, mismatch (with where it belongs instead), rotated, empty (mat
  colour), low_texture_match or unseen. Returns the homography, how far the
  content sits from the given corners (a slid assembly), and the lists of
  cells to fix. Use after a batch of placements, before building on a region,
  or before declaring the puzzle done, when the box picture is available;
  camera-only, a tenth of a second to a second on CPU.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [jigsaw, puzzle, registration, ecc, homography, verification, reference-image]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw]
    provenance:
      authors: [runtime-si]
      code: https://github.com/opencv/opencv/blob/4.x/modules/video/src/ecc.cpp
      upstream_license: Apache-2.0
      paper: "G. D. Evangelidis, E. Z. Psarakis, Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization, IEEE TPAMI 2008"
gap:
  allowed_tools: []
  required_inputs:
    camera: CameraFrame
    reference: string
    grid_cols: int
    grid_rows: int
  # Optional inputs (wired from a script node in the same graph where gap has no type for
  # them): image (RGB array or path instead of camera), board_corners_px ([[u, v] x 4],
  # top-left, top-right, bottom-right, bottom-left as the picture stands), initial_homography
  # (3x3, image -> reference pixels), placed_cells ([[col, row], ...]), board_width,
  # board_height (m), mat_rgb, motion ("auto" | "homography" | "affine"), match_ncc,
  # min_ecc, look_up, use_features.
  produces_outputs:
    ecc_correlation: float
    correction_cells: float
    n_placed: int
    n_match: int
    n_mismatch: int
    n_rotated: int
    n_empty: int
  exit_conditions:
    registered: The view aligned to the picture (final ECC correlation at least 0.7 and at least half the textured placed cells match their own place); `cells` judges every placed cell and `mismatched_cells`, `rotated_cells` and `empty_cells` list the ones to fix (all empty when the board is right).
    did_not_converge: ECC diverged or ended under the gates from the given corners or homography (or moved them more than a cell) and from SIFT features, or there was no usable image; nothing is judged. Re-take the view, give better corners, or wait for more placed pieces.
    too_little_placed: Fewer than 4 placed cells, or fewer than 3 of them textured; there is nothing to align on.
  canonical_scripts:
    - register_board: scripts/register_board.py
  examples:
    - title: Check the placed region of the puzzle against the box picture
      path: examples/canonical_subgraph.json
---

# registering-assembled-board-to-reference

The box picture is the answer key. Once the board view is aligned to it to a
few hundredths of a cell, every placed piece can be checked against the
picture it should carry: a piece in the wrong cell, a piece turned half round,
a gap where a piece fell out, or a whole section that slid on the mat. The
localizer (`localizing-a-piece-in-a-reference-image`) decides where one loose
piece goes; this skill checks what has been built.

## How

1. **Reference.** The picture is shrunk to 48 px per cell (never enlarged) and
   cut into the grid; each cell's interior (12 % off every side, where seams,
   knobs and the cut line live) is its signature. Cells with a grey standard
   deviation under 8 are low-texture.
2. **Start.** A homography from `board_corners_px` (the board's outer corners
   in the image, e.g. from fiducials or the mat's frame), or a given
   `initial_homography`; with neither, SIFT + ratio test + RANSAC.
3. **ECC.** The image is warped into the picture's frame and ECC
   (`cv2.findTransformECC` with a mask of the placed, textured cell interiors)
   fits a correction coarse-to-fine: a translation at a quarter of the
   resolution, an affine map at half, and at full resolution a homography when
   at least 20 % of the board is placed and textured, else an affine map
   (`motion="auto"`). A level that diverges keeps the previous level's warp.
4. **Outliers.** After a pass, the cells scoring NCC under 0.3 against their
   own place (wrong, turned or missing pieces) leave the mask and ECC runs
   again, at most twice more.
5. **Gates.** The alignment is accepted when the final ECC correlation is at
   least 0.7 and at least half the textured placed cells match their own place,
   and, from given corners, the correction moved no board corner more than a
   cell. Otherwise SIFT features are tried as a second start (`use_features`).
6. **Cells.** Each placed cell's interior is matched against its own place
   by NCC within 15 % of a cell either way (`shift_px`, `shift_m`: how far the
   content sits off), and also turned 180 degrees (and 90 / 270 when the cells
   are square). The verdicts:
   - `rotated`: a turn scores at least `match_ncc` (0.6) and 0.15 more than
     the unturned content; `turned_deg` says how far it is turned.
   - `match`: NCC at least 0.6.
   - `empty`: at least 60 % of the interior is within Lab distance 22 of the
     mat colour (`mat_rgb`, or the image outside the board) and the cell is
     closer to the mat than to its expected colour.
   - `low_texture_match`: a flat cell (snow, dark cloth) whose median colour,
     after a per-channel Lab gain and offset fitted on the matching cells, is
     within 28 of the picture's.
   - `mismatch`: anything else. Up to 40 mismatched cells are looked up in the
     whole picture at every allowed turn; `belongs_at` gives the cell and turn
     where that content belongs, if it scores 0.6 there.
   - `unseen`: under 90 % of the cell's interior is inside the image.
7. **Slide.** `content_shift_px` / `content_shift_m` and `correction_cells`
   say how far the content sits from where the given corners put it: a whole
   section pushed along the mat, or corners that are off.

## Inputs

- `camera` (`rgb` is used) or `image` (RGB array or path): a top-down view of
  the board. The overhead camera works when its cells are big enough (see
  Limits); otherwise a rectified mosaic of wrist views.
- `reference`: path of the box picture (or an RGB array), cropped to the
  board's edges, upright. It is prepared once and cached.
- `grid_cols`, `grid_rows`: the cut grid; (0, 0) is the picture's top-left.
- `board_corners_px` or `initial_homography`: the start (optional; SIFT
  otherwise).
- `placed_cells`: the cells believed filled; default all. A cell left out is
  neither used for alignment nor judged.
- `board_width`, `board_height` (m): shifts in metres too.
- `mat_rgb`: the mat's colour, if the image shows little of it.
- `motion` ("auto"), `match_ncc` (0.6), `min_ecc` (0.7), `look_up` (true),
  `use_features` (true).

## Outputs

- `homography`: image pixels -> reference pixels (the picture's own size).
- `ecc_correlation`, `matched_fraction`, `init` (corners / homography /
  features), `motion`.
- `correction_cells`, `content_shift_px`, `content_shift_m`.
- `cells`: one entry per placed cell: `col`, `row`, `status`, `ncc`,
  `shift_px`, `shift_m`, `texture_std`, and per status `turned_deg`,
  `turned_ncc`, `colour_distance`, `mat_fraction`, `belongs_at`.
- `n_placed`, `n_match` (match + low_texture_match), `n_mismatch`,
  `n_rotated`, `n_empty`, `mismatched_cells`, `rotated_cells`,
  `empty_cells`, `detail`.

## Measured

**Real photo against its box picture.** PuzzleMap
(https://huggingface.co/datasets/pablo-moreira/puzzle-map, CC-BY-4.0; not
redistributed here): `puzzles/100_dc.jpg` is the digital picture of a
10 x 10 puzzle and `puzzles/100_dc2.jpg` a photo of the same print (paper
texture, colour cast, slight curl). `tests/bundles/_register_board_eval.py`
puts the photo on a mat under a random perspective (corners moved up to 4 %
of the board), lights the whole view (gain 0.85-1.1 per channel, a 0.75-1.2
left-right gradient, blur, noise), swaps pairs of cells, turns cells 180
degrees and empties cells among the placed ones (3 of each on a full board,
fewer on a part-built one), and gives corners off by up to 0.3 cells. 40
trials each, seed 0:

| placed | registered | error at the placed cells' centres (median / max, cells) | right cells passed | swapped flagged | turned found | empty found |
|---|---|---|---|---|---|---|
| 100 % | 40 / 40 | 0.035 / 0.037 | 3520 / 3520 | 236 / 240 (221 mismatch, 11 empty, 4 rotated) | 112 / 120 | 120 / 120 |
| 50 % | 40 / 40 | 0.034 / 0.039 | 1680 / 1680 | 158 / 160 | 80 / 80 | 78 / 80 (2 mismatch) |
| 25 % | 40 / 40 | 0.034 / 0.049 | 840 / 840 | 78 / 80 | 40 / 40 | 40 / 40 |
| 12 % | 35 / 40 | 0.042 / 0.22 | 280 / 280 | 68 / 70 | 35 / 35 | 35 / 35 |
| 6 % (8 cells, 4 of them wrong) | 32 / 40 | 0.048 / 0.29 | 128 / 128 | 64 / 64 | 32 / 32 | 29 / 32 |

The 0.03-cell floor is the print against the digital file, not the method.
The 8 turned cells missed on the full board are 6 flat cells (judged by
colour, which a half turn does not change) and 2 near-symmetric ones; of the 4
swapped cells passed, 2 were flat (colour only) and 2 look-alikes (NCC up to
0.63 against the cell they sat in).
Corner error (full board): up to 0.3 cells 40 / 40 from the corners, 0.6 cells
40 / 40 (39 from the corners, 1 via SIFT), 1.0 cell 40 / 40 (9 from the
corners, 31 via SIFT; 9 / 40 without SIFT). SIFT alone, no corners: 40 / 40.
Time on one CPU core: median 0.10 s, max 0.33 s (full board); up to 0.9 s
with outlier passes and the SIFT retry.

Work resolution: at 32 px per cell the verdicts were the same and the
alignment 0.045 cells; at 64 px no better and twice the time.

Before the outlier passes and the gates, SIFT locked a quarter-placed board
onto the wrong place (4 cells off) with ECC 0.61-0.67; the matched-fraction
gate is what rejects that.

The target puzzle's box picture cannot be committed (copyright); its numbers
are in the pull request that added this skill.

## Limits

- **One real pair.** The photo and picture are the same print in two media;
  real assembled boards add seams, gloss and glare, and pieces that sit a
  little off. The swaps, turns and empty cells are pasted rectangles, not
  photographed pieces. Measure on the real board before trusting the
  thresholds.
- **Cell size.** At 48 px per cell the check works; the overhead Orbbec sees a
  ~2.25 cm piece at about 11 px, which is too small for NCC per cell. Use a
  wrist-camera mosaic or a high-resolution overhead camera for the cell check;
  the alignment itself still works at lower resolution.
- **Flat regions.** Low-texture cells are judged by colour only: a flat piece
  in the wrong flat cell of the same colour, or turned, passes.
- **The grid is an assumption.** Cells are equal rectangles of the given grid;
  a wrong grid shifts every cell (see `estimating-cut-grid-from-pieces`).
- **Lighting.** ECC is invariant to a global gain and offset only; strong
  glare on a glossy board lowers the correlation of the cells under it.

## The constants, and why

Each constant in `scripts/register_board.py` carries its source; in short:
`WORK_CELL_PX = 48` (see Measured);
`SEAM_INSET = 0.12` (seams, knobs and the cut line); `SHIFT_SEARCH = 0.15`;
`MATCH_NCC = 0.6` (right cells 0.87-0.97 on the pair, swapped cells up to
0.66; at 0.45, 7 swapped cells in 20 trials passed); `ROTATED_GAIN = 0.15`; `MIN_ECC = 0.7` and `MIN_MATCHED_FRACTION = 0.5`
(a wrong lock reached 0.67); `OUTLIER_NCC = 0.3`; `MAX_CORRECTION = 1.0` cell;
`AUTO_HOMOGRAPHY_FRACTION = 0.2`; `LOW_TEXTURE_STD = 8`,
`LOW_TEXTURE_COLOUR = 28`, `EMPTY_FRACTION = 0.6`, `EMPTY_COLOUR = 22` (Lab).
