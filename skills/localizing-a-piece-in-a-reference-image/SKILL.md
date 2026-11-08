---
name: localizing-a-piece-in-a-reference-image
description: Finds which grid cell of a jigsaw's box picture one piece belongs to,
  and how far the piece is turned, from a close-up wrist-camera crop and its
  mask - masked normalised cross-correlation searched over rotation at the one
  scale the grid and board size fix, coarse over the whole picture then refined
  round the best cells, with border pieces matched along their straight side.
  Returns the cell, the angle, the position in picture pixels and board metres,
  the score, the margin over the next-best cell and a ranked candidate list;
  flat pieces (white snow, black cloth) get a shortlist of flat cells of their
  colour instead. Use when a jigsaw-assembly step needs to know where a picked
  or inspected piece goes and the box picture is available; under a second per
  piece on CPU.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [jigsaw, puzzle, template-matching, ncc, localization, reference-image]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw]
    provenance:
      authors: [runtime-si]
      paper: "J. P. Lewis, Fast Normalized Cross-Correlation, Vision Interface 1995"
gap:
  allowed_tools: []
  required_inputs:
    camera: CameraFrame
    piece_mask: Mask
    reference: string
    grid_cols: int
    grid_rows: int
    board_width: float
    board_height: float
  produces_outputs:
    col: int
    row: int
    angle_deg: float
    score: float
    margin: float
  exit_conditions:
    located: The best cell's score is at least min_score (0.45) and beats every other cell by min_margin (0.04; 0.10 when the scale was guessed from the mask area); col, row and angle_deg are the answer.
    ambiguous: The best cell does not stand out (repetitive or flat art); `candidates` ranks the cells for a shape check. A low-texture piece always lands here, with the flat cells of its colour as candidates.
    not_found: The best score is under min_score, a flat piece matches no flat cell, or the input could not be used (empty mask, no scale, no placement inside the picture).
  canonical_scripts:
    - locate_piece: scripts/locate_piece.py
  examples:
    - title: Locate a piece seen by the wrist camera in the box picture
      path: examples/canonical_subgraph.json
---

# localizing-a-piece-in-a-reference-image

The box picture says where every piece goes; the piece is a small, turned,
irregular window onto it. Because the grid and the board size are known, so is
the scale: only the position and the turn are unknown, and those can be
searched exhaustively with masked normalised cross-correlation (NCC) - the
piece's own outline is the mask, so its tabs count and the mat around it does
not.

## The search

1. **Crop and scale.** The piece is cut out of the wrist frame by its mask and
   resized so that it has the reference's pixels per metre. The crop's scale
   comes from `crop_px_per_m` if given, else from the camera: focal length over
   the median depth under the mask. With neither, it is guessed from the mask
   area (one piece is about one cell; see `AREA_PER_CELL`), and the
   refinement also tries 0.94 and 1.06 times that scale.
2. **Anchor.** The piece's inner centre - the middle of the region farthest
   from its outline - is the point the search places. Tabs are narrow, so it
   sits near the body's centre (the centroid is pulled toward the tabs), and it
   is the safest place to grip: returned as `grip_pixel`.
3. **Coarse.** Grey masked NCC (OpenCV `TM_CCOEFF_NORMED` with a mask) at
   every 10 degrees (or only `allowed_angles`) over the whole picture at 18 px
   per cell. The mask is eroded by 6 % of a cell first: a real cut edge is
   darker and the segmenter's outline wanders. Each angle's score map is
   reduced to the best score per cell by where the anchor lands.
4. **Fine.** The 10 best cells, each with its two best coarse angles at least
   30 degrees apart, are refined in colour at 36 px per cell, the coarse angle
   +-6 degrees in 2 degree steps, in a window that lets the anchor move half a
   cell outside the coarse cell. The anchor's final position decides its cell.
5. **Verdict.** Best cell, its score, and the margin over the best score of any
   other cell. `located` needs score >= `min_score` and margin >= `min_margin`.

**Border pieces.** The picture is padded by a cell on every side so a template
whose bounding box reaches past the picture can still be placed. A real border
piece's mask ends at its straight side, so the padding never enters the score:
a border piece is matched on exactly the picture it carries. (The prior
measurement scored border pieces 0 % only because its test pieces carried
padding inside their masks.) `search_cells` can restrict a piece known to be
an edge or corner piece (from its shape) to the border cells.

**Flat pieces.** A piece whose masked grey standard deviation is under 6
(`low_texture`) carries no pattern for NCC to lock onto: its "best" cell is
noise. It is routed `ambiguous`, and `candidates` lists every flat reference
cell (grey std under 12 in its central 70 %) within RGB distance 90 of the
piece's median colour, closest first - the slots a shape check (the jigsaw
family's neighbour-matching skill) must decide between. Its `col`, `row` and
`angle_deg` are still the NCC best and should not be trusted.

## Inputs

- `camera`: the wrist camera frame (`rgb`; `depth` and `intrinsics` give the
  scale). Or, from Python, `piece_rgb`: a crop the size of `piece_mask`.
- `piece_mask`: the piece's mask, the size of the frame (or of `piece_rgb`),
  e.g. from `sam3.segment`. It must be the whole piece and only the piece.
- `reference`: path of the box picture (or an RGB array from Python). It is
  prepared once and cached. It must be the picture alone, cropped to the
  board's edges, upright.
- `grid_cols`, `grid_rows`: the cut grid. `board_width`, `board_height`: the
  assembled puzzle's size in metres. A reference whose aspect differs from the
  board's by over 3 % is stretched to it, with a warning.
- `crop_px_per_m`: the crop's scale, if known better than from depth.
- `piece_pose`: passed through untouched, so a piece record keeps its world
  pose next to its identity.
- `search_cells`: `[[col, row], ...]` the answer must come from.
- `allowed_angles`: piece angles (degrees) to search instead of a full turn,
  e.g. `[0, 90, 180, 270]` for a piece already squared to the camera; each is
  still refined +-6 degrees.
- `top_k` (5), `min_score` (0.45), `min_margin` (0.04).

## Outputs

- `col`, `row`: the cell (0, 0 is the picture's top-left).
- `angle_deg`: how far the piece in the camera image is turned from its
  upright place in the picture, counter-clockwise as displayed. Turning the
  piece by `-angle_deg` about the camera axis squares it to the board. For a
  top-down wrist camera, add the camera's own yaw to get it in the world.
- `position_px`: the anchor's place in the reference picture (its own pixels);
  `position_m`: the same in board metres from the picture's top-left corner,
  x along the columns, y down the rows; `cell_center_m`: the cell's centre.
- `score` (NCC, -1..1), `margin`, `candidates` (up to `top_k`: col, row,
  angle_deg, score, position_px, position_m), `low_texture`, `texture_std`,
  `grip_pixel` (in the camera frame), `piece_pose`, `detail`.

## Measured

The piece generator cuts a cell plus random tabs and blanks (straight on the
board's edge) out of a picture at 96 px per cell width, turns it by a random
angle, blurs it (sigma 1.2 px), applies a per-channel gain of 0.8-1.2 and
noise (sigma 4), and puts it on a dark mat. Right means the right cell and the
angle within 10 degrees. 22 x 25 grid, 45.72 x 60.96 cm board, every cell
once, scale known (as from depth); 12 processes in parallel on a 20-core CPU.

**Procedural picture** (792 x 1053 px, in `tests/bundles/_jigsaw_pictures.py`:
textures, shapes, a gradient, a flat white band over the top 3 rows, a plaid
patch of 9 x 7 cells):

| pieces | n | right | right cell in candidates | `located` (right) | `ambiguous` |
|---|---|---|---|---|---|
| interior, textured | 378 | 98.7 % | 99.2 % | 363 (362) | 15 |
| border (incl. 22 in the band) | 90 | 72.2 % | 100 % | 64 (64) | 26 |
| white band | 66 | 15 % (by NCC) | 98.5 % (flat shortlist) | 1 (1) | 65 |
| plaid | 42 | 45 % | 60 % | 0 | 42 |
| all | 550 | 84.7 % | 96.2 % | 428 (427) | 122 |

Angle error of right answers: median 0.6 degrees. Time per piece: median
0.50 s, 95th percentile 0.81 s, max 0.86 s (one process alone: 0.44 s).

With the scale guessed from the mask area instead (no depth, three scales
tried): textured interior pieces 95.2 % right, 345 `located` of which 344
right; all pieces 80.5 % right, 91.8 % in the candidates; median 0.8-1.1 s per
piece (three times the refinement). Give the camera's depth.

The real target's box picture cannot be committed (copyright); its numbers
are in the pull request that added this skill. Its "white bands" carry
drawn cut lines and snow texture, so they did not behave like the flat band
here.

## Limits

- **Validated on synthetic pieces only.** The pieces come from the same image
  as the reference, so print-versus-camera colour, gloss, shadows and
  perspective are only approximated (gain, blur, noise). Expect lower numbers
  on real wrist crops until it is measured on real piece photos.
- **The grid is an assumption.** A wrong `grid_cols` x `grid_rows` (or a
  reference not cropped to the board's edges) shifts every answer; the cells
  of a real puzzle are not exactly equal either. Count the real grid.
- **Flat and repetitive regions need shape.** Appearance cannot place a white
  or black piece, and a periodic pattern (plaid) gives several equally good
  cells; both come back `ambiguous` with candidates for a shape-based check.
- **A tilted camera** foreshortens the piece; the crop is not rectified, so
  keep the wrist camera roughly square to the table (the scale uses the median
  depth).
- **No feature-matching refinement.** SIFT + RANSAC placed only 19 % of
  pieces at about 35 x 41 px per cell in a prior measurement (too few
  keypoints); a LightGlue refinement against a high-resolution reference is
  not implemented.

## The constants, and why

Each constant in `scripts/locate_piece.py` carries its source; in short:

- **`WORK_CELL_PX = 36`**, **`COARSE_FACTOR = 0.5`**: 35 x 41 px cells placed
  96 % of interior pieces in the prior measurement; the coarse level (18 px)
  still finds the cell at a quarter of the cost. The reference is never
  enlarged.
- **`COARSE_STEP_DEG = 10`**, fine **+-6 in 2 degree steps**: the coarse peak
  survives +-5 degrees at 18 px per cell; the fine step leaves about 1 degree
  of error (median 0.6 measured).
- **Coarse in grey, fine in colour**: masked colour NCC over the whole
  picture cost 24 ms per angle against 7 ms in grey (0.86 s against 0.25 s
  for 36 angles); colour is kept for the decisive refinement.
- **`MIN_SCORE = 0.45`, `MIN_MARGIN = 0.04`**: on the synthetic set, right
  answers scored 0.16-0.99 (5th percentile 0.87, the low ones flat pieces);
  1 of 428 `located` answers was wrong (margin 0.106).
- **`MIN_MARGIN_GUESSED_SCALE = 0.10`**: wrong `located` answers with a
  guessed scale had margins up to 0.081 (one 0.139).
- **`LOW_TEXTURE_STD = 6`**, **`FLAT_COLOUR_DIST = 90`**: the synthetic white
  band's pieces have grey std under 5 after blur and noise; 90 covers a
  +-20 % per-channel gain on white.
- **`ERODE_FRACTION = 0.06`**, **`PAD_CELLS = 1`**: see the search above.

## Method

Rotation-searched masked NCC at a known scale, as in the jigsaw candidate
survey's prior experiment (96 % of interior pieces on a 22 x 25 grid, against
19 % for SIFT + RANSAC). Normalised cross-correlation follows Lewis (1995) as
implemented by OpenCV's `matchTemplate`. No code was copied: the coarse-to-fine
schedule, the per-cell reduction, the border handling and the flat-piece
shortlist are written for this skill.
