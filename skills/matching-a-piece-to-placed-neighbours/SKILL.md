---
name: matching-a-piece-to-placed-neighbours
description: Picks which of several loose jigsaw pieces fits an empty slot, and
  how far to turn it, by the shape of its sides alone - each piece's mask is
  split at its 4 corners into sides typed flat / tab / blank (Zolver's
  turning-angle signature), then every piece in each of its 4 rotations is
  fitted rigidly to the sides the already-placed neighbours leave at the slot
  and ranked by how closely the outlines meet. Also returns each piece's class
  (corner / edge / interior) and its grip point (the inner point farthest from
  the outline), in board metres for the winner. Use for low-texture regions
  where the box picture cannot place a piece (white snow, black cloth, blurred
  background), or to pick between a picture match's candidates; needs top-down
  piece masks (e.g. wrist-camera crops) and at least one placed neighbour or a
  border side at the slot.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [jigsaw, puzzle, shape-matching, contour, edge-matching]
  hub:
    stage: select
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw]
    provenance:
      authors: [runtime-si]
      code: https://github.com/Kawaboongawa/Zolver/blob/master/Img/filters.py
      upstream_license: MIT
gap:
  allowed_tools: []
  required_inputs: {}
  # Inputs (not listed: gap has no list or open-dict type here, so they are wired from a
  # script node in the same graph): candidates (list of {id, mask, ...}), neighbour_sides
  # ({direction: [[x, y], ...]}), border_sides (list of directions), cell_size ([w, h]).
  produces_outputs:
    best_id: string
    best_rotation_deg: float
    best_score: float
    margin: float
    piece_class: string
  exit_conditions:
    fits: One candidate fits clearly - its side types agree with the slot, its score is under max_fit (0.06) and the next candidate scores at least min_margin times worse (1.25; 1.6 with a single neighbour). best_id, best_rotation_deg and grip_point_board are the answer. With no neighbour, the only candidate whose side types fit the border sides.
    ambiguous: The best fit does not stand out from the next candidate (or, with no neighbour, several candidates' side types fit); `ranked` lists them best first for another cue (the box picture, a test placement).
    no_fit: No candidate's side types agree with the slot, the best score is over max_fit, or no candidate mask could be split into 4 sides (`pieces` says why per candidate); bad slot sides also land here.
  canonical_scripts:
    - match_piece: scripts/match_piece.py
  examples:
    - title: Choose the piece for a slot from wrist-camera masks
      path: examples/canonical_subgraph.json
---

# matching-a-piece-to-placed-neighbours

Where the picture on a piece says nothing (snow, a black hoodie, a blurred
forest), its outline still does: the slot left by placed pieces has a unique
shape, and only one piece's sides are the complement of it. This skill ranks
candidate pieces for one slot by that fit.

## How

1. **Outline.** The largest outer contour of each candidate's mask, smoothed
   3 px along itself and resampled to 850 points (the scale Zolver's
   constants were set for).
2. **Corners (Zolver).** The smoothed turning angle along the outline is
   searched for convex peaks; every 4 peaks that split the outline into sides
   of 1/8 to 1/2 of it, with Zolver's count of turning peaks inside each side,
   are candidate corner sets. Kept are the sets whose quadrilateral is near a
   rectangle about a cell in size (angles, opposite-side ratio, side length
   against the cell; a tight gate, then a loose one). Zolver's flat / indent /
   outdent cost picks one, preferring sets whose four sides all get a type.
   Each corner is then sharpened by intersecting lines fitted to the outline
   just either side of it.
3. **Side types.** Zolver's peak pattern types each side `tab` (two convex
   turns between concave ones), `blank` (the reverse) or `flat` (no turn);
   a side it cannot type is typed by its largest deviation from its chord
   (over 8 % of the side's length). The class follows from the number of
   flat sides: 0 interior, 1 edge, 2 corner.
4. **Fit.** For each of the 4 rotations, the piece sides that would face a
   placed neighbour are fitted with one rigid transform onto the neighbours'
   sides through the slot's corners (all neighbours at once), with the piece
   sides pushed outward by -1 to +3 % of their length to absorb the mask's
   edge bias. The score is the symmetric mean closest-point distance between
   each piece side and its neighbour side, over the side's length, averaged
   over the neighbours. A rotation that breaks a side-type rule (a tab against
   a tab, a knob on the border, a flat side inside the board) gets 0.02 added.
5. **Verdict.** Candidates are ranked by their best rotation. The returned
   rotation is how far to turn the piece (counter-clockwise from above) from
   where it lies; the grip point is mapped into the board by the same fit.

## Inputs

- `candidates`: `[{"id", "mask", "metres_per_pixel"?, "image_yaw_deg"?}]` -
  one top-down mask per loose piece (uint8 or bool, the whole piece and only
  it, ~60+ px across, e.g. `sam3.segment` on a wrist frame). `image_yaw_deg`
  is the board-frame angle of the image's +u axis (0 when +u is board east and
  image up is board north); `metres_per_pixel` defaults to the run's.
- `neighbour_sides`: `{"north" | "east" | "south" | "west": polyline}` - for
  each placed neighbour, its side facing the slot as `[[x, y], ...]` in board
  metres (+x east, +y north), running between the two slot corners on that
  side (either direction). From the neighbour's own analysed outline placed
  with its pose, or from a board view.
- `border_sides`: the slot's sides on the puzzle's outer edge.
- `cell_size`: `[width, height]` of a grid cell in metres; without it the
  neighbour sides' chords stand in. Corner detection uses it.
- `metres_per_pixel` (0.0004), `max_fit` (0.06), `min_margin` (1.25, or 1.6
  with one neighbour), `top_k` (5).

## Outputs

`best_id`, `best_rotation_deg`, `best_score`, `margin` (runner-up score over
the best), `piece_class`, `grip_pixel` (in the winner's mask), and
`grip_point_board` (where that pixel lands in the slot, board metres; only
with a neighbour). `ranked` (up to `top_k`): id, rotation_deg, south_side,
score, compatible, per-side scores, piece_class, side_types, grip point.
`pieces`: per candidate its corners (pixels), side types and class, or the
error that stopped its analysis. `detail` says why the route was taken.

## Measured

Procedural interlocking cuts (`tests/bundles/jigsaw_cut.py`): every interior
edge is one random Bezier side shared by its two pieces, so mating sides are
exact complements. Two styles:

- **classic**: straight shoulders, a centred knob of similar size on every
  side, grid corners jittered 4 % of a cell;
- **wavy random cut** (like the target 20 x 27, 540-piece puzzle): the whole
  side bends (half and full sine waves, up to ~13 % of its length), the knob
  sits 0.33-0.67 along it, head radius 0.12-0.22 and height 0.22-0.36 of the
  side, leaning up to 0.07; corners jittered 7 %; pieces whose knobs would
  touch are redrawn.

Masks at 64 px per cell, any angle, with "noise": outline jitter 1-2 px
(low-pass), blur sigma 0.8-1.5 px, threshold 0.35-0.65, and a 1 px erosion
half the time; the neighbour sides get 1 px jitter too. 36 puzzles of
12 x 12 per style (5184 pieces). A corner is right within 0.1 cell; a slot
answer is right when the true piece is chosen with its turn within 20 deg.
Candidates are the true piece plus N-1 random interior pieces; 360 slots per
cell of the table.

| | classic, noise | wavy, noise | classic, clean | wavy, clean |
|---|---|---|---|---|
| all 4 corners right | 99.6 % | 91.3 % | 100 % | 97.3 % |
| side types right (of all sides) | 99.6 % | 91.3 % | 100 % | 97.3 % |
| class (corner / edge / interior) | 100 % | 99.7 % | 100 % | 100 % |

Top-1 / top-3 for a slot (right piece and turn):

| neighbours, N | classic, noise | wavy, noise | classic, clean | wavy, clean |
|---|---|---|---|---|
| 1, N=10 | 46 / 77 % | 91 / 95 % | 85 / 99 % | 97 / 97 % |
| 1, N=50 | 19 / 41 % | 78 / 88 % | 57 / 88 % | 97 / 98 % |
| 2, N=10 | 94 / 100 % | 89 / 90 % | 100 / 100 % | 97 / 97 % |
| 2, N=50 | 85 / 96 % | 91 / 93 % | 100 / 100 % | 96 / 96 % |
| 3, N=10 | 99 / 99 % | 94 / 96 % | 100 / 100 % | 98 / 98 % |
| 3, N=50 | 99 / 99 % | 94 / 95 % | 100 / 100 % | 97 / 97 % |

Routes on the noisy sets: with 2-3 neighbours, 98.1-100 % of `fits` answers
were right (47-95 % of slots answered `fits`, the rest `ambiguous`); with one
neighbour (margin 1.6), `fits` was right in 81 % (classic, given for 2 % of
slots) and 98.5 % (wavy, 45 % of slots) of cases. The wavy cut's ceiling is
its corner detection: a piece whose corners were missed cannot be matched
(most of the 6-11 % misses at 2-3 neighbours). The classic cut's knobs all
look alike, so one side rarely tells pieces apart; the wavy cut's varied
knobs do. Time on one CPU core: analysing a piece median 3-5 ms, 95th
percentile 14-99 ms (a piece whose corners pass only the loose gate is
searched three times), max 0.2 s; scoring 1.8 ms per rotation per neighbour
side, so 50 candidates against 2 neighbours take about 0.7 s.

The real puzzle's cut, traced from the official flat image (approximate:
about 83 % of traced sides follow the printed seam, many lost their knob),
cannot be committed; its numbers are in the pull request that added this
skill.

## Limits

- **Synthetic validation only.** No photographed piece has been matched yet;
  the wrist-camera mask noise is imitated (jitter, blur, threshold, erosion).
  Real masks may have shadows, glare and a thick cardboard edge.
- **Corners decide everything.** A mask split at the wrong corners gives the
  piece wrong sides and it cannot win its slot. On noisy wavy pieces that is
  about 1 in 11; give `cell_size`, keep the camera square to the table (the
  rectangle gates assume a top-down view) and the mask whole.
- **One neighbour is weak.** A single side matches many pieces when the cut
  repeats its knob; prefer slots with two or three placed neighbours, or
  combine with the picture match (`localizing-a-piece-in-a-reference-image`).
- **The grid is assumed.** Cell size and the slot's corners come from the
  placed neighbours; the board frame is +x east, +y north.
- **Shape only.** Colour continuity across the seam is not used.

## The constants, and why

Each constant in `scripts/match_piece.py` carries its source. In short:
`CONTOUR_POINTS = 850`, `SIGMA 5-15`, `LOCAL_MAX_RADIUS = 10` keep Zolver's
values at its outline scale; `PEAK_FRACTION = 0.2` (Zolver 0.3) because a
random cut's big knobs turn harder than its corners; `QUAD_GATES` from the
corner sets measured on the synthetic cuts; `REFINE_SPAN = (0.04, 0.14)`
cell, inside the shoulders; `OFFSETS` -1 to +3 % of a side for the mask's
1-2 px edge bias (3-neighbour top-1 from 58 % to 99 % on noisy classic
slots); `MAX_FIT = 0.06` and `MIN_MARGIN` from the score and margin
distributions above.

## Method

Adapted from Zolver (Kawaboongawa, MIT): corner search and side typing from
`Img/filters.py` (`my_find_corner_signature`, `get_relative_angles`,
`compute_comp`, `type_peak`) and `Img/peak_detect.py`, side compatibility
from `Puzzle/Edge.py`, the edge alignment idea of `Puzzle/Mover.py` and the
edge distance of `Puzzle/Distance.py`, and the per-slot rotation search of
`Puzzle/Puzzle.py`. The script's header lists what changed; the licence is
`LICENSE.zolver`.
