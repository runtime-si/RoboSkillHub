---
name: classifying-piece-sides-for-border-first
description: Types a jigsaw piece's four sides flat / tab / blank from a top-down
  mask and calls it a corner, edge or interior piece - the side lines are
  found as the direction (1 degree steps) whose projections pile the outline's
  points into the sharpest peak near each extreme (straight shoulders and flat
  sides), then each side's middle stretch is a tab where it reaches past its
  line by 12 % of the side, a blank where it falls behind it, flat otherwise.
  Returns the outward direction of each flat side (where the board's edge must
  lie), sides near the threshold as uncertain, and, for many pieces, the ids
  sorted into corners, edges and interiors for a border-first build. Use when
  sorting loose pieces for a frame-first strategy, or to orient a border piece
  before placing it, from wrist-camera masks of pieces lying flat; CPU only,
  about 8 ms a piece.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [jigsaw, puzzle, shape, contour, edge-pieces, classification]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw]
    provenance:
      authors: [runtime-si]
gap:
  allowed_tools: []
  required_inputs: {}
  # Inputs (wired from a script node in the same graph where gap has no type for them):
  # piece_mask (one top-down mask) or pieces ([{id, mask}, ...]); max_uncertain (0).
  produces_outputs:
    piece_class: string
    uncertain_sides: int
  exit_conditions:
    classified: Every side was typed with no more than max_uncertain (0) of them within 0.35 x 12 % of the threshold; for a list of pieces, at least one mask was usable and the ids are sorted into corners / edges / interiors / uncertain.
    uncertain: One piece, but a side's deviation lies near the threshold or the piece has three or four flat sides; look closer, or decide by shape matching.
    no_piece: No usable mask (empty, under 400 px, or no square body found).
  canonical_scripts:
    - classify_sides: scripts/classify_sides.py
  examples:
    - title: Type one piece's sides from its wrist-camera mask
      path: examples/canonical_subgraph.json
---

# classifying-piece-sides-for-border-first

A border-first build needs the corner and edge pieces out of the pile first,
and needs to know which way each one's straight side faces. A piece's shape
tells both: a flat side is a straight line from corner to corner; a tab bulges
past that line, a blank cuts behind it.

## How

1. The mask's largest component (closed by 3 px) is outlined and resampled to
   480 points by arc length.
2. **Side lines.** For each direction over a quarter turn (1 degree steps) the
   outline is projected on the direction and its normal. A straight shoulder or
   flat side piles its points into one or two histogram bins (1.5 % of the
   extent); knobs spread theirs. In the outer half toward each end the densest
   pair of adjacent bins is that side's line. The direction with the highest
   sum of the four peaks wins. (A first version took the minimum-area rectangle
   of the mask opened by a knob-sized disk: a piece with two deep blanks split
   in two, and knob roots pushed the rectangle out - 194 / 300 side sequences
   right on the classic cut, against 299 / 300 now.)
3. **Typing.** Each outline point goes to the side whose line is nearest in the
   rectangle's normalised frame. In the middle 70 % of a side, the largest
   signed distance past (+) or behind (-) the line, over the side's length,
   decides: over 0.12 tab, under -0.12 blank, else flat. Within 0.35 x 0.12 of
   the threshold the side is uncertain.
4. Two flat sides: corner; one: edge; none: interior; more: unknown
   (uncertain). `flat_normals_deg` are the outward normals of the flat sides
   (image degrees, counter-clockwise from +x with y up).

## Inputs

- `piece_mask`: one top-down mask (uint8 or bool), the whole piece, nothing
  else; or `pieces`: `[{"id", "mask"}, ...]`.
- `max_uncertain` (0): uncertain sides tolerated before a piece is uncertain.

## Outputs

`piece_class`, `side_types` (counter-clockwise from the body's +u side),
`side_normals_deg`, `flat_normals_deg`, `side_deviations`, `uncertain_sides`,
`body_angle_deg`; for a list, `pieces` (per piece the same), `corners`,
`edges`, `interiors`, `uncertain` (ids), `detail`.

## Measured

`tests/bundles/_classify_sides_eval.py`.

**Generated pieces** (every piece of three 10 x 10 cuts, mask at 64 px a cell,
turned at random, outline jittered up to 1 px):

| cut | class right | sides right | whole sequence right |
|---|---|---|---|
| classic | 300 / 300 | 1199 / 1200 | 299 / 300 |
| random-cut | 270 / 300 (13 no body found) | 1040 / 1148 | 191 / 287 |

The random-cut generator's deep, angled blanks are harder than real cuts; its
misses are mostly edge pieces read as interior (16).

**Real piece photos.** PuzzleMap
(https://huggingface.co/datasets/pablo-moreira/puzzle-map, CC-BY-4.0, not
redistributed): 654 single-piece photos with per-side labels, sampled to 154
corner, 250 edge and 250 interior pieces. No learned segmenter was used: the
mask is GrabCut inside the annotated box. Many photos have fingers over the
outline or a strong shadow beside the piece. Side sequences are compared up to
a quarter turn.

- **Class:** 449 / 654 right (69 %). Pieces with no uncertain side: 387 / 482 right (80 %). 40 masks gave no square body.
- **Sides:** 2135 / 2456 right; whole sequence 369 / 614.
- **By photo set:** the 120-piece Avengers set, 212 / 262 (81 %); 195 / 225 (87 %) with no uncertain side. The "camera" set (hand-held, cluttered), 36 / 100.
- **Confusion:**

  | true \ predicted | corner | edge | interior | unknown / error |
  |---|---|---|---|---|
  | corner (154) | 103 | 25 | 10 | 16 |
  | edge (250) | 40 | 169 | 23 | 18 |
  | interior (250) | 8 | 44 | 177 | 21 |

  In a sample of 12 failures, most masks had a shadow or a finger merged into them, filling a blank (it then reads flat).

- **Time:** median 8 ms a piece.

## Limits

- **The mask decides.** On masks of generated pieces it is near perfect; on
  GrabCut masks of photos it is 69-81 %. A wrist camera over pieces lying flat
  on a plain mat, with a learned segmenter (`sam3`), should sit near the
  better end; that is not measured here.
- **Shadows and fingers** merged into a blank make it look flat: an interior
  piece reads as an edge. Light from above, and take the mask from depth where
  the piece stands 2 mm proud of the mat.
- **Square-ish bodies only** (aspect up to 1.6); a piece cut from an irregular
  grid or a whimsy cut is out of scope.
- An edge piece misread as interior is only found late (it never joins the
  frame); one misread as a corner is placed wrongly and found by
  `backtracking-a-wrong-placement`.

## The constants, and why

`CONTOUR_POINTS` 480, `ANGLE_STEP_DEG` 1, `BIN_FRACTION` 0.015, `MAX_ASPECT`
1.6, `MIDDLE` 0.7, `KNOB_FRACTION` 0.12 (knobs reach 0.20-0.30 of a side, a flat
side's wobble under 0.05), `UNCERTAIN` 0.35, `MIN_AREA_PX` 400.
