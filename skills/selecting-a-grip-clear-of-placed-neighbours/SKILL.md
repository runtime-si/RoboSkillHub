---
name: selecting-a-grip-clear-of-placed-neighbours
description: Chooses where a two-finger gripper should hold a jigsaw piece, and
  along which direction it should close, so that at the piece's slot neither
  fingertip (nor its outward travel on release) lands on an already-placed
  neighbour - the piece and its neighbours are rasterised at the slot, every
  closing direction (5 degree steps) and jaw offset (1 mm) is closed onto the
  piece's extreme points within the finger's width, and each fingertip
  footprint is checked against a distance transform of the neighbours; the
  clear grip nearest the piece's centre with the most contact and clearance
  wins. With no clear grip at the slot it searches the smallest offset at
  which the piece can be set down clear and pushed in. Also returns the same
  grip on the piece as it lies (pixel and closing yaw) for the pick. Use when
  a jigsaw piece is about to be picked for a known slot with a parallel
  gripper (suction robots never meet this), after the slot and turn are
  known; planning only, under 0.1 s on CPU (up to 1.5 s with the offset search).
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [jigsaw, puzzle, grasp, parallel-gripper, collision, clearance, planning]
  hub:
    stage: select
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw]
    provenance:
      authors: [runtime-si]
gap:
  allowed_tools: []
  required_inputs: {}
  # Inputs (not listed: gap has no list or open-dict type here, so they are wired from a
  # script node in the same graph): target_outline (piece at its slot, [[x, y], ...] board
  # metres), or piece_mask + metres_per_pixel + rotation_deg + slot_center (+ image_yaw_deg,
  # corners_px); neighbour_outlines (list of board polygons) or occupied_cells ([[col, row]])
  # + cell_size + board_origin (+ tab_reach); finger_width, finger_thickness,
  # release_travel, min_clearance, max_opening, min_grip, allow_offset, top_k.
  produces_outputs:
    closing_yaw_deg: float
    grip_width_m: float
    clearance_m: float
    n_clear: int
  exit_conditions:
    grip_found: A grip at the slot keeps both fingertip footprints (plus release travel) at least min_clearance (0.5 mm) from every placed neighbour; grip_point, closing_yaw_deg, grip_width_m and contacts are in board metres at the slot.
    only_offset_place: No grip is clear at the slot, but the piece can be set down offset_m away (up to 0.6 cell) clear of the neighbours with a clear grip there, then pushed into the slot (seating-a-piece-by-wiggle-search's offset-push); the grip is given at the offset pose.
    no_clear_grip: No grip clears at the slot and no offset placement clears (typically three or four sides already placed), or no grip spans the piece within the gripper's opening, or the input could not be used; the least-bad grip is still returned for a release-above-and-press strategy.
  canonical_scripts:
    - select_grip: scripts/select_grip.py
  examples:
    - title: Choose a grip for the next piece clear of its placed neighbours
      path: examples/canonical_subgraph.json
---

# selecting-a-grip-clear-of-placed-neighbours

Jigsaw robots pick pieces by suction at their inner centre. A parallel gripper
holds a piece by two opposite edges, and its fingertips reach down to the
piece's level: at the slot, a fingertip on the side of an already-placed
neighbour would land on that neighbour. With the west and south neighbours
placed, the free sides (north, east) are adjacent, not opposite, so the jaws
have to close diagonally, across a tab, or along the free row.

## How

1. **The piece at its slot.** `target_outline` in board metres, or a top-down
   `piece_mask` turned by `rotation_deg` (from where it lies to where it fits,
   counter-clockwise from above, as `matching-a-piece-to-placed-neighbours`
   returns it; `image_yaw_deg` is the board angle of the image's +u) with its
   body centre on `slot_center`. The body centre is the mean of `corners_px`
   when given, else the centre of the minimum-area rectangle of the mask opened
   by a disk of 0.15 of its side (knobs removed).
2. **Neighbours.** `neighbour_outlines` (board metres), or `occupied_cells`
   with `cell_size` and `board_origin`, each cell grown by `tab_reach` (0.33
   cell) - conservative for classic knobs only (see Limits). The piece's own
   area is never an obstacle.
3. **Grips.** Piece and neighbours are rasterised at 0.2 mm. For each closing
   direction (0-175 degrees, 5 degree steps) and jaw offset across the piece
   (1 mm steps), the fingers close on the piece's extreme points within the
   finger's width, and each contact must be tangent to its pad at least 1 mm
   inside the pad's edge (a side inclined to the pad touches only at the
   pad's edge, and the piece would squeeze out); the grip width must be at
   least 6 mm (not a knob's neck)
   and at most the opening minus 3 mm a side. Each fingertip's footprint
   (`finger_width` along the jaw x `finger_thickness` plus `release_travel`
   outward) is checked against the distance transform of the neighbours.
4. **Score.** Among grips with clearance >= `min_clearance`: closeness of the
   jaw line to the piece's centroid (1.0), contact length on the shorter
   finger over its width (0.5), clearance capped at 2 mm (0.5).
5. **Offset.** With none clear at the slot, the piece is moved in 16
   directions by 0.05 to 0.6 cell; the smallest offset where the piece itself
   stays 0.5 mm off every neighbour and a grip clears is returned as
   `offset_m` (put down there, push in).

## Inputs

- `target_outline`, or `piece_mask` + `metres_per_pixel` + `rotation_deg` +
  `slot_center` (+ `image_yaw_deg`, `corners_px`).
- `neighbour_outlines`, or `occupied_cells` + `cell_size` + `board_origin`
  (+ `tab_reach`).
- Gripper: `finger_width` (15.5 mm), `finger_thickness` (3.9 mm), measured on
  the YAM station model's linear_4310 blade meshes (the real tips are not
  measured); `release_travel`
  (1 mm), `max_opening` (94 mm), `min_grip` (6 mm), `min_clearance` (0.5 mm).
- `allow_offset` (true), `top_k` (5).

## Outputs

- `grip_point` (board m, centre between the contacts), `closing_yaw_deg`
  (board angle of the closing direction, 0-180), `grip_width_m`,
  `clearance_m` (inf with no neighbours), `contacts` (the two pads' centres on
  their contact lines).
- `offset_m` for `only_offset_place`.
- `pick_grip_pixel`, `pick_closing_yaw_deg`: the same grip on the piece as it
  lies in the mask's image (when the piece came as a mask).
- `candidates` (ranked), `n_clear`, `n_tried`, `detail`.

## Measured

`tests/bundles/_grip_clear_eval.py`:

- **Board:** 10 x 10 generated cuts (22.5 mm cells), in the classic style and the random-cut style.
- **Fill orders:** row by row from a corner (each new piece has its west and south neighbours); frame first, then rows; random.
- **Each placement:** the skill is asked with the placed pieces round the slot as exact outlines, and again as occupied cells only.
- **Fingertips:** 15.5 x 3.9 mm (the station model's blades).

| cut / order | grip_found | only_offset_place | no_clear_grip | median clearance |
|---|---|---|---|---|
| classic / rows | 19 | 81 | 0 | 3.5 mm |
| classic / frame first | 36 | 49 | 15 | 4.0 mm |
| classic / random | 50 | 12 | 38 | 3.0 mm |
| random-cut / rows | 19 | 76 | 5 | 2.6 mm |
| random-cut / frame first | 36 | 45 | 19 | 3.0 mm |
| random-cut / random | 51 | 7 | 42 | 3.2 mm |

**By sides placed round the slot**, over all six runs:

- 0-1 sides: 196 `grip_found`, 2 `only_offset_place` (of 198).
- 2 sides: 15 `grip_found`, 268 `only_offset_place`, 22 `no_clear_grip` (of 305). A 15.5 mm blade beside two placed neighbours almost never finds a tangent grip over free table; the piece is set down offset and pushed in.
- 3 or 4 sides: 0 / 97. Every one is `no_clear_grip`: no tangent antipodal grip has both fingertips over free table, and an offset placement is blocked by the neighbours' knobs.

A row-by-row order never makes a 3-sided slot, so it suits a parallel gripper; with these blades most of its placements are offset-and-push.

**Occupied cells only** (no outlines; cells grown by 0.33 cell):

- grips found in 25-38 % of placements, against 58-100 % with outlines;
- every grip checked against the true outlines: 0 / 94 touched a neighbour on the classic cut, 2 / 104 on the random cut, whose knobs reach up to 0.51 cell past their cell (classic: 0.30).

Give outlines (the placed pieces' masks from the wrist camera, put at their slot).

Time: median 0.08-0.48 s per placement, max 1.5 s; the offset search, which runs on most 2-sided slots with these blades, is most of it.

**Fingertip size** (classic cut; counts are grip_found / only_offset_place / no_clear_grip):

| tips | rows | frame first | random |
|---|---|---|---|
| 6 x 3 mm | 93 / 7 / 0 | 85 / 0 / 15 | 63 / 5 / 32 |
| 8 x 4 mm | 79 / 21 / 0 | 81 / 4 / 15 | 59 / 9 / 32 |
| 10 x 5 mm | 52 / 48 / 0 | 57 / 28 / 15 | 55 / 11 / 34 |
| 15.5 x 3.9 mm (station model) | 19 / 81 / 0 | 36 / 49 / 15 | 50 / 12 / 38 |
| 6 x 1.8 mm (narrow blade) | 93 / 7 / 0 | 85 / 0 / 15 | 63 / 5 / 32 |
| 4 x 1.8 mm (narrow blade) | 94 / 6 / 0 | 91 / 0 / 9 | 68 / 4 / 28 |

Narrower tips grip more pieces in place.

**Narrow blades** (fitting-narrow-blade-fingertips-for-flat-parts; box blades
1.8 mm thick along the closing axis at the linear_4310 tips, both cuts, all
three orders):

- 2-sided slots: 264 / 305 direct grips at 6 mm wide and 275 / 305 at 4 mm,
  against 15 / 305 with the station blades.
- 3- and 4-sided slots, at 4 mm: 24 / 97 direct, the first in-place grips
  there at any blade size (6 mm: 2 / 97).
- Occupied cells only: at most 1 grip per run touched a true neighbour. Trimming the blades narrower (or fitting narrow pads) is the cheapest way to more in-place grips.

## Limits

- **Validated on generated cuts only**, against exact outlines. A real
  piece's outline comes from a wrist-camera mask (about 0.3 mm per px) and
  the placed pieces' outlines from where they were put, which can be off by
  the arm's placement error: raise `min_clearance` to cover it.
- **The fingertip footprint is a rectangle on the table**, sized from the
  station model's meshes, not the real blades, at the piece's
  level. A tapered tip narrower at the bottom has more room than this; a tip
  that holds the piece above its bottom edge may clear neighbours that are
  lower than its bottom. Neither is modelled.
- **3 and 4-sided slots have no clear grip** with these fingertips. Options
  outside this skill: fill in an order that avoids them (row by row), release
  just above the neighbours and press the piece down, or narrower tips.
- **Occupied cells are only conservative up to `tab_reach`**: random-cut
  knobs reach past 0.33 cell.
- No arm model: reachability of the closing yaw at the slot is
  `robot.solve_ik`'s job; the jaw is symmetric, so yaw and yaw + 180 are the
  same grip.

## The constants, and why

Each constant in `scripts/select_grip.py` carries its source: `RES_M` 0.2 mm,
`FINGER_WIDTH_M` 15.5 mm and `FINGER_THICKNESS_M` 3.9 mm (the station model's
blade meshes), `RELEASE_TRAVEL_M` 1 mm,
`MIN_CLEARANCE_M` 0.5 mm, `MAX_OPENING_M` 94 mm (linear_4310 width_fit),
`APPROACH_MARGIN_M` 3 mm, `MIN_GRIP_M` 6 mm, `ANGLE_STEP_DEG` 5,
`OFFSET_STEP_M` 1 mm, `PAD_EDGE_M` 1 mm, `TAB_REACH` 0.33 (measured knob reach), the offset search
(16 directions, 0.05-0.6 cell), `PLACE_CLEARANCE_M` 0.5 mm and the score
weights.
