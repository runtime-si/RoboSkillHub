---
name: choosing-next-piece-and-slot
description: Chooses the next jigsaw placement (which loose piece, which grid
  cell, which turn) from the board as believed, the pieces' localization
  candidates and backtracking-a-wrong-placement's ledger. Loose pieces are
  assigned to free cells with SciPy's linear_sum_assignment on -log P(piece
  belongs in cell), and the confident pairs are ranked by a stage plan - frame
  before interior, cells with the most placed neighbours, then confidence -
  skipping forbidden pairs and deferred cells and respecting corner / edge /
  interior classes. With two arms each job stays inside one free arm's reach
  and clear of the other's work, and each arm gets a short separated queue.
  When nothing is confident it names the best-supported slot and candidates
  for a shape match, the pieces to photograph again, or asks for more pieces.
  Use when a jigsaw-assembly loop needs its next pick, after localizing-a-
  piece-in-a-reference-image and after any backtrack; planning only,
  milliseconds on CPU.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [jigsaw, puzzle, planning, assignment, ordering, bimanual]
  hub:
    stage: select
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw]
    provenance:
      authors: [runtime-si]
      code: https://github.com/scipy/scipy/blob/main/scipy/optimize/_lsap.py
      upstream_license: BSD-3-Clause
gap:
  allowed_tools: []
  required_inputs:
    grid_cols: int
    grid_rows: int
  # Inputs (not listed: gap has no list or open-dict type here, so they are wired from a
  # script node in the same graph): placements (list of {piece, col, row}), pieces (localizer
  # records {piece, candidates, piece_class?, position?, relocalized?}), ledger or
  # ledger_path (backtracking-a-wrong-placement's), arms (list of {id, reach, busy?,
  # busy_at?}) with board_pose ({origin, yaw}), board_width, board_height; exclude_cells.
  produces_outputs:
    piece: string
    col: int
    row: int
    angle_deg: float
    prob: float
    n_neighbours: int
    arm: string
  exit_conditions:
    chosen: A confident pair (P >= min_confidence, 0.5) was assigned and is reachable; piece, col, row, angle_deg (the localizer's, so turn the piece by -angle_deg) and arm are the next placement, `queue` the next few per free arm.
    need_shape_match: No pair is confident; `slot` is the open cell with the most placed neighbours (frame first on ties) and `shape_candidates` the pieces listing it, likeliest first, then the orphans (loose pieces with no confident cell). Run matching-a-piece-to-placed-neighbours on them and place its `fits` answer.
    need_relocalize: No confident pair and no slot a shape match could use; `relocalize` lists the loose pieces with no confident cell that were not re-photographed max_relocalize (1) times yet.
    need_more_pieces: The known loose pieces fit none of the open cells (or there are none); find and localize more pieces, or hand over.
    unreachable: Confident jobs exist but none can be done by a free arm with the whole pick-to-place path in its reach and clear of a busy arm (or every arm is busy); `unreachable` lists them with each arm's reason. Move those pieces to a buffer both arms reach.
    done: Every cell is placed.
  canonical_scripts:
    - choose_next: scripts/choose_next.py
  examples:
    - title: Choose the next jigsaw placement
      path: examples/canonical_subgraph.json
---

# choosing-next-piece-and-slot

The forward half of a jigsaw loop: before every pick, decide which loose piece
goes where next. The localizer says, per piece, which cells it may belong to
and how sure it is; backtracking-a-wrong-placement says which (piece, cell)
pairs failed and which cells to leave for later. This skill turns both into one
next placement and a short queue, or says what evidence is missing.

## How

1. **Board.** Free cells are those not placed and not deferred in the ledger
   (and not in `exclude_cells`). A cell is a corner, edge or interior cell of
   the grid; the frame is every corner and edge cell. Its placed neighbours
   are counted over the 4 sides.
2. **Probabilities.** P(piece belongs in cell) from the localizer's score and
   margin over the piece's best other listed cell, through
   backtracking-a-wrong-placement's calibration (same constants, a test keeps
   them equal); a flat shortlist (score null) shares the piece equally; a
   candidate with `confidence` (a shape match's verdict) uses it. The
   ledger's posterior replaces P for a pair it has seen (except a newer shape
   verdict), halved per unexplained re-seat failure; forbidden pairs are
   dropped. A known `piece_class` (corner / edge / interior, from
   matching-a-piece-to-placed-neighbours) only goes to cells of that class.
3. **Assignment.** All loose pieces against all open cells with
   `linear_sum_assignment` on -log P, a "stay loose" column per piece at
   -log 0.01. Two pieces never claim one cell, and a piece goes where it
   beats its rivals (the cell's own sure piece keeps it from an impostor).
4. **Rank** the assigned pairs with P >= `min_confidence` (0.5). `plan`:
   frontier cells (a placed neighbour, or any cell on an empty board) before
   new islands; within that, key = -log P - `neighbour_weight` (1) x placed
   neighbours - `frame_weight` (100, i.e. strictly first) on frame cells while
   the frame is unfinished; on an empty board corners get two neighbours'
   worth. Baselines for comparison: `frame_first` (frame, then confidence),
   `most_neighbours`, `confidence`, `random`.
5. **Arms** (optional). A job is feasible for a free arm when the whole
   straight pick-to-place path (1 cm steps) is inside that arm's reach (disc
   from its base with min/max radius, and/or a polygon) - so it never enters
   the zone only the other arm reaches - and at least 10 cm from a busy arm's
   `busy_at`. Jobs are dealt in rank order to the feasible arm with the
   shorter queue (then the nearer one); the jobs two arms do at the same queue
   position are at least 3 cells apart and their picks 10 cm apart. The first
   job dealt is `chosen`.
6. **Nothing confident.** `need_shape_match` for the open cell with the most
   placed neighbours (frame first on ties, at least one neighbour, or a frame
   cell on an empty board), with the pieces listing it, likeliest first, and -
   once every piece is known (`all_pieces_known`, default: at least as many
   loose pieces as free cells) - the orphans, up to 50. Else deferred cells are
   reopened for pieces not forbidden there. Else `need_relocalize` for the
   orphans not yet re-photographed `max_relocalize` (1) times. Else
   `need_more_pieces`.

Call it again after every placement or backtrack: it keeps no state (the
ledger belongs to the backtracking skill) and takes 1-8 ms for 540 pieces.

## Inputs

- `grid_cols`, `grid_rows`: the cut grid (the target is 20 x 27, see
  estimating-cut-grid-from-pieces).
- `placements`: `[{"piece", "col", "row"}]`, the board as believed (row 0 at
  the top of the picture, as the localizer counts).
- `pieces`: one record per known loose piece (placed ones are ignored):
  localizing-a-piece-in-a-reference-image's output with `piece` (an id) and
  `candidates` (`col`, `row`, `score`, `angle_deg`); optional `piece_class`,
  `position` (world `[x, y]`, or the localizer's `piece_pose`) for arms,
  `relocalized` (times photographed again), and candidates with `confidence`
  from a shape match.
- `ledger` (dict) or `ledger_path`: backtracking-a-wrong-placement's ledger;
  read, never written.
- `arms`: `[{"id", "reach": {"base": [x, y], "max_radius", "min_radius"?} |
  {"polygon": [[x, y], ...]}, "busy"?, "busy_at"?: [x, y]}]` with `board_pose`
  (`origin`: world x, y of the picture's top-left corner; `yaw`: world angle
  of the column direction, rows run at yaw - 90 degrees), `board_width`,
  `board_height` (m).
- `exclude_cells` (a shape match found no fit there; clear them after the next
  placement next to them), `policy` (`plan`), `queue_length` (3),
  `min_confidence` (0.5), `neighbour_weight` (1), `frame_weight` (100),
  `max_relocalize` (1), `allow_islands` (true), `reopen_deferred` (true),
  `all_pieces_known`, `seed` (for `random`).

## Outputs

`route`; for `chosen`: `piece`, `col`, `row`, `angle_deg` (the localizer's:
turn the piece by -angle_deg to square it), `prob`, `n_neighbours`,
`cell_class`, `tier` (0 frontier, 1 island), `arm` ("" without arms);
`queue` (`{arm id: [jobs]}`, or `{"any": [...]}` without arms); for
`need_shape_match`: `slot`, `n_neighbours`, `shape_candidates`; for
`need_relocalize`: `relocalize`; for `unreachable`: `unreachable` (up to 10
jobs with each arm's reason: `out_of_reach`, `near_busy_<id>`,
`no_position`); always `n_free`, `n_placed`, `n_jobs`, `frame_done`,
`reopened` (when deferred cells were offered again) and `detail`.

## Measured

`eval/assembly.py` (CPU, 12 processes) assembles whole 20 x 27 puzzles (the
target's grid, 540 pieces) in a loop: the skill's answer is carried out, every
placement is checked by the procedural cut's shape against its placed
neighbours (`tests/bundles/_backtrack_sim.py`: a tab on a tab does not seat,
a seam gap over 0.06 of the side fails, a piece with no placed neighbour
always passes), and every failure goes to backtracking-a-wrong-placement until
the board passes. `need_shape_match` is answered by a stand-in for
matching-a-piece-to-placed-neighbours at that skill's measured `fits` rates
and precision per cut and neighbour count (`ambiguous` / `no_fit` skip the
slot until a neighbour is placed; when only skipped slots are left and every
piece is known, its top answer is placed at its measured top-1 rate);
`need_relocalize` swaps in a second photo's record. Every piece is known and
carries its true class (the shape skill classifies 99.7-100 % right).

Evidence: **localizer** = localizing-a-piece-in-a-reference-image run on all
540 pieces of a 20 x 27 procedural picture (`eval/localize_all.py`: textured,
a flat white band over the top 3 rows, a plaid patch; 81 % of top-1 cells
right, 420 `located`, 51 flat pieces each listing all 60 band cells; a second
photo seed for re-localizing); **synth10 / synth25** = synthetic records with
10 % / 25 % of top-1 answers confidently wrong (score 0.80-0.95, margin
0-0.08, the true cell second in 70 % of them).

16 puzzles per row (cut seeds 0-15). "Finished": all 540 right. "Per
failure": checks passed per check failed. "Correct" / "wrong" are cells at the
end (a wrong piece that passed every check stays).

| evidence, cut | policy | correct | wrong | finished | failures | per failure |
|---|---|---|---|---|---|---|
| localizer, wavy | **plan** | 540.0 | 0.0 | 16 / 16 | 7.4 | 73.4 |
|  | most neighbours | 539.1 | 0.2 | 13 / 16 | 20.4 | 26.5 |
|  | frame first | 539.2 | 0.2 | 13 / 16 | 21.7 | 25.0 |
|  | confidence | 539.9 | 0.0 | 15 / 16 | 18.6 | 29.2 |
|  | random | 539.5 | 0.1 | 14 / 16 | 17.9 | 30.2 |
| localizer, classic | **plan** | 536.3 | 1.8 | 6 / 16 | 40.2 | 13.4 |
|  | most neighbours | 534.2 | 3.4 | 5 / 16 | 51.9 | 10.4 |
|  | frame first | 523.1 | 10.6 | 0 / 16 | 80.7 | 6.7 |
|  | confidence | 523.2 | 10.5 | 0 / 16 | 83.6 | 6.4 |
|  | random | 519.4 | 12.8 | 0 / 16 | 89.0 | 6.0 |
| synth10, wavy | **plan** | 539.8 | 0.0 | 13 / 16 | 11.3 | 47.9 |
|  | most neighbours | 539.8 | 0.0 | 13 / 16 | 15.3 | 35.4 |
|  | frame first | 539.9 | 0.0 | 15 / 16 | 14.8 | 36.7 |
|  | confidence | 539.7 | 0.1 | 14 / 16 | 17.5 | 30.9 |
|  | random | 539.4 | 0.1 | 9 / 16 | 16.8 | 32.4 |
| synth10, classic | **plan** | 538.6 | 0.9 | 7 / 16 | 15.6 | 34.9 |
|  | most neighbours | 538.2 | 1.1 | 6 / 16 | 19.8 | 27.5 |
|  | frame first | 538.5 | 0.9 | 8 / 16 | 15.3 | 35.4 |
|  | confidence | 538.0 | 1.4 | 4 / 16 | 17.5 | 31.0 |
|  | random | 538.1 | 1.3 | 7 / 16 | 16.8 | 32.4 |
| synth25, wavy | **plan** | 534.8 | 0.9 | 4 / 16 | 79.5 | 6.8 |
|  | most neighbours | 535.0 | 0.9 | 4 / 16 | 72.4 | 7.5 |
|  | frame first | 532.9 | 1.3 | 4 / 16 | 88.4 | 6.2 |
|  | confidence | 531.2 | 1.7 | 3 / 16 | 97.9 | 5.5 |
|  | random | 524.6 | 2.9 | 0 / 16 | 118.4 | 4.6 |
| synth25, classic | **plan** | 524.2 | 8.3 | 0 / 16 | 108.9 | 5.0 |
|  | most neighbours | 522.4 | 9.1 | 0 / 16 | 114.2 | 4.7 |
|  | frame first | 521.4 | 9.3 | 0 / 16 | 117.4 | 4.6 |
|  | confidence | 524.9 | 7.9 | 0 / 16 | 103.4 | 5.2 |
|  | random | 518.1 | 11.9 | 0 / 16 | 122.1 | 4.5 |

- **With the real localizer the plan is best on both cuts**: wavy, every
  puzzle finished with 7 failures per 540 pieces against 18-22 for every
  baseline; classic (where a wrong piece passes one seam 46 % of the time and
  a one-neighbour shape match is right 19 %), 1.8 wrong cells left and 6 of 16
  finished, against 10-13 wrong and none finished for random, frame-first and
  confidence order, and 3.4 / 5 for most-neighbours. The gain comes from where
  pieces meet their checks: the plan grows one connected region, so a wrong
  piece is usually set down against two or three neighbours and fails at
  once, and the flat band is filled by shape matches with 2-3 neighbours.
- **synth10**: every policy places 538-540 right; the plan has the fewest or
  near-fewest failures. **synth25**: plan and most-neighbours tie (wavy
  534.8-535.0 right, 72-80 failures), random is clearly worst (524.6 right,
  118 failures, none finished); on the classic cut nothing finishes.
- Unfinished runs end `need_more_pieces` with a few cells empty (a few synth25
  runs hit the 2500-step budget instead): a wrong piece
  passed its checks somewhere (so the true piece of an empty cell is on the
  board), or the true piece never listed the cell and the shape stand-in
  found it ambiguous at a test placement.
- **Ablations** (plan, cut seeds 0-7, localizer wavy / localizer classic / synth25
  wavy / synth25 classic; correct cells, failures): default 540 / 537.5 / 535.9 /
  524.2 right, 6.4 / 42.0 / 68.8 / 115.8 failures. `neighbour_weight` 0: the same
  except synth25 wavy 536.2 right, 58 failures (2: identical to 1; the shape-match
  slot order, which always puts neighbours first, carries most of the effect).
  `min_confidence` 0.3: 533.4 / 529.9 / 535.1 / 516.6 right, 72 / 59 / 80 / 168
  failures; 0.7: 538.1 / 528.1 / 533.1 / 525.9, 23 / 68 / 86 / 105. Without piece
  classes: 539.9 / 532.0 / 530.1 / 511.5 right, 24 / 68 / 123 / 160 failures.
  `frame_weight` 0, 1, 3 against 100 (an earlier version of the benchmark, 8
  puzzles): within 2 cells and 12 failures of each other, 3 identical to 100.
- **Two arms** (`--arms`, 8 puzzles): YAMs at x = +-0.45 m with 0.6 m reach,
  the board centred between them, pieces staged beside the board on their
  cell's side (10 % on the wrong side). The plan used 373 / 407 / 367 / 422
  rounds (one or two simultaneous placements) for the 555-650 placements, both
  arms busy in 178-190 of them; 19-24 pieces per puzzle went through the
  shared buffer after `unreachable`. Correct cells 539.5 / 531.9 / 532.6 /
  522.1 (plan) against 539.5 / 523.2 / 524.9 / 516.0 (random).
- **Pieces arriving in batches of 60** (`--batch 60`, wavy, 8 puzzles): with
  the localizer every policy ends with 539-540 right; random had the fewest
  failures (6.8 against 18.2 for the plan: an island set down alone is never
  checked, so "failures" also counts how often a policy checks). With synth25
  records every policy ran out of the 2500-step budget at 496-502 right
  (thousands of shape-match calls on slots whose piece is not known yet).

One call takes a median 1-5 ms (540 pieces, ~500 free cells, CPU; up to 8 ms
with two arms).

## Limits

- **Synthetic validation only.** The checks are the cut's geometry, the shape
  match is a stand-in at that skill's published rates, and the localizer ran
  on synthetic crops of a procedural picture; no robot, camera or simulator
  run. The skill moves nothing, so no simulation check applies; none was run
  (no GPU host this round either).
- **Reach is a model.** Discs or polygons and a straight pick-to-place path
  are what the skill checks; IK, wrist yaw and the arms' links are not
  (checking-dual-arm-clearance does that per move). `ARM_CLEARANCE` (10 cm)
  and `PAIR_SEPARATION_CELLS` (3) are assumptions.
- **The queue is a snapshot.** Later queue entries assume the board as it is
  now (their neighbour counts do not include earlier entries); re-plan after
  every placement.
- **Ordering helps less than evidence.** With 25 % confidently wrong top-1
  answers every policy leaves cells wrong on the classic cut; the fix is
  better localization or shape evidence, not order.
- `NEIGHBOUR_WEIGHT`, `FRAME_WEIGHT` and `MIN_CONFIDENCE` were chosen on the
  benchmark above, not on a robot.

## The constants, and why

Each constant in `scripts/choose_next.py` carries its source. In short:
`CONF_*`, `MARGIN_CAP`, `FLAT_PROB_FLOOR`, `ATTEMPT_PROB_FACTOR`,
`MIN_ASSIGN_PROB`: backtracking-a-wrong-placement's, so both skills rank pairs
alike. `MIN_CONFIDENCE = 0.5`: calibrated pairs at 0.5-0.9 were right 89 % of
the time, under 0.5 at most 21 %. `NEIGHBOUR_WEIGHT = 1`, `FRAME_WEIGHT =
100`: the ablations above. `SHAPE_MAX_CANDIDATES = 50` and
`SHAPE_MIN_NEIGHBOURS = 1`: what matching-a-piece-to-placed-neighbours was
measured with and needs; with 10 the true flat piece was often left out.
`ARM_CLEARANCE = 0.10` m, `PAIR_SEPARATION_CELLS = 3`: a wrist with a D405 is
about 8 cm across (assumption).

## Method

Written for the hub; no code copied. The assignment is SciPy's
`scipy.optimize.linear_sum_assignment` (BSD-3-Clause, the modified
Jonker-Volgenant algorithm of Crouse 2016, called as a library), as the jigsaw
survey's candidate for this skill proposes (cost -log match probability; its
travel term is replaced here by hard reach feasibility and the queue's
nearest-arm tie-break). The stage plan (frame, most neighbours, confidence),
the shape-match fallback, the ledger reading and the two-arm dealing are this
skill's design.
