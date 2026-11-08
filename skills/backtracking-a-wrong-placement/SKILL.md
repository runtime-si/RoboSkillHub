---
name: backtracking-a-wrong-placement
description: Decides what to undo after a jigsaw piece fails its check at a cell
  (it does not seat, a seam with a placed neighbour scores badly, or a re-check
  puts it elsewhere) - blames the piece or one of the cell's placed neighbours
  by a small Bayesian vote over this failure and every earlier one in a
  persistent JSON ledger (priors from the pieces' localization scores and
  margins), then re-solves the assignment of the known loose pieces and the
  lifted ones to the free cells with SciPy's linear_sum_assignment on
  -log P(piece belongs in cell). Returns the corrective actions (seat again,
  remove piece X from cell C, place piece Y at C' with its angle and attempt
  number, set aside, re-localize, search for a cell, defer a cell) under a
  retry ladder with limits, and saves the ledger so a run can resume. Use when
  a jigsaw placement has just failed verification and the pieces have records
  from localizing-a-piece-in-a-reference-image (seam scores from
  matching-a-piece-to-placed-neighbours help); planning only, no robot motion.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [jigsaw, puzzle, backtracking, assignment, recovery, tabu]
  hub:
    stage: recover
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
  # script node in the same graph): failure ({piece, col, row, kind, seam_scores?,
  # event_id?}), placements (list of {piece, col, row, score?, margin?, confidence?,
  # verified?, locked?}), pieces (localizer records {piece, candidates}), ledger_path or
  # ledger.
  produces_outputs:
    culprit_piece: string
    culprit_p_wrong: float
    p_false_alarm: float
    n_actions: int
  exit_conditions:
    retry_seat: The piece did not seat but no seam is bad and the check being wrong is likelier than any placement being wrong; seat the same piece again (once by default, max_seat_retries).
    reassigned: The failed piece itself is the likeliest wrong placement; it is lifted and the best admissible known piece for its cell is placed there.
    remove_neighbour: A placed neighbour is the likeliest wrong placement (with neighbour removals left for this cell); it and the failed piece are lifted, its cell gets its best known piece and the failed piece is placed again.
    no_alternative: A culprit was lifted but no known piece is admissible for a cell it emptied (every candidate forbidden or under min_assign_prob); `actions` asks to re-localize the lifted pieces and to search the loose pieces for that cell.
    give_up: The cell has failed with more than max_pieces_per_cell different pieces and its neighbours cannot be blamed, or max_backtracks is spent; the piece is lifted and the cell deferred (left out of every later re-solve).
  canonical_scripts:
    - backtrack: scripts/backtrack.py
  examples:
    - title: Decide the fix after a jigsaw placement failed its check
      path: examples/canonical_subgraph.json
---

# backtracking-a-wrong-placement

A forward planner (choose a piece and a slot, place it, check it) never
undoes anything. When a check fails, the question is *which* placement is
wrong: the piece just set down, or a neighbour that was placed earlier on thin
evidence and passed its own check by luck. Removing the wrong one first saves
pieces and time; blaming the right piece forever loses a cell. This skill
makes that decision from the evidence the jigsaw family already produces and
remembers it between calls.

## How

1. **Who is involved.** `not_seated`: the piece and every placed, unlocked
   neighbour of the cell, except neighbours whose reported seam score is under
   `seam_fail` (0.06, the neighbour-matching skill's `max_fit`).
   `seam_mismatch`: the piece and the neighbours whose seam score is over
   `seam_fail`. `recheck_failed`: the piece only. Neighbours are keyed by
   piece id or by direction (north = row - 1, as the localizer counts rows from
   the top of the picture).
2. **Priors.** Each placed (piece, cell) pair's probability of being right is
   a logistic function of the localizer's score for that cell and its margin
   over the piece's best other listed cell (calibrated on that skill's own
   runs, below), clipped to 0.02-0.98. A pair with `confidence` (e.g. placed by
   a shape match) uses it; `verified` (it passed a seat check before) halves
   its odds of being wrong.
3. **The vote.** The ledger keeps every failure with its involved pairs. For
   the failures linked to this one (sharing pairs), every set of up to three
   wrong pairs is weighed by the product of their prior odds and, per failure
   it does not explain, `false_alarm` (0.05). This gives each suspect's
   P(wrong) and P(the check itself was wrong). Two different pieces that fail
   beside the same neighbour make that neighbour the culprit without a
   special rule; a piece blamed on weak evidence is exonerated later.
4. **The ladder.** (a) `retry_seat` if the piece did not seat, no seam is
   bad, attempts are left and a false alarm is the likeliest explanation.
   (b) Otherwise blame the likeliest suspect still on the board (a neighbour
   only while `max_neighbour_removals` (2) for this cell are left and it is not
   `locked`); among suspects at least half as likely, prefer one whose removal
   leaves a known piece for every emptied cell. (c) `give_up` when more than
   `max_pieces_per_cell` (3) different pieces failed this cell, or
   `max_backtracks` (40) is spent.
5. **Forbid and re-solve.** The culprit's pair is left out of this call's
   re-solve, and forbidden for good once the vote puts it at P(wrong) >= 0.8.
   The failed piece (and a blamed neighbour) is lifted; every known loose
   piece plus the lifted ones is assigned to the free cells by
   `linear_sum_assignment` on -log P (posterior P for pairs the vote has seen,
   halved per unexplained re-seat failure), with a dummy "stay loose" column
   per piece at -log `min_assign_prob` (0.01).
6. **Actions.** Removals first (the failed piece, then the neighbour), then
   placements into the emptied cells (the neighbour's cell first, so the slot
   is retried against a fixed neighbour), each with `angle_deg`, `prob`,
   `attempt` and `max_attempts`; a lifted piece planned for another cell is
   `set_aside` with `planned_cell` (the forward planner places it when that
   cell is on the frontier); one with no admissible cell gets `relocalize`
   (with `exclude_cells`); an emptied cell with no piece gets
   `search_for_cell` (run the localizer on loose pieces with `search_cells`
   set to it; `exclude_pieces`).

## Inputs

- `failure`: `{"piece", "col", "row", "kind", "seam_scores"?, "event_id"?}`;
  kind `not_seated`, `seam_mismatch` or `recheck_failed`. `seam_scores` maps a
  neighbour (piece id or direction) to its seam score (lower is better, the
  neighbour-matching skill's per-side `sides`). A repeated `event_id` returns
  the stored decision (a resumed run that crashed after deciding but before
  acting).
- `placements`: the board as believed, `[{"piece", "col", "row", "score"?,
  "margin"?, "confidence"?, "verified"?, "locked"?}]`; the failed piece may be
  in it or not. Lock the seed corner and anything a person confirmed.
- `pieces`: one localizer record per known piece, placed or loose:
  `{"piece", "candidates": [{"col", "row", "score", "angle_deg"}, ...]}`
  (`localizing-a-piece-in-a-reference-image`'s `candidates`; a low-texture
  piece's shortlist with `score: null` shares its probability equally).
- `grid_cols`, `grid_rows`; `ledger_path` (JSON, written atomically) or
  `ledger` (dict); `free_cells` (default: every cell not occupied or deferred).
- Limits: `max_seat_retries` (1), `max_pieces_per_cell` (3),
  `max_neighbour_removals` (2), `max_backtracks` (40); `false_alarm` (0.05),
  `seam_fail` (0.06), `min_assign_prob` (0.01).

## Outputs

`route`, `culprit_piece`, `culprit_cell`, `culprit_p_wrong`, `p_false_alarm`,
`suspects` (each with `p_wrong` and `prior_conf`), `actions`, `n_actions`,
`assignment` (the whole re-solve: piece, cell, prob, cost, angle),
`forbidden`, `ledger` (and `ledger_path`), `detail`.

The ledger: `forbidden` (piece, cell, reason, event, p_wrong), `attempts` and
`unexplained` per pair, `p_right` (the last posterior per pair), `events`
(each failure with its involved pairs and the decision), `neighbour_removals`
per cell, `deferred_cells`, `backtracks`.

## Measured

`eval/benchmark.py` (CPU, 12 processes). A 22 x 25 board has a correctly
assembled 7 x 7 block at a random place; one wrong placement is injected and
the skill is called in a loop, its actions carried out on the board. Every
placement is checked by the shape of a procedural cut
(`tests/bundles/jigsaw_cut.py`, classic or wavy): a tab against a tab cannot
seat, otherwise the seam score is the mean gap between the two true outlines
over the side length plus a 0.005-0.03 floor; over 0.06 fails. A failed check
is the next call's input.

- `wrong_piece`: the piece set down on a free cell beside the block is the
  loose piece the localizer rates likeliest for that cell after the true one.
- `wrong_neighbour`: a block cell beside the free cell holds such an impostor,
  which passed its own check against one block neighbour (its other block
  neighbours are left empty); then the right piece is set down on the free
  cell.

Solved within k: after k calls the board holds no wrong piece and the free
cell (and the neighbour's cell) hold their true pieces. A mistake that passes
the check is "undetected" and not run (the check's limit, not this skill's).

Evidence: **localizer** = `localizing-a-piece-in-a-reference-image` run on all
550 pieces of the 22 x 25 procedural picture (photo seed 2, held out from the
calibration's seeds 0-1; top_k 5); **synthetic** = clean records (true cell
first, score 0.93-0.99, margin 0.3-0.5) with the impostor confused (score
0.85-0.95, margin 0-0.1); **hard** = the impostor as confident as a right
piece (score 0.93-0.99, margin 0.1-0.3); **decoy** = hard, plus a second wrong
piece half-rated for the free cell (so blaming the right piece there does not
leave the cell empty).

300 scenarios per row; "ran" excludes undetected mistakes (in brackets) and
impostors caught by their own check. Unsolved runs end with a later wrong
piece that passed its check ("wrong left") or with a cell left empty after
`no_alternative` / `give_up` ("empty").

| evidence, cut, case | ran (undetected) | within 1 | within 2 | within 3 | unsolved: wrong left / empty |
|---|---|---|---|---|---|
| localizer, classic, wrong piece | 144 (119) | 92.4 % | 92.4 % | 92.4 % | 8 / 3 |
| localizer, classic, wrong neighbour | 115 (102) | 82.6 % | 82.6 % | 82.6 % | 11 / 9 |
| localizer, wavy, wrong piece | 232 (31) | 90.5 % | 90.5 % | 91.4 % | 5 / 15 |
| localizer, wavy, wrong neighbour | 77 (10) | 76.6 % | 76.6 % | 76.6 % | 9 / 9 |
| synthetic, both cuts, both cases | 841 | 100 % | 100 % | 100 % | 0 / 0 |
| hard, classic, wrong neighbour | 162 (138) | 98.1 % | 98.1 % | 98.8 % | 2 / 0 |
| hard, wavy, wrong neighbour | 249 (39) | 98.4 % | 98.4 % | 100 % | 0 / 0 |
| hard, both cuts, wrong piece | 430 | 100 % | 100 % | 100 % | 0 / 0 |
| decoy, classic, wrong neighbour | 159 (141) | 0 % | 0 % | 58.5 % | 64 / 0 |
| decoy, wavy, wrong neighbour | 250 (38) | 0 % | 0 % | 82.8 % | 40 / 0 |
| decoy, both cuts, wrong piece | 418 | 100 % | 100 % | 100 % | 0 / 0 |

- With the localizer's evidence, the misses are mostly cells whose true piece
  the localizer did not list (plaid; restricted to runs where it was listed:
  93 / 90 / 95 / 81 % within 3) and the white band (flat pieces, given up),
  plus a replacement that was wrong but passed its check.
- **decoy** needs three calls by construction: the first failure is best
  explained by the right piece (equally rated, unverified), the decoy then
  fails beside the same neighbour, and the third failure (or the right piece
  failing again) turns the vote on the neighbour; the misses are a decoy that
  seated against the wrong neighbour and passed (46 % of wrong pieces pass one
  classic seam).
- Median physical operations when solved: 2 (wrong piece: remove, place),
  4 (wrong neighbour: two removals, two placements), 8 (decoy).
- With 5 % of right placements failing anyway (set down off the seat,
  `--exec-fail 0.05`): localizer rows 85.4 / 72.5 / 85.8 / 69.7 % within 1 and
  90.1 / 72.5 / 90.6 / 69.7 % within 3 (same row order); decoy wrong piece
  91.7-96.9 % within 1, 95.2-100 % within 2.
- One call takes 7-14 ms (550 known pieces, ~500 free cells, CPU).

## Limits

- **Synthetic validation only.** The checks are the cut's geometry, not a
  robot pressing a piece; the seat-check model (floor, threshold, clash) is an
  assumption. The skill moves nothing, so no simulation check applies; none
  was run (no GPU host this round either).
- **Only as good as the localizer's lists.** A piece whose true cell is not in
  its candidates (most wrong answers in a repetitive region: 78 of 93 wrong
  top-1 answers in the calibration runs) cannot be put back by the re-solve:
  the skill asks for `relocalize` / `search_for_cell`, which the benchmark does
  not carry out, so those runs count as unsolved.
- **Flat regions** (white snow) give every flat piece the same small
  probability; the skill cycles through max_pieces_per_cell of them and gives
  the cell up. Feed shape-match confidences (`confidence`) for those pieces.
- **One failure at a time, single board.** Culprit sets of up to three pairs;
  pieces that are wrong far from any failure are never suspected.
- `false_alarm`, `VERIFIED_ODDS` and the retry limits are set from the
  synthetic checks, not from robot runs.

## The constants, and why

Each constant in `scripts/backtrack.py` carries its source. In short:
`CONF_BIAS, CONF_SCORE, CONF_MARGIN = -16.73, 16.01, 51.45` with
`MARGIN_CAP = 0.15`: ridge logistic regression on 7864 (piece, listed cell)
pairs from two localizer runs over the procedural picture (`eval/calibrate.py`
reproduces them); all 844 pairs it rates over 0.9 were right, 0.5-0.9 89 %,
0.1-0.5 21 %, under 0.1 0.2 %. Right
top-1 answers had margins 0.07-0.57 (5th-95th percentile), wrong ones 0-0.02.
`VERIFIED_ODDS = 0.5`: a wrong piece passed the seam check against one right
neighbour in 46 % of 1500 tries on the classic cut (11 % wavy), a right one
always. `SEAM_FAIL = 0.06`: the neighbour-matching skill's `max_fit`.
`FORBID_P_WRONG = 0.8`, `BLAME_RATIO = 0.5`, `FALSE_ALARM = 0.05`,
`ATTEMPT_PROB_FACTOR = 0.5`: design choices, tested by the benchmark's
decoy case (a lower forbid threshold banned right pieces after one false
alarm).

## Method

Written for the hub; no code copied. The assignment is SciPy's
`scipy.optimize.linear_sum_assignment` (BSD-3-Clause, the modified
Jonker-Volgenant algorithm of Crouse 2016, called as a library, as the
jigsaw survey's choosing-next-piece-and-slot candidate proposes for forward
planning). The culprit vote, the ledger, the soft tabu and the retry ladder
are this skill's design: a tabu list of (piece, cell) pairs as in tabu search
(Glover 1989), with a Bayesian explanation of the failures instead of a fixed
"blame the newest piece" rule.
