---
name: finding-the-longest-domino-chain
description: Finds the longest chain of matching dominoes a hand can lay from a start
  value - the Mexican Train move of building your own train from the engine double or
  your train's open end, or a solitaire end-to-end line - and lays it out on the table.
  Tiles are edges between pip values; every chain is walked depth first (as
  RGBz/mexican-train-domino-solver does) with an exact bound that cuts branches which
  cannot win, ranked longest, then most pips, then not ending on a double (a double
  left open must be covered). Returns the chain with each tile turned so its matching
  half comes first, the leftover tiles, the open value at the end and, from the train
  end's world point and direction, every tile's place pose and end points - straight
  on, round a corner before the table edge or a placed tile, doubles crosswise. No
  motion. Use on a Mexican Train turn or to plan a matching-ends line for the arm to
  build, after reading the tiles' values.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [domino, game, mexican-train, longest-path, end-matching, layout, plan]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/RGBz/mexican-train-domino-solver/blob/ac758e87bbae433a5a10ba4a6907874be17d80d6/src/Graph.ts
      upstream_license: MIT
gap:
  requires: {connector: [robot.describe_workspace]}
  allowed_tools:
    - robot.describe_workspace
  required_inputs: {}
  # Inputs (not listed: gap has no list, union or open-dict type here, so they are wired
  # from a script node in the same graph): hand (list of [a, b] or
  # reading-domino-pips-and-orientation tile records), start_value (int; None tries every
  # value in the hand), start_end ({value, position, direction, crosswise}: the train's
  # open end as choosing-a-domino-move-by-end-matching takes it; its value is the start
  # value when start_value is not given), objective (longest | most_pips), max_pip,
  # max_nodes, doubles_crosswise, footprints, table, tile_size, touch_gap, edge_margin.
  produces_outputs:
    start_value: int
    length: int
    chain_pips: int
    end_value: int
    ends_in_double: bool
    leftover_pips: int
    placed_count: int
    nodes: int
    exhaustive: bool
    length_bound: int
    layout_tries: int
  exit_conditions:
    chain_found: A chain from the start value was found and, when start_end has a position, every tile has a placement inside the table and clear of the placed tiles; `chain` (per tile - `tile` as held, `hand_index`, `near` / `far` value, `layout`, `place_pose`, `place_yaw_deg` of the a-to-b axis, `target_ends_world`), `chain_tiles` ([near, far]), `leftover`, `new_end` (the train's end after the chain) and `footprints_after` are filled. `exhaustive` false means the node budget stopped the search; `length_bound` tells whether the length is still proven longest.
    no_chain: The hand is empty or no tile holds the start value; `leftover` is the hand.
    no_room: The chain was found but tile `placed_count` has no placement inside the table (less `edge_margin`) and clear of `footprints` and the tiles before it, straight or round a corner; the first `placed_count` tiles are placed and `new_end` is the end they leave.
    invalid_input: A tile outside 0..max_pip or with not two values, a tile twice, an unknown objective, a start value outside 0..max_pip, a start direction with no length in the table plane, or a footprint without an axis (also the route of a raised error).
  canonical_scripts:
    - find_chain: scripts/find_chain.py
  examples:
    - title: Find and lay out the longest train from the engine
      path: examples/canonical_subgraph.json
---

# finding-the-longest-domino-chain

The planning half of a Mexican Train turn (or of a solitaire "use as many tiles as
you can" line): which of my tiles, in which order, turned which way, and where each
one goes on the table.

## The loop

1. `reading-domino-pips-and-orientation` on the hand: each tile's `value_ab`.
   On the board: the engine (or your train's last tile) - its open value, the
   midpoint of its open edge and the outward direction, as `start_end`.
2. This skill. `chain_found`: place `chain[k]` for k = 0, 1, ...: pick the hand tile
   `hand_index`, put it at `place_pose` with its a end at `target_ends_world[0]`.
   `new_end` and `footprints_after` are the board after the chain.
3. In Mexican Train the leftover tiles go on the other trains;
   `choosing-a-domino-move-by-end-matching` chooses single plays on two open ends.

## What it does

1. **Graph** (`Graph.ts`): pip values are the nodes, tiles the edges (a double is a
   loop). A chain is a trail from the start value: each tile once, each tile holding
   the value the previous one left open (`Domino.getOtherSide`).
2. **Search** (`Graph.findMostValuableChain`): depth first over every chain from the
   start value; with no start value every value in the hand is tried.
3. **Ranking** (`compareChains`): the longer chain; on equal length the one with more
   pips; then the one that does not end on a double (an open double must be covered in
   Mexican Train). `objective: most_pips` puts pips before length (to shed points late
   in a round).
4. **Bound** (new, exact): at each node the unused tiles in the open value's connected
   component, less one non-double for every pair of odd-degree values beyond the
   chain's two ends (a chain is an Euler trail of the tiles it uses; an even start
   leaves one more pair), bound the length and the pips a chain can still reach; a
   branch that cannot strictly beat the best chain so far is cut. A full double-six
   set is proven one 28-tile chain in 121 nodes.
5. **Budget** (new): after `max_nodes` (200 000) tiles tried the search stops and
   returns its best chain with `exhaustive: false`; `length_bound` is the root bound,
   so a chain that long is still proven longest.
6. **Placement** (new; the geometry of `choosing-a-domino-move-by-end-matching`):
   from the train end at edge point P going out along d, tile L x W, gap g (1 mm):
   straight (centre P + d (g + L/2)), a double crosswise (centre P + d (g + W/2)),
   or round a corner (the matching half square in front of the end, the tile along
   d turned +-90 deg). Straight when it fits and the next tile could follow, else the
   corner with the longer straight run after it (toward the table centre on a tie).
   Inside the table box (`table`, else `robot.describe_workspace` bounds) less
   `edge_margin` (10 mm), overlapping no footprint by more than 0.5 mm. Since the
   whole chain is known, the layout is searched depth first in that order (the first
   leaf is the greedy layout) until every tile fits or `layout_tries` (20 000)
   placements were tried.

## Inputs

- `hand`: `[[a, b], ...]` or tile records with `value_ab` (or `value`).
- `start_value`: the engine double's value or the train's open value; `None` lets the
  chain start anywhere (solitaire).
- `start_end`: `{value, position, direction, crosswise}`; without a position the chain
  is returned without placement.
- `objective` (`longest`), `max_pip` (12, double-twelve), `max_nodes` (200 000),
  `doubles_crosswise` (true), `footprints` (placed tiles: `center`, `axis`, optional
  `length` / `width`), `table` (`{min, max}`), `tile_size` ([L, W], 0.048 x 0.024 m),
  `touch_gap` (0.001 m), `edge_margin` (0.01 m), `layout_tries` (20 000).

## The constants, and why

- **`TILE_SIZE` 48 x 24 mm, `TOUCH_GAP` 1 mm, `EDGE_MARGIN` 10 mm,
  `OVERLAP_TOLERANCE` 0.5 mm, `RUN_LOOKAHEAD` 4**: as
  choosing-a-domino-move-by-end-matching (the reader's tile, touching without
  pushing, a few mm of placing error).
- **`MAX_PIP` 12**: the source's `Domino.from` takes 0..12 (Mexican Train is played
  with a double-twelve set).
- **`MAX_NODES` 200 000**: about 0.3 s in Python here; every random 15-tile
  double-twelve or double-nine hand finished far below it (see Measured).
- **`LAYOUT_TRIES` 20 000**: no measured layout needed more than 31; the cap only
  bounds the time when no layout exists.

## Measured

**Fidelity to the source.** The upstream TypeScript (commit ac758e8, compiled with
tsc 4.9, run under node 24) and `best_chain` on the same random hands, half of them
turned, start value random or none: the same (length, pips, ends-on-a-double) in
**1440 / 1440** hands - 300 of 8 tiles (double-six), 300 of 10 (double-four), 300
of 12 and 300 of 15 (double-twelve), 200 of 15 (double-nine), 200 of 12 and 40 of
16 (double-six), every chain legal (starts at the start value, each tile once, ends
matching). The unit tests also check the search against brute force over every
ordered selection on 150 small hands.

**Search size** (default budget, start value random;
`tests/bundles/_domino_chain_eval.py`):

| set | hand | hands | exhaustive | nodes median / max | ms median / max |
|---|---|---|---|---|---|
| double-12 | 15 | 2000 | 2000 | 58 / 1725 | 0.1 / 3 |
| double-9 | 15 | 500 | 500 | 311 / 4611 | 0.5 / 7 |
| double-6 | 16 | 100 | 100 | 13 410 / 111 915 | 19 / 160 |
| double-6 | 20 | 50 | 12 | 200 000 (budget) | 295 / 376 |
| double-6 | 24 | 20 | 3 | 200 000 (budget) | 308 / 346 |
| double-6 | 28 (full set) | 5 | 5 | 121 / 160 | 0.4 |

Where the budget stopped the search (dense 20-24 tile double-six hands), the chain's
length equalled the root bound in 50 / 50 (20 tiles) and 19 / 20 (24 tiles) hands: the
length is proven longest, only the pip tie-break is not. The upstream walk was
1.0-2.5x faster than the port on small hands (node vs CPython) and no faster on the
16-tile double-six hands (median 18.7 vs 17.5 ms); it has no budget, so dense hands
run until every chain is walked.

**Placement** (same script; the real YAM station's tool box x -0.15..0.35, y 0.10..0.55
m from `real/motion_server.py`, less 10 mm; start ends at random points in it facing
+-x or +-y). Every placement was checked: inside the box, no two footprints
overlapping, the matching half toward the previous tile.

| set | hand | starts | with a chain | all placed (greedy alone) | no first tile fits | stopped part-way | placements tried median / max | ms median / max |
|---|---|---|---|---|---|---|---|---|
| double-12 | 15 | 500 | 450 | 418 (418) | 32 | 0 | 10 / 15 | 8 / 35 |
| double-9 | 15 | 300 | 284 | 263 (263) | 21 | 0 | 12 / 15 | 12 / 26 |
| double-6 | 28 | 100 | 100 | 95 (94) | 5 | 0 | 28 / 30 | 55 / 331 |

Every `no_room` was a start end facing the table edge with no room for even the first
tile. The full double-six set from (-0.12, 0.32) along +x is the case the greedy layout
(choosing-a-domino-move-by-end-matching's rule, one tile at a time) loses: it spirals
into its own start at tile 27 of 28; backtracking lays all 28 after 31 placements.

## Not done

- No robot motion and no perception: the hand comes from
  reading-domino-pips-and-orientation and the tiles are placed by a placing skill
  (placing-with-contact-descent or transporting-objects); none was run in this loop.
  The sim check of PORTING.md does not apply (the skill does not move the arm).
- Mexican Train rules beyond the chain are not modelled (the public train, covering a
  double someone else left open, drawing). A double inside the chain is placed
  crosswise and the chain goes on straight from its far side, as on a real table.
- Among chains of the best rank the first found is kept; the search does not prefer
  one that fits the table, so a chain can route `no_room` while another equally long
  one would have fitted.

## Provenance

Adapted from mexican-train-domino-solver (MIT, Anton C Beza Jr, commit ac758e8):
`src/Graph.ts` (`findMostValuableChain`), `src/findMostValuableChain.ts`
(`compareChains`), `src/Domino.ts` (`getOtherSide`). Changed: Python; an exact
branch-and-bound and a node budget instead of walking every chain; the connected
component split (`Graph.generateAll`) replaced by the bound; a pips-first objective;
tiles turned so the matching half comes first; routed exits. New: the world layout
(the geometry of choosing-a-domino-move-by-end-matching). The license is
`LICENSE.mexican-train-domino-solver` beside this file.
