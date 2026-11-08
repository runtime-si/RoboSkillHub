---
name: choosing-a-domino-move-by-end-matching
description: Chooses the robot's move in a game of dominoes and where the tile goes on
  the table. From the hand (values as reading-domino-pips-and-orientation reads them),
  the board's two open ends (value, world edge point, outward direction), the other
  players' tile counts and the boneyard, it lists the legal moves (a tile holding an
  end's value goes on that end, as abw333/dominoes does), ranks them with a player -
  random, greedy (biggest pip sum), doubles first, or search (the source's
  probabilistic alpha-beta over deals of the hidden tiles) - and returns the tile, the
  end and the tile's world pose with its matching half touching that end - straight
  on, round a corner before the table edge or a placed tile, doubles crosswise. With
  no legal move it routes draw, pass or game_over. Block or draw rules, 2-4 players,
  double-six or bigger sets. No motion. Use on the robot's turn in a domino game,
  after reading the hand and the board ends, before placing the tile.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [domino, game, end-matching, alpha-beta, placement, layout, plan]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/abw333/dominoes/blob/ea9f532c9b834117a5c07d214711515872f7537e/dominoes/players.py
      upstream_license: MIT
gap:
  requires: {connector: [robot.describe_workspace]}
  allowed_tools:
    - robot.describe_workspace
  required_inputs: {}
  # Inputs (not listed: gap has no list or open-dict type here, so they are wired from a
  # script node in the same graph): hand (list of [a, b] or reading-domino-pips-and-
  # orientation tile records), board ({ends: [two {value, position, direction,
  # crosswise}], footprints, tiles}; empty for the opening), player, rules,
  # boneyard_count, opponent_counts, played, opponent_missing, passes_in_a_row,
  # must_play, teams, max_pip, doubles_crosswise, table, start_pose, tile_size,
  # touch_gap, edge_margin, samples, depth, nodes, seed.
  produces_outputs:
    hand_index: int
    end: int
    end_value: int
    layout: string
    place_pose: Se3Pose
    place_yaw_deg: float
    my_pips: int
    game_over_reason: string
  exit_conditions:
    play: A legal move was chosen and fits; `tile` (the hand's [a, b], a at the tile's end a), `hand_index`, `end` (0 / 1, -1 for the opening), `layout` (straight / crosswise / corner_left / corner_right), `place_pose` (tile centre, yaw of the a-to-b axis), `target_ends_world` (where end a and end b go), `new_ends` and `new_footprint` for the next turn. Ends given without positions give the move with an empty pose.
    draw: No tile holds either end value, the rules are `draw` and the boneyard has tiles; draw one, read it, and run this skill again.
    pass: No tile holds either end value and the boneyard is empty (or the block game); also when the opening tile `must_play` is not in this hand.
    game_over: The hand is empty (`out`), an opponent has no tiles (`opponent_out`), or the game is blocked (`blocked`) - every other player passed in a row and this hand cannot play either, or every tile holding either end value is in `played`.
    no_room: Legal moves exist but none has a placement inside the table (less `edge_margin`) and clear of the placed footprints, straight or round a corner; `tile` / `end` carry the player's choice. Make room (slide the line) or play it by hand.
    invalid_input: A hand tile or end value outside 0..max_pip, a tile twice, not two ends, an end direction with no length, an unknown player or rules, search without opponent_counts, or fewer unseen tiles than the hidden places (also the route of a raised error).
  canonical_scripts:
    - choose_move: scripts/choose_move.py
  examples:
    - title: Choose the robot's domino move and the tile's world pose
      path: examples/canonical_subgraph.json
---

# choosing-a-domino-move-by-end-matching

The game half of playing dominoes with the YAM: given what the robot sees, which
tile, on which end, turned which way, and where exactly it goes on the table.

## The loop

1. `reading-domino-pips-and-orientation` on the robot's hand: each tile's
   `value_ab` (a at `ends_world[0]`) and pose. On the board: the two end tiles'
   values and poses (the open value is the half farther from the line).
2. This skill. `board.ends[k]` is `{value, position, direction, crosswise}`:
   `position` is the midpoint of the end tile's open edge (where the next tile
   touches), `direction` the outward unit vector along the line, `crosswise`
   whether the end tile is a double lying across. `board.footprints` are the
   placed tiles (`center`, `axis`, optional `length` / `width`; the reader's
   `center_world` / `axis_world` are accepted). `new_ends` / `new_footprint`
   of a `play` are the board after it, if the next turn does not re-read.
3. `play`: pick the tile at `hand_index` and place it at `place_pose`, end a
   of the tile at `target_ends_world[0]` (the placer turns the tile so its a-to-b
   axis has yaw `place_yaw_deg`). `draw`: take a boneyard tile, read it, run
   again. `pass` / `game_over`: the opponents' turn / the end.

The opponents' moves come from re-reading the board; their tile counts, the
boneyard count and the values each opponent passed or drew on
(`opponent_missing`) are bookkeeping in the graph.

## What it does

1. **Legal moves** (the source's `Game._update_valid_moves`): a tile holding the
   value of end 0 may go on end 0, one holding end 1's on end 1; when both ends
   show the same value a tile is listed once (geometry then tries both ends).
   The opening (empty board) may play any tile, or only `must_play`.
2. **Players** (`players.py`): each re-sorts the move list stably, and they are
   applied in turn, so the last one decides and the earlier ones break its
   ties. Presets: `random`; `greedy` = random, bota_gorda (biggest pip sum
   first); `double` = random, bota_gorda, double (doubles first); `search` =
   random, bota_gorda, search. A list of the source names (`identity`,
   `random`, `reverse`, `bota_gorda`, `double`, `search`) composes them freely.
3. **Search** (`probabilistic_alphabeta`): `samples` (24) times, deal the
   unseen tiles (the set less this hand and `played`) to the opponents (their
   counts), the boneyard and out of play, so that no opponent holds a value it
   passed or drew on (`Game.missing_values`; with draw-until-it-fits the values
   stay missing, since the tile that fits is played at once). Run alpha-beta on
   the deal with full information (my side maximizes; with four players the
   partners are seats 0 and 2), bota_gorda ordering, `depth` (10) plies and
   `nodes` (2000) per deal, the position past the budget valued at 0.5 x (the
   other side's pips - mine). The move that is best in most deals wins.
4. **Game end** (`Game.make_move`): a player out scores every pip left; a
   blocked game goes to the side with fewer pips (a tie scores nothing). The
   skill routes `game_over` when it sees the end; the scores are in `my_pips`
   and the counts the graph keeps.
5. **Placement** (new). Against an end at edge point P going out along d, with
   the tile L x W and the gap g (1 mm):
   - straight: centre P + d (g + L/2), long axis along d;
   - crosswise (a double, `doubles_crosswise`): centre P + d (g + W/2), long
     axis across d; the next tile meets its long side;
   - corner (turn by n = d turned +-90 deg): the matching half sits square in
     front of the end, centre P + d (g + W/2) + n (L/2 - W/2), long axis along
     n; the line then goes on along n.
   The matching half is always the one nearest P (the source's `Board.add`
   inversion), so `target_ends_world` puts the tile's a end where it belongs.
   A placement must lie inside the table box less `edge_margin` (10 mm) and
   overlap no footprint by more than 0.5 mm. It is taken in this order:
   straight on (or crosswise) when it fits and a tile could still follow; else
   the corner, of those that fit and leave room, with the longer straight run
   after it (up to 4 tiles; toward the table centre on a tie); else whatever
   fits. Placements over the cell in front of the other open end are used only
   when nothing else fits, so one branch does not wall the other in. The player's ranking is walked until a move has such a
   placement; if the player's first choice was skipped, `detail` says so. The
   table is `table` (`{min, max}` in x, y), else `robot.describe_workspace`'s
   `bounds`; the opening goes at `start_pose` or the table centre, along the
   table's longer side.

## Inputs

- `hand`: `[[a, b], ...]` or tile records with `value_ab` (or `value`).
- `board`: `{}` for the opening, else `ends` (above), `footprints`, and
  optionally `tiles` (the played tiles, used when `played` is not given).
- `player` (`greedy`), `rules` (`draw` | `block`), `boneyard_count`,
  `opponent_counts` (the others in turn order after this player; needed by
  `search`), `played`, `opponent_missing` (per opponent, the values it lacks),
  `passes_in_a_row` (consecutive passes just before this turn), `must_play`,
  `teams` (default: with four players), `max_pip` (6).
- `table`, `start_pose` (`{position, direction}`), `tile_size` ([L, W],
  0.048 x 0.024 m, the reader's default), `touch_gap` (0.001 m),
  `edge_margin` (0.01 m), `doubles_crosswise` (true).
- `samples`, `depth`, `nodes`, `seed` for the players' randomness.

## Outputs

`tile`, `hand_index`, `end`, `end_value`, `layout`, `place_pose`,
`place_yaw_deg`, `target_ends_world`, `new_ends`, `new_footprint`,
`legal_moves` and `ranked_moves` (`{tile, hand_index, end}`, the player's
order), `search_counts` (deals in which each move was best), `my_pips`,
`game_over_reason`, `detail`.

## The constants, and why

- **`TILE_SIZE` 48 x 24 mm**: as reading-domino-pips-and-orientation.
- **`TOUCH_GAP` 1 mm**: the tiles should touch, but a commanded footprint that
  overlaps the end tile makes the placer push it; the YAM places to a few mm
  without visual servoing, so 1 mm is below what the arm resolves anyway.
- **`EDGE_MARGIN` 10 mm**: a tile placed a few mm off still lies on the table.
- **`OVERLAP_TOLERANCE` 0.5 mm**: numerically touching footprints are not
  overlapping.
- **`RUN_LOOKAHEAD` 4 tiles** (~0.2 m): enough to tell a turn into open table
  from one into a pocket on the station's 0.5 x 0.45 m box.
- **Search budget `samples` 24, `depth` 10, `nodes` 2000**: see Measured; the
  source enumerates every deal and searches to the end, which with a boneyard
  is many seconds a move in Python. `EVAL_WEIGHT` 0.5 is a hub choice (half
  of the points the pips would score now).

## Measured

`tests/bundles/_domino_game.py` plays whole games in which every player's turn is
this skill's `run` with only that player's information (its hand, the two world
ends and the placed footprints, played tiles, counts, missing values). A referee
checks every answer against its own rules and every placement: inside the real
YAM station's tool box (x -0.15..0.35 m, y 0.10..0.55 m, from
`real/motion_server.py`) less 10 mm, overlapping no placed tile, 1 mm from the
end it matches, the matching half nearest it. Seats alternate; seeds 1000+.

**Legality: 100 %.** Every answer in every game below and in the placement
games (the opponents' included, since they are this skill too) was a legal play, or a
draw / pass / game_over when no tile fitted; the referee raised on none.

**Win rates** (A's side; 4 players: A holds seats 0 and 2; 3 players: one A
against two B, so chance is 1/3; a tie counts half; games ending `no_room` left
out). Points objective and budget as shipped (24 deals, 10 plies, 2000 nodes).

| rules | players | A vs B | games | A wins / ties / losses | no_room | A win rate | s per answer |
|---|---|---|---|---|---|---|---|
| draw | 2 | random vs random | 400 | 193 / 4 / 201 | 2 | 0.490 | 0.0006 |
| draw | 2 | greedy vs random | 400 | 196 / 0 / 203 | 1 | 0.491 | 0.0006 |
| draw | 2 | double vs random | 400 | 252 / 0 / 147 | 1 | 0.632 | 0.0006 |
| draw | 2 | search vs random | 200 | 128 / 0 / 72 | 0 | 0.640 | 0.005 |
| draw | 2 | search vs greedy | 200 | 120 / 1 / 79 | 0 | 0.603 | 0.006 |
| block | 2 | greedy vs random | 400 | 220 / 15 / 165 | 0 | 0.569 | 0.0008 |
| block | 2 | search vs random | 200 | 123 / 6 / 71 | 0 | 0.630 | 0.007 |
| block | 2 | search vs greedy | 200 | 110 / 5 / 85 | 0 | 0.562 | 0.008 |
| block | 4 | greedy vs random | 400 | 198 / 9 / 193 | 0 | 0.506 | 0.002 |
| block | 4 | search vs random | 200 | 143 / 2 / 55 | 0 | 0.720 | 0.021 |
| block | 4 | search vs greedy | 200 | 129 / 6 / 65 | 0 | 0.660 | 0.015 |
| draw | 3 | greedy vs random | 400 | 157 / 6 / 237 | 0 | 0.400 (chance 0.333) | 0.001 |

Greedy (bota gorda) sheds pips, which is what the source's points reward, but it
wins no more often than random in the draw game or with four players; doubles
first does (0.63), since a double can only go on one value and is the tile most
likely to be stuck. Search wins 0.60-0.72 against both. Tried and not kept: a
win-first search objective (terminal +-1000 + points, cut-off by tile counts)
gave 0.668 / 0.690 (draw, 2 players, vs random / greedy, 150 games) and 0.687 /
0.651 (block, 4) against 0.657 / 0.700 and 0.707 / 0.695 for the points
objective - no consistent gain. Budget (search vs greedy, 120 games): 8 deals /
6 plies / 500 nodes 0.637 (draw 2) and 0.617 (block 4); 24 / 10 / 2000 0.650 and
0.646; 48 / 14 / 6000 0.688 and 0.571 at 7x the time.

**Fidelity to the source.** Against the upstream `dominoes` package (installed
from the same commit): `legal_moves` equalled `Game.valid_moves` in 1013 / 1013
positions of 60 random 4-player games, and `alphabeta` without a budget gave the
same value as `dominoes.search.alphabeta` in 55 / 55 late positions (12 tiles
left), its best move reaching that value in all 55.

**Placement.** Random play, 500 games each of draw-2, block-4, draw-3 and
block-2 (36 039 placements: 24 333 straight, 5 404 crosswise, 6 189 corner left,
113 corner right): every placement inside the box and clear of the others, and
1 game of the 2000 ended `no_room`. In the 400-game rows above, 4 of 4000 did.
The failures are pockets: one branch of the line turned back alongside itself
and left the next tile no cell (the 4-tile run lookahead does not see them).
An earlier version without the run lookahead and the kept-clear cell in front of
the other end let one branch wall the other end in (1 of 200 games).

The opponents' turns in these games are this skill too; human opponents and the
real perception and placing errors were not part of any run.

## Not done

- No robot motion and no perception: the hand and the ends come from
  reading-domino-pips-and-orientation, the pick and place from a placing skill
  (placing-with-contact-descent or transporting-objects); neither was run in
  this loop. The sim check of PORTING.md does not apply (the skill does not
  move the arm).
- Spinner games (a first double open on four sides), Mexican Train, all-fives
  scoring and drawing while holding a playable tile are not modelled.
- The source's `omniscient` player (it reads the other hands) is not ported.
- OpenSpiel's block dominoes (Apache-2.0) was read for the 2-player block
  rules (7 tiles each, 14 out of play), which `rules: block` with one
  opponent covers; no OpenSpiel code is used.

## Provenance

Adapted from abw333/dominoes (MIT, Alan Wagner, commit ea9f532):
`dominoes/game.py` (`_update_valid_moves`, `make_move`, `missing_values`,
`random_possible_hands`), `dominoes/board.py` (`_add_left` / `_add_right`),
`dominoes/players.py` (`random`, `reverse`, `bota_gorda`, `double`,
`probabilistic_alphabeta`) and `dominoes/search.py` (`alphabeta`). Changed: 2-4
players with optional teams instead of 4 partners; the draw game with a
boneyard; tiles out of play; a depth / node budget and an evaluation in the
search; the board as two world ends; routed exits. New: the world placement
and the table / footprint checks. The license is `LICENSE.dominoes` beside
this file.
