# Adapted from dominoes by Alan Wagner (https://github.com/abw333/dominoes, commit
# ea9f532c9b834117a5c07d214711515872f7537e), MIT License, Copyright (c) 2016 Alan Wagner;
# the license is LICENSE.dominoes beside this skill. Taken from it:
# - the legal moves, dominoes/game.py Game._update_valid_moves: a tile holding an open end's
#   value may go on that end, and only once when both ends show the same value;
# - which half faces the end, dominoes/board.py Board._add_left / _add_right (the tile is
#   inverted so the matching value touches the end);
# - the players, dominoes/players.py identity, random, reverse, bota_gorda, double and their
#   composition (each player stably re-sorts the move list), and probabilistic_alphabeta:
#   deal the hidden tiles at random consistent with the values each opponent is known to
#   lack (Game.missing_values, Game.random_possible_hands), run alpha-beta on each deal
#   (dominoes/search.py alphabeta, better moves explored first) and prefer the move that is
#   most often best;
# - the end of a game and its points, Game.make_move (a player out scores every pip left;
#   a blocked game goes to the side with fewer pips, scoring all of them, a tie scores 0).
# Changed: 2-4 players (teams only for 4, the source's partners) instead of 4 fixed; a draw
# game (draw from the boneyard until a tile fits, pass only when it is empty) besides the
# block game; tiles out of play (the unseen tiles that are in no hand and not in the
# boneyard); alpha-beta with a depth and node budget and a pip evaluation instead of a
# search to the end of the game; the board is its two open ends in the world (value, edge
# point, outward direction) instead of a chain object; routed exits instead of exceptions.
# New: the world placement of the tile (straight on, round a corner when the line would
# leave the table or hit a placed tile, doubles crosswise) and the ends after the move.
# Not ported: the omniscient player (it reads the other players' hands).
"""Choose a legal domino move by matching the board's open ends, and the tile's world pose."""

from __future__ import annotations

import math
import random as _random
from collections import Counter
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

Tile = tuple[int, int]
Move = tuple[Tile, int]

TILE_SIZE = (0.048, 0.024, 0.008)
"""Tile length, width, thickness [m]: a common double-six size, the same default as
reading-domino-pips-and-orientation (US sets are ~51 x 25 x 9.5 mm; pass tile_size)."""

TOUCH_GAP = 0.001
"""Gap left between the new tile and the end it matches [m]. "Touching" on the table is
a placement a few mm off with the YAM's repeatability (PORTING.md); 1 mm keeps the
commanded footprint from overlapping the end tile, so a placing skill pushes nothing."""

EDGE_MARGIN = 0.01
"""A placed tile stays this far inside the table box [m] (half a pip-reading tile width,
so a tile that lands a few mm off still lies fully on the table)."""

OVERLAP_TOLERANCE = 0.0005
"""Footprints may touch: two rectangles overlap only past this depth [m]."""

RUN_LOOKAHEAD = 4
"""Straight tiles counted past a corner to choose between turning left and right: four
tiles (~0.2 m) separate a turn into open table from one into a pocket on a 0.5 m table."""

MAX_PIP = 6
"""Double-six set (the source's 28 tiles); pass max_pip=9 or 12 for bigger sets."""

SEARCH_SAMPLES = 24
"""Deals of the hidden tiles per "search" move (probabilistic_alphabeta's sample_size;
the source's default enumerates every deal, too slow with a boneyard). Measured win rates
in SKILL.md; 24 deals x SEARCH_NODES keep a move under ~0.5 s on one CPU core."""

SEARCH_DEPTH = 10
"""Plies searched per deal (play or forced pass); the source searches to the end."""

SEARCH_NODES = 2000
"""Alpha-beta nodes per deal before the search falls back to the evaluation."""

EVAL_WEIGHT = 0.5
"""Value of a cut-off position: EVAL_WEIGHT x (the other side's pips - my side's pips),
in the source's points (a player out scores every pip left). A hub choice: half of what
the pips would score now, since either side may still go out."""

SAMPLE_TRIES = 200
"""Shuffles tried per deal to honour every opponent's missing values before they are
ignored for that deal."""


class Output(TypedDict):
    route: str
    detail: str
    tile: list[int]
    hand_index: int
    end: int
    end_value: int
    layout: str
    place_pose: dict[str, Any]
    place_yaw_deg: float
    target_ends_world: list[list[float]]
    new_ends: list[dict[str, Any]]
    new_footprint: dict[str, Any]
    legal_moves: list[dict[str, Any]]
    ranked_moves: list[dict[str, Any]]
    search_counts: dict[str, int]
    my_pips: int
    game_over_reason: str


# ----------------------------------------------------------------------------- game rules


def tile_key(t: Any) -> Tile:
    a, b = int(t[0]), int(t[1])
    return (a, b) if a <= b else (b, a)


def pips(hand: list[Tile]) -> int:
    return sum(a + b for a, b in hand)


def full_set(max_pip: int = MAX_PIP) -> list[Tile]:
    return [(i, j) for i in range(max_pip + 1) for j in range(i, max_pip + 1)]


def legal_moves(hand: list[Tile], ends: tuple[int, int] | None) -> list[Move]:
    """Game._update_valid_moves; end -1 is the opening move on an empty board."""
    if ends is None:
        return [(t, -1) for t in hand]
    left, right = ends
    moves: list[Move] = []
    for t in hand:
        if left in t:
            moves.append((t, 0))
        if right in t and left != right:
            moves.append((t, 1))
    return moves


def ends_after(ends: tuple[int, int] | None, move: Move) -> tuple[int, int]:
    """The open values after a move (Board._add_left / _add_right)."""
    t, e = move
    if ends is None:
        return (t[0], t[1])
    v = ends[e]
    other = t[1] if t[0] == v else t[0]
    return (other, ends[1]) if e == 0 else (ends[0], other)


class GameState:
    """A whole deal: every hand, the boneyard in draw order, the open values, whose turn.

    Player 0 is the one choosing; ``side[p]`` is True for player 0's side (its partner
    in a 4-player team game). ``advance`` moves the turn on to the next player who can
    play, drawing (draw game) and passing on the way, like Game.make_move."""

    __slots__ = ("hands", "bone", "ends", "turn", "passes", "draw", "teams", "result")

    def __init__(self, hands, bone, ends, turn, passes, draw, teams, result=None):
        self.hands: list[list[Tile]] = hands
        self.bone: list[Tile] = bone
        self.ends: tuple[int, int] | None = ends
        self.turn: int = turn
        self.passes: int = passes
        self.draw: bool = draw
        self.teams: bool = teams
        self.result: tuple[str, int, int] | None = result

    def copy(self) -> GameState:
        return GameState(
            [list(h) for h in self.hands],
            list(self.bone),
            self.ends,
            self.turn,
            self.passes,
            self.draw,
            self.teams,
            self.result,
        )

    @property
    def n(self) -> int:
        return len(self.hands)

    def side_of(self, p: int) -> int:
        return p % 2 if self.teams else p

    def legal(self) -> list[Move]:
        return legal_moves(self.hands[self.turn], self.ends)

    def play(self, move: Move) -> GameState:
        g = self.copy()
        g.hands[g.turn].remove(move[0])
        g.ends = ends_after(g.ends, move)
        g.passes = 0
        if not g.hands[g.turn]:
            g.result = ("out", g.turn, sum(pips(h) for h in g.hands))
            return g
        g.advance()
        return g

    def advance(self) -> None:
        """Next player with a move; draws and passes on the way; a blocked game ends."""
        while True:
            self.turn = (self.turn + 1) % self.n
            hand = self.hands[self.turn]
            if legal_moves(hand, self.ends):
                return
            while self.draw and self.bone:
                hand.append(self.bone.pop())
                if legal_moves(hand, self.ends):
                    return
            self.passes += 1
            if self.passes >= self.n:
                self.result = self.blocked_result()
                return

    def blocked_result(self) -> tuple[str, int, int]:
        totals: dict[int, int] = {}
        for p, h in enumerate(self.hands):
            totals[self.side_of(p)] = totals.get(self.side_of(p), 0) + pips(h)
        low = min(totals.values())
        winners = [s for s, v in totals.items() if v == low]
        if len(winners) > 1:
            return ("blocked", -1, 0)
        return ("blocked", winners[0], sum(totals.values()))

    def value(self) -> float:
        """Points to player 0's side: + when it won, - when it lost (source signs)."""
        assert self.result is not None
        kind, who, points = self.result
        if kind == "out":
            mine = self.side_of(who) == self.side_of(0)
        else:
            if who < 0:
                return 0.0
            mine = who == self.side_of(0)
        return float(points if mine else -points)

    def evaluate(self) -> float:
        mine = sum(pips(h) for p, h in enumerate(self.hands) if self.side_of(p) == self.side_of(0))
        other = sum(pips(h) for p, h in enumerate(self.hands) if self.side_of(p) != self.side_of(0))
        return EVAL_WEIGHT * (other - mine)


# ----------------------------------------------------------------------------- players


def p_identity(moves: list[Move], rng: _random.Random, view: dict[str, Any]) -> list[Move]:
    return list(moves)


def p_random(moves: list[Move], rng: _random.Random, view: dict[str, Any]) -> list[Move]:
    keys = {i: rng.random() for i in range(len(moves))}
    return [m for _, m in sorted(enumerate(moves), key=lambda im: keys[im[0]])]


def p_reverse(moves: list[Move], rng: _random.Random, view: dict[str, Any]) -> list[Move]:
    return list(reversed(moves))


def p_bota_gorda(moves: list[Move], rng: _random.Random, view: dict[str, Any]) -> list[Move]:
    return sorted(moves, key=lambda m: -(m[0][0] + m[0][1]))


def p_double(moves: list[Move], rng: _random.Random, view: dict[str, Any]) -> list[Move]:
    return sorted(moves, key=lambda m: m[0][0] != m[0][1])


def deal_hidden(
    rng: _random.Random,
    unseen: list[Tile],
    counts: list[int],
    bone_n: int,
    missing: list[set[int]],
) -> tuple[list[list[Tile]], list[Tile], bool]:
    """Game.random_possible_hands with a boneyard: opponents get ``counts`` tiles none of
    which holds a value they are known to lack, the boneyard ``bone_n`` (draw order),
    the rest is out of play. False when the constraints had to be dropped."""
    order = sorted(range(len(counts)), key=lambda i: -len(missing[i]))
    for _ in range(SAMPLE_TRIES):
        pool = list(unseen)
        rng.shuffle(pool)
        hands: list[list[Tile]] = [[] for _ in counts]
        ok = True
        for i in order:
            take = [t for t in pool if not (missing[i] & set(t))][: counts[i]]
            if len(take) < counts[i]:
                ok = False
                break
            hands[i] = take
            taken = set(take)
            pool = [t for t in pool if t not in taken]
        if ok:
            return hands, pool[:bone_n], True
    pool = list(unseen)
    rng.shuffle(pool)
    hands, k = [], 0
    for c in counts:
        hands.append(pool[k : k + c])
        k += c
    return hands, pool[k : k + bone_n], False


def alphabeta(
    g: GameState, depth: int, alpha: float, beta: float, budget: list[int]
) -> tuple[float, Move | None]:
    """search.alphabeta with a depth and node budget; player 0's side maximizes."""
    if g.result is not None:
        return g.value(), None
    if depth <= 0 or budget[0] <= 0:
        return g.evaluate(), None
    budget[0] -= 1
    maximize = g.side_of(g.turn) == g.side_of(0)
    best = -math.inf if maximize else math.inf
    best_move: Move | None = None
    for m in sorted(g.legal(), key=lambda m: -(m[0][0] + m[0][1])):
        v, _ = alphabeta(g.play(m), depth - 1, alpha, beta, budget)
        if (maximize and v > best) or (not maximize and v < best):
            best, best_move = v, m
        if maximize:
            alpha = max(alpha, best)
        else:
            beta = min(beta, best)
        if beta <= alpha:
            break
    return best, best_move


def p_search(moves: list[Move], rng: _random.Random, view: dict[str, Any]) -> list[Move]:
    """probabilistic_alphabeta over sampled deals; ties keep the incoming order."""
    if len(moves) <= 1:
        return list(moves)
    counter: Counter = Counter()
    dropped = 0
    for _ in range(int(view["samples"])):
        hands, bone, ok = deal_hidden(
            rng, view["unseen"], view["counts"], view["boneyard"], view["missing"]
        )
        dropped += not ok
        g = GameState(
            [list(view["hand"])] + hands,
            bone,
            view["ends"],
            0,
            0,
            view["draw"],
            view["teams"],
        )
        _, best = alphabeta(g, int(view["depth"]), -math.inf, math.inf, [int(view["nodes"])])
        if best is not None:
            counter[best] += 1
    view["counter"] = counter
    view["dropped_deals"] = dropped
    return sorted(moves, key=lambda m: -counter[m])


PLAYERS = {
    "identity": p_identity,
    "random": p_random,
    "reverse": p_reverse,
    "bota_gorda": p_bota_gorda,
    "double": p_double,
    "search": p_search,
}

PRESETS = {
    "random": ["random"],
    "greedy": ["random", "bota_gorda"],
    "double": ["random", "bota_gorda", "double"],
    "search": ["random", "bota_gorda", "search"],
}
"""Named players: a list of source players applied in turn (each stably re-sorts, so the
last decides and the earlier ones break its ties). "greedy" is the source's bota_gorda
(biggest pip sum first), "search" its probabilistic_alphabeta with bota_gorda ordering."""


# ----------------------------------------------------------------------------- geometry


def _vec(v: Any, n: int = 3) -> np.ndarray:
    if isinstance(v, dict):
        v = [v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0)]
    a = np.zeros(n)
    vals = [float(x) for x in list(v)[:n]]
    a[: len(vals)] = vals
    return a


def _unit2(v: Any) -> np.ndarray:
    a = _vec(v)[:2]
    norm = float(np.linalg.norm(a))
    if norm < 1e-9:
        raise ValueError(f"direction {v} has no length in the table plane")
    return a / norm


def _perp(d: np.ndarray) -> np.ndarray:
    return np.array([-d[1], d[0]])


class Rect:
    """A tile footprint in the table plane: centre, unit long axis, half length, half width."""

    def __init__(self, c: np.ndarray, u: np.ndarray, hl: float, hw: float):
        self.c, self.u, self.hl, self.hw = np.asarray(c[:2], float), np.asarray(u, float), hl, hw

    def corners(self) -> np.ndarray:
        v = _perp(self.u)
        return np.array(
            [self.c + su * self.hl * self.u + sv * self.hw * v for su in (-1, 1) for sv in (-1, 1)]
        )

    def overlaps(self, other: Rect, tol: float = OVERLAP_TOLERANCE) -> bool:
        a, b = self.corners(), other.corners()
        for ax in (self.u, _perp(self.u), other.u, _perp(other.u)):
            pa, pb = a @ ax, b @ ax
            if min(pa.max(), pb.max()) - max(pa.min(), pb.min()) <= tol:
                return False
        return True

    def distance_to(self, p: np.ndarray) -> float:
        rel = np.asarray(p[:2], float) - self.c
        dx = max(abs(float(rel @ self.u)) - self.hl, 0.0)
        dy = max(abs(float(rel @ _perp(self.u))) - self.hw, 0.0)
        return math.hypot(dx, dy)


def _in_box(r: Rect, box: tuple[np.ndarray, np.ndarray] | None, margin: float) -> bool:
    if box is None:
        return True
    c = r.corners()
    return bool(np.all(c >= box[0] + margin - 1e-9) and np.all(c <= box[1] - margin + 1e-9))


def _box(table: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(table, dict):
        return None
    lo = table.get("min_m", table.get("min"))
    hi = table.get("max_m", table.get("max"))
    if lo is None or hi is None:
        return None
    return _vec(lo)[:2], _vec(hi)[:2]


class Geometry:
    def __init__(self, L, W, gap, margin, box, footprints, crosswise):
        self.L, self.W, self.gap, self.margin = L, W, gap, margin
        self.box, self.rects, self.crosswise = box, footprints, crosswise

    def fits(self, r: Rect, extra: list[Rect] = ()) -> bool:
        if not _in_box(r, self.box, self.margin):
            return False
        return not any(r.overlaps(o) for o in list(self.rects) + list(extra))

    def candidates(self, end: dict[str, Any], double: bool) -> list[dict[str, Any]]:
        """Placements against one end: straight on (crosswise for a double when the rules
        say so), then round a corner, left and right (the matching half sits square in
        front of the end, the tile runs along the turn)."""
        p2, d, z = end["p"][:2], end["d"], float(end["p"][2])
        L, W, g = self.L, self.W, self.gap
        left = _perp(d)
        out = []
        if double and self.crosswise:
            c = p2 + d * (g + W / 2)
            out.append(
                dict(
                    layout="crosswise", c=c, axis=left, new_p=p2 + d * (g + W), new_d=d, cross=True
                )
            )
        else:
            c = p2 + d * (g + L / 2)
            out.append(
                dict(layout="straight", c=c, axis=d, new_p=p2 + d * (g + L), new_d=d, cross=False)
            )
        for name, n in (("corner_left", left), ("corner_right", -left)):
            c = p2 + d * (g + W / 2) + n * (L / 2 - W / 2)
            out.append(dict(layout=name, c=c, axis=n, new_p=c + n * L / 2, new_d=n, cross=False))
        for cand in out:
            cand["z"] = z
            cand["rect"] = Rect(cand["c"], cand["axis"], L / 2, W / 2)
        return out

    def room_after(self, cand: dict[str, Any]) -> bool:
        """Whether a plain tile could still follow from the new end."""
        nxt = {"p": np.r_[cand["new_p"], cand["z"]], "d": cand["new_d"]}
        return any(self.fits(c["rect"], [cand["rect"]]) for c in self.candidates(nxt, False))

    def run_length(self, cand: dict[str, Any]) -> int:
        """Plain tiles that would fit straight on past the new end (up to RUN_LOOKAHEAD)."""
        p, d = np.asarray(cand["new_p"], float), cand["new_d"]
        extra = [cand["rect"]]
        for k in range(RUN_LOOKAHEAD):
            c = p + d * (self.gap + self.L / 2 + k * (self.gap + self.L))
            r = Rect(c, d, self.L / 2, self.W / 2)
            if not self.fits(r, extra):
                return k
            extra.append(r)
        return RUN_LOOKAHEAD

    def place(
        self, end: dict[str, Any], double: bool, keep_clear: list[Rect] = ()
    ) -> tuple[dict[str, Any] | None, bool]:
        """Straight on when it fits and leaves room for the next tile, else the corner with
        the longer straight run after it (toward the table centre on a tie), else whatever
        fits. Placements over ``keep_clear`` (the cell in front of the other open end) are
        taken only when nothing else fits."""
        cands = self.candidates(end, double)
        if self.box is not None:
            centre = (self.box[0] + self.box[1]) / 2
            turn_in = float(_perp(end["d"]) @ (centre - end["p"][:2]))
            if turn_in < 0:
                cands = [cands[0], cands[2], cands[1]]
        fitting = [c for c in cands if self.fits(c["rect"])]
        clear = [c for c in fitting if not any(c["rect"].overlaps(k) for k in keep_clear)]
        for pool in (clear, fitting):
            if pool and pool[0] is cands[0] and self.room_after(pool[0]):
                return pool[0], True
            roomy = [c for c in pool if self.room_after(c)]
            if roomy:
                return max(roomy, key=lambda c: self.run_length(c)), True
        return (fitting[0], False) if fitting else (None, False)

    def front_cell(self, end: dict[str, Any]) -> Rect:
        """Where a straight tile on this end would go."""
        p2, d = end["p"][:2], end["d"]
        return Rect(p2 + d * (self.gap + self.L / 2), d, self.L / 2, self.W / 2)


def _quat_yaw(yaw: float) -> dict[str, float]:
    return {"w": math.cos(yaw / 2), "x": 0.0, "y": 0.0, "z": math.sin(yaw / 2)}


def _l3(v: np.ndarray) -> list[float]:
    return [round(float(x), 6) for x in v]


# ----------------------------------------------------------------------------- inputs


def _hand(hand: Any, max_pip: int) -> list[dict[str, Any]]:
    out = []
    for i, rec in enumerate(hand or []):
        if isinstance(rec, dict):
            ab = rec.get("value_ab") or rec.get("value") or rec.get("tile")
        else:
            ab = rec
        if ab is None or len(ab) != 2:
            raise ValueError(f"hand tile {i} has no two values: {rec!r}")
        a, b = int(ab[0]), int(ab[1])
        if not (0 <= a <= max_pip and 0 <= b <= max_pip):
            raise ValueError(f"hand tile {i} {ab} is outside 0..{max_pip}")
        out.append({"ab": (a, b), "key": tile_key((a, b)), "index": i})
    keys = [h["key"] for h in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"the hand holds a tile twice: {keys}")
    return out


def _ends(board: dict[str, Any], max_pip: int) -> list[dict[str, Any]]:
    ends = board.get("ends") or []
    if len(ends) != 2:
        raise ValueError(f"board.ends needs the two open ends, got {len(ends)}")
    out = []
    for e in ends:
        if isinstance(e, dict):
            v = int(e["value"])
            pos = e.get("position", e.get("point", e.get("edge_world")))
            direction = e.get("direction", e.get("outward"))
            out.append(
                {
                    "value": v,
                    "p": None if pos is None else _vec(pos),
                    "d": None if direction is None else _unit2(direction),
                    "cross": bool(e.get("crosswise", False)),
                }
            )
        else:
            out.append({"value": int(e), "p": None, "d": None, "cross": False})
        if not 0 <= out[-1]["value"] <= max_pip:
            raise ValueError(f"end value {out[-1]['value']} is outside 0..{max_pip}")
    return out


def _footprints(items: Any, L: float, W: float) -> list[Rect]:
    rects = []
    for f in items or []:
        c = f.get("center", f.get("center_world", f.get("position")))
        u = f.get("axis", f.get("axis_world", f.get("direction")))
        size = f.get("size") or [f.get("length", L), f.get("width", W)]
        rects.append(Rect(_vec(c), _unit2(u), float(size[0]) / 2, float(size[1]) / 2))
    return rects


def _move_dict(m: Move, hand_by_key: dict[Tile, dict[str, Any]]) -> dict[str, Any]:
    return {
        "tile": list(hand_by_key[m[0]]["ab"]),
        "hand_index": hand_by_key[m[0]]["index"],
        "end": m[1],
    }


# ----------------------------------------------------------------------------- the skill


def run(
    ctx: NodeContext,
    hand: list[Any],
    board: dict[str, Any] | None = None,
    player: str | list[str] = "greedy",
    rules: str = "draw",
    boneyard_count: int = 0,
    opponent_counts: list[int] | None = None,
    played: list[Any] | None = None,
    opponent_missing: list[list[int]] | None = None,
    passes_in_a_row: int = 0,
    must_play: list[int] | None = None,
    teams: bool | None = None,
    max_pip: int = MAX_PIP,
    doubles_crosswise: bool = True,
    table: dict[str, Any] | None = None,
    start_pose: dict[str, Any] | None = None,
    tile_size: list[float] | None = None,
    touch_gap: float = TOUCH_GAP,
    edge_margin: float = EDGE_MARGIN,
    samples: int = SEARCH_SAMPLES,
    depth: int = SEARCH_DEPTH,
    nodes: int = SEARCH_NODES,
    seed: int | None = None,
) -> Output:
    out: Output = {
        "route": "invalid_input",
        "detail": "",
        "tile": [],
        "hand_index": -1,
        "end": -1,
        "end_value": -1,
        "layout": "",
        "place_pose": {},
        "place_yaw_deg": 0.0,
        "target_ends_world": [],
        "new_ends": [],
        "new_footprint": {},
        "legal_moves": [],
        "ranked_moves": [],
        "search_counts": {},
        "my_pips": 0,
        "game_over_reason": "",
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        return out

    try:
        tiles = _hand(hand, max_pip)
        board = board or {}
        empty = not board.get("ends")
        ends = None if empty else _ends(board, max_pip)
        L, W = (float(tile_size[0]), float(tile_size[1])) if tile_size else TILE_SIZE[:2]
        if rules not in ("draw", "block"):
            raise ValueError(f"rules must be 'draw' or 'block', not {rules!r}")
        names = PRESETS.get(player, [player]) if isinstance(player, str) else list(player)
        for nm in names:
            if nm not in PLAYERS:
                raise ValueError(
                    f"unknown player {nm!r}; presets {sorted(PRESETS)}, players {sorted(PLAYERS)}"
                )
        opp = [int(c) for c in opponent_counts] if opponent_counts is not None else None
        if "search" in names and not opp:
            raise ValueError("the search player needs opponent_counts")
    except (ValueError, KeyError, TypeError) as exc:
        return finish("invalid_input", str(exc))

    hand_keys = [t["key"] for t in tiles]
    by_key = {t["key"]: t for t in tiles}
    out["my_pips"] = pips(hand_keys)
    n_players = 1 + (len(opp) if opp else 1)
    use_teams = (n_players == 4) if teams is None else bool(teams) and n_players == 4
    draw = rules == "draw"
    played_keys = [
        tile_key(t) for t in (played if played is not None else board.get("tiles") or [])
    ]

    if not tiles:
        out["game_over_reason"] = "out"
        return finish("game_over", "the hand is empty: this player went out")
    if opp and min(opp) == 0:
        k = opp.index(0) + 1
        out["game_over_reason"] = "opponent_out"
        return finish("game_over", f"opponent {k} (in turn order) has no tiles left")

    values = None if empty else (ends[0]["value"], ends[1]["value"])
    moves = legal_moves(hand_keys, values)
    if empty and must_play is not None:
        mk = tile_key(must_play)
        moves = [m for m in moves if m[0] == mk]
        if not moves:
            return finish(
                "pass", f"the opening tile {list(mk)} is not in this hand: another player opens"
            )
    out["legal_moves"] = [_move_dict(m, by_key) for m in moves]

    if not moves:
        exhausted = False
        if played_keys and values is not None:
            gone = set(played_keys)
            exhausted = all(
                t in gone for t in full_set(max_pip) if values[0] in t or values[1] in t
            )
        if exhausted:
            out["game_over_reason"] = "blocked"
            return finish(
                "game_over",
                f"blocked: every tile holding {values[0]} or {values[1]} is on the board; my pips {out['my_pips']}",
            )
        if draw and int(boneyard_count) > 0:
            return finish(
                "draw",
                f"no tile holds {values[0]} or {values[1]}: draw from the boneyard, read it, choose again",
            )
        if int(passes_in_a_row) >= n_players - 1:
            out["game_over_reason"] = "blocked"
            return finish(
                "game_over",
                f"blocked: the other {n_players - 1} player(s) passed and this hand cannot play; my pips {out['my_pips']}",
            )
        why = "the boneyard is empty" if draw else "block game"
        return finish("pass", f"no tile holds {values[0]} or {values[1]} and {why}: pass")

    rng = _random.Random(seed)
    view: dict[str, Any] = {"hand": hand_keys, "ends": values, "draw": draw, "teams": use_teams}
    if "search" in names:
        seen = set(hand_keys) | set(played_keys)
        missing = [set(int(v) for v in m) for m in (opponent_missing or [])]
        missing += [set() for _ in range(len(opp) - len(missing))]
        view.update(
            unseen=[t for t in full_set(max_pip) if t not in seen],
            counts=opp,
            boneyard=int(boneyard_count) if draw else 0,
            missing=missing[: len(opp)],
            samples=samples,
            depth=depth,
            nodes=nodes,
        )
        hidden = sum(opp) + view["boneyard"]
        if len(view["unseen"]) < hidden:
            return finish(
                "invalid_input",
                f"{len(view['unseen'])} unseen tiles cannot fill {hidden} hidden places (opponents + boneyard); check played",
            )
    ranked = list(moves)
    for nm in names:
        ranked = PLAYERS[nm](ranked, rng, view)
    out["ranked_moves"] = [_move_dict(m, by_key) for m in ranked]
    notes = []
    if "counter" in view:
        out["search_counts"] = {
            f"{list(by_key[m[0]]['ab'])}@{m[1]}": int(view["counter"][m]) for m in ranked
        }
        if view["dropped_deals"]:
            notes.append(f"{view['dropped_deals']} deal(s) ignored the opponents' missing values")
        if not played_keys:
            notes.append("played tiles unknown: deals may hold tiles that are on the board")

    # ---- geometry
    if empty:
        box = _box(table)
        if box is None and start_pose is None:
            try:
                ws = ctx.tool("robot.describe_workspace") or {}
                box = _box(ws.get("bounds"))
            except Exception as exc:  # noqa: BLE001 - a graph without a workspace tool
                notes.append(f"no workspace ({exc})")
    else:
        box = _box(table)
        if box is None and table is None and all(e["p"] is not None for e in ends):
            try:
                ws = ctx.tool("robot.describe_workspace") or {}
                box = _box(ws.get("bounds"))
            except Exception as exc:  # noqa: BLE001
                notes.append(f"no workspace ({exc})")
    if box is None:
        notes.append("no table box: placements not checked against the table edge")
    try:
        geo = Geometry(
            L,
            W,
            float(touch_gap),
            float(edge_margin),
            box,
            _footprints(board.get("footprints"), L, W),
            bool(doubles_crosswise),
        )
    except (ValueError, KeyError, TypeError) as exc:
        return finish("invalid_input", f"footprints: {exc}")

    top = ranked[0]
    if empty:
        if start_pose is not None:
            sp = _vec(start_pose.get("position", start_pose))
            sd = _unit2(start_pose.get("direction", [1.0, 0.0]))
        elif box is not None:
            span = box[1] - box[0]
            sp = np.r_[(box[0] + box[1]) / 2, 0.0]
            sd = np.array([1.0, 0.0]) if span[0] >= span[1] else np.array([0.0, 1.0])
        else:
            return finish(
                "invalid_input",
                "an empty board needs start_pose, table or robot.describe_workspace",
            )
        t = top[0]
        double = t[0] == t[1]
        ab = by_key[t]["ab"]
        if double and doubles_crosswise:
            axis, half, cross, layout = _perp(sd), W / 2, True, "crosswise"
        else:
            axis, half, cross, layout = sd, L / 2, False, "straight"
        c2 = sp[:2]
        rect = Rect(c2, axis, L / 2, W / 2)
        if not geo.fits(rect):
            out.update(tile=list(ab), hand_index=by_key[t]["index"])
            return finish(
                "no_room",
                f"the opening tile does not fit at {_l3(sp)} inside the table and clear of the footprints",
            )
        a_end, b_end = c2 - axis * L / 2, c2 + axis * L / 2
        new_ends = [
            {
                "value": ab[0],
                "position": _l3(np.r_[c2 - sd * half, sp[2]]),
                "direction": _l3(-sd),
                "crosswise": cross,
            },
            {
                "value": ab[1],
                "position": _l3(np.r_[c2 + sd * half, sp[2]]),
                "direction": _l3(sd),
                "crosswise": cross,
            },
        ]
        z = float(sp[2])
        return _played(
            out,
            finish,
            top,
            by_key,
            -1,
            -1,
            layout,
            c2,
            axis,
            z,
            a_end,
            b_end,
            new_ends,
            L,
            W,
            notes,
        )

    if any(e["p"] is None or e["d"] is None for e in ends):
        t = top[0]
        e = top[1] if values[0] != values[1] else 0
        out.update(
            tile=list(by_key[t]["ab"]), hand_index=by_key[t]["index"], end=e, end_value=values[e]
        )
        notes.append("the ends carry no world position and direction: no placement computed")
        return finish(
            "play",
            f"play {list(by_key[t]['ab'])} on end {e} (value {values[e]}); " + "; ".join(notes),
        )

    fallback = None
    chosen = None
    for m in ranked:
        t, e0 = m
        for e in [0, 1] if values[0] == values[1] else [e0]:
            cand, room = geo.place(ends[e], t[0] == t[1], [geo.front_cell(ends[1 - e])])
            if cand is None:
                continue
            if room:
                chosen = (m, e, cand)
                break
            if fallback is None:
                fallback = (m, e, cand)
        if chosen:
            break
    chosen = chosen or fallback
    if chosen is None:
        t = top[0]
        out.update(
            tile=list(by_key[t]["ab"]),
            hand_index=by_key[t]["index"],
            end=top[1],
            end_value=values[max(top[1], 0)],
        )
        return finish(
            "no_room",
            "no legal move has a placement inside the table and clear of the placed tiles; "
            f"the player's choice was {list(by_key[t]['ab'])} on end {top[1]}",
        )
    m, e, cand = chosen
    if m != top:
        notes.append(
            f"the player's first choice {list(by_key[top[0]]['ab'])}@{top[1]} had no placement with room after it"
        )
    if chosen is fallback:
        notes.append("no move leaves room for a following tile on this end")
    t = m[0]
    ab = by_key[t]["ab"]
    v = ends[e]["value"]
    axis = cand["axis"]
    c2 = cand["c"]
    if ab[0] == v:  # end a of the tile is the matching half, toward the end
        a_end, b_end = c2 - axis * L / 2, c2 + axis * L / 2
    else:
        a_end, b_end = c2 + axis * L / 2, c2 - axis * L / 2
    outer = ab[1] if ab[0] == v else ab[0]
    new_end = {
        "value": outer,
        "position": _l3(np.r_[cand["new_p"], cand["z"]]),
        "direction": _l3(cand["new_d"]),
        "crosswise": cand["cross"],
    }
    new_ends = []
    for k in (0, 1):
        if k == e:
            new_ends.append(new_end)
        else:
            o = ends[k]
            new_ends.append(
                {
                    "value": o["value"],
                    "position": _l3(o["p"]),
                    "direction": _l3(o["d"]),
                    "crosswise": o["cross"],
                }
            )
    return _played(
        out,
        finish,
        m,
        by_key,
        e,
        v,
        cand["layout"],
        c2,
        axis,
        cand["z"],
        a_end,
        b_end,
        new_ends,
        L,
        W,
        notes,
    )


def _played(
    out, finish, m, by_key, e, v, layout, c2, axis, z, a_end, b_end, new_ends, L, W, notes
) -> Output:
    t = m[0]
    ab = by_key[t]["ab"]
    yaw = math.atan2(float(b_end[1] - a_end[1]), float(b_end[0] - a_end[0]))
    out.update(
        tile=list(ab),
        hand_index=by_key[t]["index"],
        end=e,
        end_value=v,
        layout=layout,
        place_pose={
            "position": {"x": float(c2[0]), "y": float(c2[1]), "z": float(z)},
            "rotation": _quat_yaw(yaw),
        },
        place_yaw_deg=round(math.degrees(yaw), 4),
        target_ends_world=[_l3(np.r_[a_end, z]), _l3(np.r_[b_end, z])],
        new_ends=new_ends,
        new_footprint={"center": _l3(np.r_[c2, z]), "axis": _l3(axis), "length": L, "width": W},
    )
    where = "to open the board" if e < 0 else f"on end {e} (value {v})"
    why = f"play {list(ab)} {where}, {layout} at ({c2[0]:.3f}, {c2[1]:.3f}) yaw {math.degrees(yaw):.1f} deg"
    return finish("play", "; ".join([why] + notes))
