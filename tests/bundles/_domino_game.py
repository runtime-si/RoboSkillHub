"""Whole domino games played through choosing-a-domino-move-by-end-matching (CPU).

    .venv/bin/python -m tests.bundles._domino_game [--games 200]

Every player's turn calls the skill's ``run`` with only what that player can see (its
hand, the board's two world ends and placed footprints, the played tiles, the other
players' tile counts and the values they passed or drew on, the boneyard count). The
referee here checks each answer against its own rules - a played tile is in the hand and
matches the end; draw / pass / game_over only when no tile fits - and each placement:
inside the table box, clear of every placed tile, the touch gap from the end it matches,
the matching half nearest that end. The win rates in the skill's SKILL.md come from here.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import time
from pathlib import Path

import numpy as np
from gap.testing import FakeContext

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skills/choosing-a-domino-move-by-end-matching/scripts/choose_move.py"

TABLE = {"min_m": [-0.15, 0.10], "max_m": [0.35, 0.55]}
"""The real YAM station's tool-centre box in x, y (real/motion_server.py BOUNDS)."""


def load():
    spec = importlib.util.spec_from_file_location("choose_move", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class IllegalAnswer(AssertionError):
    pass


def play_game(
    mod,
    players: list,
    rules: str = "draw",
    hand_size: int = 7,
    seed: int = 0,
    table: dict | None = TABLE,
    max_pip: int = 6,
    starter: int | None = None,
    skill_kw: dict | None = None,
) -> dict:
    """One game; returns the result, the counters and every placed footprint."""
    rng = random.Random(seed)
    n = len(players)
    tiles = mod.full_set(max_pip)
    rng.shuffle(tiles)
    # physical orientation of each tile in the hand is random: (a, b) with a at end a
    tiles = [t if rng.random() < 0.5 else (t[1], t[0]) for t in tiles]
    hands = [tiles[k * hand_size : (k + 1) * hand_size] for k in range(n)]
    rest = tiles[n * hand_size :]
    bone = list(rest) if rules == "draw" else []
    teams = n == 4
    ends: list[dict] | None = None
    footprints: list[dict] = []
    played: list[list[int]] = []
    missing = [set() for _ in range(n)]
    turn = rng.randrange(n) if starter is None else starter
    passes = 0
    stats = {
        "plays": 0,
        "draws": 0,
        "passes": 0,
        "no_room": 0,
        "layouts": {},
        "answers": 0,
        "seconds": 0.0,
    }
    kw = dict(skill_kw or {})
    gap = kw.get("touch_gap", mod.TOUCH_GAP)
    margin = kw.get("edge_margin", mod.EDGE_MARGIN)
    box = (np.array(table["min_m"]), np.array(table["max_m"])) if table else None
    for _ in range(400):
        me = turn
        others = [(me + k) % n for k in range(1, n)]
        board = {} if ends is None else {"ends": ends, "footprints": footprints, "tiles": played}
        t0 = time.perf_counter()
        res = mod.run(
            FakeContext(),
            hand=[list(t) for t in hands[me]],
            board=board,
            player=players[me],
            rules=rules,
            boneyard_count=len(bone),
            opponent_counts=[len(hands[o]) for o in others],
            played=played,
            opponent_missing=[sorted(missing[o]) for o in others],
            passes_in_a_row=passes,
            table=table,
            max_pip=max_pip,
            seed=rng.randrange(1 << 30),
            **kw,
        )
        stats["seconds"] += time.perf_counter() - t0
        stats["answers"] += 1
        keys = [mod.tile_key(t) for t in hands[me]]
        values = None if ends is None else (ends[0]["value"], ends[1]["value"])
        legal = mod.legal_moves(keys, values)
        route = res["route"]
        if route in ("play", "no_room"):
            t = tuple(res["tile"])
            if t not in hands[me]:
                raise IllegalAnswer(f"{t} is not in hand {hands[me]}")
            e = res["end"]
            if values is not None and values[e] not in t:
                raise IllegalAnswer(f"{t} does not match end {e} value {values[e]}")
            if values is not None and values[0] == values[1] and e not in (0, 1):
                raise IllegalAnswer(f"bad end {e}")
            if route == "no_room":
                stats["no_room"] += 1
                return {"result": ("no_room", me, 0), "stats": stats, "footprints": footprints}
            fp = res["new_footprint"]
            rect = mod.Rect(
                np.array(fp["center"]), np.array(fp["axis"]), fp["length"] / 2, fp["width"] / 2
            )
            if box is not None and not mod._in_box(rect, box, margin - 1e-6):
                raise IllegalAnswer(f"{t} placed outside the table: {rect.corners()}")
            for f in footprints:
                other = mod.Rect(
                    np.array(f["center"]), np.array(f["axis"]), f["length"] / 2, f["width"] / 2
                )
                if rect.overlaps(other):
                    raise IllegalAnswer(f"{t} overlaps a placed tile at {f['center']}")
            a_end, b_end = (np.array(p[:2]) for p in res["target_ends_world"])
            if abs(np.linalg.norm(b_end - a_end) - fp["length"]) > 1e-5:
                raise IllegalAnswer("target ends are not one tile length apart")
            if ends is not None:
                p = np.array(ends[e]["position"][:2])
                if abs(rect.distance_to(p) - gap) > 1e-5:
                    raise IllegalAnswer(f"{t} is {rect.distance_to(p)} m from end {e}, not {gap}")
                half_a = a_end + (b_end - a_end) / 4
                half_b = a_end + 3 * (b_end - a_end) / 4
                near = t[0] if np.linalg.norm(half_a - p) <= np.linalg.norm(half_b - p) else t[1]
                if near != ends[e]["value"]:
                    raise IllegalAnswer(
                        f"{t}: the half nearest end {e} holds {near}, not {ends[e]['value']}"
                    )
            hands[me].remove(t)
            played.append(list(t))
            footprints.append(fp)
            ends = res["new_ends"]
            stats["plays"] += 1
            stats["layouts"][res["layout"]] = stats["layouts"].get(res["layout"], 0) + 1
            passes = 0
            if not hands[me]:
                return {
                    "result": ("out", me, sum(mod.pips(h) for h in hands)),
                    "stats": stats,
                    "footprints": footprints,
                }
            turn = (me + 1) % n
            continue
        if legal:
            raise IllegalAnswer(f"route {route} with legal moves {legal}: {res['detail']}")
        if route == "draw":
            if rules != "draw" or not bone:
                raise IllegalAnswer("draw without a boneyard")
            hands[me].append(bone.pop())
            missing[me] |= set(values)
            stats["draws"] += 1
            continue
        if route == "pass":
            if rules == "draw" and bone:
                raise IllegalAnswer("pass while the boneyard has tiles")
            missing[me] |= set(values)
            passes += 1
            stats["passes"] += 1
            turn = (me + 1) % n
            continue
        if route == "game_over":
            exhausted = all(
                list(t) in played or [t[1], t[0]] in played
                for t in mod.full_set(max_pip)
                if values[0] in t or values[1] in t
            )
            if not exhausted and (passes < n - 1 or (rules == "draw" and bone)):
                raise IllegalAnswer(f"game_over but the game goes on: {res['detail']}")
            g = mod.GameState(
                [[mod.tile_key(t) for t in h] for h in hands], [], values, me, n, False, teams
            )
            res_ = g.blocked_result()
            return {"result": res_, "stats": stats, "footprints": footprints, "teams": teams}
        raise IllegalAnswer(f"unexpected route {route}: {res['detail']}")
    raise IllegalAnswer("game did not end in 400 answers")


def side_won(result: tuple, side_player: int, n: int) -> float:
    """1 if the side of player ``side_player`` won, 0.5 for a tie, 0 for a loss, None for no_room."""
    kind, who, points = result
    teams = n == 4
    side = (lambda p: p % 2) if teams else (lambda p: p)
    if kind == "no_room":
        return math.nan
    if kind == "out":
        return float(side(who) == side(side_player))
    if who < 0:
        return 0.5
    return float(who == side(side_player))


def match(mod, a, b, games: int, rules: str, n: int = 2, seed0: int = 0, **kw) -> dict:
    """Player a against b. Two players: seats swap each game; four: a holds seats 0 and 2
    (or 1 and 3); three: a holds one seat, rotating, against two b's."""
    wins = ties = losses = no_room = 0
    stats_total = {"answers": 0, "seconds": 0.0, "plays": 0, "layouts": {}}
    for g in range(games):
        if n == 3:
            mine = g % 3
            seats = [a if k == mine else b for k in range(n)]
        else:
            mine = g % 2
            seats = [a if k % 2 == mine else b for k in range(n)]
        res = play_game(mod, seats, rules=rules, seed=seed0 + g, starter=(g // 2) % n, **kw)
        w = side_won(res["result"], mine, n)
        if math.isnan(w):
            no_room += 1
        elif w == 1:
            wins += 1
        elif w == 0.5:
            ties += 1
        else:
            losses += 1
        for k in ("answers", "seconds", "plays"):
            stats_total[k] += res["stats"][k]
        for k, v in res["stats"]["layouts"].items():
            stats_total["layouts"][k] = stats_total["layouts"].get(k, 0) + v
    decided = wins + losses + ties
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "no_room": no_room,
        "win_rate": (wins + 0.5 * ties) / decided if decided else math.nan,
        **stats_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--search-games", type=int, default=100)
    args = ap.parse_args()
    mod = load()
    rows = [
        ("draw", 2, "random", "random", args.games),
        ("draw", 2, "greedy", "random", args.games),
        ("draw", 2, "double", "random", args.games),
        ("draw", 2, "search", "random", args.search_games),
        ("draw", 2, "search", "greedy", args.search_games),
        ("block", 2, "greedy", "random", args.games),
        ("block", 2, "search", "random", args.search_games),
        ("block", 2, "search", "greedy", args.search_games),
        ("block", 4, "greedy", "random", args.games),
        ("block", 4, "search", "random", args.search_games),
        ("block", 4, "search", "greedy", args.search_games),
        ("draw", 3, "greedy", "random", args.games),
    ]
    print(
        "rules players  A vs B          games  A wins ties  losses no_room  A win rate  s/answer  layouts"
    )
    for rules, n, a, b, games in rows:
        r = match(mod, a, b, games, rules, n=n, seed0=1000)
        print(
            f"{rules:5s} {n:7d}  {a:6s} vs {b:6s}  {games:6d}  {r['wins']:6d} {r['ties']:4d}  {r['losses']:6d} "
            f"{r['no_room']:7d}  {r['win_rate']:10.3f}  {r['seconds'] / max(r['answers'], 1):8.4f}  {r['layouts']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
