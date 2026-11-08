"""choosing-a-domino-move-by-end-matching: legal end matching, players, world placement, exits."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

from . import _domino_game as G

BUNDLE = "choosing-a-domino-move-by-end-matching"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
TABLE = G.TABLE
L, W, GAP = 0.048, 0.024, 0.001
Z = 0.004


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["choose_move"].module


def _end(value, p, d, crosswise=False):
    return {
        "value": value,
        "position": [p[0], p[1], Z],
        "direction": list(d),
        "crosswise": crosswise,
    }


def _board(e0, e1, footprints=(), tiles=None):
    b = {"ends": [e0, e1], "footprints": list(footprints)}
    if tiles is not None:
        b["tiles"] = tiles
    return b


# a board in the middle of the table running along +x: end 0 faces -x, end 1 faces +x
MID0, MID1 = (0.05, 0.33), (0.15, 0.33)


def _mid(v0, v1, **kw):
    return _board(_end(v0, MID0, (-1, 0)), _end(v1, MID1, (1, 0)), **kw)


def _run(script, hand, board, **kw):
    kw.setdefault("table", TABLE)
    return script.run(FakeContext(), hand=hand, board=board, **kw)


def _half_values_near(res, p):
    a, b = (np.array(x[:2]) for x in res["target_ends_world"])
    ha, hb = a + (b - a) / 4, a + 3 * (b - a) / 4
    return res["tile"][0] if np.linalg.norm(ha - p) <= np.linalg.norm(hb - p) else res["tile"][1]


# --- rules ported from the source -------------------------------------------------------


def test_legal_moves_match_game_update_valid_moves(script):
    # the source's docstring: board [6|6], hand with [0|6] and [2|6] -> both on the left only
    hand = [(1, 1), (3, 4), (0, 5), (0, 6), (2, 5), (1, 5), (2, 6)]
    assert script.legal_moves(hand, (6, 6)) == [((0, 6), 0), ((2, 6), 0)]
    assert script.legal_moves([(1, 2), (2, 3)], (1, 3)) == [((1, 2), 0), ((2, 3), 1)]
    assert script.legal_moves([(1, 3)], (1, 3)) == [((1, 3), 0), ((1, 3), 1)]
    assert script.legal_moves([(1, 3)], None) == [((1, 3), -1)]
    assert script.ends_after((1, 3), ((1, 5), 0)) == (5, 3)
    assert script.ends_after((1, 3), ((3, 3), 1)) == (1, 3)


def test_scores_follow_game_make_move(script):
    g = script.GameState([[], [(1, 2)], [(3, 3)], [(0, 1)]], [], (4, 5), 0, 0, False, True)
    g.result = ("out", 0, 3 + 6 + 1)
    assert g.value() == 10.0  # player 0 out: every pip left, partner's included
    g.result = ("out", 3, 10)
    assert g.value() == -10.0
    g = script.GameState([[(0, 1)], [(5, 5)], [(1, 1)], [(2, 2)]], [], (4, 6), 0, 0, False, True)
    assert g.blocked_result() == ("blocked", 0, 1 + 2 + 10 + 4)  # team 0: 3 pips vs 14
    g = script.GameState([[(0, 2)], [(1, 1)]], [], (4, 6), 0, 0, False, False)
    assert g.blocked_result() == ("blocked", -1, 0)  # tie scores nothing


def test_players_rank_like_the_source(script):
    rng = random.Random(0)
    moves = [((0, 1), 0), ((2, 2), 0), ((5, 6), 1), ((3, 3), 1)]
    assert script.p_bota_gorda(moves, rng, {})[0] == ((5, 6), 1)
    assert [m[0] for m in script.p_double(moves, rng, {})][:2] == [(2, 2), (3, 3)]
    assert script.p_reverse(moves, rng, {}) == moves[::-1]
    shuffled = script.p_random(moves, random.Random(1), {})
    assert sorted(shuffled) == sorted(moves)
    # composition: double after bota_gorda = doubles first, biggest first among them
    ranked = script.p_double(script.p_bota_gorda(moves, rng, {}), rng, {})
    assert [m[0] for m in ranked] == [(3, 3), (2, 2), (5, 6), (0, 1)]


def test_hidden_deals_honour_missing_values(script):
    rng = random.Random(3)
    unseen = [t for t in script.full_set() if t not in {(6, 6), (0, 0)}]
    for _ in range(50):
        hands, bone, ok = script.deal_hidden(rng, unseen, [7, 5], 9, [{6, 5}, set()])
        assert ok and len(hands[0]) == 7 and len(hands[1]) == 5 and len(bone) == 9
        assert not any(6 in t or 5 in t for t in hands[0])
        assert len({*hands[0], *hands[1], *bone}) == 21


def test_search_finds_the_winning_move_greedy_misses(script):
    # block game, the opponent's last tile is [6|6] (every other tile is accounted for).
    # Greedy plays [5|6] (11 pips), the opponent goes out on the 6; playing [0|1] first
    # leaves the opponent no move, and [5|6] then goes out.
    hand = [[5, 6], [0, 1]]
    played = [list(t) for t in script.full_set() if t not in {(5, 6), (0, 1), (6, 6)}]
    common = dict(rules="block", opponent_counts=[1], played=played, seed=0)
    greedy = _run(script, hand, _mid(0, 5), player="greedy", **common)
    search = _run(script, hand, _mid(0, 5), player="search", **common)
    assert greedy["tile"] == [5, 6]
    assert search["route"] == "play" and search["tile"] == [0, 1] and search["end"] == 0
    assert search["search_counts"]["[0, 1]@0"] == script.SEARCH_SAMPLES


# --- placement geometry ---------------------------------------------------------------


def test_straight_placement_puts_the_matching_half_against_the_end(script):
    res = _run(script, [[3, 5]], _mid(2, 5))
    assert res["route"] == "play" and res["end"] == 1 and res["layout"] == "straight"
    c = res["place_pose"]["position"]
    assert c["x"] == pytest.approx(MID1[0] + GAP + L / 2) and c["y"] == pytest.approx(MID1[1])
    assert c["z"] == pytest.approx(Z)
    a, b = res["target_ends_world"]
    assert b[0] == pytest.approx(MID1[0] + GAP)  # end b (the 5) touches the end
    assert a[0] == pytest.approx(MID1[0] + GAP + L)
    assert res["place_yaw_deg"] == pytest.approx(180.0)
    q = res["place_pose"]["rotation"]
    assert math.degrees(2 * math.atan2(q["z"], q["w"])) == pytest.approx(180.0)
    new = res["new_ends"][1]
    assert new["value"] == 3 and new["position"][0] == pytest.approx(MID1[0] + GAP + L)
    assert new["direction"] == [1.0, 0.0] and res["new_ends"][0]["value"] == 2


def test_double_goes_crosswise_or_inline(script):
    res = _run(script, [[4, 4]], _mid(1, 4))
    assert res["layout"] == "crosswise"
    c = res["place_pose"]["position"]
    assert c["x"] == pytest.approx(MID1[0] + GAP + W / 2)
    a, b = res["target_ends_world"]
    assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) == pytest.approx(L)  # across the line
    assert res["new_ends"][1]["position"][0] == pytest.approx(MID1[0] + GAP + W)
    assert res["new_ends"][1]["crosswise"] is True
    res = _run(script, [[4, 4]], _mid(1, 4), doubles_crosswise=False)
    assert res["layout"] == "straight"
    assert res["place_pose"]["position"]["x"] == pytest.approx(MID1[0] + GAP + L / 2)


def test_turns_a_corner_toward_the_table_centre_at_the_edge(script):
    # end 1 faces +x with just room for a corner (W + gap + 2 mm inside the margin)
    edge = (
        TABLE["max_m"][0] - 0.01 - W - GAP - 0.002,
        0.45,
    )  # upper half of the table: turn right (-y)
    res = _run(script, [[2, 5]], _board(_end(0, (0.0, 0.45), (-1, 0)), _end(5, edge, (1, 0))))
    assert res["route"] == "play" and res["layout"] == "corner_right"
    a, b = (np.array(p[:2]) for p in res["target_ends_world"])
    assert b[1] > a[1]  # the 5 (end b) is the inner half, nearest the end
    assert b[0] == pytest.approx(edge[0] + GAP + W / 2) and b[1] == pytest.approx(edge[1] + W / 2)
    assert res["new_ends"][1]["direction"] == [0.0, -1.0]
    assert res["new_ends"][1]["position"][1] == pytest.approx(edge[1] + W / 2 - L)
    rect = G.load().Rect(
        np.array(res["new_footprint"]["center"]),
        np.array(res["new_footprint"]["axis"]),
        L / 2,
        W / 2,
    )
    assert np.all(rect.corners() <= np.array(TABLE["max_m"]) - 0.01 + 1e-9)


def test_turns_before_the_edge_when_straight_would_leave_no_room(script):
    # straight fits (ends 15 mm inside the margin) but nothing could follow: turn now
    x = TABLE["max_m"][0] - 0.01 - L - GAP - 0.015
    res = _run(script, [[2, 5]], _board(_end(0, (0.0, 0.2), (-1, 0)), _end(5, (x, 0.2), (1, 0))))
    assert res["layout"] == "corner_left"  # lower half of the table: turn toward +y


def test_corner_round_a_placed_tile(script):
    blocker = {"center": [MID1[0] + 0.04, MID1[1], Z], "axis": [0, 1, 0]}
    res = _run(script, [[2, 5]], _mid(0, 5, footprints=[blocker]))
    assert res["layout"].startswith("corner")
    rect = G.load().Rect(
        np.array(res["new_footprint"]["center"]),
        np.array(res["new_footprint"]["axis"]),
        L / 2,
        W / 2,
    )
    other = G.load().Rect(np.array(blocker["center"]), np.array([0.0, 1.0]), L / 2, W / 2)
    assert not rect.overlaps(other)


def test_same_value_ends_use_the_end_with_room(script):
    edge = (TABLE["max_m"][0] - 0.012, 0.33)
    board = _board(_end(5, MID0, (-1, 0)), _end(5, edge, (1, 0)))
    blocker = {"center": [MID0[0] - 0.04, MID0[1], Z], "axis": [0, 1, 0]}
    board["footprints"] = [blocker]
    res = _run(script, [[2, 5]], board)
    # end 1 has no room even round a corner (12 mm from the edge); end 0 turns round the blocker
    assert res["route"] == "play" and res["end"] == 0 and res["layout"].startswith("corner")


def test_geometry_overrides_the_player_when_its_move_has_no_room(script):
    edge = (TABLE["max_m"][0] - 0.012, 0.33)
    board = _board(_end(1, MID0, (-1, 0)), _end(6, edge, (1, 0)))
    res = _run(script, [[6, 6], [1, 2]], board, player="greedy")
    assert res["ranked_moves"][0]["tile"] == [6, 6]
    assert res["route"] == "play" and res["tile"] == [1, 2] and res["end"] == 0
    assert "first choice [6, 6]" in res["detail"]


def test_no_room_when_no_legal_move_fits(script):
    edge = (TABLE["max_m"][0] - 0.012, 0.33)
    board = _board(_end(1, MID0, (-1, 0)), _end(6, edge, (1, 0)))
    res = _run(script, [[6, 6]], board)
    assert res["route"] == "no_room" and res["tile"] == [6, 6]


def test_opening_move_at_the_table_centre_and_from_start_pose(script, skills_registry):
    res = _run(script, [[1, 2], [5, 6]], {})
    assert res["route"] == "play" and res["tile"] == [5, 6] and res["end"] == -1
    c = res["place_pose"]["position"]
    assert c["x"] == pytest.approx(0.10) and c["y"] == pytest.approx(0.325)
    assert res["new_ends"][0]["value"] == 5 and res["new_ends"][1]["value"] == 6
    assert res["new_ends"][1]["direction"] == [1.0, 0.0]  # along the longer side (x, 0.5 m)
    res = _run(script, [[3, 3]], {}, start_pose={"position": [0.0, 0.3, Z], "direction": [0, 1]})
    assert res["layout"] == "crosswise"
    assert res["new_ends"][1]["position"] == pytest.approx([0.0, 0.3 + W / 2, Z])
    ws = {
        "robot.describe_workspace": {"bounds": {"min_m": [0.0, 0.0, 0.0], "max_m": [0.2, 0.6, 0.3]}}
    }
    res = script.run(FakeContext(ws), hand=[[1, 2]], board={})
    assert res["place_pose"]["position"]["y"] == pytest.approx(0.3)
    assert res["new_ends"][1]["direction"] == [0.0, 1.0]


def test_must_play_opening(script):
    res = _run(script, [[1, 2], [6, 6]], {}, must_play=[6, 6])
    assert res["route"] == "play" and res["tile"] == [6, 6]
    res = _run(script, [[1, 2]], {}, must_play=[6, 6])
    assert res["route"] == "pass"


def test_reader_records_and_hand_index(script):
    hand = [
        {"value": [0, 1], "value_ab": [1, 0], "center_world": [0.3, 0.2, Z]},
        {"value": [3, 4], "value_ab": [4, 3], "center_world": [0.3, 0.25, Z]},
    ]
    res = _run(script, hand, _mid(2, 3))
    assert res["tile"] == [4, 3] and res["hand_index"] == 1
    assert _half_values_near(res, np.array(MID1)) == 3


def test_value_only_ends_give_a_move_without_a_pose(script):
    res = _run(script, [[2, 3]], {"ends": [1, 3]})
    assert res["route"] == "play" and res["tile"] == [2, 3] and res["end"] == 1
    assert res["place_pose"] == {}


# --- exits ------------------------------------------------------------------------------


def test_draw_then_pass(script):
    res = _run(script, [[0, 0]], _mid(2, 5), rules="draw", boneyard_count=3)
    assert res["route"] == "draw"
    res = _run(script, [[0, 0]], _mid(2, 5), rules="draw", boneyard_count=0, opponent_counts=[4])
    assert res["route"] == "pass"
    res = _run(script, [[0, 0]], _mid(2, 5), rules="block", boneyard_count=3, opponent_counts=[4])
    assert res["route"] == "pass"


def test_game_over_routes(script):
    assert _run(script, [], _mid(2, 5))["route"] == "game_over"
    res = _run(script, [[1, 1]], _mid(2, 5), opponent_counts=[3, 0, 2])
    assert res["route"] == "game_over" and res["game_over_reason"] == "opponent_out"
    res = _run(script, [[0, 0]], _mid(2, 5), rules="block", opponent_counts=[3], passes_in_a_row=1)
    assert res["route"] == "game_over" and res["game_over_reason"] == "blocked"
    gone = [list(t) for t in G.load().full_set() if 2 in t or 5 in t]
    res = _run(script, [[0, 0]], _mid(2, 5), boneyard_count=4, played=gone, opponent_counts=[3])
    assert res["route"] == "game_over" and res["game_over_reason"] == "blocked"
    assert res["my_pips"] == 0


@pytest.mark.parametrize(
    "hand,board,kw,needle",
    [
        ([[1, 7]], {}, {}, "outside"),
        ([[1, 2], [2, 1]], {}, {}, "twice"),
        ([[1]], {}, {}, "two values"),
        ([[1, 2]], {"ends": [_end(1, MID0, (-1, 0))]}, {}, "two open ends"),
        ([[1, 2]], {}, {"player": "smart"}, "unknown player"),
        ([[1, 2]], {}, {"player": "search"}, "opponent_counts"),
        ([[1, 2]], {}, {"rules": "allfives"}, "rules"),
        ([[1, 2], [2, 3]], _mid(1, 2), {"player": "search", "opponent_counts": [30]}, "unseen"),
        ([[1, 2]], {"ends": [_end(1, MID0, (0, 0)), _end(2, MID1, (1, 0))]}, {}, "no length"),
    ],
)
def test_invalid_input(script, hand, board, kw, needle):
    res = _run(script, hand, board, **kw)
    assert res["route"] == "invalid_input" and needle in res["detail"]


def test_example_declares_every_route(script):
    ex = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert declared == {"play", "draw", "pass", "game_over", "no_room", "invalid_input"}


# --- whole games ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rules,n,a,b,games",
    [
        ("draw", 2, "greedy", "random", 40),
        ("draw", 2, "double", "random", 20),
        ("block", 2, "greedy", "random", 20),
        ("block", 4, "greedy", "random", 20),
        ("draw", 3, "random", "greedy", 15),
        ("draw", 2, "search", "random", 10),
        ("block", 4, "search", "greedy", 6),
    ],
)
def test_whole_games_are_legal_and_on_the_table(script, rules, n, a, b, games):
    # the referee raises on any illegal answer or a placement off the table / overlapping
    r = G.match(script, a, b, games, rules, n=n, seed0=77)
    assert r["no_room"] == 0
    assert r["wins"] + r["ties"] + r["losses"] == games
    assert r["plays"] > games * 5


def test_search_beats_random(script):
    # 100 seeded games: the search player won 0.65 of 200 such games (SKILL.md); random
    # against random 0.50. 0.58 is ~1.6 sigma under 0.65 at 100 games.
    r = G.match(script, "search", "random", 100, "draw", n=2, seed0=300)
    assert r["win_rate"] >= 0.58, r


def test_bigger_set_and_tile(script):
    res = G.play_game(
        script,
        ["greedy", "random"],
        rules="draw",
        seed=5,
        max_pip=9,
        table={"min_m": [-0.4, -0.4], "max_m": [0.4, 0.4]},
        skill_kw={"tile_size": [0.051, 0.025]},
    )
    assert res["result"][0] in ("out", "blocked")
    assert all(f["length"] == 0.051 for f in res["footprints"])
