"""finding-the-longest-domino-chain: the search, its ranking, the world layout and the exits."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "finding-the-longest-domino-chain"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
TABLE = {"min_m": [-0.15, 0.10], "max_m": [0.35, 0.55]}
L, W, GAP = 0.048, 0.024, 0.001
Z = 0.004


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["find_chain"].module


def _run(script, hand, ctx=None, **kw):
    return script.run(ctx or FakeContext(), hand=hand, **kw)


def _end(value, p, d):
    return {"value": value, "position": [p[0], p[1], Z], "direction": list(d)}


def _key(chain):
    if not chain:
        return (0, -1, False)
    return (len(chain), sum(a + b for a, b in chain), chain[-1][0] != chain[-1][1])


def _brute(hand, start):
    """Every ordered selection of tiles, each turned both ways: the best legal chain's key."""
    best = (0, -1, False)
    for k in range(1, len(hand) + 1):
        for perm in itertools.permutations(hand, k):
            v, chain = start, []
            for a, b in perm:
                if a == v:
                    chain.append((a, b))
                    v = b
                elif b == v:
                    chain.append((b, a))
                    v = a
                else:
                    break
            else:
                best = max(best, _key(chain))
    return best


# --- the source's own tests ---------------------------------------------------------------


def test_upstream_graph_test_most_valuable_chain(script):
    # src/Graph.test.ts: no start side -> {5x1, 3x1, 3x4}; start 12 -> nothing
    hand = [(1, 2), (7, 8), (3, 4), (5, 1), (3, 1), (6, 7)]
    res = _run(script, hand)
    assert res["route"] == "chain_found"
    assert {tuple(sorted(t["tile"])) for t in res["chain"]} == {(1, 5), (1, 3), (3, 4)}
    # 5-1-3-4 or the same chain walked from 4: both rank equal, the first found is kept
    assert res["chain_tiles"] in ([[5, 1], [1, 3], [3, 4]], [[4, 3], [3, 1], [1, 5]])
    assert {res["start_value"], res["end_value"]} == {4, 5} and res["chain_pips"] == 17
    res = _run(script, hand, start_value=12)
    assert res["route"] == "no_chain" and res["chain"] == []


def test_ranking_follows_compare_chains(script):
    # src/findMostValuableChain.test.ts: longer, then more pips, then not ending in a double
    assert script._rank("longest", 2, 3, False) > script._rank("longest", 1, 99, False)
    assert script._rank("longest", 2, 39, False) > script._rank("longest", 2, 10, False)
    assert script._rank("longest", 2, 6, False) > script._rank("longest", 2, 6, True)
    assert script._rank("longest", 1, 0, True) > script._rank("longest", 0, 0, False)
    # [0x1, 1x1] vs [0x2, 2x... ]: same length and pips, the one not ending in a double wins
    steps, *_ = script.best_chain([(0, 1), (1, 1), (0, 2)], 0)
    assert len(steps) == 2  # 0-1, 1-1 is the only two-tile chain
    steps, *_ = script.best_chain([(3, 3), (3, 0), (3, 1), (1, 2)], 3)
    # 3-3, 3-1, 1-2 (length 3, 12 pips) beats ending on the double
    assert [(n, f) for _, n, f in steps] == [(3, 3), (3, 1), (1, 2)]


def test_tie_on_length_and_pips_prefers_an_open_end_that_is_not_a_double(script):
    steps, *_ = script.best_chain([(0, 2), (2, 2), (0, 4)], 0)
    # 0-2, 2-2 (6 pips, ends on a double) vs 0-4 (4 pips, length 1): length decides
    assert len(steps) == 2
    steps, *_ = script.best_chain([(0, 3), (3, 3), (0, 2), (2, 4)], 0)
    # [0-3, 3-3] and [0-2, 2-4]: both 2 tiles, 9 vs 8 pips -> pips decide
    assert [(n, f) for _, n, f in steps] == [(0, 3), (3, 3)]
    steps, *_ = script.best_chain([(0, 2), (2, 2), (0, 1), (1, 3)], 0)
    # [0-2, 2-2] and [0-1, 1-3]: 2 tiles, 6 vs 5 -> [0-2, 2-2]; give them equal pips:
    steps, *_ = script.best_chain([(0, 2), (2, 2), (0, 3), (3, 1)], 0)
    assert [(n, f) for _, n, f in steps] == [(0, 3), (3, 1)]


# --- the search is exact -----------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_matches_brute_force_on_small_hands(script, seed):
    rng = random.Random(seed)
    full = [(a, b) for a in range(5) for b in range(a, 5)]
    for _ in range(25):
        hand = rng.sample(full, rng.randint(1, 6))
        hand = [t if rng.random() < 0.5 else t[::-1] for t in hand]
        start = rng.randint(0, 4)
        steps, nodes, exhaustive, s, bound = script.best_chain(hand, start)
        chain = [(n, f) for _, n, f in steps]
        assert exhaustive
        assert _key(chain) == _brute(hand, start)
        assert len(chain) <= bound
        if chain:
            assert chain[0][0] == start and s == start
        for (_, f), (n, _) in itertools.pairwise(chain):
            assert f == n
        assert len({i for i, _, _ in steps}) == len(steps)


def test_bound_is_never_below_the_best_chain(script):
    rng = random.Random(7)
    full = [(a, b) for a in range(7) for b in range(a, 7)]
    for _ in range(40):
        hand = rng.sample(full, 10)
        steps, _, exhaustive, _, bound = script.best_chain(hand, rng.randint(0, 6))
        assert exhaustive and len(steps) <= bound


def test_full_double_six_set_is_one_chain(script):
    full = [(a, b) for a in range(7) for b in range(a, 7)]
    res = _run(script, full, start_value=6)
    assert res["route"] == "chain_found"
    assert res["length"] == 28 and res["length_bound"] == 28 and res["exhaustive"]
    assert res["leftover"] == [] and res["chain_pips"] == 168


def test_most_pips_objective(script):
    hand = [(0, 1), (1, 2), (2, 3), (0, 12)]
    assert _run(script, hand, start_value=0)["chain_tiles"] == [[0, 1], [1, 2], [2, 3]]
    res = _run(script, hand, start_value=0, objective="most_pips")
    assert res["chain_tiles"] == [[0, 12]] and res["chain_pips"] == 12


def test_budget_stops_with_a_legal_chain(script):
    rng = random.Random(3)
    full = [(a, b) for a in range(7) for b in range(a, 7)]
    hand = rng.sample(full, 22)
    res = _run(script, hand, start_value=3, max_nodes=50)
    assert res["route"] == "chain_found"
    assert not res["exhaustive"] and res["nodes"] == 50
    assert "not proven best" in res["detail"]
    tiles = res["chain_tiles"]
    assert tiles[0][0] == 3
    for (_, f), (n, _) in itertools.pairwise(tiles):
        assert f == n


def test_chain_tiles_are_turned_and_leftovers_reported(script):
    hand = [(4, 2), (9, 9), (2, 0), {"value_ab": [5, 4]}]
    res = _run(script, hand, start_value=5)
    assert res["chain_tiles"] == [[5, 4], [4, 2], [2, 0]]
    assert res["hand_indices"] == [3, 0, 2]
    assert [c["tile"] for c in res["chain"]] == [[5, 4], [4, 2], [2, 0]]
    assert res["leftover"] == [[9, 9]] and res["leftover_pips"] == 18
    assert res["end_value"] == 0 and not res["ends_in_double"]
    assert "no placement" in res["detail"]


# --- the world layout --------------------------------------------------------------------


def _check_layout(res, start_p):
    """Each tile's matching half is the one nearer the previous tile (or the start point), and
    the tile is L long and centred on its pose."""
    prev = np.array(start_p[:2])
    for step in res["chain"][: res["placed_count"]]:
        a_end, b_end = (np.array(x[:2]) for x in step["target_ends_world"])
        c = (a_end + b_end) / 2
        pose = step["place_pose"]["position"]
        assert np.allclose(c, [pose["x"], pose["y"]], atol=1e-6)
        assert np.linalg.norm(b_end - a_end) == pytest.approx(L, abs=1e-6)
        if step["layout"] != "crosswise":
            near_end, far_end = (
                (a_end, b_end) if step["tile"][0] == step["near"] else (b_end, a_end)
            )
            hn = near_end + (far_end - near_end) / 4
            hf = near_end + 3 * (far_end - near_end) / 4
            assert np.linalg.norm(hn - prev) < np.linalg.norm(hf - prev)
        prev = c


def _corners(fp):
    c, u = np.array(fp["center"][:2]), np.array(fp["axis"][:2])
    v = np.array([-u[1], u[0]])
    return np.array([c + su * L / 2 * u + sv * W / 2 * v for su in (-1, 1) for sv in (-1, 1)])


def _overlap(f1, f2, tol=0.0005):
    a, b = _corners(f1), _corners(f2)
    for f in (f1, f2):
        u = np.array(f["axis"][:2])
        for ax in (u, np.array([-u[1], u[0]])):
            pa, pb = a @ ax, b @ ax
            if min(pa.max(), pb.max()) - max(pa.min(), pb.min()) <= tol:
                return False
    return True


def _inside(fp, margin=0.01):
    c = _corners(fp)
    lo, hi = np.array(TABLE["min_m"]), np.array(TABLE["max_m"])
    return bool(np.all(c >= lo + margin - 1e-9) and np.all(c <= hi - margin + 1e-9))


def test_straight_chain_from_the_train_end(script):
    hand = [(6, 2), (2, 5), (5, 1)]
    start = _end(6, (-0.10, 0.33), (1, 0))
    res = _run(script, hand, start_end=start, table=TABLE)
    assert res["route"] == "chain_found" and res["placed_count"] == 3
    assert [s["layout"] for s in res["chain"]] == ["straight"] * 3
    xs = [s["place_pose"]["position"]["x"] for s in res["chain"]]
    assert xs == pytest.approx([-0.10 + GAP + L / 2 + k * (GAP + L) for k in range(3)])
    # tile 0 is [6, 2]: end a (the 6) toward the start, yaw 0
    a0, b0 = res["chain"][0]["target_ends_world"]
    assert a0[0] < b0[0] and res["chain"][0]["place_yaw_deg"] == pytest.approx(0.0)
    # tile 1 is [2, 5] with a = 2 near: yaw 0; a tile held [5, 1] also yaw 0
    assert res["new_end"]["value"] == 1
    assert res["new_end"]["position"][0] == pytest.approx(-0.10 + 3 * (GAP + L))
    _check_layout(res, start["position"])
    assert all(_inside(f) for f in res["footprints_after"])


def test_tile_held_the_other_way_is_turned(script):
    res = _run(script, [(2, 6)], start_end=_end(6, (0.0, 0.33), (1, 0)), table=TABLE)
    a, b = res["chain"][0]["target_ends_world"]
    # end a (the 2) goes far from the start: yaw 180
    assert a[0] > b[0]
    assert abs(res["chain"][0]["place_yaw_deg"]) == pytest.approx(180.0)


def test_double_goes_crosswise(script):
    res = _run(script, [(6, 6), (6, 3)], start_end=_end(6, (0.0, 0.33), (1, 0)), table=TABLE)
    assert [s["layout"] for s in res["chain"]] == ["crosswise", "straight"]
    c0 = res["chain"][0]["place_pose"]["position"]
    assert c0["x"] == pytest.approx(GAP + W / 2)
    c1 = res["chain"][1]["place_pose"]["position"]
    assert c1["x"] == pytest.approx(GAP + W + GAP + L / 2)
    off = _run(
        script,
        [(6, 6), (6, 3)],
        start_end=_end(6, (0.0, 0.33), (1, 0)),
        table=TABLE,
        doubles_crosswise=False,
    )
    assert [s["layout"] for s in off["chain"]] == ["straight", "straight"]


def test_long_chain_turns_before_the_table_edge(script):
    full = [(a, b) for a in range(7) for b in range(a, 7)]
    start = _end(6, (-0.12, 0.32), (1, 0))
    res = _run(script, full, start_end=start, table=TABLE)
    assert res["length"] == 28
    assert res["route"] == "chain_found" and res["placed_count"] == 28
    # the greedy layout spirals into its own start at tile 27: backtracking finds one
    assert res["layout_tries"] > 28
    greedy = _run(script, full, start_end=start, table=TABLE, layout_tries=28)
    assert greedy["route"] == "no_room" and greedy["placed_count"] == 27
    layouts = {s["layout"] for s in res["chain"]}
    assert layouts & {"corner_left", "corner_right"}
    assert all(_inside(f) for f in res["footprints_after"])
    for i, j in itertools.combinations(range(28), 2):
        assert not _overlap(res["footprints_after"][i], res["footprints_after"][j])
    _check_layout(res, start["position"])


def test_placed_footprints_are_avoided_and_no_room_routes(script):
    start = _end(1, (0.0, 0.33), (1, 0))
    wall = [
        {"center": [GAP + L / 2, 0.33, Z], "axis": [1, 0, 0]},
        {"center": [GAP + W / 2, 0.33 + L / 2, Z], "axis": [0, 1, 0]},
        {"center": [GAP + W / 2, 0.33 - L / 2, Z], "axis": [0, 1, 0]},
    ]
    res = _run(script, [(1, 2), (2, 3)], start_end=start, table=TABLE, footprints=wall)
    assert res["route"] == "no_room" and res["placed_count"] == 0
    assert res["length"] == 2 and res["new_end"]["value"] == 1
    # a blocker one tile further on: straight would leave no room, so the chain turns now
    ahead = {"center": [2 * GAP + 1.5 * L, 0.33, Z], "axis": [1, 0, 0]}
    res = _run(script, [(1, 2), (2, 3)], start_end=start, table=TABLE, footprints=[ahead])
    assert res["route"] == "chain_found" and res["placed_count"] == 2
    assert res["chain"][0]["layout"] in ("corner_left", "corner_right")
    for fp in res["footprints_after"]:
        assert not _overlap(fp, ahead)


def test_table_comes_from_describe_workspace(script):
    ctx = FakeContext(
        {"robot.describe_workspace": {"bounds": {"min": [-0.15, 0.10], "max": [0.35, 0.55]}}}
    )
    res = _run(script, [(3, 4)], ctx=ctx, start_end=_end(3, (0.33, 0.33), (1, 0)))
    assert res["route"] == "no_room"
    res = _run(script, [(3, 4), (4, 5)], ctx=FakeContext(), start_end=_end(3, (0.30, 0.33), (1, 0)))
    assert res["route"] == "chain_found" and "no table box" in res["detail"]


def test_start_value_from_start_end(script):
    res = _run(script, [(2, 3), (3, 3)], start_end=_end(3, (0.0, 0.33), (0, 1)), table=TABLE)
    assert res["start_value"] == 3 and res["chain_tiles"][0][0] == 3
    yaws = [s["place_yaw_deg"] for s in res["chain"]]
    # [2, 3] goes on with its 3 toward the start, end a (the 2) farther along +y: the
    # a-to-b axis points back, -y
    assert res["chain"][-1]["layout"] == "straight" and yaws[-1] == pytest.approx(-90.0)


# --- exits -------------------------------------------------------------------------------


def test_no_chain(script):
    assert _run(script, [])["route"] == "no_chain"
    res = _run(script, [(1, 2)], start_value=5)
    assert res["route"] == "no_chain" and res["leftover"] == [[1, 2]] and res["leftover_pips"] == 3


@pytest.mark.parametrize(
    "kw",
    [
        {"hand": [(1, 13)]},
        {"hand": [(1, 2), (2, 1)]},
        {"hand": [(1,)]},
        {"hand": [(1, 2)], "objective": "fastest"},
        {"hand": [(1, 2)], "start_value": 20},
        {
            "hand": [(1, 2)],
            "start_end": {"value": 1, "position": [0, 0.3, 0], "direction": [0, 0, 1]},
        },
        {"hand": [(1, 2)], "footprints": [{"center": [0, 0, 0]}]},
    ],
)
def test_invalid_input(script, kw):
    hand = kw.pop("hand")
    assert _run(script, hand, **kw)["route"] == "invalid_input"


def test_example_declares_every_route(script):
    ex = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert declared == {"chain_found", "no_chain", "no_room", "invalid_input"}
