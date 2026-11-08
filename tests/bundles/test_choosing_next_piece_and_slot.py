"""choosing-next-piece-and-slot: the next jigsaw placement by stage plan, assignment and reach."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

from . import _backtrack_sim as sim

BUNDLE = "choosing-next-piece-and-slot"
COLS, ROWS = 6, 5


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["choose_next"].module


@pytest.fixture(scope="module")
def backtrack(skills_registry):
    return (
        skills_registry.get("backtracking-a-wrong-placement").canonical_scripts["backtrack"].module
    )


def _rec(piece, cells_scores, **kw):
    cands = [{"col": c, "row": r, "score": s, "angle_deg": 10.0 * c} for (c, r), s in cells_scores]
    return dict({"piece": piece, "candidates": cands}, **kw)


def _sure(piece, cell, **kw):
    """A localizer record that is sure of one cell (P 0.98)."""
    return _rec(piece, [(cell, 0.97), ((0, 0) if cell != (0, 0) else (1, 1), 0.5)], **kw)


def _pl(*cells):
    return [{"piece": f"on{c}_{r}", "col": c, "row": r} for c, r in cells]


def _run(script, placements, pieces, **kw):
    return script.run(FakeContext(), COLS, ROWS, placements=placements, pieces=pieces, **kw)


# --- probabilities agree with the backtracking skill ----------------------------------------


def test_calibration_matches_backtracking(script, backtrack):
    for name in ("CONF_BIAS", "CONF_SCORE", "CONF_MARGIN", "MARGIN_CAP", "CONF_MIN", "CONF_MAX"):
        assert getattr(script, name) == getattr(backtrack, name), name
    rec = _rec("a", [((1, 1), 0.95), ((2, 2), 0.9), ((3, 3), 0.5)])
    mine, theirs = script.cell_probs(rec), backtrack.cell_probs(rec)
    assert {k: round(v["p"], 6) for k, v in mine.items()} == {
        k: round(v["p"], 6) for k, v in theirs.items()
    }


def test_a_shape_verdict_uses_its_confidence_and_flat_shortlists_share(script):
    rec = {
        "piece": "f",
        "candidates": [{"col": 1, "row": 1, "confidence": 0.9}]
        + [{"col": c, "row": 0, "score": None} for c in range(4)],
    }
    p = script.cell_probs(rec)
    assert p[1, 1]["p"] == pytest.approx(0.9) and p[1, 1]["shape"]
    assert all(p[c, 0]["p"] == pytest.approx(0.25) for c in range(4))


# --- the stage plan --------------------------------------------------------------------------


def test_done_when_every_cell_is_placed(script):
    cells = [(c, r) for r in range(ROWS) for c in range(COLS)]
    out = _run(script, _pl(*cells), [])
    assert out["route"] == "done" and out["n_free"] == 0


def test_an_empty_board_starts_at_a_corner(script):
    pieces = [_sure("e", (2, 0)), _sure("i", (2, 2)), _sure("k", (5, 4))]
    out = _run(script, [], pieces)
    assert out["route"] == "chosen"
    assert (out["piece"], out["col"], out["row"]) == ("k", 5, 4)
    assert out["cell_class"] == "corner" and out["angle_deg"] == 50.0


def test_the_frame_comes_before_the_interior(script):
    # (1, 1) is interior with two placed neighbours; (2, 0) is a frame cell with one.
    placements = _pl((0, 0), (1, 0), (0, 1))
    pieces = [_sure("in", (1, 1)), _sure("fr", (2, 0))]
    out = _run(script, placements, pieces)
    assert out["piece"] == "fr" and out["tier"] == 0
    assert [j["piece"] for j in out["queue"]["any"]] == ["fr", "in"]
    # Once the frame is complete the interior is the active region.
    frame = [
        (c, r)
        for r in range(ROWS)
        for c in range(COLS)
        if script.cell_class(c, r, COLS, ROWS) != "interior"
    ]
    out = _run(script, _pl(*frame), [_sure("in", (1, 1)), _sure("in2", (2, 2))])
    assert out["frame_done"] and out["piece"] == "in"


def test_more_placed_neighbours_win_within_a_tier(script):
    placements = _pl((0, 0), (1, 0), (2, 0), (0, 1), (0, 2))
    # (1, 1) has 2 neighbours, (3, 0) has 1; the (3, 0) piece is a little more certain.
    pieces = [
        _rec("two", [((1, 1), 0.93), ((4, 4), 0.85)]),
        _rec("one", [((3, 0), 0.97), ((4, 4), 0.5)]),
    ]
    out = _run(script, placements, pieces, policy="most_neighbours")
    assert out["piece"] == "two"
    out = _run(script, placements, pieces, policy="confidence")
    assert out["piece"] == "one"
    # The plan puts the frame cell first even so; among frame cells the neighbours count.
    pieces.append(_rec("corner2", [((0, 3), 0.93), ((4, 4), 0.85)]))
    out = _run(script, placements, pieces)
    assert out["piece"] == "one"
    placements2 = _pl((0, 0), (1, 0), (0, 1), (0, 2), (1, 3), (1, 2))
    out = _run(
        script, placements2, [_rec("a", [((0, 3), 0.93), ((4, 4), 0.85)]), _sure("b", (2, 0))]
    )
    assert out["piece"] == "a" and out["n_neighbours"] == 2


def test_two_pieces_claiming_one_cell_are_resolved_by_the_assignment(script):
    placements = _pl((0, 0), (1, 0))
    pieces = [
        _rec("a", [((2, 0), 0.97), ((0, 1), 0.93)]),
        _rec("b", [((2, 0), 0.96), ((4, 4), 0.5)]),
    ]
    out = _run(script, placements, pieces, queue_length=5)
    cells = [(j["col"], j["row"]) for j in out["queue"]["any"]]
    assert len(cells) == len(set(cells))
    assert {j["piece"] for j in out["queue"]["any"]} <= {"a", "b"}


def test_piece_class_keeps_corner_pieces_off_edge_cells(script):
    placements = _pl((0, 0))
    pieces = [_sure("c", (1, 0), piece_class="corner"), _sure("e", (0, 1), piece_class="edge")]
    out = _run(script, placements, pieces)
    assert out["piece"] == "e"
    assert all(j["piece"] != "c" for j in out["queue"]["any"])


def test_islands_only_when_the_frontier_has_nothing(script):
    placements = _pl((0, 0))
    pieces = [_sure("far", (5, 4))]
    out = _run(script, placements, pieces)
    assert out["route"] == "chosen" and out["tier"] == 1
    out = _run(script, placements, pieces, allow_islands=False)
    assert out["route"] != "chosen"


# --- what the backtracking ledger says ------------------------------------------------------


def test_forbidden_pairs_deferred_cells_and_posteriors_are_honoured(script, tmp_path):
    placements = _pl((0, 0), (1, 0))
    pieces = [_sure("a", (2, 0)), _sure("b", (0, 1))]
    ledger = {
        "version": 1,
        "grid": [COLS, ROWS],
        "forbidden": [{"piece": "a", "col": 2, "row": 0}],
        "deferred_cells": [],
        "p_right": {},
        "unexplained": {},
    }
    out = _run(script, placements, pieces, ledger=ledger)
    assert out["piece"] == "b"
    ledger["forbidden"] = []
    ledger["deferred_cells"] = [[0, 1]]
    out = _run(script, placements, pieces, ledger=ledger)
    assert out["piece"] == "a"
    ledger["deferred_cells"] = []
    ledger["p_right"] = {"a@2,0": 0.3}
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger))
    out = _run(script, placements, pieces, ledger_path=str(path))
    assert out["piece"] == "b"
    # A fresh shape-match verdict for the pair outranks the old posterior.
    pieces[0]["candidates"].insert(0, {"col": 2, "row": 0, "confidence": 0.95, "angle_deg": 0})
    pieces[1] = _rec("b", [((0, 1), 0.9), ((4, 4), 0.88)])
    out = _run(script, placements, pieces, ledger_path=str(path))
    assert out["piece"] == "a" and out["prob"] == pytest.approx(0.95)
    with pytest.raises(ValueError, match="grid"):
        _run(script, placements, pieces, ledger=dict(ledger, grid=[3, 3]))


def test_deferred_cells_are_reopened_when_nothing_else_is_left(script):
    cells = [(c, r) for r in range(ROWS) for c in range(COLS) if (c, r) != (3, 2)]
    ledger = {
        "grid": [COLS, ROWS],
        "deferred_cells": [[3, 2]],
        "forbidden": [{"piece": "x", "col": 3, "row": 2}],
    }
    out = _run(script, _pl(*cells), [_sure("y", (3, 2)), _sure("x", (3, 2))], ledger=ledger)
    assert out["route"] == "chosen" and out["piece"] == "y" and out["reopened"]
    out = _run(script, _pl(*cells), [_sure("x", (3, 2))], ledger=ledger)
    assert out["route"] == "need_shape_match" or out["route"] == "need_more_pieces"
    assert out.get("piece") != "x"
    out = _run(script, _pl(*cells), [_sure("y", (3, 2))], ledger=ledger, reopen_deferred=False)
    assert out["route"] == "need_more_pieces"


# --- when nothing is confident ---------------------------------------------------------------


def test_need_shape_match_offers_the_best_supported_slot_with_listed_pieces_then_orphans(script):
    placements = _pl((0, 0), (1, 0), (0, 1))
    flat = [{"col": c, "row": r, "score": None} for c, r in [(1, 1), (2, 0), (3, 3)]]
    pieces = [
        {"piece": "f1", "candidates": flat},
        {"piece": "f2", "candidates": flat},
        {"piece": "orph", "candidates": []},
        {"piece": "nope", "candidates": [], "piece_class": "corner"},
    ]
    out = _run(script, placements, pieces, all_pieces_known=True)
    assert out["route"] == "need_shape_match"
    assert out["slot"] == [1, 1] and out["n_neighbours"] == 2
    assert out["shape_candidates"][:2] == ["f1", "f2"]
    assert "orph" in out["shape_candidates"] and "nope" not in out["shape_candidates"]
    out = _run(script, placements, pieces, exclude_cells=[[1, 1]], all_pieces_known=True)
    assert out["slot"] != [1, 1]
    # 4 known loose pieces for 27 free cells: the rest are still in the box, so an orphan is
    # likelier to belong there than in this slot.
    out = _run(script, placements, pieces)
    assert out["shape_candidates"] == ["f1", "f2"]


def test_need_relocalize_then_need_more_pieces(script):
    pieces = [_rec("weak", [((3, 3), 0.6), ((2, 2), 0.59)])]
    # An empty board offers frame cells to a shape match only when a piece lists them.
    out = _run(script, [], pieces)
    assert out["route"] == "need_relocalize" and out["relocalize"] == ["weak"]
    pieces[0]["relocalized"] = 1
    out = _run(script, [], pieces)
    assert out["route"] == "need_more_pieces"
    out = _run(script, [], [])
    assert out["route"] == "need_more_pieces"


# --- two arms ---------------------------------------------------------------------------------

BOARD = {"origin": [-0.15, 0.125], "yaw": 0.0}
"""A 0.30 x 0.25 m board centred on the world origin; cells are 5 cm."""
ARMS = [
    {"id": "left", "reach": {"base": [-0.40, 0.0], "max_radius": 0.48}},
    {"id": "right", "reach": {"base": [0.40, 0.0], "max_radius": 0.48}},
]


def _arm_run(script, placements, pieces, arms=ARMS, **kw):
    return _run(
        script,
        placements,
        pieces,
        arms=arms,
        board_pose=BOARD,
        board_width=0.30,
        board_height=0.25,
        **kw,
    )


def test_geometry_puts_cells_where_the_board_pose_says(script):
    g = script._Geometry(COLS, ROWS, 0.30, 0.25, BOARD)
    assert np.allclose(g.centre((0, 0)), [-0.125, 0.1])
    assert np.allclose(g.centre((5, 4)), [0.125, -0.1])
    g = script._Geometry(COLS, ROWS, 0.30, 0.25, {"origin": [0, 0], "yaw": np.pi / 2})
    assert np.allclose(g.centre((0, 0)), [0.025, 0.025])


def test_each_arm_gets_reachable_separated_jobs(script):
    placements = _pl((0, 0), (5, 0))
    pieces = [
        _sure("L1", (0, 1), position=[-0.25, 0.0]),
        _sure("R1", (5, 1), position=[0.25, 0.0]),
        _sure("L2", (1, 0), position=[-0.25, 0.05]),
        _sure("R2", (4, 0), position=[0.25, 0.05]),
    ]
    out = _arm_run(script, placements, pieces)
    assert out["route"] == "chosen"
    q = out["queue"]
    assert {j["piece"] for j in q["left"]} == {"L1", "L2"}
    assert {j["piece"] for j in q["right"]} == {"R1", "R2"}
    assert out["arm"] in ("left", "right")
    a, b = q["left"][0], q["right"][0]
    assert max(abs(a["col"] - b["col"]), abs(a["row"] - b["row"])) >= script.PAIR_SEPARATION_CELLS


def test_a_job_crossing_into_the_other_arms_zone_is_unreachable(script):
    # The piece waits far right, its cell is far left: neither arm reaches the whole path.
    placements = _pl((0, 0))
    pieces = [_sure("x", (0, 1), position=[0.30, 0.0])]
    out = _arm_run(script, placements, pieces)
    assert out["route"] == "unreachable"
    assert out["unreachable"][0]["piece"] == "x"
    assert out["unreachable"][0]["why"] == {"left": "out_of_reach", "right": "out_of_reach"}
    # From a buffer in the middle the left arm can do it.
    pieces[0]["position"] = [-0.05, -0.2]
    out = _arm_run(script, placements, pieces)
    assert out["route"] == "chosen" and out["arm"] == "left"


def test_a_busy_arm_blocks_its_working_area(script):
    placements = _pl((0, 0))
    pieces = [_sure("x", (0, 1), position=[-0.25, 0.0])]
    busy = [ARMS[0], dict(ARMS[1], busy=True, busy_at=[-0.18, 0.05])]
    out = _arm_run(script, placements, pieces, arms=busy)
    assert (
        out["route"] == "unreachable" and out["unreachable"][0]["why"]["left"] == "near_busy_right"
    )
    both = [dict(a, busy=True) for a in ARMS]
    out = _arm_run(script, placements, pieces, arms=both)
    assert out["route"] == "unreachable" and "busy" in out["detail"]
    poly = [{"id": "p", "reach": {"polygon": [[-1, -1], [1, -1], [1, 1], [-1, 1]]}}]
    out = _arm_run(script, placements, pieces, arms=poly)
    assert out["route"] == "chosen" and out["arm"] == "p"


def test_bad_inputs_raise(script):
    with pytest.raises(ValueError, match="positive"):
        script.run(FakeContext(), 0, ROWS)
    with pytest.raises(ValueError, match="policy"):
        _run(script, [], [], policy="greedy")
    with pytest.raises(ValueError, match="outside"):
        _run(script, _pl((9, 9)), [])
    with pytest.raises(ValueError, match="board_pose"):
        _run(script, [], [_sure("a", (0, 0))], arms=ARMS)


# --- closed loop on a cut board ----------------------------------------------------------------


def _assemble(script, backtrack, policy, seed, style="wavy", cols=6, rows=6):
    """Place every piece of a small cut board, checking each by its shape and sending failures
    to backtracking-a-wrong-placement. Returns (correct cells, failures, shape matches)."""
    rng = np.random.default_rng(seed)
    board = sim.Board(cols, rows, seed=seed, style=style)
    confusions = []
    for _ in range(4):
        piece = sim.pid(int(rng.integers(cols)), int(rng.integers(rows)))
        wrong = (int(rng.integers(cols)), int(rng.integers(rows)))
        if wrong != sim.true_cell(piece):
            confusions.append((piece, wrong, 0.96, 0.08))
    recs = sim.synthetic_localization(cols, rows, rng, confusions=confusions)
    for _, wrong, _, _ in confusions:
        # The cell's own piece is unsure of it, so the impostor wins the cell.
        own = recs[sim.pid(*wrong)]["candidates"]
        own[0]["score"], own[1]["score"] = 0.9, 0.89
    meta, ledger, failures, shape_matches = {}, None, 0, 0
    for step in range(200):
        out = script.run(
            FakeContext(),
            cols,
            rows,
            placements=list(meta.values()),
            pieces=[r for p, r in recs.items() if p not in board.placed.values()],
            ledger=ledger,
            policy=policy,
            seed=step,
        )
        if out["route"] == "need_shape_match":
            # A perfect shape match stands in for matching-a-piece-to-placed-neighbours.
            cell = tuple(out["slot"])
            if sim.pid(*cell) not in out["shape_candidates"]:
                break
            shape_matches += 1
            piece = sim.pid(*cell)
        elif out["route"] == "chosen":
            cell, piece = (out["col"], out["row"]), out["piece"]
        else:
            break
        f = board.check(piece, cell, rng)
        if f is None:
            board.placed[cell] = piece
            meta[cell] = sim.placement_record(piece, cell, recs)
            continue
        failures += 1
        for _ in range(6):
            pl = list(meta.values()) + [sim.placement_record(f["piece"], cell, recs, False)]
            bt = backtrack.run(FakeContext(), f, pl, list(recs.values()), cols, rows, ledger=ledger)
            ledger, f = bt["ledger"], None
            for a in bt["actions"]:
                c2 = (a.get("col"), a.get("row"))
                if a["op"] == "remove":
                    board.placed.pop(c2, None)
                    meta.pop(c2, None)
                elif a["op"] in ("place", "reseat") and c2 not in board.placed:
                    f = board.check(a["piece"], c2, rng)
                    if f is not None:
                        failures += 1
                        cell = c2
                        break
                    board.placed[c2] = a["piece"]
                    meta[c2] = sim.placement_record(a["piece"], c2, recs)
            if f is None:
                break
    correct = sum(sim.true_cell(p) == c for c, p in board.placed.items())
    return correct, failures, shape_matches


def test_closed_loop_the_plan_fails_less_often_than_random_order(script, backtrack):
    """6 x 6 boards with four confident impostors whose cells' own pieces are unsure, a perfect
    shape match for the slots nothing confident fills: the plan meets the impostors where they
    have neighbours to fail against."""
    plan = [_assemble(script, backtrack, "plan", s) for s in range(6)]
    rand = [_assemble(script, backtrack, "random", s) for s in range(6)]
    assert sum(c for c, _, _ in plan) >= 0.95 * 6 * 36, plan
    assert sum(f for _, f, _ in plan) < sum(f for _, f, _ in rand), (plan, rand)
    assert sum(m for _, _, m in plan) > 0


def test_every_route_is_declared_in_the_example():
    root = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
    example = json.loads((root / "examples/canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    routes = {
        "chosen",
        "need_shape_match",
        "need_relocalize",
        "need_more_pieces",
        "unreachable",
        "done",
    }
    assert routes <= declared
    text = (root / "scripts/choose_next.py").read_text()
    for r in routes:
        assert f'route="{r}"' in text, r
