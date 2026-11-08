"""backtracking-a-wrong-placement: wrong jigsaw placements are undone on boards checked by the cut's shape."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

from . import _backtrack_sim as sim

BUNDLE = "backtracking-a-wrong-placement"
COLS, ROWS = 8, 8


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["backtrack"].module


def _cand(cells_scores):
    return [{"col": c, "row": r, "score": s, "angle_deg": 0.0} for (c, r), s in cells_scores]


def _rec(piece, cells_scores):
    return {"piece": piece, "candidates": _cand(cells_scores)}


def _run(script, failure, placements, pieces, **kw):
    return script.run(FakeContext(), failure, placements, pieces, COLS, ROWS, **kw)


def _ops(out):
    return [(a["op"], a.get("piece"), a.get("col"), a.get("row")) for a in out["actions"]]


# --- the pieces of the decision ----------------------------------------------------------


def test_calibration_is_monotonic_and_clipped(script):
    assert script.calibrated(0.97, 0.4) == script.CONF_MAX
    assert script.calibrated(0.5, -0.2) == script.CONF_MIN
    assert (
        script.calibrated(0.9, 0.02) < script.calibrated(0.9, 0.06) < script.calibrated(0.9, 0.12)
    )
    assert script.calibrated(0.88, 0.05) < script.calibrated(0.94, 0.05)
    # Wrong top-1 answers sit near margin 0 and score 0.88 (measured): under one in four.
    assert script.calibrated(0.88, 0.0) < 0.25


def test_cell_probs_use_the_margin_to_the_best_other_cell_and_share_flat_shortlists(script):
    probs = script.cell_probs(_rec("a", [((1, 1), 0.97), ((2, 2), 0.55), ((3, 3), 0.5)]))
    assert probs[1, 1]["p"] == script.CONF_MAX
    assert probs[2, 2]["p"] < 0.05
    flat = {"piece": "f", "candidates": [{"col": c, "row": 0, "score": None} for c in range(4)]}
    fp = script.cell_probs(flat)
    assert set(fp) == {(c, 0) for c in range(4)}
    assert all(v["p"] == pytest.approx(0.25) for v in fp.values())


def test_posterior_one_failure_blames_the_less_trusted_pair(script):
    marg, none = script.culprit_posterior({"a": 0.98, "b": 0.6}, [["a", "b"]], 0.05)
    assert marg["b"] > marg["a"]
    assert 0 < none[0] < 1
    marg, _ = script.culprit_posterior({"a": 0.9, "b": 0.9}, [["a", "b"]], 0.05)
    assert marg["a"] == pytest.approx(marg["b"])


def test_posterior_two_failures_sharing_a_neighbour_blame_it(script):
    # Two different pieces each failed the same slot beside n: n explains both.
    v = {"p1": 0.8, "p2": 0.8, "n": 0.8}
    marg, none = script.culprit_posterior(v, [["p1", "n"], ["p2", "n"]], 0.05)
    assert marg["n"] > marg["p1"] and marg["n"] > marg["p2"]
    assert none[0] < 0.2


def test_reassign_is_a_global_assignment_and_respects_the_ledger(script):
    # Greedy would give a cell (0, 0) (its best) and leave b with nothing it likes.
    probs = {
        "a": {(0, 0): {"p": 0.9, "angle_deg": 0}, (1, 0): {"p": 0.85, "angle_deg": 0}},
        "b": {(0, 0): {"p": 0.8, "angle_deg": 0}, (5, 5): {"p": 0.005, "angle_deg": 0}},
    }
    book = script.new_ledger(COLS, ROWS)
    out = {
        (x["piece"]): (x["col"], x["row"])
        for x in script.reassign(["a", "b"], probs, [(0, 0), (1, 0), (5, 5)], book)
    }
    assert out == {"a": (1, 0), "b": (0, 0)}
    book["forbidden"].append({"piece": "b", "col": 0, "row": 0})
    out = {
        (x["piece"]): (x["col"], x["row"])
        for x in script.reassign(["a", "b"], probs, [(0, 0), (1, 0), (5, 5)], book)
    }
    assert out == {"a": (0, 0)}  # b's only other cell is under MIN_ASSIGN_PROB: it stays loose
    out = script.reassign(["a", "b"], probs, [(0, 0), (1, 0)], book, exclude={"a@0,0", "a@1,0"})
    assert out == []


# --- the routes on hand-built boards ------------------------------------------------------


def _row_board():
    """Cells (2,3) and (4,3) hold confident pieces; the slot is (3,3)."""
    pieces = [
        _rec("L", [((2, 3), 0.97), ((6, 6), 0.5)]),
        _rec("R", [((4, 3), 0.97), ((6, 5), 0.5)]),
        _rec("T", [((3, 3), 0.97), ((6, 4), 0.5)]),
        _rec("W", [((3, 3), 0.89), ((5, 0), 0.88)]),
    ]
    placements = [
        {"piece": "L", "col": 2, "row": 3, "score": 0.97, "margin": 0.47, "verified": True},
        {"piece": "R", "col": 4, "row": 3, "score": 0.97, "margin": 0.47, "verified": True},
    ]
    return pieces, placements


def test_a_doubtful_piece_that_does_not_seat_is_reassigned(script, tmp_path):
    pieces, placements = _row_board()
    path = tmp_path / "ledger.json"
    fail = {"piece": "W", "col": 3, "row": 3, "kind": "not_seated", "seam_scores": {"L": 0.11}}
    out = _run(script, fail, placements, pieces, ledger_path=str(path))
    assert out["route"] == "reassigned", out["detail"]
    assert out["culprit_piece"] == "W" and out["culprit_cell"] == [3, 3]
    assert _ops(out)[:2] == [("remove", "W", 3, 3), ("place", "T", 3, 3)]
    assert out["actions"][1]["attempt"] == 1 and out["actions"][1]["max_attempts"] == 2
    # W's other listed cell is free: it is set aside for it, not placed now.
    assert out["actions"][2] == {
        "op": "set_aside",
        "piece": "W",
        "planned_cell": [5, 0],
        "prob": pytest.approx(out["actions"][2]["prob"]),
    }
    saved = json.loads(path.read_text())
    assert saved == out["ledger"] and out["ledger_path"] == str(path)
    assert {(f["piece"], f["col"], f["row"]) for f in saved["forbidden"]} == {("W", 3, 3)}
    assert saved["attempts"] == {"W@3,3": 1} and saved["backtracks"] == 1
    # not_seated: every placed neighbour is a suspect unless its seam was reported good.
    assert out["suspects"][0]["piece"] == "W"
    assert {s["piece"] for s in out["suspects"]} == {"W", "L", "R"}
    assert out["n_actions"] == len(out["actions"])


def test_a_doubtful_neighbour_is_removed_and_the_slot_retried(script):
    pieces = [
        _rec("L", [((2, 3), 0.89), ((6, 6), 0.88)]),  # placed on thin evidence
        _rec("TL", [((2, 3), 0.97), ((6, 2), 0.5)]),  # loose: L's cell's true piece
        _rec("P", [((3, 3), 0.97), ((6, 4), 0.5)]),
        _rec("R", [((4, 3), 0.97), ((6, 5), 0.5)]),
    ]
    placements = [
        {"piece": "L", "col": 2, "row": 3, "score": 0.89, "margin": 0.01},
        {"piece": "R", "col": 4, "row": 3, "score": 0.97, "margin": 0.47, "verified": True},
    ]
    fail = {
        "piece": "P",
        "col": 3,
        "row": 3,
        "kind": "seam_mismatch",
        "seam_scores": {"west": 0.12, "east": 0.02},
    }
    out = _run(script, fail, placements, pieces)
    assert out["route"] == "remove_neighbour", out["detail"]
    assert out["culprit_piece"] == "L"
    assert [s["piece"] for s in out["suspects"]] == ["L", "P"]  # R's good seam cleared it
    assert _ops(out)[:4] == [
        ("remove", "P", 3, 3),
        ("remove", "L", 2, 3),
        ("place", "TL", 2, 3),
        ("place", "P", 3, 3),
    ]
    assert out["ledger"]["neighbour_removals"] == {"3,3": 1}
    assert out["actions"][3]["attempt"] == 2  # P's second try at (3, 3)


def test_a_confident_board_seats_the_piece_again_first(script):
    pieces, placements = _row_board()
    fail = {"piece": "T", "col": 3, "row": 3, "kind": "not_seated"}
    out = _run(script, fail, placements, pieces)
    assert out["route"] == "retry_seat", out["detail"]
    assert _ops(out) == [("reseat", "T", 3, 3)]
    assert out["actions"][0]["attempt"] == 2
    assert out["p_false_alarm"] > out["suspects"][0]["p_wrong"]
    # The ladder has one rung: the same failure again is decided on.
    again = _run(script, fail, placements, pieces, ledger=out["ledger"])
    assert again["route"] != "retry_seat"
    assert again["ledger"]["attempts"] == {"T@3,3": 2}


def test_a_bad_seam_is_not_retried(script):
    pieces, placements = _row_board()
    fail = {"piece": "T", "col": 3, "row": 3, "kind": "not_seated", "seam_scores": {"L": 0.15}}
    out = _run(script, fail, placements, pieces)
    assert out["route"] != "retry_seat"


def test_recheck_failed_suspects_only_the_piece(script):
    pieces, placements = _row_board()
    fail = {"piece": "W", "col": 3, "row": 3, "kind": "recheck_failed"}
    out = _run(script, fail, placements, pieces)
    assert [s["piece"] for s in out["suspects"]] == ["W"]
    assert out["route"] == "reassigned"


def test_a_locked_neighbour_is_never_blamed(script):
    pieces, placements = _row_board()
    placements[0] = dict(placements[0], score=0.89, margin=0.0, verified=False, locked=True)
    fail = {"piece": "T", "col": 3, "row": 3, "kind": "seam_mismatch", "seam_scores": {"L": 0.2}}
    out = _run(script, fail, placements, pieces)
    assert [s["piece"] for s in out["suspects"]] == ["T"]
    assert out["culprit_piece"] == "T"


def test_two_pieces_failing_one_slot_turn_the_blame_on_the_neighbour(script, tmp_path):
    """The rule of thumb, from the ledger: the second different piece that fails beside the
    same neighbour makes the neighbour the likelier culprit, across a resume from disk."""
    pieces = [
        _rec("N", [((2, 3), 0.93), ((6, 6), 0.857)]),  # wrong; P(right) 0.87
        _rec("TN", [((2, 3), 0.94), ((6, 2), 0.6)]),
        _rec("A", [((3, 3), 0.93), ((0, 7), 0.871)]),  # right; P(right) 0.77
        _rec("B", [((3, 3), 0.93), ((1, 7), 0.871)]),  # wrong; P(right) 0.77
        _rec("R", [((4, 3), 0.97), ((6, 5), 0.5)]),
    ]
    placements = [
        {"piece": "N", "col": 2, "row": 3},
        {"piece": "R", "col": 4, "row": 3, "verified": True},
    ]
    path = str(tmp_path / "ledger.json")
    f1 = {
        "piece": "A",
        "col": 3,
        "row": 3,
        "kind": "seam_mismatch",
        "seam_scores": {"N": 0.1, "R": 0.02},
        "event_id": "e1",
    }
    out1 = _run(script, f1, placements, pieces, ledger_path=path)
    assert out1["route"] == "reassigned" and out1["culprit_piece"] == "A", out1["detail"]
    assert ("place", "B", 3, 3) in _ops(out1)
    f2 = {
        "piece": "B",
        "col": 3,
        "row": 3,
        "kind": "seam_mismatch",
        "seam_scores": {"N": 0.1, "R": 0.02},
        "event_id": "e2",
    }
    out2 = _run(script, f2, placements, pieces, ledger_path=path)
    assert out2["route"] == "remove_neighbour", out2["detail"]
    assert out2["culprit_piece"] == "N"
    assert ("remove", "N", 2, 3) in _ops(out2) and ("place", "TN", 2, 3) in _ops(out2)
    # A was blamed on weak evidence only: it is back in the running for (3, 3).
    assert out2["ledger"]["p_right"]["A@3,3"] > 0.5
    assert len(json.loads(Path(path).read_text())["events"]) == 2


def test_replaying_an_event_returns_the_stored_decision(script, tmp_path):
    pieces, placements = _row_board()
    path = str(tmp_path / "ledger.json")
    fail = {
        "piece": "W",
        "col": 3,
        "row": 3,
        "kind": "not_seated",
        "seam_scores": {"L": 0.11},
        "event_id": "x",
    }
    a = _run(script, fail, placements, pieces, ledger_path=path)
    b = _run(script, fail, placements, pieces, ledger_path=path)
    assert b["route"] == a["route"] and b["actions"] == a["actions"]
    assert b["detail"].startswith("replayed")
    assert b["ledger"]["attempts"] == {"W@3,3": 1} and len(b["ledger"]["events"]) == 1


def test_no_known_piece_for_the_slot_asks_for_a_search(script):
    pieces = [
        _rec("L", [((2, 3), 0.97), ((6, 6), 0.5)]),
        _rec("W", [((3, 3), 0.88)]),
    ]
    placements = [{"piece": "L", "col": 2, "row": 3, "score": 0.97, "margin": 0.47, "locked": True}]
    fail = {"piece": "W", "col": 3, "row": 3, "kind": "seam_mismatch", "seam_scores": {"L": 0.2}}
    out = _run(script, fail, placements, pieces)
    assert out["route"] == "no_alternative", out["detail"]
    assert _ops(out) == [
        ("remove", "W", 3, 3),
        ("relocalize", "W", None, None),
        ("search_for_cell", None, 3, 3),
    ]
    assert out["actions"][1]["exclude_cells"] == [[3, 3]]
    assert out["actions"][2]["exclude_pieces"] == ["W"]


def test_a_slot_that_keeps_failing_is_given_up_and_left_out(script):
    pieces = [_rec("L", [((2, 3), 0.97), ((6, 6), 0.5)])]
    pieces += [
        _rec(f"w{k}", [((3, 3), 0.9 - 0.01 * k), ((7, k), 0.88 - 0.01 * k)]) for k in range(5)
    ]
    placements = [{"piece": "L", "col": 2, "row": 3, "score": 0.97, "margin": 0.47, "locked": True}]
    book = None
    routes = []
    for k in range(4):
        fail = {
            "piece": f"w{k}",
            "col": 3,
            "row": 3,
            "kind": "seam_mismatch",
            "seam_scores": {"L": 0.2},
        }
        out = _run(script, fail, placements, pieces, ledger=book, max_pieces_per_cell=3)
        book = out["ledger"]
        routes.append(out["route"])
    assert routes == ["reassigned", "reassigned", "reassigned", "give_up"]
    assert _ops(out)[:2] == [("remove", "w3", 3, 3), ("defer_cell", None, 3, 3)]
    assert book["deferred_cells"] == [[3, 3]]
    # A later failure elsewhere never plans a piece into the deferred cell.
    fail = {"piece": "w4", "col": 7, "row": 4, "kind": "recheck_failed"}
    out = _run(script, fail, placements, pieces, ledger=book)
    assert all((a["col"], a["row"]) != (3, 3) for a in out["assignment"])


def test_neighbour_removals_are_capped_per_slot(script):
    pieces = [
        _rec("L", [((2, 3), 0.89), ((6, 6), 0.88)]),
        _rec("TL", [((2, 3), 0.97), ((6, 2), 0.5)]),
        _rec("P", [((3, 3), 0.97), ((6, 4), 0.5)]),
    ]
    placements = [{"piece": "L", "col": 2, "row": 3, "score": 0.89, "margin": 0.01}]
    fail = {"piece": "P", "col": 3, "row": 3, "kind": "seam_mismatch", "seam_scores": {"L": 0.12}}
    out = _run(script, fail, placements, pieces, max_neighbour_removals=0)
    assert out["culprit_piece"] == "P"
    assert out["suspects"][0]["piece"] == "L"  # still the likelier one, but out of removals


def test_the_backtrack_budget_gives_up(script):
    pieces, placements = _row_board()
    book = script.new_ledger(COLS, ROWS)
    book["backtracks"] = 5
    fail = {"piece": "W", "col": 3, "row": 3, "kind": "seam_mismatch", "seam_scores": {"L": 0.2}}
    out = _run(script, fail, placements, pieces, ledger=book, max_backtracks=5)
    assert out["route"] == "give_up" and "5 backtracks" in out["detail"]


def test_a_shape_placed_piece_keeps_its_confidence_in_the_resolve(script):
    """A piece the localizer did not list for its cell (placed by a shape match) can be put
    back there after its neighbour is replaced."""
    pieces = [
        _rec("L", [((2, 3), 0.89), ((6, 6), 0.88)]),
        _rec("TL", [((2, 3), 0.97), ((6, 2), 0.5)]),
        _rec("P", [((0, 0), 0.6)]),
    ]
    placements = [
        {"piece": "L", "col": 2, "row": 3, "score": 0.89, "margin": 0.01},
        {"piece": "P", "col": 3, "row": 3, "confidence": 0.9},
    ]
    fail = {"piece": "P", "col": 3, "row": 3, "kind": "seam_mismatch", "seam_scores": {"L": 0.12}}
    out = _run(script, fail, placements, pieces)
    assert out["route"] == "remove_neighbour", out["detail"]
    assert ("place", "P", 3, 3) in _ops(out)


def test_bad_inputs_raise(script):
    pieces, placements = _row_board()
    ok = {"piece": "W", "col": 3, "row": 3, "kind": "not_seated"}
    with pytest.raises(ValueError, match="kind"):
        _run(script, dict(ok, kind="wobbly"), placements, pieces)
    with pytest.raises(ValueError, match="outside"):
        _run(script, dict(ok, col=99), placements, pieces)
    with pytest.raises(ValueError, match="grid"):
        _run(script, ok, placements, pieces, ledger=script.new_ledger(3, 3))
    with pytest.raises(ValueError, match="positive"):
        script.run(FakeContext(), ok, placements, pieces, 0, ROWS)


# --- closed loop on a cut board -----------------------------------------------------------


@pytest.mark.parametrize("style", ["classic", "wavy"])
def test_closed_loop_fixes_a_wrong_piece_and_a_wrong_neighbour(script, style):
    rng = np.random.default_rng(3)
    solved = {"piece": 0, "neighbour": 0}
    tried = {"piece": 0, "neighbour": 0}
    for trial in range(12):
        board = sim.Board(COLS, ROWS, seed=trial, style=style)
        block = [(c, r) for c in range(2, 5) for r in range(2, 5)]
        free, d_cell = (5, 3), (4, 3)
        for cell in block:
            board.placed[cell] = sim.pid(*cell)
        imp = sim.pid(*[(0, 0), (7, 7), (0, 7), (7, 0), (6, 6), (1, 6)][trial % 6])
        kind = "piece" if trial % 2 == 0 else "neighbour"
        target = free if kind == "piece" else d_cell
        recs = sim.synthetic_localization(COLS, ROWS, rng, confusions=[(imp, target, 0.9, 0.03)])
        watch = []
        if kind == "piece":
            failure = board.check(imp, free, rng)
        else:
            board.placed[d_cell] = imp
            failure = board.check(sim.pid(*free), free, rng)
            watch = [d_cell]
        if failure is None:
            continue
        tried[kind] += 1
        n, routes, _ = sim.run_episode(script, board, recs, failure, rng, watch=watch)
        solved[kind] += n is not None and n <= 3
    assert tried["piece"] >= 3 and tried["neighbour"] >= 3
    assert solved == tried


def test_every_route_is_declared_in_the_example():
    root = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
    example = json.loads((root / "examples/canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == {"reassigned", "remove_neighbour", "retry_seat", "no_alternative", "give_up"}
