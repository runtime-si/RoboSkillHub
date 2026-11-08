"""How often backtracking-a-wrong-placement finds the true fix within 1, 2, 3 decisions.

    python skills/backtracking-a-wrong-placement/eval/benchmark.py --n 300 --out /tmp/bt.json

A 22 x 25 board has a correctly assembled 7 x 7 block at a random place. One wrong placement
is injected, then the skill is called in a loop and its actions carried out on the board:
every placement is checked by the shape of the procedural cut (tests/bundles/_backtrack_sim.py),
and a failed check is the next call's input.

- ``wrong_piece``: the piece just set down on a free cell beside the block is wrong - the
  loose piece the localizer rates most likely for that cell after the true one.
- ``wrong_neighbour``: a block cell beside the free cell holds such an impostor that passed
  its own check; then the right piece is set down on the free cell.

Solved within k: after k calls the board holds no wrong piece and the watched cells (the free
cell, and the neighbour's cell) hold their true pieces. A scenario whose injected mistake
passes the check is counted as undetected and not run.

Localizer evidence: ``localizer`` runs localizing-a-piece-in-a-reference-image on every piece
of the 22 x 25 procedural picture (photo seed 2, held out from the calibration's seeds 0
and 1; cached in --cache), ``synthetic`` draws clean records (true cell first, score
0.93-0.99, margin 0.3-0.5) and gives each impostor a confusion (score 0.85-0.95, margin
0-0.1 over the true cell).
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.bundles import _backtrack_sim as sim  # noqa: E402

COLS, ROWS = 22, 25
BOARD_M = (0.4572, 0.6096)
BLOCK = 7


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCRIPT = _load("skills/backtracking-a-wrong-placement/scripts/backtrack.py", "backtrack")
_loc = None
_ref = None


def _loc_init():
    global _loc, _ref
    from tests.bundles._jigsaw_pictures import make_reference

    _loc = _load("skills/localizing-a-piece-in-a-reference-image/scripts/locate_piece.py", "lp")
    _ref = make_reference(792, 1053, seed=0, band_rows=(0.0, 3 / ROWS), plaid=True)


def _loc_one(args):
    from gap.testing import FakeContext

    from tests.bundles._jigsaw_pictures import cut_piece

    col, row, seed, top_k = args
    ppm = 96 / (BOARD_M[0] / COLS)
    img, mask, _, _ = cut_piece(
        _ref, COLS, ROWS, col, row, 96, seed=seed * 100000 + row * 100 + col
    )
    out = _loc.run(
        FakeContext(),
        mask,
        _ref,
        COLS,
        ROWS,
        *BOARD_M,
        piece_rgb=img,
        crop_px_per_m=ppm,
        top_k=top_k,
    )
    cands = [{k: c.get(k) for k in ("col", "row", "score", "angle_deg")} for c in out["candidates"]]
    return {"piece": sim.pid(col, row), "route": out["route"], "candidates": cands}


def localizer_records(cache: Path, seed: int, top_k: int, workers: int) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    jobs = [(c, r, seed, top_k) for r in range(ROWS) for c in range(COLS)]
    with ProcessPoolExecutor(workers, initializer=_loc_init) as ex:
        recs = {r["piece"]: r for r in ex.map(_loc_one, jobs, chunksize=4)}
    cache.write_text(json.dumps(recs))
    return recs


IMPOSTOR_TRIES = 30
"""Impostors tried, most likely first, until one passes its own check (wrong_neighbour)."""


def _impostors(records, cell, exclude):
    """Loose pieces (not the true one) that list ``cell``, the localizer's likeliest first."""
    ps = []
    for piece, rec in records.items():
        if piece in exclude or sim.true_cell(piece) == cell:
            continue
        p = SCRIPT.cell_probs(rec).get(cell, {}).get("p", 0.0)
        if p > 0:
            ps.append((-p, piece))
    return [piece for _, piece in sorted(ps)]


@functools.lru_cache(maxsize=16)
def _cut(seed, style):
    return sim.Board(COLS, ROWS, seed=seed, style=style).cut


def _board(seed, style, exec_fail):
    board = sim.Board.__new__(sim.Board)
    sim.Board.__init__(board, 1, 1, seed=0, style=style, exec_fail=exec_fail)
    board.cols, board.rows, board.cut = COLS, ROWS, _cut(seed, style)
    return board


def scenario(args):
    kind, style, mode, k, records, exec_fail = args
    rng = np.random.default_rng(1000 * k + {"wrong_piece": 1, "wrong_neighbour": 2}[kind])
    board = _board(k % 8, style, exec_fail)
    c0, r0 = int(rng.integers(0, COLS - BLOCK + 1)), int(rng.integers(0, ROWS - BLOCK + 1))
    block = [(c, r) for c in range(c0, c0 + BLOCK) for r in range(r0, r0 + BLOCK)]
    frontier = []
    for c, r in block:
        for d, (dc, dr) in sim.STEP.items():
            x = (c + dc, r + dr)
            if 0 <= x[0] < COLS and 0 <= x[1] < ROWS and x not in block:
                frontier.append((x, (c, r), sim.OPPOSITE[d]))
    free, d_cell, _ = frontier[int(rng.integers(len(frontier)))]
    if mode.startswith("synthetic"):
        base = sim.synthetic_localization(COLS, ROWS, rng)
    else:
        base = records
    for cell in block:
        board.placed[cell] = sim.pid(*cell)
    on_board = set(board.placed.values())
    target = free if kind == "wrong_piece" else d_cell
    loose = [
        sim.pid(c, r)
        for r in range(ROWS)
        for c in range(COLS)
        if sim.pid(c, r) not in on_board and (c, r) != target
    ]
    if mode.startswith("synthetic"):
        lo, hi = (0.0, 0.1) if mode == "synthetic" else (0.1, 0.3)
        s_lo, s_hi = (0.85, 0.95) if mode == "synthetic" else (0.93, 0.99)
        order = [loose[i] for i in rng.permutation(len(loose))[:IMPOSTOR_TRIES]]
    else:
        order = _impostors(base, target, on_board)[:IMPOSTOR_TRIES]
    imp, recs = None, base
    watch, failure = [], None
    if kind == "wrong_neighbour":
        # The impostor was set down beside one block piece (the anchor) and passed against
        # it; the block cells it touches besides the anchor are left empty.
        around = [(d_cell[0] + dc, d_cell[1] + dr) for dc, dr in sim.STEP.values()]
        around = [x for x in around if x in board.placed and x != d_cell]
        anchor = around[int(rng.integers(len(around)))]
        for x in around:
            if x != anchor:
                board.placed.pop(x)
        board.placed.pop(d_cell)
        watch = [d_cell]
    for cand in order:
        if mode.startswith("synthetic"):
            conf = [(cand, target, float(rng.uniform(s_lo, s_hi)), float(rng.uniform(lo, hi)))]
            rec = sim.synthetic_localization(COLS, ROWS, np.random.default_rng(k), confusions=conf)
            recs = dict(base, **{cand: rec[cand]})
        if kind == "wrong_piece" or board.check(cand, d_cell, rng) is None:
            imp = cand
            break
    if imp is None:
        return {"status": "impostor_caught_earlier" if order else "no_impostor"}
    if mode == "synthetic_decoy":
        # A second wrong piece the localizer half-believes belongs in the free cell, so
        # blaming the right piece there does not leave the cell without a candidate.
        decoy = next(x for x in rng.permutation(loose) if x not in (imp, sim.pid(*free)))
        dc = [(decoy, free, float(rng.uniform(0.88, 0.93)), float(rng.uniform(0.02, 0.06)))]
        recs = dict(
            recs,
            **{
                decoy: sim.synthetic_localization(
                    COLS, ROWS, np.random.default_rng(k + 1), confusions=dc
                )[decoy]
            },
        )
    true_p = SCRIPT.cell_probs(recs.get(sim.pid(*target), {})).get(target, {}).get("p", 0.0)
    natural = SCRIPT.cell_probs(recs[imp]).get(target, {}).get("p", 0.0) >= true_p
    if kind == "wrong_piece":
        failure = board.check(imp, free, rng)
    else:
        board.placed[d_cell] = imp
        failure = board.check(sim.pid(*free), free, rng)
    if failure is None:
        return {"status": "undetected"}
    listed = all(cell in SCRIPT.cell_probs(recs.get(sim.pid(*cell), {})) for cell in [free, *watch])
    t = time.time()
    solved, routes, ops = sim.run_episode(
        SCRIPT, board, recs, failure, rng, max_calls=6, watch=watch
    )
    if solved is not None:
        end = "solved"
    elif not board.correct():
        end = "wrong_piece_left"  # a later wrong placement passed its check, or calls ran out
    else:
        end = "cell_left_empty"
    return {
        "status": "run",
        "solved": solved,
        "routes": routes,
        "ops": ops,
        "natural": bool(natural),
        "first_kind": failure["kind"],
        "end": end,
        "listed": listed,
        "s_per_call": (time.time() - t) / len(routes),
    }


def summarise(rows):
    run = [r for r in rows if r["status"] == "run"]
    st = Counter(r["status"] for r in rows)
    out = {"n": len(rows), "status": dict(st), "ran": len(run)}
    for k in (1, 2, 3, 6):
        out[f"within_{k}"] = round(
            sum(r["solved"] is not None and r["solved"] <= k for r in run) / max(1, len(run)), 3
        )
    out["first_route"] = dict(Counter(r["routes"][0] for r in run))
    out["end"] = dict(Counter(r["end"] for r in run))
    out["final_route_unsolved"] = dict(Counter(r["routes"][-1] for r in run if r["solved"] is None))
    ok = [r["ops"] for r in run if r["solved"] is not None]
    out["ops_median_solved"] = float(np.median(ok)) if ok else None
    out["s_per_call_median"] = (
        round(float(np.median([r["s_per_call"] for r in run])), 3) if run else None
    )
    lst = [r for r in run if r["listed"]]
    out["listed_n"] = len(lst)
    for k in (1, 2, 3):
        out[f"listed_within_{k}"] = round(
            sum(r["solved"] is not None and r["solved"] <= k for r in lst) / max(1, len(lst)), 3
        )
    nat = [r for r in run if r["natural"]]
    out["natural_n"] = len(nat)
    out["natural_within_3"] = round(
        sum(r["solved"] is not None and r["solved"] <= 3 for r in nat) / max(1, len(nat)), 3
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cache", default="/tmp/bt_loc_seed2.json")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--exec-fail", type=float, default=0.0)
    ap.add_argument("--modes", default="localizer,synthetic,synthetic_hard,synthetic_decoy")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    records = (
        localizer_records(Path(a.cache), 2, a.top_k, a.workers) if "localizer" in a.modes else {}
    )
    results = {}
    for mode in a.modes.split(","):
        for style in ("classic", "wavy"):
            for kind in ("wrong_piece", "wrong_neighbour"):
                jobs = [(kind, style, mode, k, records, a.exec_fail) for k in range(a.n)]
                with ProcessPoolExecutor(a.workers) as ex:
                    rows = list(ex.map(scenario, jobs, chunksize=4))
                s = summarise(rows)
                results[f"{mode}/{style}/{kind}"] = s
                print(f"{mode:9s} {style:7s} {kind:15s} {json.dumps(s)}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
