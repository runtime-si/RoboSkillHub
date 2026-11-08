"""Assemble a whole 20 x 27 procedural jigsaw with each ordering policy and count the failures.

    python skills/choosing-next-piece-and-slot/eval/assembly.py --loc /tmp/np_loc_s2.json \
        --loc-alt /tmp/np_loc_s3.json --n 6 --out /tmp/np_assembly.json

Each episode: a procedural cut (tests/bundles/jigsaw_cut.py, classic or wavy) of the target's
20 x 27 grid, every piece known and loose. The skill is called in a loop and its answer carried
out:

- ``chosen``: the piece is set down and checked by the cut's shape against every placed
  neighbour (tests/bundles/_backtrack_sim.py: a tab on a tab does not seat, a seam gap over
  0.06 of the side fails; a piece with no placed neighbour always passes). A failed check goes
  to backtracking-a-wrong-placement, whose actions are carried out until the board passes.
- ``need_shape_match``: a stand-in for matching-a-piece-to-placed-neighbours answers ``fits``
  at its measured rate and precision for the cut and the slot's neighbour count
  (``SHAPE_FITS``), else ``ambiguous`` (the slot is skipped until the next placement); with the
  true piece absent it answers ``no_fit`` (skipped the same way). A ``fits`` pick is set down and checked at once (as a graph
  wires the match to the place) and written into its record as a candidate with confidence
  0.9. When every slot left was skipped and every piece is known, the next match places its
  top answer anyway (a test placement), right at the skill's measured top-1 rate
  (``SHAPE_TOP1``).
- ``need_relocalize``: the listed pieces get their record from a second photo (``--loc-alt``).
- ``need_more_pieces``: with ``--batch`` the next batch of pieces becomes known; otherwise the
  episode is stuck.
- ``unreachable`` (``--arms``): the pieces are moved to a buffer both arms reach.

Localizer evidence: ``localizer`` = localizing-a-piece-in-a-reference-image's records for every
piece (eval/localize_all.py; 81 % top-1 right on this picture), ``synth10`` / ``synth25`` =
records with 10 % / 25 % of top-1 answers wrong (see ``synthetic_records``).
"""

from __future__ import annotations

import argparse
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

COLS, ROWS = 20, 27
BOARD_M = (0.4572, 0.6096)
SHAPE_TOP1 = {
    ("wavy", 10): {1: 0.91, 2: 0.89, 3: 0.94},
    ("wavy", 50): {1: 0.78, 2: 0.91, 3: 0.94},
    ("classic", 10): {1: 0.46, 2: 0.94, 3: 0.99},
    ("classic", 50): {1: 0.19, 2: 0.85, 3: 0.99},
}
"""matching-a-piece-to-placed-neighbours' measured top-1 (noisy masks) by cut, candidate count
(10, or 50 for more than 10) and placed neighbours: used for a forced test placement."""
SHAPE_FITS = {
    ("wavy", 1): (0.45, 0.985),
    ("classic", 1): (0.02, 0.81),
    ("any", 2): (0.6, 0.99),
    ("any", 3): (0.9, 0.99),
}
"""(share of slots answered ``fits``, share of those right) from the same skill's SKILL.md: one
neighbour 45 % / 98.5 % wavy, 2 % / 81 % classic; 2-3 neighbours 47-95 % answered, 98.1-100 %
right (split here as 60 % at two and 90 % at three neighbours, an assumption)."""
SHAPE_CONFIDENCE = 0.9
BATCH_DEFAULT = 0


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CHOOSE = _load("skills/choosing-next-piece-and-slot/scripts/choose_next.py", "choose_next")
BACKTRACK = _load("skills/backtracking-a-wrong-placement/scripts/backtrack.py", "backtrack")


def synthetic_records(rng, err, top_k=5):
    """Every piece's true cell first (score 0.88-0.99, margin 0.02-0.4) except a fraction
    ``err`` whose first cell is a random wrong one (score 0.80-0.95, margin 0-0.08), the true
    cell listed second in 70 % of those."""
    cells = [(c, r) for r in range(ROWS) for c in range(COLS)]
    recs = {}
    for col, row in cells:
        others = [cells[k] for k in rng.choice(len(cells), top_k + 2, replace=False)]
        others = [x for x in others if x != (col, row)]
        if rng.random() < err:
            s = float(rng.uniform(0.80, 0.95))
            first = {"col": others[0][0], "row": others[0][1], "score": s, "angle_deg": 0.0}
            rest = others[1 : top_k - 1]
            s2 = s - float(rng.uniform(0.0, 0.08))
            cands = [first]
            if rng.random() < 0.7:
                cands.append({"col": col, "row": row, "score": s2, "angle_deg": 0.0})
            cands += [
                {"col": x[0], "row": x[1], "score": s2 - 0.1 - 0.02 * k, "angle_deg": 0.0}
                for k, x in enumerate(rest)
            ]
        else:
            s = float(rng.uniform(0.88, 0.99))
            s2 = s - float(rng.uniform(0.02, 0.4))
            cands = [{"col": col, "row": row, "score": s, "angle_deg": 0.0}]
            cands += [
                {"col": x[0], "row": x[1], "score": s2 - 0.02 * k, "angle_deg": 0.0}
                for k, x in enumerate(others[: top_k - 1])
            ]
        recs[sim.pid(col, row)] = {"piece": sim.pid(col, row), "candidates": cands}
    return recs


ARMS = [
    {"id": "left", "reach": {"base": [-0.45, 0.0], "max_radius": 0.60}},
    {"id": "right", "reach": {"base": [0.45, 0.0], "max_radius": 0.60}},
]
"""Two YAMs facing each other across the board (0.6 m usable reach, an assumption)."""
BOARD_POSE = {"origin": [-BOARD_M[0] / 2, BOARD_M[1] / 2], "yaw": 0.0}
BUFFER = [0.0, -0.36]
"""A shared buffer below the board that both arms reach (0.58 m from each base)."""


def _staging(rng, piece):
    """Pieces wait beside the board on their target's side (a staging skill's job); 10 % on the
    wrong side."""
    col, _ = sim.true_cell(piece)
    side = -1 if col < COLS / 2 else 1
    if rng.random() < 0.1:
        side = -side
    return [side * float(rng.uniform(0.27, 0.36)), float(rng.uniform(-0.29, 0.29))]


def episode(args):
    policy, mode, style, seed, opt = args
    rng = np.random.default_rng(seed * 7919 + {"wavy": 1, "classic": 2}[style])
    board = sim.Board(COLS, ROWS, seed=seed, style=style)
    if mode == "localizer":
        recs = {p: dict(r) for p, r in opt["loc"].items()}
        alt = opt["loc_alt"]
    else:
        err = {"synth10": 0.10, "synth25": 0.25}[mode]
        recs = synthetic_records(np.random.default_rng(seed), err)
        alt = synthetic_records(np.random.default_rng(seed + 10_000), err)
    if opt.get("classes", True):
        for p, r in recs.items():
            c, rw = sim.true_cell(p)
            r["piece_class"] = board.cut.piece_class(c, ROWS - 1 - rw)
    arms = opt.get("arms", False)
    if arms:
        for p, r in recs.items():
            r["position"] = _staging(rng, p)
    order = list(recs)
    rng.shuffle(order)
    batch = int(opt.get("batch", 0)) or len(order)
    known = set(order[:batch])
    revealed = batch
    meta: dict[tuple[int, int], dict] = {}
    ledger = None
    skip: list[list[int]] = []
    st = Counter()
    t_choose = []

    def placed_ids():
        return set(board.placed.values())

    def nbs(cell):
        return sum((cell[0] + dc, cell[1] + dr) in board.placed for dc, dr in sim.STEP.values())

    def record_for(piece, cell, verified):
        return sim.placement_record(piece, cell, recs, verified=verified)

    def backtrack(failure):
        nonlocal ledger
        for _ in range(8):
            st["backtrack_calls"] += 1
            pl = list(meta.values()) + [
                record_for(failure["piece"], (failure["col"], failure["row"]), False)
            ]
            out = BACKTRACK.run(
                None, failure, pl, [recs[p] for p in known], COLS, ROWS, ledger=ledger
            )
            ledger = out["ledger"]
            st["bt_" + out["route"]] += 1
            failure = None
            for a in out["actions"]:
                if a["op"] == "remove":
                    cell = (a["col"], a["row"])
                    board.placed.pop(cell, None)
                    meta.pop(cell, None)
                    st["removals"] += 1
                elif a["op"] in ("place", "reseat"):
                    cell = (a["col"], a["row"])
                    if a["piece"] in placed_ids() or cell in board.placed:
                        continue
                    st["attempts"] += 1
                    f = board.check(a["piece"], cell, rng)
                    if f is not None:
                        st["failures"] += 1
                        failure = f
                        break
                    st["passes"] += 1
                    board.placed[cell] = a["piece"]
                    meta[cell] = record_for(a["piece"], cell, True)
                elif a["op"] == "relocalize" and a["piece"] in alt:
                    recs[a["piece"]] = dict(alt[a["piece"]], **_keep(recs[a["piece"]]))
                    st["relocalized_by_backtrack"] += 1
            if failure is None:
                return

    def attempt(piece, cell):
        st["attempts"] += 1
        f = board.check(piece, cell, rng)
        if f is None:
            st["passes"] += 1
            board.placed[cell] = piece
            meta[cell] = record_for(piece, cell, nbs(cell) > 0)
            # A placement changes only its neighbours' slots; other skipped slots stay skipped.
            near = {(cell[0] + dc, cell[1] + dr) for dc, dr in sim.STEP.values()}
            skip[:] = [x for x in skip if tuple(x) not in near]
            return True
        st["failures"] += 1
        backtrack(f)
        return False

    stop = "max_steps"
    rounds = 0
    forced = False
    for step in range(int(opt.get("max_steps", 2500))):
        loose = [p for p in known if p not in placed_ids()]
        kw = {}
        if arms:
            kw = dict(
                arms=ARMS, board_pose=BOARD_POSE, board_width=BOARD_M[0], board_height=BOARD_M[1]
            )
        t = time.time()
        out = CHOOSE.run(
            None,
            COLS,
            ROWS,
            placements=list(meta.values()),
            pieces=[recs[p] for p in sorted(loose)],
            ledger=ledger,
            exclude_cells=skip,
            policy=policy,
            seed=seed * 100_000 + step,
            neighbour_weight=opt.get("neighbour_weight", CHOOSE.NEIGHBOUR_WEIGHT),
            min_confidence=opt.get("min_confidence", CHOOSE.MIN_CONFIDENCE),
            frame_weight=opt.get("frame_weight", CHOOSE.FRAME_WEIGHT),
            **kw,
        )
        t_choose.append(time.time() - t)
        route = out["route"]
        st["route_" + route] += 1
        if route == "done":
            stop = "done"
            break
        if route == "chosen":
            rounds += 1
            jobs = [out]
            if arms:
                heads = [q[0] for q in out["queue"].values() if q]
                jobs = [out] + [h for h in heads if h["piece"] != out["piece"]][:1]
                st["parallel_rounds"] += len(jobs) > 1
            for j in jobs:
                cell = (j["col"], j["row"])
                if j["piece"] in placed_ids() or cell in board.placed:
                    continue
                attempt(j["piece"], cell)
        elif route == "need_shape_match":
            slot = tuple(out["slot"])
            cands = out["shape_candidates"]
            true = sim.pid(*slot)
            n = min(3, max(1, out["n_neighbours"]))
            st["shape_matches"] += 1
            fits_rate, precision = SHAPE_FITS.get((style, n), SHAPE_FITS.get(("any", n)))
            pick = None
            if true in cands and (forced or rng.random() < fits_rate):
                ok = rng.random() < (
                    SHAPE_TOP1[style, 10 if len(cands) <= 10 else 50][n] if forced else precision
                )
                wrong = [c for c in cands if c != true]
                pick = true if ok or not wrong else wrong[int(rng.integers(len(wrong)))]
                st["shape_forced"] += forced
            elif true not in cands and forced and len(cands) > 0:
                pick = cands[int(rng.integers(len(cands)))]
                st["shape_forced"] += 1
            if pick is None:
                st["shape_ambiguous" if true in cands else "shape_no_fit"] += 1
                skip.append(list(slot))
                continue
            forced = False
            st["shape_right"] += pick == true
            r = dict(recs[pick])
            r["candidates"] = [
                {"col": slot[0], "row": slot[1], "confidence": SHAPE_CONFIDENCE, "angle_deg": 0.0}
            ] + [c for c in r.get("candidates") or [] if (c["col"], c["row"]) != slot]
            recs[pick] = r
            rounds += 1
            attempt(pick, slot)
            continue
        if route != "chosen" and skip and revealed >= len(order):
            # Every slot left was ambiguous: try again, placing the shape match's top answer.
            skip.clear()
            forced = True
            st["forced_rounds"] += 1
            continue
        elif route == "need_relocalize":
            for p in out["relocalize"]:
                n_rel = int(recs[p].get("relocalized", 0)) + 1
                recs[p] = dict(alt[p], **_keep(recs[p]), relocalized=n_rel)
                st["relocalized"] += 1
            skip.clear()
        elif route == "need_more_pieces":
            if revealed < len(order):
                known |= set(order[revealed : revealed + batch])
                revealed += batch
                skip.clear()
            else:
                stop = "stuck"
                break
        elif route == "unreachable":
            for j in out.get("unreachable") or []:
                recs[j["piece"]]["position"] = list(BUFFER)
                st["transfers"] += 1
            if not out.get("unreachable"):
                stop = "stuck_unreachable"
                break
    correct = sum(sim.true_cell(p) == c for c, p in board.placed.items())
    region = Counter()
    for c in [(c, r) for r in range(ROWS) for c in range(COLS)]:
        reg = _region(c)
        region[reg + "_n"] += 1
        region[reg + "_correct"] += board.placed.get(c) == sim.pid(*c)
    wrong = len(board.placed) - correct
    deferred = len((ledger or {}).get("deferred_cells", []))
    return {
        "policy": policy,
        "mode": mode,
        "style": style,
        "seed": seed,
        "stop": stop,
        "correct": correct,
        "wrong": wrong,
        "empty": COLS * ROWS - len(board.placed),
        "deferred": deferred,
        "rounds": rounds,
        "steps": sum(v for k, v in st.items() if k.startswith("route_")),
        "ms_per_call": 1000 * float(np.median(t_choose)) if t_choose else None,
        **dict(st),
        **dict(region),
    }


def _region(cell):
    """The procedural picture's hard regions (eval/localize_all.py's picture): the flat white
    band over the top 3 rows and the plaid patch (x 0.55-0.95, y 0.45-0.75 of the picture)."""
    c, r = cell
    if r < 3:
        return "band"
    if 0.55 * COLS - 0.5 <= c <= 0.95 * COLS - 0.5 and 0.45 * ROWS - 0.5 <= r <= 0.75 * ROWS - 0.5:
        return "plaid"
    return "textured"


def _keep(rec):
    return {k: rec[k] for k in ("piece_class", "position") if k in rec}


def summarise(rows):
    out = {"n": len(rows)}
    for k in (
        "correct",
        "wrong",
        "empty",
        "failures",
        "passes",
        "attempts",
        "shape_matches",
        "shape_right",
        "shape_forced",
        "shape_ambiguous",
        "relocalized",
        "removals",
        "steps",
        "rounds",
        "parallel_rounds",
        "transfers",
    ):
        v = [r.get(k, 0) for r in rows]
        out[k] = round(float(np.mean(v)), 1)
    out["pieces_per_failure"] = round(
        sum(r.get("passes", 0) for r in rows) / max(1, sum(r.get("failures", 0) for r in rows)), 1
    )
    for reg in ("band", "plaid", "textured"):
        out[reg + "_correct_pct"] = round(
            100
            * sum(r.get(reg + "_correct", 0) for r in rows)
            / max(1, sum(r.get(reg + "_n", 0) for r in rows)),
            1,
        )
    out["finished"] = sum(r["correct"] == COLS * ROWS for r in rows)
    out["stop"] = dict(Counter(r["stop"] for r in rows))
    out["ms_per_call"] = round(float(np.median([r["ms_per_call"] for r in rows])), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", default="/tmp/np_loc_s2.json")
    ap.add_argument("--loc-alt", default="/tmp/np_loc_s3.json")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--policies", default="plan,most_neighbours,frame_first,confidence,random")
    ap.add_argument("--modes", default="localizer,synth10,synth25")
    ap.add_argument("--styles", default="wavy,classic")
    ap.add_argument("--batch", type=int, default=BATCH_DEFAULT)
    ap.add_argument("--arms", action="store_true")
    ap.add_argument("--no-classes", action="store_true")
    ap.add_argument("--neighbour-weight", type=float, default=None)
    ap.add_argument("--frame-weight", type=float, default=None)
    ap.add_argument("--min-confidence", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    opt = {"batch": a.batch, "arms": a.arms, "classes": not a.no_classes, "max_steps": a.max_steps}
    if a.neighbour_weight is not None:
        opt["neighbour_weight"] = a.neighbour_weight
    if a.frame_weight is not None:
        opt["frame_weight"] = a.frame_weight
    if a.min_confidence is not None:
        opt["min_confidence"] = a.min_confidence
    if "localizer" in a.modes:
        opt["loc"] = json.loads(Path(a.loc).read_text())
        opt["loc_alt"] = json.loads(Path(a.loc_alt).read_text())
    results, rows_all = {}, []
    for mode in a.modes.split(","):
        for style in a.styles.split(","):
            for policy in a.policies.split(","):
                jobs = [(policy, mode, style, k, opt) for k in range(a.n)]
                with ProcessPoolExecutor(a.workers) as ex:
                    rows = list(ex.map(episode, jobs))
                rows_all += rows
                s = summarise(rows)
                results[f"{mode}/{style}/{policy}"] = s
                print(f"{mode:9s} {style:7s} {policy:15s} {json.dumps(s)}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": results, "episodes": rows_all}, indent=1))


if __name__ == "__main__":
    main()
