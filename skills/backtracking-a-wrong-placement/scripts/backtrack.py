"""Undo a jigsaw placement that failed its check: blame the piece or a neighbour, forbid, re-assign.

Written for the hub (no code copied). The re-assignment calls SciPy's
``scipy.optimize.linear_sum_assignment`` (BSD-3-Clause, used as a library) on a cost of
``-log P(piece belongs in cell)`` from the localizer's scores. The culprit is chosen by a
small Bayesian vote over every failure recorded in the ledger: each failure says "at least
one of these placed (piece, cell) pairs is wrong, or the check itself was wrong", and each
pair's prior comes from its localization score and margin.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.optimize import linear_sum_assignment

CONF_BIAS, CONF_SCORE, CONF_MARGIN = -16.73, 16.01, 51.45
"""P(a piece belongs in a cell) = logistic(bias + a * score + b * min(margin, MARGIN_CAP)),
score and margin as localizing-a-piece-in-a-reference-image returns them (the margin taken
against the piece's best other listed cell, so negative for a cell that is not its first).
Ridge logistic regression (lambda 0.1 on standardised features) on 7864 (piece, listed
cell) pairs from two runs of that skill over every textured piece of the 22 x 25 procedural
picture (top_k 8): all 844 pairs it rates over 0.9 were right; 0.5-0.9, 89 %; 0.1-0.5,
21 %; under 0.1, 0.2 % (eval/calibrate.py)."""

MARGIN_CAP = 0.15
"""Margins above this say nothing more: right answers have margins 0.07-0.62 (5th-95th
percentile 0.07-0.57), wrong ones 0-0.13 (95th percentile 0.02) in the same runs."""

CONF_MIN, CONF_MAX = 0.02, 0.98
"""The calibrated probability is clipped to this. The synthetic pieces are cleaner than
photographed ones; no placement is ever treated as certain or impossible."""

FALSE_ALARM = 0.05
"""P(a check fails although every piece it involves is right): a piece that is right but
was set down off its seat, a seam score spoiled by glare. An assumption, not measured on
a robot; the retry ladder (max_seat_retries) absorbs most of these first."""

VERIFIED_ODDS = 0.5
"""A placement that already passed a seat check against at least one neighbour has its odds
of being wrong multiplied by this (P(wrong passes) / P(right passes)). Measured on the
procedural cuts with the benchmark's seam model: a wrong piece passed against one right
neighbour in 46 % of 1500 tries (classic cut; 11 % wavy, 21 % / 2 % against two), a right
one always."""

SEAM_FAIL = 0.06
"""A seam score (mean outline gap over side length) above this clears nothing: the seam is
bad. matching-a-piece-to-placed-neighbours' max_fit."""

MAX_CULPRITS = 3
"""Culprit sets of up to this many wrong placements are enumerated."""

MAX_VARIABLES = 14
"""At most this many (piece, cell) pairs enter the vote (the most recently involved);
2^14 subsets would still be fast, but sets above MAX_CULPRITS are skipped anyway."""

MIN_ASSIGN_PROB = 0.01
"""A piece is only assigned to a cell it belongs in with at least this probability; below,
it stays loose (cost -log 0.01 = 4.6 for leaving it out). A flat piece with a 66-cell
colour shortlist gets 1/66 = 0.015."""

ATTEMPT_PROB_FACTOR = 0.5
"""Each failure of a pair that nothing explained (it was seated again, not blamed and no
neighbour was) halves the pair's probability in the re-solve."""

FORBID_P_WRONG = 0.8
"""A blamed pair is forbidden for good (listed in the ledger's ``forbidden``) when the vote
puts it at least this likely wrong; a less certain one is only left out of this call's
re-solve and comes back later at its posterior probability, so one false alarm cannot
ban the right piece from its cell."""

BLAME_RATIO = 0.5
"""A suspect at least this fraction as likely to be wrong as the likeliest one is blamed
instead when blaming the likeliest would leave a cell with no known piece to fill it: the
localizer listing no other piece for a cell is itself evidence for the piece there."""

FLAT_PROB_FLOOR = 0.015
"""A low-texture piece's shortlist cells (score None) share its probability equally, at
least this each."""


class Output(TypedDict, total=False):
    route: str
    culprit_piece: str
    culprit_cell: list[int]
    culprit_p_wrong: float
    p_false_alarm: float
    suspects: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    n_actions: int
    assignment: list[dict[str, Any]]
    forbidden: list[dict[str, Any]]
    ledger: dict[str, Any]
    ledger_path: str
    detail: str


# --- ledger -------------------------------------------------------------------------------


def _key(piece: str, col: int, row: int) -> str:
    return f"{piece}@{col},{row}"


def _ckey(col: int, row: int) -> str:
    return f"{col},{row}"


def new_ledger(cols: int, rows: int) -> dict[str, Any]:
    return {
        "version": 1,
        "grid": [cols, rows],
        "forbidden": [],
        "attempts": {},
        "unexplained": {},
        "p_right": {},
        "events": [],
        "neighbour_removals": {},
        "deferred_cells": [],
        "backtracks": 0,
    }


def load_ledger(path: str | None, ledger: dict | None, cols: int, rows: int) -> dict[str, Any]:
    if ledger is not None:
        out = json.loads(json.dumps(ledger))
    elif path and Path(path).exists():
        out = json.loads(Path(path).read_text())
    else:
        return new_ledger(cols, rows)
    if out.get("version") != 1:
        raise ValueError(f"unknown ledger version {out.get('version')!r}")
    if list(out.get("grid", [cols, rows])) != [cols, rows]:
        raise ValueError(f"ledger grid {out['grid']} is not {cols} x {rows}")
    base = new_ledger(cols, rows)
    base.update(out)
    return base


def save_ledger(path: str, ledger: dict[str, Any]) -> None:
    """Write atomically (temp file then rename) so a crash never leaves half a ledger."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name, dir=p.parent)
    with os.fdopen(fd, "w") as f:
        json.dump(ledger, f, indent=1)
    os.replace(tmp, p)


def is_forbidden(ledger: dict, piece: str, col: int, row: int) -> bool:
    return any(
        (f["piece"], f["col"], f["row"]) == (piece, col, row) for f in ledger.get("forbidden", [])
    )


# --- probabilities from the localizer ------------------------------------------------------


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def calibrated(score: float, margin: float) -> float:
    z = CONF_BIAS + CONF_SCORE * score + CONF_MARGIN * min(margin, MARGIN_CAP)
    return min(CONF_MAX, max(CONF_MIN, _logistic(z)))


def cell_probs(record: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """P(piece belongs in cell) and its angle for every cell the localizer listed."""
    cands = [c for c in record.get("candidates") or [] if "col" in c and "row" in c]
    out: dict[tuple[int, int], dict[str, Any]] = {}
    scored = [c for c in cands if c.get("score") is not None]
    flat = [c for c in cands if c.get("score") is None]
    for k, c in enumerate(scored):
        others = [o["score"] for j, o in enumerate(scored) if j != k]
        margin = c["score"] - max(others) if others else MARGIN_CAP
        cell = (int(c["col"]), int(c["row"]))
        p = calibrated(float(c["score"]), margin)
        if cell not in out or p > out[cell]["p"]:
            out[cell] = {"p": p, "angle_deg": c.get("angle_deg"), "score": c["score"]}
    for c in flat:
        cell = (int(c["col"]), int(c["row"]))
        p = max(FLAT_PROB_FLOOR, min(CONF_MAX, 1.0 / len(flat)))
        out.setdefault(cell, {"p": p, "angle_deg": None, "score": None})
    return out


def placement_conf(pl: dict[str, Any], probs: dict | None) -> float:
    """How sure we are that a placed piece is right where it is."""
    if pl.get("confidence") is not None:
        p = float(pl["confidence"])
    elif probs is not None and (pl["col"], pl["row"]) in probs:
        p = probs[pl["col"], pl["row"]]["p"]
    elif pl.get("score") is not None and pl.get("margin") is not None:
        p = calibrated(float(pl["score"]), float(pl["margin"]))
    elif probs is not None:
        p = CONF_MIN
    else:
        p = 0.5
    p = min(CONF_MAX, max(CONF_MIN, p))
    if pl.get("verified"):
        odds = (1 - p) / p * VERIFIED_ODDS
        p = 1 / (1 + odds)
    return p


# --- the culprit vote ----------------------------------------------------------------------


_DIRS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
"""Picture grid: row 0 at the top (north), as localizing-a-piece-in-a-reference-image counts."""


def culprit_posterior(
    variables: dict[str, float], events: list[list[str]], false_alarm: float
) -> tuple[dict[str, float], list[float]]:
    """Marginal P(pair is wrong) for every variable and, per event, P(none of its pairs is
    wrong). Prior odds per pair; each event is 1 if a wrong pair explains it, else
    false_alarm. Culprit sets of up to MAX_CULPRITS pairs."""
    names = list(variables)
    odds = {n: (1 - variables[n]) / variables[n] for n in names}
    ev = [set(e) for e in events]
    total = 0.0
    marg = dict.fromkeys(names, 0.0)
    none = [0.0] * len(ev)
    for k in range(0, min(MAX_CULPRITS, len(names)) + 1):
        for subset in itertools.combinations(names, k):
            s = set(subset)
            w = math.prod(odds[n] for n in subset) if subset else 1.0
            hit = [bool(s & e) for e in ev]
            w *= false_alarm ** sum(not h for h in hit)
            total += w
            for n in subset:
                marg[n] += w
            for i, h in enumerate(hit):
                if not h:
                    none[i] += w
    return {n: marg[n] / total for n in names}, [x / total for x in none]


def _component(events: list[dict], new_ids: set[str]) -> list[dict]:
    """The recorded events linked to the new one through shared pairs."""
    keep, frontier = [], set(new_ids)
    rest = list(events)
    changed = True
    while changed:
        changed = False
        for e in list(rest):
            if frontier & set(e["involved"]):
                keep.append(e)
                frontier |= set(e["involved"])
                rest.remove(e)
                changed = True
    return keep


# --- the re-solve ---------------------------------------------------------------------------


def reassign(
    pool: list[str],
    probs: dict[str, dict],
    free: list[tuple[int, int]],
    ledger: dict,
    min_prob: float = MIN_ASSIGN_PROB,
    exclude: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Assign pool pieces to free cells, maximising the product of their probabilities
    (linear_sum_assignment on -log p), never to a forbidden or excluded pair; a piece whose
    best is under min_prob stays loose (a dummy column per piece). A pair the culprit vote
    has seen uses its posterior P(right) from the ledger instead of the localizer's."""
    if not pool or not free:
        return []
    banned = {_key(f["piece"], f["col"], f["row"]) for f in ledger["forbidden"]}
    banned |= exclude or set()
    col_of = {c: j for j, c in enumerate(free)}
    big = 1e6
    cost = np.full((len(pool), len(free) + len(pool)), big)
    leave = -math.log(min_prob)
    for i, piece in enumerate(pool):
        cost[i, len(free) + i] = leave
        for cell, info in probs.get(piece, {}).items():
            j = col_of.get(cell)
            pair = _key(piece, *cell)
            if j is None or pair in banned:
                continue
            n = int(ledger["unexplained"].get(pair, 0))
            p = float(ledger["p_right"].get(pair, info["p"])) * ATTEMPT_PROB_FACTOR**n
            if p >= min_prob:
                cost[i, j] = -math.log(p)
    rows, cols = linear_sum_assignment(cost)
    out = []
    for i, j in zip(rows, cols, strict=True):
        if j < len(free) and cost[i, j] < big:
            cell = free[j]
            info = probs[pool[i]][cell]
            out.append(
                {
                    "piece": pool[i],
                    "col": cell[0],
                    "row": cell[1],
                    "prob": round(math.exp(-float(cost[i, j])), 4),
                    "cost": round(float(cost[i, j]), 4),
                    "angle_deg": info["angle_deg"],
                }
            )
    return out


# --- the node -------------------------------------------------------------------------------


def run(
    ctx: NodeContext,
    failure: dict,
    placements: list[dict],
    pieces: list[dict],
    grid_cols: int,
    grid_rows: int,
    ledger_path: str | None = None,
    ledger: dict | None = None,
    free_cells: list[list[int]] | None = None,
    max_seat_retries: int = 1,
    max_pieces_per_cell: int = 3,
    max_neighbour_removals: int = 2,
    max_backtracks: int = 40,
    false_alarm: float = FALSE_ALARM,
    seam_fail: float = SEAM_FAIL,
    min_assign_prob: float = MIN_ASSIGN_PROB,
) -> Output:
    """Decide what to undo after a placement failed its check, and what to place instead.

    ``failure``: ``{"piece", "col", "row", "kind", "seam_scores"?, "event_id"?}`` with kind
    ``not_seated`` (the piece did not go down; every placed neighbour is a suspect),
    ``seam_mismatch`` (a seam check failed; ``seam_scores`` maps a neighbour's piece id or
    its direction north / south / east / west to its seam score, and only neighbours over
    ``seam_fail`` are suspects) or ``recheck_failed`` (re-localizing the placed piece put it
    elsewhere; only the piece is a suspect). ``placements``: the board as believed,
    ``[{"piece", "col", "row", "score"?, "margin"?, "confidence"?, "verified"?, "locked"?}]``.
    ``pieces``: localizer output per known piece (placed or loose),
    ``[{"piece", "candidates": [{"col", "row", "score", "angle_deg"}, ...]}]``. The ledger
    (``ledger`` dict or ``ledger_path`` JSON) carries forbidden pairs, attempts and past
    failures between calls; it is returned and, with ``ledger_path``, saved."""
    cols, rows = int(grid_cols), int(grid_rows)
    if cols <= 0 or rows <= 0:
        raise ValueError("grid_cols and grid_rows must be positive")
    book = load_ledger(ledger_path, ledger, cols, rows)
    p_id = str(failure["piece"])
    c = (int(failure["col"]), int(failure["row"]))
    kind = str(failure.get("kind", "not_seated"))
    if kind not in ("not_seated", "seam_mismatch", "recheck_failed"):
        raise ValueError(f"unknown failure kind {kind!r}")
    if not (0 <= c[0] < cols and 0 <= c[1] < rows):
        raise ValueError(f"failure cell {list(c)} is outside the {cols} x {rows} grid")
    event_id = failure.get("event_id")
    if event_id is not None:
        for e in book["events"]:
            if e.get("id") == event_id:
                return _finish(
                    dict(
                        e["decision"], detail="replayed from the ledger: " + e["decision"]["detail"]
                    ),
                    book,
                    None,
                )

    probs = {str(r["piece"]): cell_probs(r) for r in pieces or []}
    for pl in placements or []:
        # A placement the localizer did not list (placed by a shape match) keeps its own
        # confidence as the pair's probability in the re-solve.
        cell = (int(pl["col"]), int(pl["row"]))
        if pl.get("confidence") is not None and cell not in probs.get(str(pl["piece"]), {}):
            probs.setdefault(str(pl["piece"]), {})[cell] = {
                "p": min(CONF_MAX, max(CONF_MIN, float(pl["confidence"]))),
                "angle_deg": pl.get("angle_deg"),
                "score": None,
            }
    board: dict[tuple[int, int], dict] = {}
    for pl in placements or []:
        cell = (int(pl["col"]), int(pl["row"]))
        if str(pl["piece"]) == p_id:
            continue
        board[cell] = dict(pl, piece=str(pl["piece"]), col=cell[0], row=cell[1])
    me = next((pl for pl in placements or [] if str(pl["piece"]) == p_id), {})
    me = dict(me, piece=p_id, col=c[0], row=c[1])

    # Who the failure involves.
    neighbours = {}
    for d, (dc, dr) in _DIRS.items():
        cell = (c[0] + dc, c[1] + dr)
        if cell in board:
            neighbours[d] = board[cell]
    seams = {str(k): float(v) for k, v in (failure.get("seam_scores") or {}).items()}
    suspects = [me]
    if kind != "recheck_failed":
        for d, nb in neighbours.items():
            if nb.get("locked"):
                continue
            s = seams.get(nb["piece"], seams.get(d))
            if kind == "seam_mismatch" and (s is None or s <= seam_fail):
                continue
            if kind == "not_seated" and s is not None and s <= seam_fail:
                continue
            suspects.append(nb)
    involved = [_key(x["piece"], x["col"], x["row"]) for x in suspects]

    # Record the attempt.
    my_key = _key(p_id, *c)
    book["attempts"][my_key] = int(book["attempts"].get(my_key, 0)) + 1
    attempts = book["attempts"][my_key]
    event = {
        "id": event_id,
        "piece": p_id,
        "col": c[0],
        "row": c[1],
        "kind": kind,
        "involved": involved,
        "seam_scores": seams,
    }

    # The vote over this failure and every recorded one linked to it.
    linked = _component(book["events"], set(involved))
    pairs: dict[str, None] = {}
    for e in linked + [event]:
        for k in e["involved"]:
            pairs.setdefault(k, None)
    on_board = {_key(x["piece"], x["col"], x["row"]): x for x in [me, *board.values()]}
    order = list(pairs)[-MAX_VARIABLES:]
    for k in involved:
        if k not in order:
            order.append(k)
    variables = {}
    for k in order:
        piece, cell_s = k.rsplit("@", 1)
        pc = tuple(int(v) for v in cell_s.split(","))
        pl = on_board.get(k) or {"piece": piece, "col": pc[0], "row": pc[1]}
        variables[k] = placement_conf(pl, probs.get(piece))
    evs = [[k for k in e["involved"] if k in variables] for e in linked + [event]]
    marg, none = culprit_posterior(variables, evs, false_alarm)
    p_fa = none[-1]
    for k, v in marg.items():
        book["p_right"][k] = round(1.0 - v, 4)
    ranked = sorted(
        (
            {
                "piece": x["piece"],
                "col": x["col"],
                "row": x["row"],
                "p_wrong": round(marg[k], 4),
                "prior_conf": round(variables[k], 4),
            }
            for k, x in zip(involved, suspects, strict=True)
        ),
        key=lambda s: -s["p_wrong"],
    )

    def finish(route: str, detail: str, actions: list, assignment: list, culprit=None) -> Output:
        decision = {
            "route": route,
            "culprit_piece": culprit["piece"] if culprit else "",
            "culprit_cell": [culprit["col"], culprit["row"]] if culprit else [],
            "culprit_p_wrong": culprit["p_wrong"] if culprit else 0.0,
            "p_false_alarm": round(p_fa, 4),
            "suspects": ranked,
            "actions": actions,
            "n_actions": len(actions),
            "assignment": assignment,
            "detail": detail,
        }
        event["decision"] = decision
        book["events"].append(event)
        return _finish(decision, book, ledger_path)

    failed_here = sorted({e["piece"] for e in book["events"] if (e["col"], e["row"]) == c} | {p_id})
    nb_removals = int(book["neighbour_removals"].get(_ckey(*c), 0))
    lim = {
        "max_seat_retries": max_seat_retries,
        "max_pieces_per_cell": max_pieces_per_cell,
        "max_neighbour_removals": max_neighbour_removals,
        "max_backtracks": max_backtracks,
    }

    # Rung 1: the same piece again, if the check is the likeliest thing that was wrong.
    best = ranked[0]
    bad_seam = any(s > seam_fail for s in seams.values())
    if (
        kind == "not_seated"
        and not bad_seam
        and attempts <= max_seat_retries
        and p_fa >= best["p_wrong"]
    ):
        book["unexplained"][my_key] = int(book["unexplained"].get(my_key, 0)) + 1
        act = [
            {
                "op": "reseat",
                "piece": p_id,
                "col": c[0],
                "row": c[1],
                "angle_deg": _angle(probs, p_id, c),
                "attempt": attempts + 1,
                "max_attempts": max_seat_retries + 1,
            }
        ]
        return finish(
            "retry_seat",
            f"P(check was wrong) {p_fa:.2f} >= P({best['piece']} wrong) "
            f"{best['p_wrong']:.2f}: seat {p_id} again (attempt {attempts + 1} of "
            f"{max_seat_retries + 1})",
            act,
            [],
        )

    if book["backtracks"] >= max_backtracks:
        return _give_up(
            book,
            finish,
            p_id,
            c,
            f"{book['backtracks']} backtracks already (max_backtracks {max_backtracks})",
            lim,
        )
    # Rung 2/3: the likeliest wrong placement among those still on the board; among
    # suspects nearly as likely, one whose removal leaves a known piece for every cell
    # it empties.
    deferred = {tuple(x) for x in book["deferred_cells"]}
    allowed = []
    for s in ranked:
        if s["piece"] != p_id:
            nb = board.get((s["col"], s["row"]))
            if nb is None or nb.get("locked") or nb_removals >= max_neighbour_removals:
                continue
        allowed.append(s)
    culprit = allowed[0]
    for s in allowed:
        if s["p_wrong"] < BLAME_RATIO * allowed[0]["p_wrong"]:
            break
        trial = _plan(
            s,
            p_id,
            c,
            board,
            book,
            probs,
            pieces,
            free_cells,
            deferred,
            cols,
            rows,
            min_assign_prob,
            dry_run=True,
        )
        if all(cell in trial[1] for cell in trial[0]):
            culprit = s
            break
    blame_me = culprit["piece"] == p_id
    if blame_me and len(failed_here) > max_pieces_per_cell:
        return _give_up(
            book,
            finish,
            p_id,
            c,
            f"{len(failed_here)} pieces failed cell "
            f"{list(c)} (max_pieces_per_cell {max_pieces_per_cell}) and "
            f"{nb_removals} neighbour removals",
            lim,
        )

    book["backtracks"] += 1
    if not blame_me:
        book["neighbour_removals"][_ckey(*c)] = nb_removals + 1
    fixing, by_cell, by_piece, assignment, removed = _plan(
        culprit,
        p_id,
        c,
        board,
        book,
        probs,
        pieces,
        free_cells,
        deferred,
        cols,
        rows,
        min_assign_prob,
        dry_run=False,
        reason=kind,
        event_id=event_id,
    )

    actions = [{"op": "remove", "piece": x, "col": cell[0], "row": cell[1]} for x, cell in removed]
    placed = set()
    for cell in fixing:
        a = by_cell.get(cell)
        if a is None:
            continue
        n = int(book["attempts"].get(_key(a["piece"], *cell), 0))
        actions.append(
            {
                "op": "place",
                "piece": a["piece"],
                "col": cell[0],
                "row": cell[1],
                "angle_deg": a["angle_deg"],
                "prob": a["prob"],
                "attempt": n + 1,
                "max_attempts": max_seat_retries + 1,
            }
        )
        placed.add(a["piece"])
    for x, _ in removed:
        if x in placed:
            continue
        a = by_piece.get(x)
        if a is not None:
            actions.append(
                {
                    "op": "set_aside",
                    "piece": x,
                    "planned_cell": [a["col"], a["row"]],
                    "prob": a["prob"],
                }
            )
        else:
            excl = {(f["col"], f["row"]) for f in book["forbidden"] if f["piece"] == x}
            if x == culprit["piece"]:
                excl.add((culprit["col"], culprit["row"]))
            actions.append(
                {"op": "relocalize", "piece": x, "exclude_cells": [list(e) for e in sorted(excl)]}
            )
    empty = [list(cell) for cell in fixing if cell not in by_cell]
    for cell in empty:
        excl = {f["piece"] for f in book["forbidden"] if (f["col"], f["row"]) == tuple(cell)}
        excl |= {
            e["piece"] for e in book["events"] + [event] if (e["col"], e["row"]) == tuple(cell)
        }
        actions.append(
            {
                "op": "search_for_cell",
                "col": cell[0],
                "row": cell[1],
                "exclude_pieces": sorted(excl),
            }
        )
    who = f"{culprit['piece']} at {[culprit['col'], culprit['row']]}"
    why = (
        f"P({culprit['piece']} wrong) {culprit['p_wrong']:.2f} over "
        + ", ".join(f"{s['piece']} {s['p_wrong']:.2f}" for s in ranked if s is not culprit)
        + f"; P(check wrong) {p_fa:.2f}"
    )
    if empty:
        route = "no_alternative"
        detail = f"blamed {who} ({why}); no admissible piece for {empty}"
    elif blame_me:
        route = "reassigned"
        detail = f"blamed {who} ({why}); {by_cell[c]['piece']} goes to {list(c)}"
    else:
        route = "remove_neighbour"
        detail = f"blamed neighbour {who} ({why}); {by_cell[fixing[0]]['piece']} replaces it"
    return finish(route, detail, actions, assignment, culprit)


def _plan(
    culprit,
    p_id,
    c,
    board,
    book,
    probs,
    pieces,
    free_cells,
    deferred,
    cols,
    rows,
    min_prob,
    dry_run,
    reason="",
    event_id=None,
):
    """Lift the failed piece (and the culprit), re-solve without the culprit's pair, and
    forbid that pair for good if the vote is sure enough. Returns (cells to fill,
    assignment by cell, by piece, assignment, removed pairs). A dry run changes neither the
    ledger nor the board."""
    if dry_run:
        board = dict(board)
    elif culprit["p_wrong"] >= FORBID_P_WRONG:
        book["forbidden"].append(
            {
                "piece": culprit["piece"],
                "col": culprit["col"],
                "row": culprit["row"],
                "reason": reason,
                "event": event_id,
                "p_wrong": culprit["p_wrong"],
            }
        )
    exclude = {_key(culprit["piece"], culprit["col"], culprit["row"])}
    removed, fixing = [(p_id, c)], [c]
    if culprit["piece"] != p_id:
        d = (culprit["col"], culprit["row"])
        removed.append((culprit["piece"], d))
        fixing.insert(0, d)
        board.pop(d)
    if free_cells is not None:
        free = [tuple(int(v) for v in x) for x in free_cells]
        free = [x for x in free if x not in board and x not in deferred]
        free += [x for x in fixing if x not in free]
    else:
        free = [
            (i, j)
            for j in range(rows)
            for i in range(cols)
            if (i, j) not in board and (i, j) not in deferred
        ]
    on_board_ids = {x["piece"] for x in board.values()}
    pool = sorted({str(r["piece"]) for r in pieces or []} - on_board_ids | {x for x, _ in removed})
    assignment = reassign(pool, probs, free, book, min_prob, exclude)
    by_cell = {(a["col"], a["row"]): a for a in assignment}
    by_piece = {a["piece"]: a for a in assignment}
    return fixing, by_cell, by_piece, assignment, removed


def _angle(probs, piece, cell):
    info = probs.get(piece, {}).get(tuple(cell))
    return info["angle_deg"] if info else None


def _give_up(book, finish, p_id, c, why, lim):
    book["forbidden"].append(
        {"piece": p_id, "col": c[0], "row": c[1], "reason": "give_up", "event": None}
    )
    if list(c) not in book["deferred_cells"]:
        book["deferred_cells"].append(list(c))
    actions = [
        {"op": "remove", "piece": p_id, "col": c[0], "row": c[1]},
        {"op": "defer_cell", "col": c[0], "row": c[1]},
        {"op": "set_aside", "piece": p_id, "planned_cell": None, "prob": None},
    ]
    return finish("give_up", f"giving up on cell {list(c)}: {why}; limits {lim}", actions, [])


def _finish(decision: dict, book: dict, path: str | None) -> Output:
    if path:
        save_ledger(path, book)
    out: Output = dict(decision)  # type: ignore[assignment]
    out["forbidden"] = list(book["forbidden"])
    out["ledger"] = book
    if path:
        out["ledger_path"] = str(path)
    return out
