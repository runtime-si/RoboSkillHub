"""Order pick-and-place pairs (per arm) for the least travel time: Held-Karp DP for small sets, 2-opt / Or-opt above."""

from __future__ import annotations

import itertools
import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

SPEED = 0.25
"""Cartesian hand speed of the travel estimate [m/s] (a YAM servo's typical transfer speed)."""

LIFT = 0.10
"""Each move rises to this clearance over the start and descends from it over the goal [m]; its vertical
part is added to the horizontal distance (a lift-move-lower path)."""

EXACT_MAX = 12
"""Largest task count ordered exactly (Held-Karp: 2^n * n^2 states, 0.6 M at 12)."""

ROUTES = ("ordered", "infeasible")


class Output(TypedDict):
    route: str
    detail: str
    order: dict[str, list[str]]
    travel_s: dict[str, float]
    given_travel_s: dict[str, float]
    total_travel_s: float
    total_given_s: float
    method: str
    solve_s: float


def _p(v: Any) -> np.ndarray:
    v = v.get("position", v) if isinstance(v, dict) else v
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v.get("z", 0.0))])
    return np.asarray(v, dtype=float).reshape(-1)[:3]


def travel_time(a: np.ndarray, b: np.ndarray, speed: float = SPEED, lift: float = LIFT) -> float:
    """Seconds for a lift-move-lower path from a to b at speed."""
    top = max(a[2], b[2]) + lift
    return (float(np.linalg.norm(a[:2] - b[:2])) + (top - a[2]) + (top - b[2])) / speed


def _cost(seq: list[int], start: np.ndarray, picks, places, speed, lift) -> float:
    cur, total = start, 0.0
    for i in seq:
        total += travel_time(cur, picks[i], speed, lift) + travel_time(
            picks[i], places[i], speed, lift
        )
        cur = places[i]
    return total


def _feasible(seq: list[int], preds: dict[int, set[int]]) -> bool:
    seen: set[int] = set()
    for i in seq:
        if not preds.get(i, set()) <= seen:
            return False
        seen.add(i)
    return True


def _held_karp(idx, start, picks, places, preds, speed, lift):
    n = len(idx)
    loc = {k: i for i, k in enumerate(idx)}
    pmask = [sum(1 << loc[p] for p in preds.get(k, set()) if p in loc) for k in idx]
    inner = [travel_time(picks[k], places[k], speed, lift) for k in idx]
    hop = [[travel_time(places[a], picks[b], speed, lift) for b in idx] for a in idx]
    best: dict[tuple[int, int], tuple[float, int]] = {}
    for j in range(n):
        if pmask[j] == 0:
            best[(1 << j, j)] = (travel_time(start, picks[idx[j]], speed, lift) + inner[j], -1)
    for mask in range(1, 1 << n):
        for j in range(n):
            if (mask, j) not in best:
                continue
            c, _ = best[(mask, j)]
            for k in range(n):
                if mask & (1 << k) or (pmask[k] & mask) != pmask[k]:
                    continue
                nm, nc = mask | (1 << k), c + hop[j][k] + inner[k]
                if (nm, k) not in best or nc < best[(nm, k)][0]:
                    best[(nm, k)] = (nc, j)
    full = (1 << n) - 1
    ends = [(best[(full, j)][0], j) for j in range(n) if (full, j) in best]
    if not ends:
        return None
    _, j = min(ends)
    seq, mask = [], full
    while j != -1:
        seq.append(idx[j])
        _, pj = best[(mask, j)]
        mask &= ~(1 << j)
        j = pj
    return seq[::-1]


def _nearest(idx, start, picks, places, preds, speed, lift):
    left, seq, cur = list(idx), [], start
    while left:
        ok = [k for k in left if preds.get(k, set()) & set(idx) <= set(seq)]
        if not ok:
            return None
        k = min(ok, key=lambda k: travel_time(cur, picks[k], speed, lift))
        seq.append(k)
        left.remove(k)
        cur = places[k]
    return seq


def _improve(seq, start, picks, places, preds, speed, lift, budget_s=2.0):
    """2-opt (segment reversal) and Or-opt (move a run of 1-3) until no gain or the budget runs out."""
    best, bc = list(seq), _cost(seq, start, picks, places, speed, lift)
    t_end = time.monotonic() + budget_s
    improved = True
    while improved and time.monotonic() < t_end:
        improved = False
        n = len(best)
        for i, j in itertools.combinations(range(n), 2):
            cand = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
            if _feasible(cand, preds):
                c = _cost(cand, start, picks, places, speed, lift)
                if c < bc - 1e-9:
                    best, bc, improved = cand, c, True
        for length in (1, 2, 3):
            for i in range(n - length + 1):
                seg, rest = best[i : i + length], best[:i] + best[i + length :]
                for k in range(len(rest) + 1):
                    cand = rest[:k] + seg + rest[k:]
                    if cand != best and _feasible(cand, preds):
                        c = _cost(cand, start, picks, places, speed, lift)
                        if c < bc - 1e-9:
                            best, bc, improved = cand, c, True
    return best


def run(
    ctx: NodeContext,
    tasks: list[dict[str, Any]],
    by_arm: dict[str, list[str]] | None = None,
    start: dict[str, Any] | None = None,
    speed: float = SPEED,
    lift: float = LIFT,
    method: str = "auto",
    exact_max: int = EXACT_MAX,
) -> Output:
    """tasks: [{id, pick, place, after?: [ids]}]; by_arm: #251's {arm: [ids]} (else one queue "0" in the given
    order); start: {arm: position} where each arm's hand begins (else its first pick)."""
    if method not in ("auto", "exact", "heuristic"):
        raise ValueError(f"method must be 'auto', 'exact' or 'heuristic', got {method!r}")
    if not tasks:
        raise ValueError("tasks is empty")
    if speed <= 0:
        raise ValueError("speed must be positive")
    ids = [str(t["id"]) for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("task ids must be unique")
    picks = {str(t["id"]): _p(t["pick"]) for t in tasks}
    places = {str(t["id"]): _p(t["place"]) for t in tasks}
    preds = {str(t["id"]): {str(p) for p in t.get("after", []) or []} for t in tasks}
    for k, ps in preds.items():
        if not ps <= set(ids):
            raise ValueError(f"task {k}: unknown predecessor(s) {sorted(ps - set(ids))}")
    queues = {str(a): [str(i) for i in q] for a, q in (by_arm or {"0": ids}).items()}
    out: Output = {
        "route": "ordered",
        "detail": "",
        "order": {},
        "travel_s": {},
        "given_travel_s": {},
        "total_travel_s": 0.0,
        "total_given_s": 0.0,
        "method": "",
        "solve_s": 0.0,
    }
    t0 = time.monotonic()
    methods = set()
    for arm, q in queues.items():
        if not q:
            out["order"][arm], out["travel_s"][arm], out["given_travel_s"][arm] = [], 0.0, 0.0
            continue
        s0 = _p((start or {}).get(arm)) if start and arm in start else picks[q[0]]
        # Within a queue only precedence among its own tasks orders it; cross-arm precedence is the
        # scheduler's (#251) job.
        local = {k: preds[k] & set(q) for k in q}
        given = _cost(q, s0, picks, places, speed, lift) if _feasible(q, local) else math.inf
        seq = None
        if method in ("auto", "exact") and (method == "exact" or len(q) <= int(exact_max)):
            seq, how = _held_karp(q, s0, picks, places, local, speed, lift), "held-karp"
        if seq is None:
            nn = _nearest(q, s0, picks, places, local, speed, lift)
            if nn is None:
                out["route"], out["detail"] = (
                    "infeasible",
                    f"arm {arm}: cyclic precedence among {q}",
                )
                return out
            seq, how = _improve(nn, s0, picks, places, local, speed, lift), "nearest + 2-opt/or-opt"
        methods.add(how)
        out["order"][arm] = seq
        out["travel_s"][arm] = round(_cost(seq, s0, picks, places, speed, lift), 3)
        out["given_travel_s"][arm] = round(given, 3) if math.isfinite(given) else -1.0
    out["solve_s"] = round(time.monotonic() - t0, 3)
    out["method"] = ", ".join(sorted(methods))
    out["total_travel_s"] = round(sum(out["travel_s"].values()), 3)
    finite = [v for v in out["given_travel_s"].values() if v >= 0]
    out["total_given_s"] = round(sum(finite), 3)
    out["detail"] = (
        f"travel {out['total_travel_s']} s against {out['total_given_s']} s in the given order ({out['method']})"
    )
    return out
