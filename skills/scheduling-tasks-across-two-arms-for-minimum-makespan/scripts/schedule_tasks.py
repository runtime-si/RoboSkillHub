# Generalises the hub's assigning-bricks-to-two-arms (a MILP after APEX-MR's task_assignment.py, MIT)
# from LEGO to any task list: arms are assigned and tasks timed so the makespan is minimal, with
# no-overlap on each arm, precedence, same-arm links and time-window exclusion between tasks whose
# workspace regions are closer than two YAM hand envelopes (tools/yam-spheres). The MILP is new code on
# scipy.optimize.milp (HiGHS); the greedy fallback is list scheduling.
"""Assign tasks to two arms and order them for the shortest makespan, never both arms in one region at once."""

from __future__ import annotations

import json
import logging
import math
import time
from itertools import combinations
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

MAX_EXACT = 30
"""Largest task count solved as a MILP (pairs grow as n^2; 30 tasks = 435 order binaries)."""

TIME_LIMIT = 20.0
"""MILP time limit [s]; past it the best schedule found (or the greedy one) is returned."""

HAND_ENVELOPE = 0.148
"""Radius of a YAM hand around link 6 [m]: the farthest point of tools/yam-spheres' link-6 spheres
(fingers closed) from the link origin; two tasks conflict when their regions are closer than twice
this (the hands would meet). Read from the sphere file when it is found."""

ROUTES = ("scheduled", "infeasible", "timeout")


class Output(TypedDict):
    route: str
    detail: str
    schedule: list[dict[str, Any]]
    by_arm: dict[str, list[str]]
    concurrent_steps: list[dict[str, Any]]
    makespan: float
    lower_bound: float
    method: str
    optimal: bool
    solve_s: float


def hand_envelope() -> float:
    for root in Path(__file__).resolve().parents:
        p = root / "tools" / "yam-spheres" / "yam_spheres.json"
        if p.is_file():
            ss = json.loads(p.read_text())["spheres_link6_fingers_closed"]
            return float(max(np.linalg.norm(s["center"]) + s["radius"] for s in ss))
    return HAND_ENVELOPE


def _region(r: Any) -> tuple[np.ndarray, np.ndarray] | str | None:
    """A box (lo, hi) from {'min','max'}, a point list (+ 'radius'), a Vec3, or a name string."""
    if r is None:
        return None
    if isinstance(r, str):
        return r
    if isinstance(r, dict) and "min" in r:
        return np.asarray(r["min"], float), np.asarray(r["max"], float)
    pts = r.get("points") if isinstance(r, dict) and "points" in r else r
    rad = float(r.get("radius", 0.0)) if isinstance(r, dict) else 0.0
    if isinstance(pts, dict):
        pts = [pts]
    a = np.asarray(
        [[p["x"], p["y"], p["z"]] if isinstance(p, dict) else p for p in pts], float
    ).reshape(-1, 3)
    return a.min(axis=0) - rad, a.max(axis=0) + rad


def _gap(a, b) -> float:
    if a is None or b is None:
        return math.inf
    if isinstance(a, str) or isinstance(b, str):
        return 0.0 if a == b else math.inf
    d = np.maximum(0.0, np.maximum(a[0] - b[1], b[0] - a[1]))
    return float(np.linalg.norm(d))


def _parse(tasks: list[dict[str, Any]], arms: list[int]):
    ids = [str(t["id"]) for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("task ids must be unique")
    idx = {k: i for i, k in enumerate(ids)}
    dur = np.full((len(tasks), len(arms)), np.inf)
    for i, t in enumerate(tasks):
        d = t.get("duration")
        cand = [int(a) for a in t.get("arms", arms)]
        for k, a in enumerate(arms):
            if a in cand:
                v = d.get(str(a), d.get(a)) if isinstance(d, dict) else d
                if v is not None:
                    dur[i, k] = float(v)
    prec = []
    for i, t in enumerate(tasks):
        for p in t.get("after", []) or []:
            if str(p) not in idx:
                raise ValueError(f"task {ids[i]}: unknown predecessor {p!r}")
            prec.append((idx[str(p)], i))
    same = []
    for i, t in enumerate(tasks):
        s = t.get("same_arm_as")
        if s is not None:
            if str(s) not in idx:
                raise ValueError(f"task {ids[i]}: unknown same_arm_as {s!r}")
            same.append((idx[str(s)], i))
    regions = [_region(t.get("region")) for t in tasks]
    return ids, dur, prec, same, regions


def _greedy(dur, prec, same, conflict, order=None):
    """List scheduling: tasks in topological order (longest first among ready), each on the arm that
    finishes it earliest given the arm's free time, predecessors and conflicting tasks' windows."""
    n, m = dur.shape
    preds = {i: [p for p, j in prec if j == i] for i in range(n)}
    link = {}
    for a, b in same:
        link[b] = a
    done: dict[int, tuple[int, float, float]] = {}
    free = [0.0] * m
    left = set(range(n))
    while left:
        ready = [
            i
            for i in left
            if all(p in done for p in preds[i]) and (i not in link or link[i] in done)
        ]
        if not ready:
            return None
        if order is not None:
            ready.sort(key=lambda i: order.index(i))
        else:
            ready.sort(key=lambda i: -float(np.min(dur[i])))
        i = ready[0]
        best = None
        for k in range(m):
            if not math.isfinite(dur[i, k]) or (i in link and done[link[i]][0] != k):
                continue
            s = max([free[k]] + [done[p][2] for p in preds[i]])
            moved = True
            while moved:
                moved = False
                for j, (kj, sj, ej) in done.items():
                    if kj != k and conflict[i][j] and s < ej and sj < s + dur[i, k]:
                        s, moved = ej, True
            if best is None or s + dur[i, k] < best[2]:
                best = (k, s, s + dur[i, k])
        if best is None:
            return None
        done[i] = best
        free[best[0]] = best[2]
        left.remove(i)
    return done


def _milp(dur, prec, same, conflict, time_limit):
    from scipy.optimize import Bounds, LinearConstraint, milp

    n, m = dur.shape
    big = float(np.nansum(np.where(np.isfinite(dur), dur, 0.0).max(axis=1))) + 1.0
    # Variables: x[i,k] (n*m), S[i] (n), C (1), y[p] per unordered pair (np).
    pairs = list(combinations(range(n), 2))
    nx, ns = n * m, n
    iS, iC, iy = nx, nx + ns, nx + ns + 1
    nv = iy + len(pairs)
    rows, lo, hi = [], [], []

    def row(coefs, lb, ub):
        r = np.zeros(nv)
        for k, v in coefs:
            r[k] += v
        rows.append(r)
        lo.append(lb)
        hi.append(ub)

    dfin = np.where(np.isfinite(dur), dur, 0.0)
    for i in range(n):
        row([(i * m + k, 1.0) for k in range(m)], 1.0, 1.0)
        row([(iC, 1.0), (iS + i, -1.0)] + [(i * m + k, -dfin[i, k]) for k in range(m)], 0.0, np.inf)
    for p, j in prec:
        row(
            [(iS + j, 1.0), (iS + p, -1.0)] + [(p * m + k, -dfin[p, k]) for k in range(m)],
            0.0,
            np.inf,
        )
    for a, b in same:
        for k in range(m):
            row([(a * m + k, 1.0), (b * m + k, -1.0)], 0.0, 0.0)
    for q, (i, j) in enumerate(pairs):
        y = iy + q
        # i before j:  S_j >= S_i + D_i - M (1 - y) - M * relax ; j before i: S_i >= S_j + D_j - M y - M * relax
        di = [(i * m + k, -dfin[i, k]) for k in range(m)]
        dj = [(j * m + k, -dfin[j, k]) for k in range(m)]
        if conflict[i][j]:
            row([(iS + j, 1.0), (iS + i, -1.0), (y, -big)] + di, -big, np.inf)
            row([(iS + i, 1.0), (iS + j, -1.0), (y, big)] + dj, 0.0, np.inf)
        else:
            for k in range(m):  # only when both on arm k
                row(
                    [(iS + j, 1.0), (iS + i, -1.0), (y, -big), (i * m + k, -big), (j * m + k, -big)]
                    + di,
                    -3 * big,
                    np.inf,
                )
                row(
                    [(iS + i, 1.0), (iS + j, -1.0), (y, big), (i * m + k, -big), (j * m + k, -big)]
                    + dj,
                    -2 * big,
                    np.inf,
                )
    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    ub[:nx] = np.where(np.isfinite(dur), 1.0, 0.0).reshape(-1)
    ub[iy:] = 1.0
    integrality = np.zeros(nv)
    integrality[:nx] = 1
    integrality[iy:] = 1
    c = np.zeros(nv)
    c[iC] = 1.0
    c[iS : iS + n] = 1e-4  # start early among equal makespans
    res = milp(
        c,
        constraints=LinearConstraint(np.asarray(rows), lo, hi),
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options={"time_limit": float(time_limit), "disp": False},
    )
    if res.x is None:
        return None, res
    x = res.x[:nx].reshape(n, m)
    sol = {}
    for i in range(n):
        k = int(np.argmax(x[i]))
        s = float(res.x[iS + i])
        sol[i] = (k, s, s + float(dur[i, k]))
    return sol, res


def run(
    ctx: NodeContext,
    tasks: list[dict[str, Any]],
    arms: list[int] | None = None,
    clearance: float | None = None,
    method: str = "auto",
    time_limit: float = TIME_LIMIT,
    max_exact: int = MAX_EXACT,
) -> Output:
    arms = [int(a) for a in (arms or [0, 1])]
    if method not in ("auto", "milp", "greedy"):
        raise ValueError(f"method must be 'auto', 'milp' or 'greedy', got {method!r}")
    if not tasks:
        raise ValueError("tasks is empty")
    ids, dur, prec, same, regions = _parse(tasks, arms)
    n, m = dur.shape
    sep = float(clearance) if clearance is not None else 2.0 * hand_envelope()
    conflict = [[i != j and _gap(regions[i], regions[j]) < sep for j in range(n)] for i in range(n)]
    out: Output = {
        "route": "infeasible",
        "detail": "",
        "schedule": [],
        "by_arm": {str(a): [] for a in arms},
        "concurrent_steps": [],
        "makespan": math.inf,
        "lower_bound": 0.0,
        "method": "",
        "optimal": False,
        "solve_s": 0.0,
    }
    nope = [ids[i] for i in range(n) if not np.isfinite(dur[i]).any()]
    if nope:
        out["detail"] = f"no arm can do tasks {nope}"
        return out
    best_d = np.where(np.isfinite(dur), dur, np.inf).min(axis=1)
    out["lower_bound"] = round(float(best_d.sum()) / m, 3)
    t0 = time.monotonic()
    sol, how, optimal = None, "", False
    if method in ("auto", "milp") and (method == "milp" or n <= int(max_exact)):
        sol, res = _milp(dur, prec, same, conflict, time_limit)
        how = "milp"
        optimal = sol is not None and int(getattr(res, "status", 1)) == 0
        if sol is None and int(getattr(res, "status", 2)) == 2:
            out["detail"] = (
                "the MILP is infeasible (precedence, same-arm links or reach leave no schedule)"
            )
            out["solve_s"] = round(time.monotonic() - t0, 3)
            return out
    greedy = _greedy(dur, prec, same, conflict)
    if sol is not None and greedy is not None and not optimal:
        # A MILP stopped at its time limit can be worse than list scheduling; keep the better.
        if max(e for _, _, e in greedy.values()) < max(e for _, _, e in sol.values()) - 1e-6:
            sol, how = greedy, "greedy (better than the time-limited MILP)"
    if sol is None:
        sol = greedy
        how = "greedy" if how == "" else "greedy (MILP gave no schedule in time)"
        if sol is None:
            out["detail"] = (
                "no schedule: cyclic precedence or a same-arm link to a task no shared arm can do"
            )
            out["solve_s"] = round(time.monotonic() - t0, 3)
            return out
    out["solve_s"] = round(time.monotonic() - t0, 3)
    out["method"], out["optimal"] = how, bool(optimal)
    rows = sorted(
        (
            {"id": ids[i], "arm": arms[k], "start": round(s, 3), "end": round(e, 3)}
            for i, (k, s, e) in sol.items()
        ),
        key=lambda r: (r["start"], r["arm"]),
    )
    out["schedule"] = rows
    for r in rows:
        out["by_arm"][str(r["arm"])].append(r["id"])
    out["makespan"] = round(max(r["end"] for r in rows), 3)
    # Steps for running-two-skills-concurrently-on-two-arms: each arm's next task, paired by start order.
    queues = {a: [r for r in rows if r["arm"] == a] for a in arms}
    for step in range(max(len(q) for q in queues.values())):
        out["concurrent_steps"].append(
            {f"arm_{a}": (queues[a][step]["id"] if step < len(queues[a]) else None) for a in arms}
        )
    if how.startswith("greedy (MILP"):
        out["route"] = "timeout"
        out["detail"] = f"MILP found nothing in {time_limit} s; greedy makespan {out['makespan']} s"
        return out
    out["route"] = "scheduled"
    out["detail"] = (
        f"{how}{' (optimal)' if optimal else ''}: makespan {out['makespan']} s, lower bound {out['lower_bound']} s"
    )
    return out
