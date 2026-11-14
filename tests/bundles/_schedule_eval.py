"""Benchmarks for scheduling-tasks-across-two-arms-for-minimum-makespan (CPU).

python -m tests.bundles._schedule_eval
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
from gap.testing import FakeContext

p = (
    Path(__file__).resolve().parents[2]
    / "skills/scheduling-tasks-across-two-arms-for-minimum-makespan/scripts/schedule_tasks.py"
)
spec = importlib.util.spec_from_file_location("schedule_tasks", p)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
BASES = {0: np.array([-0.3475, 0.305, 0.0]), 1: np.array([-0.3475, -0.305, 0.0])}
REACH = 0.70


def dur(arm, *pts):
    """Seconds: 3 s of grasp / release plus 6 s per metre from the arm's base to the far point."""
    d = max(np.linalg.norm(np.asarray(q) - BASES[arm]) for q in pts)
    return None if d > REACH else round(3.0 + 6.0 * d, 2)


def pens(n=12, gap=0.4):
    a, b = np.array([0.0, gap / 2, 0.05]), np.array([0.0, -gap / 2, 0.05])
    tasks = []
    for k in range(n):
        pa, pb = a + [0.02 * (k % 4), 0.0, 0.0], b + [0.02 * (k % 4), 0.0, 0.0]
        da = {str(r): v for r in (0, 1) if (v := dur(r, pa)) is not None}
        db = {str(r): v for r in (0, 1) if (v := dur(r, pb)) is not None}
        tasks.append(
            {
                "id": f"pick{k}",
                "duration": da,
                "arms": [int(r) for r in da],
                "region": {"points": [list(pa)], "radius": 0.02},
            }
        )
        tasks.append(
            {
                "id": f"place{k}",
                "duration": db,
                "arms": [int(r) for r in db],
                "region": {"points": [list(pb)], "radius": 0.02},
                "after": [f"pick{k}"],
                "same_arm_as": f"pick{k}",
            }
        )
    return tasks


def sorting(n=10, seed=0):
    rng = np.random.default_rng(seed)
    bins = {"L": np.array([0.05, 0.38, 0.05]), "R": np.array([0.05, -0.38, 0.05])}
    tasks = []
    for k in range(n):
        while True:
            o = np.array([rng.uniform(-0.15, 0.2), rng.uniform(-0.3, 0.3), 0.03])
            b = bins["L" if rng.random() < 0.5 else "R"]
            d = {str(r): v for r in (0, 1) if (v := dur(r, o, b)) is not None}
            if d:
                break
        tasks.append(
            {
                "id": f"obj{k}",
                "duration": d,
                "arms": [int(r) for r in d],
                "region": {"points": [list(o), list(b)], "radius": 0.02},
            }
        )
    return tasks


def baselines(tasks):
    ids, dur_, prec, same, regions = S._parse(tasks, [0, 1])
    sep = 2 * S.hand_envelope()
    n = len(ids)
    conflict = [
        [i != j and S._gap(regions[i], regions[j]) < sep for j in range(n)] for i in range(n)
    ]
    one = min(
        (float(dur_[:, k].sum()) for k in range(2) if np.isfinite(dur_[:, k]).all()),
        default=math.inf,
    )
    # Naive alternation: pick-like tasks (no same_arm link) go to arms in turn when reachable; linked tasks follow.
    fixed = np.full_like(dur_, np.inf)
    turn = 0
    arm_of = {}
    for i in range(n):
        link = next((a for a, b in same if b == i), None)
        if link is not None:
            k = arm_of[link]
        else:
            k = turn % 2 if np.isfinite(dur_[i, turn % 2]) else 1 - turn % 2
            turn += 1
        arm_of[i] = k
        fixed[i, k] = dur_[i, k]
    sol = S._greedy(fixed, prec, same, conflict, order=list(range(n)))
    alt = max(e for _, _, e in sol.values()) if sol else math.inf
    return one, alt


for name, tasks in (
    ("pens 12 (holders 0.4 m apart)", pens()),
    ("pens 12 (holders 0.2 m apart)", pens(gap=0.2)),
    *[(f"sorting 10, seed {s}", sorting(seed=s)) for s in range(3)],
    ("sorting 24 (greedy)", sorting(24, 7)),
):
    out = S.run(FakeContext({}), tasks=tasks, time_limit=10.0)
    g = S.run(FakeContext({}), tasks=tasks, method="greedy")
    one, alt = baselines(tasks)
    print(
        f"| {name} | {len(tasks)} | {out['makespan']:.1f} ({out['method']}{', optimal' if out['optimal'] else ''}, {out['solve_s']:.2f} s) | "
        f"{g['makespan']:.1f} | {one:.1f} | {alt:.1f} | {out['lower_bound']:.1f} |",
        flush=True,
    )
