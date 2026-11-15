"""Travel-time benchmarks for ordering-pick-and-place-tasks-for-shortest-travel (CPU).
python -m tests.bundles._order_eval"""

import importlib.util
import time
from pathlib import Path

import numpy as np
from gap.testing import FakeContext

p = (
    Path(__file__).resolve().parents[2]
    / "skills/ordering-pick-and-place-tasks-for-shortest-travel/scripts/order_tasks.py"
)
spec = importlib.util.spec_from_file_location("order_tasks", p)
OT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OT)
HOME = [0.0, 0.0, 0.2]


def pens(perm, seed=0):
    a = [[-0.25 + 0.04 * (k % 4), 0.20 + 0.04 * (k // 4), 0.05] for k in range(12)]
    b = [[-0.25 + 0.04 * (k % 4), -0.20 - 0.04 * (k // 4), 0.05] for k in range(12)]
    order = np.random.default_rng(seed).permutation(12) if perm else np.arange(12)
    return [{"id": f"p{k}", "pick": a[k], "place": b[int(order[k])]} for k in range(12)]


def sorting(n, seed):
    rng = np.random.default_rng(seed)
    bins = [[0.0, 0.35, 0.05], [0.0, -0.35, 0.05]]
    return [
        {
            "id": f"o{k}",
            "pick": [rng.uniform(-0.3, 0.2), rng.uniform(-0.3, 0.3), 0.03],
            "place": bins[int(rng.random() < 0.5)],
        }
        for k in range(n)
    ]


def nn_cost(tasks):
    ids = [t["id"] for t in tasks]
    picks = {t["id"]: np.array(t["pick"], float) for t in tasks}
    places = {t["id"]: np.array(t["place"], float) for t in tasks}
    s0 = np.array(HOME)
    seq = OT._nearest(ids, s0, picks, places, {}, OT.SPEED, OT.LIFT)
    return OT._cost(seq, s0, picks, places, OT.SPEED, OT.LIFT)


for name, tasks in (
    ("12 pens A->B, grid (slot k -> slot k)", pens(False)),
    ("12 pens A->B, random slots", pens(True)),
    ("12 pens A->B, random slots, seed 1", pens(True, 1)),
    ("sorting 10, seed 0", sorting(10, 0)),
    ("sorting 12, seed 1", sorting(12, 1)),
    ("sorting 30, seed 2 (heuristic)", sorting(30, 2)),
):
    t = time.time()
    out = OT.run(FakeContext({}), tasks=tasks, start={"0": HOME})
    nn = nn_cost(tasks)
    g = out["total_given_s"]
    print(
        f"| {name} | {len(tasks)} | {g:.1f} | {nn:.1f} | **{out['total_travel_s']:.1f}** ({100 * (1 - out['total_travel_s'] / g):.0f} % less) | {out['method']} | {time.time() - t:.2f} s |",
        flush=True,
    )
