"""Synthetic crossing tasks for inserting-pauses-to-resolve-two-arm-conflicts (CPU):
each arm goes home -> a reach into the shared middle (|y| < 0.12 m) -> home, the second starting 0-1 s
later; counts conflicts and how the pauses resolve them.  python -m tests.bundles._pauses_eval"""

import importlib.util
from pathlib import Path

import numpy as np
from gap.testing import FakeContext

p = (
    Path(__file__).resolve().parents[2]
    / "skills/inserting-pauses-to-resolve-two-arm-conflicts/scripts/insert_pauses.py"
)
spec = importlib.util.spec_from_file_location("insert_pauses", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
M = m._model()
LIM = np.array([(-2.6, 3.0), (0, 3.6), (0, 3.6), (-1.57, 1.57), (-1.57, 1.57), (-2.0, 2.0)])
rng = np.random.default_rng(0)


def reach(arm):
    base = m._base(M, arm, None)
    while True:
        q = rng.uniform(LIM[:, 0], LIM[:, 1])
        c, _ = m.spheres(q, base, M)
        tip = c[-34:].mean(axis=0)
        if abs(tip[1]) < 0.12 and 0.05 < tip[2] < 0.3 and tip[0] > -0.25 and c[:, 2].min() > 0.0:
            return q


def traj(q, start):
    z = np.zeros(6)
    t = start + np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    return {"times": t.tolist(), "joints": [z, 0.5 * q, q, 0.5 * q, z]}


rows = []
for k in range(30):
    a, b = traj(reach(0), 0.0), traj(reach(1), float(rng.uniform(0, 1.0)))
    out = m.run(FakeContext({}), trajectory_a=a, trajectory_b=b)
    rows.append(out)
    print(
        k,
        out["route"],
        out["min_clearance_before"],
        out["min_clearance_after"],
        out["added_time"],
        flush=True,
    )
conf = [r for r in rows if r["route"] != "no_conflict"]
print(
    f"tasks {len(rows)}; conflicting {len(conf)}; resolved {sum(r['route'] == 'pauses_inserted' for r in conf)}; "
    f"unresolved {sum(r['route'] == 'unresolved' for r in conf)}"
)
added = [r["added_time"] for r in conf if r["route"] == "pauses_inserted"]
if added:
    print(
        f"added time when resolved: median {np.median(added):.2f} s, max {max(added):.2f} s (of a 4 s task)"
    )
    print(
        f"min clearance before (conflicting): median {np.median([r['min_clearance_before'] for r in conf]):.3f} m; after: "
        f"min {min(r['min_clearance_after'] for r in conf if r['route'] == 'pauses_inserted'):.3f} m"
    )
