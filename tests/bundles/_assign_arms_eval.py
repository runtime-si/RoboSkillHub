"""Random straight carries across a two-YAM workspace (tests/bundles/_yam_ik.py, CPU IK): how many one arm
can carry alone, how many the DP completes with handovers.  python -m tests.bundles._assign_arms_eval"""

import importlib.util
from pathlib import Path

import numpy as np
from gap.testing import FakeContext

from . import _yam_ik as Y

p = (
    Path(__file__).resolve().parents[2]
    / "skills/assigning-motion-segments-to-arms-with-handovers/scripts/assign_arms.py"
)
spec = importlib.util.spec_from_file_location("assign_arms", p)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
rng = np.random.default_rng(0)
arms = Y.TwoArms()
ctx = FakeContext({"robot.solve_ik": arms.solve_ik})
rows = []
for t in range(30):
    a = np.array([rng.uniform(0.15, 0.42), rng.uniform(-0.45, 0.45), rng.uniform(0.05, 0.2)])
    b = np.array([rng.uniform(0.15, 0.42), rng.uniform(-0.45, 0.45), rng.uniform(0.05, 0.2)])
    wps = [list(a + (b - a) * s) for s in np.linspace(0, 1, 12)]
    out = A.run(ctx, waypoints=wps)
    one = any(
        all(arms.solve_ik({"position": dict(zip("xyz", w, strict=False))}, arm_id=k)["success"] for w in wps)
        for k in (0, 1)
    )
    rows.append((one, out["route"], out["n_handovers"], len(out["unreachable_waypoints"])))
    print(t, "single-arm" if one else "-", out["route"], out["n_handovers"], flush=True)
n = len(rows)
print(f"single arm can carry: {sum(r[0] for r in rows)}/{n}")
print(
    f"DP completes: {sum(r[1] != 'infeasible' for r in rows)}/{n} "
    f"(assignment_found {sum(r[1] == 'assignment_found' for r in rows)}, handover_required {sum(r[1] == 'handover_required' for r in rows)}, "
    f"infeasible {sum(r[1] == 'infeasible' for r in rows)}, of which with waypoints no arm reaches {sum(r[1] == 'infeasible' and r[3] > 0 for r in rows)})"
)
print(f"handovers when required: {[r[2] for r in rows if r[1] == 'handover_required']}")
