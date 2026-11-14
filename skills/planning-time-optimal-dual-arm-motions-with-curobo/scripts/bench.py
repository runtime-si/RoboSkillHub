# SPDX-License-Identifier: Apache-2.0
"""Benchmark (GPU): simultaneous 12-DoF plans vs each arm planned alone and run one after the other.

python bench.py --n 30 --out bench.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from plan_dual_arm import DualArmPlanner, arm_spheres, collisions, inter_arm_distance, plan, retime

LIMITS = [(-2.6, 3.0), (0.3, 2.6), (0.3, 2.6), (-1.5, 1.5), (-1.5, 1.5), (-2.0, 2.0)]


def tip_y(m, arm, q):
    return float(arm_spheres(m, arm, q)["link_6"][:, 1].mean())


def sample_pair(m, rng, crossing: bool):
    """Collision-free start and goal; crossing = each arm's hand changes side of the midline (y = 0)."""
    while True:
        q0 = np.array([rng.uniform(lo, hi) for lo, hi in LIMITS * 2])
        q1 = np.array([rng.uniform(lo, hi) for lo, hi in LIMITS * 2])
        if collisions(m, q0) or collisions(m, q1):
            continue
        ys = [
            tip_y(m, "left", q0[:6]),
            tip_y(m, "left", q1[:6]),
            tip_y(m, "right", q0[6:]),
            tip_y(m, "right", q1[6:]),
        ]
        if crossing and ys[0] > 0.05 and ys[1] < -0.05 and ys[2] < -0.05 and ys[3] > 0.05:
            return q0, q1
        if not crossing and ys[0] > 0.05 and ys[1] > 0.05 and ys[2] < -0.05 and ys[3] < -0.05:
            return q0, q1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench.json")
    a = ap.parse_args()
    t0 = time.time()
    P = DualArmPlanner()
    print(f"planner ready in {time.time() - t0:.1f}s; joints {P.joint_names}", flush=True)
    m = P.model
    rng = np.random.default_rng(a.seed)
    P.path(*sample_pair(m, rng, False))  # warm-up
    rows = []
    for i in range(a.n):
        crossing = i % 2 == 0
        q0, q1 = sample_pair(m, rng, crossing)
        t = time.perf_counter()
        r = plan(P, {"left": q0[:6], "right": q0[6:]}, {"left": q1[:6], "right": q1[6:]})
        t_sim = time.perf_counter() - t
        # sequential: left alone (right parked at its start), then right alone (left parked at its goal)
        t = time.perf_counter()
        mid = np.r_[q1[:6], q0[6:]]
        seq_ok, seq_T, seq_d = True, 0.0, np.inf
        for a0, a1 in ((q0, mid), (mid, q1)):
            if collisions(m, a1):
                seq_ok = False
                break
            path, _ = P.path(a0, a1)
            if path is None:
                seq_ok = False
                break
            tt, qq = retime(path)
            seq_T += float(tt[-1])
            seq_d = min(seq_d, min(inter_arm_distance(m, x) for x in qq[:: max(1, len(qq) // 60)]))
        t_seq = time.perf_counter() - t
        row = {
            "i": i,
            "kind": "crossing" if crossing else "independent",
            "exit": r["exit"],
            "plan_s": round(t_sim, 3),
            "makespan_s": round(r.get("makespan_s", float("nan")), 3),
            "min_sphere_m": round(r.get("min_inter_arm_sphere_m", float("nan")), 4),
            "seq_ok": seq_ok,
            "seq_plan_s": round(t_seq, 3),
            "seq_makespan_s": round(seq_T, 3) if seq_ok else None,
            "seq_min_sphere_m": round(seq_d, 4) if seq_ok else None,
            "detail": r.get("detail"),
        }
        rows.append(row)
        print(row, flush=True)
    summ = {}
    for kind in ("crossing", "independent", "all"):
        R = [r for r in rows if kind in ("all", r["kind"])]
        ok = [r for r in R if r["exit"] == "planned"]
        both = [r for r in ok if r["seq_ok"]]
        summ[kind] = {
            "n": len(R),
            "success": len(ok),
            "seq_success": sum(r["seq_ok"] for r in R),
            "plan_s_median": float(np.median([r["plan_s"] for r in ok])) if ok else None,
            "seq_plan_s_median": float(np.median([r["seq_plan_s"] for r in R if r["seq_ok"]]))
            if both
            else None,
            "makespan_s_mean": float(np.mean([r["makespan_s"] for r in both])) if both else None,
            "seq_makespan_s_mean": float(np.mean([r["seq_makespan_s"] for r in both]))
            if both
            else None,
            "makespan_ratio_mean": float(
                np.mean([r["makespan_s"] / r["seq_makespan_s"] for r in both])
            )
            if both
            else None,
            "min_sphere_m": float(min(r["min_sphere_m"] for r in ok)) if ok else None,
        }
    print(json.dumps(summ, indent=1))
    with open(a.out, "w") as f:
        json.dump({"summary": summ, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
