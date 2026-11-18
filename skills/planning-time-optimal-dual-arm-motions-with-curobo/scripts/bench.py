# SPDX-License-Identifier: Apache-2.0
"""Benchmark (GPU): simultaneous 12-DoF plans vs each arm planned alone and run one after the other.

Each pair is planned once; the paths are then timed at every speed fraction, raw (the joint limits alone)
and stream-capped (velocity also capped at the joint stream's per-row jump * hz).

    python bench.py --n 30 --fractions 0.25 0.5 1.0 --trajopt-seeds 12 --attempts 12 --graph-attempt 0
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from plan_dual_arm import (
    DualArmPlanner,
    arm_spheres,
    collisions,
    inter_arm_distance,
    retime,
    sample,
)

LIMITS = [(-2.6, 3.0), (0.3, 2.6), (0.3, 2.6), (-1.5, 1.5), (-1.5, 1.5), (-2.0, 2.0)]
VEL = [6, 6, 6, 18, 18, 18]
"""rad/s per joint, the proposed high-speed limits (scaled by --fractions)."""
ACC = [15, 15, 15, 40, 40, 40]
"""rad/s^2 per joint."""
JERK_PER_ACC = 10.0
"""jerk limit = this * ACC [1/s]: the acceleration may ramp fully in 0.1 s."""
STREAM_HZ = 30
STREAM_JUMP = 0.35
"""rad per row, the joint stream's per-row limit; with STREAM_HZ it caps speed at 10.5 rad/s."""


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


def timing(paths, vel, acc, jerk):
    """Makespan of running ``paths`` one after the other, and the largest per-row jump at STREAM_HZ."""
    T, jump = 0.0, 0.0
    for p in paths:
        t, q = retime(p, vel, acc, jerk)
        rows = sample(t, q, STREAM_HZ)
        T += float(t[-1])
        if len(rows) > 1:
            jump = max(jump, float(np.abs(np.diff(rows, axis=0)).max()))
    return T, jump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--trajopt-seeds", type=int, default=4)
    ap.add_argument("--ik-seeds", type=int, default=32)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--graph-attempt", type=int, default=1)
    ap.add_argument("--out", default="bench.json")
    a = ap.parse_args()
    t0 = time.time()
    P = DualArmPlanner(num_trajopt_seeds=a.trajopt_seeds, num_ik_seeds=a.ik_seeds)
    print(f"planner ready in {time.time() - t0:.1f}s", flush=True)
    m = P.model
    rng = np.random.default_rng(a.seed)
    P.path(*sample_pair(m, rng, False))  # warm-up

    def path(q0, q1):
        return P.path(q0, q1, max_attempts=a.attempts, enable_graph_attempt=a.graph_attempt)

    acc = np.r_[ACC, ACC].astype(float)
    jerk = acc * JERK_PER_ACC
    rows = []
    for i in range(a.n):
        crossing = i % 2 == 0
        q0, q1 = sample_pair(m, rng, crossing)
        t = time.perf_counter()
        sim, why = path(q0, q1)
        t_sim = time.perf_counter() - t
        t = time.perf_counter()
        mid = np.r_[q1[:6], q0[6:]]
        seq = []
        for a0, a1 in ((q0, mid), (mid, q1)):
            if collisions(m, a1):
                seq = None
                break
            p, _ = path(a0, a1)
            if p is None:
                seq = None
                break
            seq.append(p)
        t_seq = time.perf_counter() - t
        row = {
            "i": i,
            "kind": "crossing" if crossing else "independent",
            "ok": sim is not None,
            "seq_ok": seq is not None,
            "plan_s": round(t_sim, 3),
            "seq_plan_s": round(t_seq, 3),
            "detail": None if sim is not None else why,
            "by_fraction": {},
        }
        if sim is not None:
            sim[0], sim[-1] = q0, q1
            row["min_sphere_m"] = round(min(inter_arm_distance(m, x) for x in sim), 4)
        for f in a.fractions:
            vel = np.r_[VEL, VEL].astype(float) * f
            capped = np.minimum(vel, STREAM_JUMP * STREAM_HZ)
            r = {"stream_cap_binds": bool((vel > STREAM_JUMP * STREAM_HZ).any())}
            if sim is not None:
                r["makespan_s"], r["max_row_jump_rad"] = timing([sim], vel, acc, jerk)
                r["makespan_capped_s"], r["max_row_jump_capped_rad"] = timing(
                    [sim], capped, acc, jerk
                )
            if seq is not None:
                r["seq_makespan_s"], r["seq_max_row_jump_rad"] = timing(seq, vel, acc, jerk)
                r["seq_makespan_capped_s"], _ = timing(seq, capped, acc, jerk)
            row["by_fraction"][str(f)] = {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
            }
        rows.append(row)
        print(json.dumps(row), flush=True)
    summ = {
        "settings": vars(a),
        "vel": VEL,
        "acc": ACC,
        "jerk": (np.array(ACC) * JERK_PER_ACC).tolist(),
        "stream": {
            "hz": STREAM_HZ,
            "max_row_jump_rad": STREAM_JUMP,
            "speed_cap_rad_s": STREAM_JUMP * STREAM_HZ,
        },
    }
    for kind in ("crossing", "independent", "all"):
        R = [r for r in rows if kind in ("all", r["kind"])]
        ok = [r for r in R if r["ok"]]
        both = [r for r in ok if r["seq_ok"]]
        s = {
            "n": len(R),
            "success": len(ok),
            "seq_success": sum(r["seq_ok"] for r in R),
            "plan_s_median": float(np.median([r["plan_s"] for r in ok])) if ok else None,
            "seq_plan_s_median": float(np.median([r["seq_plan_s"] for r in R if r["seq_ok"]]))
            if both
            else None,
            "min_sphere_m": float(min(r["min_sphere_m"] for r in ok)) if ok else None,
            "by_fraction": {},
        }
        for f in map(str, a.fractions):
            B = [r["by_fraction"][f] for r in both]
            okf = [r["by_fraction"][f] for r in ok]
            if not B:
                continue
            s["by_fraction"][f] = {
                "makespan_s_mean": float(np.mean([b["makespan_s"] for b in B])),
                "seq_makespan_s_mean": float(np.mean([b["seq_makespan_s"] for b in B])),
                "ratio_mean": float(np.mean([b["makespan_s"] / b["seq_makespan_s"] for b in B])),
                "makespan_capped_s_mean": float(np.mean([b["makespan_capped_s"] for b in B])),
                "seq_makespan_capped_s_mean": float(
                    np.mean([b["seq_makespan_capped_s"] for b in B])
                ),
                "max_row_jump_rad": float(max(b["max_row_jump_rad"] for b in okf)),
                "max_row_jump_capped_rad": float(max(b["max_row_jump_capped_rad"] for b in okf)),
                "rows_over_jump_limit_pairs": int(
                    sum(b["max_row_jump_rad"] > STREAM_JUMP for b in okf)
                ),
                "stream_cap_binds": B[0]["stream_cap_binds"],
            }
        summ[kind] = s
    print(json.dumps(summ, indent=1))
    with open(a.out, "w") as f:
        json.dump({"summary": summ, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
