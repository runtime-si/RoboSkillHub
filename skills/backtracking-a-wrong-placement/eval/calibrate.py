"""Fit P(piece belongs in cell) from the localizer's score and margin (CONF_* in backtrack.py).

    python skills/backtracking-a-wrong-placement/eval/calibrate.py

Runs localizing-a-piece-in-a-reference-image on every piece of the 22 x 25 procedural picture
for photo seeds 0 and 1 (top_k 8, cached in /tmp), then fits a ridge logistic regression
(lambda 0.1 on standardised features) on every (piece, listed cell) pair of the textured
pieces: features the cell's score and min(margin, MARGIN_CAP), the margin taken against
the piece's best other listed cell. Prints the coefficients and a reliability table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # noqa: E402

LAMBDA = 0.1


def pairs(records):
    x, y = [], []
    for piece, rec in records.items():
        scored = [c for c in rec["candidates"] if c.get("score") is not None]
        if len(scored) < len(rec["candidates"]):
            continue  # a low-texture piece's colour shortlist
        true = bm.sim.true_cell(piece)
        for k, c in enumerate(scored):
            others = [o["score"] for j, o in enumerate(scored) if j != k]
            m = c["score"] - max(others) if others else bm.SCRIPT.MARGIN_CAP
            x.append([c["score"], min(m, bm.SCRIPT.MARGIN_CAP)])
            y.append((c["col"], c["row"]) == true)
    return np.array(x), np.array(y, float)


def main():
    xs, ys = [], []
    for seed in (0, 1):
        recs = bm.localizer_records(Path(f"/tmp/bt_loc_seed{seed}_k8.json"), seed, 8, 12)
        x, y = pairs(recs)
        xs.append(x)
        ys.append(y)
    x, y = np.vstack(xs), np.concatenate(ys)
    mu, sd = x.mean(0), x.std(0)
    a = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    w = np.zeros(3)
    reg = LAMBDA * np.diag([0.0, 1.0, 1.0])
    for _ in range(200):
        p = 1 / (1 + np.exp(-a @ w))
        h = a.T @ (a * (p * (1 - p))[:, None]) + reg
        w += np.linalg.solve(h, a.T @ (y - p) - reg @ w)
    raw = [w[0] - w[1] * mu[0] / sd[0] - w[2] * mu[1] / sd[1], w[1] / sd[0], w[2] / sd[1]]
    print(f"{len(y)} pairs, {int(y.sum())} right")
    print("CONF_BIAS, CONF_SCORE, CONF_MARGIN =", ", ".join(f"{v:.2f}" for v in raw))
    p = 1 / (1 + np.exp(-a @ w))
    for lo, hi in ((0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.01)):
        sel = (p >= lo) & (p < hi)
        print(f"rated {lo:.1f}-{hi:.1f}: {sel.sum():5d} pairs, {y[sel].mean():.3f} right")


if __name__ == "__main__":
    main()
