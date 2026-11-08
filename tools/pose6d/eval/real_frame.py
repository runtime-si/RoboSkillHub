# SPDX-License-Identifier: MIT
"""Fit a known box to a dark part in a real RGB-D frame and report residuals.

    python tools/pose6d/eval/real_frame.py --npz obs.npz --roi 220 215 310 295 --box 0.03 0.03 0.03

The frame is an ``.npz`` with ``rgb`` (H, W, 3), ``depth_m`` (H, W) and ``K``
(3x3). The part is segmented as the largest dark component in the ROI (a dark
part on a light table; no learned segmenter), the table as a RANSAC plane over
the rest of the ROI grown by 60 px. With no ground truth, the checks are
internal: residuals of the measured surface against the fitted box, how far the
fitted box's bottom face sits from the measured table plane, and the angle
between the box's up axis and the table normal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _impl  # noqa: E402


def backproject(depth, mask, K):
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u].astype(np.float64)
    return np.column_stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def ransac_plane(P, thresh=0.002, iters=500, seed=0):
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < thresh
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    Q = P[best]
    m = Q.mean(axis=0)
    n = np.linalg.svd(Q - m)[2][2]
    if n @ (-m) < 0:  # normal toward the camera (out of the table)
        n = -n
    return np.array([*n, -n @ m]), best


def draw_overlay(rgb, K, box, poses, mask, path, roi):
    from PIL import Image, ImageDraw

    h = np.asarray(box) / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
    edges = [
        (i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(corners[i] != corners[j]) == 1
    ]
    img = rgb.copy()
    img[mask] = (0.6 * img[mask] + 0.4 * np.array([0, 160, 255])).astype(np.uint8)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for (name, T), col in zip(poses.items(), [(255, 60, 60), (60, 255, 60)], strict=False):
        c = corners @ T[:3, :3].T + T[:3, 3]
        uv = (c @ K.T)[:, :2] / c[:, 2:3]
        for i, j in edges:
            dr.line([tuple(uv[i]), tuple(uv[j])], fill=col, width=1)
        dr.text((5, 5 + 12 * list(poses).index(name)), name, fill=col)
    x0, y0, x1, y1 = roi
    pad = 40
    im = im.crop((max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad))
    im.resize((im.width * 4, im.height * 4), Image.NEAREST).save(path)


def main():
    from scipy import ndimage

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--roi", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--box", type=float, nargs=3, default=[0.03, 0.03, 0.03])
    ap.add_argument(
        "--dark", type=float, default=70.0, help="grey level below which a pixel is the part"
    )
    ap.add_argument("--out", default="")
    ap.add_argument("--overlay", default="", help="PNG with the fitted boxes drawn on the RGB")
    a = ap.parse_args()

    d = np.load(a.npz)
    rgb, depth, K = d["rgb"], d["depth_m"].astype(np.float64), d["K"]
    x0, y0, x1, y1 = a.roi
    roi = np.zeros(depth.shape, bool)
    roi[y0:y1, x0:x1] = True
    dark = (rgb.mean(axis=2) < a.dark) & roi
    lab, n = ndimage.label(dark)
    if n == 0:
        raise SystemExit("no dark pixels in the ROI")
    sizes = ndimage.sum(dark, lab, range(1, n + 1))
    mask = lab == (1 + int(np.argmax(sizes)))
    P = backproject(depth, mask, K)

    ring = np.zeros_like(roi)
    ring[max(0, y0 - 60) : y1 + 60, max(0, x0 - 60) : x1 + 60] = True
    ring &= ~ndimage.binary_dilation(mask, iterations=6) & ~dark
    plane, _ = ransac_plane(backproject(depth, ring, K))
    up = plane[:3]

    spec = {"box": a.box}
    report = {"npz": a.npz, "mask_px": int(mask.sum()), "n_points": int(len(P)), "box_m": a.box}
    report["table_plane"] = plane.tolist()
    for name, kw in {
        "fast": {},
        "fast_support": {"support_plane": plane},
    }.items():
        r = _impl.register(P, spec, viewpoint=np.zeros(3), up=up, **kw)
        T = r["T"]
        surf = _impl.BoxSurface(np.asarray(a.box))
        res = _impl.surface_residuals(P, surf, T[:3, :3], T[:3, 3], np.zeros(3))
        h = np.asarray(a.box) / 2
        corners = (
            np.array(np.meshgrid(*[(-x, x) for x in h])).reshape(3, -1).T @ T[:3, :3].T + T[:3, 3]
        )
        heights = corners @ plane[:3] + plane[3]
        tilt = math.degrees(math.acos(min(1.0, abs(float(T[:3, 2] @ up)))))
        report[name] = {
            "status": r["status"],
            "n_points_fit": r["n_points"],
            "n_dropped_support": r["n_dropped_support"],
            "fitness": round(r["fitness"], 3),
            "coverage": round(r["coverage"], 3),
            "rmse_mm": round(1000 * r["rmse"], 2),
            "inlier_threshold_mm": round(1000 * r["inlier_threshold_m"], 2),
            "residual_median_mm": round(1000 * float(np.median(res)), 2),
            "residual_p90_mm": round(1000 * float(np.percentile(res, 90)), 2),
            "bottom_gap_to_table_mm": round(1000 * float(np.sort(heights)[:4].mean()), 2),
            "centre_height_mm": round(1000 * float(T[:3, 3] @ plane[:3] + plane[3]), 2),
            "tilt_vs_table_deg": round(tilt, 2),
            "yaw_ambiguity_deg": r["yaw_ambiguity_deg"],
            "symmetry_order": r["symmetry_order"],
            "ambiguous_fit": r["ambiguous_fit"],
            "runtime_ms": round(1000 * r["runtime_s"], 1),
            "T_cam_obj": np.round(T, 5).tolist(),
        }
    top = P @ plane[:3] + plane[3]
    report["measured_top_height_mm"] = round(1000 * float(np.percentile(top, 95)), 2)
    sweep = {}
    for s in np.arange(0.024, 0.0361, 0.002):
        r = _impl.register(P, {"cube": float(s)}, viewpoint=np.zeros(3), up=up, support_plane=plane)
        sweep[f"{1000 * s:.0f}"] = {
            "fitness": round(r["fitness"], 3),
            "coverage": round(r["coverage"], 3),
            "rmse_mm": round(1000 * r["rmse"], 2),
        }
    report["cube_size_sweep_mm"] = sweep
    if a.overlay:
        draw_overlay(
            rgb,
            K,
            a.box,
            {k: np.asarray(report[k]["T_cam_obj"]) for k in ("fast", "fast_support")},
            mask,
            a.overlay,
            a.roi,
        )
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
