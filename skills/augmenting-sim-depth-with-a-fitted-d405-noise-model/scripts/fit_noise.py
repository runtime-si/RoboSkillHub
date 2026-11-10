"""Fit a D405 depth-noise model to real wrist frames (npz with depth_m and K).

    python fit_noise.py 'GLOB' out.json

Per frame, 60 x 60 px patches whose depth is nearly planar (a flat table) give the per-pixel residual
after a plane fit (fine noise) and its spatial autocorrelation; 192 px windows give the smooth error left
after a plane fit, low-passed (the sensor's low-frequency error plus the table's relief); edge pixels give
the dropout rate at depth discontinuities. sigma(z) is fitted as c * z^2 (stereo: sigma = z^2 * subpix /
(f * baseline)).
"""

import glob
import json
import sys

import cv2
import numpy as np

files = sorted(f for f in glob.glob(sys.argv[1], recursive=True) if "overhead" not in f)
P, W = 60, 192
fine, zs, corr1, corr2, lowf, lowz, drop_all, drop_edge, steps, used = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    0,
)


def plane_resid(d):
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    m = d > 0
    A = np.c_[xx[m], yy[m], np.ones(m.sum())]
    c, *_ = np.linalg.lstsq(A, d[m], rcond=None)
    r = np.zeros_like(d)
    r[m] = d[m] - A @ c
    return r, m


for f in files:
    z = np.load(f)
    if "depth_m" not in z.files:
        continue
    d = z["depth_m"].astype(np.float64)
    if d.shape != (480, 640) or (d > 0).mean() < 0.5 or np.median(d[d > 0]) > 0.35:
        continue
    used += 1
    v = d[d > 0]
    u = np.unique(np.round(v[:20000] * 1e5).astype(np.int64))
    if len(u) > 10:
        steps.append(np.median(np.diff(u)) / 1e5)
    drop_all.append(float((d <= 0).mean()))
    g = cv2.morphologyEx((d > 0).astype(np.uint8), cv2.MORPH_ERODE, np.ones((3, 3), np.uint8))
    dz = np.abs(
        cv2.Sobel(np.where(d > 0, d, np.median(v)).astype(np.float32), cv2.CV_32F, 1, 0)
    ) + np.abs(cv2.Sobel(np.where(d > 0, d, np.median(v)).astype(np.float32), cv2.CV_32F, 0, 1))
    edge = dz > 0.02  # > 2 cm jump over a Sobel stencil: a depth discontinuity
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if edge.sum() > 50:
        drop_edge.append(float((d[edge] <= 0).mean()))
    for y in range(0, 480 - P + 1, P):
        for x in range(0, 640 - P + 1, P):
            p = d[y : y + P, x : x + P]
            if (p > 0).mean() < 0.98:
                continue
            r, m = plane_resid(p)
            s = r[m].std()
            if s > 0.003:  # not planar (an object edge in the patch)
                continue
            fine.append(s)
            zs.append(float(np.median(p[m])))
            rr = r - r[m].mean()
            corr1.append(float((rr[:, 1:] * rr[:, :-1]).mean() / rr.var()))
            corr2.append(float((rr[:, 3:] * rr[:, :-3]).mean() / rr.var()))
    for y in range(0, 480 - W + 1, W // 2):
        for x in range(0, 640 - W + 1, W // 2):
            p = d[y : y + W, x : x + W]
            if (p > 0).mean() < 0.97:
                continue
            r, m = plane_resid(p)
            if r[m].std() > 0.004:
                continue
            lp = cv2.GaussianBlur(r.astype(np.float32), (0, 0), 12)
            lowf.append(float(lp[m].std()))
            lowz.append(float(np.median(p[m])))

fine, zs = np.array(fine), np.array(zs)
c = float(np.median(fine / zs**2))
rho1, rho3 = float(np.median(corr1)), float(np.median(corr2))
# Gaussian-filtered white noise: rho(d) = exp(-d^2 / (4 s^2)) -> s from the lag-1 and lag-3 correlations.
s1 = float(np.sqrt(-1 / (4 * np.log(max(min(rho1, 0.999), 1e-3)))))
s3 = float(3 / np.sqrt(-4 * np.log(max(min(rho3, 0.999), 1e-3))))
out = {
    "frames_used": used,
    "files_seen": len(files),
    "fine_patches": int(len(fine)),
    "z_range_m": [round(float(zs.min()), 3), round(float(zs.max()), 3)],
    "fine_std_m": {
        "median": float(np.median(fine)),
        "p10": float(np.percentile(fine, 10)),
        "p90": float(np.percentile(fine, 90)),
    },
    "sigma_coeff_per_m": c,
    "sigma_at_0p15_m": c * 0.15**2,
    "lag1_corr": rho1,
    "lag3_corr": rho3,
    "corr_sigma_px": round((s1 + s3) / 2, 3),
    "corr_sigma_px_from_lag1_lag3": [round(s1, 3), round(s3, 3)],
    "lowfreq_windows": int(len(lowf)),
    "lowfreq_std_m": {"median": float(np.median(lowf)), "p90": float(np.percentile(lowf, 90))}
    if lowf
    else None,
    "lowfreq_coeff_per_m": float(np.median(np.array(lowf) / np.array(lowz) ** 2)) if lowf else None,
    "dropout_all": float(np.median(drop_all)),
    "dropout_at_edges": float(np.median(drop_edge)) if drop_edge else None,
    "quant_step_m": float(np.median(steps)) if steps else None,
}
open(sys.argv[2], "w").write(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
