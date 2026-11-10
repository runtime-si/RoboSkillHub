"""Temporal (texture-free) D405 noise (static pairs: RGB mean difference < 0.5, median depth within 0.05 mm, at least 0.5 % of pixels changed - not a duplicated save) from pairs of real wrist frames of a static scene.

    python fit_temporal.py 'GLOB' out.json

Frames of one run directory are ordered by name; a consecutive pair counts as static when its RGB images
differ by < 2 grey levels on average and its median depth by < 0.5 mm (robot and scene still). The depth
difference of a static pair holds only what changes between frames - the sensor's noise - not the
table's texture or relief. Per pair: fine noise = std of the difference / sqrt(2) on 60 px patches;
smooth noise = std of the difference low-passed with the same 30 px Gaussian as the model's smooth field
(and with 12 px, the round-5 fit's filter), / sqrt(2); both as coefficients of z^2.
"""

import glob
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

groups = defaultdict(list)
for f in sorted(glob.glob(sys.argv[1], recursive=True)):
    if "overhead" in f:
        continue
    groups[os.path.dirname(f)].append(f)

fine, low30, low12, nz, pairs = [], [], [], [], 0
for files in groups.values():
    frames = []
    for f in files:
        z = np.load(f)
        if "depth_m" not in z.files or "rgb" not in z.files:
            continue
        dm = z["depth_m"].astype(np.float64)
        if dm.shape != (480, 640) or (dm > 0).mean() < 0.5 or np.median(dm[dm > 0]) > 0.35:
            continue
        frames.append((f, dm, z["rgb"].astype(np.float32)))
    for (_fa, a, ra), (_fb, b, rb) in zip(frames, frames[1:], strict=False):
        if np.abs(ra - rb).mean() > 0.5:
            continue
        m = (a > 0) & (b > 0)
        if (
            m.mean() < 0.5
            or abs(np.median(a[m]) - np.median(b[m])) > 0.00005
            or (a[m] != b[m]).mean() < 0.005
        ):
            continue
        diff = np.where(m, a - b, 0.0)
        # Keep the near, valid part: drop the frame's > 2 cm depth edges.
        z2 = np.where(m, a, 0.0) ** 2
        pairs += 1
        zmed = float(np.median(a[m]))
        nz.append(zmed)
        w = m.astype(np.float64)
        for k, out in ((30, low30), (12, low12)):
            num = cv2.GaussianBlur(diff, (0, 0), k)
            den = cv2.GaussianBlur(w, (0, 0), k)
            lp = np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan)
            out.append(float(np.nanstd(lp) / np.sqrt(2) / zmed**2))
        stds = []
        for y in range(0, 480 - 60 + 1, 60):
            for x in range(0, 640 - 60 + 1, 60):
                pm = m[y : y + 60, x : x + 60]
                if pm.mean() < 0.98:
                    continue
                stds.append(
                    diff[y : y + 60, x : x + 60][pm].std()
                    / np.sqrt(2)
                    / np.median(a[y : y + 60, x : x + 60][pm]) ** 2
                )
        if stds:
            fine.append(float(np.median(stds)))

res = {
    "static_pairs": pairs,
    "z_median_m": float(np.median(nz)) if nz else None,
    "fine_coeff_per_m": {"median": float(np.median(fine)), "p90": float(np.percentile(fine, 90))}
    if fine
    else None,
    "lowfreq30_coeff_per_m": {
        "median": float(np.median(low30)),
        "p90": float(np.percentile(low30, 90)),
    }
    if low30
    else None,
    "lowfreq12_coeff_per_m": {
        "median": float(np.median(low12)),
        "p90": float(np.percentile(low12, 90)),
    }
    if low12
    else None,
}
if nz:
    zm = res["z_median_m"]
    res["at_z_median_mm"] = {
        "fine": round(res["fine_coeff_per_m"]["median"] * zm**2 * 1e3, 3),
        "low30": round(res["lowfreq30_coeff_per_m"]["median"] * zm**2 * 1e3, 3),
        "low12": round(res["lowfreq12_coeff_per_m"]["median"] * zm**2 * 1e3, 3),
    }
    res["at_0p15_m_mm"] = {
        k: round(v["median"] * 0.15**2 * 1e3, 3)
        for k, v in (
            ("fine", res["fine_coeff_per_m"]),
            ("low30", res["lowfreq30_coeff_per_m"]),
            ("low12", res["lowfreq12_coeff_per_m"]),
        )
    }
open(sys.argv[2], "w").write(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
