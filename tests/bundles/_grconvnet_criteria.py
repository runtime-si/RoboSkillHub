"""Geometric pass / fail for a top-down grasp on the station frames' cube, open box and cable.

Shared by ``_grconvnet_station_eval.py`` (GR-ConvNet) and ``_grconvnet_short_axis_eval.py``
(proposing-short-axis-grasps). Ground truth comes from the frame's own depth inside the target
mask, in the calibrated world frame:

- cube (30 mm): its top face (points > 20 mm over the table). Pass: grasp point within 8 mm of
  the face's centre, closing heading within 15 deg of a face normal, opening 32-94 mm.
- open box (100 x 100 x 40 mm, thin walls; wider than the 94 mm hand): its rim (points > 28 mm
  up). Pass: grasp point within 10 mm of a wall's line and within its length, closing heading
  within 20 deg of that wall's normal, opening 8-94 mm.
- flat cable (~4 mm): its points. Pass: grasp point within 6 mm of the cable, closing heading
  within 25 deg of the cable's local normal, opening 8-94 mm.
"""

from __future__ import annotations

import math

import numpy as np


def world_points(depth, K, T, mask):
    vs, us = np.nonzero(mask & (depth > 0.01))
    z = depth[vs, us].astype(np.float64)
    cam = np.stack([(us - K[0, 2]) / K[0, 0] * z, (vs - K[1, 2]) / K[1, 1] * z, z], axis=-1)
    return cam @ T[:3, :3].T + T[:3, 3]


def table_z(depth, K, T, mask, pad=40):
    """Median world z of measured pixels in a band around the mask (not on it)."""
    from scipy import ndimage as ndi

    ring = ndi.binary_dilation(mask, iterations=pad) & ~ndi.binary_dilation(mask, iterations=6)
    z = world_points(depth, K, T, ring)[:, 2]
    lo = np.percentile(z, 5)
    return float(np.median(z[z < lo + 0.02]))


def min_area_rect(xy):
    """``(centre, angle_rad, half_sizes)`` of the smallest rectangle around 2-D points."""
    best = None
    for a in np.radians(np.arange(0.0, 90.0, 0.5)):
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        p = xy @ R.T
        lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            centre = R.T @ ((lo + hi) / 2)
            best = (area, centre, a, (hi - lo) / 2)
    return best[1], best[2], best[3]


def _ang_err(a_deg, b_deg, period):
    d = (a_deg - b_deg) % period
    return min(d, period - d)


def ground_truth(kind, depth, K, T, mask):
    tz = table_z(depth, K, T, mask)
    P = world_points(depth, K, T, mask)
    h = P[:, 2] - tz
    if kind == "cube":
        top = P[h > 0.020]
        centre, a, half = min_area_rect(top[:, :2])
        return {
            "kind": kind,
            "table_z": tz,
            "centre": centre,
            "angle_deg": math.degrees(a),
            "half": half,
        }
    if kind == "box":
        rim = P[h > 0.028]
        centre, a, half = min_area_rect(rim[:, :2])
        return {
            "kind": kind,
            "table_z": tz,
            "centre": centre,
            "angle_deg": math.degrees(a),
            "half": half,
        }
    return {"kind": kind, "table_z": tz, "points": P[:, :2]}


def judge(gt, xy, heading_deg, opening_m):
    """``(passed, why)`` for a grasp at world ``xy`` closing along ``heading_deg``."""
    xy = np.asarray(xy, dtype=np.float64)
    kind = gt["kind"]
    if kind == "cube":
        d = float(np.linalg.norm(xy - gt["centre"]))
        e = _ang_err(heading_deg, gt["angle_deg"], 90.0)
        ok = d < 0.008 and e < 15 and 0.032 <= opening_m <= 0.094
        return ok, f"off {1e3 * d:.0f}mm, heading err {e:.0f}deg, opening {1e3 * opening_m:.0f}mm"
    if kind == "box":
        a = math.radians(gt["angle_deg"])
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        p = R @ (xy - gt["centre"])
        hx, hy = gt["half"]
        cands = []
        if abs(p[1]) <= hy + 0.005:
            cands.append((abs(abs(p[0]) - hx), gt["angle_deg"]))  # wall at x = +-hx, normal e1
        if abs(p[0]) <= hx + 0.005:
            cands.append((abs(abs(p[1]) - hy), gt["angle_deg"] + 90.0))
        if not cands:
            return False, "outside the box"
        d, normal = min(cands)
        e = _ang_err(heading_deg, normal, 180.0)
        ok = d < 0.010 and e < 20 and 0.008 <= opening_m <= 0.094
        return (
            ok,
            f"wall off {1e3 * d:.0f}mm, heading err {e:.0f}deg, opening {1e3 * opening_m:.0f}mm",
        )
    P = gt["points"]
    dist = np.linalg.norm(P - xy, axis=1)
    i = int(np.argmin(dist))
    near = P[np.linalg.norm(P - P[i], axis=1) < 0.02]
    c = near - near.mean(0)
    tangent = np.linalg.svd(c, full_matrices=False)[2][0]
    normal = math.degrees(math.atan2(tangent[0], -tangent[1]))
    e = _ang_err(heading_deg, normal, 180.0)
    ok = dist[i] < 0.006 and e < 25 and 0.008 <= opening_m <= 0.094
    return ok, f"off {1e3 * dist[i]:.0f}mm, heading err {e:.0f}deg, opening {1e3 * opening_m:.0f}mm"
