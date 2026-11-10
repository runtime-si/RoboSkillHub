# Adapted from arc-robot-vision (https://github.com/andyzeng/arc-robot-vision, commit 728139e9ad0e),
# suction-based-grasping/baseline/predict.m, Apache License 2.0, Copyright (c) 2017 Andy Zeng et al.
# Changes: numpy / scipy instead of MATLAB; normals from the cross product of the organised point
# cloud's image-space gradients instead of pcnormals' 50-neighbour PCA; the foreground is a given mask
# (a segmenter's) instead of background subtraction; candidates must lie at least a cup radius
# (plus a margin) inside the mask, so the cup's lip stays on the piece (the candidate brief's edge rule).
"""Suction affordance from one depth frame: low local surface-normal variance = good seal."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

#: Upstream's stdfilt window (25 x 25 px at 640 x 480).
STD_WINDOW_PX = 25
#: Depth smoothing before the normals (stands in for pcnormals' 50-neighbour fit): a 7 x 7 box
#: within the mask. Per-pixel gradients of a D405-like depth (0.4 mm noise at 0.6 mm pixels) are
#: noise; upstream's neighbourhood fit was not.
SMOOTH_PX = 7


def smooth_depth(depth: np.ndarray, mask: np.ndarray, size: int = SMOOTH_PX) -> np.ndarray:
    """Normalised box filter of the depth over valid mask pixels (no bleeding across the mask edge)."""
    valid = mask & np.isfinite(depth) & (depth > 0)
    num = ndimage.uniform_filter(np.where(valid, depth, 0.0), size)
    den = ndimage.uniform_filter(valid.astype(np.float64), size)
    return np.where(valid, num / np.maximum(den, 1e-9), 0.0)


def camera_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """``[H, W, 3]`` camera-frame points (OpenCV axes) of a metric depth image; 0 where depth is invalid."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float64)
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=-1)


def surface_normals(points: np.ndarray) -> np.ndarray:
    """Unit normals pointing at the sensor (upstream flips them toward the camera centre)."""
    du = np.gradient(points, axis=1)
    dv = np.gradient(points, axis=0)
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-12)
    facing = np.sum(n * points, axis=-1) > 0  # pointing away from the camera
    n[facing] *= -1
    return n


def affordance(
    depth: np.ndarray, K: np.ndarray, mask: np.ndarray, window: int = STD_WINDOW_PX
) -> tuple[np.ndarray, np.ndarray]:
    """``(affordance [H, W] in 0..1, normals [H, W, 3])``: 1 - mean local std of the normals, normalised."""
    pts = camera_points(smooth_depth(depth, mask), K)
    valid = mask & (pts[..., 2] > 0)
    n = surface_normals(pts)
    n[~valid] = 0.0
    mean = ndimage.uniform_filter(n, size=(window, window, 1))
    sq = ndimage.uniform_filter(n * n, size=(window, window, 1))
    std = np.sqrt(np.clip(sq - mean * mean, 0.0, None)).mean(axis=-1)
    top = float(std[valid].max()) if valid.any() else 0.0
    aff = 1.0 - std / top if top > 0 else np.where(valid, 1.0, 0.0)
    aff[~valid] = 0.0
    return aff, n


def best_point(
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    cup_radius_m: float,
    margin_m: float = 0.001,
    window: int = STD_WINDOW_PX,
) -> dict[str, Any] | None:
    """The best cup centre on the mask: highest affordance among pixels at least a cup radius plus a
    margin inside the mask. ``None`` when no pixel is that far inside."""
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return None
    aff, n = affordance(
        np.asarray(depth, dtype=np.float64), np.asarray(K, dtype=np.float64), mask, window
    )
    z = (
        float(np.median(depth[mask & np.isfinite(depth) & (depth > 0)]))
        if (mask & (depth > 0)).any()
        else 0.0
    )
    if z <= 0:
        return None
    need_px = (cup_radius_m + margin_m) * float(K[0, 0]) / z
    inside = ndimage.distance_transform_edt(mask)
    ok = inside >= need_px
    if not ok.any():
        return None
    score = np.where(ok, aff, -1.0)
    # among ties (flat pieces score ~1 almost everywhere), prefer the most interior pixel
    best = np.flatnonzero(score >= score.max() - 0.02)
    k = int(best[np.argmax(inside.ravel()[best])])
    v, u = divmod(k, mask.shape[1])
    pts = camera_points(depth, K)
    return {
        "pixel": [int(u), int(v)],
        "point_camera": pts[v, u].tolist(),
        "normal_camera": n[v, u].tolist(),
        "affordance": float(aff[v, u]),
        "inside_px": float(inside[v, u]),
        "need_px": float(need_px),
    }
