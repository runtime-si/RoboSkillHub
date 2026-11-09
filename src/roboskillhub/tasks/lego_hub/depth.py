# SPDX-License-Identifier: Apache-2.0
"""Depth images ray-cast from observed brick and plate poses, for the hub's camera skills on stations
without a camera (the LEGO physics station serves privileged object poses only).

Geometry in each object's contract frame (``lego_station``): a LEGO 2x4 brick is hollow - a 1 mm top, 1.2 mm
walls, three 6.5 mm underside tubes (as square prisms of the same area) - with its eight studs (4.8 mm x
1.7 mm), origin at the bottom centre, +x along the 2-stud side; the baseplate is a 3.2 mm slab with a stud
on every lattice node, its observed position the top-surface centre; the table is the plane z = 0. Noise is
the D405 model of the hub's tests (sigma = z^2 * subpixel / (f * baseline), spatially correlated, flying
pixels at depth edges, dropouts, 0.1 mm quantisation).

This is a rendering of what the station *observes* (simulation truth, including a brick resting on stud
tops, tilted or half pressed as BrickSim left it), not an Isaac camera: no hand in the view, no
specularity, no stereo shadowing.
"""

from __future__ import annotations

import math

import numpy as np

PITCH, STUD_R, STUD_H = 0.008, 0.0024, 0.0017
BRICK_H, TOP_T, WALL_T, TUBE_SIDE = 0.0096, 0.0010, 0.0012, 0.00651 * math.sqrt(math.pi) / 2
PLATE_T = 0.0032
D405_K = np.array([[394.0, 0.0, 320.0], [0.0, 394.0, 240.0], [0.0, 0.0, 1.0]])
"""640 x 480 D405 intrinsics (refining-target-from-wrist-closeup: fx ~394 px)."""


def quat_matrix(q_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in q_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _box(o: np.ndarray, d: np.ndarray, lo, hi) -> np.ndarray:
    """Entry distance of rays into an axis-aligned box (inf if missed)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1, t2 = (lo - o) * inv, (hi - o) * inv
    tmin = np.where(np.isnan(np.minimum(t1, t2)), -np.inf, np.minimum(t1, t2)).max(axis=1)
    tmax = np.where(np.isnan(np.maximum(t1, t2)), np.inf, np.maximum(t1, t2)).min(axis=1)
    return np.where((tmin <= tmax) & (tmin > 0), tmin, np.inf)


def _studs(o, d, z0, h, r, pitch, nx, ny, x0, y0) -> np.ndarray:
    """Nearest hit on studs (top disc or side) at nodes (x0 + (i + 0.5) pitch, y0 + (j + 0.5) pitch),
    0 <= i < nx, 0 <= j < ny, from z0 to z0 + h."""
    best = np.full(len(o), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        seen = set()
        for zz in (z0 + h, z0 + 0.5 * h, z0):
            t = (zz - o[:, 2]) / d[:, 2]
            p = o[:, :2] + t[:, None] * d[:, :2]
            idx = np.floor((p - [x0, y0]) / pitch).astype(np.int64)
            key = idx.tobytes()
            if key in seen:
                continue
            seen.add(key)
            ok = (idx[:, 0] >= 0) & (idx[:, 0] < nx) & (idx[:, 1] >= 0) & (idx[:, 1] < ny)
            c = (idx + 0.5) * pitch + [x0, y0]
            t_top = (z0 + h - o[:, 2]) / d[:, 2]
            q = o[:, :2] + t_top[:, None] * d[:, :2] - c
            hit = ok & (t_top > 0) & (np.einsum("ij,ij->i", q, q) <= r * r)
            best = np.where(hit & (t_top < best), t_top, best)
            oc = o[:, :2] - c
            a = np.einsum("ij,ij->i", d[:, :2], d[:, :2])
            b = 2 * np.einsum("ij,ij->i", oc, d[:, :2])
            cc = np.einsum("ij,ij->i", oc, oc) - r * r
            disc = b * b - 4 * a * cc
            t1 = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
            z1 = o[:, 2] + t1 * d[:, 2]
            side = ok & (disc > 0) & (t1 > 0) & (z1 >= z0) & (z1 <= z0 + h)
            best = np.where(side & (t1 < best), t1, best)
    return best


def _local(o, d, position, quat):
    r = quat_matrix(quat)
    return (o - np.asarray(position, float)) @ r, d @ r


def _brick(o, d, position, quat, studs=(2, 4)) -> np.ndarray:
    ob, db = _local(o, d, position, quat)
    nx, ny = studs
    hx, hy = nx * PITCH / 2 - 0.0001, ny * PITCH / 2 - 0.0001
    t = np.full(len(o), np.inf)
    for lo, hi in (
        ((-hx, -hy, BRICK_H - TOP_T), (hx, hy, BRICK_H)),
        ((-hx, -hy, 0.0), (-hx + WALL_T, hy, BRICK_H)),
        ((hx - WALL_T, -hy, 0.0), (hx, hy, BRICK_H)),
        ((-hx, -hy, 0.0), (hx, -hy + WALL_T, BRICK_H)),
        ((-hx, hy - WALL_T, 0.0), (hx, hy, BRICK_H)),
    ):
        t = np.minimum(t, _box(ob, db, lo, hi))
    for i in range(1, nx):
        for j in range(1, ny):
            cx, cy = -nx * PITCH / 2 + i * PITCH, -ny * PITCH / 2 + j * PITCH
            s = TUBE_SIDE / 2
            t = np.minimum(t, _box(ob, db, (cx - s, cy - s, 0.0), (cx + s, cy + s, BRICK_H - TOP_T)))
    return np.minimum(
        t, _studs(ob, db, BRICK_H, STUD_H, STUD_R, PITCH, nx, ny, -nx * PITCH / 2, -ny * PITCH / 2)
    )


def _plate(o, d, position, quat, size_m=0.256) -> np.ndarray:
    ob, db = _local(o, d, position, quat)
    h = size_m / 2
    n = int(round(size_m / PITCH))
    t = _box(ob, db, (-h, -h, -PLATE_T), (h, h, 0.0))
    return np.minimum(t, _studs(ob, db, 0.0, STUD_H, STUD_R, PITCH, n, n, -h, -h))


def render(
    objects: dict,
    cam_pose: np.ndarray,
    k: np.ndarray = D405_K,
    size: tuple[int, int] = (640, 480),
    table_z: float = 0.0,
) -> np.ndarray:
    """Clean depth (float32 m along the optical axis, 0 = no return) of ``objects`` (observe_objects
    entries: position_m, quaternion_xyzw, class) seen from ``cam_pose`` (4x4, OpenCV camera in world)."""
    w, h = size
    u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
    dc = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    d = dc @ cam_pose[:3, :3].T
    o = np.broadcast_to(cam_pose[:3, 3], d.shape).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (table_z - o[:, 2]) / d[:, 2]
    t = np.where(t > 0, t, np.inf)
    for obj in objects.values():
        p, q = obj.get("position_m"), obj.get("quaternion_xyzw")
        if p is None or q is None:
            continue
        cls = obj.get("class")
        if cls == "baseplate":
            size_m = float((obj.get("geometry") or {}).get("size_m", [0.256])[0])
            t = np.minimum(t, _plate(o, d, p, q, size_m))
        elif cls == "lego_brick":
            t = np.minimum(t, _brick(o, d, p, q, tuple(obj.get("studs", (2, 4)))))
    depth = t.reshape(h, w)
    return np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)


def add_noise(depth, k, rng, baseline=0.018, subpixel=0.08):
    """D405-like noise on a clean depth image (0 = no reading)."""
    import cv2

    base = np.asarray(depth, dtype=np.float64)
    valid = base > 0
    sigma = base**2 * subpixel / (float(k[0, 0]) * baseline)
    white = rng.normal(0, 1, base.shape)
    corr = cv2.GaussianBlur(white, (0, 0), 1.0)
    corr /= corr.std() or 1.0
    noisy = base + sigma * (0.6 * corr + 0.8 * white)
    dz = np.maximum(
        np.abs(np.diff(base, axis=0, append=base[-1:])),
        np.abs(np.diff(base, axis=1, append=base[:, -1:])),
    )
    fly = (dz > 0.002) & (rng.random(base.shape) < 0.5)
    mix = rng.random(base.shape)
    nb = np.roll(base, 1, axis=1)
    noisy[fly] = (mix * base + (1 - mix) * nb)[fly]
    noisy[rng.random(base.shape) < 0.005] = 0.0
    return np.where(valid, np.round(noisy / 1e-4) * 1e-4, 0.0).astype(np.float32)
