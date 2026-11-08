# SPDX-License-Identifier: MIT
"""Ray-cast depth renders of one box or cylinder on a table, with a depth-noise
model, for the bundle's tests and accuracy checks.

The noise model is the station's measured one: axial std ``sigma_ref`` at depth
``z_ref`` growing with depth squared, then quantised (overhead Orbbec Gemini:
2.9 mm at 1 m, 1 mm steps; wrist RealSense D405: 0.5 mm at 0.15 m, 0.1 mm steps).
"""

from __future__ import annotations

import numpy as np

ORBBEC = {"sigma_ref": 0.0029, "z_ref": 1.0, "quant": 0.001}
D405 = {"sigma_ref": 0.0005, "z_ref": 0.15, "quant": 0.0001}
#: The station's overhead intrinsics (Orbbec Gemini, 640x480).
K_ORBBEC = np.array([[461.2, 0.0, 320.0], [0.0, 461.5, 240.0], [0.0, 0.0, 1.0]])
#: RealSense D405 colour-aligned intrinsics at 640x480 (fx ~ 433 px for the 87 deg HFOV).
K_D405 = np.array([[433.0, 0.0, 320.0], [0.0, 433.0, 240.0], [0.0, 0.0, 1.0]])


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world 4x4 (OpenCV axes: x right, y down, z forward)."""
    eye, target, up = (np.asarray(a, dtype=np.float64) for a in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = eye
    return T


def _ray_box(o, d, h):
    """Entry distance of rays (object frame) into an axis-aligned box of half-extents h."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (-h[None, :] - o[None, :]) * inv
        t2 = (h[None, :] - o[None, :]) * inv
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax >= tmin) & (tmax > 0)
    return np.where(hit, np.maximum(tmin, 0.0), np.inf)


def _ray_cylinder(o, d, r, hz):
    t = np.full(len(d), np.inf)
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (o[0] * d[:, 0] + o[1] * d[:, 1])
    c = o[0] ** 2 + o[1] ** 2 - r**2
    disc = b**2 - 4 * a * c
    ok = (disc >= 0) & (a > 1e-12)
    ts = np.where(ok, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12)), np.inf)
    z = o[2] + ts * d[:, 2]
    side = ok & (ts > 0) & (np.abs(z) <= hz)
    t[side] = ts[side]
    for s in (1.0, -1.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            tc = (s * hz - o[2]) / d[:, 2]
        xy = o[None, :2] + tc[:, None] * d[:, :2]
        cap = (tc > 0) & (np.sum(xy**2, axis=1) <= r**2)
        t = np.where(cap & (tc < t), tc, t)
    return t


def render(
    shape: dict,
    T_world_obj: np.ndarray,
    T_world_cam: np.ndarray,
    K: np.ndarray = K_ORBBEC,
    hw: tuple[int, int] = (480, 640),
    noise: dict | None = None,
    table_z: float | None = 0.0,
    seed: int = 0,
):
    """Depth (metres, 0 = no return) and the object's pixel mask.

    ``shape`` is ``{"box": [x, y, z]}``, ``{"cube": s}`` or ``{"cylinder": [radius, height]}``.
    """
    if "cube" in shape:
        shape = {"box": [shape["cube"]] * 3}
    H, W = hw
    v, u = np.mgrid[0:H, 0:W]
    rays_c = np.column_stack(
        [(u.ravel() - K[0, 2]) / K[0, 0], (v.ravel() - K[1, 2]) / K[1, 1], np.ones(H * W)]
    )
    Rwc, twc = T_world_cam[:3, :3], T_world_cam[:3, 3]
    T_obj_world = np.linalg.inv(T_world_obj)
    o_obj = T_obj_world[:3, :3] @ twc + T_obj_world[:3, 3]
    d_obj = rays_c @ (T_obj_world[:3, :3] @ Rwc).T
    if "box" in shape:
        t_obj = _ray_box(o_obj, d_obj, np.asarray(shape["box"], dtype=np.float64) / 2)
    else:
        r, h = shape["cylinder"]
        t_obj = _ray_cylinder(o_obj, d_obj, float(r), float(h) / 2)
    t_tab = np.full(H * W, np.inf)
    if table_z is not None:
        d_w = rays_c @ Rwc.T
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (table_z - twc[2]) / d_w[:, 2]
        t_tab = np.where(tt > 0, tt, np.inf)
    t = np.minimum(t_obj, t_tab)
    mask = (t_obj <= t_tab) & np.isfinite(t_obj)
    depth = np.where(np.isfinite(t), t, 0.0)  # rays_c has z = 1, so t is the depth
    if noise is not None:
        rng = np.random.default_rng(seed)
        sig = noise["sigma_ref"] * (depth / noise["z_ref"]) ** 2
        depth = np.where(depth > 0, depth + rng.normal(size=depth.shape) * sig, 0.0)
        depth = np.where(depth > 0, np.round(depth / noise["quant"]) * noise["quant"], 0.0)
    return depth.reshape(H, W), mask.reshape(H, W)


def backproject(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame points of the masked pixels with a depth return."""
    v, u = np.nonzero((mask > 0) & (depth > 0))
    z = depth[v, u]
    return np.column_stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
