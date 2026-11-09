"""A synthetic overhead depth frame of standing, leaning, fallen or missing dominoes (for the audit tests and its
evaluation). Each tile is an oriented box splatted into a 1 mm heightmap; the camera looks straight down."""

from __future__ import annotations

import math

import numpy as np

H, W, T = 0.048, 0.024, 0.0075


def tile_points(
    x: float,
    y: float,
    facing_deg: float,
    lean_deg: float = 0.0,
    shift: float = 0.0,
    lateral: float = 0.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Surface samples of a tile at slot (x, y, facing), its base moved ``shift`` along the facing and ``lateral``
    across, turned ``yaw_deg``, leaning ``lean_deg`` (+ forward about its front bottom edge, - backward about its
    back bottom edge)."""
    f = math.radians(facing_deg + yaw_deg)
    u = np.array([math.cos(f), math.sin(f), 0.0])
    lat = np.array([-u[1], u[0], 0.0])
    fu = np.array([math.cos(math.radians(facing_deg)), math.sin(math.radians(facing_deg)), 0.0])
    flat = np.array([-fu[1], fu[0], 0.0])
    a = math.radians(abs(lean_deg))
    sx = np.linspace(-T, 0.0, 16)
    sy = np.linspace(-W / 2, W / 2, 49)
    sz = np.linspace(0.0, H, 97)
    g = (
        np.array(np.meshgrid(sx, sy, sz, indexing="ij")).reshape(3, -1).T
    )  # x from the front edge back, y, z
    if lean_deg < 0:
        g[:, 0] = -g[:, 0] - T  # pivot on the back edge: mirror the box about it
        xr = g[:, 0] * math.cos(a) - g[:, 2] * math.sin(a)
    else:
        xr = g[:, 0] * math.cos(a) + g[:, 2] * math.sin(a)
    zr = (-g[:, 0] if lean_deg >= 0 else g[:, 0]) * math.sin(a) + g[:, 2] * math.cos(a)
    if lean_deg < 0:
        pivot = np.array([x, y, 0.0]) + fu * shift + flat * lateral - u * T / 2
    else:
        pivot = np.array([x, y, 0.0]) + fu * shift + flat * lateral + u * T / 2
    return pivot + np.outer(xr, u) + np.outer(g[:, 1], lat) + np.outer(zr, [0, 0, 1.0])


def depth_frame(
    tiles: list[np.ndarray],
    centre=(0.0, 0.3),
    cam_z=3.0,
    f=3000.0,
    size=(640, 480),
    noise=0.0,
    seed=0,
    table_z=0.0,
):
    """An overhead camera's depth image (and its intrinsics and pose) of the tiles' surface samples."""
    w, h = size
    cx, cy = w / 2, h / 2
    res = 0.001
    pts = np.concatenate(tiles) if tiles else np.zeros((0, 3))
    hx0, hy0 = centre[0] - 0.25, centre[1] - 0.25
    hm = np.full((500, 500), table_z)
    if len(pts):
        ix = ((pts[:, 0] - hx0) / res).astype(int)
        iy = ((pts[:, 1] - hy0) / res).astype(int)
        ok = (ix >= 0) & (ix < 500) & (iy >= 0) & (iy < 500)
        np.maximum.at(hm, (iy[ok], ix[ok]), table_z + pts[ok, 2])
    uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    depth = np.full((h, w), cam_z - table_z)
    for _ in range(3):
        x = centre[0] + (uu - cx) / f * depth
        y = centre[1] - (vv - cy) / f * depth
        ix = np.clip(((x - hx0) / res).astype(int), 0, 499)
        iy = np.clip(((y - hy0) / res).astype(int), 0, 499)
        depth = cam_z - hm[iy, ix]
    if noise > 0:
        depth = depth + np.random.default_rng(seed).normal(0.0, noise, depth.shape)
    pose = np.eye(4)
    pose[:3, :3] = [[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]
    pose[:3, 3] = [centre[0], centre[1], cam_z]
    return {
        "name": "top",
        "depth": depth,
        "intrinsics": [[f, 0, cx], [0, f, cy], [0, 0, 1]],
        "pose": pose,
    }


def line_slots(
    n: int, pitch: float, x0: float = 0.0, y0: float = 0.2, facing_deg: float = 90.0
) -> list[list[float]]:
    a = math.radians(facing_deg)
    return [
        [x0 + k * pitch * math.cos(a), y0 + k * pitch * math.sin(a), facing_deg] for k in range(n)
    ]


def contact_deg(pitch: float) -> float:
    """The lean at which a tile's top meets the next tile on a straight line."""
    return math.degrees(math.asin(min(1.0, (pitch - T) / H)))
