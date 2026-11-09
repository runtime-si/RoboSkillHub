"""Synthetic top-down depth frames of flat parts packed on a table, for grasp-proposal tests.

Parts are vertical prisms (jigsaw pieces from ``jigsaw_cut``, bricks, dominoes, any polygon)
on a table at z = 0, seen by a pinhole camera looking straight down from ``cam_height`` with
D405-like intrinsics. Depth is ray-cast: a pixel sees a part's top, its wall (the crossing
found by bisection) or the table. Returns the CameraFrame, a per-pixel label image (0 table,
i + 1 part i) and the parts.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .jigsaw_cut import make_puzzle

K_D405 = np.array([[390.0, 0.0, 320.0], [0.0, 390.0, 240.0], [0.0, 0.0, 1.0]])
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}  # camera z = world -z, x = x, y = -y


@dataclass
class Scene:
    frame: dict
    labels: np.ndarray
    parts: list[np.ndarray]  # world outlines (N x 2), counter-clockwise
    heights: list[float]

    def mask(self, i: int) -> np.ndarray:
        return ((self.labels == i + 1) * 255).astype(np.uint8)

    def masks(self) -> list[np.ndarray]:
        return [self.mask(i) for i in range(len(self.parts))]


def jigsaw_cluster(cols=4, rows=4, gap=0.0, cell=0.02, seed=0, style="classic"):
    """Outlines of a cut ``cols x rows`` puzzle pulled apart by ``gap`` between neighbours,
    centred on the origin."""
    pz = make_puzzle(cols, rows, cell=(cell, cell), seed=seed, style=style)
    out = []
    for i in range(cols):
        for j in range(rows):
            out.append(pz.outline(i, j) + np.array([i * gap, j * gap]))
    c = np.vstack(out).mean(axis=0)
    return [o - c for o in out]


def rect(cx, cy, w, h, yaw_deg=0.0):
    a = np.radians(yaw_deg)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    p = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return p @ r.T + [cx, cy]


def brick_grid(cols, rows, gap, size=(0.0158, 0.0318)):
    w, h = size
    out = []
    for i in range(cols):
        for j in range(rows):
            out.append(rect(i * (w + gap), j * (h + gap), w, h))
    c = np.vstack(out).mean(axis=0)
    return [o - c for o in out]


def render(
    parts,
    heights,
    cam_height=0.18,
    cam_xy=(0.0, 0.0),
    k=K_D405,
    size=(480, 640),
    noise=0.0,
    holes=0.0,
    seed=0,
    res=1e-4,
):
    rng = np.random.default_rng(seed)
    heights = list(np.broadcast_to(np.asarray(heights, dtype=float), (len(parts),)))
    allp = np.vstack(parts) if parts else np.zeros((1, 2))
    lo = allp.min(axis=0) - 0.002
    hi = allp.max(axis=0) + 0.002
    n = np.ceil((hi - lo) / res).astype(int) + 1
    lab = np.zeros((n[1], n[0]), np.int32)
    top = np.zeros((n[1], n[0]))
    for i, (p, h) in enumerate(zip(parts, heights, strict=True)):
        q = np.round((p - lo) / res * 16).astype(np.int32)
        m = np.zeros_like(lab, np.uint8)
        cv2.fillPoly(m, [q], 1, lineType=cv2.LINE_8, shift=4)
        lab[m > 0] = i + 1
        top[m > 0] = h

    def look(x, y):
        ix = np.floor((x - lo[0]) / res).astype(int)
        iy = np.floor((y - lo[1]) / res).astype(int)
        ok = (ix >= 0) & (ix < n[0]) & (iy >= 0) & (iy < n[1])
        ixc, iyc = np.clip(ix, 0, n[0] - 1), np.clip(iy, 0, n[1] - 1)
        return np.where(ok, top[iyc, ixc], 0.0), np.where(ok, lab[iyc, ixc], 0)

    hh, ww = size
    vv, uu = np.mgrid[0:hh, 0:ww].astype(float)
    rx = (uu - k[0, 2]) / k[0, 0]
    ry = -(vv - k[1, 2]) / k[1, 1]

    def at(z):  # where the pixel's ray is at height z
        d = cam_height - z
        return cam_xy[0] + rx * d, cam_xy[1] + ry * d

    z_hit = np.zeros((hh, ww))
    labels = np.zeros((hh, ww), np.int32)
    for h in sorted(set(heights), reverse=True):
        x, y = at(h)
        th, tl = look(x, y)
        hit = (labels == 0) & (th >= h - 1e-9) & (tl > 0)
        z_hit[hit] = h
        labels[hit] = tl[hit]
        # a wall: the ray is inside a part of this height somewhere below h
        x0, y0 = at(0.0)
        bh, bl = look(x0, y0)
        wall = (labels == 0) & (bh >= h - 1e-9) & (bl > 0)
        a = np.zeros(wall.sum())
        b = np.full(wall.sum(), h)
        ru, rv = uu[wall], vv[wall]
        for _ in range(12):
            mid = 0.5 * (a + b)
            d = cam_height - mid
            xm = cam_xy[0] + (ru - k[0, 2]) / k[0, 0] * d
            ym = cam_xy[1] - (rv - k[1, 2]) / k[1, 1] * d
            mh, _ = look(xm, ym)
            inside = mh >= h - 1e-9
            a = np.where(inside, mid, a)
            b = np.where(inside, b, mid)
        z_hit[wall] = a
        labels[wall] = bl[wall]
    depth = cam_height - z_hit
    if noise:
        depth = depth + rng.normal(0.0, noise, depth.shape)
    if holes:
        rng_d = cv2.dilate(depth, np.ones((3, 3))) - cv2.erode(depth, np.ones((3, 3)))
        drop = (rng_d > 0.001) & (rng.random(depth.shape) < holes)
        depth[drop] = 0.0
    frame = {
        "name": "wrist",
        "depth": depth.astype(np.float32),
        "rgb": np.zeros((hh, ww, 3), np.uint8),
        "intrinsics": k.copy(),
        "pose": {
            "position": {"x": float(cam_xy[0]), "y": float(cam_xy[1]), "z": float(cam_height)},
            "rotation": DOWN,
        },
    }
    return Scene(frame, labels, list(parts), heights)
