"""Ray-cast depth of a baseplate with one brick in any pose, for verifying-brick-seated-by-height.

The plate, its studs and any grid-seated neighbour bricks come from _plate_render's tracer
(plate frame: origin at the centre of the plate's top, z up). The brick under test is a box
with its studs in its own frame (origin at the centre of its bottom face), placed by a 4 x 4
pose in the plate frame, so it can sit on the stud tops, tilt or stand a cell off. Depth noise
follows _plate_render.render (D405: sigma = z^2 * subpixel / (f * baseline), correlated, flying
pixels at edges, dropouts, 0.1 mm quantisation).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ._plate_render import SYSTEMS, Brick, Scene, _box, _studs, _trace


def brick_pose(
    system: str,
    nx: int,
    ny: int,
    x: int,
    y: int,
    plate_studs: tuple[int, int],
    layer: int = 1,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    tilt_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Plate-frame pose of a brick's bottom centre: seated at cell (x, y), then moved by
    (dx, dy, dz), tilted about its y axis through its -x bottom edge and turned by yaw."""
    g = SYSTEMS[system]
    p = g["pitch"]
    cx = -plate_studs[0] * p / 2 + (x + nx / 2) * p
    cy = -plate_studs[1] * p / 2 + (y + ny / 2) * p
    cz = (layer - 1) * g["brick_h"]
    a = -math.radians(tilt_deg)  # the +x end up
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    b = math.radians(yaw_deg)
    rz = np.array([[math.cos(b), -math.sin(b), 0], [math.sin(b), math.cos(b), 0], [0, 0, 1]])
    r = rz @ ry
    hx = nx * p / 2
    # Tilt about the -x bottom edge: that edge stays where it was.
    edge = np.array([-hx, 0.0, 0.0])
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = np.array([cx + dx, cy + dy, cz + dz]) + edge - r @ edge
    return m


def _trace_brick(o, d, pose, system, nx, ny):
    g = SYSTEMS[system]
    p, s, rad, bh = g["pitch"], g["stud_h"], g["stud_d"] / 2, g["brick_h"]
    rinv = pose[:3, :3].T
    ob = (o - pose[:3, 3]) @ rinv.T
    db = d @ rinv.T
    play = 0.0001
    hx, hy = nx * p / 2, ny * p / 2
    t, _ = _box(
        ob, db, np.array([-hx + play, -hy + play, 0.0]), np.array([hx - play, hy - play, bh])
    )

    def valid(i, j):
        return (i >= 0) & (i < nx) & (j >= 0) & (j < ny)

    ts, _, _ = _studs(
        ob - np.array([-hx, -hy, 0.0]), db, bh, s, rad, p, np.array([0.5, 0.5]), valid
    )
    return np.minimum(t, ts)


def render_depth(
    system: str,
    plate_studs: tuple[int, int],
    k: np.ndarray,
    t_plate_cam: np.ndarray,
    brick: tuple[np.ndarray, int, int] | None,
    neighbours: list[Brick] | None = None,
    size: tuple[int, int] = (640, 480),
    rng: np.random.Generator | None = None,
    noise: bool = True,
    baseline: float = 0.018,
    subpixel: float = 0.08,
    boxes: list[tuple[tuple, tuple]] = (),
) -> np.ndarray:
    """Depth (float32 metres, 0 = no reading) along the optical axis; ``boxes`` are extra
    axis-aligned (lo, hi) boxes in the plate frame (a gripper finger in view)."""
    rng = np.random.default_rng(0) if rng is None else rng
    w, h = size
    scene = Scene(system=system, studs=plate_studs, bricks=list(neighbours or []))
    _, depth = _trace(scene, k, t_plate_cam, w, h, 1)
    if brick is not None:
        pose, nx, ny = brick
        u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
        dc = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1)
        dc = dc.reshape(-1, 3)
        d = dc @ t_plate_cam[:3, :3].T
        o = np.broadcast_to(t_plate_cam[:3, 3], d.shape).copy()
        tb = _trace_brick(o, d, pose, system, nx, ny).reshape(h, w)
        depth = np.minimum(depth, tb)
    if boxes:
        u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
        dc = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1)
        d = dc.reshape(-1, 3) @ t_plate_cam[:3, :3].T
        o = np.broadcast_to(t_plate_cam[:3, 3], d.shape).copy()
        for lo, hi in boxes:
            tb, _ = _box(o, d, np.asarray(lo, float), np.asarray(hi, float))
            depth = np.minimum(depth, tb.reshape(h, w))
    depth = np.where(np.isfinite(depth), depth, 0.0)
    if noise:
        depth = add_noise(depth, k, rng, baseline, subpixel)
    return depth.astype(np.float32)


def add_noise(
    depth: np.ndarray,
    k: np.ndarray,
    rng: np.random.Generator,
    baseline: float = 0.018,
    subpixel: float = 0.08,
) -> np.ndarray:
    """D405-like noise on a clean depth image (0 = no reading)."""
    base = np.asarray(depth, dtype=np.float64)
    valid = base > 0
    f = float(k[0, 0])
    sigma = base**2 * subpixel / (f * baseline)
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
    out = np.where(valid, np.round(noisy / 1e-4) * 1e-4, 0.0)
    return out.astype(np.float32)
