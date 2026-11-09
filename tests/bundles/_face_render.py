"""Ray-cast depth of loose LEGO / DUPLO bricks lying on a table, for
classifying-brick-resting-face-from-depth.

Each brick is hollow, as real ones are: a top plate, four walls, solid underside tubes at
the interior lattice nodes (pins for 1 x N bricks; square prisms of the same area) and studs on top, in its own frame
(origin at the centre of its bottom face, z up through the studs). ``resting_pose`` puts it
studs up, upside down (on its stud tops), or on a side wall, turned by a yaw, with its
lowest point on the table (z = 0 in the table frame). Depth noise is _seat-style D405 noise
(sigma = z^2 * subpixel / (f * baseline), correlated, flying pixels, dropouts, 0.1 mm).
Dimensions are nominal (LEGO: 1.2 mm walls, 1.0 mm top, 6.51 mm tubes, 3.2 mm pins;
DUPLO: twice those), not CAD.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ._plate_render import SYSTEMS, _box, _studs

WALLS = {
    "lego": {"wall": 0.0012, "top": 0.0010, "tube_d": 0.00651, "pin_d": 0.0032},
    "duplo": {"wall": 0.0024, "top": 0.0020, "tube_d": 0.0130, "pin_d": 0.0064},
}


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


FACES = {
    "studs_up": np.eye(3),
    "upside_down": _rx(math.pi),
    "side_x": _ry(-math.pi / 2),  # the brick's x axis up, studs toward world -x before the yaw
    "side_y": _rx(math.pi / 2),  # the brick's y axis up, studs toward world -y before the yaw
}


def _hull_points(system, nx, ny, bh):
    g = SYSTEMS[system]
    p, s, r = g["pitch"], g["stud_h"], g["stud_d"] / 2
    hx, hy = nx * p / 2, ny * p / 2
    pts = [np.array([sx * hx, sy * hy, z]) for sx in (-1, 1) for sy in (-1, 1) for z in (0, bh)]
    for i in range(nx):
        for j in range(ny):
            cx, cy = -hx + (i + 0.5) * p, -hy + (j + 0.5) * p
            for a in np.linspace(0, 2 * math.pi, 12, endpoint=False):
                pts.append(np.array([cx + r * math.cos(a), cy + r * math.sin(a), bh + s]))
    return np.array(pts)


def resting_pose(system, nx, ny, face, x=0.0, y=0.0, yaw_deg=0.0, bh=None, lift=0.0):
    """Table-frame pose of a brick lying on ``face`` with its footprint centre at (x, y)."""
    g = SYSTEMS[system]
    bh = g["brick_h"] if bh is None else bh
    r = _rz(math.radians(yaw_deg)) @ FACES[face]
    hull = _hull_points(system, nx, ny, bh) @ r.T
    centre = 0.5 * (hull.min(axis=0) + hull.max(axis=0))
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = np.array([x - centre[0], y - centre[1], -hull[:, 2].min() + lift])
    return m


def _trace_brick(o, d, pose, system, nx, ny, bh):
    g, w = SYSTEMS[system], WALLS[system]
    p, s, rs = g["pitch"], g["stud_h"], g["stud_d"] / 2
    rinv = pose[:3, :3].T
    ob = (o - pose[:3, 3]) @ rinv.T
    db = d @ rinv.T
    hx, hy = nx * p / 2 - 0.0001, ny * p / 2 - 0.0001
    wt, tt = w["wall"], w["top"]
    boxes = [
        ((-hx, -hy, bh - tt), (hx, hy, bh)),
        ((-hx, -hy, 0.0), (-hx + wt, hy, bh)),
        ((hx - wt, -hy, 0.0), (hx, hy, bh)),
        ((-hx, -hy, 0.0), (hx, -hy + wt, bh)),
        ((-hx, hy - wt, 0.0), (hx, hy, bh)),
    ]
    t = np.full(len(o), np.inf)
    for lo, hi in boxes:
        tb, _ = _box(ob, db, np.array(lo), np.array(hi))
        t = np.minimum(t, tb)
    corner = np.array([-nx * p / 2, -ny * p / 2, 0.0])

    def on_top(i, j):
        return (i >= 0) & (i < nx) & (j >= 0) & (j < ny)

    ts, _, _ = _studs(ob - corner, db, bh, s, rs, p, np.array([0.5, 0.5]), on_top)
    t = np.minimum(t, ts)
    under = bh - tt
    # Tubes (pins for 1 x N) as square prisms of the same area: a cylinder seen from its
    # open end needs a bottom disc the stud tracer does not have.
    posts = []
    if nx >= 2 and ny >= 2:
        side = w["tube_d"] * math.sqrt(math.pi) / 2
        posts = [(i * p, j * p) for i in range(1, nx) for j in range(1, ny)]
    elif max(nx, ny) >= 2:
        side = w["pin_d"] * math.sqrt(math.pi) / 2
        if ny == 1:
            posts = [(i * p, 0.5 * p) for i in range(1, nx)]
        else:
            posts = [(0.5 * p, j * p) for j in range(1, ny)]
    for px, py in posts:
        c = corner[:2] + np.array([px, py])
        lo = np.array([c[0] - side / 2, c[1] - side / 2, 0.0])
        hi = np.array([c[0] + side / 2, c[1] + side / 2, under])
        tb, _ = _box(ob, db, lo, hi)
        t = np.minimum(t, tb)
    return t


def render_depth(
    k: np.ndarray,
    t_table_cam: np.ndarray,
    bricks: list[tuple[str, int, int, np.ndarray, float | None]],
    size: tuple[int, int] = (640, 480),
    boxes: list[tuple[tuple, tuple]] = (),
) -> np.ndarray:
    """Clean depth (float32 metres, 0 = no reading) of the table, ``bricks``
    (system, nx, ny, pose, brick height or None) and axis-aligned ``boxes`` (lo, hi)."""
    w, h = size
    u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
    dc = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1)
    d = dc.reshape(-1, 3) @ t_table_cam[:3, :3].T
    o = np.broadcast_to(t_table_cam[:3, 3], d.shape).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -o[:, 2] / d[:, 2]
    t = np.where(t > 0, t, np.inf)
    for system, nx, ny, pose, bh in bricks:
        bh = SYSTEMS[system]["brick_h"] if bh is None else bh
        t = np.minimum(t, _trace_brick(o, d, pose, system, nx, ny, bh))
    for lo, hi in boxes:
        tb, _ = _box(o, d, np.asarray(lo, float), np.asarray(hi, float))
        t = np.minimum(t, tb)
    depth = t.reshape(h, w)
    return np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)


def add_noise(depth, k, rng, baseline=0.018, subpixel=0.08):
    """D405-like noise on a clean depth image (0 = no reading)."""
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
