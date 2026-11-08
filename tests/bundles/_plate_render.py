"""Ray-cast RGB-D renders of a LEGO / DUPLO baseplate for calibrating-baseplate-stud-grid.

Analytic geometry in the plate frame (origin at the centre of the plate's top,
z up): the plate as a box, a stud cylinder at every lattice node (bevelled top
edge), bricks seated on the grid with their own studs, the table under the plate
and ArUco / AprilTag markers printed on it. RGB is Lambertian with an ambient
term, 2x supersampled, then blurred and noised; depth is the z along the optical
axis with D405-like noise (sigma = z^2 * subpixel / (f * baseline), spatially
correlated), flying pixels at depth edges, dropouts and 0.1 mm quantisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

SYSTEMS = {
    "lego": {
        "pitch": 0.008,
        "stud_h": 0.0017,
        "stud_d": 0.0048,
        "brick_h": 0.0096,
        "plate_t": 0.0032,
    },
    "duplo": {
        "pitch": 0.016,
        "stud_h": 0.0045,
        "stud_d": 0.0094,
        "brick_h": 0.0192,
        "plate_t": 0.005,
    },
}
D405_K = np.array([[394.0, 0.0, 320.0], [0.0, 394.0, 240.0], [0.0, 0.0, 1.0]])
"""640 x 480 wrist D405 (the hub's refining-target-from-wrist-closeup: fx ~394 px)."""
ORBBEC_K = np.array([[461.2, 0.0, 320.0], [0.0, 461.5, 240.0], [0.0, 0.0, 1.0]])
"""640 x 480 overhead camera: the intrinsics stored with the station's frames on disk."""

PLATE_RGB = np.array([0.22, 0.55, 0.26])
TABLE_RGB = np.array([0.72, 0.58, 0.42])
BRICK_RGB = [
    np.array([0.75, 0.12, 0.10]),
    np.array([0.12, 0.25, 0.70]),
    np.array([0.85, 0.75, 0.15]),
]


@dataclass
class Brick:
    x: int
    y: int
    nx: int
    ny: int
    layer: int = 1
    colour: int = 0


@dataclass
class Marker:
    id: int
    size: float
    x: float
    y: float
    z: float
    yaw_deg: float = 0.0


@dataclass
class Scene:
    system: str = "lego"
    studs: tuple[int, int] = (32, 32)
    bricks: list[Brick] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    dictionary: str = "DICT_4X4_50"
    plate_t: float | None = None
    bare: bool = False
    light: tuple[float, float, float] = (0.45, -0.35, 1.0)

    @property
    def g(self):
        return SYSTEMS[self.system]


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """OpenCV camera-to-frame pose: z towards the target, x right, y down."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def _box(o, d, lo, hi):
    """Slab test: entry t (inf if missed) and the entry face's normal."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    tmin = np.where(np.isnan(tmin), -np.inf, tmin)
    tmax = np.where(np.isnan(tmax), np.inf, tmax)
    t_in = tmin.max(axis=1)
    t_out = tmax.min(axis=1)
    hit = (t_in <= t_out) & (t_in > 0)
    axis = tmin.argmax(axis=1)
    n = np.zeros_like(o)
    n[np.arange(len(o)), axis] = -np.sign(d[np.arange(len(o)), axis])
    return np.where(hit, t_in, np.inf), n


def _studs(o, d, z0, s, r, pitch, phase, valid):
    """Nearest stud hit (top disc or side) on lattice ((i, j) + phase) * pitch at
    base height z0; ``valid(i, j)`` says which nodes carry a stud."""
    best_t = np.full(len(o), np.inf)
    best_n = np.zeros_like(o)
    best_top = np.zeros(len(o), bool)
    rho_out = np.zeros(len(o))
    with np.errstate(divide="ignore", invalid="ignore"):
        cand = []
        for zz in (z0 + s, z0 + 0.5 * s, z0):
            t = (zz - o[:, 2]) / d[:, 2]
            p = o[:, :2] + t[:, None] * d[:, :2]
            cand.append(np.round(p / pitch - phase).astype(np.int64))
        seen = set()
        for idx in cand:
            key = idx.tobytes()
            if key in seen:
                continue
            seen.add(key)
            ok = valid(idx[:, 0], idx[:, 1])
            c = (idx + phase) * pitch
            # Top disc.
            t = (z0 + s - o[:, 2]) / d[:, 2]
            p = o[:, :2] + t[:, None] * d[:, :2] - c
            rho = np.linalg.norm(p, axis=1)
            top = ok & (t > 0) & (rho <= r)
            upd = top & (t < best_t)
            best_t[upd], best_top[upd], rho_out[upd] = t[upd], True, rho[upd]
            nn = np.zeros_like(o)
            bev = rho > 0.8 * r
            tilt = np.clip((rho - 0.8 * r) / (0.2 * r), 0, 1) * math.radians(50)
            radial = np.zeros_like(o)
            radial[:, :2] = p / np.maximum(rho, 1e-12)[:, None]
            nn[:, 2] = np.where(bev, np.cos(tilt), 1.0)
            nn[:, :2] = np.where(bev[:, None], radial[:, :2] * np.sin(tilt)[:, None], 0.0)
            best_n[upd] = nn[upd]
            # Side wall.
            oc = o[:, :2] - c
            a = np.sum(d[:, :2] ** 2, axis=1)
            b = 2 * np.sum(oc * d[:, :2], axis=1)
            cc = np.sum(oc**2, axis=1) - r * r
            disc = b * b - 4 * a * cc
            t1 = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
            z1 = o[:, 2] + t1 * d[:, 2]
            side = ok & (disc > 0) & (t1 > 0) & (z1 >= z0) & (z1 <= z0 + s)
            upd = side & (t1 < best_t)
            best_t[upd], best_top[upd] = t1[upd], False
            ps = oc + t1[:, None] * d[:, :2]
            ns = np.zeros_like(o)
            ns[:, :2] = ps / r
            best_n[upd] = ns[upd]
    return best_t, best_n, best_top


def render(
    scene: Scene,
    k: np.ndarray,
    t_plate_cam: np.ndarray,
    size: tuple[int, int] = (640, 480),
    rng: np.random.Generator | None = None,
    depth_noise: bool = True,
    supersample: int = 2,
    baseline: float = 0.018,
    subpixel: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """(rgb uint8 HxWx3, depth float32 HxW metres, 0 = no reading)."""
    rng = np.random.default_rng(0) if rng is None else rng
    w, h = size
    rgb_hi, _ = _trace(scene, k, t_plate_cam, w, h, supersample)
    rgb = rgb_hi.reshape(h, supersample, w, supersample, 3).mean(axis=(1, 3))
    _, depth = _trace(scene, k, t_plate_cam, w, h, 1)
    render.last_ids = _trace.last_ids  # surface per pixel: 0 table, 1 plate, 2 + i brick i
    rgb = cv2.GaussianBlur(rgb, (0, 0), 0.6)
    rgb = rgb * 255.0 + rng.normal(0, 2.5, rgb.shape)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if depth_noise:
        f = float(k[0, 0])
        sigma = depth**2 * subpixel / (f * baseline)
        white = rng.normal(0, 1, depth.shape)
        corr = cv2.GaussianBlur(white, (0, 0), 1.0)
        corr /= corr.std() or 1.0
        noisy = depth + sigma * (0.6 * corr + 0.8 * white)
        # Flying pixels where the depth jumps.
        dz = np.maximum(
            np.abs(np.diff(depth, axis=0, append=depth[-1:])),
            np.abs(np.diff(depth, axis=1, append=depth[:, -1:])),
        )
        fly = (dz > 0.002) & (rng.random(depth.shape) < 0.5)
        mix = rng.random(depth.shape)
        nb = np.roll(depth, 1, axis=1)
        noisy[fly] = (mix * depth + (1 - mix) * nb)[fly]
        noisy[rng.random(depth.shape) < 0.005] = 0.0
        depth = np.round(noisy / 1e-4) * 1e-4
    depth[~np.isfinite(depth)] = 0.0
    return rgb, depth.astype(np.float32)


def _trace(scene, k, t_plate_cam, w, h, ss):
    g = scene.g
    pitch, s, r, bh = g["pitch"], g["stud_h"], g["stud_d"] / 2, g["brick_h"]
    pt = scene.plate_t if scene.plate_t is not None else g["plate_t"]
    nx, ny = scene.studs
    hx, hy = nx * pitch / 2, ny * pitch / 2
    u, v = np.meshgrid((np.arange(w * ss) + 0.5) / ss - 0.5, (np.arange(h * ss) + 0.5) / ss - 0.5)
    dc = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1).reshape(
        -1, 3
    )
    rmat, o0 = t_plate_cam[:3, :3], t_plate_cam[:3, 3]
    d = dc @ rmat.T
    o = np.broadcast_to(o0, d.shape).copy()
    n_rays = len(d)
    t_best = np.full(n_rays, np.inf)
    normal = np.zeros((n_rays, 3))
    albedo = np.zeros((n_rays, 3))
    sid = np.zeros(n_rays, np.int32)

    def take(t, n, col, surface=0):
        upd = t < t_best
        t_best[upd] = t[upd]
        sid[upd] = surface
        normal[upd] = n[upd] if n.ndim == 2 else n
        albedo[upd] = col[upd] if np.ndim(col) == 2 else col

    # Table.
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = (-pt - o[:, 2]) / d[:, 2]
    tt = np.where(tt > 0, tt, np.inf)
    p = o + tt[:, None] * d
    col = np.broadcast_to(TABLE_RGB, (n_rays, 3)).copy()
    if scene.markers:
        dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, scene.dictionary))
        for mk in scene.markers:
            img = cv2.aruco.generateImageMarker(dic, mk.id, 240, borderBits=1).astype(float) / 255
            a = math.radians(mk.yaw_deg)
            lx = math.cos(a) * (p[:, 0] - mk.x) + math.sin(a) * (p[:, 1] - mk.y)
            ly = -math.sin(a) * (p[:, 0] - mk.x) + math.cos(a) * (p[:, 1] - mk.y)
            half = mk.size / 2
            quiet = (np.abs(lx) <= half * 1.4) & (np.abs(ly) <= half * 1.4)
            inside = (np.abs(lx) < half) & (np.abs(ly) < half)
            n_img = img.shape[0]
            cc = np.clip(((lx + half) / mk.size * n_img).astype(int), 0, n_img - 1)
            rr = np.clip(((half - ly) / mk.size * n_img).astype(int), 0, n_img - 1)
            val = np.where(inside, img[rr, cc], 1.0) * 0.92 + 0.04
            col[quiet] = val[quiet, None]
    take(tt, np.array([0.0, 0.0, 1.0]), col)
    # Plate body.
    tb, nb = _box(o, d, np.array([-hx, -hy, -pt]), np.array([hx, hy, 0.0]))
    take(tb, nb, PLATE_RGB, 1)
    covered = np.zeros((nx, ny), bool)
    for b in scene.bricks:
        covered[b.x : b.x + b.nx, b.y : b.y + b.ny] = True

    def plate_valid(i, j):
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        out = ok.copy()
        out[ok] = ~covered[i[ok], j[ok]]
        return out

    ph = np.array([0.5, 0.5])  # cell (i, j)'s stud sits at the corner + (i + 0.5) * pitch
    origin = np.array([-hx, -hy])
    if not scene.bare:
        ts, ns, _ = _studs(o - np.r_[origin, 0], d, 0.0, s, r, pitch, ph, plate_valid)
        take(ts, ns, PLATE_RGB, 1)
    for bi, b in enumerate(scene.bricks):
        play = 0.0001
        z0 = (b.layer - 1) * bh
        lo = np.array([origin[0] + b.x * pitch + play, origin[1] + b.y * pitch + play, z0])
        hi = np.array(
            [
                origin[0] + (b.x + b.nx) * pitch - play,
                origin[1] + (b.y + b.ny) * pitch - play,
                z0 + bh,
            ]
        )
        colr = BRICK_RGB[b.colour % len(BRICK_RGB)]
        tb, nb = _box(o, d, lo, hi)
        take(tb, nb, colr, 2 + bi)

        def bvalid(i, j, b=b):
            return (i >= b.x) & (i < b.x + b.nx) & (j >= b.y) & (j < b.y + b.ny)

        ts, ns, _ = _studs(o - np.r_[origin, 0], d, z0 + bh, s, r, pitch, ph, bvalid)
        take(ts, ns, colr, 2 + bi)
    light = np.asarray(scene.light, float)
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(normal @ light, 0, 1)
    rgb = (albedo * shade[:, None]).reshape(h * ss, w * ss, 3)
    depth = t_best.reshape(h * ss, w * ss)  # d has camera-z 1, so t is the depth
    _trace.last_ids = sid.reshape(h * ss, w * ss)
    return rgb, depth


def plate_pose_world(x, y, z, yaw_deg, tilt_deg=(0.0, 0.0)) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    m = np.eye(4)
    m[:3, :3] = (
        Rotation.from_euler("z", yaw_deg, degrees=True)
        * Rotation.from_euler("xy", tilt_deg, degrees=True)
    ).as_matrix()
    m[:3, 3] = (x, y, z)
    return m
