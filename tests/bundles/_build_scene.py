"""Synthetic brick builds on a baseplate for verifying-build-against-model: a
random supported target, injected build errors, and ray-cast depth + RGB frames
of the result from the station's cameras with their depth noise.

Geometry: the plate frame has its origin at the centre of the plate's top (stud
bases), x / y along its stud rows, z up (computing-brick-pose-from-stud-grid's
plate frame). Bricks are boxes 0.2 mm short of n pitches (ICL's
brick_len_offset_) with cylinder studs on top; the plate has studs on every
cell. Noise is pose6d's station model (tools/pose6d/_synth.py): axial std
sigma_ref * (depth / z_ref)^2, then quantised - overhead Orbbec 2.9 mm at 1 m in
1 mm steps, wrist D405 0.5 mm at 0.15 m in 0.1 mm steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SYSTEMS = {
    "lego": {"pitch": 0.008, "body": 0.0096, "stud_h": 0.0017, "stud_d": 0.0048, "plate": 0.0032},
    "duplo": {"pitch": 0.016, "body": 0.0192, "stud_h": 0.0045, "stud_d": 0.0094, "plate": 0.0096},
}
ORBBEC = {"sigma_ref": 0.0029, "z_ref": 1.0, "quant": 0.001}
D405 = {"sigma_ref": 0.0005, "z_ref": 0.15, "quant": 0.0001}
K_ORBBEC = np.array([[461.2, 0.0, 320.0], [0.0, 461.5, 240.0], [0.0, 0.0, 1.0]])
K_D405 = np.array([[433.0, 0.0, 320.0], [0.0, 433.0, 240.0], [0.0, 0.0, 1.0]])
PLAY = 0.0001
SIZES = ["1x2", "1x4", "2x2", "2x3", "2x4", "1x1", "1x3", "2x6"]
COLOURS = ["red", "blue", "yellow", "white", "black", "green", "orange", "light_bluish_grey"]


@dataclass
class Brick:
    x: int
    y: int
    layer: int
    size: str
    rotation: int
    colour: str
    step: int
    lift: float = 0.0  # m, whole brick raised (resting on the stud tops)
    tilt_lift: float = 0.0  # m, one end raised (along the long side), the other seated

    @property
    def footprint(self) -> tuple[int, int]:
        a, b = (int(v) for v in self.size.split("x"))
        return (b, a) if self.rotation % 180 == 90 else (a, b)

    def cells(self) -> list[tuple[int, int, int]]:
        fx, fy = self.footprint
        return [(self.x + i, self.y + j, self.layer) for i in range(fx) for j in range(fy)]

    def as_target(self, index: int) -> dict:
        return {
            "id": index,
            "cell": {"x": self.x, "y": self.y, "layer": self.layer, "rotation": self.rotation},
            "size": self.size,
            "colour": self.colour,
            "step": self.step,
        }


@dataclass
class Scene:
    system: str
    plate: tuple[int, int]
    bricks: list[Brick] = field(default_factory=list)


def _free(bricks: list[Brick], b: Brick, plate: tuple[int, int]) -> bool:
    fx, fy = b.footprint
    if b.x < 0 or b.y < 0 or b.x + fx > plate[0] or b.y + fy > plate[1]:
        return False
    taken = {c for o in bricks for c in o.cells()}
    if any(c in taken for c in b.cells()):
        return False
    if b.layer > 1 and not any((x, y, b.layer - 1) in taken for x, y, _ in b.cells()):
        return False
    # Nothing above it (a top-down press could not reach it).
    above = {(x, y) for o in bricks if o.layer > b.layer for x, y, _ in o.cells()}
    return not any((x, y) in above for x, y, _ in b.cells())


def random_build(rng: np.random.Generator, system: str, plate=(12, 12), n=8, max_layer=3) -> Scene:
    s = Scene(system, plate)
    tries = 0
    while len(s.bricks) < n and tries < 2000:
        tries += 1
        size = SIZES[rng.integers(len(SIZES))]
        rot = int(rng.choice([0, 90]))
        layer = int(rng.integers(1, max_layer + 1))
        b = Brick(
            int(rng.integers(0, plate[0])),
            int(rng.integers(0, plate[1])),
            layer,
            size,
            rot,
            COLOURS[rng.integers(len(COLOURS))],
            len(s.bricks) + 1,
        )
        if _free(s.bricks, b, plate):
            s.bricks.append(b)
    return s


def tops(bricks: list[Brick]) -> list[int]:
    """Indices of bricks with nothing on them."""
    out = []
    for i, b in enumerate(bricks):
        cols = {(x, y) for x, y, _ in b.cells()}
        if not any(o.layer > b.layer and cols & {(x, y) for x, y, _ in o.cells()} for o in bricks):
            out.append(i)
    return out


def random_free_brick(rng, bricks, plate, size=None, colour=None, avoid=None) -> Brick | None:
    for _ in range(500):
        b = Brick(
            int(rng.integers(0, plate[0])),
            int(rng.integers(0, plate[1])),
            int(rng.integers(1, 3)),
            size or SIZES[rng.integers(len(SIZES))],
            int(rng.choice([0, 90])),
            colour or COLOURS[rng.integers(len(COLOURS))],
            99,
        )
        if avoid is not None and set(b.cells()) & avoid:
            continue
        if _free(bricks, b, plate):
            return b
    return None


# ------------------------------------------------------------------ rendering


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    eye, target, up = (np.asarray(a, dtype=float) for a in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, :3] = np.column_stack([x, y, z])
    t[:3, 3] = eye
    return t


def _box_t(o, d, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo[None] - o) * inv
        t2 = (hi[None] - o) * inv
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax >= tmin) & (tmax > 0)
    return np.where(hit, np.maximum(tmin, 0.0), np.inf)


def _stud_t(o, d, cx, cy, z0, r, h):
    """Rays (o [3], d [N,3]) against one vertical cylinder per ray (base centre
    cx, cy [N], z0), radius r, height h."""
    ox, oy, oz = o[0] - cx, o[1] - cy, o[2] - z0
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    a = dx**2 + dy**2
    b = 2 * (ox * dx + oy * dy)
    c = ox**2 + oy**2 - r * r
    disc = b * b - 4 * a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12))
        tc = (h - oz) / dz
    zs = oz + ts * dz
    t = np.where((disc >= 0) & (ts > 0) & (zs >= 0) & (zs <= h), ts, np.inf)
    px, py = ox + tc * dx, oy + tc * dy
    cap = (tc > 0) & (px**2 + py**2 <= r * r)
    return np.where(cap & (tc < t), tc, t)


def _grid_studs_t(o, d, x0, y0, z0, nx, ny, p, r, h):
    """Rays against a grid of nx x ny studs (first stud centre at x0 + p/2,
    y0 + p/2, bases at z0). Over a stud's height a ray moves well under a
    pitch sideways, so only the studs of the cells it is over at the stud's top
    and base can be hit."""
    best = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for zz in (z0, z0 + h):
            t = (zz - o[2]) / d[:, 2]
            i = np.floor((o[0] + t * d[:, 0] - x0) / p).astype(int)
            j = np.floor((o[1] + t * d[:, 1] - y0) / p).astype(int)
            ok = (t > 0) & (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i + di, j + dj
                    sel = ok & (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
                    if not sel.any():
                        continue
                    idx = np.nonzero(sel)[0]
                    tt = _stud_t(
                        o, d[idx], x0 + (ii[idx] + 0.5) * p, y0 + (jj[idx] + 0.5) * p, z0, r, h
                    )
                    best[idx] = np.minimum(best[idx], tt)
    return best


def brick_frame(b: Brick, sysd: dict, plate: tuple[int, int]) -> np.ndarray:
    """Plate-from-brick 4x4: origin at the brick's bottom centre, x along its
    footprint x; a tilt raises the far end of the long side by tilt_lift about
    the near bottom edge."""
    p = sysd["pitch"]
    fx, fy = b.footprint
    cx = -plate[0] * p / 2 + (b.x + fx / 2) * p
    cy = -plate[1] * p / 2 + (b.y + fy / 2) * p
    z0 = (b.layer - 1) * sysd["body"] + b.lift
    m = np.eye(4)
    m[:3, 3] = [cx, cy, z0]
    if b.tilt_lift:
        # Rotate about the bottom edge across the long side, raising the far end.
        along = 0 if fx >= fy else 1
        length = (fx if along == 0 else fy) * p - 2 * PLAY
        ang = math.asin(b.tilt_lift / length)
        c, s_ = math.cos(ang), math.sin(ang)
        if along == 0:
            r = np.array([[c, 0, -s_], [0, 1, 0], [s_, 0, c]])
            local_pivot = np.array([-length / 2, 0, 0])
        else:
            r = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
            local_pivot = np.array([0, -length / 2, 0])
        pivot = np.array([cx, cy, z0]) + local_pivot
        m[:3, :3] = r
        m[:3, 3] = pivot - r @ local_pivot
    return m


def render(
    scene: Scene,
    t_world_plate: np.ndarray,
    t_world_cam: np.ndarray,
    k: np.ndarray,
    noise: dict | None,
    seed: int = 0,
    hw=(480, 640),
    rgb_table: dict | None = None,
):
    """Depth [m] (0 = no return) and an RGB frame of the scene."""
    sysd = SYSTEMS[scene.system]
    p, sh, r = sysd["pitch"], sysd["stud_h"], sysd["stud_d"] / 2
    nx, ny = scene.plate
    h_img, w_img = hw
    v, u = np.mgrid[0:h_img, 0:w_img]
    rays_c = np.column_stack(
        [(u.ravel() - k[0, 2]) / k[0, 0], (v.ravel() - k[1, 2]) / k[1, 1], np.ones(h_img * w_img)]
    )
    t_pc = np.linalg.inv(t_world_plate) @ t_world_cam
    o = t_pc[:3, 3]
    d = rays_c @ t_pc[:3, :3].T
    # Only rays that pass near the plate's footprint can hit the build.
    with np.errstate(divide="ignore", invalid="ignore"):
        tz = (0.0 - o[2]) / d[:, 2]
    hx = o[0] + tz * d[:, 0]
    hy = o[1] + tz * d[:, 1]
    margin = 0.08
    near = (np.abs(hx) < nx * p / 2 + margin) & (np.abs(hy) < ny * p / 2 + margin) & (tz > 0)
    idx = np.nonzero(near)[0]
    dd = d[idx]
    t_best = np.full(len(idx), np.inf)
    label = np.full(len(idx), -2)  # -2 table, -1 plate, i brick
    thick = sysd["plate"]

    def take(t, lab):
        nonlocal t_best
        upd = t < t_best
        t_best = np.where(upd, t, t_best)
        label[upd] = lab

    take(
        _box_t(
            o,
            dd,
            np.array([-nx * p / 2, -ny * p / 2, -thick]),
            np.array([nx * p / 2, ny * p / 2, 0.0]),
        ),
        -1,
    )
    take(_grid_studs_t(o, dd, -nx * p / 2, -ny * p / 2, 0.0, nx, ny, p, r, sh), -1)
    for bi, b in enumerate(scene.bricks):
        m = brick_frame(b, sysd, scene.plate)
        inv = np.linalg.inv(m)
        ob = inv[:3, :3] @ o + inv[:3, 3]
        db = dd @ inv[:3, :3].T
        fx, fy = b.footprint
        lx, ly = fx * p - 2 * PLAY, fy * p - 2 * PLAY
        take(
            _box_t(
                ob, db, np.array([-lx / 2, -ly / 2, 0.0]), np.array([lx / 2, ly / 2, sysd["body"]])
            ),
            bi,
        )
        take(_grid_studs_t(ob, db, -fx * p / 2, -fy * p / 2, sysd["body"], fx, fy, p, r, sh), bi)
    t_all = np.full(h_img * w_img, np.inf)
    lab_all = np.full(h_img * w_img, -2)
    t_all[idx] = t_best
    lab_all[idx] = label
    # The table under everything else.
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = (-thick - o[2]) / d[:, 2]
    table = np.where(tt > 0, tt, np.inf)
    use_table = table < t_all
    t_all = np.where(use_table, table, t_all)
    lab_all = np.where(use_table, -2, lab_all)
    depth = np.where(np.isfinite(t_all), t_all, 0.0)
    rng = np.random.default_rng(seed)
    if noise is not None:
        sig = noise["sigma_ref"] * (depth / noise["z_ref"]) ** 2
        depth = np.where(depth > 0, depth + rng.normal(size=depth.shape) * sig, 0.0)
        depth = np.where(depth > 0, np.round(depth / noise["quant"]) * noise["quant"], 0.0)
    rgb = np.zeros((h_img * w_img, 3))
    if rgb_table is not None:
        rgb[:] = rgb_table.get("tan", (220, 200, 160))
        rgb[lab_all == -1] = rgb_table.get("green", (0, 133, 43))
        for bi, b in enumerate(scene.bricks):
            rgb[lab_all == bi] = rgb_table[b.colour]
        shade = rng.uniform(0.9, 1.0, size=(h_img * w_img, 1))
        rgb = np.clip(rgb * shade + rng.normal(scale=2.0, size=rgb.shape), 0, 255)
    return depth.reshape(h_img, w_img), rgb.reshape(h_img, w_img, 3).astype(np.uint8)


def cameras(rng: np.random.Generator, system: str, plate=(12, 12)):
    """(name, T_world_cam, K, noise) for the overhead camera at ~0.9 m and the
    wrist D405 over the build, both slightly oblique, plate at the origin."""
    p = SYSTEMS[system]["pitch"]
    size = max(plate) * p
    a = rng.uniform(0, 2 * math.pi)
    over = look_at(
        [0.25 * math.cos(a), 0.25 * math.sin(a), 0.86],
        [rng.normal(0, 0.01), rng.normal(0, 0.01), 0.0],
    )
    wrist_range = max(0.16, size * 1.25)  # the plate fills the D405's view
    b = rng.uniform(0, 2 * math.pi)
    tilt = math.radians(rng.uniform(5, 15))
    wrist = look_at(
        [
            wrist_range * math.sin(tilt) * math.cos(b),
            wrist_range * math.sin(tilt) * math.sin(b),
            wrist_range * math.cos(tilt),
        ],
        [0.0, 0.0, 0.0],
    )
    return [("overhead", over, K_ORBBEC, ORBBEC), ("wrist", wrist, K_D405, D405)]
