"""Synthetic RGB-D snapshots of a work surface with placed items, for snapshot-diff checks.

A pinhole camera (the station's overhead Orbbec, about 1 m from the board and tilted 31
degrees, K from the real camera; or the wrist D405 25 cm straight down) ray-casts a scene of
oriented boxes on a textured table: jigsaw pieces (25 mm, 2 mm thick, cut from a procedural
picture, sparse partial assembly), 2x4 / 2x2 bricks on a baseplate, or standing dominoes in a
line. A disturbance moves, turns, topples or removes items, or adds an object, a hand or the
robot's arm; the lighting can drift (gain, tint, gradient, soft shadow) and the camera can be
bumped. Depth is the camera z of the hit with range-dependent noise and holes.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

OVERHEAD_K = np.array([[461.2, 0.0, 320.0], [0.0, 461.5, 240.0], [0.0, 0.0, 1.0]])
WRIST_K = np.array([[394.4, 0.0, 321.9], [0.0, 393.9, 246.6], [0.0, 0.0, 1.0]])
HW = (480, 640)
TABLE_TEXEL = 0.0015  # m per table-texture pixel
TABLE_HALF = (0.6, 0.6)


def _rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * k @ k


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """Camera-to-world matrix, OpenCV axes (x right, y down, z forward)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, :3] = np.stack([x, y, z], 1)
    t[:3, 3] = eye
    return t


def camera_pose(kind):
    if kind == "overhead":
        # 0.86 m up, 0.52 m back: 1.0 m to the board, optical axis 31 deg off vertical (station).
        return look_at([0.0, -0.52, 0.86], [0.0, 0.0, 0.0], up=(0.0, 1.0, 0.0)), OVERHEAD_K
    return look_at([0.0, 0.0, 0.25], [0.0, 0.0, 0.0], up=(0.0, 1.0, 0.0)), WRIST_K


def se3(t):
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(t[:3, :3]).as_quat()  # xyzw
    return {
        "position": dict(zip("xyz", map(float, t[:3, 3]), strict=True)),
        "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
    }


@dataclass
class Box:
    id: str
    center: np.ndarray  # world, m
    half: np.ndarray  # half extents along local axes
    rot: np.ndarray  # 3x3 local-to-world
    top: np.ndarray | None = None  # texture of the local +z face (u along x, v along y)
    side: np.ndarray | None = None  # texture of the local +-y faces (u along x, v along z)
    colour: tuple = (200, 200, 200)
    item: bool = True

    def obb(self):
        from scipy.spatial.transform import Rotation

        q = Rotation.from_matrix(self.rot).as_quat()
        return {
            "center": dict(zip("xyz", map(float, self.center), strict=True)),
            "extent": dict(zip("xyz", map(float, self.half), strict=True)),
            "orientation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
        }


@dataclass
class Scene:
    kind: str
    boxes: list[Box] = field(default_factory=list)
    table: np.ndarray | None = None
    arms: list[dict] = field(default_factory=list)
    arm_bases: list | None = None

    def items(self):
        return [b for b in self.boxes if b.item]

    def get(self, iid):
        return next(b for b in self.boxes if b.id == iid)


def _smooth_noise(rng, h, w, scale, ch=3):
    small = rng.random((max(2, h // scale), max(2, w // scale), ch)).astype(np.float32)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return big.reshape(h, w, ch)


def _table(rng):
    h, w = int(2 * TABLE_HALF[1] / TABLE_TEXEL), int(2 * TABLE_HALF[0] / TABLE_TEXEL)
    yy = np.arange(h, dtype=np.float32)[:, None]
    grain = 0.5 + 0.5 * np.sin(yy / 14.0 + 3 * _smooth_noise(rng, h, w, 120, 1)[:, :, 0])
    base = np.array([150, 112, 78], np.float32)
    img = base * (0.82 + 0.18 * grain[:, :, None]) * (0.9 + 0.2 * _smooth_noise(rng, h, w, 40))
    img += rng.normal(0, 4, img.shape).astype(np.float32)
    # Static clutter round the work area: tape, a sheet with print, screws - registration features.
    for _ in range(120):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        if abs(cx - w / 2) < 90 and abs(cy - h / 2) < 80:
            continue
        col = rng.integers(20, 240, 3).astype(np.float32)
        if rng.random() < 0.5:
            cv2.circle(img, (int(cx), int(cy)), int(rng.integers(3, 12)), col.tolist(), -1)
        else:
            sz = rng.integers(8, 40, 2)
            cv2.rectangle(
                img, (int(cx), int(cy)), (int(cx + sz[0]), int(cy + sz[1])), col.tolist(), -1
            )
    return np.clip(img, 0, 255).astype(np.uint8)


def _picture(rng, w, h):
    img = 255 * _smooth_noise(rng, h, w, 18)
    for _ in range(int(w * h / 900)):
        c = rng.integers(0, 255, 3).tolist()
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        if rng.random() < 0.5:
            cv2.circle(img, p, int(rng.integers(3, 14)), c, -1)
        else:
            cv2.line(
                img,
                p,
                (int(rng.integers(0, w)), int(rng.integers(0, h))),
                c,
                int(rng.integers(1, 4)),
            )
    return np.clip(img, 0, 255).astype(np.uint8)


def build(kind, seed=0, camera="overhead"):
    rng = np.random.default_rng(seed)
    sc = Scene(kind=kind, table=_table(rng))
    up = np.eye(3)
    if kind == "jigsaw":
        cell, thick, cols, rows = 0.025, 0.002, 6, 5
        tex_px = 40
        pic = _picture(rng, cols * tex_px, rows * tex_px)
        x0, y0 = -cols * cell / 2, -rows * cell / 2
        # A partial assembly: a corner block plus a row, so a moved piece leaves a visible gap.
        cells = [
            (c, r)
            for r in range(rows)
            for c in range(cols)
            if (c < 4 and r < 3) or (r == 4 and c < 3)
        ]
        for c, r in cells:
            t = pic[r * tex_px : (r + 1) * tex_px, c * tex_px : (c + 1) * tex_px].copy()
            cv2.rectangle(t, (0, 0), (tex_px - 1, tex_px - 1), (40, 40, 40), 1)
            sc.boxes.append(
                Box(
                    f"c{c}_{r}",
                    np.array([x0 + (c + 0.5) * cell, y0 + (r + 0.5) * cell, thick / 2]),
                    np.array([cell / 2 - 0.0002, cell / 2 - 0.0002, thick / 2]),
                    up.copy(),
                    top=t,
                    colour=(150, 140, 130),
                )
            )
        sc.grid = {"cols": cols, "rows": rows, "cell": cell, "x0": x0, "y0": y0, "thick": thick}
    elif kind == "lego":
        pitch, plate_h, brick_h = 0.008, 0.0032, 0.0096
        n = 20
        ptex = np.full((n * 16, n * 16, 3), (40, 140, 60), np.uint8)
        for i in range(n):
            for j in range(n):
                cv2.circle(ptex, (j * 16 + 8, i * 16 + 8), 5, (70, 175, 90), -1)
                cv2.circle(ptex, (j * 16 + 8, i * 16 + 8), 5, (25, 100, 40), 1)
        sc.boxes.append(
            Box(
                "plate",
                np.array([0.0, 0.0, plate_h / 2]),
                np.array([n * pitch / 2, n * pitch / 2, plate_h / 2]),
                up.copy(),
                top=ptex,
                item=False,
            )
        )
        colours = [
            (200, 30, 30),
            (240, 200, 20),
            (30, 80, 200),
            (240, 240, 240),
            (20, 20, 20),
            (240, 120, 20),
            (120, 30, 150),
            (60, 170, 70),
        ]
        spots = [
            (-7, -6, 4, 2),
            (-1, -6, 2, 2),
            (4, -6, 4, 2),
            (-7, -1, 2, 4),
            (-3, 0, 4, 2),
            (3, 1, 2, 2),
            (-6, 5, 4, 2),
            (2, 5, 4, 2),
        ]
        for i, (sx, sy, lx, ly) in enumerate(spots):
            col = colours[i % len(colours)]
            top = np.full((ly * 16, lx * 16, 3), col, np.uint8)
            lighter = tuple(int(min(255, v * 1.15 + 20)) for v in col)
            darker = tuple(int(v * 0.6) for v in col)
            for a in range(lx):
                for b in range(ly):
                    cv2.circle(top, (a * 16 + 8, b * 16 + 8), 5, lighter, -1)
                    cv2.circle(top, (a * 16 + 8, b * 16 + 8), 5, darker, 1)
            cv2.rectangle(top, (0, 0), (lx * 16 - 1, ly * 16 - 1), darker, 1)
            c = np.array([(sx + lx / 2) * pitch, (sy + ly / 2) * pitch, plate_h + brick_h / 2])
            sc.boxes.append(
                Box(
                    f"b{i}",
                    c,
                    np.array([lx * pitch / 2 - 1e-4, ly * pitch / 2 - 1e-4, brick_h / 2]),
                    up.copy(),
                    top=top,
                    colour=col,
                )
            )
    elif kind == "domino":
        thick, width, height = 0.0075, 0.024, 0.048
        for i in range(8):
            x = -0.07 + i * 0.02
            top = np.full((16, 48, 3), (235, 232, 222), np.uint8)
            side = np.full((96, 48, 3), (235, 232, 222), np.uint8)
            cv2.line(side, (0, 48), (47, 48), (60, 60, 60), 1)
            for k in range(int(rng.integers(1, 6))):
                cv2.circle(
                    side, (int(10 + 14 * (k % 3)), int(10 + 14 * (k // 3))), 3, (20, 20, 20), -1
                )
            for k in range(int(rng.integers(1, 6))):
                cv2.circle(
                    side, (int(10 + 14 * (k % 3)), int(60 + 14 * (k // 3))), 3, (20, 20, 20), -1
                )
            # Local x across the line (width), y = thickness (faces the camera), z up.
            rot = _rot([0, 0, 1], math.radians(90))
            sc.boxes.append(
                Box(
                    f"d{i}",
                    np.array([x, 0.0, height / 2]),
                    np.array([width / 2, thick / 2, height / 2]),
                    rot,
                    top=top,
                    side=side,
                    colour=(230, 228, 218),
                )
            )
    else:
        raise ValueError(kind)
    return sc


# --- disturbances ------------------------------------------------------------------------------


def shift(sc, iid, dx_m=0.0, dy_m=0.0, dyaw_deg=0.0, lift=0.0005):
    sc = copy.deepcopy(sc)
    b = sc.get(iid)
    b.center = b.center + np.array([dx_m, dy_m, lift if (dx_m or dy_m or dyaw_deg) else 0.0])
    b.rot = _rot([0, 0, 1], math.radians(dyaw_deg)) @ b.rot
    return sc


def topple(sc, iid, direction_deg=0.0):
    """A standing item falls flat, its top landing one height away along direction_deg."""
    sc = copy.deepcopy(sc)
    b = sc.get(iid)
    w, t, h = b.half
    yaw = math.radians(direction_deg)
    d = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    b.center = np.array([b.center[0], b.center[1], t]) + d * h
    b.half = np.array([h, w, t])
    b.rot = _rot([0, 0, 1], yaw)
    if b.side is not None:
        b.top = np.ascontiguousarray(b.side.transpose(1, 0, 2))
    return sc


def remove(sc, iid):
    sc = copy.deepcopy(sc)
    sc.boxes = [b for b in sc.boxes if b.id != iid]
    return sc


def add_object(sc, xy, size=(0.015, 0.015, 0.010), colour=(30, 160, 200), oid="intruder"):
    sc = copy.deepcopy(sc)
    base = max([b.center[2] + b.half[2] for b in sc.boxes if not b.item] + [0.0])
    top = np.full((24, 24, 3), colour, np.uint8)
    cv2.circle(top, (12, 12), 6, tuple(int(v * 0.5) for v in colour), -1)
    sc.boxes.append(
        Box(
            oid,
            np.array([xy[0], xy[1], base + size[2] / 2]),
            np.array(size) / 2,
            np.eye(3),
            top=top,
            colour=colour,
            item=False,
        )
    )
    return sc


def add_hand(sc, tip_xy, height=0.12, direction_deg=-90.0, length=1.2):
    """A forearm (skin-coloured box) reaching in from outside the view to tip_xy."""
    sc = copy.deepcopy(sc)
    yaw = math.radians(direction_deg)
    d = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    c = np.array([tip_xy[0], tip_xy[1], height]) - d * length / 2
    top = np.full((32, 64, 3), (190, 140, 115), np.uint8)
    top = (top * (0.9 + 0.1 * _smooth_noise(np.random.default_rng(1), 32, 64, 8))).astype(np.uint8)
    sc.boxes.append(
        Box(
            "hand",
            c,
            np.array([length / 2, 0.04, 0.025]),
            _rot([0, 0, 1], yaw),
            top=top,
            colour=(180, 130, 110),
            item=False,
        )
    )
    return sc


def add_arm(sc, tcp_xyz, base_xyz):
    """The robot's hand (a vertical box above the tool centre) and forearm toward the base,
    with the matching ArmState entry (top-down hand: approach axis -z)."""
    sc = copy.deepcopy(sc)
    tcp = np.asarray(tcp_xyz, float)
    rot = _rot([1, 0, 0], math.pi)  # tool z points down
    hand_c = tcp + np.array([0, 0, 0.06])
    grey = np.full((16, 16, 3), (70, 72, 78), np.uint8)
    sc.boxes.append(
        Box(
            "arm_hand",
            hand_c,
            np.array([0.025, 0.035, 0.06]),
            np.eye(3),
            top=grey,
            colour=(80, 80, 88),
            item=False,
        )
    )
    wrist = tcp + np.array([0, 0, 0.13])
    base = np.asarray(base_xyz, float)
    d = base - wrist
    yaw = math.atan2(d[1], d[0])
    ln = float(np.linalg.norm(d[:2]))
    pitch = math.atan2(d[2], ln)
    r = _rot([0, 0, 1], yaw) @ _rot([0, 1, 0], -pitch)
    sc.boxes.append(
        Box(
            "arm_fore",
            (wrist + base) / 2,
            np.array([np.linalg.norm(d) / 2, 0.03, 0.025]),
            r,
            top=grey,
            colour=(80, 80, 88),
            item=False,
        )
    )
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = tcp
    sc.arms = [{"ee_pose": se3(t), "gripper_fraction": 1.0}]
    sc.arm_bases = [base.tolist()]
    return sc


# --- rendering ----------------------------------------------------------------------------------


@dataclass
class Lighting:
    gain: float = 1.0
    tint: tuple = (1.0, 1.0, 1.0)
    gradient: tuple = (0.0, 0.0)  # fractional change per image width, height
    shadow: tuple | None = None  # (u, v, radius_px, depth 0..1)


def bump(t_cam, rng, rot_deg=0.3, trans_m=0.002):
    t = t_cam.copy()
    axis = rng.normal(size=3)
    t[:3, :3] = _rot(axis, math.radians(rot_deg)) @ t[:3, :3]
    v = rng.normal(size=3)
    t[:3, 3] += v / np.linalg.norm(v) * trans_m
    return t


def _sample(tex, u, v):
    """Bilinear lookup of a texture at fractional coords u (cols), v (rows) in [0, 1]."""
    h, w = tex.shape[:2]
    x = np.clip(u, 0, 1) * (w - 1)
    y = np.clip(v, 0, 1) * (h - 1)
    x0 = np.minimum(np.floor(x).astype(int), w - 2 if w > 1 else 0)
    y0 = np.minimum(np.floor(y).astype(int), h - 2 if h > 1 else 0)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    t = tex.astype(np.float32)
    top = t[y0, x0] * (1 - fx) + t[y0, x1] * fx
    bot = t[y1, x0] * (1 - fx) + t[y1, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def render(sc, t_cam, k, lighting=None, seed=0, noise=2.0, depth=True, ss=3):
    """Ray-cast at ss x ss samples per pixel (area-averaged, as a sensor integrates), then
    light, blur by 0.7 px and add noise."""
    rng = np.random.default_rng(seed)
    k_out = k
    k = k.copy()
    k[:2, :2] *= ss
    k[:2, 2] = k[:2, 2] * ss + (ss - 1) / 2.0
    h, w = HW[0] * ss, HW[1] * ss
    vv, uu = np.mgrid[0:h, 0:w]
    rays_c = np.stack(
        [(uu - k[0, 2]) / k[0, 0], (vv - k[1, 2]) / k[1, 1], np.ones_like(uu, float)], -1
    ).reshape(-1, 3)
    r, o = t_cam[:3, :3], t_cam[:3, 3]
    d = rays_c @ r.T
    n = len(d)
    tbest = np.full(n, np.inf)
    col = np.zeros((n, 3), np.float32)
    ids = np.full(n, -1, np.int32)
    shade = np.ones(n, np.float32)
    # Table z = 0.
    tz = -o[2] / np.where(np.abs(d[:, 2]) < 1e-9, -1e-9, d[:, 2])
    ok = tz > 0
    p = o + d * tz[:, None]
    tu = (p[:, 0] + TABLE_HALF[0]) / (2 * TABLE_HALF[0])
    tv = 1.0 - (p[:, 1] + TABLE_HALF[1]) / (2 * TABLE_HALF[1])
    tbest[ok] = tz[ok]
    col[:] = _sample(sc.table, tu, tv)
    col[~ok] = 0
    for bi, b in enumerate(sc.boxes):
        corners = (
            np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            * b.half
            @ b.rot.T
            + b.center
        )
        inv = np.linalg.inv(t_cam)
        pc = corners @ inv[:3, :3].T + inv[:3, 3]
        if (pc[:, 2] <= 0.01).any():
            sel = np.arange(n)
        else:
            uv = pc[:, :2] / pc[:, 2:3] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
            u0, v0 = np.floor(uv.min(0)).astype(int) - 1
            u1, v1 = np.ceil(uv.max(0)).astype(int) + 1
            if u1 < 0 or v1 < 0 or u0 >= w or v0 >= h:
                continue
            u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, w - 1), min(v1, h - 1)
            sel = (vv[v0 : v1 + 1, u0 : u1 + 1] * w + uu[v0 : v1 + 1, u0 : u1 + 1]).ravel()
        ob = (o - b.center) @ b.rot
        db = d[sel] @ b.rot
        dbs = np.where(np.abs(db) < 1e-12, 1e-12, db)
        t1 = (-b.half - ob) / dbs
        t2 = (b.half - ob) / dbs
        tmin_ax = np.minimum(t1, t2)
        tmin = tmin_ax.max(1)
        tmax = np.maximum(t1, t2).min(1)
        hit = (tmax >= tmin) & (tmin > 0) & (tmin < tbest[sel])
        if not hit.any():
            continue
        s = sel[hit]
        axis = tmin_ax[hit].argmax(1)
        pb = ob + db[hit] * tmin[hit, None]
        c = np.tile(np.array(b.colour, np.float32), (len(s), 1))
        sh = np.ones(len(s), np.float32)
        topf = axis == 2
        if b.top is not None and topf.any():
            c[topf] = _sample(
                b.top, (pb[topf, 0] / b.half[0] + 1) / 2, (1 - pb[topf, 1] / b.half[1]) / 2
            )
        sidey = axis == 1
        if b.side is not None and sidey.any():
            c[sidey] = _sample(
                b.side, (pb[sidey, 0] / b.half[0] + 1) / 2, (1 - pb[sidey, 2] / b.half[2]) / 2
            )
        sh[axis == 0] = 0.7
        sh[sidey] = 0.85 if b.side is not None else 0.75
        tbest[s] = tmin[hit]
        col[s] = c
        shade[s] = sh
        ids[s] = bi
    img = col * shade[:, None]
    img = img.reshape(h, w, 3)
    zc = (tbest.reshape(h, w)) * rays_c[:, 2].reshape(h, w)
    idmap = ids.reshape(h, w)
    if ss > 1:
        img = cv2.resize(img, (HW[1], HW[0]), interpolation=cv2.INTER_AREA)
        zc = cv2.resize(
            np.where(np.isfinite(zc), zc, 0).astype(np.float32),
            (HW[1], HW[0]),
            interpolation=cv2.INTER_AREA,
        )
        idmap = idmap[ss // 2 :: ss, ss // 2 :: ss]
        h, w = HW
        vv, uu = np.mgrid[0:h, 0:w]
    lg = lighting or Lighting()
    field_ = lg.gain * (1 + lg.gradient[0] * (uu / w - 0.5) + lg.gradient[1] * (vv / h - 0.5))
    if lg.shadow is not None:
        su, sv, sr, sd = lg.shadow
        field_ = field_ * (1 - sd * np.exp(-((uu - su) ** 2 + (vv - sv) ** 2) / (2 * sr * sr)))
    img = img * field_[:, :, None] * np.array(lg.tint, np.float32)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.7)
    img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    rgb = np.clip(img, 0, 255).astype(np.uint8)
    out = {"name": "top", "rgb": rgb, "intrinsics": k_out.copy(), "pose": se3(t_cam)}
    if depth:
        z = zc.astype(np.float32)
        sigma = 0.0008 + 0.0022 * (z / 1.0) ** 2
        z = z + rng.normal(0, 1, z.shape).astype(np.float32) * sigma
        z = np.round(z / 0.001) * 0.001 if k_out is OVERHEAD_K else z
        z[rng.random(z.shape) < 0.01] = 0.0
        z[~np.isfinite(z)] = 0.0
        out["depth"] = z.astype(np.float32)
    else:
        out["depth"] = None
    return out, idmap


def item_masks(sc, idmap):
    return {b.id: idmap == i for i, b in enumerate(sc.boxes) if b.item}


def observation(frame, sc):
    return {"cameras": [frame], "arms": list(sc.arms)}
