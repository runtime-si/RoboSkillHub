"""Synthetic RGB-D clips of a domino line toppling, rendered by ray casting boxes (CPU).

Each tile is a box that rotates about its front bottom edge. Tile i starts at
``t_trigger + i * pitch / speed`` and accelerates (angle ~ t^1.5) to the angle where
its top meets the next tile's back face, which it reaches as the next one starts;
after that it leans on the next tile (top on its back face, closed form) as that
one falls. A tile that is hit but does not fall stops the wave: the one before it
rests on it at the contact angle. The last tile of the line falls flat. Planned
slots are exact; the placed tiles are off them by a little noise. A hand is an
axis-aligned box moving along a path. Depth is the camera's z with noise growing
with range, quantised to 1 mm, with random holes; rgb is shaded with noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TILE = (0.048, 0.024, 0.0075)  # height, width, thickness
TABLE_RGB = np.array([96.0, 78.0, 58.0])
TILE_RGB = np.array([236.0, 233.0, 226.0])
HAND_RGB = np.array([48.0, 50.0, 56.0])
LIGHT = np.array([0.3, 0.2, 0.93]) / np.linalg.norm([0.3, 0.2, 0.93])


@dataclass
class Hand:
    """A box hovering along a path: (time, centre) keyframes, linear between."""

    keys: list[tuple[float, tuple[float, float, float]]]
    size: tuple[float, float, float] = (0.07, 0.09, 0.04)

    def centre(self, t: float):
        ts = [k[0] for k in self.keys]
        if t < ts[0] or t > ts[-1]:
            return None
        c = np.array([np.interp(t, ts, [k[1][i] for k in self.keys]) for i in range(3)])
        return c


@dataclass
class Scene:
    n: int = 10
    pitch: float = 0.037
    stop_at: int | None = None  # first tile that is hit and stands
    fps: float = 60.0
    duration: float | None = None
    t_trigger: float = 0.15
    speed: float = 0.8  # m/s along the line
    camera: str = "overhead"
    hands: list[Hand] = field(default_factory=list)
    depth_noise: float = 0.002  # at 0.9 m
    rgb_noise: float = 4.0
    holes: float = 0.01
    place_err: float = 0.0015
    yaw_err_deg: float = 2.0
    layout: str = "line"  # or "arc"
    arc_radius: float = 0.25
    early: tuple[int, float] | None = None  # tile, time: falls alone before the trigger
    lone: tuple[int, float] | None = None  # tile, time: falls alone (bumped)
    no_trigger: bool = False
    origin: tuple[float, float] = (0.15, 0.0)
    seed: int = 0


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = eye
    return T


def mat_to_pose(T):
    from gap_core.types import matrix_to_pose

    return matrix_to_pose(T)


def slots_for(sc: Scene):
    """Planned slots (centre at table height, facing) in planning-a-domino-layout's form."""
    out = []
    for k in range(sc.n):
        if sc.layout == "arc":
            a = k * sc.pitch / sc.arc_radius
            cx, cy = sc.origin[0], sc.origin[1] + sc.arc_radius
            x, y = cx + sc.arc_radius * math.sin(a), cy - sc.arc_radius * math.cos(a)
            facing = math.degrees(a)
        else:
            x, y, facing = sc.origin[0] + k * sc.pitch, sc.origin[1], 0.0
        h = math.radians(facing) / 2.0
        out.append(
            {
                "index": k,
                "position": {"x": x, "y": y, "z": 0.0},
                "rotation": {"w": math.cos(h), "x": 0.0, "y": 0.0, "z": math.sin(h)},
                "facing_deg": facing,
                "arc_length": k * sc.pitch,
            }
        )
    return out


def camera_for(sc: Scene, slots):
    pts = np.array([[s["position"]["x"], s["position"]["y"]] for s in slots])
    mid = pts.mean(axis=0)
    if sc.camera == "station":
        # The real station's overhead Orbbec: intrinsics of cable_frame_20261103-1421
        # and its ChArUco-calibrated pose (top_camera_pose_at_capture_20261103-1421),
        # left-arm base frame with the table at z = -0.03 moved to z = 0.
        k = np.array([[461.2006, 0, 320.0015], [0, 461.5186, 240.0194], [0, 0, 1.0]])
        T = np.array(
            [
                [0.998535, -0.046372, 0.027883, 0.29002],
                [-0.054109, -0.857869, 0.511012, 0.036629],
                [0.000223, -0.511772, -0.859121, 0.908602 + 0.03],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return k, T, (480, 640)
    if sc.camera == "overhead":
        k = np.array([[461.0, 0, 320.0], [0, 461.0, 240.0], [0, 0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = [mid[0] + 0.06, mid[1] + 0.12, 0.9]
        return k, T, (480, 640)
    # A wrist D405 ~0.35 m away, looking at the line from its side and above.
    k = np.array([[390.0, 0, 320.0], [0, 390.0, 240.0], [0, 0, 1.0]])
    eye = [mid[0] - 0.05, mid[1] - 0.26, 0.24]
    return k, look_at(eye, [mid[0], mid[1], 0.015]), (480, 640)


def kinematics(sc: Scene, gaps, times):
    """Angles [n, T] in degrees and the true onset times (nan when it never falls)."""
    h, _, d = TILE
    n = sc.n
    dt_tile = sc.pitch / sc.speed
    onsets = np.full(n, np.nan)
    stop = n if sc.stop_at is None else sc.stop_at
    if not sc.no_trigger:
        for i in range(stop):
            onsets[i] = sc.t_trigger + i * dt_tile
    if sc.early is not None:
        j, t0 = sc.early
        for i in range(j, stop):
            onsets[i] = t0 + (i - j) * dt_tile
    if sc.lone is not None:
        onsets[sc.lone[0]] = sc.lone[1]
    contact = np.array(
        [math.degrees(math.asin(min(1.0, g / h))) if g < h else 90.0 for g in gaps] + [90.0]
    )
    ang = np.zeros((n, len(times)))
    for i in range(n - 1, -1, -1):
        if not np.isfinite(onsets[i]):
            continue
        lim = contact[i]
        nxt_falls = i + 1 < n and np.isfinite(onsets[i + 1]) and onsets[i + 1] > onsets[i]
        t_c = dt_tile if lim < 90.0 else dt_tile * (90.0 / 38.0) ** (1 / 1.5)
        u = np.clip((times - onsets[i]) / t_c, 0.0, 1.0)
        a = lim * u**1.5
        if lim < 90.0 and nxt_falls:
            p_eff = gaps[i] + d
            beta = np.radians(ang[i + 1])
            lean = np.degrees(beta - np.arcsin(np.clip((d - p_eff * np.cos(beta)) / h, -1, 1)))
            a = np.where(u >= 1.0, np.maximum(lim, np.minimum(lean, 90.0)), a)
        ang[i] = a
    return ang, onsets


def _boxes(sc, placed, ang_t, hand_boxes):
    """(centre, axes[3x3 columns], half extents, rgb, id) per box."""
    h, w, d = TILE
    out = []
    up = np.array([0.0, 0.0, 1.0])
    for i, (piv, f) in enumerate(placed):
        t = math.radians(ang_t[i])
        c, s = math.cos(t), math.sin(t)
        lat = np.array([-f[1], f[0], 0.0])
        ax_x = c * f - s * up
        ax_z = s * f + c * up
        centre = piv + (-d / 2 * c + h / 2 * s) * f + (d / 2 * s + h / 2 * c) * up
        out.append(
            (
                centre,
                np.stack([ax_x, lat, ax_z], axis=1),
                np.array([d / 2, w / 2, h / 2]),
                TILE_RGB,
                i + 1,
            )
        )
    for c, size in hand_boxes:
        out.append((c, np.eye(3), np.asarray(size) / 2.0, HAND_RGB, 100))
    return out


def render(sc: Scene):
    rng = np.random.default_rng(sc.seed)
    h, w, d = TILE
    slots = slots_for(sc)
    placed = []
    for s in slots:
        a = math.radians(s["facing_deg"] + rng.normal(0, sc.yaw_err_deg))
        f = np.array([math.cos(a), math.sin(a), 0.0])
        c = (
            np.array([s["position"]["x"], s["position"]["y"], 0.0])
            + np.r_[rng.normal(0, sc.place_err, 2), 0]
        )
        placed.append((c + f * d / 2, f))
    gaps = []
    for i in range(sc.n - 1):
        (p0, f0), (p1, f1) = placed[i], placed[i + 1]
        back = p1 - f1 * d
        gaps.append(float(((back - p0) @ f1) / max(float(f0 @ f1), 0.2)))
    dur = sc.duration or (sc.t_trigger + sc.n * sc.pitch / sc.speed + 0.35)
    times = np.arange(0.0, dur, 1.0 / sc.fps)
    ang, onsets = kinematics(sc, gaps, times)
    K, Tc, (H, W) = camera_for(sc, slots)
    R, o = Tc[:3, :3], Tc[:3, 3]
    vv, uu = np.mgrid[0:H, 0:W]
    dirs_c = np.stack(
        [(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], axis=-1
    )
    dirs = dirs_c @ R.T  # world, z_cam = 1 per unit t
    down = dirs[..., 2] < -1e-9
    t_table = np.where(down, -o[2] / np.where(down, dirs[..., 2], -1.0), np.inf)
    frames, ids = [], []
    pose = mat_to_pose(Tc)
    for fi, t in enumerate(times):
        zbuf = t_table.copy()
        rgb = np.broadcast_to(TABLE_RGB, (H, W, 3)).copy()
        idimg = np.zeros((H, W), np.int32)
        hands = []
        for hd in sc.hands:
            c = hd.centre(float(t))
            if c is not None:
                hands.append((c, hd.size))
        for centre, axes, half, colour, bid in _boxes(sc, placed, ang[:, fi], hands):
            sg = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
            corners = centre + (sg * half) @ axes.T
            pc = (corners - o) @ R
            if (pc[:, 2] <= 1e-3).any():
                continue
            u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
            v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
            u0, u1 = max(int(u.min()) - 1, 0), min(int(u.max()) + 2, W)
            v0, v1 = max(int(v.min()) - 1, 0), min(int(v.max()) + 2, H)
            if u1 <= u0 or v1 <= v0:
                continue
            dd = dirs[v0:v1, u0:u1] @ axes  # box frame
            oo = (o - centre) @ axes
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-half - oo) / dd
                t2 = (half - oo) / dd
            tmin_ax = np.minimum(t1, t2)
            tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
            tmin = np.nanmax(tmin_ax, axis=-1)
            hit = (tmax >= tmin) & (tmin > 0) & (tmin < zbuf[v0:v1, u0:u1])
            if not hit.any():
                continue
            k_ax = np.argmax(tmin_ax, axis=-1)
            normal_box = np.zeros(dd.shape)
            np.put_along_axis(
                normal_box,
                k_ax[..., None],
                -np.sign(np.take_along_axis(dd, k_ax[..., None], -1)),
                -1,
            )
            shade = np.clip((normal_box @ axes.T) @ LIGHT, 0.0, 1.0) * 0.45 + 0.55
            sub_z = zbuf[v0:v1, u0:u1]
            sub_z[hit] = tmin[hit]
            sub_rgb = rgb[v0:v1, u0:u1]
            sub_rgb[hit] = colour * shade[hit][:, None]
            idimg[v0:v1, u0:u1][hit] = bid
        depth = zbuf.astype(np.float64)  # t along dirs with z_cam = 1 is the camera z
        noise = sc.depth_noise * (depth / 0.9) ** 2
        depth = depth + rng.normal(0.0, 1.0, depth.shape) * noise
        depth = np.round(depth, 3)
        depth[rng.random(depth.shape) < sc.holes] = 0.0
        depth[~np.isfinite(depth)] = 0.0
        rgb = np.clip(rgb + rng.normal(0.0, sc.rgb_noise, rgb.shape), 0, 255).astype(np.uint8)
        frames.append(
            {
                "name": sc.camera,
                "rgb": rgb,
                "depth": depth.astype(np.float32),
                "intrinsics": K,
                "pose": pose,
                "timestamp": float(t),
            }
        )
        ids.append(idimg)
    fell = np.isfinite(onsets)
    stop = next((i for i in range(sc.n) if not fell[i]), -1)
    return {
        "frames": frames,
        "slots": slots,
        "ids": ids,
        "times": times,
        "angles": ang,
        "onsets": onsets,
        "fell": fell,
        "stop_index": stop,
        "placed": placed,
        "gaps": gaps,
        "K": K,
        "T": Tc,
    }
