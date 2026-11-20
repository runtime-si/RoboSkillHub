"""What a wrist RGB-D frame says about flat parts on a support and a part held over it, by depth alone.

Pure functions (numpy, OpenCV), no robot: the support is a plane fitted in the camera's own frame, and everything
else is measured against that plane in the same frame, so neither the camera's hand-eye calibration nor the arm's
height comes into a height reading. Written for yam_dominos (two YAM arms, D405 wrist cameras, 49 x 24.5 x 7.4 mm
tiles on a glossy stone counter, 2026-11); the constants say what they were measured on.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

MIN_DEPTH = 0.05
"""Nearer readings are dropped [m]: a D405 returns nothing useful under ~5 cm, and the fingers sit there."""

MAX_DEPTH = 0.60
"""Further readings are dropped [m]: the looks this is for are 7-25 cm from the support."""

PLANE_SEARCH = 0.04
"""The support is looked for within this of where the camera's pose has it [m]. The real left YAM read its counter
6-18 mm from where its joints put it, the right one 10 mm; 4 cm covers both with room."""

PLANE_BIN = 0.002
"""Histogram bin for the support's level [m]: the level most points lie at wins, so a held part, the fingers and
other parts in view do not drag the plane."""

PLANE_BAND = 0.002
"""Points within this of the plane refit it [m] (twice)."""

MIN_PLANE_POINTS = 500
"""Fewer points near the expected support than this: the support is not in view."""

MIN_PART_PX = 300
"""A blob smaller than this is not a part [px at 640 x 480]: a tile 17 cm away is ~3000 px."""


def pose_of(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def vec(v) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat(r: np.ndarray) -> dict[str, float]:
    t = float(np.trace(r))
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    return dict(zip(("w", "x", "y", "z"), (float(v) for v in q), strict=True))


def matrix(pose: dict[str, Any]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot(pose["rotation"])
    m[:3, 3] = xyz(pose)
    return m


def pose_dict(m: np.ndarray) -> dict[str, Any]:
    return {"position": vec(m[:3, 3]), "rotation": quat(m[:3, :3])}


def refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or "" (connectors that route a refusal instead of raising)."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def wrist_camera(obs: dict[str, Any], tool: np.ndarray, name: str | None) -> dict[str, Any] | None:
    """The named camera, else the eye-in-hand / wrist camera riding nearest the tool."""
    cams = [
        c
        for c in obs.get("cameras") or []
        if c.get("pose") is not None and c.get("depth") is not None
    ]
    if name:
        return next((c for c in cams if c.get("name") == name), None)
    wrist = [c for c in cams if any(k in str(c.get("name", "")) for k in ("eye_in_hand", "wrist"))]
    if not wrist:
        return None
    return min(wrist, key=lambda c: float(np.linalg.norm(xyz(c["pose"]) - tool[:3, 3])))


def cloud(depth, intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Every pixel's point in the camera's frame [h, w, 3] and which of them are valid [h, w]."""
    z = np.asarray(depth, dtype=np.float64)
    if z.ndim == 3:
        z = z[..., 0]
    k = np.asarray(intrinsics, dtype=np.float64)
    h, w = z.shape
    ok = np.isfinite(z) & (z > MIN_DEPTH) & (z < MAX_DEPTH)
    vs, us = np.mgrid[0:h, 0:w]
    pts = np.stack([(us - k[0, 2]) * z / k[0, 0], (vs - k[1, 2]) * z / k[1, 1], z], -1)
    return pts, ok


def fit_support(points: np.ndarray, t_world_cam: np.ndarray, support_z: float = 0.0):
    """The support as a plane in the CAMERA's frame: (n, d, d0), a point p's height over it being n . p - d.

    ``d0`` is where the camera's pose and ``support_z`` (the support's world height) have the plane; d - d0 is how
    far the support really is from there, along its normal. The plane is the level most points lie at within
    PLANE_SEARCH of that, fitted twice by SVD. Raises ValueError when the support is not in view."""
    n0 = t_world_cam[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    d0 = float(support_z - t_world_cam[2, 3])
    hgt = points @ n0 - d0
    near = hgt[np.abs(hgt) < PLANE_SEARCH]
    if len(near) < MIN_PLANE_POINTS:
        raise ValueError("the support is not in view")
    hist, edges = np.histogram(near, bins=np.arange(-PLANE_SEARCH, PLANE_SEARCH + 1e-9, PLANE_BIN))
    mode = float(edges[int(np.argmax(hist))] + 0.5 * PLANE_BIN)
    sel = np.abs(hgt - mode) < 2 * PLANE_BAND
    n, d = n0, d0 + mode
    for _ in range(2):
        q = points[sel]
        if len(q) < MIN_PLANE_POINTS:
            break
        c = q.mean(0)
        _u, _s, vt = np.linalg.svd(q[:: max(1, len(q) // 20000)] - c, full_matrices=False)
        n = vt[2] if vt[2] @ n0 > 0 else -vt[2]
        d = float(n @ c)
        sel = np.abs(points @ n - d) < PLANE_BAND
    return n, d, d0


def flat_parts(
    depth, intrinsics, t_world_cam: np.ndarray, size, support_z: float = 0.0
) -> dict[str, Any]:
    """The flat parts of a known size lying on the support in one wrist frame.

    ``size``: (length, width, thickness) [m], length >= width. A part is what stands between 0.45 and 1 thickness
    (+8 mm) over the fitted plane, its outline taken to the plane raised by the thickness along each pixel's ray
    and boxed in the world. Returns {"parts": [{"centre" [x, y], "yaw" [rad, the long side], "size" [long, short],
    "px", "height", "point_cam" (its top face's middle in the camera's frame)}], "rejected": [...], "plane":
    (n, d), "support_off": d - d0}, parts nearest the image's middle first.

    Why depth and not colour: the real counter is glossy, and each cream tile had a reflection and a shadow beside
    it as yellow as the tile; colour blobs came out half a tile wide of the tile. Depth has neither."""
    length, width, thick = (float(v) for v in size)
    pts, ok = cloud(depth, intrinsics)
    k = np.asarray(intrinsics, dtype=np.float64)
    h, w = ok.shape
    n, d, d0 = fit_support(pts[ok], t_world_cam, support_z)
    height = np.where(ok, pts @ n - d, 0.0)
    mask = ((height > 0.45 * thick) & (height < thick + 0.008)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
    )  # printed pips read as holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    r_wc, t_wc = t_world_cam[:3, :3], t_world_cam[:3, 3]
    parts, rejected = [], []
    for c in contours:
        if cv2.contourArea(c) < MIN_PART_PX * (w * h) / (640 * 480):
            continue
        uv = c.reshape(-1, 2).astype(np.float64)
        rays = np.stack(
            [(uv[:, 0] - k[0, 2]) / k[0, 0], (uv[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(uv))], 1
        )
        top = rays * ((d + thick) / (rays @ n))[:, None]
        world = (top @ r_wc.T + t_wc)[:, :2].astype(np.float32)
        (cx, cy), (a, b), ang = cv2.minAreaRect(world)
        if a < b:
            a, b, ang = b, a, ang + 90.0
        inside = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(inside, [c], -1, 1, -1)
        # the middle back in the camera's frame: the point of that world x, y on the raised plane
        base, up = r_wc.T @ (np.array([cx, cy, 0.0]) - t_wc), r_wc.T @ np.array([0.0, 0.0, 1.0])
        point_cam = base + (d + thick - n @ base) / (n @ up) * up
        rec = {
            "centre": [float(cx), float(cy)],
            "yaw": float(math.radians(ang)),
            "size": [float(a), float(b)],
            "px": [float(uv[:, 0].mean()), float(uv[:, 1].mean())],
            "height": float(np.median(height[(inside > 0) & ok])),
            "point_cam": [float(v) for v in point_cam],
        }
        cut = (
            uv[:, 0].min() < 3
            or uv[:, 1].min() < 3
            or uv[:, 0].max() > w - 4
            or uv[:, 1].max() > h - 4
        )
        if cut:
            rejected.append({**rec, "why": "cut by the picture's edge"})
        elif not (0.8 * length < a < 1.25 * length and 0.8 * width < b < 1.4 * width):
            rejected.append(
                {**rec, "why": f"not the part's size ({1000 * a:.0f} x {1000 * b:.0f} mm)"}
            )
        else:
            parts.append(rec)
    mid = np.array([w / 2, h / 2])
    parts.sort(key=lambda t: float(np.linalg.norm(np.asarray(t["px"]) - mid)))
    return {
        "parts": parts,
        "rejected": rejected,
        "plane": (n, float(d)),
        "support_off": float(d - d0),
    }


def held_gap(
    depth,
    intrinsics,
    t_world_cam: np.ndarray,
    part_tall: float,
    expect_cam,
    within: float = 0.03,
    rgb=None,
    min_light: float = 60.0,
    min_px: int = 250,
    support_z: float = 0.0,
) -> dict[str, Any]:
    """A part held upright in this hand over the support: how far its LOWER end is over the support.

    Both ends of the measurement are in the one depth frame: the part's upper end (what stands over half the part's
    height above the fitted plane within ``within`` of ``expect_cam``, a point in the camera's frame; its top 4 mm)
    and the support round the part's foot (the plane, then set to the support's own level within 9 cm of the part:
    over a whole frame the fit was out by a centimetre from one side to the other). The lower end is ``part_tall``
    under the upper one. ``rgb`` with ``min_light`` keeps only light pixels (CIELAB L): the YAM's fingers are black
    (under 40) and a cream tile's end in the hand's shadow read 85-125; pass rgb=None for parts that are not light.

    Why the part itself and not the grasp's model of it: held tiles' upper ends stood 6 and 11 mm lower than the
    grasp's geometry had them (a part sits where the pads took it, and may slide in them), and the hand, lowered by
    the model, pushed them into the counter by that much.

    {"seen", "gap", "top", "point_cam", "points", "support_points"}."""
    pts, ok = cloud(depth, intrinsics)
    n, d, _d0 = fit_support(pts[ok], t_world_cam, support_z)
    height = pts @ n - d
    part = (
        ok
        & (height > 0.5 * part_tall)
        & (height < 0.30)
        & (np.linalg.norm(pts - np.asarray(expect_cam, float), axis=-1) < within)
    )
    if rgb is not None:
        part &= cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2LAB)[..., 0] > min_light
    part = cv2.morphologyEx(part.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, lab, stats, _ = cv2.connectedComponentsWithStats(part)
    if count < 2 or stats[1:, cv2.CC_STAT_AREA].max() < min_px:
        return {"seen": False, "points": int(part.sum())}
    m = lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    upper = m & (height > float(np.percentile(height[m], 97)) - 0.004)
    p_top = pts[upper].mean(0)
    side = pts - p_top
    side = side - np.outer(side.reshape(-1, 3) @ n, n).reshape(side.shape)
    near = ok & (np.abs(height) < 0.02) & (np.linalg.norm(side, axis=-1) < 0.09)
    local = float(np.median(height[near])) if near.sum() > 400 else 0.0
    top = float(np.median(height[upper])) - local
    return {
        "seen": True,
        "gap": top - float(part_tall),
        "top": top,
        "point_cam": [float(v) for v in p_top],
        "points": int(m.sum()),
        "support_points": int(near.sum()),
    }
