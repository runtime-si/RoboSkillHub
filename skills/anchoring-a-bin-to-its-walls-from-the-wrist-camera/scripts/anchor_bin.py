"""Re-anchor a bin to the walls the wrist camera sees: the median of several looks, outliers left out."""

from __future__ import annotations

import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

SPAN = 0.09
"""A wall is looked for this far either side of where the bin has it [m]. tuned_on: one YAM
station, a 449 x 262 x 154 mm cardboard box seen by the wrist depth camera from about 0.3 m over
the counter (2026-11-15); the arm's error there was a few cm."""

BIN = 0.004
"""Histogram step across a wall [m]: the shift is resolved no finer than this."""

WALL_HALF_THICKNESS = 0.002
"""The densest sheet is taken as the wall's middle, this far inside its outer face [m]. tuned_on:
that box's 4 mm corrugated cardboard; pass wall_thickness for another."""

WALL_FROM = 0.05
"""Wall points are those at least this far over the bin's support [m] (below it the contents and
the floor crowd the wall), up to 3 cm over the rim. tuned_on: same."""

RIM_OVER = 0.03

END_CLEAR = 0.03
"""A wall's points are taken no nearer than this to its two corners [m]."""

MIN_POINTS, MIN_WALL_POINTS, MIN_PEAK = 800, 600, 60
"""Wall-coloured points in the frame, points near one wall, and points in the densest 4 mm sheet
below which a wall is not read. tuned_on: that camera read at every second pixel; they scale with
resolution and stride and were not swept."""

MAX_SHIFT = 0.10
"""A reading further than this from where the bin was given is not counted [m]. tuned_on: same
station; chosen as a guard."""

OUTLIER = 0.03
"""Once two readings are held, one further than this from their median is not counted [m].
tuned_on: same station: one look read the walls 7 cm out and then saw no pile where it put the
box; the looks otherwise agreed within this. Not swept."""

CARD_RED_OVER_BLUE, CARD_MIN_RED = 22, 70
"""wall_colour="cardboard": a pixel is wall when red exceeds blue by more than 22 and is over 70
(8-bit). tuned_on: brown corrugated cardboard under that station's room light, against grey and
black parts. Any other wall needs wall_mask or wall_colour="any"."""


class Output(TypedDict):
    route: str
    detail: str
    anchored_obb: dict[str, Any]
    shift: dict[str, float] | None
    used_shift: dict[str, float] | None
    shifts: list
    walls_read: list
    looks: int
    camera: str


def _rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in "wxyz")
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix(pose: Any) -> np.ndarray:
    if isinstance(pose, dict) and "pose" in pose and "position" not in pose:
        pose = pose["pose"]
    if isinstance(pose, dict):
        m = np.eye(4)
        m[:3, :3] = _rot(pose.get("rotation") or {"w": 1, "x": 0, "y": 0, "z": 0})
        p = pose["position"]
        m[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
        return m
    m = np.asarray(pose, dtype=float)
    if m.shape != (4, 4):
        raise ValueError("camera pose must be an Se3Pose {position, rotation} or a 4x4 matrix")
    return m


def footprint(obb: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    """An upright box's (centre xy, half extents along its own x / y, yaw [rad])."""
    c = obb["center"]
    e = obb["extent"]
    r = _rot(obb.get("orientation") or {"w": 1, "x": 0, "y": 0, "z": 0})
    return (
        np.array([float(c["x"]), float(c["y"])]),
        np.array([float(e["x"]), float(e["y"])]),
        math.atan2(r[1, 0], r[0, 0]),
    )


def _wrist_camera(ctx: NodeContext, frame: dict | None, name: str | None) -> dict | None:
    """The named posed depth camera; unnamed, the first on a wrist."""
    if frame is not None:
        return frame
    obs = ctx.tool("robot.get_observation")
    cams = obs.get("cameras") if isinstance(obs, dict) else getattr(obs, "cameras", None)
    if isinstance(cams, dict):
        cams = [dict(v, name=v.get("name", k)) for k, v in cams.items()]
    cams = [c for c in (cams or []) if c.get("depth") is not None and c.get("pose") is not None]
    if name:
        return next((c for c in cams if str(c.get("name")) == name), None)
    return next(
        (c for c in cams if any(k in str(c.get("name", "")) for k in ("wrist", "eye_in_hand"))),
        None,
    )


def wall_points(
    cam: dict,
    wall_colour: str = "cardboard",
    wall_mask: Any = None,
    stride: int = 2,
    min_range: float = 0.0,
) -> np.ndarray:
    """World points of the frame's pixels that are wall: by ``wall_mask`` (an image mask), by the
    cardboard colour rule, or all of them (wall_colour="any")."""
    depth = np.asarray(cam["depth"], dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    t = _matrix(cam["pose"])
    ok = np.isfinite(depth) & (depth > max(min_range, 0.0))
    if wall_mask is not None:
        m = np.asarray(wall_mask)
        ok &= (m[..., 0] if m.ndim == 3 else m) > (127 if m.dtype == np.uint8 else 0)
    elif wall_colour == "cardboard":
        rgb = np.asarray(cam["rgb"]).astype(np.float32)
        ok &= (rgb[..., 0] - rgb[..., 2] > CARD_RED_OVER_BLUE) & (rgb[..., 0] > CARD_MIN_RED)
    elif wall_colour != "any":
        raise ValueError('wall_colour must be "cardboard" or "any"')
    sub = np.zeros_like(ok)
    sub[::stride, ::stride] = ok[::stride, ::stride]
    vv, uu = np.nonzero(sub)
    z = depth[vv, uu]
    pc = np.column_stack([(uu - k[0, 2]) * z / k[0, 0], (vv - k[1, 2]) * z / k[1, 1], z])
    return pc @ t[:3, :3].T + t[:3, 3]


def wall_shift(
    points: np.ndarray,
    bin_obb: dict[str, Any],
    span: float = SPAN,
    wall_thickness: float = 2 * WALL_HALF_THICKNESS,
) -> tuple[np.ndarray | None, list[str]]:
    """How far these wall points have the bin's walls from where ``bin_obb`` has them.

    Each of the four walls is the densest ``BIN``-thick sheet of points across it, looked for
    within ``span`` of its expected place and away from its corners. The shift along the bin's
    long axis is the mean over the end walls read, along its short axis over the side walls
    read; an axis with no wall read gets 0. Returns (world (dx, dy) or None when no wall was
    read, the walls read as "+x", "-x", "+y", "-y" in the bin's frame).
    """
    centre, half, yaw = footprint(bin_obb)
    support = float(bin_obb["center"]["z"]) - float(bin_obb["extent"]["z"])
    rim = float(bin_obb["center"]["z"]) + float(bin_obb["extent"]["z"])
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[(pts[:, 2] > support + WALL_FROM) & (pts[:, 2] < rim + RIM_OVER)]
    if len(pts) < MIN_POINTS:
        return None, []
    c, s = math.cos(yaw), math.sin(yaw)
    d = pts[:, :2] - centre
    loc = np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])
    off: list[list[float]] = [[], []]
    read: list[str] = []
    for ax, other in ((0, 1), (1, 0)):
        for sign in (1.0, -1.0):
            at = sign * half[ax]
            m = (np.abs(loc[:, ax] - at) < span) & (np.abs(loc[:, other]) < half[other] - END_CLEAR)
            if m.sum() <= MIN_WALL_POINTS:
                continue
            hist, edges = np.histogram(loc[m, ax], bins=np.arange(at - span, at + span + 1e-6, BIN))
            if hist.max() <= MIN_PEAK:
                continue
            sheet = float(edges[int(np.argmax(hist))]) + BIN / 2
            off[ax].append(sheet - sign * (half[ax] - wall_thickness / 2))
            read.append(("+" if sign > 0 else "-") + "xy"[ax])
    if not read:
        return None, []
    dx = float(np.mean(off[0])) if off[0] else 0.0
    dy = float(np.mean(off[1])) if off[1] else 0.0
    return np.array([c * dx - s * dy, s * dx + c * dy]), read


def accept(
    shift: np.ndarray, past: list, max_shift: float = MAX_SHIFT, outlier: float = OUTLIER
) -> bool:
    """Whether this look's reading is counted: within ``max_shift`` of the bin as given, and,
    once two readings are held, within ``outlier`` of their median."""
    if float(np.linalg.norm(shift)) >= max_shift:
        return False
    if len(past) < 2:
        return True
    return float(np.linalg.norm(shift - np.median(np.asarray(past, float), axis=0))) < outlier


def run(
    ctx: NodeContext,
    bin_obb: dict,
    past_shifts: list | None = None,
    camera: str = "",
    camera_frame: dict | None = None,
    wall_colour: str = "cardboard",
    wall_mask: Any = None,
    wall_thickness: float = 2 * WALL_HALF_THICKNESS,
    max_shift: float = MAX_SHIFT,
    outlier: float = OUTLIER,
    min_range: float = 0.16,
    stride: int = 2,
) -> Output:
    """The bin moved to where the wrist camera's looks so far have its walls.

    Args:
        bin_obb: The bin as the fixed camera has it (upright; half extents of its outer size;
            its top the rim). Always this same box, never a previously anchored one.
        past_shifts: The ``shifts`` this skill returned on the previous look ([[dx, dy], ...]).
        camera / camera_frame: The wrist camera's name (default: the first wrist camera), or a
            CameraFrame {rgb, depth, intrinsics, pose}.
        wall_colour: "cardboard" (the colour rule) or "any" (every depth point in the wall band).
        wall_mask: An image mask of the bin's walls, instead of the colour rule.
        wall_thickness: The walls' thickness [m].
        max_shift, outlier: When a reading is not counted [m].
        min_range: Depth nearer than this is dropped (the hand's own fingers) [m].

    Returns:
        ``route`` "anchored", "outlier" or "walls_not_seen". ``anchored_obb`` is always usable:
        bin_obb moved by the median of the readings counted so far (bin_obb itself with none).
        Pass ``shifts`` back as past_shifts on the next look.
    """
    past = [[float(v[0]), float(v[1])] for v in (past_shifts or [])]
    out: Output = {
        "route": "walls_not_seen",
        "detail": "",
        "anchored_obb": bin_obb,
        "shift": None,
        "used_shift": None,
        "shifts": past,
        "walls_read": [],
        "looks": len(past),
        "camera": "",
    }

    def finish(route: str, detail: str) -> Output:
        if out["shifts"]:
            used = np.median(np.asarray(out["shifts"], float), axis=0)
            out["used_shift"] = {"x": float(used[0]), "y": float(used[1]), "z": 0.0}
            moved = dict(bin_obb)
            moved["center"] = {
                "x": float(bin_obb["center"]["x"]) + float(used[0]),
                "y": float(bin_obb["center"]["y"]) + float(used[1]),
                "z": float(bin_obb["center"]["z"]),
            }
            out["anchored_obb"] = moved
        out["looks"] = len(out["shifts"])
        out["route"], out["detail"] = route, detail
        return out

    cam = _wrist_camera(ctx, camera_frame, camera or None)
    if cam is None:
        return finish("walls_not_seen", f"no posed wrist depth camera {camera or '(any)'!r}")
    out["camera"] = str(cam.get("name", ""))
    pts = wall_points(cam, wall_colour, wall_mask, max(1, int(stride)), float(min_range))
    shift, read = wall_shift(pts, bin_obb, wall_thickness=float(wall_thickness))
    if shift is None:
        return finish("walls_not_seen", f"no wall read from {len(pts)} wall-coloured points")
    out["walls_read"] = read
    out["shift"] = {"x": float(shift[0]), "y": float(shift[1]), "z": 0.0}
    mm = f"{shift[0] * 1e3:+.0f}, {shift[1] * 1e3:+.0f} mm from walls {' '.join(read)}"
    if not accept(shift, past, float(max_shift), float(outlier)):
        return finish("outlier", f"a reading of {mm} was not counted")
    out["shifts"] = past + [[float(shift[0]), float(shift[1])]]
    return finish("anchored", f"read {mm}; the median of {len(out['shifts'])} look(s) is used")
