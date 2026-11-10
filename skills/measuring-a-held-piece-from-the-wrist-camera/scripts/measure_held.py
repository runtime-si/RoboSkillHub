"""Measure where a flat piece sits in the gripper from one wrist-camera depth frame.

Written for the hub. The wrist camera sees the held piece past the blades, partly hidden by them. The piece's
known top-down shape (its mask, e.g. from the mosaic before the pick) is fitted to what the depth shows:
points at the piece's top height (blade tips cut out) say "piece here"; points on the mat far below, traced
back along their rays to the piece's top plane, say "no piece here". Each candidate pose (x, y, yaw) scores
the first inside the shape minus the second; a coarse-to-fine grid search round the expected grip finds the
best, and the poses scoring nearly as well give the uncertainty. This is template matching with
free-space evidence (as in occupancy-based scan matching, e.g. Hess et al., "Real-Time Loop Closure in 2D
LIDAR SLAM", ICRA 2016); no code copied.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

THICKNESS_M = 0.002
"""Piece thickness (assumed, as in verifying-piece-flushness-from-wrist-depth)."""

SLAB_M = (0.003, 0.022)
"""The piece's top is searched this far under the tool centre: the blade tips reach 10 mm under it (station
model), a piece held by its edge hangs its top 8-12 mm under it; deeper is what the jaws hold over (a
placed board 28 mm, the mat 30 mm under the tool 3 cm up in the station twin), the free-space evidence."""

TOP_TOL_M = 0.001
"""Points within this of the piece's top height count as the piece."""

FREE_BELOW_M = 0.004
"""Points this far or more under the piece's top are seen past it: the piece is not on their ray."""

BLADE_HALF_WIDTH_M = 0.0085
BLADE_THICKNESS_M = 0.0045
"""The blade tips' footprint in the tool frame (station model: 15.5 x 3.9 mm, plus margin): points within it,
just outside the grip width along the closing axis, are the blades, not the piece."""

SEARCH_XY_M = 0.008
SEARCH_YAW_DEG = 20.0
"""Search window round the prior pose (a grasp puts the piece within a few mm and degrees of where it was
aimed; the twin's pick slipped 0.4-3.7 degrees)."""

GOOD_FRACTION = 0.97
"""Poses scoring at least this fraction of the best make up the uncertainty spread."""

MIN_POINTS = 60
"""Fewest piece points (of the subsampled depth) for a measurement."""

MAX_SPREAD_M = 0.0015
MAX_SPREAD_DEG = 4.0
"""Spreads beyond these route ``ambiguous`` (too little of the piece is visible to pin it down)."""


class Output(TypedDict, total=False):
    route: str
    tcp_piece: dict[str, float]
    world_pose: dict[str, float]
    sigma: dict[str, float]
    n_piece_points: int
    n_free_points: int
    score: float
    detail: str


def _rot(q: dict) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in "wxyz")
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _pose(reply: Any) -> dict:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def body_centre_px(mask: np.ndarray) -> np.ndarray:
    """Centre of the piece's square body (mask opened by a disk of 0.15 of its side, minimum-area rectangle),
    as selecting-a-grip-clear-of-placed-neighbours computes it."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    side = math.sqrt(float(m.sum()))
    r = max(1, int(round(0.15 * side)))
    body = cv2.morphologyEx(
        m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    )
    pts = np.column_stack(np.nonzero(body if body.any() else m)[::-1]).astype(np.float32)
    (cx, cy), _, _ = cv2.minAreaRect(pts)
    return np.array([cx, cy])


def _wrist(obs: dict, name: str) -> dict | None:
    cams = [
        c
        for c in obs.get("cameras") or []
        if c.get("depth") is not None and c.get("pose") is not None
    ]
    if name:
        return next((c for c in cams if c.get("name") == name), None)
    return next(
        (c for c in cams if any(k in str(c.get("name", "")) for k in ("eye_in_hand", "wrist"))),
        None,
    )


def run(
    ctx: NodeContext,
    piece_mask: Any,
    metres_per_pixel: float,
    prior: dict,
    grip_width_m: float,
    camera: dict | None = None,
    camera_name: str = "",
    image_yaw_deg: float = 0.0,
    thickness: float = THICKNESS_M,
    search_xy_m: float = SEARCH_XY_M,
    search_yaw_deg: float = SEARCH_YAW_DEG,
    arm_id: int = 0,
) -> Output:
    """Where the held piece is relative to the tool centre.

    ``piece_mask`` is the piece's top-down mask at ``metres_per_pixel`` (its +u at ``image_yaw_deg`` in the
    world, -v 90 degrees counter-clockwise from it); the piece's frame is the mask's, centred on the body.
    ``prior`` is where the piece is expected in the world: {"x", "y", "yaw_deg"} of that frame (e.g. the pick
    target and the yaw measured before the pick). ``grip_width_m`` is the jaws' grip width. The wrist frame
    is ``camera`` or a fresh observation's wrist camera. Returns ``tcp_piece`` {x, y, z, yaw_deg}: the body
    centre's position (z: the piece's bottom) and the frame's yaw in the tool frame, ``world_pose`` and
    ``sigma`` (x, y in m, yaw in degrees)."""
    kw = {"arm_id": arm_id} if arm_id else {}
    try:
        tool = _pose(ctx.tool("robot.get_ee_pose", **kw))
        cam = camera or _wrist(ctx.tool("robot.get_observation"), camera_name)
    except Exception as exc:
        return {"route": "not_seen", "detail": f"cannot read the arm or the camera: {exc}"}
    if cam is None:
        return {"route": "not_seen", "detail": "no wrist camera with depth and a pose"}
    m = (np.asarray(piece_mask) > 0).astype(np.uint8)
    if m.sum() < 50 or metres_per_pixel <= 0:
        return {"route": "not_seen", "detail": "empty piece mask or no scale"}

    rt = _rot(tool["rotation"])
    tt = np.array([float(tool["position"][k]) for k in "xyz"])
    d = np.asarray(cam["depth"], dtype=np.float64)[::2, ::2]
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    vv, uu = np.mgrid[0 : d.shape[0], 0 : d.shape[1]] * 2
    ok = np.isfinite(d) & (d > 0)
    pc = np.column_stack(
        [(uu[ok] - k[0, 2]) * d[ok] / k[0, 0], (vv[ok] - k[1, 2]) * d[ok] / k[1, 1], d[ok]]
    )
    rc = _rot(cam["pose"]["rotation"])
    tc = np.array([float(cam["pose"]["position"][a]) for a in "xyz"])
    pw = pc @ rc.T + tc
    # Tool frame: closing axis is the tool's y (linear_4310), the jaw width along its x.
    pt = (pw - tt) @ rt
    depth_below = tt[2] - pw[:, 2]
    blade = (np.abs(pt[:, 0]) <= BLADE_HALF_WIDTH_M) & (
        np.abs(np.abs(pt[:, 1]) - (0.5 * grip_width_m + 0.5 * BLADE_THICKNESS_M))
        <= 0.5 * BLADE_THICKNESS_M + 0.001
    )
    slab = (depth_below > SLAB_M[0]) & (depth_below < SLAB_M[1]) & ~blade
    if slab.sum() < MIN_POINTS:
        return {
            "route": "not_seen",
            "n_piece_points": int(slab.sum()),
            "detail": "too few points under the jaws",
        }
    hist, edges = np.histogram(
        pw[slab, 2], bins=np.arange(tt[2] - SLAB_M[1], tt[2] - SLAB_M[0] + 0.0005, 0.0005)
    )
    z_top = float(edges[int(np.argmax(hist))] + 0.00025)
    piece = (np.abs(pw[:, 2] - z_top) <= TOP_TOL_M) & ~blade & (depth_below > SLAB_M[0])
    free = pw[:, 2] < z_top - FREE_BELOW_M
    if piece.sum() < MIN_POINTS:
        return {
            "route": "not_seen",
            "n_piece_points": int(piece.sum()),
            "detail": "too few points at the piece's top",
        }
    # Free points traced back along their rays to the piece's top plane.
    ray = pw[free] - tc
    s = (z_top - tc[2]) / ray[:, 2]
    fx = tc[:2] + ray[:, :2] * s[:, None]
    near = np.hypot(fx[:, 0] - prior["x"], fx[:, 1] - prior["y"]) < 0.06
    fx = fx[near]
    px = pw[piece, :2]
    px = px[np.hypot(px[:, 0] - prior["x"], px[:, 1] - prior["y"]) < 0.06]
    if len(px) < MIN_POINTS:
        return {
            "route": "not_seen",
            "n_piece_points": int(len(px)),
            "detail": "too few piece points near the prior",
        }

    # Template: the mask, dilated 0.5 mm (the depth's edge blur), in its own frame centred on the body.
    c0 = body_centre_px(m)
    tmpl = cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(np.float32)
    h, w = tmpl.shape
    a0 = math.radians(image_yaw_deg)

    def score(x, y, yaw_deg, pts):
        a = math.radians(yaw_deg) - a0
        rel = pts - [x, y]
        # world -> template-frame metres -> mask pixels (u right, v down)
        qx = rel[:, 0] * math.cos(-a - a0) - rel[:, 1] * math.sin(-a - a0)
        qy = rel[:, 0] * math.sin(-a - a0) + rel[:, 1] * math.cos(-a - a0)
        u = np.round(c0[0] + qx / metres_per_pixel).astype(int)
        v = np.round(c0[1] - qy / metres_per_pixel).astype(int)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        val = np.zeros(len(pts))
        val[inside] = tmpl[v[inside], u[inside]]
        return float(val.sum())

    def total(x, y, yaw):
        return score(x, y, yaw, px) - 0.5 * score(x, y, yaw, fx)

    best, rows = None, []
    for step_xy, step_yaw, rx, ryaw in (
        (0.001, 2.0, search_xy_m, search_yaw_deg),
        (0.00025, 0.5, 0.0015, 3.0),
    ):
        cx, cy, cyaw = (prior["x"], prior["y"], prior["yaw_deg"]) if best is None else best[1:]
        rows = []
        for dyaw in np.arange(-ryaw, ryaw + 1e-9, step_yaw):
            for dx in np.arange(-rx, rx + 1e-12, step_xy):
                for dy in np.arange(-rx, rx + 1e-12, step_xy):
                    sc = total(cx + dx, cy + dy, cyaw + dyaw)
                    rows.append((sc, cx + dx, cy + dy, cyaw + dyaw))
        best = max(rows)
    arr = np.array(rows)
    good = arr[arr[:, 0] >= GOOD_FRACTION * best[0]]
    sig = {
        "x": float(good[:, 1].std()),
        "y": float(good[:, 2].std()),
        "yaw_deg": float(good[:, 3].std()),
    }
    _, bx, by, byaw = best
    # Tool frame: position of the body centre at the piece's bottom, yaw of the piece frame against tool x.
    rel = np.array([bx, by, z_top - thickness]) - tt
    loc = rel @ rt
    tool_heading = math.degrees(math.atan2(rt[1, 0], rt[0, 0]))
    yaw_rel = (byaw - tool_heading + 180.0) % 360.0 - 180.0
    out: Output = {
        "tcp_piece": {
            "x": float(loc[0]),
            "y": float(loc[1]),
            "z": float(loc[2]),
            "yaw_deg": float(yaw_rel),
        },
        "world_pose": {
            "x": float(bx),
            "y": float(by),
            "z": z_top - thickness,
            "yaw_deg": float(byaw),
        },
        "sigma": sig,
        "n_piece_points": int(len(px)),
        "n_free_points": int(len(fx)),
        "score": float(best[0]),
    }
    if max(sig["x"], sig["y"]) > MAX_SPREAD_M or sig["yaw_deg"] > MAX_SPREAD_DEG:
        out.update(
            route="ambiguous",
            detail=f"spread {max(sig['x'], sig['y']) * 1e3:.2f} mm / {sig['yaw_deg']:.1f} deg",
        )
    else:
        out.update(
            route="measured",
            detail=f"piece at ({bx:.4f}, {by:.4f}) yaw {byaw:+.1f} deg, +-{max(sig['x'], sig['y']) * 1e3:.2f} mm / {sig['yaw_deg']:.1f} deg, {len(px)} piece points",
        )
    return out
