# Adapted from dual-arm-cable-manipulation
# (https://github.com/HAORAN808/dual-arm-cable-manipulation, commit 11b4201, MIT,
# Copyright (c) 2026 Haoran Han, Yuxin Fu, Boxian Yao):
# perception/deliverable_perception_min/grasp_detection/core/safe_segments.py
# (apply_endpoint_margin_by_arc_length, split_centerline_by_occlusion,
# filter_short_segments, select_longest_segment) and core/dual_grasp_selector.py
# (apply_inner_margin, build_crossing_forbidden_intervals, subtract_intervals,
# generate_candidate_grasp_segments).
# Changes: the centreline is a world-frame Centerline (metres) resampled every
# 5 mm instead of image pixels; occlusion comes from the tracker's per-node
# visibility, caller-given node indices, or a stretch that reads high over the
# cable's support, instead of the upstream RGB/mask crossing reasoning; the
# cable end a curve.endpoints hint names (the plug) gets its own margin; bends
# (xy curvature) and the open fingers' footprint clearance from obstacle boxes,
# other cables and other stretches of the same cable are new forbidden zones;
# every sliding-window centre is scored and ranked instead of picking the two
# farthest; each is checked with robot.grasp_frame + robot.solve_ik and returned
# as a top-down grasp pose.
"""Rank top-down grasp points along a cable that keep off its ends, crossings, hidden stretches, bends and obstacles."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from gap_core.types import Se3Pose

logger = logging.getLogger(__name__)

STEP_M = 0.005
"""Resampling step along the cable [m]: a quarter of the fingertip window, as upstream slides its 16 px window
by 4 px."""

FINGER_HALF_M = 0.010
"""Half a fingertip's width along the held cable [m] (the YAM station's estimate). The grasp window is twice this,
the stretch the pads cover; upstream's window is 16 px."""

END_MARGIN_M = 0.05
"""Arc length kept free at a bare cable end, and beside a hidden stretch or a crossing [m]. Upstream uses one value
(8 px) for all three. 50 mm: the fitted end is 7-9 mm off the true one on the station's overhead frame
(curve.fit_centerline, smooth 1e-5, issue #10), a fingertip is 20 mm wide, and a short end flops out of a pinch."""

TIP_MARGIN_M = 0.08
"""Arc length kept free at the end a curve.endpoints hint names (the plug) [m]: the station cable's plug housing
measures about 33 mm on the overhead frame (19 px at ~1.75 mm/px), plus the end's fit error and a fingertip."""

VISIBLE_MIN = 0.5
"""A tracked node with visibility under this is hidden (curve.track_centerline's 0..1 per node)."""

MAX_RISE_M = 0.03
"""A stretch reading more than this over the cable's support level (its 10th-percentile height) is treated as
hidden [m]: either the depth sees something lying over the cable (on the station's frame the left gripper finger
read 19 cm where the cable runs under it) or the cable is lifted and swings when touched."""

MAX_CURVATURE = 25.0
"""Sharpest bend a grasp window may sit on [1/m]: bend radius 40 mm. On a tighter bend the tangent the jaws close
across is ill-defined over a 20 mm pad and the cable springs as it is pinched (estimate)."""

CURVE_HALF_M = 0.02
"""Half the chord used to read the bend [m]: wide enough that a few mm of depth jitter is not read as a bend."""

OPEN_WIDTH_M = 0.047
"""How far the jaws open over the cable [m]: the station opens its 94 mm linear_4310 half-way for a cable."""

MIN_CLEARANCE_M = 0.008
"""Free room the open fingers keep from obstacles and other cable [m] (the station's fingertip clearance from the
board's edge when taking a cable)."""

GRASP_HEIGHT_M = 0.0062
"""The tool centre's height over the cable's axis in the returned pose [m]: the station's closed fingertips reach
10.2 mm past the tool centre and pinch 4 mm under the cable's centre."""

REACH_TOLERANCE_M = 0.02
"""An IK solve reporting a larger position error is not a solve (selecting-reachable-grasp's bound)."""

END_SATURATION_M = 0.10
"""Distance past the margin beyond which a farther end no longer raises the score [m]."""

CLEAR_SATURATION_M = 0.05
"""Clearance past the minimum beyond which more room no longer raises the score [m]."""

FOLD_M = 0.02
"""Another stretch of the same cable counts as an obstacle only where the cable has folded back toward the grasp:
its xy arc length from the grasp exceeds the straight-line distance by this much [m] (a fingertip's width). The
cable's own continuation, straight or gently curved, is not something to clear."""

SUPPORT_PERCENTILE = 10.0
"""Which height percentile of the cable is its support level (robust to a stretch read over an occluder)."""


class Output(TypedDict):
    route: str
    pose: Se3Pose | None
    poses: list[Se3Pose]
    grasps: list[dict[str, Any]]
    candidate_count: int
    checked: int
    rejected: dict[str, float]
    detail: str


def _points(centerline: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(centerline, str):
        centerline = json.loads(centerline) if centerline.strip() else []
    meta = centerline if isinstance(centerline, dict) else {}
    pts = meta.get("points", []) if meta else centerline
    arr = np.asarray(pts if pts is not None else [], dtype=float).reshape(-1, 3)
    return arr, meta


def _vec(p: np.ndarray) -> dict[str, float]:
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Closed index intervals where *mask* is true."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points, counter-clockwise (monotone chain)."""
    p = sorted(map(tuple, np.round(pts, 9)))
    if len(p) <= 2:
        return np.asarray(p)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.asarray(lower[:-1] + upper[:-1])


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-18), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons."""
    for poly in (a, b):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.array([-e[1], e[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def _poly_dist(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) >= 3 and len(b) >= 3 and _overlap(a, b):
        return 0.0
    best = math.inf
    for p, q in ((a, b), (b, a)):
        for v in p:
            for i in range(len(q)):
                best = min(best, _seg_dist(v, q[i], q[(i + 1) % len(q)]))
    return best


def _rect_point_dist(
    centre: np.ndarray, t: np.ndarray, n: np.ndarray, ht: float, hn: float, pts: np.ndarray
) -> np.ndarray:
    d = pts - centre
    a = np.maximum(np.abs(d @ t) - ht, 0.0)
    b = np.maximum(np.abs(d @ n) - hn, 0.0)
    return np.hypot(a, b)


def _obb_footprints(obstacles: Any) -> list[tuple[np.ndarray, float]]:
    """Each OrientedBoundingBox's xy footprint (convex hull) and top height."""
    out = []
    for box in obstacles or []:
        c = np.array([box["center"][k] for k in "xyz"], dtype=float)
        e = np.array([box["extent"][k] for k in "xyz"], dtype=float)
        q = box.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
        w, x, y, z = (float(q[k]) for k in "wxyz")
        r = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float
        )
        corners = c + (signs * e) @ r.T
        out.append((_hull(corners[:, :2]), float(corners[:, 2].max())))
    return out


def _resample(s: np.ndarray, step: float) -> np.ndarray:
    grid = np.arange(0.0, s[-1] + 1e-9, step)
    if s[-1] - grid[-1] > 1e-6:
        grid = np.append(grid, s[-1])
    return grid


def _reachable(reply: Any) -> bool:
    if reply is None:
        return False
    if isinstance(reply, dict):
        if any(
            reply.get(k) is not None and not bool(reply[k])
            for k in ("success", "within_servo_tolerance")
        ):
            return False
        for k in ("position_error_m", "position_error", "error_m"):
            if isinstance(reply.get(k), (int, float)) and float(reply[k]) > REACH_TOLERANCE_M:
                return False
    return True


def run(
    ctx: NodeContext,
    centerline: Any,
    endpoints: dict[str, Any] | None = None,
    obstacles: list[dict[str, Any]] | None = None,
    other_cables: list[Any] | None = None,
    crossing_indices: list[int] | None = None,
    occluded_indices: list[int] | None = None,
    arm_id: int = 0,
    end_margin_m: float = END_MARGIN_M,
    tip_margin_m: float = TIP_MARGIN_M,
    max_curvature: float = MAX_CURVATURE,
    max_rise_m: float = MAX_RISE_M,
    open_width_m: float = OPEN_WIDTH_M,
    min_clearance_m: float = MIN_CLEARANCE_M,
    grasp_height_m: float = GRASP_HEIGHT_M,
    max_poses: int = 5,
    min_spacing_m: float = 0.03,
    check_reach: bool = True,
    max_checks: int = 40,
) -> Output:
    """Safe top-down grasp points along a cable, best first, each checked for reach.

    Args:
        centerline: A ``Centerline`` from curve.fit_centerline / track_centerline (or its ``points``, or that as
            JSON text), world frame, metres, in order along the cable.
        endpoints: curve.endpoints' answer for it; when its ``tip`` names an end (the plug), that end keeps
            ``tip_margin_m`` instead of ``end_margin_m``.
        obstacles: OrientedBoundingBoxes the open fingers must keep clear of (a board, ring posts, a fixture).
            A box whose top is below the cable's axis there is what the cable lies on and is ignored.
        other_cables: Other centrelines (same forms as ``centerline``) the fingers must not land on.
        crossing_indices: Node indices of ``centerline`` where another cable crosses it (upstream's crossing points).
        occluded_indices: Node indices known hidden; a tracked centreline's ``visibility`` is used as well.

    Returns:
        ``route`` ``found`` with ``poses`` (best first) and per-grasp metrics in ``grasps``; ``none_safe`` when no
        stretch passes the geometry; ``none_reachable`` when safe stretches exist but IK solved none of them.
    """
    out: Output = {
        "route": "none_safe",
        "pose": None,
        "poses": [],
        "grasps": [],
        "candidate_count": 0,
        "checked": 0,
        "rejected": {},
        "detail": "",
    }
    pts, meta = _points(centerline)
    if len(pts) < 2 or meta.get("ordered") is False:
        out["detail"] = (
            "no ordered centreline (fewer than two points, or the fitter could not order it)"
        )
        return out
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_nodes = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s_nodes[-1])
    radius = float(meta.get("radius_m") or 0.0)

    grid = _resample(s_nodes, STEP_M)
    P = np.stack([np.interp(grid, s_nodes, pts[:, k]) for k in range(3)], axis=1)
    m = len(grid)
    node_of = np.clip(np.searchsorted(s_nodes, grid, side="right") - 1, 0, len(pts) - 1)
    nearest = np.where(
        (node_of + 1 < len(pts))
        & (grid - s_nodes[node_of] > s_nodes[np.minimum(node_of + 1, len(pts) - 1)] - grid),
        node_of + 1,
        node_of,
    )

    # Per-end margins: upstream's apply_endpoint_margin_by_arc_length, with the plug end's own margin.
    tip = int((endpoints or {}).get("tip", -1)) if isinstance(endpoints, dict) else -1
    margin_a = tip_margin_m if tip == 0 else end_margin_m
    margin_b = tip_margin_m if tip == 1 else end_margin_m
    near_end = (grid < margin_a) | (total - grid < margin_b)

    # Hidden: tracker visibility, given indices, or a stretch reading high over the support.
    hidden = np.zeros(m, dtype=bool)
    vis = meta.get("visibility")
    if vis is not None and len(vis) == len(pts):
        hidden |= np.interp(grid, s_nodes, np.asarray(vis, dtype=float)) < VISIBLE_MIN
    if occluded_indices:
        hidden |= np.isin(nearest, np.asarray(list(occluded_indices), dtype=int))
    support = float(np.percentile(P[:, 2], SUPPORT_PERCENTILE))
    raised = P[:, 2] - support > max_rise_m
    hidden |= raised

    # Upstream splits the margin'd range at hidden points, drops short pieces, and keeps a margin beside each
    # break (apply_inner_margin); crossings forbid a radius around them (build_crossing_forbidden_intervals).
    forbidden_break = np.zeros(m, dtype=bool)
    for a, b in _runs(hidden):
        # Measured from the last visible sample either side: the true edge lies somewhere in that step.
        lo, hi = grid[max(a - 1, 0)], grid[min(b + 1, m - 1)]
        forbidden_break |= (grid >= lo - end_margin_m) & (grid <= hi + end_margin_m)
    forbidden_cross = np.zeros(m, dtype=bool)
    for idx in crossing_indices or []:
        if 0 <= int(idx) < len(pts):
            forbidden_cross |= np.abs(grid - s_nodes[int(idx)]) <= end_margin_m

    # Bends: turning angle of the xy chords over +-CURVE_HALF_M, per metre.
    def at(sv: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(sv, s_nodes, pts[:, k]) for k in range(2)], axis=1)

    back = P[:, :2] - at(np.clip(grid - CURVE_HALF_M, 0.0, total))
    ahead = at(np.clip(grid + CURVE_HALF_M, 0.0, total)) - P[:, :2]
    nb, na = np.linalg.norm(back, axis=1), np.linalg.norm(ahead, axis=1)
    ok = (nb > 1e-6) & (na > 1e-6)
    cosang = np.where(ok, np.einsum("ij,ij->i", back, ahead) / np.maximum(nb * na, 1e-12), 1.0)
    kappa = np.arccos(np.clip(cosang, -1.0, 1.0)) / CURVE_HALF_M
    chord = at(np.clip(grid + CURVE_HALF_M, 0.0, total)) - at(
        np.clip(grid - CURVE_HALF_M, 0.0, total)
    )
    tangent = chord / np.maximum(np.linalg.norm(chord, axis=1, keepdims=True), 1e-12)

    # Clearance of the open fingers' footprint: FINGER_HALF_M along the cable, open_width_m / 2 across it.
    boxes = _obb_footprints(obstacles)
    others = []
    for cab in other_cables or []:
        q, _ = _points(cab)
        if len(q) >= 2:
            sq = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
            gq = _resample(sq, STEP_M)
            others.append(np.stack([np.interp(gq, sq, q[:, k]) for k in range(2)], axis=1))
    other_pts = np.concatenate(others) if others else np.zeros((0, 2))
    self_arc = 0.5 * open_width_m + FINGER_HALF_M + min_clearance_m
    s_xy = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))])
    hn = 0.5 * open_width_m
    clearance = np.full(m, math.inf)
    for i in range(m):
        c, t = P[i, :2], tangent[i]
        n = np.array([-t[1], t[0]])
        best = math.inf
        if boxes:
            rect = np.array(
                [
                    c + sa * FINGER_HALF_M * t + sb * hn * n
                    for sa, sb in ((1, 1), (-1, 1), (-1, -1), (1, -1))
                ]
            )
            for poly, top in boxes:
                if top > P[i, 2]:
                    best = min(best, _poly_dist(rect, poly) if len(poly) >= 3 else math.inf)
        along = np.abs(s_xy - s_xy[i])
        folded = along - np.linalg.norm(P[:, :2] - c, axis=1) > FOLD_M
        # A raised stretch's xy is where the depth met whatever lies over the cable, not the cable.
        far = (np.abs(grid - grid[i]) > self_arc) & folded & ~raised
        cand = [P[far, :2]] if far.any() else []
        if len(other_pts):
            cand.append(other_pts)
        if cand:
            d = _rect_point_dist(c, t, n, FINGER_HALF_M, hn, np.concatenate(cand))
            if len(d):
                best = min(best, float(d.min()) - radius)
        clearance[i] = best

    bend = kappa > max_curvature
    tight = clearance < min_clearance_m
    point_ok = ~(near_end | hidden | forbidden_break | forbidden_cross | bend | tight)

    # Upstream's sliding window: a centre is a candidate when the whole fingertip window is legal.
    half = int(round(FINGER_HALF_M / STEP_M))
    window_ok = point_ok.copy()
    for k in range(1, half + 1):
        window_ok[k:] &= point_ok[:-k]
        window_ok[:-k] &= point_ok[k:]
    window_ok[:half] = False
    window_ok[m - half :] = False

    # Visible pieces (between hidden stretches, inside the end margins): upstream prefers the longest.
    visible = ~(near_end | hidden)
    pieces = [(a, b) for a, b in _runs(visible) if grid[b] - grid[a] >= 2 * FINGER_HALF_M]
    piece_len = np.zeros(m)
    for a, b in pieces:
        piece_len[a : b + 1] = grid[b] - grid[a]
    longest = max((grid[b] - grid[a] for a, b in pieces), default=0.0)

    n_pts = float(m)
    out["rejected"] = {
        "near_end": round(float(near_end.sum()) / n_pts, 3),
        "hidden": round(float(hidden.sum()) / n_pts, 3),
        "beside_hidden": round(float((forbidden_break & ~hidden).sum()) / n_pts, 3),
        "crossing": round(float(forbidden_cross.sum()) / n_pts, 3),
        "bend": round(float(bend.sum()) / n_pts, 3),
        "clearance": round(float(tight.sum()) / n_pts, 3),
    }

    idx = np.flatnonzero(window_ok)
    out["candidate_count"] = int(len(idx))
    if not len(idx):
        out["detail"] = (
            f"no {2e3 * FINGER_HALF_M:.0f} mm stretch of the {total:.2f} m cable passes; share of it ruled out: "
            + ", ".join(f"{k} {v:.0%}" for k, v in out["rejected"].items() if v > 0)
        )
        return out

    win = np.arange(-half, half + 1)
    rows = []
    for i in idx:
        w = i + win
        k_max = float(kappa[w].max())
        clear = float(clearance[w].min())
        d_end = min(grid[i] - margin_a, total - grid[i] - margin_b)
        score = (
            min(max(d_end, 0.0) / END_SATURATION_M, 1.0)
            + (1.0 - min(k_max / max_curvature, 1.0))
            + (
                1.0
                if math.isinf(clear)
                else min((clear - min_clearance_m) / CLEAR_SATURATION_M, 1.0)
            )
            + (piece_len[i] / longest if longest > 0 else 0.0)
        ) / 4.0
        rows.append((score, int(i), k_max, clear, d_end))
    rows.sort(
        key=lambda r: (-round(r[0], 9), -r[4])
    )  # ties (saturated terms) go to the farther from an end

    on_arm = {"arm_id": int(arm_id)} if int(arm_id) else {}
    chosen: list[dict[str, Any]] = []
    taken: list[float] = []
    checked = 0
    why = ""
    for score, i, k_max, clear, _ in rows:
        if len(chosen) >= max_poses or (check_reach and checked >= max_checks):
            break
        if any(abs(grid[i] - s) < min_spacing_m for s in taken):
            continue
        t = tangent[i]
        heading = math.degrees(math.atan2(t[1], t[0])) + 90.0
        z = float(np.median(P[i + win, 2])) + float(grasp_height_m)
        position = np.array([P[i, 0], P[i, 1], z])
        try:
            frame = ctx.tool("robot.grasp_frame", close_heading_deg=heading, **on_arm)
            rotations = [
                (k, frame[k])
                for k in ("rotation", "rotation_flipped")
                if isinstance(frame, dict) and frame.get(k)
            ]
        except Exception as exc:
            why = f"robot.grasp_frame: {exc}"
            rotations = []
        if not rotations:
            checked += 1
            continue
        used = None
        if check_reach:
            checked += 1
            for name, rot in rotations:
                try:
                    reply = ctx.tool(
                        "robot.solve_ik",
                        pose={"position": _vec(position), "rotation": rot},
                        **on_arm,
                    )
                except Exception as exc:
                    why = f"robot.solve_ik: {exc}"
                    continue
                if _reachable(reply):
                    used = (name, rot)
                    break
        else:
            used = rotations[0]
        if used is None:
            continue
        taken.append(float(grid[i]))
        pose: Se3Pose = {"position": _vec(position), "rotation": used[1]}
        chosen.append(
            {
                "pose": pose,
                "point": _vec(P[i]),
                "tangent": _vec(np.array([t[0], t[1], 0.0])),
                "close_heading_deg": round(heading, 2),
                "rotation_used": used[0],
                "arc_m": round(float(grid[i]), 4),
                "node": int(nearest[i]),
                "end_distance_m": round(float(min(grid[i], total - grid[i])), 4),
                "tip_distance_m": None
                if tip not in (0, 1)
                else round(float(grid[i] if tip == 0 else total - grid[i]), 4),
                "curvature": round(k_max, 2),
                "clearance_m": None if math.isinf(clear) else round(clear, 4),
                "segment_m": round(float(piece_len[i]), 4),
                "score": round(float(score), 4),
            }
        )
    out["checked"] = checked
    out["grasps"] = chosen
    out["poses"] = [g["pose"] for g in chosen]
    if not chosen:
        out["route"] = "none_reachable"
        out["detail"] = (
            f"{len(idx)} safe window(s) on the {total:.2f} m cable, but IK reached none of the "
            f"{checked} best checked" + (f" (last: {why})" if why else "")
        )
        return out
    out["route"] = "found"
    out["pose"] = chosen[0]["pose"]
    out["detail"] = (
        f"{len(chosen)} grasp(s) from {len(idx)} safe window(s) on the {total:.2f} m cable"
    )
    return out
