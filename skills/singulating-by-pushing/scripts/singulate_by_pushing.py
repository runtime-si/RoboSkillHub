# The push primitive is adapted from visual-pushing-grasping
# (https://github.com/andyzeng/visual-pushing-grasping, robot.py, Robot.push),
# BSD-2-Clause, Copyright (c) 2018 Andy Zeng: closed hand, hover a margin above
# the push point, straight down, one straight push of a bounded length kept
# inside the workspace limits, straight up. Its hover / descent / push / lift
# moves follow the hub's pushing-objects (adapted from RoboRSI, Apache-2.0).
# Changes: the push point and direction come from world-frame OBBs instead of a
# learned heightmap policy; the push is short (sized to free an open gripper,
# capped at push_step) and repeated with a re-check; robot.* connector tools;
# fingers across the push; move refusals routed rather than raised.
# The direction choice (toward the nearest point of the most free space around
# the target) is reimplemented from Kiatos et al. 2021, "Improved Reinforcement
# Learning Pushing Policies via Heuristic Rules" (arXiv 2111.11156); its code
# (github.com/robot-clutter/improved_rl) has no licence and was only read.
"""Push a target clear of touching neighbours until an open gripper's fingers fit around it."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

OPENING = 0.094
"""Jaw opening [m] when robot.describe_gripper states none: the YAM linear_4310's
measured 94 mm span (a real station's describe_gripper)."""

FINGER_THICKNESS = 0.010
"""One fingertip's depth along the closing axis [m] when describe_gripper states
none (the linear_4310 reports pad_thickness 0, unmeasured). An estimate of the
tapered blade near its tip."""

FINGER_WIDTH = 0.020
"""A fingertip's width across the closing axis [m]; estimate, as above."""

FIT_MARGIN = 0.003
"""Free space a fingertip needs on every side of its footprint [m]."""

PLAN_SLACK = 0.004
"""Extra clearance a push aims for over FIT_MARGIN [m], for the object's slip."""

HAND_RADIUS = 0.010
"""Radius of the closed fingertips' footprint around the tool centre [m] at push
height; estimate (two closed blades side by side)."""

HAND_MARGIN = 0.003
"""Free space the closed hand keeps from every neighbour on its way down and along the push [m]."""

PUSH_STEP = 0.04
"""Longest single push [m]. Kiatos et al. clip a push to 0.03-0.15 m and VPG pushes
0.10 m; short pushes, each re-checked, keep a neighbour from being dragged far."""

MIN_PUSH = 0.01
"""Shortest push worth making [m]."""

MAX_PUSHES = 3

STANDOFF = 0.02
"""Free space behind the push point where the closed hand comes down [m]. Shorter
than pushing-objects' 60 mm so the descent fits between neighbours."""

EDGE_PADDING = 0.003
"""Gap between the tool centre and the object's back face at contact [m] (RoboRSI, via pushing-objects)."""

HOVER_CLEARANCE = 0.08
"""Height of the hover over the push height [m] (VPG hovers 0.10 m; pushing-objects 0.08 m)."""

CONTACT_FRACTION = 0.5
MIN_CONTACT_HEIGHT = 0.015
"""Lowest tool-centre height over the support [m]: a real YAM station refuses a
tool centre under 14 mm over its calibrated table."""

REACH_TOLERANCE = 0.02
"""How far the hand may end from the hover or the lowered pose [m] (pushing-objects)."""

FREE_SPACE_RADIUS = 0.15
"""Radius of the window around the target searched for free space [m]: Kiatos et
al.'s longest push."""

FREE_SPACE_STEP = 0.005
FREE_SPACE_KEEP = 0.9
"""Points within this fraction of the most free space count as 'most free'; the
nearest of them sets the direction (Kiatos et al.'s 0.9 x max rule)."""

DIRECTIONS = 36
"""Candidate push headings, every 10 degrees, tried nearest the free-space heading first."""

LENGTH_STEP = 0.002
"""Resolution of the push-length search [m]."""

SLIDE_TOLERANCE = 0.001
"""How far the target may sink into a neighbour's box along its path [m]: a push
may slide it along a neighbour it touches, not into it."""


class Output(TypedDict):
    route: str
    detail: str
    pushes: int
    gap_before: float
    gap_after: float
    clearance_before: float
    clearance_after: float
    fits_before: bool
    fits_after: bool
    required_gap: float
    expected_target_obb: dict[str, Any]


# ----------------------------------------------------------------- geometry


def _rotation(obb: dict[str, Any]) -> np.ndarray:
    q = obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
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


def _centre(obb: dict[str, Any]) -> np.ndarray:
    return np.array([float(obb["center"][k]) for k in ("x", "y", "z")])


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull, counter-clockwise (monotone chain)."""
    pts = sorted({(round(float(p[0]), 9), round(float(p[1]), 9)) for p in points})
    if len(pts) <= 2:
        return np.array(pts, dtype=float)

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def footprint(obb: dict[str, Any]) -> np.ndarray:
    """The box's outline on the table: the hull of its 8 corners projected on xy."""
    half = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = (signs * half) @ _rotation(obb).T + _centre(obb)
    return _hull(corners[:, :2])


def _point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _edges(poly: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(poly) == 1:
        return [(poly[0], poly[0])]
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def signed_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two convex outlines (points, a segment or polygons) [m];
    negative is the depth they overlap by (separating axis test)."""
    axes = []
    for poly in (a, b):
        for p, q in _edges(poly):
            e = q - p
            if np.linalg.norm(e) > 1e-12:
                axes.append(np.array([-e[1], e[0]]) / np.linalg.norm(e))
    overlap = math.inf
    for n in axes:
        pa, pb = a @ n, b @ n
        overlap = min(overlap, min(pa.max(), pb.max()) - max(pa.min(), pb.min()))
    if axes and overlap > 0.0:
        return -overlap
    best = math.inf
    for src, dst in ((a, b), (b, a)):
        for p in src:
            for s, e in _edges(dst):
                best = min(best, _point_segment(p, s, e))
    return 0.0 if not axes and best < 1e-12 else best


def _rect(
    centre: np.ndarray, u: np.ndarray, along: tuple[float, float], half_across: float
) -> np.ndarray:
    v = np.array([-u[1], u[0]])
    lo, hi = along
    return np.array(
        [
            centre + u * lo - v * half_across,
            centre + u * hi - v * half_across,
            centre + u * hi + v * half_across,
            centre + u * lo + v * half_across,
        ]
    )


def grasp_axes(obb: dict[str, Any]) -> list[np.ndarray]:
    """The box's two most horizontal axes, as unit xy headings (top-down closing directions)."""
    rot = _rotation(obb)
    order = np.argsort(np.abs(rot[2, :]))[:2]
    out = []
    for i in order:
        u = rot[:2, i]
        n = float(np.linalg.norm(u))
        if n > 1e-6:
            out.append(u / n)
    return out


def finger_clearance(
    target: np.ndarray,
    centre: np.ndarray,
    u: np.ndarray,
    others: list[np.ndarray],
    hand: dict[str, float],
) -> float:
    """Least distance [m] from the two open fingertips of a top-down grasp closing
    along ``u`` at ``centre`` to any neighbour; -inf when the target does not fit
    between the open jaws along ``u``."""
    proj = (target - centre) @ u
    inner = hand["opening"] / 2.0
    if max(-proj.min(), proj.max()) >= inner:
        return -math.inf
    best = math.inf
    for side in (1.0, -1.0):
        tip = _rect(centre, u * side, (inner, inner + hand["thickness"]), hand["width"] / 2.0)
        for other in others:
            best = min(best, signed_distance(tip, other))
    return best


def clearance(
    target: np.ndarray,
    centre: np.ndarray,
    axes: list[np.ndarray],
    others: list[np.ndarray],
    hand: dict,
    need: str,
) -> float:
    values = [
        v
        for v in (finger_clearance(target, centre, u, others, hand) for u in axes)
        if v > -math.inf
    ]
    if not values:  # too wide for the jaws along every axis
        return -math.inf
    return max(values) if need == "any" else min(values)


def nearest_gap(target: np.ndarray, others: list[np.ndarray]) -> float:
    return min((signed_distance(target, o) for o in others), default=math.inf)


def _outline_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point to a convex polygon (0 inside)."""
    d = np.full(len(points), np.inf)
    inside = np.ones(len(points), dtype=bool)
    for p, q in _edges(poly):
        e = q - p
        ee = float(np.dot(e, e)) or 1e-12
        t = np.clip(((points - p) @ e) / ee, 0.0, 1.0)
        d = np.minimum(d, np.linalg.norm(points - (p + t[:, None] * e), axis=1))
        inside &= (e[0] * (points[:, 1] - p[1]) - e[1] * (points[:, 0] - p[0])) >= 0.0
    return np.where(inside, 0.0, d)


def free_space_heading(
    centre: np.ndarray, others: list[np.ndarray], workspace: dict | None
) -> float:
    """Heading [rad] toward the nearest of the most-free points around the target:
    free space at a point is its distance to the nearest neighbour (and to the
    workspace edge), Kiatos et al.'s heuristic on outlines instead of a heightmap."""
    r = np.arange(-FREE_SPACE_RADIUS, FREE_SPACE_RADIUS + 1e-9, FREE_SPACE_STEP)
    gx, gy = np.meshgrid(r, r)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    pts = pts[np.linalg.norm(pts, axis=1) <= FREE_SPACE_RADIUS] + centre
    free = np.full(len(pts), np.inf)
    for o in others:
        free = np.minimum(free, _outline_distance(pts, o))
    if workspace:
        edge = np.minimum.reduce(
            [
                pts[:, 0] - float(workspace["x_min"]),
                float(workspace["x_max"]) - pts[:, 0],
                pts[:, 1] - float(workspace["y_min"]),
                float(workspace["y_max"]) - pts[:, 1],
            ]
        )
        free = np.minimum(free, edge)
    if not np.isfinite(free).any():
        return 0.0
    keep = free >= FREE_SPACE_KEEP * free.max()
    near = pts[keep][np.argmin(np.linalg.norm(pts[keep] - centre, axis=1))]
    off = near - centre
    return math.atan2(off[1], off[0]) if np.linalg.norm(off) > 1e-9 else 0.0


def _inside(poly: np.ndarray, workspace: dict | None) -> bool:
    if not workspace:
        return True
    return bool(
        poly[:, 0].min() >= float(workspace["x_min"])
        and poly[:, 0].max() <= float(workspace["x_max"])
        and poly[:, 1].min() >= float(workspace["y_min"])
        and poly[:, 1].max() <= float(workspace["y_max"])
    )


def plan_push(
    target: np.ndarray,
    centre: np.ndarray,
    axes: list[np.ndarray],
    others: list[np.ndarray],
    hand: dict[str, float],
    need: str,
    push_step: float,
    standoff: float,
    workspace: dict | None,
) -> dict[str, Any] | None:
    """The push to make, or None when the hand can come down behind the target in no direction.

    Headings are tried nearest the free-space heading first. A heading is usable
    when the closed hand, on its way down ``standoff`` behind the push point and
    along the push, keeps HAND_MARGIN from every neighbour, and the target's path
    does not sink into one. The chosen push is the nearest-heading one that makes
    the fingers fit (shortest length that does); failing that, the one leaving the
    most finger clearance after ``push_step``.
    """
    heading = free_space_heading(centre, others, workspace)
    gaps0 = [signed_distance(target, o) for o in others]
    aim = FIT_MARGIN + PLAN_SLACK
    turns = sorted(
        (2.0 * math.pi * i / DIRECTIONS for i in range(DIRECTIONS)),
        key=lambda a: abs(math.remainder(a - heading, 2.0 * math.pi)),
    )
    best_partial: dict[str, Any] | None = None
    for angle in turns:
        d = np.array([math.cos(angle), math.sin(angle)])
        back = max(0.0, -float(((target - centre) @ d).min())) + EDGE_PADDING
        contact = centre - d * back
        lowered = contact - d * standoff
        lengths = np.arange(MIN_PUSH, push_step + 1e-9, LENGTH_STEP)
        usable = None
        for length in lengths:
            moved = target + d * length
            if not _inside(moved, workspace):
                break
            if any(
                signed_distance(moved, o) < min(g, 0.0) - SLIDE_TOLERANCE
                for o, g in zip(others, gaps0, strict=True)
            ):
                break
            sweep = np.array([lowered, contact + d * length])
            if any(signed_distance(sweep, o) - hand["radius"] < HAND_MARGIN for o in others):
                break
            usable = float(length)
            value = clearance(moved, centre + d * length, axes, others, hand, need)
            if value >= aim:
                return {
                    "heading": angle,
                    "free_heading": heading,
                    "direction": d,
                    "contact": contact,
                    "lowered": lowered,
                    "length": usable,
                    "clearance": value,
                    "fits": True,
                }
        if usable is None:
            continue
        moved = target + d * usable
        value = clearance(moved, centre + d * usable, axes, others, hand, need)
        candidate = {
            "heading": angle,
            "free_heading": heading,
            "direction": d,
            "contact": contact,
            "lowered": lowered,
            "length": usable,
            "clearance": value,
            "fits": False,
        }
        if best_partial is None or value > best_partial["clearance"] + 1e-3:
            best_partial = candidate
    return best_partial


# ------------------------------------------------------------------- motion


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or "" (connectors that route a
    refusal instead of raising)."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def _hand(
    ctx: NodeContext,
    opening: float,
    finger_thickness: float,
    finger_width: float,
    hand_radius: float,
) -> dict:
    try:
        g = ctx.tool("robot.describe_gripper") or {}
    except Exception as exc:  # the fallbacks below are the YAM's
        logger.info("singulating: no gripper description (%s)", exc)
        g = {}
    fit = g.get("width_fit") or {}
    finger = g.get("finger") or {}
    stated = float(finger.get("pad_thickness_m") or 0.0) if finger.get("stated") else 0.0
    return {
        "opening": float(opening or fit.get("at_open_m") or g.get("span_m") or OPENING),
        "thickness": float(finger_thickness or stated or FINGER_THICKNESS),
        "width": float(finger_width),
        "radius": float(hand_radius),
    }


def run(
    ctx: NodeContext,
    target_obb: dict[str, Any],
    neighbour_obb: dict[str, Any] | None = None,
    neighbour_obbs: list[dict[str, Any]] | None = None,
    need: str = "both",
    grasp_heading_deg: float | None = None,
    opening: float = 0.0,
    finger_thickness: float = 0.0,
    finger_width: float = FINGER_WIDTH,
    hand_radius: float = HAND_RADIUS,
    push_step: float = PUSH_STEP,
    max_pushes: int = MAX_PUSHES,
    standoff: float = STANDOFF,
    hover_clearance: float = HOVER_CLEARANCE,
    reach_tolerance: float = REACH_TOLERANCE,
    workspace: dict[str, float] | None = None,
    min_contact_height: float = MIN_CONTACT_HEIGHT,
) -> Output:
    hand = _hand(ctx, opening, finger_thickness, finger_width, hand_radius)
    target = footprint(target_obb)
    centre3 = _centre(target_obb)
    centre = centre3[:2].copy()
    boxes = ([neighbour_obb] if neighbour_obb else []) + list(neighbour_obbs or [])
    others = [footprint(o) for o in boxes]
    if grasp_heading_deg is not None:
        a = math.radians(float(grasp_heading_deg))
        axes = [np.array([math.cos(a), math.sin(a)])]
    else:
        axes = grasp_axes(target_obb)
    narrow = min((float(np.ptp(target @ u)) for u in axes), default=0.0)

    def gap_now() -> float:
        g = nearest_gap(target, others)
        return g if math.isfinite(g) else 1.0

    def clear_now() -> float:
        c = clearance(target, centre, axes, others, hand, need) if others else 1.0
        return c if math.isfinite(c) else (1.0 if c > 0 else -1.0)

    out: Output = {
        "route": "still_cluttered",
        "detail": "",
        "pushes": 0,
        "gap_before": gap_now(),
        "gap_after": gap_now(),
        "clearance_before": clear_now(),
        "clearance_after": clear_now(),
        "fits_before": clear_now() >= FIT_MARGIN,
        "fits_after": clear_now() >= FIT_MARGIN,
        "required_gap": (hand["opening"] - narrow) / 2.0 + hand["thickness"] + FIT_MARGIN,
        "expected_target_obb": target_obb,
    }

    def finish(route: str, detail: str) -> Output:
        out["route"] = route
        out["detail"] = detail
        out["gap_after"] = gap_now()
        out["clearance_after"] = clear_now()
        out["fits_after"] = out["clearance_after"] >= FIT_MARGIN
        moved = np.array([*centre, centre3[2]])
        out["expected_target_obb"] = {**target_obb, "center": _vec(moved)}
        return out

    if out["fits_before"]:
        return finish("singulated", "the open fingers already fit; nothing pushed")
    if all(finger_clearance(target, centre, u, [], hand) == -math.inf for u in axes):
        return finish(
            "unreachable",
            f"the target is wider than the {hand['opening'] * 1e3:.0f} mm opening along every grasp axis",
        )

    height = 2.0 * float(target_obb["extent"]["z"])
    push_z = centre3[2] - height / 2.0 + max(CONTACT_FRACTION * height, float(min_contact_height))

    def move(tool: str, position: np.ndarray, rotation: dict[str, float]) -> str:
        try:
            return _refusal(ctx.tool(tool, pose={"position": _vec(position), "rotation": rotation}))
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc)

    closed = False
    try:
        for _ in range(int(max_pushes)):
            plan = plan_push(
                target,
                centre,
                axes,
                others,
                hand,
                need,
                float(push_step),
                float(standoff),
                workspace,
            )
            if plan is None:
                return finish(
                    "unreachable" if out["pushes"] == 0 else "still_cluttered",
                    f"after {out['pushes']} push(es) the closed hand fits behind the target in no direction "
                    f"(needs {2 * (hand['radius'] + HAND_MARGIN) * 1e3:.0f} mm of free space)",
                )
            if not plan["fits"] and plan["clearance"] <= clear_now() + 1e-3:
                return finish(
                    "still_cluttered",
                    f"after {out['pushes']} push(es) no usable push direction increases the fingers' clearance",
                )
            d = plan["direction"]
            contact = np.array([*plan["contact"], push_z])
            lowered = np.array([*plan["lowered"], push_z])
            hover = lowered + np.array([0.0, 0.0, float(hover_clearance)])
            end = contact + np.array([*d, 0.0]) * plan["length"]
            logger.info(
                "singulating: push %d heading %.0f deg (free space at %.0f deg) for %.0f mm",
                out["pushes"] + 1,
                math.degrees(plan["heading"]),
                math.degrees(plan["free_heading"]),
                plan["length"] * 1e3,
            )

            start = _pose(ctx.tool("robot.get_ee_pose"))
            if not closed:
                ctx.tool("robot.close_gripper")
                closed = True
            rotations = []
            try:
                frame = ctx.tool(
                    "robot.grasp_frame", close_heading_deg=math.degrees(plan["heading"]) + 90.0
                )
                rotations = [
                    frame[k]
                    for k in ("rotation", "rotation_flipped")
                    if isinstance(frame, dict) and frame.get(k)
                ]
            except Exception as exc:
                logger.info("singulating: no top-down rotation (%s)", exc)
            rotations.append(start["rotation"])
            rotation, why = None, "no rotation reached the hover"
            for candidate in rotations:
                why = ""
                if float(_pose(ctx.tool("robot.get_ee_pose"))["position"]["z"]) < hover[2] - 1.0e-3:
                    rise = _xyz(_pose(ctx.tool("robot.get_ee_pose")))
                    rise[2] = hover[2]
                    why = move("robot.go_to_pose", rise, candidate)
                why = why or move("robot.go_to_pose", hover, candidate)
                off = float(np.linalg.norm(_xyz(_pose(ctx.tool("robot.get_ee_pose"))) - hover))
                if not why and off <= float(reach_tolerance):
                    rotation = candidate
                    break
                why = why or f"the hand ended {off * 1e3:.0f} mm from the hover"
            if rotation is None:
                return finish("unreachable", f"push {out['pushes'] + 1} hover: {why}")

            why = move("robot.go_to_pose_cartesian", lowered, rotation)
            reached = _xyz(_pose(ctx.tool("robot.get_ee_pose")))
            off = float(np.linalg.norm(reached - lowered))
            if off > float(reach_tolerance):
                return finish(
                    "unreachable",
                    f"push {out['pushes'] + 1} descent: {why or 'ended off'}; "
                    f"the hand stopped {off * 1e3:.0f} mm from the lowered pose",
                )
            if why:  # closed fingertips on the support: push at the height reached
                contact[2] = end[2] = reached[2]

            pushed_why = move("robot.go_to_pose_cartesian", end, rotation)
            stop = _xyz(_pose(ctx.tool("robot.get_ee_pose")))
            achieved = float(np.clip(np.dot(stop[:2] - contact[:2], d), 0.0, plan["length"]))
            lift_why = move(
                "robot.go_to_pose_cartesian",
                stop + np.array([0.0, 0.0, float(hover_clearance)]),
                rotation,
            )
            out["pushes"] += 1
            target = target + d * achieved
            centre = centre + d * achieved
            if lift_why:
                return finish("unreachable", f"push {out['pushes']} lift: {lift_why}")
            if achieved < 0.5 * plan["length"]:
                return finish(
                    "still_cluttered",
                    f"push {out['pushes']} travelled {achieved * 1e3:.0f} of {plan['length'] * 1e3:.0f} mm"
                    + (f" ({pushed_why})" if pushed_why else ""),
                )
            if clear_now() >= FIT_MARGIN:
                return finish(
                    "singulated",
                    f"{out['pushes']} push(es); by the hand's travel the open fingers now fit "
                    "(re-perceive before grasping)",
                )
        return finish(
            "still_cluttered", f"{out['pushes']} push(es) and the open fingers still do not fit"
        )
    finally:
        if closed:
            ctx.tool("robot.open_gripper")
