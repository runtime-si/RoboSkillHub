"""Find an opening by probing down over a hexagonal grid around its estimate, no force sensor."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

SEARCH_RADIUS = 0.006
"""Radius of the probed region around the estimate [m]: about two standard
deviations of a calibrated YAM station's hand-camera error (RMS 5.4 mm over the
workspace, 2-3 mm near the table)."""

PITCH = 0.002
"""Spacing of the hexagonal grid [m]. A hexagonal grid leaves every point of the
region within pitch / sqrt(3) of a probe, so a part with at least that much
clearance in its opening is entered by some probe; 2 mm (1.15 mm worst case)
makes 37 probes inside the default radius."""

HOVER = 0.01
"""Height over the mouth each probe starts from [m]."""

ENTER_DEPTH = 0.005
"""How far below the mouth a probe must get to count as in [m]."""

MAX_DEPTH = 0.015
"""How far below the mouth a probe is sent [m]; it stops earlier on contact."""

HOVER_TOLERANCE = 0.003
"""How far the hand may end from a probe's hover point [m] before the probe is
skipped: a probe started off its point tests a different point."""


class Output(TypedDict):
    route: str
    detail: str
    entered_at: dict[str, float]
    depth: float
    probes: int


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def hex_grid(radius: float, pitch: float) -> list[tuple[float, float]]:
    """Offsets of a hexagonal grid inside ``radius``, nearest the centre first
    (ties broken by angle, so a ring is walked round rather than zig-zagged)."""
    if pitch <= 0.0:
        return [(0.0, 0.0)]
    n = int(math.ceil(radius / pitch)) + 1
    points = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = pitch * (i + 0.5 * j)
            y = pitch * (math.sqrt(3.0) / 2.0) * j
            if math.hypot(x, y) <= radius + 1.0e-9:
                points.append((x, y))
    return sorted(
        points, key=lambda p: (round(math.hypot(*p), 9), math.atan2(p[1], p[0]) % (2 * math.pi))
    )


def run(
    ctx: NodeContext,
    mouth_position: dict[str, float],
    search_radius: float = SEARCH_RADIUS,
    pitch: float = PITCH,
    hover: float = HOVER,
    enter_depth: float = ENTER_DEPTH,
    max_depth: float = MAX_DEPTH,
    stay_in: bool = True,
    check_reach: bool = True,
) -> Output:
    mouth = np.array([float(mouth_position[k]) for k in ("x", "y", "z")])
    out: Output = {
        "route": "search_exhausted",
        "detail": "",
        "entered_at": _vec(mouth),
        "depth": 0.0,
        "probes": 0,
    }
    rotation = _pose(ctx.tool("robot.get_ee_pose"))["rotation"]

    def pose(p: np.ndarray) -> dict[str, Any]:
        return {"pose": {"position": _vec(p), "rotation": rotation}}

    def where() -> np.ndarray | None:
        try:
            return _xyz(_pose(ctx.tool("robot.get_ee_pose")))
        except Exception as exc:
            logger.info("searching-insertion-spiral: no pose reading (%s)", exc)
            return None

    def out_of_reach(p: np.ndarray) -> bool:
        """Whether the IK says this depth cannot be reached. A stall at
        the edge of reach looks exactly like contact, so a probe the arm cannot
        finish is skipped rather than read as having met a surface. A connector
        without a usable answer gets the benefit of the doubt."""
        try:
            reply = ctx.tool("robot.solve_ik", **pose(p))
        except Exception as exc:
            logger.info("searching-insertion-spiral: no IK preview (%s)", exc)
            return False
        if not isinstance(reply, dict):
            return False
        return reply.get("success") is False or reply.get("within_servo_tolerance") is False

    best, skipped = -math.inf, 0
    for k, (dx, dy) in enumerate(hex_grid(float(search_radius), float(pitch))):
        above = mouth + np.array([dx, dy, float(hover)])
        down = mouth + np.array([dx, dy, -float(max_depth)])
        # Preview the depth that proves entry, not the probe's full travel: an arm
        # that reaches past enter_depth and stalls lower has still proved it.
        if check_reach and out_of_reach(mouth + np.array([dx, dy, -float(enter_depth)])):
            logger.info("searching-insertion-spiral: probe %d out of reach, skipped", k)
            skipped += 1
            continue
        try:
            ctx.tool("robot.go_to_pose_cartesian", **pose(above))
        except Exception as exc:
            logger.info("searching-insertion-spiral: hover %d failed (%s)", k, exc)
        at = where()
        if at is None:
            out["route"], out["detail"] = "unreachable", "the hand's pose could not be read"
            return out
        if float(np.linalg.norm(at - above)) > HOVER_TOLERANCE:
            logger.info(
                "searching-insertion-spiral: probe %d skipped, hand %.1f mm off",
                k,
                np.linalg.norm(at - above) * 1e3,
            )
            skipped += 1
            continue
        try:
            # A contact stop (a routed outcome, or a station's "did not reach its
            # target" error) is how a probe ends on the surface: read where it is.
            ctx.tool("robot.move_cartesian_until_contact", **pose(down))
        except Exception as exc:
            logger.info("searching-insertion-spiral: probe %d stopped (%s)", k, exc)
        out["probes"] += 1
        at = where()
        if at is None:
            out["route"], out["detail"] = "unreachable", "the hand's pose could not be read"
            return out
        depth = float(mouth[2] - at[2])
        best = max(best, depth)
        if depth >= float(enter_depth):
            out.update(route="inserted", entered_at=_vec(at), depth=depth)
            if not stay_in:
                ctx.tool("robot.go_to_pose_cartesian", **pose(above))
            return out
        ctx.tool("robot.go_to_pose_cartesian", **pose(above))

    if out["probes"] == 0:
        out["route"] = "unreachable"
        out["detail"] = f"no probe position could be reached ({skipped} skipped)"
        return out
    out["detail"] = (
        f"{out['probes']} probes within {float(search_radius) * 1e3:.0f} mm ({skipped} skipped); "
        f"the deepest got {best * 1e3:.1f} mm below the mouth (needed {float(enter_depth) * 1e3:.1f})"
    )
    return out
