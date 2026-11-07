# Adapted from o2ac-ur (https://github.com/o2ac/o2ac-ur),
# catkin_ws/src/o2ac_routines/src/o2ac_routines/common.py, the tail of
# insert_motor_cables_without_tools_normal (linear_push, then pose_dist < 0.005
# means the tip did not go in), MIT License, Copyright (c) 2020 Team O2AC.
# Changes: o2ac's force-controlled linear_push (15 N, 30 mm, UR F/T sensor) is a
# robot.move_cartesian_until_contact move of a few millimetres judged by the
# hand's measured stop (the YAM has no force sensor); the push is along a
# world-frame axis instead of the tool's -Z; the travel is measured along that
# axis instead of as a pose distance; o2ac's single answer is split into seated
# (the hand did not go further) and not_seated (it did), and refused moves,
# unreadable poses and IK previews out of reach route blocked instead of
# raising; optional sideways pushes, the IK preview, the floor over the table
# and the return to the start are new.
"""Check an insertion by pushing the held part a little further and reading where the hand stopped."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PROBE_DISTANCE = 0.010
"""How far past the start the push is sent along the axis [m]; it stops earlier
on contact. o2ac pushes up to 30 mm with a 15 N force limit; a position-
controlled arm with no force limit presses a seated part for whatever is left
of the probe, so it is kept short. It must be shorter than any gap a part that
is not seated can have, and long enough that a free push gets well past
seat_tolerance: in a YAM station twin a free push ended about 2 mm short of
its end (the servo's arrival ball), 6.1 mm of 8, 7.3 of 10, 10.4 of 12."""

SEAT_TOLERANCE = 0.005
"""Travel along the axis under which the part counts as seated [m]: o2ac's
``pose_dist < 0.005``. A held 30 mm cube resting on a box floor in a YAM station
twin let the hand go 1.2-3.7 mm further before the stall, whatever the probe's
length: the friction grip let the cube slide up the jaws by about that much."""

LATERAL_DISTANCE = 0.0
"""How far each sideways push is sent [m]; 0 skips them. A part in a fitting
hole cannot move sideways; a part resting in a wide pocket or on a surface can.
Not in o2ac."""

LATERAL_TOLERANCE = 0.003
"""Sideways travel under which the part counts as held by its seat [m]: about
the clearance of a loose fit; the twin's servo arrives within 4 mm, so lateral
pushes shorter than about 6 mm say little there."""

MIN_TOOL_HEIGHT = 0.015
"""Lowest tool-centre height over the table the push is sent to [m], when
``table_z`` is given. A real YAM station refuses a tool centre under 14 mm over
its calibrated table."""

RETURN_TOLERANCE = 0.015
"""How far the hand may end from the start after the return [m]: no tighter
than a real YAM station's own 12 mm per-move bound."""

REFUSALS = ("unreachable", "bad_speed_scale")
"""Routed outcomes that mean a move was refused, not stopped by contact."""

STOPS = ("did not reach",)
"""Text of a raised error that means the move ran and was stopped short (a real
YAM station raises "the tool did not reach its target" when the arm stalls):
the hand's position then says how far it got. Any other error is a refusal."""


class Output(TypedDict):
    route: str
    detail: str
    travel: float
    lateral_travel: float
    start_position: dict[str, float]
    stop_position: dict[str, float]
    returned: bool


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _unit(v: Any) -> np.ndarray | None:
    a = np.array([float(v[k]) for k in ("x", "y", "z")]) if isinstance(v, dict) else np.asarray(v)
    n = float(np.linalg.norm(a))
    return a / n if n > 1.0e-9 else None


def _refusal(reply: Any) -> str:
    """Why a move's reply says it was refused, or "". A contact stop or a
    timeout is not a refusal: the move ran and the hand's position says how far."""
    if not isinstance(reply, dict):
        return ""
    outcome = reply.get("outcome")
    if outcome in REFUSALS:
        return f"{outcome}: {reply.get('reason', '')}".rstrip(": ")
    if outcome is None and reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def lateral_directions(axis: np.ndarray) -> list[np.ndarray]:
    """Four directions square to the axis: +-u, +-v, with u along world x when
    the axis is vertical."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - axis * float(np.dot(ref, axis))
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return [u, -u, v, -v]


def run(
    ctx: NodeContext,
    axis: dict[str, float] | None = None,
    probe_distance: float = PROBE_DISTANCE,
    seat_tolerance: float = SEAT_TOLERANCE,
    lateral_distance: float = LATERAL_DISTANCE,
    lateral_tolerance: float = LATERAL_TOLERANCE,
    table_z: float | None = None,
    min_tool_height: float = MIN_TOOL_HEIGHT,
    return_to_start: bool = True,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    out: Output = {
        "route": "blocked",
        "detail": "",
        "travel": 0.0,
        "lateral_travel": 0.0,
        "start_position": _vec(np.zeros(3)),
        "stop_position": _vec(np.zeros(3)),
        "returned": False,
    }

    def blocked(why: str) -> Output:
        out["route"], out["detail"] = "blocked", why
        return out

    def where() -> np.ndarray | None:
        try:
            return _xyz(_pose(ctx.tool("robot.get_ee_pose", **on_arm)))
        except Exception as exc:
            logger.info("verifying-insertion-by-probe-push: no pose reading (%s)", exc)
            return None

    direction = _unit(axis if axis is not None else {"x": 0.0, "y": 0.0, "z": -1.0})
    if direction is None:
        return blocked("the insertion axis has no length")
    try:
        start_pose = _pose(ctx.tool("robot.get_ee_pose", **on_arm))
        start = _xyz(start_pose)
        rotation = start_pose["rotation"]
    except Exception as exc:
        return blocked(f"the hand's pose could not be read ({exc})")
    out["start_position"] = out["stop_position"] = _vec(start)

    def pose(p: np.ndarray) -> dict[str, Any]:
        return {"pose": {"position": _vec(p), "rotation": rotation}, **on_arm}

    def out_of_reach(p: np.ndarray) -> bool:
        """Whether the IK says the probe's end cannot be reached: a stall at the
        edge of reach reads exactly like a seated part. A connector without a
        usable answer gets the benefit of the doubt."""
        try:
            reply = ctx.tool("robot.solve_ik", **pose(p))
        except Exception as exc:
            logger.info("verifying-insertion-by-probe-push: no IK preview (%s)", exc)
            return False
        return isinstance(reply, dict) and (
            reply.get("success") is False or reply.get("within_servo_tolerance") is False
        )

    def probe(d: np.ndarray, distance: float) -> tuple[float, np.ndarray | None, str]:
        """One push from the start along d: the measured travel along d, where
        the hand stopped, and why the push was refused ("" when it ran)."""
        try:
            why = _refusal(
                ctx.tool("robot.move_cartesian_until_contact", **pose(start + d * distance))
            )
        except Exception as exc:
            text = str(exc)
            why = "" if any(s in text for s in STOPS) else text
            logger.info("verifying-insertion-by-probe-push: push ended by an error (%s)", text)
        stop = where()
        if stop is None:
            return 0.0, None, why or "the hand's pose could not be read after the push"
        return float(np.dot(stop - start, d)), stop, why

    def back_to_start() -> str:
        try:
            why = _refusal(ctx.tool("robot.go_to_pose_cartesian", **pose(start)))
        except Exception as exc:
            why = str(exc)
        at = where()
        off = float("inf") if at is None else float(np.linalg.norm(at - start))
        if off > float(RETURN_TOLERANCE):
            return f"{why or 'ended off'}; the hand is {off * 1e3:.0f} mm from the start"
        return ""

    distance = float(probe_distance)
    if table_z is not None and direction[2] < -1.0e-6:
        room = (start[2] - (float(table_z) + float(min_tool_height))) / -direction[2]
        distance = min(distance, room)
    if distance <= float(seat_tolerance):
        return blocked(
            f"the push can go only {max(distance, 0.0) * 1e3:.1f} mm before the tool-centre "
            f"floor, not more than seat_tolerance ({float(seat_tolerance) * 1e3:.1f} mm)"
        )
    if check_reach and out_of_reach(start + direction * distance):
        return blocked("the end of the push is out of reach (IK preview)")

    travel, stop, why = probe(direction, distance)
    if stop is None or why:
        return blocked(f"push: {why}")
    out["travel"], out["stop_position"] = travel, _vec(stop)
    seated = travel < float(seat_tolerance)
    detail = (
        f"pushed {distance * 1e3:.1f} mm along the axis, the hand went {travel * 1e3:.1f} mm "
        f"({'under' if seated else 'not under'} {float(seat_tolerance) * 1e3:.1f} mm)"
    )

    if seated and float(lateral_distance) > 0.0:
        sideways = 0.0
        for k, d in enumerate(lateral_directions(direction)):
            back = back_to_start()
            if back:
                return blocked(f"{detail}; before sideways push {k}: {back}")
            if check_reach and out_of_reach(start + d * float(lateral_distance)):
                return blocked(f"{detail}; sideways push {k} is out of reach (IK preview)")
            moved, at, why = probe(d, float(lateral_distance))
            if at is None or why:
                return blocked(f"{detail}; sideways push {k}: {why}")
            sideways = max(sideways, moved)
        out["lateral_travel"] = sideways
        seated = sideways < float(lateral_tolerance)
        detail += (
            f"; sideways pushes of {float(lateral_distance) * 1e3:.1f} mm went at most "
            f"{sideways * 1e3:.1f} mm ({'under' if seated else 'not under'} "
            f"{float(lateral_tolerance) * 1e3:.1f} mm)"
        )

    out["route"] = "seated" if seated else "not_seated"
    if return_to_start:
        back = back_to_start()
        out["returned"] = not back
        if back:
            detail += f"; return to the start: {back}"
    out["detail"] = detail + "; this is the hand's motion, not a measurement of the part"
    return out
