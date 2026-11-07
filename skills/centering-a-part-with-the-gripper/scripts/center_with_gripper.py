# Adapted from o2ac-ur (https://github.com/o2ac/o2ac-ur),
# catkin_ws/src/o2ac_routines/src/o2ac_routines/common.py, center_with_gripper
# and simple_gripper_check, MIT License, Copyright (c) 2020 Team O2AC.
# Changes: robot.* connector tools in place of MoveIt and the Robotiq driver; the
# hand opens fully (robot.open_gripper) instead of to a commanded opening_width,
# which the YAM station does not offer; the "nothing held" test reads the jaw gap
# through robot.describe_gripper's width fit; the quarter turn is about the hand's
# approach axis from the tool's current rotation, judged from a fresh
# robot.get_ee_pose, falling back to turning the last joint (o2ac's own wording)
# when the pose does not turn the hand; refused moves are routed rather than
# raised; an optional release=false keeps the final close.
"""Centre a loose part between the jaws: close, open, turn the wrist a quarter turn, close again."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

MIN_HELD_WIDTH = 0.003
"""A close that ends within this of the jaws' closed stop held nothing [m].
o2ac's simple_gripper_check fails under 1 mm (6 mm on its imprecise a_bot hand);
3 mm is the hub's verifying-grasps margin, set from an empty close measured at
1.8 mm."""

TURN_DEG = 90.0
"""The wrist turn between the two closes [deg], o2ac's tau / 4."""

ROTATION_TOLERANCE_DEG = 10.0
"""How far the read-back rotation may end from the turn's target [deg]. Looser
than a servo's 3 deg and a real YAM station's 0.04 rad per move."""

POSITION_TOLERANCE = 0.015
"""How far the tool centre may drift during a turn [m], over a real YAM
station's 12 mm per move."""


class Output(TypedDict):
    route: str
    detail: str
    first_width: float
    second_width: float
    turned_deg: float
    turned_by: str
    returned: bool


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _quat(rotation: dict[str, Any]) -> np.ndarray:
    q = np.array([float(rotation[k]) for k in ("w", "x", "y", "z")])
    return q / np.linalg.norm(q)


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(2.0 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


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


def run(
    ctx: NodeContext,
    check_held: bool = True,
    min_held_width: float = MIN_HELD_WIDTH,
    turn_deg: float = TURN_DEG,
    clockwise: bool = False,
    return_to_start: bool = True,
    release: bool = True,
    skip_first_close: bool = False,
    joint_fallback: bool = True,
    rotation_tolerance_deg: float = ROTATION_TOLERANCE_DEG,
    position_tolerance: float = POSITION_TOLERANCE,
    arm_id: int = 0,
) -> Output:
    arm = {"arm_id": int(arm_id)} if arm_id else {}
    out: Output = {
        "route": "centered",
        "detail": "",
        "first_width": -1.0,
        "second_width": -1.0,
        "turned_deg": 0.0,
        "turned_by": "",
        "returned": False,
    }

    try:
        hand = ctx.tool("robot.describe_gripper", **arm)
    except Exception as exc:  # the width check falls back to "unknown", the axis to tool z
        logger.info("centering: no gripper description (%s)", exc)
        hand = {}
    fit = hand.get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or hand.get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)
    axis = hand.get("approach_axis") or {"x": 0.0, "y": 0.0, "z": 1.0}
    axis = np.array([float(axis[k]) for k in ("x", "y", "z")])
    axis = axis / np.linalg.norm(axis)

    def close() -> float:
        """Close and return the measured jaw gap [m], or -1.0 when unreadable."""
        reply = ctx.tool("robot.close_gripper", **arm)
        try:
            fraction = reply.get("position") if isinstance(reply, dict) else None
            if fraction is None:
                fraction = ctx.tool("robot.get_gripper", **arm)["position"]
            if at_open <= 0.0:
                return -1.0
            return at_closed + max(0.0, min(1.0, float(fraction))) * (at_open - at_closed)
        except Exception as exc:  # an unreadable hand is not an empty one
            logger.info("centering: could not read the gripper (%s)", exc)
            return -1.0

    def empty(width: float) -> bool:
        return bool(check_held) and 0.0 <= width <= at_closed + float(min_held_width)

    def mm(width: float) -> str:
        return "unread" if width < 0.0 else f"{width * 1e3:.1f} mm"

    def attempt(tool: str, **kwargs: Any) -> str:
        try:
            return _refusal(ctx.tool(tool, **kwargs, **arm))
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc)

    def judge(goals: dict[float, np.ndarray]) -> tuple[float | None, str]:
        """Which goal rotation a fresh reading stands at (the tool centre where it was), or why none."""
        now = _pose(ctx.tool("robot.get_ee_pose", **arm))
        q, drift = _quat(now["rotation"]), float(np.linalg.norm(_xyz(now) - origin))
        offs = {deg: _angle_deg(q, goal) for deg, goal in goals.items()}
        best = min(offs, key=offs.get)
        if offs[best] <= float(rotation_tolerance_deg) and drift <= float(position_tolerance):
            return best, ""
        return None, f"the hand ended {offs[best]:.0f} deg and {drift * 1e3:.0f} mm from the target"

    def by_pose(rotation: np.ndarray) -> str:
        wxyz = dict(zip(("w", "x", "y", "z"), map(float, rotation), strict=True))
        place = {"x": float(origin[0]), "y": float(origin[1]), "z": float(origin[2])}
        return attempt("robot.go_to_pose_cartesian", pose={"position": place, "rotation": wxyz})

    def by_joints(joints: np.ndarray) -> str:
        return attempt(
            "robot.move_to_joints", joint_config={"positions": [float(v) for v in joints]}
        )

    def go_back() -> bool:
        why = by_pose(q0)
        done, _ = judge({0.0: q0})
        if done is None and start_joints is not None:
            logger.info(
                "centering: the pose did not turn the hand back (%s); by the last joint", why
            )
            by_joints(start_joints)
            done, _ = judge({0.0: q0})
        return done is not None

    if not skip_first_close:
        out["first_width"] = close()
        if empty(out["first_width"]):
            ctx.tool("robot.open_gripper", **arm)
            out["route"] = "nothing_caught"
            out["detail"] = (
                f"the first close ended at {mm(out['first_width'])}, within "
                f"{min_held_width * 1e3:.0f} mm of the closed stop"
            )
            return out
    ctx.tool("robot.open_gripper", **arm)

    start = _pose(ctx.tool("robot.get_ee_pose", **arm))
    origin, q0 = _xyz(start), _quat(start["rotation"])
    start_joints = None
    if joint_fallback:
        try:
            arms = ctx.tool("robot.get_observation").get("arms") or []
            joints = arms[int(arm_id)].get("joint_state") or {}
            joints = joints.get("positions") if isinstance(joints, dict) else joints
            start_joints = (
                np.asarray(joints, dtype=np.float64).reshape(-1) if joints is not None else None
            )
            if start_joints is not None and start_joints.size == 0:
                start_joints = None
        except Exception as exc:  # no joint reading: the pose turn is the only one
            logger.info("centering: no joint reading (%s)", exc)

    sign = -1.0 if clockwise else 1.0
    turns = (sign * float(turn_deg), -sign * float(turn_deg))
    goals = {}
    for turn in turns:
        half = math.radians(turn) / 2.0
        goals[turn] = _mul(q0, np.array([math.cos(half), *(math.sin(half) * axis)]))
    turned, whys = None, []
    for turn in turns:  # o2ac retries the other way
        why = by_pose(goals[turn])
        turned, off = judge({turn: goals[turn]})
        if turned is not None:
            out["turned_by"] = "pose"
            break
        whys.append(f"pose {turn:+.0f} deg: {why + '; ' if why else ''}{off}")
    if turned is None and start_joints is not None:
        # A solver that leaves wrist roll free reaches the pose without turning;
        # o2ac's turn is of the last joint, so turn that.
        for turn in turns:
            joints = start_joints.copy()
            joints[-1] += math.radians(turn)
            why = by_joints(joints)
            turned, off = judge(goals)
            if turned is not None:
                out["turned_by"] = "last_joint"
                break
            whys.append(f"last joint {turn:+.0f} deg: {why + '; ' if why else ''}{off}")
    if turned is None:
        if return_to_start:
            out["returned"] = go_back()
        out["route"] = "unreachable"
        out["detail"] = "neither wrist turn arrived: " + " | ".join(whys)
        return out
    out["turned_deg"] = float(turned)

    out["second_width"] = close()
    if empty(out["second_width"]):
        ctx.tool("robot.open_gripper", **arm)
        if return_to_start:
            out["returned"] = go_back()
        out["route"] = "nothing_caught"
        out["detail"] = (
            f"the close after the turn ended at {mm(out['second_width'])}, within "
            f"{min_held_width * 1e3:.0f} mm of the closed stop"
        )
        return out
    if release:
        ctx.tool("robot.open_gripper", **arm)

    parts = [
        f"closed at {mm(out['first_width'])}, turned {out['turned_deg']:+.0f} deg by the "
        f"{out['turned_by'].replace('_', ' ')}, closed at {mm(out['second_width'])}"
    ]
    if return_to_start:
        out["returned"] = go_back()
        if not out["returned"]:
            parts.append("the turn back did not arrive")
    out["detail"] = "; ".join(parts)
    return out
