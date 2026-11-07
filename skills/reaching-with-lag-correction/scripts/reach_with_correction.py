# Adapted from MAESTRO (https://github.com/penn-pal-lab/maestro), maestro/api.py,
# _hold_and_wait_for_reach, REACH_TOLERANCE and MAX_REACH_CORRECTION, MIT License,
# Copyright (c) 2026 The MAESTRO Authors (license: LICENSE.maestro beside SKILL.md).
# Changes: a world-frame tool pose in place of a joint vector, moved through the
# connector's robot.go_to_pose (or robot.go_to_pose_cartesian) and read back with
# robot.get_ee_pose; the error is split into position [m] and rotation [rad] with
# a bound on each instead of one joint-space norm; a settle wait (pose readings
# until they stop changing) before each read-back, where MAESTRO slept 0.3 s after
# the call; a refused or failed move is read back and routed rather than raised;
# the correction fires above its own threshold, below the exit tolerance;
# check_roll=false judges only the approach axis, for IK that leaves the roll
# free; the outputs report the residual before and after the correction.
"""Move the tool to a pose; if it stops short (tracking lag, sag), command once more aimed past the shortfall."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

TOLERANCE = 0.005
"""Exit bound on the tool centre's final distance from the target [m]: above the
4 mm arrival ball of a servo connector, well inside a real YAM station's 12 mm
per-move bound (which is what lets the station accept a move that stopped short)."""

ROT_TOLERANCE = 0.05
"""Exit bound on the final rotation error [rad], just above a real YAM station's
0.04 rad per-move bound."""

CORRECT_ABOVE = 0.002
"""Position error above which the aim-past move is sent [m]. The real YAM stands
1.7-3.2 mm over its command near the table (2026-11-06 touch test), so a 2 mm
threshold catches it; MAESTRO corrects whenever the error exceeds its exit bound."""

ROT_CORRECT_ABOVE = 0.02
"""Rotation error above which the aim-past move is sent [rad]."""

MAX_CORRECTION = 0.025
"""Largest position shortfall that is aimed past [m]; a larger error is retried
as is (MAESTRO's rule, whose cap is MAX_REACH_CORRECTION = 0.1 rad per joint).
Covers the 16-21 mm a real YAM stopped short at 0.50 m reach; a larger miss is
more likely a refusal, an obstacle or the edge of reach than sag, and aiming
past it would push the hand into whatever stopped it."""

MAX_ROT_CORRECTION = 0.1
"""Largest rotation error that is aimed past [rad], MAESTRO's per-joint cap."""

REFUSED_TRAVEL = 0.001
"""A move that failed and left the hand within this of where it started [m] was
refused outright (IK, a floor or workspace guard), not cut short by lag: it is
neither aimed past nor retried."""

APPROACH = np.array([0.0, 0.0, 1.0])
"""The tool's approach axis in its own frame (the commanded pose's grasp frame,
+z toward the work). With check_roll=false only this axis's direction is
judged and corrected, for connectors whose IK leaves the roll about it free."""

SETTLE_INTERVAL = 0.1
"""Time between pose readings while the arm settles [s]."""

SETTLE_STEP = 0.0003
"""The arm counts as settled once two readings SETTLE_INTERVAL apart differ by
less than this [m]."""

SETTLE_TIMEOUT = 1.0
"""Longest settle wait [s]; MAESTRO slept a fixed 0.3 s after its reach."""


class Output(TypedDict):
    route: str
    detail: str
    residual: float
    residual_rotation: float
    first_residual: float
    first_residual_rotation: float
    corrected: bool
    correction: dict[str, float]
    reached_pose: dict[str, Any]


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _quat(pose: dict[str, Any]) -> np.ndarray:
    r = pose["rotation"]
    q = np.array([float(r["w"]), float(r["x"]), float(r["y"]), float(r["z"])])
    return q / np.linalg.norm(q)


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _rot(q: np.ndarray) -> dict[str, float]:
    return {"w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3])}


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _angle(q: np.ndarray) -> float:
    return 2.0 * math.acos(min(1.0, abs(float(q[0]))))


def _apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _mul(_mul(q, np.array([0.0, *v])), _conj(q))[1:]


def _between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The shortest rotation taking unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s, c = float(np.linalg.norm(axis)), float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-12:
        if c > 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        ortho = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        return np.array([0.0, *(ortho / np.linalg.norm(ortho))])
    half = math.atan2(s, c) / 2.0
    return np.array([math.cos(half), *(axis / s * math.sin(half))])


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
    target_pose: dict[str, Any],
    tolerance: float = TOLERANCE,
    rot_tolerance: float = ROT_TOLERANCE,
    correct_above: float = CORRECT_ABOVE,
    rot_correct_above: float = ROT_CORRECT_ABOVE,
    max_correction: float = MAX_CORRECTION,
    max_rot_correction: float = MAX_ROT_CORRECTION,
    cartesian: bool = False,
    settle_timeout: float = SETTLE_TIMEOUT,
    settle_interval: float = SETTLE_INTERVAL,
    servo_tolerance: float | None = None,
    check_roll: bool = True,
    arm_id: int = 0,
) -> Output:
    target = _pose(target_pose)
    goal_p, goal_q = _xyz(target), _quat(target)
    tool = "robot.go_to_pose_cartesian" if cartesian else "robot.go_to_pose"
    knobs: dict[str, Any] = {}
    if arm_id:
        knobs["arm_id"] = int(arm_id)
    if servo_tolerance is not None and not cartesian:
        knobs["tolerance"] = float(servo_tolerance)

    def move(p: np.ndarray, q: np.ndarray) -> str:
        try:
            reply = ctx.tool(tool, pose={"position": _vec(p), "rotation": _rot(q)}, **knobs)
            return _refusal(reply)
        except Exception as exc:  # a refused or stopped-short move is read back, not raised
            return str(exc) or type(exc).__name__

    def read() -> dict[str, Any]:
        kw = {"arm_id": int(arm_id)} if arm_id else {}
        return _pose(ctx.tool("robot.get_ee_pose", **kw))

    def settle() -> dict[str, Any]:
        """Read the tool pose until two readings agree: a lagging arm is still creeping."""
        last = read()
        deadline = time.monotonic() + float(settle_timeout)
        while time.monotonic() < deadline:
            time.sleep(max(0.0, float(settle_interval)))
            now = read()
            if float(np.linalg.norm(_xyz(now) - _xyz(last))) < SETTLE_STEP:
                return now
            last = now
        return last

    def error(pose: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float]:
        dp = goal_p - _xyz(pose)
        here = _quat(pose)
        if check_roll:
            dq = _mul(goal_q, _conj(here))  # world-frame rotation from where it is to the goal
        else:
            dq = _between(_apply(here, APPROACH), _apply(goal_q, APPROACH))
        if dq[0] < 0.0:
            dq = -dq
        return dp, dq, float(np.linalg.norm(dp)), _angle(dq)

    start = _xyz(read())
    why_first = move(goal_p, goal_q)
    pose = settle()
    dp, dq, first, first_rot = error(pose)
    moved = float(np.linalg.norm(_xyz(pose) - start))
    out: Output = {
        "route": "not_reached",
        "detail": "",
        "residual": first,
        "residual_rotation": first_rot,
        "first_residual": first,
        "first_residual_rotation": first_rot,
        "corrected": False,
        "correction": _vec(np.zeros(3)),
        "reached_pose": pose,
    }
    notes = [f"first move: {why_first}"] if why_first else []
    refused = bool(why_first) and moved < REFUSED_TRAVEL and first > float(tolerance)
    if refused:
        notes.append(f"the hand moved {moved * 1e3:.1f} mm: refused, not aimed past")

    if not refused and (first > float(correct_above) or first_rot > float(rot_correct_above)):
        # MAESTRO's rule: a shortfall within the cap is aimed past (target + error),
        # a larger one is retried as is. Each axis (position, rotation) is judged
        # on its own cap, so a small position sag is still corrected when the wrist
        # is far off, and the other way round.
        aim_p = dp if first <= float(max_correction) else np.zeros(3)
        aim_q = dq if first_rot <= float(max_rot_correction) else np.array([1.0, 0.0, 0.0, 0.0])
        command_p = goal_p + aim_p
        command_q = _mul(aim_q, goal_q)
        out["corrected"] = bool(np.any(aim_p) or _angle(aim_q) > 0.0)
        out["correction"] = _vec(aim_p)
        why_second = move(command_p, command_q)
        if why_second:
            notes.append(f"second move: {why_second}")
        if not out["corrected"]:
            notes.append(
                f"error {first * 1e3:.1f} mm / {first_rot:.3f} rad over the correction caps: retried as is"
            )
        pose = settle()
        _, _, out["residual"], out["residual_rotation"] = error(pose)
        out["reached_pose"] = pose

    res, res_rot = out["residual"], out["residual_rotation"]
    if res <= float(tolerance) and res_rot <= float(rot_tolerance):
        out["route"] = "reached"
    else:
        notes.append(
            f"ended {res * 1e3:.1f} mm / {res_rot:.3f} rad from the target "
            f"(bounds {float(tolerance) * 1e3:.1f} mm / {float(rot_tolerance):.3f} rad)"
        )
    out["detail"] = "; ".join(notes)
    logger.info(
        "reaching-with-lag-correction: %s, first %.1f mm, final %.1f mm%s",
        out["route"],
        first * 1e3,
        res * 1e3,
        " (corrected)" if out["corrected"] else "",
    )
    return out
