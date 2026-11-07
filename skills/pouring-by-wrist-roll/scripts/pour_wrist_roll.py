# Adapted from VLABench (https://github.com/OpenMOSS/VLABench),
# VLABench/utils/skill_lib.py (SkillLib.pour, SkillLib.flip), MIT,
# Copyright (c) 2024 simpler-env.
# Changes: a world-frame target box and a pour pose above it in place of the
# simulator's own placement; the roll through the connector's
# robot.move_to_joints in bounded steps paced to a joint speed (upstream stepped
# the simulator twice per pi/40 increment); the roll direction chosen by the
# room left to the joint limits; a hold and a roll back; arrival, tracking and
# hand-drift checks read from fresh observations; a pose roll about the tool
# axis for connectors that refuse joint moves; refusals routed, not raised.
"""Pour from a held container: stand above the target, roll only the last joint, hold, roll back."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

POUR_ANGLE_DEG = 120.0
"""Roll of the last joint [deg]. Upstream's default is pi; an i2rt YAM's joint 6
spans only +-120 deg, so a half turn fits only from a start past 60 deg toward
the other limit."""

POUR_HEIGHT = 0.10
"""Tool centre over the target box's top [m]. In a YAM station twin it held a
30 mm cube's bottom about 90 mm over a 40 mm box's rim."""

STEP_RAD = math.pi / 40
"""Largest joint increment per command [rad], upstream's target_q_velocity."""

MAX_SPEED = 0.5
"""Roll speed bound [rad/s]: each step takes at least STEP_RAD / MAX_SPEED. A
real YAM station's supervisor trips at 0.75 rad/s measured joint speed."""

HOLD_S = 1.0
"""Time held at the full roll before rolling back [s] (upstream: none)."""

TRACK_TOLERANCE = 0.075
"""How far the last joint may lag a step's target [rad]: a real YAM station's
single-arm settle tolerance (upstream: 0.01 rad, in a simulator)."""

RETURN_TOLERANCE = 0.08
"""Bound on every joint after the roll back [rad], as returning-to-a-recorded-pose."""

REACH_TOLERANCE = 0.02
"""How far the hand may end from the pour pose [m]."""

DRIFT_TOLERANCE = 0.02
"""How far the tool centre may wander from the pour pose during the roll [m]."""

ROLL_LIMITS = (-2.0944, 2.0944)
"""Last joint's range [rad] when the caller states none: the i2rt YAM's joint 6
(+-120 deg, its MJCF). Narrower than most arms' last joint, so safe elsewhere."""

LIMIT_MARGIN = 0.05
"""Kept clear of each roll limit [rad]."""


class Output(TypedDict):
    route: str
    detail: str
    roll_reached: float
    roll_trajectory: list[float]
    grip_trajectory: list[float]
    hand_drift: float
    final_joint_error: float
    roll_axis_elevation_deg: float
    rolled_with: str


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _wxyz(q: dict[str, float]) -> np.ndarray:
    return np.array([float(q[k]) for k in ("w", "x", "y", "z")])


def _quat(q: np.ndarray) -> dict[str, float]:
    return dict(zip(("w", "x", "y", "z"), map(float, q), strict=True))


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


def _tool_z(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


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


def _positions(joints: Any) -> np.ndarray:
    if isinstance(joints, dict):
        joints = joints.get("positions")
    return np.asarray(joints if joints is not None else [], dtype=np.float64).reshape(-1)


def run(
    ctx: NodeContext,
    target_obb: dict[str, Any],
    pour_angle_deg: float = POUR_ANGLE_DEG,
    pour_height: float = POUR_HEIGHT,
    offset: dict[str, float] | None = None,
    rotation: dict[str, float] | None = None,
    roll_direction: int = 0,
    roll_limits: list[float] | None = None,
    roll_with: str = "auto",
    step_rad: float = STEP_RAD,
    max_speed: float = MAX_SPEED,
    hold_s: float = HOLD_S,
    track_tolerance: float = TRACK_TOLERANCE,
    drift_tolerance: float = DRIFT_TOLERANCE,
    reach_tolerance: float = REACH_TOLERANCE,
    arm_id: int = 0,
    tolerance: float | None = None,
    max_steps: int | None = None,
) -> Output:
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "roll_reached": 0.0,
        "roll_trajectory": [],
        "grip_trajectory": [],
        "hand_drift": 0.0,
        "final_joint_error": -1.0,
        "roll_axis_elevation_deg": -1.0,
        "rolled_with": "",
    }

    def finish(route: str, detail: str) -> Output:
        out["route"] = route
        out["detail"] = detail
        return out

    def ee() -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **({"arm_id": arm_id} if arm_id else {})))

    grip = [-1.0]

    def joints() -> np.ndarray:
        arms = ctx.tool("robot.get_observation").get("arms") or []
        if arm_id >= len(arms):
            return np.zeros(0)
        grip[0] = float(arms[arm_id].get("gripper_fraction", -1.0))
        return _positions(arms[arm_id].get("joint_state"))

    def record(q: np.ndarray) -> None:
        out["roll_trajectory"].append(float(q[-1]))
        out["grip_trajectory"].append(grip[0])

    def move(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        kwargs: dict[str, Any] = {"pose": {"position": _vec(position), "rotation": rot}}
        if arm_id:
            kwargs["arm_id"] = arm_id
        try:
            return _refusal(ctx.tool(tool, **kwargs))
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc) or type(exc).__name__

    # 1. The pour pose: over the target's top, at pour_height, the hand's own
    # rotation unless one is given. Rise first when below it, so a held
    # container is not dragged across a rim.
    centre = np.array([float(target_obb["center"][k]) for k in ("x", "y", "z")])
    top = centre[2] + float(target_obb["extent"]["z"])
    shift = np.array([float((offset or {}).get(k, 0.0)) for k in ("x", "y")])
    pour = np.array([*(centre[:2] + shift), top + float(pour_height)])
    start = ee()
    rot = rotation or start["rotation"]
    here = _xyz(start)
    why = ""
    if here[2] < pour[2] - 1.0e-3:
        why = move("robot.go_to_pose_cartesian", np.array([here[0], here[1], pour[2]]), rot)
    why = why or move("robot.go_to_pose", pour, rot)
    at = ee()
    off = float(np.linalg.norm(_xyz(at) - pour))
    if why or off > float(reach_tolerance):
        return finish(
            "unreachable",
            f"pour pose: {why or 'ended off'}; the hand stands {off * 1e3:.0f} mm from it",
        )
    base_rot = _wxyz(at["rotation"])
    out["roll_axis_elevation_deg"] = math.degrees(
        math.asin(min(1.0, abs(float(_tool_z(base_rot)[2]))))
    )

    # 2. The roll: direction and size against the last joint's limits.
    q0 = joints()
    if q0.size == 0:
        return finish("roll_refused", f"no joint reading for arm {arm_id}")
    last = q0.size - 1
    angle = math.radians(abs(float(pour_angle_deg)))
    lo, hi = (float(v) for v in (roll_limits or ROLL_LIMITS))
    room = {1: hi - LIMIT_MARGIN - q0[last], -1: q0[last] - (lo + LIMIT_MARGIN)}
    if roll_direction:
        sign = 1 if roll_direction > 0 else -1
    else:
        sign = 1 if room[1] >= room[-1] else -1
    if angle > room[sign] + 1.0e-9:
        return finish(
            "joint_limit",
            f"joint {last + 1} at {q0[last]:.3f} rad leaves {max(0.0, room[sign]):.3f} rad toward "
            f"{'+' if sign > 0 else '-'} (limits {lo:.3f}..{hi:.3f}, {LIMIT_MARGIN} rad margin); "
            f"{angle:.3f} rad asked",
        )
    n = max(1, math.ceil(angle / max(1.0e-3, float(step_rad))))
    period = (angle / n) / max(1.0e-3, float(max_speed))
    pour_xyz = _xyz(at)
    record(q0)

    def joint_step(target: np.ndarray) -> str:
        kwargs: dict[str, Any] = {"joint_config": {"positions": target}}
        if tolerance is not None:
            kwargs["tolerance"] = float(tolerance)
        if max_steps is not None:
            kwargs["max_steps"] = int(max_steps)
        if arm_id:
            kwargs["arm_id"] = arm_id
        try:
            return _refusal(ctx.tool("robot.move_to_joints", **kwargs))
        except Exception as exc:
            return str(exc) or type(exc).__name__

    pose_sign = [1.0]

    def pose_step(delta: float) -> str:
        half = 0.5 * pose_sign[0] * delta
        q = _mul(base_rot, np.array([math.cos(half), 0.0, 0.0, math.sin(half)]))
        return move("robot.go_to_pose", pour_xyz, _quat(q))

    mode = "pose" if roll_with == "pose" else "joints"

    def go(delta: float) -> tuple[str, np.ndarray]:
        """Command the last joint to q0 + delta and read the arm back."""
        nonlocal mode
        began = time.monotonic()
        why = ""
        if mode == "joints":
            target = q0.copy()
            target[last] += delta
            why = joint_step(target)
            if why and roll_with == "auto" and len(out["roll_trajectory"]) == 1:
                now = joints()
                if now.size == q0.size and float(np.max(np.abs(now - q0))) < 1.0e-3:
                    # Nothing moved: a connector that takes poses only.
                    logger.info(
                        "pouring-by-wrist-roll: joint move refused (%s); rolling by pose", why
                    )
                    mode, why = "pose", ""
        if mode == "pose":
            why = pose_step(delta)
        now = joints()
        hand = _xyz(ee())
        out["hand_drift"] = max(out["hand_drift"], float(np.linalg.norm(hand - pour_xyz)))
        if now.size == q0.size:
            record(now)
            out["roll_reached"] = max(out["roll_reached"], abs(float(now[last] - q0[last])))
        wait = period - (time.monotonic() - began)
        if wait > 0:
            time.sleep(wait)
        return why, now

    def check(delta: float, why: str, now: np.ndarray) -> str:
        if why:
            return f"move refused: {why}"
        if now.size != q0.size:
            return "no joint reading"
        lag = abs(float(now[last] - (q0[last] + delta)))
        if lag > float(track_tolerance):
            return f"joint {last + 1} {lag:.3f} rad off the step's target (bound {float(track_tolerance):.3f})"
        if out["hand_drift"] > float(drift_tolerance):
            return (
                f"the tool centre drifted {out['hand_drift'] * 1e3:.0f} mm from the pour pose "
                f"(bound {float(drift_tolerance) * 1e3:.0f})"
            )
        return ""

    def roll_back(reached: int) -> str:
        """Step back from step ``reached`` to the start; why it stopped, or ""."""
        for i in range(reached - 1, -1, -1):
            why, _ = go(sign * angle * i / n)
            if why:
                return f"roll back refused: {why}"
        return ""

    failure = ""
    done = 0
    for i in range(1, n + 1):
        delta = sign * angle * i / n
        why, now = go(delta)
        if i == 1 and mode == "pose" and not why and now.size == q0.size:
            moved = float(now[last] - q0[last])
            if abs(moved) < 0.3 * abs(delta):
                failure = f"a roll about the tool axis moved joint {last + 1} only {moved:.3f} rad"
                done = i
                break
            if moved * delta < 0:
                pose_sign[0] = -1.0
                why, now = go(delta)
        failure = check(delta, why, now)
        done = i
        if failure:
            break
    out["rolled_with"] = mode
    if failure:
        back = roll_back(done)
        now = joints()
        if now.size == q0.size:
            out["final_joint_error"] = float(np.max(np.abs(now - q0)))
        return finish(
            "roll_refused", f"step {done} of {n}: {failure}" + (f"; {back}" if back else "")
        )

    time.sleep(max(0.0, float(hold_s)))
    held = joints()
    if held.size == q0.size:
        record(held)

    back = roll_back(n)
    now = joints()
    if back:
        return finish("roll_refused", back)
    if now.size != q0.size:
        return finish("roll_refused", "no joint reading after the roll back")
    error = float(np.max(np.abs(now - q0)))
    out["final_joint_error"] = error
    if error > RETURN_TOLERANCE:
        worst = int(np.argmax(np.abs(now - q0)))
        return finish(
            "roll_refused",
            f"rolled {math.degrees(out['roll_reached']):.0f} deg but joint {worst + 1} ended "
            f"{error:.3f} rad from its start after the roll back (bound {RETURN_TOLERANCE})",
        )
    detail = ""
    if out["roll_axis_elevation_deg"] > 60.0:
        detail = (
            f"the roll axis stood {out['roll_axis_elevation_deg']:.0f} deg from horizontal: "
            "the roll spun the container more than it tipped it"
        )
    return finish("poured", detail)
