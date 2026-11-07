# Adapted from MOSART (https://github.com/mosart-system/code),
# modules/motion_planning.py (Motion_Plan_Module.get_motion_plan and the SeqIK
# loop around inverse_kinematics_stretch_fullBody), MIT License,
# Copyright (c) 2024 Arjun Gupta.
# Changes: the handle's grasp pose and the articulation (pull direction, or
# hinge point and axis) come in world frame instead of MOSART's fixed handle
# frame and left/right hinge class; the door arc is the grasp pose rigidly
# rotated about the hinge line (MOSART's quarter circle with its matching yaw
# sweep, generalised to any axis and angle); the step counts follow MOSART's
# step sizes (35 mm, 10 deg) instead of fixed counts; the Stretch whole-body
# Pinocchio IK is replaced by the connector's robot.solve_ik, seeded with the
# previous solution only when the connector takes a seed, and otherwise
# checked for a branch jump between neighbouring solutions; the executable
# plan is the reachable prefix (MOSART kept every solved waypoint).
"""Plan the hand's path that opens a drawer or a door it already holds by the handle."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PRISMATIC_STEP = 0.035
"""Waypoint spacing along a drawer's pull [m]: MOSART's 35 cm pull in 10 steps."""

REVOLUTE_STEP_DEG = 10.0
"""Waypoint spacing along a door's arc [deg]: MOSART's quarter circle in 9 steps."""

DEFAULT_DISTANCE = 0.35
"""MOSART's drawer pull [m]. Most YAM-scale drawers want less; pass ``distance``."""

DEFAULT_ANGLE_DEG = 90.0
"""MOSART's door opening, a quarter turn [deg]."""

MIN_RADIUS = 0.01
"""A handle closer than this to the hinge line has no usable arc [m]."""

POSITION_TOLERANCE = 0.02
"""How far an IK solution may land from its waypoint and still count [m], the
hub's reachability gate (selecting-reachable-grasp)."""

MAX_JOINT_STEP = 0.8
"""Largest joint change between neighbouring waypoints [rad]. A 35 mm or 10 deg
step needs a few tenths of a radian on any joint; a wrist branch flip is about
pi. Without a seeded solver this is what keeps the chain on one branch: a hand
holding a handle cannot swing through a flip."""


class Output(TypedDict):
    route: str
    detail: str
    joint_type: str
    waypoints: dict[str, Any]
    all_poses: dict[str, Any]
    joint_trajectory: dict[str, Any]
    reachable: list[bool]
    reachable_fraction: float
    opening: float
    planned_opening: float
    radius: float
    seeded: bool


def _xyz(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=float).reshape(3)


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _unit(v: np.ndarray, name: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError(f"{name} must be a non-zero vector")
    return v / n


def _quat_to_matrix(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        raise ValueError("handle_pose.rotation is not a quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix_to_quat(m: np.ndarray, near: dict[str, float]) -> dict[str, float]:
    """wxyz, sign chosen nearest ``near`` so the sequence does not jump hemispheres."""
    t = float(np.trace(m))
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q) / np.linalg.norm(q)
    if float(np.dot(q, [near[k] for k in ("w", "x", "y", "z")])) < 0:
        q = -q
    return dict(zip(("w", "x", "y", "z"), map(float, q), strict=True))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues: rotation by ``angle`` about the unit ``axis`` (right hand)."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def _steps(amount: float, step: float, num_waypoints: int | None) -> int:
    if num_waypoints:
        return max(1, int(num_waypoints))
    return max(1, math.ceil(amount / step - 1e-9))


def _joints(reply: Any) -> list[float] | None:
    """The joint vector out of whatever shape the IK backend returned."""
    value = reply
    if isinstance(reply, dict):
        value = None
        for key in ("joint_config", "joints", "q", "solution"):
            if reply.get(key) is not None:
                value = reply[key]
                break
    if isinstance(value, dict):
        value = value.get("positions")
    if value is None:
        return None
    try:
        return [float(v) for v in np.asarray(value, dtype=float).reshape(-1)]
    except (TypeError, ValueError):
        return None


def _ik_verdict(reply: Any) -> str:
    """Why an IK reply does not hold the waypoint, or ""."""
    if not isinstance(reply, dict):
        return "" if reply is not None else "no reply"
    if reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "solve_ik reported no solution")
    if reply.get("within_servo_tolerance") is False:
        return "solution outside the servo's tolerance"
    for key in ("position_error_m", "position_error", "error_m"):
        value = reply.get(key)
        if isinstance(value, (int, float)) and value > POSITION_TOLERANCE:
            return f"solution lands {value * 1e3:.0f} mm off"
    return ""


def plan_poses(
    handle_pose: dict[str, Any],
    joint_type: str,
    pull_direction: Any = None,
    distance: float = DEFAULT_DISTANCE,
    hinge_point: Any = None,
    hinge_axis: Any = None,
    angle_deg: float = DEFAULT_ANGLE_DEG,
    num_waypoints: int | None = None,
) -> tuple[list[dict[str, Any]], list[float], float]:
    """The hand's poses from the grasp (index 0) to the full opening, the
    opening at each (m or rad), and the arc radius (0 for a drawer)."""
    p0 = _xyz(handle_pose["position"])
    q0 = handle_pose["rotation"]
    r0 = _quat_to_matrix(q0)
    kind = str(joint_type).lower()
    if kind == "prismatic":
        if pull_direction is None:
            raise ValueError("a prismatic joint needs pull_direction")
        d = _unit(_xyz(pull_direction), "pull_direction")
        total = float(distance)
        if not total > 0:
            raise ValueError("distance must be positive")
        n = _steps(total, PRISMATIC_STEP, num_waypoints)
        openings = [total * k / n for k in range(n + 1)]
        poses = [{"position": _vec(p0 + d * s), "rotation": dict(q0)} for s in openings]
        return poses, openings, 0.0
    if kind == "revolute":
        if hinge_point is None or hinge_axis is None:
            raise ValueError("a revolute joint needs hinge_point and hinge_axis")
        c = _xyz(hinge_point)
        a = _unit(_xyz(hinge_axis), "hinge_axis")
        lever = (p0 - c) - a * float(np.dot(p0 - c, a))
        radius = float(np.linalg.norm(lever))
        if radius < MIN_RADIUS:
            raise ValueError(f"the handle lies {radius * 1e3:.1f} mm from the hinge line")
        if pull_direction is not None:
            # The axis sign that starts the handle moving along the given opening direction.
            hint = _unit(_xyz(pull_direction), "pull_direction")
            if float(np.dot(np.cross(a, lever), hint)) < 0:
                a = -a
        total = math.radians(float(angle_deg))
        if not total > 0:
            raise ValueError("angle_deg must be positive")
        n = _steps(math.degrees(total), REVOLUTE_STEP_DEG, num_waypoints)
        openings = [total * k / n for k in range(n + 1)]
        poses = []
        near = q0
        for phi in openings:
            rot = _axis_angle(a, phi)
            quat = _matrix_to_quat(rot @ r0, near)
            near = quat
            poses.append({"position": _vec(c + rot @ (p0 - c)), "rotation": quat})
        return poses, openings, radius
    raise ValueError(f"joint_type must be 'prismatic' or 'revolute', got {joint_type!r}")


def run(
    ctx: NodeContext,
    handle_pose: dict[str, Any],
    joint_type: str,
    pull_direction: dict[str, float] | None = None,
    distance: float = DEFAULT_DISTANCE,
    hinge_point: dict[str, float] | None = None,
    hinge_axis: dict[str, float] | None = None,
    angle_deg: float = DEFAULT_ANGLE_DEG,
    num_waypoints: int | None = None,
    min_fraction: float = 1.0,
    seed_argument: str = "",
    max_joint_step: float = MAX_JOINT_STEP,
    arm_id: int = 0,
) -> Output:
    """Generate MOSART's opening trajectory and check every waypoint with IK.

    Args:
        handle_pose: The hand's pose holding the handle, world frame.
        joint_type: ``"prismatic"`` (drawer) or ``"revolute"`` (door, lid).
        pull_direction: Drawer: the pull, world frame. Door: optional, the way
            the handle should start moving; it picks the hinge axis's sign.
        distance: Drawer pull [m].
        hinge_point, hinge_axis: Any point on the hinge line and its direction.
            Without ``pull_direction`` the door turns right-handed about the axis.
        angle_deg: Door opening [deg].
        num_waypoints: Steps after the grasp; default from MOSART's step sizes.
        min_fraction: Reachable share of the steps that still counts as
            ``planned`` (1.0: all of them).
        seed_argument: Name of the connector's solve_ik seed keyword (e.g.
            ``"seed_joints"``); each solve then starts from the previous solution.
            Empty: not seeded. A connector that refuses the keyword is asked again
            without it and is not seeded afterwards.
        max_joint_step: Branch-jump gate between neighbouring solutions [rad].
    """
    poses, openings, radius = plan_poses(
        handle_pose,
        joint_type,
        pull_direction=pull_direction,
        distance=distance,
        hinge_point=hinge_point,
        hinge_axis=hinge_axis,
        angle_deg=angle_deg,
        num_waypoints=num_waypoints,
    )
    kind = str(joint_type).lower()
    extra = {"arm_id": int(arm_id)} if arm_id else {}
    seed_key = str(seed_argument or "")
    seeded = False

    def solve(pose: dict[str, Any], prev: list[float] | None) -> tuple[Any, str]:
        nonlocal seed_key, seeded
        if seed_key and prev is not None:
            try:
                reply = ctx.tool("robot.solve_ik", pose=pose, **extra, **{seed_key: list(prev)})
                seeded = True
                return reply, ""
            except Exception as exc:
                logger.info("planning-articulation-opening: seeded solve_ik refused (%s)", exc)
                try:
                    reply = ctx.tool("robot.solve_ik", pose=pose, **extra)
                except Exception as exc2:
                    return None, str(exc2) or type(exc2).__name__
                logger.info(
                    "planning-articulation-opening: solve_ik takes no %r; not seeding", seed_key
                )
                seed_key = ""
                return reply, ""
        try:
            return ctx.tool("robot.solve_ik", pose=pose, **extra), ""
        except Exception as exc:  # an unreachable pose raises on a real station
            return None, str(exc) or type(exc).__name__

    reachable: list[bool] = []
    solutions: list[list[float] | None] = []
    reasons: list[str] = []
    prev: list[float] | None = None
    for index, pose in enumerate(poses):
        reply, error = solve(pose, prev)
        why = error or _ik_verdict(reply)
        q = None if why else _joints(reply)
        if not why and q is not None and prev is not None and len(q) == len(prev):
            jump = float(np.max(np.abs(np.subtract(q, prev))))
            if jump > float(max_joint_step):
                why = f"joint jump {jump:.2f} rad from the last solution (branch flip)"
        if why:
            reachable.append(False)
            solutions.append(None)
            reasons.append(f"waypoint {index}: {why}")
            if index == 0:
                # MOSART: an unreachable grasp pose fails the whole plan.
                reachable.extend([False] * (len(poses) - 1))
                solutions.extend([None] * (len(poses) - 1))
                break
            continue
        reachable.append(True)
        solutions.append(q)
        reasons.append("")
        if q is not None:
            prev = q

    prefix = 0
    while prefix < len(reachable) and reachable[prefix]:
        prefix += 1
    steps = len(poses) - 1
    fraction = float(sum(reachable[1:])) / steps if steps else 0.0
    prefix_fraction = (prefix - 1) / steps if prefix > 0 and steps else 0.0
    opening = openings[prefix - 1] if prefix > 0 else 0.0

    unit = "mm" if kind == "prismatic" else "deg"
    scale = 1e3 if kind == "prismatic" else math.degrees(1.0)
    planned = prefix > 0 and prefix_fraction >= float(min_fraction) - 1e-9
    failures = [r for r in reasons if r]
    if prefix == len(poses):
        detail = f"all {len(poses)} waypoints reachable; opens {opening * scale:.0f} {unit}"
    elif prefix == 0:
        detail = f"the grasp pose itself is unreachable ({failures[0]})"
    else:
        detail = (
            f"reachable through waypoint {prefix - 1} of {steps}, opening {opening * scale:.0f} "
            f"of {openings[-1] * scale:.0f} {unit}; first failure {failures[0]}"
        )
    if radius:
        detail += f"; arc radius {radius * 1e3:.0f} mm"

    executable = poses[:prefix]
    return {
        "route": "planned" if planned else "unreachable",
        "detail": detail,
        "joint_type": kind,
        "waypoints": {
            "waypoints": [
                {"pose": p, "mode": "planned_linear", "cartesian": True} for p in executable
            ]
        },
        "all_poses": {
            "waypoints": [{"pose": p, "mode": "planned_linear", "cartesian": True} for p in poses]
        },
        "joint_trajectory": {
            "waypoints": [{"positions": q} for q in solutions[:prefix] if q is not None]
        },
        "reachable": reachable,
        "reachable_fraction": fraction,
        "opening": float(opening),
        "planned_opening": float(openings[-1]),
        "radius": float(radius),
        "seeded": seeded,
    }
