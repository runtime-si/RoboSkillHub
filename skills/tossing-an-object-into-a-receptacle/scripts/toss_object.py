# Adapted from TidyBot (https://github.com/jimmyyhwu/tidybot) @ c060278de506,
# robot/kinova.py KinovaArm.toss and server/demo.py's place-or-toss choice,
# MIT License, Copyright (c) 2023 Jimmy Wu.
# Changes: TidyBot's fixed Kinova joint swing (max_vel 140 deg/s, gripper release
# at 600 ms) becomes an analytic ballistic plan for any 6-DoF arm: a launch angle
# and speed that land the held object on the receptacle opening's centre, a
# straight run-up through the release point solved with robot.solve_ik and
# timed at a constant acceleration, and a check of the joint speed that plan
# needs against the arm's limit before anything moves; a drop from above the
# opening when the opening is reachable; the landing read back by a locate tool.
"""Toss a held object into a receptacle beyond reach, or drop it from above when the toss is not needed or not possible."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

G = 9.81

JOINT_SPEED_LIMIT = 0.25
"""Joint speed the plan may use [rad/s]. The real YAM station's motion server
streams at 0.25 rad/s by default and at most 1.5 (--stream-max-vel / --speed 6);
the YAM URDF lists 10 rad/s for the motors. A toss beyond reach needs far more
than 0.25, so at the default the skill routes not_feasible with the speed it
would need."""

ACCEL = 3.0
"""Tool acceleration along the run-up [m/s^2]; sets the run-up length v^2 / 2a."""

LAUNCH_DEG = (20.0, 30.0, 40.0, 45.0, 50.0, 60.0)
"""Launch angles tried; the one needing the least joint speed wins."""

RELEASE_DT = 0.02
"""Time step of the streamed run-up [s] and the finite difference for joint speed."""

DROP_CLEARANCE = 0.08
"""Height of the tool over the opening's rim for a drop from above [m]."""

DROPPED_MARGIN = 0.003
"""A jaw gap within this of the closed stop before release means the object is already gone [m]."""


class Output(TypedDict):
    route: str
    detail: str
    mode: str
    launch_deg: float
    release_speed_mps: float
    required_joint_speed: float
    joint_speed_limit: float
    release_position: list[float]
    landed_position: list[float]


def release_speed(d: float, h: float, theta: float) -> float:
    """Launch speed [m/s] that carries a point from height h above the target over a
    horizontal distance d at launch angle theta; inf when no speed does."""
    c = math.cos(theta)
    den = 2.0 * c * c * (d * math.tan(theta) + h)
    return math.sqrt(G * d * d / den) if den > 1e-9 else math.inf


def flight_time(d: float, v: float, theta: float) -> float:
    return d / max(v * math.cos(theta), 1e-9)


def _xyz(reply: Any) -> np.ndarray:
    p = reply.get("pose", reply) if isinstance(reply, dict) else reply
    p = p.get("position", p) if isinstance(p, dict) else p
    if isinstance(p, dict):
        return np.array([float(p["x"]), float(p["y"]), float(p["z"])])
    return np.asarray(p, dtype=float)[:3]


def _joints(reply: Any) -> np.ndarray | None:
    if not isinstance(reply, dict):
        return None if reply is None else np.asarray(reply, dtype=float)
    if reply.get("success") is False or reply.get("within_servo_tolerance") is False:
        return None
    for key in ("joint_positions", "positions", "joints", "solution", "q"):
        v = reply.get(key)
        if v is not None:
            v = v.get("positions", v) if isinstance(v, dict) else v
            return np.asarray(v, dtype=float)
    return None


def _located(reply: Any) -> np.ndarray | None:
    if not isinstance(reply, dict):
        return None
    for src in (reply, reply.get("obb") or {}, reply.get("pose") or {}):
        for key in ("position", "center"):
            v = src.get(key)
            if v is not None:
                return _xyz({"position": v})
    return None


def run(
    ctx: NodeContext,
    opening_center: list[float],
    opening_radius_m: float,
    locate_tool: str = "",
    locate_args: dict[str, Any] | None = None,
    joint_speed_limit: float = JOINT_SPEED_LIMIT,
    accel: float = ACCEL,
    release_advance_m: float = 0.10,
    allow_drop: bool = True,
    toss_only: bool = False,
    arm_id: int = 0,
) -> Output:
    arm = {"arm_id": int(arm_id)} if arm_id else {}
    target = np.asarray(opening_center, dtype=float)
    out: Output = {
        "route": "not_feasible",
        "detail": "",
        "mode": "",
        "launch_deg": 0.0,
        "release_speed_mps": 0.0,
        "required_joint_speed": 0.0,
        "joint_speed_limit": float(joint_speed_limit),
        "release_position": [],
        "landed_position": [],
    }
    ee = ctx.tool("robot.get_ee_pose", **arm)
    ee = ee.get("pose", ee)
    rotation = ee["rotation"]
    here = _xyz(ee)

    def pose(p: np.ndarray) -> dict[str, Any]:
        return {
            "position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
            "rotation": rotation,
        }

    def ik(p: np.ndarray) -> np.ndarray | None:
        try:
            return _joints(ctx.tool("robot.solve_ik", pose=pose(p), **arm))
        except Exception as exc:
            logger.info("toss: no IK at %s (%s)", np.round(p, 3).tolist(), exc)
            return None

    def still_held() -> bool:
        try:
            fit = (ctx.tool("robot.describe_gripper", **arm) or {}).get("width_fit") or {}
            frac = float((ctx.tool("robot.get_gripper", **arm) or {}).get("position", 1.0))
        except Exception:
            return True
        at_open, at_closed = (
            float(fit.get("at_open_m") or 0.0),
            float(fit.get("at_closed_m") or 0.0),
        )
        width = at_closed + frac * (at_open - at_closed) if at_open > at_closed else frac
        return width > at_closed + DROPPED_MARGIN

    def finish(mode: str) -> Output:
        out["mode"] = mode
        if not locate_tool:
            out.update(
                route="released", detail=f"{mode}: released, no locate_tool to read the landing"
            )
            return out
        try:
            found = _located(ctx.tool(locate_tool, **(locate_args or {})))
        except Exception as exc:
            found = None
            logger.info("toss: %s failed (%s)", locate_tool, exc)
        if found is None:
            out.update(route="missed", detail=f"{mode}: the object was not found after the release")
            return out
        out["landed_position"] = [round(float(v), 4) for v in found]
        off = float(np.linalg.norm(found[:2] - target[:2]))
        inside = off <= opening_radius_m and found[2] <= target[2]
        out.update(
            route="landed" if inside else "missed",
            detail=f"{mode}: landed {off * 1e3:.0f} mm from the opening's centre (radius {opening_radius_m * 1e3:.0f} mm)",
        )
        return out

    if not still_held():
        out.update(
            route="dropped_early", detail="the jaws were on the closed stop before any motion"
        )
        return out

    above = target + np.array([0.0, 0.0, DROP_CLEARANCE])
    if allow_drop and not toss_only and ik(above) is not None:
        ctx.tool("robot.go_to_pose_cartesian", pose=pose(above), **arm)
        if not still_held():
            out.update(route="dropped_early", mode="drop", detail="lost on the way to the opening")
            return out
        ctx.tool("robot.open_gripper", **arm)
        return finish("drop")

    flat = target[:2] - here[:2]
    dist = float(np.linalg.norm(flat))
    if dist < 1e-6:
        out["detail"] = "the opening is straight below the hand but out of IK reach"
        return out
    u = np.array([flat[0] / dist, flat[1] / dist, 0.0])
    release = here + u * release_advance_m
    q_rel = ik(release)
    if q_rel is None:
        out["detail"] = "no IK at the release point"
        return out
    d = float(np.linalg.norm(target[:2] - release[:2]))
    h = float(release[2] - target[2])
    best: tuple[float, float, float, list[np.ndarray]] | None = None
    for deg in LAUNCH_DEG:
        th = math.radians(deg)
        v = release_speed(d, h, th)
        if not math.isfinite(v):
            continue
        direction = math.cos(th) * u + np.array([0.0, 0.0, math.sin(th)])
        step = direction * v * RELEASE_DT
        q_prev = ik(release - step)
        if q_prev is None:
            continue
        need = float(np.max(np.abs(q_rel - q_prev))) / RELEASE_DT
        if best is None or need < best[1]:
            best = (deg, need, v, [release, direction])
    if best is None:
        out["detail"] = "no launch angle has an IK solution through the release point"
        return out
    deg, need, v, (release, direction) = best
    out.update(
        launch_deg=deg,
        release_speed_mps=round(v, 3),
        required_joint_speed=round(need, 3),
        release_position=[round(float(x), 4) for x in release],
    )
    if need > joint_speed_limit:
        out["detail"] = (
            f"a {deg:.0f} deg toss over {d * 1e3:.0f} mm needs {v:.2f} m/s at release, "
            f"{need:.2f} rad/s of joint speed; the limit is {joint_speed_limit:.2f} rad/s"
        )
        return out

    run_up = v * v / (2.0 * accel)
    n = max(2, int(math.ceil(v / accel / RELEASE_DT)))
    start = release - direction * run_up
    waypoints = []
    for i in range(n + 1):
        t = i * (v / accel) / n
        p = start + direction * 0.5 * accel * t * t
        q = ik(p)
        if q is None:
            out["detail"] = f"no IK on the run-up at {np.round(p, 3).tolist()}"
            return out
        waypoints.append({"positions": [float(x) for x in q]})
    ctx.tool("robot.move_to_joints", joint_config=waypoints[0]["positions"], **arm)
    if not still_held():
        out.update(route="dropped_early", mode="toss", detail="lost on the way to the run-up")
        return out
    ctx.tool("robot.execute_trajectory", trajectory={"waypoints": waypoints}, tolerance=0.5, **arm)
    ctx.tool("robot.open_gripper", **arm)
    logger.info("[toss] %.0f deg %.2f m/s over %.0f mm, %.2f rad/s", deg, v, d * 1e3, need)
    return finish("toss")
