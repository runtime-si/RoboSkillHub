# The pattern is geodude's two-arm pickup (github.com/personalrobotics/geodude at 80edf9d,
# src/geodude/robot.py Robot.pickup; earlier releases' bt/subtrees.py geodude_pickup_bimanual, a
# behaviour-tree Selector over two plan-grasp Sequences; MIT): try the arm on the object's side first and
# fall back to the other, keeping each attempt's failure reason. No code is copied; each attempt here is
# an IK preview, a top-down approach, descent, close, lift and a jaw-opening check through robot.* tools.
"""Pick an object with the nearer arm, and with the other arm if the first cannot reach or misses it."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

APPROACH = 0.08
"""Height of the pre-grasp over the grasp point [m]."""

LIFT = 0.08
"""Lift after closing [m]."""

EMPTY_FRACTION = 0.05
"""A jaw opening (robot.get_gripper position, 0 closed .. 1 open) at or under this after closing means the
hand closed on nothing."""

ROUTES = ("picked_by_first", "picked_by_second", "both_failed")


class Output(TypedDict):
    route: str
    detail: str
    arm_used: int
    order: list[int]
    attempts: list[dict[str, Any]]


def _xyz(v: Any) -> np.ndarray:
    v = v.get("position", v) if isinstance(v, dict) else v
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=float).reshape(3)


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _refusal(reply: Any) -> str:
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or "the move reported failure")
    return ""


def _attempt(
    ctx: NodeContext, arm: int, grasp: np.ndarray, approach: float, lift: float
) -> tuple[str, str]:
    """('picked' | 'unreachable' | 'missed', why) for one arm."""
    on = {"arm_id": int(arm)} if arm else {}
    try:
        rot = ctx.tool("robot.grasp_frame", **on)["rotation"]
    except Exception:
        rot = _pose(ctx.tool("robot.get_ee_pose", **on))["rotation"]
    poses = [
        ("pre-grasp", grasp + [0.0, 0.0, approach]),
        ("grasp", grasp),
        ("lift", grasp + [0.0, 0.0, lift]),
    ]
    for name, p in poses[:2]:
        pose = {"position": dict(zip("xyz", map(float, p), strict=True)), "rotation": rot}
        try:
            r = ctx.tool("robot.solve_ik", pose=pose, **on)
        except Exception as exc:
            return "unreachable", f"{name}: IK refused ({exc})"
        if isinstance(r, dict) and (
            r.get("success") is False or r.get("within_servo_tolerance") is False
        ):
            return "unreachable", f"{name}: out of reach (IK preview)"
    ctx.tool("robot.open_gripper", **on)
    for name, p in poses:
        pose = {"position": dict(zip("xyz", map(float, p), strict=True)), "rotation": rot}
        try:
            why = _refusal(ctx.tool("robot.go_to_pose", pose=pose, **on))
        except Exception as exc:
            why = str(exc) or type(exc).__name__
        if why:
            ctx.tool("robot.open_gripper", **on)
            return "unreachable", f"{name}: {why}"
        if name == "grasp":
            ctx.tool("robot.close_gripper", **on)
    width = float(ctx.tool("robot.get_gripper", **on).get("position", 0.0))
    if width <= EMPTY_FRACTION:
        ctx.tool("robot.open_gripper", **on)
        return "missed", f"the jaws closed to {width:.2f} (empty)"
    return "picked", f"holding (jaw opening {width:.2f})"


def run(
    ctx: NodeContext,
    target_position: dict[str, float],
    arms: list[int] | None = None,
    order: str = "nearest",
    grasp_height: float = 0.0,
    approach_height: float = APPROACH,
    lift_height: float = LIFT,
) -> Output:
    arms = [int(a) for a in (arms or [0, 1])]
    if len(arms) != 2 or arms[0] == arms[1]:
        raise ValueError("arms must be two different arm ids")
    if order not in ("nearest", "given"):
        raise ValueError(f"order must be 'nearest' or 'given', got {order!r}")
    grasp = _xyz(target_position) + [0.0, 0.0, float(grasp_height)]
    if order == "nearest":
        dist = {}
        for a in arms:
            try:
                dist[a] = float(
                    np.linalg.norm(
                        _xyz(_pose(ctx.tool("robot.get_ee_pose", **({"arm_id": a} if a else {}))))[
                            :2
                        ]
                        - grasp[:2]
                    )
                )
            except Exception:
                dist[a] = math.inf
        arms = sorted(arms, key=lambda a: dist[a])
    out: Output = {
        "route": "both_failed",
        "detail": "",
        "arm_used": -1,
        "order": arms,
        "attempts": [],
    }
    for i, arm in enumerate(arms):
        result, why = _attempt(ctx, arm, grasp, float(approach_height), float(lift_height))
        out["attempts"].append({"arm": arm, "result": result, "detail": why})
        logger.info("trying-arms-in-fallback-order: arm %d %s (%s)", arm, result, why)
        if result == "picked":
            out["route"] = "picked_by_first" if i == 0 else "picked_by_second"
            out["arm_used"] = arm
            out["detail"] = f"arm {arm}: {why}" + (
                f"; arm {arms[0]} {out['attempts'][0]['result']}: {out['attempts'][0]['detail']}"
                if i
                else ""
            )
            return out
    out["detail"] = "; ".join(
        f"arm {a['arm']} {a['result']}: {a['detail']}" for a in out["attempts"]
    )
    return out
