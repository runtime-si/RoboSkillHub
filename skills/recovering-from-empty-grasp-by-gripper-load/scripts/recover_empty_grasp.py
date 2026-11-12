# Adapted from lerobot_far (https://github.com/HyuanTan/lerobot_far) @ f8bf405bcd2e,
# src/lerobot/async_inference/smart_robot_client.py (GraspPhase, GripperDecision,
# GripperStateMonitor), MIT License, Copyright (c) 2026 Huoyuan Tan.
# Changes: the monitor's per-step classification of a streamed policy's action
# queue becomes one grasp attempt driven through robot.* connector tools; the
# empty-grasp test is FAR's position gap (the jaws ran to the commanded closed
# stop) read as robot.get_gripper's opening through robot.describe_gripper's width
# fit, with FAR's load threshold applied only when the reply carries an effort;
# FAR's re-inference after each recovery is a fresh grasp at the same pose or,
# with locate_tool, at the object's re-measured position; the rewind replays this
# script's own approach waypoints, not a policy's action history.
"""Grasp, and when the jaws close on nothing, escalate: lift and retry, rewind and retry, go home and retry, give up."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from gap import NodeContext

logger = logging.getLogger(__name__)

EMPTY_MARGIN_M = 0.003
"""A close that ends within this of the jaws' closed stop held nothing [m].
FAR's pos_gap_threshold (obs - cmd under 10 Feetech units on a fully closed
command); 3 mm is the hub's verifying-grasps margin, set from an empty close
measured at 1.8 mm on the YAM linear_4310."""

LOAD_GRASP_THRESHOLD = 0.0
"""Minimum |effort| from robot.get_gripper for a held object, in the reply's own
units; 0 disables it. FAR gates its grasp success on load_grasp_threshold as
well as the gap; the YAM connector does not report a gripper effort, so the gap
decides alone unless a connector does."""

PRE_GRASP_HEIGHT_M = 0.06
"""Height of the approach waypoint above the grasp pose [m]."""

LIFT_DZ_M = 0.05
"""How far the hand rises before a lift retry [m]; FAR's LIFT_RETRY lifts via FK/IK + dZ."""

MAX_TOTAL_RETRIES = 3
"""Retries after the first close; FAR's max_total_retries. The ladder is
lift, rewind, home; a fourth retry repeats home."""

LADDER = ("lift", "rewind", "home")


class Output(TypedDict):
    route: str
    detail: str
    attempts: int
    recovery: str
    widths_m: list[float]
    grasp_position: list[float]


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> list[float]:
    p = pose["position"]
    if isinstance(p, dict):
        return [float(p["x"]), float(p["y"]), float(p["z"])]
    return [float(v) for v in p]


def _at(xyz: list[float], rotation: Any) -> dict[str, Any]:
    return {"position": {"x": xyz[0], "y": xyz[1], "z": xyz[2]}, "rotation": rotation}


def _refusal(reply: Any) -> str:
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def _located(reply: Any) -> list[float] | None:
    """An object position from a locate tool's reply: position, center or obb.center."""
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
    grasp_pose: dict[str, Any],
    locate_tool: str = "",
    locate_args: dict[str, Any] | None = None,
    grasp_offset_m: list[float] | None = None,
    pre_grasp_height_m: float = PRE_GRASP_HEIGHT_M,
    lift_dz_m: float = LIFT_DZ_M,
    max_total_retries: int = MAX_TOTAL_RETRIES,
    empty_margin_m: float = EMPTY_MARGIN_M,
    load_grasp_threshold: float = LOAD_GRASP_THRESHOLD,
    arm_id: int = 0,
) -> Output:
    arm = {"arm_id": int(arm_id)} if arm_id else {}
    rotation = grasp_pose["rotation"]
    target = _xyz(grasp_pose)
    offset = [float(v) for v in (grasp_offset_m or [0.0, 0.0, 0.0])]
    out: Output = {
        "route": "held",
        "detail": "",
        "attempts": 0,
        "recovery": "",
        "widths_m": [],
        "grasp_position": target,
    }
    try:
        fit = (ctx.tool("robot.describe_gripper", **arm) or {}).get("width_fit") or {}
    except Exception as exc:
        logger.info("empty-grasp: no gripper description (%s)", exc)
        fit = {}
    at_open = float(fit.get("at_open_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)

    def move(xyz: list[float]) -> str:
        return _refusal(ctx.tool("robot.go_to_pose_cartesian", pose=_at(xyz, rotation), **arm))

    def above(xyz: list[float], dz: float) -> list[float]:
        return [xyz[0], xyz[1], xyz[2] + dz]

    def close() -> bool:
        """Close; True when something is between the jaws (FAR's GRASP SUCCESS)."""
        ctx.tool("robot.close_gripper", **arm)
        reply = ctx.tool("robot.get_gripper", **arm) or {}
        fraction = float(reply.get("position", 0.0))
        width = at_closed + fraction * (at_open - at_closed) if at_open > at_closed else fraction
        out["widths_m"].append(round(width, 5))
        gap_ok = width > at_closed + empty_margin_m
        effort = reply.get("effort")
        load_ok = (
            load_grasp_threshold <= 0
            or effort is None
            or abs(float(effort)) >= load_grasp_threshold
        )
        logger.info(
            "[grip] attempt=%d width_mm=%.1f effort=%s", out["attempts"], width * 1e3, effort
        )
        return gap_ok and load_ok

    def relocate() -> None:
        nonlocal target
        if not locate_tool:
            return
        try:
            found = _located(ctx.tool(locate_tool, **(locate_args or {})))
        except Exception as exc:
            logger.info("empty-grasp: %s failed (%s); keeping the last target", locate_tool, exc)
            return
        if found is not None:
            target = [found[i] + offset[i] for i in range(3)]
            out["grasp_position"] = target

    def approach() -> str:
        """Pre-grasp waypoint, then down to the target, the jaws open."""
        ctx.tool("robot.open_gripper", **arm)
        return move(above(target, pre_grasp_height_m)) or move(target)

    if why := approach():
        out.update(route="move_failed", detail=f"first approach: {why}")
        return out
    for retry in range(max_total_retries + 1):
        out["attempts"] = retry + 1
        if close():
            out["route"] = "held" if retry == 0 else "recovered"
            out["detail"] = f"held at {out['widths_m'][-1] * 1e3:.1f} mm after {retry} retries"
            return out
        if retry == max_total_retries:
            break
        level = LADDER[min(retry, len(LADDER) - 1)]
        out["recovery"] = level
        logger.info("[recovery] empty grasp %d -> %s", retry + 1, level)
        ctx.tool("robot.open_gripper", **arm)
        here = _xyz(_pose(ctx.tool("robot.get_ee_pose", **arm)))
        if level == "lift":
            why = move(above(here, lift_dz_m))
            relocate()
            why = why or move(above(target, lift_dz_m)) or move(target)
        elif level == "rewind":
            why = move(above(here, pre_grasp_height_m))
            relocate()
            why = why or approach()
        else:
            ctx.tool("robot.go_home", **arm)
            relocate()
            why = approach()
        if why:
            out.update(route="move_failed", detail=f"{level} retry: {why}")
            return out
    ctx.tool("robot.open_gripper", **arm)
    out.update(
        route="gave_up",
        detail=f"{out['attempts']} closes, all on the closed stop: {out['widths_m']}",
    )
    return out
