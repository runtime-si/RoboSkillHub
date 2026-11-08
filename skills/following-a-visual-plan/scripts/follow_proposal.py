"""Fly a proposal from propose_target_from_plan: rise, travel at the safe height, descend to the goal."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

REACH_TOLERANCE = 0.015
"""How far the hand may end from the goal [m]: no tighter than a real YAM
station's 12 mm per-move bound (placing-with-contact-descent's value)."""


class Output(TypedDict):
    route: str
    detail: str
    final_position: dict[str, float]
    position_error: float
    rotation_error_deg: float
    legs_done: list[str]


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(p: dict[str, Any]) -> np.ndarray:
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _angle_deg(a: dict[str, float], b: dict[str, float]) -> float:
    qa = np.array([a["w"], a["x"], a["y"], a["z"]], dtype=np.float64)
    qb = np.array([b["w"], b["x"], b["y"], b["z"]], dtype=np.float64)
    dot = abs(float(np.dot(qa, qb))) / max(float(np.linalg.norm(qa) * np.linalg.norm(qb)), 1.0e-12)
    return float(np.degrees(2.0 * np.arccos(min(1.0, dot))))


def _refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or "" (a simulator servo routes an
    unreachable pose as an outcome instead of raising)."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def run(
    ctx: NodeContext,
    proposal: dict[str, Any],
    arm_id: int = 0,
    tolerance: float | None = None,
    max_steps: int | None = None,
    reach_tolerance: float = REACH_TOLERANCE,
) -> Output:
    arm = {"arm_id": int(arm_id)} if arm_id else {}
    knobs: dict[str, Any] = dict(arm)
    if tolerance is not None:
        knobs["tolerance"] = float(tolerance)
    if max_steps is not None:
        knobs["max_steps"] = int(max_steps)
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "final_position": {},
        "position_error": -1.0,
        "rotation_error_deg": -1.0,
        "legs_done": [],
    }
    goal, hover = proposal["goal_pose"], proposal["hover_pose"]
    rotation = goal["rotation"]
    travel = float(proposal["travel_z"])

    start = _pose(ctx.tool("robot.get_ee_pose", **arm))
    here = _xyz(start["position"])
    legs: list[tuple[str, str, np.ndarray]] = []
    if here[2] < travel - 1.0e-3:
        # Up in place first, so no travel leg sweeps through what is under the route.
        legs.append(("rise", "robot.go_to_pose", np.array([here[0], here[1], travel])))
    for i, w in enumerate(proposal.get("waypoints") or []):
        legs.append((f"waypoint {i + 1}", "robot.go_to_pose_cartesian", _xyz(w)))
    legs.append(("hover", "robot.go_to_pose_cartesian", _xyz(hover["position"])))
    legs.append(("descend", "robot.go_to_pose_cartesian", _xyz(goal["position"])))

    timeouts: list[str] = []
    for name, tool, p in legs:
        pose = {
            "position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
            "rotation": rotation,
        }
        # go_to_pose_cartesian takes no convergence knobs on gap's connector.
        extra = knobs if tool == "robot.go_to_pose" else arm
        try:
            why = _refusal(ctx.tool(tool, pose=pose, **extra))
        except Exception as exc:  # a real station raises on a refused move
            why = str(exc)
        if why.startswith("timeout"):
            # A servo that ran out of steps short of its own 4 mm ball may still be
            # within this skill's bound: the arm's reading decides.
            now = _xyz(_pose(ctx.tool("robot.get_ee_pose", **arm))["position"])
            short = float(np.linalg.norm(now - p))
            if short <= float(reach_tolerance):
                timeouts.append(f"{name} {short * 1e3:.1f} mm short")
                why = ""
        if why:
            logger.info("following-a-visual-plan: %s move failed: %s", name, why)
            out["detail"] = f"{name} move failed: {why}"
            return out
        out["legs_done"].append(name)

    reached = _pose(ctx.tool("robot.get_ee_pose", **arm))
    error = float(np.linalg.norm(_xyz(reached["position"]) - _xyz(goal["position"])))
    out.update(
        final_position=dict(reached["position"]),
        position_error=error,
        rotation_error_deg=_angle_deg(reached["rotation"], rotation),
    )
    if error > float(reach_tolerance):
        out["detail"] = (
            f"the hand ended {error * 1e3:.1f} mm from the goal (bound {float(reach_tolerance) * 1e3:.0f} mm)"
        )
        return out
    out["route"] = "arrived"
    out["detail"] = f"at the goal within {error * 1e3:.1f} mm"
    if timeouts:
        out["detail"] += f" (move timed out: {', '.join(timeouts)})"
    return out
