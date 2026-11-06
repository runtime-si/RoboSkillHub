"""Hover the hand above a target's box, keeping its rotation, and check it got there."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

CLEARANCE = 0.12
"""Height of the tool centre over the box's top [m]: graph-as-policy's steered-policy
hand-off value (``obb_clearance``), clear of a gripper's open fingers above a
tabletop object."""

REACH_TOLERANCE = 0.01
"""How far the hand may end from the hover point [m]. A real YAM station fails a
move whose tool ends over 12 mm from its target; this stays inside that."""

ROT_TOLERANCE_DEG = 4.0
"""How far the hand's rotation may end from the one asked for [deg]. Above
a simulator IK servo's arrival ball (3 deg; a YAM twin's hover ended 2.15 deg
off) and a real YAM station's per-move bound (0.04 rad, 2.3 deg), so a move the
connector accepts is not failed here."""


class Output(TypedDict):
    route: str
    hover_pose: dict[str, Any]
    position_error: float
    rotation_error_deg: float
    detail: str


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _angle_deg(a: dict[str, float], b: dict[str, float]) -> float:
    qa = np.array([a["w"], a["x"], a["y"], a["z"]], dtype=np.float64)
    qb = np.array([b["w"], b["x"], b["y"], b["z"]], dtype=np.float64)
    dot = abs(float(np.dot(qa, qb))) / max(float(np.linalg.norm(qa) * np.linalg.norm(qb)), 1.0e-12)
    return float(np.degrees(2.0 * np.arccos(min(1.0, dot))))


def _refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or "" -- connectors that route an
    unreachable pose instead of raising (a servo's outcome) say so here."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def run(
    ctx: NodeContext,
    target_obb: dict[str, Any],
    clearance: float = CLEARANCE,
    min_z: float | None = None,
    rotation: dict[str, float] | None = None,
    arm_id: int = 0,
    tolerance: float | None = None,
    max_steps: int | None = None,
    reach_tolerance: float = REACH_TOLERANCE,
    rot_tolerance_deg: float = ROT_TOLERANCE_DEG,
) -> Output:
    arm = {"arm_id": int(arm_id)} if arm_id else {}
    move: dict[str, Any] = dict(arm)
    if tolerance is not None:
        move["tolerance"] = float(tolerance)
    if max_steps is not None:
        move["max_steps"] = int(max_steps)

    start = _pose(ctx.tool("robot.get_ee_pose", **arm))
    keep = rotation if rotation is not None else start["rotation"]
    centre, extent = target_obb["center"], target_obb["extent"]
    z = float(centre["z"]) + float(extent["z"]) + float(clearance)
    if min_z is not None:
        z = max(z, float(min_z))
    hover = {
        "position": {"x": float(centre["x"]), "y": float(centre["y"]), "z": z},
        "rotation": keep,
    }

    legs = []
    if float(start["position"]["z"]) < z - 1.0e-3:
        # Rise in place first, so the lateral leg never sweeps through the target.
        legs.append(("rise", {"position": {**start["position"], "z": z}, "rotation": keep}))
    legs.append(("hover", hover))
    for leg, pose in legs:
        try:
            why = _refusal(ctx.tool("robot.go_to_pose", pose=pose, **move))
        except Exception as exc:  # a refused or failed move is this skill's unreachable
            why = str(exc)
        if why:
            logger.info("approaching-above-a-target: %s move failed: %s", leg, why)
            return {
                "route": "unreachable",
                "hover_pose": hover,
                "position_error": -1.0,
                "rotation_error_deg": -1.0,
                "detail": f"{leg} move failed: {why}",
            }

    reached = _pose(ctx.tool("robot.get_ee_pose", **arm))
    error = float(np.linalg.norm(_xyz(reached) - _xyz(hover)))
    turn = _angle_deg(reached["rotation"], keep)
    out: Output = {
        "route": "above",
        "hover_pose": hover,
        "position_error": error,
        "rotation_error_deg": turn,
        "detail": "",
    }
    if error > float(reach_tolerance):
        out["route"] = "unreachable"
        out["detail"] = (
            f"the hand ended {error * 1e3:.1f} mm from the hover point "
            f"(bound {float(reach_tolerance) * 1e3:.0f} mm)"
        )
    elif turn > float(rot_tolerance_deg):
        out["route"] = "unreachable"
        out["detail"] = (
            f"the hand ended turned {turn:.1f} deg (bound {float(rot_tolerance_deg):.1f} deg)"
        )
    return out
