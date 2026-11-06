"""Open the hand, drive the joints back to a recorded configuration, and check arrival."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

MAX_ERROR = 0.08
"""Arrival bound on every joint [rad]: just above a real YAM station's 0.075 rad
single-arm settle tolerance (SKILL.md has the measurement)."""


class Output(TypedDict):
    route: str
    max_joint_error: float
    detail: str


def _positions(joints: Any) -> np.ndarray:
    if isinstance(joints, dict):
        joints = joints.get("positions")
    return np.asarray(joints if joints is not None else [], dtype=np.float64).reshape(-1)


def run(
    ctx: NodeContext,
    recorded_joints: dict[str, Any],
    open_first: bool = True,
    arm_id: int = 0,
    tolerance: float | None = None,
    max_steps: int | None = None,
    max_error: float = MAX_ERROR,
) -> Output:
    target = _positions(recorded_joints)
    if target.size == 0:
        return {"route": "not_returned", "max_joint_error": -1.0, "detail": "no recorded joints"}

    if open_first:
        ctx.tool("robot.open_gripper")

    move: dict[str, Any] = {"joint_config": {"positions": target}}
    if tolerance is not None:
        move["tolerance"] = float(tolerance)
    if max_steps is not None:
        move["max_steps"] = int(max_steps)
    try:
        ctx.tool("robot.move_to_joints", **move)
    except Exception as exc:  # a refused or failed move is this skill's not_returned
        logger.info("returning-to-a-recorded-pose: move failed: %s", exc)
        return {"route": "not_returned", "max_joint_error": -1.0, "detail": f"move failed: {exc}"}

    arms = ctx.tool("robot.get_observation").get("arms") or []
    if arm_id >= len(arms):
        return {
            "route": "not_returned",
            "max_joint_error": -1.0,
            "detail": f"no reading for arm {arm_id}",
        }
    now = _positions(arms[arm_id].get("joint_state"))
    n = min(len(now), len(target))
    if n == 0:
        return {"route": "not_returned", "max_joint_error": -1.0, "detail": "empty joint reading"}
    error = float(np.max(np.abs(now[:n] - target[:n])))
    if error > float(max_error):
        worst = int(np.argmax(np.abs(now[:n] - target[:n])))
        return {
            "route": "not_returned",
            "max_joint_error": error,
            "detail": f"joint {worst} ended {error:.3f} rad off (bound {float(max_error):.3f})",
        }
    return {"route": "returned", "max_joint_error": error, "detail": ""}
