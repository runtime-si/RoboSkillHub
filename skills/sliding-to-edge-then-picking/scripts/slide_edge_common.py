# Shared by slide_to_edge.py and pick_overhang.py; see slide_to_edge.py for the
# Ravens source, licence and changes.
"""Pose helpers and the last-joint jaw turn for sliding-to-edge-then-picking."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

YAW_TOLERANCE_DEG = 8.0
"""How far the jaws' heading may be off the wanted one before the last joint
finishes the turn [deg]: on a 40 mm slab an 8 deg skew moves a blade end 3 mm."""

TURN_TRIES = 3
"""Last-joint turns tried, each read back (stacking-objects)."""

ROLL_LIMITS = (-2.0944, 2.0944)
"""Last joint's range [rad] when the caller states none: the i2rt YAM's joint 6
(+-120 deg; stacking-objects' value)."""

LIMIT_MARGIN = 0.05
"""Kept clear of each roll limit [rad]."""


def pose_of(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def vec(v: Any) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def matrix(q: dict[str, float] | None) -> np.ndarray:
    q = q or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or "" (connectors that route a
    refusal instead of raising)."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def wrap(deg: float, period: float = 360.0) -> float:
    return (float(deg) + period / 2.0) % period - period / 2.0


def yaw_between(actual: dict[str, float], wanted: dict[str, float]) -> float:
    """How far [deg] the hand is turned about the vertical from the wanted
    rotation (both top-down)."""
    m = matrix(actual) @ matrix(wanted).T
    return math.degrees(math.atan2(float(m[1, 0]), float(m[0, 0])))


class Hand:
    """One arm's moves, routed rather than raised, judged by reading the hand back."""

    def __init__(self, ctx: NodeContext, arm_id: int, reach_tolerance: float) -> None:
        self.ctx = ctx
        self.arm_id = int(arm_id)
        self.on_arm: dict[str, Any] = {"arm_id": self.arm_id} if self.arm_id else {}
        self.reach_tolerance = float(reach_tolerance)
        self.notes: list[str] = []

    def tool(self, name: str, **kwargs: Any) -> Any:
        return self.ctx.tool(name, **kwargs, **self.on_arm)

    def move(self, tool: str, position: Any, rotation: dict[str, float], **extra: Any) -> str:
        try:
            return refusal(
                self.tool(tool, pose={"position": vec(position), "rotation": rotation}, **extra)
            )
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc) or type(exc).__name__

    def here(self) -> dict[str, Any]:
        return pose_of(self.tool("robot.get_ee_pose"))

    def where(self) -> np.ndarray:
        return xyz(self.here())

    def go(self, tool: str, position: Any, rotation: dict[str, float]) -> str:
        """A free move: "" when the hand arrived within reach_tolerance (a servo's
        timeout that ended that close has arrived), otherwise why not."""
        why = self.move(tool, position, rotation)
        off = float(np.linalg.norm(self.where() - np.asarray(position, dtype=float)))
        if (why and not why.startswith("timeout")) or off > self.reach_tolerance:
            return f"{why or 'ended off'}; the hand is {off * 1e3:.0f} mm from it"
        return ""

    def joints(self) -> np.ndarray:
        try:
            arms = self.ctx.tool("robot.get_observation").get("arms") or []
        except Exception as exc:
            logger.info("sliding-to-edge-then-picking: no joint reading (%s)", exc)
            return np.zeros(0)
        if self.arm_id >= len(arms):
            return np.zeros(0)
        q = arms[self.arm_id].get("joint_state")
        if isinstance(q, dict):
            q = q.get("positions")
        return np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)

    def align(
        self,
        wanted: dict[str, float],
        at: Any,
        roll_limits: list[float] | None = None,
        tolerance_deg: float = YAW_TOLERANCE_DEG,
    ) -> dict[str, float]:
        """Turn the jaws onto the wanted heading (mod 180 deg: the jaws are
        symmetric) on the last joint when the arm arrived unturned (a solver that
        leaves the wrist roll free), re-centre over ``at``, and return the
        hand's rotation as it now is."""
        now = self.here()
        lo, hi = (float(v) for v in (roll_limits or ROLL_LIMITS))
        gain = -1.0 if matrix(wanted)[2, 2] < 0 else 1.0
        for _ in range(TURN_TRIES):
            residual = -wrap(yaw_between(now["rotation"], wanted), 180.0)
            if abs(residual) <= float(tolerance_deg):
                break
            q = self.joints()
            if q.size == 0:
                self.notes.append(f"jaws {residual:+.0f} deg off, no joint reading to turn them")
                break
            room = (lo + LIMIT_MARGIN - q[-1], hi - LIMIT_MARGIN - q[-1])
            options = [residual, residual - math.copysign(180.0, residual)]
            fits = [r for r in options if room[0] <= math.radians(r) / gain <= room[1]]
            if not fits:
                self.notes.append(f"the last joint has no room to turn {residual:+.0f} deg")
                break
            command = math.radians(fits[0]) / gain
            target_q = q.copy()
            target_q[-1] += command
            try:
                why = refusal(
                    self.tool("robot.move_to_joints", joint_config={"positions": target_q.tolist()})
                )
            except Exception as exc:
                why = str(exc) or type(exc).__name__
            if why:
                self.notes.append(f"wrist turn: {why}")
                break
            was = now["rotation"]
            now = self.here()
            moved = math.radians(wrap(yaw_between(now["rotation"], was)))
            if abs(command) > math.radians(2.0) and 0.5 < abs(moved / command) < 1.5:
                gain = moved / command
            why = self.go("robot.go_to_pose", at, now["rotation"])
            if why:
                self.notes.append(f"re-centring after the wrist turn: {why}")
            now = self.here()
        return now["rotation"]
