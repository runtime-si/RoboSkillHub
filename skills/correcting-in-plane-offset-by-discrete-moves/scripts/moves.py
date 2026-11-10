"""Hand moves shared by collect_offsets.py and align.py: in-plane translation at a fixed height and
yaw about world +z through the last joint (arms whose solver leaves the wrist roll free reach a
pose without turning; the last joint turns the hand about its tool axis, which points down here)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from gap import NodeContext


def refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or "" (connectors that route instead of raising)."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def rot(q: dict[str, float]) -> np.ndarray:
    x, y, z, w = (float(q[k]) for k in ("x", "y", "z", "w"))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_deg(q: dict[str, float]) -> float:
    """Heading of the tool's x axis in the world xy plane."""
    r = rot(q)
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


class Hand:
    """One arm's hand: read back, move in the plane, turn about +z with the last joint."""

    def __init__(self, ctx: NodeContext, arm_id: int = 0):
        self.ctx = ctx
        self.arm = {"arm_id": int(arm_id)} if int(arm_id) else {}
        self.arm_id = int(arm_id)
        self.gain = None  # world yaw (deg) per last-joint degree, measured on the first turn

    def pose(self) -> dict[str, Any]:
        p = self.ctx.tool("robot.get_ee_pose", **self.arm)
        return p.get("pose", p) if isinstance(p, dict) and "pose" in p else p

    def state(self) -> tuple[np.ndarray, float, dict[str, float]]:
        p = self.pose()
        pos = np.array([float(p["position"][k]) for k in ("x", "y", "z")])
        return pos, yaw_deg(p["rotation"]), p["rotation"]

    def joints(self) -> np.ndarray:
        arms = self.ctx.tool("robot.get_observation").get("arms") or []
        q = arms[self.arm_id].get("joint_state") if self.arm_id < len(arms) else None
        if isinstance(q, dict):
            q = q.get("positions")
        return np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)

    def move_xy(
        self,
        x: float,
        y: float,
        z: float,
        rotation: dict[str, float],
        tool: str = "robot.go_to_pose",
        tolerance: float | None = None,
    ) -> str:
        """Move the tool centre. A servo connector may count a few millimetres off as arrived (its
        default ball); ``tolerance`` [m] is passed only when
        set (a servo connector reports a 1 mm step as reached without moving at its 4 mm default)."""
        pose = {"position": {"x": float(x), "y": float(y), "z": float(z)}, "rotation": rotation}
        knobs = {"tolerance": float(tolerance)} if tolerance else {}
        try:
            return refusal(self.ctx.tool(tool, pose=pose, **self.arm, **knobs))
        except Exception as exc:  # a refused move is routed, not raised
            return str(exc) or type(exc).__name__

    def turn(self, dyaw_deg: float, joint: int = -1) -> str:
        """Turn the hand by ``dyaw_deg`` about world +z through one joint (default the last)."""
        if abs(dyaw_deg) < 1e-6:
            return ""
        q = self.joints()
        if q.size == 0:
            return "no joint reading"
        _, before, _ = self.state()
        gain = (
            self.gain if self.gain is not None else -1.0
        )  # a down-pointing tool turns against its joint
        q = q.copy()
        q[joint] += math.radians(dyaw_deg / gain)
        try:
            why = refusal(
                self.ctx.tool(
                    "robot.move_to_joints", joint_config={"positions": q.tolist()}, **self.arm
                )
            )
        except Exception as exc:
            why = str(exc) or type(exc).__name__
        if why:
            return why
        _, after, _ = self.state()
        moved = wrap(after - before)
        if self.gain is None and abs(dyaw_deg) >= 0.5 and abs(moved) > 0.25 * abs(dyaw_deg):
            self.gain = gain * moved / dyaw_deg
        return ""
