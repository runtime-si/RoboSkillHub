"""<One line: what this script decides or does.>"""

from __future__ import annotations

from typing import Any, TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    reached_pose: dict[str, Any]


def run(ctx: NodeContext, target_obb: dict[str, Any]) -> Output:
    center = target_obb["center"]
    pose = ctx.tool("robot.get_ee_pose")
    goal = {
        "position": {"x": center["x"], "y": center["y"], "z": center["z"] + 0.10},
        "rotation": pose["rotation"],
    }
    result = ctx.tool("robot.go_to_pose", pose=goal)
    if not result.get("success", True):
        return {"route": "failed", "reached_pose": pose}
    return {"route": "done", "reached_pose": ctx.tool("robot.get_ee_pose")}
