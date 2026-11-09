"""Invoke the workcell capability through the same tool used by an agent."""

from __future__ import annotations

import json
from typing import Any, TypedDict


class Output(TypedDict):
    result_json: str
    status: str


def run(ctx: Any, request_json: str) -> Output:
    result_json = ctx.tool("workcell.invoke", capability="set_gripper", request_json=request_json)
    if not isinstance(result_json, str):
        raise TypeError("workcell.invoke must return a JSON string")
    result = json.loads(result_json)
    if not isinstance(result, dict) or result.get("status") not in (
        "succeeded",
        "failed",
        "unverified",
        "rejected",
    ):
        raise ValueError("workcell.invoke returned no valid terminal status")
    return {"result_json": result_json, "status": result["status"]}
