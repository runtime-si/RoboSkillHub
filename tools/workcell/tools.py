"""Expose the installed workcell runtime as graph-callable JSON tools."""

from __future__ import annotations

import json

from gap_core.tools import tool


@tool(
    name="workcell.describe",
    summary="Describe the configured station and workcell capabilities without motion.",
    tags=("workcell",),
)
def describe() -> str:
    from roboskillhub.runtime.api import describe as runtime_describe

    return json.dumps(runtime_describe(), ensure_ascii=False, allow_nan=False)


@tool(
    name="workcell.invoke",
    summary="Call one adapted workcell capability and return its status and evidence as JSON.",
    tags=("workcell",),
)
def invoke(capability: str, request_json: str) -> str:
    from roboskillhub.runtime.api import invoke as runtime_invoke

    request = json.loads(request_json)
    if not isinstance(request, dict):
        raise ValueError("request_json must encode a JSON object")
    result = runtime_invoke(capability=capability, request=request)
    return json.dumps(result, ensure_ascii=False, allow_nan=False)
