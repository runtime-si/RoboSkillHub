"""<What these tools compute.>"""

from __future__ import annotations

from typing import TypedDict

from gap_core.tools import tool


class ExampleResult(TypedDict):
    value: float


@tool(name="{{name}}.example", summary="<One line the coordinator reads.>", tags=("perception",))
def example(x: float) -> ExampleResult:
    return {"value": 2.0 * float(x)}
