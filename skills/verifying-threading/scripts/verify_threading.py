"""Has the rope gone through the loop? Judged from geometry the graph already holds.

The rope's centreline and the loop's opening both come from upstream
perception, so this node reads no camera and no simulator state: the verdict
is the same on a real arm and in any simulator. The geometry is
``predicates.threaded_through``; this script parses what the perception skills
write, routes on the verdict and says how the call was made.
"""

import json
from typing import Any, TypedDict

from gap import NodeContext


class Output(TypedDict):
    exit: str
    threaded: bool
    crossings: int
    through_m: float
    clearance_m: float
    detail: str


def _centerline(rope: Any) -> dict[str, Any]:
    """A ``Centerline``, or the JSON text ``perceiving-deformable-linear-objects``
    writes: a list of ``[x, y, z]`` points, or an object with ``points``."""
    value = json.loads(rope) if isinstance(rope, str) and rope.strip() else rope
    if isinstance(value, dict):
        return {**value, "points": list(value.get("points") or [])}
    return {"points": list(value or [])}


def _not_threaded(detail: str) -> Output:
    return {
        "exit": "not_threaded",
        "threaded": False,
        "crossings": 0,
        "through_m": 0.0,
        "clearance_m": 0.0,
        "detail": detail,
    }


def run(ctx: NodeContext, rope: Any, loop: Any, min_through_m: float = 0.01) -> Output:
    centerline = _centerline(rope)
    feature = json.loads(loop) if isinstance(loop, str) and loop.strip() else (loop or {})
    if len(centerline["points"]) < 2:
        return _not_threaded("no rope centreline to judge")
    if not feature.get("radius_inner"):
        return _not_threaded("the loop has no radius_inner; perceive it as a loop or an aperture")

    result = ctx.tool(
        "predicates.threaded_through", rope=centerline, loop=feature, min_through_m=float(min_through_m)
    )
    through_mm = result["through_m"] * 1000.0
    clearance_mm = result["clearance_m"] * 1000.0
    if result["threaded"]:
        detail = f"through: {through_mm:.0f} mm past the ring, {clearance_mm:.0f} mm clearance"
    elif result["passes"] == 0:
        detail = f"misses the opening by {-clearance_mm:.0f} mm"
    elif result["crossings"] == 0:
        detail = f"went through and came back ({result['passes']} passes, net 0)"
    else:
        detail = f"only {through_mm:.0f} mm past the ring, under {float(min_through_m) * 1000.0:.0f} mm"
    return {
        "exit": result["verdict"],
        "threaded": bool(result["threaded"]),
        "crossings": int(result["crossings"]),
        "through_m": float(result["through_m"]),
        "clearance_m": float(result["clearance_m"]),
        "detail": detail,
    }
