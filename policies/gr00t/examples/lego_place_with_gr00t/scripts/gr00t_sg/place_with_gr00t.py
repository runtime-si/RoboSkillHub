"""Hand the place-and-press step to GR00T, steered by marks: grasp at the perceived brick, release at its seat.

The graph has already decided what (the brick perception found) and where (the seat the stud grid gives);
this node marks both on the top camera's view and lets gr00t.run_skill do the how. Success here is one
grasp-and-release cycle; whether the brick is seated is the next node's (verifying-brick-seated-by-height).
"""

from typing import Any, TypedDict

from gap import NodeContext

RELEASE_ABOVE_SEAT = 0.03
"""The release mark sits this far [m] over the seat point: a DUPLO 2x2's height (19.2 mm) plus room, so the
mark lands on the brick's top face in the top view rather than on the plate beside it."""


class Output(TypedDict):
    route: str
    marks: list[dict[str, Any]]
    dropped_marks: list[dict[str, Any]]
    rows: int
    chunks: int
    latency_ms_mean: float
    detail: str


def run(
    ctx: NodeContext,
    target_obb: dict[str, Any],
    seat_position: dict[str, float],
    part_name: str,
    max_steps: int = 600,
    arms: str = "both",
    camera: str = "top",
) -> Output:
    c = target_obb["center"]
    seat = {
        "x": seat_position["x"],
        "y": seat_position["y"],
        "z": seat_position["z"] + RELEASE_ABOVE_SEAT,
    }
    obs = ctx.tool("robot.get_observation")
    projected = ctx.tool(
        "gr00t.marks_from_points",
        observation=obs,
        points=[{"kind": "grasp", "point": c}, {"kind": "release", "point": seat}],
        camera=camera,
    )
    # World-point marks are re-projected by run_skill every chunk; the pixel ones above are for the trace.
    marks = [
        {"kind": "grasp", "point": c, "camera": camera},
        {"kind": "release", "point": seat, "camera": camera},
    ]
    r = ctx.tool(
        "gr00t.run_skill",
        instruction=f"Pick up the {part_name} and press it onto the plate at the blue dot.",
        marks=marks,
        max_steps=max_steps,
        success_check="gripper_cycle",
        arms=arms,
    )
    return {
        "route": r["route"],
        "marks": projected["marks"],
        "dropped_marks": projected["dropped"],
        "rows": r["steps"],
        "chunks": r["chunks"],
        "latency_ms_mean": r["latency_ms_mean"],
        "detail": r["detail"],
    }
