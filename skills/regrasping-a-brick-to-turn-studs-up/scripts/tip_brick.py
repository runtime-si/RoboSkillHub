"""Tip a loose brick a quarter turn toward studs-up with a top-down hand: closed fingertips push its upper part
horizontally so it pivots about a bottom edge - over a fixed lip (``lip_distance``, e.g. a baseplate edge) when
the table is too slippery for a free push to tip it. One tip per call; re-classify (classifying-brick-resting-face-
from-depth) and call again: a brick on its side needs one tip, an upside-down one two (onto a side first)."""

from __future__ import annotations

import math
from typing import Any, TypedDict

from gap import NodeContext

PRESETS: dict[str, dict[str, float]] = {
    "lego": {"stud_pitch": 0.008, "stud_height": 0.0017, "brick_height": 0.0096},
    "duplo": {"stud_pitch": 0.016, "stud_height": 0.0045, "brick_height": 0.0192},
}
TIP_REACH = 0.0102
"""Closed fingertips below the TCP [m] (YAM linear_4310 / the LEGO station's hand)."""
TCP_FLOOR = 0.014
"""Lowest tool centre over the table a real YAM station accepts [m]."""
PUSH_HEIGHT_FRACTION = 0.7
"""Push this far up the brick's current height: high enough to tip rather than slide it."""
STANDOFF = 0.006
"""Fingertips start this far off the face they push [m]."""
OVERTRAVEL = 0.008
"""Push this far past the far edge of the brick's footprint [m]: the top must pass over the pivot."""
CLEARANCE = 0.04
"""Approach and leave this far above the push height [m]."""
UP_DONE = 0.9
REFUSALS = ("unreachable", "bad_speed_scale")
ROUTES = ("tipped", "already_studs_up", "blocked")


class Output(TypedDict):
    route: str
    detail: str
    push_direction: dict[str, float]
    push_height: float
    travel: float
    tips_left: int


def _xyz(v: Any) -> list[float]:
    if isinstance(v, dict) and "position" in v:
        v = v["position"]
    if isinstance(v, dict):
        return [float(v[k]) for k in "xyz"]
    return [float(x) for x in v]


def _down(yaw: float) -> dict[str, float]:
    """Tool z down, closing axis (tool y) along ``yaw``."""
    return {"w": 0.0, "x": math.cos(yaw / 2.0), "y": math.sin(yaw / 2.0), "z": 0.0}


def run(
    ctx: NodeContext,
    brick_position: dict[str, float],
    up_vector: dict[str, float],
    yaw_deg: float = 0.0,
    table_height: float = 0.0,
    brick_studs: list[int] | None = None,
    system: str = "lego",
    speed_m_s: float | None = None,
    lip_distance: float | None = None,
) -> Output:
    if system not in PRESETS:
        raise ValueError(f"system must be one of {sorted(PRESETS)}, got {system!r}")
    p = PRESETS[system]
    nx, ny = (int(v) for v in (brick_studs or [2, 4]))
    lx, ly = nx * p["stud_pitch"], ny * p["stud_pitch"]  # along the brick's x and y
    deep = p["brick_height"] + p["stud_height"]  # along its z, studs included
    c = _xyz(brick_position)
    u = _xyz(up_vector)
    out: Output = {
        "route": "blocked",
        "detail": "",
        "push_direction": {"x": 0.0, "y": 0.0, "z": 0.0},
        "push_height": 0.0,
        "travel": 0.0,
        "tips_left": 0,
    }

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        return out

    if u[2] >= UP_DONE:
        return finish("already_studs_up", "the studs already point up")
    yaw = math.radians(float(yaw_deg))
    if u[2] <= -UP_DONE:
        # Upside down: tip across the short (x) side onto a long side; the studs then point along the push.
        d = (math.cos(yaw), math.sin(yaw))
        half, height, tips_left = lx / 2.0, deep, 1
    else:
        # On a side, studs along u: push from the stud face toward -u; the brick pivots about its far bottom
        # edge and its studs swing up.
        n = math.hypot(u[0], u[1]) or 1.0
        d = (-u[0] / n, -u[1] / n)
        half, tips_left = deep / 2.0, 0
        # Its height on a side: the stud axis it lies across (the 2-stud side when the long side is down).
        height = min(lx, ly) if abs(u[2]) < 0.5 else max(lx, ly)
    out["push_direction"] = {"x": d[0], "y": d[1], "z": 0.0}
    push_h = PUSH_HEIGHT_FRACTION * height
    tcp_z = max(table_height + push_h + TIP_REACH, table_height + TCP_FLOOR)
    out["push_height"] = tcp_z - TIP_REACH - table_height
    start = (c[0] - d[0] * (half + STANDOFF), c[1] - d[1] * (half + STANDOFF))
    travel = 2 * half + STANDOFF + OVERTRAVEL
    if lip_distance is not None:
        # Tip over a fixed lip (a baseplate's edge): push until the brick's far side is past it, so its bottom
        # catches on the lip and its top rolls over. On a low-friction table a free push only slides it.
        travel = max(travel, float(lip_distance) + half + STANDOFF + 2 * half + OVERTRAVEL)
    end = (start[0] + d[0] * travel, start[1] + d[1] * travel)
    out["travel"], out["tips_left"] = travel, tips_left
    rot = _down(math.atan2(d[1], d[0]))
    kw = {"speed_m_s": speed_m_s} if speed_m_s else {}

    def go(x: float, y: float, z: float) -> str:
        try:
            reply = ctx.tool(
                "robot.go_to_pose",
                pose={"position": {"x": x, "y": y, "z": z}, "rotation": rot},
                **kw,
            )
        except Exception as exc:  # a station raises where a simulator routes
            return str(exc)
        if isinstance(reply, dict) and reply.get("outcome") in REFUSALS:
            return f"{reply['outcome']}: {reply.get('reason', '')}"
        return ""

    try:
        ctx.tool("robot.close_gripper")
    except Exception as exc:
        return finish("blocked", f"could not close the fingers ({exc})")
    for label, (x, y, z) in (
        ("above the start", (start[0], start[1], tcp_z + CLEARANCE)),
        ("down to the push height", (start[0], start[1], tcp_z)),
        ("the push", (end[0], end[1], tcp_z)),
        ("up and clear", (end[0], end[1], tcp_z + CLEARANCE)),
    ):
        why = go(x, y, z)
        if why:
            return finish("blocked", f"{label} refused: {why}")
    return finish(
        "tipped",
        f"pushed {travel * 1e3:.0f} mm along ({d[0]:+.2f}, {d[1]:+.2f}) at {out['push_height'] * 1e3:.1f} mm"
        + ("; one more tip after this (it lands on a side)" if tips_left else ""),
    )
