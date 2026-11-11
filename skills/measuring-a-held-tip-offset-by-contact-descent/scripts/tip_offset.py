"""Lower the hand over a free patch of a known surface in small steps until the held part's tip touches it (the
hand stops following its command), and read the tip's offset from the tool centre off the stop height."""

from __future__ import annotations

import math
import statistics
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

STEP = 0.0005
"""Descent step [m]."""

STEP_TOLERANCE = 0.0002
"""Arrival ball asked of each step [m]; under the step, so a blocked hand reads as lag, not as arrived."""

CONTACT_LAG = 0.0006
"""Lag of the read-back behind the command, over the free-descent baseline, that reads as touching [m]."""

BASELINE_STEPS = 6
"""Free steps whose lag makes the baseline."""

RETREAT = 0.01
"""Rise after the touch [m]."""


class Output(TypedDict):
    route: str
    detail: str
    tip_offset_m: float
    contact_tcp_z: float
    descended_m: float
    steps: int


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _approach(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def tip_offset(contact_tcp_z: float, surface_z: float, approach: np.ndarray) -> float:
    """Tip distance from the tool centre along the approach axis, the tip on a surface at ``surface_z``."""
    down = -float(approach[2])
    return (contact_tcp_z - surface_z) / max(down, 1e-6)


def run(
    ctx: NodeContext,
    surface_z: float,
    max_descent_m: float = 0.10,
    step_m: float = STEP,
    contact_lag_m: float = CONTACT_LAG,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    out: Output = {
        "route": "no_contact",
        "detail": "",
        "tip_offset_m": 0.0,
        "contact_tcp_z": 0.0,
        "descended_m": 0.0,
        "steps": 0,
    }
    start = _pose(ctx.tool("robot.get_ee_pose", **on_arm))
    rot = start["rotation"]
    p0 = _xyz(start)
    approach = _approach(rot)
    if -approach[2] < 0.5:
        out.update(
            route="invalid_input",
            detail=f"the approach axis points {math.degrees(math.acos(max(-1.0, min(1.0, -approach[2])))):.0f} deg off straight down",
        )
        return out
    lags: list[float] = []
    cmd_z = float(p0[2])
    n = int(math.ceil(float(max_descent_m) / float(step_m)))
    for i in range(n):
        cmd_z -= float(step_m)
        target = {"position": {"x": float(p0[0]), "y": float(p0[1]), "z": cmd_z}, "rotation": rot}
        try:
            ctx.tool("robot.go_to_pose", pose=target, tolerance=STEP_TOLERANCE, **on_arm)
        except Exception as exc:  # a refused step ends the descent
            out.update(route="unreachable", detail=f"step {i} refused: {exc}")
            return out
        z = float(_xyz(_pose(ctx.tool("robot.get_ee_pose", **on_arm)))[2])
        lag = z - cmd_z
        out["steps"] = i + 1
        base = statistics.median(lags) if len(lags) >= BASELINE_STEPS else None
        if base is not None and lag - base > float(contact_lag_m):
            out["contact_tcp_z"] = z
            out["descended_m"] = float(p0[2]) - z
            out["tip_offset_m"] = tip_offset(z, float(surface_z), approach)
            out["route"] = "measured"
            out["detail"] = (
                f"touched after {out['descended_m'] * 1e3:.1f} mm ({lag * 1e3:.2f} mm lag over a {base * 1e3:.2f} mm "
                f"baseline): tip {out['tip_offset_m'] * 1e3:.1f} mm from the tool centre along the approach"
            )
            back = {
                "position": {"x": float(p0[0]), "y": float(p0[1]), "z": z + RETREAT},
                "rotation": rot,
            }
            ctx.tool("robot.go_to_pose_cartesian", pose=back, **on_arm)
            return out
        if len(lags) < BASELINE_STEPS or (
            base is not None and lag - base < float(contact_lag_m) / 3
        ):
            lags.append(lag)
    out["descended_m"] = float(max_descent_m)
    out["detail"] = f"no touch over {float(max_descent_m) * 1e3:.0f} mm"
    return out
