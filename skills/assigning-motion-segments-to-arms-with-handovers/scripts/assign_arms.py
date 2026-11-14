# Adapted from Bimanual-Manipulation (github.com/Sukhvansh2004/Bimanual-Manipulation at d7940e7e2819,
# src/bimanual_manipulation/planning/assignment/solver.py and cost.py, MIT; licence in
# LICENSE.bimanual-manipulation): a Viterbi-style DP over trajectory waypoints whose state is the arm
# carrying the object, a per-waypoint cost (infinite where that arm has no IK, else the joint motion from
# the previous waypoint) and a switching cost (a handover, possible only where both arms reach). Changed:
# IK through robot.solve_ik for any two arms instead of the repo's backends; one IK solution per arm and
# waypoint (no multi-solution states or beam pruning); grasp feasibility, manipulability and collision
# terms and the leader-follower execution are not ported.
"""Choose which arm carries an object along each part of a trajectory, and where it is handed over."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

HANDOVER_COST = 2.0
"""Cost of one handover, in radians of summed joint motion (upstream's switching cost has the same
role): a handover costs a stop, a regrasp and the risk of a drop, so it wins only when a single arm
would move ~2 rad more or cannot reach at all."""

JOINT_WEIGHT = 1.0
"""Weight of the summed absolute joint change between consecutive waypoints [1/rad]."""

ROUTES = ("assignment_found", "handover_required", "infeasible")


class Output(TypedDict):
    route: str
    detail: str
    assignment: list[int]
    handovers: list[dict[str, Any]]
    n_handovers: int
    total_cost: float
    unreachable_waypoints: list[int]


def _xyz(p: Any) -> dict[str, float]:
    p = p.get("position", p) if isinstance(p, dict) else p
    if isinstance(p, dict):
        return {k: float(p[k]) for k in "xyz"}
    a = np.asarray(p, dtype=float).reshape(-1)
    return {"x": float(a[0]), "y": float(a[1]), "z": float(a[2])}


def run(
    ctx: NodeContext,
    waypoints: list[Any],
    arms: list[int] | None = None,
    rotation: dict[str, float] | None = None,
    handover_cost: float = HANDOVER_COST,
    joint_weight: float = JOINT_WEIGHT,
    initial_arm: int | None = None,
) -> Output:
    arms = [int(a) for a in (arms or [0, 1])]
    if len(arms) < 2 or len(set(arms)) != len(arms):
        raise ValueError("arms must list at least two different arm ids")
    if not waypoints or len(waypoints) < 2:
        raise ValueError("waypoints must have at least two points (the object's path)")
    if initial_arm is not None and int(initial_arm) not in arms:
        raise ValueError(f"initial_arm {initial_arm} is not in arms {arms}")
    n, m = len(waypoints), len(arms)
    out: Output = {
        "route": "infeasible",
        "detail": "",
        "assignment": [],
        "handovers": [],
        "n_handovers": 0,
        "total_cost": math.inf,
        "unreachable_waypoints": [],
    }
    q: list[list[np.ndarray | None]] = [[None] * m for _ in range(n)]
    for i, wp in enumerate(waypoints):
        pose: dict[str, Any] = {"position": _xyz(wp)}
        rot = wp.get("rotation") if isinstance(wp, dict) and "rotation" in wp else rotation
        if rot is not None:
            pose["rotation"] = rot
        for k, a in enumerate(arms):
            kw = {"arm_id": a} if a else {}
            try:
                r = ctx.tool("robot.solve_ik", pose=pose, **kw)
            except Exception as exc:  # an IK refusal is unreachable
                logger.info("IK refused for arm %s at waypoint %d: %s", a, i, exc)
                continue
            if (
                isinstance(r, dict)
                and r.get("success")
                and r.get("within_servo_tolerance", True) is not False
            ):
                js = r.get("joint_positions") or r.get("positions") or []
                q[i][k] = np.asarray(js, dtype=float)
    out["unreachable_waypoints"] = [i for i in range(n) if all(q[i][k] is None for k in range(m))]
    # Viterbi: cost[i][k] = best cost of the path ending at waypoint i carried by arm k.
    inf = math.inf
    cost = np.full((n, m), inf)
    back = np.full((n, m), -1, dtype=int)
    for k in range(m):
        if q[0][k] is not None and (initial_arm is None or arms[k] == int(initial_arm)):
            cost[0, k] = 0.0
    for i in range(1, n):
        for k in range(m):
            if q[i][k] is None:
                continue
            for j in range(m):
                if not math.isfinite(cost[i - 1, j]):
                    continue
                if j == k:
                    prev = q[i - 1][k]
                    step = (
                        joint_weight * float(np.abs(q[i][k] - prev).sum())
                        if prev is not None and len(prev) == len(q[i][k])
                        else 0.0
                    )
                    c = cost[i - 1, j] + step
                else:
                    # Handover at waypoint i-1: both arms must reach it.
                    if q[i - 1][k] is None:
                        continue
                    c = cost[i - 1, j] + float(handover_cost)
                    prev = q[i - 1][k]
                    c += (
                        joint_weight * float(np.abs(q[i][k] - prev).sum())
                        if len(prev) == len(q[i][k])
                        else 0.0
                    )
                if c < cost[i, k]:
                    cost[i, k], back[i, k] = c, j
    k = int(np.argmin(cost[-1]))
    if not math.isfinite(cost[-1, k]):
        last = max((i for i in range(n) if np.isfinite(cost[i]).any()), default=-1)
        out["detail"] = (
            f"no arm sequence reaches every waypoint; waypoints no arm reaches: {out['unreachable_waypoints'][:10]}; "
            f"the path is carried at most to waypoint {last}"
        )
        return out
    seq = [k]
    for i in range(n - 1, 0, -1):
        k = int(back[i, k])
        seq.append(k)
    seq.reverse()
    out["assignment"] = [arms[s] for s in seq]
    out["total_cost"] = round(float(cost[-1, seq[-1]]), 4)
    for i in range(1, n):
        if seq[i] != seq[i - 1]:
            out["handovers"].append(
                {
                    "index": i - 1,
                    "from_arm": arms[seq[i - 1]],
                    "to_arm": arms[seq[i]],
                    "position": _xyz(waypoints[i - 1]),
                }
            )
    out["n_handovers"] = len(out["handovers"])
    if out["handovers"]:
        out["route"] = "handover_required"
        out["detail"] = (
            f"{out['n_handovers']} handover(s) at waypoints {[h['index'] for h in out['handovers']]}"
        )
    else:
        out["route"] = "assignment_found"
        out["detail"] = f"arm {out['assignment'][0]} carries the whole path"
    return out
