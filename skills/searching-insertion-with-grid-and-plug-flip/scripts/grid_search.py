# Reimplements ShapeForce's USB grid search (third_party/utils_repo/peg_insertion/force_insert.py,
# _phase_grid_search_forusb; MIT, Copyright (c) 2021 SowmiyaNarayanan G,
# https://github.com/nolann030628/ShapeForce): find the base by a slow descent until contact, probe a small x/y
# grid at the base height, a probe that sinks past the base reads as in, one that sinks part way as the plug
# upside down, and an exhausted grid turns the plug 180 deg and searches again. Changes: contact is the hand's
# tracking lag in 0.5 mm steps (YAM has no force sensing) instead of a soft-wrist Fz threshold; the grid is
# probed centre-out instead of a corner raster; depths and steps are inputs instead of hard-coded mm.
"""Find a hole under a held plug by probing a grid at the base height; turn the plug 180 deg if none takes it."""

from __future__ import annotations

import math
import statistics
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

STEP = 0.0005
"""Descent step [m]."""

STEP_TOLERANCE = 0.0002
"""Arrival ball of each step [m], under the step so a blocked hand reads as lag."""

CONTACT_LAG = 0.0006
"""Lag over the free baseline that reads as touching [m]."""

LIFT = 0.002
"""Probe start height over the base [m]."""


class Output(TypedDict):
    route: str
    detail: str
    base_z: float
    depth: float
    offset_x: float
    offset_y: float
    probes: int
    flipped: bool


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def turned_about_tool_z(q: dict[str, float], deg: float) -> dict[str, float]:
    """``q`` (wxyz dict) turned ``deg`` about its own z axis."""
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    h = math.radians(deg) / 2
    cw, sz = math.cos(h), math.sin(h)
    return {"w": w * cw - z * sz, "x": x * cw + y * sz, "y": y * cw - x * sz, "z": z * cw + w * sz}


def grid_cells(half: float, step: float) -> list[tuple[float, float]]:
    """Grid offsets within ``half`` at ``step``, nearest the centre first."""
    n = int(round(half / step))
    cells = [(i * step, j * step) for i in range(-n, n + 1) for j in range(-n, n + 1)]
    return sorted(cells, key=lambda c: (round(math.hypot(*c), 9), math.atan2(c[1], c[0])))


def run(
    ctx: NodeContext,
    grid_half_m: float = 0.0045,
    grid_step_m: float = 0.0015,
    in_depth_m: float = 0.004,
    partial_depth_m: float = 0.0,
    probe_depth_m: float = 0.008,
    max_base_descent_m: float = 0.03,
    allow_flip: bool = True,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    out: Output = {"route": "jammed", "detail": "", "base_z": 0.0, "depth": 0.0, "offset_x": 0.0, "offset_y": 0.0,
                   "probes": 0, "flipped": False}  # fmt: skip
    start = _pose(ctx.tool("robot.get_ee_pose", **on_arm))
    rot = start["rotation"]
    x0, y0, z0 = _xyz(start)

    def go(x: float, y: float, z: float, r: dict[str, float], tol: float = STEP_TOLERANCE) -> float:
        ctx.tool(
            "robot.go_to_pose",
            pose={"position": {"x": x, "y": y, "z": z}, "rotation": r},
            tolerance=tol,
            **on_arm,
        )
        return float(_xyz(_pose(ctx.tool("robot.get_ee_pose", **on_arm)))[2])

    def descend(
        x: float, y: float, z_from: float, travel: float, r: dict[str, float]
    ) -> tuple[bool, float]:
        """Step down from z_from; (touched, read-back z at the touch or at the bottom)."""
        lags: list[float] = []
        cmd = z_from
        z = z_from
        for _ in range(int(math.ceil(travel / STEP))):
            cmd -= STEP
            z = go(x, y, cmd, r)
            lag = z - cmd
            base = statistics.median(lags) if len(lags) >= 3 else 0.0
            if lag - base > CONTACT_LAG:
                return True, z
            if lag - base < CONTACT_LAG / 3:
                lags.append(lag)
        return False, z

    touched, zb = descend(x0, y0, z0, float(max_base_descent_m), rot)
    if not touched:
        out.update(
            route="jammed",
            detail=f"no base within {max_base_descent_m * 1e3:.0f} mm under the start",
        )
        return out
    out["base_z"] = zb
    z_up = zb + LIFT
    flips = [False, True] if allow_flip else [False]
    for flipped in flips:
        r = turned_about_tool_z(rot, 180.0) if flipped else rot
        out["flipped"] = flipped
        go(x0, y0, z_up + 0.003, r, tol=0.001)
        for dx, dy in grid_cells(float(grid_half_m), float(grid_step_m)):
            go(x0 + dx, y0 + dy, z_up, r, tol=0.0005)
            touched, z = descend(x0 + dx, y0 + dy, z_up, LIFT + float(probe_depth_m), r)
            out["probes"] += 1
            depth = zb - z
            if depth >= float(in_depth_m):
                out.update(route="flipped_and_inserted" if flipped else "inserted", depth=depth, offset_x=dx, offset_y=dy,
                           detail=f"probe {out['probes']} at ({dx * 1e3:+.1f}, {dy * 1e3:+.1f}) mm sank {depth * 1e3:.1f} mm past the base")  # fmt: skip
                return out
            if (
                partial_depth_m > 0
                and depth >= float(partial_depth_m)
                and not flipped
                and allow_flip
            ):
                out["detail"] = (
                    f"probe {out['probes']} sank part way ({depth * 1e3:.1f} mm): plug upside down"
                )
                break
            go(x0 + dx, y0 + dy, z_up, r, tol=0.0005)
    go(x0, y0, z_up + 0.005, rot, tol=0.001)
    out.update(
        route="grid_exhausted",
        detail=out["detail"] or f"{out['probes']} probes, none sank {in_depth_m * 1e3:.0f} mm",
    )
    return out
