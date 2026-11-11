"""Put a held part's tip over a hole by visual servoing on the wrist RGB-D frame: fit the held part's axis, find
the hole's centre on the face, move the hand by the lateral offset (capped), check it moved, repeat."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

TOL = 0.001
"""Lateral tip-to-hole offset that counts as aligned [m]."""

MAX_STEP = 0.006
"""Largest correction per iteration [m]; a reading further off than MAX_OFFSET is a misreading."""

MAX_OFFSET = 0.03

MIN_FOLLOW = 0.4
"""Fraction of a commanded correction the hand must be seen to make (read back); under it the hand is not
following, which is what kept a real pens run's foot 6-8 mm off over three corrections."""

HELD_MARGIN = 0.006
"""Points within the part's radius plus this of the tool axis belong to the held part [m]."""

FACE_BAND = 0.003
"""Points within this of the face height are the face [m]; deeper than HOLE_DEPTH under it, a hole."""

HOLE_DEPTH = 0.005
TILT_CONE = math.tan(math.radians(15.0))
"""The first selection widens along the tool axis by this per metre, so a part tilted up to ~15 deg in the grasp
keeps its lower end (a twin peg held 10 deg off lost it to a plain cylinder and read 8-15 mm off)."""
MIN_POINTS = 60


class Output(TypedDict):
    route: str
    detail: str
    offset_x: float
    offset_y: float
    iterations: int
    moved_m: float


def _rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def cloud(depth: np.ndarray, k: np.ndarray, t_base_cam: np.ndarray, stride: int = 2):
    """(points in the base frame [N, 3], their pixel rows, cols) for every valid depth pixel."""
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[v, u].astype(float)
    ok = np.isfinite(z) & (z > 0.02) & (z < 1.5)
    v, u, z = v[ok], u[ok], z[ok]
    p = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=1)
    return p @ t_base_cam[:3, :3].T + t_base_cam[:3, 3], v, u


def held_foot(
    pts: np.ndarray,
    t_base_tool: np.ndarray,
    radius: float,
    tip_offset: float,
    face_z: float,
    cam_pos: np.ndarray,
) -> dict[str, Any]:
    """The held part's axis (a line fitted to its visible surface, moved back by the radius) and where it crosses
    the face plane: the foot. Points near the tool's approach axis, from the tool centre to the tip, above the face."""
    o, a = t_base_tool[:3, 3], t_base_tool[:3, 2]
    above = pts[:, 2] > face_z + 0.004
    # A cone around the tool axis (a part tilted in the grasp leans out of a plain cylinder), then the points
    # around the axis fitted to them, twice: the part's lower end is then kept however it leans.
    rel = pts - o
    s = rel @ a
    d = np.linalg.norm(rel - np.outer(s, a), axis=1)
    sel = (
        (d < radius + HELD_MARGIN + np.maximum(s, 0.0) * TILT_CONE)
        & (s > 0.0)
        & (s < tip_offset + 0.005)
        & above
    )
    if sel.sum() < MIN_POINTS:
        return {"found": False, "n": int(sel.sum())}
    for _ in range(2):
        q = pts[sel]
        c = q.mean(axis=0)
        axis = np.linalg.svd(q - c, full_matrices=False)[2][0]
        if axis @ a < 0:
            axis = -axis
        rel = pts - c
        s2 = rel @ axis
        d2 = np.linalg.norm(rel - np.outer(s2, axis), axis=1)
        s_tool = (pts - o) @ axis
        nxt = (d2 < radius + 0.004) & (s_tool > 0.0) & (s_tool < tip_offset + 0.01) & above
        if nxt.sum() < MIN_POINTS:
            break
        sel = nxt
    q = pts[sel]
    c = q.mean(axis=0)
    axis = np.linalg.svd(q - c, full_matrices=False)[2][0]
    if axis @ a < 0:
        axis = -axis
    # The camera sees the near side: fit the known-radius circle to the points across the axis, started behind
    # the visible surface (the surface's centroid alone sits inside the radius by an amount set by the arc seen).
    away = c - cam_pos
    away -= axis * (away @ axis)
    n = np.linalg.norm(away)
    e1 = away / n if n > 1e-9 else np.cross(axis, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    p2 = np.stack([(q - c) @ e1, (q - c) @ e2], axis=1)
    m = np.array([0.5 * radius, 0.0])
    for _ in range(30):
        d = p2 - m
        r = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        jac = -d / r[:, None]
        step = np.linalg.lstsq(jac, -(r - radius), rcond=None)[0]
        m = m + step
        if np.linalg.norm(step) < 1e-6:
            break
    c = c + m[0] * e1 + m[1] * e2
    if abs(axis[2]) < 0.3:
        return {"found": False, "n": int(sel.sum()), "why": "axis not vertical enough"}
    t = (face_z - c[2]) / axis[2]
    foot = c + axis * t
    tilt = math.degrees(math.acos(min(1.0, abs(axis[2]))))
    return {"found": True, "n": int(sel.sum()), "foot_xy": foot[:2], "axis": axis, "tilt_deg": tilt}


def hole_centre(
    pts: np.ndarray,
    expected_xy: np.ndarray,
    radius: float,
    face_z: float,
    search: float = 0.02,
    cell: float = 0.001,
) -> dict[str, Any]:
    """Centre of the hole near ``expected_xy``: the face points around it, rasterised at ``cell``, and the gaps the
    face closes around (a hole), not merely the deeper points - past the holder's edge the table is deeper too, and
    in a station twin it pulled the centre 11-17 mm off. The gap nearest the expected spot, its area at least half
    the hole's; its centre is the known-radius circle that keeps the face points out."""
    from scipy import ndimage

    xy = pts[:, :2] - expected_xy
    half = radius + search
    near = np.all(np.abs(xy) < half, axis=1)
    face = near & (np.abs(pts[:, 2] - face_z) < FACE_BAND)
    if face.sum() < MIN_POINTS:
        return {"found": False, "face": int(face.sum())}
    n = int(np.ceil(2 * half / cell))
    ij = np.clip(((xy[face] + half) / cell).astype(int), 0, n - 1)
    grid = np.zeros((n, n), bool)
    grid[ij[:, 0], ij[:, 1]] = True
    grid = ndimage.binary_closing(grid, iterations=1)
    gaps = ndimage.binary_fill_holes(grid) & ~grid
    lab, k = ndimage.label(gaps)
    best, best_d = None, np.inf
    for comp in range(1, k + 1):
        cells = np.argwhere(lab == comp)
        if len(cells) * cell * cell < 0.5 * math.pi * radius * radius:
            continue
        c = (cells.mean(axis=0) + 0.5) * cell - half
        if np.linalg.norm(c) < best_d:
            best, best_d = c, float(np.linalg.norm(c))
    if best is None:
        return {"found": False, "face": int(face.sum()), "gaps": int(k)}
    c = best
    f = xy[face]
    for _ in range(20):
        r = np.linalg.norm(f - c, axis=1)
        inside = r < radius
        if not inside.any():
            break
        push = ((f[inside] - c) / np.maximum(r[inside], 1e-6)[:, None]) * (radius - r[inside])[
            :, None
        ]
        c = c - push.mean(axis=0)
    return {"found": True, "centre_xy": c + expected_xy, "face": int(face.sum()), "gaps": int(k)}


def surface_around(pts: np.ndarray, xy: np.ndarray, hole_radius: float) -> float:
    """Median height of the points in a ring 3-10 mm outside a hole's radius around ``xy`` (under the part, where
    its own body does not hide the surface): the face if the tip is over the holder, the table if not."""
    d = np.linalg.norm(pts[:, :2] - xy, axis=1)
    ring = (d > hole_radius + 0.003) & (d < hole_radius + 0.010)
    return float(np.median(pts[ring, 2])) if ring.sum() >= MIN_POINTS // 3 else float("nan")


def measure(
    frame: dict[str, Any],
    t_base_tool: np.ndarray,
    hole_xy: np.ndarray,
    face_z: float,
    part_radius: float,
    hole_radius: float,
    tip_offset: float,
) -> dict[str, Any]:
    k = np.asarray(frame["intrinsics"], dtype=float)
    tbc = np.asarray(frame["T_base_camera"], dtype=float)
    pts, _, _ = cloud(np.asarray(frame["depth"], dtype=float), k, tbc)
    foot = held_foot(pts, t_base_tool, part_radius, tip_offset, face_z, tbc[:3, 3])
    hole = hole_centre(pts, np.asarray(hole_xy, dtype=float), hole_radius, face_z)
    out = {"foot": foot, "hole": hole}
    if foot["found"]:
        out["under"] = surface_around(pts, np.asarray(foot["foot_xy"]), hole_radius)
    if foot["found"] and hole["found"]:
        out["offset"] = np.asarray(hole["centre_xy"]) - np.asarray(foot["foot_xy"])
    return out


def _frame(
    ctx: NodeContext, camera: str | None, arm: dict[str, Any], frame_tool: str | None = None
) -> dict[str, Any] | None:
    if frame_tool:
        f = ctx.tool(frame_tool, **arm)
        if not f or f.get("depth") is None:
            return None
        return {"depth": np.asarray(f["depth"], float), "intrinsics": np.asarray(f["intrinsics"], float),
                "T_base_camera": np.asarray(f["T_base_camera"], float)}  # fmt: skip
    obs = ctx.tool("robot.get_observation")
    cams = (
        [c for c in (obs.get("cameras") or []) if isinstance(c, dict)]
        if isinstance(obs, dict)
        else []
    )
    if camera:
        cams = [c for c in cams if c.get("name") == camera]
    else:
        cams = [
            c for c in cams if any(s in str(c.get("name", "")) for s in ("wrist", "eye_in_hand"))
        ]
    for c in cams:
        if c.get("depth") is None or c.get("intrinsics") is None or c.get("pose") is None:
            continue
        p = c["pose"]
        t = np.eye(4)
        t[:3, :3] = _rot(p["rotation"])
        t[:3, 3] = [float(p["position"][a]) for a in "xyz"]
        return {"depth": c["depth"], "intrinsics": c["intrinsics"], "T_base_camera": t}
    return None


def _tool(ctx: NodeContext, arm: dict[str, Any]) -> np.ndarray:
    r = ctx.tool("robot.get_ee_pose", **arm)
    p = r.get("pose", r)
    t = np.eye(4)
    t[:3, :3] = _rot(p["rotation"])
    t[:3, 3] = [float(p["position"][a]) for a in "xyz"]
    return t


def run(
    ctx: NodeContext,
    hole_xy: list[float],
    face_z: float,
    tip_offset_m: float,
    part_diameter_m: float,
    hole_diameter_m: float,
    camera: str | None = None,
    frame_tool: str | None = None,
    table_z: float = float("nan"),
    tol_m: float = TOL,
    max_step_m: float = MAX_STEP,
    max_iterations: int = 6,
    arm_id: int = 0,
) -> Output:
    arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    out: Output = {
        "route": "not_converging",
        "detail": "",
        "offset_x": 0.0,
        "offset_y": 0.0,
        "iterations": 0,
        "moved_m": 0.0,
    }
    for i in range(int(max_iterations)):
        out["iterations"] = i + 1
        frame = _frame(ctx, camera, arm, frame_tool)
        if frame is None:
            out.update(
                route="tip_not_seen", detail="no wrist camera with depth, intrinsics and pose"
            )
            return out
        t_tool = _tool(ctx, arm)
        m = measure(frame, t_tool, np.asarray(hole_xy[:2], float), float(face_z), part_diameter_m / 2,
                    hole_diameter_m / 2, float(tip_offset_m))  # fmt: skip
        if not m["foot"]["found"]:
            out.update(route="tip_not_seen", detail=f"held part not found ({m['foot']})")
            return out
        under = m.get("under", float("nan"))
        if (
            math.isfinite(float(table_z))
            and math.isfinite(under)
            and abs(under - table_z) < abs(under - face_z)
        ):
            out.update(route="not_over_holder",
                       detail=f"the surface around the tip is at {under:.4f} m: the table ({table_z:.4f}), not the face ({face_z:.4f})")  # fmt: skip
            return out
        if not m["hole"]["found"]:
            out.update(
                route="hole_not_seen", detail=f"no hole near the expected spot ({m['hole']})"
            )
            return out
        off = m["offset"]
        out["offset_x"], out["offset_y"] = float(off[0]), float(off[1])
        dist = float(np.linalg.norm(off))
        if dist < float(tol_m):
            out.update(
                route="aligned",
                detail=f"tip {dist * 1e3:.1f} mm from the hole centre after {i} move(s)",
            )
            return out
        if dist > MAX_OFFSET:
            out.update(
                route="not_converging",
                detail=f"read {dist * 1e3:.0f} mm off: a misreading, not moved",
            )
            return out
        step = off * min(1.0, float(max_step_m) / dist)
        goal = t_tool[:3, 3] + np.array([step[0], step[1], 0.0])
        p = ctx.tool("robot.get_ee_pose", **arm)
        p = p.get("pose", p)
        ctx.tool("robot.go_to_pose", pose={"position": dict(zip("xyz", map(float, goal), strict=False)), "rotation": p["rotation"]},
                 tolerance=0.0003, **arm)  # fmt: skip
        moved = _tool(ctx, arm)[:3, 3] - t_tool[:3, 3]
        made = float(np.dot(moved[:2], step) / max(np.dot(step, step), 1e-12))
        out["moved_m"] += float(np.linalg.norm(moved[:2]))
        if made < MIN_FOLLOW:
            out.update(route="not_converging",
                       detail=f"asked {np.linalg.norm(step) * 1e3:.1f} mm, the hand made {made * 100:.0f} % of it")  # fmt: skip
            return out
    out["detail"] = (
        out["detail"] or f"still {math.hypot(out['offset_x'], out['offset_y']) * 1e3:.1f} mm off"
    )
    return out
