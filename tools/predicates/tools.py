"""Geometric predicates: yes/no answers a router can branch on."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from gap_core.tools import tool
from gap_core.types import OrientedBoundingBox, Quaternion, Vec3


class PointInObbResult(TypedDict):
    inside: bool
    verdict: str
    depth_m: float


class RestingOnResult(TypedDict):
    on: bool
    verdict: str
    xy_overlap: float
    gap_m: float


class DistanceResult(TypedDict):
    near: bool
    verdict: str
    distance_m: float


class UprightResult(TypedDict):
    upright: bool
    verdict: str
    tilt_deg: float


def _vec(v: Vec3) -> np.ndarray:
    return np.array([float(v["x"]), float(v["y"]), float(v["z"])])


def _rotation(q: Quaternion) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("orientation quaternion has zero norm")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _world_half_extents(obb: OrientedBoundingBox) -> np.ndarray:
    """Half-extents of the world-axis-aligned box that encloses *obb*."""
    return np.abs(_rotation(obb["orientation"])) @ _vec(obb["extent"])


@tool(
    name="predicates.point_in_obb",
    summary="Whether a world point lies inside an oriented box (half-extents), and by how much.",
    tags=("verification",),
)
def point_in_obb(point: Vec3, obb: OrientedBoundingBox, margin_m: float = 0.0) -> PointInObbResult:
    """``margin_m`` grows the box on every side (negative shrinks it).

    ``depth_m`` is the distance to the nearest face when inside, and minus the
    Euclidean distance to the box when outside.
    """
    local = _rotation(obb["orientation"]).T @ (_vec(point) - _vec(obb["center"]))
    slack = _vec(obb["extent"]) + float(margin_m) - np.abs(local)
    inside = bool(np.all(slack >= 0.0))
    depth = float(slack.min()) if inside else -float(np.linalg.norm(np.clip(-slack, 0.0, None)))
    return {"inside": inside, "verdict": "inside" if inside else "outside", "depth_m": depth}


@tool(
    name="predicates.resting_on",
    summary="Whether box `top` sits on box `base`: footprint overlap and the vertical gap between them.",
    tags=("verification",),
)
def resting_on(
    top: OrientedBoundingBox,
    base: OrientedBoundingBox,
    xy_overlap_min: float = 0.5,
    gap_max_m: float = 0.02,
) -> RestingOnResult:
    """``xy_overlap`` is the fraction of ``top``'s footprint over ``base``'s.

    ``gap_m`` is ``top``'s underside minus ``base``'s top face: positive is
    hovering, negative is interpenetrating. Both directions count against
    ``gap_max_m``, since a perceived box sinks into its support by its noise.
    """
    tc, th = _vec(top["center"]), _world_half_extents(top)
    bc, bh = _vec(base["center"]), _world_half_extents(base)
    lo = np.maximum(tc[:2] - th[:2], bc[:2] - bh[:2])
    hi = np.minimum(tc[:2] + th[:2], bc[:2] + bh[:2])
    shared = float(np.prod(np.clip(hi - lo, 0.0, None)))
    footprint = float(4.0 * th[0] * th[1])
    overlap = shared / footprint if footprint > 0.0 else 0.0
    gap = float((tc[2] - th[2]) - (bc[2] + bh[2]))
    on = overlap >= float(xy_overlap_min) and abs(gap) <= float(gap_max_m)
    return {"on": on, "verdict": "on" if on else "off", "xy_overlap": overlap, "gap_m": gap}


@tool(
    name="predicates.within_distance",
    summary="Whether two points are within `threshold_m`, in 3D (axes='xyz') or in the plane (axes='xy').",
    tags=("verification",),
)
def within_distance(a: Vec3, b: Vec3, threshold_m: float, axes: str = "xyz") -> DistanceResult:
    if axes not in ("xyz", "xy"):
        raise ValueError("axes is 'xyz' or 'xy'")
    delta = _vec(a) - _vec(b)
    if axes == "xy":
        delta = delta[:2]
    distance = float(np.linalg.norm(delta))
    near = distance <= float(threshold_m)
    return {"near": near, "verdict": "near" if near else "far", "distance_m": distance}


@tool(
    name="predicates.upright",
    summary="Whether a body's local axis ('x', 'y' or 'z') points up, within `max_tilt_deg` of world +Z.",
    tags=("verification",),
)
def upright(orientation: Quaternion, max_tilt_deg: float = 15.0, axis: str = "z") -> UprightResult:
    column = {"x": 0, "y": 1, "z": 2}.get(axis)
    if column is None:
        raise ValueError("axis is 'x', 'y' or 'z'")
    up = _rotation(orientation)[:, column]
    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
    ok = tilt <= float(max_tilt_deg)
    return {"upright": ok, "verdict": "upright" if ok else "tipped", "tilt_deg": tilt}
