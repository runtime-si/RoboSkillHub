"""Geometric predicates: yes/no answers a router can branch on."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from gap_core.tools import tool
from gap_core.types import Centerline, FunctionalFeature, OrientedBoundingBox, Quaternion, Vec3


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


class ThreadedResult(TypedDict):
    threaded: bool
    verdict: str
    crossings: int
    passes: int
    through_m: float
    clearance_m: float


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


@tool(
    name="predicates.threaded_through",
    summary="Whether a rope's centreline goes through a loop's opening, net of any pass back.",
    tags=("verification",),
)
def threaded_through(
    rope: Centerline, loop: FunctionalFeature, min_through_m: float = 0.01
) -> ThreadedResult:
    """Counts the centreline's passes through the disk ``loop.radius_inner`` bounds.

    A pass is a segment crossing the loop's plane inside the opening. It counts
    +1 along ``loop.axis`` and -1 against it, so a rope that went through and
    came back nets zero (``crossings``), while ``passes`` counts both.
    ``through_m`` is the shorter of the two ends beyond the first and last
    pass: how far the rope is from slipping back out. Threaded means a nonzero
    net with ``through_m >= min_through_m``.

    ``clearance_m`` is the room between the rope's surface (``rope.radius_m``)
    and the rim at the tightest pass. With no pass it is minus the distance by
    which the centreline misses the opening, measured at its nodes and at its
    crossings of the loop's plane.
    """
    points = np.asarray(rope["points"], dtype=float).reshape(-1, 3)
    if len(points) < 2:
        raise ValueError("the rope's centreline needs at least two points")
    radius = float(loop.get("radius_inner") or 0.0)
    if radius <= 0.0:
        raise ValueError("the loop needs a positive radius_inner")
    axis = _vec(loop["axis"])
    if not np.linalg.norm(axis) > 0.0:
        raise ValueError("the loop's axis has zero length")
    axis = axis / np.linalg.norm(axis)
    center = _vec(loop["pose"]["position"])

    height = (points - center) @ axis
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    along = np.concatenate([[0.0], np.cumsum(lengths)])
    cross = np.flatnonzero((height[:-1] >= 0.0) != (height[1:] >= 0.0))
    t = height[cross] / (height[cross] - height[cross + 1])
    rho = np.linalg.norm(points[cross] + t[:, None] * steps[cross] - center, axis=1)
    inside = rho < radius

    signs = np.where(height[cross + 1] > height[cross], 1, -1)[inside]
    crossings, passes = int(signs.sum()), int(inside.sum())
    at = (along[cross] + t * lengths[cross])[inside]
    through = float(min(at[0], along[-1] - at[-1])) if crossings else 0.0
    if passes:
        clearance = radius - float(rho[inside].max()) - float(rope.get("radius_m") or 0.0)
    else:
        radial = np.linalg.norm((points - center) - np.outer(height, axis), axis=1)
        to_disk = np.where(radial <= radius, np.abs(height), np.hypot(height, radial - radius))
        clearance = -float(min(to_disk.min(), (rho - radius).min(initial=np.inf)))
    threaded = crossings != 0 and through >= float(min_through_m)
    return {
        "threaded": threaded,
        "verdict": "threaded" if threaded else "not_threaded",
        "crossings": crossings,
        "passes": passes,
        "through_m": through,
        "clearance_m": clearance,
    }
