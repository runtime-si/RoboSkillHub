# SPDX-License-Identifier: Apache-2.0
"""Pure-Python pose and footprint geometry for the domino evaluation (no numpy).

Conventions: metres, quaternions xyzw, ``T_A_B`` maps B coordinates into A. A tile's
origin is its box centre; its local x is the facing (normal of the two broad faces,
the thickness axis), local y the width axis and local z the up axis (the height).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # x, y, z, w


def quat_normalize(q: Sequence[float]) -> Quat:
    if len(q) != 4:
        raise ValueError(f"quaternion needs 4 components (xyzw), got {len(q)}")
    n = math.sqrt(sum(c * c for c in q))
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError("quaternion has zero or non-finite norm")
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_from_yaw(yaw: float) -> Quat:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def quat_from_axis_angle(axis: Sequence[float], angle: float) -> Quat:
    n = math.sqrt(sum(a * a for a in axis))
    s = math.sin(angle / 2) / n
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2))


def quat_mul(a: Sequence[float], b: Sequence[float]) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_rotate(q: Sequence[float], v: Sequence[float]) -> Vec3:
    x, y, z, w = q
    # v' = v + 2w (u x v) + 2 u x (u x v)
    cx = y * v[2] - z * v[1]
    cy = z * v[0] - x * v[2]
    cz = x * v[1] - y * v[0]
    ccx = y * cz - z * cy
    ccy = z * cx - x * cz
    ccz = x * cy - y * cx
    return (v[0] + 2 * (w * cx + ccx), v[1] + 2 * (w * cy + ccy), v[2] + 2 * (w * cz + ccz))


def wxyz_to_xyzw(q: Sequence[float]) -> Quat:
    return (q[1], q[2], q[3], q[0])


def tilt_deg(q: Sequence[float], normal: Sequence[float] = (0.0, 0.0, 1.0)) -> float:
    """Angle between the tile's up axis (either sign: a box flipped end over end
    still stands) and the support normal."""
    up = quat_rotate(quat_normalize(q), (0.0, 0.0, 1.0))
    n = math.sqrt(sum(c * c for c in normal))
    d = abs(sum(u * c for u, c in zip(up, normal, strict=True))) / n
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def facing_yaw(q: Sequence[float]) -> float:
    """Heading of the tile's facing axis projected on the table plane (rad)."""
    f = quat_rotate(quat_normalize(q), (1.0, 0.0, 0.0))
    return math.atan2(f[1], f[0])


def yaw_error_deg(q: Sequence[float], target_yaw: float, symmetry: float = math.pi) -> float:
    """Heading error modulo the tile's symmetry (a box looks the same turned 180 deg
    about its up axis)."""
    d = (facing_yaw(q) - target_yaw) % symmetry
    return math.degrees(min(d, symmetry - d))


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))


def dist_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def rect_corners(cx: float, cy: float, yaw: float, half_x: float, half_y: float) -> list[tuple]:
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        lx, ly = sx * half_x, sy * half_y
        out.append((cx + c * lx - s * ly, cy + s * lx + c * ly))
    return out


def _axes(poly: Sequence[tuple]) -> Iterable[tuple]:
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        ex, ey = x2 - x1, y2 - y1
        n = math.hypot(ex, ey)
        if n > 1e-12:
            yield (-ey / n, ex / n)


def polygon_gap(a: Sequence[tuple], b: Sequence[tuple]) -> float:
    """Separating-axis gap between two convex polygons: positive is the clearance
    along the best separating axis (a lower bound on the true distance), negative
    the overlap depth."""
    best = -math.inf
    for ax in list(_axes(a)) + list(_axes(b)):
        pa = [p[0] * ax[0] + p[1] * ax[1] for p in a]
        pb = [p[0] * ax[0] + p[1] * ax[1] for p in b]
        gap = max(min(pb) - max(pa), min(pa) - max(pb))
        best = max(best, gap)
    return best


def segment_rect_sweep(
    start: Sequence[float], end: Sequence[float], yaw: float, half_x: float, half_y: float
) -> list[tuple]:
    """Convex hull (as a polygon) of a rectangle swept along a straight segment
    without turning."""
    pts = rect_corners(start[0], start[1], yaw, half_x, half_y) + rect_corners(
        end[0], end[1], yaw, half_x, half_y
    )
    return convex_hull(pts)


def convex_hull(points: Sequence[tuple]) -> list[tuple]:
    pts = sorted(set((round(p[0], 12), round(p[1], 12)) for p in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]
