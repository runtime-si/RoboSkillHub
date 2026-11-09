"""Pose algebra and the domino's geometric conventions.

Poses are ``{"frame_id", "position_m", "quaternion_xyzw"}`` (the workcell runtime's
format). ``T_A_B`` maps coordinates in frame B to frame A, so
``T_station_object = T_station_tcp * T_tcp_object``.

The domino's object frame (fixed, :data:`OBJECT_FRAME`): origin at the centre of the
support face (the bottom end of a standing tile), +z the up axis along its height,
+x the facing (the broad faces' normal, the way it topples and the row runs; the
placing skill's convention), +y across its width.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

OBJECT_FRAME = {
    "origin": "bottom_face_center",
    "up_axis": "+z",
    "facing_axis": "+x",
    "width_axis": "+y",
}


def quat_matrix(q_xyzw: Any) -> np.ndarray:
    x, y, z, w = (float(v) for v in q_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_quat(r: np.ndarray) -> list[float]:
    """Rotation matrix to a unit quaternion xyzw (w >= 0)."""
    t = float(np.trace(r))
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        w, x, y, z = (
            0.25 * s,
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
        )
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w, x, y, z = (
            (r[2, 1] - r[1, 2]) / s,
            0.25 * s,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
        )
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w, x, y, z = (
            (r[0, 2] - r[2, 0]) / s,
            (r[0, 1] + r[1, 0]) / s,
            0.25 * s,
            (r[1, 2] + r[2, 1]) / s,
        )
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w, x, y, z = (
            (r[1, 0] - r[0, 1]) / s,
            (r[0, 2] + r[2, 0]) / s,
            (r[1, 2] + r[2, 1]) / s,
            0.25 * s,
        )
    q = np.array([x, y, z, w], dtype=float)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return [float(v) for v in q]


def yaw_quat(yaw_rad: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)]


def pose(frame_id: str, position_m: Any, quaternion_xyzw: Any) -> dict:
    return {
        "frame_id": frame_id,
        "position_m": [float(v) for v in position_m],
        "quaternion_xyzw": [float(v) for v in quaternion_xyzw],
    }


def to_matrix(p: dict) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = quat_matrix(p["quaternion_xyzw"])
    t[:3, 3] = np.asarray(p["position_m"], dtype=float)
    return t


def from_matrix(t: np.ndarray, frame_id: str) -> dict:
    return pose(frame_id, t[:3, 3], matrix_quat(t[:3, :3]))


def compose(t_a_b: dict, t_b_c: dict, frame_id: str | None = None) -> dict:
    """``T_A_C = T_A_B * T_B_C``; the result is expressed in ``t_a_b``'s frame."""
    return from_matrix(to_matrix(t_a_b) @ to_matrix(t_b_c), frame_id or t_a_b["frame_id"])


def inverse(t_a_b: dict, frame_id: str) -> dict:
    """``T_B_A``; ``frame_id`` names B (the pose dict only carries its parent frame)."""
    return from_matrix(np.linalg.inv(to_matrix(t_a_b)), frame_id)


def tcp_for_object(t_station_object: dict, t_tcp_object: dict) -> dict:
    """The TCP pose that puts the held object at ``t_station_object``:
    ``T_station_tcp = T_station_object * inv(T_tcp_object)``."""
    return compose(t_station_object, inverse(t_tcp_object, "object"), t_station_object["frame_id"])


def object_from_tcp(t_station_tcp: dict, t_tcp_object: dict) -> dict:
    return compose(t_station_tcp, t_tcp_object, t_station_tcp["frame_id"])


def translated(p: dict, delta: Any) -> dict:
    return {**p, "position_m": [a + float(b) for a, b in zip(p["position_m"], delta, strict=True)]}


def unit(v: Any) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        raise ValueError("zero vector")
    return a / n


def tilt_rad(p: dict, plane_normal: Any, *, up_symmetric: bool = True) -> float:
    """Angle between the object's up axis (+z) and the support normal. A plain tile
    stood on its other end is upright too, when ``up_symmetric``."""
    up = quat_matrix(p["quaternion_xyzw"])[:, 2]
    c = float(np.clip(up @ unit(plane_normal), -1.0, 1.0))
    if up_symmetric:
        c = abs(c)
    return math.acos(c)


def facing_yaw_rad(p: dict, plane_normal: Any) -> float:
    """Heading of the object's +x projected on the support plane (about its normal,
    measured from the station x axis projected on the plane)."""
    n = unit(plane_normal)
    x = quat_matrix(p["quaternion_xyzw"])[:, 0]
    x = x - (x @ n) * n
    ref = np.array([1.0, 0.0, 0.0])
    ref = ref - (ref @ n) * n
    if np.linalg.norm(ref) < 1e-9:
        ref = np.array([0.0, 1.0, 0.0]) - n[1] * n
    ref = ref / np.linalg.norm(ref)
    return math.atan2(float(np.cross(ref, x) @ n), float(ref @ x))


def wrap(angle: float, period: float = 2.0 * math.pi) -> float:
    """Fold into [-period/2, period/2)."""
    return (angle + period / 2.0) % period - period / 2.0


def yaw_error_rad(a: dict, b: dict, plane_normal: Any, symmetry_deg: float) -> float:
    """Yaw difference about the support normal, folded by the tile's declared yaw
    symmetry (180 for a plain tile: its two broad faces are interchangeable; 360 when
    the faces differ, e.g. pips that must face the camera)."""
    d = facing_yaw_rad(a, plane_normal) - facing_yaw_rad(b, plane_normal)
    return abs(wrap(d, math.radians(symmetry_deg)))


def height_over_plane(point: Any, plane_point: Any, plane_normal: Any) -> float:
    return float(
        (np.asarray(point, dtype=float) - np.asarray(plane_point, dtype=float)) @ unit(plane_normal)
    )


def lowest_point_height(p: dict, size: dict, plane_point: Any, plane_normal: Any) -> float:
    """Signed height of the object's lowest box corner over the support plane."""
    t = to_matrix(p)
    hx, hy = size["thickness_m"] / 2.0, size["width_m"] / 2.0
    corners = [
        t[:3, :3] @ np.array([sx * hx, sy * hy, z]) + t[:3, 3]
        for sx in (-1, 1)
        for sy in (-1, 1)
        for z in (0.0, size["height_m"])
    ]
    return min(height_over_plane(c, plane_point, plane_normal) for c in corners)


def horizontal_offset(a: dict, b: dict, plane_normal: Any) -> float:
    n = unit(plane_normal)
    d = np.asarray(a["position_m"], dtype=float) - np.asarray(b["position_m"], dtype=float)
    return float(np.linalg.norm(d - (d @ n) * n))


def facing_deg_of(p: dict) -> float:
    """Station-frame heading of the object's +x (for the placing/layout skills' (xy, facing))."""
    r = quat_matrix(p["quaternion_xyzw"])
    return math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))


def xy_facing(p: dict) -> tuple[np.ndarray, float]:
    """The (xy, facing_deg) the hub's placing and layout skills take."""
    return np.asarray(p["position_m"][:2], dtype=float), facing_deg_of(p)
