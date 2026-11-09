"""Rigid transforms and the pose wire format.

Convention (00-common-contract): ``T_A_B`` maps coordinates in frame B into
frame A, so ``p_A = T_A_B @ p_B`` and ``T_A_C = T_A_B @ T_B_C``. Poses on the wire
are ``{"frame_id", "position_m", "quaternion_xyzw"}``, the kit runtime's pose
form; the frame named there is A and the posed thing is B.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def quat_xyzw_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_xyzw(r: np.ndarray) -> list[float]:
    """Shepperd's method; the sign is fixed so that w >= 0."""
    r = np.asarray(r, dtype=float)[:3, :3]
    t = float(np.trace(r))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = (
            0.25 * s,
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
        )
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (
            (r[2, 1] - r[1, 2]) / s,
            0.25 * s,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
        )
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (
            (r[0, 2] - r[2, 0]) / s,
            (r[0, 1] + r[1, 0]) / s,
            0.25 * s,
            (r[1, 2] + r[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (
            (r[1, 0] - r[0, 1]) / s,
            (r[0, 2] + r[2, 0]) / s,
            (r[1, 2] + r[2, 1]) / s,
            0.25 * s,
        )
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return [float(v) for v in q]


def make_T(
    rotation: np.ndarray | None = None, position: Sequence[float] | None = None
) -> np.ndarray:
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = np.asarray(rotation, dtype=float)[:3, :3]
    if position is not None:
        T[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def axis_angle(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * k + (1 - math.cos(angle_rad)) * (k @ k)


def rot_x(rad: float) -> np.ndarray:
    return axis_angle((1, 0, 0), rad)


def rot_y(rad: float) -> np.ndarray:
    return axis_angle((0, 1, 0), rad)


def rot_z(rad: float) -> np.ndarray:
    return axis_angle((0, 0, 1), rad)


def rotation_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle of ``Ra^T Rb`` [rad]."""
    c = (float(np.trace(np.asarray(Ra)[:3, :3].T @ np.asarray(Rb)[:3, :3])) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation vector of ``R`` [rad]."""
    th = rotation_angle(np.eye(3), R)
    if th < 1e-9:
        return np.zeros(3)
    if math.pi - th < 1e-6:
        # Near pi: axis from the symmetric part.
        M = (np.asarray(R)[:3, :3] + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(M)))
        a = M[:, k] / math.sqrt(max(M[k, k], 1e-12))
        return a / np.linalg.norm(a) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * math.sin(th)) * th


def interpolate_T(Ta: np.ndarray, Tb: np.ndarray, u: float) -> np.ndarray:
    """Straight-line position, geodesic rotation."""
    d = log_so3(Ta[:3, :3].T @ Tb[:3, :3])
    R = (
        Ta[:3, :3] @ axis_angle(d, float(np.linalg.norm(d)) * u)
        if np.linalg.norm(d) > 1e-12
        else Ta[:3, :3]
    )
    return make_T(R, (1 - u) * Ta[:3, 3] + u * Tb[:3, 3])


def pose_to_T(pose: dict[str, Any], expect_frame: str | None = None) -> np.ndarray:
    if not isinstance(pose, dict) or set(pose) != {"frame_id", "position_m", "quaternion_xyzw"}:
        raise ValueError("pose requires exactly frame_id, position_m, quaternion_xyzw")
    if expect_frame is not None and pose["frame_id"] != expect_frame:
        raise ValueError(f"pose frame is {pose['frame_id']!r}, expected {expect_frame!r}")
    return make_T(quat_xyzw_to_matrix(pose["quaternion_xyzw"]), pose["position_m"])


def T_to_pose(T: np.ndarray, frame_id: str) -> dict[str, Any]:
    return {
        "frame_id": str(frame_id),
        "position_m": [float(v) for v in np.asarray(T)[:3, 3]],
        "quaternion_xyzw": matrix_to_quat_xyzw(np.asarray(T)[:3, :3]),
    }


def is_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    R = np.asarray(R, dtype=float)
    return bool(np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1) < tol)
