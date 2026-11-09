"""Robot-world / hand-eye calibration, A_i X = Z B_i, and the arm-to-arm base transform from it.

Written for this skill from the closed form of Shah, "Solving the robot-world/hand-eye
calibration problem using the Kronecker product" (J. Mechanisms Robotics 5(3), 2013), the
method OpenCV's calibrateRobotWorldHandEye(..., CALIB_ROBOT_WORLD_HAND_EYE_SHAH) implements
(OpenCV 5's Python build no longer exposes it), followed by a nonlinear least-squares
refinement over both transforms. numpy / scipy only.

Per view i, with ``T_base_tool`` from the arm and ``T_cam_target`` from the camera:

- eye in hand (the arm's wrist camera sees a target fixed in the cell):
  ``T_base_tool_i @ T_tool_cam @ T_cam_target_i = T_base_target``,
  so A = T_base_tool, X = T_tool_cam, Z = T_base_target, B = inv(T_cam_target).
- eye to hand (a fixed camera sees a target the gripper carries):
  ``T_base_tool_i @ T_tool_target = T_base_cam @ T_cam_target_i``,
  so A = T_base_tool, X = T_tool_target, Z = T_base_cam, B = T_cam_target.

Z is then the shared thing (the target, or the camera) in this arm's base frame, and two
arms that saw the same one give ``T_base0_base1 = Z0 @ inv(Z1)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

#: Length that weighs rotation against translation in the refinement [m per rad].
ROTATION_SCALE_M = 0.1


def _project_rotation(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def _inv(t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out


def solve_axzb_closed_form(
    a: list[np.ndarray], b: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Shah's Kronecker solution of A_i X = Z B_i; returns (X, Z) as 4x4."""
    rows = []
    eye = np.eye(3)
    for ai, bi in zip(a, b, strict=True):
        # vec (column-major): vec(Ra Rx) = (I kron Ra) vec(Rx), vec(Rz Rb) = (Rb^T kron I) vec(Rz)
        rows.append(np.hstack([np.kron(eye, ai[:3, :3]), -np.kron(bi[:3, :3].T, eye)]))
    _, _, vt = np.linalg.svd(np.vstack(rows))
    v = vt[-1]
    rx = v[:9].reshape(3, 3, order="F")
    rz = v[9:].reshape(3, 3, order="F")
    det = np.linalg.det(rx)
    scale = np.sign(det) * abs(det) ** (-1.0 / 3.0)
    rx, rz = _project_rotation(rx * scale), _project_rotation(rz * scale)
    # translations: Ra tx + ta = Rz tb + tz  ->  [Ra, -I] [tx; tz] = Rz tb - ta
    lhs = np.vstack([np.hstack([ai[:3, :3], -eye]) for ai in a])
    rhs = np.concatenate([rz @ bi[:3, 3] - ai[:3, 3] for ai, bi in zip(a, b, strict=True)])
    t = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    x, z = np.eye(4), np.eye(4)
    x[:3, :3], x[:3, 3] = rx, t[:3]
    z[:3, :3], z[:3, 3] = rz, t[3:]
    return x, z


def _pack(t: np.ndarray) -> np.ndarray:
    return np.r_[Rotation.from_matrix(t[:3, :3]).as_rotvec(), t[:3, 3]]


def _unpack(p: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
    t[:3, 3] = p[3:]
    return t


def _residuals(p: np.ndarray, a: list, b: list) -> np.ndarray:
    x, z = _unpack(p[:6]), _unpack(p[6:])
    out = []
    for ai, bi in zip(a, b, strict=True):
        lhs, rhs = ai @ x, z @ bi
        out.append(lhs[:3, 3] - rhs[:3, 3])
        out.append(ROTATION_SCALE_M * Rotation.from_matrix(rhs[:3, :3].T @ lhs[:3, :3]).as_rotvec())
    return np.concatenate(out)


def solve_axzb(a: list[np.ndarray], b: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Closed form, then least squares on translation and (scaled) rotation residuals."""
    x0, z0 = solve_axzb_closed_form(a, b)
    fit = least_squares(_residuals, np.r_[_pack(x0), _pack(z0)], args=(a, b), method="lm")
    return _unpack(fit.x[:6]), _unpack(fit.x[6:])


def per_view_errors(
    a: list, b: list, x: np.ndarray, z: np.ndarray
) -> tuple[list[float], list[float]]:
    """Each view's disagreement with the fit, as where it alone puts Z: (mm, deg)."""
    mm, deg = [], []
    for ai, bi in zip(a, b, strict=True):
        zi = ai @ x @ _inv(bi)
        mm.append(float(np.linalg.norm(zi[:3, 3] - z[:3, 3]) * 1e3))
        deg.append(float(np.degrees(Rotation.from_matrix(z[:3, :3].T @ zi[:3, :3]).magnitude())))
    return mm, deg


def arm_equations(views: list[dict[str, Any]], mode: str) -> tuple[list, list]:
    """(A, B) for one arm's views ``[{T_base_tool, T_cam_target}]`` (4x4 each)."""
    a, b = [], []
    for v in views:
        tool = np.asarray(v["T_base_tool"], dtype=np.float64).reshape(4, 4)
        cam_target = np.asarray(v["T_cam_target"], dtype=np.float64).reshape(4, 4)
        a.append(tool)
        b.append(_inv(cam_target) if mode == "eye_in_hand" else cam_target)
    return a, b


def level(t: np.ndarray) -> np.ndarray:
    """``t`` with its rotation reduced to the turn about z (two bases on one flat table)."""
    out = np.eye(4)
    yaw = float(np.arctan2(t[1, 0], t[0, 0]))
    out[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    out[:3, 3] = t[:3, 3]
    return out


def base_to_base(z0: np.ndarray, z1: np.ndarray) -> np.ndarray:
    """Arm 1's base in arm 0's base frame from the shared target (or camera) in each."""
    return z0 @ _inv(z1)


def describe(t: np.ndarray) -> dict[str, Any]:
    rpy = Rotation.from_matrix(t[:3, :3]).as_euler("xyz", degrees=True)
    z_axis = t[:3, 2]
    return {
        "translation_m": [round(float(v), 5) for v in t[:3, 3]],
        "rpy_deg": [round(float(v), 3) for v in rpy],
        "yaw_deg": round(float(np.degrees(np.arctan2(t[1, 0], t[0, 0]))), 3),
        "tilt_deg": round(float(np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))), 3),
    }
