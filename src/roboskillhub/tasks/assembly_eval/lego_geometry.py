"""Pose arithmetic for the LEGO evaluator, in plain Python so the CPU tests need no numpy.

Quaternions are xyzw (the shared contract's order). BrickSim and USD use wxyz; convert at the boundary with
`wxyz_to_xyzw`. A pose is ``(p, q)`` with ``p`` a 3-list in metres and ``q`` an xyzw quaternion; ``T_A_B`` is
the pose of B in A.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vec = Sequence[float]
Quat = Sequence[float]
Pose = tuple[list[float], list[float]]


def wxyz_to_xyzw(q: Quat) -> list[float]:
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def q_normalize(q: Quat) -> list[float]:
    n = math.sqrt(sum(c * c for c in q))
    if n == 0.0:
        raise ValueError("zero quaternion")
    return [c / n for c in q]


def q_mul(a: Quat, b: Quat) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def q_conj(q: Quat) -> list[float]:
    return [-q[0], -q[1], -q[2], q[3]]


def q_rotate(q: Quat, v: Vec) -> list[float]:
    x, y, z, _ = q_mul(q_mul(q, [v[0], v[1], v[2], 0.0]), q_conj(q))
    return [x, y, z]


def q_from_axis_angle(axis: Vec, angle: float) -> list[float]:
    n = math.sqrt(sum(c * c for c in axis))
    s = math.sin(angle / 2.0) / n
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0)]


def compose(T_A_B: Pose, T_B_C: Pose) -> Pose:
    pa, qa = T_A_B
    pb, qb = T_B_C
    r = q_rotate(qa, pb)
    return [pa[0] + r[0], pa[1] + r[1], pa[2] + r[2]], q_normalize(q_mul(qa, qb))


def invert(T_A_B: Pose) -> Pose:
    p, q = T_A_B
    qi = q_conj(q)
    r = q_rotate(qi, p)
    return [-r[0], -r[1], -r[2]], qi


def relative(T_W_A: Pose, T_W_B: Pose) -> Pose:
    """T_A_B from two world poses."""
    return compose(invert(T_W_A), T_W_B)


def tilt_rad(q: Quat) -> float:
    """Angle between the frame's z axis and the reference z axis."""
    z = q_rotate(q, [0.0, 0.0, 1.0])
    return math.atan2(math.hypot(z[0], z[1]), z[2])


def yaw_rad(q: Quat) -> float:
    """Rotation about z of the frame's x axis, in (-pi, pi]."""
    x = q_rotate(q, [1.0, 0.0, 0.0])
    return math.atan2(x[1], x[0])


def wrap_symmetric(angle: float, period: float) -> float:
    """Distance of ``angle`` to the nearest multiple of ``period`` (signed, in [-period/2, period/2))."""
    return (angle + period / 2.0) % period - period / 2.0


def pose_error(T_W_goal: Pose, T_W_actual: Pose, yaw_period: float = 2.0 * math.pi) -> dict:
    """Error of ``actual`` in the goal frame: translation components, tilt and symmetric yaw."""
    p, q = relative(T_W_goal, T_W_actual)
    return {
        "dx_m": p[0],
        "dy_m": p[1],
        "dz_m": p[2],
        "dxy_m": math.hypot(p[0], p[1]),
        "tilt_rad": tilt_rad(q),
        "yaw_rad": wrap_symmetric(yaw_rad(q), yaw_period),
    }
