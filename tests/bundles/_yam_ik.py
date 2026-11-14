"""A CPU YAM arm for bimanual tests: forward kinematics from robosimstudio's yam_bimanual_arm.urdf joint
origins, a damped-least-squares IK for a tool point TOOL ahead of link_6 with its approach axis pointing
down, and two arms whose bases sit SPACING apart along y (the station twin's 0.61 m)."""

from __future__ import annotations

import numpy as np

JOINTS = [  # (xyz, rpy) of each revolute joint's origin, axis z; the URDF's values
    ((0, 0, 0.0631), (0, 0, 1.57079633)),
    ((2.5e-05, -0.02, 0.0409), (1.57079633, 0, 1.57079633)),
    ((0.264, 0, 0), (3.14159265, 0, 0)),
    ((-0.245, -0.06, 0), (0, 0, 0)),
    ((-0.074, -0.0395, 2.5e-05), (-1.57079633, 1.57079633, 0)),
    ((0, 0.0353, 0.0395), (-1.57079633, 0, 0)),
]
LIMITS = np.array(
    [
        (-2.61799, 3.05433),
        (0, 3.66519),
        (0, 3.66519),
        (-1.5708, 1.5708),
        (-1.5708, 1.5708),
        (-2.0944, 2.0944),
    ]
)
TOOL = np.array([0.0, 0.0, 0.13])
"""Tool centre in link_6 (approximate linear_4310 reach) [m]."""
SPACING = 0.61


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])  # fmt: skip


def fk(q):
    t = np.eye(4)
    for (xyz, rpy), qi in zip(JOINTS, q, strict=True):
        o = np.eye(4)
        o[:3, :3], o[:3, 3] = _rpy(*rpy), xyz
        rz = np.eye(4)
        rz[:2, :2] = [[np.cos(qi), -np.sin(qi)], [np.sin(qi), np.cos(qi)]]
        t = t @ o @ rz
    p = t[:3, :3] @ TOOL + t[:3, 3]
    return p, t[:3, :3] @ np.array([0.0, 0.0, 1.0])


def ik(p, seeds=8, rng=None, tol=0.003):
    """Joint angles reaching p with the tool axis within ~10 deg of straight down, or None."""
    rng = rng or np.random.default_rng(0)
    down = np.array([0.0, 0.0, -1.0])
    for s in range(seeds):
        q = rng.uniform(LIMITS[:, 0], LIMITS[:, 1]) if s else LIMITS.mean(axis=1)
        for _ in range(80):
            x, a = fk(q)
            e = np.r_[p - x, 0.05 * (down - a)]
            if np.linalg.norm(e[:3]) < tol and np.dot(a, down) > 0.985:
                return q
            j = np.zeros((6, 6))
            for k in range(6):
                dq = np.zeros(6)
                dq[k] = 1e-5
                x2, a2 = fk(q + dq)
                j[:, k] = np.r_[x2 - x, 0.05 * (a2 - a)] / 1e-5
            q = np.clip(
                q + j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), e), LIMITS[:, 0], LIMITS[:, 1]
            )
    return None


class TwoArms:
    """robot.solve_ik for two YAMs: arm 0 based at y = +SPACING/2, arm 1 at -SPACING/2 (world = arm 0 frame
    shifted)."""

    def __init__(self):
        self.calls = 0
        self.cache = {}

    def solve_ik(self, pose, arm_id=0, **_):
        self.calls += 1
        p = np.array([float(pose["position"][k]) for k in "xyz"])
        base = np.array([0.0, SPACING / 2 if int(arm_id) == 0 else -SPACING / 2, 0.0])
        key = (int(arm_id), *np.round(p, 4))
        if key not in self.cache:
            self.cache[key] = ik(p - base)
        q = self.cache[key]
        return {
            "success": q is not None,
            "joint_positions": [] if q is None else [float(v) for v in q],
        }
