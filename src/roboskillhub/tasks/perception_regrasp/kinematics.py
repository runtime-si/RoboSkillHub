"""YAM joint-path feasibility: IK along the whole TCP path, not just its ends.

PR #106 (standing-a-lying-domino-by-reorient-and-regrasp) found that a 90 deg
wrist pitch with a held domino jumps a YAM joint by 0.6-3.1 rad somewhere in the
middle of the turn although both ends are reachable. A plan is therefore judged
on a dense sampling of the whole TCP path, each IK solve seeded with the previous
one, with joint limits, per-sample jumps and limit margins checked.

The chain and the TCP come from the hub's ``checking-dual-arm-clearance`` skill
(derived from i2rt's YAM model, MIT), loaded through :mod:`._hub`; only the
joint limits are added here (same i2rt MJCF, ``yam_i2rt_arms.xml``).
The default :class:`YamIk` is a damped-least-squares solver in this process. A
station's own IK (``robot.solve_ik``) can be passed instead through
:class:`IkSolver`; its branch choices are then what the check sees.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from . import _hub
from .frames import interpolate_T, inv_T, log_so3, make_T, rotation_angle

#: i2rt MJCF joint1..joint6 ranges [rad] (yam_i2rt_arms.xml; MIT, Copyright (c) I2RT Robotics).
JOINT_LIMITS = np.array(
    [
        [-2.61799, 3.05433],
        [0.0, 3.65],
        [0.0, 3.66519],
        [-1.5708, 1.5708],
        [-1.5708, 1.5708],
        [-2.0944, 2.0944],
    ]
)

#: Largest joint change between neighbouring path samples [rad]. PR #106's
#: ``PITCH_JOINT_JUMP``; at a 1 cm / 5 deg sampling a continuous path moves a joint
#: well under 0.2 rad per sample, so 0.6 means the solution changed branch.
JOINT_JUMP_RAD = 0.6
#: Path sampling: TCP translation and rotation between IK solves.
CART_STEP_M = 0.01
ROT_STEP_RAD = math.radians(5.0)
#: A solve counts when the TCP is within these of the target.
IK_POS_TOL_M = 5e-4
IK_ROT_TOL_RAD = math.radians(0.5)
#: Joints within this of a limit are reported (and fail when ``min_limit_margin_rad`` asks).
LIMIT_WARN_RAD = 0.05


class IkSolver(Protocol):
    """``solve(T_base_tcp, seed) -> q (6,) or None``. Must not move anything."""

    def solve(self, T_base_tcp: np.ndarray, seed: np.ndarray) -> np.ndarray | None: ...


def _chain():
    return _hub.load("yam_chain")


def tcp_in_gripper() -> np.ndarray:
    """``T_gripper_tcp``: the i2rt grasp_site (134.7 mm along the gripper's -z,
    turned 180 deg about x). TCP +z is the approach direction (out of the
    fingers), TCP +y the jaw closing axis."""
    c = _chain()
    return c._frame(*c.DEFAULT_TCP)


def fk(q: Sequence[float], T_gripper_tcp: np.ndarray | None = None) -> np.ndarray:
    """``T_base_tcp`` for six arm joints."""
    tcp = tcp_in_gripper() if T_gripper_tcp is None else T_gripper_tcp
    return _chain().forward_kinematics(np.asarray(q, dtype=float))["gripper"] @ tcp


def _jacobian(q: np.ndarray, tcp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = _chain()
    frames = c.forward_kinematics(q)
    T = frames["gripper"] @ tcp
    p = T[:3, 3]
    J = np.zeros((6, 6))
    for name, _parent, _pos, _quat, joint, axis in c._CHAIN:
        if isinstance(joint, int):
            Tb = frames[name]
            a = Tb[:3, :3] @ np.asarray(axis, dtype=float)
            J[:3, joint] = np.cross(a, p - Tb[:3, 3])
            J[3:, joint] = a
    return J, T


@dataclass
class YamIk:
    """Damped least squares from the seed, joints clamped to their limits."""

    T_gripper_tcp: np.ndarray | None = None
    max_iters: int = 200
    damping: float = 1e-3
    pos_tol_m: float = IK_POS_TOL_M
    rot_tol_rad: float = IK_ROT_TOL_RAD

    def solve(self, T_base_tcp: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
        tcp = tcp_in_gripper() if self.T_gripper_tcp is None else self.T_gripper_tcp
        q = np.clip(np.asarray(seed, dtype=float).copy(), JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        history: list[float] = []
        for it in range(self.max_iters):
            J, T = _jacobian(q, tcp)
            ep = T_base_tcp[:3, 3] - T[:3, 3]
            er = T[:3, :3] @ log_so3(T[:3, :3].T @ T_base_tcp[:3, :3])
            if np.linalg.norm(ep) < self.pos_tol_m and np.linalg.norm(er) < self.rot_tol_rad:
                return q
            err = float(np.linalg.norm(ep) + 0.05 * np.linalg.norm(er))
            history.append(err)
            # Pinned against a limit or in a local minimum: no 2 % gain in 20 iterations.
            if it >= 20 and err > 0.98 * history[-21]:
                return None
            e = np.concatenate([ep, er * 0.2])
            Jw = J.copy()
            Jw[3:] *= 0.2
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + self.damping * np.eye(6), e)
            step = float(np.max(np.abs(dq)))
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(q + dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        T = fk(q, tcp)
        if (
            np.linalg.norm(T_base_tcp[:3, 3] - T[:3, 3]) < self.pos_tol_m
            and rotation_angle(T[:3, :3], T_base_tcp[:3, :3]) < self.rot_tol_rad
        ):
            return q
        return None


def default_seeds() -> list[np.ndarray]:
    """Starting guesses for a first solve with no current joint reading: elbow-up
    postures with the hand pointing down, at a few base yaws and wrist rolls."""
    seeds = []
    for j1 in (0.0, 0.6, -0.6, 1.2, -1.2):
        for j6 in (0.0, 1.2, -1.2):
            seeds.append(np.array([j1, 1.2, 1.0, -0.6, 0.0, j6]))
    return seeds


@dataclass
class PathReport:
    feasible: bool
    reason: str
    samples: int
    q_path: list[list[float]] = field(default_factory=list)
    max_jump_rad: float = 0.0
    max_jump_joint: int | None = None
    max_jump_at: int | None = None
    min_limit_margin_rad: float = math.inf
    min_limit_margin_joint: int | None = None
    failed_at: int | None = None
    labels: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "reason": self.reason,
            "samples": self.samples,
            "max_jump_rad": round(self.max_jump_rad, 4),
            "max_jump_joint": self.max_jump_joint,
            "max_jump_at": self.max_jump_at,
            "min_limit_margin_rad": round(self.min_limit_margin_rad, 4)
            if math.isfinite(self.min_limit_margin_rad)
            else None,
            "min_limit_margin_joint": self.min_limit_margin_joint,
            "failed_at": self.failed_at,
            "failed_label": self.labels[self.failed_at]
            if self.failed_at is not None and self.failed_at < len(self.labels)
            else None,
            "q_start": self.q_path[0] if self.q_path else None,
            "q_end": self.q_path[-1] if self.q_path else None,
        }


def densify(
    waypoints: Sequence[np.ndarray],
    labels: Sequence[str] | None = None,
    cart_step_m: float = CART_STEP_M,
    rot_step_rad: float = ROT_STEP_RAD,
) -> tuple[list[np.ndarray], list[str]]:
    """Samples along straight-line / geodesic legs between TCP waypoints."""
    labels = list(labels) if labels is not None else [f"wp{i}" for i in range(len(waypoints))]
    out, out_labels = [np.asarray(waypoints[0], dtype=float)], [labels[0]]
    for i in range(1, len(waypoints)):
        a, b = np.asarray(waypoints[i - 1], dtype=float), np.asarray(waypoints[i], dtype=float)
        n = max(
            1,
            math.ceil(np.linalg.norm(b[:3, 3] - a[:3, 3]) / cart_step_m),
            math.ceil(rotation_angle(a[:3, :3], b[:3, :3]) / rot_step_rad),
        )
        for k in range(1, n + 1):
            out.append(interpolate_T(a, b, k / n))
            out_labels.append(labels[i] if k == n else f"{labels[i - 1]}->{labels[i]}")
    return out, out_labels


def evaluate_joint_path(
    waypoints_base: Sequence[np.ndarray],
    ik: IkSolver,
    q_seed: np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    jump_rad: float = JOINT_JUMP_RAD,
    min_limit_margin_rad: float = 0.0,
    cart_step_m: float = CART_STEP_M,
    rot_step_rad: float = ROT_STEP_RAD,
    seeds: Sequence[np.ndarray] | None = None,
) -> PathReport:
    """IK every sample of the TCP path (``T_base_tcp`` waypoints), continuing from
    the previous solution. The first sample tries ``q_seed`` then ``seeds``; each
    start is continued, and the first that passes is returned (else the one that
    got furthest)."""
    samples, sample_labels = densify(waypoints_base, labels, cart_step_m, rot_step_rad)
    starts = ([np.asarray(q_seed, dtype=float)] if q_seed is not None else []) + list(
        seeds if seeds is not None else default_seeds()
    )
    best: PathReport | None = None
    tried: list[np.ndarray] = []
    for seed in starts:
        q0 = ik.solve(samples[0], seed)
        if q0 is None or any(np.max(np.abs(q0 - t)) < 1e-3 for t in tried):
            continue
        tried.append(q0)
        rep = _continue(samples, sample_labels, ik, q0, jump_rad, min_limit_margin_rad)
        if rep.feasible:
            return rep
        if best is None or (rep.failed_at or 0) > (best.failed_at or 0):
            best = rep
    if best is None:
        return PathReport(
            False,
            "unreachable: no IK solution at the first pose",
            len(samples),
            failed_at=0,
            labels=sample_labels,
        )
    return best


def _continue(samples, labels, ik, q0, jump_rad, min_margin) -> PathReport:
    rep = PathReport(True, "ok", len(samples), q_path=[q0.tolist()], labels=list(labels))
    q = q0
    _margin(rep, q, 0)
    for k in range(1, len(samples)):
        qn = ik.solve(samples[k], q)
        if qn is None:
            # The local solve stalled; is the pose reachable from fresh seeds, and how far away?
            others = [
                s for s in (ik.solve(samples[k], sd) for sd in default_seeds()) if s is not None
            ]
            if others:
                qn = min(others, key=lambda s: float(np.max(np.abs(s - q))))
            else:
                rep.feasible, rep.failed_at = False, k
                rep.reason = f"unreachable at sample {k} ({labels[k]}): no IK solution"
                return rep
        d = np.abs(qn - q)
        j = int(np.argmax(d))
        if d[j] > rep.max_jump_rad:
            rep.max_jump_rad, rep.max_jump_joint, rep.max_jump_at = float(d[j]), j + 1, k
        rep.q_path.append(qn.tolist())
        _margin(rep, qn, k)
        if d[j] > jump_rad:
            rep.feasible, rep.failed_at = False, k
            rep.reason = (
                f"joint_jump: joint{j + 1} moves {d[j]:.2f} rad between samples {k - 1} and {k} "
                f"({labels[k]}), over {jump_rad} rad"
            )
            return rep
        q = qn
    if rep.min_limit_margin_rad < min_margin:
        rep.feasible = False
        rep.reason = (
            f"limit_margin: joint{rep.min_limit_margin_joint} comes {rep.min_limit_margin_rad:.3f} rad "
            f"from its limit, under {min_margin} rad"
        )
    return rep


def _margin(rep: PathReport, q: np.ndarray, k: int) -> None:
    m = np.minimum(q - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q)
    j = int(np.argmin(m))
    if m[j] < rep.min_limit_margin_rad:
        rep.min_limit_margin_rad, rep.min_limit_margin_joint = float(m[j]), j + 1


@dataclass
class CallableIk:
    """Wrap a station IK (e.g. ``robot.solve_ik``) as an :class:`IkSolver`.

    ``fn(T_base_tcp, seed) -> q | None``. Whether the station honours the seed
    decides whether its branch choices are continuous; the path check reports
    what it gets."""

    fn: Callable[[np.ndarray, np.ndarray], Any]
    verify_fk: bool = True
    """Check each returned solution with the i2rt model's FK; one that misses the
    target by more than the tolerances (a different TCP, a bad solver, a model
    mismatch) counts as no solution and is tallied in ``mismatches``."""
    pos_tol_m: float = 0.005
    rot_tol_rad: float = math.radians(3.0)
    mismatches: int = 0

    def solve(self, T_base_tcp: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
        q = self.fn(T_base_tcp, seed)
        if q is None:
            return None
        q = np.asarray(q, dtype=float).reshape(6)
        if self.verify_fk:
            T = fk(q)
            if (
                np.linalg.norm(T[:3, 3] - T_base_tcp[:3, 3]) > self.pos_tol_m
                or rotation_angle(T[:3, :3], T_base_tcp[:3, :3]) > self.rot_tol_rad
            ):
                self.mismatches += 1
                return None
        return q


def world_to_base(T_world_base: np.ndarray, T_world_x: np.ndarray) -> np.ndarray:
    return inv_T(T_world_base) @ T_world_x


__all__ = [
    "JOINT_LIMITS",
    "JOINT_JUMP_RAD",
    "CallableIk",
    "IkSolver",
    "PathReport",
    "YamIk",
    "default_seeds",
    "densify",
    "evaluate_joint_path",
    "fk",
    "make_T",
    "tcp_in_gripper",
    "world_to_base",
]
