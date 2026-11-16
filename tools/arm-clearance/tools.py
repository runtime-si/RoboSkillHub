"""Reach and clearance of a serial arm against a scene of boxes, computed here and nowhere else.

A pose search (which hand orientation, which height, which spot) asks "can the arm be there, clear
of the fixed scene?" hundreds of times. These functions answer from a kinematic chain, link
capsules and obstacle boxes passed as arguments, with numpy only, so the search never queries a
motion server. The server still has the last word when the one chosen move is sent.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

import numpy as np
from gap_core.tools import tool

IK_ITERS = 80
IK_DAMPING = 1.0e-4
IK_STEP = 0.2
"""Largest joint step per iteration [rad]."""
IK_ROTATION_WEIGHT = 0.3
"""Weight of the rotation error against the position error (rad against m) in the solve."""
IK_CONVERGED = (2.0e-4, 2.0e-3)
"""Stop iterating under this position [m] and rotation [rad] error."""
IK_ACCEPT = (1.0e-3, 0.02)
"""A solve ending over this position [m] or rotation [rad] error is no solution. tuned_on: a 6-joint
i2rt YAM arm (0.6 m reach) in a MuJoCo twin of one station, the settings of that task's local solver.
A pose it called reachable was still sometimes refused by the motion server on the way there (its
plan followed the straight line), so a yes here is a prediction."""

CLEARANCE_LIMIT = 0.15
"""Distances are looked for up to this [m]; further reads as this."""

SPHERE_SPACING = 0.5
"""A link capsule is tested as spheres this fraction of its radius apart along its axis, which
under-reads the capsule's clearance by at most 3 % of the radius (1 mm on a 30 mm link)."""

MOVE_AWAY_SLACK = 0.002
"""A step is refused only when it ends this much [m] nearer than the arm is now: an arm already
inside the margin may always move away."""


class IkResult(TypedDict):
    ok: bool
    q: list
    position_error_m: float
    rotation_error_rad: float
    reason: str


class ClearanceResult(TypedDict):
    clearance_m: float
    link: str
    obstacle: str
    nearest: str


class StepResult(TypedDict):
    blocked: bool
    reason: str
    clearance_m: float
    clearance_now_m: float


class ReachResult(TypedDict):
    reachable: bool
    q: list
    reason: str
    clearance_m: float
    nearest: str


# ----------------------------------------------------------------------------- rotations
def _quat_mat(q: Any) -> np.ndarray:
    if isinstance(q, dict):
        q = [q["w"], q["x"], q["y"], q["z"]]
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) or 1.0)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def _rotvec(r: np.ndarray) -> np.ndarray:
    """The rotation vector of a rotation matrix (angle in [0, pi])."""
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if w > 1e-6:
        v = v / (4 * w)
    else:  # a half turn: the axis from the diagonal
        d = np.sqrt(np.maximum((np.diag(r) + 1) / 2, 0.0))
        i = int(np.argmax(d))
        v = np.array([(r[i, j] + r[j, i]) / (4 * d[i]) if j != i else d[i] for j in range(3)])
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    return v / n * (2 * math.atan2(n, w))


def _vec(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=float).reshape(3)


def _frame(pose: Any) -> np.ndarray:
    """A 4x4 from None (identity), an Se3Pose {position, rotation}, or a 4x4."""
    if pose is None:
        return np.eye(4)
    if isinstance(pose, dict):
        pose = pose.get("pose", pose)
        t = np.eye(4)
        t[:3, :3] = _quat_mat(pose.get("rotation") or pose.get("quat") or [1, 0, 0, 0])
        t[:3, 3] = _vec(pose["position"])
        return t
    return np.asarray(pose, dtype=float).reshape(4, 4)


# ----------------------------------------------------------------------------- the arm
class Arm:
    """A serial chain in the MJCF convention: each body sits at ``position`` / ``quat`` in its
    parent and, when it has a joint, turns about ``axis`` (its own frame) at its origin."""

    def __init__(self, model: dict[str, Any]) -> None:
        self.bodies = []
        for b in model["chain"]:
            t = np.eye(4)
            t[:3, :3] = _quat_mat(b.get("quat") or [1, 0, 0, 0])
            t[:3, 3] = _vec(b.get("position") or [0, 0, 0])
            joint = b.get("joint")
            axis = np.asarray(b.get("axis") or [0, 0, 1], float) if joint is not None else None
            self.bodies.append((str(b["name"]), b.get("parent"), t, joint, axis))
        names = [b[0] for b in self.bodies]
        for name, parent, *_ in self.bodies:
            if parent is not None and (
                parent not in names or names.index(parent) >= names.index(name)
            ):
                raise ValueError(f"chain: {name!r} must come after its parent {parent!r}")
        self.dof = 1 + max(int(b[3]) for b in self.bodies if b[3] is not None)
        limits = np.asarray(model.get("limits") or [[-math.pi, math.pi]] * self.dof, float)
        if limits.shape != (self.dof, 2):
            raise ValueError(f"limits must be {self.dof} [lo, hi] pairs")
        self.lo, self.hi = limits[:, 0], limits[:, 1]
        self.base = _frame(model.get("base"))
        tcp = model.get("tcp") or {}
        self.tcp_body = str(tcp.get("body") or names[-1])
        if self.tcp_body not in names:
            raise ValueError(f"tcp body {self.tcp_body!r} is not in the chain")
        self.tcp = np.eye(4)
        self.tcp[:3, :3] = _quat_mat(tcp.get("quat") or [1, 0, 0, 0])
        self.tcp[:3, 3] = _vec(tcp.get("position") or [0, 0, 0])
        # link capsules as spheres along their axes, in each body's frame
        self.spheres: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for body, caps in (model.get("capsules") or {}).items():
            if body not in names:
                raise ValueError(f"capsules: {body!r} is not in the chain")
            centres, radii = [], []
            for a, b, r in caps:
                a, b, r = _vec(a), _vec(b), float(r)
                n = 1 + int(math.ceil(float(np.linalg.norm(b - a)) / max(SPHERE_SPACING * r, 1e-6)))
                centres.append(a + np.linspace(0.0, 1.0, n)[:, None] * (b - a))
                radii.append(np.full(n, r))
            if centres:
                self.spheres[body] = (np.vstack(centres), np.concatenate(radii))

    def frames(self, q: Any) -> dict[str, np.ndarray]:
        """Every body's world frame at joints ``q``."""
        q = np.asarray(q, float).reshape(-1)[: self.dof]
        out: dict[str, np.ndarray] = {}
        for name, parent, t, joint, axis in self.bodies:
            f = (self.base if parent is None else out[parent]) @ t
            if joint is not None:
                turn = np.eye(4)
                turn[:3, :3] = _axis_angle(axis, float(q[int(joint)]))
                f = f @ turn
            out[name] = f
        return out

    def tool(self, q: Any) -> np.ndarray:
        return self.frames(q)[self.tcp_body] @ self.tcp

    def jacobian(self, q: Any) -> tuple[np.ndarray, np.ndarray]:
        """(the tool's world frame, its 6 x dof geometric Jacobian in the world)."""
        frames = self.frames(q)
        tool_frame = frames[self.tcp_body] @ self.tcp
        p = tool_frame[:3, 3]
        # which joints move the tool: those on the path from the tool's body to the root
        on_path, body = set(), self.tcp_body
        parents = {b[0]: b[1] for b in self.bodies}
        while body is not None:
            on_path.add(body)
            body = parents[body]
        jac = np.zeros((6, self.dof))
        for name, _parent, _t, joint, axis in self.bodies:
            if joint is None or name not in on_path:
                continue
            a = frames[name][:3, :3] @ (axis / np.linalg.norm(axis))
            jac[:3, int(joint)] = np.cross(a, p - frames[name][:3, 3])
            jac[3:, int(joint)] = a
        return tool_frame, jac

    def ik(
        self, target: np.ndarray, q0: Any, iters: int = IK_ITERS
    ) -> tuple[np.ndarray, float, float]:
        """Damped least squares from ``q0`` inside the joint limits: (q, position error [m],
        rotation error [rad])."""
        q = np.clip(np.asarray(q0, float).reshape(-1)[: self.dof], self.lo, self.hi)
        weight = np.diag([1, 1, 1] + [IK_ROTATION_WEIGHT] * 3)
        ep = er = float("inf")
        for _ in range(max(1, int(iters))):
            now, jac = self.jacobian(q)
            e_p = target[:3, 3] - now[:3, 3]
            e_r = _rotvec(target[:3, :3] @ now[:3, :3].T)
            ep, er = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
            if ep < IK_CONVERGED[0] and er < IK_CONVERGED[1]:
                break
            jw = weight @ jac
            dq = jw.T @ np.linalg.solve(
                jw @ jw.T + IK_DAMPING * np.eye(6), weight @ np.r_[e_p, e_r]
            )
            step = float(np.linalg.norm(dq))
            if step > IK_STEP:
                dq *= IK_STEP / step
            q = np.clip(q + dq, self.lo, self.hi)
        else:
            now = self.tool(q)
            ep = float(np.linalg.norm(target[:3, 3] - now[:3, 3]))
            er = float(np.linalg.norm(_rotvec(target[:3, :3] @ now[:3, :3].T)))
        return q, ep, er

    def clearance(
        self, q: Any, obstacles: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], limit: float
    ) -> tuple[float, str, str]:
        """The smallest distance from any link sphere to any obstacle box (negative: inside),
        up to ``limit``, with the link and the obstacle it is between."""
        frames = self.frames(q)
        best, link, what = float(limit), "", ""
        for body, (centres, radii) in self.spheres.items():
            f = frames[body]
            world = centres @ f[:3, :3].T + f[:3, 3]
            for name, centre, rot, half in obstacles:
                loc = (world - centre) @ rot
                out = np.maximum(np.abs(loc) - half, 0.0)
                outside = np.linalg.norm(out, axis=1)
                inside = -np.min(half - np.abs(loc), axis=1)
                dist = np.where(out.any(axis=1), outside, inside) - radii
                k = int(np.argmin(dist))
                if dist[k] < best:
                    best, link, what = float(dist[k]), body, name
        return best, link, what


def boxes(obstacles: Any) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Obstacle boxes as (name, centre, rotation, half sizes). Each is {name, center, half, quat}
    or an OrientedBoundingBox {center, extent, orientation} (half extents) with an optional name."""
    out = []
    for i, o in enumerate(obstacles or []):
        half = o.get("half") if o.get("half") is not None else o.get("extent")
        quat = o.get("quat") or o.get("orientation") or [1, 0, 0, 0]
        out.append(
            (str(o.get("name") or f"box {i}"), _vec(o["center"]), _quat_mat(quat), _vec(half))
        )
    return out


def step_blocked(
    arm: Arm, obstacles: list, q_now: Any, q_goal: Any, margin: float
) -> tuple[str, float, float]:
    """Why a joint step from ``q_now`` to ``q_goal`` is refused ("" when it is not), the smallest
    clearance over its middle and its end, and the clearance now.

    Refused: nearer the scene than ``margin`` at the middle or the end of the step, and nearer
    than the arm is now."""
    q_now, q_goal = np.asarray(q_now, float), np.asarray(q_goal, float)
    limit = max(CLEARANCE_LIMIT, margin + 0.01)
    here = arm.clearance(q_now, obstacles, limit)[0]
    worst, reason = float("inf"), ""
    for q in ((q_now + q_goal) / 2, q_goal):
        c, link, what = arm.clearance(q, obstacles, limit)
        worst = min(worst, c)
        if not reason and c < margin and c < here - MOVE_AWAY_SLACK:
            reason = f"{link} would be {100 * c:.1f} cm from {what} (margin {100 * margin:.0f} cm)"
    return reason, worst, here


# ----------------------------------------------------------------------------- tools
@tool(
    name="arm-clearance.solve_ik",
    summary="Local damped-least-squares IK for a serial chain given as an argument: joints for a world tool pose, or why not. Queries no robot.",
    tags=("planning",),
)
def solve_ik(
    model: dict[str, Any],
    pose: dict[str, Any],
    q0: list[float],
    bounds: dict[str, Any] | None = None,
    iters: int = IK_ITERS,
) -> IkResult:
    """Joints that put the chain's tool at ``pose`` (world Se3Pose), starting from ``q0``.

    ``model``: {chain: [{name, parent, position, quat (wxyz), joint (index or null), axis}],
    tcp: {body, position, quat}, limits: [[lo, hi], ...], base: Se3Pose of the chain's root in
    the world, capsules: {body: [[a, b, radius], ...]}}. ``bounds``: {min: [x, y, z], max: [...]}
    the tool position must lie in, in the chain root's frame (an arm controller's workspace box).
    ``ok`` is false outside the bounds or when the solve ends over 1 mm / 0.02 rad from the pose
    inside the joint limits; ``q`` is then the best found.
    """
    arm = Arm(model)
    target = _frame(pose)
    q_start = [float(v) for v in np.asarray(q0, float).reshape(-1)[: arm.dof]]
    if bounds is not None:
        local = np.linalg.inv(arm.base) @ np.append(target[:3, 3], 1.0)
        if np.any(local[:3] < _vec(bounds["min"]) - 1e-9) or np.any(
            local[:3] > _vec(bounds["max"]) + 1e-9
        ):
            return {
                "ok": False,
                "q": q_start,
                "position_error_m": -1.0,
                "rotation_error_rad": -1.0,
                "reason": "outside the arm's workspace box",
            }
    q, ep, er = arm.ik(target, q_start, iters)
    ok = ep <= IK_ACCEPT[0] and er <= IK_ACCEPT[1]
    return {
        "ok": bool(ok),
        "q": [float(v) for v in q],
        "position_error_m": ep,
        "rotation_error_rad": er,
        "reason": ""
        if ok
        else f"no solution inside the joint limits ({ep * 1e3:.1f} mm, {er:.3f} rad off)",
    }


@tool(
    name="arm-clearance.clearance",
    summary="Smallest distance from an arm's link capsules at given joints to a scene of boxes (negative inside), and which link and box.",
    tags=("planning",),
)
def clearance(
    model: dict[str, Any],
    q: list[float],
    obstacles: list[dict[str, Any]],
    limit_m: float = CLEARANCE_LIMIT,
) -> ClearanceResult:
    """``obstacles``: world boxes {name, center, half, quat (wxyz)} or OrientedBoundingBoxes.
    Distances are looked for up to ``limit_m``; with nothing nearer, ``clearance_m`` is the limit
    and ``link`` empty."""
    best, link, what = Arm(model).clearance(q, boxes(obstacles), float(limit_m))
    return {
        "clearance_m": best,
        "link": link,
        "obstacle": what,
        "nearest": f"{link} to {what}" if link else "",
    }


@tool(
    name="arm-clearance.check_step",
    summary="Whether a joint step would bring the arm inside a margin of the scene, at its middle or end, and nearer than it is now.",
    tags=("planning",),
)
def check_step(
    model: dict[str, Any],
    q_now: list[float],
    q_goal: list[float],
    obstacles: list[dict[str, Any]],
    margin_m: float = 0.02,
) -> StepResult:
    """``blocked`` when the arm would be under ``margin_m`` from a box half way or at the goal
    and nearer than at ``q_now`` (an arm already close may always move away). Two samples only:
    keep steps short (the source used 6 cm / 20 degrees of tool motion)."""
    reason, worst, here = step_blocked(Arm(model), boxes(obstacles), q_now, q_goal, float(margin_m))
    return {
        "blocked": bool(reason),
        "reason": reason,
        "clearance_m": worst,
        "clearance_now_m": here,
    }


@tool(
    name="arm-clearance.reachable",
    summary="Can the arm take a tool pose clear of the scene? Local IK inside the joint limits and workspace box, then clearance against boxes; no robot is asked.",
    tags=("planning",),
)
def reachable(
    model: dict[str, Any],
    pose: dict[str, Any],
    q_now: list[float],
    obstacles: list[dict[str, Any]] | None = None,
    margin_m: float = 0.02,
    bounds: dict[str, Any] | None = None,
) -> ReachResult:
    """``solve_ik`` from ``q_now``, then ``check_step`` from ``q_now`` to the solution. With no
    obstacles only the reach is answered. ``q`` is the solution when there is one, whether or not
    the step is blocked."""
    found = solve_ik(model, pose, q_now, bounds)
    if not found["ok"]:
        return {
            "reachable": False,
            "q": [],
            "reason": found["reason"],
            "clearance_m": float("nan"),
            "nearest": "",
        }
    arm, scene = Arm(model), boxes(obstacles)
    reason, worst, _here = step_blocked(arm, scene, q_now, found["q"], float(margin_m))
    _c, link, what = arm.clearance(found["q"], scene, max(CLEARANCE_LIMIT, margin_m + 0.01))
    return {
        "reachable": not reason,
        "q": found["q"],
        "reason": reason,
        "clearance_m": worst,
        "nearest": f"{link} to {what}" if link else "",
    }
