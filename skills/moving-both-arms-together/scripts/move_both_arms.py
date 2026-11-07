# Adapted from RoboRSI (https://github.com/nssmd/RoboRSI),
# roborsi/embodied/skills/base/move_dual_arm/robotwin/policy.py, Apache-2.0.
# Kept: both hands go to their target poses as one move, both targets are
# checked for reach before anything moves (abort without moving), and each
# arm's arrival is read back against its own target (`left_reached` /
# `right_reached`, here at 0.02 m instead of 0.03; `left_moved` /
# `right_moved` at 0.005 m).
# Changed: upstream hands both targets to one cuRobo dual-arm plan; the hub's
# connectors drive one arm per call, so the two straight paths are split into
# the same number of short segments and the arms take turns, the lead
# alternating, with each arm's arrival read back per segment. The preflight
# is robot.solve_ik on every waypoint of both paths (upstream: a cuRobo plan
# to each target from a neutral start, failing open when the checker
# errors). Added: an optional bound on the hands' relative pose for a
# two-handed carry (the second hand's waypoints ride on the first's), the
# wrist roll finished on the last joint when the arm's solver leaves it free,
# a tool-centre separation floor, and routed exits instead of an ok flag.
"""Move both hands to their targets as one coordinated motion, in interleaved short segments."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

MAX_STEP = 0.02
"""Longest straight-line segment either hand takes per turn [m]: while one arm
moves the other stands, so the hands' relative pose is off by up to one step
between turns. 20 mm keeps that under the default relative bound."""

MAX_STEP_DEG = 10.0
"""Largest rotation either hand takes per turn [deg]."""

MAX_SEGMENTS = 60
"""More segments than this are refused (1.2 m at 20 mm): each segment is two
settled servo moves."""

REACH_TOLERANCE = 0.02
"""How far a hand may end from a waypoint or its target [m]: no tighter than a
servo's 4 mm or a real YAM station's 12 mm; upstream reads 0.03 at the end."""

RELATIVE_TOLERANCE = 0.02
"""Default bound on the change of hand 2's position in hand 1's frame while
keep_relative is set [m]: two arms' 12 mm station bounds can disagree by more,
so a real carry should give its own (what the held object tolerates)."""

RELATIVE_TOLERANCE_DEG = 5.0
"""Same, for the hands' relative rotation [deg]."""

MIN_HAND_DISTANCE = 0.08
"""Planned tool centres closer than this are refused [m]: a YAM's open jaws span
94 mm, so two hands closer than ~80 mm have finger bodies within each other's
reach. A proxy, not a link clearance; lower it for hands meant to meet."""

REFUSED_TOLERANCE = 0.012
"""A move that raised or replied a refusal counts as arrived only within this
[m]: a segment can be shorter than reach_tolerance, so a refused step must not
pass on distance alone. Also the IK preview's bound on a waypoint's position
error: the twin's preview missed edge-of-reach waypoints by 4-7 mm that the
arms then ended 6.6-11.1 mm from. A real YAM station's 12 mm per-move bound."""

ROLL_TOLERANCE_DEG = 3.0
"""A hand whose roll about its approach axis ends further than this from the
commanded one has the rest turned on its last joint [deg]: a servo's own 3 deg
arrival bound (with keep_relative, at most half of relative_tolerance_deg)."""

TURN_TRIES = 2
"""Last-joint turns per arm per segment, each read back; the first one learns
which way the joint turns the hand (handing-over-between-arms' procedure)."""

ROLL_LIMITS = (-2.0944, 2.0944)
"""Last joint's range [rad]: the i2rt YAM's joint 6 (+-120 deg, its MJCF)."""

LIMIT_MARGIN = 0.05
"""Kept clear of each roll limit [rad]."""

MOVED = 0.005
"""A hand that moved less than this from its start has not moved [m] (upstream)."""

SAME_ARM = 0.001
"""Two arm_ids whose tool centres read this close are one arm [m]."""


class Output(TypedDict):
    route: str
    detail: str
    segments: int
    segment_errors: list[dict[str, Any]]
    max_relative_error_m: float
    max_relative_error_deg: float
    max_transient_error_m: float
    left_residual_m: float
    right_residual_m: float
    min_hand_distance_m: float
    roll_turns: int
    left_pose: dict[str, Any]
    right_pose: dict[str, Any]
    right_target_pose: dict[str, Any]


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _quat(q: dict[str, float]) -> np.ndarray:
    a = np.array([float(q[k]) for k in ("w", "x", "y", "z")])
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])


def _qdict(a: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip("wxyz", a, strict=True)}


def _matrix(a: np.ndarray) -> np.ndarray:
    w, x, y, z = a
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _from_matrix(m: np.ndarray) -> np.ndarray:
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    x = math.copysign(
        math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0, m[2, 1] - m[1, 2]
    )
    y = math.copysign(
        math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2.0, m[0, 2] - m[2, 0]
    )
    z = math.copysign(
        math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2.0, m[1, 0] - m[0, 1]
    )
    a = np.array([w, x, y, z])
    return a / float(np.linalg.norm(a))


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """The rotation between two unit quaternions [rad]."""
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(a, b)))))


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + t * (b - a)
        return q / float(np.linalg.norm(q))
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def _relative(p0: np.ndarray, q0: np.ndarray, p1: np.ndarray, q1: np.ndarray):
    """Hand 2's pose in hand 1's frame: (position, rotation matrix)."""
    r0 = _matrix(q0)
    return r0.T @ (p1 - p0), r0.T @ _matrix(q1)


def _relative_error(expected, measured) -> tuple[float, float]:
    (pe, re), (pm, rm) = expected, measured
    cos = (float(np.trace(re.T @ rm)) - 1.0) / 2.0
    return float(np.linalg.norm(pm - pe)), math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _pair_error(plan, now) -> tuple[float, float]:
    """How far the hands' measured relative pose is from the planned one: the
    world-frame offset between the tool centres [m], and the relative rotation
    [deg]. World frame, so a wrist roll the arm skipped (a solver that keeps
    only the approach axis) shows as rotation, not as metres of offset."""
    (p0e, q0e), (p1e, q1e) = plan
    (p0m, q0m), (p1m, q1m) = now
    _, re = _relative(p0e, q0e, p1e, q1e)
    _, rm = _relative(p0m, q0m, p1m, q1m)
    return _relative_error((p1e - p0e, re), (p1m - p0m, rm))


def _roll_residual(commanded: np.ndarray, now: np.ndarray) -> float:
    """How far the hand must still turn about its commanded approach axis (tool z)
    to reach ``commanded`` [deg], signed."""
    rc = _matrix(commanded)
    err = rc @ _matrix(now).T
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(err)) - 1.0) / 2.0)))
    if angle < 1e-9:
        return 0.0
    if math.pi - angle < 1e-6:
        sym = err + np.eye(3)
        about = sym[:, int(np.argmax(np.linalg.norm(sym, axis=0)))]
        about = about / float(np.linalg.norm(about))
    else:
        about = np.array([err[2, 1] - err[1, 2], err[0, 2] - err[2, 0], err[1, 0] - err[0, 1]])
        about = about / (2.0 * math.sin(angle))
    return math.degrees(angle * float(np.dot(about, rc[:, 2])))


def _joints(ctx: NodeContext, arm_id: int) -> np.ndarray:
    """One arm's joint positions from robot.get_observation, or empty."""
    try:
        arms = ctx.tool("robot.get_observation").get("arms") or []
    except Exception as exc:
        logger.info("moving-both-arms-together: no joint reading (%s)", exc)
        return np.zeros(0)
    if int(arm_id) >= len(arms):
        return np.zeros(0)
    q = arms[int(arm_id)].get("joint_state")
    if isinstance(q, dict):
        q = q.get("positions")
    return np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)


def _refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or "" (connectors that route a
    refusal instead of raising)."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def run(
    ctx: NodeContext,
    left_target: dict[str, Any],
    right_target: dict[str, Any] | None = None,
    left_arm: int = 0,
    right_arm: int = 1,
    keep_relative: bool = False,
    relative_tolerance: float = RELATIVE_TOLERANCE,
    relative_tolerance_deg: float = RELATIVE_TOLERANCE_DEG,
    max_step: float = MAX_STEP,
    max_step_deg: float = MAX_STEP_DEG,
    max_segments: int = MAX_SEGMENTS,
    reach_tolerance: float = REACH_TOLERANCE,
    min_hand_distance: float = MIN_HAND_DISTANCE,
    preview: bool = True,
    fix_roll: bool = True,
) -> Output:
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "segments": 0,
        "segment_errors": [],
        "max_relative_error_m": 0.0,
        "max_relative_error_deg": 0.0,
        "max_transient_error_m": 0.0,
        "left_residual_m": -1.0,
        "right_residual_m": -1.0,
        "min_hand_distance_m": math.inf,
        "roll_turns": 0,
        "left_pose": {},
        "right_pose": {},
        "right_target_pose": {},
    }
    arms = (int(left_arm), int(right_arm))

    def arm(a: int) -> dict[str, int]:
        # arm_id only when nonzero: a single-arm connector takes no such keyword.
        return {"arm_id": a} if a else {}

    def here(a: int) -> tuple[np.ndarray, np.ndarray]:
        pose = _pose(ctx.tool("robot.get_ee_pose", **arm(a)))
        return _xyz(pose), _quat(pose.get("rotation") or {"w": 0, "x": 0, "y": 0, "z": 0})

    notes: list[str] = []

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, "; ".join(d for d in (detail, *notes) if d)
        return out

    if arms[0] == arms[1]:
        return finish("unreachable", f"both hands are arm {arms[0]}")
    start_l = _pose(ctx.tool("robot.get_ee_pose", **arm(arms[0])))
    start_r = _pose(ctx.tool("robot.get_ee_pose", **arm(arms[1])))
    rot_r = start_r.get("rotation") or {}
    if sum(float(rot_r.get(k, 0.0)) ** 2 for k in "wxyz") < 0.25 or (
        float(np.linalg.norm(_xyz(start_l) - _xyz(start_r))) < SAME_ARM
    ):
        # A connector that clamps or ignores arm_id reports one arm for both.
        return finish("unreachable", f"the connector does not report a separate arm {arms[1]}")
    p0 = [_xyz(start_l), _xyz(start_r)]
    q0 = [_quat(start_l["rotation"]), _quat(start_r["rotation"])]
    rel0 = _relative(p0[0], q0[0], p0[1], q0[1])

    def target(t: dict[str, Any] | None, i: int) -> tuple[np.ndarray, np.ndarray]:
        t = _pose(t)
        return _xyz(t), _quat(t["rotation"]) if t.get("rotation") else q0[i]

    pl, ql = target(left_target, 0)
    if right_target is None:
        if not keep_relative:
            return finish("unreachable", "no right_target and keep_relative is off")
        pr, qr = pl + _matrix(ql) @ rel0[0], _from_matrix(_matrix(ql) @ rel0[1])
    else:
        pr, qr = target(right_target, 1)
    out["right_target_pose"] = {"position": _vec(pr), "rotation": _qdict(qr)}
    if keep_relative:
        off_m, off_deg = _relative_error(rel0, _relative(pl, ql, pr, qr))
        if off_m > float(relative_tolerance) or off_deg > float(relative_tolerance_deg):
            return finish(
                "unreachable",
                f"the targets change the hands' relative pose by {off_m * 1e3:.0f} mm /"
                f" {off_deg:.1f} deg, over the {float(relative_tolerance) * 1e3:.0f} mm /"
                f" {float(relative_tolerance_deg):.1f} deg kept",
            )

    step = float(max_step)
    if keep_relative:
        # Between turns the hands are off by one step; keep that to half the bound.
        step = min(step, float(relative_tolerance) / 2.0)
    step_deg = min(
        float(max_step_deg), float(relative_tolerance_deg) / 2.0 if keep_relative else 1e9
    )
    lengths = (float(np.linalg.norm(pl - p0[0])), float(np.linalg.norm(pr - p0[1])))
    turns = (math.degrees(_angle(q0[0], ql)), math.degrees(_angle(q0[1], qr)))
    n = max(1, math.ceil(max(lengths) / step - 1e-9), math.ceil(max(turns) / step_deg - 1e-9))
    if n > int(max_segments):
        return finish(
            "unreachable",
            f"the move needs {n} segments of {step * 1e3:.0f} mm / {step_deg:.0f} deg,"
            f" over the {int(max_segments)} allowed",
        )
    out["segments"] = n

    # Waypoints 0..n per hand; with keep_relative the right hand rides on the left.
    way: list[list[tuple[np.ndarray, np.ndarray]]] = [[], []]
    for k in range(n + 1):
        t = k / n
        lp, lq = p0[0] + t * (pl - p0[0]), _slerp(q0[0], ql, t)
        way[0].append((lp, lq))
        if keep_relative:
            way[1].append((lp + _matrix(lq) @ rel0[0], _from_matrix(_matrix(lq) @ rel0[1])))
        else:
            way[1].append((p0[1] + t * (pr - p0[1]), _slerp(q0[1], qr, t)))
    way[0][-1], way[1][-1] = (pl, ql), (pr, qr)

    # Hands too close anywhere along the plan, including between turns.
    closest = min(
        float(np.linalg.norm(way[0][k][0] - way[1][j][0]))
        for k in range(n + 1)
        for j in (k - 1, k, k + 1)
        if 0 <= j <= n
    )
    if closest < float(min_hand_distance):
        return finish(
            "unreachable",
            f"the planned tool centres come {closest * 1e3:.0f} mm apart, under the"
            f" {float(min_hand_distance) * 1e3:.0f} mm kept between the hands",
        )

    if preview:
        for i, a in enumerate(arms):
            for k in range(1, n + 1):
                p, q = way[i][k]
                try:
                    reply = ctx.tool(
                        "robot.solve_ik",
                        pose={"position": _vec(p), "rotation": _qdict(q)},
                        **arm(a),
                    )
                except (
                    Exception
                ) as exc:  # a station raises an unreachable pose; a connector may lack the tool
                    if "unreachable" in str(exc).lower():
                        return finish(
                            "unreachable",
                            f"arm {a} waypoint {k}/{n} is out of reach (IK preview: {exc})",
                        )
                    logger.info("moving-both-arms-together: no IK preview (%s)", exc)
                    break
                missed = reply.get("position_error_m") if isinstance(reply, dict) else None
                if isinstance(reply, dict) and (
                    reply.get("success") is False
                    or (
                        float(missed) > REFUSED_TOLERANCE
                        if isinstance(missed, (int, float))
                        else reply.get("within_servo_tolerance") is False
                    )
                ):
                    return finish(
                        "unreachable",
                        f"arm {a} waypoint {k}/{n} at {np.round(p, 3).tolist()} is out of reach"
                        + (f" by {float(missed) * 1e3:.0f} mm" if missed is not None else "")
                        + " (IK preview); nothing moved",
                    )

    now = [(p0[0], q0[0]), (p0[1], q0[1])]

    def gap() -> float:
        d = float(np.linalg.norm(now[0][0] - now[1][0]))
        out["min_hand_distance_m"] = min(out["min_hand_distance_m"], d)
        return d

    gap()
    roll_tol = min(
        ROLL_TOLERANCE_DEG, float(relative_tolerance_deg) / 2.0 if keep_relative else 1e9
    )
    gain = [1.0, 1.0]
    turning = [bool(fix_roll), bool(fix_roll)]

    def finish_roll(i: int, q: np.ndarray) -> None:
        """Turn the rest of a hand's roll on its last joint (a solver that keeps
        only the approach axis leaves it free); a full-pose arm is already there."""
        a = arms[i]
        for _ in range(TURN_TRIES):
            residual = _roll_residual(q, now[i][1])
            if not turning[i] or abs(residual) <= roll_tol:
                return
            joints = _joints(ctx, a)
            if joints.size == 0:
                turning[i] = False
                notes.append(f"no joint reading, arm {a}'s wrist roll was left")
                return
            command = math.radians(residual) / gain[i]
            lo, hi = ROLL_LIMITS
            if not lo + LIMIT_MARGIN <= joints[-1] + command <= hi - LIMIT_MARGIN:
                turning[i] = False
                notes.append(f"arm {a}'s last joint has no room to turn {residual:+.0f} deg")
                return
            target_q = joints.copy()
            target_q[-1] += command
            try:
                why = _refusal(
                    ctx.tool(
                        "robot.move_to_joints",
                        joint_config={"positions": target_q.tolist()},
                        **arm(a),
                    )
                )
            except Exception as exc:
                why = str(exc) or type(exc).__name__
            out["roll_turns"] += 1
            if why:
                turning[i] = False
                notes.append(f"arm {a} wrist turn: {why}")
                return
            now[i] = here(a)
            gap()
            moved = math.radians(residual - _roll_residual(q, now[i][1]))
            if abs(command) > math.radians(1.0) and 0.5 < abs(moved / command) < 1.5:
                gain[i] = moved / command

    def report() -> None:
        out["left_pose"] = {"position": _vec(now[0][0]), "rotation": _qdict(now[0][1])}
        out["right_pose"] = {"position": _vec(now[1][0]), "rotation": _qdict(now[1][1])}
        out["left_residual_m"] = float(np.linalg.norm(now[0][0] - pl))
        out["right_residual_m"] = float(np.linalg.norm(now[1][0] - pr))

    def stopped(detail: str) -> Output:
        report()
        moved = [float(np.linalg.norm(now[i][0] - p0[i])) > MOVED for i in (0, 1)]
        if not any(moved):
            return finish("unreachable", f"{detail}; neither hand moved")
        return finish("diverged", f"{detail}; both arms stopped there")

    for k in range(1, n + 1):
        order = (0, 1) if k % 2 else (1, 0)
        plan = (way[0][k], way[1][k])
        for turn, i in enumerate(order):
            a = arms[i]
            p, q = way[i][k]
            try:
                why = _refusal(
                    ctx.tool(
                        "robot.go_to_pose_cartesian",
                        pose={"position": _vec(p), "rotation": _qdict(q)},
                        **arm(a),
                    )
                )
            except Exception as exc:  # a refused move is routed, not raised
                why = str(exc) or type(exc).__name__
            now[i] = here(a)
            gap()
            if not why:
                finish_roll(i, q)
            off = float(np.linalg.norm(now[i][0] - p))
            if off > float(reach_tolerance) or (why and off > REFUSED_TOLERANCE):
                return stopped(
                    f"arm {a} stopped {off * 1e3:.0f} mm from waypoint {k}/{n}"
                    + (f" ({why})" if why else "")
                )
            if turn == 0:
                lead = _pair_error(plan, now)[0]
                out["max_transient_error_m"] = max(out["max_transient_error_m"], lead)
        err_m, err_deg = _pair_error(plan, now)
        out["segment_errors"].append(
            {
                "segment": k,
                "lead_arm": arms[order[0]],
                "relative_error_m": err_m,
                "relative_error_deg": err_deg,
                "hand_distance_m": float(np.linalg.norm(now[0][0] - now[1][0])),
                "rotation_off_deg": [math.degrees(_angle(way[i][k][1], now[i][1])) for i in (0, 1)],
                "roll_off_deg": [_roll_residual(way[i][k][1], now[i][1]) for i in (0, 1)],
            }
        )
        out["max_relative_error_m"] = max(out["max_relative_error_m"], err_m)
        out["max_relative_error_deg"] = max(out["max_relative_error_deg"], err_deg)
        if keep_relative and (
            err_m > float(relative_tolerance) or err_deg > float(relative_tolerance_deg)
        ):
            return stopped(
                f"after segment {k}/{n} the hands' relative pose is {err_m * 1e3:.1f} mm /"
                f" {err_deg:.1f} deg off, over the {float(relative_tolerance) * 1e3:.0f} mm /"
                f" {float(relative_tolerance_deg):.1f} deg kept"
            )

    report()
    return finish(
        "reached",
        f"{n} segments; hands' relative pose off by at most {out['max_relative_error_m'] * 1e3:.1f} mm /"
        f" {out['max_relative_error_deg']:.1f} deg at segment ends",
    )
