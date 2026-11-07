# Adapted from yam-arcade (https://github.com/zzigak/yam-arcade),
# stations/jars/robot_demo.py (Demo.twist_on, Demo.stroke, Demo.regrasp: Q6_A / Q6_B,
# the stall rule), MIT, Copyright (c) 2026 zzigak; and from RoboWaiter
# (https://github.com/guimath/RoboWaiter), MujocoController.py (MujocoController.untwist,
# twist_cap: the stroke / open / roll back / close cycle repeated N times, then
# carry the cap away), MIT, Copyright (c) 2023 Guilhem Mathieux.
# Changes: a world-frame lid box and a top-down approach and grip in place of
# the simulator's known lid pose; the stroke as speed-bounded joint-6 steps
# through robot.move_to_joints (0.5 rad/s in place of yam-arcade's 1.6 rad/s;
# a tool roll about z with robot.go_to_pose for connectors that refuse joint
# moves) in place of a torque-capped velocity-feed-forward PD target; the
# stall read from joint 6's tracking lag (yam-arcade's 0.12 rad / 0.25 s rule)
# with the target held, not advanced, while it lags, in place of a motor torque
# cap; the lid-stopped-following test (simulator lid yaw) replaced by the
# gripper reading (jaws closing on nothing) and the tool centre's drift; one
# `direction` for tightening and loosening, a lift check after loosening, and
# routed refusals.
"""Turn a lid or cap by wrist ratchet: grip from above, roll joint 6 a stroke, open, roll back, re-grip."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

Q6_A, Q6_B = -1.9, 1.9
"""Joint 6 at the two ends of a stroke [rad]: yam-arcade's Q6_A / Q6_B, a
218 deg stroke inside the YAM wrist's +-2.094 rad range."""

TIGHTEN_SIGN = 1
"""Which way joint 6 turns to tighten (+1: from Q6_A to Q6_B). yam-arcade's
YAM turns -1.9 -> +1.9 to screw a right-hand lid on, clockwise seen from above."""

STROKES = {"tighten": 4, "loosen": 5}
"""Strokes when the caller gives none: yam-arcade's twist_on loops 4 times,
RoboWaiter's untwist 5 times."""

STEP_RAD = math.pi / 40
"""Largest joint-6 increment per command [rad], as pouring-by-wrist-roll."""

MAX_SPEED = 0.5
"""Joint-6 speed bound [rad/s]: each step takes at least STEP_RAD / MAX_SPEED.
yam-arcade strokes at 1.6 rad/s in simulation; a real YAM station's supervisor
trips at 0.75 rad/s measured joint speed."""

STALL_LAG = 0.12
"""Joint 6 this far behind its step's target reads as resistance [rad]:
yam-arcade's stall lag (and its lid-stopped-following bound)."""

STALL_S = 0.25
"""How long the lagging joint is held at its target, not advanced, before the
turn is called stopped [s]: yam-arcade's stall time."""

STALL_SPEED = 0.05
"""Joint-6 speed under which a lagging wrist is not turning [rad/s]:
yam-arcade's stall speed."""

MIN_TURN = 0.12
"""A tightening stall after less than this much turn in all is no seat [rad]
(already tight, cross-threaded or blocked): not tuned, equal to STALL_LAG."""

TRACK_TOLERANCE = 0.075
"""How far joint 6 may lag a free (open-hand) roll step [rad]: a real YAM
station's single-arm settle tolerance."""

EMPTY_GRIP = 0.10
"""A gripper reading (gripper_fraction) under this is jaws closed on nothing.
Measured in the YAM station twin: 0.33 on a 30 mm cube, 0.05 empty."""

GRIP_DEPTH = 0.010
"""Tool centre below the lid's top when gripping [m]: closed linear_4310
fingertips reach ~10 mm below the tool centre, so the pads take the top 20 mm."""

APPROACH_HEIGHT = 0.08
"""Pregrasp over the grip point [m], as yam-arcade's pick (TABLE_Z + 0.08)."""

LIFT_HEIGHT = 0.025
"""Lift after loosening [m]: yam-arcade's release rise."""

LIFT_TOLERANCE = 0.010
"""How far short of lift_height the hand may stop and still count as lifted [m]
(a real YAM's 12 mm per-move bound is close to the lift itself)."""

RETREAT_HEIGHT = 0.025
"""Rise after letting go [m]: yam-arcade's release rise."""

DRIFT_TOLERANCE = 0.02
"""How far the tool centre may wander sideways from the grip point during a
stroke [m], as pouring-by-wrist-roll."""

REACH_TOLERANCE = 0.02
"""How far the hand may end from the pregrasp or grip point [m]."""

ROLL_LIMITS = (-2.0944, 2.0944)
"""Joint 6's range [rad] when the caller states none: the i2rt YAM's (+-120 deg)."""

LIMIT_MARGIN = 0.05
"""Kept clear of each joint-6 limit [rad]."""


class Output(TypedDict):
    route: str
    detail: str
    direction: str
    strokes_done: int
    turn_per_stroke: list[float]
    total_turn: float
    seated: bool
    q6_trajectory: list[float]
    grip_trajectory: list[float]
    trajectory_phase: list[str]
    grip_at_close: list[float]
    hand_drift: float
    lift_rise: float
    holding: bool
    rolled_with: str


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _wxyz(q: dict[str, float]) -> np.ndarray:
    return np.array([float(q[k]) for k in ("w", "x", "y", "z")])


def _quat(q: np.ndarray) -> dict[str, float]:
    return dict(zip(("w", "x", "y", "z"), map(float, q), strict=True))


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _half_height(obb: dict[str, Any]) -> float:
    """The box's vertical half-extent in the world [m]."""
    e = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    w, x, y, z = _wxyz(obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    row = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
    return float(np.sum(np.abs(row) * e))


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


def _positions(joints: Any) -> np.ndarray:
    if isinstance(joints, dict):
        joints = joints.get("positions")
    return np.asarray(joints if joints is not None else [], dtype=np.float64).reshape(-1)


def run(
    ctx: NodeContext,
    lid_obb: dict[str, Any],
    direction: str = "tighten",
    strokes: int = 0,
    stroke_limits: list[float] | None = None,
    tighten_sign: int = TIGHTEN_SIGN,
    pitch: float = 0.0,
    grip_depth: float = GRIP_DEPTH,
    approach_height: float = APPROACH_HEIGHT,
    rotation: dict[str, float] | None = None,
    lift_check: bool = True,
    lift_height: float = LIFT_HEIGHT,
    retreat_height: float = RETREAT_HEIGHT,
    roll_with: str = "auto",
    roll_limits: list[float] | None = None,
    step_rad: float = STEP_RAD,
    max_speed: float = MAX_SPEED,
    stall_lag: float = STALL_LAG,
    stall_s: float = STALL_S,
    min_turn: float = MIN_TURN,
    track_tolerance: float = TRACK_TOLERANCE,
    empty_grip: float = EMPTY_GRIP,
    drift_tolerance: float = DRIFT_TOLERANCE,
    reach_tolerance: float = REACH_TOLERANCE,
    arm_id: int = 0,
    tolerance: float | None = None,
    max_steps: int | None = None,
) -> Output:
    if direction not in STROKES:
        raise ValueError(f"direction must be tighten or loosen, got {direction!r}")
    if roll_with not in ("auto", "joints", "pose"):
        raise ValueError(f"roll_with must be auto, joints or pose, got {roll_with!r}")
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "direction": direction,
        "strokes_done": 0,
        "turn_per_stroke": [],
        "total_turn": 0.0,
        "seated": False,
        "q6_trajectory": [],
        "grip_trajectory": [],
        "trajectory_phase": [],
        "grip_at_close": [],
        "hand_drift": 0.0,
        "lift_rise": 0.0,
        "holding": False,
        "rolled_with": "",
    }
    on_arm: dict[str, Any] = {"arm_id": arm_id} if arm_id else {}
    n_strokes = int(strokes) if int(strokes) > 0 else STROKES[direction]
    a, b = sorted(float(v) for v in (stroke_limits or (Q6_A, Q6_B)))
    sign = (1 if tighten_sign >= 0 else -1) * (1 if direction == "tighten" else -1)
    q_from, q_to = (a, b) if sign > 0 else (b, a)
    step = max(1.0e-3, float(step_rad))
    speed = max(1.0e-3, float(max_speed))
    grip = [-1.0]

    def ee() -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **on_arm))

    def joints() -> np.ndarray:
        arms = ctx.tool("robot.get_observation").get("arms") or []
        if arm_id >= len(arms):
            return np.zeros(0)
        grip[0] = float(arms[arm_id].get("gripper_fraction", -1.0))
        return _positions(arms[arm_id].get("joint_state"))

    def record(q: np.ndarray, phase: str) -> None:
        if q.size:
            out["q6_trajectory"].append(float(q[-1]))
            out["grip_trajectory"].append(grip[0])
            out["trajectory_phase"].append(phase)

    def move(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        kwargs: dict[str, Any] = {"pose": {"position": _vec(position), "rotation": rot}, **on_arm}
        try:
            return _refusal(ctx.tool(tool, **kwargs))
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc) or type(exc).__name__

    def gripper(close: bool) -> str:
        try:
            ctx.tool("robot.close_gripper" if close else "robot.open_gripper", **on_arm)
        except Exception as exc:
            return str(exc) or type(exc).__name__
        return ""

    def let_go() -> str:
        """Open and rise retreat_height; why it failed, or ""."""
        why = gripper(False)
        if why:
            return f"open: {why}"
        out["holding"] = False
        try:
            at = ee()
        except Exception as exc:
            return f"no pose to retreat from: {exc}"
        up = _xyz(at) + np.array([0.0, 0.0, float(retreat_height)])
        why = move("robot.go_to_pose_cartesian", up, at["rotation"])
        return f"retreat: {why}" if why else ""

    notes: list[str] = []

    def finish(route: str, detail: str, release: bool = False) -> Output:
        detail = "; ".join([detail, *notes]) if detail else "; ".join(notes)
        if release:
            why = let_go()
            if why:
                detail = f"{detail}; {why}" if detail else why
        out["route"] = route
        out["detail"] = detail
        return out

    # 1. The grip point: over the lid's centre, grip_depth under its top; the
    # pregrasp approach_height over it.
    centre = np.array([float(lid_obb["center"][k]) for k in ("x", "y", "z")])
    top = centre[2] + _half_height(lid_obb)
    grip_at = np.array([centre[0], centre[1], top - float(grip_depth)])
    pre = grip_at + np.array([0.0, 0.0, float(approach_height)])

    candidates: list[dict[str, float]] = []
    if rotation:
        candidates.append(rotation)
    else:
        try:
            frame = ctx.tool("robot.grasp_frame", close_heading_deg=0.0, **on_arm)
        except Exception as exc:
            logger.info("turning-by-wrist-ratchet: no grasp frame (%s); keeping the hand's", exc)
            frame = None
        if isinstance(frame, dict):
            candidates += [frame[k] for k in ("rotation", "rotation_flipped") if frame.get(k)]
        if not candidates:
            candidates.append(ee()["rotation"])

    why = gripper(False)
    if why:
        return finish("unreachable", f"open before the approach: {why}")
    reasons = []
    for rot in candidates:
        why = move("robot.go_to_pose", pre, rot)
        off = float(np.linalg.norm(_xyz(ee()) - pre))
        if not why and off <= float(reach_tolerance):
            break
        reasons.append(why or f"ended {off * 1e3:.0f} mm off")
    else:
        return finish("unreachable", "pregrasp over the lid: " + "; ".join(reasons))

    # 2. Joint 6 to the stroke's start, hand open, over the lid.
    q0 = joints()
    if q0.size == 0:
        return finish("unreachable", f"no joint reading for arm {arm_id}")
    lo, hi = (float(v) for v in (roll_limits or ROLL_LIMITS))
    if a < lo + LIMIT_MARGIN - 1.0e-9 or b > hi - LIMIT_MARGIN + 1.0e-9:
        return finish(
            "unreachable",
            f"stroke {a:.3f}..{b:.3f} rad does not fit joint 6's limits {lo:.3f}..{hi:.3f} "
            f"less a {LIMIT_MARGIN} rad margin",
        )
    base_rot = _wxyz(ee()["rotation"])
    base_q6 = float(q0[-1])
    mode = ["pose" if roll_with == "pose" else "joints"]
    pose_sign = [1.0]
    learned = [False]
    first = [True]
    record(q0, "start")

    def joint_move(target: np.ndarray) -> str:
        kwargs: dict[str, Any] = {"joint_config": {"positions": [float(v) for v in target]}}
        if tolerance is not None:
            kwargs["tolerance"] = float(tolerance)
        if max_steps is not None:
            kwargs["max_steps"] = int(max_steps)
        kwargs.update(on_arm)
        try:
            return _refusal(ctx.tool("robot.move_to_joints", **kwargs))
        except Exception as exc:
            return str(exc) or type(exc).__name__

    def pose_rot(q6: float) -> dict[str, float]:
        half = 0.5 * pose_sign[0] * (q6 - base_q6)
        return _quat(_mul(base_rot, np.array([math.cos(half), 0.0, 0.0, math.sin(half)])))

    def command(target: np.ndarray, xyz: np.ndarray, before: np.ndarray) -> str:
        """One step to ``target`` (joints) / ``xyz`` (pose roll)."""
        why = ""
        if mode[0] == "joints":
            why = joint_move(target)
            if why and roll_with == "auto" and first[0]:
                now = joints()
                if now.size == before.size and float(np.max(np.abs(now - before))) < 1.0e-3:
                    logger.info(
                        "turning-by-wrist-ratchet: joint move refused (%s); rolling by pose", why
                    )
                    mode[0], why = "pose", ""
            first[0] = False
            if mode[0] == "joints":
                return why
        why = move("robot.go_to_pose", xyz, pose_rot(float(target[-1])))
        if why or learned[0]:
            return why
        now = joints()
        asked = float(target[-1] - before[-1])
        moved = float(now[-1] - before[-1]) if now.size == before.size else 0.0
        if abs(moved) < 0.3 * abs(asked):
            return f"a roll about the tool axis moved joint {before.size} only {moved:.3f} rad"
        learned[0] = True
        if moved * asked < 0:
            pose_sign[0] = -1.0
            return move("robot.go_to_pose", xyz, pose_rot(float(target[-1])))
        return ""

    def roll(
        goal: float,
        phase: str,
        xyz0: np.ndarray,
        xyz1: np.ndarray,
        others1: np.ndarray | None = None,
        stroke: bool = False,
    ) -> tuple[str, str]:
        """Step joint 6 to ``goal``; (kind, why) of a stop, kind "" when it got there.

        Kinds: refused, lag (a free roll off its step), stall, slip, drift.
        """
        start = joints()
        if start.size == 0:
            return "refused", "no joint reading"
        origin = float(start[-1])
        travel = goal - origin
        n = max(1, math.ceil(abs(travel) / step))
        period = abs(travel) / n / speed
        s = 1.0 if travel >= 0 else -1.0
        others0 = start[:-1].copy()
        ends = others0 if others1 is None else np.asarray(others1, dtype=np.float64)
        holds = max(1, math.ceil(float(stall_s) / max(period, 1.0e-3)))
        now = start
        for i in range(1, n + 1):
            frac = i / n
            target = np.append(others0 + frac * (ends - others0), origin + travel * frac)
            xyz = xyz0 + frac * (xyz1 - xyz0)
            began = time.monotonic()
            why = command(target, xyz, now)
            now = joints()
            record(now, phase)
            wait = period - (time.monotonic() - began)
            if wait > 0:
                time.sleep(wait)
            if now.size != start.size:
                return "refused", "no joint reading"
            lag = s * float(target[-1] - now[-1])
            if why and not (stroke and lag > float(stall_lag)):
                return "refused", why
            if not stroke:
                if abs(lag) > float(track_tolerance):
                    return "lag", (
                        f"joint 6 {abs(lag):.3f} rad off the step's target "
                        f"(bound {float(track_tolerance):.3f})"
                    )
                continue
            if 0.0 <= grip[0] < float(empty_grip):
                return (
                    "slip",
                    f"the gripper reading fell to {grip[0]:.2f} (empty under {empty_grip})",
                )
            hand = _xyz(ee())
            out["hand_drift"] = max(
                out["hand_drift"], float(np.linalg.norm(hand[:2] - grip_at[:2]))
            )
            if out["hand_drift"] > float(drift_tolerance):
                return "drift", (
                    f"the tool centre drifted {out['hand_drift'] * 1e3:.0f} mm from the grip point "
                    f"(bound {float(drift_tolerance) * 1e3:.0f})"
                )
            if lag <= float(stall_lag):
                continue
            # Lagging: hold the target, do not advance it, and see whether it turns.
            before = float(now[-1])
            for _ in range(holds):
                began = time.monotonic()
                command(target, xyz, now)
                now = joints()
                record(now, phase + "_hold")
                wait = period - (time.monotonic() - began)
                if wait > 0:
                    time.sleep(wait)
            if now.size != start.size:
                return "refused", "no joint reading"
            progress = s * float(now[-1] - before)
            lag = s * float(target[-1] - now[-1])
            if lag > float(stall_lag) and progress < float(STALL_SPEED) * holds * period:
                relieve = target.copy()
                relieve[-1] = now[-1]
                command(relieve, xyz, now)
                return "stall", (
                    f"joint 6 stood {lag:.3f} rad behind its target and turned {progress:.3f} rad "
                    f"in {holds * period:.2f} s"
                )
        return "", ""

    kind, why = roll(q_from, "to_start", pre, pre)
    out["rolled_with"] = mode[0]
    if kind:
        return finish(
            "unreachable", f"turning joint 6 to the stroke start {q_from:+.2f} rad: {why}"
        )

    # 3. Down to the grip point, keeping the turned wrist, and close.
    why = move("robot.go_to_pose_cartesian", grip_at, ee()["rotation"])
    off = float(np.linalg.norm(_xyz(ee()) - grip_at))
    if why or off > float(reach_tolerance):
        detail = f"descent to the grip point: {why or 'ended off'}; the hand stands {off * 1e3:.0f} mm from it"
        move("robot.go_to_pose_cartesian", pre, ee()["rotation"])
        return finish("unreachable", detail)
    q = joints()
    record(q, "gripped")
    if q.size and abs(float(q[-1]) - q_from) > float(track_tolerance):
        kind, why = roll(q_from, "to_start", grip_at, grip_at)
        if kind:
            return finish("unreachable", f"joint 6 left the stroke start on the descent: {why}")

    def close() -> str:
        why = gripper(True)
        q = joints()
        record(q, "close")
        out["grip_at_close"].append(grip[0])
        if why:
            return f"close: {why}"
        out["holding"] = True
        if 0.0 <= grip[0] < float(empty_grip):
            return f"the jaws closed on nothing (gripper reading {grip[0]:.2f})"
        return ""

    why = close()
    if why:
        return finish("slipped", f"grip: {why}", release=True)

    # 4. The strokes: roll, open, roll back, re-grip.
    rise = -1.0 if direction == "tighten" else 1.0
    turned = 0.0
    for k in range(1, n_strokes + 1):
        q_start = joints()
        at = _xyz(ee())
        z_end = at[2] + rise * float(pitch) * abs(q_to - float(q_start[-1])) / (2.0 * math.pi)
        end_xyz = np.array([grip_at[0], grip_at[1], z_end])
        others1 = None
        if float(pitch) > 0.0 and mode[0] == "joints":
            try:
                reply = ctx.tool(
                    "robot.solve_ik",
                    pose={"position": _vec(end_xyz), "rotation": ee()["rotation"]},
                    **on_arm,
                )
                sol = _positions((reply or {}).get("joint_config"))
                if sol.size == q_start.size and reply.get("success", True) is not False:
                    others1 = sol[:-1]
            except Exception as exc:
                logger.info("turning-by-wrist-ratchet: no IK for the thread's pitch (%s)", exc)
            if others1 is None and not notes:
                notes.append("no IK for the pitch: the hand held its height")
        kind, why = roll(
            q_to,
            f"stroke{k}",
            np.array([grip_at[0], grip_at[1], at[2]]),
            end_xyz,
            others1,
            stroke=True,
        )
        q_end = joints()
        this = (1.0 if q_to >= q_from else -1.0) * float(q_end[-1] - q_start[-1])
        out["turn_per_stroke"].append(this)
        turned += this
        out["total_turn"] = turned
        out["strokes_done"] = k
        if kind == "stall":
            if direction == "tighten" and turned >= float(min_turn):
                out["seated"] = True
                return finish(
                    "turned",
                    f"seated: stroke {k} stopped after {this:.3f} rad ({why}); "
                    f"{turned:.3f} rad turned in all",
                    release=True,
                )
            return finish("stalled", f"stroke {k}: the turn stopped: {why}", release=True)
        if kind == "slip":
            return finish("slipped", f"stroke {k}: {why}", release=True)
        if kind:
            return finish("stalled", f"stroke {k}: {why}", release=True)
        if k == n_strokes:
            break
        why = gripper(False)
        if why:
            return finish("stalled", f"open after stroke {k}: {why}")
        out["holding"] = False
        here = _xyz(ee())
        kind, why = roll(q_from, f"back{k}", here, here)
        if kind:
            return finish("stalled", f"roll back after stroke {k}: {why}", release=True)
        why = close()
        if why:
            return finish("slipped", f"re-grip after stroke {k}: {why}", release=True)

    # 5. The end: tightening lets go; loosening lifts to see whether the cap came free.
    if direction == "tighten":
        return finish(
            "turned", f"{n_strokes} strokes, {turned:.3f} rad turned; no seat felt", release=True
        )
    if not lift_check:
        return finish("turned", f"{n_strokes} strokes, {turned:.3f} rad turned; not lift-checked")
    at = ee()
    z0 = _xyz(at)[2]
    up = _xyz(at) + np.array([0.0, 0.0, float(lift_height)])
    why = move("robot.go_to_pose_cartesian", up, at["rotation"])
    out["lift_rise"] = float(_xyz(ee())[2] - z0)
    q = joints()
    record(q, "lift")
    if why or out["lift_rise"] < float(lift_height) - float(LIFT_TOLERANCE):
        return finish(
            "stalled",
            f"lift check: the hand rose {out['lift_rise'] * 1e3:.0f} of {float(lift_height) * 1e3:.0f} mm"
            + (f" ({why})" if why else "")
            + "; the cap is still on its thread",
            release=True,
        )
    if 0.0 <= grip[0] < float(empty_grip):
        return finish(
            "slipped",
            f"lift check: the hand rose but the jaws closed on nothing (gripper reading {grip[0]:.2f})",
            release=True,
        )
    return finish(
        "turned",
        f"{n_strokes} strokes, {turned:.3f} rad turned; lifted {out['lift_rise'] * 1e3:.0f} mm "
        "with the jaws still holding",
    )
