# Adapted from ShapeForce (https://github.com/nolann030628/ShapeForce, commit
# 1bc2aa8c4ee66622f0a597b2112a61cfbea2a345),
# third_party/utils_repo/peg_insertion/force_insert.py, ForceInserter._phase_spin_search,
# MIT License, Copyright (c) 2021 SowmiyaNarayanan G (the repository's LICENSE; a
# copy is LICENSE.shapeforce beside this skill).
# Changes: ShapeForce holds the part at a fixed height on its socket and turns the
# wrist 90 deg in 15 sub-steps on an xArm, stopping when its marker-tracked soft
# wrist reads a torque over 20, else opens, lifts 50 mm, turns back, re-grips and
# turns again (a ratchet, without end). Here the turn is a search for the yaw at
# which a keyed or polygonal part drops into its socket: the part is pressed
# lightly onto the socket (the command led press_depth under the resting height,
# the tracking lag acting as the push, since a YAM has no force sensor), the wrist
# is turned in the same 6 deg sub-steps but alternately either side of the start
# (0, +6, -6, +12, ... within yaw_range_deg, so the nearest yaws come first and
# the YAM's +-120 deg last joint never runs out), the press is renewed after every
# turn, and the search stops on a drop (the hand sinks drop_depth under the
# resting height) or when a turn falls short of its command by jam_deg (the
# torque stop, read from the wrist's own tracking instead of a wrist sensor). The
# regrasp ratchet is not ported (turning-by-wrist-ratchet covers it). Refused
# moves are routed. The last-joint turn with a learnt gain follows
# skills/seating-a-piece-by-wiggle-search (a YAM station twin's IK leaves the
# wrist roll free).
"""Press a held keyed part lightly onto its socket and turn it either way until it drops in."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

YAW_STEP_DEG = 6.0
"""Turn per try [deg]: ShapeForce's 90 deg stroke in 15 sub-steps."""

YAW_RANGE_DEG = 45.0
"""Largest turn either side of the start [deg]: a four-fold part (a square peg, a
2x2 brick) matches within +-45 deg; raise it to 90 for a two-fold part, 180 for
one-fold (the YAM's last joint turns +-120 deg; ROLL_LIMITS)."""

PRESS_DEPTH = 0.001
"""How far past the command at which the socket was felt the press is commanded
[m]. On the socket the push is about the servo stiffness times this plus the
contact lag; in the socket the hand follows its command, so press plus contact lag
bounds how far a drop can be seen (2-3 mm here). Kept low because the push must stay
under what the grip holds: in a YAM station twin (about 2.2 N per mm of lag) a 3 mm
press with 1 mm descent steps pushed about 10 N and the held cube slid through the
jaws on the socket's rim, which read as a drop."""

DROP_DEPTH = 0.002
"""How far under the resting height the hand must sink, following its command, for
the part to be in [m]: a press that only took up slack (the hand still held,
its lag over the free baseline above half CONTACT_LAG) is not a drop."""

JAM_DEG = 3.0
"""Shortfall of a turn against its command that reads as the rotation blocked
[deg] (ShapeForce's torque stop). A choice: over the 0.3 deg the turn is
re-commanded to, and under one step."""

START_LIFT = 0.003
MAX_DESCENT = 0.010
"""The socket is felt for from START_LIFT over the hand's start down to MAX_DESCENT
under it [m]: lifting first gives free-motion samples for the lag baseline and the
command at which the socket holds the hand, whether the part already rests on it
(after placing-with-contact-descent or pressing-in-with-guarded-push) or hovers
a few millimetres over it. 10 mm covers a real YAM standing 5.4 mm over its aim in
free air (the lifted hand rises that much off the socket) plus the contact lag."""

DESCENT_STEP = 0.0005
CONTACT_LAG = 0.0015
"""Descent step [m], and the lag over the free-motion baseline that marks the
socket [m] (half a real YAM station's 3 mm contact lag). The half-millimetre step
keeps the push at contact, and so on the rim, within a step of CONTACT_LAG."""

TURN_TRIES = 3
YAW_TOLERANCE_DEG = 0.3
ROLL_LIMITS = (-2.0944, 2.0944)
LIMIT_MARGIN = 0.05
"""Last-joint turn: re-commanded up to TURN_TRIES times until within
YAW_TOLERANCE_DEG; the YAM's last joint range [rad] and the margin kept from it
(skills/seating-a-piece-by-wiggle-search)."""

STEP_TOLERANCE = 0.0002
"""Arrival ball asked of each press and descent step [m] (go_to_pose's
``tolerance``); well under a step, or a servo calls the step reached unmoved."""

SETTLE_QUIET = 0.0003
SETTLE_PERIOD = 0.3
SETTLE_TIMEOUT = 3.0
"""The hand has settled when two reads SETTLE_PERIOD apart [s] differ by less than
SETTLE_QUIET [m], or after SETTLE_TIMEOUT [s] (a real YAM station's still())."""

CONFIRM_PROBE = 0.006
CONFIRM_STEP = 0.0005
CONFIRM_LAG = 0.001
"""Drop confirmation: after a drop the hand moves sideways CONFIRM_PROBE [m] in
CONFIRM_STEP steps; a part in its socket meets a wall within the socket's total
clearance (3 mm for a 30 mm part in a 33 mm hole), and the hand then lags its
command by over CONFIRM_LAG [m]; a part that only slid in the jaws, or walked off
on the rim, rides the rim freely. The probe must be longer than the clearance plus
CONFIRM_LAG. Choices, checked in a station twin (see SKILL.md)."""

SEAT_CHECK = 0.003
"""Before the sideways probe, the hand is commanded this much deeper [m]. A part in
its socket goes down with it (the hand follows within half CONTACT_LAG); a part
caught by its lip on the socket's edge, or resting on the rim after sliding in the
jaws, holds the hand back. In a station twin a cube caught 0.4 mm into the socket
passed the sideways probe alone. Needs a socket deeper than the drop's sink plus
this; 0 skips it."""

STOP_MARGIN = 0.002
"""Jaw gap within this of the closed stop reads as nothing held [m]
(pressing-a-brick-onto-studs' test)."""

REFUSALS = ("unreachable", "bad_speed_scale")


class Output(TypedDict):
    route: str
    detail: str
    yaw_found_deg: float
    yaw_read_deg: float
    sink: float
    confirmed: bool
    confirm_travel: float
    seat_follow: float
    rest_z: float
    tries: int


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(v: Any) -> np.ndarray:
    return np.array([float(v["x"]), float(v["y"]), float(v["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat(r: np.ndarray) -> dict[str, float]:
    t = float(np.trace(r))
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    return dict(zip(("w", "x", "y", "z"), (float(v) for v in q), strict=True))


def _rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw_between(r_from: np.ndarray, r_to: np.ndarray) -> float:
    """How far [deg] the hand turned about the vertical from ``r_from`` to ``r_to``."""
    rel = r_to @ r_from.T
    return math.degrees(math.atan2(float(rel[1, 0]), float(rel[0, 0])))


def _refusal(reply: Any) -> tuple[str, str]:
    if not isinstance(reply, dict):
        return "ok", ""
    outcome = reply.get("outcome")
    if outcome in REFUSALS:
        return "refused", f"{outcome}: {reply.get('reason', '')}".rstrip(": ")
    if outcome not in (None, "reached") or reply.get("reached") is False:
        return "short", str(outcome or reply.get("reason") or "not reached")
    if reply.get("success") is False:
        return "refused", str(reply.get("reason") or reply.get("message") or "failed")
    return "ok", ""


def _joint_state(obs: Any, arm_id: int) -> dict[str, Any] | None:
    if not isinstance(obs, dict):
        return None
    arms = obs.get("arms")
    if isinstance(arms, list) and len(arms) > arm_id and isinstance(arms[arm_id], dict):
        js = arms[arm_id].get("joint_state")
        if isinstance(js, dict):
            return js
    js = obs.get("joint_state")
    return js if isinstance(js, dict) else None


def _jaws_empty(ctx: NodeContext, on_arm: dict[str, Any]) -> float | None:
    """The jaw gap [m] when it reads at the closed stop (the part is gone), else None
    (holding, or no gripper reading)."""
    try:
        fraction = float(ctx.tool("robot.get_gripper", **on_arm)["position"])
        grip = ctx.tool("robot.describe_gripper", **on_arm)
    except Exception as exc:
        logger.info("yaw-search: gripper not read (%s)", exc)
        return None
    fit = grip.get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or grip.get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)
    if at_open <= at_closed:
        return None
    width = at_closed + (at_open - at_closed) * fraction
    return width if width <= at_closed + STOP_MARGIN else None


def yaw_order(yaw_range: float, yaw_step: float) -> list[float]:
    """0, +step, -step, +2 step, ... within +-yaw_range [deg]: the nearest yaws first."""
    order = [0.0]
    if yaw_step <= 0.0:
        return order
    k = 1
    while k * yaw_step <= yaw_range + 1e-9:
        order += [k * yaw_step, -k * yaw_step]
        k += 1
    return order


class _Arm:
    def __init__(self, ctx: NodeContext, arm_id: int, tolerance: float, max_steps: int) -> None:
        self.ctx = ctx
        self.arm_id = int(arm_id)
        self.on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
        self.knobs: dict[str, Any] = {}
        if tolerance > 0:
            self.knobs["tolerance"] = float(tolerance)
        if max_steps > 0:
            self.knobs["max_steps"] = int(max_steps)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        pose = _pose(self.ctx.tool("robot.get_ee_pose", **self.on_arm))
        return _xyz(pose["position"]), _rot(pose["rotation"])

    def move(self, p: np.ndarray, r: np.ndarray) -> tuple[str, str]:
        pose = {"position": _vec(p), "rotation": _quat(r)}
        try:
            try:
                reply = self.ctx.tool("robot.go_to_pose", pose=pose, **self.knobs, **self.on_arm)
            except TypeError as exc:
                if not self.knobs or "keyword" not in str(exc):
                    raise
                logger.info("yaw-search: go_to_pose takes no %s (%s)", sorted(self.knobs), exc)
                self.knobs = {}
                reply = self.ctx.tool("robot.go_to_pose", pose=pose, **self.on_arm)
        except Exception as exc:  # a station raises where a simulator routes
            why = str(exc)
            return ("short" if "did not reach" in why else "refused"), why
        return _refusal(reply)

    def still(self, quiet: float, period: float, timeout: float) -> tuple[np.ndarray, np.ndarray]:
        p, r = self.read()
        if period <= 0:
            return p, r
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            time.sleep(period)
            q, s = self.read()
            if float(np.linalg.norm(q - p)) < quiet:
                return q, s
            p, r = q, s
        return p, r

    def joints(self) -> np.ndarray | None:
        try:
            js = _joint_state(self.ctx.tool("robot.get_observation"), self.arm_id)
        except Exception as exc:
            logger.info("yaw-search: no joint reading (%s)", exc)
            return None
        if js is None or js.get("positions") is None:
            return None
        q = np.asarray(js["positions"], dtype=np.float64).reshape(-1)
        return q if q.size else None


def run(
    ctx: NodeContext,
    yaw_range_deg: float = YAW_RANGE_DEG,
    yaw_step_deg: float = YAW_STEP_DEG,
    press_depth: float = PRESS_DEPTH,
    drop_depth: float = DROP_DEPTH,
    jam_deg: float = JAM_DEG,
    start_lift: float = START_LIFT,
    max_descent: float = MAX_DESCENT,
    descent_step: float = DESCENT_STEP,
    contact_lag: float = CONTACT_LAG,
    return_on_fail: bool = True,
    check_holding: bool = True,
    confirm: bool = True,
    confirm_direction: dict[str, float] | None = None,
    confirm_probe: float = CONFIRM_PROBE,
    confirm_lag: float = CONFIRM_LAG,
    seat_check: float = SEAT_CHECK,
    roll_limits: list[float] | None = None,
    step_tolerance: float = STEP_TOLERANCE,
    step_max_steps: int = 0,
    settle_quiet: float = SETTLE_QUIET,
    settle_period: float = SETTLE_PERIOD,
    settle_timeout: float = SETTLE_TIMEOUT,
    arm_id: int = 0,
) -> Output:
    if press_depth <= 0 or drop_depth <= 0 or start_lift < 0 or max_descent < 0:
        raise ValueError(
            "press_depth and drop_depth must be positive, start_lift and max_descent not negative"
        )
    if yaw_step_deg <= 0 or yaw_range_deg < 0 or jam_deg <= 0:
        raise ValueError("yaw_step_deg and jam_deg must be positive, yaw_range_deg not negative")
    arm = _Arm(ctx, arm_id, step_tolerance, step_max_steps)
    p0, r0 = arm.read()
    out: Output = {
        "route": "blocked",
        "detail": "",
        "yaw_found_deg": 0.0,
        "yaw_read_deg": 0.0,
        "sink": 0.0,
        "confirmed": False,
        "confirm_travel": -1.0,
        "seat_follow": -1.0,
        "rest_z": float(p0[2]),
        "tries": 0,
    }
    notes: list[str] = []
    state = {"yaw": 0.0, "gain": -1.0 if r0[2, 2] < 0 else 1.0}

    def finish(route: str, detail: str) -> Output:
        out["route"] = route
        out["yaw_read_deg"] = state["yaw"]
        out["detail"] = detail + ("; " + "; ".join(notes) if notes else "")
        return out

    def settle() -> tuple[np.ndarray, np.ndarray]:
        return arm.still(float(settle_quiet), float(settle_period), float(settle_timeout))

    def turn(target: float) -> str:
        """Turn the hand to ``target`` deg of yaw from r0 where it stands: the last
        joint (read back, the gain learnt), else a rotation command."""
        for _ in range(TURN_TRIES):
            p, r = arm.read()
            now = _yaw_between(r0, r)
            state["yaw"] = now
            residual = target - now
            if abs(residual) <= YAW_TOLERANCE_DEG:
                return ""
            q = arm.joints()
            if q is None:
                kind, why = arm.move(p, _rz(target) @ r0)
                _, r2 = arm.read()
                state["yaw"] = _yaw_between(r0, r2)
                return "" if kind != "refused" else f"wrist turn: {why}"
            lo, hi = (float(v) for v in (roll_limits or ROLL_LIMITS))
            command = math.radians(residual) / state["gain"]
            if not lo + LIMIT_MARGIN <= q[-1] + command <= hi - LIMIT_MARGIN:
                return f"the last joint has no room to turn {residual:+.1f} deg"
            goal = q.copy()
            goal[-1] += command
            try:
                kind, why = _refusal(
                    ctx.tool(
                        "robot.move_to_joints",
                        joint_config={"positions": goal.tolist()},
                        **arm.on_arm,
                    )
                )
            except Exception as exc:
                kind, why = "refused", str(exc)
            if kind == "refused":
                return f"wrist turn: {why}"
            _, r2 = arm.read()
            moved = math.radians(_yaw_between(r, r2))
            if abs(command) > math.radians(0.3) and 0.5 < abs(moved / command) < 1.5:
                state["gain"] = moved / command
            state["yaw"] = _yaw_between(r0, r2)
        return ""

    # Feel for the socket from start_lift over the start: the free steps give the lag
    # baseline, the stop the resting height and the command that met the socket.
    top = p0 + np.array([0.0, 0.0, float(start_lift)])
    kind, why = arm.move(top, r0)
    if kind == "refused":
        return finish("blocked", f"the lift before feeling for the socket was refused: {why}")
    p, _ = settle()
    lags = [float(p[2] - top[2])]
    length = float(start_lift) + float(max_descent)
    n = math.ceil(length / float(descent_step) - 1e-9)
    rest = contact_cmd = None
    for i in range(1, n + 1):
        cmd = top - np.array([0.0, 0.0, min(i * float(descent_step), length)])
        kind, why = arm.move(cmd, r0)
        if kind == "refused":
            return finish("blocked", f"descent step {i} refused: {why}")
        p, _ = settle()
        lag = float(p[2] - cmd[2])
        over = lag - float(np.median(lags[-3:]))
        if over > float(contact_lag):
            rest, contact_cmd = p, cmd
            break
        # Only clearly free steps join the baseline: a lag building up step by step
        # on the socket must not drag the baseline up with it.
        if over < float(contact_lag) / 3:
            lags.append(lag)
    if rest is None or contact_cmd is None:
        under = float(p0[2] - p[2])
        return finish(
            "blocked",
            f"no socket felt from {start_lift * 1e3:.0f} mm over the start to "
            f"{max_descent * 1e3:.0f} mm under it (the hand went {under * 1e3:.1f} mm under "
            "the start freely: no socket there, or the part already in); nothing turned",
        )
    base = float(np.median(lags[-3:]))
    out["rest_z"] = float(rest[2])
    press_at = np.array([rest[0], rest[1], contact_cmd[2] - float(press_depth)])

    def press(r: np.ndarray) -> tuple[str, np.ndarray, float]:
        """(refusal, settled hand, its lag over the free baseline) after pressing."""
        kind, why = arm.move(press_at, r)
        if kind == "refused":
            return why, rest, 0.0
        p, _ = settle()
        return "", p, float(p[2] - press_at[2]) - base

    def confirm_drop(detail: str) -> Output:
        """Probe sideways: a wall within the probe confirms the drop."""
        if not confirm:
            out["confirmed"] = False
            return finish("dropped_in", detail + "; not confirmed (confirm false)")
        p0, r = settle()
        d = np.array([1.0, 0.0, 0.0])
        if confirm_direction is not None:
            d = np.array([float(confirm_direction[k]) for k in "xy"] + [0.0])
            d /= float(np.linalg.norm(d)) or 1.0
        if seat_check > 0:
            deep = stay(p0) - np.array([0.0, 0.0, float(seat_check)])
            kind, why = arm.move(deep, r)
            if kind == "refused":
                return finish("blocked", detail + f"; the seat check's press was refused: {why}")
            q, _ = settle()
            follow = float(p0[2] - q[2])
            out["seat_follow"] = follow
            held_back = float(q[2] - deep[2]) - base
            if held_back > float(contact_lag):
                unload(q, r)
                out["sink"] = float(rest[2] - q[2])
                return finish(
                    "unconfirmed",
                    detail.replace("; still holding", "")
                    + f"; but a press {seat_check * 1e3:.1f} mm deeper was held back "
                    f"{held_back * 1e3:.1f} mm (the hand went {follow * 1e3:.1f} mm): the part is caught on the "
                    "socket's edge or resting on the rim, not in; still holding",
                )
            p0 = q
        hold = stay(p0)
        kind, why = arm.move(hold, r)
        if kind == "refused":
            return finish("blocked", detail + f"; the confirmation's hold was refused: {why}")
        q, _ = settle()
        side0 = float(np.dot(hold - q, d))
        n = math.ceil(float(confirm_probe) / CONFIRM_STEP - 1e-9)
        hit = -1.0
        for i in range(1, n + 1):
            cmd = hold + d * min(i * CONFIRM_STEP, float(confirm_probe))
            kind, why = arm.move(cmd, r)
            if kind == "refused":
                arm.move(hold, r)
                return finish("blocked", detail + f"; confirmation step {i} refused: {why}")
            q, _ = settle()
            if float(np.dot(cmd - q, d)) - side0 > float(confirm_lag):
                hit = float(np.dot(q - p0, d))
                break
        arm.move(hold, r)
        out["confirm_travel"] = hit
        if hit >= 0.0:
            out["confirmed"] = True
            return finish(
                "dropped_in",
                detail + f"; confirmed: a wall stopped the hand {hit * 1e3:.1f} mm sideways",
            )
        return finish(
            "unconfirmed",
            detail.replace("; still holding", "")
            + f"; but the hand moved {confirm_probe * 1e3:.0f} mm sideways without meeting a wall: "
            "the part is not in a socket (it slid in the jaws or walked off on the rim); still holding",
        )

    def stay(p: np.ndarray) -> np.ndarray:
        """The command that keeps the hand at ``p`` without pushing: ``p`` less the
        free-motion lag (an arm that stands millimetres over its aim in free air would
        otherwise rise that much, lifting the part out of its socket)."""
        return p - np.array([0.0, 0.0, base])

    def unload(p: np.ndarray, r: np.ndarray) -> None:
        kind, why = arm.move(stay(p), r)
        if kind == "refused":
            notes.append(f"unloading the press was refused ({why}); the arm may still be pushing")

    order = yaw_order(float(yaw_range_deg), float(yaw_step_deg))
    last_z = float(rest[2])
    for k, yaw in enumerate(order):
        out["tries"] = k + 1
        if k:
            why = turn(yaw)
            if why:
                p, r = arm.read()
                unload(p, r)
                return finish("blocked", f"at {yaw:+.0f} deg: {why}")
            short = yaw - state["yaw"]
            if abs(short) > float(jam_deg):
                p, r = arm.read()
                unload(p, r)
                sunk = float(rest[2] - p[2])
                if sunk >= float(drop_depth):
                    # The turn swept through the matching yaw, the part dropped in on
                    # the way, and the socket's walls stopped the rest of the turn.
                    out["yaw_found_deg"] = state["yaw"]
                    out["sink"] = sunk
                    return confirm_drop(
                        f"the turn to {yaw:+.0f} deg stopped at {state['yaw']:+.1f} deg with the hand "
                        f"{sunk * 1e3:.1f} mm under the resting height: the part dropped in on the way "
                        f"(try {k + 1} of {len(order)}); still holding",
                    )
                return finish(
                    "jammed",
                    f"the turn to {yaw:+.0f} deg stopped at {state['yaw']:+.1f} deg "
                    f"({short:+.1f} deg short): the part is held against turning",
                )
        _, r = arm.read()
        why, p, held = press(r)
        if why:
            return finish("blocked", f"the press at {yaw:+.0f} deg was refused: {why}")
        sink = float(rest[2] - p[2])
        out["sink"] = sink
        if check_holding and (gone := _jaws_empty(ctx, arm.on_arm)) is not None:
            unload(p, r)
            return finish(
                "blocked",
                f"the jaws closed to their stop ({gone * 1e3:.1f} mm) at {state['yaw']:+.1f} deg "
                f"(try {k + 1}): the part is no longer in the hand",
            )
        # A drop is sudden: most of the sink comes in this press. A part creeping
        # through the jaws on the rim press after press sinks the hand too.
        sudden = last_z - float(p[2]) >= float(drop_depth) / 2
        last_z = float(p[2])
        if held <= float(contact_lag) / 2 and sink >= float(drop_depth) and sudden:
            out["yaw_found_deg"] = state["yaw"]
            unload(p, r)
            return confirm_drop(
                f"the hand sank {sink * 1e3:.1f} mm under the resting height at "
                f"{state['yaw']:+.1f} deg (try {k + 1} of {len(order)}); still holding",
            )
    if out["sink"] >= float(drop_depth) / 2:
        notes.append(
            f"the pressed hand crept {out['sink'] * 1e3:.1f} mm down over the tries without a drop: "
            "the part may be sliding in the jaws"
        )
    p, r = arm.read()
    unload(p, r)
    if return_on_fail:
        why = turn(0.0)
        if why:
            notes.append(f"turning back: {why}")
    return finish(
        "no_drop",
        f"no drop over {len(order)} yaws within +-{yaw_range_deg:.0f} deg "
        f"(last sink {out['sink'] * 1e3:.1f} mm, drop needs {drop_depth * 1e3:.1f} mm)"
        + ("; turned back to the start" if return_on_fail else ""),
    )
