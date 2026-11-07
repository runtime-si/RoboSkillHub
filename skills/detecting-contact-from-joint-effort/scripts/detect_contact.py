# Adapted from i2rt (https://github.com/i2rt-robotics/i2rt),
# i2rt/robots/utils.py, GripperForceLimiter.compute_target_gripper_torque and update
# (the "clog" test), MIT License, Copyright (c) I2RT Robotics; the thresholds are
# i2rt/robots/config/linear_4310.yml's limiter_params.
# Changes: i2rt's test watches one gripper motor inside its control loop (windowed
# mean |effort| over clog_force_threshold while |qvel| is under
# clog_speed_threshold); here it watches every arm joint from outside, through
# robot.get_observation, while the hand is stepped along a short straight line with
# robot.go_to_pose. The effort is judged as a residual against a baseline taken in
# free motion along the same line (the running mean of the windows before), which
# stands in for i2rt's KDL gravity torque that a connector does not expose. Where the
# observation carries no efforts (neither the gap connector nor a YAM station
# publishes them today) the decision falls back to tracking lag: how far the
# measured tool centre stands behind the commanded one, against the lag measured in
# free air (the real YAM threading station's settle_down rule). i2rt's response to a
# clog (holding a capped force) is not ported: this skill decides and backs off.
"""Step the hand along a short line and decide from proprioception whether it met something."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

STEP = 0.0025
"""Length of each step along the line [m]: the real YAM threading station's
SETTLE_STEP, with which it feels for the table before closing on a cable."""

LAG_THRESHOLD = 0.003
"""How much further behind its commanded position than in free motion the hand
must stand to be held by something [m]: the station's SETTLE_CONTACT (low at far
reach a real YAM stood 5.4 mm over its aim in free air, the next 1.5 mm read as
the table)."""

EFFORT_THRESHOLD = 0.5
"""Least joint effort residual read as contact [N.m]: i2rt's clog_force_threshold
for the YAM's 4310 grippers (linear_4310.yml, crank_4310.yml). Not tuned on arm
joints; EFFORT_SIGMAS raises it per joint where the free-motion windows are noisier."""

EFFORT_SIGMAS = 4.0
"""A joint's threshold is at least this many standard deviations of its free-motion
windows (with three or more of them). A choice, not a measurement."""

SPEED_THRESHOLD = 0.3
"""Joint speed above which an effort sample is a joint still moving, not pushing
[rad/s]: i2rt's clog_speed_threshold for linear_4310."""

EFFORT_SAMPLES = 5
"""Observations averaged into one effort window. i2rt averages 0.1 s of its own
control loop; a skill reads through the connector, so it counts reads."""

BASELINE_WINDOW = 3
"""Free-motion samples (lags, effort windows) the baseline is taken over, the most
recent ones, so a lag or a gravity torque that drifts along the line is followed
(a real YAM's lag grew from 3.6 to 6.7 mm as it reached lower)."""

MIN_FREE = 2
"""Free-motion samples needed before a line that met nothing counts as free."""

MAX_LINE = 0.15
"""Longest probe line [m]; this is a short feel, not a transport move."""

REACH_TOLERANCE = 0.015
"""How far from the start the hand may end and still have arrived [m]: no tighter
than a real YAM station's 12 mm per move."""

STEP_TOLERANCE = 0.001
"""Arrival ball asked of each step [m], passed as go_to_pose's ``tolerance``:
a feedback servo with a 4 mm default arrival ball calls a 2.5 mm step reached
without moving. 0 leaves the connector's default."""

BACKOFF = 0.003
"""How far back along the line the hand goes after contact, so it stops pushing [m]."""

SETTLE_QUIET = 0.0003
SETTLE_PERIOD = 0.3
SETTLE_TIMEOUT = 3.0
"""The hand has settled when two reads SETTLE_PERIOD apart [s] differ by less than
SETTLE_QUIET [m], or after SETTLE_TIMEOUT [s]: the station's still(). A real
station's move returns once the hand is within 12 mm, before a short step ends."""

STATION_LABEL = "push: feeling for contact"
"""The label a pose-list station (the YAM threading station's connector) is given;
labels starting with "push" go at its base rate, as touching moves must."""

REFUSALS = ("unreachable", "bad_speed_scale")
"""Routed outcomes that mean a move was refused, not stopped."""


class Output(TypedDict):
    route: str
    detail: str
    contact: bool
    signal: str
    contact_position: dict[str, float]
    contact_point: dict[str, float]
    travelled: float
    lag_over_baseline: float
    effort_ratio: float
    efforts_available: bool
    steps: int
    baseline_lag: float


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _arr(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=float).reshape(-1)[:3]


class _Hand:
    """The hand through whichever pose convention the connector speaks: gap's
    (``pose={"position": {x, y, z}, "rotation": {w, x, y, z}}``) or a station's
    position and quat_xyzw lists (the YAM threading station's connector)."""

    def __init__(self, ctx: NodeContext, arm_id: int, tolerance: float, max_steps: int) -> None:
        self.ctx = ctx
        self.on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
        self.knobs: dict[str, Any] = {}
        if tolerance > 0:
            self.knobs["tolerance"] = float(tolerance)
        if max_steps > 0:
            self.knobs["max_steps"] = int(max_steps)
        self.lists = False
        self.rotation: Any = None

    def read(self) -> np.ndarray:
        reply = self.ctx.tool("robot.get_ee_pose", **self.on_arm)
        pose = reply.get("pose", reply) if isinstance(reply, dict) else reply
        if "quat_xyzw" in pose:
            self.lists = True
            if self.rotation is None:
                self.rotation = [float(v) for v in pose["quat_xyzw"]]
        elif self.rotation is None:
            self.rotation = dict(pose["rotation"])
        return _arr(pose["position"])

    def move(self, position: np.ndarray, label: str = STATION_LABEL) -> tuple[str, str]:
        """("ok" | "short" | "refused", why). "short" is a move that ended without
        arriving (a servo's timeout or stall, a station's "did not reach its
        target"): the hand's measured position decides what it met."""
        try:
            if self.lists:
                reply = self.ctx.tool(
                    "robot.go_to_pose",
                    position=[float(v) for v in position],
                    quat_xyzw=self.rotation,
                    label=label,
                )
            else:
                pose = {"position": _vec(position), "rotation": self.rotation}
                try:
                    reply = self.ctx.tool(
                        "robot.go_to_pose", pose=pose, **self.knobs, **self.on_arm
                    )
                except TypeError as exc:
                    if not self.knobs or "keyword" not in str(exc):
                        raise
                    logger.info(
                        "detecting-contact: go_to_pose takes no %s (%s)", sorted(self.knobs), exc
                    )
                    self.knobs = {}
                    reply = self.ctx.tool("robot.go_to_pose", pose=pose, **self.on_arm)
        except Exception as exc:  # a station raises where a simulator routes
            why = str(exc)
            return ("short" if "did not reach" in why else "refused"), why
        if not isinstance(reply, dict):
            return "ok", ""
        outcome = reply.get("outcome")
        if outcome in REFUSALS:
            return "refused", f"{outcome}: {reply.get('reason', '')}".rstrip(": ")
        if outcome not in (None, "reached") or reply.get("reached") is False:
            return "short", str(outcome or reply.get("reason") or "not reached")
        return "ok", ""

    def still(self, quiet: float, period: float, timeout: float) -> np.ndarray:
        """Where the hand stands once it stops moving (or after ``timeout``)."""
        last = self.read()
        if period <= 0:
            return last
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            time.sleep(period)
            now = self.read()
            if float(np.linalg.norm(now - last)) < quiet:
                return now
            last = now
        return last


def _joint_state(obs: Any, arm_id: int) -> dict[str, Any] | None:
    if not isinstance(obs, dict):
        return None
    arms = obs.get("arms")
    if isinstance(arms, list) and len(arms) > arm_id and isinstance(arms[arm_id], dict):
        js = arms[arm_id].get("joint_state")
        return js if isinstance(js, dict) else None
    js = obs.get("joint_state")
    return js if isinstance(js, dict) else None


def _field(js: dict[str, Any] | None, *names: str) -> np.ndarray | None:
    for name in names:
        value = (js or {}).get(name)
        if value is not None:
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size:
                return arr
    return None


def _efforts(js: dict[str, Any] | None) -> np.ndarray | None:
    return _field(js, "efforts", "effort", "torques")


def _velocities(js: dict[str, Any] | None) -> np.ndarray | None:
    return _field(js, "velocities", "velocity")


def run(
    ctx: NodeContext,
    end: dict[str, float],
    start: dict[str, float] | None = None,
    signal: str = "auto",
    step: float = STEP,
    lag_threshold: float = LAG_THRESHOLD,
    effort_threshold: float = EFFORT_THRESHOLD,
    speed_threshold: float = SPEED_THRESHOLD,
    effort_samples: int = EFFORT_SAMPLES,
    floor_z: float | None = None,
    tip_reach: float = 0.0,
    backoff: float = BACKOFF,
    close_hand: bool = True,
    check_reach: bool = True,
    reach_tolerance: float = REACH_TOLERANCE,
    step_tolerance: float = STEP_TOLERANCE,
    step_max_steps: int = 0,
    settle_quiet: float = SETTLE_QUIET,
    settle_period: float = SETTLE_PERIOD,
    settle_timeout: float = SETTLE_TIMEOUT,
    arm_id: int = 0,
) -> Output:
    if signal not in ("auto", "effort", "lag"):
        raise ValueError(f"signal must be auto, effort or lag, got {signal!r}")
    hand = _Hand(ctx, arm_id, step_tolerance, step_max_steps)
    here = hand.read()
    out: Output = {
        "route": "blocked",
        "detail": "",
        "contact": False,
        "signal": "",
        "contact_position": _vec(here),
        "contact_point": _vec(here),
        "travelled": 0.0,
        "lag_over_baseline": 0.0,
        "effort_ratio": 0.0,
        "efforts_available": False,
        "steps": 0,
        "baseline_lag": 0.0,
    }

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        return out

    def settle() -> np.ndarray:
        return hand.still(float(settle_quiet), float(settle_period), float(settle_timeout))

    a = here if start is None else _arr(start)
    b = _arr(end)
    if floor_z is not None:
        floor = float(floor_z)
        if a[2] < floor - 1e-9:
            return finish(
                "blocked", f"the start is {(floor - a[2]) * 1e3:.1f} mm under the tool-centre floor"
            )
        if b[2] < floor:
            # The line ends where it would cross the floor.
            b = a + (b - a) * (a[2] - floor) / (a[2] - b[2])
    length = float(np.linalg.norm(b - a))
    if length < 1e-4:
        return finish(
            "blocked", "the line has no length (end at the start, or the start on the floor)"
        )
    if length > MAX_LINE:
        return finish(
            "blocked",
            f"the line is {length * 1e3:.0f} mm, over the {MAX_LINE * 1e3:.0f} mm a probe may be",
        )
    direction = (b - a) / length

    if check_reach and not hand.lists:
        try:
            reply = ctx.tool(
                "robot.solve_ik",
                pose={"position": _vec(b), "rotation": hand.rotation},
                **hand.on_arm,
            )
        except (
            Exception
        ) as exc:  # a station raises an unreachable pose; a connector may lack the tool
            if "unreachable" in str(exc).lower():
                return finish("blocked", f"the line's end is out of reach (IK preview: {exc})")
            logger.info("detecting-contact: no IK preview (%s)", exc)
            reply = None
        # A stall at the edge of reach reads exactly like contact.
        if isinstance(reply, dict) and (
            reply.get("success") is False or reply.get("within_servo_tolerance") is False
        ):
            return finish(
                "blocked",
                "the line's end is out of reach (IK preview): a stall there would read as contact",
            )

    notes: list[str] = []
    if close_hand:
        try:
            ctx.tool("robot.close_gripper", **hand.on_arm)
        except Exception as exc:
            notes.append(f"the hand was not closed ({exc})")

    # Efforts: looked for once. An observation may render every camera, so a
    # connector that publishes none is not asked again.
    efforts_on = False
    if signal in ("auto", "effort"):
        try:
            efforts_on = (
                _efforts(_joint_state(ctx.tool("robot.get_observation"), arm_id)) is not None
            )
        except Exception as exc:
            logger.info("detecting-contact: no observation (%s)", exc)
    out["efforts_available"] = efforts_on
    if signal == "effort" and not efforts_on:
        return finish(
            "no_signal",
            "signal=effort, but the observation carries no joint efforts; nothing moved",
        )
    use_lag = signal != "effort"

    def window() -> np.ndarray | None:
        """i2rt's windowed mean effort, over the samples whose joints were not moving."""
        kept = []
        for _ in range(max(1, int(effort_samples))):
            js = _joint_state(ctx.tool("robot.get_observation"), arm_id)
            eff, vel = _efforts(js), _velocities(js)
            if eff is None:
                continue
            if vel is not None and float(np.max(np.abs(vel))) >= float(speed_threshold):
                continue
            kept.append(eff)
        return np.mean(kept, axis=0) if kept else None

    lags: list[float] = []
    windows: list[np.ndarray] = []
    if start is not None:
        kind, why = hand.move(a, label="probe start")
        p = settle()
        off = float(np.linalg.norm(p - a))
        if kind == "refused" or off > float(reach_tolerance):
            return finish(
                "blocked",
                f"the start: {why or 'ended off'}; the hand is {off * 1e3:.0f} mm from it",
            )
        # Standing at a commanded pose in free air: the first free-motion sample.
        lags.append(float(np.dot(a - p, direction)))
    else:
        p = settle()
    if efforts_on:
        w = window()
        if w is not None:
            windows.append(w)

    n = math.ceil(length / float(step) - 1e-9)
    shorts = 0
    jitter = 0.0
    for i in range(1, n + 1):
        goal = a + direction * min(i * float(step), length)
        kind, why = hand.move(goal)
        out["steps"] = i
        if kind == "refused":
            out["contact_position"] = out["contact_point"] = _vec(hand.read())
            return finish("blocked", f"step {i} refused: {why}")
        shorts += kind == "short"
        p = settle()
        lag = float(np.dot(goal - p, direction))
        fired: list[str] = []
        over = 0.0
        if use_lag and lags:
            base = float(np.median(lags[-BASELINE_WINDOW:]))
            over = lag - base
            if over > float(lag_threshold):
                fired.append("lag")
        w = window() if efforts_on else None
        ratio = 0.0
        if w is not None and windows:
            recent = np.array(windows[-BASELINE_WINDOW:])
            noise = (
                np.std(np.array(windows[-5:]), axis=0) if len(windows) >= 3 else np.zeros_like(w)
            )
            thr = np.maximum(float(effort_threshold), EFFORT_SIGMAS * noise)
            ratio = float(np.max(np.abs(w - recent.mean(axis=0)) / thr))
            if ratio > 1.0:
                fired.append("effort")
        out["lag_over_baseline"], out["effort_ratio"] = over, ratio
        if fired:
            out["contact"] = True
            out["signal"] = "+".join(sorted(fired))
            out["contact_position"] = _vec(p)
            out["contact_point"] = _vec(p + direction * float(tip_reach))
            out["travelled"] = float(np.dot(p - a, direction))
            out["baseline_lag"] = float(np.median(lags[-BASELINE_WINDOW:])) if lags else 0.0
            detail = (
                f"contact at step {i} after {out['travelled'] * 1e3:.1f} mm "
                f"({out['signal']}: lag {over * 1e3:+.1f} mm over the free-motion baseline"
                + (f", effort {ratio:.2f} x its threshold" if efforts_on else "")
                + ")"
            )
            if backoff > 0:
                kind, why = hand.move(p - direction * float(backoff), label="back off the contact")
                if kind == "refused":
                    detail += f"; backing off refused ({why}), the hand may still be pushing"
                else:
                    detail += f"; backed off {backoff * 1e3:.0f} mm"
            if notes:
                detail += "; " + "; ".join(notes)
            return finish("contact", detail)
        if lags:
            jitter = max(jitter, abs(lag - lags[-1]))
        lags.append(lag)
        if w is not None:
            windows.append(w)

    out["contact_position"] = out["contact_point"] = _vec(p)
    out["travelled"] = float(np.dot(p - a, direction))
    out["baseline_lag"] = float(np.median(lags[-BASELINE_WINDOW:])) if lags else 0.0
    ended = (
        "at the tool-centre floor"
        if floor_z is not None and b[2] <= float(floor_z) + 1e-9
        else "at the end"
    )
    tail = f"{shorts} step(s) ended short" if shorts else ""
    extra = "; ".join([t for t in (tail, *notes) if t])
    extra = f"; {extra}" if extra else ""
    lag_ok = use_lag and len(lags) >= MIN_FREE and jitter <= float(lag_threshold)
    effort_ok = efforts_on and len(windows) >= MIN_FREE
    if lag_ok or effort_ok:
        return finish(
            "free",
            f"nothing met over {out['travelled'] * 1e3:.1f} mm, {ended} of the line "
            f"(free-motion lag {out['baseline_lag'] * 1e3:.1f} mm, step-to-step {jitter * 1e3:.1f} mm)"
            + extra,
        )
    why = (
        f"the hand's lag jumped {jitter * 1e3:.1f} mm between free steps, over the "
        f"{lag_threshold * 1e3:.1f} mm the test needs"
        if use_lag and len(lags) >= MIN_FREE
        else f"only {max(len(lags), len(windows))} free-motion sample(s); the line is too short to judge"
    )
    return finish(
        "no_signal", f"no usable signal over {out['travelled'] * 1e3:.1f} mm: {why}" + extra
    )
