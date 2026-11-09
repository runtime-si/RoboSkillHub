# Adapted from o2ac-ur (https://github.com/o2ac/o2ac-ur, commit
# 65ddc5496879799038cefbad061380b7d0300d73),
# catkin_ws/src/o2ac_routines/src/o2ac_routines/ur_force_control.py,
# URForceController.linear_push and do_insertion's termination_criteria,
# BSD-3-Clause, Copyright (c) 2021 OMRON SINIC X (the file's own header; the
# repository's LICENSE is MIT, Copyright (c) 2020 Team O2AC). Both texts are in
# LICENSE.o2ac beside this skill.
# Changes: o2ac pushes with a hybrid force/position controller on a UR's wrist
# force-torque sensor until the target force (STOP_ON_TARGET_FORCE), a maximum
# translation along the push axis (linear_push's termination_criteria) or a
# target position along it (do_insertion's, relaxed by goal_tolerance_if_lockup
# when the arm has stood still) is reached. A YAM has no force sensor and a
# connector offers no force control, so here the push is stepped with
# robot.go_to_pose and the force is read from proprioception: how far the hand
# lags its command along the axis beyond the lag it showed before contact (a
# position-controlled joint behaves as a spring, so the lag stands for the
# force), and joint effort residuals where the observation publishes efforts.
# "Target force reached" becomes the lag over max_lag (or an effort residual over
# effort_threshold); o2ac's zeroing of the sensor before the push becomes the
# lag baseline taken while the hand stands at its start and moves freely; the
# snap-fit detection (the lag collapsing while the hand jumps forward after a
# rise) is new, from the candidate's description, not in o2ac; the push is
# unloaded at the end (the command brought back to where the hand stands); moves
# that are refused are routed rather than raised. The pose and settle helpers
# follow skills/detecting-contact-from-joint-effort.
"""Push a held part along an axis with a bounded push and a bounded travel, and say how it ended."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

DOWN = {"x": 0.0, "y": 0.0, "z": -1.0}

MAX_TRANSLATION = 0.008
"""Travel along the axis that ends the push as seated [m]: a press-fit or a LEGO
brick's clutch is 1.7 mm and a snap-fit's travel a few mm, so 8 mm leaves room for
the part to start a little above its seat. o2ac passes it per call."""

STEP = 0.001
"""Commanded advance per step [m]. Smaller than detecting-contact's 2.5 mm: a push
already in contact raises the force by about stiffness x step each step."""

CONTACT_LAG = 0.0015
"""Lag over the free-motion baseline that marks resistance [m]: half the 3 mm a
real YAM station reads as contact (SETTLE_CONTACT); a press must notice the part
loading before it is held, and the push continues after it."""

MAX_LAG = 0.006
"""Lag over the baseline that ends the push as the force limit [m] (o2ac's target
force). Twice the station's 3 mm contact lag; the force it stands for depends on
the arm's servo stiffness at that pose and is not calibrated (no force sensor)."""

SNAP_RISE = 0.002
"""Least lag over the baseline the push must have built before a collapse counts
as a snap [m]: a snap-fit or a brick's clutch resists before it gives."""

SNAP_DROP = 0.0015
"""How much the lag must fall in one step, and the hand jump forward, to be a snap
[m]. A choice (no snap-fit was measured on a YAM): above the 0.1-0.2 mm
step-to-step jitter a station twin's lag shows in free motion."""

EFFORT_THRESHOLD = 0.5
"""Joint effort residual that ends the push like MAX_LAG does [N.m], where the
observation publishes efforts: i2rt's clog_force_threshold for the YAM's 4310
gripper motors (linear_4310.yml), as in detecting-contact-from-joint-effort; not
tuned on arm joints."""

CONTACT_EFFORT_FRACTION = 0.5
"""Fraction of effort_threshold at which an effort residual marks resistance (the
effort counterpart of CONTACT_LAG being half the contact lag). A choice."""

EFFORT_SAMPLES = 5
SPEED_THRESHOLD = 0.3
"""Observations averaged into one effort window, and the joint speed [rad/s] above
which a sample is dropped as still moving (i2rt's clog_speed_threshold)."""

BASELINE_WINDOW = 3
"""Free-motion lags the baseline is the median of (the most recent)."""

STEP_TOLERANCE = 0.001
"""Arrival ball asked of each step [m], passed as go_to_pose's ``tolerance``: a
feedback servo with a 4 mm default ball calls a 1 mm step reached without moving."""

SETTLE_QUIET = 0.0003
SETTLE_PERIOD = 0.3
SETTLE_TIMEOUT = 3.0
"""The hand has settled when two reads SETTLE_PERIOD apart [s] differ by less than
SETTLE_QUIET [m], or after SETTLE_TIMEOUT [s] (a real YAM station's still())."""

STATION_LABEL = "push: guarded press"
"""Label for a pose-list station's connector; "push..." runs at its base rate."""

REFUSALS = ("unreachable", "bad_speed_scale")


class Output(TypedDict):
    route: str
    detail: str
    travelled: float
    contact_travel: float
    peak_lag: float
    final_lag: float
    remaining: float
    steps: int
    signal: str
    efforts_available: bool
    relieved: bool
    stop_position: dict[str, float]


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _arr(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=float).reshape(-1)[:3]


class _Hand:
    """The hand through gap's pose convention or a station's position / quat_xyzw lists."""

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
        """("ok" | "short" | "refused", why); "short" ended without arriving (held)."""
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
                        "guarded-push: go_to_pose takes no %s (%s)", sorted(self.knobs), exc
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


def run(
    ctx: NodeContext,
    direction: dict[str, float] | None = None,
    max_translation: float = MAX_TRANSLATION,
    target: dict[str, float] | None = None,
    goal_tolerance_if_lockup: float = 0.0,
    step: float = STEP,
    contact_lag: float = CONTACT_LAG,
    max_lag: float = MAX_LAG,
    snap_rise: float = SNAP_RISE,
    snap_drop: float = SNAP_DROP,
    stop_on_snap: bool = True,
    signal: str = "auto",
    effort_threshold: float = EFFORT_THRESHOLD,
    effort_samples: int = EFFORT_SAMPLES,
    speed_threshold: float = SPEED_THRESHOLD,
    floor_z: float | None = None,
    relieve: bool = True,
    check_reach: bool = True,
    step_tolerance: float = STEP_TOLERANCE,
    step_max_steps: int = 0,
    settle_quiet: float = SETTLE_QUIET,
    settle_period: float = SETTLE_PERIOD,
    settle_timeout: float = SETTLE_TIMEOUT,
    arm_id: int = 0,
) -> Output:
    if signal not in ("auto", "effort", "lag"):
        raise ValueError(f"signal must be auto, effort or lag, got {signal!r}")
    if not 0 < contact_lag < max_lag:
        raise ValueError("need 0 < contact_lag < max_lag")
    d = _arr(direction or DOWN)
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        raise ValueError("direction has no length")
    d = d / norm

    hand = _Hand(ctx, arm_id, step_tolerance, step_max_steps)
    p0 = hand.read()
    out: Output = {
        "route": "blocked",
        "detail": "",
        "travelled": 0.0,
        "contact_travel": -1.0,
        "peak_lag": 0.0,
        "final_lag": 0.0,
        "remaining": 0.0,
        "steps": 0,
        "signal": "",
        "efforts_available": False,
        "relieved": False,
        "stop_position": _vec(p0),
    }
    notes: list[str] = []

    def finish(route: str, detail: str) -> Output:
        out["route"] = route
        out["detail"] = detail + ("; " + "; ".join(notes) if notes else "")
        return out

    def settle() -> np.ndarray:
        return hand.still(float(settle_quiet), float(settle_period), float(settle_timeout))

    # How far along the axis the hand must get: the target's depth or max_translation.
    goal = float(np.dot(_arr(target) - p0, d)) if target is not None else float(max_translation)
    if goal <= 0:
        out["remaining"] = goal
        return finish(
            "seated", f"the hand already stands {-goal * 1e3:.1f} mm past the target; nothing moved"
        )
    # Commands never lead the start by more than the goal plus the force limit.
    reach = goal + float(max_lag) + float(step)
    if floor_z is not None and d[2] < -1e-9:
        reach = min(reach, (p0[2] - float(floor_z)) / -d[2])
        if reach <= float(step):
            return finish("blocked", "the hand is at the tool-centre floor; nothing moved")
    end = p0 + d * reach

    if check_reach and not hand.lists:
        try:
            reply = ctx.tool(
                "robot.solve_ik",
                pose={"position": _vec(end), "rotation": hand.rotation},
                **hand.on_arm,
            )
        except Exception as exc:
            if "unreachable" in str(exc).lower():
                return finish("blocked", f"the end of the push is out of reach (IK preview: {exc})")
            logger.info("guarded-push: no IK preview (%s)", exc)
            reply = None
        if isinstance(reply, dict) and (
            reply.get("success") is False or reply.get("within_servo_tolerance") is False
        ):
            return finish(
                "blocked",
                "the end of the push is out of reach (IK preview): a stall there would read as force",
            )

    efforts_on = False
    if signal in ("auto", "effort"):
        try:
            js = _joint_state(ctx.tool("robot.get_observation"), arm_id)
            efforts_on = _field(js, "efforts", "effort", "torques") is not None
        except Exception as exc:
            logger.info("guarded-push: no observation (%s)", exc)
    out["efforts_available"] = efforts_on
    if signal == "effort" and not efforts_on:
        return finish("blocked", "signal=effort, but the observation carries no joint efforts")
    use_lag = signal != "effort"

    def window() -> np.ndarray | None:
        kept = []
        for _ in range(max(1, int(effort_samples))):
            js = _joint_state(ctx.tool("robot.get_observation"), arm_id)
            eff = _field(js, "efforts", "effort", "torques")
            vel = _field(js, "velocities", "velocity")
            if eff is None:
                continue
            if vel is not None and float(np.max(np.abs(vel))) >= float(speed_threshold):
                continue
            kept.append(eff)
        return np.mean(kept, axis=0) if kept else None

    # o2ac zeroes its force sensor before pushing; here the zero is the lag (and the
    # efforts) with the hand commanded where it stands.
    kind, why = hand.move(p0, label="push: hold before pressing")
    if kind == "refused":
        return finish("blocked", f"holding the start was refused: {why}")
    p = settle()
    lags = [float(np.dot(p0 - p, d))]
    windows = [w] if efforts_on and (w := window()) is not None else []

    cmd = p0.copy()
    prev_travel = float(np.dot(p - p0, d))
    prev_over = 0.0
    peak = 0.0
    snapped = False
    n = math.ceil(reach / float(step) - 1e-9)
    stop = ""
    for i in range(1, n + 1):
        cmd = p0 + d * min(i * float(step), reach)
        kind, why = hand.move(cmd)
        out["steps"] = i
        if kind == "refused":
            out["stop_position"] = _vec(hand.read())
            return finish("blocked", f"step {i} refused: {why}")
        p = settle()
        travel = float(np.dot(p - p0, d))
        lag = float(np.dot(cmd - p, d))
        base = float(np.median(lags[-BASELINE_WINDOW:]))
        over = lag - base if use_lag else 0.0
        ratio = 0.0
        w = window() if efforts_on else None
        if w is not None and windows:
            ratio = float(np.max(np.abs(w - np.mean(windows[-BASELINE_WINDOW:], axis=0))))
            ratio /= float(effort_threshold)
        loaded = over > float(contact_lag) or ratio > CONTACT_EFFORT_FRACTION
        if loaded and out["contact_travel"] < 0:
            out["contact_travel"] = travel
        peak = max(peak, over)
        out["travelled"], out["final_lag"], out["peak_lag"] = travel, over, peak
        out["stop_position"] = _vec(p)
        out["remaining"] = goal - travel
        if (
            use_lag
            and peak >= float(snap_rise)
            and prev_over - over >= float(snap_drop)
            and travel - prev_travel >= float(snap_drop)
        ):
            snapped = True
            if stop_on_snap:
                out["signal"] = "lag"
                stop = "snap"
                break
        if travel >= goal - 1e-9:
            stop = "goal"
            break
        fired = [s for s, on in (("lag", over > float(max_lag)), ("effort", ratio > 1.0)) if on]
        if fired:
            out["signal"] = "+".join(fired)
            stop = "limit"
            break
        if not loaded and out["contact_travel"] < 0:
            lags.append(lag)
            if w is not None:
                windows.append(w)
        prev_travel, prev_over = travel, over

    travel, over = out["travelled"], out["final_lag"]
    if relieve and over > float(contact_lag) / 2:
        # Bring the command back to where the hand stands so it stops pushing.
        hold = p + d * float(np.median(lags[-BASELINE_WINDOW:]))
        kind, why = hand.move(hold, label="push: relieve")
        if kind == "refused":
            notes.append(f"relieving the push was refused ({why}); the arm may still be pushing")
        else:
            out["relieved"] = True

    summary = (
        f"travelled {travel * 1e3:.1f} mm of {goal * 1e3:.1f} mm in {out['steps']} step(s), "
        f"lag {over * 1e3:+.1f} mm over the free baseline (peak {peak * 1e3:+.1f} mm)"
    )
    if out["contact_travel"] >= 0:
        summary += f", resistance from {out['contact_travel'] * 1e3:.1f} mm"
    else:
        summary += ", no resistance felt"
    if snapped and stop != "snap":
        summary += ", a snap felt on the way"
    if stop == "snap":
        return finish("snap_detected", f"snap: the lag fell and the hand jumped forward; {summary}")
    if stop == "goal":
        return finish(
            "seated", f"reached the {'target' if target is not None else 'travel'}; {summary}"
        )
    if stop == "limit":
        if target is not None and goal - travel <= float(goal_tolerance_if_lockup):
            return finish(
                "seated",
                f"force limit ({out['signal']}) within goal_tolerance_if_lockup of the target; {summary}",
            )
        return finish(
            "force_limit_no_travel", f"force limit ({out['signal']}) short of the goal; {summary}"
        )
    if floor_z is not None and travel < goal:
        return finish("blocked", f"the tool-centre floor ended the push; {summary}")
    return finish(
        "force_limit_no_travel",
        f"the commands ran {reach * 1e3:.1f} mm ahead without the hand reaching the goal; {summary}",
    )
