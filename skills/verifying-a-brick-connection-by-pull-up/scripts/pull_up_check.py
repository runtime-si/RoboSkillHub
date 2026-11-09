# Re-expresses the PullUp skill of CMU MFI's vibro_tactile_toolbox
# (https://github.com/cmu-mfi/vibro_tactile_toolbox, commit
# 30fb15bb05a91606dfd4725f1f37352207f0299f: src/skill/common_skills.py, PullUp and
# servo_termination_config; scripts/fts_detector.py, the outcome test;
# config/lego.yaml, skill_params.pull_up and fts_detector.force_threshold). The
# repository declares MIT only in its package.xml and ships no LICENSE file, so
# none of its code is used here; the behaviour is reimplemented.
# There: holding the brick just pressed, the arm lifts 10 mm at 0.1 % speed,
# stopping on a z reaction above the force at the start (or any large reaction),
# and the brick counts as connected when the force at the end differs from the
# start by more than 0.3 N. Here: a YAM has no force sensor, so the lift is
# stepped (0.5 mm) with robot.go_to_pose and the reaction is read from
# proprioception - the hand held below its command beyond the lag it showed at
# the start, or a joint effort residual where the observation publishes efforts -
# and the lift stops at the first sign of a hold and is put back down, so a
# clutched brick is not torn off. The lift is 3 mm, not 10 mm.
"""Lift a just-pressed brick a few millimetres and tell from the arm whether the studs held it."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

UP = np.array([0.0, 0.0, 1.0])

LIFT_HEIGHT = 0.003
"""How far the lift is commanded [m]. vibro_tactile_toolbox lifts 10 mm
(lego.yaml lift_height_offset) on a force-stopped arm; a position-servoed YAM
stopped only between steps would pull a clutched brick off over 10 mm, and a
loose brick is told within a stud height (1.7 mm LEGO): 3 mm, the hunt note's
limit for a YAM."""

STEP = 0.0005
"""Commanded rise per step [m]. Each step raises the pull on a held brick by the
servo stiffness times the step, so a clutch is felt before it lets go."""

HOLD_LAG = 0.0003
"""How far below its command beyond the start's lag the hand must stand for the
brick to be held [m]. Tuned on a YAM station twin: gripping a fixed wall, the
jaws slid once the lag reached 0.5-0.76 mm (so a 1.5 mm threshold, half the real
station's 3 mm contact lag, never fired), while a free cube lifted with 0.04-0.05
mm; 0.3 mm sits between, above the twin's 0.1-0.2 mm free-motion jitter. A real
YAM's jitter on 0.5 mm steps is not measured: retune there."""

EFFORT_THRESHOLD = 0.5
"""Joint effort residual read as a hold [N.m], where efforts are published: i2rt's
clog_force_threshold for the YAM's 4310 gripper motor (as in
detecting-contact-from-joint-effort); not tuned on arm joints. vibro_tactile_toolbox's
own threshold is 0.3 N at the wrist, which no YAM joint reading resolves."""

EFFORT_SAMPLES = 5
SPEED_THRESHOLD = 0.3
"""Observations averaged per effort window, and the joint speed [rad/s] above
which a sample is dropped as still moving (i2rt's clog_speed_threshold)."""

STOP_MARGIN = 0.002
"""Jaw gap within this of the gripper's closed width reads as nothing held [m]
(pressing-a-brick-onto-studs' test)."""

RETREAT = 0.03
"""Straight rise after releasing a connected brick [m]."""

STEP_TOLERANCE = 0.0002
"""Arrival ball asked of each step [m], passed as go_to_pose's ``tolerance``. It must
be well under the step: a YAM station twin's servo stops once inside its ball, so a
1 mm ball left a 0.5 mm step unmoved (read as lag: a hold that was not there).
0 leaves the connector's default."""

SETTLE_QUIET = 0.0003
SETTLE_PERIOD = 0.3
SETTLE_TIMEOUT = 3.0
"""The hand has settled when two reads SETTLE_PERIOD apart [s] differ by less than
SETTLE_QUIET [m], or after SETTLE_TIMEOUT [s] (a real YAM station's still())."""

STATION_LABEL = "push: pull-up check"
"""Label for a pose-list station's connector; "push..." runs at its base rate."""

REFUSALS = ("unreachable", "bad_speed_scale")


class Output(TypedDict):
    route: str
    detail: str
    connected: bool
    rise: float
    lag_over_baseline: float
    effort_ratio: float
    steps: int
    signal: str
    efforts_available: bool
    released: bool
    jaw_width: float


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
                    logger.info("pull-up: go_to_pose takes no %s (%s)", sorted(self.knobs), exc)
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


def _jaw_width(ctx: NodeContext, on_arm: dict[str, Any]) -> tuple[float, float] | None:
    """(jaw gap, closed width) from robot.get_gripper's fraction and describe_gripper's fit."""
    try:
        fraction = float(ctx.tool("robot.get_gripper", **on_arm)["position"])
        grip = ctx.tool("robot.describe_gripper", **on_arm)
    except Exception as exc:
        logger.info("pull-up: gripper not read (%s)", exc)
        return None
    fit = grip.get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or grip.get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)
    if at_open <= at_closed:
        return None
    return at_closed + (at_open - at_closed) * fraction, at_closed


def run(
    ctx: NodeContext,
    lift_height: float = LIFT_HEIGHT,
    step: float = STEP,
    hold_lag: float = HOLD_LAG,
    signal: str = "auto",
    effort_threshold: float = EFFORT_THRESHOLD,
    effort_samples: int = EFFORT_SAMPLES,
    speed_threshold: float = SPEED_THRESHOLD,
    check_holding: bool = True,
    release: bool = True,
    retreat: float = RETREAT,
    put_back: bool = True,
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
    if lift_height <= 0 or step <= 0 or hold_lag <= 0:
        raise ValueError("lift_height, step and hold_lag must be positive")
    hand = _Hand(ctx, arm_id, step_tolerance, step_max_steps)
    out: Output = {
        "route": "blocked",
        "detail": "",
        "connected": False,
        "rise": 0.0,
        "lag_over_baseline": 0.0,
        "effort_ratio": 0.0,
        "steps": 0,
        "signal": "",
        "efforts_available": False,
        "released": False,
        "jaw_width": -1.0,
    }
    notes: list[str] = []

    def finish(route: str, detail: str) -> Output:
        out["route"] = route
        out["detail"] = detail + ("; " + "; ".join(notes) if notes else "")
        return out

    def settle() -> np.ndarray:
        return hand.still(float(settle_quiet), float(settle_period), float(settle_timeout))

    def jaws_empty() -> bool:
        if not check_holding:
            return False
        jaws = _jaw_width(ctx, hand.on_arm)
        if jaws is None:
            return False
        out["jaw_width"] = jaws[0]
        return jaws[0] <= jaws[1] + STOP_MARGIN

    if jaws_empty():
        return finish(
            "not_holding",
            f"the jaws are at their stop ({out['jaw_width'] * 1e3:.1f} mm): no brick in the hand; "
            "nothing moved",
        )

    p0 = hand.read()
    top = p0 + UP * float(lift_height)
    if check_reach and not hand.lists:
        try:
            reply = ctx.tool(
                "robot.solve_ik",
                pose={"position": _vec(top), "rotation": hand.rotation},
                **hand.on_arm,
            )
        except Exception as exc:
            if "unreachable" in str(exc).lower():
                return finish("blocked", f"the top of the lift is out of reach (IK preview: {exc})")
            logger.info("pull-up: no IK preview (%s)", exc)
            reply = None
        # A stall at the edge of reach would read as a brick that held.
        if isinstance(reply, dict) and (
            reply.get("success") is False or reply.get("within_servo_tolerance") is False
        ):
            return finish(
                "blocked",
                "the top of the lift is out of reach (IK preview): a stall there would read as connected",
            )

    efforts_on = False
    if signal in ("auto", "effort"):
        try:
            js = _joint_state(ctx.tool("robot.get_observation"), arm_id)
            efforts_on = _field(js, "efforts", "effort", "torques") is not None
        except Exception as exc:
            logger.info("pull-up: no observation (%s)", exc)
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

    # The start's reading is the zero (vibro_tactile_toolbox's baseline wrench): the
    # hand commanded where it stands, which also takes any remaining press off.
    kind, why = hand.move(p0, label="push: hold before the pull-up")
    if kind == "refused":
        return finish("blocked", f"holding the start was refused: {why}")
    p = settle()
    base_lag = float(np.dot(p0 - p, UP))
    base_eff = window() if efforts_on else None

    def put_down(label: str) -> str:
        kind, why = hand.move(p0, label=label)
        return why if kind == "refused" else ""

    held = False
    n = math.ceil(float(lift_height) / float(step) - 1e-9)
    for i in range(1, n + 1):
        cmd = p0 + UP * min(i * float(step), float(lift_height))
        kind, why = hand.move(cmd)
        out["steps"] = i
        if kind == "refused":
            back = put_down("push: back down after a refused lift")
            if back:
                notes.append(f"putting the brick back was refused too ({back})")
            return finish("blocked", f"lift step {i} refused: {why}")
        p = settle()
        out["rise"] = float(np.dot(p - p0, UP))
        over = float(np.dot(cmd - p, UP)) - base_lag
        out["lag_over_baseline"] = over
        fired = []
        if use_lag and over > float(hold_lag):
            fired.append("lag")
        if base_eff is not None:
            w = window()
            if w is not None:
                ratio = float(np.max(np.abs(w - base_eff))) / float(effort_threshold)
                out["effort_ratio"] = ratio
                if ratio > 1.0:
                    fired.append("effort")
        if fired:
            out["signal"] = "+".join(fired)
            held = True
            break

    rise_txt = (
        f"the hand rose {out['rise'] * 1e3:.2f} mm of the {min(out['steps'] * step, lift_height) * 1e3:.1f} mm "
        f"commanded (lag {out['lag_over_baseline'] * 1e3:+.2f} mm over the start's"
        + (f", effort {out['effort_ratio']:.2f} x its threshold" if efforts_on else "")
        + ")"
    )
    if held:
        out["connected"] = True
        back = put_down("push: back down after the pull-up")
        if back:
            notes.append(f"easing the pull was refused ({back}); the arm may still be pulling")
        if release:
            try:
                ctx.tool("robot.open_gripper", **hand.on_arm)
                out["released"] = True
            except Exception as exc:
                notes.append(f"opening the hand failed ({exc})")
            if out["released"] and retreat > 0:
                kind, why = hand.move(
                    hand.read() + UP * float(retreat), label="retreat after the check"
                )
                if kind == "refused":
                    notes.append(f"the retreat was refused ({why})")
        return finish(
            "connected",
            f"held at lift step {out['steps']} ({out['signal']}): {rise_txt}; put back down"
            + ("; released" if out["released"] else ", still holding"),
        )

    dropped = jaws_empty()
    if put_back:
        back = put_down("push: put the loose brick back")
        if back:
            notes.append(f"putting the brick back was refused ({back})")
    if dropped:
        return finish(
            "not_holding",
            f"nothing held the lift and the jaws closed to their stop ({out['jaw_width'] * 1e3:.1f} mm): "
            f"the brick slipped out; {rise_txt}",
        )
    return finish(
        "not_connected",
        f"nothing held the lift: {rise_txt}; the brick came up with the hand (or the jaws slid on it)"
        + ("; put back on its seat, still holding" if put_back else "; still holding, lifted"),
    )
