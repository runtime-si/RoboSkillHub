# Adapted from uni-robot-agent (https://github.com/Open-RUA/uni-robot-agent),
# uni_robot_agent/tools/gripper/opening.py (OpeningTracker, native_opening) at
# a98934c6a1a16317821fed41a3b816b1d300b427, MIT, Copyright (c) 2026
# uni-robot-agent contributors.
# Changes: the skill commands the gap itself (robot.set_grip, or
# robot.close_gripper / robot.open_gripper at the ends of travel) and polls
# robot.get_gripper instead of being fed arm metadata; the gap in metres comes
# from robot.describe_gripper's width fit (upstream: an affine calibration
# record); the target is clipped to the hand's span as upstream clipped to its
# calibrated range; a stall is split by direction and progress into
# stalled_on_object (closing, stopped wide of the target) and stalled_jammed
# (opening blocked, or no travel at all), with the stall gap compared to an
# expected part width; a window that never settles routes still_moving
# (upstream stayed "pending").
"""Command a jaw gap and watch it: reached, stopped on something, jammed, or still moving."""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from gap import NodeContext

logger = logging.getLogger(__name__)

TOLERANCE_M = 0.002
"""Upstream OpeningTracker.tolerance_m: the gap counts as reached within 2 mm."""

PROGRESS_M = 0.0005
"""Upstream OpeningTracker.progress_m: less than 0.5 mm gained over the window is a stall."""

WINDOW = 8
"""Upstream OpeningTracker.window_steps: samples in the stall window."""

POLL_STEPS = 50
"""Sim steps between samples (0.1 s at 500 Hz), so the 8-sample window spans 0.7 s. Upstream
counted control steps; in the YAM twin robot.set_grip moves the jaws ~7 mm/s, and with 10
steps per sample (a 0.14 s window) a slow opening read as "did not move" (sim check)."""

POLL_S = 0.1
"""Wall-clock seconds between samples when the connector cannot step (a real station)."""

MIN_TRAVEL_M = 0.001
"""Travel from the starting gap under which a stall counts as jammed, not as stopped on
an object: a drive that never moved did not close on anything."""

WIDTH_TOLERANCE_M = 0.004
"""Stall gap vs expected part width. Sim check: a 25 mm cube read 26.4-26.7 mm (pads
and the span-linear width fit), so 4 mm."""


class Output(TypedDict):
    route: str
    target_m: float
    start_m: float
    actual_m: float
    error_m: float
    object_width_m: float
    width_error_m: float
    samples: int
    trace_m: list[float]
    reason: str


def gap_m(ctx: NodeContext, gripper: dict[str, Any], arm: dict[str, Any]) -> float | None:
    """The jaw gap in metres: ``width_m`` if the connector reports it, else the read fraction
    (1.0 = open) through describe_gripper's width fit, as verifying-grasps reads it."""
    reply = ctx.tool("robot.get_gripper", **arm) or {}
    for key in ("width_m", "opening_m", "gripper_opening_m"):
        if reply.get(key) is not None:
            return float(reply[key])
    if reply.get("position") is None:
        return None
    fraction = max(0.0, min(1.0, float(reply["position"])))
    fit = gripper.get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or gripper.get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)
    return at_closed + fraction * (at_open - at_closed)


class OpeningTracker:
    """Upstream's tracker: a target gap, the last reading, a window of absolute errors."""

    def __init__(
        self,
        target_m: float,
        tolerance_m: float = TOLERANCE_M,
        progress_m: float = PROGRESS_M,
        window: int = WINDOW,
    ):
        self.target_m = float(target_m)
        self.tolerance_m = float(tolerance_m)
        self.progress_m = float(progress_m)
        self.window = int(window)
        self.actual_m: float | None = None
        self.errors: list[float] = []

    def update(self, value: float | None) -> None:
        self.actual_m = value
        if value is None:
            self.errors.clear()
        else:
            self.errors.append(abs(value - self.target_m))
            self.errors = self.errors[-self.window :]

    @property
    def reached(self) -> bool | None:
        return None if self.actual_m is None else abs(self.actual_m - self.target_m) <= self.tolerance_m

    @property
    def stalled(self) -> bool:
        return (
            self.reached is False
            and len(self.errors) == self.window
            and self.errors[0] - min(self.errors[1:]) <= self.progress_m
        )

    @property
    def status(self) -> str:
        if self.reached is None:
            return "unknown"
        if self.reached:
            return "reached"
        return "stalled" if self.stalled else "pending"


def _refused(reply: Any) -> str:
    if not isinstance(reply, dict):
        return ""
    if reply.get("success") is False or reply.get("reached") is False:
        return str(reply.get("reason") or reply.get("message") or "the gripper command reported failure")
    return ""


def run(
    ctx: NodeContext,
    target_width_m: float | None = None,
    close: bool = False,
    command: bool = True,
    expected_width_m: float | None = None,
    width_tolerance_m: float = WIDTH_TOLERANCE_M,
    tolerance_m: float = TOLERANCE_M,
    progress_m: float = PROGRESS_M,
    window: int = WINDOW,
    max_samples: int = 100,
    poll_steps: int = POLL_STEPS,
    poll_s: float = POLL_S,
    arm_id: int = 0,
) -> Output:
    """Drive the jaws to ``target_width_m`` (or fully closed with ``close``) and track the gap.

    ``command=False`` only watches a move the graph already commanded. ``expected_width_m``
    is the part's width across the jaws; a stall farther than ``width_tolerance_m`` from it
    routes ``unexpected_width`` (wrong part, edge or corner grasp, two parts).
    """
    arm = {"arm_id": arm_id} if arm_id else {}
    gripper = ctx.tool("robot.describe_gripper", **arm) or {}
    fit = gripper.get("width_fit") or {}
    span = float(fit.get("at_open_m") or gripper.get("span_m") or 0.0)
    floor = float(fit.get("at_closed_m") or gripper.get("min_grasp_width_m") or 0.0)
    if close:
        target = floor
    elif target_width_m is None:
        raise ValueError("give target_width_m or close=True")
    else:
        target = float(target_width_m)
    if span > floor:
        target = max(floor, min(span, target))

    def out(route: str, why: str, tracker: OpeningTracker, start: float, trace: list[float]) -> Output:
        actual = -1.0 if tracker.actual_m is None else float(tracker.actual_m)
        width = actual if route in ("stalled_on_object", "unexpected_width") else -1.0
        werr = (
            width - float(expected_width_m) if width >= 0 and expected_width_m is not None else 0.0
        )
        logger.info(
            "tracking-gripper-opening-stall: %s target=%.4f actual=%.4f (%s)", route, target, actual, why
        )
        return {
            "route": route,
            "target_m": round(target, 5),
            "start_m": round(start, 5),
            "actual_m": round(actual, 5),
            "error_m": round(abs(actual - target), 5) if actual >= 0 else -1.0,
            "object_width_m": round(width, 5),
            "width_error_m": round(werr, 5),
            "samples": len(trace),
            "trace_m": [round(v, 5) for v in trace],
            "reason": why,
        }

    tracker = OpeningTracker(target, tolerance_m, progress_m, window)
    try:
        start = gap_m(ctx, gripper, arm)
    except Exception as exc:  # noqa: BLE001  (unserved / unreadable gripper)
        return out("unknown", f"the gripper could not be read ({exc})", tracker, -1.0, [])
    if start is None:
        return out("unknown", "robot.get_gripper returned no position", tracker, -1.0, [])

    if command:
        try:
            if close or target <= floor:
                why = _refused(ctx.tool("robot.close_gripper", **arm))
            elif span > floor and target >= span:
                why = _refused(ctx.tool("robot.open_gripper", **arm))
            else:
                why = _refused(ctx.tool("robot.set_grip", width_m=target, **arm))
        except Exception as exc:  # noqa: BLE001  (a real station raises on a refused command)
            why = str(exc) or type(exc).__name__
        if why:
            tracker.update(start)
            return out("refused", f"the gripper command was refused: {why}", tracker, start, [start])

    trace: list[float] = []
    for i in range(int(max_samples)):
        if i:
            try:
                ctx.tool("robot.wait_steps", steps=int(poll_steps))
            except Exception:  # noqa: BLE001  (a real connector has no stepping: wall-clock poll)
                time.sleep(float(poll_s))
        try:
            value = gap_m(ctx, gripper, arm)
        except Exception as exc:  # noqa: BLE001
            return out("unknown", f"the gripper could not be read ({exc})", tracker, start, trace)
        tracker.update(value)
        if value is None:
            return out("unknown", "robot.get_gripper returned no position", tracker, start, trace)
        trace.append(value)
        status = tracker.status
        if status == "reached":
            return out("reached", f"within {tolerance_m * 1e3:.1f} mm of the target", tracker, start, trace)
        if status != "stalled":
            continue
        travel = abs(value - start)
        closing = target < start
        if travel < MIN_TRAVEL_M:
            return out(
                "stalled_jammed",
                f"the jaws did not move ({travel * 1e3:.1f} mm from the start)",
                tracker,
                start,
                trace,
            )
        if not closing or value < target:
            where = "short of" if not closing and value < target else "past"
            return out(
                "stalled_jammed",
                f"{'opening' if not closing else 'closing'} settled {abs(value - target) * 1e3:.1f} mm "
                f"{where} the target",
                tracker,
                start,
                trace,
            )
        if expected_width_m is not None and abs(value - float(expected_width_m)) > float(width_tolerance_m):
            return out(
                "unexpected_width",
                f"stopped on something {value * 1e3:.1f} mm wide, expected "
                f"{float(expected_width_m) * 1e3:.1f} +- {float(width_tolerance_m) * 1e3:.1f} mm",
                tracker,
                start,
                trace,
            )
        return out(
            "stalled_on_object",
            f"closing stopped at {value * 1e3:.1f} mm, {abs(value - target) * 1e3:.1f} mm wide of the target",
            tracker,
            start,
            trace,
        )
    return out(
        "still_moving",
        f"neither reached nor stalled after {len(trace)} samples",
        tracker,
        start,
        trace,
    )
