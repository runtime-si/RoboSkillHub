"""Scripted test doubles for the LEGO ports. Evidence from these is ``unit_or_dryrun``;
they model no physics and must never be counted as simulation or hardware results."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import geometry as g


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeMotion:
    """Records commanded TCP poses; ``fail_on`` makes the n-th move fail."""

    def __init__(self, clock: FakeClock, *, move_s: float = 0.2, fail_on: int | None = None):
        self.clock, self.move_s, self.fail_on = clock, move_s, fail_on
        self.commands: list[dict] = []
        self.stops = 0

    def describe(self) -> dict[str, Any]:
        return {"backend_id": "fake-motion", "evidence_level": "unit_or_dryrun"}

    def move_pose(self, pose: dict, *, speed_m_s: float, timeout_s: float) -> dict[str, Any]:
        self.commands.append(pose)
        self.clock.advance(self.move_s)
        if self.fail_on is not None and len(self.commands) == self.fail_on:
            return {"ok": False, "error": "scripted failure"}
        return {"ok": True}

    def stop(self) -> dict[str, Any]:
        self.stops += 1
        return {"confirmed": True}


class FakeObserver:
    """Reports the held brick where the last commanded TCP puts it, via a *true*
    ``T_TCP_object`` that may differ from the one the skill was told (a grasp offset), plus
    a scripted per-call perturbation. Labelled ``unit_or_dryrun``."""

    def __init__(self, clock: FakeClock, motion: FakeMotion, t_tcp_object_true: dict, *,
                 sigma_m: float = 0.0001, source: str = "fake-wrist-camera",
                 kind: str = "measured", age_s: float = 0.0,
                 perturb: Callable[[int, g.Matrix], g.Matrix] | None = None,
                 visible: bool = True, occluded: bool = False, frame_id: str = "world"):
        self.clock, self.motion = clock, motion
        self.t_true = g.matrix(t_tcp_object_true)
        self.sigma_m, self.source, self.kind, self.age_s = sigma_m, source, kind, age_s
        self.perturb, self.visible, self.occluded = perturb, visible, occluded
        self.frame_id = frame_id
        self.calls = 0

    def describe(self) -> dict[str, Any]:
        return {"observer_id": self.source, "evidence_level": "unit_or_dryrun",
                "uncertainty_m": self.sigma_m}

    def observe(self, object_id: str, *, not_before_s: float, timeout_s: float) -> dict:
        self.calls += 1
        self.clock.advance(0.05)
        value = None
        if self.visible and self.motion.commands:
            t = g.mul(g.matrix(self.motion.commands[-1]), self.t_true)
            if self.perturb is not None:
                t = self.perturb(self.calls, t)
            value = g.to_pose(t, self.frame_id)
        return {"obs_id": f"obs-{self.calls}", "source": self.source, "kind": self.kind,
                "frame_id": self.frame_id, "timestamp_s": self.clock() - self.age_s,
                "value": value, "uncertainty_m": self.sigma_m, "uncertainty_deg": 0.05,
                "occluded": self.occluded, "privileged": self.kind == "sim_truth"}


class FakeContact:
    """Replays a scripted contact trace; counts guarded_move calls."""

    def __init__(self, *, capabilities: list[str] | None = None, samples: list[dict] | None = None,
                 stop_reason: str = "force_threshold", stop_confirmed: bool = True,
                 controller_kind: str = "admittance", max_force_N: float = 30.0,
                 evidence_level: str = "unit_or_dryrun", rate_hz: float = 500.0):
        self.capabilities = sorted(
            capabilities if capabilities is not None else
            ["pose_feedback", "contact_force_feedback", "guarded_motion", "stop_confirmation"])
        self.samples = samples or []
        self.stop_reason, self.stop_confirmed = stop_reason, stop_confirmed
        self.controller_kind, self.max_force_N = controller_kind, max_force_N
        self.evidence_level, self.rate_hz = evidence_level, rate_hz
        self.calls: list[dict] = []

    def describe(self) -> dict[str, Any]:
        return {
            "backend_id": "fake-contact",
            "evidence_level": self.evidence_level,
            "capabilities": self.capabilities,
            "contact_signal": {"source": "scripted", "quantity": "axial_force", "unit": "N",
                               "frame_id": "tcp", "rate_hz": self.rate_hz,
                               "max_latency_s": 0.01, "calibration_id": "fake-cal"},
            "controller": {"kind": self.controller_kind, "loop_hz": 500, "process": "fake"},
            "limits": {"max_force_N": self.max_force_N, "max_speed_m_s": 0.02,
                       "max_travel_m": 0.03},
        }

    def guarded_move(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        return {"command_issued": True, "stop_reason": self.stop_reason,
                "stop_confirmed": self.stop_confirmed, "samples": self.samples, "fault": ""}

    def stop(self) -> dict[str, Any]:
        return {"confirmed": self.stop_confirmed}


def press_trace(seat_travel_m: float, stud_height_m: float, *, final_shortfall_m: float = 0.0,
                peak_N: float = 10.0, onset_travel_m: float | None = None, dt: float = 0.002,
                step_m: float = 0.0002) -> list[dict]:
    """A synthetic force/travel trace: free motion until ``onset`` (default the stud tops),
    then force rising linearly to ``peak_N`` at the final travel."""
    onset = seat_travel_m - stud_height_m if onset_travel_m is None else onset_travel_m
    final = seat_travel_m - final_shortfall_m
    out, t, x = [], 0.0, 0.0
    while x < final - 1e-12:
        f = 0.0 if x < onset else peak_N * (x - onset) / max(final - onset, 1e-9)
        out.append({"t_s": t, "travel_m": x, "force_N": f})
        x = min(final, x + step_m)
        t += dt
    out.append({"t_s": t, "travel_m": final, "force_N": peak_N})
    return out
