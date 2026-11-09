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

    def __init__(
        self,
        clock: FakeClock,
        motion: FakeMotion,
        t_tcp_object_true: dict,
        *,
        sigma_m: float = 0.0001,
        source: str = "fake-wrist-camera",
        kind: str = "measured",
        age_s: float = 0.0,
        perturb: Callable[[int, g.Matrix], g.Matrix] | None = None,
        visible: bool = True,
        occluded: bool = False,
        frame_id: str = "world",
    ):
        self.clock, self.motion = clock, motion
        self.t_true = g.matrix(t_tcp_object_true)
        self.sigma_m, self.source, self.kind, self.age_s = sigma_m, source, kind, age_s
        self.perturb, self.visible, self.occluded = perturb, visible, occluded
        self.frame_id = frame_id
        self.calls = 0

    def describe(self) -> dict[str, Any]:
        return {
            "observer_id": self.source,
            "evidence_level": "unit_or_dryrun",
            "uncertainty_m": self.sigma_m,
        }

    def observe(self, object_id: str, *, not_before_s: float, timeout_s: float) -> dict:
        self.calls += 1
        self.clock.advance(0.05)
        value = None
        if self.visible and self.motion.commands:
            t = g.mul(g.matrix(self.motion.commands[-1]), self.t_true)
            if self.perturb is not None:
                t = self.perturb(self.calls, t)
            value = g.to_pose(t, self.frame_id)
        return {
            "obs_id": f"obs-{self.calls}",
            "source": self.source,
            "kind": self.kind,
            "frame_id": self.frame_id,
            "timestamp_s": self.clock() - self.age_s,
            "value": value,
            "uncertainty_m": self.sigma_m,
            "uncertainty_deg": 0.05,
            "occluded": self.occluded,
            "privileged": self.kind == "sim_truth",
        }


class FakeContact:
    """Replays a scripted contact trace; counts guarded_move calls."""

    def __init__(
        self,
        *,
        capabilities: list[str] | None = None,
        samples: list[dict] | None = None,
        stop_reason: str = "force_threshold",
        stop_confirmed: bool = True,
        controller_kind: str = "admittance",
        max_force_N: float = 30.0,
        evidence_level: str = "unit_or_dryrun",
        rate_hz: float = 500.0,
        insertion_force_observable: bool | None = None,
    ):
        self.capabilities = sorted(
            capabilities
            if capabilities is not None
            else ["pose_feedback", "contact_force_feedback", "compliant_motion", "stop"]
        )
        self.samples = samples or []
        self.stop_reason, self.stop_confirmed = stop_reason, stop_confirmed
        self.controller_kind, self.max_force_N = controller_kind, max_force_N
        self.evidence_level, self.rate_hz = evidence_level, rate_hz
        self.insertion_force_observable = insertion_force_observable
        self.calls: list[dict] = []

    def describe(self) -> dict[str, Any]:
        return {
            "backend_id": "fake-contact",
            "evidence_level": self.evidence_level,
            "capabilities": self.capabilities,
            "contact_signal": {
                "source": "scripted",
                "quantity": "axial_force",
                "unit": "N",
                "frame_id": "tcp",
                "rate_hz": self.rate_hz,
                "max_latency_s": 0.01,
                "calibration_id": "fake-cal",
                **(
                    {}
                    if self.insertion_force_observable is None
                    else {"insertion_force_observable": self.insertion_force_observable}
                ),
            },
            "controller": {"kind": self.controller_kind, "loop_hz": 500, "process": "fake"},
            "limits": {
                "max_force_N": self.max_force_N,
                "max_speed_m_s": 0.02,
                "max_travel_m": 0.03,
            },
        }

    def guarded_move(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        return {
            "command_issued": True,
            "stop_reason": self.stop_reason,
            "stop_confirmed": self.stop_confirmed,
            "samples": self.samples,
            "fault": "",
        }

    def stop(self) -> dict[str, Any]:
        return {"confirmed": self.stop_confirmed}


class FakeStation:
    """Scripted answers to contract primitives (move_pose, observe_objects, set_gripper and
    the requested guarded_press) for one held brick and a few placed ones. Kinematic
    bookkeeping only; no contact physics. ``scenario`` keys:

    - ``grasp_offset``: Matrix, true ``T_TCP_object`` = believed @ offset
    - ``press``: kwargs for ``press_trace`` (shortfall, peak, onset)
    - ``stop_reason`` / ``stop_confirmed``: for guarded_press
    - ``slip_m``: the brick slides this far up the jaws during the press
    - ``release_dz``: brick bottom offset (m) left after release
    - ``lifted``: the brick stays in the jaws through the retreat
    - ``disturb``: ``{brick_id: (dx, dz)}`` applied to placed bricks at the press
    - ``holding_after_release``: gripper feedback after release (default False)
    """

    def __init__(
        self,
        clock: FakeClock,
        *,
        tcp: dict,
        t_tcp_object: dict,
        held_id: str,
        placed: dict[str, dict] | None = None,
        capabilities: set[str] | None = None,
        contact_description: dict | None = None,
        sigma_m: float = 0.0001,
        stud_height_m: float = 0.0017,
        seat_travel_m: float = 0.01,
        scenario: dict | None = None,
    ):
        self.clock, self.held_id = clock, held_id
        self.tcp = g.matrix(tcp)
        self.frame = tcp["frame_id"]
        sc = scenario or {}
        self.sc = sc
        self.t_true = g.mul(g.matrix(t_tcp_object), sc.get("grasp_offset", g.tr(0, 0, 0)))
        self.attached = True
        self.brick: g.Matrix | None = None
        self.placed = {k: g.matrix(v) for k, v in (placed or {}).items()}
        self.capabilities = set(
            capabilities
            if capabilities is not None
            else {
                "pose_feedback",
                "cartesian_motion",
                "stop",
                "gripper",
                "gripper_feedback",
                "object_feedback",
                "object_pose_feedback",
                "contact_force_feedback",
                "compliant_motion",
            }
        )
        self.contact_description = contact_description
        self.sigma_m, self.stud_h, self.seat_travel = sigma_m, stud_height_m, seat_travel_m
        self.calls: list[tuple[str, dict]] = []
        self.n = 0

    def _result(self, status: str, evidence: dict, failure_type: str | None = None) -> dict:
        self.n += 1
        return {
            "status": status,
            "invocation_id": f"fake-{self.n}",
            "failure_type": failure_type,
            "evidence": evidence,
        }

    def _brick_world(self) -> g.Matrix | None:
        return g.mul(self.tcp, self.t_true) if self.attached else self.brick

    def primitive(self, capability: str, request: dict, rid: str) -> dict:
        self.calls.append((capability, request))
        self.clock.advance(0.05)
        if capability == "move_pose":
            self.tcp = g.matrix(request["target_pose"])
            return self._result("succeeded", {"measured_pose": g.to_pose(self.tcp, self.frame)})
        if capability == "observe_objects":
            obs = []
            for oid in request["object_ids"]:
                m = self._brick_world() if oid == self.held_id else self.placed.get(oid)
                obs.append(
                    {
                        "object_id": oid,
                        "frame_id": self.frame,
                        "position_m": None if m is None else [m[0][3], m[1][3], m[2][3]],
                        "quaternion_xyzw": None
                        if m is None
                        else g.to_pose(m, self.frame)["quaternion_xyzw"],
                        "timestamp_s": self.clock(),
                        "source": "fake_overhead",
                        "privileged": False,
                        "uncertainty": {"position_m": self.sigma_m, "rotation_rad": 0.001},
                        "visibility": "not_found" if m is None else "visible",
                    }
                )
            return self._result("succeeded", {"observations": obs})
        if capability == "guarded_press":
            if not {"contact_force_feedback", "compliant_motion"} <= self.capabilities:
                return self._result("rejected", {}, "capability_missing")
            kw = self.sc.get("press", {})
            samples = press_trace(self.seat_travel, self.stud_h, **kw)
            d = samples[-1]["travel_m"]
            ax = request["axis"]["vector"]
            self.tcp = g.mul(g.tr(ax[0] * d, ax[1] * d, ax[2] * d), self.tcp)
            if self.sc.get("slip_m"):  # the brick slid up the jaws instead of seating
                self.t_true = g.mul(self.t_true, g.tr(0, 0, self.sc["slip_m"]))
            for bid, (dx, dz) in self.sc.get("disturb", {}).items():
                self.placed[bid] = g.mul(self.placed[bid], g.tr(dx, 0, dz))
            return self._result(
                "succeeded",
                {
                    "samples": samples,
                    "stop_reason": self.sc.get("stop_reason", "force_threshold"),
                    "stop_confirmed": self.sc.get("stop_confirmed", True),
                },
            )
        if capability == "set_gripper":
            holding = self.sc.get("holding_after_release", False)
            if request["purpose"] == "release" and not holding and not self.sc.get("lifted"):
                self.brick = g.mul(
                    g.mul(self.tcp, self.t_true), g.tr(0, 0, self.sc.get("release_dz", 0.0))
                )
                self.attached = False
            if holding:
                return self._result(
                    "failed", {"gripper_feedback": {"holding": True}}, "release_failed"
                )
            return self._result("succeeded", {"gripper_feedback": {"holding": False}})
        return self._result("rejected", {}, "capability_missing")


def press_trace(
    seat_travel_m: float,
    stud_height_m: float,
    *,
    final_shortfall_m: float = 0.0,
    peak_N: float = 10.0,
    onset_travel_m: float | None = None,
    dt: float = 0.002,
    step_m: float = 0.0002,
) -> list[dict]:
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
