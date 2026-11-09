"""Contract tests with scripted independent observations, not simulator acceptance.

The backend records commands but never infers object movement from commands.
Measured TCP/object trajectories are supplied independently by each test. Passing
these tests establishes control-flow and evidence handling, not physical skill
performance on YAM or Isaac.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pytest

from roboskillhub.runtime import engine
from roboskillhub.runtime.contracts import ExecutionFailed, Limits
from roboskillhub.runtime.engine import Runtime


def pose(x=0.0, y=0.0, z=0.3, frame="world"):
    return {
        "frame_id": frame,
        "position_m": [float(x), float(y), float(z)],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


class Clock:
    def __init__(self):
        self.elapsed = 0.0

    def time(self):
        return 1_700_000_000.0 + self.elapsed

    def monotonic(self):
        return self.elapsed

    def sleep(self, duration):
        self.elapsed += duration


@dataclass
class ScriptedBackend:
    clock: Clock
    measured_poses: list[dict] = field(default_factory=lambda: [pose()])
    # Separate camera observations indexed by completed motion count.
    observed_objects: dict[int, list[float]] = field(default_factory=dict)
    missing_object: str | None = None
    stale_object: str | None = None
    stale_pose: str | None = None
    holding: bool | None = True
    fail_move: int | None = None
    raise_move: int | None = None
    move_duration_s: float = 0.0
    capabilities: set[str] = field(
        default_factory=lambda: {
            "pose_feedback",
            "cartesian_motion",
            "gripper",
            "object_feedback",
            "stop",
        }
    )
    calls: list[tuple] = field(default_factory=list)
    motion_count: int = 0

    def describe(self):
        return {
            "name": "scripted-contract-test",
            "frame_id": "world",
            "capabilities": sorted(self.capabilities),
        }

    def _applies(self, phase):
        return (
            phase == "before"
            and self.motion_count == 0
            or phase == "after"
            and self.motion_count > 0
        )

    def get_state(self, *, timeout_s):
        self.clock.elapsed += 0.001
        now = self.clock.time()
        index = min(self.motion_count, len(self.measured_poses) - 1)
        timestamp = now - 10 if self._applies(self.stale_pose) else now
        state = {
            "pose": copy.deepcopy(self.measured_poses[index]),
            "timestamp_s": timestamp,
            "gripper": {
                "holding": self.holding,
                "timestamp_s": now,
                "source": "scripted-gripper-measurement",
            },
            "objects": {},
        }
        if not self._applies(self.missing_object):
            object_timestamp = now - 10 if self._applies(self.stale_object) else now
            candidates = [step for step in self.observed_objects if step <= self.motion_count]
            point = self.observed_objects[max(candidates)] if candidates else [0.0, 0.0, 0.1]
            state["objects"]["target"] = {
                "position_m": list(point),
                "timestamp_s": object_timestamp,
                "frame_id": "world",
                "source": "scripted-independent-camera-observation",
            }
        return state

    def move_pose(self, target, *, speed_m_s, timeout_s, linear):
        self.motion_count += 1
        self.calls.append(("move", copy.deepcopy(target), timeout_s, linear))
        self.clock.elapsed += self.move_duration_s
        if self.motion_count == self.raise_move:
            raise TimeoutError("transport timed out during movement")
        return {"ok": self.motion_count != self.fail_move}

    def set_gripper(self, width_m, *, timeout_s):
        self.calls.append(("gripper", width_m, timeout_s))
        return {"ok": True}

    def stop(self):
        self.calls.append(("stop",))


@pytest.fixture
def limits():
    return Limits.from_dict(
        {
            "frame_id": "world",
            "workspace_min_m": [-0.5, -0.5, 0.0],
            "workspace_max_m": [0.5, 0.5, 0.6],
            "max_speed_m_s": 0.1,
            "max_contact_distance_m": 0.2,
            "gripper_min_width_m": 0.0,
            "gripper_max_width_m": 0.08,
            "max_state_age_s": 0.5,
            "max_timeout_s": 5.0,
            "position_tolerance_m": 0.005,
        }
    )


@pytest.fixture
def clock(monkeypatch):
    timer = Clock()
    monkeypatch.setattr(engine, "time", timer)
    return timer


def grasp_request():
    return {
        "pregrasp_pose": pose(z=0.25),
        "grasp_pose": pose(z=0.15),
        "lift_pose": pose(z=0.20),
        "target_id": "target",
        "open_width_m": 0.07,
        "close_width_m": 0.01,
    }


def push_request():
    return {
        "approach_pose": pose(x=-0.05, z=0.2),
        "contact_pose": pose(z=0.15),
        "target_id": "target",
        "direction_world": [1, 0, 0],
        "distance_m": 0.04,
        "step_m": 0.02,
    }


def drawer_request():
    return {
        "pregrasp_pose": pose(x=-0.05, z=0.2),
        "grasp_pose": pose(z=0.15),
        "target_id": "target",
        "direction_world": [1, 0, 0],
        "distance_m": 0.04,
        "step_m": 0.02,
        "open_width_m": 0.07,
        "close_width_m": 0.01,
        "release_on_success": True,
    }


def contact_measurements():
    return [pose(), pose(x=-0.05, z=0.2), pose(z=0.15), pose(x=0.02, z=0.15), pose(x=0.04, z=0.15)]


@pytest.mark.parametrize("problem", ["frame", "workspace", "nan", "unknown", "quaternion"])
@pytest.mark.parametrize("capability", ["move_pose", "grasp", "push", "pull_drawer"])
def test_invalid_complete_request_is_rejected_before_any_actuation(
    clock, limits, capability, problem
):
    request = {
        "move_pose": lambda: {"target_pose": pose()},
        "grasp": grasp_request,
        "push": push_request,
        "pull_drawer": drawer_request,
    }[capability]()
    key = {
        "move_pose": "target_pose",
        "grasp": "lift_pose",
        "push": "contact_pose",
        "pull_drawer": "grasp_pose",
    }[capability]
    if problem == "frame":
        request[key]["frame_id"] = "camera"
    elif problem == "workspace":
        request[key]["position_m"][0] = 1.0
    elif problem == "nan":
        request[key]["position_m"][0] = float("nan")
    elif problem == "quaternion":
        request[key]["quaternion_xyzw"] = [0, 0, 0, 0]
    else:
        request["ignore_safety"] = True
    backend = ScriptedBackend(clock)
    result = Runtime(backend, limits).invoke(capability, request)
    assert result["status"] == "rejected"
    assert backend.calls == []


@pytest.mark.parametrize(
    "capability,factory", [("push", push_request), ("pull_drawer", drawer_request)]
)
def test_out_of_workspace_contact_endpoint_is_rejected_before_approach(
    clock, limits, capability, factory
):
    request = factory()
    key = "contact_pose" if capability == "push" else "grasp_pose"
    # The contact pose itself is legal; its derived endpoint is not.
    request[key]["position_m"][0] = 0.49
    backend = ScriptedBackend(clock)
    result = Runtime(backend, limits).invoke(capability, request)
    assert result["status"] == "rejected"
    assert backend.calls == []


def test_move_acknowledgement_without_measured_arrival_fails(clock, limits):
    backend = ScriptedBackend(clock, measured_poses=[pose(), pose(x=0.0)])
    result = Runtime(backend, limits).invoke("move_pose", {"target_pose": pose(x=0.1)})
    assert result["status"] == "failed"
    assert [call[0] for call in backend.calls] == ["move", "stop"]


def test_grasp_tcp_arrival_and_holding_flag_are_insufficient_when_object_did_not_lift(
    clock, limits
):
    backend = ScriptedBackend(
        clock,
        measured_poses=[pose(), pose(z=0.25), pose(z=0.15), pose(z=0.20)],
        observed_objects={0: [0, 0, 0.1], 3: [0, 0, 0.1]},
        holding=True,
    )
    result = Runtime(backend, limits).invoke("grasp", grasp_request())
    assert result["status"] == "failed"
    outcome = next(e for e in result["trace"] if e["event"] == "outcome_verification")
    assert outcome["observed_displacement_m"] == [0, 0, 0]
    assert backend.calls[-1] == ("stop",)


@pytest.mark.parametrize("failure_kind", ["negative_ack", "transport_exception", "pose_miss"])
def test_failed_pregrasp_never_closes_gripper_or_continues(clock, limits, failure_kind):
    backend = ScriptedBackend(clock, measured_poses=[pose(), pose(z=0.25)])
    if failure_kind == "negative_ack":
        backend.fail_move = 1
    elif failure_kind == "transport_exception":
        backend.raise_move = 1
    else:
        backend.measured_poses = [pose(), pose(x=0.2, z=0.25)]
    result = Runtime(backend, limits).invoke("grasp", grasp_request())
    assert result["status"] == "failed"
    assert [call[0] for call in backend.calls] == ["gripper", "move", "stop"]
    assert backend.calls[0][1] == 0.07  # only the initial opening


@pytest.mark.parametrize(
    "displacement,status",
    [
        ([0.04, 0, 0], "succeeded"),
        ([0, 0, 0], "failed"),
        ([-0.04, 0, 0], "failed"),
        ([0.04, 0.04, 0], "failed"),
    ],
)
def test_push_outcome_uses_independently_observed_distance_and_direction(
    clock, limits, displacement, status
):
    backend = ScriptedBackend(
        clock,
        measured_poses=contact_measurements(),
        observed_objects={
            0: [0, 0, 0.1],
            4: [displacement[0], displacement[1], 0.1 + displacement[2]],
        },
    )
    result = Runtime(backend, limits).invoke("push", push_request())
    assert result["status"] == status
    assert backend.motion_count == 4
    outcome = next(e for e in result["trace"] if e["event"] == "outcome_verification")
    assert outcome["observed_displacement_m"] == displacement
    assert not [call for call in backend.calls if call[0] == "gripper"]


@pytest.mark.parametrize("failure", ["motion", "object_stationary", "object_missing"])
def test_drawer_failure_never_releases_even_when_success_release_was_requested(
    clock, limits, failure
):
    backend = ScriptedBackend(
        clock, measured_poses=contact_measurements(), observed_objects={0: [0, 0, 0.1]}
    )
    if failure == "motion":
        backend.fail_move = 3
    elif failure == "object_missing":
        backend.missing_object = "after"
    result = Runtime(backend, limits).invoke("pull_drawer", drawer_request())
    assert result["status"] in {"failed", "unverified"}
    widths = [call[1] for call in backend.calls if call[0] == "gripper"]
    assert widths == [0.07, 0.01]  # no release after the handle close
    assert backend.calls[-1] == ("stop",)


@pytest.mark.parametrize("evidence_problem", ["missing_object", "stale_object", "stale_pose"])
@pytest.mark.parametrize("phase,expected", [("before", "rejected"), ("after", "unverified")])
def test_unavailable_evidence_before_and_after_motion_has_distinct_status(
    clock, limits, evidence_problem, phase, expected
):
    backend = ScriptedBackend(
        clock,
        measured_poses=contact_measurements(),
        observed_objects={0: [0, 0, 0.1], 4: [0.04, 0, 0.1]},
    )
    setattr(backend, evidence_problem, phase)
    result = Runtime(backend, limits).invoke("push", push_request())
    assert result["status"] == expected
    if phase == "before":
        assert backend.calls == []
    else:
        assert backend.calls[-1] == ("stop",)


@pytest.mark.parametrize("timeout_mode", ["driver_exception", "late_return"])
def test_single_motion_timeout_stops_once_and_does_not_continue(clock, limits, timeout_mode):
    backend = ScriptedBackend(clock, measured_poses=[pose(), pose(x=0.1)])
    if timeout_mode == "driver_exception":
        backend.raise_move = 1
    else:
        backend.move_duration_s = 0.25
    result = Runtime(backend, limits).invoke(
        "move_pose", {"target_pose": pose(x=0.1), "timeout_s": 0.1}
    )
    assert result["status"] == "failed"
    assert [call[0] for call in backend.calls] == ["move", "stop"]
    assert 0 < backend.calls[0][2] <= 0.1


def test_missing_backend_capability_is_rejected_before_any_actuation(clock, limits):
    backend = ScriptedBackend(clock, capabilities={"pose_feedback", "cartesian_motion", "stop"})
    result = Runtime(backend, limits).invoke("grasp", grasp_request())
    assert result["status"] == "rejected"
    assert backend.calls == []


def test_grasp_must_observe_object_follow_the_lift_not_an_earlier_approach_displacement(
    clock, limits
):
    backend = ScriptedBackend(
        clock,
        measured_poses=[pose(), pose(z=0.25), pose(z=0.15), pose(z=0.20)],
        # The object moved before the lift, then stayed still as the TCP rose.
        observed_objects={0: [0, 0, 0.10], 2: [0, 0, 0.15], 3: [0, 0, 0.15]},
        holding=None,
    )
    result = Runtime(backend, limits).invoke("grasp", grasp_request())
    assert result["status"] == "failed"
    assert backend.calls[-1] == ("stop",)


def test_feedback_acquired_during_motion_is_not_post_completion_evidence(clock, limits):
    class MidMotionState(ScriptedBackend):
        def move_pose(self, *args, **kwargs):
            self.last_mid_motion_sample = self.clock.time() + self.move_duration_s / 2
            return super().move_pose(*args, **kwargs)

        def get_state(self, *, timeout_s):
            state = super().get_state(timeout_s=timeout_s)
            if self.motion_count:
                state["timestamp_s"] = self.last_mid_motion_sample
            return state

    backend = MidMotionState(clock, measured_poses=[pose(), pose(x=0.1)], move_duration_s=0.1)
    result = Runtime(backend, limits).invoke("move_pose", {"target_pose": pose(x=0.1)})
    assert result["status"] == "unverified"
    assert backend.calls[-1] == ("stop",)


def test_stale_gripper_state_does_not_gate_a_new_grasp(clock, limits):
    class OldGripperState(ScriptedBackend):
        def get_state(self, *, timeout_s):
            state = super().get_state(timeout_s=timeout_s)
            if any(c[0] == "gripper" and c[1] == 0.01 for c in self.calls):
                state["gripper"]["timestamp_s"] -= 0.05
            return state

    req = grasp_request()
    backend = OldGripperState(
        clock, measured_poses=[pose(), req["pregrasp_pose"], req["grasp_pose"]], holding=False
    )
    result = Runtime(backend, limits).invoke("grasp", req)
    assert result["status"] == "unverified"
    assert backend.motion_count == 2
    assert backend.calls[-1] == ("stop",)


def test_recent_pre_ack_samples_are_polled_until_a_new_pose_sample_arrives(clock, limits):
    class DelayedPosePublisher(ScriptedBackend):
        post_motion_reads = 0
        returned_timestamps = []

        def move_pose(self, *args, **kwargs):
            reply = super().move_pose(*args, **kwargs)
            self.ack_timestamp = self.clock.time()
            return reply

        def get_state(self, *, timeout_s):
            state = super().get_state(timeout_s=timeout_s)
            if self.motion_count:
                self.post_motion_reads += 1
                if self.post_motion_reads <= 2:
                    state["timestamp_s"] = self.ack_timestamp
                self.returned_timestamps.append(state["timestamp_s"])
            return state

    backend = DelayedPosePublisher(clock, measured_poses=[pose(), pose(x=0.1)])
    result = Runtime(backend, limits).invoke("move_pose", {"target_pose": pose(x=0.1)})
    assert result["status"] == "succeeded"
    assert backend.post_motion_reads == 3
    assert backend.returned_timestamps[:2] == [backend.ack_timestamp] * 2
    assert result["evidence"]["state_timestamp_s"] == backend.returned_timestamps[-1]
    assert result["evidence"]["state_timestamp_s"] > backend.ack_timestamp
    assert [call[0] for call in backend.calls] == ["move"]


def test_object_publisher_must_also_advance_even_when_pose_feedback_is_new(clock, limits):
    class DelayedObjectPublisher(ScriptedBackend):
        final_reads = 0

        def move_pose(self, *args, **kwargs):
            reply = super().move_pose(*args, **kwargs)
            self.ack_timestamp = self.clock.time()
            return reply

        def get_state(self, *, timeout_s):
            state = super().get_state(timeout_s=timeout_s)
            if self.motion_count == 4:
                self.final_reads += 1
                if self.final_reads <= 3:
                    state["objects"]["target"]["timestamp_s"] = self.ack_timestamp
            return state

    backend = DelayedObjectPublisher(
        clock,
        measured_poses=contact_measurements(),
        observed_objects={0: [0, 0, 0.1], 4: [0.04, 0, 0.1]},
    )
    result = Runtime(backend, limits).invoke("push", push_request())
    assert result["status"] == "succeeded"
    assert backend.final_reads == 4  # one pose read, then three object-evidence reads
    assert result["evidence"]["final"]["timestamp_s"] > backend.ack_timestamp
    assert backend.motion_count == 4
    assert [event for event in result["trace"] if event["event"] == "feedback_wait"][0][
        "target_id"
    ] == "target"


@pytest.mark.parametrize("delayed_field", ["pose", "object"])
def test_persistent_recent_old_samples_expire_as_unverified_without_motion_retry(
    clock, limits, delayed_field
):
    class FrozenPublisher(ScriptedBackend):
        post_motion_reads = 0
        read_budgets = []

        def move_pose(self, *args, **kwargs):
            reply = super().move_pose(*args, **kwargs)
            self.ack_timestamp = self.clock.time()
            return reply

        def get_state(self, *, timeout_s):
            self.read_budgets.append(timeout_s)
            state = super().get_state(timeout_s=timeout_s)
            # Freeze pose after the only move, or object after the final push segment.
            if self.motion_count == (1 if delayed_field == "pose" else 4):
                self.post_motion_reads += 1
                if delayed_field == "pose":
                    state["timestamp_s"] = self.ack_timestamp
                else:
                    state["objects"]["target"]["timestamp_s"] = self.ack_timestamp
            return state

    backend = FrozenPublisher(
        clock,
        measured_poses=[pose(), pose(x=0.1)] if delayed_field == "pose" else contact_measurements(),
        observed_objects={0: [0, 0, 0.1], 4: [0.04, 0, 0.1]},
    )
    capability = "move_pose" if delayed_field == "pose" else "push"
    request = {"target_pose": pose(x=0.1)} if delayed_field == "pose" else push_request()
    result = Runtime(backend, limits).invoke(capability, {**request, "timeout_s": 0.1})
    assert result["status"] == "unverified"
    assert result["error"]["code"] == "EvidenceMissing"
    assert "deadline" in result["error"]["message"]
    assert backend.post_motion_reads > 1
    assert backend.motion_count == (1 if delayed_field == "pose" else 4)
    assert backend.calls[-1] == ("stop",)
    assert all(
        earlier > later
        for earlier, later in zip(backend.read_budgets, backend.read_budgets[1:], strict=False)
    )
    assert result["elapsed_s"] <= 0.101


@pytest.mark.parametrize("final_holding,status", [(False, "failed"), (None, "succeeded")])
def test_lift_final_holding_feedback_can_veto_object_following_but_unknown_is_allowed(
    clock, limits, final_holding, status
):
    class LiftHoldingPublisher(ScriptedBackend):
        def get_state(self, *, timeout_s):
            state = super().get_state(timeout_s=timeout_s)
            if self.motion_count == 3:
                state["gripper"]["holding"] = final_holding
            return state

    backend = LiftHoldingPublisher(
        clock,
        measured_poses=[pose(), pose(z=0.25), pose(z=0.15), pose(z=0.20)],
        observed_objects={0: [0, 0, 0.10], 2: [0, 0, 0.10], 3: [0, 0, 0.15]},
        holding=True,
    )
    result = Runtime(backend, limits).invoke("grasp", grasp_request())
    assert result["status"] == status
    assert backend.motion_count == 3
    last_check = [event for event in result["trace"] if event["event"] == "gripper_verification"][
        -1
    ]
    assert last_check["feedback"]["holding"] is final_holding
    if final_holding is False:
        assert result["error"]["code"] == "ExecutionFailed"
        assert "no held object" in result["error"]["message"]
        assert backend.calls[-1] == ("stop",)
    else:
        assert result["evidence"]["criterion"] == "object-follows-lift"


def test_state_callback_exhausting_remaining_poll_budget_is_unverified(clock, limits):
    class BlockingPublisher(ScriptedBackend):
        post_motion_reads = 0

        def move_pose(self, *args, **kwargs):
            reply = super().move_pose(*args, **kwargs)
            self.ack_timestamp = self.clock.time()
            return reply

        def get_state(self, *, timeout_s):
            if self.motion_count:
                self.post_motion_reads += 1
                if self.post_motion_reads > 1:
                    self.clock.elapsed += timeout_s + 0.001
                    raise ExecutionFailed("state callback exceeded its deadline")
            state = super().get_state(timeout_s=timeout_s)
            if self.motion_count:
                state["timestamp_s"] = self.ack_timestamp
            return state

    backend = BlockingPublisher(clock, measured_poses=[pose(), pose(x=0.1)])
    result = Runtime(backend, limits).invoke(
        "move_pose", {"target_pose": pose(x=0.1), "timeout_s": 0.1}
    )
    assert result["status"] == "unverified"
    assert result["error"]["code"] == "EvidenceMissing"
    assert [call[0] for call in backend.calls] == ["move", "stop"]


def test_queued_request_expires_and_never_executes_after_owner_releases(clock, limits):
    import threading
    import time as real_time

    backend = ScriptedBackend(clock)
    runtime = Runtime(backend, limits)
    runtime._lock.acquire()
    results = []
    worker = threading.Thread(
        target=lambda: results.append(
            runtime.invoke("move_pose", {"target_pose": pose(), "timeout_s": 0.1})
        )
    )
    start = real_time.monotonic()
    worker.start()
    worker.join(timeout=1)
    runtime._lock.release()
    worker.join(timeout=1)
    assert real_time.monotonic() - start < 0.8
    assert results[0]["status"] == "rejected"
    assert results[0]["error"]["code"] == "QueueTimeout"
    assert backend.calls == []


def test_waiting_for_lock_consumes_original_invocation_budget(clock, limits):
    class DelayedLock:
        def acquire(self, *, timeout):
            assert timeout == pytest.approx(0.1)
            clock.elapsed += 0.11
            return True

        def release(self):
            self.released = True

    backend = ScriptedBackend(clock)
    runtime = Runtime(backend, limits)
    runtime._lock = DelayedLock()
    result = runtime.invoke("move_pose", {"target_pose": pose(), "timeout_s": 0.1})
    assert result["status"] == "rejected"
    assert "deadline" in result["error"]["message"]
    assert backend.calls == []
    assert runtime._lock.released


@pytest.mark.parametrize("fail_move", [None, 1])
def test_backend_ownership_covers_observation_motion_and_failure_stop(clock, limits, fail_move):
    class OwnedBackend(ScriptedBackend):
        owned = False
        order = []

        def begin_invocation(self, *, timeout_s):
            assert timeout_s > 0
            self.owned = True
            self.order.append("begin")

        def get_state(self, **kwargs):
            assert self.owned
            self.order.append("state")
            return super().get_state(**kwargs)

        def move_pose(self, *args, **kwargs):
            assert self.owned
            self.order.append("move")
            return super().move_pose(*args, **kwargs)

        def stop(self):
            assert self.owned
            self.order.append("stop")
            super().stop()

        def end_invocation(self):
            assert self.owned
            self.order.append("end")
            self.owned = False

    backend = OwnedBackend(clock, fail_move=fail_move)
    result = Runtime(backend, limits).invoke("move_pose", {"target_pose": pose()})
    assert result["status"] == ("succeeded" if fail_move is None else "failed")
    assert backend.order[0] == "begin" and backend.order[-1] == "end"
    assert not backend.owned
    if fail_move:
        assert backend.order[-2] == "stop"


def test_failed_lease_claim_never_releases_another_owners_lease(clock, limits):
    from roboskillhub.runtime.contracts import Rejected

    class BusyBackend(ScriptedBackend):
        def begin_invocation(self, *, timeout_s):
            raise Rejected("station already owned by another invocation")

        def end_invocation(self):
            raise AssertionError("must not release another owner")

    backend = BusyBackend(clock)
    result = Runtime(backend, limits).invoke("move_pose", {"target_pose": pose()})
    assert result["status"] == "rejected"
    assert backend.calls == []


def test_release_failure_is_visible_even_after_measured_motion_success(clock, limits):
    class LostRelease(ScriptedBackend):
        def begin_invocation(self, *, timeout_s):
            pass

        def end_invocation(self):
            raise ConnectionError("owner release reply lost")

    result = Runtime(LostRelease(clock), limits).invoke("move_pose", {"target_pose": pose()})
    assert result["status"] == "unverified"
    assert "reply lost" in result["ownership_error"]
    assert result["evidence"]["position_error_m"] == 0
