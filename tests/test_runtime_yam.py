from __future__ import annotations

import copy

import pytest

from roboskillhub.runtime.contracts import EvidenceMissing, ExecutionFailed, Limits, Rejected
from roboskillhub.runtime.yam import YamBackend, gap_pose_to_pose, pose_to_gap


@pytest.fixture
def limits():
    return Limits.from_dict(
        {
            "frame_id": "workcell",
            "workspace_min_m": [-1, -1, 0],
            "workspace_max_m": [1, 1, 1],
            "max_speed_m_s": 0.1,
            "max_contact_distance_m": 0.2,
            "gripper_min_width_m": 0,
            "gripper_max_width_m": 0.08,
            "max_timeout_s": 10,
        }
    )


@pytest.fixture
def station(limits, monkeypatch):
    monkeypatch.setattr("roboskillhub.runtime.yam.time.time", lambda: 1000.0)
    events = []
    measured = {
        "pose": {
            "frame_id": "workcell",
            "position_m": [0.1, 0.2, 0.3],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "timestamp_s": 1000.0,
        "gripper": {"holding": None},
    }

    def read_state(*, timeout_s):
        events.append(("read", timeout_s))
        return measured

    def move_cartesian(pose, **kwargs):
        events.append(("move", pose, kwargs))
        return {"ok": True, "command_id": "site-command-17"}

    def set_gripper_width(width_m, *, timeout_s):
        events.append(("gripper", width_m, timeout_s))
        return {"ok": True}

    def stop():
        events.append(("stop",))

    callbacks = {
        "read_state": read_state,
        "move_cartesian": move_cartesian,
        "set_gripper_width": set_gripper_width,
        "stop": stop,
        "limits": limits,
    }
    return YamBackend(**callbacks), callbacks, events, measured


def test_preserves_explicit_motion_arguments_and_call_order(station):
    backend, _callbacks, events, measured = station
    backend.get_state(timeout_s=2)
    target = {**measured["pose"], "position_m": [0.2, 0.2, 0.3]}
    ack = backend.move_pose(target, speed_m_s=0.04, timeout_s=3, linear=True)
    state_after_ack = backend.get_state(timeout_s=1)
    backend.set_gripper(0.035, timeout_s=2)
    backend.stop()
    assert [event[0] for event in events] == ["read", "move", "read", "gripper", "stop"]
    assert events[1][1] == target
    assert events[1][2] == {"speed_m_s": 0.04, "timeout_s": 3, "linear": True}
    assert events[3] == ("gripper", 0.035, 2)
    assert ack == {"ok": True, "command_id": "site-command-17"}
    # An accepted command must not replace independent measured feedback.
    assert state_after_ack["pose"]["position_m"] == [0.1, 0.2, 0.3]
    assert state_after_ack["timestamp_s"] == 1000
    state_after_ack["pose"]["position_m"][0] = 999
    assert measured["pose"]["position_m"][0] == 0.1


def test_describe_is_real_yam_but_not_a_claim_of_station_validation(station):
    backend, callbacks, _, _ = station
    description = backend.describe()
    assert description["kind"] == "real"
    assert description["embodiment"] == "yam_i2rt"
    assert description["frame_id"] == "workcell"
    assert description["station_validated"] is False
    assert "object_feedback" not in description["capabilities"]
    assert (
        "object_feedback"
        in YamBackend(**callbacks, object_feedback=True).describe()["capabilities"]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speed_m_s": 0},
        {"speed_m_s": 0.2},
        {"speed_m_s": float("nan")},
        {"timeout_s": 0},
        {"timeout_s": 11},
        {"linear": "true"},
    ],
)
def test_bad_motion_arguments_never_reach_driver(station, kwargs):
    backend, _, events, measured = station
    arguments = {"speed_m_s": 0.04, "timeout_s": 2, "linear": False} | kwargs
    with pytest.raises(Rejected):
        backend.move_pose(measured["pose"], **arguments)
    assert events == []


def test_wrong_frame_or_workspace_never_reaches_driver(station):
    backend, _, events, measured = station
    for bad_pose in (
        {**measured["pose"], "frame_id": "camera"},
        {**measured["pose"], "position_m": [2, 0, 0.3]},
    ):
        with pytest.raises(Rejected):
            backend.move_pose(bad_pose, speed_m_s=0.04, timeout_s=2, linear=False)
    with pytest.raises(Rejected):
        backend.set_gripper(0.1, timeout_s=2)
    assert events == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("timestamp_s", 998),
        ("timestamp_s", 1001),
        ("timestamp_s", True),
        ("timestamp_s", float("nan")),
        ("pose", {"frame_id": "camera"}),
    ],
)
def test_stale_or_invalid_measured_state_is_not_refreshed(station, field, value):
    backend, _, _, measured = station
    measured[field] = value
    with pytest.raises(EvidenceMissing):
        backend.get_state(timeout_s=1)


def test_object_provider_is_explicit_and_receives_remaining_budget(station, monkeypatch):
    _, callbacks, events, _ = station
    object_state = {
        "drawer": {
            "frame_id": "workcell",
            "position_m": [0.3, 0.4, 0.2],
            "timestamp_s": 1000,
            "source": "rgbd_handle_estimator",
        }
    }

    def provider(*, timeout_s):
        events.append(("objects", timeout_s))
        return object_state

    ticks = iter([10, 10.2, 10.3, 10.4])
    monkeypatch.setattr("roboskillhub.runtime.yam.time.monotonic", lambda: next(ticks))
    backend = YamBackend(**callbacks, object_provider=provider)
    state = backend.get_state(timeout_s=1)
    assert "object_feedback" in backend.describe()["capabilities"]
    assert state["objects"] == object_state
    assert events[0] == ("read", 1)
    assert events[1][1] == pytest.approx(0.7)
    assert state["objects"]["drawer"]["timestamp_s"] == 1000


@pytest.mark.parametrize(
    "change",
    [
        {"frame_id": "camera"},
        {"source": "commanded"},
        {"source": "simulation_truth"},
        {"timestamp_s": 998},
        {"position_m": [1, 2, float("nan")]},
    ],
)
def test_invalid_object_evidence_cannot_support_real_manipulation(station, change):
    _, callbacks, _, measured = station
    measured["objects"] = {
        "box": {
            "frame_id": "workcell",
            "position_m": [0.3, 0.4, 0.2],
            "timestamp_s": 1000,
            "source": "rgbd",
        }
        | change
    }
    backend = YamBackend(**callbacks, object_feedback=True)
    with pytest.raises(EvidenceMissing):
        backend.get_state(timeout_s=1)


@pytest.mark.parametrize("ack", [None, True, {"ok": 1}, {"accepted": True}])
def test_acknowledgement_requires_explicit_boolean_ok(station, ack):
    _, callbacks, _, measured = station
    callbacks["move_cartesian"] = lambda *_args, **_kwargs: ack
    backend = YamBackend(**callbacks)
    with pytest.raises(ExecutionFailed, match="acknowledgement"):
        backend.move_pose(measured["pose"], speed_m_s=0.04, timeout_s=1, linear=False)


def test_negative_ack_is_preserved_for_runtime_to_handle(station):
    _, callbacks, _, measured = station
    callbacks["move_cartesian"] = lambda *_args, **_kwargs: {
        "ok": False,
        "reason": "operator cancelled",
    }
    backend = YamBackend(**callbacks)
    assert backend.move_pose(measured["pose"], speed_m_s=0.04, timeout_s=1, linear=False) == {
        "ok": False,
        "reason": "operator cancelled",
    }


def test_callback_overrun_is_detected_not_reported_as_completion(station, monkeypatch):
    backend, _, events, measured = station
    ticks = iter([10, 12])
    monkeypatch.setattr("roboskillhub.runtime.yam.time.monotonic", lambda: next(ticks))
    with pytest.raises(ExecutionFailed, match="timeout"):
        backend.move_pose(measured["pose"], speed_m_s=0.04, timeout_s=1, linear=False)
    assert events[0][2]["timeout_s"] == 1


def test_no_signature_fallback_silently_discards_timeout_or_speed(station):
    _, callbacks, _, measured = station
    calls = []

    def incomplete_driver(pose):
        calls.append(pose)
        return {"ok": True}

    callbacks["move_cartesian"] = incomplete_driver
    backend = YamBackend(**callbacks)
    with pytest.raises(TypeError, match="speed_m_s"):
        backend.move_pose(measured["pose"], speed_m_s=0.04, timeout_s=1, linear=False)
    assert calls == []


def test_holding_feedback_requires_its_own_measured_timestamp(station):
    backend, _, _, measured = station
    measured["gripper"] = {"holding": True}
    with pytest.raises(EvidenceMissing):
        backend.get_state(timeout_s=1)


def test_gap_conversion_reorders_named_quaternion_without_changing_frames():
    gap = {
        "position": {"x": 0.1, "y": 0.2, "z": 0.3},
        "rotation": {"w": 0.5, "x": 0.5, "y": -0.5, "z": 0.5},
    }
    original = copy.deepcopy(gap)
    pose = gap_pose_to_pose(gap, source_frame_id="workcell", frame_id="workcell")
    assert pose["quaternion_xyzw"] == [0.5, -0.5, 0.5, 0.5]
    assert pose_to_gap(pose, gap_world_frame_id="workcell") == gap
    assert gap == original
    with pytest.raises(Rejected, match="explicit transform"):
        gap_pose_to_pose(gap, source_frame_id="camera", frame_id="workcell")
    with pytest.raises(Rejected, match="frame"):
        pose_to_gap(pose, gap_world_frame_id="camera")
