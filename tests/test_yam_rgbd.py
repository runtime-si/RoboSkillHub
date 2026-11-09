from __future__ import annotations

import json
import sys
import time
import types

import numpy as np
import pytest

from roboskillhub.runtime import yam_rgbd
from roboskillhub.runtime.contracts import EvidenceMissing, Limits, Rejected


def frame():
    return {
        "rgb": np.random.default_rng(17).integers(0, 256, (48, 48, 3), dtype=np.uint8),
        "depth_m": np.full((48, 48), 0.5, np.float32),
        "K": np.array([[100.0, 0, 24], [0, 100.0, 24], [0, 0, 1.0]]),
        "timestamp_s": time.time(),
        "camera": "test-top",
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    events, observers, received = [], [], []
    limits = Limits.from_dict(
        {
            "frame_id": "explicit_base",
            "workspace_min_m": [-1, -1, 0],
            "workspace_max_m": [1, 1, 1],
            "max_speed_m_s": 0.1,
            "max_contact_distance_m": 0.2,
            "gripper_min_width_m": 0,
            "gripper_max_width_m": 0.08,
            "max_state_age_s": 0.5,
        }
    )
    calibration = tmp_path / "calibration.json"
    # This deliberately matches the legacy layout: no machine-readable frame_id.
    calibration.write_text(
        json.dumps(
            {
                "calibrated": True,
                "T_base_from_camera": np.eye(4).tolist(),
                "convention": "untrusted descriptive text",
            }
        )
    )
    module = types.ModuleType("test_semantic_selector")

    def select(seed_frame, *, target_name):
        events.append("select")
        received.append(seed_frame)
        return {target_name: [[20, 20], [28, 28]]}

    module.select = select
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Camera:
        def __init__(self, **kwargs):
            events.append(("camera", kwargs))

        def read_frame(self, *, timeout_s):
            events.append("read_frame")
            return frame()

    class Observer:
        def __init__(self, **kwargs):
            self.args = kwargs
            self.closed = False
            observers.append(self)
            events.append("observer_construct")

        def start(self):
            events.append("observer_start")
            return self

        def __call__(self, *, timeout_s):
            events.append("observe")
            return {target: {"timestamp_s": time.time()} for target in self.args["target_points"]}

        def close(self):
            events.append("observer_close")
            self.closed = True

    class Backend:
        def __init__(self, **kwargs):
            events.append(("handshake", kwargs))
            self.kwargs = kwargs

        def describe(self):
            return {
                "kind": "real",
                "frame_id": "explicit_base",
                "station_validated": False,
                "capabilities": [
                    "pose_feedback",
                    "cartesian_motion",
                    "gripper",
                    "stop",
                    "object_feedback",
                ],
            }

        def get_state(self, *, timeout_s):
            return {"objects": self.kwargs["object_provider"](timeout_s=timeout_s)}

        def move_pose(self, *args, **kwargs):
            events.append("move")

        def set_gripper(self, *args, **kwargs):
            events.append("gripper")

        def stop(self):
            events.append("stop")

        def begin_invocation(self, *, timeout_s):
            events.append(("begin", timeout_s))

        def end_invocation(self):
            events.append("end")

    monkeypatch.setattr(yam_rgbd, "CameraClient", Camera)
    monkeypatch.setattr(yam_rgbd, "RGBDPointObserver", Observer)
    monkeypatch.setattr(yam_rgbd, "YamTcpBackend", Backend)
    kwargs = {
        "limits": limits,
        "port": 9201,
        "camera_port": 9102,
        "calibration_path": calibration,
        "calibration_frame_id": "explicit_base",
        "target_selector": "test_semantic_selector:select",
        "selector_kwargs": {"target_name": "specified_handle"},
    }
    return kwargs, events, observers, received, module


def test_factory_binds_exact_seed_and_declared_frame_without_motor_calls(setup):
    kwargs, events, observers, received, _ = setup
    backend = yam_rgbd.create_backend(**kwargs)
    obs = observers[0]
    assert received[0]["timestamp_s"] == obs.args["seed_frame"]["timestamp_s"]
    np.testing.assert_array_equal(received[0]["rgb"], obs.args["seed_frame"]["rgb"])
    assert not received[0]["rgb"].flags.writeable
    assert obs.args["frame_id"] == "explicit_base"
    assert obs.args["target_points"] == {"specified_handle": [[20, 20], [28, 28]]}
    names = [e if isinstance(e, str) else e[0] for e in events]
    assert names == [
        "camera",
        "read_frame",
        "select",
        "observer_construct",
        "observer_start",
        "handshake",
        "observe",
    ]
    handshake = next(e[1] for e in events if isinstance(e, tuple) and e[0] == "handshake")
    assert handshake["expected_kind"] == "real" and handshake["object_provider"] is obs
    info = backend.describe()["object_observer"]
    assert info["target_ids"] == ["specified_handle"]
    assert info["calibration_frame_id"] == "explicit_base"
    assert len(info["calibration_sha256"]) == 64
    backend.close()
    assert events[-1] == "observer_close" and obs.closed
    assert not {"move", "gripper", "stop", "begin", "end"} & set(names)


def test_legacy_calibration_cannot_guess_a_frame_from_free_text(setup):
    kwargs, events, *_ = setup
    kwargs.pop("calibration_frame_id")
    with pytest.raises(Rejected, match="legacy calibration"):
        yam_rgbd.create_backend(**kwargs)
    assert events == []


def test_wrapper_forwards_whole_invocation_lease_but_close_only_closes_observer(setup):
    kwargs, events, *_ = setup
    backend = yam_rgbd.create_backend(**kwargs)
    backend.begin_invocation(timeout_s=0.4)
    backend.end_invocation()
    backend.close()
    assert events[-3:] == [("begin", 0.4), "end", "observer_close"]
    assert "stop" not in events


@pytest.mark.parametrize(
    "changed", [{"frame_id": "different_base"}, {"calibrated": False}, {"calibrated": 1}]
)
def test_calibration_declarations_must_match_before_camera_or_robot_connection(setup, changed):
    kwargs, events, *_ = setup
    path = kwargs["calibration_path"]
    path.write_text(json.dumps(json.loads(path.read_text()) | changed))
    with pytest.raises(Rejected):
        yam_rgbd.create_backend(**kwargs)
    assert events == []


def test_file_frame_can_supply_the_explicit_declaration(setup):
    kwargs, *_ = setup
    path = kwargs["calibration_path"]
    path.write_text(json.dumps(json.loads(path.read_text()) | {"frame_id": "explicit_base"}))
    kwargs.pop("calibration_frame_id")
    backend = yam_rgbd.create_backend(**kwargs)
    assert backend.describe()["object_observer"]["calibration_frame_id"] == "explicit_base"
    backend.close()


def test_live_rgbd_cannot_be_mixed_with_dry_model_state(setup):
    kwargs, events, *_ = setup
    with pytest.raises(Rejected, match="dry-run"):
        yam_rgbd.create_backend(**kwargs, expected_kind="dry-run")
    assert events == []


def test_slow_selector_cannot_refresh_original_seed_timestamp(setup, monkeypatch):
    kwargs, events, observers, _, module = setup
    now = [1000.0]
    monkeypatch.setattr(yam_rgbd.time, "time", lambda: now[0])

    def slow(seed_frame, **_kwargs):
        now[0] += 1.0
        seed_frame["timestamp_s"] = now[0]  # does not mutate factory's canonical seed
        return {"specified_handle": [[20, 20]]}

    module.select = slow
    with pytest.raises(EvidenceMissing, match="seed frame is stale"):
        yam_rgbd.create_backend(**kwargs)
    assert observers == []
    assert not [e for e in events if isinstance(e, tuple) and e[0] == "handshake"]


def test_failed_handshake_closes_started_observer(setup, monkeypatch):
    kwargs, events, observers, *_ = setup

    def rejected(**_kwargs):
        raise Rejected("wrong bridge frame")

    monkeypatch.setattr(yam_rgbd, "YamTcpBackend", rejected)
    with pytest.raises(Rejected, match="wrong bridge"):
        yam_rgbd.create_backend(**kwargs)
    assert "observer_start" in events and observers[0].closed


def test_target_loss_during_initialization_closes_observer(setup, monkeypatch):
    kwargs, _, observers, *_ = setup
    monkeypatch.setattr(yam_rgbd.RGBDPointObserver, "__call__", lambda *_a, **_k: {})
    with pytest.raises(EvidenceMissing, match="targets were lost"):
        yam_rgbd.create_backend(**kwargs)
    assert observers[0].closed


def test_factory_with_real_observer_produces_fresh_calibrated_records_and_closes_worker(
    setup, monkeypatch
):
    from roboskillhub.runtime.rgbd_observer import RGBDPointObserver

    kwargs, events, _, _, _ = setup

    class ContinuousCamera:
        def __init__(self, **_kwargs):
            pass

        def read_frame(self, *, timeout_s):
            time.sleep(min(0.01, timeout_s))
            return frame()

    monkeypatch.setattr(yam_rgbd, "CameraClient", ContinuousCamera)
    monkeypatch.setattr(yam_rgbd, "RGBDPointObserver", RGBDPointObserver)
    backend = yam_rgbd.create_backend(**kwargs)
    try:
        record = backend.get_state(timeout_s=0.2)["objects"]["specified_handle"]
        assert record["frame_id"] == "explicit_base"
        assert record["source"] == "calibrated_rgbd_lk_surface_points"
        assert record["seed_count"] == 2
        assert record["timestamp_s"] > backend.describe()["object_observer"]["seed_timestamp_s"]
        assert "holding" not in record
    finally:
        backend.close()
    assert not backend._observer._worker.is_alive()
    assert not {"move", "gripper", "stop"} & {e for e in events if isinstance(e, str)}
