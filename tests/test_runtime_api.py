"""Agent/API boundary tests; no simulator or robot is connected.

The point-tracking test runs actual CPU optical flow on a saved synthetic RGB
sequence. Motion tests use controlled API substitutes and never actuate hardware.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from roboskillhub.cli import main
from roboskillhub.runtime import api
from roboskillhub.runtime.contracts import CAPABILITY_BUNDLES


def motion_request():
    return {
        "target_pose": {
            "frame_id": "world",
            "position_m": [0.0, 0.0, 0.2],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    }


@pytest.fixture(autouse=True)
def isolate_station_configuration(monkeypatch):
    monkeypatch.delenv("ROBOSKILLHUB_STATION", raising=False)
    monkeypatch.setattr(api, "_CACHE", {})
    monkeypatch.setattr(api, "_TRACKERS", {})


def test_runtime_list_exposes_every_schema_without_hardware(capsys):
    assert main(["runtime", "list"]) == 0
    data = json.loads(capsys.readouterr().out)
    entries = {row["capability"]: row for row in data["capabilities"]}
    assert set(entries) == set(CAPABILITY_BUNDLES)
    assert data["station_id"] is None
    assert data["backend"] is None
    for capability, row in entries.items():
        schema = row["input_schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"]
        assert set(schema["required"]) <= set(schema["properties"])
        assert set(row["statuses"]) == {"succeeded", "failed", "unverified", "rejected"}
        if capability != "track_points":
            assert row["configured"] is False


def test_motion_without_station_returns_structured_rejection():
    result = api.invoke("move_pose", motion_request())
    assert result["status"] == "rejected"
    assert "no station configured" in result["error"]["message"]
    assert result["manifest"]["station_id"] is None
    assert result["manifest"]["station_config_sha256"] is None
    assert result["trace"] == []
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("capability", [None, [], {}, False, 3])
def test_nonstring_capability_is_a_structured_rejection(capability):
    result = api.invoke(capability, {})
    assert result["status"] == "rejected"
    assert result["skill"] is None
    assert result["error"]["message"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", ["delete", "replace"])
def test_completed_result_keeps_configuration_snapshot_when_file_changes(
    tmp_path, monkeypatch, change
):
    path = tmp_path / "station.yaml"
    original = b"schema_version: 1\nstation_id: original-cell\n"
    path.write_bytes(original)
    expected_digest = hashlib.sha256(original).hexdigest()
    received_config = []

    class CompletedOperation:
        def invoke(self, capability, request):
            assert capability == "move_pose"
            if change == "delete":
                path.unlink()
            else:
                path.write_text("schema_version: 1\nstation_id: different-cell\n")
            return {
                "status": "succeeded",
                "invocation_id": "test-completed",
                "evidence": {"completed_operation": True},
                "trace": [],
            }

    def station(config):
        received_config.append(config)
        return CompletedOperation()

    monkeypatch.setattr(api, "_station", station)
    result = api.invoke("move_pose", motion_request(), str(path))
    assert result["status"] == "succeeded"
    assert result["evidence"]["completed_operation"] is True
    assert result["manifest"]["station_id"] == "original-cell"
    assert result["manifest"]["station_config_sha256"] == expected_digest
    assert received_config[0][2] == expected_digest


def test_interleaved_api_calls_leave_process_stdout_unchanged(monkeypatch):
    # Cross the two call lifetimes deterministically. The previous nested global
    # redirect_stdout contexts left sys.stdout pointing to stderr after this.
    original_stdout = sys.stdout
    entered_first = threading.Event()
    entered_second = threading.Event()
    finished_first = threading.Event()

    class ConcurrentOperation:
        def invoke(self, capability, request):
            if request["target_pose"]["position_m"][0] == 0:
                entered_first.set()
                assert entered_second.wait(2)
            else:
                entered_second.set()
                assert finished_first.wait(2)
            return {
                "status": "succeeded",
                "invocation_id": "parallel-contract-test",
                "evidence": {},
                "trace": [],
            }

    monkeypatch.setattr(api, "_station", lambda config: ConcurrentOperation())

    def first_call():
        try:
            return api.invoke("move_pose", motion_request())
        finally:
            finished_first.set()

    second_request = motion_request()
    second_request["target_pose"]["position_m"][0] = 0.1
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(first_call)
            assert entered_first.wait(2)
            second = pool.submit(api.invoke, "move_pose", second_request)
            assert first.result(timeout=3)["status"] == "succeeded"
            assert second.result(timeout=3)["status"] == "succeeded"
        assert sys.stdout is original_stdout
    finally:
        # Keep a failing regression from changing the remainder of pytest's IO.
        sys.stdout = original_stdout


def test_jsonlines_describe_invalid_invoke_sequence_is_clean_and_keeps_processing(
    monkeypatch, capsys
):
    describe = api.describe
    invoke = api.invoke

    def noisy_describe(*args):
        print("provider diagnostic during describe")
        return describe(*args)

    def noisy_invoke(*args):
        print("provider diagnostic during invoke")
        return invoke(*args)

    monkeypatch.setattr(api, "describe", noisy_describe)
    monkeypatch.setattr(api, "invoke", noisy_invoke)
    messages = [
        json.dumps({"id": 1, "method": "describe"}),
        "invalid json",
        json.dumps(
            {
                "id": 3,
                "method": "invoke",
                "params": {
                    "capability": "move_pose",
                    "request": motion_request(),
                },
            }
        ),
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(messages) + "\n"))
    assert main(["runtime", "serve"]) == 0
    captured = capsys.readouterr()
    responses = [json.loads(line) for line in captured.out.splitlines()]
    assert len(responses) == 3
    assert responses[0]["id"] == 1
    assert len(responses[0]["result"]["capabilities"]) == len(CAPABILITY_BUNDLES)
    assert responses[1]["id"] is None
    assert "error" in responses[1]
    assert responses[2]["id"] == 3
    assert responses[2]["result"]["status"] == "rejected"
    assert "no station configured" in responses[2]["result"]["error"]["message"]
    assert "provider diagnostic during describe" in captured.err
    assert "provider diagnostic during invoke" in captured.err


def test_actual_cpu_point_tracking_from_npy_returns_motion_and_input_digest(tmp_path):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(615)
    initial = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    frames = np.stack(
        [
            initial,
            cv2.warpAffine(initial, np.float32([[1, 0, 3], [0, 1, 2]]), (128, 96)),
            cv2.warpAffine(initial, np.float32([[1, 0, 6], [0, 1, 4]]), (128, 96)),
        ]
    )
    path = tmp_path / "frames.npy"
    np.save(path, frames, allow_pickle=False)
    points = [[65, 55], [85, 45]]
    result = api.invoke(
        "track_points",
        {
            "frames_path": str(path),
            "query_points": points,
            "backend": "opencv-lk",
        },
    )
    assert result["status"] == "succeeded"
    assert result["skill"] == "tracking-points"
    assert result["backend"]["name"] == "opencv-lk"
    tracks = np.asarray(result["evidence"]["tracks"])
    assert tracks.shape == (3, 2, 2)
    np.testing.assert_allclose(tracks[-1] - np.asarray(points), [[6, 4], [6, 4]], atol=0.3)
    assert (
        result["input_artifact"]["frames_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert result["manifest"]["station_id"] is None
    json.dumps(result, allow_nan=False)


def test_station_factory_receives_validated_limits_once_and_cannot_override(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from roboskillhub.runtime.yam import YamBackend

    calls = []

    def no_action(*args, **kwargs):
        raise AssertionError("describe must not command or observe the robot")

    def factory(*, limits, station_label):
        calls.append((limits, station_label))
        return YamBackend(
            limits=limits,
            read_state=no_action,
            move_cartesian=no_action,
            set_gripper_width=no_action,
            stop=no_action,
        )

    monkeypatch.setitem(sys.modules, "test_site_binding", SimpleNamespace(create=factory))
    config = {
        "schema_version": 1,
        "station_id": "contract-only",
        "backend": {"factory": "test_site_binding:create", "kwargs": {"station_label": "lab"}},
        "limits": {
            "frame_id": "world",
            "workspace_min_m": [-1, -1, 0],
            "workspace_max_m": [1, 1, 1],
            "max_speed_m_s": 0.1,
            "max_contact_distance_m": 0.1,
            "gripper_min_width_m": 0,
            "gripper_max_width_m": 0.08,
        },
    }
    path = tmp_path / "station.json"
    path.write_text(json.dumps(config))
    first = api.describe(str(path))
    second = api.describe(str(path))
    assert first["backend"]["kind"] == second["backend"]["kind"] == "real"
    assert len(calls) == 1
    assert calls[0][0].frame_id == "world"
    assert calls[0][1] == "lab"
    config["backend"]["kwargs"]["limits"] = {}
    path.write_text(json.dumps(config))
    result = api.describe(str(path))
    assert result["backend"] is None
    move = next(x for x in result["capabilities"] if x["capability"] == "move_pose")
    assert "cannot override" in move["readiness_note"]
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["delete", "replace"])
def test_tracking_preserves_input_snapshot_and_hash_if_file_changes(tmp_path, monkeypatch, change):
    from roboskillhub.runtime import tracking

    path = tmp_path / "clip.npy"
    source = np.zeros((2, 20, 20, 3), dtype=np.uint8)
    np.save(path, source)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    class Tracker:
        def track(self, frames, query_points):
            if change == "delete":
                path.unlink()
            else:
                np.save(path, np.full_like(source, 255))
            np.testing.assert_array_equal(frames, source)
            return {
                "status": "tracked",
                "tracks": [[[5, 5]], [[5, 5]]],
                "visible": [[True], [True]],
            }

    monkeypatch.setattr(tracking, "LKPointTracker", Tracker)
    result = api.invoke("track_points", {"frames_path": str(path), "query_points": [[5, 5]]})
    assert result["status"] == "succeeded"
    assert result["input_artifact"]["frames_sha256"] == digest
