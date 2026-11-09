from __future__ import annotations

import io
import json
import socket
import struct
import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest

from roboskillhub.runtime.contracts import EvidenceMissing, ExecutionFailed, Limits, Rejected
from roboskillhub.runtime.wire import MAX_HEADER, Client, WireError, decode, recv, send
from roboskillhub.runtime.yam_tcp import CameraClient, capture_clip, create_backend


@contextmanager
def server(handler):
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(0.05)
    address = listener.getsockname()
    stopped = threading.Event()
    calls, errors, children = [], [], []

    def connection(sock):
        with sock:
            sock.settimeout(1)
            try:
                header, arrays = recv(sock)
                calls.append(header)
                reply = handler(header, arrays)
                if reply is not None:
                    send(sock, *reply)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Intentional client timeout tests close an in-flight reply.
            except Exception as exc:
                errors.append(exc)

    def loop():
        while not stopped.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            thread = threading.Thread(target=connection, args=(conn,), daemon=True)
            children.append(thread)
            thread.start()

    worker = threading.Thread(target=loop, daemon=True)
    worker.start()
    try:
        yield address, calls
    finally:
        stopped.set()
        listener.close()
        worker.join(timeout=1)
        for child in children:
            child.join(timeout=1)
        assert not errors


@pytest.fixture
def limits():
    return Limits.from_dict(
        {
            "frame_id": "yam_left_base",
            "workspace_min_m": [-1, -1, -0.05],
            "workspace_max_m": [1, 1, 1],
            "max_speed_m_s": 0.1,
            "max_contact_distance_m": 0.2,
            "gripper_min_width_m": 0,
            "gripper_max_width_m": 0.08,
        }
    )


def description():
    return {
        "protocol": "roboskillhub-yam",
        "protocol_version": 1,
        "kind": "real",
        "mode": "confirm",
        "embodiment": "yam_i2rt",
        "frame_id": "yam_left_base",
        "capabilities": ["pose_feedback", "cartesian_motion", "gripper", "stop"],
        "supports_request_ids": True,
        "supports_deadlines": True,
        "supports_speed_limits": True,
        "supports_invocation_lease": True,
    }


def measured_state():
    return {
        "pose": {
            "frame_id": "yam_left_base",
            "position_m": [0.1, 0.2, 0.3],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "timestamp_s": time.time(),
        "objects": {},
        "gripper": {"holding": None},
    }


def healthy_handler(header, _arrays):
    op = header["op"]
    if op == "hub_describe":
        return description(), {}
    if op == "hub_state":
        return measured_state(), {}
    if op in ("hub_claim", "hub_release"):
        return {
            "ok": True,
            "owner_id": header["owner_id"],
            "status": "claimed" if op == "hub_claim" else "released",
        }, {}
    reply = {"ok": True, "command_id": header["command_id"], "owner_id": header["owner_id"]}
    if op == "hub_stop":
        reply.update(
            status="held",
            hold_confirmed=True,
            powered_hold=True,
            other_command_active=False,
            command_committed=True,
            cancel_confirmed=True,
        )
    return reply, {}


def test_real_tcp_framing_and_motion_contract(limits):
    with server(healthy_handler) as ((host, port), calls):
        backend = create_backend(limits=limits, host=host, port=port)
        owner = backend.begin_invocation(timeout_s=1)["owner_id"]
        target = {**measured_state()["pose"], "position_m": [0.2, 0.2, 0.3]}
        ack = backend.move_pose(target, speed_m_s=0.04, timeout_s=1, linear=True)
        backend.set_gripper(0.035, timeout_s=1)
        current = backend.get_state(timeout_s=1)
        backend.stop()
        backend.end_invocation()
    assert backend.describe()["mode"] == "confirm"
    assert "object_feedback" not in backend.describe()["capabilities"]
    moves = [call for call in calls if call["op"] == "hub_move"]
    assert len(moves) == 1
    assert moves[0]["command_id"] == ack["command_id"]
    assert moves[0]["position_m"] == target["position_m"]
    assert moves[0]["quat_xyzw"] == [0, 0, 0, 1]
    assert moves[0]["speed_m_s"] == 0.04 and moves[0]["linear"] is True
    assert moves[0]["frame_id"] == "yam_left_base"
    assert 0 < moves[0]["timeout_s"] <= 1
    assert current["pose"]["position_m"] == [0.1, 0.2, 0.3]  # ACK is not feedback.
    grip = next(call for call in calls if call["op"] == "hub_grip")
    assert grip["width_m"] == 0.035
    assert calls[-2] == {"op": "hub_stop", "command_id": grip["command_id"], "owner_id": owner}
    assert calls[-1] == {"op": "hub_release", "owner_id": owner}
    assert moves[0]["owner_id"] == grip["owner_id"] == owner
    assert not {"move", "grip", "finish", "reset", "to_confirm"} & {c["op"] for c in calls}


@pytest.mark.parametrize(
    "mutation",
    [
        {"mode": "dry", "kind": "dry"},
        {"supports_deadlines": False},
        {"supports_invocation_lease": False},
        {"frame_id": "other"},
        {"protocol_version": 0},
    ],
)
def test_unsupported_or_dry_server_is_rejected_before_motion(limits, mutation):
    with server(lambda *_: (description() | mutation, {})) as ((host, port), calls):
        with pytest.raises(Rejected):
            create_backend(limits=limits, host=host, port=port)
    assert [c["op"] for c in calls] == ["hub_describe"]


def test_old_server_does_not_trigger_legacy_fallback(limits):
    with server(lambda *_: ({"error": "unknown op hub_describe"}, {})) as ((host, port), calls):
        with pytest.raises(Rejected, match="legacy"):
            create_backend(limits=limits, host=host, port=port)
    assert [c["op"] for c in calls] == ["hub_describe"]


def test_explicit_dry_run_keeps_virtual_kind_in_description_state_and_ack(limits):
    def handler(header, arrays):
        reply, payload = healthy_handler(header, arrays)
        return {**reply, "kind": "dry-run", "virtual": True, "mode": "dry"}, payload

    with server(handler) as ((host, port), _):
        backend = create_backend(limits=limits, host=host, port=port, expected_kind="dry-run")
        state = backend.get_state(timeout_s=1)
        backend.begin_invocation(timeout_s=1)
        ack = backend.move_pose(state["pose"], speed_m_s=0.04, timeout_s=1, linear=True)
        grip = backend.set_gripper(0.035, timeout_s=1)
        for result in (backend.describe(), state, ack, grip):
            assert result["kind"] == "dry-run"
            assert result["virtual"] is True
        assert backend.describe()["station_validated"] is False
        backend.end_invocation()


@pytest.mark.parametrize(
    "expected,actual,mode", [("real", "dry-run", "dry"), ("dry-run", "real", "confirm")]
)
def test_expected_kind_cannot_silently_switch_to_other_execution_mode(
    limits, expected, actual, mode
):
    with server(lambda *_: ({**description(), "kind": actual, "mode": mode}, {})) as (
        (host, port),
        calls,
    ):
        with pytest.raises(Rejected, match="expected_kind"):
            create_backend(limits=limits, host=host, port=port, expected_kind=expected)
    assert [c["op"] for c in calls] == ["hub_describe"]


def test_kind_change_after_connection_is_rejected_before_action(limits):
    seen = []

    def handler(header, arrays):
        if header["op"] == "hub_describe":
            seen.append(header)
            if len(seen) > 1:
                return {**description(), "kind": "dry-run", "mode": "dry", "virtual": True}, {}
        return healthy_handler(header, arrays)

    with server(handler) as ((host, port), calls):
        backend = create_backend(limits=limits, host=host, port=port)
        backend.begin_invocation(timeout_s=1)
        with pytest.raises(Rejected, match="expected_kind"):
            backend.move_pose(measured_state()["pose"], speed_m_s=0.04, timeout_s=1, linear=True)
        backend.end_invocation()
    assert not any(c["op"] in {"hub_move", "hub_grip", "hub_stop"} for c in calls)


@pytest.mark.parametrize("failure", ["disconnect", "timeout", "wrong-id"])
def test_uncertain_motion_is_never_retried_and_is_cancelled_on_new_connection(limits, failure):
    def handler(header, arrays):
        if header["op"] == "hub_move":
            if failure == "timeout":
                time.sleep(0.15)
            if failure == "wrong-id":
                return {"ok": True, "command_id": "another-request"}, {}
            return None
        return healthy_handler(header, arrays)

    with server(handler) as ((host, port), calls):
        backend = create_backend(limits=limits, host=host, port=port, stop_timeout_s=0.5)
        backend.begin_invocation(timeout_s=1)
        with pytest.raises(ExecutionFailed, match="not retried"):
            backend.move_pose(
                measured_state()["pose"],
                speed_m_s=0.03,
                timeout_s=0.06 if failure == "timeout" else 0.5,
                linear=True,
            )
        backend.end_invocation()
    moves = [c for c in calls if c["op"] == "hub_move"]
    stops = [c for c in calls if c["op"] == "hub_stop"]
    assert len(moves) == len(stops) == 1
    assert stops[0]["command_id"] == moves[0]["command_id"]
    assert stops[0]["owner_id"] == moves[0]["owner_id"]


def test_direct_actuation_requires_explicit_whole_invocation_lease(limits):
    with server(healthy_handler) as ((host, port), calls):
        backend = create_backend(limits=limits, host=host, port=port)
        for action in (
            lambda: backend.move_pose(
                measured_state()["pose"], speed_m_s=0.04, timeout_s=1, linear=True
            ),
            lambda: backend.set_gripper(0.035, timeout_s=1),
            backend.stop,
        ):
            with pytest.raises(Rejected, match="begin_invocation"):
                action()
    assert [c["op"] for c in calls] == ["hub_describe"]


def test_lease_spans_all_actions_then_releases_and_next_owner_nonce_is_fresh(limits):
    owner = [None]

    def exclusive(header, arrays):
        if header["op"] == "hub_claim":
            if owner[0] is not None:
                return {"ok": False, "owner_id": header["owner_id"], "status": "busy"}, {}
            owner[0] = header["owner_id"]
        elif header["op"] == "hub_release":
            assert owner[0] == header["owner_id"]
            owner[0] = None
        elif header["op"] in {"hub_move", "hub_grip", "hub_stop"}:
            assert owner[0] == header["owner_id"]
        return healthy_handler(header, arrays)

    with server(exclusive) as ((host, port), calls):
        a, b = [create_backend(limits=limits, host=host, port=port) for _ in range(2)]
        first = a.begin_invocation(timeout_s=1)["owner_id"]
        a.set_gripper(0.035, timeout_s=1)
        with pytest.raises(Rejected, match="another owner"):
            b.begin_invocation(timeout_s=1)
        with pytest.raises(Rejected, match="already owns"):
            a.begin_invocation(timeout_s=1)
        with pytest.raises(Rejected, match="begin_invocation"):
            b.set_gripper(0.035, timeout_s=1)
        a.move_pose(measured_state()["pose"], speed_m_s=0.04, timeout_s=1, linear=True)
        a.end_invocation()
        second = b.begin_invocation(timeout_s=1)["owner_id"]
        b.set_gripper(0.035, timeout_s=1)
        b.end_invocation()
        third = a.begin_invocation(timeout_s=1)["owner_id"]
        a.end_invocation()
    assert len({first, second, third}) == 3
    assert [c["owner_id"] for c in calls if c["op"] in {"hub_move", "hub_grip"}] == [
        first,
        first,
        second,
    ]
    assert owner[0] is None


def test_concurrent_begin_on_same_backend_reserves_nonce_before_sending(limits):
    entered, allow_reply = threading.Event(), threading.Event()

    def delayed_claim(header, arrays):
        if header["op"] == "hub_claim":
            entered.set()
            assert allow_reply.wait(1)
        return healthy_handler(header, arrays)

    with server(delayed_claim) as ((host, port), calls):
        backend = create_backend(limits=limits, host=host, port=port)
        result = []
        worker = threading.Thread(
            target=lambda: result.append(backend.begin_invocation(timeout_s=1))
        )
        worker.start()
        assert entered.wait(1)
        try:
            with pytest.raises(Rejected, match="already owns"):
                backend.begin_invocation(timeout_s=1)
        finally:
            allow_reply.set()
            worker.join(1)
        assert result[0]["status"] == "claimed"
        backend.end_invocation()
    assert len([c for c in calls if c["op"] == "hub_claim"]) == 1


@pytest.mark.parametrize("lost_op", ["hub_claim", "hub_release"])
def test_uncertain_lease_response_retains_nonce_and_never_reclaims_or_releases_again(
    limits, lost_op
):
    def handler(header, arrays):
        return None if header["op"] == lost_op else healthy_handler(header, arrays)

    with server(handler) as ((host, port), calls):
        backend = create_backend(limits=limits, host=host, port=port)
        if lost_op == "hub_claim":
            with pytest.raises(ExecutionFailed, match="uncertain"):
                backend.begin_invocation(timeout_s=1)
        else:
            backend.begin_invocation(timeout_s=1)
            with pytest.raises(ExecutionFailed, match="uncertain"):
                backend.end_invocation()
        nonce = backend._owner_id
        assert nonce is not None
        with pytest.raises(Rejected, match="unresolved"):
            backend.begin_invocation(timeout_s=1)
        with pytest.raises(Rejected, match="begin_invocation"):
            backend.set_gripper(0.035, timeout_s=1)
        with pytest.raises((Rejected, ExecutionFailed)):
            backend.end_invocation()
        assert backend._owner_id == nonce
    assert len([c for c in calls if c["op"] == lost_op]) == 1
    assert not any(c["op"] in {"hub_move", "hub_grip", "hub_stop"} for c in calls)


@pytest.mark.parametrize(
    "bad_proof",
    [
        {"status": "superseded", "hold_confirmed": False, "powered_hold": False},
        {"powered_hold": False},
        {"other_command_active": True},
        {"owner_id": "wrong-owner"},
        {"hold_confirmed": False},
        "disconnect",
    ],
)
def test_unconfirmed_stop_latches_owner_and_prevents_release(limits, bad_proof):
    def handler(header, arrays):
        reply, payload = healthy_handler(header, arrays)
        if header["op"] == "hub_stop":
            return None if bad_proof == "disconnect" else (reply | bad_proof, payload)
        return reply, payload

    with server(handler) as ((host, port), calls):
        backend = create_backend(limits=limits, host=host, port=port)
        backend.begin_invocation(timeout_s=1)
        backend.set_gripper(0.035, timeout_s=1)
        with pytest.raises((ExecutionFailed, ConnectionError)):
            backend.stop()
        with pytest.raises(ExecutionFailed, match="cannot release"):
            backend.end_invocation()
        with pytest.raises(Rejected):
            backend.begin_invocation(timeout_s=1)
        with pytest.raises(Rejected, match="previous YAM action"):
            backend.set_gripper(0.035, timeout_s=1)
    assert not any(c["op"] == "hub_release" for c in calls)


def test_proven_never_committed_cancellation_does_not_require_fabricated_hold(limits):
    def handler(header, arrays):
        if header["op"] == "hub_move":
            return None
        reply, payload = healthy_handler(header, arrays)
        if header["op"] == "hub_stop":
            reply.update(
                status="cancelled_before_commit",
                hold_confirmed=False,
                powered_hold=False,
                command_committed=False,
                cancel_confirmed=True,
            )
        return reply, payload

    with server(handler) as ((host, port), calls):
        backend = create_backend(limits=limits, host=host, port=port)
        backend.begin_invocation(timeout_s=1)
        with pytest.raises(ExecutionFailed, match="not retried"):
            backend.move_pose(measured_state()["pose"], speed_m_s=0.04, timeout_s=1, linear=True)
        backend.end_invocation()
    assert len([c for c in calls if c["op"] == "hub_move"]) == 1
    assert calls[-1]["op"] == "hub_release"


def test_wire_roundtrip_numeric_array_and_one_connection_per_request():
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    with server(lambda h, a: ({"sequence": h["sequence"]}, {"sample": data})) as (
        (host, port),
        calls,
    ):
        client = Client(host, port)
        for sequence in (1, 2):
            header, arrays = client.call("frame", fields={"sequence": sequence})
            assert header == {"sequence": sequence}
            np.testing.assert_array_equal(arrays["sample"], data)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "spec",
    [
        {"name": "x", "dtype": "O", "shape": [1]},
        {"name": "x", "dtype": "<f8", "shape": [10**6, 10**6]},
        {"name": "x", "dtype": "u1", "shape": [-1]},
        {"name": "x", "dtype": "u1", "shape": [True]},
    ],
)
def test_wire_rejects_unsafe_arrays_before_waiting_for_payload(spec):
    with _pair() as (left, right):
        header = json.dumps({"arrays": [spec]}).encode()
        left.sendall(struct.pack("!I", len(header)) + header)
        with pytest.raises(WireError):
            recv(right)


@contextmanager
def _pair():
    left, right = socket.socketpair()
    left.settimeout(0.5)
    right.settimeout(0.5)
    try:
        yield left, right
    finally:
        left.close()
        right.close()


def test_wire_rejects_oversized_header_and_duplicate_json_keys():
    with _pair() as (left, right):
        left.sendall(struct.pack("!I", MAX_HEADER + 1))
        with pytest.raises(WireError):
            recv(right)
    with _pair() as (left, right):
        header = b'{"ok":true,"ok":false,"arrays":[]}'
        left.sendall(struct.pack("!I", len(header)) + header)
        with pytest.raises(WireError, match="duplicate"):
            recv(right)


def test_camera_raw_frame_preserves_original_timestamp_and_depth(monkeypatch):
    monkeypatch.setattr("roboskillhub.runtime.yam_tcp.time.time", lambda: 1000.0)
    arrays = {
        "rgb": np.zeros((6, 8, 3), np.uint8),
        "depth_m": np.full((6, 8), 1.25, np.float32),
        "K": np.array([[100.0, 0, 4], [0, 100, 3], [0, 0, 1]]),
    }
    with server(lambda *_: ({"time": 999.9, "camera": "top"}, arrays)) as ((host, port), calls):
        result = CameraClient(host=host, port=port).read_frame()
    assert calls == [{"op": "frame"}]
    assert result["timestamp_s"] == 999.9
    np.testing.assert_array_equal(result["depth_m"], arrays["depth_m"])
    assert "objects" not in result


def test_decode_real_jpeg_and_uint16_png():
    Image = pytest.importorskip("PIL.Image")
    rgb = np.full((6, 8, 3), [12, 80, 190], dtype=np.uint8)
    mm = np.full((6, 8), 1234, dtype=np.uint16)
    jpeg, png = io.BytesIO(), io.BytesIO()
    Image.fromarray(rgb).save(jpeg, format="JPEG")
    Image.fromarray(mm).save(png, format="PNG")
    result = decode(
        {
            "rgb_jpeg": np.frombuffer(jpeg.getvalue(), np.uint8),
            "depth_mm_png": np.frombuffer(png.getvalue(), np.uint8),
        }
    )
    assert result["rgb"].shape == (6, 8, 3)
    assert result["rgb"].dtype == np.uint8
    np.testing.assert_allclose(result["depth_m"], 1.234, atol=1e-6)


def test_capture_clip_deduplicates_timestamps_and_saves_tracking_input(tmp_path):
    class Camera:
        def __init__(self):
            self.timestamps = iter([1000.0, 1000.0, 1000.1])

        def read_frame(self, *, timeout_s):
            assert 0 < timeout_s <= 1
            return {
                "rgb": np.zeros((6, 8, 3), np.uint8),
                "camera": "top",
                "timestamp_s": next(self.timestamps),
            }

    result = capture_clip(
        Camera(), frame_count=2, interval_s=0, timeout_s=1, output_path=tmp_path / "clip.npy"
    )
    assert result["timestamps_s"] == [1000.0, 1000.1]
    assert np.load(result["frames_path"], allow_pickle=False).shape == (2, 6, 8, 3)
    assert json.loads((tmp_path / "clip.metadata.json").read_text())["color_order"] == "RGB"


def test_repeated_frame_does_not_become_a_fake_video():
    class Camera:
        def read_frame(self, *, timeout_s):
            return {"rgb": np.zeros((6, 8, 3), np.uint8), "camera": "top", "timestamp_s": 1000.0}

    with pytest.raises(ExecutionFailed, match="deadline"):
        capture_clip(Camera(), frame_count=2, interval_s=0, timeout_s=0.05)


def test_bad_camera_clock_is_rejected(monkeypatch):
    monkeypatch.setattr("roboskillhub.runtime.yam_tcp.time.time", lambda: 1000.0)
    with server(lambda *_: ({"time": 900.0, "camera": "top"}, {})) as ((host, port), _):
        with pytest.raises(EvidenceMissing, match="stale"):
            CameraClient(host=host, port=port).read_frame()
