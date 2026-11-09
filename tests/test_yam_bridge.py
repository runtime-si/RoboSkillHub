"""Local protocol/concurrency checks with API doubles; no YAM hardware acceptance."""

from __future__ import annotations

import copy
import json
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from roboskillhub.runtime.contracts import Rejected
from roboskillhub.runtime.yam_bridge import Bridge, create_bridge, make_server


@dataclass
class Pose:
    position_m: object
    orientation_xyzw: object = (0, 0, 0, 1)


class Hardware:
    """Mirrors the inspected prepare/commit/hold ownership API, without motors."""

    def __init__(self):
        self._lock = threading.RLock()
        self._updated = time.monotonic()
        self._state = SimpleNamespace(
            ee_pose=Pose([0, 0, 0.2]),
            joint_positions_rad=[0, 0, 0.2],
            gripper_closure=0.0,
            diagnostics={},
        )
        self.events = []
        self.prepare_entered = threading.Event()
        self.prepare_release = None
        self.observe_new = True
        self.reach_pose = True
        self.reach_grip = True
        self.contact = False
        self.held = False

    def state(self):
        with self._lock:
            if self.observe_new:
                self._updated = time.monotonic()
            return copy.deepcopy(self._state)

    def prepare_target(self, pose, closure, *, duration_s):
        self.prepare_entered.set()
        if self.prepare_release is not None:
            assert self.prepare_release.wait(2), "test must release native planning"
        # Native prepare_target is pure planning and cannot commit an actuator.
        return dict(pose=pose, closure=closure, expires=time.monotonic() + duration_s)

    def commit_target(self, plan):
        with self._lock:
            assert not self.held
            assert time.monotonic() < plan["expires"]
            self.events.append(("commit", plan))
            if self.reach_pose:
                self._state.ee_pose = copy.deepcopy(plan["pose"])
            self._state.gripper_closure = plan["closure"]

    def gripper_target(self, closure, *, duration_s):
        with self._lock:
            assert not self.held
            assert 0 < duration_s <= 6
            self.events.append(("gripper", closure))
            if self.reach_grip:
                self._state.gripper_closure = closure
            elif self.contact:
                self._state.diagnostics["gripper_force_backoff"] = 0.1

    def hold(self, *, measured, preserve_gripper):
        self.events.append(("hold", measured, preserve_gripper))

    def enter_operator_hold(self):
        self.events.append(("operator_hold",))
        self.held = True


class Station:
    def __init__(self, *, powered=True, mode="confirm"):
        self.arm, self.mode, self.powered = "left", mode, powered
        self.hw = Hardware()
        self.q = np.array([0.0, 0.0, 0.2])
        self.gripper_open = 1.0
        self.plan_calls = []

    def handle(self, header, arrays):
        assert header == {"op": "describe"} and arrays == {}
        return {"grip_span_m": 0.094, "bounds": [[-1, -1, 0], [1, 1, 1]]}, {}

    def current(self):
        if self.powered:
            state = self.hw.state()
            return np.asarray(state.joint_positions_rad), state.ee_pose, 1 - state.gripper_closure
        return self.q.copy(), Pose(self.q.copy()), self.gripper_open

    def plan(self, position, quaternion):
        self.plan_calls.append(position)
        _, start, _ = self.current()
        return {
            "steps": [{"pose": Pose(position, quaternion), "joints": np.asarray(position)}],
            "distance_m": float(np.linalg.norm(np.asarray(position) - start.position_m)),
            "turn_rad": 0,
        }


@pytest.fixture
def rig():
    station = Station()
    bridge = Bridge(
        station,
        allow_powered=True,
        confirmation=lambda *_: True,
        rate_limit_provider=lambda: 0.05,
    )
    bridge.claim("test_owner")
    yield bridge, station
    bridge.close()


def move(command_id="move_1", **kwargs):
    return {
        "op": "hub_move",
        "owner_id": "test_owner",
        "command_id": command_id,
        "frame_id": "yam_left_base",
        "position_m": [0.02, 0, 0.2],
        "quat_xyzw": [0, 0, 0, 1],
        "speed_m_s": 0.05,
        "linear": True,
        "timeout_s": 1,
        **kwargs,
    }


def grip(command_id="grip_1", **kwargs):
    return {
        "op": "hub_grip",
        "owner_id": "test_owner",
        "command_id": command_id,
        "width_m": 0.02,
        "timeout_s": 1,
        **kwargs,
    }


def call(bridge, request):
    if request["op"] == "hub_stop" and "owner_id" not in request:
        request = {**request, "owner_id": "test_owner"}
    result, arrays = bridge.handle(request, {})
    assert arrays == {}
    return result


def event_names(station):
    return [event[0] for event in station.hw.events]


def assert_hold(station):
    assert ("hold", True, True) in station.hw.events
    assert ("operator_hold",) in station.hw.events
    assert not {"finish", "reset", "home", "close", "off"} & set(event_names(station))


def test_describe_and_measured_state_have_explicit_provenance(rig):
    bridge, _ = rig
    description = call(bridge, {"op": "hub_describe"})
    assert description["protocol"] == "roboskillhub-yam"
    assert description["protocol_version"] == 1
    assert description["kind"] == "real" and description["mode"] == "confirm"
    assert description["capabilities"] == ["pose_feedback", "cartesian_motion", "gripper", "stop"]
    assert description["supports_request_ids"] and description["supports_deadlines"]
    assert description["supports_speed_limits"] and not description["object_feedback"]
    assert description["supports_invocation_lease"]
    assert not description["station_validated"]
    state = call(bridge, {"op": "hub_state"})
    assert state["pose"]["frame_id"] == "yam_left_base"
    assert state["source"] == "encoder_fk"
    assert state["gripper"]["holding"] is None
    assert state["gripper"]["width_m"] == pytest.approx(0.094)
    assert time.time() - state["timestamp_s"] < 0.1


def test_factory_refuses_powered_by_default_and_dry_is_labeled():
    with pytest.raises(Rejected, match="on-site"):
        create_bridge(Station())
    station = Station(powered=False, mode="dry")
    bridge = create_bridge(station, rate_limit_provider=lambda: 0.05)
    bridge.claim("test_owner")
    try:
        assert bridge.describe()["kind"] == "dry-run"
        assert bridge.describe()["virtual"]
        assert bridge.state()["source"] == "dry_model"
        result = call(bridge, move(position_m=[0.001, 0, 0.2]))
        assert result["ok"] and result["kind"] == "dry-run"
        assert result["virtual"]
        assert station.hw.events == []
        assert not call(bridge, move("too_slow", speed_m_s=0.03))["ok"]
        assert not bridge.describe()["operator_hold_latched"]
    finally:
        bridge.close()


def test_move_prepares_then_commits_and_reads_pose(rig):
    bridge, station = rig
    result = call(bridge, move())
    assert result["ok"] and result["command_id"] == "move_1"
    assert result["pose"]["position_m"] == pytest.approx([0.02, 0, 0.2])
    assert event_names(station) == ["commit"]
    assert result["timestamp_s"] > 0


@pytest.mark.parametrize("cap", [0.3, None])
def test_low_or_unknown_speed_cap_rejected_before_any_hardware_command(rig, cap):
    bridge, station = rig
    bridge.rate_limit_provider = None if cap is None else lambda: cap
    result = call(bridge, move(speed_m_s=0.03))
    assert not result["ok"] and "controller cap" in result["error"]["message"]
    assert station.hw.events == []
    assert station.plan_calls == []


@pytest.mark.parametrize("change", [{"frame_id": "map"}, {"linear": 1}, {"speed_m_s": -1}])
def test_bad_input_never_changes_hardware(rig, change):
    bridge, station = rig
    assert not call(bridge, move(**change))["ok"]
    assert station.hw.events == []


def test_completed_request_is_not_replayed(rig):
    bridge, station = rig
    first = call(bridge, move())
    assert call(bridge, move()) == first
    assert not call(bridge, move(position_m=[0.03, 0, 0.2]))["ok"]
    assert event_names(station) == ["commit"]


def test_timeout_while_operator_waits_cannot_later_commit(rig):
    bridge, station = rig
    entered, release = threading.Event(), threading.Event()

    def late_confirmation(*_):
        entered.set()
        assert release.wait(2)
        return True

    bridge.confirmation = late_confirmation
    try:
        started = time.monotonic()
        result = call(bridge, move(timeout_s=0.1))
        assert entered.is_set()
        assert not result["ok"] and time.monotonic() - started < 0.5
        assert "deadline" in result["error"]["message"]
        assert "commit" not in event_names(station)
        assert_hold(station)
    finally:
        release.set()
    bridge._planning.result(timeout=1)
    assert "commit" not in event_names(station)


def test_cancel_during_native_prepare_prevents_late_commit(rig):
    bridge, station = rig
    release = threading.Event()
    station.hw.prepare_release = release
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(call, bridge, move())
        try:
            assert station.hw.prepare_entered.wait(0.5)
            started = time.monotonic()
            state = call(bridge, {"op": "hub_state"})
            stopped = call(bridge, {"op": "hub_stop", "command_id": "move_1"})
            assert time.monotonic() - started < 0.2
            assert state["kind"] == "real" and stopped["ok"] and stopped["powered_hold"]
            result = future.result(timeout=0.5)
            assert not result["ok"] and result["status"] == "cancelled"
        finally:
            release.set()
    bridge._planning.result(timeout=1)
    assert "commit" not in event_names(station)
    assert_hold(station)


def test_cancelled_queued_request_does_not_stop_active_request(rig):
    bridge, station = rig
    entered, release = threading.Event(), threading.Event()

    def confirmation(*_):
        entered.set()
        assert release.wait(2)
        return True

    bridge.confirmation = confirmation
    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(call, bridge, move("active"))
        try:
            assert entered.wait(0.5)
            queued = pool.submit(call, bridge, move("queued"))
            # Cancel-before-arrival tombstones cover both scheduling outcomes.
            result = call(bridge, {"op": "hub_stop", "command_id": "queued"})
            assert result["ok"] and not result["powered_hold"]
            assert not queued.result(timeout=0.5)["ok"]
            assert "hold" not in event_names(station)
        finally:
            release.set()
        assert active.result(timeout=1)["ok"]
    assert event_names(station) == ["commit"]


def test_unknown_cancel_tombstone_blocks_late_request(rig):
    bridge, station = rig
    assert call(bridge, {"op": "hub_stop", "command_id": "late"})["ok"]
    assert call(bridge, move("late"))["status"] == "cancelled"
    assert station.hw.events == []


def test_stop_completed_last_command_holds_but_superseded_id_does_not(rig):
    bridge, station = rig
    assert call(bridge, move("first"))["ok"]
    assert call(bridge, move("second"))["ok"]
    superseded = call(bridge, {"op": "hub_stop", "command_id": "first"})
    assert superseded["status"] == "superseded" and not superseded["powered_hold"]
    assert "hold" not in event_names(station)
    stopped = call(bridge, {"op": "hub_stop", "command_id": "second"})
    assert stopped["ok"] and stopped["powered_hold"]
    assert_hold(station)
    assert not call(bridge, move("third"))["ok"]
    assert event_names(station).count("commit") == 2


@pytest.mark.parametrize("reach,observe", [(False, True), (True, False)])
def test_pose_requires_reached_and_distinct_measured_observations(rig, reach, observe):
    bridge, station = rig
    station.hw.reach_pose, station.hw.observe_new = reach, observe
    result = call(bridge, move(position_m=[0.1, 0, 0.2], timeout_s=0.16))
    assert not result["ok"]
    assert_hold(station)


def test_gripper_only_never_sends_an_arm_target(rig):
    bridge, station = rig
    result = call(bridge, grip())
    assert result["ok"] and result["completion"] == "width_reached"
    assert result["gripper"]["width_m"] == pytest.approx(0.02)
    assert result["gripper"]["holding"] is None
    assert event_names(station) == ["gripper", "hold"]
    assert ("hold", False, True) in station.hw.events
    assert not station.hw.held


def test_closing_force_limit_completes_without_claiming_holding(rig):
    bridge, station = rig
    station.hw.reach_grip = False
    station.hw.contact = True
    result = call(bridge, grip(width_m=0))
    assert result["ok"] and result["completion"] == "force_limited"
    assert result["gripper"]["width_m"] > 0
    assert result["gripper"]["holding"] is None
    assert result["source"] == "encoder_fk"
    assert ("hold", False, True) in station.hw.events
    assert not station.hw.held


def test_opening_cannot_reuse_closing_force_limit(rig):
    bridge, station = rig
    station.hw._state.gripper_closure = 0.8
    station.hw.reach_grip = False
    station.hw.contact = True
    result = call(bridge, grip(width_m=0.094, timeout_s=0.16))
    assert not result["ok"]
    assert_hold(station)


def test_no_force_evidence_cannot_claim_gripper_completion(rig):
    bridge, station = rig
    station.hw.reach_grip = False
    result = call(bridge, grip(width_m=0, timeout_s=0.16))
    assert not result["ok"]
    assert_hold(station)


def test_command_registry_never_evicts_ids_to_replay(rig):
    bridge, station = rig
    bridge.max_commands = 1
    assert call(bridge, move())["ok"]
    assert not call(bridge, move("new"))["ok"]
    assert call(bridge, move())["ok"]
    assert event_names(station) == ["commit"]


def rpc(address, request):
    if request["op"] == "hub_stop" and "owner_id" not in request:
        request = {**request, "owner_id": "test_owner"}
    with socket.create_connection(address, timeout=1) as stream:
        payload = json.dumps({**request, "arrays": []}).encode()
        stream.sendall(struct.pack("!I", len(payload)) + payload)

        def exact(count):
            data = bytearray()
            while len(data) < count:
                piece = stream.recv(count - len(data))
                assert piece
                data.extend(piece)
            return bytes(data)

        (size,) = struct.unpack("!I", exact(4))
        return json.loads(exact(size))


def test_threaded_wire_server_allows_state_and_cancel_during_confirmation(rig):
    bridge, station = rig
    entered = threading.Event()

    def confirmation(_message, cancel, _deadline):
        entered.set()
        cancel.wait(1)
        return True

    bridge.confirmation = confirmation
    server = make_server(bridge, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(rpc, server.server_address, move())
            assert entered.wait(0.5)
            state = rpc(server.server_address, {"op": "hub_state"})
            assert state["kind"] == "real"
            result = rpc(server.server_address, {"op": "hub_stop", "command_id": "move_1"})
            assert result["ok"] and result["powered_hold"]
            assert not future.result(timeout=0.5)["ok"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)
    assert "commit" not in event_names(station)
    assert_hold(station)


def test_server_is_local_only_and_legacy_operations_are_never_forwarded(rig):
    bridge, station = rig
    with pytest.raises(Rejected, match="localhost"):
        make_server(bridge, host="0.0.0.0")
    assert not call(bridge, {"op": "finish"})["ok"]
    assert station.hw.events == []


def test_unleased_motion_rejected_while_read_only_state_remains_available(rig):
    bridge, station = rig
    bridge.release("test_owner")
    assert call(bridge, {"op": "hub_state"})["kind"] == "real"
    assert not call(bridge, move())["ok"]
    assert not call(bridge, grip())["ok"]
    assert not call(bridge, {"op": "hub_stop", "command_id": "unknown"})["ok"]
    assert station.hw.events == []


def test_lease_covers_multiple_primitives_until_explicit_release(rig):
    bridge, station = rig
    assert call(bridge, grip())["ok"]
    busy = call(bridge, {"op": "hub_claim", "owner_id": "second_owner"})
    assert not busy["ok"] and busy["status"] == "busy" and "error" not in busy
    assert busy["owner_id"] == "second_owner"
    assert not call(bridge, move("other_move", owner_id="second_owner"))["ok"]
    assert call(bridge, move())["owner_id"] == "test_owner"
    assert not call(bridge, {"op": "hub_claim", "owner_id": "second_owner"})["ok"]
    assert call(bridge, {"op": "hub_release", "owner_id": "test_owner"})["ok"]
    assert call(bridge, {"op": "hub_claim", "owner_id": "second_owner"})["ok"]
    assert call(bridge, move("new_move", owner_id="second_owner"))["ok"]
    assert event_names(station).count("commit") == 2
    assert "operator_hold" not in event_names(station)


def test_lease_claim_release_do_not_command_hardware_and_other_owner_cannot_release(rig):
    bridge, station = rig
    assert call(bridge, {"op": "hub_claim", "owner_id": "test_owner"})["status"] == "already_owned"
    assert not call(bridge, {"op": "hub_release", "owner_id": "wrong_owner"})["ok"]
    assert bridge.describe()["invocation_owned"]
    assert call(bridge, {"op": "hub_release", "owner_id": "test_owner"})["ok"]
    assert not bridge.describe()["invocation_owned"]
    assert station.hw.events == []


def test_lease_cannot_release_an_active_confirmation(rig):
    bridge, station = rig
    entered, release = threading.Event(), threading.Event()

    def confirmation(*_):
        entered.set()
        assert release.wait(2)
        return True

    bridge.confirmation = confirmation
    with ThreadPoolExecutor(max_workers=1) as pool:
        action = pool.submit(call, bridge, move())
        try:
            assert entered.wait(0.5)
            assert not call(bridge, {"op": "hub_release", "owner_id": "test_owner"})["ok"]
            assert not call(bridge, {"op": "hub_claim", "owner_id": "second_owner"})["ok"]
        finally:
            release.set()
        assert action.result(timeout=1)["ok"]
    assert call(bridge, {"op": "hub_release", "owner_id": "test_owner"})["ok"]
    assert event_names(station) == ["commit"]


def test_release_refuses_queued_command_before_it_becomes_active(rig):
    bridge, station = rig
    bridge._motion.acquire()
    with ThreadPoolExecutor(max_workers=1) as pool:
        queued = pool.submit(call, bridge, move("queued"))
        try:
            deadline = time.monotonic() + 0.5
            while "queued" not in bridge._inflight and time.monotonic() < deadline:
                time.sleep(0.001)
            assert "queued" in bridge._inflight and bridge._active is None
            assert not call(bridge, {"op": "hub_release", "owner_id": "test_owner"})["ok"]
            cancellation = call(bridge, {"op": "hub_stop", "command_id": "queued"})
            assert cancellation["status"] == "cancelled_before_commit"
            assert cancellation["cancel_confirmed"] and not cancellation["command_committed"]
            assert not cancellation["hold_confirmed"] and not cancellation["other_command_active"]
            assert not queued.result(timeout=0.5)["ok"]
        finally:
            bridge._motion.release()
    assert call(bridge, {"op": "hub_release", "owner_id": "test_owner"})["ok"]
    assert station.hw.events == []


def test_late_previous_owner_stop_cannot_touch_new_owner_or_reuse_old_id(rig):
    bridge, station = rig
    assert call(bridge, move("first"))["ok"]
    bridge.release("test_owner")
    bridge.claim("second_owner")
    assert call(bridge, move("second", owner_id="second_owner"))["ok"]
    assert not call(bridge, {"op": "hub_stop", "owner_id": "test_owner", "command_id": "first"})[
        "ok"
    ]
    assert not call(bridge, {"op": "hub_stop", "owner_id": "second_owner", "command_id": "first"})[
        "ok"
    ]
    assert not call(bridge, move("first", owner_id="second_owner"))["ok"]
    assert "hold" not in event_names(station)
    assert event_names(station).count("commit") == 2
    stopped = call(bridge, {"op": "hub_stop", "owner_id": "second_owner", "command_id": "second"})
    assert stopped["hold_confirmed"] and stopped["powered_hold"]
    assert stopped["owner_id"] == "second_owner"


def test_failed_uncommitted_step_stop_holds_own_previous_hardware_goal(rig):
    bridge, station = rig
    assert call(bridge, move("completed"))["ok"]
    assert not call(bridge, move("rejected", speed_m_s=0.03))["ok"]
    stopped = call(bridge, {"op": "hub_stop", "command_id": "rejected"})
    assert stopped["status"] == "held" and stopped["hold_confirmed"]
    assert stopped["powered_hold"] and not stopped["command_committed"]
    assert_hold(station)


def test_stop_before_first_arrival_proves_no_commit_without_claiming_a_hold(rig):
    bridge, station = rig
    stopped = call(bridge, {"op": "hub_stop", "command_id": "never_arrived"})
    assert stopped["status"] == "cancelled_before_commit"
    assert stopped["cancel_confirmed"] and not stopped["command_committed"]
    assert not stopped["hold_confirmed"] and not stopped["powered_hold"]
    assert not stopped["other_command_active"]
    assert call(bridge, move("never_arrived"))["status"] == "cancelled"
    assert station.hw.events == []


def test_unconfirmed_hold_keeps_ownership_latched_after_command_completed(rig):
    bridge, station = rig
    assert call(bridge, move())["ok"]

    def failed_hold(**_kwargs):
        raise RuntimeError("native hold not confirmed")

    station.hw.hold = failed_hold
    stopped = call(bridge, {"op": "hub_stop", "command_id": "move_1"})
    assert not stopped["ok"]
    assert bridge.describe()["ownership_uncertain"]
    release = call(bridge, {"op": "hub_release", "owner_id": "test_owner"})
    assert not release["ok"] and "unconfirmed hold" in release["error"]["message"]
    assert call(bridge, {"op": "hub_claim", "owner_id": "second_owner"})["status"] == "busy"
    assert event_names(station).count("commit") == 1


def test_failed_deadline_hold_cannot_release_after_active_flag_clears(rig):
    bridge, station = rig
    station.hw.reach_pose = False

    def failed_hold(**_kwargs):
        raise RuntimeError("native hold failed")

    station.hw.hold = failed_hold
    result = call(bridge, move(position_m=[0.1, 0, 0.2], timeout_s=0.12))
    assert not result["ok"] and result["stop_error"] == "native hold failed"
    assert bridge._active is None and not bridge._inflight
    assert not call(bridge, {"op": "hub_release", "owner_id": "test_owner"})["ok"]
    assert bridge.describe()["ownership_uncertain"]
