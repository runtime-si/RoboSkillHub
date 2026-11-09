"""Versioned, cancellable RPC around the existing yam_gap.real.motion_server.Station.

Source contract inspected on YAM workcell, 2026-11-06: <YAM_CONTROL_ROOT>/real/
{motion_server.py,wire.py} and <YAM_MODEL_ROOT>/hardware.py.
This new module does not replace those files, construct Hardware, start motors,
change controller globals, call finish/reset, or disable torque. The station's
owner creates and (only on site) powers it using its existing confirmation flow.

Use create_bridge(existing_station) for dry mode; a powered owner must explicitly
pass allow_powered=True. make_server(...).serve_forever() runs on a separate port,
using the same length-prefixed JSON wire format without the old global RPC lock.
The owner must give this bridge exclusive policy-command access to that station;
do not also serve the legacy move/grip API or change its rate globals mid-session.
Cancel/state requests remain independent of the motion lock. Native planning runs
in a single computation worker and cannot commit actuator commands by itself.
Cancel and commit share a short critical section; canceled plans cannot later run.
One owner must claim a lease around the entire invocation, including all primitive
commands and outcome checks. The lease has no expiry or automatic takeover.

A speed request is a maximum TCP command rate. The reviewed controller has only
process-wide caps: requests below its existing cap are REJECTED, never simulated
by sleeping or by silently ignoring a requested speed. Physical overshoot is not
certified by this adapter. A hardware lock/native call that never returns still
requires the station's independent supervisor or physical stop.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import select
import socketserver
import struct
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contracts import ExecutionFailed, Rejected, number, pose_error, validate_pose

FRAME = "yam_left_base"
PROTOCOL = "roboskillhub-yam"
_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class Cancelled(ExecutionFailed):
    """This command was canceled before any further actuator commit."""


@dataclass
class _Command:
    command_id: str
    fingerprint: str
    deadline: float
    owner_id: str
    cancel: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    committed: bool = False


def _terminal_confirm(message: str, cancel: threading.Event, deadline: float) -> bool:
    if not sys.stdin.isatty():
        raise Rejected("confirm mode requires the station owner's interactive terminal")
    print(
        f"\n>>> {message}\n    Enter = execute; n/q = cancel and hold (no home/torque-off)",
        flush=True,
    )
    pending = b""
    while not cancel.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ExecutionFailed("confirmation deadline expired")
        ready, _, _ = select.select([sys.stdin.fileno()], [], [], min(0.1, remaining))
        if ready:
            chunk = os.read(sys.stdin.fileno(), 4096)
            if not chunk:
                return False
            pending += chunk
            if b"\n" in pending or b"\r" in pending:
                return not pending.splitlines()[0].strip()
    raise Cancelled("canceled while waiting for operator confirmation")


class Bridge:
    """Wrap a station already owned by its lifecycle supervisor; never power it.

    Optional rate_limit_provider is for a separately verified controller adapter
    (and contract tests). It returns the actual TCP setpoint cap, not a desired
    rate. The native path reads the reviewed _bounded_candidate globals.
    """

    def __init__(
        self,
        station: Any,
        *,
        allow_powered: bool = False,
        confirmation: Callable | None = None,
        rate_limit_provider: Callable | None = None,
        max_timeout_s: float = 120,
        max_commands: int = 1024,
    ):
        self.station, self.allow_powered = station, allow_powered
        if station.arm != "left":
            raise Rejected("this bridge binds only the left YAM base frame")
        if station.mode not in {"dry", "confirm", "auto"}:
            raise Rejected("unknown station mode")
        if station.powered and not allow_powered:
            raise Rejected(
                "powered bridge must be attached explicitly by the on-site station owner"
            )
        if station.powered and station.mode == "dry":
            raise Rejected("a powered station cannot advertise dry mode")
        info, _ = station.handle({"op": "describe"}, {})
        self.info = copy.deepcopy(info)
        self.span = number(info["grip_span_m"], "grip_span_m", 0.001, 0.5)
        self.max_timeout = number(max_timeout_s, "max_timeout_s", 0.1, 120)
        self.max_commands = int(max_commands)
        if self.max_commands < 1:
            raise Rejected("max_commands must be positive")
        self.confirmation = confirmation or _terminal_confirm
        self.rate_limit_provider = rate_limit_provider
        self._motion = threading.Lock()
        self._gate = threading.RLock()
        self._registry = threading.Lock()
        self._records: dict[str, _Command] = {}
        self._active: _Command | None = None
        self._last_committed: _Command | None = None
        self._owner_id: str | None = None
        self._inflight: set[str] = set()
        self._unsafe_ownership = False
        self._latched = False
        self._wall_minus_monotonic = time.time() - time.monotonic()
        self._planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yam-hub-plan")
        self._planning = None
        self._dry_state = self._read_station()

    def _rate(self) -> float | None:
        if self.rate_limit_provider is not None:
            value = self.rate_limit_provider()
        else:
            method = getattr(type(self.station.hw), "_bounded_candidate", None)
            values = getattr(method, "__globals__", {})
            value = values.get("EE_RATE_M_S")
        if value is None:
            return None
        # The reviewed Cartesian feedback path has its own 0.01 m/s cap.
        return max(0.01, number(float(value), "controller TCP rate cap", 1e-5, 1))

    def describe(self) -> dict:
        rate = self._rate()
        return {
            "protocol": PROTOCOL,
            "protocol_version": 1,
            "backend": "yam",
            "kind": "real" if self.station.powered else "dry-run",
            "virtual": not self.station.powered,
            "mode": self.station.mode,
            "embodiment": "yam_i2rt",
            "frame_id": FRAME,
            "capabilities": ["pose_feedback", "cartesian_motion", "gripper", "stop"],
            "supports_request_ids": True,
            "supports_deadlines": True,
            "supports_speed_limits": rate is not None,
            "supports_invocation_lease": True,
            "invocation_owned": self._owner_id is not None,
            "ownership_uncertain": self._unsafe_ownership,
            "lease_policy": "explicit whole-invocation claim/release; no expiry or automatic takeover",
            "controller_tcp_rate_cap_m_s": rate,
            "speed_limit_policy": "reject a requested maximum below the existing controller cap",
            "linear_semantics": "true uses bounded Cartesian waypoints with joint paths between them; false also selects this restricted path; no continuous line or collision guarantee",
            "operator_hold_latched": self._latched,
            "powered": bool(self.station.powered),
            "gripper_span_m": self.span,
            "bounds": self.info["bounds"],
            "max_timeout_s": self.max_timeout,
            "object_feedback": False,
            "station_validated": False,
        }

    @staticmethod
    def _pose(pose: Any) -> dict:
        return {
            "frame_id": FRAME,
            "position_m": list(map(float, pose.position_m)),
            "quaternion_xyzw": list(map(float, pose.orientation_xyzw)),
        }

    def _read_station(self) -> dict:
        if self.station.powered:
            hw = self.station.hw
            with hw._lock:
                state = hw.state()
                stamp = self._wall_minus_monotonic + float(hw._updated)
            opening = 1.0 - float(state.gripper_closure)
            pose = state.ee_pose
            diagnostics = dict(getattr(state, "diagnostics", {}) or {})
            source = "encoder_fk"
        else:
            _q, pose, opening = self.station.current()
            stamp, diagnostics, source = time.time(), {}, "dry_model"
        gripper = {
            "width_m": float(opening) * self.span,
            "holding": None,
            "timestamp_s": stamp,
            "source": source,
        }
        for key in ("gripper_effort_nm", "gripper_filtered_effort_nm", "gripper_force_backoff"):
            if key in diagnostics:
                gripper[key] = diagnostics[key]
        return {
            "pose": self._pose(pose),
            "timestamp_s": stamp,
            "objects": {},
            "gripper": gripper,
            "source": source,
            "kind": "real" if self.station.powered else "dry-run",
            "virtual": not self.station.powered,
        }

    def state(self) -> dict:
        if self.station.powered:
            return self._read_station()
        # Dry FK uses Station's mutable MjData, which its planner also uses.
        # Cache the last committed virtual state instead of racing FK in state RPC.
        with self._gate:
            result = copy.deepcopy(self._dry_state)
            result["timestamp_s"] = time.time()
            result["gripper"]["timestamp_s"] = result["timestamp_s"]
            return result

    def _check(self, cmd: _Command) -> float:
        if cmd.cancel.is_set():
            raise Cancelled("command canceled")
        remaining = cmd.deadline - time.monotonic()
        if remaining <= 0:
            raise ExecutionFailed("command deadline expired")
        if self._latched:
            raise Rejected(
                "bridge hold is latched; station owner must recover and create a new bridge"
            )
        if self.station.powered and not self.allow_powered:
            raise Rejected("powered execution was not enabled by the station owner")
        if self.station.powered and self.station.mode not in {"confirm", "auto"}:
            raise Rejected("powered execution requires explicit confirm or auto mode")
        self._require_owner(cmd.owner_id)
        return remaining

    def _require_owner(self, owner_id: str) -> None:
        self._valid_id(owner_id)
        if self._owner_id != owner_id:
            raise Rejected("request owner does not hold the station invocation lease")

    def claim(self, owner_id: str) -> dict:
        """Nonblocking ownership of an entire skill, not one primitive RPC.

        A crashed owner is never replaced automatically: after confirming that
        its process cannot continue, the station operator must recreate the bridge.
        Claiming and releasing do not command the arm.
        """
        self._valid_id(owner_id)
        with self._gate:
            if self._owner_id is not None and self._owner_id != owner_id:
                return {
                    "ok": False,
                    "owner_id": owner_id,
                    "status": "busy",
                    "message": "station invocation lease is already owned",
                }
            status = "already_owned" if self._owner_id == owner_id else "claimed"
            self._owner_id = owner_id
        return {"ok": True, "owner_id": owner_id, "status": status}

    def release(self, owner_id: str) -> dict:
        with self._gate:
            self._require_owner(owner_id)
            if self._unsafe_ownership:
                raise Rejected(
                    "cannot release after an unconfirmed hold; operator must recover and recreate the bridge"
                )
            if self._inflight:
                raise Rejected("cannot release while an invocation command is active or queued")
            self._owner_id = None
        return {"ok": True, "owner_id": owner_id, "status": "released"}

    def _hold(self) -> None:
        # Caller holds _gate, the same critical section used by commit_target.
        self._latched = True
        if self.station.powered:
            hw = self.station.hw
            try:
                with hw._lock:
                    hw.hold(measured=True, preserve_gripper=True)
                    hw.enter_operator_hold()
            except BaseException:
                # Clearing an RPC's active flag never proves that the controller
                # is held. Keep this lease until the station owner recovers it.
                self._unsafe_ownership = True
                raise

    def stop(self, command_id: str, owner_id: str) -> dict:
        self._valid_id(command_id)
        held = False
        with self._gate:
            # Keep authorization and hold atomic with lease handoff and commit.
            self._require_owner(owner_id)
            with self._registry:
                cmd = self._records.get(command_id)
                if cmd is None:
                    if len(self._records) >= self.max_commands:
                        raise Rejected("command registry full; refusing a cancellation tombstone")
                    cmd = _Command(
                        command_id, "cancelled-before-arrival", time.monotonic(), owner_id
                    )
                    self._records[command_id] = cmd
                if cmd.owner_id != owner_id:
                    raise Rejected("command ID belongs to another invocation owner")
                cmd.cancel.set()
            own_prior_goal = (
                self._last_committed is not None and self._last_committed.owner_id == owner_id
            )
            if self._active is cmd or (
                self._active is None
                and (self._last_committed is cmd or (not cmd.committed and own_prior_goal))
            ):
                self._hold()
                held = True
                status = "held"
            elif cmd.committed:
                status = "superseded"
            else:
                status = "cancelled_before_commit"
            other_active = self._active is not None and self._active is not cmd
        return {
            "ok": True,
            "owner_id": owner_id,
            "command_id": command_id,
            "status": status,
            "cancel_confirmed": True,
            "command_committed": cmd.committed,
            "other_command_active": other_active,
            "hold_confirmed": held,
            "powered_hold": held and bool(self.station.powered),
            "timestamp_s": time.time(),
        }

    @staticmethod
    def _valid_id(value: Any) -> None:
        if not isinstance(value, str) or not _ID.fullmatch(value):
            raise Rejected("command_id must be 1..128 letters, digits, underscore or hyphen")

    def _plan(self, cmd: _Command, function: Callable, *args, **kwargs):
        self._check(cmd)
        if self._planning is not None and not self._planning.done():
            raise Rejected(
                "previous native planning is still running; no second planning job started"
            )
        self._planning = self._planner.submit(function, *args, **kwargs)
        while True:
            try:
                result = self._planning.result(timeout=min(0.05, self._check(cmd)))
                self._check(cmd)
                return result
            except FutureTimeout:
                continue

    def _confirm(self, cmd: _Command, message: str) -> None:
        if self.station.mode == "confirm":
            # Confirmation is computation/input only; even a delayed custom
            # callback cannot block the RPC deadline or commit an actuator goal.
            if not self._plan(cmd, self.confirmation, message, cmd.cancel, cmd.deadline):
                raise Cancelled("operator refused the command")
        self._check(cmd)

    def _pause(self, cmd: _Command, duration: float) -> None:
        end = time.monotonic() + duration
        while time.monotonic() < end:
            cmd.cancel.wait(min(0.05, end - time.monotonic(), self._check(cmd)))
        self._check(cmd)

    def _move(self, cmd: _Command, request: dict) -> dict:
        pose = validate_pose(
            {
                "frame_id": request["frame_id"],
                "position_m": request["position_m"],
                "quaternion_xyzw": request["quat_xyzw"],
            },
            FRAME,
        )
        speed = number(request["speed_m_s"], "speed_m_s", 1e-5, 1)
        if type(request["linear"]) is not bool:
            raise Rejected("linear must be boolean")
        rate = self._rate()
        if rate is None or speed + 1e-12 < rate:
            raise Rejected(
                f"requested TCP maximum {speed:g} m/s is below or unsupported by existing controller cap {rate}; no global rate changed"
            )
        plan = self._plan(cmd, self.station.plan, pose["position_m"], pose["quaternion_xyzw"])
        if self.station.mode == "auto" and (
            plan["distance_m"] > 0.15 or plan["turn_rad"] > math.radians(50)
        ):
            raise Rejected("existing auto limits exceeded: 0.15 m / 50 degrees")
        self._confirm(cmd, f"HUB MOVE {cmd.command_id}: {pose['position_m']} at max {speed:g} m/s")
        if not self.station.powered:
            self._pause(cmd, float(plan["distance_m"]) / speed)
            with self._gate:
                self._check(cmd)
                self.station.q = np.asarray(plan["steps"][-1]["joints"], dtype=float).copy()
                self._dry_state = self._read_station()
                cmd.committed = True
                self._last_committed = cmd
        else:
            for step in plan["steps"]:
                duration = min(6.0, self._check(cmd))
                _, _, opening = self.station.current()
                prepared = self._plan(
                    cmd,
                    self.station.hw.prepare_target,
                    step["pose"],
                    1.0 - opening,
                    duration_s=duration,
                )
                with self._gate:
                    self._check(cmd)
                    self.station.hw.commit_target(prepared)
                    cmd.committed = True
                    self._last_committed = cmd
                self._wait_pose(cmd, self._pose(step["pose"]))
        state = self.state()
        return {
            "ok": True,
            "command_id": cmd.command_id,
            "timestamp_s": state["timestamp_s"],
            "pose": state["pose"],
            "kind": state["kind"],
            "virtual": state["virtual"],
            "steps": len(plan["steps"]),
            "path": "bounded Cartesian waypoints",
            "requested_linear": request["linear"],
        }

    def _wait_pose(self, cmd: _Command, target: dict) -> None:
        stable = 0
        last_stamp = -math.inf
        while True:
            self._check(cmd)
            state = self.state()
            error, rotation = pose_error(state["pose"], target)
            if state["timestamp_s"] > last_stamp:
                last_stamp = state["timestamp_s"]
                stable = stable + 1 if error <= 0.012 and rotation <= 0.04 else 0
            if stable >= 3:
                return
            self._pause(cmd, 0.05)

    def _grip(self, cmd: _Command, request: dict) -> dict:
        width = number(request["width_m"], "width_m", 0, self.span)
        initial = self.state()["gripper"]["width_m"]
        self._confirm(cmd, f"HUB GRIP {cmd.command_id}: width {width:.4f} m")
        with self._gate:
            self._check(cmd)
            if self.station.powered:
                self.station.hw.gripper_target(
                    1.0 - width / self.span, duration_s=min(6.0, self._check(cmd))
                )
            else:
                self.station.gripper_open = width / self.span
                self._dry_state = self._read_station()
            cmd.committed = True
            self._last_committed = cmd
        stable = contact = 0
        last_stamp = -math.inf
        while True:
            self._check(cmd)
            state = self.state()
            g = state["gripper"]
            if state["timestamp_s"] > last_stamp:
                last_stamp = state["timestamp_s"]
                stable = stable + 1 if abs(g["width_m"] - width) <= 0.001 else 0
                # Existing controller's force-backoff is an observed force limit,
                # not an assertion that an object is held. Opening cannot use it.
                backoff = g.get("gripper_force_backoff", 0)
                contact = (
                    contact + 1
                    if width < initial and isinstance(backoff, (int, float)) and backoff > 0
                    else 0
                )
            if stable >= 3 or contact >= 3:
                # A force-limited close must not keep advancing its old goal
                # after returning. Clear that goal while retaining both the arm
                # setpoint and capped gripper command; this is not a latched stop.
                with self._gate:
                    self._check(cmd)
                    if self.station.powered:
                        self.station.hw.hold(measured=False, preserve_gripper=True)
                return {
                    "ok": True,
                    "command_id": cmd.command_id,
                    "timestamp_s": state["timestamp_s"],
                    "gripper": g,
                    "completion": "width_reached" if stable >= 3 else "force_limited",
                    "source": g["source"],
                    "kind": state["kind"],
                    "virtual": state["virtual"],
                }
            self._pause(cmd, 0.05)

    def command(self, op: str, request: dict) -> dict:
        required = {"op", "owner_id", "command_id", "timeout_s"}
        required |= (
            {"frame_id", "position_m", "quat_xyzw", "speed_m_s", "linear"}
            if op == "hub_move"
            else {"width_m"}
        )
        if set(request) != required:
            raise Rejected(f"{op} requires exactly {sorted(required)}")
        command_id = request["command_id"]
        owner_id = request["owner_id"]
        self._valid_id(command_id)
        timeout = number(request["timeout_s"], "timeout_s", 0.001, self.max_timeout)
        fingerprint = json.dumps(request, sort_keys=True, allow_nan=False)
        with self._gate:
            self._require_owner(owner_id)
            with self._registry:
                previous = self._records.get(command_id)
                if previous is not None:
                    if previous.owner_id != owner_id:
                        raise Rejected("command ID belongs to another invocation owner")
                    if previous.cancel.is_set():
                        return {
                            "ok": False,
                            "owner_id": owner_id,
                            "command_id": command_id,
                            "status": "cancelled",
                        }
                    if previous.fingerprint != fingerprint:
                        raise Rejected("command_id already belongs to a different request")
                    return (
                        copy.deepcopy(previous.result)
                        if previous.result
                        else {
                            "ok": False,
                            "owner_id": owner_id,
                            "command_id": command_id,
                            "status": "already_running",
                        }
                    )
                if len(self._records) >= self.max_commands:
                    raise Rejected("command registry full; no old IDs evicted or replayed")
                cmd = _Command(command_id, fingerprint, time.monotonic() + timeout, owner_id)
                self._records[command_id] = cmd
                self._inflight.add(command_id)
        acquired = False
        try:
            while not acquired:
                acquired = self._motion.acquire(timeout=min(0.05, self._check(cmd)))
            with self._gate:
                self._check(cmd)
                self._active = cmd
            result = self._move(cmd, request) if op == "hub_move" else self._grip(cmd, request)
            result["owner_id"] = owner_id
        except Exception as error:
            # Terminate a late confirmation worker as well as the RPC loop.
            cmd.cancel.set()
            stop_error = None
            with self._gate:
                # Pure input/plan rejections do not alter live setpoints. An
                # explicit cancel, elapsed deadline, or failed committed action
                # does hold the selected arm under its existing supervisor.
                if self._active is cmd and (cmd.committed or not isinstance(error, Rejected)):
                    try:
                        self._hold()
                    except Exception as hold_error:
                        stop_error = str(hold_error)
            result = {
                "ok": False,
                "owner_id": owner_id,
                "command_id": command_id,
                "status": "cancelled" if isinstance(error, Cancelled) else "failed",
                "error": {"code": type(error).__name__, "message": str(error)},
            }
            if stop_error:
                result["stop_error"] = stop_error
        finally:
            with self._gate:
                if self._active is cmd:
                    self._active = None
                with self._registry:
                    cmd.result = copy.deepcopy(result)
                self._inflight.discard(command_id)
            if acquired:
                self._motion.release()
        return result

    def handle(self, header: dict, arrays: dict | None = None) -> tuple[dict, dict]:
        request = dict(header)
        request.pop("arrays", None)
        try:
            if arrays:
                raise Rejected("Hub bridge requests do not accept ndarray payloads")
            op = request.get("op")
            if op == "hub_describe" and set(request) == {"op"}:
                return self.describe(), {}
            if op == "hub_state" and set(request) == {"op"}:
                return self.state(), {}
            if op == "hub_claim" and set(request) == {"op", "owner_id"}:
                return self.claim(request["owner_id"]), {}
            if op == "hub_release" and set(request) == {"op", "owner_id"}:
                return self.release(request["owner_id"]), {}
            if op == "hub_stop" and set(request) == {"op", "command_id", "owner_id"}:
                return self.stop(request["command_id"], request["owner_id"]), {}
            if op in {"hub_move", "hub_grip"}:
                return self.command(op, request), {}
            raise Rejected(
                "unsupported Hub op or unexpected request fields; legacy ops are not forwarded"
            )
        except Exception as error:
            return {
                "ok": False,
                "owner_id": request.get("owner_id"),
                "command_id": request.get("command_id"),
                "error": {"code": type(error).__name__, "message": str(error)},
            }, {}

    def close(self) -> None:
        with self._gate:
            self._latched = True
            with self._registry:
                for command_id in self._inflight:
                    self._records[command_id].cancel.set()
            if self._active is not None:
                self._hold()
        self._planner.shutdown(wait=False, cancel_futures=True)


def create_bridge(station: Any, **kwargs) -> Bridge:
    """Attach an already created station. Defaults refuse powered execution."""
    return Bridge(station, **kwargs)


def make_server(bridge: Bridge, *, host: str = "127.0.0.1", port: int = 9201):
    """Return a threaded localhost server; caller owns serve_forever/shutdown.

    Unlike legacy wire.serve, there is no global handler lock. Stop requests use
    another TCP connection and reach Bridge.stop while motion/confirmation waits.
    """
    if host != "127.0.0.1":
        raise Rejected("YAM bridge must bind localhost; use the existing SSH tunnel")

    class Handler(socketserver.BaseRequestHandler):
        def exact(self, count):
            value = bytearray()
            while len(value) < count:
                part = self.request.recv(count - len(value))
                if not part:
                    raise EOFError
                value.extend(part)
            return bytes(value)

        def handle(self):
            self.request.settimeout(10)
            try:
                while True:
                    (size,) = struct.unpack("!I", self.exact(4))
                    if not 0 < size <= 1 << 20:
                        raise ValueError("invalid header size")
                    header = json.loads(self.exact(size))
                    if not isinstance(header, dict) or header.get("arrays", []) != []:
                        raise ValueError("only JSON headers without array payloads are supported")
                    reply, _ = bridge.handle(header)
                    payload = json.dumps({**reply, "arrays": []}, allow_nan=False).encode()
                    self.request.sendall(struct.pack("!I", len(payload)) + payload)
            except (EOFError, OSError, ValueError):
                return

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = False

    return Server((host, port), Handler)
