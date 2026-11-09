"""SDK adapter for a site's existing YAM control and observation functions.

This module knows no HTTP endpoints, motor IDs, joint order, home position,
camera calibration, or YAM gripper conversion. A station factory explicitly
binds those details through callables; existing controller/confirmation behavior
stays inside those callables. This is not a driver or a tested connection to the
team's YAM installation.

Bound functions must honor timeout_s and retain the controller's stop behavior.
The adapter passes deadlines through and detects overruns after a call returns;
it cannot interrupt a stuck native/remote call. The station needs its own watchdog.
Command acknowledgements never establish measured pose, grasp, or task success.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from typing import Any

from .contracts import (
    EvidenceMissing,
    ExecutionFailed,
    Limits,
    Rejected,
    number,
    validate_pose,
    vector,
)


def _frame(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Rejected(f"{label} must name an explicitly calibrated frame")
    return value


def gap_pose_to_pose(value: dict, *, source_frame_id: str, frame_id: str) -> dict:
    """Convert GAP position/rotation dictionaries to metres and xyzw.

    GAP Se3Pose does not carry a frame tag. Its source frame must be supplied
    explicitly and already equal the station frame; this performs no calibration
    or transform. GAP coordinates must already be metres.
    """
    if _frame(source_frame_id, "source_frame_id") != _frame(frame_id, "frame_id"):
        raise Rejected(
            "GAP source frame differs from station frame; apply an explicit transform first"
        )
    if not isinstance(value, dict) or set(value) != {"position", "rotation"}:
        raise Rejected("GAP pose must contain exactly position and rotation")
    position, rotation = value["position"], value["rotation"]
    if not isinstance(position, dict) or set(position) != {"x", "y", "z"}:
        raise Rejected("GAP position must contain x,y,z in metres")
    if not isinstance(rotation, dict) or set(rotation) != {"w", "x", "y", "z"}:
        raise Rejected("GAP rotation must contain named quaternion components w,x,y,z")
    return validate_pose(
        {
            "frame_id": frame_id,
            "position_m": [position[k] for k in ("x", "y", "z")],
            "quaternion_xyzw": [rotation[k] for k in ("x", "y", "z", "w")],
        },
        frame_id,
    )


def pose_to_gap(value: dict, *, gap_world_frame_id: str) -> dict:
    """Convert a station pose only when its frame already equals GAP's world."""
    pose = validate_pose(value, _frame(gap_world_frame_id, "gap_world_frame_id"))
    x, y, z, w = pose["quaternion_xyzw"]
    return {
        "position": dict(zip(("x", "y", "z"), pose["position_m"], strict=True)),
        "rotation": {"w": w, "x": x, "y": y, "z": z},
    }


class YamBackend:
    """Bind normalized station callbacks to the Hub StationBackend interface.

    Callbacks are invoked exactly as:
      read_state(timeout_s=...) -> normalized state
      move_cartesian(pose, speed_m_s=..., timeout_s=..., linear=...) -> {ok: bool}
      set_gripper_width(width_m, timeout_s=...) -> {ok: bool}
      stop() -> None

    All positions and gripper widths are metres. No callback signature fallback
    is attempted: a site wrapper must explicitly support speed, deadline, and
    requested linear/nonlinear mode. It must never silently discard them.

    State pose and every object record require the station frame. Timestamps are
    original acquisition Unix seconds, not request or response times. An optional
    object_provider(timeout_s=...) can supply independently observed objects.
    object_feedback=True declares that read_state already supplies such records.
    """

    def __init__(
        self,
        *,
        read_state: Callable[..., dict],
        move_cartesian: Callable[..., dict],
        set_gripper_width: Callable[..., dict],
        stop: Callable[[], None],
        limits: Limits,
        object_feedback: bool = False,
        object_provider: Callable[..., dict] | None = None,
    ) -> None:
        if not isinstance(limits, Limits):
            raise Rejected("YamBackend requires explicit calibrated station Limits")
        _frame(limits.frame_id, "limits.frame_id")
        for label, callback in (
            ("read_state", read_state),
            ("move_cartesian", move_cartesian),
            ("set_gripper_width", set_gripper_width),
            ("stop", stop),
        ):
            if not callable(callback):
                raise Rejected(f"{label} must be an explicitly bound callable")
        if type(object_feedback) is not bool:
            raise Rejected("object_feedback must be boolean")
        if object_provider is not None and not callable(object_provider):
            raise Rejected("object_provider must be callable when supplied")
        if object_feedback and object_provider is not None:
            raise Rejected("choose read_state object_feedback or a separate object_provider")
        self.limits = limits
        self._read = read_state
        self._move = move_cartesian
        self._gripper = set_gripper_width
        self._stop = stop
        self._object_feedback = object_feedback
        self._objects = object_provider

    def describe(self) -> dict:
        capabilities = ["pose_feedback", "cartesian_motion", "gripper", "stop"]
        if self._object_feedback or self._objects is not None:
            capabilities.append("object_feedback")
            # Only a provider that states it reports quaternion_xyzw adds 6D object poses.
            if getattr(self._objects, "provides_orientation", False) is True:
                capabilities.append("object_pose_feedback")
        return {
            "backend": "yam-callables",
            "kind": "real",
            "embodiment": "yam_i2rt",
            "frame_id": self.limits.frame_id,
            "units": "m",
            "quaternion_order": "xyzw",
            "capabilities": capabilities,
            "station_validated": False,
            "connection": "explicit site callbacks; no built-in network transport",
            "timeout_semantics": "passed to site callbacks; overrun detected after return",
        }

    def _timeout(self, timeout_s: float) -> float:
        return number(timeout_s, "timeout_s", 1e-6, self.limits.max_timeout_s)

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ExecutionFailed("YAM site callback exceeded its timeout")
        return remaining

    def _timestamp(self, value: Any, label: str) -> float:
        try:
            timestamp = number(value, label, 1e-6, 1e12)
        except Rejected as exc:
            raise EvidenceMissing(str(exc)) from exc
        age = time.time() - timestamp
        if age > self.limits.max_state_age_s or age < -0.1:
            raise EvidenceMissing(f"{label} is stale or its clock is not synchronized")
        return timestamp

    def _objects_record(self, value: Any) -> dict:
        if not isinstance(value, dict):
            raise EvidenceMissing("objects must map object IDs to independently observed records")
        result = copy.deepcopy(value)
        for object_id, record in result.items():
            if not isinstance(object_id, str) or not object_id or not isinstance(record, dict):
                raise EvidenceMissing(
                    "object IDs and observation records must be nonempty strings and objects"
                )
            if record.get("frame_id") != self.limits.frame_id:
                raise EvidenceMissing(f"object {object_id!r} frame differs from station frame")
            if not isinstance(record.get("source"), str) or not record["source"].strip():
                raise EvidenceMissing(f"object {object_id!r} must name its observation source")
            if record["source"] in {"simulation_truth", "sim_truth", "command", "commanded"}:
                raise EvidenceMissing(
                    "real YAM object feedback cannot be simulated or commanded state"
                )
            record["timestamp_s"] = self._timestamp(record.get("timestamp_s"), "object timestamp_s")
            if record.get("visibility") in ("occluded", "not_found"):
                continue  # an explicit miss carries no position; observe_objects reports it
            try:
                record["position_m"] = vector(record.get("position_m"), "object position_m", 3)
            except Rejected as exc:
                raise EvidenceMissing(str(exc)) from exc
        return result

    def get_state(self, *, timeout_s: float) -> dict:
        timeout = self._timeout(timeout_s)
        deadline = time.monotonic() + timeout
        state = copy.deepcopy(self._read(timeout_s=timeout))
        self._remaining(deadline)
        if not isinstance(state, dict):
            raise EvidenceMissing("YAM read_state must return a normalized state object")
        try:
            state["pose"] = validate_pose(state.get("pose"), self.limits.frame_id)
        except Rejected as exc:
            raise EvidenceMissing(f"invalid measured YAM pose: {exc}") from exc
        state["timestamp_s"] = self._timestamp(state.get("timestamp_s"), "state timestamp_s")
        objects = state.get("objects", {})
        if self._objects is not None:
            if objects:
                raise EvidenceMissing(
                    "read_state and object_provider both supplied objects; choose one source"
                )
            objects = self._objects(timeout_s=self._remaining(deadline))
            self._remaining(deadline)
        state["objects"] = self._objects_record(objects)
        gripper = state.get("gripper", {"holding": None})
        if not isinstance(gripper, dict):
            raise EvidenceMissing("gripper feedback must be an object")
        holding = gripper.get("holding")
        if holding is not None and type(holding) is not bool:
            raise EvidenceMissing("gripper holding must be boolean or null")
        if holding is not None:
            self._timestamp(gripper.get("timestamp_s"), "gripper timestamp_s")
            if not isinstance(gripper.get("source"), str) or not gripper["source"].strip():
                raise EvidenceMissing("gripper holding feedback must name its measurement source")
        if "width_m" in gripper:
            try:
                number(gripper["width_m"], "gripper width_m", 0, 1)
            except Rejected as exc:
                raise EvidenceMissing(str(exc)) from exc
        state["gripper"] = gripper
        return state

    def _ack(self, value: Any, operation: str) -> dict:
        if not isinstance(value, dict) or type(value.get("ok")) is not bool:
            raise ExecutionFailed(
                f"YAM {operation} must return an explicit boolean ok acknowledgement"
            )
        acknowledgement = copy.deepcopy(value)
        if "timestamp_s" in acknowledgement:
            self._timestamp(acknowledgement["timestamp_s"], "ack timestamp_s")
        # ok means the site's command result. Runtime independently verifies pose
        # and object state; do not synthesize measured state from this reply.
        return acknowledgement

    def move_pose(self, pose: dict, *, speed_m_s: float, timeout_s: float, linear: bool) -> dict:
        target = self.limits.pose(pose)
        speed = number(speed_m_s, "speed_m_s", 1e-5, self.limits.max_speed_m_s)
        timeout = self._timeout(timeout_s)
        if type(linear) is not bool:
            raise Rejected("linear must explicitly be true or false")
        deadline = time.monotonic() + timeout
        acknowledgement = self._move(target, speed_m_s=speed, timeout_s=timeout, linear=linear)
        self._remaining(deadline)
        return self._ack(acknowledgement, "move_cartesian")

    def set_gripper(self, width_m: float, *, timeout_s: float) -> dict:
        width = number(
            width_m, "width_m", self.limits.gripper_min_width_m, self.limits.gripper_max_width_m
        )
        timeout = self._timeout(timeout_s)
        deadline = time.monotonic() + timeout
        acknowledgement = self._gripper(width, timeout_s=timeout)
        self._remaining(deadline)
        return self._ack(acknowledgement, "set_gripper_width")

    def stop(self) -> None:
        result = self._stop()
        if result is not None:
            raise ExecutionFailed(
                "bound YAM stop callback must return None after its stop operation"
            )
