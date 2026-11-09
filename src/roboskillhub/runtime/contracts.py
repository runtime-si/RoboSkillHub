"""JSON contracts shared by agents, station adapters and skill implementations.

All spatial values are metres in the configured station frame. Quaternions use
xyzw. No conversion, implicit frame transform or default workcell is guessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

CONTRACT_VERSION = "rsh-task/1"
STATUSES = ("succeeded", "failed", "unverified", "rejected")


class _Typed(Exception):
    """Carries the contract ``failure_type`` reported alongside the terminal status."""

    default_failure = "backend_error"

    def __init__(self, message: str = "", *, failure_type: str | None = None):
        super().__init__(message)
        self.failure_type = failure_type or self.default_failure


class Rejected(_Typed, ValueError):
    """Invalid request or missing prerequisite; no execution should start."""

    default_failure = "invalid_request"


class ExecutionFailed(_Typed, RuntimeError):
    """An attempted physical action did not complete as required."""

    default_failure = "postcondition_failed"


class EvidenceMissing(_Typed, RuntimeError):
    """Execution may have occurred, but its outcome cannot be established."""

    default_failure = "evidence_missing"


def number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Rejected(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise Rejected(f"{name} must be within [{low}, {high}]")
    return value


def vector(value: Any, name: str, size: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise Rejected(f"{name} must contain {size} numbers")
    return [number(v, name, -1e6, 1e6) for v in value]


def unit_vector(value: Any, name: str) -> list[float]:
    v = vector(value, name, 3)
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-9:
        raise Rejected(f"{name} must be nonzero")
    return [x / n for x in v]


def validate_pose(value: Any, frame_id: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"frame_id", "position_m", "quaternion_xyzw"}:
        raise Rejected("pose requires exactly frame_id, position_m, quaternion_xyzw")
    if value["frame_id"] != frame_id:
        raise Rejected(f"pose frame must be {frame_id!r}; explicit transforms are required")
    p = vector(value["position_m"], "position_m", 3)
    q = vector(value["quaternion_xyzw"], "quaternion_xyzw", 4)
    norm = math.sqrt(sum(x * x for x in q))
    if abs(norm - 1) > 0.01:
        raise Rejected("quaternion_xyzw must have unit norm (tolerance 0.01)")
    return {"frame_id": frame_id, "position_m": p, "quaternion_xyzw": [x / norm for x in q]}


def pose_error(a: dict, b: dict) -> tuple[float, float]:
    distance = math.dist(a["position_m"], b["position_m"])
    dot = abs(sum(x * y for x, y in zip(a["quaternion_xyzw"], b["quaternion_xyzw"], strict=True)))
    return distance, 2 * math.acos(min(1.0, dot))


def translated(pose: dict, delta: list[float]) -> dict:
    return {**pose, "position_m": [p + d for p, d in zip(pose["position_m"], delta, strict=True)]}


def quat_multiply(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def quat_rotate(q: list[float], v: list[float]) -> list[float]:
    x, y, z, w = quat_multiply(quat_multiply(q, [*v, 0.0]), [-q[0], -q[1], -q[2], q[3]])
    return [x, y, z]


def compose(T_A_B: dict, T_B_C: dict, *, frame_id: str | None = None) -> dict:
    """``T_A_C = T_A_B · T_B_C``; the result is expressed in ``T_A_B``'s frame."""
    q = quat_multiply(T_A_B["quaternion_xyzw"], T_B_C["quaternion_xyzw"])
    n = math.sqrt(sum(x * x for x in q))
    p = quat_rotate(T_A_B["quaternion_xyzw"], T_B_C["position_m"])
    return {
        "frame_id": frame_id or T_A_B["frame_id"],
        "position_m": [a + b for a, b in zip(T_A_B["position_m"], p, strict=True)],
        "quaternion_xyzw": [x / n for x in q],
    }


def invert(T_A_B: dict, *, frame_id: str) -> dict:
    """``T_B_A``; ``frame_id`` names frame B, which the caller must state."""
    x, y, z, w = T_A_B["quaternion_xyzw"]
    q = [-x, -y, -z, w]
    p = quat_rotate(q, T_A_B["position_m"])
    return {"frame_id": frame_id, "position_m": [-v for v in p], "quaternion_xyzw": q}


def tcp_for_object_goal(T_W_object_goal: dict, T_TCP_object: dict) -> dict:
    """TCP pose that places a held object at its goal: ``T_W_object · T_object_TCP``."""
    if T_TCP_object.get("frame_id") != "tcp":
        raise Rejected("T_TCP_object must be expressed in frame 'tcp'")
    return compose(T_W_object_goal, invert(T_TCP_object, frame_id="object"))


@dataclass(frozen=True)
class Limits:
    frame_id: str
    workspace_min_m: tuple[float, float, float]
    workspace_max_m: tuple[float, float, float]
    max_speed_m_s: float
    max_contact_distance_m: float
    gripper_min_width_m: float
    gripper_max_width_m: float
    max_state_age_s: float = 1.0
    max_timeout_s: float = 30.0
    position_tolerance_m: float = 0.01
    rotation_tolerance_rad: float = 0.1
    gripper_width_tolerance_m: float = 0.004
    held_object_tolerance_m: float = 0.02
    max_task_s: float = 600.0

    @classmethod
    def from_dict(cls, config: dict) -> Limits:
        if not isinstance(config, dict):
            raise Rejected("station limits must be an object")
        required = {
            "frame_id",
            "workspace_min_m",
            "workspace_max_m",
            "max_speed_m_s",
            "max_contact_distance_m",
            "gripper_min_width_m",
            "gripper_max_width_m",
        }
        if not required <= set(config) or set(config) - set(cls.__dataclass_fields__):
            raise Rejected(f"station limits require {sorted(required)} and known optional limits")
        if not isinstance(config["frame_id"], str) or not config["frame_id"]:
            raise Rejected("frame_id must be a nonempty string")
        values = dict(config)
        for key in ("workspace_min_m", "workspace_max_m"):
            values[key] = tuple(vector(values[key], key, 3))
        if any(
            a >= b
            for a, b in zip(values["workspace_min_m"], values["workspace_max_m"], strict=True)
        ):
            raise Rejected("workspace_min_m must be below workspace_max_m on every axis")
        ranges = {
            "max_speed_m_s": (1e-5, 1),
            "max_contact_distance_m": (1e-4, 1),
            "gripper_min_width_m": (0, 1),
            "gripper_max_width_m": (1e-5, 1),
            "max_state_age_s": (0.001, 5),
            "max_timeout_s": (0.1, 120),
            "position_tolerance_m": (1e-5, 0.05),
            "rotation_tolerance_rad": (1e-4, 0.3),
            "gripper_width_tolerance_m": (1e-4, 0.02),
            "held_object_tolerance_m": (1e-3, 0.05),
            "max_task_s": (1, 7200),
        }
        for key, bounds in ranges.items():
            if key in values:
                values[key] = number(values[key], key, *bounds)
        if values["gripper_min_width_m"] >= values["gripper_max_width_m"]:
            raise Rejected("gripper_min_width_m must be smaller than gripper_max_width_m")
        return cls(**values)

    def pose(self, value: Any) -> dict:
        p = validate_pose(value, self.frame_id)
        if not all(
            lo <= x <= hi
            for lo, x, hi in zip(
                self.workspace_min_m, p["position_m"], self.workspace_max_m, strict=True
            )
        ):
            raise Rejected(
                "requested pose is outside the configured workcell workspace",
                failure_type="unreachable",
            )
        return p


class StationBackend(Protocol):
    """Blocking methods MUST honor timeout_s; stop must stop active motion.

    State: {pose, timestamp_s, objects: {id: {position_m, timestamp_s, source}},
    gripper: {holding: bool | null}}. Timestamps use Unix seconds. Object state
    must be observed (or explicitly labelled simulation truth), never fabricated
    from commanded robot motion. Move/gripper returns {ok: bool, ...}.
    """

    def describe(self) -> dict: ...
    def get_state(self, *, timeout_s: float) -> dict: ...
    def move_pose(
        self, pose: dict, *, speed_m_s: float, timeout_s: float, linear: bool
    ) -> dict: ...
    def set_gripper(self, width_m: float, *, timeout_s: float) -> dict: ...
    def stop(self) -> None: ...


POSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["frame_id", "position_m", "quaternion_xyzw"],
    "properties": {
        "frame_id": {"type": "string"},
        "position_m": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
        "quaternion_xyzw": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "number"},
        },
    },
}
_N = {"type": "number"}
_S = {"type": "string"}
_V = {"type": "array", "minItems": 3, "maxItems": 3, "items": _N}
_COMMON = {"speed_m_s": _N, "timeout_s": _N}
SCHEMAS = {
    "move_pose": {
        "required": ["target_pose"],
        "properties": {**_COMMON, "target_pose": POSE_SCHEMA},
    },
    "grasp": {
        "required": [
            "pregrasp_pose",
            "grasp_pose",
            "lift_pose",
            "target_id",
            "open_width_m",
            "close_width_m",
        ],
        "properties": {
            **_COMMON,
            **{k: POSE_SCHEMA for k in ("pregrasp_pose", "grasp_pose", "lift_pose")},
            "target_id": _S,
            "open_width_m": _N,
            "close_width_m": _N,
        },
    },
    "push": {
        "required": ["approach_pose", "contact_pose", "target_id", "direction_world", "distance_m"],
        "properties": {
            **_COMMON,
            "approach_pose": POSE_SCHEMA,
            "contact_pose": POSE_SCHEMA,
            "target_id": _S,
            "direction_world": _V,
            "distance_m": _N,
            "step_m": _N,
        },
    },
    "pull_drawer": {
        "required": [
            "pregrasp_pose",
            "grasp_pose",
            "target_id",
            "direction_world",
            "distance_m",
            "open_width_m",
            "close_width_m",
        ],
        "properties": {
            **_COMMON,
            "pregrasp_pose": POSE_SCHEMA,
            "grasp_pose": POSE_SCHEMA,
            "target_id": _S,
            "direction_world": _V,
            "distance_m": _N,
            "step_m": _N,
            "open_width_m": _N,
            "close_width_m": _N,
            "release_on_success": {"type": "boolean"},
        },
    },
    "set_gripper": {
        "required": ["width_m", "purpose"],
        "properties": {
            **_COMMON,
            "width_m": _N,
            "purpose": {"enum": ["open", "close", "release"]},
            "target_id": _S,
        },
    },
    "move_waypoints": {
        "required": ["waypoints", "linear"],
        "properties": {
            **_COMMON,
            "waypoints": {"type": "array", "minItems": 1, "maxItems": 32, "items": POSE_SCHEMA},
            "linear": {"type": "boolean"},
            "held": {
                "type": "object",
                "additionalProperties": False,
                "required": ["object_id", "T_TCP_object"],
                "properties": {"object_id": _S, "T_TCP_object": POSE_SCHEMA},
            },
        },
    },
    "observe_objects": {
        "required": ["object_ids"],
        "properties": {
            "timeout_s": _N,
            "object_ids": {"type": "array", "minItems": 1, "maxItems": 32, "items": _S},
            "require_orientation": {"type": "boolean"},
        },
    },
    "track_points": {
        "required": ["frames_path", "query_points"],
        "properties": {
            "frames_path": _S,
            "query_points": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": _N},
            },
            "backend": {"enum": ["opencv-lk", "tapir"]},
        },
    },
}
for _schema in SCHEMAS.values():
    _schema.update(type="object", additionalProperties=False)

CAPABILITY_BUNDLES = {
    "move_pose": "moving-to-pose",
    "grasp": "grasping-with-feedback",
    "track_points": "tracking-points",
    "push": "pushing-on-plane",
    "pull_drawer": "pulling-drawer",
    "set_gripper": "setting-gripper-with-feedback",
    "move_waypoints": "moving-through-waypoints",
    "observe_objects": "observing-object-poses",
}


def validate_request(capability: str, request: Any) -> dict:
    if not isinstance(capability, str) or capability not in SCHEMAS:
        raise Rejected(f"unknown capability {capability!r}")
    if not isinstance(request, dict):
        raise Rejected("request must be a JSON object")
    schema = SCHEMAS[capability]
    missing = set(schema["required"]) - set(request)
    extra = set(request) - set(schema["properties"])
    if missing or extra:
        raise Rejected(f"request fields: missing={sorted(missing)}, unknown={sorted(extra)}")
    for key in ("object_ids", "waypoints"):
        if key in request and (
            not isinstance(request[key], list)
            or not 1 <= len(request[key]) <= schema["properties"][key]["maxItems"]
        ):
            raise Rejected(f"{key} must be a list of 1 to {schema['properties'][key]['maxItems']}")
    if "object_ids" in request and (
        not all(isinstance(x, str) and x.strip() for x in request["object_ids"])
        or len(set(request["object_ids"])) != len(request["object_ids"])
    ):
        raise Rejected("object_ids must be distinct nonempty strings")
    if "held" in request:
        held = request["held"]
        if not isinstance(held, dict) or set(held) != {"object_id", "T_TCP_object"}:
            raise Rejected("held requires exactly object_id and T_TCP_object")
        if not isinstance(held["object_id"], str) or not held["object_id"].strip():
            raise Rejected("held.object_id must be a nonempty string")
        validate_pose(held["T_TCP_object"], "tcp")
    for key, value in request.items():
        spec = schema["properties"][key]
        if spec.get("type") == "string" and (not isinstance(value, str) or not value.strip()):
            raise Rejected(f"{key} must be a nonempty string")
        if spec.get("type") == "number":
            number(value, key, -1e6, 1e6)
        if spec.get("type") == "boolean" and not isinstance(value, bool):
            raise Rejected(f"{key} must be boolean")
        if "enum" in spec and value not in spec["enum"]:
            raise Rejected(f"{key} must be one of {spec['enum']}")
    return request
