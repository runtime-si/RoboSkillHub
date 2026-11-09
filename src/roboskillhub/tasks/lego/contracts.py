"""Request/observation/result semantics shared by the three LEGO skills.

Status values keep the runtime's meaning (kit 00-common-contract):
``rejected`` nothing started, preconditions not met; ``failed`` evidence the attempt did
not achieve its requirement; ``unverified`` something may have happened but the outcome
cannot be established; ``succeeded`` the skill's own complete success criterion holds.

These are module-local types awaiting Agent 05's shared contract v1; the mapping is in
integration-request.md. Nothing here talks to a robot.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from .geometry import GeometryError, validate_pose

STATUSES = ("succeeded", "failed", "unverified", "rejected")

EVIDENCE_LEVELS = (
    "source_inspected",
    "unit_or_dryrun",
    "kinematic_sim",
    "dynamic_sim_privileged",
    "dynamic_sim_observed",
    "real_observed",
)

OBSERVATION_KINDS = ("measured", "sim_truth", "commanded", "planned")
"""``commanded`` / ``planned`` values are never accepted as evidence of a physical state."""


class Rejected(ValueError):
    """Precondition or input not met; no motion was issued."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


def number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Rejected("invalid_input", f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise Rejected("invalid_input", f"{name} must be within [{low}, {high}], got {value}")
    return value


def pose(value: Any, frame_id: str | None, name: str) -> dict:
    try:
        return validate_pose(value, frame_id, name)
    except GeometryError as exc:
        raise Rejected("invalid_geometry", str(exc)) from exc


@dataclass(frozen=True)
class Observation:
    """One observed value with its provenance. ``value`` is skill-specific (a pose dict,
    a height list, ...). ``uncertainty_m`` is a 1-sigma positional uncertainty;
    ``uncertainty_deg`` an angular one."""

    obs_id: str
    source: str
    kind: str
    frame_id: str
    timestamp_s: float
    value: Any
    uncertainty_m: float
    uncertainty_deg: float = 1.0
    occluded: bool = False
    calibration_id: str = ""
    privileged: bool = False

    @classmethod
    def from_dict(cls, d: Any, name: str = "observation") -> Observation:
        if not isinstance(d, dict):
            raise Rejected("invalid_observation", f"{name} must be an object")
        need = {"obs_id", "source", "kind", "frame_id", "timestamp_s", "value", "uncertainty_m"}
        missing = need - set(d)
        if missing:
            raise Rejected("invalid_observation", f"{name} lacks {sorted(missing)}")
        if d["kind"] not in OBSERVATION_KINDS:
            raise Rejected("invalid_observation", f"{name}.kind must be one of {OBSERVATION_KINDS}")
        for k in ("obs_id", "source", "frame_id"):
            if not isinstance(d[k], str) or not d[k]:
                raise Rejected("invalid_observation", f"{name}.{k} must be a nonempty string")
        privileged = bool(d.get("privileged", d["kind"] == "sim_truth"))
        if d["kind"] == "sim_truth" and not privileged:
            raise Rejected(
                "unlabelled_privileged",
                f"{name} is simulator truth but privileged=false; label it",
            )
        return cls(
            obs_id=d["obs_id"],
            source=d["source"],
            kind=d["kind"],
            frame_id=d["frame_id"],
            timestamp_s=number(d["timestamp_s"], f"{name}.timestamp_s", 0, 1e12),
            value=d["value"],
            uncertainty_m=number(d["uncertainty_m"], f"{name}.uncertainty_m", 0, 1),
            uncertainty_deg=number(d.get("uncertainty_deg", 1.0), f"{name}.uncertainty_deg", 0, 90),
            occluded=bool(d.get("occluded", False)),
            calibration_id=str(d.get("calibration_id", "")),
            privileged=privileged,
        )

    def ref(self) -> dict[str, Any]:
        return {
            "obs_id": self.obs_id,
            "source": self.source,
            "kind": self.kind,
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "uncertainty_m": self.uncertainty_m,
            "privileged": self.privileged,
        }


def usable(
    obs: Observation,
    *,
    now_s: float,
    max_age_s: float,
    frame_id: str,
    not_after_s: float | None = None,
) -> str:
    """Empty string if ``obs`` can serve as evidence now, else a reason code."""
    if obs.kind in ("commanded", "planned"):
        return "commanded_not_observed"
    if obs.frame_id != frame_id:
        return "frame_mismatch"
    if obs.occluded:
        return "occluded"
    if obs.timestamp_s > now_s + 0.05:
        return "timestamp_in_future"
    if now_s - obs.timestamp_s > max_age_s:
        return "stale_observation"
    if not_after_s is not None and obs.timestamp_s < not_after_s:
        return "predates_action"
    return ""


@dataclass(frozen=True)
class HeldState:
    """What the hand holds: object identity, gripper identity, grasp, ``T_TCP_object``
    (object coordinates -> TCP coordinates) and the independent evidence the grasp holds."""

    object_id: str
    gripper_id: str
    grasp_id: str
    t_tcp_object: dict
    holding_evidence: Observation

    @classmethod
    def from_dict(cls, d: Any) -> HeldState:
        if not isinstance(d, dict):
            raise Rejected("invalid_input", "held must be an object")
        for k in ("object_id", "gripper_id", "grasp_id"):
            if not isinstance(d.get(k), str) or not d[k]:
                raise Rejected("invalid_input", f"held.{k} must be a nonempty string")
        if "T_TCP_object" not in d:
            if "T_object_TCP" in d:
                raise Rejected(
                    "invalid_geometry",
                    "held gives T_object_TCP; this skill takes T_TCP_object (object -> TCP), "
                    "invert it explicitly",
                )
            raise Rejected("invalid_input", "held.T_TCP_object is required")
        t = pose(d["T_TCP_object"], "tcp", "held.T_TCP_object")
        if "holding_evidence" not in d:
            raise Rejected("grasp_unconfirmed", "no independent evidence the brick is held")
        ev = Observation.from_dict(d["holding_evidence"], "held.holding_evidence")
        if ev.value is not True:
            raise Rejected("grasp_unconfirmed", "holding evidence does not show the brick held")
        return cls(d["object_id"], d["gripper_id"], d["grasp_id"], t, ev)


@dataclass
class SkillResult:
    """Uniform result. ``to_dict`` is the wire form."""

    skill_id: str
    skill_version: str
    request_id: str
    connection_id: str
    status: str = "rejected"
    reason_code: str = ""
    failure_type: str = ""
    phase: str = "precheck"
    message: str = ""
    backend: dict[str, Any] = field(default_factory=dict)
    motion_issued: bool = False
    preconditions: list[dict[str, Any]] = field(default_factory=list)
    postconditions: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    evidence_level: str = "unit_or_dryrun"
    privileged_observation: bool = False
    retries: dict[str, int] = field(default_factory=dict)
    stop: dict[str, Any] = field(default_factory=dict)
    next_allowed: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    invocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def check(self, kind: str, name: str, ok: bool | None, detail: str = "") -> bool | None:
        """Record one pre/postcondition (``ok=None`` means not determinable)."""
        entry = {"name": name, "ok": ok, "detail": detail}
        (self.preconditions if kind == "pre" else self.postconditions).append(entry)
        return ok

    def use(self, obs: Observation) -> None:
        self.evidence_refs.append(obs.ref())
        if obs.privileged:
            self.privileged_observation = True

    def finish(
        self,
        status: str,
        reason_code: str,
        message: str = "",
        *,
        failure_type: str = "",
        next_allowed: list[str] | None = None,
    ) -> SkillResult:
        assert status in STATUSES, status
        if status == "rejected" and self.motion_issued:
            raise AssertionError("rejected is only for results where no motion was issued")
        self.status, self.reason_code, self.message = status, reason_code, message
        self.failure_type = failure_type or ("" if status == "succeeded" else reason_code)
        if next_allowed is not None:
            self.next_allowed = next_allowed
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "invocation_id": self.invocation_id,
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "request_id": self.request_id,
            "connection_id": self.connection_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "failure_type": self.failure_type,
            "phase": self.phase,
            "message": self.message,
            "backend": self.backend,
            "motion_issued": self.motion_issued,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "evidence": self.evidence,
            "evidence_refs": self.evidence_refs,
            "evidence_level": self.evidence_level,
            "privileged_observation": self.privileged_observation,
            "retries": self.retries,
            "stop": self.stop,
            "next_allowed": self.next_allowed,
            "trace": self.trace,
        }


def ids(request: Any, skill_id: str, version: str) -> SkillResult:
    """Start a result; reject requests without request/connection identity."""
    rid = request.get("request_id") if isinstance(request, dict) else None
    cid = request.get("connection_id") if isinstance(request, dict) else None
    res = SkillResult(
        skill_id, version, rid if isinstance(rid, str) else "", cid if isinstance(cid, str) else ""
    )
    if not isinstance(request, dict):
        raise _Early(res, Rejected("invalid_input", "request must be a JSON object"))
    if not res.request_id or not res.connection_id:
        raise _Early(res, Rejected("invalid_input", "request_id and connection_id are required"))
    if request.get("skill_version", version) != version:
        raise _Early(
            res,
            Rejected(
                "version_mismatch",
                f"request is for {request.get('skill_version')}, this is {version}",
            ),
        )
    return res


class _Early(Exception):
    def __init__(self, result: SkillResult, exc: Rejected):
        super().__init__(str(exc))
        self.result, self.exc = result, exc


def level_for(observations: list[Observation], backend_level: str) -> str:
    """Evidence level: the backend's, downgraded to privileged if any truth was used."""
    if backend_level not in EVIDENCE_LEVELS:
        return "unit_or_dryrun"
    if backend_level == "dynamic_sim_observed" and any(o.privileged for o in observations):
        return "dynamic_sim_privileged"
    if backend_level == "real_observed" and any(o.privileged for o in observations):
        raise AssertionError("simulator truth cannot appear in a real_observed run")
    return backend_level
