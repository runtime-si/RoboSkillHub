"""Shared task contract ``rsh-task/1``: types, validators and the task-skill gate.

Task modules (``roboskillhub.tasks.<module>``) export ``SKILLS``, a list of
objects satisfying :class:`TaskSkill`. They reach the station only through the
:class:`SkillContext` they are handed, whose ``invoke`` runs primitives under the
caller's task lease. :func:`run_task_skill` applies the capability gate, input
validation and result-envelope rules every task skill is held to.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .contracts import (
    CONTRACT_VERSION,
    EvidenceMissing,
    ExecutionFailed,
    Rejected,
    _Typed,
    compose,
    invert,
    sim_time,
    tcp_for_object_goal,
    validate_pose,
)

__all__ = [
    "CONTRACT",
    "SkillContext",
    "TaskSkill",
    "compose",
    "invert",
    "make_result",
    "run_task_skill",
    "tcp_for_object_goal",
    "validate",
]

CONTRACT: dict = json.loads(
    Path(__file__).with_name("task_contract_v1.json").read_text(encoding="utf-8")
)
assert CONTRACT["contract_version"] == CONTRACT_VERSION
CORE_FAILURE_TYPES = frozenset(CONTRACT["failure_types"]["core"])
MODULE_PREFIXES = tuple(CONTRACT["failure_types"]["module_prefixes"])
BACKEND_CAPABILITIES = frozenset(CONTRACT["backend_capabilities"])
READINESS_LEVELS = tuple(CONTRACT["readiness_levels"])
RECOVERY_HINTS = frozenset(CONTRACT["recovery_hints"])
KINDS = ("tool", "skill")


def known_failure_type(value: object) -> bool:
    return isinstance(value, str) and (
        value in CORE_FAILURE_TYPES
        or (value.startswith(MODULE_PREFIXES) and len(value.split(".", 1)[1]) > 0)
    )


_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: (
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    ),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _check(value: Any, schema: Mapping, path: str, errors: list[str]) -> None:
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        _check(value, CONTRACT["schemas"][name], path, errors)
        if "frame_id" in schema and isinstance(value, dict):
            if value.get("frame_id") != schema["frame_id"]:
                errors.append(f"{path}.frame_id must be {schema['frame_id']!r}")
        return
    kinds = schema.get("type")
    if kinds is not None:
        kinds = [kinds] if isinstance(kinds, str) else kinds
        if not any(_TYPES[k](value) for k in kinds):
            errors.append(f"{path} must be {' or '.join(kinds)}")
            return
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path} must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} must be <= {schema['maximum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 1 << 30):
            errors.append(f"{path} has {len(value)} items, outside the allowed count")
        if "items" in schema:
            for i, item in enumerate(value):
                _check(item, schema["items"], f"{path}[{i}]", errors)
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", ()):
            if key not in value:
                errors.append(f"{path}.{key} is required")
        if schema.get("additionalProperties") is False:
            for key in set(value) - set(props):
                errors.append(f"{path}.{key} is not allowed")
        for key, sub in props.items():
            if key in value and isinstance(sub, Mapping):
                _check(value[key], sub, f"{path}.{key}", errors)


def schema_errors(value: Any, schema: Mapping, path: str = "$") -> list[str]:
    errors: list[str] = []
    _check(value, schema, path, errors)
    return errors


def validate(kind: str, value: Any) -> Any:
    """Validate ``value`` against a named contract schema (Observation, HeldObject, ...)."""
    if kind not in CONTRACT["schemas"]:
        raise Rejected(f"unknown contract schema {kind!r}")
    errors = schema_errors(value, CONTRACT["schemas"][kind], kind)
    if kind == "Pose" or kind in ("Observation", "HeldObject"):
        q = value.get("quaternion_xyzw") if isinstance(value, dict) else None
        if kind == "HeldObject" and isinstance(value, dict):
            q = (value.get("T_TCP_object") or {}).get("quaternion_xyzw")
        if isinstance(q, list) and len(q) == 4 and all(_TYPES["number"](x) for x in q):
            if abs(math.sqrt(sum(x * x for x in q)) - 1) > 0.01:
                errors.append(f"{kind} quaternion_xyzw must have unit norm")
    if kind == "SkillResult" and isinstance(value, dict):
        status, failure = value.get("status"), value.get("failure_type")
        if status == "succeeded" and failure is not None:
            errors.append("a succeeded result carries failure_type null")
        if status in ("failed", "unverified", "rejected") and not known_failure_type(failure):
            errors.append(f"failure_type {failure!r} is not in the contract vocabulary")
        unknown = set(value.get("recovery_hints", ())) - RECOVERY_HINTS
        if unknown:
            errors.append(f"unknown recovery_hints {sorted(unknown)}")
    if errors:
        raise Rejected("; ".join(errors[:8]), failure_type="invalid_request")
    return value


class TaskSkill(Protocol):
    skill_id: str
    version: str
    kind: str
    requires: frozenset[str]
    input_schema: dict
    readiness: str

    def run(self, request: dict, ctx: SkillContext) -> dict: ...


@dataclass
class SkillContext:
    """What a task skill may use. Built by the runtime, never by the skill."""

    task_id: str
    request_id: str
    capabilities: frozenset[str] = frozenset()
    privileged_allowed: bool = False
    primitive: Callable[[str, dict, str], dict] | None = None
    is_cancelled: Callable[[], bool] = lambda: False
    clock: Callable[[], float] = time.time
    children: list[str] = field(default_factory=list)
    primitive_results: list[dict] = field(default_factory=list)
    limits: Mapping[str, Any] = field(default_factory=dict)
    """The station's trusted limits (read-only copy); requests cannot change them."""
    backend_description: Mapping[str, Any] = field(default_factory=dict)
    evidence_level: str = "unit_or_dryrun"
    read_state: Callable[[], dict] | None = None
    sleep: Callable[[float], None] = time.sleep
    contact_description: Mapping[str, Any] | None = None
    """Contact signal, controller and contact limits of a ``guarded_press`` station, else None."""

    def invoke(self, capability: str, request: dict) -> dict:
        if self.primitive is None:
            raise Rejected(
                "this context has no station; tools run offline only",
                failure_type="capability_missing",
            )
        if self.is_cancelled():
            raise Rejected("task cancelled", failure_type="cancelled")
        rid = f"{self.request_id}/p{len(self.children)}"
        result = self.primitive(capability, request, rid)
        self.children.append(result.get("invocation_id", rid))
        self.primitive_results.append(result)
        return result

    def observe(self, object_ids: Iterable[str], *, require_orientation: bool = False) -> dict:
        """Call ``observe_objects``; returns its result (observations in evidence)."""
        request: dict = {"object_ids": list(object_ids)}
        if require_orientation:
            request["require_orientation"] = True
        return self.invoke("observe_objects", request)

    def cancelled(self) -> bool:
        return self.is_cancelled()

    def state(self) -> dict:
        """No-motion read of measured TCP pose and gripper feedback."""
        if self.read_state is None:
            raise Rejected("this context has no station state", failure_type="capability_missing")
        return self.read_state()

    def now(self) -> float:
        return self.clock()

    def result(self, skill: TaskSkill, status: str, **fields: Any) -> dict:
        return make_result(
            skill, status, task_id=self.task_id, request_id=self.request_id, ctx=self, **fields
        )


def make_result(
    skill: TaskSkill,
    status: str,
    *,
    phase: str | None = None,
    failure_type: str | None = None,
    evidence: dict | None = None,
    observations: Iterable[dict] = (),
    recovery_hints: Iterable[str] = (),
    evidence_refs: Iterable[dict] = (),
    outputs: dict | None = None,
    error: dict | None = None,
    task_id: str | None = None,
    request_id: str | None = None,
    ctx: SkillContext | None = None,
) -> dict:
    observations = list(observations)
    result = {
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "invocation_id": str(uuid.uuid4()),
        "task_id": task_id,
        "request_id": request_id,
        "capability": skill.skill_id,
        "skill": skill.skill_id,
        "version": skill.version,
        "kind": skill.kind,
        "status": status,
        "phase": phase,
        "failure_type": None if status == "succeeded" else failure_type,
        "recovery_hints": list(recovery_hints),
        "evidence": evidence or {},
        "observations": observations,
        "evidence_refs": list(evidence_refs),
        "children": list(ctx.children) if ctx else [],
        "privileged_inputs": any(o.get("privileged") for o in observations),
        "outputs": outputs or {},
        "trace": [],
    }
    if error is not None:
        result["error"] = error
    return result


def _gate_failure(skill: TaskSkill, ctx: SkillContext, status: str, exc: Exception) -> dict:
    failure = exc.failure_type if isinstance(exc, _Typed) else "backend_error"
    return make_result(
        skill,
        status,
        phase="validate" if status == "rejected" else None,
        failure_type=failure,
        error={"code": type(exc).__name__, "message": str(exc)},
        task_id=ctx.task_id,
        request_id=ctx.request_id,
        ctx=ctx,
    )


def run_task_skill(skill: TaskSkill, request: dict, ctx: SkillContext) -> dict:
    """Run one task skill under the contract's gates and envelope rules."""
    start = time.monotonic()
    try:
        if skill.kind not in KINDS or skill.readiness not in READINESS_LEVELS:
            raise Rejected(f"{skill.skill_id}: invalid kind or readiness declaration")
        unknown = set(skill.requires) - BACKEND_CAPABILITIES
        if unknown:
            raise Rejected(f"{skill.skill_id} requires undeclared capabilities {sorted(unknown)}")
        missing = set(skill.requires) - set(ctx.capabilities)
        if missing:
            raise Rejected(
                f"{skill.skill_id} needs backend capabilities {sorted(missing)}",
                failure_type="capability_missing",
            )
        errors = schema_errors(request, skill.input_schema, "request")
        if errors:
            raise Rejected("; ".join(errors[:8]), failure_type="invalid_request")
    except Rejected as exc:
        result = _gate_failure(skill, ctx, "rejected", exc)
        result["elapsed_s"] = time.monotonic() - start
        return result
    try:
        result = skill.run(request, ctx)
    except (Rejected, EvidenceMissing, ExecutionFailed) as exc:
        acted = bool(ctx.children)
        status = (
            "unverified"
            if isinstance(exc, EvidenceMissing) and acted
            else "failed"
            if acted
            else "rejected"
        )
        result = _gate_failure(skill, ctx, status, exc)
    except Exception as exc:
        result = _gate_failure(skill, ctx, "failed" if ctx.children else "rejected", exc)
    try:
        validate("SkillResult", result)
    except Rejected as exc:
        result = _gate_failure(
            skill,
            ctx,
            "failed" if ctx.children else "rejected",
            Rejected(f"malformed result: {exc}", failure_type="backend_error"),
        )
    result["children"] = list(dict.fromkeys([*result.get("children", []), *ctx.children]))
    if any(o.get("privileged") for o in result.get("observations", [])):
        result["privileged_inputs"] = True
    if result["status"] == "succeeded" and skill.kind == "skill":
        post = result["evidence"].get("postcondition")
        if not (isinstance(post, dict) and post.get("passed") is True and result["observations"]):
            result["status"] = "unverified"
            result["failure_type"] = "evidence_missing"
            result["recovery_hints"] = ["reobserve"]
            result["downgraded"] = (
                "succeeded without evidence.postcondition.passed and independent observations"
            )
    result.setdefault("elapsed_s", time.monotonic() - start)
    return result


def contact_description(description: Mapping[str, Any], limits) -> dict | None:
    """What a contact skill needs to judge a ``guarded_press`` station, from ``describe()``.

    None unless the backend declares compliant contact and describes its signal and controller.
    """
    capabilities = set(description.get("capabilities", []))
    signal, controller = description.get("contact_signal"), description.get("controller")
    if not (
        {"contact_force_feedback", "compliant_motion"} <= capabilities
        and isinstance(signal, dict)
        and isinstance(controller, dict)
    ):
        return None
    declared = description.get("contact_limits")
    declared = dict(declared) if isinstance(declared, dict) else {}
    station_max = float(getattr(limits, "max_contact_force_N", 0.0) or 0.0)
    caps = [
        x
        for x in (declared.get("max_force_N"), station_max)
        if isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0
    ]
    declared["max_force_N"] = min(caps) if caps else 0.0  # the tighter of backend and station
    return {
        "backend_id": description.get("backend", "runtime"),
        "evidence_level": description.get("evidence_level", "unit_or_dryrun"),
        "capabilities": sorted(capabilities),
        "contact_signal": dict(signal),
        "controller": {k: controller.get(k) for k in ("kind", "loop_hz", "process")},
        "limits": declared,
        "privileged": False,
    }


def primitive_skill_context(
    runtime: Any,
    lease: Any,
    *,
    request_id: str,
    privileged_allowed: bool = False,
) -> SkillContext:
    """A context whose primitives run on ``runtime`` under ``lease``."""
    description = runtime.backend.describe()
    limits = runtime.limits

    def primitive(capability: str, request: dict, rid: str) -> dict:
        return runtime.invoke(capability, request, lease=lease, request_id=rid)

    def read_state() -> dict:
        timeout = max(0.01, min(limits.max_state_age_s, lease.remaining()))
        state = runtime.backend.get_state(timeout_s=timeout)
        if not isinstance(state, dict):
            raise EvidenceMissing("backend state is not an object")
        try:
            pose = validate_pose(state.get("pose"), limits.frame_id)
        except Rejected as exc:
            raise EvidenceMissing(f"invalid measured TCP pose: {exc}") from exc
        gripper = state.get("gripper") if isinstance(state.get("gripper"), dict) else {}
        out = {
            "tcp_pose": pose,
            "timestamp_s": state.get("timestamp_s"),
            "gripper": {k: gripper.get(k) for k in ("holding", "width_m", "timestamp_s", "source")},
        }
        if sim_time(state.get("sim_time_s")) is not None:
            out["sim_time_s"] = sim_time(state["sim_time_s"])
        if isinstance(state.get("joints"), dict):
            joints = state["joints"]
            out["joints"] = {
                k: joints.get(k) for k in ("names", "positions_rad", "timestamp_s", "source")
            }
        return out

    return SkillContext(
        task_id=lease.task_id,
        request_id=request_id,
        capabilities=frozenset(description.get("capabilities", [])),
        privileged_allowed=privileged_allowed,
        primitive=primitive,
        is_cancelled=lambda: lease.cancel_reason is not None,
        limits=dict(vars(limits)),
        backend_description=dict(description),
        evidence_level=description.get("evidence_level", "unit_or_dryrun"),
        read_state=read_state,
        contact_description=contact_description(description, limits),
    )


def offline_context(task_id: str = "offline", request_id: str = "offline/0") -> SkillContext:
    """A context for tools: no station, no primitives."""
    return SkillContext(task_id=task_id, request_id=request_id)
