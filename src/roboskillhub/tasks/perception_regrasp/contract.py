"""Single-object observation contract and task-result envelope (pre contract-v1).

Field names follow 00-common-contract.md; Agent 5's contract-v1 decides the final
wire names. Until then ``SCHEMA_VERSION`` marks these as this module's draft.

An :class:`ObjectObservation` is one object's state as one provider saw it at one
time. It never carries a commanded or planned pose: a provider that did not see
the object returns ``visibility="not_seen"`` with no pose, and a consumer that
needs a pose rejects it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .frames import T_to_pose, pose_to_T

SCHEMA_VERSION = "perception_regrasp/draft-1"

VISIBILITY = ("visible", "partially_occluded", "occluded", "not_seen")
"""``visible``: fitted with full expected coverage; ``partially_occluded``: fitted,
part of the camera-facing surface missing; ``occluded``/``not_seen``: no usable
pose (the pose field is None)."""

PROVIDERS = ("sim_truth", "rgbd_pose6d", "held_prior")

STATUSES = ("succeeded", "failed", "unverified", "rejected")


@dataclass
class Uncertainty:
    position_std_m: float | None
    rotation_std_rad: float | None
    kind: str
    """``simulator_truth`` (exact, privileged), ``fit_residual`` (a heuristic from
    the registration residual, not a calibrated covariance), ``prior`` (a
    geometric assumption, e.g. a grasp offset that was never measured)."""
    ambiguous: bool = False
    """A different, non-symmetric pose explains the data almost as well."""
    symmetry_order: int = 1
    yaw_ambiguity_deg: float = 0.0
    fit: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_std_m": self.position_std_m,
            "rotation_std_rad": self.rotation_std_rad,
            "kind": self.kind,
            "ambiguous": self.ambiguous,
            "symmetry_order": self.symmetry_order,
            "yaw_ambiguity_deg": self.yaw_ambiguity_deg,
            "fit": dict(self.fit),
        }


@dataclass
class ObjectObservation:
    object_id: str
    object_class: str
    size_m: tuple[float, float, float]
    frame_id: str
    timestamp_s: float
    provider: str
    privileged: bool
    visibility: str
    T_frame_object: np.ndarray | None
    uncertainty: Uncertainty
    reason: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.visibility not in VISIBILITY:
            raise ValueError(f"visibility must be one of {VISIBILITY}")
        if self.provider == "sim_truth" and not self.privileged:
            raise ValueError("simulator truth must be labelled privileged")
        if self.visibility in ("occluded", "not_seen") and self.T_frame_object is not None:
            raise ValueError("an unseen object carries no pose")
        if self.visibility in ("visible", "partially_occluded") and self.T_frame_object is None:
            raise ValueError("a seen object needs a pose")
        if not math.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")

    @property
    def has_pose(self) -> bool:
        return self.T_frame_object is not None

    def age_s(self, now_s: float) -> float:
        return float(now_s) - float(self.timestamp_s)

    def usable(
        self, now_s: float, max_age_s: float, allow_privileged: bool = False
    ) -> tuple[bool, str]:
        """Whether a consumer may act on this pose, and why not."""
        if not self.has_pose:
            return False, f"no pose: {self.visibility} ({self.reason or 'no reason given'})"
        if self.privileged and not allow_privileged:
            return False, "privileged simulator truth used where a perception result is required"
        age = self.age_s(now_s)
        if age > max_age_s:
            return False, f"stale: observed {age:.2f} s ago, limit {max_age_s} s"
        if age < -0.05:
            return False, f"timestamp {-age:.2f} s in the future"
        if self.uncertainty.ambiguous:
            return False, "ambiguous fit: another non-symmetric pose explains the data"
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "object_id": self.object_id,
            "object_class": self.object_class,
            "size_m": [float(v) for v in self.size_m],
            "pose": None
            if self.T_frame_object is None
            else T_to_pose(self.T_frame_object, self.frame_id),
            "frame_id": self.frame_id,
            "timestamp_s": float(self.timestamp_s),
            "uncertainty": self.uncertainty.to_dict(),
            "visibility": self.visibility,
            "provider": self.provider,
            "privileged": bool(self.privileged),
            "reason": self.reason,
            "attributes": _jsonable(self.attributes),
            "evidence": _jsonable(self.evidence),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ObjectObservation:
        u = d["uncertainty"]
        return cls(
            object_id=d["object_id"],
            object_class=d["object_class"],
            size_m=tuple(float(v) for v in d["size_m"]),
            frame_id=d["frame_id"],
            timestamp_s=float(d["timestamp_s"]),
            provider=d["provider"],
            privileged=bool(d["privileged"]),
            visibility=d["visibility"],
            T_frame_object=None if d.get("pose") is None else pose_to_T(d["pose"], d["frame_id"]),
            uncertainty=Uncertainty(
                u.get("position_std_m"),
                u.get("rotation_std_rad"),
                u["kind"],
                bool(u.get("ambiguous", False)),
                int(u.get("symmetry_order", 1)),
                float(u.get("yaw_ambiguity_deg", 0.0)),
                dict(u.get("fit", {})),
            ),
            reason=d.get("reason", ""),
            attributes=dict(d.get("attributes", {})),
            evidence=dict(d.get("evidence", {})),
        )


def result(
    status: str,
    stage: str,
    *,
    failure_type: str | None = None,
    reason: str = "",
    outputs: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    recovery: list[str] | None = None,
    verification: str = "",
) -> dict[str, Any]:
    """Task-result envelope with the binding status semantics.

    ``verification`` names the strongest check behind the result, e.g.
    ``kinematic_model`` (IK on the i2rt chain), ``geometry`` or ``none``; a plan
    that passed the kinematic model is not a simulated or executed success.
    """
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    if status == "succeeded" and failure_type:
        raise ValueError("a succeeded result has no failure_type")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "stage": stage,
        "failure_type": failure_type,
        "reason": reason,
        "verification": verification,
        "outputs": _jsonable(outputs or {}),
        "evidence": _jsonable(evidence or {}),
        "recovery": list(recovery or []),
    }


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v
