"""Single-object observation and result envelope, in the shared contract ``rsh-task/1``.

:class:`ObjectObservation` is one object's state as one provider saw it at one
time. It never carries a commanded or planned pose: a provider that did not see
the object returns ``visibility="not_found"`` (or ``occluded``) with no pose, and a
consumer that needs a pose rejects it.

``to_v1()`` gives the contract's ``Observation``; ``to_backend_object()`` the entry
a backend's ``get_state().objects[id]`` carries so that the ``observe_objects``
primitive passes it through; ``from_v1()`` reads an ``Observation`` back (e.g.
from ``ctx.observe``). Orientations on the wire use the contract's object frame
(for a domino: +z the height axis, :data:`.objects.R_CONTRACT_TO_MODULE`). The fit details that the contract has no field for travel
under the extra key ``perception``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .frames import make_T, matrix_to_quat_xyzw, quat_xyzw_to_matrix
from .objects import from_contract_frame, to_contract_frame

CONTRACT_VERSION = "rsh-task/1"
SCHEMA_VERSION = CONTRACT_VERSION

VISIBILITY = ("visible", "partially_occluded", "occluded", "not_found", "unknown")
"""``visible``: fitted with full expected coverage; ``partially_occluded``: fitted,
part of the camera-facing surface missing; ``occluded`` / ``not_found``: no usable
pose (position and orientation are None)."""

SIM_TRUTH_SOURCES = ("sim_truth", "simulation_truth")

STATUSES = ("succeeded", "failed", "unverified", "rejected")
RECOVERY_HINTS = (
    "reobserve",
    "regrasp",
    "retry_same",
    "back_off",
    "replan",
    "reset_piece",
    "abort",
    "human",
)


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

    def to_v1(self) -> dict[str, Any] | None:
        if self.position_std_m is None and self.rotation_std_rad is None:
            return None
        return {"position_m": self.position_std_m, "rotation_rad": self.rotation_std_rad}

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
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.visibility not in VISIBILITY:
            raise ValueError(f"visibility must be one of {VISIBILITY}")
        if self.provider in SIM_TRUTH_SOURCES and not self.privileged:
            raise ValueError("simulator truth must be labelled privileged")
        if self.visibility in ("occluded", "not_found") and self.T_frame_object is not None:
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

    def check_usable(
        self, now_s: float, max_age_s: float, allow_privileged: bool = False
    ) -> tuple[bool, str, str | None]:
        """Whether a consumer may act on this pose: (ok, why not, contract failure_type)."""
        if not self.has_pose:
            return (
                False,
                f"no pose: {self.visibility} ({self.reason or 'no reason given'})",
                "object_not_observed",
            )
        if self.privileged and not allow_privileged:
            return (
                False,
                "privileged simulator truth used where a perception result is required",
                "perception.privileged_input",
            )
        age = self.age_s(now_s)
        if age > max_age_s:
            return (
                False,
                f"stale: observed {age:.2f} s ago, limit {max_age_s} s",
                "stale_observation",
            )
        if age < -0.1:
            return False, f"timestamp {-age:.2f} s in the future", "stale_observation"
        if self.uncertainty.ambiguous:
            return (
                False,
                "ambiguous fit: another non-symmetric pose explains the data",
                "perception.symmetry_ambiguous",
            )
        return True, "", None

    def usable(
        self, now_s: float, max_age_s: float, allow_privileged: bool = False
    ) -> tuple[bool, str]:
        ok, why, _ = self.check_usable(now_s, max_age_s, allow_privileged)
        return ok, why

    def state_label(self) -> str | None:
        d = self.attributes.get("domino")
        if d:
            return d["state"]
        lat = self.attributes.get("lattice")
        if lat:
            return "seated" if lat["route"] == "on_grid" else "loose"
        return None

    def to_v1(self) -> dict[str, Any]:
        T = self.T_frame_object
        return {
            "object_id": self.object_id,
            "class": self.object_class,
            "frame_id": self.frame_id,
            "position_m": None if T is None else [float(v) for v in T[:3, 3]],
            "quaternion_xyzw": None
            if T is None
            else matrix_to_quat_xyzw(to_contract_frame(T, self.object_class)[:3, :3]),
            "timestamp_s": float(self.timestamp_s),
            "source": self.provider,
            "privileged": bool(self.privileged),
            "confidence": self.confidence,
            "uncertainty": self.uncertainty.to_v1(),
            "visibility": self.visibility,
            "state_label": self.state_label(),
            "geometry": {
                "type": "box",
                "size_m": [float(v) for v in self.size_m],
                "origin": "box centre",
            },
            "reason": self.reason or None,
            "perception": {
                "uncertainty_detail": self.uncertainty.to_dict(),
                "attributes": jsonable(self.attributes),
                "evidence": jsonable(self.evidence),
            },
        }

    to_dict = to_v1

    def to_backend_object(self) -> dict[str, Any]:
        """``get_state().objects[id]`` entry for a backend whose object provider is this module."""
        d = self.to_v1()
        d.pop("object_id")
        return d

    @classmethod
    def from_v1(
        cls, d: dict[str, Any], size_m: tuple[float, float, float] | None = None
    ) -> ObjectObservation:
        """Read a contract ``Observation``. A missing orientation leaves the pose
        None (a centroid is not a 6D pose) and the visibility ``unknown``."""
        geom = d.get("geometry") or {}
        size = size_m or tuple(float(v) for v in geom.get("size_m", (0.0, 0.0, 0.0)))
        pos, q = d.get("position_m"), d.get("quaternion_xyzw")
        T = None if pos is None or q is None else make_T(quat_xyzw_to_matrix(q), pos)
        if T is not None:
            T = from_contract_frame(T, d.get("class") or "")
        vis = d.get("visibility", "unknown")
        reason = d.get("reason") or ""
        if T is None and vis in ("visible", "partially_occluded"):
            vis, reason = "unknown", reason or "position without orientation: not a 6D pose"
        u = d.get("uncertainty") or {}
        detail = (d.get("perception") or {}).get("uncertainty_detail") or {}
        unc = Uncertainty(
            u.get("position_m"),
            u.get("rotation_rad"),
            detail.get("kind", "simulator_truth" if d.get("privileged") else "reported"),
            bool(detail.get("ambiguous", False)),
            int(detail.get("symmetry_order", 1)),
            float(detail.get("yaw_ambiguity_deg", 0.0)),
            dict(detail.get("fit", {})),
        )
        ts = d.get("timestamp_s")
        return cls(
            object_id=d["object_id"],
            object_class=d.get("class") or "unknown",
            size_m=size,
            frame_id=d["frame_id"],
            timestamp_s=float(ts) if ts is not None else 0.0,  # no time: always stale
            provider=d.get("source", ""),
            privileged=bool(d.get("privileged")) or d.get("source") in SIM_TRUTH_SOURCES,
            visibility=vis,
            T_frame_object=T,
            uncertainty=unc,
            reason=reason,
            attributes=dict((d.get("perception") or {}).get("attributes", {})),
            evidence=dict((d.get("perception") or {}).get("evidence", {})),
            confidence=d.get("confidence"),
        )


def result(
    status: str,
    phase: str,
    *,
    failure_type: str | None = None,
    reason: str = "",
    outputs: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    recovery_hints: list[str] | None = None,
    verification: str = "",
) -> dict[str, Any]:
    """Result of a pure function, with the contract's status / failure_type /
    recovery_hints semantics; :mod:`.skills` wraps it in the full SkillResult.

    ``verification`` names the strongest check behind it: ``kinematic_model`` (IK
    on the i2rt chain), ``geometry`` or ``none``. A plan that passed the kinematic
    model is not a simulated or executed success.
    """
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    if status == "succeeded" and failure_type:
        raise ValueError("a succeeded result has no failure_type")
    if status != "succeeded" and not failure_type:
        raise ValueError("a non-succeeded result needs a failure_type")
    bad = set(recovery_hints or ()) - set(RECOVERY_HINTS)
    if bad:
        raise ValueError(f"unknown recovery hints {sorted(bad)}")
    return {
        "contract_version": CONTRACT_VERSION,
        "status": status,
        "phase": phase,
        "failure_type": failure_type,
        "reason": reason,
        "verification": verification,
        "outputs": jsonable(outputs or {}),
        "evidence": jsonable(evidence or {}),
        "recovery_hints": list(recovery_hints or []),
    }


def jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v
