"""Held-object pose ``T_TCP_object``: geometric prior vs estimate, and invalidation.

* ``geometric_prior``: the grasp candidate's ``T_object_tcp`` inverted. It is what
  the hand was *commanded* to do, not a measurement; it stays labelled so.
* ``estimated``: ``inv(T_frame_tcp) @ T_frame_object`` from an observation taken
  while holding, or a pose6d fit of the held part's depth points seeded with the
  prior (``tools/pose6d`` with ``init``, no support plane - pose6d's own advice
  for a held part).

Any slip beyond tolerance, occlusion of the held part, an identity change, a jaw
width that no longer matches the part, or an old estimate invalidates the state;
consumers must re-observe before using it (``recovery`` says so).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import _hub
from .contract import ObjectObservation, jsonable
from .frames import T_to_pose, inv_T, pose_to_T, rotation_angle
from .grasp import GraspCandidate
from .objects import ObjectSpec, from_contract_frame, symmetry_rotations, to_contract_frame

#: Slip tolerances. pose6d from the wrist D405 is within 0.5 deg / 0.3 mm (p90) and
#: from overhead with support 2.4-5 deg / 1 mm (its SKILL.md table); 3 mm / 5 deg
#: sits above the overhead noise and below what moves a domino off its target slot.
SLIP_TRANSLATION_M = 0.003
SLIP_ROTATION_DEG = 5.0
#: Jaw width off the part's width by more than this: the part is not where the prior says.
WIDTH_TOL_M = 0.0015
#: Observation and TCP reading further apart in time than this cannot be combined [s].
SYNC_TOL_S = 0.05
#: An estimate older than this is no longer the held state [s].
MAX_AGE_S = 2.0


#: Our invalidation kinds -> the v1 ``invalidated_reason`` enum (integration
#: request: add ``grasp_lost`` and ``stale`` so the last four map exactly).
V1_INVALID = {
    "slip": "slip_detected",
    "occluded": "occlusion",
    "identity_changed": "identity_changed",
    "released": "released",
    "regrasped": "regrasped",
    "width_drop": "slip_detected",
    "width_open": "slip_detected",
    "stale": "occlusion",
    "unsynchronised": "occlusion",
    "unusable_observation": "occlusion",
}


@dataclass(frozen=True)
class HeldObject:
    object_id: str
    object_class: str
    spec: ObjectSpec
    gripper_id: str
    T_tcp_object: np.ndarray
    source: str
    timestamp_s: float
    expected_width_m: float
    valid: bool = True
    invalid_reason: str = ""
    privileged: bool = False
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def invalid_kind(self) -> str | None:
        return self.invalid_reason.split(":", 1)[0] if self.invalid_reason else None

    def to_v1(self) -> dict[str, Any]:
        """The contract's ``HeldObject``. Kinds the v1 enum lacks map to its nearest
        value; the exact kind and text are kept in ``invalidated_detail``."""
        if self.source == "geometric_prior":
            estimate = "grasp_geometry_prior"
        elif self.privileged:
            estimate = "privileged_sim"
        else:
            estimate = "estimated_from_observation"
        u = self.quality.get("observation") or self.quality.get("planned_on") or {}
        pos = u.get("position_m", u.get("position_std_m"))
        rot = u.get("rotation_rad", u.get("rotation_std_rad"))
        return {
            "object_id": self.object_id,
            "gripper_id": self.gripper_id,
            "T_TCP_object": T_to_pose(
                to_contract_frame(self.T_tcp_object, self.object_class), "tcp"
            ),
            "estimate": estimate,
            "timestamp_s": float(self.timestamp_s),
            "uncertainty": None
            if pos is None and rot is None
            else {"position_m": pos, "rotation_rad": rot},
            "collision_geometry": {"type": "box", "size_m": [float(v) for v in self.spec.size_m]},
            "valid": self.valid,
            "invalidated_reason": None
            if self.valid
            else V1_INVALID.get(self.invalid_kind or "", "slip_detected"),
            "invalidated_detail": self.invalid_reason or None,
            "class": self.object_class,
            "expected_width_m": self.expected_width_m,
            "privileged": self.privileged,
            "quality": jsonable(self.quality),
        }

    to_dict = to_v1

    @classmethod
    def from_v1(cls, d: dict[str, Any], spec: ObjectSpec) -> HeldObject:
        source = "geometric_prior" if d["estimate"] == "grasp_geometry_prior" else "estimated"
        return cls(
            object_id=d["object_id"],
            object_class=d.get("class", spec.object_class),
            spec=spec,
            gripper_id=d["gripper_id"],
            T_tcp_object=from_contract_frame(
                pose_to_T(d["T_TCP_object"], "tcp"), d.get("class", spec.object_class)
            ),
            source=source,
            timestamp_s=float(d["timestamp_s"]),
            expected_width_m=float(d.get("expected_width_m", sorted(spec.size_m)[0])),
            valid=bool(d["valid"]),
            invalid_reason=d.get("invalidated_detail") or (d.get("invalidated_reason") or ""),
            privileged=bool(d.get("privileged")) or d["estimate"] == "privileged_sim",
            quality=dict(d.get("quality") or {}),
        )


def prior_from_grasp(
    cand: GraspCandidate,
    spec: ObjectSpec,
    gripper_id: str,
    timestamp_s: float,
    obs: ObjectObservation | None = None,
) -> HeldObject:
    """The commanded grasp as a held state. Its uncertainty is the pose
    uncertainty of the observation the grasp was planned on (if given), plus an
    unknown grasp-execution error, so it is never reported as measured."""
    q: dict[str, Any] = {
        "kind": "prior",
        "note": "commanded grasp offset; not measured after closing",
    }
    if obs is not None:
        q["planned_on"] = {
            "provider": obs.provider,
            "timestamp_s": obs.timestamp_s,
            "position_std_m": obs.uncertainty.position_std_m,
            "rotation_std_rad": obs.uncertainty.rotation_std_rad,
        }
    return HeldObject(
        object_id=cand.object_id,
        object_class=spec.object_class,
        spec=spec,
        gripper_id=gripper_id,
        T_tcp_object=cand.T_tcp_object,
        source="geometric_prior",
        timestamp_s=float(timestamp_s),
        expected_width_m=cand.grasp_width_m,
        privileged=bool(obs.privileged) if obs is not None else False,
        quality=q,
    )


def _fold(T_tcp_obj: np.ndarray, ref: np.ndarray, spec: ObjectSpec) -> np.ndarray:
    """Symmetric equivalent of ``T_tcp_obj`` nearest ``ref`` in rotation."""
    best, best_a = T_tcp_obj, math.inf
    for S in symmetry_rotations(spec):
        C = T_tcp_obj.copy()
        C[:3, :3] = T_tcp_obj[:3, :3] @ S
        a = rotation_angle(C[:3, :3], ref[:3, :3])
        if a < best_a:
            best, best_a = C, a
    return best


def deviation(Ta: np.ndarray, Tb: np.ndarray) -> dict[str, float]:
    return {
        "translation_m": float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])),
        "rotation_deg": math.degrees(rotation_angle(Ta[:3, :3], Tb[:3, :3])),
    }


def estimate_from_observation(
    held: HeldObject,
    obs: ObjectObservation,
    T_frame_tcp: np.ndarray,
    tcp_timestamp_s: float,
    now_s: float,
    max_age_s: float = MAX_AGE_S,
    slip_translation_m: float = SLIP_TRANSLATION_M,
    slip_rotation_deg: float = SLIP_ROTATION_DEG,
    allow_privileged: bool = False,
) -> tuple[HeldObject, dict[str, Any]]:
    """Combine an observation of the held part with the TCP pose read at the
    same time. Returns the new held state and an event record."""
    ev: dict[str, Any] = {"object_id": held.object_id, "observation_provider": obs.provider}
    if obs.object_id != held.object_id or obs.object_class != held.object_class:
        return _invalid(
            held,
            "identity_changed",
            f"observed {obs.object_id}/{obs.object_class}, holding {held.object_id}/{held.object_class}",
        ), ev
    ok, why = obs.usable(now_s, max_age_s, allow_privileged)
    if not ok:
        kind = (
            "occluded"
            if not obs.has_pose
            else ("stale" if why.startswith("stale") else "unusable_observation")
        )
        return _invalid(held, kind, why), ev
    if abs(obs.timestamp_s - tcp_timestamp_s) > SYNC_TOL_S:
        return _invalid(
            held,
            "unsynchronised",
            f"observation and TCP reading {abs(obs.timestamp_s - tcp_timestamp_s):.3f} s apart, over {SYNC_TOL_S} s",
        ), ev
    assert obs.T_frame_object is not None
    est = _fold(inv_T(T_frame_tcp) @ obs.T_frame_object, held.T_tcp_object, held.spec)
    dev = deviation(est, held.T_tcp_object)
    ev["deviation_from_previous"] = dev
    q = {
        "kind": "estimated",
        "observation": {
            "provider": obs.provider,
            "visibility": obs.visibility,
            **obs.uncertainty.to_dict(),
        },
        "deviation_from_previous": dev,
        "previous_source": held.source,
    }
    new = replace(
        held,
        T_tcp_object=est,
        source="estimated",
        timestamp_s=float(obs.timestamp_s),
        valid=True,
        invalid_reason="",
        privileged=held.privileged or obs.privileged,
        quality=q,
    )
    if dev["translation_m"] > slip_translation_m or dev["rotation_deg"] > slip_rotation_deg:
        return _invalid(
            new,
            "slip",
            f"moved {dev['translation_m'] * 1e3:.1f} mm / {dev['rotation_deg']:.1f} deg in the hand",
        ), ev
    return new, ev


def register_held_points(
    held: HeldObject,
    points_frame: np.ndarray,
    T_frame_tcp: np.ndarray,
    viewpoint_frame: np.ndarray,
    timestamp_s: float,
    frame_id: str,
) -> ObjectObservation:
    """pose6d fit of the held part's depth points (frame coordinates), seeded with
    the held prior; the result is an ordinary observation for
    :func:`estimate_from_observation`. The hand's own points must be masked out
    upstream (the fingers are boxes too)."""
    from .contract import Uncertainty

    p6 = _hub.load("pose6d")
    init = T_frame_tcp @ held.T_tcp_object
    r = p6.register(
        points_frame, {"box": list(held.spec.size_m)}, init=init, viewpoint=viewpoint_frame
    )
    ev = {k: r[k] for k in ("status", "fitness", "rmse", "coverage", "n_points")}
    ev["tool"] = "tools/pose6d primitive fast path, init = held prior"
    if r["status"] == "poor_fit":
        return ObjectObservation(
            held.object_id,
            held.object_class,
            held.spec.size_m,
            frame_id,
            timestamp_s,
            "rgbd_pose6d",
            False,
            "occluded",
            None,
            Uncertainty(None, None, "fit_residual"),
            reason="pose6d poor_fit on the held part",
            evidence=ev,
        )
    rmse = float(r["rmse"])
    return ObjectObservation(
        held.object_id,
        held.object_class,
        held.spec.size_m,
        frame_id,
        timestamp_s,
        "rgbd_pose6d",
        False,
        "visible" if r["coverage"] >= 0.6 else "partially_occluded",
        np.asarray(r["T"], dtype=float),
        Uncertainty(
            max(rmse, 5e-4),
            max(math.atan2(rmse, sorted(held.spec.size_m)[1] / 2), math.radians(0.5)),
            "fit_residual",
            r["status"] == "ambiguous_fit",
            int(r["symmetry_order"]),
            float(r["yaw_ambiguity_deg"]),
            {"fitness": float(r["fitness"]), "rmse_m": rmse, "coverage": float(r["coverage"])},
        ),
        evidence=ev,
    )


def check_held(
    held: HeldObject,
    now_s: float,
    gripper_width_m: float | None = None,
    max_age_s: float | None = None,
    width_tol_m: float = WIDTH_TOL_M,
) -> tuple[HeldObject, dict[str, Any]]:
    """Checks that need no new image: jaw width against the part, and age (only
    for an ``estimated`` state; a prior has no measurement to age)."""
    ev: dict[str, Any] = {"object_id": held.object_id}
    if not held.valid:
        return held, ev
    if gripper_width_m is not None:
        d = float(gripper_width_m) - held.expected_width_m
        ev["width_error_m"] = d
        if d < -width_tol_m:
            return _invalid(
                held,
                "width_drop",
                f"jaws at {gripper_width_m * 1e3:.1f} mm, part is {held.expected_width_m * 1e3:.1f} mm: slipped out or crushed",
            ), ev
        if d > width_tol_m:
            return _invalid(
                held,
                "width_open",
                f"jaws at {gripper_width_m * 1e3:.1f} mm, wider than the part: not gripping",
            ), ev
    if (
        held.source == "estimated"
        and max_age_s is not None
        and now_s - held.timestamp_s > max_age_s
    ):
        return _invalid(
            held, "stale", f"estimate {now_s - held.timestamp_s:.2f} s old, limit {max_age_s} s"
        ), ev
    return held, ev


def _invalid(held: HeldObject, kind: str, why: str) -> HeldObject:
    q = dict(held.quality)
    q["recovery"] = [
        "reobserve the held part (wrist close-up) before any placement",
        "if lost: observe the table and regrasp",
    ]
    return replace(held, valid=False, invalid_reason=f"{kind}: {why}", quality=q)
