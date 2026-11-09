"""Request validation, statuses, failure codes and capability gates for the domino task.

Field names follow the workcell runtime's contract (metres, quaternion xyzw, explicit
``frame_id``, Unix-second timestamps, ``succeeded / failed / unverified / rejected``).
Agent 05 owns the public schema; until its contract v1 lands these are this module's
own, versioned by :data:`CONTRACT_VERSION`.
"""

from __future__ import annotations

import math
from typing import Any

from . import upstream

CONTRACT_VERSION = "domino-task/0.1.0"
SKILL_VERSION = "0.1.0"

STATUSES = ("succeeded", "failed", "unverified", "rejected")

FAILURE_CODES = (
    "invalid_request",
    "capability_missing",
    "dependency_missing",
    "infeasible_geometry",
    "neighbor_clearance",
    "retreat_blocked",
    "holding_not_confirmed",
    "object_dropped",
    "descent_blocked",
    "placement_error",
    "support_mismatch",
    "release_not_confirmed",
    "unstable_after_release",
    "neighbor_disturbed",
    "observation_stale",
    "observation_repeated",
    "orientation_missing",
    "identity_mismatch",
    "target_occluded",
    "insufficient_settle_window",
    "timeout",
    "cancelled",
    "backend_error",
)

CAUSE_HYPOTHESES = (
    "table_calibration",
    "grasp_offset",
    "slip",
    "release_drag",
    "retreat_contact",
    "tcp_tracking",
    "gripper_release_fault",
    "unknown",
)

# Capabilities a backend must declare (``describe()["capabilities"]``) for each mode.
# Only pose_feedback, cartesian_motion, gripper, object_feedback and stop exist in the
# workcell runtime today; the rest are requested from Agent 05 (integration-request.md).
CAPABILITIES = {
    "geometric": {
        "pose_feedback",
        "cartesian_motion",
        "linear_cartesian_motion",
        "gripper_release",
        "object_pose_feedback",
        "stop",
        "station_transaction",
    },
}
CAPABILITIES["contact_feedback"] = CAPABILITIES["geometric"] | {"guarded_descent"}

PROFILE_NOTES = {
    "position_tolerance_m": "place_domino.POSITION_TOLERANCE (a 10 deg lean moves a 48 mm tile's top 8.3 mm; camera RMS 5.4 mm)",
    "tilt_tolerance_rad": "predicators domino_component fallen_threshold, 10 deg (place_domino's provenance)",
    "yaw_tolerance_rad": "place_domino.YAW_TOLERANCE_DEG (servo 3 deg + 1)",
    "support_height_tolerance_m": "place_domino.DESCENT_TOLERANCE (6 mm)",
    "tcp_tolerance_m": "place_domino.DESCENT_TOLERANCE (a descent ending this high stopped on something)",
    "drift_tolerance_m": "choice: 2 mm, under the twin's 0.1-1.9 mm placement error spread",
    "drift_tilt_tolerance_rad": "choice: 2 deg",
    "neighbor_displacement_tolerance_m": "choice: 3 mm (twin: undisturbed neighbours moved < 0.01 mm)",
    "neighbor_tilt_tolerance_rad": "choice: 5 deg",
    "settle_window_s": "choice: 0.5 s, the twin's settle time before measuring",
    "min_window_samples": "choice: 3 distinct frames",
    "max_observation_age_s": "workcell runtime Limits.max_state_age_s default (1 s)",
    "min_identity_confidence": "choice: 0.5",
    "max_holding_std_m": "choice: half the position tolerance",
    "release_clearance_m": "place_domino.RELEASE_CLEARANCE (4 mm a side)",
    "release_gap_m": "place_domino.RELEASE_GAP (1.5 mm)",
}


class RequestRejected(ValueError):
    """Invalid request or a missing precondition, found before any command."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def threshold_profile(name: str = "yam-twin-48mm-v0") -> dict[str, float]:
    """A named, sourced threshold set. Values tied to the placing skill are read from
    it, not restated; the rest are labelled choices in :data:`PROFILE_NOTES`."""
    if name != "yam-twin-48mm-v0":
        raise KeyError(f"unknown threshold profile {name!r}")
    pd = upstream.place()
    pos = float(pd.POSITION_TOLERANCE)
    return {
        "position_tolerance_m": pos,
        "tilt_tolerance_rad": math.radians(10.0),
        "yaw_tolerance_rad": math.radians(float(pd.YAW_TOLERANCE_DEG)),
        "support_height_tolerance_m": float(pd.DESCENT_TOLERANCE),
        "tcp_tolerance_m": float(pd.DESCENT_TOLERANCE),
        "drift_tolerance_m": 0.002,
        "drift_tilt_tolerance_rad": math.radians(2.0),
        "neighbor_displacement_tolerance_m": 0.003,
        "neighbor_tilt_tolerance_rad": math.radians(5.0),
        "settle_window_s": 0.5,
        "min_window_samples": 3,
        "max_observation_age_s": 1.0,
        "min_identity_confidence": 0.5,
        "max_holding_std_m": pos / 2.0,
        "release_clearance_m": float(pd.RELEASE_CLEARANCE),
        "release_gap_m": float(pd.RELEASE_GAP),
    }


THRESHOLD_KEYS = tuple(PROFILE_NOTES)


def finite(value: Any, name: str, low: float = -1e6, high: float = 1e6) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestRejected("invalid_request", f"{name} must be a number")
    v = float(value)
    if not math.isfinite(v) or not low <= v <= high:
        raise RequestRejected("invalid_request", f"{name} must be within [{low}, {high}]")
    return v


def vec(value: Any, name: str, n: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise RequestRejected("invalid_request", f"{name} must have {n} numbers")
    return [finite(v, name) for v in value]


def check_pose(value: Any, name: str, frame_id: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"frame_id", "position_m", "quaternion_xyzw"}:
        raise RequestRejected(
            "invalid_request", f"{name} needs exactly frame_id, position_m, quaternion_xyzw"
        )
    if value["frame_id"] != frame_id:
        raise RequestRejected(
            "invalid_request",
            f"{name} is in {value['frame_id']!r}, not {frame_id!r}: transform it explicitly first",
        )
    p = vec(value["position_m"], f"{name}.position_m", 3)
    q = vec(value["quaternion_xyzw"], f"{name}.quaternion_xyzw", 4)
    n = math.sqrt(sum(x * x for x in q))
    if abs(n - 1.0) > 0.01:
        raise RequestRejected("invalid_request", f"{name}.quaternion_xyzw must be unit (xyzw)")
    return {"frame_id": frame_id, "position_m": p, "quaternion_xyzw": [x / n for x in q]}


def check_geometry(value: Any) -> dict:
    if not isinstance(value, dict):
        raise RequestRejected("invalid_request", "geometry must be an object")
    g = {
        k: finite(value.get(k), f"geometry.{k}", 1e-4, 1.0)
        for k in ("height_m", "width_m", "thickness_m")
    }
    if not g["thickness_m"] < g["width_m"] < g["height_m"]:
        raise RequestRejected(
            "invalid_request", "geometry must be a tile: thickness_m < width_m < height_m"
        )
    origin = value.get("origin", "bottom_face_center")
    if origin != "bottom_face_center":
        raise RequestRejected(
            "invalid_request",
            f"geometry.origin {origin!r}: this contract fixes the origin at the bottom face centre",
        )
    sym = finite(value.get("yaw_symmetry_deg", 180.0), "geometry.yaw_symmetry_deg", 1.0, 360.0)
    if sym not in (180.0, 360.0):
        raise RequestRejected("invalid_request", "geometry.yaw_symmetry_deg must be 180 or 360")
    up_sym = value.get("up_symmetric", True)
    if not isinstance(up_sym, bool):
        raise RequestRejected("invalid_request", "geometry.up_symmetric must be boolean")
    return {**g, "origin": origin, "yaw_symmetry_deg": sym, "up_symmetric": up_sym}


def check_support(value: Any, frame_id: str) -> dict:
    if not isinstance(value, dict) or value.get("frame_id") != frame_id:
        raise RequestRejected("invalid_request", f"support must be an object in {frame_id!r}")
    point = vec(value.get("point_m"), "support.point_m", 3)
    normal = vec(value.get("normal"), "support.normal", 3)
    n = math.sqrt(sum(x * x for x in normal))
    if n < 1e-9:
        raise RequestRejected("invalid_request", "support.normal must be nonzero")
    out = {
        "frame_id": frame_id,
        "point_m": point,
        "normal": [x / n for x in normal],
        "height_std_m": finite(value.get("height_std_m"), "support.height_std_m", 0.0, 0.1),
        "calibration_version": value.get("calibration_version"),
    }
    if not isinstance(out["calibration_version"], str) or not out["calibration_version"]:
        raise RequestRejected("invalid_request", "support.calibration_version is required")
    if "region_min_m" in value or "region_max_m" in value:
        lo = vec(value.get("region_min_m"), "support.region_min_m", 2)
        hi = vec(value.get("region_max_m"), "support.region_max_m", 2)
        if lo[0] >= hi[0] or lo[1] >= hi[1]:
            raise RequestRejected("invalid_request", "support region min must be below max")
        out["region_min_m"], out["region_max_m"] = lo, hi
    return out


def check_thresholds(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise RequestRejected(
            "invalid_request", "thresholds must be given (e.g. contract.threshold_profile())"
        )
    missing = [k for k in THRESHOLD_KEYS if k not in value]
    if missing:
        raise RequestRejected("invalid_request", f"thresholds missing {missing}")
    out = {k: finite(value[k], f"thresholds.{k}", 0.0, 100.0) for k in THRESHOLD_KEYS}
    out["min_window_samples"] = int(max(2, out["min_window_samples"]))
    return out


def check_capabilities(description: dict, mode: str) -> list[str]:
    if mode not in CAPABILITIES:
        raise RequestRejected("invalid_request", f"mode must be one of {sorted(CAPABILITIES)}")
    have = set(description.get("capabilities") or [])
    return sorted(CAPABILITIES[mode] - have)


def result_skeleton(kind: str, request_id: Any) -> dict:
    return {
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "skill": f"domino.{kind}",
        "skill_version": SKILL_VERSION,
        "request_id": request_id,
        "status": "rejected",
        "failure": None,
        "cause_hypotheses": [],
        "phase_trace": [],
        "evidence": {},
        "unverified": [],
        "provenance": upstream.provenance(),
    }
