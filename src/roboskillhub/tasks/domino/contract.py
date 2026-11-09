"""Request validation, thresholds, capability gates and the result envelope, aligned to the
shared contract ``rsh-task/1`` (Agent 05: output/lego-domino/agent-05/contract-v1.*).

Metres, seconds (Unix timestamps), radians, quaternion xyzw, explicit ``frame_id``;
``succeeded / failed / unverified / rejected``. Module-internal fault codes (finer than
the shared vocabulary) map onto ``failure_type`` through :data:`FAILURE_TYPE`.
"""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

import yaml

SHARED_CONTRACT = "rsh-task/1"
MODULE_CONTRACT = "domino-task/0.2.0"
SKILL_VERSION = "0.1.0"
CONTRACT_VERSION = SHARED_CONTRACT

STATUSES = ("succeeded", "failed", "unverified", "rejected")

# Module fault code -> rsh-task/1 failure_type (core or domino.*).
FAILURE_TYPE = {
    "invalid_request": "invalid_request",
    "capability_missing": "capability_missing",
    "dependency_missing": "precondition_unmet",
    "infeasible_geometry": "unreachable",
    "neighbor_clearance": "domino.neighbour_clearance",
    "retreat_blocked": "domino.retreat_blocked",
    "holding_not_confirmed": "precondition_unmet",
    "object_dropped": "grasp_lost",
    "descent_blocked": "domino.descent_blocked",
    "placement_error": "domino.pose_out_of_tolerance",
    "tcp_not_converged": "motion_not_converged",
    "support_mismatch": "domino.support_mismatch",
    "release_failed": "release_failed",
    "release_not_confirmed": "evidence_missing",
    "not_resting_on_support": "domino.not_resting_on_support",
    "unstable_after_release": "domino.toppled_after_release",
    "neighbor_disturbed": "domino.neighbour_disturbed",
    "observation_stale": "stale_observation",
    "observation_repeated": "evidence_missing",
    "insufficient_settle_window": "evidence_missing",
    "orientation_missing": "evidence_missing",
    "identity_mismatch": "perception.low_confidence",
    "target_occluded": "object_not_observed",
    "support_unconfirmed": "evidence_missing",
    "privileged_observation": "evidence_missing",
    "timeout": "timeout",
    "cancelled": "cancelled",
    "backend_error": "backend_error",
}
FAILURE_CODES = tuple(FAILURE_TYPE)

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

# Module phases -> rsh-task/1 phases.
PHASE = {
    "validate": "validate",
    "pre_observe": "observe",
    "carry": "transport",
    "carry_check": "transport",
    "descent": "descend",
    "support_check": "contact",
    "release": "release",
    "release_check": "release",
    "retreat": "retreat",
    "settle": "observe",
    "verify": "verify",
}

# Backend capabilities (rsh-task/1 section 5) place_upright needs, before any command.
# Release is the set_gripper primitive (purpose=release), which needs only "gripper";
# straight legs are move_waypoints (linear=false, dense) unless descent_path="linear".
PLACE_REQUIRES = frozenset(
    {
        "pose_feedback",
        "cartesian_motion",
        "gripper",
        "stop",
        "object_feedback",
        "object_pose_feedback",
    }
)


def place_requires(mode: str, descent_path: str) -> frozenset[str]:
    if mode not in ("geometric", "contact_feedback"):
        raise RequestRejected("invalid_request", "mode must be geometric or contact_feedback")
    if descent_path not in ("dense_waypoints", "linear"):
        raise RequestRejected("invalid_request", "descent_path must be dense_waypoints or linear")
    extra = set()
    if descent_path == "linear":
        extra.add("linear_cartesian_motion")
    if mode == "contact_feedback":
        extra.add("contact_force_feedback")
    return PLACE_REQUIRES | extra


CAPABILITIES = {
    "geometric": place_requires("geometric", "dense_waypoints"),
    "contact_feedback": place_requires("contact_feedback", "dense_waypoints"),
}

RECOVERY_HINTS = {
    "none": [],
    "reobserve": ["reobserve"],
    "regrasp": ["reobserve", "regrasp"],
    "replan": ["replan"],
    "fix_request": ["abort"],
    "retreat_and_retry": ["back_off", "reobserve", "retry_same"],
    "recover_fallen_tile": ["reobserve", "reset_piece"],
    "restore_segment": ["reobserve", "reset_piece", "replan"],
    "regrasp_and_replace": ["reobserve", "regrasp", "retry_same"],
    "release_again": ["reobserve", "retry_same"],
    "abort": ["abort", "human"],
}


class RequestRejected(ValueError):
    """Invalid request or a missing precondition, found before any command."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# ----------------------------------------------------------------------------- thresholds

CONFIG_ROOT = Path(__file__).resolve().parent / "config"
DEFAULT_TOLERANCES = "tasks/domino/config/line_v1.yaml"
TIGHTER_IS_SMALLER = {
    "position_tolerance_m",
    "tilt_tolerance_rad",
    "yaw_tolerance_rad",
    "support_height_tolerance_m",
    "tcp_tolerance_m",
    "drift_tolerance_m",
    "drift_tilt_tolerance_rad",
    "neighbor_displacement_tolerance_m",
    "neighbor_tilt_tolerance_rad",
    "max_observation_age_s",
    "max_holding_std_m",
    "waypoint_step_m",
}
TIGHTER_IS_LARGER = {"settle_window_s", "min_window_samples", "min_identity_confidence"}
THRESHOLD_KEYS = tuple(
    sorted(TIGHTER_IS_SMALLER | TIGHTER_IS_LARGER | {"release_clearance_m", "release_gap_m"})
)


def load_config(ref: str = DEFAULT_TOLERANCES) -> dict:
    """A versioned threshold file under tasks/domino/config/ (no other paths)."""
    name = Path(str(ref)).name
    if not str(ref).replace("\\", "/").endswith(f"tasks/domino/config/{name}") and str(ref) != name:
        raise RequestRejected(
            "invalid_request", f"tolerances_ref {ref!r} is not a tasks/domino/config file"
        )
    path = CONFIG_ROOT / name
    if not path.is_file():
        raise RequestRejected("invalid_request", f"no threshold config {name!r}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    th = cfg.get("thresholds") or {}
    missing = [k for k in THRESHOLD_KEYS if k not in th]
    if missing:
        raise RequestRejected("invalid_request", f"{name} lacks {missing}")
    return {**cfg, "ref": f"tasks/domino/config/{name}"}


def thresholds_for(request: dict) -> tuple[dict[str, float], str]:
    """The config's thresholds, tightened (never loosened) by ``thresholds_override``."""
    cfg = load_config(request.get("tolerances_ref") or DEFAULT_TOLERANCES)
    th = {k: float(v) for k, v in cfg["thresholds"].items()}
    over = request.get("thresholds_override") or {}
    if not isinstance(over, dict):
        raise RequestRejected("invalid_request", "thresholds_override must be an object")
    for k, v in over.items():
        v = finite(v, f"thresholds_override.{k}", 0.0, 1e3)
        if k in TIGHTER_IS_SMALLER and v <= th[k]:
            th[k] = v
        elif k in TIGHTER_IS_LARGER and v >= th[k]:
            th[k] = v
        else:
            raise RequestRejected(
                "invalid_request",
                f"thresholds_override.{k}={v} would loosen or is not overridable ({cfg['ref']}: {th.get(k)})",
            )
    th["min_window_samples"] = int(max(2, th["min_window_samples"]))
    return th, cfg["ref"]


def threshold_profile(ref: str = DEFAULT_TOLERANCES) -> dict[str, float]:
    return thresholds_for({"tolerances_ref": ref})[0]


# ----------------------------------------------------------------------------- validation


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
    """``{"size_m": [h, w, t]}`` (rsh-task/1 piece/geometry) or ``{height_m, width_m,
    thickness_m}``; origin geometric_center (default, rsh-task/1) or bottom_face_center."""
    if not isinstance(value, dict):
        raise RequestRejected("invalid_request", "geometry must be an object")
    if "size_m" in value:
        h, w, t = vec(value["size_m"], "geometry.size_m", 3)
        value = {**value, "height_m": h, "width_m": w, "thickness_m": t}
    g = {
        k: finite(value.get(k), f"geometry.{k}", 1e-4, 1.0)
        for k in ("height_m", "width_m", "thickness_m")
    }
    if not g["thickness_m"] < g["width_m"] < g["height_m"]:
        raise RequestRejected(
            "invalid_request", "geometry must be a tile: thickness < width < height"
        )
    origin = value.get("origin", "geometric_center")
    if origin not in ("geometric_center", "bottom_face_center"):
        raise RequestRejected(
            "invalid_request", f"geometry.origin {origin!r}: geometric_center or bottom_face_center"
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
        "observation_id": value.get("observation_id"),
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


# ----------------------------------------------------------------------------- results


def envelope(
    skill_id: str,
    request: dict,
    *,
    kind: str,
    ctx: Any = None,
) -> dict:
    """A rsh-task/1 SkillResult, status rejected until filled in."""
    return {
        "schema_version": 1,
        "contract_version": SHARED_CONTRACT,
        "module_contract": MODULE_CONTRACT,
        "invocation_id": str(uuid.uuid4()),
        "task_id": getattr(ctx, "task_id", None) or request.get("task_id"),
        "request_id": getattr(ctx, "request_id", None) or request.get("request_id"),
        "capability": skill_id,
        "skill": skill_id,
        "version": SKILL_VERSION,
        "kind": kind,
        "status": "rejected",
        "phase": "validate",
        "failure_type": None,
        "failure": None,
        "cause_hypotheses": [],
        "recovery_hints": [],
        "evidence": {},
        "observations": [],
        "evidence_refs": [],
        "children": [],
        "privileged_inputs": False,
        "unverified": [],
        "outputs": {},
        "trace": [],
        "elapsed_s": 0.0,
    }


def set_failure(
    result: dict,
    code: str,
    phase: str,
    detail: str,
    status: str | None = None,
    failure_type: str | None = None,
    **extra: Any,
) -> None:
    if status:
        result["status"] = status
    result["failure"] = {"code": code, "phase": phase, "detail": detail, **extra}
    result["failure_type"] = failure_type or FAILURE_TYPE.get(code, "postcondition_failed")
    result["phase"] = PHASE.get(phase, phase)
    result["error"] = {"code": code, "message": detail}
