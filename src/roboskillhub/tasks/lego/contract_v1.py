"""Conversions between contract ``rsh-task/1`` types and this module's internal requests.

- ``Observation`` (contract) -> internal observation dict (``obs_id/kind/value/...``).
- ``HeldObject`` -> internal ``held`` (with the grasp confirmation it carries).
- ``LegoMatingTarget`` -> internal plate / brick / cell, cross-checked against its
  ``mating_pose``.
- internal result -> contract ``SkillResult`` envelope (``failure_type`` vocabulary,
  ``phase`` enum, ``recovery_hints``, ``evidence.postcondition``).

Conventions this module adds to ``LegoMatingTarget`` (open fields in the v1 schema; see
integration-request.md): ``baseplate.studs`` [nx, ny], ``baseplate.calibration_id``,
optional ``baseplate.system`` / ``brick.system`` / ``baseplate.bias_m``;
``T_W_baseplate`` is the plate's top-surface centre (stud bases), z out of the studs;
``placement.layer`` 0 stands on the plate; a lego_brick's object frame (``mating_pose``,
observations) is its bottom-face centre, z up through the studs, x along ``studs[0]``.
"""

from __future__ import annotations

import math
from typing import Any

from . import geometry as g
from .contracts import Rejected

PITCH_TO_SYSTEM = {0.008: "lego", 0.016: "duplo"}
COMMAND_SOURCES = ("command", "commanded", "target", "planned", "goal")
MATING_POSE_TOLERANCE_M = 1e-4
MATING_POSE_TOLERANCE_DEG = 0.5

FAILURE_TYPES: dict[str, str] = {
    # rejected before acting
    "invalid_input": "invalid_request",
    "invalid_geometry": "invalid_request",
    "invalid_observation": "invalid_request",
    "version_mismatch": "invalid_request",
    "axis_mismatch": "invalid_request",
    "travel_too_long": "invalid_request",
    "travel_too_short": "invalid_request",
    "unlabelled_privileged": "invalid_request",
    "frame_mismatch": "invalid_request",
    "unknown_reference": "precondition_unmet",
    "identity_mismatch": "precondition_unmet",
    "evidence_mismatch": "precondition_unmet",
    "not_aligned": "precondition_unmet",
    "uncalibrated": "precondition_unmet",
    "tolerance_exceeds_capture": "precondition_unmet",
    "out_of_plate": "precondition_unmet",
    "target_occupied": "precondition_unmet",
    "interface_mismatch": "lego.interface_mismatch",
    "insufficient_support": "lego.insufficient_support",
    "would_damage_build": "lego.would_damage_build",
    "ambiguous_orientation": "perception.symmetry_ambiguous",
    "unreachable": "unreachable",
    "grasp_unconfirmed": "precondition_unmet",
    "missing_capabilities": "capability_missing",
    "resolution_insufficient": "perception.low_confidence",
    "force_limit_unsupported": "lego.contact_control_unavailable",
    "torque_limit_unsupported": "lego.contact_control_unavailable",
    "speed_unsupported": "lego.contact_control_unavailable",
    "travel_unsupported": "lego.contact_control_unavailable",
    # observation problems
    "stale_observation": "stale_observation",
    "predates_action": "stale_observation",
    "timestamp_in_future": "evidence_missing",
    "commanded_not_observed": "evidence_missing",
    "occluded": "object_not_observed",
    "brick_not_visible": "object_not_observed",
    "no_usable_observation": "object_not_observed",
    "orientation_missing": "evidence_missing",
    "uncertainty_unknown": "evidence_missing",
    "window_incomplete": "evidence_missing",
    "release_unconfirmed": "evidence_missing",
    "neighbour_unobserved": "evidence_missing",
    "contact_signal_lost": "evidence_missing",
    "no_contact_record": "evidence_missing",
    "invalid_corners": "evidence_missing",
    "seat_depth_without_threshold": "lego.seat_depth_without_threshold",
    # execution
    "alignment_not_converged": "motion_not_converged",
    "correction_budget_exhausted": "lego.misaligned",
    "deadline_exceeded": "timeout",
    "timeout": "timeout",
    "motion_failed": "backend_error",
    "backend_fault": "backend_error",
    "not_executed": "backend_error",
    "stop_unconfirmed": "lego.stop_unconfirmed",
    "held_brick_tilted": "held_object_slipped",
    "force_limit_exceeded": "lego.force_limit_exceeded",
    "early_contact": "lego.early_contact",
    "no_contact": "lego.no_contact",
    "blocked_on_studs": "lego.misaligned",
    # verification verdicts
    "lateral_offset": "lego.misaligned",
    "wrong_cell": "lego.misaligned",
    "angle_error": "lego.misaligned",
    "not_pressed_fully": "lego.not_seated",
    "resting_on_studs": "lego.not_seated",
    "off_studs": "lego.not_seated",
    "below_seat": "lego.insufficient_support",
    "partial_engagement": "lego.partially_seated",
    "tilted": "lego.partially_seated",
    "lifted_on_release": "lego.lifted_with_gripper",
    "moved_after_release": "lego.moved_after_release",
    "brick_missing": "lego.brick_missing",
    "build_disturbed": "lego.structure_damaged",
    "release_failed": "release_failed",
}

PHASES = {
    "precheck": "validate",
    "approach": "approach",
    "correct": "approach",
    "observe": "observe",
    "press": "contact",
    "verify": "verify",
    "release": "release",
    "retreat": "retreat",
}
HINTS = {
    "reobserve": "reobserve",
    "reobserve_grasp": "reobserve",
    "replan": "replan",
    "regrasp": "regrasp",
    "human": "human",
    "use_finer_observer": "replan",
    "bounded_retreat_if_state_fresh": "back_off",
    "report_connection_failure": "abort",
}


def failure_type(reason_code: str) -> str:
    return FAILURE_TYPES.get(reason_code, f"lego.{reason_code}" if reason_code else "backend_error")


def observation(o: Any, *, brick_height_m: float | None = None) -> dict[str, Any]:
    """Contract Observation -> internal observation dict (never raises on content; problems
    become ``unusable_reason``)."""
    if not isinstance(o, dict):
        raise Rejected("invalid_observation", "observation must be an object")
    oid, src = str(o.get("object_id", "")), str(o.get("source", ""))
    ts = o.get("timestamp_s")
    unc = o.get("uncertainty") or {}
    privileged = bool(o.get("privileged")) or src in ("simulation_truth", "sim_truth")
    kind = "sim_truth" if privileged else ("commanded" if src in COMMAND_SOURCES else "measured")
    reason = ""
    vis = o.get("visibility", "unknown")
    value: Any = None
    if vis in ("occluded", "unknown"):
        reason = "occluded"
    elif vis == "not_found":
        value = None
    elif o.get("position_m") is None:
        reason = "occluded"
    elif o.get("quaternion_xyzw") is None:
        reason = "orientation_missing"
    else:
        pose = {
            "frame_id": o.get("frame_id", ""),
            "position_m": list(o["position_m"]),
            "quaternion_xyzw": list(o["quaternion_xyzw"]),
        }
        origin = (o.get("geometry") or {}).get("origin", "bottom_centre")
        if origin != "bottom_centre":
            if brick_height_m is None or origin not in ("body_centre", "top_centre"):
                reason = "unknown_object_frame"
            else:
                down = brick_height_m / 2 if origin == "body_centre" else brick_height_m
                pose = g.to_pose(g.mul(g.matrix(pose), g.tr(0, 0, -down)), pose["frame_id"])
        value = pose
        corners = o.get("corner_bottom_heights_m") or (o.get("geometry") or {}).get(
            "corner_bottom_heights_m"
        )
        if corners is not None:
            value = {"pose": pose, "corner_bottom_heights_m": corners}
    sigma_m = unc.get("position_m")
    sigma_r = unc.get("rotation_rad")
    if sigma_m is None:
        if privileged:
            sigma_m = 0.0
        elif not reason:
            reason = "uncertainty_unknown"
    if ts is None and not reason:
        reason = "stale_observation"
    return {
        "obs_id": f"{src}:{oid}@{ts}",
        "source": src or "unknown",
        "kind": kind,
        "frame_id": o.get("frame_id") or "unknown",
        "timestamp_s": float(ts or 0.0),
        "value": value,
        "uncertainty_m": float(sigma_m or 0.0),
        "uncertainty_deg": math.degrees(sigma_r)
        if sigma_r is not None
        else (0.0 if privileged else 1.0),
        "privileged": privileged,
        "unusable_reason": reason,
    }


def held(h: dict, *, close_axis: int = 0) -> dict[str, Any]:
    """Contract HeldObject -> internal ``held``. ``valid`` with a timestamp is the grasp
    confirmation (from whoever produced the HeldObject: Agent 01's skill or a gripper)."""
    est = h.get("estimate")
    return {
        "object_id": h["object_id"],
        "gripper_id": h["gripper_id"],
        "grasp_id": f"{h['gripper_id']}:{h.get('timestamp_s')}",
        "close_axis": close_axis,
        "T_TCP_object": h["T_TCP_object"],
        "holding_evidence": {
            "obs_id": f"held:{h['object_id']}@{h.get('timestamp_s')}",
            "source": f"held_object:{est}",
            "kind": "sim_truth" if est == "privileged_sim" else "measured",
            "frame_id": "tcp",
            "timestamp_s": h["timestamp_s"],
            "value": bool(h.get("valid")) and h.get("invalidated_reason") is None,
            "uncertainty_m": float((h.get("uncertainty") or {}).get("position_m") or 0.0),
            "privileged": est == "privileged_sim",
        },
    }


def _system(t: dict) -> str:
    bp = t["baseplate"]
    pitch = bp.get("stud_pitch_m")
    by_pitch = next(
        (
            s
            for p, s in PITCH_TO_SYSTEM.items()
            if isinstance(pitch, (int, float)) and abs(pitch - p) < 1e-6
        ),
        None,
    )
    if by_pitch is None:
        raise Rejected("invalid_geometry", f"stud_pitch_m {pitch!r} is neither LEGO nor DUPLO")
    if bp.get("system", by_pitch) != by_pitch:
        raise Rejected("invalid_geometry", "baseplate.system disagrees with stud_pitch_m")
    return by_pitch


def mating_target(t: dict) -> dict[str, Any]:
    """LegoMatingTarget -> internal ``brick, plate, target_cell`` (+ checks of its
    ``mating_pose`` against the placement)."""
    if not isinstance(t, dict) or t.get("kind") != "lego_mating":
        raise Rejected("invalid_input", "target must be a LegoMatingTarget")
    system = _system(t)
    bp, brick, pl = t["baseplate"], t["brick"], t["placement"]
    if "studs" not in bp or not bp.get("calibration_id"):
        raise Rejected(
            "uncalibrated", "baseplate needs studs [nx, ny] and calibration_id (module convention)"
        )
    for k in ("stud_x", "stud_y", "layer", "yaw_quarter_turns"):
        if not isinstance(pl.get(k), int) or isinstance(pl.get(k), bool):
            raise Rejected("invalid_geometry", f"placement.{k} must be an integer (grid index)")
    out = {
        "brick": {"system": brick.get("system", system), "studs": brick["studs"]},
        "plate": {
            "plate_id": bp["object_id"],
            "pose": bp["T_W_baseplate"],
            "studs": bp["studs"],
            "system": system,
            "calibration_id": bp["calibration_id"],
            **({"bias_m": bp["bias_m"]} if "bias_m" in bp else {}),
        },
        "target_cell": {
            "x": pl["stud_x"],
            "y": pl["stud_y"],
            "layer": pl["layer"] + 1,
            "rotation": 90 * (pl["yaw_quarter_turns"] % 4),
        },
    }
    try:
        bt = g.brick_type(out["brick"]["system"], out["brick"]["studs"])
        plate = g.Plate(
            g.validate_pose(bp["T_W_baseplate"], t["frame_id"], "T_W_baseplate"),
            tuple(bp["studs"]),
            g.SYSTEMS[system],
            bp["calibration_id"],
            tuple(bp.get("bias_m", (0.0, 0.0))),
        )
        seat = g.seat_pose(plate, bt, g.Cell.from_dict(out["target_cell"]))
        mp = g.validate_pose(t["mating_pose"], t["frame_id"], "mating_pose")
    except g.GeometryError as exc:
        raise Rejected("invalid_geometry", str(exc)) from exc
    d = g.pose_delta(g.matrix(seat), g.matrix(mp))
    period = 90.0 if bt.studs[0] == bt.studs[1] else 180.0
    yaw = g.wrap_deg(d["yaw_deg"], period)
    if (
        d["lateral_m"] > MATING_POSE_TOLERANCE_M
        or abs(d["dz_m"]) > MATING_POSE_TOLERANCE_M
        or abs(yaw) > MATING_POSE_TOLERANCE_DEG
        or d["tilt_deg"] > MATING_POSE_TOLERANCE_DEG
    ):
        raise Rejected(
            "invalid_geometry",
            f"mating_pose is {d['lateral_m'] * 1e3:.2f} mm / {d['dz_m'] * 1e3:+.2f} mm / "
            f"{yaw:.2f} deg from the placement's seat: frame or layer convention mismatch",
        )
    out["seat_pose"] = seat
    out["requires_support"] = bool(t.get("requires_support", False))
    out["depends_on"] = list(t.get("depends_on", []))
    return out


def placed(t: dict) -> dict[str, Any]:
    """A placed brick (a LegoMatingTarget already done) -> internal build entry."""
    m = mating_target(t)
    return {
        "brick_id": t["target_id"],
        "system": m["brick"]["system"],
        "studs": m["brick"]["studs"],
        "cell": m["target_cell"],
    }


def postcondition(
    name: str, passed: bool, criterion: str, measured: dict, thresholds_ref: str
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "criterion": criterion,
        "measured": measured,
        "thresholds_ref": thresholds_ref,
    }


def envelope(
    skill: Any, ctx: Any, internal: dict, *, observations: list[dict], postcond: dict, outputs: dict
) -> dict[str, Any]:
    """Internal result -> contract SkillResult via the runtime's make_result."""
    from roboskillhub.runtime.task import make_result

    status = internal["status"]
    ft = None if status == "succeeded" else failure_type(internal["reason_code"])
    hints = list(dict.fromkeys(HINTS[h] for h in internal.get("next_allowed", []) if h in HINTS))
    res = make_result(
        skill,
        status,
        phase=PHASES.get(internal.get("phase", ""), "validate"),
        failure_type=ft,
        evidence={
            "postcondition": postcond,
            "lego": {
                k: internal[k]
                for k in (
                    "reason_code",
                    "message",
                    "preconditions",
                    "postconditions",
                    "evidence",
                    "evidence_level",
                    "privileged_observation",
                    "retries",
                    "stop",
                    "motion_issued",
                    "connection_id",
                    "request_id",
                    "invocation_id",
                )
            },
        },
        observations=observations,
        recovery_hints=hints,
        outputs=outputs,
        error=None
        if status == "succeeded"
        else {"code": internal["reason_code"], "message": internal.get("message", "")},
        task_id=getattr(ctx, "task_id", None),
        request_id=getattr(ctx, "request_id", None),
        ctx=ctx,
    )
    res["trace"] = internal.get("trace", [])
    if internal.get("privileged_observation"):
        res["privileged_inputs"] = True
    return res
