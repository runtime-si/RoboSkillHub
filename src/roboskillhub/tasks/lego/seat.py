"""lego.guarded_seat: bounded contact press along the engagement axis.

Requires a ContactBackend with a measured, calibrated contact-force signal, a controller
that stops on it, and confirmed stop. POSITION-ONLY PRESSING IS NOT COMPLIANT FORCE CONTROL:
a backend that only offers position moves with a travel limit (and tracking lag or
joint-effort heuristics, as skills/pressing-a-brick-onto-studs uses) is rejected here
before any motion. Use that skill, labelled as a bounded position approach, if that is all
the station has.

``succeeded`` means the declared bounded operation completed with sufficient evidence
(``operation_complete``). It never means the brick is connected: that is
lego.verify_connection's job (``connection_verified`` is always false here).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from .contracts import Rejected, SkillResult, _Early, ids, number
from .geometry import SYSTEMS
from .ports import CONTACT_CAPABILITIES, ContactBackend

SKILL_ID = "lego.guarded_seat"
VERSION = "0.1.0"

NOT_FORCE_CONTROL = (
    "position-only press (move_pose / travel limit / tracking lag / joint effort) is NOT "
    "compliant force control and is not accepted by lego.guarded_seat"
)

SEAT_FRACTION = 0.35
"""Final travel within this fraction of the stud height of the seated travel is a complete
press (skills/pressing-a-brick-onto-studs' SEAT_FRACTION: 0.6 mm LEGO, 1.6 mm DUPLO)."""

RESTING_FRACTION = 0.6
"""Stopped this fraction of the stud height or more short of the seat: the walls are on the
stud tops (same skill's RESTING_FRACTION)."""

OVERTRAVEL_FRACTION = 0.5
"""max_travel_m may exceed the seated travel by at most this fraction of the stud height."""

EARLY_CONTACT_MARGIN_FRACTION = 0.5
"""Contact this fraction of a stud height before the brick bottom can reach the stud tops
is early: something other than the target studs is in the way."""

FORCE_OVERSHOOT = 1.25
"""Peak force over this multiple of force_limit_N: the controller did not hold its limit."""

CONTACT_FRACTION = 0.2
"""Contact onset: the first sample whose force exceeds this fraction of force_limit_N."""


def _axis(req: dict, alignment: dict) -> list[float]:
    ax = req.get("axis")
    if not isinstance(ax, dict) or set(ax) != {"frame_id", "vector"}:
        raise Rejected("invalid_input", "axis must be {frame_id, vector}")
    planned = alignment.get("engagement_axis")
    if not isinstance(planned, dict):
        raise Rejected("invalid_input", "alignment evidence lacks its engagement_axis")
    if ax["frame_id"] != planned.get("frame_id"):
        raise Rejected("frame_mismatch", "axis frame differs from the alignment's frame")
    v = ax["vector"]
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise Rejected("invalid_input", "axis.vector needs 3 numbers")
    v = [number(x, "axis.vector", -1, 1) for x in v]
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-6:
        raise Rejected("invalid_input", "axis.vector is zero")
    v = [x / n for x in v]
    c = sum(a * b for a, b in zip(v, planned["vector"], strict=True))
    if c < math.cos(math.radians(2.0)):
        raise Rejected("axis_mismatch",
                       f"axis is {math.degrees(math.acos(max(-1, min(1, c)))):.1f} deg off the "
                       "connection's engagement axis")
    return v


def check_contact_capabilities(desc: dict) -> list[str]:
    """Missing capabilities / signal declarations (empty when the backend qualifies)."""
    caps = set(desc.get("capabilities", []))
    missing = sorted(CONTACT_CAPABILITIES - caps)
    sig = desc.get("contact_signal") or {}
    for k in ("source", "quantity", "unit", "frame_id", "rate_hz", "max_latency_s",
              "calibration_id"):
        if not sig.get(k):
            missing.append(f"contact_signal.{k}")
    if sig.get("unit") and sig.get("unit") != "N":
        missing.append("contact_signal.unit==N")
    ctl = desc.get("controller") or {}
    for k in ("kind", "loop_hz", "process"):
        if not ctl.get(k):
            missing.append(f"controller.{k}")
    if ctl.get("kind") in ("position", "guarded_position"):
        missing.append("controller.kind in {force, admittance, impedance, hybrid}")
    if not (desc.get("limits") or {}).get("max_force_N"):
        missing.append("limits.max_force_N")
    return missing


def guarded_seat(
    request: dict,
    contact: ContactBackend | None,
    *,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Execute lego.guarded_seat. Returns the SkillResult wire dict."""
    try:
        res = ids(request, SKILL_ID, VERSION)
    except _Early as e:
        return e.result.finish("rejected", e.exc.reason_code, str(e.exc)).to_dict()
    res.evidence["operation_complete"] = False
    res.evidence["connection_verified"] = False
    res.evidence["control_statement"] = NOT_FORCE_CONTROL
    try:
        p = _params(request)
    except Rejected as exc:
        return res.finish("rejected", exc.reason_code, str(exc)).to_dict()
    res.check("pre", "aligned", True, f"alignment {p['alignment']['invocation_id']}")
    if contact is None:
        res.evidence["missing_capabilities"] = sorted(CONTACT_CAPABILITIES)
        return res.finish("rejected", "missing_capabilities",
                          "no contact backend; " + NOT_FORCE_CONTROL).to_dict()
    desc = contact.describe()
    res.backend = desc
    missing = check_contact_capabilities(desc)
    res.check("pre", "contact_capabilities", not missing, ", ".join(missing))
    if missing:
        res.evidence["missing_capabilities"] = missing
        msg = f"backend lacks {missing}"
        if "bounded_position_approach" in set(desc.get("capabilities", [])):
            msg += "; it offers only bounded_position_approach: " + NOT_FORCE_CONTROL
        return res.finish("rejected", "missing_capabilities", msg).to_dict()
    lim = desc["limits"]
    if p["force_limit_N"] > float(lim["max_force_N"]):
        return res.finish("rejected", "force_limit_unsupported",
                          f"force_limit_N {p['force_limit_N']} > backend max "
                          f"{lim['max_force_N']}").to_dict()
    if lim.get("max_travel_m") and p["max_travel_m"] > float(lim["max_travel_m"]):
        return res.finish("rejected", "travel_unsupported", "max_travel_m beyond backend").to_dict()
    if lim.get("max_speed_m_s") and p["speed_m_s"] > float(lim["max_speed_m_s"]):
        return res.finish("rejected", "speed_unsupported", "speed_m_s beyond backend").to_dict()
    if p["torque_limit_Nm"] is not None and "contact_torque_feedback" not in desc["capabilities"]:
        return res.finish("rejected", "torque_limit_unsupported",
                          "torque_limit_Nm given but the backend measures no torque").to_dict()

    res.phase = "press"
    res.motion_issued = True
    started = clock()
    trace = contact.guarded_move(axis={"frame_id": p["frame_id"], "vector": p["axis"]},
                                 max_travel_m=p["max_travel_m"], speed_m_s=p["speed_m_s"],
                                 force_limit_N=p["force_limit_N"], hold_s=p["hold_s"],
                                 timeout_s=p["timeout_s"])
    if not trace.get("command_issued", True):
        res.motion_issued = False
    res.stop = {"reason": trace.get("stop_reason"), "confirmed": bool(trace.get("stop_confirmed"))}
    res.evidence_level = desc.get("evidence_level", "unit_or_dryrun")
    res.privileged_observation = bool(desc.get("privileged", False))
    res.trace.append({"event": "guarded_move", "t_s": started,
                      "n_samples": len(trace.get("samples", []))})
    res.evidence["contact_trace"] = trace.get("samples", [])
    return _classify(res, p, trace, desc).to_dict()


def _params(req: dict) -> dict[str, Any]:
    al = req.get("alignment")
    if not isinstance(al, dict):
        raise Rejected("not_aligned", "alignment evidence (an align_connection result) required")
    if al.get("skill_id") != "lego.align_connection" or al.get("status") != "succeeded":
        raise Rejected("not_aligned", "alignment evidence is not a succeeded align_connection")
    if al.get("connection_id") != req["connection_id"]:
        raise Rejected("evidence_mismatch", "alignment evidence is for another connection")
    plan = (al.get("evidence") or {}).get("plan") or {}
    sysname = req.get("system")
    if sysname not in SYSTEMS:
        raise Rejected("invalid_input", f"system must be one of {sorted(SYSTEMS)}")
    s = SYSTEMS[sysname]
    axis = _axis(req, plan)
    pre_h = plan.get("preassembly_height_m")
    if not isinstance(pre_h, (int, float)):
        raise Rejected("invalid_input", "alignment evidence lacks preassembly_height_m")
    for k in ("max_travel_m", "speed_m_s", "timeout_s", "force_limit_N", "limits_source"):
        if k not in req:
            raise Rejected("invalid_input", f"{k} is required (no default contact limits)")
    max_travel = number(req["max_travel_m"], "max_travel_m", 1e-4, 0.05)
    if max_travel > pre_h + OVERTRAVEL_FRACTION * s.stud_height_m:
        raise Rejected("travel_too_long",
                       f"max_travel_m {max_travel * 1e3:.2f} mm exceeds the seat "
                       f"({pre_h * 1e3:.2f} mm) by more than half a stud height")
    if max_travel < pre_h:
        raise Rejected("travel_too_short", "max_travel_m does not reach the seat")
    return {
        "alignment": al,
        "system": s,
        "frame_id": req["axis"]["frame_id"],
        "axis": axis,
        "seat_travel_m": float(pre_h),
        "max_travel_m": max_travel,
        "speed_m_s": number(req["speed_m_s"], "speed_m_s", 1e-5, 0.05),
        "timeout_s": number(req["timeout_s"], "timeout_s", 0.1, 60),
        "force_limit_N": number(req["force_limit_N"], "force_limit_N", 0.1, 200),
        "torque_limit_Nm": (number(req["torque_limit_Nm"], "torque_limit_Nm", 0.01, 20)
                            if "torque_limit_Nm" in req else None),
        "hold_s": number(req.get("hold_s", 0.0), "hold_s", 0, 5),
        "limits_source": str(req["limits_source"]),
    }


def _classify(res: SkillResult, p: dict, trace: dict, desc: dict) -> SkillResult:
    s = p["system"]
    samples = trace.get("samples") or []
    stop_reason = trace.get("stop_reason")
    confirmed = bool(trace.get("stop_confirmed"))
    after = ["bounded_retreat_if_state_fresh", "reobserve", "human"]
    if not confirmed:
        return res.finish("failed", "stop_unconfirmed",
                          f"backend stopped for {stop_reason!r} but did not confirm the stop",
                          next_allowed=["human"])
    if not samples:
        if res.motion_issued:
            return res.finish("unverified", "no_contact_record",
                              "motion issued but no contact samples returned",
                              next_allowed=["reobserve", "human"])
        return res.finish("failed", "not_executed", str(trace.get("fault", "")),
                          next_allowed=after)
    # Signal integrity: monotonic time, gaps within the declared latency.
    max_gap = float(desc["contact_signal"]["max_latency_s"])
    ts = [float(x["t_s"]) for x in samples]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    if any(gp <= 0 for gp in gaps) or (gaps and max(gaps) > max_gap) or stop_reason == "signal_lost":
        return res.finish("unverified", "contact_signal_lost",
                          f"contact signal gap {max(gaps, default=0):.3f} s > {max_gap} s or "
                          "non-monotonic; the press outcome is not established",
                          next_allowed=["reobserve", "human"])
    travel = [float(x["travel_m"]) for x in samples]
    force = [float(x["force_N"]) for x in samples]
    peak = max(force)
    final = travel[-1]
    seat = p["seat_travel_m"]
    fl = p["force_limit_N"]
    onset = next((t for t, f in zip(travel, force) if f > CONTACT_FRACTION * fl), None)
    stud_top_travel = seat - s.stud_height_m
    res.evidence.update(
        peak_force_N=peak, final_travel_m=final, seat_travel_m=seat,
        contact_onset_travel_m=onset, force_limit_N=fl, limits_source=p["limits_source"],
        shortfall_m=seat - final,
    )
    res.check("post", "force_within_limit", peak <= FORCE_OVERSHOOT * fl,
              f"peak {peak:.2f} N, limit {fl:.2f} N")
    if peak > FORCE_OVERSHOOT * fl:
        return res.finish("failed", "force_limit_exceeded",
                          f"peak {peak:.2f} N over {FORCE_OVERSHOOT}x the {fl:.2f} N limit",
                          next_allowed=["human"])
    if stop_reason == "fault":
        return res.finish("failed", "backend_fault", str(trace.get("fault", "")),
                          next_allowed=after)
    if onset is not None and onset < stud_top_travel - EARLY_CONTACT_MARGIN_FRACTION * s.stud_height_m:
        return res.finish("failed", "early_contact",
                          f"contact at {onset * 1e3:.2f} mm travel, studs expected from "
                          f"{stud_top_travel * 1e3:.2f} mm: obstacle or wrong seat height",
                          next_allowed=after)
    if stop_reason == "timeout":
        return res.finish("failed", "timeout", "press timed out", next_allowed=after)
    if onset is None:
        return res.finish("failed", "no_contact",
                          f"travelled {final * 1e3:.2f} mm without contact: brick missing, "
                          "slipped in the grasp, or seat lower than planned",
                          next_allowed=["reobserve", "human"])
    shortfall = seat - final
    res.check("post", "reached_seat_depth", abs(shortfall) <= SEAT_FRACTION * s.stud_height_m,
              f"{shortfall * 1e3:+.2f} mm short of the seat")
    if shortfall >= RESTING_FRACTION * s.stud_height_m:
        return res.finish("failed", "blocked_on_studs",
                          f"stopped {shortfall * 1e3:.2f} mm short (stud {s.stud_height_m * 1e3:.1f}"
                          " mm): walls on the stud tops, misaligned", next_allowed=after)
    if shortfall > SEAT_FRACTION * s.stud_height_m:
        return res.finish("failed", "not_pressed_fully",
                          f"stopped {shortfall * 1e3:.2f} mm short at {peak:.2f} N",
                          next_allowed=after)
    if stop_reason != "force_threshold":
        return res.finish("unverified", "seat_depth_without_threshold",
                          f"reached seat depth on {stop_reason!r} without the force rise of a "
                          "press-fit; whether the clutch engaged is not established",
                          next_allowed=["release_and_retreat", "lego.verify_connection"])
    res.evidence["operation_complete"] = True
    return res.finish("succeeded", "press_complete",
                      "bounded press completed at seat depth; connection NOT yet verified",
                      next_allowed=["release_and_retreat", "lego.verify_connection"])
