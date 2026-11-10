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

Success criterion (station-agnostic). After a confirmed stop within the force limit, with the
contact where the stud tops are expected:

1. **Seated pose** - if an independent observation of the pressed brick, taken after the
   press, is usable (fresh, after the press started, same frame, not commanded, its 2-sigma band
   narrower than the seat band): its bottom must lie within ``seat_depth_tolerance_m`` of the
   seat height and its lateral / yaw / tilt error within ``lego.verify_connection``'s bands
   (``verify._thresholds``: the same geometry + 2 sigma); tilt is judged through the bottom-face
   corners (none may be above the seat band), because the hand still holds the brick and can tilt
   it; verify applies the strict tilt band after the release. The observed brick decides; the TCP
   travel does not (a brick can move in the grasp, e.g. a simulator that snaps it to the seat).
2. **Force rise** - required only when the station declares that its force signal shows the
   press-fit (``contact_signal.insertion_force_observable``, default true): then the press must
   also have stopped on ``force_threshold``. A station whose clutch model has no insertion
   travel or force (BrickSim snaps the brick to the seat) declares false, and the pose alone
   decides.
3. Without a usable observation: a force-observable station falls back to the measured travel
   (stop on force_threshold within ``seat_depth_tolerance_m`` of the seat travel); a station
   that cannot show the press-fit force returns ``unverified/seat_unobserved``.

``seat_depth_tolerance_m`` is declared by the request (press limits) or defaults to the
documented ``SEAT_FRACTION`` x stud height; the value and its source are always in the evidence.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from . import geometry as g
from .contracts import Observation, Rejected, SkillResult, _Early, ids, number, usable
from .geometry import SYSTEMS
from .ports import CONTACT_CAPABILITIES, ContactBackend
from .verify import _thresholds as seat_bands

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

POST_PRESS_MAX_AGE_S = 2.0
"""A post-press observation older than this (or taken before the press began) is not used."""


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
        raise Rejected(
            "axis_mismatch",
            f"axis is {math.degrees(math.acos(max(-1, min(1, c)))):.1f} deg off the "
            "connection's engagement axis",
        )
    return v


def check_contact_capabilities(desc: dict) -> list[str]:
    """Missing capabilities / signal declarations (empty when the backend qualifies)."""
    caps = set(desc.get("capabilities", []))
    missing = sorted(CONTACT_CAPABILITIES - caps)
    sig = desc.get("contact_signal") or {}
    for k in (
        "source",
        "quantity",
        "unit",
        "frame_id",
        "rate_hz",
        "max_latency_s",
        "calibration_id",
    ):
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
    observe: Callable[[], dict | None] | None = None,
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
        return res.finish(
            "rejected", "missing_capabilities", "no contact backend; " + NOT_FORCE_CONTROL
        ).to_dict()
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
        return res.finish(
            "rejected",
            "force_limit_unsupported",
            f"force_limit_N {p['force_limit_N']} > backend max {lim['max_force_N']}",
        ).to_dict()
    if lim.get("max_travel_m") and p["max_travel_m"] > float(lim["max_travel_m"]):
        return res.finish("rejected", "travel_unsupported", "max_travel_m beyond backend").to_dict()
    if lim.get("max_speed_m_s") and p["speed_m_s"] > float(lim["max_speed_m_s"]):
        return res.finish("rejected", "speed_unsupported", "speed_m_s beyond backend").to_dict()
    if p["torque_limit_Nm"] is not None and "contact_torque_feedback" not in desc["capabilities"]:
        return res.finish(
            "rejected",
            "torque_limit_unsupported",
            "torque_limit_Nm given but the backend measures no torque",
        ).to_dict()

    res.phase = "press"
    res.motion_issued = True
    started = clock()
    trace = contact.guarded_move(
        axis={"frame_id": p["frame_id"], "vector": p["axis"]},
        max_travel_m=p["max_travel_m"],
        speed_m_s=p["speed_m_s"],
        force_limit_N=p["force_limit_N"],
        hold_s=p["hold_s"],
        timeout_s=p["timeout_s"],
    )
    if not trace.get("command_issued", True):
        res.motion_issued = False
    res.stop = {"reason": trace.get("stop_reason"), "confirmed": bool(trace.get("stop_confirmed"))}
    res.evidence_level = desc.get("evidence_level", "unit_or_dryrun")
    res.privileged_observation = bool(desc.get("privileged", False))
    res.trace.append(
        {"event": "guarded_move", "t_s": started, "n_samples": len(trace.get("samples", []))}
    )
    res.evidence["contact_trace"] = trace.get("samples", [])
    post = None
    if observe is not None and res.motion_issued:
        try:
            post = observe()
        except Exception as exc:  # an observer failure is evidence missing, not a press result
            res.evidence["post_press_observation_error"] = repr(exc)
    p["post_press"] = post
    p["press_started_s"] = started
    p["now_s"] = clock()
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
    if str(plan.get("brick_type", "")).split(":")[0] != sysname:
        raise Rejected(
            "evidence_mismatch",
            f"alignment was for {plan.get('brick_type')!r}, press is for {sysname}",
        )
    axis = _axis(req, plan)
    pre_h = plan.get("preassembly_height_m")
    if not isinstance(pre_h, (int, float)):
        raise Rejected("invalid_input", "alignment evidence lacks preassembly_height_m")
    for k in ("max_travel_m", "speed_m_s", "timeout_s", "force_limit_N", "limits_source"):
        if k not in req:
            raise Rejected("invalid_input", f"{k} is required (no default contact limits)")
    max_travel = number(req["max_travel_m"], "max_travel_m", 1e-4, 0.05)
    if max_travel > pre_h + OVERTRAVEL_FRACTION * s.stud_height_m:
        raise Rejected(
            "travel_too_long",
            f"max_travel_m {max_travel * 1e3:.2f} mm exceeds the seat "
            f"({pre_h * 1e3:.2f} mm) by more than half a stud height",
        )
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
        "torque_limit_Nm": (
            number(req["torque_limit_Nm"], "torque_limit_Nm", 0.01, 20)
            if "torque_limit_Nm" in req
            else None
        ),
        "hold_s": number(req.get("hold_s", 0.0), "hold_s", 0, 5),
        "limits_source": str(req["limits_source"]),
        "seat_depth_tolerance_m": (
            number(
                req["seat_depth_tolerance_m"],
                "seat_depth_tolerance_m",
                1e-5,
                RESTING_FRACTION * s.stud_height_m,
            )
            if "seat_depth_tolerance_m" in req
            else SEAT_FRACTION * s.stud_height_m
        ),
        "seat_depth_tolerance_source": (
            f"request ({req['limits_source']})"
            if "seat_depth_tolerance_m" in req
            else f"lego.guarded_seat default: SEAT_FRACTION {SEAT_FRACTION} x stud height"
        ),
        "plan": plan,
    }


def _classify(res: SkillResult, p: dict, trace: dict, desc: dict) -> SkillResult:
    s = p["system"]
    samples = trace.get("samples") or []
    stop_reason = trace.get("stop_reason")
    confirmed = bool(trace.get("stop_confirmed"))
    after = ["bounded_retreat_if_state_fresh", "reobserve", "human"]
    if not confirmed:
        return res.finish(
            "failed",
            "stop_unconfirmed",
            f"backend stopped for {stop_reason!r} but did not confirm the stop",
            next_allowed=["human"],
        )
    if not samples:
        if res.motion_issued:
            return res.finish(
                "unverified",
                "no_contact_record",
                "motion issued but no contact samples returned",
                next_allowed=["reobserve", "human"],
            )
        return res.finish("failed", "not_executed", str(trace.get("fault", "")), next_allowed=after)
    # Signal integrity: monotonic time, gaps within the declared latency.
    max_gap = float(desc["contact_signal"]["max_latency_s"])
    ts = [float(x["t_s"]) for x in samples]
    gaps = [b - a for a, b in zip(ts, ts[1:], strict=False)]
    if (
        any(gp <= 0 for gp in gaps)
        or (gaps and max(gaps) > max_gap)
        or stop_reason == "signal_lost"
    ):
        return res.finish(
            "unverified",
            "contact_signal_lost",
            f"contact signal gap {max(gaps, default=0):.3f} s > {max_gap} s or "
            "non-monotonic; the press outcome is not established",
            next_allowed=["reobserve", "human"],
        )
    travel = [float(x["travel_m"]) for x in samples]
    force = [float(x["force_N"]) for x in samples]
    peak = max(force)
    final = travel[-1]
    seat = p["seat_travel_m"]
    fl = p["force_limit_N"]
    onset = next(
        (t for t, f in zip(travel, force, strict=False) if f > CONTACT_FRACTION * fl), None
    )
    stud_top_travel = seat - s.stud_height_m
    res.evidence.update(
        peak_force_N=peak,
        final_travel_m=final,
        seat_travel_m=seat,
        contact_onset_travel_m=onset,
        force_limit_N=fl,
        limits_source=p["limits_source"],
        shortfall_m=seat - final,
    )
    res.check(
        "post",
        "force_within_limit",
        peak <= FORCE_OVERSHOOT * fl,
        f"peak {peak:.2f} N, limit {fl:.2f} N",
    )
    if peak > FORCE_OVERSHOOT * fl:
        return res.finish(
            "failed",
            "force_limit_exceeded",
            f"peak {peak:.2f} N over {FORCE_OVERSHOOT}x the {fl:.2f} N limit",
            next_allowed=["human"],
        )
    if stop_reason == "fault":
        return res.finish(
            "failed", "backend_fault", str(trace.get("fault", "")), next_allowed=after
        )
    if (
        onset is not None
        and onset < stud_top_travel - EARLY_CONTACT_MARGIN_FRACTION * s.stud_height_m
    ):
        return res.finish(
            "failed",
            "early_contact",
            f"contact at {onset * 1e3:.2f} mm travel, studs expected from "
            f"{stud_top_travel * 1e3:.2f} mm: obstacle or wrong seat height",
            next_allowed=after,
        )
    if stop_reason == "timeout":
        return res.finish("failed", "timeout", "press timed out", next_allowed=after)
    if onset is None:
        return res.finish(
            "failed",
            "no_contact",
            f"travelled {final * 1e3:.2f} mm without contact: brick missing, "
            "slipped in the grasp, or seat lower than planned",
            next_allowed=["reobserve", "human"],
        )
    shortfall = seat - final
    depth_tol = p["seat_depth_tolerance_m"]
    force_observable = bool(desc["contact_signal"].get("insertion_force_observable", True))
    res.evidence["seat_check"] = {
        "seat_depth_tolerance_m": depth_tol,
        "seat_depth_tolerance_source": p["seat_depth_tolerance_source"],
        "insertion_force_observable": force_observable,
        "tcp_shortfall_m": shortfall,
    }
    seated = _seated_by_observation(res, p, s)
    done = ["release_and_retreat", "lego.verify_connection"]
    if seated is not None:
        ok, d, th = seated
        res.evidence["seat_check"].update(basis="post_press_observation", delta=d, bands=th)
        # The brick may have moved in the grasp (e.g. snapped to the seat): report the gap
        # between the TCP travel and the observed brick depth, but judge the brick.
        res.evidence["seat_check"]["brick_vs_tcp_m"] = shortfall - d["dz_m"]
        if not ok:
            return _off_seat(res, d, th, depth_tol, s, after)
        if force_observable and stop_reason != "force_threshold":
            return res.finish(
                "unverified",
                "seated_without_force_rise",
                f"brick observed seated ({d['dz_m'] * 1e3:+.2f} mm) but the press ended on "
                f"{stop_reason!r}: the station declares a force signal that shows a press-fit "
                "and none was seen",
                next_allowed=["release_and_retreat", "lego.verify_connection"],
            )
        res.evidence["operation_complete"] = True
        basis = "observed seated pose" + (" and force threshold" if force_observable else "")
        return res.finish(
            "succeeded",
            "press_complete",
            f"bounded press completed; {basis}; connection NOT yet verified",
            next_allowed=done,
        )
    res.check(
        "post",
        "reached_seat_depth",
        abs(shortfall) <= depth_tol,
        f"{shortfall * 1e3:+.2f} mm short of the seat (TCP travel; no usable brick observation)",
    )
    if not force_observable:
        res.evidence["seat_check"]["basis"] = "none"
        return res.finish(
            "unverified",
            "seat_unobserved",
            "the station's force signal cannot show a press-fit and no usable post-press "
            "observation of the brick exists: seating is not established",
            next_allowed=["reobserve", "lego.verify_connection"],
        )
    res.evidence["seat_check"]["basis"] = "tcp_travel_and_force"
    if shortfall >= RESTING_FRACTION * s.stud_height_m:
        return res.finish(
            "failed",
            "blocked_on_studs",
            f"stopped {shortfall * 1e3:.2f} mm short (stud {s.stud_height_m * 1e3:.1f}"
            " mm): walls on the stud tops, misaligned",
            next_allowed=after,
        )
    if shortfall > depth_tol:
        return res.finish(
            "failed",
            "not_pressed_fully",
            f"stopped {shortfall * 1e3:.2f} mm short at {peak:.2f} N",
            next_allowed=after,
        )
    if stop_reason != "force_threshold":
        return res.finish(
            "unverified",
            "seat_depth_without_threshold",
            f"reached seat depth on {stop_reason!r} without the force rise of a "
            "press-fit; whether the clutch engaged is not established",
            next_allowed=done,
        )
    res.evidence["operation_complete"] = True
    return res.finish(
        "succeeded",
        "press_complete",
        "bounded press completed at seat depth; connection NOT yet verified",
        next_allowed=done,
    )


def _seated_by_observation(res: SkillResult, p: dict, s: Any) -> tuple[bool, dict, dict] | None:
    """(seated, pose delta, bands) from a usable post-press observation of the brick, else None
    (the reason is recorded)."""
    post = p.get("post_press")
    plan = p["plan"]
    if post is None:
        res.evidence["seat_check_observation"] = "none"
        return None
    try:
        obs = Observation.from_dict(post, "post_press_observation")
        seat_pose = plan["seat_pose"]
        bt = _brick_type(plan, s)
    except (Rejected, KeyError, g.GeometryError) as exc:
        res.evidence["seat_check_observation"] = f"unusable: {exc}"
        return None
    why = usable(
        obs,
        now_s=p["now_s"],
        max_age_s=POST_PRESS_MAX_AGE_S,
        frame_id=seat_pose["frame_id"],
        not_after_s=p["press_started_s"],
    )
    if not why and obs.value is None:
        why = "brick_not_visible"
    if why:
        res.evidence["seat_check_observation"] = f"unusable: {why}"
        return None
    pose = obs.value["pose"] if isinstance(obs.value, dict) and "pose" in obs.value else obs.value
    th = seat_bands(bt, obs.uncertainty_m, obs.uncertainty_deg)
    if th["sigma_band_m"] >= p["seat_depth_tolerance_m"]:
        res.evidence["seat_check_observation"] = (
            "unusable: observation too uncertain for the seat band"
        )
        return None
    res.use(obs)
    res.evidence["seat_check_observation"] = obs.obs_id
    t_seat, t_obs = g.matrix(seat_pose), g.matrix(pose)
    d = g.pose_delta(t_seat, t_obs)
    period = 90.0 if bt.studs[0] == bt.studs[1] else 180.0
    d["yaw_deg"] = (d["yaw_deg"] + period / 2) % period - period / 2
    # Heights of the four bottom-face corners over the seat plane. The brick is still in the hand:
    # a clamp on a brick that is already held by the studs can tilt the observed pose (BrickSim, 15 N
    # grip: 1.5-2.2 deg). What makes it unseated is a corner still above the seat band (riding on the
    # stud tops), so tilt is judged through the corners here; lego.verify_connection applies the
    # strict tilt band after the release.
    rel = g.relative(t_seat, t_obs)
    lx, ly = bt.body_m[0] / 2, bt.body_m[1] / 2
    corners = [rel[2][0] * x + rel[2][1] * y + rel[2][3] for x in (-lx, lx) for y in (-ly, ly)]
    d["corner_max_dz_m"], d["corner_min_dz_m"] = max(corners), min(corners)
    tol = p["seat_depth_tolerance_m"]
    # Only height over the seat means unseated. Deeper than the nominal seat is the support sitting low:
    # BrickSim seats each layer ~0.4 mm under the nominal model, so a second-layer brick read -0.74 mm and
    # was failed as misaligned (pyramid6 / wall7 brick 5-7). Deeper than a stud height is still refused.
    depth_ok = -s.stud_height_m <= d["dz_m"] <= tol
    corners_ok = d["corner_max_dz_m"] <= tol
    lat_ok = d["lateral_m"] <= th["lateral_m"]
    yaw_ok = abs(d["yaw_deg"]) <= th["yaw_deg"]
    res.check("post", "seat_depth_observed", depth_ok, f"{d['dz_m'] * 1e3:+.2f} mm over the seat")
    res.check(
        "post",
        "seat_corners_observed",
        corners_ok,
        f"highest bottom corner {d['corner_max_dz_m'] * 1e3:+.2f} mm over the seat "
        f"(tilt {d['tilt_deg']:.2f} deg, judged after release by verify)",
    )
    res.check("post", "seat_lateral_observed", lat_ok, f"{d['lateral_m'] * 1e3:.2f} mm")
    res.check("post", "seat_yaw_observed", yaw_ok, f"{d['yaw_deg']:+.2f} deg")
    return depth_ok and corners_ok and lat_ok and yaw_ok, d, th


def _brick_type(plan: dict, s: Any) -> g.BrickType:
    """The aligned brick type from the alignment plan (``system:NxM@version``)."""
    tid = str(plan["brick_type"])
    studs = tid.split(":", 1)[1].split("@", 1)[0]
    return g.brick_type(s.name, studs)


def _off_seat(
    res: SkillResult, d: dict, th: dict, depth_tol: float, s: Any, after: list
) -> SkillResult:
    dz = max(d["dz_m"], d.get("corner_max_dz_m", d["dz_m"]))
    if d["lateral_m"] > th["wrong_cell_m"] or abs(d["yaw_deg"]) > th["yaw_deg"]:
        return res.finish(
            "failed",
            "blocked_on_studs"
            if dz >= RESTING_FRACTION * s.stud_height_m
            else "seat_pose_misaligned",
            f"brick observed {d['lateral_m'] * 1e3:.2f} mm / {d['yaw_deg']:+.2f} deg off its seat, "
            f"{dz * 1e3:+.2f} mm over it",
            next_allowed=after,
        )
    if dz >= RESTING_FRACTION * s.stud_height_m:
        return res.finish(
            "failed",
            "blocked_on_studs",
            f"brick observed {dz * 1e3:+.2f} mm over its seat (stud {s.stud_height_m * 1e3:.1f} mm): "
            "on the stud tops",
            next_allowed=after,
        )
    if dz > depth_tol:
        return res.finish(
            "failed",
            "not_pressed_fully",
            f"brick observed {dz * 1e3:+.2f} mm over its seat (tolerance {depth_tol * 1e3:.2f} mm)",
            next_allowed=after,
        )
    return res.finish(
        "failed",
        "seat_pose_misaligned",
        f"brick at seat depth ({dz * 1e3:+.2f} mm) but {d['lateral_m'] * 1e3:.2f} mm / "
        f"{d['yaw_deg']:+.2f} deg / tilt {d['tilt_deg']:.2f} deg off (bands {th['lateral_m'] * 1e3:.2f} mm, "
        f"{th['yaw_deg']:.2f} deg, {th['tilt_deg']:.2f} deg)",
        next_allowed=after,
    )
