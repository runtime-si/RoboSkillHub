"""lego.align_connection: non-contact pre-alignment of a held brick over its target studs.

Planning (``plan_alignment``) is pure. Execution (``align_connection``) moves the hand to a
pre-assembly hover, then corrects from fresh independent observations of the held brick,
within a correction count, a cumulative correction budget and one deadline. A computed
pose is a plan, not an alignment: success needs an observation, taken after the last move,
showing the brick within tolerance.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from . import geometry as g
from .contracts import (
    HeldState,
    Observation,
    Rejected,
    SkillResult,
    _Early,
    ids,
    level_for,
    number,
    pose,
    usable,
)
from .ports import BrickObserver, MotionPort

SKILL_ID = "lego.align_connection"
VERSION = "0.1.0"

CAPTURE_EVIDENCE_M = {
    "duplo": (
        0.001,
        "pressing-a-brick-onto-studs, YAM twin MuJoCo: 2/2 seated at <=1 mm, "
        "refused 5/5 at 2-4 mm (kinematic contact, no clutch)",
    ),
    "lego": (0.0005, "DUPLO bound scaled by pitch (8/16 mm); NOT validated for LEGO"),
}
"""Largest lateral alignment tolerance accepted per system: lateral error the straight
press has been seen to capture. A request asking for a looser tolerance is rejected."""

UNCERTAINTY_SIGMAS = 2.0
"""An observation resolves a tolerance only if this many sigmas fit inside it."""

TILT_LIMIT_FRACTION = 0.5
"""Held-brick tilt limit: the angle that lifts one end of the brick by this fraction of
the stud height over its longest side (a tilted brick cannot be fixed by an xy move)."""

DEFAULTS = {
    "preassembly_height_m": 0.01,
    "max_corrections": 3,
    "max_total_correction_m": 0.006,
    "speed_m_s": 0.02,
    "deadline_s": 30.0,
    "max_obs_age_s": 0.5,
    "yaw_ambiguity_deg": 10.0,
    "support_required_fraction": 0.25,
}


def _limits(req: dict, system: str) -> dict[str, Any]:
    lim = req.get("limits")
    if not isinstance(lim, dict):
        raise Rejected("invalid_input", "limits is required (tolerances come from config)")
    for k in (
        "xy_tolerance_m",
        "yaw_tolerance_deg",
        "workspace_min_m",
        "workspace_max_m",
        "limits_source",
    ):
        if k not in lim:
            raise Rejected("invalid_input", f"limits.{k} is required; no default tolerance")
    out = {**DEFAULTS, **{k: v for k, v in lim.items() if k in DEFAULTS}}
    unknown = (
        set(lim)
        - set(DEFAULTS)
        - {
            "xy_tolerance_m",
            "yaw_tolerance_deg",
            "workspace_min_m",
            "workspace_max_m",
            "limits_source",
        }
    )
    if unknown:
        raise Rejected("invalid_input", f"unknown limits {sorted(unknown)}")
    out["xy_tolerance_m"] = number(lim["xy_tolerance_m"], "xy_tolerance_m", 1e-5, 0.01)
    out["yaw_tolerance_deg"] = number(lim["yaw_tolerance_deg"], "yaw_tolerance_deg", 0.01, 10)
    out["preassembly_height_m"] = number(
        out["preassembly_height_m"], "preassembly_height_m", 0.002, 0.1
    )
    out["max_corrections"] = int(number(out["max_corrections"], "max_corrections", 0, 20))
    out["max_total_correction_m"] = number(
        out["max_total_correction_m"], "max_total_correction_m", 0, 0.05
    )
    out["speed_m_s"] = number(out["speed_m_s"], "speed_m_s", 1e-4, 0.25)
    out["deadline_s"] = number(out["deadline_s"], "deadline_s", 0.1, 300)
    out["max_obs_age_s"] = number(out["max_obs_age_s"], "max_obs_age_s", 0.001, 10)
    out["limits_source"] = str(lim["limits_source"])
    for k in ("workspace_min_m", "workspace_max_m"):
        v = lim[k]
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise Rejected("invalid_input", f"limits.{k} needs 3 numbers")
        out[k] = [number(x, k, -100, 100) for x in v]
    cap, why = CAPTURE_EVIDENCE_M[system]
    if out["xy_tolerance_m"] > cap:
        raise Rejected(
            "tolerance_exceeds_capture",
            f"xy_tolerance_m {out['xy_tolerance_m'] * 1e3:.2f} mm is looser than the "
            f"{cap * 1e3:.2f} mm a press is known to capture for {system} ({why})",
        )
    return out


def _parse(req: dict) -> dict[str, Any]:
    """All geometric preconditions; raises Rejected. No motion, no ports."""
    for k in ("held_brick_id", "reference_id", "brick", "plate", "target_cell", "held"):
        if k not in req:
            raise Rejected("invalid_input", f"{k} is required")
    b = req["brick"]
    if not isinstance(b, dict):
        raise Rejected("invalid_input", "brick must be {system, studs, version?}")
    try:
        bt = g.brick_type(b.get("system"), b.get("studs"))
        if b.get("version", bt.system.version) != bt.system.version:
            raise g.GeometryError(
                f"brick geometry version {b.get('version')!r} is not {bt.system.version!r}"
            )
        p = req["plate"]
        if not isinstance(p, dict):
            raise g.GeometryError("plate must be {pose, studs, system, calibration_id}")
        plate_sys = g.SYSTEMS.get(p.get("system"))
        if plate_sys is None:
            raise g.GeometryError(f"unknown plate system {p.get('system')!r}")
        if not p.get("calibration_id"):
            raise Rejected("uncalibrated", "plate.calibration_id is required (calibrated plate)")
        studs = p.get("studs")
        if not (
            isinstance(studs, (list, tuple))
            and len(studs) == 2
            and all(isinstance(v, int) and v > 0 for v in studs)
        ):
            raise g.GeometryError("plate.studs must be two positive integers")
        bias = tuple(float(v) for v in p.get("bias_m", (0.0, 0.0)))
        plate = g.Plate(
            g.validate_pose(p["pose"], None, "plate.pose"),
            (studs[0], studs[1]),
            plate_sys,
            str(p["calibration_id"]),
            (bias[0], bias[1]),
        )
        cell = g.Cell.from_dict(req["target_cell"])
        build = []
        for _i, item in enumerate(req.get("build", [])):
            build.append(
                g.PlacedBrick(
                    str(item["brick_id"]),
                    g.brick_type(item["system"], item["studs"]),
                    g.Cell.from_dict(item["cell"]),
                )
            )
    except g.GeometryError as exc:
        raise Rejected("invalid_geometry", str(exc)) from exc
    except (KeyError, TypeError) as exc:
        raise Rejected("invalid_input", f"malformed build entry: {exc}") from exc
    held = HeldState.from_dict(req["held"])
    if held.object_id != req["held_brick_id"]:
        raise Rejected(
            "identity_mismatch", f"held object {held.object_id!r} is not {req['held_brick_id']!r}"
        )
    if any(pb.brick_id == req["held_brick_id"] for pb in build):
        raise Rejected("identity_mismatch", "the held brick is listed as already placed")
    ref = req["reference_id"]
    plate_id = req["plate"].get("plate_id", "baseplate")
    if ref != plate_id and ref not in {pb.brick_id for pb in build}:
        raise Rejected("unknown_reference", f"reference {ref!r} is neither the plate nor placed")
    close_axis = req["held"].get("close_axis", 0)
    if close_axis not in (0, 1):
        raise Rejected("invalid_input", "held.close_axis is 0 (brick x) or 1 (brick y)")
    return {
        "bt": bt,
        "plate": plate,
        "cell": cell,
        "build": build,
        "held": held,
        "close_axis": close_axis,
        "reference_id": ref,
        "plate_id": plate_id,
    }


def plan_alignment(request: dict, *, tcp_pose: dict | None = None) -> dict[str, Any]:
    """Pure planning: checks, target seat, hover brick pose and hover TCP pose.

    Returns ``{ok, checks, reason_code?, message?, ...poses}``. This is a planning result:
    it says nothing about where the brick physically is."""
    try:
        parsed = _parse(request)
        lim = _limits(request, parsed["bt"].system.name)
    except Rejected as exc:
        return {
            "kind": "plan",
            "ok": False,
            "reason_code": exc.reason_code,
            "message": str(exc),
            "checks": [],
        }
    return _plan(parsed, lim, request, tcp_pose)


def _plan(parsed: dict, lim: dict, request: dict, tcp_pose: dict | None) -> dict[str, Any]:
    bt, plate, cell, build = parsed["bt"], parsed["plate"], parsed["cell"], parsed["build"]
    checks = g.connection_checks(
        plate, bt, cell, build, parsed["close_axis"], lim["support_required_fraction"]
    )
    out: dict[str, Any] = {
        "kind": "plan",
        "ok": True,
        "checks": checks,
        "brick_type": bt.type_id,
        "cell": cell.as_dict(),
    }
    bad = next((c for c in checks if not c["ok"]), None)
    if bad:
        code = {
            "interface_match": "interface_mismatch",
            "on_plate": "out_of_plate",
            "collision_free": "target_occupied",
            "supported": "insufficient_support",
            "approach_clear_of_build": "would_damage_build",
        }[bad["name"]]
        return {**out, "ok": False, "reason_code": code, "message": bad["detail"]}
    t_seat = g.matrix(g.seat_pose(plate, bt, cell))
    t_tcp_obj = g.matrix(parsed["held"].t_tcp_object)
    extra = 0
    if tcp_pose is not None:
        try:
            extra, cands = g.choose_yaw(
                bt, cell, t_seat, g.matrix(tcp_pose), t_tcp_obj, lim["yaw_ambiguity_deg"]
            )
        except g.GeometryError as exc:
            return {**out, "ok": False, "reason_code": "ambiguous_orientation", "message": str(exc)}
        out["orientation_candidates"] = cands
    t_seat = g.mul(t_seat, g.rz(extra))
    up = g.column(g.matrix(plate.pose), 2)
    h = lim["preassembly_height_m"]
    t_hover = g.mul(g.tr(up[0] * h, up[1] * h, up[2] * h), t_seat)
    t_tcp_hover = g.tcp_target(t_hover, t_tcp_obj)
    p = [t_tcp_hover[i][3] for i in range(3)]
    if not all(
        lo <= v <= hi
        for lo, v, hi in zip(lim["workspace_min_m"], p, lim["workspace_max_m"], strict=True)
    ):
        return {
            **out,
            "ok": False,
            "reason_code": "unreachable",
            "message": f"hover TCP {p} outside the configured workspace",
        }
    if h <= bt.system.stud_height_m:
        return {
            **out,
            "ok": False,
            "reason_code": "invalid_input",
            "message": "preassembly_height_m must clear the stud height (non-contact)",
        }
    fid = plate.frame_id
    out.update(
        extra_yaw_deg=extra,
        seat_pose=g.to_pose(t_seat, fid),
        hover_brick_pose=g.to_pose(t_hover, fid),
        hover_tcp_pose=g.to_pose(t_tcp_hover, fid),
        engagement_axis={"frame_id": fid, "vector": [-up[0], -up[1], -up[2]]},
        preassembly_height_m=h,
    )
    return out


def align_connection(
    request: dict,
    motion: MotionPort | None,
    observer: BrickObserver | None,
    *,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Execute lego.align_connection. Returns the SkillResult wire dict."""
    t0 = clock()
    try:
        res = ids(request, SKILL_ID, VERSION)
    except _Early as e:
        return e.result.finish("rejected", e.exc.reason_code, str(e.exc)).to_dict()
    try:
        parsed = _parse(request)
        lim = _limits(request, parsed["bt"].system.name)
    except Rejected as exc:
        return res.finish("rejected", exc.reason_code, str(exc)).to_dict()
    deadline = t0 + lim["deadline_s"]
    res.evidence["limits_source"] = lim["limits_source"]
    res.evidence["plate_calibration_id"] = parsed["plate"].calibration_id
    missing = [n for n, p in (("motion", motion), ("brick_observer", observer)) if p is None]
    if missing:
        res.evidence["missing_capabilities"] = missing
        return res.finish(
            "rejected",
            "missing_capabilities",
            f"no {missing}: alignment cannot be executed or confirmed",
        ).to_dict()
    res.backend = {"motion": motion.describe(), "observer": observer.describe()}
    fid = parsed["plate"].frame_id

    held = parsed["held"]
    reason = usable(
        held.holding_evidence,
        now_s=clock(),
        max_age_s=lim["max_obs_age_s"],
        frame_id=held.holding_evidence.frame_id,
    )
    res.check("pre", "grasp_confirmed", not reason, reason or held.holding_evidence.source)
    if reason:
        return res.finish(
            "rejected",
            "grasp_unconfirmed",
            f"holding evidence: {reason}",
            next_allowed=["reobserve_grasp"],
        ).to_dict()
    res.use(held.holding_evidence)
    tcp = request.get("tcp_observation")
    if tcp is None:
        return res.finish(
            "rejected", "missing_observation", "tcp_observation is required"
        ).to_dict()
    try:
        tcp_obs = Observation.from_dict(tcp, "tcp_observation")
        tcp_pose = pose(tcp_obs.value, fid, "tcp_observation.value")
    except Rejected as exc:
        return res.finish("rejected", exc.reason_code, str(exc)).to_dict()
    reason = usable(tcp_obs, now_s=clock(), max_age_s=lim["max_obs_age_s"], frame_id=fid)
    if reason:
        return res.finish("rejected", reason, "tcp_observation not usable").to_dict()
    res.use(tcp_obs)
    plan = _plan(parsed, lim, request, tcp_pose)
    for c in plan["checks"]:
        res.check("pre", c["name"], c["ok"], c["detail"])
    if not plan["ok"]:
        return res.finish(
            "rejected", plan["reason_code"], plan["message"], next_allowed=["replan"]
        ).to_dict()
    res.evidence["plan"] = {
        k: plan[k]
        for k in (
            "brick_type",
            "cell",
            "seat_pose",
            "hover_brick_pose",
            "hover_tcp_pose",
            "engagement_axis",
            "extra_yaw_deg",
            "preassembly_height_m",
        )
    }
    res.evidence["orientation_candidates"] = plan.get("orientation_candidates", [])
    sigma = observer.describe().get("uncertainty_m")
    if isinstance(sigma, (int, float)) and UNCERTAINTY_SIGMAS * sigma > lim["xy_tolerance_m"]:
        return res.finish(
            "rejected",
            "resolution_insufficient",
            f"observer sigma {sigma * 1e3:.2f} mm cannot resolve "
            f"{lim['xy_tolerance_m'] * 1e3:.2f} mm at {UNCERTAINTY_SIGMAS:g} sigma",
            next_allowed=["use_finer_observer"],
        ).to_dict()

    # ---- execution
    res.phase = "approach"
    t_cmd = g.matrix(plan["hover_tcp_pose"])
    t_goal = g.matrix(plan["hover_brick_pose"])
    bt = parsed["bt"]
    period = 90.0 if bt.studs[0] == bt.studs[1] else 180.0
    longest = max(bt.body_m[:2])
    tilt_limit = math.degrees(math.atan(TILT_LIMIT_FRACTION * bt.system.stud_height_m / longest))
    corrections, total = 0, 0.0
    observations: list[Observation] = [held.holding_evidence, tcp_obs]

    def remaining() -> float:
        return deadline - clock()

    while True:
        if remaining() <= 0:
            return _done(
                res,
                observations,
                observer,
                "failed",
                "deadline_exceeded",
                "alignment deadline spent",
                corrections,
                total,
            )
        res.motion_issued = True
        move_at = clock()
        reply = motion.move_pose(
            g.to_pose(t_cmd, fid), speed_m_s=lim["speed_m_s"], timeout_s=max(0.01, remaining())
        )
        res.trace.append(
            {
                "event": "move_pose",
                "t_s": move_at,
                "ok": bool(reply.get("ok")),
                "target": g.to_pose(t_cmd, fid),
            }
        )
        if remaining() <= 0:
            stop = motion.stop()
            res.stop = {"requested": True, "confirmed": bool(stop.get("confirmed"))}
            return _done(
                res,
                observations,
                observer,
                "failed",
                "deadline_exceeded",
                "move returned after the alignment deadline",
                corrections,
                total,
                next_allowed=["reobserve", "human"],
            )
        if not reply.get("ok"):
            stop = motion.stop()
            res.stop = {"requested": True, "confirmed": bool(stop.get("confirmed"))}
            return _done(
                res,
                observations,
                observer,
                "failed",
                "motion_failed",
                str(reply.get("error", "move_pose refused or failed")),
                corrections,
                total,
                next_allowed=["reobserve"],
            )
        res.phase = "observe"
        try:
            obs = Observation.from_dict(
                observer.observe(
                    held.object_id, not_before_s=clock(), timeout_s=max(0.01, remaining())
                ),
                "brick_observation",
            )
        except Rejected as exc:
            return _done(
                res,
                observations,
                observer,
                "unverified",
                exc.reason_code,
                str(exc),
                corrections,
                total,
                next_allowed=["reobserve"],
            )
        observations.append(obs)
        reason = usable(
            obs, now_s=clock(), max_age_s=lim["max_obs_age_s"], frame_id=fid, not_after_s=move_at
        )
        if not reason and obs.value is None:
            reason = "brick_not_visible"
        if not reason and UNCERTAINTY_SIGMAS * obs.uncertainty_m > lim["xy_tolerance_m"]:
            reason = "resolution_insufficient"
        if reason:
            res.trace.append({"event": "observation_unusable", "reason": reason, "obs": obs.ref()})
            return _done(
                res,
                observations,
                observer,
                "unverified",
                reason,
                "alignment cannot be confirmed from this observation",
                corrections,
                total,
                next_allowed=["reobserve", "human"],
            )
        res.use(obs)
        try:
            t_obs = g.matrix(pose(obs.value, fid, "brick_observation.value"))
        except Rejected as exc:
            return _done(
                res,
                observations,
                observer,
                "unverified",
                exc.reason_code,
                str(exc),
                corrections,
                total,
                next_allowed=["reobserve"],
            )
        d = g.pose_delta(t_goal, t_obs)
        d["yaw_deg"] = g.wrap_deg(d["yaw_deg"], period)
        res.trace.append({"event": "alignment_error", "obs_id": obs.obs_id, **d})
        res.evidence["observed_error"] = d
        if d["tilt_deg"] > tilt_limit:
            return _done(
                res,
                observations,
                observer,
                "failed",
                "held_brick_tilted",
                f"held brick tilted {d['tilt_deg']:.2f} deg (limit {tilt_limit:.2f}): "
                "grasp slipped; an xy correction cannot fix it",
                corrections,
                total,
                next_allowed=["regrasp"],
            )
        ok_xy = d["lateral_m"] <= lim["xy_tolerance_m"]
        ok_yaw = abs(d["yaw_deg"]) <= lim["yaw_tolerance_deg"]
        if ok_xy and ok_yaw:
            res.check(
                "post",
                "lateral_within_tolerance",
                True,
                f"{d['lateral_m'] * 1e3:.3f} <= {lim['xy_tolerance_m'] * 1e3:.3f} mm",
            )
            res.check(
                "post",
                "yaw_within_tolerance",
                True,
                f"{d['yaw_deg']:.3f} <= {lim['yaw_tolerance_deg']:.3f} deg",
            )
            res.evidence["commanded_tcp_pose"] = g.to_pose(t_cmd, fid)
            return _done(
                res,
                observations,
                observer,
                "succeeded",
                "aligned",
                "fresh independent observation within tolerance",
                corrections,
                total,
                next_allowed=["lego.guarded_seat"],
            )
        if corrections >= lim["max_corrections"]:
            return _done(
                res,
                observations,
                observer,
                "failed",
                "alignment_not_converged",
                f"{corrections} corrections, error {d['lateral_m'] * 1e3:.2f} mm / "
                f"{d['yaw_deg']:.2f} deg",
                corrections,
                total,
                next_allowed=["reobserve", "replan"],
            )
        step = d["lateral_m"]
        if total + step > lim["max_total_correction_m"]:
            return _done(
                res,
                observations,
                observer,
                "failed",
                "correction_budget_exhausted",
                f"next correction {step * 1e3:.2f} mm would exceed the "
                f"{lim['max_total_correction_m'] * 1e3:.2f} mm budget",
                corrections,
                total,
                next_allowed=["regrasp", "replan"],
            )
        # X @ O = goal in the plane: X = G Rz(-yaw) T(-dx,-dy,0) G^-1; the rigidly held brick
        # moves with the hand, so the hand gets the same correction.
        x = g.mul(
            g.mul(g.mul(t_goal, g.rz(-d["yaw_deg"])), g.tr(-d["dx_m"], -d["dy_m"], 0.0)),
            g.inv(t_goal),
        )
        t_cmd = g.mul(x, t_cmd)
        corrections += 1
        total += step
        res.phase = "correct"


def _done(
    res: SkillResult,
    observations: list[Observation],
    observer: BrickObserver,
    status: str,
    reason: str,
    message: str,
    corrections: int,
    total: float,
    next_allowed: list[str] | None = None,
) -> dict[str, Any]:
    res.retries = {"corrections": corrections}
    res.evidence["total_correction_m"] = total
    res.evidence["alignment_is_physical"] = status == "succeeded"
    res.evidence_level = level_for(
        observations, observer.describe().get("evidence_level", "unit_or_dryrun")
    )
    return res.finish(status, reason, message, next_allowed=next_allowed).to_dict()
