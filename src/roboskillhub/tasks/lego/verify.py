"""lego.verify_connection: is the brick on the right studs, yaw and engagement height, and
did it (and the build around it) stay put after release and retreat?

Pure function over observations; issues no motion and applies no pull/wiggle test. It
accepts no observation from the executor that pressed (``executor_id``), nor commanded or
planned values. What it can establish is geometric seating (``geometric_seating_verified``),
never retention force or mechanical interlock.

Verdict -> status: satisfied -> succeeded, violated -> failed, unknown -> unverified;
malformed or unusable requests -> rejected. The height bands are
skills/pressing-a-brick-onto-studs' (seated within 0.35 stud heights, resting on the studs
from 0.6) and the cell/residual inverse is skills/computing-brick-pose-from-stud-grid's.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from . import geometry as g
from .contracts import (
    EVIDENCE_LEVELS,
    Observation,
    Rejected,
    _Early,
    ids,
    level_for,
    number,
    usable,
)

SKILL_ID = "lego.verify_connection"
VERSION = "0.1.0"

SEAT_FRACTION = 0.35
RESTING_FRACTION = 0.6
LIFTED_FRACTION = 1.5
"""Bottom more than this many stud heights over the seat after release: off the studs."""
TILT_FRACTION = SEAT_FRACTION
"""Tilt that lifts one end by this fraction of the stud height over the brick's length:
the seat band, so a brick tilted past it has one end outside the band."""
LATERAL_CLEARANCE_M = 0.0002
"""Lateral play of a seated brick on its studs (2 x the 0.1 mm footprint play)."""
SIGMAS = 2.0
"""Thresholds are widened by this many observation sigmas; an observation whose sigma
band is as wide as the seat band cannot resolve seating (unknown)."""

DEFAULTS = {"window_s": 0.5, "min_samples": 2, "max_obs_age_s": 2.0}


def _thresholds(bt: g.BrickType, sigma_m: float, sigma_deg: float) -> dict[str, float]:
    s = bt.system
    longest = max(bt.body_m[:2])
    return {
        "seat_m": SEAT_FRACTION * s.stud_height_m,
        "resting_m": RESTING_FRACTION * s.stud_height_m,
        "lifted_m": LIFTED_FRACTION * s.stud_height_m,
        "lateral_m": LATERAL_CLEARANCE_M + SIGMAS * sigma_m,
        "wrong_cell_m": 0.25 * s.stud_pitch_m,
        "yaw_deg": math.degrees(math.atan(2 * LATERAL_CLEARANCE_M / longest)) + SIGMAS * sigma_deg,
        "tilt_deg": math.degrees(math.atan(TILT_FRACTION * s.stud_height_m / longest))
        + SIGMAS * sigma_deg,
        "drift_m": LATERAL_CLEARANCE_M + SIGMAS * math.sqrt(2) * sigma_m,
        "sigma_band_m": SIGMAS * sigma_m,
    }


def _pose_of(obs: Observation, frame_id: str) -> tuple[dict | None, list[float] | None]:
    v = obs.value
    if v is None:
        return None, None
    corners = None
    if isinstance(v, dict) and "pose" in v:
        corners = v.get("corner_bottom_heights_m")
        v = v["pose"]
    return g.validate_pose(v, frame_id, f"{obs.obs_id}.value"), corners


def verify_connection(request: dict, *, clock: Callable[[], float] = time.time) -> dict[str, Any]:
    try:
        res = ids(request, SKILL_ID, VERSION)
    except _Early as e:
        return e.result.finish("rejected", e.exc.reason_code, str(e.exc)).to_dict()
    res.phase = "verify"
    res.evidence["claim"] = "none"
    res.evidence["mechanical_interlock_verified"] = False
    try:
        ctx = _parse(request)
    except Rejected as exc:
        return res.finish("rejected", exc.reason_code, str(exc)).to_dict()
    except g.GeometryError as exc:
        return res.finish("rejected", "invalid_geometry", str(exc)).to_dict()
    bt, plate, cell, fid = ctx["bt"], ctx["plate"], ctx["cell"], ctx["plate"].frame_id
    ctx["res_checks"] = lambda name, ok, detail: res.check("post", name, ok, detail)
    now = clock()
    released_at = ctx["released_at_s"]
    verdicts: list[tuple[str, str, str]] = []  # (verdict, reason, detail)

    def unknown(reason: str, detail: str) -> None:
        verdicts.append(("unknown", reason, detail))

    def violated(reason: str, detail: str) -> None:
        verdicts.append(("violated", reason, detail))

    # Release evidence: must be observed, independent, and the retreat done.
    rel = ctx["release"]
    r = usable(rel, now_s=now, max_age_s=1e9, frame_id=rel.frame_id)
    if not r and rel.source == ctx["executor_id"]:
        r = "executor_self_report"
    if r or not (
        isinstance(rel.value, dict) and rel.value.get("gripper_open") and rel.value.get("retreated")
    ):
        unknown("release_unconfirmed", r or "release evidence lacks gripper_open and retreated")
    else:
        res.use(rel)

    # Target brick observations after release.
    good: list[tuple[Observation, dict | None, list[float] | None]] = []
    rejected_obs = []
    for o in ctx["observations"]:
        reason = usable(
            o, now_s=now, max_age_s=ctx["max_obs_age_s"], frame_id=fid, not_after_s=released_at
        )
        if not reason and o.source == ctx["executor_id"]:
            reason = "executor_self_report"
        if reason:
            rejected_obs.append({"obs_id": o.obs_id, "reason": reason})
            continue
        try:
            p, corners = _pose_of(o, fid)
        except g.GeometryError as exc:
            rejected_obs.append({"obs_id": o.obs_id, "reason": f"invalid_geometry: {exc}"})
            continue
        good.append((o, p, corners))
        res.use(o)
    res.evidence["unusable_observations"] = rejected_obs
    if not good:
        reasons = sorted({x["reason"] for x in rejected_obs}) or ["no_observation"]
        unknown("no_usable_observation", ", ".join(reasons))
        return _finish(res, verdicts, ctx, [])
    sigma_m = max(o.uncertainty_m for o, _, _ in good)
    sigma_deg = max(o.uncertainty_deg for o, _, _ in good)
    th = _thresholds(bt, sigma_m, sigma_deg)
    res.evidence["thresholds"] = th
    res.evidence["thresholds_source"] = (
        "geometry of " + bt.type_id + f" + {SIGMAS:g} sigma of the observations"
    )
    if th["sigma_band_m"] >= th["seat_m"]:
        unknown(
            "resolution_insufficient",
            f"{SIGMAS:g} sigma = {th['sigma_band_m'] * 1e3:.2f} mm >= seat band "
            f"{th['seat_m'] * 1e3:.2f} mm: cannot tell seated from not pressed",
        )
        return _finish(res, verdicts, ctx, [o for o, _, _ in good])
    span = good[-1][0].timestamp_s - good[0][0].timestamp_s
    if len(good) < ctx["min_samples"] or span < ctx["window_s"]:
        unknown(
            "window_incomplete",
            f"{len(good)} usable samples over {span:.2f} s; need {ctx['min_samples']} over "
            f"{ctx['window_s']:.2f} s after release",
        )

    t_seat = g.matrix(g.seat_pose(plate, bt, cell))
    period = 90.0 if bt.studs[0] == bt.studs[1] else 180.0
    deltas = []
    for o, p, corners in good:
        if p is None:
            violated("brick_missing", f"{o.obs_id}: brick not found at the seat")
            continue
        d = g.pose_delta(t_seat, g.matrix(p))
        d["yaw_deg"] = g.wrap_deg(d["yaw_deg"], period)
        d["obs_id"] = o.obs_id
        if corners is not None:
            if not (isinstance(corners, list) and len(corners) == 4):
                unknown("invalid_corners", f"{o.obs_id}: corner_bottom_heights_m needs 4 values")
            else:
                d["corner_dz_m"] = [float(c) for c in corners]
        deltas.append(d)
    res.evidence["deltas"] = deltas
    if deltas:
        last = deltas[-1]
        _judge(last, th, ctx, violated)
        first = deltas[0]
        drift = math.dist(
            (first["dx_m"], first["dy_m"], first["dz_m"]),
            (last["dx_m"], last["dy_m"], last["dz_m"]),
        )
        res.check(
            "post",
            "stable_over_window",
            drift <= th["drift_m"],
            f"moved {drift * 1e3:.2f} mm over the window",
        )
        if drift > th["drift_m"]:
            violated("moved_after_release", f"moved {drift * 1e3:.2f} mm during the window")
        pre = ctx["pre_release"]
        if pre is not None and last["dz_m"] > th["seat_m"]:
            try:
                pp, _ = _pose_of(pre, fid)
            except g.GeometryError:
                pp = None
            if (
                pp is not None
                and pre.source != ctx["executor_id"]
                and pre.kind in ("measured", "sim_truth")
            ):
                dpre = g.pose_delta(t_seat, g.matrix(pp))
                if abs(dpre["dz_m"]) <= th["seat_m"]:
                    violated(
                        "lifted_on_release",
                        f"seated before release ({dpre['dz_m'] * 1e3:+.2f} mm), "
                        f"{last['dz_m'] * 1e3:+.2f} mm after: pulled up by the jaws",
                    )
                    res.use(pre)

    # The rest of the build: every expected neighbour must still be where it was.
    _neighbours(res, ctx, now, th, violated, unknown)
    return _finish(res, verdicts, ctx, [o for o, _, _ in good])


def _judge(d: dict, th: dict, ctx: dict, violated: Callable[[str, str], None]) -> None:
    res_checks = ctx["res_checks"]
    lat_ok = d["lateral_m"] <= th["lateral_m"]
    res_checks(
        "lateral", lat_ok, f"{d['lateral_m'] * 1e3:.3f} mm (tol {th['lateral_m'] * 1e3:.3f})"
    )
    if d["lateral_m"] > th["wrong_cell_m"]:
        violated("wrong_cell", f"{d['lateral_m'] * 1e3:.2f} mm off: a different stud cell")
    elif not lat_ok:
        violated(
            "lateral_offset",
            f"{d['dx_m'] * 1e3:+.2f} / {d['dy_m'] * 1e3:+.2f} mm off the stud grid",
        )
    yaw_ok = abs(d["yaw_deg"]) <= th["yaw_deg"]
    res_checks("yaw", yaw_ok, f"{d['yaw_deg']:+.2f} deg (tol {th['yaw_deg']:.2f})")
    if not yaw_ok:
        violated("angle_error", f"yawed {d['yaw_deg']:+.2f} deg off the connection")
    dz = d["dz_m"]
    corners = d.get("corner_dz_m")
    if corners is not None:
        low = [c for c in corners if abs(c) <= th["seat_m"]]
        high = [c for c in corners if c > th["seat_m"]]
        res_checks("corners", not high, f"corner heights {[round(c * 1e3, 2) for c in corners]} mm")
        if low and high:
            violated(
                "partial_engagement",
                f"{len(low)}/4 corners seated, {len(high)} up to {max(high) * 1e3:.2f} mm high",
            )
            return
    tilt_ok = d["tilt_deg"] <= th["tilt_deg"]
    res_checks("tilt", tilt_ok, f"{d['tilt_deg']:.2f} deg (tol {th['tilt_deg']:.2f})")
    if not tilt_ok and corners is None:
        violated(
            "partial_engagement" if abs(dz) <= th["resting_m"] else "tilted",
            f"tilted {d['tilt_deg']:.2f} deg: engaged at one end only",
        )
    h_ok = abs(dz) <= th["seat_m"]
    res_checks(
        "height", h_ok, f"bottom {dz * 1e3:+.2f} mm from the seat (tol {th['seat_m'] * 1e3:.2f})"
    )
    if dz > th["lifted_m"]:
        violated("off_studs", f"bottom {dz * 1e3:.2f} mm over the seat: not on the studs")
    elif dz >= th["resting_m"]:
        violated("resting_on_studs", f"bottom {dz * 1e3:.2f} mm high: resting on the stud tops")
    elif dz > th["seat_m"]:
        violated("not_pressed_fully", f"bottom {dz * 1e3:.2f} mm high: partly pressed")
    elif dz < -th["seat_m"]:
        violated("below_seat", f"bottom {dz * 1e3:.2f} mm under the seat: wrong layer or support")


def _neighbours(res, ctx, now, th, violated, unknown) -> None:
    fid = ctx["plate"].frame_id
    seen: dict[str, Observation] = {}
    for o in ctx["neighbours"]:
        bid = (o.value or {}).get("brick_id") if isinstance(o.value, dict) else None
        reason = usable(
            o,
            now_s=now,
            max_age_s=ctx["max_obs_age_s"],
            frame_id=fid,
            not_after_s=ctx["released_at_s"],
        )
        if not reason and o.source == ctx["executor_id"]:
            reason = "executor_self_report"
        if reason or not bid:
            continue
        seen[bid] = o
    report = []
    for pb in ctx["build"]:
        o = seen.get(pb.brick_id)
        if o is None:
            if pb.brick_id in ctx["must_observe"]:
                unknown("neighbour_unobserved", f"{pb.brick_id} not observed after release")
            report.append({"brick_id": pb.brick_id, "observed": False})
            continue
        res.use(o)
        if o.value.get("pose") is None:
            violated("build_disturbed", f"{pb.brick_id} is gone from its place")
            report.append({"brick_id": pb.brick_id, "observed": True, "present": False})
            continue
        p = g.validate_pose(o.value["pose"], fid, f"{o.obs_id}.pose")
        d = g.pose_delta(g.matrix(g.seat_pose(ctx["plate"], pb.type, pb.cell)), g.matrix(p))
        moved = (
            d["lateral_m"] > th["lateral_m"]
            or abs(d["dz_m"]) > th["seat_m"]
            or d["tilt_deg"] > th["tilt_deg"]
        )
        report.append(
            {
                "brick_id": pb.brick_id,
                "observed": True,
                "lateral_m": d["lateral_m"],
                "dz_m": d["dz_m"],
                "tilt_deg": d["tilt_deg"],
                "disturbed": moved,
            }
        )
        if moved:
            violated(
                "build_disturbed",
                f"{pb.brick_id} moved {d['lateral_m'] * 1e3:.2f} mm / "
                f"{d['dz_m'] * 1e3:+.2f} mm / {d['tilt_deg']:.2f} deg",
            )
    res.evidence["neighbours"] = report


def _parse(req: dict) -> dict[str, Any]:
    from .align import _parse as parse_geometry  # same plate/brick/cell/build parsing

    for k in ("executor_id", "release", "observations", "released_at_s"):
        if k not in req:
            raise Rejected("invalid_input", f"{k} is required")
    if not isinstance(req["executor_id"], str) or not req["executor_id"]:
        raise Rejected("invalid_input", "executor_id must name the backend that pressed")
    for k in ("brick", "plate", "target_cell"):
        if k not in req:
            raise Rejected("invalid_input", f"{k} is required")
    base = {k: req[k] for k in ("brick", "plate", "target_cell")}
    base.update(
        held_brick_id=req.get("brick_id", ""),
        reference_id=req.get("plate", {}).get("plate_id", "baseplate"),
        build=req.get("build", []),
    )
    if not base["held_brick_id"]:
        raise Rejected("invalid_input", "brick_id is required")
    parsed = _geometry_only(base, parse_geometry)
    obs = req["observations"]
    if not isinstance(obs, list):
        raise Rejected("invalid_input", "observations must be a list")
    win = {**DEFAULTS, **(req.get("window") or {})}
    out = {
        **parsed,
        "executor_id": req["executor_id"],
        "released_at_s": number(req["released_at_s"], "released_at_s", 0, 1e12),
        "release": Observation.from_dict(req["release"], "release"),
        "observations": sorted(
            (Observation.from_dict(o, f"observations[{i}]") for i, o in enumerate(obs)),
            key=lambda o: o.timestamp_s,
        ),
        "pre_release": (
            Observation.from_dict(req["pre_release_observation"], "pre_release_observation")
            if req.get("pre_release_observation")
            else None
        ),
        "neighbours": [
            Observation.from_dict(o, f"neighbours[{i}]")
            for i, o in enumerate(req.get("neighbours", []))
        ],
        "must_observe": set(req.get("must_observe_neighbours", [])),
        "window_s": number(win["window_s"], "window_s", 0, 60),
        "min_samples": int(number(win["min_samples"], "min_samples", 1, 100)),
        "max_obs_age_s": number(win["max_obs_age_s"], "max_obs_age_s", 0.01, 60),
        "evidence_level": req.get("observer_evidence_level", "unit_or_dryrun"),
    }
    if out["evidence_level"] not in EVIDENCE_LEVELS:
        raise Rejected("invalid_input", f"observer_evidence_level must be one of {EVIDENCE_LEVELS}")
    return out


def _geometry_only(base: dict, parse_geometry) -> dict[str, Any]:
    """Run align's geometric parsing without its held-state requirements."""
    dummy_held = {
        "object_id": base["held_brick_id"],
        "gripper_id": "-",
        "grasp_id": "-",
        "T_TCP_object": {
            "frame_id": "tcp",
            "position_m": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "holding_evidence": {
            "obs_id": "-",
            "source": "-",
            "kind": "measured",
            "frame_id": "-",
            "timestamp_s": 0,
            "value": True,
            "uncertainty_m": 0,
        },
    }
    parsed = parse_geometry({**base, "held": dummy_held})
    parsed.pop("held")
    return parsed


def _finish(res, verdicts, ctx, used: list[Observation]) -> dict[str, Any]:
    res.evidence_level = level_for(used, ctx.get("evidence_level", "unit_or_dryrun"))
    viol = sorted(
        (v for v in verdicts if v[0] == "violated"),
        key=lambda v: v[1] not in ("lifted_on_release", "brick_missing"),
    )
    unk = [v for v in verdicts if v[0] == "unknown"]
    res.evidence["findings"] = [{"verdict": v, "reason": r, "detail": d} for v, r, d in verdicts]
    if viol:
        res.evidence["verdict"] = "violated"
        return res.finish(
            "failed",
            viol[0][1],
            "; ".join(d for _, _, d in viol),
            next_allowed=["report_connection_failure", "human"],
        ).to_dict()
    if unk:
        res.evidence["verdict"] = "unknown"
        return res.finish(
            "unverified",
            unk[0][1],
            "; ".join(d for _, _, d in unk),
            next_allowed=["reobserve", "human"],
        ).to_dict()
    res.evidence["verdict"] = "satisfied"
    res.evidence["claim"] = "geometric_seating_verified"
    return res.finish(
        "succeeded",
        "geometric_seating_verified",
        "pose, height and stability within thresholds after release; retention "
        "force and mechanical interlock NOT tested",
        next_allowed=["next_brick"],
    ).to_dict()
