"""verify_stable_placement: an independent judgement from observations alone.

It never reads what the executor said it did. Inputs are rsh-task/1 ``Observation``
records (``object_id, frame_id, position_m, quaternion_xyzw | null, timestamp_s, source,
privileged, confidence, visibility, state_label``), the measured gripper states and TCP
poses, the release / retreat times, a baseline of the neighbours observed before the
action, and thresholds from a versioned config. It decides:

1. the target is the right tile, within tolerance of its target pose, observed after
   the release and after the retreat;
2. its tilt (object up axis against the support normal, not world z or centre height)
   stays within tolerance; yaw folds by the tile's declared symmetry;
3. it rests on the support (lowest corner on the plane) and the gripper let go
   (holding false, or the measured TCP clear above the tile); otherwise unknown;
4. distinct timestamps spanning the settle window agree (one frame repeated is one);
5. every neighbour in the baseline is still where it was and still upright;
6. stale, future, wrong-frame, unidentified, occluded or orientation-less records are
   never read as zero error. A record without orientation can still show a fall
   (``state_label`` lying / toppled, e.g. from the placing skill's depth check).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import geometry as geo
from .contract import (
    RequestRejected,
    check_geometry,
    check_pose,
    check_support,
    envelope,
    finite,
    set_failure,
    thresholds_for,
)

SKILL_ID = "domino.verify_stable_placement"
VISIBLE = ("visible", "partially_occluded")
PRIVILEGED_SOURCES = ("simulation_truth", "sim_truth")
FALLEN_LABELS = ("lying", "toppled")
FAILURE_PRIORITY = (
    "neighbor_disturbed",
    "unstable_after_release",
    "release_failed",
    "not_resting_on_support",
    "placement_error",
)


def tier_of(rec: dict) -> str:
    src = str(rec.get("source") or "")
    if src.startswith("synthetic"):
        return "synthetic_contract_test"
    if rec.get("privileged") or src in PRIVILEGED_SOURCES:
        return "simulation_truth"
    return "sensor"


def _pose_of(rec: dict | None, frame: str, min_conf: float) -> tuple[dict | None, str]:
    """(pose, "") or (None, why the record cannot give an orientation-bearing pose).
    A sensor record without a confidence has no identity evidence; privileged or
    synthetic records know their identity by construction."""
    if rec is None or rec.get("visibility") not in VISIBLE:
        return None, "target_occluded"
    if rec.get("position_m") is None or rec.get("quaternion_xyzw") is None:
        return None, "orientation_missing"
    conf = rec.get("confidence")
    if conf is None and tier_of(rec) == "sensor":
        return None, "identity_mismatch"
    if conf is not None and conf < min_conf:
        return None, "identity_mismatch"
    try:
        p = check_pose(
            {
                "frame_id": rec["frame_id"],
                "position_m": rec["position_m"],
                "quaternion_xyzw": rec["quaternion_xyzw"],
            },
            "observation",
            frame,
        )
    except RequestRejected:
        return None, "orientation_missing"
    return p, ""


def clean_observations(records: Any, frame: str, now: float, unknown: list[dict]) -> list[dict]:
    """Well-formed, past, labelled records in time order; one per (object, timestamp)."""
    if not isinstance(records, list):
        raise RequestRejected(
            "invalid_request", "observations must be a list of Observation records"
        )
    seen: set[tuple[str, float]] = set()
    out, repeated, bad = [], 0, 0
    for r in records:
        t = r.get("timestamp_s") if isinstance(r, dict) else None
        if (
            not isinstance(r, dict)
            or r.get("frame_id") != frame
            or not isinstance(r.get("object_id"), str)
            or not r.get("source")
            or not isinstance(r.get("privileged"), bool)
            or isinstance(t, bool)
            or not isinstance(t, (int, float))
            or not math.isfinite(t)
            or t > now + 0.1
        ):
            bad += 1
            continue
        key = (r["object_id"], float(t))
        if key in seen:
            repeated += 1
            continue
        seen.add(key)
        out.append(r)
    if repeated:
        unknown.append(
            {
                "code": "observation_repeated",
                "detail": f"{repeated} record(s) repeat an earlier (object, timestamp) and were not counted",
            }
        )
    if bad:
        unknown.append(
            {
                "code": "observation_stale",
                "detail": f"{bad} record(s) dropped: wrong frame, missing/future timestamp, or no source/privileged label",
            }
        )
    return sorted(out, key=lambda r: float(r["timestamp_s"]))


def verify_stable_placement(request: dict, ctx: Any = None) -> dict:
    result = envelope(SKILL_ID, request, kind="tool", ctx=ctx)
    result["phase"] = "verify"
    failures: list[dict] = []
    unknown: list[dict] = []
    try:
        frame = request.get("frame_id")
        if not isinstance(frame, str) or not frame:
            raise RequestRejected("invalid_request", "frame_id is required")
        target_id = request.get("target_id")
        if not isinstance(target_id, str) or not target_id:
            raise RequestRejected("invalid_request", "target_id is required")
        size = check_geometry(request.get("geometry"))
        goal = check_pose(request.get("target_object_pose"), "target_object_pose", frame)
        support = check_support(request.get("support"), frame)
        th, th_ref = thresholds_for(request)
        events = request.get("events") or {}
        t_start = finite(events.get("action_start_s"), "events.action_start_s", 0, 1e12)
        t_release = finite(events.get("release_s"), "events.release_s", 0, 1e12)
        t_retreat = finite(events.get("retreat_done_s"), "events.retreat_done_s", 0, 1e12)
        if not t_start <= t_release <= t_retreat:
            raise RequestRejected(
                "invalid_request", "events must be ordered start <= release <= retreat_done"
            )
        now = finite(request.get("now_s", ctx.now() if ctx is not None else None), "now_s", 0, 1e12)
        tip_reach = finite(request.get("gripper_tip_reach_m", 0.0), "gripper_tip_reach_m", 0, 0.5)
        base_poses = {}
        for b in request.get("neighbors_baseline") or []:
            nid = b.get("object_id") if isinstance(b, dict) else None
            if not nid or not b.get("source"):
                raise RequestRejected(
                    "invalid_request", "neighbors_baseline entries are Observation records"
                )
            if finite(b.get("timestamp_s"), f"neighbour {nid} timestamp_s", 0, 1e12) > t_start:
                raise RequestRejected(
                    "invalid_request",
                    f"neighbour {nid}: its baseline was observed after the action started",
                )
            p, why = _pose_of(b, frame, th["min_identity_confidence"])
            if p is None:
                raise RequestRejected(
                    "invalid_request", f"neighbour {nid}: baseline unusable ({why})"
                )
            base_poses[nid] = p
    except RequestRejected as exc:
        set_failure(result, exc.code, "validate", str(exc), "rejected")
        return result

    normal = support["normal"]
    up_sym, yaw_sym = size["up_symmetric"], size["yaw_symmetry_deg"]
    records = clean_observations(request.get("observations"), frame, now, unknown)
    tiers = sorted({tier_of(r) for r in records})
    privileged = any(t == "simulation_truth" for t in tiers)
    result["privileged_inputs"] = privileged
    evidence: dict[str, Any] = {
        "evidence_tiers": tiers,
        "evidence_tier": "synthetic_contract_test"
        if "synthetic_contract_test" in tiers
        else "simulation_truth"
        if privileged
        else "sensor"
        if tiers
        else "none",
        "records_used": len(records),
        "thresholds": th,
        "thresholds_ref": th_ref,
    }
    if request.get("require_unprivileged") and privileged:
        unknown.append(
            {
                "code": "privileged_observation",
                "detail": "unprivileged observations required; got simulation truth",
            }
        )

    def of(oid: str) -> list[dict]:
        return [r for r in records if r["object_id"] == oid and float(r["timestamp_s"]) > t_release]

    # --- the target after the release; the settle window after the retreat
    target_recs = of(target_id)
    window_recs = [r for r in target_recs if float(r["timestamp_s"]) >= t_retreat]

    def metrics(p: dict) -> dict:
        return {
            "position_error_m": geo.horizontal_offset(p, goal, normal),
            "yaw_error_rad": geo.yaw_error_rad(p, goal, normal, yaw_sym),
            "tilt_rad": geo.tilt_rad(p, normal, up_symmetric=up_sym),
            "lowest_point_height_m": geo.lowest_point_height(p, size, support["point_m"], normal),
        }

    track, label_down, reasons = [], None, {}
    for r in target_recs:
        t = float(r["timestamp_s"])
        p, why = _pose_of(r, frame, th["min_identity_confidence"])
        if p is None and r.get("visibility") in VISIBLE and r.get("state_label") in FALLEN_LABELS:
            label_down = label_down or (t, r)
        if p is None:
            if t >= t_retreat:
                reasons.setdefault(why, t)
            continue
        track.append((t, p, metrics(p)))
    for why, t in reasons.items():
        unknown.append({"code": why, "detail": f"target {target_id} at t={t:.3f}: {why}"})
    in_window = [x for x in track if x[0] >= t_retreat]

    if not window_recs:
        unknown.append(
            {
                "code": "insufficient_settle_window",
                "detail": "no observation of the target after the retreat",
            }
        )
    else:
        ts = [float(r["timestamp_s"]) for r in window_recs]
        usable_ts = [x[0] for x in in_window]
        span = (max(usable_ts) - min(usable_ts)) if usable_ts else 0.0
        age = now - max(ts)
        evidence["window"] = {
            "first_s": min(ts),
            "last_s": max(ts),
            "usable_span_s": span,
            "usable_samples": len(usable_ts),
            "last_age_s": age,
        }
        if age > th["max_observation_age_s"]:
            unknown.append(
                {
                    "code": "observation_stale",
                    "detail": f"newest target observation is {age:.2f} s old",
                }
            )
        if span + 1e-9 < th["settle_window_s"] or len(usable_ts) < th["min_window_samples"]:
            unknown.append(
                {
                    "code": "insufficient_settle_window",
                    "detail": f"{len(usable_ts)} usable frame(s) over {span:.2f} s; need {th['min_window_samples']} over {th['settle_window_s']:.2f} s",
                }
            )

    fell_at = next((x for x in track if x[2]["tilt_rad"] > th["tilt_tolerance_rad"]), None)
    if fell_at is not None:
        failures.append(
            {
                "code": "unstable_after_release",
                "phase": "release" if fell_at[0] < t_retreat else "settle",
                "detail": f"tilt {math.degrees(fell_at[2]['tilt_rad']):.1f} deg at t={fell_at[0]:.3f}",
            }
        )
    elif label_down is not None:
        failures.append(
            {
                "code": "unstable_after_release",
                "phase": "release" if label_down[0] < t_retreat else "settle",
                "detail": f"{label_down[1]['source']} labels the target {label_down[1]['state_label']} at t={label_down[0]:.3f}",
            }
        )
    if in_window:
        last = in_window[-1][2]
        first_p = in_window[0][1]
        drift = max(
            float(np.linalg.norm(np.subtract(p["position_m"], first_p["position_m"])))
            for _, p, _ in in_window
        )
        tilts = [m["tilt_rad"] for _, _, m in in_window]
        evidence["target"] = {
            "final": last,
            "final_pose": in_window[-1][1],
            "max_drift_m": drift,
            "tilt_change_rad": max(tilts) - min(tilts),
        }
        if fell_at is None:
            if (
                drift > th["drift_tolerance_m"]
                or max(tilts) - min(tilts) > th["drift_tilt_tolerance_rad"]
            ):
                failures.append(
                    {
                        "code": "unstable_after_release",
                        "phase": "settle",
                        "detail": f"drifted {drift * 1e3:.1f} mm / {math.degrees(max(tilts) - min(tilts)):.1f} deg in the window",
                    }
                )
            if (
                last["position_error_m"] > th["position_tolerance_m"]
                or last["yaw_error_rad"] > th["yaw_tolerance_rad"]
            ):
                failures.append(
                    {
                        "code": "placement_error",
                        "phase": "settle",
                        "detail": f"{last['position_error_m'] * 1e3:.1f} mm, yaw {math.degrees(last['yaw_error_rad']):.1f} deg off target",
                    }
                )
            low = last["lowest_point_height_m"]
            if low > th["support_height_tolerance_m"]:
                failures.append(
                    {
                        "code": "not_resting_on_support",
                        "phase": "settle",
                        "detail": f"lowest corner {low * 1e3:.1f} mm over the support: held up, not resting on it",
                    }
                )
            elif low < -th["support_height_tolerance_m"] - support["height_std_m"]:
                unknown.append(
                    {
                        "code": "support_unconfirmed",
                        "detail": f"lowest corner {low * 1e3:.1f} mm below the calibrated support ({support['calibration_version']}): plane or pose estimate is off",
                    }
                )

    # --- the release: measured gripper state, or the measured TCP clear of the tile
    release = {"confirmed": False, "by": None}
    for g in request.get("gripper_states") or []:
        if (
            not isinstance(g, dict)
            or not g.get("source")
            or not isinstance(g.get("timestamp_s"), (int, float))
        ):
            continue
        if float(g["timestamp_s"]) <= t_release:
            continue
        if g.get("holding") is True:
            failures.append(
                {
                    "code": "release_failed",
                    "phase": "release",
                    "detail": f"gripper ({g['source']}) reports holding at t={g['timestamp_s']:.3f}",
                }
            )
            release = {"confirmed": False, "by": "gripper_holding_true"}
            break
        if g.get("holding") is False:
            release = {"confirmed": True, "by": f"gripper:{g['source']}"}
    if not release["confirmed"] and release["by"] is None:
        for s in request.get("tcp_states") or []:
            if (
                not isinstance(s, dict)
                or not isinstance(s.get("timestamp_s"), (int, float))
                or float(s["timestamp_s"]) < t_release
            ):
                continue
            try:
                tcp = check_pose(s.get("pose"), "tcp_states.pose", frame)
            except RequestRejected:
                continue
            top = size["height_m"] + geo.bottom_z(size) if in_window else size["height_m"]
            obj_h = (
                geo.height_over_plane(in_window[-1][1]["position_m"], support["point_m"], normal)
                if in_window
                else 0.0
            )
            clear = (
                geo.height_over_plane(tcp["position_m"], support["point_m"], normal)
                - tip_reach
                - (obj_h + top)
            )
            if clear > th["release_clearance_m"]:
                release = {"confirmed": True, "by": f"measured_tcp_clear_{clear * 1e3:.0f}mm"}
                break
    evidence["release"] = release
    if not release["confirmed"] and release["by"] is None:
        unknown.append(
            {
                "code": "release_not_confirmed",
                "detail": "no gripper state or measured TCP shows the hand let go",
            }
        )

    # --- neighbours, against their own pre-action baseline
    neighbours = {}
    for nid, b in base_poses.items():
        recs = [
            (float(r["timestamp_s"]), *_pose_of(r, frame, th["min_identity_confidence"]), r)
            for r in of(nid)
        ]
        last_rec = next((x for x in reversed(recs) if x[1] is not None and x[0] >= t_retreat), None)
        down = next(
            (
                x
                for x in recs
                if x[1] is None
                and x[3].get("visibility") in VISIBLE
                and x[3].get("state_label") in FALLEN_LABELS
            ),
            None,
        )

        def moved(p: dict, b: dict = b) -> tuple[float, float]:
            d = float(np.linalg.norm(np.subtract(p["position_m"], b["position_m"])))
            dt = abs(
                geo.tilt_rad(p, normal, up_symmetric=up_sym)
                - geo.tilt_rad(b, normal, up_symmetric=up_sym)
            )
            return d, dt

        def bad(x: tuple[float, float]) -> bool:
            return (
                x[0] > th["neighbor_displacement_tolerance_m"]
                or x[1] > th["neighbor_tilt_tolerance_rad"]
            )

        if last_rec is None and down is None:
            unknown.append(
                {
                    "code": "target_occluded",
                    "detail": f"neighbour {nid} not observed with a pose after the retreat",
                }
            )
            neighbours[nid] = {"observed": False}
            continue
        if last_rec is not None:
            d, dt = moved(last_rec[1])
            neighbours[nid] = {"observed": True, "displacement_m": d, "tilt_change_rad": dt}
            disturbed = bad((d, dt))
            detail = (
                f"neighbour {nid} moved {d * 1e3:.1f} mm, tilt changed {math.degrees(dt):.1f} deg"
            )
        else:
            neighbours[nid] = {"observed": True, "state_label": down[3].get("state_label")}
            disturbed, detail = (
                True,
                f"neighbour {nid} labelled {down[3].get('state_label')} by {down[3]['source']}",
            )
        if disturbed:
            pre = [moved(p) for t, p, _, _ in recs if p is not None and t < t_retreat]
            phase = "unknown" if not pre else "release" if any(bad(x) for x in pre) else "retreat"
            failures.append(
                {"code": "neighbor_disturbed", "phase": phase, "neighbor_id": nid, "detail": detail}
            )
    diff = request.get("snapshot_diff")
    if isinstance(diff, dict):
        for nid in diff.get("disturbed") or []:
            if nid in base_poses and not any(f.get("neighbor_id") == nid for f in failures):
                failures.append(
                    {
                        "code": "neighbor_disturbed",
                        "phase": "unknown",
                        "neighbor_id": nid,
                        "detail": f"detecting-disturbed-pieces-by-snapshot-diff reports {nid} disturbed",
                    }
                )
    evidence["neighbors"] = neighbours

    passed = not failures and not unknown
    measured = dict(evidence.get("target", {}).get("final", {}))
    measured.update(
        {k: evidence["target"][k] for k in ("max_drift_m", "tilt_change_rad")}
        if "target" in evidence
        else {}
    )
    evidence["postcondition"] = {
        "name": "standing_independently_at_target",
        "passed": passed,
        "criterion": "position/yaw within tolerance, tilt <= tilt_tolerance_rad and drift bounded over settle_window_s "
        "after the retreat, resting on the support, gripper released, every baseline neighbour undisturbed",
        "measured": measured,
        "thresholds_ref": th_ref,
    }
    result["evidence"] = evidence
    result["observations"] = [r for r in records if float(r["timestamp_s"]) > t_release]
    result["faults"] = failures
    result["unknowns"] = unknown
    result["unverified"] = [u["code"] for u in unknown]
    if failures:
        primary = min(failures, key=lambda f: FAILURE_PRIORITY.index(f["code"]))
        set_failure(
            result,
            primary["code"],
            primary["phase"],
            primary["detail"],
            "failed",
            **{k: v for k, v in primary.items() if k not in ("code", "phase", "detail")},
        )
        result["phase"] = "verify"
    elif unknown:
        set_failure(result, unknown[0]["code"], "verify", unknown[0]["detail"], "unverified")
    else:
        result["status"] = "succeeded"
    return result
