"""verify_stable_placement: an independent judgement from observations alone.

It never reads what the executor said it did. It takes the target pose, a sequence of
timestamped observations (sensor output or explicitly labelled simulation truth), the
release / retreat event times, a baseline of the neighbours observed before the action
and the thresholds, and decides:

1. the target is the right tile, in tolerance of its target pose, observed after the
   release and after the retreat;
2. its tilt (object up axis against the support normal, not world z or centre height)
   stays within tolerance; yaw folds by the tile's declared symmetry;
3. it rests on the support (lowest corner on the plane) and the gripper is no longer
   holding it; when neither can be observed the release stays unknown;
4. distinct frames spanning the settle window agree (one frame repeated is one frame);
5. every neighbour in the baseline is still where it was and still upright;
6. stale, future, wrong-frame, unidentified, occluded or orientation-less observations
   are never read as zero error.

Observation sample::

    {"sample_id": str, "timestamp_s": float, "frame_id": str, "source": str,
     "source_kind": "sensor" | "simulation_truth" | "synthetic_contract_test",
     "objects": {id: {"pose": pose | None, "identity_confidence": float, "visible": bool}},
     "gripper": {"holding": bool | None, "width_m": float, "source": str},   # optional
     "tcp_pose": pose}                                                       # optional, measured
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
    check_thresholds,
    finite,
    result_skeleton,
)

SOURCE_KINDS = ("sensor", "simulation_truth", "synthetic_contract_test")

# Failed codes in the order they are reported as the primary failure.
FAILURE_PRIORITY = (
    "neighbor_disturbed",
    "unstable_after_release",
    "release_not_confirmed",
    "placement_error",
)


def _record(sample: dict, oid: str) -> dict | None:
    objs = sample.get("objects")
    rec = objs.get(oid) if isinstance(objs, dict) else None
    return rec if isinstance(rec, dict) else None


def _usable_pose(rec: dict | None, frame_id: str, min_conf: float) -> tuple[dict | None, str]:
    """(pose, "") or (None, the reason it cannot be used)."""
    if rec is None:
        return None, "target_occluded"
    if rec.get("visible") is False:
        return None, "target_occluded"
    conf = rec.get("identity_confidence")
    if not isinstance(conf, (int, float)) or conf < min_conf:
        return None, "identity_mismatch"
    p = rec.get("pose")
    if not isinstance(p, dict) or p.get("quaternion_xyzw") is None:
        return None, "orientation_missing"
    try:
        return check_pose(p, "observed pose", frame_id), ""
    except RequestRejected:
        return None, "orientation_missing"


def _clean_samples(
    samples: Any, frame_id: str, now: float, unknown: list[dict]
) -> tuple[list[dict], set[str]]:
    """Distinct, well-formed, past samples in time order, and their source kinds."""
    if not isinstance(samples, list):
        raise RequestRejected("invalid_request", "observations must be a list")
    seen_ids: set[Any] = set()
    seen_t: set[float] = set()
    out, kinds, repeated, bad = [], set(), 0, 0
    for s in samples:
        if not isinstance(s, dict) or s.get("frame_id") != frame_id:
            bad += 1
            continue
        t = s.get("timestamp_s")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
            bad += 1
            continue
        if t > now + 0.1 or s.get("source_kind") not in SOURCE_KINDS or not s.get("source"):
            bad += 1
            continue
        sid = s.get("sample_id")
        if sid is None or sid in seen_ids or float(t) in seen_t:
            repeated += 1
            continue
        seen_ids.add(sid)
        seen_t.add(float(t))
        kinds.add(s["source_kind"])
        out.append(s)
    if repeated:
        unknown.append(
            {
                "code": "observation_repeated",
                "detail": f"{repeated} sample(s) repeat an earlier frame id or timestamp and were not counted",
            }
        )
    if bad:
        unknown.append(
            {
                "code": "observation_stale",
                "detail": f"{bad} sample(s) dropped: wrong frame, bad or future timestamp, or no source label",
            }
        )
    out.sort(key=lambda s: float(s["timestamp_s"]))
    return out, kinds


def verify_stable_placement(request: dict) -> dict:
    result = result_skeleton("verify_stable_placement", request.get("request_id"))
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
        th = check_thresholds(request.get("thresholds"))
        events = request.get("events") or {}
        t_start = finite(events.get("action_start_s"), "events.action_start_s", 0, 1e12)
        t_release = finite(events.get("release_s"), "events.release_s", 0, 1e12)
        t_retreat = finite(events.get("retreat_done_s"), "events.retreat_done_s", 0, 1e12)
        if not t_start <= t_release <= t_retreat:
            raise RequestRejected(
                "invalid_request", "events must be ordered start <= release <= retreat_done"
            )
        now = finite(request.get("now_s"), "now_s", 0, 1e12)
        tip_reach = finite(request.get("gripper_tip_reach_m", 0.0), "gripper_tip_reach_m", 0, 0.5)
        baseline = request.get("neighbors_baseline") or {}
        if not isinstance(baseline, dict):
            raise RequestRejected(
                "invalid_request", "neighbors_baseline must be {id: {pose, timestamp_s, source}}"
            )
        base_poses = {}
        for nid, b in baseline.items():
            if not isinstance(b, dict) or not b.get("source"):
                raise RequestRejected(
                    "invalid_request", f"neighbour {nid}: baseline needs pose, timestamp_s, source"
                )
            bt = finite(b.get("timestamp_s"), f"neighbour {nid} timestamp_s", 0, 1e12)
            if bt > t_start:
                raise RequestRejected(
                    "invalid_request",
                    f"neighbour {nid}: its baseline was observed after the action started",
                )
            base_poses[nid] = check_pose(b.get("pose"), f"neighbour {nid} baseline", frame)
    except RequestRejected as exc:
        result["failure"] = {"code": exc.code, "detail": str(exc)}
        return result

    normal = support["normal"]
    up_sym, yaw_sym = size["up_symmetric"], size["yaw_symmetry_deg"]
    samples, kinds = _clean_samples(request.get("observations"), frame, now, unknown)
    after_release = [s for s in samples if float(s["timestamp_s"]) > t_release]
    window = [s for s in samples if float(s["timestamp_s"]) >= t_retreat]
    [s for s in after_release if float(s["timestamp_s"]) < t_retreat]

    evidence: dict[str, Any] = {
        "source_kinds": sorted(kinds),
        "evidence_tier": (
            "synthetic_contract_test"
            if "synthetic_contract_test" in kinds
            else "simulation_truth"
            if "simulation_truth" in kinds
            else "sensor"
            if kinds
            else "none"
        ),
        "samples_used": len(samples),
        "samples_after_release": len(after_release),
        "thresholds": th,
    }
    if request.get("require_sensor_source") and kinds - {"sensor"}:
        unknown.append(
            {
                "code": "observation_stale",
                "detail": f"sensor observations required; got {sorted(kinds)}",
            }
        )

    # --- the settle window: distinct frames, after the retreat, fresh, long enough
    if not window:
        unknown.append(
            {"code": "insufficient_settle_window", "detail": "no observation after the retreat"}
        )
    else:
        span = float(window[-1]["timestamp_s"]) - float(window[0]["timestamp_s"])
        age = now - float(window[-1]["timestamp_s"])
        evidence["window"] = {
            "first_s": float(window[0]["timestamp_s"]),
            "last_s": float(window[-1]["timestamp_s"]),
            "span_s": span,
            "samples": len(window),
            "last_age_s": age,
        }
        if age > th["max_observation_age_s"]:
            unknown.append(
                {"code": "observation_stale", "detail": f"newest observation is {age:.2f} s old"}
            )
        if span + 1e-9 < th["settle_window_s"] or len(window) < th["min_window_samples"]:
            unknown.append(
                {
                    "code": "insufficient_settle_window",
                    "detail": f"{len(window)} distinct frame(s) over {span:.2f} s; need "
                    f"{th['min_window_samples']} over {th['settle_window_s']:.2f} s",
                }
            )

    # --- the target over release + window
    def metrics(p: dict) -> dict:
        return {
            "position_error_m": geo.horizontal_offset(p, goal, normal),
            "yaw_error_rad": geo.yaw_error_rad(p, goal, normal, yaw_sym),
            "tilt_rad": geo.tilt_rad(p, normal, up_symmetric=up_sym),
            "lowest_point_height_m": geo.lowest_point_height(p, size, support["point_m"], normal),
        }

    track = []
    depth_down = None
    for s in after_release:
        rec = _record(s, target_id)
        p, why = _usable_pose(rec, frame, th["min_identity_confidence"])
        t = float(s["timestamp_s"])
        depth = rec.get("depth_upright") if isinstance(rec, dict) else None
        if (
            p is None
            and isinstance(depth, dict)
            and depth.get("seen")
            and not depth.get("standing")
        ):
            # The placing skill's depth check sees no standing tile at the slot: a
            # fall is observable without an orientation estimate.
            depth_down = depth_down or (t, depth)
        if p is None:
            if t >= t_retreat:
                unknown.append({"code": why, "detail": f"target at t={t:.3f}: {why}"})
            continue
        track.append((t, p, metrics(p)))
    in_window = [x for x in track if x[0] >= t_retreat]
    fell_at = next((x for x in track if x[2]["tilt_rad"] > th["tilt_tolerance_rad"]), None)
    if fell_at is None and depth_down is not None:
        failures.append(
            {
                "code": "unstable_after_release",
                "phase": "release" if depth_down[0] < t_retreat else "settle",
                "detail": f"depth check at t={depth_down[0]:.3f}: {depth_down[1].get('why', 'not standing')}",
            }
        )
    if fell_at is not None:
        failures.append(
            {
                "code": "unstable_after_release",
                "phase": "release" if fell_at[0] < t_retreat else "settle",
                "detail": f"tilt {math.degrees(fell_at[2]['tilt_rad']):.1f} deg at t={fell_at[0]:.3f}",
            }
        )
    if in_window:
        last = in_window[-1][2]
        evidence["target"] = {"final": last, "final_pose": in_window[-1][1]}
        first_p = in_window[0][1]
        drift = max(
            float(np.linalg.norm(np.subtract(p["position_m"], first_p["position_m"])))
            for _, p, _ in in_window
        )
        tilts = [m["tilt_rad"] for _, _, m in in_window]
        evidence["target"]["max_drift_m"] = drift
        evidence["target"]["tilt_change_rad"] = max(tilts) - min(tilts)
        if fell_at is None and (
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
        if fell_at is None and (
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
        if fell_at is None and low > th["support_height_tolerance_m"]:
            failures.append(
                {
                    "code": "release_not_confirmed",
                    "phase": "settle",
                    "detail": f"lowest corner {low * 1e3:.1f} mm over the support: held up, not resting on it",
                }
            )
        elif low < -th["support_height_tolerance_m"] - support["height_std_m"]:
            unknown.append(
                {
                    "code": "support_mismatch",
                    "detail": f"lowest corner {low * 1e3:.1f} mm below the calibrated support (calibration "
                    f"{support['calibration_version']}): the plane or the pose estimate is off",
                }
            )

    # --- the release: gripper not holding, or the hand clear of the tile
    release = {"confirmed": False, "by": None}
    for s in window:
        g = s.get("gripper")
        if isinstance(g, dict) and g.get("source") and g.get("holding") is True:
            failures.append(
                {
                    "code": "release_not_confirmed",
                    "phase": "release",
                    "detail": f"gripper ({g['source']}) still reports holding at t={float(s['timestamp_s']):.3f}",
                }
            )
            release = {"confirmed": False, "by": "gripper_holding_true"}
            break
        if isinstance(g, dict) and g.get("source") and g.get("holding") is False:
            release = {"confirmed": True, "by": f"gripper:{g['source']}"}
        tcp = s.get("tcp_pose")
        if not release["confirmed"] and isinstance(tcp, dict):
            try:
                tcp = check_pose(tcp, "tcp_pose", frame)
            except RequestRejected:
                continue
            clear = (
                geo.height_over_plane(tcp["position_m"], support["point_m"], normal)
                - tip_reach
                - size["height_m"]
            )
            if clear > th["release_clearance_m"]:
                release = {"confirmed": True, "by": f"measured_tcp_clear_{clear * 1e3:.0f}mm"}
    evidence["release"] = release
    if not release["confirmed"] and not any(f["code"] == "release_not_confirmed" for f in failures):
        unknown.append(
            {
                "code": "release_not_confirmed",
                "detail": "no gripper state or measured TCP shows the hand let go",
            }
        )

    # --- neighbours: compared with their own pre-action baseline
    neighbours = {}
    for nid, b in base_poses.items():
        recs = []
        for s in after_release:
            p, why = _usable_pose(_record(s, nid), frame, th["min_identity_confidence"])
            recs.append((float(s["timestamp_s"]), p, why))
        last = next((r for r in reversed(recs) if r[1] is not None and r[0] >= t_retreat), None)
        if last is None:
            unknown.append(
                {
                    "code": "target_occluded",
                    "detail": f"neighbour {nid} not observed after the retreat",
                }
            )
            neighbours[nid] = {"observed": False}
            continue

        def moved(p: dict, b: dict = b) -> tuple[float, float]:
            d = float(np.linalg.norm(np.subtract(p["position_m"], b["position_m"])))
            dt = abs(
                geo.tilt_rad(p, normal, up_symmetric=up_sym)
                - geo.tilt_rad(b, normal, up_symmetric=up_sym)
            )
            return d, dt

        d, dt = moved(last[1])
        neighbours[nid] = {"observed": True, "displacement_m": d, "tilt_change_rad": dt}
        if d > th["neighbor_displacement_tolerance_m"] or dt > th["neighbor_tilt_tolerance_rad"]:
            pre = [moved(p) for t, p, _ in recs if p is not None and t < t_retreat]
            if not pre:
                phase = "unknown"
            elif any(
                x > th["neighbor_displacement_tolerance_m"] or y > th["neighbor_tilt_tolerance_rad"]
                for x, y in pre
            ):
                phase = "release"
            else:
                phase = "retreat"
            failures.append(
                {
                    "code": "neighbor_disturbed",
                    "phase": phase,
                    "neighbor_id": nid,
                    "detail": f"neighbour {nid} moved {d * 1e3:.1f} mm, tilt changed {math.degrees(dt):.1f} deg",
                }
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

    result["evidence"] = evidence
    result["faults"] = failures
    result["unknowns"] = unknown
    if failures:
        result["status"] = "failed"
        primary = min(failures, key=lambda f: FAILURE_PRIORITY.index(f["code"]))
        result["failure"] = primary
    elif unknown:
        result["status"] = "unverified"
        result["failure"] = unknown[0]
    else:
        result["status"] = "succeeded"
    result["unverified"] = [u["code"] for u in unknown]
    return result
