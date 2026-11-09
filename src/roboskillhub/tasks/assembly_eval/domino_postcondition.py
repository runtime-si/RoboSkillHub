# SPDX-License-Identifier: Apache-2.0
"""Independent post-conditions for upright domino placement.

The verdict comes only from an observation stream of object poses (and, when the
backend reports them, contacts) plus the event times; the executor's own report is
recorded next to it and never decides it. Statuses keep the shared semantics:

- ``rejected``: the evaluation request itself is malformed (frames, quaternions, events);
- ``failed``: fresh observations in the stability window show the task is not done;
- ``unverified``: the action may have happened but the window lacks fresh evidence
  (stale/repeated frames, target not seen, support/contact unknown);
- ``succeeded``: every check passed on fresh, distinct samples covering the window.

Observation sample (dict)::

    {"t": sim_time_s, "stamp": observation_time_s, "seq": int, "source": "simulation_truth",
     "frame": "table",
     "objects": {"tile_0": {"position": [x, y, z], "quat_xyzw": [x, y, z, w]}, ...},
     "contacts": {"tile_0": ["table", "gripper_pad_l"], ...} | None}

``contacts`` maps a body to the bodies it touches at that sample; ``None`` (or absent)
means the backend gave no contact information, so support/release stays unknown.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from .domino_geometry import dist, dist_xy, quat_normalize, quat_rotate, tilt_deg, yaw_error_deg
from .domino_scene import DominoSceneSpec

EVAL_VERSION = "domino_postcondition/0.1.0"
GRIPPER_BODIES = ("gripper_pad_l", "gripper_pad_r", "gripper_palm")


@dataclass(frozen=True)
class Thresholds:
    position_tol: float = 0.005  # slot centre error in the table plane
    yaw_tol_deg: float = 5.0
    tilt_tol_deg: float = 5.0  # predicators Upright: roll < 5 deg
    fallen_deg: float = 10.0  # predicators Toppled: roll > 10 deg
    bottom_tol: float = 0.002  # tile bottom vs table plane
    drift_tol: float = 0.001  # position drift inside the window
    tilt_drift_deg: float = 1.0
    neighbour_disp_tol: float = 0.002
    neighbour_tilt_tol_deg: float = 2.0
    window_s: float = 1.0
    min_samples: int = 5
    max_obs_age_s: float = 0.2
    not_placed_dist: float = 0.03  # farther than this from the slot: tile is not there
    source: str = (
        "evaluation choices on simulator truth: tilt 5/10 deg from predicators' pybullet_domino "
        "Upright/Toppled; position/drift/neighbour bounds chosen well above the MuJoCo twin's "
        "0.1-1.9 mm placement scatter; not calibrated for a camera observer"
    )


DEFAULT_THRESHOLDS = Thresholds()


def _bad(code: str, detail: str) -> dict:
    return {"code": code, "detail": detail}


def validate_request(samples: list[dict], events: dict) -> list[dict]:
    errs = []
    for key in ("release_t", "retreat_done_t", "eval_t"):
        v = events.get(key)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            errs.append(_bad("events", f"events.{key} missing or not finite"))
    if not errs and not events["release_t"] <= events["retreat_done_t"] <= events["eval_t"]:
        errs.append(_bad("events", "need release_t <= retreat_done_t <= eval_t"))
    for i, s in enumerate(samples):
        if s.get("frame") != "table":
            errs.append(_bad("frame", f"sample {i} frame {s.get('frame')!r}, expected 'table'"))
            break
        for oid, o in (s.get("objects") or {}).items():
            pos, q = o.get("position"), o.get("quat_xyzw")
            if pos is None or len(pos) != 3 or not all(math.isfinite(c) for c in pos):
                errs.append(_bad("position", f"sample {i} {oid} position invalid"))
            if q is not None:
                try:
                    quat_normalize(q)
                except ValueError as e:
                    errs.append(_bad("quaternion", f"sample {i} {oid}: {e}"))
        if errs:
            break
    return errs


def fresh_samples(samples: Iterable[dict], max_age: float) -> tuple[list[dict], dict]:
    """Keep samples whose stamp strictly advances, whose sequence number is new, and
    whose age (sim time minus stamp) is within ``max_age``. Repeated frames do not fill
    a stability window."""
    out, last_stamp, seen = [], -math.inf, set()
    stats = {"total": 0, "repeated": 0, "stale": 0, "out_of_order": 0}
    for s in samples:
        stats["total"] += 1
        stamp, seq = s.get("stamp"), s.get("seq")
        if stamp is None:
            stats["stale"] += 1
            continue
        if seq in seen or stamp == last_stamp:
            stats["repeated"] += 1
            continue
        if stamp < last_stamp:
            stats["out_of_order"] += 1
            continue
        if s.get("t", stamp) - stamp > max_age:
            stats["stale"] += 1
            continue
        seen.add(seq)
        last_stamp = stamp
        out.append(s)
    return out, stats


def _pose(s: dict, oid: str) -> dict | None:
    o = (s.get("objects") or {}).get(oid)
    if o is None or o.get("quat_xyzw") is None:
        return None
    return o


def tile_metrics(spec: DominoSceneSpec, pose: dict, target: dict) -> dict:
    pos, q = pose["position"], pose["quat_xyzw"]
    tilt = tilt_deg(q)
    # Bottom of the box above the table: lowest corner of the (possibly tilted) box.
    hx, hy, hz = spec.tile.half
    lowest = min(
        pos[2] + quat_rotate(q, (sx * hx, sy * hy, sz * hz))[2]
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    )
    return {
        "position_error_m": dist_xy(pos, target["position"]),
        "yaw_error_deg": yaw_error_deg(q, target["yaw"]),
        "tilt_deg": tilt,
        "bottom_height_m": lowest - spec.table.top_z,
        "center_height_error_m": pos[2] - target["position"][2],
    }


def _phase_at(events: dict, t: float) -> str | None:
    for ph in events.get("phases") or []:
        if ph["t0"] <= t <= ph["t1"]:
            return ph["name"]
    return None


def evaluate_placement(
    spec: DominoSceneSpec,
    slot_index: int,
    target_id: str,
    neighbours: dict[str, dict],
    samples: list[dict],
    events: dict,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    executor_report: dict | None = None,
    feeder_pose: dict | None = None,
) -> dict:
    """Judge one placement. ``neighbours`` maps the tiles that stood before this
    placement to their baseline pose (from a fresh observation before the pick)."""
    th = thresholds
    target = spec.slot_pose(slot_index)
    result: dict[str, Any] = {
        "evaluator": EVAL_VERSION,
        "slot_index": slot_index,
        "target_id": target_id,
        "target_pose": target,
        "thresholds": asdict(th),
        "executor_claim": (executor_report or {}).get("status"),
        "failure_types": [],
        "cause_hypotheses": [],
        "unverified_reasons": [],
        "observation_sources": sorted({s.get("source", "?") for s in samples}),
    }
    errs = validate_request(samples, events)
    if errs:
        result.update(status="rejected", errors=errs)
        return result
    fresh, stats = fresh_samples(samples, th.max_obs_age_s)
    result["observation_stats"] = stats
    t_ret, t_eval = events["retreat_done_t"], events["eval_t"]
    window = [s for s in fresh if t_ret <= s["stamp"] <= t_eval]
    with_target = [s for s in window if _pose(s, target_id) is not None]
    span = (with_target[-1]["stamp"] - with_target[0]["stamp"]) if len(with_target) >= 2 else 0.0
    result["window"] = {
        "t0": t_ret,
        "t1": t_eval,
        "fresh_samples": len(window),
        "samples_with_target": len(with_target),
        "covered_s": span,
    }

    # Attribution (not the verdict): when did a neighbour first move, using every fresh
    # sample from the release on?
    for nid, base in neighbours.items():
        for s in fresh:
            if s["stamp"] < events["release_t"] - 1e-9 and not events.get("phases"):
                continue
            p = _pose(s, nid)
            if p is None:
                continue
            moved = dist(p["position"], base["position"]) > th.neighbour_disp_tol
            tilted = tilt_deg(p["quat_xyzw"]) > th.neighbour_tilt_tol_deg
            if moved or tilted:
                result.setdefault("neighbour_first_moved", {})[nid] = {
                    "t": s["stamp"],
                    "phase": _phase_at(events, s["stamp"]),
                }
                break

    covered = len(with_target) >= th.min_samples and span >= 0.9 * th.window_s
    if t_eval - t_ret < th.window_s - 1e-9:
        result["unverified_reasons"].append("stability window shorter than configured")
        covered = False
    if not covered:
        if stats["repeated"] or stats["stale"]:
            result["unverified_reasons"].append(
                f"observation_stale: {stats['repeated']} repeated and {stats['stale']} stale samples; "
                f"{len(with_target)} fresh samples with the target cover {span:.2f} s of {th.window_s:.2f} s"
            )
        elif len(window) and not with_target:
            result["unverified_reasons"].append(
                "orientation_missing: target pose not observed in the window"
            )
        else:
            result["unverified_reasons"].append(
                f"insufficient window: {len(with_target)} samples over {span:.2f} s"
            )

    per_sample = []
    for s in with_target:
        m = tile_metrics(spec, _pose(s, target_id), target)
        m["stamp"] = s["stamp"]
        c = s.get("contacts")
        m["contacts_known"] = c is not None
        if c is not None:
            touching = set(c.get(target_id, []))
            m["touches_table"] = "table" in touching
            m["touches_gripper"] = bool(touching & set(GRIPPER_BODIES))
            m["touches"] = sorted(touching)
        nb = {}
        for nid, base in neighbours.items():
            p = _pose(s, nid)
            if p is None:
                nb[nid] = None
                continue
            nb[nid] = {
                "displacement_m": dist(p["position"], base["position"]),
                "tilt_deg": tilt_deg(p["quat_xyzw"]),
            }
        m["neighbours"] = nb
        per_sample.append(m)

    failures: list[str] = []
    hyps: list[dict] = []
    if per_sample:
        last = per_sample[-1]
        max_tilt = max(m["tilt_deg"] for m in per_sample)
        result["final"] = {k: v for k, v in last.items() if k != "neighbours"}
        result["final"]["neighbours"] = last["neighbours"]
        result["max_tilt_deg"] = max_tilt
        sp = [_pose(s, target_id)["position"] for s in with_target]
        pos_drift = max(dist(p, sp[0]) for p in sp)
        tilt_drift = max(m["tilt_deg"] for m in per_sample) - min(m["tilt_deg"] for m in per_sample)
        result["drift_m"] = pos_drift
        result["tilt_drift_deg"] = tilt_drift
        if last["position_error_m"] > th.not_placed_dist:
            failures.append("not_at_slot")
            if (
                feeder_pose is not None
                and dist(_pose(with_target[-1], target_id)["position"], feeder_pose["position"])
                < 0.01
            ):
                hyps.append(
                    {
                        "cause": "grasp_missed",
                        "evidence": "tile still at its feeder pose",
                        "confidence": "high",
                    }
                )
            else:
                hyps.append(
                    {
                        "cause": "dropped_or_knocked_away",
                        "evidence": f"tile {last['position_error_m'] * 1e3:.0f} mm from the slot",
                        "confidence": "medium",
                    }
                )
        if max_tilt > th.fallen_deg:
            failures.append("toppled")
        elif max_tilt > th.tilt_tol_deg:
            failures.append("not_upright")
        if "not_at_slot" not in failures and (
            last["position_error_m"] > th.position_tol or last["yaw_error_deg"] > th.yaw_tol_deg
        ):
            failures.append("placement_error")
        if pos_drift > th.drift_tol or tilt_drift > th.tilt_drift_deg:
            failures.append("unstable_after_release")
        if (
            "toppled" not in failures
            and "not_at_slot" not in failures
            and (abs(last["bottom_height_m"]) > th.bottom_tol)
        ):
            failures.append("not_supported")
        if not all(m["contacts_known"] for m in per_sample):
            result["unverified_reasons"].append("support_unknown: backend reported no contacts")
        else:
            if any(m.get("touches_gripper") for m in per_sample):
                failures.append("release_not_confirmed")
                hyps.append(
                    {
                        "cause": "gripper_still_holding",
                        "evidence": "gripper contact with the target inside the window",
                        "confidence": "high",
                    }
                )
            if "toppled" not in failures and not last.get("touches_table"):
                failures.append("not_supported")
        disturbed = []
        unseen = []
        for nid in neighbours:
            vals = [m["neighbours"].get(nid) for m in per_sample]
            seen = [v for v in vals if v is not None]
            if not seen:
                unseen.append(nid)
                continue
            if any(
                v["displacement_m"] > th.neighbour_disp_tol
                or v["tilt_deg"] > th.neighbour_tilt_tol_deg
                for v in seen
            ):
                disturbed.append(nid)
        if disturbed:
            failures.append("neighbor_disturbed")
            for nid in disturbed:
                fm = (result.get("neighbour_first_moved") or {}).get(nid)
                hyps.append(
                    {
                        "cause": f"{fm['phase']}_contact"
                        if fm and fm.get("phase")
                        else "unknown_contact",
                        "evidence": f"{nid} first moved at t={fm['t']:.3f}s"
                        if fm
                        else f"{nid} moved",
                        "confidence": "medium",
                    }
                )
        if unseen:
            result["unverified_reasons"].append(f"neighbours not observed in the window: {unseen}")
        result["neighbours_disturbed"] = disturbed

    seen_f = []
    for f in failures:
        if f not in seen_f:
            seen_f.append(f)
    result["failure_types"] = seen_f
    result["cause_hypotheses"] = hyps
    if seen_f:
        status = "failed"
    elif result["unverified_reasons"] or not per_sample:
        status = "unverified"
    else:
        status = "succeeded"
    result["status"] = status
    claim = result["executor_claim"]
    result["executor_claim_contradicted"] = bool(
        claim in ("finished", "ok", "succeeded") and status != "succeeded"
    )
    return result


def evaluate_layout(
    spec: DominoSceneSpec,
    samples: list[dict],
    t0: float,
    t1: float,
    placement_results: list[dict],
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> dict:
    """Whole-episode verdict: every placement succeeded and, in a final window, every
    tile stands at its slot with no gripper contact."""
    th = thresholds
    fresh, stats = fresh_samples(samples, th.max_obs_age_s)
    window = [s for s in fresh if t0 <= s["stamp"] <= t1]
    tiles = {}
    for k, tid in enumerate(spec.tile_ids()):
        with_t = [s for s in window if _pose(s, tid) is not None]
        if len(with_t) < th.min_samples:
            tiles[tid] = {"status": "unverified", "reason": "not observed in the final window"}
            continue
        m = tile_metrics(spec, _pose(with_t[-1], tid), spec.slot_pose(k))
        max_tilt = max(tilt_deg(_pose(s, tid)["quat_xyzw"]) for s in with_t)
        ok = (
            m["position_error_m"] <= th.position_tol
            and m["yaw_error_deg"] <= th.yaw_tol_deg
            and max_tilt <= th.tilt_tol_deg
            and abs(m["bottom_height_m"]) <= th.bottom_tol
        )
        c = with_t[-1].get("contacts")
        if c is not None and set(c.get(tid, [])) & set(GRIPPER_BODIES):
            ok = False
        tiles[tid] = {"status": "succeeded" if ok else "failed", **m, "max_tilt_deg": max_tilt}
    statuses = [p.get("status") for p in placement_results] + [v["status"] for v in tiles.values()]
    if len(placement_results) < spec.n_tiles:
        statuses.append("failed" if any(s == "failed" for s in statuses) else "unverified")
    if "rejected" in statuses:
        status = "rejected"
    elif "failed" in statuses:
        status = "failed"
    elif "unverified" in statuses:
        status = "unverified"
    else:
        status = "succeeded"
    return {
        "evaluator": EVAL_VERSION,
        "status": status,
        "tiles": tiles,
        "standing_at_slot": sum(1 for v in tiles.values() if v["status"] == "succeeded"),
        "observation_stats": stats,
        "window": {"t0": t0, "t1": t1, "fresh_samples": len(window)},
    }
