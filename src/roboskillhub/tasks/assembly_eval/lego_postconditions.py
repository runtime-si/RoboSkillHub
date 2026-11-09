"""Independent post-conditions and failure attribution for one LEGO placement.

Inputs are simulator snapshots recorded by the Isaac runner (privileged ground truth, labelled as such), never
the policy's own targets or the injected fault: brick poses, the engine's connection list, the contact-triggered
assembly/disassembly events and an upward pull probe. A placement succeeds only if every goal connection of the
placed brick exists with the goal footprint, was created by a contact event during this placement (not
authored), the brick is seated within tolerance, stayed after release and retreat, held the probe, and every
brick accepted earlier is still where it was and still connected.

Snapshot format (JSON-able)::

    {"step": int, "parts": {pid: {"p": [x, y, z], "q": [x, y, z, w]}},
     "conns": [{"stud": pid, "hole": pid, "offset": [i, j], "yaw": k, "utilization": u}], ...}
"""

from __future__ import annotations

import math

from .lego_clutch import STUD_HEIGHT_M, STUD_PITCH_M
from .lego_geometry import pose_error

SNAPSHOTS = ("before_place", "released", "retreated", "settled")


def footprint_cells(stud_lw, hole_lw, offset, yaw_q) -> frozenset[tuple[int, int]]:
    """Stud cells of the stud interface covered by the hole interface at (offset, quarter-turn yaw)."""
    L, W = hole_lw
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][int(yaw_q) % 4]
    x0, y0 = offset
    x1, y1 = x0 + c * L - s * W, y0 + s * L + c * W
    xs = range(max(0, min(x0, x1)), min(stud_lw[0], max(x0, x1)))
    ys = range(max(0, min(y0, y1)), min(stud_lw[1], max(y0, y1)))
    return frozenset((x, y) for x in xs for y in ys)


def _pose(snap: dict, pid: str):
    e = snap["parts"].get(pid)
    return None if e is None else (e["p"], e["q"])


def _yaw_period(lw) -> float:
    return math.pi / 2 if lw[0] == lw[1] else math.pi


def _conns_of_hole(snap: dict, pid: str) -> list[dict]:
    return [c for c in snap.get("conns", []) if c["hole"] == pid]


def _check(name: str, passed: bool | None, criterion: str, measured: dict) -> dict:
    return {"name": name, "passed": passed, "criterion": criterion, "measured": measured}


def evaluate_placement(record: dict, index: int, cfg: dict, accepted: dict[str, dict]) -> dict:
    """Verdict for ``record["steps"][index]``.

    ``accepted`` maps every earlier successfully placed part to its pose (``{"p", "q"}``) at acceptance; it is
    updated in place when this placement succeeds.
    """
    pc = cfg["postconditions"]
    step = record["steps"][index]
    pid = step["part"]
    parts = record["parts"]
    lw = tuple(parts[pid]["lw"])
    goal = record["goal_poses"][pid]
    goal_conns = [c for c in record["goal_connections"] if c["hole"] == pid]
    snaps = step.get("snapshots", {})
    out = {"part": pid, "index": index, "postconditions": [], "measured": {}}

    if step.get("skipped"):
        out.update(status="rejected", failure_type="precondition_unmet", phase="validate")
        return out
    missing = [s for s in SNAPSHOTS if s not in snaps]
    if missing:
        out.update(
            status="unverified", failure_type="evidence_missing", phase="observe", missing=missing
        )
        return out
    if step.get("timed_out"):
        out.update(status="failed", failure_type="timeout", phase=step.get("last_phase", "unknown"))
        return out

    settled = snaps["settled"]
    T_goal = (goal["p"], goal["q"])
    err = pose_error(T_goal, _pose(settled, pid), _yaw_period(lw))
    out["measured"]["settled_error"] = err

    # Earlier bricks: still where they were accepted and still connected.
    moved, lost = {}, []
    final_snaps = [settled] + ([snaps["probe_end"]] if "probe_end" in snaps else [])
    for other, pose in accepted.items():
        for snap in final_snaps:
            e = pose_error((pose["p"], pose["q"]), _pose(snap, other))
            if (
                e["dxy_m"] > pc["structure_move_tol_m"]
                or abs(e["dz_m"]) > pc["structure_move_tol_m"]
            ):
                moved[other] = max(moved.get(other, 0.0), math.hypot(e["dxy_m"], e["dz_m"]))
        for gc in record["goal_connections"]:
            if gc["hole"] == other and not _has_goal_conn(settled, gc, parts):
                lost.append(f"{gc['stud']}->{other}")
    lost += [
        f"{ev['stud']}->{ev['hole']}"
        for ev in step.get("events", [])
        if ev["kind"] == "disassembled" and ev["hole"] in accepted
    ]
    intact = not moved and not lost
    out["postconditions"].append(
        _check(
            "structure_intact",
            intact,
            "earlier bricks within structure_move_tol and still connected",
            {"moved_m": moved, "lost_connections": sorted(set(lost))},
        )
    )

    # Release: the brick did not leave with the gripper.
    z_rel = snaps["released"]["parts"][pid]["p"][2]
    z_ret = snaps["retreated"]["parts"][pid]["p"][2]
    rise = max(z_ret - z_rel, settled["parts"][pid]["p"][2] - z_rel)
    stayed = rise <= pc["lifted_z_m"]
    out["postconditions"].append(
        _check(
            "stays_after_release",
            stayed,
            "brick z rise after release <= lifted_z_m",
            {"rise_m": rise},
        )
    )

    # Support: goal parents (other than the plate) at their goal pose before placing.
    before = snaps["before_place"]
    absent = []
    for gc in goal_conns:
        parent = gc["stud"]
        if parts[parent].get("role") == "plate":
            continue
        g = record["goal_poses"][parent]
        e = pose_error(
            (g["p"], g["q"]), _pose(before, parent), _yaw_period(tuple(parts[parent]["lw"]))
        )
        if e["dxy_m"] > pc["structure_move_tol_m"] or abs(e["dz_m"]) > pc["structure_move_tol_m"]:
            absent.append(parent)
    supported = not absent
    out["postconditions"].append(
        _check(
            "parents_supported",
            supported,
            "every goal parent at its goal pose before placing",
            {"absent_parents": absent},
        )
    )

    # Clutch: engine connections of this brick against the goal footprint.
    have = _conns_of_hole(settled, pid)
    matched = [gc for gc in goal_conns if _has_goal_conn(settled, gc, parts)]
    all_matched = len(matched) == len(goal_conns) and bool(goal_conns)
    goal_cells = sum(gc_cells(gc, parts) for gc in goal_conns)
    have_cells = sum(len(_cells(c, parts)) for c in have)
    out["postconditions"].append(
        _check(
            "clutch_engaged_goal",
            all_matched,
            "engine connection for every goal stud/hole pair with the goal footprint",
            {"connections": have, "engaged_studs": have_cells, "goal_studs": goal_cells},
        )
    )
    triggered = {
        (ev["stud"], ev["hole"]) for ev in step.get("events", []) if ev["kind"] == "assembled"
    }
    by_contact = all((gc["stud"], pid) in triggered for gc in goal_conns)
    out["postconditions"].append(
        _check(
            "engaged_by_contact",
            by_contact,
            "assembly event raised by the contact trigger during this placement",
            {
                "assembled_events": sorted(f"{s}->{h}" for s, h in triggered),
                "max_press_force_n": step.get("max_press_force_n"),
            },
        )
    )
    seated = (
        err["dxy_m"] <= pc["seated_xy_tol_m"]
        and abs(err["dz_m"]) <= pc["seated_z_tol_m"]
        and err["tilt_rad"] <= math.radians(pc["seated_tilt_tol_deg"])
        and abs(err["yaw_rad"]) <= math.radians(pc["seated_yaw_tol_deg"])
    )
    out["postconditions"].append(
        _check("seated_pose", seated, "settled pose within seated tolerances", err)
    )

    probe = pc["probe"]
    if probe["enabled"] and "probe_end" in snaps:
        lift = snaps["probe_end"]["parts"][pid]["p"][2] - settled["parts"][pid]["p"][2]
        held = lift <= probe["lift_tol_m"]
        out["postconditions"].append(
            _check(
                "holds_under_probe",
                held,
                f"z rise under {probe['upward_force_n']} N upward <= lift_tol_m",
                {"lift_m": lift},
            )
        )
    else:
        held = None
        out["postconditions"].append(_check("holds_under_probe", None, "probe not run", {}))

    out["phase"] = "verify"
    if not intact:
        out.update(status="failed", failure_type="lego.structure_damaged")
    elif not stayed:
        out.update(status="failed", failure_type="lego.lifted_with_gripper")
    elif not supported:
        out.update(status="failed", failure_type="lego.insufficient_support")
    elif all_matched:
        if not by_contact:
            out.update(status="unverified", failure_type="evidence_missing")
        elif held is False:
            out.update(status="failed", failure_type="lego.not_seated")
        elif not seated:
            out.update(status="failed", failure_type="lego.partially_seated")
        else:
            out.update(status="succeeded", failure_type=None)
    elif have:
        goal_set = (
            set().union(*(_cells_of_goal(gc, parts) for gc in goal_conns)) if goal_conns else set()
        )
        have_set = set().union(*(_cells_keyed(c, parts) for c in have))
        partial = bool(goal_set & have_set) and have_cells < goal_cells
        out.update(
            status="failed", failure_type="lego.partially_seated" if partial else "lego.misaligned"
        )
    else:
        near = (
            err["dxy_m"] <= STUD_PITCH_M / 2
            and abs(err["yaw_rad"]) <= math.radians(10.0)
            and abs(err["dz_m"]) <= STUD_HEIGHT_M + pc["seated_z_tol_m"]
        )
        out.update(status="failed", failure_type="lego.not_seated" if near else "lego.misaligned")
    if out["status"] == "succeeded":
        accepted[pid] = settled["parts"][pid]
    return out


def _cells(conn: dict, parts: dict) -> frozenset:
    return footprint_cells(
        tuple(parts[conn["stud"]]["lw"]),
        tuple(parts[conn["hole"]]["lw"]),
        tuple(conn["offset"]),
        conn["yaw"],
    )


def _cells_keyed(conn: dict, parts: dict) -> set:
    return {(conn["stud"], c) for c in _cells(conn, parts)}


def _cells_of_goal(gc: dict, parts: dict) -> set:
    return _cells_keyed(gc, parts)


def gc_cells(gc: dict, parts: dict) -> int:
    return len(_cells(gc, parts))


def _has_goal_conn(snap: dict, gc: dict, parts: dict) -> bool:
    want = _cells(gc, parts)
    return any(
        c["stud"] == gc["stud"] and c["hole"] == gc["hole"] and _cells(c, parts) == want
        for c in snap.get("conns", [])
    )


def evaluate_episode(record: dict, cfg: dict) -> dict:
    """Per-placement verdicts and the episode outcome (first failing placement decides the failure type)."""
    accepted: dict[str, dict] = {
        pid: record["start_poses"][pid]
        for pid, p in record["parts"].items()
        if p.get("role") == "preplaced"
    }
    verdicts = [evaluate_placement(record, i, cfg, accepted) for i in range(len(record["steps"]))]
    first_bad = next((v for v in verdicts if v["status"] != "succeeded"), None)
    final = record.get("final_snapshot")
    final_ok = final is not None and all(
        _has_goal_conn(final, gc, record["parts"]) for gc in record["goal_connections"]
    )
    if first_bad is None and final_ok:
        outcome = {"status": "succeeded", "failure_type": None}
    elif first_bad is None:
        outcome = {
            "status": "failed" if final is not None else "unverified",
            "failure_type": "postcondition_failed" if final is not None else "evidence_missing",
        }
    else:
        outcome = {
            "status": first_bad["status"],
            "failure_type": first_bad["failure_type"],
            "failed_part": first_bad["part"],
        }
    outcome["placements_succeeded"] = sum(v["status"] == "succeeded" for v in verdicts)
    outcome["placements"] = len(verdicts)
    outcome["final_structure_complete"] = final_ok
    return {"outcome": outcome, "placements": verdicts}
