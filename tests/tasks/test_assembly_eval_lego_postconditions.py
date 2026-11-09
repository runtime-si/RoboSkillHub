"""Failure attribution of the LEGO evaluator on synthetic snapshots (logic only, no physics)."""

from __future__ import annotations

import copy
import math

import pytest

from roboskillhub.tasks.assembly_eval.lego_episode import load_config
from roboskillhub.tasks.assembly_eval.lego_geometry import q_from_axis_angle
from roboskillhub.tasks.assembly_eval.lego_postconditions import (
    evaluate_episode,
    evaluate_placement,
    footprint_cells,
)

CFG = load_config()
IDENT = [0.0, 0.0, 0.0, 1.0]
BRICK = 0.0096


def pose(x=0.0, y=0.0, z=0.0, q=IDENT):
    return {"p": [x, y, z], "q": list(q)}


def snap(parts, conns=()):
    return {"step": 0, "parts": copy.deepcopy(parts), "conns": [dict(c) for c in conns]}


GOAL = {"0": pose(), "1": pose(z=0.0032)}
PLATE_CONN = {"stud": "0", "hole": "1", "offset": [0, 0], "yaw": 0}


def record(
    settled_parts,
    conns,
    events,
    *,
    released=None,
    retreated=None,
    before=None,
    probe=None,
    goal_conns=None,
):
    released = released or settled_parts
    snaps = {
        "before_place": snap(before or GOAL),
        "released": snap(released, conns),
        "retreated": snap(retreated or released, conns),
        "settled": snap(settled_parts, conns),
    }
    if probe is not None:
        snaps["probe_end"] = snap(probe, conns)
    return {
        "parts": {"0": {"lw": [32, 32], "role": "plate"}, "1": {"lw": [4, 2], "role": "to_place"}},
        "goal_poses": GOAL,
        "start_poses": GOAL,
        "goal_connections": [goal_conns or PLATE_CONN],
        "steps": [{"part": "1", "snapshots": snaps, "events": events, "max_press_force_n": 3.0}],
        "final_snapshot": snap(settled_parts, conns),
    }


ASSEMBLED = [{"kind": "assembled", "stud": "0", "hole": "1", "offset": [0, 0], "yaw": 0}]


def verdict(rec):
    return evaluate_placement(rec, 0, CFG, {})


def test_engaged_seated_and_held_succeeds():
    v = verdict(record(GOAL, [PLATE_CONN], ASSEMBLED, probe=GOAL))
    assert v["status"] == "succeeded", v
    assert v["failure_type"] is None
    assert all(c["passed"] for c in v["postconditions"])


def test_connection_without_contact_event_is_not_success():
    v = verdict(record(GOAL, [PLATE_CONN], [], probe=GOAL))
    assert v["status"] == "unverified"
    assert v["failure_type"] == "evidence_missing"


def test_arrived_on_seat_without_clutch_is_not_seated():
    resting = {"0": pose(), "1": pose(z=0.0032)}
    lifted_by_probe = {"0": pose(), "1": pose(z=0.0032 + 0.004)}
    v = verdict(record(resting, [], [], probe=lifted_by_probe))
    assert v["status"] == "failed"
    assert v["failure_type"] == "lego.not_seated"
    seated = next(c for c in v["postconditions"] if c["name"] == "seated_pose")
    assert seated["passed"], "geometry alone says seated; only the clutch and probe say otherwise"


def test_connection_that_fails_the_probe_is_not_seated():
    up = {"0": pose(), "1": pose(z=0.0032 + 0.003)}
    v = verdict(record(GOAL, [PLATE_CONN], ASSEMBLED, probe=up))
    assert v["failure_type"] == "lego.not_seated"


def test_far_off_brick_is_misaligned():
    off = {
        "0": pose(),
        "1": pose(x=0.006, z=0.005, q=q_from_axis_angle([0, 0, 1], math.radians(12))),
    }
    v = verdict(record(off, [], [], probe=off))
    assert v["failure_type"] == "lego.misaligned"


def test_one_stud_shift_is_partial_engagement():
    shifted = {"0": pose(), "1": pose(y=0.008, z=0.0032)}
    goal_conn = {"stud": "2", "hole": "1", "offset": [0, 0], "yaw": 0}
    rec = record(
        shifted,
        [{**goal_conn, "offset": [0, 1]}],
        [{**ASSEMBLED[0], "stud": "2", "offset": [0, 1]}],
        probe=shifted,
        goal_conns=goal_conn,
    )
    rec["parts"]["2"] = {"lw": [2, 4], "role": "to_place"}
    rec["parts"]["1"]["lw"] = [2, 4]
    rec["goal_poses"] = {**GOAL, "2": pose()}
    rec["steps"][0]["snapshots"]["before_place"]["parts"]["2"] = pose()
    v = verdict(rec)
    assert v["failure_type"] == "lego.partially_seated"
    engaged = next(c for c in v["postconditions"] if c["name"] == "clutch_engaged_goal")["measured"]
    assert (engaged["engaged_studs"], engaged["goal_studs"]) == (6, 8)


def test_brick_carried_up_by_the_gripper_is_lifted():
    up = {"0": pose(), "1": pose(z=0.03)}
    v = verdict(record(up, [], [], released=GOAL, retreated=up, probe=up))
    assert v["failure_type"] == "lego.lifted_with_gripper"


def test_missing_parent_is_insufficient_support():
    goal_conn = {"stud": "2", "hole": "1", "offset": [0, 0], "yaw": 0}
    dropped = {"0": pose(), "1": pose(z=0.0032), "2": pose(x=0.1)}
    rec = record(dropped, [], [], before=dropped, probe=dropped, goal_conns=goal_conn)
    rec["parts"]["2"] = {"lw": [4, 2], "role": "to_place"}
    rec["goal_poses"] = {"0": pose(), "1": pose(z=0.0032 + BRICK), "2": pose(z=0.0032)}
    v = verdict(rec)
    assert v["failure_type"] == "lego.insufficient_support"


def test_knocked_earlier_brick_is_structure_damage():
    rec = record(GOAL, [PLATE_CONN], ASSEMBLED, probe=GOAL)
    rec["parts"]["3"] = {"lw": [4, 2], "role": "to_place"}
    for s in rec["steps"][0]["snapshots"].values():
        s["parts"]["3"] = pose(x=0.04, z=0.0032)
    accepted = {"3": pose(x=0.016, z=0.0032)}
    v = evaluate_placement(rec, 0, CFG, accepted)
    assert v["failure_type"] == "lego.structure_damaged"


def test_missing_snapshot_is_unverified():
    rec = record(GOAL, [PLATE_CONN], ASSEMBLED, probe=GOAL)
    del rec["steps"][0]["snapshots"]["released"]
    v = verdict(rec)
    assert (v["status"], v["failure_type"]) == ("unverified", "evidence_missing")


def test_episode_outcome_needs_final_structure():
    rec = record(GOAL, [PLATE_CONN], ASSEMBLED, probe=GOAL)
    assert evaluate_episode(rec, CFG)["outcome"]["status"] == "succeeded"
    rec["final_snapshot"]["conns"] = []
    out = evaluate_episode(rec, CFG)["outcome"]
    assert (out["status"], out["failure_type"]) == ("failed", "postcondition_failed")


@pytest.mark.parametrize("yaw_q", [0, 1, 2, 3])
def test_footprint_cells_count(yaw_q):
    cells = footprint_cells((32, 32), (4, 2), (10, 10), yaw_q)
    assert len(cells) == 8
