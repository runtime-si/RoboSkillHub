"""place_upright composed against a synthetic kinematic world (tier: synthetic_contract_test).

The world's faults change where tiles end up; the executor always acknowledges
``ok: true``. Each negative checks the status, the observed fault, where it was seen,
which commands were (not) issued, the cause hypothesis and the bounded recovery plan.
"""

from __future__ import annotations

import math

import pytest
from _domino_world import FakeClock, SyntheticStation, request

from roboskillhub.tasks.domino import Clock, place_upright
from roboskillhub.tasks.domino.contract import CAPABILITIES


def run(faults=(), clock=None, cancel=None, **kw):
    clock = clock or FakeClock()
    world_kw = {
        k: kw.pop(k) for k in ("grasp_dx", "table_z", "capabilities", "neighbors") if k in kw
    }
    world = SyntheticStation(clock, faults=set(faults), **world_kw)
    out = place_upright(
        request(clock, **kw), world, clock=Clock(clock.wall, clock.mono, clock.sleep), cancel=cancel
    )
    return out, world


def causes(out):
    return {h["cause"]: h["confidence"] for h in out["cause_hypotheses"]}


def test_happy_path_succeeds_only_through_the_observer():
    out, world = run()
    assert out["status"] == "succeeded", out["failure"]
    assert world.calls == ["begin", "move:free", "move:linear", "release", "move:linear", "end"]
    v = out["evidence"]["verification"]
    assert (
        v["status"] == "succeeded" and v["evidence"]["evidence_tier"] == "synthetic_contract_test"
    )
    assert v["evidence"]["target"]["final"]["position_error_m"] < 1e-3
    assert out["recovery"]["action"] == "none"
    assert "support_contact_not_sensed_before_release" in out["unverified"]
    phases = [e["phase"] for e in out["phase_trace"]]
    assert phases.index("release") < phases.index("retreat") < phases.index("settle")


def test_missing_release_capability_rejects_before_any_command():
    caps = CAPABILITIES["geometric"] - {"gripper_release", "linear_cartesian_motion"}
    out, world = run(capabilities=caps)
    assert out["status"] == "rejected" and out["failure"]["code"] == "capability_missing"
    assert "gripper_release" in out["failure"]["detail"]
    assert world.calls == []


def test_contact_mode_requires_guarded_descent():
    out, world = run(mode="contact_feedback")
    assert out["status"] == "rejected" and "guarded_descent" in out["failure"]["detail"]
    out, world = run(mode="contact_feedback", capabilities=CAPABILITIES["contact_feedback"])
    assert out["status"] == "succeeded"


def test_missed_grasp_rejected_before_motion():
    # The TCP is wherever it is; the tile lies on the table elsewhere.
    out, world = run(faults={"missed_grasp"})
    assert out["status"] == "rejected" and out["failure"]["code"] == "holding_not_confirmed"
    assert not any(c.startswith("move") or c == "release" for c in world.calls)
    assert out["recovery"]["action"] == "regrasp"


def test_grasp_offset_beyond_tolerance_is_caught_before_motion():
    out, world = run(grasp_dx=0.010)
    assert out["status"] == "rejected" and out["failure"]["code"] == "holding_not_confirmed"
    assert not any(c.startswith("move") for c in world.calls)


def test_drop_in_transit_fails_and_does_not_release():
    out, world = run(faults={"drop_in_transit"})
    assert out["status"] == "failed" and out["failure"]["code"] == "object_dropped"
    assert "release" not in world.calls and "stop" in world.calls
    assert causes(out)["slip"] == "medium"
    assert out["recovery"]["action"] == "regrasp"


def test_wrong_release_pose_is_not_released():
    # The executor acknowledges the descent but arrives 12 mm off.
    out, world = run(faults={"tcp_offset"})
    assert out["status"] == "failed" and out["failure"] == {
        **out["failure"],
        "code": "placement_error",
        "phase": "descent",
    }
    assert "release" not in world.calls
    assert causes(out)["tcp_tracking"] == "high"
    assert out["recovery"]["action"] == "retreat_and_retry"


def test_knocked_over_at_release():
    out, world = run(faults={"fall_on_release"})
    assert out["status"] == "failed"
    assert (
        out["failure"]["code"] == "unstable_after_release" and out["failure"]["phase"] == "release"
    )
    assert "release_drag" in causes(out)
    assert out["recovery"]["action"] == "recover_fallen_tile" and out["recovery"]["affected"] == [
        "d2"
    ]
    assert "agent01.stage_and_regrasp" in out["recovery"]["requires_capabilities"]


def test_retreat_sweeps_a_neighbour():
    out, world = run(faults={"retreat_knocks"})
    assert out["status"] == "failed"
    f = out["failure"]
    assert (
        f["code"] == "neighbor_disturbed" and f["neighbor_id"] == "d1" and f["phase"] == "retreat"
    )
    assert causes(out) == {"retreat_contact": "medium"}
    assert out["recovery"]["action"] == "restore_segment" and out["recovery"]["affected"] == ["d1"]


def test_table_higher_than_calibrated_blocks_the_descent():
    out, world = run(table_z=0.010)
    assert out["status"] == "failed" and out["failure"]["code"] == "descent_blocked"
    assert "release" not in world.calls
    assert "table_calibration" in causes(out)


def test_table_lower_than_calibrated_seen_before_release():
    out, world = run(faults={"observe_support"}, table_z=-0.012)
    assert out["status"] == "failed" and out["failure"]["code"] == "support_mismatch"
    assert "release" not in world.calls
    assert causes(out)["table_calibration"] == "high"


def test_table_lower_than_calibrated_unsensed_drops_the_tile():
    out, world = run(table_z=-0.012)
    assert out["status"] == "failed" and out["failure"]["code"] == "unstable_after_release"


def test_stale_observation_after_release_is_unverified():
    out, world = run(faults={"stale_obs"})
    assert out["status"] == "unverified" and out["failure"]["code"] == "observation_stale"
    assert "stop" in world.calls and world.calls[-1] == "end"
    assert out["recovery"]["action"] == "reobserve"
    assert all(s["capability"] != "domino.place_upright" for s in out["recovery"]["steps"])


def test_repeated_frames_cannot_fill_the_settle_window():
    out, world = run(faults={"repeated_frames"})
    assert out["status"] == "unverified"
    assert out["failure"]["code"] in ("observation_stale", "insufficient_settle_window")


def test_executor_finished_but_tile_fell_later():
    out, world = run(faults={"fall_in_settle"})
    assert all(e.get("ack_ok", True) for e in out["phase_trace"])
    assert out["status"] == "failed" and out["failure"]["code"] == "unstable_after_release"


def test_gripper_never_let_go():
    out, world = run(faults={"still_holding"})
    assert out["status"] == "failed" and out["failure"]["code"] == "release_not_confirmed"
    assert out["recovery"]["action"] == "release_again"


def test_orientation_missing_or_identity_lost_after_release():
    out, _ = run(faults={"no_orientation"})
    assert out["status"] == "unverified" and out["failure"]["code"] == "orientation_missing"
    out, _ = run(faults={"wrong_identity"})
    assert out["status"] == "unverified" and out["failure"]["code"] == "identity_mismatch"


def test_deadline_covers_the_whole_sequence():
    out, world = run(timeout_s=12.0)
    assert out["status"] == "failed" and out["failure"]["code"] == "timeout"
    assert "stop" in world.calls and world.calls[-1] == "end"
    assert out["recovery"]["action"] == "abort"


def test_cancel_after_motion_is_unverified_and_stops():
    clock = FakeClock()
    out, world = run(clock=clock, cancel=lambda: clock.t > 1.0)
    assert out["status"] == "unverified" and out["failure"]["code"] == "cancelled"
    assert "stop" in world.calls and "release" not in world.calls


def test_backend_exception_is_failure_not_success():
    clock = FakeClock()
    world = SyntheticStation(clock)

    def boom(*a, **k):
        raise ConnectionError("driver went away")

    world.release = boom
    out = place_upright(request(clock), world, clock=Clock(clock.wall, clock.mono, clock.sleep))
    assert out["status"] == "failed" and out["failure"]["code"] == "backend_error"
    assert "stop" in world.calls


@pytest.mark.parametrize(
    "over, code",
    [
        ({"target_id": "d9"}, "holding_not_confirmed"),
        ({"frame_id": "base"}, "invalid_request"),
        ({"thresholds": None}, "invalid_request"),
        ({"release": {"width_m": 0.02}}, "infeasible_geometry"),
        ({"release": {"width_m": 0.2}}, "infeasible_geometry"),
        ({"workspace": {"min_m": [0, -0.5, 0], "max_m": [0.8, 0.5, 0.06]}}, "infeasible_geometry"),
        ({"retreat": {"direction": [0, -0.8, 0.6], "distance_m": 0.06}}, "retreat_blocked"),
        ({"approach": {"direction": [0, 0.6, -0.8], "distance_m": 0.06}}, "neighbor_clearance"),
    ],
)
def test_offline_checks_reject_before_any_command(over, code):
    out, world = run(**over)
    assert out["status"] == "rejected" and out["failure"]["code"] == code, out["failure"]
    assert world.calls == []


def test_neighbour_too_close_for_the_fingers():
    clock = FakeClock()
    r = request(clock)
    r["neighbors"][1]["pose"]["position_m"][1] = 0.068
    out = place_upright(
        r, SyntheticStation(clock), clock=Clock(clock.wall, clock.mono, clock.sleep)
    )
    assert out["status"] == "rejected" and out["failure"]["code"] == "neighbor_clearance"


def test_stale_neighbour_or_holding_evidence_rejected():
    clock = FakeClock()
    r = request(clock)
    r["neighbors"][0]["timestamp_s"] -= 10
    out = place_upright(
        r, SyntheticStation(clock), clock=Clock(clock.wall, clock.mono, clock.sleep)
    )
    assert out["failure"]["code"] == "observation_stale"
    r = request(clock)
    r["holding"]["timestamp_s"] -= 10
    out = place_upright(
        r, SyntheticStation(clock), clock=Clock(clock.wall, clock.mono, clock.sleep)
    )
    assert out["failure"]["code"] == "observation_stale"


def test_tilted_target_rejected():
    clock = FakeClock()
    r = request(clock)
    r["target_object_pose"]["quaternion_xyzw"] = [math.sin(0.2), 0.0, 0.0, math.cos(0.2)]
    out = place_upright(
        r, SyntheticStation(clock), clock=Clock(clock.wall, clock.mono, clock.sleep)
    )
    assert out["status"] == "rejected" and out["failure"]["code"] == "infeasible_geometry"


def test_recovery_budget_is_bounded():
    out, _ = run(faults={"fall_on_release"}, recovery_attempts_used=2, max_recovery_attempts=2)
    assert out["recovery"]["action"] == "abort" and out["recovery"]["steps"] == []


def test_slanted_retreat_toward_the_free_end_is_allowed():
    out, world = run(retreat={"direction": [0, 0.6, 0.8], "distance_m": 0.06})
    assert out["status"] == "succeeded", out["failure"]
