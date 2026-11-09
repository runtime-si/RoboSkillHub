"""place_upright composed through a fake rsh-task/1 task context over a synthetic
kinematic world (tier: synthetic_contract_test).

The world's faults change where tiles end up; every primitive the context runs reports
its own (runtime-level) result. Each negative checks the status, the observed fault and
where it was seen, which primitives were (not) issued, the cause hypothesis and the
bounded recovery plan / hints.
"""

from __future__ import annotations

import math

import pytest
from _domino_world import FakeTaskContext, request

from roboskillhub.tasks.domino import SKILLS, place_upright
from roboskillhub.tasks.domino.contract import CAPABILITIES


def run(faults=(), **kw):
    ctx_kw = {
        k: kw.pop(k) for k in ("grasp_dx", "table_z", "capabilities", "cancel_after_s") if k in kw
    }
    ctx = FakeTaskContext(faults=set(faults), **ctx_kw)
    return place_upright(request(ctx, **kw), ctx), ctx


def motions(ctx):
    return [c for c in ctx.calls if c != "observe_objects"]


def causes(out):
    return {h["cause"]: h["confidence"] for h in out["cause_hypotheses"]}


def test_happy_path_succeeds_only_through_the_observer():
    out, ctx = run()
    assert out["status"] == "succeeded", out["failure"]
    assert motions(ctx) == ["move_waypoints", "move_waypoints", "set_gripper", "move_waypoints"]
    assert ctx.calls[0] == "observe_objects"
    assert out["contract_version"] == "rsh-task/1" and out["skill"] == "domino.place_upright"
    pc = out["evidence"]["postcondition"]
    assert pc["passed"] and pc["measured"]["position_error_m"] < 1e-3
    assert pc["thresholds_ref"] == "tasks/domino/config/line_v1.yaml"
    assert out["observations"] and out["evidence"]["evidence_tier"] == "synthetic_contract_test"
    assert (
        len(out["children"]) == len(ctx.calls)
        and out["failure_type"] is None
        and out["recovery_hints"] == []
    )
    assert "support_contact_not_sensed_before_release" in out["unverified"]
    assert (
        "descent_and_retreat_are_waypoint_sequences_not_linear_interpolation" in out["unverified"]
    )
    phases = [e["phase"] for e in out["trace"]]
    assert phases.index("release") < phases.index("retreat") < phases.index("settle")


def test_registered_skill_entry_runs_the_same():
    entry = next(s for s in SKILLS if s.skill_id == "domino.place_upright")
    assert entry.kind == "skill" and entry.requires == CAPABILITIES["geometric"]
    assert "gripper" in entry.requires and "object_pose_feedback" in entry.requires
    ctx = FakeTaskContext()
    assert entry.run(request(ctx), ctx)["status"] == "succeeded"


def test_missing_orientation_feedback_rejects_before_any_command():
    out, ctx = run(capabilities=CAPABILITIES["geometric"] - {"object_pose_feedback"})
    assert out["status"] == "rejected" and out["failure_type"] == "capability_missing"
    assert motions(ctx) == []


def test_linear_descent_needs_linear_cartesian_motion():
    out, ctx = run(descent_path="linear")
    assert out["status"] == "rejected" and "linear_cartesian_motion" in out["failure"]["detail"]
    caps = CAPABILITIES["geometric"] | {"linear_cartesian_motion"}
    out, _ = run(descent_path="linear", capabilities=caps)
    assert out["status"] == "succeeded"
    assert (
        "descent_and_retreat_are_waypoint_sequences_not_linear_interpolation"
        not in out["unverified"]
    )


CONTACT_CAPS = CAPABILITIES["contact_feedback"]


def test_contact_mode_needs_force_feedback_and_compliance():
    out, ctx = run(mode="contact_feedback", contact={"force_limit_N": 2.0})
    assert out["status"] == "rejected" and out["failure_type"] == "capability_missing"
    assert {"contact_force_feedback", "compliant_motion"} <= set(
        out["requires"] if "requires" in out else CONTACT_CAPS
    )
    assert motions(ctx) == []


def test_contact_mode_confirms_the_support_by_guarded_press():
    out, ctx = run(
        mode="contact_feedback", contact={"force_limit_N": 2.0}, capabilities=CONTACT_CAPS
    )
    assert out["status"] == "succeeded", out["failure"]
    assert motions(ctx) == [
        "move_waypoints",
        "move_waypoints",
        "guarded_press",
        "set_gripper",
        "move_waypoints",
    ]
    press = next(r for c, r in ctx.requests if c == "guarded_press")
    assert press["axis"]["vector"] == [0.0, 0.0, -1.0] and press["force_limit_N"] == 2.0
    assert abs(out["evidence"]["support_contact"]["offset_m"]) < 1e-3
    assert "support_contact_not_sensed_before_release" not in out["unverified"]


@pytest.mark.parametrize("table_z", [-0.012, 0.010])
def test_contact_mode_refuses_to_release_off_the_calibrated_support(table_z):
    out, ctx = run(
        mode="contact_feedback",
        contact={"force_limit_N": 2.0},
        capabilities=CONTACT_CAPS,
        table_z=table_z,
    )
    assert out["status"] == "failed" and out["failure_type"] == "domino.support_mismatch"
    assert "set_gripper" not in ctx.calls
    assert "table_calibration" in causes(out)


@pytest.mark.parametrize(
    "limits, over, code",
    [
        ({"max_speed_m_s": 0.05}, {}, "invalid_request"),
        ({"gripper_min_width_m": 0.0, "gripper_max_width_m": 0.03}, {}, "infeasible_geometry"),
        (
            {"workspace_min_m": (0.0, -0.5, 0.0), "workspace_max_m": (0.8, 0.5, 0.08)},
            {},
            "infeasible_geometry",
        ),
        (
            {"max_contact_force_N": 0.0},
            {"mode": "contact_feedback", "contact": {"force_limit_N": 2.0}},
            "capability_missing",
        ),
        (
            {"max_contact_force_N": 1.0},
            {"mode": "contact_feedback", "contact": {"force_limit_N": 2.0}},
            "invalid_request",
        ),
        ({"frame_id": "other"}, {}, "invalid_request"),
    ],
)
def test_station_limits_reject_before_any_command(limits, over, code):
    ctx = FakeTaskContext(capabilities=CONTACT_CAPS)
    ctx.limits = limits
    out = place_upright(request(ctx, **over), ctx)
    assert out["status"] == "rejected" and out["failure"]["code"] == code, out["failure"]
    assert ctx.calls == []


def test_primitive_timeout_follows_the_station_limit():
    ctx = FakeTaskContext()
    ctx.limits = {"max_timeout_s": 7.0}
    out = place_upright(request(ctx), ctx)
    assert out["status"] == "succeeded"
    assert all(r["timeout_s"] <= 7.0 for _, r in ctx.requests)


def test_missed_grasp_rejected_before_motion():
    out, ctx = run(faults={"missed_grasp"})
    assert out["status"] == "rejected" and out["failure"]["code"] == "holding_not_confirmed"
    assert motions(ctx) == []
    assert out["recovery"]["action"] == "regrasp" and "regrasp" in out["recovery_hints"]


def test_missed_grasp_without_gripper_feedback_still_caught_by_observation():
    out, ctx = run(faults={"missed_grasp", "gripper_unknown"})
    assert out["status"] == "rejected" and out["failure"]["code"] == "holding_not_confirmed"
    assert motions(ctx) == []


def test_grasp_offset_beyond_tolerance_is_caught_before_motion():
    out, ctx = run(grasp_dx=0.010)
    assert out["status"] == "rejected" and out["failure"]["code"] == "holding_not_confirmed"
    assert motions(ctx) == []


def test_drop_in_transit_fails_and_does_not_release():
    out, ctx = run(faults={"drop_in_transit"})
    assert out["status"] == "failed" and out["failure_type"] == "held_object_slipped"
    assert out["failure"]["code"] == "object_dropped" and out["phase"] == "transport"
    assert "set_gripper" not in ctx.calls
    assert causes(out)["slip"] == "medium"
    assert out["recovery"]["action"] == "regrasp"


def test_wrong_release_pose_is_not_released():
    # The descent ends 12 mm off; the tile is never let go there.
    out, ctx = run(faults={"tcp_offset"})
    assert out["status"] == "failed" and out["failure_type"] == "motion_not_converged"
    assert out["failure"]["phase"] == "descent" and out["phase"] == "descend"
    assert "set_gripper" not in ctx.calls
    assert causes(out)["tcp_tracking"] == "high"
    assert (
        out["recovery"]["action"] == "retreat_and_retry" and out["recovery_hints"][0] == "back_off"
    )


def test_knocked_over_at_release():
    out, ctx = run(faults={"fall_on_release"})
    assert out["status"] == "failed" and out["failure_type"] == "domino.toppled_after_release"
    assert out["failure"]["phase"] == "release"
    assert "release_drag" in causes(out)
    assert out["recovery"]["action"] == "recover_fallen_tile" and out["recovery"]["affected"] == [
        "d2"
    ]
    assert "agent01.stage_and_regrasp" in out["recovery"]["requires_capabilities"]
    assert out["recovery_hints"] == ["reobserve", "reset_piece"]


def test_retreat_sweeps_a_neighbour():
    out, ctx = run(faults={"retreat_knocks"})
    assert out["status"] == "failed" and out["failure_type"] == "domino.neighbour_disturbed"
    f = out["failure"]
    assert f["neighbor_id"] == "d1" and f["phase"] == "retreat"
    assert causes(out) == {"retreat_contact": "medium"}
    assert out["recovery"]["action"] == "restore_segment" and out["recovery"]["affected"] == ["d1"]


def test_table_higher_than_calibrated_blocks_the_descent():
    out, ctx = run(table_z=0.010)
    assert out["status"] == "failed" and out["failure"]["code"] == "descent_blocked"
    assert "set_gripper" not in ctx.calls
    assert "table_calibration" in causes(out)


def test_table_lower_than_calibrated_seen_before_release():
    ctx = FakeTaskContext(table_z=-0.012)
    r = request(ctx)
    r["support"]["observation_id"] = "table"
    out = place_upright(r, ctx)
    assert out["status"] == "failed" and out["failure_type"] == "domino.support_mismatch"
    assert "set_gripper" not in ctx.calls
    assert causes(out)["table_calibration"] == "high"


def test_table_lower_than_calibrated_unsensed_drops_the_tile():
    out, ctx = run(table_z=-0.012)
    assert out["status"] == "failed" and out["failure_type"] == "domino.toppled_after_release"


def test_stale_observation_after_release_is_unverified():
    out, ctx = run(faults={"stale_obs"})
    assert out["status"] == "unverified" and out["failure_type"] == "stale_observation"
    assert out["recovery"]["action"] == "reobserve"
    assert all(s["capability"] != "domino.place_upright" for s in out["recovery"]["steps"])


def test_repeated_frames_cannot_fill_the_settle_window():
    out, ctx = run(faults={"repeated_frames"})
    assert out["status"] == "unverified"
    assert out["failure_type"] in ("stale_observation", "evidence_missing")


def test_executor_finished_but_tile_fell_later():
    out, ctx = run(faults={"fall_in_settle"})
    assert all(e.get("status", "succeeded") == "succeeded" for e in out["trace"])
    assert out["status"] == "failed" and out["failure_type"] == "domino.toppled_after_release"
    assert out["failure"]["phase"] == "settle"


def test_gripper_reports_release_failed():
    out, ctx = run(faults={"still_holding"})
    assert out["status"] == "failed" and out["failure_type"] == "release_failed"
    assert out["recovery"]["action"] == "release_again"


def test_gripper_says_open_but_tile_still_carried():
    out, ctx = run(faults={"sticky_release"})
    assert out["status"] == "failed" and out["failure_type"] == "domino.not_resting_on_support"


def test_orientation_missing_or_identity_lost_after_release():
    out, _ = run(faults={"no_orientation"})
    assert out["status"] == "unverified" and out["failure"]["code"] == "orientation_missing"
    out, _ = run(faults={"wrong_identity"})
    assert out["status"] == "unverified" and out["failure_type"] == "perception.low_confidence"


def test_deadline_covers_the_whole_sequence():
    out, ctx = run(timeout_s=15.0)
    assert out["status"] == "failed" and out["failure_type"] == "timeout"
    assert out["recovery"]["action"] == "abort"


def test_cancel_after_motion_is_unverified():
    out, ctx = run(cancel_after_s=1.0)
    assert out["status"] == "unverified" and out["failure_type"] == "cancelled"
    assert "set_gripper" not in ctx.calls


def test_context_exception_is_failure_not_success():
    ctx = FakeTaskContext()
    real = ctx.invoke

    def boom(capability, req):
        if capability == "set_gripper":
            raise ConnectionError("driver went away")
        return real(capability, req)

    ctx.invoke = boom
    out = place_upright(request(ctx), ctx)
    assert out["status"] == "failed" and out["failure_type"] == "backend_error"


@pytest.mark.parametrize(
    "over, code",
    [
        ({"target_id": "d9"}, "holding_not_confirmed"),
        ({"frame_id": "base"}, "invalid_request"),
        ({"tolerances_ref": "/etc/passwd"}, "invalid_request"),
        ({"thresholds_override": {"position_tolerance_m": 0.05}}, "invalid_request"),
        ({"thresholds_override": {"release_gap_m": 0.0}}, "invalid_request"),
        ({"release": {"width_m": 0.02}}, "infeasible_geometry"),
        ({"release": {"width_m": 0.2}}, "infeasible_geometry"),
        ({"workspace": {"min_m": [0, -0.5, 0], "max_m": [0.8, 0.5, 0.06]}}, "infeasible_geometry"),
        ({"retreat": {"direction": [0, -0.8, 0.6], "distance_m": 0.06}}, "retreat_blocked"),
        ({"approach": {"direction": [0, 0.6, -0.8], "distance_m": 0.09}}, "neighbor_clearance"),
        ({"approach": {"distance_m": 0.05}}, "neighbor_clearance"),
    ],
)
def test_offline_checks_reject_before_any_command(over, code):
    out, ctx = run(**over)
    assert out["status"] == "rejected" and out["failure"]["code"] == code, out["failure"]
    assert motions(ctx) == []


def test_tighter_override_is_accepted():
    out, _ = run(thresholds_override={"position_tolerance_m": 0.004, "settle_window_s": 0.8})
    assert out["status"] == "succeeded"


def test_slanted_retreat_toward_the_free_end_is_allowed():
    out, _ = run(retreat={"direction": [0, 0.6, 0.8], "distance_m": 0.06})
    assert out["status"] == "succeeded", out["failure"]


def test_neighbour_too_close_for_the_fingers():
    ctx = FakeTaskContext()
    r = request(ctx)
    r["neighbors"][1]["position_m"][1] = 0.068
    out = place_upright(r, ctx)
    assert out["status"] == "rejected" and out["failure"]["code"] == "neighbor_clearance"


def test_stale_or_orientationless_neighbour_or_holding_evidence_rejected():
    ctx = FakeTaskContext()
    r = request(ctx)
    r["neighbors"][0]["timestamp_s"] -= 10
    assert place_upright(r, ctx)["failure_type"] == "stale_observation"
    r = request(ctx)
    r["neighbors"][0]["quaternion_xyzw"] = None
    assert place_upright(r, ctx)["failure"]["code"] == "invalid_request"
    r = request(ctx)
    r["held"]["timestamp_s"] -= 10
    assert place_upright(r, ctx)["failure_type"] == "stale_observation"
    r = request(ctx)
    r["held"]["valid"], r["held"]["invalidated_reason"] = False, "slip_detected"
    assert place_upright(r, ctx)["failure_type"] == "precondition_unmet"


def test_tilted_or_floating_target_rejected():
    ctx = FakeTaskContext()
    r = request(ctx)
    r["target_object_pose"]["quaternion_xyzw"] = [math.sin(0.2), 0.0, 0.0, math.cos(0.2)]
    assert place_upright(r, ctx)["failure"]["code"] == "infeasible_geometry"
    r = request(ctx)
    r["target_object_pose"]["position_m"][2] = 0.0  # bottom-face height given for a centre origin
    out = place_upright(r, ctx)
    assert (
        out["failure"]["code"] == "infeasible_geometry"
        and "geometric_center" in out["failure"]["detail"]
    )


def test_recovery_budget_is_bounded():
    out, _ = run(faults={"fall_on_release"}, recovery_attempts_used=2, max_recovery_attempts=2)
    assert out["recovery"]["action"] == "abort" and out["recovery"]["steps"] == []


def test_carry_rises_before_crossing_and_keeps_transit_clearance():
    ctx = FakeTaskContext()
    ctx.tcp["position_m"] = [0.30, 0.10, 0.05]
    out = place_upright(request(ctx), ctx)
    assert out["status"] == "succeeded"
    cap, carry = ctx.requests[0]
    rise, preplace = carry["waypoints"]
    assert rise["position_m"][:2] == pytest.approx([0.30, 0.10]) and rise["position_m"][
        2
    ] == pytest.approx(preplace["position_m"][2])
    assert carry["held"]["object_id"] == "d2" and carry["linear"] is False
    # the held tile's bottom over the standing tops at the end of the carry: 20 mm
    assert preplace["position_m"][2] - 0.030 - 0.048 == pytest.approx(0.020 + 0.0015)
    descent = ctx.requests[1][1]["waypoints"]
    assert len(descent) > 1 and all(
        w["position_m"][:2] == pytest.approx(descent[-1]["position_m"][:2]) for w in descent
    )


def test_baseline_stamped_just_after_now_does_not_break_the_verdict():
    # Agent 05's integration: neighbour stamps a fraction of a ms after ctx.now() made the
    # observer reject its own request and place_upright raise KeyError('faults').
    out, _ = run(faults={"clock_ahead"})
    assert out["status"] == "succeeded", out["failure"]


def test_a_rejected_verdict_becomes_a_structured_unverified_result(monkeypatch):
    from roboskillhub.runtime.task import validate
    from roboskillhub.tasks.domino import place as place_mod

    real = place_mod.verify_stable_placement

    def baseline_after_start(req, *a, **k):
        return real({**req, "events": {**req["events"], "action_start_s": 0.0}}, *a, **k)

    monkeypatch.setattr(place_mod, "verify_stable_placement", baseline_after_start)
    ctx = FakeTaskContext()
    out = place_upright(request(ctx), ctx)
    assert out["status"] == "unverified" and out["failure_type"] == "evidence_missing"
    assert out["failure"]["code"] == "verification_rejected"
    assert out["failure"]["verification_code"] == "invalid_request"
    assert "baseline was observed after the action started" in out["failure"]["detail"]
    assert out["evidence"]["verification"]["status"] == "rejected"
    assert out["recovery"]["action"] == "reobserve"
    validate("SkillResult", out)
