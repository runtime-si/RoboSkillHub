"""lego.align_connection: planning, prechecks (zero motion on rejection), bounded correction,
observation problems. Fakes only: unit_or_dryrun."""

from __future__ import annotations

import pytest
from lego_fixtures import T_TCP_OBJECT, WORLD, align_request, obs, placed, shifted

from roboskillhub.tasks.lego import geometry as g
from roboskillhub.tasks.lego.align import align_connection, plan_alignment
from roboskillhub.tasks.lego.fakes import FakeClock, FakeMotion, FakeObserver


def _run(req, clock, *, grasp_offset=None, **obs_kw):
    motion = FakeMotion(clock)
    true = (
        T_TCP_OBJECT
        if grasp_offset is None
        else g.to_pose(g.mul(g.matrix(T_TCP_OBJECT), grasp_offset), "tcp")
    )
    observer = FakeObserver(clock, motion, true, **obs_kw)
    out = align_connection(req, motion, observer, clock=clock)
    return out, motion, observer


def test_plan_is_a_plan_not_an_alignment():
    clock = FakeClock()
    plan = plan_alignment(align_request(clock))
    assert plan["kind"] == "plan" and plan["ok"]
    assert plan["engagement_axis"]["vector"] == pytest.approx([0, 0, -1])
    seat, hover = plan["seat_pose"], plan["hover_brick_pose"]
    assert hover["position_m"][2] - seat["position_m"][2] == pytest.approx(0.01)
    assert "status" not in plan


def test_aligns_when_the_grasp_is_as_believed():
    clock = FakeClock()
    out, motion, _ = _run(align_request(clock), clock)
    assert out["status"] == "succeeded" and out["reason_code"] == "aligned"
    assert out["retries"]["corrections"] == 0 and len(motion.commands) == 1
    assert out["evidence"]["alignment_is_physical"] is True
    assert out["evidence_level"] == "unit_or_dryrun"


def test_corrects_a_grasp_offset_within_budget():
    clock = FakeClock()
    # The brick sits 1.5 mm off and 2 deg turned in the hand versus what the skill was told.
    off = g.mul(g.tr(0.0015, -0.001, 0.0), g.rz(2.0))
    out, motion, _ = _run(align_request(clock), clock, grasp_offset=off)
    assert out["status"] == "succeeded"
    assert 1 <= out["retries"]["corrections"] <= 3
    assert len(motion.commands) == out["retries"]["corrections"] + 1
    assert out["evidence"]["observed_error"]["lateral_m"] <= 0.0004


def test_lateral_error_too_large_exhausts_the_correction_budget():
    clock = FakeClock()
    req = align_request(clock)
    req["limits"]["max_total_correction_m"] = 0.002
    out, _, _ = _run(req, clock, grasp_offset=g.tr(0.004, 0, 0))
    assert out["status"] == "failed" and out["reason_code"] == "correction_budget_exhausted"
    assert out["motion_issued"] is True


def test_non_converging_alignment_stops_at_max_corrections():
    clock = FakeClock()
    req = align_request(clock)
    req["limits"]["max_corrections"] = 2

    def jitter(i, t):  # the brick keeps landing 1 mm off whatever is commanded
        return g.mul(t, g.tr(0.001 * (-1) ** i, 0, 0))

    out, motion, _ = _run(req, clock, perturb=jitter)
    assert out["status"] == "failed" and out["reason_code"] == "alignment_not_converged"
    assert out["retries"]["corrections"] == 2 and len(motion.commands) == 3


def test_angle_error_is_corrected_or_reported():
    clock = FakeClock()
    req = align_request(clock)
    req["limits"]["max_corrections"] = 0
    out, _, _ = _run(req, clock, grasp_offset=g.rz(3.0))
    assert out["status"] == "failed" and out["reason_code"] == "alignment_not_converged"
    assert abs(out["evidence"]["observed_error"]["yaw_deg"]) == pytest.approx(3.0, abs=1e-6)


def test_tilted_held_brick_fails_without_correcting():
    clock = FakeClock()
    off = g.matrix(
        shifted(
            {"frame_id": "x", "position_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}, tilt_x=5.0
        )
    )
    out, motion, _ = _run(align_request(clock), clock, grasp_offset=off)
    assert out["reason_code"] == "held_brick_tilted" and len(motion.commands) == 1
    assert "regrasp" in out["next_allowed"]


def test_deadline_spent_by_slow_moves():
    clock = FakeClock()
    req = align_request(clock)
    req["limits"]["deadline_s"] = 0.5
    motion = FakeMotion(clock, move_s=0.3)
    observer = FakeObserver(
        clock, motion, g.to_pose(g.mul(g.matrix(T_TCP_OBJECT), g.tr(0.001, 0, 0)), "tcp")
    )
    out = align_connection(req, motion, observer, clock=clock)
    assert out["status"] == "failed" and out["reason_code"] == "deadline_exceeded"


# ---------------------------------------------------------------- rejected: zero motion


def _rejected(req, clock, code, **kw):
    out, motion, observer = _run(req, clock, **kw)
    assert out["status"] == "rejected", out
    assert out["reason_code"] == code, out
    assert motion.commands == [] and out["motion_issued"] is False
    return out


def test_insufficient_support_rejected_before_motion():
    clock = FakeClock()
    req = align_request(
        clock, cell={"x": 4, "y": 4, "layer": 2, "rotation": 0}, build=[placed("u", 10, 10)]
    )
    _rejected(req, clock, "insufficient_support")


def test_partial_support_threshold_from_config():
    clock = FakeClock()
    # 2x4 on layer 2 overlapping a 2x4 below by one 2x1 row: 2/8 studs engaged.
    req = align_request(
        clock, cell={"x": 4, "y": 4, "layer": 2, "rotation": 0}, build=[placed("u", 4, 7)]
    )
    req["limits"]["support_required_fraction"] = 0.5
    _rejected(req, clock, "insufficient_support")
    req["limits"]["support_required_fraction"] = 0.25
    req["build"][0]["brick_id"] = "u"
    out, _, _ = _run(req, clock)
    assert out["status"] == "succeeded"


def test_damaging_the_existing_build_rejected_before_motion():
    clock = FakeClock()
    req = align_request(clock, build=[placed("n", 6, 4, layer=1)])
    out = _rejected(req, clock, "would_damage_build")
    assert "n" in out["message"]


def test_target_occupied_rejected():
    clock = FakeClock()
    _rejected(align_request(clock, build=[placed("o", 4, 5)]), clock, "target_occupied")


def test_interface_mismatch_rejected():
    clock = FakeClock()
    req = align_request(clock)
    req["plate"]["system"] = "duplo"
    req["limits"]["xy_tolerance_m"] = 0.0004
    _rejected(req, clock, "interface_mismatch")


def test_ambiguous_symmetric_orientation_rejected():
    clock = FakeClock()
    _rejected(align_request(clock, tcp_yaw_deg=90.0), clock, "ambiguous_orientation")


def test_unreachable_hover_rejected():
    clock = FakeClock()
    req = align_request(clock)
    req["limits"]["workspace_max_m"] = [0.8, 0.5, 0.03]
    _rejected(req, clock, "unreachable")


@pytest.mark.parametrize(
    "mutate, code",
    [
        (
            lambda r: r["held"].update(
                holding_evidence=obs(True, r["tcp_observation"]["timestamp_s"] - 5)
            ),
            "grasp_unconfirmed",
        ),
        (
            lambda r: r["held"].update(
                holding_evidence=obs(False, r["tcp_observation"]["timestamp_s"])
            ),
            "grasp_unconfirmed",
        ),
        (lambda r: r["held"].pop("holding_evidence"), "grasp_unconfirmed"),
        (lambda r: r["held"].update(object_id="b2"), "identity_mismatch"),
        (lambda r: r["tcp_observation"].update(kind="commanded"), "commanded_not_observed"),
        (
            lambda r: r["tcp_observation"].update(
                timestamp_s=r["tcp_observation"]["timestamp_s"] - 3
            ),
            "stale_observation",
        ),
        (lambda r: r["tcp_observation"].update(frame_id="camera"), "frame_mismatch"),
        (
            lambda r: r["tcp_observation"].update(kind="sim_truth", privileged=False),
            "unlabelled_privileged",
        ),
        (
            lambda r: r["held"].update(T_object_TCP=r["held"].pop("T_TCP_object")),
            "invalid_geometry",
        ),
        (lambda r: r["plate"].update(calibration_id=""), "uncalibrated"),
        (lambda r: r["limits"].update(xy_tolerance_m=0.002), "tolerance_exceeds_capture"),
        (lambda r: r["limits"].pop("xy_tolerance_m"), "invalid_input"),
        (lambda r: r.update(reference_id="ghost"), "unknown_reference"),
        (lambda r: r.pop("connection_id"), "invalid_input"),
        (lambda r: r["brick"].update(version="lego-std-8mm/0"), "invalid_geometry"),
        (lambda r: r["target_cell"].update(x=0.032), "invalid_geometry"),
        (lambda r: r["plate"]["pose"].update(quaternion_xyzw=[0, 0, 0, 3]), "invalid_geometry"),
    ],
)
def test_input_and_observation_negatives_reject_without_motion(mutate, code):
    clock = FakeClock()
    req = align_request(clock)
    mutate(req)
    _rejected(req, clock, code)


def test_missing_observer_is_rejected_not_executed_blind():
    clock = FakeClock()
    motion = FakeMotion(clock)
    out = align_connection(align_request(clock), motion, None, clock=clock)
    assert out["status"] == "rejected" and out["reason_code"] == "missing_capabilities"
    assert motion.commands == []


def test_coarse_observer_rejected_before_motion():
    clock = FakeClock()
    _rejected(align_request(clock), clock, "resolution_insufficient", sigma_m=0.0054)


# ---------------------------------------------------------------- unverified after motion


@pytest.mark.parametrize(
    "kw, code",
    [
        ({"age_s": 5.0}, "stale_observation"),
        ({"visible": False}, "brick_not_visible"),
        ({"occluded": True}, "occluded"),
        ({"kind": "commanded"}, "commanded_not_observed"),
        ({"frame_id": "camera"}, "frame_mismatch"),
    ],
)
def test_bad_observation_after_motion_is_unverified(kw, code):
    clock = FakeClock()
    out, motion, _ = _run(align_request(clock), clock, **kw)
    assert out["status"] == "unverified" and out["reason_code"] == code
    assert out["motion_issued"] is True and len(motion.commands) == 1


def test_sim_truth_marks_result_privileged():
    clock = FakeClock()
    out, _, _ = _run(align_request(clock), clock, kind="sim_truth")
    assert out["status"] == "succeeded" and out["privileged_observation"] is True


def test_motion_failure_stops_and_reports():
    clock = FakeClock()
    motion = FakeMotion(clock, fail_on=1)
    observer = FakeObserver(clock, motion, T_TCP_OBJECT)
    out = align_connection(align_request(clock), motion, observer, clock=clock)
    assert out["status"] == "failed" and out["reason_code"] == "motion_failed"
    assert motion.stops == 1 and out["stop"]["confirmed"] is True


def test_requests_do_not_share_evidence():
    clock = FakeClock()
    a, _, _ = _run(align_request(clock, request_id="r-a", connection_id="c-a"), clock)
    clock.advance(1.0)
    t_b = clock()
    b, _, _ = _run(align_request(clock, request_id="r-b", connection_id="c-b"), clock)
    assert a["invocation_id"] != b["invocation_id"]
    assert (a["connection_id"], b["connection_id"]) == ("c-a", "c-b")
    brick_refs = [r for r in b["evidence_refs"] if r["source"] == "fake-wrist-camera"]
    assert brick_refs and all(r["timestamp_s"] >= t_b for r in brick_refs)
    # Reusing a's alignment result for b's press is refused (see test_lego_seat).


def test_duplo_geometry_aligns_too():
    clock = FakeClock()
    out, _, _ = _run(
        align_request(
            clock, system="duplo", studs="2x2", cell={"x": 2, "y": 2, "layer": 1, "rotation": 0}
        ),
        clock,
    )
    assert out["status"] == "succeeded"
    seat = out["evidence"]["plan"]["seat_pose"]["position_m"]
    assert seat[0] == pytest.approx(0.30 + (-8 + 2 + 1) * 0.016)
    assert WORLD == out["evidence"]["plan"]["seat_pose"]["frame_id"]
