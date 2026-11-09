"""lego.guarded_seat: capability gate (no position-only fallback, zero motion), and trace
classification (complete, short, blocked, early contact, over-force, lost signal, stop
unconfirmed). Scripted traces only: unit_or_dryrun."""

from __future__ import annotations

import pytest
from lego_fixtures import align_request

from roboskillhub.tasks.lego import NOT_FORCE_CONTROL, check_contact_capabilities, guarded_seat
from roboskillhub.tasks.lego.align import align_connection
from roboskillhub.tasks.lego.fakes import (
    FakeClock,
    FakeContact,
    FakeMotion,
    FakeObserver,
    press_trace,
)
from roboskillhub.tasks.lego.geometry import SYSTEMS

PRE_H = 0.01
LEGO = SYSTEMS["lego"]


@pytest.fixture(scope="module")
def alignment():
    clock = FakeClock()
    req = align_request(clock)
    motion = FakeMotion(clock)
    out = align_connection(
        req, motion, FakeObserver(clock, motion, req["held"]["T_TCP_object"]), clock=clock
    )
    assert out["status"] == "succeeded"
    return out


def seat_request(alignment, **over):
    req = {
        "request_id": "req-2",
        "connection_id": "conn-1",
        "system": "lego",
        "alignment": alignment,
        "axis": {"frame_id": "world", "vector": [0, 0, -1]},
        "max_travel_m": PRE_H + 0.0005,
        "speed_m_s": 0.005,
        "timeout_s": 5.0,
        "force_limit_N": 15.0,
        "limits_source": "test-config (not a YAM calibration)",
    }
    req.update(over)
    return req


def test_complete_press_is_operation_complete_not_connection_verified(alignment):
    contact = FakeContact(samples=press_trace(PRE_H, LEGO.stud_height_m, peak_N=15.0))
    out = guarded_seat(seat_request(alignment), contact)
    assert out["status"] == "succeeded" and out["reason_code"] == "press_complete"
    assert out["evidence"]["operation_complete"] is True
    assert out["evidence"]["connection_verified"] is False
    assert "lego.verify_connection" in out["next_allowed"]
    assert len(contact.calls) == 1 and contact.calls[0]["force_limit_N"] == 15.0


# ------------------------------------------------------------ capability gate: zero motion


@pytest.mark.parametrize(
    "contact, missing",
    [
        (None, "pose_feedback"),
        (
            FakeContact(capabilities=["pose_feedback", "bounded_position_approach", "stop"]),
            "contact_force_feedback",
        ),
        (
            FakeContact(
                capabilities=["pose_feedback", "contact_force_feedback", "compliant_motion"]
            ),
            "stop",
        ),
        (FakeContact(controller_kind="guarded_position"), "controller.kind"),
    ],
)
def test_missing_contact_capability_rejects_with_zero_motion(alignment, contact, missing):
    out = guarded_seat(seat_request(alignment), contact)
    assert out["status"] == "rejected" and out["reason_code"] == "missing_capabilities"
    assert out["motion_issued"] is False
    assert any(missing in m for m in out["evidence"]["missing_capabilities"])
    assert out["evidence"]["control_statement"] == NOT_FORCE_CONTROL
    if contact is not None:
        assert contact.calls == []


def test_position_only_backend_is_told_it_is_not_force_control(alignment):
    contact = FakeContact(
        capabilities=["pose_feedback", "bounded_position_approach"], controller_kind="position"
    )
    out = guarded_seat(seat_request(alignment), contact)
    assert out["status"] == "rejected"
    assert "NOT compliant force control" in out["message"]
    assert contact.calls == []


def test_capability_check_requires_declared_signal_units():
    desc = FakeContact().describe()
    assert check_contact_capabilities(desc) == []
    desc["contact_signal"]["unit"] = "A"  # motor current is not a force
    assert "contact_signal.unit==N" in check_contact_capabilities(desc)
    desc["contact_signal"].pop("calibration_id")
    assert "contact_signal.calibration_id" in check_contact_capabilities(desc)


@pytest.mark.parametrize(
    "over, code",
    [
        ({"force_limit_N": 50.0}, "force_limit_unsupported"),
        ({"torque_limit_Nm": 0.5}, "torque_limit_unsupported"),
        ({"max_travel_m": PRE_H + 0.005}, "travel_too_long"),
        ({"max_travel_m": PRE_H - 0.002}, "travel_too_short"),
        ({"axis": {"frame_id": "world", "vector": [0, 0.2, -1]}}, "axis_mismatch"),
        ({"axis": {"frame_id": "camera", "vector": [0, 0, -1]}}, "frame_mismatch"),
        ({"connection_id": "conn-other"}, "evidence_mismatch"),
        ({"force_limit_N": None}, "invalid_input"),
    ],
)
def test_request_negatives_reject_with_zero_motion(alignment, over, code):
    contact = FakeContact(samples=press_trace(PRE_H, LEGO.stud_height_m))
    req = seat_request(alignment, **over)
    if over.get("force_limit_N", 0) is None:
        req.pop("force_limit_N")
    out = guarded_seat(req, contact)
    assert out["status"] == "rejected" and out["reason_code"] == code, out
    assert contact.calls == []


def test_unaligned_or_failed_alignment_blocks_the_press(alignment):
    contact = FakeContact(samples=press_trace(PRE_H, LEGO.stud_height_m))
    bad = {**alignment, "status": "failed"}
    out = guarded_seat(seat_request(bad), contact)
    assert out["reason_code"] == "not_aligned" and contact.calls == []
    out = guarded_seat(
        {k: v for k, v in seat_request(alignment).items() if k != "alignment"}, contact
    )
    assert out["reason_code"] == "not_aligned" and contact.calls == []


# ------------------------------------------------------------ trace classification


def _press(alignment, samples, **kw):
    return guarded_seat(seat_request(alignment), FakeContact(samples=samples, **kw))


def test_not_pressed_fully(alignment):
    out = _press(alignment, press_trace(PRE_H, LEGO.stud_height_m, final_shortfall_m=0.0008))
    assert out["status"] == "failed" and out["reason_code"] == "not_pressed_fully"
    assert out["evidence"]["operation_complete"] is False


def test_blocked_on_stud_tops_is_misalignment(alignment):
    out = _press(alignment, press_trace(PRE_H, LEGO.stud_height_m, final_shortfall_m=0.0016))
    assert out["status"] == "failed" and out["reason_code"] == "blocked_on_studs"


def test_early_contact_is_an_obstacle(alignment):
    out = _press(
        alignment,
        press_trace(PRE_H, LEGO.stud_height_m, onset_travel_m=0.004, final_shortfall_m=0.005),
    )
    assert out["status"] == "failed" and out["reason_code"] == "early_contact"


def test_force_overshoot_fails_even_at_seat_depth(alignment):
    out = _press(alignment, press_trace(PRE_H, LEGO.stud_height_m, peak_N=40.0))
    assert out["status"] == "failed" and out["reason_code"] == "force_limit_exceeded"


def test_no_contact_means_brick_missing_or_slipped(alignment):
    samples = [{"t_s": i * 0.002, "travel_m": i * 0.0002, "force_N": 0.0} for i in range(53)]
    out = _press(alignment, samples, stop_reason="travel_limit")
    assert out["status"] == "failed" and out["reason_code"] == "no_contact"


def test_force_threshold_short_of_seat_is_not_success(alignment):
    # The force threshold was reached, but 1 mm short: a threshold alone is not success.
    out = _press(alignment, press_trace(PRE_H, LEGO.stud_height_m, final_shortfall_m=0.001))
    assert out["status"] == "failed"


def test_seat_depth_without_force_rise_is_unverified(alignment):
    out = _press(
        alignment, press_trace(PRE_H, LEGO.stud_height_m, peak_N=5.0), stop_reason="travel_limit"
    )
    assert out["status"] == "unverified"
    assert out["reason_code"] == "seat_depth_without_threshold"


def test_stop_not_confirmed_fails_to_human(alignment):
    out = _press(alignment, press_trace(PRE_H, LEGO.stud_height_m), stop_confirmed=False)
    assert out["status"] == "failed" and out["reason_code"] == "stop_unconfirmed"
    assert out["next_allowed"] == ["human"] and out["stop"]["confirmed"] is False


def test_signal_gap_mid_press_is_unverified(alignment):
    samples = press_trace(PRE_H, LEGO.stud_height_m)
    for s in samples[len(samples) // 2 :]:
        s["t_s"] += 0.2
    out = _press(alignment, samples)
    assert out["status"] == "unverified" and out["reason_code"] == "contact_signal_lost"


def test_timeout_fails(alignment):
    out = _press(
        alignment,
        press_trace(PRE_H, LEGO.stud_height_m, final_shortfall_m=0.0008),
        stop_reason="timeout",
    )
    assert out["status"] == "failed" and out["reason_code"] == "timeout"


def test_press_system_must_match_the_alignment(alignment):
    contact = FakeContact(samples=press_trace(PRE_H, LEGO.stud_height_m))
    out = guarded_seat(seat_request(alignment, system="duplo"), contact)
    assert out["reason_code"] == "evidence_mismatch" and contact.calls == []


def test_duplo_bands_scale_with_stud_height(alignment):
    duplo = SYSTEMS["duplo"]
    clock = FakeClock()
    areq = align_request(
        clock, system="duplo", studs="2x2", cell={"x": 2, "y": 2, "layer": 1, "rotation": 0}
    )
    motion = FakeMotion(clock)
    dal = align_connection(
        areq, motion, FakeObserver(clock, motion, areq["held"]["T_TCP_object"]), clock=clock
    )
    assert dal["status"] == "succeeded"
    req = seat_request(dal, system="duplo", max_travel_m=PRE_H + 0.002)
    ok = guarded_seat(
        req, FakeContact(samples=press_trace(PRE_H, duplo.stud_height_m, final_shortfall_m=0.0012))
    )
    assert ok["status"] == "succeeded"  # 1.2 mm < 0.35 * 4.5 mm
    lego = guarded_seat(
        seat_request(alignment),
        FakeContact(samples=press_trace(PRE_H, LEGO.stud_height_m, final_shortfall_m=0.0012)),
    )
    assert lego["status"] == "failed"  # 1.2 mm > 0.6 * 1.7 mm on LEGO


def test_sim_backend_level_is_carried(alignment):
    out = _press(
        alignment, press_trace(PRE_H, LEGO.stud_height_m), evidence_level="dynamic_sim_observed"
    )
    assert out["evidence_level"] == "dynamic_sim_observed"
