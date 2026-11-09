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


# ------------------------------------------------------------ post-press seated pose (sim snap / slip)

NOW = 100.0


def _seat_obs(alignment, *, dz=0.0, dx=0.0, yaw_deg=0.0, tilt_deg=0.0, t=NOW, sigma=0.0001):
    """An independent post-press observation of the pressed brick, offset from its seat."""
    import math

    from roboskillhub.tasks.lego import geometry as g

    seat = alignment["evidence"]["plan"]["seat_pose"]
    c, s_ = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))
    rx = [[1, 0, 0, 0], [0, c, -s_, 0], [0, s_, c, 0], [0, 0, 0, 1]]  # tilt about the brick's x
    m = g.mul(g.matrix(seat), g.mul(g.mul(g.tr(dx, 0.0, dz), g.rz(yaw_deg)), rx))
    return {
        "obs_id": f"camera:brick@{t}",
        "source": "camera",
        "kind": "measured",
        "frame_id": seat["frame_id"],
        "timestamp_s": t,
        "value": g.to_pose(m, seat["frame_id"]),
        "uncertainty_m": sigma,
        "uncertainty_deg": 0.05,
    }


def _press_obs(alignment, samples, obs, *, observable=None, **kw):
    contact = FakeContact(samples=samples, insertion_force_observable=observable, **kw)
    return guarded_seat(seat_request(alignment), contact, clock=lambda: NOW, observe=lambda: obs)


@pytest.mark.parametrize("observable", [False, True, None])
def test_regression_firm_grip_snap_tcp_short_but_brick_seated_succeeds(alignment, observable):
    # 15 N grip in BrickSim: the clutch snapped the brick 1.66 mm down to its seat, the hand did not
    # follow; the press stopped on force 1.09 mm short (TCP). Before: "blocked"/"not_seated".
    samples = press_trace(PRE_H, LEGO.stud_height_m, final_shortfall_m=0.00109, peak_N=6.0)
    out = _press_obs(alignment, samples, _seat_obs(alignment), observable=observable)
    assert out["status"] == "succeeded" and out["reason_code"] == "press_complete", out["message"]
    sc = out["evidence"]["seat_check"]
    assert sc["basis"] == "post_press_observation"
    assert sc["tcp_shortfall_m"] == pytest.approx(0.00109)
    assert sc["brick_vs_tcp_m"] == pytest.approx(0.00109, abs=1e-6)
    assert out["evidence"]["connection_verified"] is False


def test_snap_station_travel_stop_with_seated_brick_succeeds_with_declared_threshold(alignment):
    out = _press_obs(
        alignment,
        press_trace(PRE_H, LEGO.stud_height_m, peak_N=5.0),
        _seat_obs(alignment, dz=0.0002),
        observable=False,
        stop_reason="travel_limit",
    )
    assert out["status"] == "succeeded"
    sc = out["evidence"]["seat_check"]
    assert sc["seat_depth_tolerance_m"] == pytest.approx(0.35 * LEGO.stud_height_m)
    assert "SEAT_FRACTION" in sc["seat_depth_tolerance_source"]


def test_force_observable_station_needs_the_force_rise_even_when_seated(alignment):
    out = _press_obs(
        alignment,
        press_trace(PRE_H, LEGO.stud_height_m, peak_N=5.0),
        _seat_obs(alignment),
        observable=True,
        stop_reason="travel_limit",
    )
    assert out["status"] == "unverified" and out["reason_code"] == "seated_without_force_rise"


@pytest.mark.parametrize(
    "obs_kw, code",
    [
        ({"dz": 0.0017}, "blocked_on_studs"),
        ({"dz": 0.0009}, "not_pressed_fully"),
        ({"dx": 0.001}, "seat_pose_misaligned"),
        ({"dx": 0.004}, "seat_pose_misaligned"),
        ({"yaw_deg": 4.0}, "seat_pose_misaligned"),
    ],
)
def test_observed_brick_off_its_seat_fails_whatever_the_tcp_says(alignment, obs_kw, code):
    # The TCP reached the seat and the force rose: the observed brick still decides.
    out = _press_obs(
        alignment, press_trace(PRE_H, LEGO.stud_height_m), _seat_obs(alignment, **obs_kw)
    )
    assert out["status"] == "failed" and out["reason_code"] == code, out["message"]


def test_slip_in_the_grasp_is_caught_by_the_seat_check(alignment):
    # TCP at seat depth with the force rise, brick 0.9 mm high in the grasp: not seated.
    out = _press_obs(
        alignment, press_trace(PRE_H, LEGO.stud_height_m), _seat_obs(alignment, dz=0.0009)
    )
    assert out["reason_code"] == "not_pressed_fully"


def test_no_observation_on_a_snap_station_is_unverified_not_success(alignment):
    out = _press_obs(alignment, press_trace(PRE_H, LEGO.stud_height_m), None, observable=False)
    assert out["status"] == "unverified" and out["reason_code"] == "seat_unobserved"


@pytest.mark.parametrize(
    "obs_kw",
    [{"t": NOW - 10.0}, {"sigma": 0.001}],  # before the press / 2-sigma wider than the seat band
)
def test_unusable_observation_falls_back_to_force_and_travel(alignment, obs_kw):
    out = _press_obs(
        alignment, press_trace(PRE_H, LEGO.stud_height_m), _seat_obs(alignment, **obs_kw)
    )
    assert out["evidence"]["seat_check"]["basis"] == "tcp_travel_and_force"
    assert out["evidence"]["seat_check_observation"].startswith("unusable")
    assert out["status"] == "succeeded"  # force-observable default, seat depth, force threshold


def test_declared_seat_depth_tolerance_is_used_and_cited(alignment):
    req = seat_request(alignment, seat_depth_tolerance_m=0.0003)
    out = guarded_seat(
        req,
        FakeContact(samples=press_trace(PRE_H, LEGO.stud_height_m)),
        clock=lambda: NOW,
        observe=lambda: _seat_obs(alignment, dz=0.0004),
    )
    assert out["reason_code"] == "not_pressed_fully"
    assert out["evidence"]["seat_check"]["seat_depth_tolerance_source"].startswith("request")


@pytest.mark.parametrize("tilt_deg", [1.49, 2.2])
def test_regression_firm_grip_tilt_in_the_hand_with_corners_in_band_succeeds(alignment, tilt_deg):
    # 15 N grip, brick on the plate: snapped, clutched and seated, then observed tilted 1.5-2.2 deg
    # by the clamp while still held (centre 0.33 mm low). Before: "misaligned" on the tilt band.
    samples = press_trace(PRE_H, LEGO.stud_height_m, final_shortfall_m=0.00096, peak_N=6.0)
    out = _press_obs(
        alignment, samples, _seat_obs(alignment, dz=-0.00033, tilt_deg=tilt_deg), observable=False
    )
    assert out["status"] == "succeeded", out["message"]
    d = out["evidence"]["seat_check"]["delta"]
    assert d["corner_max_dz_m"] <= 0.000595 and d["tilt_deg"] == pytest.approx(tilt_deg, abs=0.01)


@pytest.mark.parametrize(
    "tilt_deg, code",
    [(2.0, "not_pressed_fully"), (4.0, "blocked_on_studs")],
)
def test_tilt_that_lifts_a_corner_fails(alignment, tilt_deg, code):
    # Tilt about x raises one end of the 32 mm side by 16 mm x sin(tilt): with the centre 0.3 mm
    # high, 2 deg puts that corner 0.86 mm over the seat (band 0.6 mm), 4 deg 1.42 mm (on the studs).
    out = _press_obs(
        alignment,
        press_trace(PRE_H, LEGO.stud_height_m),
        _seat_obs(alignment, dz=0.0003, tilt_deg=tilt_deg),
    )
    assert out["status"] == "failed" and out["reason_code"] == code, out["message"]
    assert out["evidence"]["seat_check"]["delta"]["corner_max_dz_m"] > 0.000595
