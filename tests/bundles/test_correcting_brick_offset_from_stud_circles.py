"""correcting-brick-offset-from-stud-circles: FakeContext tests with a fake YAM arm whose wrist D405 is
ray-cast over a LEGO / DUPLO brick (tests/bundles/_brick_offset_sim.py): the stud-circle measurement,
the closed loop under landing, reporting and hand-eye errors, every exit, and the connector knobs."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext

from . import _brick_offset_sim as S

BUNDLE = "correcting-brick-offset-from-stud-circles"
P = 0.008


@pytest.fixture(scope="module")
def co(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["correct_offset"].module


def _ctx(arm, **tools):
    return FakeContext(
        {
            "robot.get_observation": arm.obs,
            "robot.get_ee_pose": arm.ee,
            "robot.go_to_pose": arm.go,
            **tools,
        }
    )


def _pair_err(sc, pose):
    e = np.linalg.inv(sc.pair) @ S.mat(pose)
    return np.abs(e[:3, 3]) * 1e3, abs(math.degrees(math.atan2(e[1, 0], e[0, 0])))


# -- the measurement ------------------------------------------------------------------------


@pytest.mark.parametrize("yaw", [-35.0, 0.0, 20.0])
def test_measure_finds_the_pair_to_a_tenth_of_a_millimetre(co, yaw):
    sc = S.Scene(yaw_deg=yaw)
    arm = S.Arm(sc, S.start_over(sc, 0.002, -0.001, 3.0))
    frame = arm.obs()["cameras"][0]
    m = co.measure(frame, S.mat(S.guess(sc, 0.0015, 0.001, 2.0)), co.PRESETS["lego"])
    assert m["ok"], m["why"]
    xyz, dyaw = _pair_err(sc, S.se3(m["pair"]))
    assert xyz.max() < 0.12 and dyaw < 0.6
    assert m["lattice_studs"] == 8 and len(m["studs"]) == 2


def test_a_block_hanging_in_the_view_hides_studs_but_not_the_pair(co):
    sc = S.Scene(floaters=[(4, 8, 1, 1, 3, 1)])
    arm = S.Arm(sc, S.start_over(sc))
    frame = arm.obs()["cameras"][0]
    m = co.measure(frame, sc.pair, co.PRESETS["lego"])
    assert m["ok"], m["why"]
    assert m["lattice_studs"] < 8
    xyz, _ = _pair_err(sc, S.se3(m["pair"]))
    assert xyz[:2].max() < 0.15


def test_a_stud_half_hidden_is_fitted_from_its_seen_rim(co):
    sc = S.Scene()
    arm = S.Arm(sc, S.start_over(sc))
    frame = dict(arm.obs()["cameras"][0])
    # Blank the depth over one side of a pair stud, as a finger or a dropout would.
    stud = sc.pair @ np.array([-0.5 * P, 0.0, 0.0017, 1.0])
    pc = np.linalg.inv(S.mat(frame["pose"])) @ stud
    k = S.R.D405_K
    u, v = k[0, 0] * pc[0] / pc[2] + k[0, 2], k[1, 1] * pc[1] / pc[2] + k[1, 2]
    depth = frame["depth"].copy()
    rad = k[0, 0] * 0.0024 / pc[2]
    depth[int(v - 2 * rad) : int(v + 2 * rad), int(u + 0.3 * rad) : int(u + 2 * rad)] = 0.0
    frame["depth"] = depth
    m = co.measure(frame, sc.pair, co.PRESETS["lego"])
    assert m["ok"], m["why"]
    assert min(m["arc"]) < 200.0
    xyz, _ = _pair_err(sc, S.se3(m["pair"]))
    assert xyz[:2].max() < 0.15


def test_duplo_is_measured(co):
    sc = S.Scene(system="duplo", brick=(3, 3, 2, 4))
    arm = S.Arm(sc, S.start_over(sc, 0.003, 0.0, 0.0, height=0.04))
    frame = arm.obs()["cameras"][0]
    m = co.measure(frame, S.mat(S.guess(sc, 0.002)), co.PRESETS["duplo"])
    assert m["ok"], m["why"]
    xyz, dyaw = _pair_err(sc, S.se3(m["pair"]))
    assert xyz[:2].max() < 0.25 and dyaw < 1.0


def test_a_prior_half_a_pitch_off_is_ambiguous(co):
    sc = S.Scene()
    arm = S.Arm(sc, S.start_over(sc))
    frame = arm.obs()["cameras"][0]
    m = co.measure(frame, S.mat(S.guess(sc, 0.5 * P)), co.PRESETS["lego"])
    assert not m["ok"] and m.get("ambiguous")


# -- the loop -------------------------------------------------------------------------------


def test_the_loop_aligns_the_hand_over_the_pair(co):
    sc = S.Scene()
    arm = S.Arm(sc, S.start_over(sc, 0.003, -0.002, 4.0))
    out = co.run(_ctx(arm), S.guess(sc, 0.0015, -0.001, 2.0))
    assert out["route"] == "aligned", out["detail"]
    lat, dyaw = arm.error()
    assert lat < 0.15 and dyaw < 0.6
    assert out["moves"] >= 2  # the peek pose, then at least one correction
    assert len(out["history"]) == out["iterations"] and math.hypot(*out["offset_mm"]) <= 0.2
    xyz, _ = _pair_err(sc, out["pair_pose"])
    assert xyz.max() < 0.12
    assert out["camera_name"] == "left_wrist" and out["tool_axis_source"] == "ee_pose"
    # The hand stays at the peek height over the stud tops.
    top = sc.pair[2, 3] + 0.0017
    assert arm.true[2, 3] - top == pytest.approx(0.03, abs=3e-4)
    assert len(out["pair_px"]) == 2 and len(out["stud_centres"]) == 2


def test_landing_and_reporting_errors_are_closed_out(co):
    sc = S.Scene(yaw_deg=-25.0)
    arm = S.Arm(
        sc,
        S.start_over(sc, -0.002, 0.003, -3.0),
        land_frac=0.15,
        land_sigma=0.0001,
        land_yaw_deg=0.1,
        read_bias=(0.001, -0.0005, 0.0),
        seed=4,
    )
    out = co.run(_ctx(arm), S.guess(sc, -0.001, 0.0005, -2.0), move_to_peek=False)
    assert out["route"] == "aligned", out["detail"]
    lat, dyaw = arm.error()
    assert lat < 0.25 and dyaw < 0.8
    assert out["moves"] >= 2


def test_a_calibrated_tool_centre_pixel_takes_out_a_hand_eye_error(co):
    he = S.small(0.0006, -0.0004, 0.0, 0.2, -0.2, 0.0)
    cal_sc = S.Scene(yaw_deg=0.0)
    cal = co.run(
        _ctx(S.Arm(cal_sc, S.start_over(cal_sc), hand_eye_error=he)),
        S.guess(cal_sc),
        max_iterations=1,
        move_to_peek=False,
    )
    px, rng = cal["pair_px"], cal["pair_range_m"]
    assert 0.1 < rng < 0.2
    errs = {}
    for name, kw in (
        ("ee", {}),
        ("px", {"tool_center_px": px}),
        ("px+range", {"tool_center_px": px, "tool_center_range": rng}),
    ):
        sc = S.Scene(yaw_deg=15.0)
        arm = S.Arm(sc, S.start_over(sc, 0.002, 0.002, 2.0), hand_eye_error=he)
        out = co.run(_ctx(arm), S.guess(sc, 0.001, 0.001), move_to_peek=False, **kw)
        assert out["route"] == "aligned", out["detail"]
        errs[name] = arm.error()[0]
    assert out["tool_axis_source"] == "tool_center_px"
    assert errs["ee"] > 0.4 and errs["px+range"] < 0.25


def test_yaw_in_pair_turns_the_hand_across_the_pair(co):
    sc = S.Scene()
    arm = S.Arm(sc, S.start_over(sc, 0.001, 0.0, 0.0))
    out = co.run(_ctx(arm), S.guess(sc), yaw_in_pair_deg=90.0)
    assert out["route"] == "aligned", out["detail"]
    assert arm.error(90.0)[1] < 0.6 and arm.error()[0] < 0.15


def test_a_rise_comes_before_the_peek_move_from_below(co):
    sc = S.Scene()
    arm = S.Arm(sc, S.start_over(sc, 0.004, 0.0, 0.0, height=0.015))
    out = co.run(_ctx(arm), S.guess(sc))
    assert out["route"] == "aligned"
    first = arm.moves[0]
    assert np.allclose(first[:2, 3], S.start_over(sc, 0.004, 0.0, 0.0, height=0.015)[:2, 3])
    assert first[2, 3] == pytest.approx(sc.pair[2, 3] + 0.0017 + 0.03, abs=1e-6)


def test_one_iteration_measures_without_moving(co):
    sc = S.Scene()
    arm = S.Arm(sc, S.start_over(sc, 0.002, 0.0, 0.0))
    out = co.run(_ctx(arm), S.guess(sc), max_iterations=1, move_to_peek=False)
    assert out["route"] == "max_iterations" and arm.moves == []
    assert out["offset_mm"][0] == pytest.approx(-2.0, abs=0.15)


def test_an_ambiguous_prior_stops_before_any_correction(co):
    sc = S.Scene()
    arm = S.Arm(sc, S.start_over(sc))
    out = co.run(_ctx(arm), S.guess(sc, 0.5 * P), move_to_peek=False)
    assert out["route"] == "ambiguous_stud_pair" and arm.moves == []


def test_no_brick_under_the_prior_is_lost_studs(co):
    sc = S.Scene()
    arm = S.Arm(sc, S.start_over(sc))
    far = S.se3(sc.pair @ S.small(0.0, 5 * P))  # the bare plate, a brick height lower
    out = co.run(_ctx(arm), far, move_to_peek=False)
    assert out["route"] == "lost_studs" and arm.moves == []


def test_no_wrist_camera_is_lost_studs(co):
    sc = S.Scene()
    arm = S.Arm(sc, S.start_over(sc))
    ctx = _ctx(arm, **{"robot.get_observation": {"cameras": [], "arms": []}})
    out = co.run(ctx, S.guess(sc), move_to_peek=False)
    assert out["route"] == "lost_studs" and "no posed wrist" in out["detail"]


def test_routed_and_raised_refusals_are_unreachable(co):
    sc = S.Scene()
    for refuse, text in (
        (lambda pose: "collision", "collision"),
        (lambda pose: "raise", "joint limit"),
    ):
        arm = S.Arm(sc, S.start_over(sc, 0.002), refuse=refuse)
        out = co.run(_ctx(arm), S.guess(sc))
        assert out["route"] == "unreachable" and text in out["detail"]
        assert "peek pose" in out["detail"] and out["moves"] == 1


def test_a_refused_correction_is_unreachable(co):
    sc = S.Scene()
    arm = S.Arm(sc, S.start_over(sc, 0.002), refuse=lambda pose: "collision")
    out = co.run(_ctx(arm), S.guess(sc), move_to_peek=False)
    assert out["route"] == "unreachable" and "correction 1" in out["detail"]


def test_a_large_correction_is_clipped_to_half_a_pitch(co):
    sc = S.Scene()
    arm = S.Arm(sc, S.start_over(sc, 0.004, 0.003))
    co.run(_ctx(arm), S.guess(sc), move_to_peek=False, max_iterations=2)
    step = arm.moves[0][:3, 3] - S.start_over(sc, 0.004, 0.003)[:3, 3]
    assert np.linalg.norm(step[:2]) == pytest.approx(0.5 * P, abs=1e-6)


def test_connector_knobs_are_passed_only_when_set(co):
    sc = S.Scene()
    arm = S.Arm(sc, S.start_over(sc, 0.002))
    co.run(_ctx(arm), S.guess(sc), max_iterations=2)
    assert arm.kwargs and all(k == {} for k in arm.kwargs)
    arm = S.Arm(sc, S.start_over(sc, 0.002))
    co.run(_ctx(arm), S.guess(sc), max_iterations=2, tolerance=0.002, max_steps=300)
    assert all(k == {"tolerance": 0.002, "max_steps": 300} for k in arm.kwargs)


def test_bad_arguments_raise(co):
    ctx = FakeContext({})
    with pytest.raises(ValueError, match="system"):
        co.run(ctx, {"position": [0, 0, 0]}, system="mega")
    with pytest.raises(ValueError, match="pair_pose"):
        co.run(ctx, {})
    with pytest.raises(ValueError, match="gain"):
        co.run(ctx, {"position": [0, 0, 0]}, gain=0.0)
    with pytest.raises(ValueError, match="tool_center_px"):
        co.run(ctx, {"position": [0, 0, 0]}, tool_center_px=[1.0])
    with pytest.raises(ValueError, match="needs tool_center_px"):
        co.run(ctx, {"position": [0, 0, 0]}, tool_center_range=0.15)
