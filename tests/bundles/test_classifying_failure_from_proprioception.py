"""classifying-failure-from-proprioception: FakeContext tests of every exit, the jaw and lag signals and the sampling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "classifying-failure-from-proprioception"
SURFACE = 0.0
REACH = 0.0102
SPAN = 0.094
STOP = 0.0
OPEN = 1.0
CUBE = 0.030 / SPAN  # the open fraction of a jaw closed on a 30 mm cube
EMPTY = 0.0015 / SPAN  # a jaw closed on air, 1.5 mm off the stop
GRIPPER = {
    "gripper": "linear_4310",
    "span_m": SPAN,
    "width_fit": {"source": "measured", "at_open_m": SPAN, "at_closed_m": STOP},
    "finger": {"stated": True, "reach_m": REACH},
}
JOINTS = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["classify_failure"].module


def s(z, frac, target=None, x=0.40, closed=None, joints=JOINTS):
    out = {
        "ee_position": {"x": x, "y": 0.0, "z": z},
        "gripper_fraction": frac,
        "joints": list(joints),
    }
    if target is not None:
        out["target_position"] = {"x": x, "y": 0.0, "z": target}
    if closed is not None:
        out["commanded_closed"] = closed
    return out


def ramp(z0, z1, n, frac, target=None):
    return [
        s(float(z), frac, target if target is not None else float(z))
        for z in np.linspace(z0, z1, n)
    ]


def obs_of(sample):
    p = sample["ee_position"]
    return {
        "cameras": [],
        "arms": [
            {
                "joint_state": {"positions": np.array(sample["joints"])},
                "gripper_fraction": sample["gripper_fraction"],
                "ee_pose": {
                    "position": dict(p),
                    "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
                },
            }
        ],
    }


def ctx_for(now, gripper=GRIPPER, workspace=None):
    return FakeContext(
        {
            "robot.get_observation": lambda **_: obs_of(now),
            "robot.describe_gripper": gripper,
            "robot.describe_workspace": workspace or {"surface_z": SURFACE},
        }
    )


def classify(script, history, now=None, **kw):
    now = now or history[-1]
    kw.setdefault("sample_interval_s", 0.0)
    return script.run(ctx_for(now), history=history, **kw)


# -- the four staged episodes of the simulation check -----------------------------------------------


def good_lift():
    """Close on the cube at grasp height, lift 10 cm in 1 cm segments."""
    return [s(0.020, OPEN), s(0.020, CUBE)] + ramp(0.030, 0.120, 10, CUBE)


def test_a_held_lift_is_ok_held(script):
    out = classify(script, good_lift())
    assert out["route"] == "ok" and out["state"] == "held"
    assert out["features"]["held_final"] and not out["features"]["empty_final"]


def test_a_close_on_nothing_is_never_gripped(script):
    hist = [s(0.020, OPEN), s(0.020, EMPTY)] + ramp(0.030, 0.120, 10, EMPTY)
    out = classify(script, hist)
    assert out["route"] == "never_gripped" and out["state"] == "empty"


def test_opening_mid_air_is_a_drop(script):
    hist = good_lift() + [s(0.120, OPEN)] + ramp(0.130, 0.150, 3, OPEN)
    out = classify(script, hist)
    assert out["route"] == "drop"
    assert out["features"]["ee_z_at_last_open"] == pytest.approx(0.120)


def test_pushing_into_the_table_is_a_stall(script):
    below = -0.04
    hist = ramp(0.10, 0.02, 9, EMPTY, target=None)
    hist += [s(0.0105, EMPTY, target=below) for _ in range(12)]
    out = classify(script, hist)
    assert out["route"] == "stall"
    assert out["features"]["lag_final_m"] == pytest.approx(0.0505, abs=1e-3)


# -- the other exits --------------------------------------------------------------------------------


def test_frozen_while_holding_in_the_air_short_of_target_is_stuck_mid_air(script):
    hist = [s(0.15, CUBE, target=0.25) for _ in range(12)]
    out = classify(script, hist)
    assert out["route"] == "stuck_mid_air"


def test_frozen_holding_on_target_is_ok_held_not_stuck(script):
    hist = [s(0.15, CUBE, target=0.15) for _ in range(12)]
    out = classify(script, hist)
    assert (out["route"], out["state"]) == ("ok", "held")


def test_frozen_without_target_is_a_stall_only_when_motion_was_expected(script):
    hist = [s(0.03, EMPTY) for _ in range(12)]
    assert classify(script, hist, expect_motion=True)["route"] == "stall"
    # Not expected (a blocking connector's arm is still between nodes): the empty jaw speaks instead.
    assert classify(script, hist)["route"] == "never_gripped"


def test_holding_frozen_without_target_and_motion_expected_is_stuck_mid_air(script):
    hist = [s(0.15, CUBE) for _ in range(12)]
    assert classify(script, hist, expect_motion=True)["route"] == "stuck_mid_air"


def test_stall_waits_for_chunks_elapsed(script):
    hist = [s(0.03, CUBE, target=-0.04) for _ in range(12)]
    out = classify(script, hist, chunks_elapsed=5)
    assert out["route"] != "stall"


def test_repeated_open_close_at_the_table_is_a_fumble(script):
    hist = []
    for i in range(16):
        hist.append(s(0.015 + 0.002 * (i % 2), CUBE if i % 2 else OPEN, x=0.40 + 0.003 * i))
    out = classify(script, hist)
    assert out["route"] == "fumble" and "low height" in out["reason"]


def test_four_transitions_while_moving_is_a_general_fumble(script):
    hist = ramp(0.05, 0.12, 6, OPEN)
    for i, frac in enumerate([CUBE, OPEN, CUBE, OPEN]):
        hist.append(s(0.12 - 0.002 * i, frac))
    hist.append(s(0.03, OPEN))
    out = classify(script, hist)
    assert out["route"] == "fumble" and out["features"]["n_grip_transitions"] == 4


def test_a_held_object_slipping_out_while_shut_is_a_drop(script):
    hist = good_lift() + [s(0.120, EMPTY), s(0.120, EMPTY)]
    out = classify(script, hist)
    assert out["route"] == "drop" and "slipped" in out["reason"]


def test_a_slip_at_the_table_is_not_called_a_drop(script):
    hist = [s(0.020, CUBE), s(0.020, CUBE), s(0.021, EMPTY)]
    out = classify(script, hist)
    assert out["route"] != "drop"


def test_opening_after_a_descent_is_ok_placed(script):
    hist = ramp(0.12, 0.03, 10, CUBE) + [s(0.03, OPEN)] + ramp(0.04, 0.06, 2, OPEN)
    out = classify(script, hist)
    assert (out["route"], out["state"]) == ("ok", "placed")


def test_opening_low_without_descent_is_ok_placed(script):
    hist = [s(0.025, OPEN) for _ in range(4)]
    out = classify(script, hist, chunks_elapsed=0)
    assert (out["route"], out["state"]) == ("ok", "placed")


def test_an_open_hand_that_never_closed_is_never_gripped_once_the_step_ran_long(script):
    hist = ramp(0.15, 0.08, 8, OPEN)
    assert classify(script, hist)["route"] == "never_gripped"
    assert classify(script, hist)["state"] == "open"
    assert classify(script, hist, chunks_elapsed=10)["route"] == "ok"


def test_a_closed_hand_low_and_moving_is_approaching(script):
    hist = ramp(0.05, 0.035, 5, CUBE)
    out = classify(script, hist)
    assert (out["route"], out["state"]) == ("ok", "approaching")


# -- jaw convention, sampling, window ---------------------------------------------------------------


def test_a_wide_object_held_at_70_mm_counts_as_closed_and_held(script):
    wide = 0.070 / SPAN
    hist = [s(0.020, OPEN), s(0.020, wide)] + ramp(0.03, 0.12, 10, wide)
    out = classify(script, hist)
    assert (out["route"], out["state"]) == ("ok", "held")


def test_commanded_closed_wins_over_the_measured_opening(script):
    nearly_open = (SPAN - 0.001) / SPAN
    hist = [s(z, nearly_open, closed=True) for z in np.linspace(0.03, 0.12, 10)]
    assert classify(script, hist, n_samples=0)["features"]["grip_final_closed"]


def test_sampling_alone_reads_the_arm_and_stamps_the_target(script):
    now = s(0.12, CUBE)
    ctx = ctx_for(now)
    out = script.run(
        ctx, n_samples=3, sample_interval_s=0.0, target_position={"x": 0.4, "y": 0.0, "z": 0.12}
    )
    assert ctx.call_count("robot.get_observation") == 3
    assert out["samples"] == 3
    assert all(h["target_position"]["z"] == pytest.approx(0.12) for h in out["history"])
    assert out["history"][0]["joints"] == JOINTS
    assert (out["route"], out["state"]) == ("ok", "held")


def test_self_samples_inherit_the_last_history_target(script):
    hist = ramp(0.10, 0.02, 9, EMPTY) + [s(0.0105, EMPTY, target=-0.04) for _ in range(10)]
    out = classify(script, hist, now=s(0.0105, EMPTY))
    assert out["history"][-1]["target_position"]["z"] == pytest.approx(-0.04)
    assert out["route"] == "stall"


def test_the_window_keeps_the_last_samples(script):
    hist = good_lift() + [s(0.12, CUBE) for _ in range(30)]
    out = classify(script, hist, n_samples=0, window=16)
    assert out["samples"] == 16


def test_surface_z_input_skips_describe_workspace(script):
    ctx = FakeContext(
        {
            "robot.get_observation": lambda **_: obs_of(s(0.12, CUBE)),
            "robot.describe_gripper": GRIPPER,
        }
    )
    out = script.run(ctx, history=good_lift(), surface_z=SURFACE, sample_interval_s=0.0)
    assert out["route"] == "ok"
    assert ctx.call_count("robot.describe_workspace") == 0


def test_heights_are_over_the_surface_and_fingertip_reach(script):
    raised = 0.75
    hist = [
        dict(h, ee_position={**h["ee_position"], "z": h["ee_position"]["z"] + raised})
        for h in good_lift()
    ]
    hist += [s(0.12 + raised, OPEN)] + ramp(0.13 + raised, 0.15 + raised, 3, OPEN)
    out = script.run(
        ctx_for(hist[-1], workspace={"surface_z": raised}), history=hist, sample_interval_s=0.0
    )
    assert out["route"] == "drop"
    assert out["features"]["table_level_z"] == pytest.approx(raised + REACH)


def test_a_gripper_without_width_fit_reads_against_its_span(script):
    real = {
        "gripper": "linear_4310",
        "span_m": SPAN,
        "min_grasp_width_m": 0.0,
        "finger": {"reach_m": REACH},
    }
    hist = [s(0.020, OPEN), s(0.020, EMPTY)] + ramp(0.030, 0.120, 10, EMPTY)
    out = script.run(ctx_for(hist[-1], gripper=real), history=hist, sample_interval_s=0.0)
    assert out["route"] == "never_gripped"
    assert out["features"]["gripper_stop_m"] == 0.0


def test_describe_gripper_failure_falls_back_to_the_linear_4310(script):
    def boom(**_):
        raise RuntimeError("no hand")

    ctx = FakeContext(
        {
            "robot.get_observation": lambda **_: obs_of(s(0.12, CUBE)),
            "robot.describe_gripper": boom,
            "robot.describe_workspace": {"surface_z": SURFACE},
        }
    )
    out = script.run(ctx, history=good_lift(), sample_interval_s=0.0)
    assert out["features"]["gripper_span_m"] == pytest.approx(SPAN)
    assert out["route"] == "ok"


def test_too_few_samples_raises(script):
    with pytest.raises(ValueError):
        script.run(ctx_for(s(0.1, OPEN)), n_samples=1, sample_interval_s=0.0)


def test_no_surface_anywhere_raises(script):
    ctx = FakeContext(
        {
            "robot.get_observation": lambda **_: obs_of(s(0.12, CUBE)),
            "robot.describe_gripper": GRIPPER,
            "robot.describe_workspace": {},
        }
    )
    with pytest.raises(ValueError):
        script.run(ctx, history=good_lift(), sample_interval_s=0.0)


def test_arm_id_is_passed_only_when_nonzero(script):
    ctx = ctx_for(s(0.12, CUBE))
    script.run(ctx, history=good_lift(), sample_interval_s=0.0)
    assert "arm_id" not in ctx.calls_to("robot.describe_gripper")[0].kwargs


def test_every_route_is_declared_in_the_example(script, skills_registry):
    bundle = skills_registry.get(BUNDLE)
    graph = json.loads(
        (Path(bundle.bundle_dir) / "examples" / "canonical_subgraph.json").read_text()
    )
    declared = set(graph["exit"]["success_values"]) | {graph["on_error"]}
    assert set(script.ROUTES) <= declared
    assert set(script.ROUTES) | {"unknown"} == set(bundle.meta.exit_conditions)
