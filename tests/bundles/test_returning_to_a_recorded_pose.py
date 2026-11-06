"""returning-to-a-recorded-pose: FakeContext tests of the return and its arrival check."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "returning-to-a-recorded-pose"
RECORDED = {"positions": np.array([0.1, 0.4, 0.9, -0.2, 0.0, 0.3])}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["return_to_pose"].module


def _obs(positions):
    return {
        "cameras": [],
        "arms": [
            {
                "joint_state": {"positions": np.asarray(positions, dtype=np.float64)},
                "gripper_fraction": 1.0,
                "ee_pose": {},
            }
        ],
    }


def _ctx(after, move=None):
    return FakeContext(
        {
            "robot.open_gripper": {"position": 1.0},
            "robot.move_to_joints": move,
            "robot.get_observation": _obs(after),
        }
    )


def test_returns_and_reports_the_worst_joint(script):
    after = RECORDED["positions"] + np.array([0.0, 0.03, 0.0, -0.05, 0.0, 0.0])
    ctx = _ctx(after)
    out = script.run(ctx, recorded_joints=RECORDED)
    assert out["route"] == "returned"
    assert out["max_joint_error"] == pytest.approx(0.05)
    assert [c.tool for c in ctx.calls] == [
        "robot.open_gripper",
        "robot.move_to_joints",
        "robot.get_observation",
    ]
    move = ctx.calls_to("robot.move_to_joints")[0].kwargs
    np.testing.assert_allclose(move["joint_config"]["positions"], RECORDED["positions"])
    # Convergence knobs are gap-connector specific: not passed unless asked.
    assert set(move) == {"joint_config"}


def test_a_joint_left_off_is_not_returned(script):
    after = RECORDED["positions"] + np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0])
    out = script.run(_ctx(after), recorded_joints=RECORDED)
    assert out["route"] == "not_returned"
    assert out["max_joint_error"] == pytest.approx(0.2)
    assert "joint 2" in out["detail"]


def test_default_bound_accepts_what_a_yam_station_settles_to(script):
    # A real YAM return stopped 0.0517 rad off on joint 3; that is arrived.
    after = RECORDED["positions"] + np.array([0.0, 0.0, 0.0, 0.0517, 0.0, 0.0])
    assert script.run(_ctx(after), recorded_joints=RECORDED)["route"] == "returned"
    tight = script.run(_ctx(after), recorded_joints=RECORDED, max_error=0.01)
    assert tight["route"] == "not_returned"


def test_a_refused_move_is_not_returned(script):
    def refuse(**kwargs):
        raise ToolError("robot.move_to_joints", "outside the workspace box")

    ctx = _ctx(RECORDED["positions"], move=refuse)
    out = script.run(ctx, recorded_joints=RECORDED, tolerance=0.005, max_steps=800)
    assert out["route"] == "not_returned"
    assert "outside the workspace box" in out["detail"]
    assert out["max_joint_error"] == -1.0
    assert ctx.calls_to("robot.move_to_joints")[0].kwargs["tolerance"] == 0.005
    assert ctx.calls_to("robot.move_to_joints")[0].kwargs["max_steps"] == 800
    assert ctx.call_count("robot.get_observation") == 0


def test_keeps_the_grasp_when_asked_and_takes_a_bare_list(script):
    ctx = _ctx(RECORDED["positions"])
    out = script.run(ctx, recorded_joints=list(RECORDED["positions"]), open_first=False)
    assert out["route"] == "returned"
    assert ctx.call_count("robot.open_gripper") == 0


def test_nothing_recorded_or_no_reading_is_not_returned(script):
    ctx = _ctx(RECORDED["positions"])
    assert script.run(ctx, recorded_joints={"positions": []})["route"] == "not_returned"
    assert ctx.calls == []
    out = script.run(_ctx(RECORDED["positions"]), recorded_joints=RECORDED, arm_id=1)
    assert out == {
        "route": "not_returned",
        "max_joint_error": -1.0,
        "detail": "no reading for arm 1",
    }
