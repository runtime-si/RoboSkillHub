"""approaching-above-a-target: FakeContext tests of the hover moves and the arrival check."""

from __future__ import annotations

import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "approaching-above-a-target"
ROT = {"w": 0.0, "x": 0.7071, "y": 0.7071, "z": 0.0}
OBB = {
    "center": {"x": 0.40, "y": 0.10, "z": 0.02},
    "extent": {"x": 0.02, "y": 0.02, "z": 0.02},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}
HOVER_Z = 0.02 + 0.02 + 0.12


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["approach_above_target"].module


def _pose(x, y, z, rot=ROT):
    return {"pose": {"position": {"x": x, "y": y, "z": z}, "rotation": rot}}


def _ctx(start, reached, move=None):
    return FakeContext({"robot.get_ee_pose": [start, reached], "robot.go_to_pose": move})


def test_rises_in_place_then_hovers_with_the_current_rotation(script):
    ctx = _ctx(_pose(0.2, -0.1, 0.05), _pose(0.401, 0.1, HOVER_Z))
    out = script.run(ctx, target_obb=OBB)
    assert out["route"] == "above"
    assert out["position_error"] == pytest.approx(0.001, abs=1e-9)
    rise, hover = (c.kwargs for c in ctx.calls_to("robot.go_to_pose"))
    assert rise["pose"]["position"] == {"x": 0.2, "y": -0.1, "z": pytest.approx(HOVER_Z)}
    assert hover["pose"]["position"] == {"x": 0.40, "y": 0.10, "z": pytest.approx(HOVER_Z)}
    assert rise["pose"]["rotation"] == ROT and hover["pose"]["rotation"] == ROT
    # gap-connector knobs and arm 0 are left to the connector's defaults.
    assert set(rise) == {"pose"} and set(hover) == {"pose"}
    assert out["hover_pose"] == hover["pose"]


def test_no_rise_when_already_high_and_a_floor_and_rotation_apply(script):
    down = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
    ctx = _ctx(_pose(0.2, -0.1, 0.50), _pose(0.40, 0.10, 0.30, down))
    out = script.run(
        ctx, target_obb=OBB, min_z=0.30, rotation=down, tolerance=0.003, max_steps=400, arm_id=1
    )
    assert out["route"] == "above"
    (hover,) = (c.kwargs for c in ctx.calls_to("robot.go_to_pose"))
    assert hover["pose"]["position"]["z"] == pytest.approx(0.30)
    assert hover["pose"]["rotation"] == down
    assert hover["tolerance"] == 0.003 and hover["max_steps"] == 400 and hover["arm_id"] == 1
    assert all(c.kwargs == {"arm_id": 1} for c in ctx.calls_to("robot.get_ee_pose"))


def test_ending_off_the_hover_point_is_unreachable(script):
    ctx = _ctx(_pose(0.40, 0.10, 0.30), _pose(0.40, 0.10, HOVER_Z - 0.016))
    out = script.run(ctx, target_obb=OBB)
    assert out["route"] == "unreachable"
    assert out["position_error"] == pytest.approx(0.016)
    assert "16.0 mm" in out["detail"]


def test_a_refused_move_is_unreachable(script):
    def refuse(**kwargs):
        raise ToolError("robot.go_to_pose", "no IK solution")

    ctx = _ctx(_pose(0.2, -0.1, 0.05), _pose(0.2, -0.1, 0.05), move=refuse)
    out = script.run(ctx, target_obb=OBB)
    assert out["route"] == "unreachable"
    assert out["position_error"] == -1.0
    assert "no IK solution" in out["detail"]
    assert ctx.call_count("robot.get_ee_pose") == 1


@pytest.mark.parametrize(
    "reply",
    [
        {"reached": False, "outcome": "unreachable", "reason": "no IK within 30 mm"},
        {"outcome": "timeout"},
        {"success": False, "message": "outside the workspace box"},
    ],
)
def test_a_move_that_replies_it_did_not_arrive_is_unreachable(script, reply):
    ctx = _ctx(_pose(0.40, 0.10, 0.30), _pose(0.40, 0.10, HOVER_Z), move=reply)
    out = script.run(ctx, target_obb=OBB)
    assert out["route"] == "unreachable"
    assert out["detail"].startswith("hover move failed")
    assert ctx.call_count("robot.get_ee_pose") == 1


def test_a_reached_reply_passes(script):
    ctx = _ctx(
        _pose(0.40, 0.10, 0.30),
        _pose(0.40, 0.10, HOVER_Z),
        move={"reached": True, "outcome": "reached"},
    )
    assert script.run(ctx, target_obb=OBB)["route"] == "above"


def test_a_turned_wrist_is_unreachable_past_the_bound(script):
    # 2.15 deg about z (what a YAM twin's servo left) passes; 6 deg does not.
    import math

    def about_z(deg):
        h = math.radians(deg) / 2
        return {"w": math.cos(h), "x": 0.0, "y": 0.0, "z": math.sin(h)}

    ok = script.run(
        _ctx(_pose(0.4, 0.1, 0.3, about_z(0)), _pose(0.4, 0.1, HOVER_Z, about_z(2.15))),
        target_obb=OBB,
    )
    assert ok["route"] == "above" and ok["rotation_error_deg"] == pytest.approx(2.15, abs=1e-6)
    off = script.run(
        _ctx(_pose(0.4, 0.1, 0.3, about_z(0)), _pose(0.4, 0.1, HOVER_Z, about_z(6.0))),
        target_obb=OBB,
    )
    assert off["route"] == "unreachable" and "turned 6.0 deg" in off["detail"]


def test_a_bare_pose_reply_is_accepted(script):
    ctx = _ctx(_pose(0.40, 0.10, 0.30)["pose"], _pose(0.40, 0.10, HOVER_Z)["pose"])
    assert script.run(ctx, target_obb=OBB)["route"] == "above"
