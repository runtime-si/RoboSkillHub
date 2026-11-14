"""assigning-motion-segments-to-arms-with-handovers: the Viterbi DP with a fake IK (arm 0 reaches y >= -0.05,
arm 1 y <= 0.05) and with the CPU YAM model (tests/bundles/_yam_ik.py)."""

from __future__ import annotations

import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

from . import _yam_ik as Y


@pytest.fixture(scope="module")
def aa(skills_registry):
    return (
        skills_registry.get("assigning-motion-segments-to-arms-with-handovers")
        .canonical_scripts["assign_arms"]
        .module
    )


def _ik(pose, arm_id=0, **_):
    y = pose["position"]["y"]
    ok = y >= -0.05 if arm_id == 0 else y <= 0.05
    return {"success": ok, "joint_positions": [y, 0, 0, 0, 0, 0] if ok else []}


def _line(y0, y1, n=11):
    return [[0.3, y0 + (y1 - y0) * i / (n - 1), 0.1] for i in range(n)]


def test_one_arm_side_needs_no_handover(aa):
    out = aa.run(FakeContext({"robot.solve_ik": _ik}), waypoints=_line(0.3, 0.1))
    assert (
        out["route"] == "assignment_found"
        and set(out["assignment"]) == {0}
        and out["handovers"] == []
    )


def test_crossing_the_middle_hands_over_where_both_reach(aa):
    out = aa.run(FakeContext({"robot.solve_ik": _ik}), waypoints=_line(0.3, -0.3))
    assert out["route"] == "handover_required" and out["n_handovers"] == 1
    h = out["handovers"][0]
    assert (h["from_arm"], h["to_arm"]) == (0, 1) and abs(h["position"]["y"]) <= 0.05
    assert out["assignment"][0] == 0 and out["assignment"][-1] == 1


def test_a_waypoint_no_arm_reaches_is_infeasible(aa):
    def ik(pose, arm_id=0, **_):
        if pose["position"]["y"] == 0.0:
            return {"success": False}
        return _ik(pose, arm_id)

    out = aa.run(FakeContext({"robot.solve_ik": ik}), waypoints=_line(0.3, -0.3))
    assert out["route"] == "infeasible" and out["unreachable_waypoints"] == [5]


def test_raised_ik_counts_as_unreachable_and_initial_arm_is_honoured(aa):
    def ik(pose, arm_id=0, **_):
        if arm_id == 1:
            raise ToolError("robot.solve_ik", "unreachable")
        return _ik(pose, arm_id)

    out = aa.run(FakeContext({"robot.solve_ik": ik}), waypoints=_line(0.3, 0.1))
    assert out["route"] == "assignment_found"
    out = aa.run(FakeContext({"robot.solve_ik": _ik}), waypoints=_line(0.03, -0.3), initial_arm=0)
    assert out["assignment"][0] == 0 and out["route"] == "handover_required"


def test_handover_cost_trades_against_joint_motion(aa):
    # Both arms reach everything; arm 1's joints move 10x more: a free handover is taken, a costly one not.
    def ik(pose, arm_id=0, **_):
        y = pose["position"]["y"]
        return {"success": True, "joint_positions": [y * (1 if arm_id == 0 else 10)] * 6}

    path = _line(0.3, -0.3)
    assert (
        aa.run(FakeContext({"robot.solve_ik": ik}), waypoints=path, handover_cost=0.0)[
            "n_handovers"
        ]
        == 0
    )
    assert (
        aa.run(
            FakeContext({"robot.solve_ik": ik}), waypoints=path, initial_arm=1, handover_cost=0.01
        )["n_handovers"]
        == 1
    )
    assert (
        aa.run(
            FakeContext({"robot.solve_ik": ik}), waypoints=path, initial_arm=1, handover_cost=100.0
        )["n_handovers"]
        == 0
    )


def test_with_the_yam_model_a_left_to_right_carry_needs_a_handover(aa):
    arms = Y.TwoArms()
    out = aa.run(FakeContext({"robot.solve_ik": arms.solve_ik}), waypoints=_line(0.35, -0.35, 8))
    assert out["route"] == "handover_required" and out["n_handovers"] == 1
    assert -0.1 <= out["handovers"][0]["position"]["y"] <= 0.1


def test_bad_arguments(aa):
    with pytest.raises(ValueError, match="arms"):
        aa.run(FakeContext({}), waypoints=_line(0, 1), arms=[0])
    with pytest.raises(ValueError, match="waypoints"):
        aa.run(FakeContext({}), waypoints=[[0, 0, 0]])
    with pytest.raises(ValueError, match="initial_arm"):
        aa.run(FakeContext({}), waypoints=_line(0, 1), initial_arm=3)
