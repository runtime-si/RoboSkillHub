"""trying-arms-in-fallback-order with two fake arms."""

from __future__ import annotations

import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
TARGET = {"x": 0.3, "y": 0.1, "z": 0.02}


@pytest.fixture(scope="module")
def ta(skills_registry):
    return skills_registry.get("trying-arms-in-fallback-order").canonical_scripts["try_arms"].module


class Arms:
    def __init__(self, reach=(True, True), holds=(True, True), refuse=(None, None)):
        self.home = {0: [0.3, 0.3, 0.3], 1: [0.3, -0.3, 0.3]}
        self.reach, self.holds, self.refuse = reach, holds, refuse
        self.closed = {0: False, 1: False}
        self.log = []

    def tools(self):
        return {
            "robot.get_ee_pose": lambda arm_id=0, **_: {
                "pose": {
                    "position": dict(zip("xyz", self.home[arm_id], strict=False)),
                    "rotation": DOWN,
                }
            },
            "robot.grasp_frame": lambda arm_id=0, **_: {"rotation": DOWN},
            "robot.solve_ik": lambda pose, arm_id=0, **_: {"success": self.reach[arm_id]},
            "robot.open_gripper": lambda arm_id=0, **_: self.log.append(("open", arm_id)) or {},
            "robot.close_gripper": lambda arm_id=0, **_: (
                self.log.append(("close", arm_id)) or self.closed.update({arm_id: True}) or {}
            ),
            "robot.get_gripper": lambda arm_id=0, **_: {
                "position": 0.4 if self.holds[arm_id] else 0.0
            },
            "robot.go_to_pose": self.go,
        }

    def go(self, pose, arm_id=0, **_):
        self.log.append(("go", arm_id, round(pose["position"]["z"], 3)))
        r = self.refuse[arm_id]
        if r == "raise":
            raise ToolError("robot.go_to_pose", "joint limit")
        if r:
            return {"reached": False, "outcome": "unreachable"}
        return {"reached": True, "outcome": "reached"}


def test_the_nearer_arm_picks(ta):
    arms = Arms()
    out = ta.run(FakeContext(arms.tools()), target_position=TARGET)
    assert out["route"] == "picked_by_first" and out["arm_used"] == 0 and out["order"] == [0, 1]
    assert [e for e in arms.log if e[0] == "go"] == [
        ("go", 0, 0.1),
        ("go", 0, 0.02),
        ("go", 0, 0.1),
    ]


def test_out_of_reach_falls_back_to_the_other_arm(ta):
    arms = Arms(reach=(False, True))
    out = ta.run(FakeContext(arms.tools()), target_position=TARGET)
    assert out["route"] == "picked_by_second" and out["arm_used"] == 1
    assert out["attempts"][0]["result"] == "unreachable" and not any(
        e[1] == 0 for e in arms.log if e[0] == "go"
    )


def test_a_miss_falls_back_and_reopens(ta):
    arms = Arms(holds=(False, True))
    out = ta.run(FakeContext(arms.tools()), target_position=TARGET)
    assert out["route"] == "picked_by_second" and out["attempts"][0]["result"] == "missed"
    assert ("open", 0) in arms.log[arms.log.index(("close", 0)) :]


@pytest.mark.parametrize("refuse", ["routed", "raise"])
def test_refused_moves_count_as_unreachable(ta, refuse):
    arms = Arms(refuse=(refuse, None))
    out = ta.run(FakeContext(arms.tools()), target_position=TARGET)
    assert out["route"] == "picked_by_second" and out["attempts"][0]["result"] == "unreachable"


def test_both_failed_and_given_order(ta):
    arms = Arms(reach=(False, False))
    out = ta.run(FakeContext(arms.tools()), target_position=TARGET)
    assert out["route"] == "both_failed" and len(out["attempts"]) == 2 and out["arm_used"] == -1
    arms = Arms()
    out = ta.run(
        FakeContext(arms.tools()),
        target_position={"x": 0.3, "y": -0.2, "z": 0.02},
        arms=[0, 1],
        order="given",
    )
    assert out["order"] == [0, 1] and out["arm_used"] == 0
    out = ta.run(FakeContext(Arms().tools()), target_position={"x": 0.3, "y": -0.2, "z": 0.02})
    assert out["order"] == [1, 0] and out["arm_used"] == 1


def test_bad_arguments(ta):
    with pytest.raises(ValueError, match="arms"):
        ta.run(FakeContext({}), target_position=TARGET, arms=[0, 0])
    with pytest.raises(ValueError, match="order"):
        ta.run(FakeContext({}), target_position=TARGET, order="random")
