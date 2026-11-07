"""handing-over-between-arms: FakeContext tests of the two-arm sequence, the hold checks and the routing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "handing-over-between-arms"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
ROT_A = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
ROT_B = {"w": 0.0, "x": 0.0, "y": 1.0, "z": 0.0}
# A 30 mm cube held top-down by arm 0, its tool centre 8 mm above the cube's centre.
CUBE = {
    "center": {"x": 0.60, "y": 0.27, "z": 0.10},
    "extent": {"x": 0.015, "y": 0.015, "z": 0.015},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}
MEET = {"x": 0.50, "y": 0.0, "z": 0.30}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["hand_over"].module


def _about_y(deg: float) -> dict[str, float]:
    h = math.radians(deg) / 2.0
    return {"w": math.cos(h), "x": 0.0, "y": math.sin(h), "z": 0.0}


class Arms:
    """Two arms; moves land where commanded unless ``refuse(tool, arm, pose)`` says why not."""

    def __init__(self, refuse=None, receiver_close=0.33, roll_lag_deg=0.0, joints=True):
        self.p = {0: np.array([0.60, 0.27, 0.108]), 1: np.array([0.50, -0.30, 0.30])}
        self.rot = {0: DOWN, 1: ROT_A}
        self.grip = {0: 0.33, 1: 1.0}
        self.q = {0: [0.0] * 6, 1: [0.0] * 6}
        self.refuse = refuse or (lambda tool, a, pose: None)
        self.receiver_close = receiver_close
        self.roll_lag_deg = roll_lag_deg
        self.joints = joints
        self.on_donor_open = None
        self.donor_opens = True
        self.log = []

    def _move(self, tool):
        def go(pose, arm_id=0, **kw):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.log.append((tool, arm_id, target.copy(), pose["rotation"]))
            why = self.refuse(tool, arm_id, pose)
            if why == "raise":
                raise ToolError(tool, "unreachable")
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            self.p[arm_id] = target
            rot = pose["rotation"]
            if arm_id == 1 and self.roll_lag_deg and tool == "robot.go_to_pose":
                rot = _about_y(-self.roll_lag_deg)  # the solver leaves the roll unturned
            self.rot[arm_id] = rot
            return {"reached": True, "outcome": "reached"}

        return go

    def ctx(self):
        def pose(arm_id=0):
            return {
                "pose": {
                    "position": dict(zip("xyz", map(float, self.p[arm_id]), strict=True)),
                    "rotation": self.rot[arm_id],
                }
            }

        def open_gripper(arm_id=0, **kw):
            self.log.append(("open", arm_id))
            if arm_id == 0 and not self.donor_opens:
                return {"position": self.grip[0]}
            self.grip[arm_id] = 1.0
            if arm_id == 0 and self.on_donor_open:
                self.on_donor_open(self)
            return {"position": 1.0}

        def close_gripper(arm_id=0, **kw):
            self.log.append(("close", arm_id))
            self.grip[arm_id] = self.receiver_close if arm_id == 1 else 0.33
            return {"position": self.grip[arm_id]}

        def move_to_joints(joint_config, arm_id=0, **kw):
            self.log.append(("joints", arm_id, list(joint_config["positions"])))
            self.q[arm_id] = list(joint_config["positions"])
            self.rot[arm_id] = ROT_A
            self.roll_lag_deg = 0.0
            return None

        def observation():
            if not self.joints:
                raise ToolError("robot.get_observation", "not served")
            return {"arms": [{"joint_state": {"positions": self.q[a]}} for a in (0, 1)]}

        return FakeContext(
            {
                "robot.get_ee_pose": pose,
                "robot.get_gripper": lambda arm_id=0: {"position": self.grip[arm_id]},
                "robot.go_to_pose": self._move("robot.go_to_pose"),
                "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
                "robot.open_gripper": open_gripper,
                "robot.close_gripper": close_gripper,
                "robot.grasp_frame": {"rotation": ROT_A, "rotation_flipped": ROT_B},
                "robot.move_to_joints": move_to_joints,
                "robot.get_observation": observation,
            }
        )

    def moves(self, arm_id=None):
        return [
            m for m in self.log if m[0].startswith("robot.") and (arm_id is None or m[1] == arm_id)
        ]


def run(script, arms, **kw):
    return script.run(arms.ctx(), object_obb=CUBE, handover_position=MEET, **kw)


def test_happy_path_geometry_and_order(script):
    arms = Arms()
    ctx = arms.ctx()
    out = script.run(ctx, object_obb=CUBE, handover_position=MEET)
    assert out["route"] == "transferred", out["detail"]
    donor = arms.moves(0)
    # Donor carries the cube's centre onto the meet point, keeping its rotation.
    np.testing.assert_allclose(donor[0][2], [0.50, 0.0, 0.308], atol=1e-9)
    assert donor[0][3] == DOWN
    # Receiver: horizontal approach +y; free side is down (away from the donor's
    # tool centre), one half-extent: grasp at z = 0.30 - 0.015; pre-grasp 0.10 back.
    recv = arms.moves(1)
    np.testing.assert_allclose(recv[0][2], [0.50, -0.10, 0.285], atol=1e-9)
    np.testing.assert_allclose(recv[1][2], [0.50, 0.0, 0.285], atol=1e-9)
    # Donor backs out straight up (its tool centre is above the cube), receiver draws back 0.05.
    np.testing.assert_allclose(donor[1][2], [0.50, 0.0, 0.408], atol=1e-9)
    np.testing.assert_allclose(recv[-1][2], [0.50, -0.05, 0.285], atol=1e-9)
    order = [m[:2] for m in arms.log if m[0] in ("open", "close")]
    assert order == [("open", 1), ("close", 1), ("open", 0)]
    # arm_id only for the nonzero arm.
    for call in ctx.calls:
        if call.tool.startswith("robot.") and "arm_id" in call.kwargs:
            assert call.kwargs["arm_id"] == 1
    heading = ctx.calls_to("robot.grasp_frame")[0].kwargs
    assert heading["close_heading_deg"] == pytest.approx(180.0)  # jaws across a +y approach
    assert heading["arm_id"] == 1
    assert out["object_position"] == {
        "x": pytest.approx(0.50),
        "y": pytest.approx(-0.05),
        "z": pytest.approx(0.30),
    }
    assert out["receiver_grip"] == pytest.approx(0.33)


def test_refuses_one_arm_twice_or_a_missing_second_arm(script):
    out = run(script, Arms(), receiver_arm=0)
    assert out["route"] == "donor_stuck" and "both arm 0" in out["detail"]

    arms = Arms()
    arms.p[1] = arms.p[0].copy()  # a connector that clamps arm_id reports the donor
    out = run(script, arms)
    assert out["route"] == "donor_stuck" and "separate arm" in out["detail"]
    assert not arms.moves()

    arms = Arms()
    arms.rot[1] = {"w": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}  # no pose for arm 1 at all
    assert run(script, arms)["route"] == "donor_stuck"


def test_an_empty_donor_is_dropped(script):
    arms = Arms()
    arms.grip[0] = 0.01
    out = run(script, arms)
    assert out["route"] == "dropped" and "holds nothing" in out["detail"]
    assert not arms.moves()


@pytest.mark.parametrize("how", ["raise", "unreachable"])
def test_donor_cannot_reach_the_handover(script, how):
    arms = Arms(refuse=lambda tool, a, pose: how if a == 0 else None)
    out = run(script, arms)
    assert out["route"] == "donor_stuck" and "handover pose" in out["detail"]
    assert not arms.moves(1)


def test_receiver_tries_the_flipped_wrist_then_misses(script):
    arms = Arms(
        refuse=lambda tool, a, pose: "no IK" if a == 1 and pose["rotation"] == ROT_A else None
    )
    assert run(script, arms)["route"] == "transferred"
    assert [m[3] for m in arms.moves(1)][:2] == [ROT_A, ROT_B]

    arms = Arms(refuse=lambda tool, a, pose: "raise" if a == 1 else None)
    out = run(script, arms)
    assert out["route"] == "receiver_missed" and "pre-grasp" in out["detail"]
    assert ("open", 0) not in arms.log


def test_receiver_closing_on_air_backs_off_and_the_donor_keeps_it(script):
    arms = Arms(receiver_close=0.01)
    out = run(script, arms)
    assert out["route"] == "receiver_missed" and "to the stop" in out["detail"]
    assert ("open", 0) not in arms.log
    assert arms.log[-2][:2] == ("open", 1)
    np.testing.assert_allclose(arms.moves(1)[-1][2], [0.50, -0.10, 0.285], atol=1e-9)


def test_receiver_stopped_moving_in(script):
    arms = Arms(
        refuse=lambda tool, a, pose: "contact" if a == 1 and tool.endswith("cartesian") else None
    )
    out = run(script, arms)
    assert out["route"] == "receiver_missed" and "onto the object" in out["detail"]
    assert ("close", 1) not in arms.log


def test_receiver_loses_it_when_the_donor_lets_go(script):
    arms = Arms()

    def slip(a):
        a.grip[1] = 0.0

    arms.on_donor_open = slip
    out = run(script, arms)
    assert out["route"] == "dropped" and "let go" in out["detail"]
    np.testing.assert_allclose(arms.moves(0)[-1][2], [0.50, 0.0, 0.408], atol=1e-9)


def test_a_donor_that_does_not_open_is_stuck(script):
    arms = Arms()
    arms.donor_opens = False
    out = run(script, arms)
    assert out["route"] == "donor_stuck" and "both hands" in out["detail"]


def test_donor_backs_out_sideways_when_up_is_out_of_reach(script):
    def refuse(tool, a, pose):
        up = pose["position"]["z"] > 0.40
        return "unreachable" if a == 0 and tool.endswith("cartesian") and up else None

    arms = Arms(refuse=refuse)
    out = run(script, arms)
    assert out["route"] == "transferred", out["detail"]
    # Away from the receiver, along its approach (+y).
    np.testing.assert_allclose(arms.moves(0)[-1][2], [0.50, 0.10, 0.308], atol=1e-9)

    arms = Arms(
        refuse=lambda tool, a, pose: "raise" if a == 0 and tool.endswith("cartesian") else None
    )
    out = run(script, arms)
    assert out["route"] == "donor_stuck" and "back out" in out["detail"]


def test_receiver_grip_lost_while_the_donor_backs_out(script):
    arms = Arms()
    original = arms._move

    def watch(tool):
        go = original(tool)

        def wrapped(pose, arm_id=0, **kw):
            out = go(pose, arm_id=arm_id, **kw)
            if arm_id == 0 and arms.grip[0] == 1.0:
                arms.grip[1] = 0.02
            return out

        return wrapped

    arms._move = watch
    out = run(script, arms)
    assert out["route"] == "dropped" and "backed out" in out["detail"]


def test_a_free_wrist_roll_is_finished_on_the_last_joint(script):
    arms = Arms(roll_lag_deg=30.0)
    out = run(script, arms)
    assert out["route"] == "transferred", out["detail"]
    turns = [m for m in arms.log if m[0] == "joints"]
    assert len(turns) == 1 and turns[0][1] == 1
    assert turns[0][2][-1] == pytest.approx(math.radians(30.0))
    assert abs(out["receiver_roll_error_deg"]) < 1e-6


def test_no_joint_reading_leaves_the_roll_and_says_so(script):
    arms = Arms(roll_lag_deg=30.0, joints=False)
    out = run(script, arms)
    assert out["route"] == "transferred"
    assert out["receiver_roll_error_deg"] == pytest.approx(30.0)
    assert "wrist was not turned" in out["detail"]


def test_roll_residual_folds_half_turns(script):
    assert script._roll_residual(ROT_A, _about_y(-30), np.array([0, 1.0, 0])) == pytest.approx(30.0)
    assert script._roll_residual(ROT_A, _about_y(170), np.array([0, 1.0, 0])) == pytest.approx(10.0)
    assert script._roll_residual(ROT_A, ROT_A, np.array([0, 1.0, 0])) == 0.0


def test_example_declares_every_route():
    example = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    source = (SKILL_DIR / "scripts" / "hand_over.py").read_text()
    for route in ("transferred", "receiver_missed", "dropped", "donor_stuck"):
        assert f'"{route}"' in source
    assert declared == {"transferred", "receiver_missed", "dropped", "donor_stuck"}
