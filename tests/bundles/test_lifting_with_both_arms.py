"""lifting-with-both-arms: FakeContext tests of the grasp geometry, the interleaved lift and the routing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "lifting-with-both-arms"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
ROT_A = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
ROT_B = {"w": 0.0, "x": 0.0, "y": 1.0, "z": 0.0}
# A 200 x 40 x 40 mm bar lying along world y between the arms.
BAR = {
    "center": {"x": 0.50, "y": 0.0, "z": 0.02},
    "extent": {"x": 0.02, "y": 0.10, "z": 0.02},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}
TILT = math.radians(15.0)
HOVER_Z = 0.02 + 0.08 * math.cos(TILT)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["lift_with_both_arms"].module


def _yaw(deg: float) -> dict[str, float]:
    h = math.radians(deg) / 2.0
    return {"w": math.cos(h), "x": 0.0, "y": 0.0, "z": math.sin(h)}


def _down_rolled(deg: float) -> dict[str, float]:
    """ROT_A (top-down) turned by ``deg`` about the world vertical."""
    h = math.radians(deg) / 2.0
    w, z = math.cos(h), math.sin(h)
    # q_yaw * ROT_A, ROT_A = (0, 1, 0, 0)
    return {"w": 0.0, "x": w, "y": z, "z": 0.0}


class Arms:
    """Two arms; moves land where commanded unless ``refuse(tool, arm, pose)`` says why not."""

    def __init__(self, refuse=None, close_to=(0.43, 0.43), roll_lag_deg=0.0, joints=True):
        self.p = {0: np.array([0.40, 0.30, 0.25]), 1: np.array([0.40, -0.30, 0.25])}
        self.rot = {0: ROT_A, 1: ROT_A}
        self.grip = {0: 1.0, 1: 1.0}
        self.q = {0: [0.0] * 6, 1: [0.0] * 6}
        self.refuse = refuse or (lambda tool, a, pose: None)
        self.close_to = close_to
        self.roll_lag_deg = roll_lag_deg
        self.joints = joints
        self.after_move = None
        self.turned = set()
        self.log = []

    def _move(self, tool):
        def go(pose, arm_id=0, **kw):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.log.append((tool, arm_id, target.copy(), pose["rotation"]))
            why = self.refuse(tool, arm_id, pose)
            if isinstance(why, np.ndarray):  # stalls at this position, reports reached
                self.p[arm_id] = why
                return {"reached": True, "outcome": "reached"}
            if why == "raise":
                raise ToolError(tool, "unreachable")
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            self.p[arm_id] = target
            rot = pose["rotation"]
            if self.roll_lag_deg and tool == "robot.go_to_pose" and arm_id not in self.turned:
                rot = _down_rolled(self.roll_lag_deg)  # the solver leaves the roll unturned
            self.rot[arm_id] = rot
            if self.after_move:
                self.after_move(self, tool, arm_id)
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
            self.grip[arm_id] = 1.0
            return {"position": 1.0}

        def close_gripper(arm_id=0, **kw):
            self.log.append(("close", arm_id))
            self.grip[arm_id] = self.close_to[arm_id]
            return {"position": self.grip[arm_id]}

        def move_to_joints(joint_config, arm_id=0, **kw):
            self.log.append(("joints", arm_id, list(joint_config["positions"])))
            self.q[arm_id] = list(joint_config["positions"])
            self.rot[arm_id] = ROT_A
            self.turned.add(arm_id)
            return None

        def observation():
            if not self.joints:
                raise ToolError("robot.get_observation", "not served")
            return {"arms": [{"joint_state": {"positions": self.q[a]}} for a in (0, 1)]}

        def grasp_frame(close_heading_deg=None, arm_id=0, approach=None, **kw):
            self.log.append(("frame", arm_id, close_heading_deg, approach))
            return {"rotation": ROT_A, "rotation_flipped": ROT_B}

        return FakeContext(
            {
                "robot.get_ee_pose": pose,
                "robot.get_gripper": lambda arm_id=0: {"position": self.grip[arm_id]},
                "robot.go_to_pose": self._move("robot.go_to_pose"),
                "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
                "robot.open_gripper": open_gripper,
                "robot.close_gripper": close_gripper,
                "robot.grasp_frame": grasp_frame,
                "robot.move_to_joints": move_to_joints,
                "robot.get_observation": observation,
            }
        )

    def moves(self, arm_id=None, tool=None):
        return [
            m
            for m in self.log
            if m[0].startswith("robot.")
            and (arm_id is None or m[1] == arm_id)
            and (tool is None or m[0] == tool)
        ]


def run(script, arms, obb=BAR, **kw):
    return script.run(arms.ctx(), object_obb=obb, **kw)


def test_happy_path_geometry_and_interleaving(script):
    arms = Arms()
    out = run(script, arms)
    assert out["route"] == "lifted", out["detail"]
    # Each arm takes its own end: 30 mm in from the ends, tool centre at the bar's centre height.
    np.testing.assert_allclose(_pos(out["first_grasp_pose"]), [0.50, 0.07, 0.02], atol=1e-9)
    np.testing.assert_allclose(_pos(out["second_grasp_pose"]), [0.50, -0.07, 0.02], atol=1e-9)
    # Jaws close across the long axis (y): heading 180 deg = along x.
    assert [m[2] for m in arms.log if m[0] == "frame"] == [180.0, 180.0]
    # Hover, descend, then 10 interleaved 10 mm steps with the leader alternating.
    lifts = [m for m in arms.moves(tool="robot.go_to_pose_cartesian")][2:]
    assert len(lifts) == 20
    assert [m[1] for m in lifts[:4]] == [0, 1, 1, 0]
    heights = {0: [], 1: []}
    for m in lifts:
        heights[m[1]].append(m[2][2])
    for a in (0, 1):
        np.testing.assert_allclose(np.diff([0.02, *heights[a]]), 0.01, atol=1e-9)
        assert heights[a][-1] == pytest.approx(0.12)
    # The hands' heights never differ by more than one step after any move.
    z = {0: 0.02, 1: 0.02}
    for m in lifts:
        z[m[1]] = m[2][2]
        assert abs(z[0] - z[1]) <= 0.01 + 1e-9
    assert out["rise_m"] == pytest.approx(0.10)
    assert out["hand_tilt_deg"] == pytest.approx(0.0, abs=1e-9)
    assert out["separation_change_m"] == pytest.approx(0.0, abs=1e-9)
    assert out["object_obb"]["center"]["z"] == pytest.approx(0.12)
    assert out["first_grip"] == pytest.approx(0.43)


def test_wrists_lean_outward_by_default(script):
    arms = Arms()
    run(script, arms)
    frames = {m[1]: m[3] for m in arms.log if m[0] == "frame"}
    # Arm 0 holds the +y end: its approach points down and toward -y (the other hand).
    assert [frames[0][k] for k in "xyz"] == pytest.approx([0.0, -math.sin(TILT), -math.cos(TILT)])
    assert [frames[1][k] for k in "xyz"] == pytest.approx([0.0, math.sin(TILT), -math.cos(TILT)])
    hovers = {m[1]: m[2] for m in arms.moves(tool="robot.go_to_pose")}
    np.testing.assert_allclose(hovers[0], [0.50, 0.07 + 0.08 * math.sin(TILT), HOVER_Z], atol=1e-9)
    np.testing.assert_allclose(hovers[1], [0.50, -0.07 - 0.08 * math.sin(TILT), HOVER_Z], atol=1e-9)


def test_zero_tilt_is_straight_down(script):
    arms = Arms()
    run(script, arms, outward_tilt_deg=0.0)
    frames = [m[3] for m in arms.log if m[0] == "frame"]
    assert all([f[k] for k in "xyz"] == pytest.approx([0.0, 0.0, -1.0]) for f in frames)
    hovers = [m[2] for m in arms.moves(tool="robot.go_to_pose")]
    assert all(h[2] == pytest.approx(0.10) for h in hovers)


def _pos(pose):
    return [pose["position"][k] for k in "xyz"]


def test_each_arm_takes_its_nearer_end_when_the_ids_are_swapped(script):
    arms = Arms()
    arms.p = {0: np.array([0.40, -0.30, 0.25]), 1: np.array([0.40, 0.30, 0.25])}
    out = run(script, arms)
    assert out["route"] == "lifted"
    assert _pos(out["first_grasp_pose"])[1] == pytest.approx(-0.07)
    assert _pos(out["second_grasp_pose"])[1] == pytest.approx(0.07)


def test_rotated_bar_ends_and_heading(script):
    obb = {**BAR, "orientation": _yaw(20.0)}
    arms = Arms()
    out = run(script, obb=obb, arms=arms)
    assert out["route"] == "lifted"
    along = np.array([-math.sin(math.radians(20)), math.cos(math.radians(20)), 0.0])
    np.testing.assert_allclose(
        _pos(out["first_grasp_pose"]), np.array([0.5, 0, 0.02]) + 0.07 * along, atol=1e-9
    )
    assert [m[2] for m in arms.log if m[0] == "frame"][0] == pytest.approx(200.0)


def test_handles_override_the_box(script):
    arms = Arms()
    handles = [{"x": 0.45, "y": -0.12, "z": 0.08}, {"x": 0.45, "y": 0.12, "z": 0.08}]
    out = run(script, arms, handles=handles)
    assert out["route"] == "lifted"
    assert _pos(out["first_grasp_pose"]) == pytest.approx([0.45, 0.12, 0.08])
    assert _pos(out["second_grasp_pose"]) == pytest.approx([0.45, -0.12, 0.08])


def test_one_handle_is_refused(script):
    out = run(script, Arms(), handles=[{"x": 0.4, "y": 0.0, "z": 0.1}])
    assert out["route"] == "unreachable"


def test_short_object_keeps_the_hands_apart(script):
    obb = {**BAR, "extent": {"x": 0.02, "y": 0.035, "z": 0.02}}
    out = run(script, Arms(), obb=obb)
    # half-length 35 mm less 30 mm inset = 5 mm < a quarter of 35 mm: 8.75 mm.
    assert _pos(out["first_grasp_pose"])[1] == pytest.approx(0.00875)


def test_tall_box_grasp_depth_from_the_top(script):
    obb = {
        **BAR,
        "center": {"x": 0.5, "y": 0.0, "z": 0.05},
        "extent": {"x": 0.02, "y": 0.1, "z": 0.05},
    }
    out = run(script, Arms(), obb=obb)
    assert _pos(out["first_grasp_pose"])[2] == pytest.approx(0.08)


def test_upright_object_is_refused(script):
    obb = {**BAR, "extent": {"x": 0.02, "y": 0.02, "z": 0.10}}
    out = run(script, Arms(), obb=obb)
    assert out["route"] == "unreachable"
    assert "upright" in out["detail"]


def test_same_arm_ids_refused(script):
    out = run(script, Arms(), first_arm=1, second_arm=1)
    assert out["route"] == "unreachable"


def test_connector_without_a_second_arm_is_refused(script):
    arms = Arms()
    arms.p[1] = arms.p[0].copy()
    out = run(script, arms)
    assert out["route"] == "unreachable"
    assert "separate arm" in out["detail"]
    assert not arms.moves()


def test_flipped_wrist_used_when_the_first_is_refused(script):
    arms = Arms(refuse=lambda t, a, p: "joint limit" if a == 1 and p["rotation"] == ROT_A else None)
    out = run(script, arms)
    assert out["route"] == "lifted", out["detail"]
    assert out["second_grasp_pose"]["rotation"] == ROT_B


def test_hover_unreachable_routes(script):
    arms = Arms(refuse=lambda t, a, p: "raise" if a == 1 and t == "robot.go_to_pose" else None)
    out = run(script, arms)
    assert out["route"] == "unreachable"
    assert "arm 1 to its hover" in out["detail"]
    assert ("close", 0) not in arms.log


def test_descent_refused_backs_both_off_open(script):
    arms = Arms(
        refuse=lambda t, a, p: (
            "contact"
            if a == 1 and t == "robot.go_to_pose_cartesian" and p["position"]["z"] < 0.05
            else None
        )
    )
    out = run(script, arms)
    assert out["route"] == "unreachable"
    assert "arm 1 down onto its grasp point" in out["detail"]
    assert arms.grip == {0: 1.0, 1: 1.0}
    assert arms.p[0][2] == pytest.approx(HOVER_Z)
    assert ("close", 0) not in arms.log


def test_empty_jaw_at_the_grasp_opens_and_backs_off(script):
    arms = Arms(close_to=(0.43, 0.01))
    out = run(script, arms)
    assert out["route"] == "slipped"
    assert "arm 1 closed to the stop" in out["detail"]
    assert arms.grip == {0: 1.0, 1: 1.0}
    assert arms.p[0][2] == pytest.approx(HOVER_Z) and arms.p[1][2] == pytest.approx(HOVER_Z)


def test_jaw_runs_to_stop_mid_lift(script):
    arms = Arms()

    def slip(self, tool, a):
        if tool == "robot.go_to_pose_cartesian" and self.p[a][2] >= 0.0549:
            self.grip[1] = 0.0

    arms.after_move = slip
    out = run(script, arms)
    assert out["route"] == "slipped"
    assert "arm 1's jaws ran to the stop" in out["detail"]
    assert out["rise_m"] == pytest.approx(0.04)
    assert ("open", 0) not in arms.log[-5:]  # the hands stay closed where they stopped


def test_separation_drift_routes_slipped(script):
    # Arm 1 reports reached but drifts 4 mm further out on each step.
    def refuse(t, a, p):
        if a == 1 and t == "robot.go_to_pose_cartesian" and p["position"]["z"] > 0.025:
            pos = np.array([p["position"][k] for k in "xyz"])
            pos[1] -= 0.004 * round((pos[2] - 0.02) / 0.01)
            return pos
        return None

    out = run(script, Arms(refuse=refuse))
    assert out["route"] == "slipped"
    assert "separation changed +16 mm" in out["detail"]
    assert out["separation_change_m"] == pytest.approx(0.016, abs=1e-6)


def test_arm_that_cannot_follow_levels_the_other(script):
    # Arm 1 refuses anything over 60 mm up; arm 0 leads on odd steps.
    def refuse(t, a, p):
        if a == 1 and t == "robot.go_to_pose_cartesian" and p["position"]["z"] > 0.0801:
            return "unreachable: out of workspace"
        return None

    arms = Arms(refuse=refuse)
    out = run(script, arms)
    assert out["route"] == "unreachable"
    assert "step 7/10, arm 1" in out["detail"]
    assert "arm 0 brought back level" in out["detail"]
    assert arms.p[0][2] == pytest.approx(0.08)
    assert arms.p[1][2] == pytest.approx(0.08)
    assert out["rise_m"] == pytest.approx(0.06)
    assert arms.grip == {0: 0.43, 1: 0.43}


def test_stall_reported_as_reached_is_caught_by_the_level_check(script):
    def refuse(t, a, p):
        if a == 0 and t == "robot.go_to_pose_cartesian" and p["position"]["z"] > 0.0801:
            return np.array([p["position"]["x"], p["position"]["y"], 0.08])
        return None

    out = run(script, Arms(refuse=refuse))
    assert out["route"] == "unreachable"
    assert "-10 mm off its step height" in out["detail"]


def test_raised_lift_move_is_routed(script):
    def refuse(t, a, p):
        if a == 0 and t == "robot.go_to_pose_cartesian" and p["position"]["z"] > 0.0301:
            return "raise"
        return None

    out = run(script, Arms(refuse=refuse))
    assert out["route"] == "unreachable"
    assert "step 1/10" not in out["detail"] and "step 2/10, arm 0" in out["detail"]


def test_free_wrist_roll_is_finished_on_the_last_joint(script):
    arms = Arms(roll_lag_deg=20.0)
    out = run(script, arms, outward_tilt_deg=0.0)
    assert out["route"] == "lifted", out["detail"]
    turns = [m for m in arms.log if m[0] == "joints"]
    assert [t[1] for t in turns] == [0, 1]
    assert abs(turns[0][2][-1]) == pytest.approx(math.radians(20.0), abs=1e-6)


def test_no_joint_reading_carries_on_and_says_so(script):
    out = run(script, Arms(roll_lag_deg=20.0, joints=False), outward_tilt_deg=0.0)
    assert out["route"] == "lifted"
    assert "wrist was not turned" in out["detail"]
    assert "off square" in out["detail"]


def test_step_count_rounds_up(script):
    arms = Arms()
    out = run(script, arms, lift_height=0.025, lift_step=0.01)
    assert out["route"] == "lifted"
    lifts = arms.moves(tool="robot.go_to_pose_cartesian")[2:]
    assert len(lifts) == 6
    assert lifts[0][2][2] == pytest.approx(0.02 + 0.025 / 3)


def test_example_declares_every_route(script):
    example = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == {"lifted", "slipped", "unreachable"}
