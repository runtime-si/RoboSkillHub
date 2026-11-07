"""moving-both-arms-together: FakeContext tests of the interleaved segments, the per-segment checks and the routing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "moving-both-arms-together"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["move_both_arms"].module


def _target(x, y, z, rotation=DOWN):
    return {"position": {"x": x, "y": y, "z": z}, "rotation": rotation}


def _about_z(deg: float) -> dict[str, float]:
    """Top-down hand turned ``deg`` about world z."""
    h = math.radians(deg) / 2.0
    # (rotation about z) * DOWN
    return {"w": 0.0, "x": math.cos(h), "y": math.sin(h), "z": 0.0}


def _turn_z(deg: float, q: dict[str, float]) -> dict[str, float]:
    """``q`` turned ``deg`` about world z."""
    h = math.radians(deg) / 2.0
    a = np.array([math.cos(h), 0.0, 0.0, math.sin(h)])
    b = np.array([q[k] for k in "wxyz"])
    w = a[0] * b[0] - a[1:] @ b[1:]
    v = a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])
    return dict(zip("wxyz", map(float, (w, *v)), strict=True))


class Arms:
    """Two arms; a move lands where commanded plus ``slip(arm, target)`` unless ``refuse`` says why not.

    ``roll_drift`` [deg about world z] is added to an arm's commanded rotation on
    every Cartesian move (a solver that leaves the wrist roll free); a last-joint
    turn of +d rad turns the hand ``joint_sign`` * d about its approach (tool z,
    world -z for a top-down hand).
    """

    def __init__(
        self,
        left=(0.45, 0.25, 0.92),
        right=(0.45, -0.25, 0.92),
        refuse=None,
        slip=None,
        roll_drift=(0.0, 0.0),
        joint_sign=1.0,
        joints=True,
    ):
        self.p = {0: np.array(left, float), 1: np.array(right, float)}
        self.rot = {0: DOWN, 1: DOWN}
        self.q = {0: [0.0] * 6, 1: [0.0] * 6}
        self.refuse = refuse or (lambda a, target, n: None)
        self.slip = slip or (lambda a, target, n: np.zeros(3))
        self.roll_drift = roll_drift
        self.joint_sign = joint_sign
        self.joints = joints
        self.log: list[tuple[int, np.ndarray]] = []
        self.turns: list[tuple[int, float]] = []
        self.ik_reply = {"success": True, "within_servo_tolerance": True}
        self.ik_calls = 0

    def ctx(self):
        def pose(arm_id=0):
            return {
                "pose": {
                    "position": dict(zip("xyz", map(float, self.p[arm_id]), strict=True)),
                    "rotation": self.rot[arm_id],
                }
            }

        def go(pose, arm_id=0, **kw):
            target = np.array([pose["position"][k] for k in "xyz"])
            n = sum(1 for a, _ in self.log if a == arm_id)
            self.log.append((arm_id, target.copy()))
            why = self.refuse(arm_id, target, n)
            if why == "raise":
                raise ToolError("robot.go_to_pose_cartesian", "unreachable")
            if why == "stall":
                self.p[arm_id] = self.p[arm_id] + 0.2 * (target - self.p[arm_id])
                return {"reached": False, "outcome": "timeout"}
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            self.p[arm_id] = target + self.slip(arm_id, target, n)
            self.rot[arm_id] = _turn_z(self.roll_drift[arm_id], pose["rotation"])
            return {"reached": True, "outcome": "reached"}

        def move_to_joints(joint_config, arm_id=0, **kw):
            new = list(joint_config["positions"])
            d = new[-1] - self.q[arm_id][-1]
            self.turns.append((arm_id, d))
            self.q[arm_id] = new
            # +d about tool z (world -z) is -d about world z.
            self.rot[arm_id] = _turn_z(-math.degrees(d) * self.joint_sign, self.rot[arm_id])
            return None

        def observation():
            if not self.joints:
                raise ToolError("robot.get_observation", "not served")
            return {"arms": [{"joint_state": {"positions": self.q[a]}} for a in (0, 1)]}

        def solve_ik(pose, arm_id=0, **kw):
            self.ik_calls += 1
            reply = self.ik_reply
            if callable(reply):
                return reply(arm_id, np.array([pose["position"][k] for k in "xyz"]))
            if reply == "raise":
                raise ToolError("robot.solve_ik", "pose unreachable")
            if reply == "missing":
                raise ToolError("robot.solve_ik", "not served")
            return reply

        return FakeContext(
            {
                "robot.get_ee_pose": pose,
                "robot.go_to_pose_cartesian": go,
                "robot.solve_ik": solve_ik,
                "robot.move_to_joints": move_to_joints,
                "robot.get_observation": observation,
            }
        )

    def moves(self, arm_id):
        return [t for a, t in self.log if a == arm_id]


def test_parallel_move_is_split_and_interleaved(script):
    arms = Arms()
    ctx = arms.ctx()
    out = script.run(
        ctx, left_target=_target(0.58, 0.25, 0.92), right_target=_target(0.58, -0.25, 0.92)
    )
    assert out["route"] == "reached", out["detail"]
    # 130 mm at 20 mm per step: 7 segments, each arm one move per segment.
    assert out["segments"] == 7
    left, right = arms.moves(0), arms.moves(1)
    assert len(left) == len(right) == 7
    np.testing.assert_allclose(left[0], [0.45 + 0.13 / 7, 0.25, 0.92], atol=1e-9)
    np.testing.assert_allclose(left[-1], [0.58, 0.25, 0.92], atol=1e-12)
    np.testing.assert_allclose(right[-1], [0.58, -0.25, 0.92], atol=1e-12)
    # The lead alternates: left first on odd segments, right first on even ones.
    assert [a for a, _ in arms.log[:4]] == [0, 1, 1, 0]
    assert [s["lead_arm"] for s in out["segment_errors"][:3]] == [0, 1, 0]
    # Straight lines: every waypoint on the segment from start to target.
    for k, p in enumerate(left, 1):
        np.testing.assert_allclose(p, [0.45 + 0.13 * k / 7, 0.25, 0.92], atol=1e-9)
    assert out["max_relative_error_m"] == pytest.approx(0.0, abs=1e-9)
    # Between turns the leading hand is one step ahead.
    assert out["max_transient_error_m"] == pytest.approx(0.13 / 7, abs=1e-9)
    assert out["left_residual_m"] == pytest.approx(0.0, abs=1e-9)
    assert out["min_hand_distance_m"] == pytest.approx(0.50)
    # arm_id only for the nonzero arm.
    for call in ctx.calls:
        if "arm_id" in call.kwargs:
            assert call.kwargs["arm_id"] == 1
    # Every waypoint of both arms previewed before the first move.
    assert arms.ik_calls == 14
    first_move = next(i for i, c in enumerate(ctx.calls) if c.tool.endswith("cartesian"))
    assert all(c.tool != "robot.solve_ik" for c in ctx.calls[first_move:])


def test_mirrored_paths_share_one_segment_count(script):
    # Left travels 160 mm, right only 40: both take 8 segments, so both arrive together.
    arms = Arms(left=(0.50, 0.28, 0.96), right=(0.50, -0.28, 0.96))
    out = script.run(
        arms.ctx(), left_target=_target(0.50, 0.12, 0.96), right_target=_target(0.50, -0.24, 0.96)
    )
    assert out["route"] == "reached"
    assert out["segments"] == 8
    right = arms.moves(1)
    assert len(right) == 8
    np.testing.assert_allclose(right[0], [0.50, -0.275, 0.96], atol=1e-9)
    # The planned relative pose changes along the way; measured against the plan, no error.
    assert out["max_relative_error_m"] == pytest.approx(0.0, abs=1e-9)


def test_rotation_counts_toward_the_segments(script):
    arms = Arms()
    out = script.run(
        arms.ctx(),
        left_target=_target(0.45, 0.25, 0.92, _about_z(45.0)),
        right_target=_target(0.45, -0.25, 0.92),
    )
    assert out["route"] == "reached"
    assert out["segments"] == 5  # 45 deg at 10 deg per segment
    assert arms.rot[0] == pytest.approx(_about_z(45.0))


def test_keep_relative_derives_the_right_target_and_halves_the_step(script):
    arms = Arms(left=(0.45, 0.10, 0.90), right=(0.45, -0.10, 0.90))
    out = script.run(arms.ctx(), left_target=_target(0.56, 0.05, 0.97), keep_relative=True)
    assert out["route"] == "reached", out["detail"]
    goal = out["right_target_pose"]["position"]
    assert [goal[k] for k in "xyz"] == pytest.approx([0.56, -0.15, 0.97])
    # Path 0.140 m at 10 mm (half the 20 mm bound): 14 segments.
    assert out["segments"] == 14
    for p0, p1 in zip(arms.moves(0), arms.moves(1), strict=True):
        np.testing.assert_allclose(p0 - p1, [0.0, 0.20, 0.0], atol=1e-9)
    assert out["max_transient_error_m"] <= 0.0101


def test_keep_relative_rides_the_right_hand_on_a_turning_left(script):
    arms = Arms(left=(0.45, 0.10, 0.90), right=(0.45, -0.10, 0.90))
    out = script.run(
        arms.ctx(), left_target=_target(0.45, 0.10, 0.90, _about_z(20.0)), keep_relative=True
    )
    assert out["route"] == "reached", out["detail"]
    # The right hand swings on an arc about the left, not a straight chord.
    goal = np.array([out["right_target_pose"]["position"][k] for k in "xyz"])
    assert float(np.linalg.norm(goal - np.array([0.45, 0.10, 0.90]))) == pytest.approx(0.20)
    for p0, p1 in zip(arms.moves(0), arms.moves(1), strict=True):
        assert float(np.linalg.norm(p1 - p0)) == pytest.approx(0.20)
    assert out["max_relative_error_deg"] == pytest.approx(0.0, abs=1e-4)


def test_keep_relative_refuses_targets_that_break_it(script):
    arms = Arms(left=(0.45, 0.10, 0.90), right=(0.45, -0.10, 0.90))
    out = script.run(
        arms.ctx(),
        left_target=_target(0.55, 0.10, 0.90),
        right_target=_target(0.55, -0.15, 0.90),
        keep_relative=True,
    )
    assert out["route"] == "unreachable" and "relative pose by 50 mm" in out["detail"]
    assert not arms.log


def test_keep_relative_diverges_when_a_hand_lags(script):
    # The hands drift apart 3 mm more each segment (each within its own reach
    # tolerance): over the 20 mm bound by segment 7.
    arms = Arms(
        left=(0.45, 0.10, 0.90),
        right=(0.45, -0.10, 0.90),
        slip=lambda a, t, n: np.array([0.0, 0.0, 0.0015 * (n + 1) * (1 if a == 0 else -1)]),
    )
    out = script.run(arms.ctx(), left_target=_target(0.60, 0.10, 0.90), keep_relative=True)
    assert out["route"] == "diverged", out["detail"]
    assert "relative pose" in out["detail"] and "after segment 7/" in out["detail"]
    assert len(out["segment_errors"]) == 7
    assert out["segment_errors"][-1]["relative_error_m"] == pytest.approx(0.021)
    # Stopped there: no move after the breach.
    assert len(arms.moves(0)) == len(arms.moves(1)) == 7


def test_without_keep_relative_drift_is_reported_not_routed(script):
    arms = Arms(slip=lambda a, t, n: np.array([0.0, 0.0, -0.015]) if a == 1 else np.zeros(3))
    out = script.run(
        arms.ctx(), left_target=_target(0.58, 0.25, 0.92), right_target=_target(0.58, -0.25, 0.92)
    )
    assert out["route"] == "reached"
    assert out["max_relative_error_m"] == pytest.approx(0.015)
    assert out["right_residual_m"] == pytest.approx(0.015)


@pytest.mark.parametrize("how", ["raise", "unreachable", "stall"])
def test_a_hand_stopping_mid_way_diverges_and_both_stop(script, how):
    arms = Arms(refuse=lambda a, t, n: how if a == 1 and n == 3 else None)
    out = script.run(
        arms.ctx(), left_target=_target(0.58, 0.25, 0.92), right_target=_target(0.58, -0.25, 0.92)
    )
    assert out["route"] == "diverged", out["detail"]
    assert "arm 1 stopped" in out["detail"] and "waypoint 4/7" in out["detail"]
    # Segment 4 is right-led: the left never took its 4th step.
    assert len(arms.moves(0)) == 3 and len(arms.moves(1)) == 4
    assert out["right_pose"]["position"]["x"] < 0.45 + 0.13 * 4 / 7


def test_a_first_move_refused_in_place_is_unreachable(script):
    arms = Arms(refuse=lambda a, t, n: "raise" if a == 0 else None)
    out = script.run(
        arms.ctx(), left_target=_target(0.58, 0.25, 0.92), right_target=_target(0.58, -0.25, 0.92)
    )
    assert out["route"] == "unreachable" and "neither hand moved" in out["detail"]
    assert not arms.moves(1)


def test_a_waypoint_out_of_reach_is_refused_before_moving(script):
    arms = Arms()
    arms.ik_reply = lambda a, p: (
        {"success": True, "within_servo_tolerance": p[0] < 0.55 or a == 0}
    )
    out = script.run(
        arms.ctx(), left_target=_target(0.58, 0.25, 0.92), right_target=_target(0.58, -0.25, 0.92)
    )
    assert out["route"] == "unreachable" and "arm 1 waypoint 6/7" in out["detail"]
    assert not arms.log

    arms = Arms()
    arms.ik_reply = "raise"  # a real station raises an unreachable pose
    out = script.run(
        arms.ctx(), left_target=_target(0.58, 0.25, 0.92), right_target=_target(0.58, -0.25, 0.92)
    )
    assert out["route"] == "unreachable" and not arms.log


def test_preview_judges_the_reported_miss_not_the_servo_ball(script):
    # A waypoint the solver misses by 5 mm (outside a servo's 4 mm ball) is
    # still one the arm ends near; 30 mm is not.
    def reply(miss):
        return lambda a, p: {
            "success": True,
            "position_error_m": miss if p[0] > 0.55 else 0.0,
            "within_servo_tolerance": not (p[0] > 0.55 and miss > 0.004),
        }

    for miss, route in ((0.005, "reached"), (0.030, "unreachable")):
        arms = Arms()
        arms.ik_reply = reply(miss)
        out = script.run(
            arms.ctx(),
            left_target=_target(0.58, 0.25, 0.92),
            right_target=_target(0.58, -0.25, 0.92),
        )
        assert out["route"] == route, out["detail"]
    assert "by 30 mm" in out["detail"]


def test_a_free_wrist_roll_is_finished_on_the_last_joint(script):
    # Arm 1's wrist ends 12 deg off whatever it is sent; the joint turns the
    # other way from what the first guess assumes, so the second turn learns it.
    arms = Arms(roll_drift=(0.0, 12.0), joint_sign=-1.0)
    out = script.run(
        arms.ctx(), left_target=_target(0.50, 0.25, 0.92), right_target=_target(0.50, -0.25, 0.92)
    )
    assert out["route"] == "reached", out["detail"]
    assert {a for a, _ in arms.turns} == {1}
    for seg in out["segment_errors"]:
        assert seg["rotation_off_deg"][1] <= 3.0 + 1e-6
    assert out["max_relative_error_deg"] <= 3.0 + 1e-6
    assert out["roll_turns"] == len(arms.turns)


def test_an_unfinished_roll_is_reported_and_bounds_a_carry(script):
    arms = Arms(roll_drift=(0.0, 12.0), joints=False)
    out = script.run(
        arms.ctx(), left_target=_target(0.50, 0.25, 0.92), right_target=_target(0.50, -0.25, 0.92)
    )
    assert out["route"] == "reached"
    seg = out["segment_errors"][0]
    assert seg["rotation_off_deg"] == pytest.approx([0.0, 12.0], abs=1e-4)
    assert seg["relative_error_deg"] == pytest.approx(12.0, abs=1e-4)
    assert seg["relative_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert "wrist roll was left" in out["detail"]

    arms = Arms(left=(0.45, 0.10, 0.90), right=(0.45, -0.10, 0.90), roll_drift=(0.0, 12.0))
    out = script.run(
        arms.ctx(), left_target=_target(0.50, 0.10, 0.90), keep_relative=True, fix_roll=False
    )
    assert out["route"] == "diverged" and "12.0 deg off" in out["detail"]
    assert not arms.turns


def test_a_roll_past_the_joint_limit_is_left_and_said(script):
    arms = Arms(roll_drift=(0.0, 12.0))
    arms.q[1][-1] = 2.0
    out = script.run(
        arms.ctx(), left_target=_target(0.50, 0.25, 0.92), right_target=_target(0.50, -0.25, 0.92)
    )
    assert out["route"] == "reached"
    assert "no room to turn" in out["detail"] and not arms.turns


def test_no_ik_preview_still_moves(script):
    arms = Arms()
    arms.ik_reply = "missing"
    out = script.run(
        arms.ctx(), left_target=_target(0.58, 0.25, 0.92), right_target=_target(0.58, -0.25, 0.92)
    )
    assert out["route"] == "reached"


def test_hands_planned_too_close_are_refused(script):
    arms = Arms(left=(0.50, 0.20, 0.90), right=(0.50, -0.20, 0.90))
    out = script.run(
        arms.ctx(), left_target=_target(0.50, 0.03, 0.90), right_target=_target(0.50, -0.03, 0.90)
    )
    assert out["route"] == "unreachable" and "60 mm apart" in out["detail"]
    assert not arms.log
    out = script.run(
        arms.ctx(),
        left_target=_target(0.50, 0.03, 0.90),
        right_target=_target(0.50, -0.03, 0.90),
        min_hand_distance=0.05,
    )
    assert out["route"] == "reached"
    assert out["min_hand_distance_m"] == pytest.approx(0.06)


def test_refuses_one_arm_twice_or_a_missing_second_arm(script):
    arms = Arms()
    out = script.run(arms.ctx(), left_target=_target(0.5, 0.25, 0.92), right_arm=0)
    assert out["route"] == "unreachable" and "both hands are arm 0" in out["detail"]

    arms = Arms(right=(0.45, 0.25, 0.92))  # a connector that clamps arm_id reports arm 0
    out = script.run(
        arms.ctx(), left_target=_target(0.5, 0.25, 0.92), right_target=_target(0.5, -0.2, 0.92)
    )
    assert out["route"] == "unreachable" and "separate arm 1" in out["detail"]
    assert not arms.log

    arms = Arms()
    arms.rot[1] = {"w": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}
    out = script.run(
        arms.ctx(), left_target=_target(0.5, 0.25, 0.92), right_target=_target(0.5, -0.2, 0.92)
    )
    assert out["route"] == "unreachable"


def test_needs_a_right_target_unless_it_rides_on_the_left(script):
    out = script.run(Arms().ctx(), left_target=_target(0.5, 0.25, 0.92))
    assert out["route"] == "unreachable" and "no right_target" in out["detail"]


def test_too_long_a_move_is_refused(script):
    arms = Arms()
    out = script.run(
        arms.ctx(),
        left_target=_target(0.58, 0.25, 0.92),
        right_target=_target(0.58, -0.25, 0.92),
        max_segments=5,
    )
    assert out["route"] == "unreachable" and "7 segments" in out["detail"]
    assert not arms.log


def test_targets_without_rotation_keep_the_hands_rotation(script):
    arms = Arms()
    out = script.run(
        arms.ctx(),
        left_target={"position": {"x": 0.5, "y": 0.25, "z": 0.92}},
        right_target={"position": {"x": 0.5, "y": -0.25, "z": 0.92}},
    )
    assert out["route"] == "reached"
    assert arms.rot[0] == pytest.approx(DOWN) and arms.rot[1] == pytest.approx(DOWN)


def test_example_declares_every_route():
    example = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    source = (SKILL_DIR / "scripts" / "move_both_arms.py").read_text()
    for route in ("reached", "diverged", "unreachable"):
        assert f'"{route}"' in source
    assert declared == {"reached", "diverged", "unreachable"}
