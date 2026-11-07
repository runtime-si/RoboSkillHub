"""centering-a-part-with-the-gripper: FakeContext tests of the close-turn-close sequence, the jaw-gap check and the routing."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "centering-a-part-with-the-gripper"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
HAND = {
    "span_m": 0.094,
    "width_fit": {"source": "measured", "at_open_m": 0.094, "at_closed_m": 0.0},
    "approach_axis": {"x": 0.0, "y": 0.0, "z": 1.0},
}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["center_with_gripper"].module


def quat(d):
    return np.array([d[k] for k in "wxyz"])


def turn_about_z(q, deg):
    """q followed by a turn of deg about the tool's own z."""
    h = math.radians(deg) / 2.0
    w, x, y, z = q
    c, s = math.cos(h), math.sin(h)
    return np.array([w * c - z * s, x * c + y * s, -x * s + y * c, w * s + z * c])


def signed_turn(a, b):
    """The turn about the tool's z from a to b [deg] (b = a followed by it)."""
    w, x, y, z = a
    rel = np.array(
        [
            w * b[0] + x * b[1] + y * b[2] + z * b[3],
            w * b[1] - x * b[0] - y * b[3] + z * b[2],
            w * b[2] + x * b[3] - y * b[0] - z * b[1],
            w * b[3] - x * b[2] + y * b[1] - z * b[0],
        ]
    )
    return math.degrees(2.0 * math.atan2(rel[3], rel[0]))


def angle_deg(a, b):
    return math.degrees(2.0 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


class Hand:
    """An arm that turns where commanded unless ``refuse`` says otherwise (or keeps
    its roll, ``roll_free``, as a solver that leaves wrist roll free does), a last
    joint that turns the tool about its own z, and jaws that close to ``widths``
    in turn (an open fraction of the 94 mm span)."""

    def __init__(
        self, widths=(0.030, 0.030), refuse=None, drift=None, roll_free=False, joints_refused=False
    ):
        self.p = np.array([0.40, 0.10, 0.018])
        self.q = quat(DOWN)
        self.j = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.3])
        self.widths = list(widths)
        self.refuse = refuse or (lambda pose: None)
        self.drift = drift
        self.roll_free = roll_free
        self.joints_refused = joints_refused
        self.moves = []
        self.joint_moves = []
        self.jaw = []

    def go(self, pose, **kwargs):
        target = quat(pose["rotation"])
        self.moves.append(target)
        why = self.refuse(pose)
        if why == "raise":
            raise ToolError("robot.go_to_pose_cartesian", "joint limit")
        if why:
            return {"reached": False, "outcome": "unreachable", "reason": why}
        if not self.roll_free:
            self.j = self.j.copy()
            self.j[-1] += math.radians(signed_turn(self.q, target))
            self.q = target
        if self.drift is not None:
            self.p = self.p + self.drift
        return {"reached": True, "outcome": "reached"}

    def to_joints(self, joint_config, **kwargs):
        target = np.array(joint_config["positions"])
        self.joint_moves.append(target)
        if self.joints_refused:
            raise ToolError("robot.move_to_joints", "the station bridge takes poses only")
        self.q = turn_about_z(self.q, math.degrees(target[-1] - self.j[-1]))
        self.j = target
        return None

    def close(self, **kwargs):
        self.jaw.append("close")
        return {"position": self.widths.pop(0) / 0.094}

    def open(self, **kwargs):
        self.jaw.append("open")
        return {"position": 1.0}

    def ctx(self, hand=HAND, **overrides):
        return FakeContext(
            {
                "robot.describe_gripper": hand,
                "robot.get_ee_pose": lambda **kw: {
                    "pose": {
                        "position": dict(zip("xyz", map(float, self.p), strict=True)),
                        "rotation": dict(zip("wxyz", map(float, self.q), strict=True)),
                    }
                },
                "robot.go_to_pose_cartesian": self.go,
                "robot.close_gripper": self.close,
                "robot.open_gripper": self.open,
                "robot.get_gripper": {"position": 0.0},
                "robot.get_observation": lambda **kw: {
                    "arms": [{"joint_state": {"positions": self.j.copy()}}]
                },
                "robot.move_to_joints": self.to_joints,
                **{k.replace("__", "."): v for k, v in overrides.items()},
            }
        )


def test_close_open_turn_close_open_and_turn_back(script):
    hand = Hand(widths=(0.031, 0.030))
    ctx = hand.ctx()
    out = script.run(ctx)
    assert out["route"] == "centered", out["detail"]
    assert hand.jaw == ["close", "open", "close", "open"]
    assert out["first_width"] == pytest.approx(0.031)
    assert out["second_width"] == pytest.approx(0.030)
    assert out["turned_deg"] == pytest.approx(90.0) and out["turned_by"] == "pose"
    assert out["returned"] is True and hand.joint_moves == []
    turned, back = hand.moves
    assert angle_deg(turned, quat(DOWN)) == pytest.approx(90.0)
    np.testing.assert_allclose(np.abs(turned), np.abs(turn_about_z(quat(DOWN), 90.0)), atol=1e-9)
    np.testing.assert_allclose(back, quat(DOWN), atol=1e-9)
    # The turn is in place: the tool centre is commanded where it stood.
    pose = ctx.calls_to("robot.go_to_pose_cartesian")[0].kwargs["pose"]
    assert pose["position"] == {"x": 0.40, "y": 0.10, "z": pytest.approx(0.018)}
    calls = [c.tool for c in ctx.calls]
    assert calls.index("robot.open_gripper") < calls.index("robot.go_to_pose_cartesian")
    assert "arm_id" not in ctx.calls_to("robot.close_gripper")[0].kwargs


def test_clockwise_turns_the_other_way_first(script):
    hand = Hand()
    out = script.run(hand.ctx(), clockwise=True)
    assert out["turned_deg"] == pytest.approx(-90.0)
    np.testing.assert_allclose(
        np.abs(hand.moves[0]), np.abs(turn_about_z(quat(DOWN), -90.0)), atol=1e-9
    )


def test_nothing_caught_on_the_first_close_does_not_turn(script):
    hand = Hand(widths=(0.0018,))
    out = script.run(hand.ctx())
    assert out["route"] == "nothing_caught"
    assert "1.8 mm" in out["detail"] and "first close" in out["detail"]
    assert hand.jaw == ["close", "open"]
    assert hand.moves == []


def test_nothing_caught_after_the_turn_opens_and_turns_back(script):
    hand = Hand(widths=(0.030, 0.001))
    out = script.run(hand.ctx())
    assert out["route"] == "nothing_caught" and "after the turn" in out["detail"]
    assert hand.jaw == ["close", "open", "close", "open"]
    assert len(hand.moves) == 2 and out["returned"] is True
    np.testing.assert_allclose(hand.q, quat(DOWN), atol=1e-9)


def test_check_held_false_and_a_thin_part_threshold(script):
    out = script.run(Hand(widths=(0.001, 0.001)).ctx(), check_held=False)
    assert out["route"] == "centered"
    # A 2.5 mm part is held under a 1 mm threshold, empty under the 3 mm default.
    assert (
        script.run(Hand(widths=(0.0025, 0.0025)).ctx(), min_held_width=0.001)["route"] == "centered"
    )
    assert script.run(Hand(widths=(0.0025,)).ctx())["route"] == "nothing_caught"


def test_the_closed_stop_comes_from_the_width_fit(script):
    hand = {**HAND, "width_fit": {"at_open_m": 0.094, "at_closed_m": 0.010}}
    # Fraction 0.05 reads 0.010 + 0.05 * 0.084 = 14.2 mm: 4.2 mm over the stop, held.
    h = Hand(widths=(0.05 * 0.094, 0.01 * 0.094))
    out = script.run(h.ctx(hand=hand))
    assert out["first_width"] == pytest.approx(0.0142)
    assert out["route"] == "nothing_caught"  # 10.84 mm is within 3 mm of the 10 mm stop


def test_an_unreadable_hand_is_not_an_empty_one(script):
    hand = Hand()
    ctx = hand.ctx(hand={})  # no span: the gap cannot be converted
    out = script.run(ctx)
    assert out["route"] == "centered"
    assert out["first_width"] == -1.0 and "unread" in out["detail"]


def test_the_gap_falls_back_to_get_gripper(script):
    hand = Hand()
    ctx = hand.ctx(robot__close_gripper={"ok": True})  # no position in the reply
    out = script.run(ctx)
    assert out["route"] == "nothing_caught"  # get_gripper reads 0.0: shut on nothing
    assert ctx.calls_to("robot.get_gripper")


def test_a_refused_turn_tries_the_other_way(script):
    hand = Hand(refuse=lambda pose: "joint limit" if len(hand.moves) == 1 else None)
    out = script.run(hand.ctx())
    assert out["route"] == "centered", out["detail"]
    assert out["turned_deg"] == pytest.approx(-90.0)
    assert len(hand.moves) == 3


@pytest.mark.parametrize("refusal", ["raise", "timeout"])
def test_both_turns_refused_and_no_joint_moves_is_unreachable(script, refusal):
    hand = Hand(refuse=lambda pose: refusal if len(hand.moves) <= 2 else None, joints_refused=True)
    out = script.run(hand.ctx())
    assert out["route"] == "unreachable"
    assert "neither wrist turn" in out["detail"] and "last joint +90" in out["detail"]
    assert hand.jaw == ["close", "open"]
    assert len(hand.joint_moves) == 2
    assert out["returned"] is True  # the third pose move, back to the start rotation, arrived


def test_a_roll_free_solver_turns_by_the_last_joint_and_back(script):
    hand = Hand(roll_free=True)
    ctx = hand.ctx()
    out = script.run(ctx)
    assert out["route"] == "centered", out["detail"]
    assert out["turned_by"] == "last_joint" and out["turned_deg"] == pytest.approx(90.0)
    assert len(hand.moves) == 3  # +90 and -90 by pose (both kept the roll), then back by pose
    first, back = hand.joint_moves
    assert first[-1] == pytest.approx(0.3 + math.pi / 2) and np.allclose(first[:-1], back[:-1])
    np.testing.assert_allclose(back, [0.0, 0.5, 0.5, 0.0, 0.0, 0.3])
    assert out["returned"] is True
    np.testing.assert_allclose(hand.q, quat(DOWN), atol=1e-9)
    assert ctx.call_count("robot.get_observation") == 1


def test_a_last_joint_turning_the_other_way_still_counts(script):
    hand = Hand(roll_free=True)
    real = hand.to_joints

    def mirrored(joint_config, **kwargs):  # a joint whose axis points against the approach axis
        q = np.array(joint_config["positions"])
        q[-1] = 2 * hand.j[-1] - q[-1] if abs(q[-1] - 0.3) > 1e-9 else q[-1]
        return real({"positions": q})

    out = script.run(hand.ctx(robot__move_to_joints=mirrored))
    assert out["route"] == "centered" and out["turned_deg"] == pytest.approx(-90.0)


def test_roll_free_without_joint_moves_is_unreachable(script):
    hand = Hand(roll_free=True, joints_refused=True)
    out = script.run(hand.ctx())
    assert out["route"] == "unreachable"
    assert "90 deg and 0 mm from the target" in out["detail"]


def test_joint_fallback_off_never_reads_joints(script):
    hand = Hand(roll_free=True)
    ctx = hand.ctx()
    out = script.run(ctx, joint_fallback=False)
    assert out["route"] == "unreachable"
    assert ctx.call_count("robot.get_observation") == 0 and hand.joint_moves == []


def test_a_turn_that_drifts_too_far_does_not_count(script):
    hand = Hand(drift=np.array([0.02, 0.0, 0.0]))
    out = script.run(hand.ctx())
    assert out["route"] == "unreachable"
    assert "mm from the target" in out["detail"]


def test_release_false_keeps_the_part_and_return_false_stays_turned(script):
    hand = Hand()
    out = script.run(hand.ctx(), release=False, return_to_start=False)
    assert out["route"] == "centered"
    assert hand.jaw == ["close", "open", "close"]
    assert len(hand.moves) == 1 and out["returned"] is False


def test_skip_first_close(script):
    hand = Hand(widths=(0.030,))
    out = script.run(hand.ctx(), skip_first_close=True)
    assert out["route"] == "centered"
    assert hand.jaw == ["open", "close", "open"]
    assert out["first_width"] == -1.0


def test_a_failed_turn_back_is_reported_but_still_centred(script):
    hand = Hand(
        refuse=lambda pose: "joint limit" if len(hand.moves) == 2 else None, joints_refused=True
    )
    out = script.run(hand.ctx())
    assert out["route"] == "centered"
    assert out["returned"] is False and "turn back did not arrive" in out["detail"]


def test_arm_id_is_passed_when_nonzero(script):
    hand = Hand()
    ctx = hand.ctx()
    script.run(ctx, arm_id=1)
    for tool in (
        "robot.close_gripper",
        "robot.open_gripper",
        "robot.get_ee_pose",
        "robot.go_to_pose_cartesian",
    ):
        assert all(c.kwargs.get("arm_id") == 1 for c in ctx.calls_to(tool)), tool
