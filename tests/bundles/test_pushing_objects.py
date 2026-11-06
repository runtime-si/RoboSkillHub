"""pushing-objects: FakeContext tests of the push geometry, the fallback rotation and the routing."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "pushing-objects"
KEEP = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
TOP_A = {"w": 0.0, "x": 0.7071, "y": 0.7071, "z": 0.0}
TOP_B = {"w": 0.0, "x": -0.7071, "y": 0.7071, "z": 0.0}
# A 30 mm cube on a table at z = 0, centre (0.40, 0.10, 0.015).
CUBE = {
    "center": {"x": 0.40, "y": 0.10, "z": 0.015},
    "extent": {"x": 0.015, "y": 0.015, "z": 0.015},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}
GOAL_PLUS_X = {"x": 0.50, "y": 0.10, "z": 0.015}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["push_object"].module


class Arm:
    """Moves land where commanded unless ``block`` (a predicate on the call) says otherwise."""

    def __init__(self, start=(0.30, 0.0, 0.25), refuse=None, stop_push_at=None):
        self.p = np.array(start, dtype=float)
        self.rot = KEEP
        self.refuse = refuse or (lambda tool, pose: None)
        self.stop_push_at = stop_push_at
        self.moves = []

    def _move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy(), pose["rotation"]))
            why = self.refuse(tool, pose)
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            if (
                self.stop_push_at is not None
                and tool.endswith("cartesian")
                and target[0] > self.p[0] + 0.01
            ):
                target = target.copy()
                target[0] = min(target[0], self.stop_push_at)
            self.p, self.rot = target, pose["rotation"]
            return {"reached": True, "outcome": "reached"}

        return go

    def ctx(self, frame=True):
        tools = {
            "robot.get_ee_pose": lambda **kw: {
                "pose": {
                    "position": dict(zip("xyz", map(float, self.p), strict=True)),
                    "rotation": self.rot,
                }
            },
            "robot.go_to_pose": self._move("robot.go_to_pose"),
            "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
            "robot.close_gripper": {"position": 0.0},
            "robot.open_gripper": {"position": 1.0},
        }
        if frame:
            tools["robot.grasp_frame"] = {"rotation": TOP_A, "rotation_flipped": TOP_B}
        else:

            def missing(**kw):
                raise ToolError("robot.grasp_frame", "not served")

            tools["robot.grasp_frame"] = missing
        return FakeContext(tools)


def test_push_geometry_and_sequence(script):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(ctx, target_obb=CUBE, goal_position=GOAL_PLUS_X)
    assert out["route"] == "pushed", out["detail"]
    assert out["push_distance"] == pytest.approx(0.10)
    # Contact: back face at x = 0.385, 3 mm padding -> 0.382; z = max(0.5 * 0.03, 0.015) = 0.015.
    # Hover backoff cap: min(0.06, max(0.02, 0.09 - 0.018)) = 0.06 behind contact, 0.08 up.
    names = [m[0] for m in arm.moves]
    assert names == [
        "robot.go_to_pose",  # hover (start is already above it)
        "robot.go_to_pose_cartesian",  # straight down
        "robot.go_to_pose_cartesian",  # push
        "robot.go_to_pose_cartesian",  # lift
    ]
    hover, lowered, finish, lift = (m[1] for m in arm.moves)
    np.testing.assert_allclose(hover, [0.382 - 0.06, 0.10, 0.015 + 0.08], atol=1e-9)
    np.testing.assert_allclose(lowered, [0.322, 0.10, 0.015], atol=1e-9)
    np.testing.assert_allclose(finish, [0.382 + 0.10, 0.10, 0.015], atol=1e-9)
    np.testing.assert_allclose(lift, [0.482, 0.10, 0.095], atol=1e-9)
    assert all(m[2] == TOP_A for m in arm.moves)  # fingers across the push first
    assert out["achieved"] == pytest.approx(0.10)
    assert out["expected_center"] == {
        "x": pytest.approx(0.50),
        "y": pytest.approx(0.10),
        "z": pytest.approx(0.015),
    }
    calls = [c.tool for c in ctx.calls]
    assert calls.index("robot.close_gripper") < calls.index("robot.go_to_pose")
    assert calls[-1] == "robot.open_gripper"


def test_rises_first_from_below_and_floors_the_contact_on_a_flat_object(script):
    flat = {
        **CUBE,
        "center": {"x": 0.40, "y": 0.10, "z": 0.004},
        "extent": {"x": 0.03, "y": 0.03, "z": 0.004},
    }
    arm = Arm(start=(0.30, 0.0, 0.02))
    out = script.run(arm.ctx(), target_obb=flat, goal_position={"x": 0.40, "y": 0.25, "z": 0.0})
    assert out["route"] == "pushed"
    rise, hover, lowered = (m[1] for m in arm.moves[:3])
    np.testing.assert_allclose(rise, [0.30, 0.0, hover[2]], atol=1e-9)
    # 8 mm tall: the middle (4 mm) is under the 15 mm floor, so the hand pushes at 15 mm.
    assert lowered[2] == pytest.approx(0.0 + 0.015)


def test_a_corner_first_box_is_met_at_its_corner(script):
    q = {"w": math.cos(math.pi / 8), "x": 0.0, "y": 0.0, "z": math.sin(math.pi / 8)}
    arm = Arm()
    script.run(arm.ctx(), target_obb={**CUBE, "orientation": q}, goal_position=GOAL_PLUS_X)
    lowered = arm.moves[1][1]
    # Rotated 45 deg, the back corner reaches 0.015 * sqrt(2) behind the centre.
    assert lowered[0] == pytest.approx(0.40 - 0.015 * math.sqrt(2) - 0.003 - 0.06, abs=1e-9)


def test_tries_the_flipped_wrist_then_the_hand_s_own_rotation(script):
    def refuse(tool, pose):
        return (
            "joint limit"
            if pose["rotation"] in (TOP_A, TOP_B) and tool == "robot.go_to_pose"
            else None
        )

    arm = Arm(refuse=refuse)
    ctx = arm.ctx()
    out = script.run(ctx, target_obb=CUBE, goal_position=GOAL_PLUS_X)
    assert out["route"] == "pushed"
    assert [m[2] for m in arm.moves[:3]] == [TOP_A, TOP_B, KEEP]
    assert all(m[2] == KEEP for m in arm.moves[3:])
    heading = ctx.calls_to("robot.grasp_frame")[0].kwargs["close_heading_deg"]
    assert heading == pytest.approx(90.0)  # fingers across a +x push


def test_keep_rotation_tries_the_hand_s_own_first(script):
    arm = Arm()
    out = script.run(arm.ctx(), target_obb=CUBE, goal_position=GOAL_PLUS_X, keep_rotation=True)
    assert out["route"] == "pushed"
    assert all(m[2] == KEEP for m in arm.moves)


def test_unreachable_hover_and_descent(script):
    arm = Arm(refuse=lambda tool, pose: "no IK" if tool == "robot.go_to_pose" else None)
    out = script.run(arm.ctx(frame=False), target_obb=CUBE, goal_position=GOAL_PLUS_X)
    assert out["route"] == "unreachable" and out["detail"].startswith("hover")
    assert out["achieved"] == 0.0

    arm = Arm(refuse=lambda tool, pose: "contact" if tool.endswith("cartesian") else None)
    arm.p = np.array([0.30, 0.0, 0.25])
    out = script.run(arm.ctx(), target_obb=CUBE, goal_position=GOAL_PLUS_X)
    assert out["route"] == "unreachable" and out["detail"].startswith("descent")
    assert "stopped" in out["detail"]


def test_a_descent_the_table_stops_short_pushes_from_where_it_stopped(script):
    class Short(Arm):
        def _move(self, tool):
            go = super()._move(tool)

            def wrapped(pose, **kwargs):
                z = pose["position"]["z"]
                if tool.endswith("cartesian") and z < 0.02 and self.p[2] > 0.05:
                    go({**pose, "position": {**pose["position"], "z": 0.0206}}, **kwargs)
                    return {"reached": False, "outcome": "timeout", "remaining_distance": 0.0056}
                return go(pose, **kwargs)

            return wrapped

    arm = Short()
    out = script.run(arm.ctx(), target_obb=CUBE, goal_position=GOAL_PLUS_X)
    assert out["route"] == "pushed", out["detail"]
    finish = arm.moves[2][1]
    assert finish[2] == pytest.approx(0.0206)  # pushed at the height the table allowed


def test_a_blocked_push_is_not_pushed_and_still_opens(script):
    arm = Arm(stop_push_at=0.40)  # the hand stops 18 mm past contact
    ctx = arm.ctx()
    out = script.run(ctx, target_obb=CUBE, goal_position=GOAL_PLUS_X)
    assert out["route"] == "not_pushed"
    assert out["achieved"] == pytest.approx(0.018, abs=1e-9)
    assert "travelled 18 mm" in out["detail"] and "needed 75" in out["detail"]
    assert ctx.calls[-1].tool == "robot.open_gripper"


def test_a_goal_too_close_does_nothing(script):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(ctx, target_obb=CUBE, goal_position={"x": 0.41, "y": 0.11, "z": 0.0})
    assert out["route"] == "not_pushed" and "under 30 mm" in out["detail"]
    assert ctx.calls == []


def test_max_distance_caps_the_push(script):
    arm = Arm()
    out = script.run(
        arm.ctx(),
        target_obb=CUBE,
        goal_position={"x": 0.80, "y": 0.10, "z": 0.0},
        max_distance=0.12,
    )
    assert out["push_distance"] == pytest.approx(0.12)
    assert arm.moves[2][1][0] == pytest.approx(0.382 + 0.12)
