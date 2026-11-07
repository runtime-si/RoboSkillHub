"""picking-and-placing-two-objects: FakeContext tests of the assignment, the interleaving, the clearance gate and the routing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "picking-and-placing-two-objects"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
ROT_A = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
ROT_B = {"w": 0.0, "x": 0.0, "y": 1.0, "z": 0.0}
LEFT_HOME = np.array([0.30, 0.30, 0.20])
RIGHT_HOME = np.array([0.30, -0.30, 0.20])


def _cube(x, y, yaw_deg=0.0, half=0.015):
    h = math.radians(yaw_deg) / 2.0
    return {
        "center": {"x": x, "y": y, "z": half},
        "extent": {"x": half, "y": half, "z": half},
        "orientation": {"w": math.cos(h), "x": 0.0, "y": 0.0, "z": math.sin(h)},
    }


def _v(x, y, z):
    return {"x": x, "y": y, "z": z}


A = _cube(0.40, 0.25)  # on the left arm's side
B = _cube(0.40, -0.25)  # on the right arm's side
PLACE_A = _v(0.32, 0.25, 0.015)
PLACE_B = _v(0.32, -0.25, 0.015)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["pick_place_two"].module


class Arms:
    """Two arms; moves land where commanded unless ``refuse(tool, arm, target)`` says why not.

    ``reach(arm, target)`` is the IK preview; ``close_on(arm)`` the jaw reading after a close.
    """

    def __init__(self, refuse=None, reach=None, close_on=None, second_arm=True):
        self.p = {0: LEFT_HOME.copy(), 1: RIGHT_HOME.copy() if second_arm else LEFT_HOME.copy()}
        self.rot = {0: ROT_A, 1: ROT_A}
        self.grip = {0: 1.0, 1: 1.0}
        self.q = {0: [0.0] * 6, 1: [0.0] * 6}
        self.refuse = refuse or (lambda tool, a, target: None)
        self.reach = reach or (lambda a, target: True)
        self.close_on = close_on or (lambda a: 0.33)
        self.after_move = None
        self.log = []
        self.gaps = []

    def _move(self, tool):
        def go(pose, arm_id=0, **kw):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.log.append((tool, arm_id, target.copy()))
            why = self.refuse(tool, arm_id, target)
            if why == "raise":
                raise ToolError(tool, "unreachable")
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            self.p[arm_id] = target
            self.rot[arm_id] = pose["rotation"]
            self.gaps.append(float(np.linalg.norm(self.p[0] - self.p[1])))
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
            self.grip[arm_id] = self.close_on(arm_id)
            return {"position": self.grip[arm_id]}

        def solve_ik(pose, arm_id=0, **kw):
            target = np.array([pose["position"][k] for k in "xyz"])
            ok = self.reach(arm_id, target)
            if ok == "raise":
                raise ToolError("robot.solve_ik", "no solution")
            return {
                "success": bool(ok),
                "position_error_m": 0.001 if ok else 0.08,
                "within_servo_tolerance": bool(ok),
            }

        return FakeContext(
            {
                "robot.get_ee_pose": pose,
                "robot.get_gripper": lambda arm_id=0: {"position": self.grip[arm_id]},
                "robot.go_to_pose": self._move("robot.go_to_pose"),
                "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
                "robot.open_gripper": open_gripper,
                "robot.close_gripper": close_gripper,
                "robot.grasp_frame": {"rotation": ROT_A, "rotation_flipped": ROT_B},
                "robot.solve_ik": solve_ik,
                "robot.get_observation": {
                    "arms": [{"joint_state": {"positions": [0.0] * 6}} for _ in (0, 1)]
                },
                "robot.move_to_joints": None,
            }
        )

    def events(self, arm_id=None):
        """(kind, arm) in call order: kind is the tool's short name or open / close."""
        out = []
        for e in self.log:
            kind = e[0].split(".")[-1]
            if arm_id is None or e[1] == arm_id:
                out.append((kind, e[1]))
        return out

    def targets(self, arm_id):
        return [e[2] for e in self.log if e[0].startswith("robot.") and e[1] == arm_id]


def run(script, arms, a=A, b=B, pa=PLACE_A, pb=PLACE_B, **kw):
    return script.run(arms.ctx(), object_a_obb=a, object_b_obb=b, place_a=pa, place_b=pb, **kw)


def test_happy_path_assigns_by_side_and_places_both(script):
    arms = Arms()
    out = run(script, arms)
    assert out["route"] == "both_placed", out["detail"]
    assert (out["arm_a"], out["arm_b"]) == (0, 1)
    assert (out["outcome_a"], out["outcome_b"]) == ("placed", "placed")
    assert out["failed_arms"] == ""
    assert not out["staggered"]
    left = arms.targets(0)
    # hover 0.10 over the cube's centre, down to the centre, lift 0.10, carry to
    # 0.07 over the release, release 0.01 over rest, raise, home.
    np.testing.assert_allclose(left[0], [0.40, 0.25, 0.115], atol=1e-9)
    np.testing.assert_allclose(left[1], [0.40, 0.25, 0.015], atol=1e-9)
    np.testing.assert_allclose(left[2], [0.40, 0.25, 0.115], atol=1e-9)
    np.testing.assert_allclose(left[3], [0.32, 0.25, 0.095], atol=1e-9)
    np.testing.assert_allclose(left[4], [0.32, 0.25, 0.025], atol=1e-9)
    np.testing.assert_allclose(left[5], [0.32, 0.25, 0.095], atol=1e-9)
    np.testing.assert_allclose(left[6], LEFT_HOME, atol=1e-9)
    assert out["release_pose_a"]["position"]["z"] == pytest.approx(0.025)
    assert out["min_hand_gap_m"] == pytest.approx(0.50, abs=1e-6)


def test_the_two_arms_interleave_step_by_step(script):
    arms = Arms()
    run(script, arms)
    kinds = [k for k, _ in arms.events()]
    order = [a for _, a in arms.events()]
    # open, hover, descend, close, lift alternate arms: 0, 1, 0, 1, ...
    assert order[:10] == [0, 1] * 5
    assert (
        kinds[:10]
        == ["open", "open", "go_to_pose", "go_to_pose"]
        + ["go_to_pose_cartesian"] * 2
        + [
            "close",
            "close",
        ]
        + ["go_to_pose_cartesian"] * 2
    )
    # Neither arm starts carrying before both have lifted.
    first_carry = next(
        i for i, e in enumerate(arms.log) if e[0] == "robot.go_to_pose" and e[2][0] < 0.35
    )
    lifts = [
        i
        for i, e in enumerate(arms.log)
        if e[0] == "robot.go_to_pose_cartesian" and e[2][2] > 0.1 and e[2][0] == 0.40
    ]
    assert len(lifts) == 2 and max(lifts) < first_carry


def test_objects_on_swapped_sides_swap_arms(script):
    arms = Arms()
    out = run(script, arms, a=B, b=A, pa=PLACE_B, pb=PLACE_A)
    assert out["route"] == "both_placed"
    assert (out["arm_a"], out["arm_b"]) == (1, 0)


def test_ik_preview_overrides_the_nearer_side(script):
    # Arm 0 cannot reach object a's side at all: a goes to arm 1, b to arm 0.
    arms = Arms(reach=lambda a, t: not (a == 0 and t[1] > 0.2))
    a_far = _cube(0.40, 0.25)
    b_mid = _cube(0.40, 0.05)
    out = run(script, arms, a=a_far, b=b_mid, pa=_v(0.40, 0.30, 0.015), pb=_v(0.40, -0.10, 0.015))
    assert (out["arm_a"], out["arm_b"]) == (1, 0), out["detail"]
    assert out["route"] == "both_placed", out["detail"]


def test_unreachable_names_the_arm_and_moves_nothing(script):
    arms = Arms(reach=lambda a, t: t[0] < 0.6)
    far = _cube(0.70, -0.25)
    out = run(script, arms, b=far, pb=_v(0.70, -0.20, 0.015))
    assert out["route"] == "unreachable"
    assert out["failed_arms"] == "1"
    assert out["outcome_b"] == "unreachable" and out["outcome_a"] == ""
    assert "object b with arm 1" in out["detail"] and "nothing moved" in out["detail"]
    assert not [e for e in arms.log if e[0].startswith("robot.")]


def test_ik_preview_that_raises_counts_as_unreachable(script):
    arms = Arms(reach=lambda a, t: "raise" if t[0] > 0.6 else True)
    out = run(script, arms, b=_cube(0.70, -0.25), pb=_v(0.70, -0.20, 0.015))
    assert out["route"] == "unreachable"


def test_a_place_point_out_of_reach_is_unreachable(script):
    arms = Arms(reach=lambda a, t: t[0] > 0.0)
    out = run(script, arms, pa=_v(-0.10, 0.25, 0.015))
    assert out["route"] == "unreachable"
    assert "pre_place pose is out of reach" in out["detail"]


def test_same_arm_ids_refused(script):
    arms = Arms()
    out = run(script, arms, arm_ids=[1, 1])
    assert out["route"] == "unreachable"
    assert not arms.log


def test_connector_without_a_second_arm_is_refused(script):
    arms = Arms(second_arm=False)
    out = run(script, arms)
    assert out["route"] == "unreachable"
    assert "separate arm 1" in out["detail"]
    assert out["failed_arms"] == "1"
    assert not arms.log


def test_one_grasp_missed_routes_one_placed(script):
    arms = Arms(close_on=lambda a: 0.01 if a == 1 else 0.33)
    out = run(script, arms)
    assert out["route"] == "one_placed"
    assert out["failed_arms"] == "1"
    assert out["outcome_b"] == "grasp_missed"
    right = arms.events(1)
    # it opened, backed up to its hover and went home
    assert right[-3:] == [("open", 1), ("go_to_pose_cartesian", 1), ("go_to_pose", 1)]
    np.testing.assert_allclose(arms.p[1], RIGHT_HOME)


def test_both_missed_routes_none_placed(script):
    arms = Arms(close_on=lambda a: 0.0)
    out = run(script, arms)
    assert out["route"] == "none_placed"
    assert out["failed_arms"] == "0,1"


def test_drop_during_the_carry(script):
    def after(arms, tool, arm_id):
        # arm 0 loses the object when it reaches its pre-place point
        if (
            arm_id == 0
            and tool == "robot.go_to_pose"
            and arms.p[0][0] < 0.35
            and arms.p[0][2] > 0.09
        ):
            arms.grip[0] = 0.0

    arms = Arms()
    arms.after_move = after
    out = run(script, arms)
    assert out["route"] == "one_placed"
    assert out["outcome_a"] == "dropped"
    assert "after the lowering" in out["detail"]


def test_refused_hover_routed_not_raised(script):
    arms = Arms(
        refuse=lambda tool, a, t: "raise" if a == 0 and t[2] > 0.11 and t[0] == 0.40 else None
    )
    out = run(script, arms)
    assert out["route"] == "one_placed"
    assert out["outcome_a"] == "move_refused"
    assert "hover" in out["detail"]


def test_refused_carry_keeps_the_object(script):
    def refuse(tool, a, t):
        if a == 1 and tool == "robot.go_to_pose" and abs(t[0] - 0.32) < 1e-9 and t[2] > 0.09:
            return "out of reach"
        return None

    arms = Arms(refuse=refuse)
    out = run(script, arms)
    assert out["route"] == "one_placed"
    assert out["outcome_b"] == "move_refused"
    assert "still holds the object" in out["detail"]
    # it went back over where it picked, still closed
    np.testing.assert_allclose(arms.p[1], [0.40, -0.25, 0.115])
    assert arms.grip[1] == pytest.approx(0.33)


def test_close_place_points_stagger_the_arms(script):
    # Both into one box: the place points are 50 mm apart.
    arms = Arms()
    out = run(script, arms, pa=_v(0.30, 0.025, 0.015), pb=_v(0.30, -0.025, 0.015))
    assert out["route"] == "both_placed", out["detail"]
    assert out["staggered"]
    # Arm 1 starts its carry only after arm 0 is home.
    home0 = max(i for i, e in enumerate(arms.log) if e[0] == "robot.go_to_pose" and e[1] == 0)
    carry1 = next(
        i
        for i, e in enumerate(arms.log)
        if e[0] == "robot.go_to_pose" and e[1] == 1 and abs(e[2][0] - 0.30) < 1e-9
    )
    assert carry1 > home0
    assert min(arms.gaps) >= 0.15 - 1e-9


def test_close_objects_are_picked_one_after_the_other(script):
    arms = Arms()
    a, b = _cube(0.40, 0.05), _cube(0.40, -0.05)
    out = run(script, arms, a=a, b=b, pa=_v(0.40, 0.25, 0.015), pb=_v(0.40, -0.25, 0.015))
    assert out["route"] == "both_placed", out["detail"]
    assert out["staggered"]
    assert min(arms.gaps) >= 0.15 - 1e-9


def test_hands_that_can_never_clear_route_clearance(script):
    # Arm 1's carry is refused, so it holds its object over where it picked it;
    # arm 0's place point is right beside that, and arm 0 never gets clear to go.
    def refuse(tool, a, t):
        if a == 1 and tool == "robot.go_to_pose" and t[0] < 0.35:
            return "out of reach"
        return None

    arms = Arms(refuse=refuse)
    out = run(script, arms, pa=_v(0.40, -0.18, 0.015), pb=_v(0.30, -0.25, 0.015))
    assert out["route"] == "none_placed", out["detail"]
    assert (out["outcome_a"], out["outcome_b"]) == ("clearance", "move_refused")
    assert "would come within 150 mm of arm 1" in out["detail"]
    assert min(arms.gaps) >= 0.15 - 1e-9
    assert arms.grip[0] == pytest.approx(0.33)  # still holds its object


def test_rectangular_object_closed_across_its_narrow_side(script):
    bar = {
        "center": {"x": 0.40, "y": 0.25, "z": 0.02},
        "extent": {"x": 0.05, "y": 0.015, "z": 0.02},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    arms = Arms()
    ctx = arms.ctx()
    out = script.run(ctx, object_a_obb=bar, object_b_obb=B, place_a=PLACE_A, place_b=PLACE_B)
    assert out["route"] == "both_placed"
    heading = ctx.calls_to("robot.grasp_frame")[0].kwargs["close_heading_deg"]
    assert heading == pytest.approx(90.0)  # across y, the 30 mm side


def test_square_object_tries_both_sides(script):
    # The first heading's poses are out of reach for arm 0: it closes across the other side.
    arms = Arms()
    ctx = arms.ctx()
    seen = []
    frames = {
        0.0: {"rotation": ROT_A, "rotation_flipped": ROT_A},
        90.0: {"rotation": ROT_B, "rotation_flipped": ROT_B},
    }

    def grasp_frame(close_heading_deg=0.0, arm_id=0, **kw):
        seen.append((arm_id, round(close_heading_deg)))
        return frames[round(close_heading_deg) % 180 * 1.0]

    def solve_ik(pose, arm_id=0, **kw):
        ok = not (arm_id == 0 and pose["rotation"] == ROT_A)
        return {"success": ok, "within_servo_tolerance": ok}

    ctx._responses["robot.grasp_frame"] = grasp_frame
    ctx._responses["robot.solve_ik"] = solve_ik
    out = script.run(ctx, object_a_obb=A, object_b_obb=B, place_a=PLACE_A, place_b=PLACE_B)
    assert out["route"] == "both_placed", out["detail"]
    assert (0, 0) in seen and (0, 90) in seen
    hover = next(c for c in ctx.calls_to("robot.go_to_pose") if c.kwargs.get("arm_id", 0) == 0)
    assert hover.kwargs["pose"]["rotation"] == ROT_B


def test_grasp_floored_over_a_flat_object(script):
    flat = {
        "center": {"x": 0.40, "y": 0.25, "z": 0.005},
        "extent": {"x": 0.015, "y": 0.015, "z": 0.005},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    arms = Arms()
    run(script, arms, a=flat)
    # fingertips 10 mm under the tool centre, kept 4 mm over the bottom
    assert arms.targets(0)[1][2] == pytest.approx(0.014)


def test_wrist_roll_left_free_is_turned_on_the_last_joint(script):
    arms = Arms()
    ctx = arms.ctx()
    h = math.radians(20.0) / 2.0
    rolled = {
        "w": 0.0,
        "x": math.cos(h),
        "y": math.sin(h),
        "z": 0.0,
    }  # top-down, 20 deg off in roll
    go = ctx._responses["robot.go_to_pose"]
    turned = []

    def go_to_pose(pose, arm_id=0, **kw):
        reply = go(pose, arm_id=arm_id, **kw)
        if pose["position"]["z"] == pytest.approx(0.115) and arm_id not in turned:
            arms.rot[arm_id] = rolled  # the solver kept the approach, not the roll
        return reply

    def move_to_joints(joint_config, arm_id=0, **kw):
        turned.append(arm_id)
        arms.rot[arm_id] = ROT_A
        return None

    ctx._responses["robot.go_to_pose"] = go_to_pose
    ctx._responses["robot.move_to_joints"] = move_to_joints
    out = script.run(ctx, object_a_obb=A, object_b_obb=B, place_a=PLACE_A, place_b=PLACE_B)
    assert out["route"] == "both_placed", out["detail"]
    assert sorted(turned) == [0, 1]
    turn = ctx.calls_to("robot.move_to_joints")[0].kwargs["joint_config"]["positions"]
    assert abs(turn[-1]) == pytest.approx(math.radians(20.0), abs=1e-6)


def test_no_joint_reading_carries_on_and_says_so(script):
    arms = Arms()
    ctx = arms.ctx()
    h = math.radians(20.0) / 2.0
    rolled = {"w": 0.0, "x": math.cos(h), "y": math.sin(h), "z": 0.0}
    go = ctx._responses["robot.go_to_pose"]

    def go_to_pose(pose, arm_id=0, **kw):
        reply = go(pose, arm_id=arm_id, **kw)
        if pose["position"]["z"] == pytest.approx(0.115):
            arms.rot[arm_id] = rolled
        return reply

    def no_obs():
        raise ToolError("robot.get_observation", "not served")

    ctx._responses["robot.go_to_pose"] = go_to_pose
    ctx._responses["robot.get_observation"] = no_obs
    out = script.run(ctx, object_a_obb=A, object_b_obb=B, place_a=PLACE_A, place_b=PLACE_B)
    assert out["route"] == "both_placed"
    assert "the wrist was not turned" in out["detail"]


def test_example_declares_every_route(script):
    example = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == {"both_placed", "one_placed", "none_placed", "unreachable"}
