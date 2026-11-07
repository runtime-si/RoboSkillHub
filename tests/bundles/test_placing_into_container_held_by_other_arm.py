"""placing-into-container-held-by-other-arm: FakeContext tests of the two-arm sequence, the geometry and the routing."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "placing-into-container-held-by-other-arm"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
IDENT = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
# A 70 x 100 x 40 mm tray held by arm 0 on its +y wall (tool centre 3 mm in from
# the outer face); a 30 mm cube held top-down by arm 1, its tool centre 8 mm
# above the cube's centre.
CUP = {
    "center": {"x": 0.40, "y": 0.20, "z": 0.10},
    "extent": {"x": 0.035, "y": 0.05, "z": 0.02},
    "orientation": IDENT,
}
CUBE = {
    "center": {"x": 0.40, "y": -0.20, "z": 0.15},
    "extent": {"x": 0.015, "y": 0.015, "z": 0.015},
    "orientation": IDENT,
}
PRESENT = {"x": 0.45, "y": 0.0, "z": 0.15}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["place_in_held_container"].module


def _about(axis, deg):
    h = math.radians(deg) / 2.0
    a = np.asarray(axis, float) / float(np.linalg.norm(axis))
    return {
        "w": math.cos(h),
        "x": a[0] * math.sin(h),
        "y": a[1] * math.sin(h),
        "z": a[2] * math.sin(h),
    }


def _mul(q1, q2):
    w1, x1, y1, z1 = (q1[k] for k in "wxyz")
    w2, x2, y2, z2 = (q2[k] for k in "wxyz")
    return {
        "w": w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        "x": w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        "y": w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        "z": w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    }


class Arms:
    """Two arms; moves land where commanded unless ``refuse(tool, arm, pose)`` says why not."""

    def __init__(self, refuse=None, ik=None, stuck=False):
        self.p = {0: np.array([0.40, 0.247, 0.112]), 1: np.array([0.40, -0.20, 0.158])}
        self.rot = {0: DOWN, 1: DOWN}
        self.grip = {0: 0.064, 1: 0.33}
        self.holding = True
        self.stuck = stuck
        self.refuse = refuse or (lambda tool, a, pose: None)
        self.ik = ik or (lambda a, pose: {"success": True, "within_servo_tolerance": True})
        self.after_move = None
        self.opens = True
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
            self.rot[arm_id] = pose["rotation"]
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
            if arm_id == 1 and not self.opens:
                return {"position": self.grip[1]}
            self.grip[arm_id] = 1.0
            if arm_id == 1 and not self.stuck:
                self.holding = False
            return {"position": 1.0}

        def close_gripper(arm_id=0, **kw):
            self.log.append(("close", arm_id))
            self.grip[arm_id] = 0.33 if (arm_id == 1 and self.holding) else 0.0
            return {"position": self.grip[arm_id]}

        def solve_ik(pose, arm_id=0, **kw):
            self.log.append(("ik", arm_id, pose["rotation"]))
            return self.ik(arm_id, pose)

        return FakeContext(
            {
                "robot.get_ee_pose": pose,
                "robot.get_gripper": lambda arm_id=0: {"position": self.grip[arm_id]},
                "robot.go_to_pose": self._move("robot.go_to_pose"),
                "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
                "robot.open_gripper": open_gripper,
                "robot.close_gripper": close_gripper,
                "robot.solve_ik": solve_ik,
            }
        )

    def moves(self, arm_id=None):
        return [
            m for m in self.log if m[0].startswith("robot.") and (arm_id is None or m[1] == arm_id)
        ]


def run(script, arms, **kw):
    kw.setdefault("presentation_position", PRESENT)
    return script.run(arms.ctx(), container_obb=CUP, object_obb=CUBE, **kw)


def test_happy_path_geometry_and_order(script):
    arms = Arms()
    ctx = arms.ctx()
    out = script.run(
        ctx, container_obb=CUP, object_obb=CUBE, presentation_position=PRESENT, rotations=["own"]
    )
    assert out["route"] == "placed", out["detail"]
    holder = arms.moves(0)
    # The holder carries the cup's centre onto the presentation point, straight, keeping its rotation.
    assert len(holder) == 1 and holder[0][0] == "robot.go_to_pose_cartesian"
    np.testing.assert_allclose(holder[0][2], [0.45, 0.047, 0.162], atol=1e-9)
    assert holder[0][3] == DOWN
    # Drop point: away from the holder's +y wall, as far as the cube fits
    # (50 - 8 - 15 = 27 mm); the cube's bottom 60 then 15 mm over the 0.17 rim.
    carrier = arms.moves(1)
    np.testing.assert_allclose(carrier[0][2], [0.45, -0.027, 0.253], atol=1e-9)
    assert carrier[0][0] == "robot.go_to_pose"
    np.testing.assert_allclose(carrier[1][2], [0.45, -0.027, 0.208], atol=1e-9)
    assert carrier[1][0] == "robot.go_to_pose_cartesian"
    np.testing.assert_allclose(carrier[2][2], [0.45, -0.027, 0.308], atol=1e-9)
    order = [m[:2] for m in arms.log if m[0] in ("open", "close")]
    assert order == [("open", 1), ("close", 1), ("open", 1)]
    assert out["chosen_rotation"] == "own"
    # The cube's near edge 74 - 15 mm from the holder's tool centre.
    assert out["hand_gap"] == pytest.approx(0.059)
    assert out["container_tilt_deg"] == pytest.approx(0.0, abs=1e-6)
    assert out["release_position"] == {
        "x": pytest.approx(0.45),
        "y": pytest.approx(-0.027),
        "z": pytest.approx(0.20),
    }
    assert out["container_position"]["y"] == pytest.approx(0.0)
    # arm_id only for the nonzero arm.
    for call in ctx.calls:
        if call.tool.startswith("robot.") and "arm_id" in call.kwargs:
            assert call.kwargs["arm_id"] == 1


def test_default_presentation_is_between_the_hands(script):
    arms = Arms()
    out = script.run(arms.ctx(), container_obb=CUP, object_obb=CUBE)
    assert out["route"] == "placed", out["detail"]
    # Midpoint of the tool centres (0.40, 0.0235), at the tray's own height.
    np.testing.assert_allclose(arms.moves(0)[0][2], [0.40, 0.0705, 0.112], atol=1e-9)


def test_refuses_one_arm_twice_or_a_missing_second_arm(script):
    out = run(script, Arms(), object_arm=0)
    assert out["route"] == "unreachable" and "both on arm 0" in out["detail"]

    arms = Arms()
    arms.p[1] = arms.p[0].copy()
    out = run(script, arms)
    assert out["route"] == "unreachable" and "separate arm" in out["detail"]
    assert not arms.moves()


def test_empty_hands(script):
    arms = Arms()
    arms.grip[0] = 0.01
    out = run(script, arms)
    assert out["route"] == "unreachable" and "no container" in out["detail"]
    assert not arms.moves()

    arms = Arms()
    arms.grip[1] = 0.0
    out = run(script, arms)
    assert out["route"] == "missed" and "no object" in out["detail"]
    assert not arms.moves()


def test_a_leaning_container_is_refused(script):
    arms = Arms()
    cup = {**CUP, "orientation": _about((1, 0, 0), 20.0)}
    out = script.run(arms.ctx(), container_obb=cup, object_obb=CUBE, presentation_position=PRESENT)
    assert out["route"] == "unreachable" and "leans 20 deg" in out["detail"]
    assert not arms.moves()
    # A box lying on its side still has a vertical axis: its opening is that face.
    side = {
        **CUP,
        "extent": {"x": 0.03, "y": 0.03, "z": 0.03},
        "orientation": _about((1, 0, 0), 90.0),
    }
    arms = Arms()
    out = script.run(arms.ctx(), container_obb=side, object_obb=CUBE, presentation_position=PRESENT)
    assert out["route"] == "placed", out["detail"]


@pytest.mark.parametrize("how", ["raise", "unreachable"])
def test_holder_cannot_present(script, how):
    arms = Arms(refuse=lambda tool, a, pose: how if a == 0 else None)
    out = run(script, arms)
    assert out["route"] == "unreachable" and "presentation" in out["detail"]
    assert not arms.moves(1)
    assert [m[0] for m in arms.moves(0)] == ["robot.go_to_pose_cartesian", "robot.go_to_pose"]


def test_holder_falls_back_to_a_free_move(script):
    arms = Arms(
        refuse=lambda tool, a, pose: "raise" if a == 0 and tool.endswith("cartesian") else None
    )
    out = run(script, arms)
    assert out["route"] == "placed"
    assert "straight-line presentation refused" in out["detail"]


def test_a_tilt_is_taken_when_the_own_rotation_is_out_of_reach(script):
    arms = Arms(ik=lambda a, pose: {"success": pose["rotation"] != DOWN})
    out = run(script, arms)
    assert out["route"] == "placed", out["detail"]
    assert out["chosen_rotation"] == "tilt_30_+x"
    approach = arms.moves(1)[0]
    rot = approach[3]
    expect = _mul(_about((1, 0, 0), 30.0), DOWN)
    assert all(
        rot[k] == pytest.approx(expect[k]) or rot[k] == pytest.approx(-expect[k]) for k in "wxyz"
    )
    # The tool centre rides the 8 mm hold offset turned 30 deg about x; the
    # cube's bottom reach grows to 15 (cos 30 + sin 30) mm.
    bottom = 0.015 * (math.cos(math.radians(30)) + math.sin(math.radians(30)))
    offset = np.array(
        [0.0, -0.008 * math.sin(math.radians(30)), 0.008 * math.cos(math.radians(30))]
    )
    release = np.array([0.45, None, 0.17 + 0.015 + bottom], dtype=object)
    np.testing.assert_allclose(arms.moves(1)[1][2][2], release[2] + offset[2], atol=1e-9)
    assert out["release_position"]["z"] == pytest.approx(0.17 + 0.015 + bottom)


def test_no_rotation_reaches(script):
    arms = Arms(refuse=lambda tool, a, pose: "raise" if a == 1 else None)
    out = run(script, arms)
    assert out["route"] == "unreachable" and "no carrying-hand rotation" in out["detail"]
    for label in (
        "own",
        "tilt_30_+x",
        "tilt_30_-x",
        "tilt_30_+y",
        "tilt_30_-y",
        "tilt_45_+x",
        "tilt_45_-x",
    ):
        assert f"{label}: " in out["detail"]
    assert ("open", 1) not in arms.log
    assert len(arms.moves(1)) == 7


def test_holder_hand_over_the_opening_is_refused(script):
    arms = Arms()
    arms.p[0] = np.array([0.40, 0.20, 0.112])  # gripping across the tray's middle
    out = run(script, arms)
    assert out["route"] == "unreachable" and "from the holder's hand" in out["detail"]
    assert not arms.moves(1)

    # A 60 mm cup held by its wall: the holder's 30 mm blade fills the side the cube would need.
    cup = {
        **CUP,
        "center": {"x": 0.40, "y": 0.22, "z": 0.10},
        "extent": {"x": 0.03, "y": 0.03, "z": 0.02},
    }
    arms = Arms()
    out = script.run(arms.ctx(), container_obb=cup, object_obb=CUBE, presentation_position=PRESENT)
    assert out["route"] == "unreachable" and "19 mm from the holder's hand" in out["detail"]
    assert not arms.moves(1)


def test_an_object_too_big_for_the_opening(script):
    big = {**CUBE, "extent": {"x": 0.03, "y": 0.03, "z": 0.03}}
    arms = Arms()
    out = script.run(arms.ctx(), container_obb=CUP, object_obb=big, presentation_position=PRESENT)
    assert out["route"] == "unreachable" and "does not fit" in out["detail"]
    assert not arms.moves(1)


def test_descent_stopped_backs_up_without_releasing(script):
    arms = Arms(
        refuse=lambda tool, a, pose: (
            "contact"
            if a == 1 and tool.endswith("cartesian") and pose["position"]["z"] < 0.21
            else None
        )
    )
    out = run(script, arms, rotations=["own"])
    assert out["route"] == "unreachable" and "down to the release" in out["detail"]
    assert ("open", 1) not in arms.log
    np.testing.assert_allclose(arms.p[1], [0.45, -0.027, 0.253], atol=1e-9)


def test_jaws_that_do_not_open(script):
    arms = Arms()
    arms.opens = False
    out = run(script, arms)
    assert out["route"] == "missed" and "after opening" in out["detail"]


def test_object_stuck_in_the_jaws(script):
    arms = Arms(stuck=True)
    out = run(script, arms)
    assert out["route"] == "missed" and "still in them" in out["detail"]
    assert arms.log[-1][:2] == ("open", 1)


def test_container_dropped(script):
    arms = Arms()

    def drop(a, tool, arm_id):
        if arm_id == 1 and a.p[1][2] > 0.30:
            a.grip[0] = 0.0

    arms.after_move = drop
    out = run(script, arms)
    assert out["route"] == "missed" and "container fell" in out["detail"]


def test_holder_pushed_away_misses_the_opening(script):
    arms = Arms()

    def push(a, tool, arm_id):
        if arm_id == 1 and tool.endswith("cartesian") and a.p[1][2] < 0.21:
            a.p[0] = a.p[0] + np.array([0.0, 0.03, 0.0])

    arms.after_move = push
    out = run(script, arms, rotations=["own"])
    assert out["route"] == "missed" and "outside the container's opening" in out["detail"]


def test_free_wrist_rolls_are_read_back(script):
    # Both wrists come back turned 90 deg about the vertical: the cup turns with
    # its holder, and the carrier is re-aimed so the cube still lands on the drop point.
    turn = _about((0, 0, 1), 90.0)
    arms = Arms()
    arms.p[1] = np.array([0.405, -0.20, 0.158])  # held 5 mm off the cube's centre in x

    def roll(a, tool, arm_id):
        if tool == "robot.go_to_pose" or (arm_id == 0 and a.rot[0] == DOWN):
            a.rot[arm_id] = _mul(turn, a.rot[arm_id])

    arms.after_move = roll
    out = run(script, arms, rotations=["own"])
    assert out["route"] == "placed", out["detail"]
    # Tray centre: the holder's tool centre plus its (0, -0.047) offset turned 90 deg.
    assert out["container_position"]["x"] == pytest.approx(0.45 + 0.047)
    assert out["container_position"]["y"] == pytest.approx(0.047)
    # The carrier's descent ends at the drop point plus the 5 mm offset turned 90 deg.
    release = arms.moves(1)[1][2]
    drop = np.array([out["release_position"][k] for k in "xy"])
    np.testing.assert_allclose(release[:2] - drop, [0.0, 0.005], atol=1e-9)


def test_quat_round_trip(script):
    for q in (DOWN, IDENT, _about((0.3, -0.5, 0.8), 137.0), _about((0, 1, 0), 180.0)):
        r = script._rotation(q)
        np.testing.assert_allclose(script._rotation(script._quat(r)), r, atol=1e-9)


def test_example_declares_every_route():
    example = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    source = (SKILL_DIR / "scripts" / "place_in_held_container.py").read_text()
    for route in ("placed", "missed", "unreachable"):
        assert f'"{route}"' in source
    assert declared == {"placed", "missed", "unreachable"}


def test_a_caller_can_order_the_rotations(script):
    arms = Arms()
    out = run(script, arms, rotations=["tilt_30_+x", "own"])
    assert out["route"] == "placed" and out["chosen_rotation"] == "tilt_30_+x"
    out = run(script, Arms(), rotations=["sideways"])
    assert out["route"] == "unreachable" and "unknown carrier rotations" in out["detail"]


def test_default_order_leans_the_carrier_away_from_the_holder(script):
    # The holder is on the tray's +y side: tilts about +x lean the carrier's wrist
    # toward -y, away from it, and go first (30 before 45), then RoboRSI's order.
    arms = Arms(refuse=lambda tool, a, pose: "raise" if a == 1 else None)
    out = run(script, arms)
    order = re.findall(r"(own|tilt_\d\d_[+-][xy]): ", out["detail"])
    assert order == [
        "tilt_30_+x",
        "tilt_45_+x",
        "own",
        "tilt_30_-x",
        "tilt_30_+y",
        "tilt_30_-y",
        "tilt_45_-x",
    ]
    arms = Arms()
    out = run(script, arms)
    assert out["route"] == "placed" and out["chosen_rotation"] == "tilt_30_+x"
