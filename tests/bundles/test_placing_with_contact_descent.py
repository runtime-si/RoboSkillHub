"""placing-with-contact-descent: FakeContext tests of the descent geometry, the contact verdicts and the routing."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "placing-with-contact-descent"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
START = (0.40, 0.10, 0.20)
# A 30 mm cube held with its centre 2 mm along +x and 7 mm under the tool centre:
# its bottom hangs 22 mm under the hand.
HELD = {
    "center": {"x": 0.402, "y": 0.10, "z": 0.193},
    "extent": {"x": 0.015, "y": 0.015, "z": 0.015},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}
HANG = 0.022
PLACE = {"x": 0.50, "y": 0.0, "z": 0.0}
# The hand's targets for PLACE: x shifted back by the cube's offset, contact at
# the hang, the window top 12 mm over it, the bottom floored 15 mm over the table.
HAND_XY = (0.498, 0.0)
EXPECTED = HANG
WINDOW_TOP = HANG + 0.012
FLOOR = 0.015
APPROACH = HANG + 0.08


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["place_with_contact_descent"].module


class Arm:
    """Moves land where commanded. A contact move stops where the held object's bottom
    meets ``surface`` (routed ``contact``, or raised like a real station with
    ``raise_contact``); with ``slide`` the object slides up the jaws and the move
    just finishes. ``refuse`` (tool, pose) -> reason refuses a move, routed;
    ``land_off`` (tool, target) -> where a move lands; ``outcomes`` routes another
    outcome for a move that landed."""

    def __init__(
        self,
        start=START,
        surface=0.0,
        hang=HANG,
        slide=False,
        raise_contact=False,
        refuse=None,
        land_off=None,
        timeouts=0,
        outcomes=None,
    ):
        self.outcomes = outcomes or {}
        self.p = np.array(start, dtype=float)
        self.rot = DOWN
        self.surface, self.hang = surface, hang
        self.slide, self.raise_contact = slide, raise_contact
        self.refuse = refuse or (lambda tool, pose: None)
        self.land_off = land_off or (lambda tool, target: target)
        self.timeouts = timeouts
        self.moves = []

    def _move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy(), pose["rotation"], kwargs))
            why = self.refuse(tool, pose)
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            self.rot = pose["rotation"]
            if tool == "robot.move_cartesian_until_contact":
                if self.timeouts:
                    self.timeouts -= 1
                    self.p = self.p + (target - self.p) / 2.0
                    return {"reached": False, "outcome": "timeout"}
                contact = self.surface + self.hang
                if not self.slide and target[2] < contact <= self.p[2]:
                    self.p = np.array([target[0], target[1], contact])
                    if self.raise_contact:
                        raise RuntimeError("the arm did not reach its target")
                    return {"reached": False, "outcome": "contact"}
            self.p = self.land_off(tool, target)
            outcome = self.outcomes.get(tool, "reached")
            return {"reached": outcome == "reached", "outcome": outcome}

        return go

    def ctx(self, ik=None):
        tools = {
            "robot.get_ee_pose": lambda **kw: {
                "pose": {
                    "position": dict(zip("xyz", map(float, self.p), strict=True)),
                    "rotation": self.rot,
                }
            },
            "robot.go_to_pose": self._move("robot.go_to_pose"),
            "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
            "robot.move_cartesian_until_contact": self._move("robot.move_cartesian_until_contact"),
            "robot.solve_ik": ik
            if ik is not None
            else {"success": True, "within_servo_tolerance": True},
            "robot.open_gripper": {"position": 1.0},
        }
        return FakeContext(tools)


def _xy(v):
    return np.asarray(v)[:2]


def test_places_on_contact_and_retreats(script):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(ctx, place_position=PLACE, held_obb=HELD, table_z=0.0)
    assert out["route"] == "placed", out["detail"]
    assert out["released"] is True
    names = [m[0] for m in arm.moves]
    assert names == [
        "robot.go_to_pose",  # approach (the hand starts above it)
        "robot.move_cartesian_until_contact",  # to the top of the contact window
        "robot.move_cartesian_until_contact",  # through it
        "robot.go_to_pose_cartesian",  # retreat
    ]
    approach, top, bottom, retreat = (m[1] for m in arm.moves)
    np.testing.assert_allclose(approach, [*HAND_XY, APPROACH], atol=1e-9)
    np.testing.assert_allclose(top, [*HAND_XY, WINDOW_TOP], atol=1e-9)
    # 20 mm under the expected contact would be 2 mm; the floor holds it at 15 mm.
    np.testing.assert_allclose(bottom, [*HAND_XY, FLOOR], atol=1e-9)
    np.testing.assert_allclose(retreat, [*HAND_XY, EXPECTED + 0.08], atol=1e-9)
    assert all(m[2] == DOWN for m in arm.moves)  # the hand keeps its rotation
    assert out["expected_contact_z"] == pytest.approx(EXPECTED)
    assert out["contact_offset"] == pytest.approx(0.0, abs=1e-9)
    placed = out["placed_position"]
    assert [placed[k] for k in "xyz"] == pytest.approx([0.50, 0.0, 0.0], abs=1e-9)
    calls = [c.tool for c in ctx.calls]
    assert calls[0] == "robot.get_ee_pose"
    assert calls.index("robot.solve_ik") < calls.index("robot.go_to_pose")
    assert calls.index("robot.open_gripper") < max(
        i for i, c in enumerate(calls) if c == "robot.go_to_pose_cartesian"
    )
    assert "arm_id" not in ctx.calls_to("robot.go_to_pose")[0].kwargs


def test_a_real_station_raising_on_contact_is_a_stop(script):
    arm = Arm(raise_contact=True)
    out = script.run(arm.ctx(), place_position=PLACE, held_obb=HELD, table_z=0.0)
    assert out["route"] == "placed", out["detail"]
    assert out["contact_z"] == pytest.approx(EXPECTED)


def test_rises_straight_up_first_when_under_the_approach(script):
    arm = Arm(start=(0.40, 0.10, 0.05))
    held = {**HELD, "center": {**HELD["center"], "z": 0.05 - 0.007}}
    out = script.run(arm.ctx(), place_position=PLACE, held_obb=held, table_z=0.0)
    assert out["route"] == "placed"
    tool, rise = arm.moves[0][:2]
    assert tool == "robot.go_to_pose_cartesian"
    np.testing.assert_allclose(rise, [0.40, 0.10, APPROACH], atol=1e-9)
    assert arm.moves[1][0] == "robot.go_to_pose"


def test_contact_above_the_window_is_too_early_and_not_released(script):
    # Something 30 mm tall under the place point: the cube meets it 30 mm early.
    arm = Arm(surface=0.03)
    ctx = arm.ctx()
    out = script.run(ctx, place_position=PLACE, held_obb=HELD, table_z=0.0)
    assert out["route"] == "contact_too_early", out["detail"]
    assert out["released"] is False
    assert ctx.call_count("robot.open_gripper") == 0
    assert out["contact_offset"] == pytest.approx(0.03)
    tool, lift = arm.moves[-1][:2]
    assert tool == "robot.go_to_pose_cartesian"
    np.testing.assert_allclose(lift, [*HAND_XY, APPROACH], atol=1e-9)
    assert "still holding" in out["detail"]


def test_a_support_lower_than_stated_is_no_contact(script):
    # Stated 30 mm higher than the table it really is: the bottom (20 mm under the
    # expected contact) still leaves the cube 10 mm over the table.
    arm = Arm(surface=0.0)
    ctx = arm.ctx()
    out = script.run(ctx, place_position={**PLACE, "z": 0.03}, held_obb=HELD, table_z=0.0)
    assert out["route"] == "no_contact", out["detail"]
    assert out["released"] is False
    assert ctx.call_count("robot.open_gripper") == 0
    assert out["contact_z"] == pytest.approx(0.03 + HANG - 0.02)
    np.testing.assert_allclose(arm.moves[-1][1], [*HAND_XY, 0.03 + APPROACH], atol=1e-9)


def test_a_slide_to_the_floor_over_the_table_is_placed(script):
    # The cube slides up the jaws instead of stalling the arm: the hand finishes at
    # the floor, where the cube's bottom would be 7 mm under the table.
    arm = Arm(slide=True)
    out = script.run(arm.ctx(), place_position=PLACE, held_obb=HELD, table_z=0.0)
    assert out["route"] == "placed", out["detail"]
    assert "at or under the table" in out["detail"]
    assert out["contact_z"] == pytest.approx(FLOOR)


def test_the_same_slide_without_the_table_height_is_no_contact(script):
    arm = Arm(slide=True)
    ctx = arm.ctx()
    out = script.run(ctx, place_position=PLACE, held_obb=HELD)
    assert out["route"] == "no_contact", out["detail"]
    assert ctx.call_count("robot.open_gripper") == 0


def test_a_lower_support_inside_the_window_is_placed_low(script):
    arm = Arm(surface=-0.004)
    out = script.run(arm.ctx(), place_position=PLACE, held_obb=HELD, table_z=-0.01)
    assert out["route"] == "placed", out["detail"]
    assert out["contact_offset"] == pytest.approx(-0.004)


def test_the_ik_preview_refusing_blocks_before_moving(script):
    arm = Arm()
    out = script.run(
        arm.ctx(ik={"success": True, "within_servo_tolerance": False}),
        place_position=PLACE,
        held_obb=HELD,
    )
    assert out["route"] == "blocked"
    assert "IK" in out["detail"]
    assert arm.moves == []


def test_no_ik_preview_gets_the_benefit_of_the_doubt(script):
    def missing(**kw):
        raise ToolError("robot.solve_ik", "not served")

    arm = Arm()
    out = script.run(arm.ctx(ik=missing), place_position=PLACE, held_obb=HELD, table_z=0.0)
    assert out["route"] == "placed"


def test_check_reach_false_skips_the_preview(script):
    arm = Arm()
    ctx = arm.ctx()
    script.run(ctx, place_position=PLACE, held_obb=HELD, table_z=0.0, check_reach=False)
    assert ctx.call_count("robot.solve_ik") == 0


@pytest.mark.parametrize("raised", [False, True])
def test_a_refused_approach_is_blocked(script, raised):
    def refuse(tool, pose):
        if tool == "robot.go_to_pose":
            if raised:
                raise RuntimeError("IK failed")
            return "out of reach"
        return None

    arm = Arm(refuse=refuse)
    ctx = arm.ctx()
    out = script.run(ctx, place_position=PLACE, held_obb=HELD)
    assert out["route"] == "blocked"
    assert out["detail"].startswith("approach:")
    assert ctx.call_count("robot.move_cartesian_until_contact") == 0
    assert out["released"] is False


def test_an_approach_that_ends_off_is_blocked(script):
    def off(tool, target):
        return target + np.array([0.03, 0.0, 0.0]) if tool == "robot.go_to_pose" else target

    arm = Arm(land_off=off)
    out = script.run(arm.ctx(), place_position=PLACE, held_obb=HELD)
    assert out["route"] == "blocked"
    assert "30 mm" in out["detail"]


@pytest.mark.parametrize(("miss", "route"), [(0.004, "placed"), (0.03, "blocked")])
def test_an_approach_that_times_out_is_judged_by_where_the_hand_is(script, miss, route):
    def short(tool, target):
        return target + np.array([miss, 0.0, 0.0]) if tool == "robot.go_to_pose" else target

    arm = Arm(land_off=short, outcomes={"robot.go_to_pose": "timeout"})
    out = script.run(arm.ctx(), place_position=PLACE, held_obb=HELD, table_z=0.0)
    assert out["route"] == route, out["detail"]


def test_a_refused_descent_is_blocked_not_read_as_contact(script):
    def refuse(tool, pose):
        if (
            tool == "robot.move_cartesian_until_contact"
            and pose["position"]["z"] < WINDOW_TOP - 1e-6
        ):
            return "IK failed"
        return None

    arm = Arm(refuse=refuse)
    ctx = arm.ctx()
    out = script.run(ctx, place_position=PLACE, held_obb=HELD, table_z=0.0)
    assert out["route"] == "blocked", out["detail"]
    assert ctx.call_count("robot.open_gripper") == 0


def test_a_failed_retreat_after_release_is_blocked_but_released(script):
    def refuse(tool, pose):
        if tool == "robot.go_to_pose_cartesian":
            return "joint limit"
        return None

    arm = Arm(refuse=refuse)
    out = script.run(arm.ctx(), place_position=PLACE, held_obb=HELD, table_z=0.0)
    assert out["route"] == "blocked"
    assert out["released"] is True
    assert out["detail"].startswith("released, but the retreat")


def test_timeouts_still_descending_are_resent(script):
    arm = Arm(timeouts=2)
    out = script.run(arm.ctx(), place_position=PLACE, held_obb=HELD, table_z=0.0)
    assert out["route"] == "placed", out["detail"]
    contact_moves = [m for m in arm.moves if m[0] == "robot.move_cartesian_until_contact"]
    assert len(contact_moves) == 4  # two timeouts, then the window top, then through


def test_without_a_held_box_the_place_point_is_the_tool_centre(script):
    arm = Arm(hang=0.0, surface=0.10)
    out = script.run(arm.ctx(), place_position={"x": 0.45, "y": 0.05, "z": 0.10})
    assert out["route"] == "placed", out["detail"]
    approach, top, bottom = (m[1] for m in arm.moves[:3])
    np.testing.assert_allclose(approach, [0.45, 0.05, 0.18], atol=1e-9)
    np.testing.assert_allclose(top, [0.45, 0.05, 0.112], atol=1e-9)
    np.testing.assert_allclose(bottom, [0.45, 0.05, 0.08], atol=1e-9)  # no floor without a support


def test_arm_id_is_passed_only_when_set(script):
    arm = Arm()
    ctx = arm.ctx()
    script.run(ctx, place_position=PLACE, held_obb=HELD, table_z=0.0, arm_id=1)
    robot_calls = [c for c in ctx.calls if c.tool.startswith("robot.")]
    assert all(c.kwargs.get("arm_id") == 1 for c in robot_calls)


def test_the_bottom_of_a_tilted_box(script):
    q = {"w": math.cos(math.pi / 8), "x": math.sin(math.pi / 8), "y": 0.0, "z": 0.0}
    tilted = {**HELD, "orientation": q}
    # Turned 45 deg about x, the lowest edge sits 0.015 * sqrt(2) under the centre.
    assert script._bottom_z(tilted) == pytest.approx(0.193 - 0.015 * math.sqrt(2))
