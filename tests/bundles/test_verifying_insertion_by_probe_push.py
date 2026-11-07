"""verifying-insertion-by-probe-push: FakeContext tests of the push, the verdicts and the routing."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "verifying-insertion-by-probe-push"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
START = np.array([0.40, 0.10, 0.20])


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["probe_push"].module


class Arm:
    """Moves land where commanded, except a contact move, which stops where the held
    part meets its seat: ``give`` [m] past the start along the push (the hand's give
    and the part's slide in the jaws), or nowhere when ``give`` is None (nothing in
    the way). ``side_give`` does the same for pushes square to the axis. With
    ``raise_contact`` a stop raises like a real station; ``refuse`` (tool) -> routed
    reply refuses that tool's moves; ``short`` [m] ends a free move that far short
    (a servo's arrival ball)."""

    def __init__(
        self,
        give=0.002,
        side_give=None,
        axis=(0.0, 0.0, -1.0),
        raise_contact=False,
        refuse=None,
        short=0.0,
        ik_far=False,
    ):
        self.p = START.copy()
        self.start = START.copy()
        self.give, self.side_give = give, side_give
        self.axis = np.asarray(axis, dtype=float)
        self.raise_contact = raise_contact
        self.refuse = refuse or {}
        self.short = short
        self.ik_far = ik_far
        self.refused_drift = 0.0
        self.pushes: list[np.ndarray] = []
        self.calls: list[tuple[str, dict]] = []

    def ee(self, **kw):
        self.calls.append(("robot.get_ee_pose", kw))
        return {
            "pose": {
                "position": dict(zip("xyz", map(float, self.p), strict=True)),
                "rotation": DOWN,
            }
        }

    def cart(self, pose, **kw):
        self.calls.append(("robot.go_to_pose_cartesian", kw))
        if "robot.go_to_pose_cartesian" in self.refuse:
            self.p = self.p + np.array([0.0, 0.0, self.refused_drift])
            return self.refuse["robot.go_to_pose_cartesian"]
        self.p = np.array([pose["position"][k] for k in "xyz"])
        return {"reached": True, "outcome": "reached"}

    def push(self, pose, **kw):
        self.calls.append(("robot.move_cartesian_until_contact", kw))
        if "robot.move_cartesian_until_contact" in self.refuse:
            reply = self.refuse["robot.move_cartesian_until_contact"]
            if isinstance(reply, Exception):
                raise reply
            return reply
        target = np.array([pose["position"][k] for k in "xyz"])
        step = target - self.start
        self.pushes.append(step)
        d = step / np.linalg.norm(step)
        along = abs(float(np.dot(d, self.axis))) > 0.5
        give = self.give if along else self.side_give
        if give is None or give >= np.linalg.norm(step):
            self.p = target - d * self.short
            return {"reached": True, "outcome": "reached"}
        self.p = self.start + d * give
        if self.raise_contact:
            raise ToolError(
                "robot.move_cartesian_until_contact", "step 1/1: the tool did not reach its target"
            )
        return {"reached": False, "outcome": "contact"}

    def ik(self, pose, **kw):
        self.calls.append(("robot.solve_ik", kw))
        return {"success": True, "within_servo_tolerance": not self.ik_far}

    def ctx(self, **overrides):
        tools = {
            "robot.get_ee_pose": self.ee,
            "robot.go_to_pose_cartesian": self.cart,
            "robot.move_cartesian_until_contact": self.push,
            "robot.solve_ik": self.ik,
        }
        tools.update(overrides)
        return FakeContext(tools)

    def called(self, tool):
        return [c for c in self.calls if c[0] == tool]


def test_a_part_that_does_not_go_further_is_seated_and_the_press_is_taken_off(script):
    arm = Arm(give=0.0035)
    out = script.run(arm.ctx())
    assert out["route"] == "seated"
    assert out["travel"] == pytest.approx(0.0035)
    assert out["stop_position"]["z"] == pytest.approx(START[2] - 0.0035)
    assert np.allclose(arm.pushes[0], [0.0, 0.0, -script.PROBE_DISTANCE])
    assert out["returned"] and np.allclose(arm.p, START)
    assert "3.5 mm" in out["detail"] and "not a measurement of the part" in out["detail"]


def test_a_part_that_goes_further_is_not_seated(script):
    arm = Arm(give=None)
    out = script.run(arm.ctx())
    assert out["route"] == "not_seated"
    assert out["travel"] == pytest.approx(script.PROBE_DISTANCE)
    assert np.allclose(arm.p, START)


def test_a_free_push_ending_in_the_servos_arrival_ball_is_still_not_seated(script):
    # The twin's servo ended free pushes about 2 mm short (7.3 mm of 10).
    out = script.run(Arm(give=None, short=0.0027).ctx())
    assert out["route"] == "not_seated" and out["travel"] == pytest.approx(0.0073)


def test_a_part_that_stops_after_part_of_its_gap_is_not_seated(script):
    out = script.run(Arm(give=0.006).ctx())
    assert out["route"] == "not_seated" and out["travel"] == pytest.approx(0.006)


def test_a_station_style_error_on_a_stall_is_read_as_a_stop(script):
    out = script.run(Arm(give=0.001, raise_contact=True).ctx())
    assert out["route"] == "seated" and out["travel"] == pytest.approx(0.001)


@pytest.mark.parametrize(
    "reply",
    [
        {"reached": False, "outcome": "unreachable", "reason": "no IK solution"},
        {"success": False, "message": "planner failed"},
        ToolError("robot.move_cartesian_until_contact", "operator refused this move"),
    ],
)
def test_a_refused_push_is_blocked_not_seated(script, reply):
    arm = Arm(refuse={"robot.move_cartesian_until_contact": reply})
    out = script.run(arm.ctx())
    assert out["route"] == "blocked"
    assert out["detail"].startswith("push:")
    assert not arm.called("robot.go_to_pose_cartesian")


def test_a_contact_outcome_is_not_a_refusal(script):
    assert script._refusal({"reached": False, "outcome": "contact"}) == ""
    assert script._refusal({"reached": False, "outcome": "timeout"}) == ""
    assert script._refusal({"reached": False, "outcome": "unreachable"}).startswith("unreachable")


def test_an_end_out_of_reach_is_blocked_before_pushing(script):
    arm = Arm(ik_far=True)
    out = script.run(arm.ctx())
    assert out["route"] == "blocked" and "IK preview" in out["detail"]
    assert not arm.pushes


def test_an_ik_preview_that_errors_does_not_block(script):
    arm = Arm(give=0.001)

    def broken(**kw):
        raise ToolError("robot.solve_ik", "not offered")

    out = script.run(arm.ctx(**{"robot.solve_ik": broken}))
    assert out["route"] == "seated"


def test_check_reach_false_skips_the_preview(script):
    arm = Arm(ik_far=True, give=0.001)
    out = script.run(arm.ctx(), check_reach=False)
    assert out["route"] == "seated" and not arm.called("robot.solve_ik")


def test_the_table_floor_shortens_the_push(script):
    arm = Arm(give=None)
    table_z = START[2] - 0.015 - 0.007  # 7 mm of room over the tool-centre floor
    out = script.run(arm.ctx(), table_z=table_z)
    assert np.allclose(arm.pushes[0], [0.0, 0.0, -0.007])
    assert out["route"] == "not_seated" and out["travel"] == pytest.approx(0.007)


def test_no_room_over_the_floor_is_blocked(script):
    arm = Arm()
    out = script.run(arm.ctx(), table_z=START[2] - 0.015 - 0.004)
    assert out["route"] == "blocked" and "floor" in out["detail"]
    assert not arm.pushes


def test_the_floor_does_not_limit_an_upward_or_sideways_push(script):
    arm = Arm(give=0.001, axis=(1.0, 0.0, 0.0))
    out = script.run(arm.ctx(), axis={"x": 1.0, "y": 0.0, "z": 0.0}, table_z=START[2] - 0.01)
    assert out["route"] == "seated"
    assert np.allclose(arm.pushes[0], [script.PROBE_DISTANCE, 0.0, 0.0])


def test_a_horizontal_axis_is_normalised(script):
    arm = Arm(give=None, axis=(0.0, 1.0, 0.0))
    out = script.run(arm.ctx(), axis={"x": 0.0, "y": 3.0, "z": 0.0})
    assert out["route"] == "not_seated"
    assert np.allclose(arm.pushes[0], [0.0, script.PROBE_DISTANCE, 0.0])


def test_a_zero_axis_is_blocked(script):
    out = script.run(Arm().ctx(), axis={"x": 0.0, "y": 0.0, "z": 0.0})
    assert out["route"] == "blocked" and "no length" in out["detail"]


def test_an_unreadable_pose_is_blocked(script):
    def broken(**kw):
        raise ToolError("robot.get_ee_pose", "no reading")

    out = script.run(Arm().ctx(**{"robot.get_ee_pose": broken}))
    assert out["route"] == "blocked" and "could not be read" in out["detail"]


def test_a_part_held_sideways_by_its_seat_is_seated(script):
    arm = Arm(give=0.001, side_give=0.001)
    out = script.run(arm.ctx(), lateral_distance=0.006)
    assert out["route"] == "seated"
    assert out["lateral_travel"] == pytest.approx(0.001)
    sideways = arm.pushes[1:]
    assert len(sideways) == 4
    for step in sideways:
        assert np.linalg.norm(step) == pytest.approx(0.006)
        assert abs(step[2]) < 1e-12
    assert np.allclose(arm.p, START)


def test_a_part_that_moves_sideways_is_not_seated(script):
    out = script.run(Arm(give=0.001, side_give=None).ctx(), lateral_distance=0.006)
    assert out["route"] == "not_seated"
    assert out["lateral_travel"] == pytest.approx(0.006)
    assert "sideways" in out["detail"]


def test_no_sideways_pushes_after_a_part_went_further(script):
    arm = Arm(give=None)
    out = script.run(arm.ctx(), lateral_distance=0.006)
    assert out["route"] == "not_seated" and len(arm.pushes) == 1


REFUSED = {"robot.go_to_pose_cartesian": {"reached": False, "outcome": "unreachable"}}


def test_a_failed_return_before_a_sideways_push_is_blocked(script):
    arm = Arm(give=0.001, refuse=REFUSED)
    arm.refused_drift = 0.02
    out = script.run(arm.ctx(), lateral_distance=0.006)
    assert out["route"] == "blocked" and "before sideways push 0" in out["detail"]


def test_a_failed_return_keeps_the_verdict_and_says_so(script):
    arm = Arm(give=0.001, refuse=REFUSED)
    arm.refused_drift = 0.02
    out = script.run(arm.ctx())
    assert out["route"] == "seated" and not out["returned"]
    assert "return to the start" in out["detail"]


def test_a_return_is_judged_by_where_the_hand_is(script):
    # A servo inside its arrival ball does not move; the hand is still at the start.
    out = script.run(Arm(give=0.001, refuse=REFUSED).ctx())
    assert out["route"] == "seated" and out["returned"]


def test_return_to_start_false_leaves_the_hand_at_the_stop(script):
    arm = Arm(give=0.002)
    out = script.run(arm.ctx(), return_to_start=False)
    assert out["route"] == "seated" and not out["returned"]
    assert arm.p[2] == pytest.approx(START[2] - 0.002)


def test_lateral_directions_are_square_to_the_axis(script):
    for axis in ([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.6, 0.0, -0.8]):
        a = np.asarray(axis)
        dirs = script.lateral_directions(a)
        assert len(dirs) == 4
        for d in dirs:
            assert np.linalg.norm(d) == pytest.approx(1.0)
            assert abs(float(np.dot(d, a))) < 1e-9
    down = script.lateral_directions(np.array([0.0, 0.0, -1.0]))
    assert np.allclose(down[0], [1.0, 0.0, 0.0]) and np.allclose(abs(down[2]), [0.0, 1.0, 0.0])


def test_arm_id_is_passed_only_when_nonzero(script):
    arm = Arm(give=0.001)
    script.run(arm.ctx())
    assert all("arm_id" not in kw for _, kw in arm.calls)
    arm = Arm(give=0.001)
    script.run(arm.ctx(), arm_id=1)
    assert arm.calls and all(kw.get("arm_id") == 1 for _, kw in arm.calls)
