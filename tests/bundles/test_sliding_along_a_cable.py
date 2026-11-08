"""sliding-along-a-cable: FakeContext tests of the cage, the path along the centreline, the stops and the routing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "sliding-along-a-cable"
SPAN = 0.094
R = 0.00175  # a 3.5 mm cable
PINCHED = 0.0054  # what a YAM station twin's jaws read closed on it
# Straight along +y on a table at z = 0, 0.26 m long; its end (the tip) at node 0, y = 0.10.
CABLE = {
    "points": [[0.40, 0.10 + 0.01 * k, R] for k in range(27)],
    "radius_m": R,
    "ordered": True,
}
TIP = {"node": 0, "position": [0.40, 0.10, R]}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["slide_along_cable"].module


def down(a_deg: float) -> dict[str, float]:
    """Top-down: a half turn about the horizontal axis at a_deg, which takes the
    tool's x to the heading 2 * a_deg."""
    a = math.radians(a_deg)
    return {"w": 0.0, "x": math.cos(a), "y": math.sin(a), "z": 0.0}


def jaw_heading(rotation: dict[str, float]) -> float:
    """Heading of the tool's x (the fake hand's closing axis) [deg]."""
    return math.degrees(2.0 * math.atan2(rotation["y"], rotation["x"]))


class Arm:
    """A top-down hand closed on the cable. Moves land where commanded, unless
    ``stall_at`` (a y the hand cannot pass, going toward smaller y) stops them or
    ``refuse`` says no. ``free_roll`` keeps the hand's yaw on pose moves (a solver
    that leaves the wrist roll free); the last joint turns it (gain -1)."""

    def __init__(
        self,
        start=(0.40, 0.25, 0.0112),
        heading=0.0,
        jaws=PINCHED,
        pinched=PINCHED,
        stall_at=None,
        refuse=None,
        ik=None,
        free_roll=False,
        widen_at=None,
        empty_after=False,
        q6=0.0,
        stuck_wrist=False,
    ):
        self.p = np.array(start, dtype=float)
        self.a = heading / 2.0
        self.q = np.zeros(6)
        self.q[-1] = q6
        self.stuck_wrist = stuck_wrist
        self.width = jaws
        self.pinched = pinched
        self.stall_at = stall_at
        self.refuse = refuse or (lambda pose: None)
        self.ik = ik or (lambda pose: {"success": True, "within_servo_tolerance": True})
        self.free_roll = free_roll
        self.widen_at = widen_at
        self.empty_after = empty_after
        self.moves = []
        self.grips = []
        self.previews = []

    def _move(self, pose, **kwargs):
        target = np.array([pose["position"][k] for k in "xyz"])
        self.moves.append((target.copy(), pose["rotation"]))
        why = self.refuse(pose)
        if why:
            return {"reached": False, "outcome": "unreachable", "reason": why}
        if self.stall_at is not None and target[1] < self.stall_at:
            frac = 0.2 if self.p[1] > self.stall_at else 0.0
            self.p = self.p + frac * (target - self.p)
            if not self.free_roll:
                self.a = math.degrees(math.atan2(pose["rotation"]["y"], pose["rotation"]["x"]))
            return {"reached": False, "outcome": "timeout"}
        self.p = target
        if not self.free_roll:
            self.a = math.degrees(math.atan2(pose["rotation"]["y"], pose["rotation"]["x"]))
        return {"reached": True, "outcome": "reached"}

    def _joints(self, joint_config, **kwargs):
        if self.stuck_wrist:
            return {"reached": False, "outcome": "unreachable", "reason": "joint limit"}
        q = np.asarray(joint_config["positions"], dtype=float)
        self.a -= math.degrees(q[-1] - self.q[-1]) / 2.0
        self.q = q
        return {"reached": True, "outcome": "reached"}

    def _width(self):
        if self.widen_at is not None and self.p[1] < self.widen_at:
            return self.width + 0.004
        return self.width

    def _close(self, **kwargs):
        self.width = 0.0 if self.empty_after else self.pinched
        self.grips.append("close")
        return {"position": self.width / SPAN}

    def _set_grip(self, width_m, **kwargs):
        self.grips.append(round(float(width_m), 6))
        self.width = float(width_m)
        return {"requested_width_m": width_m}

    def ctx(self):
        return FakeContext(
            {
                "robot.get_ee_pose": lambda **kw: {
                    "pose": {
                        "position": dict(zip("xyz", map(float, self.p), strict=True)),
                        "rotation": down(self.a),
                    }
                },
                "robot.go_to_pose_cartesian": self._move,
                "robot.move_to_joints": self._joints,
                "robot.solve_ik": lambda pose, **kw: self.previews.append(pose) or self.ik(pose),
                "robot.describe_gripper": {
                    "span_m": SPAN,
                    "width_fit": {"at_open_m": SPAN, "at_closed_m": 0.0},
                },
                "robot.get_gripper": lambda **kw: {"position": self._width() / SPAN},
                "robot.set_grip": self._set_grip,
                "robot.close_gripper": self._close,
                "robot.open_gripper": lambda **kw: self.grips.append("open") or {"position": 1.0},
                "robot.get_observation": lambda **kw: {
                    "arms": [{"joint_state": {"positions": self.q.tolist()}}]
                },
            }
        )


def run(script, arm, **kwargs):
    kwargs.setdefault("centerline", CABLE)
    kwargs.setdefault("end", TIP)
    return script.run(arm.ctx(), **kwargs)


def test_happy_path_cages_slides_to_the_planned_stop_and_closes(script):
    arm = Arm()
    out = run(script, arm)
    assert out["route"] == "reached_end", out["detail"]
    assert out["stopped_by"] == "planned_stop"
    # Cage: the jaws' reading on the cable plus 1.5 mm; then the firm close.
    assert arm.grips == [pytest.approx(PINCHED + 0.0015), "close"]
    assert out["cage_width"] == pytest.approx(PINCHED + 0.0015)
    assert out["jaw_width"] == pytest.approx(PINCHED)
    # 150 mm from the tip to 40 mm from it, in 1 cm steps, 1 mm up off the table.
    assert out["start_from_end"] == pytest.approx(0.15)
    assert out["end_from_end"] == pytest.approx(0.04)
    assert out["slid"] == pytest.approx(0.11)
    assert out["steps"] == 11
    rise, *steps, press = arm.moves
    np.testing.assert_allclose(rise[0], [0.40, 0.25, 0.0122], atol=1e-9)
    ys = [m[0][1] for m in steps]
    np.testing.assert_allclose(ys, [0.25 - 0.01 * (k + 1) for k in range(11)], atol=1e-9)
    assert all(m[0][0] == pytest.approx(0.40) and m[0][2] == pytest.approx(0.0122) for m in steps)
    # Square to the cable all the way (a straight cable: no turn).
    assert all(abs(jaw_heading(m[1]) % 180.0) < 1e-6 for m in steps)
    assert len(arm.previews) == 11 and out["wrist_turns"] == 0
    # Back down, 3 mm under where it started, while the jaws shut.
    np.testing.assert_allclose(press[0], [0.40, 0.14, 0.0112 - 0.003], atol=1e-9)
    assert "re-perceive" in out["detail"]


def test_the_end_at_the_last_node_slides_the_other_way(script):
    arm = Arm(start=(0.40, 0.20, 0.0112))
    end = {"node": 26, "position": [0.40, 0.36, R]}
    out = run(script, arm, end=end, stop_before_end=0.05)
    assert out["route"] == "reached_end"
    assert out["start_from_end"] == pytest.approx(0.16)
    assert arm.p[1] == pytest.approx(0.31)


def test_a_tip_past_the_last_node_counts_toward_the_stop(script):
    # curve.endpoints moves the end out by the radius: the stop is measured from there.
    arm = Arm()
    end = {"node": 0, "position": [0.40, 0.10 - R, R]}
    out = run(script, arm, end=end)
    assert out["start_from_end"] == pytest.approx(0.15 + R)
    assert arm.p[1] == pytest.approx(0.10 - R + 0.04)


def test_an_end_given_as_a_point_picks_the_nearer_end(script):
    arm = Arm()
    out = run(script, arm, end=[0.401, 0.099, 0.0])
    assert out["route"] == "reached_end"
    assert arm.p[1] == pytest.approx(0.10 + 0.04, abs=2e-3)


def test_an_open_hand_cages_at_the_diameter_plus_clearance(script):
    arm = Arm(jaws=SPAN)
    out = run(script, arm)
    assert out["route"] == "reached_end"
    assert arm.grips[0] == pytest.approx(2 * R + 0.0025)


def test_a_given_cage_width_is_used(script):
    arm = Arm()
    run(script, arm, cage_width=0.006)
    assert arm.grips[0] == pytest.approx(0.006)


def test_jaws_closed_on_nothing_route_lost_cable_without_moving(script):
    arm = Arm(jaws=0.0)
    out = run(script, arm)
    assert out["route"] == "lost_cable"
    assert "closed stop" in out["detail"]
    assert arm.moves == [] and arm.grips == []


def test_a_hand_off_the_centreline_routes_lost_cable(script):
    arm = Arm(start=(0.42, 0.25, 0.0112))
    out = run(script, arm)
    assert out["route"] == "lost_cable"
    assert "20.0 mm from the cable" in out["detail"]
    assert arm.moves == []


def test_a_small_offset_is_kept_along_the_path(script):
    arm = Arm(start=(0.403, 0.25, 0.0112))
    out = run(script, arm)
    assert out["route"] == "reached_end"
    assert all(m[0][0] == pytest.approx(0.403) for m in arm.moves)


def test_a_stall_near_the_tip_is_the_end_feature(script):
    # The plug's boot holds the hand back 58 mm from the tip (a fifth of a step past 60 mm), inside the 60 mm window.
    arm = Arm(stall_at=0.155)
    out = run(script, arm)
    assert out["route"] == "reached_end", out["detail"]
    assert out["stopped_by"] == "stall"
    assert "end feature" in out["detail"]
    assert out["end_from_end"] == pytest.approx(0.058, abs=1e-6)
    assert arm.grips[-1] == "close"


def test_a_stall_far_from_the_tip_routes_unreachable(script):
    arm = Arm(stall_at=0.215)
    out = run(script, arm)
    assert out["route"] == "unreachable"
    assert out["stopped_by"] == "stall" and "snag" in out["detail"]
    assert arm.grips[-1] == "close"  # held where it stopped


def test_jaws_pushed_open_stop_the_slide(script):
    arm = Arm(widen_at=0.165)
    out = run(script, arm)
    assert out["route"] == "reached_end"
    assert out["stopped_by"] == "jaws"
    assert arm.p[1] == pytest.approx(0.16)


def test_a_refused_step_routes_unreachable_and_holds(script):
    arm = Arm(refuse=lambda pose: "no solution" if pose["position"]["y"] < 0.20 else None)
    out = run(script, arm)
    assert out["route"] == "unreachable"
    assert out["stopped_by"] == "refused"
    assert arm.grips[-1] == "close"


def test_a_raised_refusal_is_routed_too(script):
    def refuse(pose):
        if pose["position"]["y"] < 0.22:
            raise RuntimeError("motion server refused the pose")

    arm = Arm(refuse=refuse)
    out = run(script, arm)
    assert out["route"] == "unreachable" and "motion server refused" in out["detail"]
    assert arm.grips[-1] == "close"


def test_an_unreachable_preview_stops_before_anything_moves(script):
    arm = Arm(
        ik=lambda pose: {"success": True, "within_servo_tolerance": pose["position"]["y"] > 0.16}
    )
    out = run(script, arm)
    assert out["route"] == "unreachable"
    assert "no arm solution" in out["detail"]
    assert arm.moves == [] and arm.grips == []


def test_check_reach_off_skips_the_preview(script):
    arm = Arm(ik=lambda pose: {"success": False})
    out = run(script, arm, check_reach=False)
    assert out["route"] == "reached_end" and arm.previews == []


def test_an_empty_firm_close_routes_lost_cable_and_rises(script):
    arm = Arm(empty_after=True)
    out = run(script, arm)
    assert out["route"] == "lost_cable"
    assert arm.grips[-2:] == ["close", "open"]
    assert arm.moves[-1][0][2] == pytest.approx(0.0112 - 0.003 + 0.04)


def test_press_off_closes_where_the_hand_stands(script):
    arm = Arm()
    out = run(script, arm, press=0.0)
    assert out["route"] == "reached_end"
    assert arm.moves[-1][0][2] == pytest.approx(0.0122)


def test_a_refused_press_is_noted_and_the_close_goes_on(script):
    arm = Arm(refuse=lambda pose: "under the floor" if pose["position"]["z"] < 0.0112 else None)
    out = run(script, arm)
    assert out["route"] == "reached_end"
    assert "press before the close was refused" in out["detail"]
    assert arm.grips[-1] == "close"


def test_jaws_held_open_by_the_cable_route_lost_cable(script):
    # A missed grasp left the jaws at 26 mm on the cable; set_grip does not close them.
    arm = Arm(jaws=0.026)
    arm._set_grip = lambda width_m, **kw: arm.grips.append(width_m) or {}
    out = run(script, arm)
    assert out["route"] == "lost_cable"
    assert "holds them open" in out["detail"]
    assert arm.moves == []


def test_already_at_the_stop_just_closes(script):
    arm = Arm(start=(0.40, 0.141, 0.0112))
    out = run(script, arm)
    assert out["route"] == "reached_end" and out["stopped_by"] == "already_there"
    assert len(arm.moves) == 1 and out["steps"] == 0  # only the press while closing
    assert arm.grips == [pytest.approx(PINCHED + 0.0015), "close"]


def test_a_curved_cable_turns_the_jaws_with_it(script):
    # A quarter circle of radius 0.2 m, from the tip at (0.40, 0.10) heading +y, turning left.
    pts = [
        [0.40 + 0.2 * (math.cos(t) - 1.0), 0.10 + 0.2 * math.sin(t), R]
        for t in np.linspace(0.0, math.pi / 2, 40)
    ]
    curve = {"points": pts, "radius_m": R}
    s0 = 0.15
    start = (0.40 + 0.2 * (math.cos(s0 / 0.2) - 1.0), 0.10 + 0.2 * math.sin(s0 / 0.2), 0.0112)
    tangent_deg = math.degrees(s0 / 0.2) + 90.0
    arm = Arm(start=start, heading=tangent_deg + 90.0, free_roll=True)
    out = run(script, arm, centerline=curve, end={"node": 0, "position": pts[0]})
    assert out["route"] == "reached_end", out["detail"]
    assert out["end_from_end"] == pytest.approx(0.04, abs=1e-3)
    # The arm left the roll alone: the last joint turned the jaws, and they end square.
    assert out["wrist_turns"] >= 1
    end_tangent = math.degrees(0.04 / 0.2) + 90.0
    assert abs(((jaw_heading(down(arm.a)) - end_tangent - 90.0) + 90.0) % 180.0 - 90.0) <= 4.0


def quarter_circle():
    """A quarter circle of radius 0.2 m from the tip at (0.40, 0.10) heading +y,
    turning left, and the hand 0.15 m along it, square to it."""
    pts = [
        [0.40 + 0.2 * (math.cos(t) - 1.0), 0.10 + 0.2 * math.sin(t), R]
        for t in np.linspace(0.0, math.pi / 2, 40)
    ]
    start = (0.40 + 0.2 * (math.cos(0.75) - 1.0), 0.10 + 0.2 * math.sin(0.75), 0.0112)
    return {"points": pts, "radius_m": R}, start, math.degrees(0.75) + 180.0


def test_a_turn_the_last_joint_cannot_follow_is_unreachable_up_front(script):
    curve, start, heading = quarter_circle()
    arm = Arm(start=start, heading=heading, free_roll=True, q6=1.9)
    out = run(script, arm, centerline=curve, end={"node": 0, "position": curve["points"][0]})
    assert out["route"] == "unreachable"
    assert "no room to follow" in out["detail"] and "other wrist solution" in out["detail"]
    assert arm.moves == [] and arm.grips == []


def test_jaws_left_off_square_stop_the_slide(script):
    curve, start, heading = quarter_circle()
    arm = Arm(start=start, heading=heading, free_roll=True, stuck_wrist=True)
    out = run(script, arm, centerline=curve, end={"node": 0, "position": curve["points"][0]})
    assert out["route"] == "unreachable"
    assert out["stopped_by"] == "refused" and "could not turn them back" in out["detail"]
    assert out["end_from_end"] > 0.08  # stopped early, about 10 deg of curve in
    assert arm.grips[-1] == "close"


def test_a_skewed_grasp_keeps_its_angle(script):
    arm = Arm(heading=10.0)
    run(script, arm, close_heading_deg=10.0)
    assert all(abs(jaw_heading(m[1]) - 10.0) < 1e-6 for m in arm.moves)


def test_missing_inputs_raise(script):
    arm = Arm()
    with pytest.raises(ValueError):
        script.run(arm.ctx(), end=TIP)
    with pytest.raises(ValueError):
        script.run(arm.ctx(), centerline=CABLE)


def test_a_short_centreline_is_unreachable(script):
    out = run(script, Arm(), centerline={"points": [[0.4, 0.1, R]]})
    assert out["route"] == "unreachable"


def test_example_declares_every_route(script):
    path = (
        Path(__file__).resolve().parents[2]
        / "skills"
        / BUNDLE
        / "examples"
        / "canonical_subgraph.json"
    )
    example = json.loads(path.read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert {"reached_end", "lost_cable", "unreachable"} <= declared
