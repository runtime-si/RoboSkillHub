"""grasping-a-cable-lying-flat: FakeContext tests of the depth, the jaws across the cable, the wrist turn and the routing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "grasping-a-cable-lying-flat"
REACH = 0.0102
SPAN = 0.094
R = 0.00175  # a 3.5 mm cable
# Straight along +y on a table at z = 0, 0.20 m long, centre a radius up.
CABLE = {
    "points": [[0.40, 0.10 + 0.01 * k, R] for k in range(21)],
    "arclength_m": 0.20,
    "radius_m": R,
    "nodes": 21,
    "ordered": True,
}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["grasp_flat_cable"].module


def down(a_deg: float) -> dict[str, float]:
    """Top-down: a half turn about the horizontal axis at a_deg, which takes the
    tool's x to the heading 2 * a_deg."""
    a = math.radians(a_deg)
    return {"w": 0.0, "x": math.cos(a), "y": math.sin(a), "z": 0.0}


def closing_heading(rotation: dict[str, float]) -> float:
    """Heading of the tool's x (the fake hand's closing axis) [deg]."""
    return math.degrees(2.0 * math.atan2(rotation["y"], rotation["x"]))


class Arm:
    """A top-down hand over a table at z = 0. Moves land where commanded except
    that the fingertips (REACH under the tool centre) stop on the table; a
    contact move stopped there reports contact. ``free_roll`` keeps the hand's
    own yaw on pose moves (a solver that leaves the wrist roll free); the last
    joint then turns it (gain -1: +theta on the joint is -theta of yaw)."""

    def __init__(
        self,
        start=(0.30, 0.0, 0.25),
        close_width=0.0054,
        free_roll=False,
        refuse=None,
        stand_at=None,
        ik=None,
        rise=None,
    ):
        self.p = np.array(start, dtype=float)
        self.a = 0.0  # half the closing heading [deg]
        self.q = np.zeros(6)
        self.width = SPAN
        self.close_width = close_width
        self.free_roll = free_roll
        self.refuse = refuse or (lambda tool, pose: None)
        self.stand_at = stand_at  # height a contact move stops at, instead of the table
        self.ik = ik or (lambda rotation: {"success": True, "within_servo_tolerance": True})
        self.rise = rise  # the cable beside the fingers after the lift, over its height before
        self.moves = []
        self.grips = []

    def _move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy(), pose["rotation"]))
            why = self.refuse(tool, pose)
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            lowest = (
                REACH
                if self.stand_at is None or tool != "robot.move_cartesian_until_contact"
                else self.stand_at
            )
            stopped = target[2] < lowest - 1e-9
            self.p = np.array([target[0], target[1], max(target[2], lowest)])
            if not self.free_roll:
                self.a = math.degrees(math.atan2(pose["rotation"]["y"], pose["rotation"]["x"]))
            if stopped and tool == "robot.move_cartesian_until_contact":
                return {"reached": False, "outcome": "contact"}
            if stopped:
                return {"reached": False, "outcome": "timeout"}
            return {"reached": True, "outcome": "reached"}

        return go

    def _joints(self, joint_config, **kwargs):
        q = np.asarray(joint_config["positions"], dtype=float)
        self.a -= math.degrees(q[-1] - self.q[-1]) / 2.0
        self.q = q
        return {"reached": True, "outcome": "reached"}

    def _close(self, **kwargs):
        self.width = self.close_width
        self.grips.append("close")
        return {"position": self.width / SPAN}

    def _observation(self):
        obs = {"arms": [{"joint_state": {"positions": self.q.tolist()}}], "cameras": []}
        if self.rise is not None:
            obs["cameras"] = [
                {
                    "name": "overhead",
                    "rgb": "o",
                    "depth": np.zeros((2, 2)),
                    "intrinsics": None,
                    "pose": None,
                },
                {
                    "name": "left_wrist",
                    "rgb": "w",
                    "depth": np.zeros((2, 2)),
                    "intrinsics": None,
                    "pose": None,
                },
            ]
        return obs

    def _points(self, **kwargs):
        ring = [
            [self.p[0] + 0.015 * math.cos(t), self.p[1] + 0.015 * math.sin(t), R + self.rise]
            for t in np.linspace(0.0, 2 * math.pi, 30)
        ]
        far = [[self.p[0] + 0.1, self.p[1], R]] * 30  # the rest of the cable, still on the table
        return {"points": {"points": ring + far}}

    def ctx(self, workspace_z=None):
        tools = {
            "robot.get_ee_pose": lambda **kw: {
                "pose": {
                    "position": dict(zip("xyz", map(float, self.p), strict=True)),
                    "rotation": down(self.a),
                }
            },
            "robot.go_to_pose": self._move("robot.go_to_pose"),
            "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
            "robot.move_cartesian_until_contact": self._move("robot.move_cartesian_until_contact"),
            "robot.move_to_joints": self._joints,
            "robot.solve_ik": lambda pose, **kw: self.ik(pose["rotation"]),
            "robot.grasp_frame": lambda close_heading_deg, **kw: {
                "rotation": down(close_heading_deg / 2.0),
                "rotation_flipped": down(close_heading_deg / 2.0 + 90.0),
            },
            "robot.describe_gripper": {
                "span_m": SPAN,
                "width_fit": {"at_open_m": SPAN, "at_closed_m": 0.0},
                "finger": {"reach_m": REACH},
            },
            "robot.open_gripper": lambda **kw: self.grips.append("open") or {"position": 1.0},
            "robot.close_gripper": self._close,
            "robot.set_grip": lambda width_m, **kw: self.grips.append(width_m) or {"ok": True},
            "robot.get_observation": lambda **kw: self._observation(),
            "sam3.segment_text": lambda image, **kw: {
                "masks": [np.ones((2, 2))] if image == "w" else []
            },
            "geometry.mask_to_world_points": self._points,
        }
        if workspace_z is not None:
            tools["robot.describe_workspace"] = {"surface_z": workspace_z}
        else:

            def missing(**kw):
                raise ToolError("robot.describe_workspace", "not served")

            tools["robot.describe_workspace"] = missing
        return FakeContext(tools)


def run(script, arm, **kwargs):
    kwargs.setdefault("centerline", CABLE)
    kwargs.setdefault("table_z", 0.0)
    kwargs.setdefault("min_tool_height", 0.005)
    return script.run(arm.ctx(), **kwargs)


def test_happy_path_sequence_and_depth(script):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(ctx, centerline=CABLE, table_z=0.0, min_tool_height=0.005)
    assert out["route"] == "grasped", out["detail"]
    assert not out["lift_checked"] and "jaw gap only" in out["detail"]
    # Mid-length of the cable along +y; the jaws close across it, along x.
    assert out["grasp_point"] == {
        "x": pytest.approx(0.40),
        "y": pytest.approx(0.20),
        "z": pytest.approx(R),
    }
    assert (
        abs(out["close_heading_deg"]) == pytest.approx(0.0, abs=1e-6)
        or abs(out["close_heading_deg"]) == 90.0
    )
    heading = ctx.calls_to("robot.grasp_frame")[0].kwargs["close_heading_deg"]
    assert heading % 180.0 == pytest.approx(0.0, abs=1e-6)  # square to +y
    # 3.5 mm is not thin: fingertips 4 mm under the centre, floored 0.8 mm over the table.
    tool_z = 0.0008 + REACH
    assert out["planned_tool_z"] == pytest.approx(tool_z)
    names = [m[0] for m in arm.moves]
    assert names == [
        "robot.go_to_pose",  # over the cable
        "robot.go_to_pose_cartesian",  # down to narrow the fingers
        "robot.move_cartesian_until_contact",  # to the table
        "robot.go_to_pose_cartesian",  # pressed while closing
        "robot.go_to_pose_cartesian",  # the test lift
    ]
    above, narrow, contact, press, lift = (m[1] for m in arm.moves)
    np.testing.assert_allclose(above, [0.40, 0.20, tool_z + 0.06], atol=1e-9)
    np.testing.assert_allclose(narrow, [0.40, 0.20, tool_z + 0.012], atol=1e-9)
    np.testing.assert_allclose(
        contact, [0.40, 0.20, 0.005], atol=1e-9
    )  # 8 mm under, floored at 5 mm
    np.testing.assert_allclose(press, [0.40, 0.20, REACH - 0.003], atol=1e-9)
    np.testing.assert_allclose(lift, [0.40, 0.20, REACH + 0.04], atol=1e-9)
    assert arm.grips == ["open", 0.010, "close"]
    assert out["touched_table"] and out["pressed"] and out["pre_closed"]
    assert out["fingertips_over_table"] == pytest.approx(0.0)
    assert out["jaw_width"] == pytest.approx(0.0054)
    assert all(m[2] == arm.moves[0][2] for m in arm.moves)


def test_the_real_station_floor_routes_unreachable_without_moving(script):
    arm = Arm()
    out = script.run(arm.ctx(), centerline=CABLE, table_z=0.0)  # default 14 mm floor
    assert out["route"] == "unreachable"
    assert "floor" in out["detail"] and "11.0 mm" in out["detail"]
    assert arm.moves == [] and arm.grips == []


def test_a_thin_cable_goes_deeper_and_nearer_the_table(script):
    thin = {**CABLE, "points": [[x, y, 0.001] for x, y, _ in CABLE["points"]], "radius_m": 0.001}
    out = run(script, Arm(), centerline=thin)
    assert out["planned_tool_z"] == pytest.approx(0.0004 + REACH)


def test_a_thick_cable_puts_the_fingertips_under_its_middle(script):
    thick = {**CABLE, "points": [[x, y, 0.008] for x, y, _ in CABLE["points"]], "radius_m": 0.008}
    out = run(script, Arm(), centerline=thick)
    assert out["planned_tool_z"] == pytest.approx(0.008 - 0.004 + REACH)


def test_a_given_point_and_tangent_close_square_to_it(script):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(
        ctx,
        grasp_point={"x": 0.35, "y": 0.05, "z": 0.004},
        tangent={"x": 1.0, "y": 1.0, "z": 0.0},
        table_z=0.0,
        cable_radius=R,
        min_tool_height=0.005,
    )
    assert out["route"] == "grasped"
    assert ctx.calls_to("robot.grasp_frame")[0].kwargs["close_heading_deg"] == pytest.approx(135.0)
    assert out["close_heading_deg"] == pytest.approx(-45.0)
    assert out["grasp_point"]["z"] == pytest.approx(
        R
    )  # flat: a radius over the table, not the camera's 4 mm
    assert closing_heading(arm.moves[-1][2]) % 180.0 == pytest.approx(135.0)


def test_back_from_either_end(script):
    out = run(script, Arm(), end={"node": 0}, back_from_end=0.03)
    assert out["grasp_point"]["y"] == pytest.approx(0.13)
    out = run(script, Arm(), end={"node": 20}, back_from_end=0.05)
    assert out["grasp_point"]["y"] == pytest.approx(0.25)


def test_a_bent_centreline_takes_the_local_tangent(script):
    # Along +y to (0.40, 0.20), then along +x.
    pts = [[0.40, 0.10 + 0.01 * k, R] for k in range(11)] + [
        [0.40 + 0.01 * k, 0.20, R] for k in range(1, 11)
    ]
    arm = Arm()
    ctx = arm.ctx()
    script.run(
        ctx,
        centerline={"points": pts, "radius_m": R},
        grasp_s=0.8,
        table_z=0.0,
        min_tool_height=0.005,
    )
    assert ctx.calls_to("robot.grasp_frame")[0].kwargs[
        "close_heading_deg"
    ] % 180.0 == pytest.approx(90.0)


def test_table_from_the_workspace_when_not_given(script):
    arm = Arm()
    out = script.run(arm.ctx(workspace_z=0.0), centerline=CABLE, min_tool_height=0.005)
    assert out["route"] == "grasped"
    assert out["planned_tool_z"] == pytest.approx(0.0008 + REACH)


def test_a_cable_off_the_table_is_not_lying_flat(script):
    up = {**CABLE, "points": [[x, y, 0.03] for x, y, _ in CABLE["points"]]}
    arm = Arm()
    out = run(script, arm, centerline=up)
    assert out["route"] == "unreachable" and "not lying flat" in out["detail"]
    assert arm.moves == []


def test_jaws_at_the_closed_stop_missed_and_open(script):
    arm = Arm(close_width=0.0)
    out = run(script, arm)
    assert out["route"] == "missed" and "closed stop" in out["detail"]
    assert arm.grips[-1] == "open"


def test_a_long_close_is_sent_twice(script):
    class Twice(Arm):
        def _close(self, **kwargs):
            out = super()._close(**kwargs)
            if self.grips.count("close") == 1:
                self.width = 0.03
                return {"position": self.width / SPAN}
            return out

    arm = Twice()
    out = run(script, arm)
    assert arm.grips.count("close") == 2
    assert out["route"] == "grasped" and "first close stopped at 30.0 mm" in out["detail"]


def test_the_camera_sees_the_cable_rise_with_the_hand(script):
    arm = Arm(rise=0.03)
    ctx = arm.ctx()
    out = script.run(
        ctx,
        centerline=CABLE,
        table_z=0.0,
        min_tool_height=0.005,
        cable_description="the white cable",
    )
    assert out["route"] == "grasped" and out["lift_checked"]
    assert out["cable_rise"] == pytest.approx(0.03)
    assert ctx.calls_to("sam3.segment_text")[0].kwargs["image"] == "w"  # the wrist camera


def test_the_camera_sees_the_cable_stay_on_the_table(script):
    arm = Arm(rise=0.002)
    out = run(script, arm, cable_description="the white cable")
    assert out["route"] == "missed" and out["lift_checked"]
    assert "rose 2.0 mm" in out["detail"]
    assert arm.grips[-1] == "open"


def test_a_free_wrist_roll_is_finished_on_the_last_joint(script):
    arm = Arm(free_roll=True)
    arm.a = 30.0  # the hand arrives closing at 60 deg, the cable wants 0 (mod 180)
    ctx = arm.ctx()
    out = script.run(ctx, centerline=CABLE, table_z=0.0, min_tool_height=0.005)
    assert out["route"] == "grasped", out["detail"]
    assert ctx.calls_to("robot.move_to_joints")
    final = closing_heading(arm.moves[-1][2])
    assert abs(((final + 90.0) % 180.0) - 90.0) < 5.0
    assert "off square" not in out["detail"]


def test_ik_preview_takes_the_flipped_wrist_then_gives_up(script):
    arm = Arm(ik=lambda rot: {"success": rot["x"] < 0.5, "within_servo_tolerance": True})
    out = run(script, arm)
    assert out["route"] == "grasped"
    arm = Arm(ik=lambda rot: {"success": True, "within_servo_tolerance": False})
    out = run(script, arm)
    assert out["route"] == "unreachable" and "no wrist solution" in out["detail"]
    assert arm.moves == []


def test_refused_moves_are_routed(script):
    arm = Arm(refuse=lambda tool, pose: "no IK" if tool == "robot.go_to_pose" else None)
    out = run(script, arm)
    assert out["route"] == "unreachable" and out["detail"].startswith("approach")

    class Raises(Arm):
        def _move(self, tool):
            go = super()._move(tool)

            def wrapped(pose, **kwargs):
                reply = go(pose, **kwargs)
                if reply["outcome"] == "contact":
                    raise RuntimeError("the tool did not reach its target")
                return reply

            return wrapped

    arm = Raises()
    out = run(script, arm)
    # A real station raises when the arm stops short: read as a stop where the hand stands.
    assert out["route"] == "grasped" and out["touched_table"]

    arm = Arm(
        refuse=lambda tool, pose: (
            "joint limits" if tool == "robot.move_cartesian_until_contact" else None
        )
    )
    out = run(script, arm)
    assert out["route"] == "unreachable" and out["detail"].startswith("descent: unreachable")
    assert arm.grips[-1] == "open"


def test_no_press_no_pre_close(script):
    arm = Arm()
    out = run(script, arm, press=0.0, pre_close_width=0.0)
    assert out["route"] == "grasped"
    assert [m[0] for m in arm.moves] == [
        "robot.go_to_pose",
        "robot.move_cartesian_until_contact",
        "robot.go_to_pose_cartesian",  # back to where it stands
        "robot.go_to_pose_cartesian",
    ]
    assert arm.moves[2][1][2] == pytest.approx(REACH)
    assert arm.grips == ["open", "close"]
    assert not out["pressed"] and not out["pre_closed"]


def test_a_stop_well_over_the_planned_depth_is_noted(script):
    arm = Arm(stand_at=0.02)
    out = run(script, arm)
    assert out["route"] == "grasped"
    assert not out["touched_table"]
    assert "over the planned depth" in out["detail"]


def test_a_stop_over_an_aim_above_the_table_is_not_the_table(script):
    # Aimed at a 14 mm floor (fingertips 3.8 mm over the table), the arm stops 2.6 mm over its aim in the air.
    arm = Arm(stand_at=0.0166)
    out = run(script, arm, pinch_below=-0.002, overtravel=0.0, press=0.0, min_tool_height=0.005)
    assert not out["touched_table"]
    assert "over its aim, which is over the table" in out["detail"]


def test_example_declares_every_route(script):
    example = (
        Path(__file__).resolve().parents[2]
        / "skills"
        / BUNDLE
        / "examples"
        / "canonical_subgraph.json"
    )
    graph = json.loads(example.read_text())
    declared = set(graph["exit"]["success_values"]) | {graph["on_error"]}
    assert declared == {"grasped", "missed", "unreachable"}
