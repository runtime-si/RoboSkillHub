"""grasping-a-cable-by-its-plug: FakeContext tests of locating the plug, the depth, the jaws across it and the routing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "grasping-a-cable-by-its-plug"
REACH = 0.0102
SPAN = 0.094
R = 0.00175
# A cable along +y on a table at z = 0; its plug end is node 0 at y = 0.10, pointing -y.
CABLE = {
    "points": [[0.40, 0.10 + 0.01 * k, R] for k in range(21)],
    "radius_m": R,
    "nodes": 21,
    "ordered": True,
}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["grasp_plug"].module


def down(a_deg: float) -> dict[str, float]:
    """Top-down: a half turn about the horizontal axis at a_deg, which takes the
    tool's x to the heading 2 * a_deg."""
    a = math.radians(a_deg)
    return {"w": 0.0, "x": math.cos(a), "y": math.sin(a), "z": 0.0}


def closing_heading(rotation: dict[str, float]) -> float:
    """Heading of the tool's x (the fake hand's closing axis) [deg]."""
    return math.degrees(2.0 * math.atan2(rotation["y"], rotation["x"]))


class Arm:
    """A top-down hand over a table at z = 0. Moves land where commanded, except
    that a descent under ``stand_at`` stops there (a fingertip on the plug) and
    reports contact. ``free_roll`` keeps the hand's own yaw on pose moves; the last
    joint then turns it (gain -1)."""

    def __init__(
        self,
        start=(0.30, 0.0, 0.25),
        close_width=0.0125,
        lifted_width=None,
        free_roll=False,
        refuse=None,
        stand_at=None,
        ik=None,
        seen=None,
        pointed=None,
    ):
        self.p = np.array(start, dtype=float)
        self.a = 0.0
        self.q = np.zeros(6)
        self.width = SPAN
        self.close_width = close_width
        self.lifted_width = close_width if lifted_width is None else lifted_width
        self.free_roll = free_roll
        self.refuse = refuse or (lambda tool, pose: None)
        self.stand_at = stand_at
        self.ik = ik or (lambda pose: {"success": True, "within_servo_tolerance": True})
        self.seen = seen  # the plug's points after the lift: height over the table
        self.pointed = pointed  # the world point under gemini-er's point
        self.moves = []
        self.grips = []

    def _move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy(), pose["rotation"]))
            why = self.refuse(tool, pose)
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            stopped = self.stand_at is not None and target[2] < self.stand_at - 1e-9
            self.p = np.array([target[0], target[1], self.stand_at if stopped else target[2]])
            if not self.free_roll:
                self.a = math.degrees(math.atan2(pose["rotation"]["y"], pose["rotation"]["x"]))
            if stopped:
                return {"reached": False, "outcome": "contact"}
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

    def _gripper(self, **kwargs):
        if "close" in self.grips:
            self.width = self.lifted_width
        return {"position": self.width / SPAN}

    def _observation(self):
        cams = [
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
        return {"arms": [{"joint_state": {"positions": self.q.tolist()}}], "cameras": cams}

    def _point(self, image, **kwargs):
        if self.pointed is None:
            return {"pixel_x": 0.0, "pixel_y": 0.0, "found": False, "points": []}
        assert image == "o"
        return {"pixel_x": 10.0, "pixel_y": 20.0, "found": True, "points": []}

    def _points(self, **kwargs):
        cluster = [
            [self.p[0] + 0.005 * math.cos(t), self.p[1] + 0.005 * math.sin(t), self.seen]
            for t in np.linspace(0.0, 2 * math.pi, 30)
        ]
        far = [[self.p[0] + 0.1, self.p[1], 0.004]] * 30
        return {"points": {"points": cluster + far}}

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
            "robot.move_to_joints": self._joints,
            "robot.solve_ik": lambda pose, **kw: self.ik(pose),
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
            "robot.get_gripper": self._gripper,
            "robot.set_grip": lambda width_m, **kw: self.grips.append(width_m) or {"ok": True},
            "robot.get_observation": lambda **kw: self._observation(),
            "gemini-er.point": self._point,
            "geometry.pixel_to_world_point": lambda **kw: {
                "point": dict(zip("xyz", map(float, self.pointed), strict=True))
            },
            "sam3.segment_text": lambda image, **kw: {
                "masks": [np.ones((2, 2))] if image == "w" and self.seen is not None else []
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
    return script.run(arm.ctx(), **kwargs)


def test_happy_path_sequence_and_depth(script):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(ctx, centerline=CABLE, table_z=0.0)
    assert out["route"] == "grasped", out["detail"]
    assert not out["lift_checked"] and "jaw gap only" in out["detail"]
    # The body centre 10 mm behind the tip (half the 20 mm plug), at half its 8 mm height.
    assert out["grasp_point"] == {
        "x": pytest.approx(0.40),
        "y": pytest.approx(0.11),
        "z": pytest.approx(0.004),
    }
    heading = ctx.calls_to("robot.grasp_frame")[0].kwargs["close_heading_deg"]
    assert heading % 180.0 == pytest.approx(0.0, abs=1e-6)  # across the plug's axis (y)
    # 1.5 mm + reach is under the 14 mm floor: the floor, fingertips 3.8 mm up (under 4 mm).
    assert out["planned_tool_z"] == pytest.approx(0.014)
    assert out["fingertips_over_table"] == pytest.approx(0.014 - REACH)
    assert [m[0] for m in arm.moves] == [
        "robot.go_to_pose",  # over the plug
        "robot.go_to_pose_cartesian",  # down around it
        "robot.go_to_pose_cartesian",  # the test lift
    ]
    above, grasp, lift = (m[1] for m in arm.moves)
    np.testing.assert_allclose(above, [0.40, 0.11, 0.014 + 0.06], atol=1e-9)
    np.testing.assert_allclose(grasp, [0.40, 0.11, 0.014], atol=1e-9)
    np.testing.assert_allclose(lift, [0.40, 0.11, 0.054], atol=1e-9)
    assert arm.grips == [pytest.approx(0.028), "close"]  # 12 mm + 16 mm, then close
    assert out["jaw_width"] == pytest.approx(0.0125)
    assert out["jaw_width_lifted"] == pytest.approx(0.0125)
    assert all(m[2] == arm.moves[0][2] for m in arm.moves)
    # IK previewed at the grasp and over it.
    zs = [c.kwargs["pose"]["position"]["z"] for c in ctx.calls_to("robot.solve_ik")]
    assert zs == [pytest.approx(0.014), pytest.approx(0.074)]


def test_the_far_end_an_end_dict_and_a_tip_with_direction(script):
    out = run(script, Arm(), end_node=-1)
    assert out["grasp_point"]["y"] == pytest.approx(0.29)  # the y = 0.30 end, pointing +y
    end = {"node": 0, "position": [0.35, 0.05, R], "direction": [1.0, 1.0, 0.0]}
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(ctx, centerline=None, end=end, table_z=0.0)
    assert out["route"] == "grasped"
    assert ctx.calls_to("robot.grasp_frame")[0].kwargs["close_heading_deg"] == pytest.approx(135.0)
    assert out["close_heading_deg"] == pytest.approx(-45.0)
    d = 0.01 / math.sqrt(2.0)
    assert (out["grasp_point"]["x"], out["grasp_point"]["y"]) == (
        pytest.approx(0.35 - d),
        pytest.approx(0.05 - d),
    )
    assert closing_heading(arm.moves[-1][2]) % 180.0 == pytest.approx(135.0)
    out = run(
        script,
        Arm(),
        centerline=None,
        tip={"x": 0.4, "y": 0.1, "z": R},
        direction={"x": 0, "y": -1, "z": 0},
    )
    assert out["grasp_point"]["y"] == pytest.approx(0.11)
    with pytest.raises(ValueError):
        run(script, Arm(), centerline=None)


def test_body_offset_and_plug_size(script):
    out = run(script, Arm(), body_offset=0.0185, plug_length=0.020, plug_height=0.010)
    assert out["grasp_point"]["y"] == pytest.approx(0.1185)
    assert out["grasp_point"]["z"] == pytest.approx(0.005)
    assert out["plug_height"] == pytest.approx(0.010)


def test_a_thin_plug_at_the_real_floor_is_unreachable_without_moving(script):
    arm = Arm()
    out = run(script, arm, plug_height=0.0065)
    assert out["route"] == "unreachable"
    assert "3.8 mm over the table" in out["detail"] and "upper edge" in out["detail"]
    assert arm.moves == [] and arm.grips == []
    out = run(script, Arm(), plug_height=0.0065, min_tool_height=0.005)
    assert out["route"] == "grasped"
    assert out["planned_tool_z"] == pytest.approx(0.0015 + REACH)


def test_a_plug_wider_than_the_hand(script):
    arm = Arm()
    out = run(script, arm, plug_width=0.095)
    assert out["route"] == "unreachable" and "wider than the hand opens" in out["detail"]
    assert arm.moves == []


def test_the_jaws_open_no_wider_than_the_hand(script):
    arm = Arm(close_width=0.085)
    out = run(script, arm, plug_width=0.085)
    assert arm.grips[0] == pytest.approx(SPAN)
    assert out["route"] == "grasped"


def test_a_raised_end_is_not_resting_on_the_table(script):
    up = {**CABLE, "points": [[x, y, 0.03] for x, y, _ in CABLE["points"]]}
    arm = Arm()
    out = run(script, arm, centerline=up)
    assert out["route"] == "unreachable" and "not resting" in out["detail"]
    assert arm.moves == []


def test_a_vertical_end_is_not_lying_flat(script):
    out = run(script, Arm(), centerline=None, tip=[0.4, 0.1, 0.0], direction=[0.0, 0.0, 1.0])
    assert out["route"] == "unreachable" and "not lying flat" in out["detail"]


def test_table_from_the_workspace_or_unreachable(script):
    arm = Arm()
    out = script.run(arm.ctx(workspace_z=0.0), centerline=CABLE)
    assert out["route"] == "grasped"
    out = script.run(Arm().ctx(), centerline=CABLE)
    assert out["route"] == "unreachable" and "table height" in out["detail"]


def test_jaws_at_the_closed_stop_missed_and_open(script):
    arm = Arm(close_width=0.0)
    out = run(script, arm)
    assert out["route"] == "missed" and "closed stop" in out["detail"]
    assert arm.grips[-1] == "open"
    assert arm.moves[-1][1][2] == pytest.approx(0.074)  # back up at the approach height


def test_jaws_on_the_jacket_missed(script):
    arm = Arm(close_width=0.0054)
    out = run(script, arm)
    assert out["route"] == "missed" and "jacket" in out["detail"]
    assert arm.grips[-1] == "open"
    assert "robot.go_to_pose_cartesian" in [m[0] for m in arm.moves]


def test_a_plug_that_slips_out_with_the_lift(script):
    arm = Arm(close_width=0.0125, lifted_width=0.002)
    out = run(script, arm)
    assert out["route"] == "missed" and "slipped out" in out["detail"]
    assert out["jaw_width_lifted"] == pytest.approx(0.002)
    assert arm.grips[-1] == "open"


def test_a_wide_reading_is_noted_as_askew(script):
    out = run(script, Arm(close_width=0.019))
    assert out["route"] == "grasped" and "askew" in out["detail"]


def test_a_fingertip_landing_on_the_plug_missed(script):
    arm = Arm(stand_at=0.024)
    out = run(script, arm)
    assert out["route"] == "missed" and "fingertip on the plug" in out["detail"]
    assert "close" not in arm.grips and arm.grips[-1] == "open"


def test_a_small_stop_over_the_grasp_still_closes(script):
    # The servo calls it arrived 3 mm high (a real arm stands 1.7-3.2 mm over near the table).
    out = run(script, Arm(stand_at=0.017))
    assert out["route"] == "grasped"
    assert out["fingertips_over_table"] == pytest.approx(0.017 - REACH)


def test_pointing_moves_the_centre_and_measures_the_height(script):
    arm = Arm(pointed=(0.405, 0.115, 0.0095))
    ctx = arm.ctx()
    out = script.run(ctx, centerline=CABLE, table_z=0.0, plug_description="the black plug")
    assert out["located_by"] == "point"
    assert out["plug_height"] == pytest.approx(0.0095)
    assert out["grasp_point"] == {
        "x": pytest.approx(0.405),
        "y": pytest.approx(0.115),
        "z": pytest.approx(0.00475),
    }
    assert ctx.calls_to("gemini-er.point")[0].kwargs["query"] == "the black plug"


def test_a_pointed_plug_far_from_the_end_is_ignored(script):
    out = run(script, Arm(pointed=(0.50, 0.11, 0.008)), plug_description="the black plug")
    assert out["located_by"] == "end"
    assert out["grasp_point"]["x"] == pytest.approx(0.40)
    assert "from the cable end's" in out["detail"]


def test_a_pointed_plug_standing_on_something(script):
    arm = Arm(pointed=(0.40, 0.11, 0.025))
    out = run(script, arm, plug_description="the black plug", plug_height=0.008)
    assert out["route"] == "unreachable" and "lies on something" in out["detail"]
    assert arm.moves == []


def test_nothing_pointed_keeps_the_end(script):
    out = run(script, Arm(), plug_description="the black plug")
    assert out["located_by"] == "end" and "not pointed" in out["detail"]


def test_the_camera_sees_the_plug_rise_with_the_hand(script):
    arm = Arm(seen=0.044)
    ctx = arm.ctx()
    out = script.run(ctx, centerline=CABLE, table_z=0.0, plug_description="the black plug")
    assert out["route"] == "grasped" and out["lift_checked"]
    assert out["plug_rise"] == pytest.approx(0.040)
    assert ctx.calls_to("sam3.segment_text")[0].kwargs["image"] == "w"


def test_the_camera_sees_the_plug_stay_down(script):
    arm = Arm(seen=0.004)
    out = run(script, arm, plug_description="the black plug")
    assert out["route"] == "missed" and out["lift_checked"]
    assert "rose 0.0 mm" in out["detail"]
    assert arm.grips[-1] == "open"


def test_a_free_wrist_roll_is_finished_on_the_last_joint(script):
    arm = Arm(free_roll=True)
    arm.a = 30.0
    ctx = arm.ctx()
    out = script.run(ctx, centerline=CABLE, table_z=0.0)
    assert out["route"] == "grasped", out["detail"]
    assert ctx.calls_to("robot.move_to_joints")
    final = closing_heading(arm.moves[-1][2])
    assert abs(((final + 90.0) % 180.0) - 90.0) < 5.0
    assert "off the plug's short axis" not in out["detail"]


def test_ik_preview_takes_the_flipped_wrist_then_gives_up(script):
    arm = Arm(
        ik=lambda pose: {"success": pose["rotation"]["x"] < 0.5, "within_servo_tolerance": True}
    )
    out = run(script, arm)
    assert out["route"] == "grasped"
    assert closing_heading(arm.moves[0][2]) % 360.0 == pytest.approx(180.0)
    # Reaches the grasp but not over it: refused too.
    arm = Arm(
        ik=lambda pose: {"success": True, "within_servo_tolerance": pose["position"]["z"] < 0.05}
    )
    out = run(script, arm)
    assert out["route"] == "unreachable" and "no wrist solution" in out["detail"]
    assert arm.moves == []


def test_refused_moves_are_routed(script):
    arm = Arm(refuse=lambda tool, pose: "no IK" if tool == "robot.go_to_pose" else None)
    out = run(script, arm)
    assert out["route"] == "unreachable" and out["detail"].startswith("approach")

    arm = Arm(
        refuse=lambda tool, pose: (
            "joint limits"
            if tool == "robot.go_to_pose_cartesian" and pose["position"]["z"] < 0.02
            else None
        )
    )
    out = run(script, arm)
    assert out["route"] == "unreachable" and out["detail"].startswith("descent: unreachable")
    assert arm.grips[-1] == "open"

    class Raises(Arm):
        def _move(self, tool):
            go = super()._move(tool)

            def wrapped(pose, **kwargs):
                reply = go(pose, **kwargs)
                if reply["outcome"] == "contact":
                    raise RuntimeError("the tool did not reach its target")
                return reply

            return wrapped

    # A real station raises when the arm stops short: judged by where the hand stands.
    arm = Raises(stand_at=0.03)
    out = run(script, arm)
    assert out["route"] == "missed" and "did not reach" in out["detail"]


def test_a_low_start_rises_first(script):
    arm = Arm(start=(0.30, 0.0, 0.03))
    out = run(script, arm)
    assert out["route"] == "grasped"
    first = arm.moves[0]
    assert first[0] == "robot.go_to_pose_cartesian"
    np.testing.assert_allclose(first[1], [0.30, 0.0, 0.074], atol=1e-9)


def test_arm_id_is_passed_only_when_nonzero(script):
    arm = Arm()
    ctx = arm.ctx()
    script.run(ctx, centerline=CABLE, table_z=0.0)
    assert "arm_id" not in ctx.calls_to("robot.close_gripper")[0].kwargs
    arm = Arm()
    ctx = arm.ctx()
    script.run(ctx, centerline=CABLE, table_z=0.0, arm_id=1, check_reach=False)
    assert ctx.calls_to("robot.close_gripper")[0].kwargs["arm_id"] == 1


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
