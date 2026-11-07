"""sliding-to-edge-then-picking: FakeContext tests of the edge geometry, the drag, the pinch and the routing."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "sliding-to-edge-then-picking"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
FLIP = {"w": 0.0, "x": 0.0, "y": 1.0, "z": 0.0}
# A 60 x 40 x 8 mm slab on a table at z = 0: long side along x, its +x face
# 80 mm from the table's edge at x = 0.50 (centre x = 0.39).
SLAB = {
    "center": {"x": 0.39, "y": 0.10, "z": 0.004},
    "extent": {"x": 0.03, "y": 0.02, "z": 0.004},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}
EDGE = {"x": 0.50, "y": 0.0, "z": 0.0}
ALONG_Y = {"x": 0.0, "y": 1.0, "z": 0.0}
GRIPPER = {
    "span_m": 0.094,
    "max_grasp_width_m": 0.086,
    "width_fit": {"at_open_m": 0.094, "at_closed_m": 0.0},
}


@pytest.fixture(scope="module")
def scripts(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["slide_to_edge"].module, bundle.canonical_scripts[
        "pick_overhang"
    ].module


def yaw_quat(deg: float) -> dict[str, float]:
    """DOWN turned about world z by ``deg``."""
    a = math.radians(deg) / 2.0
    qz = np.array([math.cos(a), 0.0, 0.0, math.sin(a)])
    # qz * DOWN, DOWN = (0, 1, 0, 0)
    w, x, y, z = qz
    return {"w": float(-x), "x": float(w), "y": float(z), "z": float(-y)}


class Arm:
    """Moves land where commanded unless a hook says otherwise; the jaws close
    on ``held`` metres when the hand is over the overhang, else on nothing."""

    def __init__(
        self, start=(0.30, 0.0, 0.25), refuse=None, stop_drag_at=None, held=0.040, grasp_stop=None
    ):
        self.p = np.array(start, dtype=float)
        self.rot = DOWN
        self.refuse = refuse or (lambda tool, pose: None)
        self.stop_drag_at = stop_drag_at
        self.held = held
        self.grasp_stop = grasp_stop
        self.fraction = 1.0
        self.moves: list[tuple[str, np.ndarray, dict]] = []
        self.q = np.zeros(6)

    def _move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy(), pose["rotation"]))
            why = self.refuse(tool, pose)
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            if self.stop_drag_at is not None and tool.endswith("cartesian") and target[2] < 0.03:
                target = target.copy()
                target[0] = min(target[0], self.stop_drag_at)
            if tool == "robot.move_cartesian_until_contact":
                if target[2] < 0.03 and self.p[0] < 0.50 and self.grasp_stop is None:
                    # pressing onto the slab top: the fingertips meet it 10 mm under the tool centre
                    stop = max(target[2], 0.008 + 0.010)
                    hit = stop > target[2] + 1e-9
                    self.p = np.array([target[0], target[1], stop])
                    return {"reached": not hit, "outcome": "contact" if hit else "reached"}
                if self.grasp_stop is not None and target[2] < 0.03:
                    self.p = np.array([target[0], target[1], self.grasp_stop])
                    return {"reached": False, "outcome": "contact"}
            self.p, self.rot = target, pose["rotation"]
            return {"reached": True, "outcome": "reached"}

        return go

    def close(self, **kw):
        self.fraction = (self.held / 0.094) if self.p[0] > 0.50 else 0.0
        return {"position": self.fraction}

    def open(self, **kw):
        self.fraction = 1.0
        return {"position": 1.0}

    def ctx(self, frame=True, extra=None):
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
            "robot.close_gripper": self.close,
            "robot.open_gripper": self.open,
            "robot.get_gripper": lambda **kw: {"position": self.fraction},
            "robot.describe_gripper": GRIPPER,
            "robot.solve_ik": {"success": True, "within_servo_tolerance": True},
            "robot.get_observation": lambda **kw: {
                "arms": [{"joint_state": {"positions": self.q.tolist()}}]
            },
        }
        if frame:
            tools["robot.grasp_frame"] = {"rotation": DOWN, "rotation_flipped": FLIP}
        else:

            def missing(**kw):
                raise ToolError("robot.grasp_frame", "not served")

            tools["robot.grasp_frame"] = missing
        tools.update(extra or {})
        return FakeContext(tools)


def slide_then_pick(scripts, arm, ctx=None, **kwargs):
    slide, pick = scripts
    ctx = ctx or arm.ctx()
    a = slide.run(
        ctx, target_obb=kwargs.pop("obb", SLAB), edge_point=kwargs.pop("edge", EDGE), **kwargs
    )
    if a["route"] != "slid":
        return a, None
    return a, pick.run(ctx, plan=a["plan"], slide_route=a["route"])


def test_push_slides_to_the_overhang_and_picks(scripts):
    arm = Arm()
    a, b = slide_then_pick(scripts, arm, edge_direction=ALONG_Y)
    assert a["route"] == "slid", a["detail"]
    # Overhang wanted: 0.4 x 60 = 24 mm; now -80 mm; slide 104 mm.
    assert a["overhang_before"] == pytest.approx(-0.080)
    assert a["overhang_target"] == pytest.approx(0.024)
    assert a["slide_planned"] == pytest.approx(0.104)
    assert a["slide_achieved"] == pytest.approx(0.104)
    assert a["overhang_expected"] == pytest.approx(0.024)
    tools = [m[0] for m in arm.moves]
    # hover, lowered, 13 drag steps (124 mm in 10 mm), lift, ...
    assert tools[:2] == ["robot.go_to_pose", "robot.go_to_pose_cartesian"]
    hover, lowered = arm.moves[0][1], arm.moves[1][1]
    # Back face 0.36; the blades stand 10 mm ahead of the tool centre, 3 mm
    # padding: contact 0.347, standoff 0.02 behind it.
    np.testing.assert_allclose(lowered, [0.327, 0.10, 0.015], atol=1e-9)
    np.testing.assert_allclose(hover, [0.327, 0.10, 0.095], atol=1e-9)
    drags = [m[1] for m in arm.moves[2:15]]
    assert len(drags) == 13  # 20 + 3 + 104 = 127 mm in 10 mm steps
    np.testing.assert_allclose(drags[0], [0.337, 0.10, 0.015], atol=1e-9)
    np.testing.assert_allclose(drags[-1], [0.454, 0.10, 0.015], atol=1e-9)
    assert b["route"] == "picked", b["detail"]
    assert b["grasp_position"] == {
        "x": pytest.approx(0.513),
        "y": pytest.approx(0.10),
        "z": pytest.approx(0.004),
    }
    assert b["held_width"] == pytest.approx(0.040)
    assert arm.p[2] == pytest.approx(0.004 + 0.05)


def test_an_edge_point_alone_takes_the_edge_square_to_the_slab(scripts):
    slide, _ = scripts
    out = slide.run(Arm().ctx(), target_obb=SLAB, edge_point={"x": 0.50, "y": 0.10, "z": 0.0})
    assert out["slide_planned"] == pytest.approx(0.104)
    assert out["plan"]["edge_normal"] == {
        "x": pytest.approx(1.0),
        "y": pytest.approx(0.0),
        "z": 0.0,
    }


def test_the_normal_points_away_from_the_slab_whatever_the_direction_s_sign(scripts):
    slide, _ = scripts
    out = slide.run(
        Arm().ctx(),
        target_obb=SLAB,
        edge_point=EDGE,
        edge_direction={"x": 0.0, "y": -1.0, "z": 0.0},
    )
    assert out["plan"]["edge_normal"]["x"] == pytest.approx(1.0)


def test_a_turned_slab_overhangs_by_its_corner(scripts):
    slide, _ = scripts
    q = {"w": math.cos(math.pi / 12), "x": 0.0, "y": 0.0, "z": math.sin(math.pi / 12)}  # 30 deg
    out = slide.run(
        Arm().ctx(), target_obb={**SLAB, "orientation": q}, edge_point=EDGE, edge_direction=ALONG_Y
    )
    front = 0.03 * math.cos(math.pi / 6) + 0.02 * math.sin(math.pi / 6)
    assert out["overhang_before"] == pytest.approx(front - 0.11)


def test_press_mode_presses_on_the_top_then_drags_deeper(scripts):
    arm = Arm()
    slide, _ = scripts
    out = slide.run(
        arm.ctx(), target_obb=SLAB, edge_point=EDGE, edge_direction=ALONG_Y, mode="press"
    )
    assert out["route"] == "slid", out["detail"]
    contact = [m for m in arm.moves if m[0] == "robot.move_cartesian_until_contact"][0][1]
    # 12 mm inside the back edge (x = 0.36), down to the 15 mm floor; stopped at 18 mm.
    np.testing.assert_allclose(contact, [0.372, 0.10, 0.015], atol=1e-9)
    drags = [m[1] for m in arm.moves if m[0] == "robot.go_to_pose_cartesian" and m[1][2] < 0.03]
    assert drags[0][2] == pytest.approx(0.018 - 0.0015)
    assert out["slide_achieved"] == pytest.approx(0.104)


def test_press_stop_inside_the_servo_ball_still_counts_as_the_slab(scripts):
    arm = Arm()
    slide, _ = scripts

    def lands_on_top(pose, **kw):  # stops 3 mm above the floor and calls it reached
        arm.p = np.array([pose["position"]["x"], pose["position"]["y"], 0.018])
        return {"outcome": "reached", "reached": True}

    ctx = arm.ctx(extra={"robot.move_cartesian_until_contact": lands_on_top})
    out = slide.run(ctx, target_obb=SLAB, edge_point=EDGE, edge_direction=ALONG_Y, mode="press")
    assert out["route"] == "slid", out["detail"]


def test_press_that_meets_nothing_is_unreachable(scripts):
    arm = Arm()
    slide, _ = scripts
    ctx = arm.ctx(
        extra={
            "robot.move_cartesian_until_contact": lambda pose, **kw: (
                setattr(arm, "p", np.array([pose["position"][k] for k in "xyz"]))
                or {"outcome": "reached"}
            )
        }
    )
    out = slide.run(ctx, target_obb=SLAB, edge_point=EDGE, edge_direction=ALONG_Y, mode="press")
    assert out["route"] == "unreachable" and "without meeting" in out["detail"]


def test_a_blocked_drag_is_slid_not_picked(scripts):
    arm = Arm(stop_drag_at=0.40)
    a, b = slide_then_pick(scripts, arm, edge_direction=ALONG_Y)
    assert b is None
    assert a["route"] == "slid_not_picked"
    assert a["slide_achieved"] == pytest.approx(0.40 - 0.347 - 0.003)
    assert "stopped advancing" in a["detail"]


def test_ik_preview_refusal_touches_nothing(scripts):
    arm = Arm()
    slide, _ = scripts
    ctx = arm.ctx(extra={"robot.solve_ik": {"success": True, "within_servo_tolerance": False}})
    out = slide.run(ctx, target_obb=SLAB, edge_point=EDGE, edge_direction=ALONG_Y)
    assert out["route"] == "unreachable" and "IK preview" in out["detail"]
    assert arm.moves == []


def test_an_ik_preview_that_raises_unreachable_refuses_but_other_errors_do_not(scripts):
    slide, _ = scripts

    def raises(msg):
        def solve(**kw):
            raise ToolError("robot.solve_ik", msg)

        return solve

    arm = Arm()
    out = slide.run(
        arm.ctx(extra={"robot.solve_ik": raises("Target pose is unreachable")}),
        target_obb=SLAB,
        edge_point=EDGE,
    )
    assert out["route"] == "unreachable" and "IK preview" in out["detail"]
    assert arm.moves == []
    out = slide.run(
        Arm().ctx(extra={"robot.solve_ik": raises("not served")}), target_obb=SLAB, edge_point=EDGE
    )
    assert out["route"] == "slid"


def test_a_slab_too_short_across_the_edge_is_refused(scripts):
    slide, _ = scripts
    turned = {**SLAB, "extent": {"x": 0.015, "y": 0.03, "z": 0.004}}
    out = slide.run(Arm().ctx(), target_obb=turned, edge_point=EDGE, edge_direction=ALONG_Y)
    assert out["route"] == "unreachable" and "deep across the edge" in out["detail"]


def test_a_slab_wider_than_the_jaws_is_refused(scripts):
    slide, _ = scripts
    wide = {**SLAB, "extent": {"x": 0.03, "y": 0.05, "z": 0.004}}
    out = slide.run(Arm().ctx(), target_obb=wide, edge_point=EDGE, edge_direction=ALONG_Y)
    assert out["route"] == "unreachable" and "wide along the edge" in out["detail"]


def test_too_far_from_the_edge(scripts):
    slide, _ = scripts
    out = slide.run(
        Arm().ctx(),
        target_obb=SLAB,
        edge_point={"x": 0.80, "y": 0.0, "z": 0.0},
        edge_direction=ALONG_Y,
    )
    assert out["route"] == "unreachable" and "over 250" in out["detail"]


def test_a_centre_past_the_edge_is_refused(scripts):
    slide, _ = scripts
    out = slide.run(
        Arm().ctx(),
        target_obb=SLAB,
        edge_point={"x": 0.39, "y": 0.0, "z": 0.0},
        edge_direction=ALONG_Y,
    )
    assert out["route"] == "unreachable"


def test_already_overhanging_skips_the_slide(scripts):
    arm = Arm()
    over = {**SLAB, "center": {"x": 0.495, "y": 0.10, "z": 0.004}}  # overhangs 25 mm already
    a, b = slide_then_pick(scripts, arm, obb=over, edge_direction=ALONG_Y)
    assert a["route"] == "slid" and "nothing to slide" in a["detail"]
    assert a["slide_planned"] == 0.0
    assert b["route"] == "picked"


def test_hover_refused_for_every_rotation(scripts):
    arm = Arm(refuse=lambda tool, pose: "no IK" if tool == "robot.go_to_pose" else None)
    slide, _ = scripts
    out = slide.run(arm.ctx(frame=False), target_obb=SLAB, edge_point=EDGE, edge_direction=ALONG_Y)
    assert out["route"] == "unreachable" and out["detail"].startswith("hover")


def test_a_raised_move_is_routed(scripts):
    arm = Arm()
    slide, _ = scripts

    def boom(pose, **kw):
        raise ToolError("robot.go_to_pose", "joint limit")

    out = slide.run(arm.ctx(extra={"robot.go_to_pose": boom}), target_obb=SLAB, edge_point=EDGE)
    assert out["route"] == "unreachable" and "joint limit" in out["detail"]


def test_jaws_closing_on_nothing_are_slid_not_picked(scripts):
    arm = Arm(held=0.0)
    a, b = slide_then_pick(scripts, arm, edge_direction=ALONG_Y)
    assert a["route"] == "slid"
    assert b["route"] == "slid_not_picked" and "closed on nothing" in b["detail"]
    assert arm.fraction == 1.0  # opened again


def test_a_blade_landing_on_something_backs_up(scripts):
    arm = Arm(grasp_stop=0.018)
    a, b = slide_then_pick(scripts, arm, edge_direction=ALONG_Y)
    assert b["route"] == "slid_not_picked" and "landed" in b["detail"]
    assert arm.p[2] == pytest.approx(0.004 + 0.08)


def test_pick_after_a_failed_slide_does_nothing(scripts):
    _, pick = scripts
    ctx = Arm().ctx()
    out = pick.run(ctx, plan={"grasp_position": EDGE, "rotation": DOWN}, slide_route="unreachable")
    assert out["route"] == "slid_not_picked"
    assert ctx.calls == []


def test_an_unturned_wrist_is_finished_on_the_last_joint(scripts):
    arm = Arm()
    arm.rot = yaw_quat(40.0)
    turns = []

    def go(pose, **kw):  # the solver keeps the hand's roll
        arm.p = np.array([pose["position"][k] for k in "xyz"])
        arm.moves.append(("robot.go_to_pose", arm.p.copy(), pose["rotation"]))
        return {"outcome": "reached"}

    def to_joints(joint_config, **kw):
        q = np.array(joint_config["positions"])
        turns.append(float(q[-1] - arm.q[-1]))
        arm.q = q
        arm.rot = yaw_quat(40.0 + math.degrees(q[-1]))
        return {"outcome": "reached"}

    slide, _ = scripts
    ctx = arm.ctx(extra={"robot.go_to_pose": go, "robot.move_to_joints": to_joints})
    out = slide.run(ctx, target_obb=SLAB, edge_point=EDGE, edge_direction=ALONG_Y)
    assert out["route"] == "slid", out["detail"]
    assert turns, "the last joint was never turned"
    m = slide.C.yaw_between(arm.rot, DOWN)
    assert abs(slide.C.wrap(m, 180.0)) <= 8.0


def test_arm_id_and_speed_scale_pass_only_when_set(scripts):
    arm = Arm()
    slide, _ = scripts
    ctx = arm.ctx()
    slide.run(ctx, target_obb=SLAB, edge_point=EDGE, edge_direction=ALONG_Y)
    assert all("arm_id" not in c.kwargs and "speed_scale" not in c.kwargs for c in ctx.calls)
    arm = Arm()
    ctx = arm.ctx()
    slide.run(
        ctx, target_obb=SLAB, edge_point=EDGE, edge_direction=ALONG_Y, arm_id=1, speed_scale=0.3
    )
    moves = ctx.calls_to("robot.go_to_pose_cartesian")
    assert all(c.kwargs.get("arm_id") == 1 for c in moves)
    assert any(c.kwargs.get("speed_scale") == 0.3 for c in moves)
