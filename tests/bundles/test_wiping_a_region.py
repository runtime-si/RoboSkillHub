"""wiping-a-region: the pass plan, the coverage raster, and the wipe against a fake arm over a table."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "wiping-a-region"
KEEP = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
TOP_A = {"w": 0.0, "x": 0.7071, "y": 0.7071, "z": 0.0}
TOP_B = {"w": 0.0, "x": -0.7071, "y": 0.7071, "z": 0.0}
# A 10 x 8 cm region on a table at z = 0, long side along x.
REGION = {
    "center": {"x": 0.40, "y": 0.10, "z": 0.0005},
    "extent": {"x": 0.05, "y": 0.04, "z": 0.0005},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}


def yawed(region, yaw):
    q = {"w": math.cos(yaw / 2), "x": 0.0, "y": 0.0, "z": math.sin(yaw / 2)}
    return {**region, "orientation": q}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["wipe_region"].module


class Arm:
    """Moves land where commanded unless told otherwise; a contact move stops on a
    surface at ``surface`` (tool centre ``tip`` above it) or on ``bump`` regions."""

    def __init__(
        self,
        start=(0.30, 0.0, 0.25),
        surface=0.0,
        tip=0.0,
        refuse=None,
        ik=None,
        no_contact_tool=False,
        contact_raises=False,
    ):
        self.p = np.array(start, dtype=float)
        self.rot = KEEP
        self.surface = surface
        self.tip = tip
        self.refuse = refuse or (lambda tool, target: None)
        self.ik = ik or (lambda target: {"success": True, "within_servo_tolerance": True})
        self.no_contact_tool = no_contact_tool
        self.contact_raises = contact_raises
        self.moves = []
        self.closed = False

    def _go(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy(), pose["rotation"]))
            why = self.refuse(tool, target)
            if why == "raise":
                raise ToolError(tool, "the station refused the move")
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            self.p, self.rot = target, pose["rotation"]
            return {"reached": True, "outcome": "reached"}

        return go

    def contact(self, pose, **kwargs):
        if self.no_contact_tool:
            raise ToolError("robot.move_cartesian_until_contact", "unknown tool")
        target = np.array([pose["position"][k] for k in "xyz"])
        self.moves.append(("contact", target.copy(), pose["rotation"]))
        floor = self.surface_at(target[:2]) + self.tip
        if target[2] < floor:
            self.p = np.array([target[0], target[1], floor])
            if self.contact_raises:
                raise ToolError("robot.move_cartesian_until_contact", "did not reach its target")
            return {"reached": False, "outcome": "contact"}
        self.p = target
        return {"reached": True, "outcome": "reached"}

    def surface_at(self, xy):
        return self.surface(xy) if callable(self.surface) else self.surface

    def ee(self, **kwargs):
        return {
            "pose": {
                "position": dict(zip("xyz", map(float, self.p), strict=True)),
                "rotation": self.rot,
            }
        }

    def ctx(self, frame=None):
        def grasp_frame(**kwargs):
            if frame == "raise":
                raise ToolError("robot.grasp_frame", "no frame")
            return frame or {"rotation": TOP_A, "rotation_flipped": TOP_B}

        def close(**kwargs):
            self.closed = True
            return {}

        return FakeContext(
            {
                "robot.get_ee_pose": self.ee,
                "robot.go_to_pose": self._go("go_to_pose"),
                "robot.go_to_pose_cartesian": self._go("cartesian"),
                "robot.move_cartesian_until_contact": self.contact,
                "robot.solve_ik": lambda pose, **kw: self.ik(
                    np.array([pose["position"][k] for k in "xyz"])
                ),
                "robot.close_gripper": close,
                "robot.grasp_frame": grasp_frame,
            }
        )

    def strokes(self):
        return [m[1] for m in self.moves if m[0] == "cartesian" and abs(m[1][2] - 0.015) < 1e-6]


# --- the plan ---------------------------------------------------------------


def test_passes_alternate_and_keep_the_tool_inside_the_region(script):
    passes = script.plan_passes(0.05, 0.04, 0.03, 0.05)
    assert [round(off, 4) for off, _ in passes] == [-0.025, 0.0, 0.025]
    assert passes[0][1] == pytest.approx([-0.035, 0.0, 0.035])
    assert passes[1][1] == pytest.approx([0.035, 0.0, -0.035])
    for _, along in passes:
        assert all(abs(a) <= 0.05 - 0.015 + 1e-9 for a in along)


def test_pass_spacing_never_exceeds_step_or_tool_width(script):
    for half_v, width, step in [(0.04, 0.03, 0.05), (0.10, 0.08, 0.05), (0.2, 0.05, 0.05)]:
        offsets = [off for off, _ in script.plan_passes(0.1, half_v, width, step)]
        gaps = np.diff(offsets)
        assert gaps.max() <= min(width, step) + 1e-9
        assert offsets[0] == pytest.approx(-(half_v - width / 2))
        assert offsets[-1] == pytest.approx(half_v - width / 2)


def test_waypoints_are_at_most_a_step_apart(script):
    _, along = script.plan_passes(0.20, 0.01, 0.03, 0.05)[0]
    assert np.diff(along).max() <= 0.05 + 1e-9
    assert along[0] == pytest.approx(-0.185) and along[-1] == pytest.approx(0.185)


def test_a_region_narrower_than_the_tool_gets_one_pass_down_its_middle(script):
    passes = script.plan_passes(0.01, 0.01, 0.05, 0.05)
    assert len(passes) == 1
    assert passes[0][0] == 0.0 and passes[0][1] == [0.0, 0.0]


def test_the_region_frame_takes_the_long_axis_and_the_bottom(script):
    centre, u, v, hu, hv, bottom = script.region_frame(yawed(REGION, math.pi / 2))
    assert np.allclose(centre, [0.40, 0.10])
    assert np.allclose(np.abs(u), [0.0, 1.0], atol=1e-9)  # the 10 cm side, turned onto y
    assert hu == pytest.approx(0.05) and hv == pytest.approx(0.04)
    assert bottom == pytest.approx(0.0)
    tall = {**REGION, "extent": {"x": 0.02, "y": 0.06, "z": 0.003}}
    _, u, _, hu, hv, bottom = script.region_frame(tall)
    assert np.allclose(np.abs(u), [0.0, 1.0]) and hu == pytest.approx(0.06)
    assert bottom == pytest.approx(0.0005 - 0.003)


def test_coverage_counts_the_square_swath(script):
    centre, u, v = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    full = [np.array([[-0.035, y, 0.0], [0.035, y, 0.0]]) for y in (-0.025, 0.0, 0.025)]
    assert script.coverage(full, centre, u, v, 0.05, 0.04, 0.03) == pytest.approx(1.0)
    one = full[:1]
    assert script.coverage(one, centre, u, v, 0.05, 0.04, 0.03) == pytest.approx(0.375, abs=0.01)
    assert script.coverage([], centre, u, v, 0.05, 0.04, 0.03) == 0.0


# --- the wipe ---------------------------------------------------------------


def test_happy_path_wipes_the_whole_region_and_lifts(script):
    arm = Arm()
    out = script.run(arm.ctx(), region_obb=REGION, tool_width=0.03)
    assert out["route"] == "wiped", out["detail"]
    assert out["coverage"] == pytest.approx(1.0)
    assert out["passes_planned"] == out["passes_done"] == 3
    assert out["min_height"] == pytest.approx(0.015) and out["max_height"] == pytest.approx(0.015)
    assert arm.closed
    # Every stroke stays inside the region, inset by half the tool.
    for p in arm.strokes():
        assert 0.40 - 0.035 - 1e-9 <= p[0] <= 0.40 + 0.035 + 1e-9
        assert 0.10 - 0.025 - 1e-9 <= p[1] <= 0.10 + 0.025 + 1e-9
    # The top-down, fingers-across rotation was used, and the hand ends lifted.
    assert all(m[2] == TOP_A for m in arm.moves)
    assert arm.p[2] == pytest.approx(0.115)
    phases = [p["phase"] for p in out["path"]]
    assert phases.count("lift") == 3 and phases.count("stroke") == 6


def test_grasp_frame_is_asked_for_fingers_across_the_strokes(script):
    arm = Arm()
    seen = {}

    def frame(**kw):
        seen.update(kw)
        return {"rotation": TOP_A}

    ctx = arm.ctx()
    ctx._responses["robot.grasp_frame"] = frame
    script.run(ctx, region_obb=yawed(REGION, math.radians(30)), tool_width=0.03)
    assert seen["close_heading_deg"] % 180 == pytest.approx(120.0)


def test_contact_height_is_floored(script):
    arm = Arm()
    out = script.run(arm.ctx(), region_obb=REGION, tool_width=0.03, contact_height=0.002)
    assert out["wipe_height"] == pytest.approx(0.015)
    assert min(m[1][2] for m in arm.moves) == pytest.approx(0.015)


def test_a_surface_higher_than_estimated_is_met_and_wiped_where_it_is(script):
    # The true surface is 8 mm above the box's bottom; the held tool reaches 15 mm
    # below the tool centre, so the descent stops 8 mm high.
    arm = Arm(surface=0.008, tip=0.015)
    out = script.run(arm.ctx(), region_obb=REGION, tool_width=0.03)
    assert out["route"] == "wiped", out["detail"]
    assert out["min_height"] == pytest.approx(0.023)
    strokes = [m[1] for m in arm.moves if m[0] == "cartesian" and m[1][2] < 0.05]
    assert strokes and all(p[2] == pytest.approx(0.023) for p in strokes)


def test_an_object_in_a_pass_skips_it_and_reports_partial(script):
    # Something 40 mm tall under the middle pass's start.
    def surface(xy):
        return 0.04 if abs(xy[1] - 0.10) < 0.005 else 0.0

    arm = Arm(surface=surface, tip=0.015)
    out = script.run(arm.ctx(), region_obb=REGION, tool_width=0.03)
    assert out["route"] == "partial"
    assert out["passes_done"] == 2
    assert "pass 1: descent stopped" in out["detail"]
    assert out["coverage"] == pytest.approx(0.75, abs=0.02)


def test_a_contact_stop_that_raises_is_read_from_the_pose(script):
    arm = Arm(surface=0.004, tip=0.015, contact_raises=True)
    out = script.run(arm.ctx(), region_obb=REGION, tool_width=0.03)
    assert out["route"] == "wiped", out["detail"]
    assert out["min_height"] == pytest.approx(0.015)  # stopped 4 mm high: inside the tolerance


def test_a_connector_without_the_contact_move_descends_the_ordinary_way(script):
    arm = Arm(no_contact_tool=True)
    out = script.run(arm.ctx(), region_obb=REGION, tool_width=0.03)
    assert out["route"] == "wiped", out["detail"]
    assert not any(m[0] == "contact" for m in arm.moves)


def test_probe_surface_false_never_calls_the_contact_move(script):
    arm = Arm()
    out = script.run(arm.ctx(), region_obb=REGION, tool_width=0.03, probe_surface=False)
    assert out["route"] == "wiped"
    assert not any(m[0] == "contact" for m in arm.moves)


def test_passes_out_of_reach_are_skipped(script):
    arm = Arm(ik=lambda t: {"success": True, "within_servo_tolerance": t[1] < 0.11})
    out = script.run(arm.ctx(), region_obb=REGION, tool_width=0.03)
    assert out["route"] == "partial"
    assert out["passes_done"] == 2
    assert "pass 2: out of reach" in out["detail"]


def test_every_pass_out_of_reach_is_unreachable(script):
    arm = Arm(ik=lambda t: {"success": False})
    out = script.run(arm.ctx(), region_obb=REGION, tool_width=0.03)
    assert out["route"] == "unreachable"
    assert out["passes_done"] == 0 and out["coverage"] == 0.0


def test_check_reach_false_ignores_the_ik(script):
    arm = Arm(ik=lambda t: {"success": False})
    out = script.run(arm.ctx(), region_obb=REGION, tool_width=0.03, check_reach=False)
    assert out["route"] == "wiped"


def test_a_refused_hover_falls_back_to_the_next_rotation(script):
    arm = Arm()
    arm.refuse = lambda tool, t: "no IK" if tool == "go_to_pose" and len(arm.moves) == 1 else None
    out = script.run(arm.ctx(), region_obb=REGION, tool_width=0.03)
    assert out["route"] == "wiped"
    assert arm.moves[0][2] == TOP_A
    assert all(m[2] == TOP_B for m in arm.moves[1:])


def test_every_rotation_refused_is_unreachable(script):
    arm = Arm(refuse=lambda tool, t: "raise" if tool == "go_to_pose" else None)
    out = script.run(arm.ctx(), region_obb=REGION, tool_width=0.03)
    assert out["route"] == "unreachable"
    assert out["detail"].startswith("hover:")


def test_without_a_grasp_frame_the_hand_keeps_its_rotation(script):
    arm = Arm()
    out = script.run(arm.ctx(frame="raise"), region_obb=REGION, tool_width=0.03)
    assert out["route"] == "wiped"
    assert all(m[2] == KEEP for m in arm.moves)


def test_a_stroke_refused_part_way_counts_what_was_swept(script):
    def refuse(tool, t):
        return "unreachable" if tool == "cartesian" and t[2] < 0.05 and t[0] > 0.43 else None

    arm = Arm(refuse=refuse)
    out = script.run(arm.ctx(), region_obb=REGION, tool_width=0.03)
    assert out["route"] == "partial"
    assert out["passes_done"] == 3
    assert "stroke stopped" in out["detail"]
    assert 0.5 < out["coverage"] < 0.95


def test_a_refused_lift_is_partial_even_with_full_coverage(script):
    def refuse(tool, t):
        return (
            "raise" if tool == "cartesian" and t[2] > 0.1 and t[0] > 0.43 and t[1] > 0.12 else None
        )

    arm = Arm(refuse=refuse)
    out = script.run(arm.ctx(), region_obb=REGION, tool_width=0.03)
    assert out["route"] == "partial"
    assert out["coverage"] == pytest.approx(1.0)
    assert "final lift" in out["detail"]


def test_an_explicit_surface_z_sets_the_wiping_height(script):
    arm = Arm()
    out = script.run(arm.ctx(), region_obb=REGION, surface_z=0.01, tool_width=0.03)
    assert out["surface_z"] == pytest.approx(0.01)
    assert min(m[1][2] for m in arm.moves) == pytest.approx(0.025)


def test_close_hand_false_leaves_the_gripper(script):
    arm = Arm()
    script.run(arm.ctx(), region_obb=REGION, tool_width=0.03, close_hand=False)
    assert not arm.closed
