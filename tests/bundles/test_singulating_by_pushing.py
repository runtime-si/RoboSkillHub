"""singulating-by-pushing: FakeContext tests of the clearance geometry, the push choice and the routing."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "singulating-by-pushing"
KEEP = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
TOP_A = {"w": 0.0, "x": 0.7071, "y": 0.7071, "z": 0.0}
TOP_B = {"w": 0.0, "x": -0.7071, "y": 0.7071, "z": 0.0}
YAM = {"span_m": 0.094, "finger": {"stated": True, "pad_thickness_m": 0.0}}


def cube(x, y, yaw_deg=0.0, half=0.015):
    yaw = math.radians(yaw_deg)
    return {
        "center": {"x": x, "y": y, "z": half},
        "extent": {"x": half, "y": half, "z": half},
        "orientation": {"w": math.cos(yaw / 2), "x": 0.0, "y": 0.0, "z": math.sin(yaw / 2)},
    }


# A 30 mm cube on a table at z = 0, centre (0.40, 0.10).
TARGET = cube(0.40, 0.10)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["singulate_by_pushing"].module


class Arm:
    """Moves land where commanded unless ``refuse`` (a predicate on the call) or ``short`` say otherwise."""

    def __init__(self, start=(0.30, 0.0, 0.25), refuse=None, short=None, gripper=YAM):
        self.p = np.array(start, dtype=float)
        self.rot = KEEP
        self.refuse = refuse or (lambda tool, pose: None)
        self.short = short  # fraction of a horizontal cartesian move actually made
        self.gripper = gripper
        self.moves = []

    def _move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy(), pose["rotation"]))
            why = self.refuse(tool, pose)
            if isinstance(why, Exception):
                raise why
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            horizontal = (
                np.linalg.norm((target - self.p)[:2]) > 1e-3 and abs(target[2] - self.p[2]) < 1e-6
            )
            if self.short is not None and tool.endswith("cartesian") and horizontal:
                target = self.p + (target - self.p) * self.short
            self.p, self.rot = target, pose["rotation"]
            return {"reached": True, "outcome": "reached"}

        return go

    def ctx(self):
        return FakeContext(
            {
                "robot.describe_gripper": self.gripper,
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
                "robot.grasp_frame": {"rotation": TOP_A, "rotation_flipped": TOP_B},
            }
        )

    def pushes(self):
        """(lowered, end) of each push: the descent target and the horizontal move after it."""
        cart = [m[1] for m in self.moves if m[0] == "robot.go_to_pose_cartesian"]
        return [(cart[i], cart[i + 1]) for i in range(0, len(cart) - 2, 3)]


# ------------------------------------------------------------------ geometry


def test_signed_distance_and_footprint(script):
    a = script.footprint(cube(0.0, 0.0))
    assert len(a) == 4
    assert script.signed_distance(a, script.footprint(cube(0.030, 0.0))) == pytest.approx(
        0.0, abs=1e-9
    )
    assert script.signed_distance(a, script.footprint(cube(0.035, 0.0))) == pytest.approx(0.005)
    assert script.signed_distance(a, script.footprint(cube(0.020, 0.0))) == pytest.approx(-0.010)
    diag = script.footprint(cube(0.0, 0.0, yaw_deg=45))
    assert np.ptp(diag[:, 0]) == pytest.approx(0.030 * math.sqrt(2))
    seg = np.array([[0.05, -0.05], [0.05, 0.05]])
    assert script.signed_distance(seg, a) == pytest.approx(0.035)


def test_free_space_heading_points_away_from_the_neighbours(script):
    c = np.array([0.40, 0.10])
    one = [script.footprint(cube(0.43, 0.10))]
    assert math.degrees(script.free_space_heading(c, one, None)) == pytest.approx(180.0, abs=3.0)
    two = [script.footprint(cube(0.43, 0.10)), script.footprint(cube(0.40, 0.13))]
    assert math.degrees(script.free_space_heading(c, two, None)) % 360 == pytest.approx(
        225.0, abs=8.0
    )


# ------------------------------------------------------------------ routes


def test_already_free_target_is_singulated_without_moving(script):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(ctx, target_obb=TARGET, neighbour_obb=cube(0.50, 0.10))
    assert out["route"] == "singulated"
    assert out["pushes"] == 0 and out["fits_before"] and out["fits_after"]
    assert out["gap_before"] == pytest.approx(0.070)
    assert arm.moves == []
    assert "robot.close_gripper" not in [c.tool for c in ctx.calls]


def test_no_neighbours_is_singulated(script):
    out = script.run(Arm().ctx(), target_obb=TARGET, neighbour_obb=None)
    assert out["route"] == "singulated" and out["pushes"] == 0


def test_five_mm_gap_is_pushed_away_from_a_free_side(script):
    arm = Arm()
    ctx = arm.ctx()
    neighbour = cube(0.435, 0.10)
    out = script.run(ctx, target_obb=TARGET, neighbour_obb=neighbour)
    assert out["route"] == "singulated", out["detail"]
    assert out["pushes"] == 1
    assert out["gap_before"] == pytest.approx(0.005)
    assert not out["fits_before"] and out["fits_after"]
    assert out["gap_after"] > out["gap_before"] + 0.01
    assert out["clearance_after"] >= 0.007 - 1e-9
    ((lowered, end),) = arm.pushes()
    d = (end - lowered)[:2] / np.linalg.norm((end - lowered)[:2])
    assert d[0] < 0.0  # away from the +x neighbour
    # The hand came down where it keeps 3 mm from the neighbour (10 mm hand radius).
    seg = np.array([lowered[:2], end[:2]])
    assert script.signed_distance(seg, script.footprint(neighbour)) >= 0.013 - 1e-9
    assert lowered[2] == pytest.approx(0.015)  # half of 30 mm, the 15 mm floor
    names = [m[0] for m in arm.moves]
    assert names == ["robot.go_to_pose"] + ["robot.go_to_pose_cartesian"] * 3
    assert all(m[2] == TOP_A for m in arm.moves)
    calls = [c.tool for c in ctx.calls]
    assert calls.index("robot.close_gripper") < calls.index("robot.go_to_pose")
    assert calls[-1] == "robot.open_gripper"
    moved = out["expected_target_obb"]["center"]
    assert moved["x"] < 0.40


def test_touching_neighbour_is_not_pushed_from_inside_it(script):
    arm = Arm()
    out = script.run(arm.ctx(), target_obb=TARGET, neighbour_obb=cube(0.43, 0.10))
    assert out["route"] == "singulated", out["detail"]
    assert out["gap_before"] == pytest.approx(0.0, abs=1e-9)
    lowered, end = arm.pushes()[0]
    # Straight away (-x) would put the hand inside the neighbour: it comes down on a free side.
    assert lowered[0] < 0.43 - 0.015 - 0.013 + 1e-9
    assert np.linalg.norm((end - lowered)[:2]) <= 0.02 + 0.003 + 0.015 + 0.04 + 1e-6


def test_boxed_in_corner_is_unreachable_without_moving(script):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(
        ctx, target_obb=TARGET, neighbour_obb=cube(0.43, 0.10), neighbour_obbs=[cube(0.40, 0.13)]
    )
    assert out["route"] == "unreachable"
    assert "no direction" in out["detail"]
    assert arm.moves == []
    assert "robot.close_gripper" not in [c.tool for c in ctx.calls]


def test_too_wide_target_is_unreachable(script):
    wide = cube(0.40, 0.10, half=0.05)
    out = script.run(Arm().ctx(), target_obb=wide, neighbour_obb=cube(0.47, 0.10))
    assert out["route"] == "unreachable" and "wider" in out["detail"]


def test_a_blocked_push_is_still_cluttered(script):
    arm = Arm(short=0.2)
    out = script.run(arm.ctx(), target_obb=TARGET, neighbour_obb=cube(0.435, 0.10))
    assert out["route"] == "still_cluttered"
    assert out["pushes"] == 1 and "travelled" in out["detail"]
    assert not out["fits_after"]


def test_pushes_run_out(script):
    out = script.run(
        Arm().ctx(),
        target_obb=TARGET,
        neighbour_obb=cube(0.435, 0.10),
        push_step=0.012,
        max_pushes=1,
    )
    assert out["route"] == "still_cluttered"
    assert out["pushes"] == 1 and "still do not fit" in out["detail"]
    assert out["clearance_after"] > out["clearance_before"]


def test_short_pushes_repeat_until_free(script):
    arm = Arm()
    out = script.run(arm.ctx(), target_obb=TARGET, neighbour_obb=cube(0.435, 0.10), push_step=0.016)
    assert out["route"] == "singulated", out["detail"]
    assert out["pushes"] >= 2
    assert all(
        np.linalg.norm((e - lw)[:2]) <= 0.02 + 0.003 + 0.0152 + 0.016 + 1e-6
        for lw, e in arm.pushes()
    )


def test_refused_hover_is_unreachable_and_opens(script):
    arm = Arm(refuse=lambda tool, pose: "joint limit" if tool == "robot.go_to_pose" else None)
    ctx = arm.ctx()
    out = script.run(ctx, target_obb=TARGET, neighbour_obb=cube(0.435, 0.10))
    assert out["route"] == "unreachable" and "hover" in out["detail"]
    assert [m[2] for m in arm.moves] == [TOP_A, TOP_B, KEEP]
    assert out["pushes"] == 0
    assert [c.tool for c in ctx.calls][-1] == "robot.open_gripper"


def test_raised_descent_is_routed(script):
    def refuse(tool, pose):
        if tool.endswith("cartesian") and pose["position"]["z"] < 0.05:
            return ToolError("robot.go_to_pose_cartesian", "IK failed")
        return None

    out = script.run(Arm(refuse=refuse).ctx(), target_obb=TARGET, neighbour_obb=cube(0.435, 0.10))
    assert out["route"] == "unreachable" and "descent" in out["detail"]


def test_a_chosen_grasp_heading_along_the_contact_already_fits(script):
    out = script.run(
        Arm().ctx(), target_obb=TARGET, neighbour_obb=cube(0.435, 0.10), grasp_heading_deg=90.0
    )
    assert out["route"] == "singulated" and out["pushes"] == 0


def test_any_axis_is_looser_than_both(script):
    out = script.run(Arm().ctx(), target_obb=TARGET, neighbour_obb=cube(0.435, 0.10), need="any")
    assert out["route"] == "singulated" and out["pushes"] == 0


def test_opening_comes_from_describe_gripper(script):
    small = {"width_fit": {"at_open_m": 0.05}, "finger": {"stated": True, "pad_thickness_m": 0.006}}
    out = script.run(Arm(gripper=small).ctx(), target_obb=TARGET, neighbour_obb=cube(0.50, 0.10))
    assert out["required_gap"] == pytest.approx((0.05 - 0.03) / 2 + 0.006 + 0.003)


def test_describe_gripper_missing_falls_back_to_the_yam(script):
    def missing(**kw):
        raise ToolError("robot.describe_gripper", "not served")

    arm = Arm()
    arm.gripper = missing
    out = script.run(arm.ctx(), target_obb=TARGET, neighbour_obb=cube(0.50, 0.10))
    assert out["required_gap"] == pytest.approx((0.094 - 0.03) / 2 + 0.010 + 0.003)


def test_workspace_keeps_the_target_inside(script):
    ws = {"x_min": 0.30, "x_max": 0.60, "y_min": 0.0, "y_max": 0.1155}
    arm = Arm()
    out = script.run(arm.ctx(), target_obb=TARGET, neighbour_obb=cube(0.435, 0.10), workspace=ws)
    assert out["route"] in ("singulated", "still_cluttered")
    centre = out["expected_target_obb"]["center"]
    assert centre["y"] + 0.015 <= ws["y_max"] + 1e-9
