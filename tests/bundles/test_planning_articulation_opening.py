"""planning-articulation-opening: the drawer and door waypoint geometry, the IK chain and the routing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "planning-articulation-opening"
# A hand approaching a cabinet front along +x, fingers horizontal.
FRONT = {"w": 0.7071068, "x": 0.0, "y": 0.7071068, "z": 0.0}
HANDLE = {"position": {"x": 0.40, "y": 0.05, "z": 0.12}, "rotation": FRONT}
TOWARD_ROBOT = {"x": -1.0, "y": 0.0, "z": 0.0}
UP = {"x": 0.0, "y": 0.0, "z": 1.0}
HINGE = {"x": 0.40, "y": 0.17, "z": 0.0}  # 12 cm to the handle's left, axis vertical


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["plan_articulation_opening"].module


def poses_of(out):
    return [w["pose"] for w in out["all_poses"]["waypoints"]]


def xyz(v):
    return np.array([v["x"], v["y"], v["z"]], dtype=float)


def rot(q):
    w, x, y, z = (q[k] for k in ("w", "x", "y", "z"))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def axis_angle(a, phi):
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(phi) * k + (1 - math.cos(phi)) * (k @ k)


class Arm:
    """A fake IK whose joints follow the pose smoothly; tells it which waypoints fail and how."""

    def __init__(self, fail=None, flip_at=None, takes_seed=True, no_joints=False):
        self.fail = fail or {}  # index -> "raise" | "success_false" | "servo" | "far"
        self.flip_at = flip_at
        self.takes_seed = takes_seed
        self.no_joints = no_joints
        self.calls = 0
        self.index = 0
        self.poses = []

    def solve(self, pose, **kw):
        if "seed_joints" in kw and not self.takes_seed:
            raise ToolError("robot.solve_ik", "unexpected keyword argument 'seed_joints'")
        p = xyz(pose["position"])
        i = self._index_of(p)
        self.calls += 1
        how = self.fail.get(i)
        if how == "raise":
            raise ToolError("robot.solve_ik", "Target pose is unreachable")
        if how == "success_false":
            return {"success": False, "joint_config": {"positions": [0.0] * 7}}
        q = [*(5.0 * p), *(pose["rotation"][k] for k in ("w", "x", "y", "z"))]
        if i == self.flip_at:
            q[-1] += math.pi
        reply = {"success": True, "joint_config": {"positions": q}}
        if how == "servo":
            reply["within_servo_tolerance"] = False
        if how == "far":
            reply["position_error_m"] = 0.05
        if self.no_joints:
            reply = {"success": True}
        return reply

    def _index_of(self, p):
        for i, seen in enumerate(self.poses):
            if np.allclose(seen, p, atol=1e-9):
                return i
        self.poses.append(p)
        return len(self.poses) - 1

    def ctx(self):
        return FakeContext({"robot.solve_ik": self.solve})


def drawer(script, arm=None, **kw):
    arm = arm or Arm()
    ctx = arm.ctx()
    args = {"pull_direction": TOWARD_ROBOT, "distance": 0.15, **kw}
    return script.run(ctx, handle_pose=HANDLE, joint_type="prismatic", **args), ctx


def door(script, arm=None, **kw):
    arm = arm or Arm()
    ctx = arm.ctx()
    args = {"hinge_point": HINGE, "hinge_axis": UP, "angle_deg": 60.0, **kw}
    return script.run(ctx, handle_pose=HANDLE, joint_type="revolute", **args), ctx


# ---------------------------------------------------------------- geometry


def test_drawer_pull_is_a_straight_evenly_spaced_line_with_the_rotation_held(script):
    out, _ = drawer(script, pull_direction={"x": -3.0, "y": 0.0, "z": 0.0})
    poses = poses_of(out)
    assert len(poses) == 6  # the grasp plus ceil(150 / 35) = 5 steps
    pts = np.array([xyz(p["position"]) for p in poses])
    steps = np.diff(pts, axis=0)
    np.testing.assert_allclose(steps, np.tile([-0.03, 0.0, 0.0], (5, 1)), atol=1e-12)
    for p in poses:
        assert p["rotation"] == FRONT
    assert out["planned_opening"] == pytest.approx(0.15)
    assert out["radius"] == 0.0


def test_mosart_defaults_give_its_waypoint_counts(script):
    arm = Arm()
    out = script.run(
        arm.ctx(), handle_pose=HANDLE, joint_type="prismatic", pull_direction=TOWARD_ROBOT
    )
    assert len(poses_of(out)) == 11 and out["planned_opening"] == pytest.approx(0.35)
    out = script.run(
        Arm().ctx(), handle_pose=HANDLE, joint_type="revolute", hinge_point=HINGE, hinge_axis=UP
    )
    assert len(poses_of(out)) == 10 and out["planned_opening"] == pytest.approx(math.pi / 2)


def test_num_waypoints_overrides_the_step_size(script):
    out, _ = drawer(script, num_waypoints=3)
    assert len(poses_of(out)) == 4


def test_door_arc_keeps_its_radius_and_its_height(script):
    out, _ = door(script)
    poses = poses_of(out)
    assert len(poses) == 7  # 60 deg in 10 deg steps
    c = xyz(HINGE)
    for p in poses:
        q = xyz(p["position"])
        assert math.hypot(*(q - c)[:2]) == pytest.approx(0.12, abs=1e-9)
        assert q[2] == pytest.approx(0.12, abs=1e-12)
    assert out["radius"] == pytest.approx(0.12)


def test_door_arc_keeps_the_hand_fixed_in_the_door_frame(script):
    out, _ = door(script)
    c, p0, r0 = xyz(HINGE), xyz(HANDLE["position"]), rot(FRONT)
    angles = out_angles(out)
    for p, phi in zip(poses_of(out), angles, strict=True):
        door_rot = axis_angle([0, 0, 1], phi)
        np.testing.assert_allclose(door_rot.T @ (xyz(p["position"]) - c), p0 - c, atol=1e-9)
        np.testing.assert_allclose(door_rot.T @ rot(p["rotation"]), r0, atol=1e-6)


def out_angles(out):
    """Signed opening of each waypoint about +z, read back from the positions."""
    c = xyz(HINGE)
    lever0 = (xyz(HANDLE["position"]) - c)[:2]
    angles = []
    for p in poses_of(out):
        lever = (xyz(p["position"]) - c)[:2]
        cross = lever0[0] * lever[1] - lever0[1] * lever[0]
        angles.append(math.atan2(cross, float(np.dot(lever0, lever))))
    return angles


def test_door_opening_is_monotone_and_ends_at_the_asked_angle(script):
    out, _ = door(script)
    angles = out_angles(out)
    assert angles[0] == pytest.approx(0.0)
    assert all(b > a for a, b in zip(angles, angles[1:], strict=False))
    assert angles[-1] == pytest.approx(math.radians(60))
    np.testing.assert_allclose(np.diff(angles), math.radians(10), atol=1e-9)


def test_drawer_opening_is_monotone(script):
    out, _ = drawer(script)
    d = np.array([-1.0, 0.0, 0.0])
    along = [float(np.dot(xyz(p["position"]) - xyz(HANDLE["position"]), d)) for p in poses_of(out)]
    assert along[0] == 0.0 and all(b > a for a, b in zip(along, along[1:], strict=False))


def test_without_a_hint_the_door_turns_right_handed_about_the_axis(script):
    out, _ = door(script)
    # Handle at -y from a +z hinge: a positive turn swings it toward +x.
    assert poses_of(out)[1]["position"]["x"] > HANDLE["position"]["x"]
    out, _ = door(script, hinge_axis={"x": 0.0, "y": 0.0, "z": -1.0})
    assert poses_of(out)[1]["position"]["x"] < HANDLE["position"]["x"]


def test_pull_direction_picks_the_hinge_sign(script):
    for axis in (UP, {"x": 0.0, "y": 0.0, "z": -1.0}):
        out, _ = door(script, hinge_axis=axis, pull_direction=TOWARD_ROBOT)
        first = xyz(poses_of(out)[1]["position"]) - xyz(HANDLE["position"])
        assert first[0] < 0  # the handle starts toward the robot either way
        assert poses_of(out)[-1]["position"]["x"] == pytest.approx(
            0.40 - 0.12 * math.sin(math.radians(60))
        )


def test_hinge_point_anywhere_on_the_line_gives_the_same_arc(script):
    a, _ = door(script)
    b, _ = door(script, hinge_point={"x": 0.40, "y": 0.17, "z": 0.55})
    for p, q in zip(poses_of(a), poses_of(b), strict=True):
        np.testing.assert_allclose(xyz(p["position"]), xyz(q["position"]), atol=1e-12)


def test_a_tilted_axis_lid_keeps_radius_and_relation(script):
    axis = np.array([0.0, 1.0, 0.3])
    hinge = {"x": 0.50, "y": 0.0, "z": 0.10}
    out = script.run(
        Arm().ctx(),
        handle_pose=HANDLE,
        joint_type="revolute",
        hinge_point=hinge,
        hinge_axis=dict(zip("xyz", axis, strict=True)),
        angle_deg=45.0,
    )
    a = axis / np.linalg.norm(axis)
    c, p0 = xyz(hinge), xyz(HANDLE["position"])
    lever0 = (p0 - c) - a * np.dot(p0 - c, a)
    for p in poses_of(out):
        v = xyz(p["position"]) - c
        assert np.dot(v, a) == pytest.approx(np.dot(p0 - c, a), abs=1e-9)
        assert np.linalg.norm(v - a * np.dot(v, a)) == pytest.approx(
            np.linalg.norm(lever0), abs=1e-9
        )
    assert out["radius"] == pytest.approx(np.linalg.norm(lever0))


def test_quaternions_stay_in_one_hemisphere(script):
    out = script.run(
        Arm().ctx(),
        handle_pose=HANDLE,
        joint_type="revolute",
        hinge_point=HINGE,
        hinge_axis=UP,
        angle_deg=170.0,
    )
    qs = [np.array([p["rotation"][k] for k in "wxyz"]) for p in poses_of(out)]
    assert all(float(np.dot(a, b)) > 0 for a, b in zip(qs, qs[1:], strict=False))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"joint_type": "prismatic"},
        {"joint_type": "prismatic", "pull_direction": {"x": 0.0, "y": 0.0, "z": 0.0}},
        {"joint_type": "prismatic", "pull_direction": TOWARD_ROBOT, "distance": 0.0},
        {"joint_type": "revolute", "hinge_axis": UP},
        {
            "joint_type": "revolute",
            "hinge_point": {"x": 0.40, "y": 0.05, "z": 0.0},
            "hinge_axis": UP,
        },
        {"joint_type": "revolute", "hinge_point": HINGE, "hinge_axis": UP, "angle_deg": -10},
        {"joint_type": "screw"},
    ],
)
def test_malformed_articulations_raise(script, kwargs):
    with pytest.raises(ValueError):
        script.run(Arm().ctx(), handle_pose=HANDLE, **kwargs)


# ---------------------------------------------------------------- IK chain and routing


def test_all_reachable_is_planned_with_poses_and_joints(script):
    out, ctx = drawer(script)
    assert out["route"] == "planned"
    assert out["reachable"] == [True] * 6 and out["reachable_fraction"] == 1.0
    assert len(out["waypoints"]["waypoints"]) == 6
    assert out["waypoints"]["waypoints"][0]["pose"] == poses_of(out)[0]
    assert len(out["joint_trajectory"]["waypoints"]) == 6
    assert out["opening"] == pytest.approx(0.15)
    assert ctx.call_count("robot.solve_ik") == 6
    assert all("arm_id" not in c.kwargs for c in ctx.calls)


def test_door_all_reachable_is_planned(script):
    out, _ = door(script)
    assert out["route"] == "planned" and out["opening"] == pytest.approx(math.radians(60))
    assert "arc radius 120 mm" in out["detail"]


@pytest.mark.parametrize("how", ["raise", "success_false", "servo", "far"])
def test_a_failed_waypoint_cuts_the_plan_to_its_prefix(script, how):
    out, ctx = drawer(script, arm=Arm(fail={3: how}))
    assert out["route"] == "unreachable"
    assert out["reachable"] == [True, True, True, False, True, True]
    assert len(out["waypoints"]["waypoints"]) == 3
    assert len(out["joint_trajectory"]["waypoints"]) == 3
    assert out["opening"] == pytest.approx(0.06)
    assert out["reachable_fraction"] == pytest.approx(4 / 5)
    assert "waypoint 3" in out["detail"]
    # MOSART keeps solving past a failure, so the graph sees the whole mask.
    assert ctx.call_count("robot.solve_ik") == 6


def test_an_unreachable_grasp_fails_the_whole_plan(script):
    out, ctx = door(script, arm=Arm(fail={0: "raise"}))
    assert out["route"] == "unreachable"
    assert out["reachable"] == [False] * 7
    assert out["waypoints"]["waypoints"] == [] and out["joint_trajectory"]["waypoints"] == []
    assert out["opening"] == 0.0 and out["reachable_fraction"] == 0.0
    assert ctx.call_count("robot.solve_ik") == 1
    assert "grasp pose itself" in out["detail"]


def test_min_fraction_accepts_a_partial_opening(script):
    out, _ = drawer(script, arm=Arm(fail={4: "raise", 5: "raise"}), min_fraction=0.6)
    assert out["route"] == "planned"
    assert out["opening"] == pytest.approx(0.09) and len(out["waypoints"]["waypoints"]) == 4
    out, _ = drawer(script, arm=Arm(fail={3: "raise"}), min_fraction=0.6)
    assert out["route"] == "unreachable"


def test_a_branch_flip_counts_as_unreachable(script):
    out, _ = door(script, arm=Arm(flip_at=2))
    assert out["route"] == "unreachable"
    assert out["reachable"][:3] == [True, True, False]
    assert "branch flip" in out["detail"]
    # The chain carries on from the last good solution, which the flip did not replace.
    assert all(out["reachable"][3:])


def test_the_jump_gate_can_be_widened(script):
    out, _ = door(script, arm=Arm(flip_at=2), max_joint_step=4.0)
    assert out["route"] == "planned"


def test_seeded_solves_start_from_the_previous_solution(script):
    out, ctx = drawer(script, seed_argument="seed_joints")
    calls = ctx.calls_to("robot.solve_ik")
    assert "seed_joints" not in calls[0].kwargs
    for prev, call, q in zip(
        out["joint_trajectory"]["waypoints"], calls[1:], range(5), strict=False
    ):
        assert call.kwargs["seed_joints"] == prev["positions"], q
    assert out["seeded"] is True and out["route"] == "planned"


def test_a_connector_without_a_seed_is_asked_again_and_not_seeded_after(script):
    out, ctx = drawer(script, arm=Arm(takes_seed=False), seed_argument="seed_joints")
    assert out["route"] == "planned" and out["seeded"] is False
    calls = ctx.calls_to("robot.solve_ik")
    assert sum("seed_joints" in c.kwargs for c in calls) == 1
    assert len(calls) == 7  # one refused seeded call, then six plain ones


def test_unseeded_by_default(script):
    out, ctx = drawer(script)
    assert out["seeded"] is False
    assert all("seed_joints" not in c.kwargs for c in ctx.calls)


def test_arm_id_is_passed_only_when_nonzero(script):
    _, ctx = drawer(script, arm_id=1)
    assert all(c.kwargs.get("arm_id") == 1 for c in ctx.calls)


def test_a_connector_reporting_no_joints_still_plans_poses(script):
    out, _ = drawer(script, arm=Arm(no_joints=True))
    assert out["route"] == "planned"
    assert len(out["waypoints"]["waypoints"]) == 6
    assert out["joint_trajectory"]["waypoints"] == []


def test_example_declares_every_route():
    path = Path(__file__).parents[2] / "skills" / BUNDLE / "examples" / "canonical_subgraph.json"
    example = json.loads(path.read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert {"planned", "unreachable"} <= declared
