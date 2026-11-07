"""selecting-safe-cable-grasp-segments: FakeContext tests of the safety gates, the ranking and the routing.

Cables are synthetic centrelines on a table at z = 0 (axis 2 mm up). robot.grasp_frame is a small top-down frame
builder; robot.solve_ik answers from a reach predicate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "selecting-safe-cable-grasp-segments"
Z = 0.002


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["select_cable_grasps"].module


def _quat(heading_deg: float) -> dict[str, float]:
    a = math.radians(heading_deg) / 2.0
    return {"w": 0.0, "x": math.cos(a), "y": math.sin(a), "z": 0.0}


def grasp_frame(close_heading_deg=0.0, arm_id=0):
    return {
        "rotation": _quat(close_heading_deg),
        "rotation_flipped": _quat(close_heading_deg + 180.0),
    }


def _ctx(reach=lambda p, rot: True, frame=grasp_frame):
    def solve_ik(pose, **kwargs):
        p = np.array([pose["position"][k] for k in "xyz"])
        if not reach(p, pose["rotation"]):
            raise ToolError("robot.solve_ik", "Target pose is unreachable")
        return {"joint_config": {"positions": [0.0] * 6}, "success": True}

    return FakeContext({"robot.grasp_frame": frame, "robot.solve_ik": solve_ik})


def _line(x0, x1, y=0.30, n=61, z=Z):
    xs = np.linspace(x0, x1, n)
    return {
        "points": [[float(x), y, z] for x in xs],
        "arclength_m": abs(x1 - x0),
        "radius_m": 0.00175,
        "nodes": n,
        "ordered": True,
    }


def _xy(g):
    return np.array([g["point"]["x"], g["point"]["y"]])


def _box(cx, cy, hx, hy, top, bottom=0.0):
    return {
        "center": {"x": cx, "y": cy, "z": (top + bottom) / 2},
        "extent": {"x": hx, "y": hy, "z": (top - bottom) / 2},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


# --- happy path and the pose ------------------------------------------------------------------------------------


def test_straight_cable_found_middle_first_jaws_across(script):
    ctx = _ctx()
    out = script.run(ctx, centerline=_line(0.0, 0.60))
    assert out["route"] == "found"
    assert out["pose"] == out["poses"][0]
    best = out["grasps"][0]
    assert abs(best["point"]["x"] - 0.30) < 0.06  # the farthest from both ends ranks first
    assert best["curvature"] < 1.0 and best["clearance_m"] is None
    # tangent along +x: the jaws close along y
    assert abs((best["close_heading_deg"] - 90.0) % 180.0) < 1e-6
    assert best["pose"]["position"]["z"] == pytest.approx(Z + script.GRASP_HEIGHT_M)
    assert best["pose"]["rotation"] == _quat(best["close_heading_deg"])


def test_never_within_the_end_margin_and_spaced(script):
    out = script.run(_ctx(), centerline=_line(0.0, 0.60), max_poses=10)
    xs = sorted(g["point"]["x"] for g in out["grasps"])
    assert len(xs) == 10
    assert xs[0] >= script.END_MARGIN_M + script.FINGER_HALF_M - 1e-9
    assert xs[-1] <= 0.60 - script.END_MARGIN_M - script.FINGER_HALF_M + 1e-9
    assert min(np.diff(xs)) >= 0.03 - 1e-9


def test_tip_end_gets_the_larger_margin(script):
    cab = _line(0.0, 0.40)
    plain = script.run(_ctx(), centerline=cab, max_poses=500, min_spacing_m=0.0, check_reach=False)
    tipped = script.run(
        _ctx(),
        centerline=cab,
        endpoints={"tip": 0},
        max_poses=500,
        min_spacing_m=0.0,
        check_reach=False,
    )
    lo = min(g["point"]["x"] for g in plain["grasps"])
    lo_tip = min(g["point"]["x"] for g in tipped["grasps"])
    hi_tip = max(g["point"]["x"] for g in tipped["grasps"])
    assert lo < script.TIP_MARGIN_M
    assert lo_tip >= script.TIP_MARGIN_M + script.FINGER_HALF_M - 1e-9
    assert hi_tip == pytest.approx(max(g["point"]["x"] for g in plain["grasps"]))
    assert all(g["tip_distance_m"] == pytest.approx(g["arc_m"]) for g in tipped["grasps"])


def test_accepts_points_list_and_json_text(script):
    cab = _line(0.0, 0.5)
    a = script.run(_ctx(), centerline=cab["points"])
    b = script.run(_ctx(), centerline=json.dumps(cab))
    assert a["route"] == b["route"] == "found"
    assert a["grasps"][0]["arc_m"] == b["grasps"][0]["arc_m"]


# --- the geometry gates ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cab",
    [
        _line(0.0, 0.09),  # shorter than both margins
        {"points": [[0.0, 0.3, Z]]},
        {"points": [], "ordered": True},
        {**_line(0.0, 0.6), "ordered": False},
    ],
)
def test_none_safe_without_a_usable_cable(script, cab):
    ctx = _ctx()
    out = script.run(ctx, centerline=cab)
    assert out["route"] == "none_safe" and out["pose"] is None and out["poses"] == []
    assert not ctx.calls_to("robot.solve_ik")


def test_hidden_stretch_from_visibility_splits_with_margin(script):
    cab = _line(0.0, 0.60)
    xs = np.linspace(0.0, 0.60, 61)
    cab["visibility"] = [0.0 if 0.25 <= x <= 0.35 else 1.0 for x in xs]
    out = script.run(_ctx(), centerline=cab, max_poses=500, min_spacing_m=0.0, check_reach=False)
    assert out["route"] == "found"
    for g in out["grasps"]:
        x = g["point"]["x"]
        assert (
            x <= 0.25 - script.END_MARGIN_M - script.FINGER_HALF_M + 0.006
            or x >= 0.35 + script.END_MARGIN_M + script.FINGER_HALF_M - 0.006
        )
    assert out["rejected"]["hidden"] > 0.1


def test_occluded_indices_and_raised_stretch_are_hidden(script):
    cab = _line(0.0, 0.60)
    pts = np.array(cab["points"])
    pts[25:36, 2] = 0.15  # depth met something over the cable here
    cab["points"] = pts.tolist()
    out = script.run(
        _ctx(),
        centerline=cab,
        occluded_indices=[10],
        max_poses=500,
        min_spacing_m=0.0,
        check_reach=False,
    )
    # Where the raised stretch lies in arc length (the climb onto the occluder counts as cable).
    s_nodes = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    fine = np.linspace(0.0, s_nodes[-1], 20001)
    up = fine[np.interp(fine, s_nodes, pts[:, 2]) - Z > script.MAX_RISE_M]
    keep_off = script.END_MARGIN_M + script.FINGER_HALF_M - 1e-3
    assert out["route"] == "found"
    for g in out["grasps"]:
        assert g["arc_m"] < up.min() - keep_off or g["arc_m"] > up.max() + keep_off
        assert abs(g["point"]["x"] - 0.10) > script.END_MARGIN_M
        assert g["pose"]["position"]["z"] < 0.02


def test_crossing_forbids_its_radius(script):
    out = script.run(
        _ctx(),
        centerline=_line(0.0, 0.60),
        crossing_indices=[30],
        max_poses=500,
        min_spacing_m=0.0,
        check_reach=False,
    )
    assert all(abs(g["point"]["x"] - 0.30) > script.END_MARGIN_M for g in out["grasps"])
    assert out["rejected"]["crossing"] > 0


def test_tight_bend_is_avoided(script):
    # Two 0.3 m straights joined by a 20 mm-radius half turn: the turn and the legs beside it are out.
    r = 0.02
    leg = np.linspace(0.0, 0.30, 31)
    a = np.linspace(-math.pi / 2, math.pi / 2, 13)
    pts = (
        [[float(x), 0.30, Z] for x in leg]
        + [[0.30 + r * math.cos(t), 0.30 + r + r * math.sin(t), Z] for t in a[1:-1]]
        + [[float(x), 0.30 + 2 * r, Z] for x in leg[::-1]]
    )
    out = script.run(
        _ctx(),
        centerline={"points": pts, "radius_m": 0.00175},
        max_poses=500,
        min_spacing_m=0.0,
        check_reach=False,
    )
    assert out["rejected"]["bend"] > 0
    for g in out["grasps"]:
        assert g["curvature"] <= script.MAX_CURVATURE
        assert (
            g["point"]["x"] <= 0.30 - script.FINGER_HALF_M + 1e-6
        )  # the window never reaches the turn
    # The legs are 40 mm apart: fingers opened 47 mm across one still clear the other.
    assert any(g["clearance_m"] is not None for g in out["grasps"])
    assert all(
        g["clearance_m"] is None or g["clearance_m"] >= script.MIN_CLEARANCE_M
        for g in out["grasps"]
    )


def test_parallel_stretch_of_the_same_cable_blocks_the_fingers(script):
    r = 0.012
    leg = np.linspace(0.0, 0.40, 41)
    a = np.linspace(-math.pi / 2, math.pi / 2, 9)
    pts = (
        [[float(x), 0.30, Z] for x in leg]
        + [[0.40 + r * math.cos(t), 0.30 + r + r * math.sin(t), Z] for t in a[1:-1]]
        + [[float(x), 0.30 + 2 * r, Z] for x in leg[::-1]]
    )
    out = script.run(_ctx(), centerline={"points": pts}, check_reach=False)
    assert out["route"] == "none_safe"
    assert out["rejected"]["clearance"] > 0.8


def test_obstacle_beside_the_cable_and_box_under_it(script):
    cab = _line(0.0, 0.60)
    # A board 15 mm beside the cable's right half (its edge at y = 0.285), standing 13 mm.
    board = _box(0.50, 0.185, 0.10, 0.10, top=0.013)
    out = script.run(
        _ctx(),
        centerline=cab,
        obstacles=[board],
        max_poses=500,
        min_spacing_m=0.0,
        check_reach=False,
    )
    assert out["rejected"]["clearance"] > 0.2
    assert all(g["point"]["x"] < 0.40 - script.MIN_CLEARANCE_M for g in out["grasps"])
    assert all(
        g["clearance_m"] is None or g["clearance_m"] >= script.MIN_CLEARANCE_M
        for g in out["grasps"]
    )
    # The same board under the cable (the cable lies on it) is no obstacle.
    on = _line(0.0, 0.60, z=0.013 + Z)
    under = _box(0.30, 0.30, 0.40, 0.10, top=0.013)
    out2 = script.run(_ctx(), centerline=on, obstacles=[under], check_reach=False)
    assert out2["rejected"]["clearance"] == 0.0 and out2["route"] == "found"


def test_rotated_obstacle_box(script):
    a = math.radians(45.0) / 2
    box = _box(0.30, 0.30, 0.03, 0.03, top=0.05)
    box["orientation"] = {"w": math.cos(a), "x": 0.0, "y": 0.0, "z": math.sin(a)}
    out = script.run(
        _ctx(),
        centerline=_line(0.0, 0.60),
        obstacles=[box],
        max_poses=500,
        min_spacing_m=0.0,
        check_reach=False,
    )
    # The diamond's corner reaches 42 mm along the cable from its centre.
    reach = 0.03 * math.sqrt(2) + script.FINGER_HALF_M + script.MIN_CLEARANCE_M
    assert all(abs(g["point"]["x"] - 0.30) > reach - 1e-6 for g in out["grasps"])


def test_other_cable_crossing_is_kept_clear(script):
    other = {"points": [[0.30, y, 0.006] for y in np.linspace(0.1, 0.5, 41)]}
    out = script.run(
        _ctx(),
        centerline=_line(0.0, 0.60),
        other_cables=[other],
        max_poses=500,
        min_spacing_m=0.0,
        check_reach=False,
    )
    assert all(
        abs(g["point"]["x"] - 0.30) > script.FINGER_HALF_M + script.MIN_CLEARANCE_M
        for g in out["grasps"]
    )


# --- reach ---------------------------------------------------------------------------------------------------------


def test_reach_filters_and_ranks_within_the_box(script):
    ctx = _ctx(reach=lambda p, rot: p[0] <= 0.20)
    out = script.run(ctx, centerline=_line(0.0, 0.60))
    assert out["route"] == "found"
    assert all(g["point"]["x"] <= 0.20 for g in out["grasps"])
    assert out["checked"] <= 40
    xs = [g["point"]["x"] for g in out["grasps"]]
    assert xs[0] == max(xs)  # nearest the middle among the reachable


def test_flipped_rotation_used_when_the_first_fails(script):
    ctx = _ctx(reach=lambda p, rot: rot["x"] < 0)  # only the flipped wrist solution solves
    out = script.run(ctx, centerline=_line(0.0, 0.60), max_poses=1)
    assert out["route"] == "found"
    g = out["grasps"][0]
    assert g["rotation_used"] == "rotation_flipped"
    assert g["pose"]["rotation"]["x"] < 0


@pytest.mark.parametrize(
    "reply",
    [
        {"success": False},
        {"joint_config": [0.0] * 6, "within_servo_tolerance": False},
        {"joint_config": [0] * 6, "position_error_m": 0.05},
        None,
    ],
)
def test_ik_misses_route_none_reachable(script, reply):
    ctx = FakeContext({"robot.grasp_frame": grasp_frame, "robot.solve_ik": reply})
    out = script.run(ctx, centerline=_line(0.0, 0.60), max_checks=6)
    assert out["route"] == "none_reachable"
    assert out["pose"] is None and out["checked"] == 6
    assert len(ctx.calls_to("robot.solve_ik")) == 12  # both wrist solutions per candidate
    assert out["candidate_count"] > 0


def test_raised_ik_routes_none_reachable(script):
    out = script.run(_ctx(reach=lambda p, rot: False), centerline=_line(0.0, 0.60))
    assert out["route"] == "none_reachable"
    assert "unreachable" in out["detail"]


def test_grasp_frame_failure_is_routed(script):
    def frame(**kwargs):
        raise ToolError("robot.grasp_frame", "no gripper")

    out = script.run(_ctx(frame=frame), centerline=_line(0.0, 0.60), max_checks=3)
    assert out["route"] == "none_reachable"
    assert "grasp_frame" in out["detail"]


def test_arm_id_passed_only_when_nonzero(script):
    ctx = _ctx()
    script.run(ctx, centerline=_line(0.0, 0.60), max_poses=1)
    assert all("arm_id" not in c.kwargs for c in ctx.calls)
    ctx = _ctx()
    script.run(ctx, centerline=_line(0.0, 0.60), max_poses=1, arm_id=1)
    assert all(c.kwargs.get("arm_id") == 1 for c in ctx.calls)


def test_check_reach_off_makes_no_ik_calls(script):
    ctx = _ctx()
    out = script.run(ctx, centerline=_line(0.0, 0.60), check_reach=False)
    assert out["route"] == "found" and not ctx.calls_to("robot.solve_ik")


# --- the bundle -----------------------------------------------------------------------------------------------------


def test_example_declares_every_route(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    root = (
        Path(bundle.path)
        if hasattr(bundle, "path")
        else Path(__file__).resolve().parents[2] / "skills" / BUNDLE
    )
    ex = json.loads((root / "examples" / "canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert {"found", "none_safe", "none_reachable"} <= declared
