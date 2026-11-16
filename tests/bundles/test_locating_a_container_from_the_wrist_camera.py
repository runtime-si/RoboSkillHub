"""locating-a-container-from-the-wrist-camera: the in-view fit, the view search and its IK budget."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

from ._bin_scenes import DOWN, Scene, obb

BUNDLE = "locating-a-container-from-the-wrist-camera"
TRUE = (0.40, -0.07)  # where the box really is
SEEN_FROM = (0.40, -0.07, 0.50)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["locate_container"].module


def _scene():
    return Scene().open_box(*TRUE, 0.24, 0.18, -50.0, 0.066)


def _expected(dx=0.0, dy=0.0, yaw=-50.0):
    return obb(TRUE[0] + dx, TRUE[1] + dy, 0.24, 0.18, yaw, 0.066)


def _wrist(scene=None, cam=SEEN_FROM, **kw):
    return (scene or _scene()).frame(cam_xyz=cam, focal=400.0, name="right_wrist", **kw)


def test_the_shift_from_the_expected_place_is_measured(script):
    # the overhead camera's place for the box is 48, -39 mm from where this camera has it
    out = script.run(FakeContext({}), _expected(-0.048, 0.039), camera_frame=_wrist())
    assert out["route"] == "located", out["detail"]
    assert out["shift"]["x"] == pytest.approx(0.048, abs=0.006)
    assert out["shift"]["y"] == pytest.approx(-0.039, abs=0.006)
    assert out["aim_point"]["x"] == pytest.approx(TRUE[0], abs=0.006)
    assert out["aim_point"]["z"] == pytest.approx(0.066)
    assert out["located_obb"]["extent"]["x"] == pytest.approx(0.12)
    assert abs(out["yaw_deg"] + 50.0) <= 2.01


def test_a_rim_the_view_cuts_is_placed_by_its_size(script):
    # the camera stands off to one side: under half of the rim is in the picture
    frame = _wrist(cam=(0.40 + 0.20, -0.07 - 0.16, 0.36))
    pts = script.world_points(frame, 2)
    rim = pts[(pts[:, 2] > 0.05) & (pts[:, 2] < 0.08)]
    assert np.linalg.norm(rim[:, :2].mean(0) - TRUE) > 0.03  # what is seen is not centred on it
    out = script.run(FakeContext({}), _expected(0.02, 0.02), camera_frame=frame)
    assert out["route"] == "located", out["detail"]
    assert out["coverage"] < 0.8
    assert np.hypot(out["aim_point"]["x"] - TRUE[0], out["aim_point"]["y"] - TRUE[1]) < 0.012


def test_a_neighbours_wall_is_not_taken_for_the_rim(script):
    tall = (0.40, 0.14, 0.40, 0.20, 0.0, 0.15)
    scene = _scene().open_box(*tall)
    pts = script.world_points(_wrist(scene), 2)
    with_wall = script.box_in_view(pts, _expected(0.0, 0.03))
    without = script.box_in_view(pts, _expected(0.0, 0.03), [obb(*tall)])
    assert without is not None
    assert np.linalg.norm(without["center"] - TRUE) < 0.008
    assert with_wall is None or with_wall["rim_points"] > without["rim_points"]


def test_a_fit_too_far_from_the_expected_place_is_not_accepted(script):
    out = script.run(FakeContext({}), _expected(-0.07, 0.0), camera_frame=_wrist(), max_shift=0.05)
    assert out["route"] == "not_seen" and "not accepted" in out["detail"]
    assert out["located_obb"] is None and out["coverage"] > 0.3


def test_nothing_in_view_is_not_seen(script):
    out = script.run(FakeContext({}), _expected(), camera_frame=_wrist(Scene()))
    assert out["route"] == "not_seen" and "rim" in out["detail"]
    no_cam = FakeContext({"robot.get_observation": {"cameras": [Scene().frame(name="top")]}})
    out = script.run(no_cam, _expected(), move=False)
    assert out["route"] == "not_seen" and "wrist" in out["detail"]


def test_the_hands_own_fingers_are_dropped_by_range(script):
    frame = _wrist()
    frame["depth"][200:280, 300:340] = 0.10  # a fingertip in front of the lens
    near = script.world_points(frame, 1, 0.0)
    far = script.world_points(frame, 1, 0.16)
    assert len(near) - len(far) == 80 * 40


def test_views_stand_short_of_the_centre_on_the_hands_side(script):
    views = script.view_candidates(
        np.array([0.4, 0.0]), 0.066, np.array([0.4, -0.5]), [DOWN], [DOWN]
    )
    assert [round(float(v[0][1]), 2) for v in views] == [-0.14, -0.20, -0.08, -0.05, -0.09, 0.0]
    assert views[0][2][0] == pytest.approx(0.066 + 0.33) and views[3][2][0] == pytest.approx(0.296)
    assert len(script.view_candidates(np.zeros(2), 0.0, np.zeros(2), [], [DOWN])) == 3


def _pose(x, y, z):
    return {"pose": {"position": {"x": x, "y": y, "z": z}, "rotation": DOWN}}


def _robot(ik, start_z=0.10, frame=None):
    state = {"at": (0.40, -0.45, start_z)}

    def go(pose, **_):
        p = pose["position"]
        state["at"] = (p["x"], p["y"], p["z"])
        return {"success": True}

    return FakeContext(
        {
            "robot.get_ee_pose": lambda **_: _pose(*state["at"]),
            "robot.grasp_frame": {"rotation": DOWN},
            "robot.solve_ik": ik,
            "robot.go_to_pose": go,
            "robot.go_to_pose_cartesian": go,
            "robot.get_observation": {"cameras": [frame or _wrist()]},
        }
    )


def test_it_rises_then_goes_to_the_first_view_that_passes_the_preview(script):
    def ik(pose, **_):
        if pose["position"]["z"] > 0.25:
            raise ToolError("robot.solve_ik", "Target pose is unreachable")
        return {"joint_config": {"positions": [0.0] * 6}, "success": True}

    ctx = _robot(ik)
    out = script.run(ctx, _expected(-0.03, 0.02))
    assert out["route"] == "located", out["detail"]
    assert out["views_tried"] == 1 and out["ik_queries"] == 3
    (rise,) = ctx.calls_to("robot.go_to_pose_cartesian")
    assert rise.kwargs["pose"]["position"]["z"] == pytest.approx(0.066 + 0.11 + 0.02)
    (view,) = ctx.calls_to("robot.go_to_pose")
    assert view.kwargs["pose"]["position"]["z"] == pytest.approx(0.066 + 0.17)
    assert out["view_pose"]["position"] == view.kwargs["pose"]["position"]
    assert out["shift"]["x"] == pytest.approx(0.03, abs=0.006)


def test_the_search_stops_at_its_ik_budget_and_moves_nothing(script):
    def never(**_):
        raise ToolError("robot.solve_ik", "Target pose is unreachable")

    ctx = _robot(never)
    out = script.run(ctx, _expected(), rotations=[DOWN], max_ik_queries=7)
    assert out["route"] == "unreachable" and out["ik_queries"] == 7
    assert ctx.call_count("robot.solve_ik") == 7
    assert ctx.call_count("robot.go_to_pose") == 0 and "7 IK previews" in out["detail"]


def test_given_view_poses_are_sent_without_a_preview_and_all_are_tried(script):
    ctx = _robot(None, start_z=0.40, frame=_wrist(Scene()))
    views = [_pose(0.40, -0.20, 0.30)["pose"], _pose(0.40, -0.10, 0.25)["pose"]]
    out = script.run(ctx, _expected(), view_poses=views)
    assert out["route"] == "not_seen" and out["views_tried"] == 2
    assert ctx.call_count("robot.solve_ik") == 0 and ctx.call_count("robot.go_to_pose") == 2
    assert ctx.call_count("robot.go_to_pose_cartesian") == 0  # already over the rims


def test_refused_view_moves_are_unreachable(script):
    def refuse(**_):
        raise ToolError("robot.go_to_pose", "no plan")

    ctx = FakeContext(
        {
            "robot.get_ee_pose": _pose(0.4, -0.45, 0.40),
            "robot.go_to_pose": refuse,
        }
    )
    out = script.run(ctx, _expected(), view_poses=[_pose(0.40, -0.20, 0.30)["pose"]])
    assert out["route"] == "unreachable" and "refused" in out["detail"]
