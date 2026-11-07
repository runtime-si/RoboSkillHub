"""previewing-target-overlay: FakeContext tests of the projection, the drawing, the question and the routing.

The scene is synthetic: a z-up world with a table at z = 0, a 30 mm cube at (0.40, 0, 0.015) and an
overhead camera 1 m over (0.40, 0) looking straight down (image x along world +x, image y along
world -y), 640 x 480, f = 460 px. The hand's tool frame is the YAM station's: fingers along tool +z,
jaws along tool +y.
"""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError
from scipy.spatial.transform import Rotation

BUNDLE = "previewing-target-overlay"
W, H, F = 640, 480, 460.0
K = np.array([[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]])
CAM_AT = np.array([0.40, 0.0, 1.0])
DOWN_CAM = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}  # camera z -> world -z, camera y -> world -y
CUBE_C = np.array([0.40, 0.0, 0.015])
GRIPPER = {"span_m": 0.094, "finger": {"stated": True, "reach_m": 0.0102, "clearance_m": 0.004}}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["preview_target"].module


def _depth() -> np.ndarray:
    depth = np.full((H, W), CAM_AT[2], dtype=np.float32)
    half_px = int(F * 0.015 / (CAM_AT[2] - 0.03))
    depth[H // 2 - half_px : H // 2 + half_px, W // 2 - half_px : W // 2 + half_px] = (
        CAM_AT[2] - 0.03
    )
    return depth


def _camera(name="top", pose=None, rgb_shape=(H, W), depth=True, K_=K):
    return {
        "name": name,
        "rgb": np.full((*rgb_shape, 3), 128, dtype=np.uint8),
        "depth": _depth() if depth and rgb_shape == (H, W) else None,
        "intrinsics": K_,
        "pose": pose
        if pose is not None
        else {"position": dict(zip("xyz", map(float, CAM_AT), strict=True)), "rotation": DOWN_CAM},
    }


def _obs(*cams):
    return {"cameras": list(cams) or [_camera()]}


def station_frame(approach, heading_deg=0.0):
    down = np.asarray(approach, float) / np.linalg.norm(approach)
    yaw = np.radians(heading_deg)
    h = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    h -= (h @ down) * down
    if np.linalg.norm(h) < 1e-6:
        h = np.cross(down, [1.0, 0.0, 0.0])
    h /= np.linalg.norm(h)
    return np.column_stack([np.cross(h, down), h, down])


def _wxyz(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}


def _pose(p, R):
    return {"position": dict(zip("xyz", map(float, p), strict=True)), "rotation": _wxyz(R)}


def grasp_frame(approach=None, close_heading_deg=0.0, **_):
    R = station_frame([approach[k] for k in "xyz"] if approach else [0, 0, -1], close_heading_deg)
    return {
        "rotation": _wxyz(R),
        "approach": dict(zip("xyz", map(float, R[:, 2]), strict=True)),
        "close_axis": dict(zip("xyz", map(float, R[:, 1]), strict=True)),
    }


def _ctx(answer=True, obs=None, **overrides):
    tools = {
        "robot.describe_gripper": GRIPPER,
        "robot.grasp_frame": grasp_frame,
        "robot.get_observation": obs or _obs(),
        "vlm.query_yes_no": {"answer": answer, "text": ("YES fits." if answer else "NO, wrong.")},
    }
    tools.update(overrides)
    return FakeContext(tools)


TOP_DOWN = _pose(CUBE_C, station_frame([0, 0, -1]))


# ------------------------------------------------------------------ projection


def test_world_to_image_is_maestros_pinhole(script):
    T = np.eye(4)
    uv, scale, z = script.world_to_image(np.array([0.1, -0.05, 2.0]), K, T)
    assert uv == pytest.approx([W / 2 + F * 0.05, H / 2 - F * 0.025])
    assert scale == pytest.approx(F / 2.0)
    assert z == pytest.approx(2.0)
    assert script.world_to_image(np.array([0.0, 0.0, -1.0]), K, T) == (None, None, None)


def test_target_under_the_camera_lands_at_the_principal_point(script):
    out = script.run(_ctx(), target_pose=TOP_DOWN, intent="grasp the cube from above")
    assert out["pixels"]["top"] == pytest.approx([W / 2, H / 2], abs=1e-6)


@pytest.mark.parametrize("form", ["matrix", "packed"])
def test_camera_pose_as_matrix_or_packed_array(script, form):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([1.0, 0.0, 0.0, 0.0]).as_matrix()
    T[:3, 3] = CAM_AT
    pose = T if form == "matrix" else np.array([*CAM_AT, 0.0, 1.0, 0.0, 0.0])
    obs = _obs(_camera(pose=pose))
    out = script.run(_ctx(obs=obs), target_pose=TOP_DOWN, intent="grasp")
    assert out["pixels"]["top"] == pytest.approx([W / 2, H / 2], abs=1e-6)


# ------------------------------------------------------------------ routing


def test_yes_routes_approved_with_the_overlay_and_the_question(script):
    ctx = _ctx(answer=True)
    out = script.run(ctx, target_pose=TOP_DOWN, intent="grasp the small grey cube from above")
    assert out["route"] == "approved" and out["approved"] is True
    assert out["answer"] == "YES fits."
    assert out["views"] == ["top"]
    (call,) = ctx.calls_to("vlm.query_yes_no")
    assert call.kwargs["image"].shape == (H, W, 3)
    assert call.kwargs["image"] is out["overlay"]
    prompt = call.kwargs["prompt"]
    assert '"grasp the small grey cube from above"' in prompt
    assert "94 mm apart" in prompt
    assert "from above (top-down)" in prompt and "0 degrees from straight down" in prompt
    assert "model" not in call.kwargs


def test_no_routes_rejected(script):
    out = script.run(_ctx(answer=False), target_pose=TOP_DOWN, intent="grasp the box")
    assert out["route"] == "rejected" and out["approved"] is False
    assert out["reason"] == "NO, wrong."


def test_model_is_passed_through(script):
    ctx = _ctx()
    script.run(ctx, target_pose=TOP_DOWN, intent="grasp", model="gemini-x")
    assert ctx.calls_to("vlm.query_yes_no")[0].kwargs["model"] == "gemini-x"


def test_vlm_failure_routes_no_view_with_the_overlay(script):
    def boom(**_):
        raise ToolError("vlm", "backend unavailable")

    out = script.run(_ctx(**{"vlm.query_yes_no": boom}), target_pose=TOP_DOWN, intent="grasp")
    assert out["route"] == "no_view"
    assert "vlm.query_yes_no failed" in out["reason"]
    assert out["overlay"] is not None


@pytest.mark.parametrize(
    ("position", "why"),
    [
        ((0.40 + 0.9, 0.0, 0.015), "outside the image"),
        ((0.40, 0.0, 1.5), "outside the image"),  # behind the camera
    ],
)
def test_target_no_camera_sees_routes_no_view_without_asking(script, position, why):
    ctx = _ctx()
    out = script.run(ctx, target_pose=_pose(position, np.eye(3)), intent="grasp")
    assert out["route"] == "no_view"
    assert why in out["reason"]
    assert ctx.call_count("vlm.query_yes_no") == 0


def test_cameras_without_calibration_or_rgb_route_no_view(script):
    no_k = _camera()
    no_k["intrinsics"] = None
    no_rgb = _camera(name="side")
    no_rgb["rgb"] = None
    ctx = _ctx(obs={"cameras": [no_k, no_rgb]})
    out = script.run(ctx, target_pose=TOP_DOWN, intent="grasp")
    assert out["route"] == "no_view"
    assert "no rgb / intrinsics / pose" in out["reason"]
    assert ctx.call_count("vlm.query_yes_no") == 0


def test_empty_observation_routes_no_view(script):
    out = script.run(_ctx(obs={"cameras": []}), target_pose=TOP_DOWN, intent="grasp")
    assert out["route"] == "no_view"


def test_given_observation_is_used_instead_of_a_fresh_one(script):
    ctx = _ctx()
    script.run(ctx, target_pose=TOP_DOWN, intent="grasp", observation=_obs())
    assert ctx.call_count("robot.get_observation") == 0


# ------------------------------------------------------------------ views


def test_views_go_side_by_side_wrist_first_padded_to_the_tallest(script):
    wrist_pose = {"position": {"x": 0.40, "y": 0.0, "z": 0.30}, "rotation": DOWN_CAM}
    wrist = _camera(name="wrist", pose=wrist_pose, rgb_shape=(400, 300), K_=K * [[0.5], [0.5], [1]])
    ctx = _ctx(obs=_obs(_camera(), wrist))
    out = script.run(ctx, target_pose=TOP_DOWN, intent="grasp")
    assert out["views"] == ["wrist", "top"]
    assert out["overlay"].shape == (H, 300 + W, 3)
    assert "view 1 (left to right): wrist camera 'wrist'" in out["prompt"]


def test_cameras_filter_by_name(script):
    side = _camera(name="side")
    out = script.run(
        _ctx(obs=_obs(_camera(), side)), target_pose=TOP_DOWN, intent="g", cameras=["side"]
    )
    assert out["views"] == ["side"]


def test_only_cameras_that_see_the_target_are_sent(script):
    away = {"position": {"x": 3.0, "y": 0.0, "z": 1.0}, "rotation": DOWN_CAM}
    out = script.run(
        _ctx(obs=_obs(_camera(), _camera(name="far", pose=away))), target_pose=TOP_DOWN, intent="g"
    )
    assert out["views"] == ["top"]


# ------------------------------------------------------------------ drawing


def _colour_pixels(image, rgb, tol=40):
    return int(np.all(np.abs(image.astype(int) - np.array(rgb)) <= tol, axis=-1).sum())


def test_overlay_has_the_tool_centre_jaws_and_label(script):
    out = script.run(_ctx(), target_pose=TOP_DOWN, intent="g")
    img = out["overlay"]
    assert _colour_pixels(img, script.RED) > 0
    assert _colour_pixels(img, script.GREEN) > 0
    assert _colour_pixels(img, script.LABEL_BLUE) > 0
    # The jaws (heading 0: along world x, image x) span 94 mm; the finger roots, 45 mm nearer the
    # camera, are the widest green, plus the line's width.
    row = img[H // 2]
    green = np.where(np.all(np.abs(row.astype(int) - np.array(script.GREEN)) <= 40, axis=-1))[0]
    span_px = green.max() - green.min()
    assert span_px == pytest.approx(F * 0.094 / (1.0 - 0.015 - script.FINGER_DRAW_M), abs=3)


def test_gripper_segments_follow_span_reach_and_approach(script):
    c = np.zeros(3)
    a, j = np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])
    seg = script.gripper_segments(c, a, j, span=0.08, reach=0.01)
    (jaw,) = seg["jaw"]
    assert np.linalg.norm(jaw[1] - jaw[0]) == pytest.approx(0.08)
    (finger, _) = seg["fingers"]
    assert finger[1] == pytest.approx([0.04, 0.0, -0.01])  # tip: reach past the tool centre
    assert finger[0] == pytest.approx([0.04, 0.0, script.FINGER_DRAW_M])
    ((start, end),) = seg["approach"]
    assert start == pytest.approx([0.0, 0.0, script.APPROACH_LINE_M]) and end == pytest.approx(c)


def test_surface_gap_and_dashes_inside_the_table(script):
    below = _pose((0.30, 0.0, -0.04), station_frame([0, 0, -1]))
    out = script.run(_ctx(), target_pose=below, intent="grasp")
    assert out["surface_gap_m"]["top"] == pytest.approx(-0.04, abs=1e-3)
    assert "4.0 cm behind the surface" in out["prompt"]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([1.0, 0.0, 0.0, 0.0]).as_matrix()
    T[:3, 3] = CAM_AT
    img = np.zeros((H, W, 3), np.uint8)
    depth = _depth().astype(np.float64)
    hidden = script._draw_segment(
        img, np.array([0.28, 0, -0.04]), np.array([0.32, 0, -0.04]), K, T, depth, (0, 255, 0), 2
    )
    shown = script._draw_segment(
        img, np.array([0.28, 0, 0.04]), np.array([0.32, 0, 0.04]), K, T, depth, (0, 255, 0), 2
    )
    assert hidden > 0 and shown == 0


def test_surface_gap_in_the_air_and_inside_the_cube(script):
    air = script.run(_ctx(), target_pose=_pose(CUBE_C + [0, 0, 0.15], np.eye(3)), intent="g")
    assert air["surface_gap_m"]["top"] == pytest.approx(0.135, abs=1e-3)
    assert "13.5 cm in front of the surface" in air["prompt"]
    grasp = script.run(_ctx(), target_pose=TOP_DOWN, intent="g")
    assert grasp["surface_gap_m"]["top"] == pytest.approx(-0.015, abs=1e-3)


def test_geometry_cues_off_draws_as_maestro_did(script):
    out = script.run(_ctx(), target_pose=TOP_DOWN, intent="g", geometry_cues=False)
    assert out["surface_gap_m"]["top"] is None
    assert "Dashed" not in out["prompt"] and "degrees from straight down" not in out["prompt"]


# ------------------------------------------------------------------ axes and the hand


@pytest.mark.parametrize(
    ("R", "words", "tilt"),
    [
        (station_frame([0, 0, -1], 30.0), "from above (top-down)", 0),
        (station_frame([1, 0, -1]), "from above at a slant", 45),
        (station_frame([1, 0, 0], 90.0), "from the side (roughly horizontally)", 90),
        (station_frame([0, 0, 1]), "from below (moving upward)", 180),
    ],
)
def test_approach_is_read_off_the_rotation_through_grasp_frame(script, R, words, tilt):
    out = script.run(_ctx(), target_pose=_pose(CUBE_C, R), intent="g")
    assert words in out["prompt"] and f"{tilt} degrees from straight down" in out["prompt"]


def test_another_tool_frame_convention_is_followed(script):
    """A connector whose fingers point along tool -x: grasp_frame says so, the drawing follows."""
    flip = Rotation.from_euler("y", 90, degrees=True).as_matrix()  # tool -x -> old tool +z

    def other_frame(approach=None, close_heading_deg=0.0, **_):
        f = grasp_frame(approach, close_heading_deg)
        R = station_frame([0, 0, -1], close_heading_deg) @ flip.T
        return {**f, "rotation": _wxyz(R)}

    R_target = station_frame([0, 0, -1]) @ flip.T
    ctx = _ctx(**{"robot.grasp_frame": other_frame})
    out = script.run(ctx, target_pose=_pose(CUBE_C, R_target), intent="g")
    assert "0 degrees from straight down" in out["prompt"]


def test_without_grasp_frame_tool_z_is_assumed(script):
    def missing(**_):
        raise ToolError("robot.grasp_frame", "not served")

    out = script.run(_ctx(**{"robot.grasp_frame": missing}), target_pose=TOP_DOWN, intent="g")
    assert out["route"] == "approved"
    assert "0 degrees from straight down" in out["prompt"]


def test_given_axes_override_the_rotation(script):
    ctx = _ctx()
    out = script.run(
        ctx,
        target_pose=TOP_DOWN,
        intent="g",
        approach={"x": 1.0, "y": 0.0, "z": 0.0},
        close_axis={"x": 0.0, "y": 1.0, "z": 0.0},
    )
    assert "from the side" in out["prompt"]
    assert ctx.call_count("robot.grasp_frame") == 0


def test_jaw_width_and_describe_gripper_fallback(script):
    out = script.run(_ctx(), target_pose=TOP_DOWN, intent="g", jaw_width=0.06)
    assert "60 mm apart" in out["prompt"]

    def missing(**_):
        raise ToolError("robot.describe_gripper", "not served")

    out = script.run(_ctx(**{"robot.describe_gripper": missing}), target_pose=TOP_DOWN, intent="g")
    assert f"{script.DEFAULT_SPAN_M * 1000:.0f} mm apart" in out["prompt"]


@pytest.mark.parametrize(("arm_id", "expected"), [(None, {}), (0, {}), (1, {"arm_id": 1})])
def test_arm_id_is_passed_only_when_nonzero(script, arm_id, expected):
    ctx = _ctx()
    script.run(ctx, target_pose=TOP_DOWN, intent="g", arm_id=arm_id)
    assert ctx.calls_to("robot.describe_gripper")[0].kwargs == expected
    gf = ctx.calls_to("robot.grasp_frame")[0].kwargs
    assert {k: v for k, v in gf.items() if k == "arm_id"} == expected
