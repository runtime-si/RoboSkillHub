"""framing-a-target-by-mask-edge-steering: FakeContext tests with a rendered wrist camera riding a fake arm."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError
from scipy import ndimage

BUNDLE = "framing-a-target-by-mask-edge-steering"
TOP_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
# The real YAM station's calibrated left wrist D405, camera-to-tool (2026-11-05 dual-arm fit).
TOOL_CAM = np.array(
    [
        [0.0058, 0.701, 0.7131, -0.086],
        [-0.9995, 0.0253, -0.0168, 0.0088],
        [-0.0298, -0.7127, 0.7008, -0.0749],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
u_, s_, vt_ = np.linalg.svd(TOOL_CAM[:3, :3])
TOOL_CAM[:3, :3] = u_ @ vt_
# The D405's 640x480 intrinsics at half the resolution.
K = np.array([[197.0, 0.0, 160.0], [0.0, 196.8, 120.0], [0.0, 0.0, 1.0]])
HW = (240, 320)
RED = (220, 40, 40)
# A 30 mm cube on a table at z = 0.
CUBE = ((0.40, 0.10, 0.015), (0.03, 0.03, 0.03))
TOP = np.array([0.40, 0.10, 0.03])
CORNERS = np.array([[x, y, z] for x in (0.385, 0.415) for y in (0.085, 0.115) for z in (0.0, 0.03)])


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["frame_target"].module


def _rot(q):
    w, x, y, z = (q[k] for k in "wxyz")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def render(cam_world, boxes):
    """rgb and z-depth of a table at z = 0 and axis-aligned boxes, seen from ``cam_world``."""
    h, w = HW
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dirs_cam = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
    dirs = dirs_cam @ cam_world[:3, :3].T
    origin = cam_world[:3, 3]
    depth = np.full((h, w), np.inf)
    rgb = np.full((h, w, 3), 235, np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -origin[2] / dirs[..., 2]
    table = np.isfinite(t) & (t > 0)
    depth[table] = t[table]
    rgb[table] = 110
    for centre, size in boxes:
        lo = np.array(centre) - np.array(size) / 2
        hi = np.array(centre) + np.array(size) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - origin) / dirs, (hi - origin) / dirs
        tmin, tmax = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
        hit = (tmax >= tmin) & (tmin > 0) & (tmin < depth)
        depth[hit] = tmin[hit]
        rgb[hit] = RED
    depth[~np.isfinite(depth)] = 0.0
    # z-depth, as an RGB-D camera reports it.
    return rgb, (depth * dirs_cam[..., 2]).astype(np.float32)


def project(cam_world, points):
    pc = (np.linalg.inv(cam_world) @ np.c_[points, np.ones(len(points))].T)[:3]
    return np.stack([K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]], -1)


def look_down_tool(script, aim_point, rng=0.22):
    """The tool pose whose camera looks straight down at ``aim_point`` from ``rng``."""
    aim = script.aim_module()
    r0 = _rot(TOP_DOWN)
    r = aim._align(r0 @ TOOL_CAM[:3, 2], np.array([0.0, 0.0, -1.0])) @ r0
    tool = np.eye(4)
    tool[:3, :3] = r
    eye = np.asarray(aim_point, dtype=float) + np.array([0.0, 0.0, rng])
    tool[:3, 3] = eye - r @ TOOL_CAM[:3, 3]
    return tool


def offset_tool(script, du, dv, rng=0.22):
    """A look-down pose where the cube's top projects ``(du, dv)`` px from the image centre."""
    tool = look_down_tool(script, TOP, rng)
    cam = tool @ TOOL_CAM
    shift = -(du / K[0, 0]) * rng * cam[:3, 0] - (dv / K[1, 1]) * rng * cam[:3, 1]
    tool[:3, 3] += shift
    return tool


def _pose(script, m):
    aim = script.aim_module()
    return {"position": aim._vec(m[:3, 3]), "rotation": aim._quat(m[:3, :3])}


class Rig:
    """A fake arm whose moves land where commanded, with a wrist camera on its tool."""

    def __init__(self, script, tool, refuse=None):
        self.script = script
        self.tool = tool.copy()
        self.refuse = refuse or (lambda pose: None)
        self.moves = []
        self.boxes = [CUBE]
        self.wrist_name = "left_eye_in_hand"
        self.camera_world = None
        self.seg_calls = []
        self.blind = lambda n: False

    def get_observation(self, **kw):
        cam = self.tool @ TOOL_CAM
        self.camera_world = cam
        rgb, depth = render(cam, self.boxes)
        cameras = []
        if self.wrist_name:
            cameras.append(
                {
                    "name": self.wrist_name,
                    "rgb": rgb,
                    "depth": depth,
                    "intrinsics": K,
                    "pose": _pose(self.script, cam),
                }
            )
        return {"cameras": cameras, "arms": [{"ee_pose": _pose(self.script, self.tool)}]}

    def go_to_pose(self, pose, **kw):
        aim = self.script.aim_module()
        target = aim._matrix(pose)
        self.moves.append((target.copy(), kw))
        why = self.refuse(pose)
        if isinstance(why, Exception):
            raise why
        if why:
            return {"reached": False, "outcome": "unreachable", "reason": why}
        self.tool = target
        return {"reached": True, "outcome": "reached"}

    def segment(self, image, query, max_results=3, **kw):
        self.seg_calls.append((query, max_results))
        if self.blind(len(self.seg_calls)):
            return {"masks": [], "scores": [], "boxes": []}
        red = np.all(np.asarray(image) == RED, axis=-1)
        labels, n = ndimage.label(red)
        masks = [(labels == i).astype(np.uint8) for i in range(1, n + 1)]
        return {"masks": masks, "scores": [0.9] * n, "boxes": [[0, 0, 1, 1]] * n}

    def ctx(self, segment=None):
        return FakeContext(
            {
                "robot.get_observation": self.get_observation,
                "robot.get_ee_pose": lambda **kw: {"pose": _pose(self.script, self.tool)},
                "robot.go_to_pose": self.go_to_pose,
                "sam3.segment_text": segment or self.segment,
            }
        )

    def cube_pixels(self):
        return project(self.tool @ TOOL_CAM, CORNERS)


def _inside(px, margin):
    h, w = HW
    return bool(
        (px[:, 0] >= margin).all()
        and (px[:, 0] <= w - 1 - margin).all()
        and (px[:, 1] >= margin).all()
        and (px[:, 1] <= h - 1 - margin).all()
    )


# --- the image analysis -------------------------------------------------------------------


def test_analyze_reports_the_edges_within_the_margin(script):
    m = np.zeros(HW, bool)
    m[100:140, 0:30] = True
    r = script.analyze_mask(m, margin_px=10)
    assert r.seen and not r.full
    assert r.edges == frozenset({"left"})
    assert r.clearance_px == (0.0, 290.0, 100.0, 100.0)
    m = np.zeros(HW, bool)
    m[100:140, 140:180] = True
    r = script.analyze_mask(m, margin_px=10)
    assert r.full and r.edges == frozenset()
    assert r.center_error == pytest.approx(
        ((159.5 - 159.5) / 159.5, (119.5 - 119.5) / 119.5), abs=1e-6
    )


def test_the_margin_grows_with_the_target(script):
    m = np.zeros(HW, bool)
    m[30:210, 25:295] = True  # 270 px wide: pad 27 px > 15 px
    r = script.analyze_mask(m)
    assert r.context_pad_px == pytest.approx(27.0)
    assert r.edges == frozenset({"left", "right"})


def test_specks_do_not_read_as_a_clipped_edge(script):
    m = np.zeros(HW, bool)
    m[100:140, 140:180] = True
    m[0:3, 0:3] = True
    r = script.analyze_mask(m)
    assert r.full and r.bbox == (140, 100, 179, 139)
    # A 60 px glint by the edge is under 5 % of a large body; a large detached piece is kept.
    m = np.zeros(HW, bool)
    m[60:180, 60:260] = True
    m[100:106, 2:12] = True
    assert script.analyze_mask(m).edges == frozenset()
    m[100:140, 0:30] = True
    assert script.analyze_mask(m).edges == frozenset({"left"})


def test_the_bottom_band_counts_as_the_bottom_edge(script):
    m = np.zeros(HW, bool)
    m[150:180, 140:180] = True
    assert script.analyze_mask(m).full
    r = script.analyze_mask(m, ignore_bottom_frac=0.3)
    assert "bottom" in r.edges
    m[:] = False
    m[200:230, 140:180] = True
    assert not script.analyze_mask(m, ignore_bottom_frac=0.3).seen


def test_actions_follow_the_source(script):
    a = script.analyze_mask
    left = np.zeros(HW, bool)
    left[100:140, 0:30] = True
    act = script.adaptive_action(a(left), [])
    assert act.mode == "translate" and act.direction_image[0] < 0 and act.step_scale == 1.0
    wide = np.zeros(HW, bool)
    wide[60:180, 0:320] = True
    assert script.adaptive_action(a(wide), []).mode == "backoff"
    small = np.zeros(HW, bool)
    small[110:118, 150:158] = True  # 64 px, under 0.3 % of the image
    act = script.adaptive_action(a(small), [])
    assert act.mode == "approach"
    # The same edge three times running with no gain: too close, back off.
    r = a(left)
    assert script.adaptive_action(r, [r, r, r]).mode == "backoff"
    assert script.adaptive_action(script.MaskReport(False, False), []).mode == "hold"


def test_world_delta_moves_the_camera_toward_the_clipped_edge(script):
    right, down, away = (
        np.array([0.0, -1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    d = script.world_delta(np.array([-1.0, 0.0]), False, 0.03, right, down, away)
    np.testing.assert_allclose(d, [0.0, 0.03, 0.0])
    np.testing.assert_allclose(
        script.world_delta(np.zeros(2), True, 0.02, right, down, away), [0, 0, 0.02]
    )


# --- the loop on a fake arm -------------------------------------------------------------------


def test_an_already_framed_target_is_framed_without_moving(script):
    rig = Rig(script, offset_tool(script, 0, 0))
    out = script.run(rig.ctx(), object_name="red cube")
    assert out["route"] == "framed", out["detail"]
    assert rig.moves == [] and out["moves"] == 0
    assert out["edges"] == [] and out["camera_name"] == "left_eye_in_hand"
    np.testing.assert_allclose([out["target_point"][k] for k in "xyz"], TOP, atol=0.003)
    assert out["depth"] == pytest.approx(0.22, abs=0.003)
    assert rig.seg_calls[0] == ("red cube", 3)


@pytest.mark.parametrize(
    "du,dv,edge",
    [
        (165, 0, "right"),
        (-165, 0, "left"),
        (0, -125, "top"),
        (0, 125, "bottom"),
        (160, 118, "right"),
    ],
)
def test_a_clipped_target_is_brought_inside_the_margin(script, du, dv, edge):
    rig = Rig(script, offset_tool(script, du, dv))
    assert not _inside(rig.cube_pixels(), 0)
    out = script.run(rig.ctx(), object_name="red cube")
    assert out["route"] == "framed", out["detail"]
    assert out["trace"][0].startswith("translate") and edge in out["trace"][0]
    assert 1 <= out["moves"] <= 6
    assert _inside(rig.cube_pixels(), 15)
    # The hand only translated, keeping its rotation.
    np.testing.assert_allclose(rig.tool[:3, :3], offset_tool(script, 0, 0)[:3, :3], atol=1e-9)
    assert out["travel"] == pytest.approx(
        np.linalg.norm(rig.tool[:3, 3] - offset_tool(script, du, dv)[:3, 3])
    )
    # The median of the visible surface: the side faces seen near the border pull it a few mm.
    np.testing.assert_allclose([out["target_point"][k] for k in "xyz"], TOP, atol=0.008)


@pytest.mark.parametrize("du,dv", [(165, 0), (0, 125), (0, -125)])
def test_the_oblique_45_degree_view_of_a_top_down_hand_is_framed_too(script, du, dv):
    rng = 0.25
    tool = np.eye(4)
    tool[:3, :3] = _rot(TOP_DOWN)
    axis = tool[:3, :3] @ TOOL_CAM[:3, 2]
    tool[:3, 3] = TOP - rng * axis - tool[:3, :3] @ TOOL_CAM[:3, 3]
    cam = tool @ TOOL_CAM
    tool[:3, 3] += -(du / K[0, 0]) * rng * cam[:3, 0] - (dv / K[1, 1]) * rng * cam[:3, 1]
    rig = Rig(script, tool)
    assert axis[2] == pytest.approx(-0.70, abs=0.02)
    assert not _inside(rig.cube_pixels(), 0)
    out = script.run(rig.ctx(), object_name="red cube")
    assert out["route"] == "framed", out["detail"]
    assert _inside(rig.cube_pixels(), 15)
    np.testing.assert_allclose(rig.tool[:3, :3], _rot(TOP_DOWN), atol=1e-9)


def test_a_far_small_target_is_approached(script):
    rig = Rig(script, look_down_tool(script, TOP, rng=0.48))
    out = script.run(rig.ctx(), object_name="red cube")
    assert out["route"] == "framed", out["detail"]
    assert out["trace"][0].startswith("approach")
    assert out["area_frac"] >= script.MIN_AREA_FRAC
    assert out["depth"] < 0.48


def test_a_small_target_at_the_nearest_range_is_framed_without_approaching(script):
    rig = Rig(script, look_down_tool(script, TOP, rng=0.48))
    out = script.run(rig.ctx(), object_name="red cube", min_area_frac=0.05, min_range=0.49)
    assert out["route"] == "framed" and "small" in out["detail"]
    assert rig.moves == []


def test_a_target_spanning_opposite_edges_is_backed_off_from(script):
    rig = Rig(script, look_down_tool(script, TOP, rng=0.22))
    rig.boxes = [((0.40, 0.10, 0.015), (0.40, 0.03, 0.03))]
    out = script.run(rig.ctx(), object_name="red cube", context_pad_frac=0.0)
    assert out["route"] == "framed", out["detail"]
    assert out["trace"][0].startswith("backoff")
    first = rig.moves[0][0]
    start = look_down_tool(script, TOP, rng=0.22)
    np.testing.assert_allclose(first[:3, 3] - start[:3, 3], [0, 0, 0.03], atol=1e-6)


def test_backing_off_past_the_depth_range_is_unreachable(script):
    rig = Rig(script, look_down_tool(script, TOP, rng=0.22))
    rig.boxes = [((0.40, 0.10, 0.015), (0.40, 0.03, 0.03))]
    out = script.run(rig.ctx(), object_name="red cube", context_pad_frac=0.0, max_range=0.23)
    assert out["route"] == "unreachable" and "farthest" in out["detail"]
    assert rig.moves == []


def test_approach_keeps_the_tool_centre_clear_of_the_target(script):
    rig = Rig(script, look_down_tool(script, TOP, rng=0.48))
    out = script.run(
        rig.ctx(), object_name="red cube", min_area_frac=0.2, min_range=0.0, max_moves=30
    )
    assert out["route"] == "unreachable" and "within 50 mm" in out["detail"]
    for target, _ in rig.moves:
        assert target[2, 3] >= 0.03 + 0.05 - 1e-6


def test_nothing_seen_is_not_seen_without_moving(script):
    rig = Rig(script, offset_tool(script, 0, 0))
    rig.boxes = []
    out = script.run(rig.ctx(), object_name="red cube")
    assert out["route"] == "not_seen" and "found no" in out["detail"]
    assert rig.moves == []


def test_a_failing_segmenter_is_not_seen(script):
    rig = Rig(script, offset_tool(script, 0, 0))

    def broken(image, query, **kw):
        raise ToolError("sam3.segment_text", "CUDA OOM")

    out = script.run(rig.ctx(segment=broken), object_name="red cube")
    assert out["route"] == "not_seen" and "CUDA OOM" in out["detail"]


def test_low_scores_are_ignored(script):
    rig = Rig(script, offset_tool(script, 0, 0))

    def weak(image, query, **kw):
        reply = rig.segment(image, query, **kw)
        return {**reply, "scores": [0.1] * len(reply["masks"])}

    out = script.run(rig.ctx(segment=weak), object_name="red cube")
    assert out["route"] == "not_seen"


def test_no_wrist_camera_is_not_seen(script):
    rig = Rig(script, offset_tool(script, 0, 0))
    rig.wrist_name = ""
    out = script.run(rig.ctx(), object_name="red cube")
    assert out["route"] == "not_seen" and "no wrist camera" in out["detail"]


def test_a_target_lost_after_a_move_sends_the_hand_back_and_halves_the_step(script):
    start = offset_tool(script, 165, 0)
    rig = Rig(script, start)
    rig.blind = lambda n: n == 2
    out = script.run(rig.ctx(), object_name="red cube")
    assert out["route"] == "framed", out["detail"]
    assert any(t.startswith("lost") for t in out["trace"])
    first, back, second = (m[0] for m in rig.moves[:3])
    np.testing.assert_allclose(back[:3, 3], start[:3, 3], atol=1e-9)
    step1 = np.linalg.norm(first[:3, 3] - start[:3, 3])
    step2 = np.linalg.norm(second[:3, 3] - start[:3, 3])
    assert step2 == pytest.approx(step1 / 2, rel=0.05)


def test_a_target_lost_twice_is_not_seen_back_at_the_last_view(script):
    start = offset_tool(script, 165, 0)
    rig = Rig(script, start)
    rig.blind = lambda n: n >= 2
    out = script.run(rig.ctx(), object_name="red cube")
    assert out["route"] == "not_seen" and "did not find it again" in out["detail"]
    np.testing.assert_allclose(rig.tool[:3, 3], start[:3, 3], atol=1e-9)


@pytest.mark.parametrize("refusal", ["joint limit", ToolError("robot.go_to_pose", "IK failed")])
def test_refused_steps_are_retried_at_half_length_then_unreachable(script, refusal):
    start = offset_tool(script, 165, 0)
    rig = Rig(script, start, refuse=lambda pose: refusal)
    out = script.run(rig.ctx(), object_name="red cube")
    assert out["route"] == "unreachable"
    assert "joint limit" in out["detail"] or "IK failed" in out["detail"]
    full, half = (np.linalg.norm(m[0][:3, 3] - start[:3, 3]) for m in rig.moves)
    assert len(rig.moves) == 2 and half == pytest.approx(full / 2)


def test_a_hand_that_ends_off_counts_as_refused(script):
    start = offset_tool(script, 165, 0)
    rig = Rig(script, start)
    real = rig.go_to_pose

    def short(pose, **kw):
        reply = real(pose, **kw)
        rig.tool[2, 3] += 0.03
        return reply

    rig.go_to_pose = short
    out = script.run(rig.ctx(), object_name="red cube")
    assert out["route"] == "unreachable" and "30 mm from the step" in out["detail"]


def test_the_move_cap_ends_with_step_limit(script):
    rig = Rig(script, offset_tool(script, 165, 0))
    out = script.run(rig.ctx(), object_name="red cube", max_moves=1, step=0.005)
    assert out["route"] == "step_limit" and "right" in out["detail"]
    assert out["moves"] == 1 and len(rig.moves) == 1


def test_a_target_point_picks_the_nearest_mask(script):
    rig = Rig(script, offset_tool(script, 0, 0))
    other = ((0.40, 0.16, 0.015), (0.03, 0.03, 0.03))
    rig.boxes = [CUBE, other]

    def biased(image, query, **kw):
        reply = rig.segment(image, query, **kw)
        n = len(reply["masks"])
        # The other cube scores higher.
        centroids = [np.nonzero(m)[1].mean() for m in reply["masks"]]
        order = np.argsort(centroids)
        return {**reply, "scores": [0.9 if i == order[0] else 0.95 for i in range(n)]}

    cube_u = project(rig.tool @ TOOL_CAM, TOP[None])[0, 0]
    other_u = project(rig.tool @ TOOL_CAM, np.array([[0.40, 0.16, 0.03]]))[0, 0]
    assert cube_u < other_u
    out = script.run(
        rig.ctx(segment=biased),
        object_name="red cube",
        target_point=dict(zip("xyz", TOP, strict=True)),
    )
    assert out["route"] == "framed", out["detail"]
    np.testing.assert_allclose([out["target_point"][k] for k in "xyz"], TOP, atol=0.003)
    out = script.run(rig.ctx(segment=biased), object_name="red cube")
    np.testing.assert_allclose(
        [out["target_point"][k] for k in "xyz"], [0.40, 0.16, 0.03], atol=0.008
    )


def test_connector_knobs_are_passed_only_when_set(script):
    rig = Rig(script, offset_tool(script, 165, 0))
    script.run(rig.ctx(), object_name="red cube")
    assert all(kw == {} for _, kw in rig.moves)
    rig = Rig(script, offset_tool(script, 165, 0))
    script.run(rig.ctx(), object_name="red cube", tolerance=0.004, max_steps=500)
    assert all(kw == {"tolerance": 0.004, "max_steps": 500} for _, kw in rig.moves)


def test_a_named_camera_is_used(script):
    rig = Rig(script, offset_tool(script, 0, 0))
    rig.wrist_name = "d405"
    out = script.run(rig.ctx(), object_name="red cube", camera="d405")
    assert out["route"] == "framed" and out["camera_name"] == "d405"
