"""refining-target-from-wrist-closeup: FakeContext tests with a rendered wrist camera riding a fake arm."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "refining-target-from-wrist-closeup"
TOP_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
TOP_DOWN_FLIPPED = {"w": 0.0, "x": 0.0, "y": 1.0, "z": 0.0}
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
# The D405's 640x480 intrinsics at a quarter of the resolution.
K = np.array([[98.5, 0.0, 80.0], [0.0, 98.4, 60.0], [0.0, 0.0, 1.0]])
HW = (120, 160)
# A 30 mm cube on a table at z = 0, and a second one 120 mm away along +y.
CUBE = ((0.40, 0.10, 0.015), (0.03, 0.03, 0.03))
OTHER = ((0.40, 0.22, 0.015), (0.03, 0.03, 0.03))
TRUE_TOP = np.array([0.40, 0.10, 0.03])
# The overhead estimate, 15 mm and -10 mm off.
COARSE = {
    "center": {"x": 0.415, "y": 0.09, "z": 0.015},
    "extent": {"x": 0.015, "y": 0.015, "z": 0.015},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["refine_from_wrist"].module


def _rot(q):
    w, x, y, z = (q[k] for k in "wxyz")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat(r):
    w = np.sqrt(max(1e-12, 1.0 + np.trace(r))) / 2.0
    if w > 1e-3:
        return {
            "w": w,
            "x": (r[2, 1] - r[1, 2]) / (4 * w),
            "y": (r[0, 2] - r[2, 0]) / (4 * w),
            "z": (r[1, 0] - r[0, 1]) / (4 * w),
        }
    x = np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2
    x, y, z = (
        x,
        np.copysign(y, r[0, 1] + r[1, 0]) if x > 0 else y,
        np.copysign(z, r[0, 2] + r[2, 0]) if x > 0 else z,
    )
    return {"w": 0.0, "x": x, "y": y, "z": z}


def _pose(m):
    return {
        "position": dict(zip("xyz", map(float, m[:3, 3]), strict=True)),
        "rotation": _quat(m[:3, :3]),
    }


def render(cam_world, boxes, depth_scale=1.0):
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
        rgb[hit] = (220, 40, 40)
    depth[~np.isfinite(depth)] = 0.0
    return rgb, (depth * depth_scale).astype(np.float32)


class Rig:
    """A fake arm whose moves land where commanded, with a wrist camera on its tool."""

    def __init__(self, start=(0.30, 0.0, 0.30), rotation=TOP_DOWN, refuse=None, ik=None, land=None):
        self.tool = np.eye(4)
        self.tool[:3, :3] = _rot(rotation)
        self.tool[:3, 3] = start
        self.refuse = refuse or (lambda pose: None)
        self.ik = ik or (lambda pose: {"success": True, "within_servo_tolerance": True})
        self.land = land
        self.moves = []
        self.boxes = [CUBE]
        self.wrist_name = "left_eye_in_hand"
        self.depth_scale = 1.0
        self.camera_world = None

    def get_observation(self, **kw):
        cam = self.tool @ TOOL_CAM
        self.camera_world = cam
        rgb, depth = render(cam, self.boxes, self.depth_scale)
        top = np.eye(4)
        top[:3, :3] = _rot({"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0})
        top[:3, 3] = (0.4, 0.0, 1.0)
        cameras = [{"name": "top", "rgb": rgb, "depth": depth, "intrinsics": K, "pose": _pose(top)}]
        if self.wrist_name:
            cameras.append(
                {
                    "name": self.wrist_name,
                    "rgb": rgb,
                    "depth": depth,
                    "intrinsics": K,
                    "pose": _pose(cam),
                }
            )
        return {"cameras": cameras, "arms": [{"ee_pose": _pose(self.tool)}]}

    def go_to_pose(self, pose, **kw):
        target = np.eye(4)
        target[:3, :3] = _rot(pose["rotation"])
        target[:3, 3] = [pose["position"][k] for k in "xyz"]
        self.moves.append((target.copy(), kw))
        why = self.refuse(pose)
        if isinstance(why, Exception):
            raise why
        if why:
            return {"reached": False, "outcome": "unreachable", "reason": why}
        self.tool = self.land(target) if self.land else target
        return {"reached": True, "outcome": "reached"}

    def point_at(self, world):
        def point(image, query, **kw):
            self.point_calls.append((query, kw))
            pc = np.linalg.inv(self.camera_world) @ np.append(world, 1.0)
            u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
            v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
            p = {"pixel_x": float(u), "pixel_y": float(v), "label": query}
            return {"pixel_x": p["pixel_x"], "pixel_y": p["pixel_y"], "found": True, "points": [p]}

        return point

    def ctx(self, pointer=None, pointer_name="gemini-er.point", frame=True):
        self.point_calls = []
        tools = {
            "robot.get_observation": self.get_observation,
            "robot.get_ee_pose": lambda **kw: {"pose": _pose(self.tool)},
            "robot.go_to_pose": self.go_to_pose,
            "robot.solve_ik": lambda pose, **kw: self.ik(pose),
            pointer_name: pointer or self.point_at(TRUE_TOP),
        }
        if frame:
            tools["robot.grasp_frame"] = {
                "rotation": TOP_DOWN,
                "rotation_flipped": TOP_DOWN_FLIPPED,
            }
        else:

            def missing(**kw):
                raise ToolError("robot.grasp_frame", "not served")

            tools["robot.grasp_frame"] = missing
        return FakeContext(tools)


def _camera_axis(tool):
    return (tool @ TOOL_CAM)[:3, 2]


def test_refines_a_coarse_estimate_looking_straight_down(script):
    rig = Rig()
    out = script.run(rig.ctx(), object_name="red cube", target_obb=COARSE)
    assert out["route"] == "refined", out["detail"]
    np.testing.assert_allclose([out["refined_point"][k] for k in "xyz"], TRUE_TOP, atol=0.002)
    assert out["camera_name"] == "left_eye_in_hand"
    assert out["camera_range"] == pytest.approx(0.20)
    assert out["depth"] == pytest.approx(0.20, abs=0.003)
    # One move: the hand starts above the view pose, so no rise.
    assert len(rig.moves) == 1
    np.testing.assert_allclose(_camera_axis(rig.tool), [0.0, 0.0, -1.0], atol=1e-6)
    # The camera's axis passes through the coarse top, 0.20 m away.
    eye = (rig.tool @ TOOL_CAM)[:3, 3]
    np.testing.assert_allclose(eye, [0.415, 0.09, 0.03 + 0.20], atol=1e-6)
    assert out["drift"] == pytest.approx(np.hypot(0.015, 0.01), abs=0.002)
    c = out["refined_obb"]["center"]
    assert (c["x"], c["y"], c["z"]) == (
        pytest.approx(0.40, abs=0.002),
        pytest.approx(0.10, abs=0.002),
        0.015,
    )
    assert out["refined_obb"]["extent"] == COARSE["extent"]
    query, kwargs = rig.point_calls[0]
    assert query == "the centre of the top face of the red cube"
    assert kwargs == {"max_points": 3}


def test_rises_in_place_first_from_below(script):
    rig = Rig(start=(0.30, 0.0, 0.08))
    out = script.run(rig.ctx(), object_name="red cube", target_obb=COARSE)
    assert out["route"] == "refined", out["detail"]
    rise, view = (m[0] for m in rig.moves)
    np.testing.assert_allclose(rise[:3, 3], [0.30, 0.0, view[2, 3]], atol=1e-9)
    np.testing.assert_allclose(rise[:3, :3], _rot(TOP_DOWN), atol=1e-9)


def test_falls_back_to_the_oblique_view_when_looking_down_is_out_of_reach(script):
    def ik(pose):
        tilted = abs(_rot(pose["rotation"])[2, 2] + 1.0) > 1e-3
        return {"success": not tilted}

    rig = Rig(ik=ik)
    ctx = rig.ctx()
    out = script.run(ctx, object_name="red cube", target_obb=COARSE)
    assert out["route"] == "refined", out["detail"]
    np.testing.assert_allclose(rig.tool[:3, :3], _rot(TOP_DOWN), atol=1e-9)
    # Oblique: the camera looks 45 deg down; the refined point is as good.
    assert _camera_axis(rig.tool)[2] == pytest.approx(-0.70, abs=0.02)
    np.testing.assert_allclose([out["refined_point"][k] for k in "xyz"], TRUE_TOP, atol=0.002)
    # The look-down view's ranges (0.15 m brings the hand too close) were previewed before the first move.
    tilted = [
        c
        for c in ctx.calls_to("robot.solve_ik")
        if abs(_rot(c.kwargs["pose"]["rotation"])[2, 2] + 1) > 1e-3
    ]
    assert len(tilted) == 3
    assert len(rig.moves) == 1


def test_skips_ranges_that_bring_the_hand_too_close(script):
    rig = Rig()
    out = script.run(rig.ctx(), object_name="red cube", target_obb=COARSE, camera_range=0.08)
    assert out["route"] == "refined", out["detail"]
    assert out["camera_range"] == pytest.approx(0.20)
    for target, _ in rig.moves:
        assert target[2, 3] >= 0.03 + 0.05 - 1e-9


def test_routed_and_raised_refusals_end_unreachable_after_three_moves(script):
    for refusal in ("joint limit", ToolError("robot.go_to_pose", "IK failed")):
        rig = Rig(refuse=lambda pose, r=refusal: r)
        out = script.run(rig.ctx(), object_name="red cube", target_obb=COARSE)
        assert out["route"] == "unreachable"
        assert len(rig.moves) == 3
        assert "IK failed" in out["detail"] or "joint limit" in out["detail"]
        assert out["refined_point"] == out["coarse_point"]


def test_ik_preview_refusing_everything_never_moves(script):
    rig = Rig(ik=lambda pose: {"success": False})
    out = script.run(rig.ctx(frame=False), object_name="red cube", target_obb=COARSE)
    assert out["route"] == "unreachable"
    assert rig.moves == []
    assert "IK preview" in out["detail"]


def test_a_hand_that_lands_unturned_still_back_projects_through_the_camera_it_reached(script):
    # A twin whose IK leaves the wrist free: the hand arrives keeping its start rotation.
    def land(target):
        kept = target.copy()
        kept[:3, :3] = _rot(TOP_DOWN)
        return kept

    rig = Rig(land=land)
    out = script.run(rig.ctx(), object_name="red cube", target_obb=COARSE)
    assert out["route"] == "refined", out["detail"]
    assert len(rig.moves) == 1
    np.testing.assert_allclose([out["refined_point"][k] for k in "xyz"], TRUE_TOP, atol=0.002)


def test_a_view_that_misses_the_target_takes_the_next(script):
    # The first move lands with the wrist turned half a turn: the camera faces away.
    def land(target):
        if len(rig.moves) > 1:
            return target
        turned = target.copy()
        turned[:3, :3] = target[:3, :3] @ np.diag([-1.0, -1.0, 1.0])
        return turned

    rig = Rig(land=land)
    out = script.run(rig.ctx(), object_name="red cube", target_obb=COARSE)
    assert out["route"] == "refined", out["detail"]
    assert len(rig.moves) == 2
    np.testing.assert_allclose([out["refined_point"][k] for k in "xyz"], TRUE_TOP, atol=0.002)


def test_waits_for_the_hand_to_settle_before_using_a_frame(script):
    class Settling(Rig):
        def __init__(self):
            super().__init__()
            self.wobble = []

        def go_to_pose(self, pose, **kw):
            reply = super().go_to_pose(pose, **kw)
            self.wobble = [0.004, 0.002, 0.0]
            return reply

        def get_observation(self, **kw):
            if self.wobble:
                self.tool[0, 3] += self.wobble.pop(0)
            return super().get_observation()

    rig = Settling()
    ctx = rig.ctx()
    out = script.run(ctx, object_name="red cube", target_obb=COARSE)
    assert out["route"] == "refined", out["detail"]
    # The opening read; after the move, two reads 2 mm apart, then a third that agrees.
    assert len(ctx.calls_to("robot.get_observation")) == 1 + 3
    assert rig.wobble == []
    np.testing.assert_allclose([out["refined_point"][k] for k in "xyz"], TRUE_TOP, atol=0.002)


def test_a_hand_that_ends_off_is_not_trusted(script):
    def land(target):
        off = target.copy()
        off[2, 3] += 0.03
        return off

    rig = Rig(land=land)
    out = script.run(rig.ctx(), object_name="red cube", target_obb=COARSE)
    assert out["route"] == "unreachable"
    assert "30 mm from the view pose" in out["detail"]


def test_no_wrist_camera_is_not_seen_without_moving(script):
    rig = Rig()
    rig.wrist_name = ""
    out = script.run(rig.ctx(), object_name="red cube", target_obb=COARSE)
    assert out["route"] == "not_seen"
    assert "no wrist camera" in out["detail"] and "top" in out["detail"]
    assert rig.moves == []


def test_a_named_camera_is_used(script):
    rig = Rig()
    rig.wrist_name = "d405"
    out = script.run(rig.ctx(), object_name="red cube", target_obb=COARSE, camera="d405")
    assert out["route"] == "refined", out["detail"]
    assert out["camera_name"] == "d405"


def test_pointer_finding_nothing_or_failing_is_not_seen(script):
    rig = Rig()

    def nothing(image, query, **kw):
        return {"pixel_x": 0.0, "pixel_y": 0.0, "found": False, "points": []}

    out = script.run(rig.ctx(pointer=nothing), object_name="red cube", target_obb=COARSE)
    assert out["route"] == "not_seen" and "found no" in out["detail"]

    def broken(image, query, **kw):
        raise ToolError("gemini-er.point", "quota")

    out = script.run(Rig().ctx(pointer=broken), object_name="red cube", target_obb=COARSE)
    assert out["route"] == "not_seen" and "quota" in out["detail"]


def test_the_point_nearest_the_coarse_estimate_wins(script):
    rig = Rig()
    rig.boxes = [CUBE, OTHER]
    one, two = rig.point_at(np.array([0.40, 0.22, 0.03])), rig.point_at(TRUE_TOP)

    def both(image, query, **kw):
        a, b = one(image, query, **kw), two(image, query, **kw)
        return {**a, "points": a["points"] + b["points"]}

    out = script.run(rig.ctx(pointer=both), object_name="red cube", target_obb=COARSE)
    assert out["route"] == "refined", out["detail"]
    np.testing.assert_allclose([out["refined_point"][k] for k in "xyz"], TRUE_TOP, atol=0.002)


def test_a_point_on_another_object_is_refused(script):
    rig = Rig()
    rig.boxes = [CUBE, OTHER]
    out = script.run(
        rig.ctx(pointer=rig.point_at(np.array([0.40, 0.22, 0.03]))),
        object_name="red cube",
        target_obb=COARSE,
    )
    assert out["route"] == "not_seen"
    assert "likely another object" in out["detail"]
    assert out["refined_obb"] == COARSE
    assert out["refined_point"]["y"] == pytest.approx(0.22, abs=0.003)


def test_no_depth_round_the_point_is_not_seen(script):
    rig = Rig()
    rig.depth_scale = 0.0
    out = script.run(rig.ctx(), object_name="red cube", target_obb=COARSE)
    assert out["route"] == "not_seen" and "depth readings" in out["detail"]


def test_molmo_and_a_point_target(script):
    rig = Rig()
    ctx = rig.ctx(pointer_name="molmo.point_prompt")
    point = {"x": 0.41, "y": 0.095, "z": 0.03}
    out = script.run(
        ctx,
        object_name="red cube",
        target_point=point,
        pointer="molmo.point_prompt",
        query="the red cube",
    )
    assert out["route"] == "refined", out["detail"]
    assert rig.point_calls[0] == ("the red cube", {})
    assert out["refined_obb"] == {}
    np.testing.assert_allclose([out["refined_point"][k] for k in "xyz"], TRUE_TOP, atol=0.002)
    out = script.run(
        Rig().ctx(pointer_name="molmo.point_prompt"),
        object_name="red cube",
        target_point=point,
        pointer="molmo.point_prompt",
        max_drift=0.005,
    )
    assert out["route"] == "not_seen"


def test_connector_knobs_pass_only_when_set(script):
    rig = Rig()
    script.run(rig.ctx(), object_name="red cube", target_obb=COARSE)
    assert rig.moves[0][1] == {}

    class TwoArms(Rig):
        def get_observation(self, **kw):
            o = super().get_observation()
            o["arms"] = [{"ee_pose": _pose(np.eye(4))}, o["arms"][0]]
            return o

    rig = TwoArms()
    ctx = rig.ctx()
    out = script.run(
        ctx, object_name="red cube", target_obb=COARSE, arm_id=1, tolerance=0.004, max_steps=900
    )
    assert out["route"] == "refined", out["detail"]
    assert rig.moves[0][1] == {"arm_id": 1, "tolerance": 0.004, "max_steps": 900}
    assert ctx.calls_to("robot.solve_ik")[0].kwargs["arm_id"] == 1


def test_a_given_rotation_is_the_only_one_tried(script):
    rig = Rig(refuse=lambda pose: "no")
    out = script.run(
        rig.ctx(), object_name="red cube", target_obb=COARSE, rotation=TOP_DOWN_FLIPPED
    )
    assert out["route"] == "unreachable"
    for target, _ in rig.moves:
        np.testing.assert_allclose(target[:3, :3], _rot(TOP_DOWN_FLIPPED), atol=1e-9)


def test_needs_a_target(script):
    with pytest.raises(ValueError):
        script.run(Rig().ctx(), object_name="red cube")
