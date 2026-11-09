"""grasping-by-wrist-camera-servoing: FakeContext tests with a ray-cast wrist camera riding a fake arm."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError
from scipy import ndimage

BUNDLE = "grasping-by-wrist-camera-servoing"
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
# The D405's 640x480 intrinsics at half the resolution.
K = np.array([[197.0, 0.0, 160.0], [0.0, 196.8, 120.0], [0.0, 0.0, 1.0]])
HW = (240, 320)
SIDE = 0.025
CUBE = np.array([0.40, 0.10, SIDE / 2])
RED = (220, 40, 40)
SPAN, CLOSED = 0.094, 0.0


def coarse_obb(dx=0.0, dy=0.0, side=SIDE, centre=CUBE):
    return {
        "center": {"x": centre[0] + dx, "y": centre[1] + dy, "z": side / 2},
        "extent": {"x": side / 2, "y": side / 2, "z": side / 2},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["servo_grasp"].module


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


def render(cam_world, boxes):
    """rgb and z-depth of a table at z = 0 and axis-aligned red boxes, seen from ``cam_world``."""
    h, w = HW
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dirs_cam = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
    dirs = dirs_cam @ cam_world[:3, :3].T
    origin = cam_world[:3, 3]
    depth = np.full((h, w), np.inf)
    rgb = np.full((h, w, 3), 110, np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -origin[2] / dirs[..., 2]
    table = np.isfinite(t) & (t > 0)
    depth[table] = t[table]
    for centre, size in boxes:
        lo, hi = np.array(centre) - np.array(size) / 2, np.array(centre) + np.array(size) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - origin) / dirs, (hi - origin) / dirs
        tmin, tmax = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
        hit = (tmax >= tmin) & (tmin > 0) & (tmin < depth)
        depth[hit] = tmin[hit]
        rgb[hit] = RED
    depth[~np.isfinite(depth)] = 0.0
    return rgb, depth.astype(np.float32)


def red_segmenter(calls=None, blind=lambda n: False):
    """A sam3.segment_text stand-in: one mask per connected red blob, biggest first."""

    def segment(image, query, max_results=5, **kw):
        n = len(calls) if calls is not None else 0
        if calls is not None:
            calls.append(query)
        if blind(n):
            return {"masks": [], "scores": [], "boxes": []}
        red = np.all(np.asarray(image) == RED, axis=-1)
        labels, count = ndimage.label(red)
        masks = sorted(
            ((labels == i).astype(np.uint8) * 255 for i in range(1, count + 1)),
            key=lambda m: -int(m.sum()),
        )[:max_results]
        return {"masks": masks, "scores": [0.9] * len(masks), "boxes": []}

    return segment


class Rig:
    """A fake arm (moves land where commanded unless told otherwise) with a wrist camera
    and parallel jaws that hold a cube only when it lies between the pads."""

    def __init__(
        self, start=(0.30, 0.10, 0.25), cubes=None, side=SIDE, refuse=None, ik=None, land=None
    ):
        self.tool = np.eye(4)
        self.tool[:3, :3] = _rot(TOP_DOWN)
        self.tool[:3, 3] = start
        self.side = side
        self.cubes = [np.array(c) for c in (cubes or [CUBE])]
        self.refuse = refuse or (lambda tool, pose: None)
        self.ik = ik or (lambda pose: {"success": True, "within_servo_tolerance": True})
        self.land = land
        self.moves = []
        self.opening = SPAN
        self.grip_calls = []
        self.wrist = True
        self.slips = False

    def get_observation(self, **kw):
        cam = self.tool @ TOOL_CAM
        rgb, depth = render(cam, [(c, (self.side,) * 3) for c in self.cubes])
        cameras = []
        if self.wrist:
            cameras.append(
                {
                    "name": "left_eye_in_hand",
                    "rgb": rgb,
                    "depth": depth,
                    "intrinsics": K,
                    "pose": _pose(cam),
                }
            )
        return {"cameras": cameras, "arms": [{"ee_pose": _pose(self.tool)}]}

    def move(self, tool, pose, **kw):
        target = np.eye(4)
        target[:3, :3] = _rot(pose["rotation"])
        target[:3, 3] = [pose["position"][k] for k in "xyz"]
        self.moves.append((tool, target.copy(), kw))
        why = self.refuse(tool, pose)
        if isinstance(why, Exception):
            raise why
        if why:
            return {"reached": False, "outcome": "unreachable", "reason": why}
        self.tool = self.land(self.tool, target) if self.land else target
        return {"reached": True, "outcome": "reached"}

    def held(self):
        p = self.tool[:3, 3]
        for c in self.cubes:
            off = np.linalg.inv(self.tool[:3, :3]) @ (c - p)
            if (
                abs(off[0]) <= self.opening / 2 - self.side / 2
                and abs(off[1]) <= self.side / 2
                and p[2] <= c[2] + self.side / 2 - 0.003
            ):
                return True
        return False

    def close(self, **kw):
        self.grip_calls.append(("close", kw))
        w = self.side if self.held() else CLOSED
        self.opening = w
        return {"position": (w - CLOSED) / (SPAN - CLOSED)}

    def get_gripper(self, **kw):
        if self.slips:
            self.opening = CLOSED
        return {"position": (self.opening - CLOSED) / (SPAN - CLOSED)}

    def set_grip(self, width_m, **kw):
        self.grip_calls.append(("set_grip", width_m))
        self.opening = width_m
        return {"ok": True}

    def open(self, **kw):
        self.grip_calls.append(("open", kw))
        self.opening = SPAN
        return {"ok": True}

    def ctx(self, segmenter=None, extra=None):
        tools = {
            "robot.get_observation": self.get_observation,
            "robot.get_ee_pose": lambda **kw: {"pose": _pose(self.tool)},
            "robot.go_to_pose": lambda pose, **kw: self.move("go_to_pose", pose, **kw),
            "robot.go_to_pose_cartesian": lambda pose, **kw: self.move("cartesian", pose, **kw),
            "robot.solve_ik": lambda pose, **kw: self.ik(pose),
            "robot.grasp_frame": {"rotation": TOP_DOWN, "rotation_flipped": TOP_DOWN_FLIPPED},
            "robot.describe_gripper": {
                "span_m": SPAN,
                "width_fit": {"at_open_m": SPAN, "at_closed_m": CLOSED},
                "finger": {"stated": True, "reach_m": 0.0102, "clearance_m": 0.002},
            },
            "robot.get_gripper": self.get_gripper,
            "robot.set_grip": self.set_grip,
            "robot.open_gripper": self.open,
            "robot.close_gripper": self.close,
            "sam3.segment_text": segmenter or red_segmenter(),
        }
        tools.update(extra or {})
        return FakeContext(tools)


def _cartesian_steps(rig):
    return [m for m in rig.moves if m[0] == "cartesian"]


def test_servo_grasps_a_cube_the_coarse_estimate_missed_by_8_mm(script):
    rig = Rig()
    out = script.run(rig.ctx(), object_name="red cube", target_obb=coarse_obb(0.008, 0.0))
    assert out["route"] == "grasped", out["detail"]
    assert out["mode"] == "servo"
    closed_at = rig.tool[:3, 3] - np.array([0, 0, out["lifted"]])
    assert np.linalg.norm(closed_at[:2] - CUBE[:2]) <= 0.0015
    gp = np.array([out["grasp_point"][k] for k in "xyz"])
    assert np.linalg.norm(gp[:2] - CUBE[:2]) < 0.001
    assert abs(gp[2] - SIDE) < 0.001
    assert out["correction"] == pytest.approx(0.008, abs=0.001)
    assert out["final_error"] <= 0.0015
    assert out["grip_width"] == pytest.approx(SIDE)
    assert out["lifted"] == pytest.approx(0.05)
    # Jaws pre-shaped to the cube's width + 6 mm a side before the servo.
    assert ("set_grip", pytest.approx(SIDE + 0.012)) in rig.grip_calls


def test_open_loop_misses_the_same_cube_with_the_jaws_pre_shaped(script):
    rig = Rig()
    out = script.run(
        rig.ctx(), object_name="red cube", target_obb=coarse_obb(0.008, 0.0), servo=False
    )
    assert out["route"] == "missed"
    assert out["mode"] == "open_loop"
    assert ("open", {}) in rig.grip_calls[-1:]


def test_open_loop_grasps_when_the_estimate_is_good(script):
    rig = Rig()
    out = script.run(
        rig.ctx(), object_name="red cube", target_obb=coarse_obb(0.002, 0.001), servo=False
    )
    assert out["route"] == "grasped"


def test_servo_steps_are_small_and_each_previewed(script):
    previews = []

    def ik(pose):
        previews.append(pose)
        return {"success": True, "within_servo_tolerance": True}

    rig = Rig(ik=ik)
    out = script.run(rig.ctx(), object_name="red cube", target_obb=coarse_obb(-0.006, 0.005))
    assert out["route"] == "grasped", out["detail"]
    steps = _cartesian_steps(rig)
    servo_steps = steps[:-1]  # the last is the lift
    prev = None
    for _, target, _ in servo_steps:
        if prev is not None:
            d = target[:3, 3] - prev
            assert np.linalg.norm(d[:2]) <= 0.010 + 1e-9
            assert -d[2] <= 0.015 + 1e-9
        prev = target[:3, 3]
    assert len(previews) >= len(servo_steps)


def test_fingertips_stay_above_the_top_until_aligned(script):
    rig = Rig()
    out = script.run(rig.ctx(), object_name="red cube", target_obb=coarse_obb(0.008, -0.004))
    assert out["route"] == "grasped"
    top = SIDE
    for _, target, _ in _cartesian_steps(rig)[:-1]:
        p = target[:3, 3]
        if p[2] - 0.0102 < top:
            assert np.linalg.norm(p[:2] - CUBE[:2]) <= 0.0015 + 0.0005


def test_lost_track_falls_back_to_open_loop_at_the_last_seen_point(script):
    calls = []
    rig = Rig()
    seg = red_segmenter(calls, blind=lambda n: n >= 3)
    out = script.run(rig.ctx(seg), object_name="red cube", target_obb=coarse_obb(0.006, 0.0))
    assert out["mode"] == "open_loop_fallback"
    assert "lost the target" in out["fallback_reason"]
    gp = np.array([out["grasp_point"][k] for k in "xyz"])
    assert np.linalg.norm(gp[:2] - CUBE[:2]) < 0.001
    assert out["route"] == "grasped"
    assert out["lost_frames"] == 3


def test_lost_track_without_fallback_is_lost_target(script):
    rig = Rig()
    seg = red_segmenter([], blind=lambda n: n >= 2)
    out = script.run(
        rig.ctx(seg), object_name="red cube", target_obb=coarse_obb(0.006, 0.0), fallback=False
    )
    assert out["route"] == "lost_target"
    assert not any(c[0] == "close" for c in rig.grip_calls)


def test_never_seen_is_lost_target_without_descending(script):
    rig = Rig(cubes=[CUBE + np.array([0.0, 0.0, 0.0])])
    seg = red_segmenter([], blind=lambda n: True)
    out = script.run(rig.ctx(seg), object_name="red cube", target_obb=coarse_obb())
    assert out["route"] == "lost_target"
    assert out["iterations"] == 3
    assert _cartesian_steps(rig) == []


def test_another_object_far_from_the_estimate_is_not_taken(script):
    rig = Rig(cubes=[CUBE + np.array([0.0, 0.07, 0.0])])
    out = script.run(rig.ctx(), object_name="red cube", target_obb=coarse_obb())
    assert out["route"] == "lost_target"
    assert "coarse estimate" in out["detail"]


def test_the_target_is_chosen_over_a_neighbour(script):
    rig = Rig(cubes=[CUBE, CUBE + np.array([0.0, 0.045, 0.0])])
    out = script.run(rig.ctx(), object_name="red cube", target_obb=coarse_obb(0.004, 0.004))
    assert out["route"] == "grasped", out["detail"]
    gp = np.array([out["grasp_point"][k] for k in "xyz"])
    assert np.linalg.norm(gp[:2] - CUBE[:2]) < 0.001


def test_no_wrist_camera_is_lost_target_without_moving(script):
    rig = Rig()
    rig.wrist = False
    out = script.run(rig.ctx(), object_name="red cube", target_obb=coarse_obb())
    assert out["route"] == "lost_target"
    assert rig.moves == []


def test_a_stalled_arm_falls_back(script):
    def land(tool, target):
        if target[2, 3] < 0.06 and tool[2, 3] < 0.065:
            return tool  # the arm will not go lower
        return target

    rig = Rig(land=land)
    out = script.run(rig.ctx(), object_name="red cube", target_obb=coarse_obb(0.005, 0.0))
    assert out["mode"] == "open_loop_fallback"
    assert "stalled" in out["fallback_reason"]


def test_pregrasp_out_of_reach_is_unreachable_without_moving(script):
    rig = Rig(ik=lambda pose: {"success": False})
    out = script.run(rig.ctx(), object_name="red cube", target_obb=coarse_obb())
    assert out["route"] == "unreachable"
    assert rig.moves == []
    assert "IK preview" in out["detail"]


def test_refused_servo_steps_fall_back_and_a_refused_fallback_is_unreachable(script):
    def refuse(tool, pose):
        return (
            ToolError("robot.go_to_pose_cartesian", "joint limit") if tool == "cartesian" else None
        )

    rig = Rig(refuse=refuse)
    out = script.run(rig.ctx(), object_name="red cube", target_obb=coarse_obb(0.004, 0.0))
    assert out["route"] == "unreachable"
    assert out["mode"] == "open_loop_fallback"
    assert "refused at full and half length" in out["fallback_reason"]


def test_jaws_closing_on_nothing_is_missed_and_reopened(script):
    rig = Rig()
    rig.held = lambda: False
    out = script.run(rig.ctx(), object_name="red cube", target_obb=coarse_obb(0.003, 0.0))
    assert out["route"] == "missed"
    assert "nothing" in out["detail"]
    assert rig.grip_calls[-1][0] == "open"


def test_a_part_slipping_out_in_the_lift_is_missed(script):
    rig = Rig()
    real_close = rig.close

    def close(**kw):
        r = real_close(**kw)
        rig.slips = True
        return r

    out = script.run(
        rig.ctx(extra={"robot.close_gripper": close}),
        object_name="red cube",
        target_obb=coarse_obb(0.003, 0.0),
    )
    assert out["route"] == "missed"
    assert "slipped" in out["detail"]


def test_without_set_grip_the_jaws_open_fully(script):
    rig = Rig()

    def no_grip(**kw):
        raise ToolError("robot.set_grip", "not served")

    out = script.run(
        rig.ctx(extra={"robot.set_grip": no_grip}),
        object_name="red cube",
        target_obb=coarse_obb(0.005, 0.0),
    )
    assert out["route"] == "grasped"
    assert ("open", {}) in rig.grip_calls


def test_a_short_part_is_grasped_no_lower_than_14_mm_over_the_support(script):
    side = 0.020
    cube = np.array([0.40, 0.10, side / 2])
    rig = Rig(cubes=[cube], side=side)
    out = script.run(
        rig.ctx(), object_name="red cube", target_obb=coarse_obb(0.004, 0.0, side=side, centre=cube)
    )
    assert out["route"] == "grasped", out["detail"]
    closed_z = rig.tool[2, 3] - out["lifted"]
    assert closed_z == pytest.approx(0.014, abs=0.0021)


def test_a_point_target_uses_the_default_grasp_depth_and_opens_fully(script):
    rig = Rig()
    out = script.run(
        rig.ctx(), object_name="red cube", target_point={"x": 0.405, "y": 0.10, "z": SIDE}
    )
    assert out["route"] == "grasped", out["detail"]
    assert ("open", {}) in rig.grip_calls
    closed_z = rig.tool[2, 3] - out["lifted"]
    assert closed_z == pytest.approx(SIDE - 0.012, abs=0.0021)


def test_top_face_centre_is_not_pulled_toward_the_camera(script):
    """A plain centroid of an obliquely seen cube leans toward the camera; the top band does not."""
    aim = script.aim_module()
    tool = np.eye(4)
    tool[:3, :3] = _rot(TOP_DOWN)
    tool[:3, 3] = (0.39, 0.10, 0.06)
    cam = tool @ TOOL_CAM
    rgb, depth = render(cam, [(CUBE, (SIDE,) * 3)])
    frame = {"rgb": rgb, "depth": depth, "intrinsics": K, "pose": _pose(cam)}
    mask = np.all(rgb == RED, axis=-1)
    pts = script._mask_points(frame, mask, aim.MIN_DEPTH, aim.MAX_DEPTH)
    top = script._top_face(pts, script.TOP_BAND)
    assert np.linalg.norm(top[:2] - CUBE[:2]) < 0.001
    assert abs(top[2] - SIDE) < 0.0005
    assert np.linalg.norm(np.median(pts, axis=0)[:2] - CUBE[:2]) > 0.002


def test_connector_knobs_pass_only_when_set(script):
    rig = Rig()
    script.run(rig.ctx(), object_name="red cube", target_obb=coarse_obb(0.002, 0.0))
    assert all(kw == {} for _, _, kw in rig.moves)
    rig = Rig()
    script.run(
        rig.ctx(),
        object_name="red cube",
        target_obb=coarse_obb(0.002, 0.0),
        tolerance=0.004,
        max_steps=500,
    )
    pregrasp = [kw for t, _, kw in rig.moves if t == "go_to_pose"]
    assert pregrasp and all(kw == {"tolerance": 0.004, "max_steps": 500} for kw in pregrasp)
    assert all(kw == {} for t, _, kw in rig.moves if t == "cartesian")


def test_needs_a_target(script):
    with pytest.raises(ValueError):
        script.run(Rig().ctx(), object_name="red cube")
