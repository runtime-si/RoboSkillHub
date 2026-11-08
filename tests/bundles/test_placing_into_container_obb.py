"""placing-into-container-obb: FakeContext tests of the opening fit, the footprint room, the
drop candidates and every exit."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "placing-into-container-obb"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
BOX_C = np.array([0.45, 0.0])
FLOOR = 0.003
RIM = 0.040
# A 30 mm cube held with its centre 7 mm under the tool centre at (0.30, 0.10, 0.10).
CUBE = {
    "center": {"x": 0.30, "y": 0.10, "z": 0.093},
    "extent": {"x": 0.015, "y": 0.015, "z": 0.015},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}
HANG = 0.10 - (0.093 - 0.015)  # tool centre over the cube's bottom: 22 mm
OBB = {
    "center": {"x": 0.45, "y": 0.0, "z": 0.02},
    "extent": {"x": 0.05, "y": 0.05, "z": 0.02},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}


def box_cloud(outer=0.05, wall=0.003, floor=FLOOR, rim=RIM, step=0.002):
    """An overhead view of an open 100 x 100 x 40 mm box: rim tops, inner walls, floor."""
    pts = []
    g = np.arange(-outer, outer + 1e-9, step)
    inner = outer - wall
    for x in g:
        for y in g:
            if max(abs(x), abs(y)) > inner:
                pts.append((x, y, rim))
            elif max(abs(x), abs(y)) < inner - 0.004:
                pts.append((x, y, floor))
    for z in np.arange(floor, rim, 0.003):
        for t in np.arange(-inner, inner + 1e-9, 0.004):
            pts += [(inner, t, z), (-inner, t, z), (t, inner, z), (t, -inner, z)]
    p = np.array(pts, dtype=float)
    p[:, 0] += BOX_C[0]
    p[:, 1] += BOX_C[1]
    return p


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["place_into_container"].module


def frame():
    """A 40 x 50 image and a container mask over 10% of it."""
    rgb = np.zeros((40, 50, 3), dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[:4, :] = 255
    return rgb, mask


class Arm:
    """Moves land where commanded unless ``refuse`` (tool, target) says why not, or
    ``stop_z`` (target) gives a height the hand stops at on the way down."""

    def __init__(self, start=(0.30, 0.10, 0.10), refuse=None, stop_z=None, raises=False):
        self.p = np.array(start, dtype=float)
        self.rot = DOWN
        self.refuse = refuse or (lambda tool, target: None)
        self.stop_z = stop_z or (lambda target: None)
        self.raises = raises
        self.moves = []

    def _move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy(), kwargs))
            why = self.refuse(tool, target)
            if why:
                if self.raises:
                    raise ToolError(tool, why)
                return {"reached": False, "outcome": "unreachable", "reason": why}
            stop = self.stop_z(target)
            if stop is not None and tool.endswith("cartesian") and target[2] < stop:
                self.p = np.array([target[0], target[1], stop])
                return {"reached": False, "outcome": "contact"}
            self.p, self.rot = target, pose["rotation"]
            return {"reached": True, "outcome": "reached"}

        return go

    def ctx(self, *, cloud=None, masks=None, scores=None, grip=0.3, opened=1.0, ik=None, **extra):
        rgb, mask = frame()
        cloud = box_cloud() if cloud is None else cloud
        tools = {
            "robot.get_ee_pose": lambda **kw: {
                "pose": {
                    "position": dict(zip("xyz", map(float, self.p), strict=True)),
                    "rotation": self.rot,
                }
            },
            "robot.get_gripper": {"position": grip},
            "robot.go_to_pose": self._move("robot.go_to_pose"),
            "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
            "robot.open_gripper": {"position": opened},
            "robot.solve_ik": ik or {"success": True, "within_servo_tolerance": True},
            "robot.get_observation": {
                "cameras": [
                    {
                        "name": "top",
                        "rgb": rgb,
                        "depth": np.ones((40, 50), dtype=np.float32),
                        "intrinsics": np.eye(3),
                        "pose": {"position": {"x": 0, "y": 0, "z": 1}, "rotation": DOWN},
                    }
                ]
            },
            "sam3.segment_text": {
                "masks": [mask] if masks is None else masks,
                "scores": [0.6] if scores is None else scores,
            },
            "geometry.mask_to_world_points": {"points": {"points": cloud}},
            "geometry.filter_noise": lambda points, **kw: {"points": {"points": points}},
            "geometry.compute_obb": {"obb": OBB},
        }
        tools.update(extra)
        return FakeContext(tools)


def test_places_the_cube_in_the_box_centre(script):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(ctx, container_description="open box", held_obb=CUBE)
    assert out["route"] == "placed", out["detail"]
    assert out["candidate"] == "centre"
    assert out["opening_source"] == "rim"
    assert out["rim_z"] == pytest.approx(RIM)
    assert out["floor_z"] == pytest.approx(FLOOR)
    assert out["aperture_center"]["x"] == pytest.approx(0.45, abs=1e-3)
    assert out["aperture_center"]["y"] == pytest.approx(0.0, abs=1e-3)
    assert out["aperture_center"]["z"] == pytest.approx(RIM)
    # Half the rim band's 10th-90th percentile span (47 mm) less a 3 mm wall.
    ext = out["opening_obb"]["extent"]
    assert [ext["x"], ext["y"]] == pytest.approx([0.044, 0.044], abs=0.002)
    assert ext["z"] == pytest.approx((RIM - FLOOR) / 2)
    assert out["opening_obb"]["center"]["z"] == pytest.approx((RIM + FLOOR) / 2)
    names = [m[0] for m in arm.moves]
    assert names == [
        "robot.go_to_pose_cartesian",  # lift
        "robot.go_to_pose",  # over the opening centre
        "robot.go_to_pose_cartesian",  # straight down
        "robot.go_to_pose_cartesian",  # retreat
    ]
    lift, above, release, retreat = (m[1] for m in arm.moves)
    np.testing.assert_allclose(lift, [0.30, 0.10, 0.20], atol=1e-9)
    # The cube's centre is on the tool centre in xy, so the hand goes over the opening centre,
    # the cube's bottom 30 mm over the rim.
    np.testing.assert_allclose(above, [0.45, 0.0, RIM + 0.03 + HANG], atol=1e-3)
    np.testing.assert_allclose(release, [0.45, 0.0, FLOOR + 0.01 + HANG], atol=1e-3)
    np.testing.assert_allclose(retreat, release + [0, 0, 0.08], atol=1e-9)
    assert out["release_position"]["z"] == pytest.approx(FLOOR + 0.01, abs=1e-6)
    calls = [c.tool for c in ctx.calls]
    assert calls.index("robot.get_observation") > calls.index("robot.go_to_pose_cartesian")
    assert calls.index("robot.open_gripper") > calls.index("robot.go_to_pose")
    assert all(m[2] == {} for m in arm.moves)  # no arm_id on arm 0


def test_the_object_offset_in_the_hand_shifts_the_hand(script):
    off = {**CUBE, "center": {"x": 0.31, "y": 0.10, "z": 0.093}}
    arm = Arm()
    out = script.run(arm.ctx(), container_description="open box", held_obb=off)
    assert out["route"] == "placed"
    above = arm.moves[1][1]
    assert above[0] == pytest.approx(0.45 - 0.01, abs=1e-3)
    assert out["release_position"]["x"] == pytest.approx(0.45, abs=1e-3)


def test_not_holding_moves_nothing(script):
    arm = Arm()
    out = script.run(arm.ctx(grip=0.01), container_description="open box", held_obb=CUBE)
    assert out["route"] == "not_holding"
    assert arm.moves == []


def test_an_unreadable_hand_is_not_an_empty_one(script):
    def unreadable(**kw):
        raise ToolError("robot.get_gripper", "not served")

    arm = Arm()
    ctx = arm.ctx(**{"robot.get_gripper": unreadable})
    ctx._responses["robot.open_gripper"] = {}
    out = script.run(ctx, container_description="open box", held_obb=CUBE)
    assert out["route"] == "placed", out["detail"]


def test_no_room_for_a_wide_object(script):
    wide = {**CUBE, "extent": {"x": 0.045, "y": 0.015, "z": 0.015}}
    arm = Arm()
    out = script.run(arm.ctx(), container_description="open box", held_obb=wide)
    assert out["route"] == "no_room", out["detail"]
    assert [m[0] for m in arm.moves] == ["robot.go_to_pose_cartesian"]  # the lift only
    assert out["aperture_center"]  # what was seen is still reported


def test_a_turned_object_needs_its_diagonal(script):
    # A 60 x 20 mm bar turned 45 deg reaches (30 + 10) / sqrt(2) = 28 mm along each box axis.
    c = np.cos(np.pi / 8)
    s = np.sin(np.pi / 8)
    bar = {
        **CUBE,
        "extent": {"x": 0.03, "y": 0.01, "z": 0.015},
        "orientation": {"w": c, "x": 0, "y": 0, "z": s},
    }
    assert script.footprint_reach(bar, np.array([1.0, 0.0, 0.0])) == pytest.approx(
        0.04 / np.sqrt(2)
    )
    out = script.run(
        Arm().ctx(), container_description="open box", held_obb=bar, footprint_margin=0.02
    )
    assert out["route"] == "no_room"  # 28 + 20 > 44
    out = script.run(Arm().ctx(), container_description="open box", held_obb=bar)
    assert out["route"] == "placed"


@pytest.mark.parametrize(
    "masks,scores,why",
    [
        ([], [], "found no"),
        (None, [0.1], "best score"),
        ("whole", [0.9], "of the frame"),
    ],
)
def test_container_not_found(script, masks, scores, why):
    if masks == "whole":
        masks = [np.full((40, 50), 255, dtype=np.uint8)]
    arm = Arm()
    out = script.run(
        arm.ctx(masks=masks, scores=scores), container_description="bin", held_obb=CUBE
    )
    assert out["route"] == "container_not_found"
    assert why in out["detail"]


def test_too_few_depth_points_is_not_found(script):
    out = script.run(Arm().ctx(cloud=box_cloud()[:10]), container_description="bin", held_obb=CUBE)
    assert out["route"] == "container_not_found"
    assert "depth points" in out["detail"]


def test_no_camera_is_not_found(script):
    arm = Arm()
    out = script.run(arm.ctx(), container_description="bin", held_obb=CUBE, camera_name="wrist")
    assert out["route"] == "container_not_found"


def test_container_pixel_picks_the_mask_under_it(script):
    a = np.zeros((40, 50), dtype=np.uint8)
    a[:4, :] = 255
    b = np.zeros((40, 50), dtype=np.uint8)
    b[30:34, :] = 255
    arm = Arm()
    ctx = arm.ctx(masks=[a, b], scores=[0.9, 0.05])
    out = script.run(ctx, container_description="bin", held_obb=CUBE, container_pixel=[10, 31])
    assert out["route"] == "placed"
    sent = ctx.calls_to("geometry.mask_to_world_points")[0].kwargs["mask"]
    assert sent[31, 10] == 255 and sent[0, 0] == 0
    out = script.run(
        Arm().ctx(masks=[a, b]),
        container_description="bin",
        held_obb=CUBE,
        container_pixel=[10, 20],
    )
    assert out["route"] == "container_not_found"


def test_a_descent_stopped_on_the_rim_backs_up_and_tries_the_next_point(script):
    # The first descent (over the centre) stops 30 mm up; the next candidate goes down.
    stops = [0.06]
    arm = Arm(stop_z=lambda t: stops.pop(0) if stops and t[2] < 0.05 else None)
    out = script.run(arm.ctx(), container_description="open box", held_obb=CUBE)
    assert out["route"] == "placed", out["detail"]
    assert out["candidate"] == "-u"
    assert "before: centre: descent" in out["detail"]
    names = [m[0] for m in arm.moves]
    assert names[:4] == [
        "robot.go_to_pose_cartesian",
        "robot.go_to_pose",
        "robot.go_to_pose_cartesian",  # stopped
        "robot.go_to_pose_cartesian",  # back up
    ]
    # -u: within the room (44 - 15 - 8 = 21 mm), short of half-way to the inner edge (22 mm).
    assert arm.moves[4][1][0] == pytest.approx(0.45 - 0.021, abs=1.5e-3)


def test_unreachable_when_no_drop_point_previews(script):
    arm = Arm()
    ctx = arm.ctx(ik={"success": True, "within_servo_tolerance": False})
    out = script.run(ctx, container_description="open box", held_obb=CUBE)
    assert out["route"] == "unreachable"
    assert out["detail"].count("IK preview") == 5
    assert [m[0] for m in arm.moves] == ["robot.go_to_pose_cartesian"]
    assert ctx.call_count("robot.open_gripper") == 0


@pytest.mark.parametrize("raises", [False, True])
def test_refused_moves_are_routed(script, raises):
    arm = Arm(refuse=lambda tool, t: "no IK" if tool == "robot.go_to_pose" else None, raises=raises)
    ctx = arm.ctx()
    out = script.run(ctx, container_description="open box", held_obb=CUBE)
    assert out["route"] == "unreachable"
    assert "no IK" in out["detail"]
    assert ctx.call_count("robot.open_gripper") == 0


def test_a_refused_lift_is_noted_and_the_place_goes_on(script):
    arm = Arm(refuse=lambda tool, t: "joint limit" if t[2] > 0.15 else None)
    out = script.run(arm.ctx(), container_description="open box", held_obb=CUBE)
    assert out["route"] == "placed"
    assert "lift: unreachable: joint limit" in out["detail"]
    assert arm.moves[1][1][2] == pytest.approx(RIM + 0.03 + HANG)


def test_jaws_that_stay_closed_are_not_released(script):
    arm = Arm()
    out = script.run(arm.ctx(opened=0.2), container_description="open box", held_obb=CUBE)
    assert out["route"] == "not_released"
    assert [m[0] for m in arm.moves][-1] == "robot.go_to_pose_cartesian"  # no retreat after it
    assert len(arm.moves) == 3


def test_tool_centre_floor_raises_a_short_hang(script):
    # The cube's bottom hangs 3 mm under the tool centre: the hand stops at floor + 15 mm
    # (the object's bottom 12 mm up), not floor + 10 + 3 mm.
    short = {**CUBE, "center": {"x": 0.30, "y": 0.10, "z": 0.112}}
    arm = Arm()
    out = script.run(arm.ctx(), container_description="open box", held_obb=short)
    assert out["route"] == "placed"
    assert arm.moves[2][1][2] == pytest.approx(FLOOR + 0.015)
    assert "to keep the tool centre" in out["detail"]


def test_arm_id_is_passed_when_nonzero(script):
    arm = Arm()
    ctx = arm.ctx()
    script.run(ctx, container_description="open box", held_obb=CUBE, arm_id=1)
    assert all(m[2] == {"arm_id": 1} for m in arm.moves)
    assert ctx.calls_to("robot.get_gripper")[0].kwargs == {"arm_id": 1}


def test_opening_falls_back_to_the_box_and_its_bottom(script):
    op = script.container_opening(np.zeros((5, 3)), OBB, wall_thickness=0.003)
    assert op["source"] == "obb"
    assert op["rim_z"] == pytest.approx(0.04)
    np.testing.assert_allclose(op["inner_half"], [0.047, 0.047])
    assert op["floor_source"] == "box bottom + wall"
    assert op["floor_z"] == pytest.approx(0.003)


def test_a_wall_seen_from_the_side_does_not_pull_the_opening(script):
    # A tilted camera sees the -x wall's inner face densely, inside the rim band: the band's
    # mean moves toward it, the opening's centre and size must not.
    cloud = box_cloud()
    y, z = np.meshgrid(np.arange(-0.047, 0.047, 0.001), np.arange(RIM - 0.019, RIM, 0.001))
    face = np.column_stack([np.full(y.size, BOX_C[0] - 0.047), y.ravel() + BOX_C[1], z.ravel()])
    cloud = np.vstack([cloud, face])
    rim = cloud[cloud[:, 2] >= RIM - 0.02]
    assert rim[:, 0].mean() < BOX_C[0] - 0.01
    op = script.container_opening(cloud, OBB)
    np.testing.assert_allclose(op["center_xy"], BOX_C, atol=0.002)
    np.testing.assert_allclose(op["inner_half"], [0.044, 0.044], atol=0.003)


def test_floor_is_the_top_of_the_contents(script):
    cloud = box_cloud(floor=0.02)
    op = script.container_opening(cloud, OBB)
    assert op["floor_source"] == "seen"
    assert op["floor_z"] == pytest.approx(0.02)


def test_drop_candidates_keep_the_margin(script):
    inner = np.array([0.046, 0.030])
    reach = np.array([0.015, 0.015])
    c = script.drop_candidates(inner, reach, 0.008)
    assert [n for n, *_ in c] == ["centre", "-u", "+u", "-v", "+v"]
    d = dict((n, (u, v)) for n, u, v in c)
    assert d["+u"] == pytest.approx((0.023, 0.0))  # half-way: 23 <= room 23
    assert d["+v"] == pytest.approx((0.0, 0.007))  # pulled in to the room: 30 - 15 - 8
    assert script.drop_candidates(inner, np.array([0.015, 0.025]), 0.008) == []
    # No room to move sideways along v: the centre and the u points only.
    assert [n for n, *_ in script.drop_candidates(inner, np.array([0.015, 0.022]), 0.008)] == [
        "centre",
        "-u",
        "+u",
    ]
