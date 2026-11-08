"""following-a-visual-plan: FakeContext tests of the lifting, the checks, the replan and the motion.

The scene reproduces the real YAM station frame RoboBrain was asked about (g_sota 20261103-0511
frame_02): the calibrated overhead camera (pose and intrinsics from the station's calibration), a
table plane at z = -0.021 in the left arm's base frame, the open 100 x 100 x 40 mm box and the
30 mm cube where the twin fitted to that frame puts them. Its depth image is ray-cast here. The
plans are robobrain.trace's real answers on that frame (data/robobrain_traces_station.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "following-a-visual-plan"
DATA = json.loads(
    (Path(__file__).resolve().parent / "data" / "robobrain_traces_station.json").read_text()
)
K = np.asarray(DATA["intrinsics"], dtype=np.float64)
TABLE_Z = float(DATA["table_z"])
SCENE = DATA["scene_cube_box"]
BOX_C = np.array(SCENE["box_center_xy"])
BOX_YAW = float(SCENE["box_yaw"])
BOX_H, BOX_OUT, BOX_WALL = SCENE["box_height"], SCENE["box_outer"], SCENE["box_wall"]
CUBE_C = np.array(SCENE["cube_center_xy"])
W, H = 640, 480
KEEP = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
TOP_A = {"w": 0.0, "x": 0.7071, "y": 0.7071, "z": 0.0}
TOP_B = {"w": 0.0, "x": -0.7071, "y": 0.7071, "z": 0.0}
HAND = (0.05, 0.25, 0.06)


def traces(frame):
    return [t for t in DATA["traces"] if t["frame"] == frame]


# ---------------------------------------------------------------------------
# A ray-cast depth image of oriented boxes and a (finite) table
# ---------------------------------------------------------------------------


def _slabs(table_xy=((-1.0, 1.0), (-1.0, 1.0)), extra=()):
    """(centre, half extents, yaw) of every solid: the table top, the box's floor and walls, the cube."""
    (x0, x1), (y0, y1) = table_xy
    out = [
        (
            np.array([(x0 + x1) / 2, (y0 + y1) / 2, TABLE_Z - 0.01]),
            np.array([(x1 - x0) / 2, (y1 - y0) / 2, 0.01]),
            0.0,
        )
    ]
    h, w = BOX_OUT / 2, BOX_WALL
    parts = [
        ((0.0, 0.0, w / 2), (h, h, w / 2)),
        ((h - w / 2, 0.0, BOX_H / 2), (w / 2, h, BOX_H / 2)),
        ((-h + w / 2, 0.0, BOX_H / 2), (w / 2, h, BOX_H / 2)),
        ((0.0, h - w / 2, BOX_H / 2), (h - w, w / 2, BOX_H / 2)),
        ((0.0, -h + w / 2, BOX_H / 2), (h - w, w / 2, BOX_H / 2)),
    ]
    c, s = np.cos(BOX_YAW), np.sin(BOX_YAW)
    for (dx, dy, z), half in parts:
        at = np.array([BOX_C[0] + c * dx - s * dy, BOX_C[1] + s * dx + c * dy, TABLE_Z + z])
        out.append((at, np.array(half), BOX_YAW))
    q = SCENE["cube_size"] / 2
    out.append((np.array([CUBE_C[0], CUBE_C[1], TABLE_Z + q]), np.array([q, q, q]), 0.0))
    out.extend(extra)
    return out


def render_depth(T, solids, floor_z=TABLE_Z - 0.75):
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rays_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    rays = rays_c @ T[:3, :3].T  # world direction per unit camera z
    o = T[:3, 3]
    best = np.where(rays[..., 2] < 0, (floor_z - o[2]) / np.minimum(rays[..., 2], -1e-9), np.inf)
    for centre, half, yaw in solids:
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        lo_o = (o - centre) @ R
        d = rays @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-half - lo_o) / d
            t2 = (half - lo_o) / d
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = (tmax >= tmin) & (tmin > 0)
        best = np.where(hit & (tmin < best), tmin, best)
    return best.astype(np.float32)


def camera(frame="cube_box", solids=None, depth=None):
    T = np.asarray(DATA["camera_pose"][frame], dtype=np.float64)
    if depth is None:
        depth = render_depth(T, _slabs() if solids is None else solids)
    return {
        "name": "top",
        "rgb": np.zeros((H, W, 3), np.uint8),
        "depth": depth,
        "intrinsics": K,
        "pose": T,
    }


def project(T, p):
    cam = (np.asarray(p) - T[:3, 3]) @ T[:3, :3]
    return np.array([K[0, 0] * cam[0] / cam[2] + K[0, 2], K[1, 1] * cam[1] / cam[2] + K[1, 2]])


@pytest.fixture(scope="module")
def cube_obs():
    return {"cameras": [camera()]}


@pytest.fixture(scope="module")
def propose(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["propose_target_from_plan"].module


@pytest.fixture(scope="module")
def follow(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["follow_proposal"].module


class Arm:
    """Moves land where commanded unless ``refuse`` (tool, pose) says why not; IK reaches unless ``ik`` says no."""

    def __init__(self, start=HAND, refuse=None, ik=None, land_off=None):
        self.p = np.array(start, dtype=float)
        self.rot = KEEP
        self.refuse = refuse or (lambda tool, pose: None)
        self.ik = ik or (lambda pose: True)
        self.land_off = land_off
        self.moves = []

    def _move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy(), pose["rotation"], kwargs))
            why = self.refuse(tool, pose)
            if why == "raise":
                raise ToolError(tool, "IK failed: target out of reach")
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            self.p, self.rot = target, pose["rotation"]
            if self.land_off is not None:
                self.p = self.p + np.asarray(self.land_off)
            return {"reached": True, "outcome": "reached"}

        return go

    def tools(self, obs=None, **more):
        tools = {
            "robot.get_ee_pose": lambda **kw: {
                "pose": {
                    "position": dict(zip("xyz", map(float, self.p), strict=True)),
                    "rotation": self.rot,
                }
            },
            "robot.go_to_pose": self._move("robot.go_to_pose"),
            "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
            "robot.solve_ik": lambda pose, **kw: {
                "success": True,
                "within_servo_tolerance": bool(self.ik(pose)),
            },
            "robot.grasp_frame": {"rotation": TOP_A, "rotation_flipped": TOP_B},
            "robot.describe_workspace": {"surface_z": TABLE_Z},
        }
        if obs is not None:
            tools["robot.get_observation"] = obs
        tools.update(more)
        return tools


def box_local(p):
    d = np.array([p["x"], p["y"]]) - BOX_C
    c, s = np.cos(-BOX_YAW), np.sin(-BOX_YAW)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def plan_with(pixels, query=(184.0, 340.0)):
    return {
        "found": True,
        "tracks": [{"query_pixel": list(query), "pixels": [list(p) for p in pixels]}],
        "dt_s": None,
    }


INNER = BOX_OUT / 2 - BOX_WALL


# ---------------------------------------------------------------------------
# The real RoboBrain plans on the cube + box frame
# ---------------------------------------------------------------------------


def test_the_scene_matches_the_frame():
    """The box and cube as placed project where the twin's fit saw them in the frame."""
    T = np.asarray(DATA["camera_pose"]["cube_box"])
    assert np.linalg.norm(project(T, [*BOX_C, TABLE_Z]) - [303.3, 330.6]) < 8.0
    assert np.linalg.norm(project(T, [*CUBE_C, TABLE_Z + 0.015]) - [183.7, 339.5]) < 8.0


@pytest.mark.parametrize("rec", traces("cube_box"), ids=lambda r: r["label"])
def test_every_real_cube_to_box_plan_lands_over_the_box(propose, cube_obs, rec):
    arm = Arm()
    out = propose.run(FakeContext(arm.tools()), plan=rec["plan"], observation=cube_obs)
    assert out["route"] == "proposed", out["detail"]
    local = box_local(out["target_point"])
    assert np.all(np.abs(local) < BOX_OUT / 2), local  # over the box's footprint
    tp = out["target_point"]
    assert (
        TABLE_Z < tp["z"] <= TABLE_Z + BOX_H + 0.002
    )  # the measured floor (or a wall), not the model's height
    assert out["object_top_z"] == pytest.approx(TABLE_Z + BOX_H, abs=0.004)  # the rim
    goal = out["goal_pose"]["position"]
    assert goal["z"] == pytest.approx(out["object_top_z"] + 0.03)
    assert (goal["x"], goal["y"]) == (tp["x"], tp["y"])
    assert out["hover_pose"]["position"]["z"] >= TABLE_Z + 0.12
    # The model's own end point hovers 14-28 cm over the table; ours is on the box.
    model_end = rec["plan"]["tracks"][0]["points_world"][-1]
    assert model_end[2] - TABLE_Z > 0.12
    d = out["approach_direction"]
    assert d["x"] ** 2 + d["y"] ** 2 == pytest.approx(1.0)


def test_eleven_of_twelve_real_plans_end_inside_the_walls(propose, cube_obs):
    inside = []
    for rec in traces("cube_box"):
        out = propose.run(FakeContext(Arm().tools()), plan=rec["plan"], observation=cube_obs)
        if np.all(np.abs(box_local(out["target_point"])) < INNER):
            inside.append(rec["label"])
    # cube0/end_effector/start ends 3.7 px inside the box's far edge in the image: on the far wall.
    assert len(inside) == 11 and "cube0/end_effector/start" not in inside


def test_the_whole_path_direction_points_from_the_cube_to_the_box(propose, cube_obs):
    """The last segment of a descending arc projects mostly toward the camera (-y here); the whole path is +x."""
    for rec in traces("cube_box"):
        out = propose.run(
            FakeContext(Arm().tools()),
            plan=rec["plan"],
            observation=cube_obs,
            approach_from="whole_path",
        )
        assert out["approach_direction"]["x"] > 0.6, rec["label"]


def test_follow_brings_the_hand_above_the_box(propose, follow, cube_obs):
    rec = traces("cube_box")[0]
    arm = Arm()
    ctx = FakeContext(arm.tools())
    prop = propose.run(ctx, plan=rec["plan"], observation=cube_obs)
    out = follow.run(ctx, proposal=prop["proposal"])
    assert out["route"] == "arrived", out["detail"]
    assert out["legs_done"] == ["rise", "hover", "descend"]
    names = [m[0] for m in arm.moves]
    assert names == ["robot.go_to_pose", "robot.go_to_pose_cartesian", "robot.go_to_pose_cartesian"]
    rise, hover, down = (m[1] for m in arm.moves)
    travel = prop["travel_z"]
    np.testing.assert_allclose(rise, [HAND[0], HAND[1], travel])
    np.testing.assert_allclose(hover[2], travel)
    goal = prop["goal_pose"]["position"]
    np.testing.assert_allclose(down, [goal["x"], goal["y"], goal["z"]])
    assert all(m[2] == KEEP for m in arm.moves)  # the hand's own rotation reaches: kept
    assert np.all(np.abs(box_local(out["final_position"])) < INNER)
    assert out["position_error"] == pytest.approx(0.0)


def test_cable_endpoint_against_a_pointing_cross_check(propose):
    """The cable frame: RoboBrain's endpoints near the left ring, checked against a point on the ring."""
    T = np.asarray(DATA["camera_pose"]["cable"])
    obs = {"cameras": [camera("cable", solids=_slabs()[:1])]}
    ring = {
        "pixel_x": 235.0,
        "pixel_y": 365.0,
        "found": True,
        "points": [{"pixel_x": 235.0, "pixel_y": 365.0}],
    }
    by = {r["label"]: r for r in traces("cable")}
    near = by["cable_var/object/insert"]["plan"]  # ends on the left ring's top edge
    far = by["cable/object/start"]["plan"]  # ends on the board 40 px off the ring
    arm = Arm(start=(0.0, 0.2, 0.15))
    ok = propose.run(
        FakeContext(arm.tools(**{"gemini-er.point": ring})),
        plan=near,
        observation=obs,
        target_phrase="the left black ring",
    )
    assert ok["route"] == "proposed", ok["detail"]
    assert 0.0 <= ok["cross_check_distance"] < 0.05
    bad = propose.run(
        FakeContext(arm.tools(**{"gemini-er.point": ring})),
        plan=far,
        observation=obs,
        target_phrase="the left black ring",
    )
    assert bad["route"] == "plan_rejected"
    assert "cross-check disagrees" in bad["detail"]
    assert bad["cross_check_distance"] > 0.05
    assert project(T, [0.2, 0.3, TABLE_Z]).shape == (2,)


# ---------------------------------------------------------------------------
# Rejections and no plan
# ---------------------------------------------------------------------------


def test_endpoint_off_the_table_is_rejected(propose):
    T = np.asarray(DATA["camera_pose"]["cube_box"])
    solids = _slabs(table_xy=((-0.3, 0.6), (-0.1, 0.45)))
    obs = {"cameras": [camera(solids=solids)]}
    far = project(T, [0.25, 0.85, TABLE_Z - 0.75])  # past the table's far edge, on the floor
    assert 0 <= far[0] < W and 0 <= far[1] < H
    arm = Arm()
    out = propose.run(FakeContext(arm.tools()), plan=plan_with([(240, 300), far]), observation=obs)
    assert out["route"] == "plan_rejected"
    assert "off the table" in out["detail"]
    assert arm.moves == []


def test_endpoint_outside_the_image_or_without_depth_is_rejected(propose, cube_obs):
    arm = Arm()
    out = propose.run(
        FakeContext(arm.tools()), plan=plan_with([(250, 300), (700, 300)]), observation=cube_obs
    )
    assert out["route"] == "plan_rejected" and "outside the image" in out["detail"]
    cam = camera()
    cam["depth"] = cam["depth"].copy()
    cam["depth"][310:340, 290:320] = np.nan
    out = propose.run(
        FakeContext(arm.tools()),
        plan=plan_with([(250, 300), (303, 325)]),
        observation={"cameras": [cam]},
    )
    assert out["route"] == "plan_rejected" and "no valid depth" in out["detail"]


def test_workspace_bounds_reject(propose, cube_obs):
    out = propose.run(
        FakeContext(Arm().tools()),
        plan=traces("cube_box")[0]["plan"],
        observation=cube_obs,
        workspace_bounds={"x": [-0.3, 0.2], "y": [0.0, 0.6]},
    )
    assert out["route"] == "plan_rejected" and "outside the workspace" in out["detail"]


def test_a_still_forecast_is_no_plan(propose, cube_obs):
    still = plan_with([(184.5, 340.2)] * 30)
    out = propose.run(FakeContext(Arm().tools()), plan=still, observation=cube_obs)
    assert out["route"] == "no_plan" and "no motion proposed" in out["detail"]


def test_no_plan_cases(propose, cube_obs):
    ctx = FakeContext(Arm().tools())
    assert (
        propose.run(ctx, plan={"found": False, "tracks": []}, observation=cube_obs)["route"]
        == "no_plan"
    )
    out = propose.run(ctx, observation=cube_obs)
    assert out["route"] == "no_plan" and "no instruction" in out["detail"]
    out = propose.run(ctx, plan=plan_with([(250, 300)]), observation=cube_obs, track=2)
    assert out["route"] == "no_plan"
    out = propose.run(
        ctx,
        plan={"found": True, "tracks": [{"query_pixel": [1, 1], "pixels": []}]},
        observation=cube_obs,
    )
    assert out["route"] == "no_plan"


def test_out_of_reach_tries_top_down_then_rejects(propose, cube_obs):
    plan = traces("cube_box")[0]["plan"]
    arm = Arm(ik=lambda pose: pose["rotation"] == TOP_B)
    out = propose.run(FakeContext(arm.tools()), plan=plan, observation=cube_obs)
    assert out["route"] == "proposed", out["detail"]
    assert out["proposal"]["rotation_source"] == "top-down flipped"
    assert out["goal_pose"]["rotation"] == TOP_B
    arm = Arm(ik=lambda pose: False)
    out = propose.run(FakeContext(arm.tools()), plan=plan, observation=cube_obs)
    assert out["route"] == "plan_rejected" and "out of reach" in out["detail"]
    out = propose.run(FakeContext(arm.tools()), plan=plan, observation=cube_obs, check_reach=False)
    assert out["route"] == "proposed"


def test_no_ik_preview_lets_the_moves_decide(propose, cube_obs):
    def no_ik(**kw):
        raise ToolError("robot.solve_ik", "not served")

    out = propose.run(
        FakeContext(Arm().tools(**{"robot.solve_ik": no_ik})),
        plan=traces("cube_box")[0]["plan"],
        observation=cube_obs,
    )
    assert out["route"] == "proposed"
    assert out["proposal"]["rotation_source"] == "current"


def test_cross_check_not_found_raised_and_agreeing(propose, cube_obs):
    plan = traces("cube_box")[0]["plan"]
    none = {"pixel_x": 0.0, "pixel_y": 0.0, "found": False, "points": []}
    out = propose.run(
        FakeContext(Arm().tools(**{"gemini-er.point": none})),
        plan=plan,
        observation=cube_obs,
        target_phrase="the open box",
    )
    assert out["route"] == "plan_rejected" and "found no" in out["detail"]

    def down(**kw):
        raise ToolError("gemini-er.point", "quota")

    out = propose.run(
        FakeContext(Arm().tools(**{"gemini-er.point": down})),
        plan=plan,
        observation=cube_obs,
        target_phrase="the open box",
    )
    assert out["route"] == "proposed" and "cross-check unavailable" in out["detail"]
    box = {
        "pixel_x": 303.0,
        "pixel_y": 331.0,
        "found": True,
        "points": [{"pixel_x": 303.0, "pixel_y": 331.0}],
    }
    ctx = FakeContext(Arm().tools(**{"locate-anything.point": box}))
    out = propose.run(
        ctx,
        plan=plan,
        observation=cube_obs,
        target_phrase="the open box",
        cross_check_tool="locate-anything.point",
    )
    assert out["route"] == "proposed"
    assert 0.0 <= out["cross_check_distance"] < 0.05
    assert [c.tool for c in ctx.calls].count("locate-anything.point") == 1


def test_goal_height_given_and_floored(propose, cube_obs):
    plan = traces("cube_box")[0]["plan"]
    out = propose.run(
        FakeContext(Arm().tools()), plan=plan, observation=cube_obs, goal_height=TABLE_Z + 0.05
    )
    assert out["goal_pose"]["position"]["z"] == pytest.approx(TABLE_Z + 0.05)
    out = propose.run(
        FakeContext(Arm().tools()), plan=plan, observation=cube_obs, goal_height=TABLE_Z
    )
    assert out["goal_pose"]["position"]["z"] == pytest.approx(TABLE_Z + 0.015)


def test_table_height_falls_back_to_the_depth_image(propose, cube_obs):
    def none(**kw):
        raise ToolError("robot.describe_workspace", "not served")

    out = propose.run(
        FakeContext(Arm().tools(**{"robot.describe_workspace": none})),
        plan=traces("cube_box")[0]["plan"],
        observation=cube_obs,
    )
    assert out["route"] == "proposed"
    assert out["table_z"] == pytest.approx(TABLE_Z, abs=0.004)


def test_an_obstacle_under_the_route_raises_the_travel_height(propose):
    post = (
        np.array([0.16, 0.30, TABLE_Z + 0.09]),
        np.array([0.02, 0.02, 0.09]),
        0.0,
    )  # 18 cm post between hand and box
    obs = {"cameras": [camera(solids=_slabs(extra=[post]))]}
    arm = Arm(start=(0.03, 0.30, 0.15))
    out = propose.run(FakeContext(arm.tools()), plan=traces("cube_box")[0]["plan"], observation=obs)
    assert out["route"] == "proposed", out["detail"]
    assert out["travel_z"] == pytest.approx(TABLE_Z + 0.18 + 0.08, abs=0.01)
    tall = (post[0] + [0, 0, 0.1], post[1] + [0, 0, 0.1], 0.0)  # 38 cm
    obs = {"cameras": [camera(solids=_slabs(extra=[tall]))]}
    out = propose.run(FakeContext(arm.tools()), plan=traces("cube_box")[0]["plan"], observation=obs)
    assert out["route"] == "plan_rejected" and "travel height" in out["detail"]


def test_follow_path_waypoints_sit_on_the_plan_rays_at_travel_height(propose, follow, cube_obs):
    rec = next(
        r for r in traces("cube_box") if r["label"] == "cube1/end_effector"
    )  # 6 keypoints, from the gripper
    T = np.asarray(DATA["camera_pose"]["cube_box"])
    arm = Arm()
    ctx = FakeContext(arm.tools())
    out = propose.run(ctx, plan=rec["plan"], observation=cube_obs, follow_path=True)
    assert out["route"] == "proposed", out["detail"]
    assert out["waypoints"], "expected waypoints from the path"
    pix = [np.array(p) for p in rec["plan"]["tracks"][0]["pixels"][:-1]]
    for w in out["waypoints"]:
        assert w["z"] == pytest.approx(out["travel_z"])
        uv = project(T, [w["x"], w["y"], w["z"]])
        assert min(np.linalg.norm(uv - p) for p in pix) < 0.5
    moved = follow.run(ctx, proposal=out["proposal"])
    assert moved["route"] == "arrived"
    assert moved["legs_done"][1:-2] == [f"waypoint {i + 1}" for i in range(len(out["waypoints"]))]


def test_approach_offset_adds_a_pre_hover_on_the_plans_side(propose, cube_obs):
    out = propose.run(
        FakeContext(Arm().tools()),
        plan=traces("cube_box")[0]["plan"],
        observation=cube_obs,
        approach_offset=0.05,
    )
    pre = out["waypoints"][-1]
    tp = out["target_point"]
    d = np.array([tp["x"] - pre["x"], tp["y"] - pre["y"]])
    assert np.linalg.norm(d) == pytest.approx(0.05)
    # The last segment runs down the image, toward the camera (-y): in from the far side.
    assert d[1] < -0.03


# ---------------------------------------------------------------------------
# Asking the planner, and replanning once
# ---------------------------------------------------------------------------


def test_asks_the_planner_with_the_frame_and_start(propose, cube_obs):
    good = traces("cube_box")[0]["plan"]
    seen = {}

    def planner(**kw):
        seen.update(kw)
        return good

    ctx = FakeContext(Arm().tools(obs=cube_obs, **{"robobrain.trace": planner}))
    out = propose.run(
        ctx,
        instruction="put the cube in the box",
        start_pixel=[184, 340],
        planner_args={"mode": "object"},
    )
    assert out["route"] == "proposed" and out["planner_calls"] == 1
    assert seen["start_point"] == [184.0, 340.0] and seen["mode"] == "object"
    assert seen["instruction"] == "put the cube in the box"
    assert seen["depth"].shape == (H, W) and np.asarray(seen["intrinsics"]).shape == (3, 3)


def test_a_forecaster_gets_query_points_and_needs_a_start(propose, cube_obs):
    seen = {}

    def forecast(**kw):
        seen.update(kw)
        return plan_with([(200, 335), (260, 325), (303, 330)])

    ctx = FakeContext(Arm().tools(obs=cube_obs, **{"molmo-motion.forecast": forecast}))
    out = propose.run(
        ctx,
        instruction="the cube is moved into the box",
        planner="molmo-motion.forecast",
        start_pixel=[184, 340],
    )
    assert out["route"] == "proposed", out["detail"]
    assert seen["query_points"] == [[184.0, 340.0]] and "start_point" not in seen
    out = propose.run(ctx, instruction="x", planner="molmo-motion.forecast")
    assert out["route"] == "no_plan" and "start_pixel" in out["detail"]


def test_replans_once_from_a_fresh_frame(propose, cube_obs):
    bad = plan_with([(250, 300), (700, 300)])
    good = traces("cube_box")[0]["plan"]
    ctx = FakeContext(Arm().tools(obs=cube_obs, **{"robobrain.trace": good}))
    out = propose.run(
        ctx, plan=bad, observation=cube_obs, instruction="put the cube in the box", replan=True
    )
    assert out["route"] == "proposed", out["detail"]
    assert out["planner_calls"] == 1
    assert "plan 1: plan_rejected" in out["detail"] and "plan 2: proposed" in out["detail"]
    tools = [c.tool for c in ctx.calls]
    assert tools.index("robot.get_observation") < tools.index("robobrain.trace")


def test_replan_that_fails_again_reports_both(propose, cube_obs):
    bad = plan_with([(250, 300), (700, 300)])
    ctx = FakeContext(Arm().tools(obs=cube_obs, **{"robobrain.trace": [bad, bad]}))
    out = propose.run(ctx, instruction="put the cube in the box", observation=cube_obs, replan=True)
    assert out["route"] == "plan_rejected"
    assert out["planner_calls"] == 2
    assert out["detail"].count("plan_rejected") == 2

    def broken(**kw):
        raise ToolError("robobrain.trace", "CUDA out of memory")

    ctx = FakeContext(Arm().tools(obs=cube_obs, **{"robobrain.trace": broken}))
    out = propose.run(ctx, instruction="put the cube in the box", observation=cube_obs)
    assert out["route"] == "no_plan" and "raised" in out["detail"]


def test_without_replan_a_rejection_is_final(propose, cube_obs):
    bad = plan_with([(250, 300), (700, 300)])
    ctx = FakeContext(
        Arm().tools(obs=cube_obs, **{"robobrain.trace": traces("cube_box")[0]["plan"]})
    )
    out = propose.run(ctx, plan=bad, observation=cube_obs, instruction="put the cube in the box")
    assert out["route"] == "plan_rejected"
    assert "robobrain.trace" not in [c.tool for c in ctx.calls]


# ---------------------------------------------------------------------------
# Following: refusals and arrival
# ---------------------------------------------------------------------------


@pytest.fixture
def proposal(propose, cube_obs):
    return propose.run(
        FakeContext(Arm().tools()), plan=traces("cube_box")[0]["plan"], observation=cube_obs
    )["proposal"]


@pytest.mark.parametrize("how", ["raise", "servo says unreachable"])
def test_a_refused_descent_is_unreachable(follow, proposal, how):
    arm = Arm(
        refuse=lambda tool, pose: (
            how if pose["position"]["z"] < proposal["travel_z"] - 0.01 else None
        )
    )
    out = follow.run(FakeContext(arm.tools()), proposal=proposal)
    assert out["route"] == "unreachable"
    assert out["detail"].startswith("descend move failed")
    assert out["legs_done"] == ["rise", "hover"]


def test_ending_off_the_goal_is_unreachable(follow, proposal):
    arm = Arm(land_off=(0.0, 0.0, 0.02))
    out = follow.run(FakeContext(arm.tools()), proposal=proposal)
    assert out["route"] == "unreachable" and "from the goal" in out["detail"]
    assert out["position_error"] == pytest.approx(0.02)


def test_no_rise_when_already_high_and_knobs_only_when_set(follow, proposal):
    arm = Arm(start=(0.05, 0.25, proposal["travel_z"] + 0.02))
    out = follow.run(FakeContext(arm.tools()), proposal=proposal)
    assert out["route"] == "arrived"
    assert out["legs_done"] == ["hover", "descend"]
    arm = Arm(start=(0.05, 0.25, 0.03))
    follow.run(FakeContext(arm.tools()), proposal=proposal, tolerance=0.004, arm_id=1)
    assert arm.moves[0][3] == {"arm_id": 1, "tolerance": 0.004}
    assert arm.moves[1][3] == {"arm_id": 1}


def test_a_servo_timeout_short_of_the_goal_is_judged_by_the_reading(follow, proposal):
    """The YAM twin's servo timed out 3.8 mm over the goal on some descents: within the bound, arrived."""

    class Stalls(Arm):
        def _move(self, tool):
            go = super()._move(tool)

            def stall(pose, **kw):
                reply = go(pose, **kw)
                if pose["position"]["z"] < proposal["travel_z"] - 0.01:
                    self.p = self.p + np.array([0.0, 0.0, self.short])
                    return {"reached": False, "outcome": "timeout"}
                return reply

            return stall

    arm = Stalls()
    arm.short = 0.0038
    out = follow.run(FakeContext(arm.tools()), proposal=proposal)
    assert out["route"] == "arrived", out["detail"]
    assert "descend 3.8 mm short" in out["detail"]
    assert out["position_error"] == pytest.approx(0.0038)
    arm = Stalls()
    arm.short = 0.03
    out = follow.run(FakeContext(arm.tools()), proposal=proposal)
    assert out["route"] == "unreachable" and out["detail"] == "descend move failed: timeout"
