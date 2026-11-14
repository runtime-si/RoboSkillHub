"""requesting-a-typed-visual-plan: FakeContext tests of the request, the checks, the ladder and reuse.

The scenes are following-a-visual-plan's (test_following_a_visual_plan.py): the real YAM station
frame's calibrated overhead camera with a ray-cast depth image of the box and the cube where the
twin fitted to that frame puts them, and the cable frame's camera over a bare table. The text
answers are real: what four models answered this skill's own request on those two frames through
the hub's vlm.query (data/typed_plan_answers_station.json), replayed. The schema answers are
robobrain.trace's real traces on the same frames (data/robobrain_traces_station.json) and what
it answered this skill's script live on the cube frame (``robobrain_live`` in the answers file).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

from .test_following_a_visual_plan import (
    BOX_C,
    BOX_H,
    BOX_OUT,
    BOX_YAW,
    TABLE_Z,
    H,
    K,
    W,
    _slabs,
    box_local,
    camera,
    project,
)
from .test_following_a_visual_plan import DATA as STATION

BUNDLE = "requesting-a-typed-visual-plan"
ANSWERS = json.loads(
    (Path(__file__).resolve().parent / "data" / "typed_plan_answers_station.json").read_text()
)
BY_LABEL = {a["label"]: a for a in ANSWERS["answers"]}
LIVE = {t["label"]: t for t in ANSWERS["robobrain_live"]}
TRACES = {t["label"]: t for t in STATION["traces"]}
LIFT = float(ANSWERS["hand_over_table_m"])
TRAVEL = TABLE_Z + 0.12
KEEP = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
CUBE_PX = (183.7, 339.5)
BOX_PX = tuple(ANSWERS["box_px"])
CUBE_TO_BOX = "pick up the cube and place it in the box"


def pose_of(frame="cube_box"):
    return np.asarray(STATION["camera_pose"][frame], dtype=np.float64)


def on_ray(uv, z, frame="cube_box"):
    """The world point on a pixel's ray at world height z."""
    T = pose_of(frame)
    ray = T[:3, :3] @ np.array([(uv[0] - K[0, 2]) / K[0, 0], (uv[1] - K[1, 2]) / K[1, 1], 1.0])
    return T[:3, 3] + (z - T[2, 3]) / ray[2] * ray


def to_camera(p, frame="cube_box"):
    T = pose_of(frame)
    return (np.asarray(p, dtype=np.float64) - T[:3, 3]) @ T[:3, :3]


def hands(frame="cube_box"):
    """Each arm's tool centre where the live check put it: on its ring's ray, 8 cm over the table."""
    return {
        int(arm): on_ray(px, TABLE_Z + LIFT, frame)
        for arm, px in ANSWERS["hands_px"][frame].items()
    }


def hand_px(arm, frame="cube_box"):
    return np.array(ANSWERS["hands_px"][frame][str(arm)], dtype=np.float64)


def tools(at=None, **more):
    at = hands() if at is None else at

    def ee(arm_id=0, **kw):
        p = at[int(arm_id)]
        return {
            "pose": {"position": dict(zip("xyz", map(float, p), strict=True)), "rotation": KEEP}
        }

    out = {"robot.get_ee_pose": ee, "robot.describe_workspace": {"surface_z": TABLE_Z}}
    out.update(more)
    return out


def says(subtask="reach", **paths):
    """A text planner's reply: ``{colour: [[u, v] px, ...]}`` (the ring's centre first), in the answer's 0..1000 units."""
    body = {"subtask": subtask}
    for colour, points in paths.items():
        body[colour] = [
            [round(float(u) * 1000 / W, 4), round(float(v) * 1000 / H, 4)] for u, v in points
        ]
    return {"text": json.dumps(body)}


def line(a, b, n=10):
    """``a``, then ``n`` points evenly spaced to ``b``."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return [a + (b - a) * i / n for i in range(n + 1)]


def seen(route, frame="cube_box"):
    """The pixels a horizontal route ``[[x, y], ...]`` is seen at: flown at the travel height, ended on the table."""
    T = pose_of(frame)
    return [
        *(project(T, [x, y, TRAVEL]) for x, y in route[:-1]),
        project(T, [*route[-1], TABLE_Z]),
    ]


def flight(arm, to, frame="cube_box", n=10):
    """A text path for an arm: its ring, then ``n`` waypoints of the straight flight to ``to`` (x, y on the table)."""
    return [hand_px(arm, frame), *seen(line(hands(frame)[arm][:2], to, n)[1:], frame)]


@pytest.fixture(scope="module")
def request_plan(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["request_typed_plan"].module


@pytest.fixture(scope="module")
def follower(skills_registry):
    scripts = skills_registry.get("following-a-visual-plan").canonical_scripts
    return scripts["propose_target_from_plan"].module


@pytest.fixture(scope="module")
def obs():
    return {
        "cube_box": {"cameras": [camera()]},
        "cable": {"cameras": [camera("cable", solids=_slabs()[:1])]},
    }


def replay(mod, obs, rec, **kw):
    """Ask with one text planner that answers a recorded answer."""
    ctx = FakeContext(tools(hands(rec["frame"]), **{"vlm.query": {"text": rec["text"]}}))
    out = mod.run(
        ctx,
        instruction=rec["instruction"],
        arms=rec["arms"],
        planners=["vlm.query"],
        observation=obs[rec["frame"]],
        start_pixels=rec["start_pixels"],
        **kw,
    )
    return out, ctx


# ---------------------------------------------------------------------------
# The request and its answer
# ---------------------------------------------------------------------------


def test_the_request_declares_the_shape(request_plan):
    text = request_plan.text_request(
        "pick up the cube", 640, 480, [(0, "red", True), (1, "blue", True)], 10
    )
    assert "(640 x 480 px)" in text and "Task: pick up the cube\n" in text
    assert "a red ring around the tool centre of arm 0's gripper" in text
    assert "a blue ring around the tool centre of arm 1's gripper" in text
    assert "exactly 10 waypoints" in text
    assert '"red": [[x, y], ... exactly 11 pairs], "blue": [[x, y], ... exactly 11 pairs]' in text
    assert "from 0 to 1000" in text and "carried point" not in text
    carried = request_plan.text_request(
        "thread the cable", 640, 480, [(0, "red", False)], 5, ["through the left ring"]
    )
    assert "a red ring around a point arm 0 carries" in carried
    assert "The red ring's part of the task: through the left ring\n" in carried
    assert "exactly 5 waypoints" in carried and '"red": [[x, y], ... exactly 6 pairs]' in carried
    assert "blue" not in carried and "tool centre" not in carried


def test_an_answer_in_the_shape_is_read_in_pixels(request_plan):
    parse = request_plan.parse_text_answer
    good = json.dumps(
        {"subtask": " reach the cube ", "red": [[100, 500], [110, 500], [120, 500], [130, 500]]}
    )
    subtask, paths = parse(good, ["red"], 3, 640, 480)
    assert subtask == "reach the cube"
    assert paths["red"]["start"] == [64.0, 240.0]
    assert paths["red"]["pixels"] == pytest.approx(
        np.array([[70.4, 240], [76.8, 240], [83.2, 240]])
    )
    fenced = "```json\n" + good + "\n```"  # as Gemini answers
    assert parse(fenced, ["red"], 3, 640, 480)[1] == paths
    edge = json.dumps({"red": [[1000, 1000]] * 4})  # the far corner is the last pixel, not past it
    assert parse(edge, ["red"], 3, 640, 480) == (
        "",
        {"red": {"start": [639.0, 479.0], "pixels": [[639.0, 479.0]] * 3}},
    )


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("I cannot see a cube in this image.", "no JSON object"),
        ('{"red": [[1, 2], [3, 4]', "no JSON object"),
        ('{"red": [[1, 2], [3, 4]}', "not one JSON object"),
        ('{"subtask": "reach"}', "no list of points for the red ring"),
        ('{"red": "to the left"}', "no list of points for the red ring"),
        (
            '{"red": [[1, 2], [3, 4], [5, 6]]}',
            "has 2 waypoints after its ring's centre, 3 declared",
        ),
        (
            '{"red": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]}',
            "has 4 waypoints after its ring's centre, 3 declared",
        ),
        ('{"red": [[1, 2], [3, 4], [5, 1200], [7, 8]]}', "waypoint 2 [5, 1200] is outside 0..1000"),
        ('{"red": [[-1, 2], [3, 4], [5, 6], [7, 8]]}', "start [-1, 2] is outside 0..1000"),
        ('{"red": [[1, 2], [3, 4], [5], [7, 8]]}', "waypoint 2 is not an [x, y] pair"),
        ('{"red": [[1, 2], [3, 4], [5, 6, 7], [7, 8]]}', "waypoint 2 is not an [x, y] pair"),
        ('{"red": [["a", 2], [3, 4], [5, 6], [7, 8]]}', "start is not an [x, y] pair"),
        ('{"red": [[true, 2], [3, 4], [5, 6], [7, 8]]}', "start is not an [x, y] pair"),
        ('{"red": [[1, 2], [3, NaN], [5, 6], [7, 8]]}', "waypoint 1 is not an [x, y] pair"),
        (
            '{"red": [[1, 2], [3, 4], [5, 6], [7, 8]], "blue": [[1, 2], [3, 4], [5, 6], [7, 8]]}',
            "a blue path, which was not declared",
        ),
    ],
)
def test_an_answer_out_of_the_shape_names_the_rule_it_breaks(request_plan, text, why):
    with pytest.raises(ValueError, match="^.*" + why.replace("[", r"\[").replace("]", r"\]")):
        request_plan.parse_text_answer(text, ["red"], 3, 640, 480)


def test_the_rings_are_drawn_around_the_starts(request_plan):
    rgb = np.full((H, W, 3), 90, np.uint8)
    marks = [
        (np.array([134.0, 366.0]), (255, 40, 40)),
        (np.array([499.0, 362.0]), (40, 110, 255)),
    ]
    out = request_plan.mark_image(rgb, marks)
    assert out.shape == rgb.shape and (rgb == 90).all()  # a copy: the frame itself is untouched
    assert tuple(out[366, 134]) == (90, 90, 90)  # the ringed point stays visible
    assert tuple(out[366, 134 + 9]) == (255, 40, 40)
    assert tuple(out[362, 499 - 9]) == (40, 110, 255)
    v, u = np.nonzero((out != 90).any(-1))
    r = np.minimum(np.hypot(u - 134, v - 366), np.hypot(u - 499, v - 362))
    assert 0.014 * W - 4.0 <= r.min() and r.max() <= 0.014 * W + 3.0


def test_resample_spaces_the_waypoints_evenly_and_keeps_the_end(request_plan):
    start, corner, end = [0.0, 0.0], [30.0, 0.0], [30.0, 40.0]  # an L, 70 px long
    px, z = request_plan.resample(
        np.array(start), np.array([corner, end]), 7, np.array([1.0, 0.7, 0.9])
    )
    assert px.tolist() == pytest.approx(
        np.array([[10, 0], [20, 0], [30, 0], [30, 10], [30, 20], [30, 30], [30, 40]])
    )
    # Depths follow the distance along the path, not the keypoint index.
    assert z == pytest.approx([0.9, 0.8, 0.7, 0.75, 0.8, 0.85, 0.9])
    px, z = request_plan.resample(np.array(start), np.array([start, start]), 4)
    assert px.tolist() == [[0.0, 0.0]] * 4 and z is None


def test_separation_follows_the_order_the_arms_move_in(request_plan):
    a = np.array([[0.0, 0.0], [1.0, 0.0]])  # left to right
    b = np.array([[1.0, 0.1], [0.0, 0.1]])  # right to left, 10 cm beside it
    least, at = request_plan.separation(a, b)
    assert least == pytest.approx(0.1) and at == pytest.approx(0.5)  # they pass mid-way
    # One after the other: a ends where b still waits (half-way through the plan).
    least, at = request_plan.separation(a, b, "a_then_b")
    assert least == pytest.approx(0.1) and at == pytest.approx(0.5, abs=0.03)
    c = np.array([[1.0, 0.5], [1.0, 1.5]])  # an arm that leaves from half a metre beside a's end
    assert request_plan.separation(a, c) == pytest.approx((1.125**0.5, 0.25))
    least, at = request_plan.separation(a, c, "a_then_b")  # a arrives while c still waits there
    assert least == pytest.approx(0.5) and at == pytest.approx(0.5, abs=0.03)
    assert request_plan.separation(a, c, "b_then_a") == pytest.approx((1.25**0.5, 0.0))  # c left
    still = np.repeat([[0.5, 0.3]], 2, 0)
    assert request_plan.separation(a, still) == pytest.approx((0.3, 0.5))
    with pytest.raises(ValueError, match="order is one of"):
        request_plan.separation(a, b, "in_turns")


# ---------------------------------------------------------------------------
# Real answers of four models on the real frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rec", ANSWERS["answers"], ids=lambda r: r["label"])
def test_a_recorded_answer_routes_as_it_did_live(request_plan, obs, rec):
    out, ctx = replay(request_plan, obs, rec)
    assert out["route"] == rec["live_route"], out["detail"]
    assert out["planner_calls"] == 1  # both arms in one call
    assert len(ctx.calls_to("vlm.query")) == 1
    if out["route"] != "planned":
        assert out["plan"] == {} and out["endpoint_pixels"] == []
        return
    plan = out["plan"]
    json.dumps(plan)  # goes from node to node
    assert plan["found"] and plan["horizon"] == 10 and plan["frame"] == "pixel"
    assert plan["typed"] == {"arms": rec["arms"], "waypoints": 10, "require_metric": False}
    assert [t["arm_id"] for t in plan["tracks"]] == rec["arms"]
    assert all(len(t["pixels"]) == 10 and t["predicted"] == [True] * 10 for t in plan["tracks"])
    assert out["subtask"] == plan["subtask"] != ""
    assert out["endpoint_pixels"] == [t["pixels"][-1] for t in plan["tracks"]]


@pytest.mark.parametrize(
    "label",
    [a["label"] for a in ANSWERS["answers"] if not a["label"].startswith("flash-lite")],
)
def test_the_capable_planners_end_on_their_targets(request_plan, obs, label):
    rec = BY_LABEL[label]
    out, _ = replay(request_plan, obs, rec)
    if rec["live_route"] != "planned":
        return
    for track in out["plan"]["tracks"]:
        target = rec["targets_px"][str(track["arm_id"])]
        if target is None:  # the arm told not to move
            assert not track["active"]
            continue
        assert track["active"] and track["start_error_px"] < 2.0  # it saw its ring
        assert np.linalg.norm(np.subtract(track["pixels"][-1], target)) < 12.0


def test_the_request_shows_the_rings_and_asks_once_for_both_arms(request_plan, obs):
    rec = BY_LABEL["azure/dual_cube_box"]
    out, ctx = replay(request_plan, obs, rec)
    call = ctx.calls_to("vlm.query")[0].kwargs
    assert set(call) == {"prompt", "image"}
    assert "a red ring around the tool centre of arm 0's gripper" in call["prompt"]
    assert "a blue ring around the tool centre of arm 1's gripper" in call["prompt"]
    assert f"Task: {rec['instruction']}\n" in call["prompt"]
    image = call["image"]
    assert image.shape == (H, W, 3) and image.dtype == np.uint8
    assert tuple(image[366, 134 + 9]) == (255, 40, 40)  # the rings, on the frame it was sent
    assert tuple(image[362, 499 - 9]) == (40, 110, 255)
    assert out["start_pixels"] == pytest.approx(np.array([[134.0, 366.0], [499.0, 362.0]]))
    # 67.5 cm apart at the start, the cube and the box 24 cm apart at the end.
    assert out["min_arm_separation_m"] == pytest.approx(0.25, abs=0.03)
    assert out["separation_at_fraction"] == pytest.approx(1.0)
    assert "2 arm(s) x 10 waypoints, arms at least" in out["attempts"][0]


def test_a_recorded_plan_feeds_following_a_visual_plan(request_plan, follower, obs):
    """The plan is the schema that skill takes: one track per arm, picked with ``track``."""
    out, _ = replay(request_plan, obs, BY_LABEL["azure/dual_cube_box"])
    ctx = FakeContext(
        tools(
            **{
                "robot.solve_ik": {"success": True, "within_servo_tolerance": True},
                "robot.grasp_frame": {"rotation": KEEP, "rotation_flipped": KEEP},
            }
        )
    )
    cube = follower.run(ctx, plan=out["plan"], observation=obs["cube_box"], track=0)
    assert cube["route"] == "proposed", cube["detail"]
    assert cube["target_point"]["z"] == pytest.approx(TABLE_Z + 0.03, abs=0.004)  # the cube's top
    box = follower.run(
        ctx, plan=out["plan"], observation=obs["cube_box"], track=1, arm_id=1, follow_path=True
    )
    assert box["route"] == "proposed", box["detail"]
    assert np.all(np.abs(box_local(box["target_point"])) < BOX_OUT / 2)  # over the box
    assert len(box["waypoints"]) >= 3  # the path's own waypoints, carried to the travel height


def test_a_path_asked_around_the_box_stays_off_it(request_plan, obs):
    half, c, s = BOX_OUT / 2, np.cos(BOX_YAW), np.sin(BOX_YAW)
    rim = np.array(  # the box's rim as the camera sees it: about 48 px wide
        [
            project(pose_of(), [*(BOX_C + [c * dx - s * dy, s * dx + c * dy]), TABLE_Z + BOX_H])
            for dx, dy in ((half, half), (half, -half), (-half, -half), (-half, half))
        ]
    )

    def over_the_box(points):
        """How many of the path's points (20 per segment) lie inside the rim."""
        dense = np.concatenate(
            [np.linspace(a, b, 20) for a, b in zip(points, points[1:], strict=False)]
        )
        edges, to = np.roll(rim, -1, 0) - rim, dense[:, None, :] - rim[None]
        side = edges[None, :, 0] * to[..., 1] - edges[None, :, 1] * to[..., 0]
        return int(((side > 0).all(1) | (side < 0).all(1)).sum())

    assert over_the_box([hand_px(1), np.array(CUBE_PX)]) > 0  # the straight way crosses it
    for label in ("azure/around_box", "er2/around_box"):
        out, _ = replay(request_plan, obs, BY_LABEL[label])
        track = out["plan"]["tracks"][0]
        path = [np.array(track["query_pixel"]), *map(np.array, track["pixels"])]
        assert over_the_box(path) == 0, label
        assert np.linalg.norm(path[-1] - CUBE_PX) < 5.0


def test_a_carried_point_is_planned_from_its_own_pixel(request_plan, obs):
    rec = BY_LABEL["er2/cable_thread"]
    out, ctx = replay(request_plan, obs, rec)
    assert out["route"] == "planned", out["detail"]
    prompt = ctx.calls_to("vlm.query")[0].kwargs["prompt"]
    assert "a red ring around a point arm 0 carries" in prompt
    assert out["start_pixels"] == [[132.0, 335.0]]
    track = out["plan"]["tracks"][0]
    assert track["query_pixel"] == [132.0, 335.0]
    assert np.linalg.norm(np.subtract(track["pixels"][-1], [235.0, 365.0])) < 12.0  # the left ring
    assert track["hand_world"] == pytest.approx(hands("cable")[0].tolist())  # still the arm's hand


def test_answers_out_of_the_shape_are_refused_with_the_reason(request_plan, obs):
    cut, _ = replay(request_plan, obs, BY_LABEL["g3flash/dual_one_stays"])
    assert cut["route"] == "plan_rejected"
    assert cut["attempts"] == ["vlm.query: rejected: no JSON object in the answer"]
    other, _ = replay(request_plan, obs, BY_LABEL["g3flash/cable_thread"])
    assert other["route"] == "plan_rejected"
    assert "the path starts 74 px from the given start (bound 32 px)" in other["detail"]
    weak, _ = replay(request_plan, obs, BY_LABEL["flash-lite/cube_pick"])
    assert weak["route"] == "plan_rejected"
    assert "the path starts 81 px from its hand (bound 32 px)" in weak["detail"]
    loose, _ = replay(request_plan, obs, BY_LABEL["flash-lite/cube_pick"], start_tolerance=0.2)
    assert loose["route"] == "planned"  # the bound is the caller's to widen


def test_a_second_planner_catches_a_wrong_endpoint_the_shape_cannot(request_plan, obs):
    """The weak planner's answer is in the declared shape and ends 144 px from the box."""
    texts = {m: BY_LABEL[f"{m}/box_reach"]["text"] for m in ("flash-lite", "azure", "er2")}
    rec = BY_LABEL["azure/box_reach"]

    def ladder(*models):
        return [{"tool": "vlm.query", "args": {"model": m}} for m in models]

    def run(planners, **kw):
        ctx = FakeContext(
            tools(**{"vlm.query": lambda prompt, image, model: {"text": texts[model]}})
        )
        out = request_plan.run(
            ctx,
            instruction=rec["instruction"],
            arms=[1],
            planners=planners,
            observation=obs["cube_box"],
            **kw,
        )
        return out, ctx

    alone, _ = run(ladder("flash-lite"))
    assert alone["route"] == "planned"
    assert np.linalg.norm(np.subtract(alone["endpoint_pixels"][0], BOX_PX)) > 140.0

    out, ctx = run(ladder("flash-lite", "azure", "er2"), agree=2)
    assert out["route"] == "planned", out["detail"]
    assert out["planner"] == "vlm.query(model=azure)"  # the first in the ladder of those agreeing
    assert out["plan"]["confirmed_by"] == ["vlm.query(model=er2)"]
    assert out["planner_calls"] == 3 and len(ctx.calls) == 3 + 1  # and the hand's pose
    assert np.linalg.norm(np.subtract(out["endpoint_pixels"][0], BOX_PX)) < 5.0
    assert out["attempts"][0].startswith("vlm.query(model=flash-lite): valid (1 arm(s) x 10")
    assert out["attempts"][0].endswith("1 of 2 planners agree on the endpoints")
    assert out["attempts"][2].endswith("endpoints within 26 px of vlm.query(model=er2)'s")

    out, ctx = run(ladder("azure", "er2", "flash-lite"), agree=2)
    assert out["route"] == "planned" and out["planner_calls"] == 2  # the third is not asked

    out, _ = run(ladder("flash-lite", "azure"), agree=2)
    assert out["route"] == "plan_rejected" and out["plan"] == {}
    assert out["attempts"][-1] == "no 2 planners agree on the endpoints"

    named = [{"tool": "vlm.query", "args": {"model": "er2"}, "name": "gemini-er"}]
    assert run(named)[0]["planner"] == "gemini-er"


# ---------------------------------------------------------------------------
# The ladder, and planners that return the schema
# ---------------------------------------------------------------------------


def test_the_ladder_falls_to_the_next_planner(request_plan, obs):
    trace = TRACES["cube1/object/start"]["plan"]  # robobrain.trace from the cube: 3 keypoints
    ctx = FakeContext(
        tools(**{"vlm.query": {"text": "I cannot see a cube."}, "robobrain.trace": trace})
    )
    out = request_plan.run(
        ctx, instruction=CUBE_TO_BOX, start_pixels=[[184.0, 340.0]], observation=obs["cube_box"]
    )
    assert out["route"] == "planned", out["detail"]
    assert out["planner"] == "robobrain.trace" and out["planner_calls"] == 2
    assert out["attempts"][0] == "vlm.query: rejected: no JSON object in the answer"
    # The model put the cube 245 mm nearer than the depth image does, and its path ends in the
    # air: its 3D is not passed on, and both are named.
    dropped = (
        "dropped: arm 0: its start is 245 mm off the depth measured at the given start and it "
        "ends 303 mm in front of the surface measured under its endpoint"
    )
    assert out["attempts"][1] == (
        "robobrain.trace: planned: 1 arm(s) x 10 waypoints, arm 0 resampled from 3, "
        f"metric part {dropped}"
    )
    call = ctx.calls_to("robobrain.trace")[0].kwargs
    assert call["instruction"] == CUBE_TO_BOX and call["start_point"] == [184.0, 340.0]
    assert call["image"].shape == (H, W, 3) and call["depth"].dtype == np.float32
    assert np.array_equal(call["intrinsics"], K) and np.array_equal(call["camera_pose"], pose_of())
    plan, track = out["plan"], out["plan"]["tracks"][0]
    assert plan["backend"] == "robobrain" and plan["subtask"] == ""
    assert len(track["pixels"]) == 10
    assert track["pixels"][-1] == pytest.approx(trace["tracks"][0]["pixels"][-1])
    assert out["metric"] == dropped
    assert plan["frame"] == "pixel" and "points_world" not in track and "points_camera" not in track


def test_a_schema_answer_with_another_count_is_refused_when_resampling_is_off(request_plan, obs):
    trace = TRACES["cube1/object/start"]["plan"]
    out = request_plan.run(
        FakeContext(tools(**{"robobrain.trace": trace})),
        instruction=CUBE_TO_BOX,
        start_pixels=[[184.0, 340.0]],
        planners=["robobrain.trace"],
        observation=obs["cube_box"],
        resample_paths=False,
    )
    assert out["route"] == "plan_rejected"
    assert out["detail"] == "robobrain.trace: rejected: arm 0: 3 waypoints, 10 declared"
    out = request_plan.run(
        FakeContext(tools(**{"robobrain.trace": trace})),
        instruction=CUBE_TO_BOX,
        start_pixels=[[184.0, 340.0]],
        planners=["robobrain.trace"],
        observation=obs["cube_box"],
        resample_paths=False,
        waypoints=3,
    )
    assert out["route"] == "planned" and "resampled" not in out["detail"]
    assert out["plan"]["tracks"][0]["pixels"] == trace["tracks"][0]["pixels"]


def test_every_real_robobrain_trace_is_planned_in_pixels_only(request_plan, obs):
    """Given the start it was asked from, each trace passes the image checks; none keeps its depth."""
    reasons = []
    for label, rec in TRACES.items():
        start = rec["args"].get("start_point")
        if start is None:
            continue
        frame = rec["frame"]
        out = request_plan.run(
            FakeContext(tools(hands(frame), **{"robobrain.trace": rec["plan"]})),
            instruction=rec["args"]["instruction"],
            start_pixels=[start],
            planners=[{"tool": "robobrain.trace", "args": {"mode": rec["args"]["mode"]}}],
            observation=obs[frame],
        )
        assert out["route"] == "planned", (label, out["detail"])
        assert out["planner"] == f"robobrain.trace(mode={rec['args']['mode']})"
        assert out["metric"].startswith("dropped: arm 0: "), (label, out["metric"])
        reasons.append(out["metric"].removeprefix("dropped: arm 0: "))
        strict = request_plan.run(
            FakeContext(tools(hands(frame), **{"robobrain.trace": rec["plan"]})),
            instruction=rec["args"]["instruction"],
            start_pixels=[start],
            planners=["robobrain.trace"],
            observation=obs[frame],
            require_metric=True,
        )
        assert strict["route"] == "plan_rejected", label
        assert "no metric part that agrees with the measured depth (dropped" in strict["detail"]
    # Without a depth anchor the model's start is 24-35 cm off; with one its end is 15-28 cm short.
    assert len(reasons) == 12
    assert sum(r.startswith("its start is ") for r in reasons) == 4
    assert sum(r.startswith("it ends ") and "in front of the surface" in r for r in reasons) == 8


def test_a_trace_that_starts_away_from_the_hand_is_refused(request_plan, obs):
    """robobrain.trace without a start put the gripper 37 px from where the ring is."""
    trace = TRACES["cube1/end_effector"]["plan"]
    out = request_plan.run(
        FakeContext(tools(**{"robobrain.trace": trace})),
        instruction=CUBE_TO_BOX,
        planners=["robobrain.trace"],
        observation=obs["cube_box"],
    )
    assert out["route"] == "plan_rejected"
    assert "arm 0: the path starts 37 px from its hand (bound 32 px)" in out["detail"]


START_OFF_HAND_MM = {0: 79, 1: 76}
"""The live traces' start (the table's depth under the ring) against the hand 8 cm over it."""
END_SHORT_MM = {
    "robobrain/cube_pick": [182],
    "robobrain/box_reach": [272],
    "robobrain/cube_to_box": [254],
    "robobrain/around_box": [132],
    "robobrain/dual_cube_box": [182, 272],
}
"""How far in front of the surface under its endpoint each live trace ends, per arm."""


def ends_short(label):
    return [
        f"it ends {mm} mm in front of the surface measured under its endpoint"
        for mm in END_SHORT_MM[label]
    ]


def ask_live(mod, obs, rec, at=None, **kw):
    """Ask robobrain.trace alone; it answers what it answered this script live, one reply per arm."""
    ctx = FakeContext(tools(at, **{"robobrain.trace": copy.deepcopy(rec["replies"])}))
    out = mod.run(
        ctx,
        instruction=rec["instruction"],
        arms=rec["arms"],
        arm_instructions=rec["arm_instructions"],
        planners=["robobrain.trace"],
        observation=obs[rec["frame"]],
        **kw,
    )
    return out, ctx


@pytest.mark.parametrize("rec", ANSWERS["robobrain_live"], ids=lambda r: r["label"])
def test_a_live_robobrain_answer_routes_as_it_did_live(request_plan, obs, rec):
    """Asked from a hand's ring with the depth image: a pixel plan, its own depths not passed on."""
    out, ctx = ask_live(request_plan, obs, rec)
    assert out["route"] == rec["live_route"] == "planned", out["detail"]
    calls = [c.kwargs for c in ctx.calls_to("robobrain.trace")]
    assert out["planner_calls"] == len(calls) == len(rec["arms"])  # one call per arm
    plan = out["plan"]
    assert plan["backend"] == "robobrain" and plan["frame"] == "pixel"
    for i, (arm, call, reply) in enumerate(zip(rec["arms"], calls, rec["replies"], strict=True)):
        asked = rec["arm_instructions"][i] if rec["arm_instructions"] else rec["instruction"]
        assert call["instruction"] == asked
        assert call["start_point"] == pytest.approx(hand_px(arm).tolist(), abs=1e-3)
        own, track = reply["tracks"][0], plan["tracks"][i]
        assert reply["depth_source"] == "model, start from depth map"
        assert f"arm {arm} resampled from {len(own['pixels'])}" in out["attempts"][0]
        assert track["arm_id"] == arm and track["active"] and track["start_error_px"] < 0.01
        assert len(track["pixels"]) == 10 and track["pixels"][-1] == pytest.approx(
            own["pixels"][-1]
        )
        assert "points_world" not in track and "points_camera" not in track
    # Two things are named for every arm. The trace's start is the depth image's reading under
    # the ring - the table between the fingers - and the hand is 8 cm over that (where the live
    # check put it: the script named arm 0's 79 mm, arm 1's 76 mm). And the trace ends in the air.
    assert rec["live_start_off_hand_mm"] == START_OFF_HAND_MM[rec["arms"][0]]
    assert out["metric"] == "dropped: " + "; ".join(
        f"arm {arm}: its start is {START_OFF_HAND_MM[arm]} mm off its hand's depth and {end}"
        for arm, end in zip(rec["arms"], ends_short(rec["label"]), strict=True)
    )
    assert out["min_arm_separation_m"] == pytest.approx(rec["live_min_arm_separation_m"], abs=1e-3)


@pytest.mark.parametrize(
    "label", [label for label, short in END_SHORT_MM.items() if len(short) == 1]
)
def test_a_live_trace_ends_in_the_air_over_what_it_reaches(request_plan, obs, label):
    """With the hand on the table under its ring the start agrees; the end is 13-27 cm short of the depth."""
    rec, (end,) = LIVE[label], ends_short(label)
    resting = {arm: on_ray(hand_px(arm), TABLE_Z) for arm in (0, 1)}
    out, _ = ask_live(request_plan, obs, rec, at=resting)
    assert out["route"] == "planned", out["detail"]
    assert out["metric"] == f"dropped: arm {rec['arms'][0]}: {end}"
    strict, _ = ask_live(request_plan, obs, rec, at=resting, require_metric=True)
    assert strict["route"] == "plan_rejected" and end in strict["detail"]


def test_robobrain_confirms_a_text_plan_only_where_its_own_endpoint_is_right(request_plan, obs):
    """``agree: 2`` on the default ladder: the text planner's plan, if robobrain.trace ends there too."""

    def both(text_label, trace_label):
        rec = BY_LABEL[text_label]
        ctx = FakeContext(
            tools(
                **{
                    "vlm.query": {"text": rec["text"]},
                    "robobrain.trace": copy.deepcopy(LIVE[trace_label]["replies"]),
                }
            )
        )
        out = request_plan.run(
            ctx,
            instruction=rec["instruction"],
            arms=rec["arms"],
            observation=obs["cube_box"],
            agree=2,
        )
        return out, np.array(LIVE[trace_label]["replies"][0]["tracks"][0]["pixels"][-1])

    for task, target in (("cube_pick", CUBE_PX), ("box_reach", BOX_PX)):
        out, traced = both(f"azure/{task}", f"robobrain/{task}")
        assert out["route"] == "planned", out["detail"]
        assert out["planner"] == "vlm.query" and out["planner_calls"] == 2
        assert out["plan"]["confirmed_by"] == ["robobrain.trace"]
        assert out["attempts"][-1].endswith("endpoints within 26 px of robobrain.trace's")
        assert np.linalg.norm(np.subtract(out["endpoint_pixels"][0], target)) < 5.0  # the text plan
        assert np.linalg.norm(traced - target) < 15.0

    # Asked to reach the cube around the box, the trace ended in the box: no agreement, no plan,
    # though the text planner's path ends on the cube.
    out, traced = both("azure/around_box", "robobrain/around_box")
    assert out["route"] == "plan_rejected" and out["plan"] == {}
    assert out["attempts"][0].startswith("vlm.query: valid (1 arm(s) x 10 waypoints)")
    assert out["attempts"][1].startswith(
        "robobrain.trace: valid (1 arm(s) x 10 waypoints, arm 1 resampled"
    )
    assert out["attempts"][-1] == "no 2 planners agree on the endpoints"
    assert np.linalg.norm(traced - CUBE_PX) > 110.0 and np.linalg.norm(traced - BOX_PX) < 20.0

    # A two-step instruction: the text planner plans the first step (to the cube), the trace
    # draws the whole manipulation (to the box). They cannot agree; give one step.
    out, traced = both("g3flash/cube_to_box", "robobrain/cube_to_box")
    assert out["route"] == "plan_rejected"
    assert np.linalg.norm(traced - BOX_PX) < 20.0


def test_a_schema_planner_is_asked_once_per_arm_with_its_own_instruction(request_plan, obs):
    left = {"found": True, "tracks": [{"pixels": [list(p) for p in flight(0, (0.10, 0.50))[1:]]}]}
    right = {"found": True, "tracks": [{"pixels": [list(p) for p in flight(1, (0.50, 0.50))[1:]]}]}
    ctx = FakeContext(tools(**{"molmo-motion.forecast": [left, right]}))
    out = request_plan.run(
        ctx,
        instruction="tidy the table",
        arms=[0, 1],
        arm_instructions=["sweep the left side", ""],
        planners=[{"tool": "molmo-motion.forecast", "args": {"horizon": 10}}],
        observation=obs["cube_box"],
    )
    assert out["route"] == "planned", out["detail"]
    assert out["planner"] == "molmo-motion.forecast(horizon=10)" and out["planner_calls"] == 2
    first, second = (c.kwargs for c in ctx.calls_to("molmo-motion.forecast"))
    assert first["instruction"] == "sweep the left side"
    assert second["instruction"] == "tidy the table"
    assert first["query_points"] == [pytest.approx(hand_px(0).tolist())]
    assert "start_point" not in first
    assert second["query_points"] == [pytest.approx(hand_px(1).tolist())]
    assert first["horizon"] == second["horizon"] == 10
    assert [c.kwargs for c in ctx.calls_to("robot.get_ee_pose")] == [{}, {"arm_id": 1}]
    assert out["min_arm_separation_m"] == pytest.approx(0.40, abs=0.005)  # the two ends


# ---------------------------------------------------------------------------
# The metric part
# ---------------------------------------------------------------------------


def metric_plan(n=10, end_lift=0.0, start_shift=0.0, sink=None, pixel_shift=0.0, arm=0, to=None):
    """A schema answer whose metric waypoints run straight from an arm's hand to the cube's top
    (or to ``to``, an (x, y) on the bare table)."""
    hand = hands()[arm]
    if to is None:
        top = on_ray(CUBE_PX, TABLE_Z + 0.03 + end_lift)
    else:
        top = np.array([to[0], to[1], TABLE_Z + end_lift])
    world = np.array(line(hand, top, n)[1:])
    if sink is not None:
        world[sink, 2] = TABLE_Z - 0.08  # under the table
    cam = np.array([to_camera(p) for p in world])
    pixels = cam[:, :2] / cam[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]] + pixel_shift
    start = to_camera(hand) + [0.0, 0.0, start_shift]
    track = {
        "query_pixel": hand_px(arm).tolist(),
        "pixels": pixels.tolist(),
        "start_camera": start.tolist(),
        "points_camera": cam.tolist(),
    }
    return {"found": True, "backend": "metric-planner", "tracks": [track]}, world


def ask_metric(request_plan, obs, plan, **kw):
    return request_plan.run(
        FakeContext(tools(**{"robobrain.trace": plan})),
        instruction="reach the cube",
        planners=["robobrain.trace"],
        observation=obs["cube_box"],
        **kw,
    )


def test_a_metric_part_that_agrees_with_the_depth_is_passed_on(request_plan, obs):
    plan, world = metric_plan()
    out = ask_metric(request_plan, obs, plan, require_metric=True)
    assert out["route"] == "planned", out["detail"]
    assert out["metric"] == "anchored" and out["plan"]["frame"] == "world"
    assert out["plan"]["backend"] == "metric-planner"
    track = out["plan"]["tracks"][0]
    np.testing.assert_allclose(track["points_world"], world, atol=1e-6)
    np.testing.assert_allclose(track["points_camera"], plan["tracks"][0]["points_camera"])
    assert "metric part anchored" in out["attempts"][0]


def test_a_resampled_metric_part_keeps_its_depths_along_the_path(request_plan, obs):
    plan, world = metric_plan(n=5)
    out = ask_metric(request_plan, obs, plan)
    assert out["route"] == "planned" and out["metric"] == "anchored", out["detail"]
    points = np.asarray(out["plan"]["tracks"][0]["points_world"])
    assert points.shape == (10, 3)
    np.testing.assert_allclose(points[-1], world[-1], atol=1e-6)
    # Even steps in the image along a straight 3D line: on the line, to the interpolation's error.
    hand = hands()[0]
    along = (world[-1] - hand) / np.linalg.norm(world[-1] - hand)
    off = (points - hand) - np.outer((points - hand) @ along, along)
    assert np.linalg.norm(off, axis=1).max() < 0.004


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"end_lift": 0.10}, "it ends 104 mm in front of the surface measured under its endpoint"),
        ({"start_shift": -0.10}, "its start is 100 mm off its hand's depth"),
        ({"sink": 4}, "metric waypoint 5 lies 80 mm behind a measured surface"),
        ({"pixel_shift": 25.0}, "metric waypoint 1 projects 35 px from its own pixel"),
    ],
)
def test_a_metric_part_that_disagrees_is_dropped_and_named(request_plan, obs, kwargs, why):
    plan, _ = metric_plan(**kwargs)
    out = ask_metric(request_plan, obs, plan)
    assert out["route"] == "planned", out["detail"]  # the image part still stands
    assert out["metric"] == f"dropped: arm 0: {why}"
    assert out["plan"]["frame"] == "pixel" and "points_world" not in out["plan"]["tracks"][0]
    strict = ask_metric(request_plan, obs, plan, require_metric=True)
    assert strict["route"] == "plan_rejected" and why in strict["detail"]
    loose = ask_metric(request_plan, obs, plan, metric_tolerance=0.2)
    assert loose["metric"] == ("anchored" if "pixel_shift" not in kwargs else out["metric"])


def test_every_disagreement_of_a_metric_part_is_named(request_plan, obs):
    plan, _ = metric_plan(end_lift=0.10, start_shift=-0.10, sink=4)
    out = ask_metric(request_plan, obs, plan)
    assert out["route"] == "planned", out["detail"]
    assert out["metric"] == (
        "dropped: arm 0: its start is 100 mm off its hand's depth and it ends 104 mm in front of "
        "the surface measured under its endpoint and metric waypoint 5 lies 79 mm behind a "
        "measured surface"
    )
    # Points that are not on their own pixels are compared with nothing else.
    plan, _ = metric_plan(end_lift=0.10, pixel_shift=25.0)
    out = ask_metric(request_plan, obs, plan)
    assert out["metric"] == "dropped: arm 0: metric waypoint 1 projects 35 px from its own pixel"


def test_a_metric_part_is_passed_on_for_every_moving_arm_or_for_none(request_plan, obs):
    def ask(left, right, **kw):
        return request_plan.run(
            FakeContext(tools(**{"robobrain.trace": [left, right]})),
            instruction="left hand to the cube, right hand to its side of the table",
            arms=[0, 1],
            planners=["robobrain.trace"],
            observation=obs["cube_box"],
            **kw,
        )

    def carried(out):
        return ["points_world" in t and "points_camera" in t for t in out["plan"]["tracks"]]

    left, left_world = metric_plan()
    right, right_world = metric_plan(arm=1, to=(0.50, 0.50))
    out = ask(left, right, require_metric=True)
    assert out["route"] == "planned", out["detail"]
    assert out["metric"] == "anchored" and out["plan"]["frame"] == "world"
    assert carried(out) == [True, True]
    np.testing.assert_allclose(out["plan"]["tracks"][0]["points_world"], left_world, atol=1e-6)
    np.testing.assert_allclose(out["plan"]["tracks"][1]["points_world"], right_world, atol=1e-6)

    # One arm's metric part disagrees: that arm is named, and neither arm's is passed on.
    high = "it ends 116 mm in front of the surface measured under its endpoint"  # along its ray
    out = ask(left, metric_plan(arm=1, to=(0.50, 0.50), end_lift=0.10)[0])
    assert out["route"] == "planned" and out["metric"] == f"dropped: arm 1: {high}"
    assert out["plan"]["frame"] == "pixel" and carried(out) == [False, False]
    strict = ask(left, metric_plan(arm=1, to=(0.50, 0.50), end_lift=0.10)[0], require_metric=True)
    assert strict["route"] == "plan_rejected" and f"(dropped: arm 1: {high})" in strict["detail"]

    # Both disagree: both are named.
    out = ask(
        metric_plan(start_shift=-0.10)[0], metric_plan(arm=1, to=(0.50, 0.50), end_lift=0.10)[0]
    )
    assert out["metric"] == (
        f"dropped: arm 0: its start is 100 mm off its hand's depth; arm 1: {high}"
    )

    # One arm's answer has no metric part at all: none for the plan.
    bare = {"found": True, "tracks": [{"pixels": right["tracks"][0]["pixels"]}]}
    out = ask(left, bare)
    assert out["route"] == "planned" and out["metric"] == "none"
    assert out["plan"]["frame"] == "pixel" and carried(out) == [False, False]


def test_a_text_plan_has_no_metric_part(request_plan, obs):
    out, _ = replay(request_plan, obs, BY_LABEL["azure/cube_pick"])
    assert out["metric"] == "none" and out["plan"]["frame"] == "pixel"
    strict, _ = replay(request_plan, obs, BY_LABEL["azure/cube_pick"], require_metric=True)
    assert strict["route"] == "plan_rejected"
    assert "no metric part that agrees with the measured depth (none)" in strict["detail"]


# ---------------------------------------------------------------------------
# Checks against the image and the depth
# ---------------------------------------------------------------------------


def ask_text(request_plan, observation, reply, at=None, **kw):
    ctx = FakeContext(tools(at, **{"vlm.query": reply}))
    kw.setdefault("planners", ["vlm.query"])
    out = request_plan.run(ctx, instruction="reach", observation=observation, **kw)
    return out, ctx


def test_a_waypoint_outside_the_image_is_refused(request_plan):
    # A camera whose answers can leave the frame: half as wide, same scene.
    cam = camera()
    cam = {**cam, "rgb": cam["rgb"][:, :400], "depth": cam["depth"][:, :400]}
    path = line(hand_px(0), (399.5, 300.0))
    red = [[u * 1000 / 400, v * 1000 / H] for u, v in path]
    reply = {"text": json.dumps({"subtask": "", "red": red})}
    out, _ = ask_text(request_plan, {"cameras": [cam]}, reply)
    assert out["route"] == "planned", out["detail"]  # the last pixel column is inside
    edge = [list(p) for p in path[1:]]
    wide = FakeContext(tools(**{"robobrain.trace": {"found": True, "tracks": [{"pixels": edge}]}}))
    out = request_plan.run(
        wide,
        instruction="reach",
        planners=["robobrain.trace"],
        observation={"cameras": [cam]},
        resample_paths=False,
    )
    assert out["route"] == "planned", out["detail"]
    past = [list(p) for p in line(hand_px(0), (420.0, 300.0))[1:]]
    wide = FakeContext(tools(**{"robobrain.trace": {"found": True, "tracks": [{"pixels": past}]}}))
    out = request_plan.run(
        wide, instruction="reach", planners=["robobrain.trace"], observation={"cameras": [cam]}
    )
    assert out["route"] == "plan_rejected"
    assert "arm 0: waypoint 10 (420, 300) px is outside the image" in out["detail"]


def test_an_endpoint_without_depth_is_refused(request_plan):
    cam = camera()
    cam["depth"] = cam["depth"].copy()
    cam["depth"][310:350, 285:325] = np.nan
    reply = says(red=line(hand_px(0), BOX_PX))
    out, _ = ask_text(request_plan, {"cameras": [cam]}, reply)
    assert out["route"] == "plan_rejected"
    assert "arm 0: no valid depth under the endpoint (303, 331) px" in out["detail"]


def test_no_motion_is_no_plan_and_one_still_arm_is_fine(request_plan, obs):
    still = [hand_px(0)] * 11
    out, _ = ask_text(request_plan, obs["cube_box"], says(red=still))
    assert out["route"] == "no_plan"
    assert out["attempts"] == [
        "vlm.query: no motion proposed: every path ends within 12 px of its start"
    ]
    both = says(red=still, blue=[hand_px(1)] * 11)
    out, _ = ask_text(request_plan, obs["cube_box"], both, arms=[0, 1])
    assert out["route"] == "no_plan"
    one = says(red=still, blue=line(hand_px(1), BOX_PX))
    out, _ = ask_text(request_plan, obs["cube_box"], one, arms=[0, 1])
    assert out["route"] == "planned", out["detail"]
    assert [t["active"] for t in out["plan"]["tracks"]] == [False, True]
    assert out["plan"]["tracks"][0]["endpoint_depth_m"] is None


# ---------------------------------------------------------------------------
# Two arms: the distance between them along the plan
# ---------------------------------------------------------------------------

P_RIGHT, Q_LEFT = (0.45, 0.55), (0.15, 0.50)  # arm 0 crosses to the right, arm 1 to the left


def crossing(request_plan, obs, **kw):
    reply = says(red=flight(0, P_RIGHT), blue=flight(1, Q_LEFT))
    return ask_text(request_plan, obs["cube_box"], reply, arms=[0, 1], **kw)[0]


def test_arms_that_cross_at_the_same_time_are_refused(request_plan, obs):
    out = crossing(request_plan, obs)
    assert out["route"] == "plan_rejected"
    assert out["attempts"] == [
        "vlm.query: rejected: arms 0 and 1 come within 44 mm (bound 150 mm) at 69 % of the plan "
        "(together)"
    ]


def test_the_order_decides_whether_crossing_paths_are_clear(request_plan, obs):
    # Arm 0 first: arm 1 then passes 178 mm from where arm 0 came to rest.
    out = crossing(request_plan, obs, order="a_then_b")
    assert out["route"] == "planned", out["detail"]
    assert out["min_arm_separation_m"] == pytest.approx(0.178, abs=0.003)
    assert 0.5 < out["separation_at_fraction"] < 1.0  # during the second arm's flight
    assert out["plan"]["min_arm_separation_m"] == out["min_arm_separation_m"]
    # Arm 1 first: arm 0 then passes 96 mm from where arm 1 came to rest.
    out = crossing(request_plan, obs, order="b_then_a")
    assert out["route"] == "plan_rejected"
    assert "come within 96 mm (bound 150 mm)" in out["detail"] and "(b_then_a)" in out["detail"]
    # The bound is the caller's: two closed hands may pass closer.
    assert crossing(request_plan, obs, order="b_then_a", min_hand_gap=0.09)["route"] == "planned"


def test_a_moving_arm_is_kept_off_a_resting_one(request_plan, obs):
    rest = hands()[1][:2]
    reply = says(red=flight(0, rest + [-0.06, 0.05]), blue=[hand_px(1)] * 11)
    out, _ = ask_text(request_plan, obs["cube_box"], reply, arms=[0, 1])
    assert out["route"] == "plan_rejected"
    assert "arms 0 and 1 come within 78 mm (bound 150 mm) at 100 % of the plan" in out["detail"]


def test_hands_already_close_may_move_apart_but_not_closer(request_plan, obs):
    at = {0: np.array([0.25, 0.30, TABLE_Z + LIFT]), 1: np.array([0.35, 0.30, TABLE_Z + LIFT])}
    T = pose_of()
    rings = [project(T, at[0]), project(T, at[1])]

    def reply(to0, to1):
        return says(
            red=[rings[0], *seen(line(at[0][:2], to0)[1:])],
            blue=[rings[1], *seen(line(at[1][:2], to1)[1:])],
        )

    apart, _ = ask_text(
        request_plan, obs["cube_box"], reply((0.10, 0.50), (0.50, 0.50)), at, arms=[0, 1]
    )
    assert apart["route"] == "planned", apart["detail"]
    assert apart["min_arm_separation_m"] == pytest.approx(0.10, abs=0.002)
    assert apart["separation_at_fraction"] == 0.0
    closer, _ = ask_text(
        request_plan, obs["cube_box"], reply((0.27, 0.50), (0.33, 0.50)), at, arms=[0, 1]
    )
    assert closer["route"] == "plan_rejected"
    assert "come within 60 mm" in closer["detail"]


def test_the_table_height_falls_back_to_the_depth_image(request_plan, obs):
    def none(**kw):
        raise ToolError("robot.describe_workspace", "not served")

    reply = says(red=flight(0, P_RIGHT), blue=flight(1, Q_LEFT))
    out, ctx = ask_text(request_plan, obs["cube_box"], reply, arms=[0, 1], order="a_then_b")
    assert out["route"] == "planned" and len(ctx.calls_to("robot.describe_workspace")) == 1
    with_tool = out["min_arm_separation_m"]
    ctx = FakeContext(tools(**{"vlm.query": reply, "robot.describe_workspace": none}))
    out = request_plan.run(
        ctx,
        instruction="reach",
        arms=[0, 1],
        planners=["vlm.query"],
        observation=obs["cube_box"],
        order="a_then_b",
    )
    assert out["route"] == "planned", out["detail"]
    assert out["min_arm_separation_m"] == pytest.approx(with_tool, abs=0.004)  # a 4 mm height bin
    given = FakeContext(tools(**{"vlm.query": reply, "robot.describe_workspace": none}))
    out = request_plan.run(
        given,
        instruction="reach",
        arms=[0, 1],
        planners=["vlm.query"],
        observation=obs["cube_box"],
        order="a_then_b",
        table_z=TABLE_Z,
    )
    assert out["min_arm_separation_m"] == pytest.approx(with_tool)
    assert given.calls_to("robot.describe_workspace") == []
    single, ctx = replay(request_plan, obs, BY_LABEL["azure/cube_pick"])
    assert ctx.calls_to("robot.describe_workspace") == []  # one arm: no table needed
    assert single["min_arm_separation_m"] == -1.0 and single["separation_at_fraction"] == -1.0


# ---------------------------------------------------------------------------
# An earlier plan
# ---------------------------------------------------------------------------


def test_an_earlier_plan_is_handed_out_again_while_nothing_changed(request_plan, obs):
    rec = BY_LABEL["azure/dual_cube_box"]
    first, _ = replay(request_plan, obs, rec)
    again, ctx = replay(request_plan, obs, rec, plan=json.loads(json.dumps(first["plan"])))
    assert again["route"] == "planned" and again["reused"] and again["planner_calls"] == 0
    assert ctx.calls_to("vlm.query") == []
    assert again["plan"] == first["plan"] and again["endpoint_pixels"] == first["endpoint_pixels"]
    assert again["planner"] == "vlm.query" and again["subtask"] == first["subtask"]
    assert again["min_arm_separation_m"] == first["min_arm_separation_m"]
    assert again["attempts"][0].startswith("earlier plan: reused (")
    assert not first["reused"]


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ("age", "it is 31 s old (bound 30 s)"),
        ("future", "s old (bound 30 s)"),
        ("hand", "arm 1's hand moved 30 mm since it was made"),
        ("scene", "the surface under arm 0's endpoint moved 30 mm"),
        ("no_depth", "the surface under arm 1's endpoint has no depth now"),
        ("instruction", "it was made for another request"),
        ("arms", "it was made for another request"),
        ("waypoints", "it was made for another request"),
        ("camera", "it was made for another request"),
        ("not_found", "it was made for another request"),
        ("foreign", "it was made for another request"),
        ("metric", "it has no metric part"),
    ],
)
def test_an_earlier_plan_is_asked_again_when_it_no_longer_holds(request_plan, obs, change, why):
    rec = BY_LABEL["azure/dual_cube_box"]
    first, _ = replay(request_plan, obs, rec)
    plan = copy.deepcopy(first["plan"])
    at, observation, kw = hands(), obs["cube_box"], {"instruction": rec["instruction"]}
    if change == "age":
        plan["planned_at"] -= 31.0
    elif change == "future":
        plan["planned_at"] += 3600.0  # another machine's clock
    elif change == "hand":
        at[1] = at[1] + [0.03, 0.0, 0.0]
    elif change in ("scene", "no_depth"):
        cam = camera()
        cam["depth"] = cam["depth"].copy()
        u, v = (int(round(c)) for c in plan["tracks"][0 if change == "scene" else 1]["pixels"][-1])
        patch = cam["depth"][v - 8 : v + 9, u - 8 : u + 9]
        # 3 cm nearer (something put there), or nothing measured (a hand over it).
        patch[...] = patch - 0.03 if change == "scene" else np.nan
        observation = {"cameras": [cam]}
    elif change == "instruction":
        kw["instruction"] = "put the cube back"
    elif change == "arms":
        plan["typed"]["arms"] = [1, 0]
    elif change == "waypoints":
        plan["typed"]["waypoints"] = 8
    elif change == "camera":
        plan["camera"] = "wrist"
    elif change == "not_found":
        plan["found"] = False
    elif change == "foreign":
        plan = TRACES["cube1/object/start"]["plan"]  # a plan this skill did not make
    elif change == "metric":
        kw["require_metric"] = True
    ctx = FakeContext(tools(at, **{"vlm.query": {"text": rec["text"]}}))
    out = request_plan.run(
        ctx, arms=[0, 1], planners=["vlm.query"], observation=observation, plan=plan, **kw
    )
    assert not out["reused"] and out["planner_calls"] == 1
    assert out["attempts"][0].startswith("earlier plan: not reused: ")
    assert why in out["attempts"][0]
    assert len(out["attempts"]) == 2  # and what the planner asked instead answered


def test_an_old_plan_is_reused_when_the_caller_sets_no_age_bound(request_plan, obs):
    rec = BY_LABEL["azure/cube_pick"]
    first, _ = replay(request_plan, obs, rec)
    old = {**first["plan"], "planned_at": first["plan"]["planned_at"] - 3600.0}
    assert not replay(request_plan, obs, rec, plan=old)[0]["reused"]
    assert replay(request_plan, obs, rec, plan=old, max_age_s=0.0)[0]["reused"]
    assert replay(request_plan, obs, rec, plan=old, max_age_s=7200.0)[0]["reused"]


# ---------------------------------------------------------------------------
# No plan, and declarations that are wrong
# ---------------------------------------------------------------------------


def test_no_plan_cases(request_plan, obs):
    reply = says(red=line(hand_px(0), CUBE_PX))
    out, ctx = ask_text(request_plan, obs["cube_box"], reply, planners=[])
    assert out["route"] == "no_plan" and out["detail"] == "no planner to ask"
    out = request_plan.run(FakeContext(tools()), instruction="  ", observation=obs["cube_box"])
    assert out["route"] == "no_plan" and out["detail"] == "no instruction to ask a planner with"
    no_depth = {"cameras": [{**camera(), "depth": None}]}
    out, _ = ask_text(request_plan, no_depth, reply)
    assert out["route"] == "no_plan" and "no camera with rgb, depth" in out["detail"]
    out, _ = ask_text(request_plan, obs["cube_box"], reply, camera_name="wrist")
    assert out["route"] == "no_plan" and "no usable camera named 'wrist'" in out["detail"]
    behind = {0: np.array([-1.5, 0.3, 0.06])}  # far left of what the camera sees
    out, ctx = ask_text(request_plan, obs["cube_box"], reply, behind)
    assert out["route"] == "no_plan" and ctx.calls_to("vlm.query") == []
    assert "arm 0's tool centre is not in the top image" in out["detail"]
    out, _ = ask_text(request_plan, obs["cube_box"], reply, behind, start_pixels=[[134.0, 366.0]])
    assert out["route"] == "planned"  # a start the caller gives needs no hand in view


def test_planners_that_raise_or_have_no_answer_are_no_plan(request_plan, obs):
    def down(**kw):
        raise ToolError("vlm.query", "quota exceeded")

    none = {"found": False, "tracks": [], "error": "no trajectory in the model's answer"}
    ctx = FakeContext(tools(**{"vlm.query": down, "robobrain.trace": none}))
    out = request_plan.run(ctx, instruction="reach the cube", observation=obs["cube_box"])
    assert out["route"] == "no_plan" and out["planner_calls"] == 2 and out["plan"] == {}
    assert out["attempts"][0].startswith("vlm.query: raised: ") and "quota" in out["attempts"][0]
    assert out["attempts"][1] == (
        "robobrain.trace: no answer: arm 0: no trajectory in the model's answer"
    )
    assert out["detail"] == " | ".join(out["attempts"])
    # One refusal among them makes it plan_rejected: a planner did answer.
    ctx = FakeContext(tools(**{"vlm.query": {"text": "no"}, "robobrain.trace": none}))
    out = request_plan.run(ctx, instruction="reach the cube", observation=obs["cube_box"])
    assert out["route"] == "plan_rejected"
    empty = {"found": True, "tracks": [{"query_pixel": hand_px(0).tolist(), "pixels": []}]}
    out = request_plan.run(
        FakeContext(tools(**{"robobrain.trace": empty})),
        instruction="reach the cube",
        planners=["robobrain.trace"],
        observation=obs["cube_box"],
    )
    assert out["route"] == "plan_rejected"
    assert "arm 0: a waypoint is missing or not a finite number" in out["detail"]


def test_the_observation_is_taken_from_the_robot_when_not_given(request_plan, obs):
    wrist = {**camera(), "name": "wrist", "depth": None}  # no depth: not usable
    seen_by = {"cameras": [wrist, *obs["cube_box"]["cameras"]]}
    reply = says(red=line(hand_px(0), CUBE_PX), subtask="reach the cube")
    ctx = FakeContext(tools(**{"vlm.query": reply, "robot.get_observation": seen_by}))
    out = request_plan.run(ctx, instruction="reach the cube", planners=["vlm.query"])
    assert out["route"] == "planned", out["detail"]
    assert len(ctx.calls_to("robot.get_observation")) == 1
    assert out["plan"]["camera"] == "top" and out["plan"]["image_size"] == [W, H]
    assert out["plan"]["instruction"] == "reach the cube" and out["subtask"] == "reach the cube"


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"arms": [0, 1, 2]}, "arms is one or two different arm ids"),
        ({"arms": []}, "arms is one or two different arm ids"),
        ({"arms": [1, 1]}, "arms is one or two different arm ids"),
        ({"waypoints": 1}, "waypoints is at least 2"),
        ({"order": "in_turns"}, "order is one of"),
        (
            {"arms": [0, 1], "start_pixels": [[1.0, 2.0]]},
            "start_pixels has one entry per declared arm",
        ),
        ({"arm_instructions": ["a", "b"]}, "arm_instructions has one entry per declared arm"),
    ],
)
def test_a_wrong_declaration_raises(request_plan, obs, kwargs, why):
    with pytest.raises(ValueError, match=why):
        request_plan.run(
            FakeContext(tools()), instruction="reach", observation=obs["cube_box"], **kwargs
        )


def test_other_counts_of_waypoints(request_plan, obs):
    for n in (2, 5, 20):
        reply = says(red=line(hand_px(0), CUBE_PX, n))
        out, ctx = ask_text(request_plan, obs["cube_box"], reply, waypoints=n)
        assert out["route"] == "planned", out["detail"]
        assert f"exactly {n} waypoints" in ctx.calls_to("vlm.query")[0].kwargs["prompt"]
        assert len(out["plan"]["tracks"][0]["pixels"]) == n and out["plan"]["horizon"] == n
    out, _ = ask_text(request_plan, obs["cube_box"], says(red=line(hand_px(0), CUBE_PX, 8)))
    assert out["route"] == "plan_rejected"
    assert "the red path has 8 waypoints after its ring's centre, 10 declared" in out["detail"]
