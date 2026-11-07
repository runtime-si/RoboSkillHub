"""robobrain bundle: prompts, parsing and the visual-plan output with the model replaced by a fake."""

from __future__ import annotations

import numpy as np
import pytest
from gap_core.errors import ToolError

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
W, H = 640, 480


@pytest.fixture(scope="module")
def rb(skills_registry):
    return skills_registry.get("robobrain").tools_module


class FakeRuntime:
    """Answers with scripted text and records what the model would have been given."""

    def __init__(self, answers=None, fail=None):
        self.answers = list(answers or [])
        self.fail = fail
        self.calls = []

    def generate(self, segments, *, answer_prefix="", max_new_tokens=768):
        if self.fail:
            raise self.fail
        self.calls.append(
            {"segments": segments, "prefix": answer_prefix, "max_new_tokens": max_new_tokens}
        )
        return self.answers.pop(0), 0.01, 0.5


@pytest.fixture
def fake(rb, monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(rb, "_get_runtime", lambda: (runtime, 0.0))
    return runtime


def frame(h=H, w=W, value=0):
    return np.full((h, w, 3), value, dtype=np.uint8)


def texts(call):
    return [s for s in call["segments"] if isinstance(s, str)]


def images(call):
    return [s for s in call["segments"] if not isinstance(s, str)]


# --- registration ---------------------------------------------------------


def test_tools_registered_with_their_schemas(tool_registry):
    for name in ("robobrain.point", "robobrain.trace", "robobrain.progress"):
        assert name in tool_registry
    trace = tool_registry.get("robobrain.trace").schema
    assert set(trace.inputs) == {
        "image",
        "instruction",
        "start_point",
        "depth",
        "intrinsics",
        "camera_pose",
        "mode",
    }
    assert trace.inputs["mode"].default == "end_effector"
    # The visual-plan fields molmo-motion.forecast returns, plus robobrain's extras.
    forecast = tool_registry.get("molmo-motion.forecast").schema
    assert set(forecast.outputs) <= set(trace.outputs)
    assert set(trace.outputs) - set(forecast.outputs) == {
        "mode",
        "depth_source",
        "measured_start_depth_m",
    }
    point = tool_registry.get("robobrain.point").schema
    gemini = tool_registry.get("gemini-er.point").schema
    assert set(gemini.outputs) <= set(point.outputs)
    assert set(tool_registry.get("robobrain.progress").schema.inputs) == {
        "image",
        "task",
        "start_image",
        "goal_image",
        "wrist_images",
        "start_wrist_images",
    }


def test_import_does_not_pull_torch(rb):
    assert rb._runtime is None
    assert not {"torch", "transformers"} & set(vars(rb))


# --- point ------------------------------------------------------------------


def test_point_uses_the_card_prompt_and_converts_0_1000_to_pixels(rb, fake):
    fake.answers = ["[(500, 250)]"]
    out = rb.point(frame(), "the small cube")
    assert out["found"] and out["pixel_x"] == pytest.approx(320.0)
    assert out["pixel_y"] == pytest.approx(120.0)
    assert out["points"] == [{"pixel_x": 320.0, "pixel_y": 120.0, "label": "the small cube"}]
    (call,) = fake.calls
    assert texts(call) == [
        "the small cube. Please provide its 2D coordinates. Your answer should be formatted as a "
        "tuple, i.e. [(x, y)], where the tuple contains the x and y coordinates of a point "
        "satisfying the conditions above."
    ]
    assert images(call)[0].shape == (H, W, 3)
    assert call["prefix"] == ""


def test_point_keeps_several_points_up_to_the_cap_and_clamps(rb, fake):
    fake.answers = ["[(100, 100), (1000, 1000), (200.5, 300.5)]"]
    out = rb.point(frame(), "rings", max_points=2)
    assert [(p["pixel_x"], p["pixel_y"]) for p in out["points"]] == [
        (64.0, 48.0),
        (W - 1.0, H - 1.0),
    ]


def test_point_reads_bracketed_pairs(rb, fake):
    fake.answers = ["[[250, 500]]"]
    out = rb.point(frame(), "x")
    assert (out["pixel_x"], out["pixel_y"]) == (160.0, 240.0)


def test_point_nothing_parsed_is_not_found(rb, fake):
    fake.answers = ["I cannot see that."]
    out = rb.point(frame(), "a unicorn")
    assert out == {
        "pixel_x": 0.0,
        "pixel_y": 0.0,
        "found": False,
        "points": [],
        "raw": "I cannot see that.",
    }


def test_point_rejects_bad_inputs(rb, fake):
    with pytest.raises(ValueError):
        rb.point(np.zeros((4, 4), dtype=np.uint8), "x")
    with pytest.raises(ValueError):
        rb.point(frame(), "  ")
    assert fake.calls == []


# --- trace ------------------------------------------------------------------


def test_trace_without_camera_is_pixels_in_the_plan_schema(rb, fake):
    fake.answers = ["[(500, 500, 0.80), (600, 500, 0.70), (700, 250, 0.60)]"]
    out = rb.trace(frame(), "move the cube right")
    assert out["found"] and out["frame"] == "pixel" and out["backend"] == "robobrain"
    assert out["horizon"] == 2 and out["dt_s"] is None and out["mode"] == "end_effector"
    (track,) = out["tracks"]
    assert track["query_pixel"] == [320.0, 240.0]
    assert track["pixels"] == [[384.0, 240.0], [448.0, 120.0]]
    assert track["predicted"] == [True, True]
    assert track["start_depth_m"] == 0.8 and track["depths_m"] == [0.7, 0.6]
    assert track["points_camera"] == [] and track["points_world"] == []
    assert out["depth_source"] == "model" and out["measured_start_depth_m"] is None
    (call,) = fake.calls
    assert texts(call)[0].startswith(
        "Please predict 3D end-effector-centric waypoints to complete the task successfully. "
        'The task is "move the cube right".'
    )


def test_trace_with_intrinsics_and_pose_gives_camera_and_world_points(rb, fake):
    fake.answers = ["[(500, 500, 0.80), (600, 500, 0.50)]"]
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    out = rb.trace(frame(), "lift", intrinsics=K, camera_pose=T)
    (track,) = out["tracks"]
    assert out["frame"] == "world"
    assert track["start_camera"] == pytest.approx([0.0, 0.0, 0.8])
    # u = 384 at z = 0.5: x = (384 - 320) / 500 * 0.5
    assert track["points_camera"][0] == pytest.approx([0.064, 0.0, 0.5])
    assert track["start_world"] == pytest.approx([1.0, 2.0, 3.8])
    assert track["points_world"][0] == pytest.approx([1.064, 2.0, 3.5])


def test_trace_world_points_from_an_se3_pose_dict(rb, fake):
    fake.answers = ["[(500, 500, 1.0), (500, 500, 0.5)]"]
    pose = {
        "position": {"x": 0.0, "y": 0.0, "z": 1.0},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    out = rb.trace(frame(), "x", intrinsics=K, camera_pose=pose)
    assert out["tracks"][0]["points_world"][0] == pytest.approx([0.0, 0.0, 1.5])


def test_start_point_is_written_as_the_start_of_the_answer(rb, fake):
    fake.answers = ["0.75), (600, 500, 0.70)]"]
    out = rb.trace(frame(), "push it", start_point=[321.0, 239.0], mode="object")
    (call,) = fake.calls
    assert call["prefix"] == "[(502, 498, "
    assert texts(call)[0].startswith(
        'Point the 3D object-centric visual trace for the task "push it"'
    )
    (track,) = out["tracks"]
    assert track["query_pixel"] == [321.0, 239.0]  # exact, not the 0-1000 rounding
    assert track["start_depth_m"] == 0.75 and track["pixels"] == [[384.0, 240.0]]
    assert out["raw"] == "[(502, 498, 0.75), (600, 500, 0.70)]"
    assert out["mode"] == "object"


def test_start_point_with_depth_writes_the_measured_depth_too(rb, fake):
    fake.answers = ["600, 500, 0.70)]"]
    depth = np.full((H, W), 0.9, dtype=np.float32)
    out = rb.trace(frame(), "x", start_point=[320.0, 240.0], depth=depth, intrinsics=K)
    assert fake.calls[0]["prefix"] == "[(500, 500, 0.90), ("
    (track,) = out["tracks"]
    assert track["start_camera"] == pytest.approx([0.0, 0.0, 0.9])
    assert out["depth_source"] == "model, start from depth map"
    assert out["measured_start_depth_m"] == pytest.approx(0.9)


def test_start_on_a_depth_hole_falls_back_to_the_model_depth(rb, fake):
    fake.answers = ["0.8), (600, 500, 0.7)]"]
    depth = np.zeros((H, W), dtype=np.float32)
    out = rb.trace(frame(), "x", start_point=[320.0, 240.0], depth=depth, intrinsics=K)
    assert fake.calls[0]["prefix"] == "[(500, 500, "
    assert out["depth_source"] == "model" and out["measured_start_depth_m"] is None


def test_depth_without_start_reports_the_measured_depth_at_the_models_start(rb, fake):
    fake.answers = ["[(500, 500, 0.80), (600, 500, 0.70)]"]
    depth = np.full((H, W), 0.85, dtype=np.float32)
    out = rb.trace(frame(), "x", depth=depth, intrinsics=K)
    assert fake.calls[0]["prefix"] == ""
    assert out["measured_start_depth_m"] == pytest.approx(0.85)
    assert out["tracks"][0]["start_camera"][2] == pytest.approx(0.8)  # the model's own


def test_trace_of_2d_tuples_stays_in_pixels(rb, fake):
    fake.answers = ["[(500, 500), (600, 500)]"]
    out = rb.trace(frame(), "x", intrinsics=K)
    assert out["frame"] == "pixel" and out["depth_source"] == "none"
    assert out["tracks"][0]["depths_m"] == [None]


def test_single_or_no_keypoint_is_not_found(rb, fake):
    fake.answers = ["[(500, 500, 0.8)]", "no idea"]
    one = rb.trace(frame(), "x")
    assert not one["found"] and one["horizon"] == 0
    assert one["tracks"][0]["query_pixel"] == [320.0, 240.0]
    none = rb.trace(frame(), "x", start_point=[10.0, 20.0])
    assert not none["found"] and none["tracks"][0]["query_pixel"] == [10.0, 20.0]


def test_trace_rejects_bad_inputs(rb, fake):
    with pytest.raises(ValueError, match="mode"):
        rb.trace(frame(), "x", mode="joint")
    with pytest.raises(ValueError, match="instruction"):
        rb.trace(frame(), "")
    with pytest.raises(ValueError, match="depth"):
        rb.trace(frame(), "x", depth=np.zeros((10, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="start_point"):
        rb.trace(frame(), "x", start_point=[1.0, 2.0, 3.0])
    assert fake.calls == []


def test_model_failures_raise_tool_error(rb, monkeypatch):
    monkeypatch.setattr(
        rb, "_get_runtime", lambda: (FakeRuntime(fail=RuntimeError("CUDA out of memory")), 0.0)
    )
    with pytest.raises(ToolError, match="out of memory"):
        rb.trace(frame(), "x")

    def broken():
        raise OSError("no snapshot")

    monkeypatch.setattr(rb, "_get_runtime", broken)
    with pytest.raises(ToolError, match="failed to load"):
        rb.point(frame(), "x")


# --- progress ---------------------------------------------------------------


def test_progress_builds_robo_dopamines_eight_image_forward_prompt(rb, fake):
    fake.answers = ["<score>+40%</score>"]
    start, now = frame(value=10), frame(value=200)
    out = rb.progress(now, "push the cube into the box", start)
    assert out == {
        "progress": 0.4,
        "score": 0.4,
        "found": True,
        "raw": "<score>+40%</score>",
        "timing": out["timing"],
    }
    (call,) = fake.calls
    ims = images(call)
    assert len(ims) == 8
    # 640 x 480 is scaled to Robo-Dopamine's 76800-pixel budget.
    assert ims[0].shape == (240, 320, 3)
    means = [int(im.mean()) for im in ims]
    # ref start, ref end (blank grey), before x3 (start), after x3 (now)
    assert means == [10, 237, 10, 10, 10, 200, 200, 200]
    assert "`push the cube into the box`" in texts(call)[0]
    assert len(texts(call)) == 9 and "<score>" in texts(call)[-1]


def test_progress_uses_goal_and_wrist_views_when_given(rb, fake):
    fake.answers = ["<score>-25%</score>"]
    out = rb.progress(
        frame(value=1),
        "t",
        frame(value=2),
        goal_image=frame(value=3),
        wrist_images=[frame(value=4), frame(value=5)],
        start_wrist_images=[frame(value=6), frame(value=7)],
    )
    assert out["score"] == -0.25 and out["progress"] == 0.0
    assert [int(im.mean()) for im in images(fake.calls[0])] == [2, 3, 2, 6, 7, 1, 4, 5]


def test_progress_clips_and_handles_a_missing_score(rb, fake):
    fake.answers = ["<score>+150%</score>", "maybe half"]
    assert rb.progress(frame(), "t", frame())["progress"] == 1.0
    out = rb.progress(frame(), "t", frame())
    assert not out["found"] and out["progress"] == 0.0


def test_progress_rejects_bad_inputs(rb, fake):
    with pytest.raises(ValueError, match="wrist"):
        rb.progress(frame(), "t", frame(), wrist_images=[frame()])
    with pytest.raises(ValueError, match="task"):
        rb.progress(frame(), "", frame())
    assert fake.calls == []
