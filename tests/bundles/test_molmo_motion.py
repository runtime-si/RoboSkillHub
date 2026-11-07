"""molmo-motion bundle: forecast plumbing with the model replaced by a fake runtime."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from gap_core.errors import ToolError

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


@pytest.fixture(scope="module")
def mm(skills_registry):
    return skills_registry.get("molmo-motion").tools_module


class FakeRuntime:
    """Records what the model would get; every point moves by ``step`` metres a frame."""

    history_size = 3
    num_points = 8

    def __init__(self, step=(0.01, 0.0, 0.0), emitted=None, fail=None):
        self.step = np.asarray(step, dtype=np.float64)
        self.emitted = emitted
        self.fail = fail
        self.calls = []

    def predict(self, frames, points_2d, points_3d_history, action, horizon):
        if self.fail:
            raise self.fail
        self.calls.append(
            {
                "frames": frames,
                "points_2d": np.array(points_2d),
                "history": np.array(points_3d_history),
                "action": action,
                "horizon": horizon,
            }
        )
        start = points_3d_history[-1]
        k = np.arange(1, horizon + 1)[None, :, None]
        future = start[:, None, :] + k * self.step[None, None, :]
        predicted = np.ones((self.num_points, horizon), dtype=bool)
        if self.emitted is not None:
            predicted[:, :] = False
            predicted[:, : self.emitted] = True
        return future.astype(np.float32), predicted, "<tracks coords=...>", 0.01


@pytest.fixture
def fake(mm, monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(mm, "_get_runtime", lambda: (runtime, 0.0))
    return runtime


def frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def flat_depth(z=0.8, h=480, w=640):
    return np.full((h, w), z, dtype=np.float32)


def test_forecast_registered_with_its_schema(tool_registry):
    assert "molmo-motion.forecast" in tool_registry
    schema = tool_registry.get("molmo-motion.forecast").schema
    assert set(schema.inputs) == {
        "image",
        "query_points",
        "instruction",
        "depth",
        "intrinsics",
        "camera_pose",
        "query_history",
        "horizon",
    }
    for name in ("image", "query_points", "instruction"):
        assert schema.inputs[name].required
    assert schema.inputs["horizon"].default == 30
    assert set(schema.outputs) == {
        "found",
        "tracks",
        "horizon",
        "dt_s",
        "frame",
        "backend",
        "model",
        "timing",
        "raw",
    }


def test_import_does_not_pull_torch(mm):
    assert mm._runtime is None
    assert "molmo_motion" not in sys.modules


def test_depth_and_camera_give_world_tracks(mm, fake):
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    out = mm.forecast(
        frame(),
        [[320.0, 240.0]],
        "push it right",
        depth=flat_depth(0.8),
        intrinsics=K,
        camera_pose=T,
        horizon=5,
    )
    assert out["found"] and out["frame"] == "world" and out["backend"] == "molmo-motion"
    assert out["horizon"] == 5 and out["dt_s"] == pytest.approx(1 / 15)
    (track,) = out["tracks"]
    assert track["query_pixel"] == [320.0, 240.0]
    assert track["start_camera"] == pytest.approx([0.0, 0.0, 0.8])
    assert track["start_world"] == pytest.approx([1.0, 2.0, 3.8])
    assert len(track["points_camera"]) == 5 and len(track["pixels"]) == 5
    assert track["points_camera"][-1] == pytest.approx([0.05, 0.0, 0.8])
    assert track["points_world"][-1] == pytest.approx([1.05, 2.0, 3.8])
    # 5 cm right at 0.8 m with f=500 is 31.25 px right
    assert track["pixels"][-1] == pytest.approx([351.25, 240.0])
    assert track["predicted"] == [True] * 5


def test_se3_camera_pose_is_accepted(mm, fake):
    pose = {
        "position": {"x": 0.0, "y": 0.0, "z": 1.0},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    out = mm.forecast(
        frame(), [[320, 240]], "x", depth=flat_depth(0.5), intrinsics=K, camera_pose=pose
    )
    assert out["tracks"][0]["start_world"] == pytest.approx([0.0, 0.0, 1.5])


def test_model_gets_eight_points_anchored_on_the_first_query(mm, fake):
    mm.forecast(
        frame(), [[100, 200], [300, 250]], "pull", depth=flat_depth(), intrinsics=K, horizon=4
    )
    (call,) = fake.calls
    assert call["points_2d"].shape == (8, 2) and call["history"].shape == (3, 8, 3)
    assert call["points_2d"][:2] == pytest.approx(np.array([[100, 200], [300, 250]]))
    # padding sits on a 4 px ring around the queries, taking turns
    off = call["points_2d"][2:] - call["points_2d"][[0, 1, 0, 1, 0, 1]]
    assert np.linalg.norm(off, axis=1) == pytest.approx([4.0] * 6)
    # without a query history the points stand still over the 3 frames
    assert np.allclose(call["history"][0], call["history"][2])
    assert call["action"] == "pull" and call["horizon"] == 4


def test_single_frame_is_repeated_to_the_history_size(mm, fake):
    img = frame()
    img[0, 0] = 7
    mm.forecast(img, [[10, 10]], "x")
    frames = fake.calls[0]["frames"]
    assert len(frames) == 3 and all(f[0, 0, 0] == 7 for f in frames)


def test_longer_history_keeps_the_last_frames(mm, fake):
    seq = [np.full((48, 64, 3), i, dtype=np.uint8) for i in range(5)]
    mm.forecast(seq, [[10, 10]], "x")
    assert [int(f[0, 0, 0]) for f in fake.calls[0]["frames"]] == [2, 3, 4]


def test_query_history_moves_the_3d_history(mm, fake):
    hist = [[[300.0, 240.0]], [[310.0, 240.0]], [[320.0, 240.0]]]
    mm.forecast(
        frame(),
        [[320.0, 240.0]],
        "x",
        depth=[flat_depth(1.0)] * 3,
        intrinsics=K,
        query_history=hist,
    )
    xs = fake.calls[0]["history"][:, 0, 0]
    assert xs == pytest.approx([-0.04, -0.02, 0.0])


def test_more_than_eight_queries_take_two_calls(mm, fake):
    queries = [[10.0 * i + 50, 100.0] for i in range(10)]
    out = mm.forecast(frame(), queries, "x", depth=flat_depth(), intrinsics=K, horizon=3)
    assert len(fake.calls) == 2
    assert [t["query_pixel"] for t in out["tracks"]] == queries
    assert fake.calls[1]["points_2d"][:2] == pytest.approx(np.array(queries[8:]))


def test_without_depth_only_pixels_come_back(mm, fake):
    out = mm.forecast(frame(), [[320, 240]], "x", horizon=3)
    assert out["frame"] == "pixel"
    track = out["tracks"][0]
    assert track["points_camera"] == [] and track["points_world"] == []
    assert len(track["pixels"]) == 3
    # the 1 m assumed depth and a focal of 640 px: 1 cm a step is 6.4 px
    assert track["pixels"][0] == pytest.approx([326.4, 240.0])


def test_camera_frame_without_pose(mm, fake):
    out = mm.forecast(frame(), [[320, 240]], "x", depth=flat_depth(), intrinsics=K)
    assert out["frame"] == "camera" and out["tracks"][0]["points_world"] == []


def test_depth_window_skips_holes(mm, fake):
    depth = flat_depth(0.9)
    depth[240, 320] = 0.0
    out = mm.forecast(frame(), [[320, 240]], "x", depth=depth, intrinsics=K)
    assert out["tracks"][0]["start_camera"][2] == pytest.approx(0.9)


def test_no_depth_near_a_query_raises(mm, fake):
    depth = flat_depth(0.9)
    depth[230:251, 310:331] = np.nan
    with pytest.raises(ToolError, match="no valid depth"):
        mm.forecast(frame(), [[320, 240]], "x", depth=depth, intrinsics=K)


def test_steps_the_model_did_not_emit_hold_the_last_one(mm, monkeypatch):
    runtime = FakeRuntime(emitted=2)
    monkeypatch.setattr(mm, "_get_runtime", lambda: (runtime, 0.0))
    out = mm.forecast(frame(), [[320, 240]], "x", depth=flat_depth(1.0), intrinsics=K, horizon=4)
    track = out["tracks"][0]
    assert track["predicted"] == [True, True, False, False]
    xs = [p[0] for p in track["points_camera"]]
    assert xs == pytest.approx([0.01, 0.02, 0.02, 0.02])


def test_nothing_emitted_is_not_found(mm, monkeypatch):
    runtime = FakeRuntime(emitted=0)
    monkeypatch.setattr(mm, "_get_runtime", lambda: (runtime, 0.0))
    out = mm.forecast(frame(), [[320, 240]], "x", depth=flat_depth(1.0), intrinsics=K, horizon=3)
    assert out["found"] is False
    assert out["tracks"][0]["points_camera"] == [[0.0, 0.0, 1.0]] * 3


def test_model_failure_is_a_tool_error(mm, monkeypatch):
    runtime = FakeRuntime(fail=RuntimeError("CUDA out of memory"))
    monkeypatch.setattr(mm, "_get_runtime", lambda: (runtime, 0.0))
    with pytest.raises(ToolError, match="out of memory"):
        mm.forecast(frame(), [[320, 240]], "x")


def test_load_failure_is_a_tool_error(mm, monkeypatch):
    def boom():
        raise OSError("disk full")

    monkeypatch.setattr(mm, "_get_runtime", boom)
    with pytest.raises(ToolError, match="failed to load"):
        mm.forecast(frame(), [[320, 240]], "x")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"instruction": " "}, "instruction"),
        ({"horizon": 0}, "horizon"),
        ({"horizon": 33}, "horizon"),
        ({"query_points": []}, "query_points"),
        ({"image": np.zeros((4, 4), dtype=np.uint8)}, "RGB"),
        ({"image": np.zeros((4, 4, 3), dtype=np.float32)}, "RGB"),
        ({"depth": flat_depth()}, "intrinsics"),
        ({"depth": flat_depth(h=10), "intrinsics": K}, "depth must be"),
        ({"query_history": [[[1.0, 2.0]]] * 3, "query_points": [[1, 2], [3, 4]]}, "query_history"),
    ],
)
def test_bad_requests_raise_before_the_model(mm, fake, kwargs, match):
    args = {"image": frame(), "query_points": [[320, 240]], "instruction": "x"}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        mm.forecast(**args)
    assert fake.calls == []


def test_pinned_checkpoint_and_model_id(mm, monkeypatch):
    monkeypatch.delenv("GAP_MOLMO_MOTION_MODEL", raising=False)
    monkeypatch.delenv("GAP_MOLMO_MOTION_REPO", raising=False)
    assert (
        mm._model_id() == "allenai/MolmoMotion-4B-H3-F30@3f5e790a511ff2cdf21c8d2a14cb4d8409c94629"
    )
    monkeypatch.setenv("GAP_MOLMO_MOTION_REPO", "allenai/MolmoMotion-4B-H1-F32")
    assert mm._model_id().endswith("@14e69b0d5dd55b2f09c885030b8756cd053ea6a4")
    monkeypatch.setenv("GAP_MOLMO_MOTION_REPO", "someone/else")
    with pytest.raises(ValueError, match="GAP_MOLMO_MOTION_REPO"):
        mm._model_id()


def test_weights_cached_with_a_local_dir(mm, monkeypatch, tmp_path):
    monkeypatch.setenv("GAP_MOLMO_MOTION_MODEL", str(tmp_path))
    assert mm.weights_cached() is False
    (tmp_path / "config.yaml").write_text("model: {}")
    (tmp_path / "model.pt").write_bytes(b"")
    assert mm.weights_cached() is True
    assert mm._model_id() == str(tmp_path)


def test_skill_md_matches_the_pinned_revision(mm):
    from pathlib import Path

    text = (Path(mm.__file__).parent / "SKILL.md").read_text()
    repo = mm.DEFAULT_REPO
    assert f"hf://{repo}@{mm.REVISIONS[repo]}" in text
