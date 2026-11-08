"""track bundle: pixel <-> model mapping, clip and session plumbing with TAPNext++ replaced by a fake."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
from gap_core.errors import ToolError

W, H = 640, 480
BUNDLE = Path(__file__).resolve().parents[2] / "tools" / "track"


@pytest.fixture(scope="module")
def tr(skills_registry):
    return skills_registry.get("track").tools_module


class FakeRuntime:
    """Every query moves ``delta`` model px ([y, x]) per frame stepped; frames are told apart by value.

    ``vis`` maps a step to the visibility logit of every query (default +5).
    """

    def __init__(self, delta=(0.0, 0.0), vis=None, cert=1.0, fail=None):
        self.delta = np.asarray(delta, dtype=np.float64)
        self.vis = vis or {}
        self.cert = cert
        self.fail = fail
        self.seen = []  # frame values in the order the model got them
        self.queries = []

    def _out(self, frame, state):
        if self.fail:
            raise self.fail
        self.seen.append(int(frame[0, 0, 0]))
        q, k = state
        n = len(q)
        vis = np.full(n, float(self.vis.get(k, 5.0)))
        return q + k * self.delta, vis, np.full(n, self.cert), (q, k)

    def init(self, frame, queries_yx):
        self.queries.append(np.array(queries_yx))
        return self._out(frame, (np.array(queries_yx, dtype=np.float64), 0))

    def step(self, frame, state):
        q, k = state
        return self._out(frame, (q, k + 1))


@pytest.fixture
def fake(tr, monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(tr, "_get_runtime", lambda: (runtime, 0.0))
    monkeypatch.setattr(tr, "_sessions", type(tr._sessions)())
    return runtime


def frame(value=0, h=H, w=W):
    return np.full((h, w, 3), value, dtype=np.uint8)


def clip(n, h=H, w=W):
    return [frame(k, h, w) for k in range(n)]


def pixels(out, j=0):
    return np.array(out["tracks"][j]["pixels"])


# --- registration ---------------------------------------------------------


def test_tools_registered_with_their_schemas(tool_registry):
    for name in ("track.points", "track.start", "track.update", "track.stop"):
        assert name in tool_registry
    points = tool_registry.get("track.points").schema
    assert set(points.inputs) == {
        "frames",
        "query_points",
        "query_frame",
        "support_points",
        "min_confidence",
    }
    assert points.inputs["support_points"].default == 0
    assert {"found", "tracks", "num_frames", "query_frame", "timing"} <= set(points.outputs)
    assert set(tool_registry.get("track.update").schema.inputs) == {
        "session_id",
        "frame",
        "min_confidence",
    }


def test_import_does_not_pull_torch(tr):
    assert tr._runtime is None
    assert not {"torch", "tapnet"} & set(vars(tr))


def test_pins_agree_with_skill_md_and_pyproject(tr):
    skill = (BUNDLE / "SKILL.md").read_text()
    spec = tr.CHECKPOINTS[tr.DEFAULT_CHECKPOINT]
    assert f"uri: {spec['url']}" in skill and f"sha256: {spec['sha256']}" in skill
    assert tr.TAPNET_COMMIT in skill
    assert f'rev = "{tr.TAPNET_COMMIT}"' in (BUNDLE / "pyproject.toml").read_text()
    for s in tr.CHECKPOINTS.values():
        assert re.fullmatch(r"[0-9a-f]{64}", s["sha256"])


# --- pixel <-> model mapping ---------------------------------------------------


def test_pixel_centres_map_to_model_cell_centres(tr):
    # OpenCV pixel centre i is model coordinate (i + 0.5) * 256 / size.
    yx = tr._to_model(np.array([[0.0, 0.0], [W - 1.0, H - 1.0]]), W, H)
    assert yx[0] == pytest.approx([0.5 * 256 / H, 0.5 * 256 / W])
    assert yx[1] == pytest.approx([256 - 0.5 * 256 / H, 256 - 0.5 * 256 / W])
    px = np.array([[12.3, 400.7], [320.0, 240.0]])
    assert tr._from_model(tr._to_model(px, W, H), W, H) == pytest.approx(px)


def test_support_points_follow_upstreams_grid(tr):
    q = np.array([[320.0, 240.0]])
    pts = tr._support_points(q, 64, W, H)
    assert pts.shape == (64, 2)
    rx, ry = 32 / 512 * W, 32 / 512 * H  # 40 x 30 px
    # tracker.py's grid_support_points: cols = round(sqrt(n w / h)) = 9, rows = ceil(64 / 9) = 8
    xs = np.unique(pts[:, 0])
    assert len(xs) == 9 and xs[0] == pytest.approx(320 - rx + rx / 9)
    assert np.unique(pts[:, 1])[0] == pytest.approx(240 - ry + ry / 8)
    assert np.abs(pts - q).max(axis=0) == pytest.approx([rx * 8 / 9, ry * 7 / 8])


def test_support_points_are_clamped_and_optional(tr):
    pts = tr._support_points(np.array([[0.0, H - 1.0]]), 16, W, H)
    assert pts[:, 0].min() == 0.0 and pts[:, 1].max() == H - 1.0
    assert tr._support_points(np.array([[5.0, 5.0]]), 0, W, H).shape == (0, 2)


# --- track.points -----------------------------------------------------------------


def test_static_points_come_back_where_they_were_queried(tr, fake):
    out = tr.points(clip(5), [[100.0, 200.0], [600.5, 10.25]], support_points=8)
    assert out["found"] and out["num_frames"] == 5 and out["query_frame"] == 0
    assert out["backend"] == "tapnext++" and out["model"].startswith("tapnextpp_512.ckpt@sha256:")
    assert len(out["tracks"]) == 2  # support points dropped
    assert pixels(out, 1) == pytest.approx(np.tile([600.5, 10.25], (5, 1)))
    assert out["tracks"][0]["query_pixel"] == [100.0, 200.0]
    assert out["tracks"][0]["visible"] == [True] * 5
    assert len(fake.queries[0]) == 2 * 9 and fake.seen == [0, 1, 2, 3, 4]


def test_motion_is_scaled_back_to_frame_pixels(tr, fake):
    fake.delta = np.array([1.0, 2.0])  # model px per frame: y 1 -> 480/256, x 2 -> 5 px
    out = tr.points(clip(3), [[100.0, 100.0]], support_points=0)
    expected = np.array([[100, 100], [105, 101.875], [110, 103.75]])
    assert pixels(out) == pytest.approx(expected, abs=0.005)  # pixels are rounded to 0.01


def test_a_query_on_a_later_frame_is_tracked_both_ways(tr, fake):
    fake.delta = np.array([0.0, 1.0])  # 2.5 px per frame stepped away from the query frame
    out = tr.points(clip(5), [[200.0, 50.0]], query_frame=2, support_points=0)
    assert out["query_frame"] == 2
    assert fake.seen == [2, 3, 4, 2, 1, 0]  # forwards, then the reversed head
    assert pixels(out)[:, 0] == pytest.approx([205.0, 202.5, 200.0, 202.5, 205.0])
    assert out["timing"]["per_frame_ms"] >= 0


def test_a_point_never_seen_again_is_not_found(tr, fake):
    fake.vis = {1: -3.0, 2: -3.0}
    out = tr.points(clip(3), [[10.0, 10.0]], support_points=0)
    assert out["tracks"][0]["visible"] == [True, False, False]
    assert not out["found"]
    fake.vis = {0: -3.0}
    assert tr.points(clip(1), [[10.0, 10.0]], support_points=0)["found"] is False


def test_confidence_is_visibility_times_certainty(tr, fake):
    fake.cert = 0.5
    fake.vis = {1: 0.0}
    out = tr.points(clip(2), [[10.0, 10.0]], support_points=0)
    p5 = 1 / (1 + np.exp(-5.0))
    assert out["tracks"][0]["confidence"] == pytest.approx([round(0.5 * p5, 3), 0.25])
    assert out["tracks"][0]["visible"] == [True, False]  # logit 0 is not > 0
    gated = tr.points(clip(2), [[10.0, 10.0]], support_points=0, min_confidence=0.3)
    assert gated["tracks"][0]["visible"] == [True, False]
    assert not tr.points(clip(2), [[10.0, 10.0]], support_points=0, min_confidence=0.6)["found"]


def test_a_stacked_array_clip_and_a_single_query_pair(tr, fake):
    out = tr.points(np.stack(clip(2)), [50.0, 60.0], support_points=0)
    assert out["num_frames"] == 2 and pixels(out)[0] == pytest.approx([50.0, 60.0])


def test_points_rejects_bad_inputs(tr, fake):
    with pytest.raises(ValueError, match="uint8"):
        tr.points([np.zeros((H, W, 3), dtype=np.float32)], [[1.0, 1.0]])
    with pytest.raises(ValueError, match="same size"):
        tr.points([frame(), frame(h=240)], [[1.0, 1.0]])
    with pytest.raises(ValueError, match="at least one"):
        tr.points([], [[1.0, 1.0]])
    with pytest.raises(ValueError, match="outside"):
        tr.points(clip(2), [[W - 0.4, 1.0]])
    with pytest.raises(ValueError, match="query_points"):
        tr.points(clip(2), [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="query_frame"):
        tr.points(clip(2), [[1.0, 1.0]], query_frame=2)
    with pytest.raises(ValueError, match="support_points"):
        tr.points(clip(2), [[1.0, 1.0]], support_points=257)
    assert fake.seen == []


def test_model_failures_raise_tool_error(tr, monkeypatch):
    monkeypatch.setattr(
        tr, "_get_runtime", lambda: (FakeRuntime(fail=RuntimeError("CUDA out of memory")), 0.0)
    )
    with pytest.raises(ToolError, match="out of memory"):
        tr.points(clip(2), [[1.0, 1.0]])

    def broken():
        raise OSError("no checkpoint")

    monkeypatch.setattr(tr, "_get_runtime", broken)
    with pytest.raises(ToolError, match="failed to load"):
        tr.start(frame(), [[1.0, 1.0]])


# --- online sessions ------------------------------------------------------------


def test_a_session_matches_the_clip_frame_by_frame(tr, fake):
    fake.delta = np.array([0.0, 1.0])
    first = tr.start(frame(0), [[100.0, 100.0]], support_points=4)
    assert first["step"] == 0 and first["pixels"] == [[100.0, 100.0]] and first["evicted"] == ""
    sid = first["session_id"]
    steps = [tr.update(sid, frame(k)) for k in (1, 2)]
    assert [s["step"] for s in steps] == [1, 2]
    assert [s["pixels"][0][0] for s in steps] == pytest.approx([102.5, 105.0])
    assert len(fake.queries[0]) == 5 and steps[-1]["visible"] == [True]
    assert tr.stop(sid) == {"stopped": True}
    assert tr.stop(sid) == {"stopped": False}
    with pytest.raises(ToolError, match="unknown session"):
        tr.update(sid, frame(3))


def test_a_session_keeps_its_frame_size(tr, fake):
    sid = tr.start(frame(), [[1.0, 1.0]])["session_id"]
    with pytest.raises(ValueError, match="640x480"):
        tr.update(sid, frame(h=240, w=320))


def test_the_least_recently_used_session_is_evicted(tr, fake, monkeypatch):
    monkeypatch.setattr(tr, "MAX_SESSIONS", 2)
    a = tr.start(frame(), [[1.0, 1.0]])["session_id"]
    b = tr.start(frame(), [[1.0, 1.0]])["session_id"]
    tr.update(a, frame(1))  # b is now the least recently used
    c = tr.start(frame(), [[1.0, 1.0]])
    assert c["evicted"] == b and set(tr._sessions) == {a, c["session_id"]}


# --- checkpoint --------------------------------------------------------------------


def test_checkpoint_choice_and_local_override(tr, monkeypatch, tmp_path):
    monkeypatch.setenv("GAP_TRACK_CHECKPOINT", "256")
    assert tr._model_id().startswith("tapnextpp_ckpt.pt@sha256:cb96a43444cc")
    monkeypatch.setenv("GAP_TRACK_CHECKPOINT", "1024")
    with pytest.raises(ValueError, match="GAP_TRACK_CHECKPOINT"):
        tr._model_id()
    local = tmp_path / "mine.pt"
    monkeypatch.setenv("GAP_TRACK_CHECKPOINT_PATH", str(local))
    assert tr._model_id() == str(local) and tr.weights_cached() is False
    local.write_bytes(b"x")
    assert tr.weights_cached() is True and tr._checkpoint_file() == str(local)


def _fake_bucket(tr, monkeypatch, tmp_path, payload, digest):
    spec = dict(tr.CHECKPOINTS["512"], bytes=len(payload), sha256=digest)
    monkeypatch.setitem(tr.CHECKPOINTS, "512", spec)
    monkeypatch.delenv("GAP_TRACK_CHECKPOINT", raising=False)
    monkeypatch.delenv("GAP_TRACK_CHECKPOINT_PATH", raising=False)
    monkeypatch.setenv("GAP_TRACK_CACHE", str(tmp_path))
    fetched = []

    def urlretrieve(url, dest):
        fetched.append(url)
        Path(dest).write_bytes(payload)

    monkeypatch.setattr(tr.urllib.request, "urlretrieve", urlretrieve)
    return fetched


def test_the_download_is_checked_and_cached(tr, monkeypatch, tmp_path):
    payload = b"tapnext++ weights"
    fetched = _fake_bucket(tr, monkeypatch, tmp_path, payload, hashlib.sha256(payload).hexdigest())
    assert tr.weights_cached() is False
    tr.prefetch()
    assert tr.weights_cached() is True
    assert tr._checkpoint_file() == str(tmp_path / "tapnextpp_512.ckpt")
    assert fetched == [tr.CHECKPOINTS["512"]["url"]]  # the second call used the cache


def test_a_download_with_the_wrong_digest_is_discarded(tr, monkeypatch, tmp_path):
    _fake_bucket(tr, monkeypatch, tmp_path, b"tampered", "0" * 64)
    with pytest.raises(RuntimeError, match="expected 0000"):
        tr._checkpoint_file()
    assert list(tmp_path.iterdir()) == []
