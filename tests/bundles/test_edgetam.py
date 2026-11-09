"""edgetam bundle: prompts, sessions (open, step, evict, close), presence from the occlusion head,
the clip path and the session pruning, with the EdgeTAM / SAM 2.1 runtime faked."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from gap_core.errors import PerceptionFailed, ToolError

H, W = 120, 160


@pytest.fixture(scope="module")
def et(skills_registry):
    return skills_registry.get("edgetam").tools_module


def frame(v: int = 0) -> np.ndarray:
    return np.full((H, W, 3), v, dtype=np.uint8)


def box_mask(x1, y1, x2, y2) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[y1:y2, x1:x2] = True
    return m


class FakeRuntime:
    """The object is a box that moves 2 px right per frame; frames whose pixel value is 255 hide it
    (the occlusion head goes negative); a prompt over an empty region gives an empty mask."""

    def __init__(self):
        self.started: list[dict] = []
        self.steps: list[int] = []

    def _at(self, prompt, k):
        if prompt.get("box") is not None:
            x1, y1, x2, y2 = (int(v) for v in prompt["box"])
        elif prompt.get("points") is not None:
            x, y = prompt["points"][0]
            x1, y1, x2, y2 = int(x) - 5, int(y) - 5, int(x) + 5, int(y) + 5
        else:
            ys, xs = np.nonzero(prompt["mask"])
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        if x1 >= W - 4:
            return np.zeros((H, W), bool)
        return box_mask(x1 + 2 * k, y1, min(x2 + 2 * k, W), y2)

    def start(self, image, prompt):
        self.started.append(prompt)
        state = {"prompt": prompt}
        m = self._at(prompt, 0)
        return state, m, 3.0 if m.any() else -3.0

    def step(self, state, image, k):
        self.steps.append(k)
        if image[0, 0, 0] == 255:
            return np.zeros((H, W), bool), -4.0
        return self._at(state["prompt"], k), 2.0

    def clip(self, frames, pf, prompt):
        out = []
        for i, f in enumerate(frames):
            if f[0, 0, 0] == 255:
                out.append((np.zeros((H, W), bool), -4.0))
            else:
                out.append((self._at(prompt, i - pf), 2.0))
        return out


@pytest.fixture
def fake(et, monkeypatch):
    rt = FakeRuntime()
    monkeypatch.setattr(et, "_get_runtime", lambda: rt)
    monkeypatch.setattr(et, "_sessions", type(et._sessions)())
    return rt


def xyxy(b):
    return [b["x1"], b["y1"], b["x2"], b["y2"]]


# --- registration, pins, imports ----------------------------------------------------


def test_tools_registered_with_their_schemas(tool_registry):
    init = tool_registry.get("edgetam.tracker_init").schema
    assert {"image", "box", "pixel_x", "pixel_y", "points", "point_labels", "mask"} <= set(
        init.inputs
    )
    assert set(init.outputs) == {
        "tracker_id",
        "initial_mask",
        "initial_box",
        "score",
        "object_present",
    }
    upd = tool_registry.get("edgetam.tracker_update").schema
    assert set(upd.inputs) == {"tracker_id", "image"}
    assert {"mask", "box", "confidence", "object_present"} <= set(upd.outputs)
    assert set(tool_registry.get("edgetam.tracker_close").schema.inputs) == {"tracker_id"}
    trk = tool_registry.get("edgetam.track").schema
    assert {"frames", "box", "prompt_frame", "return_masks"} <= set(trk.inputs)
    for n in ("tracker_init", "tracker_update", "tracker_close", "track"):
        assert tool_registry.get(f"edgetam.{n}").tags == ("perception",)


def test_import_does_not_pull_torch_and_models_are_pinned(et):
    assert "torch" not in vars(et) and "transformers" not in vars(et)
    assert et.DEFAULT_MODEL in et.MODELS
    for rev in et.MODELS.values():
        assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def test_model_choice_is_checked(et, monkeypatch, tmp_path):
    monkeypatch.setenv("GAP_EDGETAM_MODEL", "facebook/sam2.1-hiera-small")
    assert et._model_source()[1] == et.MODELS["facebook/sam2.1-hiera-small"]
    monkeypatch.setenv("GAP_EDGETAM_MODEL", str(tmp_path))
    assert et._model_source() == (str(tmp_path), None)
    assert et.weights_cached() is False
    monkeypatch.setenv("GAP_EDGETAM_MODEL", "someone/sam2-finetune")
    with pytest.raises(ValueError, match="must be one of"):
        et._model_source()


# --- prompts --------------------------------------------------------------------------


def test_prompt_forms(et):
    size = (H, W)
    assert et._prompt("t", size, [10, 20, 30, 40], None, None, None) == {"box": [10, 20, 30, 40]}
    assert et._prompt("t", size, {"x1": -5, "y1": 0, "x2": 500, "y2": 10}, None, None, None) == {
        "box": [0.0, 0.0, float(W), 10.0]
    }
    p = et._prompt("t", size, None, [[5, 6], [7, 8]], [1, 0], None)
    assert p == {"points": [[5.0, 6.0], [7.0, 8.0]], "labels": [1, 0]}
    p = et._prompt("t", size, [10, 20, 30, 40], [[15, 25]], None, None)
    assert p["box"] == [10, 20, 30, 40] and p["labels"] == [1]
    m = et._prompt("t", size, None, None, None, box_mask(1, 2, 3, 4).astype(np.uint8) * 255)
    assert m["mask"].dtype == np.float32 and m["mask"].sum() == 4


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((None, None, None, None), "give one prompt"),
        (([0, 0, 0.5, 0.5], None, None, None), "empty"),
        (([1, 2, 3], None, None, None), "x1, y1, x2, y2"),
        ((None, [[1, 1]], [0], None), "at least one object point"),
        ((None, [[1, 1]], [1, 1], None), "one 1"),
        ((None, [[1, 1]], [2], None), "one 1"),
        ((None, None, None, np.zeros((H, W))), "empty"),
        ((None, None, None, np.ones((H + 1, W))), "differs"),
        (([0, 0, 9, 9], None, None, np.ones((H, W))), "mask alone"),
    ],
)
def test_bad_prompts_are_refused(et, args, match):
    with pytest.raises(ToolError, match=match):
        et._prompt("t", (H, W), *args)


# --- live sessions ----------------------------------------------------------------------


def test_init_update_close_follow_the_object(et, fake):
    r = et.tracker_init(frame(), box=[10, 20, 30, 40], object_name="cube")
    assert r["object_present"] and r["tracker_id"]
    assert xyxy(r["initial_box"]) == [10, 20, 30, 40]
    assert r["initial_mask"].dtype == np.uint8 and set(np.unique(r["initial_mask"])) == {0, 255}
    assert r["score"] == pytest.approx(1 / (1 + np.exp(-3.0)))
    u = et.tracker_update(r["tracker_id"], frame())
    assert u["object_present"] and xyxy(u["box"]) == [12, 20, 32, 40]
    assert u["frame_index"] == 1 and u["area_px"] == 400
    u = et.tracker_update(r["tracker_id"], frame())
    assert xyxy(u["box"]) == [14, 20, 34, 40] and fake.steps == [1, 2]
    assert et.tracker_close(r["tracker_id"]) == {"closed": True}
    assert et.tracker_close(r["tracker_id"]) == {"closed": False}
    with pytest.raises(ToolError, match="unknown tracker_id"):
        et.tracker_update(r["tracker_id"], frame())


def test_occluded_frames_report_absent_and_the_session_continues(et, fake):
    tid = et.tracker_init(frame(), box=[10, 20, 30, 40])["tracker_id"]
    u = et.tracker_update(tid, frame(255))
    assert not u["object_present"] and u["mask"] is None and u["box"] is None
    assert u["confidence"] < 0.5 and u["area_px"] == 0
    u = et.tracker_update(tid, frame())
    assert u["object_present"] and u["frame_index"] == 2


def test_point_and_mask_prompts_reach_the_runtime(et, fake):
    et.tracker_init(frame(), pixel_x=50, pixel_y=60)
    assert fake.started[-1] == {"points": [[50.0, 60.0]], "labels": [1]}
    et.tracker_init(frame(), mask=box_mask(5, 5, 9, 9))
    assert set(fake.started[-1]) == {"mask"}
    with pytest.raises(ToolError, match="together"):
        et.tracker_init(frame(), pixel_x=50)
    with pytest.raises(ToolError, match="together"):
        et.tracker_init(frame(), pixel_x=1, pixel_y=2, points=[[1, 2]])


def test_an_empty_first_mask_opens_no_session(et, fake):
    r = et.tracker_init(frame(), box=[W - 3, 0, W, 10])
    assert r == {
        "tracker_id": "",
        "initial_mask": None,
        "initial_box": None,
        "score": pytest.approx(1 / (1 + np.exp(3.0))),
        "object_present": False,
    }
    assert len(et._sessions) == 0


def test_frames_must_match_the_session(et, fake):
    tid = et.tracker_init(frame(), box=[10, 20, 30, 40])["tracker_id"]
    with pytest.raises(ToolError, match="differs"):
        et.tracker_update(tid, np.zeros((H, W + 1, 3), np.uint8))
    with pytest.raises(ToolError, match="RGB uint8"):
        et.tracker_update(tid, np.zeros((H, W), np.uint8))


def test_least_recently_used_session_is_evicted(et, fake, monkeypatch):
    monkeypatch.setattr(et, "MAX_SESSIONS", 2)
    a = et.tracker_init(frame(), box=[10, 20, 30, 40])["tracker_id"]
    b = et.tracker_init(frame(), box=[10, 20, 30, 40])["tracker_id"]
    et.tracker_update(a, frame())  # a is now more recent than b
    c = et.tracker_init(frame(), box=[10, 20, 30, 40])["tracker_id"]
    assert set(et._sessions) == {a, c}
    with pytest.raises(ToolError, match="unknown"):
        et.tracker_update(b, frame())


def test_stale_sessions_are_reaped(et, fake, monkeypatch):
    tid = et.tracker_init(frame(), box=[10, 20, 30, 40])["tracker_id"]
    et._sessions[tid].touched -= et.SESSION_TTL_S + 1
    et.tracker_init(frame(), box=[10, 20, 30, 40])
    assert tid not in et._sessions


def test_model_errors_become_perception_failed(et, fake, monkeypatch):
    tid = et.tracker_init(frame(), box=[10, 20, 30, 40])["tracker_id"]

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(fake, "step", boom)
    with pytest.raises(PerceptionFailed, match="out of memory"):
        et.tracker_update(tid, frame())
    monkeypatch.setattr(fake, "start", boom)
    with pytest.raises(PerceptionFailed, match="init failed"):
        et.tracker_init(frame(), box=[10, 20, 30, 40])


# --- recorded clips ---------------------------------------------------------------------


def test_track_a_clip_from_a_middle_frame(et, fake):
    clip = [frame(), frame(), frame(255), frame(), frame()]
    r = et.track(clip, box=[40, 20, 60, 40], prompt_frame=1, return_masks=True)
    assert r["present"] == [True, True, False, True, True]
    assert xyxy(r["boxes"][0]) == [38, 20, 58, 40] and r["boxes"][2] is None
    assert xyxy(r["boxes"][4]) == [46, 20, 66, 40]
    assert r["area_px"][2] == 0 and r["masks"][2].max() == 0 and r["masks"][1].max() == 255
    assert r["prompt_frame"] == 1 and r["seconds"] >= 0
    assert et.track(clip, box=[40, 20, 60, 40])["masks"] is None


def test_track_refuses_bad_clips(et, fake):
    with pytest.raises(ToolError, match="empty"):
        et.track([], box=[0, 0, 5, 5])
    with pytest.raises(ToolError, match="same size"):
        et.track([frame(), np.zeros((H + 1, W, 3), np.uint8)], box=[0, 0, 5, 5])
    with pytest.raises(ToolError, match="outside"):
        et.track([frame()], box=[0, 0, 5, 5], prompt_frame=1)


# --- pruning ----------------------------------------------------------------------------


def test_prune_keeps_the_memory_window_and_conditioning_frames(et):
    s = SimpleNamespace(
        processed_frames={k: object() for k in range(30)},
        output_dict_per_obj={
            0: {
                "cond_frame_outputs": {0: "prompt"},
                "non_cond_frame_outputs": {k: object() for k in range(1, 30)},
            }
        },
        frames_tracked_per_obj={0: {k: {} for k in range(1, 30)}},
    )
    et._prune(s, 29, 17)
    assert list(s.processed_frames) == [29]
    assert s.output_dict_per_obj[0]["cond_frame_outputs"] == {0: "prompt"}
    assert sorted(s.output_dict_per_obj[0]["non_cond_frame_outputs"]) == list(range(13, 30))
    assert sorted(s.frames_tracked_per_obj[0]) == list(range(13, 30))
    et._prune(SimpleNamespace(), 5, 17)  # tolerates a session without those attributes
