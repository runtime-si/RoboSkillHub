"""judging-attempt-success-from-frames-robojev and tools/robojev with the model faked: the prompt, the frame
picking, the calibrated read-out, the server path, and the skill's frame sources, cameras and routes."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import PerceptionFailed, ToolError

SKILL = "judging-attempt-success-from-frames-robojev"


@pytest.fixture(scope="module")
def rj(skills_registry):
    return skills_registry.get("robojev").tools_module


@pytest.fixture(scope="module")
def judge(skills_registry):
    bundle = skills_registry.get(SKILL)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["judge_attempt"].module


class FakeRuntime:
    name = "fake-robojev"

    def __init__(self, lf=-3.0, lt=-1.0):
        self.lf, self.lt = lf, lt
        self.calls = []

    def logits(self, text, images):
        self.calls.append((text, images))
        return self.lf, self.lt


@pytest.fixture
def runtime(rj, monkeypatch):
    rt = FakeRuntime()
    monkeypatch.setattr(rj, "_runtime", rt)
    return rt


def _img(v, h=48, w=64):
    return np.full((h, w, 3), v, np.uint8)


# -- tools/robojev ----------------------------------------------------------------------------


def test_pick_frames_spans_start_to_end_and_pads_short_clips_with_the_end(rj):
    assert rj.pick_frames(12) == [0, 1, 2, 4, 5, 6, 7, 9, 10, 11]
    assert rj.pick_frames(10) == list(range(10))
    assert rj.pick_frames(2) == [0] + [1] * 9
    assert rj.pick_frames(4) == [0, 1, 2] + [3] * 7
    assert rj.pick_frames(1) == [0] * 10
    with pytest.raises(ValueError):
        rj.pick_frames(0)


def test_the_calibrated_read_out_is_the_released_one(rj):
    # decision.json: bias +1.196e-4 / -1.198e-4, temperature 0.9.
    lf, lt = -2.0, -0.5
    want = 1.0 / (
        1.0 + math.exp(((lf + 0.00011960109259234741) - (lt - 0.00011981795250903815)) / 0.9)
    )
    assert rj.p_done(lf, lt) == pytest.approx(want)
    assert rj.p_done(0.0, 0.0) == pytest.approx(0.5, abs=1e-3)
    assert rj.p_done(-200.0, 0.0) == pytest.approx(1.0)


def test_the_prompt_is_robojevs_dense_one(rj, runtime):
    frames = [_img(i * 20) for i in range(12)]
    out = rj.judge(
        task=" Put the cube in the box. ",
        frames=frames,
        views={"left_wrist": _img(7)},
        camera="top",
    )
    text, images = runtime.calls[0]
    assert text.startswith(
        "[kind] noul\n[state] Task given to the robot arm: Put the cube in the box.\n"
    )
    assert (
        "Images 1-10: overhead camera, evenly spaced from the start to the end of the attempt.\n"
        in text
    )
    assert "Image 11: wrist camera at the end.\n" in text
    assert text.count("<|vision_start|><|image_pad|><|vision_end|>") == 11
    assert text.endswith("\n[options]\nfalse\ntrue\n[decision]:")
    assert rj.DENSE_QUESTION in text
    assert len(images) == 11 and min(images[0].size) == 288
    assert out["frames_used"] == [0, 1, 2, 4, 5, 6, 7, 9, 10, 11] and out["views_used"] == [
        "left_wrist"
    ]
    assert out["p_done"] == pytest.approx(rj.p_done(-3.0, -1.0)) and out["model"] == "fake-robojev"


def test_camera_names_follow_the_client(rj):
    assert rj.camera_name("top") == "overhead camera"
    assert rj.camera_name("observation.images.front") == "front camera"
    assert rj.camera_name("left_wrist") == "wrist camera"
    assert rj.camera_name("side") == "side camera"


def test_bad_inputs_raise(rj, runtime):
    with pytest.raises(ToolError, match="task"):
        rj.judge(task=" ", frames=[_img(0)])
    with pytest.raises(ToolError, match="frames"):
        rj.judge(task="t", frames=[])
    with pytest.raises(ToolError, match="RGB uint8"):
        rj.judge(task="t", frames=[np.zeros((4, 4), np.uint8)])


def test_a_model_failure_is_perception_failed(rj, monkeypatch):
    class Broken:
        name = "x"

        def logits(self, text, images):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(rj, "_runtime", Broken())
    with pytest.raises(PerceptionFailed, match="out of memory"):
        rj.judge(task="t", frames=[_img(0), _img(1)])


def test_the_server_path_sends_upstreams_request(rj, monkeypatch):
    import io
    import json
    import urllib.request

    sent = {}

    def urlopen(req, timeout):
        sent.update(url=req.full_url, body=json.loads(req.data))
        top = [
            {"token": "token_id:3721", "logprob": -2.0},
            {"token": "token_id:1802", "logprob": -0.2},
        ]
        reply = {"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]}
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(rj, "_runtime", rj._Server("http://gpu:8000/", "robojev"))
    out = rj.judge(task="t", frames=[_img(0), _img(1)])
    body = sent["body"]
    assert sent["url"] == "http://gpu:8000/v1/chat/completions"
    assert body["allowed_token_ids"] == [3721, 1802] and body["max_tokens"] == 1
    content = body["messages"][0]["content"]
    assert content[0]["text"].startswith("[kind] noul") and len(content) == 12
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert out["p_done"] == pytest.approx(rj.p_done(-2.0, -0.2))


def test_weights_probe_never_downloads(rj, monkeypatch):
    monkeypatch.setenv("GAP_ROBOJEV_MODEL", "no/such-model")
    assert rj.weights_cached() is False


# -- the skill --------------------------------------------------------------------------------


def _obs(**cams):
    return {"cameras": [{"name": n, "rgb": v} for n, v in cams.items()], "arms": []}


def _judge_stub(p, calls):
    def judge(task, frames, views=None, camera="main"):
        calls.append(
            {"task": task, "n": len(frames), "views": sorted(views or {}), "camera": camera}
        )
        return {
            "p_done": p,
            "logit_false": 0.0,
            "logit_true": 0.0,
            "frames_used": [],
            "views_used": [],
        }

    return judge


@pytest.mark.parametrize(("p", "route"), [(0.93, "success"), (0.05, "failure"), (0.5, "uncertain")])
def test_routes_follow_the_probability(judge, p, route):
    calls = []
    ctx = FakeContext(
        {
            "robot.get_observation": _obs(top=_img(1), left_wrist=_img(2)),
            "robojev.judge": _judge_stub(p, calls),
        }
    )
    out = judge.run(ctx, task="Put the cube in the box.", frames=[_img(i) for i in range(12)])
    assert out["route"] == route and out["p_done"] == p
    assert calls == [
        {"task": "Put the cube in the box.", "n": 12, "views": ["left_wrist"], "camera": "top"}
    ]
    assert out["main_camera"] == "top" and out["views"] == ["left_wrist"] and out["n_frames"] == 12


def test_before_and_now_make_a_two_frame_clip(judge):
    calls = []
    ctx = FakeContext(
        {
            "robot.get_observation": _obs(top=_img(9), left_wrist=_img(2)),
            "robojev.judge": _judge_stub(0.9, calls),
        }
    )
    out = judge.run(ctx, task="t", before=_obs(top=_img(0), left_wrist=_img(1)))
    assert out["route"] == "success" and calls[0]["n"] == 2 and out["n_frames"] == 2


def test_camera_frames_as_dicts_and_float_images_are_accepted(judge):
    calls = []
    ctx = FakeContext(
        {"robot.get_observation": _obs(top=_img(9)), "robojev.judge": _judge_stub(0.1, calls)}
    )
    frames = [{"name": "top", "rgb": np.full((8, 8, 3), 0.5, np.float32)}, {"rgb": _img(3)}]
    out = judge.run(ctx, task="t", frames=frames, use_views=False)
    assert out["route"] == "failure" and calls[0]["views"] == [] and calls[0]["n"] == 2


def test_views_and_camera_can_be_named(judge):
    calls = []
    ctx = FakeContext(
        {
            "robot.get_observation": _obs(top=_img(1), front=_img(2), left_wrist=_img(3)),
            "robojev.judge": _judge_stub(0.8, calls),
        }
    )
    judge.run(ctx, task="t", frames=[_img(0), _img(1)], camera="front", views=["left_wrist"])
    assert calls[0]["camera"] == "front" and calls[0]["views"] == ["left_wrist"]


def test_only_a_wrist_camera_is_used_as_main(judge):
    calls = []
    ctx = FakeContext(
        {
            "robot.get_observation": _obs(left_wrist=_img(1)),
            "robojev.judge": _judge_stub(0.8, calls),
        }
    )
    out = judge.run(ctx, task="t", before=_obs(left_wrist=_img(0)))
    assert out["main_camera"] == "left_wrist" and calls[0]["views"] == []


def test_no_frames(judge):
    ctx = FakeContext({"robot.get_observation": _obs(top=_img(1))})
    out = judge.run(ctx, task="t")
    assert out["route"] == "no_frames" and "1 frame(s)" in out["detail"]
    out = judge.run(FakeContext({"robot.get_observation": {"cameras": []}}), task="t")
    assert out["route"] == "no_frames" and "no camera" in out["detail"]


def test_bad_arguments_raise(judge):
    with pytest.raises(ValueError, match="task"):
        judge.run(FakeContext({}), task="")
    with pytest.raises(ValueError, match="failure_at"):
        judge.run(FakeContext({}), task="t", success_at=0.2, failure_at=0.6)
