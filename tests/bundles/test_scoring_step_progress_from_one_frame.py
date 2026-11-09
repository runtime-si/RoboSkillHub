"""scoring-step-progress-from-one-frame: FakeContext tests of frame choice, tool choice and routing."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "scoring-step-progress-from-one-frame"
TASK = "put the cube in the black box"


def _cam(name, value):
    return {"name": name, "rgb": np.full((4, 6, 3), value, dtype=np.uint8)}


OBS = {"cameras": [_cam("left_wrist", 1), _cam("top", 2)]}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["score_progress"].module


class Scorer:
    """A densereward tool double: records calls, answers ``reply`` (a dict or an exception)."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return {"model": "densereward/x@abc", "raw": "r", "reason": "", **self.reply}


def _ctx(one=None, three=None, obs=OBS):
    one = Scorer(one if one is not None else {"progress": 0.5, "found": True})
    three = Scorer(three if three is not None else {"progress": 0.5, "found": True})
    tools = {
        "densereward.progress": one,
        "densereward.progress_with_history": three,
        "robot.get_observation": obs,
    }
    return FakeContext(tools), one, three


@pytest.mark.parametrize(
    ("progress", "route"),
    [(0.94, "done"), (0.85, "done"), (0.849, "in_progress"), (0.0, "in_progress")],
)
def test_routes_on_the_done_threshold(script, progress, route):
    ctx, one, _ = _ctx({"progress": progress, "found": True})
    out = script.run(ctx, task=TASK)
    assert out["route"] == route
    assert out["progress"] == pytest.approx(progress)
    assert out["camera"] == "top"
    assert out["model"] == "densereward/x@abc"
    assert out["scores"] == [pytest.approx(progress)] * 3
    assert len(one.calls) == 3
    assert all(c["task"] == TASK and int(c["image"][0, 0, 0]) == 2 for c in one.calls)


def test_external_camera_is_preferred_and_a_named_one_is_used(script):
    ctx, one, _ = _ctx()
    script.run(ctx, task=TASK, camera="left_wrist")
    assert int(one.calls[0]["image"][0, 0, 0]) == 1
    wrist_only = {"cameras": {"left_wrist": {"rgb": np.zeros((2, 2, 3), np.uint8)}}}
    ctx, one, _ = _ctx(obs=wrist_only)
    assert script.run(ctx, task=TASK, samples=1)["camera"] == "left_wrist"


def test_given_observation_is_used_instead_of_a_fresh_one(script):
    ctx, one, _ = _ctx(obs=RuntimeError("must not be called"))
    script.run(ctx, task=TASK, observation={"cameras": [_cam("top", 9)]}, samples=1)
    assert int(one.calls[0]["image"][0, 0, 0]) == 9


class Seq(Scorer):
    """Answers each call with the next progress in ``values``."""

    def __init__(self, values):
        super().__init__({})
        self.values = list(values)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "model": "m",
            "raw": "r",
            "reason": "",
            "found": True,
            "progress": self.values.pop(0),
        }


def _seq_ctx(values, observations):
    one = Seq(values)
    obs = iter(observations)
    tools = {"densereward.progress": one, "robot.get_observation": lambda: next(obs)}
    return FakeContext(tools), one


def test_samples_fresh_frames_and_routes_on_their_median(script):
    frames = [{"cameras": [_cam("top", v)]} for v in (3, 4, 5)]
    ctx, one = _seq_ctx([0.86, 0.90, 0.70], frames)
    out = script.run(ctx, task=TASK)
    assert [int(c["image"][0, 0, 0]) for c in one.calls] == [3, 4, 5]
    assert out["route"] == "done" and out["progress"] == pytest.approx(0.86)
    assert out["scores"] == [0.86, 0.9, 0.7]
    assert out["raw"] == "r | r | r"


def test_samples_that_disagree_route_low_confidence(script):
    frames = [{"cameras": [_cam("top", v)]} for v in (3, 4, 5)]
    ctx, _ = _seq_ctx([0.33, 0.94, 0.37], frames)
    out = script.run(ctx, task=TASK)
    assert out["route"] == "low_confidence" and "disagree" in out["reason"]
    assert out["progress"] == pytest.approx(0.37)
    ctx, _ = _seq_ctx([0.33, 0.94, 0.37], frames)
    assert script.run(ctx, task=TASK, max_spread=0.7)["route"] == "in_progress"


def test_a_fresh_frame_without_the_camera_is_skipped(script):
    frames = [{"cameras": [_cam("top", 3)]}, {"cameras": []}, {"cameras": [_cam("top", 5)]}]
    ctx, one = _seq_ctx([0.5, 0.6], frames)
    out = script.run(ctx, task=TASK)
    assert len(one.calls) == 2 and out["scores"] == [0.5, 0.6]


def test_float_frames_are_converted_to_uint8(script):
    obs = {"cameras": [{"name": "top", "rgb": np.full((2, 2, 4), 0.5, dtype=np.float32)}]}
    ctx, one, _ = _ctx(obs=obs)
    script.run(ctx, task=TASK)
    img = one.calls[0]["image"]
    assert img.dtype == np.uint8 and img.shape == (2, 2, 3) and int(img[0, 0, 0]) == 127


def test_history_uses_the_three_frame_model_on_the_same_camera(script):
    history = [{"cameras": [_cam("top", v)]} for v in (5, 6, 7)]
    ctx, one, three = _ctx(three={"progress": 0.9, "found": True, "reason": "correct"})
    out = script.run(ctx, task=TASK, history=history)
    assert not one.calls
    assert [int(im[0, 0, 0]) for im in three.calls[0]["images"]] == [6, 7, 2]
    assert out["route"] == "done" and "(correct)" in out["reason"]


def test_history_without_that_camera_falls_back_to_one_frame(script):
    ctx, one, three = _ctx()
    script.run(ctx, task=TASK, history=[{"cameras": [_cam("other", 3)]}])
    assert one.calls and not three.calls


@pytest.mark.parametrize("word", ["miss", "collision", "fall", "smooth", "failure"])
def test_a_failure_word_routes_regressed_even_when_high(script, word):
    history = [{"cameras": [_cam("top", 1)]}]
    ctx, _, _ = _ctx(three={"progress": 0.95, "found": True, "reason": word})
    out = script.run(ctx, task=TASK, history=history)
    assert out["route"] == "regressed" and word in out["reason"]


def test_a_drop_from_previous_progress_routes_regressed(script):
    ctx, _, _ = _ctx({"progress": 0.29, "found": True})
    assert script.run(ctx, task=TASK, previous_progress=0.6)["route"] == "regressed"
    ctx, _, _ = _ctx({"progress": 0.31, "found": True})
    assert script.run(ctx, task=TASK, previous_progress=0.6)["route"] == "in_progress"
    ctx, _, _ = _ctx({"progress": 0.5, "found": True})
    out = script.run(ctx, task=TASK, previous_progress=0.6, regress_drop=0.05)
    assert out["route"] == "regressed" and "0.600 -> 0.500" in out["reason"]


def test_threshold_is_a_knob(script):
    ctx, _, _ = _ctx({"progress": 0.7, "found": True})
    assert script.run(ctx, task=TASK, done_threshold=0.6)["route"] == "done"


@pytest.mark.parametrize(
    ("kwargs", "obs", "reply", "why"),
    [
        ({}, {"cameras": []}, None, "no RGB frame"),
        ({"camera": "side"}, OBS, None, "no RGB frame"),
        ({}, OBS, {"progress": 0.0, "found": False}, "no progress value"),
        ({}, OBS, ToolError("densereward.progress", "CUDA OOM"), "CUDA OOM"),
    ],
)
def test_low_confidence_exits(script, kwargs, obs, reply, why):
    ctx, _, _ = _ctx(reply, obs=obs)
    out = script.run(ctx, task=TASK, **kwargs)
    assert out["route"] == "low_confidence" and why in out["reason"]
    assert out["progress"] == 0.0


def test_empty_task_is_an_error(script):
    ctx, _, _ = _ctx()
    with pytest.raises(ValueError, match="task"):
        script.run(ctx, task=" ")


def test_example_declares_every_route(skills_registry):
    import json
    from pathlib import Path

    bundle = skills_registry.get(BUNDLE)
    example = json.loads(
        (Path(bundle.bundle_dir) / "examples" / "canonical_subgraph.json").read_text()
    )
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == {"done", "in_progress", "regressed", "low_confidence"}
