"""detecting-subtask-success: FakeContext tests of the prompt, the parse, the poll loop and the exits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "detecting-subtask-success"
SUBTASK = "put the cube in the box"
YES = "Q1: cube in box. Q4: yes.\nFINAL ANSWER: yes"
NO = "Q1: cube on table. Q4: no.\nFINAL ANSWER: no"


def _frame(name, value):
    return {"name": name, "rgb": np.full((4, 6, 3), value, dtype=np.uint8)}


def _obs(*values, names=("top", "wrist")):
    return {"cameras": [_frame(n, v) for n, v in zip(names, values, strict=False)]}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["detect_success"].module


class Clock:
    """Stands in for the script's ``time``: sleep advances it, each VLM call costs ``call_s``."""

    def __init__(self):
        self.t = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


@pytest.fixture
def clock(script, monkeypatch):
    c = Clock()
    monkeypatch.setattr(script, "time", c)
    return c


class Vlm:
    """Answers each vlm.query with the next reply (a string or an exception; the last one repeats)."""

    def __init__(self, replies, clock=None, call_s=0.0):
        self.replies = list(replies)
        self.calls = []
        self.clock, self.call_s = clock, call_s

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.clock is not None:
            self.clock.t += self.call_s
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return {"text": reply}


def _ctx(replies, observations, clock=None, call_s=0.0):
    vlm = Vlm(replies if isinstance(replies, list) else [replies], clock, call_s)
    return FakeContext({"vlm.query": vlm, "robot.get_observation": observations}), vlm


def test_yes_succeeds_on_one_check(script, clock):
    ctx, vlm = _ctx(YES, _obs(1, 2))
    out = script.run(ctx, subtask=SUBTASK)
    assert (out["route"], out["success"], out["answer"], out["checks"]) == (
        "succeeded",
        True,
        "yes",
        1,
    )
    assert out["cameras"] == ["top", "wrist"] and out["raw"] == YES and not clock.sleeps


def test_no_is_not_yet_when_asked_once(script, clock):
    ctx, vlm = _ctx(NO, _obs(1, 2))
    out = script.run(ctx, subtask=SUBTASK)
    assert (out["route"], out["success"], out["answer"], out["checks"]) == (
        "not_yet",
        False,
        "no",
        1,
    )
    assert out["reason"] == "FINAL ANSWER: no" and len(vlm.calls) == 1


def test_unparsed_reply_and_follow_up_count_as_no(script, clock):
    ctx, vlm = _ctx("The cube looks like it is in the box.", _obs(1, 2))
    out = script.run(ctx, subtask=SUBTASK)
    assert out["route"] == "not_yet" and out["answer"] == "" and out["checks"] == 1
    assert out["follow_ups"] == 1 and len(vlm.calls) == 2
    assert "no FINAL ANSWER" in out["reason"]


def test_cut_off_reply_gets_one_follow_up(script, clock):
    cut = "Q1: the cube is held by the gripper.\nQ2: The cube is"
    ctx, vlm = _ctx([cut, "FINAL ANSWER: yes"], _obs(1, 2))
    out = script.run(ctx, subtask="pick up the cube")
    assert out["route"] == "succeeded" and out["follow_ups"] == 1 and out["checks"] == 1
    first, second = vlm.calls
    assert second["prompt"].startswith(first["prompt"])
    assert "cut off before its final line:\n" + cut in second["prompt"]
    assert "Reply with only the final line" in second["prompt"]
    assert [int(im[0, 0, 0]) for im in second["images"]] == [1, 2]
    assert out["raw"] == cut + "\n[follow-up] FINAL ANSWER: yes"


def test_follow_up_can_be_turned_off(script, clock):
    ctx, vlm = _ctx(["Q1: cut", YES], _obs(1))
    out = script.run(ctx, subtask=SUBTASK, follow_up=False)
    assert out["route"] == "not_yet" and len(vlm.calls) == 1 and out["follow_ups"] == 0


def test_answered_reply_gets_no_follow_up(script, clock):
    ctx, vlm = _ctx(NO, _obs(1))
    out = script.run(ctx, subtask=SUBTASK)
    assert len(vlm.calls) == 1 and out["follow_ups"] == 0


def test_every_camera_is_sent_in_order_and_announced(script, clock):
    ctx, vlm = _ctx(NO, _obs(1, 2))
    script.run(ctx, subtask=SUBTASK)
    call = vlm.calls[0]
    assert [int(im[0, 0, 0]) for im in call["images"]] == [1, 2] and "image" not in call
    prompt = call["prompt"]
    assert "what the robot sees now" in prompt and "at the start of the task" not in prompt
    assert "Image 1: top\nImage 2: wrist" in prompt
    assert "The robot's task is: put the cube in the box" in prompt
    assert "answer the following questions first:" in prompt
    assert "Please limit your response to 50 words." in prompt
    assert prompt.index("The robot's task is") < prompt.index("Q1 Tell me about the state")


def test_prompt_keeps_upstream_questions_and_leniency(script, clock):
    ctx, vlm = _ctx(NO, _obs(1))
    script.run(ctx, subtask=SUBTASK)
    prompt = vlm.calls[0]["prompt"]
    for line in (
        "Q2 What are the relationships between the relevant objects at each time?",
        "Q3 What state should the objects be in if the robot successfully completed the task?",
        "put the spoon in the compost bin",
        "Use the most linient way to intrepret task completion.",
        "If you think the robot finished the task, answer: 'FINAL ANSWER: yes'",
        "Do not mention the word 'FINAL ANSWER' anywhere else.",
    ):
        assert line in prompt


def test_strict_replaces_the_lenient_line(script, clock):
    ctx, vlm = _ctx(NO, _obs(1))
    script.run(ctx, subtask=SUBTASK, lenient=False)
    prompt = vlm.calls[0]["prompt"]
    assert "linient" not in prompt and "Use a strict way to interpret task completion." in prompt


def test_word_limit_can_be_changed_or_dropped(script, clock):
    ctx, vlm = _ctx(NO, _obs(1))
    script.run(ctx, subtask=SUBTASK, word_limit=120)
    assert "limit your response to 120 words" in vlm.calls[0]["prompt"]
    ctx, vlm = _ctx(NO, _obs(1))
    script.run(ctx, subtask=SUBTASK, word_limit=None)
    assert "limit your response" not in vlm.calls[0]["prompt"]


def test_start_images_come_first(script, clock):
    ctx, vlm = _ctx(NO, _obs(11, 12))
    script.run(ctx, subtask=SUBTASK, start=_obs(1, 2))
    call = vlm.calls[0]
    assert [int(im[0, 0, 0]) for im in call["images"]] == [1, 2, 11, 12]
    prompt = call["prompt"]
    assert prompt.index("at the start of the task") < prompt.index("what the robot sees now")
    assert "Image 1: top\nImage 2: wrist" in prompt and "Image 3: top\nImage 4: wrist" in prompt


def test_named_cameras_pick_and_order(script, clock):
    ctx, vlm = _ctx(YES, _obs(1, 2))
    out = script.run(ctx, subtask=SUBTASK, cameras=["wrist", "side", "top"])
    assert [int(im[0, 0, 0]) for im in vlm.calls[0]["images"]] == [2, 1]
    assert out["cameras"] == ["wrist", "top"]


def test_now_is_used_instead_of_a_fresh_observation(script, clock):
    ctx, vlm = _ctx(YES, _obs(5, 6))
    script.run(ctx, subtask=SUBTASK, now=_obs(7, names=("top",)))
    assert [int(im[0, 0, 0]) for im in vlm.calls[0]["images"]] == [7]
    assert [c.tool for c in ctx.calls] == ["vlm.query"]


def test_mapping_cameras_float_rgb_and_missing_rgb(script, clock):
    now = {
        "cameras": {
            "top": {"rgb": np.full((2, 2, 4), 0.5, dtype=np.float32)},
            "wrist": {"rgb": None},
            "depth_only": {"depth": np.zeros((2, 2))},
        }
    }
    ctx, vlm = _ctx(YES, now)
    out = script.run(ctx, subtask=SUBTASK)
    image = vlm.calls[0]["images"][0]
    assert image.dtype == np.uint8 and image.shape == (2, 2, 3) and int(image[0, 0, 0]) == 127
    assert out["cameras"] == ["top"] and len(vlm.calls[0]["images"]) == 1


def test_no_rgb_is_no_view_without_a_call(script, clock):
    ctx, vlm = _ctx(YES, {"cameras": [{"name": "top", "rgb": None}]})
    out = script.run(ctx, subtask=SUBTASK)
    assert out["route"] == "no_view" and not vlm.calls and out["checks"] == 0
    assert "no camera with RGB" in out["reason"] and "top" in out["reason"]


def test_vlm_error_once_is_no_view(script, clock):
    ctx, _ = _ctx(ToolError("vlm", "backend unavailable"), _obs(1))
    out = script.run(ctx, subtask=SUBTASK)
    assert (
        out["route"] == "no_view" and out["errors"] == 1 and "backend unavailable" in out["reason"]
    )


def test_polls_fresh_frames_until_yes(script, clock):
    frames = [_obs(1), _obs(2), _obs(3)]
    ctx, vlm = _ctx([NO, NO, YES], frames, clock, call_s=2.0)
    out = script.run(ctx, subtask=SUBTASK, timeout_s=30.0)
    assert out["route"] == "succeeded" and out["checks"] == 3
    assert [int(c["images"][0][0, 0, 0]) for c in vlm.calls] == [1, 2, 3]
    assert clock.sleeps == [0.2, 0.2] and out["elapsed_s"] == pytest.approx(6.4)


def test_poll_times_out(script, clock):
    frames = [_obs(i) for i in range(1, 50)]
    ctx, vlm = _ctx(NO, frames, clock, call_s=3.0)
    out = script.run(ctx, subtask=SUBTASK, timeout_s=10.0, interval_s=1.0)
    assert out["route"] == "timed_out" and out["success"] is False
    assert out["checks"] == 3 and out["answer"] == "no"
    assert "no yes in 3 checks over 10 s" in out["reason"]


def test_identical_frames_are_not_asked_again(script, clock):
    ctx, vlm = _ctx(NO, [_obs(1), _obs(1), _obs(1), _obs(2), _obs(2)], clock)
    out = script.run(ctx, subtask=SUBTASK, timeout_s=0.7, interval_s=0.2)
    assert out["route"] == "timed_out"
    assert [int(c["images"][0][0, 0, 0]) for c in vlm.calls] == [1, 2] and out["checks"] == 2


def test_a_new_timestamp_is_a_new_frame(script, clock):
    a, b = {**_obs(1), "timestamp": 1.0}, {**_obs(1), "timestamp": 2.0}
    ctx, vlm = _ctx([NO, YES], [a, b], clock)
    out = script.run(ctx, subtask=SUBTASK, timeout_s=5.0)
    assert out["route"] == "succeeded" and len(vlm.calls) == 2


def test_failed_call_is_retried_on_the_same_frame(script, clock):
    ctx, vlm = _ctx([ToolError("vlm", "429"), YES], [_obs(1), _obs(1)], clock)
    out = script.run(ctx, subtask=SUBTASK, timeout_s=5.0)
    assert out["route"] == "succeeded" and out["errors"] == 1 and out["checks"] == 1


def test_poll_with_every_call_failing_is_no_view(script, clock):
    ctx, vlm = _ctx(ToolError("vlm", "down"), [_obs(i) for i in range(20)], clock, call_s=1.0)
    out = script.run(ctx, subtask=SUBTASK, timeout_s=2.5)
    assert out["route"] == "no_view" and out["errors"] == len(vlm.calls) >= 2
    assert "every vlm.query failed" in out["reason"]


def test_poll_without_any_view_is_no_view(script, clock):
    ctx, vlm = _ctx(YES, {"cameras": []}, clock)
    out = script.run(ctx, subtask=SUBTASK, timeout_s=1.0, interval_s=0.5)
    assert out["route"] == "no_view" and not vlm.calls


def test_a_camera_appearing_mid_poll_is_judged(script, clock):
    ctx, vlm = _ctx(YES, [{"cameras": []}, _obs(3)], clock)
    out = script.run(ctx, subtask=SUBTASK, timeout_s=1.0)
    assert out["route"] == "succeeded" and out["cameras"] == ["top"]


def test_model_is_passed_only_when_set(script, clock):
    ctx, vlm = _ctx(NO, _obs(1))
    script.run(ctx, subtask=SUBTASK)
    assert "model" not in vlm.calls[0]
    ctx, vlm = _ctx(NO, _obs(1))
    script.run(ctx, subtask=SUBTASK, model="gemini-robotics-er-2-preview")
    assert vlm.calls[0]["model"] == "gemini-robotics-er-2-preview"


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        ("FINAL ANSWER: yes", "yes"),
        ("final answer: No", "no"),
        ("**Final Answer:** yes", "yes"),
        ("FINAL ANSWER: **yes**", "yes"),
        ("$\\boxed{yes}$", "yes"),
        ("FINAL ANSWER: no ... on reflection FINAL ANSWER: yes", "yes"),
        ("FINAL ANSWER: yesterday", ""),
        ("yes, the cube is in the box", ""),
        ("", ""),
    ],
)
def test_parse_answer(script, text, answer):
    assert script.parse_answer(text) == answer


def test_every_route_is_declared_in_the_example():
    path = (
        Path(__file__).resolve().parents[2]
        / "skills"
        / BUNDLE
        / "examples"
        / "canonical_subgraph.json"
    )
    example = json.loads(path.read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == {"succeeded", "not_yet", "timed_out", "no_view"}
