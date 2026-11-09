"""densereward bundle: prompts, user turns and parsing with the model replaced by a fake."""

from __future__ import annotations

import numpy as np
import pytest
from gap_core.errors import ToolError


@pytest.fixture(scope="module")
def dr(skills_registry):
    return skills_registry.get("densereward").tools_module


class FakeRuntime:
    """Answers with scripted text and records what the model would have been given."""

    def __init__(self, answers=None, fail=None):
        self.answers = list(answers or [])
        self.fail = fail
        self.calls = []

    def generate(self, system, images, text, max_new_tokens):
        if self.fail:
            raise self.fail
        self.calls.append(
            {"system": system, "images": images, "text": text, "max_new_tokens": max_new_tokens}
        )
        return self.answers.pop(0), 0.01, 0.2


@pytest.fixture
def fake(dr, monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(dr, "_get_runtime", lambda variant: (runtime, 0.0))
    return runtime


def frame(value=0, h=48, w=64):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_tools_registered_with_their_schemas(tool_registry):
    assert set(tool_registry.get("densereward.progress").schema.inputs) == {"image", "task"}
    assert set(tool_registry.get("densereward.progress_with_history").schema.inputs) == {
        "images",
        "task",
    }
    for name in ("densereward.progress", "densereward.progress_with_history"):
        assert set(tool_registry.get(name).schema.outputs) == {
            "progress",
            "found",
            "reason",
            "raw",
            "model",
            "timing",
        }


def test_import_does_not_pull_torch(dr):
    assert dr._runtimes == {}
    assert not {"torch", "transformers"} & set(vars(dr))


def test_one_frame_uses_the_shipped_system_prompt_and_the_card_user_turn(dr, fake):
    fake.answers = ["0.374"]
    out = dr.progress(frame(7), "put the black bowl on the plate")
    assert out["found"] and out["progress"] == pytest.approx(0.374)
    assert out["reason"] == "" and out["raw"] == "0.374"
    assert out["model"].startswith("densereward/densereward-1frame@9796cb2a")
    (call,) = fake.calls
    assert call["system"] == dr.SYSTEM_PROMPT_1FRAME
    assert call["system"].startswith("You are a robot manipulation evaluator. You analyze the image")
    assert call["system"].endswith("Do NOT generate <tool_call> tokens.")
    assert call["text"] == "Task: put the black bowl on the plate."
    assert call["max_new_tokens"] == 8
    assert len(call["images"]) == 1 and call["images"][0].shape == (48, 64, 3)


@pytest.mark.parametrize(
    ("task", "text"),
    [
        ("Task: stack the cups.", "Task: stack the cups."),
        ("  close the drawer!  ", "Task: close the drawer!"),
    ],
)
def test_one_frame_task_text_is_not_doubled(dr, task, text):
    assert dr.task_text(task, "1frame") == text


@pytest.mark.parametrize(
    ("raw", "value", "found"),
    [
        ("1.000", 1.0, True),
        ("0.006", 0.006, True),
        ("reward: 0.75 maybe", 0.75, True),
        ("1.7", 1.0, True),  # the card's regex reads "1.7" whole; clipped
        ("no idea", 0.0, False),
        ("", 0.0, False),
    ],
)
def test_one_frame_parse_is_guarded(dr, fake, raw, value, found):
    fake.answers = [raw]
    out = dr.progress(frame(), "put the cube in the box")
    assert out["found"] is found
    assert out["progress"] == pytest.approx(value)


def test_three_frames_send_oldest_first_with_the_thinking_prompt(dr, fake):
    fake.answers = ["<think>\nmiss\n</think>\n\n0.521"]
    out = dr.progress_with_history([frame(1), frame(2), frame(3)], "put the cube in the box")
    assert out == {
        **out,
        "progress": pytest.approx(0.521),
        "found": True,
        "reason": "miss",
    }
    assert out["model"].startswith("densereward/densereward-3frame-thinking@e9c4ed9c")
    (call,) = fake.calls
    assert call["system"] == dr.SYSTEM_PROMPT_3FRAME
    assert call["text"] == "put the cube in the box"
    assert call["max_new_tokens"] == 32
    assert [int(im[0, 0, 0]) for im in call["images"]] == [1, 2, 3]


def test_three_frames_pad_short_history_and_keep_the_last_three(dr, fake):
    fake.answers = ["<think>correct</think> 0.2", "<think>correct</think> 0.3"]
    dr.progress_with_history([frame(5), frame(6)], "t")
    dr.progress_with_history([frame(v) for v in range(1, 6)], "t")
    assert [int(im[0, 0, 0]) for im in fake.calls[0]["images"]] == [5, 5, 6]
    assert [int(im[0, 0, 0]) for im in fake.calls[1]["images"]] == [3, 4, 5]


def test_three_frames_float_inside_think_is_not_the_reward(dr, fake):
    fake.answers = ["<think>1</think>"]
    out = dr.progress_with_history([frame()], "t")
    assert out["found"] is False and out["reason"] == "1"


def test_rgba_is_accepted_and_bad_images_rejected(dr, fake):
    fake.answers = ["0.5"]
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    assert dr.progress(rgba, "t")["progress"] == 0.5
    assert fake.calls[0]["images"][0].shape == (4, 4, 3)
    with pytest.raises(ValueError, match="uint8"):
        dr.progress(np.zeros((4, 4, 3), dtype=np.float32), "t")
    with pytest.raises(ValueError, match="task"):
        dr.progress(frame(), "  ")
    with pytest.raises(ValueError, match="at least one"):
        dr.progress_with_history([], "t")


def test_model_failures_become_tool_errors(dr, monkeypatch):
    monkeypatch.setattr(
        dr, "_get_runtime", lambda variant: (FakeRuntime(fail=RuntimeError("CUDA OOM")), 0.0)
    )
    with pytest.raises(ToolError, match="CUDA OOM"):
        dr.progress(frame(), "t")

    def no_load(variant):
        raise OSError("no snapshot")

    monkeypatch.setattr(dr, "_get_runtime", no_load)
    with pytest.raises(ToolError, match="failed to load"):
        dr.progress_with_history([frame()], "t")


def test_local_model_override(dr, monkeypatch, tmp_path):
    monkeypatch.setenv("GAP_DENSEREWARD_MODEL_1FRAME", str(tmp_path))
    assert dr._model_id("1frame") == str(tmp_path)
    assert dr._model_dir("1frame") == str(tmp_path)
    assert dr._model_id("3frame").startswith("densereward/")
