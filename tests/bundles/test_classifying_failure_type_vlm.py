"""classifying-failure-type-vlm: FakeContext tests of the prompt, the frame strip and the routing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "classifying-failure-type-vlm"
TASK = "put the cube in the box"
STEP = "grasp the cube and lift it"
CLASSES = [
    "missed_grasp",
    "slipped",
    "wrong_object",
    "misaligned",
    "collision_or_stuck",
    "incomplete",
]


def _frame(name, value):
    return {"name": name, "rgb": np.full((4, 6, 3), value, dtype=np.uint8)}


BEFORE = {"cameras": [_frame("wrist", 1), _frame("overhead", 2)]}
AFTER = {"cameras": [_frame("wrist", 11), _frame("overhead", 12)]}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["classify_failure_vlm"].module


class Vlm:
    """Records each vlm.query call and answers with ``reply`` (a string, or an exception)."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return {"text": self.reply}


def _ctx(reply, after=AFTER):
    vlm = Vlm(reply if isinstance(reply, (str, Exception)) else json.dumps(reply))
    return FakeContext({"vlm.query": vlm, "robot.get_observation": after}), vlm


def _fail(cls, robofac="Grasping error.", **extra):
    return {
        "success": False,
        "robofac_type": robofac,
        "failure_class": cls,
        "explanation": "e",
        "correction": "c",
        "low_level_correction": "l",
        **extra,
    }


@pytest.mark.parametrize("cls", CLASSES)
def test_each_class_routes_to_its_exit(script, cls):
    ctx, _ = _ctx(_fail(cls))
    out = script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    assert out["route"] == cls and out["failure_class"] == cls and out["failed"] is True
    assert (out["explanation"], out["correction"], out["low_level_correction"]) == ("e", "c", "l")
    assert out["robofac_type"] == "Grasping error."


def test_success_is_no_failure_whatever_the_class(script):
    ctx, _ = _ctx({"success": True, "failure_class": "slipped", "explanation": ""})
    out = script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    assert out["route"] == "no_failure" and out["failed"] is False and out["failure_class"] == ""


@pytest.mark.parametrize(("value", "route"), [("yes", "no_failure"), ("No.", "slipped")])
def test_success_as_a_word(script, value, route):
    ctx, _ = _ctx({**_fail("slipped"), "success": value})
    assert script.run(ctx, before=BEFORE, task=TASK, step=STEP)["route"] == route


def test_every_exit_is_declared_in_the_example(script):
    path = Path(__file__).parents[2] / "skills" / BUNDLE / "examples" / "canonical_subgraph.json"
    example = json.loads(path.read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == {*script.CLASSES, script.NO_FAILURE, script.NO_VERDICT}


@pytest.mark.parametrize(
    ("robofac", "route", "misalignment"),
    [
        ("Wrong target object.", "wrong_object", ""),
        ("Position deviation.", "misaligned", "translation"),
        ("orientation deviation", "misaligned", "rotation"),
        ("Step omission.", "incomplete", ""),
        ("Grasping error.", "no_verdict", ""),
        ("Timing error.", "no_verdict", ""),
    ],
)
def test_unknown_class_falls_back_on_the_robofac_type(script, robofac, route, misalignment):
    ctx, _ = _ctx(_fail("dropped it", robofac=robofac))
    out = script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    assert out["route"] == route and out["misalignment"] == misalignment
    assert out["robofac_type"] == robofac.capitalize().rstrip(".") + "."


def test_misalignment_only_for_the_misaligned_class(script):
    ctx, _ = _ctx(_fail("misaligned", robofac="Orientation deviation."))
    assert script.run(ctx, before=BEFORE, task=TASK, step=STEP)["misalignment"] == "rotation"
    ctx, _ = _ctx(_fail("collision_or_stuck", robofac="Position deviation."))
    assert script.run(ctx, before=BEFORE, task=TASK, step=STEP)["misalignment"] == ""


def test_invalid_robofac_type_is_blank(script):
    ctx, _ = _ctx(_fail("slipped", robofac="butterfingers"))
    out = script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    assert out["route"] == "slipped" and out["robofac_type"] == ""


def test_class_is_case_insensitive(script):
    ctx, _ = _ctx(_fail(" Missed_Grasp "))
    assert script.run(ctx, before=BEFORE, task=TASK, step=STEP)["route"] == "missed_grasp"


def test_missing_success_is_no_verdict_and_keeps_the_text(script):
    reply = _fail("slipped")
    del reply["success"]
    ctx, _ = _ctx(reply)
    out = script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    assert out["route"] == "no_verdict" and out["explanation"] == "no success verdict: e"
    assert out["failed"] is False


def test_failure_without_any_class_is_no_verdict(script):
    ctx, _ = _ctx({"success": False})
    out = script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    assert out["route"] == "no_verdict" and out["explanation"] == "no failure class"


def test_frames_are_before_during_after_per_camera_external_first(script):
    ctx, vlm = _ctx(_fail("slipped"))
    during = [{"cameras": [_frame("wrist", 21), _frame("overhead", 22)]}]
    out = script.run(ctx, before=BEFORE, task=TASK, step=STEP, during=during)
    assert [int(im[0, 0, 0]) for im in vlm.calls[0]["images"]] == [2, 22, 12, 1, 21, 11]
    assert out["cameras"] == ["overhead", "wrist"] and out["frames"] == 6
    prompt = vlm.calls[0]["prompt"]
    for line in (
        "Image 1: BEFORE - external camera 'overhead'",
        "Image 2: DURING 1/1 - external camera 'overhead'",
        "Image 3: AFTER - external camera 'overhead'",
        "Image 4: BEFORE - wrist camera 'wrist'",
        "Image 6: AFTER - wrist camera 'wrist'",
    ):
        assert line in prompt


def test_during_is_thinned_evenly(script):
    ctx, vlm = _ctx(_fail("slipped"))
    during = [{"cameras": [_frame("overhead", 30 + i)]} for i in range(9)]
    script.run(
        ctx, before=BEFORE, task=TASK, step=STEP, during=during, cameras=["overhead"], max_during=3
    )
    assert [int(im[0, 0, 0]) for im in vlm.calls[0]["images"]] == [2, 30, 34, 38, 12]
    ctx, vlm = _ctx(_fail("slipped"))
    script.run(
        ctx, before=BEFORE, task=TASK, step=STEP, during=during, cameras=["overhead"], max_during=0
    )
    assert [int(im[0, 0, 0]) for im in vlm.calls[0]["images"]] == [2, 12]


def test_during_frame_missing_a_camera_is_skipped(script):
    ctx, vlm = _ctx(_fail("slipped"))
    during = [{"cameras": [_frame("wrist", 21)]}, {"cameras": [_frame("overhead", 23)]}]
    out = script.run(ctx, before=BEFORE, task=TASK, step=STEP, during=during)
    assert [int(im[0, 0, 0]) for im in vlm.calls[0]["images"]] == [2, 23, 12, 1, 21, 11]
    assert "DURING 1/1 - external camera 'overhead'" in vlm.calls[0]["prompt"]
    assert out["frames"] == 6


def test_after_defaults_to_a_fresh_observation_and_can_be_given(script):
    ctx, vlm = _ctx(_fail("slipped"))
    script.run(
        ctx, before=BEFORE, task=TASK, step=STEP, after={"cameras": [_frame("overhead", 99)]}
    )
    assert [int(im[0, 0, 0]) for im in vlm.calls[0]["images"]] == [2, 99]
    ctx, vlm = _ctx(_fail("slipped"))
    script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    assert [int(im[0, 0, 0]) for im in vlm.calls[0]["images"]] == [2, 12, 1, 11]


def test_named_cameras_and_mapping_and_float_rgb(script):
    before = {"cameras": {"top": {"rgb": np.full((2, 2, 4), 0.5, dtype=np.float32)}}}
    after = {"cameras": {"top": {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}}}
    ctx, vlm = _ctx(_fail("slipped"), after=after)
    out = script.run(ctx, before=before, task=TASK, step=STEP, cameras=["top"])
    first = vlm.calls[0]["images"][0]
    assert first.dtype == np.uint8 and first.shape == (2, 2, 3) and int(first[0, 0, 0]) == 127
    assert out["cameras"] == ["top"]


def test_no_shared_camera_is_no_verdict_without_a_call(script):
    ctx, vlm = _ctx(_fail("slipped"), after={"cameras": [_frame("side", 5)]})
    out = script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    assert out["route"] == "no_verdict" and not vlm.calls
    assert "no camera with RGB" in out["explanation"]


def test_prompt_carries_task_step_robofac_questions_and_classes(script):
    ctx, vlm = _ctx(_fail("slipped"))
    script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    prompt = vlm.calls[0]["prompt"]
    assert (
        f'Overall task: "{TASK}"' in prompt
        and f'The step the robot just executed: "{STEP}"' in prompt
    )
    assert "can you determine whether it was successful?" in prompt
    assert (
        "['Orientation deviation.', 'Step omission.', 'Wrong target object.', "
        "'Timing error.', 'Grasping error.', 'Position deviation.']" in prompt
    )
    assert "provide a detailed explanation of why the robotic arm failed" in prompt
    assert "which direction (relative to the robot arm) and how much" in prompt
    for cls in CLASSES:
        assert f'- "{cls}": ' in prompt


def test_fenced_and_embedded_json_parse(script):
    ctx, _ = _ctx("```json\n" + json.dumps(_fail("wrong_object")) + "\n```")
    assert script.run(ctx, before=BEFORE, task=TASK, step=STEP)["route"] == "wrong_object"
    ctx, _ = _ctx("Here: " + json.dumps(_fail("incomplete")) + " done.")
    assert script.run(ctx, before=BEFORE, task=TASK, step=STEP)["route"] == "incomplete"


def test_unparseable_reply_is_no_verdict(script):
    ctx, _ = _ctx("The cube fell, I think.")
    out = script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    assert out["route"] == "no_verdict" and out["explanation"] == "parse_failure"
    assert out["raw"] == "The cube fell, I think." and out["frames"] == 4


def test_reply_cut_off_mid_json_is_salvaged(script):
    cut = json.dumps(_fail("collision_or_stuck", robofac="Position deviation."))[:-30]
    ctx, _ = _ctx(cut)
    out = script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    assert out["route"] == "collision_or_stuck" and out["robofac_type"] == "Position deviation."
    assert out["explanation"] == "e" and out["low_level_correction"] == ""


def test_reply_cut_off_before_the_verdict_is_no_verdict(script):
    ctx, _ = _ctx('{"observations": ["Image 1: the cube on the table", "Image 2: the')
    out = script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    assert out["route"] == "no_verdict" and out["explanation"] == "parse_failure"


def test_observations_are_returned(script):
    ctx, vlm = _ctx(_fail("slipped", observations=["Image 1: a", "Image 2: b"]))
    out = script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    assert out["observations"] == ["Image 1: a", "Image 2: b"]
    assert '"observations": ["Image 1: ...", ...]' in vlm.calls[0]["prompt"]
    ctx, _ = _ctx(_fail("slipped", observations="not a list"))
    assert script.run(ctx, before=BEFORE, task=TASK, step=STEP)["observations"] == []


def test_crop_cuts_every_frame_of_a_camera(script):
    big = {
        "cameras": [
            {"name": "overhead", "rgb": np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)}
        ]
    }
    during = [big]
    ctx, vlm = _ctx(_fail("slipped"), after=big)
    script.run(ctx, before=big, task=TASK, step=STEP, during=during, crop=[2, 3, 8, 7])
    assert [im.shape for im in vlm.calls[0]["images"]] == [(4, 6, 3)] * 3
    assert int(vlm.calls[0]["images"][0][0, 0, 0]) == int(big["cameras"][0]["rgb"][3, 2, 0])


def test_crop_per_camera_and_out_of_bounds(script):
    ctx, vlm = _ctx(_fail("slipped"))
    script.run(ctx, before=BEFORE, task=TASK, step=STEP, crop={"wrist": [-5, 1, 99, 3]})
    shapes = [im.shape for im in vlm.calls[0]["images"]]
    assert shapes == [(4, 6, 3), (4, 6, 3), (2, 6, 3), (2, 6, 3)]
    ctx, vlm = _ctx(_fail("slipped"))
    script.run(ctx, before=BEFORE, task=TASK, step=STEP, crop=[5, 3, 5, 3])
    assert [im.shape for im in vlm.calls[0]["images"]] == [(4, 6, 3)] * 4


def test_vlm_error_is_no_verdict(script):
    ctx, _ = _ctx(ToolError("vlm", "backend unavailable"))
    out = script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    assert out["route"] == "no_verdict" and "backend unavailable" in out["explanation"]


def test_model_is_passed_only_when_set(script):
    ctx, vlm = _ctx(_fail("slipped"))
    script.run(ctx, before=BEFORE, task=TASK, step=STEP)
    assert "model" not in vlm.calls[0]
    ctx, vlm = _ctx(_fail("slipped"))
    script.run(ctx, before=BEFORE, task=TASK, step=STEP, model="gemini-3-flash-preview")
    assert vlm.calls[0]["model"] == "gemini-3-flash-preview"
