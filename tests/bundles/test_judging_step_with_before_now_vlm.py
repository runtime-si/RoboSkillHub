"""judging-step-with-before-now-vlm: FakeContext tests of the prompt, the image order and the routing."""

from __future__ import annotations

import json

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "judging-step-with-before-now-vlm"
TASK = "put the cube in the box"
SUBGOAL = "pick up the cube"


def _frame(name, value):
    return {"name": name, "rgb": np.full((4, 6, 3), value, dtype=np.uint8)}


BEFORE = {"cameras": [_frame("wrist", 1), _frame("overhead", 2)]}
NOW = {"cameras": [_frame("wrist", 11), _frame("overhead", 12)]}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["judge_step"].module


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


def _ctx(reply, now=NOW):
    vlm = Vlm(reply if isinstance(reply, (str, Exception)) else json.dumps(reply))
    return FakeContext({"vlm.query": vlm, "robot.get_observation": now}), vlm


@pytest.mark.parametrize(
    ("status", "action", "route"),
    [
        ("complete", "next", "complete"),
        ("in_progress", "continue", "in_progress"),
        ("failure", "replan", "failed"),
        ("failure", "grasp_tool", "wrong_object"),
        ("failure", "place_tool", "wrong_destination"),
    ],
)
def test_each_action_routes_to_its_exit(script, status, action, route):
    ctx, _ = _ctx({"status": status, "action": action, "reason": "r"})
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)
    assert (out["route"], out["status"], out["action"], out["reason"]) == (
        route,
        status,
        action,
        "r",
    )


def test_route_follows_the_action_when_status_disagrees(script):
    ctx, _ = _ctx({"status": "failure", "action": "continue", "reason": "odd"})
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)
    assert out["route"] == "in_progress" and out["status"] == "failure"


def test_before_images_come_first_external_camera_first(script):
    ctx, vlm = _ctx({"status": "in_progress", "action": "continue"})
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)
    images = vlm.calls[0]["images"]
    assert [int(im[0, 0, 0]) for im in images] == [2, 1, 12, 11]
    assert out["cameras"] == ["overhead", "wrist"]
    prompt = vlm.calls[0]["prompt"]
    assert "Image 1: BEFORE — External camera 'overhead' (scene before this step)" in prompt
    assert "Image 2: BEFORE — Wrist camera 'wrist'" in prompt
    assert "Image 3: NOW — External camera 'overhead' (current scene)" in prompt
    assert "Image 4: NOW — Wrist camera 'wrist'" in prompt
    assert "image" not in vlm.calls[0]


def test_named_cameras_only(script):
    ctx, vlm = _ctx({"status": "in_progress", "action": "continue"})
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL, cameras=["wrist"])
    assert [int(im[0, 0, 0]) for im in vlm.calls[0]["images"]] == [1, 11]
    assert out["cameras"] == ["wrist"]


def test_now_defaults_to_a_fresh_observation_and_can_be_given(script):
    ctx, vlm = _ctx({"status": "complete", "action": "next"})
    given = {"cameras": [_frame("overhead", 99)]}
    script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL, now=given)
    assert [int(im[0, 0, 0]) for im in vlm.calls[0]["images"]] == [2, 99]


def test_cameras_as_a_mapping_and_float_rgb(script):
    before = {"cameras": {"top": {"rgb": np.full((2, 2, 4), 0.5, dtype=np.float32)}}}
    now = {"cameras": {"top": {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}}}
    ctx, vlm = _ctx({"status": "in_progress", "action": "continue"}, now=now)
    out = script.run(ctx, before=before, task=TASK, subgoal=SUBGOAL)
    first = vlm.calls[0]["images"][0]
    assert first.dtype == np.uint8 and first.shape == (2, 2, 3) and int(first[0, 0, 0]) == 127
    assert out["cameras"] == ["top"]


def test_task_text_lists_remaining_subgoals(script):
    ctx, vlm = _ctx({"status": "in_progress", "action": "continue"})
    script.run(
        ctx,
        before=BEFORE,
        task=TASK,
        subgoal=SUBGOAL,
        remaining_subgoals=["carry it over the box", "release it"],
        subgoal_index=2,
    )
    prompt = vlm.calls[0]["prompt"]
    assert 'Overall task: "put the cube in the box"' in prompt
    assert 'Current subgoal (2/4): "pick up the cube"' in prompt
    assert "  3. carry it over the box\n  4. release it" in prompt


def test_last_subgoal_is_said_so(script):
    ctx, vlm = _ctx({"status": "in_progress", "action": "continue"})
    script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL, remaining_subgoals="")
    assert "This is the LAST subgoal." in vlm.calls[0]["prompt"]


def test_recovery_mode_limits_the_vocabulary(script):
    ctx, vlm = _ctx({"status": "failure", "action": "grasp_tool", "grasp_target": "cube"})
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL, recovery_mode="replan")
    assert out["route"] == "in_progress" and out["action"] == "continue"
    prompt = vlm.calls[0]["prompt"]
    assert '"replan"' in prompt and '"grasp_tool"' not in prompt and '"place_tool"' not in prompt


def test_unknown_recovery_mode_raises(script):
    ctx, _ = _ctx({})
    with pytest.raises(ValueError, match="recovery_mode"):
        script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL, recovery_mode="bogus")


def test_invalid_status_defaults_to_in_progress(script):
    ctx, _ = _ctx({"status": "done!", "action": "next"})
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)
    assert out["status"] == "in_progress" and out["route"] == "complete"


def test_missing_fields_default_to_continue(script):
    ctx, _ = _ctx({})
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)
    assert (out["status"], out["action"], out["route"]) == (
        "in_progress",
        "continue",
        "in_progress",
    )


def test_fenced_and_embedded_json_parse(script):
    fenced = '```json\n{"status": "complete", "action": "next", "reason": "in box"}\n```'
    ctx, _ = _ctx(fenced)
    assert script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)["route"] == "complete"
    prose = 'Sure. {"status": "failure", "action": "replan", "reason": "dropped"} Hope this helps.'
    ctx, _ = _ctx(prose)
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)
    assert out["route"] == "failed" and out["reason"] == "dropped"


def test_unparseable_reply_is_no_verdict(script):
    ctx, _ = _ctx("I think the robot is doing fine.")
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)
    assert out["route"] == "no_verdict" and out["reason"] == "parse_failure"
    assert out["raw"] == "I think the robot is doing fine."


def test_vlm_error_is_no_verdict(script):
    ctx, _ = _ctx(ToolError("vlm", "backend unavailable"))
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)
    assert out["route"] == "no_verdict" and "backend unavailable" in out["reason"]


def test_no_shared_camera_is_no_verdict_without_a_call(script):
    ctx, vlm = _ctx({"status": "complete", "action": "next"})
    out = script.run(
        ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL, now={"cameras": [_frame("side", 5)]}
    )
    assert out["route"] == "no_verdict" and not vlm.calls
    assert "no camera with RGB" in out["reason"]


def test_frame_without_rgb_is_skipped(script):
    ctx, vlm = _ctx({"status": "in_progress", "action": "continue"})
    now = {"cameras": [{"name": "wrist", "rgb": None}, _frame("overhead", 12)]}
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL, now=now)
    assert out["cameras"] == ["overhead"] and len(vlm.calls[0]["images"]) == 2


def test_wrong_object_carries_the_target(script):
    ctx, _ = _ctx({"status": "failure", "action": "grasp_tool", "grasp_target": "the red cube"})
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)
    assert out["route"] == "wrong_object" and out["grasp_target"] == "the red cube"


def test_wrong_destination_carries_the_phrase(script):
    reply = {
        "status": "failure",
        "action": "place_tool",
        "place_destination": "in the grey box",
        "place_held_object": "cube",
        "place_stack": True,
    }
    ctx, _ = _ctx(reply)
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)
    assert out["place_destination"] == "in the grey box" and out["place_held_object"] == "cube"
    assert out["place_stack"] is False
    ctx, _ = _ctx(reply)
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL, stack_mode=True)
    assert out["place_stack"] is True


def test_wrong_destination_from_legacy_fields(script):
    reply = {
        "status": "failure",
        "action": "place_tool",
        "place_target": "bowl",
        "place_relation": "on",
    }
    ctx, _ = _ctx(reply)
    out = script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)
    assert out["place_destination"] == "on the bowl"


def test_stack_block_only_in_stack_mode(script):
    ctx, vlm = _ctx({})
    script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)
    assert "STACK CONTROL" not in vlm.calls[0]["prompt"]
    ctx, vlm = _ctx({})
    script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL, stack_mode=True)
    assert "STACK CONTROL" in vlm.calls[0]["prompt"]


def test_model_is_passed_only_when_set(script):
    ctx, vlm = _ctx({})
    script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)
    assert "model" not in vlm.calls[0]
    ctx, vlm = _ctx({})
    script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL, model="gemini-3-flash-preview")
    assert vlm.calls[0]["model"] == "gemini-3-flash-preview"


def test_prompt_keeps_the_upstream_rules(script):
    ctx, vlm = _ctx({})
    script.run(ctx, before=BEFORE, task=TASK, subgoal=SUBGOAL)
    prompt = vlm.calls[0]["prompt"]
    for line in (
        'When in doubt, say "in_progress" with action "continue".',
        "Do NOT replan just because the scene looks different from BEFORE.",
        "fire only for WRONG-OBJECT recovery",
        "fire only for WRONG-DESTINATION recovery",
        '{"status": "...", "action": "...", "reason": "brief explanation"}',
    ):
        assert line in prompt
    assert "Front camera" not in prompt
