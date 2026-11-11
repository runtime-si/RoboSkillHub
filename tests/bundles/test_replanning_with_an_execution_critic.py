"""replanning-with-an-execution-critic: prompt, parsing and escalation with a fake VLM."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext

OBS = {"cameras": [{"name": "top", "rgb": np.zeros((8, 8, 3), np.uint8)}]}


@pytest.fixture(scope="module")
def critic(skills_registry):
    return (
        skills_registry.get("replanning-with-an-execution-critic")
        .canonical_scripts["critique_action"]
        .module
    )


def run(critic, reply, **kw):
    calls = []

    def q(**k):
        calls.append(k)
        if isinstance(reply, Exception):
            raise reply
        return {"text": reply}

    ctx = FakeContext({"vlm.query": q, "robot.get_observation": OBS})
    return critic.run(ctx, task="t", subgoal="s", proposed_action="a", **kw), calls


def test_suitable_continues_and_prompt_has_the_sections(critic):
    out, calls = run(
        critic,
        'Reasoning...\n```YAML\nAction Suitability: True\nFeedback: "fine"\n```',
        history="h",
    )
    assert out["route"] == "continue" and out["suitable"] and out["feedback"] == "fine"
    p = calls[0]["prompt"]
    assert (
        "Robot's Subgoal:\ns" in p
        and "Belief:\nh" in p
        and "Proposed Action:\na" in p
        and "Knowledge" not in p
    )
    assert len(calls[0]["images"]) == 1


def test_rejection_replans_the_subtask_then_the_task(critic):
    reply = '```YAML\nAction Suitability: False\nFeedback: "the brick is not pressed"\n```'
    out, _ = run(critic, reply)
    assert out["route"] == "replan_subtask" and out["rejections"] == 1
    out, _ = run(critic, reply, rejections=1)
    assert out["route"] == "replan_task" and out["rejections"] == 2


@pytest.mark.parametrize("reply", ["no verdict here", RuntimeError("down")])
def test_no_verdict(critic, reply):
    out, _ = run(critic, reply)
    assert out["route"] == "no_verdict"


def test_no_frames_is_no_verdict(critic):
    ctx = FakeContext({"vlm.query": {"text": "Action Suitability: True"}})
    assert (
        critic.run(ctx, task="t", subgoal="s", proposed_action="a", observation={"cameras": []})[
            "route"
        ]
        == "no_verdict"
    )
