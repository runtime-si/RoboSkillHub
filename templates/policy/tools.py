"""{{name}} as a closed-loop policy skill over gap's shared policy loop."""

from __future__ import annotations

from typing import Any

from gap.runtime.policy_skill import PolicyLoopOutput, PolicyLoopSkill
from gap_core.tools import tool


class Policy(PolicyLoopSkill):
    preset = "{{name}}"


@tool(
    name="{{name}}.run", summary="Run the checkpoint in closed loop against its own server.", tags=("policy",)
)
def run_policy(
    ctx,
    observation_stream: Any,
    prompt: str,
    termination_prompt: str = "",
    max_windows: int = 20,
    replan_every: int = 5,
) -> PolicyLoopOutput:
    return Policy().run(
        ctx,
        observation_stream=observation_stream,
        prompt=prompt,
        termination_prompt=termination_prompt,
        max_windows=max_windows,
        replan_every=replan_every,
    )
