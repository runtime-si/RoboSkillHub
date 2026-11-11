# Adapted from CLEA (https://github.com/sp4595/clea-closed-loop-embodied-agent) at
# eea73b60cf09: CLEA/templates/prompts/Critic.prompt and CLEA/Modules/Critic.py
# (get_feed_back: the YAML "Action Suitability" / "Feedback" reply), Apache-2.0.
# Changes: one call through the hub's vlm.query with the current camera frames (upstream used
# its own LangChain client and a two-robot side-by-side image); the household-simulator lines
# of the prompt (two-robot hand-off, navigation points, the API list, "encourage exploration,
# give less False") are dropped and the optional sections are filled only when given; the
# verdict routes continue / replan_subtask, and repeated rejections (``rejections`` so far)
# escalate to replan_task as CLEA's hierarchical re-planning does; an unparseable reply routes
# no_verdict (upstream raised).
"""Critique a proposed next action against the subgoal and the current view, before executing it."""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

MAX_REJECTIONS = 2
"""Rejections of actions for one subgoal after which the subgoal itself is re-planned (CLEA re-plans the
subtask on a rejection and the task when re-planning the subtask keeps failing)."""


class Output(TypedDict):
    route: str
    suitable: bool
    feedback: str
    rejections: int
    raw: str


SYSTEM = (
    "You are a highly advanced language model tasked with critically evaluating the suitability of a robot's "
    "proposed next action given its assigned subgoal, summarized execution history, and current environment. "
    "Each action executed by the robot alters the environment: reflect on the action's immediate significance "
    "in the current context and anticipate how it changes what can be done next.\n\n"
    "Your responsibilities:\n"
    "1. Reasoning: analyze the subgoal, execution history, current environment (the images) and proposed "
    "action; assess whether the action is suitable and supports progress toward the subgoal.\n"
    "2. Action Suitability: True (the action is relevant and beneficial now) or False (unsuitable or "
    "potentially counterproductive).\n"
    "3. Constructive Feedback: if unsuitable, why, the key problem in relation to the current environment and "
    "subgoal, and how to improve.\n\n"
    "Output: reason step by step, then end with\n"
    '```YAML\nAction Suitability: True/False\nFeedback: "..."\n```'
)


def prompt_text(task: str, subgoal: str, action: str, history: str, knowledge: str) -> str:
    parts = [SYSTEM, "", f"Robot's task:\n{task}", f"Robot's Subgoal:\n{subgoal}"]
    if history:
        parts.append(f"Belief:\n{history}")
    if knowledge:
        parts.append(f"Knowledge:\n{knowledge}")
    parts += [
        "Current Environment: the images that follow this text.",
        f"Proposed Action:\n{action}",
    ]
    return "\n\n".join(parts)


def parse_reply(text: str) -> tuple[bool | None, str]:
    m = re.search(r"Action\s+Suitability\s*:\s*\**\s*(True|False)", text, re.IGNORECASE)
    f = re.search(r"Feedback\s*:\s*\"?(.*?)\"?\s*(?:```|$)", text, re.IGNORECASE | re.S)
    return (None if m is None else m.group(1).lower() == "true"), (
        f.group(1).strip()[:400] if f else ""
    )


def _images(obs: Any) -> list[np.ndarray]:
    cams = (obs or {}).get("cameras") or []
    if isinstance(cams, dict):
        cams = list(cams.values())
    out = []
    for c in cams:
        a = c.get("rgb") if isinstance(c, dict) else None
        if a is None:
            continue
        a = np.asarray(a)
        if a.ndim == 3 and a.shape[2] >= 3:
            out.append(np.ascontiguousarray(a[..., :3].astype(np.uint8)))
    return out


def run(
    ctx: NodeContext,
    task: str,
    subgoal: str,
    proposed_action: str,
    history: str = "",
    knowledge: str = "",
    observation: dict[str, Any] | None = None,
    rejections: int = 0,
    max_rejections: int = MAX_REJECTIONS,
    model: str | None = None,
) -> Output:
    """Is ``proposed_action`` suitable now? ``observation`` defaults to a fresh robot.get_observation;
    ``rejections`` counts earlier rejections for this subgoal."""
    if observation is None:
        observation = ctx.tool("robot.get_observation")
    imgs = _images(observation)
    out: Output = {
        "route": "no_verdict",
        "suitable": False,
        "feedback": "",
        "rejections": int(rejections),
        "raw": "",
    }
    if not imgs:
        out["feedback"] = "no camera frame with RGB"
        return out
    kw: dict[str, Any] = {
        "prompt": prompt_text(task, subgoal, proposed_action, history, knowledge),
        "images": imgs,
    }
    if model:
        kw["model"] = model
    try:
        raw = str(ctx.tool("vlm.query", **kw).get("text") or "")
    except Exception as exc:  # noqa: BLE001  (auth / network / unserved tool)
        out["feedback"] = f"vlm.query failed: {exc}"
        return out
    ok, feedback = parse_reply(raw)
    out.update(raw=raw, feedback=feedback)
    if ok is None:
        out["feedback"] = "reply has no Action Suitability"
        return out
    out["suitable"] = ok
    if ok:
        out["route"] = "continue"
    else:
        out["rejections"] = int(rejections) + 1
        out["route"] = (
            "replan_task" if out["rejections"] >= int(max_rejections) else "replan_subtask"
        )
    logger.info("critique_action: %s (%s)", out["route"], feedback[:120])
    return out
