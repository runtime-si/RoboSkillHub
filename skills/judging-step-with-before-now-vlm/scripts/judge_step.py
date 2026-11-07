# Adapted from VoLoAgent (https://github.com/NVlabs/VoLoAgent),
# vlm_orchestrator/failure_handlers/vlm.py (prompt, parser) and
# vlm_orchestrator/strategies/subgoal_base.py (BEFORE / NOW labels), Apache-2.0,
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# Changes: one judgment per call instead of a periodic handler; BEFORE is the
# observation a graph captured before the step (upstream: the episode start);
# the system prompt and the task text go into one vlm.query prompt with the
# images listed by number (the hub's vlm tool has no system prompt and appends
# images after the text); the mirrored-front-camera block is dropped; the
# action tokens become graph exits; a failed or unparseable reply routes
# no_verdict (upstream logged it and continued).
"""Judge the current subgoal from BEFORE / NOW frames with a VLM: complete, in progress or failed."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

STATUS_COMPLETE = "complete"
STATUS_FAILURE = "failure"
STATUS_IN_PROGRESS = "in_progress"

ACTION_NEXT = "next"
ACTION_CONTINUE = "continue"
ACTION_REPLAN = "replan"
ACTION_GRASP = "grasp_tool"
ACTION_PLACE = "place_tool"

ACTION_SETS: dict[str, set[str]] = {
    "replan": {ACTION_NEXT, ACTION_CONTINUE, ACTION_REPLAN},
    "replan_grasp": {ACTION_NEXT, ACTION_CONTINUE, ACTION_REPLAN, ACTION_GRASP},
    "grasp": {ACTION_NEXT, ACTION_CONTINUE, ACTION_GRASP},
    "place": {ACTION_NEXT, ACTION_CONTINUE, ACTION_PLACE},
    "replan_place": {ACTION_NEXT, ACTION_CONTINUE, ACTION_REPLAN, ACTION_PLACE},
    "tools": {ACTION_NEXT, ACTION_CONTINUE, ACTION_GRASP, ACTION_PLACE},
    "replan_tools": {ACTION_NEXT, ACTION_CONTINUE, ACTION_REPLAN, ACTION_GRASP, ACTION_PLACE},
}
"""VoLoAgent's ``--recovery-mode`` action vocabularies, verbatim."""

DEFAULT_RECOVERY_MODE = "replan_tools"
"""Every action, so every exit is reachable (upstream defaults to ``replan``)."""

ROUTES = {
    ACTION_NEXT: "complete",
    ACTION_CONTINUE: "in_progress",
    ACTION_REPLAN: "failed",
    ACTION_GRASP: "wrong_object",
    ACTION_PLACE: "wrong_destination",
}
"""Upstream acts on the action token, not the status; so does the route."""

NO_VERDICT = "no_verdict"


class Output(TypedDict):
    route: str
    status: str
    action: str
    reason: str
    grasp_target: str
    place_destination: str
    place_held_object: str
    place_stack: bool
    cameras: list[str]
    raw: str


def _action_descriptions(actions: set[str], stack_mode: bool) -> str:
    lines = []
    if ACTION_NEXT in actions:
        lines.append(
            '- "next" — this subgoal is COMPLETE. The target object has '
            "moved to its destination. Advance to the next subgoal."
        )
    if ACTION_CONTINUE in actions:
        lines.append(
            '- "continue" — the robot is making progress or the subgoal '
            "is not yet complete. Keep working with the current instruction."
        )
    if ACTION_REPLAN in actions:
        lines.append(
            '- "replan" — something went wrong (object dropped, stuck, '
            "wrong object picked, no visible progress) OR the current "
            "plan is not working. Replan the remaining subtasks from "
            "the current scene state."
        )
    if ACTION_GRASP in actions:
        lines.append(
            '- "grasp_tool" — fire only for WRONG-OBJECT recovery: the '
            "gripper holds (or is closing on) an object whose identity "
            "differs from the subgoal's target. Do NOT fire while the "
            "robot is still attempting the correct object — that is "
            'in-progress execution; emit "continue". When firing, set '
            '"grasp_target" to the noun phrase of the correct object.'
        )
    if ACTION_PLACE in actions:
        stack = (
            "\n"
            'STACK CONTROL: place_tool accepts an optional boolean "place_stack" arg.\n'
            '  - "place_stack": false (DEFAULT) — plain top-down release. Use for '
            "ordinary pick-and-place: dropping an object into a bowl/bin, onto an "
            "open surface, or beside another object.\n"
            '  - "place_stack": true — keep the held object\'s current orientation '
            "at release instead of forcing top-down. Use ONLY when setting an object "
            "ON TOP of another in a way that preserves how it is held (stacking a "
            "block on a block, nesting a lid on a container). When in doubt, prefer false."
        )
        lines.append(
            '- "place_tool" — fire only for WRONG-DESTINATION recovery: '
            "the held object is heading to or already released at a "
            "destination different from the subgoal's destination. Do "
            "NOT fire while the robot is en route to the correct "
            'destination — that is in-progress execution; emit "continue". '
            'When firing, set "place_destination" to a FREEFORM destination '
            "phrase. PHRASING RULES:\n"
            '  1. Inside a container → "in <container>" (e.g. "in the white bowl").\n'
            '  2. Beside/near an anchor → prepend "empty space" '
            '(e.g. "empty space next to the green lemon").\n'
            '  3. Open table → "empty space on the table".\n'
            '  4. On a surface object → "on <surface>" (e.g. "on the wire rack shelf").\n'
            'Optionally also set "place_held_object" with the noun phrase of the '
            "object in the gripper.\n"
            "LIMITATION: place_tool is unreliable for high containers "
            "(e.g. a shelf, a high vase). For those destinations prefer "
            '"continue" or "replan" instead of firing place_tool.' + (stack if stack_mode else "")
        )
    return "\n".join(lines)


def system_prompt(actions: set[str], stack_mode: bool = False) -> str:
    """VoLoAgent's monitoring prompt; BEFORE is the scene before the current step."""
    return f"""\
You are monitoring a robot arm executing a tabletop manipulation task.

You receive two sets of images:
- **BEFORE** images: the scene before the current step started (for reference).
- **NOW** images: the current scene.

You also receive the overall task instruction, the current subgoal, and \
any remaining subgoals after the current one.

HOW TO REASON:
- Use the BEFORE images and overall task to understand the goal.
- Use the NOW images, current subgoal, and remaining subgoals to \
understand the current state and what still needs to happen.
- Decide whether the robot can keep executing the current subgoal \
from the NOW scene. If yes → "in_progress" / "continue". \
If the current subgoal is already done → "complete" / "next". \
If something actively prevents progress (wrong object in gripper, \
object dropped, robot stuck) → "failure".
- The subgoal list may have been replanned mid-episode. The current \
subgoal and remaining subgoals already reflect what needs to be \
done from the current state — do not treat them as stale.

Your job:
1. **Assess status** — look at the NOW images and determine the status \
of the CURRENT SUBGOAL.
2. **Decide action** — choose the best action for the robot.

STATUS:
- "complete" — the current subgoal is done (the target object is at \
its destination in the NOW images).
- "failure" — the robot is clearly stuck, holding the wrong object, \
or the target object is unreachable.
- "in_progress" — the subgoal is not yet complete but is still \
achievable. The robot may be approaching, reaching, or grasping.

ACTION (what to do?):
{_action_descriptions(actions, stack_mode)}

IMPORTANT RULES:
- Only say "complete" if you see CLEAR visual evidence the subgoal is \
done in the NOW images (target object at its destination).
- Say "in_progress" if the robot has not yet started or is still \
working on the current subgoal — this is NORMAL, not a failure.
- Only say "failure" if the robot is clearly stuck, has picked up the \
WRONG object, or has been making no progress for a long time.
- When in doubt, say "in_progress" with action "continue".
- Do NOT replan just because the scene looks different from BEFORE. \
Only replan if the current subgoal is impossible or the plan no \
longer makes sense given the NOW scene.

Output ONLY valid JSON (no markdown, no explanation):
{{"status": "...", "action": "...", "reason": "brief explanation"}}"""


def task_text(
    task: str, subgoal: str, remaining: list[str], index: int = 1, total: int | None = None
) -> str:
    """VoLoAgent's per-check user text."""
    total = total if total is not None else index + len(remaining)
    if remaining:
        remaining_str = "Remaining subgoals after this:\n" + "\n".join(
            f"  {i}. {s}" for i, s in enumerate(remaining, start=index + 1)
        )
    else:
        remaining_str = "This is the LAST subgoal."
    return (
        f'Overall task: "{task}"\n'
        f'Current subgoal ({index}/{total}): "{subgoal}"\n'
        f"{remaining_str}\n\n"
        f"Assess the current subgoal status and decide the action."
    )


def parse_json(text: str) -> dict[str, Any]:
    """VoLoAgent's tolerant JSON extraction: bare, fenced, or embedded in prose."""
    for candidate in (
        text,
        *(m.group(1) for m in [re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.S)] if m),
        *(m.group(0) for m in [re.search(r"\{.*\}", text, re.S)] if m),
    ):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    raise ValueError(f"cannot parse JSON from: {text[:200]}")


def _frames(observation: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    cams = (observation or {}).get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} if "name" not in v else v for k, v in cams.items()]
    return {str(c.get("name") or f"camera{i}"): c for i, c in enumerate(cams)}


def _rgb(frame: dict[str, Any]) -> np.ndarray | None:
    rgb = frame.get("rgb")
    if rgb is None:
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _is_wrist(name: str) -> bool:
    return any(k in name.lower() for k in ("wrist", "eye_in_hand", "hand"))


def _label(name: str) -> str:
    return f"{'Wrist' if _is_wrist(name) else 'External'} camera '{name}'"


def _verdict(route: str, **fields: Any) -> Output:
    out: Output = {
        "route": route,
        "status": "",
        "action": "",
        "reason": "",
        "grasp_target": "",
        "place_destination": "",
        "place_held_object": "",
        "place_stack": False,
        "cameras": [],
        "raw": "",
    }
    out.update(fields)  # type: ignore[typeddict-item]
    return out


def run(
    ctx: NodeContext,
    before: dict[str, Any],
    task: str,
    subgoal: str,
    remaining_subgoals: list[str] | str | None = None,
    subgoal_index: int = 1,
    subgoal_count: int | None = None,
    now: dict[str, Any] | None = None,
    cameras: list[str] | None = None,
    recovery_mode: str = DEFAULT_RECOVERY_MODE,
    stack_mode: bool = False,
    model: str | None = None,
) -> Output:
    """Ask the VLM whether ``subgoal`` is done, still running, or failed, BEFORE vs NOW.

    ``before`` is an observation taken before the step (a ``robot.get_observation``
    tool node); ``now`` defaults to a fresh ``robot.get_observation``. ``cameras``
    picks frames by name (default: every camera with RGB in both, external ones first).
    """
    if recovery_mode not in ACTION_SETS:
        raise ValueError(f"recovery_mode {recovery_mode!r} not in {sorted(ACTION_SETS)}")
    actions = ACTION_SETS[recovery_mode]
    if isinstance(remaining_subgoals, str):
        remaining = [remaining_subgoals] if remaining_subgoals.strip() else []
    else:
        remaining = [str(s) for s in remaining_subgoals or []]

    if now is None:
        now = ctx.tool("robot.get_observation")
    before_frames, now_frames = _frames(before), _frames(now)
    if cameras:
        names = [str(n) for n in cameras]
    else:
        names = [n for n in now_frames if n in before_frames]
        names.sort(key=_is_wrist)
    pairs = []
    for name in names:
        b = _rgb(before_frames[name]) if name in before_frames else None
        n = _rgb(now_frames[name]) if name in now_frames else None
        if b is not None and n is not None:
            pairs.append((name, b, n))
    if not pairs:
        return _verdict(
            NO_VERDICT,
            reason=f"no camera with RGB in both BEFORE and NOW (asked {names or 'all'}, "
            f"before has {sorted(before_frames)}, now has {sorted(now_frames)})",
        )

    images: list[np.ndarray] = []
    legend = []
    for when, which, scene in (
        ("BEFORE", 1, "scene before this step"),
        ("NOW", 2, "current scene"),
    ):
        for k, (name, b, n) in enumerate(pairs):
            images.append(b if which == 1 else n)
            note = f" ({scene})" if k == 0 else ""
            legend.append(f"Image {len(images)}: {when} — {_label(name)}{note}")
    prompt = (
        system_prompt(actions, stack_mode)
        + "\n\nThe images follow this text, in this order:\n"
        + "\n".join(legend)
        + "\n\n"
        + task_text(task, subgoal, remaining, subgoal_index, subgoal_count)
    )
    kwargs: dict[str, Any] = {"prompt": prompt, "images": images}
    if model:
        kwargs["model"] = model
    cams = [p[0] for p in pairs]
    try:
        raw = str(ctx.tool("vlm.query", **kwargs).get("text") or "")
    except Exception as exc:  # noqa: BLE001  (auth / network / unserved tool)
        logger.warning("judge_step: vlm.query failed: %s", exc)
        return _verdict(NO_VERDICT, reason=f"vlm.query failed: {exc}", cameras=cams)

    try:
        data = parse_json(raw)
    except ValueError:
        logger.warning("judge_step: unparseable reply: %s", raw[:200])
        return _verdict(NO_VERDICT, reason="parse_failure", cameras=cams, raw=raw)

    status = data.get("status", STATUS_IN_PROGRESS)
    if status not in (STATUS_COMPLETE, STATUS_FAILURE, STATUS_IN_PROGRESS):
        logger.warning("judge_step: invalid status %r, defaulting to in_progress", status)
        status = STATUS_IN_PROGRESS
    action = data.get("action", ACTION_CONTINUE)
    if action not in actions:
        logger.warning("judge_step: action %r not in %s, defaulting to continue", action, actions)
        action = ACTION_CONTINUE
    out = _verdict(
        ROUTES[action],
        status=status,
        action=action,
        reason=str(data.get("reason") or ""),
        grasp_target=str(data.get("grasp_target") or ""),
        place_destination=str(data.get("place_destination") or ""),
        place_held_object=str(data.get("place_held_object") or ""),
        place_stack=bool(data.get("place_stack", False)) if stack_mode else False,
        cameras=cams,
        raw=raw,
    )
    if action == ACTION_PLACE and not out["place_destination"]:
        target = str(data.get("place_target") or "").strip()
        if target:
            relation = str(data.get("place_relation") or "in").strip()
            out["place_destination"] = f"{relation} the {target}"
    logger.info(
        "judge_step: status=%s action=%s route=%s — %s",
        status,
        action,
        out["route"],
        out["reason"],
    )
    return out
