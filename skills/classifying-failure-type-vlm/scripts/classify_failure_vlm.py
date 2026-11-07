# Adapted from RoboFAC (https://github.com/MINT-SJTU/RoboFAC),
# evaluation/eval.py and evaluation/eval_utils.py, MIT License,
# Copyright (c) 2025 MINT-SJTU; the question texts and the failure-type options
# are RoboFAC's test QA (MINT-SJTU/RoboFAC-dataset, test_qa_sim, MIT).
# Changes: RoboFAC asks each question in its own request about a whole video
# (one key frame per second, 512 x 512); here one vlm.query asks failure
# detection, failure identification, failure explanation and the high- and
# low-level corrections together about BEFORE / DURING / AFTER frames of one
# step, with the task and the step in the prompt, and wants JSON back. A
# routing class (missed grasp, slip, wrong object, misalignment, collision or
# stuck, incomplete) is asked beside RoboFAC's six types and becomes the exit;
# an unparseable reply or a failed call routes no_verdict.
"""Classify why a step failed from camera frames around it, with a VLM, into one exit per failure type."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

ROBOFAC_TYPES = (
    "Orientation deviation.",
    "Step omission.",
    "Wrong target object.",
    "Timing error.",
    "Grasping error.",
    "Position deviation.",
)
"""RoboFAC's failure-identification options, verbatim and in its order."""

CLASSES: dict[str, str] = {
    "missed_grasp": (
        "the gripper closed but the object is not in it: the hand missed the object, closed beside "
        "it or on air, or pushed it away while closing (RoboFAC: Grasping error or Position "
        "deviation at the grasp)"
    ),
    "slipped": (
        "the object WAS held at some point but is no longer in the hand: it slipped out or fell "
        "while lifting or carrying and now lies somewhere it was not put (RoboFAC: Grasping error "
        "or Timing error after the grasp)"
    ),
    "wrong_object": (
        "the robot grasped, moved or is holding an object other than the one the task names "
        "(RoboFAC: Wrong target object)"
    ),
    "misaligned": (
        "the right object was handled and the step was carried out, but the object or the hand "
        "ended offset or turned relative to its target: placed beside the container or target, "
        "tilted, not lined up with a slot (RoboFAC: Position deviation or Orientation deviation)"
    ),
    "collision_or_stuck": (
        "the hand or the object ran into something - a wall, a fixture, another object, the "
        "table - and stopped or jammed before the step could finish"
    ),
    "incomplete": (
        "the step stopped short or a part of it was never done: no collision and no wrong "
        "object, it simply did not get far enough (RoboFAC: Step omission or Timing error)"
    ),
}
"""The routing taxonomy, one exit each; the text is what the VLM is told."""

NO_FAILURE = "no_failure"
NO_VERDICT = "no_verdict"

FALLBACK: dict[str, str] = {
    "Wrong target object.": "wrong_object",
    "Position deviation.": "misaligned",
    "Orientation deviation.": "misaligned",
    "Step omission.": "incomplete",
}
"""Route from RoboFAC's type when the reply names no valid class; Grasping error and Timing
error are left out because they split between missed_grasp and slipped."""

MISALIGNMENT = {"Position deviation.": "translation", "Orientation deviation.": "rotation"}

MAX_DURING = 4
"""DURING frames kept per camera, evenly spaced. RoboFAC sent one key frame per second of a
clip; with BEFORE and AFTER this is 6 frames per camera, a few seconds of a step."""

DETECTION_Q = (
    "In the video, the robotic arm executes a task, can you determine whether it was successful?"
)
IDENTIFICATION_Q = (
    "This is a video of a robotic arm performing a task, please identify the type of error that "
    "occurred during execution. (Your answer should choose one of the following options: "
    f"{list(ROBOFAC_TYPES)})"
)
EXPLANATION_Q = (
    "Based on the video, provide a detailed explanation of why the robotic arm failed to complete "
    "the task. (Your answer should be a detailed textual description of about two to three "
    "sentences.)"
)
HIGH_LEVEL_Q = (
    "Based on the video showing an error during the robotic arm 's execution of a task, give "
    "detailed high-level guidance for correcting the error and enabling task completion. (Your "
    "answer should be a detailed textual suggestion of about two to three sentences.)"
)
LOW_LEVEL_Q = (
    "According to the video of the robotic arm executing a task, please suggest specific "
    "low-level recovery actions to enable successful task completion. (Your answer should "
    "describe which direction (relative to the robot arm) and how much the robot arm should move "
    "to recover from the failure.)"
)
"""RoboFAC's questions as its eval.py sends them to models other than RoboFAC-7B (with the
answer-format suffix)."""


class Output(TypedDict):
    route: str
    failed: bool
    failure_class: str
    robofac_type: str
    misalignment: str
    explanation: str
    correction: str
    low_level_correction: str
    observations: list[str]
    cameras: list[str]
    frames: int
    raw: str


def build_prompt(task: str, step: str, legend: list[str]) -> str:
    classes = "\n".join(f'- "{name}": {text}' for name, text in CLASSES.items())
    return f"""\
You are analysing a robot manipulation step that may have failed.

Overall task: "{task}"
The step the robot just executed: "{step}"

The images follow this text, in this order (BEFORE is the scene just before the step,
DURING frames are in time order, AFTER is the scene now; together they are the video):
{chr(10).join(legend)}

Answer these questions about the step:
1. {DETECTION_Q} Judge the STEP above: it succeeded only if AFTER shows its intended result.
2. If it failed: {IDENTIFICATION_Q}
3. If it failed, which of these classes describes what went wrong (pick exactly one):
{classes}
4. If it failed: {EXPLANATION_Q}
5. If it failed: {HIGH_LEVEL_Q}
6. If it failed: {LOW_LEVEL_Q}

Before answering, find the hand and every object the task names in each image. If an object
is no longer where it was in BEFORE, find where it went: in the closed gripper, elsewhere on
the table, inside a container. A small object held by the gripper can be mostly hidden by the
fingers: an object that left its BEFORE spot, is nowhere else on the table, and the raised
closed gripper is where it went, is held. A step whose object ended where the step says
succeeded, even if the hand still covers part of it.

Output ONLY valid JSON (no markdown), keys in this order; "observations" is one short line
per image saying where the hand and the task's objects are:
{{"observations": ["Image 1: ...", ...], "success": true or false,
"robofac_type": "<option from question 2, or empty>",
"failure_class": "<class from question 3, or empty>", "explanation": "...",
"correction": "...", "low_level_correction": "..."}}"""


def parse_json(text: str) -> dict[str, Any]:
    """Bare, fenced or embedded JSON object."""
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
    salvaged = _salvage(text)
    if "success" in salvaged:
        return salvaged
    raise ValueError(f"cannot parse JSON from: {text[:200]}")


def _salvage(text: str) -> dict[str, Any]:
    """The complete scalar fields of a reply cut off mid-JSON (a thinking model's output cap)."""
    out: dict[str, Any] = {}
    for m in re.finditer(r'"(\w+)"\s*:\s*("(?:[^"\\]|\\.)*"|true|false)', text):
        key, value = m.group(1), m.group(2)
        out.setdefault(key, json.loads(value))
    return out


def _frames(observation: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    cams = (observation or {}).get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} if "name" not in v else v for k, v in cams.items()]
    return {str(c.get("name") or f"camera{i}"): c for i, c in enumerate(cams)}


def _rgb(frame: dict[str, Any] | None) -> np.ndarray | None:
    rgb = (frame or {}).get("rgb")
    if rgb is None:
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _crop(img: np.ndarray, box: Any) -> np.ndarray:
    if not box:
        return img
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return img
    return np.ascontiguousarray(img[y1:y2, x1:x2])


def _is_wrist(name: str) -> bool:
    return any(k in name.lower() for k in ("wrist", "eye_in_hand", "hand"))


def _subsample(items: list[Any], k: int) -> list[Any]:
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def _boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower().rstrip(".")
        if v in ("true", "yes", "success", "successful"):
            return True
        if v in ("false", "no", "failure", "failed"):
            return False
    return None


def _robofac_type(value: Any) -> str:
    text = str(value or "").strip().lower().rstrip(".")
    for option in ROBOFAC_TYPES:
        if text == option.lower().rstrip("."):
            return option
    return ""


def _verdict(route: str, **fields: Any) -> Output:
    out: Output = {
        "route": route,
        "failed": route not in (NO_FAILURE, NO_VERDICT),
        "failure_class": "",
        "robofac_type": "",
        "misalignment": "",
        "explanation": "",
        "correction": "",
        "low_level_correction": "",
        "observations": [],
        "cameras": [],
        "frames": 0,
        "raw": "",
    }
    out.update(fields)  # type: ignore[typeddict-item]
    return out


def run(
    ctx: NodeContext,
    before: dict[str, Any],
    task: str,
    step: str,
    after: dict[str, Any] | None = None,
    during: list[dict[str, Any]] | None = None,
    cameras: list[str] | None = None,
    max_during: int = MAX_DURING,
    crop: list[float] | dict[str, list[float]] | None = None,
    model: str | None = None,
) -> Output:
    """Ask the VLM whether ``step`` failed and, if so, how, from BEFORE / DURING / AFTER frames.

    ``before`` is an observation taken before the step (a ``robot.get_observation`` tool node);
    ``after`` defaults to a fresh ``robot.get_observation``; ``during`` is an optional list of
    observations taken while the step ran (a short video as frames), thinned to ``max_during``.
    ``cameras`` picks frames by name (default: every camera with RGB in BEFORE and AFTER,
    external ones first). ``crop`` is a pixel box ``[x1, y1, x2, y2]`` cut from every frame,
    or a mapping camera name -> box.
    """
    if after is None:
        after = ctx.tool("robot.get_observation")
    before_frames, after_frames = _frames(before), _frames(after)
    during_frames = [_frames(o) for o in _subsample(list(during or []), int(max_during))]
    if cameras:
        names = [str(n) for n in cameras]
    else:
        names = sorted((n for n in after_frames if n in before_frames), key=_is_wrist)

    images: list[np.ndarray] = []
    legend: list[str] = []
    used: list[str] = []
    for name in names:
        b, a = _rgb(before_frames.get(name)), _rgb(after_frames.get(name))
        if b is None or a is None:
            continue
        used.append(name)
        kind = "wrist" if _is_wrist(name) else "external"
        strip = [("BEFORE", b)]
        mids = [m for m in (_rgb(f.get(name)) for f in during_frames) if m is not None]
        strip += [(f"DURING {i}/{len(mids)}", m) for i, m in enumerate(mids, start=1)]
        strip.append(("AFTER", a))
        box = crop.get(name) if isinstance(crop, dict) else crop
        for when, img in strip:
            images.append(_crop(img, box))
            legend.append(f"Image {len(images)}: {when} - {kind} camera '{name}'")
    if not images:
        return _verdict(
            NO_VERDICT,
            explanation=f"no camera with RGB in both BEFORE and AFTER (asked {names or 'all'}, "
            f"before has {sorted(before_frames)}, after has {sorted(after_frames)})",
        )

    kwargs: dict[str, Any] = {"prompt": build_prompt(task, step, legend), "images": images}
    if model:
        kwargs["model"] = model
    seen = {"cameras": used, "frames": len(images)}
    try:
        raw = str(ctx.tool("vlm.query", **kwargs).get("text") or "")
    except Exception as exc:  # noqa: BLE001  (auth / network / unserved tool)
        logger.warning("classifying-failure-type-vlm: vlm.query failed: %s", exc)
        return _verdict(NO_VERDICT, explanation=f"vlm.query failed: {exc}", **seen)
    try:
        data = parse_json(raw)
    except ValueError:
        logger.warning("classifying-failure-type-vlm: unparseable reply: %s", raw[:200])
        return _verdict(NO_VERDICT, explanation="parse_failure", raw=raw, **seen)

    success = _boolish(data.get("success"))
    robofac = _robofac_type(data.get("robofac_type"))
    cls = str(data.get("failure_class") or "").strip().lower()
    observations = data.get("observations")
    text = {
        "observations": [str(o) for o in observations] if isinstance(observations, list) else [],
        "explanation": str(data.get("explanation") or ""),
        "correction": str(data.get("correction") or ""),
        "low_level_correction": str(data.get("low_level_correction") or ""),
    }
    if success is True:
        return _verdict(NO_FAILURE, raw=raw, **text, **seen)
    if cls not in CLASSES:
        cls = FALLBACK.get(robofac, "")
    if success is None or not cls:
        why = "no success verdict" if success is None else "no failure class"
        text["explanation"] = f"{why}: {text['explanation']}".rstrip(": ")
        return _verdict(NO_VERDICT, robofac_type=robofac, raw=raw, **text, **seen)
    out = _verdict(
        cls,
        failure_class=cls,
        robofac_type=robofac,
        misalignment=MISALIGNMENT.get(robofac, "") if cls == "misaligned" else "",
        raw=raw,
        **text,
        **seen,
    )
    logger.info(
        "classifying-failure-type-vlm: %s (%s) - %s", cls, robofac or "-", out["explanation"]
    )
    return out
