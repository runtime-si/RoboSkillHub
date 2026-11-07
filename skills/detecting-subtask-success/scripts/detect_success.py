# Adapted from the Gemini Robotics SDK (https://github.com/google-deepmind/gemini-robotics-sdk),
# safari_sdk/agent/framework/tools/success_detection.py (SubtaskSuccessDetectorV4,
# _GUIDED_THINKING_SD_QUESTIONS) at 26b49f1a15b96f1e65b05160f5b991d6167aba74, Apache-2.0,
# Copyright 2025 Google LLC.
# Changes: the event-bus image buffer and the asyncio background queries become a
# synchronous poll of robot.get_observation inside run(); the stale-image guard
# becomes "do not ask again about frames identical to the last judged ones"; the
# prompt goes through the hub's vlm.query, which appends the images after the text,
# so each image is announced by number and camera in the text instead of being
# interleaved; start images are optional (upstream sent them by default); a strict
# reading can replace upstream's lenient one; the FINAL ANSWER parse is
# case-insensitive and also reports "no" and unparsed replies; a reply with no FINAL
# ANSWER gets one follow-up asking for that line only; the external success
# signal and cancellation events are dropped (a graph stops a node by not running it).
"""Ask a VLM whether a subtask is done, from every camera's latest frame, with guided questions."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

SUCCEEDED = "succeeded"
NOT_YET = "not_yet"
TIMED_OUT = "timed_out"
NO_VIEW = "no_view"

GUIDED_QUESTIONS = """
  Q1 Tell me about the state of task relevent objects at each time by combining all views to form a complete picture.
  Q2 What are the relationships between the relevant objects at each time?
  Q3 What state should the objects be in if the robot successfully completed the task?
  Q4 Are the objects currently in the state consistent with the state where the robot successfully completed at the current time? What are the differences? Tell me what views and times do you base your answer on first.

  Example answers for the questions above for the robot task: put the spoon in the compost bin:
  Q1:
    1) The spoon is on the table now and at start, based on left and right wrist and overhead cameras.
    2) The compost bin is empty with the lid open now and at start, based on the overhead cameras.
  Q2:
    1) the spoon is outside of the compost bin now and at start
  Q3:
    1) The spoon should be inside the compost bin. The overhead camera should show the spoon in the compost bin.
  Q4:
    1) No, the spoon is on the table instead of in the compost bin based on the overhead camera and answers to Q1.

  After you answered these questions, tell me if the robot successfully completed the task.
{leniency}
  If the task specifies a quantity (e.g., one, a, two), the minimum requirement is met as soon as at least that number of items is in the correct final state.
  The state of any other similar items that were not part of the task does not matter.

  If you think the robot finished the task, answer: 'FINAL ANSWER: yes'
  If you think the roboy did not finish the task, answer: 'FINAL ANSWER: no'.
  Do not mention the word 'FINAL ANSWER' anywhere else."""
"""Upstream's _GUIDED_THINKING_SD_QUESTIONS verbatim (typos included), with its leniency line as a slot."""

LENIENT = (
    "  Use the most linient way to intrepret task completion. If you think the robot partially "
    "completed or it may have completed the task (cannot be determined based on the current views), "
    "count it as completed."
)
"""Upstream's line: partial or undeterminable counts as done."""

STRICT = (
    "  Use a strict way to interpret task completion. Count the task as completed only if the current "
    "views clearly show the required end state. If the robot only partially completed it, or it cannot "
    "be determined based on the current views, count it as not completed."
)
"""The opposite reading, for a gate where a false yes costs more than another check."""

WORD_LIMIT = 50
"""Upstream sd_guided_thinking_word_limit default."""

INTERVAL_S = 0.2
"""Upstream sd_async_sd_interval_s: the wait between checks (upstream launched a query every 0.2 s)."""

FOLLOW_UP = (
    "\n\nYour reply to this was cut off before its final line:\n{reply}\n\n"
    "Do not repeat the answers. Reply with only the final line: 'FINAL ANSWER: yes' or 'FINAL ANSWER: no'."
)
"""Asked once, with the same images, when a reply has no FINAL ANSWER (the hub's vlm.query caps output at
1024 tokens, thinking included, and Gemini 3 replies to the guided questions got cut there)."""

_FINAL = re.compile(r"final\s+answer\W{0,6}(yes|no)\b", re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{\s*(yes|no)\s*\}", re.IGNORECASE)


class Output(TypedDict):
    route: str
    success: bool
    answer: str
    checks: int
    follow_ups: int
    errors: int
    elapsed_s: float
    cameras: list[str]
    reason: str
    raw: str


def parse_answer(text: str) -> str:
    """'yes' / 'no' from the last FINAL ANSWER (or upstream's \\boxed{} form), '' when there is none."""
    hits = [m for rx in (_FINAL, _BOXED) for m in rx.finditer(text or "")]
    if not hits:
        return ""
    return max(hits, key=lambda m: m.start()).group(1).lower()


def _frames(observation: dict[str, Any] | None) -> list[tuple[str, dict[str, Any]]]:
    cams = (observation or {}).get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} if "name" not in v else v for k, v in cams.items()]
    return [(str(c.get("name") or f"camera{i}"), c) for i, c in enumerate(cams)]


def _rgb(frame: dict[str, Any]) -> np.ndarray | None:
    rgb = frame.get("rgb")
    if rgb is None:
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
        return None
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def views(
    observation: dict[str, Any] | None, cameras: list[str] | None
) -> list[tuple[str, np.ndarray]]:
    """(camera name, RGB) for every camera with RGB, in ``cameras`` order when given."""
    frames = _frames(observation)
    if cameras:
        by_name = dict(frames)
        frames = [(str(n), by_name[str(n)]) for n in cameras if str(n) in by_name]
    out = []
    for name, frame in frames:
        rgb = _rgb(frame)
        if rgb is not None:
            out.append((name, rgb))
    return out


def build_prompt(
    subtask: str,
    now: list[tuple[str, np.ndarray]],
    start: list[tuple[str, np.ndarray]] | None = None,
    lenient: bool = True,
    word_limit: int | None = WORD_LIMIT,
) -> tuple[str, list[np.ndarray]]:
    """Upstream's _build_prompt (explicit thinking) with the images listed by number, start ones first."""
    lines: list[str] = []
    images: list[np.ndarray] = []
    if start:
        lines.append(
            "The images below show what the robot saw at the start of the task"
            " (the robot may have more than one camera):"
        )
        for name, rgb in start:
            images.append(rgb)
            lines.append(f"Image {len(images)}: {name.replace('/', '').strip()}")
    lines.append(
        "The images below show what the robot sees now (the robot may have more than one camera):"
    )
    for name, rgb in now:
        images.append(rgb)
        lines.append(f"Image {len(images)}: {name.replace('/', '').strip()}")
    lines.append(
        "(The images are attached after this text, in the numbered order above.)\n"
        f"The robot's task is: {subtask}\n"
        "Given the information above, answer the following questions first:"
    )
    lines.append(GUIDED_QUESTIONS.format(leniency=LENIENT if lenient else STRICT))
    if word_limit is not None:
        lines.append(f"Please limit your response to {word_limit} words.")
    return "\n".join(lines), images


def _fingerprint(observation: dict[str, Any], frames: list[tuple[str, np.ndarray]]) -> str:
    h = hashlib.sha1()
    for key in ("timestamp", "stamp", "time"):
        if key in (observation or {}):
            h.update(f"{key}={observation[key]}".encode())
    for name, rgb in frames:
        h.update(name.encode())
        h.update(str(rgb.shape).encode())
        h.update(rgb.tobytes())
    return h.hexdigest()


def run(
    ctx: NodeContext,
    subtask: str,
    now: dict[str, Any] | None = None,
    start: dict[str, Any] | None = None,
    cameras: list[str] | None = None,
    timeout_s: float = 0.0,
    interval_s: float = INTERVAL_S,
    lenient: bool = True,
    word_limit: int | None = WORD_LIMIT,
    follow_up: bool = True,
    model: str | None = None,
) -> Output:
    """Ask whether ``subtask`` is done in the latest frame of every camera, until yes or ``timeout_s``.

    ``timeout_s`` 0 asks once (``now`` if given, else a fresh ``robot.get_observation``) and routes
    ``succeeded`` / ``not_yet``. Above 0 it polls fresh observations every ``interval_s`` and routes
    ``succeeded`` on the first yes, ``timed_out`` at the deadline. ``start`` adds upstream's
    start-of-task images in front of the current ones. ``follow_up`` re-asks once for the final line
    when a reply has none; without it (or when the follow-up has none either) the reply reads as no.
    """
    t0 = time.monotonic()
    deadline = t0 + max(float(timeout_s), 0.0)
    start_views = views(start, cameras) if start is not None else []
    checks = errors = follow_ups = 0
    last_print = ""
    last: dict[str, Any] = {"answer": "", "raw": "", "cameras": [], "error": ""}
    seen_cameras: list[str] = []
    first = True

    while True:
        observation = now if (first and now is not None) else ctx.tool("robot.get_observation")
        first = False
        current = views(observation, cameras)
        if current:
            seen_cameras = [n for n, _ in current]
            stamp = _fingerprint(observation, current)
            if stamp != last_print:
                last_print = stamp
                prompt, images = build_prompt(subtask, current, start_views, lenient, word_limit)
                kwargs: dict[str, Any] = {"prompt": prompt, "images": images}
                if model:
                    kwargs["model"] = model
                try:
                    raw = str(ctx.tool("vlm.query", **kwargs).get("text") or "")
                    answer = parse_answer(raw)
                    if not answer and follow_up:
                        follow_ups += 1
                        kwargs["prompt"] = prompt + FOLLOW_UP.format(reply=raw.strip() or "(empty)")
                        tail = str(ctx.tool("vlm.query", **kwargs).get("text") or "")
                        raw = f"{raw}\n[follow-up] {tail}"
                        answer = parse_answer(tail)
                except Exception as exc:  # noqa: BLE001  (auth / network / unserved tool)
                    errors += 1
                    last_print = ""
                    last["error"] = str(exc)
                    logger.warning("detect_success: vlm.query failed: %s", exc)
                else:
                    checks += 1
                    last = {"answer": answer, "raw": raw, "cameras": seen_cameras, "error": ""}
                    logger.info("detect_success: %r check %d -> %s", subtask, checks, answer or "?")
                    if answer == "yes":
                        return _out(
                            SUCCEEDED, t0, checks, follow_ups, errors, last, "FINAL ANSWER: yes"
                        )
        if time.monotonic() >= deadline:
            break
        time.sleep(max(float(interval_s), 0.0))

    if checks == 0:
        if errors:
            why = f"every vlm.query failed ({errors}x), last: {last['error']}"
        else:
            names = [n for n, _ in _frames(observation)]
            why = f"no camera with RGB (asked {cameras or 'all'}, observation has {names})"
        return _out(NO_VIEW, t0, 0, follow_ups, errors, {**last, "cameras": seen_cameras}, why)
    why = (
        "FINAL ANSWER: no"
        if last["answer"] == "no"
        else "no FINAL ANSWER in the reply (read as no)"
    )
    if timeout_s > 0:
        return _out(
            TIMED_OUT,
            t0,
            checks,
            follow_ups,
            errors,
            last,
            f"no yes in {checks} checks over {timeout_s:g} s; last: {why}",
        )
    return _out(NOT_YET, t0, checks, follow_ups, errors, last, why)


def _out(
    route: str,
    t0: float,
    checks: int,
    follow_ups: int,
    errors: int,
    last: dict[str, Any],
    reason: str,
) -> Output:
    return {
        "route": route,
        "success": route == SUCCEEDED,
        "answer": last.get("answer", ""),
        "checks": checks,
        "follow_ups": follow_ups,
        "errors": errors,
        "elapsed_s": round(time.monotonic() - t0, 3),
        "cameras": list(last.get("cameras") or []),
        "reason": reason,
        "raw": last.get("raw", ""),
    }
