"""Score how far a step has got from one camera frame (DenseReward) and route done / in progress / regressed.

The model is the hub's ``densereward`` tool (DenseReward-1frame, or -3frame-thinking
when history frames are given; Apache-2.0, see tools/densereward). This script
only picks the frame, calls the tool and turns its 0..1 progress into an exit.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

DONE_THRESHOLD = 0.85
"""Median progress at or above which the step counts as done. Real station
frames (SKILL.md, Measured): cube-in-box frames scored 0.87-0.93 when the model
recognised them (5 of 7), and with the 3-sample median 1 of 91 not-done frames
reached 0.85 (0.94, mid-reach); at 0.9 no done frame's median passed."""

SAMPLES = 3
"""Frames scored per call (the given or a fresh observation, then fresh ones); the
median is the progress. One real episode: consecutive camera frames 1/15 s apart
moved the score by more than 0.1 on 8 of 33 frames, and the median of three
removed 2 of the 3 false 'done' scores of single frames."""

MAX_SPREAD = 0.3
"""More disagreement (max - min) than this between the samples routes
``low_confidence``. On the real episode the 8 unstable frames spread 0.40-0.61
and the other 25 at most 0.10; it caught the remaining false 'done'."""

REGRESS_DROP = 0.3
"""A fall below ``previous_progress`` larger than this routes ``regressed``. The
score is not smooth (successive 2 s frames of one successful real episode fell
by 0.3-0.5 three times while the arm moved): compare scores taken at rest."""

FAILURE_REASONS = ("miss", "collision", "fall", "smooth", "failure")
"""The 3-frame model's words for a drop it saw (its system prompt's vocabulary)."""


class Output(TypedDict):
    route: str
    progress: float
    scores: list[float]
    reason: str
    camera: str
    model: str
    raw: str


def _cameras(observation: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
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


def _is_wrist(name: str) -> bool:
    return any(k in name.lower() for k in ("wrist", "eye_in_hand", "hand"))


def _pick_camera(frames: dict[str, dict[str, Any]], camera: str | None) -> str | None:
    if camera:
        return camera if _rgb(frames.get(camera)) is not None else None
    names = sorted((n for n in frames if _rgb(frames[n]) is not None), key=_is_wrist)
    return names[0] if names else None


def _out(route: str, **fields: Any) -> Output:
    out: Output = {
        "route": route,
        "progress": 0.0,
        "scores": [],
        "reason": "",
        "camera": "",
        "model": "",
        "raw": "",
    }
    out.update(fields)  # type: ignore[typeddict-item]
    return out


def run(
    ctx: NodeContext,
    task: str,
    observation: dict[str, Any] | None = None,
    camera: str | None = None,
    history: list[dict[str, Any]] | None = None,
    previous_progress: float | None = None,
    samples: int = SAMPLES,
    done_threshold: float = DONE_THRESHOLD,
    max_spread: float = MAX_SPREAD,
    regress_drop: float = REGRESS_DROP,
) -> Output:
    """Score ``task``'s progress in one frame and route on it.

    ``observation`` defaults to a fresh ``robot.get_observation``; ``camera``
    picks its frame by name (default: the first external camera, else a wrist
    one). ``samples`` frames are scored (the first from ``observation``, the
    rest from fresh observations) and the median is the progress; a spread
    over ``max_spread`` routes ``low_confidence``. With ``history`` (earlier
    observations, oldest first) the last two frames of the same camera and
    the current one go once to ``densereward.progress_with_history``, which
    also names a failure it saw. ``previous_progress`` (this skill's last
    ``progress`` for the same step, taken at rest) enables ``regressed`` by drop.
    """
    if not task or not str(task).strip():
        raise ValueError("scoring-step-progress-from-one-frame needs a task")
    if observation is None:
        observation = ctx.tool("robot.get_observation")
    frames = _cameras(observation)
    name = _pick_camera(frames, camera)
    if name is None:
        return _out(
            "low_confidence",
            reason=f"no RGB frame (asked {camera or 'any'}, observation has {sorted(frames)})",
        )
    now = _rgb(frames[name])
    past = [_rgb(_cameras(h).get(name)) for h in (history or [])]
    past = [p for p in past if p is not None][-2:]

    task = str(task)
    try:
        if past:
            results = [
                ctx.tool("densereward.progress_with_history", images=[*past, now], task=task)
            ]
        else:
            results = [ctx.tool("densereward.progress", image=now, task=task)]
            for _ in range(max(1, int(samples)) - 1):
                extra = _rgb(_cameras(ctx.tool("robot.get_observation")).get(name))
                if extra is not None:
                    results.append(ctx.tool("densereward.progress", image=extra, task=task))
    except Exception as exc:  # noqa: BLE001  (unserved tool, model load, CUDA, camera)
        logger.warning("score_progress: densereward failed: %s", exc)
        return _out("low_confidence", reason=f"densereward failed: {exc}", camera=name)

    fields: dict[str, Any] = {
        "camera": name,
        "model": str(results[0].get("model") or ""),
        "raw": " | ".join(str(r.get("raw") or "") for r in results),
    }
    scores = [float(r.get("progress") or 0.0) for r in results if r.get("found")]
    if not scores:
        return _out("low_confidence", reason="no progress value in the model's answer", **fields)
    progress = float(np.median(scores))
    word = str(results[0].get("reason") or "")
    fields["progress"] = round(progress, 3)
    fields["scores"] = [round(v, 3) for v in scores]
    spread = max(scores) - min(scores)
    if spread > max_spread:
        return _out(
            "low_confidence",
            reason=f"samples disagree: {fields['scores']} (spread {spread:.3f} > {max_spread})",
            **fields,
        )

    if word in FAILURE_REASONS:
        route, why = "regressed", f"the model saw '{word}'"
    elif previous_progress is not None and progress < float(previous_progress) - regress_drop:
        route = "regressed"
        why = f"progress fell {float(previous_progress):.3f} -> {progress:.3f}"
    elif progress >= done_threshold:
        route, why = "done", f"progress {progress:.3f} >= {done_threshold}"
    else:
        route, why = "in_progress", f"progress {progress:.3f} < {done_threshold}"
    if word:
        why += f" ({word})"
    logger.info("score_progress: %s on %s: %s", route, name, why)
    return _out(route, reason=why, **fields)
