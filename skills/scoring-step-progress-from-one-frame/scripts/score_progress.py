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
"""Progress at or above which the step counts as done. On real station frames
(SKILL.md, Measured) the finished pick-place scored 0.87-0.94 and no frame of an
unfinished or failed one scored above 0.80."""

REGRESS_DROP = 0.15
"""A fall in progress below ``previous_progress`` larger than this routes
``regressed``. Upstream trains a drop on irreversible failures (dropped, missed,
wrong target); successive frames of one real episode moved by up to 0.1 without one."""

FAILURE_REASONS = ("miss", "collision", "fall", "smooth", "failure")
"""The 3-frame model's words for a drop it saw (its system prompt's vocabulary)."""


class Output(TypedDict):
    route: str
    progress: float
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
    out: Output = {"route": route, "progress": 0.0, "reason": "", "camera": "", "model": "", "raw": ""}
    out.update(fields)  # type: ignore[typeddict-item]
    return out


def run(
    ctx: NodeContext,
    task: str,
    observation: dict[str, Any] | None = None,
    camera: str | None = None,
    history: list[dict[str, Any]] | None = None,
    previous_progress: float | None = None,
    done_threshold: float = DONE_THRESHOLD,
    regress_drop: float = REGRESS_DROP,
) -> Output:
    """Score ``task``'s progress in one frame and route on it.

    ``observation`` defaults to a fresh ``robot.get_observation``; ``camera``
    picks its frame by name (default: the first external camera, else a wrist
    one). With ``history`` (earlier observations, oldest first) the last two
    frames of the same camera go to ``densereward.progress_with_history``,
    which also names a failure it saw. ``previous_progress`` (this skill's
    last ``progress`` for the same step) enables the ``regressed`` exit by
    drop.
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

    try:
        if past:
            result = ctx.tool(
                "densereward.progress_with_history", images=[*past, now], task=str(task)
            )
        else:
            result = ctx.tool("densereward.progress", image=now, task=str(task))
    except Exception as exc:  # noqa: BLE001  (unserved tool, model load, CUDA)
        logger.warning("score_progress: densereward failed: %s", exc)
        return _out("low_confidence", reason=f"densereward failed: {exc}", camera=name)

    fields = {
        "camera": name,
        "model": str(result.get("model") or ""),
        "raw": str(result.get("raw") or ""),
    }
    if not result.get("found"):
        return _out("low_confidence", reason="no progress value in the model's answer", **fields)
    progress = float(result.get("progress") or 0.0)
    word = str(result.get("reason") or "")
    fields["progress"] = round(progress, 3)

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
