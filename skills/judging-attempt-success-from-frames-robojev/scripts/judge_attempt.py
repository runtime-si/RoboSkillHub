# The protocol is RoboJev's (github.com/josefchen/robojev at 79a5e2f, robojev/robojev.py, Apache-2.0):
# ten main-camera frames from the start to the end of the attempt plus the final frame of each other
# camera, one calibrated P(done) - asked through tools/robojev, which adapts that client. New here:
# where the frames come from in a graph (a clip the caller recorded, or the observation captured
# before the step and the one now), which cameras are main and extra, and the three-way decision
# with a band in which the answer is not trusted.
"""Did the step succeed? One calibrated P(done) from the attempt's frames (RoboJev-27B), and a route."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

SUCCESS_AT = 0.7
"""P(done) at or above which the attempt counts as a success. The model card's balanced accuracy
uses 0.5; its calibration was fitted on SO-101 runs and its ECE on unseen robots is 0.148, so the
band 0.3-0.7 is left `uncertain` until it is refitted on labelled YAM steps."""

FAILURE_AT = 0.3
"""P(done) at or below which the attempt counts as a failure."""

MIN_FRAMES = 2
"""Fewest main-camera frames: a start and an end (RoboJev repeats frames to make ten, as its
client does for short clips)."""

ROUTES = ("success", "failure", "uncertain", "no_frames")


class Output(TypedDict):
    route: str
    detail: str
    p_done: float
    main_camera: str
    views: list[str]
    n_frames: int


def _is_wrist(name: str) -> bool:
    return any(k in name for k in ("wrist", "eye_in_hand"))


def _rgb(frame: Any) -> np.ndarray | None:
    if frame is None:
        return None
    img = frame.get("rgb") if isinstance(frame, dict) else frame
    if img is None:
        return None
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        return None
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(a * (255.0 if float(a.max()) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def _cams(obs: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not obs:
        return {}
    return {
        str(c.get("name", i)): c
        for i, c in enumerate(obs.get("cameras") or [])
        if c.get("rgb") is not None
    }


def run(
    ctx: NodeContext,
    task: str,
    frames: list[Any] | None = None,
    before: dict[str, Any] | None = None,
    camera: str | None = None,
    views: list[str] | None = None,
    use_views: bool = True,
    success_at: float = SUCCESS_AT,
    failure_at: float = FAILURE_AT,
) -> Output:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task (the instruction the step carried out, one sentence) is required")
    if not 0.0 <= failure_at <= success_at <= 1.0:
        raise ValueError(f"need 0 <= failure_at <= success_at <= 1, got {failure_at}, {success_at}")
    out: Output = {
        "route": "no_frames",
        "detail": "",
        "p_done": float("nan"),
        "main_camera": "",
        "views": [],
        "n_frames": 0,
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        logger.info("judging-attempt-success: %s (%s)", route, why)
        return out

    now = _cams(ctx.tool("robot.get_observation"))
    start = _cams(before)
    if camera is None:
        names = [n for n in now if not _is_wrist(n)] + [n for n in now if _is_wrist(n)]
        names += [n for n in start if n not in names]
        camera = next((n for n in names if not _is_wrist(n)), names[0] if names else None)
    if camera is None:
        return finish("no_frames", "no camera with RGB in the observation")
    out["main_camera"] = camera

    clip: list[np.ndarray] = []
    if frames:
        clip = [f for f in (_rgb(x) for x in frames) if f is not None]
    else:
        if camera in start:
            clip.append(_rgb(start[camera]))
        if camera in now:
            clip.append(_rgb(now[camera]))
    clip = [c for c in clip if c is not None]
    out["n_frames"] = len(clip)
    if len(clip) < MIN_FRAMES:
        return finish(
            "no_frames",
            f"{len(clip)} frame(s) of {camera!r}: pass `frames` (the attempt, start to end) or `before` "
            "(the observation captured before the step)",
        )

    extra: dict[str, np.ndarray] = {}
    if use_views:
        for name in views if views is not None else [n for n in now if n != camera]:
            img = _rgb(now.get(name))
            if img is not None and name != camera:
                extra[name] = img
    out["views"] = list(extra)
    res = ctx.tool(
        "robojev.judge", task=task.strip(), frames=clip, views=extra or None, camera=camera
    )
    p = float(res["p_done"])
    out["p_done"] = p
    msg = f"P(done) {p:.3f} from {len(clip)} frame(s) of {camera}" + (
        f" + final {', '.join(extra)}" if extra else ""
    )
    if p >= success_at:
        return finish("success", msg)
    if p <= failure_at:
        return finish("failure", msg)
    return finish("uncertain", msg + f" (between {failure_at} and {success_at})")
