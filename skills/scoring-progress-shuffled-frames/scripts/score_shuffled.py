# Adapted from OpenGVL (https://github.com/budzianowski/opengvl) at
# 36a39f3edede1935d3c0f5c8ba4ca724d5c0408b: opengvl/utils/prompts.py (get_prompt),
# configs/prompt_phrases/default.yaml, opengvl/clients/base.py (_iter_prompt_events),
# opengvl/mapper/regex_mapper.py and opengvl/metrics/voc.py, MIT License,
# Copyright (c) 2025 Pawel Budzianowski, Gracjan Goral, Emilia Wisnios.
# Changes: frames come from the graph (oldest first) and are shuffled here with a seeded
# generator; the hub's vlm.query takes the text first and the images after it, so the
# prompt lists the images by number instead of interleaving them; percentages are read
# per "Frame i" label (upstream read them in order of appearance and, for fractional
# values, rescaled them to sum to 100); the value-order correlation and the last frame's
# percentage become graph exits; no few-shot context episodes.
"""Generative Value Learning: a VLM rates shuffled frames of a step 0-100 %; the order of its ratings
against time says whether the step progressed, and the last frame's rating whether it is done."""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

DONE_PCT = 90.0
"""The last frame's rating at or over which the step counts as complete (DenseReward's 0.85 for 0-1 progress
and RynnValue's sub-2 s remaining play the same role); calibrate on your frames."""

PROGRESS_VOC = 0.5
"""Value-order correlation (Spearman of the ratings, in time order, against time) at or over which the step
is progressing; GVL reports VOC as its quality and success signal."""

REGRESS_DROP_PCT = 20.0
"""The last frame rated this many points under the best earlier one: regressed (something was undone)."""

_FRAME_RE = re.compile(r"Frame\s*(\d+)[\s*_]*:[^%\n]*?(-?\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


class Output(TypedDict):
    route: str
    progress: list[float]
    last_pct: float
    voc: float
    order: list[int]
    reason: str
    raw: str


def prompt_text(instruction: str, n: int) -> str:
    """OpenGVL's default prompt and phrases, with the images listed by number."""
    lines = [
        "You are an expert roboticist tasked to predict task completion percentages for frames of a robot "
        f"for the task of {instruction}. The task completion percentages are between 0 and 100, where 100 "
        "corresponds to full task completion. The frames may be in random order; reason about each frame "
        "independently when estimating completion.",
        "",
        "The images follow this text, in this order:",
        "Image 1: Initial robot scene. In the initial robot scene, the task completion percentage is 0%.",
    ]
    lines += [f"Image {i + 2}: Frame {i + 1}:" for i in range(n)]
    lines += [
        "",
        f"Now, for the task of {instruction}, output the task completion percentage for the following frames "
        "that are presented in random order. For each frame, format your response as follow: "
        "Frame {i}: Task Completion Percentages:{}%",
        "Be rigorous and precise; percentage reflects task completion.",
        "Remember: frames are in random order.",
    ]
    return "\n".join(lines)


def parse_ratings(text: str, n: int) -> dict[int, float]:
    """Frame number (1-based) -> rating, the first one given per frame, clipped to 0-100."""
    out: dict[int, float] = {}
    for m in _FRAME_RE.finditer(text):
        i = int(m.group(1))
        if 1 <= i <= n and i not in out:
            out[i] = float(np.clip(float(m.group(2)), 0.0, 100.0))
    return out


def value_order_correlation(values: list[float]) -> float:
    """Spearman correlation of ``values`` (in time order) with time; nan when undefined (upstream voc.py)."""
    v = np.asarray(values, dtype=float)
    if v.size < 2 or np.allclose(v, v[0]):
        return float("nan")
    rv = np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    # Average ranks for ties, as scipy.stats.spearmanr does.
    for val in np.unique(v):
        idx = np.flatnonzero(v == val)
        rv[idx] = rv[idx].mean()
    rt = np.arange(v.size, dtype=float)
    return float(np.corrcoef(rv, rt)[0, 1])


def _rgb(frame: Any) -> np.ndarray | None:
    if isinstance(frame, dict):
        cams = frame.get("cameras")
        if cams is not None:
            cams = list(cams.values()) if isinstance(cams, dict) else list(cams)
            frame = next((c.get("rgb") for c in cams if c.get("rgb") is not None), None)
        else:
            frame = frame.get("rgb")
    if frame is None:
        return None
    a = np.asarray(frame)
    if a.ndim != 3 or a.shape[2] < 3:
        return None
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0 if a.max() <= 1.0 else a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def run(
    ctx: NodeContext,
    frames: list[Any],
    instruction: str,
    initial: Any = None,
    seed: int = 0,
    done_pct: float = DONE_PCT,
    progress_voc: float = PROGRESS_VOC,
    regress_drop_pct: float = REGRESS_DROP_PCT,
    model: str | None = None,
) -> Output:
    """Rate ``frames`` (RGB arrays or observations, oldest first) shuffled; ``initial`` (default the first
    frame) is shown as the 0 % scene."""
    imgs = [_rgb(f) for f in frames or []]
    imgs = [a for a in imgs if a is not None]
    out: Output = {
        "route": "no_verdict",
        "progress": [],
        "last_pct": -1.0,
        "voc": float("nan"),
        "order": [],
        "reason": "",
        "raw": "",
    }
    if len(imgs) < 2:
        out["reason"] = f"need at least 2 frames with RGB, got {len(imgs)}"
        return out
    start = _rgb(initial) if initial is not None else imgs[0]
    order = [int(i) for i in np.random.default_rng(seed).permutation(len(imgs))]
    out["order"] = order
    kw: dict[str, Any] = {
        "prompt": prompt_text(instruction, len(imgs)),
        "images": [start] + [imgs[i] for i in order],
    }
    if model:
        kw["model"] = model
    try:
        raw = str(ctx.tool("vlm.query", **kw).get("text") or "")
    except Exception as exc:  # noqa: BLE001  (auth / network / unserved tool)
        out["reason"] = f"vlm.query failed: {exc}"
        return out
    out["raw"] = raw
    got = parse_ratings(raw, len(imgs))
    if len(got) < len(imgs):
        out["reason"] = f"ratings for {len(got)} of {len(imgs)} frames"
        return out
    chrono = [0.0] * len(imgs)
    for k, i in enumerate(order):
        chrono[i] = got[k + 1]
    voc = value_order_correlation(chrono)
    last, best_before = chrono[-1], max(chrono[:-1])
    out.update(progress=chrono, last_pct=last, voc=voc)
    if last >= done_pct:
        route, why = "complete", f"last frame rated {last:.0f} % (>= {done_pct:.0f})"
    elif best_before - last > regress_drop_pct:
        route, why = "regressed", f"last frame {last:.0f} % after {best_before:.0f} % earlier"
    elif voc == voc and voc >= progress_voc:
        route, why = (
            "progress_increasing",
            f"ratings rise with time (VOC {voc:.2f}); last {last:.0f} %",
        )
    else:
        route, why = "stalled", f"ratings do not rise with time (VOC {voc:.2f}); last {last:.0f} %"
    out.update(route=route, reason=why)
    logger.info("score_shuffled: %s (%s)", route, why)
    return out
