"""RynnValue tool bundle -- remaining time to task completion for every frame of a clip.

RynnValue (Alibaba DAMO Academy, arXiv 2608.09853, Apache-2.0) is a RynnBrain / Qwen3-VL
model that reads a task instruction and up to K sampled frames of a robot clip and predicts,
for every frame, the **remaining time in seconds until the task is complete** (a 256-bin
quantile head, decoded as the expected bin centre), the signed time between consecutive
frames, and a short generated analysis ending in ``Match: Yes/No`` (does the clip match the
instruction) and ``Success: Yes/No``.

One tool, ``rynnvalue.remaining_time(images, instruction)``. Frames are oldest first; the
value read for "now" is the last one. Runs on a GPU (bf16: ~9 GB for the 4B, ~17 GB for the
8B); out of process through ``gap_core.rpc.server``.

Environment:

- ``GAP_RYNNVALUE_MODEL`` -- ``8b`` (default, RynnValue-8B-Quantile) or ``4b``
  (RynnValue-4B-Quantile), or a local model directory.
- ``GAP_RYNNVALUE_DEVICE`` -- torch device (default ``cuda``).
- ``GAP_RYNNVALUE_ATTN`` -- inference attention (default ``sdpa``: the causal mask the
  model card's headline numbers use; ``pred_slot_isolated_eager`` is the training mask,
  which upstream's inference.py forces and the card reports ranking worse).

The inference follows upstream ``rynn_infer/inference.py`` (process_episode, one forward
pass, the value head averaged over its ensemble, then a greedy generate of the Analysis
block parsed with the same regexes). Upstream: https://github.com/alibaba-damo-academy/RynnValue
at cd29294f1ac56bc817074bd665ba92b630936ad7 (Apache-2.0); the modelling code itself is
loaded from the pinned Hugging Face revision with ``trust_remote_code``.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, TypedDict

import numpy as np
from gap_core.tools import tool

logger = logging.getLogger(__name__)

MODELS = {
    "8b": (
        "Alibaba-DAMO-Academy/RynnValue-8B-Quantile",
        "0fb3ac12c73844ea1abaf4f767c895aae248a9ec",
    ),
    "4b": (
        "Alibaba-DAMO-Academy/RynnValue-4B-Quantile",
        "d48ffefcefaf8b5381ae1e412e9a62faf1456bcd",
    ),
}
"""Pinned Hugging Face revisions (both Apache-2.0 per their model cards)."""

MAX_FRAMES = 8
"""K = 8 sampled observations per clip in training and in the card's evaluation."""

MAX_IMAGE_SIDE = 640
"""Upstream inference.py's ``--max_image_side`` default."""

DEFAULT_ROBOT = "a robot arm with a parallel-jaw gripper"
DEFAULT_CAMERA = "a fixed third-person camera looking down at the table"
"""Used when the caller gives neither description: the released checkpoints were trained with
the embodiment / camera meta block (``use_meta=True``) and refuse a prompt without it."""

_DESCRIPTION_RE = re.compile(r"-\s*Video Description:\s*(.+)", re.IGNORECASE)
_MATCH_RE = re.compile(r"-\s*Match:\s*(Yes|No)", re.IGNORECASE)
_SUCCESS_RE = re.compile(r"-\s*Success:\s*(Yes|No)", re.IGNORECASE)


class RemainingTime(TypedDict):
    remaining_s: list[float]
    final_remaining_s: float
    delta_s: list[float]
    entropy: list[float]
    match: bool | None
    success: bool | None
    description: str
    analysis: str
    frames_used: int
    model: str
    timing: dict[str, float]


def parse_analysis(text: str) -> dict[str, Any]:
    """Upstream's Analysis parser: description, Match and Success (None when omitted)."""

    def first(pattern: re.Pattern) -> str | None:
        m = pattern.search(text)
        return m.group(1).strip() if m else None

    def yes(v: str | None) -> bool | None:
        return None if v is None else v.lower() == "yes"

    return {
        "description": first(_DESCRIPTION_RE) or "",
        "match": yes(first(_MATCH_RE)),
        "success": yes(first(_SUCCESS_RE)),
    }


def select_frames(n: int, k: int = MAX_FRAMES) -> list[int]:
    """Indices of at most ``k`` frames, uniform over the clip, always keeping the last
    (upstream builds each prefix with ``np.linspace(0, end, num_frames, dtype=int)``)."""
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return [int(i) for i in np.linspace(0, n - 1, k, dtype=int)]


def _validate_image(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"rynnvalue expects RGB [H, W, 3] frames, got shape {arr.shape}")
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return arr


def _pil(arr: np.ndarray) -> Any:
    from PIL import Image

    img = Image.fromarray(arr, "RGB")
    side = max(img.size)
    if side > MAX_IMAGE_SIDE:
        s = MAX_IMAGE_SIDE / side
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.BICUBIC)
    return img


def _model_ref() -> tuple[str, str | None, str]:
    raw = os.environ.get("GAP_RYNNVALUE_MODEL", "8b").strip()
    if raw.lower() in MODELS:
        repo, rev = MODELS[raw.lower()]
        return repo, rev, f"{repo}@{rev[:12]}"
    return raw, None, raw


class _Runtime:
    def __init__(self) -> None:
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor

        repo, rev, self.name = _model_ref()
        self.device = torch.device(os.environ.get("GAP_RYNNVALUE_DEVICE", "cuda"))
        attn = os.environ.get("GAP_RYNNVALUE_ATTN", "sdpa")
        cfg = AutoConfig.from_pretrained(repo, revision=rev, trust_remote_code=True)
        cfg._attn_implementation = attn
        self.model = (
            AutoModel.from_pretrained(
                repo, revision=rev, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
            .to(device=self.device, dtype=torch.bfloat16)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(repo, revision=rev, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.attn = attn

    def run(
        self,
        images: list[Any],
        instruction: str,
        robot_description: str | None,
        camera_description: str | None,
        analysis: bool,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        import torch

        sample = self.processor.process_episode(
            instruction=instruction,
            images=images,
            robot_description=robot_description,
            camera_description=camera_description,
        )
        kw = dict(
            input_ids=sample["input_ids"].to(self.device).long(),
            attention_mask=sample["attention_mask"].to(self.device).long(),
            pixel_values=sample["pixel_values"].flatten(0, 1).to(self.device),
            image_grid_thw=sample["image_grid_thw"].flatten(0, 1).to(self.device).long(),
        )
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model(**kw)
        value = out.value.pred_value.float()
        if value.dim() == 3:  # (heads, batch, frames)
            value = value.mean(dim=0)
        remaining = value.reshape(-1).tolist()
        ent = out.value.entropy.float()
        if ent.dim() == 3:
            ent = ent.mean(dim=0)
        entropy = ent.reshape(-1).tolist()
        rel = getattr(getattr(out, "relative", None), "pred_value", None)
        delta = [] if rel is None else rel.float().reshape(-1).tolist()
        t1 = time.perf_counter()
        text = ""
        if analysis:
            with torch.inference_mode():
                gen = self.model.generate(
                    **kw,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=self.eos,
                    pad_token_id=self.eos,
                    use_cache=True,
                )
            text = self.tokenizer.decode(
                gen[0, kw["input_ids"].shape[1] :], skip_special_tokens=True
            )
        t2 = time.perf_counter()
        return {
            "remaining": remaining,
            "entropy": entropy,
            "delta": delta,
            "text": text,
            "timing": {"value_s": round(t1 - t0, 3), "analysis_s": round(t2 - t1, 3)},
        }


_runtime: _Runtime | None = None
_lock = threading.Lock()


def _get_runtime() -> _Runtime:
    global _runtime
    with _lock:
        if _runtime is None:
            logger.info("Loading RynnValue %s", _model_ref()[2])
            _runtime = _Runtime()
        return _runtime


def remaining_time_impl(
    images: list[Any],
    instruction: str,
    robot_description: str | None = None,
    camera_description: str | None = None,
    analysis: bool = True,
    max_new_tokens: int = 128,
    runtime: Any = None,
) -> RemainingTime:
    """The tool body, with the runtime injectable for tests."""
    if not instruction or not str(instruction).strip():
        raise ValueError("instruction is required")
    frames = [_validate_image(f) for f in images or []]
    if not frames:
        raise ValueError("give at least one frame")
    keep = select_frames(len(frames))
    pil = [_pil(frames[i]) for i in keep]
    if robot_description is None and camera_description is None:
        robot_description, camera_description = DEFAULT_ROBOT, DEFAULT_CAMERA
    rt = runtime or _get_runtime()
    t0 = time.perf_counter()
    raw = rt.run(
        pil, str(instruction), robot_description, camera_description, analysis, max_new_tokens
    )
    parsed = parse_analysis(raw["text"])
    remaining = [round(float(v), 3) for v in raw["remaining"]][: len(keep)]
    return {
        "remaining_s": remaining,
        "final_remaining_s": remaining[-1] if remaining else float("nan"),
        "delta_s": [round(float(v), 3) for v in raw["delta"]],
        "entropy": [round(float(v), 4) for v in raw["entropy"]][: len(keep)],
        "match": parsed["match"],
        "success": parsed["success"],
        "description": parsed["description"],
        "analysis": raw["text"],
        "frames_used": len(keep),
        "model": getattr(rt, "name", "rynnvalue"),
        "timing": {**raw["timing"], "total_s": round(time.perf_counter() - t0, 3)},
    }


@tool(
    name="rynnvalue.remaining_time",
    summary="Remaining seconds to task completion for each frame of a clip, plus Match/Success (RynnValue).",
    tags=("perception", "judge"),
)
def remaining_time(
    images: list[np.ndarray],
    instruction: str,
    robot_description: str | None = None,
    camera_description: str | None = None,
    analysis: bool = True,
    max_new_tokens: int = 128,
) -> RemainingTime:
    """Frames oldest first (at most 8 are used: uniform over the clip, the last always kept)."""
    return remaining_time_impl(
        images, instruction, robot_description, camera_description, analysis, max_new_tokens
    )
