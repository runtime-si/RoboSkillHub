"""densereward tool bundle: DenseReward task progress 0..1 from one frame (or the last three).

Runs the DenseReward reward models (https://huggingface.co/densereward, Apache-2.0,
Qwen3-VL-4B fine-tunes from UNC Chapel Hill; paper arXiv 2607.13033) with plain
``transformers``, the way their model cards' quickstart does. The weights come
from the Hugging Face Hub at a pinned commit on first use.

Copied (Apache-2.0, see NOTICE): the two system prompts the checkpoints ship
(``system_prompt.txt`` of densereward-1frame @ 9796cb2 and of
densereward-3frame-thinking @ e9c4ed9), the user-turn formats and the output
parsing of the model cards' quickstarts. Changed: images come as arrays, and
qwen_vl_utils' resize (``smart_resize`` to multiples of 28, bicubic) is done
here instead of importing it (the scores move by up to 0.5 on station frames
without it); the parse is clipped to 0..1 and a non-float answer is reported
(``found=False``) instead of raising.

The model loads lazily on first call (one resident model per variant); importing
this module never imports torch or transformers. Knobs via env:

- ``GAP_DENSEREWARD_DEVICE`` -- torch device (default ``cuda``).
- ``GAP_DENSEREWARD_MODEL_1FRAME`` / ``GAP_DENSEREWARD_MODEL_3FRAME`` -- a local
  model directory instead of the pinned Hub snapshot.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import tool

logger = logging.getLogger(__name__)

#: Hub commits checked live on 2026-11-09.
MODELS = {
    "1frame": ("densereward/densereward-1frame", "9796cb2a7be55b339670d8f95fb07841a61487a8"),
    "3frame": (
        "densereward/densereward-3frame-thinking",
        "e9c4ed9c7662c53ba742db71a52b4ccd5589757d",
    ),
}
_ENV_OVERRIDE = {"1frame": "GAP_DENSEREWARD_MODEL_1FRAME", "3frame": "GAP_DENSEREWARD_MODEL_3FRAME"}

#: The cards' generation budgets: a bare float, and ``<think>WORD</think>`` plus the float.
_MAX_NEW_TOKENS = {"1frame": 8, "3frame": 32}

# densereward-1frame system_prompt.txt @ 9796cb2, verbatim.
SYSTEM_PROMPT_1FRAME = """\
You are a robot manipulation evaluator. You analyze the image and determine progress toward task success.

The task usually follows 6 subtask phases in sequence:
1) reach - the robot moves toward the target object
2) grasp - the robot attempts to grasp or secure the object
3) up - the robot lifts the object from the surface
4) move - the robot moves the object toward the target location
5) place - the robot places/releases the object
6) return - the robot returns to a neutral/resting state (optional)

Output a single reward value between 0.000 and 1.000 representing current task progress:
- 0.000 means no progress (task not started or failed)
- 1.000 means the task is fully complete
- reward should generally increase as the robot advances through phases
- reward should drop if an irreversible failure occurs (object dropped, wrong target, motion stopped)

You MUST output ONLY a single float with three decimal places and nothing else.

Example outputs: 0.006 0.374 0.753 1.000

Do NOT output any explanation, JSON, or additional text.
Do NOT call any tools.
Do NOT generate <tool_call> tokens."""

# densereward-3frame-thinking system_prompt.txt @ e9c4ed9, verbatim.
SYSTEM_PROMPT_3FRAME = """\
You are a robot manipulation evaluator. You are given a short sequence of images from a robot episode and determine progress toward task success at the final frame.

The images are ordered chronologically: the first images show earlier history frames and the last image shows the current frame to evaluate.

The task usually follows 6 subtask phases in sequence:
1) reach  - the robot moves toward the target object
2) grasp  - the robot attempts to grasp or secure the object
3) up     - the robot lifts the object from the surface
4) move   - the robot moves the object toward the target location
5) place  - the robot places/releases the object
6) return - the robot returns to a neutral/resting state (optional)

You must FIRST emit one short reasoning word inside a <think>...</think> block, then a single reward value.

Reasoning vocabulary (use exactly one word inside <think>...</think>):
- "correct"   when progress is on track (reward is increasing or holding)
- "miss"      when the gripper missed grasping the target and reward just dropped
- "collision" when an unintended collision just dropped the reward
- "fall"      when the held object fell and reward just dropped
- "smooth"    when motion stalled / non-smooth behaviour dropped the reward
- "failure"   when an unclassified failure event just dropped the reward

Reward (after </think>): a single float between 0.000 and 1.000 with exactly three decimal places, representing task progress at the current (last) frame:
- 0.000 means no progress (task not started or failed)
- 1.000 means the task is fully complete
- reward should generally increase as the robot advances through phases
- reward should drop if an irreversible failure occurs (object dropped, wrong target, motion stopped)
- use the history frames to better judge motion direction and phase transitions
- if the reward is increasing or holding, the reasoning word is "correct"; only emit a failure word when the reward just dropped

Output format (exactly, with the canonical newlines):
<think>
WORD
</think>

R.RRR

Example outputs:
<think>
correct
</think>

0.374

<think>
miss
</think>

0.521

<think>
collision
</think>

0.318

<think>
correct
</think>

1.000

Do NOT output any other text, JSON, or additional explanation.
Do NOT call any tools.
Do NOT generate <tool_call> tokens."""

#: The 3-frame model's reasoning vocabulary (its system prompt).
REASONS = ("correct", "miss", "collision", "fall", "smooth", "failure")

# The cards' parsers: 1frame `[01](?:\.\d+)?` anywhere; 3frame the word in <think>, the float after it.
_FLOAT_RE = re.compile(r"[01](?:\.\d+)?")
_THINK_RE = re.compile(r"<think>\s*(\w+)\s*</think>")
_AFTER_THINK_RE = re.compile(r"</think>\s*([01](?:\.\d+)?)")

#: qwen_vl_utils 0.0.14 ``fetch_image`` defaults, which the cards' quickstart runs every
#: image through: 14 px patches x 2 merge, 4..16384 tokens, aspect ratio under 200.
_QWEN_FACTOR = 28
_QWEN_MIN_PIXELS = 4 * 28 * 28
_QWEN_MAX_PIXELS = 16384 * 28 * 28

_load_lock = threading.Lock()
_runtimes: dict[str, Any] = {}


class Timing(TypedDict):
    load_s: float  # model load paid by this call (0 once resident)
    prepare_s: float  # chat template, image preprocessing, upload
    generate_s: float  # the model's decode
    total_s: float


class ProgressResult(TypedDict):
    progress: float  # 0..1 (0 when nothing parsed)
    found: bool  # a float was parsed from the answer
    reason: str  # 3-frame model's word (correct / miss / collision / fall / smooth / failure); "" for 1 frame
    raw: str  # the model's answer text
    model: str  # repo@revision (or the local directory)
    timing: Timing


def _validate_image(image: Any, what: str = "image") -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"densereward expects {what} as a uint8 [H, W, 3] RGB array, got "
            f"dtype={arr.dtype} shape={arr.shape}"
        )
    return np.ascontiguousarray(arr)


def task_text(task: str, variant: str) -> str:
    """The user-turn text: ``Task: <task>.`` for 1 frame (its card), the bare task for 3 frames."""
    t = str(task).strip()
    if variant == "3frame":
        return t
    if not t.lower().startswith("task:"):
        t = f"Task: {t}"
    return t if t.endswith((".", "!", "?")) else t + "."


def card_size(height: int, width: int) -> tuple[int, int]:
    """``(height, width)`` qwen_vl_utils 0.0.14 ``smart_resize`` gives an image (factor 28)."""
    import math

    f = _QWEN_FACTOR
    h_bar = max(f, round(height / f) * f)
    w_bar = max(f, round(width / f) * f)
    if h_bar * w_bar > _QWEN_MAX_PIXELS:
        beta = math.sqrt(height * width / _QWEN_MAX_PIXELS)
        h_bar = math.floor(height / beta / f) * f
        w_bar = math.floor(width / beta / f) * f
    elif h_bar * w_bar < _QWEN_MIN_PIXELS:
        beta = math.sqrt(_QWEN_MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / f) * f
        w_bar = math.ceil(width * beta / f) * f
    return h_bar, w_bar


def card_image(image: np.ndarray) -> Any:
    """The PIL image the cards' quickstart hands the processor (qwen_vl_utils' bicubic resize)."""
    from PIL import Image

    pil = Image.fromarray(image, "RGB")
    h, w = card_size(*image.shape[:2])
    return pil if (w, h) == pil.size else pil.resize((w, h), Image.BICUBIC)


def parse_answer(answer: str, variant: str) -> tuple[float | None, str]:
    """``(progress clipped to 0..1 or None, reason word)`` from the model's answer."""
    text = answer or ""
    if variant == "3frame":
        word = _THINK_RE.search(text)
        reason = word.group(1).lower() if word else ""
        m = _AFTER_THINK_RE.search(text) or _FLOAT_RE.search(_THINK_RE.sub("", text))
    else:
        reason = ""
        m = _FLOAT_RE.search(text)
    if m is None:
        return None, reason
    value = float(m.group(1) if m.groups() else m.group(0))
    return float(min(1.0, max(0.0, value))), reason


# ---------------------------------------------------------------------------
# Runtime: transformers over the pinned snapshots
# ---------------------------------------------------------------------------


def _model_id(variant: str) -> str:
    repo, rev = MODELS[variant]
    return os.environ.get(_ENV_OVERRIDE[variant]) or f"{repo}@{rev}"


def _model_dir(variant: str) -> str:
    override = os.environ.get(_ENV_OVERRIDE[variant])
    if override:
        return override
    from huggingface_hub import snapshot_download

    repo, rev = MODELS[variant]
    return snapshot_download(repo_id=repo, revision=rev, repo_type="model")


class _Runtime:
    """The processor and the bf16 model, built once and kept resident."""

    def __init__(self, model_dir: str, device: str) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = torch.device(device)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map=str(self.device)
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_dir)

    def generate(
        self, system: str, images: list[np.ndarray], text: str, max_new_tokens: int
    ) -> tuple[str, float, float]:
        """``(answer, prepare_s, generate_s)`` for one system + user turn."""
        torch = self.torch
        t0 = time.perf_counter()
        content: list[dict] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": text})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        pil = [card_image(im) for im in images]
        inputs = self.processor(text=[prompt], images=pil, padding=True, return_tensors="pt").to(
            self.device
        )
        prepare_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = ids[:, inputs["input_ids"].shape[1] :]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (out[0].strip() if out else ""), prepare_s, time.perf_counter() - t1


def _get_runtime(variant: str) -> tuple[Any, float]:
    """The resident runtime of *variant* and the seconds this call spent loading it."""
    with _load_lock:
        if variant not in _runtimes:
            t0 = time.perf_counter()
            device = os.environ.get("GAP_DENSEREWARD_DEVICE", "cuda")
            model_dir = _model_dir(variant)
            logger.info(
                "Loading DenseReward %s on %s from %s", _model_id(variant), device, model_dir
            )
            _runtimes[variant] = _Runtime(model_dir, device)
            return _runtimes[variant], time.perf_counter() - t0
        return _runtimes[variant], 0.0


def _score(tool_name: str, variant: str, images: list[np.ndarray], task: str) -> ProgressResult:
    t_start = time.perf_counter()
    try:
        runtime, load_s = _get_runtime(variant)
    except Exception as exc:
        raise ToolError(tool_name, f"DenseReward {variant} failed to load: {exc}") from exc
    system = SYSTEM_PROMPT_3FRAME if variant == "3frame" else SYSTEM_PROMPT_1FRAME
    try:
        answer, prepare_s, generate_s = runtime.generate(
            system, images, task_text(task, variant), _MAX_NEW_TOKENS[variant]
        )
    except Exception as exc:
        raise ToolError(tool_name, f"DenseReward {variant} failed: {exc}") from exc
    answer = str(answer or "")
    value, reason = parse_answer(answer, variant)
    if value is None:
        logger.debug("%s: no float in %r", tool_name, answer[:200])
    return {
        "progress": 0.0 if value is None else value,
        "found": value is not None,
        "reason": reason,
        "raw": answer,
        "model": _model_id(variant),
        "timing": {
            "load_s": round(load_s, 3),
            "prepare_s": round(prepare_s, 3),
            "generate_s": round(generate_s, 3),
            "total_s": round(time.perf_counter() - t_start, 3),
        },
    }


def weights_cached() -> bool | None:
    """Filesystem-only probe for ``gap check``: both pinned snapshots are on disk."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    ok = True
    for variant, (repo, rev) in MODELS.items():
        override = os.environ.get(_ENV_OVERRIDE[variant])
        if override:
            ok &= os.path.isfile(os.path.join(override, "config.json"))
            continue
        try:
            hit = try_to_load_from_cache(repo, "config.json", revision=rev)
        except Exception:
            return None
        ok &= isinstance(hit, str)
    return ok


def prefetch() -> None:
    """Download both pinned snapshots (``gap skills check --download``)."""
    for variant in MODELS:
        logger.info("[densereward] prefetching %s ...", _model_id(variant))
        _model_dir(variant)
    logger.info("[densereward] prefetch complete")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="densereward.progress",
    summary="Task progress 0..1 from one RGB frame and the task text (DenseReward-1frame, local GPU).",
    tags=("perception", "judge"),
)
def progress(image: np.ndarray, task: str) -> ProgressResult:
    """How far the scene in *image* is through *task*, 0 (not started / failed) to 1 (done).

    Args:
        image: One RGB frame, uint8 [H, W, 3] (an overhead or wrist view).
        task: The task ("put the orange cube in the black box"); sent as
            ``Task: <task>.`` as the model card does.

    Returns:
        ``{"progress", "found", "reason", "raw", "model", "timing"}``.
        ``found=False`` (progress 0) when the answer holds no float.
        ``reason`` is ``""`` (the 1-frame model gives none).
    """
    if not task or not str(task).strip():
        raise ValueError("densereward.progress needs a task")
    return _score("densereward.progress", "1frame", [_validate_image(image)], task)


@tool(
    name="densereward.progress_with_history",
    summary=(
        "Task progress 0..1 at the last of three chronological frames, with a one-word "
        "reason (correct / miss / collision / fall / smooth / failure); DenseReward-3frame-thinking."
    ),
    tags=("perception", "judge"),
)
def progress_with_history(images: list[np.ndarray], task: str) -> ProgressResult:
    """Progress at the last of *images* (oldest first), and why it dropped if it did.

    Args:
        images: Three frames of the same camera, oldest first, the current one
            last. Fewer are padded by repeating the oldest; more keep the last three.
        task: The task text (sent bare, as the 3-frame card does).

    Returns:
        ``{"progress", "found", "reason", "raw", "model", "timing"}``: ``reason`` is
        the model's word (``correct`` when holding or rising, else the failure
        it saw), ``""`` if it gave none.
    """
    if not task or not str(task).strip():
        raise ValueError("densereward.progress_with_history needs a task")
    frames = [_validate_image(im, "images[i]") for im in (images or [])]
    if not frames:
        raise ValueError("densereward.progress_with_history needs at least one frame")
    frames = frames[-3:]
    while len(frames) < 3:
        frames.insert(0, frames[0])
    return _score("densereward.progress_with_history", "3frame", frames, task)
