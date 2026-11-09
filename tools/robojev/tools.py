"""robojev tool bundle -- P(task done) for a robot attempt from its frames (RoboJev-27B).

RoboJev (Chen, "RoboJev: An Open Vision-Language Judge for Robot Success Detection", arXiv 2610.33718;
weights josefchen/RoboJev-27B, Apache-2.0, a LoRA-adapted autotrust/JEV-27B-VL, Qwen3.5 architecture)
reads the task sentence, ten main-camera frames evenly spaced over the attempt and the final frame of
each other camera, and answers false / true at one position; the released calibration
(``decision.json``: per-token bias, temperature 0.9) turns the two log-probabilities into P(done).

Adapted from the RoboJev client (github.com/josefchen/robojev at 79a5e2f, ``robojev/robojev.py``,
Apache-2.0, licence in LICENSE.robojev): the prompt strings (``DENSE_QUESTION``, ``dense_state_text``,
``camera_name``, the ``[kind] noul`` / ``[options]`` / ``[decision]:`` frame and the image
placeholders), the frame picking (``pick_frames``), the 288 px short side with a JPEG round trip,
and the read-out ``1 / (1 + exp(((lp_false + b_false) - (lp_true + b_true)) / T))``. Changed: the
model runs in this process through transformers (``Qwen3_5ForConditionalGeneration`` and its
processor) instead of behind a vLLM server - one forward pass, the last position's logits of the
two tokens; with vLLM's ``allowed_token_ids`` the two log-probabilities are renormalised over the
pair, which leaves their difference, and so P(done), unchanged. ``GAP_ROBOJEV_URL`` sends the same
request to a vLLM server instead (upstream's protocol, through urllib instead of requests).
``camera_name`` also finds the known names inside a longer one (``left_wrist`` -> "wrist camera").

The model loads lazily on first call (bf16, ~55 GB of GPU memory); importing this module never pulls
torch / transformers. Env knobs: ``GAP_ROBOJEV_MODEL`` (the pinned repo or a local directory),
``GAP_ROBOJEV_DEVICE`` (default ``cuda``), ``GAP_ROBOJEV_URL`` (a vLLM server serving the model as
``GAP_ROBOJEV_SERVED`` - default ``robojev``).
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import threading
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import PerceptionFailed, ToolError
from gap_core.tools import tool

logger = logging.getLogger(__name__)

#: Hugging Face repo -> the commit the bundle loads: the evaluated weights the model card pins.
MODELS = {"josefchen/RoboJev-27B": "e089a9b69a365744b80765057d86583647af95a9"}
DEFAULT_MODEL = "josefchen/RoboJev-27B"

#: The released decision.json (the model card's calibration), repeated so no download is needed.
FALSE_TOKEN, TRUE_TOKEN = 3721, 1802
BIAS = {"false": 0.00011960109259234741, "true": -0.00011981795250903815}
TEMPERATURE = 0.9
SHORT_SIDE = 288
FRAMES = 10

CAMERA_NAMES = {"front": "front camera", "top": "overhead camera", "wrist": "wrist camera"}
DENSE_QUESTION = (
    "The images are frames from a robot arm's attempt at the task in the state. The first images come from "
    "one camera, evenly spaced in time from the start of the attempt to its end; any further images show the "
    "end of the attempt from other cameras. Did the robot complete the task? "
    "True means: By the end of the attempt the task is completed as instructed. "
    "False means: The task is not completed: wrong place, dropped, not attempted, done wrongly or still in progress."
)
IMAGE_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>"

_load_lock = threading.Lock()
_runtime: Any = None


class JudgeResult(TypedDict):
    p_done: float  # calibrated P(the task was completed)
    logit_false: (
        float  # the two answer tokens' log-probabilities (or logits; only the difference matters)
    )
    logit_true: float
    frames_used: list[int]  # indices into the given frames
    views_used: list[str]
    prompt_state: str
    model: str


# ---------------------------------------------------------------------------
# Prompt (adapted from robojev/robojev.py)
# ---------------------------------------------------------------------------


def camera_name(camera: str) -> str:
    """How the prompt names a camera: the known names, else the last part of a dotted key."""
    short = camera.split(".")[-1]
    for key, name in CAMERA_NAMES.items():
        if key in short.lower():
            return name
    return CAMERA_NAMES.get(camera) or f"{short} camera"


def dense_state_text(task: str, cameras: list[str], frames: int = FRAMES) -> str:
    names = [camera_name(c) for c in cameras]
    lines = [
        f"Images 1-{frames}: {names[0]}, evenly spaced from the start to the end of the attempt."
    ]
    lines += [f"Image {frames + i + 1}: {name} at the end." for i, name in enumerate(names[1:])]
    return f"Task given to the robot arm: {task}\n" + "\n".join(lines)


def pick_frames(count: int, frames: int = FRAMES) -> list[int]:
    """``frames`` indices evenly spaced over ``count`` frames, first and last included (repeats when short)."""
    if count < 1:
        raise ValueError("at least one frame is needed")
    if frames == 1:
        return [count - 1]
    return [round(i * (count - 1) / (frames - 1)) for i in range(frames)]


def _resized(image: np.ndarray) -> Any:
    """A PIL image at RoboJev's 288 px short side, through a JPEG round trip (quality 92) as upstream."""
    from PIL import Image

    im = Image.fromarray(image)
    scale = SHORT_SIDE / min(im.size)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), buf.getvalue()


def p_done(logit_false: float, logit_true: float) -> float:
    no = (logit_false + BIAS["false"]) / TEMPERATURE
    yes = (logit_true + BIAS["true"]) / TEMPERATURE
    return 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, no - yes))))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _model_source() -> tuple[str, str | None]:
    name = os.environ.get("GAP_ROBOJEV_MODEL") or DEFAULT_MODEL
    if name in MODELS:
        return name, MODELS[name]
    if os.path.isdir(name):
        return name, None
    raise ValueError(
        f"GAP_ROBOJEV_MODEL must be one of {sorted(MODELS)} or a local directory, got {name!r}"
    )


class _Runtime:
    """RoboJev in this process: prompt + images in, the two answer logits out."""

    def __init__(self, source: str, revision: str | None, device: str) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.torch = torch
        self.name = source
        logger.info("Loading RoboJev %s@%s on %s ...", source, revision, device)
        # The tokenizer and the (slow, PIL) image processor rather than AutoProcessor, which also
        # builds the video processor and with it needs torchvision; the image tokens are expanded
        # here as the processor would.
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
        try:
            self.images = AutoImageProcessor.from_pretrained(
                source, revision=revision, backend="pil"
            )
        except (TypeError, ValueError):  # transformers before the backend argument
            self.images = AutoImageProcessor.from_pretrained(
                source, revision=revision, use_fast=False
            )
        self.merge = int(getattr(self.images, "merge_size", 2))
        self.model = (
            Qwen3_5ForConditionalGeneration.from_pretrained(
                source, revision=revision, dtype=torch.bfloat16
            )
            .to(device)
            .eval()
        )
        self.device = next(self.model.parameters()).device

    def logits(self, text: str, images: list[Any]) -> tuple[float, float]:
        torch = self.torch
        px = self.images(images=images, return_tensors="pt")
        grid = px["image_grid_thw"]
        parts = text.split("<|image_pad|>")
        if len(parts) != len(images) + 1:
            raise ValueError(f"{len(parts) - 1} image placeholders for {len(images)} images")
        counts = [int(g.prod()) // (self.merge**2) for g in grid]
        text = parts[0] + "".join(
            "<|image_pad|>" * n + p for n, p in zip(counts, parts[1:], strict=True)
        )
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        image_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        batch = {
            "input_ids": ids["input_ids"].to(self.device),
            # Modality per token (text 0, image 1), for the multimodal RoPE.
            "mm_token_type_ids": (ids["input_ids"] == image_id).int().to(self.device),
            "attention_mask": ids["attention_mask"].to(self.device),
            "pixel_values": px["pixel_values"].to(self.device, dtype=torch.bfloat16),
            "image_grid_thw": grid.to(self.device),
        }
        with torch.no_grad():
            out = self.model(**batch, logits_to_keep=1)
        last = out.logits[0, -1].float().log_softmax(-1)
        return float(last[FALSE_TOKEN]), float(last[TRUE_TOKEN])


class _Server:
    """The same question sent to a vLLM server (upstream's client protocol)."""

    def __init__(self, url: str, served: str) -> None:
        self.url, self.served, self.name = url.rstrip("/"), served, f"{served}@{url}"

    def logits(
        self, text: str, images: list[Any], jpegs: list[bytes], head: str, tail: str
    ) -> tuple[float, float]:
        import json
        import urllib.request

        template = (
            "{%- for m in messages -%}{%- for c in m['content'] -%}{%- if c['type'] == 'text' -%}{{ c['text'] }}"
            "{%- else -%}<|vision_start|><|image_pad|><|vision_end|>{%- endif -%}{%- endfor -%}{%- endfor -%}"
        )
        content = [{"type": "text", "text": head}]
        content += [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()},
            }
            for j in jpegs
        ]
        content.append({"type": "text", "text": tail})
        body = {
            "model": self.served,
            "messages": [{"role": "user", "content": content}],
            "chat_template": template,
            "add_generation_prompt": False,
            "add_special_tokens": False,
            "max_tokens": 1,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "logprobs": True,
            "top_logprobs": 2,
            "allowed_token_ids": [FALSE_TOKEN, TRUE_TOKEN],
            "return_tokens_as_token_ids": True,
        }
        req = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            reply = json.loads(resp.read())
        top = reply["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        lp = {int(t["token"].split(":")[1]): t["logprob"] for t in top}
        return float(lp.get(FALSE_TOKEN, -1e9)), float(lp.get(TRUE_TOKEN, -1e9))


def _get_runtime() -> Any:
    global _runtime
    with _load_lock:
        if _runtime is None:
            url = os.environ.get("GAP_ROBOJEV_URL")
            if url:
                _runtime = _Server(url, os.environ.get("GAP_ROBOJEV_SERVED", "robojev"))
            else:
                source, revision = _model_source()
                _runtime = _Runtime(source, revision, os.environ.get("GAP_ROBOJEV_DEVICE", "cuda"))
        return _runtime


def weights_cached() -> bool | None:
    """Filesystem-only weight-cache probe for ``gap check`` (never downloads, never imports torch)."""
    try:
        source, revision = _model_source()
    except ValueError:
        return False
    if revision is None:
        return os.path.isfile(os.path.join(source, "config.json"))
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        hit = try_to_load_from_cache(source, "model.safetensors.index.json", revision=revision)
    except Exception:
        return None
    return isinstance(hit, str)


def prefetch() -> None:
    """Snapshot-download the pinned weights into the HF cache (``gap skills check --download``)."""
    source, revision = _model_source()
    if revision is None:
        return
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=source,
        revision=revision,
        repo_type="model",
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def _rgb(image: Any, name: str, what: str) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(image))
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ToolError(name, f"{what}: expected RGB uint8 [H, W, 3], got {a.dtype} {a.shape}")
    return np.ascontiguousarray(a)


@tool(
    name="robojev.judge",
    summary="P(task completed) for one robot attempt from the task sentence, its main-camera frames (start to end) and final frames of other cameras (RoboJev-27B, calibrated).",
    tags=("perception", "verification"),
)
def judge(
    task: str,
    frames: list[np.ndarray],
    views: dict[str, np.ndarray] | None = None,
    camera: str = "main",
) -> JudgeResult:
    """``frames``: the attempt from one camera in time order, first = start, last = end (any number;
    ten are picked evenly, repeating when fewer). ``views``: camera name -> that camera's final frame.
    ``camera`` names the main camera in the prompt ("top" / "front" / "wrist" get the model's own
    wording). Returns the calibrated P(done) and the two raw log-probabilities."""
    name = "robojev.judge"
    if not isinstance(task, str) or not task.strip():
        raise ToolError(name, "task must be a non-empty sentence")
    if not isinstance(frames, (list, tuple)) or not frames:
        raise ToolError(name, "frames must be a non-empty list of RGB images (start to end)")
    idx = pick_frames(len(frames))
    imgs = [_rgb(frames[i], name, f"frame {i}") for i in idx]
    view_names = list((views or {}).keys())
    imgs += [_rgb(views[v], name, f"view {v!r}") for v in view_names]
    state = dense_state_text(task.strip(), [camera, *view_names])
    head = f"[kind] noul\n[state] {state}\n"
    tail = f"\n[question] {DENSE_QUESTION}\n[options]\nfalse\ntrue\n[decision]:"
    small = [_resized(im) for im in imgs]
    try:
        runtime = _get_runtime()
        if isinstance(runtime, _Server):
            lf, lt = runtime.logits("", [], [j for _, j in small], head, tail)
        else:
            lf, lt = runtime.logits(head + IMAGE_TOKENS * len(small) + tail, [p for p, _ in small])
    except Exception as e:
        raise PerceptionFailed(f"RoboJev failed: {e}") from e
    return {
        "p_done": float(p_done(lf, lt)),
        "logit_false": lf,
        "logit_true": lt,
        "frames_used": idx,
        "views_used": view_names,
        "prompt_state": state,
        "model": str(runtime.name),
    }
