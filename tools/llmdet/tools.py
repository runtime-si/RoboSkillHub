"""llmdet tool bundle -- open-vocabulary detection with LLMDet (CVPR 2025, iSEE-Laboratory, Apache-2.0).

LLMDet: Grounding DINO's architecture co-trained with LLM-generated captions; weights
``iSEE-Laboratory/llmdet_{tiny,base,large}`` (Apache-2.0 model cards), loaded through Hugging Face
transformers' ``AutoModelForZeroShotObjectDetection`` as LLMDet's README ("Use LLMDet in Huggingface")
does. Adapted from tools/mm-grounding-dino/tools.py of this repository (the processor -> model ->
``post_process_grounded_object_detection`` path, period-terminated query, detection dicts), so it is a
drop-in for grounding-dino / mm-grounding-dino in a skill and as a sahi backend. Changed: LLMDet's
pinned checkpoints; transformers 5.x is allowed (checked live: boxes decode correctly on 5.22.2).

Knobs via env: ``GAP_LLMDET_MODEL`` (one of ``MODELS`` or a local directory; default large),
``GAP_LLMDET_DEVICE`` (default ``cuda``).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import PerceptionFailed, ToolError
from gap_core.tools import tool
from gap_core.types import BoundingBox2D

logger = logging.getLogger(__name__)

#: Hugging Face repo -> the commit the bundle loads (model cards: Apache-2.0).
MODELS = {
    "iSEE-Laboratory/llmdet_large": "bec37f296f05b22f6c6b39bc05a6c611239f4e31",
    "iSEE-Laboratory/llmdet_base": "3537869413e7dca01cfd86d6ab10553b72e4fe29",
    "iSEE-Laboratory/llmdet_tiny": "d05199165a19320a9236396e20fca0a5e065189c",
}
DEFAULT_MODEL = "iSEE-Laboratory/llmdet_large"

#: Box floor. On the 7 station frames (SKILL.md), large on a table ROI: 0.3 gave
#: 6 / 12 small parts, 3 / 4 open boxes and 2 false positives; 0.25 gave 7 / 12,
#: 4 / 4 and 9; grounding-dino's 0.2 gave 8 / 12, 4 / 4 and 18.
DEFAULT_BOX_THRESHOLD = 0.3
#: Token floor that names a box (grounding-dino's default; the station check used it).
DEFAULT_TEXT_THRESHOLD = 0.2
#: The transformers release the bundle is locked to and checked with.
TRANSFORMERS_MAX_MAJOR = (
    5  # LLMDet decodes correctly on 5.22.2 (live check), unlike MM-Grounding-DINO
)

_load_lock = threading.Lock()
_runtime: Any = None


class Detection(TypedDict):
    box: BoundingBox2D  # [x1, y1, x2, y2] in pixels
    label: str  # matched text span ("" when no token cleared text_threshold)
    score: float  # box confidence [0, 1]


class DetectResult(TypedDict):
    detections: list[Detection]


def _model_source() -> tuple[str, str | None]:
    name = os.environ.get("GAP_LLMDET_MODEL") or DEFAULT_MODEL
    if name in MODELS:
        return name, MODELS[name]
    if os.path.isdir(name):
        return name, None
    raise ValueError(
        f"GAP_LLMDET_MODEL must be one of {sorted(MODELS)} or a local directory, got {name!r}"
    )


def _check_transformers(version: str) -> None:
    major = int(version.split(".", 1)[0])
    if major > TRANSFORMERS_MAX_MAJOR:
        raise PerceptionFailed(
            f"llmdet needs transformers < 6 (found {version}): checked on 4.55+ and 5.22; "
            "use the bundle's own environment (uv sync --project tools/llmdet)"
        )


class _Runtime:
    def __init__(self, source: str, revision: str | None, device: str) -> None:
        import torch
        import transformers
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        _check_transformers(transformers.__version__)
        self.torch, self.device = torch, device
        logger.info("Loading LLMDet %s@%s on %s ...", source, revision, device)
        self.processor = AutoProcessor.from_pretrained(source, revision=revision)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(source, revision=revision)
        self.model = model.to(device).eval()

    def __call__(self, image: np.ndarray, text: str, box_thr: float, text_thr: float) -> dict:
        from PIL import Image

        pil = Image.fromarray(image, "RGB")
        inputs = self.processor(images=pil, text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_thr,
            text_threshold=text_thr,
            target_sizes=[pil.size[::-1]],
        )[0]
        labels = res.get("text_labels", res["labels"])
        return {
            "boxes": res["boxes"].float().cpu().numpy(),
            "scores": res["scores"].float().cpu().numpy(),
            "labels": [str(x) for x in labels],
        }


def _get_runtime() -> Any:
    global _runtime
    with _load_lock:
        if _runtime is None:
            source, revision = _model_source()
            _runtime = _Runtime(source, revision, os.environ.get("GAP_LLMDET_DEVICE", "cuda"))
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
        hit = try_to_load_from_cache(source, "config.json", revision=revision)
    except Exception:
        return None
    return isinstance(hit, str)


def prefetch() -> None:
    """Snapshot-download the pinned weights into the HF cache (``gap skills check --download``)."""
    source, revision = _model_source()
    if revision is None:
        return
    from huggingface_hub import snapshot_download

    logger.info("[llmdet] prefetching %s@%s ...", source, revision)
    snapshot_download(repo_id=source, revision=revision, repo_type="model")
    logger.info("[llmdet] prefetch complete")


def _query(query: str | None, prompts: list[str] | str | None) -> str:
    """Grounding DINO's period-terminated phrase list from a query string or a list of phrases."""
    if prompts is not None:
        items = prompts.split(".") if isinstance(prompts, str) else list(prompts)
        phrases = [str(p).strip().rstrip(".").strip() for p in items]
        phrases = [p for p in phrases if p]
        if not phrases:
            raise ToolError("llmdet.detect", f"no phrase in prompts={prompts!r}")
        return " ".join(f"{p}." for p in phrases)
    if not query or not query.strip(" ."):
        raise ToolError("llmdet.detect", "give query ('cube. plug.') or prompts")
    q = query.strip()
    return q if q.endswith(".") else q + "."


@tool(
    name="llmdet.detect",
    summary="Open-vocabulary detection with LLMDet (labelled boxes + scores, grounding-dino's shape).",
    tags=("perception",),
)
def detect(
    image: np.ndarray,
    query: str | None = None,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    prompts: list[str] | None = None,
    max_detections: int = 0,
) -> DetectResult:
    """Detect the phrases of ``query`` (``"cube. open box."``) or ``prompts``
    (a list) in an RGB uint8 ``[H, W, 3]`` image, best-first.

    Same call and answer as ``grounding-dino.detect``: ``label`` is the matched
    text span (group by it; sahi snaps it to the prompt). Zero or negative
    thresholds fall back to the defaults. ``max_detections`` > 0 keeps the best N.
    """
    arr = np.ascontiguousarray(np.asarray(image))
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ToolError(
            "llmdet.detect",
            f"expected RGB uint8 [H, W, 3] image, got {arr.dtype} {arr.shape}",
        )
    text = _query(query, prompts)
    box_thr = box_threshold if box_threshold > 0 else DEFAULT_BOX_THRESHOLD
    text_thr = text_threshold if text_threshold > 0 else DEFAULT_TEXT_THRESHOLD
    try:
        r = _get_runtime()(arr, text, box_thr, text_thr)
    except (ToolError, PerceptionFailed):
        raise
    except Exception as e:
        raise PerceptionFailed(f"LLMDet detection failed: {e}") from e
    order = np.argsort(-np.asarray(r["scores"], dtype=np.float64), kind="stable")
    if max_detections > 0:
        order = order[:max_detections]
    dets: list[Detection] = []
    for i in order:
        b = r["boxes"][i]
        dets.append(
            {
                "box": {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])},
                "label": r["labels"][i],
                "score": float(r["scores"][i]),
            }
        )
    logger.info("llmdet.detect returning %d detections", len(dets))
    return {"detections": dets}
