"""mm-grounding-dino tool bundle -- open-vocabulary detection with MM-Grounding-DINO.

MM-Grounding-DINO (Zhao et al., "An Open and Comprehensive Pipeline for Unified
Object Grounding and Detection", 2024; open-mmlab/mmdetection, Apache-2.0):
Grounding DINO's architecture retrained by OpenMMLab on open data, through
Hugging Face transformers' ``MMGroundingDinoForObjectDetection``
(https://github.com/huggingface/transformers, Apache-2.0), weights
``openmmlab-community/mm_grounding_dino_*`` (Apache-2.0 model cards).

Adapted from tools/grounding-dino/tools.py of this repository (MIT, itself
from graph-robots/open-robot-skills): the processor -> model ->
``post_process_grounded_object_detection`` path, the period-terminated
query and the detection dicts are the same, so the two tools are
interchangeable in a skill and as a sahi backend. Changed: the checkpoints
are pinned to commits; ``prompts`` takes a list; ``max_detections``; and the
runtime refuses transformers 5.x, which loads these checkpoints without an
error but decodes collapsed boxes (SKILL.md).

The model loads lazily on first call; importing this module never pulls torch
/ transformers. Knobs via env:

- ``GAP_MMGDINO_MODEL``  -- one of ``MODELS`` (pinned revision) or a local directory
  (default ``openmmlab-community/mm_grounding_dino_large_all``).
- ``GAP_MMGDINO_DEVICE`` -- torch device (default ``cuda``).
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
    "openmmlab-community/mm_grounding_dino_large_all": "0445d0a0ae5f0fa6be689cf7243d4d0ef4ba7de0",
    "openmmlab-community/mm_grounding_dino_base_all": "b32293f50a56f7218d9f73ff940ffb9eba84d63c",
    "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det": "ef2053aa1e0cc9d9cee1df31ca919ee9aae4fc8e",
}
DEFAULT_MODEL = "openmmlab-community/mm_grounding_dino_large_all"

#: Box floor. On the 7 station frames (SKILL.md), large on a table ROI: 0.3 gave
#: 6 / 12 small parts, 3 / 4 open boxes and 2 false positives; 0.25 gave 7 / 12,
#: 4 / 4 and 9; grounding-dino's 0.2 gave 8 / 12, 4 / 4 and 18.
DEFAULT_BOX_THRESHOLD = 0.3
#: Token floor that names a box (grounding-dino's default; the station check used it).
DEFAULT_TEXT_THRESHOLD = 0.2
#: The transformers release the bundle is locked to and checked with.
TRANSFORMERS_MAX_MAJOR = 4

_load_lock = threading.Lock()
_runtime: Any = None


class Detection(TypedDict):
    box: BoundingBox2D  # [x1, y1, x2, y2] in pixels
    label: str  # matched text span ("" when no token cleared text_threshold)
    score: float  # box confidence [0, 1]


class DetectResult(TypedDict):
    detections: list[Detection]


def _model_source() -> tuple[str, str | None]:
    name = os.environ.get("GAP_MMGDINO_MODEL") or DEFAULT_MODEL
    if name in MODELS:
        return name, MODELS[name]
    if os.path.isdir(name):
        return name, None
    raise ValueError(
        f"GAP_MMGDINO_MODEL must be one of {sorted(MODELS)} or a local directory, got {name!r}"
    )


def _check_transformers(version: str) -> None:
    major = int(version.split(".", 1)[0])
    if major > TRANSFORMERS_MAX_MAJOR:
        raise PerceptionFailed(
            f"mm-grounding-dino needs transformers < 5 (found {version}): 5.x loads the "
            "MM-Grounding-DINO checkpoints without an error but returns collapsed boxes; "
            "use the bundle's own environment (uv sync --project tools/mm-grounding-dino)"
        )


class _Runtime:
    def __init__(self, source: str, revision: str | None, device: str) -> None:
        import torch
        import transformers
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        _check_transformers(transformers.__version__)
        self.torch, self.device = torch, device
        logger.info("Loading MM-Grounding-DINO %s@%s on %s ...", source, revision, device)
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
            _runtime = _Runtime(source, revision, os.environ.get("GAP_MMGDINO_DEVICE", "cuda"))
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

    logger.info("[mm-grounding-dino] prefetching %s@%s ...", source, revision)
    snapshot_download(repo_id=source, revision=revision, repo_type="model")
    logger.info("[mm-grounding-dino] prefetch complete")


def _query(query: str | None, prompts: list[str] | str | None) -> str:
    """Grounding DINO's period-terminated phrase list from a query string or a list of phrases."""
    if prompts is not None:
        items = prompts.split(".") if isinstance(prompts, str) else list(prompts)
        phrases = [str(p).strip().rstrip(".").strip() for p in items]
        phrases = [p for p in phrases if p]
        if not phrases:
            raise ToolError("mm-grounding-dino.detect", f"no phrase in prompts={prompts!r}")
        return " ".join(f"{p}." for p in phrases)
    if not query or not query.strip(" ."):
        raise ToolError("mm-grounding-dino.detect", "give query ('cube. plug.') or prompts")
    q = query.strip()
    return q if q.endswith(".") else q + "."


@tool(
    name="mm-grounding-dino.detect",
    summary="Open-vocabulary detection with MM-Grounding-DINO (labelled boxes + scores, grounding-dino's shape).",
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
            "mm-grounding-dino.detect",
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
        raise PerceptionFailed(f"MM-Grounding-DINO detection failed: {e}") from e
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
    logger.info("mm-grounding-dino.detect returning %d detections", len(dets))
    return {"detections": dets}
