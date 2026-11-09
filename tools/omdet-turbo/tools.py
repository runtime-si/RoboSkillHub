"""omdet-turbo tool bundle -- fast open-vocabulary detection (OmDet-Turbo).

OmDet-Turbo (Zhao et al., "Real-time Transformer-based Open-Vocabulary
Detection with Efficient Fusion Head", 2024): a Swin-T backbone, a CLIP text
encoder and an efficient fusion head, through Hugging Face transformers'
``OmDetTurboForObjectDetection`` (https://github.com/huggingface/transformers,
Apache-2.0, src/transformers/models/omdet_turbo/), weights
``omlab/omdet-turbo-swin-tiny-hf`` (Apache-2.0 model card; original code
om-ai-lab/OmDet, Apache-2.0).

Adapted from transformers (Apache-2.0, license in LICENSE.transformers):

- ``_decode`` follows ``OmDetTurboProcessor.post_process_grounded_object_detection``
  and ``_post_process_boxes_for_image``: per-query sigmoid scores of every
  class, the top ``num_queries`` (query, class) pairs, centre-to-corner boxes
  scaled to the frame, a score floor, class-wise NMS, then clipping to the
  frame. Changed: done in numpy (no torchvision ``batched_nms``), boxes are
  returned as gap ``BoundingBox2D``, labels are the caller's prompts, and
  ``max_detections`` caps the result after NMS (upstream's ``max_num_det``
  caps the top-k before it).
- The task sentence defaults to upstream's ``"Detect {a}, {b}."``.

The model loads lazily on first call; importing this module never pulls torch
/ transformers. The class-prompt embeddings are cached by the model (its LRU
language cache), so repeating the same prompts costs only the image pass.
Knobs via env:

- ``GAP_OMDET_MODEL``  -- ``omlab/omdet-turbo-swin-tiny-hf`` (pinned revision) or a local directory.
- ``GAP_OMDET_DEVICE`` -- torch device (default ``cuda``).
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

#: Hugging Face repo -> the commit the bundle loads (model card: Apache-2.0).
MODELS = {"omlab/omdet-turbo-swin-tiny-hf": "7fe93cecfb770c4d76cf71163956221249cab566"}
DEFAULT_MODEL = "omlab/omdet-turbo-swin-tiny-hf"

#: Score floor. Upstream's post-process default is 0.3; on the 7 station frames
#: (SKILL.md) 0.2 kept the same small-part hits as 0.1 with fewer false positives.
DEFAULT_THRESHOLD = 0.2
#: Class-wise NMS IoU (upstream post-process default).
DEFAULT_NMS_IOU = 0.5

_load_lock = threading.Lock()
_runtime: Any = None


class Detection(TypedDict):
    box: BoundingBox2D  # [x1, y1, x2, y2] in pixels
    label: str  # the prompt it matched
    score: float  # sigmoid of the fusion head's class logit [0, 1]


class DetectResult(TypedDict):
    detections: list[Detection]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _model_source() -> tuple[str, str | None]:
    name = os.environ.get("GAP_OMDET_MODEL") or DEFAULT_MODEL
    if name in MODELS:
        return name, MODELS[name]
    if os.path.isdir(name):
        return name, None
    raise ValueError(
        f"GAP_OMDET_MODEL must be one of {sorted(MODELS)} or a local directory, got {name!r}"
    )


class _Runtime:
    """One OmDetTurboForObjectDetection on the device; numpy in, numpy out."""

    def __init__(self, source: str, revision: str | None, device: str) -> None:
        import torch
        from transformers import AutoProcessor, OmDetTurboForObjectDetection

        self.torch = torch
        self.device = device
        logger.info("Loading OmDet-Turbo %s@%s on %s ...", source, revision, device)
        self.processor = AutoProcessor.from_pretrained(source, revision=revision)
        self.model = OmDetTurboForObjectDetection.from_pretrained(source, revision=revision)
        self.model = self.model.to(device).eval()

    def __call__(
        self, image: np.ndarray, prompts: list[str], task: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Class logits ``[Q, C]`` and boxes ``[Q, 4]`` (cx, cy, w, h normalised to the frame)."""
        from PIL import Image

        inputs = self.processor(
            images=Image.fromarray(image, "RGB"), text=list(prompts), task=task, return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        return (
            out.decoder_class_logits[0].float().cpu().numpy(),
            out.decoder_coord_logits[0].float().cpu().numpy(),
        )


def _get_runtime() -> Any:
    global _runtime
    with _load_lock:
        if _runtime is None:
            source, revision = _model_source()
            _runtime = _Runtime(source, revision, os.environ.get("GAP_OMDET_DEVICE", "cuda"))
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

    logger.info("[omdet-turbo] prefetching %s@%s ...", source, revision)
    snapshot_download(repo_id=source, revision=revision, repo_type="model")
    logger.info("[omdet-turbo] prefetch complete")


# ---------------------------------------------------------------------------
# Post-processing (numpy)
# ---------------------------------------------------------------------------


def _image(image: Any) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(image))
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ToolError(
            "omdet-turbo.detect", f"expected RGB uint8 [H, W, 3] image, got {arr.dtype} {arr.shape}"
        )
    return arr


def _prompts(prompts: Any, query: str | None) -> list[str]:
    """A list of class phrases from a list or from grounding-dino's ``"a. b."`` / ``"a, b"`` string."""
    raw = prompts if prompts is not None else query
    if raw is None:
        raise ToolError("omdet-turbo.detect", "give prompts (a list of phrases) or query")
    if isinstance(raw, str):
        sep = "." if "." in raw else ","
        raw = raw.split(sep)
    out: list[str] = []
    for p in raw:
        p = str(p).strip().rstrip(".").strip()
        if p and p not in out:
            out.append(p)
    if not out:
        raise ToolError("omdet-turbo.detect", f"no phrase in prompts={prompts!r} query={query!r}")
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou: float) -> np.ndarray:
    """Indices kept, best-first: greedy suppression of lower-scored boxes of the same label over ``iou``."""
    order = np.argsort(-scores, kind="stable")
    if iou >= 1.0 or len(order) == 0:
        return order
    b, lab = boxes[order], labels[order]
    ix = np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0])
    iy = np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1])
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
    areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    over = inter / np.maximum(areas[:, None] + areas[None, :] - inter, 1e-9) > iou
    over &= lab[:, None] == lab[None, :]
    kept = np.ones(len(order), dtype=bool)
    for i in range(len(order)):
        if kept[i]:
            kept[i + 1 :] &= ~over[i, i + 1 :]
    return order[kept]


def _decode(
    logits: np.ndarray,
    cxcywh: np.ndarray,
    n_classes: int,
    width: int,
    height: int,
    threshold: float,
    nms_iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame-pixel ``xyxy`` boxes, scores and class indices, best-first (upstream's post-process)."""
    logits = np.asarray(logits, dtype=np.float64)[:, :n_classes]
    n_q = logits.shape[0]
    scores = _sigmoid(logits).reshape(-1)
    top = np.argsort(-scores, kind="stable")[:n_q]
    sc = scores[top]
    q, cls = np.divmod(top, n_classes)
    b = np.asarray(cxcywh, dtype=np.float64).reshape(-1, 4)[q]
    cx, cy, w, h = b[:, 0] * width, b[:, 1] * height, b[:, 2] * width, b[:, 3] * height
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep = sc > threshold
    xyxy, sc, cls = xyxy[keep], sc[keep], cls[keep]
    order = _nms(xyxy, sc, cls, nms_iou)
    xyxy = np.clip(xyxy[order], 0.0, [width, height, width, height])
    return xyxy, sc[order], cls[order]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(
    name="omdet-turbo.detect",
    summary="Fast open-vocabulary detection from text prompts (labelled boxes + scores, grounding-dino's shape).",
    tags=("perception",),
)
def detect(
    image: np.ndarray,
    prompts: list[str] | str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    query: str | None = None,
    nms_iou: float = DEFAULT_NMS_IOU,
    task: str | None = None,
    max_detections: int = 0,
) -> DetectResult:
    """Boxes of every ``prompts`` phrase in an RGB uint8 ``[H, W, 3]`` image, best-first.

    ``prompts`` is a list of phrases (``["cube", "plug"]``) or one string,
    ``"cube. plug."`` (grounding-dino's form) or ``"cube, plug"``; ``query`` is
    the same string (the name ``sahi.detect_sliced`` passes). ``label`` is the
    phrase as given. ``threshold`` is the floor on the class sigmoid;
    ``nms_iou`` suppresses lower-scored boxes of the same label;
    ``task`` overrides the sentence the fusion head is conditioned on
    (default ``"Detect a, b."``); ``max_detections`` > 0 keeps the best N.
    """
    arr = _image(image)
    names = _prompts(prompts, query)
    if not 0.0 <= threshold < 1.0:
        raise ToolError("omdet-turbo.detect", f"threshold must be in [0, 1), got {threshold}")
    sentence = task or f"Detect {', '.join(names)}."
    h, w = arr.shape[:2]
    try:
        logits, boxes = _get_runtime()(arr, names, sentence)
    except ToolError:
        raise
    except Exception as e:
        raise PerceptionFailed(f"OmDet-Turbo detection failed: {e}") from e
    xyxy, scores, cls = _decode(logits, boxes, len(names), w, h, threshold, nms_iou)
    if max_detections > 0:
        xyxy, scores, cls = xyxy[:max_detections], scores[:max_detections], cls[:max_detections]
    dets: list[Detection] = [
        {
            "box": {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])},
            "label": names[int(c)],
            "score": float(s),
        }
        for b, s, c in zip(xyxy, scores, cls, strict=True)
    ]
    logger.info("omdet-turbo.detect returning %d detections", len(dets))
    return {"detections": dets}
