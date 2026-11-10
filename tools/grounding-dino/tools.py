"""grounding-dino tool bundle — zero-shot object detection.

Extracted from the original Grounding DINO gRPC servicer in the dev
tree. The transformers pipeline
(processor → model → post_process_grounded_object_detection) is verbatim;
the proto byte decode/encode is replaced by numpy arrays + gap.types dicts.

The model loads lazily on first call (module-level singleton); importing
this module never pulls torch/transformers. Knobs via env:

- ``GAP_DINO_DEVICE`` — torch device (default ``cuda``).
- ``GAP_DINO_MODEL``  — HuggingFace model name
  (default ``IDEA-Research/grounding-dino-base``, loaded at a pinned commit).

transformers 5.x (5.12.1 and 5.19.0 checked) loads Grounding DINO's decoder
box head wrong without a warning: with ``decoder_bbox_embed_share`` (the
IDEA checkpoints) the shared head keeps its random initialisation, whose
last layer is zero, so the six decoder layers stop refining the boxes; without
it (MM-Grounding-DINO) every layer is tied to layer 0's head.
``_repair_box_heads`` loads the head from the checkpoint file after
``from_pretrained`` and ``_verify_against_checkpoint`` refuses to serve a
model whose tensors still differ from the checkpoint (SKILL.md).
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

_DEFAULT_MODEL_NAME = "IDEA-Research/grounding-dino-base"
#: Commits the bundle loads for the checkpoints it knows (model card: Apache-2.0).
_PINNED = {"IDEA-Research/grounding-dino-base": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"}
_DEFAULT_BOX_THRESHOLD = 0.20
_DEFAULT_TEXT_THRESHOLD = 0.20

_DEVICE = os.environ.get("GAP_DINO_DEVICE", "cuda")
_MODEL_NAME = os.environ.get("GAP_DINO_MODEL", _DEFAULT_MODEL_NAME)
_REVISION = _PINNED.get(_MODEL_NAME)

_load_lock = threading.Lock()
_model: Any = None
_processor: Any = None


class Detection(TypedDict):
    box: BoundingBox2D  # [x1, y1, x2, y2] in pixels
    label: str  # matched text label
    score: float  # confidence [0, 1]


class DetectResult(TypedDict):
    detections: list[Detection]


def weights_cached() -> bool | None:
    """Filesystem-only weight-cache probe for ``gap check``.

    Checks the Hugging Face cache for the configured model's config.json
    (the canonical presence marker) — never downloads, never imports
    torch. ``None`` when huggingface_hub is unavailable ("unknown").
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        result = try_to_load_from_cache(_MODEL_NAME, "config.json", revision=_REVISION)
    except Exception:
        return None
    return isinstance(result, str)


def prefetch() -> None:
    """Snapshot-download the configured GDINO weights into the HF cache.

    Called by ``gap skills check --download``. Idempotent: re-running
    against an already-cached snapshot is a near-no-op (HF revision
    check + symlink refresh). Logs the bytes added so the user sees
    progress; raises on network / auth / disk errors so ``gap skills
    check --download`` exits non-zero.

    Uses ``snapshot_download`` rather than ``AutoModel.from_pretrained``
    so we never load torch / instantiate the model at prefetch time —
    important for CI lanes and for the bare-engine venv.
    """
    from huggingface_hub import snapshot_download

    logger.info("[grounding-dino] prefetching weights for %s ...", _MODEL_NAME)
    snapshot_download(repo_id=_MODEL_NAME, revision=_REVISION, repo_type="model")
    logger.info("[grounding-dino] prefetch complete (cached at HF default)")


def _checkpoint_tensors(source: str, revision: str | None) -> dict[str, Any]:
    """Every tensor of the checkpoint's ``model.safetensors``, by name (CPU)."""
    from safetensors.torch import load_file

    if os.path.isdir(source):
        path = os.path.join(source, "model.safetensors")
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(source, "model.safetensors", revision=revision)
    return load_file(path)


def _head_tensors(checkpoint: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """The checkpoint's decoder box heads by layer: ``model.decoder.bbox_embed.<i>.*`` or ``bbox_embed.<i>.*``."""
    import re

    heads: dict[int, dict[str, Any]] = {}
    for key, value in checkpoint.items():
        m = re.match(r"(?:model\.decoder\.)?bbox_embed\.(\d+)\.(.+)$", key)
        if m:
            heads.setdefault(int(m.group(1)), {})[m.group(2)] = value
    return heads


def _head_plan(share: bool, n_layers: int, heads: dict[int, dict[str, Any]]) -> list[int]:
    """Which checkpoint head each of the ``n_layers`` decoder layers loads.

    ``decoder_bbox_embed_share``: one module for every layer, head 0 (the IDEA checkpoints store
    only that one). Otherwise one module per layer, its own head.
    """
    if not heads:
        return []
    if share:
        if 0 not in heads:
            raise PerceptionFailed("grounding-dino: the checkpoint has no decoder box head 0")
        return [0] * n_layers
    missing = [i for i in range(n_layers) if i not in heads]
    if missing:
        raise PerceptionFailed(f"grounding-dino: the checkpoint lacks decoder box heads {missing}")
    return list(range(n_layers))


def _repair_box_heads(model: Any, checkpoint: dict[str, Any]) -> bool:
    """Load the decoder's box heads from ``checkpoint`` into ``model`` (True when it did).

    Shared heads stay one module; unshared ones get a module per layer (untied, so loading
    layer 5 does not overwrite layer 0). The model's own ``bbox_embed`` and the decoder's
    are pointed at the same list, as transformers builds them.
    """
    import copy

    import torch

    heads = _head_tensors(checkpoint)
    if not heads or not hasattr(model, "bbox_embed"):
        return False
    n = len(model.bbox_embed)
    share = bool(getattr(model.config, "decoder_bbox_embed_share", True))
    plan = _head_plan(share, n, heads)
    if share:
        module = model.bbox_embed[0]
        module.load_state_dict(heads[0], strict=True)
        new = torch.nn.ModuleList([module] * n)
    else:
        new = torch.nn.ModuleList([copy.deepcopy(model.bbox_embed[0]) for _ in range(n)])
        for i, h in enumerate(plan):
            new[i].load_state_dict(heads[h], strict=True)
    model.bbox_embed = new
    model.model.decoder.bbox_embed = new
    return True


def _verify_against_checkpoint(model: Any, checkpoint: dict[str, Any]) -> list[str]:
    """Names of checkpoint tensors the loaded model holds with other values (same name and shape)."""
    import torch

    own = model.state_dict()
    bad = []
    for key, value in checkpoint.items():
        mine = own.get(key)
        if mine is None or tuple(mine.shape) != tuple(value.shape):
            continue
        if not torch.equal(mine.detach().cpu(), value.to(mine.dtype)):
            bad.append(key)
    return bad


def _get_model() -> tuple[Any, Any]:
    """Load the Grounding DINO model + processor once (lazy singleton)."""
    global _model, _processor
    with _load_lock:
        if _model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            logger.info("Loading Grounding DINO model: %s on %s ...", _MODEL_NAME, _DEVICE)
            _processor = AutoProcessor.from_pretrained(_MODEL_NAME, revision=_REVISION)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                _MODEL_NAME, revision=_REVISION
            )
            try:
                checkpoint = _checkpoint_tensors(_MODEL_NAME, _REVISION)
            except Exception as e:  # e.g. a model with only pytorch_model.bin
                logger.warning(
                    "grounding-dino: cannot read %s's checkpoint to check it: %s", _MODEL_NAME, e
                )
                checkpoint = {}
            bad = _verify_against_checkpoint(model, checkpoint)
            repaired = bool(bad) and _repair_box_heads(model, checkpoint)
            if repaired:
                bad = _verify_against_checkpoint(model, checkpoint)
            if bad:
                raise PerceptionFailed(
                    f"grounding-dino: {len(bad)} tensors of {_MODEL_NAME} differ from its checkpoint "
                    f"after loading (e.g. {bad[:3]}); this transformers release loads it wrong"
                )
            if repaired:
                logger.info("grounding-dino: decoder box head reloaded from the checkpoint")
            model = model.to(_DEVICE)
            model.eval()
            _model = model
            logger.info("Grounding DINO model loaded successfully on %s.", _DEVICE)
        return _model, _processor


@tool(
    name="grounding-dino.detect",
    summary="Zero-shot object detection from a text prompt; returns labeled boxes with confidence scores.",
    tags=("perception",),
)
def detect(
    image: np.ndarray,
    query: str,
    box_threshold: float = _DEFAULT_BOX_THRESHOLD,
    text_threshold: float = _DEFAULT_TEXT_THRESHOLD,
) -> DetectResult:
    """Detect objects matching ``query`` in an RGB uint8 [H, W, 3] image.

    Grounding DINO expects period-separated phrases (``"red cube. green
    cube."``); a missing trailing period is appended automatically. Returns
    an empty detections list when nothing clears the thresholds — select the
    best box downstream (e.g. closest to a pointing-model pixel) and feed it
    to ``sam3.segment_box`` for a pixel-accurate mask.
    """
    import torch
    from PIL import Image

    model, processor = _get_model()

    arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ToolError(
            "grounding-dino.detect",
            f"expected RGB uint8 [H, W, 3] image, got shape {arr.shape}",
        )
    pil_image = Image.fromarray(arr, "RGB")

    # GDINO requires period-terminated phrases
    text_prompt = query
    if not text_prompt.endswith("."):
        text_prompt = text_prompt + "."

    # Defaults if zero/unset (mirrors the servicer's proto-default handling)
    box_threshold = box_threshold if box_threshold > 0 else _DEFAULT_BOX_THRESHOLD
    text_threshold = text_threshold if text_threshold > 0 else _DEFAULT_TEXT_THRESHOLD

    try:
        inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]
    except Exception as e:
        raise PerceptionFailed(f"Grounding DINO detection failed: {e}") from e

    detections: list[Detection] = []
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]

    for box, score, label in zip(boxes, scores, labels, strict=False):
        detections.append(
            {
                "box": {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                },
                "label": str(label),
                "score": float(score),
            }
        )

    logger.info("grounding-dino.detect returning %d detections.", len(detections))
    return {"detections": detections}
