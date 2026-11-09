"""owlv2 tool bundle -- open-vocabulary detection by text and by exemplar (OWLv2).

OWLv2 (Minderer et al., "Scaling Open-Vocabulary Object Detection", 2023)
through Hugging Face transformers' ``Owlv2ForObjectDetection``
(https://github.com/huggingface/transformers, Apache-2.0,
src/transformers/models/owlv2/modeling_owlv2.py and
image_processing_owlv2.py), weights ``google/owlv2-base-patch16-ensemble`` /
``google/owlv2-large-patch14-ensemble`` (Apache-2.0 model cards).

Adapted from transformers (Apache-2.0, license in LICENSE.transformers):

- ``_decode_boxes`` follows ``Owlv2ImageProcessor.post_process_object_detection``
  and ``_scale_boxes``: centre-to-corner boxes scaled by the padded square's
  side (the processor pads the frame bottom / right to a square before
  resizing). Changed: boxes whose centre falls in the padding are dropped and
  the rest are clipped to the frame; done in numpy.
- ``_pick_exemplar`` is the query-box selection of ``OwlViTForObjectDetection.
  embed_image_query`` (transformers v4.30, owlvit/modeling_owlvit.py): the
  predicted boxes overlapping the query box by at least 0.8 of the best IoU
  are candidates, and the one least similar to the image's mean class
  embedding is the query. Changed: the query box is the caller's (upstream
  used the whole query image), in numpy.
- ``_nms`` is the greedy suppression of ``post_process_image_guided_detection``
  (suppress every lower-scored box over ``nms_iou`` IoU), applied per label to
  the text mode as well.
- ``_Runtime.embed`` is the image half of ``Owlv2ForObjectDetection.
  image_guided_detection`` (image_embedder, class / box / objectness heads),
  so a query embedding can come from a box in any frame, be reused across
  frames and slices, and be passed as a string. ``_Runtime.pixels`` is
  ``Owlv2ImageProcessor``'s preprocessing redone in torch on the device.

Exemplar scores are the cosine similarity of each box's class embedding with
the exemplar's, not the class head's sigmoid (it saturates at 0.99+ for most
boxes of a frame against an image embedding) nor
``post_process_image_guided_detection``'s display alphas (those rescale every
image so its best box scores 1.0, even when nothing in it resembles the
query). The class head's own normalisation is the same cosine before its
learned shift and scale.

The model loads lazily on first call; importing this module never pulls
torch / transformers. Knobs via env:

- ``GAP_OWLV2_MODEL``  -- one of ``MODELS`` (pinned revision) or a local
  directory (default ``google/owlv2-base-patch16-ensemble``).
- ``GAP_OWLV2_DEVICE`` -- torch device (default ``cuda``).
"""

from __future__ import annotations

import base64
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
    "google/owlv2-base-patch16-ensemble": "cfd3195ba4ea9592eec887ded089f4c08eff231d",
    "google/owlv2-large-patch14-ensemble": "95e26936e865f87db1742128404b3c035d47d89d",
}
DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"

#: Text-mode score floor. post_process_object_detection's default is 0.1; on
#: the 7 station frames (SKILL.md) base found the same 7 / 12 small parts at
#: 0.15 and 0.2 with 33 vs 18 false positives, and 56 at 0.1.
DEFAULT_THRESHOLD = 0.2
#: Exemplar-mode floor on the cosine similarity with the exemplar's embedding.
#: Station frames (SKILL.md): a cube from another frame scores 0.89-0.97, the
#: open box of the same grey plastic 0.77-0.88; at 0.87 base found 12 / 12
#: cross-frame cubes with no false positive (0.85: 1 on the open box).
DEFAULT_LIKE_THRESHOLD = 0.87
#: IoU above which a lower-scored box of the same label is suppressed
#: (post_process_image_guided_detection's nms_threshold default).
DEFAULT_NMS_IOU = 0.3
#: Text template the ensemble checkpoints were queried with in the model
#: card / transformers examples ("a photo of a cat").
DEFAULT_TEMPLATE = "a photo of a {}"
#: Upstream's candidate rule: boxes within this share of the best IoU with the query box.
EXEMPLAR_IOU_SHARE = 0.8
#: A detection overlapping the exemplar's own box by this IoU is the exemplar
#: itself (``exclude_query``).
SELF_IOU = 0.5
#: Prefix of the string form of an exemplar embedding (``owlv2.embed_exemplar``).
TOKEN_PREFIX = "owlv2-exemplar"

_load_lock = threading.Lock()
_runtime: Any = None


class Detection(TypedDict):
    box: BoundingBox2D  # [x1, y1, x2, y2] in pixels
    label: str  # the prompt it matched ("exemplar" / ``label`` in exemplar mode)
    score: float  # class-head sigmoid [0, 1]


class DetectResult(TypedDict):
    detections: list[Detection]


class ExemplarInfo(TypedDict):
    box: BoundingBox2D  # the model's box the embedding was taken from (query image pixels)
    iou: float  # its IoU with the query box (1.0 when no box was given)
    objectness: float  # OWLv2's objectness sigmoid for that box


class EmbedResult(TypedDict):
    query: str  # pass to owlv2.detect_like(query=...) or as a sahi prompt
    exemplar: ExemplarInfo
    dim: int


class DetectLikeResult(TypedDict):
    detections: list[Detection]
    boxes: list[BoundingBox2D]
    scores: list[float]
    found: bool
    count: int
    exemplar: ExemplarInfo | None  # None when the query came as a string


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _model_source() -> tuple[str, str | None]:
    name = os.environ.get("GAP_OWLV2_MODEL") or DEFAULT_MODEL
    if name in MODELS:
        return name, MODELS[name]
    if os.path.isdir(name):
        return name, None
    raise ValueError(
        f"GAP_OWLV2_MODEL must be one of {sorted(MODELS)} or a local directory, got {name!r}"
    )


class _Runtime:
    """One Owlv2ForObjectDetection on the device; numpy in, numpy out."""

    def __init__(self, source: str, revision: str | None, device: str) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = device
        logger.info("Loading OWLv2 %s@%s on %s ...", source, revision, device)
        self.processor = Owlv2Processor.from_pretrained(source, revision=revision)
        self.model = Owlv2ForObjectDetection.from_pretrained(source, revision=revision).to(device)
        self.model.eval()
        ip = self.processor.image_processor
        self.side = int(ip.size["height"])
        self.mean = torch.tensor(ip.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=device).view(1, 3, 1, 1)
        self.hf_preprocess = os.environ.get("GAP_OWLV2_PREPROCESS", "device") == "hf"

    def pixels(self, image: np.ndarray) -> Any:
        """``Owlv2ImageProcessor``'s steps on the device: rescale, pad bottom / right to a square
        with 0, Gaussian anti-alias (only when shrinking), bilinear resize, CLIP normalise.

        The processor runs them on the CPU (scipy / PIL without torchvision): 0.27 s of a 0.35 s
        call on a 640 x 480 frame; ``GAP_OWLV2_PREPROCESS=hf`` uses it instead.
        """
        torch = self.torch
        if self.hf_preprocess:
            from PIL import Image

            px = self.processor(images=Image.fromarray(image, "RGB"), return_tensors="pt")
            return px["pixel_values"].to(self.device)
        F = torch.nn.functional
        x = torch.tensor(image, device=self.device).permute(2, 0, 1)[None].float() / 255.0
        h, w = x.shape[-2:]
        s = max(h, w)
        x = F.pad(x, (0, s - w, 0, s - h), value=0.0)
        sigma = max((s / self.side - 1.0) / 2.0, 0.0)
        if sigma > 0:
            r = int(np.ceil(3 * sigma))
            t = torch.arange(-r, r + 1, device=self.device, dtype=x.dtype)
            k = torch.exp(-0.5 * (t / sigma) ** 2)
            k = (k / k.sum()).view(1, 1, 1, -1).repeat(3, 1, 1, 1)
            x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k, groups=3)
            x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.transpose(2, 3), groups=3)
        x = F.interpolate(x, size=(self.side, self.side), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def text(self, image: np.ndarray, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Class logits ``[N, Q]`` and boxes ``[N, 4]`` (cx, cy, w, h over the padded square)."""
        tok = self.processor(text=[texts], return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(
                input_ids=tok["input_ids"].to(self.device),
                attention_mask=tok["attention_mask"].to(self.device),
                pixel_values=self.pixels(image),
            )
        return (
            out.logits[0].float().cpu().numpy(),
            out.pred_boxes[0].float().cpu().numpy(),
        )

    def embed(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Class embeddings ``[N, D]``, boxes ``[N, 4]`` (as ``text``) and objectness logits ``[N]``."""
        with self.torch.no_grad():
            fmap = self.model.image_embedder(pixel_values=self.pixels(image))[0]
            b, h, w, d = fmap.shape
            feats = fmap.reshape(b, h * w, d)
            _, class_embeds = self.model.class_predictor(feats)
            boxes = self.model.box_predictor(feats, fmap)
            objectness = self.model.objectness_predictor(feats)
        return (
            class_embeds[0].float().cpu().numpy(),
            boxes[0].float().cpu().numpy(),
            objectness[0].float().cpu().numpy(),
        )


def _get_runtime() -> Any:
    global _runtime
    with _load_lock:
        if _runtime is None:
            source, revision = _model_source()
            _runtime = _Runtime(source, revision, os.environ.get("GAP_OWLV2_DEVICE", "cuda"))
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

    logger.info("[owlv2] prefetching %s@%s ...", source, revision)
    snapshot_download(repo_id=source, revision=revision, repo_type="model")
    logger.info("[owlv2] prefetch complete")


# ---------------------------------------------------------------------------
# Geometry and post-processing (numpy)
# ---------------------------------------------------------------------------


def _image(image: Any, name: str) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(image))
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ToolError(name, f"expected RGB uint8 [H, W, 3] image, got {arr.dtype} {arr.shape}")
    return arr


def _box_xyxy(box: Any, name: str) -> np.ndarray:
    if isinstance(box, dict):
        b = np.array([box["x1"], box["y1"], box["x2"], box["y2"]], dtype=np.float64)
    else:
        b = np.asarray(box, dtype=np.float64).reshape(-1)
        if b.size != 4:
            raise ToolError(name, f"box must be [x1, y1, x2, y2] or a BoundingBox2D, got {box!r}")
    if b[2] <= b[0] or b[3] <= b[1]:
        raise ToolError(name, f"box {b.tolist()} is empty")
    return b


def _box_dict(b: np.ndarray) -> BoundingBox2D:
    return {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _decode_boxes(cxcywh: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame-pixel ``xyxy`` boxes and a mask of those whose centre lies in the frame (not the padding)."""
    side = float(max(width, height))
    b = np.asarray(cxcywh, dtype=np.float64).reshape(-1, 4) * side
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    inside = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
    xyxy = np.clip(xyxy, 0.0, [width, height, width, height])
    return xyxy, inside


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou: float) -> np.ndarray:
    """Indices kept, best-first: greedy suppression of lower-scored boxes of the same label over ``iou``."""
    order = np.argsort(-scores, kind="stable")
    if iou >= 1.0:
        return order
    if len(order) <= 2000:
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
    suppressed = np.zeros(len(scores), dtype=bool)
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        same = labels == labels[i]
        over = _iou(boxes[i], boxes) > iou
        suppressed |= same & over
    return np.array(keep, dtype=np.int64)


def _pick_exemplar(
    boxes: np.ndarray, class_embeds: np.ndarray, query_box: np.ndarray, valid: np.ndarray
) -> tuple[int, float]:
    """Index of the box whose embedding is the query, and its IoU with ``query_box``."""
    ious = np.where(valid, _iou(query_box, boxes), 0.0)
    best = float(ious.max()) if len(ious) else 0.0
    if best <= 0.0:
        return -1, 0.0
    cand = np.flatnonzero(ious >= EXEMPLAR_IOU_SHARE * best)
    mean = class_embeds.mean(axis=0)
    sim = class_embeds[cand] @ mean
    i = int(cand[int(np.argmin(sim))])
    return i, float(ious[i])


def _prompts(prompts: Any, query: Any, name: str) -> list[str]:
    items: list[str] = []
    for src in (prompts, query):
        if src is None:
            continue
        for p in [src] if isinstance(src, str) else list(src):
            items += str(p).split(".")
    out: list[str] = []
    for p in items:
        p = p.strip()
        if p and p not in out:
            out.append(p)
    if not out:
        raise ToolError(name, "give at least one object to look for (prompts or query)")
    return out


def _encode_query(vec: np.ndarray) -> str:
    raw = np.asarray(vec, dtype="<f2").tobytes()
    return f"{TOKEN_PREFIX}:{vec.size}:{base64.b64encode(raw).decode('ascii')}"


def _decode_query(token: str, name: str) -> np.ndarray:
    text = str(token).strip().rstrip(".").strip()
    parts = text.split(":")
    if len(parts) != 3 or parts[0] != TOKEN_PREFIX:
        raise ToolError(
            name, f"query must be a string from owlv2.embed_exemplar ({TOKEN_PREFIX}:...)"
        )
    try:
        dim = int(parts[1])
        vec = np.frombuffer(base64.b64decode(parts[2], validate=True), dtype="<f2").astype(
            np.float32
        )
    except (ValueError, TypeError) as exc:
        raise ToolError(name, f"query is not a valid exemplar string: {exc}") from exc
    if vec.size != dim:
        raise ToolError(name, f"exemplar string holds {vec.size} values, says {dim}")
    return vec


def _detections(
    boxes: np.ndarray, scores: np.ndarray, labels: list[str], keep: np.ndarray, max_detections: int
) -> list[Detection]:
    if max_detections > 0:
        keep = keep[:max_detections]
    return [
        {"box": _box_dict(boxes[i]), "label": labels[i], "score": float(scores[i])} for i in keep
    ]


def _embed_exemplar(runtime: Any, image: np.ndarray, query_box: np.ndarray | None, name: str):
    """(embedding, ExemplarInfo, the image's features) for the exemplar ``query_box`` in ``image``."""
    H, W = image.shape[:2]
    class_embeds, raw_boxes, objectness = runtime.embed(image)
    boxes, inside = _decode_boxes(raw_boxes, W, H)
    if query_box is None:
        i = int(np.argmax(np.where(inside, objectness, -np.inf)))
        iou = 1.0
    else:
        qb = np.clip(query_box, 0.0, [W, H, W, H])
        i, iou = _pick_exemplar(boxes, class_embeds, qb, inside)
        if i < 0:
            raise PerceptionFailed(
                f"{name}: no OWLv2 box overlaps the query box {query_box.tolist()}; "
                "is it inside the query image?"
            )
    info: ExemplarInfo = {
        "box": _box_dict(boxes[i]),
        "iou": round(float(iou), 4),
        "objectness": float(_sigmoid(objectness[i])),
    }
    return np.asarray(class_embeds[i], dtype=np.float32), info, (class_embeds, boxes, inside)


def _cosine(class_embeds: np.ndarray, query: np.ndarray) -> np.ndarray:
    e = np.asarray(class_embeds, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    if e.shape[1] != q.size:
        raise ToolError(
            "owlv2.detect_like",
            f"the exemplar has {q.size} dimensions, the loaded model {e.shape[1]} (embedded by another OWLv2 size?)",
        )
    norms = np.linalg.norm(e, axis=1) * max(float(np.linalg.norm(q)), 1e-12)
    return (e @ q) / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="owlv2.detect",
    summary="Open-vocabulary detection from text prompts with OWLv2; labeled boxes with confidence scores (grounding-dino's shape).",
    tags=("perception",),
)
def detect(
    image: np.ndarray,
    prompts: list[str] | str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    query: str | None = None,
    template: str = DEFAULT_TEMPLATE,
    nms_iou: float = DEFAULT_NMS_IOU,
    max_detections: int = 0,
) -> DetectResult:
    """Detect every instance of the prompted objects in an RGB uint8 [H, W, 3] image.

    ``prompts`` is a list of object names or one string; strings are split on
    ``.`` (grounding-dino's ``"red cube. plug."`` works, and so does sahi's
    generic backend call ``query="cube. plug"``). Each prompt is embedded as
    ``template.format(prompt)``; every box is labelled with the prompt it
    scores highest on (the bare prompt, not the template) and kept when that
    score clears ``threshold``. Boxes of one label overlapping by more than
    ``nms_iou`` IoU are suppressed (1.0 keeps all). Best-first; an empty list
    means nothing cleared the threshold.
    """
    name = "owlv2.detect"
    arr = _image(image, name)
    names = _prompts(prompts, query, name)
    texts = [template.format(p) if "{}" in template else p for p in names]
    H, W = arr.shape[:2]
    try:
        runtime = _get_runtime()
        logits, raw_boxes = runtime.text(arr, texts)
    except Exception as e:
        raise PerceptionFailed(f"OWLv2 detection failed: {e}") from e
    logits = np.asarray(logits, dtype=np.float64).reshape(len(raw_boxes), -1)
    best = logits.argmax(axis=1)
    scores = _sigmoid(logits[np.arange(len(best)), best])
    boxes, inside = _decode_boxes(raw_boxes, W, H)
    sel = np.flatnonzero(
        inside & (scores > threshold) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    )
    keep = sel[_nms(boxes[sel], scores[sel], best[sel], nms_iou)] if len(sel) else sel
    labels = [names[j] for j in best]
    dets = _detections(boxes, scores, labels, keep, int(max_detections))
    logger.info("owlv2.detect returning %d detections.", len(dets))
    return {"detections": dets}


@tool(
    name="owlv2.embed_exemplar",
    summary="OWLv2 query embedding of one object (a box in a frame, or a crop) as a string for owlv2.detect_like or sahi.",
    tags=("perception",),
)
def embed_exemplar(
    query_image: np.ndarray, query_box: BoundingBox2D | list[float] | None = None
) -> EmbedResult:
    """Embed the object in ``query_box`` of ``query_image`` (or the most object-like box of a crop when no box).

    The returned ``query`` string carries the embedding (float16, base64):
    give it to ``owlv2.detect_like(query=...)`` on other frames, or to
    ``sahi.detect_sliced(prompts=[query], backend="owlv2.detect_like")`` to
    search sliced frames with one embedding.
    """
    name = "owlv2.embed_exemplar"
    arr = _image(query_image, name)
    qb = None if query_box is None else _box_xyxy(query_box, name)
    try:
        vec, info, _ = _embed_exemplar(_get_runtime(), arr, qb, name)
    except PerceptionFailed:
        raise
    except Exception as e:
        raise PerceptionFailed(f"OWLv2 exemplar embedding failed: {e}") from e
    return {"query": _encode_query(vec), "exemplar": info, "dim": int(vec.size)}


@tool(
    name="owlv2.detect_like",
    summary="Image-guided one-shot detection with OWLv2: every object in the image that looks like the exemplar (a box in a frame, a crop, or an embedding string).",
    tags=("perception",),
)
def detect_like(
    image: np.ndarray,
    query_image: np.ndarray | None = None,
    query_box: BoundingBox2D | list[float] | None = None,
    threshold: float = DEFAULT_LIKE_THRESHOLD,
    query: str | None = None,
    nms_iou: float = DEFAULT_NMS_IOU,
    exclude_query: bool = False,
    label: str = "exemplar",
    max_detections: int = 0,
) -> DetectLikeResult:
    """Find every object in ``image`` that looks like the exemplar: "all the bricks like this one".

    The exemplar is, in order of precedence: ``query`` (a string from
    ``owlv2.embed_exemplar``; sahi passes its prompt here), ``query_box`` in
    ``query_image``, ``query_box`` in ``image`` itself (``query_image`` None),
    or ``query_image`` alone as a crop of the object. The query embedding is
    the class embedding of OWLv2's box that best covers ``query_box``
    (transformers' OWL-ViT rule); ``exemplar.iou`` says how well it does --
    under ~0.3 the model saw no object there and the search is unreliable.

    Scores are the class head's sigmoid against that embedding; boxes over
    ``threshold`` are kept and suppressed by ``nms_iou`` IoU. With the
    exemplar in ``image`` itself it is found too; ``exclude_query`` drops the
    detections overlapping it by IoU >= 0.5 ("find the others").
    """
    name = "owlv2.detect_like"
    arr = _image(image, name)
    H, W = arr.shape[:2]
    qb = None if query_box is None else _box_xyxy(query_box, name)
    same_image = False
    try:
        runtime = _get_runtime()
        info: ExemplarInfo | None = None
        if query:
            vec = _decode_query(query, name)
            embeds, raw_boxes, _ = runtime.embed(arr)
            boxes, inside = _decode_boxes(raw_boxes, W, H)
        elif query_image is None and qb is not None:
            same_image = True
            vec, info, (embeds, boxes, inside) = _embed_exemplar(runtime, arr, qb, name)
        elif query_image is not None:
            qimg = _image(query_image, name)
            vec, info, _ = _embed_exemplar(runtime, qimg, qb, name)
            embeds, raw_boxes, _ = runtime.embed(arr)
            boxes, inside = _decode_boxes(raw_boxes, W, H)
        else:
            raise ToolError(name, "give the exemplar: query, query_box, or query_image")
    except (ToolError, PerceptionFailed):
        raise
    except Exception as e:
        raise PerceptionFailed(f"OWLv2 image-guided detection failed: {e}") from e
    scores = _cosine(embeds, vec)
    ok = inside & (scores > threshold) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if exclude_query and same_image and qb is not None:
        ok &= _iou(qb, boxes) < SELF_IOU
    sel = np.flatnonzero(ok)
    zeros = np.zeros(len(sel), dtype=np.int64)
    keep = sel[_nms(boxes[sel], scores[sel], zeros, nms_iou)] if len(sel) else sel
    dets = _detections(boxes, scores, [label] * len(boxes), keep, int(max_detections))
    logger.info("owlv2.detect_like returning %d detections.", len(dets))
    return {
        "detections": dets,
        "boxes": [d["box"] for d in dets],
        "scores": [d["score"] for d in dets],
        "found": bool(dets),
        "count": len(dets),
        "exemplar": info,
    }
