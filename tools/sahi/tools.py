"""sahi tool bundle: slicing-aided detection of small parts with the hub's detectors.

Adapted from SAHI (https://github.com/obss/sahi at commit
10061afd5fa076e00f8eda55e2ee290f9d72f3c1, release 0.14.0), MIT License,
Copyright (c) 2020 obss; the licence is tools/sahi/LICENSE.sahi. Ported:

- :func:`_slice_bboxes` from ``sahi/slicing.py`` ``get_slice_bboxes`` (the
  explicit slice size branch);
- the merge from ``sahi/postprocess/_numpy_backend.py``
  (``_compute_metric_matrix_full``, ``_score_tiebreak_order``,
  ``nms_from_matrix``, ``greedy_nmm_from_matrix``, ``nmm_from_matrix``),
  ``sahi/postprocess/combine.py`` (``_batched_apply``, ``_apply_merge``) and
  ``sahi/postprocess/utils.py`` (``has_match``, ``merge_object_prediction_pair``:
  union box, higher score, the higher-scored label, union mask);
- the flow of ``sahi/predict.py`` ``get_sliced_prediction``: every slice, then
  the whole image when there is more than one slice (``perform_standard_pred``),
  everything shifted to frame pixels, then one postprocess over all of it with
  SAHI's defaults (GREEDYNMM, IOS, 0.5, per label).

Changed: no sahi package (it pulls shapely, OpenCV and an optional AGPL
ultralytics backend), no torch: dense numpy matrices only (the sparse, numba
and torchvision backends are dropped; a station frame gives hundreds of raw
boxes, not thousands); masks are unioned as pixel arrays instead of shapely
polygons; the detector is a hub tool called through ``ctx`` instead of a
``DetectionModel`` subclass. Added: a region of interest, per-box filters
before the merge (score floor, size bounds, a box covering most of the slice
it came from), labels snapped to the prompts. Not copied: SAHI's silent switch
to NMS / IOU when the model's confidence threshold is below 0.1.

The bundle runs in process (no ``gap.serving``): ``detect_sliced`` takes the
node's ``ctx`` and calls the backend tool once per slice (``sam3``: once per
slice and prompt), so the backend bundle must be installed and is what needs
the GPU. ``slice_windows`` and ``merge`` are pure numpy.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import tool
from gap_core.types import BoundingBox2D, Mask

logger = logging.getLogger(__name__)

#: Backend name -> the tool it calls. Any other name containing a dot is called
#: as a tool taking ``(image, query)`` and returning ``{detections: [...]}``
#: (``locate-anything.ground``, ``gemini-er.detect``).
BACKEND_TOOLS = {
    "grounding-dino": "grounding-dino.detect",
    "sam3": "sam3.segment_text",
    "locate-anything": "locate-anything.detect",
}
POSTPROCESS_TYPES = ("GREEDYNMM", "NMM", "NMS")
MATCH_METRICS = ("IOS", "IOU")

#: Slice side in pixels: 6 slices on a 640 x 480 frame with the overlap below.
#: grounding-dino upsamples a 320 px slice to its 800 px short side (x2.5) and
#: found the three 16-22 px cubes at ~1 m with it; SAM3 (x3.2) needed 200-256 px
#: slices for them (SKILL.md).
DEFAULT_SLICE_PX = 320
#: Overlap of neighbouring slices as a share of the slice (SAHI's default 0.2).
DEFAULT_OVERLAP = 0.2
#: SAHI get_sliced_prediction defaults.
DEFAULT_POSTPROCESS = "GREEDYNMM"
DEFAULT_MATCH_METRIC = "IOS"
DEFAULT_MATCH_THRESHOLD = 0.5
#: A box covering more than this share of the image it was found in (a slice,
#: or the whole image / ROI) is dropped: open-vocabulary detectors asked for a
#: part a slice does not hold tend to box the whole slice. Measured: SKILL.md.
DEFAULT_MAX_BOX_FRACTION = 0.5
#: Boxes whose longer side is under this many frame pixels are dropped (a
#: 1-2 px sliver left by clipping at a slice edge).
DEFAULT_MIN_BOX_PX = 3.0
#: Score floor applied before the merge when ``min_score`` is not given. The
#: sam3 bundle returns every instance down to score 0 (hundreds per slice);
#: unfloored, the greedy IOS merge unions the low-score boxes around a part
#: into one large box and the part is lost (2 of 12 small parts found on the
#: station frames and ~130 false positives a frame, 4-8 of 12 and 1-3 at 0.5;
#: SKILL.md). The other backends filter with their own thresholds.
DEFAULT_MIN_SCORE = {"sam3": 0.5}


class Detection(TypedDict):
    box: BoundingBox2D  # frame pixels
    label: str  # the prompt it matched (the backend's label when none matches)
    score: float
    n_merged: int  # raw boxes merged into this one (1 = not merged)
    sources: list[str]  # "slice" and / or "full" (the whole image / ROI pass)


class DetectTiming(TypedDict):
    backend_s: float  # all backend tool calls, transfers included
    merge_s: float
    total_s: float


class DetectSlicedResult(TypedDict):
    detections: list[Detection]  # best-first; grounding-dino.detect's shape plus two fields
    boxes: list[BoundingBox2D]  # the same detections as parallel lists
    labels: list[str]
    scores: list[float]
    masks: list[Mask] | None  # uint8 [H, W] 0 / 255 per detection (sam3 backend, return_masks)
    found: bool
    count: int
    raw_slice: int  # boxes from the slices that passed the filters, before the merge
    raw_full: int  # same, from the whole-image pass
    n_slices: int
    n_backend_calls: int
    slice_windows: list[BoundingBox2D]  # frame pixels
    backend: str
    timing: DetectTiming


class SliceWindowsResult(TypedDict):
    windows: list[BoundingBox2D]
    count: int


class MergeResult(TypedDict):
    detections: list[Detection]
    count: int


# ---------------------------------------------------------------------------
# Slicing (sahi/slicing.py get_slice_bboxes)
# ---------------------------------------------------------------------------


def _slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
) -> list[list[int]]:
    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                x_max = min(image_width, x_max)
                y_max = min(image_height, y_max)
                x_min = max(0, x_max - slice_width)
                y_min = max(0, y_max - slice_height)
            slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


# ---------------------------------------------------------------------------
# Postprocess (sahi/postprocess/_numpy_backend.py, combine.py, utils.py)
# ---------------------------------------------------------------------------


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(
            np.broadcast(numerator, denominator).shape, dtype=np.result_type(numerator, denominator)
        ),
        where=denominator > 0,
    )


def _metric_matrix(boxes: np.ndarray, match_metric: str) -> np.ndarray:
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    inter_x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    inter_y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    inter_x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    inter_y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    if match_metric == "IOU":
        return _safe_ratio(inter, areas[:, None] + areas[None, :] - inter)
    return _safe_ratio(inter, np.minimum(areas[:, None], areas[None, :]))


def _score_order(boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Score descending, ties broken by coordinates (y2, x2, y1, x1) as SAHI does."""
    return np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0], -scores))


def _nms(matrix: np.ndarray, order: np.ndarray, threshold: float) -> dict[int, list[int]]:
    keep: dict[int, list[int]] = {}
    suppressed = np.zeros(matrix.shape[0], dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        keep[int(idx)] = []
        suppressed |= matrix[idx] >= threshold
    return keep


def _greedy_nmm(matrix: np.ndarray, order: np.ndarray, threshold: float) -> dict[int, list[int]]:
    keep: dict[int, list[int]] = {}
    suppressed = np.zeros(matrix.shape[0], dtype=bool)
    for i, idx in enumerate(order):
        if suppressed[idx]:
            continue
        remaining = order[i + 1 :]
        candidates = remaining[~suppressed[remaining]]
        merged = candidates[matrix[idx, candidates] >= threshold]
        suppressed[merged] = True
        keep[int(idx)] = [int(m) for m in merged]
    return keep


def _nmm(
    matrix: np.ndarray, order: np.ndarray, scores: np.ndarray, boxes: np.ndarray, threshold: float
) -> dict[int, list[int]]:
    """Transitive merge: a box merged into a keeper brings what it matches along."""
    n = len(order)
    lower = scores[:, None] > scores[None, :]
    equal = scores[:, None] == scores[None, :]
    lt = np.zeros((n, n), dtype=bool)
    still_eq = np.ones((n, n), dtype=bool)
    for col in range(4):
        lt |= still_eq & (boxes[:, col][:, None] < boxes[:, col][None, :])
        still_eq &= boxes[:, col][:, None] == boxes[:, col][None, :]
    candidates_of = ~np.eye(n, dtype=bool) & (lower | (equal & ~lt)) & (matrix >= threshold)

    keep: dict[int, list[int]] = {}
    merge_to_keep = np.full(n, -1, dtype=np.intp)
    for idx in order:
        cur = int(idx)
        if merge_to_keep[cur] < 0:
            merge_to_keep[cur] = cur
            keep_idx = cur
            keep[keep_idx] = []
        else:
            keep_idx = int(merge_to_keep[cur])
        for m in np.where(candidates_of[cur])[0]:
            if merge_to_keep[m] < 0:
                keep[keep_idx].append(int(m))
                merge_to_keep[m] = keep_idx
    return keep


def _keep_to_merge(
    boxes: np.ndarray,
    scores: np.ndarray,
    postprocess: str,
    match_metric: str,
    match_threshold: float,
) -> dict[int, list[int]]:
    matrix = _metric_matrix(boxes, match_metric)
    order = _score_order(boxes, scores)
    if postprocess == "NMS":
        return _nms(matrix, order, match_threshold)
    if postprocess == "NMM":
        return _nmm(matrix, order, scores, boxes, match_threshold)
    return _greedy_nmm(matrix, order, match_threshold)


def _has_match(a: np.ndarray, b: np.ndarray, match_metric: str, threshold: float) -> bool:
    return float(_metric_matrix(np.stack([a, b]), match_metric)[0, 1]) >= threshold


def _merge_pair(p1: dict[str, Any], p2: dict[str, Any]) -> dict[str, Any]:
    box = np.concatenate(
        [np.minimum(p1["box"][:2], p2["box"][:2]), np.maximum(p1["box"][2:], p2["box"][2:])]
    )
    hi = p1 if p1["score"] > p2["score"] else p2
    pieces = p1["pieces"] + p2["pieces"] if p1["pieces"] and p2["pieces"] else []
    return {
        "box": box,
        "score": max(p1["score"], p2["score"]),
        "label": hi["label"],
        "pieces": pieces,
        "n": p1["n"] + p2["n"],
        "sources": p1["sources"] | p2["sources"],
    }


def _merge_predictions(
    preds: list[dict[str, Any]],
    postprocess: str,
    match_metric: str,
    match_threshold: float,
    class_agnostic: bool,
) -> list[dict[str, Any]]:
    if len(preds) <= 1:
        return list(preds)
    preds = list(preds)
    # float32 as SAHI's ObjectPredictionList.tonumpy: an IOS of exactly 0.5 must decide the same way.
    boxes = np.array([p["box"] for p in preds], dtype=np.float32)
    scores = np.array([p["score"] for p in preds], dtype=np.float32)
    groups = (
        [np.arange(len(preds))]
        if class_agnostic
        else [
            np.array([i for i, p in enumerate(preds) if p["label"] == lab])
            for lab in sorted({p["label"] for p in preds})
        ]
    )
    out: list[dict[str, Any]] = []
    for idx in groups:
        local = _keep_to_merge(boxes[idx], scores[idx], postprocess, match_metric, match_threshold)
        for k, merge_list in local.items():
            keep_i = int(idx[k])
            keep = preds[keep_i]
            for m in merge_list:
                other = preds[int(idx[m])]
                if _has_match(keep["box"], other["box"], match_metric, match_threshold):
                    keep = _merge_pair(keep, other)
                    preds[keep_i] = keep
            out.append(keep)
    out.sort(key=lambda p: -p["score"])
    return out


# ---------------------------------------------------------------------------
# Inputs, outputs
# ---------------------------------------------------------------------------


def _image(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ToolError("sahi", f"expected RGB uint8 [H, W, 3] image, got {arr.dtype} {arr.shape}")
    return arr


def _prompts(prompts: Any) -> list[str]:
    items = [prompts] if isinstance(prompts, str) else list(prompts or [])
    out = []
    for p in items:
        p = str(p).strip().rstrip(".").strip()
        if p and p not in out:
            out.append(p)
    if not out:
        raise ToolError("sahi", "prompts must name at least one object")
    return out


def _box_xyxy(box: Any) -> np.ndarray:
    if isinstance(box, dict):
        return np.array([box["x1"], box["y1"], box["x2"], box["y2"]], dtype=np.float64)
    return np.asarray(box, dtype=np.float64).reshape(4)


def _box_dict(b: np.ndarray) -> BoundingBox2D:
    return {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])}


def _roi(roi: Any, width: int, height: int) -> tuple[int, int, int, int]:
    if roi is None:
        return 0, 0, width, height
    b = _box_xyxy(roi)
    x1, y1 = int(np.clip(np.floor(b[0]), 0, width)), int(np.clip(np.floor(b[1]), 0, height))
    x2, y2 = int(np.clip(np.ceil(b[2]), 0, width)), int(np.clip(np.ceil(b[3]), 0, height))
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ToolError("sahi", f"roi {list(b)} leaves under 8 px of the {width} x {height} image")
    return x1, y1, x2, y2


def _snap_label(label: str, prompts: list[str]) -> str:
    """The prompt a backend label belongs to (grounding-dino returns matched text spans)."""
    raw = str(label).strip().rstrip(".").strip()
    low = raw.lower()
    for p in prompts:
        if p.lower() == low:
            return p
    for p in sorted(prompts, key=len, reverse=True):
        if p.lower() in low or (low and low in p.lower()):
            return p
    return raw


def _windows(width: int, height: int, slice_px: int, overlap: float) -> list[list[int]]:
    if slice_px <= 0:
        return [[0, 0, width, height]]
    if not 0.0 <= overlap < 1.0:
        raise ToolError("sahi", f"overlap must be in [0, 1), got {overlap}")
    if slice_px < 32:
        raise ToolError("sahi", f"slice_px must be 0 (no slicing) or >= 32, got {slice_px}")
    return _slice_bboxes(height, width, slice_px, slice_px, overlap, overlap)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _run_backend(
    ctx: Any,
    backend: str,
    tile: np.ndarray,
    prompts: list[str],
    *,
    box_threshold: float,
    text_threshold: float,
    want_masks: bool,
) -> tuple[list[tuple[np.ndarray, float, str, np.ndarray | None]], int]:
    """Raw predictions in tile pixels: (xyxy, score, label, bool mask or None), and the call count."""
    out: list[tuple[np.ndarray, float, str, np.ndarray | None]] = []
    if backend == "sam3":
        for p in prompts:
            r = ctx.tool("sam3.segment_text", image=tile, query=p, max_results=0)
            masks = r.get("masks") or []
            for i, (box, score) in enumerate(
                zip(r.get("boxes") or [], r.get("scores") or [], strict=False)
            ):
                m = np.asarray(masks[i]) > 0 if want_masks and i < len(masks) else None
                out.append((_box_xyxy(box), float(score), p, m))
        return out, len(prompts)
    if backend == "grounding-dino":
        kwargs: dict[str, Any] = {"image": tile, "query": " ".join(f"{p}." for p in prompts)}
        if box_threshold > 0:
            kwargs["box_threshold"] = box_threshold
        if text_threshold > 0:
            kwargs["text_threshold"] = text_threshold
        r = ctx.tool("grounding-dino.detect", **kwargs)
    elif backend == "locate-anything":
        r = ctx.tool("locate-anything.detect", image=tile, categories=list(prompts))
    elif "." in backend:
        r = ctx.tool(backend, image=tile, query=". ".join(prompts))
    else:
        raise ToolError(
            "sahi.detect_sliced",
            f"unknown backend {backend!r}; use one of {sorted(BACKEND_TOOLS)} or a tool name returning detections",
        )
    for d in r.get("detections") or []:
        out.append((_box_xyxy(d["box"]), float(d.get("score", 1.0)), str(d.get("label", "")), None))
    return out, 1


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="sahi.detect_sliced",
    summary="Detect small parts by running a detector (grounding-dino, sam3, locate-anything) on overlapping slices and merging the boxes (SAHI).",
    tags=("perception",),
)
def detect_sliced(
    ctx,
    image: np.ndarray,
    prompts: list[str] | str,
    backend: str = "grounding-dino",
    slice_px: int = DEFAULT_SLICE_PX,
    overlap: float = DEFAULT_OVERLAP,
    full_frame: bool = True,
    roi: BoundingBox2D | None = None,
    min_score: float | None = None,
    box_threshold: float = 0.0,
    text_threshold: float = 0.0,
    max_box_fraction: float = DEFAULT_MAX_BOX_FRACTION,
    min_box_px: float = DEFAULT_MIN_BOX_PX,
    max_box_px: float = 0.0,
    postprocess: str = DEFAULT_POSTPROCESS,
    match_metric: str = DEFAULT_MATCH_METRIC,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    class_agnostic: bool = False,
    return_masks: bool = False,
    max_detections: int = 0,
) -> DetectSlicedResult:
    """Find every instance of the prompted parts, small ones included.

    The image (or ``roi`` of it) is cut into ``slice_px`` square slices that
    overlap by ``overlap`` of a side; the backend runs on each slice (which it
    upsamples to its own input size, so a 15 px part becomes ~40 px) and, with
    ``full_frame``, once on the whole image / ROI for parts larger than a
    slice. Boxes are shifted to frame pixels, filtered (``min_score``, by
    default 0.5 for sam3 and none for the others; longer side within
    ``min_box_px``..``max_box_px``, 0 = no upper bound; area at most
    ``max_box_fraction`` of the slice or image they came from) and merged
    per label with SAHI's postprocess (GREEDYNMM over IOS >= 0.5 by default:
    boxes cut at a slice edge join the whole one from the neighbouring slice).

    ``slice_px=0`` runs the backend once on the image / ROI through the same
    filters: the unsliced baseline. ``box_threshold`` / ``text_threshold`` go
    to grounding-dino when > 0. ``return_masks`` gives full-frame masks with
    the sam3 backend (unioned across slices); other backends give none, so
    ``masks`` is ``None``. Nothing found is an empty list, never an error.
    """
    t0 = time.perf_counter()
    if min_score is None:
        min_score = DEFAULT_MIN_SCORE.get(backend, 0.0)
    img = _image(image)
    names = _prompts(prompts)
    if postprocess not in POSTPROCESS_TYPES:
        raise ToolError(
            "sahi.detect_sliced",
            f"postprocess must be one of {POSTPROCESS_TYPES}, got {postprocess!r}",
        )
    if match_metric not in MATCH_METRICS:
        raise ToolError(
            "sahi.detect_sliced",
            f"match_metric must be one of {MATCH_METRICS}, got {match_metric!r}",
        )
    if backend not in BACKEND_TOOLS and "." not in backend:
        raise ToolError(
            "sahi.detect_sliced",
            f"unknown backend {backend!r}; use one of {sorted(BACKEND_TOOLS)} or a tool name returning detections",
        )
    H, W = img.shape[:2]
    rx1, ry1, rx2, ry2 = _roi(roi, W, H)
    region = img[ry1:ry2, rx1:rx2]
    windows = _windows(rx2 - rx1, ry2 - ry1, int(slice_px), float(overlap))
    passes = [(w, "slice") for w in windows] if len(windows) > 1 else [(windows[0], "full")]
    if len(windows) > 1 and full_frame:
        passes.append(([0, 0, rx2 - rx1, ry2 - ry1], "full"))
    want_masks = bool(return_masks) and backend == "sam3"

    preds: list[dict[str, Any]] = []
    calls = 0
    backend_s = 0.0
    for (x0, y0, x1, y1), source in passes:
        tile = np.ascontiguousarray(region[y0:y1, x0:x1])
        tb = time.perf_counter()
        raw, n = _run_backend(
            ctx,
            backend,
            tile,
            names,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            want_masks=want_masks,
        )
        backend_s += time.perf_counter() - tb
        calls += n
        tw, th = x1 - x0, y1 - y0
        for box, score, label, mask in raw:
            b = np.clip(box, [0, 0, 0, 0], [tw, th, tw, th])
            bw, bh = b[2] - b[0], b[3] - b[1]
            if bw <= 0 or bh <= 0 or score < min_score:
                continue
            if bw * bh > max_box_fraction * tw * th:
                continue
            side = max(bw, bh)
            if side < min_box_px or (max_box_px > 0 and side > max_box_px):
                continue
            pieces = []
            if mask is not None and mask.shape == (th, tw):
                pieces = [(mask, rx1 + x0, ry1 + y0)]
            preds.append(
                {
                    "box": b + np.array([rx1 + x0, ry1 + y0, rx1 + x0, ry1 + y0], dtype=np.float64),
                    "score": score,
                    "label": _snap_label(label, names),
                    "pieces": pieces,
                    "n": 1,
                    "sources": {source},
                }
            )
    raw_slice = sum(1 for p in preds if "slice" in p["sources"])
    raw_full = len(preds) - raw_slice

    tm = time.perf_counter()
    merged = _merge_predictions(
        preds, postprocess, match_metric, float(match_threshold), bool(class_agnostic)
    )
    if max_detections > 0:
        merged = merged[: int(max_detections)]
    masks: list[Mask] | None = None
    if want_masks:
        masks = []
        for p in merged:
            full = np.zeros((H, W), dtype=np.uint8)
            for m, ox, oy in p["pieces"]:
                view = full[oy : oy + m.shape[0], ox : ox + m.shape[1]]
                view[m] = 255
            masks.append(full)
    merge_s = time.perf_counter() - tm

    dets: list[Detection] = [
        {
            "box": _box_dict(p["box"]),
            "label": p["label"],
            "score": float(p["score"]),
            "n_merged": int(p["n"]),
            "sources": sorted(p["sources"]),
        }
        for p in merged
    ]
    logger.info(
        "sahi.detect_sliced: %d slices, %d backend calls, %d raw -> %d detections",
        len(windows),
        calls,
        len(preds),
        len(dets),
    )
    return {
        "detections": dets,
        "boxes": [d["box"] for d in dets],
        "labels": [d["label"] for d in dets],
        "scores": [d["score"] for d in dets],
        "masks": masks,
        "found": bool(dets),
        "count": len(dets),
        "raw_slice": raw_slice,
        "raw_full": raw_full,
        "n_slices": len(windows) if len(windows) > 1 else 0,
        "n_backend_calls": calls,
        "slice_windows": [
            _box_dict(np.array(w, dtype=np.float64) + [rx1, ry1, rx1, ry1]) for w in windows
        ],
        "backend": backend,
        "timing": {
            "backend_s": round(backend_s, 4),
            "merge_s": round(merge_s, 4),
            "total_s": round(time.perf_counter() - t0, 4),
        },
    }


@tool(
    name="sahi.slice_windows",
    summary="The overlapping slice windows SAHI cuts an image (or a region of it) into, in frame pixels.",
)
def slice_windows(
    width: int,
    height: int,
    slice_px: int = DEFAULT_SLICE_PX,
    overlap: float = DEFAULT_OVERLAP,
    roi: BoundingBox2D | None = None,
) -> SliceWindowsResult:
    """Windows ``detect_sliced`` would run the backend on (the whole-image pass not included).

    For a backend ``detect_sliced`` does not call: crop each window, run the
    detector, shift its boxes by the window's ``x1, y1`` and hand them all to
    ``sahi.merge``.
    """
    rx1, ry1, rx2, ry2 = _roi(roi, int(width), int(height))
    wins = _windows(rx2 - rx1, ry2 - ry1, int(slice_px), float(overlap))
    out = [_box_dict(np.array(w, dtype=np.float64) + [rx1, ry1, rx1, ry1]) for w in wins]
    return {"windows": out, "count": len(out)}


@tool(
    name="sahi.merge",
    summary="Merge overlapping detections from several slices with SAHI's postprocess (greedy NMM / NMM / NMS over IOS or IOU).",
)
def merge(
    detections: list[dict],
    postprocess: str = DEFAULT_POSTPROCESS,
    match_metric: str = DEFAULT_MATCH_METRIC,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    class_agnostic: bool = False,
) -> MergeResult:
    """Merge ``[{box, label, score}]`` already in frame pixels, best-first out.

    Merged boxes are the union of their members, keep the highest score and
    that member's label; ``n_merged`` counts the members.
    """
    if postprocess not in POSTPROCESS_TYPES:
        raise ToolError(
            "sahi.merge", f"postprocess must be one of {POSTPROCESS_TYPES}, got {postprocess!r}"
        )
    if match_metric not in MATCH_METRICS:
        raise ToolError(
            "sahi.merge", f"match_metric must be one of {MATCH_METRICS}, got {match_metric!r}"
        )
    preds = [
        {
            "box": _box_xyxy(d["box"]),
            "score": float(d.get("score", 1.0)),
            "label": str(d.get("label", "")),
            "pieces": [],
            "n": int(d.get("n_merged", 1)),
            "sources": set(d.get("sources") or []),
        }
        for d in detections
    ]
    merged = _merge_predictions(
        preds, postprocess, match_metric, float(match_threshold), bool(class_agnostic)
    )
    dets: list[Detection] = [
        {
            "box": _box_dict(p["box"]),
            "label": p["label"],
            "score": float(p["score"]),
            "n_merged": int(p["n"]),
            "sources": sorted(p["sources"]),
        }
        for p in merged
    ]
    return {"detections": dets, "count": len(dets)}
