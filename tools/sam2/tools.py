"""sam2 tool bundle -- promptable video mask tracking (SAM 2.1; EdgeTAM as the fast option).

SAM 2.1 (Ravi et al., "SAM 2: Segment Anything in Images and Videos", 2024;
https://github.com/facebookresearch/sam2, Apache-2.0): promptable video
segmentation with a memory bank, object pointers and an occlusion head.
EdgeTAM (Zhou et al., "EdgeTAM: On-Device Track Anything Model", CVPR 2025;
https://github.com/facebookresearch/EdgeTAM, Apache-2.0) keeps that design and
feeds the memory attention through a 2D spatial perceiver, so a frame is
cheaper. Both run through Hugging Face transformers' ``Sam2VideoModel`` /
``EdgeTamVideoModel`` with ``Sam2VideoProcessor``
(https://github.com/huggingface/transformers, Apache-2.0,
src/transformers/models/sam2_video/, edgetam_video/), weights
``facebook/sam2.1-hiera-*`` (Apache-2.0 model cards) or
``yonigozlan/EdgeTAM-hf`` (the transformers conversion of facebook/EdgeTAM's
edgetam.pt; Apache-2.0 model card).

Adapted from transformers (Apache-2.0, license in LICENSE.transformers):

- ``_Runtime.start`` / ``step`` follow the streaming recipe of the SAM 2 video
  docs (``init_video_session`` without a video, ``add_inputs_to_inference_session``
  with ``original_size`` on the first frame, ``model(inference_session, frame=...)``
  per frame, ``post_process_masks``). Changed: the frame index is passed
  explicitly and the session is pruned after every frame to the frames the
  memory can still read (``num_maskmem`` and ``max_object_pointers_in_encoder``),
  so a live session holds constant GPU memory instead of every frame it has seen.
- ``_Runtime.clip`` follows ``propagate_in_video_iterator``: the prompt on one
  frame, propagated forward and then backward from it.

The presence call is the model's occlusion head (``object_score_logits`` > 0,
as SAM 2 gates its own masks); ``confidence`` is its sigmoid.

The model loads lazily on first call; importing this module never pulls torch
/ transformers. Knobs via env:

- ``GAP_SAM2_MODEL``  -- one of ``MODELS`` (pinned revision) or a local directory
  (default ``facebook/sam2.1-hiera-base-plus``).
- ``GAP_SAM2_DEVICE`` -- torch device (default ``cuda``).
- ``GAP_SAM2_DTYPE``  -- ``bfloat16`` (default on CUDA) or ``float32``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import PerceptionFailed, ToolError
from gap_core.tools import tool
from gap_core.types import BoundingBox2D, Mask

logger = logging.getLogger(__name__)

#: Hugging Face repo -> the commit the bundle loads (model cards: Apache-2.0).
MODELS = {
    "yonigozlan/EdgeTAM-hf": "c266ce53b3fc00f0f495b583f6a116c4e57f53bb",
    "facebook/sam2.1-hiera-tiny": "de431c4043854a71d8101e17995dfe596bf101a5",
    "facebook/sam2.1-hiera-small": "ee5bba1d82bb8749febdf90f45e84b687142ba03",
    "facebook/sam2.1-hiera-base-plus": "b7320756a13354e7530a63935656d35b2f91a290",
    "facebook/sam2.1-hiera-large": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
}
DEFAULT_MODEL = "facebook/sam2.1-hiera-base-plus"

#: Sessions kept open at once; opening one more evicts the least recently used.
MAX_SESSIONS = 8
#: A session not touched for this long is closed on the next tracker call.
SESSION_TTL_S = 600.0

_load_lock = threading.Lock()
_runtime: Any = None
_sessions: OrderedDict[str, _Session] = OrderedDict()
_sessions_lock = threading.Lock()


class TrackerInitResult(TypedDict):
    tracker_id: str  # opaque session id
    initial_mask: Mask | None  # uint8 [H, W] 0 / 255
    initial_box: BoundingBox2D | None
    score: float  # occlusion-head sigmoid on the prompt frame
    object_present: bool


class TrackerUpdateResult(TypedDict):
    mask: Mask | None
    box: BoundingBox2D | None
    confidence: float  # occlusion-head sigmoid [0, 1]
    object_present: bool
    area_px: int
    frame_index: int


class TrackerCloseResult(TypedDict):
    closed: bool


class TrackResult(TypedDict):
    boxes: list[BoundingBox2D | None]  # per frame; None where the object is absent
    present: list[bool]
    confidence: list[float]
    area_px: list[int]
    masks: list[Mask] | None  # per frame when return_masks
    prompt_frame: int
    seconds: float


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _model_source() -> tuple[str, str | None]:
    name = os.environ.get("GAP_SAM2_MODEL") or DEFAULT_MODEL
    if name in MODELS:
        return name, MODELS[name]
    if os.path.isdir(name):
        return name, None
    raise ValueError(
        f"GAP_SAM2_MODEL must be one of {sorted(MODELS)} or a local directory, got {name!r}"
    )


class _Runtime:
    """One SAM 2.1 / EdgeTAM video model on the device; numpy in, numpy out."""

    def __init__(self, source: str, revision: str | None, device: str) -> None:
        import torch
        from transformers import AutoConfig, Sam2VideoProcessor

        self.torch = torch
        self.device = device
        name = os.environ.get("GAP_SAM2_DTYPE") or ("bfloat16" if device != "cpu" else "float32")
        self.dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]
        cfg = AutoConfig.from_pretrained(source, revision=revision)
        if cfg.model_type == "edgetam_video":
            from transformers import EdgeTamVideoModel as Model
        else:
            from transformers import Sam2VideoModel as Model
        logger.info(
            "Loading %s %s@%s on %s (%s) ...", cfg.model_type, source, revision, device, name
        )
        self.processor = Sam2VideoProcessor.from_pretrained(source, revision=revision)
        self.model = Model.from_pretrained(source, revision=revision).to(device, dtype=self.dtype)
        self.model.eval()
        # non-conditioning outputs a frame can still read: num_maskmem (7) memories and
        # max_object_pointers_in_encoder (16) object pointers back, plus the current one
        self.keep = (
            max(
                int(getattr(cfg, "num_maskmem", 7)),
                int(getattr(cfg, "max_object_pointers_in_encoder", 16)),
            )
            + 1
        )

    def _pixels(self, image: np.ndarray) -> tuple[Any, tuple[int, int]]:
        inputs = self.processor(images=image, device=self.device, return_tensors="pt")
        return inputs.pixel_values[0], tuple(int(v) for v in inputs.original_sizes[0])

    def _mask(self, out: Any, size: tuple[int, int]) -> tuple[np.ndarray, float]:
        m = self.processor.post_process_masks(
            [out.pred_masks.float()], original_sizes=[size], binarize=True
        )[0]
        logit = float(out.object_score_logits.float().reshape(-1)[0])
        return m.reshape(-1, *size)[0].cpu().numpy().astype(bool), logit

    def _prompt(self, session: Any, frame_idx: int, prompt: dict, size: tuple[int, int]) -> None:
        kw: dict[str, Any] = {"inference_session": session, "frame_idx": frame_idx, "obj_ids": 1}
        if prompt.get("mask") is not None:
            kw["input_masks"] = prompt["mask"]
        else:
            kw["original_size"] = size
            if prompt.get("box") is not None:
                kw["input_boxes"] = [[list(prompt["box"])]]
            if prompt.get("points") is not None:
                kw["input_points"] = [[[list(map(float, p)) for p in prompt["points"]]]]
                kw["input_labels"] = [[list(map(int, prompt["labels"]))]]
        self.processor.add_inputs_to_inference_session(**kw)

    def start(self, image: np.ndarray, prompt: dict) -> tuple[Any, np.ndarray, float]:
        """A streaming session seeded on ``image`` (frame 0): the session, its mask, its score logit."""
        session = self.processor.init_video_session(inference_device=self.device, dtype=self.dtype)
        pixels, size = self._pixels(image)
        self._prompt(session, 0, prompt, size)
        with self.torch.inference_mode():
            out = self.model(inference_session=session, frame=pixels, frame_idx=0)
        mask, logit = self._mask(out, size)
        return session, mask, logit

    def step(self, session: Any, image: np.ndarray, frame_idx: int) -> tuple[np.ndarray, float]:
        pixels, size = self._pixels(image)
        with self.torch.inference_mode():
            out = self.model(inference_session=session, frame=pixels, frame_idx=frame_idx)
        mask, logit = self._mask(out, size)
        _prune(session, frame_idx, self.keep)
        return mask, logit

    def clip(
        self, frames: list[np.ndarray], prompt_frame: int, prompt: dict
    ) -> list[tuple[np.ndarray, float]]:
        """Per-frame (mask, logit), propagated forward then backward from ``prompt_frame``."""
        session = self.processor.init_video_session(
            video=frames, inference_device=self.device, dtype=self.dtype
        )
        size = (int(frames[0].shape[0]), int(frames[0].shape[1]))
        self._prompt(session, prompt_frame, prompt, size)
        res: dict[int, tuple[np.ndarray, float]] = {}
        with self.torch.inference_mode():
            for reverse in (False, True):
                if reverse and prompt_frame == 0:
                    break
                for out in self.model.propagate_in_video_iterator(
                    session, start_frame_idx=prompt_frame, reverse=reverse
                ):
                    res[int(out.frame_idx)] = self._mask(out, size)
        return [res[i] for i in range(len(frames))]


def _prune(session: Any, frame_idx: int, keep: int) -> None:
    """Drop streamed frames and non-conditioning outputs the memory can no longer read."""
    frames = getattr(session, "processed_frames", None)
    if frames:
        for k in [k for k in frames if k < frame_idx]:
            del frames[k]
    for per_obj in getattr(session, "output_dict_per_obj", {}).values():
        non_cond = per_obj.get("non_cond_frame_outputs", {})
        for k in [k for k in non_cond if k <= frame_idx - keep]:
            del non_cond[k]
    for tracked in getattr(session, "frames_tracked_per_obj", {}).values():
        for k in [k for k in tracked if k <= frame_idx - keep]:
            del tracked[k]


def _get_runtime() -> Any:
    global _runtime
    with _load_lock:
        if _runtime is None:
            source, revision = _model_source()
            _runtime = _Runtime(source, revision, os.environ.get("GAP_SAM2_DEVICE", "cuda"))
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

    logger.info("[sam2] prefetching %s@%s ...", source, revision)
    snapshot_download(repo_id=source, revision=revision, repo_type="model")
    logger.info("[sam2] prefetch complete")


# ---------------------------------------------------------------------------
# Inputs and outputs (numpy)
# ---------------------------------------------------------------------------


class _Session:
    def __init__(self, state: Any, size: tuple[int, int]) -> None:
        self.state, self.size = state, size
        self.frame_index = 0
        self.touched = time.monotonic()
        self.lock = threading.Lock()


def _image(image: Any, name: str) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(image))
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ToolError(name, f"expected RGB uint8 [H, W, 3] image, got {arr.dtype} {arr.shape}")
    return arr


def _box_xyxy(box: Any, name: str, size: tuple[int, int]) -> list[float]:
    if isinstance(box, dict):
        b = [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
    else:
        b = [float(v) for v in np.asarray(box, dtype=np.float64).reshape(-1)]
        if len(b) != 4:
            raise ToolError(name, f"box must be [x1, y1, x2, y2] or a BoundingBox2D, got {box!r}")
    h, w = size
    b = [
        min(max(b[0], 0.0), w),
        min(max(b[1], 0.0), h),
        min(max(b[2], 0.0), w),
        min(max(b[3], 0.0), h),
    ]
    if b[2] - b[0] < 1 or b[3] - b[1] < 1:
        raise ToolError(name, f"box {b} is empty inside the {w} x {h} image")
    return b


def _prompt(
    name: str,
    size: tuple[int, int],
    box: Any,
    points: Any,
    point_labels: Any,
    mask: Any,
) -> dict:
    """Exactly one of box (optionally with points), points, mask."""
    if mask is not None:
        if box is not None or points is not None:
            raise ToolError(name, "give a mask alone, or a box and / or points")
        m = np.asarray(mask)
        if m.shape != size:
            raise ToolError(name, f"mask shape {m.shape} differs from the image's {size}")
        if not (m > 0).any():
            raise ToolError(name, "the mask prompt is empty")
        return {"mask": (m > 0).astype(np.float32)}
    out: dict[str, Any] = {}
    if box is not None:
        out["box"] = _box_xyxy(box, name, size)
    if points is not None:
        p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(p) == 0:
            raise ToolError(name, "points is empty")
        lab = np.ones(len(p), int) if point_labels is None else np.asarray(point_labels, int)
        if lab.shape != (len(p),) or not set(lab.tolist()) <= {0, 1}:
            raise ToolError(name, "point_labels must be one 1 (object) / 0 (background) per point")
        if not (lab == 1).any() and box is None:
            raise ToolError(name, "a point prompt needs at least one object point (label 1)")
        out["points"], out["labels"] = p.tolist(), lab.tolist()
    if not out:
        raise ToolError(name, "give one prompt: box, points (pixel_x / pixel_y) or mask")
    return out


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _box_of(mask: np.ndarray) -> BoundingBox2D | None:
    ys = np.flatnonzero(mask.any(axis=1))
    if len(ys) == 0:
        return None
    xs = np.flatnonzero(mask.any(axis=0))
    return {
        "x1": float(xs[0]),
        "y1": float(ys[0]),
        "x2": float(xs[-1] + 1),
        "y2": float(ys[-1] + 1),
    }


def _u8(mask: np.ndarray) -> Mask:
    return mask.astype(np.uint8) * 255


def _reap(now: float) -> None:
    with _sessions_lock:
        for tid in [t for t, s in _sessions.items() if now - s.touched > SESSION_TTL_S]:
            _sessions.pop(tid, None)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="sam2.tracker_init",
    summary="Open a SAM 2.1 / EdgeTAM mask-tracking session on a live frame from a box, points or a mask; returns a tracker_id and the first mask.",
    tags=("perception",),
)
def tracker_init(
    image: np.ndarray,
    box: BoundingBox2D | list[float] | None = None,
    pixel_x: float | None = None,
    pixel_y: float | None = None,
    points: list[list[float]] | None = None,
    point_labels: list[int] | None = None,
    mask: Mask | None = None,
    object_name: str = "",
) -> TrackerInitResult:
    """Seed a session on ``image`` (its frame 0) with one object.

    Prompts: ``box`` (pixels, e.g. from a detector), optionally refined by
    ``points`` / ``point_labels`` (1 object, 0 background); or a single
    pixel (``pixel_x``, ``pixel_y``); or ``points`` alone; or a ``mask``
    (uint8 [H, W], e.g. from sam3.segment_box). Not open-vocabulary: find the
    object with a detector first. ``object_present`` False (empty mask or the
    occlusion head says absent) still returns a ``tracker_id`` only when
    the mask is not empty.
    """
    name = "sam2.tracker_init"
    arr = _image(image, name)
    size = (int(arr.shape[0]), int(arr.shape[1]))
    if pixel_x is not None or pixel_y is not None:
        if pixel_x is None or pixel_y is None or points is not None:
            raise ToolError(name, "give pixel_x and pixel_y together, or points, not both")
        points, point_labels = [[float(pixel_x), float(pixel_y)]], [1]
    prompt = _prompt(name, size, box, points, point_labels, mask)
    now = time.monotonic()
    _reap(now)
    try:
        state, m, logit = _get_runtime().start(arr, prompt)
    except ToolError:
        raise
    except Exception as e:
        raise PerceptionFailed(f"SAM 2 init failed: {e}") from e
    box_out = _box_of(m)
    if box_out is None:
        return {
            "tracker_id": "",
            "initial_mask": None,
            "initial_box": None,
            "score": _sigmoid(logit),
            "object_present": False,
        }
    tid = uuid.uuid4().hex
    with _sessions_lock:
        _sessions[tid] = _Session(state, size)
        while len(_sessions) > MAX_SESSIONS:
            old, _ = _sessions.popitem(last=False)
            logger.info("sam2: evicted session %s (over %d open)", old, MAX_SESSIONS)
    logger.info("sam2.tracker_init %s %r: %d px, score %.2f", tid, object_name, m.sum(), logit)
    return {
        "tracker_id": tid,
        "initial_mask": _u8(m),
        "initial_box": box_out,
        "score": _sigmoid(logit),
        "object_present": logit > 0,
    }


@tool(
    name="sam2.tracker_update",
    summary="Advance a SAM 2 session by one frame of the same camera; returns the mask, box, confidence and presence.",
    tags=("perception",),
)
def tracker_update(tracker_id: str, image: np.ndarray) -> TrackerUpdateResult:
    """Track the session's object into ``image``. ``object_present`` is the
    model's occlusion head (and a non-empty mask); when it is False the mask
    and box are None and the memory keeps going, so the object is picked up
    again when it reappears. Raises ToolError for an unknown ``tracker_id``.
    """
    name = "sam2.tracker_update"
    now = time.monotonic()
    _reap(now)
    with _sessions_lock:
        sess = _sessions.get(tracker_id)
        if sess is not None:
            _sessions.move_to_end(tracker_id)
    if sess is None:
        raise ToolError(
            name, f"unknown tracker_id {tracker_id!r} (closed, evicted or never opened)"
        )
    arr = _image(image, name)
    if (int(arr.shape[0]), int(arr.shape[1])) != sess.size:
        raise ToolError(name, f"frame size {arr.shape[:2]} differs from the session's {sess.size}")
    with sess.lock:
        sess.frame_index += 1
        try:
            m, logit = _get_runtime().step(sess.state, arr, sess.frame_index)
        except Exception as e:
            raise PerceptionFailed(f"SAM 2 update failed: {e}") from e
        sess.touched = time.monotonic()
        box_out = _box_of(m)
        present = logit > 0 and box_out is not None
        return {
            "mask": _u8(m) if present else None,
            "box": box_out if present else None,
            "confidence": _sigmoid(logit),
            "object_present": present,
            "area_px": int(m.sum()) if present else 0,
            "frame_index": sess.frame_index,
        }


@tool(
    name="sam2.tracker_close",
    summary="Free a SAM 2 session. Idempotent: closing an unknown id is not an error.",
    tags=("perception",),
)
def tracker_close(tracker_id: str) -> TrackerCloseResult:
    with _sessions_lock:
        sess = _sessions.pop(tracker_id, None)
    return {"closed": sess is not None}


@tool(
    name="sam2.track",
    summary="Track one object through a recorded clip from a box, points or mask on any frame; per-frame boxes, presence and (optionally) masks.",
    tags=("perception",),
)
def track(
    frames: list[np.ndarray],
    box: BoundingBox2D | list[float] | None = None,
    points: list[list[float]] | None = None,
    point_labels: list[int] | None = None,
    mask: Mask | None = None,
    prompt_frame: int = 0,
    return_masks: bool = False,
) -> TrackResult:
    """Prompt one object on ``frames[prompt_frame]`` and propagate forward to
    the end, then backward to the start. All frames: RGB uint8, one size.
    ``boxes[i]`` is None where the object is absent (occlusion head or empty
    mask); ``masks`` (uint8 0 / 255) only with ``return_masks``.
    """
    name = "sam2.track"
    if not frames:
        raise ToolError(name, "frames is empty")
    arrs = [_image(f, name) for f in frames]
    size = (int(arrs[0].shape[0]), int(arrs[0].shape[1]))
    if any((int(a.shape[0]), int(a.shape[1])) != size for a in arrs):
        raise ToolError(name, "all frames must have the same size")
    if not 0 <= prompt_frame < len(arrs):
        raise ToolError(name, f"prompt_frame {prompt_frame} is outside 0..{len(arrs) - 1}")
    prompt = _prompt(name, size, box, points, point_labels, mask)
    t = time.perf_counter()
    try:
        res = _get_runtime().clip(arrs, int(prompt_frame), prompt)
    except Exception as e:
        raise PerceptionFailed(f"SAM 2 clip tracking failed: {e}") from e
    boxes: list[BoundingBox2D | None] = []
    present, conf, area = [], [], []
    masks: list[Mask] = []
    for m, logit in res:
        b = _box_of(m)
        p = logit > 0 and b is not None
        boxes.append(b if p else None)
        present.append(bool(p))
        conf.append(_sigmoid(logit))
        area.append(int(m.sum()) if p else 0)
        if return_masks:
            masks.append(_u8(m) if p else np.zeros(size, np.uint8))
    return {
        "boxes": boxes,
        "present": present,
        "confidence": conf,
        "area_px": area,
        "masks": masks if return_masks else None,
        "prompt_frame": int(prompt_frame),
        "seconds": time.perf_counter() - t,
    }
