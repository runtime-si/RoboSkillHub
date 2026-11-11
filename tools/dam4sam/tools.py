"""dam4sam tool bundle -- SAM 2.1 mask tracking with a distractor-aware memory (DAM4SAM).

DAM4SAM (Videnovic, Lukezic, Kristan, "A Distractor-Aware Memory for Visual Object Tracking with
SAM2", CVPR 2025; github.com/jovanavidenovic/DAM4SAM at aad389b85c22, Apache-2.0, licence in
LICENSE.dam4sam). SAM 2.1's memory keeps the last frames; DAM4SAM splits it into a recent-appearance
memory (RAM) and a distractor-resolving memory (DRM) of anchor frames, added when the chosen mask is
confident and steady in size while SAM's alternative masks point somewhere else (a look-alike next
to the target). The modified SAM 2 video predictor (``add_to_drm``, ``return_all_masks``) is the
fork's ``sam2`` package, installed from the pinned commit; ``_Tracker`` below is the fork's
``dam4sam_tracker.DAM4SAMTracker``.

Adapted from dam4sam_tracker.py (Apache-2.0):

- ``_Tracker`` keeps ``initialize`` / ``track`` / ``estimate_mask_from_box`` and the DRM rule
  (predicted IoU > 0.8, size within +-20 % of the recent median, 5+ frames since the last anchor,
  an alternative mask whose box overlaps the chosen one's by <= 0.7). Changed: the VOT toolkit's
  ``Mask`` / ``calculate_overlaps`` (bounding box of a mask, box IoU) are numpy here; the checkpoint
  comes from the pinned Hugging Face SAM 2.1 repo (Meta's original ``.pt``) instead of
  ``checkpoints/``; the module no longer reseeds the global RNGs on import; boxes are [x1, y1, x2,
  y2]; frames are numpy RGB, not PIL.

The model loads lazily on first call; importing this module never pulls torch. Env knobs:
``GAP_DAM4SAM_MODEL`` (``base-plus``, the one pinned so far), ``GAP_DAM4SAM_DEVICE`` (default ``cuda``).
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

logger = logging.getLogger(__name__)

#: model -> (Hugging Face repo, pinned commit, Meta's checkpoint file, DAM4SAM's config).
MODELS = {
    "base-plus": ("facebook/sam2.1-hiera-base-plus", "b7320756a13354e7530a63935656d35b2f91a290",
                  "sam2.1_hiera_base_plus.pt", "sam21pp_hiera_b+.yaml"),
}  # fmt: skip
DEFAULT_MODEL = "base-plus"

#: DAM4SAM's DRM gates (dam4sam_tracker.py).
DRM_MIN_IOU = 0.8
DRM_SIZE_RATIO = (0.8, 1.2)
DRM_MIN_GAP = 5
DRM_MAX_BOX_IOU = 0.7

#: Open sessions at most (the least recently used is closed), and idle seconds before one is closed.
MAX_SESSIONS = 4
IDLE_S = 600.0

_load_lock = threading.Lock()
_predictor: Any = None
_sessions: dict[str, dict[str, Any]] = {}


class UpdateResult(TypedDict):
    mask: np.ndarray | None  # uint8 [H, W] 0 / 255
    box: list[int] | None  # [x1, y1, x2, y2] px
    object_present: bool
    area_px: int
    frame_index: int
    anchors: int  # frames in the distractor-resolving memory


# ---------------------------------------------------------------------------
# numpy stand-ins for the VOT toolkit
# ---------------------------------------------------------------------------


def mask_box(m: np.ndarray) -> np.ndarray | None:
    """[x1, y1, x2, y2] (exclusive) of a mask's pixels, or None when empty."""
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float64)


def box_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == k).astype(np.uint8)


def drm_should_add(
    chosen: np.ndarray,
    alternatives: list[np.ndarray],
    m_iou: float,
    size_ratio: float,
    gap_ok: bool,
) -> bool:
    """DAM4SAM's rule for adding the current frame to the distractor-resolving memory."""
    lo, hi = DRM_SIZE_RATIO
    if not (m_iou > DRM_MIN_IOU and lo <= size_ratio <= hi and chosen.any() and gap_ok):
        return False
    cbox = mask_box(chosen)
    alts = [np.logical_and(a, np.logical_not(chosen)).astype(np.uint8) for a in alternatives]
    alts = [keep_largest_component(a) for a in alts if a.sum() >= 1]
    if not alts:
        return False
    ious = [box_iou(cbox, mask_box(np.logical_or(a, chosen))) for a in alts]
    return min(ious) <= DRM_MAX_BOX_IOU


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _model() -> tuple[str, str, str, str]:
    name = os.environ.get("GAP_DAM4SAM_MODEL") or DEFAULT_MODEL
    if name not in MODELS:
        raise ValueError(f"GAP_DAM4SAM_MODEL must be one of {sorted(MODELS)}, got {name!r}")
    return MODELS[name]


def _get_predictor() -> Any:
    global _predictor
    with _load_lock:
        if _predictor is None:
            from huggingface_hub import hf_hub_download
            from sam2.build_sam import build_sam2_video_predictor

            repo, rev, fname, cfg = _model()
            ckpt = hf_hub_download(repo, fname, revision=rev)
            device = os.environ.get("GAP_DAM4SAM_DEVICE", "cuda")
            logger.info("Loading DAM4SAM %s (%s@%s) on %s ...", cfg, repo, rev, device)
            _predictor = build_sam2_video_predictor(cfg, ckpt, device=device)
        return _predictor


def weights_cached() -> bool | None:
    """Filesystem-only weight-cache probe for ``gap check`` (never downloads, never imports torch)."""
    try:
        repo, rev, fname, _ = _model()
        from huggingface_hub import try_to_load_from_cache
    except (ValueError, ImportError):
        return None
    try:
        return isinstance(try_to_load_from_cache(repo, fname, revision=rev), str)
    except Exception:
        return None


def prefetch() -> None:
    """Download the pinned checkpoint into the HF cache (``gap skills check --download``)."""
    from huggingface_hub import hf_hub_download

    repo, rev, fname, _ = _model()
    hf_hub_download(repo, fname, revision=rev)


class _Tracker:
    """DAM4SAMTracker (dam4sam_tracker.py, Apache-2.0) on one stream; see the module docstring."""

    def __init__(self, predictor: Any) -> None:
        import torch

        self.torch = torch
        self.predictor = predictor
        self.size = 1024
        dev = predictor.device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
        self.anchors = 0

    def _prep(self, rgb: np.ndarray) -> Any:
        import torchvision.transforms.functional as F

        t = self.torch.from_numpy(np.ascontiguousarray(rgb[..., :3])).to(self.predictor.device)
        t = F.resize(t.permute(2, 0, 1).float() / 255.0, [self.size, self.size])
        return (t - self.mean) / self.std

    def _state(self, h: int, w: int) -> dict[str, Any]:
        dev = self.predictor.device
        return {
            "images": None, "num_frames": 0, "offload_video_to_cpu": False,
            "offload_state_to_cpu": False, "video_height": h, "video_width": w,
            "device": dev, "storage_device": dev, "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {}, "cached_features": {}, "constants": {},
            "obj_id_to_idx": OrderedDict(), "obj_idx_to_id": OrderedDict(), "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {}, "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()},
            "tracking_has_started": False, "frames_already_tracked": {},
            "frames_tracked_per_obj": {}, "adds_in_drm_per_obj": {},
        }  # fmt: skip

    def initialize(
        self, rgb: np.ndarray, mask: np.ndarray | None, box: np.ndarray | None
    ) -> np.ndarray:
        with self.torch.inference_mode():
            h, w = rgb.shape[:2]
            self.h, self.w = h, w
            self.frame_index = 0
            self.object_sizes: list[int] = []
            self.last_added = -1
            self.state = self._state(h, w)
            self.state["images"] = {0: self._prep(rgb)}
            self.state["num_frames"] = 1
            self.predictor.reset_state(self.state)
            self.predictor._get_image_feature(self.state, frame_idx=0, batch_size=1)
            if mask is None:
                mask = self._mask_from_box(box)
            _, _, logits = self.predictor.add_new_mask(
                inference_state=self.state, frame_idx=0, obj_id=0, mask=mask
            )
            m = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
            self.state["images"].pop(0)
            return m

    def track(self, rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            self.frame_index += 1
            self.state["num_frames"] += 1
            self.state["images"][self.frame_index] = self._prep(rgb).unsqueeze(0)
            m = np.zeros((self.h, self.w), np.uint8)
            for out in self.predictor.propagate_in_video(
                self.state, start_frame_idx=self.frame_index, max_frame_num_to_track=0,
                return_all_masks=True,
            ):  # fmt: skip
                m = (out[2][0][0] > 0.0).cpu().numpy().astype(np.uint8)
                if len(out) == 4:
                    self._maybe_anchor(out[0], m, *out[3])
            self.state["images"].pop(self.frame_index)
            return m

    def _maybe_anchor(self, frame_idx: int, m: np.ndarray, masks: list, ious: Any) -> None:
        i = int(np.argmax(ious))
        alts = [
            (a[0][0] > 0.0).cpu().numpy().astype(np.uint8) for k, a in enumerate(masks) if k != i
        ]
        n = int(m.sum())
        self.object_sizes.append(n)
        recent = [s for s in self.object_sizes[-300:] if s >= 1][-10:]
        ratio = n / float(np.median(recent)) if len(self.object_sizes) > 1 and n >= 1 else -1.0
        gap_ok = self.last_added == -1 or self.frame_index - self.last_added > DRM_MIN_GAP
        if drm_should_add(m, alts, float(ious[i]), ratio, gap_ok):
            self.last_added = self.frame_index
            self.anchors += 1
            self.predictor.add_to_drm(inference_state=self.state, frame_idx=frame_idx, obj_id=0)

    def _mask_from_box(self, box: np.ndarray) -> np.ndarray:
        torch = self.torch
        from sam2.utils.transforms import SAM2Transforms

        p = self.predictor
        _, _, feats, _, sizes = p._get_image_feature(self.state, 0, 1)
        b = torch.as_tensor(np.asarray(box, np.float32)[None], device=feats[0].device)
        tr = SAM2Transforms(resolution=p.image_size, mask_threshold=0.0, max_hole_area=0.0,
                            max_sprinkle_area=0.0)  # fmt: skip
        coords = tr.transform_boxes(b, normalize=True, orig_hw=(self.h, self.w)).reshape(-1, 2, 2)
        labels = torch.tensor([[2, 3]], dtype=torch.int, device=coords.device)
        sparse, dense = p.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        hr = []
        for i in range(2):
            _, b_, c_ = feats[i].shape
            hr.append(feats[i].permute(1, 2, 0).view(b_, c_, sizes[i][0], sizes[i][1]))
        emb = feats[2] + p.no_mem_embed if p.directly_add_no_mem_embed else feats[2]
        _, b_, c_ = feats[2].shape
        emb = emb.permute(1, 2, 0).view(b_, c_, sizes[2][0], sizes[2][1])
        low, _, _, _ = p.sam_mask_decoder(
            image_embeddings=emb, image_pe=p.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=False, high_res_features=hr,
        )  # fmt: skip
        masks = tr.postprocess_masks(low, (self.h, self.w)) > 0
        return masks.squeeze(0).float().cpu().numpy()[0]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _image(image: Any, name: str) -> np.ndarray:
    a = np.asarray(image)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ToolError(name, f"image must be RGB [H, W, 3], got {a.shape}")
    return a[..., :3].astype(np.uint8)


def _box(box: Any, name: str) -> np.ndarray:
    b = box
    if isinstance(b, dict):
        b = [b.get("x1", b.get("x_min")), b.get("y1", b.get("y_min")),
             b.get("x2", b.get("x_max")), b.get("y2", b.get("y_max"))]  # fmt: skip
    try:
        a = np.asarray([float(v) for v in b], np.float64)
    except (TypeError, ValueError) as e:
        raise ToolError(name, f"box must be [x1, y1, x2, y2]: {e}") from e
    if a.shape != (4,) or a[2] <= a[0] or a[3] <= a[1]:
        raise ToolError(name, f"box must be [x1, y1, x2, y2] with x2 > x1, y2 > y1, got {b}")
    return a


def _result(m: np.ndarray, idx: int, anchors: int) -> UpdateResult:
    b = mask_box(m)
    return {
        "mask": (m > 0).astype(np.uint8) * 255 if b is not None else None,
        "box": [int(v) for v in b] if b is not None else None,
        "object_present": b is not None,
        "area_px": int(m.sum()),
        "frame_index": idx,
        "anchors": anchors,
    }


def _evict() -> None:
    now = time.monotonic()
    for k in [k for k, s in _sessions.items() if now - s["t"] > IDLE_S]:
        _sessions.pop(k, None)
    while len(_sessions) >= MAX_SESSIONS:
        _sessions.pop(min(_sessions, key=lambda k: _sessions[k]["t"]), None)


def _new_tracker(name: str) -> _Tracker:
    try:
        return _Tracker(_get_predictor())
    except Exception as e:  # noqa: BLE001 - model load / CUDA failures surface as a perception failure
        raise PerceptionFailed(f"{name}: {type(e).__name__}: {e}") from e


@tool(
    name="dam4sam.tracker_init",
    summary="Open a DAM4SAM (SAM 2.1 + distractor-aware memory) mask-tracking session on a frame from a box or a mask; for parts among look-alikes.",
    tags=("perception",),
)
def tracker_init(
    image: np.ndarray, box: Any = None, mask: Any = None, object_name: str = ""
) -> dict:
    """Seed a session with ``box`` [x1, y1, x2, y2] or ``mask`` [H, W]; returns its id and first mask."""
    name = "dam4sam.tracker_init"
    rgb = _image(image, name)
    if mask is None and box is None:
        raise ToolError(name, "give box or mask")
    m0 = None if mask is None else (np.asarray(mask) > 0).astype(np.float32)
    if m0 is not None and m0.shape != rgb.shape[:2]:
        raise ToolError(name, f"mask shape {m0.shape} != image {rgb.shape[:2]}")
    b0 = None if box is None else _box(box, name)
    tr = _new_tracker(name)
    try:
        m = tr.initialize(rgb, m0, b0)
    except Exception as e:  # noqa: BLE001
        raise PerceptionFailed(f"{name}: {type(e).__name__}: {e}") from e
    res = _result(m, 0, 0)
    if not res["object_present"]:
        return {
            "tracker_id": "",
            "initial_mask": None,
            "initial_box": None,
            "object_present": False,
        }
    _evict()
    tid = uuid.uuid4().hex[:12]
    _sessions[tid] = {"tracker": tr, "t": time.monotonic(), "name": object_name}
    return {"tracker_id": tid, "initial_mask": res["mask"], "initial_box": res["box"],
            "object_present": True}  # fmt: skip


@tool(
    name="dam4sam.tracker_update",
    summary="Advance a DAM4SAM session by one frame; mask, box, presence and the anchor count.",
    tags=("perception",),
)
def tracker_update(tracker_id: str, image: np.ndarray) -> UpdateResult:
    name = "dam4sam.tracker_update"
    s = _sessions.get(tracker_id)
    if s is None:
        raise ToolError(name, f"no open session {tracker_id!r} (closed or evicted)")
    rgb = _image(image, name)
    tr: _Tracker = s["tracker"]
    if rgb.shape[:2] != (tr.h, tr.w):
        raise ToolError(name, f"frame {rgb.shape[:2]} != session's {(tr.h, tr.w)}")
    try:
        m = tr.track(rgb)
    except Exception as e:  # noqa: BLE001
        raise PerceptionFailed(f"{name}: {type(e).__name__}: {e}") from e
    s["t"] = time.monotonic()
    return _result(m, tr.frame_index, tr.anchors)


@tool(
    name="dam4sam.tracker_close",
    summary="Free a DAM4SAM session (idempotent).",
    tags=("perception",),
)
def tracker_close(tracker_id: str) -> dict:
    return {"closed": _sessions.pop(tracker_id, None) is not None}


@tool(
    name="dam4sam.track",
    summary="Track one part through a recorded clip with DAM4SAM from a box or mask on its first frame; per-frame boxes and presence.",
    tags=("perception",),
)
def track(
    frames: list[np.ndarray], box: Any = None, mask: Any = None, return_masks: bool = False
) -> dict:
    """Forward from frame 0 (seed ``box`` or ``mask`` there)."""
    name = "dam4sam.track"
    if not frames:
        raise ToolError(name, "no frames")
    init = tracker_init(frames[0], box=box, mask=mask)
    tid = init["tracker_id"]
    boxes, present, masks = [init["initial_box"]], [init["object_present"]], [init["initial_mask"]]
    anchors = 0
    try:
        for f in frames[1:]:
            if not tid:
                boxes.append(None)
                present.append(False)
                masks.append(None)
                continue
            r = tracker_update(tid, f)
            boxes.append(r["box"])
            present.append(r["object_present"])
            masks.append(r["mask"])
            anchors = r["anchors"]
    finally:
        if tid:
            tracker_close(tid)
    out = {"boxes": boxes, "present": present, "anchors": anchors}
    if return_masks:
        out["masks"] = masks
    return out
