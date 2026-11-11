"""dinov2 tool bundle -- DINOv2 embeddings of image regions (boxes or masks).

DINOv2 (Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023;
facebook/dinov2-small and facebook/dinov2-base through transformers, Apache-2.0, pinned revisions).
Each region is cut out of the frame as a square crop around its box (a margin of context), the pixels
outside its mask filled with the ImageNet mean colour, resized and encoded; the region's vector is its
mask-pooled patch tokens (the mean of the final-layer tokens whose patch is mostly inside the mask, as
No-Time-to-Train pools DINOv2 features under SAM masks) concatenated with the CLS token, each half
L2-normalised, the whole L2-normalised. Two regions' cosine is then the mean of their pooled and
CLS cosines.

The model loads lazily on first call; importing this module never pulls torch / transformers. Env
knobs: ``GAP_DINOV2_MODEL`` (a pinned repo or a local directory), ``GAP_DINOV2_DEVICE`` (default
``cuda``).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import PerceptionFailed, ToolError
from gap_core.tools import tool

logger = logging.getLogger(__name__)

#: Hugging Face repo -> the commit the bundle loads (model cards: Apache-2.0).
MODELS = {
    "facebook/dinov2-small": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
    "facebook/dinov2-base": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
}
DEFAULT_MODEL = "facebook/dinov2-base"

#: Crop side sent to the encoder [px]: 16 x 16 patches of 14 px.
IMAGE_SIZE = 224
#: Context around a region's box, as a share of its long side on each side.
MARGIN = 0.15
#: Smallest crop side cut from the frame [px]; a 12 px part is cut with its surroundings.
MIN_CROP = 24
#: A patch is pooled when at least this share of it is inside the region's mask.
MASK_SHARE = 0.5
#: ImageNet mean (RGB, 0-255): the fill outside a region's mask, so it encodes to ~zero input.
FILL = (124, 116, 104)

_load_lock = threading.Lock()
_runtime: Any = None


class EmbedResult(TypedDict):
    embeddings: np.ndarray  # [N, 2C] float32, unit rows: [pooled patches | CLS], each half unit
    boxes: list[list[int]]  # per region: the crop cut from the frame [x1, y1, x2, y2] px
    pooled_patches: list[int]  # per region: patches pooled (0 -> the CLS half doubles for it)
    model: str


def _model_source() -> tuple[str, str | None]:
    name = os.environ.get("GAP_DINOV2_MODEL") or DEFAULT_MODEL
    if name in MODELS:
        return name, MODELS[name]
    if os.path.isdir(name):
        return name, None
    raise ValueError(
        f"GAP_DINOV2_MODEL must be one of {sorted(MODELS)} or a local directory, got {name!r}"
    )


class _Runtime:
    """One DINOv2 encoder on the device: square RGB crops in, (CLS, patch tokens) out."""

    def __init__(self, source: str, revision: str | None, device: str) -> None:
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device
        self.name = source
        logger.info("Loading DINOv2 %s@%s on %s ...", source, revision, device)
        self.model = AutoModel.from_pretrained(source, revision=revision).to(device).eval()
        self.patch = int(self.model.config.patch_size)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def encode(self, crops: list[np.ndarray], size: int) -> tuple[np.ndarray, np.ndarray]:
        """CLS [N, C] and patch tokens [N, grid * grid, C] (final layer norm applied)."""
        torch = self.torch
        F = torch.nn.functional
        x = torch.cat(
            [
                F.interpolate(
                    torch.tensor(c, device=self.device).permute(2, 0, 1)[None].float() / 255.0,
                    size=(size, size),
                    mode="bicubic",
                    align_corners=False,
                )
                for c in crops
            ]
        ).clamp(0, 1)
        x = (x - self.mean) / self.std
        cls, pat = [], []
        with torch.no_grad():
            for s in range(0, len(x), 64):
                h = self.model(pixel_values=x[s : s + 64]).last_hidden_state
                cls.append(h[:, 0].float().cpu().numpy())
                pat.append(h[:, 1:].float().cpu().numpy())
        return np.concatenate(cls), np.concatenate(pat)


def _get_runtime() -> Any:
    global _runtime
    with _load_lock:
        if _runtime is None:
            source, revision = _model_source()
            _runtime = _Runtime(source, revision, os.environ.get("GAP_DINOV2_DEVICE", "cuda"))
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

    snapshot_download(repo_id=source, revision=revision, repo_type="model")


# ---------------------------------------------------------------------------
# Regions (numpy)
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _region_masks(
    shape: tuple[int, int], boxes: list[Any] | None, masks: list[Any] | None
) -> list[np.ndarray]:
    h, w = shape
    out = []
    if masks is not None:
        for m in masks:
            a = np.asarray(m)
            if a.ndim == 3:
                a = a[..., 0]
            if a.shape != (h, w):
                raise ToolError(f"dinov2: mask shape {a.shape} != image {(h, w)}")
            out.append(a > 0)
        return out
    for b in boxes or []:
        if isinstance(b, dict):
            b = [b.get("x1", b.get("x_min")), b.get("y1", b.get("y_min")),
                 b.get("x2", b.get("x_max")), b.get("y2", b.get("y_max"))]  # fmt: skip
        x1, y1, x2, y2 = (int(round(float(v))) for v in b)
        m = np.zeros((h, w), bool)
        m[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = True
        out.append(m)
    return out


def cut_regions(
    image: np.ndarray, regions: list[np.ndarray], margin: float, use_mask: bool
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[int]]]:
    """Square crops around each region (clipped to the frame) and the region's mask in each."""
    h, w = image.shape[:2]
    crops, cmasks, boxes = [], [], []
    for m in regions:
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            raise ToolError("dinov2: empty region")
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        side = max(x2 - x1, y2 - y1)
        side = max(int(round(side * (1 + 2 * margin))), MIN_CROP)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bx1 = int(round(cx - side / 2))
        by1 = int(round(cy - side / 2))
        bx1 = min(max(bx1, 0), max(w - side, 0))
        by1 = min(max(by1, 0), max(h - side, 0))
        bx2, by2 = min(bx1 + side, w), min(by1 + side, h)
        crop = np.ascontiguousarray(image[by1:by2, bx1:bx2, :3]).copy()
        cm = m[by1:by2, bx1:bx2]
        if use_mask:
            crop[~cm] = FILL
        crops.append(crop)
        cmasks.append(cm)
        boxes.append([bx1, by1, bx2, by2])
    return crops, cmasks, boxes


def pool(
    cls: np.ndarray, patches: np.ndarray, cmasks: list[np.ndarray], grid: int
) -> tuple[np.ndarray, list[int]]:
    """[pooled | CLS] unit vectors; the pooled half is the mean of the patches inside each mask."""
    out, counts = [], []
    for i, cm in enumerate(cmasks):
        share = _patch_share(cm, grid)
        sel = share >= MASK_SHARE
        if not sel.any():
            sel = share >= share.max() - 1e-9 if share.max() > 0 else np.ones_like(sel)
        counts.append(int(sel.sum()))
        p = _unit(patches[i][sel].mean(axis=0))
        out.append(np.concatenate([p, _unit(cls[i])]))
    return _unit(np.stack(out)).astype(np.float32), counts


def _patch_share(cm: np.ndarray, grid: int) -> np.ndarray:
    """Share of each of grid x grid patches inside the crop's mask (row-major)."""
    import cv2

    return cv2.resize(cm.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA).reshape(-1)


@tool(
    name="dinov2.embed_regions",
    summary="DINOv2 embeddings of regions (boxes or masks) of one RGB frame: unit vectors [mask-pooled patch tokens | CLS]; compare by dot product.",
    tags=("perception",),
)
def embed_regions(
    image: np.ndarray,
    boxes: list[Any] | None = None,
    masks: list[Any] | None = None,
    use_mask: bool = True,
    margin: float = MARGIN,
    image_size: int = IMAGE_SIZE,
) -> EmbedResult:
    """Embed each region of ``image`` (HxWx3 uint8 RGB): ``boxes`` [x1, y1, x2, y2] px or ``masks``
    (HxW). ``use_mask`` fills the crop outside the mask (masks only; a box fills nothing)."""
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ToolError(f"dinov2: image must be HxWx3, got {img.shape}")
    if boxes is None and masks is None:
        raise ToolError("dinov2: give boxes or masks")
    regions = _region_masks(img.shape[:2], boxes, masks)
    if not regions:
        return {"embeddings": np.zeros((0, 0), np.float32), "boxes": [], "pooled_patches": [],
                "model": ""}  # fmt: skip
    if image_size % 14:
        raise ToolError("dinov2: image_size must be a multiple of 14")
    crops, cmasks, cboxes = cut_regions(img.astype(np.uint8), regions, margin,
                                        use_mask and masks is not None)  # fmt: skip
    try:
        rt = _get_runtime()
        cls, pat = rt.encode(crops, image_size)
    except ToolError:
        raise
    except Exception as e:  # noqa: BLE001 - model load / CUDA failures surface as a perception failure
        raise PerceptionFailed(f"dinov2: {type(e).__name__}: {e}") from e
    grid = image_size // rt.patch
    emb, counts = pool(cls, pat, cmasks, grid)
    return {"embeddings": emb, "boxes": cboxes, "pooled_patches": counts, "model": rt.name}
