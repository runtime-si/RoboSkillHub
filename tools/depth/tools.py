"""depth tool bundle: complete and denoise a sensor depth image with its RGB frame.

Runs Robbyant's LingBot-Depth v0.5 (masked depth modeling; code
https://github.com/Robbyant/lingbot-depth, weights
https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5, both
Apache-2.0) through the ``mdm`` package at a pinned commit, the way upstream's
README "Quick Start" and ``example.py`` call ``MDMModel.infer``. The package is
a dependency of this bundle's environment and the checkpoint comes from the
Hugging Face Hub at a pinned commit on first use. Nothing is copied from
upstream; the preprocessing (RGB / 255, depth in metres with 0 for a hole)
follows ``example.py``. Changed: the bundle does not install xFormers, so
attention runs through upstream's PyTorch SDPA fallback and a one-frame token
list goes through the encoder blocks' tensor path
(:func:`_run_single_sequences_without_xformers`, the same attention for a
single sequence); the output keeps the input's convention (metres, 0 = no
depth) instead of upstream's ``inf`` for masked pixels.

LingBot-Depth takes an RGB image and the raw depth of the same camera (holes
as 0) and returns a dense metric depth image of the same size plus a validity
probability per pixel. It does not condition on the intrinsics: ``K`` is
checked and used only for the optional point output.

The model loads lazily on first call (module-level singleton); importing this
module never imports torch or mdm. Knobs via env:

- ``GAP_DEPTH_DEVICE`` -- torch device (default ``cuda``).
- ``GAP_DEPTH_MODEL`` -- a local ``model.pt`` instead of the pinned Hub
  snapshot (e.g. the depth-completion variant
  ``robbyant/lingbot-depth-postrain-dc-vitl14``; only v0.5 was checked).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import tool
from gap_core.types import Mask

logger = logging.getLogger(__name__)

MODEL_REPO = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
#: Hub commit of the v0.5 checkpoint (uploaded 2026-02-14), checked live on 2026-11-07.
MODEL_REVISION = "79204ed6b837f4fdd192cf563e59481fecfa0295"
MODEL_FILE = "model.pt"
#: SHA-256 of ``model.pt`` at that commit (the Hub's LFS oid), 1,284,837,952 bytes.
MODEL_SHA256 = "b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23"
#: lingbot-depth commit the bundle was checked with (pyproject pins the same).
LINGBOT_COMMIT = "f3a237e434ae987bc38281476d6cfb5df3e4d739"

#: ``MDMModel.infer``'s default: the most image tokens of the checkpoint's
#: ``num_tokens_range`` (resolution_level 0..9 spans it linearly).
DEFAULT_RESOLUTION_LEVEL = 9
#: The encoder treats depth <= 0.01 m as a hole (modules_rgbd_encoder.py); the
#: same bound marks a valid input pixel here.
MIN_VALID_DEPTH_M = 0.01
#: ``MDMModel.infer`` calls a pixel valid when the mask head's probability is above 0.5.
MASK_THRESHOLD = 0.5
#: Fewest measured pixels to fit the model's scale and shift to; below it the
#: model's depth is used as it comes (a frame with ~1 % depth).
MIN_ALIGN_PIXELS = 3000
#: Robust refit of the scale / shift: rounds, and residuals beyond this many
#: MADs dropped each round (a sensor's flying pixels and the model's misses).
ALIGN_ROUNDS = 3
ALIGN_MAD_CUT = 3.0

_load_lock = threading.Lock()
_run_lock = threading.Lock()
_runtime: Any = None


class RefineTiming(TypedDict):
    load_s: float  # model load paid by this call (0 once resident)
    infer_s: float  # the model's forward pass, transfers included
    total_s: float


class RefineResult(TypedDict):
    depth_m: np.ndarray  # float32 [H, W] metres, 0 where not valid
    valid: Mask  # uint8 [H, W] 0 / 255: measured, or the model's mask head says valid
    filled: Mask  # uint8 [H, W] 0 / 255: the model's depth where the input had a hole
    raw_valid_fraction: float  # input pixels with depth > 0.01 m
    valid_fraction: float  # output pixels marked valid
    filled_fraction: float  # of the input's holes, the share now valid
    aligned: bool  # the model's depth was fitted to the measured depth
    align_scale: float  # model depth x scale + shift (1, 0 when not aligned)
    align_shift_mm: float
    model_vs_measured_mm: float  # median |model - measured| after alignment, measured pixels
    points_m: np.ndarray | None  # float32 [H, W, 3] camera frame (OpenCV), when asked
    backend: str  # "lingbot-depth"
    model: str  # repo@revision (or the local file)
    timing: RefineTiming


# ---------------------------------------------------------------------------
# Inputs (numpy only)
# ---------------------------------------------------------------------------


def _rgb(rgb: Any) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"depth: rgb must be uint8 [H, W, 3], got {arr.dtype} {arr.shape}")
    return np.ascontiguousarray(arr)


def _depth(depth_m: Any, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(depth_m)
    if arr.ndim != 2 or arr.shape != shape:
        raise ValueError(f"depth: depth_m must be [H, W] = {list(shape)}, got {list(arr.shape)}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(
            f"depth: depth_m must be float metres, got {arr.dtype} "
            "(divide a uint16 millimetre image by 1000)"
        )
    out = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out[out <= MIN_VALID_DEPTH_M] = 0.0
    return out


def _intrinsics(K: Any) -> np.ndarray | None:
    if K is None:
        return None
    k = np.asarray(K, dtype=np.float64)
    if k.shape != (3, 3) or not (k[0, 0] > 0 and k[1, 1] > 0):
        raise ValueError(f"depth: K must be a 3x3 pinhole matrix with fx, fy > 0, got {k.tolist()}")
    return k


def _fit_scale_shift(pred: np.ndarray, raw: np.ndarray, use: np.ndarray) -> tuple[float, float]:
    """Robust ``raw ~ scale * pred + shift`` over ``use``; ``(1, 0)`` when it cannot be fitted."""
    x = pred[use].astype(np.float64)
    y = raw[use].astype(np.float64)
    if len(x) < MIN_ALIGN_PIXELS:
        return 1.0, 0.0
    # Seeded at scale 1 and the median offset, so a band the model misses
    # badly cannot lever the first fit (the station's scales were 0.91-1.08).
    scale, shift = 1.0, float(np.median(y - x))
    for _ in range(ALIGN_ROUNDS):
        r = y - (scale * x + shift)
        mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        keep = np.abs(r) < ALIGN_MAD_CUT * max(mad, 1e-4)
        if keep.sum() < MIN_ALIGN_PIXELS:
            break
        A = np.stack([x[keep], np.ones(int(keep.sum()))], axis=1)
        scale, shift = (float(v) for v in np.linalg.lstsq(A, y[keep], rcond=None)[0])
    if not scale > 0.0:
        return 1.0, 0.0
    return scale, shift


def _mask(b: np.ndarray) -> np.ndarray:
    return np.where(b, 255, 0).astype(np.uint8)


def _points(depth: np.ndarray, valid: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame points (OpenCV axes) of every pixel; NaN where not valid."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    z = np.where(valid, depth, np.nan).astype(np.float32)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Runtime: the pinned package over the pinned checkpoint
# ---------------------------------------------------------------------------


def _model_id() -> str:
    return os.environ.get("GAP_DEPTH_MODEL") or f"{MODEL_REPO}@{MODEL_REVISION}"


def _model_file() -> str:
    override = os.environ.get("GAP_DEPTH_MODEL")
    if override:
        return override
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, repo_type="model")
    return os.path.join(root, MODEL_FILE)


def _run_single_sequences_without_xformers() -> None:
    """Let mdm's encoder blocks take its one-frame token list without xFormers.

    The encoder drops the depth tokens of all-hole patches, so each frame's
    sequence has its own length and the blocks get a list of sequences, which
    upstream runs only through xFormers' block-diagonal attention. A list of
    one sequence is plain full attention over it: run it through the block's
    tensor path (PyTorch SDPA). ``refine`` sends one frame at a time.
    """
    from mdm.model.dinov2_rgbd.layers import block

    if block.XFORMERS_AVAILABLE or getattr(block.NestedTensorBlock, "_gap_single", False):
        return
    nested_forward = block.NestedTensorBlock.forward

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, list) and len(x_or_x_list) == 1:
            return [block.Block.forward(self, x_or_x_list[0])]
        return nested_forward(self, x_or_x_list)

    block.NestedTensorBlock.forward = forward
    block.NestedTensorBlock._gap_single = True


class _Runtime:
    """LingBot-Depth on the device, one RGB-D frame per call."""

    def __init__(self, model_file: str, device: str) -> None:
        import torch
        from mdm.model.v2 import MDMModel

        _run_single_sequences_without_xformers()
        self.torch = torch
        self.device = torch.device(device)
        model = MDMModel.from_pretrained(model_file)
        self.model = model.to(self.device).eval()

    def infer(
        self, rgb: np.ndarray, depth: np.ndarray, resolution_level: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """``(depth [H, W] metres, mask head's call [H, W] as 0 / 1, or None)``."""
        torch = self.torch
        image = torch.from_numpy(np.array(rgb)).to(self.device).permute(2, 0, 1).float().div_(255.0)
        depth_in = torch.from_numpy(np.array(depth)).to(self.device)
        with torch.inference_mode():
            out = self.model.infer(
                image[None],
                depth_in=depth_in[None],
                resolution_level=resolution_level,
                apply_mask=False,
            )
        pred = out["depth"][0].float().cpu().numpy()
        mask = out.get("mask")
        return pred, None if mask is None else mask[0].float().cpu().numpy()


def _build_runtime() -> _Runtime:
    device = os.environ.get("GAP_DEPTH_DEVICE", "cuda")
    path = _model_file()
    logger.info("Loading LingBot-Depth (%s) on %s from %s", _model_id(), device, path)
    return _Runtime(path, device)


def _get_runtime() -> tuple[Any, float]:
    """The resident runtime and the seconds this call spent loading it."""
    global _runtime
    with _load_lock:
        if _runtime is None:
            t0 = time.perf_counter()
            _runtime = _build_runtime()
            return _runtime, time.perf_counter() - t0
        return _runtime, 0.0


def weights_cached() -> bool | None:
    """Filesystem-only probe for ``gap check``: the pinned checkpoint is on disk."""
    override = os.environ.get("GAP_DEPTH_MODEL")
    if override:
        return os.path.isfile(override)
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        hit = try_to_load_from_cache(MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION)
    except Exception:
        return None
    return isinstance(hit, str)


def prefetch() -> None:
    """Download the pinned snapshot (``gap skills check --download``)."""
    logger.info("[depth] prefetching %s ...", _model_id())
    _model_file()
    logger.info("[depth] prefetch complete")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="depth.refine",
    summary=(
        "Complete (or denoise) a sensor depth image with its RGB frame (LingBot-Depth): "
        "same-size metric depth, validity and filled masks."
    ),
    tags=("perception", "depth"),
)
def refine(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray | None = None,
    keep_measured: bool = True,
    align: bool = True,
    resolution_level: int = DEFAULT_RESOLUTION_LEVEL,
    return_points: bool = False,
) -> RefineResult:
    """Dense metric depth for one RGB-D frame of a camera.

    Args:
        rgb: uint8 ``[H, W, 3]`` RGB of the camera the depth is registered to
            (the overhead Orbbec's or the wrist D405's colour frame).
        depth_m: float ``[H, W]`` metres, aligned to ``rgb``; 0 / NaN / inf
            mark holes.
        K: The 3x3 pinhole intrinsics of that frame. The model does not use
            them; they are checked and needed only for ``return_points``.
        keep_measured: Keep the sensor's value wherever it has one and use the
            model only in the holes (default: completion). False returns the
            model's depth everywhere (denoising): cleaner on dark, speckled
            surfaces, but it flattens thin upright parts seen from ~1 m and
            warps a close, well-measured table (see SKILL.md).
        align: Fit the model's depth to the measured pixels (robust
            ``scale * model + shift``) before using it; the raw model sits
            4-14 mm off the sensors on the station's tables.
        resolution_level: 0..9, the model's image-token budget (9 = upstream's
            default, the finest and slowest; 0 a third of the tokens, ~2x faster).
        return_points: Also return ``points_m`` ``[H, W, 3]`` in the camera
            frame (NaN where not valid); needs ``K``.

    Returns:
        ``RefineResult``: ``depth_m`` (0 where not valid), ``valid`` and
        ``filled`` masks (uint8 0 / 255, as ``sam3`` returns them), how much
        of the frame had depth before and after, the alignment, the model's
        median disagreement with the measured depth, and timings.
    """
    t_start = time.perf_counter()
    img = _rgb(rgb)
    raw = _depth(depth_m, img.shape[:2])
    k = _intrinsics(K)
    if return_points and k is None:
        raise ValueError("depth.refine: return_points needs K")
    level = int(resolution_level)
    if not 0 <= level <= 9:
        raise ValueError(f"depth.refine: resolution_level must be 0..9, got {resolution_level}")

    try:
        runtime, load_s = _get_runtime()
    except Exception as exc:
        raise ToolError("depth.refine", f"LingBot-Depth failed to load: {exc}") from exc
    t_infer = time.perf_counter()
    try:
        with _run_lock:
            pred, mask = runtime.infer(img, raw, level)
    except Exception as exc:
        raise ToolError("depth.refine", f"LingBot-Depth failed: {exc}") from exc
    infer_s = time.perf_counter() - t_infer

    pred = np.asarray(pred, dtype=np.float32)
    if pred.shape != raw.shape:
        raise ToolError("depth.refine", f"model returned {pred.shape}, expected {raw.shape}")
    measured = raw > 0.0
    model_ok = np.isfinite(pred) & (pred > MIN_VALID_DEPTH_M)
    if mask is not None:
        model_ok &= np.asarray(mask) > MASK_THRESHOLD
    both = model_ok & measured
    scale, shift = _fit_scale_shift(pred, raw, both) if align else (1.0, 0.0)
    aligned = align and (scale, shift) != (1.0, 0.0)
    pred = scale * pred + shift
    model_ok &= pred > MIN_VALID_DEPTH_M
    both &= model_ok
    if keep_measured:
        out = np.where(measured, raw, np.where(model_ok, pred, 0.0))
        valid = measured | model_ok
    else:
        out = np.where(model_ok, pred, 0.0)
        valid = model_ok
    out = out.astype(np.float32)
    filled = valid & ~measured
    holes = int((~measured).sum())
    return {
        "depth_m": out,
        "valid": _mask(valid),
        "filled": _mask(filled),
        "raw_valid_fraction": round(float(measured.mean()), 4),
        "valid_fraction": round(float(valid.mean()), 4),
        "filled_fraction": round(float(filled.sum()) / holes, 4) if holes else 1.0,
        "aligned": bool(aligned),
        "align_scale": round(scale, 5),
        "align_shift_mm": round(1000.0 * shift, 2),
        "model_vs_measured_mm": (
            round(1000.0 * float(np.median(np.abs(pred[both] - raw[both]))), 2)
            if both.any()
            else 0.0
        ),
        "points_m": _points(out, valid, k) if return_points else None,
        "backend": "lingbot-depth",
        "model": _model_id(),
        "timing": {
            "load_s": round(load_s, 3),
            "infer_s": round(infer_s, 3),
            "total_s": round(time.perf_counter() - t_start, 3),
        },
    }
