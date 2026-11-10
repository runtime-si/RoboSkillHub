"""d405-stereo tool bundle: depth from a rendered passive IR stereo pair the way a RealSense D405 gets it.

The D405 has no projector: its two global-shutter imagers 18 mm apart match the scene's own texture,
so depth has holes on untextured or glossy faces, fattened edges and noise growing with z^2. Here the
pair is matched with OpenCV's semi-global block matching (Hirschmuller 2008; the same family as the
D4 ASIC and SimSense's GPU SGM): census-like robustness is approximated by a pre-filter cap, a
uniqueness test, a left-right consistency check, a speckle filter and sub-pixel disparity. Optional
sensor noise is added to the images first, and the depth can be quantised to the stream's step.

CPU only (OpenCV); no weights. SimSense (MIT) would be the GPU drop-in for real-time rates.
"""

from __future__ import annotations

import time
from typing import TypedDict

import numpy as np
from gap_core.tools import tool

BASELINE_M = 0.018
"""D405 stereo baseline (Intel D405 datasheet)."""
MIN_DEPTH_M, MAX_DEPTH_M = 0.07, 0.5
"""D405 ideal range (datasheet: 7 cm - 50 cm)."""
QUANT_M = 0.001
"""The YAM station's depth stream step (#185 measured 1 mm on 401 real frames)."""


class StereoResult(TypedDict):
    depth: list  # [H][W] metres in the left camera, 0 = no depth (hole)
    hole_fraction: float  # holes inside [min_depth, max_depth] candidates (all pixels)
    disparity_range_px: list  # [min, max] searched
    seconds: float


def disparity_bounds(
    fx: float, baseline_m: float, min_depth_m: float, max_depth_m: float
) -> tuple[int, int]:
    """(min disparity, number of disparities rounded up to 16) covering the depth range."""
    dmin = int(np.floor(fx * baseline_m / max_depth_m))
    dmax = int(np.ceil(fx * baseline_m / min_depth_m))
    n = int(np.ceil((dmax - dmin + 1) / 16.0) * 16)
    return dmin, n


def add_ir_noise(
    img: np.ndarray, rng: np.random.Generator, read_sigma: float = 2.0, shot: float = 0.02
) -> np.ndarray:
    """Read noise plus signal-dependent (shot-like) noise on an 8-bit image."""
    f = img.astype(np.float64)
    f = f + rng.normal(0.0, 1.0, f.shape) * np.sqrt(read_sigma**2 + (shot * f) ** 2)
    return np.clip(f, 0, 255).astype(np.uint8)


def stereo_depth(
    left: np.ndarray,
    right: np.ndarray,
    fx: float,
    baseline_m: float = BASELINE_M,
    min_depth_m: float = MIN_DEPTH_M,
    max_depth_m: float = MAX_DEPTH_M,
    block: int = 7,
    uniqueness: int = 10,
    quant_m: float = QUANT_M,
    ir_noise: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Depth [H, W] (metres, 0 = hole) of the rectified pair (``left`` the reference camera)."""
    import cv2

    rng = np.random.default_rng(seed)
    L = np.asarray(left)
    R = np.asarray(right)
    if L.ndim == 3:
        L = cv2.cvtColor(L[..., :3], cv2.COLOR_RGB2GRAY)
        R = cv2.cvtColor(R[..., :3], cv2.COLOR_RGB2GRAY)
    if ir_noise:
        L, R = add_ir_noise(L, rng), add_ir_noise(R, rng)
    dmin, n = disparity_bounds(fx, baseline_m, min_depth_m, max_depth_m)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=dmin,
        numDisparities=n,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=uniqueness,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_HH,
    )
    disp = sgbm.compute(L, R).astype(np.float64) / 16.0
    valid = disp > max(dmin, 0) - 0.5
    valid &= disp > 0
    depth = np.zeros_like(disp)
    depth[valid] = fx * baseline_m / disp[valid]
    depth[(depth < min_depth_m) | (depth > max_depth_m)] = 0.0
    if quant_m:
        depth = np.round(depth / quant_m) * quant_m
    return depth


@tool(
    name="d405-stereo.depth_from_pair",
    summary="D405-like passive stereo depth (holes, edge fattening, z^2 noise) from a rendered left/right image pair.",
    tags=("perception", "simulation"),
)
def depth_from_pair(
    left: np.ndarray,
    right: np.ndarray,
    intrinsics: list[list[float]],
    baseline_m: float = BASELINE_M,
    min_depth_m: float = MIN_DEPTH_M,
    max_depth_m: float = MAX_DEPTH_M,
    quant_m: float = QUANT_M,
    ir_noise: bool = True,
    seed: int = 0,
) -> StereoResult:
    """``left`` / ``right``: a rectified pair (same intrinsics, ``right`` = left shifted +x by the baseline),
    uint8 grey or RGB. Returns depth in the left camera."""
    t0 = time.perf_counter()
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    d = stereo_depth(
        left,
        right,
        float(K[0, 0]),
        baseline_m,
        min_depth_m,
        max_depth_m,
        quant_m=quant_m,
        ir_noise=ir_noise,
        seed=seed,
    )
    dmin, n = disparity_bounds(float(K[0, 0]), baseline_m, min_depth_m, max_depth_m)
    return {
        "depth": d.tolist(),
        "hole_fraction": round(float((d <= 0).mean()), 4),
        "disparity_range_px": [dmin, dmin + n],
        "seconds": round(time.perf_counter() - t0, 3),
    }
