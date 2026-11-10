"""Add D405 depth noise fitted to real station frames to every simulated depth image of an observation."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

D405_FIT = {
    "sigma_coeff_per_m": 0.0228,
    "corr_sigma_px": 2.4,
    "lowfreq_coeff_per_m": 0.0072,
    "lowfreq_scale_px": 30.0,
    "edge_dropout": 0.41,
    "quant_step_m": 0.001,
}
"""Fitted on 401 real wrist D405 frames of the YAM station (13166 flat 60 px patches, wood table)
by skills/augmenting-sim-depth-with-a-fitted-d405-noise-model/scripts/fit_noise.py: fine noise
sigma = 0.0228 z^2 (0.51 mm at 15 cm) with a 2.4 px correlation, 41 % dropout on depth edges, depth stored
in 1 mm steps. The smooth error is the frame-to-frame one (scripts/fit_temporal.py, static repeated
captures, so no table relief): 0.0072 z^2 (0.16 mm at 15 cm) at the model's 30 px scale; the first fit's
0.0452 z^2 (1.0 mm) measured the wood table's relief."""


def add_fitted_noise(depth, k, rng, params=None, scale=1.0):
    """D405 noise fitted to real frames (``D405_FIT``) on a clean depth image (0 = no reading): correlated
    fine noise + a smooth low-frequency error, both growing with z^2, dropouts on depth edges, then the
    stream's quantisation. ``scale`` multiplies both noise amplitudes (stress tests)."""
    import cv2

    p = {**D405_FIT, **(params or {})}
    base = np.asarray(depth, dtype=np.float64)
    valid = base > 0
    z2 = base**2
    fine = cv2.GaussianBlur(rng.normal(0, 1, base.shape), (0, 0), p["corr_sigma_px"])
    fine /= fine.std() or 1.0
    low = cv2.GaussianBlur(rng.normal(0, 1, base.shape), (0, 0), p["lowfreq_scale_px"])
    low /= low.std() or 1.0
    noisy = base + scale * z2 * (p["sigma_coeff_per_m"] * fine + p["lowfreq_coeff_per_m"] * low)
    dz = np.maximum(
        np.abs(np.diff(base, axis=0, append=base[-1:])),
        np.abs(np.diff(base, axis=1, append=base[:, -1:])),
    )
    edge = cv2.dilate((dz > 0.002).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    noisy[edge & (rng.random(base.shape) < p["edge_dropout"])] = 0.0
    q = p["quant_step_m"]
    return np.where(valid & (noisy > 0), np.round(noisy / q) * q, 0.0).astype(np.float32)


class Output(TypedDict):
    route: str
    observation: dict[str, Any]
    cameras: list[str]
    scale: float


def run(
    ctx: NodeContext,
    observation: dict[str, Any] | None = None,
    cameras: list[str] | None = None,
    scale: float = 1.0,
    seed: int = 0,
    params: dict[str, float] | None = None,
) -> Output:
    """``observation`` (default: a fresh robot.get_observation) with each named camera's depth (all with
    depth by default) replaced by ``add_fitted_noise(depth, scale)``; ``params`` overrides ``D405_FIT``."""
    if observation is None:
        observation = ctx.tool("robot.get_observation")
    cams = observation.get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} for k, v in cams.items()]
    rng = np.random.default_rng(seed)
    out_cams, done = [], []
    for c in cams:
        c = dict(c)
        if c.get("depth") is not None and (not cameras or c.get("name") in cameras):
            k = np.asarray(c.get("intrinsics") if c.get("intrinsics") is not None else np.eye(3))
            c["depth"] = add_fitted_noise(c["depth"], k, rng, {**D405_FIT, **(params or {})}, scale)
            c["source"] = f"{c.get('source', 'simulated')} + fitted D405 noise x{scale:g}"
            done.append(str(c.get("name")))
        out_cams.append(c)
    return {
        "route": "augmented" if done else "no_depth",
        "observation": {**observation, "cameras": out_cams},
        "cameras": done,
        "scale": float(scale),
    }
