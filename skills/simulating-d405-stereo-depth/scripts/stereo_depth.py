"""Replace a simulated camera's perfect depth with D405-like passive stereo depth from its rendered pair."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PAIR_KEYS = (("ir_left", "ir_right"), ("rgb", "rgb_right"))
"""Where a sim camera carries its pair: the two IR imagers, or the colour frame plus a right image."""


class Output(TypedDict):
    route: str
    observation: dict[str, Any]
    hole_fraction: float
    reason: str


def run(
    ctx: NodeContext,
    camera: str,
    observation: dict[str, Any] | None = None,
    baseline_m: float = 0.018,
    min_depth_m: float = 0.07,
    max_depth_m: float = 0.5,
    quant_m: float = 0.001,
    ir_noise: bool = True,
    seed: int = 0,
) -> Output:
    """``observation`` (default a fresh robot.get_observation): its ``camera`` must carry a rectified pair
    (``ir_left``/``ir_right`` or ``rgb``/``rgb_right``) and ``intrinsics``. Returns the observation with that
    camera's ``depth`` replaced (``depth_true`` keeps the original)."""
    obs = observation if observation is not None else ctx.tool("robot.get_observation")
    cams = obs.get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} for k, v in cams.items()]
    cam = next((c for c in cams if str(c.get("name")) == camera), None)
    pair = None
    if cam is not None:
        pair = next(
            (
                (cam[a], cam[b])
                for a, b in PAIR_KEYS
                if cam.get(a) is not None and cam.get(b) is not None
            ),
            None,
        )
    if cam is None or pair is None or cam.get("intrinsics") is None:
        return {
            "route": "no_pair",
            "observation": obs,
            "hole_fraction": 0.0,
            "reason": f"camera {camera!r} has no rendered stereo pair with intrinsics",
        }
    try:
        r = ctx.tool(
            "d405-stereo.depth_from_pair",
            left=np.asarray(pair[0]),
            right=np.asarray(pair[1]),
            intrinsics=np.asarray(cam["intrinsics"]).tolist(),
            baseline_m=baseline_m,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            quant_m=quant_m,
            ir_noise=ir_noise,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "route": "no_pair",
            "observation": obs,
            "hole_fraction": 0.0,
            "reason": f"d405-stereo failed: {exc}",
        }
    new = dict(cam)
    if cam.get("depth") is not None:
        new["depth_true"] = cam["depth"]
    new["depth"] = np.asarray(r["depth"], dtype=np.float32)
    out_cams = [new if c is cam else c for c in cams]
    return {
        "route": "depth_rendered",
        "observation": {**obs, "cameras": out_cams},
        "hole_fraction": float(r["hole_fraction"]),
        "reason": f"{r['hole_fraction']:.1%} holes",
    }
