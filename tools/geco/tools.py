# Adapted from GeCo (https://github.com/jerpelhan/GeCo, commit c0fad20), demo.py (inference: resize_and_pad,
# normalise, model(img, exemplar boxes), keep boxes scoring over max / 4, NMS 0.5), MIT License,
# Copyright (c) 2024 jerpelhan. The model code under geco_core/ is GeCo's (models/, utils/box_ops.py,
# resize_and_pad of utils/data.py), vendored unchanged. Changes here: image and exemplar boxes in, boxes in
# the input image's pixels out (no matplotlib), the score-ratio threshold is a parameter.
"""geco tool bundle: GeCo low-shot detection and counting - every instance like 1-3 exemplar boxes.

Weights: GeCo.pth from the authors' Google Drive link (README); its licence is NOT stated - treat it as
research-only until the authors say otherwise. Set ``GAP_GECO_WEIGHTS`` to its path.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import tool

WEIGHTS_URL = "https://drive.google.com/file/d/1wjOF9MWkrVJVo5uG3gVqZEW9pwRq_aIk"
#: GeCo's demo keeps boxes scoring above max / 4.
SCORE_RATIO = 0.25

_lock = threading.Lock()
_model: dict[str, Any] = {}


class DetectResult(TypedDict):
    boxes: list[list[float]]  # [x1, y1, x2, y2] pixels of the input image
    scores: list[float]
    count: int
    seconds: float


def _load():
    with _lock:
        if "m" not in _model:
            import argparse

            import torch

            here = str(Path(__file__).resolve().parent / "geco_core")
            if here not in sys.path:
                sys.path.insert(0, here)
            from models.geco_infer import build_model

            path = os.environ.get("GAP_GECO_WEIGHTS") or str(Path.home() / ".cache/gap/GeCo.pth")
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"GeCo weights not at {path}; download {WEIGHTS_URL} (licence unstated)"
                )
            args = argparse.Namespace(
                image_size=1024,
                num_objects=3,
                zero_shot=False,
                emb_dim=256,
                num_heads=8,
                kernel_dim=1,
                backbone_lr=0,
                reduction=16,
                model_path=None,
                output_masks=False,
            )
            dev = os.environ.get("GAP_GECO_DEVICE") or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            m = build_model(args).to(dev)
            state = torch.load(path, map_location=dev, weights_only=True)["model"]
            state = {k.removeprefix("module."): v for k, v in state.items()}
            m.load_state_dict(state, strict=False)
            m.eval()
            _model.update(m=m, dev=dev, torch=torch)
        return _model


@tool(
    name="geco.detect_like",
    summary="Every instance like 1-3 exemplar boxes in an RGB image (GeCo low-shot detection + count).",
    tags=("perception",),
)
def detect_like(
    image: np.ndarray, exemplar_boxes: list[list[float]], score_ratio: float = SCORE_RATIO
) -> DetectResult:
    """``exemplar_boxes``: 1-3 ``[x1, y1, x2, y2]`` pixel boxes around examples in this image."""
    t0 = time.perf_counter()
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3:
        raise ValueError(f"geco expects a uint8 [H, W, 3] image, got {arr.dtype} {arr.shape}")
    ex = np.asarray(exemplar_boxes, dtype=np.float32).reshape(-1, 4)
    if not 1 <= len(ex) <= 3:
        raise ValueError("geco.detect_like needs 1-3 exemplar boxes")
    try:
        M = _load()
    except Exception as exc:
        raise ToolError("geco.detect_like", f"GeCo failed to load: {exc}") from exc
    torch = M["torch"]
    from torchvision import ops
    from torchvision import transforms as T
    from utils.data import resize_and_pad

    img = torch.from_numpy(np.ascontiguousarray(arr[..., :3])).permute(2, 0, 1).float() / 255.0
    # GeCo was trained with three exemplars: repeat the given ones to three (as the authors' 1-shot setting)
    ex3 = np.concatenate([ex] * 3)[:3]
    img_t, boxes_t, scale = resize_and_pad(img, torch.from_numpy(ex3), full_stretch=False)
    img_t = (
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_t)
        .unsqueeze(0)
        .to(M["dev"])
    )
    with torch.no_grad():
        outputs = M["m"](img_t, boxes_t.unsqueeze(0).to(M["dev"]))[0]
    v = outputs[0]["box_v"]
    keep_mask = v > v.max() * float(score_ratio)
    pb, ps = outputs[0]["pred_boxes"][keep_mask], v[keep_mask]
    keep = ops.nms(pb, ps, 0.5)
    pb = torch.clamp(pb[keep], 0, 1) * img_t.shape[-1] / scale
    return {
        "boxes": [[round(float(c), 2) for c in b] for b in pb.cpu().numpy()],
        "scores": [round(float(s), 4) for s in ps[keep].cpu().numpy()],
        "count": int(len(keep)),
        "seconds": round(time.perf_counter() - t0, 3),
    }
