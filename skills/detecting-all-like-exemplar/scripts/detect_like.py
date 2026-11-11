"""Find every instance like 1-3 exemplar boxes in a camera frame (GeCo low-shot detection, or owlv2)."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    route: str
    boxes: list[list[float]]
    scores: list[float]
    count: int
    backend: str
    reason: str


def _rgb(obs: dict[str, Any], camera: str) -> np.ndarray | None:
    cams = obs.get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} for k, v in cams.items()]
    for c in cams:
        if (not camera or str(c.get("name")) == camera) and c.get("rgb") is not None:
            return np.ascontiguousarray(np.asarray(c["rgb"])[..., :3]).astype(np.uint8)
    return None


def run(
    ctx: NodeContext,
    exemplar_boxes: list[list[float]],
    camera: str = "",
    observation: dict[str, Any] | None = None,
    backend: str = "geco",
    min_count: int = 1,
) -> Output:
    """``exemplar_boxes``: 1-3 ``[x1, y1, x2, y2]`` boxes around examples in ``camera``'s frame.
    ``backend``: ``geco`` (geco.detect_like) or ``owlv2`` (owlv2.detect_like with the first box)."""
    out: Output = {
        "route": "not_found",
        "boxes": [],
        "scores": [],
        "count": 0,
        "backend": backend,
        "reason": "",
    }
    obs = observation if observation is not None else ctx.tool("robot.get_observation")
    img = _rgb(obs, camera)
    if img is None:
        out["reason"] = f"no RGB frame from {camera or 'any camera'}"
        return out
    ex = [[float(v) for v in b] for b in exemplar_boxes][:3]
    if not ex:
        raise ValueError("detecting-all-like-exemplar needs 1-3 exemplar boxes")
    try:
        if backend == "geco":
            r = ctx.tool("geco.detect_like", image=img, exemplar_boxes=ex)
            boxes, scores = r["boxes"], r["scores"]
        elif backend == "owlv2":
            r = ctx.tool("owlv2.detect_like", image=img, query_box=ex[0])
            dets = r.get("detections") or []
            boxes = [
                [d["box"][k] for k in ("x1", "y1", "x2", "y2")]
                if isinstance(d["box"], dict)
                else list(d["box"])
                for d in dets
            ]
            scores = [float(d.get("score", 0.0)) for d in dets]
        else:
            raise ValueError(f"backend must be geco or owlv2, got {backend!r}")
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"{backend} failed: {exc}"
        return out
    out.update(boxes=boxes, scores=scores, count=len(boxes))
    out["route"] = "found" if len(boxes) >= min_count else "not_found"
    out["reason"] = f"{len(boxes)} instances"
    return out
