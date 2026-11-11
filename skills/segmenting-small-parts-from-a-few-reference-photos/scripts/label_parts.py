# After No-Time-to-Train (Espinosa et al., "No time to train! Training-Free Reference-Based Instance
# Segmentation", 2025; github.com/miquel-espinosa/no-time-to-train at e93a181837bd, Apache-2.0):
# a memory bank of DINOv2 features pooled under the masks of a few labelled reference parts, and
# mask proposals of a new frame labelled by feature matching against it. Re-implemented on the hub's
# tools (no upstream code): proposals come from the caller or sam3.segment_text, features from
# tools/dinov2, and the matching is the mean of the top-k cosines per class with a score and a margin
# gate in place of upstream's semantic-aware soft merging of SAM2 proposals.
"""Label part masks in a frame from a handful of labelled reference parts (no training)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

TOP_K = 3
"""A class's score is the mean of a proposal's TOP_K highest cosines to that class's references
(fewer when the class has fewer): one look-alike reference cannot carry a class alone."""

MIN_SCORE = 0.45
"""Lowest class score [cosine of the tools/dinov2 vectors] for a proposal to get a label. On the
held-out renders (SKILL.md) correct labels scored a median 0.73, wrong ones 0.58; below 0.45 is
something not in the bank."""

MIN_MARGIN = 0.0
"""Lowest gap between the best and second class score; raise it to trade recall for precision
between look-alike classes (a 2x3 against a 2x4 brick)."""

MIN_AREA_PX = 20
"""Proposals smaller than this [px] are dropped: under ~4 x 5 px no DINOv2 patch sees the part."""


def classify(
    queries: np.ndarray,
    bank: np.ndarray,
    labels: list[str],
    min_score: float = MIN_SCORE,
    min_margin: float = MIN_MARGIN,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """Per query row: {label (None under the gates), score, margin, scores per class}."""
    q = np.asarray(queries, np.float32)
    b = np.asarray(bank, np.float32)
    classes = sorted(set(labels))
    lab = np.asarray(labels)
    sim = q @ b.T
    per = np.zeros((len(q), len(classes)), np.float32)
    for j, c in enumerate(classes):
        s = np.sort(sim[:, lab == c], axis=1)[:, ::-1]
        per[:, j] = s[:, : min(top_k, s.shape[1])].mean(axis=1)
    out = []
    for i in range(len(q)):
        order = np.argsort(-per[i])
        best = float(per[i, order[0]])
        second = float(per[i, order[1]]) if len(classes) > 1 else -1.0
        margin = best - second
        ok = best >= min_score and margin >= min_margin
        out.append(
            {
                "label": classes[order[0]] if ok else None,
                "best_class": classes[order[0]],
                "score": round(best, 4),
                "margin": round(margin, 4),
                "scores": {c: round(float(per[i, j]), 4) for j, c in enumerate(classes)},
            }
        )
    return out


def _frame(ctx: NodeContext, frame: dict[str, Any] | None, camera: str | None) -> Any:
    if frame is not None:
        return frame
    obs = ctx.tool("robot.get_observation")
    cams = [c for c in obs.get("cameras") or [] if c.get("rgb") is not None]
    if camera:
        return next((c for c in cams if c.get("name") == camera), None)
    return (cams or [None])[0]


def _rgb(f: Any) -> np.ndarray | None:
    if f is None:
        return None
    a = f.get("rgb") if isinstance(f, dict) else f
    return None if a is None else np.asarray(a)


def _area(m: Any, box: Any) -> float:
    if m is not None:
        return float(np.count_nonzero(np.asarray(m)))
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox(m: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def build_bank(ctx: NodeContext, references: list[dict[str, Any]], default_image: Any) -> dict:
    """Embed every reference part: {labels, embeddings [N, D]}. Each reference is {label, mask | box,
    image (optional; else ``default_image``)}; references sharing an image are embedded in one call."""
    groups: dict[int, tuple[np.ndarray, list[dict[str, Any]]]] = {}
    for r in references:
        img = _rgb(r.get("image")) if r.get("image") is not None else _rgb(default_image)
        if img is None or ("mask" not in r and "box" not in r) or not r.get("label"):
            continue
        groups.setdefault(id(img), (img, []))[1].append(r)
    labels, embs = [], []
    for img, refs in groups.values():
        for kind in ("mask", "box"):
            rs = [r for r in refs if kind in r]
            if not rs:
                continue
            arg = (
                {"masks": [r["mask"] for r in rs]}
                if kind == "mask"
                else {"boxes": [r["box"] for r in rs]}
            )
            res = ctx.tool("dinov2.embed_regions", image=img, **arg)
            embs.append(np.asarray(res["embeddings"], np.float32))
            labels += [str(r["label"]) for r in rs]
    if not labels:
        return {"labels": [], "embeddings": np.zeros((0, 0), np.float32)}
    return {"labels": labels, "embeddings": np.concatenate(embs)}


def run(
    ctx: NodeContext,
    references: list[dict[str, Any]] | None = None,
    bank: dict[str, Any] | None = None,
    masks: list[Any] | None = None,
    boxes: list[Any] | None = None,
    prompt: str | None = None,
    frame: dict[str, Any] | None = None,
    camera: str | None = None,
    reference_frame: dict[str, Any] | None = None,
    min_score: float = MIN_SCORE,
    min_margin: float = MIN_MARGIN,
    top_k: int = TOP_K,
    keep_unknown: bool = False,
) -> dict[str, Any]:
    """Label the frame's part proposals (``masks`` | ``boxes`` | sam3 ``prompt``) against a bank."""
    out: dict[str, Any] = {"instances": [], "counts": {}, "n_proposals": 0, "detail": ""}
    if bank is None or not bank.get("labels"):
        bank = build_bank(ctx, references or [], reference_frame) if references else None
    if bank is None or not bank.get("labels"):
        out.update(
            route="bank_empty", bank=None, detail="no usable reference (label + mask/box + image)"
        )
        return out
    out["bank"] = {"labels": list(bank["labels"]), "embeddings": np.asarray(bank["embeddings"])}
    out["classes"] = sorted(set(bank["labels"]))

    f = _frame(ctx, frame, camera)
    img = _rgb(f)
    if img is None:
        out.update(route="none_found", detail="no RGB frame")
        return out
    if masks is None and boxes is None and prompt:
        seg = ctx.tool("sam3.segment_text", image=img, prompt=prompt)
        masks = [d["mask"] for d in seg.get("detections", seg.get("instances", [])) if "mask" in d]
    props = masks if masks is not None else boxes or []
    keep = [i for i, p in enumerate(props)
            if _area(p if masks is not None else None, None if masks is not None else p) >= MIN_AREA_PX]  # fmt: skip
    out["n_proposals"] = len(props)
    if not keep:
        out.update(route="none_found", detail=f"{len(props)} proposals, none of {MIN_AREA_PX}+ px")
        return out
    arg = ({"masks": [props[i] for i in keep]} if masks is not None
           else {"boxes": [props[i] for i in keep]})  # fmt: skip
    res = ctx.tool("dinov2.embed_regions", image=img, **arg)
    labs = classify(res["embeddings"], out["bank"]["embeddings"], out["bank"]["labels"],
                    min_score, min_margin, top_k)  # fmt: skip
    for i, lab in zip(keep, labs, strict=True):
        if lab["label"] is None and not keep_unknown:
            continue
        p = props[i]
        box = _bbox(np.asarray(p) > 0) if masks is not None else [int(round(float(v))) for v in p]
        out["instances"].append({"index": i, "box": box, **lab})
        if lab["label"] is not None:
            out["counts"][lab["label"]] = out["counts"].get(lab["label"], 0) + 1
    n = sum(out["counts"].values())
    out["route"] = "instances_found" if n else "none_found"
    out["detail"] = (
        f"{n} of {len(keep)} proposals labelled ({len(out['bank']['labels'])} references)"
    )
    logger.info("label_parts: %s", out["detail"])
    return out
