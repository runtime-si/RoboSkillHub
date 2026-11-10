# Training a part detector from auto-labels: a big open-vocabulary teacher boxes a folder of
# station frames from a caption ontology, the boxes become a COCO dataset, a small RF-DETR student
# is fine-tuned on it.
#
# Reimplemented from autodistill (https://github.com/autodistill/autodistill, Apache-2.0, commit
# 838e1c39f9fd): CaptionOntology (prompt -> class), base_model.label (label a folder, write a
# dataset with a train / valid split) and target_model.train. No code copied. Changed: the teacher
# is any hub detector reached through sahi.detect_sliced (slicing for 10-30 px parts), the target
# is tools/rfdetr (Apache-2.0 sizes only, not autodistill's YOLOv8 / ultralytics AGPL target),
# and the labels are checked before training (per-class counts, empty frames, score floor).
"""Label station frames with an open-vocabulary teacher and train a small detector on the labels."""

from __future__ import annotations

import json
import os
import random
import shutil
from typing import Any, TypedDict

import numpy as np

#: Below this many labels of a class there is nothing to learn it from (autodistill's examples use
#: tens to hundreds of images; a 2-class run here had 300-1100 labels a class).
DEFAULT_MIN_LABELS = 50
#: Share of frames the teacher left empty above which its labels are not trusted (the frames were
#: chosen to show the parts; an empty majority means wrong prompts or wrong scale).
DEFAULT_MAX_EMPTY = 0.5
#: Share of frames kept for validation (autodistill's default split is 0.8 / 0.2; 0.15 here).
DEFAULT_VALID = 0.15


class Result(TypedDict):
    route: str
    model_path: str
    counts: dict[str, int]
    n_images: int
    n_empty: int
    median_score: float
    dataset_dir: str
    reason: str


def _load(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _coco(entries: list[dict], classes: list[str], split_dir: str) -> None:
    os.makedirs(split_dir, exist_ok=True)
    images, anns = [], []
    for i, e in enumerate(entries):
        name = f"{i:05d}{os.path.splitext(e['path'])[1] or '.png'}"
        shutil.copyfile(e["path"], os.path.join(split_dir, name))
        images.append({"id": i, "file_name": name, "height": e["h"], "width": e["w"]})
        for b, c in e["boxes"]:
            x1, y1, x2, y2 = b
            anns.append(
                {
                    "id": len(anns),
                    "image_id": i,
                    "category_id": classes.index(c),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
    cats = [{"id": k, "name": c, "supercategory": "part"} for k, c in enumerate(classes)]
    json.dump(
        {"images": images, "annotations": anns, "categories": cats},
        open(os.path.join(split_dir, "_annotations.coco.json"), "w"),
    )


IMAGE_EXT = (".png", ".jpg", ".jpeg")


def parse_ontology(ontology: Any) -> dict[str, str]:
    """``{"lego brick": "brick"}`` or ``"lego brick=brick; domino=domino"`` (a bare prompt is its own class)."""
    if isinstance(ontology, dict):
        out = {str(k).strip(): str(v).strip() for k, v in ontology.items()}
    else:
        out = {}
        for item in str(ontology).split(";"):
            if not item.strip():
                continue
            prompt, _, cls = item.partition("=")
            out[prompt.strip()] = (cls or prompt).strip()
    out = {k: v for k, v in out.items() if k and v}
    if not out:
        raise ValueError(f"ontology {ontology!r} names no prompt")
    return out


def run(
    ctx: Any,
    frames_dir: str,
    ontology: Any,
    out_dir: str,
    teacher: str = "mm-grounding-dino.detect",
    slice_px: int = 320,
    min_score: float = 0.25,
    min_labels: int = DEFAULT_MIN_LABELS,
    max_empty: float = DEFAULT_MAX_EMPTY,
    valid_fraction: float = DEFAULT_VALID,
    size: str = "nano",
    epochs: int = 40,
    resolution: int = 640,
    seed: int = 0,
    image_paths: list[str] | None = None,
) -> Result:
    """Label the frames in ``frames_dir`` (or ``image_paths``) with ``teacher`` and train ``rfdetr``.

    ``ontology`` maps a teacher prompt to a class name (``"lego brick=brick; domino=domino"`` or a
    dict; several prompts may share a class). Every frame goes through
    ``sahi.detect_sliced`` (``slice_px=0``: one call on the frame) and boxes at ``min_score`` or
    more keep the class of their prompt. The dataset is written to ``out_dir/dataset`` and the
    model to ``out_dir/model``.
    """
    if image_paths is None:
        if not os.path.isdir(frames_dir):
            raise ValueError(f"frames_dir {frames_dir!r} is not a folder")
        image_paths = sorted(
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith(IMAGE_EXT)
        )
    if not image_paths:
        raise ValueError(f"no frames ({', '.join(IMAGE_EXT)}) in {frames_dir!r}")
    ontology = parse_ontology(ontology)
    prompts = list(ontology)
    classes = sorted(set(ontology.values()))
    entries, scores = [], []
    for path in image_paths:
        img = _load(path)
        r = ctx.tool(
            "sahi.detect_sliced",
            image=img,
            prompts=prompts,
            backend=teacher,
            slice_px=int(slice_px),
            min_score=float(min_score),
        )
        boxes = []
        for d in r["detections"]:
            cls = ontology.get(d["label"])
            if cls is None:
                continue
            b = d["box"]
            box = [float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])]
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            boxes.append((box, cls))
            scores.append(float(d["score"]))
        entries.append(
            {"path": path, "h": int(img.shape[0]), "w": int(img.shape[1]), "boxes": boxes}
        )
    counts = {c: sum(c == k for e in entries for _, k in e["boxes"]) for c in classes}
    n_empty = sum(not e["boxes"] for e in entries)
    median = float(np.median(scores)) if scores else 0.0
    base: dict[str, Any] = {
        "model_path": "",
        "counts": counts,
        "n_images": len(entries),
        "n_empty": n_empty,
        "median_score": median,
        "dataset_dir": "",
    }
    if n_empty / len(entries) > max_empty:
        return {
            "route": "label_quality_low",
            **base,
            "reason": f"the teacher found nothing in {n_empty} of {len(entries)} frames "
            f"(more than {max_empty:.0%}): wrong prompts, scale or frames",
        }
    few = {c: n for c, n in counts.items() if n < min_labels}
    if few:
        return {
            "route": "too_few_labels",
            **base,
            "reason": f"classes under {min_labels} labels: {few}; add frames showing them",
        }
    rng = random.Random(seed)
    order = list(range(len(entries)))
    rng.shuffle(order)
    n_valid = max(1, round(valid_fraction * len(entries)))
    valid = [entries[i] for i in order[:n_valid]]
    train = [entries[i] for i in order[n_valid:]]
    ds = os.path.join(out_dir, "dataset")
    shutil.rmtree(ds, ignore_errors=True)
    _coco(train, classes, os.path.join(ds, "train"))
    _coco(valid, classes, os.path.join(ds, "valid"))
    _coco(valid, classes, os.path.join(ds, "test"))
    t = ctx.tool(
        "rfdetr.train",
        dataset_dir=ds,
        out_dir=os.path.join(out_dir, "model"),
        size=size,
        epochs=int(epochs),
        resolution=int(resolution),
    )
    return {
        "route": "detector_trained",
        **base,
        "model_path": t["model_path"],
        "dataset_dir": ds,
        "reason": f"trained {size} for {epochs} epochs on {len(train)} frames "
        f"({sum(len(e['boxes']) for e in train)} labels), validated on {len(valid)}",
    }
