"""rfdetr tool bundle -- train and run a small station-specific detector (RF-DETR, Apache-2.0).

RF-DETR (Roboflow; https://github.com/roboflow/rf-detr, Apache-2.0, package ``rfdetr``): a real-time
DETR on a DINOv2 backbone, fine-tuned from its COCO checkpoint on a COCO-format dataset. Only the
Apache-2.0 sizes are offered (Nano, Small, Medium, Large); the PML-1.0 sizes of ``rfdetr[plus]``
are not installed.

This bundle calls the package's public API (``RFDETRNano(...).train(...)``, ``.predict(...)``) and
copies no code from it. Changed around it: boxes come back in grounding-dino's
``{detections: [{box, label, score}]}`` shape with the dataset's class names, and a trained model is
a directory (``checkpoint_best_total.pth`` + ``classes.json`` + ``train.json``) that ``detect``
loads once and caches.

The model loads lazily; importing this module never pulls torch. Knobs via env:

- ``GAP_RFDETR_DEVICE`` -- torch device (default ``cuda``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import PerceptionFailed, ToolError
from gap_core.tools import tool
from gap_core.types import BoundingBox2D

logger = logging.getLogger(__name__)

#: Apache-2.0 detection sizes of rfdetr 1.12 (README "License") and their input side in pixels.
SIZES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}
#: Checkpoint rfdetr writes for the best validation score (ema or regular, whichever is better).
BEST = "checkpoint_best_total.pth"

_lock = threading.Lock()
_loaded: dict[str, Any] = {}


class Detection(TypedDict):
    box: BoundingBox2D
    label: str
    score: float


class DetectResult(TypedDict):
    detections: list[Detection]


class TrainResult(TypedDict):
    model_path: str  # directory: checkpoint + classes.json + train.json
    classes: list[str]
    n_train: int
    n_valid: int
    epochs: int
    seconds: float


def _model_class(size: str) -> Any:
    if size not in SIZES:
        raise ToolError("rfdetr", f"size must be one of {sorted(SIZES)} (Apache-2.0), got {size!r}")
    import rfdetr

    return getattr(rfdetr, SIZES[size])


def _coco_split(dataset_dir: str, split: str) -> dict:
    path = os.path.join(dataset_dir, split, "_annotations.coco.json")
    if not os.path.isfile(path):
        raise ToolError("rfdetr.train", f"no COCO annotations at {path}")
    return json.load(open(path))


def _classes(coco: dict) -> list[str]:
    """Class names in category-id order (the order rfdetr's class ids follow)."""
    return [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]


@tool(
    name="rfdetr.train",
    summary="Fine-tune an RF-DETR detector (Apache-2.0 sizes) on a COCO-format dataset folder; returns the model directory.",
    tags=("perception",),
)
def train(
    dataset_dir: str,
    out_dir: str,
    size: str = "nano",
    epochs: int = 40,
    batch_size: int = 8,
    resolution: int = 640,
    lr: float = 1e-4,
) -> TrainResult:
    """``dataset_dir`` holds ``train/`` and ``valid/`` (and optionally ``test/``), each with images and
    ``_annotations.coco.json``. ``resolution`` (multiple of 32) is the square input side: 640 keeps
    10-20 px parts of a 640 x 480 frame at their size (the Nano default, 384, would shrink them).
    Writes the model directory ``out_dir`` and returns it as ``model_path``.
    """
    if resolution % 32:
        raise ToolError("rfdetr.train", f"resolution must be a multiple of 32, got {resolution}")
    if epochs < 1 or batch_size < 1:
        raise ToolError("rfdetr.train", "epochs and batch_size must be >= 1")
    tr, va = _coco_split(dataset_dir, "train"), _coco_split(dataset_dir, "valid")
    classes = _classes(tr)
    if _classes(va) != classes:
        raise ToolError("rfdetr.train", "train and valid list different categories")
    if not tr["annotations"]:
        raise ToolError("rfdetr.train", "the train split has no annotations")
    cls = _model_class(size)
    os.makedirs(out_dir, exist_ok=True)
    t = time.perf_counter()
    try:
        model = cls()
        model.train(
            dataset_dir=dataset_dir,
            output_dir=out_dir,
            epochs=int(epochs),
            batch_size=int(batch_size),
            resolution=int(resolution),
            lr=float(lr),
            device=os.environ.get("GAP_RFDETR_DEVICE", "cuda"),
        )
    except ToolError:
        raise
    except Exception as e:
        raise PerceptionFailed(f"RF-DETR training failed: {e}") from e
    if not os.path.isfile(os.path.join(out_dir, BEST)):
        raise PerceptionFailed(f"RF-DETR training wrote no {BEST} in {out_dir}")
    secs = time.perf_counter() - t
    json.dump(classes, open(os.path.join(out_dir, "classes.json"), "w"))
    meta = {
        "size": size,
        "resolution": int(resolution),
        "epochs": int(epochs),
        "n_train": len(tr["images"]),
        "n_valid": len(va["images"]),
        "seconds": secs,
    }
    json.dump(meta, open(os.path.join(out_dir, "train.json"), "w"))
    return {
        "model_path": out_dir,
        "classes": classes,
        "n_train": meta["n_train"],
        "n_valid": meta["n_valid"],
        "epochs": int(epochs),
        "seconds": secs,
    }


def _load(model_path: str) -> tuple[Any, list[str]]:
    with _lock:
        if model_path not in _loaded:
            ckpt = os.path.join(model_path, BEST)
            if not os.path.isfile(ckpt):
                raise ToolError(
                    "rfdetr.detect", f"no trained model at {model_path} (missing {BEST})"
                )
            meta = json.load(open(os.path.join(model_path, "train.json")))
            classes = json.load(open(os.path.join(model_path, "classes.json")))
            cls = _model_class(meta["size"])
            model = cls(pretrain_weights=ckpt, resolution=meta["resolution"])
            _loaded[model_path] = (model, classes)
        return _loaded[model_path]


def _decode(xyxy: Any, conf: Any, class_id: Any, classes: list[str]) -> list[Detection]:
    """Best-first detections; class ids outside the class list are dropped."""
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    cid = np.asarray(class_id).reshape(-1).astype(int)
    out: list[Detection] = []
    for i in np.argsort(-conf, kind="stable"):
        if not 0 <= cid[i] < len(classes):
            continue
        b = xyxy[i]
        out.append(
            {
                "box": {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])},
                "label": classes[cid[i]],
                "score": float(conf[i]),
            }
        )
    return out


@tool(
    name="rfdetr.detect",
    summary="Run a trained RF-DETR model on an image (labelled boxes + scores, grounding-dino's shape).",
    tags=("perception",),
)
def detect(
    image: np.ndarray, model_path: str, threshold: float = 0.5, query: str | None = None
) -> DetectResult:
    """``model_path``: a directory from ``rfdetr.train``. ``query`` (``"brick. domino"``) keeps only
    those classes, so the tool can be a sahi backend; None keeps all."""
    arr = np.ascontiguousarray(np.asarray(image))
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ToolError(
            "rfdetr.detect", f"expected RGB uint8 [H, W, 3] image, got {arr.dtype} {arr.shape}"
        )
    model, classes = _load(model_path)
    try:
        from PIL import Image

        det = model.predict(Image.fromarray(arr, "RGB"), threshold=float(threshold))
    except Exception as e:
        raise PerceptionFailed(f"RF-DETR detection failed: {e}") from e
    out = _decode(det.xyxy, det.confidence, det.class_id, classes)
    if query:
        keep = {q.strip().rstrip(".").strip() for q in query.split(".") if q.strip(" .")}
        out = [d for d in out if d["label"] in keep]
    return {"detections": out}
