# Adapted from vision-guided-rack-insertion-ot2
# (https://github.com/cooper-group-uol-robotics/vision-guided-rack-insertion-ot2, commit 3bfe3fe87b64),
# reference_implementation/vlm_cnn_train/train_cnn.py (27-class ResNet18 head, augmentation, schedule)
# and training/dataset/build_llava_dataset.py (offset -> label dead-bands), MIT License,
# Copyright (c) 2026 Hadi Beyzaee Juneghani. See NOTICE.
# Changes: labels name world-frame corrections by sign ("+x", "-x", "0" per axis, yaw about +z)
# instead of upstream's "Move Up / Move Left / Rotate Clockwise" strings, whose direction differs
# between upstream's two loops; the dataset is a folder with index.jsonl of measured offsets, not
# LLaVA JSON; a held-out split is reported per axis; inference is in-process.
"""move-classifier tool bundle: train and run a 27-class 'which way to nudge' classifier.

A ResNet18 (ImageNet weights) with upstream's 1024-unit head predicts, from one cropped wrist
image, the correction that would bring the held part over its target: per axis x, y (world,
metres) and yaw (about world +z) one of "+", "-" or "0". Training data is self-labelled: the arm
is jogged from a taught aligned pose to random offsets, and each image's label is the inverse of
its measured offset (skills/correcting-in-plane-offset-by-discrete-moves collects it).

torch / torchvision load lazily; importing this module does not import them.
``GAP_MOVE_CLASSIFIER_DEVICE`` picks the device (default cuda when available).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import tool

logger = logging.getLogger(__name__)

AXES = ("x", "y", "yaw")
SIGNS = ("+", "-", "0")
CLASSES = [f"{a}{b}{c}" for a in SIGNS for b in SIGNS for c in SIGNS]
"""27 classes: the x, y and yaw correction signs, e.g. "+0-" = move +x, keep y, turn -yaw."""

ALIGNED = "000"

#: Upstream's dead-bands (build_llava_dataset.py): 0.5 mm in x and y, 0.2 deg in yaw.
DEADBAND_M = 0.0005
DEADBAND_DEG = 0.2
#: Upstream's training settings (train_cnn.py).
IMAGE_SIZE = 336
EPOCHS = 20
LEARNING_RATE = 1e-4
BATCH_SIZE = 16

_lock = threading.Lock()
_models: dict[str, Any] = {}


def label_for_offset(dx: float, dy: float, dyaw_deg: float, deadband_m: float = DEADBAND_M,
                     deadband_deg: float = DEADBAND_DEG) -> str:
    """The class that undoes an offset of the hand from its aligned pose (move against it)."""

    def sign(v: float, band: float) -> str:
        if abs(v) <= band:
            return "0"
        return "-" if v > 0 else "+"

    return sign(dx, deadband_m) + sign(dy, deadband_m) + sign(dyaw_deg, deadband_deg)


class TrainResult(TypedDict):
    model_path: str
    n_train: int
    n_val: int
    val_accuracy: float  # all three axes right
    val_axis_accuracy: dict[str, float]
    class_counts: dict[str, int]
    epochs: int
    seconds: float


class PredictResult(TypedDict):
    label: str  # e.g. "+0-"
    x: str
    y: str
    yaw: str
    confidence: float  # softmax probability of the label
    aligned: bool  # label == "000"


def _device(torch) -> str:
    want = os.environ.get("GAP_MOVE_CLASSIFIER_DEVICE")
    if want:
        return want
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build(torch, pretrained: bool):
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    class MoveClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            n = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.classifier = nn.Sequential(nn.Linear(n, 1024), nn.ReLU(), nn.Linear(1024, len(CLASSES)))

        def forward(self, x):
            return self.classifier(torch.flatten(self.backbone(x), 1))

    return MoveClassifier()


def _transforms(train: bool, size: int):
    from torchvision import transforms as T

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if not train:
        return T.Compose([T.Resize((size, size)), T.ToTensor(), norm])
    return T.Compose(
        [
            T.Resize((size, size)),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.3),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
            T.ToTensor(),
            norm,
        ]
    )


def read_index(dataset_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(dataset_dir)
    rows = [json.loads(line) for line in (root / "index.jsonl").read_text().splitlines() if line.strip()]
    for r in rows:
        r["path"] = str(root / r["image"])
    return rows


@tool(
    name="move-classifier.train",
    summary="Train the 27-class nudge classifier on a folder of self-labelled wrist crops (index.jsonl).",
    tags=("learning",),
)
def train(
    dataset_dir: str,
    model_path: str,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    image_size: int = IMAGE_SIZE,
    val_fraction: float = 0.15,
    deadband_m: float = DEADBAND_M,
    deadband_deg: float = DEADBAND_DEG,
    seed: int = 0,
) -> TrainResult:
    """Fit on ``dataset_dir/index.jsonl`` rows ``{image, dx, dy, dyaw_deg}`` (the hand's measured
    offset from the aligned pose, world frame) and write the best-validation weights to ``model_path``."""
    t0 = time.perf_counter()
    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise ToolError("move-classifier.train", f"torch / torchvision unavailable: {exc}") from exc
    rows = read_index(dataset_dir)
    labels = [CLASSES.index(label_for_offset(r["dx"], r["dy"], r["dyaw_deg"], deadband_m, deadband_deg)) for r in rows]
    if len(rows) < 20:
        raise ToolError("move-classifier.train", f"only {len(rows)} samples in {dataset_dir}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    n_val = max(1, int(len(rows) * val_fraction))
    val_idx, tr_idx = order[:n_val], order[n_val:]

    class Rows(Dataset):
        def __init__(self, idx, train_mode):
            self.idx, self.tf = list(idx), _transforms(train_mode, image_size)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, k):
            i = self.idx[k]
            return self.tf(Image.open(rows[i]["path"]).convert("RGB")), labels[i]

    torch.manual_seed(seed)
    dev = _device(torch)
    model = _build(torch, pretrained=True).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    tl = DataLoader(Rows(tr_idx, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    vl = DataLoader(Rows(val_idx, False), batch_size=BATCH_SIZE, num_workers=4)
    best, best_pred = float("inf"), None
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(epochs)):
        model.train()
        for x, y in tl:
            opt.zero_grad()
            loss = loss_fn(model(x.to(dev)), y.to(dev))
            loss.backward()
            opt.step()
        model.eval()
        vloss, preds = 0.0, []
        with torch.no_grad():
            for x, y in vl:
                out = model(x.to(dev))
                vloss += float(loss_fn(out, y.to(dev))) * len(y)
                preds += out.argmax(1).tolist()
        vloss /= len(val_idx)
        logger.info("move-classifier.train: epoch %d val loss %.4f", epoch + 1, vloss)
        if vloss < best:
            best, best_pred = vloss, preds
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "image_size": image_size}, model_path)
        sched.step()
    truth = [CLASSES[labels[i]] for i in val_idx]
    pred = [CLASSES[p] for p in best_pred]
    axis_acc = {a: float(np.mean([t[k] == p[k] for t, p in zip(truth, pred, strict=True)])) for k, a in enumerate(AXES)}
    counts: dict[str, int] = {}
    for i in labels:
        counts[CLASSES[i]] = counts.get(CLASSES[i], 0) + 1
    return {
        "model_path": str(model_path),
        "n_train": len(tr_idx),
        "n_val": len(val_idx),
        "val_accuracy": float(np.mean([t == p for t, p in zip(truth, pred, strict=True)])),
        "val_axis_accuracy": axis_acc,
        "class_counts": counts,
        "epochs": int(epochs),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def _load(model_path: str):
    with _lock:
        if model_path not in _models:
            import torch

            ck = torch.load(model_path, map_location="cpu", weights_only=False)
            if list(ck.get("classes", [])) != CLASSES:
                raise ValueError(f"{model_path} was trained with other classes")
            dev = _device(torch)
            model = _build(torch, pretrained=False)
            model.load_state_dict(ck["state_dict"])
            model.to(dev).eval()
            _models[model_path] = (torch, model, dev, _transforms(False, int(ck.get("image_size", IMAGE_SIZE))))
        return _models[model_path]


@tool(
    name="move-classifier.predict",
    summary="Which way to nudge (x, y, yaw signs, world frame) from one cropped wrist image.",
    tags=("perception",),
)
def predict(image: np.ndarray, model_path: str) -> PredictResult:
    """Classify a uint8 RGB crop with the weights at ``model_path`` (from move-classifier.train)."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"move-classifier expects a uint8 [H, W, 3] image, got {arr.dtype} {arr.shape}")
    try:
        torch, model, dev, tf = _load(str(model_path))
    except Exception as exc:
        raise ToolError("move-classifier.predict", f"cannot load {model_path}: {exc}") from exc
    from PIL import Image

    x = tf(Image.fromarray(np.ascontiguousarray(arr[..., :3]), "RGB")).unsqueeze(0).to(dev)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0].cpu().numpy()
    k = int(np.argmax(p))
    lab = CLASSES[k]
    return {"label": lab, "x": lab[0], "y": lab[1], "yaw": lab[2], "confidence": round(float(p[k]), 4), "aligned": lab == ALIGNED}
