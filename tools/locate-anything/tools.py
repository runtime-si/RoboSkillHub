"""LocateAnything-3B tool bundle: detection, phrase grounding and pointing, local GPU.

Runs NVIDIA's LocateAnything-3B (https://huggingface.co/nvidia/LocateAnything-3B)
through the reference worker NVlabs ships in Eagle Embodied
(https://github.com/NVlabs/Eagle/blob/main/Embodied/locateanything_worker.py).
Neither is copied into this repository: the weights come from the Hugging Face
Hub at a pinned commit, the worker from GitHub at a pinned commit with its
sha256 checked, both on first use. Both are under the NVIDIA License,
non-commercial (research or evaluation) use only, so every tool refuses to run
until ``GAP_LOCATE_ANYTHING_NONCOMMERCIAL=1`` acknowledges that.

The model answers in its own token format, ``<ref>label</ref><box><x1><y1><x2><y2></box>``
for boxes and ``<box><x><y></box>`` for points, coordinates normalized to
0-1000 in x, y order (Gemini's are y, x). They are converted to pixels here
and returned in the shapes ``gemini-er.detect`` / ``gemini-er.point`` return,
so a skill can switch backend by tool name.

The worker loads lazily on first call (module-level singleton); importing this
module never imports torch or transformers. Knobs via env:

- ``GAP_LOCATE_ANYTHING_NONCOMMERCIAL`` -- must be ``1``: the user's statement
  that the use is non-commercial research or evaluation.
- ``GAP_LOCATE_ANYTHING_DEVICE`` -- torch device (default ``cuda``).
- ``GAP_LOCATE_ANYTHING_MODEL`` -- a local model directory instead of the
  pinned Hub snapshot.
- ``GAP_LOCATE_ANYTHING_WORKER`` -- a local ``locateanything_worker.py``
  instead of the pinned download (its digest is not checked).
- ``GAP_LOCATE_ANYTHING_RUNTIME`` -- ``batch`` (default, the release's
  ``batch_utils`` hybrid scheduler) or ``standard`` (transformers ``generate``).
- ``GAP_LOCATE_ANYTHING_ATTN`` -- batch runtime attention: ``sdpa`` (default)
  or ``la_flash`` (needs flash-attn).
"""

from __future__ import annotations

import difflib
import hashlib
import importlib.util
import logging
import os
import re
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import tool
from gap_core.types import BoundingBox2D
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_REPO = "nvidia/LocateAnything-3B"
#: Hub commit of the checkpoint, checked live on 2026-11-07.
MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"

#: Eagle commit the worker is fetched at (main on 2026-11-07; the worker last
#: changed in bb860a7, 2026-06-24). VFlash lives in the same tree.
EAGLE_COMMIT = "06edcff802dbdd167d196c7a1c5b44929e07d951"
WORKER_URL = f"https://raw.githubusercontent.com/NVlabs/Eagle/{EAGLE_COMMIT}/Embodied/locateanything_worker.py"
WORKER_SHA256 = "021f26b07220b9dcb090023286ef52f0a55392d0c43ec1a0490645a2c5c165a4"

ACK_ENV = "GAP_LOCATE_ANYTHING_NONCOMMERCIAL"
_ACK_VALUES = frozenset({"1", "true", "yes"})

#: Greedy decoding. The worker's default is 0.7 with sampling; greedy keeps two
#: calls on the same frame identical, which a graph that re-perceives relies on.
_TEMPERATURE = 0.0
#: The worker's default; a dense detection emits ~9 tokens a box.
_MAX_NEW_TOKENS = 2048

#: The release's batch_utils runtime with sdpa attention. On the 2026-11-07
#: live check (4 YAM station frames, 23 grounding/pointing queries) it scored
#: 18/23 to the standard transformers path's 14/23 at the same greedy settings,
#: and was no slower.
DEFAULT_RUNTIME = "batch"

_load_lock = threading.Lock()
_worker: Any = None


class Detection(TypedDict):
    box: BoundingBox2D  # [x1, y1, x2, y2] in pixels
    label: str
    score: float


class DetectResult(TypedDict):
    detections: list[Detection]


class Point(TypedDict):
    pixel_x: float
    pixel_y: float
    label: str


class PointResult(TypedDict):
    pixel_x: float
    pixel_y: float
    found: bool
    points: list[Point]


# ---------------------------------------------------------------------------
# Runtime: the pinned worker over the pinned snapshot
# ---------------------------------------------------------------------------


def _require_ack(tool_name: str) -> None:
    if os.environ.get(ACK_ENV, "").strip().lower() not in _ACK_VALUES:
        raise ToolError(
            tool_name,
            "LocateAnything-3B is under the NVIDIA License for non-commercial research or "
            f"evaluation use only; set {ACK_ENV}=1 to confirm the use is non-commercial, "
            "or use gemini-er / grounding-dino instead",
        )


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "roboskillhub" / "locate-anything" / f"eagle-{EAGLE_COMMIT[:12]}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _worker_path(*, download: bool = True) -> Path | None:
    """The worker source: the env override, else the pinned download (fetched once)."""
    override = os.environ.get("GAP_LOCATE_ANYTHING_WORKER")
    if override:
        return Path(override)
    path = _cache_dir() / "locateanything_worker.py"
    if path.is_file() and _sha256(path) == WORKER_SHA256:
        return path
    if not download:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(WORKER_URL, timeout=60) as resp:
        tmp.write_bytes(resp.read())
    got = _sha256(tmp)
    if got != WORKER_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{WORKER_URL}: sha256 {got} is not the pinned {WORKER_SHA256}")
    tmp.replace(path)
    return path


def _model_dir() -> str:
    override = os.environ.get("GAP_LOCATE_ANYTHING_MODEL")
    if override:
        return override
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, repo_type="model")


def _import_worker_class(path: Path):
    spec = importlib.util.spec_from_file_location("locateanything_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LocateAnythingWorker


def _build_worker() -> Any:
    model_dir = _model_dir()
    worker_cls = _import_worker_class(_worker_path())
    device = os.environ.get("GAP_LOCATE_ANYTHING_DEVICE", "cuda")
    runtime = os.environ.get("GAP_LOCATE_ANYTHING_RUNTIME", DEFAULT_RUNTIME).strip().lower()
    if runtime == "batch":
        # batch_utils / kernel_utils ship in the model snapshot, not as a package.
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        attn = os.environ.get("GAP_LOCATE_ANYTHING_ATTN", "sdpa")
        logger.info("Loading LocateAnything-3B (batch runtime, attn=%s) from %s", attn, model_dir)
        return worker_cls(model_dir, device=device, use_batch_runtime=True, attn=attn)
    if runtime != "standard":
        raise ValueError(f"GAP_LOCATE_ANYTHING_RUNTIME must be standard or batch, got {runtime!r}")
    logger.info("Loading LocateAnything-3B on %s from %s", device, model_dir)
    return worker_cls(model_dir, device=device)


def _get_worker() -> Any:
    global _worker
    with _load_lock:
        if _worker is None:
            _worker = _build_worker()
        return _worker


def weights_cached() -> bool | None:
    """Filesystem-only probe for ``gap check``: the pinned snapshot and worker are on disk."""
    if os.environ.get("GAP_LOCATE_ANYTHING_MODEL"):
        model_ok = Path(os.environ["GAP_LOCATE_ANYTHING_MODEL"], "config.json").is_file()
    else:
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return None
        try:
            hit = try_to_load_from_cache(MODEL_REPO, "config.json", revision=MODEL_REVISION)
        except Exception:
            return None
        model_ok = isinstance(hit, str)
    return model_ok and _worker_path(download=False) is not None


def prefetch() -> None:
    """Download the pinned snapshot and worker (``gap skills check --download``)."""
    logger.info("[locate-anything] prefetching %s@%s ...", MODEL_REPO, MODEL_REVISION)
    _model_dir()
    _worker_path()
    logger.info("[locate-anything] prefetch complete")


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"<ref>(.*?)</ref>|<box>(.*?)</box>", flags=re.DOTALL)
_COORD_RE = re.compile(r"<(\d+(?:\.\d+)?)>")


def _clamp(v: float, hi: float) -> float:
    return max(0.0, min(hi, v))


def _parse(answer: str, *, width: int, height: int, default_label: str):
    """Boxes and points of one answer in pixels, each labelled by the ``<ref>`` before it.

    ``<box>None</box>`` (the model's "not found") and malformed boxes are skipped.
    """
    boxes: list[tuple[BoundingBox2D, str]] = []
    points: list[Point] = []
    label = default_label
    for ref, body in _TOKEN_RE.findall(answer or ""):
        if ref:
            # The batch runtime echoes the prompt's closing period into the ref.
            label = ref.strip().rstrip(".").strip() or default_label
            continue
        coords = [float(c) for c in _COORD_RE.findall(body)]
        if len(coords) == 4:
            x1, y1, x2, y2 = coords
            box: BoundingBox2D = {
                "x1": _clamp(min(x1, x2) / 1000.0 * width, width),
                "y1": _clamp(min(y1, y2) / 1000.0 * height, height),
                "x2": _clamp(max(x1, x2) / 1000.0 * width, width),
                "y2": _clamp(max(y1, y2) / 1000.0 * height, height),
            }
            boxes.append((box, label))
        elif len(coords) == 2:
            points.append(
                {
                    "pixel_x": _clamp(coords[0] / 1000.0 * width, width),
                    "pixel_y": _clamp(coords[1] / 1000.0 * height, height),
                    "label": label,
                }
            )
    return boxes, points


def _validate_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"locate-anything expects a uint8 [H, W, 3] RGB array, got "
            f"dtype={arr.dtype} shape={arr.shape}"
        )
    return np.ascontiguousarray(arr)


def _ask(tool_name: str, method: str, image: np.ndarray, arg: Any) -> tuple[str, int, int]:
    """Run one worker query; return the raw answer and the frame size."""
    _require_ack(tool_name)
    arr = _validate_image(image)
    height, width = arr.shape[:2]
    pil_image = Image.fromarray(arr, "RGB")
    try:
        worker = _get_worker()
        result = getattr(worker, method)(
            pil_image,
            arg,
            temperature=_TEMPERATURE,
            max_new_tokens=_MAX_NEW_TOKENS,
            verbose=False,
        )
    except Exception as exc:
        raise ToolError(tool_name, f"LocateAnything-3B failed: {exc}") from exc
    answer = result.get("answer", "") if isinstance(result, dict) else str(result)
    return str(answer or ""), width, height


def _detections(boxes) -> DetectResult:
    return {"detections": [{"box": box, "label": label, "score": 1.0} for box, label in boxes]}


def _snap_label(label: str, categories: list[str]) -> str:
    """The requested category a ``<ref>`` names.

    The batch runtime's parallel decoding sometimes garbles the ref text
    ("gripperper" for "gripper" on the 2026-11-07 frames); a ref within
    difflib ratio 0.6 of a category is that category, anything else is kept.
    """
    by_lower = {c.lower(): c for c in categories}
    if label.lower() in by_lower:
        return by_lower[label.lower()]
    close = difflib.get_close_matches(label.lower(), list(by_lower), n=1, cutoff=0.6)
    return by_lower[close[0]] if close else label


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="locate-anything.detect",
    summary="Detect every instance of one or more categories with LocateAnything-3B (non-commercial).",
    tags=("perception",),
)
def detect(image: np.ndarray, categories: list[str] | str) -> DetectResult:
    """Boxes for every instance of each category in *image*.

    Args:
        image: uint8 [H, W, 3] RGB array.
        categories: Category names (``["cube", "box"]``) or one name.

    Returns:
        ``{"detections": [{"box": BoundingBox2D, "label": str, "score": float}]}``
        in pixels, ``label`` the category the box was emitted under. The model
        reports no confidence, so ``score`` is 1.0. Empty when nothing is
        found (never an error).
    """
    cats = [categories] if isinstance(categories, str) else [str(c) for c in categories]
    cats = [c.strip() for c in cats if c and c.strip()]
    if not cats:
        raise ValueError("locate-anything.detect needs at least one category")
    answer, width, height = _ask("locate-anything.detect", "detect", image, cats)
    boxes, _ = _parse(answer, width=width, height=height, default_label=cats[0])
    if not boxes:
        logger.debug("locate-anything.detect: nothing for %r (raw=%r)", cats, answer[:200])
    return _detections([(box, _snap_label(label, cats)) for box, label in boxes])


@tool(
    name="locate-anything.ground",
    summary="Boxes for what a referring phrase names, via LocateAnything-3B (non-commercial).",
    tags=("perception",),
)
def ground(image: np.ndarray, query: str, single: bool = False) -> DetectResult:
    """Boxes for the object(s) *query* refers to; the drop-in for ``gemini-er.detect``.

    Args:
        image: uint8 [H, W, 3] RGB array.
        query: Referring phrase ("the small dark grey cube left of the box").
        single: Ask for one instance (the model's single-instance grounding
            prompt) instead of every instance.

    Returns:
        ``{"detections": [...]}`` as :func:`detect`, ``label`` the phrase.
    """
    method = "ground_single" if single else "ground_multi"
    answer, width, height = _ask("locate-anything.ground", method, image, query)
    boxes, _ = _parse(answer, width=width, height=height, default_label=query)
    if not boxes:
        logger.debug("locate-anything.ground: nothing for %r (raw=%r)", query, answer[:200])
    return _detections(boxes)


@tool(
    name="locate-anything.point",
    summary="Point at what a phrase names via LocateAnything-3B (non-commercial); returns pixel points.",
    tags=("perception",),
)
def point(image: np.ndarray, query: str, max_points: int = 10) -> PointResult:
    """Point at *query* in *image*; the drop-in for ``gemini-er.point``.

    Args:
        image: uint8 [H, W, 3] RGB array.
        query: What to point at ("the free end of the white cable").
        max_points: Most points returned (the model is not told the cap).

    Returns:
        ``{"pixel_x", "pixel_y", "found", "points"}``: the first point in
        pixels and every point as ``{"pixel_x", "pixel_y", "label"}``.
        ``found=False`` with zeros and no points when nothing is found.
    """
    answer, width, height = _ask("locate-anything.point", "point", image, query)
    _, points = _parse(answer, width=width, height=height, default_label=query)
    points = points[: max(1, int(max_points))]
    if not points:
        logger.debug("locate-anything.point: nothing for %r (raw=%r)", query, answer[:200])
        return {"pixel_x": 0.0, "pixel_y": 0.0, "found": False, "points": []}
    return {
        "pixel_x": points[0]["pixel_x"],
        "pixel_y": points[0]["pixel_y"],
        "found": True,
        "points": points,
    }
