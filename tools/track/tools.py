"""track tool bundle: follow chosen pixels through a video, occlusions included.

Runs Google DeepMind's TAPNext++ (https://github.com/google-deepmind/tapnet,
Apache-2.0 for the code and, per its README's "License and Disclaimer", for
every released checkpoint including TAPNext++) through the ``tapnet`` package at
a pinned commit. The package is a dependency of this bundle's environment and
the checkpoint is downloaded from DeepMind's bucket on first use, checked
against its pinned SHA-256. Adapted from tapnet (Apache-2.0, Copyright 2026
Google LLC): the frame preprocessing of ``tapnextpp/votsp2026/utils.py``
(``preprocess_frame``, taking RGB instead of BGR) and the support-point layout
of ``tapnextpp/votsp2026/tracker.py`` (``grid_support_points`` /
``local_support_points``, vectorised). Changed: the pixel <-> model mapping
adds the half-pixel shift ``display_to_model`` leaves out, and confidence uses
``tapnext_torch_utils.tracker_certainty`` as upstream's colab does.

TAPNext++ is an online (causal, recurrent) point tracker: it is given query
pixels on one frame and then one frame at a time, and for each frame returns
every query's position and a visibility logit, carrying a recurrent state
between frames. It keeps tracking through occlusions and re-detects a point
when it comes back. This module wraps it two ways:

- ``track.points(frames, query_points)``: a whole clip at once; queries may sit
  on any frame (the clip is tracked forwards and backwards from it).
- ``track.start`` / ``track.update`` / ``track.stop``: a session that takes one
  live frame per call, for closed-loop feedback (a cable tip in the wrist
  camera while the arm moves).

Pixels in and out are OpenCV pixel coordinates ``[u, v]`` (pixel centres on
integers) of the frames as given; the model sees each frame resized to a square.

The model loads lazily on first call (module-level singleton); importing this
module never imports torch or tapnet. Knobs via env:

- ``GAP_TRACK_DEVICE`` -- torch device (default ``cuda``).
- ``GAP_TRACK_CHECKPOINT`` -- ``512`` (default) or ``256``: the released
  TAPNext++ checkpoint and the square input resolution it runs at.
- ``GAP_TRACK_CHECKPOINT_PATH`` -- a local checkpoint file instead of the
  pinned download (its digest is not checked).
- ``GAP_TRACK_CACHE`` -- download directory (default ``~/.cache/roboskillhub/track``).
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import os
import threading
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import tool

logger = logging.getLogger(__name__)

#: tapnet commit the bundle was checked with (pyproject pins the same).
TAPNET_COMMIT = "ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b"

#: The two released TAPNext++ checkpoints (tapnet README, "TAPNext++" rows),
#: with the size and MD5 the bucket reported on 2026-11-07 and the SHA-256 of
#: the downloaded file.
CHECKPOINTS: dict[str, dict[str, Any]] = {
    "512": {
        "url": "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt",
        "file": "tapnextpp_512.ckpt",
        "bytes": 2532283010,
        "md5": "eea4fffa043f4f28503c130d826b116c",
        "sha256": "6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f",
        "input_resolution": 512,
    },
    "256": {
        "url": "https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt",
        "file": "tapnextpp_ckpt.pt",
        "bytes": 2532282370,
        "md5": "a6388e0b911f2e3644fafa8a9a118dfb",
        "sha256": "cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da",
        "input_resolution": 256,
    },
}
DEFAULT_CHECKPOINT = "512"

#: TAPNext's coordinate space: queries and tracks are in a 256 x 256 frame
#: whatever the input resolution (``TAPNextPP.MODEL_SIZE``).
MODEL_SIZE = 256
#: Support points co-tracked around each query and then dropped. tapnet's
#: VOTSp2026 tracker (tapnet/tapnextpp/votsp2026/tracker.py) uses 64 in a
#: 32 px radius of its 512 px input, i.e. 1/16 of the frame on each axis.
DEFAULT_SUPPORT_POINTS = 0
SUPPORT_RADIUS_FRACTION = 32.0 / 512.0
MAX_SUPPORT_POINTS = 256
#: Radius (model px) of the window ``tracker_certainty`` sums the coordinate
#: distribution over; upstream's colab passes 8.
_CERTAINTY_RADIUS = 8
#: Open online sessions kept on the GPU; the least recently used is dropped
#: when a new one starts (each holds a recurrent state of ~0.15-0.6 GB).
MAX_SESSIONS = 4

_load_lock = threading.Lock()
_run_lock = threading.Lock()
_runtime: Any = None
_sessions: OrderedDict[str, _Session] = OrderedDict()
_session_ids = itertools.count(1)


class PointTrack(TypedDict):
    query_pixel: list[float]  # [u, v] on the query frame, as given
    pixels: list[list[float]]  # [T][u, v] one per frame
    visible: list[bool]  # [T] the model's visibility call per frame
    confidence: list[float]  # [T] p(visible) x localisation certainty, 0..1


class TrackTiming(TypedDict):
    load_s: float  # model load paid by this call (0 once resident)
    track_s: float  # all model steps of this call
    per_frame_ms: float  # track_s / frames stepped
    total_s: float


class TrackResult(TypedDict):
    found: bool  # every query is visible on at least one frame besides its query frame
    tracks: list[PointTrack]  # one per query, in query order
    num_frames: int  # T
    query_frame: int  # the frame the queries are on
    backend: str  # "tapnext++"
    model: str  # checkpoint file name and its pinned digest (or the local path)
    timing: TrackTiming


class SessionFrame(TypedDict):
    session_id: str
    step: int  # frames this session has seen, minus one (0 for the query frame)
    pixels: list[list[float]]  # [Q][u, v] on this frame
    visible: list[bool]  # [Q]
    confidence: list[float]  # [Q] 0..1
    evicted: str  # a session dropped to make room for this one ("" if none)
    timing: TrackTiming


class StopResult(TypedDict):
    stopped: bool  # False when the id was unknown (already stopped or evicted)


# ---------------------------------------------------------------------------
# Geometry (numpy only)
# ---------------------------------------------------------------------------


def _frame(frame: Any, what: str = "frame") -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"track: {what} must be uint8 [H, W, 3] RGB, got dtype={arr.dtype} shape={arr.shape}"
        )
    return np.ascontiguousarray(arr)


def _frames(frames: Any) -> list[np.ndarray]:
    """A ``[T, H, W, 3]`` array or a list of ``[H, W, 3]`` frames, all one size."""
    seq = list(frames) if isinstance(frames, (list, tuple)) else list(np.asarray(frames))
    if not seq:
        raise ValueError("track.points needs at least one frame")
    out = [_frame(f, f"frame {k}") for k, f in enumerate(seq)]
    if any(f.shape != out[0].shape for f in out):
        raise ValueError("track.points: all frames must have the same size")
    return out


def _query_array(query_points: Any, width: int, height: int) -> np.ndarray:
    pts = np.asarray(query_points, dtype=np.float64)
    if pts.ndim == 1 and pts.size == 2:
        pts = pts[None, :]
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        raise ValueError(f"query_points must be [[u, v], ...], got shape {pts.shape}")
    inside = (pts[:, 0] > -0.5) & (pts[:, 0] < width - 0.5)
    inside &= (pts[:, 1] > -0.5) & (pts[:, 1] < height - 0.5)
    if not inside.all():
        raise ValueError(
            f"query_points {pts[~inside].tolist()} lie outside the {width}x{height} frame"
        )
    return pts


def _to_model(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    """OpenCV ``[u, v]`` pixels -> model ``[y, x]`` in the 256 x 256 frame.

    The model's grid has pixel ``i``'s centre at ``i + 0.5`` (``grid_sample``
    with ``align_corners=False``), OpenCV at ``i``: shift by half a pixel
    before scaling (upstream's wrapper scales without the shift, ~0.5 px off
    at 640 px).
    """
    y = (pixels[:, 1] + 0.5) * (MODEL_SIZE / height)
    x = (pixels[:, 0] + 0.5) * (MODEL_SIZE / width)
    return np.stack([y, x], axis=-1)


def _from_model(yx: np.ndarray, width: int, height: int) -> np.ndarray:
    """Model ``[y, x]`` -> OpenCV ``[u, v]`` pixels of the frame as given."""
    u = yx[:, 1] * (width / MODEL_SIZE) - 0.5
    v = yx[:, 0] * (height / MODEL_SIZE) - 0.5
    return np.stack([u, v], axis=-1)


def _support_points(queries: np.ndarray, per_query: int, width: int, height: int) -> np.ndarray:
    """``per_query`` points on a grid around each query, clamped into the frame.

    The layout of tapnet's VOTSp2026 tracker (``local_support_points``): an
    aspect-matched grid filling a box of half-size ``SUPPORT_RADIUS_FRACTION``
    of the frame centred on the query.
    """
    if per_query <= 0:
        return np.zeros((0, 2))
    rx, ry = SUPPORT_RADIUS_FRACTION * width, SUPPORT_RADIUS_FRACTION * height
    cols = max(1, round(float(np.sqrt(per_query * rx / ry))))
    rows = max(1, int(np.ceil(per_query / cols)))
    xs = (np.arange(cols) + 0.5) * (2 * rx / cols) - rx
    ys = (np.arange(rows) + 0.5) * (2 * ry / rows) - ry
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)[:per_query]
    out = (queries[:, None, :] + grid[None, :, :]).reshape(-1, 2)
    out[:, 0] = np.clip(out[:, 0], 0.0, width - 1.0)
    out[:, 1] = np.clip(out[:, 1], 0.0, height - 1.0)
    return out


def _support_count(support_points: int) -> int:
    n = int(support_points)
    if not 0 <= n <= MAX_SUPPORT_POINTS:
        raise ValueError(f"support_points must be 0..{MAX_SUPPORT_POINTS}, got {n}")
    return n


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _round(rows: Any, nd: int) -> list:
    return np.round(np.asarray(rows, dtype=np.float64), nd).tolist()


# ---------------------------------------------------------------------------
# Runtime: the pinned package over the pinned checkpoint
# ---------------------------------------------------------------------------


def _checkpoint_key() -> str:
    key = os.environ.get("GAP_TRACK_CHECKPOINT", DEFAULT_CHECKPOINT)
    if key not in CHECKPOINTS:
        raise ValueError(f"GAP_TRACK_CHECKPOINT must be one of {sorted(CHECKPOINTS)}, got {key!r}")
    return key


def _model_id() -> str:
    override = os.environ.get("GAP_TRACK_CHECKPOINT_PATH")
    if override:
        return override
    spec = CHECKPOINTS[_checkpoint_key()]
    return f"{spec['file']}@sha256:{spec['sha256'][:12]}"


def _cache_path() -> Path:
    root = os.environ.get("GAP_TRACK_CACHE") or Path.home() / ".cache" / "roboskillhub" / "track"
    return Path(root) / CHECKPOINTS[_checkpoint_key()]["file"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_file() -> str:
    """The checkpoint on disk, downloading and checking it the first time."""
    override = os.environ.get("GAP_TRACK_CHECKPOINT_PATH")
    if override:
        return override
    spec = CHECKPOINTS[_checkpoint_key()]
    dest = _cache_path()
    if dest.is_file() and dest.stat().st_size == spec["bytes"]:
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    logger.info("[track] downloading %s (%.2f GB)", spec["url"], spec["bytes"] / 1e9)
    urllib.request.urlretrieve(spec["url"], part)
    got = _sha256(part)
    if got != spec["sha256"]:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{spec['url']} has sha256 {got}, expected {spec['sha256']}")
    part.replace(dest)
    return str(dest)


class _Runtime:
    """The TAPNext model on the device, one frame per call."""

    def __init__(self, ckpt: str, input_resolution: int, device: str) -> None:
        import torch
        import torch.nn.functional as F
        from tapnet.tapnext.tapnext_torch_utils import tracker_certainty
        from tapnet.tapnextpp.votsp2026.model import TAPNextPP

        self.torch = torch
        self.F = F
        self.certainty = tracker_certainty
        self.device = torch.device(device)
        self.resolution = int(input_resolution)
        wrapper = TAPNextPP.from_checkpoint(
            ckpt, device=self.device, input_resolution=self.resolution
        )
        self.model = wrapper._model

    def _video(self, frame: np.ndarray):
        """uint8 RGB ``[H, W, 3]`` -> ``[1, 1, R, R, 3]`` in [-1, 1] (upstream's preprocess_frame)."""
        t = self.torch.from_numpy(frame).to(self.device, non_blocking=True)
        t = t.permute(2, 0, 1).unsqueeze(0).float()
        t = self.F.interpolate(
            t, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        t = t.div_(127.5).sub_(1.0)
        return t.permute(0, 2, 3, 1).unsqueeze(0)

    def _run(self, frame: np.ndarray, queries, state):
        torch = self.torch
        autocast = torch.autocast(
            self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
        )
        with torch.inference_mode(), autocast:
            tracks, track_logits, vis_logits, state = self.model(
                video=self._video(frame), query_points=queries, state=state
            )
            cert = self.certainty(tracks.float(), track_logits.float(), _CERTAINTY_RADIUS)
        yx = tracks[0, 0].float().cpu().numpy().astype(np.float64)
        vis = vis_logits[0, 0, :, 0].float().cpu().numpy().astype(np.float64)
        return yx, vis, cert[0, 0, :, 0].float().cpu().numpy().astype(np.float64), state

    def init(self, frame: np.ndarray, queries_yx: np.ndarray):
        """``(yx [Q, 2] model, visibility logits [Q], certainty [Q], state)`` on the query frame."""
        q = np.zeros((len(queries_yx), 3), dtype=np.float32)
        q[:, 1:] = queries_yx
        queries = self.torch.from_numpy(q).to(self.device).unsqueeze(0)
        return self._run(frame, queries, None)

    def step(self, frame: np.ndarray, state):
        """The same for the next frame of a running track."""
        return self._run(frame, None, state)


def _build_runtime() -> _Runtime:
    device = os.environ.get("GAP_TRACK_DEVICE", "cuda")
    key = _checkpoint_key()
    ckpt = _checkpoint_file()
    logger.info("Loading TAPNext++ (%s) on %s from %s", _model_id(), device, ckpt)
    return _Runtime(ckpt, CHECKPOINTS[key]["input_resolution"], device)


def _get_runtime() -> tuple[Any, float]:
    """The resident runtime and the seconds this call spent loading it."""
    global _runtime
    with _load_lock:
        if _runtime is None:
            t0 = time.perf_counter()
            _runtime = _build_runtime()
            return _runtime, time.perf_counter() - t0
        return _runtime, 0.0


def weights_cached() -> bool | None:
    """Filesystem-only probe for ``gap check``: the pinned checkpoint is on disk."""
    override = os.environ.get("GAP_TRACK_CHECKPOINT_PATH")
    if override:
        return os.path.isfile(override)
    path = _cache_path()
    return path.is_file() and path.stat().st_size == CHECKPOINTS[_checkpoint_key()]["bytes"]


def prefetch() -> None:
    """Download the pinned checkpoint (``gap skills check --download``)."""
    logger.info("[track] prefetching %s ...", _model_id())
    _checkpoint_file()
    logger.info("[track] prefetch complete")


def _runtime_or_raise(name: str) -> tuple[Any, float]:
    try:
        return _get_runtime()
    except Exception as exc:
        raise ToolError(name, f"TAPNext++ failed to load: {exc}") from exc


def _rollout(runtime: Any, frames: list[np.ndarray], queries_yx: np.ndarray) -> tuple:
    """Track from ``frames[0]``: ``(yx [T, Q, 2], vis logits [T, Q], certainty [T, Q])``."""
    yx, vis, cert, state = runtime.init(frames[0], queries_yx)
    out_yx, out_vis, out_cert = [yx], [vis], [cert]
    for frame in frames[1:]:
        yx, vis, cert, state = runtime.step(frame, state)
        out_yx.append(yx)
        out_vis.append(vis)
        out_cert.append(cert)
    return np.stack(out_yx), np.stack(out_vis), np.stack(out_cert)


def _visible(vis_logits: np.ndarray, confidence: np.ndarray, min_confidence: float | None):
    if min_confidence is None:
        return vis_logits > 0.0
    return confidence >= float(min_confidence)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="track.points",
    summary=(
        "Track query pixels through a clip (TAPNext++): per-frame pixel position and "
        "visibility, through occlusions and back."
    ),
    tags=("perception", "tracking"),
)
def points(
    frames: list[np.ndarray] | np.ndarray,
    query_points: list[list[float]] | np.ndarray,
    query_frame: int = 0,
    support_points: int = DEFAULT_SUPPORT_POINTS,
    min_confidence: float | None = None,
) -> TrackResult:
    """Follow each query pixel through ``frames``.

    Args:
        frames: The clip, earliest first: a list of uint8 ``[H, W, 3]`` RGB
            frames or one ``[T, H, W, 3]`` array, all the same size.
        query_points: ``[[u, v], ...]`` pixels on ``frames[query_frame]``,
            e.g. ``gemini-er.point``'s answer or ``curve.endpoints``' tip pixel.
        query_frame: Index of the frame the queries are on. Frames after it
            are tracked forwards, frames before it backwards (the clip
            reversed), each from the queries.
        support_points: Extra points co-tracked on a grid around each query
            and then dropped (tapnet's VOTSp2026 setting is 64); 0 tracks the
            queries alone.
        min_confidence: When set, ``visible`` is ``confidence >= min_confidence``
            instead of the model's visibility logit > 0.

    Returns:
        ``TrackResult``: one track per query with ``pixels`` ``[T][u, v]``,
        ``visible`` and ``confidence`` per frame. A point the model calls
        occluded still has a position: its estimate of where the point is
        behind the occluder. ``found`` is False when some query is never seen
        again after its query frame.
    """
    t_start = time.perf_counter()
    seq = _frames(frames)
    height, width = seq[0].shape[:2]
    q0 = int(query_frame)
    if not 0 <= q0 < len(seq):
        raise ValueError(f"query_frame must be 0..{len(seq) - 1}, got {query_frame}")
    queries = _query_array(query_points, width, height)
    n_support = _support_count(support_points)
    all_px = np.concatenate([queries, _support_points(queries, n_support, width, height)])
    queries_yx = _to_model(all_px, width, height)
    n = len(queries)

    runtime, load_s = _runtime_or_raise("track.points")
    t_track = time.perf_counter()
    stepped = 0
    try:
        with _run_lock:
            yx, vis, cert = _rollout(runtime, seq[q0:], queries_yx)
            stepped += len(seq) - q0
            if q0 > 0:
                b_yx, b_vis, b_cert = _rollout(runtime, seq[q0::-1], queries_yx)
                stepped += q0 + 1
                yx = np.concatenate([b_yx[:0:-1], yx])
                vis = np.concatenate([b_vis[:0:-1], vis])
                cert = np.concatenate([b_cert[:0:-1], cert])
    except Exception as exc:
        raise ToolError("track.points", f"TAPNext++ failed: {exc}") from exc
    track_s = time.perf_counter() - t_track

    confidence = _sigmoid(vis) * np.clip(cert, 0.0, 1.0)
    visible = _visible(vis, confidence, min_confidence)
    tracks: list[PointTrack] = []
    found = True
    for j in range(n):
        px = _from_model(yx[:, j, :], width, height)
        later = np.delete(visible[:, j], q0)
        found = found and (bool(later.any()) if len(later) else bool(visible[q0, j]))
        tracks.append(
            {
                "query_pixel": [float(queries[j, 0]), float(queries[j, 1])],
                "pixels": _round(px, 2),
                "visible": [bool(b) for b in visible[:, j]],
                "confidence": _round(confidence[:, j], 3),
            }
        )
    return {
        "found": found,
        "tracks": tracks,
        "num_frames": len(seq),
        "query_frame": q0,
        "backend": "tapnext++",
        "model": _model_id(),
        "timing": {
            "load_s": round(load_s, 3),
            "track_s": round(track_s, 3),
            "per_frame_ms": round(1000.0 * track_s / max(stepped, 1), 2),
            "total_s": round(time.perf_counter() - t_start, 3),
        },
    }


class _Session:
    def __init__(self, state: Any, width: int, height: int, n: int) -> None:
        self.state = state
        self.width = width
        self.height = height
        self.n = n
        self.step = 0


def _session_frame(
    session_id: str,
    session: _Session,
    yx: np.ndarray,
    vis: np.ndarray,
    cert: np.ndarray,
    min_confidence: float | None,
    evicted: str,
    load_s: float,
    track_s: float,
    t_start: float,
) -> SessionFrame:
    n = session.n
    confidence = _sigmoid(vis[:n]) * np.clip(cert[:n], 0.0, 1.0)
    return {
        "session_id": session_id,
        "step": session.step,
        "pixels": _round(_from_model(yx[:n], session.width, session.height), 2),
        "visible": [bool(b) for b in _visible(vis[:n], confidence, min_confidence)],
        "confidence": _round(confidence, 3),
        "evicted": evicted,
        "timing": {
            "load_s": round(load_s, 3),
            "track_s": round(track_s, 3),
            "per_frame_ms": round(1000.0 * track_s, 2),
            "total_s": round(time.perf_counter() - t_start, 3),
        },
    }


@tool(
    name="track.start",
    summary=(
        "Open an online tracking session on a live frame and query pixels (TAPNext++); "
        "feed later frames to track.update."
    ),
    tags=("perception", "tracking"),
)
def start(
    frame: np.ndarray,
    query_points: list[list[float]] | np.ndarray,
    support_points: int = DEFAULT_SUPPORT_POINTS,
    min_confidence: float | None = None,
) -> SessionFrame:
    """Start following ``query_points`` from ``frame``; returns a ``session_id``.

    Args:
        frame: The current uint8 ``[H, W, 3]`` RGB frame; later frames of the
            session must have the same size (the same camera).
        query_points: ``[[u, v], ...]`` pixels on ``frame``.
        support_points: As in ``track.points``.
        min_confidence: As in ``track.points``, for this first frame.

    Returns:
        ``SessionFrame`` for the query frame (``step`` 0). At most
        ``MAX_SESSIONS`` sessions stay open; starting one more drops the least
        recently used, named in ``evicted``.
    """
    t_start = time.perf_counter()
    img = _frame(frame)
    height, width = img.shape[:2]
    queries = _query_array(query_points, width, height)
    n_support = _support_count(support_points)
    all_px = np.concatenate([queries, _support_points(queries, n_support, width, height)])
    runtime, load_s = _runtime_or_raise("track.start")
    t_track = time.perf_counter()
    try:
        with _run_lock:
            yx, vis, cert, state = runtime.init(img, _to_model(all_px, width, height))
    except Exception as exc:
        raise ToolError("track.start", f"TAPNext++ failed: {exc}") from exc
    track_s = time.perf_counter() - t_track
    evicted = ""
    with _load_lock:
        while len(_sessions) >= MAX_SESSIONS:
            evicted, _ = _sessions.popitem(last=False)
            logger.info("[track] dropped session %s to open a new one", evicted)
        session_id = f"track-{next(_session_ids)}"
        session = _Session(state, width, height, len(queries))
        _sessions[session_id] = session
    return _session_frame(
        session_id, session, yx, vis, cert, min_confidence, evicted, load_s, track_s, t_start
    )


@tool(
    name="track.update",
    summary="Advance an online tracking session by one frame: each query's pixel and visibility.",
    tags=("perception", "tracking"),
)
def update(
    session_id: str,
    frame: np.ndarray,
    min_confidence: float | None = None,
) -> SessionFrame:
    """Track the session's queries into ``frame`` (the next frame of the same camera).

    Args:
        session_id: From ``track.start``.
        frame: uint8 ``[H, W, 3]`` RGB, the size of the session's first frame.
        min_confidence: As in ``track.points``.

    Returns:
        ``SessionFrame`` with ``step`` counting frames since the query frame.
    """
    t_start = time.perf_counter()
    with _load_lock:
        session = _sessions.get(session_id)
        if session is not None:
            _sessions.move_to_end(session_id)
    if session is None:
        raise ToolError("track.update", f"unknown session {session_id!r} (stopped or evicted)")
    img = _frame(frame)
    if img.shape[:2] != (session.height, session.width):
        raise ValueError(
            f"track.update: frame is {img.shape[1]}x{img.shape[0]}, the session's "
            f"{session.width}x{session.height}"
        )
    runtime, load_s = _runtime_or_raise("track.update")
    t_track = time.perf_counter()
    try:
        with _run_lock:
            yx, vis, cert, session.state = runtime.step(img, session.state)
    except Exception as exc:
        raise ToolError("track.update", f"TAPNext++ failed: {exc}") from exc
    session.step += 1
    return _session_frame(
        session_id,
        session,
        yx,
        vis,
        cert,
        min_confidence,
        "",
        load_s,
        time.perf_counter() - t_track,
        t_start,
    )


@tool(
    name="track.stop",
    summary="Close an online tracking session and free its GPU state. Unknown ids are not an error.",
    tags=("perception", "tracking"),
)
def stop(session_id: str) -> StopResult:
    """Drop the session's recurrent state."""
    with _load_lock:
        return {"stopped": _sessions.pop(session_id, None) is not None}
