"""molmo-motion tool bundle: language-conditioned forecasts of where query points move.

Runs AllenAI's MolmoMotion-4B (https://huggingface.co/allenai/MolmoMotion-4B-H3-F30,
Apache-2.0) through the ``molmo_motion`` package of
https://github.com/allenai/molmo-motion (Apache-2.0), both at pinned commits.
Nothing of either is copied here: the package is a dependency of this bundle's
environment, the weights come from the Hugging Face Hub on first use.

The model forecasts the 3D positions, in the camera frame of the last
observed frame (t0), of exactly 8 query points over F future steps at 15 fps,
from up to H RGB frames, the points' pixels at t0, their 3D history and a
short action caption. This module turns a hub-shaped request (an RGB frame or
short history, query pixels, optional depth + intrinsics + camera-to-world
pose, an instruction) into that input -- padding the queries to 8, lifting them
to 3D, repeating frames up to H -- and returns each query's forecast as pixels
and, when depth was given, as camera and world points. The result shape is
backend-neutral (``backend`` names the model) so another forecaster, e.g. a
future ``vflash.forecast``, can return the same structure.

The model loads lazily on first call (module-level singleton); importing this
module never imports torch or molmo_motion. Knobs via env:

- ``GAP_MOLMO_MOTION_DEVICE`` -- torch device (default ``cuda``).
- ``GAP_MOLMO_MOTION_REPO`` -- ``allenai/MolmoMotion-4B-H3-F30`` (default) or
  ``allenai/MolmoMotion-4B-H1-F32``, each at its pinned commit.
- ``GAP_MOLMO_MOTION_MODEL`` -- a local checkpoint directory (``config.yaml`` +
  ``model.pt``) instead of the pinned Hub snapshot.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import tool
from gap_core.types import Se3Pose, pose_to_matrix

logger = logging.getLogger(__name__)

#: Hub commits of the two released checkpoints, checked live on 2026-11-07.
REVISIONS = {
    "allenai/MolmoMotion-4B-H3-F30": "3f5e790a511ff2cdf21c8d2a14cb4d8409c94629",
    "allenai/MolmoMotion-4B-H1-F32": "14e69b0d5dd55b2f09c885030b8756cd053ea6a4",
}
DEFAULT_REPO = "allenai/MolmoMotion-4B-H3-F30"
#: The native checkpoint the package loads; the safetensors shards in the same
#: repo are an HF-format copy it does not read, so they are not downloaded.
_CHECKPOINT_FILES = ["config.yaml", "model.pt"]

#: The model's fixed query count (``num_points`` in both checkpoints' config).
NUM_POINTS = 8
#: Forecast step: the model card's "~2 s at 15 fps" for F=30.
DT_S = 1.0 / 15.0
#: The released checkpoints' trained horizons (H3-F30: 30, H1-F32: 32).
DEFAULT_HORIZON = 30
MAX_HORIZON = 32
#: Padding points sit on a ring this far around the queries: the model was
#: trained on 8 points sampled on one moving object, so the padding stays on it.
_PAD_RADIUS_PX = 4.0
#: Half-size of the window a query's depth is the median of (invalid pixels skipped).
_DEPTH_WINDOW_PX = 3
#: Depth along the ray when none is given; only the 2D forecast is returned then.
_ASSUMED_DEPTH_M = 1.0

_load_lock = threading.Lock()
_runtime: Any = None


class ForecastTrack(TypedDict):
    query_pixel: list[float]  # [u, v] at t0, as given
    pixels: list[list[float]]  # [F][u, v], the forecast steps t0 + k * dt_s, k = 1..F
    start_camera: list[float]  # [x, y, z] m at t0, camera frame; [] without depth
    points_camera: list[list[float]]  # [F][x, y, z] m, camera frame at t0; [] without depth
    start_world: list[float]  # [x, y, z] m at t0; [] without depth + camera_pose
    points_world: list[list[float]]  # [F][x, y, z] m, world; [] without depth + camera_pose
    predicted: list[bool]  # [F] the model emitted this step (else held from the last one)


class ForecastTiming(TypedDict):
    load_s: float  # model load paid by this call (0 once resident)
    prepare_s: float  # lifting queries to 3D, tokenizing, image preprocessing
    generate_s: float  # the model's autoregressive decode
    total_s: float


class ForecastResult(TypedDict):
    found: bool  # at least one step of one track came back
    tracks: list[ForecastTrack]  # one per query, in query order
    horizon: int  # F
    dt_s: float  # seconds between forecast steps
    frame: str  # "world" | "camera" | "pixel": the richest frame the tracks carry
    backend: str  # "molmo-motion"
    model: str  # repo@revision (or the local directory)
    timing: ForecastTiming
    raw: str  # the model's answer text


# ---------------------------------------------------------------------------
# Geometry (numpy only)
# ---------------------------------------------------------------------------


def _frames(image: Any) -> list[np.ndarray]:
    """One ``uint8 [H, W, 3]`` frame or a list of them (earliest first)."""
    seq = list(image) if isinstance(image, (list, tuple)) else [image]
    if not seq:
        raise ValueError("molmo-motion.forecast needs at least one frame")
    out = []
    for frame in seq:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"molmo-motion.forecast expects uint8 [H, W, 3] RGB frames, got "
                f"dtype={arr.dtype} shape={arr.shape}"
            )
        out.append(np.ascontiguousarray(arr))
    if any(f.shape != out[-1].shape for f in out):
        raise ValueError("molmo-motion.forecast: all frames must have the same size")
    return out


def _fit_history(items: list, size: int) -> list:
    """The last ``size`` items, the earliest repeated in front when there are fewer."""
    items = list(items)[-size:]
    return [items[0]] * (size - len(items)) + items


def _query_array(query_points: Any) -> np.ndarray:
    pts = np.asarray(query_points, dtype=np.float64)
    if pts.ndim == 1 and pts.size == 2:
        pts = pts[None, :]
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        raise ValueError(f"query_points must be [[u, v], ...], got shape {pts.shape}")
    return pts


def _pad_queries(queries: np.ndarray, count: int = NUM_POINTS) -> np.ndarray:
    """``count`` pixels: the queries first, then a ring of points around them in turn."""
    n = len(queries)
    if n > count:
        raise ValueError(f"at most {count} queries per model call, got {n}")
    extra = count - n
    out = [q for q in queries]
    for k in range(extra):
        angle = 2.0 * np.pi * k / max(extra, 1)
        base = queries[k % n]
        out.append(base + _PAD_RADIUS_PX * np.array([np.cos(angle), np.sin(angle)]))
    return np.asarray(out, dtype=np.float64)


def _clip_pixels(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    out = pixels.copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, width - 1.0)
    out[:, 1] = np.clip(out[:, 1], 0.0, height - 1.0)
    return out


def _depth_at(depth: np.ndarray, u: float, v: float) -> float:
    """Median of the valid depths in a small window around (u, v); NaN if none."""
    h, w = depth.shape
    cu, cv = int(round(u)), int(round(v))
    r = _DEPTH_WINDOW_PX
    patch = depth[max(0, cv - r) : min(h, cv + r + 1), max(0, cu - r) : min(w, cu + r + 1)]
    good = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(good)) if good.size else float("nan")


def _deproject(pixels: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    x = (pixels[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (pixels[:, 1] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=-1)


def _project(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = np.clip(xyz[..., 2], 1e-6, None)
    u = K[0, 0] * xyz[..., 0] / z + K[0, 2]
    v = K[1, 1] * xyz[..., 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1)


def _to_world(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    return xyz @ T[:3, :3].T + T[:3, 3]


def _camera_matrix(camera_pose: Se3Pose | np.ndarray) -> np.ndarray:
    """A camera-to-world 4x4 from either an ``Se3Pose`` dict or a 4x4 array."""
    if isinstance(camera_pose, dict):
        return pose_to_matrix(camera_pose)
    return np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)


def _default_intrinsics(width: int, height: int) -> np.ndarray:
    """A pinhole guess (focal = the longer side, ~53 deg across it) when none is given."""
    f = float(max(width, height))
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def _hold_missing(track: np.ndarray, predicted: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Steps the model did not emit take the last emitted value (the start before any)."""
    out = track.copy()
    last = start
    for k in range(len(out)):
        if predicted[k]:
            last = out[k]
        else:
            out[k] = last
    return out


def _round(rows: np.ndarray, nd: int) -> list:
    return np.round(np.asarray(rows, dtype=np.float64), nd).tolist()


# ---------------------------------------------------------------------------
# Runtime: the pinned package over the pinned checkpoint
# ---------------------------------------------------------------------------


def _repo() -> str:
    repo = os.environ.get("GAP_MOLMO_MOTION_REPO", DEFAULT_REPO)
    if repo not in REVISIONS:
        raise ValueError(f"GAP_MOLMO_MOTION_REPO must be one of {sorted(REVISIONS)}, got {repo!r}")
    return repo


def _model_id() -> str:
    override = os.environ.get("GAP_MOLMO_MOTION_MODEL")
    if override:
        return override
    repo = _repo()
    return f"{repo}@{REVISIONS[repo]}"


def _checkpoint_dir() -> str:
    override = os.environ.get("GAP_MOLMO_MOTION_MODEL")
    if override:
        return override
    from huggingface_hub import snapshot_download

    repo = _repo()
    return snapshot_download(
        repo_id=repo, revision=REVISIONS[repo], allow_patterns=_CHECKPOINT_FILES
    )


class _Runtime:
    """The processor and the bf16 model, built once and kept resident."""

    def __init__(self, ckpt_dir: str, device: str) -> None:
        import torch
        from molmo_motion import MolmoMotion, MolmoMotionProcessor
        from molmo_motion.models.molmo2.molmo2_trajectory import Molmo2TrajectoryConfig
        from molmo_motion.public_config import MolmoMotionConfig

        self.torch = torch
        self.device = torch.device(device)
        self.processor = MolmoMotionProcessor.from_pretrained(ckpt_dir)
        cfg = Molmo2TrajectoryConfig.load(
            os.path.join(ckpt_dir, "config.yaml"), key="model", validate_paths=False
        )
        public = MolmoMotionConfig(
            num_points=self.processor.config.num_points,
            history_size=self.processor.config.history_size,
            future_size=self.processor.config.future_size,
            max_sequence_length=self.processor.config.max_sequence_length,
        )
        # Upstream's from_pretrained builds the model in fp32 on the CPU and
        # torch.loads the 17.8 GB model.pt into RAM next to it before casting
        # to bf16 (~36 GB of host memory). Building in bf16 on the device and
        # mmap-loading the state dict gives the same bf16 weights (the model
        # registers no buffers; RoPE tables are computed in fp32 on use) with
        # ~9 GB on the GPU and the checkpoint paged in from disk.
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with self.device:
                model = MolmoMotion(public, internal_config=cfg)
        finally:
            torch.set_default_dtype(previous)
        state = torch.load(
            os.path.join(ckpt_dir, "model.pt"), map_location="cpu", mmap=True, weights_only=True
        )
        missing, unexpected = model._internal.load_state_dict(state, strict=False)
        del state
        if missing:
            raise RuntimeError(f"checkpoint lacks {len(missing)} weights, e.g. {missing[:3]}")
        if unexpected:
            logger.warning("molmo-motion: %d unused checkpoint keys", len(unexpected))
        model._internal.eval()
        self.model = model
        self.history_size = int(public.history_size)
        self.num_points = int(public.num_points)

    def predict(
        self,
        frames: list[np.ndarray],
        points_2d: np.ndarray,
        points_3d_history: np.ndarray,
        action: str,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, str, float]:
        """``(future_3d [P, F, 3] camera m, predicted [P, F], answer text, prepare_s)``."""
        from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array
        from PIL import Image

        torch = self.torch
        t0 = time.perf_counter()
        inputs = self.processor(
            history_frames=[Image.fromarray(f, "RGB") for f in frames],
            points_2d_at_t0=torch.as_tensor(points_2d, dtype=torch.float32),
            points_3d_history=torch.as_tensor(points_3d_history, dtype=torch.float32),
            action=action,
            future_horizon=horizon,
        )
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        prepare_s = time.perf_counter() - t0
        autocast = torch.autocast(self.device.type, dtype=torch.bfloat16)
        with torch.inference_mode(), autocast:
            out = self.model.predict_trajectory(**inputs)
        future = out.future_3d.float().cpu().numpy()
        parsed = parse_tracks_text(out.future_text)
        if parsed is None:
            predicted = np.zeros(future.shape[:2], dtype=bool)
        else:
            _, predicted = tracks_to_array(
                parsed,
                num_points=self.num_points,
                num_frames=horizon,
                start_timestamp=float(self.history_size),
            )
        return future, np.asarray(predicted, dtype=bool), out.future_text, prepare_s


def _build_runtime() -> _Runtime:
    device = os.environ.get("GAP_MOLMO_MOTION_DEVICE", "cuda")
    ckpt = _checkpoint_dir()
    logger.info("Loading MolmoMotion (%s) on %s from %s", _model_id(), device, ckpt)
    return _Runtime(ckpt, device)


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
    override = os.environ.get("GAP_MOLMO_MOTION_MODEL")
    if override:
        return all(os.path.isfile(os.path.join(override, f)) for f in _CHECKPOINT_FILES)
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    repo = _repo()
    try:
        hits = [
            try_to_load_from_cache(repo, f, revision=REVISIONS[repo]) for f in _CHECKPOINT_FILES
        ]
    except Exception:
        return None
    return all(isinstance(h, str) for h in hits)


def prefetch() -> None:
    """Download the pinned checkpoint (``gap skills check --download``)."""
    logger.info("[molmo-motion] prefetching %s ...", _model_id())
    _checkpoint_dir()
    logger.info("[molmo-motion] prefetch complete")


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(
    name="molmo-motion.forecast",
    summary=(
        "Forecast where query pixels move over the next ~2 s under a language instruction "
        "(MolmoMotion-4B); pixels per step, plus camera/world points when depth is given."
    ),
    tags=("perception", "planning"),
)
def forecast(
    image: np.ndarray | list[np.ndarray],
    query_points: list[list[float]] | np.ndarray,
    instruction: str,
    depth: np.ndarray | list[np.ndarray] | None = None,
    intrinsics: np.ndarray | None = None,
    camera_pose: Se3Pose | np.ndarray | None = None,
    query_history: list[list[list[float]]] | np.ndarray | None = None,
    horizon: int = DEFAULT_HORIZON,
) -> ForecastResult:
    """Forecast each query pixel's next ``horizon`` positions at 15 fps.

    Args:
        image: The current RGB frame ``uint8 [H, W, 3]``, or a short history
            of frames, earliest first, the last one current (t0). The H3
            checkpoint reads 3 frames: fewer are padded by repeating the
            earliest (a still history), more are cut to the last 3.
        query_points: ``[[u, v], ...]`` pixels in the current frame, e.g.
            ``gemini-er.point``'s answer or ``curve.endpoints``' tip pixel.
            Up to 8 per model call; more are forecast 8 at a time.
        instruction: The action the forecast is conditioned on, as a short
            caption ("the cable tip is pulled through the ring").
        depth: ``float32 [H, W]`` metres for the current frame, or one per
            history frame. With ``intrinsics`` it lifts the queries to 3D and
            the result carries camera points (and world points with
            ``camera_pose``). Without it the queries sit at an assumed 1 m
            along their rays and only pixels come back.
        intrinsics: ``[3, 3]`` pinhole K. Required with ``depth``; without
            both a focal of the image's longer side is assumed.
        camera_pose: Camera-to-world ``Se3Pose`` or 4x4, for world points.
        query_history: ``[H][N][2]`` pixels of the queries in each history
            frame (e.g. from a tracker), earliest first, last = ``query_points``.
            Default: the queries did not move.
        horizon: Forecast steps F (1-32; the checkpoint was trained for 30).

    Returns:
        ``ForecastResult``: one track per query with ``pixels`` ``[F][u, v]``,
        ``points_camera`` / ``points_world`` ``[F][x, y, z]`` (empty when not
        computable), ``predicted`` per step, ``dt_s``, ``frame`` and timings.
        ``found=False`` when the model emitted no parseable step.
    """
    t_start = time.perf_counter()
    if not instruction or not str(instruction).strip():
        raise ValueError("molmo-motion.forecast needs an instruction")
    horizon = int(horizon)
    if not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be 1..{MAX_HORIZON}, got {horizon}")
    frames = _frames(image)
    height, width = frames[-1].shape[:2]
    queries = _query_array(query_points)

    depths: list[np.ndarray] | None = None
    if depth is not None:
        seq = list(depth) if isinstance(depth, (list, tuple)) else [depth]
        depths = [np.asarray(d, dtype=np.float32) for d in seq]
        if any(d.shape != (height, width) for d in depths):
            raise ValueError(f"depth must be [{height}, {width}] like the frame")
        if intrinsics is None:
            raise ValueError("molmo-motion.forecast: depth needs intrinsics")
    K = (
        np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        if intrinsics is not None
        else _default_intrinsics(width, height)
    )
    T = _camera_matrix(camera_pose) if camera_pose is not None and depths is not None else None

    history_px = None
    if query_history is not None:
        history_px = np.asarray(query_history, dtype=np.float64)
        if history_px.ndim != 3 or history_px.shape[1:] != queries.shape:
            raise ValueError(
                f"query_history must be [H][{len(queries)}][2], got shape {history_px.shape}"
            )

    try:
        runtime, load_s = _get_runtime()
    except Exception as exc:
        raise ToolError("molmo-motion.forecast", f"MolmoMotion failed to load: {exc}") from exc
    H = runtime.history_size
    P = runtime.num_points
    frames_in = _fit_history(frames, H)
    depth_hist = _fit_history(depths, H) if depths is not None else None
    px_hist = (
        _fit_history(list(history_px), H) if history_px is not None else [queries] * H
    )  # [H] x [N, 2]

    tracks: list[ForecastTrack] = []
    raw: list[str] = []
    prepare_s = generate_s = 0.0
    any_found = False
    for lo in range(0, len(queries), P):
        idx = slice(lo, lo + P)
        n = len(queries[idx])
        padded_hist = [_clip_pixels(_pad_queries(h[idx], P), width, height) for h in px_hist]
        z_hist = []
        for k, pix in enumerate(padded_hist):
            if depth_hist is None:
                z = np.full(P, _ASSUMED_DEPTH_M)
            else:
                z = np.array([_depth_at(depth_hist[k], u, v) for u, v in pix])
                if not np.all(np.isfinite(z[:n])):
                    bad = [
                        tuple(q)
                        for q, ok in zip(pix[:n], np.isfinite(z[:n]), strict=True)
                        if not ok
                    ]
                    raise ToolError(
                        "molmo-motion.forecast", f"no valid depth near query pixel(s) {bad}"
                    )
                # A padding point off the depth map takes its query's depth.
                fill = np.where(np.isfinite(z), z, np.resize(z[:n], P))
                z = fill
            z_hist.append(z)
        points_3d = np.stack(
            [_deproject(p, z, K) for p, z in zip(padded_hist, z_hist, strict=True)]
        )
        t_gen = time.perf_counter()
        try:
            future, predicted, text, prep = runtime.predict(
                frames_in, padded_hist[-1], points_3d, str(instruction).strip(), horizon
            )
        except Exception as exc:
            raise ToolError("molmo-motion.forecast", f"MolmoMotion failed: {exc}") from exc
        prepare_s += prep
        generate_s += time.perf_counter() - t_gen - prep
        raw.append(text)
        start = points_3d[-1]
        for j in range(n):
            pred_j = np.asarray(predicted[j], dtype=bool)
            any_found = any_found or bool(pred_j.any())
            cam = _hold_missing(np.asarray(future[j], dtype=np.float64), pred_j, start[j])
            pix = _project(cam, K)
            track: ForecastTrack = {
                "query_pixel": [float(queries[lo + j, 0]), float(queries[lo + j, 1])],
                "pixels": _round(pix, 2),
                "start_camera": [],
                "points_camera": [],
                "start_world": [],
                "points_world": [],
                "predicted": [bool(b) for b in pred_j],
            }
            if depth_hist is not None:
                track["start_camera"] = _round(start[j], 4)
                track["points_camera"] = _round(cam, 4)
                if T is not None:
                    track["start_world"] = _round(_to_world(start[j][None], T)[0], 4)
                    track["points_world"] = _round(_to_world(cam, T), 4)
            tracks.append(track)

    frame = "pixel" if depth_hist is None else ("world" if T is not None else "camera")
    return {
        "found": any_found,
        "tracks": tracks,
        "horizon": horizon,
        "dt_s": DT_S,
        "frame": frame,
        "backend": "molmo-motion",
        "model": _model_id(),
        "timing": {
            "load_s": round(load_s, 3),
            "prepare_s": round(prepare_s, 3),
            "generate_s": round(generate_s, 3),
            "total_s": round(time.perf_counter() - t_start, 3),
        },
        "raw": "\n".join(raw),
    }
