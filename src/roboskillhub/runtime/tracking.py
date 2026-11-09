"""Image-coordinate point tracking, independent of robot and simulator backends.

LK is an explicit CPU baseline, not a substitute named TAPIR. TAPIR's tensor
contract follows google-deepmind/tapnet at
ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b: tapnet/torch/tapir_model.py and
colabs/torch_tapir_demo.ipynb. No upstream source or weights are bundled here.
Neither tracker infers metric 3-D positions, contact, or task success.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

TAPNET_REVISION = "ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b"


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Point tracking requires numpy; install the runtime extra.") from exc
    return np


def _inputs(frames: Any, query_points: Any):
    np = _numpy()
    try:
        video = np.asarray(frames)
        points = np.asarray(query_points, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("frames and query_points must be rectangular numeric arrays") from exc
    if video.ndim != 4 or video.shape[-1] != 3 or video.shape[0] < 2:
        raise ValueError("frames must contain at least two RGB frames with shape [T,H,W,3]")
    if min(video.shape[1:3]) < 2:
        raise ValueError("image height and width must both be at least two pixels")
    if video.dtype.kind not in "ui" or video.min() < 0 or video.max() > 255:
        raise ValueError("frames must be uint8 RGB or nested integer lists in [0,255]")
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("query_points must be a nonempty [N,2] array of [x,y] pixels")
    height, width = video.shape[1:3]
    if not np.isfinite(points).all() or not _inside(points, width, height).all():
        raise ValueError("query_points must be finite and inside the first image")
    return np.ascontiguousarray(video, dtype=np.uint8), np.ascontiguousarray(points)


def _inside(points: Any, width: int, height: int):
    np = _numpy()
    return (
        np.isfinite(points).all(axis=-1)
        & (points[..., 0] >= 0)
        & (points[..., 0] < width)
        & (points[..., 1] >= 0)
        & (points[..., 1] < height)
    )


def _result(tracks: Any, visible: Any, backend: str, height: int, width: int, **extra):
    np = _numpy()
    valid = visible & np.isfinite(tracks).all(axis=-1)
    # JSON cannot carry NaN. Coordinates for rejected points are deliberately null.
    positions = tracks.astype(object)
    positions[~valid] = None
    status = "tracked" if valid.all() else "partial" if valid[-1].any() else "lost"
    return {
        "tracks": positions.tolist(),
        "visible": valid.tolist(),
        "backend": backend,
        "coordinate_frame": "image_pixels",
        "coordinate_order": "xy",
        "image_size_hw": [height, width],
        "query_frame": 0,
        "status": status,
        "notes": "2-D tracking confidence only; not metric motion, grasp or task verification.",
        **extra,
    }


class LKPointTracker:
    """Pyramidal Lucas-Kanade tracking with forward/backward and photometric gates.

    Tracks start at the supplied first-frame pixels. Once lost they stay lost;
    this implementation does not silently reinitialize a point onto another object.
    Use short RGB clips with small inter-frame motion and textured query regions.
    """

    def __init__(
        self,
        *,
        max_fb_error_px: float = 1.5,
        max_lk_error: float = 20.0,
        window_size: int = 21,
        pyramid_levels: int = 3,
    ):
        np = _numpy()
        if not np.isfinite(max_fb_error_px) or max_fb_error_px <= 0:
            raise ValueError("max_fb_error_px must be finite and positive")
        if not np.isfinite(max_lk_error) or max_lk_error <= 0:
            raise ValueError("max_lk_error must be finite and positive")
        if not isinstance(window_size, int) or window_size < 3 or window_size % 2 != 1:
            raise ValueError("window_size must be an odd integer >= 3")
        if not isinstance(pyramid_levels, int) or not 0 <= pyramid_levels <= 8:
            raise ValueError("pyramid_levels must be an integer in [0,8]")
        self.max_fb_error_px = float(max_fb_error_px)
        self.max_lk_error = float(max_lk_error)
        self.window_size = window_size
        self.pyramid_levels = pyramid_levels

    def track(self, frames: Any, query_points: Any) -> dict:
        np = _numpy()
        video, points = _inputs(frames, query_points)
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("LK tracking requires opencv-python-headless.") from exc
        count, height, width, _ = video.shape
        tracks = np.full((count, len(points), 2), np.nan, dtype=np.float32)
        visible = np.zeros((count, len(points)), dtype=bool)
        fb_error = np.full((count, len(points)), np.nan, dtype=np.float32)
        tracks[0], visible[0], fb_error[0] = points, True, 0
        previous = cv2.cvtColor(video[0], cv2.COLOR_RGB2GRAY)
        options = {
            "winSize": (self.window_size, self.window_size),
            "maxLevel": self.pyramid_levels,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }
        for frame_index in range(1, count):
            active = np.flatnonzero(visible[frame_index - 1])
            if not len(active):
                break
            current = cv2.cvtColor(video[frame_index], cv2.COLOR_RGB2GRAY)
            seeds = tracks[frame_index - 1, active].reshape(-1, 1, 2)
            forward, forward_ok, photometric = cv2.calcOpticalFlowPyrLK(
                previous, current, seeds, None, **options
            )
            if forward is None or forward_ok is None or photometric is None:
                break
            candidate = forward.reshape(-1, 2)
            forward_valid = (
                forward_ok.reshape(-1).astype(bool)
                & _inside(candidate, width, height)
                & np.isfinite(photometric.reshape(-1))
                & (photometric.reshape(-1) <= self.max_lk_error)
            )
            indices = np.flatnonzero(forward_valid)
            if len(indices):
                backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
                    current, previous, forward[indices], None, **options
                )
                if backward is not None and backward_ok is not None:
                    error = np.linalg.norm(backward.reshape(-1, 2) - seeds[indices, 0], axis=1)
                    accepted = (
                        backward_ok.reshape(-1).astype(bool)
                        & np.isfinite(error)
                        & (error <= self.max_fb_error_px)
                    )
                    matched = active[indices[accepted]]
                    tracks[frame_index, matched] = candidate[indices[accepted]]
                    visible[frame_index, matched] = True
                    fb_error[frame_index, active[indices]] = error
            previous = current
        error_values = fb_error.astype(object)
        error_values[~np.isfinite(fb_error)] = None
        return _result(
            tracks,
            visible,
            "opencv-lk",
            height,
            width,
            forward_backward_error_px=error_values.tolist(),
        )


class TapirPointTracker:
    """Offline TAPIR clip inference with original-image pixel outputs.

    This is the torch TAPIR.forward path, not TAPIR's causal streaming API.
    A caller may supply an initialized model or explicitly load a local checkpoint.
    Model inference can run on CPU or CUDA; this module never downloads weights.
    """

    def __init__(
        self,
        model: Any,
        *,
        device: str = "cpu",
        resize_hw: tuple[int, int] = (256, 256),
        visibility_threshold: float = 0.5,
    ):
        if len(resize_hw) != 2 or any(not isinstance(v, int) or v < 8 or v % 8 for v in resize_hw):
            raise ValueError("resize_hw must contain two positive multiples of eight")
        if not 0 < visibility_threshold < 1:
            raise ValueError("visibility_threshold must be between zero and one")
        if not callable(model):
            raise ValueError("model must implement the callable torch TAPIR.forward contract")
        self.model, self.device = model, device
        self.resize_hw = tuple(resize_hw)
        self.visibility_threshold = float(visibility_threshold)
        self.checkpoint_sha256: str | None = None

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        model_kwargs: dict | None = None,
        **kwargs,
    ) -> TapirPointTracker:
        checkpoint = Path(path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Local TAPIR checkpoint not found: {checkpoint}")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        sha256 = digest.hexdigest()
        if expected_sha256 is not None and sha256 != expected_sha256.lower():
            raise ValueError("TAPIR checkpoint SHA-256 does not match expected_sha256")
        try:
            import torch
            from tapnet.torch.tapir_model import TAPIR
        except ImportError as exc:
            raise RuntimeError(
                "TAPIR requires torch and tapnet[torch]; install tapnet at " + TAPNET_REVISION
            ) from exc
        model = TAPIR(**({"pyramid_level": 1} | (model_kwargs or {})))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        tracker = cls(model, **kwargs)
        tracker.checkpoint_sha256 = sha256
        return tracker

    def track(self, frames: Any, query_points: Any) -> dict:
        np = _numpy()
        video, points = _inputs(frames, query_points)
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise RuntimeError("TAPIR tracking requires torch and tapnet[torch].") from exc
        count, height, width, _ = video.shape
        resized_h, resized_w = self.resize_hw
        tensor = torch.as_tensor(video, dtype=torch.float32, device=self.device)
        # Resize spatial dimensions, keeping RGB channel-last at the model boundary.
        tensor = functional.interpolate(
            tensor.permute(0, 3, 1, 2), size=self.resize_hw, mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        tensor = (tensor / 255.0 * 2.0 - 1.0).contiguous().unsqueeze(0)
        queries = np.zeros((1, len(points), 3), dtype=np.float32)
        queries[0, :, 1] = points[:, 1] * resized_h / height
        queries[0, :, 2] = points[:, 0] * resized_w / width
        query_tensor = torch.as_tensor(queries, device=self.device)
        model = self.model
        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model = model.eval()
        with torch.inference_mode():
            output = model(tensor, query_tensor, is_training=False)
        try:
            tracks = output["tracks"]
            occlusion = output["occlusion"]
            expected_dist = output["expected_dist"]
            if (
                tuple(tracks.shape) != (1, len(points), count, 2)
                or tuple(occlusion.shape) != (1, len(points), count)
                or tuple(expected_dist.shape) != (1, len(points), count)
            ):
                raise ValueError("unexpected output shapes")
            confidence = (1 - torch.sigmoid(occlusion)) * (1 - torch.sigmoid(expected_dist))
            confidence = confidence[0].transpose(0, 1).detach().cpu().numpy()
            tracks = tracks[0].permute(1, 0, 2).detach().cpu().numpy().copy()
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError("model output does not match the TAPIR.forward contract") from exc
        tracks[..., 0] *= width / resized_w
        tracks[..., 1] *= height / resized_h
        visible = (
            np.isfinite(confidence)
            & (confidence > self.visibility_threshold)
            & _inside(tracks, width, height)
        )
        confidence_values = confidence.astype(object)
        confidence_values[~np.isfinite(confidence)] = None
        return _result(
            tracks,
            visible,
            "tapir",
            height,
            width,
            visibility_confidence=confidence_values.tolist(),
            checkpoint_sha256=self.checkpoint_sha256,
            tapnet_contract_revision=TAPNET_REVISION,
        )
