"""grconvnet tool bundle: ranked top-down parallel-jaw grasps from one RGB-D frame.

Runs GR-ConvNet (Kumra, Joshi, Sahin, "Antipodal Robotic Grasping using
Generative Residual Convolutional Neural Network", IROS 2020,
https://arxiv.org/abs/1909.04810) with the checkpoints released in
https://github.com/skumra/robotic-grasping (BSD-3-Clause, Copyright (c) 2019
Sulabh Kumra, MABL, RIT; Copyright (c) 2018 Douglas Morrison, ACRV, QUT). The
checkpoint is downloaded from that repository at a pinned commit on first use
and checked against its SHA-256.

Adapted from skumra/robotic-grasping at commit 183c6f68 (BSD-3-Clause):

- ``inference/models/grconvnet3.py`` (``GenerativeResnet``) and
  ``inference/models/grasp_model.py`` (``ResidualBlock``): copied, the layers
  unchanged so the released state dicts load; training code and the dropout
  heads' branch dropped (the released checkpoints were saved in eval use).
- ``utils/data/camera_data.py`` (``CameraData``) and
  ``utils/dataset_processing/image.py`` (``normalise``): the input
  normalisation (RGB / 255 minus its mean; depth in metres minus its mean,
  clipped to [-1, 1]; depth channel first). Changed: the square crop is centred
  on a mask / ROI and resampled to the network's input at a chosen zoom (4/3
  by default) instead of a fixed centre crop at native resolution; depth holes
  are filled from the nearest measured pixel (upstream inpaints its Cornell
  depth with OpenCV; its camera path fills nothing).
- ``inference/post_process.py`` and ``utils/dataset_processing/grasp.py``
  (``post_process_output``, ``detect_grasps``): Gaussian smoothing (quality
  sigma 2, width sigma 1, width x 150 px) and peak picking (min distance 20 px,
  threshold 0.2, border excluded, as ``skimage.feature.peak_local_max``).
  Changed: cos 2a / sin 2a are smoothed before the angle is taken (upstream
  smooths the angle itself, which tears where it wraps at +-90 deg); peaks are
  found with scipy, not scikit-image.

New here: pixel -> metric width, the jaw axis and grasp point in the world
from a camera pose, an object box measured from the depth along the jaw line
(so the merged OBB-driven grasp skills can take it), and the gripper-span check.

The model loads lazily on first call (module-level singleton); importing this
module never imports torch. Knobs via env:

- ``GAP_GRCONVNET_DEVICE`` -- torch device (default ``cpu``; ``cuda`` works).
- ``GAP_GRCONVNET_THREADS`` -- CPU threads (default ``min(8, cores)``).
- ``GAP_GRCONVNET_CHECKPOINT`` -- default checkpoint key (see ``CHECKPOINTS``).
- ``GAP_GRCONVNET_CHECKPOINT_PATH`` -- a local checkpoint file for that key
  (digest not checked).
- ``GAP_GRCONVNET_CACHE`` -- download directory
  (default ``~/.cache/roboskillhub/grconvnet``).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import tool
from gap_core.types import OrientedBoundingBox, Se3Pose

logger = logging.getLogger(__name__)

#: skumra/robotic-grasping commit the checkpoints come from (master, 2021-11-09).
UPSTREAM_COMMIT = "183c6f68c44c1c7ff0f07707e2db6fcfd6840d2d"
_RAW = (
    f"https://raw.githubusercontent.com/skumra/robotic-grasping/{UPSTREAM_COMMIT}/trained-models/"
)

#: The released RGB-D / depth GR-ConvNet3 checkpoints, best epoch of each
#: (README "Model Training": Jacquard at --input-size 300, Cornell at 224),
#: with the SHA-256 of the file at the pinned commit (checked 2026-11-08).
CHECKPOINTS: dict[str, dict[str, Any]] = {
    "cornell-rgbd": {
        "path": "cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98",
        "sha256": "e6c03afc0f8266d29ec92141088cc6a557215fabc2681d4ca5a162f451853d28",
        "input_size": 224,
        "channels": 4,
    },
    "jacquard-rgbd": {
        "path": "jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93",
        "sha256": "adfb2cbbb8df2708a732e12ddc4db114f3ec399ffb5d403ca75c5b5b9e769171",
        "input_size": 300,
        "channels": 4,
    },
    "jacquard-d": {
        "path": "jacquard-d-grconvnet3-drop0-ch32/epoch_50_iou_0.94",
        "sha256": "67d8d2a948bd24f68547bbfd851f6ae2b229f07cf7ed1b75ed8c6730c6626bc6",
        "input_size": 300,
        "channels": 1,
    },
}
#: Depth only: on the station frames its top grasp held on the cube and on the open box's
#: wall; both RGB-D checkpoints missed the cube (SKILL.md, "Measured").
DEFAULT_CHECKPOINT = "jacquard-d"

#: The YAM linear_4310's jaw span (94 mm, the station twin's gripper spec; the
#: number skills/classifying-failure-from-proprioception uses).
GRIPPER_WIDTH_MAX_M = 0.094
#: Upstream's post-processing: width head x 150 px, smoothing sigmas (px of
#: the network input), peak spacing and quality floor (detect_grasps).
WIDTH_SCALE_PX = 150.0
QUALITY_SIGMA = 2.0
ANGLE_SIGMA = 2.0
WIDTH_SIGMA = 1.0
PEAK_MIN_DISTANCE = 20
QUALITY_THRESHOLD = 0.2
#: Network px per frame px of the crop around the target: 4/3 (a 225 px crop
#: for Jacquard's 300 px input) did best of 0.67 / 1 / 1.33 and mask-filling
#: crops on the station frames (SKILL.md, "Measured"); 1 is upstream's camera path.
ZOOM = 4.0 / 3.0
#: A mask whose longer side exceeds this share of the zoomed crop gets a larger crop.
MAX_MASK_SHARE = 0.9
#: Smallest crop side (px of the frame) ``object_fill`` may ask for: below ~64 px a
#: 25 px cube is upsampled past where the RGB carries any texture.
MIN_CROP_PX = 64
#: Grasp centres may sit this far (px of the frame) outside the mask: its
#: edge pixels are the noisiest.
MASK_DILATE_PX = 3
#: Depth <= 1 cm is a hole (the depth bundle's bound).
MIN_VALID_DEPTH_M = 0.01
#: Height over the support that counts as object along the jaw line: the
#: larger of this and a third of the centre's height (sensor noise ~1-4 mm
#: on the station's tables, tools/depth "Measured").
MIN_OBJECT_HEIGHT_M = 0.004
#: Window pixels further than this below the grasp centre are not the
#: support (the floor past the table edge on the overhead frames).
SUPPORT_BAND_M = 0.15

#: CPU threads by default: 8 ran fastest on a 20-core Core Ultra 7 265KF (75 ms a
#: call against 685 ms on all 20); on a 128-core Xeon 8352S 16 were barely faster
#: (115 against 123 ms) with a worse p90.
DEFAULT_THREADS = 8

_load_lock = threading.Lock()
_run_lock = threading.Lock()
_runtimes: dict[str, Any] = {}
_net_classes: Any = None


class PlanarGrasp(TypedDict):
    rank: int
    quality: float  # the network's grasp quality at the centre, 0..1
    center_px: list[float]  # [u, v] in the frame as given
    angle_rad: float  # jaw closing axis in the image, anticlockwise from +u as seen, [-pi/2, pi/2)
    angle_deg: float
    width_px: float  # the network's jaw opening, px of the frame
    width_m: float  # the same at the centre's depth (world-horizontal with a camera pose)
    jaw_px: list[list[float]]  # [[u, v], [u, v]] the two jaw ends of the opening
    depth_m: float  # depth at the centre (median of measured pixels around it, else filled)
    depth_measured: bool
    in_mask: bool
    # Only with camera_pose (else None):
    position: (
        dict[str, float] | None
    )  # world grasp point: centre xy, mid-height of the object there
    top_z: float | None  # world z of the surface at the centre
    support_z: float | None
    object_height_m: float | None
    close_heading_deg: float | None  # world heading the jaws close ALONG, for robot.grasp_frame
    object_width_m: float | None  # object across the jaws, measured from depth (net width when not)
    object_length_m: float | None  # object along the jaws' faces, measured from depth
    obb: OrientedBoundingBox | None  # local x = closing axis; half-extents
    closing_is_short_axis: bool | None  # obb's short horizontal side is the closing axis


class Crop(TypedDict):
    x0: float  # frame px of the crop's top-left corner (may be negative: padded)
    y0: float
    side: float  # crop side in frame px
    scale: float  # frame px per network px


class ProposeTiming(TypedDict):
    load_s: float
    preprocess_s: float
    infer_s: float
    postprocess_s: float
    total_s: float


class ProposeResult(TypedDict):
    grasps: list[PlanarGrasp]
    n_peaks: int  # peaks above the quality threshold before the span check
    n_too_wide: int  # peaks dropped because the opening exceeds gripper_width_max
    best_quality: float  # the quality map's maximum in the (masked) crop
    crop: Crop
    checkpoint: str
    model: str
    device: str
    timing: ProposeTiming


# ---------------------------------------------------------------------------
# The network (copied from skumra/robotic-grasping, BSD-3-Clause)
# ---------------------------------------------------------------------------


def _network():
    """``(GenerativeResnet, ResidualBlock)``, built on first use so importing stays torch-free."""
    global _net_classes
    if _net_classes is not None:
        return _net_classes
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
            self.bn2 = nn.BatchNorm2d(in_channels)

        def forward(self, x_in):
            x = F.relu(self.bn1(self.conv1(x_in)))
            x = self.bn2(self.conv2(x))
            return x + x_in

    class GenerativeResnet(nn.Module):
        def __init__(
            self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0
        ):
            super().__init__()
            c = channel_size
            self.conv1 = nn.Conv2d(input_channels, c, kernel_size=9, stride=1, padding=4)
            self.bn1 = nn.BatchNorm2d(c)
            self.conv2 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(c * 2)
            self.conv3 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(c * 4)
            self.res1 = ResidualBlock(c * 4, c * 4)
            self.res2 = ResidualBlock(c * 4, c * 4)
            self.res3 = ResidualBlock(c * 4, c * 4)
            self.res4 = ResidualBlock(c * 4, c * 4)
            self.res5 = ResidualBlock(c * 4, c * 4)
            self.conv4 = nn.ConvTranspose2d(
                c * 4, c * 2, kernel_size=4, stride=2, padding=1, output_padding=1
            )
            self.bn4 = nn.BatchNorm2d(c * 2)
            self.conv5 = nn.ConvTranspose2d(
                c * 2, c, kernel_size=4, stride=2, padding=2, output_padding=1
            )
            self.bn5 = nn.BatchNorm2d(c)
            self.conv6 = nn.ConvTranspose2d(c, c, kernel_size=9, stride=1, padding=4)
            self.pos_output = nn.Conv2d(c, output_channels, kernel_size=2)
            self.cos_output = nn.Conv2d(c, output_channels, kernel_size=2)
            self.sin_output = nn.Conv2d(c, output_channels, kernel_size=2)
            self.width_output = nn.Conv2d(c, output_channels, kernel_size=2)

        def forward(self, x_in):
            x = F.relu(self.bn1(self.conv1(x_in)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.res5(self.res4(self.res3(self.res2(self.res1(x)))))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            x = self.conv6(x)
            return self.pos_output(x), self.cos_output(x), self.sin_output(x), self.width_output(x)

    _net_classes = (GenerativeResnet, ResidualBlock)
    return _net_classes


class _Unpickler(pickle.Unpickler):
    """Upstream saved whole modules (``torch.save(net)``); map its classes onto the copies above."""

    def find_class(self, module, name):
        if module.startswith("inference.models."):
            net, block = _network()
            if name == "GenerativeResnet":
                return net
            if name == "ResidualBlock":
                return block
            raise pickle.UnpicklingError(f"unexpected class {module}.{name} in the checkpoint")
        return super().find_class(module, name)


class _PickleModule:
    Unpickler = _Unpickler
    load = pickle.load


def _load_state_dict(path: str):
    """The checkpoint's state dict. It is a pickled module, so it cannot load with weights_only."""
    import torch

    model = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_PickleModule)
    return model.state_dict()


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def _checkpoint_key(name: str | None) -> str:
    key = name or os.environ.get("GAP_GRCONVNET_CHECKPOINT") or DEFAULT_CHECKPOINT
    if key not in CHECKPOINTS:
        raise ValueError(f"grconvnet: checkpoint must be one of {sorted(CHECKPOINTS)}, got {key!r}")
    return key


def _cache_path(key: str) -> Path:
    root = (
        os.environ.get("GAP_GRCONVNET_CACHE")
        or Path.home() / ".cache" / "roboskillhub" / "grconvnet"
    )
    return Path(root) / CHECKPOINTS[key]["path"].replace("/", "__")


def _model_id(key: str) -> str:
    override = os.environ.get("GAP_GRCONVNET_CHECKPOINT_PATH")
    if override and key == _checkpoint_key(None):
        return override
    return f"skumra/robotic-grasping@{UPSTREAM_COMMIT[:8]}:{CHECKPOINTS[key]['path']}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_file(key: str) -> str:
    """The checkpoint on disk, downloading and checking it the first time."""
    override = os.environ.get("GAP_GRCONVNET_CHECKPOINT_PATH")
    if override and key == _checkpoint_key(None):
        return override
    spec = CHECKPOINTS[key]
    dest = _cache_path(key)
    if dest.is_file() and _sha256(dest) == spec["sha256"]:
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    url = _RAW + spec["path"]
    logger.info("[grconvnet] downloading %s", url)
    urllib.request.urlretrieve(url, part)
    got = _sha256(part)
    if got != spec["sha256"]:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{url} has sha256 {got}, expected {spec['sha256']}")
    part.replace(dest)
    return str(dest)


def weights_cached() -> bool | None:
    """Filesystem-only probe for ``gap check``: the default checkpoint is on disk."""
    override = os.environ.get("GAP_GRCONVNET_CHECKPOINT_PATH")
    if override:
        return os.path.isfile(override)
    return _cache_path(_checkpoint_key(None)).is_file()


def prefetch() -> None:
    """Download every pinned checkpoint (``gap skills check --download``; 7.7 MB each)."""
    for key in CHECKPOINTS:
        logger.info("[grconvnet] prefetching %s ...", _model_id(key))
        _checkpoint_file(key)
    logger.info("[grconvnet] prefetch complete")


class _Runtime:
    """One GR-ConvNet3 on the device; ``infer`` takes a normalised [C, S, S] input."""

    def __init__(self, key: str, path: str, device: str) -> None:
        import torch

        spec = CHECKPOINTS[key]
        net_cls, _ = _network()
        net = net_cls(input_channels=spec["channels"], channel_size=32)
        net.load_state_dict(_load_state_dict(path))
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cpu":
            threads = os.environ.get("GAP_GRCONVNET_THREADS") or min(
                DEFAULT_THREADS, os.cpu_count() or 1
            )
            torch.set_num_threads(int(threads))
        self.net = net.to(self.device).eval()

    def infer(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Raw ``(quality, cos 2a, sin 2a, width / 150)`` maps, each ``[S, S]``."""
        torch = self.torch
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None].to(self.device)
            out = self.net(t)
        return tuple(o[0, 0].float().cpu().numpy() for o in out)  # type: ignore[return-value]


def _get_runtime(key: str) -> tuple[Any, float]:
    """The resident runtime for ``key`` and the seconds this call spent loading it."""
    with _load_lock:
        if key not in _runtimes:
            t0 = time.perf_counter()
            device = os.environ.get("GAP_GRCONVNET_DEVICE", "cpu")
            path = _checkpoint_file(key)
            logger.info("Loading GR-ConvNet (%s) on %s from %s", key, device, path)
            _runtimes[key] = _Runtime(key, path, device)
            return _runtimes[key], time.perf_counter() - t0
        return _runtimes[key], 0.0


# ---------------------------------------------------------------------------
# Inputs and geometry (numpy only)
# ---------------------------------------------------------------------------


def _rgb(rgb: Any) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"grconvnet: rgb must be uint8 [H, W, 3], got {arr.dtype} {arr.shape}")
    return arr


def _depth(depth_m: Any, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(depth_m)
    if arr.ndim != 2 or arr.shape != shape:
        raise ValueError(
            f"grconvnet: depth_m must be [H, W] = {list(shape)}, got {list(arr.shape)}"
        )
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(
            f"grconvnet: depth_m must be float metres, got {arr.dtype} (divide a uint16 millimetre image by 1000)"
        )
    out = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out[out <= MIN_VALID_DEPTH_M] = 0.0
    return out


def _intrinsics(K: Any) -> np.ndarray:
    k = np.asarray(K, dtype=np.float64)
    if k.shape != (3, 3) or not (k[0, 0] > 0 and k[1, 1] > 0):
        raise ValueError(
            f"grconvnet: K must be a 3x3 pinhole matrix with fx, fy > 0, got {k.tolist()}"
        )
    return k


def _mask(mask: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.shape != shape:
        raise ValueError(f"grconvnet: mask must be [H, W] = {list(shape)}, got {list(arr.shape)}")
    out = arr > 0
    if not out.any():
        raise ValueError("grconvnet: mask is empty")
    return out


def _pose_matrix(pose: Any) -> np.ndarray | None:
    """Camera-to-world 4x4 from an ``Se3Pose`` (wxyz) or a 4x4 matrix (OpenCV camera axes)."""
    if pose is None:
        return None
    if isinstance(pose, dict):
        p, q = pose["position"], pose["rotation"]
        w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
        n = math.sqrt(w * w + x * x + y * y + z * z)
        w, x, y, z = w / n, x / n, y / n, z / n
        m = np.eye(4)
        m[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
        m[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
        return m
    m = np.asarray(pose, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(
            "grconvnet: camera_pose must be an Se3Pose or a 4x4 camera-to-world matrix"
        )
    return m


def _crop_box(
    shape: tuple[int, int],
    mask: np.ndarray | None,
    roi: Any,
    crop_px: float | None,
    input_size: int,
    zoom: float,
    object_fill: float | None,
) -> tuple[float, float, float]:
    """``(x0, y0, side)`` of the square crop, frame px (it may run off the frame: padded)."""
    h, w = shape
    if roi is not None:
        u0, v0, u1, v1 = (float(v) for v in roi)
        if not (u1 > u0 and v1 > v0):
            raise ValueError(
                f"grconvnet: roi must be [u0, v0, u1, v1] with u1 > u0, v1 > v0, got {roi}"
            )
        cu, cv, side = (u0 + u1) / 2, (v0 + v1) / 2, max(u1 - u0, v1 - v0)
    elif mask is not None:
        vs, us = np.nonzero(mask)
        cu, cv = (us.min() + us.max() + 1) / 2, (vs.min() + vs.max() + 1) / 2
        long_side = max(us.max() - us.min() + 1, vs.max() - vs.min() + 1)
        if object_fill is not None:
            side = max(long_side / object_fill, MIN_CROP_PX)
        else:
            side = max(input_size / zoom, long_side / MAX_MASK_SHARE)
    else:
        cu, cv, side = w / 2, h / 2, input_size / zoom
    if crop_px is not None:
        side = float(crop_px)
    side = float(round(side))
    return cu - side / 2, cv - side / 2, side


def _sample(
    img: np.ndarray, x0: float, y0: float, side: float, size: int, order: int
) -> np.ndarray:
    """Resample the square ``(x0, y0, side)`` of ``img`` onto ``size x size`` (edge-padded)."""
    from scipy.ndimage import map_coordinates

    s = side / size
    c = (np.arange(size) + 0.5) * s - 0.5
    rows, cols = np.meshgrid(c + y0, c + x0, indexing="ij")
    rows = np.clip(rows, 0, img.shape[0] - 1)
    cols = np.clip(cols, 0, img.shape[1] - 1)
    if img.ndim == 2:
        return map_coordinates(img, [rows, cols], order=order, mode="nearest")
    return np.stack(
        [
            map_coordinates(img[..., i], [rows, cols], order=order, mode="nearest")
            for i in range(img.shape[2])
        ],
        axis=-1,
    )


def _fill_holes(depth: np.ndarray) -> np.ndarray:
    """Every hole takes its nearest measured pixel's depth (all-hole stays 0)."""
    from scipy.ndimage import distance_transform_edt

    holes = depth <= 0
    if not holes.any() or holes.all():
        return depth
    idx = distance_transform_edt(holes, return_distances=False, return_indices=True)
    return depth[tuple(idx)]


def _network_input(rgb_c: np.ndarray, depth_c: np.ndarray, channels: int) -> np.ndarray:
    """Upstream's normalisation, depth channel first: ``[C, S, S]`` float32."""
    d = depth_c.astype(np.float32)
    d = np.clip(d - d.mean(), -1.0, 1.0)
    if channels == 1:
        return d[None]
    c = rgb_c.astype(np.float32) / 255.0
    c -= c.mean()
    return np.concatenate([d[None], c.transpose(2, 0, 1)], axis=0)


def _post_process(q, cos2, sin2, width):
    from scipy.ndimage import gaussian_filter

    q = gaussian_filter(np.asarray(q, np.float64), QUALITY_SIGMA)
    ang = 0.5 * np.arctan2(
        gaussian_filter(np.asarray(sin2, np.float64), ANGLE_SIGMA),
        gaussian_filter(np.asarray(cos2, np.float64), ANGLE_SIGMA),
    )
    wid = gaussian_filter(np.asarray(width, np.float64) * WIDTH_SCALE_PX, WIDTH_SIGMA)
    return q, ang, wid


def _peaks(q: np.ndarray, min_distance: int, threshold: float, limit: int) -> list[tuple[int, int]]:
    """Local maxima of ``q`` as ``peak_local_max(exclude_border=min_distance)`` picks them."""
    from scipy.ndimage import maximum_filter

    size = 2 * min_distance + 1
    peak = (q == maximum_filter(q, size=size, mode="nearest")) & (q > threshold)
    b = min_distance
    peak[:b, :] = peak[-b:, :] = False
    peak[:, :b] = peak[:, -b:] = False
    rows, cols = np.nonzero(peak)
    order = np.argsort(-q[rows, cols], kind="stable")
    out: list[tuple[int, int]] = []
    for i in order:
        r, c = int(rows[i]), int(cols[i])
        if all(max(abs(r - r2), abs(c - c2)) > min_distance for r2, c2 in out):
            out.append((r, c))
        if len(out) >= limit:
            break
    return out


def _depth_at(raw: np.ndarray, filled: np.ndarray, u: float, v: float) -> tuple[float, bool]:
    h, w = raw.shape
    r, c = int(round(v)), int(round(u))
    r, c = min(max(r, 0), h - 1), min(max(c, 0), w - 1)
    win = raw[max(r - 2, 0) : r + 3, max(c - 2, 0) : c + 3]
    good = win[win > 0]
    if good.size:
        return float(np.median(good)), True
    return float(filled[r, c]), False


def _ray(K: np.ndarray, u: float, v: float) -> np.ndarray:
    return np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])


def _on_plane(
    T: np.ndarray, K: np.ndarray, u: float, v: float, z_world: float
) -> np.ndarray | None:
    """Where pixel ``(u, v)``'s ray meets the horizontal world plane ``z = z_world``."""
    d = T[:3, :3] @ _ray(K, u, v)
    o = T[:3, 3]
    if abs(d[2]) < 1e-9:
        return None
    t = (z_world - o[2]) / d[2]
    return o + t * d if t > 0 else None


def _world_points(
    T: np.ndarray, K: np.ndarray, depth: np.ndarray, us: np.ndarray, vs: np.ndarray
) -> np.ndarray:
    z = depth[vs, us]
    cam = np.stack([(us - K[0, 2]) / K[0, 0] * z, (vs - K[1, 2]) / K[1, 1] * z, z], axis=-1)
    return cam @ T[:3, :3].T + T[:3, 3]


def _support_z(T, K, raw, u, v, top_z, width_px, mask) -> float | None:
    """Support height under a grasp: low quartile of measured world z in a window around it."""
    h, w = raw.shape
    half = int(max(1.5 * width_px, 20))
    r0, r1 = max(int(v) - half, 0), min(int(v) + half + 1, h)
    c0, c1 = max(int(u) - half, 0), min(int(u) + half + 1, w)
    vs, us = np.mgrid[r0:r1, c0:c1]
    keep = raw[vs, us] > 0
    if mask is not None:
        keep &= ~mask[vs, us]
    if keep.sum() < 10:
        return None
    z = _world_points(T, K, raw, us[keep], vs[keep])[:, 2]
    z = z[(z > top_z - SUPPORT_BAND_M) & (z < top_z + 0.01)]
    if z.size < 10:
        return None
    return float(np.percentile(z, 25))


def _extent_along(
    T, K, raw, u, v, direction_px, reach_px, centre, axis_world, support_z, h_min
) -> float | None:
    """How far (m, from the centre along ``axis_world``) the object reaches each way, summed."""
    h, w = raw.shape
    total = 0.0
    for sign in (1.0, -1.0):
        best, misses = 0.0, 0
        for step in range(1, int(reach_px) + 1):
            uu = u + sign * direction_px[0] * step
            vv = v + sign * direction_px[1] * step
            r, c = int(round(vv)), int(round(uu))
            if not (0 <= r < h and 0 <= c < w):
                break
            if raw[r, c] <= 0:
                continue
            p = _world_points(T, K, raw, np.array([c]), np.array([r]))[0]
            if p[2] - support_z >= h_min:
                best, misses = max(best, abs(float(np.dot(p[:2] - centre[:2], axis_world)))), 0
            else:
                misses += 1
                if misses >= 2:
                    break
        total += best
    return total if total > 0 else None


def _quat_z(theta: float) -> dict[str, float]:
    return {"w": math.cos(theta / 2), "x": 0.0, "y": 0.0, "z": math.sin(theta / 2)}


def _wrap_half_deg(a: float) -> float:
    """Wrap to [-90, 90): parallel jaws are symmetric under a half turn."""
    return (a + 90.0) % 180.0 - 90.0


def _world_grasp(T, K, raw, mask, u, v, depth, angle, width_px, support_z_given):
    centre = T[:3, :3] @ (_ray(K, u, v) * depth) + T[:3, 3]
    top_z = float(centre[2])
    d = np.array([math.cos(angle), -math.sin(angle)])
    a = _on_plane(T, K, u - d[0] * width_px / 2, v - d[1] * width_px / 2, top_z)
    b = _on_plane(T, K, u + d[0] * width_px / 2, v + d[1] * width_px / 2, top_z)
    if a is None or b is None:
        return None
    close = (b - a)[:2]
    width_m = float(np.linalg.norm(close))
    if width_m < 1e-6:
        return None
    axis = close / width_m
    heading = _wrap_half_deg(math.degrees(math.atan2(axis[1], axis[0])))
    axis = np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading))])
    perp = np.array([-axis[1], axis[0]])

    if support_z_given is not None:
        support = float(support_z_given)
    else:
        support = _support_z(T, K, raw, u, v, top_z, width_px, mask)
    if support is None:
        support = top_z
    height = max(top_z - support, 0.0)
    h_min = max(MIN_OBJECT_HEIGHT_M, height / 3)
    across = along = None
    if height >= MIN_OBJECT_HEIGHT_M:
        px_per_m = width_px / width_m
        # The image direction of the world perpendicular, from a short step on the plane.
        tip = centre[:2] + perp * 0.01
        cam = np.linalg.inv(T) @ np.array([tip[0], tip[1], top_z, 1.0])
        du = K[0, 0] * cam[0] / cam[2] + K[0, 2] - u
        dv = K[1, 1] * cam[1] / cam[2] + K[1, 2] - v
        n = math.hypot(du, dv) or 1.0
        across = _extent_along(T, K, raw, u, v, d, 1.5 * width_px, centre, axis, support, h_min)
        along = _extent_along(
            T,
            K,
            raw,
            u,
            v,
            (du / n, dv / n),
            max(2.0 * width_px, 0.15 * px_per_m),
            centre,
            perp,
            support,
            h_min,
        )
    obj_w = across if across is not None else width_m
    obj_l = along if along is not None else obj_w
    mid = support + height / 2
    return {
        "position": {"x": float(centre[0]), "y": float(centre[1]), "z": float(mid)},
        "top_z": top_z,
        "support_z": float(support),
        "object_height_m": float(height),
        "close_heading_deg": float(heading),
        "width_m": width_m,
        "object_width_m": float(obj_w),
        "object_length_m": float(obj_l),
        "obb": {
            "center": {"x": float(centre[0]), "y": float(centre[1]), "z": float(mid)},
            "extent": {"x": obj_w / 2, "y": obj_l / 2, "z": max(height, 1e-3) / 2},
            "orientation": _quat_z(math.radians(heading)),
        },
        "closing_is_short_axis": bool(obj_w <= obj_l),
    }


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(
    name="grconvnet.propose",
    summary=(
        "Ranked top-down parallel-jaw grasps from an RGB-D frame (GR-ConvNet): pixel centre, "
        "jaw angle, width in px and m, quality; world grasp point, closing heading and box with a camera pose."
    ),
    tags=("perception", "grasping"),
)
def propose(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray | None = None,
    camera_pose: Se3Pose | None = None,
    n: int = 10,
    gripper_width_max: float = GRIPPER_WIDTH_MAX_M,
    roi: list[float] | None = None,
    zoom: float = ZOOM,
    crop_px: float | None = None,
    object_fill: float | None = None,
    quality_threshold: float = QUALITY_THRESHOLD,
    support_z: float | None = None,
    checkpoint: str | None = None,
) -> ProposeResult:
    """Grasp proposals for one RGB-D frame, best first.

    Args:
        rgb: uint8 ``[H, W, 3]`` RGB.
        depth_m: float ``[H, W]`` metres registered to ``rgb`` (0 / NaN = hole).
        K: 3x3 pinhole intrinsics of the frame.
        mask: Optional uint8 / bool ``[H, W]`` of the target (``sam3``'s
            0 / 255). The crop is centred on it and grasp centres must lie on
            it (dilated 3 px). Without a mask or ``roi`` the crop is centred
            on the frame.
        camera_pose: Camera-to-world ``Se3Pose`` (OpenCV camera axes) or 4x4.
            With it each grasp gets a world point, closing heading and box.
        n: Most grasps returned.
        gripper_width_max: The hand's opening (m); wider proposals are dropped
            (counted in ``n_too_wide``). Default: the YAM linear_4310's 94 mm.
        roi: ``[u0, v0, u1, v1]`` square-ified crop instead of the mask's.
        zoom: Network px per frame px (crop side = input size / zoom, grown
            to hold the whole mask). 1 = upstream's native-resolution crop.
        crop_px: Crop side in frame px, overriding every other sizing.
        object_fill: Size the crop so the mask's longer side fills this share
            of it instead of using ``zoom``.
        quality_threshold: Peaks at or under this quality are not proposals.
        support_z: World height of the surface under the object, if known
            (else estimated from the depth around each grasp).
        checkpoint: ``jacquard-d`` (default, depth only: solid parts),
            ``cornell-rgbd`` (RGB-D: thin parts seen in colour, e.g. a flat
            cable from the overhead camera) or ``jacquard-rgbd``.

    Returns:
        ``ProposeResult``. Feed ``grasps[i]["close_heading_deg"]`` to
        ``robot.grasp_frame(close_heading_deg=...)`` and ``position`` as the
        tool-centre target, or ``obb`` to an OBB-driven grasp skill.
    """
    t_start = time.perf_counter()
    img = _rgb(rgb)
    shape = img.shape[:2]
    raw = _depth(depth_m, shape)
    k = _intrinsics(K)
    m = _mask(mask, shape)
    T = _pose_matrix(camera_pose)
    key = _checkpoint_key(checkpoint)
    spec = CHECKPOINTS[key]
    size = int(spec["input_size"])
    if int(n) < 1:
        raise ValueError(f"grconvnet.propose: n must be >= 1, got {n}")
    if object_fill is not None and not 0.05 <= float(object_fill) <= 1.0:
        raise ValueError(f"grconvnet.propose: object_fill must be in [0.05, 1], got {object_fill}")
    if not 0.1 <= float(zoom) <= 10.0:
        raise ValueError(f"grconvnet.propose: zoom must be in [0.1, 10], got {zoom}")
    if not float(gripper_width_max) > 0:
        raise ValueError(
            f"grconvnet.propose: gripper_width_max must be > 0, got {gripper_width_max}"
        )
    if not (raw > 0).any():
        raise ValueError("grconvnet.propose: depth_m has no valid pixel")

    x0, y0, side = _crop_box(
        shape,
        m,
        roi,
        crop_px,
        size,
        float(zoom),
        None if object_fill is None else float(object_fill),
    )
    scale = side / size
    # Work on the crop's window of the frame only (+1 px for the bilinear taps).
    r0, r1 = max(int(math.floor(y0)) - 1, 0), min(int(math.ceil(y0 + side)) + 1, shape[0])
    c0, c1 = max(int(math.floor(x0)) - 1, 0), min(int(math.ceil(x0 + side)) + 1, shape[1])
    win = (slice(r0, r1), slice(c0, c1))
    filled = raw.copy()
    filled[win] = _fill_holes(raw[win])
    rgb_c = _sample(img[win].astype(np.float32), x0 - c0, y0 - r0, side, size, order=1)
    depth_c = _sample(filled[win], x0 - c0, y0 - r0, side, size, order=1)
    x = _network_input(rgb_c, depth_c, spec["channels"])
    t_pre = time.perf_counter()

    try:
        runtime, load_s = _get_runtime(key)
    except Exception as exc:
        raise ToolError("grconvnet.propose", f"GR-ConvNet failed to load: {exc}") from exc
    t_infer = time.perf_counter()
    try:
        with _run_lock:
            q_raw, cos2, sin2, wid = runtime.infer(x)
    except Exception as exc:
        raise ToolError("grconvnet.propose", f"GR-ConvNet failed: {exc}") from exc
    infer_s = time.perf_counter() - t_infer
    if np.shape(q_raw) != (size, size):
        raise ToolError(
            "grconvnet.propose", f"model returned {np.shape(q_raw)}, expected {(size, size)}"
        )

    t_post = time.perf_counter()
    q, ang, width_net = _post_process(q_raw, cos2, sin2, wid)
    # Off the frame the crop is edge padding: no grasp centre there.
    c = (np.arange(size) + 0.5) * scale - 0.5
    gate = ((c + y0 >= 0) & (c + y0 <= shape[0] - 1))[:, None] & (
        (c + x0 >= 0) & (c + x0 <= shape[1] - 1)
    )[None]
    if m is not None:
        from scipy.ndimage import binary_dilation

        grown = binary_dilation(m[win], iterations=MASK_DILATE_PX)
        gate &= _sample(grown.astype(np.float32), x0 - c0, y0 - r0, side, size, order=0) > 0.5
    q = np.where(gate, q, 0.0)
    best = float(q.max())
    peaks = _peaks(q, PEAK_MIN_DISTANCE, float(quality_threshold), limit=max(4 * int(n), 20))

    grasps: list[PlanarGrasp] = []
    too_wide = 0
    for r, c in peaks:
        u = x0 + (c + 0.5) * scale - 0.5
        v = y0 + (r + 0.5) * scale - 0.5
        a = float(ang[r, c])
        # The width head is unbounded below; a non-positive opening is no grasp.
        w_px = max(float(width_net[r, c]) * scale, 1.0)
        depth, measured = _depth_at(raw, filled, u, v)
        world = None
        if T is not None and depth > 0:
            world = _world_grasp(T, k, raw, m, u, v, depth, a, w_px, support_z)
        width_m = world["width_m"] if world else w_px * depth / (0.5 * (k[0, 0] + k[1, 1]))
        if width_m > float(gripper_width_max):
            too_wide += 1
            continue
        d = (math.cos(a), -math.sin(a))
        g: PlanarGrasp = {
            "rank": len(grasps),
            "quality": round(float(q[r, c]), 4),
            "center_px": [round(u, 2), round(v, 2)],
            "angle_rad": round(a, 5),
            "angle_deg": round(math.degrees(a), 2),
            "width_px": round(w_px, 2),
            "width_m": round(float(width_m), 5),
            "jaw_px": [
                [round(u - d[0] * w_px / 2, 2), round(v - d[1] * w_px / 2, 2)],
                [round(u + d[0] * w_px / 2, 2), round(v + d[1] * w_px / 2, 2)],
            ],
            "depth_m": round(depth, 4),
            "depth_measured": measured,
            "in_mask": bool(
                m[
                    min(max(int(round(v)), 0), shape[0] - 1),
                    min(max(int(round(u)), 0), shape[1] - 1),
                ]
            )
            if m is not None
            else True,
            "position": None,
            "top_z": None,
            "support_z": None,
            "object_height_m": None,
            "close_heading_deg": None,
            "object_width_m": None,
            "object_length_m": None,
            "obb": None,
            "closing_is_short_axis": None,
        }
        if world:
            for name in (
                "position",
                "top_z",
                "support_z",
                "object_height_m",
                "close_heading_deg",
                "object_width_m",
                "object_length_m",
                "obb",
                "closing_is_short_axis",
            ):
                g[name] = world[name]  # type: ignore[literal-required]
        grasps.append(g)
        if len(grasps) >= int(n):
            break
    t_end = time.perf_counter()
    return {
        "grasps": grasps,
        "n_peaks": len(peaks),
        "n_too_wide": too_wide,
        "best_quality": round(best, 4),
        "crop": {"x0": round(x0, 2), "y0": round(y0, 2), "side": side, "scale": round(scale, 5)},
        "checkpoint": key,
        "model": _model_id(key),
        "device": str(getattr(runtime, "device", "?")),
        "timing": {
            "load_s": round(load_s, 4),
            "preprocess_s": round(t_pre - t_start, 4),
            "infer_s": round(infer_s, 4),
            "postprocess_s": round(t_end - t_post, 4),
            "total_s": round(t_end - t_start - load_s, 4),
        },
    }
