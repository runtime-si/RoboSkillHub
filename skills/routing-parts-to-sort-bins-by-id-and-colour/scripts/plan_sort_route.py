# Adapted from lego-sorter-public (https://github.com/mbric/lego-sorter-public, commit
# 525683d3da4ec03017d9355afff8df2f5218b179), src/orchestrator/sort_cycle.py (LandingRoi,
# _to_gray_small, _match_exposure, _roi_diff_mask, _largest_blob_label, _largest_blob_bbox_overlapping,
# _competing_blob_sizes, significant_secondary_blob, blob_touches_roi_edge, crop_content_scores,
# crop_looks_empty, _motion_score and the ADVANCING -> SETTLING -> CAPTURED part of SortCycle),
# src/orchestrator/routing.py (virtual_bin) and src/demo_brickognize/brickognize_client.py (predict),
# MIT License, Copyright (c) 2026 Matt Brickner (LICENSE.lego-sorter beside this skill).
# Kept: the empty reference frame; grey frames shrunk for the comparison, the live frame gain-matched
# to the reference (gain clipped to 0.55-1.85), a pixel changed at 16 grey levels; occupancy as the
# largest 4-connected changed blob in percent of the landing zone (8 px floor), a part landed after
# 2 frames at >= 0.8 %; settling as the zone's mean absolute frame difference under 8 grey levels for
# 0.5 s; the capture gates (a second blob >= 0.35 x the largest and >= 12 px, not touching it within
# 2 px, is another part; a blob touching the zone's edge is still arriving; a crop with no edge
# energy is an empty-tray ghost) with 4 re-settles before giving up, and after 4 edge rejects the
# blob is taken whole even past the zone's edge; the bin label by priority colour > category >
# unknown; the optional Brickognize upload of the padded crop (off by default here).
# Changed: the exposure gain is a ratio of medians over the zone (a 2 cm part fills 5-15 % of a 10 cm
# zone and pulled the mean) and is reused for the part's mask; the conveyor is gone - the landing zone is a staging pad on the table, given in world
# metres (projected through the camera) or as frame fractions; frames come from robot.get_observation
# and the loop ends when a part is captured or the wait runs out (no belt start / stop, no WaitClear:
# the arm takes the part away); the comparison window is the zone itself (plus a margin) shrunk to at
# most 160 px, not the whole frame shrunk to 160 x 120, because a 2 cm part is 10-20 px in a YAM
# station's overhead camera; a depth rise of 8 mm also marks a pixel changed when both frames have
# depth (a part the colour of the table); the part's mask for colour is the full-resolution change
# (largest channel difference) in the padded blob, its colour from colour.classify (CIEDE2000 against
# a palette) instead of HSV hue rules, and its footprint an upright box fitted to its depth points
# (cv2.minAreaRect in xy, the top at the 95th height percentile) for the pick, optionally replaced by
# the best pose6d fit of known part models (which also names the part id); the routing is a
# configurable rule list (colour, category, part id, size bounds) over bins with capacities and an
# overflow bin, falling back to the source's colour > category > unknown labels, and outputs the
# chosen bin's world drop position; the source's lighting EMA, feeder-fringe and AE-ghost heuristics
# (a webcam on a conveyor) are not ported.
"""Wait for a part on the staging pad, classify it, and choose its sort bin and drop position."""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.request
import uuid
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PIXEL_DIFF_THR = 16.0
"""Grey levels a pixel must change by: lego-sorter's PIXEL_DIFF_THR ("lower catches dark/black
bricks on light trays")."""

DEPTH_RISE = 0.008
"""A pixel whose depth came this much nearer the camera than in the reference also changed [m]:
over 2-3x an Orbbec Gemini's noise at 0.9 m, under the 9.6 mm of a LEGO plate. Not in lego-sorter
(RGB only); catches a part the grey of the table."""

OCCUPIED_THRESHOLD = 0.8
"""Largest changed blob, percent of the staging zone, for a part to have landed: lego-sorter's
DEFAULT_OCCUPIED_THRESHOLD."""

OCCUPIED_FRAMES = 2
"""Consecutive frames over OCCUPIED_THRESHOLD: lego-sorter's DEFAULT_OCCUPIED_FRAMES."""

SETTLE_S = 0.5
"""How long the zone must stay still before the capture [s]: lego-sorter's DEFAULT_SETTLE_S."""

MOTION_THRESHOLD = 8.0
"""Mean absolute grey difference between consecutive frames over the zone that still counts as
moving: lego-sorter's DEFAULT_MOTION_THRESHOLD."""

MIN_BLOB_PIXELS = 8
"""Smallest blob (compare pixels) that is a part at all: lego-sorter's MIN_BLOB_PIXELS."""

MULTI_BLOB_MIN_FRAC = 0.35
MULTI_BLOB_MIN_PIXELS = 12
MULTI_BLOB_NEAR_GAP_PX = 2
"""A second blob at least 0.35 x the largest and 12 px, more than 2 px from it (a shadow or a split
silhouette touches the part), is another part: lego-sorter's values."""

BLOB_EDGE_MARGIN_PX = 1
"""A blob within 1 compare pixel of the zone's border is still arriving: lego-sorter's value."""

CAPTURE_GATE_MAX_REJECTS = 4
"""Re-settles after a rejected capture before giving up (edge contact: before taking the blob
whole past the zone's edge): lego-sorter's CAPTURE_GATE_MAX_REJECTS."""

CROP_MIN_EDGE_MEAN = 2.0
CROP_MIN_LAP_VAR = 10.0
CROP_MIN_STRONG_EDGE_FRAC = 0.015
CROP_STRONG_EDGE_THR = 8.0
"""A padded crop with mean gradient under 2, Laplacian variance under 10 and under 1.5 % of its
pixels with a gradient of 8 has no part in it (exposure or shadow ghost): lego-sorter's values."""

CROP_PAD_FRAC = 0.25
"""Padding around the blob for the colour mask and the Brickognize crop: lego-sorter's
API_CROP_PAD_FRAC."""

EXPOSURE_GAIN_LIMITS = (0.55, 1.85)
"""Gain range of the live-to-reference exposure match: lego-sorter's _match_exposure."""

COMPARE_MAX_SIDE = 160
"""The comparison window is shrunk until its longer side is at most this [px]. lego-sorter shrinks
the whole frame to 160 x 120; on a YAM station's overhead camera a 2 cm part is 10-20 px across,
which that would turn into 2-5 px, under MIN_BLOB_PIXELS."""

WINDOW_MARGIN_FRAC = 0.25
"""The comparison window is the zone grown by this fraction of its size on each side, so a part
straddling the zone's edge can be taken whole (lego-sorter's expand_outside_roi)."""

MAX_WAIT_S = 20.0
"""Longest wait for a part to land and settle [s]. A choice: a feeder or the other arm drops one
every few seconds."""

POLL_S = 0.1
"""Pause between frames while waiting [s]."""

MIN_PART_HEIGHT = 0.004
"""Depth points this far over the support are the part's for its footprint [m]: over the
calibrated YAM station camera's 5.4 mm RMS only in the mean; flat parts (jigsaw pieces, 2-4 mm)
fall back to every point in the mask."""

TOP_PERCENTILE = 95.0
"""The part's top is the 95th height percentile of its points: robust to flying pixels at its rim."""

MIN_POINTS = 12
"""Fewest depth points to fit the part's box."""

BRICKOGNIZE_URL = "https://api.brickognize.com/predict/parts/"
"""lego-sorter's Brickognize parts endpoint (a third-party service: its terms and rate limits are
not checked here, so use_brickognize is off by default)."""

BRICKOGNIZE_MIN_SCORE = 0.5
"""Least Brickognize score to accept its part id and category. A choice."""


class Output(TypedDict):
    route: str
    detail: str
    bin_name: str
    sort_bin: dict[str, Any]
    drop_position: dict[str, float]
    label: str
    matched_by: str
    part_colour: str
    colour_status: str
    colour_delta_e: float
    category: str
    part_id: str
    part_obb: dict[str, Any]
    support_z: float
    long_mm: float
    short_mm: float
    height_mm: float
    occupancy: float
    part_pixel: list[float]
    edge_override: bool
    pose6d_model: str
    pose6d_fitness: float
    bin_counts: dict[str, int]


# ---------------------------------------------------------------- frames


def _frame(source: Any, camera_name: str) -> dict[str, Any] | None:
    """A CameraFrame-like dict {rgb, depth?, intrinsics?, pose?} from an Observation, a CameraFrame
    or a bare RGB array."""
    if source is None:
        return None
    if isinstance(source, np.ndarray):
        return {"rgb": source}
    if not isinstance(source, dict):
        return None
    if source.get("rgb") is not None:
        return source
    cams = source.get("cameras") or []
    if isinstance(cams, dict):
        cams = list(cams.values())
    cams = [c for c in cams if isinstance(c, dict) and c.get("rgb") is not None]
    if camera_name:
        cams = [c for c in cams if c.get("name") == camera_name]
    return cams[0] if cams else None


def _rgb(frame: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(frame["rgb"])
    if arr.dtype != np.uint8:
        top = float(np.nanmax(arr)) if arr.size else 0.0
        arr = (np.clip(arr, 0, 1) * 255.0 if top <= 1.0 + 1e-6 else np.clip(arr, 0, 255)).astype(
            np.uint8
        )
    if arr.ndim == 3 and arr.shape[2] >= 4:
        arr = arr[..., :3]
    return arr


def _grey(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32).mean(axis=2) if rgb.ndim == 3 else rgb.astype(np.float32)


def _depth(frame: dict[str, Any]) -> np.ndarray | None:
    d = frame.get("depth")
    if d is None:
        return None
    d = np.asarray(d, dtype=np.float32)
    return d.reshape(d.shape[0], d.shape[1]) if d.ndim >= 2 else None


def _quat_matrix(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _camera(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(K, R_world_cam, t_world_cam) when the frame carries intrinsics and an (OpenCV-convention)
    camera pose."""
    k, pose = frame.get("intrinsics"), frame.get("pose")
    if k is None or not isinstance(pose, dict):
        return None
    p = pose.get("position") or {}
    t = np.array([float(p.get(c, 0.0)) for c in "xyz"])
    return np.asarray(k, dtype=np.float64).reshape(3, 3), _quat_matrix(pose["rotation"]), t


def _project(points: np.ndarray, cam: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    k, r, t = cam
    pc = (points - t) @ r
    z = np.maximum(pc[:, 2], 1e-6)
    return np.stack([k[0, 0] * pc[:, 0] / z + k[0, 2], k[1, 1] * pc[:, 1] / z + k[1, 2]], axis=1)


def _back_project(
    us: np.ndarray, vs: np.ndarray, z: np.ndarray, cam: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    k, r, t = cam
    pc = np.stack([(us - k[0, 2]) * z / k[0, 0], (vs - k[1, 2]) * z / k[1, 1], z], axis=1)
    return pc @ r.T + t


def zone_rect(
    shape_hw: tuple[int, int],
    frame: dict[str, Any],
    staging_zone: dict[str, Any] | None,
    staging_roi: dict[str, float] | None,
) -> tuple[tuple[int, int, int, int], np.ndarray | None]:
    """The staging zone in pixels: (x0, y0, x1, y1) and, for a world zone, the polygon mask of its
    projected corners inside that rectangle (None for a fractional zone)."""
    h, w = shape_hw
    if staging_zone:
        cam = _camera(frame)
        if cam is None:
            raise ValueError("a world staging_zone needs a frame with intrinsics and pose")
        c = staging_zone["center"]
        centre = np.array([float(c[k]) for k in "xyz"])
        hx, hy = (float(v) for v in staging_zone.get("half", [0.05, 0.05]))
        yaw = math.radians(float(staging_zone.get("yaw_deg", 0.0)))
        ax = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        ay = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        corners = np.array(
            [
                centre + sx * hx * ax + sy * hy * ay
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ]
        )
        px = _project(corners, cam)
        x0, y0 = (int(max(0, math.floor(v))) for v in px.min(axis=0))
        x1, y1 = int(min(w, math.ceil(px[:, 0].max()))), int(min(h, math.ceil(px[:, 1].max())))
        if x1 - x0 < 4 or y1 - y0 < 4:
            raise ValueError(
                f"the staging zone projects outside the frame ({px.round(1).tolist()})"
            )
        import cv2

        poly = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(poly, [np.round(px - [x0, y0]).astype(np.int32)], 1)
        return (x0, y0, x1, y1), poly.astype(bool)
    roi = staging_roi or {"top": 0.40, "bottom": 0.92, "left": 0.15, "right": 0.85}
    top = float(np.clip(roi.get("top", 0.40), 0.0, 0.95))
    bottom = float(np.clip(roi.get("bottom", 0.92), top + 0.05, 1.0))
    left = float(np.clip(roi.get("left", 0.15), 0.0, 0.95))
    right = float(np.clip(roi.get("right", 0.85), left + 0.05, 1.0))
    y0, x0 = int(h * top), int(w * left)
    return (x0, y0, max(x0 + 1, int(w * right)), max(y0 + 1, int(h * bottom))), None


# ---------------------------------------------------------------- lego-sorter's diff and blobs


def _shrink(img: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return img.astype(np.float32)
    import cv2

    h, w = img.shape[:2]
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(img.astype(np.float32), size, interpolation=cv2.INTER_AREA)


def exposure_gain(ref: np.ndarray, live: np.ndarray) -> float:
    """The gain that matches live's median to the reference's, so auto exposure does not read as a
    part (lego-sorter's _match_exposure, on medians instead of means: a 2 cm part fills 5-15 % of a
    10 cm zone and would pull a mean)."""
    ma, mb = float(np.median(ref)), float(np.median(live))
    if mb < 1.0 or ma < 1.0:
        return 1.0
    gain = float(np.clip(ma / mb, *EXPOSURE_GAIN_LIMITS))
    return 1.0 if abs(gain - 1.0) < 0.02 else gain


def label_blobs(changed: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected components (lego-sorter's _largest_blob_label labelling)."""
    from scipy import ndimage

    labels, n = ndimage.label(changed)
    return labels, int(n)


def blob_boxes(changed: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], np.ndarray]]:
    """Every blob of at least MIN_BLOB_PIXELS: (size, (x0, y0, x1, y1), its mask), largest first."""
    labels, n = label_blobs(changed)
    out = []
    for lab in range(1, n + 1):
        m = labels == lab
        size = int(m.sum())
        if size < MIN_BLOB_PIXELS:
            continue
        ys, xs = np.nonzero(m)
        out.append((size, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), m))
    out.sort(key=lambda b: b[0], reverse=True)
    return out


def bbox_separation(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Least gap between two boxes; 0 when they overlap or touch."""
    dx = max(0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0, max(a[1] - b[3], b[1] - a[3]))
    if dx == 0 or dy == 0:
        return max(dx, dy)
    return int(round(math.hypot(dx, dy)))


def competing_blob_sizes(changed: np.ndarray) -> list[int]:
    """Blob sizes after folding blobs within MULTI_BLOB_NEAR_GAP_PX of the largest into it."""
    blobs = blob_boxes(changed)
    if not blobs:
        return []
    primary, box = blobs[0][0], blobs[0][1]
    rest = []
    for size, bb, _ in blobs[1:]:
        if bbox_separation(box, bb) <= MULTI_BLOB_NEAR_GAP_PX:
            primary += size
        else:
            rest.append(size)
    return sorted([primary, *rest], reverse=True)


def significant_secondary_blob(changed: np.ndarray) -> bool:
    """A second blob big enough to be another part, not noise or a shadow."""
    sizes = competing_blob_sizes(changed)
    if len(sizes) < 2 or sizes[0] < MIN_BLOB_PIXELS:
        return False
    return sizes[1] >= MULTI_BLOB_MIN_PIXELS and sizes[1] >= MULTI_BLOB_MIN_FRAC * sizes[0]


def touches_edge(bbox: tuple[int, int, int, int], shape_hw: tuple[int, int]) -> bool:
    """True when the blob's box touches the zone's border (the part is still arriving)."""
    rh, rw = shape_hw
    m = BLOB_EDGE_MARGIN_PX
    return bbox[0] <= m or bbox[1] <= m or bbox[2] >= rw - m or bbox[3] >= rh - m


def crop_looks_empty(crop: np.ndarray) -> bool:
    """A crop with no edge structure: a blank tray under an exposure or shadow change."""
    luma = _grey(crop) if crop.ndim == 3 else crop.astype(np.float32)
    if luma.size < 16 or min(luma.shape[:2]) < 3:
        return True
    gy = np.abs(np.diff(luma, axis=0))
    gx = np.abs(np.diff(luma, axis=1))
    g = 0.5 * (gy[:, 1:] + gx[1:, :])
    edge = float(g.mean()) if g.size else 0.0
    strong = float((g >= CROP_STRONG_EDGE_THR).mean()) if g.size else 0.0
    c = luma[1:-1, 1:-1]
    lap = float(
        np.var(luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:] - 4.0 * c)
    )
    return (
        edge < CROP_MIN_EDGE_MEAN and lap < CROP_MIN_LAP_VAR and strong < CROP_MIN_STRONG_EDGE_FRAC
    )


class Window:
    """The comparison window around the staging zone, at compare resolution."""

    def __init__(
        self, ref: dict[str, Any], zone: tuple[int, int, int, int], poly: np.ndarray | None
    ):
        rgb = _rgb(ref)
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = zone
        mx, my = int((x1 - x0) * WINDOW_MARGIN_FRAC), int((y1 - y0) * WINDOW_MARGIN_FRAC)
        self.full = (max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my))
        wx0, wy0, wx1, wy1 = self.full
        self.scale = min(1.0, COMPARE_MAX_SIDE / max(wx1 - wx0, wy1 - wy0))
        self.ref_grey = _shrink(_grey(rgb)[wy0:wy1, wx0:wx1], self.scale)
        d = _depth(ref)
        self.ref_depth = (
            None if d is None else _shrink(np.nan_to_num(d[wy0:wy1, wx0:wx1]), self.scale)
        )
        ch, cw = self.ref_grey.shape
        sx, sy = cw / (wx1 - wx0), ch / (wy1 - wy0)
        self.zone = (
            int(round((x0 - wx0) * sx)),
            int(round((y0 - wy0) * sy)),
            max(int(round((x0 - wx0) * sx)) + 1, int(round((x1 - wx0) * sx))),
            max(int(round((y0 - wy0) * sy)) + 1, int(round((y1 - wy0) * sy))),
        )
        zx0, zy0, zx1, zy1 = self.zone
        self.in_zone = np.zeros((ch, cw), bool)
        if poly is None:
            self.in_zone[zy0:zy1, zx0:zx1] = True
        else:
            import cv2

            small = cv2.resize(
                poly.astype(np.uint8), (zx1 - zx0, zy1 - zy0), interpolation=cv2.INTER_NEAREST
            )
            self.in_zone[zy0:zy1, zx0:zx1] = small.astype(bool)
        self.zone_px = max(1, int(self.in_zone.sum()))
        self.gain = 1.0

    def grey(self, frame: dict[str, Any]) -> np.ndarray:
        wx0, wy0, wx1, wy1 = self.full
        return _shrink(_grey(_rgb(frame))[wy0:wy1, wx0:wx1], self.scale)

    def changed(self, frame: dict[str, Any], grey: np.ndarray) -> np.ndarray:
        """Changed pixels of the whole window vs the reference (exposure-matched grey, and depth)."""
        if grey.shape != self.ref_grey.shape:
            raise ValueError(
                f"frame size changed against the reference ({grey.shape} vs {self.ref_grey.shape})"
            )
        gain = exposure_gain(self.ref_grey[self.in_zone], grey[self.in_zone])
        self.gain = gain
        out = np.abs(self.ref_grey - np.clip(grey * gain, 0, 255)) >= PIXEL_DIFF_THR
        d = _depth(frame)
        if self.ref_depth is not None and d is not None:
            wx0, wy0, wx1, wy1 = self.full
            live_d = _shrink(np.nan_to_num(d[wy0:wy1, wx0:wx1]), self.scale)
            ok = (self.ref_depth > 0) & (live_d > 0)
            out |= ok & (self.ref_depth - live_d >= DEPTH_RISE)
        return out

    def occupancy(self, changed: np.ndarray) -> float:
        """Largest changed blob inside the zone, percent of the zone."""
        blobs = blob_boxes(changed & self.in_zone)
        return 100.0 * blobs[0][0] / self.zone_px if blobs else 0.0

    def motion(self, prev: np.ndarray, cur: np.ndarray) -> float:
        return float(np.mean(np.abs(prev[self.in_zone] - cur[self.in_zone])))

    def zone_changed(self, changed: np.ndarray) -> np.ndarray:
        zx0, zy0, zx1, zy1 = self.zone
        return (changed & self.in_zone)[zy0:zy1, zx0:zx1]

    def capture_blob(
        self, changed: np.ndarray, whole: bool
    ) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        """The part's blob: the zone's largest (lego-sorter) or, ``whole``, the window blob with the
        most pixels inside the zone (_largest_blob_bbox_overlapping), in compare pixels of the window."""
        if not whole:
            blobs = blob_boxes(changed & self.in_zone)
            return (blobs[0][1], blobs[0][2]) if blobs else None
        best = None
        for _size, bb, m in blob_boxes(changed):
            inside = int((m & self.in_zone).sum())
            if inside >= MIN_BLOB_PIXELS and (best is None or inside > best[0]):
                best = (inside, bb, m)
        return (best[1], best[2]) if best else None

    def to_full(
        self, bb: tuple[int, int, int, int], pad: float, shape_hw: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        wx0, wy0, _, _ = self.full
        s = self.scale
        x0, y0 = wx0 + int(bb[0] / s), wy0 + int(bb[1] / s)
        x1, y1 = wx0 + int(math.ceil(bb[2] / s)), wy0 + int(math.ceil(bb[3] / s))
        px, py = max(2, int(round((x1 - x0) * pad))), max(2, int(round((y1 - y0) * pad)))
        h, w = shape_hw
        return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)


def capture_reject(window: Window, changed: np.ndarray, allow_edge: bool) -> str | None:
    """lego-sorter's capture_reject_reason, before the crop: another part, or still arriving."""
    zc = window.zone_changed(changed)
    if significant_secondary_blob(zc):
        sizes = competing_blob_sizes(zc)
        return f"multiple parts in the zone (blobs {sizes[0]}+{sizes[1]} px)"
    blobs = blob_boxes(zc)
    if not blobs:
        return "empty crop"
    bb = blobs[0][1]
    rh, rw = zc.shape
    fills = (bb[2] - bb[0]) * (bb[3] - bb[1]) >= 0.45 * rh * rw
    if not allow_edge and not fills and touches_edge(bb, zc.shape):
        return "part touching the zone's edge"
    return None


# ---------------------------------------------------------------- the part


def part_mask(
    ref: dict[str, Any],
    cur: dict[str, Any],
    rect: tuple[int, int, int, int],
    seed: np.ndarray,
    gain: float = 1.0,
) -> np.ndarray:
    """Full-resolution mask of the part inside ``rect``: pixels whose largest channel moved by
    PIXEL_DIFF_THR (after the zone's exposure ``gain``: the crop is mostly part, so its own gain
    would be wrong) or whose depth rose DEPTH_RISE, the component that overlaps the
    compare-resolution blob ``seed`` (already scaled to ``rect``), holes filled."""
    from scipy import ndimage

    x0, y0, x1, y1 = rect
    a = _rgb(ref)[y0:y1, x0:x1].astype(np.float32)
    b = _rgb(cur)[y0:y1, x0:x1].astype(np.float32)
    ch = np.abs(a - np.clip(b * gain, 0, 255)).max(axis=2) >= PIXEL_DIFF_THR
    da, db = _depth(ref), _depth(cur)
    if da is not None and db is not None:
        za, zb = np.nan_to_num(da[y0:y1, x0:x1]), np.nan_to_num(db[y0:y1, x0:x1])
        ch |= (za > 0) & (zb > 0) & (za - zb >= DEPTH_RISE)
    ch = ndimage.binary_opening(ch, iterations=1) if ch.sum() > 50 else ch
    labels, n = ndimage.label(ch)
    if n == 0:
        return np.zeros_like(ch)
    hits = np.bincount(labels[seed & (labels > 0)].ravel(), minlength=n + 1)
    hits[0] = 0
    lab = (
        int(hits.argmax()) if hits.max() > 0 else int(np.bincount(labels.ravel())[1:].argmax()) + 1
    )
    return ndimage.binary_fill_holes(labels == lab)


def footprint(
    ref: dict[str, Any], cur: dict[str, Any], mask_full: np.ndarray, zone: tuple[int, int, int, int]
) -> dict[str, Any] | None:
    """The part's upright box in the world from depth: support height from the reference's zone,
    the part's points (over the support by MIN_PART_HEIGHT, else all), cv2.minAreaRect in xy and
    the top at the TOP_PERCENTILE height."""
    import cv2

    cam = _camera(cur)
    d, dref = _depth(cur), _depth(ref)
    if cam is None or d is None:
        return None
    ys, xs = np.nonzero(mask_full)
    z = d[ys, xs]
    ok = np.isfinite(z) & (z > 0.015)
    if int(ok.sum()) < MIN_POINTS:
        return None
    pts = _back_project(xs[ok].astype(float), ys[ok].astype(float), z[ok].astype(float), cam)
    support = None
    if dref is not None:
        x0, y0, x1, y1 = zone
        vv, uu = np.mgrid[y0:y1:2, x0:x1:2]
        zr = dref[vv, uu]
        good = np.isfinite(zr) & (zr > 0.015)
        if int(good.sum()) >= MIN_POINTS:
            ref_pts = _back_project(
                uu[good].astype(float), vv[good].astype(float), zr[good].astype(float), cam
            )
            support = float(np.median(ref_pts[:, 2]))
    if support is None:
        support = float(np.percentile(pts[:, 2], 2.0))
    high = pts[pts[:, 2] >= support + MIN_PART_HEIGHT]
    use = high if len(high) >= MIN_POINTS else pts
    top = float(np.percentile(use[:, 2], TOP_PERCENTILE))
    (cx, cy), (along, across), ang = cv2.minAreaRect((use[:, :2] * 1000.0).astype(np.float32))
    yaw = math.radians(ang)
    length, width = along, across
    if across > along:
        length, width, yaw = across, along, yaw + math.pi / 2
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    height = max(top - support, 1e-3)
    return {
        "obb": {
            "center": {"x": cx / 1000.0, "y": cy / 1000.0, "z": support + height / 2},
            "extent": {"x": length / 2000.0, "y": width / 2000.0, "z": height / 2},
            "orientation": {"w": math.cos(yaw / 2), "x": 0.0, "y": 0.0, "z": math.sin(yaw / 2)},
        },
        "support_z": support,
        "long_mm": float(length),
        "short_mm": float(width),
        "height_mm": height * 1000.0,
        "n_points": len(use),
    }


def model_obb(pose: dict[str, Any], mesh: Any) -> dict[str, Any] | None:
    """An upright OrientedBoundingBox (yaw only, as footprint() makes) from a registered pose and its
    primitive's edge lengths (box / cube / cylinder): the model axis nearest vertical is the height,
    the longer of the other two sets the yaw. None for a mesh file (its size is not known here)."""
    if isinstance(mesh, dict) and "box" in mesh:
        size = [float(s) for s in mesh["box"]]
    elif isinstance(mesh, dict) and "cube" in mesh:
        size = [float(mesh["cube"])] * 3
    elif isinstance(mesh, dict) and "cylinder" in mesh:
        r, h = (float(s) for s in mesh["cylinder"])
        size = [2 * r, 2 * r, h]
    else:
        return None
    R = _quat_matrix(pose["rotation"])
    up = int(np.argmax(np.abs(R[2, :])))
    flat = sorted((i for i in range(3) if i != up), key=lambda i: -size[i])
    axis = R[:2, flat[0]]
    yaw = (math.atan2(axis[1], axis[0]) + math.pi / 2) % math.pi - math.pi / 2
    return {
        "center": {k: float(pose["position"][k]) for k in "xyz"},
        "extent": {"x": size[flat[0]] / 2, "y": size[flat[1]] / 2, "z": size[up] / 2},
        "orientation": {"w": math.cos(yaw / 2), "x": 0.0, "y": 0.0, "z": math.sin(yaw / 2)},
    }


def fit_models(
    ctx: NodeContext,
    frame: dict[str, Any],
    mask: np.ndarray,
    support_z: float,
    models: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, str]:
    """pose6d.register_mask of every known part model on the part's mask; the best ``registered``
    fit by fitness. Returns (model name, its result + obb, note)."""
    cam = _camera(frame)
    d = _depth(frame)
    if cam is None or d is None:
        return "", None, "pose6d skipped: no depth or camera pose"
    K, R, t = cam
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    best: tuple[float, str, dict[str, Any]] | None = None
    seen = []
    for name, mesh in models.items():
        try:
            r = ctx.tool(
                "pose6d.register_mask",
                mask=mask.astype(np.uint8),
                depth=d,
                intrinsics=K,
                mesh=mesh,
                camera_pose=T,
                up=[0.0, 0.0, 1.0],
                support_plane=[0.0, 0.0, 1.0, -float(support_z)],
            )
        except Exception as exc:  # no pose6d server: keep the depth box
            return "", None, f"pose6d.register_mask failed ({exc})"
        status, fit = str(r.get("status", "")), float(r.get("fitness") or 0.0)
        seen.append(f"{name} {status} {fit:.2f}")
        if status == "registered" and (best is None or fit > best[0]):
            best = (fit, str(name), r)
    if best is None:
        return "", None, "pose6d: no model registered (" + ", ".join(seen) + ")"
    _, name, r = best
    obb = model_obb(r["pose"], models[name])
    return name, {**r, "obb": obb}, f"pose6d: {name} (" + ", ".join(seen) + ")"


def brickognize(crop: np.ndarray, timeout_s: float) -> dict[str, Any]:
    """lego-sorter's Brickognize predict: the padded crop as JPEG to the parts endpoint; the best
    item's id, category and score."""
    import cv2

    ok, jpeg = cv2.imencode(
        ".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="query_image"; filename="frame.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        + jpeg.tobytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        BRICKOGNIZE_URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - fixed https URL
        data = json.loads(resp.read().decode())
    items = data.get("items") or []
    if not items:
        return {"part_id": "", "category": "", "score": 0.0}
    best = items[0]
    return {
        "part_id": str(best.get("id", "")),
        "category": str(best.get("category") or best.get("type") or ""),
        "score": float(best.get("score", 0.0)),
    }


# ---------------------------------------------------------------- routing


def slug(text: str) -> str:
    """lego-sorter's bin-label normalisation ("Brick, Modified" -> "brick_modified")."""
    return str(text).strip().lower().replace(",", "").replace(" ", "_")


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    return [slug(x) for x in (v if isinstance(v, (list, tuple)) else [v])]


def size_category(
    long_mm: float, short_mm: float, height_mm: float, classes: list[dict[str, Any]]
) -> str:
    """The first size class whose bounds hold (``max_long_mm``, ``min_long_mm``, ``max_short_mm``,
    ``min_short_mm``, ``max_height_mm``, ``min_height_mm``)."""
    for c in classes or []:
        if _within(c, long_mm, short_mm, height_mm):
            return slug(c.get("name", ""))
    return ""


def _within(c: dict[str, Any], long_mm: float, short_mm: float, height_mm: float) -> bool:
    for key, val in (("long", long_mm), ("short", short_mm), ("height", height_mm)):
        lo, hi = c.get(f"min_{key}_mm"), c.get(f"max_{key}_mm")
        if (lo is not None or hi is not None) and val < 0:
            return False
        if lo is not None and val < float(lo):
            return False
        if hi is not None and val > float(hi):
            return False
    return True


def rule_bin(part: dict[str, Any], rules: list[dict[str, Any]]) -> tuple[str, str]:
    """The first rule every given key of which holds: (bin name, which rule)."""
    for i, r in enumerate(rules):
        if r.get("colour") is not None and part["colour"] not in _as_list(r["colour"]):
            continue
        if r.get("category") is not None and part["category"] not in _as_list(r["category"]):
            continue
        if r.get("part_id") is not None and part["part_id"] not in _as_list(r["part_id"]):
            continue
        if not _within(r, part["long_mm"], part["short_mm"], part["height_mm"]):
            continue
        return str(r["bin"]), f"rule {i}"
    return "", ""


def label_bin(part: dict[str, Any], bins: list[dict[str, Any]]) -> tuple[str, str, str]:
    """lego-sorter's virtual_bin priority over the bins' labels: the colour, then the part id, then
    the category, then "unknown"; (bin name, label, matched_by) of the first label a bin accepts."""
    for by in ("colour", "part_id", "category"):
        label = part[by]
        if not label or label == "unknown":
            continue
        for b in bins:
            if label in _accepts(b):
                return str(b["name"]), label, by
    for b in bins:
        if "unknown" in _accepts(b):
            return str(b["name"]), "unknown", "unknown"
    return "", "unknown", ""


def _accepts(b: dict[str, Any]) -> list[str]:
    return _as_list(b.get("accepts")) or [slug(b["name"])]


def choose_bin(
    part: dict[str, Any],
    bins: list[dict[str, Any]],
    rules: list[dict[str, Any]] | None,
    counts: dict[str, int],
) -> dict[str, Any]:
    """The rule's (or label's) bin, then capacity: a full bin sends the part to the first overflow
    bin with room. Returns {route, bin, label, matched_by, detail}."""
    name, label, by = "", "", ""
    if rules:
        name, by = rule_bin(part, rules)
        label = name
    if not name:
        name, label, by = label_bin(part, bins)
    if not name:
        return {"route": "unknown_part", "bin": None, "label": label, "matched_by": "",
                "detail": "no rule or bin label matches the part and no bin accepts 'unknown'"}  # fmt: skip
    by_name = {str(b["name"]): b for b in bins}
    if name not in by_name:
        return {"route": "unknown_part", "bin": None, "label": label, "matched_by": by,
                "detail": f"the matching rule names bin {name!r}, which is not in bins"}  # fmt: skip

    def full(b: dict[str, Any]) -> bool:
        cap = b.get("capacity")
        return cap is not None and int(counts.get(str(b["name"]), 0)) >= int(cap)

    chosen = by_name[name]
    if full(chosen):
        spare = [b for b in bins if b.get("overflow") and str(b["name"]) != name and not full(b)]
        if not spare:
            return {"route": "bin_full", "bin": chosen, "label": label, "matched_by": by,
                    "detail": f"bin {name!r} holds {counts.get(name, 0)} of {chosen.get('capacity')} "
                              "and no overflow bin has room"}  # fmt: skip
        return {"route": "routed", "bin": spare[0], "label": label, "matched_by": f"{by}, overflow",
                "detail": f"bin {name!r} is full ({counts.get(name, 0)}/{chosen.get('capacity')}); "
                          f"overflow to {spare[0]['name']!r}"}  # fmt: skip
    return {"route": "routed", "bin": chosen, "label": label, "matched_by": by, "detail": ""}


def drop_position(b: dict[str, Any]) -> dict[str, float]:
    p = b.get("drop_position") or b.get("center") or {}
    return {k: float(p.get(k, 0.0)) for k in "xyz"}


# ---------------------------------------------------------------- the node


def run(
    ctx: NodeContext,
    reference: Any,
    bins: list[dict[str, Any]] | None = None,
    staging_zone: dict[str, Any] | None = None,
    staging_roi: dict[str, float] | None = None,
    rules: list[dict[str, Any]] | None = None,
    size_classes: list[dict[str, Any]] | None = None,
    bin_counts: dict[str, int] | None = None,
    camera_name: str = "",
    palette: Any = "lego_basic",
    white_ref: Any = None,
    accept_ambiguous: bool = False,
    part_id: str = "",
    categories: dict[str, str] | None = None,
    use_brickognize: bool = False,
    brickognize_timeout_s: float = 10.0,
    part_models: dict[str, Any] | None = None,
    wait: bool = True,
    max_wait_s: float = MAX_WAIT_S,
    poll_s: float = POLL_S,
    settle_s: float = SETTLE_S,
    occupied_threshold: float = OCCUPIED_THRESHOLD,
    occupied_frames: int = OCCUPIED_FRAMES,
    motion_threshold: float = MOTION_THRESHOLD,
) -> Output:
    counts = {str(k): int(v) for k, v in (bin_counts or {}).items()}
    out: Output = {
        "route": "staging_empty",
        "detail": "",
        "bin_name": "",
        "sort_bin": {},
        "drop_position": {},
        "label": "",
        "matched_by": "",
        "part_colour": "",
        "colour_status": "",
        "colour_delta_e": -1.0,
        "category": "",
        "part_id": slug(part_id) if part_id else "",
        "part_obb": {},
        "support_z": 0.0,
        "long_mm": -1.0,
        "short_mm": -1.0,
        "height_mm": -1.0,
        "occupancy": 0.0,
        "part_pixel": [],
        "edge_override": False,
        "pose6d_model": "",
        "pose6d_fitness": -1.0,
        "bin_counts": counts,
    }
    notes: list[str] = []

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, "; ".join(d for d in (detail, *notes) if d)
        return out

    if not bins:
        return finish("unknown_part", "no bins given")
    ref = _frame(reference, camera_name)
    if ref is None:
        return finish("staging_empty", f"no reference frame (camera {camera_name or '(any)'!r})")

    def grab() -> dict[str, Any] | None:
        return _frame(ctx.tool("robot.get_observation"), camera_name)

    first = grab()
    if first is None:
        return finish("staging_empty", f"no camera frame (camera {camera_name or '(any)'!r})")
    shape = _rgb(first).shape[:2]
    try:
        zone, poly = zone_rect(shape, ref, staging_zone, staging_roi)
        window = Window(ref, zone, poly)
    except ValueError as exc:
        return finish("staging_empty", str(exc))

    # ADVANCING -> SETTLING -> CAPTURED (lego-sorter's SortCycle, the belt replaced by waiting).
    deadline = time.monotonic() + float(max_wait_s)
    state, streak, rejects, edge_rejects = "waiting", 0, 0, 0
    prev: np.ndarray | None = None
    stable_since: float | None = None
    frame, captured = first, None
    whole = False
    last_reject = ""
    while True:
        grey = window.grey(frame)
        changed = window.changed(frame, grey)
        occ = window.occupancy(changed)
        out["occupancy"] = occ
        now = time.monotonic()
        if state == "waiting":
            streak = streak + 1 if occ >= float(occupied_threshold) else 0
            if streak >= int(occupied_frames):
                state, prev, stable_since = "settling", None, None
        if state == "settling":
            if occ < float(occupied_threshold):
                state, streak = "waiting", 0
                notes.append(f"false trigger (occupancy {occ:.2f} %)")
            else:
                motion = window.motion(prev, grey) if prev is not None else float("inf")
                prev = grey
                if motion > float(motion_threshold):
                    stable_since = None
                else:
                    stable_since = now if stable_since is None else stable_since
                    if now - stable_since >= float(settle_s):
                        reject = capture_reject(window, changed, allow_edge=whole)
                        rect = None
                        if reject is None:
                            blob = window.capture_blob(changed, whole)
                            if blob is None:
                                reject = "empty crop"
                            else:
                                rect = window.to_full(blob[0], CROP_PAD_FRAC, shape)
                                x0, y0, x1, y1 = rect
                                if crop_looks_empty(_rgb(frame)[y0:y1, x0:x1]):
                                    reject = "empty crop"
                        if reject is None:
                            captured = (frame, changed, blob, rect, window.gain)
                            break
                        last_reject = reject
                        if reject == "empty crop":
                            state, streak = "waiting", 0
                            notes.append("empty-crop ghost ignored")
                        elif reject.startswith("part touching"):
                            edge_rejects += 1
                            if edge_rejects >= CAPTURE_GATE_MAX_REJECTS:
                                whole = True
                            stable_since, prev = None, None
                        else:
                            rejects += 1
                            if rejects >= CAPTURE_GATE_MAX_REJECTS:
                                return finish(
                                    "staging_not_clear",
                                    f"capture rejected {rejects} times: {reject} (occupancy {occ:.2f} %)",
                                )
                            stable_since, prev = None, None
        if not wait and state == "waiting":
            return finish("staging_empty", f"nothing on the staging zone (occupancy {occ:.2f} %)")
        if now >= deadline:
            if state == "waiting":
                return finish(
                    "staging_empty",
                    f"no part landed in {max_wait_s:.1f} s (occupancy {occ:.2f} % < {occupied_threshold:.2f} %)",
                )
            return finish(
                "staging_not_clear",
                f"the part did not settle in {max_wait_s:.1f} s"
                + (f" (last capture reject: {last_reject})" if last_reject else ""),
            )
        if poll_s > 0:
            time.sleep(float(poll_s))
        nxt = grab()
        frame = nxt if nxt is not None else frame

    frame, _changed, (_bb, blob_mask), rect, gain = captured
    out["edge_override"] = whole
    x0, y0, x1, y1 = rect
    import cv2

    wx0, wy0, wx1, wy1 = window.full
    seed_full = cv2.resize(
        blob_mask.astype(np.uint8), (wx1 - wx0, wy1 - wy0), interpolation=cv2.INTER_NEAREST
    )
    seed = np.zeros(shape, bool)
    seed[wy0:wy1, wx0:wx1] = seed_full.astype(bool)
    local = part_mask(ref, frame, rect, seed[y0:y1, x0:x1], gain)
    mask = np.zeros(shape, bool)
    mask[y0:y1, x0:x1] = local
    if not mask.any():
        mask = seed
    ys, xs = np.nonzero(mask)
    out["part_pixel"] = [float(xs.mean()), float(ys.mean())]

    # Colour (colour.classify), footprint (depth), id (optional).
    colour = ""
    try:
        cls = ctx.tool(
            "colour.classify",
            rgb=_rgb(frame),
            mask=mask.astype(np.uint8),
            palette=palette,
            **({"white_ref": white_ref} if white_ref is not None else {}),
        )
        status = str(cls.get("status", ""))
        out["colour_status"] = status
        out["colour_delta_e"] = float(cls.get("delta_e") or -1.0)
        name = slug(cls.get("name") or "")
        if status == "classified" or (status == "ambiguous" and accept_ambiguous):
            colour = name
        elif status == "ambiguous":
            notes.append(f"colour ambiguous: {cls.get('name')} or {cls.get('second_name')}")
    except Exception as exc:  # no colour tool: route on the other labels
        out["colour_status"] = "error"
        notes.append(f"colour.classify failed ({exc})")
    out["part_colour"] = colour or "unknown"

    fp = footprint(ref, frame, mask, zone)
    if fp is not None:
        out["part_obb"], out["support_z"] = fp["obb"], fp["support_z"]
        out["long_mm"], out["short_mm"], out["height_mm"] = (
            fp["long_mm"],
            fp["short_mm"],
            fp["height_mm"],
        )
    else:
        notes.append("no depth or camera pose: no part box for the pick")
    pid = out["part_id"]
    if part_models and fp is not None:
        model, fit, note = fit_models(ctx, frame, mask, fp["support_z"], part_models)
        notes.append(note)
        if fit is not None:
            out["pose6d_model"], out["pose6d_fitness"] = model, float(fit.get("fitness") or 0.0)
            if fit["obb"] is not None:
                out["part_obb"] = fit["obb"]
            pid = pid or slug(model)

    category = ""
    if use_brickognize and not pid:
        try:
            got = brickognize(_rgb(frame)[y0:y1, x0:x1], float(brickognize_timeout_s))
            if got["score"] >= BRICKOGNIZE_MIN_SCORE:
                pid, category = slug(got["part_id"]), slug(got["category"])
            else:
                notes.append(f"Brickognize unsure ({got['part_id']} at {got['score']:.2f})")
        except Exception as exc:  # network, rate limit or terms: route without an id
            notes.append(f"Brickognize unavailable ({exc})")
    if pid and categories and not category:
        category = slug(categories.get(pid, ""))
    if not category and size_classes:
        category = size_category(out["long_mm"], out["short_mm"], out["height_mm"], size_classes)
    out["part_id"], out["category"] = pid, category

    part = {
        "colour": out["part_colour"],
        "category": category,
        "part_id": pid,
        "long_mm": out["long_mm"],
        "short_mm": out["short_mm"],
        "height_mm": out["height_mm"],
    }
    pick = choose_bin(part, bins, rules, counts)
    out["label"], out["matched_by"] = pick["label"], pick["matched_by"]
    seen = (
        f"part {out['part_colour']} ({out['colour_status']}), "
        + (
            f"{out['long_mm']:.0f} x {out['short_mm']:.0f} x {out['height_mm']:.0f} mm, "
            if fp
            else ""
        )
        + (f"category {category}, " if category else "")
        + (f"id {pid}, " if pid else "")
        + f"occupancy {out['occupancy']:.2f} %"
    )
    if pick["bin"] is not None:
        out["bin_name"] = str(pick["bin"]["name"])
        out["sort_bin"] = dict(pick["bin"])
        out["drop_position"] = drop_position(pick["bin"])
    if pick["route"] != "routed":
        return finish(pick["route"], f"{pick['detail']}; {seen}")
    return finish(
        "routed",
        f"bin {out['bin_name']!r} by {pick['matched_by']} ({pick['label']}); {seen}"
        + (f"; {pick['detail']}" if pick["detail"] else ""),
    )
