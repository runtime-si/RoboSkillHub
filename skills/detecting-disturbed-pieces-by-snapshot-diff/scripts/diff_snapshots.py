# Adapted from scikit-image (https://github.com/scikit-image/scikit-image,
# src/skimage/metrics/_structural_similarity.py, structural_similarity with
# gaussian_weights=True and full=True, commit e098ebbb684788df60b42e3bde112c014d0d9517),
# BSD-3-Clause, Copyright (c) 2009-2022 the scikit-image team; licence in
# LICENSE.scikit-image. Changes: one grey float32 channel; OpenCV's GaussianBlur (sigma 1.5,
# 11 taps, reflected border) in place of scipy's gaussian filter with truncate 3.5, which is
# the same 11-tap window; only the full SSIM map is returned (no mean, no gradient).
# Alignment calls OpenCV's findTransformECC (https://github.com/opencv/opencv,
# modules/video/src/ecc.cpp, Apache-2.0) with a mask; nothing of it is copied.
# Everything else (registration masks and pyramid, lighting gain field, depth drift
# plane, per-item local registration, occluder and blob classification, routing) is
# written for this skill.
"""Which known items moved between a reference snapshot and now: register, diff, explain.

The reference snapshot is taken after the last verified step; the current one now. Both are
RGB(-D) frames of the same camera. The current frame is registered onto the reference on the
static background (ORB + RANSAC homography, refined by masked ECC), lighting drift is divided
out with a smooth gain field, depth drift with a fitted plane, and every pixel gets an SSIM,
colour and depth change. Each known item (mask, pixel polygon, world box or grid cell) is then
re-found in the current frame by a local rigid registration (masked NCC, then Euclidean ECC):
its translation (mm when depth, a camera pose or a scale is known), rotation and depth change
say whether it moved. Changed pixels no item explains are blobs: an occluder (a hand or the
arm, touching the frame edge or the arm, nearer than the reference) or an unexpected object.
"""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

MAX_WORK_WIDTH = 960
"""Frames wider than this are shrunk before anything else [px]; outputs are in the input's
pixels. 640 x 480 station frames run at full size."""

# --- registration -----------------------------------------------------------------------------

ITEM_EXCLUDE_PX = 6
"""Items are dilated by this before they are taken out of the registration mask [px]: their
edges move with them."""

MIN_STATIC_FRACTION = 0.10
"""If the background (frame minus items, ignore and arm masks) is under this fraction of the
frame, the items are registered on too (a board covered with pieces): most of them did not
move, and RANSAC and the second ECC pass drop the ones that did."""

ORB_FEATURES = 3000
ORB_MIN_INLIERS = 25
ORB_RANSAC_PX = 3.0
"""ORB + RANSAC homography initialises ECC when it finds at least this many inliers within
3 px; otherwise the camera-pose prior (or identity) does. 640 x 480 station frames give
400-1500 inliers between consecutive frames."""

ECC_LEVELS = (0.25, 0.5, 1.0)
ECC_ITERS = (80, 50, 30)
ECC_EPS = 1e-5
"""Masked homography ECC, coarse to fine. At a quarter scale a few-pixel bump of the camera
is under a pixel, so ECC converges from identity when ORB is not available."""

HIGHPASS_PRE = 1.0
HIGHPASS_SIGMA = 8.0
HIGHPASS_EPS = 10.0
"""Registration runs on grey (blurred by 1 px) minus its 8 px Gaussian blur, divided by the
local standard deviation + 10 grey levels: a lamp's gradient or a white-balance change does
not pull ECC, and flat regions' sensor noise is not blown up. On the synthetic boards this
took the residual camera shift under lighting drift from 5 px to 0.05 px; with 5 px and +3 the
registered background correlated at only 0.75 between identical noisy frames."""

NEAR_REGION_PX = 60
MIN_NEAR_PX = 3000
"""The second ECC pass uses the background within 60 px of the work region (when it has at
least 3000 pixels): the residual of a whole-frame homography (lens distortion, a table that is
not flat) is a pixel or two, i.e. 2-4 mm at the overhead camera."""

MIN_RHO = 0.80
"""Masked correlation of the registered background below which the pair is
``unregistrable`` (another scene, a covered lens, a big camera move). Station frames of one
scene register at 0.993-0.999; the synthetic boards at 0.90-0.995 (flat wrist views are the
low ones); wrist views mostly covered by a hand fell to 0.05-0.78."""

RHO_TILES = (6, 8)
RHO_TILE_MIN_PX = 300
"""The background correlation is the median over 6 x 8 tiles with at least 300 background
pixels each."""

MAX_CAMERA_SHIFT_FRACTION = 0.08
"""Largest corner displacement of the registration, as a fraction of the frame diagonal,
still accepted (64 px at 640 x 480). Beyond it the views differ too much for a pixel diff
(parallax, new occlusions): ``unregistrable``; re-take the reference."""

# --- change map -------------------------------------------------------------------------------

GAIN_SIGMA_FRACTION = 0.08
"""Lighting drift is divided out by a per-channel gain field: the ratio of the two frames'
masked Gaussian blurs with sigma = this x frame width (51 px at 640). Wider than any item, so
a moved item barely changes it; narrow enough to follow a lamp's gradient and a soft
shadow."""

GAIN_CLIP = (0.33, 3.0)

GAIN_LOCAL = 0.2
"""The local gain may depart from the quadratic by +-20 %: a soft shadow of the station's
lamps (the synthetic nuisance darkens by 10-30 %, so the deepest are partly left as change),
but not a hand or a sheet over the board, which differ by more."""

SSIM_SIGMA = 1.5
SSIM_K1, SSIM_K2 = 0.01, 0.03
"""Wang et al. 2004 and scikit-image's defaults for gaussian_weights=True."""

T_DSSIM = 0.35
T_COLOUR_MIN = 14.0
T_COLOUR = 40.0
"""A pixel changed in appearance if 1 - SSIM > 0.35 and its gain-corrected colour moved by
over 14 (0-255, largest channel), or its colour moved by over 40 alone. On 20 consecutive
station frame pairs no pixel of the table's work area survived the 3 x 3 opening; a 2 mm
shift of a synthetic piece marks its edges."""

DEPTH_CHANGE_M = 0.006
DEPTH_CHANGE_REL = 0.012
"""A pixel changed in depth if it moved by over max(6 mm, 1.2 % of its range): the Orbbec
at 0.9 m has 2-4 mm of noise per pixel, the D405 at 0.2 m under 1 mm."""

DEPTH_DRIFT_INIT_M = 0.010
"""The depth drift plane starts from the background's median change and keeps what is within
10 mm of it (a camera bump moves the background a few mm), before least squares."""

DEPTH_MEDIAN_PX = 5
MIN_BLOB_PX = 30
"""Changed pixels are opened by a 3 x 3 square; components under this many pixels are
dropped (a 9 x 9 mm patch is about 17 px at the overhead camera, 200 px at the wrist)."""

MIN_BLOB_MM2 = 80.0
"""Smallest unexplained change reported as an unexpected object when the scale is known
[mm^2]: a 9 x 9 mm patch, under a quarter of a 20 mm jigsaw piece, a LEGO 1x1 (8 x 8 mm)
being the smallest item in the families."""

BORDER_PX = 4
"""A changed blob within this of the frame edge (plus the camera shift) entered from outside
the view: an occluder (hand, arm, tool), not an object set down."""

OCCLUDER_NEARER_M = 0.025
"""With depth, an occluder must also be at least this much nearer the camera than the
reference surface (median) - a hand or the arm hovers, an object pushed in from the edge does
not."""

ARM_RADIUS_M = 0.05
ARM_HAND_FORWARD_M = 0.03
ARM_HAND_BACK_M = 0.15
"""The arm's mask is a chain of balls of this radius from 3 cm past the tool centre along the
approach axis back 15 cm (the YAM hand and wrist: 9 cm fingers, 4 cm wrist motor), then on
to the arm base when ``arm_base_positions`` gives it."""

# --- items ------------------------------------------------------------------------------------

EDGE_PX = 3
"""Each item's template is its mask dilated by this [px]: a flat-topped item (white domino,
uniform brick) is registered on its outline."""

SEARCH_RADIUS_MM = 25.0
SEARCH_RADIUS_PX = 24
MAX_SEARCH_PX = 60
"""How far an item is searched for: 25 mm (a bump pushes a piece a few mm, a slide of a
section rarely over a piece width) at the item's scale, or 24 px when no scale is known,
never over 60 px."""

DISTANCE_PENALTY = 0.08
"""The coarse search's score is lowered by this times the displacement over the search
radius: of two equally good places (a row of identical dominoes 20 mm apart, a repeating
brick colour), the nearer is the item."""

CANDIDATE_BAND = 0.3
MAX_CANDIDATES = 3
"""Up to 3 local peaks of the coarse map within 0.3 of its best are refined by ECC and the
best refined match (less the distance penalty) wins: at whole-pixel steps a thin item (a
standing domino is 4 px wide at the overhead camera) half a pixel off scores well below an
identical neighbour that happens to sit on the grid."""

OVERLAP_PENALTY = 0.15
"""A candidate is also lowered by this times the fraction of the moved item that would lie on
another item's reference footprint: a bumped domino that now stands against its neighbour
looks less like itself (its background ring is gone) than the untouched neighbour does."""

COARSE_ANGLES_DEG = (-16.0, -8.0, 8.0, 16.0)
COARSE_GOOD = 0.9
"""Rotations tried for the coarse placement when the unrotated template scores under
0.9; ECC refines from the best."""

MIN_VISIBLE = 0.6
"""An item with less of its mask visible (the rest under the arm, an occluder, an ignore
mask or outside the registered frame) is ``occluded``: not judged."""

MIN_MATCH = 0.55
LOST_CHANGED = 0.25
"""An item whose best local match correlates under 0.55 and whose mask changed by over 25 %
is ``lost``: not where it was and not nearby as it looked (toppled, flipped, removed, pushed
past the search radius). Unchanged synthetic items correlate 0.9-1.0."""

MIN_SHIFT_MM = 1.0
MIN_SHIFT_NOISE_PX = 0.6
MIN_SHIFT_PX = 1.0
MIN_ROTATION_DEG = 2.0
"""An item moved when its local registration translates it by at least max(1 mm, 0.6 px at
its scale) - 1.3 mm at the overhead camera (2.2 mm per px at 1 m), 1 mm at the wrist - or by
1 px when no scale is known, or turns it by 2 degrees (but see MIN_ROTATION_ARC_PX).
Unmoved items on the synthetic boards (lighting drift and camera bumps) read a median 0.1 px
and a 99th percentile of 0.35-0.47 px overhead, 0.2-0.77 px at the wrist (the high ones are
neighbours of a moved item)."""

MIN_ROTATION_ARC_PX = 0.8
"""...and only if that turn moves the item's rim (sqrt(2) x its RMS radius) by 0.8 px: a 25 mm
jigsaw piece is 11 px across at the overhead camera, where its turn reads +-5 degrees on an
unchanged board; it must turn about 8 degrees to count there, 2 degrees at the wrist camera."""

MIN_SIGNIFICANCE = 100.0
NOISE_FLOOR = 1.0
MIN_TEXTURE = 2.0
"""A move also has to stand out of the noise given the item's texture: p' J'J p / (2 sigma^2)
>= 100, with p = (dx, dy, turn) and J the template's gradients and their rotational moments
(the Gauss-Newton information of the local registration), sigma the grey-level noise of the
registered frames (1.4826 MAD over the unchanged background, at least 1). And the template's
weakest gradient direction must carry a mean squared gradient of at least 2 (grey levels per
px, squared; sensor noise alone gives about 0.2): below it (white cloth, plain wood across
the grain) the move cannot be measured and only the change map, depth and match judge the
item. On unchanged station pairs, 50 mm table cells read up to 28 mm on a white tablecloth
and 2.4 mm across wood grain before these two gates, nothing after."""

MIN_DEPTH_CHANGE_M = 0.005
"""An item also moved if the median depth over its mask changed by this much (lifted,
pressed down, tipped, toppled)."""

MIN_DEPTH_PX = 20


class Output(TypedDict):
    route: str
    items: list[dict[str, Any]]
    disturbed: list[str]
    occluded: list[str]
    unexpected: list[dict[str, Any]]
    occluders: list[dict[str, Any]]
    n_disturbed: int
    n_unexpected: int
    max_shift_mm: float
    camera_shift_px: float
    registration: dict[str, Any]
    changed_fraction: float
    detail: str


# --- frames and geometry ----------------------------------------------------------------------


def _pose_matrix(pose: Any) -> np.ndarray | None:
    if pose is None:
        return None
    if isinstance(pose, dict):
        p, q = pose["position"], pose["rotation"]
        w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
        n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
        w, x, y, z = w / n, x / n, y / n, z / n
        m = np.eye(4)
        m[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
        m[:3, 3] = [float(p[k]) for k in ("x", "y", "z")]
        return m
    m = np.asarray(pose, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError("a camera pose is an Se3Pose or a 4x4 camera-to-world matrix")
    return m


def _rgb(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"rgb must be H x W x 3, got {arr.shape}")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _frame(snapshot: Any, camera_name: str) -> dict[str, Any]:
    """A CameraFrame, an Observation (its camera ``camera_name``) or a bare RGB array."""
    if snapshot is None:
        raise ValueError("no snapshot")
    arms: list[dict[str, Any]] = []
    cam = snapshot
    if isinstance(snapshot, dict) and "cameras" in snapshot:
        arms = list(snapshot.get("arms") or [])
        cams = snapshot.get("cameras") or []
        if isinstance(cams, dict):
            cams = [{"name": k, **v} for k, v in cams.items()]
        cam = None
        for c in cams:
            if camera_name and c.get("name") == camera_name:
                cam = c
        if cam is None:
            if camera_name and len(cams) != 1:
                raise ValueError(
                    f"observation has no camera {camera_name!r} (has {[c.get('name') for c in cams]})"
                )
            if not cams:
                raise ValueError("observation has no cameras")
            cam = cams[0]
    if not isinstance(cam, dict):
        cam = {"rgb": cam}
    rgb = _rgb(cam["rgb"])
    depth = cam.get("depth")
    if depth is not None:
        depth = np.asarray(depth, dtype=np.float32).reshape(rgb.shape[:2]).copy()
        depth[~np.isfinite(depth) | (depth < 0.05) | (depth > 10.0)] = np.nan
        if not np.isfinite(depth).any():
            depth = None
    k = cam.get("intrinsics")
    k = None if k is None else np.asarray(k, dtype=np.float64).reshape(3, 3).copy()
    return {
        "name": cam.get("name", camera_name),
        "rgb": rgb,
        "depth": depth,
        "K": k,
        "T": _pose_matrix(cam.get("pose")),
        "arms": arms,
    }


def _scale_frame(f: dict[str, Any], s: float) -> dict[str, Any]:
    if s == 1.0:
        return f
    h, w = f["rgb"].shape[:2]
    size = (int(round(w * s)), int(round(h * s)))
    out = dict(f)
    out["rgb"] = cv2.resize(f["rgb"], size, interpolation=cv2.INTER_AREA)
    if f["depth"] is not None:
        out["depth"] = cv2.resize(f["depth"], size, interpolation=cv2.INTER_NEAREST)
    if f["K"] is not None:
        k = f["K"].copy()
        k[:2] *= s
        out["K"] = k
    return out


def _mask(m: Any, shape: tuple[int, int]) -> np.ndarray:
    a = np.asarray(m)
    if a.ndim > 2:
        a = a.reshape(a.shape[-2:])
    a = a > 0
    if a.shape != shape:
        a = cv2.resize(a.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        a = a > 0
    return a


def _project(k: np.ndarray, t_cam: np.ndarray, pts_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv = np.linalg.inv(t_cam)
    pc = pts_w @ inv[:3, :3].T + inv[:3, 3]
    z = pc[:, 2]
    uv = (pc[:, :2] / np.maximum(z, 1e-6)[:, None]) * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    return uv, z


def _ray(k: np.ndarray, u: Any, v: Any) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], axis=-1)


def _on_plane(k: np.ndarray, plane: tuple[np.ndarray, float], u: Any, v: Any) -> np.ndarray:
    """Camera-frame points where pixel rays meet the plane n . X = d."""
    n, d = plane
    r = _ray(k, u, v)
    den = r @ n
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    return r * (d / den)[..., None]


def _fill_polygon(pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    if len(pts) >= 3 and np.isfinite(pts).all():
        hull = cv2.convexHull(np.round(pts).astype(np.int32))
        cv2.fillConvexPoly(out, hull, 1)
    return out > 0


def _box_corners(box: dict[str, Any]) -> np.ndarray:
    c = np.array([box["center"][a] for a in "xyz"], dtype=np.float64)
    e = np.array([box["extent"][a] for a in "xyz"], dtype=np.float64)
    r = _pose_matrix(
        {
            "position": {"x": 0, "y": 0, "z": 0},
            "rotation": box.get("orientation") or {"w": 1, "x": 0, "y": 0, "z": 0},
        }
    )[:3, :3]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return c + (signs * e) @ r.T


def _quat_wxyz(r: np.ndarray) -> dict[str, float]:
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    if w > 1e-6:
        x, y, z = (
            (r[2, 1] - r[1, 2]) / (4 * w),
            (r[0, 2] - r[2, 0]) / (4 * w),
            (r[1, 0] - r[0, 1]) / (4 * w),
        )
    else:
        i = int(np.argmax(np.diag(r)))
        j, kk = (i + 1) % 3, (i + 2) % 3
        q = np.zeros(4)
        q[i + 1] = math.sqrt(max(0.0, 1.0 + r[i, i] - r[j, j] - r[kk, kk])) / 2.0
        q[j + 1] = (r[j, i] + r[i, j]) / (4 * q[i + 1])
        q[kk + 1] = (r[kk, i] + r[i, kk]) / (4 * q[i + 1])
        q[0] = (r[kk, j] - r[j, kk]) / (4 * q[i + 1])
        w, x, y, z = q
    return {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}


def _cell_box(grid: dict[str, Any], col: int, row: int) -> dict[str, Any]:
    t = _pose_matrix(grid["origin"])
    size = grid.get("cell_size", grid.get("pitch"))
    cw, ch = (float(size), float(size)) if np.isscalar(size) else (float(size[0]), float(size[1]))
    h = float(grid.get("item_height", 0.004))
    c = t[:3, :3] @ np.array([(col + 0.5) * cw, (row + 0.5) * ch, h / 2]) + t[:3, 3]
    return {
        "center": dict(zip("xyz", map(float, c), strict=True)),
        "extent": {"x": cw / 2, "y": ch / 2, "z": h / 2},
        "orientation": _quat_wxyz(t[:3, :3]),
    }


def _cell_polygon_px(grid: dict[str, Any], col: int, row: int) -> np.ndarray:
    corners = np.asarray(grid["corners_px"], dtype=np.float32).reshape(4, 2)
    cols, rows = int(grid["cols"]), int(grid["rows"])
    unit = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    m = cv2.getPerspectiveTransform(unit, corners)
    q = np.float32([[col, row], [col + 1, row], [col + 1, row + 1], [col, row + 1]])
    return cv2.perspectiveTransform(q[None], m)[0]


def _region_mask(region: Any, ref: dict[str, Any], shape: tuple[int, int], s: float) -> np.ndarray:
    if isinstance(region, dict) and "center" in region:
        if ref["K"] is None or ref["T"] is None:
            raise ValueError("a world-box region needs the reference camera's intrinsics and pose")
        uv, z = _project(ref["K"], ref["T"], _box_corners(region))
        return _fill_polygon(uv[z > 0], shape)
    if isinstance(region, dict) and "polygon_px" in region:
        return _fill_polygon(np.asarray(region["polygon_px"], float) * s, shape)
    return _mask(region, shape)


def _item_mask(
    item: dict[str, Any],
    ref: dict[str, Any],
    grid: dict[str, Any] | None,
    shape: tuple[int, int],
    s: float,
) -> tuple[np.ndarray, float | None]:
    """The item's mask in the (working) reference image, and its height above the support
    plane [m] when a world box says so."""
    if item.get("mask") is not None:
        return _mask(item["mask"], shape), None
    if item.get("polygon_px") is not None:
        return _fill_polygon(np.asarray(item["polygon_px"], float) * s, shape), None
    box = item.get("box")
    if box is None and item.get("cell") is not None:
        if grid is None:
            raise ValueError(f"item {item.get('id')!r} is a cell but no grid was given")
        col, row = (int(v) for v in item["cell"])
        if "corners_px" in grid:
            return _fill_polygon(_cell_polygon_px(grid, col, row) * s, shape), None
        box = _cell_box(grid, col, row)
    if box is None:
        raise ValueError(f"item {item.get('id')!r} has no mask, polygon_px, box or cell")
    if ref["K"] is None or ref["T"] is None:
        raise ValueError(
            "world-box and grid-origin items need the reference camera's intrinsics and pose"
        )
    corners = _box_corners(box)
    uv, z = _project(ref["K"], ref["T"], corners)
    top = float(corners[:, 2].max())
    return _fill_polygon(uv[z > 0], shape), top


def _arm_mask(frame: dict[str, Any], shape: tuple[int, int], bases: list[Any] | None) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    k, t = frame["K"], frame["T"]
    if k is None or t is None:
        return out > 0
    for i, arm in enumerate(frame["arms"]):
        pose = arm.get("ee_pose") if isinstance(arm, dict) else None
        if pose is None:
            continue
        m = _pose_matrix(pose)
        tcp, approach = m[:3, 3], m[:3, 2]
        wrist = tcp - ARM_HAND_BACK_M * approach
        chain = [np.linspace(tcp + ARM_HAND_FORWARD_M * approach, wrist, 10)]
        if bases is not None and i < len(bases) and bases[i] is not None:
            b = bases[i]
            b = np.array([b[a] for a in "xyz"] if isinstance(b, dict) else b, dtype=np.float64)
            n = max(2, int(np.linalg.norm(b - wrist) / 0.02))
            chain.append(np.linspace(wrist, b, n))
        pts = np.concatenate(chain)
        uv, z = _project(k, t, pts)
        for (u, v), zz in zip(uv, z, strict=True):
            if zz <= 0.02:
                continue
            r = int(math.ceil(k[0, 0] * ARM_RADIUS_M / zz))
            if -r <= u < shape[1] + r and -r <= v < shape[0] + r:
                cv2.circle(out, (int(round(u)), int(round(v))), r, 1, -1)
    return out > 0


def _fit_plane(pts: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Least-squares plane through camera-frame points, trimmed twice to 10 mm."""
    if len(pts) < 50:
        return None
    keep = np.ones(len(pts), bool)
    n, d = np.array([0.0, 0.0, 1.0]), 0.0
    for tol in (None, 0.02, 0.01):
        p = pts[keep]
        if len(p) < 30:
            return None
        c = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - c, full_matrices=False)
        n = vt[2] * (1.0 if vt[2][2] < 0 else -1.0)
        d = float(n @ c)
        res = np.abs(pts @ n - d)
        keep = res < (tol if tol is not None else max(0.03, 3 * np.median(res)))
    return n, d


# --- image measures ---------------------------------------------------------------------------


def _ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """scikit-image structural_similarity(gaussian_weights=True, full=True)'s S map."""
    win = 2 * int(3.5 * SSIM_SIGMA + 0.5) + 1
    npix = win * win
    cov_norm = npix / (npix - 1)

    def blur(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (win, win), SSIM_SIGMA, borderType=cv2.BORDER_REFLECT)

    ux, uy = blur(a), blur(b)
    vx = cov_norm * (blur(a * a) - ux * ux)
    vy = cov_norm * (blur(b * b) - uy * uy)
    vxy = cov_norm * (blur(a * b) - ux * uy)
    c1, c2 = (SSIM_K1 * 255.0) ** 2, (SSIM_K2 * 255.0) ** 2
    a1, a2 = 2 * ux * uy + c1, 2 * vxy + c2
    b1, b2 = ux**2 + uy**2 + c1, vx + vy + c2
    return (a1 * a2) / (b1 * b2)


def _highpass(g: np.ndarray) -> np.ndarray:
    """Locally normalised grey: lighting gradients and gain drop out before ECC."""
    f = cv2.GaussianBlur(g.astype(np.float32), (0, 0), HIGHPASS_PRE)
    hp = f - cv2.GaussianBlur(f, (0, 0), HIGHPASS_SIGMA)
    sd = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), HIGHPASS_SIGMA))
    return hp / (sd + HIGHPASS_EPS)


def _masked_ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    if m.sum() < 20:
        return 0.0
    x, y = a[m].astype(np.float64), b[m].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    den = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 1e-9 else 0.0


def _tile_rho(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    """Median masked correlation over a grid of tiles: a hand over a third of the view does not
    sink it, another scene sinks every tile."""
    hh, ww = m.shape
    vals = []
    for i in range(RHO_TILES[0]):
        for j in range(RHO_TILES[1]):
            sl = (
                slice(i * hh // RHO_TILES[0], (i + 1) * hh // RHO_TILES[0]),
                slice(j * ww // RHO_TILES[1], (j + 1) * ww // RHO_TILES[1]),
            )
            if m[sl].sum() >= RHO_TILE_MIN_PX:
                vals.append(_masked_ncc(a[sl], b[sl], m[sl]))
    if not vals:
        return _masked_ncc(a, b, m)
    return float(np.median(vals))


def _warp_to_ref(
    img: np.ndarray, h: np.ndarray, shape: tuple[int, int], nearest: bool = False
) -> np.ndarray:
    flags = (cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP
    border = np.nan if img.dtype == np.float32 and nearest else 0
    return cv2.warpPerspective(
        img,
        h,
        (shape[1], shape[0]),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def _orb_homography(ref_g, cur_g, ref_m, cur_m) -> tuple[np.ndarray | None, int]:
    orb = cv2.ORB_create(ORB_FEATURES)
    k1, d1 = orb.detectAndCompute(ref_g, ref_m.astype(np.uint8) * 255)
    k2, d2 = orb.detectAndCompute(cur_g, cur_m.astype(np.uint8) * 255)
    if d1 is None or d2 is None or len(k1) < ORB_MIN_INLIERS or len(k2) < ORB_MIN_INLIERS:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < ORB_MIN_INLIERS:
        return None, 0
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    h, inl = cv2.findHomography(p1, p2, cv2.RANSAC, ORB_RANSAC_PX)
    n = int(inl.sum()) if inl is not None else 0
    if h is None or n < ORB_MIN_INLIERS:
        return None, n
    return h, n


def _ecc_call(a, b, tmask, imask, warp, motion, iters, eps, filt):
    """ECC with a template mask and an input mask (findTransformECCWithMask, OpenCV >= 4.13);
    older OpenCV takes one mask, applied to the input image."""
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    tm = (tmask > 0).astype(np.uint8)
    im = (imask > 0).astype(np.uint8)
    if hasattr(cv2, "findTransformECCWithMask"):
        return cv2.findTransformECCWithMask(a, b, tm, im, warp, motion, crit, filt)
    return cv2.findTransformECC(a, b, warp, motion, crit, tm & im, filt)


def _ecc(
    ref_g, cur_g, mask, h0, cur_mask=None, levels=ECC_LEVELS, iters=ECC_ITERS
) -> tuple[np.ndarray, bool]:
    h = h0.copy()
    ok = False
    for s, it in zip(levels, iters, strict=True):
        sc = np.diag([s, s, 1.0])
        hs = sc @ h @ np.linalg.inv(sc)
        cm = np.ones(cur_g.shape, np.uint8) if cur_mask is None else cur_mask.astype(np.uint8)
        if s == 1.0:
            a, b, m = ref_g, cur_g, mask.astype(np.uint8)
        else:
            size = (max(8, int(ref_g.shape[1] * s)), max(8, int(ref_g.shape[0] * s)))
            a = cv2.resize(ref_g, size, interpolation=cv2.INTER_AREA)
            b = cv2.resize(cur_g, size, interpolation=cv2.INTER_AREA)
            m = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            cm = cv2.resize(cm, size, interpolation=cv2.INTER_NEAREST)
        try:
            _, hs = _ecc_call(
                a, b, m, cm, hs.astype(np.float32), cv2.MOTION_HOMOGRAPHY, it, ECC_EPS, 5
            )
            h = np.linalg.inv(sc) @ hs.astype(np.float64) @ sc
            ok = True
        except cv2.error:
            continue
    return h / h[2, 2], ok


def _corner_shift(h: np.ndarray, shape: tuple[int, int]) -> float:
    hh, ww = shape
    c = np.float64([[0, 0], [ww - 1, 0], [ww - 1, hh - 1], [0, hh - 1]])
    p = cv2.perspectiveTransform(c[None], h)[0]
    return float(np.linalg.norm(p - c, axis=1).max())


def _pose_prior(ref: dict[str, Any], cur: dict[str, Any], plane, shape) -> np.ndarray | None:
    """Homography (reference pixels to current pixels) the two camera poses imply for the
    support plane; None when the poses are missing or equal."""
    if ref["K"] is None or cur["K"] is None or ref["T"] is None or cur["T"] is None:
        return None
    rel = np.linalg.inv(cur["T"]) @ ref["T"]
    ang = math.degrees(math.acos(np.clip((np.trace(rel[:3, :3]) - 1) / 2, -1, 1)))
    if np.linalg.norm(rel[:3, 3]) < 5e-4 and ang < 0.05:
        return None
    hh, ww = shape
    c = np.float64([[0, 0], [ww - 1, 0], [ww - 1, hh - 1], [0, hh - 1]])
    if plane is not None:
        pc = _on_plane(ref["K"], plane, c[:, 0], c[:, 1])
    else:
        pc = _ray(ref["K"], c[:, 0], c[:, 1]) * 1.0
    q = pc @ rel[:3, :3].T + (rel[:3, 3] if plane is not None else 0.0)
    if (q[:, 2] <= 0).any():
        return None
    uv = q[:, :2] / q[:, 2:3] * [cur["K"][0, 0], cur["K"][1, 1]] + [cur["K"][0, 2], cur["K"][1, 2]]
    return cv2.getPerspectiveTransform(c.astype(np.float32), uv.astype(np.float32)).astype(
        np.float64
    )


def _gain_correct(ref: np.ndarray, cur: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    """Divide ``cur`` by a per-channel gain so it matches ``ref``'s lighting: a quadratic in the
    pixel position fitted robustly to the log ratio (a lamp's gain, tint and gradient), times a
    smooth local residual clipped to +-GAIN_LOCAL (a soft shadow). A large uniform change (a
    hand, a sheet of paper) is brighter or darker than that and stays a change."""
    hh, ww = m.shape
    mf = m.astype(np.float32)
    bm = cv2.GaussianBlur(mf, (0, 0), sigma)
    vv, uu = np.mgrid[0:hh, 0:ww]
    x, y = (uu / ww - 0.5).astype(np.float32), (vv / hh - 0.5).astype(np.float32)
    basis = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)
    step = max(1, int(math.sqrt(hh * ww / 20000)))
    out = np.empty_like(cur)
    for c in range(cur.shape[2]):
        a, b = ref[:, :, c], cur[:, :, c]
        sel = m & (a > 8) & (b > 8) & (a < 250) & (b < 250)
        sub = sel[::step, ::step]
        if sub.sum() > 50:
            lr = np.log(b[::step, ::step][sub] / a[::step, ::step][sub])
            bs = basis[::step, ::step][sub]
            keep = np.abs(lr - np.median(lr)) < 0.5
            coef = np.zeros(6)
            for _ in range(3):
                if keep.sum() < 20:
                    break
                coef, *_ = np.linalg.lstsq(bs[keep], lr[keep], rcond=None)
                r = np.abs(bs @ coef - lr)
                keep = r < max(0.02, 2.5 * 1.4826 * np.median(r[keep]))
            gq = np.exp(basis @ coef.astype(np.float32))
        else:
            gq = np.ones((hh, ww), np.float32)
        aq = a * gq
        num = cv2.GaussianBlur(b * mf, (0, 0), sigma)
        den = cv2.GaussianBlur(aq * mf, (0, 0), sigma)
        gl = np.where((bm > 0.05) & (den > 1e-3), num / np.maximum(den, 1e-3), 1.0)
        gl = np.clip(gl, 1.0 / (1.0 + GAIN_LOCAL), 1.0 + GAIN_LOCAL)
        out[:, :, c] = b / np.clip(gq * gl, *GAIN_CLIP)
    return out


def _depth_change(dr0: np.ndarray, dc0: np.ndarray, ok_mask: np.ndarray, static: np.ndarray):
    """Signed depth change [m] (NaN where unknown) less a drift plane fitted on the
    background, the pixels whose change is over max(6 mm, 1.2 % of range), and the drift."""
    hh, ww = dr0.shape
    dr = cv2.medianBlur(np.nan_to_num(dr0, nan=0.0), DEPTH_MEDIAN_PX)
    dc = cv2.medianBlur(np.nan_to_num(dc0, nan=0.0), DEPTH_MEDIAN_PX)
    ok = (dr > 0.05) & (dc > 0.05) & ok_mask
    ddz = np.where(ok, dc - dr, np.nan).astype(np.float32)
    drift = np.zeros((hh, ww), np.float32)
    bg = ok & static
    if bg.sum() > 200:
        v, u = np.nonzero(bg)
        sel = np.random.default_rng(0).choice(len(u), min(len(u), 5000), replace=False)
        a = np.stack([u[sel], v[sel], np.ones(len(sel))], 1).astype(np.float64)
        y = ddz[v[sel], u[sel]].astype(np.float64)
        # Start from the median, so a hand over a third of the background cannot tilt the fit.
        keep = np.abs(y - np.median(y)) < DEPTH_DRIFT_INIT_M
        coef = np.array([0.0, 0.0, float(np.median(y))])
        for _ in range(3):
            if keep.sum() < 50:
                break
            coef, *_ = np.linalg.lstsq(a[keep], y[keep], rcond=None)
            r = np.abs(a @ coef - y)
            keep = r < max(0.002, 3.0 * 1.4826 * np.median(r[keep]))
        vv, uu = np.mgrid[0:hh, 0:ww]
        drift = (coef[0] * uu + coef[1] * vv + coef[2]).astype(np.float32)
        ddz = ddz - drift
    thr = np.maximum(DEPTH_CHANGE_M, DEPTH_CHANGE_REL * dr)
    changed = np.isfinite(ddz) & (np.abs(np.nan_to_num(ddz)) > thr)
    changed = (
        cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    )
    return ddz, changed, drift


def _change_maps(ref_rgb, cur_rgb, valid, gain_mask, sigma):
    a = cv2.GaussianBlur(ref_rgb.astype(np.float32), (3, 3), 0)
    b = cv2.GaussianBlur(cur_rgb.astype(np.float32), (3, 3), 0)
    b = _gain_correct(a, b, gain_mask & valid, sigma)
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(np.clip(b, 0, 255), cv2.COLOR_RGB2GRAY)
    dssim = 1.0 - _ssim_map(ga, gb)
    dcol = np.abs(b - a).max(axis=2)
    changed = ((dssim > T_DSSIM) & (dcol > T_COLOUR_MIN)) | (dcol > T_COLOUR)
    return changed & valid, ga, gb, a, b


# --- per item ---------------------------------------------------------------------------------


def _rot_about(theta_deg: float, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    t = math.radians(theta_deg)
    r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    w = np.zeros((2, 3), np.float32)
    w[:, :2] = r
    w[:, 2] = c + d - r @ c
    return w


def _local_motion(ref_g, cur_g, mask, search_px, cur_valid, others) -> dict[str, Any]:
    """Re-find one item: masked NCC over translations (and a few turns), then Euclidean ECC.
    Returns the warp (reference pixels to current pixels), its correlation and the centroid."""
    hh, ww = mask.shape
    ys, xs = np.nonzero(mask)
    c = np.array([xs.mean(), ys.mean()])
    tm = cv2.dilate(mask.astype(np.uint8), np.ones((2 * EDGE_PX + 1,) * 2, np.uint8))
    tm[others & ~mask] = 0
    own = mask.astype(np.uint8)
    oth = others & ~mask
    y0, y1 = max(0, ys.min() - EDGE_PX), min(hh, ys.max() + EDGE_PX + 1)
    x0, x1 = max(0, xs.min() - EDGE_PX), min(ww, xs.max() + EDGE_PX + 1)
    pad = int(search_px)
    wy0, wy1 = max(0, y0 - pad), min(hh, y1 + pad)
    wx0, wx1 = max(0, x0 - pad), min(ww, x1 + pad)
    win = cur_g[wy0:wy1, wx0:wx1]

    cands: list[tuple[float, float, float, np.ndarray]] = []  # (penalised, score, angle, shift)
    for ang in (0.0, *COARSE_ANGLES_DEG):
        if ang != 0.0 and cands and max(cd[1] for cd in cands) >= COARSE_GOOD:
            break
        if ang == 0.0:
            tpl, tmk = ref_g[y0:y1, x0:x1], tm[y0:y1, x0:x1]
        else:
            rw = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), -ang, 1.0)
            rg = cv2.warpAffine(ref_g, rw, (ww, hh), flags=cv2.INTER_LINEAR)
            rm = cv2.warpAffine(tm, rw, (ww, hh), flags=cv2.INTER_NEAREST)
            tpl, tmk = rg[y0:y1, x0:x1], rm[y0:y1, x0:x1]
        if tmk.sum() < 10 or win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
            continue
        if float(tpl[tmk > 0].std()) < 1e-3:
            continue
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED, mask=tmk)
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)
        gy, gx = np.mgrid[0 : res.shape[0], 0 : res.shape[1]]
        dist = np.hypot(gx + wx0 - x0, gy + wy0 - y0)
        peaks = (res >= cv2.dilate(res, np.ones((3, 3), np.uint8))) & (
            res >= res.max() - CANDIDATE_BAND
        )
        for py, px in zip(*np.nonzero(peaks), strict=True):
            pen = float(res[py, px]) - DISTANCE_PENALTY * float(dist[py, px]) / max(
                1.0, float(search_px)
            )
            cands.append(
                (pen, float(res[py, px]), ang, np.array([wx0 + px - x0, wy0 + py - y0], float))
            )
    if not cands:
        cands = [(0.0, 0.0, 0.0, np.zeros(2))]
    cands.sort(key=lambda cd: -cd[0])

    ry0, ry1 = max(0, y0 - pad - 4), min(hh, y1 + pad + 4)
    rx0, rx1 = max(0, x0 - pad - 4), min(ww, x1 + pad + 4)
    a = ref_g[ry0:ry1, rx0:rx1]
    b = cur_g[ry0:ry1, rx0:rx1]
    m = tm[ry0:ry1, rx0:rx1]
    cvm = cur_valid[ry0:ry1, rx0:rx1].astype(np.uint8)
    off = np.array([rx0, ry0], float)
    best: tuple[float, float, np.ndarray] | None = None  # (penalised, rho, warp)
    for _, score, ang, d in cands[:MAX_CANDIDATES]:
        w0 = _rot_about(ang, c, d)
        wl = w0.copy()
        wl[:, 2] = w0[:, 2] + w0[:, :2] @ off - off
        rho = score
        try:
            rho, wl = _ecc_call(a, b, m, cvm, wl, cv2.MOTION_EUCLIDEAN, 60, 1e-4, 3)
            w = wl.astype(np.float64)
            w[:, 2] = wl[:, 2] - wl[:, :2] @ off + off
        except cv2.error:
            w = w0.astype(np.float64)
        newc = w[:, :2] @ c + w[:, 2]
        if np.linalg.norm(newc - c) > search_px + 2:  # ECC ran off: keep the coarse answer
            w, rho = w0.astype(np.float64), score
            newc = w[:, :2] @ c + w[:, 2]
        pen = float(rho) - DISTANCE_PENALTY * float(np.linalg.norm(newc - c)) / max(
            1.0, float(search_px)
        )
        if np.linalg.norm(newc - c) > 0.5:
            wl2 = w.copy()
            wl2[:, 2] = w[:, 2] + w[:, :2] @ off - off
            moved = cv2.warpAffine(
                own[ry0:ry1, rx0:rx1],
                wl2.astype(np.float32),
                (rx1 - rx0, ry1 - ry0),
                flags=cv2.INTER_NEAREST,
            )
            overlap = float((moved.astype(bool) & oth[ry0:ry1, rx0:rx1]).sum()) / max(
                1.0, float(moved.sum())
            )
            pen -= OVERLAP_PENALTY * overlap
        if best is None or pen > best[0]:
            best = (pen, float(rho), w)
    _, rho, w = best
    newc = w[:, :2] @ c + w[:, 2]
    tb = tm[y0:y1, x0:x1] > 0
    gx = cv2.Sobel(ref_g[y0:y1, x0:x1], cv2.CV_32F, 1, 0, ksize=3)[tb] / 8.0
    gy = cv2.Sobel(ref_g[y0:y1, x0:x1], cv2.CV_32F, 0, 1, ksize=3)[tb] / 8.0
    py, px = np.nonzero(tb)
    rx, ry = px + x0 - c[0], py + y0 - c[1]
    jac = np.stack([gx, gy, rx * gy - ry * gx], 1).astype(np.float64)
    info = jac.T @ jac
    texture = float(np.linalg.eigvalsh(info[:2, :2])[0]) / max(1, len(gx))
    return {
        "texture": texture,
        "information": info,
        "warp": w,
        "centroid": c,
        "new_centroid": newc,
        "rotation_deg": -math.degrees(math.atan2(w[1, 0], w[0, 0])),
        "match": float(rho),
    }


def _metric_motion(lm, k, plane, t_cam) -> dict[str, Any] | None:
    """The item's in-plane translation [m] and turn [deg] on its own height's plane."""
    if k is None or plane is None:
        return None
    c, w = lm["centroid"], lm["warp"]
    e = c + np.array([10.0, 0.0])
    before = _on_plane(k, plane, np.array([c[0], e[0]]), np.array([c[1], e[1]]))
    pa = w[:, :2] @ np.stack([c, e]).T + w[:, 2:3]
    after = _on_plane(k, plane, pa[0], pa[1])
    dt = after[0] - before[0]
    n = plane[0]
    v0, v1 = before[1] - before[0], after[1] - after[0]
    v0 -= n * (v0 @ n)
    v1 -= n * (v1 @ n)
    ang = math.degrees(math.atan2(float(np.cross(v0, v1) @ n), float(v0 @ v1)))
    world = None if t_cam is None else t_cam[:3, :3] @ dt
    return {"translation_cam": dt, "translation_world": world, "rotation_deg": ang}


def _blobs(mask: np.ndarray, min_px: int) -> list[np.ndarray]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [lab == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]


def _vec(v: np.ndarray | None) -> dict[str, float] | None:
    return None if v is None else {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def run(
    ctx: NodeContext,
    reference: Any,
    items: list[dict[str, Any]] | None = None,
    current: Any = None,
    camera_name: str = "",
    grid: dict[str, Any] | None = None,
    region: Any = None,
    ignore_mask: Any = None,
    reference_ignore_mask: Any = None,
    expected_ids: list[str] | None = None,
    mask_arms: bool = True,
    arm_base_positions: list[Any] | None = None,
    plane_z: float | None = None,
    mm_per_px: float | None = None,
    search_radius_mm: float = SEARCH_RADIUS_MM,
    min_shift_mm: float = MIN_SHIFT_MM,
    min_shift_px: float = MIN_SHIFT_PX,
    min_rotation_deg: float = MIN_ROTATION_DEG,
    min_depth_change_m: float = MIN_DEPTH_CHANGE_M,
    min_blob_mm2: float = MIN_BLOB_MM2,
    min_blob_px: int = MIN_BLOB_PX,
    max_camera_shift_fraction: float = MAX_CAMERA_SHIFT_FRACTION,
    min_rho: float = MIN_RHO,
) -> Output:
    """Compare ``current`` (default: a fresh ``robot.get_observation``) with ``reference``.

    ``items``: ``[{id, mask | polygon_px | box | cell, expected?}]`` in the reference frame
    (``cell`` needs ``grid``: ``{cols, rows, corners_px}`` in reference pixels, or
    ``{origin: Se3Pose, cell_size, item_height}`` in the world). ``region``: where an
    unexpected object counts (mask, ``{polygon_px}`` or a world box); default the grid, else
    the items' hull widened by the search radius, else the whole frame. ``ignore_mask`` (current
    frame) and ``reference_ignore_mask`` are left out (the slot just filled, a tool);
    ``expected_ids`` may change without counting as disturbed. ``plane_z`` (world) is the
    support height when depth is missing; ``mm_per_px`` the scale when depth and pose are both
    missing.
    """
    items = list(items or [])
    expected = {str(x) for x in (expected_ids or [])}
    for it in items:
        if it.get("expected"):
            expected.add(str(it.get("id")))
    if current is None:
        current = ctx.tool("robot.get_observation")
    ref0 = _frame(reference, camera_name)
    cur0 = _frame(current, camera_name or ref0["name"] or "")
    h0, w0 = ref0["rgb"].shape[:2]
    if cur0["rgb"].shape[:2] != (h0, w0):
        cur0["rgb"] = cv2.resize(cur0["rgb"], (w0, h0), interpolation=cv2.INTER_AREA)
        if cur0["depth"] is not None:
            cur0["depth"] = cv2.resize(cur0["depth"], (w0, h0), interpolation=cv2.INTER_NEAREST)
    s = min(1.0, MAX_WORK_WIDTH / w0)
    ref, cur = _scale_frame(ref0, s), _scale_frame(cur0, s)
    shape = ref["rgb"].shape[:2]
    hh, ww = shape

    # Support plane (camera frame of the reference): from its depth, else the pose and plane_z.
    plane = None
    if ref["depth"] is not None and ref["K"] is not None:
        vv, uu = np.mgrid[0:hh:4, 0:ww:4]
        z = ref["depth"][vv, uu]
        ok = np.isfinite(z)
        pts = _ray(ref["K"], uu[ok], vv[ok]) * z[ok][:, None]
        plane = _fit_plane(pts)
    if plane is None and plane_z is not None and ref["K"] is not None and ref["T"] is not None:
        r, t = ref["T"][:3, :3], ref["T"][:3, 3]
        n = r.T @ np.array([0.0, 0.0, 1.0])
        p = r.T @ (np.array([0.0, 0.0, float(plane_z)]) - t)
        plane = (n, float(n @ p))

    # Masks in the reference's working pixels.
    masks, tops, ids = [], [], []
    for i, it in enumerate(items):
        m, top = _item_mask(it, ref, grid, shape, s)
        masks.append(m)
        tops.append(top)
        ids.append(str(it.get("id", i)))
    ign_cur = np.zeros(shape, bool) if ignore_mask is None else _mask(ignore_mask, shape)
    ign_ref = (
        np.zeros(shape, bool)
        if reference_ignore_mask is None
        else _mask(reference_ignore_mask, shape)
    )
    arm_ref = _arm_mask(ref, shape, arm_base_positions) if mask_arms else np.zeros(shape, bool)
    arm_cur = _arm_mask(cur, shape, arm_base_positions) if mask_arms else np.zeros(shape, bool)

    if region is not None:
        roi = _region_mask(region, ref, shape, s)
    elif grid is not None and "corners_px" in grid:
        roi = _fill_polygon(np.asarray(grid["corners_px"], float) * s, shape)
    elif grid is not None and "origin" in grid and ref["K"] is not None and ref["T"] is not None:
        cols, rows = int(grid["cols"]), int(grid["rows"])
        roi = np.zeros(shape, bool)
        for col, row in ((0, 0), (cols - 1, 0), (cols - 1, rows - 1), (0, rows - 1)):
            roi |= _item_mask({"cell": [col, row]}, ref, grid, shape, s)[0]
        roi = _fill_polygon(np.argwhere(roi)[:, ::-1].astype(float), shape)
    elif masks:
        allm = np.zeros(shape, bool)
        for m in masks:
            allm |= m
        roi = _fill_polygon(np.argwhere(allm)[:, ::-1].astype(float), shape) if allm.any() else allm
        roi = (
            cv2.dilate(roi.astype(np.uint8), np.ones((2 * SEARCH_RADIUS_PX + 1,) * 2, np.uint8)) > 0
        )
    else:
        roi = np.ones(shape, bool)

    # Registration on the static background.
    ref_g = cv2.cvtColor(ref["rgb"], cv2.COLOR_RGB2GRAY)
    cur_g = cv2.cvtColor(cur["rgb"], cv2.COLOR_RGB2GRAY)
    items_all = np.zeros(shape, bool)
    for m in masks:
        items_all |= m
    items_dil = (
        cv2.dilate(items_all.astype(np.uint8), np.ones((2 * ITEM_EXCLUDE_PX + 1,) * 2, np.uint8))
        > 0
    )
    static = ~(items_dil | ign_ref | arm_ref)
    reg_on = "background"
    if static.mean() < MIN_STATIC_FRACTION:
        static = ~(ign_ref | arm_ref)
        reg_on = "background+items"
    cur_free = ~(ign_cur | arm_cur)
    prior = _pose_prior(ref, cur, plane, shape)
    h_orb, n_inl = _orb_homography(ref_g, cur_g, static, cur_free)
    ref_hp, cur_hp = _highpass(ref_g), _highpass(cur_g)
    # Start ECC from whichever of ORB, the pose prior and identity fits the background best
    # (a low-texture view can give ORB a wrong homography with enough inliers).
    inits = [("identity", np.eye(3))]
    if prior is not None:
        inits.append(("pose", prior))
    if h_orb is not None and _corner_shift(h_orb, shape) < max_camera_shift_fraction * math.hypot(
        ww, hh
    ):
        inits.append(("orb", h_orb))
    scored = [
        (_tile_rho(ref_hp, _warp_to_ref(cur_hp, hm, shape), static), name, hm) for name, hm in inits
    ]
    _, init_name, h_init = max(scored, key=lambda x: x[0])
    h, ecc_ok = _ecc(ref_hp, cur_hp, static, h_init, cur_free)
    if (
        ecc_ok
        and _tile_rho(ref_hp, _warp_to_ref(cur_hp, h_init, shape), static)
        > _tile_rho(ref_hp, _warp_to_ref(cur_hp, h, shape), static) + 0.01
    ):
        h, ecc_ok = h_init, False  # ECC was pulled off by something large that changed

    def warp_all(hm: np.ndarray):
        cw = _warp_to_ref(cur["rgb"], hm, shape)
        valid = _warp_to_ref(np.ones(cur_g.shape, np.uint8), hm, shape, nearest=True) > 0
        valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        free = _warp_to_ref(cur_free.astype(np.uint8), hm, shape, nearest=True) > 0
        return cw, valid, free

    cur_w, valid, free_w = warp_all(h)
    rho = _tile_rho(ref_hp, _warp_to_ref(cur_hp, h, shape), static & valid & free_w)
    sigma = GAIN_SIGMA_FRACTION * ww
    blocked = ign_ref | arm_ref | ~free_w
    changed, ga, gb, col_a, col_b = _change_maps(
        ref["rgb"], cur_w, valid & ~blocked, ~blocked, sigma
    )
    # Second pass: the background round the work region only, without the pixels that changed:
    # a plane's homography fitted over the whole frame can leave a pixel of error at the items.
    if ecc_ok:
        excl = cv2.dilate((changed & static).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        near = static & ~excl
        if not roi.all():
            ring = (
                cv2.dilate(roi.astype(np.uint8), np.ones((2 * NEAR_REGION_PX + 1,) * 2, np.uint8))
                > 0
            )
            if (near & ring).sum() >= MIN_NEAR_PX:
                near &= ring
        h2, ok2 = _ecc(ref_hp, cur_hp, near, h, cur_free, levels=(0.5, 1.0), iters=(40, 40))
        if ok2 and _corner_shift(
            np.linalg.inv(h) @ h2, shape
        ) < 0.5 * max_camera_shift_fraction * math.hypot(ww, hh):
            cw2, valid2, free2 = warp_all(h2)
            m1 = near & valid & free_w & valid2 & free2
            r1 = _masked_ncc(ref_hp, _warp_to_ref(cur_hp, h, shape), m1)
            r2 = _masked_ncc(ref_hp, _warp_to_ref(cur_hp, h2, shape), m1)
            if r2 >= r1:
                h, cur_w, valid, free_w = h2, cw2, valid2, free2
                rho = _tile_rho(ref_hp, _warp_to_ref(cur_hp, h, shape), static & valid & free_w)
                blocked = ign_ref | arm_ref | ~free_w
                changed, ga, gb, col_a, col_b = _change_maps(
                    ref["rgb"], cur_w, valid & ~blocked, ~blocked, sigma
                )
    cam_shift = _corner_shift(h, shape) / s
    registration = {
        "method": f"{init_name}+ecc" if ecc_ok else init_name,
        "registered_on": reg_on,
        "orb_inliers": n_inl,
        "rho": round(float(rho), 4),
        "homography": (np.diag([1 / s, 1 / s, 1]) @ h @ np.diag([s, s, 1])).tolist(),
    }
    diag = math.hypot(w0, h0)

    def empty(route: str, detail: str) -> Output:
        return {
            "route": route,
            "items": [],
            "disturbed": [],
            "occluded": [],
            "unexpected": [],
            "occluders": [],
            "n_disturbed": 0,
            "n_unexpected": 0,
            "max_shift_mm": 0.0,
            "camera_shift_px": round(cam_shift, 2),
            "registration": registration,
            "changed_fraction": 0.0,
            "detail": detail,
        }

    if rho < min_rho or not np.isfinite(rho):
        return empty(
            "unregistrable",
            f"background correlation {rho:.3f} < {min_rho} after {registration['method']} registration: the views do not match (re-take the reference)",
        )
    if cam_shift > max_camera_shift_fraction * diag:
        return empty(
            "unregistrable",
            f"camera moved {cam_shift:.0f} px (> {max_camera_shift_fraction:.0%} of the diagonal): re-take the reference",
        )

    # Depth change, with the camera's drift taken out by a plane fitted on the background; then
    # the appearance change, with the lighting fitted where the depth did not change.
    ddz = dc_w = dr_s = None
    depth_changed = np.zeros(shape, bool)
    if ref["depth"] is not None and cur["depth"] is not None:
        dc_w = _warp_to_ref(cur["depth"], h, shape, nearest=True)
        ddz, depth_changed, drift = _depth_change(ref["depth"], dc_w, valid & ~blocked, static)
        dc_w = np.nan_to_num(dc_w, nan=0.0) - drift
        dr_s = np.nan_to_num(ref["depth"], nan=0.0)
    grow = cv2.dilate(depth_changed.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    changed, ga, gb, col_a, col_b = _change_maps(
        ref["rgb"], cur_w, valid & ~blocked, ~blocked & ~grow, sigma
    )
    changed |= depth_changed

    changed = (
        cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    )
    changed &= ~blocked

    # Scale [mm per working pixel] for areas and the search radius.
    def mm_px_at(m: np.ndarray) -> float | None:
        if ref["K"] is not None:
            z = None
            if ref["depth"] is not None:
                zz = ref["depth"][m]
                zz = zz[np.isfinite(zz)]
                if len(zz) >= MIN_DEPTH_PX:
                    z = float(np.median(zz))
            if z is None and plane is not None:
                ys, xs = np.nonzero(m)
                z = float(_on_plane(ref["K"], plane, xs.mean(), ys.mean())[2])
            if z is not None and z > 0:
                return 1000.0 * z / ref["K"][0, 0]
        return None if mm_per_px is None else float(mm_per_px) / s

    frame_mm_px = mm_px_at(roi if roi.any() else np.ones(shape, bool))

    # Occluders: changed blobs that came in from outside the view or from the arm.
    edge = int(BORDER_PX + cam_shift * s)
    near_edge = np.zeros(shape, bool)
    near_edge[: edge + 1, :] = near_edge[-edge - 1 :, :] = True
    near_edge[:, : edge + 1] = near_edge[:, -edge - 1 :] = True
    if cam_shift * s > 1:
        near_edge |= cv2.dilate((~valid).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    arm_touch = cv2.dilate((arm_ref | ~free_w).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    occl = np.zeros(shape, bool)
    occluders: list[dict[str, Any]] = []
    raw = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    for b in _blobs(raw, min_blob_px):
        if not (b & (near_edge | arm_touch)).any():
            continue
        dz = None
        if ddz is not None:
            vals = ddz[b]
            vals = vals[np.isfinite(vals)]
            dz = float(np.median(vals)) if len(vals) >= MIN_DEPTH_PX else None
            if dz is not None and dz > -OCCLUDER_NEARER_M:
                continue
        occl |= b
        ys, xs = np.nonzero(b)
        occluders.append(
            {
                "area_px": int(b.sum() / (s * s)),
                "centroid_px": [float(xs.mean() / s), float(ys.mean() / s)],
                "bbox_px": [
                    float(xs.min() / s),
                    float(ys.min() / s),
                    float(xs.max() / s),
                    float(ys.max() / s),
                ],
                "depth_change_mm": None if dz is None else round(1000 * dz, 1),
            }
        )
    occl = cv2.dilate(occl.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    # Items.
    hidden = blocked | occl | ~valid
    quiet = valid & ~blocked & ~changed & static
    diffs = (gb - ga)[quiet] if quiet.sum() > 500 else (gb - ga)[valid & ~blocked]
    noise_sigma = (
        max(NOISE_FLOOR, 1.4826 * float(np.median(np.abs(diffs - np.median(diffs)))))
        if len(diffs)
        else 2.0
    )
    gb_items = np.where(hidden, ga, gb)  # hidden pixels read as unchanged, so they do not pull
    out_items: list[dict[str, Any]] = []
    explained = np.zeros(shape, bool)
    for idx, (iid, m) in enumerate(zip(ids, masks, strict=True)):
        rec: dict[str, Any] = {"id": iid, "expected": iid in expected}
        area = int(m.sum())
        if area < 4:
            rec.update(
                status="occluded", visible_fraction=0.0, detail="mask is empty or outside the frame"
            )
            out_items.append(rec)
            continue
        vis = m & ~hidden
        vis_frac = float(vis.sum()) / area
        rec["visible_fraction"] = round(vis_frac, 3)
        explained |= cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if vis_frac < MIN_VISIBLE:
            rec.update(status="occluded")
            out_items.append(rec)
            continue
        chg = float(changed[vis].mean())
        mmpx = mm_px_at(m)
        search = int(
            min(MAX_SEARCH_PX, round(search_radius_mm / mmpx) if mmpx else SEARCH_RADIUS_PX)
        )
        lm = _local_motion(ga, gb_items, m, max(4, search), valid & ~hidden, items_all)
        dpx = (lm["new_centroid"] - lm["centroid"]) / s
        rot = lm["rotation_deg"]
        item_plane = plane
        if plane is not None and ref["K"] is not None:
            zz = ref["depth"][m] if ref["depth"] is not None else np.array([])
            zz = zz[np.isfinite(zz)]
            ys, xs = np.nonzero(m)
            if len(zz) >= MIN_DEPTH_PX:
                pc = _ray(ref["K"], xs.mean(), ys.mean()) * float(np.median(zz))
                item_plane = (plane[0], float(plane[0] @ pc))
            elif tops[idx] is not None and ref["T"] is not None:
                r, t = ref["T"][:3, :3], ref["T"][:3, 3]
                p = r.T @ (np.array([0.0, 0.0, tops[idx]]) - t)
                item_plane = (plane[0], float(plane[0] @ p))
        met = _metric_motion(lm, ref["K"], item_plane, ref["T"])
        if met is not None:
            shift_mm = 1000.0 * float(np.linalg.norm(met["translation_cam"]))
            rot = met["rotation_deg"]
        elif mm_per_px is not None:
            shift_mm = float(np.linalg.norm(dpx)) * float(mm_per_px)
        else:
            shift_mm = None
        dz_mm = None
        if ddz is not None:
            vals = ddz[vis]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= MIN_DEPTH_PX:
                dz_mm = 1000.0 * float(np.median(vals))
        if shift_mm is not None:
            floor = MIN_SHIFT_NOISE_PX * mmpx * s if mmpx else 0.0
            moved_xy = shift_mm >= max(min_shift_mm, floor)
        else:
            moved_xy = float(np.linalg.norm(dpx)) >= min_shift_px
        dw = lm["new_centroid"] - lm["centroid"]
        pvec = np.array([dw[0], dw[1], -math.radians(lm["rotation_deg"])])
        significance = float(pvec @ lm["information"] @ pvec) / (2.0 * noise_sigma**2)
        measurable = lm["texture"] >= MIN_TEXTURE
        moved_xy = moved_xy and measurable and significance >= MIN_SIGNIFICANCE
        ys_, xs_ = np.nonzero(m)
        radius = (
            math.sqrt(2.0 * float(((xs_ - xs_.mean()) ** 2 + (ys_ - ys_.mean()) ** 2).mean())) / s
        )
        moved_rot = abs(rot) >= max(
            min_rotation_deg, math.degrees(MIN_ROTATION_ARC_PX / max(radius, 1.0))
        )
        moved_rot = moved_rot and measurable and significance >= MIN_SIGNIFICANCE
        moved_z = dz_mm is not None and abs(dz_mm) >= 1000.0 * min_depth_change_m
        poor = lm["match"] < MIN_MATCH
        if poor and (chg >= LOST_CHANGED or moved_z):
            status = "lost"
        elif (moved_xy or moved_rot) and not poor:
            status = "moved"
        elif moved_z or (poor and chg >= LOST_CHANGED / 2):
            status = "lost" if poor else "moved"
        else:
            status = "unchanged"
        new_dz_mm = new_dcol = None
        if status == "moved":
            nm = cv2.warpAffine(
                m.astype(np.uint8), lm["warp"].astype(np.float32), (ww, hh), flags=cv2.INTER_NEAREST
            )
            core_old = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            core_new = (cv2.erode(nm, np.ones((3, 3), np.uint8)) > 0) & ~hidden
            if core_old.sum() >= 4 and core_new.sum() >= 4:
                # A slid item keeps its colour and its height at the new place; a removed one
                # matched to a look-alike patch (studs of the plate) does not.
                new_dcol = float(
                    np.abs(
                        np.median(col_b[core_new], axis=0) - np.median(col_a[core_old], axis=0)
                    ).max()
                )
                if ddz is not None and dc_w is not None:
                    zn, zo = dc_w[core_new], dr_s[core_old]
                    zn, zo = zn[zn > 0.05], zo[zo > 0.05]
                    if len(zn) >= MIN_DEPTH_PX and len(zo) >= MIN_DEPTH_PX:
                        new_dz_mm = 1000.0 * (float(np.median(zn)) - float(np.median(zo)))
            if (new_dcol is not None and new_dcol > T_COLOUR) or (
                new_dz_mm is not None and abs(new_dz_mm) >= 1000.0 * min_depth_change_m
            ):
                status = "lost"
            else:
                explained |= cv2.dilate(nm, np.ones((9, 9), np.uint8)) > 0
        if status == "lost":
            r_ = 2 * max(4, search) + 1
            explained |= cv2.dilate(m.astype(np.uint8), np.ones((r_, r_), np.uint8)) > 0
        rec.update(
            status=status,
            shift_mm=None if shift_mm is None else round(shift_mm, 2),
            translation_px=[round(float(dpx[0]), 2), round(float(dpx[1]), 2)],
            translation_world=_vec(met["translation_world"]) if met else None,
            rotation_deg=round(float(rot), 2),
            depth_change_mm=None if dz_mm is None else round(dz_mm, 1),
            changed_fraction=round(chg, 3),
            significance=round(significance, 1),
            texture=round(lm["texture"], 2),
            match=round(lm["match"], 3),
            centroid_px=[
                round(float(lm["centroid"][0] / s), 1),
                round(float(lm["centroid"][1] / s), 1),
            ],
        )
        out_items.append(rec)

    # Unexpected blobs: changes inside the region that no item explains.
    rest = changed & roi & ~explained & ~occl
    rest = cv2.morphologyEx(rest.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    unexpected: list[dict[str, Any]] = []
    for b in _blobs(rest, min_blob_px):
        ys, xs = np.nonzero(b)
        mmpx = mm_px_at(b)
        area_mm2 = None if mmpx is None else float(b.sum()) * mmpx * mmpx
        if area_mm2 is not None and area_mm2 < min_blob_mm2:
            continue
        kind = "appearance"
        dz = None
        if ddz is not None:
            vals = ddz[b]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= MIN_DEPTH_PX:
                dz = float(np.median(vals))
                if dz < -DEPTH_CHANGE_M:
                    kind = "appeared"
                elif dz > DEPTH_CHANGE_M:
                    kind = "vacated"
        cu, cv_ = float(xs.mean()), float(ys.mean())
        world = None
        if ref["K"] is not None and ref["T"] is not None:
            z = None
            if ref["depth"] is not None:
                zz = ref["depth"][b]
                zz = zz[np.isfinite(zz)]
                z = float(np.median(zz)) if len(zz) else None
                pc = None if z is None else _ray(ref["K"], cu, cv_) * z
            if z is None and plane is not None:
                pc = _on_plane(ref["K"], plane, cu, cv_)
            if z is not None or plane is not None:
                world = ref["T"][:3, :3] @ pc + ref["T"][:3, 3]
        rec = {
            "kind": kind,
            "area_px": int(b.sum() / (s * s)),
            "area_mm2": None if area_mm2 is None else round(area_mm2, 1),
            "centroid_px": [round(cu / s, 1), round(cv_ / s, 1)],
            "bbox_px": [
                float(xs.min() / s),
                float(ys.min() / s),
                float(xs.max() / s),
                float(ys.max() / s),
            ],
            "depth_change_mm": None if dz is None else round(1000 * dz, 1),
            "position_world": _vec(world),
        }
        if grid is not None and "corners_px" in grid:
            corners = np.asarray(grid["corners_px"], np.float32).reshape(4, 2)
            unit = np.float32(
                [[0, 0], [grid["cols"], 0], [grid["cols"], grid["rows"]], [0, grid["rows"]]]
            )
            inv = cv2.getPerspectiveTransform(corners, unit)
            q = cv2.perspectiveTransform(np.float32([[[cu / s, cv_ / s]]]), inv)[0, 0]
            if 0 <= q[0] < grid["cols"] and 0 <= q[1] < grid["rows"]:
                rec["cell"] = [int(q[0]), int(q[1])]
        unexpected.append(rec)

    judged = [r for r in out_items if not r["expected"]]
    disturbed = [r["id"] for r in judged if r["status"] in ("moved", "lost")]
    occluded = [r["id"] for r in judged if r["status"] == "occluded"]
    shifts = [r.get("shift_mm") or 0.0 for r in judged if r["status"] in ("moved", "lost")]
    roi_valid = roi & valid & ~blocked
    changed_fraction = float(changed[roi_valid].mean()) if roi_valid.any() else 0.0
    scale_note = (
        "" if frame_mm_px is not None else " (no depth, pose or mm_per_px: shifts in px only)"
    )
    if disturbed:
        route = "disturbed"
        lead = ", ".join(
            f"{r['id']} {r['status']}"
            + (
                f" {r['shift_mm']:.1f} mm"
                if r.get("shift_mm") is not None
                else f" {np.hypot(*r['translation_px']):.1f} px"
            )
            + (
                f" {r['rotation_deg']:+.1f} deg"
                if abs(r.get("rotation_deg") or 0) >= min_rotation_deg
                else ""
            )
            for r in judged
            if r["id"] in disturbed[:6]
        )
        detail = f"{len(disturbed)} item(s) disturbed: {lead}"
    elif unexpected:
        route = "unexpected_object"
        detail = f"{len(unexpected)} unexplained change(s) in the region, largest {max(u['area_px'] for u in unexpected)} px"
    elif occluded:
        route = "occluded"
        detail = f"{len(occluded)} item(s) hidden (under the arm, an occluder or an ignore mask): {', '.join(occluded[:8])}"
    else:
        route = "unchanged"
        detail = f"{len(judged)} item(s) where they were"
    if unexpected and route == "disturbed":
        detail += f"; also {len(unexpected)} unexplained change(s)"
    if occluders:
        detail += f"; {len(occluders)} occluder(s) entering the view"
    detail += f"; registered by {registration['method']} (rho {rho:.3f}, camera shift {cam_shift:.1f} px){scale_note}"
    return {
        "route": route,
        "items": out_items,
        "disturbed": disturbed,
        "occluded": occluded,
        "unexpected": unexpected,
        "occluders": occluders,
        "n_disturbed": len(disturbed),
        "n_unexpected": len(unexpected),
        "max_shift_mm": round(max(shifts), 2) if shifts else 0.0,
        "camera_shift_px": round(cam_shift, 2),
        "registration": registration,
        "changed_fraction": round(changed_fraction, 4),
        "detail": detail,
    }
