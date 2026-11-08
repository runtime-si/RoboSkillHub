# Adapted from AMMistry18/aic (https://github.com/AMMistry18/aic, commit fd27bbe8),
# aic_model/aic_model/insertion/visual_gap.py (detect_dark_port_opening) and
# aic_model/aic_model/insertion/visual_gap_recovery.py (VisualGapRecoveryMixin:
# _visual_gap_project_point, _visual_gap_ray_to_port_plane, _visual_gap_consensus,
# _run_visual_gap_wedge_recovery), Apache License 2.0.
#
# Changes from the source: the views come from robot.get_observation (each camera's rgb,
# depth, intrinsics and camera-to-world pose) instead of ROS topics and TF; the port's mouth
# quad is built from a perceived centre, inward axis and opening size instead of the task's
# CAD keypoints; the calibrated per-camera gripper PNG masks are replaced by a depth test
# (pixels nearer the camera than the port plane by more than mask_above_plane are the hand
# or the held part) plus optional caller masks, so fixed and wrist cameras both work; an
# optional sam3 text mask can stand in for the dark threshold (an opening that is not darker
# than its surround); the hand is stepped with robot.go_to_pose (which takes a tolerance), each target the
# previous target plus the capped step (so an arm that lags its command still converges);
# the failure paths are named exits instead of a fall-back to the source's force sweep;
# the debug image encoder is dropped.
"""Centre a held part over a port opening seen by two or more cameras, in capped steps."""

from __future__ import annotations

import itertools
import logging
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

MAX_STEPS = 8
"""Correction moves before giving up (upstream VISUAL_GAP_MAX_STEPS)."""

MAX_STEP = 0.001
"""Longest single lateral move [m] (upstream VISUAL_GAP_MAX_STEP_M)."""

MAX_TOTAL = 0.006
"""Longest total commanded lateral excursion [m] (upstream VISUAL_GAP_MAX_TOTAL_M)."""

MAX_PORT_OFFSET = 0.008
"""Furthest the cameras' opening may lie from the estimate [m] before the estimate is
called wrong rather than corrected (upstream VISUAL_GAP_MAX_PORT_OFFSET_M)."""

MAX_VIEW_DISAGREEMENT = 0.003
"""Furthest two views' opening centres on the port plane may lie apart and still agree [m]
(upstream VISUAL_GAP_MAX_VIEW_DISAGREE_M)."""

PIXEL_TOL_FRAC = 0.12
"""Centred in a view when the part is within this fraction of sqrt(projected opening area)
of the opening's centre, in pixels (upstream VISUAL_GAP_PIXEL_TOL_FRAC). In metric terms
about this fraction of the opening's size: 2.9 mm on a 24 mm opening, too loose for a part
with 1-2 mm of clearance; set it near clearance / opening_size."""

MIN_PIXEL_TOL = 2.0
"""Floor of that pixel tolerance [px] (upstream VISUAL_GAP_MIN_PIXEL_TOL)."""

MIN_VIEWS = 2
"""Views that must see the opening, agree, and see the part centred (upstream: 2)."""

MASK_ABOVE_PLANE = 0.008
"""Pixels whose depth is this much nearer than the port plane are ignored [m]: the hand and
the held part. Replaces upstream's calibrated gripper masks; 8 mm clears a depth camera's
noise at half a metre (D405 / Orbbec Gemini 335: ~2-5 mm) and is thinner than any gripper."""

SEARCH_SCALE = 1.55
"""Search polygon = the projected mouth quad scaled about its centre (upstream default)."""

MIN_CONTRAST = 24.0
"""Least p90 - p10 grey spread inside the search polygon (upstream default)."""

MIN_AREA_RATIO = 0.16
MAX_AREA_RATIO = 1.35
"""Accepted component area / projected mouth area (upstream defaults)."""

MAX_CENTER_DISTANCE = 0.75
"""Furthest a component's centroid may lie from the quad's centre, in sqrt(area) units
(upstream default)."""

OCCLUSION_REFIT = 0.15
"""When more of the found opening's outline than this lies on ignored pixels, an occluder
cuts it and its centroid leans away from the occluder: its centre is refitted with the
projected mouth quad instead. Measured in a YAM station twin: the wrist camera of the arm
holding the part sees the fingertips over the far half of the mouth (40% of its area
hidden, 0.28 of its outline), and the centroid sat 3.5 mm from the true centre; refitted, 0.7 mm."""

SAM3_SCORE_MIN = 0.2
"""Least sam3 score for an opening mask when opening_query is given."""


class Output(TypedDict):
    route: str
    detail: str
    hole_position: dict[str, float]
    hole_offset: float
    correction: dict[str, float]
    steps: int
    views: int
    pixel_errors: dict[str, float]


# ----------------------------------------------------------------------------- detector


def _gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(np.ascontiguousarray(array).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif array.ndim == 3 and array.shape[2] == 4:
        gray = cv2.cvtColor(np.ascontiguousarray(array).astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    else:
        raise ValueError("image must be grayscale, RGB, or RGBA")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(gray)


def _search_region(
    shape: tuple[int, int], quad: np.ndarray, ignored: np.ndarray | None, search_scale: float
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Valid pixels (scaled quad minus ignored), the quad's area, and its centre."""
    height, width = shape
    expected_area = float(cv2.contourArea(cv2.convexHull(quad.astype(np.float32))))
    if expected_area < 20.0:
        return None
    center = quad.mean(axis=0)
    search_hull = cv2.convexHull(np.rint(center + search_scale * (quad - center)).astype(np.int32))
    search_hull[:, 0, 0] = np.clip(search_hull[:, 0, 0], 0, width - 1)
    search_hull[:, 0, 1] = np.clip(search_hull[:, 0, 1], 0, height - 1)
    valid = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(valid, search_hull, 255)
    if ignored is not None:
        valid[np.asarray(ignored, dtype=bool)] = 0
    return valid, expected_area, center


def _select(
    gray: np.ndarray,
    candidates: list[np.ndarray],
    expected_area: float,
    center: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float,
    max_center_distance: float,
) -> dict[str, Any] | None:
    """Upstream's gates and score over candidate component masks; lowest score wins."""
    norm_scale = max(np.sqrt(expected_area), 1.0)
    best = None
    for component in candidates:
        area = int(np.count_nonzero(component))
        area_ratio = area / expected_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        rows, cols = np.nonzero(component)
        centroid = np.array([cols.mean(), rows.mean()], dtype=np.float64)
        center_distance = float(np.linalg.norm(centroid - center) / norm_scale)
        if center_distance > max_center_distance:
            continue
        mean_darkness = float(gray[component].mean()) / 255.0
        # A real opening normally fills roughly half the projected mouth quad (upstream).
        area_cost = abs(float(np.log(max(area_ratio, 1e-6) / 0.55)))
        score = 2.5 * center_distance + 0.35 * area_cost + 0.30 * mean_darkness
        if best is None or score < best["score"]:
            best = {
                "center_uv": centroid,
                "area_px": area,
                "score": float(score),
                "component": component,
            }
    if best is not None:
        best["expected_area_px"] = expected_area
    return best


def occluded_fraction(component: np.ndarray, ignored: np.ndarray | None) -> float:
    """Share of a component's 2 px outer ring that lies on ignored pixels: how much of its
    outline an occluder (the hand, the held part) cuts off."""
    if ignored is None:
        return 0.0
    blob = component.astype(np.uint8)
    ring = (cv2.dilate(blob, np.ones((3, 3), np.uint8), iterations=2) > 0) & ~component
    n = int(np.count_nonzero(ring))
    return 0.0 if n == 0 else float(np.count_nonzero(ring & ignored)) / n


def refit_center(
    component: np.ndarray,
    quad: np.ndarray,
    seen: np.ndarray,
    max_shift: float,
    center_uv: np.ndarray | None = None,
) -> np.ndarray:
    """The projected mouth quad's centre after sliding the quad over the image to best cover
    the component, counting only the ``seen`` pixels (the search polygon less the ignored
    ones): the centre of an opening an occluder partly hides, which the visible part's
    centroid is not. ``center_uv`` is the projected mouth centre (under perspective not the
    corners' mean, which is the default)."""
    h, w = component.shape
    reach = int(np.ceil(max_shift)) + 2
    lo = np.maximum(np.floor(quad.min(axis=0)) - reach, 0).astype(int)
    hi = np.minimum(np.ceil(quad.max(axis=0)) + reach + 1, [w, h]).astype(int)
    comp = component[lo[1] : hi[1], lo[0] : hi[0]]
    seen = seen[lo[1] : hi[1], lo[0] : hi[0]]
    target = comp & seen
    local = quad - lo

    def cost(shift: np.ndarray) -> int:
        q = np.zeros(comp.shape, np.uint8)
        cv2.fillConvexPoly(q, np.rint(local + shift).astype(np.int32), 1)
        q = (q > 0) & seen
        return int(np.count_nonzero(q ^ target))

    def search(center: np.ndarray, radius: int, stride: int) -> np.ndarray:
        best = (cost(center), float(np.linalg.norm(center)), center)
        for dx in range(-radius, radius + 1, stride):
            for dy in range(-radius, radius + 1, stride):
                s = center + np.array([dx, dy], dtype=np.float64)
                if np.linalg.norm(s) > max_shift:
                    continue
                c = cost(s)
                if (c, np.linalg.norm(s)) < best[:2]:
                    best = (c, float(np.linalg.norm(s)), s)
        return best[2]

    shift = search(np.zeros(2), int(np.ceil(max_shift)), 3)
    shift = search(shift, 3, 1)
    anchor = quad.mean(axis=0) if center_uv is None else np.asarray(center_uv, dtype=np.float64)
    return anchor + shift


def detect_dark_port_opening(
    image: np.ndarray,
    expected_quad_uv: np.ndarray,
    ignored_pixels: np.ndarray | None = None,
    *,
    search_scale: float = SEARCH_SCALE,
    min_contrast: float = MIN_CONTRAST,
    min_area_ratio: float = MIN_AREA_RATIO,
    max_area_ratio: float = MAX_AREA_RATIO,
    max_center_distance: float = MAX_CENTER_DISTANCE,
    center_uv: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Find the dark opening near its projected quadrilateral (upstream's detector).

    Returns ``center_uv``, ``area_px``, ``expected_area_px``, ``score``, or None.
    ``ignored_pixels`` is True where pixels must not participate."""
    gray = _gray(image)
    quad = np.asarray(expected_quad_uv, dtype=np.float64)
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        raise ValueError("expected_quad_uv must be a finite 4x2 array")
    region = _search_region(gray.shape, quad, ignored_pixels, search_scale)
    if region is None:
        return None
    valid, expected_area, center = region
    values = gray[valid != 0]
    if values.size < max(25, int(0.35 * expected_area)):
        return None
    p10, p50, p90 = np.percentile(values, [10.0, 50.0, 90.0])
    contrast = float(p90 - p10)
    if contrast < min_contrast:
        return None
    otsu_threshold, _ = cv2.threshold(
        values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    # Keep the threshold on the dark side of the local distribution (upstream): this
    # rejects medium-grey walls while retaining the nearly black opening.
    threshold = float(min(otsu_threshold, p50 - 0.10 * contrast))
    if threshold <= float(p10) + 2.0:
        threshold = float(p10) + 0.25 * contrast
    dark = np.zeros_like(valid)
    dark[(valid != 0) & (gray <= threshold)] = 255
    min_edge = max(1.0, min(float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)))
    if min_edge >= 9.0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    count, labels = cv2.connectedComponents(dark, connectivity=8)
    candidates = [labels == label for label in range(1, count)]
    found = _select(
        gray,
        candidates,
        expected_area,
        center,
        min_area_ratio,
        max_area_ratio,
        max_center_distance,
    )
    if found is not None:
        found["contrast"] = contrast
    return _refit_if_occluded(found, quad, ignored_pixels, valid, max_center_distance, center_uv)


def detect_masked_opening(
    image: np.ndarray,
    expected_quad_uv: np.ndarray,
    masks: list[np.ndarray],
    ignored_pixels: np.ndarray | None = None,
    *,
    search_scale: float = SEARCH_SCALE,
    min_area_ratio: float = MIN_AREA_RATIO,
    max_area_ratio: float = MAX_AREA_RATIO,
    max_center_distance: float = MAX_CENTER_DISTANCE,
    center_uv: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """The same gates and score as the dark detector, over segmentation masks clipped to the
    search polygon: for an opening that is not darker than its surround."""
    gray = _gray(image)
    quad = np.asarray(expected_quad_uv, dtype=np.float64)
    region = _search_region(gray.shape, quad, ignored_pixels, search_scale)
    if region is None:
        return None
    valid, expected_area, center = region
    clipped = [(np.asarray(m) > 0) & (valid != 0) for m in masks]
    found = _select(
        gray,
        [c for c in clipped if c.any()],
        expected_area,
        center,
        min_area_ratio,
        max_area_ratio,
        max_center_distance,
    )
    return _refit_if_occluded(found, quad, ignored_pixels, valid, max_center_distance, center_uv)


def _refit_if_occluded(
    found: dict[str, Any] | None,
    quad: np.ndarray,
    ignored: np.ndarray | None,
    valid: np.ndarray,
    max_center_distance: float,
    center_uv: np.ndarray | None,
) -> dict[str, Any] | None:
    if found is None or ignored is None:
        return found
    found["occluded"] = occluded_fraction(found["component"], np.asarray(ignored, dtype=bool))
    if found["occluded"] > OCCLUSION_REFIT:
        found["centroid_uv"] = found["center_uv"]
        found["center_uv"] = refit_center(
            found["component"],
            quad,
            valid != 0,
            max_center_distance * np.sqrt(found["expected_area_px"]),
            center_uv,
        )
    return found


# ----------------------------------------------------------------------------- geometry


def _vec3(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=np.float64).reshape(3)


def _out3(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _quat_matrix(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _pose_matrix(pose: Any) -> np.ndarray:
    T = np.eye(4)
    if isinstance(pose, dict):
        T[:3, :3] = _quat_matrix(pose["rotation"])
        T[:3, 3] = _vec3(pose["position"])
        return T
    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    flat = arr.reshape(-1)
    T[:3, 3] = flat[:3]
    T[:3, :3] = _quat_matrix(dict(zip(("w", "x", "y", "z"), flat[3:7], strict=True)))
    return T


def port_frame(axis: np.ndarray, x_hint: np.ndarray | None = None) -> np.ndarray:
    """Columns: the port's in-plane x and y, and its inward axis (upstream's Rp)."""
    z = axis / np.linalg.norm(axis)
    hint = np.array([1.0, 0.0, 0.0]) if x_hint is None else np.asarray(x_hint, dtype=np.float64)
    x = hint - (hint @ z) * z
    if np.linalg.norm(x) < 1e-6:
        hint = np.array([0.0, 1.0, 0.0])
        x = hint - (hint @ z) * z
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def project_point(K: np.ndarray, T_world_cam: np.ndarray, point: np.ndarray) -> np.ndarray | None:
    """World point to pixel (upstream _visual_gap_project_point), or None behind the camera."""
    pc = T_world_cam[:3, :3].T @ (np.asarray(point, dtype=np.float64) - T_world_cam[:3, 3])
    if not np.all(np.isfinite(pc)) or pc[2] <= 1e-6:
        return None
    uvw = K @ pc
    return uvw[:2] / uvw[2]


def ray_to_plane(
    uv: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray, port_pos: np.ndarray, normal: np.ndarray
) -> np.ndarray | None:
    """A pixel's ray met with the port plane (upstream _visual_gap_ray_to_port_plane)."""
    origin = T_world_cam[:3, 3]
    ray = T_world_cam[:3, :3] @ np.linalg.solve(K, np.array([float(uv[0]), float(uv[1]), 1.0]))
    norm = float(np.linalg.norm(ray))
    if norm <= 1e-9:
        return None
    ray /= norm
    denominator = float(normal @ ray)
    if abs(denominator) <= 1e-6:
        return None
    distance = float(normal @ (port_pos - origin) / denominator)
    if not np.isfinite(distance) or distance <= 0.0:
        return None
    return origin + distance * ray


def consensus(
    points: list[np.ndarray], Rp: np.ndarray, max_disagreement: float, min_views: int
) -> tuple[np.ndarray | None, float]:
    """Upstream _visual_gap_consensus without its offset gate: the closest-agreeing pair,
    plus every other view within the bound of its mean. Returns (point or None, the best
    pair's in-plane disagreement)."""
    if len(points) < max(1, min_views):
        return None, float("nan")
    if len(points) == 1:
        return points[0], 0.0
    best = None
    for i, j in itertools.combinations(range(len(points)), 2):
        disagreement = float(np.linalg.norm((Rp.T @ (points[i] - points[j]))[:2]))
        if best is None or disagreement < best[0]:
            best = (disagreement, i, j)
    disagreement, i, j = best
    if disagreement > max_disagreement:
        return None, disagreement
    pair_mean = np.mean([points[i], points[j]], axis=0)
    kept = [points[i], points[j]] + [
        p
        for k, p in enumerate(points)
        if k not in (i, j)
        and float(np.linalg.norm((Rp.T @ (p - pair_mean))[:2])) <= max_disagreement
    ]
    if len(kept) < min_views:
        return None, disagreement
    return np.mean(kept, axis=0), disagreement


def plane_depth_mask(
    depth: np.ndarray,
    K: np.ndarray,
    T_world_cam: np.ndarray,
    port_pos: np.ndarray,
    normal: np.ndarray,
    margin: float,
) -> np.ndarray:
    """True where the measured depth is nearer than the port plane by more than ``margin``:
    whatever stands above the mouth (the hand, the held part). Pixels without depth stay."""
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    h, w = d.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    Kinv = np.linalg.inv(K)
    rays_cam = np.stack([u, v, np.ones_like(u)], axis=-1) @ Kinv.T
    rays_world = rays_cam @ T_world_cam[:3, :3].T
    denominator = rays_world @ normal
    numerator = float(normal @ (port_pos - T_world_cam[:3, 3]))
    with np.errstate(divide="ignore", invalid="ignore"):
        plane_z = numerator / denominator  # rays_cam has z = 1, so this is the plane's z-depth
    ok = np.isfinite(d) & (d > 1e-4)
    return ok & np.isfinite(plane_z) & (plane_z > 0) & (d < plane_z - margin)


# ----------------------------------------------------------------------------- the skill


def _views(observation: Any, names: list[str] | None) -> dict[str, dict[str, Any]]:
    """Every camera with rgb, intrinsics and pose; the observation's frames may be a dict
    or a list of named blocks, with channels inline or under ``images``."""
    frames = observation.get("cameras") if isinstance(observation, dict) else None
    if isinstance(frames, dict):
        blocks = [dict(v, name=v.get("name", k)) for k, v in frames.items()]
    else:
        blocks = list(frames or [])
    out: dict[str, dict[str, Any]] = {}
    for block in blocks:
        name = str(block.get("name", ""))
        if names is not None and name not in names:
            continue
        images = block.get("images") if isinstance(block.get("images"), dict) else {}

        def first(*keys, _images=images, _block=block):
            for src in (_images, _block):
                for key in keys:
                    if src.get(key) is not None:
                        return src[key]
            return None

        rgb, K, pose = first("rgb", "color"), block.get("intrinsics"), block.get("pose")
        if rgb is None or K is None or pose is None:
            continue
        out[name] = {
            "rgb": np.asarray(rgb),
            "depth": first("depth", "depth_data"),
            "K": np.asarray(K, dtype=np.float64).reshape(3, 3),
            "T": _pose_matrix(pose),
        }
    return out


def _ee(ctx: NodeContext) -> tuple[np.ndarray, dict[str, float]] | None:
    try:
        reply = ctx.tool("robot.get_ee_pose")
        pose = reply.get("pose", reply) if isinstance(reply, dict) else reply
        return _vec3(pose["position"]), dict(pose["rotation"])
    except Exception as exc:
        logger.info("correcting-insertion-offset: no pose reading (%s)", exc)
        return None


def run(
    ctx: NodeContext,
    port_position: dict[str, float],
    opening_size: float,
    port_axis: dict[str, float] | None = None,
    opening_height: float | None = None,
    port_x_axis: dict[str, float] | None = None,
    tip_in_tcp: dict[str, float] | None = None,
    cameras: list[str] | None = None,
    opening_query: str | None = None,
    ignore_masks: dict[str, Any] | None = None,
    mask_above_plane: float = MASK_ABOVE_PLANE,
    min_views: int = MIN_VIEWS,
    max_steps: int = MAX_STEPS,
    max_step: float = MAX_STEP,
    max_total: float = MAX_TOTAL,
    max_port_offset: float = MAX_PORT_OFFSET,
    max_view_disagreement: float = MAX_VIEW_DISAGREEMENT,
    pixel_tol_frac: float = PIXEL_TOL_FRAC,
    move_tolerance: float | None = None,
) -> Output:
    port_pos = _vec3(port_position)
    axis = _vec3(port_axis) if port_axis is not None else np.array([0.0, 0.0, -1.0])
    Rp = port_frame(axis, None if port_x_axis is None else _vec3(port_x_axis))
    insert_axis = Rp[:, 2]
    hw = 0.5 * float(opening_size)
    hh = 0.5 * float(opening_height if opening_height is not None else opening_size)
    mouth = [port_pos + Rp[:, 0] * sx * hw + Rp[:, 1] * sy * hh for sx, sy in
             ((-1, -1), (1, -1), (1, 1), (-1, 1))]  # fmt: skip
    tip_offset = np.zeros(3) if tip_in_tcp is None else _vec3(tip_in_tcp)
    min_views = max(1, int(min_views))
    out: Output = {
        "route": "no_convergence",
        "detail": "",
        "hole_position": _out3(port_pos),
        "hole_offset": 0.0,
        "correction": _out3(np.zeros(3)),
        "steps": 0,
        "views": 0,
        "pixel_errors": {},
    }

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        logger.info("correcting-insertion-offset: %s (%s)", route, detail)
        return out

    commanded = np.zeros(3)
    target: np.ndarray | None = None
    for step in range(int(max_steps) + 1):
        ee = _ee(ctx)
        if ee is None:
            return finish("unreachable", "the hand's pose could not be read")
        tcp, rotation = ee
        if target is None:
            target = tcp.copy()
        tip = tcp + _quat_matrix(rotation) @ tip_offset
        # Compare the opening with the part's descent line at the mouth plane, not the tip
        # itself: a tip above or below the plane would add camera-dependent parallax.
        intercept = tip - insert_axis * float((tip - port_pos) @ insert_axis)
        try:
            views = _views(ctx.tool("robot.get_observation"), cameras)
        except Exception as exc:
            return finish("opening_not_found", f"no observation ({exc})")
        if len(views) < min_views:
            return finish(
                "opening_not_found",
                f"{len(views)} usable camera(s) ({sorted(views)}), need {min_views}",
            )

        hits = []
        for name, view in views.items():
            K, T = view["K"], view["T"]
            quad = [project_point(K, T, p) for p in mouth]
            if any(q is None for q in quad):
                continue
            quad = np.asarray(quad)
            port_uv = project_point(K, T, port_pos)
            ignored = np.zeros(view["rgb"].shape[:2], dtype=bool)
            if view["depth"] is not None and mask_above_plane > 0:
                ignored |= plane_depth_mask(
                    view["depth"], K, T, port_pos, insert_axis, float(mask_above_plane)
                )
            if ignore_masks and ignore_masks.get(name) is not None:
                ignored |= np.asarray(ignore_masks[name]) > 0
            if opening_query:
                found = ctx.tool(
                    "sam3.segment_text", image=view["rgb"], query=opening_query, max_results=5
                )
                masks = [
                    m
                    for m, s in zip(
                        found.get("masks") or [], found.get("scores") or [], strict=False
                    )
                    if float(s) >= SAM3_SCORE_MIN
                ]
                detection = detect_masked_opening(
                    view["rgb"], quad, masks, ignored, center_uv=port_uv
                )
            else:
                detection = detect_dark_port_opening(view["rgb"], quad, ignored, center_uv=port_uv)
            if detection is None:
                logger.info("correcting-insertion-offset: %s: opening not found", name)
                continue
            part_uv = project_point(K, T, intercept)
            plane_point = ray_to_plane(detection["center_uv"], K, T, port_pos, insert_axis)
            if part_uv is None or plane_point is None:
                continue
            hits.append(
                {
                    "camera": name,
                    "plane_point": plane_point,
                    "pixel_error": float(np.linalg.norm(detection["center_uv"] - part_uv)),
                    "pixel_tolerance": max(
                        MIN_PIXEL_TOL,
                        float(pixel_tol_frac) * np.sqrt(detection["expected_area_px"]),
                    ),
                }
            )
        out["views"] = len(hits)
        out["pixel_errors"] = {h["camera"]: round(h["pixel_error"], 2) for h in hits}
        errors = ", ".join(
            f"{h['camera']}={h['pixel_error']:.1f}/{h['pixel_tolerance']:.1f}px" for h in hits
        )
        if len(hits) < min_views:
            return finish(
                "opening_not_found",
                f"the opening was found in {len(hits)} of {len(views)} camera(s) "
                f"({sorted(h['camera'] for h in hits)}), need {min_views}",
            )

        hole, disagreement = consensus(
            [h["plane_point"] for h in hits], Rp, float(max_view_disagreement), min_views
        )
        if hole is None:
            return finish(
                "views_disagree",
                f"the closest pair of views put the opening {disagreement * 1e3:.1f} mm apart "
                f"on the port plane (bound {float(max_view_disagreement) * 1e3:.1f} mm)",
            )
        offset = Rp.T @ (hole - port_pos)
        hole = hole - insert_axis * float(offset[2])
        out["hole_position"] = _out3(hole)
        out["hole_offset"] = float(np.linalg.norm(offset[:2]))
        if out["hole_offset"] > float(max_port_offset):
            return finish(
                "offset_too_large",
                f"the cameras put the opening {out['hole_offset'] * 1e3:.1f} mm from the "
                f"estimate (bound {float(max_port_offset) * 1e3:.1f} mm); re-perceive the port",
            )

        centred = [h for h in hits if h["pixel_error"] <= h["pixel_tolerance"]]
        if len(centred) >= min_views:
            return finish(
                "centered",
                f"centred in {len(centred)}/{len(hits)} views after {step} step(s) ({errors}); "
                f"the opening is {out['hole_offset'] * 1e3:.1f} mm from the estimate",
            )
        if step >= int(max_steps):
            return finish("no_convergence", f"not centred after {step} steps ({errors})")

        correction = hole - intercept
        correction = correction - insert_axis * float(correction @ insert_axis)
        norm = float(np.linalg.norm(correction))
        if not np.isfinite(norm) or norm <= 1e-6:
            return finish("no_convergence", f"no usable correction ({errors})")
        step_vec = correction * min(1.0, float(max_step) / norm)
        if float(np.linalg.norm(commanded + step_vec)) > float(max_total) + 1e-9:
            return finish(
                "offset_too_large",
                f"the next step would take the commanded correction past "
                f"{float(max_total) * 1e3:.1f} mm ({errors})",
            )
        commanded = commanded + step_vec
        target = target + step_vec
        out["correction"], out["steps"] = _out3(commanded), step + 1
        kwargs: dict[str, Any] = {}
        if move_tolerance is not None:
            kwargs["tolerance"] = float(move_tolerance)
        try:
            reply = ctx.tool(
                "robot.go_to_pose",
                pose={"position": _out3(target), "rotation": rotation},
                **kwargs,
            )
        except Exception as exc:
            return finish("unreachable", f"step {step + 1} was refused ({exc})")
        if isinstance(reply, dict) and reply.get("outcome") == "unreachable":
            return finish("unreachable", f"step {step + 1} is out of reach")
        logger.info(
            "correcting-insertion-offset: step %d: %s; nudge %.2f mm, total %.2f mm",
            step + 1,
            errors,
            np.linalg.norm(step_vec) * 1e3,
            np.linalg.norm(commanded) * 1e3,
        )
    return finish("no_convergence", "the step budget ran out")
