# Adapted from MAESTRO (https://github.com/penn-pal-lab/maestro, MIT,
# Copyright (c) 2026 The MAESTRO Authors) at commit
# e60d326233a238ddc496bf7a08acb7a278e9e66d: maestro/api.py
# (RobotAPI.generate_best_grid_from_mask, _sample_points_from_mask,
# locate_object_part) and maestro/vlm.py (SELECTION_PROMPT, select_points).
# Changes: the mask comes from the caller or sam3.segment_text instead of
# GroundedSAM on the wrist camera; boundary samples come from the largest
# outline (upstream took the first) and an interior marker that would cover
# an outline one is dropped; when no stride gives a count inside the range the count
# nearest to it is kept (upstream kept the largest count); markers without
# depth are not drawn and the rest are numbered 1..N (upstream drew all and
# mapped an unoffered index to the nearest offered number); the crop is scaled
# by the smaller of the height and width ratios (upstream: height only) and
# marker size follows the marker spacing; the VLM goes through the hub's
# vlm.query, is told it may answer [] and is asked again on a bad reply; the
# prompt lists the offered numbers instead of the 3D positions; a marker's
# world point is the near-side depth of a small window inside the mask,
# back-projected by geometry.mask_to_world_points (upstream read one pixel of
# the wrist point cloud); the outputs are world-frame hub types and routed exits.
"""Number a grid of markers inside an object mask, let a VLM pick the marker(s) for an instruction, return world points."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

TARGET_RANGE = (35, 45)
"""Interior markers wanted: the largest stride whose grid leaves this many inside
the mask wins (upstream locate_object_part's call; the function's own default was
25-35)."""

BOUNDARY_SAMPLES = 20
"""Markers spaced evenly along the mask's outline (upstream locate_object_part's
call; the function's default was 10)."""

GRID_INSET = 5
"""Interior grid starts this many pixels inside the mask's box (upstream's
min_x + 5 .. max_x - 4)."""

CROP_MARGIN = 10
"""Context kept around the mask's box in the crop the VLM sees [px] (upstream)."""

MARKER_RADIUS = 17
"""Marker disc radius on the upscaled crop [px] (upstream); shrunk to fit when
markers are closer than two radii."""

MIN_MARKER_RADIUS = 8
"""Smallest disc that still holds a legible two-digit number."""

DEPTH_WINDOW = 3
"""Half-size of the pixel window a marker's depth is read from [px]; the
executing-keypoint-motion choice, so both read a point the same way."""

DEPTH_PERCENTILE = 30.0
"""The window's depths at or under this percentile (inside the mask) make the
point: the near side, so a marker on an object's edge reads the object and not
the table behind it."""

MIN_DEPTH, MAX_DEPTH = 0.015, 20.0
"""Valid depth [m], the bounds geometry.mask_to_world_points applies."""

MIN_MASK_PIXELS = 20
"""A mask smaller than this is treated as not seen."""

SAM_SCORE_FLOOR = 0.3
"""sam3.segment_text masks scoring under this are ignored (the sam3 bundle's advice)."""

ATTEMPTS = 2
"""VLM replies tried before giving up on a bad one (upstream asked once)."""

SELECTION_PROMPT = (
    "You are given an image with some numbered points and a language instruction. "
    "Based on the instruction and the image, return a list of indices of the points. "
    "Please only output a list of points in format of "
    '[{"index": 0, "label": "left top corner of the cube"}].'
)
"""maestro/vlm.py SELECTION_PROMPT, verbatim."""

NONE_CLAUSE = "If no numbered point fits the instruction, output []."


class Output(TypedDict):
    route: str
    detail: str
    point: dict[str, float]
    points: list[dict[str, float]]
    selected: list[dict[str, Any]]
    pixel: list[float]
    marker_count: int
    marker_pixels: list[list[float]]
    marked_image: np.ndarray
    selection_image: np.ndarray
    reply: str


# ---------------------------------------------------------------------------
# The marker grid (maestro generate_best_grid_from_mask)
# ---------------------------------------------------------------------------


def _grid_inside(mask: np.ndarray, stride: int, box: tuple[int, int, int, int]) -> np.ndarray:
    min_x, max_x, min_y, max_y = box
    gx = np.arange(min_x + GRID_INSET, max_x - GRID_INSET + 1, stride)
    gy = np.arange(min_y + GRID_INSET, max_y - GRID_INSET + 1, stride)
    if not len(gx) or not len(gy):
        return np.zeros((0, 2), int)
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    rc = np.stack([yy.ravel(), xx.ravel()], axis=1)
    return rc[mask[rc[:, 0], rc[:, 1]]]


def best_grid(
    mask: np.ndarray, target_range: tuple[int, int] = TARGET_RANGE
) -> tuple[np.ndarray, int]:
    """(row, col) interior grid points and the stride: the largest stride whose
    count lands in target_range, else the one whose count is nearest to it."""
    ys, xs = np.nonzero(mask)
    box = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
    lo, hi = int(target_range[0]), int(target_range[1])
    best, best_stride, best_gap = np.zeros((0, 2), int), 0, None
    for stride in range(min(box[1] - box[0], box[3] - box[2]) + 1, 1, -1):
        pts = _grid_inside(mask, stride, box)
        if lo <= len(pts) <= hi:
            return pts, stride
        gap = lo - len(pts) if len(pts) < lo else len(pts) - hi
        if best_gap is None or gap < best_gap:
            best, best_stride, best_gap = pts, stride, gap
    return best, best_stride


def boundary_samples(mask: np.ndarray, n: int = BOUNDARY_SAMPLES) -> np.ndarray:
    """(row, col) points spaced evenly along the largest outer contour."""
    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    if not contours or n <= 0:
        return np.zeros((0, 2), int)
    outline = max(contours, key=len).reshape(-1, 2)
    idx = np.linspace(0, len(outline) - 1, int(n)).astype(int)
    return outline[idx][:, ::-1]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    if name:
        for camera in cameras:
            if camera.get("name") == name:
                return camera
        raise ValueError(f"observation has no camera named {name!r}")
    for camera in cameras:
        cam = str(camera.get("name") or "").lower()
        if not any(k in cam for k in ("wrist", "eye_in_hand", "hand")):
            return camera
    if cameras:
        return cameras[0]
    raise ValueError("observation has no cameras")


def _as_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.size == 0:
        return None
    m = m.reshape(m.shape[-2:]) if m.ndim > 2 else m
    if m.shape != shape:
        raise ValueError(f"mask {m.shape} and image {shape} differ in size")
    return m > 0


def _segment(ctx: NodeContext, rgb: np.ndarray, query: str) -> tuple[np.ndarray | None, str]:
    found = ctx.tool("sam3.segment_text", image=rgb, query=query, max_results=3)
    scores = found.get("scores") or []
    for m, score in zip(found.get("masks") or [], scores, strict=False):
        if float(score) >= SAM_SCORE_FLOOR:
            seg = _as_mask(m, rgb.shape[:2])
            if seg is not None and int(seg.sum()) >= MIN_MASK_PIXELS:
                return seg, f"sam3 {float(score):.2f}"
    best = max((float(s) for s in scores), default=None)
    return None, "sam3 found nothing" if best is None else f"sam3's best mask scored {best:.2f}"


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------


def _near_pixels(depth: np.ndarray, mask: np.ndarray, rc: np.ndarray) -> np.ndarray | None:
    """Boolean image of the near-side valid mask pixels around rc, or None."""
    h, w = depth.shape
    for half in (DEPTH_WINDOW, 2 * DEPTH_WINDOW):
        r0, r1 = max(0, rc[0] - half), min(h, rc[0] + half + 1)
        c0, c1 = max(0, rc[1] - half), min(w, rc[1] + half + 1)
        d = depth[r0:r1, c0:c1]
        ok = mask[r0:r1, c0:c1] & np.isfinite(d) & (d >= MIN_DEPTH) & (d <= MAX_DEPTH)
        if ok.any():
            near = ok & (d <= np.percentile(d[ok], DEPTH_PERCENTILE))
            out = np.zeros_like(mask)
            out[r0:r1, c0:c1] = near
            return out
    return None


def _world_point(ctx: NodeContext, cam: dict[str, Any], pixels: np.ndarray) -> np.ndarray | None:
    reply = ctx.tool(
        "geometry.mask_to_world_points",
        mask=pixels.astype(np.uint8) * 255,
        depth=np.asarray(cam["depth"], dtype=np.float32),
        intrinsics=np.asarray(cam["intrinsics"], dtype=np.float64),
        camera_pose=cam["pose"],
    )
    pts = reply.get("points")
    pts = pts.get("points") if isinstance(pts, dict) else pts
    pts = np.asarray(pts if pts is not None else [], dtype=np.float64).reshape(-1, 3)
    return np.median(pts, axis=0) if len(pts) else None


# ---------------------------------------------------------------------------
# Drawing (maestro _sample_points_from_mask)
# ---------------------------------------------------------------------------


def _crop_box(mask: np.ndarray, margin: int) -> tuple[int, int, int, int, int]:
    """(r0, r1, c0, c1, scale): the mask's box plus margin, and an integer upscale."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    r0, r1 = max(0, int(ys.min()) - margin), min(h, int(ys.max()) + margin + 1)
    c0, c1 = max(0, int(xs.min()) - margin), min(w, int(xs.max()) + margin + 1)
    scale = max(1, min(h // (r1 - r0), w // (c1 - c0)))
    return r0, r1, c0, c1, scale


def _drop_overlaps(rc: np.ndarray, min_dist: float) -> np.ndarray:
    """In order, a point within min_dist of an already kept one goes."""
    kept: list[np.ndarray] = []
    for p in rc:
        if all(float(np.hypot(*(p - q))) >= min_dist for q in kept):
            kept.append(p)
    return np.array(kept, dtype=int).reshape(-1, 2)


def _draw_markers(
    rgb: np.ndarray, rc: np.ndarray, box: tuple[int, int, int, int, int], radius: int
) -> np.ndarray:
    r0, r1, c0, c1, scale = box
    crop = cv2.resize(
        np.ascontiguousarray(rgb[r0:r1, c0:c1]),
        ((c1 - c0) * scale, (r1 - r0) * scale),
        interpolation=cv2.INTER_LINEAR,
    )
    font = 0.6 * radius / MARKER_RADIUS
    thick = 2 if radius >= 12 else 1
    for i, (r, c) in enumerate(rc, 1):
        centre = (int((c - c0) * scale + scale // 2), int((r - r0) * scale + scale // 2))
        cv2.circle(crop, centre, radius, (255, 0, 0), -1)
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        cv2.putText(
            crop,
            label,
            (centre[0] - tw // 2, centre[1] + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )
    return crop


def _draw_selection(rgb: np.ndarray, mask: np.ndarray, chosen: list[tuple[int, np.ndarray]]):
    out = np.ascontiguousarray(rgb.copy())
    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    cv2.drawContours(out, contours, -1, (255, 220, 0), 1)
    for index, (r, c) in chosen:
        cv2.circle(out, (int(c), int(r)), 6, (0, 255, 0), 2)
        cv2.putText(
            out,
            str(index),
            (int(c) + 7, int(r) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# The VLM (maestro vlm.select_points)
# ---------------------------------------------------------------------------


def parse_selection(text: str, count: int) -> list[tuple[int, str]]:
    """[(index, label)] from the reply's outermost [...]; raises on no list or an
    index outside 1..count (upstream: the text between the first '[' and last ']')."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end < start:
        raise ValueError("the reply holds no list")
    raw = re.sub(r"//[^\n]*", "", text[start : end + 1])
    try:
        items = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"the reply's list is not JSON ({exc})") from exc
    if not isinstance(items, list):
        raise ValueError("the reply's list is not a list")
    out: list[tuple[int, str]] = []
    for item in items:
        if isinstance(item, dict):
            value, label = item.get("index"), str(item.get("label", ""))
        else:
            value, label = item, ""
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{value!r} is not a point number") from exc
        if not 1 <= index <= count:
            raise ValueError(f"point {index} was not offered (1..{count})")
        if index not in [i for i, _ in out]:
            out.append((index, label))
    return out


def _ask(ctx: NodeContext, tool: str, prompt: str, image: np.ndarray, model: str) -> str:
    kwargs: dict[str, Any] = {"prompt": prompt, "image": image}
    if model:
        kwargs["model"] = model
    reply = ctx.tool(tool, **kwargs)
    return str(reply.get("text", "") if isinstance(reply, dict) else reply)


def _vec(p: np.ndarray) -> dict[str, float]:
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    instruction: str,
    object_name: str = "",
    mask: Any = None,
    camera_name: str = "",
    target_min: int = TARGET_RANGE[0],
    target_max: int = TARGET_RANGE[1],
    boundary_count: int = BOUNDARY_SAMPLES,
    crop_margin: int = CROP_MARGIN,
    max_points: int = 0,
    vlm_tool: str = "vlm.query",
    vlm_model: str = "",
    attempts: int = ATTEMPTS,
) -> Output:
    cam = _camera(observation, camera_name)
    rgb = np.asarray(cam["rgb"], dtype=np.uint8)
    depth = np.asarray(cam["depth"], dtype=np.float64).reshape(rgb.shape[:2])
    out: Output = {
        "route": "not_seen",
        "detail": "",
        "point": {},
        "points": [],
        "selected": [],
        "pixel": [],
        "marker_count": 0,
        "marker_pixels": [],
        "marked_image": np.zeros((1, 1, 3), np.uint8),
        "selection_image": rgb,
        "reply": "",
    }

    def leave(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        return out

    # 1. The mask: given, else sam3 on the object's name.
    seg = _as_mask(mask, rgb.shape[:2])
    if seg is not None and seg.any():
        source = "given mask"
    elif mask is not None:
        return leave("not_seen", "the given mask is empty")
    elif object_name:
        seg, source = _segment(ctx, rgb, object_name)
        if seg is None:
            return leave("not_seen", f"no mask of {object_name!r}: {source}")
    else:
        raise ValueError("give a mask or an object_name to segment")
    if int(seg.sum()) < MIN_MASK_PIXELS:
        return leave("not_seen", f"the mask has {int(seg.sum())} pixels (< {MIN_MASK_PIXELS})")

    # 2. Markers: boundary samples, then the interior grid; those with depth are kept.
    interior, stride = best_grid(seg, (int(target_min), int(target_max)))
    boundary = boundary_samples(seg, int(boundary_count))
    box = _crop_box(seg, int(crop_margin))
    scale = box[4]
    radius = int(np.clip(0.45 * max(stride, 1) * scale, MIN_MARKER_RADIUS, MARKER_RADIUS))
    # Outline markers first (upstream's numbering); an interior one that would cover one goes.
    rc = np.concatenate([boundary, interior], axis=0) if len(boundary) else interior
    rc = _drop_overlaps(rc, 2.0 * radius / scale)
    near = [_near_pixels(depth, seg, p) for p in rc]
    keep = [i for i, n in enumerate(near) if n is not None]
    markers = rc[keep]
    near = [near[i] for i in keep]
    out["marker_count"] = len(markers)
    out["marker_pixels"] = [[float(c), float(r)] for r, c in markers]
    if not len(markers):
        return leave(
            "none_suitable", f"none of the {len(rc)} markers in the mask ({source}) has depth"
        )
    marked = _draw_markers(rgb, markers, box, radius)
    out["marked_image"] = marked

    # 3. Ask which marker(s) fit the instruction.
    prompt = (
        f"{SELECTION_PROMPT} {NONE_CLAUSE}\n"
        f"Points: {json.dumps(list(range(1, len(markers) + 1)))}\n"
        f"Instruction: {instruction}"
    )
    picks: list[tuple[int, str]] | None = None
    why = "no reply"
    for _ in range(max(1, int(attempts))):
        reply = _ask(ctx, vlm_tool, prompt, marked, vlm_model)
        out["reply"] = reply
        try:
            picks = parse_selection(reply, len(markers))
            break
        except ValueError as exc:
            why = f"the VLM's reply was unusable: {exc}"
    if picks is None:
        return leave("none_suitable", why)
    if not picks:
        return leave("none_suitable", f"the VLM chose none of the {len(markers)} markers")
    if int(max_points) > 0:
        picks = picks[: int(max_points)]

    # 4. World points of the chosen markers.
    selected = []
    for index, label in picks:
        p = _world_point(ctx, cam, near[index - 1])
        r, c = (int(v) for v in markers[index - 1])
        if p is None:
            continue
        selected.append(
            {"index": index, "label": label, "pixel": [float(c), float(r)], "position": _vec(p)}
        )
    out["selection_image"] = _draw_selection(
        rgb, seg, [(s["index"], markers[s["index"] - 1]) for s in selected]
    )
    if not selected:
        return leave("none_suitable", "the chosen markers did not back-project")
    out["selected"] = selected
    out["points"] = [s["position"] for s in selected]
    out["point"] = selected[0]["position"]
    out["pixel"] = selected[0]["pixel"]
    out["route"] = "selected"
    out["detail"] = (
        f"{len(selected)} of {len(markers)} markers ({source}, stride {stride} px): "
        + ", ".join(
            f"{s['index']} ({s['label']})" if s["label"] else str(s["index"]) for s in selected
        )
    )
    return out
