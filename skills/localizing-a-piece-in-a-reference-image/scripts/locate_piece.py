"""Find where one jigsaw piece goes in the box picture: rotation-searched masked NCC at the known scale.

Reimplemented for the hub (no code copied). The method is masked, zero-mean normalised
cross-correlation (OpenCV ``TM_CCOEFF_NORMED`` with a mask) of the piece crop against the
reference picture, searched over rotation at the one scale the grid and board size fix,
coarse-to-fine: every angle over the whole picture at half resolution, then a 2 degree
refinement in a window round the best cells at full resolution.
"""

from __future__ import annotations

import logging
import math
import zlib
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext
from gap_core.types import CameraFrame, Mask, Se3Pose

logger = logging.getLogger(__name__)

WORK_CELL_PX = 36.0
"""Width of one grid cell in the picture the fine search runs on [px]. The reference is
shrunk to this (never enlarged). 35 x 41 px cells were enough for 96 % of interior pieces in
the prior measurement on a 768 x 1024 reference of a 22 x 25 grid; larger only costs time."""

COARSE_FACTOR = 0.5
"""The coarse search runs on the fine picture shrunk by this (18 px cells): the cell is
still found there, at a quarter of the cost."""

COARSE_STEP_DEG = 10.0
"""Angle step of the coarse search. Masked NCC of an 18 px cell stays near its peak within
+-5 degrees on the synthetic set, so a 10 degree step does not lose the right cell."""

FINE_HALF_RANGE_DEG = 6.0
FINE_STEP_DEG = 2.0
"""The refinement tries the coarse angle +-6 degrees in 2 degree steps (covers half a coarse
step plus slack); the angle is then good to about 1-2 degrees."""

FINE_CELLS = 10
"""Cells (by coarse score) refined at full resolution; each with its best two coarse angles
at least ANGLE_SEPARATION_DEG apart."""

ANGLE_SEPARATION_DEG = 30.0

FINE_SLACK_CELLS = 0.5
"""The fine window lets the piece's centre move this far outside the coarse cell [cells]."""

PAD_CELLS = 1.0
"""The reference is padded by this much on every side so a border piece's template (whose
tabs and canvas reach past the picture) can be placed; the mask keeps the padding out of the
score, since a border piece's mask ends at its straight side."""

ERODE_FRACTION = 0.06
"""The mask is eroded by this fraction of a cell before matching (2 px at 36 px cells): the
cut edge of a real piece is darker and blurred, and the segmenter's outline wanders by a
pixel or two."""

MIN_SCORE = 0.45
"""Below this best NCC the piece is not in the picture (or the crop is not a piece):
``not_found``. Synthetic set: right-cell answers scored 0.55-0.99 (median 0.93); the
wrong-cell answers that scored highest were flat white-band pieces near 0.5-0.9."""

MIN_MARGIN = 0.04
"""Best score minus the best score of any other cell. Below this the answer is
``ambiguous`` and the candidates are returned for a shape check. On the synthetic set every
right-cell answer with margin >= 0.04 was right; most flat-region pieces fall below it."""

MIN_MARGIN_GUESSED_SCALE = 0.10
"""``min_margin`` when the scale came from the mask area: with the scale off by a few
percent a wrong cell can win by more. Synthetic set, area scale: wrong ``located`` answers had
margins 0.043-0.081 and one 0.139; local reference 0.045-0.074."""

AREA_SCALE_SET = (0.94, 1.0, 1.06)
"""Scales tried at the fine stage when the crop's scale had to be guessed from the mask area
(no depth, no ``crop_px_per_m``): a piece's area varies about +-12 % with its tabs and
blanks, so its side about +-6 %."""

AREA_PER_CELL = 1.0
"""Piece area divided by cell area, for that guess. Two tabs and two blanks cancel; the
synthetic generator's pieces average 1.0 (0.87-1.13)."""

LOW_TEXTURE_STD = 6.0
"""Masked grey-level standard deviation (0-255) under which a piece is flagged
``low_texture`` in the output: white snow, black cloth. Appearance cannot place those."""

FLAT_CELL_STD = 2.0 * LOW_TEXTURE_STD
"""A reference cell whose central 70 % has a grey standard deviation under this counts as
flat, a home for a low-texture piece."""

FLAT_COLOUR_DIST = 90.0
"""Largest RGB distance between a low-texture piece's median colour and a flat cell's mean
for the cell to stay a candidate. Covers a +-20 % per-channel gain between print and camera
on white (238 reads 190-255 in each channel, up to 83 apart); at 60, 2 of 22 synthetic
white-band pieces lost every candidate."""

MAX_FLAT_CANDIDATES = 200

TOP_K = 5

_REF_CACHE: dict[tuple, dict[str, Any]] = {}


class Output(TypedDict):
    route: str
    col: int
    row: int
    angle_deg: float
    position_px: dict[str, float]
    position_m: dict[str, float]
    cell_center_m: dict[str, float]
    score: float
    margin: float
    candidates: list[dict[str, Any]]
    low_texture: bool
    texture_std: float
    grip_pixel: dict[str, float]
    piece_pose: Se3Pose | None
    detail: str


def _rgb(image: Any) -> np.ndarray:
    if isinstance(image, str):
        bgr = cv2.imread(image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read reference image {image!r}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[:, :, :3])


def _prepare_reference(
    reference: Any, cols: int, rows: int, board_w: float, board_h: float
) -> dict[str, Any]:
    if isinstance(reference, str):
        key: tuple = (reference, cols, rows, board_w, board_h)
    else:
        arr = np.ascontiguousarray(reference)
        key = (arr.shape, zlib.crc32(arr.tobytes()), cols, rows, board_w, board_h)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    native = _rgb(reference)
    h, w = native.shape[:2]
    ppm_native = w / board_w
    if abs((h / board_h) / ppm_native - 1.0) > 0.03:
        logger.warning(
            "reference %dx%d px does not have the board's aspect %.3f x %.3f m; it is stretched",
            w,
            h,
            board_w,
            board_h,
        )
    ppm = min(ppm_native, WORK_CELL_PX * cols / board_w)
    size = (round(board_w * ppm), round(board_h * ppm))
    fine = cv2.resize(native, size, interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(
        fine,
        (round(size[0] * COARSE_FACTOR), round(size[1] * COARSE_FACTOR)),
        interpolation=cv2.INTER_AREA,
    )
    cell_fine = (size[0] / cols, size[1] / rows)
    levels = {}
    for name, img, factor in (("fine", fine, 1.0), ("coarse", coarse, COARSE_FACTOR)):
        cw, ch = cell_fine[0] * factor, cell_fine[1] * factor
        pad = int(math.ceil(PAD_CELLS * max(cw, ch)))
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        levels[name] = {
            "img": padded,
            "grey": cv2.cvtColor(padded, cv2.COLOR_RGB2GRAY),
            "pad": pad,
            "cell": (cw, ch),
            "ppm": ppm * factor,
        }
    cw, ch = cell_fine
    flat = np.zeros((rows, cols), np.float32)
    mean = np.zeros((rows, cols, 3), np.float32)
    grey = cv2.cvtColor(fine, cv2.COLOR_RGB2GRAY)
    for r in range(rows):
        for c in range(cols):
            ys = slice(round((r + 0.15) * ch), round((r + 0.85) * ch))
            xs = slice(round((c + 0.15) * cw), round((c + 0.85) * cw))
            flat[r, c] = grey[ys, xs].std()
            mean[r, c] = fine[ys, xs].reshape(-1, 3).mean(axis=0)
    prepared = {
        "cell_std": flat,
        "cell_mean": mean,
        "levels": levels,
        "native_size": (w, h),
        "ppm": ppm,
        "cols": cols,
        "rows": rows,
        "board": (board_w, board_h),
    }
    if len(_REF_CACHE) > 4:
        _REF_CACHE.clear()
    _REF_CACHE[key] = prepared
    return prepared


def _anchor(mask: np.ndarray) -> tuple[float, float]:
    """The piece's inner centre: the middle of the region farthest from its outline.
    Tabs are narrower than the body, so this sits near the body's centre (the centroid is
    pulled toward the tabs) and is also the safest point to grip."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    top = dist >= 0.8 * dist.max()
    ys, xs = np.nonzero(top)
    return float(xs.mean()), float(ys.mean())


def _rotated(
    crop: np.ndarray, mask: np.ndarray, anchor: tuple[float, float], rot_deg: float, erode: int
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Rotate the crop about the anchor (CCW as displayed for positive angles) and trim to the
    rotated mask. Returns template, mask and the anchor's position inside the template."""
    ys, xs = np.nonzero(mask)
    radius = int(math.ceil(np.hypot(xs - anchor[0], ys - anchor[1]).max())) + 2
    side = 2 * radius + 1
    m = cv2.getRotationMatrix2D(anchor, rot_deg, 1.0)
    m[0, 2] += radius - anchor[0]
    m[1, 2] += radius - anchor[1]
    tpl = cv2.warpAffine(crop, m, (side, side), flags=cv2.INTER_LINEAR)
    tm = cv2.warpAffine(mask.astype(np.uint8) * 255, m, (side, side), flags=cv2.INTER_NEAREST)
    if erode > 0:
        tm = cv2.erode(tm, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    ys, xs = np.nonzero(tm)
    if len(xs) < 16:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return (
        np.ascontiguousarray(tpl[y0:y1, x0:x1]),
        np.ascontiguousarray(tm[y0:y1, x0:x1]),
        float(radius - x0),
        float(radius - y0),
    )


def _match(img: np.ndarray, tpl: np.ndarray, tm: np.ndarray) -> np.ndarray:
    res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED, mask=tm)
    return np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)


def _cell_max(
    res: np.ndarray, ox: float, oy: float, level: dict, cols: int, rows: int
) -> np.ndarray:
    """Best score per grid cell, by where the template's anchor lands."""
    pad = level["pad"]
    cw, ch = level["cell"]
    out = np.full((rows, cols), -np.inf, dtype=np.float32)
    cx = np.floor((np.arange(res.shape[1]) + ox - pad) / cw).astype(int)
    cy = np.floor((np.arange(res.shape[0]) + oy - pad) / ch).astype(int)
    kx = (cx >= 0) & (cx < cols)
    ky = (cy >= 0) & (cy < rows)
    if not kx.any() or not ky.any():
        return out
    sub = res[np.ix_(ky, kx)]
    cxk, cyk = cx[kx], cy[ky]
    starts_x = np.flatnonzero(np.r_[True, np.diff(cxk) != 0])
    starts_y = np.flatnonzero(np.r_[True, np.diff(cyk) != 0])
    reduced = np.maximum.reduceat(np.maximum.reduceat(sub, starts_x, axis=1), starts_y, axis=0)
    out[np.ix_(cyk[starts_y], cxk[starts_x])] = reduced
    return out


def _wrap(deg: float) -> float:
    return float((deg + 180.0) % 360.0 - 180.0)


def _crop_from_camera(
    camera: CameraFrame, piece_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float | None, tuple[int, int]]:
    rgb = _rgb(camera["rgb"])
    mask = np.asarray(piece_mask).astype(bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"piece_mask {mask.shape} does not match the camera image {rgb.shape[:2]}")
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("piece_mask is empty")
    m = 4
    x0, x1 = max(0, xs.min() - m), min(rgb.shape[1], xs.max() + 1 + m)
    y0, y1 = max(0, ys.min() - m), min(rgb.shape[0], ys.max() + 1 + m)
    px_per_m = None
    depth = camera.get("depth")
    intr = camera.get("intrinsics")
    if depth is not None and intr is not None:
        d = np.asarray(depth, dtype=np.float64)[mask]
        d = d[np.isfinite(d) & (d > 0)]
        if len(d) >= 0.2 * len(xs):
            k = np.asarray(intr, dtype=np.float64)
            px_per_m = float(0.5 * (k[0, 0] + k[1, 1]) / np.median(d))
    return rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1], px_per_m, (int(x0), int(y0))


def run(
    ctx: NodeContext,
    piece_mask: Mask,
    reference: Any,
    grid_cols: int,
    grid_rows: int,
    board_width: float,
    board_height: float,
    camera: CameraFrame | None = None,
    piece_rgb: Any = None,
    crop_px_per_m: float | None = None,
    piece_pose: Se3Pose | None = None,
    search_cells: list | None = None,
    allowed_angles: list | None = None,
    top_k: int = TOP_K,
    min_score: float = MIN_SCORE,
    min_margin: float = MIN_MARGIN,
) -> Output:
    cols, rows = int(grid_cols), int(grid_rows)
    if cols < 1 or rows < 1 or board_width <= 0 or board_height <= 0:
        raise ValueError("grid_cols, grid_rows, board_width and board_height must be positive")
    if piece_rgb is not None:
        crop, mask = _rgb(piece_rgb), np.asarray(piece_mask).astype(bool)
        if mask.shape != crop.shape[:2]:
            raise ValueError(f"piece_mask {mask.shape} does not match piece_rgb {crop.shape[:2]}")
        offset, depth_ppm = (0, 0), None
    elif camera is not None:
        crop, mask, depth_ppm, offset = _crop_from_camera(camera, piece_mask)
    else:
        raise ValueError("give the wrist camera frame (camera) or a piece crop (piece_rgb)")
    if mask.sum() < 64:
        raise ValueError(f"piece_mask has only {int(mask.sum())} pixels")

    ref = _prepare_reference(reference, cols, rows, float(board_width), float(board_height))
    fine, coarse = ref["levels"]["fine"], ref["levels"]["coarse"]
    cell_m = (board_width / cols, board_height / rows)

    scales = (1.0,)
    if crop_px_per_m:
        src_ppm = float(crop_px_per_m)
        scale_from = "crop_px_per_m"
    elif depth_ppm:
        src_ppm = depth_ppm
        scale_from = "depth"
    else:
        area_m2 = AREA_PER_CELL * cell_m[0] * cell_m[1]
        src_ppm = math.sqrt(float(mask.sum()) / area_m2)
        scales = AREA_SCALE_SET
        scale_from = "mask area"

    # Crop at the fine picture's scale; the anchor and texture are measured there.
    factor = fine["ppm"] / src_ppm
    size = (max(4, round(crop.shape[1] * factor)), max(4, round(crop.shape[0] * factor)))
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    crop_f = cv2.resize(crop, size, interpolation=interp)
    mask_f = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    if mask_f.sum() < 16:
        raise ValueError("the piece is too small at the reference's scale")
    anchor_f = _anchor(mask_f)
    anchor_cam = _anchor(mask)
    grey = cv2.cvtColor(crop_f, cv2.COLOR_RGB2GRAY)
    inner = cv2.erode(mask_f.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    texture = float(grey[inner if inner.any() else mask_f].std())

    if allowed_angles:
        rotations = sorted({_wrap(-float(a)) for a in allowed_angles})
    else:
        n = int(round(360.0 / COARSE_STEP_DEG))
        rotations = [_wrap(i * COARSE_STEP_DEG) for i in range(n)]

    allowed = None
    if search_cells:
        allowed = np.zeros((rows, cols), bool)
        for c, r in search_cells:
            if 0 <= int(c) < cols and 0 <= int(r) < rows:
                allowed[int(r), int(c)] = True
        if not allowed.any():
            raise ValueError("search_cells names no cell inside the grid")

    # Coarse: every rotation over the whole picture, best score per cell.
    crop_c = cv2.resize(
        crop_f,
        (max(4, round(size[0] * COARSE_FACTOR)), max(4, round(size[1] * COARSE_FACTOR))),
        interpolation=cv2.INTER_AREA,
    )
    mask_c = cv2.resize(
        mask_f.astype(np.uint8), crop_c.shape[1::-1], interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    anchor_c = (anchor_f[0] * COARSE_FACTOR, anchor_f[1] * COARSE_FACTOR)
    erode_c = max(1, round(ERODE_FRACTION * coarse["cell"][0]))
    per_rot = np.full((len(rotations), rows, cols), -np.inf, dtype=np.float32)
    crop_c = cv2.cvtColor(crop_c, cv2.COLOR_RGB2GRAY)
    coarse_img = coarse["grey"]
    for i, rot in enumerate(rotations):
        t = _rotated(crop_c, mask_c, anchor_c, rot, erode_c)
        if t is None or t[0].shape[0] > coarse_img.shape[0] or t[0].shape[1] > coarse_img.shape[1]:
            continue
        per_rot[i] = _cell_max(_match(coarse_img, t[0], t[1]), t[2], t[3], coarse, cols, rows)
    if allowed is not None:
        per_rot[:, ~allowed] = -np.inf
    cell_best = per_rot.max(axis=0)
    order = np.argsort(cell_best, axis=None)[::-1]
    seeds: list[tuple[int, int, float]] = []
    for flat in order[:FINE_CELLS]:
        r, c = divmod(int(flat), cols)
        if not np.isfinite(cell_best[r, c]):
            break
        picked: list[float] = []
        for i in np.argsort(per_rot[:, r, c])[::-1]:
            rot = rotations[int(i)]
            if not np.isfinite(per_rot[i, r, c]):
                break
            if all(abs(_wrap(rot - p)) >= ANGLE_SEPARATION_DEG for p in picked):
                picked.append(rot)
                seeds.append((c, r, rot))
            if len(picked) == 2:
                break
    if not seeds:
        raise ValueError("no placement of the piece fits inside the reference")

    # Fine: refine angle (and scale, when guessed) in a window round each seed cell.
    erode_f = max(1, round(ERODE_FRACTION * fine["cell"][0]))
    offsets = np.arange(-FINE_HALF_RANGE_DEG, FINE_HALF_RANGE_DEG + 1e-9, FINE_STEP_DEG)
    cw, ch = fine["cell"]
    pad = fine["pad"]
    big = fine["img"]
    templates: dict[tuple[float, float], Any] = {}
    best_by_cell: dict[tuple[int, int], dict[str, Any]] = {}
    for c, r, rot0 in seeds:
        for s in scales:
            for d in offsets:
                rot = _wrap(rot0 + float(d))
                key = (round(rot, 3), s)
                if key not in templates:
                    if s == 1.0:
                        cf, mf, af = crop_f, mask_f, anchor_f
                    else:
                        sz = (max(4, round(size[0] * s)), max(4, round(size[1] * s)))
                        cf = cv2.resize(crop_f, sz, interpolation=cv2.INTER_LINEAR)
                        mf = (
                            cv2.resize(mask_f.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST)
                            > 0
                        )
                        af = (anchor_f[0] * s, anchor_f[1] * s)
                    templates[key] = _rotated(cf, mf, af, rot, erode_f)
                t = templates[key]
                if t is None:
                    continue
                tpl, tm, ox, oy = t
                ax0 = pad + (c - FINE_SLACK_CELLS) * cw
                ax1 = pad + (c + 1 + FINE_SLACK_CELLS) * cw
                ay0 = pad + (r - FINE_SLACK_CELLS) * ch
                ay1 = pad + (r + 1 + FINE_SLACK_CELLS) * ch
                x0 = max(0, int(math.floor(ax0 - ox)))
                y0 = max(0, int(math.floor(ay0 - oy)))
                x1 = min(big.shape[1], int(math.ceil(ax1 - ox)) + tpl.shape[1])
                y1 = min(big.shape[0], int(math.ceil(ay1 - oy)) + tpl.shape[0])
                if x1 - x0 < tpl.shape[1] or y1 - y0 < tpl.shape[0]:
                    continue
                res = _match(big[y0:y1, x0:x1], tpl, tm)
                _, score, _, loc = cv2.minMaxLoc(res)
                px = x0 + loc[0] + ox - pad
                py = y0 + loc[1] + oy - pad
                cc, rr = int(math.floor(px / cw)), int(math.floor(py / ch))
                if not (0 <= cc < cols and 0 <= rr < rows):
                    continue
                if allowed is not None and not allowed[rr, cc]:
                    continue
                prev = best_by_cell.get((cc, rr))
                if prev is None or score > prev["score"]:
                    best_by_cell[(cc, rr)] = {
                        "col": cc,
                        "row": rr,
                        "angle_deg": _wrap(-rot),
                        "score": float(score),
                        "scale": s,
                        "px": px,
                        "py": py,
                    }

    if not best_by_cell:
        raise ValueError("no refined placement landed inside the grid")
    ranked = sorted(best_by_cell.values(), key=lambda b: -b["score"])
    best = ranked[0]
    runner = ranked[1]["score"] if len(ranked) > 1 else -1.0
    margin = float(best["score"] - runner)
    to_native = ref["native_size"][0] / (ref["ppm"] * board_width)

    def describe(b: dict[str, Any]) -> dict[str, Any]:
        return {
            "col": b["col"],
            "row": b["row"],
            "angle_deg": round(b["angle_deg"], 2),
            "score": round(b["score"], 4),
            "position_px": {"x": b["px"] * to_native, "y": b["py"] * to_native},
            "position_m": {"x": b["px"] / ref["ppm"], "y": b["py"] / ref["ppm"]},
        }

    candidates = [describe(b) for b in ranked[: max(1, int(top_k))]]
    low_texture = texture < LOW_TEXTURE_STD
    if low_texture:
        # NCC on a flat patch correlates noise; shortlist the flat cells of the piece's colour.
        core = cv2.erode(mask_f.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        piece_mean = np.median(crop_f[core if core.any() else mask_f].reshape(-1, 3), axis=0)
        dist = np.linalg.norm(ref["cell_mean"] - piece_mean, axis=2)
        ok = (ref["cell_std"] < FLAT_CELL_STD) & (dist < FLAT_COLOUR_DIST)
        if allowed is not None:
            ok &= allowed
        flat_cells = sorted(zip(*np.nonzero(ok), strict=True), key=lambda rc: dist[rc])
        candidates = [
            {
                "col": int(c),
                "row": int(r),
                "angle_deg": None,
                "score": None,
                "colour_distance": round(float(dist[r, c]), 1),
                "position_m": {"x": (c + 0.5) * cell_m[0], "y": (r + 0.5) * cell_m[1]},
            }
            for r, c in flat_cells[:MAX_FLAT_CANDIDATES]
        ]
        route = "ambiguous" if candidates else "not_found"
    elif best["score"] < min_score:
        route = "not_found"
    elif margin < (max(min_margin, MIN_MARGIN_GUESSED_SCALE) if len(scales) > 1 else min_margin):
        route = "ambiguous"
    else:
        route = "located"
    detail = (
        f"{route}: cell ({best['col']}, {best['row']}) at {best['angle_deg']:.1f} deg, "
        f"score {best['score']:.3f}, margin {margin:.3f} over the next cell; scale from "
        f"{scale_from}; {len(rotations)} coarse angles, {len(seeds)} refined seeds"
        + (
            f"; low texture: appearance cannot place this piece, {len(candidates)} flat cells "
            "of its colour are the candidates"
            if low_texture
            else ""
        )
    )
    logger.info(detail)
    top = describe(best)
    return Output(
        route=route,
        col=int(best["col"]),
        row=int(best["row"]),
        angle_deg=float(top["angle_deg"]),
        position_px=top["position_px"],
        position_m=top["position_m"],
        cell_center_m={"x": (best["col"] + 0.5) * cell_m[0], "y": (best["row"] + 0.5) * cell_m[1]},
        score=float(best["score"]),
        margin=margin,
        candidates=candidates,
        low_texture=bool(low_texture),
        texture_std=texture,
        grip_pixel={"x": anchor_cam[0] + offset[0], "y": anchor_cam[1] + offset[1]},
        piece_pose=piece_pose,
        detail=detail,
    )
