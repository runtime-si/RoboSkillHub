# Adapted from Jigsaw-Puzzle-Building-Robot by JP Strydom
# (https://github.com/JPStrydom/Jigsaw-Puzzle-Building-Robot, commit
# be867bda743740fac9d843f606b9211e663777f7), MIT License, Copyright (c) 2016 JP Strydom; the
# license is LICENSE.jigsaw-puzzle-building-robot beside this skill. Taken from
# code/software/Functions_1.py:
# - Clear_Background: the staging surface is removed by an HSV colour band (kept as
#   ``background_hsv``, the source's calibration values work unchanged) and a 1 % frame border
#   is not trusted;
# - Binarise: 3 x 3 Gaussian blur, then everything not background is foreground;
# - Detect_Pieces: external contours sorted by area and the rotated-box aspect into pieces
#   (piece_min..piece_max area, aspect under 1.75), overlapping blobs (up to 3 x piece_max,
#   aspect under 3.6) and foreign objects (anything else over 1/15 of a piece);
# - Separate_Overlap: for a blob that stays a blob, a robust (Huber) centre line is fitted and
#   walked for points at least a gripper radius inside the outline (30 px, shrinking by 5 to
#   15 px), the end farther from the centroid is where to take it from, pointing away from the
#   centroid; and an open area at least 0.8 piece wide is where to put it;
# - Classify_Pieces: the piece's square-side angle is the mean of the largest group (within
#   0.35 rad) of Hough lines on its outline, modulo 90 degrees.
# Changed: the calibration constants (absolute pixel areas, an HSV band per mat, 30 px gripper
# radius) are scaled to the piece size, taken from depth and ``piece_size_m`` or from the median
# blob; the default background model is estimated from the frame (the mat's chromaticity and
# brightness, tolerant to shadows) instead of a fixed HSV band; holes left by mat-coloured print
# are filled; Python 2 / OpenCV 2 calls (cv2.cv.CV_DIST_HUBER, xrange) are updated; the
# Hough-angle groups wrap at 90 degrees (the source's grouping did not wrap); the open-area
# search uses the free space's distance transform instead of a coarse grid scan.
# New: touching pieces are split (distance-transform peaks as markers, a priority-flood
# watershed on the distance transform, the source only moved such blobs aside); the pick point
# is the largest inscribed circle's centre (distance-transform maximum) instead of the
# contour centroid; touching-neighbour flags and gaps; a two-finger grip check across the
# piece's sides; a face-up / face-down cue from texture (the source has none); lifting to the
# world with the hub's geometry tools.
# Not ported: the perspective correction from corner markers (the camera's calibration
# replaces it), the tab / blank classification and the solver (Functions_2), the CNC commands
# (Functions_3).
"""Segment loose jigsaw pieces on a staging mat, split touching ones, and pick each at its incenter."""

from __future__ import annotations

import heapq
import logging
import math
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext
from gap_core.errors import ToolError
from gap_core.types import Mask, Vec3, pose_to_matrix

logger = logging.getLogger(__name__)

EDGE_BLACKOUT = 0.01
"""Fraction of the frame on each side that is not trusted (source: edge_blackout = 0.01).
Blobs reaching it are flagged ``clipped`` (part of the piece may be out of view) instead of
being blacked out."""

BLUR_KSIZE = 3
"""Gaussian kernel of the binarisation (source: Binarise, (3, 3))."""

PIECE_AREA_RANGE = (0.55, 1.5)
"""A blob is one piece when its area is this fraction of the nominal piece area. Source:
15000-22000 px for its 18500 px pieces (0.81-1.19), one puzzle at one scale; widened for wavy
random cuts (areas 0.87-1.13 of the cell on the hub's cut generator), a cell size guessed from
the median blob, and masks that lose a pixel at 20 px per piece (synthetic set: no single piece
fell outside 0.62-1.38)."""

OVERLAP_MAX = 3.0
"""Largest blob, in units of the largest single piece, that is taken for touching or
overlapping pieces (source: overlap_max_area = 3 * piece_max_area). Bigger blobs are foreign:
the arm, a tool, a heap that must be spread first."""

FOREIGN_MIN = 1.0 / 15.0
"""Blobs under this fraction of a piece are noise and dropped (source: min_fo_area = 1000 px
for 15000 px pieces)."""

MAX_ASPECT = 1.75
"""Longest over shortest side of a single piece's rotated box (source: min_wh_ratio 0.575 /
max_wh_ratio 1.75 on the upright box; the rotated box does not grow with the piece's turn)."""

MIN_SINGLES = 3
"""Single pieces needed in the frame before their median area replaces the nominal one (from
``piece_size_m`` or the blobs) for the size ranges and for counting the pieces in a blob."""

MAX_ASPECT_OVERLAP = 3.6
"""Same for a blob of touching pieces (source: max_wh_ratio_overlap = 3.6)."""

MIN_PEAK_FRACTION = 0.12
"""A distance-transform peak seeds a piece only if it is at least this fraction of the piece
side from the outline. A tab's head (radius 0.13-0.17 of the side on the cut generator) can
pass; the body's peak (0.23-0.40) is taller, and only the n tallest separated peaks are kept."""

CUT_LINE_WEIGHT = 2.0
"""Weight of the dark cut line against the distance transform (in piece sides) in the
watershed's relief. Synthetic set at 48 px, seeds 1000-1035: weights 0 / 0.3 / 1 / 2 / 4 split
14 / 28 / 39 / 53 / 58 % of still-interlocked pairs (touching pairs 96-99 % throughout). The
synthetic cut edge is 35 % darker for one pixel; a real cut shows a dark gap too, but this was
not measured on real interlocked pieces."""

CUT_LINE_CONTRAST = 40.0
"""Black-hat grey levels that count as a full cut line."""

PEAK_SEPARATION = 0.65
"""Seeds are at least this fraction of a piece side apart (two bodies side by side are at
least 0.8 apart). Synthetic set at 48 px, seeds 1000-1035: 0.55 / 0.65 / 0.75 split 98.6 / 100 /
100 % of touching pairs and 64 / 67 / 56 % of still-interlocked pairs."""

GRIP_RADIUS = (0.22, 0.11)
"""Inscribed-circle radius a pick point should have, wanted and least, as fractions of the
piece side (source: man_rad = 30 px, min_man_rad = 15 px on ~136 px pieces). Under the least
the piece is not ``pickable``: a sliver, or a split that went wrong."""

DRAG_RADIUS_STEP = 5.0 / 136.0
"""Shrink of the gripper radius per centre-line search pass (source: 5 px on ~136 px pieces)."""

FREE_SPOT_CLEARANCE = 0.8
"""An open area must be clear this far, in piece sides, from its centre (source: a 220 px
empty square, search_radius = 110 px, on ~136 px pieces)."""

HOUGH_VOTES = 0.22
"""Hough line votes as a fraction of the piece side (source: 30 votes on ~136 px pieces)."""

ANGLE_GROUP_RAD = 0.35
"""Hough lines within this of a group's mean join it (source: ang_dev = 0.35 rad)."""

NOMINAL_PIECE_M = 0.0225
"""Piece side when nothing else gives the scale of the fingers [m]: the target puzzle's
~5.07 cm^2 per piece (550 pieces, 45.72 x 60.96 cm)."""

FINGER_RADIUS_M = 0.004
"""Half the width of a fingertip where it meets the table [m]. Assumed for the YAM's tapered
linear_4310 fingertips, not measured: pass ``finger_radius_m``."""

FINGER_MARGIN_M = 0.002
"""Gap left between the piece's edge and a fingertip on the way down [m]. Assumed."""

TOUCH_GAP_M = 0.001
"""Two pieces closer than this touch [m]; 1 px when there is no scale."""

LOW_TEXTURE_STD = 8.0
"""Robust grey-level spread (0.7413 x interquartile range) inside the piece, eroded by a tenth
of its side, under which its face is flat: the plain cardboard back, or flat art (white snow).
Synthetic backs read 8-11 as a plain standard deviation with a 2 px erosion (lighting
gradient, edge pixels); localizing-a-piece-in-a-reference-image uses 6 for its flat-art test."""

BACK_CHROMA_DIST = 0.015
BACK_BRIGHTNESS_RANGE = (0.6, 1.4)
"""A flat piece is face down when its median colour's rg chromaticity is within
BACK_CHROMA_DIST of ``back_rgb``'s and its brightness within this ratio of it (lighting
varies; the hue of plain cardboard does not much). Flat white art (238, 238, 238) is 0.025
from a (170, 160, 145) back: at 0.03, 6 % of face-up synthetic pieces read face down."""

STACK_HEIGHT_M = 0.003
"""Height over the table above which a piece lies on another [m]: a piece is ~2 mm thick, so
a stacked one is ~4 mm up; D405 depth at 15-30 cm is good to a fraction of a mm. Overhead
depth at 0.9 m (~3 mm noise) cannot tell, and is not used (see HEIGHT_NOISE_MAX_M)."""

HEIGHT_FG_M = 0.001
"""Height over the table above which a pixel is a piece whatever its colour [m]: half a
piece's thickness."""

HEIGHT_NOISE_MAX_M = 0.0006
"""Depth is used to find and stack pieces only when the mat's height noise (1.4826 x MAD,
before the 5 x 5 median) is under this [m]; the median then leaves ~0.25 mm, a quarter of
HEIGHT_FG_M. A D405 at 15-30 cm qualifies; the overhead Orbbec at 0.9 m (2-3 mm) does not."""

STACK_FRACTION = (0.15, 0.9)
"""A blob whose raised part (over STACK_HEIGHT_M) is this fraction of it is a stack."""

COVERED_MIN = 0.25
"""The covered piece's visible part is reported (not pickable) when at least this fraction of
a piece."""

CHROMA_TOL = 0.05
"""Background when the pixel's rg chromaticity is within this of the mat's (distance scaled by
min(brightness, mat brightness) / mat brightness, so dark pixels' noisy chromaticity counts
less). Synthetic mats: mat pixels under shadow and a 0.75-1.1 lighting gradient stay within
0.02; printed pieces mostly differ by more than 0.08."""

BRIGHTNESS_RANGE = (0.35, 1.45)
"""First pass: background brightness relative to the mat's median, shadows down to 35 %, the
lighting gradient up to 145 %."""

LOCAL_BRIGHTNESS_MAX = 1.15
"""Second pass: brightest background relative to the mat's brightness nearby (a Gaussian of
the lit mat pixels, sigma a tenth of the frame). A cardboard back (170, 160, 145) on a grey
mat (120) is 1.32: foreground; mat texture and noise stay under 1.1."""

DARK_MAT = 45.0
"""A mat darker than this (mean of R, G, B) has no usable chromaticity (black felt): only
brightness separates it, foreground above max(2.2 x mat, mat + 30)."""


class Output(TypedDict):
    route: str
    n_pieces: int
    n_touching: int
    n_pickable: int
    n_clusters: int
    pieces: list[dict[str, Any]]
    clusters: list[dict[str, Any]]
    foreign: list[dict[str, Any]]
    piece_index: int
    piece_mask: Mask
    pick_point: Vec3 | None
    pick_pixel: dict[str, float] | None
    angle_deg: float
    piece_area_px: float
    px_per_m: float | None
    table_z: float | None
    free_spot: dict[str, Any] | None
    labels: np.ndarray
    background: dict[str, Any]
    detail: str


# --------------------------------------------------------------------------- inputs


def _camera(observation: Any, camera: Any, camera_name: str) -> dict[str, Any]:
    if camera is not None:
        return dict(camera)
    if observation is None:
        raise ValueError("give observation or camera")
    cams = observation.get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} for k, v in cams.items()]
    if camera_name:
        for c in cams:
            if c.get("name") == camera_name:
                return dict(c)
        raise ValueError(f"observation has no camera named {camera_name!r}")
    if not cams:
        raise ValueError("observation has no cameras")
    return dict(cams[0])


def _rgb(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[:, :, :3])


# --------------------------------------------------------------------------- background


def mat_model(rgb: np.ndarray, roi: np.ndarray) -> dict[str, Any]:
    """The mat's colour: the most populated (chromaticity, brightness) bin of the region."""
    f = rgb.astype(np.float32)
    s = f.sum(axis=2) + 1e-3
    r, g, bright = f[:, :, 0] / s, f[:, :, 1] / s, s / 3.0
    sel = roi.copy()
    step = max(1, int(math.sqrt(sel.sum() / 40000.0)))
    sub = np.zeros_like(sel)
    sub[::step, ::step] = True
    sel &= sub
    rv, gv, iv = r[sel], g[sel], bright[sel]
    if rv.size < 50:
        raise ValueError("region too small to estimate the mat")
    hist, edges = np.histogramdd(
        np.column_stack([rv, gv, iv]), bins=(25, 25, 8), range=((0, 1), (0, 1), (0, 256))
    )
    i, j, k = np.unravel_index(int(np.argmax(hist)), hist.shape)
    near = (
        (np.abs(rv - (edges[0][i] + edges[0][i + 1]) / 2) <= 0.04)
        & (np.abs(gv - (edges[1][j] + edges[1][j + 1]) / 2) <= 0.04)
        & (np.abs(iv - (edges[2][k] + edges[2][k + 1]) / 2) <= 48)
    )
    return {
        "chroma": [float(np.median(rv[near])), float(np.median(gv[near]))],
        "brightness": float(np.median(iv[near])),
        "rgb": [float(np.median(f[:, :, c][sel][near])) for c in range(3)],
        "fraction": float(near.mean()),
    }


def background_mask(
    rgb: np.ndarray, roi: np.ndarray, background_hsv: dict | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """True where the pixel is the staging surface."""
    if background_hsv:
        lo = np.array(background_hsv["lower"], dtype=np.uint8)
        hi = np.array(background_hsv["upper"], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return cv2.inRange(hsv, lo, hi) > 0, {"mode": "hsv", **background_hsv}
    m = mat_model(rgb, roi)
    f = rgb.astype(np.float32)
    s = f.sum(axis=2) + 1e-3
    bright = s / 3.0
    mb = max(m["brightness"], 1.0)
    if mb < DARK_MAT:
        return bright < max(2.2 * mb, mb + 30.0), {"mode": "auto", **m}
    dc = np.hypot(f[:, :, 0] / s - m["chroma"][0], f[:, :, 1] / s - m["chroma"][1])
    dc *= np.minimum(bright, mb) / mb
    ratio = bright / mb
    bg = (dc < CHROMA_TOL) & (ratio > BRIGHTNESS_RANGE[0]) & (ratio < BRIGHTNESS_RANGE[1])
    # The lighting varies over the frame: compare with the mat's own brightness nearby, from
    # the first estimate's mat pixels (lit ones only, shadows would pull it down).
    lit = (bg & roi & (ratio > 0.8)).astype(np.float32)
    h, w = rgb.shape[:2]
    ds = max(1, max(h, w) // 160)  # the blur runs on a small copy
    small = (max(1, w // ds), max(1, h // ds))
    sigma = max(h, w) / 10.0 / ds
    num = cv2.GaussianBlur(
        cv2.resize(bright * lit, small, interpolation=cv2.INTER_AREA), (0, 0), sigma
    )
    den = cv2.GaussianBlur(cv2.resize(lit, small, interpolation=cv2.INTER_AREA), (0, 0), sigma)
    num = cv2.resize(num, (w, h), interpolation=cv2.INTER_LINEAR)
    den = cv2.resize(den, (w, h), interpolation=cv2.INTER_LINEAR)
    local = np.where(den > 0.05, num / np.maximum(den, 1e-6), mb)
    ratio = bright / np.maximum(local, 1.0)
    bg = (dc < CHROMA_TOL) & (ratio > BRIGHTNESS_RANGE[0]) & (ratio < LOCAL_BRIGHTNESS_MAX)
    return bg, {"mode": "auto", **m}


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, cnts, -1, 1, thickness=cv2.FILLED)
    return out > 0


def foreground(
    rgb: np.ndarray,
    roi: np.ndarray,
    background_hsv: dict | None,
    side_px: float | None,
    height: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    bg, model = background_mask(rgb, roi, background_hsv)
    fg = ~bg
    if height is not None:
        # A piece stands off the table; its shadow and a mat-coloured print do not.
        fg |= height > HEIGHT_FG_M
    fg = (fg & roi).astype(np.uint8) * 255
    # Binarise: a 3 x 3 Gaussian, then anything left is foreground.
    fg = cv2.GaussianBlur(fg, (BLUR_KSIZE, BLUR_KSIZE), 0)
    fg = (fg > 127).astype(np.uint8)
    k = 3 if side_px is None else max(3, int(round(0.06 * side_px)) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ker)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ker)
    return _fill_holes(fg > 0) & roi, model


# --------------------------------------------------------------------------- splitting


def _flood(alt: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Marker-based priority-flood watershed (Meyer) on ``alt`` inside ``mask``, 8-connected."""
    h, w = alt.shape
    lab = markers.copy()
    heap: list[tuple[float, int, int, int]] = []
    cnt = 0
    ys, xs = np.nonzero(lab)
    nb = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    queued = lab > 0
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        for dy, dx in nb:
            v, u = y + dy, x + dx
            if 0 <= v < h and 0 <= u < w and mask[v, u] and not queued[v, u]:
                queued[v, u] = True
                heapq.heappush(heap, (float(alt[v, u]), cnt, v, u))
                cnt += 1
    while heap:
        _, _, y, x = heapq.heappop(heap)
        best = 0
        for dy, dx in nb:
            v, u = y + dy, x + dx
            if 0 <= v < h and 0 <= u < w and lab[v, u] > 0:
                best = lab[v, u]
                break
        lab[y, x] = best
        for dy, dx in nb:
            v, u = y + dy, x + dx
            if 0 <= v < h and 0 <= u < w and mask[v, u] and not queued[v, u]:
                queued[v, u] = True
                heapq.heappush(heap, (float(alt[v, u]), cnt, v, u))
                cnt += 1
    return lab


def split_blob(
    mask: np.ndarray, n: int, side_px: float, lines: np.ndarray | None = None
) -> list[np.ndarray]:
    """Split a blob of touching pieces into at most ``n`` parts: the ``n`` tallest separated
    distance-transform points seed a watershed of the inverted distance transform plus the
    dark cut lines (``lines``, 0..1). One part (the blob itself) when fewer than two qualify."""
    m8 = mask.astype(np.uint8)
    dt = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    # Greedy: the tallest point, then the tallest at least PEAK_SEPARATION from every seed. A
    # body's peak on a ridge shared with its neighbour need not be a local maximum.
    left = dt.copy()
    seeds: list[tuple[int, int]] = []
    sep = int(round(PEAK_SEPARATION * side_px))
    while len(seeds) < n:
        y, x = np.unravel_index(int(np.argmax(left)), left.shape)
        if left[y, x] < MIN_PEAK_FRACTION * side_px:
            break
        seeds.append((int(y), int(x)))
        cv2.circle(left, (int(x), int(y)), sep, 0.0, thickness=-1)
    if len(seeds) < 2:
        return [mask.copy()]
    markers = np.zeros(mask.shape, np.int32)
    for i, (y, x) in enumerate(seeds, 1):
        markers[y, x] = i
    alt = -dt / side_px
    if lines is not None:
        alt = alt + CUT_LINE_WEIGHT * lines
    lab = _flood(alt, markers, mask)
    return [lab == i for i in range(1, len(seeds) + 1)]


# --------------------------------------------------------------------------- per piece


def cut_lines(rgb: np.ndarray, side_px: float) -> np.ndarray:
    """Thin dark lines (a morphological black-hat of the grey image, about a twentieth of a
    piece wide), scaled to 0..1: where two touching pieces' cut edges meet."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = max(3, int(round(0.08 * side_px)) | 1)
    bh = cv2.morphologyEx(
        grey, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    bh = cv2.GaussianBlur(bh.astype(np.float32), (3, 3), 0)
    return np.clip(bh / CUT_LINE_CONTRAST, 0.0, 1.0)


def incenter(mask: np.ndarray) -> tuple[float, float, float]:
    """Centre and radius of the largest inscribed circle (distance-transform maximum), px."""
    dt = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return float(x), float(y), float(dt[y, x])


def side_angle(mask: np.ndarray, side_px: float) -> float:
    """Image angle of the piece's square sides in [0, 90) degrees, counter-clockwise as
    displayed: the largest group of Hough lines on the outline (source: Classify_Pieces), or the
    rotated box when no line is found."""
    m8 = mask.astype(np.uint8) * 255
    edges = cv2.Canny(m8, 50, 150, apertureSize=3)
    votes = max(6, int(round(HOUGH_VOTES * side_px)))
    lines = cv2.HoughLines(edges, 1, np.pi / 180, votes)
    if lines is not None and len(lines):
        th = np.mod(lines[:, 0, 1], np.pi / 2)  # sides are 90 degrees apart: fold
        groups: list[list[float]] = []
        for t in th.tolist():  # source order: strongest first
            for grp in groups:
                mean = _fold_mean(grp)
                d = abs((t - mean + np.pi / 4) % (np.pi / 2) - np.pi / 4)
                if d <= ANGLE_GROUP_RAD:
                    grp.append(t)
                    break
            else:
                if len(groups) < 4:
                    groups.append([t])
        best = max(groups, key=len)
        theta = _fold_mean(best)
        # theta is a side's normal in image (y-down) axes; the sides make the same angles mod 90.
        return float((-math.degrees(theta)) % 90.0)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (_, _), (_, _), a = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    return float((-a) % 90.0)


def _fold_mean(angles: list[float]) -> float:
    z = np.exp(4j * np.asarray(angles))
    return float(np.mod(np.angle(z.mean()) / 4.0, np.pi / 2))


def _chord(mask: np.ndarray, x: float, y: float, ux: float, uy: float) -> tuple[float, float]:
    """Distances from (x, y) to where the line along +-(ux, uy) leaves the mask."""
    h, w = mask.shape
    out = []
    for sgn in (1.0, -1.0):
        t = 0.0
        while True:
            t += 0.5
            u, v = int(round(x + sgn * t * ux)), int(round(y + sgn * t * uy))
            if not (0 <= u < w and 0 <= v < h) or not mask[v, u]:
                break
        out.append(t)
    return out[0], out[1]


def _disc_hits(other: np.ndarray, x: float, y: float, r: float) -> bool | None:
    h, w = other.shape
    if not (r <= x < w - r and r <= y < h - r):
        return None
    x0, x1 = max(0, int(math.floor(x - r))), min(w, int(math.ceil(x + r)) + 1)
    y0, y1 = max(0, int(math.floor(y - r))), min(h, int(math.ceil(y + r)) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    return bool(np.any(other[y0:y1, x0:x1][disc]))


def grip_check(
    mask: np.ndarray,
    other: np.ndarray,
    x: float,
    y: float,
    angle_deg: float,
    finger_r: float,
    margin: float,
) -> dict[str, Any]:
    """Two fingertips closing across the piece through (x, y), along either side direction:
    the narrower axis whose landing discs (just outside the piece) are clear of everything else."""
    cands = []
    for extra in (0.0, 90.0):
        a = math.radians(angle_deg + extra)
        ux, uy = math.cos(a), -math.sin(a)  # counter-clockwise as displayed (y down)
        d1, d2 = _chord(mask, x, y, ux, uy)
        hits = []
        for sgn, d in ((1.0, d1), (-1.0, d2)):
            off = d + margin + finger_r
            hits.append(_disc_hits(other, x + sgn * off * ux, y + sgn * off * uy, finger_r))
        clear = all(h is False for h in hits)
        cands.append(
            {
                "axis_deg": float((angle_deg + extra) % 180.0),
                "width_px": float(d1 + d2),
                "offset_px": float((d1 - d2) / 2.0),
                "clear": clear,
                "in_view": all(h is not None for h in hits),
                "u": (ux, uy),
            }
        )
    cands.sort(key=lambda c: (not c["clear"], c["width_px"]))
    return cands[0]


def drag_point(cnt: np.ndarray, side_px: float, shape: tuple[int, int]) -> dict[str, Any] | None:
    """Source Separate_Overlap: walk the blob's Huber centre line for points at least a gripper
    radius inside; the end farther from the centroid, pushed away from the centroid."""
    h, w = shape
    m = cv2.moments(cnt)
    if m["m00"] <= 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    vx, vy, x0, y0 = (float(v) for v in cv2.fitLine(cnt, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
    rad, rmin, step = GRIP_RADIUS[0] * side_px, GRIP_RADIUS[1] * side_px, DRAG_RADIUS_STEP * side_px
    span = float(np.hypot(h, w))
    ts = np.arange(-span, span, max(1.0, 0.02 * side_px))
    pts = [(x0 + t * vx, y0 + t * vy) for t in ts]
    pts = [(x, y) for x, y in pts if 0 <= x < w and 0 <= y < h]
    found: list[tuple[float, float]] = []
    while not found and rad >= rmin:
        found = [p for p in pts if cv2.pointPolygonTest(cnt, p, True) >= rad]
        if not found:
            rad -= step
    if not found:
        return None
    a, b = found[0], found[-1]
    p = a if math.hypot(a[0] - cx, a[1] - cy) > math.hypot(b[0] - cx, b[1] - cy) else b
    d = np.array([p[0] - cx, p[1] - cy])
    d = d / (np.linalg.norm(d) or 1.0)
    return {
        "pixel": {"x": float(p[0]), "y": float(p[1])},
        "radius_px": float(rad),
        "direction_px": [float(d[0]), float(d[1])],
    }


# --------------------------------------------------------------------------- world


class _Lift:
    """Pixel -> world through the hub's geometry tools, with a ray-plane fallback."""

    def __init__(self, ctx, cam: dict[str, Any], table_z: float | None):
        self.ctx = ctx
        self.depth = cam.get("depth")
        k = cam.get("intrinsics")
        self.K = None if k is None else np.asarray(k, dtype=np.float64)
        self.pose = cam.get("pose")
        self.T = None if self.pose is None else pose_to_matrix(self.pose)
        if self.depth is not None:
            self.depth = np.asarray(self.depth, dtype=np.float32)
        self.table_z = table_z
        self.ok = self.K is not None and self.T is not None

    def plane(self, uv: np.ndarray, z: float) -> np.ndarray | None:
        """Pixels (N, 2) on the horizontal plane at world z."""
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        k = self.K
        rays = np.column_stack(
            [(uv[:, 0] - k[0, 2]) / k[0, 0], (uv[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(uv))]
        )
        d = rays @ self.T[:3, :3].T
        o = self.T[:3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - o[2]) / d[:, 2]
        if not np.all(np.isfinite(s)) or np.any(s <= 0):
            return None
        return o + s[:, None] * d

    def points(self, mask: np.ndarray) -> np.ndarray:
        if self.depth is None:
            return np.zeros((0, 3))
        cloud = self.ctx.tool(
            "geometry.mask_to_world_points",
            mask=mask.astype(np.uint8) * 255,
            depth=self.depth,
            intrinsics=self.K,
            camera_pose=self.pose,
        )["points"]
        return np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)

    def pixel(self, x: float, y: float, z_fallback: float | None) -> np.ndarray | None:
        if self.depth is not None:
            try:
                p = self.ctx.tool(
                    "geometry.pixel_to_world_point",
                    pixel_x=float(x),
                    pixel_y=float(y),
                    depth=self.depth,
                    intrinsics=self.K,
                    camera_pose=self.pose,
                )["point"]
                return np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
            except ToolError as exc:
                logger.debug("no depth at (%.0f, %.0f): %s", x, y, exc)
        if z_fallback is None:
            return None
        q = self.plane(np.array([[x, y]]), z_fallback)
        return None if q is None else q[0]


def _vec(p: np.ndarray | None) -> Vec3 | None:
    return None if p is None else {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}


# --------------------------------------------------------------------------- core


def segment(
    rgb: np.ndarray,
    *,
    roi: np.ndarray | None = None,
    background_hsv: dict | None = None,
    piece_area_px: float | None = None,
    height: np.ndarray | None = None,
) -> dict[str, Any]:
    """Pixel-only pipeline: foreground, blob sorting, splitting. ``height``: height over the
    table per pixel [m] (median-filtered), only when the depth is good enough to show a piece.
    Returns the piece masks with their cluster ids, unsplit clusters and foreign blobs."""
    h, w = rgb.shape[:2]
    roi = np.ones((h, w), bool) if roi is None else np.asarray(roi).astype(bool)
    side = None if not piece_area_px else math.sqrt(piece_area_px)
    fg, model = foreground(rgb, roi, background_hsv, side, height)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    if not piece_area_px:
        # The piece size is the blob size most foreground pixels belong to (speckles are many
        # but small), over blobs clear of the frame's edge (the area outside a mat is huge).
        st = stats[1:]
        inner = (
            (st[:, 0] > 1)
            & (st[:, 1] > 1)
            & (st[:, 0] + st[:, 2] < w - 1)
            & (st[:, 1] + st[:, 3] < h - 1)
        )
        big = st[inner & (st[:, cv2.CC_STAT_AREA] >= 30), cv2.CC_STAT_AREA].astype(float)
        if big.size == 0:
            return {
                "pieces": [],
                "clusters": [],
                "foreign": [],
                "fg": fg,
                "background": model,
                "piece_area_px": 0.0,
            }
        big = np.sort(big)
        cum = np.cumsum(big)
        a0 = float(big[int(np.searchsorted(cum, 0.5 * cum[-1]))])
        near = big[(big >= 0.5 * a0) & (big <= 1.5 * a0)]
        piece_area_px = float(np.median(near)) if near.size else a0
        side = math.sqrt(piece_area_px)
        fg, model = foreground(rgb, roi, background_hsv, side, height)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    a_min, a_max = PIECE_AREA_RANGE[0] * piece_area_px, PIECE_AREA_RANGE[1] * piece_area_px
    # One piece's area: the median single piece of this frame when there are a few (tabs make
    # a piece larger than its cell), else the nominal one; the size ranges follow it.
    singles = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a_min <= a <= a_max:
            pts = np.column_stack(
                np.nonzero(
                    lab[
                        stats[i, 1] : stats[i, 1] + stats[i, 3],
                        stats[i, 0] : stats[i, 0] + stats[i, 2],
                    ]
                    == i
                )
            ).astype(np.float32)
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            if max(rw, rh) / max(min(rw, rh), 1.0) <= MAX_ASPECT:
                singles.append(float(a))
    unit = float(np.median(singles)) if len(singles) >= MIN_SINGLES else piece_area_px
    a_min, a_max = PIECE_AREA_RANGE[0] * unit, PIECE_AREA_RANGE[1] * unit
    cut = cut_lines(rgb, side) if CUT_LINE_WEIGHT > 0 else None
    border = np.zeros((h, w), bool)
    bx, by = max(1, int(EDGE_BLACKOUT * w)), max(1, int(EDGE_BLACKOUT * h))
    border[:by] = border[-by:] = True
    border[:, :bx] = border[:, -bx:] = True
    border |= ~cv2.erode(roi.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    pieces, clusters, foreign = [], [], []
    for i in range(1, n):
        x0, y0, bw, bh, area = (int(v) for v in stats[i])
        if area < FOREIGN_MIN * piece_area_px:
            continue
        sl = (slice(y0, y0 + bh), slice(x0, x0 + bw))
        m = lab[sl] == i
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        aspect = max(rw, rh) / max(min(rw, rh), 1.0)
        clipped = bool(np.any(border[sl][m]))
        bbox = [x0, y0, x0 + bw, y0 + bh]
        layers = None if height is None else _layers(m, height[sl], piece_area_px)
        if layers is not None and area <= OVERLAP_MAX * a_max:
            cid = len(clusters)
            top, covered = layers
            pieces.append(
                {"mask": top, "bbox": bbox, "cluster": cid, "clipped": clipped, "layer": "top"}
            )
            if covered is not None:
                pieces.append(
                    {
                        "mask": covered,
                        "bbox": bbox,
                        "cluster": cid,
                        "clipped": clipped,
                        "layer": "covered",
                    }
                )
            rest = m & ~top if covered is None else np.zeros_like(m)
            clusters.append(
                {
                    "cluster": cid,
                    "bbox": bbox,
                    "area_px": int(area),
                    "kind": "stacked",
                    "n_estimated": 2,
                    "n_split": 1 + (covered is not None),
                    "resolved": not rest.any(),
                    "rest": rest,
                    "contour": cnt,
                    "clipped": clipped,
                }
            )
            continue
        if a_min <= area <= a_max and aspect <= MAX_ASPECT:
            pieces.append(
                {"mask": m, "bbox": bbox, "cluster": None, "clipped": clipped, "layer": "table"}
            )
        elif (
            a_max < area <= OVERLAP_MAX * a_max
            and aspect <= MAX_ASPECT_OVERLAP
            or (a_min <= area <= a_max and aspect > MAX_ASPECT)
        ):
            cid = len(clusters)
            parts = _split_recursive(
                m,
                unit,
                side,
                (a_min, a_max),
                2,
                None if cut is None else cut[sl],
                whole_ok=aspect <= MAX_ASPECT,
            )
            # A piece-sized blob too long for one piece is two small or overlapping pieces:
            # left whole, it is not a piece.
            whole = len(parts) == 1
            good = [p for p in parts if a_min <= p.sum() <= a_max and not whole]
            for p in good:
                pieces.append(
                    {"mask": p, "bbox": bbox, "cluster": cid, "clipped": clipped, "layer": "table"}
                )
            rest = m.copy() if whole else np.zeros_like(m)
            for p in parts:
                if not (a_min <= p.sum() <= a_max):
                    rest |= p
            clusters.append(
                {
                    "cluster": cid,
                    "bbox": bbox,
                    "area_px": int(area),
                    "kind": "touching",
                    "n_estimated": int(max(2, round(area / unit))),
                    "n_split": len(good),
                    "resolved": not rest.any(),
                    "rest": rest,
                    "contour": cnt,
                    "clipped": clipped,
                }
            )
        elif area < a_min:
            foreign.append(
                {"bbox": bbox, "area_px": int(area), "kind": "fragment", "clipped": clipped}
            )
        else:
            foreign.append(
                {"bbox": bbox, "area_px": int(area), "kind": "too_large", "clipped": clipped}
            )
    return {
        "pieces": pieces,
        "clusters": clusters,
        "foreign": foreign,
        "fg": fg,
        "background": model,
        "piece_area_px": float(piece_area_px),
    }


def _split_recursive(
    m: np.ndarray,
    unit: float,
    side: float,
    rng: tuple[float, float],
    depth: int,
    lines: np.ndarray | None,
    whole_ok: bool = True,
) -> list[np.ndarray]:
    """Split into the piece count the area suggests (``unit``: one piece's area), rounded down
    and up, keeping whichever of those splits and the blob itself (if ``whole_ok``) has its
    parts nearest the piece size range (then the fewest parts); parts still too big are tried
    again."""

    def key(parts):
        dev = sum(max(0.0, rng[0] - p.sum(), p.sum() - rng[1]) for p in parts) / unit
        return (dev == 0.0, -dev, -len(parts))

    r = m.sum() / unit
    best = [m]
    for n in sorted({max(2, math.floor(r)), max(2, math.ceil(r))}):
        parts = split_blob(m, n, side, lines)
        if (len(best) == 1 and not whole_ok and len(parts) > 1) or key(parts) > key(best):
            best = parts
    if len(best) < 2 or depth <= 1:
        return best
    out = []
    for p in best:
        if p.sum() > rng[1]:
            out.extend(_split_recursive(p, unit, side, rng, depth - 1, lines))
        else:
            out.append(p)
    return out


def _layers(
    m: np.ndarray, hgt: np.ndarray, area: float
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """A blob with a raised part: (the piece on top, the covered piece's visible part or None).
    None when the blob is one level."""
    high = (m & (hgt > STACK_HEIGHT_M)).astype(np.uint8)
    frac = high.sum() / max(m.sum(), 1)
    if (
        not (STACK_FRACTION[0] <= frac <= STACK_FRACTION[1])
        or high.sum() < PIECE_AREA_RANGE[0] * area
    ):
        return None
    k = max(3, int(round(0.06 * math.sqrt(area))) | 1)
    high = cv2.morphologyEx(high, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(high, connectivity=8)
    if n < 2:
        return None
    top = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    top = _fill_holes(top) & m
    low = m & ~(cv2.dilate(top.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    n, lab, st, _ = cv2.connectedComponentsWithStats(low.astype(np.uint8), connectivity=8)
    if n < 2:
        return top, None
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    covered = lab == big
    if covered.sum() < COVERED_MIN * area:
        return top, None
    return top, covered


def _height_map(depth: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """World z of every pixel [m] (NaN without depth)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    d = np.where(np.isfinite(depth) & (depth >= 0.015), depth, np.nan).astype(np.float32)
    x = (u - K[0, 2]) * d / K[0, 0]
    y = (v - K[1, 2]) * d / K[1, 1]
    r = T[2, :3]
    return (r[0] * x + r[1] * y + r[2] * d + T[2, 3]).astype(np.float32)


def _same_colour(a: np.ndarray, b: np.ndarray) -> bool:
    sa, sb = a.sum() + 1e-6, b.sum() + 1e-6
    chroma = float(np.hypot(a[0] / sa - b[0] / sb, a[1] / sa - b[1] / sb))
    return (
        chroma < BACK_CHROMA_DIST
        and BACK_BRIGHTNESS_RANGE[0] <= sa / sb <= BACK_BRIGHTNESS_RANGE[1]
    )


def _full(mask_crop: np.ndarray, bbox: list[int], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, bool)
    x0, y0, x1, y1 = bbox
    out[y0:y1, x0:x1] = mask_crop
    return out


def run(
    ctx: NodeContext,
    observation: Any = None,
    camera: Any = None,
    camera_name: str = "",
    region: Any = None,
    background_hsv: dict | None = None,
    piece_size_m: float | None = None,
    piece_area_px: float | None = None,
    table_z: float | None = None,
    back_rgb: list[float] | None = None,
    finger_radius_m: float = FINGER_RADIUS_M,
    finger_margin_m: float = FINGER_MARGIN_M,
) -> Output:
    """Segment the loose pieces in one top-down RGB(-D) frame and give each a pick point.

    ``observation`` (or ``camera``, a CameraFrame; ``camera_name`` picks one of the
    observation's) must look roughly straight down at the staging surface. ``region``: a mask of
    where pieces may lie. ``background_hsv``: ``{"lower": [h, s, v], "upper": [h, s, v]}`` (OpenCV
    HSV) to use the source's fixed colour band instead of the estimated mat model.
    ``piece_size_m`` (nominal side) with depth, or ``piece_area_px``, fixes the piece size;
    otherwise it is the median blob. ``table_z`` overrides the table height measured from depth.
    ``back_rgb``: the colour of the pieces' backs, to call a flat piece face down."""
    cam = _camera(observation, camera, camera_name)
    rgb = _rgb(cam["rgb"])
    h, w = rgb.shape[:2]
    roi = None if region is None else np.asarray(region).astype(bool)
    lift = _Lift(ctx, cam, table_z)
    # The scale: px per metre on the table, from depth.
    px_per_m = None
    if lift.depth is not None and lift.K is not None:
        d = lift.depth[np.isfinite(lift.depth) & (lift.depth > 0.015)]
        if d.size > 0.2 * lift.depth.size:
            px_per_m = float(0.5 * (lift.K[0, 0] + lift.K[1, 1]) / np.percentile(d, 90))
    if piece_area_px is None and piece_size_m and px_per_m:
        piece_area_px = (piece_size_m * px_per_m) ** 2
    # The table's height and, when the depth is fine enough, every pixel's height over it.
    height, height_noise = None, None
    if lift.depth is not None and lift.ok:
        z = _height_map(lift.depth, lift.K, lift.T)
        ok = np.isfinite(z) if roi is None else np.isfinite(z) & roi
        if ok.sum() > 0.2 * ok.size:
            zs = z[ok][:: max(1, int(ok.sum() // 200000))]
            if table_z is None:
                hist, edges = np.histogram(zs, bins=np.arange(zs.min(), zs.max() + 1e-3, 5e-4))
                mode = (
                    float(edges[int(np.argmax(hist))] + 2.5e-4)
                    if hist.size
                    else float(np.median(zs))
                )
                table_z = float(np.median(zs[np.abs(zs - mode) < 1e-3]))
            near = zs[np.abs(zs - table_z) < 3e-3] - table_z
            height_noise = float(1.4826 * np.median(np.abs(near - np.median(near))))
            if height_noise < HEIGHT_NOISE_MAX_M:
                height = cv2.medianBlur(np.nan_to_num(z - table_z, nan=0.0).astype(np.float32), 5)
    seg = segment(
        rgb, roi=roi, background_hsv=background_hsv, piece_area_px=piece_area_px, height=height
    )
    area_px = seg["piece_area_px"]
    empty = np.zeros((h, w), np.uint8)
    if not seg["pieces"] and not seg["clusters"]:
        return Output(
            route="nothing_found",
            n_pieces=0,
            n_touching=0,
            n_pickable=0,
            n_clusters=0,
            pieces=[],
            clusters=[],
            foreign=_public_foreign(seg["foreign"]),
            piece_index=-1,
            piece_mask=empty,
            pick_point=None,
            pick_pixel=None,
            angle_deg=0.0,
            piece_area_px=area_px,
            px_per_m=px_per_m,
            table_z=table_z,
            free_spot=None,
            labels=np.zeros((h, w), np.int32),
            background=seg["background"],
            detail=f"no piece-sized blob ({len(seg['foreign'])} foreign blobs)",
        )
    side = math.sqrt(area_px)
    mpp_side = (piece_size_m or NOMINAL_PIECE_M) / side  # metres per px when depth is missing
    mpp = 1.0 / px_per_m if px_per_m else mpp_side

    # Labels: pieces 1..n, cluster remnants and foreign blobs as "other".
    labels = np.zeros((h, w), np.int32)
    for i, p in enumerate(seg["pieces"]):
        x0, y0, x1, y1 = p["bbox"]
        labels[y0:y1, x0:x1][p["mask"]] = i + 1
    other_fg = seg["fg"].copy()

    finger_r = finger_radius_m / mpp
    margin = finger_margin_m / mpp
    touch_px = max(1, int(round(TOUCH_GAP_M / mpp)))
    pieces_out: list[dict[str, Any]] = []
    for i, p in enumerate(seg["pieces"]):
        x0, y0, x1, y1 = p["bbox"]
        pad = int(math.ceil(side * 1.2 + finger_r + margin))
        X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
        X1, Y1 = min(w, x1 + pad), min(h, y1 + pad)
        win = labels[Y0:Y1, X0:X1]
        mine = win == i + 1
        others = other_fg[Y0:Y1, X0:X1] & ~mine
        if p["layer"] == "top":
            # The piece underneath is no obstacle to fingertips pinching the one on top.
            for j, q in enumerate(seg["pieces"]):
                if q["cluster"] == p["cluster"] and q["layer"] == "covered":
                    others &= win != j + 1
        # Touching neighbours and the gap to the nearest thing.
        grown = cv2.dilate(mine.astype(np.uint8), np.ones((2 * touch_px + 1,) * 2, np.uint8)) > 0
        neigh = sorted({int(v) - 1 for v in np.unique(win[grown & others]) if v > 0})
        touches_other = bool(np.any(grown & others))
        if others.any():
            dt_o = cv2.distanceTransform((~others).astype(np.uint8), cv2.DIST_L2, 5)
            gap_px = float(dt_o[mine].min()) - 1.0
        else:
            gap_px = float(pad)
        cx, cy, r = incenter(mine)
        ang = side_angle(mine, side)
        grip = grip_check(mine, others, cx, cy, ang, finger_r, margin)
        px, py = cx + X0, cy + Y0
        m = cv2.moments(mine.astype(np.uint8), binaryImage=True)
        cen = {"x": m["m10"] / m["m00"] + X0, "y": m["m01"] / m["m00"] + Y0}
        crop_mask = mine[y0 - Y0 : y1 - Y0, x0 - X0 : x1 - X0]
        crop_rgb = rgb[y0:y1, x0:x1]
        grey = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        core = (
            cv2.erode(
                crop_mask.astype(np.uint8),
                np.ones((3, 3), np.uint8),
                iterations=max(1, int(0.1 * side)),
            )
            > 0
        )
        if core.sum() < 10:
            core = crop_mask
        q25, q75 = np.percentile(grey[core], [25, 75])
        tex = float(0.7413 * (q75 - q25))
        face = "up" if tex >= LOW_TEXTURE_STD else "unknown"
        if face == "unknown" and back_rgb is not None:
            med = np.median(crop_rgb[core].reshape(-1, 3), axis=0).astype(np.float64)
            face = "down" if _same_colour(med, np.asarray(back_rgb, np.float64)) else "up"
        # World.
        full = _full(crop_mask, [x0, y0, x1, y1], (h, w))
        z_piece, piece_h = None, None
        if lift.depth is not None and lift.ok:
            pts = lift.points(full)
            if len(pts) >= 10:
                z_piece = float(np.median(pts[:, 2]))
                if table_z is not None:
                    piece_h = z_piece - table_z
        if z_piece is None:
            z_piece = table_z
        pick = lift.pixel(px, py, z_piece) if lift.ok else None
        yaw = grip_yaw = None
        area_m2 = None
        if lift.ok and z_piece is not None:
            ref = lift.plane(np.array([[px, py]]), z_piece)
            a = math.radians(ang)
            ga = math.radians(grip["axis_deg"])
            q = lift.plane(
                np.array(
                    [
                        [px + side * math.cos(a), py - side * math.sin(a)],
                        [px + side * math.cos(ga), py - side * math.sin(ga)],
                    ]
                ),
                z_piece,
            )
            if ref is not None and q is not None:
                yaw = float(math.atan2(q[0, 1] - ref[0, 1], q[0, 0] - ref[0, 0]) % (math.pi / 2))
                grip_yaw = float(math.atan2(q[1, 1] - ref[0, 1], q[1, 0] - ref[0, 0]) % math.pi)
            cnts, _ = cv2.findContours(
                crop_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64) + [x0, y0]
            poly = lift.plane(c, z_piece)
            if poly is not None:
                xs, ys = poly[:, 0], poly[:, 1]
                area_m2 = float(0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))
        layer = p["layer"]
        stacked = layer == "top" or (piece_h is not None and piece_h > STACK_HEIGHT_M)
        pickable = bool(
            grip["clear"]
            and r >= GRIP_RADIUS[1] * side
            and not p["clipped"]
            and layer != "covered"
            and not (stacked and layer != "top")
        )
        pieces_out.append(
            {
                "piece": f"p{i}",
                "index": i,
                "bbox_px": [int(v) for v in p["bbox"]],
                "mask_crop": (crop_mask.astype(np.uint8) * 255),
                "rgb_crop": crop_rgb.copy(),
                "area_px": int(mine.sum()),
                "area_m2": area_m2,
                "pick_pixel": {"x": float(px), "y": float(py)},
                "centroid_pixel": cen,
                "inradius_px": float(r),
                "inradius_m": float(r * mpp) if px_per_m else None,
                "position": _vec(pick),
                "height_m": piece_h,
                "angle_deg": float(ang),
                "yaw": yaw,
                "grip": {
                    "axis_deg": grip["axis_deg"],
                    "yaw": grip_yaw,
                    "width_px": grip["width_px"],
                    "width_m": float(grip["width_px"] * mpp) if px_per_m else None,
                    "opening_m": float(
                        grip["width_px"] * mpp + 2 * (finger_margin_m + finger_radius_m)
                    ),
                    "offset_px": grip["offset_px"],
                    "clear": grip["clear"],
                },
                "touching": touches_other,
                "neighbours": [f"p{j}" for j in neigh],
                "gap_px": max(0.0, gap_px),
                "gap_m": max(0.0, gap_px) * mpp if px_per_m else None,
                "split": p["cluster"] is not None
                and sum(q["cluster"] == p["cluster"] for q in seg["pieces"]) > 1,
                "cluster": p["cluster"],
                "clipped": p["clipped"],
                "stacked": bool(stacked),
                "layer": layer,
                "face": face,
                "texture_std": tex,
                "pickable": pickable,
            }
        )

    clusters_out = []
    for c in seg["clusters"]:
        rec = {
            k: c[k]
            for k in ("cluster", "bbox", "area_px", "n_estimated", "n_split", "resolved", "clipped")
        }
        rec["pieces"] = [q["piece"] for q in pieces_out if q["cluster"] == c["cluster"]]
        rec["drag"] = None
        if not c["resolved"]:
            x0, y0 = c["bbox"][:2]
            dp = drag_point(c["contour"].astype(np.float32), side, (h - y0, w - x0))
            if dp is not None:
                dp["pixel"] = {"x": dp["pixel"]["x"] + x0, "y": dp["pixel"]["y"] + y0}
                dp["point"] = (
                    _vec(lift.pixel(dp["pixel"]["x"], dp["pixel"]["y"], table_z))
                    if lift.ok
                    else None
                )
                rec["drag"] = dp
        clusters_out.append(rec)

    free = _free_spot(seg["fg"], roi, side, lift, table_z)
    touching = [q for q in pieces_out if q["touching"]]
    unresolved = [c for c in clusters_out if not c["resolved"]]
    pickable = [q for q in pieces_out if q["pickable"]]
    pool = pickable or pieces_out
    best = (
        max(pool, key=lambda q: (not q["touching"], min(q["gap_px"], side), q["inradius_px"]))
        if pool
        else None
    )
    best_mask = empty
    if best is not None:
        best_mask = _full(best["mask_crop"] > 0, best["bbox_px"], (h, w)).astype(np.uint8) * 255
    route = "touching_pieces" if touching or unresolved else "segmented"
    detail = (
        f"{len(pieces_out)} pieces ({len(touching)} touching, {len(pickable)} pickable), "
        f"{len(clusters_out)} touching blobs ({len(unresolved)} not fully split), "
        f"{len(seg['foreign'])} foreign; piece ~{side:.0f} px"
    )
    return Output(
        route=route,
        n_pieces=len(pieces_out),
        n_touching=len(touching),
        n_pickable=len(pickable),
        n_clusters=len(unresolved),
        pieces=pieces_out,
        clusters=clusters_out,
        foreign=_public_foreign(seg["foreign"]),
        piece_index=-1 if best is None else int(best["index"]),
        piece_mask=best_mask,
        pick_point=None if best is None else best["position"],
        pick_pixel=None if best is None else best["pick_pixel"],
        angle_deg=0.0 if best is None else float(best["angle_deg"]),
        piece_area_px=area_px,
        px_per_m=px_per_m,
        table_z=table_z,
        free_spot=free,
        labels=labels,
        background=seg["background"],
        detail=detail,
    )


def _public_foreign(foreign: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(f) for f in foreign]


def _free_spot(fg, roi, side, lift: _Lift, table_z) -> dict[str, Any] | None:
    """Source's open-area search: the mat point farthest from anything, if it is clear by
    FREE_SPOT_CLEARANCE pieces (and that far from the edge of the region / frame)."""
    h, w = fg.shape
    free = ~fg if roi is None else (~fg & roi)
    free = free.astype(np.uint8)
    free[0, :] = free[-1, :] = 0
    free[:, 0] = free[:, -1] = 0
    dt = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
    r = float(dt[y, x])
    if r < FREE_SPOT_CLEARANCE * side:
        return None
    point = lift.pixel(float(x), float(y), table_z) if lift.ok else None
    return {"pixel": {"x": float(x), "y": float(y)}, "clearance_px": r, "point": _vec(point)}
