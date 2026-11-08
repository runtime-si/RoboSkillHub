# Adapted from Domino-Arm (https://github.com/SponTanious/Domino-Arm),
# Functions/Domino_Detection.m and Class/Domino.m, MIT License,
# Copyright (c) 2016 Samuel Comerford, Cameron Gear, Lachlan Gear, Daniel Tanious,
# McKay Wight (license: skills/reading-domino-pips-and-orientation/LICENSE.domino-arm).
# Kept: the centre divider is the anchor - an MSER region at least four times
# longer than wide (and dark) gives where the tile splits; pips are dark circles
# found by circle detection plus round MSER regions, duplicates merged, and each
# pip is counted in the half rectangle it falls in, the rectangles running from
# 5 % of the tile length past the divider to the tile's end; the value is the
# sorted pair. Changes: MATLAB -> OpenCV (cv2.MSER, cv2.HoughCircles, and
# OpenCV's SimpleBlobDetector, Apache-2.0, in place of the MSER-round branch's
# size test); the tile comes from a mask (sam3.segment_text or upstream) and is
# rectified to a canonical 2:1 crop, so every pixel constant is a fraction of the
# tile's width instead of a Kinect-at-one-height pixel count; each candidate pip
# is scored by its contrast against its surround; each half's count is decoded
# against the seven double-six pip layouts (a glint or a missed pip shows as a
# poor fit) and reported with a margin; light pips on dark tiles; the tile is
# lifted to the world with geometry.mask_to_world_points for its ends, long axis
# and flat / upright / fallen state from its height over the table; routed exits
# instead of a figure with numbers drawn on it. The arm planning, the goal-slot
# table (get_domino_location.m) and the Kinect calibration are not ported.
"""Read each domino's two pip values, its ends and long axis, and whether it lies flat, stands or has fallen."""

from __future__ import annotations

import itertools
import logging
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

DEFAULT_QUERY = "domino"
"""On the photos in the SKILL.md SAM3 scored no mask over 0.15 for "domino tile"
(0.004 on the two near-top-down ones); for "domino" the tiles scored 0.64-0.95 and
the best non-tile mask 0.28."""
DEFAULT_SEGMENTER = "sam3.segment_text"

TILE_SIZE = (0.048, 0.024, 0.008)
"""Nominal tile length, width, thickness [m]: a common double-six size (48 x 24 x
8 mm; US sets are ~51 x 25 x 9.5). Only the midpoints between the three resting
heights use it, so a set 20 % off still classifies; pass ``tile_size`` for yours."""

MIN_SCORE = 0.3
"""Lowest segmenter score kept (the sam3 bundle's advice, as in
framing-a-target-by-mask-edge-steering)."""

MIN_MASK_PX = 40
"""Masks smaller than this are dropped [px]: a 24 mm tile at 1 m on the station's
overhead Orbbec (fx 461) is ~11 x 22 px = 240 px; 40 px is a fragment."""

MIN_TILE_WIDTH_PX = 30
"""Narrowest tile, across its short side, the pips are read at [px]; pips are then
~5 px (17 % of the width). Synthetic renders with 1 px blur and noise read 168/168
at 28 px, 157/168 at 24, 149/168 at 20 and 17/168 with 4 wrong at 16; 30 leaves room
for a real sensor. The overhead Orbbec sees a tile ~11 px wide, a wrist D405 at
0.20 m ~47 px."""

CANON_W = 64
"""Canonical tile width [px]: the tile is warped to 2*CANON_W x CANON_W, each half a
square, so the pixel constants below are fixed fractions of the tile."""

PIP_DIAMETER_FRAC = 0.17
"""Pip diameter over tile width: 4 mm pips on a 24 mm tile (3-5 mm pips are within
the detector's radius range below)."""

PIP_RADIUS_RANGE = (0.55, 1.6)
"""Pip radii searched, as multiples of the nominal radius."""

DIVIDER_GAP_FRAC = 0.05
"""Pips within this fraction of the tile length of the divider are in neither half:
Domino_Detection.m's rectangles start at 0.05 * height from the divider (a centre
spinner rivet sits in that band)."""

DIVIDER_ELONGATION = 4.0
"""An MSER region counts as the divider when its long axis is at least this many
times its short axis (Domino_Detection.m: ``axe(1) > 4*axe(2)``)."""

DIVIDER_LENGTH_RANGE = (0.3, 1.1)
"""Divider segment lengths accepted, over the tile width (the source accepted
30-60 px for a ~50 px-wide tile; 0.3 lets a line broken by a spinner count)."""

DIVIDER_ANGLE_DEG = 20.0
"""Most the divider may lean from square to the long axis [deg]."""

DIVIDER_CENTRE_BAND = 0.15
"""An MSER divider's centre must lie within this fraction of the tile length of the
middle (the source had no bound; a photo taken at ~40 deg put it 0.1 off)."""

PROFILE_CENTRE_BAND = 0.07
"""The fallback divider column must lie within this fraction of the length of the
middle: a six's inner pip column, a dark streak down most rows, sits 0.125 off it
(at 0.15 the fallback took it for the divider and misread sixes)."""

MIN_FILL = 0.85
"""Least mask area over its minimum-area rectangle for one flat tile: single tiles
on the near-top-down photos filled 0.88-0.97, a sam3 mask joining halves of two
touching tiles less."""

MSER_MAX_VARIATION = 1.0
"""cv2.MSER max variation. The source's ``MaxAreaVariation`` was 0.25 on a whole
Kinect frame; a 2-4 px line in the canonical crop grows by its whole length per
grey step, and at 0.25 or 0.5 MSER missed the divider in up to a third of the
synthetic blur / noise renders, at 1.0 in none."""

MSER_SIGMA = 0.7
"""Gaussian blur before MSER [canonical px]."""

DIVIDER_PROFILE_MIN = 0.15
"""Fallback divider: least darkness that 80 % of the middle rows of its column
reach (a six's pip column is dark over ~70 % of them)."""

EXCLUSION_RADII = 2.0
"""A pip taken by one layout position is masked out of the others' probes within
this many nominal radii of its centre (its own radius plus the probe disc's)."""

SATURATED = 250
"""Grey level counted as glare: a pip under it cannot be seen."""

GLARE_FRAC = 0.3
"""A layout position whose pip disc is at least this fraction glare is left out of
the decoding (missing data): on a light tile only one read as empty, since a pip
seen under glare was seen."""

PROBE_WINDOW = 0.08
"""Each empty layout position is probed for a pip within this many half-widths of
it (a disc there stays clear of the neighbouring positions' pips)."""

MERGE_RADII = 1.2
"""Candidate pips closer than this many nominal radii are one pip (the source merged
within 10 px for 4-7 px circles)."""

MIN_PIP_SCORE = 0.25
"""A candidate is a pip when its Weber contrast (surround minus disc over surround)
is at least this; black pips on a white tile score ~0.8, ivory and blur ~0.5."""

STRAY_SCORE = 0.5
"""A pip-like spot the decoded layout leaves out (on or off the grid) makes the half
uncertain when it scores this: pips on the photos in the SKILL.md scored 0.85-0.97,
an obliquely seen tile's shaded rim 0.28 next to a layout position."""

MIN_CANDIDATE_SCORE = 0.08
"""Weaker detections are dropped (and a darker region must reach it to be the
divider); the decoder reads every layout position's own contrast, so a pip half
washed out by glare still adds what it has."""

LAYOUT_TOLERANCE = 0.16
"""Most a detected pip may sit from its layout position, in half-widths (positions
are 0.25 apart; sets that put pips at 0.20 / 0.80 or 0.30 / 0.70 are 0.05 off). A
detection answers only for the position nearest it."""

MIN_MARGIN = 0.3
"""A half is read when its best layout beats the best layout of another value by
this much (contrast units). A pip that left no trace at all costs its layout only
MIN_PIP_SCORE (0.25), so a margin under 0.3 cannot rule one out; a clear pip
(contrast ~0.8) separates neighbouring values by ~1.1."""

EDGE_MARGIN_FRAC = 0.05
"""Candidates within this fraction of the width of the tile's outline are dropped
(the tile edge and the segmenter's outline are dark lines)."""

TABLE_RING_FRAC = 0.35
"""The table ring round the tiles is this many tile widths wide."""

LAYOUTS: dict[int, list[list[tuple[int, int]]]] = {
    0: [[]],
    1: [[(1, 1)]],
    2: [[(0, 0), (2, 2)], [(2, 0), (0, 2)]],
    3: [[(0, 0), (1, 1), (2, 2)], [(2, 0), (1, 1), (0, 2)]],
    4: [[(0, 0), (2, 0), (0, 2), (2, 2)]],
    5: [[(0, 0), (2, 0), (0, 2), (2, 2), (1, 1)]],
    6: [
        [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)],
        [(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2)],
    ],
}
"""Double-six pip layouts on a 3 x 3 grid (column, row), each variant a quarter turn
of the other; grid positions sit at 0.25 / 0.5 / 0.75 of the half."""


class Output(TypedDict):
    route: str
    tiles: list[dict[str, Any]]
    tile: dict[str, Any]
    value: list[int]
    value_ab: list[int]
    state: str
    center_world: dict[str, float]
    axis_world: dict[str, float]
    ends_world: list[dict[str, float]]
    closeup_targets: list[dict[str, float]]
    camera_name: str
    detail: str


def _vec(v: Any) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _pick_camera(obs: dict[str, Any], name: str | None, arm_id: int) -> dict[str, Any] | None:
    """The named camera, else the wrist camera nearest the arm's tool, else the first."""
    cams = [c for c in obs.get("cameras") or [] if c.get("rgb") is not None]
    if name:
        return next((c for c in cams if c.get("name") == name), None)
    wrist = [c for c in cams if any(k in str(c.get("name", "")) for k in ("eye_in_hand", "wrist"))]
    arms = obs.get("arms") or []
    if len(wrist) > 1 and arm_id < len(arms) and arms[arm_id].get("ee_pose"):
        tool = _xyz(arms[arm_id]["ee_pose"])
        posed = [c for c in wrist if c.get("pose") is not None]
        if posed:
            return min(posed, key=lambda c: float(np.linalg.norm(_xyz(c["pose"]) - tool)))
    return (wrist or cams or [None])[0]


def _gray(rgb: np.ndarray) -> np.ndarray:
    """The darkest channel: coloured pips (yellow, light blue, pink on a white tile)
    are dark in it, while luminance puts a yellow pip within 0.2 of the tile."""
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        return rgb.astype(np.uint8)
    return rgb[..., :3].min(axis=2).astype(np.uint8)


def _bool_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = m > (127 if m.dtype == np.uint8 and m.max() > 1 else 0)
    if m.shape != shape:
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        m = m > 0
    return m


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def tile_frame(mask: np.ndarray) -> dict[str, Any]:
    """The mask's minimum-area rectangle: centre, long-axis unit vector (pointing to
    image-right, or down when vertical), long and short side in pixels."""
    pts = np.argwhere(mask)[:, ::-1].astype(np.float32)
    (cx, cy), _, _ = rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    long_e, short_e = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
    length, width = float(np.linalg.norm(long_e)), float(np.linalg.norm(short_e))
    u = long_e / max(length, 1e-9)
    if u[0] < -1e-6 or (abs(u[0]) <= 1e-6 and u[1] < 0):
        u = -u
    n = np.array([-u[1], u[0]])
    return {
        "center": np.array([cx, cy]),
        "u": u.astype(np.float64),
        "n": n.astype(np.float64),
        "length": length,
        "width": width,
    }


def _canon_to_image(frame: dict[str, Any]) -> np.ndarray:
    """2x3 affine taking canonical crop pixels (x along the tile, end a at x=0) to image pixels."""
    sx = frame["length"] / (2.0 * CANON_W)
    sy = frame["width"] / CANON_W
    u, n, c = frame["u"], frame["n"], frame["center"]
    a = np.column_stack([u * sx, n * sy])
    t = c - a @ np.array([CANON_W, CANON_W / 2.0])
    return np.column_stack([a, t])


def _to_image(m: np.ndarray, xy: Any) -> np.ndarray:
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    return xy @ m[:, :2].T + m[:, 2]


def _region_axes(pts: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Centroid, segment length and width (sqrt(12 var) along each principal axis), major direction."""
    p = pts.astype(np.float64)
    c = p.mean(0)
    cov = np.cov((p - c).T) if len(p) > 2 else np.eye(2) * 1e-3
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-6)
    return c, float(np.sqrt(12 * vals[1])), float(np.sqrt(12 * vals[0])), vecs[:, 1]


def _contrast(gray: np.ndarray, x: float, y: float, r: float, tile_level: float) -> float:
    """Weber contrast of the disc at (x, y) against its surround (capped at the tile's level)."""
    h, w = gray.shape
    rr = int(np.ceil(2.0 * r)) + 1
    x0, x1 = max(0, int(x) - rr), min(w, int(x) + rr + 1)
    y0, y1 = max(0, int(y) - rr), min(h, int(y) + rr + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xs - x, ys - y)
    patch = gray[y0:y1, x0:x1].astype(np.float64)
    disc = patch[d <= 0.6 * r]
    ring = patch[(d >= 1.3 * r) & (d <= 1.9 * r)]
    if disc.size == 0 or ring.size == 0:
        return 0.0
    surround = min(float(np.median(ring)), tile_level)
    return float(np.clip((surround - float(disc.mean())) / max(surround, 1.0), 0.0, 1.0))


def _contrast_map(gray: np.ndarray, r0: float, tile_level: float) -> np.ndarray:
    """Weber contrast of a pip-sized disc at every pixel against the mean of its surround."""

    def kernel(lo: float, hi: float) -> np.ndarray:
        rr = int(np.ceil(hi)) + 1
        ys, xs = np.mgrid[-rr : rr + 1, -rr : rr + 1]
        d = np.hypot(xs, ys)
        k = ((d >= lo) & (d <= hi)).astype(np.float32)
        return k / max(float(k.sum()), 1.0)

    g = gray.astype(np.float32)
    disc = cv2.filter2D(g, -1, kernel(0.0, 0.6 * r0), borderType=cv2.BORDER_REPLICATE)
    ring = cv2.filter2D(g, -1, kernel(1.3 * r0, 1.9 * r0), borderType=cv2.BORDER_REPLICATE)
    surround = np.minimum(ring, tile_level)
    return np.clip((surround - disc) / np.maximum(surround, 1.0), 0.0, 1.0)


def _grid_evidence(
    cmap: np.ndarray, found: list[tuple[float, float, float, float]], x0: float, x1: float
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float, float]]]:
    """Pip evidence at the half's nine layout positions (column-major from end a): the
    strongest detection within LAYOUT_TOLERANCE whose nearest position it is, else the best contrast probed within
    PROBE_WINDOW (a pip the detectors missed still counts, and an empty position says
    how empty it is). Positions take pips greedily, strongest first, and a pip taken by
    one position is masked out of the others' probes. Also the detections near no
    position (strays)."""
    span = max(x1 - x0, 1.0)
    r0 = 0.5 * PIP_DIAMETER_FRAC * CANON_W
    grid = [(x0 + gx * span, gy * CANON_W) for gx in (0.25, 0.5, 0.75) for gy in (0.25, 0.5, 0.75)]
    wx, wy = PROBE_WINDOW * span, PROBE_WINDOW * CANON_W
    ys, xs = np.mgrid[0 : cmap.shape[0], 0 : cmap.shape[1]]

    def nearest(p: tuple[float, float, float, float]) -> tuple[int, float]:
        d = [np.hypot((p[0] - x) / span, (p[1] - y) / CANON_W) for x, y in grid]
        return int(np.argmin(d)), float(min(d))

    owner = [nearest(p) for p in found]

    def evidence(i: int, taken: list[tuple[float, float]]) -> tuple[float, float, float]:
        x, y = grid[i]
        ok = [
            p
            for p, (cell, d) in zip(found, owner, strict=True)
            if cell == i
            and d <= LAYOUT_TOLERANCE
            and all(np.hypot(p[0] - tx, p[1] - ty) > EXCLUSION_RADII * r0 for tx, ty in taken)
        ]
        ya, yb = max(0, int(y - wy)), min(cmap.shape[0], int(y + wy) + 1)
        xa, xb = max(0, int(x - wx)), min(cmap.shape[1], int(x + wx) + 1)
        win = cmap[ya:yb, xa:xb].copy()
        for tx, ty in taken:
            win[np.hypot(xs[ya:yb, xa:xb] - tx, ys[ya:yb, xa:xb] - ty) <= EXCLUSION_RADII * r0] = (
                0.0
            )
        best = (x, y, 0.0)
        if win.size:
            j = np.unravel_index(int(np.argmax(win)), win.shape)
            best = (float(xa + j[1]), float(ya + j[0]), float(win[j]))
        for p in ok:
            if p[3] > best[2]:
                best = (p[0], p[1], p[3])
        return best

    first = [evidence(i, []) for i in range(9)]
    cells: list[tuple[float, float, float]] = list(first)
    taken: list[tuple[float, float]] = []
    for i in sorted(range(9), key=lambda k: -first[k][2]):
        cells[i] = evidence(i, taken) if taken else first[i]
        if cells[i][2] >= MIN_PIP_SCORE:
            taken.append((cells[i][0], cells[i][1]))
    strays = [p for p, (_, d) in zip(found, owner, strict=True) if d > LAYOUT_TOLERANCE]
    return cells, strays


def _decode_half(
    cells: list[tuple[float, float, float]], ignore: set[int] | None = None
) -> dict[str, Any]:
    """Best double-six layout for one half's nine position scores: each position in the
    layout adds (score - MIN_PIP_SCORE), each outside it (MIN_PIP_SCORE - score); an
    ignored position adds nothing either way."""
    c = np.array([cell[2] for cell in cells])
    keep = np.ones(9, bool)
    keep[list(ignore or ())] = False
    best_by_value: dict[int, tuple[float, list[int]]] = {}
    for value, variants in LAYOUTS.items():
        for layout in variants:
            used = [3 * col + row for col, row in layout]
            inside = np.zeros(9, bool)
            inside[used] = True
            total = float(
                np.sum(c[inside & keep] - MIN_PIP_SCORE) + np.sum(MIN_PIP_SCORE - c[~inside & keep])
            )
            if value not in best_by_value or total > best_by_value[value][0]:
                best_by_value[value] = (total, used)
    ranked = sorted(best_by_value.items(), key=lambda kv: -kv[1][0])
    (value, (top, used)), (runner_up, (second, _)) = ranked[0], ranked[1]
    return {
        "value": int(value),
        "fit": float(top),
        "margin": float(top - second),
        "runner_up": int(runner_up),
        "used": used,
    }


def read_face(rgb: np.ndarray, mask: np.ndarray, polarity: str = "auto") -> dict[str, Any]:
    """Read the pips of one tile seen face-on: rectify, find the divider, detect and
    score pips, split them into the two halves, decode each half. Pixel outputs are
    in the input image."""
    gray_full = _gray(rgb)
    frame = tile_frame(mask)
    m = _canon_to_image(frame)
    size = (2 * CANON_W, CANON_W)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    canon = cv2.warpAffine(gray_full, m, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
    inside = cv2.warpAffine(mask.astype(np.uint8) * 255, m, size, flags=flags, borderValue=0) > 127
    if inside.sum() < 0.3 * inside.size:
        inside[:] = True
    level = float(np.median(canon[inside]))
    glare = (canon >= SATURATED) & inside
    dark_tile = level < 110 if polarity == "auto" else polarity == "light"
    if dark_tile:
        canon = 255 - canon
        level = float(np.median(canon[inside]))
    canon = canon.astype(np.uint8)
    canon[~inside] = int(level)
    edge = max(1, int(round(EDGE_MARGIN_FRAC * CANON_W)))
    interior = cv2.erode(inside.astype(np.uint8), np.ones((2 * edge + 1, 2 * edge + 1), np.uint8))
    interior[:edge, :] = interior[-edge:, :] = 0
    interior[:, :edge] = interior[:, -edge:] = 0

    r0 = 0.5 * PIP_DIAMETER_FRAC * CANON_W
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 4)).apply(canon)
    smooth = cv2.GaussianBlur(eq, (3, 3), 0)

    # Divider: an elongated dark MSER region across the tile near its middle.
    mser = cv2.MSER_create()
    mser.setDelta(4)
    mser.setMinArea(max(5, int(0.3 * r0 * r0)))
    mser.setMaxArea(int(0.25 * canon.size))
    mser.setMaxVariation(MSER_MAX_VARIATION)
    # MSER finds no region whose area is flat over every threshold (a hard-edged,
    # noise-free line); a camera's own blur is about this much.
    regions, _ = mser.detectRegions(cv2.GaussianBlur(eq, (0, 0), MSER_SIGMA))
    divider_x: list[float] = []
    divider_scores: list[float] = []
    divider_mask = np.zeros(canon.shape, np.uint8)
    candidates: list[tuple[float, float, float]] = []
    lo_len, hi_len = (f * CANON_W for f in DIVIDER_LENGTH_RANGE)
    for reg in regions:
        c, length, width, major = _region_axes(reg)
        darkness = (level - float(canon[reg[:, 1], reg[:, 0]].mean())) / max(level, 1.0)
        if darkness < MIN_CANDIDATE_SCORE:
            continue
        lean = np.degrees(np.arccos(min(1.0, abs(float(major[1])))))
        if (
            length > DIVIDER_ELONGATION * max(width, 1.0)
            and lo_len <= length <= hi_len
            and lean <= DIVIDER_ANGLE_DEG
            and abs(c[0] - CANON_W) <= DIVIDER_CENTRE_BAND * 2 * CANON_W
        ):
            divider_x.append(float(c[0]))
            divider_scores.append(float(darkness))
            divider_mask[reg[:, 1], reg[:, 0]] = 1
        elif length <= 2.0 * max(width, 1.0):
            r = 0.5 * float(np.sqrt(length * width))
            if PIP_RADIUS_RANGE[0] * r0 <= r <= PIP_RADIUS_RANGE[1] * r0:
                candidates.append((float(c[0]), float(c[1]), r))
    divider_method = "mser" if divider_x else ""
    xd = float(np.median(divider_x)) if divider_x else float(CANON_W)
    divider_score = float(max(divider_scores)) if divider_x else 0.0
    if not divider_x:
        # Fallback: the column near the centre that stays dark down most of the
        # tile's middle rows (the divider spans them all, a pip a fifth).
        rows = slice(int(0.2 * CANON_W), int(0.8 * CANON_W))
        dark = (level - canon[rows].astype(np.float64)) / max(level, 1.0)
        profile = np.percentile(dark, 20, axis=0)
        band = int(PROFILE_CENTRE_BAND * 2 * CANON_W)
        lo_x = CANON_W - band
        peak = lo_x + int(np.argmax(profile[lo_x : CANON_W + band + 1]))
        if profile[peak] >= DIVIDER_PROFILE_MIN:
            xd, divider_score, divider_method = float(peak), float(profile[peak]), "profile"
    divider_found = bool(divider_method)

    # Pip candidates: dark blobs (SimpleBlobDetector) and circles (Hough), with the round MSERs.
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor, params.blobColor = True, 0
    params.minThreshold, params.maxThreshold, params.thresholdStep = 10, 240, 10
    params.minRepeatability = 2
    params.filterByArea = True
    params.minArea = float(np.pi * (PIP_RADIUS_RANGE[0] * r0) ** 2)
    params.maxArea = float(np.pi * (PIP_RADIUS_RANGE[1] * r0) ** 2)
    params.filterByCircularity, params.minCircularity = True, 0.6
    params.filterByConvexity, params.minConvexity = True, 0.8
    params.filterByInertia, params.minInertiaRatio = True, 0.4
    params.minDistBetweenBlobs = float(r0)
    for kp in cv2.SimpleBlobDetector_create(params).detect(smooth):
        candidates.append((float(kp.pt[0]), float(kp.pt[1]), 0.5 * float(kp.size)))
    circles = cv2.HoughCircles(
        smooth,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1.4 * r0,
        param1=100,
        param2=10,
        minRadius=int(np.floor(PIP_RADIUS_RANGE[0] * r0)),
        maxRadius=int(np.ceil(PIP_RADIUS_RANGE[1] * r0)),
    )
    if circles is not None:
        candidates += [(float(x), float(y), float(r)) for x, y, r in circles[0]]

    merged: list[list[float]] = []
    for x, y, r in candidates:
        for g in merged:
            if np.hypot(g[0] / g[3] - x, g[1] / g[3] - y) <= MERGE_RADII * r0:
                g[0] += x
                g[1] += y
                g[2] += r
                g[3] += 1
                break
        else:
            merged.append([x, y, r, 1.0])
    pips: list[tuple[float, float, float, float]] = []
    for sx, sy, sr, k in merged:
        x, y, r = sx / k, sy / k, float(np.clip(sr / k, 0.6 * r0, 1.5 * r0))
        if (
            not (0 <= int(y) < CANON_W and 0 <= int(x) < 2 * CANON_W)
            or not interior[int(y), int(x)]
        ):
            continue
        score = _contrast(canon, x, y, r, level)
        if score >= MIN_CANDIDATE_SCORE:
            pips.append((x, y, r, score))

    contrast_map = _contrast_map(canon, r0, level)
    # The divider is not a pip: seen obliquely it leans across the gap band, and a
    # position probed next to it would read it as one.
    on_divider = cv2.dilate(divider_mask, np.ones((2 * int(r0) + 1, 2 * int(r0) + 1), np.uint8)) > 0
    contrast_map[on_divider] = 0.0
    pips = [p for p in pips if not on_divider[int(p[1]), int(p[0])]]
    ys_c, xs_c = np.mgrid[0:CANON_W, 0 : 2 * CANON_W]
    gap = DIVIDER_GAP_FRAC * 2 * CANON_W
    halves = []
    for lo, hi in ((0.0, xd - gap), (xd + gap, 2.0 * CANON_W)):
        x0, x1 = (0.0, xd) if lo == 0.0 else (xd, 2.0 * CANON_W)
        found = [p for p in pips if lo <= p[0] <= hi]
        cells, strays = _grid_evidence(contrast_map, found, x0, x1)
        dec = _decode_half(cells)
        # Glare hides a dark pip on a light tile (a position read empty under it is a
        # guess) and looks like a pip once a dark tile is inverted (any position).
        glared: set[int] = set()
        for i, (gx, gy) in enumerate(itertools.product((0.25, 0.5, 0.75), repeat=2)):
            if not dark_tile and cells[i][2] >= MIN_PIP_SCORE:
                continue
            disc = np.hypot(xs_c - (x0 + gx * (x1 - x0)), ys_c - gy * CANON_W) <= r0
            if glare[disc].mean() >= GLARE_FRAC:
                glared.add(i)
        if glared:
            dec = _decode_half(cells, glared)
        dec["glare_positions"] = len(glared)
        chosen = [cells[i] for i in dec["used"]]
        # A pip-like position the layout leaves out: a seven-to-nine of a bigger set,
        # dirt, or a misplaced half.
        dec["unexplained"] = sum(
            1
            for i in range(9)
            if i not in dec["used"] and i not in glared and cells[i][2] >= STRAY_SCORE
        )
        halves.append(
            {
                **dec,
                "raw_count": int(sum(1 for p in found if p[3] >= MIN_PIP_SCORE)),
                "strays": int(sum(1 for p in strays if p[3] >= STRAY_SCORE)) + dec["unexplained"],
                "pips_px": [[float(v) for v in _to_image(m, (p[0], p[1]))[0]] for p in chosen],
                "pip_scores": [round(float(p[2]), 3) for p in chosen],
                "position_scores": [round(float(p[2]), 3) for p in cells],
                "candidates": len(found),
                "center_canon": ((x0 + x1) / 2.0, CANON_W / 2.0),
                "rect_canon": (x0, x1),
            }
        )
    strong = sum(1 for p in pips if p[3] >= MIN_PIP_SCORE)
    ends = _to_image(m, [(0.0, CANON_W / 2.0), (2.0 * CANON_W, CANON_W / 2.0)])
    return {
        "frame": frame,
        "affine": m,
        "divider_found": divider_found,
        "divider_method": divider_method,
        "divider_score": round(divider_score, 3),
        "divider_px": [float(v) for v in _to_image(m, (xd, CANON_W / 2.0))[0]],
        "halves": halves,
        "strong_pips": strong,
        "dark_tile": bool(dark_tile),
        "ends_px": [[float(v) for v in e] for e in ends],
    }


def _half_mask(mask: np.ndarray, m: np.ndarray, x0: float, x1: float) -> np.ndarray:
    corners = _to_image(m, [(x0, 0.0), (x1, 0.0), (x1, CANON_W), (x0, CANON_W)])
    poly = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(poly, [np.round(corners).astype(np.int32)], 1)
    return mask & (poly > 0)


def _lift(ctx: NodeContext, mask: np.ndarray, cam: dict[str, Any]) -> np.ndarray:
    if cam.get("depth") is None or cam.get("intrinsics") is None or cam.get("pose") is None:
        return np.zeros((0, 3))
    if not mask.any():
        return np.zeros((0, 3))
    depth = np.asarray(cam["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.shape != mask.shape:
        depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    reply = ctx.tool(
        "geometry.mask_to_world_points",
        mask=mask.astype(np.uint8) * 255,
        depth=depth,
        intrinsics=np.asarray(cam["intrinsics"], dtype=np.float64),
        camera_pose=cam["pose"],
    )
    pts = reply.get("points", reply) if isinstance(reply, dict) else reply
    if isinstance(pts, dict):
        pts = pts.get("points")
    return np.asarray(pts if pts is not None else [], dtype=np.float64).reshape(-1, 3)


def _state(height: float | None, tile_size: tuple[float, float, float]) -> str:
    length, width, thick = tile_size
    if height is None:
        return "unknown"
    if height < 0.5 * (thick + width):
        return "flat"
    if height > 0.5 * (width + length):
        return "upright"
    return "fallen"


def run(
    ctx: NodeContext,
    camera: dict[str, Any] | None = None,
    masks: list[Any] | None = None,
    mask_scores: list[float] | None = None,
    query: str = DEFAULT_QUERY,
    segmenter: str = DEFAULT_SEGMENTER,
    camera_name: str | None = None,
    target_point: dict[str, float] | None = None,
    table_z: float | None = None,
    tile_size: list[float] | tuple[float, float, float] = TILE_SIZE,
    pip_polarity: str = "auto",
    min_score: float = MIN_SCORE,
    min_tile_width_px: float = MIN_TILE_WIDTH_PX,
    max_tiles: int = 8,
    arm_id: int = 0,
) -> Output:
    size3 = (float(tile_size[0]), float(tile_size[1]), float(tile_size[2]))
    out: Output = {
        "route": "not_seen",
        "tiles": [],
        "tile": {},
        "value": [],
        "value_ab": [],
        "state": "unknown",
        "center_world": {},
        "axis_world": {},
        "ends_world": [],
        "closeup_targets": [],
        "camera_name": "",
        "detail": "",
    }

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        logger.info("reading-domino-pips-and-orientation: %s (%s)", route, detail)
        return out

    cam = camera
    if cam is None:
        cam = _pick_camera(ctx.tool("robot.get_observation"), camera_name, int(arm_id))
    if cam is None or cam.get("rgb") is None:
        return finish("not_seen", "no camera frame with an rgb image")
    out["camera_name"] = str(cam.get("name", ""))
    rgb = np.asarray(cam["rgb"])
    shape = rgb.shape[:2]

    if masks is None:
        try:
            reply = ctx.tool(segmenter, image=rgb, query=query, max_results=int(max_tiles))
        except Exception as exc:
            return finish("not_seen", f"{segmenter} failed: {exc}")
        masks = list(reply.get("masks") or [])
        mask_scores = list(reply.get("scores") or [1.0] * len(masks))
    scores = list(mask_scores) if mask_scores is not None else [1.0] * len(masks)
    kept: list[tuple[np.ndarray, float]] = []
    for raw, s in sorted(zip(masks, scores, strict=False), key=lambda ms: -float(ms[1])):
        if float(s) < min_score:
            continue
        mk = _largest_component(_bool_mask(raw, shape))
        if mk.sum() < MIN_MASK_PX:
            continue
        if any((mk & k).sum() > 0.5 * min(mk.sum(), k.sum()) for k, _ in kept):
            continue
        kept.append((mk, float(s)))
        if len(kept) >= max_tiles:
            break
    if not kept:
        return finish("not_seen", f"no {query!r} mask scoring {min_score} and {MIN_MASK_PX} px")

    if table_z is None:
        union = np.zeros(shape, bool)
        for mk, _ in kept:
            union |= mk
        widths = [tile_frame(mk)["width"] for mk, _ in kept]
        k = max(3, int(round(TABLE_RING_FRAC * float(np.median(widths)))))
        grown = cv2.dilate(union.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
        ring = _lift(ctx, grown & ~union, cam)
        if len(ring) >= 10:
            table_z = float(np.median(ring[:, 2]))

    tiles: list[dict[str, Any]] = []
    for idx, (mk, s) in enumerate(kept):
        tiles.append(
            _read_tile(ctx, cam, rgb, mk, s, idx, table_z, size3, pip_polarity, min_tile_width_px)
        )

    out["tiles"] = tiles
    judged = tiles
    if target_point is not None:
        tp = np.array([float(target_point[c]) for c in ("x", "y")])
        with_world = [t for t in tiles if t["center_world"]]
        if with_world:
            judged = [
                min(
                    with_world,
                    key=lambda t: float(
                        np.hypot(t["center_world"]["x"] - tp[0], t["center_world"]["y"] - tp[1])
                    ),
                )
            ]
    main = next((t for t in judged if t["status"] == "read"), judged[0])
    out["tile"] = main
    out["value"], out["value_ab"] = main["value"], main["value_ab"]
    out["state"], out["center_world"] = main["state"], main["center_world"]
    out["axis_world"], out["ends_world"] = main["axis_world"], main["ends_world"]
    small = [t for t in judged if t["status"] == "too_small"]
    out["closeup_targets"] = [t["center_world"] for t in small if t["center_world"]]
    summary = "; ".join(f"#{t['index']} {t['status']} {t['detail']}" for t in judged)
    if all(t["status"] == "read" for t in judged):
        return finish("read", summary)
    if small:
        return finish("needs_closeup", summary)
    return finish("ambiguous", summary)


def _read_tile(
    ctx: NodeContext,
    cam: dict[str, Any],
    rgb: np.ndarray,
    mask: np.ndarray,
    mask_score: float,
    idx: int,
    table_z: float | None,
    tile_size: tuple[float, float, float],
    polarity: str,
    min_width_px: float,
) -> dict[str, Any]:
    frame = tile_frame(mask)
    tile: dict[str, Any] = {
        "index": idx,
        "status": "ambiguous",
        "state": "unknown",
        "value": [],
        "value_ab": [],
        "mask_score": round(mask_score, 3),
        "mask_area_px": int(mask.sum()),
        "width_px": round(frame["width"], 1),
        "length_px": round(frame["length"], 1),
        "center_px": [float(v) for v in frame["center"]],
        "ends_px": [
            [float(v) for v in frame["center"] - 0.5 * frame["length"] * frame["u"]],
            [float(v) for v in frame["center"] + 0.5 * frame["length"] * frame["u"]],
        ],
        "half_centers_px": [],
        "pips_px": [[], []],
        "pip_scores": [[], []],
        "pip_counts_raw": [],
        "layout_margin": [],
        "divider_found": False,
        "divider_score": 0.0,
        "center_world": {},
        "axis_world": {},
        "ends_world": [],
        "half_centers_world": [],
        "height": None,
        "footprint": [],
        "detail": "",
    }
    cloud = _lift(ctx, mask, cam)
    height = None
    if len(cloud) >= 10:
        top = float(np.percentile(cloud[:, 2], 95))
        centre = np.median(cloud, axis=0)
        xy = cloud[:, :2] - centre[:2]
        vals, vecs = np.linalg.eigh(np.cov(xy.T))
        proj = xy @ vecs
        ext = np.percentile(proj, 98, axis=0) - np.percentile(proj, 2, axis=0)
        tile["footprint"] = [round(float(ext[1]), 4), round(float(ext[0]), 4)]
        tile["center_world"] = _vec(centre)
        if table_z is not None:
            height = top - float(table_z)
            tile["height"] = round(height, 4)
    state = _state(height, tile_size)
    tile["state"] = state

    if state in ("upright", "fallen"):
        if state == "upright" and tile["center_world"] and table_z is not None:
            c = tile["center_world"]
            tile["axis_world"] = {"x": 0.0, "y": 0.0, "z": 1.0}
            tile["ends_world"] = [
                {"x": c["x"], "y": c["y"], "z": float(table_z)},
                {"x": c["x"], "y": c["y"], "z": float(table_z) + float(height)},
            ]
        elif len(cloud) >= 10:
            _axis_from_halves(
                ctx,
                cam,
                mask,
                _canon_to_image(frame),
                (0.0, CANON_W),
                (CANON_W, 2.0 * CANON_W),
                cloud,
                tile,
            )
        tile["status"] = "not_flat"
        tile["detail"] = f"{state} (height {height * 1e3:.0f} mm): its face is not up"
        return tile

    if frame["width"] < min_width_px:
        tile["status"] = "too_small"
        tile["detail"] = (
            f"{frame['width']:.0f} px wide, under {min_width_px:.0f} px: pips ~{PIP_DIAMETER_FRAC * frame['width']:.1f} px, read from a wrist close-up"
        )
        return tile
    aspect = frame["length"] / max(frame["width"], 1e-6)
    face = read_face(rgb, mask, polarity)
    m = face["affine"]
    tile["divider_found"], tile["divider_score"] = face["divider_found"], face["divider_score"]
    tile["divider_px"] = face["divider_px"]
    tile["half_centers_px"] = [
        [float(v) for v in _to_image(m, h["center_canon"])[0]] for h in face["halves"]
    ]
    tile["pips_px"] = [h["pips_px"] for h in face["halves"]]
    tile["pip_scores"] = [h["pip_scores"] for h in face["halves"]]
    tile["pip_counts_raw"] = [h["raw_count"] for h in face["halves"]]
    tile["layout_margin"] = [round(h["margin"], 3) for h in face["halves"]]
    a, b = (h["value"] for h in face["halves"])
    tile["value_ab"] = [a, b]
    tile["value"] = sorted([a, b])
    if len(cloud) >= 10:
        (ha, hb) = (h["rect_canon"] for h in face["halves"])
        _axis_from_halves(ctx, cam, mask, m, ha, hb, cloud, tile)

    fill = float(mask.sum()) / max(frame["length"] * frame["width"], 1.0)
    tile["fill"] = round(fill, 3)
    weak = [i for i, h in enumerate(face["halves"]) if h["margin"] < MIN_MARGIN]
    stray = [i for i, h in enumerate(face["halves"]) if h["strays"]]
    tile["glare_positions"] = [h["glare_positions"] for h in face["halves"]]
    tile["stray_pips"] = [h["strays"] for h in face["halves"]]
    if not 1.5 <= aspect <= 2.7 or fill < MIN_FILL:
        tile["status"] = "ambiguous"
        tile["detail"] = f"mask aspect {aspect:.2f}, fill {fill:.2f}: not one flat tile"
    elif not face["divider_found"] and face["strong_pips"] == 0:
        tile["status"] = "face_down"
        tile["value"], tile["value_ab"] = [], []
        tile["detail"] = "no divider and no pips: face down (or a blank whose divider is not seen)"
    elif not face["divider_found"]:
        tile["status"] = "ambiguous"
        tile["detail"] = (
            f"value {a}-{b} uncertain: divider not seen, the halves were split at the "
            "mask's middle (a tilted view moves the divider off it)"
        )
    elif weak:
        glare = sum(face["halves"][i]["glare_positions"] for i in weak)
        tile["status"] = "ambiguous"
        tile["detail"] = (
            f"value {a}-{b} uncertain: half {'/'.join('ab'[i] for i in weak)} layout margin "
            f"{min(face['halves'][i]['margin'] for i in weak):.2f} < {MIN_MARGIN}"
            + (f" ({glare} pip position(s) under glare)" if glare else "")
        )
    elif stray:
        tile["status"] = "ambiguous"
        tile["detail"] = (
            f"value {a}-{b} uncertain: a pip the double-six layout does not explain in half "
            f"{'/'.join('ab'[i] for i in stray)}"
        )
    else:
        tile["status"] = "read"
        tile["detail"] = f"{a}-{b}"
    return tile


def _axis_from_halves(
    ctx: NodeContext,
    cam: dict[str, Any],
    mask: np.ndarray,
    m: np.ndarray,
    half_a: tuple[float, float],
    half_b: tuple[float, float],
    cloud: np.ndarray,
    tile: dict[str, Any],
) -> None:
    """Long axis (end a -> end b) from the two halves' world medians; ends at the cloud's extent along it."""
    pa = _lift(ctx, _half_mask(mask, m, *half_a), cam)
    pb = _lift(ctx, _half_mask(mask, m, *half_b), cam)
    if len(pa) < 5 or len(pb) < 5:
        return
    ca, cb = np.median(pa, axis=0), np.median(pb, axis=0)
    axis = cb - ca
    if np.linalg.norm(axis) < 1e-6:
        return
    axis /= np.linalg.norm(axis)
    centre = np.median(cloud, axis=0)
    proj = (cloud - centre) @ axis
    lo, hi = float(np.percentile(proj, 1)), float(np.percentile(proj, 99))
    tile["axis_world"] = _vec(axis)
    tile["half_centers_world"] = [_vec(ca), _vec(cb)]
    tile["ends_world"] = [_vec(centre + lo * axis), _vec(centre + hi * axis)]
    tile["length"] = round(hi - lo, 4)
