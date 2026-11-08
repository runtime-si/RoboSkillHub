# Adapted from Zolver (https://github.com/Kawaboongawa/Zolver, MIT, Copyright (c) 2018
# Kawaboongawa and co): Img/filters.py (get_relative_angles, is_maximum_local,
# longest_peak, distance_signature, flat_score, indent_score, outdent_score,
# compute_comp, peaks_inside, is_pattern, is_acceptable_comb, type_peak,
# my_find_corner_signature), Img/peak_detect.py (detect_peaks, rising edges only),
# Puzzle/Edge.py (is_compatible), Puzzle/Mover.py (stick_pieces) and Puzzle/Distance.py
# (diff_match_edges2, dist_edge), and the per-slot rotation search of
# Puzzle/Puzzle.py (best_diff, Strategy.FILL).
# Changes: the outline is smoothed 3 px along itself, resampled to a fixed 850 points by
# arc length (Zolver's pieces had 750-1000 contour points, so its sigma 5-15 and radius 10
# stay as they were) and oriented so convex turns are positive; the peak threshold is 0.2
# of the largest turn instead of 0.3; combinations of 4 corners are drawn as sorted index
# sets instead of all permutations filtered by order (the same set); the peak-count rule
# is relaxed in two steps when no set passes it; a set must make a near-rectangle about a
# cell in size (two gates, tight then loose) before Zolver's cost picks among them, and
# sets whose sides all get a type are preferred; per-side scores are cached; each corner is
# then sharpened by intersecting lines fitted to the outline either side of it; a side
# Zolver leaves UNDEFINED is typed by its deviation from its chord. Matching is against
# placed neighbour sides given in board metres rather than against a growing block of
# pieces: one rigid transform per candidate rotation, fitted to the slot corners of every
# placed neighbour at once (stick_pieces aligned one edge), with the piece's sides pushed
# out by a few small offsets to absorb the mask's edge bias, then each side is scored by
# the symmetric mean closest-point distance (diff_match_edges2 compared points by index
# and truncated the longer edge). Colour is not used (Zolver's real-puzzle score was
# colour only). Border and interior-side rules follow Strategy.BORDER's flat-side
# alignment, as a penalty instead of a hard reject. The grip point (deepest interior
# point) and the corner / edge / interior class (PuzzlePiece.number_of_border) are
# returned per piece.
"""Fit candidate jigsaw pieces into a slot by the shape of their sides against the placed neighbours."""

from __future__ import annotations

import math
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

DIRECTIONS = ("south", "east", "north", "west")
"""Slot sides in counter-clockwise order seen from above (board +x east, +y north)."""

_TRAVERSE = {
    "south": (1.0, 0.0),
    "east": (0.0, 1.0),
    "north": (-1.0, 0.0),
    "west": (0.0, -1.0),
}
"""Direction a counter-clockwise walk round the slot runs along each side."""

OUTLINE_SMOOTH_PX = 3.0
"""Gaussian smoothing of the mask outline along itself [contour pixels] before anything
else; see SKILL.md for the sweep."""

CONTOUR_POINTS = 850
"""Outline resampled to this many points. Zolver's sample pieces have 750-1000 contour
points (resources/jigsaw-samples, measured), the scale its sigma and peak radius were set
for; a 64 px wrist-camera piece has ~330."""

SIGMA_MIN, SIGMA_MAX = 5, 15
"""Gaussian smoothing of the turning-angle signature, tried in increasing order until no
side is UNDEFINED (Zolver, non green-screen mode)."""

PEAK_FRACTION = 0.2
"""A turn counts as a peak above this fraction of the largest (Zolver's mph was 0.3). The
big round knobs of a random cut turn harder than a corner: at 0.3 a corner of 1 in 20
noisy wavy pieces made no peak at any sigma; 0.2 halved that, 0.1 let in more wrong sets."""

PEAK_COUNTS = ((0, 2, 3), (0, 1, 2, 3), None)
"""Turning peaks of each sign allowed inside one side, tried in order: Zolver's (0, 2, 3),
then any up to 3, then any number (None). A wavy random-cut side bends its shoulders and
puts 4+ peaks inside; with only Zolver's rule 9% of noisy wavy pieces found no
near-rectangular corner set and fell back to an ungated one, wrong in 2 of 3 cases."""

LOCAL_MAX_RADIUS = 10
"""Points either side within which an indent's ends must be local maxima (Zolver)."""

QUAD_GATES = ((15.0, 1.2, 0.2), (25.0, 1.35, 0.3))
"""Corner sets kept before Zolver's cost picks one, as (angle_tol deg, side_ratio,
cell_tol): angles within angle_tol of square, opposite sides within side_ratio of each
other and, when the cell size is known, every side within cell_tol of it. The looser
gate is tried only when no set passes the tight one, and every set only when none passes
either. On the classic synthetic cut with mask noise the true corners made at most
6 deg / x1.11 / 8%; the wrong sets Zolver's cost preferred made 2-29 deg, x1.0-1.4, and
sides 20-35% off a square cell. Wavy random-cut pieces (corners jittered 7% of a cell,
curved sides) failed the tight gate in about 1 in 12 noisy pieces."""

REFINE_SPAN: tuple[float, float] | None = (0.04, 0.14)
"""Stretch of outline either side of a corner, in cells, fitted with a line to sharpen
it (None: keep Zolver's peak). Knob necks start 0.2+ of a side from a corner."""

SIDE_POINTS = 64
"""Points per side for the side-to-side comparison."""

FLAT_DEVIATION = 0.08
"""A side whose outline stays within this fraction of its chord length of the chord is
flat; used for neighbour sides and Zolver's UNDEFINED sides. Generated tabs reach
0.20-0.28 of the side; 1-2 px jitter on a 64 px side is 0.03."""

OFFSETS = (-0.01, 0.0, 0.01, 0.02, 0.03)
"""Outward shifts of the piece's sides tried in each fit, as fractions of the side length
(the best is kept). A mask's edge sits 1-2 px off the true cut (blur, threshold, erosion),
which on 64 px sides thins every tab and widens every blank by about 0.02 of the side.
Without them the right piece won only 58% of noisy 3-neighbour slots from 50 candidates
on the classic cut; with them 99%."""

MISMATCH_FRACTION = 0.05
"""Zolver counted points more than 5 px apart on ~100-200 px sides; here a point is a
mismatch further than this fraction of the side length."""

MAX_FIT = 0.06
"""Best score (mean closest-point distance over the side length after the rigid fit, plus
INCOMPATIBLE_PENALTY when a side-type rule breaks) still called a fit. Synthetic cuts,
64 px sides, blur / threshold / erosion / 1-2 px jitter: the right piece scored median
0.014-0.015, 95th percentile 0.018 (classic) and 0.023 (wavy), max 0.051."""

MIN_MARGIN = 1.25
MIN_MARGIN_ONE_SIDE = 1.6
"""Runner-up (a different piece) must score at least this times the best for `fits`; the
second with a single placed neighbour, where one side leaves more look-alikes. Noisy
synthetic slots, 10 or 50 candidates: with 2-3 neighbours 98.6-100% of `fits` at 1.25
were right; with one neighbour 61% (classic) / 95% (wavy) at 1.25, 81% / 98.5% at 1.6."""

INCOMPATIBLE_PENALTY = 0.02
"""Added to the score of a rotation that breaks a side-type rule (tab against tab, a flat
side inside the board, a knob on the border). Zolver dropped such pairs; a penalty of
about one right fit's score keeps a piece whose side was misread in the running. On the
synthetic cuts 1.0 and 0.02 ranked identically; on the traced real cut, whose traced
sides often lost their knob and read flat, 0.02 doubled the right piece's top-1 rate."""

FLAT, TAB, BLANK, UNDEFINED = "flat", "tab", "blank", "undefined"


class MatchResult(TypedDict, total=False):
    route: str
    best_id: str
    best_rotation_deg: float
    best_score: float
    margin: float
    piece_class: str
    grip_pixel: list[float]
    grip_point_board: list[float]
    ranked: list[dict[str, Any]]
    pieces: list[dict[str, Any]]
    detail: str


# --- outline and turning-angle signature (Zolver Img/filters.py) -------------------------


def outline(mask: np.ndarray, n: int = CONTOUR_POINTS, smooth_px: float = 0.0) -> np.ndarray:
    """Largest outer contour of a mask, smoothed along itself by a Gaussian of ``smooth_px``
    contour pixels, ``n`` points evenly spaced by arc length, in image pixels, oriented
    counter-clockwise in the y-up frame (convex turns positive)."""
    m = (np.asarray(mask) > (127 if np.asarray(mask).dtype == np.uint8 else 0)).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty mask")
    c = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    if len(c) < 16:
        raise ValueError("mask outline too small")
    x, v = c[:, 0], c[:, 1]
    if np.sum(x * np.roll(v, -1) - np.roll(x, -1) * v) > 0:  # image-frame area > 0 = clockwise y-up
        c = c[::-1]
    if smooth_px > 0:
        r = int(4 * smooth_px + 0.5)
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / smooth_px) ** 2)
        k /= k.sum()
        wrap = np.vstack([c[-r:], c, c[:r]])
        c = np.column_stack([np.convolve(wrap[:, i], k, mode="valid") for i in range(2)])
    closed = np.vstack([c, c[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


def _gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    r = int(4.0 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(x, k / k.sum(), mode="same")


def relative_angles(pts: np.ndarray, sigma: float) -> np.ndarray:
    """Smoothed turning angle at every outline point (Zolver get_relative_angles); index i
    is the turn at point i."""
    n = len(pts)
    p = np.vstack([pts, pts, pts])
    d = np.diff(p, axis=0)
    ang = np.unwrap(np.arctan2(-d[:, 1], d[:, 0]))
    turn = np.diff(ang)
    turn = np.convolve(turn, [0.33] * 5, mode="same")
    turn = _gaussian(turn, sigma)
    return turn[n - 1 : 2 * n - 1]


def detect_peaks(x: np.ndarray, mph: float) -> np.ndarray:
    """Rising-edge local maxima at or above ``mph`` (Zolver's detect_peaks defaults)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return np.array([], dtype=int)
    dx = x[1:] - x[:-1]
    ind = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
    ind = ind[(ind > 0) & (ind < x.size - 1)]
    return ind[x[ind] >= mph]


def _circular_peaks(x: np.ndarray) -> np.ndarray:
    n = len(x)
    mph = PEAK_FRACTION * np.max(x)
    a = detect_peaks(x, mph)
    b = (detect_peaks(np.roll(x, n // 2), mph) - n // 2) % n
    return np.unique(np.concatenate([a, b]).astype(int))


def is_maximum_local(index: int, x: np.ndarray, radius: int) -> bool:
    lo, hi = max(0, index - radius), min(x.shape[0] - 1, index + radius)
    return not np.any(x[lo : hi + 1] > x[index])


def longest_peak(x: np.ndarray) -> tuple[int, int]:
    longest, j = (0, 0), 0
    for i in range(x.shape[0]):
        if x[i] >= 0:
            j = i
        if i - j > longest[1] - longest[0]:
            longest = (j, i)
    return longest


def flat_score(x: np.ndarray) -> float:
    """Largest distance of the signature from the line through its ends."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    l1 = np.array([0.0, x[0]])
    d = np.array([n - 1.0, x[-1]]) - l1
    pts = np.column_stack([np.arange(n, dtype=np.float64), x]) - l1
    return float(np.max(np.abs(d[0] * pts[:, 1] - d[1] * pts[:, 0])) / np.linalg.norm(d))


def indent_score(x: np.ndarray) -> float:
    n = x.shape[0]
    p0, p1 = longest_peak(x)
    while p0 > 0 and not is_maximum_local(p0, x, LOCAL_MAX_RADIUS):
        p0 -= 1
    while p1 < n - 1 and not is_maximum_local(p1, x, LOCAL_MAX_RADIUS):
        p1 += 1
    shape = np.concatenate([x[:p0], x[p1:]])
    if shape.shape[0] <= 1:
        return flat_score(x)
    return flat_score(shape)


def outdent_score(x: np.ndarray) -> float:
    return indent_score(-x)


def _side_cost(x: np.ndarray) -> float:
    hole, head, border = indent_score(x), outdent_score(x), flat_score(x)
    return min(hole, head) if hole != border else border


def peaks_inside(span, peaks) -> list[int]:
    if len(span) == 0:
        return []
    return [p for p in peaks if span[0] < p < span[-1]]


def _sides_of(corners) -> list[tuple[int, int]]:
    a, b, c, d = corners
    return [(a, b), (b, c), (c, d), (d, a)]


def _inside(span: tuple[int, int], peaks: np.ndarray, n: int) -> list[int]:
    """Peaks strictly inside the outline span start -> end (wrapping), as offsets from start."""
    s, e = span
    length = (e - s) % n
    off = (peaks - s) % n
    return sorted(int(o) for o in off if 0 < o < length)


def type_peak(pos: list[int], neg: list[int]) -> str:
    """Zolver's side type from the turning peaks inside it: none = flat; two concave turns
    inside the convex ones = blank (HOLE); two convex inside concave = tab (HEAD)."""
    if not pos and not neg:
        return FLAT
    if len(peaks_inside(pos, neg)) == 2:
        return BLANK
    if len(peaks_inside(neg, pos)) == 2:
        return TAB
    return UNDEFINED


def chord_type(side: np.ndarray, y_up: bool = True) -> tuple[str, float]:
    """Flat / tab / blank from the side's largest deviation from its chord, for a side
    walked counter-clockwise round its piece (interior on the left). Returns the type and
    the signed deviation over the chord length (negative = outward)."""
    a, b = side[0], side[-1]
    chord = b - a
    length = float(np.linalg.norm(chord))
    if length == 0:
        return UNDEFINED, 0.0
    u = chord / length
    dev = (u[0] * (side[:, 1] - a[1]) - u[1] * (side[:, 0] - a[0])) / length
    if not y_up:
        dev = -dev
    lo, hi = float(dev.min()), float(dev.max())
    peak = lo if -lo > hi else hi
    if abs(peak) < FLAT_DEVIATION:
        return FLAT, peak
    return (TAB if peak < 0 else BLANK), peak


def find_corners(
    pts: np.ndarray, quad_gate: bool = True, cell_px: tuple[float, float] | None = None
) -> tuple[list[int], list[str]] | None:
    """Zolver's my_find_corner_signature on an outline from :func:`outline`: the 4 corner
    indices (ascending) and the Zolver types of sides corner0->1, 1->2, 2->3, 3->0."""
    for gate in (*QUAD_GATES, None) if quad_gate else (None,):
        fallback = None
        for allowed in PEAK_COUNTS:
            found = _corner_search(pts, allowed, gate, cell_px)
            if found is not None and UNDEFINED not in found[1]:
                return found
            fallback = fallback or found
        if fallback is not None:
            return fallback
    return None


def plausible_quad(
    corners: np.ndarray,
    cell_px: tuple[float, float] | None = None,
    gate: tuple[float, float, float] | None = None,
) -> bool:
    """The 4 corners make a near-rectangle: every angle within ``angle_tol`` of 90 deg,
    opposite sides within ``side_ratio`` of each other and, when the cell size is known,
    every side within ``cell_tol`` of it (either way round); ``gate`` is
    (angle_tol, side_ratio, cell_tol), the first of QUAD_GATES by default."""
    angle_tol, side_ratio, cell_tol = gate or QUAD_GATES[0]
    v = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(v, axis=1)
    if np.any(lengths == 0):
        return False
    for k in range(4):
        a, b = -v[k - 1], v[k]
        cos = float(a @ b) / (lengths[k - 1] * lengths[k])
        if abs(math.degrees(math.acos(max(-1.0, min(1.0, cos)))) - 90.0) > angle_tol:
            return False
    r1 = max(lengths[0], lengths[2]) / min(lengths[0], lengths[2])
    r2 = max(lengths[1], lengths[3]) / min(lengths[1], lengths[3])
    if max(r1, r2) > side_ratio:
        return False
    if cell_px is None:
        return True
    w, h = cell_px
    for a, b in ((w, h), (h, w)):
        want = np.array([a, b, a, b])
        if np.all(np.abs(lengths / want - 1.0) <= cell_tol):
            return True
    return False


def _corner_search(
    pts: np.ndarray,
    allowed: tuple[int, ...] | None,
    gate: tuple[float, float, float] | None,
    cell_px,
) -> tuple[list[int], list[str]] | None:
    n = len(pts)
    lo, hi = n / 8.0, n / 2.0
    best_fit, types = None, None
    for sigma in range(SIGMA_MIN, SIGMA_MAX + 1):
        ra = relative_angles(pts, sigma)
        pos = _circular_peaks(ra)
        neg = _circular_peaks(-ra)
        ran = ra / (np.linalg.norm(ra) or 1.0)
        cache: dict[tuple[int, int], float] = {}

        def side_cost(s: int, e: int, ran=ran, cache=cache) -> float:
            if (s, e) not in cache:
                seg = ran[s:e] if e > s else np.concatenate([ran[s:], ran[:e]])
                cache[s, e] = _side_cost(seg)
            return cache[s, e]

        good = []
        k = len(pos)
        for i in range(k):
            for j in range(i + 1, k):
                if not lo < pos[j] - pos[i] < hi:
                    continue
                for m in range(j + 1, k):
                    if not lo < pos[m] - pos[j] < hi:
                        continue
                    for q in range(m + 1, k):
                        comb = (int(pos[i]), int(pos[j]), int(pos[m]), int(pos[q]))
                        if not lo < comb[3] - comb[2] < hi or not lo < comb[0] + n - comb[3] < hi:
                            continue
                        if allowed is None or all(
                            len(_inside(sp, pos, n)) in allowed
                            and len(_inside(sp, neg, n)) in allowed
                            for sp in _sides_of(comb)
                        ):
                            good.append(comb)
        if not good:
            continue
        if gate is not None:
            good = [c for c in good if plausible_quad(pts[list(c)], cell_px, gate)]
            if not good:
                continue
        typed = [
            [type_peak(_inside(sp, pos, n), _inside(sp, neg, n)) for sp in _sides_of(c)]
            for c in good
        ]
        defined = [k for k, t in enumerate(typed) if UNDEFINED not in t]
        pool = defined or list(range(len(good)))
        costs = [sum(side_cost(s, e) for s, e in _sides_of(good[k])) for k in pool]
        best = pool[int(np.argmin(costs))]
        best_fit, types = good[best], typed[best]
        if defined:
            break
    if best_fit is None:
        return None
    return list(best_fit), types


def side_points(
    pts: np.ndarray, corners: list[int], vertices: np.ndarray | None = None
) -> list[np.ndarray]:
    """The outline between consecutive corners; with ``vertices`` each side starts and ends
    at the refined corner points instead of the outline points at the split."""
    out = []
    for k, (s, e) in enumerate(_sides_of(corners)):
        side = pts[s : e + 1] if e > s else np.vstack([pts[s:], pts[: e + 1]])
        if vertices is not None:
            side = np.vstack([vertices[k], side[1:-1], vertices[(k + 1) % 4]])
        out.append(side)
    return out


def refine_corner(pts: np.ndarray, idx: int, cell_px: float) -> tuple[int, np.ndarray]:
    """Sharpen one corner: fit a line to the outline from REFINE_SPAN[0] to REFINE_SPAN[1]
    cells along it on either side of ``idx`` and intersect the two. Smoothing and a mask's
    rounding cut a corner short; the shoulders next to it are nearly straight. Returns the
    outline index nearest the intersection (within REFINE_SPAN[1]) and the intersection,
    or ``idx`` and its own point when the lines meet at under 30 deg or too far away."""
    n = len(pts)
    step = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    a, b = (int(round(f * cell_px / step)) for f in REFINE_SPAN)
    if b <= a + 2:
        return idx, pts[idx]
    lines = []
    for sign in (1, -1):
        seg = pts[(idx + sign * np.arange(a, b + 1)) % n]
        c = seg.mean(axis=0)
        _, _, vt = np.linalg.svd(seg - c)
        lines.append((c, vt[0]))
    (c1, d1), (c2, d2) = lines
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < math.sin(math.radians(30.0)):
        return idx, pts[idx]
    t = ((c2[0] - c1[0]) * d2[1] - (c2[1] - c1[1]) * d2[0]) / cross
    vertex = c1 + t * d1
    window = (idx + np.arange(-b, b + 1)) % n
    dist = np.linalg.norm(pts[window] - vertex, axis=1)
    if dist.min() > REFINE_SPAN[1] * cell_px:
        return idx, pts[idx]
    return int(window[int(np.argmin(dist))]), vertex


def resample(poly: np.ndarray, m: int = SIDE_POINTS) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float64)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], m)
    return np.column_stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])])


def _quad_side(corners: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)))


def grip_pixel(mask: np.ndarray) -> list[float]:
    """Interior point farthest from the outline (distance-transform maximum), (u, v) pixels."""
    m = (np.asarray(mask) > (127 if np.asarray(mask).dtype == np.uint8 else 0)).astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    v, u = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return [float(u), float(v)]


def classify(types: list[str]) -> str:
    """corner / edge / interior from the number of flat sides (PuzzlePiece.number_of_border)."""
    flats = sum(t == FLAT for t in types)
    return {0: "interior", 1: "edge", 2: "corner"}.get(flats, "unknown")


def analyse_piece(
    mask: np.ndarray,
    metres_per_pixel: float = 1.0,
    cell_size: tuple[float, float] | None = None,
    quad_gate: bool = True,
    smooth_px: float | None = None,
) -> dict[str, Any]:
    """Corners, sides, side types, class and grip point of one piece mask.

    ``cell_size`` is the grid cell (width, height) in metres when known. ``sides`` are in the
    piece's metric frame: x = u * mpp, y = -v * mpp (y up), each walked counter-clockwise
    round the piece from one corner to the next."""
    pts = outline(mask, smooth_px=OUTLINE_SMOOTH_PX if smooth_px is None else smooth_px)
    cell_px = None
    if cell_size is not None:
        cell_px = (cell_size[0] / metres_per_pixel, cell_size[1] / metres_per_pixel)
    found = find_corners(pts, quad_gate, cell_px)
    if found is None:
        raise ValueError("no 4-corner split of the outline")
    corners, ztypes = found
    vertices = pts[corners]
    if REFINE_SPAN is not None:
        scale = float(np.mean(cell_px)) if cell_px else _quad_side(pts[corners])
        refined = [refine_corner(pts, c, scale) for c in corners]
        order = np.argsort([r[0] for r in refined])
        if len({r[0] for r in refined}) == 4 and list(order) in (
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
        ):
            corners = [r[0] for r in refined]
            vertices = np.array([r[1] for r in refined])
            shift = int(np.argmin(corners))
            corners = corners[shift:] + corners[:shift]
            vertices = np.roll(vertices, -shift, axis=0)
            ztypes = ztypes[shift:] + ztypes[:shift]
    metric = np.column_stack([pts[:, 0], -pts[:, 1]]) * metres_per_pixel
    vmetric = np.column_stack([vertices[:, 0], -vertices[:, 1]]) * metres_per_pixel
    sides = side_points(metric, corners, vmetric)
    types = []
    for zt, side in zip(ztypes, sides, strict=True):
        types.append(zt if zt != UNDEFINED else chord_type(side)[0])
    return {
        "corners_px": [[float(x), float(y)] for x, y in vertices],
        "zolver_types": ztypes,
        "side_types": types,
        "piece_class": classify(types),
        "grip_pixel": grip_pixel(mask),
        "sides": sides,
        "metres_per_pixel": metres_per_pixel,
    }


# --- matching against the slot (Zolver Mover.stick_pieces, Distance.diff_match_edges2) ----


def _rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation R and translation t with R @ src + t ~ dst (2-D Kabsch)."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    h = (src - cs).T @ (dst - cd)
    ang = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    r = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    return r, cd - r @ cs


def _outward(p: np.ndarray) -> np.ndarray:
    """Unit outward normals of a side walked counter-clockwise round its piece."""
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    return np.column_stack([t[:, 1], -t[:, 0]])


def _fit_sides(pairs, offset: float, fit: str):
    """Rigid fit of the piece sides (pushed ``offset`` outward) onto the neighbour sides and
    the mean point distance over the side length."""
    moved_sides = [p + offset * _outward(p) for _, p, _ in pairs]
    if fit == "corners":
        src = np.vstack([[p[0], p[-1]] for p in moved_sides])
        dst = np.vstack([[q[0], q[-1]] for _, _, q in pairs])
    else:
        src = np.vstack(moved_sides)
        dst = np.vstack([q for _, _, q in pairs])
    r, t = _rigid(src, dst)
    per_side, total = {}, 0.0
    for (d, _, q), p in zip(pairs, moved_sides, strict=True):
        moved = p @ r.T + t
        length = float(np.linalg.norm(q[-1] - q[0]))
        dense = resample(q, 4 * SIDE_POINTS)
        dmat = np.linalg.norm(moved[:, None] - dense[None], axis=2)
        dist = dmat.min(axis=1)
        cost = 0.5 * (float(dist.mean()) + float(dmat.min(axis=0)[::4].mean())) / length
        per_side[d] = {
            "score": cost,
            "mismatch_fraction": float(np.mean(dist > MISMATCH_FRACTION * length)),
        }
        total += cost
    return total / len(pairs), r, t, per_side, offset


def _rotation_for(sides: list[np.ndarray], south: int) -> np.ndarray:
    chord = sides[south][-1] - sides[south][0]
    ang = -math.atan2(chord[1], chord[0])
    return np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])


def _oriented(direction: str, poly) -> np.ndarray:
    q = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    tx, ty = _TRAVERSE[direction]
    if (q[-1, 0] - q[0, 0]) * tx + (q[-1, 1] - q[0, 1]) * ty < 0:
        q = q[::-1]
    return q


def neighbour_type(direction: str, poly) -> str:
    """Type of the slot side a placed neighbour leaves, from the slot's point of view:
    ``tab`` when the neighbour's blank leaves room for a tab of the new piece, ``blank``
    when the neighbour's tab pokes into the slot."""
    t, _ = chord_type(_oriented(direction, poly))
    return t


def _compatible(piece_type: str, slot_type: str) -> bool:
    """Edge.is_compatible extended with flat: an undefined side matches anything."""
    if UNDEFINED in (piece_type, slot_type):
        return True
    return piece_type == slot_type


def score_rotation(
    piece: dict[str, Any],
    south: int,
    neighbours: dict[str, np.ndarray],
    border: set[str],
    fit: str = "corners",
    offsets: tuple[float, ...] = OFFSETS,
) -> dict[str, Any]:
    """Fit one piece with side ``south`` facing south against the slot."""
    sides, types = piece["sides"], piece["side_types"]
    compatible = True
    for i, d in enumerate(DIRECTIONS):
        t = types[(south + i) % 4]
        if d in neighbours:
            if not _compatible(t, neighbour_type(d, neighbours[d])):
                compatible = False
        elif d in border:
            compatible &= _compatible(t, FLAT)
        elif t == FLAT:
            compatible = False
    if neighbours:
        pairs = []
        for i, d in enumerate(DIRECTIONS):
            if d in neighbours:
                pairs.append((d, resample(sides[(south + i) % 4]), resample(neighbours[d])))
        length = float(np.mean([np.linalg.norm(q[-1] - q[0]) for _, _, q in pairs]))
        fits = [_fit_sides(pairs, f * length, fit) for f in offsets]
        score, r, t, per_side, offset = min(fits, key=lambda x: x[0])
        rotation = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    else:
        r, t = _rotation_for(sides, south), None
        per_side, score, offset = {}, 0.0, 0.0
        rotation = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    if not compatible:
        score += INCOMPATIBLE_PENALTY
    return {
        "south_side": south,
        "rotation_deg": rotation,
        "score": score,
        "compatible": compatible,
        "sides": per_side,
        "offset_m": offset,
        "_r": r,
        "_t": t,
    }


def _cell_from(cell_size, neighbours) -> tuple[float, float] | None:
    if cell_size:
        w, h = (float(x) for x in cell_size)
        return (w, h)
    chord = {d: float(np.linalg.norm(q[-1] - q[0])) for d, q in neighbours.items()}
    w = [chord[d] for d in ("north", "south") if d in chord]
    h = [chord[d] for d in ("east", "west") if d in chord]
    if not w and not h:
        return None
    w_, h_ = float(np.mean(w or h)), float(np.mean(h or w))
    return (w_, h_)


def run(
    ctx: NodeContext,
    candidates: list[dict],
    neighbour_sides: dict | None = None,
    border_sides: list[str] | None = None,
    metres_per_pixel: float = 0.0004,
    cell_size: list[float] | None = None,
    max_fit: float = MAX_FIT,
    min_margin: float | None = None,
    top_k: int = 5,
) -> MatchResult:
    """Rank candidate pieces and rotations for one slot by side shape.

    Each candidate is ``{"id", "mask"}`` (uint8 mask seen from above), optionally
    ``"metres_per_pixel"`` and ``"image_yaw_deg"`` (the board-frame angle of the image's
    +u axis; 0 when the image's +u is board east and -v board north). The returned
    rotation is how far to turn the piece, counter-clockwise seen from above, from where
    it lies to where it fits. ``neighbour_sides`` maps ``north / east / south / west`` of
    the slot to the facing side of the placed neighbour there, a polyline in board metres
    whose ends are the slot's corners. ``border_sides`` lists the slot's sides on the
    puzzle's outer edge. ``cell_size`` is the grid cell (width, height) in metres; without
    it the neighbour sides' chords stand in for it."""
    neighbours = {}
    for d, poly in (neighbour_sides or {}).items():
        if d not in _TRAVERSE:
            return {
                "route": "no_fit",
                "detail": f"unknown slot side {d!r}",
                "ranked": [],
                "pieces": [],
            }
        try:
            neighbours[d] = _oriented(d, poly)
        except ValueError:
            return {
                "route": "no_fit",
                "detail": f"slot side {d!r} is not a polyline",
                "ranked": [],
                "pieces": [],
            }
        if len(neighbours[d]) < 2:
            return {
                "route": "no_fit",
                "detail": f"slot side {d!r} has under 2 points",
                "ranked": [],
                "pieces": [],
            }
    border = set(border_sides or [])
    cell = _cell_from(cell_size, neighbours)
    pieces, ranked = [], []
    for k, cand in enumerate(candidates or []):
        cid = str(cand.get("id", k))
        mpp = float(cand.get("metres_per_pixel") or metres_per_pixel)
        try:
            info = analyse_piece(cand["mask"], mpp, cell_size=cell)
        except (KeyError, ValueError) as exc:
            pieces.append({"id": cid, "error": str(exc)})
            continue
        pieces.append({key: val for key, val in info.items() if key != "sides"} | {"id": cid})
        fits = [score_rotation(info, s, neighbours, border) for s in range(4)]
        best = min(fits, key=lambda f: f["score"])
        yaw = float(cand.get("image_yaw_deg") or 0.0)
        entry = {
            "id": cid,
            "rotation_deg": (best["rotation_deg"] - yaw + 180.0) % 360.0 - 180.0,
            "south_side": best["south_side"],
            "score": best["score"],
            "compatible": best["compatible"],
            "sides": best["sides"],
            "piece_class": info["piece_class"],
            "side_types": info["side_types"],
        }
        if best["_t"] is not None:
            g = np.array([info["grip_pixel"][0], -info["grip_pixel"][1]]) * mpp
            entry["grip_point_board"] = [float(x) for x in best["_r"] @ g + best["_t"]]
        entry["grip_pixel"] = info["grip_pixel"]
        ranked.append(entry)
    ranked.sort(key=lambda e: e["score"])
    out: MatchResult = {"ranked": ranked[: max(1, top_k)], "pieces": pieces}
    if not ranked:
        out.update(route="no_fit", detail="no candidate outline split into 4 sides")
        return out
    best = ranked[0]
    out.update(
        best_id=best["id"],
        best_rotation_deg=best["rotation_deg"],
        best_score=best["score"],
        piece_class=best["piece_class"],
        grip_pixel=best["grip_pixel"],
    )
    if "grip_point_board" in best:
        out["grip_point_board"] = best["grip_point_board"]
    runner = ranked[1]["score"] if len(ranked) > 1 else math.inf
    out["margin"] = runner / best["score"] if best["score"] > 0 else math.inf
    if not best["compatible"]:
        out.update(route="no_fit", detail="no candidate's side types fit the slot")
    elif not neighbours:
        n_ok = sum(e["compatible"] for e in ranked)
        out.update(
            route="fits" if n_ok == 1 else "ambiguous",
            detail=f"no placed neighbour: {n_ok} candidate(s) fit the side types only",
        )
    elif best["score"] > max_fit:
        out.update(route="no_fit", detail=f"best score {best['score']:.3f} over {max_fit}")
    elif out["margin"] < (
        min_margin or (MIN_MARGIN_ONE_SIDE if len(neighbours) == 1 else MIN_MARGIN)
    ):
        out.update(route="ambiguous", detail=f"runner-up within x{out['margin']:.2f} of the best")
    else:
        out.update(route="fits", detail=f"score {best['score']:.3f}, margin x{out['margin']:.2f}")
    return out
