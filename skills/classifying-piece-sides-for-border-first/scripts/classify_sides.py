"""Type a jigsaw piece's four sides (flat / tab / blank) from a top-down mask, and sort pieces for a border-first build.

Written for the hub (no code copied). The piece's body is its mask opened by a disk wider than a knob's neck
(the knobs fall away, the square body stays); the body's minimum-area rectangle gives the four side lines.
Each outline point belongs to the side whose line it is nearest in the rectangle's normalised frame; a side
is a tab where its middle stretch reaches past the line, a blank where it falls behind it, and flat where it
stays on it. Two flat sides make a corner piece, one an edge piece.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

CONTOUR_POINTS = 480
"""The outline is resampled to this many points, evenly by arc length, before the line search."""

ANGLE_STEP_DEG = 1.0
"""The side lines' direction is searched over a quarter turn in these steps."""

BIN_FRACTION = 0.015
"""Histogram bin of the projected outline, as a fraction of its extent: 1.5 % (about 1 px at 64 px a
piece). A straight side piles its points into one or two bins; a knob spreads them."""

MAX_ASPECT = 1.6
"""Widest body side over the narrowest still taken for a piece (real cuts: 1.0-1.3)."""

MIDDLE = 0.7
"""Only the middle 70 % of each side is used to type it: the corners' rounding and a neighbouring side's
knob root live in the outer 15 % either end."""

KNOB_FRACTION = 0.12
"""A side is a tab (blank) where its outline reaches this fraction of the side's length past (behind) the
side line. Knobs reach 0.20-0.30 of a side; a flat side's wobble from the mask stays under 0.05."""

UNCERTAIN = 0.35
"""A side is uncertain when its deciding deviation lies within this fraction of KNOB_FRACTION of the
threshold (between 0.078 and 0.162 of the side)."""

MIN_AREA_PX = 400
"""Smallest mask (pixels) classified: a 20 x 20 px piece. Below that a knob is a few pixels."""

SIDE_NAMES = ("right", "top", "left", "bottom")
"""Sides in counter-clockwise order in the image seen with y up (the rectangle's +u, +v, -u, -v)."""


class Output(TypedDict, total=False):
    route: str
    piece_class: str
    side_types: list[str]
    side_normals_deg: list[float]
    flat_normals_deg: list[float]
    side_deviations: list[float]
    uncertain_sides: int
    body_angle_deg: float
    pieces: list[dict[str, Any]]
    corners: list[str]
    edges: list[str]
    interiors: list[str]
    uncertain: list[str]
    detail: str


def _mask(m: Any) -> np.ndarray:
    a = np.asarray(m)
    if a.dtype == bool:
        return a.astype(np.uint8)
    return (a > (127 if a.dtype == np.uint8 and a.max() > 1 else 0)).astype(np.uint8)


def _resample(poly: np.ndarray, n: int) -> np.ndarray:
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


def _peak(proj: np.ndarray, lo: float, hi: float, binw: float, upper: bool) -> tuple[float, int]:
    """Position and height of the densest bin of ``proj`` in the outer half toward ``hi`` (or ``lo``)."""
    mid = 0.5 * (lo + hi)
    sel = proj[proj >= mid] if upper else proj[proj <= mid]
    if len(sel) == 0:
        return (hi if upper else lo), 0
    edges = np.arange(lo, hi + binw, binw)
    hist, edges = np.histogram(sel, bins=edges)
    # Two adjacent bins together: a line falling on a bin edge splits its points.
    pair = hist[:-1] + hist[1:]
    k = int(np.argmax(pair))
    near = sel[(sel >= edges[k]) & (sel < edges[k + 2])]
    return float(np.median(near)), int(pair[k])


def _side_lines(pts: np.ndarray) -> tuple[float, tuple[float, float, float, float]]:
    """Direction (rad) and the four side lines (u_min, u_max, v_min, v_max) of a piece's outline: the
    direction whose projections pile the outline's points into the sharpest peak near each extreme
    (straight shoulders and flat sides), and those peaks."""
    best = None
    for deg in np.arange(0.0, 90.0, ANGLE_STEP_DEG):
        a = math.radians(deg)
        eu = np.array([math.cos(a), math.sin(a)])
        ev = np.array([-eu[1], eu[0]])
        pu, pv = pts @ eu, pts @ ev
        lines, score = [], 0
        for proj in (pu, pv):
            lo, hi = float(proj.min()), float(proj.max())
            binw = max(1e-6, BIN_FRACTION * (hi - lo))
            for upper in (False, True):
                pos, h = _peak(proj, lo, hi, binw, upper)
                lines.append(pos)
                score += h
        if best is None or score > best[0]:
            best = (score, a, tuple(lines))
    return best[1], best[2]


def classify_mask(mask: Any) -> dict[str, Any]:
    """Side types of one piece mask. Raises ValueError for an unusable mask."""
    m = _mask(mask)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        raise ValueError("empty mask")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    area = float(m.sum())
    if area < MIN_AREA_PX:
        raise ValueError(f"mask of {int(area)} px is too small")
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    raw = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    pts = _resample(np.column_stack([raw[:, 0], -raw[:, 1]]), CONTOUR_POINTS)  # y up
    theta, lines = _side_lines(pts)
    eu = np.array([math.cos(theta), math.sin(theta)])
    ev = np.array([-eu[1], eu[0]])
    u0, u1, v0, v1 = lines
    c = 0.5 * (u0 + u1) * eu + 0.5 * (v0 + v1) * ev
    hw, hh = 0.5 * (u1 - u0), 0.5 * (v1 - v0)
    if min(hw, hh) <= 0 or max(hw, hh) / min(hw, hh) > MAX_ASPECT:
        raise ValueError("no square body found")
    rel = pts - c
    u, v = rel @ eu, rel @ ev
    nu, nv = u / hw, v / hh
    side_of = np.where(np.abs(nu) >= np.abs(nv), np.where(nu >= 0, 0, 2), np.where(nv >= 0, 1, 3))
    types, devs, uncertain = [], [], 0
    for s in range(4):
        along = v if s in (0, 2) else u
        half = hh if s in (0, 2) else hw
        length = 2 * half
        across = (u if s in (0, 2) else v) * (1 if s in (0, 1) else -1)
        sel = (side_of == s) & (np.abs(along) <= MIDDLE * half)
        if sel.sum() < 3:
            types.append("undefined")
            devs.append(0.0)
            uncertain += 1
            continue
        d = (across[sel] - (hw if s in (0, 2) else hh)) / length
        hi, lo = float(d.max()), float(d.min())
        dev = hi if hi >= -lo else lo
        devs.append(dev)
        if abs(dev) < KNOB_FRACTION:
            types.append("flat")
        else:
            types.append("tab" if dev > 0 else "blank")
        if abs(abs(dev) - KNOB_FRACTION) < UNCERTAIN * KNOB_FRACTION:
            uncertain += 1
    base = math.degrees(theta)
    normals = [(base + 90.0 * s + 180.0) % 360.0 - 180.0 for s in range(4)]
    flats = sum(t == "flat" for t in types)
    cls = {0: "interior", 1: "edge", 2: "corner"}.get(flats, "unknown")
    return {
        "piece_class": cls,
        "side_types": types,
        "side_normals_deg": normals,
        "flat_normals_deg": [nrm for t, nrm in zip(types, normals, strict=True) if t == "flat"],
        "side_deviations": devs,
        "uncertain_sides": uncertain,
        "body_angle_deg": base,
    }


def run(
    ctx: NodeContext,
    piece_mask: Any = None,
    pieces: Any = None,
    max_uncertain: int = 0,
) -> Output:
    """Classify one piece (``piece_mask``: a top-down mask) or many (``pieces``: [{"id", "mask"}, ...]).

    One piece: its class (corner / edge / interior), side types counter-clockwise from the body's +u side
    with their outward normals (image degrees, counter-clockwise from +x with y up) and the flat sides'
    normals - the direction the board's edge must lie. Many: the same per piece, plus the ids sorted into
    corners, edges and interiors for a border-first build, and the uncertain ones. A piece with more than
    ``max_uncertain`` sides near the threshold is uncertain (look again closer, or match by shape)."""
    if pieces is not None:
        recs, groups = [], {"corner": [], "edge": [], "interior": [], "uncertain": []}
        for i, p in enumerate(pieces):
            pid = str(p.get("id", i))
            try:
                info = classify_mask(p["mask"])
            except (KeyError, ValueError) as exc:
                recs.append({"id": pid, "error": str(exc)})
                groups["uncertain"].append(pid)
                continue
            recs.append({"id": pid, **info})
            if info["uncertain_sides"] > max_uncertain or info["piece_class"] == "unknown":
                groups["uncertain"].append(pid)
            else:
                groups[info["piece_class"]].append(pid)
        if not any(r for r in recs if "error" not in r):
            return {"route": "no_piece", "pieces": recs, "detail": "no usable piece mask"}
        return {
            "route": "classified",
            "pieces": recs,
            "corners": groups["corner"],
            "edges": groups["edge"],
            "interiors": groups["interior"],
            "uncertain": groups["uncertain"],
            "detail": (
                f"{len(groups['corner'])} corner, {len(groups['edge'])} edge, {len(groups['interior'])} interior,"
                f" {len(groups['uncertain'])} uncertain of {len(recs)}"
            ),
        }
    if piece_mask is None:
        return {"route": "no_piece", "detail": "give piece_mask or pieces"}
    try:
        info = classify_mask(piece_mask)
    except ValueError as exc:
        return {"route": "no_piece", "detail": str(exc)}
    route = "classified"
    if info["uncertain_sides"] > max_uncertain or info["piece_class"] == "unknown":
        route = "uncertain"
    return {
        "route": route,
        **info,
        "detail": f"{info['piece_class']}: {' '.join(info['side_types'])} ({info['uncertain_sides']} uncertain)",
    }
