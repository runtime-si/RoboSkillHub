"""Pure geometry for resolving-reconstruction-doubts-with-wrist-views: look poses round a
target, wrist RGB-D frames to base-frame points, and the measurements written back
into the twin (band-diameter profile under a top, height over the surface, lean from an
axis fit, footprint rectangle, position), each per view and fused, with the agreement
between views. No robot, no tools: everything here runs on saved frames.

A frame is a dict with ``rgb`` (H x W x 3 uint8), ``depth`` (H x W metres), ``K``
(3 x 3) and ``T`` (4 x 4 camera-to-base), plus an ``id``. The pen-shape capture's
npz files (``rgb``, ``depth_m``, ``K``, ``T_base_camera``) load with :func:`load_frame`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

NEAR, FAR = 0.07, 0.50
"""Depth kept [m]: a D405 reads nothing under ~7 cm and is best inside 50 cm (Intel's
ideal range at 640 x 480); the pen-shape views stood 0.15-0.25 m off."""

TOP_PERCENTILE = 99.0
"""Percentile of an object's point heights taken as its top. tuned_on: pen-shape capture
2026-11-09, green cap fused over 11 views: 99 -> 89.7 mm (base z), 95 -> 88.4 (into the
knob's rounded edge), max 91.0 (a few high pixels of the noisiest view)."""

EDGE_FRACTION = 0.5
EDGE_BIN_M = 0.002
"""The ``edge`` top: the highest 2 mm height bin with half the fullest bin's points.
tuned_on: pen-shape capture, black holder (dark pixels within 60 mm of its centre, under
60 mm over the table): dense to 24-26 mm, then a tail of dark pens to 60 mm; the edge
reads 28 mm fused and 24-29 mm in the views at 30-75 deg (a 99th percentile: 59 mm)."""

PEN_BANDS_MM = ((0, 3), (3, 6), (6, 10), (10, 16), (16, 24))
"""Default depth bands under the top [mm], the ones the pen caps were read in: 0-3 mm
held the knob (11-12 mm), 3-6 the knob's foot, 6-10 and 10-16 the cap (18-19 mm),
16-24 the cap's lower edge over the barrel."""

DIAMETER_PERCENTILE = 90.0
"""Radius percentile per band, doubled to a diameter. tuned_on: pen-shape capture, green
cap, fused: p90 -> 18.1 mm in the 6-10 mm band and 15.9 mm in the 16-24 mm one, against
the 18-19 mm cap and 16 mm barrel of the journal's reading of the same frames; the max is
pulled out by edge pixels mixing with the neighbours."""

MIN_BAND_POINTS = 20
"""Points a band needs before its diameter is read. tuned_on: pen-shape views, where a
green-cap band seen in fewer than 20 points read anywhere in 12-21 mm (4 cases)."""

MIN_COVERAGE = 0.8
"""Share of the 36 10-degree sectors round the axis a band's points must fill to count:
with less, a view saw one side only and its p90 radius is a guess. tuned_on: pen-shape
views, green cap, bands under its top round the fused axis: the 22 per-view bands under
0.8 read 9.5-20.4 mm; the 6-10 mm ones at or over 0.8 read 15.7-18.3 mm."""

SECTORS = 36

LEAN_SLICE_M = 0.004
"""Slice height for the lean fit [m]: each slice's centre is one point of the axis."""

MIN_LEAN_SLICES = 4
"""Slices with MIN_BAND_POINTS points the lean fit needs."""

FOOTPRINT_MIN_HEIGHT = 0.004
"""Points this far over the surface belong to the object's footprint [m]: above the
D405's depth noise on the table at 0.2-0.25 m (sd ~1 mm, Intel's 2 % RMS at 0.2 m)."""

FOOTPRINT_TRIM = (1.0, 99.0)
"""Percentiles of the points along each rectangle axis taken as its sides: one stray
pixel does not widen it."""

TOLERANCE = {
    "top_z": 0.005,
    "height_over_surface": 0.005,
    "diameter": 0.004,
    "lean_deg": 3.0,
    "footprint_side": 0.005,
    "position": 0.005,
}
"""How far one view may read from the fused value and still agree [m, deg]. tuned_on:
pen-shape capture, green cap: of the ten views that saw its top round, eight read it
87.1-90.8 mm (3.7 mm apart), the two at 85 deg 84.9 and 73.9 (the side, not the top),
against 89.7 fused, so 9 of 10 agree; the 6-10 mm band read 15.7-18.3 mm in the three
views that saw it round, against 18.1 fused. Lean: the pens' 3-7 deg spread."""

MIN_AGREEMENT = 0.5
"""Share of the supporting views that must agree with the fused value, else the
quantity is inconsistent between views."""

MIN_CHECK_VIEWS = 2
"""Views that must read a quantity on their own before their agreement is judged; with
fewer the value stands on the fused cloud alone and says ``checked: false``."""


# ----------------------------------------------------------------------------- poses


def view_direction(tilt_deg: float, azimuth_deg: float) -> np.ndarray:
    """Unit vector from the target to the camera: ``tilt_deg`` off straight above,
    toward ``azimuth_deg`` (from +x, counter-clockwise, in the base frame)."""
    t, a = math.radians(tilt_deg), math.radians(azimuth_deg)
    return np.array([math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), math.cos(t)])


def align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1.0e-9:
        if c > 0.0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1.0e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def look_pose(
    target: np.ndarray,
    tilt_deg: float,
    azimuth_deg: float,
    distance: float,
    hand_rotation: np.ndarray,
    tool_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The tool pose that puts the wrist camera's optical axis on ``target`` from
    ``distance``, looking down ``tilt_deg`` off vertical from ``azimuth_deg``: the hand
    is ``hand_rotation`` turned by the smallest rotation that aims the camera, so it keeps
    its roll. ``tool_cam`` is camera-to-tool. Returns (tool position, tool rotation, eye).
    The pen probe's look_pose (steps.py) with the camera's direction given instead of a
    pinch's."""
    u = view_direction(tilt_deg, azimuth_deg)
    rot = align(hand_rotation @ tool_cam[:3, 2], -u) @ hand_rotation
    eye = np.asarray(target, dtype=float) + distance * u
    return eye - rot @ tool_cam[:3, 3], rot, eye


def segment_distance(p0: np.ndarray, p1: np.ndarray, q: np.ndarray) -> float:
    """Distance from point ``q`` to the segment p0-p1."""
    p0, p1, q = (np.asarray(x, dtype=float) for x in (p0, p1, q))
    d = p1 - p0
    n = float(d @ d)
    s = 0.0 if n < 1.0e-12 else min(1.0, max(0.0, float((q - p0) @ d) / n))
    return float(np.linalg.norm(p0 + s * d - q))


def project(K: np.ndarray, T: np.ndarray, point: np.ndarray) -> tuple[float, float, float]:
    """Pixel (u, v) and depth of a base-frame point in a camera posed by ``T``."""
    pc = np.linalg.inv(T) @ np.append(np.asarray(point, dtype=float), 1.0)
    z = float(pc[2])
    if z <= 1.0e-6:
        return -1.0, -1.0, z
    return float(K[0, 0] * pc[0] / z + K[0, 2]), float(K[1, 1] * pc[1] / z + K[1, 2]), z


# ----------------------------------------------------------------------------- frames


def load_frame(path: str | Path) -> dict[str, Any]:
    """A saved frame (this skill's npz or the pen-shape probe's) as a frame dict."""
    with np.load(path) as f:
        depth = f["depth_m"] if "depth_m" in f.files else f["depth"]
        T = f["T_base_camera"] if "T_base_camera" in f.files else f["T"]
        return {
            "id": Path(path).stem,
            "rgb": np.asarray(f["rgb"]),
            "depth": np.asarray(depth, dtype=np.float32),
            "K": np.asarray(f["K"], dtype=float),
            "T": np.asarray(T, dtype=float),
        }


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """OpenCV's HSV scale (H 0-180, S and V 0-255) without OpenCV."""
    c = np.asarray(rgb, dtype=np.float64)[..., :3] / 255.0
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    v = c.max(-1)
    span = v - c.min(-1)
    s = np.where(v > 0, span / np.where(v > 0, v, 1.0), 0.0)
    safe = np.where(span > 0, span, 1.0)
    h = np.where(
        v == r,
        (g - b) / safe,
        np.where(v == g, 2.0 + (b - r) / safe, 4.0 + (r - g) / safe),
    )
    h = np.where(span > 0, (h * 60.0) % 360.0, 0.0)
    return np.stack([h / 2.0, s * 255.0, v * 255.0], -1)


def colour_mask(rgb: np.ndarray, h=None, s=None, v=None) -> np.ndarray:
    """Pixels inside the HSV ranges (OpenCV scale). A hue range with lo > hi wraps
    through red (e.g. [170, 8])."""
    hsv = rgb_to_hsv(rgb)
    m = np.ones(hsv.shape[:2], dtype=bool)
    if h is not None:
        lo, hi = float(h[0]), float(h[1])
        hh = hsv[..., 0]
        m &= ((hh >= lo) & (hh <= hi)) if lo <= hi else ((hh >= lo) | (hh <= hi))
    for k, rng in ((1, s), (2, v)):
        if rng is not None:
            m &= (hsv[..., k] >= float(rng[0])) & (hsv[..., k] <= float(rng[1]))
    return m


def back_project(frame: dict[str, Any], mask: np.ndarray | None = None) -> np.ndarray:
    """The frame's depth inside ``mask`` as base-frame points (N x 3). Depth smaller than
    the colour image is read at the colour pixel's scaled position."""
    depth = np.asarray(frame["depth"], dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    K, T = np.asarray(frame["K"], dtype=float), np.asarray(frame["T"], dtype=float)
    ok = np.isfinite(depth) & (depth > NEAR) & (depth < FAR)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        if m.shape != depth.shape:
            rows = (np.arange(depth.shape[0]) * m.shape[0] / depth.shape[0]).astype(int)
            cols = (np.arange(depth.shape[1]) * m.shape[1] / depth.shape[1]).astype(int)
            m = m[np.ix_(rows, cols)]
        ok &= m
    v, u = np.nonzero(ok)
    z = depth[v, u]
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    return cam @ T[:3, :3].T + T[:3, 3]


def crop_cylinder(
    P: np.ndarray, centre_xy, radius: float, z_min: float | None = None, iters: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Points within ``radius`` of a vertical axis, the axis moved to their median xy
    ``iters`` times (an object a few mm off its prior is followed, a neighbour further
    than ``radius`` is not). Returns (points, axis xy)."""
    c = np.asarray(centre_xy, dtype=float)[:2].copy()
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    if z_min is not None:
        P = P[P[:, 2] > z_min]
    Q = P[:0]
    for _ in range(max(1, iters)):
        Q = P[np.linalg.norm(P[:, :2] - c, axis=1) < radius]
        if len(Q) == 0:
            break
        c = np.median(Q[:, :2], axis=0)
    return Q, c


# ----------------------------------------------------------------------------- measures


def top_z(P: np.ndarray, percentile: float = TOP_PERCENTILE) -> float | None:
    return float(np.percentile(P[:, 2], percentile)) if len(P) else None


def edge_top_z(
    P: np.ndarray, fraction: float = EDGE_FRACTION, bin_m: float = EDGE_BIN_M
) -> float | None:
    """The top of the dense part of a cloud: the upper edge of the highest ``bin_m``
    height bin holding at least ``fraction`` of the fullest bin's points. For a body with
    other things standing on or in it (a holder full of pens), whose sparse points above
    would set a percentile top."""
    if len(P) == 0:
        return None
    z = P[:, 2]
    edges = np.arange(z.min(), z.max() + bin_m, bin_m)
    if len(edges) < 2:
        return float(z.max())
    h, edges = np.histogram(z, bins=edges)
    return float(edges[int(np.nonzero(h >= fraction * h.max())[0].max()) + 1])


def extent_centre(xy: np.ndarray, trim: float = 5.0) -> np.ndarray:
    """Middle of the points' ``trim``-``100 - trim`` percentile extent in x and y: where
    the points lie, not how densely (a median follows the side most views saw)."""
    lo = np.percentile(xy, trim, axis=0)
    hi = np.percentile(xy, 100.0 - trim, axis=0)
    return (lo + hi) / 2.0


def coverage(P: np.ndarray, centre_xy) -> float:
    """Share of the SECTORS round ``centre_xy`` that hold at least one point."""
    if len(P) == 0:
        return 0.0
    d = P[:, :2] - np.asarray(centre_xy, dtype=float)[:2]
    a = np.arctan2(d[:, 1], d[:, 0])
    k = np.floor((a + math.pi) / (2 * math.pi / SECTORS)).astype(int) % SECTORS
    return len(np.unique(k)) / SECTORS


def band_profile(
    P: np.ndarray,
    top: float,
    bands_mm=PEN_BANDS_MM,
    centres: list[np.ndarray | None] | None = None,
    min_points: int = MIN_BAND_POINTS,
    min_coverage: float = MIN_COVERAGE,
    percentile: float = DIAMETER_PERCENTILE,
) -> list[dict[str, Any]]:
    """Per band of depth under ``top``: the band's axis (the middle of its points' trimmed
    extent unless ``centres`` gives one, e.g. the fused band's for a single view; a median
    of a ring wanders by r / sqrt(n) and widens the p90), the share of sectors
    round it its points fill, and the ``percentile`` radius doubled (``diameter``, None
    when the band has fewer than ``min_points`` points or covers less than
    ``min_coverage``)."""
    out = []
    for i, (a, b) in enumerate(bands_mm):
        sel = (top - P[:, 2] >= a / 1000.0) & (top - P[:, 2] < b / 1000.0)
        B = P[sel]
        given = centres[i] if centres is not None and i < len(centres) else None
        c = (
            np.asarray(given, dtype=float)
            if given is not None
            else (extent_centre(B[:, :2]) if len(B) else None)
        )
        cov = coverage(B, c) if c is not None else 0.0
        d = None
        if c is not None and len(B) >= min_points and cov >= min_coverage:
            d = 2.0 * float(np.percentile(np.linalg.norm(B[:, :2] - c, axis=1), percentile))
        out.append(
            {
                "from_mm": float(a),
                "to_mm": float(b),
                "diameter": d,
                "points": int(len(B)),
                "coverage": round(cov, 3),
                "centre": None if c is None else [float(c[0]), float(c[1])],
            }
        )
    return out


def fit_lean(
    P: np.ndarray,
    top: float,
    depth: float,
    slice_m: float = LEAN_SLICE_M,
    min_points: int = MIN_BAND_POINTS,
    min_slices: int = MIN_LEAN_SLICES,
    min_coverage: float = MIN_COVERAGE,
) -> dict[str, Any] | None:
    """The object's axis from the centres of horizontal slices from ``top`` down
    ``depth``: a weighted line fit x, y = a + b z. Returns the lean off vertical (deg),
    the azimuth it leans toward (deg, from +x), the axis' xy at ``top`` and the fit's rms,
    or None with fewer than ``min_slices`` usable slices. A slice counts only when its
    points go round it (``min_coverage``), and its centre is the middle of its trimmed
    extent, not its median: views see different sides at different heights, which moves
    a median slice by slice (a 20-40 deg "lean" of an upright cap in the pen capture)."""
    zs, xy, w = [], [], []
    n = max(1, int(round(depth / slice_m)))
    for i in range(n):
        hi, lo = top - i * slice_m, top - (i + 1) * slice_m
        S = P[(P[:, 2] <= hi) & (P[:, 2] > lo)]
        if len(S) < min_points:
            continue
        c = extent_centre(S[:, :2])
        if coverage(S, c) >= min_coverage:
            zs.append(0.5 * (hi + lo))
            xy.append(c)
            w.append(math.sqrt(len(S)))
    if len(zs) < min_slices:
        return None
    z, xy, w = np.array(zs), np.array(xy), np.array(w)
    A = np.stack([np.ones_like(z), z - top], 1) * w[:, None]
    coef, *_ = np.linalg.lstsq(A, xy * w[:, None], rcond=None)
    slope = coef[1]
    resid = xy - (coef[0] + np.outer(z - top, slope))
    return {
        "lean_deg": math.degrees(math.atan(float(np.linalg.norm(slope)))),
        "azimuth_deg": math.degrees(math.atan2(float(slope[1]), float(slope[0]))),
        "top_xy": [float(coef[0][0]), float(coef[0][1])],
        "rms": float(np.sqrt((resid**2).sum(1).mean())),
        "slices": int(len(z)),
    }


def lean_from_base(top_xyz, base_xyz) -> dict[str, float]:
    """Lean of the line from a known base point (a socket, a contact) to the top centre."""
    d = np.asarray(top_xyz, dtype=float) - np.asarray(base_xyz, dtype=float)
    h = float(np.linalg.norm(d[:2]))
    return {
        "lean_deg": math.degrees(math.atan2(h, max(1.0e-9, float(d[2])))),
        "azimuth_deg": math.degrees(math.atan2(float(d[1]), float(d[0]))),
        "offset": h,
    }


def _hull(xy: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(np.asarray(xy, dtype=float), 6), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq):
        h: list[np.ndarray] = []
        for p in seq:
            while len(h) >= 2:
                a, b = h[-2], h[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 0:
                    h.pop()
                else:
                    break
            h.append(p)
        return h

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def footprint_rect(xy: np.ndarray, trim=FOOTPRINT_TRIM) -> dict[str, Any] | None:
    """The smallest-area rectangle round the points (its yaw from the convex hull's
    edges), its sides then taken between the ``trim`` percentiles along its axes.
    Returns centre, length (long side), width (short side) and yaw of the long side
    (deg, folded into (-90, 90])."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) < 3:
        return None
    hull = _hull(xy)
    if len(hull) < 3:
        return None
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        if np.linalg.norm(e) < 1.0e-9:
            continue
        a = math.atan2(e[1], e[0])
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        q = hull @ R.T
        area = float(np.ptp(q[:, 0]) * np.ptp(q[:, 1]))
        if best is None or area < best[0]:
            best = (area, a)
    if best is None:
        return None
    a = best[1]
    R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    q = xy @ R.T
    lo = np.percentile(q, trim[0], axis=0)
    hi = np.percentile(q, trim[1], axis=0)
    size = hi - lo
    centre = R.T @ ((lo + hi) / 2.0)
    if size[1] > size[0]:
        size = size[::-1]
        a += math.pi / 2
    yaw = math.degrees(a)
    yaw = (yaw + 90.0) % 180.0 - 90.0
    if yaw <= -90.0:
        yaw += 180.0
    return {
        "centre": [float(centre[0]), float(centre[1])],
        "length": float(size[0]),
        "width": float(size[1]),
        "yaw_deg": float(yaw),
    }


def agreement(values: list[float], fused: float, tol: float) -> float:
    """Share of the per-view values within ``tol`` of the fused one (1.0 for none)."""
    if not values:
        return 1.0
    return float(np.mean([abs(v - fused) <= tol for v in values]))


# ----------------------------------------------------------------------------- one doubt

QUANTITIES = ("height", "profile", "width", "footprint", "lean", "position")

DEFAULT_RADIUS = 0.03
"""Crop radius round the target's vertical axis when the doubt gives none [m]: a
graspable object (up to 60 mm across) and little of its neighbours. Touching neighbours
(the pen caps, 22 mm apart) need the doubt's own radius and a colour or a mask."""

POSITION_RADIUS = 0.06
"""Crop radius for a ``position`` doubt [m]: the overhead estimate on the YAM station is
1-2 cm off, a pushed holder 10 cm (runs/real-v1); further than this, re-run A."""

MIN_EDGE_POINTS = 200
"""Points a view needs for its own ``edge`` top: its 2 mm bins must hold tens of points.
tuned_on: pen-shape holder views, 300-3000 dark points each."""


def _xyz(v: Any) -> list[float]:
    if isinstance(v, dict):
        return [float(v["x"]), float(v["y"]), float(v["z"])]
    return [float(x) for x in v][:3]


def normalize_doubt(doubt: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """A doubt with every key filled: ``id``, ``point`` [x, y, z] (base frame; a Vec3
    or a list), ``what`` (a list of QUANTITIES), ``radius``, ``top`` (``percentile`` or
    ``edge``) and the optional ``colour``, ``bands_mm``, ``base_point``, ``lean_depth``,
    ``max_height``, ``object_name``, ``prior`` and ``write``."""
    if "point" not in doubt:
        raise ValueError(f"doubt {index} has no point")
    what = doubt.get("what") or ["height"]
    what = [what] if isinstance(what, str) else list(what)
    unknown = [w for w in what if w not in QUANTITIES]
    if unknown:
        raise ValueError(f"doubt {index}: unknown quantities {unknown}; known: {list(QUANTITIES)}")
    out = dict(doubt)
    out["id"] = str(doubt.get("id") or f"doubt{index}")
    out["point"] = _xyz(doubt["point"])
    out["what"] = what
    out["radius"] = float(
        doubt.get("radius") or (POSITION_RADIUS if "position" in what else DEFAULT_RADIUS)
    )
    out["top"] = str(doubt.get("top") or "percentile")
    if out["top"] not in ("percentile", "edge"):
        raise ValueError(f"doubt {index}: top must be 'percentile' or 'edge'")
    if doubt.get("base_point") is not None:
        out["base_point"] = _xyz(doubt["base_point"])
    return out


def select_points(
    frame: dict[str, Any],
    doubt: dict[str, Any],
    surface_z: float | None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """The doubted object's points in one frame: inside ``mask`` (a segmenter's) and the
    doubt's HSV ``colour`` ranges when given, above the surface (and under its
    ``max_height`` over it, which keeps the gripper's own fingers out), within the doubt's
    radius of the target's vertical axis."""
    m = None if mask is None else np.asarray(mask).astype(bool)
    colour = doubt.get("colour")
    if colour:
        cm = colour_mask(frame["rgb"], colour.get("h"), colour.get("s"), colour.get("v"))
        m = cm if m is None or m.shape != cm.shape else m & cm
    P = back_project(frame, m)
    if surface_z is not None:
        P = P[P[:, 2] > surface_z + FOOTPRINT_MIN_HEIGHT]
        if doubt.get("max_height") is not None:
            P = P[P[:, 2] < surface_z + float(doubt["max_height"])]
    Q, _ = crop_cylinder(P, doubt["point"][:2], float(doubt["radius"]), iters=1)
    return Q


def measure(
    views: list[tuple[str, np.ndarray]],
    doubt: dict[str, Any],
    surface_z: float | None,
) -> dict[str, Any]:
    """Every quantity the doubt asks for, fused over ``views`` ((id, points) pairs, the
    points already selected by :func:`select_points`), each with the views that
    supported it, their own readings and their agreement with the fused value.

    ``status``: ``measured`` (every asked quantity read, the views agree), ``inconsistent``
    (a quantity read, but fewer than MIN_AGREEMENT of its views agree with the fused
    value), ``no_view`` (no points, or an asked quantity no view could read)."""
    doubt = normalize_doubt(doubt) if "radius" not in doubt else doubt
    what, radius = doubt["what"], float(doubt["radius"])
    views = [(i, P) for i, P in views if len(P)]
    fused_all = np.concatenate([P for _, P in views]) if views else np.zeros((0, 3))
    fused, axis = crop_cylinder(fused_all, doubt["point"][:2], radius)
    if len(fused) < MIN_BAND_POINTS:
        return {
            "status": "no_view",
            "detail": f"{len(fused)} object points within {radius * 1e3:.0f} mm of the target "
            f"in {len(views)} views",
            "quantities": {},
            "points": int(len(fused)),
        }
    keep = {i: P[np.linalg.norm(P[:, :2] - axis, axis=1) < radius] for i, P in views}
    edge = doubt["top"] == "edge"
    top = edge_top_z(fused) if edge else top_z(fused)
    bands = [tuple(b) for b in doubt.get("bands_mm") or PEN_BANDS_MM]
    fused_bands = band_profile(fused, top, bands)
    centres = [b["centre"] for b in fused_bands]

    # A view reads the top only when it sees the top: round (percentile) or in bulk (edge).
    # Round its own centre: a view of an object that has moved since is judged, not dropped.
    top_views: dict[str, float] = {}
    for i, P in keep.items():
        if edge:
            if len(P) >= MIN_EDGE_POINTS:
                top_views[i] = float(edge_top_z(P))
        elif len(P) >= MIN_BAND_POINTS:
            b0 = band_profile(P, float(top_z(P)), bands[:1])[0]
            if b0["points"] >= MIN_BAND_POINTS and b0["coverage"] >= MIN_COVERAGE:
                top_views[i] = float(top_z(P))

    q: dict[str, Any] = {}
    failed: list[str] = []

    from_views = [i for i, P in keep.items() if len(P)]

    def checked(per_view: dict[str, float], value: float, tol: float) -> dict[str, Any]:
        agree = agreement(list(per_view.values()), value, tol)
        enough = len(per_view) >= MIN_CHECK_VIEWS
        return {
            "views": list(per_view),
            "per_view": per_view,
            "agreement": round(agree, 3),
            "checked": enough,
            "consistent": bool(agree >= MIN_AGREEMENT) if enough else True,
        }

    def judge(name: str, value: float, per_view: dict[str, float], tol: float, **extra) -> None:
        q[name] = {
            "value": value,
            "from_views": from_views,
            **checked(per_view, value, tol),
            **extra,
        }

    if "height" in what or "profile" in what:
        judge("top_z", top, top_views, TOLERANCE["top_z"], unit="m", method=doubt["top"])
        if surface_z is not None:
            judge(
                "height_over_surface",
                top - surface_z,
                {k: v - surface_z for k, v in top_views.items()},
                TOLERANCE["height_over_surface"],
                unit="m",
                surface_z=surface_z,
            )
        elif "height" in what:
            failed.append("height_over_surface: no surface_z")

    if "profile" in what:
        profile = []
        for k, band in enumerate(fused_bands):
            entry = dict(band)
            if band["diameter"] is not None:
                per_view = {}
                for i, P in keep.items():
                    d = band_profile(P, top, [bands[k]], centres=[centres[k]])[0]["diameter"]
                    if d is not None:
                        per_view[i] = d
                entry.update(checked(per_view, band["diameter"], TOLERANCE["diameter"]))
            profile.append(entry)
        read = [b for b in profile if b["diameter"] is not None]
        if read:
            q["profile"] = {
                "value": profile,
                "unit": "m",
                "from_views": from_views,
                "views": sorted({v for b in read for v in b["views"]}),
                "checked": any(b["checked"] for b in read),
                "consistent": all(b["consistent"] for b in read),
            }
        else:
            failed.append("profile: no band seen round enough")

    if "width" in what or "footprint" in what:
        rect = footprint_rect(fused[:, :2])
        # A view reads the footprint alone only when its points go round the object.
        per_rect = {
            i: r
            for i, P in keep.items()
            if len(P) >= MIN_BAND_POINTS
            and rect is not None
            and coverage(P, rect["centre"]) >= MIN_COVERAGE
            and (r := footprint_rect(P[:, :2])) is not None
        }
        if rect is None:
            failed.append("footprint: too few points")
        else:
            tol = TOLERANCE["footprint_side"]
            judge(
                "width", rect["width"], {i: r["width"] for i, r in per_rect.items()}, tol, unit="m"
            )
            judge(
                "length",
                rect["length"],
                {i: r["length"] for i, r in per_rect.items()},
                tol,
                unit="m",
            )
            if "footprint" in what:
                q["footprint"] = {
                    "value": rect,
                    "unit": "m",
                    "from_views": from_views,
                    "views": list(per_rect),
                    "checked": q["width"]["checked"],
                    "consistent": q["width"]["consistent"] and q["length"]["consistent"],
                }

    # The top's centre: the top band seen round has no one-sided bias (a side view's
    # median sits 2r/pi toward the camera, 6 mm on an 18.5 mm cap).
    top_band = fused[top - fused[:, 2] < bands[0][1] / 1000.0]
    top_centre = (
        np.asarray(centres[0], dtype=float)
        if fused_bands[0]["diameter"] is not None
        else (extent_centre(top_band[:, :2]) if len(top_band) else np.asarray(axis))
    )
    per_view_centre = {}
    for i in top_views:
        B = keep[i][top - keep[i][:, 2] < bands[0][1] / 1000.0]
        if len(B) >= MIN_BAND_POINTS:
            per_view_centre[i] = extent_centre(B[:, :2])

    if "position" in what:
        prior = np.asarray(doubt["point"], dtype=float)
        per_view = {i: float(np.linalg.norm(c - top_centre)) for i, c in per_view_centre.items()}
        q["position"] = {
            "value": [float(top_centre[0]), float(top_centre[1]), float(top)],
            "unit": "m",
            "offset_from_prior": float(np.linalg.norm(top_centre - prior[:2])),
            "from_views": from_views,
            **checked(per_view, 0.0, TOLERANCE["position"]),
        }

    if "lean" in what:
        base = doubt.get("base_point")
        if base is not None:
            lb = lean_from_base([*top_centre, top], base)
            per_view = {
                i: lean_from_base([*c, top_views[i]], base)["lean_deg"]
                for i, c in per_view_centre.items()
            }
            judge(
                "lean_deg",
                lb["lean_deg"],
                per_view,
                TOLERANCE["lean_deg"],
                unit="deg",
                azimuth_deg=lb["azimuth_deg"],
                offset=lb["offset"],
                method="base_point",
            )
        else:
            floor = surface_z + FOOTPRINT_MIN_HEIGHT if surface_z is not None else top - 0.03
            depth = float(doubt.get("lean_depth") or (top - floor))
            fit = fit_lean(fused, top, depth)
            if fit is None:
                failed.append("lean: fewer than 4 slices seen round")
            else:
                per_view = {
                    i: f["lean_deg"] for i, P in keep.items() if (f := fit_lean(P, top, depth))
                }
                judge(
                    "lean_deg",
                    fit["lean_deg"],
                    per_view,
                    TOLERANCE["lean_deg"],
                    unit="deg",
                    azimuth_deg=fit["azimuth_deg"],
                    rms=fit["rms"],
                    slices=fit["slices"],
                    method="axis_fit",
                )

    bad = [k for k, v in q.items() if not v.get("consistent", True)]
    if bad:
        status, detail = "inconsistent", f"views disagree on {', '.join(bad)}"
    elif failed:
        status, detail = "no_view", "; ".join(failed)
    else:
        status, detail = "measured", f"{len(q)} quantities from {len(views)} views"
    return {
        "status": status,
        "detail": detail,
        "quantities": q,
        "points": int(len(fused)),
        "axis_xy": [float(axis[0]), float(axis[1])],
    }
