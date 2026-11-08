"""Procedural interlocking jigsaw cut and wrist-camera-like piece masks.

Every interior edge of a ``cols x rows`` grid is one random Bezier side (neck, round head,
wavy shoulders, jittered grid corners) shared by the two pieces it separates, so mating
sides are exact complements: one piece walks it forwards, the other backwards. Board frame:
metres, +x east, +y north, row 0 at the south.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

DIRECTIONS = ("south", "east", "north", "west")


def _bezier(p, n=40):
    p = np.asarray(p, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (
        (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t**2 * p[2] + t**3 * p[3]
    )


def _tab_profile(rng):
    """One "classic" side in local coordinates: u along the edge 0..1, w the bump
    (fraction of length). Straight shoulders, a centred round head of similar size on
    every side."""
    c = rng.uniform(0.42, 0.58)
    n = rng.uniform(0.07, 0.10)
    r = rng.uniform(0.13, 0.17)
    h = rng.uniform(0.20, 0.27)
    sk = rng.uniform(-0.03, 0.03)
    j1, j2 = rng.uniform(-0.03, 0.03, 2)
    return _knob(c, n, r, h, sk, j1, j2)


def _wavy_profile(rng):
    """One "random cut" side: the whole side bends (a half and a full sine wave along it,
    up to ~13% of its length), the knob sits anywhere from 0.33 to 0.67 along it, its
    head is 0.12-0.22 of the side in radius and 0.22-0.36 high, and it leans up to 0.07
    either way. Imitates the wavy random-cut die of mass-market 500-1000 piece puzzles,
    whose sides are never straight and whose knobs differ in place and size."""
    n = rng.uniform(0.05, 0.10)
    r = rng.uniform(max(0.12, n + 0.05), 0.22)
    c = rng.uniform(max(0.33, r + 0.15), min(0.67, 0.85 - r))
    h = rng.uniform(0.22, 0.36)
    lean = rng.uniform(-0.07, 0.07)
    j1, j2 = rng.uniform(-0.05, 0.05, 2)
    pts = _knob(c, n, r, h, lean, j1, j2)
    a1, a2 = rng.uniform(-0.09, 0.09), rng.uniform(-0.05, 0.05)
    u = pts[:, 0]
    pts[:, 1] += a1 * np.sin(np.pi * u) + a2 * np.sin(2 * np.pi * u)
    return pts


def _knob(c, n, r, h, sk, j1, j2):
    """Shoulder, neck, round head, neck, shoulder as cubic Beziers; ``sk`` shifts the head
    along the side relative to the neck (a leaning knob)."""
    segs = [
        [(0, 0), (0.25, j1), (c - n - 0.08, 0.5 * j1), (c - n, 0)],
        [
            (c - n, 0),
            (c - n + 0.03, 0.15 * h),
            (c - r - 0.02 + sk, 0.35 * h),
            (c - r + sk, 0.6 * h),
        ],
        [(c - r + sk, 0.6 * h), (c - r + sk, 0.85 * h), (c - 0.6 * r + sk, h), (c + sk, h)],
        [(c + sk, h), (c + 0.6 * r + sk, h), (c + r + sk, 0.85 * h), (c + r + sk, 0.6 * h)],
        [
            (c + r + sk, 0.6 * h),
            (c + r + 0.02 + sk, 0.35 * h),
            (c + n - 0.03, 0.15 * h),
            (c + n, 0),
        ],
        [(c + n, 0), (c + n + 0.08, 0.5 * j2), (0.75, j2), (1, 0)],
    ]
    pts = np.vstack([_bezier(s)[:-1] for s in segs] + [[[1.0, 0.0]]])
    return pts


def _place(profile, a, b, sign):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    length = np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]]) / length
    return a + profile[:, :1] * d + sign * profile[:, 1:2] * length * nrm


@dataclass
class Puzzle:
    cols: int
    rows: int
    cell: tuple[float, float]
    vertices: np.ndarray  # (cols + 1, rows + 1, 2)
    h_edges: dict  # (i, j) -> (polyline V[i,j] -> V[i+1,j], sign or 0)
    v_edges: dict  # (i, j) -> (polyline V[i,j] -> V[i,j+1], sign or 0)

    def side(self, i: int, j: int, direction: str) -> np.ndarray:
        """Piece (i, j)'s side, walked counter-clockwise round the piece."""
        return self._side(i, j, direction)[0]

    def _side(self, i, j, direction):
        if direction == "south":
            poly, sign = self.h_edges[i, j]
            return poly, sign, +1
        if direction == "east":
            poly, sign = self.v_edges[i + 1, j]
            return poly, sign, +1
        if direction == "north":
            poly, sign = self.h_edges[i, j + 1]
            return poly[::-1], sign, -1
        poly, sign = self.v_edges[i, j]
        return poly[::-1], sign, -1

    def side_type(self, i: int, j: int, direction: str) -> str:
        _, sign, walk = self._side(i, j, direction)
        if sign == 0:
            return "flat"
        return "blank" if sign * walk > 0 else "tab"

    def side_types(self, i, j):
        return [self.side_type(i, j, d) for d in DIRECTIONS]

    def piece_class(self, i, j):
        flats = sum(t == "flat" for t in self.side_types(i, j))
        return {0: "interior", 1: "edge", 2: "corner"}[flats]

    def outline(self, i: int, j: int) -> np.ndarray:
        return np.vstack([self.side(i, j, d)[:-1] for d in DIRECTIONS])

    def is_simple(self, i: int, j: int, clearance: float = 0.04) -> bool:
        """The outline never comes within ``clearance`` of a cell of a part of itself more
        than half a cell away along it (two knobs or blanks at one corner touching)."""
        poly = np.vstack(
            [self.side(i, j, d)[:-1] for d in DIRECTIONS] + [self.side(i, j, "south")[:1]]
        )
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        t = np.linspace(0.0, s[-1], 240, endpoint=False)
        p = np.column_stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])])
        cell = min(self.cell)
        dist = np.linalg.norm(p[:, None] - p[None], axis=2)
        along = np.abs(t[:, None] - t[None])
        along = np.minimum(along, s[-1] - along)
        return not np.any((dist < clearance * cell) & (along > 0.5 * cell))

    def corners(self, i: int, j: int) -> np.ndarray:
        """South-west, south-east, north-east, north-west corners (counter-clockwise)."""
        v = self.vertices
        return np.array([v[i, j], v[i + 1, j], v[i + 1, j + 1], v[i, j + 1]])

    def centre(self, i, j):
        return self.corners(i, j).mean(axis=0)

    def facing_side(self, i: int, j: int, direction: str) -> np.ndarray:
        """What a placed neighbour shows the empty slot (i, j) on ``direction``."""
        di, dj, opposite = {
            "south": (0, -1, "north"),
            "north": (0, 1, "south"),
            "east": (1, 0, "west"),
            "west": (-1, 0, "east"),
        }[direction]
        return self.side(i + di, j + dj, opposite)

    def interior(self):
        return [(i, j) for i in range(1, self.cols - 1) for j in range(1, self.rows - 1)]

    def border_sides(self, i, j):
        out = []
        if j == 0:
            out.append("south")
        if j == self.rows - 1:
            out.append("north")
        if i == 0:
            out.append("west")
        if i == self.cols - 1:
            out.append("east")
        return out


def make_puzzle(
    cols: int,
    rows: int,
    cell=(0.02, 0.02),
    seed: int = 0,
    corner_jitter: float | None = None,
    style: str = "classic",
) -> Puzzle:
    """``style`` "classic" (straight shoulders, centred knob, corners jittered 4% of a cell)
    or "wavy" (random-cut sides from :func:`_wavy_profile`, corners jittered 7%)."""
    profile = {"classic": _tab_profile, "wavy": _wavy_profile}[style]
    if corner_jitter is None:
        corner_jitter = 0.04 if style == "classic" else 0.07
    rng = np.random.default_rng(seed)
    cw, ch = cell
    gx, gy = np.meshgrid(np.arange(cols + 1) * cw, np.arange(rows + 1) * ch, indexing="ij")
    v = np.stack([gx, gy], axis=-1).astype(np.float64)
    jx = rng.uniform(-corner_jitter, corner_jitter, (cols + 1, rows + 1)) * cw
    jy = rng.uniform(-corner_jitter, corner_jitter, (cols + 1, rows + 1)) * ch
    jx[0, :] = jx[-1, :] = 0
    jy[:, 0] = jy[:, -1] = 0
    v[..., 0] += jx
    v[..., 1] += jy
    h_edges, v_edges = {}, {}

    def draw(edges, key, a, b, straight):
        if straight:
            edges[key] = (np.linspace(a, b, 120), 0)
        else:
            sign = int(rng.choice([-1, 1]))
            edges[key] = (_place(profile(rng), a, b, sign), sign)

    for i in range(cols):
        for j in range(rows + 1):
            draw(h_edges, (i, j), v[i, j], v[i + 1, j], j in (0, rows))
    for i in range(cols + 1):
        for j in range(rows):
            draw(v_edges, (i, j), v[i, j], v[i, j + 1], i in (0, cols))
    puzzle = Puzzle(cols, rows, (cw, ch), v, h_edges, v_edges)
    check = {(i, j) for i in range(cols) for j in range(rows)}
    for _ in range(200):
        bad = [p for p in sorted(check) if not puzzle.is_simple(*p)]
        if not bad:
            return puzzle
        check = set()
        for i, j in bad:
            inner = [d for d in DIRECTIONS if d not in puzzle.border_sides(i, j)]
            d = inner[int(rng.integers(len(inner)))]
            if d == "south":
                draw(h_edges, (i, j), v[i, j], v[i + 1, j], False)
                check |= {(i, j), (i, j - 1)}
            elif d == "north":
                draw(h_edges, (i, j + 1), v[i, j + 1], v[i + 1, j + 1], False)
                check |= {(i, j), (i, j + 1)}
            elif d == "west":
                draw(v_edges, (i, j), v[i, j], v[i, j + 1], False)
                check |= {(i, j), (i - 1, j)}
            else:
                draw(v_edges, (i + 1, j), v[i + 1, j], v[i + 1, j + 1], False)
                check |= {(i, j), (i + 1, j)}
    raise RuntimeError("could not draw a cut whose pieces do not touch themselves")


def _smooth_noise(n, amplitude, rng, corr=12):
    """Closed-loop low-pass noise, peak about ``amplitude``."""
    if amplitude <= 0:
        return np.zeros(n)
    w = rng.normal(size=n)
    k = np.exp(-0.5 * (np.arange(-3 * corr, 3 * corr + 1) / corr) ** 2)
    x = np.real(
        np.fft.ifft(np.fft.fft(w) * np.fft.fft(np.roll(np.pad(k, (0, n - k.size)), -3 * corr)))
    )
    x /= np.max(np.abs(x)) or 1.0
    return amplitude * x


def jitter_polyline(poly, amplitude, rng, closed=False):
    poly = np.asarray(poly, float)
    t = np.gradient(poly, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    nrm = np.column_stack([-t[:, 1], t[:, 0]])
    return poly + nrm * _smooth_noise(len(poly), amplitude, rng)[:, None]


@dataclass
class Rendered:
    mask: np.ndarray
    angle_deg: float  # the piece image is the board view turned by this (counter-clockwise)
    corners_px: np.ndarray  # SW, SE, NE, NW of the cell, image pixels
    metres_per_pixel: float


def render_piece(
    puzzle: Puzzle,
    i: int,
    j: int,
    rng,
    px_per_cell: float = 64.0,
    angle_deg: float | None = None,
    noise: bool = True,
    margin: int = 12,
    jitter_px=(1.0, 2.0),
    blur_px=(0.8, 1.5),
    threshold=(0.35, 0.65),
    erode_probability: float = 0.5,
) -> Rendered:
    """A top-down mask of piece (i, j) turned by ``angle_deg``; with ``noise`` the outline is
    jittered 1-2 px (low-pass), the raster blurred (sigma 0.8-1.5 px), thresholded at
    0.35-0.65 and, half the time, eroded by 1 px."""
    return render_outline(
        puzzle.outline(i, j),
        puzzle.corners(i, j),
        puzzle.cell[0] / px_per_cell,
        rng,
        angle_deg=angle_deg,
        noise=noise,
        margin=margin,
        jitter_px=jitter_px,
        blur_px=blur_px,
        threshold=threshold,
        erode_probability=erode_probability,
    )


def render_outline(
    poly,
    corners,
    mpp: float,
    rng,
    angle_deg: float | None = None,
    noise: bool = True,
    margin: int = 12,
    jitter_px=(1.0, 2.0),
    blur_px=(0.8, 1.5),
    threshold=(0.35, 0.65),
    erode_probability: float = 0.5,
) -> Rendered:
    """:func:`render_piece` for any closed outline in board metres (counter-clockwise) and
    its 4 corners, at ``mpp`` metres per pixel."""
    if angle_deg is None:
        angle_deg = float(rng.uniform(0, 360))
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    poly = np.asarray(poly, dtype=np.float64)
    corners = np.asarray(corners, dtype=np.float64)
    c = corners.mean(axis=0)
    seg = np.linalg.norm(np.diff(np.vstack([poly, poly[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1], 0.25 * mpp)
    closed = np.vstack([poly, poly[:1]])
    poly = np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])

    def to_px(p):
        q = (np.asarray(p) - c) @ rot.T / mpp
        return np.column_stack([q[:, 0], -q[:, 1]])

    px = to_px(poly)
    if noise:
        px = jitter_polyline(px, rng.uniform(*jitter_px), rng)
    lo = px.min(axis=0) - margin
    size = (np.ceil(px.max(axis=0) - lo) + margin).astype(int)
    px = px - lo
    ss = 4
    big = np.zeros((size[1] * ss, size[0] * ss), np.uint8)
    cv2.fillPoly(big, [np.round(px * ss * 16).astype(np.int32)], 255, lineType=cv2.LINE_8, shift=4)
    img = cv2.resize(big, (size[0], size[1]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    if noise:
        if blur_px[1] > 0:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(*blur_px))
        mask = (img > rng.uniform(*threshold)).astype(np.uint8) * 255
        if rng.random() < erode_probability:
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    else:
        mask = (img > 0.5).astype(np.uint8) * 255
    return Rendered(mask, angle_deg, to_px(corners) - lo, mpp)
