"""Choose a two-finger grip on a jigsaw piece whose fingertips stay clear of the placed neighbours at its slot.

Written for the hub (no upstream code: the jigsaw robots found - puzzle-bot, Mark Rober's
puzzle_bot, DeepClaw - pick with suction at the piece's inner centre and never meet this
problem). The piece is rasterised at its final pose with the neighbours round it; for every
closing direction and every offset of the jaws across the piece, the fingers close on the
piece's extreme points inside their width, and each fingertip's footprint (plus the outward
travel to release) is checked against a distance transform of the neighbours. Among the clear
grips the one nearest the piece's centre with the most clearance wins. When none is clear at
the slot, the piece is shifted away from its neighbours to find the smallest offset at which
it can be put down and gripped (then pushed in).
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

RES_M = 0.0002
"""Raster resolution [m]: 0.2 mm, a tenth of a 2 mm fingertip margin."""

FINGER_WIDTH_M = 0.008
FINGER_THICKNESS_M = 0.004
"""Fingertip footprint on the table: width along the jaw (across the closing direction) and
thickness along the closing direction. Assumed for the YAM linear_4310 tapered tips (the
same 4 mm half-width segmenting-and-picking-pieces-classical assumes); measure the real tips
and pass them."""

RELEASE_TRAVEL_M = 0.001
"""Outward travel of each fingertip to let go of the piece before lifting."""

MIN_CLEARANCE_M = 0.0005
"""Least distance between a fingertip's footprint (with its release travel) and a placed
neighbour for the grip to count as clear: about 1.5 px of wrist-camera outline error at
0.3 mm per px. The arm's placement error is not in it; raise it to cover that."""

MAX_OPENING_M = 0.094
"""Widest the jaws open (YAM linear_4310, robot.describe_gripper width_fit at_open_m in the
hub's tests). A grip wider than this minus 2 * APPROACH_MARGIN_M is skipped."""

APPROACH_MARGIN_M = 0.003
"""Extra opening on each side while the open hand comes down over the piece at the pick."""

MIN_GRIP_M = 0.006
"""Narrowest span gripped: under it the fingers close on a tab's neck, which bends."""

ANGLE_STEP_DEG = 5.0
OFFSET_STEP_M = 0.001
"""Closing directions every 5 degrees over a half turn; jaw offsets every 1 mm across the
piece."""

TAB_REACH = 0.33
"""Without neighbour outlines, an occupied cell is its rectangle grown by this fraction of a
cell on every side, minus the piece itself. The hub's classic generated cut (straight
shoulders, centred knobs) reaches at most 0.30 cell past its cell (3 boards of 100 pieces);
its random-cut style reaches 0.47 at the 99th percentile and 0.51 at most, which no cell box
covers without blocking every grip: give outlines for such cuts (see SKILL.md)."""

CONTACT_TOL_M = 0.0003
"""Outline within this of a finger's contact line counts as touching it; the contact length
on each finger is part of the score."""

PAD_EDGE_M = 0.001
"""A contact must reach this far inside the pad's edge. Closing at 45 degrees across a
square's corner region touches the two sides at the pads' edges only: the sides are inclined
to the pads and the piece would squeeze out (it needs friction of 1). At 0.4 mm (two raster
cells) the raster's stair-steps on a 45 degree side still passed as a 0.28 mm contact."""

OFFSET_DIRECTIONS = 16
OFFSET_MAX_CELLS = 0.6
OFFSET_STEP_CELLS = 0.05
"""Offset search: 16 directions, up to 0.6 cell (past a tab's reach) in steps of 0.05 cell."""

PLACE_CLEARANCE_M = 0.0005
"""An offset placement must keep the piece itself this far from every neighbour."""

W_CENTRE, W_CONTACT, W_CLEAR = 1.0, 0.5, 0.5
"""Score weights: jaw offset from the piece's centroid (over half the piece's size), contact
length (over the finger's width, the shorter side), clearance (capped at 2 mm, over 2 mm)."""


class Output(TypedDict, total=False):
    route: str
    grip_point: list[float]
    closing_yaw_deg: float
    grip_width_m: float
    clearance_m: float
    contacts: list[list[float]]
    offset_m: list[float]
    pick_grip_pixel: list[float]
    pick_closing_yaw_deg: float
    candidates: list[dict[str, Any]]
    n_clear: int
    n_tried: int
    detail: str


def _rot(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def _outline_from_mask(mask: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty piece mask")
    return max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)


def body_centre_px(mask: np.ndarray, corners_px: Any = None) -> np.ndarray:
    """Centre of the piece's square body, (u, v) pixels: the mean of its four corners when
    given (matching-a-piece-to-placed-neighbours returns them), else the centre of the
    minimum-area rectangle of the mask opened by a disk of 0.15 of its side (knobs gone)."""
    if corners_px is not None:
        return np.asarray(corners_px, dtype=np.float64).reshape(4, 2).mean(axis=0)
    m = (np.asarray(mask) > 0).astype(np.uint8)
    side = math.sqrt(float(m.sum()))
    r = max(1, int(round(0.15 * side)))
    body = cv2.morphologyEx(
        m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    )
    pts = np.column_stack(np.nonzero(body if body.any() else m)[::-1]).astype(np.float32)
    (cx, cy), _, _ = cv2.minAreaRect(pts)
    return np.array([cx, cy])


def target_from_mask(
    mask: np.ndarray,
    metres_per_pixel: float,
    rotation_deg: float,
    slot_center: Any,
    image_yaw_deg: float = 0.0,
    corners_px: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The piece's outline at its slot in board metres, from a top-down mask. Returns the
    outline, the 2x2 rotation and the translation taking piece-image metres (x = u * mpp,
    y = -v * mpp) to the board."""
    out = _outline_from_mask(mask)
    centre = body_centre_px(mask, corners_px)
    to_metric = lambda p: np.column_stack([p[..., 0], -p[..., 1]]) * metres_per_pixel  # noqa: E731
    r = _rot(image_yaw_deg + rotation_deg)
    t = np.asarray(slot_center, dtype=np.float64) - r @ to_metric(centre[None])[0]
    return to_metric(out) @ r.T + t, r, t


def _cell_obstacles(cells, cell_size, origin, reach: float = TAB_REACH) -> list[np.ndarray]:
    w, h = (float(v) for v in cell_size)
    ox, oy = (float(v) for v in origin)
    g = reach
    out = []
    for c, r in cells:
        x0, y0 = ox + (c - g) * w, oy + (r - g) * h
        x1, y1 = ox + (c + 1 + g) * w, oy + (r + 1 + g) * h
        out.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    return out


class _Raster:
    def __init__(self, piece: np.ndarray, obstacles: list[np.ndarray], pad: float):
        lo = piece.min(axis=0) - pad
        hi = piece.max(axis=0) + pad
        self.lo = lo
        self.n = np.ceil((hi - lo) / RES_M).astype(int) + 1
        self.piece = self._fill([piece])
        obs = self._fill(obstacles) if obstacles else np.zeros(self.piece.shape, np.uint8)
        # The piece's own area is never an obstacle (a conservative cell box overlaps it).
        obs[cv2.dilate(self.piece, np.ones((3, 3), np.uint8)) > 0] = 0
        self.obstacle = obs
        free = (obs == 0).astype(np.uint8)
        self.dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * RES_M if obs.any() else None

    def _fill(self, polys) -> np.ndarray:
        m = np.zeros((self.n[1], self.n[0]), np.uint8)
        for p in polys:
            cv2.fillPoly(m, [np.round((np.asarray(p) - self.lo) / RES_M).astype(np.int32)], 1)
        return m

    def clearance(self, pts: np.ndarray) -> float:
        """Least distance from any of ``pts`` (board m) to an obstacle; inf without any."""
        if self.dist is None:
            return math.inf
        ij = np.round((pts - self.lo) / RES_M).astype(int)
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < self.n[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n[1])
        if not inside.all():
            # Outside the raster is farther than the pad from the piece: count it as free.
            ij = ij[inside]
        if len(ij) == 0:
            return math.inf
        return float(self.dist[ij[:, 1], ij[:, 0]].min())


def _grips(
    raster: _Raster, piece_pts: np.ndarray, centroid: np.ndarray, size: float, p: dict
) -> list[dict]:
    """Every (closing direction, jaw offset) grip on the rasterised piece, scored."""
    out = []
    fw, ft = p["finger_width"], p["finger_thickness"]
    travel = p["release_travel"]
    for ang in np.arange(0.0, 180.0, ANGLE_STEP_DEG):
        u = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        v = np.array([-u[1], u[0]])
        pu, pv = piece_pts @ u, piece_pts @ v
        cv = float(centroid @ v)
        for s in np.arange(pv.min() + 0.5 * fw, pv.max() - 0.5 * fw + 1e-9, OFFSET_STEP_M):
            band = np.abs(pv - s) <= 0.5 * fw
            if band.sum() < 3:
                continue
            umin, umax = float(pu[band].min()), float(pu[band].max())
            width = umax - umin
            if width < p["min_grip"] or width + 2 * APPROACH_MARGIN_M > p["max_opening"]:
                continue
            touch_a = band & (pu <= umin + CONTACT_TOL_M)
            touch_b = band & (pu >= umax - CONTACT_TOL_M)
            # A flat pad holds where the outline is tangent to it, inside its width. Touching
            # only at the pad's edge means the outline still slopes there: it squeezes out.
            inner = 0.5 * fw - PAD_EDGE_M
            if not (
                np.any(np.abs(pv[touch_a] - s) < inner) and np.any(np.abs(pv[touch_b] - s) < inner)
            ):
                continue
            contact_a = float(np.ptp(pv[touch_a]))
            contact_b = float(np.ptp(pv[touch_b]))
            # Fingertip footprints, each from its contact line outward by thickness + travel.
            su = np.arange(0.0, ft + travel + 1e-9, RES_M * 2)
            sv = np.arange(-0.5 * fw, 0.5 * fw + 1e-9, RES_M * 2)
            gu, gv = np.meshgrid(su, sv)
            gu, gv = gu.ravel(), gv.ravel()
            pa = np.outer(umin - gu, u) + np.outer(s + gv, v)
            pb = np.outer(umax + gu, u) + np.outer(s + gv, v)
            clear = min(raster.clearance(pa), raster.clearance(pb))
            centre = 0.5 * (umin + umax) * u + s * v
            off = abs(s - cv) / (0.5 * size)
            contact = min(contact_a, contact_b) / fw
            score = (
                W_CENTRE * (1.0 - min(1.0, off))
                + W_CONTACT * min(1.0, contact)
                + W_CLEAR * min(1.0, (clear if math.isfinite(clear) else 0.002) / 0.002)
            )
            out.append(
                {
                    "closing_yaw_deg": float(ang),
                    "grip_point": [float(centre[0]), float(centre[1])],
                    "grip_width_m": width,
                    "clearance_m": clear,
                    "contacts": [
                        [float(x) for x in umin * u + s * v],
                        [float(x) for x in umax * u + s * v],
                    ],
                    "contact_m": [contact_a, contact_b],
                    "score": float(score),
                }
            )
    return out


def run(
    ctx: NodeContext,
    target_outline: Any = None,
    piece_mask: Any = None,
    metres_per_pixel: float = 0.0,
    rotation_deg: float = 0.0,
    slot_center: Any = None,
    image_yaw_deg: float = 0.0,
    corners_px: Any = None,
    neighbour_outlines: Any = None,
    occupied_cells: Any = None,
    cell_size: Any = None,
    board_origin: Any = (0.0, 0.0),
    tab_reach: float = TAB_REACH,
    finger_width: float = FINGER_WIDTH_M,
    finger_thickness: float = FINGER_THICKNESS_M,
    release_travel: float = RELEASE_TRAVEL_M,
    min_clearance: float = MIN_CLEARANCE_M,
    max_opening: float = MAX_OPENING_M,
    min_grip: float = MIN_GRIP_M,
    allow_offset: bool = True,
    top_k: int = 5,
) -> Output:
    """Pick the contact pair and closing direction for placing one piece into its slot.

    The piece at its slot is ``target_outline`` (board metres) or comes from ``piece_mask``
    (top-down, ``metres_per_pixel``) turned by ``rotation_deg`` (counter-clockwise from above,
    from where it lies to where it fits, as matching-a-piece-to-placed-neighbours returns it;
    ``image_yaw_deg`` is the board angle of the image's +u axis) with its body centre on
    ``slot_center``. The placed neighbours are ``neighbour_outlines`` (board metres) or
    ``occupied_cells`` ([[col, row], ...]) with ``cell_size`` and ``board_origin`` (the
    corner of cell (0, 0); +x along columns, +y along rows)."""
    params = {
        "finger_width": float(finger_width),
        "finger_thickness": float(finger_thickness),
        "release_travel": float(release_travel),
        "max_opening": float(max_opening),
        "min_grip": float(min_grip),
    }
    r_place, t_place = None, None
    try:
        if target_outline is not None:
            piece = np.asarray(target_outline, dtype=np.float64).reshape(-1, 2)
        elif piece_mask is not None and metres_per_pixel > 0 and slot_center is not None:
            piece, r_place, t_place = target_from_mask(
                piece_mask, metres_per_pixel, rotation_deg, slot_center, image_yaw_deg, corners_px
            )
        else:
            return {
                "route": "no_clear_grip",
                "detail": "give target_outline, or piece_mask with metres_per_pixel and slot_center",
            }
    except ValueError as exc:
        return {"route": "no_clear_grip", "detail": str(exc)}
    if len(piece) < 3:
        return {"route": "no_clear_grip", "detail": "the piece outline has under 3 points"}

    obstacles: list[np.ndarray] = []
    if neighbour_outlines:
        obstacles = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in neighbour_outlines]
    elif occupied_cells:
        if cell_size is None:
            return {"route": "no_clear_grip", "detail": "occupied_cells need cell_size"}
        obstacles = _cell_obstacles(occupied_cells, cell_size, board_origin, tab_reach)

    size = float(np.max(np.ptp(piece, axis=0)))
    pad = 2.0 * (finger_thickness + release_travel) + max(finger_width, size) + 0.004
    out: Output = {}

    def solve(poly: np.ndarray):
        raster = _Raster(poly, obstacles, pad)
        ys, xs = np.nonzero(raster.piece)
        pts = np.column_stack([xs, ys]) * RES_M + raster.lo
        centroid = pts.mean(axis=0)
        grips = _grips(raster, pts, centroid, size, params)
        clear = [g for g in grips if g["clearance_m"] >= min_clearance]
        clear.sort(key=lambda g: -g["score"])
        return raster, grips, clear

    raster, grips, clear = solve(piece)
    out["n_tried"] = len(grips)
    out["n_clear"] = len(clear)
    offset = np.zeros(2)
    if not grips:
        out.update(
            route="no_clear_grip",
            detail="no closing direction spans the piece within the gripper's range",
        )
        return out
    if not clear and allow_offset and obstacles:
        cell = size / 1.25
        best = None
        for frac in np.arange(OFFSET_STEP_CELLS, OFFSET_MAX_CELLS + 1e-9, OFFSET_STEP_CELLS):
            for k in range(OFFSET_DIRECTIONS):
                a = 2 * math.pi * k / OFFSET_DIRECTIONS
                d = frac * cell * np.array([math.cos(a), math.sin(a)])
                moved = piece + d
                r2 = _Raster(moved, obstacles, pad)
                ys, xs = np.nonzero(r2.piece)
                body = np.column_stack([xs, ys]) * RES_M + r2.lo
                if r2.clearance(body) < PLACE_CLEARANCE_M:
                    continue
                g2 = [
                    g
                    for g in _grips(r2, body, body.mean(axis=0), size, params)
                    if g["clearance_m"] >= min_clearance
                ]
                if g2:
                    g2.sort(key=lambda g: -g["score"])
                    if best is None or g2[0]["score"] > best[1][0]["score"]:
                        best = (d, g2)
            if best is not None:
                break
        if best is not None:
            offset, clear = best
            out["offset_m"] = [float(offset[0]), float(offset[1])]

    pool = clear if clear else sorted(grips, key=lambda g: -g["clearance_m"])
    top = pool[0]
    out.update(
        grip_point=top["grip_point"],
        closing_yaw_deg=top["closing_yaw_deg"],
        grip_width_m=top["grip_width_m"],
        clearance_m=top["clearance_m"],
        contacts=top["contacts"],
        candidates=[{k: v for k, v in g.items() if k != "contacts"} for g in pool[: max(1, top_k)]],
    )
    if r_place is not None:
        # The same grip on the piece as it lies: board -> piece-image metres -> pixels.
        local = r_place.T @ (np.asarray(top["grip_point"]) - offset - t_place)
        out["pick_grip_pixel"] = [
            float(local[0] / metres_per_pixel),
            float(-local[1] / metres_per_pixel),
        ]
        out["pick_closing_yaw_deg"] = float(
            (top["closing_yaw_deg"] - rotation_deg - image_yaw_deg) % 180.0
        )
    if clear and not offset.any():
        out.update(
            route="grip_found",
            detail=(
                f"closing at {top['closing_yaw_deg']:.0f} deg, {top['grip_width_m'] * 1e3:.1f} mm wide,"
                f" {('%.1f mm' % (top['clearance_m'] * 1e3)) if math.isfinite(top['clearance_m']) else 'no'}"
                f" clearance to the neighbours ({len(clear)} of {len(grips)} grips clear)"
            ),
        )
    elif clear:
        out.update(
            route="only_offset_place",
            detail=(
                f"no clear grip at the slot; put it down {np.linalg.norm(offset) * 1e3:.1f} mm off toward"
                f" ({offset[0] / np.linalg.norm(offset):+.2f}, {offset[1] / np.linalg.norm(offset):+.2f})"
                " and push it in"
            ),
        )
    else:
        out.update(
            route="no_clear_grip",
            detail=(
                f"best clearance {top['clearance_m'] * 1e3:.1f} mm under {min_clearance * 1e3:.1f} mm and no"
                " offset placement clears"
            ),
        )
    return out
