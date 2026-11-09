"""Judge from one close wrist depth frame whether a just-placed jigsaw piece lies flush.

The reference surface is a plane fitted by RANSAC to the placed neighbours' tops (or the table
round the piece), as Open3D's ``PointCloud::segment_plane`` does
(https://github.com/isl-org/Open3D/blob/main/cpp/open3d/geometry/PointCloudSegmentation.cpp,
MIT): random 3-point planes, the one with the most points within a distance threshold, then a
least-squares refit on its inliers. Reimplemented in NumPy here (no code copied; Open3D is not
a hub dependency). The piece's own points are then measured against that plane: their median
height (raised / sunk), a plane fitted to them (tilted, and which side is high) and the seam
band between the piece and its neighbours (open seam).
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext
from gap_core.types import CameraFrame

PIECE_THICKNESS_M = 0.002
"""Assumed thickness of a cardboard jigsaw piece (common 500-1000 piece puzzles are about
2 mm; the target puzzle has not been measured). Used when the reference is the table, and
for the seam test."""

RAISED_M = 0.0005
"""Median height of the piece over its expected top beyond which it is ``raised``. Real D405
frames at 8-16 cm (the station's wood table): per-pixel plane-fit noise 0.39-0.44 mm std
(60 x 60 px patches); a flat piece-sized disc against the ring round it read a level of
0.06 mm median, 0.19 mm 90th percentile, 0.37 mm 99th, 0.44 mm max (113 windows)."""

SUNK_M = 0.0006
"""Median height this far below the expected top: ``gap`` (kind ``low``) - the neighbours
are raised, or the piece is in the wrong slot under a tab."""

TILT_RISE_M = 0.001
"""Rise of the plane fitted to the piece's top across the piece (its highest minus its
lowest point inside the piece) beyond which it is ``tilted``: one side or a tab is up.
Real D405 depth of the station's wood table, 113 windows at 8-16 cm, a flat piece-sized disc
against the ring round it: rise 0.30 mm median, 0.77 mm 90th percentile, 1.16 mm 99th (the
sensor's low-frequency error and the table's own relief; a quadratic reference did not help).
1 mm (2.5 degrees across a 22.5 mm piece) is the 95th percentile or so; a tab resting on a
neighbour raises one side by a whole thickness."""

EDGE_MARGIN_M = 0.0012
"""The piece mask is eroded and the reference region kept this far from the piece: the D405
smooths depth over about 3 px (1 mm at 12 cm) at a 2 mm step."""

RING_M = (0.002, 0.012)
"""Without neighbour outlines, the reference points are an annulus this far round the piece."""

SEAM_BAND_M = 0.0025
"""The seam band: the strip this far either side of the piece's outline, where it faces a
neighbour. A point there lower than the reference by half a thickness is an open seam (the
table shows). It straddles the outline because the outline is usually the planned slot: a
piece that stopped short leaves its gap inside the slot."""

SEAM_MIN_LENGTH_M = 0.003
"""Shortest run of open seam along the outline that counts as ``gap``: about a knob's neck.
Single noisy pixels make runs of a pixel or two (0.3-0.6 mm)."""

RANSAC_THRESHOLD_M = 0.0008
RANSAC_ITERATIONS = 200
"""segment_plane's distance threshold and iterations: 2 sigma of the per-pixel noise; the
inlier set of a 2-level ring (table and neighbour tops 2 mm apart) is then one level."""

MAX_PLANE_TILT_DEG = 10.0
"""RANSAC planes tilted more than this from the world's horizontal are not tried: a ring of
table and neighbour tops 2 mm apart otherwise admits a plane sloping from one to the other
(seen on the synthetic scenes). The world z must be the table's normal to within this."""

MIN_VALID_FRACTION = 0.4
"""Least fraction of the piece's (eroded) pixels with valid depth. Glossy black print and
the black breadboard of the station drop 30-60 % of D405 pixels at 10-16 cm."""

MIN_POINTS = 150
"""Least valid points on the piece and in the reference region."""

MIN_LEVEL_FRACTION = 0.15
"""In the ring, a level (table, or neighbours' tops a thickness above it) counts when this
fraction of the ring's points lie on it."""

TABLE_RULE = 0.75
"""Ring of one level, no table_z: it is taken for the table when the piece's median sits more
than this many thicknesses above it, else for the neighbours' tops. A piece riding up by more
than this on a ring of neighbours only is then misread as flat on the table: give
``neighbour_outlines`` or ``table_z``."""


class Output(TypedDict, total=False):
    route: str
    level_m: float
    tilt_rise_m: float
    tilt_deg: float
    high_side: list[float]
    residual_p90_m: float
    uncovered_fraction: float
    seam_gap_m: float
    seam_gap_points: list[list[float]]
    gap_kind: str
    reference: str
    valid_fraction: float
    n_piece_points: int
    n_reference_points: int
    detail: str


def _pose_matrix(pose: Any) -> np.ndarray:
    if pose is None:
        return np.eye(4)
    arr = np.asarray(pose, dtype=object) if not isinstance(pose, dict) else None
    if arr is not None and arr.shape == (4, 4):
        return np.asarray(pose, dtype=np.float64)
    p, q = pose["position"], pose["rotation"]
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
    return t


def _project(
    poly_world: np.ndarray, z: float, k: np.ndarray, cam_to_world: np.ndarray
) -> np.ndarray:
    pts = np.column_stack([poly_world, np.full(len(poly_world), z), np.ones(len(poly_world))])
    cam = (np.linalg.inv(cam_to_world) @ pts.T).T[:, :3]
    cam[:, 2] = np.maximum(cam[:, 2], 1e-6)
    return np.column_stack(
        [k[0, 0] * cam[:, 0] / cam[:, 2] + k[0, 2], k[1, 1] * cam[:, 1] / cam[:, 2] + k[1, 2]]
    )


def _fill(poly_px: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(poly_px).astype(np.int32)], 1)
    return m.astype(bool)


def _disk(radius_px: float) -> np.ndarray:
    r = max(1, int(round(radius_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def segment_plane(
    pts: np.ndarray,
    threshold: float = RANSAC_THRESHOLD_M,
    iterations: int = RANSAC_ITERATIONS,
    seed: int = 0,
    max_tilt_deg: float = MAX_PLANE_TILT_DEG,
    offsets: tuple[float, ...] = (0.0,),
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC plane (Open3D segment_plane): returns (n, d) with n . p + d = 0, |n| = 1, and
    the inlier mask, refit by least squares on the inliers. Candidate planes tilted more than
    ``max_tilt_deg`` from the world's horizontal are skipped (not in Open3D). With
    ``offsets`` a candidate also collects the points near its parallel copies at those
    heights (a two-level surface: table and piece tops); the refit uses its own level only."""
    min_nz = math.cos(math.radians(max_tilt_deg))
    rng = np.random.default_rng(seed)
    best, best_count = None, -1
    n_pts = len(pts)
    for _ in range(iterations):
        a, b, c = pts[rng.choice(n_pts, 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        if abs(n[2]) < min_nz:
            continue
        d = -float(n @ a)
        if n[2] < 0:
            n, d = -n, -d
        h = pts @ n + d
        count = int(sum(np.sum(np.abs(h - o) < threshold) for o in offsets))
        if count > best_count:
            best, best_count = (n, d), count
    if best is None:
        best = (np.array([0.0, 0.0, 1.0]), -float(np.median(pts[:, 2])))
    n, d = best
    inl = np.abs(pts @ n + d) < threshold
    c = pts[inl].mean(axis=0)
    _, _, vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
    n = vt[2]
    d = -float(n @ c)
    inl = np.abs(pts @ n + d) < threshold
    return np.array([*n, d]), inl


def _points(depth: np.ndarray, k: np.ndarray, cam_to_world: np.ndarray, mask: np.ndarray):
    v, u = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    z = depth[v, u].astype(np.float64)
    cam = np.column_stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z])
    return cam @ cam_to_world[:3, :3].T + cam_to_world[:3, 3], np.column_stack([u, v])


def _refit(pts: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane through ``pts`` (normal kept upward); ``n`` if too few points."""
    if len(pts) < 3:
        return n, -float(np.median(pts @ n)) if len(pts) else 0.0
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    m = vt[2] if vt[2][2] > 0 else -vt[2]
    return m, -float(m @ c)


def _open_seam(piece, neighbours, depth, k, cam_to_world, px_m, height, thickness):
    """Longest run of the outline facing a neighbour along which an open seam shows: on the
    line across the outline (SEAM_BAND_M either way) a stretch lower than the reference by
    half a thickness with surface at the reference level on both sides of it. A free corner
    (the table of an empty diagonal cell) is low on one side only."""
    contour = max(
        cv2.findContours(piece.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
        key=cv2.contourArea,
    )[:, 0, :].astype(np.float64)
    band_px = SEAM_BAND_M / px_m
    near_nb = cv2.dilate(neighbours.astype(np.uint8), _disk(band_px)).astype(bool)
    ci = contour.astype(int)
    facing = near_nb[ci[:, 1], ci[:, 0]]
    if not facing.any():
        return 0.0, []
    shape = depth.shape
    region = cv2.dilate(piece.astype(np.uint8), _disk(band_px + 2)).astype(bool)
    pts, px = _points(depth, k, cam_to_world, region)
    hmap = np.full(shape, np.nan)
    hmap[px[:, 1], px[:, 0]] = height(pts)
    # Outward normals of the (counter-clockwise in the image) contour, smoothed over ~1 mm.
    w = max(2, int(round(0.001 / px_m)))
    tangent = np.roll(contour, -w, axis=0) - np.roll(contour, w, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    probe = contour + 2.0 * normal
    pi = np.clip(np.round(probe).astype(int), 0, [shape[1] - 1, shape[0] - 1])
    if piece[pi[:, 1], pi[:, 0]].mean() > 0.5:
        normal = -normal
    steps = np.arange(-band_px, band_px + 0.5, 0.5)
    low_t, high_t = -0.5 * thickness, -0.25 * thickness
    hit = np.zeros(len(contour), bool)
    gap_px = []
    for i in np.nonzero(facing)[0]:
        line = contour[i] + steps[:, None] * normal[i]
        li = np.round(line).astype(int)
        ok = (li[:, 0] >= 0) & (li[:, 0] < shape[1]) & (li[:, 1] >= 0) & (li[:, 1] < shape[0])
        vals = np.full(len(steps), np.nan)
        vals[ok] = hmap[li[ok, 1], li[ok, 0]]
        low = vals < low_t
        if not low.any():
            continue
        first, last = int(np.argmax(low)), len(low) - 1 - int(np.argmax(low[::-1]))
        if np.any(vals[:first] >= high_t) and np.any(vals[last + 1 :] >= high_t):
            hit[i] = True
            gap_px.append(line[(first + last) // 2])
    if not hit.any():
        return 0.0, []
    run_len = best = 0
    for v in np.concatenate([hit, hit]):
        run_len = run_len + 1 if v else 0
        best = max(best, min(run_len, len(hit)))
    step = float(np.mean(np.linalg.norm(np.diff(contour, axis=0), axis=1)))
    sample = gap_px[:: max(1, len(gap_px) // 50)]
    world = []
    for u, v in sample:
        z = depth[int(round(v)), int(round(u))]
        if np.isfinite(z) and z > 0:
            cam = np.array([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z])
            w_pt = cam_to_world[:3, :3] @ cam + cam_to_world[:3, 3]
            world.append([float(w_pt[0]), float(w_pt[1])])
    return best * step * px_m, world


def run(
    ctx: NodeContext,
    camera: CameraFrame,
    piece_outline: Any = None,
    piece_mask: Any = None,
    neighbour_outlines: Any = None,
    table_z: float | None = None,
    piece_thickness: float = PIECE_THICKNESS_M,
    raised_m: float = RAISED_M,
    tilt_rise_m: float = TILT_RISE_M,
) -> Output:
    """Is the piece flush with its neighbours (or flat on the table)?

    ``camera`` is a fresh wrist frame from above the piece (depth in metres, intrinsics, pose
    camera-to-world). The piece is ``piece_outline`` (its outline in world x, y metres, e.g.
    the slot from the placement plan) or ``piece_mask`` (a mask of the image).
    ``neighbour_outlines`` are the placed neighbours' outlines in world x, y; they become the
    reference surface and the seam test runs against them. Without them the reference is a
    ring round the piece. ``table_z`` (world z of the table top) settles whether that ring is
    the table or the neighbours' tops."""
    depth = (
        np.asarray(camera.get("depth"), dtype=np.float64)
        if camera.get("depth") is not None
        else None
    )
    k = camera.get("intrinsics")
    if depth is None or k is None:
        return {"route": "unverifiable", "detail": "the camera frame has no depth or intrinsics"}
    k = np.asarray(k, dtype=np.float64)
    cam_to_world = _pose_matrix(camera.get("pose"))
    shape = depth.shape
    valid_depth = np.isfinite(depth) & (depth > 0)
    if not valid_depth.any():
        return {"route": "unverifiable", "detail": "no valid depth"}
    z_cam = float(np.median(depth[valid_depth]))
    px_m = z_cam / (0.5 * (k[0, 0] + k[1, 1]))

    level_z = (table_z + piece_thickness) if table_z is not None else None
    if piece_outline is not None:
        poly = np.asarray(piece_outline, dtype=np.float64).reshape(-1, 2)
        z_guess = level_z if level_z is not None else float(cam_to_world[2, 3] - z_cam)
        piece = _fill(_project(poly, z_guess, k, cam_to_world), shape)
    elif piece_mask is not None:
        piece = np.asarray(piece_mask).astype(bool)
        if piece.shape != shape:
            return {
                "route": "unverifiable",
                "detail": f"piece_mask {piece.shape} is not the depth's {shape}",
            }
        z_guess = level_z if level_z is not None else float(cam_to_world[2, 3] - z_cam)
    else:
        return {"route": "unverifiable", "detail": "give piece_outline or piece_mask"}
    if not piece.any():
        return {"route": "unverifiable", "detail": "the piece is outside the image"}

    margin = _disk(EDGE_MARGIN_M / px_m)
    inner = cv2.erode(piece.astype(np.uint8), margin).astype(bool)
    keep_out = cv2.dilate(piece.astype(np.uint8), margin).astype(bool)
    neighbours = None
    if neighbour_outlines:
        neighbours = np.zeros(shape, bool)
        for poly_n in neighbour_outlines:
            pn = np.asarray(poly_n, dtype=np.float64).reshape(-1, 2)
            neighbours |= _fill(_project(pn, z_guess, k, cam_to_world), shape)
        ref_region = cv2.erode(neighbours.astype(np.uint8), margin).astype(bool) & ~keep_out
    else:
        outer = cv2.dilate(piece.astype(np.uint8), _disk(RING_M[1] / px_m)).astype(bool)
        near = cv2.dilate(piece.astype(np.uint8), _disk(RING_M[0] / px_m)).astype(bool)
        ref_region = outer & ~near

    valid_fraction = float(valid_depth[inner].mean()) if inner.any() else 0.0
    p_pts, _ = _points(depth, k, cam_to_world, inner)
    r_pts, _ = _points(depth, k, cam_to_world, ref_region)
    out: Output = {
        "valid_fraction": valid_fraction,
        "n_piece_points": len(p_pts),
        "n_reference_points": len(r_pts),
    }
    if valid_fraction < MIN_VALID_FRACTION or len(p_pts) < MIN_POINTS or len(r_pts) < MIN_POINTS:
        out.update(
            route="unverifiable",
            detail=(
                f"{valid_fraction:.0%} of the piece has depth, {len(p_pts)} piece and"
                f" {len(r_pts)} reference points"
            ),
        )
        return out

    plane, inl = segment_plane(r_pts)
    n = plane[:3] if plane[2] > 0 else -plane[:3]
    d = plane[3] if plane[2] > 0 else -plane[3]

    def height(p: np.ndarray) -> np.ndarray:
        return p @ n + d

    # What the plane is, and how high the piece's top should sit over it.
    reference, offset = "neighbours", 0.0
    if neighbours is None:
        # The ring may hold the table and neighbours' tops: fit two parallel levels a
        # thickness apart, then take the lowest well-populated one as the base.
        t = piece_thickness
        plane, _ = segment_plane(r_pts, offsets=(-t, 0.0, t))
        n, d = (plane[:3], plane[3]) if plane[2] > 0 else (-plane[:3], -plane[3])
        h_ring = height(r_pts)
        if np.mean(np.abs(h_ring + t) < RANSAC_THRESHOLD_M) >= MIN_LEVEL_FRACTION:
            d += t
            h_ring = height(r_pts)
        base = np.abs(h_ring) < RANSAC_THRESHOLD_M
        upper = np.abs(h_ring - t) < RANSAC_THRESHOLD_M
        n, d = _refit(r_pts[base], n)
        if upper.mean() >= MIN_LEVEL_FRACTION and base.mean() >= MIN_LEVEL_FRACTION:
            # Table and neighbour tops both round the piece: the tops are the reference.
            n, d = _refit(r_pts[upper], n)
        elif table_z is not None:
            plane_z = float(-(d + n[0] * p_pts[:, 0].mean() + n[1] * p_pts[:, 1].mean()) / n[2])
            if abs(plane_z - table_z) < 0.5 * t:
                reference, offset = "table", t
        elif float(np.median(height(p_pts))) > TABLE_RULE * t:
            # One level round the piece and the piece well above it: that level is the
            # table (the piece cannot be under its neighbours' tops by most of a thickness).
            reference, offset = "table", t

    r_all = height(p_pts) - offset
    # Points of the slot that show the table (a piece stopped short of its slot) are seam
    # evidence, not the piece's top.
    on_piece = r_all > -0.75 * piece_thickness
    uncovered = float(1.0 - on_piece.mean())
    if on_piece.sum() < MIN_POINTS:
        out.update(
            route="unverifiable",
            uncovered_fraction=uncovered,
            detail=f"{uncovered:.0%} of the slot shows the surface under the piece",
        )
        return out
    p_pts, r = p_pts[on_piece], r_all[on_piece]
    level = float(np.median(r))
    # Plane through the piece's top: r ~ a x + b y + c over its own points.
    xy = p_pts[:, :2] - p_pts[:, :2].mean(axis=0)
    a_mat = np.column_stack([xy, np.ones(len(xy))])
    coef, *_ = np.linalg.lstsq(a_mat, r, rcond=None)
    fitted = a_mat @ coef
    rise = float(fitted.max() - fitted.min())
    slope = float(np.hypot(coef[0], coef[1]))
    high = [float(coef[0] / slope), float(coef[1] / slope)] if slope > 0 else [0.0, 0.0]
    out.update(
        level_m=level,
        tilt_rise_m=rise,
        tilt_deg=math.degrees(math.atan(slope)),
        high_side=high,
        residual_p90_m=float(np.percentile(np.abs(r - level), 90)),
        uncovered_fraction=uncovered,
        reference=reference,
    )

    seam = 0.0
    if neighbours is not None:
        seam, gap_pts = _open_seam(
            piece, neighbours, depth, k, cam_to_world, px_m, height, piece_thickness
        )
        if gap_pts:
            out["seam_gap_points"] = gap_pts
    out["seam_gap_m"] = seam

    if rise > tilt_rise_m:
        route, why = "tilted", f"rises {rise * 1e3:.2f} mm across the piece"
    elif level > raised_m:
        route, why = "raised", f"sits {level * 1e3:.2f} mm above its neighbours' level"
    elif level < -SUNK_M:
        route, why = "gap", f"sits {-level * 1e3:.2f} mm below its neighbours' level"
        out["gap_kind"] = "low"
    elif seam >= SEAM_MIN_LENGTH_M:
        route, why = "gap", f"open seam over {seam * 1e3:.1f} mm of the outline"
        out["gap_kind"] = "seam"
    else:
        route, why = "flush", f"level {level * 1e3:+.2f} mm, rise {rise * 1e3:.2f} mm"
    out.update(route=route, detail=f"{why} (reference: {reference})")
    return out
