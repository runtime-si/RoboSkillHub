# Stud detection follows OpenCV's HoughCircles (modules/imgproc/src/hough.cpp,
# samples/python/houghcircles.py, https://github.com/opencv/opencv), Apache-2.0,
# called through cv2. The stud-lattice idea (CLAHE + Hough circles, then a
# regular grid fitted to the centres) is the one BrickScan's
# ml/preprocessing/stud_detector.py describes; BrickScan has no licence, so none
# of its code is used. Here the image is first rectified onto the plate plane
# with the depth (the circles get their metric radius and lose the
# perspective), a depth detector runs beside the RGB one, and the grid is a
# known-pitch 2D lattice fitted by RANSAC over neighbour pairs and a rigid
# Procrustes refinement, instead of 1D histogram clustering in image axes.
"""Stud-lattice geometry shared by calibrate_plate.py and read_studs.py."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

PRESETS: dict[str, dict[str, float]] = {
    "lego": {
        "stud_pitch": 0.008,
        "stud_height": 0.0017,
        "stud_diameter": 0.0048,
        "brick_height": 0.0096,
    },
    "duplo": {
        "stud_pitch": 0.016,
        "stud_height": 0.0045,
        "stud_diameter": 0.0094,
        "brick_height": 0.0192,
    },
}
"""Brick systems [m], the values computing-brick-pose-from-stud-grid and
pressing-a-brick-onto-studs use (ICL's Lego.hpp P_len_ 8 mm, knob 1.7 mm; DUPLO
twice the pitch, 9.4 mm studs 4.5 mm high; bricks 9.6 / 19.2 mm)."""

MIN_DEPTH = 0.07
MAX_DEPTH = 0.60
"""Depth trusted from a wrist D405 [m]: Intel's ~7 cm minimum Z at 640x480; 60 cm
is past its ideal 7-50 cm range but keeps a plate seen obliquely whole."""

PLANE_ITERATIONS = 300
"""RANSAC draws for a plane. The plate top is ~70 % of a LEGO close-up's pixels
(studs cover pi * 2.4^2 / 64 = 28 %), so 300 draws miss it with p < 1e-20."""

LATTICE_HYPOTHESES = 400
"""Most neighbour pairs tried as lattice hypotheses (each one a yaw and an anchor)."""

RECT_PX_PER_PITCH = 32
"""Rectified-image resolution: pixels per stud pitch (0.25 mm for LEGO). A stud's
radius is then 9.6 px, far over HoughCircles' practical minimum of ~5 px."""

RECT_MAX_SIDE = 1400
"""Largest rectified image side [px]; a wider view gets a coarser resolution."""


def rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat(r: np.ndarray) -> dict[str, float]:
    t = float(np.trace(r))
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    if q[0] < 0.0:
        q = tuple(-v for v in q)
    return dict(zip(("w", "x", "y", "z"), (float(v) for v in q), strict=True))


def xyz(v: Any) -> np.ndarray:
    if isinstance(v, dict) and "position" in v:
        v = v["position"]
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=float).reshape(3)


def vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def matrix(pose: Any) -> np.ndarray:
    if isinstance(pose, dict) and "pose" in pose and "position" not in pose:
        pose = pose["pose"]
    m = np.eye(4)
    if isinstance(pose, dict):
        m[:3, :3] = rot(pose.get("rotation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
        m[:3, 3] = xyz(pose["position"])
        return m
    arr = np.asarray(pose, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError("a pose must be an Se3Pose {position, rotation} or a 4x4 matrix")
    return arr


def se3(m: np.ndarray) -> dict[str, Any]:
    return {"position": vec(m[:3, 3]), "rotation": quat(m[:3, :3])}


def rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def wrap(angle: float, period: float) -> float:
    return (angle + period / 2.0) % period - period / 2.0


def as_depth(depth: Any) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float64)
    return d[..., 0] if d.ndim == 3 else d


def as_gray(rgb: Any) -> np.ndarray:
    img = np.asarray(rgb)
    if img.ndim == 2:
        return np.ascontiguousarray(img).astype(np.uint8)
    img = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def backproject(
    depth: np.ndarray,
    k: np.ndarray,
    t_ref_cam: np.ndarray,
    stride: int = 1,
    mask: np.ndarray | None = None,
    zmin: float = MIN_DEPTH,
    zmax: float = MAX_DEPTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Valid depth pixels as points in the reference frame, and their (row, col)."""
    d = depth[::stride, ::stride]
    rows, cols = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    ok = np.isfinite(d) & (d >= zmin) & (d <= zmax)
    if mask is not None:
        ok &= np.asarray(mask)[::stride, ::stride] > 0
    z = d[ok]
    u, v = cols[ok].astype(np.float64), rows[ok].astype(np.float64)
    pc = np.stack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z], axis=1)
    pts = pc @ t_ref_cam[:3, :3].T + t_ref_cam[:3, 3]
    return pts, np.stack([rows[ok], cols[ok]], axis=1)


def fit_plane(
    points: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
    iterations: int = PLANE_ITERATIONS,
    up: np.ndarray | None = None,
    max_tilt_deg: float = 90.0,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """RANSAC plane n . p + d = 0 (n towards ``up``), refitted to its inliers."""
    n_pts = len(points)
    if n_pts < 30:
        return None
    up = np.array([0.0, 0.0, 1.0]) if up is None else up
    cos_max = math.cos(math.radians(max_tilt_deg))
    best: tuple[int, np.ndarray, float] | None = None
    for _ in range(iterations):
        a, b, c = points[rng.choice(n_pts, 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            continue
        n /= norm
        if float(n @ up) < 0.0:
            n = -n
        if float(n @ up) < cos_max:
            continue
        d = -float(n @ a)
        count = int(np.count_nonzero(np.abs(points @ n + d) < threshold))
        if best is None or count > best[0]:
            best = (count, n, d)
    if best is None or best[0] < 30:
        return None
    n, d = best[1], best[2]
    for _ in range(2):
        inl = np.abs(points @ n + d) < threshold
        sel = points[inl]
        if len(sel) < 30:
            break
        centre = sel.mean(axis=0)
        n = np.linalg.svd(sel - centre, full_matrices=False)[2][2]
        if float(n @ up) < 0.0:
            n = -n
        d = -float(n @ centre)
    inl = np.abs(points @ n + d) < threshold
    return n, d, inl


STEP_PLANE_SHARE = 0.25
"""A plane under (over) the dominant one replaces it when it holds at least this
share of the dominant plane's points: bricks covering most of a close-up must not
pass their tops off as the plate, and a brick's top must win over the plate
round a small brick's mask."""


def stepped_plane(
    points: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
    drop: float,
    direction: int = -1,
    up: np.ndarray | None = None,
    max_tilt_deg: float = 30.0,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """The dominant up-facing plane, stepped down (direction -1) or up (+1) to
    another more than ``drop`` beyond it that holds STEP_PLANE_SHARE of its support."""
    best = fit_plane(points, threshold, rng, up=up, max_tilt_deg=max_tilt_deg)
    for _ in range(3):
        if best is None:
            return None
        n, d, inl = best
        beyond = points[direction * (points @ n + d) > drop]
        support = int(np.count_nonzero(inl))
        if len(beyond) < STEP_PLANE_SHARE * support:
            break
        other = fit_plane(beyond, threshold, rng, up=up, max_tilt_deg=max_tilt_deg)
        if other is None or np.count_nonzero(other[2]) < STEP_PLANE_SHARE * support:
            break
        best = (other[0], other[1], np.abs(points @ other[0] + other[1]) < threshold)
    return best


def plane_frame(n: np.ndarray, d: float, x_hint: np.ndarray, origin_hint: np.ndarray) -> np.ndarray:
    """A frame on the plane: z = n, origin the foot of ``origin_hint``, x along ``x_hint``."""
    x = x_hint - n * float(x_hint @ n)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(n, [0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.cross(n, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = x, np.cross(n, x), n
    m[:3, 3] = origin_hint - n * (float(n @ origin_hint) + d)
    return m


class Raster:
    """A metric grid on a frame's z = 0 plane: pixel (row, col) centre is
    (x0 + (col + 0.5) * res, y0 + (row + 0.5) * res)."""

    def __init__(self, x0: float, y0: float, x1: float, y1: float, res: float) -> None:
        span = max(x1 - x0, y1 - y0)
        self.res = max(res, span / RECT_MAX_SIDE)
        self.x0, self.y0 = x0, y0
        self.w = max(1, int(math.ceil((x1 - x0) / self.res)))
        self.h = max(1, int(math.ceil((y1 - y0) / self.res)))

    def to_xy(self, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return np.stack(
            [self.x0 + (cols + 0.5) * self.res, self.y0 + (rows + 0.5) * self.res], axis=-1
        )

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        return np.stack(
            [(xy[..., 0] - self.x0) / self.res - 0.5, (xy[..., 1] - self.y0) / self.res - 0.5],
            axis=-1,
        )

    def homography(self, k: np.ndarray, t_cam_frame: np.ndarray, z: float) -> np.ndarray:
        """Raster pixel (col, row, 1) -> image pixel, through the frame's plane z."""
        r, t = t_cam_frame[:3, :3], t_cam_frame[:3, 3]
        a = np.array(
            [
                [self.res, 0.0, self.x0 + 0.5 * self.res],
                [0.0, self.res, self.y0 + 0.5 * self.res],
                [0.0, 0.0, 1.0],
            ]
        )
        m = np.column_stack([r[:, 0], r[:, 1], r[:, 2] * z + t])
        return k @ m @ a


def height_map(points_frame: np.ndarray, raster: Raster, sigma_px: float) -> np.ndarray:
    """Points' heights (frame z) splatted at their true (x, y) and filled by
    normalised Gaussian convolution; NaN where no point lands nearby."""
    px = raster.to_px(points_frame[:, :2])
    col = np.floor(px[:, 0] + 0.5).astype(int)
    row = np.floor(px[:, 1] + 0.5).astype(int)
    ok = (col >= 0) & (col < raster.w) & (row >= 0) & (row < raster.h)
    acc = np.zeros((raster.h, raster.w), np.float64)
    cnt = np.zeros_like(acc)
    np.add.at(acc, (row[ok], col[ok]), points_frame[ok, 2])
    np.add.at(cnt, (row[ok], col[ok]), 1.0)
    acc = cv2.GaussianBlur(acc, (0, 0), sigma_px)
    cnt = cv2.GaussianBlur(cnt, (0, 0), sigma_px)
    out = np.full_like(acc, np.nan)
    good = cnt > 0.05
    out[good] = acc[good] / cnt[good]
    return out


def rectify_gray(
    gray: np.ndarray, k: np.ndarray, t_cam_frame: np.ndarray, raster: Raster, z: float
) -> np.ndarray:
    h = raster.homography(k, t_cam_frame, z)
    return cv2.warpPerspective(
        gray,
        h,
        (raster.w, raster.h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def rectify_mask(
    mask: np.ndarray, k: np.ndarray, t_cam_frame: np.ndarray, raster: Raster, z: float
) -> np.ndarray:
    h = raster.homography(k, t_cam_frame, z)
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    out = cv2.warpPerspective(
        m, h, (raster.w, raster.h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    )
    return out > 0


def studs_from_depth(
    hmap: np.ndarray, stud_h: float, r_px: float, region: np.ndarray | None = None
) -> np.ndarray:
    """Raster (col, row) centroids of blobs standing stud_h over the plane. Only
    the top disc counts: the band starts 1 mm under the stud top (half the
    height for LEGO), since the side wall the camera sees would pull the
    centroid towards it (0.4 mm on DUPLO in the renders with half the height)."""
    lo = max(0.5 * stud_h, stud_h - 0.001)
    band = np.isfinite(hmap) & (hmap > lo) & (hmap < 1.6 * stud_h)
    if region is not None:
        band &= region
    b = band.astype(np.uint8)
    ks = max(3, int(round(0.5 * r_px)) | 1)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    n, _, stats, cents = cv2.connectedComponentsWithStats(b, connectivity=8)
    area0 = math.pi * r_px * r_px
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not 0.4 * area0 <= area <= 1.7 * area0:
            continue
        if not (1.2 * r_px <= w <= 2.8 * r_px and 1.2 * r_px <= h <= 2.8 * r_px):
            continue
        if area / float(w * h) < 0.55:
            continue
        out.append(cents[i])
    return np.array(out, dtype=np.float64).reshape(-1, 2)


def studs_from_rgb(
    rect_gray: np.ndarray, r_px: float, pitch_px: float, region: np.ndarray | None = None
) -> np.ndarray:
    """Raster (col, row) centres of HoughCircles at the stud's metric radius."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(rect_gray)
    g = cv2.GaussianBlur(g, (0, 0), max(1.0, 0.12 * r_px))
    circles = cv2.HoughCircles(
        g,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=0.6 * pitch_px,
        param1=60,
        param2=max(8.0, 1.1 * r_px),
        minRadius=max(2, int(0.8 * r_px)),
        maxRadius=int(math.ceil(1.2 * r_px)),
    )
    if circles is None:
        return np.zeros((0, 2))
    c = circles[0][:, :2].astype(np.float64)
    if region is not None and len(c):
        rr = np.clip(np.round(c[:, 1]).astype(int), 0, region.shape[0] - 1)
        cc = np.clip(np.round(c[:, 0]).astype(int), 0, region.shape[1] - 1)
        c = c[region[rr, cc]]
    return c


def merge_candidates(a: np.ndarray, b: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Union of two candidate sets; of a pair closer than ``radius`` the point
    from ``a`` (depth) is kept: the RGB circle centre shifts with the shading
    (0.5 mm in the renders, light 40 deg off vertical). Returns the points and
    how many detectors saw each (1 or 2)."""
    if len(a) == 0 or len(b) == 0:
        both = a if len(b) == 0 else b
        return both.reshape(-1, 2), np.ones(len(both), int)
    tree = cKDTree(b)
    dist, j = tree.query(a, distance_upper_bound=radius)
    used = np.zeros(len(b), bool)
    pts, votes = [], []
    for i, (dd, jj) in enumerate(zip(dist, j, strict=True)):
        if np.isfinite(dd) and not used[jj]:
            used[jj] = True
            pts.append(a[i])
            votes.append(2)
        else:
            pts.append(a[i])
            votes.append(1)
    for jj in np.flatnonzero(~used):
        pts.append(b[jj])
        votes.append(1)
    return np.array(pts), np.array(votes)


TOP_BAND = 0.0006
"""Half-width of the height band round the stud top whose raw points re-centre a
candidate [m]: ~1.5 sigma of a D405 at 0.2 m. A stud's top disc is never hidden
by its own stud from above, so its points fill the disc evenly; the visible side
wall only enters in its top 0.6 mm."""

MIN_TOP_POINTS = 8
"""Fewest top-disc points to re-centre a candidate (a LEGO stud top is ~90 depth
pixels at 0.2 m)."""


def refine_centres(xy: np.ndarray, pts: np.ndarray, stud_h: float, stud_r: float) -> np.ndarray:
    """Re-centre candidates on the centroid of their top-disc depth points (frame
    x, y; z within TOP_BAND of stud_h), twice; a candidate with too few keeps its place."""
    if len(xy) == 0:
        return xy
    top = pts[np.abs(pts[:, 2] - stud_h) < TOP_BAND]
    if len(top) < MIN_TOP_POINTS:
        return xy
    tree = cKDTree(top[:, :2])
    out = xy.copy()
    for _ in range(2):
        for i, groups in enumerate(tree.query_ball_point(out, 1.05 * stud_r)):
            if len(groups) >= MIN_TOP_POINTS:
                out[i] = top[groups, :2].mean(axis=0)
    return out


def _assign(q: np.ndarray, yaw: float, t: np.ndarray, pitch: float, phase: np.ndarray):
    r = rz(yaw)
    g_cont = (q - t) @ r  # R^T (q - t), row-wise
    idx = np.round(g_cont / pitch - phase).astype(int)
    g = (idx + phase) * pitch
    res = np.linalg.norm(q - (g @ r.T + t), axis=1)
    return idx, g, res


def _procrustes(g: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
    gc, qc = g.mean(axis=0), q.mean(axis=0)
    a, b = g - gc, q - qc
    yaw = math.atan2(
        float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])),
        float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])),
    )
    return yaw, qc - rz(yaw) @ gc


def _unique_nodes(idx: np.ndarray, res: np.ndarray, inl: np.ndarray) -> np.ndarray:
    """Inliers with at most one point per lattice node (the closest)."""
    keep = np.zeros_like(inl)
    best: dict[tuple[int, int], int] = {}
    for i in np.flatnonzero(inl):
        key = (int(idx[i, 0]), int(idx[i, 1]))
        if key not in best or res[i] < res[best[key]]:
            best[key] = i
    keep[list(best.values())] = True
    return keep


def fit_lattice(
    q: np.ndarray,
    pitch: float,
    rng: np.random.Generator,
    phase: tuple[float, float] = (0.0, 0.0),
    inlier_tol: float | None = None,
    yaw_prior: float | None = None,
) -> dict[str, Any] | None:
    """Fit q ~ R(yaw) g + t, g on the lattice ((i, j) + phase) * pitch, yaw in
    [-45, 45) deg. RANSAC over neighbour pairs (a pair gives the yaw, one of its
    points the offset), then rigid Procrustes on the assigned nodes."""
    q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
    if len(q) == 0:
        return None
    ph = np.asarray(phase, dtype=np.float64)
    tol = float(inlier_tol if inlier_tol is not None else 0.2 * pitch)
    hyps: list[tuple[float, int]] = []
    if len(q) >= 2:
        tree = cKDTree(q)
        pairs = np.array(sorted(tree.query_pairs(1.25 * pitch)), dtype=int).reshape(-1, 2)
        if len(pairs):
            d = q[pairs[:, 1]] - q[pairs[:, 0]]
            ln = np.linalg.norm(d, axis=1)
            pairs = pairs[(ln > 0.75 * pitch)]
            if len(pairs) > LATTICE_HYPOTHESES:
                pairs = pairs[rng.choice(len(pairs), LATTICE_HYPOTHESES, replace=False)]
            for a, b in pairs:
                dv = q[b] - q[a]
                hyps.append((wrap(math.atan2(dv[1], dv[0]), math.pi / 2.0), int(a)))
    if yaw_prior is not None:
        anchors = rng.choice(len(q), min(len(q), 50), replace=False)
        hyps += [(wrap(float(yaw_prior), math.pi / 2.0), int(a)) for a in anchors]
    if not hyps:
        return None

    best: tuple[int, float, float, np.ndarray] | None = None
    for yaw, a in hyps:
        r = rz(yaw)
        g_a = q[a] @ r  # R^T q_a
        frac = g_a / pitch - ph
        t = r @ ((frac - np.round(frac)) * pitch)
        _, _, res = _assign(q, yaw, t, pitch, ph)
        inl = res < tol
        score = int(np.count_nonzero(inl))
        cost = float(np.sum(np.minimum(res, tol)))
        if best is None or score > best[0] or (score == best[0] and cost < best[1]):
            best = (score, cost, yaw, t)
    assert best is not None
    yaw, t = best[2], best[3]
    inl = np.zeros(len(q), bool)
    for _ in range(5):
        idx, g, res = _assign(q, yaw, t, pitch, ph)
        inl = _unique_nodes(idx, res, res < tol)
        if np.count_nonzero(inl) >= 2:
            yaw, t = _procrustes(g[inl], q[inl])
        elif np.count_nonzero(inl) == 1:
            t = t + (q[inl][0] - (g[inl][0] @ rz(yaw).T + t))
        w = wrap(yaw, math.pi / 2.0)
        if w != yaw:
            idx, g, res = _assign(q, w, t, pitch, ph)
            yaw = w
            t = np.mean(q[inl] - g[inl] @ rz(yaw).T, axis=0) if inl.any() else t
    idx, g, res = _assign(q, yaw, t, pitch, ph)
    inl = _unique_nodes(idx, res, res < tol)
    n_in = int(np.count_nonzero(inl))
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if n_in else float("inf")
    return {
        "yaw": float(yaw),
        "t": np.asarray(t, dtype=np.float64),
        "inliers": inl,
        "indices": idx,
        "residuals": res,
        "n_inliers": n_in,
        "rms": rms,
        "max_residual": float(res[inl].max()) if n_in else float("inf"),
    }


def minimal_shift(
    yaw: float, t: np.ndarray, pitch: float, at: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Among the lattice-equivalent offsets t + R(yaw) * pitch * k, the one whose
    correction at point ``at`` is smallest. Returns (t, correction at ``at``,
    the largest component of that correction in pitches)."""
    r = rz(yaw)
    f = (r - np.eye(2)) @ at + t
    f_rot = r.T @ f
    f_rot = f_rot - pitch * np.round(f_rot / pitch)
    f = r @ f_rot
    return f - (r - np.eye(2)) @ at, f, float(np.max(np.abs(f_rot)) / pitch)


def detect_and_fit(
    rgb: Any,
    depth: np.ndarray,
    k: np.ndarray,
    t_frame_cam: np.ndarray,
    raster: Raster,
    stud_h: float,
    stud_r: float,
    pitch: float,
    detector: str,
    region_cam: np.ndarray | None = None,
    region_raster: np.ndarray | None = None,
    stride: int = 1,
) -> dict[str, Any]:
    """Stud candidates in one frame, in the plane frame's (x, y) [m]."""
    t_cam_frame = np.linalg.inv(t_frame_cam)
    r_px = stud_r / raster.res
    pitch_px = pitch / raster.res
    pts, _ = backproject(depth, k, t_frame_cam, stride=stride, mask=region_cam)
    hmap = height_map(pts, raster, sigma_px=max(1.0, 0.25 * r_px))
    region = region_raster
    if region_cam is not None:
        rm = rectify_mask(region_cam, k, t_cam_frame, raster, stud_h)
        region = rm if region is None else (region & rm)
    dep = np.zeros((0, 2))
    rgbc = np.zeros((0, 2))
    if detector in ("both", "depth"):
        dep = studs_from_depth(hmap, stud_h, r_px, region)
    if detector in ("both", "rgb") and rgb is not None:
        rect = rectify_gray(as_gray(rgb), k, t_cam_frame, raster, stud_h)
        rgbc = studs_from_rgb(rect, r_px, pitch_px, region)
        if len(rgbc) and detector == "both":
            rr = np.clip(np.round(rgbc[:, 1]).astype(int), 0, raster.h - 1)
            cc = np.clip(np.round(rgbc[:, 0]).astype(int), 0, raster.w - 1)
            hv = hmap[rr, cc]
            # A circle where depth says the surface is not a plate stud top
            # (a brick's top, the plate between studs) is dropped; no depth keeps it.
            keep = ~np.isfinite(hv) | ((hv > 0.4 * stud_h) & (hv < 1.6 * stud_h))
            rgbc = rgbc[keep]
    cands, votes = merge_candidates(dep, rgbc, 0.35 * pitch_px)
    xy = raster.to_xy(cands[:, 0], cands[:, 1]) if len(cands) else np.zeros((0, 2))
    xy = refine_centres(xy, pts, stud_h, stud_r)
    return {
        "xy": xy,
        "votes": votes,
        "n_depth": len(dep),
        "n_rgb": len(rgbc),
        "hmap": hmap,
        "n_points": len(pts),
    }
