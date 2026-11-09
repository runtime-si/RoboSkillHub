"""Synthetic top-down RGB-D frames of loose jigsaw pieces on a staging mat, with ground truth.

Pieces are cut by ``jigsaw_cut.make_puzzle`` (classic or wavy random cut) and printed with a
procedural picture (``_jigsaw_pictures.make_reference``) or shown face down (plain cardboard,
mirrored outline). The frame has a lighting gradient, mat texture, drop shadows, dark cut
edges, blur and noise. Scenes mix isolated pieces, touching pairs (moved together until their
outlines meet), flush pairs (two mating pieces still interlocked) and stacked pairs (one piece
partly on another, 2 mm higher in depth). World frame: table at z = 0, camera straight down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    from ._jigsaw_pictures import make_reference
    from .jigsaw_cut import make_puzzle
except ImportError:  # run as a script by _jigsaw_staging_eval.py
    from _jigsaw_pictures import make_reference
    from jigsaw_cut import make_puzzle

MATS = {
    "green": (40, 120, 70),
    "magenta": (225, 30, 120),
    "blue": (40, 70, 150),
    "grey": (120, 120, 120),
    "black": (22, 22, 24),
    "white": (225, 225, 220),
}
BACK = (170, 160, 145)
"""Plain cardboard back."""

THICK = 0.002
SS = 2  # supersampling


@dataclass
class Scene:
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    pose: dict
    masks: list = field(default_factory=list)  # visible ground-truth mask per piece
    kinds: list = field(
        default_factory=list
    )  # single / touch / flush / stacked_top / stacked_bottom
    faces: list = field(default_factory=list)  # up / down
    groups: list = field(default_factory=list)  # index of the pair, or -1
    angles: list = field(
        default_factory=list
    )  # side angle in the image, CCW as displayed, deg mod 90
    cells: list = field(
        default_factory=list
    )  # (col, row) of the piece in the picture, row 0 at the top
    picture: np.ndarray | None = None
    grid: int = 8
    px_per_piece: float = 0.0
    cell_m: float = 0.0225


def _camera(width, height, px_per_piece, cell_m, f=600.0):
    zc = f * cell_m / px_per_piece
    K = np.array([[f, 0, width / 2 - 0.5], [0, f, height / 2 - 0.5], [0, 0, 1]], float)
    pose = {
        "position": {"x": 0.0, "y": 0.0, "z": zc},
        "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
    }
    return K, pose, zc


class _Piece:
    def __init__(self, puzzle, i, j, picture, face_down):
        self.cell_ij = (i, j)
        self.outline = puzzle.outline(i, j)
        self.centre = puzzle.centre(i, j)
        self.picture = picture
        self.cell = puzzle.cell[0]
        self.board_h = puzzle.rows * puzzle.cell[1]
        self.ppm_pic = picture.shape[1] / (puzzle.cols * puzzle.cell[0])
        self.face_down = face_down

    def local(self):
        p = self.outline - self.centre
        if self.face_down:
            p = p * [-1, 1]
        return p


def _to_px(local, x, y, th, K, zc, z):
    c, s = math.cos(th), math.sin(th)
    w = local @ np.array([[c, s], [-s, c]]) + [x, y]
    depth = zc - z
    u = w[:, 0] * K[0, 0] / depth + K[0, 2]
    v = -w[:, 1] * K[1, 1] / depth + K[1, 2]
    return np.column_stack([u, v])


def _affine_pic_to_px(piece, x, y, th, K, zc, z):
    """Picture pixel -> image pixel (all affine for a top-down camera)."""
    d = zc - z
    # picture px -> board metres (y north, picture row 0 at the north edge)
    a1 = np.array([[1 / piece.ppm_pic, 0, 0], [0, -1 / piece.ppm_pic, piece.board_h], [0, 0, 1]])
    a2 = np.array([[1, 0, -piece.centre[0]], [0, 1, -piece.centre[1]], [0, 0, 1]])
    c, s = math.cos(th), math.sin(th)
    a3 = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
    a4 = np.array([[K[0, 0] / d, 0, K[0, 2]], [0, -K[1, 1] / d, K[1, 2]], [0, 0, 1]])
    return (a4 @ a3 @ a2 @ a1)[:2]


def _raster(px, shape):
    big = np.zeros((shape[0] * SS, shape[1] * SS), np.uint8)
    cv2.fillPoly(
        big,
        [np.round((px + 0.5) * SS * 16 - 0.5 * 16).astype(np.int32)],
        255,
        lineType=cv2.LINE_8,
        shift=4,
    )
    return (
        cv2.resize(big, (shape[1], shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    )


def make_scene(
    seed: int = 0,
    width: int = 640,
    height: int = 480,
    px_per_piece: float = 48.0,
    n_single: int = 6,
    n_touch: int = 2,
    n_flush: int = 0,
    n_stacked: int = 0,
    mat: str = "green",
    shadows: bool = True,
    face_down: float = 0.15,
    style: str | None = None,
    cell_m: float = 0.0225,
    depth_noise: float = 0.0003,
) -> Scene:
    rng = np.random.default_rng(seed)
    style = style or ("classic" if rng.random() < 0.5 else "wavy")
    puzzle = make_puzzle(8, 8, cell=(cell_m, cell_m), seed=seed, style=style)
    picture = make_reference(36 * 8, 36 * 8, seed=seed, band_rows=(0.0, 0.1), plaid=True)
    K, pose, zc = _camera(width, height, px_per_piece, cell_m)
    shape = (height, width)
    cells = [(i, j) for i in range(8) for j in range(8)]
    rng.shuffle(cells)
    cells = iter(cells)
    occupied = np.zeros(shape, bool)
    placed: list[dict] = []  # each: piece, x, y, th, z, alpha, kind, group
    half_w = (width / 2 - 0.9 * px_per_piece) * (zc / K[0, 0])
    half_h = (height / 2 - 0.9 * px_per_piece) * (zc / K[1, 1])

    def new_piece(fd=None):
        i, j = next(cells)
        fd = bool(rng.random() < face_down) if fd is None else fd
        return _Piece(puzzle, i, j, picture, fd)

    def alpha_of(pc, x, y, th, z=THICK):
        return _raster(_to_px(pc.local(), x, y, th, K, zc, z), shape)

    def free(a, gap_px):
        m = a > 0.5
        if gap_px > 0:
            m = cv2.dilate(m.astype(np.uint8), np.ones((2 * gap_px + 1,) * 2, np.uint8)) > 0
        return not np.any(m & occupied)

    def add(pc, x, y, th, a, kind, group, z=THICK):
        placed.append(dict(piece=pc, x=x, y=y, th=th, z=z, alpha=a, kind=kind, group=group))
        occupied[:] |= a > 0.5

    gap = max(3, int(0.3 * px_per_piece))
    group = 0
    for _ in range(n_single):
        pc = new_piece()
        for _ in range(200):
            x, y, th = (
                rng.uniform(-half_w, half_w),
                rng.uniform(-half_h, half_h),
                rng.uniform(0, 2 * np.pi),
            )
            a = alpha_of(pc, x, y, th)
            if free(a, gap):
                add(pc, x, y, th, a, "single", -1)
                break
    for _ in range(n_touch):
        pa, pb = new_piece(), new_piece()
        for _ in range(200):
            x, y, th = (
                rng.uniform(-half_w, half_w),
                rng.uniform(-half_h, half_h),
                rng.uniform(0, 2 * np.pi),
            )
            aa = alpha_of(pa, x, y, th)
            if not free(aa, gap):
                continue
            phi, thb = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
            ma = cv2.dilate((aa > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            ok, ab, xb, yb = False, None, x, y
            step = 0.25 * zc / K[0, 0]
            for k in range(int(2.0 * cell_m / step), 0, -1):
                xb, yb = x + k * step * math.cos(phi), y + k * step * math.sin(phi)
                ab = alpha_of(pb, xb, yb, thb)
                if np.any((ab > 0.5) & ma):
                    ok = not np.any((ab > 0.5) & (aa > 0.5))
                    break
            if ok and _inside(ab) and free(ab, gap):
                add(pa, x, y, th, aa, "touch", group)
                add(pb, xb, yb, thb, ab, "touch", group)
                group += 1
                break
    for _ in range(n_flush):
        i = int(rng.integers(0, 7))
        j = int(rng.integers(0, 8))
        pa, pb = _Piece(puzzle, i, j, picture, False), _Piece(puzzle, i + 1, j, picture, False)
        for _ in range(200):
            x, y, th = (
                rng.uniform(-half_w, half_w),
                rng.uniform(-half_h, half_h),
                rng.uniform(0, 2 * np.pi),
            )
            off = pb.centre - pa.centre
            c, s = math.cos(th), math.sin(th)
            xb, yb = x + c * off[0] - s * off[1], y + s * off[0] + c * off[1]
            aa, ab = alpha_of(pa, x, y, th), alpha_of(pb, xb, yb, th)
            if free(np.maximum(aa, ab), gap) and _inside(aa) and _inside(ab):
                ab = np.minimum(ab, 1.0 - aa)  # the shared cut belongs to one piece
                add(pa, x, y, th, aa, "flush", group)
                add(pb, xb, yb, th, ab, "flush", group)
                group += 1
                break
    for _ in range(n_stacked):
        pa, pb = new_piece(False), new_piece()
        for _ in range(200):
            x, y, th = (
                rng.uniform(-half_w, half_w),
                rng.uniform(-half_h, half_h),
                rng.uniform(0, 2 * np.pi),
            )
            phi, thb = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
            d = rng.uniform(0.45, 0.65) * cell_m
            xb, yb = x + d * math.cos(phi), y + d * math.sin(phi)
            aa, ab = alpha_of(pa, x, y, th), alpha_of(pb, xb, yb, thb, 2 * THICK)
            if free(np.maximum(aa, ab), gap) and _inside(ab) and _inside(aa):
                add(pa, x, y, th, aa, "stacked_bottom", group)
                add(pb, xb, yb, thb, ab, "stacked_top", group, z=2 * THICK)
                group += 1
                break
    scene = _render(
        placed, shape, K, pose, zc, mat, shadows, rng, px_per_piece, cell_m, depth_noise
    )
    scene.picture = picture
    return scene


def _inside(a, margin=4):
    m = a > 0.5
    return not (
        m[:margin].any() or m[-margin:].any() or m[:, :margin].any() or m[:, -margin:].any()
    )


def _render(placed, shape, K, pose, zc, mat, shadows, rng, px_per_piece, cell_m, depth_noise):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = rng.uniform(-1, 1, 2)
    light = 0.92 + 0.17 * (gx * (xx / w - 0.5) + gy * (yy / h - 0.5))
    base = np.array(MATS[mat], np.float32)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.0)
    img = base[None, None, :] * (1 + 0.04 * tex[:, :, None])
    if shadows:
        sh = np.zeros(shape, np.float32)
        off = 0.06 * px_per_piece
        sx, sy = off * rng.uniform(0.5, 1.0), off * rng.uniform(0.5, 1.0)
        for p in placed:
            sh = np.maximum(
                sh, cv2.warpAffine(p["alpha"], np.float32([[1, 0, sx], [0, 1, sy]]), (w, h))
            )
        sh = cv2.GaussianBlur(sh, (0, 0), 0.04 * px_per_piece + 0.5)
        img *= (1 - 0.45 * sh)[:, :, None]
    depth = np.full(shape, zc, np.float32)
    order = sorted(range(len(placed)), key=lambda k: placed[k]["z"])
    vis = [None] * len(placed)
    for k in order:
        p = placed[k]
        pc, a = p["piece"], p["alpha"]
        if pc.face_down:
            face = np.empty((h, w, 3), np.float32)
            face[:] = np.array(BACK, np.float32)
            face += rng.normal(0, 2.0, face.shape).astype(np.float32)
        else:
            A = _affine_pic_to_px(pc, p["x"], p["y"], p["th"], K, zc, p["z"])
            face = cv2.warpAffine(
                pc.picture, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            ).astype(np.float32)
        # dark cut edge: a thin rim inside the outline
        m = (a > 0.5).astype(np.uint8)
        rim = (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(np.float32)
        face *= (1 - 0.35 * rim)[:, :, None]
        img = img * (1 - a[:, :, None]) + face * a[:, :, None]
        depth[a > 0.5] = zc - p["z"]
        vis[k] = a > 0.5
        for q in range(len(placed)):
            if q != k and vis[q] is not None and placed[q]["z"] <= p["z"]:
                vis[q] = vis[q] & ~(a > 0.5)
    img *= light[:, :, None]
    img = cv2.GaussianBlur(img, (0, 0), 0.7) + rng.normal(0, 3.0, img.shape)
    depth = depth + rng.normal(0, depth_noise, shape).astype(np.float32)
    s = Scene(
        np.clip(img, 0, 255).astype(np.uint8),
        depth,
        K,
        pose,
        px_per_piece=px_per_piece,
        cell_m=cell_m,
    )
    for k, p in enumerate(placed):
        s.masks.append(vis[k])
        s.kinds.append(p["kind"])
        s.angles.append(math.degrees(p["th"]) % 90.0)
        i, j = p["piece"].cell_ij
        s.cells.append((i, 7 - j))
        s.faces.append("down" if p["piece"].face_down else "up")
        s.groups.append(p["group"])
    return s


def camera_frame(scene: Scene, name: str = "wrist") -> dict:
    return {
        "name": name,
        "rgb": scene.rgb,
        "depth": scene.depth,
        "intrinsics": scene.K,
        "pose": scene.pose,
    }


def match(pred_masks, gt_masks):
    """Hungarian match on IoU; returns [(pred, gt, iou)] for pairs with IoU > 0."""
    from scipy.optimize import linear_sum_assignment

    if not pred_masks or not gt_masks:
        return []
    iou = np.zeros((len(pred_masks), len(gt_masks)))
    for a, pm in enumerate(pred_masks):
        for b, gm in enumerate(gt_masks):
            inter = np.logical_and(pm, gm).sum()
            if inter:
                iou[a, b] = inter / np.logical_or(pm, gm).sum()
    r, c = linear_sum_assignment(-iou)
    return [(int(a), int(b), float(iou[a, b])) for a, b in zip(r, c, strict=True) if iou[a, b] > 0]
