"""Register a view of the assembled jigsaw area to the box picture and judge every placed cell.

Reimplemented for the hub (no code copied). The alignment is OpenCV's ECC
(``cv2.findTransformECC`` with an input mask, G. D. Evangelidis and E. Z. Psarakis,
"Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization", IEEE TPAMI
2008; OpenCV modules/video/src/ecc.cpp, Apache-2.0), run coarse-to-fine on a pyramid after an
initial homography from the board's corners (or SIFT + RANSAC when no corners are given). The
mask keeps only the cells believed placed, shrunk away from the seams. Each placed cell is then
compared with its own cell of the picture by zero-mean normalised cross-correlation searched
over a small shift, and over the turns the cell shape allows; cells that do not match are
looked up in the whole picture.
"""

from __future__ import annotations

import logging
import math
import zlib
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext
from gap_core.types import CameraFrame

logger = logging.getLogger(__name__)

WORK_CELL_PX = 48.0
"""Width of one grid cell in the reference frame the work runs in [px]; the picture is shrunk
to this, never enlarged. PuzzleMap 10 x 10 pair, 40 trials (tests/bundles/_register_board_eval.py):
32 px gave the same cell verdicts but aligned to 0.045 cells (median) instead of 0.035;
64 px was no better and took twice as long."""

PYRAMID = (0.25, 0.5, 1.0)
LEVEL_MOTIONS = (cv2.MOTION_TRANSLATION, cv2.MOTION_AFFINE)
"""ECC runs at these fractions of the work resolution in turn, each level starting from the
last one's warp (Evangelidis' coarse-to-fine scheme), fitting a translation at the coarsest
(12 px cells), an affine map at the middle and the requested model at full resolution. With
the full homography at every level, a quarter-placed board diverged at the coarse level in
most trials (8 free parameters on a strip of texture)."""

ECC_ITERATIONS = 80
ECC_EPS = 1e-5
ECC_GAUSS = 5
"""ECC stopping rule and its pre-smoothing kernel (OpenCV's defaults are 50 / 1e-3 / 5; more
iterations and a tighter epsilon cost little at these sizes)."""

SEAM_INSET = 0.12
"""Fraction of a cell cut off every side of a cell before it enters the ECC mask or the
cell comparison. Seams, tabs and the dark cut line live there, and a tab of the neighbour
reaches about 0.2-0.25 cell into a cell (the jigsaw generators in the hub's tests); 0.12
keeps the knob tips in but leaves most of the cell's own picture."""

SHIFT_SEARCH = 0.15
"""A cell's content is searched this far (fraction of the cell) either way round its place.
A piece pushed in by hand or sitting a millimetre off still matches; its offset is
reported. Larger and a neighbouring cell's texture starts to win on repetitive art."""

MATCH_NCC = 0.6
"""A textured cell whose best NCC against its own reference cell is at least this is
``match``. PuzzleMap 10 x 10 pair (a real photo of the print against the digital picture,
40 trials): right cells scored 0.87-0.97 (median 0.97), swapped cells up to 0.66 against the
picture of the cell they sat in (look-alikes; most under 0.4). At 0.45 seven swapped cells
in 20 trials passed as matches."""

ROTATED_GAIN = 0.15
"""A non-matching cell is ``rotated`` when one of its allowed turns reaches MATCH_NCC and beats
the unturned score by this much."""

LOW_TEXTURE_STD = 8.0
"""Reference cells whose interior grey standard deviation (0-255) is under this carry too
little pattern for NCC (white snow, black cloth); they are judged by colour instead."""

LOW_TEXTURE_COLOUR = 28.0
"""Largest Lab distance (after the global colour fit) for a low-texture cell to be
``low_texture_match``. Two different flat whites or blacks of one print differ by about
10-20; the same white under the photo's gloss by under 15 after the fit."""

EMPTY_FRACTION = 0.6
"""A cell is ``empty`` when this fraction of its interior is within EMPTY_COLOUR of the mat's
colour (and the reference cell itself is not that colour)."""

EMPTY_COLOUR = 22.0
"""Lab distance from the mat colour that counts as mat."""

MIN_ECC = 0.7
"""Smallest final ECC correlation coefficient for ``registered``. PuzzleMap pair, after the outlier
passes: right alignments 0.75-0.99 (median 0.98); SIFT locking a quarter-placed board onto the wrong place (4 cells off)
still reached 0.61-0.67, which is why MIN_MATCHED_FRACTION is also required."""

MIN_MATCHED_FRACTION = 0.5
"""Fraction of the textured placed cells that must match their own place (NCC >= OUTLIER_NCC
without a shift search) after alignment. A right alignment of a board with a few wrong
pieces keeps nearly all of them; a wrong one keeps a few by chance."""

OUTLIER_NCC = 0.3
OUTLIER_PASSES = 2
"""After each ECC pass the textured cells scoring under OUTLIER_NCC against their own place
are dropped from the mask and ECC runs again, at most OUTLIER_PASSES more times."""

MAX_CORRECTION = 1.0
"""Largest move of any board corner by the ECC correction from the given corners or
homography, in cells; past it the start is treated as wrong and SIFT features are tried
(``use_features``). PuzzleMap pair: from corners off by up to 0.6 cells ECC moved them back to
within 0.04 cells; it does not reliably pull in more than about half a cell's error."""

MIN_PLACED_CELLS = 4
MIN_TEXTURED_CELLS = 3
"""ECC needs texture in the mask: at least this many placed cells, this many of them
textured, before alignment is tried (``too_little_placed``)."""

AUTO_HOMOGRAPHY_FRACTION = 0.2
"""With ``motion="auto"`` the full-resolution correction is a homography when at least this
fraction of the board's cells are placed and textured, else affine (the initial homography
from the corners already holds the perspective; 8 parameters on a narrow strip of texture
tilt the far side of the board). PuzzleMap pair, 12 % placed: see SKILL.md."""

MAX_LOOKUPS = 40
"""Mismatched cells looked up in the whole picture (``belongs_at``); each costs a few ms."""

_REF_CACHE: dict[tuple, dict[str, Any]] = {}


class Output(TypedDict, total=False):
    route: str
    homography: list[list[float]]
    ecc_correlation: float
    correction_cells: float
    init: str
    motion: str
    matched_fraction: float
    content_shift_px: list[float]
    content_shift_m: list[float]
    cells: list[dict[str, Any]]
    n_placed: int
    n_match: int
    n_mismatch: int
    n_rotated: int
    n_empty: int
    mismatched_cells: list[list[int]]
    rotated_cells: list[list[int]]
    empty_cells: list[list[int]]
    detail: str


def _rgb(image: Any) -> np.ndarray:
    if isinstance(image, str):
        bgr = cv2.imread(image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read image {image!r}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[:, :, :3])


def _prepare_reference(reference: Any, cols: int, rows: int) -> dict[str, Any]:
    if isinstance(reference, str):
        key: tuple = (reference, cols, rows)
    else:
        arr = np.ascontiguousarray(reference)
        key = (arr.shape, zlib.crc32(arr.tobytes()), cols, rows)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    native = _rgb(reference)
    h, w = native.shape[:2]
    scale = min(1.0, WORK_CELL_PX * cols / w)
    size = (round(w * scale), round(h * scale))
    work = cv2.resize(native, size, interpolation=cv2.INTER_AREA) if scale < 1 else native.copy()
    grey = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB).astype(np.float32)
    cw, ch = size[0] / cols, size[1] / rows
    std = np.zeros((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            ys, xs = _interior(c, r, cw, ch, SEAM_INSET)
            std[r, c] = grey[ys, xs].std()
    prepared = {
        "native": native,
        "native_size": (w, h),
        "scale": scale,
        "work": work,
        "grey": grey,
        "lab": lab,
        "size": size,
        "cell": (cw, ch),
        "std": std,
    }
    if len(_REF_CACHE) > 4:
        _REF_CACHE.clear()
    _REF_CACHE[key] = prepared
    return prepared


def _interior(c: int, r: int, cw: float, ch: float, inset: float) -> tuple[slice, slice]:
    return (
        slice(round((r + inset) * ch), round((r + 1 - inset) * ch)),
        slice(round((c + inset) * cw), round((c + 1 - inset) * cw)),
    )


def _corners_homography(corners_px: Any, size: tuple[int, int]) -> np.ndarray:
    src = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    w, h = size
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(
        np.float64
    )


def _feature_homography(image: np.ndarray, ref: dict[str, Any]) -> np.ndarray | None:
    """SIFT + ratio test + RANSAC from the image to the work reference (image -> reference)."""
    sift = cv2.SIFT_create(nfeatures=4000)
    g_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    k1, d1 = sift.detectAndCompute(g_img, None)
    k2, d2 = sift.detectAndCompute(ref["grey"], None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if h is None or int(inliers.sum()) < 10:
        return None
    return h.astype(np.float64)


def _mat_colour(image: np.ndarray, h_img2ref: np.ndarray, size: tuple[int, int]) -> Any:
    """Median Lab colour of the image outside the board's outline, if there is enough of it."""
    w, h = size
    corners = np.float64([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    poly = cv2.perspectiveTransform(corners, np.linalg.inv(h_img2ref)).reshape(-1, 2)
    inside = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(inside, np.round(poly).astype(np.int32), 1)
    outside = inside == 0
    if outside.mean() < 0.05:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).reshape(-1, 3)[outside.reshape(-1)]
    return np.median(lab, axis=0).astype(np.float32)


def _ecc(ref_grey: np.ndarray, warped: np.ndarray, mask: np.ndarray, motion: int):
    """Coarse-to-fine ECC of ``warped`` (already in the reference frame) onto the reference:
    returns the 3x3 correction (reference -> warped coordinates) and the final correlation.
    The levels fit LEVEL_MOTIONS in turn, ending with ``motion``; a level whose iterations
    diverge keeps the previous level's warp. Raises cv2.error when every level failed."""
    warp = np.eye(3, dtype=np.float64)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERATIONS, ECC_EPS)
    rho, ok, last_error = 0.0, False, None
    for s, level_motion in zip(PYRAMID, (*LEVEL_MOTIONS, motion), strict=False):
        if level_motion == cv2.MOTION_HOMOGRAPHY and motion != cv2.MOTION_HOMOGRAPHY:
            level_motion = motion
        k = np.diag([s, s, 1.0])
        scaled = k @ warp @ np.linalg.inv(k)
        size = (max(8, round(ref_grey.shape[1] * s)), max(8, round(ref_grey.shape[0] * s)))
        t = cv2.resize(ref_grey, size, interpolation=cv2.INTER_AREA)
        i = cv2.resize(warped, size, interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        if level_motion == cv2.MOTION_HOMOGRAPHY:
            start = (scaled / scaled[2, 2]).astype(np.float32)
        else:
            start = scaled[:2].astype(np.float32)
        try:
            rho, found = cv2.findTransformECC(t, i, start, level_motion, criteria, m, ECC_GAUSS)
        except cv2.error as exc:
            last_error = exc
            continue
        found = found.astype(np.float64)
        full = found if found.shape == (3, 3) else np.vstack([found, [0.0, 0.0, 1.0]])
        warp = np.linalg.inv(k) @ full @ k
        ok = True
    if not ok:
        raise last_error
    return warp, float(rho)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def _best_in_window(patch: np.ndarray, ref_grey: np.ndarray, ys: slice, xs: slice, pad: int):
    """Best NCC of ``patch`` placed within ``pad`` px of (ys, xs) in the reference, and its shift."""
    y0, x0 = max(0, ys.start - pad), max(0, xs.start - pad)
    y1, x1 = min(ref_grey.shape[0], ys.stop + pad), min(ref_grey.shape[1], xs.stop + pad)
    win = ref_grey[y0:y1, x0:x1]
    if win.shape[0] < patch.shape[0] or win.shape[1] < patch.shape[1] or patch.std() < 1e-3:
        return _ncc(patch, ref_grey[ys, xs]), (0.0, 0.0)
    res = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
    _, best, _, loc = cv2.minMaxLoc(res)
    # The shift is how far the content sits from where it belongs (content - expected).
    return float(best), (float(xs.start - (x0 + loc[0])), float(ys.start - (y0 + loc[1])))


def _turns(cw: float, ch: float) -> list[int]:
    return [90, 180, 270] if abs(cw / ch - 1.0) < 0.15 else [180]


def _lab_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel gain and offset taking ``src`` Lab values to ``dst`` (least squares)."""
    gain, off = np.ones(3, np.float32), np.zeros(3, np.float32)
    for k in range(3):
        a = np.vstack([src[:, k], np.ones(len(src))]).T
        sol, *_ = np.linalg.lstsq(a, dst[:, k], rcond=None)
        gain[k], off[k] = sol
    return gain, off


def run(
    ctx: NodeContext,
    reference: Any,
    grid_cols: int,
    grid_rows: int,
    camera: CameraFrame | None = None,
    image: Any = None,
    board_corners_px: Any = None,
    initial_homography: Any = None,
    placed_cells: Any = None,
    board_width: float = 0.0,
    board_height: float = 0.0,
    mat_rgb: Any = None,
    motion: str = "auto",
    match_ncc: float = MATCH_NCC,
    min_ecc: float = MIN_ECC,
    look_up: bool = True,
    use_features: bool = True,
) -> Output:
    """Align a view of the assembled area to the box picture and judge the placed cells.

    ``camera`` (or ``image``, an RGB array or path) shows the assembly from above (an overhead
    frame, a rectified view or a wrist mosaic). ``board_corners_px`` are the board's outer
    corners in that image, top-left, top-right, bottom-right, bottom-left as the picture
    stands; or ``initial_homography`` (image -> reference pixels); with neither, SIFT
    features are matched. ``placed_cells`` ([[col, row], ...], (0, 0) top-left) are the cells
    believed filled; by default every cell. ``mat_rgb`` (or the image outside the board)
    lets empty cells be told from wrong ones. Board size in metres converts shifts to metres.
    """
    cols, rows = int(grid_cols), int(grid_rows)
    if cols < 1 or rows < 1:
        return {"route": "did_not_converge", "detail": "grid_cols and grid_rows must be positive"}
    try:
        ref = _prepare_reference(reference, cols, rows)
        src = image if image is not None else (camera or {}).get("rgb")
        if src is None:
            return {"route": "did_not_converge", "detail": "no camera or image given"}
        img = _rgb(src)
    except ValueError as exc:
        return {"route": "did_not_converge", "detail": str(exc)}
    cw, ch = ref["cell"]
    size = ref["size"]
    s_native = ref["scale"]
    if placed_cells is None:
        placed = [(c, r) for r in range(rows) for c in range(cols)]
    else:
        placed = sorted(
            {(int(c), int(r)) for c, r in placed_cells if 0 <= c < cols and 0 <= r < rows}
        )
    textured = [(c, r) for c, r in placed if ref["std"][r, c] >= LOW_TEXTURE_STD]
    if len(placed) < MIN_PLACED_CELLS or len(textured) < MIN_TEXTURED_CELLS:
        return {
            "route": "too_little_placed",
            "n_placed": len(placed),
            "detail": f"{len(placed)} placed cells, {len(textured)} textured: too little to align",
        }

    h_given = None
    if initial_homography is not None:
        h_given = np.asarray(initial_homography, dtype=np.float64).reshape(3, 3)
        h_given = np.diag([s_native, s_native, 1.0]) @ h_given
    elif board_corners_px is not None:
        h_given = _corners_homography(board_corners_px, size)

    mask = np.zeros(size[::-1], np.uint8)
    for c, r in textured:
        ys, xs = _interior(c, r, cw, ch, SEAM_INSET)
        mask[ys, xs] = 255
    if motion == "auto":
        motion = (
            "homography" if len(textured) >= AUTO_HOMOGRAPHY_FRACTION * cols * rows else "affine"
        )
    mode = cv2.MOTION_AFFINE if motion == "affine" else cv2.MOTION_HOMOGRAPHY
    w, h = size
    box = np.float64([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    def correction_of(h_final: np.ndarray, h_start: np.ndarray) -> float:
        rel = h_start @ np.linalg.inv(h_final)
        moved = cv2.perspectiveTransform(box, rel).reshape(-1, 2) - box.reshape(-1, 2)
        return float(np.max(np.linalg.norm(moved / [cw, ch], axis=1)))

    def align(h_start: np.ndarray):
        """ECC from ``h_start``, then again without the cells that do not match their place
        (a wrong, turned or missing piece pulls the fit toward itself)."""
        use = mask.copy()
        h_cur, rho = h_start, 0.0
        for _ in range(1 + OUTLIER_PASSES):
            warped = cv2.cvtColor(
                cv2.warpPerspective(img, h_cur, size, flags=cv2.INTER_LINEAR), cv2.COLOR_RGB2GRAY
            )
            valid = cv2.warpPerspective(
                np.full(img.shape[:2], 255, np.uint8), h_cur, size, flags=cv2.INTER_NEAREST
            )
            try:
                corr, rho = _ecc(ref["grey"], warped, use & valid, mode)
            except cv2.error as exc:
                return None, 0.0, 0.0, f"ECC failed ({str(exc).strip().splitlines()[-1][:80]})"
            # corr maps reference -> pre-warped coordinates: image -> reference is inv(corr) @ h_cur.
            h_cur = np.linalg.inv(corr) @ h_cur
            grey = cv2.cvtColor(
                cv2.warpPerspective(img, h_cur, size, flags=cv2.INTER_LINEAR), cv2.COLOR_RGB2GRAY
            )
            good, drop = 0, []
            for c, r in textured:
                ys, xs = _interior(c, r, cw, ch, SEAM_INSET)
                if _ncc(grey[ys, xs], ref["grey"][ys, xs]) >= OUTLIER_NCC:
                    good += 1
                elif use[ys, xs].any():
                    drop.append((ys, xs))
            frac = good / len(textured)
            if not drop or len(textured) - len(drop) < MIN_TEXTURED_CELLS:
                break
            for ys, xs in drop:
                use[ys, xs] = 0
        return h_cur, float(rho), frac, ""

    def accepted(rho_try: float, frac: float) -> bool:
        return rho_try >= min_ecc and frac >= MIN_MATCHED_FRACTION

    attempts: list[str] = []
    h_final, rho, init, matched = None, 0.0, "", 0.0
    if h_given is not None:
        h_try, rho_try, frac, why = align(h_given)
        init = "corners" if initial_homography is None else "homography"
        if (
            h_try is not None
            and accepted(rho_try, frac)
            and correction_of(h_try, h_given) <= MAX_CORRECTION
        ):
            h_final, rho, matched = h_try, rho_try, frac
        else:
            attempts.append(
                why
                or f"from the {init}: ECC {rho_try:.2f}, {frac:.0%} of textured cells match,"
                f" correction {correction_of(h_try, h_given):.2f} cells"
            )
    if h_final is None and use_features:
        h_feat = _feature_homography(img, ref)
        if h_feat is None:
            attempts.append("too few feature matches")
        else:
            h_try, rho_try, frac, why = align(h_feat)
            if h_try is not None and accepted(rho_try, frac):
                h_final, rho, init, matched = h_try, rho_try, "features", frac
            else:
                attempts.append(
                    why or f"from features: ECC {rho_try:.2f}, {frac:.0%} of textured cells match"
                )
    if h_final is None:
        return {
            "route": "did_not_converge",
            "n_placed": len(placed),
            "detail": "; ".join(attempts) or "no board corners, homography or features",
        }

    # How the content sits against the given corners: a slide of the assembly, or wrong corners.
    h_ref = h_given if h_given is not None else h_final
    rel = h_ref @ np.linalg.inv(h_final)
    correction = correction_of(h_final, h_ref)
    centre = np.float64([[[w / 2, h / 2]]])
    shift_px = (cv2.perspectiveTransform(centre, rel) - centre).reshape(2)
    out: Output = {
        "homography": (np.diag([1 / s_native, 1 / s_native, 1.0]) @ h_final).tolist(),
        "ecc_correlation": rho,
        "correction_cells": correction,
        "content_shift_px": [float(v / s_native) for v in shift_px],
        "n_placed": len(placed),
        "init": init,
        "motion": motion,
        "matched_fraction": matched,
    }
    if board_width and board_height:
        out["content_shift_m"] = [
            float(shift_px[0] / size[0] * board_width),
            float(shift_px[1] / size[1] * board_height),
        ]
    final_rgb = cv2.warpPerspective(img, h_final, size, flags=cv2.INTER_LINEAR)
    final_valid = cv2.warpPerspective(
        np.full(img.shape[:2], 255, np.uint8), h_final, size, flags=cv2.INTER_NEAREST
    )
    final_grey = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2GRAY)
    final_lab = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    mat = None
    if mat_rgb is not None:
        m = np.uint8(np.asarray(mat_rgb, dtype=np.float64).reshape(1, 1, 3))
        mat = cv2.cvtColor(m, cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
    else:
        mat = _mat_colour(img, h_final, size)

    pad = round(SHIFT_SEARCH * max(cw, ch))
    turns = _turns(cw, ch)
    cells: list[dict[str, Any]] = []
    lab_src, lab_dst = [], []
    for c, r in placed:
        ys, xs = _interior(c, r, cw, ch, SEAM_INSET)
        cover = float((final_valid[ys, xs] > 0).mean())
        entry: dict[str, Any] = {"col": c, "row": r, "texture_std": float(ref["std"][r, c])}
        if cover < 0.9:
            entry.update(status="unseen", coverage=cover)
            cells.append(entry)
            continue
        patch = final_grey[ys, xs]
        if entry["texture_std"] < LOW_TEXTURE_STD:
            entry["status"] = "low_texture"
            cells.append(entry)
            continue
        ncc, shift = _best_in_window(patch, ref["grey"], ys, xs, pad)
        entry.update(ncc=ncc, shift_px=[float(v / s_native) for v in shift])
        if board_width and board_height:
            entry["shift_m"] = [
                float(shift[0] / size[0] * board_width),
                float(shift[1] / size[1] * board_height),
            ]
        best_turn, best_turn_ncc = 0, ncc
        for t in turns:
            turned = np.ascontiguousarray(np.rot90(patch, t // 90))
            if turned.shape != patch.shape:
                turned = cv2.resize(turned, patch.shape[::-1])
            tn, _ = _best_in_window(turned, ref["grey"], ys, xs, pad)
            if tn > best_turn_ncc:
                best_turn, best_turn_ncc = t, tn
        if best_turn and best_turn_ncc >= match_ncc and best_turn_ncc - ncc >= ROTATED_GAIN:
            # np.rot90 turns the content counter-clockwise as displayed; the piece is turned
            # the other way from where it belongs.
            entry.update(status="rotated", turned_deg=-best_turn % 360, turned_ncc=best_turn_ncc)
        elif ncc >= match_ncc:
            entry["status"] = "match"
            lab_src.append(final_lab[ys, xs].reshape(-1, 3)[::7])
            lab_dst.append(ref["lab"][ys, xs].reshape(-1, 3)[::7])
        else:
            entry["status"] = "mismatch"
        cells.append(entry)

    # Low-texture cells: colour after a global Lab fit learnt on the matching cells.
    if lab_src:
        gain, off = _lab_fit(np.vstack(lab_src), np.vstack(lab_dst))
    else:
        gain, off = np.ones(3, np.float32), np.zeros(3, np.float32)
    for entry in cells:
        if entry.get("status") not in ("low_texture", "mismatch"):
            continue
        ys, xs = _interior(entry["col"], entry["row"], cw, ch, SEAM_INSET)
        seen_lab = final_lab[ys, xs].reshape(-1, 3)
        seen = np.median(seen_lab, axis=0) * gain + off
        want = np.median(ref["lab"][ys, xs].reshape(-1, 3), axis=0)
        dist = float(np.linalg.norm(seen - want))
        entry["colour_distance"] = dist
        if mat is not None:
            # The mat is seen raw (the fit is for the print), so it is compared unfitted.
            to_mat = np.linalg.norm(seen_lab - mat, axis=1)
            frac = float((to_mat < EMPTY_COLOUR).mean())
            entry["mat_fraction"] = frac
            if frac >= EMPTY_FRACTION and float(np.median(to_mat)) < dist:
                entry["status"] = "empty"
                continue
        if entry["status"] == "low_texture":
            entry["status"] = "low_texture_match" if dist <= LOW_TEXTURE_COLOUR else "mismatch"

    if look_up:
        lookups = 0
        for entry in cells:
            if entry.get("status") != "mismatch" or entry["texture_std"] < LOW_TEXTURE_STD:
                continue
            if lookups >= MAX_LOOKUPS:
                break
            lookups += 1
            ys, xs = _interior(entry["col"], entry["row"], cw, ch, SEAM_INSET)
            best = None
            for t in [0, *turns]:
                p = np.ascontiguousarray(np.rot90(final_grey[ys, xs], t // 90))
                res = cv2.matchTemplate(ref["grey"], p, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(res)
                if best is None or score > best[0]:
                    best = (score, loc, t, p.shape)
            score, loc, t, shape = best
            cx = (loc[0] + shape[1] / 2) / cw
            cy = (loc[1] + shape[0] / 2) / ch
            if score >= match_ncc:
                entry["belongs_at"] = {
                    "col": int(min(cols - 1, max(0, cx))),
                    "row": int(min(rows - 1, max(0, cy))),
                    "turned_deg": -t % 360,
                    "ncc": float(score),
                }

    def _list(status: str) -> list[list[int]]:
        return [[e["col"], e["row"]] for e in cells if e.get("status") == status]

    mismatched, rotated, empty = _list("mismatch"), _list("rotated"), _list("empty")
    n_match = sum(e.get("status") in ("match", "low_texture_match") for e in cells)
    out.update(
        cells=cells,
        n_match=n_match,
        n_mismatch=len(mismatched),
        n_rotated=len(rotated),
        n_empty=len(empty),
        mismatched_cells=mismatched,
        rotated_cells=rotated,
        empty_cells=empty,
        route="registered",
        detail=(
            f"ECC {rho:.2f} from the {init}, correction {correction:.2f} cells; {n_match}/{len(placed)} cells match,"
            f" {len(mismatched)} mismatch, {len(rotated)} rotated, {len(empty)} empty"
        ),
    )
    return out
