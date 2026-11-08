"""Estimate a jigsaw's cut grid (cols x rows) from its printed piece count, the board's
shape and the corner-to-corner side lengths of a few measured pieces.

Every grid (c, r) whose c * r is within MAX_COUNT_DEVIATION of the printed count is a
candidate. Each predicts a cell (board_width / c, board_height / r); the measured pieces'
side lengths (one loose piece does not say which of its side pairs is the width, so both
ways are mixed) are scored against it in log space with an outlier term, under one common
scale factor the fit may move a little (depth, mask edge bias), plus a prior on how far
manufacturers round the printed count and, when given, a full count of edge pieces
(2 * (c + r) - 4). The candidates' posterior probabilities give the answer, its confidence
and the runner-up. Written for this hub; no code was copied (corner detection is
matching-a-piece-to-placed-neighbours', loaded from beside this skill).
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

MATCH_SKILL = "matching-a-piece-to-placed-neighbours"
MATCH_SCRIPT = "scripts/match_piece.py"

MAX_COUNT_DEVIATION = 0.10
"""Candidates are the grids whose c * r is within this fraction of the printed count.
Printed counts are rounded: the 550-piece target is cut 20 x 27 = 540 (1.8 % short,
measured from the seams printed on its official flat image); 1000-piece boxes are often
38 x 27 = 1026 (+2.6 %)."""

COUNT_SIGMA = 0.03
"""Width of the Gaussian prior on (c * r - count) / count. With 0.03 a grid 1.8 % off the
count (the target's) loses a factor 1.2 to one that hits it exactly, so the pieces decide."""

SIDE_SIGMA_RANGE = (0.03, 0.15)
"""Per-piece relative error of a side-pair mean, estimated from the pieces themselves
(see side_sigma), clipped to this range. Synthetic cuts, 64 px cells, mask noise:
0.035-0.06 (corners jittered 4-7 % of a cell plus 1.5-1.8 % detection); the traced real
cut 0.058. A floor of 0.045 cost 12 points of top-1 without orientation, 0.02 gained
under 1; 0.03 keeps a few tight pieces from making the fit overconfident."""

OUTLIER_FRACTION = 0.08
"""Share of pieces assumed mis-measured (wrong corner split, two pieces in one mask). About
1 in 11 noisy wavy pieces gets a wrong corner set in matching-a-piece-to-placed-neighbours'
own measurement; such a piece's sides are spread over the outlier density below."""

OUTLIER_SPAN = 0.7
"""The outlier density is uniform over this span of log side length per pair (x0.70 to
x1.42 of the cell), so it is 1 / OUTLIER_SPAN ** 2 per piece."""

SCALE_SIGMA = 0.08
"""Prior on the common scale factor between measured and predicted cells (log), when the
board size is given: depth off by a few mm at 15 cm and a mask edge 1-2 px inside the cut
shrink every measured side by 1-5 % (synthetic noisy masks: -2.8 % after CHORD_BIAS_PX;
the traced real cut: -4.5 %). 0.05 pulled the traced cut toward finer grids (25 of 40
right at 48 oriented pieces against 26 at 0.08); on the synthetic sets 0.03-0.15 changed
top-1 by under 1.5 points."""

SCALE_RANGE = 0.25
"""Common scale factors searched, +- this in log (x0.78 to x1.28). Without a board size
(aspect only) the board is sized so that count cells match the pieces' median cell, and
the scale is free within this range, so only the cells' shape and the count decide."""

CELL_ASPECT_SIGMA = 0.15
"""Weak prior on log(cell width / cell height): manufacturers cut near-square cells
(the target's 20 x 27 cells are 22.9 x 22.6 mm, 1.3 % off square). It decides only when
the pieces cannot (none, or too few)."""

MAX_CELL_ASPECT = 1.6
"""Grids whose predicted cell is longer than this times its width are not scored; the
cell-aspect prior already puts them over 3 sigma out (log 1.6 = 0.47 = 3.1 x 0.15)."""

EDGE_COUNT_SIGMA = (1.0, 0.02)
"""A full count of edge pieces (one or two straight sides) is trusted to max(1 piece,
2 % of the expected count): sorting 90 edge pieces by eye misses or adds one or two."""

EDGE_DISAGREE_Z = 3.0
"""The edge count disagrees with the chosen grid beyond this many sigmas: route ambiguous."""

CHORD_BIAS_PX = 1.0
"""Added to every corner-to-corner side measured on a mask [px]. The corner search
(outline smoothed 3 px, corners sharpened on the shoulders) puts corners slightly inside
the cut: on clean synthetic masks sides came out 0.8-1.3 px short at 64 and 100 px per
cell, classic and wavy cuts alike. Blur, threshold and erosion add another 1-1.5 px that
depends on the segmenter; the common scale factor absorbs that."""

MIN_PIECES = 6
"""Fewer usable measured pieces than this: insufficient_pieces (the prior-only guess is
still returned). See SKILL.md for the accuracy by number of pieces."""

MIN_CONFIDENCE = 0.9
"""Posterior probability the best grid needs for `estimated`. On the synthetic sets in
SKILL.md, `estimated` answers at 0.9 were right in 98.6-99.9 % of cases (6-48 pieces)."""

PIECE_QUERY = "jigsaw puzzle piece"
SEGMENT_SCORE = 0.5
"""sam3.segment_text masks scoring under this are ignored (as in other hub skills)."""

AREA_OUTLIER = 1.8
"""A view's mask whose area is over this times, or under one over this times, the view's
median is dropped (two pieces in one mask, a fragment); interior jigsaw pieces vary about
+-25 % in area (the traced real cut: CV 0.21)."""


class GridResult(TypedDict, total=False):
    route: str
    grid_cols: int
    grid_rows: int
    cell_width: float
    cell_height: float
    confidence: float
    runner_up_cols: int
    runner_up_rows: int
    runner_up_confidence: float
    grid_count: int
    expected_edge_pieces: int
    pieces_used: int
    scale_ratio: float
    side_sigma: float
    residual: float
    edge_check: str
    candidates: list[dict[str, Any]]
    pieces: list[dict[str, Any]]
    detail: str


_MATCH: Any = None


def match_module() -> Any:
    """matching-a-piece-to-placed-neighbours' script (its corner search), from the skill
    installed beside this one or under a GAP_SKILLS_PATH root."""
    global _MATCH
    if _MATCH is not None:
        return _MATCH
    roots = [Path(__file__).resolve().parents[2]]
    roots += [Path(p) for p in os.environ.get("GAP_SKILLS_PATH", "").split(os.pathsep) if p]
    for root in roots:
        for path in (
            root / MATCH_SKILL / MATCH_SCRIPT,
            root / "skills" / MATCH_SKILL / MATCH_SCRIPT,
        ):
            if path.is_file():
                spec = importlib.util.spec_from_file_location("_estimate_grid_match", path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _MATCH = module
                return module
    raise RuntimeError(
        f"{MATCH_SKILL} is not installed beside this skill (looked under "
        f"{', '.join(str(r) for r in roots)})"
    )


# --- measuring pieces ---------------------------------------------------------------------


def _width_pair(corners_px: np.ndarray, angle_deg: float) -> int:
    """Which side pair (0: sides 0 and 2, 1: sides 1 and 3) runs along the board's x once
    the piece is turned upright; ``angle_deg`` is how far the piece in the image is turned
    from upright, counter-clockwise as displayed (localizing-a-piece-in-a-reference-image's
    angle_deg)."""
    d = corners_px[1] - corners_px[0]
    x, y = float(d[0]), float(-d[1])
    a = math.radians(-angle_deg)
    ux = x * math.cos(a) - y * math.sin(a)
    uy = x * math.sin(a) + y * math.cos(a)
    return 0 if abs(ux) >= abs(uy) else 1


def measure_mask(
    mask: np.ndarray,
    metres_per_pixel: float,
    angle_deg: float | None = None,
    cell_size: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Corner-to-corner side lengths [m], side types and class of one top-down piece mask."""
    info = match_module().analyse_piece(mask, metres_per_pixel, cell_size=cell_size)
    corners = np.asarray(info["corners_px"], dtype=np.float64)
    sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1) + CHORD_BIAS_PX
    sides = sides * metres_per_pixel
    out = {
        "side_lengths": [float(s) for s in sides],
        "side_types": info["side_types"],
        "piece_class": info["piece_class"],
        "corners_px": info["corners_px"],
        "metres_per_pixel": float(metres_per_pixel),
    }
    if angle_deg is not None:
        out["width_pair"] = _width_pair(corners, float(angle_deg))
    return out


def _intrinsics(cam: dict[str, Any]) -> np.ndarray | None:
    k = cam.get("intrinsics")
    if k is None:
        return None
    k = np.asarray(k, dtype=np.float64)
    if k.shape == (3, 3):
        return k
    if k.size == 4:  # fx, fy, cx, cy
        fx, fy, cx, cy = k.ravel()
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return None


def _mask_mpp(mask: np.ndarray, cam: dict[str, Any]) -> float | None:
    """Metres per pixel at the piece: median valid depth under the mask over the focal
    length (camera square to the table)."""
    depth, k = cam.get("depth"), _intrinsics(cam)
    if depth is None or k is None:
        return None
    d = np.asarray(depth, dtype=np.float64)
    if d.shape != mask.shape:
        return None
    vals = d[mask & np.isfinite(d) & (d > 0)]
    if vals.size < 20:
        return None
    if np.median(vals) > 20:  # millimetres
        vals = vals / 1000.0
    return float(np.median(vals)) / float(0.5 * (k[0, 0] + k[1, 1]))


def _as_bool(mask) -> np.ndarray:
    m = np.asarray(mask)
    return m > (127 if m.dtype == np.uint8 else 0)


def _touches_border(mask: np.ndarray, pad: int = 2) -> bool:
    return bool(
        mask[:pad].any() or mask[-pad:].any() or mask[:, :pad].any() or mask[:, -pad:].any()
    )


def _view_masks(ctx: NodeContext, camera_name: str | None, query: str):
    obs = ctx.tool("robot.get_observation")
    cams = obs.get("cameras") if isinstance(obs, dict) else getattr(obs, "cameras", None)
    if isinstance(cams, dict):
        cams = [dict(v, name=v.get("name", k)) for k, v in cams.items()]
    cams = list(cams or [])
    if camera_name:
        cam = next((c for c in cams if str(c.get("name")) == camera_name), None)
    else:
        cam = next(
            (c for c in cams if any(k in str(c.get("name", "")) for k in ("wrist", "eye_in_hand"))),
            None,
        )
    if cam is None:
        raise ValueError(f"no camera {camera_name or 'wrist'!r} in the observation")
    seg = ctx.tool("sam3.segment_text", image=np.asarray(cam["rgb"]), query=query, max_results=0)
    masks = seg.get("masks") or []
    scores = [float(s) for s in (seg.get("scores") or [1.0] * len(masks))]
    return cam, [_as_bool(m) for m, s in zip(masks, scores, strict=False) if s >= SEGMENT_SCORE]


def _collect(
    ctx: NodeContext,
    pieces: list[dict] | None,
    camera: dict | None,
    piece_masks: list | None,
    measure_view: bool,
    camera_name: str | None,
    metres_per_pixel: float | None,
    cell_hint: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    """Every measured piece as {id, side_lengths, width_pair?, piece_class?} or {id, error}."""
    raw: list[dict[str, Any]] = []
    for k, p in enumerate(pieces or []):
        raw.append(dict(p, id=str(p.get("id", f"p{k}"))))
    masks: list[tuple[str, np.ndarray, dict | None]] = []
    if piece_masks:
        for k, m in enumerate(piece_masks):
            masks.append((f"m{k}", _as_bool(m), camera))
    if measure_view:
        cam, view = _view_masks(ctx, camera_name, PIECE_QUERY)
        areas = [int(m.sum()) for m in view]
        med = float(np.median(areas)) if areas else 0.0
        for k, m in enumerate(view):
            if _touches_border(m) or not (med / AREA_OUTLIER <= m.sum() <= med * AREA_OUTLIER):
                raw.append({"id": f"v{k}", "error": "mask clipped by the image or not one piece"})
                continue
            masks.append((f"v{k}", m, cam))
    for pid, m, cam in masks:
        mpp = _mask_mpp(m, cam) if cam is not None else None
        raw.append({"id": pid, "mask": m, "metres_per_pixel": mpp or metres_per_pixel})

    out = []
    for p in raw:
        if "error" in p:
            out.append(p)
            continue
        if p.get("side_lengths") is not None:
            s = [float(x) for x in p["side_lengths"]]
            if len(s) != 4 or min(s) <= 0:
                out.append({"id": p["id"], "error": "side_lengths must be 4 positive lengths"})
                continue
            rec = {"id": p["id"], "side_lengths": s}
            for key in ("width_pair", "piece_class", "side_types", "corners_px"):
                if p.get(key) is not None:
                    rec[key] = p[key]
            if "width_pair" not in rec and p.get("angle_deg") is not None and p.get("corners_px"):
                corners = np.asarray(p["corners_px"], dtype=np.float64)
                rec["width_pair"] = _width_pair(corners, float(p["angle_deg"]))
            out.append(rec)
            continue
        if p.get("mask") is None:
            out.append({"id": p["id"], "error": "no side_lengths and no mask"})
            continue
        mpp = p.get("metres_per_pixel")
        if not mpp:
            out.append({"id": p["id"], "error": "no scale (metres_per_pixel or depth)"})
            continue
        try:
            rec = measure_mask(p["mask"], float(mpp), p.get("angle_deg"), cell_hint)
        except ValueError as exc:
            out.append({"id": p["id"], "error": str(exc)})
            continue
        out.append({"id": p["id"], **rec})
    return out


# --- scoring grids ------------------------------------------------------------------------


def candidate_grids(count: int, max_dev: float = MAX_COUNT_DEVIATION) -> list[tuple[int, int]]:
    lo, hi = count * (1 - max_dev), count * (1 + max_dev)
    out = []
    for c in range(2, int(hi) + 1):
        for r in range(max(2, math.ceil(lo / c)), int(hi // c) + 1):
            out.append((c, r))
    return out


def _pairs(rec: dict[str, Any]) -> tuple[float, float, int | None]:
    s = rec["side_lengths"]
    return 0.5 * (s[0] + s[2]), 0.5 * (s[1] + s[3]), rec.get("width_pair")


def side_sigma(sides: np.ndarray) -> float:
    """Relative per-piece error of a side-pair mean, from how much each piece's opposite
    sides differ (they share no corner, so log(s0 / s2) has twice a side's variance and a
    pair mean half of it). Needs no candidate grid and no orientation."""
    if len(sides) < 3:
        return SIDE_SIGMA_RANGE[1]
    ls = np.log(sides)
    d = np.concatenate([ls[:, 0] - ls[:, 2], ls[:, 1] - ls[:, 3]])
    mad = 1.4826 * float(np.median(np.abs(d - np.median(d))))
    return float(np.clip(0.5 * mad, *SIDE_SIGMA_RANGE))


def score_grids(
    count: int,
    shape: tuple[float, float],
    absolute: bool,
    a: np.ndarray,
    b: np.ndarray,
    oriented: np.ndarray,
    sigma: float,
    edge_count: int | None = None,
) -> list[dict[str, Any]]:
    """Log posterior of every candidate grid. ``shape`` is (board_width, board_height) in
    metres when ``absolute``, else any pair with the board's aspect. ``a``, ``b`` are the
    side-pair means of each piece; ``oriented[i]`` is 0 / 1 when pair a / b is known to be
    the width, -1 when unknown."""
    W, H = shape
    grids = [
        (c, r)
        for c, r in candidate_grids(count)
        if abs(math.log((W / c) / (H / r))) <= math.log(MAX_CELL_ASPECT)
    ]
    span = SCALE_RANGE
    ks = np.linspace(-span, span, 601)
    la, lb = np.log(a), np.log(b)
    log_out = math.log(OUTLIER_FRACTION) - 2 * math.log(OUTLIER_SPAN)
    norm = -math.log(2 * math.pi * sigma**2)
    out = []
    for c, r in grids:
        cw, ch = W / c, H / r
        lcw, lch = math.log(cw), math.log(ch)
        lp = -0.5 * ((c * r - count) / (count * COUNT_SIGMA)) ** 2
        lp += -0.5 * (math.log(cw / ch) / CELL_ASPECT_SIGMA) ** 2
        if edge_count is not None:
            exp_e = 2 * (c + r) - 4
            s_e = max(EDGE_COUNT_SIGMA[0], EDGE_COUNT_SIGMA[1] * exp_e)
            lp += -0.5 * ((edge_count - exp_e) / s_e) ** 2 - math.log(s_e)
        if len(a):
            mw = lcw + ks[:, None]
            mh = lch + ks[:, None]
            e1 = ((la - mw) ** 2 + (lb - mh) ** 2) / (2 * sigma**2)
            e2 = ((la - mh) ** 2 + (lb - mw) ** 2) / (2 * sigma**2)
            l1, l2 = norm - e1, norm - e2
            mixed = np.logaddexp(l1, l2) - math.log(2)
            good = np.where(oriented == 0, l1, np.where(oriented == 1, l2, mixed))
            per = np.logaddexp(math.log(1 - OUTLIER_FRACTION) + good, log_out)
            total = per.sum(axis=1)
            if absolute:
                total = total - 0.5 * (ks / SCALE_SIGMA) ** 2
            best = int(np.argmax(total))
            lp += float(total[best])
            k = float(ks[best])
            res_w = np.where(
                oriented == 1,
                lb,
                np.where(oriented == 0, la, np.where(e1[best] <= e2[best], la, lb)),
            )
            res_h = np.where(
                oriented == 1,
                la,
                np.where(oriented == 0, lb, np.where(e1[best] <= e2[best], lb, la)),
            )
            r_all = np.concatenate([res_w - (lcw + k), res_h - (lch + k)])
            residual = float(np.sqrt(np.median(r_all**2)))
        else:
            k, residual = 0.0, float("nan")
        out.append(
            {
                "cols": c,
                "rows": r,
                "count": c * r,
                "cell_width": cw * (1.0 if absolute else math.exp(k)),
                "cell_height": ch * (1.0 if absolute else math.exp(k)),
                "scale_ratio": math.exp(k),
                "residual": residual,
                "expected_edge_pieces": 2 * (c + r) - 4,
                "log_posterior": lp,
            }
        )
    if not out:
        return out
    top = max(o["log_posterior"] for o in out)
    z = sum(math.exp(o["log_posterior"] - top) for o in out)
    for o in out:
        o["probability"] = math.exp(o["log_posterior"] - top) / z
    out.sort(key=lambda o: -o["probability"])
    return out


def _board_shape(
    board_width: float | None,
    board_height: float | None,
    board_aspect: float | None,
    reference: Any,
) -> tuple[tuple[float, float], bool, str]:
    ref_aspect = None
    if reference is not None:
        if isinstance(reference, str | os.PathLike):
            import cv2

            img = cv2.imread(str(reference))
            if img is None:
                raise ValueError(f"cannot read reference {reference!r}")
        else:
            img = np.asarray(reference)
        ref_aspect = img.shape[1] / img.shape[0]
    note = ""
    if board_width and board_height:
        if ref_aspect and abs(math.log(ref_aspect / (board_width / board_height))) > 0.03:
            note = (
                f"; the reference's aspect {ref_aspect:.3f} differs from the board's "
                f"{board_width / board_height:.3f} (the board size is used)"
            )
        return (float(board_width), float(board_height)), True, note
    aspect = board_aspect or ref_aspect
    if not aspect:
        raise ValueError("give board_width and board_height, board_aspect or the reference picture")
    return (math.sqrt(aspect), 1.0 / math.sqrt(aspect)), False, note


def run(
    ctx: NodeContext,
    piece_count: int,
    board_width: float | None = None,
    board_height: float | None = None,
    board_aspect: float | None = None,
    reference: Any = None,
    pieces: list[dict] | None = None,
    camera: dict | None = None,
    piece_masks: list | None = None,
    measure_view: bool = False,
    camera_name: str | None = None,
    metres_per_pixel: float | None = None,
    edge_piece_count: int | None = None,
    min_pieces: int = MIN_PIECES,
    min_confidence: float = MIN_CONFIDENCE,
    top_k: int = 5,
) -> GridResult:
    """Estimate the cut grid of a ``piece_count``-piece jigsaw.

    The board's shape comes from ``board_width`` x ``board_height`` (metres, from the box),
    else ``board_aspect`` (width / height) or the ``reference`` box picture cropped to the
    board (path or array). Pieces come from any of: ``pieces`` (``[{"id", "side_lengths":
    [4 corner-to-corner lengths, m, in order round the piece], "width_pair"?: 0|1}]`` or
    ``[{"id", "mask", "metres_per_pixel"?, "angle_deg"?}]``), ``piece_masks`` on
    ``camera`` (a CameraFrame whose depth gives each mask's scale), or, with
    ``measure_view``, every whole piece sam3 finds in a fresh wrist frame.
    ``edge_piece_count`` is a full count of the puzzle's pieces with one or two straight
    sides, if one has been made."""
    count = int(piece_count)
    if count < 4:
        return {"route": "insufficient_pieces", "detail": f"piece_count {count} is not a puzzle"}
    try:
        shape, absolute, note = _board_shape(board_width, board_height, board_aspect, reference)
    except ValueError as exc:
        return {"route": "insufficient_pieces", "detail": str(exc), "candidates": [], "pieces": []}

    hint = None
    if absolute:
        side = math.sqrt(shape[0] * shape[1] / count)
        hint = (side, side)
    try:
        measured = _collect(
            ctx, pieces, camera, piece_masks, measure_view, camera_name, metres_per_pixel, hint
        )
    except (RuntimeError, ValueError) as exc:
        return {"route": "insufficient_pieces", "detail": str(exc), "candidates": [], "pieces": []}
    good = [p for p in measured if "side_lengths" in p]
    if good:
        a = np.array([_pairs(p)[0] for p in good])
        b = np.array([_pairs(p)[1] for p in good])
        oriented = np.array([-1 if _pairs(p)[2] is None else int(_pairs(p)[2]) for p in good])
    else:
        a = b = np.zeros(0)
        oriented = np.zeros(0, dtype=int)
    sigma = side_sigma(np.array([p["side_lengths"] for p in good]).reshape(-1, 4))
    if not absolute and len(a):
        # Aspect only: size the board so its cells match the pieces; only their shape counts.
        area = count * float(np.median(a * b))
        aspect = shape[0] / shape[1]
        shape = (math.sqrt(area * aspect), math.sqrt(area / aspect))
    edge = int(edge_piece_count) if edge_piece_count is not None else None
    ranked = score_grids(count, shape, absolute, a, b, oriented, sigma, edge)
    if not ranked:
        return {"route": "insufficient_pieces", "detail": "no candidate grid", "pieces": measured}
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    public = [{k: v for k, v in o.items() if k != "log_posterior"} for o in ranked[: max(1, top_k)]]
    out: GridResult = {
        "grid_cols": best["cols"],
        "grid_rows": best["rows"],
        "cell_width": best["cell_width"],
        "cell_height": best["cell_height"],
        "confidence": best["probability"],
        "grid_count": best["count"],
        "expected_edge_pieces": best["expected_edge_pieces"],
        "pieces_used": len(good),
        "scale_ratio": best["scale_ratio"],
        "side_sigma": sigma,
        "residual": best["residual"],
        "candidates": public,
        "pieces": [{k: v for k, v in p.items() if k != "mask"} for p in measured],
    }
    if second is not None:
        out.update(
            runner_up_cols=second["cols"],
            runner_up_rows=second["rows"],
            runner_up_confidence=second["probability"],
        )
    edge_check = "not_given"
    if edge is not None:
        exp_e = best["expected_edge_pieces"]
        z = (edge - exp_e) / max(EDGE_COUNT_SIGMA[0], EDGE_COUNT_SIGMA[1] * exp_e)
        edge_check = "agrees" if abs(z) <= EDGE_DISAGREE_Z else "disagrees"
    out["edge_check"] = edge_check
    what = (
        f"{best['cols']} x {best['rows']} = {best['count']} (p {best['probability']:.2f}"
        + (
            f"; runner-up {second['cols']} x {second['rows']} p {second['probability']:.2f}"
            if second
            else ""
        )
        + f") from {len(good)} piece(s)"
    )
    if len(good) < min_pieces:
        out.update(
            route="insufficient_pieces",
            detail=f"{len(good)} usable piece(s), need {min_pieces}; prior-only guess {what}{note}",
        )
    elif edge_check == "disagrees":
        out.update(
            route="ambiguous",
            detail=f"edge count {edge} disagrees with {best['expected_edge_pieces']} for {what}{note}",
        )
    elif best["probability"] < min_confidence:
        out.update(route="ambiguous", detail=f"best grid not certain: {what}{note}")
    else:
        out.update(route="estimated", detail=f"{what}{note}")
    return out
