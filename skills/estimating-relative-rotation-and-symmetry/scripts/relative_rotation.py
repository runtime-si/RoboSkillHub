# Relative in-plane rotation of a part between a reference view and the current view, with the
# rotation's period (its apparent symmetry), from the two masks and the pixels inside them.
#
# Candidate: Orient Anything V2 (SpatialVision/Orient-Anything-V2, CC-BY-4.0, commit 73b11c9),
# which predicts a relative rotation and a symmetry order from two crops. It was measured on the
# same pairs as this script (SKILL.md) and rejected: median error 4.1 deg on masked crops (40 deg
# on raw ones) against 0.34 deg here, 2-fold parts called 4-fold 45 times in 49, and 1 of 6 domino
# half-turns resolved. No code of it is used; this is own code: principal-axis angles of the masks
# give the rotation up to the part's outline symmetry, and comparing the rotated reference pixels
# with the current ones picks among the candidate turns (the pips of a domino, studs, print).
"""Relative in-plane rotation and its symmetry period of one part between two views (masks + pixels)."""

from __future__ import annotations

import math
from typing import Any, TypedDict

import cv2
import numpy as np

#: Long / short principal-axis ratio above which the outline counts as 2-fold (a rectangle);
#: below it 4-fold (a square or round part). 1 x 1 and 2 x 2 bricks measure 1.0-1.2 in the
#: station renders, 2 x 3 and longer 1.6 and more.
ELONGATION_2FOLD = 1.3
#: Candidate turns whose appearance score is within this of the best are indistinguishable.
#: On the 12 face-up domino pairs of SKILL.md the right half-turn won by 0.29-0.93 in 10; the
#: other 2 differed by 0.005 and 0.036 (pips at the frame's resolution limit) and would have
#: picked the wrong turn.
AMBIGUOUS_MARGIN = 0.05
#: Below this normalised correlation of the best candidate the views do not match (other part,
#: occluded, very different lighting).
MIN_SCORE = 0.3
#: Fewer mask pixels than this in either view: too small to measure an axis.
MIN_AREA_PX = 60
#: The principal-axis angle is refined over +-REFINE_DEG in steps of REFINE_STEP_DEG.
REFINE_DEG, REFINE_STEP_DEG = 4.0, 0.5


class Result(TypedDict):
    route: str  # rotation_found | symmetric_ambiguous | low_confidence
    rotation_deg: (
        float  # current = reference turned by this, in (-180, 180], image plane, clockwise +
    )
    period_deg: float  # the rotation is known up to multiples of this (360 = fully resolved)
    symmetry_order: int  # 360 / period_deg
    outline_order: int  # 2 (elongated outline) or 4 (square / round)
    scores: list[float]  # appearance score of each candidate turn, best first
    elongation: float
    reason: str


def _mask(m: Any, shape: tuple[int, int]) -> np.ndarray:
    a = np.asarray(m)
    if a.shape != shape:
        raise ValueError(f"mask shape {a.shape} differs from its image's {shape}")
    return a > 0


def _image(im: Any) -> np.ndarray:
    """RGB [H, W, 3] from a CameraFrame (its ``rgb``) or an array."""
    a = np.asarray(im["rgb"] if isinstance(im, dict) else im)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected an RGB [H, W, 3] image, got {a.shape}")
    return a


def axis(mask: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Principal-axis angle (deg, image plane, mod 180), elongation (sqrt of the eigenvalue ratio), centroid."""
    ys, xs = np.nonzero(mask)
    c = np.array([xs.mean(), ys.mean()])
    cov = np.cov(np.stack([xs - c[0], ys - c[1]]))
    w, v = np.linalg.eigh(cov)
    a = v[:, 1]
    return math.degrees(math.atan2(a[1], a[0])), float(math.sqrt(w[1] / max(w[0], 1e-9))), c


def _patch(
    img: np.ndarray, mask: np.ndarray, c: np.ndarray, angle: float, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """A size x size gray patch and its mask, centred on c, turned by ``angle`` degrees counter-clockwise as seen (cv2)."""
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), float(angle), 1.0)
    M[:, 2] += np.array([size / 2 - c[0], size / 2 - c[1]])
    p = cv2.warpAffine(gray, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)
    m = (
        cv2.warpAffine(
            mask.astype(np.uint8), M, (size, size), flags=cv2.INTER_NEAREST, borderValue=0
        )
        > 0
    )
    return p, m


def _ncc(a: np.ndarray, ma: np.ndarray, b: np.ndarray, mb: np.ndarray) -> float:
    m = ma & mb
    if m.sum() < 20:
        return -1.0
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 1e-6 else 0.0


def _wrap(deg: float) -> float:
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def outline_angle(mask: np.ndarray, order: int) -> float:
    """The outline's angle (deg, image plane): principal axis mod 180 for elongated parts; for square
    ones (order 4) the minimum-area rectangle's side mod 90, where the principal axis is unstable."""
    if order == 2:
        return axis(mask)[0]
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (_, _), (_, _), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    return float(ang)


def _fold(deg: float, period: float) -> float:
    """``deg`` moved by multiples of ``period`` into [-period / 2, period / 2)."""
    return ((deg + period / 2) % period) - period / 2


def run(
    ctx: Any,
    reference_frame: Any,
    reference_mask: Any,
    frame: Any,
    mask: Any,
    symmetry_order: int = 0,
    outline_order: int = 0,
) -> Result:
    """How far the part in ``mask`` (of ``frame``) is turned in the image plane from the part in
    ``reference_mask`` (of ``reference_frame``); frames are CameraFrames or RGB arrays.

    Both views should come from one camera at about the same distance and tilt (the wrist at a fixed
    height over the table). ``outline_order``: 2 (elongated) or 4 (square / round) outline; measured
    from the masks when 0. ``symmetry_order``: how many turns of the part look the same (1 for a
    face-up domino or a printed piece, 2 for a 2 x 4 brick, 4 for a 2 x 2 brick); defaults to the
    outline's. Below the outline's, the pixels inside the masks pick among the outline's turns.
    ``ctx`` is unused (camera-only; masks from sam2 / sam3 / a detector).
    """
    ref = _image(reference_frame)
    cur = _image(frame)
    mr = _mask(reference_mask, ref.shape[:2])
    mc = _mask(mask, cur.shape[:2])
    out: dict[str, Any] = {
        "rotation_deg": 0.0,
        "period_deg": 360.0,
        "symmetry_order": 1,
        "outline_order": 0,
        "scores": [],
        "elongation": 0.0,
    }
    if mr.sum() < MIN_AREA_PX or mc.sum() < MIN_AREA_PX:
        return {
            "route": "low_confidence",
            **out,
            "reason": f"masks of {int(mr.sum())} / {int(mc.sum())} px: under {MIN_AREA_PX}",
        }
    _, e_ref, c_ref = axis(mr)
    _, e_cur, c_cur = axis(mc)
    elong = (e_ref + e_cur) / 2
    order = (
        int(outline_order) if outline_order in (2, 4) else (2 if elong > ELONGATION_2FOLD else 4)
    )
    sym = (
        int(symmetry_order)
        if symmetry_order in (1, 2, 4) and int(symmetry_order) <= order
        else order
    )
    period = 360.0 / order
    base = _fold(outline_angle(mc, order) - outline_angle(mr, order), period)
    out.update({"outline_order": order, "elongation": round(elong, 3)})
    if sym == order:
        return {
            "route": "rotation_found",
            **out,
            "rotation_deg": base,
            "period_deg": period,
            "symmetry_order": sym,
            "reason": f"outline turned {base:.1f} deg; a {order}-fold part: known modulo {period:.0f} deg",
        }
    ys, xs = np.nonzero(mr)
    size = int(2 * max(np.ptp(xs), np.ptp(ys)) + 8)
    pc, pmc = _patch(cur, mc, c_cur, 0.0, size)
    scores = []
    for k in range(order):
        best = (-2.0, 0.0)
        for d in np.arange(-REFINE_DEG, REFINE_DEG + 1e-9, REFINE_STEP_DEG):
            rot = base + k * period + d
            pr, pmr = _patch(ref, mr, c_ref, -rot, size)
            s = _ncc(pr, pmr, pc, pmc)
            if s > best[0]:
                best = (s, base + k * period)
        scores.append((best[0], _wrap(best[1])))
    scores.sort(key=lambda s: -s[0])
    top, rot = scores[0]
    out["scores"] = [round(s, 4) for s, _ in scores]
    if top < MIN_SCORE:
        return {
            "route": "low_confidence",
            **out,
            "rotation_deg": base,
            "period_deg": period,
            "symmetry_order": order,
            "reason": f"best appearance match {top:.2f} < {MIN_SCORE}: not the same part, occluded or relit; "
            f"outline turn {base:.1f} deg modulo {period:.0f}",
        }
    if top - scores[1][0] <= AMBIGUOUS_MARGIN:
        return {
            "route": "symmetric_ambiguous",
            **out,
            "rotation_deg": base,
            "period_deg": period,
            "symmetry_order": order,
            "reason": f"the part's {order} outline turns look alike ({top:.2f} vs {scores[1][0]:.2f}): "
            f"known modulo {period:.0f} deg only",
        }
    return {
        "route": "rotation_found",
        **out,
        "rotation_deg": _fold(rot, 360.0 / sym),
        "period_deg": 360.0 / sym,
        "symmetry_order": sym,
        "reason": f"turn picked by appearance ({top:.2f} vs {scores[1][0]:.2f} for the next outline turn)",
    }
