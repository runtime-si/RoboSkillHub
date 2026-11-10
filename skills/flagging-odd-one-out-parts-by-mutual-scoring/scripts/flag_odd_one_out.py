# Mutual scoring after MuSc / MuSc-V2 (github.com/HUST-SLOW/MuSc-V2 at b5e911dbc089, MIT; the RGB
# scoring runs in tools/musc, which adapts its code - see that file and NOTICE). Here: the parts are
# cut out of one frame and normalised (each crop turned to its mask's long axis, the 180 deg turn
# chosen to match the others, the background filled) so that "nominally identical" holds patch by
# patch, a depth branch scores each part's height map against the others' with the same
# nearest-patch rule (MuSc-V2's 3D branch uses Point-MAE tokens; this one uses the heights
# themselves, no model), and a robust z-score over the parts decides which ones are odd.
"""Find the part that differs from its siblings (fallen, tilted, flipped, wrong) with no reference image."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

MIN_PARTS = 4
"""Fewest parts scored. MuSc averages over the nearest 30 % of the others, so with fewer than ~8 one
odd part is compared with one or two siblings only; the result says so under 8."""

Z_THRESHOLD = 3.5
"""Robust z (median / 1.4826 MAD) over which a part is odd: Iglewicz-Hoaglin's outlier cut."""

MIN_RATIO = 1.4
"""An odd part's appearance score must also be this many times the median: with near-identical parts
the MAD is tiny and z alone flags noise. On 12 rendered control scenes (10 identical tiles, no odd
one) the highest score was 1.37 x the median at 448 px crops."""

MAD_FLOOR = 0.02
"""Floor of the MAD as a share of the median score, for the same reason."""

CROP_MARGIN = 0.25
"""Crop margin around a part's rotated box, as a share of its long side (context for the patches)."""

CROP_SIZE = 448
"""Crop side sent to musc.score [px] (32 x 32 DINOv2 patches; see tools/musc IMAGE_SIZE)."""

MIN_APPEARANCE_PX = 20
"""Median short side of the parts' boxes under which appearance is not scored [px]: a part under
~1.5 DINOv2 patches wide at its native size is upsampled blur (the top of a standing domino seen
from 0.2 m is 18 px)."""

ALIGNED_SPREAD_DEG = 10.0
"""The parts count as laid out aligned (a domino line, a row of bricks) when their box angles'
robust spread (1.4826 MAD, mod 180 deg) is under this; then one turned away is odd."""

TURNED_DEG = 20.0
"""How far from the parts' median angle an aligned part may be turned [deg] before it is odd."""

FOOTPRINT_RATIO = 1.25
"""A part's box side may differ from the median side by this factor before it is odd (with z over
z_threshold): a fallen domino's footprint is 2-3x a standing one's."""

FILL = 128
"""Grey the background outside a part's mask is filled with."""

DEPTH_GRID = 24
"""Height-map cells per crop side for the depth branch."""

DEPTH_PATCH = 3
"""Cells per side of a depth patch: 3 x 3 heights over the table, not mean-centred (absolute height is
what tells a fallen domino); only patches wholly inside the part's mask are used."""

DEPTH_NOISE = 0.0005
"""Floor of the depth scores' MAD [m]: a D405 / Orbbec's noise on a table at 0.2-1 m is ~0.2-2 mm;
identical renders have none, and z would explode on numerical noise."""

DEPTH_MIN_SCORE = 0.002
"""A part is odd by height only when its worst patch is at least 2 mm from every sibling's nearest one
(the station cameras' depth noise at the table)."""

TOPMIN = 0.3
"""Share of the other parts whose nearest-patch distances are averaged (MuSc's topmin_max)."""

MAX_DEPTH = 3.0

ROUTES = ("all_consistent", "outliers_found", "too_few_parts")


class Output(TypedDict):
    route: str
    detail: str
    outliers: list[int]
    parts: list[dict[str, Any]]
    scores: list[float]
    z_scores: list[float]
    depth_scores: list[float]
    depth_z_scores: list[float]
    outlier_positions: list[list[float]]
    n_parts: int
    camera_name: str
    appearance_scored: bool
    aligned_layout: bool


def _rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in "wxyz")
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _depth(d: Any) -> np.ndarray:
    a = np.asarray(d, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., 0]
    if np.nanmax(a) > 20.0:
        a = a / 1000.0
    a = np.where(np.isfinite(a) & (a > 0) & (a < MAX_DEPTH), a, 0.0)
    return a


def _frame(
    ctx: NodeContext, frame: dict[str, Any] | None, camera: str | None
) -> dict[str, Any] | None:
    if frame is not None:
        return frame
    obs = ctx.tool("robot.get_observation")
    cams = [c for c in obs.get("cameras") or [] if c.get("rgb") is not None]
    if camera:
        return next((c for c in cams if c.get("name") == camera), None)
    # The overhead camera sees a row of parts at once; a wrist camera only if nothing else is there.
    top = [
        c for c in cams if not any(k in str(c.get("name", "")) for k in ("wrist", "eye_in_hand"))
    ]
    return (top or cams or [None])[0]


def _as_mask(m: Any, shape: tuple[int, int]) -> np.ndarray:
    a = np.asarray(m)
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape != shape:
        a = cv2.resize(a.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return a > 0


def _box_mask(box: Any, shape: tuple[int, int]) -> np.ndarray:
    b = box
    if isinstance(b, dict):
        b = [
            b.get("x1", b.get("x_min")),
            b.get("y1", b.get("y_min")),
            b.get("x2", b.get("x_max")),
            b.get("y2", b.get("y_max")),
        ]
    x1, y1, x2, y2 = (int(round(float(v))) for v in b)
    m = np.zeros(shape, bool)
    m[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = True
    return m


def _plane_heights(depth: np.ndarray, k: np.ndarray, union: np.ndarray) -> np.ndarray | None:
    """Height of every pixel over the support plane (RANSAC on the pixels outside the parts) [m]."""
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    z = depth
    pts = np.stack([(uu - k[0, 2]) * z / k[0, 0], (vv - k[1, 2]) * z / k[1, 1], z], axis=-1)
    ring = cv2.dilate(union.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    bg = (
        ring & ~cv2.dilate(union.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & (z > 0)
    )
    p = pts[bg]
    if len(p) < 50:
        return None
    rng = np.random.default_rng(0)
    best_n, best_d, best_c = None, 0.0, -1
    for _ in range(200):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        d = -float(n @ a)
        cnt = int((np.abs(p @ n + d) < 0.003).sum())
        if cnt > best_c:
            best_n, best_d, best_c = n, d, cnt
    if best_n is None:
        return None
    if best_d < 0:  # normal towards the camera: heights positive above the plane
        best_n, best_d = -best_n, -best_d
    hgt = pts @ best_n + best_d
    return np.where(z > 0, hgt, np.nan)


def _crop(img: np.ndarray, rect: tuple, flip: bool, size: int, interp: int) -> np.ndarray:
    (cx, cy), (rw, rh), ang = rect
    if rw < rh:
        rw, rh, ang = rh, rw, ang + 90.0
    if flip:
        ang += 180.0
    side = max(rw, rh) * (1.0 + 2 * CROP_MARGIN)
    m = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    m[:, 2] += np.array([side / 2 - cx, side / 2 - cy])
    s = int(math.ceil(side))
    border = cv2.BORDER_CONSTANT
    out = cv2.warpAffine(img, m, (s, s), flags=interp, borderMode=border, borderValue=0)
    return cv2.resize(out, (size, size), interpolation=interp)


def _robust_z(s: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) * 1.4826
    return (s - med) / max(mad, MAD_FLOOR * abs(med), floor)


def _depth_scores(grids: list[np.ndarray], topmin: float) -> np.ndarray:
    """MuSc's nearest-patch rule on height maps: each part's 3 x 3 height patches against each other
    part's; the part's score is its worst patch (max), the patch's the mean over the nearest share."""
    feats, valid = [], []
    p = DEPTH_PATCH
    for g in grids:
        gg = np.nan_to_num(g, nan=0.0)
        ok = np.isfinite(g)
        cells = []
        oks = []
        for r in range(0, g.shape[0] - p + 1):
            for c in range(0, g.shape[1] - p + 1):
                cells.append(gg[r : r + p, c : c + p].ravel())
                oks.append(bool(ok[r : r + p, c : c + p].all()))
        feats.append(np.asarray(cells))
        valid.append(np.asarray(oks))
    n = len(grids)
    k = max(1, int((n - 1) * topmin))
    out = np.zeros(n)
    for i in range(n):
        if not valid[i].any():
            continue
        d = []
        for j in range(n):
            if j == i or not valid[j].any():
                continue
            a, b = feats[i][valid[i]], feats[j][valid[j]]
            d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
            d.append(np.sqrt(np.maximum(d2.min(axis=1), 0.0)))
        if not d:
            continue
        dm = np.sort(np.stack(d, axis=1), axis=1)[:, :k].mean(axis=1)
        out[i] = float(dm.max())
    return out


def run(
    ctx: NodeContext,
    prompt: str | None = None,
    masks: list[Any] | None = None,
    boxes: list[Any] | None = None,
    frame: dict[str, Any] | None = None,
    camera: str | None = None,
    use_depth: bool = True,
    max_parts: int = 40,
    min_parts: int = MIN_PARTS,
    z_threshold: float = Z_THRESHOLD,
    min_ratio: float = MIN_RATIO,
    align_flip: bool = True,
    use_appearance: bool = True,
    check_orientation: bool | None = None,
) -> Output:
    if masks is None and boxes is None and not prompt:
        raise ValueError("give the parts as masks, boxes, or a prompt for sam3.segment_text")
    if min_parts < 3:
        raise ValueError("min_parts must be at least 3")
    out: Output = {
        "route": "too_few_parts",
        "detail": "",
        "outliers": [],
        "parts": [],
        "scores": [],
        "z_scores": [],
        "depth_scores": [],
        "depth_z_scores": [],
        "outlier_positions": [],
        "n_parts": 0,
        "camera_name": "",
        "appearance_scored": False,
        "aligned_layout": False,
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        logger.info("flagging-odd-one-out: %s (%s)", route, why)
        return out

    cam = _frame(ctx, frame, camera)
    if cam is None or cam.get("rgb") is None:
        return finish("too_few_parts", "no camera frame with RGB")
    out["camera_name"] = str(cam.get("name", ""))
    rgb = np.ascontiguousarray(np.asarray(cam["rgb"])[..., :3]).astype(np.uint8)
    shape = rgb.shape[:2]
    if masks is not None:
        parts = [_as_mask(m, shape) for m in masks]
    elif boxes is not None:
        parts = [_box_mask(b, shape) for b in boxes]
    else:
        seg = ctx.tool(
            "sam3.segment_text", image=rgb, query=str(prompt), max_results=int(max_parts)
        )
        parts = [_as_mask(m, shape) for m in seg.get("masks") or []]
    parts = [m for m in parts if m.sum() >= 30][: int(max_parts)]
    out["n_parts"] = len(parts)
    if len(parts) < int(min_parts):
        return finish("too_few_parts", f"{len(parts)} part(s) found, {min_parts} needed")

    rects = []
    for m in parts:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rects.append(cv2.minAreaRect(max(cnts, key=cv2.contourArea)))
    union = np.any(parts, axis=0)
    filled = []
    for m in parts:
        im = rgb.copy()
        im[~m] = FILL
        filled.append(im)
    flips = [False] * len(parts)
    if align_flip:
        thumbs = [
            [
                cv2.cvtColor(
                    _crop(filled[i], rects[i], f, 32, cv2.INTER_AREA), cv2.COLOR_RGB2GRAY
                ).astype(np.float32)
                for f in (False, True)
            ]
            for i in range(len(parts))
        ]
        for _ in range(2):
            ref = np.median([thumbs[i][int(flips[i])] for i in range(len(parts))], axis=0)
            flips = [
                float(np.sum((thumbs[i][1] - ref) ** 2)) < float(np.sum((thumbs[i][0] - ref) ** 2))
                for i in range(len(parts))
            ]
    crops = [
        _crop(filled[i], rects[i], flips[i], CROP_SIZE, cv2.INTER_LINEAR) for i in range(len(parts))
    ]
    cmasks = [
        _crop(parts[i].astype(np.uint8) * 255, rects[i], flips[i], CROP_SIZE, cv2.INTER_NEAREST)
        for i in range(len(parts))
    ]
    short = np.array([min(r[1]) for r in rects])
    long_ = np.array([max(r[1]) for r in rects])
    appearance = bool(use_appearance) and float(np.median(short)) >= MIN_APPEARANCE_PX
    s = np.zeros(len(parts))
    z = np.zeros(len(parts))
    odd_rgb = np.zeros(len(parts), bool)
    if appearance:
        res = ctx.tool("musc.score", images=crops, masks=cmasks)
        s = np.asarray(res["scores"], dtype=np.float64)
        z = _robust_z(s)
        med = float(np.median(s))
        odd_rgb = (z > z_threshold) & (s > min_ratio * med)

    # Geometry: the parts' boxes against each other (the crops above were turned upright, so a part
    # turned in place looks like its siblings there).
    ang = np.array([(r[2] + (90.0 if r[1][0] < r[1][1] else 0.0)) % 180.0 for r in rects])
    ref = math.degrees(np.angle(np.mean(np.exp(2j * np.radians(ang))))) / 2.0
    dev = np.array([abs(((a - ref) + 90.0) % 180.0 - 90.0) for a in ang])
    spread = float(np.median(dev)) * 1.4826
    aligned = spread < ALIGNED_SPREAD_DEG if check_orientation is None else bool(check_orientation)
    odd_turn = aligned & (dev > TURNED_DEG)
    odd_size = np.zeros(len(parts), bool)
    for side in (long_, short):
        zs = _robust_z(side, 1.0)
        rs = side / max(float(np.median(side)), 1e-9)
        odd_size |= (np.abs(zs) > z_threshold) & (
            (rs > FOOTPRINT_RATIO) | (rs < 1.0 / FOOTPRINT_RATIO)
        )

    dz = np.zeros(len(parts))
    ds = np.zeros(len(parts))
    odd_depth = np.zeros(len(parts), bool)
    hmap = None
    if use_depth and cam.get("depth") is not None and cam.get("intrinsics") is not None:
        depth = _depth(cam["depth"])
        if depth.shape != shape:
            depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        hmap = _plane_heights(depth, np.asarray(cam["intrinsics"], dtype=np.float64), union)
        if hmap is not None:
            grids = []
            for i, m in enumerate(parts):
                h = np.where(m, hmap, np.nan).astype(np.float32)
                g = _crop(
                    np.nan_to_num(h, nan=-1.0), rects[i], flips[i], DEPTH_GRID, cv2.INTER_NEAREST
                )
                grids.append(np.where(g > -0.5, g, np.nan))
            ds = _depth_scores(grids, TOPMIN)
            dz = _robust_z(ds, DEPTH_NOISE)
            dmed = float(np.median(ds))
            # Depth scores are metres of height difference: also demand 2 mm, under which the D405 /
            # Orbbec noise at the table is.
            odd_depth = (dz > z_threshold) & (ds > max(min_ratio * dmed, DEPTH_MIN_SCORE))

    k = (
        np.asarray(cam.get("intrinsics"), dtype=np.float64)
        if cam.get("intrinsics") is not None
        else None
    )
    t_wc = None
    if cam.get("pose") is not None:
        t_wc = np.eye(4)
        t_wc[:3, :3] = _rot(cam["pose"]["rotation"])
        pos = cam["pose"]["position"]
        t_wc[:3, 3] = (
            [float(pos[a]) for a in "xyz"] if isinstance(pos, dict) else np.asarray(pos, float)
        )
    depth_img = _depth(cam["depth"]) if cam.get("depth") is not None else None
    for i, m in enumerate(parts):
        (cx, cy), (rw, rh), ang = rects[i]
        why = []
        if odd_rgb[i]:
            why.append("appearance")
        if odd_depth[i]:
            why.append("height")
        if odd_turn[i]:
            why.append("orientation")
        if odd_size[i]:
            why.append("footprint")
        rec = {
            "index": i,
            "centre_px": [float(cx), float(cy)],
            "obb_px": {
                "cx": float(cx),
                "cy": float(cy),
                "w": float(rw),
                "h": float(rh),
                "angle_deg": float(ang),
            },
            "area_px": int(m.sum()),
            "score": float(s[i]),
            "z": float(z[i]),
            "depth_score": float(ds[i]),
            "depth_z": float(dz[i]),
            "turn_deg": round(float(dev[i]), 2),
            "flipped": bool(flips[i]),
            "odd": bool(why),
            "why": why,
        }
        out["parts"].append(rec)
        if why:
            out["outliers"].append(i)
            if k is not None and t_wc is not None and depth_img is not None:
                zz = depth_img[m & (depth_img > 0)]
                if zz.size:
                    zc = float(np.median(zz))
                    pc = np.array(
                        [(cx - k[0, 2]) * zc / k[0, 0], (cy - k[1, 2]) * zc / k[1, 1], zc, 1.0]
                    )
                    out["outlier_positions"].append([float(v) for v in (t_wc @ pc)[:3]])
    out["scores"] = [round(float(v), 5) for v in s]
    out["z_scores"] = [round(float(v), 3) for v in z]
    out["depth_scores"] = [round(float(v), 6) for v in ds]
    out["depth_z_scores"] = [round(float(v), 3) for v in dz]
    few = (
        f" (only {len(parts)} parts: under 8 the mutual scores are weak)" if len(parts) < 8 else ""
    )
    if not appearance and use_appearance:
        few += f" (appearance not scored: parts {float(np.median(short)):.0f} px wide)"
    out["appearance_scored"] = appearance
    out["aligned_layout"] = bool(aligned)
    if out["outliers"]:
        return finish(
            "outliers_found",
            f"{len(out['outliers'])} of {len(parts)} parts odd: "
            + ", ".join(f"#{i} ({'+'.join(out['parts'][i]['why'])})" for i in out["outliers"])
            + few,
        )
    return finish(
        "all_consistent",
        f"{len(parts)} parts, highest z {float(max(z.max(), dz.max())):.1f} under {z_threshold}{few}",
    )
