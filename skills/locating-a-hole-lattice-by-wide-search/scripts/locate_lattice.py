"""Where a holder's lattice of round holes really is, searched wide: translation over +-1.5 pitch, yaw +-15 deg.

**The failure this exists for.** A holder that was pushed, or a lattice fitted one column off, sits a whole pitch
from where the graph believes it is. A local refinement (a few millimetres round the prior) then finds a local
optimum, reports it as found, and the graph inserts over the table beside the holder: on a real YAM (pens task,
probe-insert-1 round 2, 2026-11-10) the believed lattice was about one pitch off, the wrist camera's +-5 mm shift
stopped at its edge on every frame, and 18 insertions went down into empty air next to the holder.

**How.** The frame's depth is lifted to the world and binned on a 1 mm grid at the hole face's height: cells with a
point within a few mm of the face are *face*; cells whose points lie well under it (a hole's floor, the table
past the holder) or that read dark in RGB at face height are *open*. A node is a matched hole when the disc of the
hole's radius is open and the ring round it is face: the holder's own top surface round every hole. Over the table
there is no face, so no node matches there. Matched counts over a dense grid of poses come from per-cell maps
(one convolution each), so the whole +-1.5 pitch x +-15 deg search is a lookup; the best pose is refined to 0.25 mm
/ 0.5 deg. Every pose a whole pitch away from the best that matches nearly as many holes is a second hypothesis:
a lattice shifted by one pitch along a row still fits the holes it shares, which is exactly the mistake to catch.
"""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np

logger = logging.getLogger(__name__)

CELL = 0.001
"""Grid cell [m] at least; coarser when the camera's pixel on the face is wider (an overhead camera)."""
FACE_BAND = 0.004
"""A point within this of the face height is on the face."""
DEEP = 0.006
"""A point this far under the face is seen past it (into a hole, or the table past the holder)."""


class Output(TypedDict):
    route: str
    pose: dict
    shift: dict
    matched: int
    visible: int
    contradicted: int
    of: int
    residual_mm: float
    second: dict
    face_z: float
    nodes: list
    detail: str


def _rot(q: dict) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in "wxyz")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _T(cam: dict) -> np.ndarray:
    if cam.get("T_world_cam") is not None:
        return np.asarray(cam["T_world_cam"], dtype=float)
    p = cam["pose"]
    T = np.eye(4)
    T[:3, :3] = _rot(p["rotation"])
    pos = p["position"]
    T[:3, 3] = [float(pos[a]) for a in "xyz"] if isinstance(pos, dict) else pos
    return T


def _camera(obs: dict, name: str) -> dict | None:
    cams = [
        c
        for c in obs.get("cameras") or []
        if c.get("depth") is not None
        and (c.get("pose") is not None or c.get("T_world_cam") is not None)
    ]
    if name:
        return next((c for c in cams if c.get("name") == name), None)
    return cams[0] if cams else None


def lattice_nodes(prior: dict) -> np.ndarray:
    """The prior's hole centres [N, 2] (world xy): ``nodes`` as given, else rows x cols at ``pitch_m`` round
    ``centre`` turned by ``yaw_deg`` (columns along the turned x)."""
    if prior.get("nodes"):
        return np.asarray(
            [
                [float(n["x"]), float(n["y"])] if isinstance(n, dict) else n[:2]
                for n in prior["nodes"]
            ],
            dtype=float,
        )
    p = prior["pitch_m"]
    px, py = (float(p), float(p)) if np.isscalar(p) else (float(p[0]), float(p[1]))
    rows, cols = int(prior["rows"]), int(prior["cols"])
    c = prior["centre"]
    cx, cy = (float(c["x"]), float(c["y"])) if isinstance(c, dict) else (float(c[0]), float(c[1]))
    a = math.radians(float(prior.get("yaw_deg", 0.0)))
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    g = np.array(
        [
            [(j - (cols - 1) / 2) * px, (i - (rows - 1) / 2) * py]
            for i in range(rows)
            for j in range(cols)
        ]
    )
    return g @ R.T + [cx, cy]


def border_nodes(rel: np.ndarray, pitch: float) -> np.ndarray:
    """The lattice's next ring: each node moved one pitch along each of the lattice's two directions (from the
    nearest-neighbour steps), kept where no node is. No hole may be there: a lattice placed one pitch off puts this
    ring onto the real holder's outer row or column."""
    if len(rel) < 2:
        return np.zeros((0, 2))
    d = rel[None] - rel[:, None]
    n = np.linalg.norm(d, axis=-1)
    steps = d[(n > 0.5 * pitch) & (n < 1.3 * pitch)]
    steps = steps[(steps[:, 0] > 1e-9) | ((np.abs(steps[:, 0]) <= 1e-9) & (steps[:, 1] > 0))]
    if not len(steps):
        return np.zeros((0, 2))
    a = steps[np.argmin(np.abs(np.arctan2(steps[:, 1], steps[:, 0])))]
    rest = steps[np.abs(steps[:, 0] * a[1] - steps[:, 1] * a[0]) > 0.5 * pitch * pitch]
    basis = [a] + ([rest[0]] if len(rest) else [])
    out = []
    for p in rel:
        for b in basis:
            for q in (p + b, p - b):
                if np.min(np.linalg.norm(rel - q, axis=1)) > 0.3 * pitch and (
                    not out or np.min(np.linalg.norm(np.asarray(out) - q, axis=1)) > 0.3 * pitch
                ):
                    out.append(q)
    return np.asarray(out).reshape(-1, 2)


def _points(cam: dict, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    dep = np.asarray(cam["depth"], dtype=float)[::stride, ::stride]
    K = np.asarray(cam["intrinsics"], dtype=float)
    v, u = np.nonzero((dep > 0.05) & (dep < 2.5))
    z = dep[v, u]
    uu, vv = u * stride, v * stride
    C = np.stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z], 1)
    T = _T(cam)
    P = C @ T[:3, :3].T + T[:3, 3]
    dark = np.zeros(len(P), dtype=bool)
    if cam.get("rgb") is not None:
        rgb = np.asarray(cam["rgb"])[::stride, ::stride]
        dark = rgb[v, u].max(axis=-1) < 60
    return P, dark


def _face_height(P: np.ndarray, xy: np.ndarray, guess: float | None, span: float) -> float | None:
    """The face's height over the lattice's area: the densest 1 mm band within ``guess`` +-15 mm, else the highest band
    holding 15 % of the densest one's points."""
    near = np.linalg.norm(P[:, :2] - xy.mean(axis=0), axis=1) < span
    z = P[near, 2]
    if guess is not None:
        z = z[np.abs(z - guess) < 0.015]
    if len(z) < 200:
        return guess
    h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.001, 0.001))
    if guess is not None:
        return float(e[h.argmax()] + 0.0005)
    # no guess: the holder's face is the highest band holding a good share of the points (the table round it may
    # hold more)
    k = int(np.nonzero(h >= 0.15 * h.max())[0].max())
    return float(e[k] + 0.0005)


def _disc(r: float, cell: float = CELL) -> np.ndarray:
    n = int(math.ceil(r / cell))
    y, x = np.mgrid[-n : n + 1, -n : n + 1] * cell
    return (np.hypot(x, y) <= r).astype(float)


def _ring(r0: float, r1: float, cell: float = CELL) -> np.ndarray:
    n = int(math.ceil(r1 / cell))
    y, x = np.mgrid[-n : n + 1, -n : n + 1] * cell
    d = np.hypot(x, y)
    return ((d >= r0) & (d <= r1)).astype(float)


def _conv(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    from numpy.fft import irfft2, rfft2

    s = (img.shape[0] + k.shape[0] - 1, img.shape[1] + k.shape[1] - 1)
    out = irfft2(rfft2(img, s) * rfft2(k, s), s)
    oy, ox = k.shape[0] // 2, k.shape[1] // 2
    return out[oy : oy + img.shape[0], ox : ox + img.shape[1]]


def hole_map(
    P: np.ndarray,
    dark: np.ndarray,
    face_z: float,
    lo: np.ndarray,
    shape: tuple,
    radius: float,
    cell: float = CELL,
):
    """Per cell: how much a hole centred there looks like one (ring face fraction - disc face fraction, when both
    are seen), and whether the cell's neighbourhood was seen at all."""
    ij = np.floor((P[:, :2] - lo) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0])
    ij, z, dk = ij[ok], P[ok, 2], dark[ok]
    face = (np.abs(z - face_z) < FACE_BAND) & ~dk
    open_ = (z < face_z - DEEP) | (dk & (np.abs(z - face_z) < 0.03))
    F = np.zeros(shape)
    Op = np.zeros(shape)
    np.add.at(F, (ij[face, 1], ij[face, 0]), 1.0)
    np.add.at(Op, (ij[open_, 1], ij[open_, 0]), 1.0)
    F, Op = (F > 0).astype(float), ((Op > 0) & (F == 0)).astype(float)
    disc, ring = (
        _disc(0.75 * radius, cell),
        _ring(1.25 * radius, 1.25 * radius + max(0.004, 2 * cell), cell),
    )
    fd, od = _conv(F, disc), _conv(Op, disc)
    fr, orr = _conv(F, ring), _conv(Op, ring)
    seen_d, seen_r = fd + od, fr + orr
    score = np.where(
        (seen_d > 0.3 * disc.sum()) & (seen_r > 0.3 * ring.sum()),
        fr / np.maximum(seen_r, 1e-9) - fd / np.maximum(seen_d, 1e-9),
        np.nan,
    )
    # the holder's body: the face closed over its holes (dilated, then eroded, by a hole's radius and 3 mm). A
    # centre outside it is past the holder's edge, where the ring is half face, half table and scores like a poor
    # hole (0.2-0.3): it is no hole
    k = _disc(radius + 0.003, cell)
    grown = _conv(F, k) > 0.5
    body = _conv((~grown).astype(float), k) < 0.5
    score = np.where(np.isnan(score) | body, score, -1.0)
    return score, seen_d + seen_r


def run(
    ctx: Any,
    prior: dict,
    camera: dict | None = None,
    camera_name: str = "",
    hole_d: float = 0.0187,
    face_z: float | None = None,
    search_pitch: float = 1.5,
    search_yaw_deg: float = 15.0,
    match_score: float = 0.25,
    miss_score: float | None = None,
    min_matched: int = 4,
    ambiguous_ratio: float = 0.85,
) -> Output:
    """The lattice's pose from one RGB-D frame (``camera``, or ``camera_name`` / the first camera with depth of a
    fresh robot.get_observation): ``prior`` {"nodes": [[x, y], ...]} or {"centre", "yaw_deg", "pitch_m", "rows",
    "cols"} in the world frame; ``face_z`` the holes' face height (else the densest height there).

    Searches the prior moved by up to ``search_pitch`` pitches and turned up to ``search_yaw_deg`` about its centre.
    A node matches when its hole score (ring on the face, disc open) is over ``match_score``. Routes:
    located        the best pose matches at least ``min_matched`` holes, is inside the window, and no pose a pitch
                   or more away matches over ``ambiguous_ratio`` of its holes
    ambiguous      such a second pose exists (``second`` holds it): a lattice one pitch off fits nearly as well
    at_search_edge the best pose lies on the window's border: the truth may be further out
    not_found      fewer than ``min_matched`` holes match anywhere (no holder face in view, or no frame)"""
    empty = {
        "pose": {},
        "shift": {},
        "matched": 0,
        "visible": 0,
        "contradicted": 0,
        "of": 0,
        "residual_mm": -1.0,
        "second": {},
        "face_z": float(face_z or 0.0),
        "nodes": [],
    }
    try:
        cam = camera or _camera(ctx.tool("robot.get_observation"), camera_name)
    except Exception as exc:  # noqa: BLE001
        return {"route": "not_found", **empty, "detail": f"no observation: {exc}"}
    if cam is None or cam.get("depth") is None:
        return {"route": "not_found", **empty, "detail": "no camera with depth and a pose"}
    N0 = lattice_nodes(prior)
    c0 = N0.mean(axis=0)
    rel = N0 - c0
    d = np.linalg.norm(rel[:, None] - rel[None], axis=-1)
    pitch = float(np.min(d[d > 1e-6])) if len(N0) > 1 else 0.02
    reach = search_pitch * pitch
    ring = border_nodes(rel, pitch)
    span = float(np.max(np.linalg.norm(rel, axis=1))) + reach + 0.02
    P, dark = _points(cam, stride=1 if np.asarray(cam["depth"]).size < 400_000 else 2)
    fz = _face_height(
        P,
        N0,
        face_z,
        span if face_z is not None else float(np.max(np.linalg.norm(rel, axis=1))) + pitch,
    )
    if fz is None:
        return {"route": "not_found", **empty, "detail": "no points near the lattice"}
    # a node in view that does not look like a hole contradicts the pose: past the holder's last row or column the
    # ring is half face, half table (0.1-0.25 here), so a neutral band would let a lattice one pitch off tie
    miss = match_score if miss_score is None else float(miss_score)
    T = _T(cam)
    K = np.asarray(cam["intrinsics"], dtype=float)
    cell = max(CELL, 1.2 * abs(float(T[2, 3]) - fz) / float(K[0, 0]))
    lo = c0 - span
    shape = (int(2 * span / cell) + 1, int(2 * span / cell) + 1)
    score, seen = hole_map(P, dark, fz, lo, shape, hole_d / 2, cell)

    def lookup(pts: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        X = pts[None, :, 0] + dx[:, None]
        Y = pts[None, :, 1] + dy[:, None]
        i = np.clip(np.round((X - lo[0]) / cell).astype(int), 0, shape[1] - 1)
        j = np.clip(np.round((Y - lo[1]) / cell).astype(int), 0, shape[0] - 1)
        return score[j, i]

    def evaluate(dx: np.ndarray, dy: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        a = math.radians(yaw)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        s = lookup(rel @ R.T + c0, dx, dy)
        seen_n = ~np.isnan(s)
        m = np.nan_to_num(s, nan=0.0)
        hit = seen_n & (m > match_score)
        matched = hit.sum(axis=1)
        contra = (seen_n & (m < miss)).sum(axis=1)
        # the ring past the border: a hole there scoring like this pose's own holes (0.8 of their median; a cell
        # just past the holder's edge scores 0.1-0.3) says the lattice is a row or column off
        med = np.where(
            matched > 0, np.sum(np.where(hit, m, 0.0), axis=1) / np.maximum(matched, 1), 1.0
        )
        b = (
            np.nan_to_num(lookup(ring @ R.T + c0, dx, dy), nan=0.0)
            if len(ring)
            else np.zeros((len(dx), 0))
        )
        outside = (b > np.maximum(match_score, 0.8 * med)[:, None]).sum(axis=1)
        return matched - contra - outside + 0.01 * np.where(hit, m, 0.0).sum(axis=1), np.stack(
            [matched, contra + outside, seen_n.sum(axis=1)], 1
        )

    step = 0.001
    g = np.arange(-reach, reach + 1e-9, step)
    DX, DY = [a.ravel() for a in np.meshgrid(g, g)]
    yaws = np.arange(-search_yaw_deg, search_yaw_deg + 1e-9, 1.0)
    cand = []  # (value, dx, dy, yaw)
    for yaw in yaws:
        v, _ = evaluate(DX, DY, float(yaw))
        for k in np.argsort(v)[-40:]:
            cand.append((float(v[k]), float(DX[k]), float(DY[k]), float(yaw)))
    cand.sort(key=lambda c: -c[0])
    best = cand[0]
    # refine round the best: 0.25 mm, 0.5 deg
    fg = np.arange(-0.002, 0.002 + 1e-9, 0.00025)
    FX, FY = [a.ravel() for a in np.meshgrid(fg, fg)]
    top = best
    for yaw in np.arange(best[3] - 1.0, best[3] + 1.0 + 1e-9, 0.5):
        v, _ = evaluate(best[1] + FX, best[2] + FY, float(yaw))
        k = int(np.argmax(v))
        if v[k] > top[0] + 1e-9:
            top = (float(v[k]), float(best[1] + FX[k]), float(best[2] + FY[k]), float(yaw))
    best = top
    _, mc = evaluate(np.array([best[1]]), np.array([best[2]]), best[3])
    matched, contra, vis = (int(v) for v in mc[0])
    # the best other hypothesis: at least ~a pitch away in translation (or 8 deg in yaw)
    far = [
        c
        for c in cand
        if math.hypot(c[1] - best[1], c[2] - best[2]) > 0.7 * pitch or abs(c[3] - best[3]) > 8.0
    ]
    second = far[0] if far else None
    a = math.radians(best[3])
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    nodes = rel @ R.T + c0 + [best[1], best[2]]
    # residual: each matched node's own best cell within 2 mm
    res = []
    for p in nodes:
        i, j = int(round((p[0] - lo[0]) / cell)), int(round((p[1] - lo[1]) / cell))
        w = score[max(0, j - 2) : j + 3, max(0, i - 2) : i + 3]
        if (
            w.size
            and np.nanmax(np.nan_to_num(w, nan=-1)) > match_score
            and np.nan_to_num(score[j, i], nan=-1) > match_score
        ):
            jj, ii = np.unravel_index(np.nanargmax(np.nan_to_num(w, nan=-1)), w.shape)
            res.append(math.hypot(ii - min(2, i), jj - min(2, j)) * cell)
    out = {
        "pose": {
            "x": float(c0[0] + best[1]),
            "y": float(c0[1] + best[2]),
            "yaw_deg": float(best[3]),
        },
        "shift": {"x": best[1], "y": best[2], "yaw_deg": best[3]},
        "matched": matched,
        "visible": vis,
        "contradicted": contra,
        "of": int(len(N0)),
        "residual_mm": round(float(np.median(res)) * 1000, 2) if res else -1.0,
        "second": (
            {
                "x": float(c0[0] + second[1]),
                "y": float(c0[1] + second[2]),
                "yaw_deg": second[3],
                "value": round(second[0], 2),
                "best_value": round(best[0], 2),
            }
            if second
            else {}
        ),
        "face_z": float(fz),
        "nodes": [[round(float(x), 5), round(float(y), 5)] for x, y in nodes],
    }
    edge = (
        abs(abs(best[1]) - reach) < 1.5 * step
        or abs(abs(best[2]) - reach) < 1.5 * step
        or abs(abs(best[3]) - search_yaw_deg) < 0.75
    )
    if matched < min_matched or best[0] < min_matched:
        route, why = "not_found", f"{matched} holes matched (under {min_matched})"
    elif edge:
        route, why = "at_search_edge", "the best pose lies on the search window's border"
    elif second and second[0] >= ambiguous_ratio * best[0]:
        route, why = (
            "ambiguous",
            (
                f"a pose {math.hypot(second[1] - best[1], second[2] - best[2]) * 1000:.0f} mm away "
                f"scores {second[0]:.1f} against {best[0]:.1f} (matched - contradicted)"
            ),
        )
    else:
        route, why = "located", ""
    logger.info(
        "[lattice] %s matched=%d/%d shift=(%.1f, %.1f) mm yaw=%.1f",
        route,
        matched,
        len(N0),
        best[1] * 1000,
        best[2] * 1000,
        best[3],
    )
    return {"route": route, **out, "detail": why}
