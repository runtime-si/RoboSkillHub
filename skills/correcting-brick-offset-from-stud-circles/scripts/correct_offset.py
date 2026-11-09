# Reimplemented from the Eye-in-Finger paper (arXiv 2503.06848, Sec. III-B, Algorithm 1) and the
# behaviour of github.com/kevinpoi/legoscope (detect_offset.py at 2abdc205ef36), which has no licence:
# reference only, none of its code is used. Kept from the paper: peek at a fixed height, take the two
# studs nearest the expected pair, fit each stud's circle at the known stud radius using only the rim
# that is actually seen (the paper forces its circle tangent to the visible mask edges), compare the
# pair's midpoint and line with the tool axis (optionally a calibrated tool-centre pixel), step the hand
# and repeat until under tolerance. New here: the studs come from the wrist RGB-D depth (no YOLO model),
# the comparison is metric on the fitted brick-top plane, the pair is identified against a tracked
# prior with the half-pitch ambiguity rule, and the yaw is folded to the pair's 180 deg symmetry.
"""Close the last millimetre over a LEGO / DUPLO brick: peek at its studs, fit circles, step the hand, repeat."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PRESETS = {
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
"""The hub's LEGO / DUPLO dimensions (calibrating-baseplate-stud-grid, pressing-a-brick-onto-studs)."""

PEEK_HEIGHT = 0.03
"""Tool centre over the stud tops while peeking [m]. A YAM wrist D405 (86 mm behind, 75 mm above the
tool, 45 deg forward) then sees the studs under the tool ~6 deg off its axis at ~0.14 m, past its
~7 cm minimum range; a LEGO stud is ~13 px wide there. The paper's EiF camera peeks at 17-30 mm."""

TOLERANCE = 0.0002
"""Residual tool-to-pair offset that counts as aligned [m]. The paper's loop ends at its detector's
noise (~0.3-0.4 mm); here one frame's midpoint scatters 0.025 mm (sd) on the renders, so 0.2 mm leaves
room for a real D405's noise."""

TOLERANCE_DEG = 1.0
"""Residual yaw that counts as aligned [deg]: two studs 8 mm apart at 0.15 mm each give ~1 deg."""

MAX_ITERATIONS = 6
"""Most measure-and-step rounds (the paper's loop converges in 2-4)."""

GAIN = 1.0
"""Fraction of the measured correction commanded per step."""

MAX_STEP_PITCHES = 0.5
"""Largest lateral step per round, in pitches (4 mm LEGO): a bigger one is not a last-centimetre
correction and would pass the next pair."""

MAX_TURN_DEG = 10.0
"""Largest yaw step per round [deg]."""

MAX_PRIOR_ERROR = 0.4
"""Largest distance between the expected and the found pair midpoint, in pitches. At half a pitch a
neighbouring pair is as near (the paper's 'within half a stud' precondition)."""

MIN_PAIR_MARGIN = 0.2
"""The second-nearest pair must be this much farther than the nearest, in pitches, or the pair is
ambiguous."""

MAX_PAIR_YAW_DEG = 30.0
"""A stud pair's line must be within this of the expected pair line [deg]; perpendicular pairs of a
2-wide brick are other pairs."""

PAIR_SPACING_TOL = 0.15
"""Stud spacing accepted for a pair: one pitch +- this fraction."""

LATTICE_INLIER = 0.2
"""A stud within this many pitches of a node of the pair's lattice joins the yaw / midpoint fit. On
the renders a pair alone gave yaw to ~0.5 deg (one frame, sd); a 2 x 4 top's 8 studs to ~0.2."""

CELL_PITCHES = 1.0 / 16.0
"""Height-map cell, in pitches (0.5 mm LEGO), about a D405 pixel's footprint at 0.14 m. On 10 renders
1/16 gave a 0.02 mm mean error along the view direction against 0.07 at 1/12 and 0.13 at 1/10."""

TOP_FRACTION = 0.5
"""A seen cell is a stud top when its mean height over the brick top is over half a stud height
(0.85 mm LEGO, ~4 sigma of a D405 at 0.14 m). A cell cut by the rim averages the two heights in
proportion to its area, so half a stud height puts the edge on the rim; 0.6 left a 0.2 mm bias away
from the camera on the renders."""

PLANE_BAND = 0.0006
"""Inlier band of the brick-top plane [m]: ~2.5 sigma of a D405 at 0.14 m, under a LEGO stud's 1.7 mm."""

MIN_RIM_POINTS = 8
MIN_RIM_ARC_DEG = 100.0
"""A stud's circle is used when this many rim edges over at least this arc are seen; the rest of the
rim may be hidden (by the stud itself from an oblique camera, a finger, a neighbour)."""

MAX_DEPTH = 0.6
MIN_DEPTH = 0.04

SETTLE_READS = 5
SETTLE_DISTANCE = 0.0002
"""Two tool readings this close count as settled [m]."""

ROUTES = ("aligned", "ambiguous_stud_pair", "lost_studs", "max_iterations", "unreachable")


class Output(TypedDict):
    route: str
    detail: str
    pair_pose: dict[str, Any]
    tool_pose: dict[str, Any]
    offset_mm: list[float]
    yaw_offset_deg: float
    iterations: int
    moves: int
    history: list[dict[str, Any]]
    stud_centres: list[list[float]]
    pair_px: list[float]
    pair_range_m: float
    rim_arc_deg: list[float]
    tool_axis_source: str
    camera_name: str
    system: str
    stud_pitch: float


# -- poses ----------------------------------------------------------------------------------


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


def _quat(r: np.ndarray) -> dict[str, float]:
    m = r
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    return dict(zip("wxyz", (float(v) for v in q), strict=True))


def _xyz(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=np.float64).reshape(3)


def _matrix(pose: Any) -> np.ndarray:
    pose = pose.get("pose", pose) if isinstance(pose, dict) and "pose" in pose else pose
    m = np.eye(4)
    m[:3, :3] = _rot(pose["rotation"]) if pose.get("rotation") else np.eye(3)
    m[:3, 3] = _xyz(pose["position"])
    return m


def _se3(m: np.ndarray) -> dict[str, Any]:
    return {
        "position": dict(zip("xyz", (float(v) for v in m[:3, 3]), strict=True)),
        "rotation": _quat(m[:3, :3]),
    }


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _wrap(angle: float, period: float) -> float:
    return (angle + period / 2.0) % period - period / 2.0


def _refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or ""."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


# -- the wrist frame ------------------------------------------------------------------------


def _is_wrist(cam: dict[str, Any]) -> bool:
    return any(k in str(cam.get("name", "")) for k in ("eye_in_hand", "wrist"))


def _wrist_frame(obs: dict[str, Any], arm_id: int, name: str | None) -> dict[str, Any] | None:
    cams = [
        c
        for c in obs.get("cameras") or []
        if c.get("pose") is not None
        and c.get("intrinsics") is not None
        and c.get("depth") is not None
    ]
    if name:
        return next((c for c in cams if c.get("name") == name), None)
    wrist = [c for c in cams if _is_wrist(c)]
    if not wrist:
        return None
    arms = obs.get("arms") or []
    if len(arms) > arm_id and arms[arm_id].get("ee_pose") is not None:
        tool = _matrix(arms[arm_id]["ee_pose"])[:3, 3]
        return min(wrist, key=lambda c: float(np.linalg.norm(_matrix(c["pose"])[:3, 3] - tool)))
    return wrist[0]


def _depth(d: Any) -> np.ndarray:
    a = np.asarray(d, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., 0]
    if np.nanmax(a) > 20.0:  # millimetres
        a = a / 1000.0
    return np.where(np.isfinite(a), a, 0.0)


# -- one measurement ------------------------------------------------------------------------


def _plane(
    pts: np.ndarray, band: float, rng: np.random.Generator
) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane (n, d) with n.p + d = 0, n roughly +z of the given points, and its inliers."""
    best = (np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(len(pts), bool))
    if len(pts) < 3:
        return best
    for _ in range(150):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12 or abs(n[2]) / nn < 0.8:
            continue
        n = n / nn
        if n[2] < 0:
            n = -n
        inl = np.abs(pts @ n - n @ a) < band
        if inl.sum() > best[2].sum():
            best = (n, -float(n @ a), inl)
    n, _, inl = best
    if inl.sum() >= 3:
        q = pts[inl]
        c = q.mean(axis=0)
        n = np.linalg.svd(q - c)[2][2]
        n = n if n[2] > 0 else -n
        inl = np.abs((pts - c) @ n) < band
        return n, -float(n @ c), inl
    return best


def _fit_circle(edges: np.ndarray, r: float, c0: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre of the radius-r circle through the rim edges (Gauss-Newton, Huber at a quarter radius)."""
    c = c0.astype(np.float64).copy()
    for _ in range(25):
        d = edges - c
        rho = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        res = rho - r
        w = np.where(np.abs(res) < 0.25 * r, 1.0, 0.25 * r / np.maximum(np.abs(res), 1e-12))
        jac = -d / rho[:, None]
        h = (jac * w[:, None]).T @ jac + 1e-12 * np.eye(2)
        step = -np.linalg.solve(h, (jac * w[:, None]).T @ res)
        c += step
        if np.linalg.norm(step) < 1e-7:
            break
    rho = np.linalg.norm(edges - c, axis=1)
    return c, float(np.sqrt(np.mean((rho - r) ** 2)))


def _arc_deg(edges: np.ndarray, c: np.ndarray) -> float:
    """Angular extent of the rim edges around c (360 minus the largest gap) [deg]."""
    ang = np.sort(np.arctan2(edges[:, 1] - c[1], edges[:, 0] - c[0]))
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
    return float(math.degrees(2 * math.pi - gaps.max()))


def measure(
    frame: dict[str, Any],
    prior: np.ndarray,
    geom: dict[str, float],
    seed: int = 0,
) -> dict[str, Any]:
    """Find the stud pair nearest ``prior`` (4x4: midpoint at the brick top, x along the pair, z up).

    Returns {ok, why, pair (4x4 world, at the brick top), studs (list of world xyz at the stud tops),
    arc (deg per stud), cands, plane_n, plane_d}."""
    pitch, stud_h, r = geom["stud_pitch"], geom["stud_height"], geom["stud_diameter"] / 2.0
    out: dict[str, Any] = {"ok": False, "why": "", "studs": [], "arc": [], "cands": 0}
    depth = _depth(frame["depth"])
    k = np.asarray(frame["intrinsics"], dtype=np.float64)
    t_wc = _matrix(frame["pose"])
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    ok = (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
    z = depth[ok]
    pc = np.stack([(uu[ok] - k[0, 2]) * z / k[0, 0], (vv[ok] - k[1, 2]) * z / k[1, 1], z], axis=1)
    pw = pc @ t_wc[:3, :3].T + t_wc[:3, 3]
    # Prior frame: points near the expected pair, heights over its brick top.
    pl = (pw - prior[:3, 3]) @ prior[:3, :3]
    win = 2.0 * pitch + r
    near = (np.hypot(pl[:, 0], pl[:, 1]) < win) & (np.abs(pl[:, 2]) < 0.5 * geom["brick_height"])
    pl, pw = pl[near], pw[near]
    if len(pl) < 50:
        out["why"] = f"only {len(pl)} depth points near the expected pair"
        return out
    rng = np.random.default_rng(seed)
    n, d, inl = _plane(pl, PLANE_BAND, rng)
    # When the stud tops won the plane, the brick top is a stud height below it.
    lower = np.abs(pl @ n + d + stud_h) < PLANE_BAND
    if lower.sum() > 0.25 * inl.sum():
        n, d, inl = _plane(pl[lower], PLANE_BAND, rng)
        inl = np.abs(pl @ n + d) < PLANE_BAND
    if inl.sum() < 30:
        out["why"] = "no brick-top plane near the expected pair"
        return out
    # Plane frame (in the prior frame): origin under the prior midpoint, x along the prior pair line.
    x = np.array([1.0, 0.0, 0.0]) - n[0] * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    o = -d * n
    q = np.stack([(pl - o) @ x, (pl - o) @ y, (pl - o) @ n], axis=1)
    cell = CELL_PITCHES * pitch
    size = int(math.ceil(2 * win / cell)) + 1
    ij = np.floor((q[:, :2] + win) / cell).astype(np.int64)
    good = (ij >= 0).all(axis=1) & (ij < size).all(axis=1)
    ij, hh = ij[good], q[good, 2]
    flat = ij[:, 1] * size + ij[:, 0]
    cnt = np.bincount(flat, minlength=size * size).reshape(size, size)
    hsum = np.bincount(flat, weights=hh, minlength=size * size).reshape(size, size)
    mean_h = np.where(cnt > 0, hsum / np.maximum(cnt, 1), np.nan)
    seen = cnt > 0
    top = seen & (mean_h > TOP_FRACTION * stud_h) & (mean_h < 1.6 * stud_h + PLANE_BAND)
    not_top = seen & ~top
    top_c = cv2.morphologyEx(top.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    top_c &= ~not_top | top
    nlab, lab, stats, cent = cv2.connectedComponentsWithStats(
        top_c.astype(np.uint8), connectivity=4
    )
    disc = math.pi * (r / cell) ** 2
    studs, arcs = [], []
    for i in range(1, nlab):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (0.3 * disc <= area <= 1.7 * disc):
            continue
        comp = lab == i
        # Rim edges: between a top cell and a seen non-top 4-neighbour; unseen neighbours say nothing.
        edges = []
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = np.zeros_like(comp)
            src = comp[max(0, -dr) : size - max(0, dr), max(0, -dc) : size - max(0, dc)]
            tgt = not_top[max(0, dr) : size - max(0, -dr), max(0, dc) : size - max(0, -dc)]
            nb[max(0, -dr) : size - max(0, dr), max(0, -dc) : size - max(0, dc)] = src & tgt
            rr, cc = np.nonzero(nb)
            edges += [
                ((c + 0.5 + dc / 2.0) * cell - win, (rr_ + 0.5 + dr / 2.0) * cell - win)
                for rr_, c in zip(rr, cc, strict=True)
            ]
        if len(edges) < MIN_RIM_POINTS:
            continue
        e = np.asarray(edges)
        c0 = (cent[i] + 0.5) * cell - win
        c, rms = _fit_circle(e, r, c0)
        arc = _arc_deg(e, c)
        if arc < MIN_RIM_ARC_DEG or rms > 0.35 * r or np.linalg.norm(c - c0) > r:
            continue
        studs.append(c)
        arcs.append(arc)
    out["cands"] = len(studs)
    t_prior_plane = np.eye(4)
    t_prior_plane[:3, :3] = np.column_stack([x, y, n])
    t_prior_plane[:3, 3] = o
    t_w_plane = prior @ t_prior_plane
    out["plane"] = t_w_plane
    if len(studs) < 2:
        out["why"] = f"{len(studs)} stud circle(s) found near the expected pair"
        return out
    s = np.asarray(studs)
    pairs = []
    for a in range(len(s)):
        for b in range(a + 1, len(s)):
            v = s[b] - s[a]
            sep = float(np.linalg.norm(v))
            if abs(sep - pitch) > PAIR_SPACING_TOL * pitch:
                continue
            yaw = _wrap(math.atan2(v[1], v[0]), math.pi)
            if abs(math.degrees(yaw)) > MAX_PAIR_YAW_DEG:
                continue
            mid = 0.5 * (s[a] + s[b])
            pairs.append((float(np.linalg.norm(mid)), a, b, mid, yaw))
    if not pairs:
        out["why"] = f"{len(s)} stud circles but no pair one pitch apart along the expected line"
        out["ambiguous"] = False
        return out
    pairs.sort(key=lambda p: p[0])
    d1, a, b, mid, yaw = pairs[0]
    d2 = pairs[1][0] if len(pairs) > 1 else math.inf
    out["margin_pitches"] = (d2 - d1) / pitch
    out["prior_error_pitches"] = d1 / pitch
    # Every stud on the pair's lattice (same top, nodes ((i + 1/2) p, j p) in the pair frame) refines
    # the pair's yaw and midpoint by a rigid 2D fit; the pair itself only says which node is which.
    c, sn = math.cos(yaw), math.sin(yaw)
    local = (s - mid) @ np.array([[c, -sn], [sn, c]])
    node = (
        np.stack([np.round(local[:, 0] / pitch - 0.5) + 0.5, np.round(local[:, 1] / pitch)], axis=1)
        * pitch
    )
    inl = np.linalg.norm(local - node, axis=1) < LATTICE_INLIER * pitch
    out["lattice_studs"] = int(inl.sum())
    if inl.sum() >= 3:
        g, o2 = node[inl], s[inl]
        gm, om = g.mean(axis=0), o2.mean(axis=0)
        hm = (g - gm).T @ (o2 - om)
        yaw = math.atan2(hm[0, 1] - hm[1, 0], hm[0, 0] + hm[1, 1])
        rr = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        mid = om - rr @ gm
    pair_plane = np.eye(4)
    pair_plane[:3, :3] = _axis_angle(np.array([0.0, 0.0, 1.0]), yaw)
    pair_plane[:2, 3] = mid
    pair = t_w_plane @ pair_plane
    out["pair"] = pair
    for c in (s[a], s[b]):
        out["studs"].append((t_w_plane @ np.array([c[0], c[1], stud_h, 1.0]))[:3])
    out["arc"] = [arcs[a], arcs[b]]
    if d1 > MAX_PRIOR_ERROR * pitch:
        out["why"] = (
            f"the nearest pair is {d1 / pitch:.2f} pitch from the expected one (over {MAX_PRIOR_ERROR})"
        )
        out["ambiguous"] = True
        return out
    if d2 - d1 < MIN_PAIR_MARGIN * pitch:
        out["why"] = f"two pairs are within {(d2 - d1) / pitch:.2f} pitch of each other's distance"
        out["ambiguous"] = True
        return out
    out["ok"] = True
    return out


def _pixel(frame: dict[str, Any], p: np.ndarray) -> list[float]:
    k = np.asarray(frame["intrinsics"], dtype=np.float64)
    pc = np.linalg.inv(_matrix(frame["pose"])) @ np.r_[p, 1.0]
    if pc[2] <= 1e-9:
        return []
    return [float(k[0, 0] * pc[0] / pc[2] + k[0, 2]), float(k[1, 1] * pc[1] / pc[2] + k[1, 2])]


def _ray_on_plane(
    frame: dict[str, Any], px: list[float], origin: np.ndarray, n: np.ndarray
) -> np.ndarray | None:
    k = np.asarray(frame["intrinsics"], dtype=np.float64)
    t = _matrix(frame["pose"])
    d = t[:3, :3] @ np.array([(px[0] - k[0, 2]) / k[0, 0], (px[1] - k[1, 2]) / k[1, 1], 1.0])
    den = float(d @ n)
    if abs(den) < 1e-6:
        return None
    s = float((origin - t[:3, 3]) @ n) / den
    return t[:3, 3] + s * d if s > 0 else None


# -- the loop -------------------------------------------------------------------------------


def _settled(ctx: NodeContext, arm_id: int) -> dict[str, Any]:
    obs = ctx.tool("robot.get_observation")
    for _ in range(SETTLE_READS):
        arms = obs.get("arms") or []
        if len(arms) <= arm_id or arms[arm_id].get("ee_pose") is None:
            return obs
        nxt = ctx.tool("robot.get_observation")
        a = _matrix(arms[arm_id]["ee_pose"])[:3, 3]
        nb = nxt.get("arms") or []
        obs = nxt
        if (
            len(nb) > arm_id
            and float(np.linalg.norm(_matrix(nb[arm_id]["ee_pose"])[:3, 3] - a)) <= SETTLE_DISTANCE
        ):
            break
    return obs


def run(
    ctx: NodeContext,
    pair_pose: dict[str, Any],
    system: str = "lego",
    stud_pitch: float | None = None,
    stud_height: float | None = None,
    stud_diameter: float | None = None,
    peek_height: float = PEEK_HEIGHT,
    yaw_in_pair_deg: float | None = None,
    tool_center_px: list[float] | None = None,
    tool_center_range: float | None = None,
    tolerance_mm: float = TOLERANCE * 1e3,
    tolerance_deg: float = TOLERANCE_DEG,
    max_iterations: int = MAX_ITERATIONS,
    gain: float = GAIN,
    move_to_peek: bool = True,
    camera: str | None = None,
    arm_id: int = 0,
    tolerance: float | None = None,
    max_steps: int | None = None,
    seed: int = 0,
) -> Output:
    if system not in PRESETS:
        raise ValueError(f"system must be one of {sorted(PRESETS)}, got {system!r}")
    if not pair_pose or "position" not in pair_pose:
        raise ValueError("pair_pose (the expected stud-pair midpoint at the brick top) is required")
    if not 0.0 < gain <= 1.5:
        raise ValueError(f"gain must be in (0, 1.5], got {gain}")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if tool_center_px is not None and len(tool_center_px) != 2:
        raise ValueError("tool_center_px is [u, v]")
    if tool_center_range is not None and tool_center_px is None:
        raise ValueError("tool_center_range needs tool_center_px")
    geom = dict(PRESETS[system])
    for key, val in (
        ("stud_pitch", stud_pitch),
        ("stud_height", stud_height),
        ("stud_diameter", stud_diameter),
    ):
        if val is not None:
            geom[key] = float(val)
    pitch = geom["stud_pitch"]
    tol, tol_yaw = float(tolerance_mm) * 1e-3, math.radians(float(tolerance_deg))
    arm = {"arm_id": int(arm_id)} if arm_id else {}
    move: dict[str, Any] = dict(arm)
    if tolerance is not None:
        move["tolerance"] = float(tolerance)
    if max_steps is not None:
        move["max_steps"] = int(max_steps)

    prior = _matrix(pair_pose)
    up = np.array([0.0, 0.0, 1.0])
    # The pair frame is kept level: z up, x along the pair line.
    xa = prior[:3, 0] - (prior[:3, 0] @ up) * up
    if np.linalg.norm(xa) < 1e-6:
        raise ValueError("pair_pose's x axis (the pair line) must not be vertical")
    xa /= np.linalg.norm(xa)
    prior[:3, :3] = np.column_stack([xa, np.cross(up, xa), up])
    out: Output = {
        "route": "lost_studs",
        "detail": "",
        "pair_pose": _se3(prior),
        "tool_pose": {},
        "offset_mm": [],
        "yaw_offset_deg": 0.0,
        "iterations": 0,
        "moves": 0,
        "history": [],
        "stud_centres": [],
        "pair_px": [],
        "pair_range_m": 0.0,
        "rim_arc_deg": [],
        "tool_axis_source": "tool_center_px" if tool_center_px is not None else "ee_pose",
        "camera_name": "",
        "system": system,
        "stud_pitch": pitch,
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        try:
            out["tool_pose"] = _se3(_matrix(ctx.tool("robot.get_ee_pose", **arm)))
        except Exception as exc:  # the result stands without it
            logger.info("correcting-brick-offset: no final tool pose (%s)", exc)
        logger.info("correcting-brick-offset-from-stud-circles: %s (%s)", route, why)
        return out

    def go(target: np.ndarray, what: str) -> str:
        try:
            why = _refusal(ctx.tool("robot.go_to_pose", pose=_se3(target), **move))
        except Exception as exc:
            why = str(exc) or type(exc).__name__
        out["moves"] += 1
        return f"{what}: {why}" if why else ""

    tool = _matrix(ctx.tool("robot.get_ee_pose", **arm))
    yaw_tool = math.atan2(float(tool[:3, 0] @ prior[:3, 1]), float(tool[:3, 0] @ prior[:3, 0]))
    if yaw_in_pair_deg is not None:
        rel = math.radians(float(yaw_in_pair_deg))
    else:  # the hand's x along or across the pair line, whichever is nearer now
        rel = round(yaw_tool / (math.pi / 2)) * (math.pi / 2)
    peek_z = geom["stud_height"] + float(peek_height)

    def target_for(pair: np.ndarray, tool_now: np.ndarray, axis_now: np.ndarray) -> np.ndarray:
        """Tool pose putting the tool axis over the pair midpoint at the peek height, at yaw rel."""
        yaw_now = math.atan2(
            float(tool_now[:3, 0] @ pair[:3, 1]), float(tool_now[:3, 0] @ pair[:3, 0])
        )
        r_turn = _axis_angle(up, _wrap(rel - yaw_now, math.pi))
        lever = axis_now - tool_now[:3, 3]
        lever -= (lever @ up) * up
        t = np.eye(4)
        t[:3, :3] = r_turn @ tool_now[:3, :3]
        t[:3, 3] = pair[:3, 3] - r_turn @ lever
        t[:3, 3] += (float((pair[:3, 3] - t[:3, 3]) @ up) + peek_z) * up
        return t

    if move_to_peek:
        want = target_for(prior, tool, tool[:3, 3])
        if float(np.linalg.norm(want[:3, 3] - tool[:3, 3])) > 0.001 or abs(
            rel - yaw_tool
        ) > math.radians(0.5):
            if tool[2, 3] < want[2, 3] - 0.001:
                rise = tool.copy()
                rise[2, 3] = want[2, 3]
                why = go(rise, "rise to the peek height")
                if why:
                    return finish("unreachable", why)
            why = go(want, "peek pose")
            if why:
                return finish("unreachable", why)

    for it in range(int(max_iterations)):
        obs = _settled(ctx, int(arm_id))
        frame = _wrist_frame(obs, int(arm_id), camera)
        out["iterations"] = it + 1
        if frame is None:
            return finish("lost_studs", "no posed wrist RGB-D frame in the observation")
        out["camera_name"] = str(frame.get("name", ""))
        arms = obs.get("arms") or []
        tool = (
            _matrix(arms[arm_id]["ee_pose"])
            if len(arms) > arm_id and arms[arm_id].get("ee_pose") is not None
            else _matrix(ctx.tool("robot.get_ee_pose", **arm))
        )
        m = measure(frame, prior, geom, seed=seed + it)
        out["stud_centres"] = [[float(v) for v in s] for s in m["studs"]]
        out["rim_arc_deg"] = [round(float(a), 1) for a in m["arc"]]
        if not m["ok"]:
            if m.get("ambiguous"):
                return finish("ambiguous_stud_pair", m["why"])
            return finish("lost_studs", m["why"])
        pair = m["pair"]
        top = pair[:3, 3] + geom["stud_height"] * pair[:3, 2]
        cam = _matrix(frame["pose"])[:3, 3]
        out["pair_px"] = _pixel(frame, top)
        out["pair_range_m"] = round(float(np.linalg.norm(top - cam)), 6)
        dh = 0.0
        if tool_center_px is not None:
            axis = _ray_on_plane(frame, list(tool_center_px), top, pair[:3, 2])
            if axis is None:
                return finish("lost_studs", "tool_center_px's ray does not meet the stud-top plane")
            if tool_center_range is not None:
                # Hold the camera's range to the studs along the pixel's ray (camera-frame depth, free
                # of hand-eye error) at the calibrated one: the ray meets the tool axis only there.
                ray = (axis - cam) / np.linalg.norm(axis - cam)
                dh = -(float(np.linalg.norm(axis - cam)) - float(tool_center_range)) * abs(
                    float(ray @ up)
                )
        else:
            # Dropped vertically: the fitted plane's normal is good to ~0.3 deg, 0.2 mm at the peek height.
            axis = tool[:3, 3] - float((tool[:3, 3] - top) @ up) * up
        off = top - axis
        x_level = pair[:3, 0] - (pair[:3, 0] @ up) * up
        x_level /= np.linalg.norm(x_level)
        off_pair = [float(off @ x_level), float(off @ np.cross(up, x_level))]
        yaw_now = math.atan2(float(tool[:3, 0] @ pair[:3, 1]), float(tool[:3, 0] @ pair[:3, 0]))
        dyaw = _wrap(rel - yaw_now, math.pi)
        out["pair_pose"] = _se3(pair)
        out["offset_mm"] = [round(v * 1e3, 3) for v in off_pair]
        out["yaw_offset_deg"] = round(math.degrees(dyaw), 3)
        lateral = math.hypot(*off_pair)
        out["history"].append(
            {
                "offset_mm": out["offset_mm"],
                "yaw_offset_deg": out["yaw_offset_deg"],
                "prior_error_pitches": round(float(m.get("prior_error_pitches", 0.0)), 3),
                "studs_found": int(m["cands"]),
            }
        )
        prior = pair
        if lateral <= tol and abs(dyaw) <= tol_yaw and abs(dh) <= tol:
            return finish(
                "aligned",
                f"tool axis {lateral * 1e3:.2f} mm and {math.degrees(dyaw):.2f} deg from the pair after "
                f"{out['moves']} move(s)",
            )
        if it == int(max_iterations) - 1:
            break
        want = target_for(pair, tool, axis)
        if tool_center_range is not None:
            want[:3, 3] += (float((tool[:3, 3] - want[:3, 3]) @ up) + dh) * up
        step = want[:3, 3] - tool[:3, 3]
        lat_step = step - (step @ up) * up
        scale = float(gain)
        if np.linalg.norm(lat_step) * scale > MAX_STEP_PITCHES * pitch:
            scale = MAX_STEP_PITCHES * pitch / float(np.linalg.norm(lat_step))
        turn = _wrap(rel - yaw_now, math.pi) * float(gain)
        turn = max(-math.radians(MAX_TURN_DEG), min(math.radians(MAX_TURN_DEG), turn))
        nxt = np.eye(4)
        nxt[:3, :3] = _axis_angle(up, turn) @ tool[:3, :3]
        nxt[:3, 3] = tool[:3, 3] + scale * lat_step + (step @ up) * up
        why = go(nxt, f"correction {it + 1}")
        if why:
            return finish("unreachable", why)
    return finish(
        "max_iterations",
        f"still {math.hypot(*out['offset_mm']):.2f} mm and {out['yaw_offset_deg']:.2f} deg off after "
        f"{out['iterations']} measurement(s)",
    )
