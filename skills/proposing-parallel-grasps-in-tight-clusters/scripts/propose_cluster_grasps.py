# Adapted from arc-robot-vision (https://github.com/andyzeng/arc-robot-vision),
# parallel-jaw-grasping/baseline/predict.m (commit 728139e9ad0e003106c71a7c0d0a2fc9d8430644),
# Apache-2.0, Andy Zeng et al. (MIT-Princeton, Amazon Robotics Challenge 2017).
# Kept: brute-force search over grasp centres, 8 jaw rotations 22.5 degrees apart and growing
# jaw widths; two finger footprints (width across the jaw motion, thickness along it) beside
# the centre; a grasp is valid when the 90th percentile of the heights under both fingers lies
# a set depth below the part's median top; the smallest valid width per centre and rotation;
# the confidence as the fraction of cells between the fingers above the finger floor.
# Changed: numpy instead of MATLAB; a world-frame heightmap built from any posed depth camera
# (wrist or overhead) at 0.5-2 mm cells instead of a fixed 2 mm bin heightmap; candidate
# centres inside each part's mask (or height blob) instead of 10 x 10 px windows; the 2 cm
# depth below the top becomes half the part's height (1-20 mm) for 2-12 mm parts; fingers are
# also checked at the opening the hand descends with (contact width + OPEN_MARGIN per side);
# no other part may lie between the jaws; the finger body above the tip must clear tall
# neighbours; the part's minimum-area-rectangle axes join the 8 rotations; the plates must
# meet an edge of the part, not a corner; the support is a fitted plane and the thresholds
# scale with its depth noise; grasps are ranked by finger clearance, centring, the upstream
# confidence and narrowness instead of the confidence alone; world grasp poses through
# robot.grasp_frame; the flush (bin wall) grasps are not ported.
"""Propose top-down parallel-jaw grasps on small flat parts packed close together, from one depth frame."""

from __future__ import annotations

import math
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext
from scipy import ndimage as ndi

FINGER_THICKNESS = 0.010
"""Fingertip size along the jaw motion [m]. An estimate of the tapered linear_4310 tips (the
same as singulating-by-pushing's); robot.describe_gripper states `pad_thickness_m: 0`
(unmeasured). Upstream: 0.036 m."""

FINGER_WIDTH = 0.020
"""Fingertip size across the jaw motion [m]; estimate, unmeasured. Upstream: 0.06 m."""

TIP_REACH = 0.0102
"""Fingertips past the tool centre [m] when describe_gripper states none (real YAM station
measurement)."""

MAX_GRASP_WIDTH = 0.086
"""Widest opening a grasp may use [m] when describe_gripper states none (a real station's
`max_grasp_width_m`: the 94 mm span less 8 mm)."""

ROTATIONS = 8
"""Jaw headings over 180 degrees (upstream: 0 to 157.5 in 22.5 degree steps)."""

FINGER_PERCENTILE = 90.0
"""Upstream's percentile of the heights under the fingers: up to 10 % of the footprint may sit
on something taller (noise, a mask edge, a knob's rim)."""

GRIP_FRACTION = 0.5
"""How far the fingertips must reach below the part's top, as a fraction of its height; new
(upstream asks a fixed 2 cm of 2-10 cm tall items)."""

MIN_INSERTION = 0.001
"""Least depth below the part's top the fingertips must reach [m]; about twice a D405's depth
noise at 0.15-0.2 m. A 2 mm jigsaw piece needs its neighbours' gaps open down to 1 mm."""

MAX_INSERTION = 0.02
"""Upstream's depth below the top (its 2 cm), now the cap."""

OPEN_MARGIN = 0.002
"""Extra opening per side when the hand comes down [m]; the fingers are checked there too."""

MIN_PART_HEIGHT = 0.0012
"""A cell this far over the support is part of something [m]; a 2 mm jigsaw piece clears it."""

MAX_PART_HEIGHT = 0.06
"""Points higher over the support are dropped (own fingertips in a wrist view, the arm) [m];
their cells read unknown."""

NOISE_STANDS = 3.0
"""A cell stands over the support at max(min_part_height, 3 sigma) of the support's depth noise
(sigma from the spread of the heights below the support)."""

NOISE_INSERTION = 2.0
"""The fingertips reach at least 2 sigma below the part's top, so table noise under them (90th
percentile 1.3 sigma) does not read as a neighbour."""

PLANE_MIN_SPAN = 0.03
"""The support is fitted as a plane only over at least this much table each way [m]."""

PLANE_MAX_SLOPE = 0.2
"""A steeper fitted support (11 degrees) is refused for a flat one."""

MIN_AREA = 2.0e-5
"""Least top area of a part [m^2] (20 mm^2); smaller height blobs are noise."""

BETWEEN_AREA = 4.0e-6
"""Largest area of other parts between the open jaws [m^2] (2 x 2 mm of mask noise)."""

CONTACT_DEPTH = 0.001
"""A part's edge within this of its outermost point along the jaw touches the plate first [m]
(at least two heightmap cells)."""

CONTACT_RATIO = 4.0
"""The touching edge must span CONTACT_RATIO x CONTACT_DEPTH across the jaw, each side: an
edge within atan(1/4) = 14 degrees of the plate (inside a 0.3-0.5 friction cone, 17-27
degrees), a jigsaw knob's round head (r >= 2.6 mm gives >= 4.1 mm at 1 mm), not a corner (a
rectangle met at 22.5 degrees gives 2.8 mm, a 45 degree wedge 2 mm). New: upstream has no
contact test."""

APPROACH_MARGIN = 0.001
"""The finger body's footprint is grown this much each way for the approach check [m]."""

CLEARANCE_REF = 0.003
"""Finger clearance that earns full marks [m] (singulating-by-pushing's FIT_MARGIN)."""

SHADOW = 0.003
"""Cells without depth this close to the part are its own occlusion shadow [m]: not obstacles
for the clearance (they still are under a finger)."""

CLEARANCE_WINDOW = 0.006
"""Clearance is searched this far round each finger [m]; more reads as this."""

TIP_LIFT = 0.0005
"""Fingertips are set this far over the floor under them [m]."""

WEIGHTS = (0.35, 0.35, 0.15, 0.15)
"""Score = clearance, centring, upstream confidence (fraction between the jaws over the floor),
narrowness (1 - opening / widest: the short side first, as proposing-short-axis-grasps)."""

UNKNOWN_HIGH = 1.0
"""Height given to cells without depth under the fingers [m] (finite, so percentiles hold)."""

MAX_CENTRES = 9
"""Candidate centres per part, nearest its centroid first."""

MAX_GRID = 0.5
"""Largest heightmap side [m] without a mask or roi; centred on what stands on the support."""


class Output(TypedDict):
    route: str
    reason: str
    grasps: list[dict[str, Any]]
    position: dict[str, float] | None
    close_heading_deg: float
    jaw_width_m: float
    object_width_m: float
    tip_z: float
    tool_z: float
    clearance_m: float
    score: float
    grasp_pose: dict[str, Any] | None
    rotation_flipped: dict[str, float] | None
    pieces: list[dict[str, Any]]
    n_pieces: int
    n_graspable: int
    fraction_graspable: float
    support_z: float
    noise_m: float
    voxel_m: float


def _rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in "wxyz")
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix(pose: Any) -> np.ndarray:
    if isinstance(pose, dict) and "pose" in pose and "position" not in pose:
        pose = pose["pose"]
    if isinstance(pose, dict):
        m = np.eye(4)
        m[:3, :3] = _rot(
            pose.get("rotation") or pose.get("orientation") or {"w": 1, "x": 0, "y": 0, "z": 0}
        )
        p = pose["position"]
        m[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
        return m
    m = np.asarray(pose, dtype=float)
    if m.shape != (4, 4):
        raise ValueError("camera pose must be an Se3Pose {position, rotation} or a 4x4 matrix")
    return m


def _as_bool(mask: Any) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    return m > (127 if m.dtype == np.uint8 else 0)


def _camera(ctx: NodeContext, frame: dict | None, name: str | None) -> dict:
    if frame is not None:
        return frame
    obs = ctx.tool("robot.get_observation")
    cams = obs.get("cameras") if isinstance(obs, dict) else getattr(obs, "cameras", None)
    if isinstance(cams, dict):
        cams = [dict(v, name=v.get("name", k)) for k, v in cams.items()]
    cams = [c for c in (cams or []) if c.get("depth") is not None and c.get("pose") is not None]
    if name:
        cam = next((c for c in cams if str(c.get("name")) == name), None)
    else:
        cam = next(
            (c for c in cams if any(k in str(c.get("name", "")) for k in ("wrist", "eye_in_hand"))),
            cams[0] if cams else None,
        )
    if cam is None:
        raise ValueError(f"no posed depth camera {name or '(any)'!r} in the observation")
    return cam


def _support(z: np.ndarray) -> float:
    """The lowest well-populated 0.5 mm height band (the table under a cluster)."""
    lo, hi = np.percentile(z, [0.5, 99.5])
    if hi - lo < 1e-4:
        return float(np.median(z))
    edges = np.arange(lo, hi + 5e-4, 5e-4)
    hist, edges = np.histogram(z, bins=edges)
    hist = np.convolve(hist, np.ones(3), mode="same")
    i = int(np.argmax(hist >= 0.15 * hist.max()))
    band = z[(z >= edges[i] - 1e-3) & (z <= edges[i + 1] + 1e-3)]
    return float(np.median(band)) if band.size else float(edges[i])


def _support_plane(pool: np.ndarray, fixed: float | None) -> tuple[np.ndarray, float]:
    """The support as a plane z = a x + b y + c (a tilted camera calibration tilts the table) and
    its depth noise sigma, fitted to the lowest well-populated band; flat at ``fixed`` if given."""
    z = pool[:, 2]
    coef = np.array([0.0, 0.0, _support(z) if fixed is None else float(fixed)])
    sigma = 0.001
    a_mat = np.column_stack([pool[:, 0], pool[:, 1], np.ones(len(pool))])
    span = min(np.ptp(pool[:, 0]), np.ptp(pool[:, 1])) if len(pool) else 0.0
    for _ in range(4):
        r = z - a_mat @ coef
        below = -r[(r <= 0) & (r > -0.01)]
        if below.size >= 20:
            sigma = max(float(np.median(below) / 0.6745), 1e-5)
        if fixed is not None or span < PLANE_MIN_SPAN:
            continue
        inl = (r > -max(0.005, 4 * sigma)) & (r < max(0.0008, 2.5 * sigma))
        if inl.sum() < 50:
            break
        fit = np.linalg.lstsq(a_mat[inl], z[inl], rcond=None)[0]
        if math.hypot(fit[0], fit[1]) > PLANE_MAX_SLOPE:
            break
        coef = fit
    return coef, sigma


def _wxyz(q: Any) -> dict[str, float] | None:
    return {k: float(q[k]) for k in "wxyz"} if isinstance(q, dict) else None


class _Map:
    """A world-frame heightmap: cell (iy, ix) covers x0 + ix*voxel .. +voxel, y0 + iy*voxel .."""

    def __init__(self, pts, labels, x0, y0, nx, ny, voxel, support_at):
        self.x0, self.y0, self.voxel, self.support_at = x0, y0, voxel, support_at
        ix = np.floor((pts[:, 0] - x0) / voxel).astype(int)
        iy = np.floor((pts[:, 1] - y0) / voxel).astype(int)
        keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy, z, lab = ix[keep], iy[keep], pts[keep, 2], labels[keep]
        order = np.argsort(z)  # the highest point of a cell wins, as in a heightmap
        h = np.full((ny, nx), np.nan)
        lb = np.zeros((ny, nx), dtype=np.int32)
        h[iy[order], ix[order]] = z[order]
        lb[iy[order], ix[order]] = lab[order]
        for _ in range(2):  # close pinholes left by sparse projection with their neighbours' median
            pad_h = np.pad(h, 1, constant_values=np.nan)
            pad_l = np.pad(lb, 1)
            shifts = [(dy, dx) for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)]
            nh = np.stack([pad_h[dy : dy + ny, dx : dx + nx] for dy, dx in shifts])
            nl = np.stack([pad_l[dy : dy + ny, dx : dx + nx] for dy, dx in shifts])
            hole = ~np.isfinite(h) & (np.isfinite(nh).sum(axis=0) >= 5)
            if not hole.any():
                break
            med = np.nanmedian(nh[:, hole], axis=0)
            pick = np.nanargmin(np.abs(nh[:, hole] - med), axis=0)
            h[hole] = med
            lb[hole] = nl[:, hole][pick, np.arange(hole.sum())]
        self.h, self.labels = h, lb

    def centre(self, iy, ix):
        return self.x0 + (ix + 0.5) * self.voxel, self.y0 + (iy + 0.5) * self.voxel

    def sample(self, x, y, extra=None):
        ix = np.floor((x - self.x0) / self.voxel).astype(int)
        iy = np.floor((y - self.y0) / self.voxel).astype(int)
        ny, nx = self.h.shape
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ixc, iyc = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)
        h = np.where(inside, self.h[iyc, ixc], np.nan)
        lab = np.where(inside, self.labels[iyc, ixc], 0)
        more = np.where(inside, extra[iyc, ixc], False) if extra is not None else None
        return h, lab, more


def _join_unmasked(m: _Map, min_h: float) -> None:
    """Unmasked standing cells connected to exactly one masked part join it (depth smear at its
    edges, a mask a few pixels short of the part); touching none or several, they stay
    anonymous neighbours."""
    stand = np.nan_to_num(m.h, nan=-1.0) > min_h
    comp, n = ndi.label(stand & (m.labels == 0), np.ones((3, 3)))
    if not n:
        return
    near = ndi.grey_dilation(np.where(stand, m.labels, 0), size=(3, 3))
    near_min = -ndi.grey_dilation(
        np.where(stand & (m.labels > 0), -m.labels, -(10**9)), size=(3, 3)
    )
    for i, sl in enumerate(ndi.find_objects(comp), start=1):
        sl = tuple(slice(max(a.start - 1, 0), a.stop + 1) for a in sl)
        here = comp[sl] == i
        hi = near[sl][here]
        lo = near_min[sl][here]
        touching = hi[hi > 0]
        if touching.size and touching.max() == lo[lo < 10**9].min():
            m.labels[sl][here] = int(touching.max())


def _search_piece(m: _Map, lab: int, p: dict[str, Any]) -> tuple[list[dict], dict[str, int]]:
    """Upstream's brute-force finger fit around one part: every centre, heading and width."""
    v = m.voxel
    h, labels = m.h, m.labels
    own = (labels == lab) & (h > p["min_part_height"])
    iy, ix = np.nonzero(own)
    xs, ys = m.centre(iy, ix)
    part_top = float(np.median(h[own]))
    insertion = float(np.clip(GRIP_FRACTION * part_top, p["min_insertion"], MAX_INSERTION))
    floor_limit = part_top - insertion
    near_own = ndi.distance_transform_edt(~own) * v <= SHADOW
    shadow = near_own & ~np.isfinite(h)
    top = np.column_stack([xs, ys])
    cxy = top.mean(axis=0)
    rel = top - cxy
    (_, _), (rw, rh), ang = cv2.minAreaRect(rel.astype(np.float32))
    a = math.radians(ang)
    ax1, ax2 = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    minor, major = (ax1, ax2) if rw <= rh else (ax2, ax1)
    ext_minor = float(np.ptp(rel @ minor)) + v
    ext_major = float(np.ptp(rel @ major)) + v

    thetas = [math.pi * k / p["rotations"] for k in range(p["rotations"])]
    for axis in (minor, major):  # the part's own axes: across and along it
        t = math.atan2(axis[1], axis[0]) % math.pi
        if all(
            abs((t - u + math.pi / 2) % math.pi - math.pi / 2) > math.radians(5) for u in thetas
        ):
            thetas.append(t)

    stride = max(0.002, 0.25 * ext_minor)
    dist = np.hypot(*rel.T)
    centres: list[np.ndarray] = []
    for i in np.argsort(dist):
        q = np.array([xs[i], ys[i]])
        if all(np.hypot(*(q - c)) >= stride - 1e-9 for c in centres):
            centres.append(q)
            if len(centres) >= MAX_CENTRES:
                break

    t_tip = max(1, round(p["tip_thickness"] / v))
    t_body = max(t_tip, round(p["finger_thickness"] / v))
    half_w = 0.5 * p["finger_width"]
    clear_cells = round(CLEARANCE_WINDOW / v)
    m_open = max(0, round(OPEN_MARGIN / v))
    k_hi = int(math.floor(0.5 * p["max_width"] / v)) - m_open
    J = k_hi + m_open + t_body + clear_cells + 2
    u = (np.arange(2 * J + 1) - J) * v
    v_rows = np.arange(
        -(half_w + APPROACH_MARGIN + CLEARANCE_WINDOW),
        half_w + APPROACH_MARGIN + CLEARANCE_WINDOW + 1e-9,
        v,
    )
    tip_rows = np.abs(v_rows) <= half_w + 1e-9
    body_rows = np.abs(v_rows) <= half_w + APPROACH_MARGIN + 1e-9
    unknown_h = UNKNOWN_HIGH if p["unknown_is_obstacle"] else -UNKNOWN_HIGH

    fails = {"collision": 0, "neighbour_between": 0, "too_wide": 0, "poor_contact": 0}
    grasps = []
    for c in centres:
        for theta in thetas:
            d = np.array([math.cos(theta), math.sin(theta)])
            n = np.array([-d[1], d[0]])
            gx = c[0] + u[None, :] * d[0] + v_rows[:, None] * n[0]
            gy = c[1] + u[None, :] * d[1] + v_rows[:, None] * n[1]
            hb, lb, shade = m.sample(gx, gy, shadow)
            unknown = ~np.isfinite(hb)
            hb_obst = np.where(unknown, unknown_h, hb)
            own_b = (lb == lab) & (hb > p["min_part_height"])
            other_b = (lb != lab) & ~unknown & (hb > p["min_part_height"])

            own_cols = np.nonzero(own_b[tip_rows].any(axis=0))[0]
            if own_cols.size == 0:
                continue
            umin, umax = u[own_cols.min()], u[own_cols.max()]
            flat, smin, smax = _contacts(top - c, d, n, half_w, 0.5 * p["max_width"], v)
            if not flat:
                fails["poor_contact"] += 1
                continue
            k_need = int(math.ceil(max(-umin, umax) / v + 0.5)) - 1
            if k_need > k_hi:
                fails["too_wide"] += 1
                continue
            k_lo = max(1, k_need)
            ks = np.arange(k_lo, k_hi + m_open + 1)
            tip = hb_obst[tip_rows]
            body = hb_obst[body_rows]
            # windows of finger columns for each half-opening k: right J+k+1.., left ..J-k-1
            r_idx = J + ks[:, None] + 1 + np.arange(t_tip)[None, :]
            l_idx = J - ks[:, None] - 1 - np.arange(t_tip)[None, :]
            tip_cells = np.concatenate([tip[:, r_idx], tip[:, l_idx]], axis=2)  # rows x K x 2t
            tip_cells = np.moveaxis(tip_cells, 1, 0).reshape(len(ks), -1)
            floor = np.percentile(tip_cells, FINGER_PERCENTILE, axis=1)
            fit = floor <= floor_limit
            if not fit.any():
                fails["collision"] += 1
                continue
            rb_idx = J + ks[:, None] + 1 + np.arange(t_body)[None, :]
            lb_idx = J - ks[:, None] - 1 - np.arange(t_body)[None, :]
            body_cells = np.concatenate([body[:, rb_idx], body[:, lb_idx]], axis=2)
            body_cells = np.moveaxis(body_cells, 1, 0).reshape(len(ks), -1)
            body_top = np.percentile(body_cells, FINGER_PERCENTILE, axis=1)
            body_ok = body_top <= np.maximum(floor, 0.0) + p["tip_reach"]
            col_other = np.concatenate([[0], np.cumsum(other_b[tip_rows].sum(axis=0))])
            n_between = (col_other[J + ks + 1] - col_other[J - ks]).astype(float)
            between_ok = n_between * v * v <= BETWEEN_AREA

            n_k = len(ks) - m_open
            if n_k <= 0:
                fails["too_wide"] += 1
                continue
            ok = (
                fit[:n_k]
                & fit[m_open : m_open + n_k]
                & body_ok[m_open : m_open + n_k]
                & between_ok[m_open : m_open + n_k]
            )
            if not ok.any():
                if (fit[:n_k] & fit[m_open : m_open + n_k] & body_ok[m_open : m_open + n_k]).any():
                    fails["neighbour_between"] += 1
                else:
                    fails["collision"] += 1
                continue
            i = int(np.argmax(ok))  # the smallest valid width, as upstream
            k, kd = int(ks[i]), int(ks[i + m_open])
            tip_floor = max(float(floor[i + m_open]), 0.0)

            between = hb[tip_rows][:, J - k : J + k + 1]
            between = between[np.isfinite(between)]
            coverage = float((between > floor[i]).mean()) if between.size else 0.0

            # clearance: nearest obstacle round the open fingers, outside the jaws (between
            # them is only this part and its occlusion shadow, others were refused above)
            uu, vv = np.meshgrid(u, v_rows)
            inside = (np.abs(uu) < (kd + 0.5) * v) & (np.abs(vv) <= half_w)
            obst = ~own_b & ~inside & ~shade & (hb_obst > tip_floor + max(5e-4, 0.5 * insertion))
            clr = CLEARANCE_WINDOW
            for side in (1, -1):
                u_in = side * (kd + 0.5) * v
                u_mid = u_in + side * 0.5 * t_body * v
                du = np.maximum(np.abs(uu - u_mid) - 0.5 * t_body * v, 0.0)
                dv = np.maximum(np.abs(vv) - half_w, 0.0)
                dd = np.hypot(du, dv)[obst]
                if dd.size:
                    clr = min(clr, float(dd.min()))
            centroid_off = float(abs((cxy - c) @ n))
            mid_off = float(abs(0.5 * (smin + smax)))
            centring = math.hypot(centroid_off, mid_off)
            centring_ref = max(0.003, 0.25 * ext_minor)
            w_c, w_k, w_q, w_n = WEIGHTS
            score = (
                w_c * min(clr / CLEARANCE_REF, 1.0)
                + w_k * (1.0 - min(centring / centring_ref, 1.0))
                + w_q * coverage
                + w_n * (1.0 - (2 * kd + 1) * v / p["max_width"])
            )
            heading = math.degrees(theta)
            heading = (heading + 90.0) % 180.0 - 90.0
            tip_z = m.support_at(*c) + tip_floor + TIP_LIFT
            grasps.append(
                {
                    "piece": lab,
                    "xy": c.copy(),
                    "theta": theta,
                    "close_heading_deg": heading,
                    "jaw_width_m": (2 * kd + 1) * v,
                    "contact_width_m": (2 * k + 1) * v,
                    "object_width_m": smax - smin,
                    "part_height_m": part_top,
                    "insertion_m": insertion,
                    "finger_floor_m": float(floor[i + m_open]),
                    "tip_z": tip_z,
                    "clearance_m": clr,
                    "centring_m": centring,
                    "coverage": coverage,
                    "score": float(score),
                    "ext_minor": ext_minor,
                    "ext_major": ext_major,
                    "along_m": float(np.ptp((top - c) @ n) + v),
                }
            )
    return grasps, fails


def _contacts(
    rel: np.ndarray, d: np.ndarray, n: np.ndarray, half_w: float, reach: float, v: float
) -> tuple[bool, float, float]:
    """Whether both plates meet an edge of the part, not a corner (see CONTACT_RATIO), and the
    part's extent along the jaw between them. ``rel``: the part's top cells relative to the
    grasp centre, measured on cell centres (not resampled); only cells across the fingers'
    width and within ``reach`` along the jaw (the widest opening's half) count."""
    depth = max(CONTACT_DEPTH, 2.0 * v)
    need = CONTACT_RATIO * depth
    t = rel @ n
    band = (np.abs(t) <= half_w) & (np.abs(rel @ d) <= reach)
    if band.sum() < 3:
        return False, 0.0, 0.0
    s, t = rel[band] @ d, t[band]
    ok = True
    for prof in (s, -s):
        touch = t[prof >= prof.max() - depth]
        if np.ptp(touch) + v < need - 1e-9:
            ok = False
    return ok, float(s.min() - 0.5 * v), float(s.max() + 0.5 * v)


def _distinct(grasps: list[dict], n: int) -> list[dict]:
    out: list[dict] = []
    for g in sorted(grasps, key=lambda g: -g["score"]):
        if all(
            np.hypot(*(g["xy"] - o["xy"])) > 0.003
            or abs((g["theta"] - o["theta"] + math.pi / 2) % math.pi - math.pi / 2)
            > math.radians(20)
            for o in out
        ):
            out.append(g)
            if len(out) >= n:
                break
    return out


def _quat_from_axes(x_axis: np.ndarray) -> dict[str, float]:
    a = math.atan2(x_axis[1], x_axis[0])
    return {"w": math.cos(a / 2), "x": 0.0, "y": 0.0, "z": math.sin(a / 2)}


def _public(g: dict, support_at: Any, reach: float, to_px: Any) -> dict[str, Any]:
    d = np.array([math.cos(g["theta"]), math.sin(g["theta"])])
    support = support_at(*g["xy"])
    mid_z = support + 0.5 * g["part_height_m"]
    out = {
        "piece": int(g["piece"]),
        "position": {"x": float(g["xy"][0]), "y": float(g["xy"][1]), "z": float(mid_z)},
        "close_heading_deg": float(g["close_heading_deg"]),
        "jaw_width_m": float(g["jaw_width_m"]),
        "contact_width_m": float(g["contact_width_m"]),
        "object_width_m": float(g["object_width_m"]),
        "part_height_m": float(g["part_height_m"]),
        "support_z": float(support),
        "tip_z": float(g["tip_z"]),
        "tool_z": float(g["tip_z"] + reach),
        "clearance_m": float(g["clearance_m"]),
        "centring_m": float(g["centring_m"]),
        "coverage": float(g["coverage"]),
        "score": float(g["score"]),
        "closing_is_short_axis": bool(g["object_width_m"] <= g["along_m"] + 1e-9),
        "obb": {
            "center": {"x": float(g["xy"][0]), "y": float(g["xy"][1]), "z": float(mid_z)},
            "extent": {
                "x": 0.5 * float(g["object_width_m"]),
                "y": 0.5 * float(g["along_m"]),
                "z": 0.5 * float(g["part_height_m"]),
            },
            "orientation": _quat_from_axes(d),
        },
    }
    out["center_px"] = to_px(np.array([g["xy"][0], g["xy"][1], mid_z]))
    return out


def run(
    ctx: NodeContext,
    mask: Any = None,
    piece_masks: list | None = None,
    camera: str | None = None,
    camera_frame: dict | None = None,
    ignore_mask: Any = None,
    support_z: float | None = None,
    roi: dict[str, float] | None = None,
    finger_thickness: float = 0.0,
    finger_width: float = 0.0,
    tip_thickness: float = 0.0,
    max_width: float = 0.0,
    voxel: float = 0.0,
    rotations: int = ROTATIONS,
    min_part_height: float = MIN_PART_HEIGHT,
    min_insertion: float = MIN_INSERTION,
    unknown_is_obstacle: bool = True,
    per_piece: int = 3,
    max_pieces: int = 60,
    frame: bool = True,
    arm_id: int = 0,
) -> Output:
    """Ranked top-down parallel-jaw grasps on small parts in a tight cluster.

    Args:
        mask: The target part's mask in the camera image (uint8 0/255 or bool). Given, only
            it gets grasps and every other part is a neighbour; omitted, every part does.
        piece_masks: Masks of the other parts (optional; a list, so wired from a script
            node). Without them parts are told apart by height, and touching parts merge.
        camera / camera_frame: Which observation camera to read (default: the wrist one,
            else the first posed depth camera), or a CameraFrame {depth, intrinsics, pose}.
        ignore_mask: Pixels to drop (own fingertips in a wrist view).
        support_z: The surface the parts lie on [m, world]; default the lowest
            well-populated height band of the depth (round the target when masked).
        roi: {x_min, x_max, y_min, y_max} [m, world] to search; default round the target, or
            what stands on the support (at most MAX_GRID a side).
        finger_thickness, finger_width: Fingertip footprint [m] along and across the jaw
            motion (0 = FINGER_THICKNESS, FINGER_WIDTH: estimates). tip_thickness: the tip's
            thickness at its bottom when tapered (0 = finger_thickness).
        max_width: Widest jaw opening [m] (0 = describe_gripper's max_grasp_width_m).
        voxel: Heightmap cell [m] (0 = the camera's footprint per pixel, 0.5-2 mm).
        rotations: Jaw headings over 180 degrees (upstream 8).
        unknown_is_obstacle: Cells without depth count as obstacles under the fingers.
        per_piece: Grasps kept per part. frame: call robot.grasp_frame for the best one.

    Returns:
        ``route`` "proposed" (``grasps`` best first; the best's fields flattened),
        "too_crowded" (parts seen, no finger fit beside them) or "nothing_graspable".
    """
    hand: dict[str, Any] = {}
    on_arm = {"arm_id": arm_id} if arm_id else {}
    try:
        hand = ctx.tool("robot.describe_gripper", **on_arm) or {}
    except Exception:  # noqa: BLE001 - the hand's numbers are optional
        hand = {}
    finger = hand.get("finger") or {}
    reach = float(finger.get("reach_m") or TIP_REACH) if finger.get("stated", True) else TIP_REACH
    widest = float(
        max_width or hand.get("max_grasp_width_m") or hand.get("span_m") or MAX_GRASP_WIDTH
    )
    params = {
        "finger_thickness": float(finger_thickness or FINGER_THICKNESS),
        "finger_width": float(finger_width or FINGER_WIDTH),
        "tip_thickness": float(tip_thickness or finger_thickness or FINGER_THICKNESS),
        "max_width": widest,
        "rotations": max(1, int(rotations)),
        "min_part_height": float(min_part_height),
        "min_insertion": float(min_insertion),
        "unknown_is_obstacle": bool(unknown_is_obstacle),
        "tip_reach": reach,
    }

    cam = _camera(ctx, camera_frame, camera)
    depth = np.asarray(cam["depth"], dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    t_wc = _matrix(cam["pose"])
    hgt, wid = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    if ignore_mask is not None:
        valid &= ~_as_bool(ignore_mask)
    target = _as_bool(mask) if mask is not None else None
    others = [_as_bool(pm) for pm in (piece_masks or [])]
    if target is not None:
        others = [o for o in others if (o & target).sum() < 0.5 * max(1, o.sum())]

    vv, uu = np.nonzero(valid)
    zc = depth[vv, uu]
    pc = np.column_stack([(uu - k[0, 2]) * zc / k[0, 0], (vv - k[1, 2]) * zc / k[1, 1], zc])
    pts = pc @ t_wc[:3, :3].T + t_wc[:3, 3]
    labels = np.zeros(len(pts), dtype=np.int32)
    for i, o in enumerate(others):
        labels[o[vv, uu]] = 2 + i
    if target is not None:
        labels[target[vv, uu]] = 1

    def to_px(p: np.ndarray) -> list[float]:
        q = np.linalg.inv(t_wc) @ np.append(p, 1.0)
        return [float(k[0, 0] * q[0] / q[2] + k[0, 2]), float(k[1, 1] * q[1] / q[2] + k[1, 2])]

    empty: Output = {
        "route": "nothing_graspable",
        "reason": "",
        "grasps": [],
        "position": None,
        "close_heading_deg": 0.0,
        "jaw_width_m": 0.0,
        "object_width_m": 0.0,
        "tip_z": 0.0,
        "tool_z": 0.0,
        "clearance_m": 0.0,
        "score": 0.0,
        "grasp_pose": None,
        "rotation_flipped": None,
        "pieces": [],
        "n_pieces": 0,
        "n_graspable": 0,
        "fraction_graspable": 0.0,
        "support_z": 0.0,
        "noise_m": 0.0,
        "voxel_m": 0.0,
    }
    if target is not None and not (labels == 1).any():
        return {**empty, "reason": "the target mask covers no pixel with depth"}
    if len(pts) < 50:
        return {**empty, "reason": "too few depth pixels"}

    reach_r = 0.5 * widest + params["finger_thickness"] + CLEARANCE_WINDOW + 0.01
    if roi is not None:
        x0, x1, y0, y1 = (float(roi[key]) for key in ("x_min", "x_max", "y_min", "y_max"))
    elif target is not None:
        tp = pts[labels == 1]
        cx, cy = np.median(tp[:, 0]), np.median(tp[:, 1])
        r = reach_r + 0.5 * float(np.hypot(np.ptp(tp[:, 0]), np.ptp(tp[:, 1])))
        x0, x1, y0, y1 = cx - r, cx + r, cy - r, cy + r
    else:
        x0 = x1 = y0 = y1 = None  # type: ignore[assignment]

    if x0 is not None:
        near = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        free = near & (labels == 0)
        pool = pts[free] if free.sum() >= 200 else pts[near]
    else:
        centre = (np.abs(uu - wid / 2) < wid / 4) & (np.abs(vv - hgt / 2) < hgt / 4)
        pool = pts[centre] if centre.sum() >= 200 else pts
    if len(pool) < 50:
        return {**empty, "reason": "too few depth pixels round the target"}
    plane, noise = _support_plane(pool, support_z)
    params["min_part_height"] = max(params["min_part_height"], NOISE_STANDS * noise)
    params["min_insertion"] = max(params["min_insertion"], NOISE_INSERTION * noise)
    pts = pts.copy()
    pts[:, 2] -= pts[:, 0] * plane[0] + pts[:, 1] * plane[1] + plane[2]  # heights over the support
    keep = pts[:, 2] <= MAX_PART_HEIGHT
    pts, labels, zc_k = pts[keep], labels[keep], zc[keep]

    def support_at(x: float, y: float) -> float:
        return float(plane[0] * x + plane[1] * y + plane[2])

    if x0 is None:
        stand = pts[:, 2] > params["min_part_height"]
        base = pts[stand] if stand.sum() >= 20 else pts
        mx, my = np.median(base[:, 0]), np.median(base[:, 1])
        x0 = max(base[:, 0].min() - reach_r, mx - MAX_GRID / 2)
        x1 = min(base[:, 0].max() + reach_r, mx + MAX_GRID / 2)
        y0 = max(base[:, 1].min() - reach_r, my - MAX_GRID / 2)
        y1 = min(base[:, 1].max() + reach_r, my + MAX_GRID / 2)

    inroi = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    if inroi.sum() < 50:
        return {**empty, "reason": "too few depth pixels in the search area"}
    gsd = float(np.median(zc_k[inroi])) / float(0.5 * (k[0, 0] + k[1, 1]))
    vox = float(voxel) if voxel else float(np.clip(gsd, 0.0005, 0.002))
    nx, ny = int(math.ceil((x1 - x0) / vox)), int(math.ceil((y1 - y0) / vox))
    hm = _Map(pts[inroi], labels[inroi], x0, y0, nx, ny, vox, support_at)

    if target is None and not others:
        blobs, n_blobs = ndi.label(
            np.nan_to_num(hm.h, nan=-1.0) > params["min_part_height"], np.ones((3, 3))
        )
        hm.labels = blobs.astype(np.int32)
        ids = list(range(1, n_blobs + 1))
    else:
        _join_unmasked(hm, params["min_part_height"])
        ids = [1] if target is not None else sorted(int(i) for i in np.unique(hm.labels) if i > 0)

    high = np.nan_to_num(hm.h, nan=-1.0) > params["min_part_height"]
    cell_area = vox * vox
    sized = []
    for lab in ids:
        n_own = int(((hm.labels == lab) & high).sum())
        n_seen = int((hm.labels == lab).sum())
        sized.append((lab, n_own, n_seen))
    if target is None:
        sized = [s for s in sized if s[1] * cell_area >= MIN_AREA]
        cx0, cy0 = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

        def dist_to_centre(s):
            yy, xx = np.nonzero((hm.labels == s[0]) & high)
            px, py = hm.centre(yy.mean(), xx.mean())
            return math.hypot(px - cx0, py - cy0)

        sized = sorted(sized, key=dist_to_centre)[: max(1, int(max_pieces))]

    pieces: list[dict[str, Any]] = []
    all_grasps: list[dict] = []
    for lab, n_own, n_seen in sized:
        if n_own * cell_area < MIN_AREA:
            why = "too_flat" if n_seen * cell_area >= MIN_AREA else "not_seen"
            pieces.append({"piece": lab, "status": why, "grasps": 0})
            continue
        grasps, fails = _search_piece(hm, lab, params)
        best = _distinct(grasps, max(1, int(per_piece)))
        if best:
            status = "graspable"
        elif not (fails["collision"] or fails["neighbour_between"]):
            status = "too_wide" if fails["too_wide"] >= fails["poor_contact"] else "poor_contact"
        elif fails["neighbour_between"] > fails["collision"]:
            status = "neighbour_between"
        else:
            status = "crowded"
        yy, xx = np.nonzero((hm.labels == lab) & high)
        px, py = hm.centre(yy.mean(), xx.mean())
        pieces.append(
            {
                "piece": lab,
                "status": status,
                "grasps": len(best),
                "centroid": {"x": float(px), "y": float(py)},
                "height_m": float(np.median(hm.h[(hm.labels == lab) & high])),
                "fails": fails,
                "best_clearance_m": float(best[0]["clearance_m"]) if best else None,
            }
        )
        all_grasps.extend(best)

    n_pieces = len(pieces)
    n_ok = sum(p["status"] == "graspable" for p in pieces)
    base_out = {
        **empty,
        "pieces": pieces,
        "n_pieces": n_pieces,
        "n_graspable": n_ok,
        "fraction_graspable": n_ok / n_pieces if n_pieces else 0.0,
        "support_z": support_at(0.5 * (x0 + x1), 0.5 * (y0 + y1)),
        "noise_m": noise,
        "voxel_m": vox,
    }
    if not all_grasps:
        statuses = [p["status"] for p in pieces]
        if any(s in ("crowded", "neighbour_between") for s in statuses):
            return {
                **base_out,
                "route": "too_crowded",
                "reason": (
                    f"{n_pieces} part(s) seen, but no {params['finger_thickness'] * 1e3:.0f} x "
                    f"{params['finger_width'] * 1e3:.0f} mm fingertip fits beside any of them down to "
                    "half their height (or another part lies between the jaws): singulate or spread them"
                ),
            }
        why = ", ".join(sorted(set(statuses))) or "no part stands out of the support"
        return {**base_out, "reason": f"nothing to grasp: {why}"}

    ranked = sorted(all_grasps, key=lambda g: -g["score"])
    public = [_public(g, support_at, reach, to_px) for g in ranked]
    best = public[0]
    pose = None
    flipped = None
    if frame:
        f = ctx.tool("robot.grasp_frame", close_heading_deg=best["close_heading_deg"], **on_arm)
        rot = _wxyz((f or {}).get("rotation"))
        flipped = _wxyz((f or {}).get("rotation_flipped"))
        if rot is not None:
            pose = {
                "position": {
                    "x": best["position"]["x"],
                    "y": best["position"]["y"],
                    "z": best["tool_z"],
                },
                "rotation": rot,
            }
    return {
        **base_out,
        "route": "proposed",
        "reason": "",
        "grasps": public,
        "position": best["position"],
        "close_heading_deg": best["close_heading_deg"],
        "jaw_width_m": best["jaw_width_m"],
        "object_width_m": best["object_width_m"],
        "tip_z": best["tip_z"],
        "tool_z": best["tool_z"],
        "clearance_m": best["clearance_m"],
        "score": best["score"],
        "grasp_pose": pose,
        "rotation_flipped": flipped,
        "support_z": best["support_z"],
    }
