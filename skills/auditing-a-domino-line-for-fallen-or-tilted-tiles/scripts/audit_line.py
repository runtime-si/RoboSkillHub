# The tile model and the depth profile fit are those of
# skills/verifying-a-domino-cascade-from-video/scripts/verify_cascade.py (Tile, _profiles, _profile_obs,
# _fit_profile, Cam, _lift), which adapts predicators (https://github.com/BasisResearch/predicators, commit
# afb6f2918091f5ca372a75b9a335aea317a8a209), MIT, Copyright (c) 2022 Rohan Chitnis and Tom Silver (license:
# ../LICENSE.predicators): the upright / toppled bands of
# predicators/envs/pybullet_domino/components/domino_component.py (upright under 5 deg, toppled from 10 deg).
# Changed: one frame instead of a clip, so no onset, no visibility history and no first-frame position: the
# tile's shift along the row is fitted jointly with its lean; a lean backward (about the back edge) is fitted
# as well as forward; per tile states (upright, displaced, tilted, leaning on its neighbour, fallen, missing)
# instead of a fall time; the gaps between the tiles as they stand checked against the critical gap.
"""Audit a standing domino line from one depth frame: which tiles are upright, displaced, tilted, leaning, fallen or missing."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

TILE_SIZE = (0.048, 0.024, 0.0075)
"""Tile height, width, thickness [m]: planning-a-domino-layout's default tile."""

UPRIGHT_DEG = 5.0
"""A tile leaning less than this stands upright [deg]: predicators' domino_roll_threshold
(5 deg). A 5 deg lean moves the top 4.2 mm, two 2 mm depth bins."""

FALLEN_DEG = 60.0
"""A tile leaning this far or more has fallen [deg], whatever holds it (a tile lying on a
fallen neighbour rests at 75-80 deg, one flat at 90)."""

CONTACT_BAND = (8.0, 4.0)
"""A tile leaning forward from this far under to this far over the angle at which its top
meets the next tile [deg] rests on that tile: leaning, propped (verify_cascade's 8 deg
CONTACT_MARGIN_DEG; 4 deg over for the lean's fit error)."""

SHIFT_RANGE = 0.010
SHIFT_STEP = 0.001
"""Along-row offsets of the tile's base fitted with its lean [m]: a placed tile is within
a few mm of its slot (verify_cascade's MAX_SHIFT is 8 mm)."""

DISPLACED = 0.004
"""An upright tile more than this off its slot, along or across the row, is displaced [m]
(the placing skill puts tiles within 2 mm in the twin; its depth check allows 8)."""

YAW_TOLERANCE_DEG = 10.0
"""An upright tile turned more than this from its slot's facing is displaced [deg]."""

CRITICAL_GAP_H = 0.996
"""The widest face-to-face gap a running wave crosses, in tile heights: 47.8 mm for a 48 mm
tile (planning-a-domino-layout's GapChain result)."""

BIN = 0.002
LATERAL_MARGIN = 0.003
OCCLUDER_ABOVE = 0.012
BIN_TOP_PCT = 90.0
OVER_WEIGHT = 0.3
FIT_CAP = 0.015
MAX_FIT_COST = 0.006
MIN_BINS = 4
GAP_MARGIN = 0.002
ANGLE_STEP = 0.5
"""The profile fit (verify_cascade): bin width, side margin, what over a tile is an
occluder, the bin's top percentile, the weight of a bin higher than the model, the
residual cap, the largest mean residual of a usable fit, fewest bins, the margin kept
from the neighbour's face, and the angle grid."""

BACKWARD_MARGIN = 0.0005
"""A backward lean is taken only when it fits this much better [m] than the best forward
one: a cascade topples forward, and in a fallen run a pile of tiles can be explained
leaning either way (in the station twin, without it, 4 of 13 fallen tiles read backward)."""

SECTOR_FROM_DEG = 45.0
"""A neighbour leaning this far or more is taken out of a tile's window over every angle
from its fitted one down to flat [deg]: in a fallen run the fit sees only part of a tile
lying across the next slot (in the twin a flat tile read 67.5 deg and its body was then
read as the next slot's own fallen tile)."""

LEAN_SLACK_DEG = 2.0
"""A neighbour's body is taken out over its fitted lean +- this [deg]: its fit is good to
about a degree, and the sliver a degree leaves behind made the tile it leans on read
tilted at 1 mm depth noise."""

RAISED = 0.004
"""Depth this far over the table is something on it [m]."""

MIN_POINTS = 6
"""Fewest raised depth points that make a tile present."""

PREFERRED_CAMERAS = ("overhead", "top")


class Output(TypedDict):
    route: str
    detail: str
    n_tiles: int
    n_upright: int
    first_fault: int
    states: list
    lean_deg: list
    shift_mm: list
    lateral_mm: list
    yaw_error_deg: list
    fit_cost_mm: list
    faults: list
    fallen_runs: list
    wide_gaps: list
    camera: str


# ----------------------------------------------------------------------------- inputs


def _matrix(q: Any) -> np.ndarray:
    if isinstance(q, dict):
        w, x, y, z = (float(q.get(k, 0.0)) for k in "wxyz")
    else:
        w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _xyz(p: Any) -> np.ndarray:
    if isinstance(p, dict):
        return np.array([float(p["x"]), float(p["y"]), float(p.get("z", 0.0))])
    v = [float(a) for a in list(p)[:3]]
    return np.array(v + [0.0] * (3 - len(v)))


def _slot(s: Any) -> tuple[np.ndarray, np.ndarray]:
    """(centre at the table, unit facing) of a slot: an Se3Pose (x axis the facing), a
    planner slot with facing_deg, or [x, y, facing_deg]."""
    if isinstance(s, dict):
        pose = s.get("pose", s) if "position" not in s else s
        c = _xyz(pose["position"] if "position" in pose else pose)
        if "facing_deg" in s:
            a = math.radians(float(s["facing_deg"]))
            f = np.array([math.cos(a), math.sin(a), 0.0])
        elif pose.get("rotation") is not None:
            f = _matrix(pose["rotation"])[:, 0].copy()
            f[2] = 0.0
        else:
            raise ValueError(f"slot {s!r} has no facing")
    else:
        v = [float(a) for a in s]
        if len(v) < 3:
            raise ValueError(f"slot {s!r} is not [x, y, facing_deg]")
        c = np.array([v[0], v[1], 0.0])
        a = math.radians(v[2])
        f = np.array([math.cos(a), math.sin(a), 0.0])
    n = float(np.linalg.norm(f))
    if n < 1e-6:
        raise ValueError(f"slot {s!r} faces straight up")
    return c, f / n


# ----------------------------------------------------------------------------- tile geometry


def _profiles(grid: np.ndarray, bins: np.ndarray, h: float, d: float) -> np.ndarray:
    """Upper height envelope over each bin of the tile rotated by each grid angle about its
    front bottom edge at x = 0, its back face at x = -d (verify_cascade)."""
    t = np.radians(grid)[:, None]
    c, s = np.cos(t), np.sin(t)
    corners = [
        (np.zeros_like(c), np.zeros_like(c)),
        (-d * c, d * s),
        (-d * c + h * s, d * s + h * c),
        (h * s, h * c),
    ]
    x = bins[None, :]
    top = np.zeros((len(grid), len(bins)))
    for k in range(4):
        (x0, z0), (x1, z1) = corners[k], corners[(k + 1) % 4]
        lo, hi = np.minimum(x0, x1), np.maximum(x0, x1)
        span = np.where(np.abs(x1 - x0) < 1e-12, 1e-12, x1 - x0)
        z = z0 + (x - x0) * (z1 - z0) / span
        inside = (x >= lo) & (x <= hi)
        top = np.where(inside, np.maximum(top, z), top)
    return top


class Cam:
    def __init__(self, cam: dict[str, Any]):
        self.name = str(cam.get("name", ""))
        self.K = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
        pose = cam["pose"]
        if isinstance(pose, dict) and "position" in pose:
            t = np.eye(4)
            t[:3, :3] = _matrix(pose.get("rotation") or pose.get("orientation"))
            t[:3, 3] = _xyz(pose["position"])
            self.T = t
        else:
            self.T = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        self.R, self.t = self.T[:3, :3], self.T[:3, 3]
        depth = np.asarray(cam["depth"], dtype=np.float64)
        self.depth = depth[..., 0] if depth.ndim == 3 else depth

    def points(self, world_corners: np.ndarray) -> np.ndarray:
        """World points of the depth pixels inside the image box of ``world_corners``;
        none when the box is not wholly in the image."""
        pc = (world_corners - self.t) @ self.R
        if np.any(pc[:, 2] <= 0.05):
            return np.zeros((0, 3))
        u = self.K[0, 0] * pc[:, 0] / pc[:, 2] + self.K[0, 2]
        v = self.K[1, 1] * pc[:, 1] / pc[:, 2] + self.K[1, 2]
        h, w = self.depth.shape
        u0, u1 = int(math.floor(u.min())) - 2, int(math.ceil(u.max())) + 3
        v0, v1 = int(math.floor(v.min())) - 2, int(math.ceil(v.max())) + 3
        if u0 < 0 or v0 < 0 or u1 > w or v1 > h:
            return np.zeros((0, 3))
        z = self.depth[v0:v1, u0:u1]
        vv, uu = np.mgrid[v0:v1, u0:u1]
        ok = np.isfinite(z) & (z >= 0.015) & (z <= 20.0)
        z, uu, vv = z[ok], uu[ok], vv[ok]
        x = (uu - self.K[0, 2]) * z / self.K[0, 0]
        y = (vv - self.K[1, 2]) * z / self.K[1, 1]
        return np.stack([x, y, z], axis=1) @ self.R.T + self.t


def fit_tile(
    local: np.ndarray,
    h: float,
    w: float,
    d: float,
    front_room: float,
    back_room: float,
    bias: float = 0.0,
) -> dict[str, Any]:
    """Joint fit of a tile's along-row shift and lean from the depth points in its window.
    ``local``: x forward from the slot's front bottom edge, y across, z over the table.
    Forward lean pivots on the front edge, backward on the back edge. Returns the best
    (lean_deg, signed: + forward), shift [m], cost [m] and the bins used."""
    band = np.abs(local[:, 1]) <= w / 2.0 - LATERAL_MARGIN
    pts = local[band & (local[:, 2] <= h + OCCLUDER_ABOVE)]
    x_lo, x_hi = -d - back_room, front_room
    bins = np.arange(x_lo + BIN / 2.0, x_hi, BIN)
    nb = len(bins)
    if nb < MIN_BINS:
        return {"lean": math.nan, "shift": 0.0, "cost": math.inf, "bins": 0}
    k = np.floor((pts[:, 0] - x_lo) / BIN).astype(int)
    keep = (k >= 0) & (k < nb)
    k, z = k[keep], pts[keep, 2]
    obs = np.full(nb, np.nan)
    for b in np.unique(k):
        obs[b] = float(np.percentile(z[k == b], BIN_TOP_PCT)) - bias
    ok = np.isfinite(obs)
    if ok.sum() < MIN_BINS:
        return {"lean": math.nan, "shift": 0.0, "cost": math.inf, "bins": int(ok.sum())}
    grid = np.arange(0.0, 90.0 + 1e-9, ANGLE_STEP)
    best_by_sign = {1.0: (math.inf, 0.0, 0.0), -1.0: (math.inf, 0.0, 0.0)}
    for s in np.arange(-SHIFT_RANGE, SHIFT_RANGE + 1e-9, SHIFT_STEP):
        for sign in (1.0, -1.0):
            # forward: x' = x - s; backward: mirror so the back edge is the pivot
            # the model's highest point over the whole bin (its edges and centre): a face
            # seen at an angle lies on a bin edge, and a centre sample would miss it
            prof = None
            for off in (-BIN / 2.0, 0.0, BIN / 2.0):
                xs = bins + off - s if sign > 0 else -(bins + off - s) - d
                p = _profiles(grid, xs, h, d)
                prof = p if prof is None else np.maximum(prof, p)
            diff = obs[ok][None, :] - prof[:, ok]
            res = np.minimum(np.where(diff > 0, OVER_WEIGHT * diff, -diff), FIT_CAP).mean(axis=1)
            i = int(np.argmin(res))
            if res[i] < best_by_sign[sign][0] - 1e-12:
                best_by_sign[sign] = (float(res[i]), sign * float(grid[i]), float(s))
    fwd, bwd = best_by_sign[1.0], best_by_sign[-1.0]
    best = bwd if bwd[0] < fwd[0] - BACKWARD_MARGIN else fwd
    return {"lean": best[1], "shift": best[2], "cost": best[0], "bins": int(ok.sum())}


def noise_sigma(local: np.ndarray) -> float:
    """The depth noise [m] from the points near the table in a window (their MAD)."""
    z = local[np.abs(local[:, 2]) < 0.006, 2] if len(local) else np.zeros(0)
    if len(z) < 20:
        return 0.0
    return float(1.4826 * np.median(np.abs(z - np.median(z))))


def _yaw_error(local: np.ndarray, h: float) -> float:
    """The tile's turn about the vertical from its slot's facing [deg], from the long axis
    of its top face's points (nan with too few)."""
    top = local[(local[:, 2] > h - 0.006) & (local[:, 2] < h + OCCLUDER_ABOVE)]
    if len(top) < 12:
        return math.nan
    xy = top[:, :2] - top[:, :2].mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(xy.T))
    long_axis = vecs[:, -1]  # the top's width runs across the facing
    ang = math.degrees(math.atan2(-long_axis[0], long_axis[1]))
    return (ang + 90.0) % 180.0 - 90.0


# ----------------------------------------------------------------------------- the skill


def run(
    ctx: NodeContext,
    slots: list[Any],
    table_z: float | None = None,
    tile_size: Any = TILE_SIZE,
    frame: dict[str, Any] | None = None,
    camera_name: str = "",
    upright_deg: float = UPRIGHT_DEG,
    fallen_deg: float = FALLEN_DEG,
    displaced: float = DISPLACED,
    yaw_tolerance_deg: float = YAW_TOLERANCE_DEG,
) -> Output:
    out: Output = {
        "route": "invalid_input",
        "detail": "",
        "n_tiles": 0,
        "n_upright": 0,
        "first_fault": -1,
        "states": [],
        "lean_deg": [],
        "shift_mm": [],
        "lateral_mm": [],
        "yaw_error_deg": [],
        "fit_cost_mm": [],
        "faults": [],
        "fallen_runs": [],
        "wide_gaps": [],
        "camera": "",
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        return out

    try:
        if isinstance(tile_size, dict):
            h, w, d = (
                float(tile_size.get(k, v))
                for k, v in zip(("height", "width", "thickness"), TILE_SIZE, strict=True)
            )
        else:
            h, w, d = (float(v) for v in tile_size)
        if not h > w > d > 0:
            raise ValueError(f"tile_size {tile_size} is not height > width > thickness > 0")
        items = [_slot(s) for s in slots or []]
        if not items:
            raise ValueError("no slots")
        if table_z is None:
            table_z = float(np.median([c[2] for c, _ in items]))
        table_z = float(table_z)
    except (ValueError, KeyError, TypeError) as exc:
        return finish("invalid_input", str(exc))
    n = len(items)
    out["n_tiles"] = n

    # --- the frame: the given one, else the observation's cameras with depth
    try:
        raw = (
            [frame]
            if frame is not None
            else (ctx.tool("robot.get_observation").get("cameras") or [])
        )
    except Exception as exc:  # noqa: BLE001
        return finish("not_seen", f"no observation ({exc})")
    if isinstance(raw, dict):
        raw = [dict(v, name=k) for k, v in raw.items()]
    cams = []
    for c in raw:
        if (
            not isinstance(c, dict)
            or c.get("depth") is None
            or c.get("intrinsics") is None
            or c.get("pose") is None
        ):
            continue
        if camera_name and c.get("name") != camera_name:
            continue
        try:
            cams.append(Cam(c))
        except (ValueError, KeyError, TypeError) as exc:
            logger.info("auditing-a-domino-line: camera skipped (%s)", exc)
    rank = {nm: i for i, nm in enumerate(PREFERRED_CAMERAS)}
    cams.sort(key=lambda c: rank.get(c.name, len(rank)))
    if not cams:
        return finish("not_seen", "no camera frame with depth, intrinsics and pose")

    # --- the room each tile has to lean into: up to the neighbours' faces
    centres = [c.copy() for c, _ in items]
    for c in centres:
        c[2] = table_z
    facing = [f for _, f in items]

    def face_gap(a: int, b: int, sa: float = 0.0, sb: float = 0.0) -> float:
        """From a's front face to b's back face, along a's facing [m]."""
        front_a = centres[a] + facing[a] * (d / 2 + sa)
        back_b = centres[b] + facing[b] * (sb - d / 2)
        return float(((back_b - front_a) @ facing[b]) / max(float(facing[a] @ facing[b]), 0.2))

    def window(i: int) -> tuple[list[tuple[str, np.ndarray]], tuple[float, float]]:
        """Per camera that sees it whole: tile i's window points in its slot frame (x forward
        from the front bottom edge, y across, z over the table), up to GAP_MARGIN short of
        the neighbours' faces; and the room it has to lean into, forward and back."""
        f = facing[i]
        lat = np.array([-f[1], f[0], 0.0])
        pivot = centres[i] + f * d / 2
        front = min(face_gap(i, i + 1) - GAP_MARGIN, h + 0.002) if i + 1 < n else h + 0.002
        back = min(face_gap(i - 1, i) - GAP_MARGIN, h + 0.002) if i > 0 else h + 0.002
        front, back = max(front, 0.004), max(back, 0.004)
        xs = (-d - back, front)
        ys = (-w / 2 - 0.008, w / 2 + 0.008)
        corners = np.array(
            [
                pivot + x * f + y * lat + np.array([0.0, 0.0, z])
                for x in xs
                for y in ys
                for z in (0.0, h + 0.04)
            ]
        )
        seen = []
        for cam in cams:
            pts = cam.points(corners)
            if not len(pts):
                continue
            rel = pts - pivot
            local = np.stack([rel @ f, rel @ lat, pts[:, 2] - table_z], axis=1)
            win = local[
                (local[:, 0] >= xs[0]) & (local[:, 0] <= xs[1]) & (np.abs(local[:, 1]) <= ys[1])
            ]
            if len(win) and np.mean(win[:, 2] > h + OCCLUDER_ABOVE) > 0.25:
                continue  # a hand or the arm over it
            seen.append((cam.name, win))
        return seen, (front, back)

    def to_world(i: int, local: np.ndarray) -> np.ndarray:
        f = facing[i]
        lat = np.array([-f[1], f[0], 0.0])
        pivot = centres[i] + f * d / 2
        return (
            pivot
            + np.outer(local[:, 0], f)
            + np.outer(local[:, 1], lat)
            + np.outer(local[:, 2], [0, 0, 1.0])
        )

    def inside_fit(j: int, fit: dict[str, Any], world: np.ndarray, g: float = 0.0015) -> np.ndarray:
        """Which world points lie inside tile j's fitted (leaning) box, 1.5 mm grown."""
        f = facing[j]
        lat = np.array([-f[1], f[0], 0.0])
        pivot = centres[j] + f * d / 2
        rel = world - pivot
        x, y, z = rel @ f - fit["shift"], rel @ lat, world[:, 2] - table_z
        if fit["lean"] < 0:
            x = -x - d  # mirror: pivot on the back edge
        lean = abs(fit["lean"])
        # a fallen tile lies somewhere from its fitted angle down to flat: take out the sector
        if lean < SECTOR_FROM_DEG:
            angles = [max(0.0, lean - LEAN_SLACK_DEG), lean, lean + LEAN_SLACK_DEG]
        else:
            angles = list(np.arange(max(0.0, lean - LEAN_SLACK_DEG), 90.0 + 1e-9, 2.0)) + [90.0]
        inside = np.zeros(len(world), bool)
        for ang in angles:
            a = math.radians(ang)
            # rotate back by the lean about the pivot (y axis): tile frame x' in [-d, 0], z' in [0, h]
            xr = x * math.cos(a) - z * math.sin(a)
            zr = x * math.sin(a) + z * math.cos(a)
            inside |= (
                (xr >= -d - g) & (xr <= g) & (zr >= -g) & (zr <= h + g) & (np.abs(y) <= w / 2 + g)
            )
        return inside

    def fit_one(i: int, drop: list[tuple[int, dict[str, Any]]]) -> dict[str, Any] | None:
        seen, (front, back) = window(i)
        best: dict[str, Any] | None = None
        for name, win in seen:
            sigma = noise_sigma(win)
            lift = max(RAISED, 3.0 * sigma)
            had_body = len(win) > 0 and int(np.sum(win[:, 2] > lift)) >= MIN_POINTS
            for j, fj in drop:
                if len(win):
                    win = win[~inside_fit(j, fj, to_world(i, win), max(0.002, 3.0 * sigma))]
            raised = win[win[:, 2] > lift] if len(win) else win
            if len(raised) < MIN_POINTS:
                # nothing left of its own: empty, or wholly hidden in a fallen pile
                return {"state": "missing", "camera": name, "local": win, "explained": had_body}
            # the tile's turn from its top face, then the fit in its own frame
            body = win[np.abs(win[:, 0] + d / 2) <= d / 2 + SHIFT_RANGE]
            yaw = _yaw_error(body, h) if len(body) else math.nan
            loc = win
            if math.isfinite(yaw) and abs(yaw) > 2.0:
                c0 = np.array([-d / 2, 0.0])
                r = math.radians(-yaw)
                xy = (win[:, :2] - c0) @ np.array(
                    [[math.cos(r), math.sin(r)], [-math.sin(r), math.cos(r)]]
                ) + c0
                loc = np.column_stack([xy, win[:, 2]])
            fit = fit_tile(loc, h, w, d, front, back, bias=1.28 * sigma)
            fit.update(camera=name, local=loc, yaw=yaw, sigma=sigma)
            if best is None or fit["cost"] < best.get("cost", math.inf):
                best = fit
            if fit["cost"] <= MAX_FIT_COST:
                break
        return best

    fits = [fit_one(i, []) for i in range(n)]
    # second pass: a leaning or fallen neighbour's body lies in a tile's window; take it out
    leaning = {
        j: fj
        for j, fj in enumerate(fits)
        if fj is not None
        and fj.get("state") != "missing"
        and math.isfinite(fj.get("lean", math.nan))
        and abs(fj["lean"]) >= float(upright_deg)
    }
    for _ in range(2):  # in line order, each refit with its neighbours' latest fits taken out
        for i in range(n):
            drop = [(j, leaning[j]) for j in (i - 1, i + 1) if j in leaning]
            if not drop:
                continue
            fits[i] = fit_one(i, drop)
            fi = fits[i]
            if (
                fi is not None
                and fi.get("state") != "missing"
                and math.isfinite(fi.get("lean", math.nan))
                and abs(fi["lean"]) >= float(upright_deg)
            ):
                leaning[i] = fi
            else:
                leaning.pop(i, None)

    if all(f is None for f in fits):
        return finish("not_seen", "no camera saw any slot whole and unoccluded")

    # --- states
    states, faults = [], []
    for i, fit in enumerate(fits):
        lean = shift = lateral = yaw = cost = math.nan
        if fit is None:
            state = "unseen"
        elif fit.get("state") == "missing":
            state = "missing"
            out["camera"] = out["camera"] or fit["camera"]
        else:
            out["camera"] = out["camera"] or fit["camera"]
            lean, shift, cost = fit["lean"], fit["shift"], fit["cost"]
            if cost > MAX_FIT_COST:
                state = "unclear"
            else:
                contact = 90.0
                if lean >= 0 and i + 1 < n:
                    gap = face_gap(i, i + 1, shift)
                    contact = (
                        math.degrees(math.asin(min(1.0, max(gap, 1e-4) / h))) if gap < h else 90.0
                    )
                elif lean < 0 and i > 0:
                    gap = face_gap(i - 1, i, 0.0, shift)
                    contact = (
                        math.degrees(math.asin(min(1.0, max(gap, 1e-4) / h))) if gap < h else 90.0
                    )
                a = abs(lean)
                if a < float(upright_deg):
                    loc = fit["local"]
                    top = loc[(loc[:, 2] > h - 0.006) & (loc[:, 2] < h + OCCLUDER_ABOVE)]
                    lateral = float(np.median(top[:, 1])) if len(top) >= 6 else 0.0
                    yaw = fit.get("yaw", math.nan)
                    off = abs(shift) > float(displaced) or abs(lateral) > float(displaced)
                    turned = math.isfinite(yaw) and abs(yaw) > float(yaw_tolerance_deg)
                    state = "displaced" if (off or turned) else "upright"
                elif a >= float(fallen_deg) or a > contact + CONTACT_BAND[1]:
                    state = "fallen"
                elif contact < 90.0 and a >= contact - CONTACT_BAND[0]:
                    state = "leaning"
                else:
                    state = "tilted"
        states.append(state)
        out["lean_deg"].append(None if not math.isfinite(lean) else round(lean, 1))
        out["shift_mm"].append(None if not math.isfinite(shift) else round(shift * 1e3, 1))
        out["lateral_mm"].append(None if not math.isfinite(lateral) else round(lateral * 1e3, 1))
        out["yaw_error_deg"].append(None if not math.isfinite(yaw) else round(yaw, 1))
        out["fit_cost_mm"].append(None if not math.isfinite(cost) else round(cost * 1e3, 2))
        if state != "upright":
            faults.append(
                {
                    "index": i,
                    "state": state,
                    "lean_deg": out["lean_deg"][-1],
                    "shift_mm": out["shift_mm"][-1],
                    "direction": "forward"
                    if (math.isfinite(lean) and lean > 0)
                    else ("backward" if math.isfinite(lean) and lean < 0 else ""),
                }
            )

    # physical consistency between neighbours
    def contact_to(i: int, j: int) -> float:
        g = face_gap(i, j) if j > i else face_gap(j, i)
        return math.degrees(math.asin(min(1.0, max(g, 1e-4) / h))) if g < h else 90.0

    standing = ("upright", "displaced")
    for i in range(n):
        fi = fits[i]
        hidden = states[i] == "unclear" or (
            states[i] == "missing" and fi is not None and fi.get("explained")
        )
        if hidden and 0 < i < n - 1 and states[i - 1] == "fallen" and states[i + 1] == "fallen":
            states[i] = "fallen"  # inside a fallen run its body cannot be told from its neighbours'
    for i in range(n):
        lean = out["lean_deg"][i]
        if states[i] == "fallen" and lean is not None:
            j = i + 1 if lean > 0 else i - 1
            if (
                0 <= j < n
                and states[j] in standing
                and abs(lean) > contact_to(i, j) + CONTACT_BAND[1]
            ):
                # its own tile would rest on that standing neighbour, not lie past it: the
                # body there is the tile behind it, fallen across this slot
                states[i] = "missing"
        if states[i] == "tilted" and lean is not None and abs(lean) >= SECTOR_FROM_DEG:
            if any(0 <= j < n and states[j] == "fallen" for j in (i - 1, i + 1)):
                states[i] = "fallen"  # part of a fallen run
    faults = [dict(f, state=states[f["index"]]) for f in faults]
    out["states"] = states
    out["faults"] = faults
    out["n_upright"] = sum(s == "upright" for s in states)
    out["first_fault"] = faults[0]["index"] if faults else -1

    # runs of fallen / leaning tiles (a stopped wave), and gaps the wave cannot cross
    run_start = None
    for i, s in enumerate(states + ["end"]):
        down = s in ("fallen", "leaning")
        if down and run_start is None:
            run_start = i
        elif not down and run_start is not None:
            out["fallen_runs"].append([run_start, i - 1])
            run_start = None
    crit = CRITICAL_GAP_H * h
    for i in range(n - 1):
        if states[i] in ("upright", "displaced") and states[i + 1] in ("upright", "displaced"):
            s_i = (out["shift_mm"][i] or 0.0) / 1e3
            s_j = (out["shift_mm"][i + 1] or 0.0) / 1e3
            g = face_gap(i, i + 1, s_i, s_j)
            if g > crit:
                out["wide_gaps"].append({"pair": [i, i + 1], "gap_mm": round(g * 1e3, 1)})

    unseen = [i for i, s in enumerate(states) if s in ("unseen", "unclear")]
    down = [f for f in faults if f["state"] in ("fallen", "missing")]
    other = [f for f in faults if f["state"] in ("leaning", "tilted", "displaced")]
    summary = ", ".join(
        f"{f['index']} {f['state']}"
        + (
            f" {f['lean_deg']:+.0f} deg"
            if f["lean_deg"] is not None and f["state"] in ("tilted", "leaning", "fallen")
            else ""
        )
        for f in faults
        if f["state"] not in ("unseen", "unclear")
    )
    gaps = (
        "; gaps the wave cannot cross: "
        + ", ".join(f"{g['pair'][0]}-{g['pair'][1]} {g['gap_mm']:.0f} mm" for g in out["wide_gaps"])
        if out["wide_gaps"]
        else ""
    )
    tail = (f"; not judged: {unseen}" if unseen else "") + gaps
    if down:
        return finish("fallen", f"{out['n_upright']}/{n} upright; {summary}{tail}")
    if other or out["wide_gaps"]:
        return finish(
            "fix_needed", f"{out['n_upright']}/{n} upright; {summary or 'every tile upright'}{tail}"
        )
    if unseen:
        return finish("not_seen", f"{out['n_upright']}/{n} upright, the rest not judged: {unseen}")
    return finish("line_ok", f"all {n} tiles upright on their slots")
