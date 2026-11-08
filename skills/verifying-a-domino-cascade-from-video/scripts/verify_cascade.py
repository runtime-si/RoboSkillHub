# Adapted from predicators (https://github.com/BasisResearch/predicators) at
# afb6f2918091f5ca372a75b9a335aea317a8a209, MIT, Copyright (c) 2022 Rohan Chitnis
# and Tom Silver (license: ../LICENSE.predicators):
# predicators/envs/pybullet_domino/cascade_certificate.py (_topple_onset,
# _TOPPLE_MIN_STEPS) and the onset-detection rules of
# docs/envs/domino/continuous-perception.md sections 3.3-3.4, with the upright /
# fallen bands of predicators/envs/pybullet_domino/components/domino_component.py.
# Kept: a fall counts only once the tile stays at or past the fallen threshold
# for 3 consecutive frames (or in a run that reaches the end of the clip, the
# certificate's asymmetry: a topple in the last frames still counts), and its
# onset is then walked back to the frame after the tile was last seen inside the
# upright band; with no upright frame before the fall the onset is the first
# usable frame. From the design note: frames where too little of the tile is
# seen (40-70 % of its usual visibility) are missing data, neither extending nor
# breaking a run; each tile is measured against its own first frames; the fall
# is confirmed well past any drift or occlusion artefact and dated back; the
# result is scored as onset times and propagation intervals, and no jump or
# speed gate is used (a real cascade is fast motion).
# Changes: the simulator roll per state becomes a per-frame fall angle measured
# from an RGB(-D) clip at planned slots (a rotated-box height profile fitted to
# the depth points along each tile's fall corridor; without depth, the fraction
# of the gap in front of the tile its body covers; optionally a tracked point
# (track) or a tracked mask (sam3) fitted to the rotated box's projection);
# is_held becomes the visibility gate (a hand or the arm over the tile); the
# carry floor of the backward search is dropped (nothing is carried during a
# cascade clip); the confirm threshold is capped below the angle at which the
# tile meets its neighbour, where a tile stopped by a standing neighbour rests;
# the certificate's staging, Push-label and counterfactual-probe rules are not
# ported (they need the simulator); per-tile states, the stop index, the order
# check and a robust wave speed are added, with routed exits.
"""Decide from an RGB(-D) clip which dominoes of a planned line fell, when, how fast the wave ran and where it stopped."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

TILE_SIZE = (0.048, 0.024, 0.0075)
"""Tile height, width, thickness [m]: planning-a-domino-layout's default tile."""

TOPPLE_MIN_FRAMES = 3
"""Consecutive usable frames at or past the fallen threshold before a fall counts:
upstream ``_TOPPLE_MIN_STEPS = 3``."""

END_RUN_MIN_FRAMES = 2
"""A run cut short by the end of the clip counts from this many frames (upstream
counts any length; the design note asks that one bad frame never emit an onset, and
a single noisy last frame did in the synthetic sweep with 4 mm depth noise)."""

FALLEN_DEG = 45.0
"""Confirm a fall at this angle [deg] (capped per tile below its contact angle, see
``CONTACT_MARGIN_DEG``): the design note's 45-60 deg, above the 13 deg drift and
29 deg occlusion artefacts it measured. Upstream's simulator used 10 deg."""

UPRIGHT_DEG = 15.0
"""Upright band [deg] the onset is dated back to: the design note's markerless check
(#130) used 15 deg; upstream's simulator band is 5 deg (``domino_roll_threshold``),
inside a perceived angle's noise."""

CONTACT_MARGIN_DEG = 8.0
"""The confirm threshold stays this far under the angle at which a tile's top meets
its neighbour [deg]: a tile the next one stops rests there (a propped leaner, which
upstream counts as toppled), at 38 deg for the auto pitch of a 48 mm tile. 5 mm of
placement error moves that angle by 7.6 deg at that pitch; with 1.5 mm placement
noise a 5 deg margin missed 2 of 12 stopped chains' last tile in the synthetic sweep."""

MIN_VISIBILITY = 0.5
"""A frame is used for a tile when at least this fraction of the tile's usual
visibility is seen: the design note's 40-70 % band (at 80 % the gate deleted a real
topple, which passed through 78-85 % visibility); upstream's default was 0.6."""

VIS_REFERENCE_PCT = 90.0
"""A tile's usual visibility is this percentile of its visibility over the clip (the
note used the median mask area; a clip where a hand hovers for most of it would drag
a median down)."""

BIN = 0.002
"""Width of the depth profile's bins along the fall direction [m]: the overhead
camera's pixel at the table (~2 mm at 0.9 m, fx 461)."""

LATERAL_MARGIN = 0.003
"""Depth points within this of the tile's side edges are left out [m], so a few mm of
placement error or yaw does not mix in the table beside it."""

OCCLUDER_ABOVE = 0.012
"""Depth over the tile's top by more than this is an occluder (a hand) [m]."""

BIN_TOP_PCT = 90.0
"""A depth bin's height is this percentile of its points, less the same percentile's
excess over the median on the table (the noise): a wrist camera from the side sees
the table under a leaning tile beside the tile, and a median read 0 deg for tiles at
23-48 deg there; a plain 90th percentile read a table with 4 mm noise as a flat tile."""

OVER_WEIGHT = 0.3
"""Weight of a bin seen higher than the rotated tile explains (the tile before it
lying on its back in a fallen row; a hand's fingers) against one seen lower, which
only a wrong angle explains."""

FIT_CAP = 0.015
"""Per-bin residual cap of the profile fit [m] (truncated L1)."""

MAX_FIT_COST = 0.006
"""A frame whose best profile fit misses by more than this mean residual (or 3 times
the tile's median residual in its first frames, for a noisier camera) is not used
[m]: something other than the tile is there. Synthetic clips with 2 mm depth noise
fit at 1-2 mm."""

MIN_BINS = 4
"""Fewest usable profile bins (8 mm) for a depth reading."""

GAP_MARGIN = 0.002
"""The depth corridor ends this short of the next tile's back face [m], so the next
tile's own points stay out of it."""

MAX_SHIFT = 0.008
"""Largest correction of a tile's position from its first frames [m] (a placed tile
is within a few mm of its slot)."""

ANGLE_STEP = 0.5
"""Fall-angle grid [deg]."""

TRACK_MAX_RESIDUAL_PX = 4.0
"""A tracked top point more than this many px off the rotated top's image path (plus
10 % of the tile's height in px) is not used."""

ORDER_TOLERANCE_FRAMES = 1.0
"""Onsets may run backwards by this many frames (plus their uncertainty) before the
order counts as broken: onsets are 2-6 frames apart at 30 fps (the design note)."""

MIN_INTERVAL_FRAMES = 2.0
"""Successive onsets closer than this are flagged: at 15 fps the design note found
the last two onsets 1 frame apart and the order unresolvable."""

RGB_SHADE = (0.4, 1.25)
"""A pixel is tile coloured when it lies near the line through the tile's colour, at
this range of brightness of it: a falling tile's faces turn from the lit top to its
shaded back (0.55-1.0 in the renders), a dark grey hand sits at ~0.2 of a white tile."""

RGB_SPREAD = 4.0
"""A pixel is table coloured within this many median deviations of the table's
colour in the first frame (floored at ``RGB_MIN_CLASS_DIST``); neither tile nor table
is an occluder."""

RGB_CONFIRM_COVER = 0.75
"""Without depth a fall is confirmed when the tile covers this share of the gap in
front of it (25 deg at the auto pitch; a tile stopped by a neighbour 5 mm closer than
planned still covers 0.89 of it), and the upright band is half that angle."""

RGB_STRIP_WIDTH = 0.6
"""Share of the tile's width the rgb gap strip spans, centred on the planned slot: the
rgb signal cannot see where a tile really stands, so the strip leaves room for a few
mm of lateral placement error and yaw (at 0.8 a tile 6 mm and 6.6 deg off its slot
covered too little of it once flat)."""

RGB_MIN_CLASS_DIST = 18.0
"""Floor of the table's colour radius, and two thirds of the tile colour line's
[RGB units]."""

DEFAULT_FPS = 30.0


class Output(TypedDict):
    route: str
    detail: str
    n_tiles: int
    n_fell: int
    stop_index: int
    last_fallen_index: int
    stop_point: dict[str, float]
    wave_speed: float
    wave_rate: float
    first_onset_time: float
    last_onset_time: float
    signal: str
    tracker_fallback: str
    tiles: list[dict[str, Any]]
    intervals: list[float]
    angles: list[list[float | None]]
    visibility: list[list[float]]
    times: list[float]
    problems: list[str]


# ----------------------------------------------------------------------------- inputs


def _pose_matrix(pose: Any) -> np.ndarray:
    if pose is None:
        raise ValueError("a camera frame has no pose")
    if isinstance(pose, dict):
        from gap_core.types import pose_to_matrix

        return pose_to_matrix(pose)
    m = np.asarray(pose, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"camera pose must be an Se3Pose or a 4x4 matrix, got {m.shape}")
    return m


def _frames(frames: Any) -> list[dict[str, Any]]:
    """A list of CameraFrame-like dicts from a list of them or a dict of stacked arrays."""
    if isinstance(frames, dict):
        rgb = frames.get("rgb")
        n = len(rgb) if rgb is not None else len(frames.get("depth") or [])
        depth = frames.get("depth")
        stamps = frames.get("timestamps", frames.get("times"))
        poses = frames.get("poses")
        out = []
        for i in range(n):
            out.append(
                {
                    "rgb": None if rgb is None else rgb[i],
                    "depth": None if depth is None else depth[i],
                    "intrinsics": frames.get("intrinsics"),
                    "pose": poses[i] if poses is not None else frames.get("pose"),
                    "timestamp": None if stamps is None else stamps[i],
                }
            )
        return out
    return [dict(f) for f in frames]


def _times(frames: list[dict[str, Any]], fps: float, problems: list[str]) -> np.ndarray:
    stamps = [f.get("timestamp", f.get("time")) for f in frames]
    if all(s is not None for s in stamps):
        t = np.asarray(stamps, dtype=np.float64)
        if t.max() > 1e12:  # ns
            t = t * 1e-9
        t = t - t[0]
        if np.all(np.diff(t) > 0):
            return t
        problems.append("frame timestamps are not increasing; frame rate used instead")
    if not fps or fps <= 0:
        fps = DEFAULT_FPS
        problems.append(f"no timestamps and no fps: {DEFAULT_FPS:g} fps assumed")
    return np.arange(len(frames), dtype=np.float64) / float(fps)


def _tile_size(tile_size: Any) -> tuple[float, float, float]:
    if isinstance(tile_size, dict):
        return (
            float(tile_size.get("height", TILE_SIZE[0])),
            float(tile_size.get("width", TILE_SIZE[1])),
            float(tile_size.get("thickness", TILE_SIZE[2])),
        )
    h, w, d = (float(v) for v in tile_size)
    return h, w, d


def _xyz(p: Any) -> np.ndarray:
    if isinstance(p, dict):
        return np.array([float(p["x"]), float(p["y"]), float(p.get("z", 0.0))])
    return np.asarray(p, dtype=np.float64).reshape(-1)[:3]


def _facing(slot: dict[str, Any]) -> np.ndarray:
    if "facing_deg" in slot:
        a = math.radians(float(slot["facing_deg"]))
        return np.array([math.cos(a), math.sin(a), 0.0])
    rot = slot.get("rotation")
    if rot is None:
        raise ValueError("a slot has neither rotation nor facing_deg")
    from gap_core.types import pose_to_matrix

    x = pose_to_matrix({"position": {"x": 0, "y": 0, "z": 0}, "rotation": rot})[:3, 0]
    x[2] = 0.0
    n = np.linalg.norm(x)
    if n < 1e-6:
        raise ValueError("a slot's facing (its x axis) is vertical")
    return x / n


def _slots(slots: Any, tiles: Any, table_z: float | None) -> tuple[list[dict], float]:
    """Slot centres (tile bottom centre) and facings in path order; tiles from a reader
    replace a slot's xy when one is within 15 mm."""
    items = []
    for s in slots:
        if isinstance(s, dict) and "pose" in s and "position" not in s:
            s = {**s["pose"], **{k: v for k, v in s.items() if k != "pose"}}
        items.append({"c": _xyz(s["position"] if isinstance(s, dict) else s), "f": _facing(s)})
    if table_z is None:
        table_z = float(np.median([it["c"][2] for it in items]))
    for it in items:
        it["c"][2] = table_z
    for t in tiles or []:
        c = t.get("center_world") if isinstance(t, dict) else t
        if not c:
            continue
        c = _xyz(c)
        k = min(range(len(items)), key=lambda i: float(np.hypot(*(items[i]["c"][:2] - c[:2]))))
        if np.hypot(*(items[k]["c"][:2] - c[:2])) <= 0.015:
            items[k]["c"][:2] = c[:2]
    return items, float(table_z)


# ----------------------------------------------------------------------------- tile geometry


class Tile:
    """One planned tile: its fall frame (x forward from the front bottom edge along the
    facing, y lateral, z up from the table) and what its fall can look like."""

    def __init__(self, idx: int, centre: np.ndarray, facing: np.ndarray, size, table_z, gap):
        self.idx = idx
        self.h, self.w, self.d = size
        self.f = facing
        self.l = np.array([-facing[1], facing[0], 0.0])
        self.table_z = table_z
        self.pivot = centre + facing * (self.d / 2.0)
        self.pivot[2] = table_z
        self.shift = np.zeros(2)  # forward, lateral correction from the first frames
        self.grid = np.arange(0.0, 90.0 + 1e-9, ANGLE_STEP)
        self.set_gap(gap)

    def set_gap(self, gap: float | None) -> None:
        """The face gap to the next tile (None for the last one): where the corridor
        ends and the angle at which the top meets the next tile."""
        self.gap = gap
        if gap is None:
            self.x_end, self.contact_deg = self.h + 0.002, 90.0
        else:
            self.x_end = min(max(gap - GAP_MARGIN, 0.004), self.h + 0.002)
            self.contact_deg = (
                math.degrees(math.asin(max(gap, 1e-4) / self.h)) if gap < self.h else 90.0
            )
        self.bins = np.arange(-self.d + 0.001, self.x_end, BIN) + BIN / 2.0
        self.profiles = _profiles(self.grid, self.bins, self.h, self.d)

    def to_local(self, pts: np.ndarray) -> np.ndarray:
        rel = pts - self.pivot
        x = rel @ self.f - self.shift[0]
        y = rel @ self.l - self.shift[1]
        return np.stack([x, y, pts[:, 2] - self.table_z], axis=1)

    def to_world(self, local: np.ndarray) -> np.ndarray:
        local = np.atleast_2d(local)
        x = local[:, 0] + self.shift[0]
        y = local[:, 1] + self.shift[1]
        out = self.pivot + np.outer(x, self.f) + np.outer(y, self.l)
        out[:, 2] = self.table_z + local[:, 2]
        return out

    def box_local(self, theta_deg: float) -> np.ndarray:
        """The 8 corners of the tile rotated by theta about its front bottom edge."""
        t = math.radians(theta_deg)
        c, s = math.cos(t), math.sin(t)
        xz = [(0.0, 0.0), (-self.d, 0.0), (-self.d, self.h), (0.0, self.h)]
        out = []
        for x, z in xz:
            rx, rz = x * c + z * s, -x * s + z * c
            for y in (-self.w / 2.0, self.w / 2.0):
                out.append((rx, y, rz))
        return np.array(out)

    def top_local(self, theta_deg: float) -> np.ndarray:
        t = math.radians(theta_deg)
        x, z = -self.d / 2.0, self.h
        return np.array(
            [x * math.cos(t) + z * math.sin(t), 0.0, -x * math.sin(t) + z * math.cos(t)]
        )


def _profiles(grid: np.ndarray, bins: np.ndarray, h: float, d: float) -> np.ndarray:
    """Upper height envelope over each bin of the tile rotated by each grid angle
    (0 where the tile is not: the table)."""
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


def _face_gap(a: Tile, b: Tile) -> float:
    """Distance along a's facing from a's front face to b's back face plane."""
    piv_a = a.pivot + a.f * a.shift[0] + a.l * a.shift[1]
    back_b = b.pivot + b.f * (b.shift[0] - b.d) + b.l * b.shift[1]
    return float(((back_b - piv_a) @ b.f) / max(float(a.f @ b.f), 0.2))


def _build_tiles(items: list[dict], size, table_z: float) -> list[Tile]:
    tiles = [Tile(i, it["c"].copy(), it["f"], size, table_z, None) for i, it in enumerate(items)]
    for a, b in zip(tiles, tiles[1:], strict=False):
        a.set_gap(_face_gap(a, b))
    return tiles


def _thresholds(
    tile: Tile, upright_deg: float, fallen_deg: float, mode: str
) -> tuple[float, float]:
    if mode == "rgb" and tile.gap is not None:
        # The rgb angle is the gap's covered fraction: confirm on a share of the gap
        # (placement error changes where a stopped tile rests, not that it covers it).
        conf = min(
            fallen_deg, math.degrees(math.asin(min(1.0, RGB_CONFIRM_COVER * tile.x_end / tile.h)))
        )
        return min(upright_deg, conf / 2.0), conf
    up = min(upright_deg, tile.contact_deg / 2.0)
    conf = min(fallen_deg, max(up + 5.0, tile.contact_deg - CONTACT_MARGIN_DEG))
    return up, conf


# ----------------------------------------------------------------------------- cameras


class Cam:
    def __init__(self, frame: dict[str, Any]):
        self.K = np.asarray(frame["intrinsics"], dtype=np.float64).reshape(3, 3)
        self.T = _pose_matrix(frame["pose"])
        self.R, self.t = self.T[:3, :3], self.T[:3, 3]

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pc = (np.atleast_2d(pts) - self.t) @ self.R
        z = pc[:, 2]
        zs = np.where(z > 1e-6, z, 1e-6)
        u = self.K[0, 0] * pc[:, 0] / zs + self.K[0, 2]
        v = self.K[1, 1] * pc[:, 1] / zs + self.K[1, 2]
        return np.stack([u, v], axis=1), z

    def same_as(self, other: Cam) -> bool:
        return bool(np.allclose(self.T, other.T, atol=1e-3) and np.allclose(self.K, other.K))


def _bbox(cam: Cam, pts: np.ndarray, shape: tuple[int, int], pad: int = 3):
    uv, z = cam.project(pts)
    uv = uv[z > 0]
    if len(uv) == 0:
        return None
    u0, v0 = np.floor(uv.min(axis=0)).astype(int) - pad
    u1, v1 = np.ceil(uv.max(axis=0)).astype(int) + pad
    u0, v0 = max(u0, 0), max(v0, 0)
    u1, v1 = min(u1, shape[1]), min(v1, shape[0])
    if u1 <= u0 or v1 <= v0:
        return None
    return u0, v0, u1, v1


def _corridor_corners(tile: Tile, z_top: float) -> np.ndarray:
    xs = (-tile.d - 0.004, tile.x_end + 0.004)
    ys = (-tile.w / 2.0, tile.w / 2.0)
    zs = (0.0, z_top)
    return tile.to_world(np.array([(x, y, z) for x in xs for y in ys for z in zs]))


def _lift(depth: np.ndarray, cam: Cam, box) -> np.ndarray:
    """Valid depth pixels of a crop as world points (geometry.mask_to_world_points'
    pinhole model and its 0.015-20 m bounds)."""
    u0, v0, u1, v1 = box
    z = np.asarray(depth, dtype=np.float64)[v0:v1, u0:u1]
    vv, uu = np.mgrid[v0:v1, u0:u1]
    ok = np.isfinite(z) & (z >= 0.015) & (z <= 20.0)
    z, uu, vv = z[ok], uu[ok], vv[ok]
    x = (uu - cam.K[0, 2]) * z / cam.K[0, 0]
    y = (vv - cam.K[1, 2]) * z / cam.K[1, 1]
    return np.stack([x, y, z], axis=1) @ cam.R.T + cam.t


# ----------------------------------------------------------------------------- depth signal


def _profile_obs(tile: Tile, local: np.ndarray, bias: float = 0.0):
    """Per bin: the top height (its 90th percentile, less the noise's share ``bias``)
    and whether an occluder is over it (its median over the tile's top)."""
    band = np.abs(local[:, 1]) <= tile.w / 2.0 - LATERAL_MARGIN
    x0 = tile.bins[0] - BIN / 2.0
    sel = band & (local[:, 0] >= x0) & (local[:, 0] < tile.bins[-1] + BIN / 2.0)
    loc = local[sel]
    nb = len(tile.bins)
    obs = np.full(nb, np.nan)
    med = np.full(nb, np.nan)
    occl = np.zeros(nb, bool)
    if len(loc) == 0:
        return obs, occl, med
    k = np.clip(((loc[:, 0] - x0) / BIN).astype(int), 0, nb - 1)
    order = np.argsort(k, kind="stable")
    k, z = k[order], loc[order, 2]
    starts = np.searchsorted(k, np.arange(nb))
    ends = np.searchsorted(k, np.arange(nb), side="right")
    z_occl = tile.h + OCCLUDER_ABOVE
    for b in range(nb):
        zz = z[starts[b] : ends[b]]
        if len(zz) == 0:
            continue
        med[b] = float(np.median(zz))
        if med[b] > z_occl:
            occl[b] = True
        else:
            # A camera from the side sees the table under a leaning tile beside the
            # tile itself: the bin's top, not its median, is the tile.
            obs[b] = min(float(np.percentile(zz, BIN_TOP_PCT)) - bias, z_occl)
    return obs, occl, med


def _fit_profile(tile: Tile, obs: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(obs)
    if ok.sum() < MIN_BINS:
        return math.nan, math.inf
    diff = obs[ok][None, :] - tile.profiles[:, ok]
    # Height over the model is cheap: a fallen neighbour lies on a fallen tile's back.
    res = np.minimum(np.where(diff > 0, OVER_WEIGHT * diff, -diff), FIT_CAP).mean(axis=1)
    k = int(np.argmin(res))
    return float(tile.grid[k]), float(res[k])


def _depth_signal(frames, cams, tiles: list[Tile]):
    n, T = len(tiles), len(frames)
    ang = np.full((n, T), np.nan)
    vis_raw = np.zeros((n, T))
    cost = np.full((n, T), np.inf)
    z_top = max(t.h for t in tiles) + 0.04
    corners = np.concatenate([_corridor_corners(t, z_top) for t in tiles])
    locals_first: list[list[np.ndarray]] = [[] for _ in tiles]
    clouds = []
    for fr, cam in zip(frames, cams, strict=True):
        depth = fr.get("depth")
        if depth is None:
            clouds.append(None)
            continue
        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[..., 0]
        box = _bbox(cam, corners, depth.shape)
        clouds.append(None if box is None else _lift(depth, cam, box))
    # The tile's own position from its first usable frames (top-surface points).
    for pts in clouds[: max(3, T // 10)]:
        if pts is None:
            continue
        for ti, tile in enumerate(tiles):
            loc = tile.to_local(pts)
            hi = loc[
                (loc[:, 2] > 0.8 * tile.h)
                & (loc[:, 2] < tile.h + OCCLUDER_ABOVE)
                & (np.abs(loc[:, 0] + tile.d / 2.0) < tile.d / 2.0 + MAX_SHIFT)
                & (np.abs(loc[:, 1]) < tile.w / 2.0 + MAX_SHIFT)
            ]
            if len(hi) >= 3:
                locals_first[ti].append(hi)
    for ti, tile in enumerate(tiles):
        if locals_first[ti]:
            hi = np.concatenate(locals_first[ti])
            dx = float(np.median(hi[:, 0])) + tile.d / 2.0
            lo, up = np.percentile(hi[:, 1], [5, 95])
            dy = float((lo + up) / 2.0) if up - lo > 0.6 * tile.w else 0.0
            tile.shift = np.clip([dx, dy], -MAX_SHIFT, MAX_SHIFT)
    # The gaps between the tiles as placed, not as planned.
    for a, b in zip(tiles, tiles[1:], strict=False):
        a.set_gap(_face_gap(a, b))
    # The depth noise's share of a bin's top: its 90th percentile over its median on
    # the table in front of the standing tiles (a camera property, pooled over the
    # tiles so a hand over one of them does not set it).
    excess = []
    for tile in tiles:
        for pts in clouds[: max(3, T // 10)]:
            if pts is None:
                continue
            obs, _, med = _profile_obs(tile, tile.to_local(pts))
            table = (tile.bins > 0.002) & np.isfinite(med) & (med < 0.25 * tile.h)
            excess.extend((obs[table] - med[table]).tolist())
    bias = np.full(n, float(np.median(excess)) if excess else 0.0)
    for fi, pts in enumerate(clouds):
        if pts is None:
            continue
        for ti, tile in enumerate(tiles):
            obs, occl, _ = _profile_obs(tile, tile.to_local(pts), bias[ti])
            vis_raw[ti, fi] = float(np.isfinite(obs).sum())
            if occl.sum() > 0.5 * len(occl):
                vis_raw[ti, fi] = 0.0
            a, c = _fit_profile(tile, obs)
            ang[ti, fi], cost[ti, fi] = a, c
    first = cost[:, : max(3, T // 10)]
    ref = np.array([np.median(r[np.isfinite(r)]) if np.isfinite(r).any() else 0.0 for r in first])
    usable = cost <= np.maximum(MAX_FIT_COST, 3.0 * ref)[:, None]
    return ang, vis_raw, usable


# ----------------------------------------------------------------------------- rgb signal


def _poly_mask(cam: Cam, world: np.ndarray, box) -> np.ndarray:
    u0, v0, u1, v1 = box
    m = np.zeros((v1 - v0, u1 - u0), np.uint8)
    uv, z = cam.project(world)
    if (z <= 0).any():
        return m.astype(bool)
    hull = cv2.convexHull((uv - [u0, v0]).astype(np.float32))
    cv2.fillConvexPoly(m, np.round(hull).astype(np.int32), 1)
    return m.astype(bool)


def _strip_world(tile: Tile, z: float) -> np.ndarray:
    lat = tile.w / 2.0 * RGB_STRIP_WIDTH
    return tile.to_world(
        np.array([(0.0, -lat, z), (0.0, lat, z), (tile.x_end, lat, z), (tile.x_end, -lat, z)])
    )


def _top_world(tile: Tile) -> np.ndarray:
    lat = tile.w / 2.0 * 0.8
    x0, x1 = -tile.d * 0.85, -tile.d * 0.15
    return tile.to_world(
        np.array([(x0, -lat, tile.h), (x0, lat, tile.h), (x1, lat, tile.h), (x1, -lat, tile.h)])
    )


def _rgb_signal(frames, cams, tiles: list[Tile], tile_colour, problems: list[str]):
    n, T = len(tiles), len(frames)
    ang = np.full((n, T), np.nan)
    vis_raw = np.zeros((n, T))
    shape = np.asarray(frames[0]["rgb"]).shape[:2]
    z_top = max(t.h for t in tiles) + 0.01
    corners = np.concatenate([_corridor_corners(t, z_top) for t in tiles])

    def masks_for(cam: Cam):
        box = _bbox(cam, corners, shape)
        if box is None:
            return None
        per = []
        for tile in tiles:
            strip = _poly_mask(cam, _strip_world(tile, tile.d), box)
            top = _poly_mask(cam, _top_world(tile), box)
            foot = _poly_mask(cam, _corridor_corners(tile, tile.h), box)
            per.append({"strip": strip, "top": top, "roi": foot | strip})
        return box, per

    first = masks_for(cams[0])
    if first is None:
        problems.append("the planned slots are outside the camera's view")
        return ang, vis_raw, np.zeros((n, T), bool)
    box0, per0 = first
    rgb0 = np.asarray(frames[0]["rgb"], dtype=np.float32)[..., :3]
    crop0 = rgb0[box0[1] : box0[3], box0[0] : box0[2]]
    tops = np.concatenate([crop0[m["top"]] for m in per0 if m["top"].any()] or [np.zeros((0, 3))])
    strips = np.concatenate([crop0[m["strip"]] for m in per0 if m["strip"].any()])
    c_tile = (
        np.asarray(tile_colour, dtype=np.float32)
        if tile_colour is not None
        else np.median(tops, axis=0)
    )
    c_table = np.median(strips, axis=0)
    r_table = max(
        RGB_MIN_CLASS_DIST, RGB_SPREAD * float(np.median(np.linalg.norm(strips - c_table, axis=1)))
    )
    tile_sq = float(c_tile @ c_tile) or 1.0
    s_table = float(c_table @ c_tile) / tile_sq
    table_off = float(np.linalg.norm(c_table - s_table * c_tile))
    if RGB_SHADE[0] <= s_table <= RGB_SHADE[1] and table_off < 2.0 * RGB_MIN_CLASS_DIST:
        problems.append(
            "the table looks like a shaded tile (same hue, "
            f"{table_off:.0f} off its colour line): the rgb signal cannot tell them apart"
        )

    def classify(crop):
        # Tile: any shade of the tile's colour (its faces turn darker as it falls);
        # table: within its own spread; anything else (a hand, the arm) occludes.
        s = (crop @ c_tile) / tile_sq
        off = np.linalg.norm(crop - s[..., None] * c_tile, axis=2)
        db = np.linalg.norm(crop - c_table, axis=2)
        tile_px = (off < RGB_MIN_CLASS_DIST * 1.5) & (s >= RGB_SHADE[0]) & (s <= RGB_SHADE[1])
        table_px = (db < r_table) & ~(tile_px & (off < db))
        tile_px &= ~table_px
        return tile_px, ~(tile_px | table_px)

    # The scene at rest per pixel: its first frame without an occluder over it.
    ref_tile, ref_unknown = classify(crop0)
    cache = (cams[0], box0, per0)
    for fi, (fr, cam) in enumerate(zip(frames, cams, strict=True)):
        if not cam.same_as(cache[0]):
            got = masks_for(cam)
            if got is None:
                continue
            cache = (cam, *got)
        _, box, per = cache
        static = box == box0
        crop = np.asarray(fr["rgb"], dtype=np.float32)[box[1] : box[3], box[0] : box[2], :3]
        tile_px, other = classify(crop)
        if static and ref_unknown.any():
            fill = ref_unknown & ~other
            ref_tile[fill] = tile_px[fill]
            ref_unknown &= other
        for ti, (tile, m) in enumerate(zip(tiles, per, strict=True)):
            roi = m["roi"]
            if not roi.any():
                continue
            vis_raw[ti, fi] = 1.0 - float(other[roi].mean())
            fresh = m["strip"] & ~other
            if static:
                fresh &= ~ref_tile
            if fresh.sum() < 3:
                continue
            cov = float(tile_px[fresh].mean())
            ang[ti, fi] = math.degrees(math.asin(min(1.0, cov * tile.x_end / tile.h)))
    return ang, vis_raw, np.isfinite(ang)


# ----------------------------------------------------------------------------- optional trackers


def _track_signal(ctx, frames, cams, tiles: list[Tile]):
    """track.points on each tile's top centre; the angle whose rotated top projects
    nearest the tracked pixel."""
    n, T = len(tiles), len(frames)
    queries = [cams[0].project(t.to_world(t.top_local(0.0)))[0][0].tolist() for t in tiles]
    rgbs = [np.asarray(f["rgb"]) for f in frames]
    reply = ctx.tool("track.points", frames=rgbs, query_points=queries, query_frame=0)
    tracks = reply.get("tracks") or []
    if len(tracks) != n:
        raise RuntimeError(f"track.points returned {len(tracks)} tracks for {n} queries")
    ang = np.full((n, T), np.nan)
    vis = np.zeros((n, T))
    for ti, (tile, tr) in enumerate(zip(tiles, tracks, strict=True)):
        path_cache: dict[int, np.ndarray] = {}
        for fi in range(T):
            if not bool(tr["visible"][fi]):
                continue
            cam = cams[fi]
            key = id(cam) if not cam.same_as(cams[0]) else -1
            if key not in path_cache:
                pts = tile.to_world(np.array([tile.top_local(a) for a in tile.grid]))
                path_cache[key] = cam.project(pts)[0]
            path = path_cache[key]
            px = np.asarray(tr["pixels"][fi], dtype=np.float64)
            dist = np.linalg.norm(path - px, axis=1)
            k = int(np.argmin(dist))
            tile_px = float(np.linalg.norm(path[0] - cam.project(tile.to_world([[0, 0, 0]]))[0][0]))
            if dist[k] <= TRACK_MAX_RESIDUAL_PX + 0.1 * tile_px:
                ang[ti, fi] = float(tile.grid[k])
                vis[ti, fi] = 1.0
    return ang, vis, np.isfinite(ang)


def _silhouette_fit(tile: Tile, cam: Cam, mask: np.ndarray):
    """Angle whose rotated-box projection best overlaps the mask, and the mask's area
    over that projection's."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 4:
        return math.nan, 0.0
    sweep = np.concatenate([tile.to_world(tile.box_local(a)) for a in (0.0, 45.0, 90.0)])
    box = _bbox(cam, sweep, mask.shape, pad=4)
    if box is None:
        return math.nan, 0.0
    u0, v0, u1, v1 = box
    m = mask[v0:v1, u0:u1]
    best = (-1.0, math.nan, 0.0)
    coarse = np.arange(0.0, 90.0 + 1e-9, 3.0)
    for grid in (coarse, None):
        if grid is None:
            a0 = best[1]
            grid = np.arange(max(0.0, a0 - 3.0), min(90.0, a0 + 3.0) + 1e-9, ANGLE_STEP)
        for a in grid:
            s = _poly_mask(cam, tile.to_world(tile.box_local(float(a))), box)
            inter = float((s & m).sum())
            union = float((s | m).sum())
            iou = inter / union if union else 0.0
            if iou > best[0]:
                best = (iou, float(a), float(m.sum()) / max(float(s.sum()), 1.0))
    return best[1], min(best[2], 1.5)


def _sam3_signal(ctx, frames, cams, tiles: list[Tile]):
    n, T = len(tiles), len(frames)
    ang = np.full((n, T), np.nan)
    vis = np.zeros((n, T))
    shape = np.asarray(frames[0]["rgb"]).shape[:2]
    for ti, tile in enumerate(tiles):
        b = _bbox(cams[0], tile.to_world(tile.box_local(0.0)), shape, pad=2)
        if b is None:
            continue
        init = ctx.tool(
            "sam3.tracker_init",
            image=np.asarray(frames[0]["rgb"]),
            box={"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])},
            object_name=f"domino {ti}",
        )
        tid = init.get("tracker_id") or ""
        if not tid:
            continue
        try:
            for fi in range(T):
                if fi == 0:
                    mask, conf = init.get("initial_mask"), 1.0
                else:
                    up = ctx.tool(
                        "sam3.tracker_update", tracker_id=tid, image=np.asarray(frames[fi]["rgb"])
                    )
                    mask = up.get("mask") if up.get("object_present", True) else None
                    conf = float(up.get("confidence", 1.0))
                if mask is None or conf <= 0.0:
                    continue
                mk = np.asarray(mask)
                mk = mk > (127 if mk.dtype == np.uint8 and mk.max() > 1 else 0)
                a, v = _silhouette_fit(tile, cams[fi], mk)
                ang[ti, fi], vis[ti, fi] = a, v
        finally:
            ctx.tool("sam3.tracker_close", tracker_id=tid)
    return ang, vis, np.isfinite(ang)


# ----------------------------------------------------------------------------- onsets


def topple_onset(
    angle: np.ndarray, usable: np.ndarray, upright_deg: float, fallen_deg: float, min_frames: int
) -> dict[str, Any] | None:
    """Upstream ``_topple_onset`` over a perceived angle trace: unusable frames are
    missing data (they neither extend nor break a run), a run cut by the clip's end
    needs ``END_RUN_MIN_FRAMES``, and there is no carry floor."""
    fall_idx = None
    run_start = None
    run_len = 0
    for t in range(len(angle)):
        if not usable[t]:
            continue
        if angle[t] < fallen_deg:
            run_start, run_len = None, 0
            continue
        if run_start is None:
            run_start, run_len = t, 0
        run_len += 1
        if run_len >= min_frames:
            fall_idx = run_start
            break
    confirmed = fall_idx is not None
    if fall_idx is None and run_start is not None and run_len >= END_RUN_MIN_FRAMES:
        # The run was still going when the clip ended (upstream: a false accept beats
        # a false reject for a topple in the last frames).
        fall_idx = run_start
    if fall_idx is None:
        return None
    last_up = None
    for t in range(fall_idx - 1, -1, -1):
        if usable[t] and angle[t] < upright_deg:
            last_up = t
            break
    if last_up is None:
        first = next(t for t in range(fall_idx + 1) if usable[t])
        return {
            "fall_frame": fall_idx,
            "onset_frame": first,
            "last_upright_frame": None,
            "next_seen_frame": first,
            "confirmed": confirmed,
        }
    nxt = next(t for t in range(last_up + 1, fall_idx + 1) if usable[t])
    return {
        "fall_frame": fall_idx,
        "onset_frame": last_up + 1,
        "last_upright_frame": last_up,
        "next_seen_frame": nxt,
        "confirmed": confirmed,
    }


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(len(x))
        for j in range(i + 1, len(x))
        if abs(x[j] - x[i]) > 1e-9
    ]
    return float(np.median(slopes)) if slopes else math.nan


# ----------------------------------------------------------------------------- the skill


def run(
    ctx: NodeContext,
    slots: list[Any] | None = None,
    frames: Any = None,
    first_slot: dict[str, Any] | None = None,
    pitch: float = 0.0,
    n_tiles: int = 0,
    tiles: list[Any] | None = None,
    tile_size: Any = TILE_SIZE,
    table_z: float | None = None,
    fps: float = 0.0,
    trigger_frame: int = -1,
    trigger_time: float = -1.0,
    signal: str = "auto",
    tracker: str = "none",
    tile_colour: list[float] | None = None,
    min_visibility: float = MIN_VISIBILITY,
    upright_deg: float = UPRIGHT_DEG,
    fallen_deg: float = FALLEN_DEG,
    min_frames: int = TOPPLE_MIN_FRAMES,
    camera_name: str | None = None,
    record_frames: int = 0,
) -> Output:
    out: Output = {
        "route": "unverifiable",
        "detail": "",
        "n_tiles": 0,
        "n_fell": 0,
        "stop_index": -1,
        "last_fallen_index": -1,
        "stop_point": {},
        "wave_speed": 0.0,
        "wave_rate": 0.0,
        "first_onset_time": -1.0,
        "last_onset_time": -1.0,
        "signal": "",
        "tracker_fallback": "",
        "tiles": [],
        "intervals": [],
        "angles": [],
        "visibility": [],
        "times": [],
        "problems": [],
    }
    problems = out["problems"]

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        logger.info("verifying-a-domino-cascade-from-video: %s (%s)", route, detail)
        return out

    if not slots and first_slot and n_tiles >= 2 and pitch > 0:
        f = _facing(first_slot)
        c0 = _xyz(first_slot["position"])
        slots = [
            {
                "position": dict(zip("xyz", (c0 + f * k * pitch).tolist(), strict=True)),
                "rotation": first_slot["rotation"],
            }
            for k in range(int(n_tiles))
        ]
    if not slots:
        return finish(
            "unverifiable",
            "no slots: pass the planned slot list, or first_slot + pitch + n_tiles (planning-a-domino-layout)",
        )
    if frames is None and record_frames > 0:
        frames = []
        for _ in range(int(record_frames)):
            obs = ctx.tool("robot.get_observation")
            cams_obs = [c for c in obs.get("cameras") or [] if c.get("rgb") is not None]
            cam = next(
                (c for c in cams_obs if not camera_name or c.get("name") == camera_name), None
            )
            if cam is None:
                return finish("unverifiable", f"no camera {camera_name!r} in the observation")
            frames.append({**cam, "timestamp": cam.get("timestamp", time.monotonic())})
    if frames is None:
        return finish("unverifiable", "no frames: pass a recorded clip (or record_frames > 0)")
    frames = _frames(frames)
    if len(frames) < 2:
        return finish("unverifiable", f"{len(frames)} frame(s): a clip needs at least two")

    size = _tile_size(tile_size)
    items, tz = _slots(slots, tiles, table_z)
    tl = _build_tiles(items, size, tz)
    n, T = len(tl), len(frames)
    out["n_tiles"] = n
    times = _times(frames, fps, problems)
    out["times"] = [round(float(t), 5) for t in times]
    cams = [Cam(f) for f in frames]
    cams = [cams[0] if c.same_as(cams[0]) else c for c in cams]

    has_depth = all(f.get("depth") is not None for f in frames)
    mode = signal if signal != "auto" else ("depth" if has_depth else "rgb")
    if mode == "depth" and not has_depth:
        return finish("unverifiable", "signal 'depth' asked but some frames have no depth")
    if mode not in ("depth", "rgb"):
        return finish("unverifiable", f"unknown signal {signal!r} (auto, depth or rgb)")

    used = mode
    ang = vis_raw = usable = None
    if tracker not in ("none", "", None):
        try:
            if tracker == "track":
                ang, vis_raw, usable = _track_signal(ctx, frames, cams, tl)
            elif tracker == "sam3":
                ang, vis_raw, usable = _sam3_signal(ctx, frames, cams, tl)
            else:
                raise ValueError(f"unknown tracker {tracker!r} (none, track or sam3)")
            used = tracker
        except Exception as exc:
            out["tracker_fallback"] = f"{tracker} unavailable ({exc}); {mode} signal used"
            problems.append(out["tracker_fallback"])
            ang = None
    if ang is None:
        if mode == "depth":
            ang, vis_raw, usable = _depth_signal(frames, cams, tl)
        else:
            ang, vis_raw, usable = _rgb_signal(frames, cams, tl, tile_colour, problems)
    out["signal"] = used

    # Visibility against the tile's usual visibility (gate-dropped frames are missing).
    vis = np.zeros_like(vis_raw)
    for ti in range(n):
        ref = float(np.percentile(vis_raw[ti], VIS_REFERENCE_PCT)) if T else 0.0
        vis[ti] = vis_raw[ti] / ref if ref > 0 else 0.0
    usable = usable & (vis >= min_visibility) & np.isfinite(ang)

    trig = -1
    if trigger_frame >= 0:
        trig = int(trigger_frame)
    elif trigger_time >= 0:
        trig = int(np.searchsorted(times, float(trigger_time)))

    per: list[dict[str, Any]] = []
    rel = np.full_like(ang, np.nan)
    for ti, tile in enumerate(tl):
        up, conf = _thresholds(tile, upright_deg, fallen_deg, used)
        seen = np.flatnonzero(usable[ti])
        rec: dict[str, Any] = {
            "index": ti,
            "state": "unknown",
            "onset_frame": -1,
            "onset_time": -1.0,
            "onset_uncertainty_frames": 0,
            "fall_frame": -1,
            "confirmed": False,
            "inferred": False,
            "start_angle": None,
            "max_angle": None,
            "final_angle": None,
            "usable_frames": int(len(seen)),
            "upright_deg": round(up, 2),
            "fallen_deg": round(conf, 2),
            "contact_deg": round(tile.contact_deg, 2),
            "position": {
                k: float(v)
                for k, v in zip("xyz", tile.to_world([[-tile.d / 2.0, 0.0, 0.0]])[0], strict=True)
            },
            "face_gap": None if tile.gap is None else round(tile.gap, 4),
        }
        per.append(rec)
        if len(seen) == 0:
            continue
        a0 = float(np.median(ang[ti, seen[: min(3, len(seen))]]))
        rec["start_angle"] = round(a0, 2)
        first_seen_before_wave = trig < 0 or seen[0] <= trig
        if a0 >= conf and first_seen_before_wave:
            rec["state"] = "down_at_start"
            continue
        if a0 >= up and first_seen_before_wave:
            problems.append(f"tile {ti} leaned {a0:.0f} deg in its first frames")
        # Each tile against its own first frames, unless those already lean.
        base = a0 if a0 < up else 0.0
        rel[ti] = ang[ti] - base
        r = rel[ti]
        rec["max_angle"] = round(float(np.nanmax(r[seen])), 2)
        rec["final_angle"] = round(float(r[seen[-1]]), 2)
        ev = topple_onset(r, usable[ti], up, conf, int(min_frames))
        if ev is not None:
            rec.update(
                state="fell",
                onset_frame=int(ev["onset_frame"]),
                fall_frame=int(ev["fall_frame"]),
                confirmed=bool(ev["confirmed"]),
                onset_uncertainty_frames=int(ev["next_seen_frame"] - ev["onset_frame"]),
            )
            lu, ns = ev["last_upright_frame"], ev["next_seen_frame"]
            if lu is not None and ns > lu:
                # Sub-frame: where the angle crosses the upright band between the
                # last upright frame and the next one seen.
                a_lo, a_hi = float(r[lu]), float(r[ns])
                frac = (up - a_lo) / (a_hi - a_lo) if a_hi > a_lo else 1.0
                rec["onset_time"] = float(
                    times[lu] + np.clip(frac, 0.0, 1.0) * (times[ns] - times[lu])
                )
            else:
                rec["onset_time"] = float(times[ev["onset_frame"]])
            if not ev["confirmed"]:
                problems.append(
                    f"tile {ti} is past {conf:.0f} deg only in the clip's last "
                    f"{T - ev['fall_frame']} usable frame(s): counted as fallen, unconfirmed"
                )

    out["angles"] = [
        [None if not np.isfinite(v) else round(float(v), 1) for v in row] for row in rel
    ]
    out["visibility"] = [[round(float(v), 3) for v in row] for row in vis]

    # Standing needs the tile seen upright after the wave could have reached it.
    for ti, rec in enumerate(per):
        if rec["state"] != "unknown":
            continue
        prev = [p["onset_frame"] for p in per[:ti] if p["state"] == "fell"]
        since = max(prev) if prev else max(trig, 0)
        seen_after = [t for t in np.flatnonzero(usable[ti]) if t >= since]
        if len(seen_after) >= int(min_frames) and rel[ti, seen_after[-1]] < float(
            rec["fallen_deg"]
        ):
            rec["state"] = "standing"
    # A tile nobody saw fall but with a fallen tile after it: the wave went through it.
    last_fell = max((p["index"] for p in per if p["state"] == "fell"), default=0)
    for rec in per[:last_fell]:
        if rec["state"] == "unknown":
            rec["state"], rec["inferred"] = "fell", True
    out["tiles"] = per

    fell = [p for p in per if p["state"] == "fell"]
    timed = [p for p in fell if not p["inferred"]]
    out["n_fell"] = len(fell)
    if timed:
        out["first_onset_time"] = round(min(p["onset_time"] for p in timed), 5)
        out["last_onset_time"] = round(max(p["onset_time"] for p in timed), 5)
    frame_dt = float(np.median(np.diff(times))) if T > 1 else 1.0 / DEFAULT_FPS
    if len(timed) >= 2:
        idx = np.array([p["index"] for p in timed], dtype=np.float64)
        arc = np.concatenate(
            [
                [0.0],
                np.cumsum(
                    [np.linalg.norm(tl[i + 1].pivot[:2] - tl[i].pivot[:2]) for i in range(n - 1)]
                ),
            ]
        )
        s = arc[idx.astype(int)]
        t = np.array([p["onset_time"] for p in timed])
        slope_s = _theil_sen(s, t)
        slope_i = _theil_sen(idx, t)
        if np.isfinite(slope_s) and slope_s > 0:
            out["wave_speed"] = round(1.0 / slope_s, 4)
        if np.isfinite(slope_i) and slope_i > 0:
            out["wave_rate"] = round(1.0 / slope_i, 3)
        pairs = [
            (a, b) for a, b in zip(timed, timed[1:], strict=False) if b["index"] == a["index"] + 1
        ]
        out["intervals"] = [round(b["onset_time"] - a["onset_time"], 5) for a, b in pairs]
        short = [x for x in out["intervals"] if x < MIN_INTERVAL_FRAMES * frame_dt]
        if short:
            problems.append(
                f"{len(short)} successive onset(s) under {MIN_INTERVAL_FRAMES:g} frames apart: "
                "their order is not resolvable at this frame rate (record at 60 fps or more)"
            )

    down = [p["index"] for p in per if p["state"] == "down_at_start"]
    if down:
        return finish(
            "unverifiable",
            f"tile(s) {down} already down in their first frames: the line was disturbed before the clip",
        )
    if trig >= 0:
        early = [
            p
            for p in timed
            if p["onset_frame"] < trig - ORDER_TOLERANCE_FRAMES - p["onset_uncertainty_frames"]
        ]
        if early:
            names = ", ".join(f"{p['index']} (frame {p['onset_frame']})" for p in early)
            return finish(
                "fell_before_trigger",
                f"tile(s) {names} started falling before the trigger at frame {trig}",
            )
    if not fell:
        if per[0]["state"] == "standing":
            return finish("not_triggered", "no tile fell; the first tile is standing")
        return finish(
            "unverifiable",
            "no fall seen and the first tile is not seen standing (occluded or out of view)",
        )
    if per[0]["state"] != "fell":
        return finish(
            "irregular", f"tile(s) {[p['index'] for p in fell]} fell but the first tile did not"
        )
    k = next((p["index"] for p in per if p["state"] != "fell"), n)
    for a, b in zip(timed, timed[1:], strict=False):
        tol = ORDER_TOLERANCE_FRAMES + a["onset_uncertainty_frames"] + b["onset_uncertainty_frames"]
        if b["onset_time"] < a["onset_time"] - tol * frame_dt:
            return finish(
                "irregular",
                f"tile {b['index']} started falling {1e3 * (a['onset_time'] - b['onset_time']):.0f} ms "
                f"before tile {a['index']}: not one wave from the first tile",
            )
    if k == n:
        speed = f", wave {out['wave_speed']:.2f} m/s" if out["wave_speed"] else ""
        return finish("all_fell", f"all {n} tiles fell{speed}")
    out["stop_index"], out["last_fallen_index"] = k, k - 1
    out["stop_point"] = per[k]["position"]
    if per[k]["state"] == "unknown":
        return finish(
            "unverifiable",
            f"tiles 0-{k - 1} fell; tile {k} is not seen well enough after the wave to tell",
        )
    later = [p["index"] for p in per[k + 1 :] if p["state"] == "fell"]
    if later:
        return finish("irregular", f"tile {k} stood but tile(s) {later} fell")
    stuck = [p["index"] for p in per[k + 1 :] if p["state"] == "unknown"]
    tail = f"; tiles {stuck} after it not seen well" if stuck else ""
    return finish("stopped_at", f"tiles 0-{k - 1} fell; the wave stopped at tile {k}{tail}")
