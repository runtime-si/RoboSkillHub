# Adapted from BrickGPT (https://github.com/AvaLovelace1/BrickGPT) at commit
# 9c53c7cb65f9de0df89562cc74b2d2e270fbcab4: src/brickgpt/data/brick_structure.py
# (Brick, Brick.from_ldr, BrickStructure.brick_in_bounds, has_collisions,
# brick_floats, is_connected), src/brickgpt/stability_analysis/connectivity_analysis.py
# (connectivity_score, _connected, _connected_to_ground) and the masses of
# src/brickgpt/data/brick_library.json; MIT License, Copyright (c) 2025 Ava Pun
# (LICENSE.brickgpt beside this bundle).
# Changes: the out-of-bounds, voxel-collision, local floating and ground-connected
# component rules are BrickGPT's; the world is a rectangular plate of
# plate_size studs and max_layers layers instead of a 20^3 cube; the connectivity
# graph is a breadth-first search over brick indices (networkx and the Brick
# objects as node keys are dropped, so two identical bricks stay two nodes);
# LDraw input accepts several parts per 0 STEP, any of the four upright
# quarter-turn rotations and more brick part numbers, and refuses what is not on
# the grid instead of truncating it. Added: per-step checks (each brick connected
# to the plate, and not under an already placed brick, when its turn comes),
# single-stud joints that carry an overhang, and a stability score. The stability
# score keeps the idea of StableLego / BrickGPT's stability_analysis.py (a brick
# is stable when static equilibrium needs no stud clutch over its capacity T,
# T = 100 g x g as there) but is a linear program solved with scipy's HiGHS
# instead of Gurobi: vertical contact forces only, compression at the corners of
# each connected stud cell, clutch tension at the stud, minimising the largest
# clutch tension. Results are routed instead of returned as booleans.
"""Check a brick design on a stud grid before it is built: plate bounds, collisions, connectivity, per-step support, single-stud overhangs and stability."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PLATE_SIZE = "32x32"
"""Default plate, studs along x by y: the standard 32 x 32 stud LEGO baseplate.
BrickGPT used a 20 x 20 grid (world_dim 20)."""

MAX_LAYERS = 20
"""Layers a brick may sit in (0 .. MAX_LAYERS - 1): BrickGPT's world_dim."""

CLUTCH_N = 0.98
"""Stud clutch capacity T [N]: BrickGPT's StabilityConfig T = 100 (grams) x g / 1000,
from StableLego. Measured for LEGO; for DUPLO pass your own."""

GRAVITY = 9.8
"""[m/s^2], BrickGPT's StabilityConfig g."""

STABLE_BELOW = 1.0
"""A structure is stable when its largest clutch utilisation is under this
(BrickGPT's is_stable: stability_scores().max() < 1)."""

BRICK_IDS: dict[int, tuple[int, int]] = {
    2: (2, 4),
    3: (2, 6),
    4: (1, 8),
    5: (1, 4),
    6: (1, 6),
    7: (1, 4),
    8: (1, 4),
    9: (1, 2),
    10: (1, 1),
    11: (1, 2),
    12: (2, 2),
}
"""Brick ids of BrickGPT's brick_library.json (2-6, 9, 10, 12) and the ICL
Robotic_Lego_Manipulation config/lego_library.json (adds 7, 8, 11): id ->
(height, width) studs, height along x at ori 0."""

MASS_G: dict[tuple[int, int], float] = {
    (2, 4): 2.16,
    (2, 6): 3.23,
    (1, 8): 3.03,
    (1, 4): 1.57,
    (1, 6): 2.28,
    (1, 2): 0.81,
    (1, 1): 0.43,
    (2, 2): 1.15,
}
"""Brick masses [g] from BrickGPT's brick_library.json (short side first)."""

MASS_FIT = (0.1033, 0.2740, -0.2558)
"""Mass [g] of a size not in MASS_G: a*studs + b*(short + long) + c, a least-squares
fit to MASS_G (largest residual 0.10 g, on the 2x2)."""

LDRAW_PARTS: dict[str, tuple[int, int]] = {
    "3005": (1, 1),
    "3004": (2, 1),
    "3622": (3, 1),
    "3010": (4, 1),
    "3009": (6, 1),
    "3008": (8, 1),
    "6111": (10, 1),
    "6112": (12, 1),
    "3003": (2, 2),
    "3002": (3, 2),
    "3001": (4, 2),
    "2456": (6, 2),
    "3007": (8, 2),
    "3006": (10, 2),
}
"""LDraw basic bricks (one brick tall, 24 LDU): part number -> (studs along the
part's own x, studs along its z). BrickGPT's library has 3001, 2456, 3008, 3010,
3009, 3004, 3005 and 3003; the rest are the same family. Plates, slopes and
anything else are refused."""

LDU_STUD = 20.0
LDU_BRICK = 24.0
"""LDraw units per stud and per brick height (BrickGPT's to_ldr / from_ldr)."""

GRID_TOLERANCE_LDU = 0.5
"""An LDraw position further than this from the stud grid is refused [LDU]."""

ROUTES = (
    "invalid_input",
    "out_of_plate",
    "collision",
    "floating",
    "step_unsupported",
    "step_blocked",
    "weakly_connected",
    "unstable",
    "valid",
)
"""Every route, in the order they are tried: the first problem found is the route,
every list in the output is filled regardless."""


class Output(TypedDict):
    route: str
    detail: str
    n_bricks: int
    n_steps: int
    out_of_plate: list[int]
    collisions: list[list[int]]
    floating: list[int]
    isolated: list[int]
    step_unsupported: list[int]
    step_blocked: list[list[int]]
    weak_joints: list[list[int]]
    weak_bricks: list[int]
    weak_at_step: list[int]
    stability_max: float
    stability_per_brick: list[float]
    worst_brick: int
    unstable_step: int
    bricks: list[dict[str, Any]]
    elapsed_s: float


@dataclass(frozen=True)
class Brick:
    """BrickGPT's Brick: a one-layer rectangular brick. h studs along x, w along y,
    lowest corner cell (x, y), layer z (0 = on the plate)."""

    h: int
    w: int
    x: int
    y: int
    z: int

    @property
    def area(self) -> int:
        return self.h * self.w

    @property
    def slice_2d(self) -> tuple[slice, slice]:
        return slice(self.x, self.x + self.h), slice(self.y, self.y + self.w)

    def cells(self) -> list[tuple[int, int]]:
        return [
            (i, j) for i in range(self.x, self.x + self.h) for j in range(self.y, self.y + self.w)
        ]


# ----------------------------------------------------------------- BrickGPT rules


def _brick_in_bounds(b: Brick, nx: int, ny: int, layers: int) -> bool:
    """BrickStructure.brick_in_bounds, on an nx x ny plate."""
    return b.x >= 0 and b.x + b.h <= nx and b.y >= 0 and b.y + b.w <= ny and 0 <= b.z < layers


def _overlap(b1: Brick, b2: Brick) -> int:
    """Studs the two footprints share in the x-y plane."""
    ox = min(b1.x + b1.h, b2.x + b2.h) - max(b1.x, b2.x)
    oy = min(b1.y + b1.w, b2.y + b2.w) - max(b1.y, b2.y)
    return max(ox, 0) * max(oy, 0)


def _connected(b1: Brick, b2: Brick) -> bool:
    """connectivity_analysis._connected: one directly on top of the other, footprints overlapping."""
    if b1.z > b2.z:
        b1, b2 = b2, b1
    if b1.z != b2.z - 1:
        return False
    return b1.x < b2.x + b2.h and b1.x + b1.h > b2.x and b1.y < b2.y + b2.w and b1.y + b1.w > b2.y


def _connected_to_ground(b: Brick) -> bool:
    return b.z == 0


def _occupancy(bricks: list[Brick]) -> tuple[np.ndarray, tuple[int, int, int]]:
    """BrickStructure.voxel_occupancy (count of bricks per cell), over the bricks'
    bounding box so out-of-plate bricks fit too."""
    lo = (
        min(b.x for b in bricks),
        min(b.y for b in bricks),
        min(b.z for b in bricks),
    )
    hi = (
        max(b.x + b.h for b in bricks),
        max(b.y + b.w for b in bricks),
        max(b.z + 1 for b in bricks),
    )
    occ = np.zeros((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), dtype=int)
    for b in bricks:
        occ[b.x - lo[0] : b.x - lo[0] + b.h, b.y - lo[1] : b.y - lo[1] + b.w, b.z - lo[2]] += 1
    return occ, lo


def _brick_floats(b: Brick, occ: np.ndarray, lo: tuple[int, int, int]) -> bool:
    """BrickStructure.brick_floats: not on the ground and nothing directly below or above."""
    if b.z == 0:
        return False
    sx = slice(b.x - lo[0], b.x - lo[0] + b.h)
    sy = slice(b.y - lo[1], b.y - lo[1] + b.w)
    k = b.z - lo[2]
    if k - 1 >= 0 and np.any(occ[sx, sy, k - 1]):
        return False
    if k + 1 < occ.shape[2] and np.any(occ[sx, sy, k + 1]):
        return False
    return True


def _graph(bricks: list[Brick]) -> dict[int, set[int]]:
    """connectivity_score's graph: node -1 is the ground, an edge per _connected pair."""
    adj: dict[int, set[int]] = {-1: set()}
    for i in range(len(bricks)):
        adj[i] = set()
    for i, b in enumerate(bricks):
        if _connected_to_ground(b):
            adj[-1].add(i)
            adj[i].add(-1)
    by_layer: dict[int, list[int]] = {}
    for i, b in enumerate(bricks):
        by_layer.setdefault(b.z, []).append(i)
    for i, b in enumerate(bricks):
        for j in by_layer.get(b.z + 1, ()):
            if _connected(b, bricks[j]):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _ground_component(adj: dict[int, set[int]], cut: tuple[int, int] | None = None) -> set[int]:
    """nx.node_connected_component(graph, 'ground'), optionally with one edge removed."""
    seen = {-1}
    queue = deque([-1])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if cut is not None and {u, v} == set(cut):
                continue
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return seen


# ----------------------------------------------------------------- added checks


def _weak_joints(
    bricks: list[Brick], adj: dict[int, set[int]], grounded: set[int]
) -> tuple[list[list[int]], list[int]]:
    """Single-stud joints carrying an overhang: an edge of one shared stud whose removal
    cuts some bricks off the plate, where the cut-off bricks cover more than that one
    stud's column. Returns the joints [lower, upper] (-1 = the plate) and the bricks
    hanging on them."""
    joints: list[list[int]] = []
    hanging: set[int] = set()
    edges: list[tuple[int, int, tuple[int, int]]] = []
    for i in sorted(grounded - {-1}):
        b = bricks[i]
        for j in adj[i]:
            if j == -1:
                if b.area == 1:
                    edges.append((-1, i, (b.x, b.y)))
                continue
            up = bricks[j]
            if up.z == b.z + 1 and _overlap(b, up) == 1:
                edges.append((i, j, (max(b.x, up.x), max(b.y, up.y))))
    for lower, upper, stud in edges:
        cut_off = grounded - _ground_component(adj, cut=(lower, upper))
        if cut_off and any(c != stud for k in cut_off for c in bricks[k].cells()):
            joints.append([lower, upper])
            hanging |= cut_off
    return joints, sorted(hanging)


def _mass_g(b: Brick) -> float:
    key = (min(b.h, b.w), max(b.h, b.w))
    if key in MASS_G:
        return MASS_G[key]
    a, c, d = MASS_FIT
    return a * b.area + c * (b.h + b.w) + d


def _stability(
    bricks: list[Brick], clutch_n: float, mass_scale: float
) -> tuple[float, list[float], str]:
    """Largest clutch utilisation (tension / clutch_n) the structure needs to stand.

    Every stud cell where a brick sits on another (or on the plate) is a joint with
    compression c >= 0 at its four corners and clutch tension t >= 0 at the stud.
    Each brick's vertical force and its two tipping moments must balance its weight.
    The linear program minimises the largest t (the most favourable load path; if it
    is within capacity the structure can stand, by the lower-bound theorem).
    Assumes every brick is connected to the plate."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    joints: list[tuple[int, int, int, int]] = []  # (upper, lower or -1, cx, cy)
    by_layer: dict[int, list[int]] = {}
    for i, b in enumerate(bricks):
        by_layer.setdefault(b.z, []).append(i)
    for i, b in enumerate(bricks):
        if b.z == 0:
            joints += [(i, -1, cx, cy) for cx, cy in b.cells()]
        for j in by_layer.get(b.z - 1, ()):
            lo = bricks[j]
            for cx in range(max(b.x, lo.x), min(b.x + b.h, lo.x + lo.h)):
                for cy in range(max(b.y, lo.y), min(b.y + b.w, lo.y + lo.w)):
                    joints.append((i, j, cx, cy))
    n_b = len(bricks)
    n_v = 5 * len(joints) + 1
    s_col = n_v - 1
    centre = [(b.x + b.h / 2.0, b.y + b.w / 2.0) for b in bricks]
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def put(brick: int, col: int, force: float, px: float, py: float) -> None:
        cx, cy = centre[brick]
        for r, v in ((0, force), (1, force * (px - cx)), (2, force * (py - cy))):
            if v != 0.0:
                rows.append(3 * brick + r)
                cols.append(col)
                vals.append(v)

    corners = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    for k, (up, lo, cx, cy) in enumerate(joints):
        for m, (dx, dy) in enumerate(corners):
            put(up, 5 * k + m, 1.0, cx + dx, cy + dy)
            if lo >= 0:
                put(lo, 5 * k + m, -1.0, cx + dx, cy + dy)
        put(up, 5 * k + 4, -1.0, cx + 0.5, cy + 0.5)
        if lo >= 0:
            put(lo, 5 * k + 4, 1.0, cx + 0.5, cy + 0.5)
    a_eq = coo_matrix((vals, (rows, cols)), shape=(3 * n_b, n_v)).tocsr()
    b_eq = np.zeros(3 * n_b)
    for i, b in enumerate(bricks):
        b_eq[3 * i] = _mass_g(b) * mass_scale / 1000.0 * GRAVITY
    t_cols = np.arange(len(joints)) * 5 + 4
    a_ub = coo_matrix(
        (
            np.concatenate([np.ones(len(joints)), -np.ones(len(joints))]),
            (
                np.tile(np.arange(len(joints)), 2),
                np.concatenate([t_cols, np.full(len(joints), s_col)]),
            ),
        ),
        shape=(len(joints), n_v),
    ).tocsr()
    cost = np.zeros(n_v)
    cost[t_cols] = 1e-6
    cost[s_col] = 1.0
    res = linprog(
        cost,
        A_ub=a_ub,
        b_ub=np.zeros(len(joints)),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method="highs",
    )
    if res.status != 0:
        return math.inf, [math.inf] * n_b, f"the stability program did not solve ({res.message})"
    t = res.x[t_cols]
    per = [0.0] * n_b
    for k, (up, lo, _, _) in enumerate(joints):
        per[up] = max(per[up], float(t[k] / clutch_n))
        if lo >= 0:
            per[lo] = max(per[lo], float(t[k] / clutch_n))
    return float(res.x[s_col] / clutch_n), per, ""


# ----------------------------------------------------------------- input parsing


def _size(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*", value)
        if not m:
            raise ValueError(f"size {value!r} is not 'AxB'")
        a, b = int(m.group(1)), int(m.group(2))
    else:
        a, b = (int(v) for v in value)
    if a < 1 or b < 1:
        raise ValueError(f"size {value!r} must be at least 1x1")
    return a, b


def _brick_from_dict(d: dict[str, Any], first_layer: int) -> tuple[Brick, str]:
    """A hub brick {cell: [x, y, layer], size: 'AxB', rotation: 0|90|180|270}, or the
    BrickGPT / ICL form {brick_id, x, y, z, ori}. Size A runs along the plate's x at
    rotation 0 (ICL's ori 0: height along x); 90 / 270 swap the axes."""
    if "cell" in d:
        x, y, layer = (int(v) for v in d["cell"])
    else:
        x, y = int(d["x"]), int(d["y"])
        layer = int(d["layer"] if "layer" in d else d["z"])
    if "size" in d:
        a, b = _size(d["size"])
        label = f"{a}x{b}"
    elif "brick_id" in d:
        bid = int(d["brick_id"])
        if bid not in BRICK_IDS:
            raise ValueError(f"unknown brick_id {bid} (known: {sorted(BRICK_IDS)})")
        a, b = BRICK_IDS[bid]
        label = f"{a}x{b}"
    else:
        raise ValueError(f"brick {d!r} has no size or brick_id")
    if "rotation" in d:
        rot = int(round(float(d["rotation"]))) % 360
        if rot not in (0, 90, 180, 270):
            raise ValueError(f"rotation {d['rotation']!r} is not a quarter turn")
        turned = rot in (90, 270)
    else:
        ori = int(d.get("ori", 0))
        if ori not in (0, 1):
            raise ValueError(f"ori {ori!r} is not 0 or 1")
        turned = ori == 1
    if turned:
        a, b = b, a
    return Brick(h=a, w=b, x=x, y=y, z=layer - first_layer), label


def _parse_ldraw(
    text: str, offset: tuple[int, int, int], extra_parts: dict[str, Any]
) -> tuple[list[Brick], list[int], list[str]]:
    """BrickGPT's Brick.from_ldr / BrickStructure.from_ldr, generalised: parts between
    '0 STEP' lines share a step; x = x0/20 - h/2, y = z0/20 - w/2, layer = -y0/24."""
    parts = dict(LDRAW_PARTS)
    for k, v in (extra_parts or {}).items():
        parts[str(k).lower().removesuffix(".dat")] = _size(v)
    bricks: list[Brick] = []
    steps: list[int] = []
    labels: list[str] = []
    step = 0
    for n, raw in enumerate(text.splitlines(), start=1):
        tok = raw.split()
        if not tok:
            continue
        if tok[0] == "0":
            if len(tok) > 1 and tok[1].upper() == "STEP":
                step += 1
            continue
        if tok[0] != "1":
            continue
        if len(tok) < 15:
            raise ValueError(f"LDraw line {n} is ill-formed: {raw.strip()!r}")
        x0, y0, z0 = (float(v) for v in tok[2:5])
        r = np.array([float(v) for v in tok[5:14]]).reshape(3, 3)
        part = " ".join(tok[14:]).lower().replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".dat")
        if part not in parts:
            raise ValueError(f"LDraw line {n}: part {part!r} is not a supported one-layer brick")
        lx, lz = parts[part]
        if not (
            np.allclose(np.abs(r), np.round(np.abs(r)), atol=1e-3) and abs(r[1, 1] - 1.0) < 1e-3
        ):
            raise ValueError(f"LDraw line {n}: rotation is not an upright quarter turn")
        h = int(round(abs(r[0, 0]) * lx + abs(r[0, 2]) * lz))
        w = int(round(abs(r[2, 0]) * lx + abs(r[2, 2]) * lz))
        fx = x0 / LDU_STUD - h * 0.5
        fy = z0 / LDU_STUD - w * 0.5
        fz = -y0 / LDU_BRICK
        for value, unit, axis in ((fx, LDU_STUD, "x"), (fy, LDU_STUD, "z"), (fz, LDU_BRICK, "y")):
            if abs(value - round(value)) * unit > GRID_TOLERANCE_LDU:
                raise ValueError(f"LDraw line {n}: {axis} is off the stud grid")
        bricks.append(
            Brick(
                h=h,
                w=w,
                x=int(round(fx)) + offset[0],
                y=int(round(fy)) + offset[1],
                z=int(round(fz)) + offset[2],
            )
        )
        steps.append(step)
        labels.append(f"{lx}x{lz}")
    if step == 0:
        steps = list(range(len(bricks)))
    return bricks, steps, labels


def _plate(value: Any) -> tuple[int, int]:
    return _size(value)


# ----------------------------------------------------------------- run


def run(
    ctx: NodeContext,
    bricks: list | str | None = None,
    ldraw: str = "",
    plate_size: str = PLATE_SIZE,
    max_layers: int = MAX_LAYERS,
    first_layer: int = 0,
    ldraw_offset: list | None = None,
    ldraw_fit: bool = False,
    extra_parts: dict | None = None,
    allow_weak: bool = False,
    stability: str = "final",
    clutch_n: float = CLUTCH_N,
    mass_scale: float = 1.0,
) -> Output:
    t0 = time.perf_counter()
    nan = float("nan")
    out: Output = {
        "route": "invalid_input",
        "detail": "",
        "n_bricks": 0,
        "n_steps": 0,
        "out_of_plate": [],
        "collisions": [],
        "floating": [],
        "isolated": [],
        "step_unsupported": [],
        "step_blocked": [],
        "weak_joints": [],
        "weak_bricks": [],
        "weak_at_step": [],
        "stability_max": nan,
        "stability_per_brick": [],
        "worst_brick": -1,
        "unstable_step": -1,
        "bricks": [],
        "elapsed_s": 0.0,
    }

    def done(route: str, detail: str) -> Output:
        out["route"] = route
        out["detail"] = detail
        out["elapsed_s"] = time.perf_counter() - t0
        logger.info("check_design: %s - %s", route, detail)
        return out

    try:
        nx, ny = _plate(plate_size)
        if int(max_layers) < 1:
            raise ValueError("max_layers must be at least 1")
        if stability not in ("off", "final", "steps"):
            raise ValueError(f"stability {stability!r} is not off / final / steps")
        if clutch_n <= 0 or mass_scale <= 0:
            raise ValueError("clutch_n and mass_scale must be positive")
        labels: list[str]
        if ldraw:
            off = tuple(int(v) for v in (ldraw_offset or (0, 0, 0)))
            if len(off) != 3:
                raise ValueError("ldraw_offset is [dx, dy, dlayer] in studs / layers")
            parsed, steps, labels = _parse_ldraw(ldraw, off, extra_parts or {})  # type: ignore[arg-type]
            if ldraw_fit and parsed:
                mx = min(b.x for b in parsed)
                my = min(b.y for b in parsed)
                mz = min(b.z for b in parsed)
                parsed = [Brick(h=b.h, w=b.w, x=b.x - mx, y=b.y - my, z=b.z - mz) for b in parsed]
            first_layer = 0
        else:
            if isinstance(bricks, str):
                bricks = json.loads(bricks) if bricks.strip() else []
            if isinstance(bricks, dict):
                bricks = [v for k, v in bricks.items() if str(k).isdigit()]
            parsed, steps, labels = [], [], []
            for k, d in enumerate(bricks or []):
                b, label = _brick_from_dict(d, int(first_layer))
                parsed.append(b)
                labels.append(label)
                steps.append(int(d.get("step", d.get("brick_seq", k))))
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return done("invalid_input", str(exc))
    if not parsed:
        return done("invalid_input", "no bricks given (bricks or ldraw)")

    n = len(parsed)
    order = sorted(range(n), key=lambda i: (steps[i], i))
    out["n_bricks"] = n
    out["n_steps"] = len(set(steps))

    # BrickGPT: bounds, collisions, local floating, ground-connected component.
    out["out_of_plate"] = [
        i for i, b in enumerate(parsed) if not _brick_in_bounds(b, nx, ny, int(max_layers))
    ]
    occ, lo = _occupancy(parsed)
    if np.any(occ > 1):
        by_layer: dict[int, list[int]] = {}
        for i, b in enumerate(parsed):
            by_layer.setdefault(b.z, []).append(i)
        for ids in by_layer.values():
            for a_i, i in enumerate(ids):
                for j in ids[a_i + 1 :]:
                    if _overlap(parsed[i], parsed[j]) > 0:
                        out["collisions"].append([i, j])
    out["isolated"] = [i for i, b in enumerate(parsed) if _brick_floats(b, occ, lo)]
    adj = _graph(parsed)
    grounded = _ground_component(adj)
    out["floating"] = [i for i in range(n) if i not in grounded]

    # Per step: connected to the plate through what is already built, nothing built above it.
    placed: list[int] = []
    for i in order:
        b = parsed[i]
        for j in placed:
            if parsed[j].z > b.z and _overlap(parsed[j], b) > 0:
                out["step_blocked"].append([i, j])
        placed.append(i)
        sub = [parsed[k] for k in placed]
        if _ground_component(_graph(sub)).isdisjoint({len(placed) - 1}):
            out["step_unsupported"].append(i)
        studs = sum(_overlap(b, parsed[k]) for k in placed[:-1] if _connected(b, parsed[k]))
        if b.z == 0:
            studs += b.area
        if studs == 1 and b.area > 1:
            out["weak_at_step"].append(i)

    out["weak_joints"], out["weak_bricks"] = _weak_joints(parsed, adj, grounded)

    blocked_by_geometry = out["collisions"] or out["floating"] or out["out_of_plate"]
    if stability != "off" and not blocked_by_geometry:
        smax, per, err = _stability(parsed, clutch_n, mass_scale)
        if err:
            return done("invalid_input", err)
        out["stability_max"] = smax
        out["stability_per_brick"] = per
        out["worst_brick"] = int(np.argmax(per))
        if stability == "steps":
            unsupported = set(out["step_unsupported"])
            for k in range(1, n + 1):
                prefix = order[:k]
                if unsupported.intersection(prefix):
                    break
                s_k, _, err = _stability([parsed[i] for i in prefix], clutch_n, mass_scale)
                if s_k >= STABLE_BELOW:
                    out["unstable_step"] = order[k - 1]
                    break

    out["bricks"] = [
        {
            "index": i,
            "cell": [b.x, b.y, b.z + int(first_layer)],
            "size": labels[i],
            "footprint": [b.h, b.w],
            "step": steps[i],
            "connected": i in grounded,
            "utilisation": out["stability_per_brick"][i] if out["stability_per_brick"] else nan,
        }
        for i, b in enumerate(parsed)
    ]

    if out["out_of_plate"]:
        first = parsed[out["out_of_plate"][0]]
        return done(
            "out_of_plate",
            f"{len(out['out_of_plate'])} brick(s) leave the {nx}x{ny} plate or layers 0-{int(max_layers) - 1}"
            f" (first: brick {out['out_of_plate'][0]} at ({first.x}, {first.y}, {first.z}) size {first.h}x{first.w})",
        )
    if out["collisions"]:
        i, j = out["collisions"][0]
        return done(
            "collision", f"{len(out['collisions'])} overlapping pair(s), first bricks {i} and {j}"
        )
    if out["floating"]:
        return done(
            "floating",
            f"{len(out['floating'])} brick(s) not connected to the plate through studs: {out['floating'][:10]}"
            f" ({len(out['isolated'])} touch no other brick at all)",
        )
    if out["step_unsupported"]:
        i = out["step_unsupported"][0]
        return done(
            "step_unsupported",
            f"{len(out['step_unsupported'])} brick(s) not connected to the plate when placed; first brick {i} at step {steps[i]}",
        )
    if out["step_blocked"]:
        i, j = out["step_blocked"][0]
        return done(
            "step_blocked",
            f"{len(out['step_blocked'])} placement(s) under an already placed brick; first brick {i} (step {steps[i]}) under brick {j}",
        )
    if out["weak_joints"] and not allow_weak:
        lo_i, up_i = out["weak_joints"][0]
        below = "the plate" if lo_i < 0 else f"brick {lo_i}"
        return done(
            "weakly_connected",
            f"{len(out['weak_joints'])} single-stud joint(s) carry an overhang; first brick {up_i} on {below},"
            f" {len(out['weak_bricks'])} brick(s) hang on them",
        )
    if stability != "off":
        if out["stability_max"] >= STABLE_BELOW:
            return done(
                "unstable",
                f"needs {out['stability_max']:.2f} x the stud clutch ({clutch_n} N) at brick {out['worst_brick']}",
            )
        if out["unstable_step"] >= 0:
            return done(
                "unstable",
                f"the finished design holds (max {out['stability_max']:.2f} x clutch) but the build is unstable"
                f" once brick {out['unstable_step']} is placed",
            )
    tail = "" if stability == "off" else f"; max clutch utilisation {out['stability_max']:.2f}"
    weak = (
        f"; {len(out['weak_joints'])} single-stud overhang(s) allowed" if out["weak_joints"] else ""
    )
    return done("valid", f"{n} bricks in {out['n_steps']} steps{tail}{weak}")
