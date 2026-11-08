# The tile choice is DominoArtSolver's integer program (ArnoldF/DominoArtSolver, MIT; see
# mosaic_model.py's header for the files, the commit and what changed). Everything after the solve
# is new here: world poses of each tile on the table (grid origin, cell pitch, yaw), which end of
# the tile shows which value, the copy of the set each tile comes from, a placing order that never
# has an arm reach across a placed tile (farthest from the arm first, with an explicit check of the
# line from the arm's base to the tile), the free sides for the fingers at each step, a preview
# render and error measures.
"""Plan a flat domino mosaic of an image: which tile goes where, which way round, in what order."""

from __future__ import annotations

import heapq
import json
import logging
import math
import os
import sys
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mosaic_model as mm  # noqa: E402

logger = logging.getLogger(__name__)

TILE_SIZE = (0.048, 0.024, 0.0075)
"""A double-six domino lying flat, length x width x thickness [m]: the hub's domino skills' tile
(planning-a-domino-layout, reading-domino-pips-and-orientation use 48 x 24 x 7.5-8 mm)."""

GAP = 0.002
"""Default gap between neighbouring tiles [m] (a choice): the cell pitch defaults to the tile width
plus this. It is far under what fingers need between tiles (FINGER_* below), so a tight mosaic is
placed by dropping the tile from just above or sliding it in; `finger_room` says when."""

FINGER_THICKNESS = 0.010
FINGER_WIDTH = 0.020
FINGER_MARGIN = 0.003
"""A linear_4310 fingertip's depth along the closing axis, its width across it, and the free space
it needs around it [m]: singulating-by-pushing's estimates, as planning-brick-assembly-order uses."""

CORRIDOR_SHRINK = 0.001
"""A placed tile blocks the line from an arm's base to a later tile only if the line crosses the
tile shrunk by this much [m], so a line along a gap between two tiles does not count."""

DEFAULT_TIME_LIMIT = 120.0
"""Solver time limit [s]: twice the slowest solve measured up to 24 double-six sets (HiGHS, 53 s
at 16 sets; CBC at most 17 s; SKILL.md, Measured)."""

PIP_LAYOUTS: dict[int, list[tuple[float, float]]] = {
    0: [],
    1: [(0.5, 0.5)],
    2: [(0.25, 0.25), (0.75, 0.75)],
    3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
    4: [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)],
    5: [(0.25, 0.25), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (0.75, 0.75)],
    6: [(0.25, 0.25), (0.25, 0.5), (0.25, 0.75), (0.75, 0.25), (0.75, 0.5), (0.75, 0.75)],
    7: [
        (0.25, 0.25),
        (0.25, 0.5),
        (0.25, 0.75),
        (0.5, 0.5),
        (0.75, 0.25),
        (0.75, 0.5),
        (0.75, 0.75),
    ],
    8: [(0.25, y) for y in (0.25, 0.5, 0.75)]
    + [(0.5, 0.25), (0.5, 0.75)]
    + [(0.75, y) for y in (0.25, 0.5, 0.75)],
    9: [(x, y) for x in (0.25, 0.5, 0.75) for y in (0.25, 0.5, 0.75)],
}
"""Pip positions in a tile half (unit square) for the preview: the usual layouts (upstream's
plotting used 0.2 / 0.8 for the same patterns)."""

KNOWN_INPUTS = {
    "sets",
    "image",
    "image_path",
    "max_pips",
    "rows",
    "cols",
    "grid_origin",
    "anchor",
    "grid_yaw_deg",
    "pitch_m",
    "tile_size",
    "pip_contrast",
    "arms",
    "solver",
    "time_limit_s",
    "accept_suboptimal",
    "cost",
    "preview_path",
    "preview_cell_px",
    "finger_thickness",
    "finger_width",
    "finger_margin",
}


class Output(TypedDict):
    route: str
    detail: str
    placements: list[dict[str, Any]]
    placements_json: str
    pose_sequence: dict[str, Any]
    first_pose: dict[str, Any] | None
    by_arm: dict[str, list[int]]
    rows: int
    cols: int
    n_tiles: int
    sets: int
    max_pips: int
    per_set: list[dict[str, Any]]
    supply: dict[str, int]
    grid: dict[str, Any]
    target: list[list[float]]
    values: list[list[int]]
    preview: np.ndarray | None
    target_preview: np.ndarray | None
    preview_path: str
    objective: float
    cell_rmse: float
    floor_rmse: float
    image_rmse: float
    correlation: float
    solver: str
    solver_status: str
    solve_time_s: float
    reach_over: int
    steps_without_finger_room: int
    out_of_reach: list[list[int]]
    problems: list[str]


class _Route(Exception):
    def __init__(self, route: str, detail: str):
        super().__init__(detail)
        self.route, self.detail = route, detail


def _empty(route: str, detail: str, **kw: Any) -> Output:
    out: dict[str, Any] = {
        "route": route,
        "detail": detail,
        "placements": [],
        "placements_json": "[]",
        "pose_sequence": {"waypoints": []},
        "first_pose": None,
        "by_arm": {},
        "rows": 0,
        "cols": 0,
        "n_tiles": 0,
        "sets": 0,
        "max_pips": 0,
        "per_set": [],
        "supply": {},
        "grid": {},
        "target": [],
        "values": [],
        "preview": None,
        "target_preview": None,
        "preview_path": "",
        "objective": float("nan"),
        "cell_rmse": float("nan"),
        "floor_rmse": float("nan"),
        "image_rmse": float("nan"),
        "correlation": float("nan"),
        "solver": "",
        "solver_status": "",
        "solve_time_s": 0.0,
        "reach_over": 0,
        "steps_without_finger_room": 0,
        "out_of_reach": [],
        "problems": [],
    }
    out.update(kw)
    return Output(**out)  # type: ignore[typeddict-item]


# -- inputs ---------------------------------------------------------------------------------


def _grey(image: Any, image_path: str | None) -> np.ndarray:
    if image is None and not image_path:
        raise _Route("invalid_image", "no `image` and no `image_path`")
    if image is None:
        grey = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if grey is None:
            raise _Route("invalid_image", f"cannot read an image from {image_path!r}")
        return grey.astype(np.float64)
    if isinstance(image, dict):
        image = image.get("rgb")
    arr = np.asarray(image) if image is not None else None
    if arr is None or arr.size == 0 or arr.ndim not in (2, 3):
        raise _Route("invalid_image", "`image` is not an H x W or H x W x 3 array")
    if arr.ndim == 3:
        if arr.shape[2] not in (3, 4):
            raise _Route("invalid_image", f"`image` has {arr.shape[2]} channels")
        a = arr[..., :3].astype(np.float64)
        arr = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]  # RGB, as gap's rgb
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise _Route("invalid_image", "`image` has non-finite values")
    return arr


def _xyz(value: Any, name: str) -> tuple[np.ndarray, float | None]:
    """A point and, for an Se3Pose, its yaw."""
    yaw = None
    if isinstance(value, dict):
        if "position" in value:
            rot = value.get("rotation")
            if isinstance(rot, dict):
                w, x, y, z = (float(rot.get(k, 0.0)) for k in ("w", "x", "y", "z"))
                yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            value = value["position"]
        if isinstance(value, dict):
            value = [value.get("x", 0.0), value.get("y", 0.0), value.get("z", 0.0)]
    try:
        v = [float(c) for c in value]
    except (TypeError, ValueError) as exc:
        raise _Route("invalid_input", f"`{name}` is not a point or pose") from exc
    if len(v) == 2:
        v.append(0.0)
    if len(v) != 3 or not all(math.isfinite(c) for c in v):
        raise _Route("invalid_input", f"`{name}` is not a point or pose")
    return np.array(v), yaw


def _arms(arms: Any, origin_given: bool, near_default: np.ndarray) -> list[dict]:
    if arms is None:
        if origin_given:
            return [{"arm_id": 0, "base": np.zeros(3), "reach": None, "min_reach": 0.0}]
        return [{"arm_id": 0, "base": near_default, "reach": None, "min_reach": 0.0}]
    if isinstance(arms, dict):
        arms = [arms]
    out = []
    for i, arm in enumerate(arms):
        if not isinstance(arm, dict) or "base" not in arm:
            raise _Route("invalid_input", f"arms[{i}] needs a `base` point")
        base, _ = _xyz(arm["base"], f"arms[{i}].base")
        reach = arm.get("reach_m")
        out.append(
            {
                "arm_id": arm.get("arm_id", i),
                "base": base,
                "reach": float(reach) if reach is not None else None,
                "min_reach": float(arm.get("min_reach_m", 0.0)),
            }
        )
    if not out:
        raise _Route("invalid_input", "`arms` is empty")
    return out


def _nearest_arm(arm_list: list[dict], bases_local: np.ndarray, p: np.ndarray) -> int:
    """The nearest arm that reaches `p` (every cell has one: checked before the solve)."""
    d = np.linalg.norm(bases_local - p, axis=1)
    ok = [
        k
        for k, a in enumerate(arm_list)
        if (a["reach"] is None or d[k] <= a["reach"]) and d[k] >= a["min_reach"]
    ]
    return int(min(ok or range(len(arm_list)), key=lambda k: d[k]))


# -- geometry -------------------------------------------------------------------------------


def _segment_hits_box(p0: np.ndarray, p1: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Liang-Barsky: does the segment p0 -> p1 cross each axis-aligned box [lo, hi] (N x 2)?"""
    d = p1 - p0
    t0 = np.zeros(len(lo))
    t1 = np.ones(len(lo))
    ok = np.ones(len(lo), dtype=bool)
    for k in range(2):
        if abs(d[k]) < 1e-12:
            ok &= (p0[k] >= lo[:, k]) & (p0[k] <= hi[:, k])
            continue
        ta = (lo[:, k] - p0[k]) / d[k]
        tb = (hi[:, k] - p0[k]) / d[k]
        t0 = np.maximum(t0, np.minimum(ta, tb))
        t1 = np.minimum(t1, np.maximum(ta, tb))
    return ok & (t0 <= t1)


def _boxes_overlap(lo: np.ndarray, hi: np.ndarray, blo: np.ndarray, bhi: np.ndarray) -> bool:
    """Open boxes: touching (to 1 um) is not overlapping."""
    if not len(blo):
        return False
    return bool(((blo < hi - 1e-6) & (bhi > lo + 1e-6)).all(axis=1).any())


def _order(
    centres: np.ndarray,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    arm_of: np.ndarray,
    bases: np.ndarray,
    keys: list[tuple[int, int]],
) -> tuple[list[int], list[list[int]]]:
    """Farthest from its arm first, never after a tile on the line from its arm's base to it.

    Tile i must go before tile j when j lies on i's line (placing i later would reach across j).
    A topological order of those constraints, farthest first among the ready ones; a cycle (not
    seen on a grid) is broken farthest first and reported per step as `reach_over`.
    """
    n = len(centres)
    dist = np.linalg.norm(centres - bases[arm_of], axis=1)
    lo_s, hi_s = box_lo + CORRIDOR_SHRINK, box_hi - CORRIDOR_SHRINK
    before: list[set[int]] = [set() for _ in range(n)]  # i must precede every j in before[i]
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        hits = _segment_hits_box(bases[arm_of[i]], centres[i], lo_s, hi_s)
        hits[i] = False
        for j in np.flatnonzero(hits):
            before[i].add(int(j))
            indeg[j] += 1
    heap = [(-dist[i], keys[i], i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    done = np.zeros(n, dtype=bool)
    order: list[int] = []
    while len(order) < n:
        if not heap:
            i = max((k for k in range(n) if not done[k]), key=lambda k: (dist[k], keys[k]))
            heap = [(-dist[i], keys[i], i)]
            indeg[i] = 0
        _, _, i = heapq.heappop(heap)
        if done[i]:
            continue
        done[i] = True
        order.append(i)
        for j in before[i]:
            indeg[j] -= 1
            if indeg[j] == 0 and not done[j]:
                heapq.heappush(heap, (-dist[j], keys[j], j))
    blocked: list[list[int]] = []
    for step, i in enumerate(order):
        placed = np.array(order[:step], dtype=int)
        if len(placed):
            hits = _segment_hits_box(bases[arm_of[i]], centres[i], lo_s[placed], hi_s[placed])
            blocked.append([int(k) for k in placed[hits]])
        else:
            blocked.append([])
    return order, blocked


def _finger_room(
    i: int, placed: list[int], box_lo, box_hi, axis_u: bool, size: tuple[float, float], fingers
) -> list[str]:
    """Which grasp axes have both finger spots clear of the tiles already placed."""
    thick, width, margin = fingers
    length, w = size
    centre = (box_lo[i] + box_hi[i]) / 2
    blo, bhi = box_lo[placed], box_hi[placed]
    free = []
    for name, along_long in (("across", False), ("along", True)):
        ok = True
        for sign in (-1.0, 1.0):
            if along_long:  # fingers on the tile's ends
                off, half_off, half_side = length / 2, thick + margin, width / 2 + margin
            else:  # fingers on the long sides
                off, half_off, half_side = w / 2, thick + margin, width / 2 + margin
            # the finger spot in the tile's own frame: x along the long axis
            if along_long:
                x0, x1 = sign * off, sign * (off + half_off)
                y0, y1 = -half_side, half_side
            else:
                x0, x1 = -half_side, half_side
                y0, y1 = sign * off, sign * (off + half_off)
            lx, hx = min(x0, x1), max(x0, x1)
            ly, hy = min(y0, y1), max(y0, y1)
            if axis_u:
                lo = centre + np.array([lx, ly])
                hi = centre + np.array([hx, hy])
            else:
                lo = centre + np.array([ly, lx])
                hi = centre + np.array([hy, hx])
            if _boxes_overlap(lo, hi, blo, bhi):
                ok = False
                break
        if ok:
            free.append(name)
    return free


# -- rendering ------------------------------------------------------------------------------


def _render(
    rows: int,
    cols: int,
    tiles: list[dict[str, Any]],
    cell_px: int,
    pitch: float,
    size: tuple[float, float],
    dark_pips: bool,
    max_pips: int,
) -> np.ndarray:
    img = np.full((rows * cell_px, cols * cell_px, 3), 150, dtype=np.uint8)
    body = (240, 238, 228) if dark_pips else (25, 25, 25)
    pip = (15, 15, 15) if dark_pips else (245, 245, 245)
    scale = cell_px / pitch
    gap_w = max(0, round((pitch - size[1]) * scale / 2))
    gap_l = max(0, round((2 * pitch - size[0]) * scale / 2))
    r_pip = max(1, round(0.09 * cell_px))
    for t in tiles:
        (ra, ca), (rb, cb) = t["cells"]
        r0, r1 = min(ra, rb), max(ra, rb)
        c0, c1 = min(ca, cb), max(ca, cb)
        y0, y1 = r0 * cell_px, (r1 + 1) * cell_px - 1
        x0, x1 = c0 * cell_px, (c1 + 1) * cell_px - 1
        horizontal = ra == rb
        dy, dx = (gap_w, gap_l) if horizontal else (gap_l, gap_w)
        cv2.rectangle(img, (x0 + dx, y0 + dy), (x1 - dx, y1 - dy), body, -1)
        if horizontal:
            xm = (c0 + 1) * cell_px
            cv2.line(img, (xm, y0 + dy + 2), (xm, y1 - dy - 2), pip, 1)
        else:
            ym = (r0 + 1) * cell_px
            cv2.line(img, (x0 + dx + 2, ym), (x1 - dx - 2, ym), pip, 1)
        for (r, c), v in zip(t["cells"], t["value_ab"], strict=True):
            for px, py in PIP_LAYOUTS[min(int(v), 9)]:
                cx = round(c * cell_px + px * (cell_px - 1))
                cy = round(r * cell_px + py * (cell_px - 1))
                cv2.circle(img, (cx, cy), r_pip, pip, -1, lineType=cv2.LINE_AA)
    return img


# -- the skill ------------------------------------------------------------------------------


def run(
    ctx: NodeContext,
    sets: int = 1,
    image: Any = None,
    image_path: str | None = None,
    max_pips: int = 6,
    rows: int | None = None,
    cols: int | None = None,
    grid_origin: Any = None,
    anchor: str = "top_left",
    grid_yaw_deg: float | None = None,
    pitch_m: float | None = None,
    tile_size: list[float] | None = None,
    pip_contrast: str = "dark_on_light",
    arms: Any = None,
    solver: str = "auto",
    time_limit_s: float = DEFAULT_TIME_LIMIT,
    accept_suboptimal: bool = True,
    cost: str = "upstream",
    preview_path: str | None = None,
    preview_cell_px: int = 24,
    finger_thickness: float = FINGER_THICKNESS,
    finger_width: float = FINGER_WIDTH,
    finger_margin: float = FINGER_MARGIN,
    **unknown: Any,
) -> Output:
    """Choose a tile, a cell pair and an end order for every domino of `sets` complete sets so the
    pips reproduce the image, then turn the plan into ordered world poses on the table.

    Args:
        sets: complete double-`max_pips` sets available (each tile type used `sets` times).
        image / image_path: the picture (H x W grey or RGB array, a CameraFrame, or a file).
        max_pips: 6 for double-six (28 tiles a set), 9 for upstream's double-nine.
        rows, cols: the grid in cells (two per tile); default: upstream's choice, rows x cols =
            2 x tiles x sets closest to the image's aspect.
        grid_origin: world point or Se3Pose of the anchor cell's centre on the table top
            (`anchor` top_left: cell (0, 0), the image's top-left; center: the grid's centre).
            Columns run along the grid's +x, rows along its -y (the image upright seen from above
            with +y up). Yaw from the pose's rotation or `grid_yaw_deg`.
        pitch_m: cell pitch, default tile width + 2 mm; a tile spans 2 x pitch.
        tile_size: [length, width(, thickness)] in m, default 48 x 24 x 7.5 mm.
        pip_contrast: dark_on_light (white tiles, black pips: more pips = darker) or
            light_on_dark (upstream's black tiles with white pips).
        arms: [{arm_id, base, reach_m?, min_reach_m?}] in world; default one arm at the world
            origin (or 0.3 m beyond the image's bottom edge without `grid_origin`).
        solver: auto (CBC through PuLP when installed, else HiGHS through SciPy) | cbc | highs.
        time_limit_s: solver time limit. accept_suboptimal: a feasible plan found by the time
            limit is `planned` (else `solver_timeout`).
        cost: upstream (squared error truncated to an int, as DominoArtSolver) | squared.
        preview_path: also write [target | mosaic] side by side as an image file.
    """
    del ctx
    try:
        if unknown:
            raise _Route("invalid_input", f"unknown input(s): {sorted(unknown)}")
        return _plan(**{k: v for k, v in locals().items() if k in KNOWN_INPUTS})
    except _Route as r:
        return _empty(r.route, r.detail)


def _plan(
    sets: int,
    image: Any,
    image_path: str | None,
    max_pips: int,
    rows: int | None,
    cols: int | None,
    grid_origin: Any,
    anchor: str,
    grid_yaw_deg: float | None,
    pitch_m: float | None,
    tile_size: list[float] | None,
    pip_contrast: str,
    arms: Any,
    solver: str,
    time_limit_s: float,
    accept_suboptimal: bool,
    cost: str,
    preview_path: str | None,
    preview_cell_px: int,
    finger_thickness: float,
    finger_width: float,
    finger_margin: float,
) -> Output:
    problems: list[str] = []
    # -- options
    if pip_contrast not in ("dark_on_light", "light_on_dark"):
        raise _Route("invalid_input", f"pip_contrast {pip_contrast!r}")
    if solver not in ("auto", "cbc", "highs"):
        raise _Route("invalid_input", f"solver {solver!r}")
    if cost not in ("upstream", "squared"):
        raise _Route("invalid_input", f"cost {cost!r}")
    if anchor not in ("top_left", "center"):
        raise _Route("invalid_input", f"anchor {anchor!r}")
    try:
        max_pips = int(max_pips)
        sets = int(sets)
        time_limit_s = float(time_limit_s)
        preview_cell_px = int(preview_cell_px)
    except (TypeError, ValueError) as exc:
        raise _Route("invalid_input", f"a numeric input is not a number ({exc})") from exc
    if not 1 <= max_pips <= 9:
        raise _Route("invalid_input", f"max_pips {max_pips}: 1 to 9")
    if time_limit_s <= 0 or preview_cell_px < 8:
        raise _Route("invalid_input", "time_limit_s must be > 0 and preview_cell_px >= 8")
    size = tuple(float(v) for v in (tile_size or TILE_SIZE))
    if len(size) == 2:
        size = (*size, TILE_SIZE[2])
    if len(size) != 3 or min(size) <= 0:
        raise _Route("invalid_input", "tile_size is [length, width(, thickness)] > 0")
    length, width, thickness = size
    pitch = float(pitch_m) if pitch_m is not None else width + GAP
    if pitch < width - 1e-9 or 2 * pitch < length - 1e-9:
        raise _Route(
            "invalid_input",
            f"pitch {pitch * 1000:.1f} mm: tiles of {length * 1000:.0f} x {width * 1000:.0f} mm "
            f"need >= {max(width, length / 2) * 1000:.1f} mm",
        )
    dark_pips = pip_contrast == "dark_on_light"
    tiles = mm.domino_tiles(max_pips)
    per_set = len(tiles)

    # -- image (upstream checks it first)
    grey = _grey(image, image_path)
    h, w = grey.shape
    if sets < 1:
        raise _Route("not_enough_sets", f"sets = {sets}: at least one complete set is needed")
    # -- grid
    if (rows is None) != (cols is None):
        raise _Route("invalid_input", "give both `rows` and `cols`, or neither")
    if rows is None:
        n_fields = 2 * per_set * sets
        rows, cols = mm.grid_for_sets(n_fields, w / h)
        exact = True
    else:
        rows, cols = int(rows), int(cols)
        if rows < 1 or cols < 1 or (rows * cols) % 2:
            raise _Route("invalid_input", f"grid {rows} x {cols}: needs an even number of cells")
        if rows * cols > 2 * per_set * sets:
            raise _Route(
                "not_enough_sets",
                f"a {rows} x {cols} grid takes {rows * cols // 2} tiles; {sets} set(s) of "
                f"{per_set} hold {per_set * sets} (needs {math.ceil(rows * cols / 2 / per_set)})",
            )
        exact = rows * cols == 2 * per_set * sets
    if rows > h or cols > w:
        raise _Route(
            "invalid_image", f"a {w} x {h} px image is smaller than the {cols} x {rows} grid"
        )
    means, fh, fw = mm.field_brightness(grey, rows, cols)
    if float(np.ptp(means)) < 1e-6:
        raise _Route("invalid_image", "the image has no contrast at this grid (all fields equal)")
    target = mm.target_values(means, max_pips, dark_pips)

    # -- world frame of the grid
    origin_given = grid_origin is not None
    origin, pose_yaw = _xyz(grid_origin, "grid_origin") if origin_given else (np.zeros(3), None)
    if not origin_given:
        problems.append("no grid_origin: poses are in the grid's own frame (cell (0, 0) at 0, 0)")
    yaw = math.radians(float(grid_yaw_deg)) if grid_yaw_deg is not None else (pose_yaw or 0.0)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot = np.array([[cy, -sy], [sy, cy]])
    shift = np.zeros(2)
    if anchor == "center":
        shift = np.array([-(cols - 1) / 2 * pitch, (rows - 1) / 2 * pitch])

    def cell_local(r: float, c: float) -> np.ndarray:
        return np.array([c * pitch, -r * pitch]) + shift

    def to_world(p: np.ndarray) -> np.ndarray:
        return origin[:2] + rot @ p

    def to_local(p: np.ndarray) -> np.ndarray:
        return rot.T @ (p[:2] - origin[:2])

    near = to_world(cell_local(rows - 1 + 0.3 / pitch, (cols - 1) / 2))
    arm_list = _arms(arms, origin_given, np.array([near[0], near[1], origin[2]]))
    bases_local = np.array([to_local(a["base"]) for a in arm_list])

    # -- reach, before the solve: every cell must be reachable by some arm
    far_cells = []
    for r in range(rows):
        for c in range(cols):
            p = cell_local(r, c)
            d = np.linalg.norm(bases_local - p, axis=1)
            if not any(
                (a["reach"] is None or d[k] <= a["reach"]) and d[k] >= a["min_reach"]
                for k, a in enumerate(arm_list)
            ):
                far_cells.append([r, c])
    grid_info = {
        "rows": rows,
        "cols": cols,
        "pitch_m": pitch,
        "tile_size_m": list(size),
        "origin": origin.tolist(),
        "yaw_rad": yaw,
        "anchor": anchor,
        "frame": "world" if origin_given else "grid",
        "extent_m": [cols * pitch, rows * pitch],
        "field_px": [fh, fw],
        "exact_usage": exact,
        "image_aspect": w / h,
        "grid_aspect": cols / rows,
    }
    stretch = (cols / rows) / (w / h)
    if abs(math.log(stretch)) > math.log(1.1):
        problems.append(
            f"the {rows} x {cols} grid stretches the image {stretch:.2f}x in width (no divisor of "
            f"the cell count fits its aspect better): give rows/cols for fewer cells to keep it"
        )
    if far_cells:
        return _empty(
            "out_of_reach",
            f"{len(far_cells)} of {rows * cols} cells are outside every arm's reach",
            out_of_reach=far_cells,
            rows=rows,
            cols=cols,
            sets=sets,
            max_pips=max_pips,
            grid=grid_info,
        )

    # -- the integer program
    places = mm.placements(rows, cols)
    costs = mm.cost_matrix(target, tiles, places, truncate=cost == "upstream")
    model = mm.DominoModel(costs, places, rows, cols, sets, exact)
    try:
        sol = model.solve(solver, time_limit_s)
    except mm.SolverUnavailable as exc:
        raise _Route("solver_unavailable", str(exc)) from exc
    base_kw = {
        "rows": rows,
        "cols": cols,
        "sets": sets,
        "max_pips": max_pips,
        "grid": grid_info,
        "solver": sol.solver,
        "solver_status": sol.status,
        "solve_time_s": sol.seconds,
    }
    if sol.status in ("no_solution", "infeasible"):
        route = "solver_timeout" if sol.status == "no_solution" else "solver_failed"
        return _empty(
            route,
            f"{sol.detail}: no plan in {time_limit_s:.0f} s ({rows} x {cols} cells, "
            f"{costs.size} binaries)",
            **base_kw,
        )
    bad = model.check(sol.chosen)
    if bad:
        return _empty(
            "solver_failed", f"{sol.detail}: the solution is not a tiling ({bad})", **base_kw
        )
    if sol.status == "feasible":
        msg = f"time limit {time_limit_s:.0f} s hit: a feasible plan, not proven optimal"
        if not accept_suboptimal:
            return _empty("solver_timeout", f"{sol.detail}: {msg}", **base_kw)
        problems.append(msg)

    # -- tiles in the world
    n = len(sol.chosen)
    recs = []
    values = np.full((rows, cols), -1, dtype=int)
    for t, p in sorted(sol.chosen, key=lambda tp: places[tp[1]]):
        lo, hi = tiles[t]
        a_cell, b_cell = places[p]
        if not mm.lower_value_first(target, tiles[t], places[p]):
            a_cell, b_cell = b_cell, a_cell
        values[a_cell], values[b_cell] = lo, hi
        recs.append({"tile_id": t, "value_ab": [lo, hi], "cells": [list(a_cell), list(b_cell)]})
    centres_l = np.zeros((n, 2))
    box_lo = np.zeros((n, 2))
    box_hi = np.zeros((n, 2))
    axis_u = np.zeros(n, dtype=bool)
    for i, rec in enumerate(recs):
        pa = cell_local(*rec["cells"][0])
        pb = cell_local(*rec["cells"][1])
        centre = (pa + pb) / 2
        horizontal = rec["cells"][0][0] == rec["cells"][1][0]
        half = (
            np.array([length / 2, width / 2]) if horizontal else np.array([width / 2, length / 2])
        )
        centres_l[i], box_lo[i], box_hi[i], axis_u[i] = (
            centre,
            centre - half,
            centre + half,
            horizontal,
        )
    arm_of = np.array([_nearest_arm(arm_list, bases_local, centres_l[i]) for i in range(n)])
    keys = [
        (min(r["cells"][0][0], r["cells"][1][0]), min(r["cells"][0][1], r["cells"][1][1]))
        for r in recs
    ]
    order, blocked = _order(centres_l, box_lo, box_hi, arm_of, bases_local, keys)

    fingers = (float(finger_thickness), float(finger_width), float(finger_margin))
    copies: dict[int, int] = {}
    placements: list[dict[str, Any]] = []
    by_arm: dict[str, list[int]] = {}
    no_room = 0
    step_of = {i: step for step, i in enumerate(order)}
    for step, i in enumerate(order):
        rec = recs[i]
        a_cell, b_cell = rec["cells"]
        pa_w = to_world(cell_local(*a_cell))
        pb_w = to_world(cell_local(*b_cell))
        centre_w = to_world(centres_l[i])
        axis = (pb_w - pa_w) / np.linalg.norm(pb_w - pa_w)
        tile_yaw = math.atan2(axis[1], axis[0])
        z = float(origin[2])
        set_index = copies.get(rec["tile_id"], 0)
        copies[rec["tile_id"]] = set_index + 1
        room = _finger_room(
            i, order[:step], box_lo, box_hi, bool(axis_u[i]), (length, width), fingers
        )
        no_room += not room
        arm = arm_list[arm_of[i]]
        lo, hi = rec["value_ab"]
        q = [math.cos(tile_yaw / 2), 0.0, 0.0, math.sin(tile_yaw / 2)]
        placements.append(
            {
                "step": step,
                "arm_id": arm["arm_id"],
                "value": [lo, hi],
                "value_ab": [lo, hi],
                "is_double": lo == hi,
                "set_index": set_index,
                "cells": [list(a_cell), list(b_cell)],
                "orientation": "row" if axis_u[i] else "column",
                "position": [float(centre_w[0]), float(centre_w[1]), z],
                "yaw_rad": tile_yaw,
                "yaw_deg": math.degrees(tile_yaw),
                "axis_world": [float(axis[0]), float(axis[1]), 0.0],
                "ends_world": [
                    [*(centre_w - axis * length / 2).tolist(), z],
                    [*(centre_w + axis * length / 2).tolist(), z],
                ],
                "half_centers_world": [
                    [*(centre_w - axis * length / 4).tolist(), z],
                    [*(centre_w + axis * length / 4).tolist(), z],
                ],
                "pose": {
                    "position": {"x": float(centre_w[0]), "y": float(centre_w[1]), "z": z},
                    "rotation": {"w": q[0], "x": q[1], "y": q[2], "z": q[3]},
                },
                "top_z": z + thickness,
                "distance_to_arm": float(np.linalg.norm(centres_l[i] - bases_local[arm_of[i]])),
                "finger_room": room,
                "reaches_over": sorted(step_of[k] for k in blocked[step]),
            }
        )
        by_arm.setdefault(str(arm["arm_id"]), []).append(step)
    reach_over = sum(1 for p in placements if p["reaches_over"])
    if reach_over:
        problems.append(f"{reach_over} steps reach across a placed tile (no order avoids it)")
    if no_room:
        need = finger_thickness + finger_margin
        problems.append(
            f"{no_room} of {n} steps have no grasp axis with {need * 1000:.0f} mm free on both "
            f"sides (gap {(pitch - width) * 1000:.1f} mm): release from just above or slide the "
            f"tile in, or use a pitch >= {(width + need) * 1000:.0f} mm"
        )

    # -- per set usage
    counts = np.zeros(per_set, dtype=int)
    for rec in recs:
        counts[rec["tile_id"]] += 1
    per_set_out = []
    for s in range(sets):
        used = [list(tiles[t]) for t in range(per_set) if counts[t] > s]
        unused = [list(tiles[t]) for t in range(per_set) if counts[t] <= s]
        per_set_out.append({"set_index": s, "used": len(used), "unused": unused})
    supply = {f"{lo}-{hi}": int(counts[t]) for t, (lo, hi) in enumerate(tiles) if counts[t]}

    # -- error measures
    cell_rmse = float(np.sqrt(np.mean((values - target) ** 2)))
    floor_rmse = float(np.sqrt(np.mean((np.clip(np.rint(target), 0, max_pips) - target) ** 2)))
    corr = float(np.corrcoef(values.ravel(), target.ravel())[0, 1]) if values.std() > 0 else 0.0
    crop = grey[: fh * rows, : fw * cols]
    g = np.clip((crop - means.min()) / np.ptp(means), 0, 1)
    shade = values / max_pips
    shade = 1 - shade if dark_pips else shade
    mosaic = np.repeat(np.repeat(shade, fh, axis=0), fw, axis=1)
    image_rmse = float(np.sqrt(np.mean((mosaic - g) ** 2)))
    objective = float(sum(costs[t, p] for t, p in sol.chosen))

    # -- previews
    ordered_recs = [recs[i] for i in order]
    preview = _render(
        rows, cols, ordered_recs, preview_cell_px, pitch, (length, width), dark_pips, max_pips
    )
    tgt = np.clip((target + 0.5) / (max_pips + 1), 0, 1)
    tgt = 1 - tgt if dark_pips else tgt
    target_preview = np.repeat(
        np.repeat((tgt * 255).astype(np.uint8), preview_cell_px, axis=0), preview_cell_px, axis=1
    )
    target_preview = np.stack([target_preview] * 3, axis=-1)
    written = ""
    if preview_path:
        pair = np.concatenate(
            [target_preview, np.full((rows * preview_cell_px, 8, 3), 255, np.uint8), preview],
            axis=1,
        )
        if cv2.imwrite(str(preview_path), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR)):
            written = str(preview_path)
        else:
            problems.append(f"could not write the preview to {preview_path}")

    waypoints = [{"pose": p["pose"]} for p in placements]
    detail = (
        f"{n} tiles ({sets} set(s) of double-{max_pips}) on a {rows} x {cols} grid, "
        f"{cols * pitch * 1000:.0f} x {rows * pitch * 1000:.0f} mm; {sol.solver} {sol.status} in "
        f"{sol.seconds:.2f} s, cell RMSE {cell_rmse:.2f} pips"
    )
    return Output(
        route="planned",
        detail=detail,
        placements=placements,
        placements_json=json.dumps(placements),
        pose_sequence={"waypoints": waypoints},
        first_pose=placements[0]["pose"] if placements else None,
        by_arm=by_arm,
        rows=rows,
        cols=cols,
        n_tiles=n,
        sets=sets,
        max_pips=max_pips,
        per_set=per_set_out,
        supply=supply,
        grid=grid_info,
        target=target.round(3).tolist(),
        values=values.tolist(),
        preview=preview,
        target_preview=target_preview,
        preview_path=written,
        objective=objective,
        cell_rmse=cell_rmse,
        floor_rmse=floor_rmse,
        image_rmse=image_rmse,
        correlation=corr,
        solver=sol.solver,
        solver_status=sol.status,
        solve_time_s=sol.seconds,
        reach_over=reach_over,
        steps_without_finger_room=no_room,
        out_of_reach=[],
        problems=problems,
    )
