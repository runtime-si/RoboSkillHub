# Adapted from MOKA (https://github.com/moka-manipulation/moka, MIT,
# copyright 2024 Kuan Fang, Fangchen Liu), moka/planners/visual_prompt_utils.py:
# annotate_grid, sample_waypoint_in_tile, parse_json_string, and the tile
# convention of prompts/visual_prompt_planner/select_motion.txt; and the
# free-space question of RoboPoint (https://github.com/wentaoyuan/RoboPoint,
# Apache-2.0), robopoint/serve/gradio_web_server.py.
# Changes: the grid is drawn with OpenCV and its labels are outlined; the model
# names free tiles for one region instead of a motion; each chosen tile is
# sampled on a fixed 3 x 3 lattice instead of one uniform draw; RoboPoint's
# question goes to gemini-er.point instead of a RoboPoint checkpoint; every
# pixel is back-projected through calibrated depth and kept only when the
# footprint around it is level, flat and clear; spots come back world-frame,
# ranked by free radius.
"""Pick free spots on a named surface with a pointing model or a tile grid, then check them in depth."""

from __future__ import annotations

import json
import logging
import math
import re
from string import ascii_lowercase
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext
from gap_core.errors import ToolError

logger = logging.getLogger(__name__)

FOOTPRINT_RADIUS = 0.025
"""Radius of the held object's footprint when no ``object_obb`` is given [m]
(a 50 mm part)."""

CLEARANCE = 0.01
"""Free margin around the footprint [m]."""

MAX_SPOTS = 8
"""Points or tiles asked of the model."""

GRID_COLS = 8
GRID_ROWS = 6
"""Tile grid. MOKA used 5 x 5 on its cropped workspace views; on a full
640 x 480 station frame 8 x 6 gives 80 px tiles, about 8 cm on a table 1 m
from the camera, so a small board still spans more than one tile."""

TILE_LATTICE = (1 / 6, 1 / 2, 5 / 6)
"""Where in a chosen tile candidates are sampled, as fractions of its width and
height (3 x 3). MOKA drew one uniform sample per tile."""

MIN_POINTS = 40
"""Fewest depth points inside the footprint for a verdict; a 35 mm footprint
1 m away holds about 600 at 640 x 480."""

MAX_TILT_DEG = 4.0
"""Steepest support the object is set on. Fitted over twice the footprint on
the overhead camera's depth of the station: bare table median 1.3 deg, 90th
percentile 3.4; breadboard interior at most 2.8; footprints hanging 8 mm or
more off the breadboard's edge fit a compromise plane of 2.7-5.9 deg
(median 4.3), which is how most of them are caught."""

FLAT_RMS = 0.006
"""Largest RMS of the trimmed plane residual [m]. The overhead camera's depth
on the bare station table and breadboard: 2nd-98th percentile residuals
-8.5..+8.3 mm about a 30 mm-radius plane fit, RMS 3-4 mm."""

OBSTACLE_HEIGHT = 0.012
"""Height above the fitted support that counts as something standing on it [m].
Over the measured noise (98th percentile +8.3 mm); so anything lower, a cable
lying flat (4 mm), is invisible to this check and only the model guards it."""

DROP_DEPTH = 0.010
"""Depth below the fitted support that counts as off its edge [m]: the station
breadboard stands 16 mm over the table."""

CELL = 0.01
DROP_CELL = 0.02
CELL_MAJORITY = 0.5
MIN_CELL_POINTS = 4
"""The footprint is binned in x-y; a cell with at least MIN_CELL_POINTS points,
most of them above OBSTACLE_HEIGHT, is occupied (10 mm cells), and one most
of them below DROP_DEPTH is off the edge (20 mm cells: the breadboard's 6 mm
holes read deep and fill most of a 10 mm cell). One such cell rejects the
spot. A share of points over the whole footprint does not scale: a gripper
finger filled 3 % of a 110 mm footprint on the station table. Single noisy
points do not fill a cell."""

SECTORS = 8
MIN_SECTOR_SHARE = 0.3
"""Every eighth of the footprint's outer half (and of each ranking ring) must
hold at least this share of the median eighth's points. Past a table's edge
the depth lands on the floor far away, so the support ends as missing points,
not as points below it; occlusion behind a tall object reads the same way."""

SUPPORT_BAND = 0.008
"""Points within this height of the footprint core's median depth [m] are the
support the plane is fitted to. Half the breadboard's 16 mm step, so a
footprint whose core is on the board fits the board, not a compromise between
board and table."""

OUTER_FACTOR = 2.0
"""The free radius used for ranking is measured out to this multiple of the
footprint radius."""

RING_STEPS = 4

GRID_PROMPT = """\
The image is from a camera looking down at a robot's table-top workspace. It is \
annotated with a grid that divides it into tiles: columns are marked 'a', 'b', \
'c', ... from left to right and rows 1, 2, 3, ... from bottom to top, and every \
tile's label ('[column][row]', e.g. 'c2') is written at its centre.

Task: find tiles where an object about {size} cm across could be set down on \
{region}. Choose only tiles whose centre shows the bare surface of {region}, \
with nothing on it (no object, cable, tool, fixture or robot part) within about \
{size} cm of the centre, and not at the edge of {region}.

The response should be a dictionary in JSON form: {{"tiles": [...]}} with at \
most {n} tile labels, the tile with the most free room around it first. \
Return {{"tiles": []}} if no tile qualifies. Remember that the columns are \
marked from left to right and the rows from bottom to top."""

POINT_QUERY = (
    "several spots within the vacant area on {region} where an object about "
    "{size} cm across can be set down without touching anything"
)
"""RoboPoint's free-space question ("Find a few spots within the vacant area on
the rightmost white plate."), phrased as a gemini-er.point query."""


class Output(TypedDict):
    route: str
    place_position: dict[str, float]
    place_positions: list[dict[str, float]]
    spots: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    spot_count: int
    footprint_radius: float
    detail: str


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    if name:
        for camera in cameras:
            if camera.get("name") == name:
                return camera
        raise ValueError(f"observation has no camera named {name!r}")
    for camera in cameras:
        cam = str(camera.get("name") or "").lower()
        if not any(k in cam for k in ("wrist", "eye_in_hand", "hand")):
            return camera
    if cameras:
        return cameras[0]
    raise ValueError("observation has no cameras")


def _rotate(q: dict[str, float], v: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in "wxyz")
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return v @ r.T


def footprint_of(obb: dict[str, Any]) -> float:
    """Radius of the circle round the box's corners seen from above [m]."""
    e = np.array([float(obb["extent"][k]) for k in "xyz"])
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = _rotate(obb.get("orientation") or {"w": 1, "x": 0, "y": 0, "z": 0}, signs * e)
    return float(np.max(np.hypot(corners[:, 0], corners[:, 1])))


def annotate_grid(image: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """MOKA's tile grid: lines, and '[column][row]' at each tile's centre."""
    out = np.ascontiguousarray(image.copy())
    h, w = out.shape[:2]
    for i in range(1, rows):
        y = round(h * i / rows)
        cv2.line(out, (0, y), (w - 1, y), (0, 0, 0), 1, cv2.LINE_AA)
    for j in range(1, cols):
        x = round(w * j / cols)
        cv2.line(out, (x, 0), (x, h - 1), (0, 0, 0), 1, cv2.LINE_AA)
    scale = max(0.35, min(w / cols, h / rows) / 160.0)
    for j in range(cols):
        for i in range(rows):
            label = f"{ascii_lowercase[j]}{rows - i}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            org = (round(w * (j + 0.5) / cols - tw / 2), round(h * (i + 0.5) / rows + th / 2))
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(
                out, label, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA
            )
    return out


def parse_json_string(text: str) -> Any:
    """MOKA's reply parser: strip a code fence and // comments, then JSON. None on failure."""
    body = text
    fence = re.search(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        body = fence.group(1)
    body = "\n".join(line.split("//")[0].strip() for line in body.split("\n"))
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        start, end = body.find("{"), body.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(body[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                return None
    return None


def parse_tiles(text: str, cols: int, rows: int) -> list[tuple[int, int]]:
    """Tile labels from the reply as (column, row-from-top) indices, in the model's order."""
    data = parse_json_string(text)
    labels: list[str] = []
    if isinstance(data, dict):
        data = data.get("tiles", [])
    if isinstance(data, list):
        labels = [str(v) for v in data]
    else:
        labels = re.findall(r"\b[a-zA-Z]\d{1,2}\b", text)
    tiles: list[tuple[int, int]] = []
    for label in labels:
        m = re.fullmatch(r"\s*([a-zA-Z])(\d{1,2})\s*", label)
        if not m:
            continue
        col = ascii_lowercase.index(m.group(1).lower())
        row = int(m.group(2))
        if col < cols and 1 <= row <= rows and (col, rows - row) not in tiles:
            tiles.append((col, rows - row))
    return tiles


def tile_samples(width: int, height: int, cols: int, rows: int, tile: tuple[int, int]):
    """MOKA's sample_waypoint_in_tile on a fixed lattice: pixels inside the tile, centre first."""
    col, row = tile
    x0, x1 = width * col / cols, width * (col + 1) / cols
    y0, y1 = height * row / rows, height * (row + 1) / rows
    out = [((x0 + x1) / 2, (y0 + y1) / 2)]
    for fy in TILE_LATTICE:
        for fx in TILE_LATTICE:
            p = (x0 + fx * (x1 - x0), y0 + fy * (y1 - y0))
            if p not in out:
                out.append(p)
    return out


def _fit_plane(pts: np.ndarray) -> np.ndarray:
    """z = a x + b y + c by least squares, refit twice without points past 3 robust sigmas."""
    keep = np.ones(len(pts), dtype=bool)
    coef = np.zeros(3)
    for _ in range(3):
        a = np.c_[pts[keep, 0], pts[keep, 1], np.ones(int(keep.sum()))]
        coef, *_ = np.linalg.lstsq(a, pts[keep, 2], rcond=None)
        res = pts[:, 2] - (pts[:, 0] * coef[0] + pts[:, 1] * coef[1] + coef[2])
        mad = float(np.median(np.abs(res[keep] - np.median(res[keep]))))
        keep = np.abs(res) <= max(3.0 * 1.4826 * mad, 0.003)
        if keep.sum() < 10:
            break
    return coef


def _trimmed_rms(res: np.ndarray) -> float:
    """RMS of the residuals within 3 robust sigmas (at least 3 mm) of their median."""
    med = float(np.median(res))
    mad = float(np.median(np.abs(res - med)))
    keep = np.abs(res - med) <= max(3.0 * 1.4826 * mad, 0.003)
    return float(np.sqrt(np.mean(res[keep] ** 2)))


def _cells(dx: np.ndarray, dy: np.ndarray, flag: np.ndarray, size: float) -> int:
    """Cells of ``size`` holding MIN_CELL_POINTS or more points, most of them flagged."""
    if not len(flag):
        return 0
    ij = np.floor(np.c_[dx, dy] / size).astype(np.int64)
    _, cell, counts = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    hits = np.bincount(cell.reshape(-1), weights=flag, minlength=len(counts))
    return int(np.sum((counts >= MIN_CELL_POINTS) & (hits > CELL_MAJORITY * counts)))


def _bad_cells(
    dx: np.ndarray, dy: np.ndarray, res: np.ndarray, sel: np.ndarray, obstacle_height: float
) -> tuple[int, int]:
    """Occupied cells and off-the-edge cells among the points ``sel``."""
    x, y, r = dx[sel], dy[sel], res[sel]
    return _cells(x, y, r > obstacle_height, CELL), _cells(x, y, r < -DROP_DEPTH, DROP_CELL)


def _covered(dx: np.ndarray, dy: np.ndarray, ring: np.ndarray) -> bool:
    """True when every angular sector of ``ring`` holds a fair share of its points."""
    sector = ((np.arctan2(dy[ring], dx[ring]) + math.pi) / (2 * math.pi) * SECTORS).astype(int)
    counts = np.bincount(np.clip(sector, 0, SECTORS - 1), minlength=SECTORS)
    return bool(counts.min() >= max(3.0, MIN_SECTOR_SHARE * float(np.median(counts))))


def check_spot(
    ctx: NodeContext,
    camera: dict[str, Any],
    pixel_x: float,
    pixel_y: float,
    radius: float,
    max_tilt_deg: float,
    flat_rms: float,
    obstacle_height: float,
) -> dict[str, Any]:
    """Back-project a pixel and judge the footprint round it. ``reason`` is empty when it is free."""
    depth = np.asarray(camera["depth"], dtype=np.float32)
    k = np.asarray(camera["intrinsics"], dtype=np.float64)
    h, w = depth.shape[:2]
    px = min(max(float(pixel_x), 0.0), w - 1.0)
    py = min(max(float(pixel_y), 0.0), h - 1.0)
    verdict: dict[str, Any] = {"pixel_x": px, "pixel_y": py, "reason": ""}
    try:
        centre = ctx.tool(
            "geometry.pixel_to_world_point",
            pixel_x=px,
            pixel_y=py,
            depth=depth,
            intrinsics=k,
            camera_pose=camera["pose"],
        )["point"]
    except ToolError as exc:
        logger.debug("no depth at (%.0f, %.0f): %s", px, py, exc)
        return {**verdict, "reason": "no_depth"}
    u, v = round(px), round(py)
    d = float(depth[v, u])
    if not d > 0:
        return {**verdict, "reason": "no_depth"}
    outer = OUTER_FACTOR * radius
    half = int(math.ceil(1.25 * max(k[0, 0], k[1, 1]) * outer / d))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[max(0, v - half) : v + half + 1, max(0, u - half) : u + half + 1] = 255
    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=mask,
        depth=depth,
        intrinsics=k,
        camera_pose=camera["pose"],
    )["points"]
    pts = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
    cx, cy = float(centre["x"]), float(centre["y"])
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    dxy = np.hypot(dx, dy)
    inner = pts[dxy <= radius]
    if len(inner) < MIN_POINTS:
        return {**verdict, "reason": "no_depth", "points": len(inner)}
    core = pts[dxy <= radius / 2, 2]
    z_core = float(np.median(core)) if len(core) else float(np.median(inner[:, 2]))
    same = (np.abs(pts[:, 2] - z_core) <= SUPPORT_BAND) & (dxy <= outer)
    if same.sum() < MIN_POINTS:
        return {**verdict, "reason": "rough", "points": len(inner)}
    coef = _fit_plane(pts[same])
    res_all = pts[:, 2] - (pts[:, 0] * coef[0] + pts[:, 1] * coef[1] + coef[2])
    res = res_all[dxy <= radius]
    tilt = math.degrees(math.atan(math.hypot(coef[0], coef[1])))
    rms = _trimmed_rms(res)
    in_foot = dxy <= radius
    occupied, dropped = _bad_cells(dx, dy, res_all, in_foot, obstacle_height)
    z = float(coef[0] * cx + coef[1] * cy + coef[2])
    verdict.update(
        position={"x": cx, "y": cy, "z": z},
        tilt_deg=tilt,
        rms_m=rms,
        occupied_cells=occupied,
        edge_cells=dropped,
        points=len(inner),
    )
    if tilt > max_tilt_deg:
        verdict["reason"] = "tilted"
    elif occupied:
        verdict["reason"] = "occupied"
    elif dropped or not _covered(dx, dy, (dxy > radius / 2) & in_foot):
        verdict["reason"] = "edge"
    elif rms > flat_rms:
        verdict["reason"] = "rough"
    if verdict["reason"]:
        return verdict
    free = radius
    edges = np.linspace(radius, outer, RING_STEPS + 1)
    for r0, r1 in zip(edges[:-1], edges[1:], strict=True):
        ring = (dxy > r0) & (dxy <= r1)
        if ring.sum() < MIN_POINTS or not _covered(dx, dy, ring):
            break
        if any(_bad_cells(dx, dy, res_all, ring, obstacle_height)):
            break
        free = float(r1)
    verdict["free_radius"] = free
    return verdict


def _vec(p: dict[str, float]) -> dict[str, float]:
    return {"x": float(p["x"]), "y": float(p["y"]), "z": float(p["z"])}


def _candidates(
    ctx: NodeContext,
    rgb: np.ndarray,
    method: str,
    region: str,
    size_cm: int,
    max_spots: int,
    cols: int,
    rows: int,
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Groups of candidate pixels in the model's order; one group per point or per tile."""
    if method == "point":
        reply = ctx.tool(
            "gemini-er.point",
            image=rgb,
            query=POINT_QUERY.format(region=region, size=size_cm),
            max_points=max_spots,
        )
        points = reply.get("points") or []
        if not points and reply.get("found"):
            points = [reply]
        return [
            (f"point {i + 1}", [(float(p["pixel_x"]), float(p["pixel_y"]))])
            for i, p in enumerate(points)
        ]
    if method == "grid":
        h, w = rgb.shape[:2]
        reply = ctx.tool(
            "vlm.query",
            prompt=GRID_PROMPT.format(region=region, size=size_cm, n=max_spots),
            image=annotate_grid(rgb, cols, rows),
        )
        tiles = parse_tiles(str(reply.get("text", "")), cols, rows)[:max_spots]
        return [
            (f"tile {ascii_lowercase[c]}{rows - r}", tile_samples(w, h, cols, rows, (c, r)))
            for c, r in tiles
        ]
    raise ValueError(f"method must be 'point' or 'grid', not {method!r}")


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    region: str = "the table",
    object_obb: dict[str, Any] | None = None,
    footprint_radius: float = FOOTPRINT_RADIUS,
    clearance: float = CLEARANCE,
    method: str = "point",
    max_spots: int = MAX_SPOTS,
    grid_cols: int = GRID_COLS,
    grid_rows: int = GRID_ROWS,
    camera_name: str = "",
    max_tilt_deg: float = MAX_TILT_DEG,
    flat_rms: float = FLAT_RMS,
    obstacle_height: float = OBSTACLE_HEIGHT,
) -> Output:
    """Ask the model for free spots on ``region`` and keep those whose footprint is free in depth."""
    camera = _camera(observation, camera_name)
    rgb = np.asarray(camera["rgb"])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb[..., :3])
    base = footprint_of(object_obb) if object_obb else float(footprint_radius)
    radius = base + float(clearance)
    size_cm = max(1, round(200 * radius))

    groups = _candidates(
        ctx, rgb, method, region, size_cm, int(max_spots), int(grid_cols), int(grid_rows)
    )
    spots: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for order, (source, pixels) in enumerate(groups):
        best = None
        misses = []
        for px, py in pixels:
            v = check_spot(ctx, camera, px, py, radius, max_tilt_deg, flat_rms, obstacle_height)
            v.update(source=source, model_rank=order)
            if v["reason"]:
                misses.append(v)
            elif best is None or v["free_radius"] > best["free_radius"]:
                best = v
        if best is not None:
            spots.append(best)
        else:
            rejected.append(misses[0] if len(misses) == 1 else {**misses[0], "tried": len(misses)})

    spots.sort(key=lambda s: (-s["free_radius"], s["model_rank"]))
    kept: list[dict[str, Any]] = []
    for s in spots:
        p = s["position"]
        if all(
            math.hypot(p["x"] - k["position"]["x"], p["y"] - k["position"]["y"]) >= radius
            for k in kept
        ):
            kept.append(s)
        else:
            rejected.append({**s, "reason": "duplicate"})

    positions = [_vec(s["position"]) for s in kept]
    reasons = sorted({r["reason"] for r in rejected})
    if not groups:
        detail = f"the model named no free {'tile' if method == 'grid' else 'spot'} on {region}"
    elif not kept:
        detail = (
            f"all {len(groups)} candidates on {region} failed the depth check "
            f"({', '.join(reasons)}) for a {radius * 1000:.0f} mm footprint"
        )
    else:
        detail = (
            f"{len(kept)} free spot(s) on {region} for a {radius * 1000:.0f} mm footprint "
            f"out of {len(groups)} candidates"
        )
    return {
        "route": "found" if kept else "none_free",
        "place_position": positions[0] if positions else {"x": 0.0, "y": 0.0, "z": 0.0},
        "place_positions": positions,
        "spots": kept,
        "rejected": rejected,
        "spot_count": len(kept),
        "footprint_radius": radius,
        "detail": detail,
    }
