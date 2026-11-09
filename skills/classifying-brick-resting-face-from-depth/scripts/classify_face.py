# Adapted from UR5-Pick-and-Place-Simulation (https://github.com/pietrolechthaler/UR5-Pick-and-Place-Simulation),
# catkin_ws/src/vision/scripts/lego-vision.py (getDepthAxis, process_item) at commit
# 596c496867c11ba63f47127833e0f49b7480ffd3, MIT License, Copyright (c) 2022 Pietro Lechthaler.
# Changes: the original crops a YOLO box from a fixed overhead Gazebo camera, takes the brick's
# height as table depth minus its nearest depth, matches it against the brick's dimensions to
# tell which axis is vertical (getDepthAxis), and leaves studs-up versus upside-down to a
# second YOLOv5 model run on the depth image; the silhouette's and the top surface's
# minAreaRect give the yaw, and their centres' offset the stud direction of a brick on its
# side. Here the depth is back-projected to world points (no perspective correction needed),
# the table level is measured around the brick, the brick is the connected cluster of points
# over it nearest the given position, every resting face of the given brick is scored on its
# height and its footprint, and studs-up versus upside-down is decided geometrically: an
# upside-down brick rests on its stud tops and shows its hollow underside (most of the inner
# footprint far below the rim), a studs-up one a flat face a stud height under its stud tops.
# The YOLO models (YOLOv5, AGPL) are not used. The view move follows the hub's
# pressing-a-brick-onto-studs.
"""Tell from depth whether a loose LEGO / DUPLO brick lies studs up, upside down or on its side."""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PRESETS: dict[str, dict[str, float]] = {
    "lego": {"stud_pitch": 0.008, "stud_height": 0.0017, "brick_height": 0.0096},
    "duplo": {"stud_pitch": 0.016, "stud_height": 0.0045, "brick_height": 0.0192},
}
"""Brick systems [m] (pressing-a-brick-onto-studs' presets: BrickSim's LEGO units, ICL's
1.7 mm knob height; published DUPLO dimensions). A LEGO plate: brick_height 0.0032."""

FOOTPRINT_PLAY = 0.0001
"""Each wall stands this far inside the stud lattice [m] (BrickSim's brick model)."""

HEIGHT_TOLERANCE_FRACTION = 0.3
"""A face is accepted when the measured height is within this fraction of the pitch of
its expected height (2.4 mm LEGO, 4.8 mm DUPLO). The source matched heights to whole
studs (round(height / unit)), i.e. half a unit."""

FOOTPRINT_TOLERANCE_FRACTION = 0.4
"""... and each side of the silhouette within this fraction of the pitch of its
expected length."""

MARGIN = 1.5
"""The best face's score must beat the next different face's by this factor; else
'unknown' (two faces explain the reading about equally)."""

DEEP_FRACTION = 0.45
"""Upside-down test: inner points more than this fraction of a brick height under the
rim (and at least 1.5 stud heights) are the hollow underside."""

TOP_THICKNESS_FRACTION = 0.1
"""Top wall about a tenth of the brick height (1 mm LEGO): the cavity is 0.9 brick
heights deep, so DEEP_FRACTION sits in its middle."""

UPSIDE_DOWN_SHARE = 0.25
STUDS_UP_SHARE = 0.10
"""Deep share at or over UPSIDE_DOWN_SHARE: upside down; at or under STUDS_UP_SHARE:
studs up (the source's YOLO 'sopra' / 'sotto' classes); between: unknown."""

INNER_FRACTION = 0.2
"""The inner footprint for the up/down test is the silhouette inset by this fraction of
the pitch (walls and edge pixels left out)."""

STUD_SIDE_FRACTION = 0.5
"""On a side: the stud side is where the silhouette reaches past the top wall's face;
the two sides must differ by this fraction of the stud height to name it."""

GRID = 0.001
"""Raster cell for the brick's connected cluster [m]."""

OBJECT_FRACTION = 0.35
"""Points this fraction of the lowest possible brick height over the table are brick."""

SEARCH_MARGIN = 0.02
"""Points within the brick's half-diagonal plus this of brick_position are searched [m]."""

TABLE_RING = 0.03
"""... and the table level is read within this much further out [m]."""

MIN_POINTS = 40
DEPTH_RANGE = (0.07, 0.6)
FRAMES = 3

VIEW_RANGES = (0.20, 0.25)
MIN_VIEW_CLEARANCE = 0.05
"""Camera ranges over the brick's expected top and the lowest tool centre over it [m]
(refining-target-from-wrist-closeup / pressing-a-brick-onto-studs)."""

REFUSALS = ("unreachable", "bad_speed_scale")

ROUTES = ("studs_up", "upside_down", "on_side", "unknown", "no_brick", "unreachable")


class Output(TypedDict):
    route: str
    detail: str
    vertical_axis: str
    up_vector: dict[str, float]
    stud_direction_known: bool
    yaw_deg: float
    centre: dict[str, float]
    height: float
    footprint: list[float]
    table_height: float
    deep_share: float
    scores: dict[str, float]
    samples: int


def _xyz(v: Any) -> np.ndarray:
    if isinstance(v, dict) and "position" in v:
        v = v["position"]
    return np.array([float(v["x"]), float(v["y"]), float(v["z"])])


def _vec(v: Any) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
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
    t = float(np.trace(r))
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    return dict(zip(("w", "x", "y", "z"), (float(v) for v in q), strict=True))


def _matrix(pose: dict[str, Any]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rot(pose["rotation"])
    m[:3, 3] = _xyz(pose)
    return m


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1.0e-9:
        if c > 0.0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1.0e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def _refusal(reply: Any) -> str:
    if not isinstance(reply, dict):
        return ""
    outcome = reply.get("outcome")
    if outcome in REFUSALS:
        return f"{outcome}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "failed")
    return ""


def _pick_camera(obs: dict[str, Any], tool: np.ndarray | None, name: str | None) -> dict | None:
    cams = [c for c in obs.get("cameras") or [] if c.get("pose") is not None]
    if name:
        return next((c for c in cams if c.get("name") == name), None)
    wrist = [c for c in cams if any(k in str(c.get("name", "")) for k in ("eye_in_hand", "wrist"))]
    if not wrist:
        return None
    if tool is None:
        return wrist[0]
    return min(wrist, key=lambda c: float(np.linalg.norm(_xyz(c["pose"]) - tool[:3, 3])))


def _tool_pose(obs: dict[str, Any], arm_id: int) -> np.ndarray | None:
    try:
        return _matrix(obs["arms"][arm_id]["ee_pose"])
    except (KeyError, IndexError, TypeError):
        return None


def world_points(cam: dict[str, Any], depth: np.ndarray) -> np.ndarray:
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    m = _matrix(cam["pose"])
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    ok = np.isfinite(depth) & (depth >= DEPTH_RANGE[0]) & (depth <= DEPTH_RANGE[1])
    z = depth[ok]
    pc = np.stack([(u[ok] - k[0, 2]) / k[0, 0] * z, (v[ok] - k[1, 2]) / k[1, 1] * z, z], axis=1)
    return pc @ m[:3, :3].T + m[:3, 3]


def _lowest_level(z: np.ndarray, cap: float) -> float:
    """The lowest well-supported level of heights ``z`` (residuals capped at ``cap``)."""
    lo, hi = np.percentile(z, [1, 50])
    grid = np.arange(lo, hi + 5.0e-5, 5.0e-5)
    cost = np.minimum((z[None, :] - grid[:, None]) ** 2, cap**2).sum(axis=1)
    return float(grid[int(np.argmin(cost))])


def faces(nx: int, ny: int, pitch: float, stud_h: float, body_h: float) -> dict[str, dict]:
    """Every resting face of an nx x ny brick: height over the table and silhouette
    sides (long, short)."""
    lx, ly = nx * pitch - 2 * FOOTPRINT_PLAY, ny * pitch - 2 * FOOTPRINT_PLAY
    deep = body_h + stud_h

    def sides(a: float, b: float) -> tuple[float, float]:
        return (max(a, b), min(a, b))

    out = {
        "z": {"height": deep, "sides": sides(lx, ly), "vertical": "z"},
        "x": {"height": lx, "sides": sides(ly, deep), "vertical": "x"},
        "y": {"height": ly, "sides": sides(lx, deep), "vertical": "y"},
    }
    if nx == ny:
        del out["y"]
    return out


def measure(
    pts: np.ndarray,
    near: np.ndarray,
    nx: int,
    ny: int,
    pitch: float,
    stud_h: float,
    body_h: float,
    table_z: float | None = None,
) -> dict[str, Any]:
    """Features of the brick nearest ``near`` (xy) among world points ``pts``."""
    hyp = faces(nx, ny, pitch, stud_h, body_h)
    half_diag = 0.5 * math.hypot(nx * pitch, ny * pitch) + body_h
    r = np.linalg.norm(pts[:, :2] - near[:2], axis=1)
    out: dict[str, Any] = {"error": ""}
    if table_z is None:
        ring = (r > half_diag + SEARCH_MARGIN) & (r < half_diag + SEARCH_MARGIN + TABLE_RING)
        if int(ring.sum()) < MIN_POINTS:
            ring = r < half_diag + SEARCH_MARGIN + TABLE_RING
        if int(ring.sum()) < MIN_POINTS:
            out["error"] = f"{int(ring.sum())} depth points around the brick"
            return out
        table_z = _lowest_level(pts[ring, 2], 0.002)
    out["table_height"] = float(table_z)
    h_min = min(f["height"] for f in hyp.values())
    sel = (r < half_diag + SEARCH_MARGIN) & (pts[:, 2] > table_z + OBJECT_FRACTION * h_min)
    cand = pts[sel]
    if len(cand) < MIN_POINTS:
        out["no_brick"] = True
        out["samples"] = len(cand)
        return out
    # The connected cluster nearest the given position.
    org = cand[:, :2].min(axis=0)
    ij = np.floor((cand[:, :2] - org) / GRID).astype(int)
    img = np.zeros((ij[:, 1].max() + 1, ij[:, 0].max() + 1), np.uint8)
    img[ij[:, 1], ij[:, 0]] = 1
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n_lab, lab = cv2.connectedComponents(img, connectivity=8)
    labels = lab[ij[:, 1], ij[:, 0]]
    best, best_d = 0, math.inf
    for k in range(1, n_lab):
        m = labels == k
        if m.sum() < MIN_POINTS:
            continue
        d = float(np.min(np.linalg.norm(cand[m, :2] - near[:2], axis=1)))
        if d < best_d:
            best, best_d = k, d
    if best == 0:
        out["no_brick"] = True
        out["samples"] = len(cand)
        return out
    # An upside-down brick's walls and its tubes are separate clusters over the hollow
    # between them: take the cluster whose hull encloses the nearest one, and every
    # cluster inside that hull.
    hulls = {}
    for k in range(1, n_lab):
        cells_k = np.argwhere(lab == k)[:, ::-1].astype(np.int32)
        if len(cells_k) >= 3:
            hulls[k] = (cv2.convexHull(cells_k), cells_k.mean(axis=0), len(cells_k))

    def inside(k: int, outer: int) -> bool:
        c = hulls[k][1]
        return cv2.pointPolygonTest(hulls[outer][0], (float(c[0]), float(c[1])), False) >= 0

    outer = best
    if best in hulls:
        enclosing = [k for k in hulls if k != best and inside(best, k)]
        if enclosing:
            outer = max(enclosing, key=lambda k: hulls[k][2])
    members = {outer} | ({k for k in hulls if inside(k, outer)} if outer in hulls else set())
    sel_b = np.isin(labels, list(members))
    brick = cand[sel_b]
    # Drop isolated cells (flying pixels between the brick and the table).
    cells, counts = np.unique(ij[sel_b], axis=0, return_counts=True)
    dense = {tuple(c) for c, n in zip(cells, counts, strict=True) if n >= 2}
    keep = np.array([tuple(c) in dense for c in ij[sel_b]])
    if keep.sum() >= MIN_POINTS:
        brick = brick[keep]
    out["samples"] = len(brick)
    heights = brick[:, 2] - table_z
    top = float(np.percentile(heights, 98))
    rect = cv2.minAreaRect(brick[:, :2].astype(np.float32))
    (cx, cy), (w, h), ang = rect
    long_side, short_side = max(w, h), min(w, h)
    # The long side's direction in the plane.
    a = math.radians(ang) if w >= h else math.radians(ang + 90.0)
    long_dir = np.array([math.cos(a), math.sin(a)])
    out.update(
        height=top,
        sides=(float(long_side), float(short_side)),
        centre_xy=np.array([cx, cy]),
        long_dir=long_dir,
    )
    h_tol = HEIGHT_TOLERANCE_FRACTION * pitch
    f_tol = FOOTPRINT_TOLERANCE_FRACTION * pitch
    scores = {}
    for key, f in hyp.items():
        dh = abs(top - f["height"]) / h_tol
        ds = max(abs(long_side - f["sides"][0]), abs(short_side - f["sides"][1])) / f_tol
        scores[key] = float(max(dh, ds))
    out["scores"] = scores
    # Up / down: the share of the inner footprint far under the rim, from every depth
    # point there (the hollow underside's floor is under the brick-point cut).
    perp = np.array([-long_dir[1], long_dir[0]])
    inset = INNER_FRACTION * pitch
    rel_all = pts[:, :2] - np.array([cx, cy])
    inner = (np.abs(rel_all @ long_dir) <= long_side / 2 - inset) & (
        np.abs(rel_all @ perp) <= short_side / 2 - inset
    )
    depth_cut = max(DEEP_FRACTION * body_h, 1.5 * stud_h)
    if inner.sum() >= MIN_POINTS // 2:
        out["deep_share"] = float(np.mean(pts[inner, 2] - table_z < top - depth_cut))
    else:
        out["deep_share"] = float("nan")
    rel = brick[:, :2] - np.array([cx, cy])
    u, v = rel @ long_dir, rel @ perp
    # On a side: which way the studs point (the silhouette reaches past the top wall).
    wall = heights > top - 0.25 * stud_h - 0.001
    out["stud_side"] = 0.0
    if wall.sum() >= MIN_POINTS // 2:
        out["wall_uv"] = (u[wall], v[wall])
        out["all_uv"] = (u, v)
    return out


def run(
    ctx: NodeContext,
    brick_position: dict[str, float],
    brick_studs: list[int] | None = None,
    system: str = "duplo",
    stud_pitch: float | None = None,
    stud_height: float | None = None,
    brick_height: float | None = None,
    table_height: float | None = None,
    view: str = "move",
    camera: str | None = None,
    frames: int = FRAMES,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    if system not in PRESETS:
        raise ValueError(f"system must be one of {sorted(PRESETS)}, got {system!r}")
    if view not in ("move", "current"):
        raise ValueError(f"view must be 'move' or 'current', got {view!r}")
    studs = [int(v) for v in (brick_studs or [2, 2])]
    if len(studs) != 2 or min(studs) < 1:
        raise ValueError(f"brick_studs must be two positive integers, got {brick_studs!r}")
    preset = PRESETS[system]
    pitch = float(stud_pitch if stud_pitch is not None else preset["stud_pitch"])
    stud_h = float(stud_height if stud_height is not None else preset["stud_height"])
    body_h = float(brick_height if brick_height is not None else preset["brick_height"])
    nx, ny = studs
    near = _xyz(brick_position)
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}

    out: Output = {
        "route": "unknown",
        "detail": "",
        "vertical_axis": "",
        "up_vector": _vec(np.zeros(3)),
        "stud_direction_known": False,
        "yaw_deg": 0.0,
        "centre": _vec(near),
        "height": 0.0,
        "footprint": [0.0, 0.0],
        "table_height": 0.0,
        "deep_share": 0.0,
        "scores": {},
        "samples": 0,
    }

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        return out

    try:
        obs = ctx.tool("robot.get_observation")
    except Exception as exc:
        return finish("unknown", f"no observation ({exc})")
    tool = _tool_pose(obs, arm_id)
    cam = _pick_camera(obs, tool, camera)
    if cam is None:
        return finish("unknown", "no camera with a pose (named, or a wrist camera)")
    table_guess = float(table_height) if table_height is not None else float(near[2])
    top_guess = table_guess + max(f["height"] for f in faces(nx, ny, pitch, stud_h, body_h).values())
    target = np.array([near[0], near[1], top_guess])

    tries: list[str] = []
    if view == "current":
        candidates: list[np.ndarray | None] = [None]
    else:
        if tool is None:
            return finish("unknown", "no tool pose in the observation to place the view")
        tool_cam = np.linalg.inv(tool) @ _matrix(cam["pose"])
        candidates = list(_views(tool, tool_cam, target))
    for view_pose in candidates:
        if view_pose is not None:
            if view_pose[2, 3] < top_guess + MIN_VIEW_CLEARANCE:
                tries.append("a view under the clearance floor")
                continue
            if check_reach and not _ik_ok(ctx, view_pose, on_arm):
                tries.append("a view out of reach (IK preview)")
                continue
            pose = {"position": _vec(view_pose[:3, 3]), "rotation": _quat(view_pose[:3, :3])}
            try:
                why = _refusal(ctx.tool("robot.go_to_pose", pose=pose, **on_arm))
            except Exception as exc:  # a station raises where a simulator routes
                why = str(exc)
            if why and any(r in why for r in REFUSALS):
                tries.append(f"a view move refused: {why}")
                continue
        depth, cam_now, why = _read_depth(ctx, arm_id, camera, frames)
        if depth is None:
            tries.append(why)
            continue
        m = measure(world_points(cam_now, depth), near, nx, ny, pitch, stud_h, body_h, table_height)
        if m["error"]:
            tries.append(m["error"])
            continue
        return judge(m, out, finish, nx, ny, pitch, stud_h, body_h)
    if view == "move" and tries and all("refused" in t or "reach" in t for t in tries):
        return finish("unreachable", "no view of the brick: " + "; ".join(tries))
    return finish("unknown", "not read: " + "; ".join(tries))


def judge(
    m: dict[str, Any],
    out: Output,
    finish: Any,
    nx: int,
    ny: int,
    pitch: float,
    stud_h: float,
    body_h: float,
) -> Output:
    out["samples"] = int(m.get("samples", 0))
    out["table_height"] = float(m.get("table_height", 0.0))
    if m.get("no_brick"):
        return finish(
            "no_brick", f"{out['samples']} points over the table near brick_position: no brick"
        )
    scores = m["scores"]
    out["scores"] = {k: round(v, 3) for k, v in scores.items()}
    out["height"], out["footprint"] = float(m["height"]), [float(s) for s in m["sides"]]
    ds = m["deep_share"]
    out["deep_share"] = 0.0 if math.isnan(ds) else float(ds)
    table_z = out["table_height"]
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    key, best = ranked[0]
    read = (
        f"top {m['height'] * 1e3:.1f} mm over the table, silhouette "
        f"{m['sides'][0] * 1e3:.1f} x {m['sides'][1] * 1e3:.1f} mm, "
        + ", ".join(f"{k} {v:.2f}" for k, v in ranked)
    )
    if best > 1.0:
        return finish("unknown", f"{read}: no resting face of a {nx}x{ny} brick fits")
    if len(ranked) > 1 and ranked[1][1] < MARGIN * best:
        return finish("unknown", f"{read}: {key} and {ranked[1][0]} fit about equally")
    long_dir = m["long_dir"]
    lx, ly = nx * pitch, ny * pitch
    cxy = m["centre_xy"]
    if key == "z":
        # The brick's x axis: along the long side if nx is the longer count.
        x_dir = long_dir if lx >= ly else np.array([-long_dir[1], long_dir[0]])
        period = 90.0 if nx == ny else 180.0
        out["yaw_deg"] = _wrap(math.degrees(math.atan2(x_dir[1], x_dir[0])), period)
        out["vertical_axis"] = "z"
        if math.isnan(ds):
            return finish("unknown", f"{read}: too few inner points to tell up from down")
        if ds >= UPSIDE_DOWN_SHARE:
            out["up_vector"] = _vec([0.0, 0.0, -1.0])
            out["stud_direction_known"] = True
            out["centre"] = _vec([cxy[0], cxy[1], table_z + stud_h + body_h / 2])
            return finish(
                "upside_down", f"{read}; {ds:.0%} of the inner footprint is hollow underside"
            )
        if ds <= STUDS_UP_SHARE:
            out["up_vector"] = _vec([0.0, 0.0, 1.0])
            out["stud_direction_known"] = True
            out["centre"] = _vec([cxy[0], cxy[1], table_z + body_h / 2])
            return finish("studs_up", f"{read}; {ds:.0%} of the inner footprint deep")
        return finish(
            "unknown", f"{read}; {ds:.0%} of the inner footprint deep: neither face nor hollow"
        )
    # On a side: the remaining stud axis lies along the silhouette side of its length,
    # the brick's z (the studs) along the other side, length brick height + stud height.
    stud_len = lx if key == "y" else ly
    deep = body_h + stud_h
    along_long = abs(stud_len - max(stud_len, deep)) < 1e-9
    axis_dir = long_dir if along_long else np.array([-long_dir[1], long_dir[0]])
    z_dir = np.array([-axis_dir[1], axis_dir[0]])
    out["vertical_axis"] = key
    out["yaw_deg"] = _wrap(math.degrees(math.atan2(axis_dir[1], axis_dir[0])), 180.0)
    up = np.array([z_dir[0], z_dir[1], 0.0])
    known = False
    if "wall_uv" in m:
        wu, wv = m["wall_uv"]
        au, av = m["all_uv"]
        cu, cv = float(z_dir @ long_dir), float(z_dir @ np.array([-long_dir[1], long_dir[0]]))
        wz, az = wu * cu + wv * cv, au * cu + av * cv
        plus = float(np.percentile(az, 99) - np.percentile(wz, 99))
        minus = float(np.percentile(wz, 1) - np.percentile(az, 1))
        if abs(plus - minus) >= STUD_SIDE_FRACTION * stud_h:
            known = True
            if minus > plus:
                up = -up
    out["up_vector"] = _vec(up if known else np.abs(up))
    out["stud_direction_known"] = known
    height = (lx if key == "x" else ly) - 2 * FOOTPRINT_PLAY
    out["centre"] = _vec([cxy[0], cxy[1], table_z + height / 2])
    side = "studs toward " + (
        f"({up[0]:+.2f}, {up[1]:+.2f})" if known else "an unresolved side"
    )
    return finish("on_side", f"{read}; the brick's {key} axis is vertical, {side}")


def _wrap(deg: float, period: float) -> float:
    return (deg + period / 2.0) % period - period / 2.0


def _read_depth(
    ctx: NodeContext, arm_id: int, camera: str | None, frames: int
) -> tuple[np.ndarray | None, dict[str, Any], str]:
    stack, cam = [], {}
    for _ in range(max(1, int(frames))):
        try:
            obs = ctx.tool("robot.get_observation")
        except Exception as exc:
            return None, {}, f"no observation ({exc})"
        c = _pick_camera(obs, _tool_pose(obs, arm_id), camera)
        if c is None or c.get("depth") is None:
            return None, {}, "the camera frame has no depth"
        d = np.asarray(c["depth"], dtype=np.float64)
        if d.ndim == 3:
            d = d[..., 0]
        if stack and d.shape != stack[0].shape:
            continue
        stack.append(np.where(d > 0, d, np.nan))
        cam = c
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels: no reading
        depth = np.nanmedian(np.stack(stack), axis=0)
    return np.nan_to_num(depth, nan=0.0), cam, ""


def _views(tool: np.ndarray, tool_cam: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    cam_now = tool @ tool_cam
    down = _align(cam_now[:3, 2], np.array([0.0, 0.0, -1.0])) @ cam_now[:3, :3]
    views = []
    for rng in VIEW_RANGES:
        for cam_rot in (down, cam_now[:3, :3]):
            cam = np.eye(4)
            cam[:3, :3] = cam_rot
            cam[:3, 3] = target - cam_rot[:, 2] * rng
            views.append(cam @ np.linalg.inv(tool_cam))
    return views


def _ik_ok(ctx: NodeContext, m: np.ndarray, on_arm: dict[str, Any]) -> bool:
    try:
        reply = ctx.tool(
            "robot.solve_ik",
            pose={"position": _vec(m[:3, 3]), "rotation": _quat(m[:3, :3])},
            **on_arm,
        )
    except Exception as exc:
        if "unreachable" in str(exc).lower():
            return False
        logger.info("classifying-brick-resting-face: no IK preview (%s)", exc)
        return True
    return not (
        isinstance(reply, dict)
        and (reply.get("success") is False or reply.get("within_servo_tolerance") is False)
    )
