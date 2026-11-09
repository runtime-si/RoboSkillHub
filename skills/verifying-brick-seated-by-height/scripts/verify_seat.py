# Acceptance tolerances adapted from BrickSim (https://github.com/intelligent-control-lab/BrickSim),
# python/bricksim/mdp/connection_thresholds.py configure_assembly_thresholds at commit
# cbd5de3238e6ac12f44ef699e1507a9f16bdafc3 (distance 1 mm, z-angle 5 deg, planar position 2 mm),
# MIT License, Copyright (c) 2026 Haowei Wen; and the LEGO stud height of
# Robotic_Lego_Manipulation (https://github.com/intelligent-control-lab/Robotic_Lego_Manipulation),
# include/Lego.hpp knob_height_ at commit 9bc91654e708e4d8428ecfa76840103cacb2a19a, MIT License.
# Changes: BrickSim applies these limits to privileged stud/hole poses inside its simulator to
# decide whether a connection forms; here they are acceptance tolerances on what a depth camera
# sees after the brick is released: a RANSAC plane through the brick's top face inside its
# expected footprint (stud tops told apart from the face by their height), its height measured
# against the plate surface around the footprint (so a camera's extrinsic error cancels), its
# tilt, and how much of the footprint the brick covers (a brick on the wrong studs leaves part of
# the footprint at plate height). The view move and depth reading follow
# skills/pressing-a-brick-onto-studs (same hub); the planar tolerance scales with the stud pitch.
"""Read a placed LEGO / DUPLO brick's top face from depth and judge whether it is seated."""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PRESETS: dict[str, dict[str, float]] = {
    "lego": {
        "stud_pitch": 0.008,
        "stud_height": 0.0017,
        "brick_height": 0.0096,
    },
    "duplo": {
        "stud_pitch": 0.016,
        "stud_height": 0.0045,
        "brick_height": 0.0192,
    },
}
"""Brick systems [m]. LEGO: 8 mm pitch, 1.7 mm studs (ICL's knob_height_), 9.6 mm bricks.
DUPLO: 16 mm pitch, 4.5 mm studs, 19.2 mm bricks (published DUPLO dimensions; the same
presets as pressing-a-brick-onto-studs)."""

FOOTPRINT_PLAY = 0.0001
"""Each wall stands this far inside the stud lattice [m] (BrickSim's brick model)."""

DISTANCE_TOLERANCE = 0.001
"""BrickSim's assembly distance_tolerance [m]: the default height tolerance, capped at
SEAT_FRACTION of a stud height (0.6 mm LEGO)."""

SEAT_FRACTION = 0.35
"""Seated within this fraction of a stud height (pressing-a-brick-onto-studs)."""

RESTING_FRACTION = 0.6
"""A top this fraction of a stud height or more high is resting on the stud tops
(pressing-a-brick-onto-studs: the twin read +4.5 mm for DUPLO bricks on the studs)."""

Z_ANGLE_TOLERANCE_DEG = 5.0
"""BrickSim's z_angle_tolerance: never accept more tilt than this."""

TILT_FRACTION = 0.5
"""Also tilted when one end of the top rises this fraction of a stud height over the
other along the brick's long side (DUPLO 2x2: 4.0 deg; LEGO 2x4: 1.5 deg), the spread
limit of pressing-a-brick-onto-studs."""

POSITION_FRACTION = 0.25
"""Planar tolerance as a fraction of the pitch: BrickSim's 2 mm at LEGO's 8 mm pitch."""

EDGE_INSET = 0.0015
"""Points are taken this far inside the expected footprint's edge [m] (flying pixels and
the walls' play at the edge; pressing-a-brick-onto-studs' SAMPLE_INSET)."""

RING_INNER = 0.3
RING_OUTER = 1.5
"""The reference ring around the footprint, in pitches outside its edge."""

REFERENCE_WINDOW_FRACTION = 0.5
"""The ring's lowest surface counts as the seat's surface when it reads within this
fraction of a brick height of the given seat height; otherwise the seat's own z is used
(a brick stacked on one of the same footprint has the plate a whole brick lower around it)."""

LOWER_SUPPORT = 0.1
"""A lower surface needs this share of the ring's points to be taken as the reference."""

LEVEL_STEP = 0.00005
"""Grid step of the two-level search [m]."""

PLANE_ITERS = 6
TRIM_SIGMAS = 2.5
"""Two-layer plane fit: reassign / refit rounds, and the residual cut in robust sigmas."""

MIN_POINTS = 40
"""Fewer depth points in the footprint than this: the brick was not read."""

NO_BRICK_COVERAGE = 0.25
"""Less of the footprint than this at brick height: nothing is there."""

DEPTH_RANGE = (0.07, 0.6)
"""Depth readings kept [m] (D405 minimum range 7 cm)."""

FRAMES = 3
"""Depth frames read and median-combined per view."""

VIEW_RANGES = (0.20, 0.25)
MIN_VIEW_CLEARANCE = 0.05
"""Camera ranges over the brick's top and the lowest tool centre over it [m]
(refining-target-from-wrist-closeup / pressing-a-brick-onto-studs)."""

REFUSALS = ("unreachable", "bad_speed_scale")

ROUTES = (
    "seated",
    "not_seated",
    "high_by_stud",
    "tilted",
    "shifted_off_grid",
    "no_brick",
    "unverified",
    "unreachable",
)


class Output(TypedDict):
    route: str
    detail: str
    height_error: float
    tilt_deg: float
    edge_rise: float
    coverage: float
    shift_estimate: dict[str, float]
    top_height: float
    reference: str
    reference_offset: float
    samples: int
    viewed_from: str


def _xyz(v: Any) -> np.ndarray:
    if isinstance(v, dict) and "position" in v:
        v = v["position"]
    return np.array([float(v["x"]), float(v["y"]), float(v["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
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
    """Why a move did not arrive ('' when it did), for connectors that route
    instead of raising."""
    if not isinstance(reply, dict):
        return ""
    outcome = reply.get("outcome")
    if outcome in REFUSALS:
        return f"{outcome}: {reply.get('reason', '')}".rstrip(": ")
    if outcome not in (None, "reached") or reply.get("reached") is False:
        return str(outcome or reply.get("reason") or "not reached")
    if reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "failed")
    return ""


def _pick_camera(obs: dict[str, Any], tool: np.ndarray | None, name: str | None) -> dict | None:
    """The named camera, else the eye-in-hand / wrist camera riding nearest the tool."""
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
    """N x 3 world points of the valid depth pixels."""
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    m = _matrix(cam["pose"])
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    ok = np.isfinite(depth) & (depth >= DEPTH_RANGE[0]) & (depth <= DEPTH_RANGE[1])
    z = depth[ok]
    pc = np.stack([(u[ok] - k[0, 2]) / k[0, 0] * z, (v[ok] - k[1, 2]) / k[1, 1] * z, z], axis=1)
    return pc @ m[:3, :3].T + m[:3, 3]


def _fit_plane(pts: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    """Least squares z = a x + b y + c."""
    a = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
    return np.linalg.lstsq(a, pts[:, 2] if z is None else z, rcond=None)[0]


def _two_layer_level(z: np.ndarray, stud_h: float) -> float:
    """The lower of two levels a stud height apart that best explain heights ``z`` (a
    surface and its stud tops), each residual capped at a stud height so that taller
    things (neighbour bricks) do not pull it."""
    lo, hi = np.percentile(z, [2, 60])
    grid = np.arange(lo, hi + LEVEL_STEP, LEVEL_STEP)
    r = z[None, :] - grid[:, None]
    cost = np.minimum(np.minimum(r**2, (r - stud_h) ** 2), stud_h**2).sum(axis=1)
    return float(grid[int(np.argmin(cost))])


def _lowest_surface(z: np.ndarray, stud_h: float, gap: float) -> float:
    """The lowest stud-covered surface with enough support among heights ``z``: the
    two-level search, repeated on what lies more than ``gap`` under its answer (the plate
    between neighbour bricks that fill most of the ring)."""
    level = _two_layer_level(z, stud_h)
    for _ in range(4):
        lower = z[z < level - gap]
        if lower.size < max(MIN_POINTS, LOWER_SUPPORT * z.size):
            break
        level = _two_layer_level(lower, stud_h)
    return level


def _two_layer_plane(pts: np.ndarray, stud_h: float) -> tuple[np.ndarray, float]:
    """Plane of a stud-covered face: each point is taken as on the face or on a stud
    top (a stud height up, lowered by it), the plane refitted, the worst residuals
    (over TRIM_SIGMAS robust sigmas) dropped. Returns (a, b, c) and the face share."""
    # The slope from a plain fit (studs spread evenly over the face), the level from
    # the two-layer search on what is left.
    plane = _fit_plane(pts)
    plane[2] = 0.0
    plane[2] = _two_layer_level(pts[:, 2] - pts[:, :2] @ plane[:2], stud_h)
    keep = np.ones(len(pts), bool)
    on_face = keep
    for _ in range(PLANE_ITERS):
        base = pts[:, :2] @ plane[:2] + plane[2]
        r = pts[:, 2] - base
        on_face = np.abs(r) <= np.abs(r - stud_h)
        zc = pts[:, 2] - np.where(on_face, 0.0, stud_h)
        rc = zc - base
        mad = float(np.median(np.abs(rc[keep] - np.median(rc[keep])))) * 1.4826 or 1.0e-4
        keep = np.abs(rc) <= TRIM_SIGMAS * mad
        if keep.sum() < 3:
            break
        plane = _fit_plane(pts[keep], zc[keep])
    return plane, float(on_face[keep].mean()) if keep.any() else 0.0


def measure(
    pts: np.ndarray,
    seat: np.ndarray,
    yaw_deg: float,
    len_x: float,
    len_y: float,
    pitch: float,
    stud_h: float,
    body_h: float,
    reference: str = "auto",
) -> dict[str, Any]:
    """Height, tilt and coverage of the brick top over the footprint at ``seat``.

    ``pts`` are world points; the footprint is centred on ``seat`` (x, y) with its x
    axis at ``yaw_deg``. Returns a dict, with ``error`` set when nothing could be read.
    """
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    rel = pts[:, :2] - seat[:2]
    lx = rel @ np.array([c, s])
    ly = rel @ np.array([-s, c])
    hx, hy = len_x / 2.0, len_y / 2.0
    out: dict[str, Any] = {"error": ""}

    # The reference: the lowest surface in a ring around the footprint.
    ox, oy = np.abs(lx) - hx, np.abs(ly) - hy
    outside = np.maximum(ox, oy)
    ring = (outside > RING_INNER * pitch) & (outside < RING_OUTER * pitch)
    seat_z = float(seat[2])
    ref_z, ref_used = seat_z, "world"
    if reference != "world" and int(ring.sum()) >= MIN_POINTS:
        cand = _lowest_surface(pts[ring, 2], stud_h, 0.5 * body_h)
        if abs(cand - seat_z) <= REFERENCE_WINDOW_FRACTION * body_h:
            ref_z, ref_used = cand, "local"
    if reference == "local" and ref_used != "local":
        out["error"] = (
            f"no plate surface within {REFERENCE_WINDOW_FRACTION * body_h * 1e3:.1f} mm of the "
            f"seat height in the ring around the footprint ({int(ring.sum())} points)"
        )
    out["reference"], out["reference_offset"] = ref_used, ref_z - seat_z

    # The footprint, inset; brick points are those over half a brick above the reference.
    inside = (np.abs(lx) <= hx - EDGE_INSET) & (np.abs(ly) <= hy - EDGE_INSET)
    n_in = int(inside.sum())
    out["samples"] = n_in
    if n_in < MIN_POINTS:
        out["error"] = out["error"] or f"{n_in} depth points inside the footprint"
        return out
    fp = pts[inside]
    flx, fly = lx[inside], ly[inside]
    high = fp[:, 2] > ref_z + 0.5 * body_h
    coverage = float(high.mean())
    out["coverage"] = coverage
    if int(high.sum()) < MIN_POINTS or coverage < NO_BRICK_COVERAGE:
        out["no_brick"] = True
        return out
    plane, face_share = _two_layer_plane(fp[high], stud_h)
    out["face_share"] = face_share
    # The brick's top plane at the footprint centre.
    z_top = float(seat[0] * plane[0] + seat[1] * plane[1] + plane[2])
    grad = math.hypot(float(plane[0]), float(plane[1]))
    out["top_height"] = z_top
    out["height_error"] = z_top - (ref_z + body_h)
    out["tilt_deg"] = math.degrees(math.atan(grad))
    out["edge_rise"] = grad * max(len_x, len_y)
    # Which way the uncovered part lies: the brick sits toward its covered side.
    low = ~high
    shift = np.zeros(2)
    if low.any() and coverage < 1.0:
        mx, my = float(np.mean(flx[low])), float(np.mean(fly[low]))
        d = np.array([-mx, -my])
        d /= np.linalg.norm(d) or 1.0
        width = abs(d[0]) * (len_x - 2 * EDGE_INSET) + abs(d[1]) * (len_y - 2 * EDGE_INSET)
        mag = (1.0 - coverage) * width
        shift = mag * (d[0] * np.array([c, s]) + d[1] * np.array([-s, c]))
    out["shift"] = shift
    return out


def run(
    ctx: NodeContext,
    seat_position: dict[str, float],
    brick_studs: list[int] | None = None,
    seat_yaw_deg: float = 0.0,
    system: str = "duplo",
    stud_pitch: float | None = None,
    stud_height: float | None = None,
    brick_height: float | None = None,
    view: str = "move",
    camera: str | None = None,
    reference: str = "auto",
    height_tolerance: float | None = None,
    tilt_tolerance_deg: float | None = None,
    position_tolerance: float | None = None,
    frames: int = FRAMES,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    if system not in PRESETS:
        raise ValueError(f"system must be one of {sorted(PRESETS)}, got {system!r}")
    if view not in ("move", "current"):
        raise ValueError(f"view must be 'move' or 'current', got {view!r}")
    if reference not in ("auto", "local", "world"):
        raise ValueError(f"reference must be 'auto', 'local' or 'world', got {reference!r}")
    preset = PRESETS[system]
    pitch = float(stud_pitch if stud_pitch is not None else preset["stud_pitch"])
    stud_h = float(stud_height if stud_height is not None else preset["stud_height"])
    body_h = float(brick_height if brick_height is not None else preset["brick_height"])
    studs = [int(v) for v in (brick_studs or [2, 2])]
    if len(studs) != 2 or min(studs) < 1:
        raise ValueError(f"brick_studs must be two positive integers, got {brick_studs!r}")
    len_x = studs[0] * pitch - 2 * FOOTPRINT_PLAY
    len_y = studs[1] * pitch - 2 * FOOTPRINT_PLAY
    h_tol = (
        float(height_tolerance)
        if height_tolerance is not None
        else min(DISTANCE_TOLERANCE, SEAT_FRACTION * stud_h)
    )
    tilt_tol = (
        float(tilt_tolerance_deg)
        if tilt_tolerance_deg is not None
        else min(
            Z_ANGLE_TOLERANCE_DEG,
            math.degrees(math.atan(TILT_FRACTION * stud_h / max(len_x, len_y))),
        )
    )
    pos_tol = (
        float(position_tolerance) if position_tolerance is not None else (POSITION_FRACTION * pitch)
    )
    seat = _xyz(seat_position)
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    top_z = float(seat[2] + body_h)
    target = np.array([seat[0], seat[1], top_z])

    out: Output = {
        "route": "unverified",
        "detail": "",
        "height_error": 0.0,
        "tilt_deg": 0.0,
        "edge_rise": 0.0,
        "coverage": 0.0,
        "shift_estimate": _vec(np.zeros(3)),
        "top_height": 0.0,
        "reference": "",
        "reference_offset": 0.0,
        "samples": 0,
        "viewed_from": "",
    }

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        return out

    try:
        obs = ctx.tool("robot.get_observation")
    except Exception as exc:
        return finish("unverified", f"no observation ({exc})")
    tool = _tool_pose(obs, arm_id)
    cam = _pick_camera(obs, tool, camera)
    if cam is None:
        return finish("unverified", "no camera with a pose (named, or a wrist camera)")

    tries: list[str] = []
    if view == "current":
        candidates: list[np.ndarray | None] = [None]
    else:
        if tool is None:
            return finish("unverified", "no tool pose in the observation to place the view")
        tool_cam = np.linalg.inv(tool) @ _matrix(cam["pose"])
        candidates = list(_views(tool, tool_cam, target))
    for view_pose in candidates:
        if view_pose is not None:
            if view_pose[2, 3] < top_z + MIN_VIEW_CLEARANCE:
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
        pts = world_points(cam_now, depth)
        m = measure(pts, seat, float(seat_yaw_deg), len_x, len_y, pitch, stud_h, body_h, reference)
        out["samples"] = int(m.get("samples", 0))
        out["reference"] = str(m.get("reference", ""))
        out["reference_offset"] = float(m.get("reference_offset", 0.0))
        out["coverage"] = float(m.get("coverage", 0.0))
        cpos = _matrix(cam_now["pose"])[:3, 3]
        out["viewed_from"] = (
            f"{cam_now.get('name', 'camera')} {np.linalg.norm(cpos - target) * 1e3:.0f} mm "
            "from the brick's top"
        )
        if m["error"]:
            tries.append(m["error"])
            continue
        return judge(m, out, finish, h_tol, tilt_tol, pos_tol, stud_h)
    if view == "move" and tries and all("refused" in t or "reach" in t for t in tries):
        return finish("unreachable", "no view of the brick: " + "; ".join(tries))
    return finish("unverified", "not read: " + "; ".join(tries))


def judge(
    m: dict[str, Any],
    out: Output,
    finish: Any,
    h_tol: float,
    tilt_tol: float,
    pos_tol: float,
    stud_h: float,
) -> Output:
    cov = float(m.get("coverage", 0.0))
    if m.get("no_brick"):
        return finish(
            "no_brick",
            f"{cov:.0%} of the footprint reads at brick height ({out['samples']} points): "
            "no brick at the seat",
        )
    dh, tilt = float(m["height_error"]), float(m["tilt_deg"])
    shift = np.asarray(m["shift"], dtype=float)
    out["height_error"], out["tilt_deg"] = dh, tilt
    out["edge_rise"], out["top_height"] = float(m["edge_rise"]), float(m["top_height"])
    out["shift_estimate"] = _vec(np.array([shift[0], shift[1], 0.0]))
    off = float(np.linalg.norm(shift))
    read = (
        f"top {dh * 1e3:+.2f} mm from seated ({out['reference']} reference), tilt {tilt:.1f} deg, "
        f"{cov:.0%} of the footprint covered, {out['samples']} points"
    )
    resting = dh >= RESTING_FRACTION * stud_h
    # A brick on the wrong studs leaves only part of the footprint to fit the plane
    # to, so its tilt is not trusted; one resting on the studs half a cell off is
    # reported as resting, with the offset.
    if off > pos_tol and not resting:
        return finish(
            "shifted_off_grid",
            f"{read}: about {off * 1e3:.1f} mm off the footprint (tolerance {pos_tol * 1e3:.1f} mm)",
        )
    if tilt > tilt_tol:
        return finish("tilted", f"{read}: tilted over {tilt_tol:.1f} deg (one end on the studs)")
    if resting:
        return finish(
            "high_by_stud",
            f"{read}: {dh / stud_h:.2f} stud heights high, resting on the stud tops"
            + (f"; about {off * 1e3:.1f} mm off the footprint" if off > pos_tol else ""),
        )
    if abs(dh) <= h_tol:
        return finish("seated", read)
    if dh > 0.0:
        return finish("not_seated", f"{read}: partly pressed (tolerance {h_tol * 1e3:.2f} mm)")
    return finish(
        "unverified", f"{read}: below the seated height; the seat height or calibration is off"
    )


def _read_depth(
    ctx: NodeContext, arm_id: int, camera: str | None, frames: int
) -> tuple[np.ndarray | None, dict[str, Any], str]:
    """The camera's depth, median over ``frames`` observations, with its pose."""
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
    """Hand poses putting ``target`` on the wrist camera's optical axis: looking
    straight down, then with the camera's rotation as it is."""
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
        logger.info("verifying-brick-seated: no IK preview (%s)", exc)
        return True
    return not (
        isinstance(reply, dict)
        and (reply.get("success") is False or reply.get("within_servo_tolerance") is False)
    )
