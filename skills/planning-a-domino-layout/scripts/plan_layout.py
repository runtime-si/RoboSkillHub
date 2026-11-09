# The pitch, its toppling margin and the gate width come from the 2D toppling models in
# topple_physics.py (adapted from maxencelaunayquerre/domino-fall-simulation, MIT, and
# eigenmatt/domino-simulation, CC BY 4.0; see that file's header).
# The layout itself is reimplemented, no code copied: slots along a path at a pitch,
# each yawed to the path's tangent, with a footprint overlap check, is the idea of
# Franka-Domino-Project's place_dominoes/pattern.py (Zhu-Chenyu/Franka-Domino-Project,
# no licence: reference only), extended with arc-length resampling, filleted corners,
# a per-pair check of where a falling tile's top meets the next one on a curve, safety
# gates and a build order. Placing the tiles farthest from the arm first, toward the
# arm, is the ordering rule of alexbaucom17/DominoRobot's FieldPlanner (GPL-3.0: the
# rule only, no code). The hand's clearance check mirrors
# placing-an-upright-domino-beside-neighbours' finger_conflicts.
"""Plan a domino line along a path: slot poses, pitch and toppling margin, safety gates, a build order."""

from __future__ import annotations

import functools
import logging
import math
import os
import sys
from typing import Any, TypedDict

import numpy as np

try:
    from gap import NodeContext
except ImportError:  # annotation only; lets the domino task module import this without gap
    NodeContext = Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topple_physics as physics  # noqa: E402

logger = logging.getLogger(__name__)

TILE_SIZE = {"height": 0.048, "width": 0.024, "thickness": 0.0075}
"""A standard domino standing on its end [m]: 48 x 24 x 7.5 mm (zmatt.net's H5 tiles,
Franka-Domino-Project's), the placing skill's default."""

DOMINO_FRICTION = 0.25
"""Domino-domino dynamic friction: 0.25 +- 0.05 measured on 48 x 24 x 7.5 mm tiles
(zmatt.net/domino-physics). The critical gap barely moves with it: 47.9 mm at 0,
47.8 at 0.25, 47.4 at 0.5, 46.8 at 0.8 (GapChain, 37 mm pitch)."""

PUSH_SPEED_FACTOR = 1.2
"""The trigger push the line must survive, as a multiple of the angular speed that
just tips a lone tile over (2.69 rad/s for a 48 x 7.5 mm tile): a slow push. At this
speed GapChain's line of 48 x 24 x 7.5 mm tiles stalls in the second tile at a 16 mm
pitch and runs at 20 mm and wider."""

RUN_UP = 20
"""Tiles the wave runs before it meets a gate when the critical gap is computed
(upstream create_gap's n_wall = 20): by then it runs at its settled speed, which a
hard knock does not change."""

TOPPLE_MARGIN_MIN = 0.005
"""Least room between a pair's face gap and the critical gap [m]: one tile 5 mm off its
slot (a calibrated YAM station camera's RMS is 5.4 mm; the placing skill's twin placed
within 2 mm). It admits the 1.0 h pitch the twin saw cascade (40.5 mm against 47.8).
The models assume no sliding on the table, which only flatters wide gaps."""

GATE_MARGIN = 0.010
"""Least room between a gate's face gap and the critical gap [m]: two tiles 5 mm off
their slots away from each other (a gate errs on the wide side; it costs only a slot)."""

VALIDATED_PITCH = 1.0
"""The widest pitch, in tile heights, a placed line has been seen to cascade: the
placing skill's YAM station twin placed 6- and 8-tile lines at 1.0 h (48 mm) that all
toppled from one push, and a line at 1.5 h stopped after the first tile. The auto
pitch stays under it; a wider given pitch is planned with a note."""

GATE_EVERY = 10
"""Tiles between two safety gates when none are given (a choice: an accidental topple
while building takes down at most this run)."""

GATE_MAX_SLOTS = 4
"""Widest gate, in empty slots."""

MIN_RUN = 2
"""Fewest standing tiles between two gates, and after the trigger tile."""

GATE_SEARCH = 3
"""How many slots an automatic gate may move from its nominal place to find one the
arm can close without reaching past standing tiles."""

# The placing skill's hand (placing-an-upright-domino-beside-neighbours), mirrored so
# that every slot this plan emits passes its finger check at the step it is placed.
FINGER_HALF_DEPTH = 0.012
"""Half the open linear_4310 blade's extent along the row [m] (the placing skill's)."""

FINGER_BLADE = 0.030
"""One linear_4310 blade's depth along its closing direction [m] (the placing skill's)."""

NEIGHBOUR_CLEARANCE = 0.004
"""Least gap between an open finger and a standing tile or obstacle [m] (the placing skill's)."""

GRIP_CLEARANCE = 0.004
"""The jaws open to the tile's width plus this each side [m] (the placing skill's)."""

LEAD_IN = 0.03
"""The placing skill's last leg along the row from its free end [m]; its start must
be inside the arm's box too."""

OVERLAP_CLEARANCE = 0.001
"""Least gap between two tiles' footprints [m]; under it the slots overlap."""

CONTACT_SHARE = 1.0
"""On a curve the centre of a falling tile's top edge must land within this share of
the next tile's half width (a choice: at 1.0 it lands on the face, and at least half
the falling edge overlaps it; a hit nearer the edge twists the tile instead of
toppling it). On a circle the landing point is about pitch^2 / (2 R) off centre."""

CORNER_RADIUS = 0.08
"""Radius of the arc each polyline corner is rounded with [m] (clipped to fit the
legs). At the auto pitch (36.9 mm) the contact check passes circles down to about
60 mm; 80 mm keeps the landing point about 8 mm off centre, 4 mm inside the face."""

CORRIDOR_HALF_WIDTH = 0.047
"""Half the width of the strip from the arm's base to a slot that the arm sweeps [m]:
half the linear_4310's 94 mm span (the YAM station twin's gripper spec)."""

REACH_TOLERANCE = 0.005
"""A standing tile counts as nearer the arm than a slot only when it is this much
nearer [m]."""

SAMPLE_STEP = 0.0005
"""Spacing of the dense polyline the path is resampled on [m]."""

TANGENT_HALF = 0.002
"""Half the window over which a slot's facing is taken from the path [m]."""


class Output(TypedDict):
    route: str
    detail: str
    slots: list[dict[str, Any]]
    slot_poses: list[dict[str, Any]]
    first_slot: dict[str, Any]
    build_order: list[int]
    build_steps: list[dict[str, Any]]
    gates: list[dict[str, Any]]
    gate_count: int
    gate_width_slots: int
    n_tiles: int
    pitch: float
    pitch_source: str
    face_gap: float
    critical_gap: float
    critical_pitch: float
    topple_margin: float
    max_face_gap: float
    gate_margin: float
    min_inner_gap: float
    max_turn_deg: float
    max_contact_offset: float
    path_length: float
    used_length: float
    trigger: dict[str, Any]
    problems: list[str]


# ----------------------------------------------------------------------------- physics (cached)


@functools.lru_cache(maxsize=64)
def critical_gap(height: float, thickness: float, pitch: float, mu: float) -> float:
    """The widest face-to-face gap [m] a wave settled at ``pitch`` still crosses."""
    return physics.critical_gap(height, thickness, pitch, mu, RUN_UP)


@functools.lru_cache(maxsize=64)
def push_runs(height: float, thickness: float, pitch: float, mu: float) -> bool:
    """Whether a slow trigger push (PUSH_SPEED_FACTOR) runs a straight line at ``pitch``."""
    w0 = PUSH_SPEED_FACTOR * physics.tipping_speed(height, thickness)
    return physics.chain_crosses(height, thickness, pitch, mu, None, 11, 0, omega0=w0)


def auto_pitch(crit: float, thickness: float, height: float, lo: float) -> float:
    """The pitch whose one-slot gate clears the critical gap by as much as the line's
    face gap stays under it: crit - (p - d) = (2p - d) - crit, so p = 2 (crit + d) / 3;
    kept between ``lo`` and VALIDATED_PITCH tile heights."""
    p = 2.0 * (crit + thickness) / 3.0
    return float(min(max(p, lo), VALIDATED_PITCH * height))


# ----------------------------------------------------------------------------- the path


def _xy(p: Any) -> np.ndarray:
    if isinstance(p, dict):
        q = p.get("position") or p.get("center") or p
        if not isinstance(q, dict):
            return _xy(q)
        return np.array([float(q["x"]), float(q["y"])])
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    if a.size < 2:
        raise ValueError(f"a point needs x and y: {p!r}")
    return a[:2].copy()


def _z(p: Any) -> float | None:
    if isinstance(p, dict):
        q = p.get("position") or p.get("center") or p
        if not isinstance(q, dict):
            return _z(q)
        return float(q["z"]) if "z" in q else None
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    return float(a[2]) if a.size > 2 else None


def _densify(points: np.ndarray, step: float = SAMPLE_STEP) -> np.ndarray:
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:], strict=True):
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def _arc_points(center: np.ndarray, radius: float, start: float, sweep: float) -> np.ndarray:
    n = max(2, int(math.ceil(abs(sweep) * radius / SAMPLE_STEP)) + 1)
    a = start + np.linspace(0.0, sweep, n)
    return center + radius * np.stack([np.cos(a), np.sin(a)], axis=1)


def _fillet(points: np.ndarray, radius: float) -> np.ndarray:
    """A polyline with each corner replaced by a tangent arc of ``radius`` (smaller
    where the legs are too short for it)."""
    if len(points) < 3 or radius <= 0.0:
        return points
    out = [points[0]]
    for i in range(1, len(points) - 1):
        a, b, c = out[-1], points[i], points[i + 1]
        u, v = b - a, c - b
        lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        if lu < 1e-9 or lv < 1e-9:
            continue
        u, v = u / lu, v / lv
        turn = math.atan2(u[0] * v[1] - u[1] * v[0], float(u @ v))
        if abs(turn) < 1e-6:
            out.append(b)
            continue
        cut = min(radius * math.tan(abs(turn) / 2.0), 0.5 * lu, 0.5 * lv)
        r = cut / math.tan(abs(turn) / 2.0)
        p0 = b - u * cut
        normal = np.array([-u[1], u[0]]) * math.copysign(1.0, turn)
        centre = p0 + normal * r
        start = math.atan2(p0[1] - centre[1], p0[0] - centre[0])
        out.extend(_arc_points(centre, r, start, turn))
    out.append(points[-1])
    return np.array(out)


def parse_path(path: Any) -> tuple[np.ndarray, str, float | None]:
    """The path as a dense world-xy polyline from its start (the trigger end), its
    kind, and a height when the points carry one."""
    if isinstance(path, (list, tuple)):
        path = {"type": "polyline", "points": list(path)}
    if not isinstance(path, dict):
        raise ValueError("path must be a dict or a list of points")
    kind = str(path.get("type") or ("curve" if "points" in path else "line")).lower()
    z = None
    if kind == "line":
        start = _xy(path["start"])
        z = _z(path["start"])
        if "end" in path:
            end = _xy(path["end"])
        else:
            h = math.radians(float(path["heading_deg"]))
            end = start + float(path["length"]) * np.array([math.cos(h), math.sin(h)])
        pts = _densify(np.array([start, end]))
    elif kind == "arc":
        centre = _xy(path["center"])
        z = _z(path["center"])
        radius = float(path["radius"])
        if radius <= 0.0:
            raise ValueError("an arc needs a positive radius")
        if "sweep_deg" in path:
            sweep = math.radians(float(path["sweep_deg"]))
        else:
            sweep = math.radians(float(path["end_deg"]) - float(path["start_deg"]))
        pts = _arc_points(centre, radius, math.radians(float(path.get("start_deg", 0.0))), sweep)
    elif kind in ("polyline", "curve"):
        raw = list(path.get("points") or [])
        if len(raw) < 2:
            raise ValueError(f"a {kind} needs at least two points")
        z = _z(raw[0])
        pts = np.array([_xy(p) for p in raw])
        keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9])
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError(f"the {kind}'s points are all the same")
        if kind == "polyline":
            pts = _fillet(pts, float(path.get("corner_radius", CORNER_RADIUS)))
        pts = _densify(pts)
    else:
        raise ValueError(f"unknown path type {kind!r} (line, arc, polyline, curve)")
    return pts, kind, z


def _arc_lengths(pts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])


def _at(pts: np.ndarray, s_all: np.ndarray, s: float) -> np.ndarray:
    s = min(max(s, 0.0), float(s_all[-1]))
    return np.array([np.interp(s, s_all, pts[:, 0]), np.interp(s, s_all, pts[:, 1])])


def _heading(pts: np.ndarray, s_all: np.ndarray, s: float) -> float:
    lo = max(0.0, s - TANGENT_HALF)
    hi = min(float(s_all[-1]), s + TANGENT_HALF)
    d = _at(pts, s_all, hi) - _at(pts, s_all, lo)
    return math.degrees(math.atan2(float(d[1]), float(d[0])))


# ----------------------------------------------------------------------------- tile geometry


def _axes(facing_deg: float) -> tuple[np.ndarray, np.ndarray]:
    a = math.radians(facing_deg)
    return np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])


def _rect(xy: np.ndarray, facing_deg: float, half_along: float, half_across: float) -> np.ndarray:
    along, across = _axes(facing_deg)
    return np.array(
        [
            xy + sa * half_along * along + sw * half_across * across
            for sa, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
    )


def footprint(xy: np.ndarray, facing_deg: float, size: dict[str, float]) -> np.ndarray:
    return _rect(xy, facing_deg, size["thickness"] / 2.0, size["width"] / 2.0)


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-18), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _separated(a: np.ndarray, b: np.ndarray) -> bool:
    for poly in (a, b):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return True
    return False


def poly_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two convex polygons [m]; 0 when they touch or overlap."""
    if not _separated(a, b):
        return 0.0
    best = math.inf
    for p, q in ((a, b), (b, a)):
        for v in p:
            for i in range(len(q)):
                best = min(best, _seg_dist(v, q[i], q[(i + 1) % len(q)]))
    return best


def contact(
    xy_i: np.ndarray, f_i: float, xy_j: np.ndarray, f_j: float, size: dict[str, float]
) -> tuple[float, float]:
    """Where tile i, falling along its facing, meets tile j: the distance its top
    travels from i's front face to j's back face along i's centre line (the face gap
    the wave must cross; p - d on a straight line), and how far off j's centre across
    its width it lands [m]. inf when the faces are parallel or turned away."""
    d = size["thickness"]
    u_i, _ = _axes(f_i)
    u_j, v_j = _axes(f_j)
    cos = float(u_i @ u_j)
    if cos <= 1e-6:
        return math.inf, math.inf
    front = xy_i + 0.5 * d * u_i
    back = xy_j - 0.5 * d * u_j
    s = float((back - front) @ u_j) / cos
    hit = front + s * u_i
    return s, abs(float((hit - xy_j) @ v_j))


def finger_conflicts(
    slot_xy: np.ndarray,
    facing_deg: float,
    neighbours: list[tuple[np.ndarray, float]],
    size: dict[str, float],
    open_width: float,
) -> list[int]:
    """Neighbours whose footprint comes within NEIGHBOUR_CLEARANCE of the open hand
    over the slot (placing-an-upright-domino-beside-neighbours' check, same rule)."""
    along, across = _axes(facing_deg)
    reach_a = max(FINGER_HALF_DEPTH, size["thickness"] / 2.0) + NEIGHBOUR_CLEARANCE
    reach_w = open_width / 2.0 + FINGER_BLADE + NEIGHBOUR_CLEARANCE
    bad = []
    for i, (xy, f) in enumerate(neighbours):
        c = footprint(xy, f, size) - slot_xy
        a, w = c @ along, c @ across
        if a.min() < reach_a and a.max() > -reach_a and w.min() < reach_w and w.max() > -reach_w:
            bad.append(i)
    return bad


def hand_envelope(
    xy: np.ndarray, facing_deg: float, size: dict[str, float], open_width: float
) -> np.ndarray:
    """The open hand's footprint over a slot: both blades beside the tile, across its width."""
    return _rect(
        xy,
        facing_deg,
        max(FINGER_HALF_DEPTH, size["thickness"] / 2.0),
        open_width / 2.0 + FINGER_BLADE,
    )


def obstacle_shape(ob: dict[str, Any], size: dict[str, float]) -> tuple[str, Any]:
    """("circle", (xy, r)) or ("poly", corners) from a circle (``radius``), a box
    (``extent`` half sizes or ``size`` full sizes, with ``yaw_deg`` or a rotation),
    an axis-aligned ``min``/``max`` box, or a standing tile's pose (``facing_deg`` or a
    rotation and no size)."""
    if "min" in ob and "max" in ob:
        lo, hi = _xy(ob["min"]), _xy(ob["max"])
        return "poly", np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])
    xy = _xy(ob)
    if "radius" in ob:
        return "circle", (xy, float(ob["radius"]))
    yaw = float(ob.get("yaw_deg", ob.get("facing_deg", 0.0)))
    rot = ob.get("rotation") or ob.get("orientation")
    if rot and "yaw_deg" not in ob and "facing_deg" not in ob:
        w, x, y, z = (float(rot[k]) for k in "wxyz")
        yaw = math.degrees(math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))
    if "extent" in ob or "size" in ob:
        e = ob.get("extent")
        if e is not None:
            half = _xy(e) if not isinstance(e, dict) else np.array([float(e["x"]), float(e["y"])])
        else:
            s = ob["size"]
            half = 0.5 * (
                _xy(s) if not isinstance(s, dict) else np.array([float(s["x"]), float(s["y"])])
            )
        return "poly", _rect(xy, yaw, float(half[0]), float(half[1]))
    return "poly", footprint(xy, yaw, size)


def shape_distance(poly: np.ndarray, shape: tuple[str, Any]) -> float:
    kind, val = shape
    if kind == "circle":
        c, r = val
        inside = not _separated(poly, np.array([c, c + 1e-9, c + np.array([0.0, 1e-9])]))
        if inside:
            return 0.0
        d = min(_seg_dist(c, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))
        return max(0.0, d - r)
    return poly_distance(poly, val)


# ----------------------------------------------------------------------------- the arms


def _bounds_xy(b: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(b, dict):
        return None
    lo = b.get("min_m", b.get("min"))
    hi = b.get("max_m", b.get("max"))
    if lo is None or hi is None:
        return None
    return _xy(lo), _xy(hi)


def _arms(
    ctx: NodeContext, arms: list[dict[str, Any]] | None, problems: list[str]
) -> list[dict[str, Any]]:
    """Each arm's id, base (world xy), tool-centre box (xy) and reach annulus."""
    if not arms:
        arms = [{}]
        try:
            ws = ctx.tool("robot.describe_workspace")
            ws = ws if isinstance(ws, dict) else {}
            arms = [
                {
                    "arm_id": int(ws.get("arm_id", 0) or 0),
                    "base": ws.get("base_in_left_m") or [0.0, 0.0],
                    "bounds": ws.get("bounds"),
                }
            ]
        except Exception as exc:  # a planner without a workspace checks no reach
            problems.append(f"no workspace description ({exc}); reach not checked")
    out = []
    for k, a in enumerate(arms):
        reach = a.get("reach") or a.get("reach_m")
        out.append(
            {
                "arm_id": int(a.get("arm_id", k)),
                "base": _xy(a.get("base", [0.0, 0.0])),
                "bounds": _bounds_xy(a.get("bounds")),
                "reach": (float(reach[0]), float(reach[1])) if reach else None,
            }
        )
    return out


def _in_box(xy: np.ndarray, arm: dict[str, Any]) -> bool:
    b = arm["bounds"]
    if b is not None and not (np.all(xy >= b[0] - 1e-9) and np.all(xy <= b[1] + 1e-9)):
        return False
    if arm["reach"] is not None:
        r = float(np.linalg.norm(xy - arm["base"]))
        if not arm["reach"][0] <= r <= arm["reach"][1]:
            return False
    return True


def in_corridor(base: np.ndarray, slot: np.ndarray, xy: np.ndarray) -> bool:
    """Whether a standing tile at ``xy`` lies on the arm's way from its base to the
    slot: nearer the base along that line by REACH_TOLERANCE and within
    CORRIDOR_HALF_WIDTH of it."""
    way = slot - base
    dist = float(np.linalg.norm(way))
    if dist < 1e-9:
        return False
    u = way / dist
    rel = xy - base
    along = float(rel @ u)
    across = abs(float(rel[0] * u[1] - rel[1] * u[0]))
    return -CORRIDOR_HALF_WIDTH < along < dist - REACH_TOLERANCE and across < CORRIDOR_HALF_WIDTH


def _quat(facing_deg: float) -> dict[str, float]:
    h = math.radians(facing_deg) / 2.0
    return {"w": math.cos(h), "x": 0.0, "y": 0.0, "z": math.sin(h)}


# ----------------------------------------------------------------------------- the skill


def run(
    ctx: NodeContext,
    path: Any,
    table_z: float | None = None,
    pitch: float | str = "auto",
    n_tiles: int = 0,
    length: float = 0.0,
    tile_size: dict[str, float] | None = None,
    gates: list[Any] | None = None,
    gate_every: int = GATE_EVERY,
    arms: list[dict[str, Any]] | None = None,
    table: dict[str, Any] | None = None,
    obstacles: list[dict[str, Any]] | None = None,
    friction: float = DOMINO_FRICTION,
    topple_margin_min: float = TOPPLE_MARGIN_MIN,
    gate_margin_min: float = GATE_MARGIN,
) -> Output:
    size = {**TILE_SIZE, **{k: float(v) for k, v in (tile_size or {}).items()}}
    h, w, d = size["height"], size["width"], size["thickness"]
    mu = float(friction)
    problems: list[str] = []
    out: Output = {
        "route": "invalid_input",
        "detail": "",
        "slots": [],
        "slot_poses": [],
        "first_slot": {},
        "build_order": [],
        "build_steps": [],
        "gates": [],
        "gate_count": 0,
        "gate_width_slots": 0,
        "n_tiles": 0,
        "pitch": 0.0,
        "pitch_source": "",
        "face_gap": 0.0,
        "critical_gap": 0.0,
        "critical_pitch": 0.0,
        "topple_margin": 0.0,
        "max_face_gap": 0.0,
        "gate_margin": 0.0,
        "min_inner_gap": 0.0,
        "max_turn_deg": 0.0,
        "max_contact_offset": 0.0,
        "path_length": 0.0,
        "used_length": 0.0,
        "trigger": {},
        "problems": problems,
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        return out

    # --- the path and the table
    try:
        pts, kind, path_z = parse_path(path)
    except (KeyError, TypeError, ValueError) as exc:
        return finish("invalid_input", f"path: {exc}")
    if min(h, w, d) <= 0.0 or d >= h:
        return finish("invalid_input", f"tile_size {size} is not a standing tile")
    s_all = _arc_lengths(pts)
    total = float(s_all[-1])
    out["path_length"] = total
    usable = min(total, float(length)) if length and float(length) > 0.0 else total
    if table_z is None:
        try:
            table_z = float(ctx.tool("robot.describe_workspace")["surface_z"])
        except Exception as exc:  # the path's own height, else 0
            logger.info("planning-a-domino-layout: no workspace surface (%s)", exc)
            table_z = path_z if path_z is not None else 0.0
    table_z = float(table_z)

    # --- the pitch: the hand's floor, the physics' ceiling
    open_width = w + 2.0 * GRIP_CLEARANCE
    finger_pitch = max(FINGER_HALF_DEPTH, d / 2.0) + NEIGHBOUR_CLEARANCE + d / 2.0
    guess = float(pitch) if not isinstance(pitch, str) else 0.75 * h
    if guess <= d:
        return finish(
            "too_tight",
            f"pitch {guess * 1e3:.1f} mm is not over the tile's {d * 1e3:.1f} mm thickness",
        )
    # The wave that meets a gap runs at a pitch the line itself carries.
    ref = min(max(guess, finger_pitch), VALIDATED_PITCH * h)
    crit = critical_gap(h, d, round(ref, 4), mu)
    if crit <= 0.0:
        return finish(
            "too_loose", f"a straight line of these tiles at {ref * 1e3:.1f} mm does not topple"
        )
    out["critical_gap"] = crit
    out["critical_pitch"] = crit + d
    if isinstance(pitch, str):
        if pitch.strip().lower() != "auto":
            return finish("invalid_input", f"pitch must be a number or 'auto', not {pitch!r}")
        p = auto_pitch(crit, d, h, finger_pitch)
        source = "auto"
        if n_tiles and int(n_tiles) > 1 and (int(n_tiles) - 1) * p > usable:
            p = usable / (int(n_tiles) - 1)
            source = "auto, shrunk to fit"
    else:
        p, source = float(pitch), "given"
    out["pitch"], out["pitch_source"] = p, source
    out["face_gap"] = p - d
    if n_tiles and int(n_tiles) > 0:
        n = int(n_tiles)
        if (n - 1) * p > usable + 1e-9:
            return finish(
                "too_short",
                f"{n} tiles at {p * 1e3:.1f} mm need {(n - 1) * p * 1e3:.0f} mm of path; "
                f"it has {usable * 1e3:.0f} mm",
            )
    else:
        n = int(math.floor(usable / p + 1e-9)) + 1
    if n < 2:
        return finish(
            "too_short", f"the path ({usable * 1e3:.0f} mm) holds one tile at {p * 1e3:.1f} mm"
        )
    out["n_tiles"] = n
    out["used_length"] = (n - 1) * p
    if p < finger_pitch - 1e-9:
        return finish(
            "too_tight",
            f"pitch {p * 1e3:.1f} mm: the open fingers over a slot need {finger_pitch * 1e3:.1f} mm "
            "to the next tile's centre (placing-an-upright-domino-beside-neighbours' check)",
        )

    # --- the slots
    xy = [_at(pts, s_all, k * p) for k in range(n)]
    facing = [_heading(pts, s_all, k * p) for k in range(n)]
    out["slots"] = [
        {
            "index": k,
            "position": {"x": float(xy[k][0]), "y": float(xy[k][1]), "z": table_z},
            "rotation": _quat(facing[k]),
            "facing_deg": float(facing[k]),
            "arc_length": float(k * p),
            "gate": False,
        }
        for k in range(n)
    ]
    out["slot_poses"] = [
        {"position": s["position"], "rotation": s["rotation"]} for s in out["slots"]
    ]
    out["first_slot"] = out["slot_poses"][0]
    u0, _ = _axes(facing[0])
    out["trigger"] = {
        "slot_index": 0,
        "pose": out["slot_poses"][0],
        "push_direction": {"x": float(u0[0]), "y": float(u0[1]), "z": 0.0},
    }

    # --- the pairs: overlap at inner corners, where the falling top lands, the gap
    feet = [footprint(xy[k], facing[k], size) for k in range(n)]
    gaps, offsets, inner, turns = [], [], [], []
    for k in range(n - 1):
        s, off = contact(xy[k], facing[k], xy[k + 1], facing[k + 1], size)
        gaps.append(s)
        offsets.append(off)
        inner.append(poly_distance(feet[k], feet[k + 1]))
        turns.append(abs((facing[k + 1] - facing[k] + 180.0) % 360.0 - 180.0))
    out["min_inner_gap"] = float(min(inner))
    out["max_turn_deg"] = float(max(turns))
    out["max_contact_offset"] = float(max(offsets))
    out["max_face_gap"] = float(max(gaps))
    out["topple_margin"] = float(crit - max(gaps))
    tight = [k for k in range(n - 1) if inner[k] < OVERLAP_CLEARANCE]
    if tight:
        k = tight[0]
        return finish(
            "too_tight",
            f"slots {k} and {k + 1} overlap at the inner corner ({inner[k] * 1e3:.1f} mm apart, "
            f"{turns[k]:.0f} deg turn): the curve is too sharp for {p * 1e3:.1f} mm",
        )
    for a in range(n):
        for b in range(a + 2, n):
            if poly_distance(feet[a], feet[b]) < OVERLAP_CLEARANCE:
                return finish(
                    "too_tight", f"slots {a} and {b} overlap: the path comes back on itself"
                )
    glancing = [k for k in range(n - 1) if offsets[k] > CONTACT_SHARE * w / 2.0]
    if glancing:
        k = glancing[0]
        return finish(
            "too_tight",
            f"slot {k}'s top meets slot {k + 1} {offsets[k] * 1e3:.1f} mm off its centre "
            f"(over {CONTACT_SHARE * w / 2.0 * 1e3:.0f} mm, {turns[k]:.0f} deg turn): the curve is too sharp",
        )
    loose = [k for k in range(n - 1) if gaps[k] > crit - float(topple_margin_min)]
    if loose:
        k = max(loose, key=lambda i: gaps[i])
        return finish(
            "too_loose",
            f"the wave must cross {gaps[k] * 1e3:.1f} mm from slot {k} to {k + 1}; it crosses "
            f"{crit * 1e3:.1f} mm at most, less the {float(topple_margin_min) * 1e3:.0f} mm margin",
        )
    if not push_runs(h, d, round(p, 4), mu):
        return finish(
            "too_tight",
            f"pitch {p * 1e3:.1f} mm: a slow trigger push stalls in the second tile "
            "(it leans on the next one before the first has tipped over)",
        )
    if p > VALIDATED_PITCH * h + 1e-9:
        problems.append(
            f"pitch {p * 1e3:.1f} mm is over the {VALIDATED_PITCH:.1f} h a placed line was seen to cascade at"
        )

    # --- reach and the table
    arm_list = _arms(ctx, arms, problems)
    if table:
        tb = _bounds_xy(table)
        if tb is not None:
            off = [
                k
                for k in range(n)
                if not (np.all(feet[k] >= tb[0] - 1e-9) and np.all(feet[k] <= tb[1] + 1e-9))
            ]
            if off:
                return finish("out_of_reach", f"slots {off} are off the table")
    owner: list[int] = []
    for k in range(n):
        ok = [i for i, a in enumerate(arm_list) if _in_box(xy[k], a)]
        if not ok:
            far = [k2 for k2 in range(n) if not any(_in_box(xy[k2], a) for a in arm_list)]
            return finish("out_of_reach", f"slots {far} are outside every arm's box or reach")
        owner.append(min(ok, key=lambda i: float(np.linalg.norm(xy[k] - arm_list[i]["base"]))))

    # --- obstacles
    shapes = [obstacle_shape(ob, size) for ob in (obstacles or [])]
    for k in range(n):
        env = hand_envelope(xy[k], facing[k], size, open_width)
        for j, sh in enumerate(shapes):
            if shape_distance(env, sh) < NEIGHBOUR_CLEARANCE:
                return finish("blocked", f"obstacle {j} is where the open hand goes over slot {k}")

    # --- gates
    def gate_gap(first: int, k: int) -> float:
        a, b = first - 1, first + k
        return contact(xy[a], facing[a], xy[b], facing[b], size)[0]

    def width_for(first: int) -> int:
        for k in range(1, GATE_MAX_SLOTS + 1):
            if first + k > n - 1:
                return 0
            if gate_gap(first, k) >= crit + float(gate_margin_min):
                return k
        return 0

    def lead_in_fits(k: int, arm: int, standing: set[int]) -> bool:
        """Whether the placing skill's lead-in point at slot k (LEAD_IN along the row,
        on the side away from the nearest standing tile; either side on an empty
        table) is inside the arm's box."""
        along, _ = _axes(facing[k])
        if standing:
            near = min(standing, key=lambda t: float(np.linalg.norm(xy[t] - xy[k])))
            side = -1.0 if float((xy[near] - xy[k]) @ along) > 0.0 else 1.0
            leads = [xy[k] + side * LEAD_IN * along]
        else:
            leads = [xy[k] + LEAD_IN * along, xy[k] - LEAD_IN * along]
        return any(_in_box(q, arm_list[arm]) for q in leads)

    def arm_for(k: int, standing: set[int], beside: set[int]) -> tuple[int | None, str]:
        """An arm that can place slot k now: the slot and the lead-in in its box, and no
        standing tile on its way other than the tiles ``beside`` it on the path (the
        hand's own neighbours, which the placing skill's finger check clears).
        (None, "corridor" | "lead-in") when there is none."""
        cands = sorted(
            (i for i, a in enumerate(arm_list) if _in_box(xy[k], a)),
            key=lambda i: (i != owner[k], float(np.linalg.norm(xy[k] - arm_list[i]["base"]))),
        )
        why = "corridor"
        for i in cands:
            base = arm_list[i]["base"]
            if any(in_corridor(base, xy[k], xy[t]) for t in standing - beside):
                continue
            if not lead_in_fits(k, i, standing):
                why = "lead-in"
                continue
            return i, ""
        return None, why

    def run_of(s: int, step: int, standing: set[int]) -> int:
        """Standing tiles in a row from s + step on."""
        count, t = 0, s + step
        while 0 <= t < n and t in standing:
            count += 1
            t += step
        return count

    groups: list[list[int]] = []
    skipped: list[int] = []
    if gates is not None:
        for g in gates:
            first = (
                int(g)
                if isinstance(g, (int, np.integer))
                else int(np.argmin([float(np.linalg.norm(_xy(g) - c)) for c in xy]))
            )
            if not MIN_RUN <= first <= n - 1 - MIN_RUN:
                return finish(
                    "blocked",
                    f"gate at slot {first}: a gate needs {MIN_RUN} standing tiles before and after it",
                )
            k = width_for(first)
            if not k or first + k > n - MIN_RUN:
                return finish(
                    "blocked",
                    f"gate at slot {first}: no gate of up to {GATE_MAX_SLOTS} slots stops the wave there",
                )
            g_slots = list(range(first, first + k))
            if any(abs(s - t) < MIN_RUN + 1 for grp in groups for s in g_slots for t in grp):
                return finish("blocked", f"gate at slot {first} is too close to another gate")
            groups.append(g_slots)
    elif int(gate_every) > 0:
        nominal = int(gate_every)
        while nominal <= n - 1 - MIN_RUN:
            placed = None
            for shift in sorted(range(-GATE_SEARCH, GATE_SEARCH + 1), key=abs):
                first = nominal + shift
                if not MIN_RUN <= first <= n - 1 - MIN_RUN:
                    continue
                if groups and first - groups[-1][-1] - 1 < MIN_RUN:
                    continue
                k = width_for(first)
                if not k or first + k > n - MIN_RUN:
                    continue
                g_slots = list(range(first, first + k))
                standing = set(range(n)) - set(g_slots)
                beside = {g_slots[0] - 1, g_slots[-1] + 1}
                if all(arm_for(s, standing, beside)[0] is not None for s in g_slots):
                    placed = g_slots
                    break
            if placed is None:
                skipped.append(nominal)
                nominal += int(gate_every)
                continue
            groups.append(placed)
            nominal = placed[-1] + 1 + int(gate_every)
    if skipped:
        problems.append(
            f"no closable gate near slots {skipped} (the arm would reach past standing tiles there)"
        )
    gate_set = {s for grp in groups for s in grp}
    for s in gate_set:
        out["slots"][s]["gate"] = True
    out["gate_count"] = len(groups)
    out["gate_width_slots"] = max((len(g) for g in groups), default=0)

    # --- the build order: farthest from its arm first, then the gates
    steps: list[dict[str, Any]] = []
    standing: set[int] = set()

    def place(k: int, arm: int, gate: bool) -> tuple[str, str]:
        """("", "") when slot k can be placed now, else (route, why)."""
        nbrs = sorted(standing)
        clash = finger_conflicts(
            xy[k], facing[k], [(xy[t], facing[t]) for t in nbrs], size, open_width
        )
        if clash:
            return (
                "too_tight",
                f"slot {k}: the open fingers would touch standing slot(s) {[nbrs[c] for c in clash]}",
            )
        steps.append(
            {
                "slot_index": k,
                "arm_id": arm_list[arm]["arm_id"],
                "neighbours": nbrs,
                "gate": gate,
            }
        )
        standing.add(k)
        return "", ""

    todo = [k for k in range(n) if k not in gate_set]
    holes = 0
    while todo:
        # Farthest from its arm first; a slot at the free end of a run before one
        # between two standing tiles; the trigger end last.
        todo.sort(
            key=lambda k: (
                (k - 1 in standing) and (k + 1 in standing),
                -float(np.linalg.norm(xy[k] - arm_list[owner[k]]["base"])),
                -k,
            )
        )
        reasons = []
        for k in todo:
            arm, why = arm_for(k, standing, {k - 1, k + 1})
            if arm is not None:
                break
            reasons.append(why)
        else:
            k = todo[0]
            if reasons[0] == "lead-in":
                return finish(
                    "out_of_reach",
                    f"slot {k}: the placing skill's lead-in ({LEAD_IN * 1e3:.0f} mm along the row "
                    "from its free end) is outside every arm's box that reaches the slot",
                )
            base = arm_list[owner[k]]["base"]
            hit = [t for t in sorted(standing - {k - 1, k + 1}) if in_corridor(base, xy[k], xy[t])]
            return finish(
                "blocked",
                f"no order places slots {sorted(todo)} without reaching past standing tiles "
                f"(slot {k} is behind {hit})",
            )
        hole = (k - 1 in standing) and (k + 1 in standing)
        holes += int(hole)
        at_risk = run_of(k, -1, standing) + run_of(k, 1, standing) + 1
        route, why = place(k, arm, False)
        if route:
            return finish(route, why)
        steps[-1]["between_standing"] = hole
        steps[-1]["at_risk"] = at_risk
        todo.remove(k)

    # Gates close smallest merged run first: a slip while closing one takes down only
    # the runs it joins.
    margins = []
    open_groups = list(groups)
    gate_out = []
    while open_groups:
        open_groups.sort(
            key=lambda g: (
                run_of(g[0], -1, standing) + run_of(g[-1], 1, standing) + len(g),
                -float(np.linalg.norm(xy[g[0]] - arm_list[owner[g[0]]]["base"])),
            )
        )
        for g in open_groups:
            beside = {g[0] - 1, g[-1] + 1}
            arms_ok = [
                arm_for(s, standing | set(g[:i]), beside | set(g))[0] for i, s in enumerate(g)
            ]
            if all(a is not None for a in arms_ok):
                break
        else:
            g = open_groups[0]
            return finish(
                "blocked",
                f"the gate at slots {g} cannot be closed without reaching past standing tiles "
                "(or with the placing skill's lead-in inside the arm's box)",
            )
        at_risk = run_of(g[0], -1, standing) + run_of(g[-1], 1, standing) + len(g)
        gap = gate_gap(g[0], len(g))
        margins.append(gap - crit)
        first_step = len(steps)
        for s, a in zip(g, arms_ok, strict=True):
            route, why = place(s, int(a), True)
            if route:
                return finish(route, why)
            steps[-1]["between_standing"] = s == g[-1]
            steps[-1]["at_risk"] = at_risk
        gate_out.append(
            {
                "slots": g,
                "gap": float(gap),
                "margin": float(gap - crit),
                "close_step": first_step,
                "at_risk": at_risk,
            }
        )
        open_groups.remove(g)
    out["gates"] = gate_out
    out["gate_margin"] = float(min(margins)) if margins else 0.0
    out["build_steps"] = steps
    out["build_order"] = [s["slot_index"] for s in steps]
    if holes:
        problems.append(f"{holes} line slot(s) go between two standing tiles (where the runs meet)")

    gate_note = (
        f"{len(groups)} gate(s) of {out['gate_width_slots']} slot(s), "
        f"{out['gate_margin'] * 1e3:.1f} mm past the critical gap"
        if groups
        else "no gates"
    )
    return finish(
        "planned",
        f"{n} tiles along a {kind} at {p * 1e3:.1f} mm ({source}); face gap up to "
        f"{out['max_face_gap'] * 1e3:.1f} mm, critical {crit * 1e3:.1f} mm "
        f"(margin {out['topple_margin'] * 1e3:.1f} mm); {gate_note}"
        + (f"; {'; '.join(problems)}" if problems else ""),
    )
