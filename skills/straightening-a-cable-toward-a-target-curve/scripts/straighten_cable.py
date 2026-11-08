# Adapted from deformable-ravens (https://github.com/DanielTakeshi/deformable-ravens),
# task.py of the ravens.tasks package at 73982748bdfe756e04553e28698948facccecb41: Task.oracle's
# cable-ring branch (try every ordering of the beads against the targets, keep the
# one with the least mean distance, pick the bead with the largest distance and
# move it to its target), its cable-shape branch (pick_place: the bead furthest
# from its target first) and Task.params_no_rots (place a little past the target,
# along the pull), Apache-2.0, Copyright (c) Daniel Seita and the Ravens authors.
# Changes: the cable is a perceived centreline (sam3.segment_text and
# curve.fit_centerline, or a given one) instead of the simulator's bead poses,
# resampled at equal arc length in plan and laid on the table along the camera's
# rays (its depth only feeds the fit); the target is a world-frame polyline
# resampled the same way; the orderings are the open cable's two directions and,
# when cable and target differ in length, every placement of the shorter along
# the longer; the pick is grasping-a-cable-lying-flat (top-down, fingertips to
# the table, test lift) kept clear of the cable's ends, and an unreachable pick
# falls back to the next-largest discrepancy; the place is a carry at the lifted
# height, a set-down on the table and a release; the loop stops on a measured
# max error under a tolerance (upstream: every bead within 3.5 radii of a
# target, or 21 actions), a cap on corrections, a stall, or a lost cable, and
# routes those as exits.
"""Pull a cable lying on a table toward a target curve, one largest-discrepancy pick-and-place at a time."""

from __future__ import annotations

import importlib.util
import logging
import math
import os
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

TOLERANCE = 0.010
"""Largest plan distance from a matched target node that counts as on the curve
[m]. Upstream counts a bead on the target within 3.5 of its radii (17.5 mm for
its 5 mm beads, "2 * radius is too strict"); 3.5 radii of a 3.5 mm cable is
6 mm, under a real overhead fit's end error (7-9 mm, issue #10), so 10 mm."""

MAX_ITERS = 8
"""Most pick-and-place corrections [-]. Upstream's cable-shape episodes stop at
21 actions for 24 beads; a YAM pick-and-place takes about 20 s."""

SPACING = 0.010
"""Node spacing of both resampled curves [m]: about upstream's (24 beads 14 mm
apart) and the real station's cable trace (1 cm)."""

OVERSHOOT = 0.010
"""How far past its target the picked point is put down, along the pull [m].
Upstream: 0 for the cable ring, 0.02 for bag rings, 0.03 for cloth ("otherwise
the physics will favor the cloth resetting"); a flat cable held by table
friction springs back less, so half the bag-ring value."""

OVERSHOOT_SHARE = 0.5
"""Largest overshoot as a share of the move [-]. With the full 10 mm past the
target, a node moved a little that does not spring back ends 10 mm off on the
other side, at the tolerance, instead of nearer than it started."""

ALONG_SLACK = 0.03
"""How far along the target a node may lie from its matched place and still be
measured across to the target [m]. Upstream's oracle matches bead to target by
order only, and its success metric counts a bead near any target. Matched by
order only, a straightened cable lying on the line but 2.3 mm-per-cm crept
along it read 13.6 mm off at one node in a YAM station twin, and pulling that
node along the cable buckled it to 30.6 mm (true distance to the line 12.8 ->
30.6 mm); three nodes' worth lets such a node count as on the line."""

END_MARGIN = 0.015
"""Nearest a pick comes to the cable's ends [m]: the skeleton's ends erode by
about a radius and a real fit's ends are 7-9 mm off (issue #10), so the
fingertips stay on the jacket."""

MIN_COVERAGE = 0.8
"""Shortest perceived cable, as a share of the target's length, that is still
taken to be the whole cable [-]: shorter, part of it is hidden or out of view."""

ALONG_CONE_DEG = 30.0
"""A move within this angle of the cable's own way at the pick runs along the
cable [deg]: pushed that way a cable buckles instead of sliding."""

BUCKLE = "the move runs along the cable away from its ends (it would buckle)"

END_ZONE = 0.02
"""Picks within END_MARGIN plus this of an end count as at that end [m]: moves
along the cable there slide it, and moves across get no overshoot (in a YAM
station twin an end 25 mm off, pulled 15 mm in from it with a 10 mm overshoot,
swung to 17 mm off on the other side)."""

PICK_TRIES = 3
"""Pick points tried, in falling discrepancy, when the grasp routes unreachable."""

PICK_GAP = 0.03
"""Least distance along the cable between two tried pick points [m]."""

MAX_MISSES = 2
"""Grasps in a row that close on nothing before the skill gives up [-]."""

STALL_ITERS = 3
"""Corrections without a better max error (by STALL_GAIN) before the skill stops."""

STALL_GAIN = 0.002
"""Improvement of the best max error that counts as progress [m]."""

FIT_SMOOTH = 1.0e-5
"""``smooth`` passed to curve.fit_centerline. Its default (2.0) is applied in
metres and over-smooths a long real cable to one cubic (ends 54-58 mm off on a
real overhead frame); 1e-5 gave 7-9 mm ends there, with a 3D length inflated by
depth jitter (issue #10), which this skill drops by working in plan."""

FIT_NODES = 60
"""Nodes asked of curve.fit_centerline: about 1.5 cm apart on a 0.9 m cable,
resampled to SPACING here."""

PLACE_OVER = 0.003
"""Tool centre over the grasp's contact height when the cable is set down [m]:
the held cable reaches the table, the fingertips do not press into it."""

RETREAT = 0.05
"""Rise after the release [m], before the next look."""

TANGENT_SPAN = 0.010
"""Half the stretch of centreline the cable's way at the pick is taken over [m]."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from the carry [m]: no tighter than a real YAM
station's own 12 mm per-move bound."""

CABLE_RADIUS = 0.00175
"""Radius when neither the input nor the fit states one [m]: a 3.5 mm cable."""

GRASP_SKILL = "grasping-a-cable-lying-flat"
GRASP_SCRIPT = Path("scripts") / "grasp_flat_cable.py"


class Output(TypedDict):
    route: str
    detail: str
    max_error: float
    mean_error: float
    initial_max_error: float
    errors: list[float]
    corrections: int
    misses: int
    reversed: bool
    offset_m: float
    coverage: float
    pick_point: dict[str, float]
    place_point: dict[str, float]
    centerline: dict[str, Any]


# --- geometry -------------------------------------------------------------------


def curve_points(curve: Any) -> np.ndarray:
    """``[[x, y, z], ...]`` from a Centerline, a dict with ``points``, or a list."""
    pts = curve.get("points") if isinstance(curve, dict) else curve
    if pts is None:
        return np.zeros((0, 3))
    rows = []
    for p in pts:
        if isinstance(p, dict):
            rows.append([float(p["x"]), float(p["y"]), float(p.get("z", 0.0))])
        else:
            q = [float(v) for v in p]
            rows.append(q[:3] if len(q) >= 3 else [q[0], q[1], 0.0])
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def plan_arcs(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length in plan (x, y) along a polyline [m]."""
    if len(points) < 2:
        return np.zeros(len(points))
    seg = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def at_arcs(points: np.ndarray, arcs: np.ndarray, s: Any) -> np.ndarray:
    """Points at plan arc lengths ``s`` along the polyline (clamped to its ends)."""
    s = np.clip(np.atleast_1d(np.asarray(s, dtype=float)), 0.0, arcs[-1])
    return np.stack([np.interp(s, arcs, points[:, i]) for i in range(3)], axis=1)


def resample(points: np.ndarray, n: int) -> np.ndarray:
    """``n`` points at equal plan arc length along the polyline."""
    arcs = plan_arcs(points)
    return at_arcs(points, arcs, np.linspace(0.0, arcs[-1], int(n)))


def _matrix(q: dict[str, float]) -> np.ndarray:
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


def camera_matrix(pose: Any) -> np.ndarray | None:
    """Camera-to-world 4x4 from an Se3Pose or a 4x4; None when unreadable."""
    try:
        if isinstance(pose, dict):
            p = pose["position"]
            T = np.eye(4)
            T[:3, :3] = _matrix(pose["rotation"])
            T[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
            return T
        return np.asarray(pose, dtype=float).reshape(4, 4)
    except Exception:  # noqa: BLE001 - no pose, no correction
        return None


def onto_table(points: np.ndarray, camera_to_world: np.ndarray, plane_z: float) -> np.ndarray:
    """Each point slid along its camera ray to the height ``plane_z``. A cable
    lying flat has its axis there; the depth of a 3.5 mm cable is noisier than
    its thickness, and a point read high is also off in plan along the ray."""
    c = np.asarray(camera_to_world, dtype=float)[:3, 3]
    out = np.array(points, dtype=float)
    dz = out[:, 2] - c[2]
    ok = np.abs(dz) > 1.0e-6
    t = np.ones(len(out))
    t[ok] = (float(plane_z) - c[2]) / dz[ok]
    ok &= t > 0.0
    out[ok] = c + (out[ok] - c) * t[ok, None]
    out[:, 2] = float(plane_z)
    return out


def match_curves(
    current: Any,
    target: Any,
    spacing: float = SPACING,
    mode: str = "slide",
    along_slack: float = ALONG_SLACK,
) -> dict[str, Any]:
    """Upstream's ordering search on an open cable: the shorter of cable and
    target is laid along the longer at every offset (``mode="slide"``, steps of
    half the spacing) or with an end on an end (``"ends"``), in both directions;
    the placement with the least mean plan distance wins (upstream's
    ``average_distance``), and in it the node with the largest distance is the
    one to move (``max_idx``). With ``along_slack`` each node's distance is to
    the nearest target point within that arc of its matched place."""
    cur = curve_points(current)
    tgt = curve_points(target)
    if len(cur) < 2 or len(tgt) < 2:
        raise ValueError("cable and target need at least two points each")
    ca, ta = plan_arcs(cur), plan_arcs(tgt)
    lc, lt = float(ca[-1]), float(ta[-1])
    if lc <= 0.0 or lt <= 0.0:
        raise ValueError("cable and target need a length in plan")
    cable_longer = lc >= lt
    short = min(lc, lt)
    n = max(2, int(round(short / float(spacing))) + 1)
    u = np.linspace(0.0, short, n)  # arc along the shorter curve
    slack = abs(lc - lt)
    if mode == "ends" or slack < 1e-9:
        offsets = np.array([0.0, slack]) if slack > 1e-9 else np.array([0.0])
    else:
        offsets = np.unique(np.append(np.arange(0.0, slack, float(spacing) / 2.0), slack))
    best: dict[str, Any] | None = None
    for rev in (False, True):
        for o in offsets:
            along = o + (short - u if rev else u)  # arc on the longer curve, per node
            cs, ts = (along, u) if cable_longer else (u, along)
            C = at_arcs(cur, ca, cs)
            T = at_arcs(tgt, ta, ts)
            d = np.linalg.norm(T[:, :2] - C[:, :2], axis=1)
            mean = float(d.mean())
            if best is None or mean < best["mean_error"] - 1e-12:
                best = {
                    "errors": d,
                    "mean_error": mean,
                    "max_error": float(d.max()),
                    "worst": int(np.argmax(d)),
                    "cable_s": cs,
                    "target_s": ts,
                    "cable_nodes": C,
                    "target_nodes": T,
                    "reversed": bool(rev),
                    "offset_m": float(o),
                }
    assert best is not None
    best.update(
        cable_length=lc,
        target_length=lt,
        coverage=float(min(1.0, lc / lt)),
        cable_points=cur,
        cable_arcs=ca,
        target_points=tgt,
        target_arcs=ta,
    )
    if along_slack > 0.0:
        # Each node's target is the nearest target point within along_slack of
        # its matched place: a node lying on the curve a little along from its
        # place is on it, and the pull is across the curve, not along the cable.
        ts = np.array(best["target_s"], dtype=float)
        T = np.array(best["target_nodes"], dtype=float)
        for i, p in enumerate(best["cable_nodes"]):
            ts[i], T[i] = nearest_on(tgt, ta, p, ts[i], along_slack, spacing)
        d = np.linalg.norm(T[:, :2] - np.asarray(best["cable_nodes"])[:, :2], axis=1)
        best.update(
            errors=d,
            mean_error=float(d.mean()),
            max_error=float(d.max()),
            worst=int(np.argmax(d)),
            target_s=ts,
            target_nodes=T,
        )
    best["along_slack"] = float(along_slack)
    best["spacing"] = float(spacing)
    return best


def nearest_on(
    points: np.ndarray, arcs: np.ndarray, p: np.ndarray, s: float, slack: float, spacing: float
) -> tuple[float, np.ndarray]:
    """The point of the polyline nearest ``p`` in plan among arc lengths within
    ``slack`` of ``s`` (sampled every tenth of the spacing): (arc, point)."""
    k = max(3, int(math.ceil(2.0 * slack / (float(spacing) / 10.0))) + 1)
    cand = np.clip(np.linspace(s - slack, s + slack, k), 0.0, arcs[-1])
    pts = at_arcs(points, arcs, cand)
    j = int(np.argmin(np.linalg.norm(pts[:, :2] - p[:2], axis=1)))
    return float(cand[j]), pts[j]


def pick_candidates(
    m: dict[str, Any], tolerance: float, end_margin: float, tries: int, gap: float
) -> list[int]:
    """Matched nodes to move, largest discrepancy first, over the tolerance and
    at least ``gap`` apart along the cable."""
    order = np.argsort(-np.asarray(m["errors"]))
    chosen: list[int] = []
    lc = float(m["cable_length"])
    for i in order:
        if m["errors"][i] < tolerance or len(chosen) >= tries:
            break
        s = _clamp_arc(float(m["cable_s"][i]), lc, end_margin)
        if all(abs(s - _clamp_arc(float(m["cable_s"][j]), lc, end_margin)) >= gap for j in chosen):
            chosen.append(int(i))
    return chosen


def _clamp_arc(s: float, length: float, margin: float) -> float:
    if length <= 2.0 * margin:
        return 0.5 * length
    return float(np.clip(s, margin, length - margin))


def pull_not_push(
    points: np.ndarray,
    arcs: np.ndarray,
    s: float,
    pick: np.ndarray,
    goal: np.ndarray,
    end_margin: float,
) -> tuple[float, np.ndarray, np.ndarray, bool, str]:
    """A cable can be pulled along itself, not pushed. A move within
    ALONG_CONE_DEG of the cable's own way is kept only at an end and outward
    from it (it slides the cable); one into the cable at an end is made by
    pulling the other end the same way instead; one along the cable away from
    its ends is refused (it would buckle one side). Returns (arc, pick, goal,
    slides, why-refused)."""
    lc = float(arcs[-1])
    d = np.array([goal[0] - pick[0], goal[1] - pick[1], 0.0])
    if float(np.linalg.norm(d)) < 1e-9:
        return s, pick, goal, False, ""
    way = at_arcs(points, arcs, s + TANGENT_SPAN)[0] - at_arcs(points, arcs, s - TANGENT_SPAN)[0]
    way[2] = 0.0
    if float(np.linalg.norm(way)) < 1e-9:
        return s, pick, goal, False, ""
    way /= float(np.linalg.norm(way))
    along = float(d @ way) / float(np.linalg.norm(d))
    if abs(along) < math.cos(math.radians(ALONG_CONE_DEG)):
        return s, pick, goal, False, ""
    near = end_margin + END_ZONE
    outward = -1.0 if s <= near else 1.0 if s >= lc - near else 0.0
    if outward == 0.0:
        return s, pick, goal, False, BUCKLE
    if along * outward > 0.0:
        return s, pick, goal, True, ""
    other = _clamp_arc(lc - s, lc, end_margin)
    p2 = at_arcs(points, arcs, other)[0]
    g2 = p2 + d
    g2[2] = p2[2]
    return other, p2, g2, True, ""


def place_past(pick: np.ndarray, goal: np.ndarray, overshoot: float) -> np.ndarray:
    """Upstream's ``params_no_rots``: the place is the target pushed
    ``overshoot`` further along the pull from the pick (plan; the pick's height),
    but never more than OVERSHOOT_SHARE of the move itself."""
    d = np.array([goal[0] - pick[0], goal[1] - pick[1]])
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return np.array([goal[0], goal[1], pick[2]])
    past = min(float(overshoot), OVERSHOOT_SHARE * length)
    xy = pick[:2] + d / length * (length + past)
    return np.array([xy[0], xy[1], pick[2]])


# --- tools ----------------------------------------------------------------------


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or "" (connectors that route a
    refusal instead of raising)."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


_GRASP: Any = None


def grasp_module() -> Any:
    """grasping-a-cable-lying-flat's script, from the skill installed beside this
    one (or under a GAP_SKILLS_PATH root)."""
    global _GRASP
    if _GRASP is not None:
        return _GRASP
    roots = [Path(__file__).resolve().parents[2]]
    roots += [Path(p) for p in os.environ.get("GAP_SKILLS_PATH", "").split(os.pathsep) if p]
    for root in roots:
        for path in (
            root / GRASP_SKILL / GRASP_SCRIPT,
            root / "skills" / GRASP_SKILL / GRASP_SCRIPT,
        ):
            if path.is_file():
                spec = importlib.util.spec_from_file_location("_straighten_grasp_flat_cable", path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _GRASP = module
                return module
    raise RuntimeError(
        f"{GRASP_SKILL} is not installed beside this skill (looked under "
        f"{', '.join(str(r) for r in roots)})"
    )


def perceive(
    ctx: NodeContext,
    description: str,
    camera: str,
    plane_z: float | None,
    smooth: float,
    nodes: int,
) -> tuple[np.ndarray | None, float | None, str]:
    """The cable's centreline from one camera: sam3.segment_text, then
    curve.fit_centerline, then laid on the table along the camera's rays.
    Returns (points, radius, why-not)."""
    obs = ctx.tool("robot.get_observation")
    cams = obs.get("cameras") if isinstance(obs, dict) else getattr(obs, "cameras", None)
    cams = list(cams.values()) if isinstance(cams, dict) else list(cams or [])
    cams = [c for c in cams if c.get("depth") is not None and c.get("rgb") is not None]
    named = [c for c in cams if camera and camera in str(c.get("name", ""))]
    cam = (named or cams or [None])[0]
    if cam is None:
        return None, None, "no camera frame with depth"
    found = ctx.tool("sam3.segment_text", image=cam["rgb"], query=description, max_results=1)
    masks = found.get("masks") if isinstance(found, dict) else None
    if masks is None or len(masks) == 0:
        return None, None, f"{description!r} not found in the {cam.get('name', 'camera')} frame"
    fit = ctx.tool(
        "curve.fit_centerline",
        mask=np.asarray(masks[0], dtype=np.uint8),
        depth=cam["depth"],
        intrinsics=cam["intrinsics"],
        camera_pose=cam["pose"],
        nodes=int(nodes),
        smooth=float(smooth),
    )
    pts = curve_points(fit)
    if len(pts) < 2:
        return None, None, "the centreline fit came back empty"
    radius = fit.get("radius_m") if isinstance(fit, dict) else None
    T = camera_matrix(cam.get("pose"))
    if plane_z is not None and T is not None:
        r = float(radius) if radius else CABLE_RADIUS
        pts = onto_table(pts, T, float(plane_z) + r)
    return pts, (float(radius) if radius else None), ""


# --- the skill ------------------------------------------------------------------


def run(
    ctx: NodeContext,
    target: Any,
    centerline: Any = None,
    cable_description: str = "",
    camera: str = "overhead",
    table_z: float | None = None,
    cable_radius: float | None = None,
    tolerance: float = TOLERANCE,
    max_iters: int = MAX_ITERS,
    spacing: float = SPACING,
    match: str = "slide",
    along_slack: float = ALONG_SLACK,
    overshoot: float = OVERSHOOT,
    end_margin: float = END_MARGIN,
    min_coverage: float = MIN_COVERAGE,
    pick_tries: int = PICK_TRIES,
    max_misses: int = MAX_MISSES,
    fit_smooth: float = FIT_SMOOTH,
    fit_nodes: int = FIT_NODES,
    lift_check: bool = True,
    grasp_camera: str = "wrist",
    place_over: float = PLACE_OVER,
    retreat: float = RETREAT,
    clear_pose: dict[str, Any] | None = None,
    press: float | None = None,
    pre_close_width: float | None = None,
    min_tool_height: float | None = None,
    pinch_below: float | None = None,
    overtravel: float | None = None,
    check_reach: bool = True,
    reach_tolerance: float = REACH_TOLERANCE,
    arm_id: int = 0,
) -> Output:
    if match not in ("slide", "ends"):
        raise ValueError(f"match must be 'slide' or 'ends', not {match!r}")
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    out: Output = {
        "route": "lost_cable",
        "detail": "",
        "max_error": -1.0,
        "mean_error": -1.0,
        "initial_max_error": -1.0,
        "errors": [],
        "corrections": 0,
        "misses": 0,
        "reversed": False,
        "offset_m": 0.0,
        "coverage": 0.0,
        "pick_point": {"x": 0.0, "y": 0.0, "z": 0.0},
        "place_point": {"x": 0.0, "y": 0.0, "z": 0.0},
        "centerline": {},
    }
    tgt = curve_points(target)
    if len(tgt) < 2 or plan_arcs(tgt)[-1] <= 0.0:
        raise ValueError("target needs at least two points apart in plan")

    if table_z is None:
        try:
            table_z = float(ctx.tool("robot.describe_workspace", **on_arm)["surface_z"])
        except Exception as exc:  # noqa: BLE001 - fall back to the cable's own height
            logger.info("straightening: no workspace surface (%s)", exc)
    camera_on = bool(cable_description)
    radius = float(cable_radius) if cable_radius else None

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        return out

    # The first look: the given centreline, else the camera.
    if centerline is not None:
        cur = curve_points(centerline)
        if radius is None and isinstance(centerline, dict) and centerline.get("radius_m"):
            radius = float(centerline["radius_m"])
        if len(cur) < 2:
            return finish("lost_cable", "the given centreline has fewer than two points")
    elif camera_on:
        cur, fit_r, why = perceive(ctx, cable_description, camera, table_z, fit_smooth, fit_nodes)
        if cur is None:
            return finish("lost_cable", why)
        radius = radius or fit_r
    else:
        raise ValueError("give a centerline, or a cable_description to perceive it by camera")
    radius = radius or CABLE_RADIUS
    if table_z is None:
        table_z = float(np.median(cur[:, 2])) - radius
    table_z = float(table_z)
    cur[:, 2] = table_z + radius

    grasp = grasp_module()
    knobs = {
        k: v
        for k, v in {
            "press": press,
            "pre_close_width": pre_close_width,
            "min_tool_height": min_tool_height,
            "pinch_below": pinch_below,
            "overtravel": overtravel,
        }.items()
        if v is not None
    }
    floor = table_z + float(
        min_tool_height if min_tool_height is not None else grasp.MIN_TOOL_HEIGHT
    )

    def here() -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **on_arm))

    def move(position: np.ndarray, rot: dict[str, float]) -> str:
        try:
            reply = ctx.tool(
                "robot.go_to_pose_cartesian",
                pose={"position": _vec(position), "rotation": rot},
                **on_arm,
            )
            return _refusal(reply)
        except Exception as exc:  # noqa: BLE001 - a refused move is routed, not raised
            return str(exc) or type(exc).__name__

    def set_down(at: np.ndarray, rot: dict[str, float], low_z: float) -> str:
        """Lower the held cable to the table where the hand is, release, rise."""
        notes = []
        why = move(np.array([at[0], at[1], low_z]), rot)
        if why:
            notes.append(f"lowering: {why}")
        ctx.tool("robot.open_gripper", **on_arm)
        up = _xyz(here())
        why = move(np.array([up[0], up[1], max(up[2], low_z) + float(retreat)]), rot)
        if why:
            notes.append(f"rise: {why}")
        return "; ".join(notes)

    def summary() -> str:
        trail = " -> ".join(f"{e * 1e3:.0f}" for e in out["errors"])
        return f"max error {trail} mm over {out['corrections']} correction(s)"

    best = math.inf
    since_best = 0
    misses_in_row = 0
    while True:
        m = match_curves(cur, tgt, spacing, match, along_slack)
        out.update(
            max_error=m["max_error"],
            mean_error=m["mean_error"],
            reversed=m["reversed"],
            offset_m=m["offset_m"],
            coverage=m["coverage"],
            centerline={
                "points": cur.tolist(),
                "arclength_m": m["cable_length"],
                "radius_m": radius,
                "nodes": len(cur),
                "ordered": True,
            },
        )
        out["errors"].append(float(m["max_error"]))
        if out["initial_max_error"] < 0.0:
            out["initial_max_error"] = float(m["max_error"])
        if m["coverage"] < float(min_coverage):
            return finish(
                "lost_cable",
                f"the cable seen is {m['cable_length'] * 1e3:.0f} mm in plan, "
                f"{m['coverage']:.0%} of the {m['target_length'] * 1e3:.0f} mm target: part of it is hidden "
                f"or out of view; {summary()}",
            )
        if m["max_error"] < float(tolerance):
            return finish(
                "shaped",
                f"every node within {m['max_error'] * 1e3:.1f} mm of the target "
                f"(tolerance {float(tolerance) * 1e3:.0f} mm, mean {m['mean_error'] * 1e3:.1f} mm); {summary()}",
            )
        if m["max_error"] < best - STALL_GAIN:
            best, since_best = float(m["max_error"]), 0
        elif out["corrections"] > 0:
            since_best += 1
        if out["corrections"] >= int(max_iters) or since_best >= STALL_ITERS:
            why = "no better for" if since_best >= STALL_ITERS else "after"
            n = since_best if since_best >= STALL_ITERS else out["corrections"]
            return finish(
                "not_converged",
                f"max error still {m['max_error'] * 1e3:.1f} mm {why} {n} correction(s) "
                f"(tolerance {float(tolerance) * 1e3:.0f} mm); {summary()}",
            )

        # The pick: largest discrepancy first, the next ones if it is out of reach.
        candidates = pick_candidates(
            m, float(tolerance), float(end_margin), int(pick_tries), PICK_GAP
        )
        arcs = m["cable_arcs"]
        lc = float(m["cable_length"])
        g: dict[str, Any] | None = None
        refused: list[str] = []
        buckles: list[str] = []
        pick = goal = place = np.zeros(3)
        for i in candidates:
            s = _clamp_arc(float(m["cable_s"][i]), lc, float(end_margin))
            pick = at_arcs(cur, arcs, s)[0]
            # The node's own correction, applied where the fingers can hold
            # (the same point unless it was too near an end).
            goal = pick + (m["target_nodes"][i] - m["cable_nodes"][i])
            goal[2] = pick[2]
            s, pick, goal, slides, why = pull_not_push(cur, arcs, s, pick, goal, float(end_margin))
            if why:
                buckles.append(f"{s * 1e3:.0f} mm along: {why}")
                continue
            way = at_arcs(cur, arcs, s + TANGENT_SPAN)[0] - at_arcs(cur, arcs, s - TANGENT_SPAN)[0]
            # Upstream's overshoot is against the cloth springing back across
            # the pull; a slide along the cable, or a pick near an end (whose
            # free end swings with the hand, further than it), is put exactly.
            at_end = min(s, lc - s) <= float(end_margin) + END_ZONE
            place = goal.copy() if slides or at_end else place_past(pick, goal, overshoot)
            reply = grasp.run(
                ctx,
                grasp_point=_vec(pick),
                tangent=_vec(way),
                table_z=table_z,
                cable_radius=radius,
                cable_description=cable_description if lift_check else "",
                camera=grasp_camera,
                check_reach=check_reach,
                arm_id=arm_id,
                **knobs,
            )
            if reply["route"] == "unreachable":
                refused.append(f"{s * 1e3:.0f} mm along: {reply['detail']}")
                continue
            g = reply
            break
        out["pick_point"], out["place_point"] = _vec(pick), _vec(place)
        if g is None and buckles and not refused:
            return finish(
                "not_converged",
                f"every node off the target ({m['max_error'] * 1e3:.1f} mm) would have to move along "
                f"the cable ({'; '.join(buckles)}); {summary()}",
            )
        if g is None:
            refused += buckles
            return finish(
                "unreachable",
                f"no pick reached ({'; '.join(refused) or 'no node over the tolerance'}); {summary()}",
            )
        if g["route"] != "grasped":
            out["misses"] += 1
            out["corrections"] += 1
            misses_in_row += 1
            if misses_in_row >= int(max_misses) or not camera_on:
                return finish(
                    "grasp_failed",
                    f"{misses_in_row} grasp(s) in a row closed on nothing (last: {g['detail']}); {summary()}",
                )
        else:
            misses_in_row = 0
            # The place: carry at the lifted height, set down, release, rise.
            now = here()
            rot = now["rotation"]
            lifted = _xyz(now)
            contact = float(g.get("contact_z") or g.get("planned_tool_z") or lifted[2])
            low_z = max(contact + float(place_over), floor)
            carry = np.array([place[0], place[1], lifted[2]])
            why = move(carry, rot)
            arrived = _xyz(here())
            off = float(np.linalg.norm(arrived[:2] - carry[:2]))
            if (why and not why.startswith("timeout")) or off > float(reach_tolerance):
                note = set_down(arrived, rot, low_z)
                out["corrections"] += 1
                return finish(
                    "unreachable",
                    f"the carry to the place was refused or ended {off * 1e3:.0f} mm off ({why or 'off'}); "
                    f"the cable was set down where the hand stood{'; ' + note if note else ''}; {summary()}",
                )
            note = set_down(arrived, rot, low_z)
            if note:
                logger.info("straightening: %s", note)
            out["corrections"] += 1
        if clear_pose is not None:
            try:
                why = _refusal(ctx.tool("robot.go_to_pose", pose=clear_pose, **on_arm))
            except Exception as exc:  # noqa: BLE001
                why = str(exc) or type(exc).__name__
            if why:
                logger.info("straightening: clear pose not reached (%s)", why)

        # Look again.
        if not camera_on:
            return finish(
                "corrected",
                f"moved the point {math.dist(pick[:2], goal[:2]) * 1e3:.0f} mm off its target "
                f"to {math.dist(goal[:2], place[:2]) * 1e3:.0f} mm past it; not seen since (no cable_description): "
                f"fit the centreline again and call again; {summary()}",
            )
        seen, fit_r, why = perceive(ctx, cable_description, camera, table_z, fit_smooth, fit_nodes)
        if seen is None:
            return finish(
                "lost_cable", f"after correction {out['corrections']}: {why}; {summary()}"
            )
        cur = seen
        cur[:, 2] = table_z + radius
