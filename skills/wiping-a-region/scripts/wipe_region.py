# Adapted from CaP-X (https://github.com/capgym/cap-x), its Franka spill-wipe
# oracle (franka_spill_wipe.py, ORACLE_CODE), MIT, Copyright (c) 2026 Max Fu.
# Changes: a world-frame region box in place of get_object_pose's SAM3 extent;
# strokes along the box's long axis (CaP-X: along world x) with the tool kept
# inside the box (CaP-X's np.arange overshoots its far edge by up to a step);
# the tool centre held at a floored contact height over the box's bottom rather
# than at z = 0; each pass's descent made with robot.move_cartesian_until_contact
# so a surface higher than estimated is met, not pressed into; an IK preview per
# pass; routed move refusals; coverage measured from the hand's read-back path.
"""Wipe a flat region of a surface with back-and-forth strokes at a contact height, no force sensor."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

STEP = 0.05
"""Longest move along a stroke and widest spacing between strokes [m]: CaP-X's
``step`` (its oracle keeps IK deltas small with 5 cm waypoints)."""

APPROACH_HEIGHT = 0.10
"""Height over the wiping height the hand travels at between passes [m]: CaP-X's
``z_approach``."""

TOOL_WIDTH = 0.05
"""Width of the swath one stroke wipes [m]. CaP-X's 5 cm stroke spacing assumes a
sponge at least that wide; pass the held tool's width (a closed linear_4310's
fingers across the stroke cover about 30 mm)."""

CONTACT_HEIGHT = 0.015
"""Tool centre over the surface while wiping [m]. CaP-X wipes at the surface
(z = 0) with a sponge below its tool centre; for a held tool set this to how far
its face reaches below the tool centre, minus any press-in."""

MIN_CONTACT_HEIGHT = 0.015
"""Lowest tool-centre height over the surface [m], whatever is asked. A real YAM
station refuses a tool centre under 14 mm over its calibrated table; the closed
fingertips then clear it by about 5 mm."""

HEIGHT_TOLERANCE = 0.006
"""How far over the wiping height a descent may stop and still count as having
arrived [m]: above a servo's 4 mm ball and a real arm's 2-3 mm sag. A stroke
point higher than this over its pass's height is not counted as wiping."""

MAX_RAISE = 0.02
"""A descent that stops higher than this over the wiping height met an object or
stalled [m]; the pass is skipped rather than wiped in the air."""

REACH_TOLERANCE = 0.02
"""How far the hand may end from a hover or a stroke waypoint [m]: a real YAM
station's 12 mm per-move bound plus margin."""

MIN_COVERAGE = 0.95
"""Fraction of the region the measured strokes must sweep to count as wiped."""

GRID = 0.002
"""Cell size of the coverage raster [m]."""


class Output(TypedDict):
    route: str
    detail: str
    coverage: float
    passes_planned: int
    passes_done: int
    surface_z: float
    wipe_height: float
    min_height: float
    max_height: float
    path: list[dict[str, Any]]


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


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


def _rotation_matrix(q: dict[str, float] | None) -> np.ndarray:
    q = q or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def region_frame(
    obb: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """The region in the table plane: centre (xy), long axis u, short axis v (unit,
    xy), the half-lengths along them, and the box's lowest z (its bottom, the
    surface under a flat region). The box axis nearest vertical is its normal."""
    rot = _rotation_matrix(obb.get("orientation"))
    half = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    centre = np.array([float(obb["center"][k]) for k in ("x", "y", "z")])
    normal = int(np.argmax(np.abs(rot[2, :])))
    planar = [i for i in range(3) if i != normal]
    axes, halves = [], []
    for i in planar:
        a = rot[:2, i]
        n = float(np.linalg.norm(a))
        # A planar axis is horizontal up to the box's tilt; its xy length keeps the
        # half-length's footprint.
        axes.append(a / n if n > 1e-9 else np.array([1.0, 0.0]))
        halves.append(half[i] * n)
    if halves[1] > halves[0]:
        axes, halves = axes[::-1], halves[::-1]
    u = axes[0]
    v = np.array([-u[1], u[0]])
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    bottom = float(centre[2] + (corners @ rot.T)[:, 2].min())
    return centre[:2], u, v, float(halves[0]), float(halves[1]), bottom


def plan_passes(
    half_u: float, half_v: float, tool_width: float, step: float
) -> list[tuple[float, list[float]]]:
    """CaP-X's boustrophedon in the region's frame: passes along u at offsets
    across v no more than min(step, tool_width) apart, alternating direction,
    each a list of waypoints along u no more than ``step`` apart. The tool's
    centre stays half its width inside every edge, so its swath ends at the
    region's edge; a region narrower than the tool gets one pass down its middle."""
    reach_u = max(0.0, half_u - tool_width / 2.0)
    reach_v = max(0.0, half_v - tool_width / 2.0)
    pitch = min(step, tool_width)
    rows = 1 if reach_v <= 1e-9 else int(math.ceil(2.0 * reach_v / pitch - 1e-9)) + 1
    offsets = [0.0] if rows == 1 else list(np.linspace(-reach_v, reach_v, rows))
    segments = max(1, int(math.ceil(2.0 * reach_u / step - 1e-9)))
    along = list(np.linspace(-reach_u, reach_u, segments + 1))
    return [
        (float(off), [float(a) for a in (along if i % 2 == 0 else along[::-1])])
        for i, off in enumerate(offsets)
    ]


def coverage(
    strokes: list[np.ndarray],
    centre: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    half_u: float,
    half_v: float,
    tool_width: float,
    grid: float = GRID,
) -> float:
    """Fraction of the region (a raster of ``grid`` cells) inside the swath of the
    strokes: each segment between consecutive measured points sweeps a square
    tool of side ``tool_width`` along it, so a rectangle ``tool_width`` wide that
    reaches half the width past each end."""
    nu = max(1, int(round(2.0 * half_u / grid)))
    nv = max(1, int(round(2.0 * half_v / grid)))
    gu = (np.arange(nu) + 0.5) / nu * 2.0 * half_u - half_u
    gv = (np.arange(nv) + 0.5) / nv * 2.0 * half_v - half_v
    cu, cv = np.meshgrid(gu, gv, indexing="ij")
    covered = np.zeros_like(cu, dtype=bool)
    half_w = tool_width / 2.0
    for stroke in strokes:
        local = np.stack([(stroke[:, :2] - centre) @ u, (stroke[:, :2] - centre) @ v], axis=1)
        for a, b in zip(local[:-1], local[1:], strict=False):
            d = b - a
            length = float(np.linalg.norm(d))
            t = d / length if length > 1e-9 else np.array([1.0, 0.0])
            n = np.array([-t[1], t[0]])
            pu, pv = cu - a[0], cv - a[1]
            along = pu * t[0] + pv * t[1]
            across = pu * n[0] + pv * n[1]
            covered |= (along >= -half_w) & (along <= length + half_w) & (np.abs(across) <= half_w)
        if len(local) == 1:
            p = local[0]
            covered |= (np.abs(cu - p[0]) <= half_w) & (np.abs(cv - p[1]) <= half_w)
    return float(covered.mean())


def run(
    ctx: NodeContext,
    region_obb: dict[str, Any],
    surface_z: float | None = None,
    tool_width: float = TOOL_WIDTH,
    step: float = STEP,
    contact_height: float = CONTACT_HEIGHT,
    min_contact_height: float = MIN_CONTACT_HEIGHT,
    approach_height: float = APPROACH_HEIGHT,
    min_coverage: float = MIN_COVERAGE,
    close_hand: bool = True,
    probe_surface: bool = True,
    check_reach: bool = True,
    keep_rotation: bool = False,
) -> Output:
    centre, u, v, half_u, half_v, bottom = region_frame(region_obb)
    surface = float(surface_z) if surface_z is not None else bottom
    tool_width = max(float(tool_width), 1.0e-3)
    step = max(float(step), 1.0e-3)
    above_surface = max(float(contact_height), float(min_contact_height))
    wipe_z = surface + above_surface
    hover_dz = float(approach_height)
    passes = plan_passes(half_u, half_v, tool_width, step)

    out: Output = {
        "route": "unreachable",
        "detail": "",
        "coverage": 0.0,
        "passes_planned": len(passes),
        "passes_done": 0,
        "surface_z": surface,
        "wipe_height": above_surface,
        "min_height": 0.0,
        "max_height": 0.0,
        "path": [],
    }

    def point(along: float, across: float, z: float) -> np.ndarray:
        return np.array([*(centre + u * along + v * across), z])

    def move(tool: str, position: np.ndarray, rotation: dict[str, float]) -> str:
        try:
            return _refusal(ctx.tool(tool, pose={"position": _vec(position), "rotation": rotation}))
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc) or type(exc).__name__

    def where(phase: str, k: int) -> np.ndarray:
        at = _xyz(_pose(ctx.tool("robot.get_ee_pose")))
        out["path"].append({**_vec(at), "phase": phase, "pass": k})
        return at

    def out_of_reach(p: np.ndarray, rotation: dict[str, float]) -> bool:
        """Whether IK says the pose cannot be held: a stall at the edge of reach
        looks like contact. A connector without a usable answer gets the benefit
        of the doubt."""
        try:
            reply = ctx.tool("robot.solve_ik", pose={"position": _vec(p), "rotation": rotation})
        except Exception as exc:
            logger.info("wiping-a-region: no IK preview (%s)", exc)
            return False
        if not isinstance(reply, dict):
            return False
        return any(
            reply.get(k) is not None and not bool(reply[k])
            for k in ("success", "within_servo_tolerance")
        )

    start = _pose(ctx.tool("robot.get_ee_pose"))
    if close_hand:
        ctx.tool("robot.close_gripper")

    # Rotations, in order: top-down with the closed fingers side by side across
    # the strokes (both wrist solutions), then the hand's own.
    across = []
    try:
        heading = math.degrees(math.atan2(u[1], u[0])) + 90.0
        frame = ctx.tool("robot.grasp_frame", close_heading_deg=heading)
        across = [
            frame[k]
            for k in ("rotation", "rotation_flipped")
            if isinstance(frame, dict) and frame.get(k)
        ]
    except Exception as exc:
        logger.info("wiping-a-region: no top-down rotation (%s)", exc)
    own = [start["rotation"]]
    rotations = own + across if keep_rotation else across + own

    first_offset, first_along = passes[0]
    first_hover = point(first_along[0], first_offset, wipe_z + hover_dz)
    rotation, why = None, "no rotation reached the first pass's hover"
    for candidate in rotations:
        why = ""
        if float(start["position"]["z"]) < first_hover[2] - 1.0e-3:
            rise = _xyz(start)
            rise[2] = first_hover[2]
            why = move("robot.go_to_pose", rise, candidate)
        why = why or move("robot.go_to_pose", first_hover, candidate)
        off = float(np.linalg.norm(where("hover", 0) - first_hover))
        if not why and off <= REACH_TOLERANCE:
            rotation = candidate
            break
        why = why or f"the hand ended {off * 1e3:.0f} mm from the hover"
    if rotation is None:
        out["detail"] = f"hover: {why}"
        return out

    strokes: list[np.ndarray] = []
    heights: list[float] = []
    notes: list[str] = []
    lift_why = ""
    for k, (offset, along) in enumerate(passes):
        begin = point(along[0], offset, wipe_z)
        end = point(along[-1], offset, wipe_z)
        if check_reach and (out_of_reach(begin, rotation) or out_of_reach(end, rotation)):
            notes.append(f"pass {k}: out of reach")
            continue
        hover = begin + np.array([0.0, 0.0, hover_dz])
        why = move("robot.go_to_pose_cartesian", hover, rotation)
        off = float(np.linalg.norm(where("hover", k) - hover))
        if why or off > REACH_TOLERANCE:
            notes.append(f"pass {k}: hover {why or f'ended {off * 1e3:.0f} mm off'}")
            continue

        # The descent is the surface probe: it stops where the tool meets
        # something, so a surface higher than estimated is wiped where it is.
        descended = False
        if probe_surface:
            why = move("robot.move_cartesian_until_contact", begin, rotation)
            at = where("down", k)
            # A connector without the tool (or one that refused before moving)
            # leaves the hand at the hover: descend the ordinary way.
            descended = float(np.linalg.norm(at - hover)) > HEIGHT_TOLERANCE or not why
        if not descended:
            why = move("robot.go_to_pose_cartesian", begin, rotation)
            at = where("down", k)
        if float(np.linalg.norm(at[:2] - begin[:2])) > REACH_TOLERANCE:
            notes.append(
                f"pass {k}: descent ended {np.linalg.norm(at[:2] - begin[:2]) * 1e3:.0f} mm aside"
            )
            lift = move("robot.go_to_pose_cartesian", at + np.array([0.0, 0.0, hover_dz]), rotation)
            where("lift", k)
            lift_why = lift and f"pass {k}: {lift}"
            continue
        raise_ = float(at[2] - wipe_z)
        if raise_ > MAX_RAISE:
            notes.append(f"pass {k}: descent stopped {raise_ * 1e3:.0f} mm over the wiping height")
            lift = move("robot.go_to_pose_cartesian", at + np.array([0.0, 0.0, hover_dz]), rotation)
            where("lift", k)
            lift_why = lift and f"pass {k}: {lift}"
            continue
        z = float(at[2]) if raise_ > HEIGHT_TOLERANCE else wipe_z
        if z != wipe_z:
            logger.info("wiping-a-region: pass %d met the surface %.1f mm high", k, raise_ * 1e3)

        stroke = [at]
        stopped = ""
        for a in along[1:]:
            target = point(a, offset, z)
            why = move("robot.go_to_pose_cartesian", target, rotation)
            at = where("stroke", k)
            stroke.append(at)
            off = float(np.linalg.norm(at - target))
            if why or off > REACH_TOLERANCE:
                stopped = why or f"ended {off * 1e3:.0f} mm off a waypoint"
                break
        touching = np.array([p for p in stroke if p[2] <= z + HEIGHT_TOLERANCE])
        if len(touching):
            strokes.append(touching)
            heights.extend(float(p[2] - surface) for p in touching)
        out["passes_done"] += 1
        if stopped:
            notes.append(f"pass {k}: stroke stopped ({stopped})")
        lift = move("robot.go_to_pose_cartesian", at + np.array([0.0, 0.0, hover_dz]), rotation)
        where("lift", k)
        lift_why = lift and f"pass {k}: {lift}"

    if out["passes_done"] == 0:
        out["detail"] = "; ".join(notes) or "no pass ran"
        return out
    out["coverage"] = coverage(strokes, centre, u, v, half_u, half_v, tool_width)
    if heights:
        out["min_height"], out["max_height"] = min(heights), max(heights)
    if out["coverage"] >= float(min_coverage) and not lift_why:
        out["route"] = "wiped"
        out["detail"] = "; ".join(notes)
        return out
    out["route"] = "partial"
    parts = [f"coverage {out['coverage']:.2f} (needed {float(min_coverage):.2f})", *notes]
    if lift_why:
        parts.append(f"final lift: {lift_why}")
    out["detail"] = "; ".join(parts)
    return out
