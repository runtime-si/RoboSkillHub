# Adapted from predicators (https://github.com/BasisResearch/predicators) at
# afb6f2918091f5ca372a75b9a335aea317a8a209, MIT, Copyright (c) 2022 Rohan Chitnis
# and Tom Silver (license: ../LICENSE.predicators):
# predicators/ground_truth_models/skill_factories/push.py (create_push_skill:
# CloseFingers, Waypoint_0 above and behind the target at transport height,
# Waypoint_1 the descent to contact height, Waypoint_2 the stroke along the
# facing with an optional overshoot and a speed clamp, Waypoint_3 the retreat)
# and predicators/ground_truth_models/domino/options.py (_create_sf_push,
# _transport_z_push = 1.5 tile heights, _offset_x = 3 tile depths, _offset_z =
# 0.55 tile heights over the tile's centre).
# Changes: the domino's simulator pose is a planned slot (planning-a-domino-layout's
# slot 0, world frame, its x axis the facing); the contact height is the closed
# fingertips' height over the table as a fraction of the tile's height, on its
# upper half; the stroke ends a computed distance past the tile's back face (the
# travel that carries the tile past its tipping angle at that height, plus an
# overshoot) instead of at the tile's centre, and is refused when it would reach
# the next slot; the hand closes across the tile's width so the closed blades
# span it; the stroke's speed clamp is the connector's speed_scale; BiRRT's
# collision-free transit is a rise to transport height, a move over the start
# and a vertical descent, with robot.solve_ik previews; the retreat backs off
# along the stroke and rises instead of going home; and the simulator's
# "toppled" state is a depth check of the slot from the observation's RGB-D
# cameras (the depth judge of placing-an-upright-domino-beside-neighbours).
"""Topple the first domino of a line: come down behind it, push it along its facing past its tipping point, back off."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

TILE_SIZE = {"height": 0.048, "width": 0.024, "thickness": 0.0075}
"""A standard domino standing on its end [m]: 48 x 24 x 7.5 mm (Franka-Domino-Project's
and zmatt.net's tiles; the planning and placing skills' default)."""

CONTACT_FRACTION = 0.7
"""Height of the closed fingertips over the table at the stroke, as a fraction of the
tile's height: on its upper half, where the least force tips it (upstream's 0.55 tile
heights over the tile's centre puts the Panda's pads at its top edge). The fingertips
are the lowest point of the hand, and a tile tipping away from a vertical finger face
stays in contact with it there, so this is the contact height. Measured in a YAM
station twin over 0.4-0.85 (see the SKILL.md)."""

CONTACT_BAND = (0.35, 0.9)
"""Contact fractions accepted [-]. Under 0.35 a fingertip a few millimetres low meets the
table and a tile on a slippery table slides rather than tips (it tips when pushed higher
than half its thickness over the table friction coefficient); over 0.9 a table or
tile-height error of 4 mm lets the hand pass over the tile. 0.35 and 0.85 both toppled
the tile in a YAM station twin."""

MIN_OVERLAP = 0.004
"""Least height of tile above the fingertips [m]: a 4 mm table-height error must not
put the hand over the tile."""

TIP_REACH = 0.0102
"""How far the closed linear_4310 fingertips reach under the tool centre [m] when
robot.describe_gripper does not say."""

MIN_TOOL_HEIGHT = 0.015
"""Lowest tool-centre height over the table [m]: a real YAM station refuses under 14 mm."""

FINGER_HALF_DEPTH = 0.012
"""Half the closed hand's extent along the stroke [m] with the jaws closing across the
tile (a linear_4310 blade about 24 mm wide; placing-an-upright-domino-beside-neighbours'
value, not measured). The stroke's distances are the blade front's."""

CLOSE_HEADING_OFFSET_DEG = 90.0
"""Jaw closing heading minus the facing [deg]: across the tile's width, so the two
closed blades (30 mm each along the closing axis) span the 24 mm tile and the push
stays centred on it with a lateral error (upstream: the robot's push_ee_yaw_offset)."""

APPROACH_CLEARANCE = 0.008
"""Gap between the blade front and the tile's back face where the hand comes down [m].
Upstream's _offset_x is 3 tile depths from the centre (22.5 mm for a 7.5 mm tile),
which a 24 mm blade would not clear."""

OVERSHOOT = 0.008
"""How far past the tipping travel the blade front pushes [m] (upstream's
stroke_overshoot_fn): covers a servo's 4 mm arrival ball and a few mm of slot error."""

NEXT_CLEARANCE = 0.006
"""Least gap kept between the stroke's end and the next tile's back face [m], so the
hand never pushes the second tile itself."""

SPEED_SCALE = 0.25
"""The stroke's speed as the connector's speed_scale on robot.go_to_pose_cartesian
(upstream's stroke_step_norm_fn speed clamp): a slow push. In a YAM station twin
0.25 ran the stroke at about 0.04-0.05 m/s (see the SKILL.md)."""

TRANSIT_CLEARANCE = 0.02
"""Height the fingertips keep over the tile's top on the way to the start and away
from it [m]; upstream's _transport_z_push is 1.5 tile heights (24 mm over a 48 mm tile)."""

RETREAT = 0.03
"""How far back along the stroke the hand goes before it rises [m]: the tile falls
away from the hand, so backing off first never meets it."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from a free move's target [m]: no tighter than a real
YAM station's 12 mm per move."""

DESCENT_TOLERANCE = 0.006
"""A descent to the stroke's start that ends this far high stopped on something [m]:
over a servo's 4 mm arrival ball."""

STROKE_TOLERANCE = 0.005
"""A stroke that ends this much short of its target was held, not just inside the
servo's arrival ball [m]: a servo's 4 mm ball plus a millimetre (the YAM station twin
ended its strokes 1.4-2.8 mm short in free air)."""

LATERAL_TOLERANCE = 0.006
"""How far across the facing the hand may end from the stroke's line [m]."""

HEIGHT_TOLERANCE = 0.010
POSITION_TOLERANCE = 0.008
SEARCH_MARGIN = 0.006
OCCLUDER_RISE = 0.015
OCCLUDED_SHARE = 0.25
TOP_BAND = 0.006
MIN_POINTS = 4
DEPTH_RANGE = (0.07, 3.0)
"""The depth judge's constants, placing-an-upright-domino-beside-neighbours' (a standing
tile's top within 10 mm of its height and 8 mm of the slot; the window widened by 6 mm;
a window a quarter covered by something 15 mm over the tile is occluded)."""

PREFERRED_CAMERAS = ("overhead", "top")
"""Camera names tried first when no camera_name is given (the station's calibrated overhead)."""

REFUSALS = ("unreachable", "bad_speed_scale")
"""Routed outcomes that mean a move was refused, not stopped."""


class Output(TypedDict):
    route: str
    detail: str
    pushed: bool
    contact_height: float
    tip_travel: float
    push_travel: float
    stroke_length: float
    stroke_time: float
    stroke_speed: float
    stroke_steps: int
    trigger_time: float
    push_direction: dict[str, float]
    verified: bool
    first_standing: bool
    next_standing: bool


# ----------------------------------------------------------------------------- helpers


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _matrix(q: dict[str, float] | None) -> np.ndarray:
    q = q or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
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


def tile_pose(pose: dict[str, Any]) -> tuple[np.ndarray, float]:
    """A slot's (xy, facing [deg]): an Se3Pose whose x axis is the facing (the way
    the tile topples and the row runs), or ``facing_deg`` beside a ``position``."""
    p = pose.get("position") or pose.get("center") or pose
    if "facing_deg" in pose or "yaw_deg" in pose:
        facing = float(pose.get("facing_deg", pose.get("yaw_deg", 0.0)))
    elif pose.get("rotation") or pose.get("orientation"):
        r = _matrix(pose.get("rotation") or pose.get("orientation"))
        facing = math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))
    else:
        facing = 0.0
    return np.array([float(p["x"]), float(p["y"])]), facing


def _axes(facing_deg: float) -> tuple[np.ndarray, np.ndarray]:
    a = math.radians(facing_deg)
    return np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])


def tip_travel(height: float, thickness: float, contact: float) -> float:
    """How far a point of the tile's back face ``contact`` over the table moves along
    the facing before the tile's centre of mass passes over its front foot (the
    tipping angle atan(thickness / height)) [m]. Past it the tile falls on its own."""
    theta = math.atan2(thickness, height)
    return thickness * (1.0 - math.cos(theta)) + contact * math.sin(theta)


# ----------------------------------------------------------------------------- depth check


def _cameras(obs: Any) -> list[dict[str, Any]]:
    cams = obs.get("cameras") if isinstance(obs, dict) else None
    if isinstance(cams, dict):
        return [dict(v, name=k) for k, v in cams.items() if isinstance(v, dict)]
    return [c for c in (cams or []) if isinstance(c, dict)]


def order_cameras(cams: list[dict[str, Any]], camera_name: str) -> list[dict[str, Any]]:
    if camera_name:
        return [c for c in cams if c.get("name") == camera_name]
    rank = {n: i for i, n in enumerate(PREFERRED_CAMERAS)}
    return sorted(cams, key=lambda c: rank.get(str(c.get("name", "")), len(rank)))


def _camera_to_world(pose: Any) -> np.ndarray:
    if isinstance(pose, dict) and "position" in pose:
        t = np.eye(4)
        t[:3, :3] = _matrix(pose.get("rotation") or pose.get("orientation"))
        t[:3, 3] = _xyz(pose)
        return t
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)


def camera_points(cam: dict[str, Any], corners: np.ndarray) -> np.ndarray:
    """World points (N x 3) of the camera's depth pixels inside the image box of the
    world ``corners`` (OpenCV pinhole); none when that box is not wholly in the image."""
    if cam.get("depth") is None or cam.get("intrinsics") is None or cam.get("pose") is None:
        return np.zeros((0, 3))
    depth = np.asarray(cam["depth"], dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    k = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
    t = _camera_to_world(cam["pose"])
    inv = np.linalg.inv(t)
    c = (inv[:3, :3] @ corners.T).T + inv[:3, 3]
    if np.any(c[:, 2] <= DEPTH_RANGE[0]):
        return np.zeros((0, 3))
    u = k[0, 0] * c[:, 0] / c[:, 2] + k[0, 2]
    v = k[1, 1] * c[:, 1] / c[:, 2] + k[1, 2]
    h, w = depth.shape
    u0, u1 = int(math.floor(u.min())) - 2, int(math.ceil(u.max())) + 3
    v0, v1 = int(math.floor(v.min())) - 2, int(math.ceil(v.max())) + 3
    if u0 < 0 or v0 < 0 or u1 > w or v1 > h:
        return np.zeros((0, 3))
    z = depth[v0:v1, u0:u1]
    uu, vv = np.meshgrid(np.arange(u0, u1, dtype=np.float64), np.arange(v0, v1, dtype=np.float64))
    ok = np.isfinite(z) & (z >= DEPTH_RANGE[0]) & (z <= DEPTH_RANGE[1])
    z, uu, vv = z[ok], uu[ok], vv[ok]
    pts = np.stack([(uu - k[0, 2]) * z / k[0, 0], (vv - k[1, 2]) * z / k[1, 1], z], axis=1)
    return (t[:3, :3] @ pts.T).T + t[:3, 3]


def judge_tile(
    cams: list[dict[str, Any]],
    xy: np.ndarray,
    facing_deg: float,
    table_z: float,
    size: dict[str, float],
) -> dict[str, Any]:
    """Whether a tile stands at (xy, facing): the highest depth points inside its
    footprint widened by SEARCH_MARGIN are its top; it stands when that top is within
    HEIGHT_TOLERANCE of its height and POSITION_TOLERANCE of the slot. Judged by the
    first camera that sees the whole footprint unoccluded."""
    along, across = _axes(facing_deg)
    h = size["height"]
    half_a = size["thickness"] / 2.0 + SEARCH_MARGIN
    half_w = size["width"] / 2.0 + SEARCH_MARGIN
    corners = np.array(
        [
            [*(xy + sa * half_a * along + sw * half_w * across), table_z + z]
            for sa in (-1, 1)
            for sw in (-1, 1)
            for z in (0.0, h + OCCLUDER_RISE)
        ]
    )
    best: dict[str, Any] = {
        "seen": False,
        "standing": False,
        "points": 0,
        "why": "no camera sees it",
    }
    for cam in cams:
        pts = camera_points(cam, corners)
        if not len(pts):
            continue
        rel = pts[:, :2] - xy
        a, w, z = rel @ along, rel @ across, pts[:, 2] - table_z
        inside = (np.abs(a) <= half_a) & (np.abs(w) <= half_w)
        over = int(np.sum(z > h + OCCLUDER_RISE))
        keep = inside & (z > -0.005) & (z <= h + OCCLUDER_RISE)
        raised = keep & (z > 0.004)
        n = int(raised.sum())
        name = cam.get("name", "")
        if over >= OCCLUDED_SHARE * len(z):
            verdict = {
                "seen": False,
                "standing": False,
                "points": n,
                "camera": name,
                "why": "occluded",
            }
        elif int(keep.sum()) < MIN_POINTS:
            verdict = {
                "seen": False,
                "standing": False,
                "points": n,
                "camera": name,
                "why": "no depth",
            }
        elif n < MIN_POINTS:
            verdict = {
                "seen": True,
                "standing": False,
                "points": int(keep.sum()),
                "camera": name,
                "top": float(np.max(z[keep])),
                "why": "nothing stands there",
            }
        else:
            top = float(np.percentile(z[raised], 95))
            band = raised & (z >= top - TOP_BAND)
            offset = math.hypot(float(np.mean(a[band])), float(np.mean(w[band])))
            standing = top >= h - HEIGHT_TOLERANCE and offset <= POSITION_TOLERANCE
            verdict = {
                "seen": True,
                "standing": bool(standing),
                "points": n,
                "camera": name,
                "top": top,
                "offset": offset,
                "why": ""
                if standing
                else f"its top is {top * 1e3:.0f} mm over the table, {offset * 1e3:.0f} mm off the slot",
            }
        if verdict["seen"]:
            return verdict
        if verdict["points"] >= best["points"]:
            best = verdict
    return best


# ----------------------------------------------------------------------------- the skill


def run(
    ctx: NodeContext,
    table_z: float,
    slot: dict[str, Any] | None = None,
    slots: list[dict[str, Any]] | None = None,
    slot_index: int = 0,
    next_slot: dict[str, Any] | None = None,
    tile_size: dict[str, float] | None = None,
    contact_fraction: float = CONTACT_FRACTION,
    overshoot: float = OVERSHOOT,
    speed_scale: float = SPEED_SCALE,
    approach_clearance: float = APPROACH_CLEARANCE,
    retreat: float = RETREAT,
    transit_clearance: float = TRANSIT_CLEARANCE,
    close_heading_offset_deg: float = CLOSE_HEADING_OFFSET_DEG,
    finger_half_depth: float = FINGER_HALF_DEPTH,
    verify: bool = True,
    camera_name: str = "",
    park_pose: dict[str, Any] | None = None,
    reach_tolerance: float = REACH_TOLERANCE,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    size = {**TILE_SIZE, **{k: float(v) for k, v in (tile_size or {}).items()}}
    h, t = size["height"], size["thickness"]
    table_z = float(table_z)
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "pushed": False,
        "contact_height": -1.0,
        "tip_travel": 0.0,
        "push_travel": 0.0,
        "stroke_length": 0.0,
        "stroke_time": 0.0,
        "stroke_speed": 0.0,
        "stroke_steps": -1,
        "trigger_time": 0.0,
        "push_direction": {"x": 0.0, "y": 0.0, "z": 0.0},
        "verified": False,
        "first_standing": True,
        "next_standing": True,
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        return out

    # --- the line: the tile to push and, when known, the next one and the one behind
    behind_slot = None
    if slot is None:
        idx = int(slot_index)
        if not slots or not 0 <= idx < len(slots):
            return finish(
                "unreachable",
                f"no slot: slot_index {slot_index} is not one of the {len(slots or [])} slots",
            )
        slot = slots[idx]
        if next_slot is None and idx + 1 < len(slots):
            next_slot = slots[idx + 1]
        if idx > 0:
            behind_slot = slots[idx - 1]
    xy, facing = tile_pose(slot)
    along, across = _axes(facing)
    out["push_direction"] = {"x": float(along[0]), "y": float(along[1]), "z": 0.0}

    try:
        hand = ctx.tool("robot.describe_gripper", **on_arm)
    except Exception as exc:  # the fingertip reach falls back to the linear_4310's
        logger.info("triggering-a-domino-chain: no gripper description (%s)", exc)
        hand = {}
    finger = (hand if isinstance(hand, dict) else {}).get("finger") or {}
    tip_reach = float(finger.get("reach_m") or 0.0) if finger.get("stated", True) else 0.0
    tip_reach = tip_reach or TIP_REACH

    # --- the geometry of the stroke (push.py's waypoints, distances from the blade front)
    frac = float(contact_fraction)
    contact = frac * h
    if not CONTACT_BAND[0] <= frac <= CONTACT_BAND[1] or h - contact < MIN_OVERLAP:
        return finish(
            "unreachable",
            f"contact_fraction {frac:.2f} is outside {CONTACT_BAND[0]:.2f}-{CONTACT_BAND[1]:.2f} "
            f"(or leaves under {MIN_OVERLAP * 1e3:.0f} mm of tile over the fingertips)",
        )
    tool_z = table_z + contact + tip_reach
    if tool_z < table_z + MIN_TOOL_HEIGHT:
        return finish(
            "unreachable",
            f"the contact height puts the tool centre {(tool_z - table_z) * 1e3:.0f} mm over the table "
            f"(at least {MIN_TOOL_HEIGHT * 1e3:.0f} mm)",
        )
    travel = tip_travel(h, t, contact)
    past = travel + float(overshoot)
    out["tip_travel"] = travel
    back = xy - along * (t / 2.0)
    fhd = float(finger_half_depth)
    start_xy = back - along * (float(approach_clearance) + fhd)
    end_xy = back + along * (past - fhd)
    if next_slot is not None:
        n_xy, _ = tile_pose(next_slot)
        room = float((n_xy - back) @ along) - t / 2.0
        if past > room - NEXT_CLEARANCE:
            return finish(
                "unreachable",
                f"the stroke must carry the blade {past * 1e3:.1f} mm past the tile's back face to tip it, "
                f"but the next tile's back face is {room * 1e3:.1f} mm away (keeping {NEXT_CLEARANCE * 1e3:.0f} mm): "
                "the line is too tight to push the first tile alone",
            )
    if behind_slot is not None:
        b_xy, _ = tile_pose(behind_slot)
        behind = float((back - b_xy) @ along)
        gap = behind - t / 2.0
        need = float(approach_clearance) + 2.0 * fhd
        if behind > 0.0 and gap < need:
            return finish(
                "unreachable",
                f"the tile behind stands {gap * 1e3:.0f} mm back, where the hand comes down "
                f"(it needs {need * 1e3:.0f} mm)",
            )
    out["stroke_length"] = float(np.linalg.norm(end_xy - start_xy))
    start = np.array([*start_xy, tool_z])
    end = np.array([*end_xy, tool_z])
    transit_z = table_z + h + float(transit_clearance) + tip_reach

    def here() -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **on_arm))

    def move(
        tool: str, position: np.ndarray, rot: dict[str, float], **knobs: Any
    ) -> tuple[str, Any]:
        """(why it did not arrive or "", the reply)."""
        pose = {"position": _vec(position), "rotation": rot}
        try:
            try:
                reply = ctx.tool(tool, pose=pose, **knobs, **on_arm)
            except TypeError as exc:
                if not knobs or "keyword" not in str(exc):
                    raise
                logger.info(
                    "triggering-a-domino-chain: %s takes no %s (%s)", tool, sorted(knobs), exc
                )
                reply = ctx.tool(tool, pose=pose, **on_arm)
        except Exception as exc:  # a refused or failed move is routed, not raised
            return (str(exc) or type(exc).__name__), None
        return _refusal(reply), reply

    def go(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        """A free move judged by where the hand is: "" within reach_tolerance (a servo
        timeout that ended that close has arrived)."""
        why, _ = move(tool, position, rot)
        off = float(np.linalg.norm(_xyz(here()) - position))
        if (why and not why.startswith("timeout")) or off > float(reach_tolerance):
            return f"{why or 'ended off'}; the hand is {off * 1e3:.0f} mm from it"
        return ""

    def observe() -> list[dict[str, Any]]:
        try:
            return order_cameras(_cameras(ctx.tool("robot.get_observation")), camera_name)
        except Exception as exc:
            logger.info("triggering-a-domino-chain: no observation (%s)", exc)
            return []

    # --- something must stand at the slot (when a camera can tell)
    notes: list[str] = []
    if verify:
        before = judge_tile(observe(), xy, facing, table_z, size)
        if before["seen"] and not before["standing"]:
            out["first_standing"] = False
            return finish(
                "no_contact",
                f"nothing stands at the trigger slot before the push ({before['why']}, "
                f"{before.get('camera', '')}); nothing moved",
            )
        if not before["seen"]:
            notes.append(f"the slot was not checked before the push ({before['why']})")

    # --- the hand: closed, blades across the tile, both wrist solutions previewed
    heading = facing + float(close_heading_offset_deg)
    try:
        gf = ctx.tool("robot.grasp_frame", close_heading_deg=heading, **on_arm)
    except Exception as exc:
        return finish("unreachable", f"no hand frame for the push ({exc})")
    rots = [gf[k] for k in ("rotation", "rotation_flipped") if isinstance(gf, dict) and gf.get(k)]
    if not rots:
        return finish("unreachable", "no hand frame for the push")

    def reachable(rot: dict[str, float]) -> bool:
        if not check_reach:
            return True
        for point in (start, end):
            try:
                reply = ctx.tool(
                    "robot.solve_ik", pose={"position": _vec(point), "rotation": rot}, **on_arm
                )
            except Exception as exc:
                logger.info("triggering-a-domino-chain: IK preview refused (%s)", exc)
                return False
            # A stall at the edge of reach reads like a blocked stroke.
            if isinstance(reply, dict) and (
                reply.get("success") is False or reply.get("within_servo_tolerance") is False
            ):
                return False
        return True

    rots = [r for r in rots if reachable(r)]
    if not rots:
        return finish(
            "unreachable",
            "the stroke's start or end is out of reach (IK preview, both wrist solutions)",
        )
    try:
        _refusal(ctx.tool("robot.close_gripper", **on_arm))
    except Exception as exc:
        notes.append(f"the hand was not closed ({exc})")

    # --- Waypoint_0: over the start at transport height; Waypoint_1: straight down
    now = _xyz(here())
    start_pose = now.copy()
    if now[2] < transit_z - 1.0e-3:
        why = go("robot.go_to_pose_cartesian", np.array([now[0], now[1], transit_z]), rots[0])
        if why:
            return finish("unreachable", f"rise before the push: {why}")
    above = np.array([*start_xy, transit_z])
    why, rot = "", rots[0]
    for rot in rots:
        why = go("robot.go_to_pose", above, rot)
        if not why:
            break
    if why:
        return finish("unreachable", f"over the stroke's start: {why}")
    rot = here()["rotation"]
    why, _ = move("robot.go_to_pose_cartesian", start, rot)
    reached = _xyz(here())
    high = float(reached[2] - start[2])
    if why.split(":")[0] in REFUSALS:
        return finish("unreachable", f"the descent behind the tile was refused ({why})")
    if high > DESCENT_TOLERANCE:
        go("robot.go_to_pose_cartesian", np.array([*reached[:2], transit_z]), rot)
        return finish(
            "blocked",
            f"the descent behind the tile stopped {high * 1e3:.1f} mm high ({why or 'stalled'}): "
            "something stands where the hand comes down (a tile behind, the tile not where planned); lifted back",
        )
    if float(np.linalg.norm(reached[:2] - start_xy)) > float(reach_tolerance):
        go("robot.go_to_pose_cartesian", np.array([*reached[:2], transit_z]), rot)
        return finish("unreachable", "the descent ended off the stroke's start; lifted back")
    out["contact_height"] = float(reached[2] - tip_reach - table_z)

    # --- Waypoint_2: the stroke along the facing, slow, a short way past the tipping travel
    knobs = {"speed_scale": float(speed_scale)} if 0.0 < float(speed_scale) < 1.0 else {}
    out["trigger_time"] = time.time()
    t0 = time.monotonic()
    why, reply = move("robot.go_to_pose_cartesian", end, rot, **knobs)
    out["stroke_time"] = time.monotonic() - t0
    if isinstance(reply, dict) and isinstance(reply.get("steps"), int | float):
        out["stroke_steps"] = int(reply["steps"])
    stop = _xyz(here())
    blade = float((stop[:2] - back) @ along) + fhd
    out["push_travel"] = blade
    moved = float((stop[:2] - start_xy) @ along)
    out["stroke_speed"] = moved / out["stroke_time"] if out["stroke_time"] > 0 else 0.0
    lateral = abs(float((stop[:2] - start_xy) @ across))
    if why.split(":")[0] in REFUSALS and moved < 0.002:
        return finish(
            "unreachable", f"the stroke was refused ({why}); the hand is behind the tile, low"
        )

    # --- Waypoint_3: back off along the stroke, then up (not home: the chain is falling)
    back_off = stop.copy()
    back_off[:2] -= along * float(retreat)
    retreat_why = go("robot.go_to_pose_cartesian", back_off, rot)
    if retreat_why:
        notes.append(f"backing off: {retreat_why}")
    up = _xyz(here())
    rise_why = go("robot.go_to_pose_cartesian", np.array([up[0], up[1], transit_z]), rot)
    if rise_why:
        notes.append(f"rising: {rise_why}")

    hand_note = (
        f"fingertips {out['contact_height'] * 1e3:.1f} mm over the table, blade {blade * 1e3:.1f} mm past "
        f"the back face (tipping needs {travel * 1e3:.1f} mm), {lateral * 1e3:.1f} mm off the line"
    )
    short = float(np.linalg.norm(end_xy - start_xy)) - moved
    held = why.split(":")[0] in ("timeout", "contact") or short > STROKE_TOLERANCE
    if blade < travel and held:
        return finish(
            "blocked",
            f"the stroke stopped {short * 1e3:.1f} mm short ({why or 'ended short'}): {hand_note}; the hand was "
            "held before the tile could tip (a tile that does not give, or the arm at its limit)"
            + (f"; {'; '.join(notes)}" if notes else ""),
        )
    if blade < travel:
        notes.append(
            f"the stroke ended {short * 1e3:.1f} mm short of its end, inside the arrival tolerance, "
            "and short of the tipping travel"
        )
    if lateral > LATERAL_TOLERANCE:
        notes.append(f"the stroke ran {lateral * 1e3:.1f} mm off its line")
    out["pushed"] = blade >= travel
    tail = f"; {'; '.join(notes)}" if notes else ""
    if not verify:
        if not out["pushed"]:
            return finish(
                "no_contact", f"{hand_note}; the tile was not carried past its tipping angle{tail}"
            )
        return finish(
            "triggered", f"{hand_note}; what the hand did only, not checked from depth{tail}"
        )

    # --- the check: out of the cameras' way, then the slot (and the next one)
    park = (
        _xyz(_pose(park_pose))
        if park_pose
        else np.array([*start_pose[:2], max(float(start_pose[2]), transit_z)])
    )
    why = go("robot.go_to_pose", park, rot)
    if why:
        notes.append(f"park: {why}")
        tail = f"; {'; '.join(notes)}"
    cams = observe()
    first = judge_tile(cams, xy, facing, table_z, size)
    nxt = (
        judge_tile(cams, tile_pose(next_slot)[0], tile_pose(next_slot)[1], table_z, size)
        if next_slot
        else None
    )
    if nxt is not None and nxt["seen"]:
        out["next_standing"] = bool(nxt["standing"])
    next_note = (
        ""
        if nxt is None
        else (
            f"; the next tile {'stands' if nxt['standing'] else 'is down'}"
            if nxt["seen"]
            else f"; the next tile not seen ({nxt['why']})"
        )
    )
    if not first["seen"]:
        return finish(
            "unverified",
            f"{hand_note}; the slot could not be checked ({first['why']}){next_note}{tail}",
        )
    out["verified"] = True
    out["first_standing"] = bool(first["standing"])
    if first["standing"]:
        why_up = (
            "the hand passed it, or it is not where the slot says"
            if out["pushed"]
            else "the stroke did not carry it past its tipping angle and it rocked back"
        )
        return finish(
            "no_contact",
            f"{hand_note}; but the tile still stands at the slot (top {first['top'] * 1e3:.0f} mm, "
            f"{first['camera']}): {why_up}{next_note}{tail}",
        )
    return finish(
        "triggered",
        f"{hand_note}; the slot reads down ({first['why']}, {first['camera']}){next_note}{tail}",
    )
