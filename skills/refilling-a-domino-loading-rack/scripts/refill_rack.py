# The rack geometry, the slot judge, the jaw read-back and the pick are those of
# skills/picking-upright-dominoes-from-a-loading-rack/scripts/pick_from_rack.py and
# skills/placing-an-upright-domino-beside-neighbours/scripts/place_domino.py, which adapt predicators
# (https://github.com/BasisResearch/predicators, commit afb6f2918091f5ca372a75b9a335aea317a8a209), MIT,
# Copyright (c) 2022 Rohan Chitnis and Tom Silver (license: ../LICENSE.predicators):
# predicators/ground_truth_models/skill_factories/pick.py (MoveAbove, PreOpen, MoveToGrasp, Grasp, Lift) and
# skill_factories/place.py (MoveAbove, Descend, OpenFingers a step, Retreat holding that width).
# New here: the empty slot chosen from depth, the hold height kept over the rack's walls, a descent into the slot
# from just over the walls with offsets along the row tried when it stops on a wall (a held tile has 1 mm a side
# in a 9.5 mm slot), and the slot read again after the release.
"""Put a standing domino back into an empty slot of a loading rack: pick it, lower it into the slot, check it stands there."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

TILE_SIZE = {"height": 0.048, "width": 0.024, "thickness": 0.0075}
"""A standard domino standing on its end [m], as placing-an-upright-domino-beside-neighbours."""

WALL_HEIGHT = 0.020
"""How high the rack's walls rise over its floor [m]: 42 % of a 48 mm tile. The fingertips
must close over them inside GRASP_BAND: 20 mm leaves tool-centre heights of 32-46 mm,
the candidate's 25 mm leaves 37-46 mm, and over 33.8 mm leaves none for the linear_4310."""

GRASP_HEIGHT = 0.030
"""Preferred tool-centre height over the tile's bottom at the close [m] (the placing
skill's); raised to clear the walls when they are higher."""

GRASP_BAND = (0.015, 0.046)
"""Tool-centre heights over the tile's bottom that carried and stood a 48 mm tile in a
YAM station twin (placing-an-upright-domino-beside-neighbours' measurement) [m]."""

TIP_REACH = 0.0102
"""How far the closed linear_4310 fingertips reach under the tool centre [m] when
robot.describe_gripper does not say."""

MIN_TOOL_HEIGHT = 0.015
"""Lowest tool-centre height over the table [m]: a real YAM station refuses under 14 mm."""

MIN_BITE = 0.010
"""Least height of tile between the fingertips and its top [m] (placing skill: a 2.9 mm
bite lost the tile in transit, 9.9 mm carried it)."""

WALL_CLEARANCE = 0.002
"""Least gap between the fingertips and the walls' top at the close [m]."""

LIFT_CLEARANCE = 0.010
"""How far over the walls' top the tile's bottom is lifted straight up before any
sideways move [m]."""

GRIP_CLEARANCE = 0.004
"""Jaw clearance each side of the tile when opening for the grasp [m] (placing skill)."""

TRANSIT_CLEARANCE = 0.02
"""Height the held tile's bottom (or the open fingertips) keeps over every tile top in the
rack in transit [m] (placing skill)."""

FINGER_HALF_DEPTH = 0.012
FINGER_BLADE = 0.030
NEIGHBOUR_CLEARANCE = 0.004
"""The open linear_4310 finger's half extent along the row, its blade depth, and the least
gap kept to a standing neighbour [m] (placing skill): a rack pitch under 19.8 mm puts an
open finger on the next tile."""

EMPTY_MARGIN = 0.003
NARROW_MARGIN = 0.006
FULL_MARGIN = 0.008
"""Jaw-gap bands [m]: within EMPTY_MARGIN of closed holds nothing, NARROW_MARGIN under the
tile's width did not close across it, FULL_MARGIN over it closed on something else
(grasping-flat-objects, placing skill)."""

HELD_MARGIN = 0.005
"""A jaw gap from the tile's width less NARROW_MARGIN up to its width plus this is a tile in
the hand [m]: in a YAM station twin a held 24 mm tile read 25.4-26.9 mm, and the placing
skill's release leaves the jaws at 32 mm."""

DESCENT_TOLERANCE = 0.006
"""A straight move that ends this far short of its target was held [m]: over a servo's
4 mm arrival bound."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from a free move's target [m]: a real YAM station's 12 mm per
move bound, and no tighter."""

YAW_TOLERANCE_DEG = 4.0
TURN_TRIES = 3
ROLL_LIMITS = (-2.0944, 2.0944)
LIMIT_MARGIN = 0.05
"""Jaw heading read-back (placing skill): tolerance [deg], last-joint turns tried, the YAM's
joint 6 range and the margin kept from it [rad]."""

HEIGHT_TOLERANCE = 0.010
POSITION_TOLERANCE = 0.008
"""A slot is full when the highest depth points in it are within 10 mm of a tile's height
over the floor and within 8 mm of the slot (placing skill's standing test)."""

EMPTY_TOP = 0.005
"""A slot whose highest points are at most this over the walls' top holds nothing [m]."""

SEARCH_MARGIN = 0.006
OCCLUDER_RISE = 0.015
OCCLUDED_SHARE = 0.25
TOP_BAND = 0.006
MIN_POINTS = 4
DEPTH_RANGE = (0.07, 3.0)
PREFERRED_CAMERAS = ("overhead", "top")
"""Depth judge (placing skill): window margin, what over a tile is an occluder, the share
of occluder that blinds a camera, the top band, fewest points, valid depth, and the
station's calibrated overhead first."""


ENTRY_CLEARANCE = 0.003
"""The held tile's bottom over the walls' top where the descent into the slot starts [m]."""

RELEASE_GAP = 0.0015
"""The tile's bottom over the rack's floor when the jaws open [m] (placing skill)."""

SEAT_PRESS = 0.001
"""How far under the floor the held tile's bottom is commanded before the release [m]: a
tile held a few degrees tilted about the jaw axis squares up on the floor first (a YAM
station twin with its grip fixed lifts parts 4-6.5 deg tilted, SIM_GRIP.md)."""

RELEASE_CLEARANCE = 0.004
"""Jaw opening each side of the tile at the release, held through the retreat [m]
(placing skill)."""

INSERT_TOLERANCE = 0.004
"""A descent into the slot that ends this far higher than the hand's own arrival offset
(read on the free move to just over the walls) stopped on a wall [m]. A tile on a wall's
top stops wall height less the release gap high (18.5 mm for 20 mm walls). In a YAM station
twin the hand arrived 2.2-3.1 mm over every command; judged against the command, at 3 mm,
every seated tile read jammed and the offset retries wedged it out of the slot."""

OFFSETS = (0.0, 0.001, -0.001, 0.002, -0.002)
"""Offsets along the row tried for the descent into the slot [m]: a 9.5 mm slot leaves
1 mm a side for a 7.5 mm tile, and the twin's hand landed 1-4 mm off its target."""

HELD_TURN_STEP_DEG = 15.0
"""Largest turn of the last joint in one move while the hand holds a tile [deg]. In a YAM
station twin a single 90 deg turn of a pinched tile left the tile where it was and the
jaws ended across its thickness (9.3 mm) in 4 of 5 refills."""

ALIGN_TOLERANCE = 0.0005
ALIGN_TRIES = 3
"""Over the slot the held tile's bottom is brought within this of the slot along the row [m],
with up to this many corrective moves read back: a 9.5 mm slot leaves 1 mm a side. Its
place in the hand (from the grasp's read-back) and its tilt (a hand tilted at the grasp
leaves the tile tilted under a level hand) are both taken in. A tile that meets a wall's top
slides up in the pinch while the hand reaches its target, so the hand's height cannot tell
it; in a YAM station twin it was then released on the wall and fell back."""

SEATED_TOP = 0.004
"""After the release a slot counts as refilled only when its top reads within this of a
tile's height over the floor [m]. A tile left leaning on a wall's top reads 36-38 mm,
inside the 10 mm of the pick-side judge; a seated one read 48 mm in a YAM station twin."""

FILL_ORDERS = ("index", "reverse", "nearest")


class Output(TypedDict):
    route: str
    detail: str
    slot_index: int
    picked: bool
    released: bool
    grasp_height: float
    grip_width: float
    jaw_error_deg: float
    wrist_turn_deg: float
    insert_offset_mm: float
    align_error_mm: float
    tile_tilt_deg: float
    attempts: int
    slot_states: list
    slot_state_after: str
    empty_count: int


# ----------------------------------------------------------------------------- geometry


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    if isinstance(p, dict):
        return np.array([float(p["x"]), float(p["y"]), float(p.get("z", 0.0))])
    return np.array([float(v) for v in list(p)[:3]] + [0.0] * (3 - len(list(p)[:3])))


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


def _quat_yaw(deg: float) -> dict[str, float]:
    a = math.radians(deg) / 2
    return {"w": math.cos(a), "x": 0.0, "y": 0.0, "z": math.sin(a)}


def _refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or "" (connectors that route a refusal
    instead of raising)."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def _wrap_half(deg: float) -> float:
    """An angle folded into [-90, 90): the jaws close the same both ways round."""
    return (float(deg) + 90.0) % 180.0 - 90.0


def jaw_error_deg(now: dict[str, float], target: dict[str, float], close: np.ndarray) -> float:
    """How far the hand at ``now`` has its jaws turned about the vertical from where
    ``target`` puts them (``close`` is the target's close axis, world) [deg]."""
    turned = _matrix(now) @ _matrix(target).T @ close
    return math.degrees(
        math.atan2(float(close[0] * turned[1] - close[1] * turned[0]), float(close @ turned))
    )


def _axes(facing_deg: float) -> tuple[np.ndarray, np.ndarray]:
    a = math.radians(facing_deg)
    return np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])


def rack_slots(rack: dict[str, Any]) -> list[tuple[np.ndarray, float, float]]:
    """(xy, facing_deg, floor_z) of every slot: from ``slots`` (poses at the tiles' bottom
    centres, x axis the facing) or from ``origin`` + ``facing_deg`` + ``pitch`` + ``n_slots``
    (+ ``row_deg``, the way the slots advance; default the facing)."""
    if rack.get("slots"):
        out = []
        for s in rack["slots"]:
            p = _xyz(s)
            if "facing_deg" in s:
                f = float(s["facing_deg"])
            elif s.get("rotation"):
                r = _matrix(s["rotation"])
                f = math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))
            else:
                raise ValueError(f"rack slot {s!r} has no facing")
            out.append((p[:2], f, float(p[2])))
        return out
    o = rack["origin"]
    origin = _xyz(o if isinstance(o, dict) and "position" in o else {"position": o})
    facing = float(rack["facing_deg"])
    pitch, n = float(rack["pitch"]), int(rack["n_slots"])
    if pitch <= 0 or n < 1:
        raise ValueError(f"rack pitch {pitch} and n_slots {n} must be positive")
    row = math.radians(float(rack.get("row_deg", facing)))
    step = np.array([math.cos(row), math.sin(row)]) * pitch
    return [(origin[:2] + k * step, facing, float(origin[2])) for k in range(n)]


def finger_conflicts(
    xy: np.ndarray,
    facing_deg: float,
    others: list[tuple[np.ndarray, float]],
    size: dict[str, float],
    open_width: float,
) -> list[int]:
    """Standing tiles whose footprint comes within NEIGHBOUR_CLEARANCE of the open hand over
    ``xy`` (both fingers beside the tile across its width) - the placing skill's check."""
    along, across = _axes(facing_deg)
    reach_a = max(FINGER_HALF_DEPTH, size["thickness"] / 2.0) + NEIGHBOUR_CLEARANCE
    reach_w = open_width / 2.0 + FINGER_BLADE + NEIGHBOUR_CLEARANCE
    bad = []
    for i, (oxy, o_facing) in enumerate(others):
        o_along, o_across = _axes(o_facing)
        corners = [
            oxy + sa * size["thickness"] / 2.0 * o_along + sw * size["width"] / 2.0 * o_across
            for sa in (-1, 1)
            for sw in (-1, 1)
        ]
        d = np.array([[(c - xy) @ along, (c - xy) @ across] for c in corners])
        if (
            d[:, 0].min() < reach_a
            and d[:, 0].max() > -reach_a
            and d[:, 1].min() < reach_w
            and d[:, 1].max() > -reach_w
        ):
            bad.append(i)
    return bad


# ----------------------------------------------------------------------------- depth


def _cameras(obs: Any) -> list[dict[str, Any]]:
    cams = obs.get("cameras") if isinstance(obs, dict) else None
    if isinstance(cams, dict):
        return [dict(v, name=k) for k, v in cams.items() if isinstance(v, dict)]
    return [c for c in (cams or []) if isinstance(c, dict)]


def _camera_to_world(pose: Any) -> np.ndarray:
    if isinstance(pose, dict) and "position" in pose:
        t = np.eye(4)
        t[:3, :3] = _matrix(pose.get("rotation") or pose.get("orientation"))
        t[:3, 3] = _xyz(pose)
        return t
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)


def camera_points(cam: dict[str, Any], corners: np.ndarray) -> np.ndarray:
    """World points (N x 3) of the depth pixels inside the image box of the world
    ``corners``; none when that box is not wholly inside the image."""
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


def judge_slot(
    cams: list[dict[str, Any]],
    xy: np.ndarray,
    facing_deg: float,
    floor_z: float,
    size: dict[str, float],
    wall_height: float,
) -> dict[str, Any]:
    """full / empty / jammed / unseen for one slot: the highest depth points inside its
    footprint widened by SEARCH_MARGIN. A full slot's top is a tile's height over the
    floor, within 8 mm of the slot; an empty one shows only the walls' tops; anything
    else (a tile leaning, half out, two in one slot) is jammed."""
    along, across = _axes(facing_deg)
    h = size["height"]
    half_a = size["thickness"] / 2.0 + SEARCH_MARGIN
    half_w = size["width"] / 2.0 + SEARCH_MARGIN
    corners = np.array(
        [
            [*(xy + sa * half_a * along + sw * half_w * across), floor_z + z]
            for sa in (-1, 1)
            for sw in (-1, 1)
            for z in (0.0, h + OCCLUDER_RISE)
        ]
    )
    best = {
        "state": "unseen",
        "camera": "",
        "top": -1.0,
        "offset": -1.0,
        "why": "no camera sees it",
    }
    for cam in cams:
        pts = camera_points(cam, corners)
        if not len(pts):
            continue
        rel = pts[:, :2] - xy
        a, w, z = rel @ along, rel @ across, pts[:, 2] - floor_z
        inside = (np.abs(a) <= half_a) & (np.abs(w) <= half_w)
        over = int(np.sum(z > h + OCCLUDER_RISE))
        keep = inside & (z > -0.005) & (z <= h + OCCLUDER_RISE)
        name = cam.get("name", "")
        if over >= OCCLUDED_SHARE * max(1, len(z)):
            best = {**best, "camera": name, "why": "occluded"}
            continue
        if int(keep.sum()) < MIN_POINTS:
            best = {**best, "camera": name, "why": "no depth"}
            continue
        top = float(np.percentile(z[keep], 95))
        if top <= wall_height + EMPTY_TOP:
            return {"state": "empty", "camera": name, "top": top, "offset": -1.0, "why": ""}
        band = keep & (z >= top - TOP_BAND)
        offset = math.hypot(float(np.mean(a[band])), float(np.mean(w[band])))
        if top >= h - HEIGHT_TOLERANCE and offset <= POSITION_TOLERANCE:
            return {"state": "full", "camera": name, "top": top, "offset": offset, "why": ""}
        why = (
            f"its top is {top * 1e3:.0f} mm over the floor (a tile's is {h * 1e3:.0f})"
            if top < h - HEIGHT_TOLERANCE
            else f"its top is {offset * 1e3:.0f} mm off the slot"
        )
        return {"state": "jammed", "camera": name, "top": top, "offset": offset, "why": why}
    return best


def order_cameras(cams: list[dict[str, Any]], camera_name: str) -> list[dict[str, Any]]:
    if camera_name:
        return [c for c in cams if c.get("name") == camera_name]
    rank = {n: i for i, n in enumerate(PREFERRED_CAMERAS)}
    return sorted(cams, key=lambda c: rank.get(str(c.get("name", "")), len(rank)))


def grasp_band(
    size: dict[str, float], wall_height: float, tip_reach: float, floor_over_table: float
) -> tuple[float, float]:
    """Tool-centre heights over the tile's bottom that close the fingertips over the walls
    (plus WALL_CLEARANCE), leave MIN_BITE of tile above them, stay inside GRASP_BAND and keep
    the tool centre MIN_TOOL_HEIGHT over the table; lo > hi when there is none."""
    lo = max(
        wall_height + WALL_CLEARANCE + tip_reach, GRASP_BAND[0], MIN_TOOL_HEIGHT - floor_over_table
    )
    hi = min(size["height"] - MIN_BITE + tip_reach, GRASP_BAND[1])
    return lo, hi


def _tile_xy_facing(pose: dict[str, Any]) -> tuple[np.ndarray, float]:
    """A standing tile's (xy, facing [deg]): an Se3Pose's x axis, or ``facing_deg``."""
    p = pose.get("position", pose)
    xy = _xyz({"position": p})[:2]
    if "facing_deg" in pose:
        return xy, float(pose["facing_deg"])
    r = _matrix(pose.get("rotation"))
    return xy, math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))


# ----------------------------------------------------------------------------- the skill


def run(
    ctx: NodeContext,
    table_z: float,
    rack: dict[str, Any],
    tile: dict[str, Any] | None = None,
    grasp_height: float | None = None,
    slot_index: int = -1,
    order: str = "index",
    tile_size: dict[str, float] | None = None,
    wall_height: float | None = None,
    offsets: list[float] | None = None,
    seat_press: float = SEAT_PRESS,
    grip_clearance: float = GRIP_CLEARANCE,
    release_clearance: float = RELEASE_CLEARANCE,
    transit_clearance: float = TRANSIT_CLEARANCE,
    read_slots: bool = True,
    verify: bool = True,
    camera_name: str = "",
    look_pose: dict[str, Any] | None = None,
    yaw_tolerance_deg: float = YAW_TOLERANCE_DEG,
    roll_limits: list[float] | None = None,
    reach_tolerance: float = REACH_TOLERANCE,
    descent_tolerance: float = DESCENT_TOLERANCE,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    out: Output = {
        "route": "invalid_input",
        "detail": "",
        "slot_index": -1,
        "picked": False,
        "released": False,
        "grasp_height": -1.0,
        "grip_width": -1.0,
        "jaw_error_deg": 0.0,
        "wrist_turn_deg": 0.0,
        "insert_offset_mm": 0.0,
        "align_error_mm": 0.0,
        "tile_tilt_deg": 0.0,
        "attempts": 0,
        "slot_states": [],
        "slot_state_after": "",
        "empty_count": 0,
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        return out

    try:
        size = {**TILE_SIZE, **{k: float(v) for k, v in (tile_size or {}).items()}}
        if not size["height"] > size["width"] > size["thickness"] > 0:
            raise ValueError(f"tile_size {size} is not height > width > thickness > 0")
        table_z = float(table_z)
        slots = rack_slots(rack or {})
        if not slots:
            raise ValueError("the rack has no slots")
        wall = float(
            wall_height if wall_height is not None else (rack or {}).get("wall_height", WALL_HEIGHT)
        )
        if wall < 0 or wall >= size["height"]:
            raise ValueError(f"wall_height {wall} is not between 0 and the tile's height")
        if order not in FILL_ORDERS:
            raise ValueError(f"order must be one of {FILL_ORDERS}, not {order!r}")
        if int(slot_index) >= len(slots):
            raise ValueError(f"slot_index {slot_index} is not one of the {len(slots)} slots")
        if tile is None and grasp_height is None:
            raise ValueError("with no tile to pick, grasp_height (the held tile's hang) is needed")
        offs = [float(v) for v in (offsets if offsets is not None else OFFSETS)] or [0.0]
        t_xy, t_facing = _tile_xy_facing(tile) if tile is not None else (None, 0.0)
    except (ValueError, KeyError, TypeError) as exc:
        return finish("invalid_input", str(exc))
    h, tw = size["height"], size["width"]

    try:
        hand = ctx.tool("robot.describe_gripper", **on_arm)
    except Exception as exc:  # the width checks and the reach fall back to defaults
        logger.info("refilling-a-domino-loading-rack: no gripper description (%s)", exc)
        hand = {}
    hand = hand if isinstance(hand, dict) else {}
    fit = hand.get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or hand.get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)
    finger = hand.get("finger") or {}
    tip_reach = float(finger.get("reach_m") or 0.0) if finger.get("stated", True) else 0.0
    tip_reach = tip_reach or TIP_REACH
    lo_roll, hi_roll = (float(v) for v in (roll_limits or ROLL_LIMITS))
    grip_open = tw + 2.0 * float(grip_clearance)
    release_open = tw + 2.0 * float(release_clearance)

    def move(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        try:
            return _refusal(
                ctx.tool(tool, pose={"position": _vec(position), "rotation": rot}, **on_arm)
            )
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc) or type(exc).__name__

    def here() -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **on_arm))

    def go(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        why = move(tool, position, rot)
        off = float(np.linalg.norm(_xyz(here()) - position))
        if (why and not why.startswith("timeout")) or off > float(reach_tolerance):
            return f"{why or 'ended off'}; the hand is {off * 1e3:.0f} mm from it"
        return ""

    def width(reply: Any = None) -> float:
        try:
            fraction = reply.get("position") if isinstance(reply, dict) else None
            if fraction is None:
                fraction = ctx.tool("robot.get_gripper", **on_arm)["position"]
            if at_open <= 0.0:
                return -1.0
            return at_closed + max(0.0, min(1.0, float(fraction))) * (at_open - at_closed)
        except Exception as exc:  # an unreadable hand is not an empty one
            logger.info("refilling-a-domino-loading-rack: could not read the gripper (%s)", exc)
            return -1.0

    def holding(w: float) -> str:
        if w < 0.0:
            return ""
        if w <= at_closed + EMPTY_MARGIN:
            return f"the jaws closed on nothing ({w * 1e3:.1f} mm)"
        if w < tw - NARROW_MARGIN:
            return f"the jaws closed to {w * 1e3:.1f} mm, under the tile's {tw * 1e3:.0f} mm: not across it"
        if w > tw + FULL_MARGIN:
            return (
                f"the jaws read {w * 1e3:.1f} mm, over the tile's {tw * 1e3:.0f} mm: not holding it"
            )
        return ""

    def set_width(w: float) -> str:
        try:
            return _refusal(ctx.tool("robot.set_grip", width_m=float(w), **on_arm))
        except Exception as exc:
            return str(exc) or type(exc).__name__

    def joints() -> np.ndarray:
        try:
            arms = ctx.tool("robot.get_observation").get("arms") or []
        except Exception as exc:
            logger.info("refilling-a-domino-loading-rack: no joint reading (%s)", exc)
            return np.zeros(0)
        if int(arm_id) >= len(arms):
            return np.zeros(0)
        q = arms[int(arm_id)].get("joint_state")
        if isinstance(q, dict):
            q = q.get("positions")
        return np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)

    def frames(facing_deg: float) -> tuple[list[dict[str, float]], np.ndarray]:
        heading = facing_deg + 90.0
        gf = ctx.tool("robot.grasp_frame", close_heading_deg=heading, **on_arm)
        rots = [
            gf[key]
            for key in ("rotation", "rotation_flipped")
            if isinstance(gf, dict) and gf.get(key)
        ]
        c = gf.get("close_axis") if isinstance(gf, dict) else None
        close = (
            np.array([float(c["x"]), float(c["y"]), 0.0])
            if c
            else np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading)), 0.0])
        )
        close = close / (np.linalg.norm(close) or 1.0)
        cur = here()["rotation"]
        rots.sort(key=lambda r: abs(jaw_error_deg(cur, r, close)))
        return rots, close

    def reachable(rots: list[dict[str, float]], point: np.ndarray) -> list[dict[str, float]]:
        if not check_reach:
            return rots
        ok = []
        for rot in rots:
            try:
                reply = ctx.tool(
                    "robot.solve_ik", pose={"position": _vec(point), "rotation": rot}, **on_arm
                )
            except Exception as exc:
                logger.info("refilling-a-domino-loading-rack: IK preview refused (%s)", exc)
                continue
            if isinstance(reply, dict) and (
                reply.get("success") is False or reply.get("within_servo_tolerance") is False
            ):
                continue
            ok.append(rot)
        return ok

    def square_jaws(
        target: dict[str, float],
        close: np.ndarray,
        at: np.ndarray,
        notes: list[str],
        step_deg: float = 180.0,
    ) -> dict[str, Any]:
        """Read the jaws' heading back and finish the turn on the last joint, at most
        ``step_deg`` a move (a held tile is turned in small steps)."""
        now = here()
        gain = -1.0
        for _ in range(TURN_TRIES + int(math.ceil(90.0 / max(step_deg, 1.0)))):
            err = _wrap_half(jaw_error_deg(now["rotation"], target, close))
            if abs(err) <= float(yaw_tolerance_deg):
                break
            q = joints()
            if q.size == 0:
                notes.append("no joint reading, the wrist was not turned")
                break
            options = [-err, -err - math.copysign(180.0, -err)]
            fits = [
                r
                for r in options
                if lo_roll + LIMIT_MARGIN
                <= q[-1] + math.radians(r) / gain
                <= hi_roll - LIMIT_MARGIN
            ]
            if not fits:
                notes.append(f"the last joint has no room to turn {-err:+.0f} deg")
                break
            turn = max(-step_deg, min(step_deg, fits[0]))
            command = math.radians(turn) / gain
            goal = q.copy()
            goal[-1] += command
            try:
                why = _refusal(
                    ctx.tool(
                        "robot.move_to_joints", joint_config={"positions": goal.tolist()}, **on_arm
                    )
                )
            except Exception as exc:
                why = str(exc) or type(exc).__name__
            if why:
                notes.append(f"wrist turn: {why}")
                break
            before = now["rotation"]
            now = here()
            moved = math.radians(_wrap_half(jaw_error_deg(now["rotation"], before, close)))
            out["wrist_turn_deg"] += math.degrees(moved)
            if abs(command) > math.radians(2.0) and 0.5 < abs(moved / command) < 1.5:
                gain = moved / command
            why = go("robot.go_to_pose", at, now["rotation"])
            if why:
                notes.append(f"re-centring after the wrist turn: {why}")
                break
            now = here()
        return now

    def read() -> list[dict[str, Any]]:
        try:
            cams = order_cameras(_cameras(ctx.tool("robot.get_observation")), camera_name)
        except Exception as exc:
            logger.info("refilling-a-domino-loading-rack: no observation (%s)", exc)
            cams = []
        return [judge_slot(cams, xy, f, fz, size, wall) for xy, f, fz in slots]

    def look() -> None:
        if look_pose is not None:
            why = go(
                "robot.go_to_pose", _xyz(look_pose), look_pose.get("rotation") or here()["rotation"]
            )
            if why:
                logger.info("refilling-a-domino-loading-rack: look pose not reached (%s)", why)

    # --- the hand: empty to pick, holding to place a held tile
    w0 = width()
    held = w0 >= 0.0 and tw - NARROW_MARGIN <= w0 <= tw + HELD_MARGIN
    if tile is not None and held:
        out["grip_width"] = w0
        return finish(
            "hand_full",
            f"the jaws read {w0 * 1e3:.1f} mm, a tile's width: place the held tile (tile None) or set it down",
        )
    if tile is None and w0 >= 0.0 and not held:
        out["grip_width"] = w0
        return finish(
            "not_holding",
            f"no tile to pick and the jaws read {w0 * 1e3:.1f} mm: nothing in the hand",
        )

    # --- the slots
    states = [{"state": "unseen", "why": "not read"} for _ in slots]
    if read_slots:
        states = read()
        if look_pose is not None and any(st["state"] == "unseen" for st in states):
            look()
            states = read()
    out["slot_states"] = [st["state"] for st in states]
    empty = [k for k, st in enumerate(states) if st["state"] == "empty"]
    out["empty_count"] = len(empty)
    if read_slots and all(st["state"] == "unseen" for st in states):
        return finish("not_seen", f"no camera saw the rack ({states[0].get('why', '')})")
    standing = [
        (slots[k][0], slots[k][1])
        for k, st in enumerate(states)
        if st["state"] in ("full", "jammed")
    ]
    standing_idx = [k for k, st in enumerate(states) if st["state"] in ("full", "jammed")]

    def clash(k: int) -> list[int]:
        return [
            standing_idx[i]
            for i in finger_conflicts(slots[k][0], slots[k][1], standing, size, release_open)
        ]

    if int(slot_index) >= 0:
        k = int(slot_index)
        if read_slots and states[k]["state"] != "empty":
            out["slot_index"] = k
            return finish("rack_full", f"slot {k} is {states[k]['state']}, not empty")
        candidates = [k]
    else:
        if order == "nearest":
            hxy = _xyz(here())[:2]
            candidates = sorted(empty, key=lambda j: float(np.linalg.norm(slots[j][0] - hxy)))
        else:
            candidates = sorted(empty, reverse=order == "reverse")
    if not candidates:
        unseen = [k for k, st in enumerate(states) if st["state"] == "unseen"]
        if unseen:
            return finish(
                "not_seen",
                f"no empty slot among those seen; slots {unseen} not seen (give look_pose)",
            )
        jammed = [j for j, st in enumerate(states) if st["state"] == "jammed"]
        return finish(
            "rack_full",
            "no empty slot" + (f"; jammed: {jammed} (clear them first)" if jammed else ""),
        )
    k = next((j for j in candidates if not clash(j)), None)
    if k is None:
        return finish(
            "unreachable",
            f"every empty slot has a standing neighbour where an open finger goes (slot {candidates[0]}: "
            f"{clash(candidates[0])}); the rack's pitch is under what the open hand needs",
        )
    xy, facing, floor_z = slots[k]
    out["slot_index"] = k

    lo, hi = grasp_band(size, wall, tip_reach, 0.0)
    if tile is not None:
        if lo > hi:
            return finish(
                "unreachable",
                f"no grasp height keeps the fingertips over the {wall * 1e3:.0f} mm walls",
            )
        gh = float(grasp_height) if grasp_height is not None else min(max(GRASP_HEIGHT, lo), hi)
        if gh - tip_reach > h - MIN_BITE or gh < MIN_TOOL_HEIGHT:
            return finish(
                "unreachable",
                f"grasp_height {gh * 1e3:.1f} mm is outside what the fingertips can do",
            )
    else:
        gh = float(grasp_height)
    tops = max([slots[j][2] for j in standing_idx] + [floor_z, table_z]) + h
    transit_z = tops + float(transit_clearance) + max(gh, tip_reach)
    notes: list[str] = []
    in_hand = np.zeros(3)
    axis_in_hand: np.ndarray | None = None

    # --- the pick (pick.py: MoveAbove, PreOpen, MoveToGrasp, Grasp, Lift)
    if tile is not None:
        grasp_point = np.array([*t_xy, table_z + gh])
        try:
            rots, close = frames(t_facing)
        except Exception as exc:
            return finish("unreachable", f"no grasp frame ({exc})")
        rots = reachable(rots, grasp_point)
        if not rots:
            return finish(
                "unreachable",
                "the grasp over the tile is out of reach (IK preview, both wrist solutions)",
            )
        why = set_width(grip_open)
        if why:
            return finish(
                "unreachable", f"the jaws did not open to {grip_open * 1e3:.0f} mm ({why})"
            )
        now = _xyz(here())
        if now[2] < transit_z - 1.0e-3:
            why = go(
                "robot.go_to_pose_cartesian",
                np.array([now[0], now[1], transit_z]),
                here()["rotation"],
            )
            if why:
                return finish("unreachable", f"rise before the pick: {why}")
        above = np.array([*t_xy, transit_z])
        why, rot = "", rots[0]
        for rot in rots:
            why = go("robot.go_to_pose", above, rot)
            if not why:
                break
        if why:
            return finish("unreachable", f"over the tile: {why}")
        pose = square_jaws(rot, close, above, notes)
        err = _wrap_half(jaw_error_deg(pose["rotation"], rot, close))
        if abs(err) > float(yaw_tolerance_deg):
            return finish(
                "unreachable", f"the jaws stayed {err:+.0f} deg off the tile's width axis"
            )
        rot = pose["rotation"]
        why = move("robot.go_to_pose_cartesian", grasp_point, rot)
        reached = _xyz(here())
        if float(reached[2] - grasp_point[2]) > float(descent_tolerance):
            go("robot.go_to_pose_cartesian", np.array([*reached[:2], transit_z]), rot)
            return finish(
                "not_holding",
                f"the descent to the tile stopped {(reached[2] - grasp_point[2]) * 1e3:.1f} mm high; nothing closed",
            )
        out["grip_width"] = width(ctx.tool("robot.close_gripper", **on_arm))
        why = holding(out["grip_width"])
        if why:
            set_width(grip_open)
            go("robot.go_to_pose_cartesian", np.array([*reached[:2], transit_z]), rot)
            return finish("not_holding", f"grasp: {why}; the hand is open over the tile")
        gh = float(reached[2] - table_z)
        # where the tile sits in the hand: its pose less where the hand closed, in the hand's frame
        r_grasp = _matrix(here()["rotation"])
        in_hand = r_grasp.T @ np.array([*(t_xy - reached[:2]), 0.0])
        axis_in_hand = r_grasp.T @ np.array([0.0, 0.0, 1.0])  # the standing tile's up axis
        out["picked"] = True
        why = go("robot.go_to_pose_cartesian", np.array([*reached[:2], transit_z]), rot)
        if why:
            return finish("unreachable", f"lift after the grasp: {why}; still holding")
    out["grasp_height"] = gh

    # --- over the slot, the jaws across the slot's width
    release_bottom = max(-float(seat_press), wall + WALL_CLEARANCE + tip_reach - gh)
    if release_bottom > 1e-9:
        notes.append(
            f"held {gh * 1e3:.1f} mm over its bottom: released {release_bottom * 1e3:.1f} mm over the floor so the fingertips clear the walls"
        )
    entry = np.array([*xy, floor_z + wall + ENTRY_CLEARANCE + gh])
    try:
        rots, close = frames(facing)
    except Exception as exc:
        return finish("unreachable", f"no grasp frame at the slot ({exc}); still holding")
    rots = reachable(rots, np.array([*xy, floor_z + release_bottom + gh]))
    if not rots:
        return finish(
            "unreachable", f"the release in slot {k} is out of reach (IK preview); still holding"
        )
    now = _xyz(here())
    if now[2] < transit_z - 1.0e-3:
        why = go(
            "robot.go_to_pose_cartesian", np.array([now[0], now[1], transit_z]), here()["rotation"]
        )
        if why:
            return finish("unreachable", f"rise before the transit: {why}; still holding")
    above = np.array([*xy, transit_z])
    why, rot = "", rots[0]
    for rot in rots:
        why = go("robot.go_to_pose", above, rot)
        if not why:
            break
    if why:
        return finish("unreachable", f"over slot {k}: {why}; still holding")
    pose = square_jaws(rot, close, above, notes, HELD_TURN_STEP_DEG)
    err = _wrap_half(jaw_error_deg(pose["rotation"], rot, close))
    out["jaw_error_deg"] = err
    if abs(err) > float(yaw_tolerance_deg):
        return finish(
            "unreachable",
            f"the jaws stayed {err:+.0f} deg off the slot's width axis; still holding",
        )
    rot = pose["rotation"]
    w = width()
    lost = holding(w)
    if lost:
        return finish("not_holding", f"over slot {k}: {lost} (dropped in transit?)")
    why = go("robot.go_to_pose_cartesian", entry, rot)
    if why:
        go("robot.go_to_pose_cartesian", np.array([*_xyz(here())[:2], transit_z]), rot)
        return finish(
            "unreachable", f"down to just over the walls: {why}; lifted back, still holding"
        )
    arrival = float(_xyz(here())[2] - entry[2])  # the hand's own offset on a free move
    # bring the held tile (not the hand) over the slot: its offset in the grip, then the
    # hand's own landing error, read back
    along = np.array([math.cos(math.radians(facing)), math.sin(math.radians(facing))])
    across = np.array([-along[1], along[0]])
    r_now = _matrix(here()["rotation"])
    tile_off = (r_now @ in_hand)[:2]
    if axis_in_hand is not None:
        # a tile gripped by a tilted hand hangs tilted under a level one: aim its bottom,
        # gh below the pads along its own axis, not its top
        up = r_now @ axis_in_hand
        tile_off = tile_off - gh * up[:2] / max(float(up[2]), 0.5)
        out["tile_tilt_deg"] = round(math.degrees(math.acos(min(1.0, abs(float(up[2]))))), 2)
    aim = entry.copy()
    aim[:2] = xy - tile_off
    for _ in range(ALIGN_TRIES):
        now = _xyz(here())
        err = now[:2] + tile_off - xy
        if (
            abs(float(err @ along)) <= ALIGN_TOLERANCE
            and abs(float(err @ across)) <= 4 * ALIGN_TOLERANCE
        ):
            break
        aim[:2] = aim[:2] - err
        move("robot.go_to_pose_cartesian", aim, rot)
    centre = _xyz(here())[:2] + tile_off
    out["align_error_mm"] = round(float((centre - xy) @ along) * 1e3, 2)

    # --- into the slot, along-row offsets tried when the descent stops on a wall
    release_z = floor_z + release_bottom + gh
    base = _xyz(here())[:2]
    seated = False
    for i, off in enumerate(offs):
        out["attempts"] = i + 1
        at = np.array([*(base + along * off), entry[2]])
        if i:
            why = go("robot.go_to_pose_cartesian", at, rot)
            if why:
                notes.append(f"offset {off * 1e3:+.0f} mm: {why}")
                continue
        move("robot.go_to_pose_cartesian", np.array([*at[:2], release_z]), rot)
        high = float(_xyz(here())[2] - release_z) - max(0.0, arrival)
        if high <= INSERT_TOLERANCE:
            seated = True
            out["insert_offset_mm"] = round(off * 1e3, 1)
            break
        notes.append(f"offset {off * 1e3:+.0f} mm stopped {high * 1e3:.1f} mm high")
        move("robot.go_to_pose_cartesian", at, rot)
    if not seated:
        go("robot.go_to_pose_cartesian", np.array([*_xyz(here())[:2], transit_z]), rot)
        return finish(
            "jammed",
            f"the tile would not go into slot {k} at {len(offs)} offset(s) along the row ({'; '.join(notes)}); "
            "lifted back, still holding",
        )

    # --- release (place.py: OpenFingers a step, Retreat holding that width)
    why = set_width(release_open)
    if why:
        try:
            ctx.tool("robot.open_gripper", **on_arm)
            notes.append(f"opened fully ({why})")
        except Exception as exc:
            return finish(
                "unreachable", f"the jaws did not open in slot {k} ({exc}); still holding"
            )
    out["released"] = True
    why = go("robot.go_to_pose_cartesian", np.array([*_xyz(here())[:2], transit_z]), rot)
    if why:
        return finish("unreachable", f"released, but the vertical retreat: {why}")

    # --- the slot again from depth
    tail = f" ({'; '.join(notes)})" if notes else ""
    if not verify:
        return finish(
            "refilled", f"slot {k}: released after {out['attempts']} descent(s), not checked{tail}"
        )
    look()
    after = judge_slot(
        order_cameras(_cameras(ctx.tool("robot.get_observation")), camera_name),
        xy,
        facing,
        floor_z,
        size,
        wall,
    )
    if after["state"] == "full" and after.get("top", h) < h - SEATED_TOP:
        after = {
            **after,
            "state": "jammed",
            "why": f"its top is {after['top'] * 1e3:.0f} mm over the floor, not seated",
        }
    out["slot_state_after"] = after["state"]
    if after["state"] == "full":
        out["empty_count"] = max(0, len(empty) - 1)
        return finish(
            "refilled",
            f"slot {k} reads full after the release ({out['attempts']} descent(s)){tail}",
        )
    if after["state"] == "unseen":
        return finish(
            "unverified",
            f"slot {k}: released, but no camera saw it after ({after.get('why', '')}){tail}",
        )
    return finish(
        "fell", f"slot {k} reads {after['state']} after the release ({after.get('why', '')}){tail}"
    )
