# Adapted from predicators (https://github.com/BasisResearch/predicators) at
# afb6f2918091f5ca372a75b9a335aea317a8a209, MIT, Copyright (c) 2022 Rohan Chitnis
# and Tom Silver (license: ../LICENSE.predicators):
# predicators/ground_truth_models/domino/options.py
# (_pick_param_defs, _create_sf_pick, _create_sf_place),
# predicators/ground_truth_models/skill_factories/pick.py (create_pick_skill:
# MoveAbove, PreOpen to the object's width plus slack, straight MoveToGrasp,
# Grasp, LiftSlightly) and skill_factories/place.py (create_place_skill with
# use_move_above and release_until_ungrasped: MoveAbove, straight Descend to the
# release height, OpenFingers only a step, ClearFingers a few mm more, a
# straight Retreat holding that width), with the upright / fallen thresholds of
# predicators/envs/pybullet_domino/components/domino_component.py.
# Changes: the domino's simulator state is a perceived standing tile (x, y,
# facing) and a list of slots and neighbours, all world frame; the jaws close
# across the tile's width so the fingers stand beside the row and never between
# tiles (upstream closes the Panda across whatever the planned yaw gives);
# BiRRT's collision-checked approach is a transit at a height that clears every
# standing tile with the held one, then a lead-in along the row from its free
# end and a straight vertical descent; the grasp height is a tool-centre height
# over the tile's bottom whose band is re-derived for the YAM's linear_4310
# fingertips (upstream: a z offset over the Panda's tile origin, 0.045-0.080 m
# on a 0.15 m tile); the simulator's grasp-constraint release (open until
# is_held flips) is a jaw width of the tile plus a clearance each side, held
# through the vertical retreat; holding is read from the jaw gap
# (robot.close_gripper / robot.get_gripper) instead of is_held; the jaw heading
# is read back and finished on the last joint (robot.move_to_joints) when the
# arm's solver leaves the wrist roll free; and the Upright / Toppled predicates
# read off simulator roll become a depth check of the slot and of every
# neighbour from the observation's RGB-D cameras.
"""Pick a standing domino top-down and set it upright at a slot beside its neighbours, then check from depth."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np

try:
    from gap import NodeContext
except ImportError:  # annotation only; lets the domino task module import this without gap
    NodeContext = Any

logger = logging.getLogger(__name__)

TILE_SIZE = {"height": 0.048, "width": 0.024, "thickness": 0.0075}
"""A standard domino standing on its end [m]: 48 x 24 x 7.5 mm (Franka-Domino-Project's
and zmatt.net's tiles)."""

GRASP_HEIGHT = 0.030
"""Tool-centre height over the tile's bottom at the close [m]. The linear_4310's
fingertips reach 10.2 mm under the tool centre, so they meet the tile at 19.8 mm
and the pads cover its middle. Inside the band measured in a YAM station twin
(GRASP_BAND); upstream's Panda band is 0.045-0.080 m over the centre of a 0.15 m tile."""

GRASP_BAND = (0.015, 0.046)
"""Commanded tool-centre heights over the tile's bottom that grasped, carried and
stood a 48 mm tile in a YAM station twin (15, 22, 30, 38, 46 mm: all placed; 53 mm
dropped it in transit) [m]. The floor is a real YAM station's: it refuses a tool
centre under 14 mm over its table; the ceiling is MIN_BITE. Sample inside it."""

TIP_REACH = 0.0102
"""How far the closed linear_4310 fingertips reach under the tool centre [m] when
robot.describe_gripper does not say."""

MIN_TOOL_HEIGHT = 0.015
"""Lowest tool-centre height over the table [m]: a real YAM station refuses under 14 mm."""

MIN_BITE = 0.010
"""Least height of tile between the fingertips and its top [m], for the commanded
grasp. In a YAM station twin the hand arrived 2.2 mm over every commanded grasp;
a 9.9 mm bite (commanded 46 mm) carried the tile, a 2.9 mm bite (commanded 53 mm)
closed on it and lost it in transit (upstream: the Panda's fingers 'close above
the domino without grasping it' past their band)."""

GRIP_CLEARANCE = 0.004
"""Jaw clearance each side of the tile when opening for the grasp [m]: upstream's
PreOpen slack keeps a narrow opening so the pads do not reach a neighbour; 4 mm a
side is robot.describe_gripper's clearance on the YAM station twin."""

RELEASE_CLEARANCE = 0.004
"""Jaw opening each side of the tile at the release, held through the retreat [m]:
upstream's _RELEASE_CLEAR_SLACK (8 mm in all, ~4 mm a side)."""

RELEASE_GAP = 0.0015
"""Height of the tile's bottom over the table when the jaws open [m]. Upstream drops
from a sampled release_z; a millimetre and a half is under the twin's 4 mm servo
bound and does not topple a 48 mm tile."""

TRANSIT_CLEARANCE = 0.02
"""Height the held tile's bottom (or the open fingertips) keeps over the tops of the
standing tiles in transit [m]. Upstream plans a 15 mm held-bystander clearance with
BiRRT because 'a graze topples a standing domino'."""

LEAD_IN = 0.03
"""Length of the last horizontal leg, along the row from its free end, at transit
height [m]. A choice: the hand arrives over the slot moving along the row, so the
fingers beside the tile never cross over a neighbour."""

FINGER_HALF_DEPTH = 0.012
"""Half the open finger's extent along the row [m], for the neighbour clearance
check (a linear_4310 blade, about 24 mm wide; not measured)."""

FINGER_BLADE = 0.030
"""One linear_4310 blade's depth along its closing direction [m] (the twin's trimmed
30 mm blade, JAW_THICKNESS 29.9 mm)."""

NEIGHBOUR_CLEARANCE = 0.004
"""Least gap between an open finger and a standing neighbour [m]."""

EMPTY_MARGIN = 0.003
"""A jaw gap within this of the closed stop holds nothing [m] (grasping-flat-objects)."""

NARROW_MARGIN = 0.006
"""A jaw gap this far under the tile's width did not close across it [m]
(grasping-flat-objects)."""

FULL_MARGIN = 0.008
"""A jaw gap this far over the tile's width closed on something else [m]."""

DESCENT_TOLERANCE = 0.006
"""A straight descent that ends this far over its target stopped on something [m]:
over a servo's 4 mm arrival bound (grasping-flat-objects)."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from a free move's target [m]: no tighter than a real
YAM station's 12 mm per-move bound."""

YAW_TOLERANCE_DEG = 4.0
"""How far the jaws may end from the tile's width axis before the rest is turned on
the last joint [deg]: a servo's 3 deg arrival bound plus a degree."""

TURN_TRIES = 3
"""Last-joint turns tried, each read back (the first learns which way it turns)."""

ROLL_LIMITS = (-2.0944, 2.0944)
"""The last joint's range [rad] when not stated: the i2rt YAM's joint 6 (+-120 deg)."""

LIMIT_MARGIN = 0.05
"""Kept clear of each roll limit [rad]."""

HEIGHT_TOLERANCE = 0.010
"""A standing tile's top reads at least its height less this [m]. A tile leaning
on a neighbour at 30 deg reads 7 mm lower at most; one lying reads 7.5 or 24 mm."""

POSITION_TOLERANCE = 0.008
"""Largest offset of a standing tile's top from its slot [m]: a tile leaning 10 deg
(upstream's fallen_threshold) moves its top 8.3 mm; a calibrated YAM station's
camera RMS is 5.4 mm."""

SEARCH_MARGIN = 0.006
"""How far beyond the tile's footprint the depth check looks for its top [m]."""

LEAN_SEARCH = 0.020
"""How far along the row the depth check follows a leaning tile's top [m]."""

OCCLUDER_RISE = 0.015
"""Points this far over a tile's top inside its footprint belong to something over
it (the hand) [m]."""

OCCLUDED_SHARE = 0.25
"""A camera whose window over a tile is this much something higher than a standing
tile does not judge it (a choice)."""

TOP_BAND = 0.006
"""Points within this of the highest one form the tile's top [m]."""

MIN_POINTS = 4
"""Fewest depth points on a tile's top to judge it (a 24 x 7.5 mm top is about
12 x 4 px from a 0.9 m overhead camera)."""

DEPTH_RANGE = (0.07, 3.0)
"""Valid depth [m]: a D405's 7 cm minimum, an overhead camera's far bound."""

PREFERRED_CAMERAS = ("overhead", "top")
"""Camera names tried first when no camera_name is given: the station's calibrated
overhead (RMS 5.4 mm). A wrist camera's pose follows the hand's calibration; in a YAM
station twin its view of a parked-away row read tiles 7-12 mm off their true spot."""


class Output(TypedDict):
    route: str
    detail: str
    released: bool
    picked: bool
    grasp_height: float
    grip_width: float
    held_width: float
    jaw_error_deg: float
    wrist_turn_deg: float
    release_position: dict[str, float]
    verified: bool
    tile_top: float
    placement_offset: float
    neighbours_standing: int
    neighbours_knocked: list[int]


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


def tile_pose(pose: dict[str, Any]) -> tuple[np.ndarray, float]:
    """A standing tile's or a slot's (xy, facing [deg]). The facing is the heading
    of its broad faces' normal - the way it topples, the way the row runs: an
    Se3Pose's x axis, or ``facing_deg`` beside a ``position`` (or ``center``)."""
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
    """The row's direction (along the facing) and the tile's width axis, world xy."""
    a = math.radians(facing_deg)
    return np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])


def finger_conflicts(
    slot_xy: np.ndarray,
    facing_deg: float,
    neighbours: list[tuple[np.ndarray, float]],
    size: dict[str, float],
    open_width: float,
) -> list[int]:
    """Neighbours whose footprint comes within NEIGHBOUR_CLEARANCE of the open hand
    over the slot: both fingers beside the tile, across its width, and the tile
    between them."""
    along, across = _axes(facing_deg)
    reach_a = max(FINGER_HALF_DEPTH, size["thickness"] / 2.0) + NEIGHBOUR_CLEARANCE
    reach_w = open_width / 2.0 + FINGER_BLADE + NEIGHBOUR_CLEARANCE
    bad = []
    for i, (xy, n_facing) in enumerate(neighbours):
        n_along, n_across = _axes(n_facing)
        corners = [
            xy + sa * size["thickness"] / 2.0 * n_along + sw * size["width"] / 2.0 * n_across
            for sa in (-1, 1)
            for sw in (-1, 1)
        ]
        d = np.array([[(c - slot_xy) @ along, (c - slot_xy) @ across] for c in corners])
        if (
            d[:, 0].min() < reach_a
            and d[:, 0].max() > -reach_a
            and d[:, 1].min() < reach_w
            and d[:, 1].max() > -reach_w
        ):
            bad.append(i)
    return bad


# ----------------------------------------------------------------------------- depth check


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
    """World points (N x 3) of the camera's depth pixels inside the image box of the
    world ``corners`` (OpenCV pinhole: x right, y down, z forward); none when that
    box is not wholly inside the image (a partial view shifts a tile's top)."""
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
    along_room: float = LEAN_SEARCH,
) -> dict[str, Any]:
    """Whether a tile stands at (xy, facing): the highest points inside its footprint
    widened by SEARCH_MARGIN (and by ``along_room`` along the row, to follow a lean)
    are its top. Judged by the first camera in ``cams`` that sees the whole footprint
    unoccluded; when none does, the best of the rest says why."""
    along, across = _axes(facing_deg)
    h = size["height"]
    half_a = size["thickness"] / 2.0 + SEARCH_MARGIN + max(0.0, along_room)
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
        "camera": "",
        "why": "no camera sees it",
    }
    for cam in cams:
        pts = camera_points(cam, corners)
        if not len(pts):
            continue
        rel = pts[:, :2] - xy
        a, w, z = rel @ along, rel @ across, pts[:, 2] - table_z
        inside = (np.abs(a) <= half_a) & (np.abs(w) <= half_w)
        # The window is the footprint's image, so anything in it higher than a
        # standing tile stands between the camera and the footprint.
        over = int(np.sum(z > h + OCCLUDER_RISE))
        keep = inside & (z > -0.005) & (z <= h + OCCLUDER_RISE)
        raised = keep & (z > 0.004)
        n = int(raised.sum())
        if over >= OCCLUDED_SHARE * len(z):
            verdict = {
                "seen": False,
                "standing": False,
                "points": n,
                "camera": cam.get("name", ""),
                "why": "occluded",
            }
        elif int(keep.sum()) < MIN_POINTS:
            verdict = {
                "seen": False,
                "standing": False,
                "points": n,
                "camera": cam.get("name", ""),
                "why": "no depth",
            }
        elif n < MIN_POINTS:
            verdict = {
                "seen": True,
                "standing": False,
                "points": int(keep.sum()),
                "camera": cam.get("name", ""),
                "top": float(np.max(z[keep])),
                "offset": float("nan"),
                "why": "nothing stands there",
            }
        else:
            top = float(np.percentile(z[raised], 95))
            band = raised & (z >= top - TOP_BAND)
            off_a, off_w = float(np.mean(a[band])), float(np.mean(w[band]))
            offset = math.hypot(off_a, off_w)
            standing = top >= h - HEIGHT_TOLERANCE and offset <= POSITION_TOLERANCE
            why = (
                ""
                if standing
                else (
                    f"its top is {top * 1e3:.0f} mm over the table (a standing tile's is {h * 1e3:.0f})"
                    if top < h - HEIGHT_TOLERANCE
                    else f"its top is {offset * 1e3:.0f} mm off the slot (leaning or misplaced)"
                )
            )
            verdict = {
                "seen": True,
                "standing": bool(standing),
                "points": n,
                "camera": cam.get("name", ""),
                "top": top,
                "offset": offset,
                "offset_along": off_a,
                "offset_across": off_w,
                "why": why,
            }
        if verdict["seen"]:
            return verdict
        if verdict["points"] >= best["points"]:
            best = verdict
    return best


def order_cameras(cams: list[dict[str, Any]], camera_name: str) -> list[dict[str, Any]]:
    """The cameras to judge with, in order: only ``camera_name`` when given, else the
    fixed overhead first (PREFERRED_CAMERAS), then the rest as the observation lists them."""
    if camera_name:
        return [c for c in cams if c.get("name") == camera_name]
    rank = {n: i for i, n in enumerate(PREFERRED_CAMERAS)}
    return sorted(cams, key=lambda c: rank.get(str(c.get("name", "")), len(rank)))


# ----------------------------------------------------------------------------- the skill


def run(
    ctx: NodeContext,
    table_z: float,
    slot: dict[str, Any] | None = None,
    tile: dict[str, Any] | None = None,
    slots: list[dict[str, Any]] | None = None,
    slot_index: int = 0,
    neighbours: list[dict[str, Any]] | None = None,
    tile_size: dict[str, float] | None = None,
    grasp_height: float = GRASP_HEIGHT,
    release_gap: float = RELEASE_GAP,
    grip_clearance: float = GRIP_CLEARANCE,
    release_clearance: float = RELEASE_CLEARANCE,
    transit_clearance: float = TRANSIT_CLEARANCE,
    lead_in: float = LEAD_IN,
    verify: bool = True,
    camera_name: str = "",
    park_pose: dict[str, Any] | None = None,
    yaw_tolerance_deg: float = YAW_TOLERANCE_DEG,
    roll_limits: list[float] | None = None,
    reach_tolerance: float = REACH_TOLERANCE,
    descent_tolerance: float = DESCENT_TOLERANCE,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    size = {**TILE_SIZE, **{k: float(v) for k, v in (tile_size or {}).items()}}
    h, tw = size["height"], size["width"]
    table_z = float(table_z)

    out: Output = {
        "route": "unreachable",
        "detail": "",
        "released": False,
        "picked": False,
        "grasp_height": float(grasp_height),
        "grip_width": -1.0,
        "held_width": -1.0,
        "jaw_error_deg": 0.0,
        "wrist_turn_deg": 0.0,
        "release_position": {"x": 0.0, "y": 0.0, "z": table_z},
        "verified": False,
        "tile_top": -1.0,
        "placement_offset": -1.0,
        "neighbours_standing": 0,
        "neighbours_knocked": [],
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        return out

    if slot is None:
        if not slots or not 0 <= int(slot_index) < len(slots):
            return finish(
                "unreachable",
                f"no slot: slot_index {slot_index} is not one of the {len(slots or [])} slots",
            )
        slot = slots[int(slot_index)]
        if neighbours is None:
            neighbours = list(slots[: int(slot_index)])
    slot_xy, slot_facing = tile_pose(slot)
    neighbours = list(neighbours or [])
    nbrs = [tile_pose(n) for n in neighbours]
    out["release_position"] = _vec(np.array([*slot_xy, table_z]))

    try:
        hand = ctx.tool("robot.describe_gripper", **on_arm)
    except Exception as exc:  # the width checks and the reach fall back to defaults
        logger.info("placing-an-upright-domino: no gripper description (%s)", exc)
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
    if at_open > 0.0 and max(grip_open, release_open) > at_open:
        return finish(
            "unreachable",
            f"a {tw * 1e3:.0f} mm tile does not fit the {at_open * 1e3:.0f} mm open hand",
        )
    clash = finger_conflicts(slot_xy, slot_facing, nbrs, size, max(grip_open, release_open))
    if clash:
        return finish(
            "unreachable",
            f"neighbour(s) {clash} stand where an open finger goes over the slot: "
            "the slot is too close to them for this hand",
        )

    if tile is not None:
        gh = float(grasp_height)
        if gh < MIN_TOOL_HEIGHT or gh - tip_reach > h - MIN_BITE:
            return finish(
                "unreachable",
                f"grasp_height {gh * 1e3:.0f} mm is outside what the fingertips can do on a "
                f"{h * 1e3:.0f} mm tile ({MIN_TOOL_HEIGHT * 1e3:.0f}-{(h - MIN_BITE + tip_reach) * 1e3:.0f} mm)",
            )
    # Transit: the held tile's bottom (hanging grasp_height under the tool centre)
    # or the open fingertips clear every standing tile's top.
    hang = float(grasp_height)
    transit_z = table_z + h + float(transit_clearance) + max(hang, tip_reach)

    start = _pose(ctx.tool("robot.get_ee_pose", **on_arm))
    start_xyz = _xyz(start)

    def move(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        try:
            reply = ctx.tool(tool, pose={"position": _vec(position), "rotation": rot}, **on_arm)
            return _refusal(reply)
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc) or type(exc).__name__

    def here() -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **on_arm))

    def go(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        """A move judged by where the hand is: "" when it arrived within
        reach_tolerance (a servo timeout that ended that close has arrived)."""
        why = move(tool, position, rot)
        off = float(np.linalg.norm(_xyz(here()) - position))
        if (why and not why.startswith("timeout")) or off > float(reach_tolerance):
            return f"{why or 'ended off'}; the hand is {off * 1e3:.0f} mm from it"
        return ""

    def width(reply: Any = None) -> float:
        """The jaw gap [m] (robot.get_gripper's 1.0 = open), or -1.0 when unreadable."""
        try:
            fraction = reply.get("position") if isinstance(reply, dict) else None
            if fraction is None:
                fraction = ctx.tool("robot.get_gripper", **on_arm)["position"]
            if at_open <= 0.0:
                return -1.0
            return at_closed + max(0.0, min(1.0, float(fraction))) * (at_open - at_closed)
        except Exception as exc:  # an unreadable hand is not an empty one
            logger.info("placing-an-upright-domino: could not read the gripper (%s)", exc)
            return -1.0

    def holding(w: float) -> str:
        """ "" when the gap reads the tile's width between the pads (or cannot be read)."""
        if w < 0.0:
            return ""
        if w <= at_closed + EMPTY_MARGIN:
            return f"the jaws closed on nothing ({w * 1e3:.1f} mm)"
        if w < tw - NARROW_MARGIN:
            return f"the jaws closed to {w * 1e3:.1f} mm, under the tile's {tw * 1e3:.0f} mm: not across it"
        if w > tw + FULL_MARGIN:
            return (
                f"the jaws hold something {w * 1e3:.1f} mm wide, over the tile's {tw * 1e3:.0f} mm"
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
            logger.info("placing-an-upright-domino: no joint reading (%s)", exc)
            return np.zeros(0)
        if int(arm_id) >= len(arms):
            return np.zeros(0)
        q = arms[int(arm_id)].get("joint_state")
        if isinstance(q, dict):
            q = q.get("positions")
        return np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)

    def frames(facing_deg: float) -> tuple[list[dict[str, float]], np.ndarray]:
        """Both top-down wrist solutions with the jaws closing across the tile's width."""
        heading = facing_deg + 90.0
        gf = ctx.tool("robot.grasp_frame", close_heading_deg=heading, **on_arm)
        rots = [
            gf[k] for k in ("rotation", "rotation_flipped") if isinstance(gf, dict) and gf.get(k)
        ]
        c = gf.get("close_axis") if isinstance(gf, dict) else None
        close = (
            np.array([float(c["x"]), float(c["y"]), 0.0])
            if c
            else np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading)), 0.0])
        )
        return rots, close / (np.linalg.norm(close) or 1.0)

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
                logger.info("placing-an-upright-domino: IK preview refused (%s)", exc)
                continue
            # A stall at the edge of reach reads like contact.
            if isinstance(reply, dict) and (
                reply.get("success") is False or reply.get("within_servo_tolerance") is False
            ):
                continue
            ok.append(rot)
        return ok

    def square_jaws(
        target: dict[str, float], close: np.ndarray, at: np.ndarray, notes: list[str]
    ) -> dict[str, Any]:
        """Read the jaws' heading back and, when the solver left the wrist roll free,
        finish the turn on the last joint and re-centre over ``at``."""
        now = here()
        gain = -1.0  # a top-down hand turns opposite to its last joint, until measured
        for _ in range(TURN_TRIES):
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
            command = math.radians(fits[0]) / gain
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

    def ordered(rots: list[dict[str, float]], close: np.ndarray) -> list[dict[str, float]]:
        """The wrist solutions, the one nearer the hand's present jaw heading first."""
        cur = here()["rotation"]
        return sorted(rots, key=lambda r: abs(jaw_error_deg(cur, r, close)))

    def rise(rot: dict[str, float]) -> str:
        now = _xyz(here())
        if now[2] < transit_z - 1.0e-3:
            return go("robot.go_to_pose_cartesian", np.array([now[0], now[1], transit_z]), rot)
        return ""

    # Neighbours as they stand before anything moves, so a tile that was already
    # down is not blamed on this placement.
    cams_before: list[dict[str, Any]] = []
    if verify and nbrs:
        try:
            cams_before = order_cameras(_cameras(ctx.tool("robot.get_observation")), camera_name)
        except Exception as exc:
            logger.info("placing-an-upright-domino: no observation before the place (%s)", exc)
    before = (
        [judge_tile(cams_before, xy, f, table_z, size, 0.0) for xy, f in nbrs]
        if cams_before
        else []
    )

    # --- the pick (pick.py: MoveAbove, PreOpen, MoveToGrasp, Grasp, LiftSlightly)
    notes: list[str] = []
    if tile is not None:
        t_xy, t_facing = tile_pose(tile)
        grasp_point = np.array([*t_xy, table_z + float(grasp_height)])
        try:
            rots, close = frames(t_facing)
        except Exception as exc:
            return finish("unreachable", f"no grasp frame ({exc})")
        rots = reachable(ordered(rots, close), grasp_point)
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
        why = rise(here()["rotation"])
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
        out["jaw_error_deg"] = err
        if abs(err) > float(yaw_tolerance_deg):
            return finish(
                "unreachable",
                f"the jaws stayed {err:+.0f} deg off the tile's width axis"
                + (f" ({'; '.join(notes)})" if notes else ""),
            )
        why = move("robot.go_to_pose_cartesian", grasp_point, pose["rotation"])
        reached = _xyz(here())
        high = float(reached[2] - grasp_point[2])
        if high > float(descent_tolerance) or float(
            np.linalg.norm(reached[:2] - grasp_point[:2])
        ) > float(reach_tolerance):
            go("robot.go_to_pose_cartesian", np.array([*reached[:2], transit_z]), pose["rotation"])
            return finish(
                "not_holding",
                f"the descent to the tile stopped {high * 1e3:.1f} mm high ({why or 'stalled'}): "
                "a fingertip on its top edge, or the tile is not where it was seen; nothing closed",
            )
        out["grip_width"] = width(ctx.tool("robot.close_gripper", **on_arm))
        why = holding(out["grip_width"])
        if why:
            set_width(grip_open)
            go("robot.go_to_pose_cartesian", np.array([*reached[:2], transit_z]), pose["rotation"])
            return finish("not_holding", f"grasp: {why}; the hand is open over the tile")
        hang = float(reached[2] - table_z)
        out["grasp_height"] = hang
        out["picked"] = True
        why = go(
            "robot.go_to_pose_cartesian", np.array([*reached[:2], transit_z]), pose["rotation"]
        )
        if why:
            return finish("unreachable", f"lift after the grasp: {why}; still holding")
    else:
        out["grip_width"] = width()
        why = holding(out["grip_width"])
        if why:
            return finish("not_holding", f"nothing in the hand to place: {why}")

    # --- the place (place.py with use_move_above: MoveAbove, Descend, OpenFingers,
    # ClearFingers, Retreat), the last leg along the row from its free end.
    release = np.array([*slot_xy, table_z + hang + float(release_gap)])
    out["release_position"] = _vec(np.array([*slot_xy, table_z + float(release_gap)]))
    if release[2] < table_z + MIN_TOOL_HEIGHT:
        return finish(
            "unreachable",
            f"the release would put the tool centre {hang * 1e3:.0f} mm over the table",
        )
    along, _ = _axes(slot_facing)
    if nbrs:
        nearest = min(nbrs, key=lambda n: float(np.linalg.norm(n[0] - slot_xy)))
        side = -1.0 if float((nearest[0] - slot_xy) @ along) > 0.0 else 1.0
    else:
        side = 1.0 if float((start_xyz[:2] - slot_xy) @ along) >= 0.0 else -1.0
    lead = np.array([*(slot_xy + side * float(lead_in) * along), transit_z])
    over = np.array([*slot_xy, transit_z])
    try:
        rots, close = frames(slot_facing)
    except Exception as exc:
        return finish("unreachable", f"no grasp frame at the slot ({exc}); still holding")
    rots = reachable(ordered(rots, close), release)
    if not rots:
        return finish(
            "unreachable", "the release over the slot is out of reach (IK preview); still holding"
        )
    why = rise(here()["rotation"])
    if why:
        return finish("unreachable", f"rise before the transit: {why}; still holding")
    why, rot = "", rots[0]
    for rot in rots:
        why = go("robot.go_to_pose", lead, rot)
        if not why:
            break
    if why:
        return finish("unreachable", f"transit to the lead-in: {why}; still holding")
    notes = []
    pose = square_jaws(rot, close, lead, notes)
    err = _wrap_half(jaw_error_deg(pose["rotation"], rot, close))
    out["jaw_error_deg"] = err
    if abs(err) > float(yaw_tolerance_deg):
        return finish(
            "unreachable",
            f"the jaws stayed {err:+.0f} deg off the slot's facing"
            + (f" ({'; '.join(notes)})" if notes else "")
            + "; still holding",
        )
    rot = pose["rotation"]
    why = go("robot.go_to_pose_cartesian", over, rot)
    if why:
        return finish("unreachable", f"along the row to over the slot: {why}; still holding")
    out["held_width"] = width()
    why = holding(out["held_width"])
    if why:
        return finish("not_holding", f"over the slot: {why} (dropped in transit?)")
    why = move("robot.go_to_pose_cartesian", release, rot)
    reached = _xyz(here())
    high = float(reached[2] - release[2])
    if high > float(descent_tolerance):
        go("robot.go_to_pose_cartesian", np.array([*reached[:2], transit_z]), rot)
        return finish(
            "unreachable",
            f"the descent to the slot stopped {high * 1e3:.1f} mm high ({why or 'stalled'}): "
            "something under the held tile; lifted back, still holding",
        )
    out["release_position"] = _vec(np.array([*reached[:2], reached[2] - hang]))
    why = set_width(release_open)
    if why:
        try:
            ctx.tool("robot.open_gripper", **on_arm)
            notes.append(f"opened fully ({why})")
        except Exception as exc:
            return finish(
                "unreachable", f"the jaws did not open at the slot ({exc}); still holding"
            )
    out["released"] = True
    why = go("robot.go_to_pose_cartesian", np.array([*reached[:2], transit_z]), rot)
    if why:
        return finish("unreachable", f"released, but the vertical retreat: {why}")

    hand_note = (
        f"released {np.linalg.norm(reached[:2] - slot_xy) * 1e3:.1f} mm from the slot, "
        f"jaws {out['jaw_error_deg']:+.1f} deg off its facing"
    ) + (f"; {'; '.join(notes)}" if notes else "")
    if not verify:
        return finish("placed", hand_note + "; not checked from depth")

    # --- the check: out of the cameras' way, then the slot and every neighbour.
    park = (
        _xyz(_pose(park_pose))
        if park_pose
        else np.array([*start_xyz[:2], max(start_xyz[2], transit_z)])
    )
    why = go("robot.go_to_pose", park, rot)
    if why:
        notes.append(f"park: {why}")
    try:
        cams = order_cameras(_cameras(ctx.tool("robot.get_observation")), camera_name)
    except Exception as exc:
        return finish("unverified", f"{hand_note}; no observation for the check ({exc})")
    # Follow a lean along the row, but stop short of the nearest neighbour's face.
    room = LEAN_SEARCH
    if nbrs:
        nearest_d = min(float(np.linalg.norm(n[0] - slot_xy)) for n in nbrs)
        room = max(0.0, min(LEAN_SEARCH, nearest_d - size["thickness"] - 2.0 * SEARCH_MARGIN))
    placed = judge_tile(cams, slot_xy, slot_facing, table_z, size, room)
    knocked: list[int] = []
    standing = 0
    unseen: list[int] = []
    for i, (xy, f) in enumerate(nbrs):
        v = judge_tile(cams, xy, f, table_z, size, 0.0)
        was = before[i] if i < len(before) else {"seen": False, "standing": True}
        if not v["seen"]:
            unseen.append(i)
        elif v["standing"]:
            standing += 1
        elif was["standing"] or not was["seen"]:
            knocked.append(i)
    out["neighbours_standing"] = standing
    out["neighbours_knocked"] = knocked
    if placed["seen"]:
        out["verified"] = True
        out["tile_top"] = float(placed.get("top", -1.0))
        offset = float(placed.get("offset", -1.0))
        out["placement_offset"] = offset if math.isfinite(offset) else -1.0
    seen_note = f"neighbours standing {standing}/{len(nbrs)}" + (
        f", unseen {unseen}" if unseen else ""
    )
    if knocked:
        tile_note = (
            "the tile stands"
            if placed.get("standing")
            else f"the tile: {placed.get('why') or 'not standing'}"
        )
        return finish(
            "knocked_neighbour",
            f"{hand_note}; neighbour(s) {knocked} no longer stand; {tile_note}; {seen_note}",
        )
    if not placed["seen"]:
        return finish(
            "unverified",
            f"{hand_note}; the slot could not be checked ({placed['why']}); {seen_note}",
        )
    if not placed["standing"]:
        return finish("fell", f"{hand_note}; {placed['why']} ({placed['camera']}); {seen_note}")
    return finish(
        "placed",
        f"{hand_note}; stands, top {placed['top'] * 1e3:.0f} mm, {placed['offset'] * 1e3:.1f} mm off the slot "
        f"({placed['camera']}, {placed['points']} points); {seen_note}",
    )
