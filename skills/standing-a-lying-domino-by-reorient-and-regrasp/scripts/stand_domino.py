# Reimplemented from the behaviour described for normalize_domino in Zhu-Chenyu/Franka-Domino-Project
# (place_dominoes/place_dominoes/place_dominoes.py): pick the lying tile top-down, turn the wrist so its long axis
# is vertical, set it down on its end at a staging spot, release, retreat - so a top-down pick can take it from
# there. That repository has no licence: no code was copied; this script is written from the description only.
# What is new here: the inputs are a world-frame centre and long axis; the 90 deg turn is made on the arm's own
# wrist-pitch joint, found by a small test turn of each wrist joint (a pose command asking for the pitched hand
# stalled in a YAM station twin whose solver leaves the wrist roll free), with the tile first turned about the
# vertical on the last joint so its long axis lies in the plane that joint swings; the tile's lean is read back
# from the hand's rotation before the release; Franka's force-limited descent is a measured height; and whether the
# tile stands is judged from depth.
"""Stand a flat-lying domino upright: pinch it across its width, lift, pitch the wrist 90 deg, set it on its end."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

TILE_SIZE = {"height": 0.048, "width": 0.024, "thickness": 0.0075}
"""A standard domino [m]: 48 x 24 x 7.5 mm (Franka-Domino-Project's and zmatt.net's tiles). Lying flat, its
height is the long axis and its thickness is vertical."""

GRASP_HEIGHT = 0.011
"""Commanded tool-centre height over the table at the close [m]. The linear_4310's fingertips reach 10.2 mm under
the tool centre; in a YAM station twin the hand arrived 2.2-2.7 mm over the command (13.2-13.7 mm), leaving the
pads 4.0-4.5 mm of the 7.5 mm edge. A real YAM station refuses a tool centre under 14 mm by default: see the
SKILL.md."""

MIN_BITE = 0.002
"""Least height of tile edge between the commanded fingertips and the tile's top [m] (a choice)."""

GRIP_OFFSET = 0.012
"""How far along the long axis from the tile's centre the jaws close [m], toward the end that ends up on top. The
tool centre then stands half the tile's height plus this over the table at the release, which keeps the pitched
hand's housing off the table; a quarter of a 48 mm tile, the pads still over the tile."""

LIFT_HEIGHT = 0.10
"""Tool-centre height over the table for the lift, the test turns, the turn about the vertical and the pitch
[m]: the held tile swings through half its height plus GRIP_OFFSET under the tool centre."""

RELEASE_GAP = 0.0015
"""Height of the tile's bottom over the table when the jaws open [m]: under a servo's 4 mm arrival bound."""

GRIP_CLEARANCE = 0.004
"""Jaw clearance each side of the tile when opening for the grasp and at the release [m]
(robot.describe_gripper's clearance on the YAM station twin)."""

RETREAT = 0.03
"""How far the hand backs off along its approach after the release before it rises [m]: the fingertips reach
10.2 mm past the tool centre, the tile's near face about 3-5 mm."""

TEST_TURN = 0.15
"""Each wrist joint's test turn [rad] (8.6 deg) to learn which way, and about which axis, it turns the hand."""

PITCH_AXIS_TILT = 0.25
"""Largest vertical component of a usable pitch joint's axis (unit) at the top-down hand: ~14 deg off level."""

JOINT_MARGIN = 0.05
"""Kept clear of each joint limit [rad]."""

YAM_LIMITS = (
    (-2.61799, 0.0, 0.0, -1.5708, -1.5708, -2.0944),
    (3.05433, 3.65, 3.66519, 1.5708, 1.5708, 2.0944),
)
"""Joint limits [rad] when robot.describe_arm does not give them: the i2rt YAM's (its MJCF)."""

ROLL_TOLERANCE_DEG = 4.0
"""How far the held tile may stay turned off its target about the hand's approach before the rest is turned on
the last joint [deg]: a servo's 3 deg arrival bound plus a degree."""

TILT_TOLERANCE_DEG = 6.0
"""Largest lean of the held tile's long axis from vertical at the release [deg]. A 48 x 7.5 mm tile tips past its
edge at 8.9 deg."""

TURN_TRIES = 3
"""Corrections tried per turn, each read back."""

TIP_REACH = 0.0102
"""How far the closed linear_4310 fingertips reach under the tool centre [m] when robot.describe_gripper does not
say."""

MIN_TOOL_HEIGHT = 0.010
"""Lowest tool-centre height over the table this skill commands [m]. A real YAM station refuses under 14 mm by
default: lower its floor or raise grasp_height (see the SKILL.md)."""

EMPTY_MARGIN = 0.003
"""A jaw gap within this of the closed stop holds nothing [m] (grasping-flat-objects)."""

NARROW_MARGIN = 0.006
"""A jaw gap this far under the tile's width did not close across it [m] (grasping-flat-objects)."""

FULL_MARGIN = 0.008
"""A jaw gap this far over the tile's width closed on something else [m]."""

DESCENT_TOLERANCE = 0.006
"""A straight descent that ends this far over its target stopped on something [m]: over a servo's 4 mm arrival
bound (grasping-flat-objects)."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from a free move's target [m]: no tighter than a real YAM station's 12 mm."""

STAGING_MOVE = 0.01
"""A staging spot farther than this from where the tile lay is carried to, and the pitch joint learned again
there [m]."""

HEIGHT_TOLERANCE = 0.010
"""A standing tile's top reads at least its height less this [m]; a lying one reads 7.5 or 24 mm."""

POSITION_TOLERANCE = 0.008
"""Largest offset of a standing tile's top from the staging spot [m]: a 10 deg lean moves a 48 mm tile's top
8.3 mm; a calibrated YAM station's camera RMS is 5.4 mm."""

SEARCH_MARGIN = 0.010
"""How far beyond the standing tile's footprint the depth check looks for its top [m]."""

OCCLUDER_RISE = 0.015
"""Points this far over a standing tile's top inside its footprint belong to something over it [m]."""

OCCLUDED_SHARE = 0.25
"""A camera whose window is this much something higher than a standing tile does not judge (a choice)."""

TOP_BAND = 0.006
"""Points within this of the highest one form the tile's top [m]."""

MIN_POINTS = 4
"""Fewest depth points on a tile's top to judge it (a 24 x 7.5 mm top is ~12 x 4 px from 0.9 m)."""

DEPTH_RANGE = (0.07, 3.0)
"""Valid depth [m]: a D405's 7 cm minimum, an overhead camera's far bound."""

PREFERRED_CAMERAS = ("overhead", "top")
"""Camera names tried first when no camera_name is given: the station's calibrated overhead."""


class Output(TypedDict):
    route: str
    detail: str
    picked: bool
    released: bool
    grasp_height: float
    grip_width: float
    pitch_joint: int
    facing_deg: float
    yaw_turn_deg: float
    pitch_turn_deg: float
    roll_turn_deg: float
    tilt_at_release_deg: float
    release_position: dict[str, float]
    standing_tile: dict[str, Any]
    verified: bool
    tile_top: float
    standing_offset: float


# ----------------------------------------------------------------------------- geometry


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"] if "position" in pose else pose
    return np.array([float(p["x"]), float(p["y"]), float(p.get("z", 0.0))])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def matrix(q: dict[str, float] | None) -> np.ndarray:
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


def quat(m: np.ndarray) -> dict[str, float]:
    """A rotation matrix as a unit quaternion {w, x, y, z} (w >= 0)."""
    t = float(np.trace(m))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x, y, z = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}


def axis_angle(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rotation matrix about a unit ``axis`` by ``deg``."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / (np.linalg.norm(k) or 1.0)
    a = math.radians(deg)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * kx + (1.0 - math.cos(a)) * (kx @ kx)


def rotation_vector(m: np.ndarray) -> np.ndarray:
    """The axis times the angle [rad] of a rotation matrix (the turn it makes)."""
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(m)) - 1.0) / 2.0)))
    if angle < 1.0e-9:
        return np.zeros(3)
    v = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    n = float(np.linalg.norm(v))
    if n < 1.0e-9:  # a half turn: the axis from the symmetric part
        k = np.sqrt(np.maximum(0.0, (np.diag(m) + 1.0) / 2.0))
        return k * angle
    return v / n * angle


def heading(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a), 0.0])


def wrap(deg: float, half: bool) -> float:
    """An angle folded into [-180, 180), or into [-90, 90) when a half turn is the same grasp."""
    period = 180.0 if half else 360.0
    return (float(deg) + period / 2.0) % period - period / 2.0


def turn_about(axis: np.ndarray, frm: np.ndarray, to: np.ndarray) -> float:
    """Signed angle [deg] that turns ``frm`` onto ``to`` about ``axis`` (both projected on its normal plane)."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) or 1.0)
    f = frm - (frm @ a) * a
    t = to - (to @ a) * a
    return math.degrees(math.atan2(float(a @ np.cross(f, t)), float(f @ t)))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    c = float(a @ b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def plan_release(
    staging_xy: np.ndarray,
    facing_deg: float,
    table_z: float,
    size: dict[str, float],
    grasp_height: float,
    grip_offset: float,
    release_gap: float,
) -> np.ndarray:
    """Where the tool centre goes to set the tile down standing at ``staging_xy``, facing ``facing_deg`` (the
    pitched approach): the tile hangs half its height plus ``grip_offset`` under the tool centre, its centre
    ``grasp_height`` less half its thickness ahead of it along the approach."""
    u = heading(facing_deg)
    ahead = float(grasp_height) - size["thickness"] / 2.0
    xy = np.asarray(staging_xy, dtype=np.float64)[:2] - ahead * u[:2]
    return np.array([*xy, table_z + size["height"] / 2.0 + float(grip_offset) + float(release_gap)])


def standing_pose(
    xy: np.ndarray, facing_deg: float, table_z: float, size: dict[str, float]
) -> dict[str, Any]:
    """The standing tile as placing-an-upright-domino-beside-neighbours reads it: centre, rotation whose x axis
    is the facing (the broad faces' normal), and facing_deg."""
    return {
        "position": _vec(np.array([float(xy[0]), float(xy[1]), table_z + size["height"] / 2.0])),
        "rotation": quat(axis_angle(np.array([0.0, 0.0, 1.0]), facing_deg)),
        "facing_deg": float(facing_deg),
    }


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
        t[:3, :3] = matrix(pose.get("rotation") or pose.get("orientation"))
        t[:3, 3] = _xyz(pose)
        return t
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)


def camera_points(cam: dict[str, Any], corners: np.ndarray) -> np.ndarray:
    """World points (N x 3) of the camera's depth pixels inside the image box of the world ``corners``
    (OpenCV pinhole); none when that box is not wholly inside the image."""
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


def judge_standing(
    cams: list[dict[str, Any]],
    xy: np.ndarray,
    facing_deg: float,
    table_z: float,
    size: dict[str, float],
) -> dict[str, Any]:
    """Whether a tile stands at (xy, facing): the highest depth points inside its footprint widened by
    SEARCH_MARGIN are its top; it stands when that top is within HEIGHT_TOLERANCE of the tile's height and
    within POSITION_TOLERANCE of xy. The first camera that sees the window whole and unoccluded judges."""
    along = heading(facing_deg)[:2]
    across = np.array([-along[1], along[0]])
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
                "offset": float("nan"),
                "why": "nothing stands there",
            }
        else:
            top = float(np.percentile(z[raised], 95))
            band = raised & (z >= top - TOP_BAND)
            offset = math.hypot(float(np.mean(a[band])), float(np.mean(w[band])))
            standing = top >= h - HEIGHT_TOLERANCE and offset <= POSITION_TOLERANCE
            why = (
                ""
                if standing
                else (
                    f"its top is {top * 1e3:.0f} mm over the table (a standing tile's is {h * 1e3:.0f})"
                    if top < h - HEIGHT_TOLERANCE
                    else f"its top is {offset * 1e3:.0f} mm off the spot (leaning or misplaced)"
                )
            )
            verdict = {
                "seen": True,
                "standing": bool(standing),
                "points": n,
                "camera": name,
                "top": top,
                "offset": offset,
                "why": why,
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
    tile_center: dict[str, float],
    long_axis: dict[str, float],
    staging: dict[str, float] | None = None,
    tile_size: dict[str, float] | None = None,
    grasp_height: float = GRASP_HEIGHT,
    grip_offset: float = GRIP_OFFSET,
    lift_height: float = LIFT_HEIGHT,
    release_gap: float = RELEASE_GAP,
    grip_clearance: float = GRIP_CLEARANCE,
    retreat: float = RETREAT,
    min_tool_height: float = MIN_TOOL_HEIGHT,
    pitch_joint: int = -1,
    joint_limits: list[list[float]] | None = None,
    verify: bool = True,
    camera_name: str = "",
    park_pose: dict[str, Any] | None = None,
    roll_tolerance_deg: float = ROLL_TOLERANCE_DEG,
    tilt_tolerance_deg: float = TILT_TOLERANCE_DEG,
    reach_tolerance: float = REACH_TOLERANCE,
    descent_tolerance: float = DESCENT_TOLERANCE,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    size = {**TILE_SIZE, **{k: float(v) for k, v in (tile_size or {}).items()}}
    tl, tw, tt = size["height"], size["width"], size["thickness"]
    table_z = float(table_z)
    center = _xyz(tile_center)
    ax = _xyz(long_axis)
    spot = _xyz(staging)[:2] if staging else center[:2].copy()
    s = max(0.0, min(float(grip_offset), tl / 2.0))
    z_up = np.array([0.0, 0.0, 1.0])

    out: Output = {
        "route": "unreachable",
        "detail": "",
        "picked": False,
        "released": False,
        "grasp_height": float(grasp_height),
        "grip_width": -1.0,
        "pitch_joint": -1,
        "facing_deg": 0.0,
        "yaw_turn_deg": 0.0,
        "pitch_turn_deg": 0.0,
        "roll_turn_deg": 0.0,
        "tilt_at_release_deg": -1.0,
        "release_position": _vec(np.array([*spot, table_z])),
        "standing_tile": standing_pose(spot, 0.0, table_z, size),
        "verified": False,
        "tile_top": -1.0,
        "standing_offset": -1.0,
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        return out

    if float(np.hypot(ax[0], ax[1])) < 1.0e-6:
        return finish(
            "unreachable", "the long axis has no horizontal direction: the tile does not lie flat"
        )
    theta = math.degrees(math.atan2(ax[1], ax[0]))

    try:
        hand = ctx.tool("robot.describe_gripper", **on_arm)
    except Exception as exc:  # the width checks and the reach fall back to defaults
        logger.info("standing-a-lying-domino: no gripper description (%s)", exc)
        hand = {}
    hand = hand if isinstance(hand, dict) else {}
    fit = hand.get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or hand.get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)
    finger = hand.get("finger") or {}
    tip_reach = float(finger.get("reach_m") or 0.0) if finger.get("stated", True) else 0.0
    tip_reach = tip_reach or TIP_REACH
    limits = joint_limits
    if limits is None:
        try:
            arm = ctx.tool("robot.describe_arm", **on_arm)
            jl = arm.get("joint_limits") if isinstance(arm, dict) else None
            limits = [jl["lower"], jl["upper"]] if jl else None
        except Exception as exc:
            logger.info("standing-a-lying-domino: no arm description (%s)", exc)
    lower, upper = (np.asarray(v, dtype=np.float64) for v in (limits or YAM_LIMITS))

    gh = float(grasp_height)
    if gh < float(min_tool_height):
        return finish(
            "unreachable",
            f"grasp_height {gh * 1e3:.1f} mm is under the {float(min_tool_height) * 1e3:.0f} mm tool-centre floor",
        )
    if tt - (gh - tip_reach) < MIN_BITE:
        return finish(
            "ungraspable",
            f"at grasp_height {gh * 1e3:.1f} mm the fingertips ({tip_reach * 1e3:.1f} mm under the tool centre) "
            f"stand over the {tt * 1e3:.1f} mm tile's top less {MIN_BITE * 1e3:.0f} mm: nothing to pinch",
        )
    grip_open = tw + 2.0 * float(grip_clearance)
    if at_open > 0.0 and grip_open > at_open:
        return finish(
            "ungraspable",
            f"a {tw * 1e3:.0f} mm tile does not fit the {at_open * 1e3:.0f} mm open hand",
        )
    lift_z = table_z + max(float(lift_height), tl / 2.0 + s + tip_reach + 0.02)

    start = _pose(ctx.tool("robot.get_ee_pose", **on_arm))
    start_xyz = _xyz(start)

    def here() -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **on_arm))

    def rot_now() -> np.ndarray:
        return matrix(here()["rotation"])

    def refusal(reply: Any) -> str:
        if not isinstance(reply, dict):
            return ""
        if reply.get("outcome") not in (None, "reached"):
            return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
        if reply.get("reached") is False or reply.get("success") is False:
            return str(reply.get("reason") or reply.get("message") or "the move reported failure")
        return ""

    def move(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        try:
            return refusal(
                ctx.tool(tool, pose={"position": _vec(position), "rotation": rot}, **on_arm)
            )
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc) or type(exc).__name__

    def go(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        """A move judged by where the hand is: "" when it ended within reach_tolerance."""
        why = move(tool, position, rot)
        off = float(np.linalg.norm(_xyz(here()) - position))
        if (why and not why.startswith("timeout")) or off > float(reach_tolerance):
            return f"{why or 'ended off'}; the hand is {off * 1e3:.0f} mm from it"
        return ""

    def joints() -> np.ndarray:
        try:
            arms = ctx.tool("robot.get_observation").get("arms") or []
        except Exception as exc:
            logger.info("standing-a-lying-domino: no joint reading (%s)", exc)
            return np.zeros(0)
        if int(arm_id) >= len(arms):
            return np.zeros(0)
        q = arms[int(arm_id)].get("joint_state")
        if isinstance(q, dict):
            q = q.get("positions")
        return np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)

    def to_joints(q: np.ndarray) -> str:
        try:
            return refusal(
                ctx.tool("robot.move_to_joints", joint_config={"positions": q.tolist()}, **on_arm)
            )
        except Exception as exc:
            return str(exc) or type(exc).__name__

    def fits(j: int, value: float) -> bool:
        return (
            j >= len(lower)
            or float(lower[j]) + JOINT_MARGIN <= value <= float(upper[j]) - JOINT_MARGIN
        )

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
            logger.info("standing-a-lying-domino: could not read the gripper (%s)", exc)
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
            return refusal(ctx.tool("robot.set_grip", width_m=float(w), **on_arm))
        except Exception as exc:
            return str(exc) or type(exc).__name__

    def solves(point: np.ndarray, rot: np.ndarray) -> bool:
        if not check_reach:
            return True
        try:
            reply = ctx.tool(
                "robot.solve_ik", pose={"position": _vec(point), "rotation": quat(rot)}, **on_arm
            )
        except Exception as exc:
            logger.info("standing-a-lying-domino: IK preview refused (%s)", exc)
            return False
        return not (
            isinstance(reply, dict)
            and (reply.get("success") is False or reply.get("within_servo_tolerance") is False)
        )

    # The hand-frame axes, from the hand's own grasp frame: the closing axis and the approach.
    try:
        gf = ctx.tool("robot.grasp_frame", close_heading_deg=theta + 90.0, **on_arm)
    except Exception as exc:
        return finish("unreachable", f"no grasp frame ({exc})")
    grasp_rots = [
        matrix(gf[k])
        for k in ("rotation", "rotation_flipped")
        if isinstance(gf, dict) and gf.get(k)
    ]
    if not grasp_rots:
        return finish("unreachable", "the grasp frame returned no rotation")
    local_close = grasp_rots[0].T @ _xyz(gf.get("close_axis") or _vec(heading(theta + 90.0)))
    local_approach = grasp_rots[0].T @ _xyz(gf.get("approach") or {"x": 0.0, "y": 0.0, "z": -1.0})

    def jaw_error(r: np.ndarray, target: np.ndarray) -> float:
        """How far the jaws must turn about the approach to close along ``target``'s closing axis [deg]."""
        return wrap(turn_about(r @ local_approach, r @ local_close, target @ local_close), True)

    # A preview of the top-down grasp, the nearer wrist solution first.
    cur = rot_now()
    grasp_rots.sort(key=lambda g: abs(jaw_error(cur, g)))
    grasp_rots = [g for g in grasp_rots if solves(np.array([*center[:2], table_z + gh]), g)]
    if not grasp_rots:
        return finish(
            "unreachable",
            "the grasp over the tile is out of reach (IK preview, both wrist solutions)",
        )
    g_rot = grasp_rots[0]

    roll_gain = [0.0]

    def turn_last(err_fn: Any, at: np.ndarray, notes: list[str], key: str) -> np.ndarray:
        """Turn the last joint until ``err_fn(rotation)`` (deg about the approach) is within roll_tolerance_deg,
        re-centring over ``at`` after each turn. Returns the hand's rotation."""
        r = rot_now()
        for _ in range(TURN_TRIES):
            err = float(err_fn(r))
            if abs(err) <= float(roll_tolerance_deg):
                break
            q = joints()
            if q.size == 0:
                notes.append("no joint reading, the wrist was not turned")
                break
            options = [err, err - math.copysign(360.0, err)]
            gain = roll_gain[0] or 1.0
            ok = [o for o in options if fits(q.size - 1, q[-1] + math.radians(o) / gain)]
            if not ok:
                notes.append(f"the last joint has no room to turn {err:+.0f} deg")
                break
            command = math.radians(ok[0]) / gain
            goal = q.copy()
            goal[-1] += command
            why = to_joints(goal)
            if why:
                notes.append(f"wrist turn: {why}")
                break
            before, r = r, rot_now()
            moved = turn_about(r @ local_approach, before @ local_close, r @ local_close)
            out[key] += moved  # type: ignore[literal-required]
            if abs(command) > math.radians(2.0) and 0.5 < abs(math.radians(moved) / command) < 1.5:
                roll_gain[0] = math.radians(moved) / command
            why = go("robot.go_to_pose", at, quat(r))
            if why:
                notes.append(f"re-centring after the wrist turn: {why}")
                break
            r = rot_now()
        return r

    def learn(notes: list[str]) -> tuple[int, np.ndarray, float] | None:
        """Test-turn the last joint (its sign about the approach) and the wrist joints before it; return the
        joint that pitches the top-down hand about a level axis by a quarter turn within its limits, that axis
        (unit, world) and the joint change, or None."""
        q0 = joints()
        if q0.size < 3:
            notes.append("no joint reading")
            return None
        n = q0.size

        def test(j: int) -> tuple[np.ndarray, float] | None:
            step = TEST_TURN if fits(j, q0[j] + TEST_TURN) else -TEST_TURN
            r0 = rot_now()
            q = q0.copy()
            q[j] += step
            why = to_joints(q)
            r1 = rot_now()
            back = to_joints(q0)
            if why or back:
                notes.append(f"test turn of joint {j + 1}: {why or back}")
                return None
            w = rotation_vector(r1 @ r0.T) / step
            return w, step

        t = test(n - 1)
        if t is not None:
            w, _ = t
            along = float(w @ (rot_now() @ local_approach))
            if abs(along) > 0.5:
                roll_gain[0] = along
        joints_to_try = [int(pitch_joint)] if int(pitch_joint) >= 0 else [n - 3, n - 2]
        best: tuple[int, np.ndarray, float] | None = None
        best_room = -math.inf
        for j in joints_to_try:
            t = test(j)
            if t is None:
                continue
            w, _ = t
            rate = float(np.linalg.norm(w))
            if rate < 0.5:
                continue
            k = w / rate
            if abs(float(k[2])) > PITCH_AXIS_TILT:
                notes.append(
                    f"joint {j + 1} turns the hand about an axis {math.degrees(math.asin(abs(k[2]))):.0f} deg off level"
                )
                continue
            for sign in (1.0, -1.0):
                dq = sign * (math.pi / 2.0) / rate
                if not fits(j, q0[j] + dq):
                    continue
                room = (
                    min(q0[j] + dq - float(lower[j]), float(upper[j]) - q0[j] - dq)
                    if j < len(lower)
                    else 0.0
                )
                if room > best_room:
                    best_room, best = room, (j, sign * k, dq)
            if best is None:
                notes.append(f"joint {j + 1} has no room for a quarter turn either way")
        return best

    # --- the pick: over the tile, learn the wrist, square the jaws, straight down, close, lift.
    why = set_width(grip_open)
    if why:
        return finish("unreachable", f"the jaws did not open to {grip_open * 1e3:.0f} mm ({why})")
    now = _xyz(here())
    if now[2] < lift_z - 1.0e-3:
        why = go(
            "robot.go_to_pose_cartesian", np.array([now[0], now[1], lift_z]), here()["rotation"]
        )
        if why:
            return finish("unreachable", f"rise before the pick: {why}")
    above = np.array([*center[:2], lift_z])
    why = go("robot.go_to_pose", above, quat(g_rot))
    if why:
        return finish("unreachable", f"over the tile: {why}")
    notes: list[str] = []
    pitch = learn(notes)
    if pitch is None:
        return finish(
            "unreachable",
            "no wrist joint can pitch the hand a quarter turn here"
            + (f" ({'; '.join(notes)})" if notes else ""),
        )
    j, k, dq = pitch
    out["pitch_joint"] = j + 1
    # The pitch turns the approach (down) toward u; the end of the tile along u goes up.
    u = np.cross(k, np.array([0.0, 0.0, -1.0]))
    u = np.array([u[0], u[1], 0.0]) / (float(np.hypot(u[0], u[1])) or 1.0)
    e0 = heading(theta) if float(heading(theta) @ u) >= 0.0 else heading(theta + 180.0)
    grasp_point = np.array([*(center[:2] + s * e0[:2]), table_z + gh])
    r_now = turn_last(lambda r: jaw_error(r, g_rot), above, notes, "yaw_turn_deg")
    err = jaw_error(r_now, g_rot)
    if abs(err) > float(roll_tolerance_deg):
        return finish(
            "unreachable",
            f"the jaws stayed {err:+.0f} deg off the tile's width axis ({'; '.join(notes) or 'not turned'})",
        )
    out["yaw_turn_deg"] = 0.0
    why = go("robot.go_to_pose", np.array([*grasp_point[:2], lift_z]), quat(r_now))
    if why:
        return finish("unreachable", f"over the grasp point: {why}")
    why = move("robot.go_to_pose_cartesian", grasp_point, quat(r_now))
    reached = _xyz(here())
    high = float(reached[2] - grasp_point[2])
    if high > float(descent_tolerance) or float(
        np.linalg.norm(reached[:2] - grasp_point[:2])
    ) > float(reach_tolerance):
        go("robot.go_to_pose_cartesian", np.array([*reached[:2], lift_z]), quat(r_now))
        return finish(
            "ungraspable",
            f"the descent to the tile stopped {high * 1e3:.1f} mm high ({why or 'stalled'}): a fingertip on the "
            "tile, or it is not where it was seen; nothing closed",
        )
    out["grip_width"] = width(ctx.tool("robot.close_gripper", **on_arm))
    why = holding(out["grip_width"])
    if why:
        set_width(grip_open)
        go("robot.go_to_pose_cartesian", np.array([*reached[:2], lift_z]), quat(r_now))
        return finish("ungraspable", f"grasp: {why}; the hand is open over the tile")
    g_meas = float(reached[2] - table_z)
    out["grasp_height"] = g_meas
    out["picked"] = True
    r_grip = rot_now()
    e_local = r_grip.T @ e0  # the end that goes up, in the hand's frame
    why = go("robot.go_to_pose_cartesian", np.array([*reached[:2], lift_z]), quat(r_grip))
    if why:
        return finish("unreachable", f"lift after the grasp: {why}; still holding")
    why = holding(width())
    if why:
        return finish("dropped", f"after the lift: {why}")

    # --- carried to another staging spot: learn the wrist again there.
    if float(np.linalg.norm(spot - center[:2])) > STAGING_MOVE:
        why = go("robot.go_to_pose", np.array([*spot, lift_z]), quat(r_grip))
        if why:
            return finish("unreachable", f"to over the staging spot: {why}; still holding")
        pitch = learn(notes)
        if pitch is None:
            return finish(
                "unreachable",
                "no wrist joint can pitch the hand a quarter turn over the staging spot ("
                + "; ".join(notes)
                + "); still holding",
            )
        j, k, dq = pitch
        out["pitch_joint"] = j + 1
        u = np.cross(k, np.array([0.0, 0.0, -1.0]))
        u = np.array([u[0], u[1], 0.0]) / (float(np.hypot(u[0], u[1])) or 1.0)
    at = _xyz(here())

    # --- turn the held tile about the vertical so the end that goes up points along u.
    notes = []
    r_now = turn_last(
        lambda r: turn_about(r @ local_approach, r @ e_local, u), at, notes, "yaw_turn_deg"
    )
    err = turn_about(r_now @ local_approach, r_now @ e_local, u)
    if abs(err) > float(roll_tolerance_deg):
        return finish(
            "unreachable",
            f"the held tile stayed {err:+.0f} deg off the pitch's plane ({'; '.join(notes) or 'not turned'}); "
            "still holding",
        )

    # --- the pitch, on the wrist joint, then read back: the tile's long axis from vertical.
    q = joints()
    goal = q.copy()
    goal[j] += dq
    why = to_joints(goal)
    if why:
        return finish("unreachable", f"the pitch on joint {j + 1}: {why}; still holding")
    out["pitch_turn_deg"] = math.degrees(dq)
    r_now = rot_now()
    lean = turn_about(k, r_now @ e_local, z_up)
    if abs(lean) > float(roll_tolerance_deg):
        q = joints()
        rate = abs(dq) / (math.pi / 2.0)
        fix = math.radians(lean) / rate * (1.0 if dq > 0 else -1.0)
        if q.size and fits(j, q[j] + fix):
            q[j] += fix
            to_joints(q)
            out["pitch_turn_deg"] += math.degrees(fix)
            r_now = rot_now()
    why = holding(width())
    if why:
        return finish("dropped", f"after the pitch: {why}")
    a_now = r_now @ local_approach
    if float(np.hypot(a_now[0], a_now[1])) < 0.5:
        return finish(
            "unreachable",
            f"after the pitch the hand still points {np.round(a_now, 2).tolist()}; still holding",
        )
    psi = math.degrees(math.atan2(a_now[1], a_now[0]))
    out["facing_deg"] = float(wrap(psi, False))
    out["standing_tile"] = standing_pose(spot, out["facing_deg"], table_z, size)
    u = heading(psi)

    # --- over the release point, the roll read back (the tile's up end about the approach), straight down.
    release = plan_release(spot, psi, table_z, size, g_meas, s, float(release_gap))
    high_pt = np.array([*release[:2], lift_z])
    if not solves(release, r_now):
        return finish(
            "unreachable",
            "the release with the pitched hand is out of reach (IK preview); still holding",
        )
    why = go("robot.go_to_pose", high_pt, quat(r_now))
    if why:
        return finish("unreachable", f"to over the staging spot, pitched: {why}; still holding")
    notes = []
    r_now = turn_last(
        lambda r: turn_about(r @ local_approach, r @ e_local, z_up), high_pt, notes, "roll_turn_deg"
    )
    why = move("robot.go_to_pose_cartesian", release, quat(r_now))
    reached = _xyz(here())
    r_now = rot_now()
    high = float(reached[2] - release[2])
    tilt = angle_between(r_now @ e_local, z_up)
    out["tilt_at_release_deg"] = tilt
    if high > float(descent_tolerance) or float(np.linalg.norm(reached[:2] - release[:2])) > float(
        reach_tolerance
    ):
        go("robot.go_to_pose_cartesian", high_pt, quat(r_now))
        return finish(
            "unreachable",
            f"the descent to the staging spot stopped {high * 1e3:.1f} mm high and "
            f"{np.linalg.norm(reached[:2] - release[:2]) * 1e3:.1f} mm aside ({why or 'stalled'}): something under "
            "the tile or the hand; lifted back, still holding",
        )
    if tilt > float(tilt_tolerance_deg):
        go("robot.go_to_pose_cartesian", high_pt, quat(r_now))
        return finish(
            "unreachable",
            f"at the release the tile's long axis leaned {tilt:.1f} deg from vertical"
            + (f" ({'; '.join(notes)})" if notes else "")
            + "; lifted back, still holding",
        )
    why = holding(width())
    if why:
        return finish("dropped", f"over the staging spot: {why}")
    landed = reached[:2] + (g_meas - tt / 2.0) * u[:2]
    out["release_position"] = _vec(np.array([*landed, reached[2] - tl / 2.0 - s]))
    why = set_width(grip_open)
    if why:
        try:
            ctx.tool("robot.open_gripper", **on_arm)
            notes.append(f"opened fully ({why})")
        except Exception as exc:
            return finish(
                "unreachable", f"the jaws did not open at the staging spot ({exc}); still holding"
            )
    out["released"] = True
    back = reached - float(retreat) * u
    why = go("robot.go_to_pose_cartesian", back, quat(r_now))
    if not why:
        why = go("robot.go_to_pose_cartesian", np.array([*back[:2], lift_z]), quat(r_now))
    if why:
        return finish("unreachable", f"released, but the retreat: {why}")

    hand_note = (
        f"set down {np.linalg.norm(landed - spot) * 1e3:.1f} mm from the spot, facing {out['facing_deg']:.0f} deg, "
        f"pitched on joint {j + 1} by {out['pitch_turn_deg']:+.0f} deg, leaning {tilt:.1f} deg at the release"
    ) + (f"; {'; '.join(notes)}" if notes else "")
    if not verify:
        return finish("standing", hand_note + "; not checked from depth")

    # --- the check: out of the cameras' way, then the staging spot from depth.
    park = (
        _xyz(_pose(park_pose))
        if park_pose
        else np.array([*start_xyz[:2], max(start_xyz[2], lift_z)])
    )
    park_rot = (_pose(park_pose) or {}).get("rotation") if park_pose else None
    why = go("robot.go_to_pose", park, park_rot or start["rotation"])
    if why:
        hand_note += f"; park: {why}"
    try:
        cams = order_cameras(_cameras(ctx.tool("robot.get_observation")), camera_name)
    except Exception as exc:
        return finish("unverified", f"{hand_note}; no observation for the check ({exc})")
    v = judge_standing(cams, spot, out["facing_deg"], table_z, size)
    if v["seen"]:
        out["verified"] = True
        out["tile_top"] = float(v.get("top", -1.0))
        off = float(v.get("offset", -1.0))
        out["standing_offset"] = off if math.isfinite(off) else -1.0
    if not v["seen"]:
        return finish("unverified", f"{hand_note}; the spot could not be checked ({v['why']})")
    if not v["standing"]:
        return finish("fell", f"{hand_note}; {v['why']} ({v['camera']})")
    return finish(
        "standing",
        f"{hand_note}; stands, top {v['top'] * 1e3:.0f} mm, {v['offset'] * 1e3:.1f} mm off the spot "
        f"({v['camera']}, {v['points']} points)",
    )
