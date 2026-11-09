# Reimplemented from the behaviour described for normalize_domino in Zhu-Chenyu/Franka-Domino-Project
# (place_dominoes/place_dominoes/place_dominoes.py): pick the lying tile top-down, turn the wrist so its long axis
# is vertical, set it down on its end at a staging spot, release, retreat - so a top-down pick can take it from
# there. That repository has no licence: no code was copied; this script is written from the description only.
# What is new here: the inputs are a world-frame centre and long axis; the heading of the pitched approach (the
# facing) is chosen by IK preview before anything moves, since a pitched hand near the table reaches only some
# headings on a YAM; the quarter turn is driven to the previewed joints in small steps, then the wrist roll is read
# back and finished on the last joint (a YAM station twin's solver leaves the roll free); the tile's lean is read
# back from the hand's rotation before the release; Franka's force-limited descent is a measured height; and
# whether the tile stands is judged from depth.
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

GRIP_OFFSET = 0.014
"""How far along the long axis from the tile's centre the jaws close [m], toward the end that ends up on top. The
tool centre then stands half the tile's height plus this over the table at the release (39.5 mm here): in a YAM
station twin the pitched hand stalled 37.7 mm over the table when asked for 25.5 mm (gripped at the centre). Held
16 mm off its centre and lifted, a tile there sagged 1.7 deg in the pinch."""

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

FACING_STEP_DEG = 10.0
"""Spacing of the facings (headings of the pitched approach) tried in the IK preview [deg]. The release with the
pitched hand solves only in bands of about this width in a YAM station twin (a 30 deg probe grid found one or two
facing pairs per spot, none along the arm's radius)."""

PITCH_STEP_DEG = 15.0
"""The quarter turn is commanded as pose moves of at most this turn about the level axis through the tool centre
[deg]. In a YAM station twin one 90 deg pose command stalled (the servo's solver lost the path), and one jump of
the joints to a solved pitched pose (1.6 rad on a wrist joint) flung the held tile out of the jaws or knocked a
neighbour."""

PITCH_JOINT_JUMP = 0.6
"""Largest change of any joint but the last between the IK solutions of two consecutive pitch steps [rad]. In a
YAM station twin the steps of a passing facing moved the joints 0.2-0.5 rad each, then the last 15 deg step jumped
the wrist pitch 1.27 rad onto its 1.5708 limit, and the tile was lost in that step. Allowed 0.9 rad, five of six
facings that passed still lost the tile in the pitch."""

STRATEGIES = ("table_pivot", "pose_pitch", "hinge_tip", "handover")
"""table_pivot: find the wrist joint that pitches the hand about a level axis, carry the tile flat into that joint's
plane far enough out that the upright hand reaches the table, and turn it upright about its inner bottom edge on the
table by chained IK steps. pose_pitch: the quarter turn in the air in pose steps about the tool centre over the
staging spot (stood 0 tiles in a YAM station twin; kept for arms whose solver holds the whole pose)."""

PROBE_TURN = 0.15
"""Test turn of each wrist joint with the hand empty over the tile, to find the pitch joint [rad]. In a YAM station
twin 0.15 rad of joint 4 turned the hand 8.6 deg and swung the tool centre 37 mm."""

LEVEL_TOLERANCE = 0.2
"""Largest vertical share of a joint's unit axis for it to count as level (a choice: 11.5 deg)."""

AIR_PITCH_DEG = 0.0
"""How far the pitch joint alone turns the held tile in the air before it is set on the table [deg]. In a YAM station
twin the tile slid down the pads as it tilted (8-11 mm by 45-55 deg) and fell out between 45 and 80 deg, so by
default the whole turn is made with the tile's lower edge on the table."""

PITCH_ROOM = 0.3
"""A pitch joint whose quarter turn leaves this much travel to its limit [rad] is taken without testing the joints
behind it (a YAM's joint 5 is level too, but its quarter turn ends on its limit)."""

PIVOT_STEP_DEG = 5.0
"""Turn per chained IK step while the tile is stood up about its lower edge on the table [deg]."""

IK_STEP = 0.02
"""Longest straight chunk of a chained IK move [m]. In a YAM station twin 20 mm descents of the pitched hand moved
each joint 0.07-0.08 rad."""

SETDOWN_GAP = 0.001
"""Height of the flat tile's bottom over the table when the hand stops over the spot [m]."""

PRESS = 0.0
"""How far under the table the edge the tile turns about is commanded [m]. With 1 mm the tile left the jaws halfway
up in a YAM station twin: the pinch is weak along the tile, and the press pushes the tile through it."""

STAGE_STEP = 0.03
"""With no staging spot given, spots are tried this far apart [m], out from where the tile lies along the pitch
plane; the upright hand at each is previewed with the empty hand turned upright (the rehearsal)."""

STAGE_REACH = 0.30
"""How far out the spots go [m]: in a YAM station twin the upright hand reached 40 mm over the table at 0.55 m from
the base, a tile lies 0.25-0.40 m out."""

STAGE_ATTEMPTS = 3
"""Spots at which the turn upright is started at most; a turn that fails is backed out the way it came."""

HINGE_OFFSET = 0.016
"""hinge_tip: how far from the tile's centre toward the end that goes up the jaws pinch [m] (8 mm from that end)."""

HINGE_PRESS = 0.002
"""hinge_tip: how much shorter than the pin's distance to the far bottom edge the arc's radius is [m], so the pin
pushes the tile's foot onto the table (tipping-over-a-surface-edge presses while it pulls)."""

HANDOVER_HEIGHT = 0.15
"""handover: height of the tile's centre over the table where the helper takes it [m] (a choice: clear of the
table for a level hand)."""

HELPER_OFFSET = 0.012
"""handover: how far from the tile's centre toward its far end the helper pinches [m] (12 mm from that end): clear
of this arm's pads, which sit 6-22 mm toward the other end."""

HELPER_BITE = 0.010
"""handover: how far the helper's tool centre goes past the tile's near long edge [m]; its fingertips reach 10.2 mm
further."""

TOP_BITE = 0.008
"""handover: how far down the upright tile this arm's fingertips reach when it takes the top end [m]."""

TURN_STEP_DEG = 10.0
"""Largest turn of the hand per chained IK step when a move also turns it [deg]."""

JOINT_SETTLE = 0.075
"""A joint move that ends farther than this from its target on any joint did not arrive [rad]: a real YAM station's
settle bound (PORTING lesson 3)."""

CENTRE_TOLERANCE = 0.004
"""A wrist turn that moved the tool centre more than this is followed by a re-centring move [m] (a servo's 4 mm
arrival bound)."""

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
    strategy: str
    pitch_joint: int
    picked: bool
    released: bool
    grasp_height: float
    grip_width: float
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


def rotvec(m: np.ndarray) -> np.ndarray:
    """The rotation vector (axis times angle [rad]) of a rotation matrix."""
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(m)) - 1.0) / 2.0)))
    if angle < 1.0e-9:
        return np.zeros(3)
    v = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    return v / (2.0 * math.sin(angle)) * angle


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    c = float(a @ b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def pitched_rotation(
    u: np.ndarray, local_approach: np.ndarray, local_close: np.ndarray, local_up: np.ndarray
) -> np.ndarray:
    """The hand's rotation after the quarter turn: its approach along the level heading ``u``, the hand-frame
    direction ``local_up`` (the tile end that goes up) vertical, its closing axis level."""
    a = np.asarray(local_approach, dtype=np.float64)
    a = a / (np.linalg.norm(a) or 1.0)
    c = np.asarray(local_close, dtype=np.float64) - float(np.asarray(local_close) @ a) * a
    c = c / (np.linalg.norm(c) or 1.0)
    e = np.cross(a, c)
    if float(e @ np.asarray(local_up, dtype=np.float64)) < 0.0:
        e, c = -e, -c
    z = np.array([0.0, 0.0, 1.0])
    level = np.asarray(u, dtype=np.float64)
    b_world = np.stack([level, np.cross(z, level), z], axis=1)
    return b_world @ np.stack([a, c, e], axis=1).T


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


def _refusal(reply: Any) -> str:
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def _frame(
    approach: np.ndarray, close: np.ndarray, local_approach: np.ndarray, local_close: np.ndarray
) -> np.ndarray:
    """The hand rotation whose approach is ``approach`` and whose closing axis is ``close`` (made square to it)."""
    a = np.asarray(approach, dtype=np.float64)
    a = a / (np.linalg.norm(a) or 1.0)
    c = np.asarray(close, dtype=np.float64) - float(np.asarray(close) @ a) * a
    c = c / (np.linalg.norm(c) or 1.0)
    la = np.asarray(local_approach, dtype=np.float64)
    la = la / (np.linalg.norm(la) or 1.0)
    lc = np.asarray(local_close, dtype=np.float64) - float(np.asarray(local_close) @ la) * la
    lc = lc / (np.linalg.norm(lc) or 1.0)
    return np.stack([a, c, np.cross(a, c)], axis=1) @ np.stack([la, lc, np.cross(la, lc)], axis=1).T


class _Hand:
    """One arm through robot.* tools (the two-arm strategy drives two): pose, joints, chained IK steps refused when
    a joint but the last would jump, the jaws."""

    def __init__(
        self, ctx: Any, arm_id: int, lower: np.ndarray, upper: np.ndarray, jump: float
    ) -> None:
        self.ctx, self.id = ctx, int(arm_id)
        self.kw: dict[str, Any] = {"arm_id": self.id} if self.id else {}
        self.lower, self.upper, self.jump = np.asarray(lower), np.asarray(upper), float(jump)
        self.at_open, self.at_closed = 0.0, 0.0
        try:
            hand = ctx.tool("robot.describe_gripper", **self.kw)
            fit = (hand or {}).get("width_fit") or {}
            self.at_open = float(fit.get("at_open_m") or (hand or {}).get("span_m") or 0.0)
            self.at_closed = float(fit.get("at_closed_m") or 0.0)
        except Exception as exc:
            logger.info(
                "standing-a-lying-domino: no gripper description for arm %d (%s)", self.id, exc
            )

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        p = _pose(self.ctx.tool("robot.get_ee_pose", **self.kw))
        return _xyz(p), matrix(p["rotation"])

    def q(self) -> np.ndarray:
        try:
            arms = self.ctx.tool("robot.get_observation").get("arms") or []
        except Exception as exc:
            logger.info("standing-a-lying-domino: no joint reading (%s)", exc)
            return np.zeros(0)
        if self.id >= len(arms):
            return np.zeros(0)
        q = arms[self.id].get("joint_state")
        if isinstance(q, dict):
            q = q.get("positions")
        return np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)

    def fits(self, j: int, value: float) -> bool:
        return (
            j >= len(self.lower)
            or float(self.lower[j]) + JOINT_MARGIN <= value <= float(self.upper[j]) - JOINT_MARGIN
        )

    def to_q(self, q: np.ndarray) -> str:
        try:
            return _refusal(
                self.ctx.tool(
                    "robot.move_to_joints",
                    joint_config={"positions": [float(v) for v in q]},
                    **self.kw,
                )
            )
        except Exception as exc:
            return str(exc) or type(exc).__name__

    def ik(self, point: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, str]:
        try:
            reply = self.ctx.tool(
                "robot.solve_ik", pose={"position": _vec(point), "rotation": quat(rot)}, **self.kw
            )
        except Exception as exc:
            return np.zeros(0), f"IK refused ({exc})"
        reply = reply if isinstance(reply, dict) else {}
        if reply.get("success") is False or reply.get("within_servo_tolerance") is False:
            return np.zeros(0), "no IK solution"
        q = (reply.get("joint_config") or {}).get("positions")
        q = np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)
        if q.size == 0:
            return q, "the IK reply has no joints"
        bad = [j for j in range(q.size) if not self.fits(j, float(q[j]))]
        if bad:
            return q, f"joint {bad[0] + 1} would be at {q[bad[0]]:+.2f} rad, at its limit"
        return q, ""

    def step(self, point: np.ndarray, rot: np.ndarray) -> str:
        q_now = self.q()
        q, why = self.ik(point, rot)
        if why:
            return why
        if q_now.size == q.size and q.size > 1:
            jump = float(np.max(np.abs(q[:-1] - q_now[:-1])))
            if jump > self.jump:
                return f"a joint would jump {jump:.2f} rad (limit {self.jump:.2f})"
        why = self.to_q(q)
        if why:
            return why
        q_after = self.q()
        if q_after.size == q.size:
            off = float(np.max(np.abs(q_after - q)))
            if off > JOINT_SETTLE:
                return f"the joints stopped {off:.2f} rad short"
        return ""

    def line(self, target: np.ndarray, rot: np.ndarray, turn: bool = False) -> str:
        """A straight move in chained IK steps of at most IK_STEP; with ``turn`` the rotation is turned toward
        ``rot`` along the way in steps of at most TURN_STEP_DEG, else held at ``rot``."""
        p0, r0 = self.pose()
        rel = rotvec(np.asarray(rot) @ r0.T) if turn else np.zeros(3)
        n = max(
            1,
            int(math.ceil(float(np.linalg.norm(target - p0)) / IK_STEP)),
            int(math.ceil(math.degrees(float(np.linalg.norm(rel))) / TURN_STEP_DEG)),
        )
        for i in range(1, n + 1):
            r_i = (
                axis_angle(rel, math.degrees(float(np.linalg.norm(rel))) * i / n) @ r0
                if turn
                else rot
            )
            why = self.step(p0 + (target - p0) * i / n, r_i)
            if why:
                return f"{why} (step {i} of {n})"
        return ""

    def grip(self, w: float) -> str:
        try:
            return _refusal(self.ctx.tool("robot.set_grip", width_m=float(w), **self.kw))
        except Exception as exc:
            return str(exc) or type(exc).__name__

    def width(self, reply: Any = None) -> float:
        try:
            fraction = reply.get("position") if isinstance(reply, dict) else None
            if fraction is None:
                fraction = self.ctx.tool("robot.get_gripper", **self.kw)["position"]
            if self.at_open <= 0.0:
                return -1.0
            return self.at_closed + max(0.0, min(1.0, float(fraction))) * (
                self.at_open - self.at_closed
            )
        except Exception as exc:
            logger.info(
                "standing-a-lying-domino: could not read arm %d's gripper (%s)", self.id, exc
            )
            return -1.0

    def close(self) -> float:
        try:
            return self.width(self.ctx.tool("robot.close_gripper", **self.kw))
        except Exception as exc:
            logger.info("standing-a-lying-domino: arm %d did not close (%s)", self.id, exc)
            return -1.0

    def roll(self, deg: float, step_deg: float, local_approach: np.ndarray) -> str:
        """Turn the hand ``deg`` about its approach on the last joint, in steps; the sign of the joint is read
        from the first step."""
        q0 = self.q()
        if q0.size == 0:
            return "no joint reading"
        last = q0.size - 1
        _, r0 = self.pose()
        options = [deg + m * 360.0 for m in (-1, 0, 1)]
        n = max(1, int(math.ceil(abs(deg) / max(1.0, step_deg))))
        sign = 1.0
        for i in range(1, n + 1):
            goal = q0.copy()
            goal[last] += sign * math.radians(deg) * i / n
            if not self.fits(last, float(goal[last])):
                alt = [o for o in options if self.fits(last, q0[last] + sign * math.radians(o))]
                return f"the last joint has no room to roll {deg:+.0f} deg" + (
                    f" (a turn of {alt[0]:+.0f} would fit)" if alt else ""
                )
            why = self.to_q(goal)
            if why:
                return why
            if i == 1:
                _, r1 = self.pose()
                turned = float(
                    rotvec(r1 @ r0.T) @ (r0 @ np.asarray(local_approach, dtype=np.float64))
                )
                if turned * deg < 0.0:
                    sign = -1.0
        return ""

    def base(self) -> np.ndarray | None:
        """Where this arm's first joint stands (its vertical axis), from a small test turn of it."""
        q0 = self.q()
        if q0.size == 0:
            return None
        d = 0.1 if self.fits(0, float(q0[0]) + 0.1) else -0.1
        p0, _ = self.pose()
        goal = q0.copy()
        goal[0] += d
        why = self.to_q(goal)
        p1, _ = self.pose()
        q1 = self.q()
        self.to_q(q0)
        if why:
            return None
        dq = float(q1[0] - q0[0]) if q1.size == q0.size else d
        if abs(dq) < 0.02:
            return None
        c, s_ = math.cos(dq), math.sin(dq)
        m = np.array([[c - 1.0, -s_], [s_, c - 1.0]])
        v = np.linalg.solve(m, (p1 - p0)[:2])
        return np.array([*(p0[:2] - v), p0[2]])


# ----------------------------------------------------------------------------- the skill


def run(
    ctx: NodeContext,
    table_z: float,
    tile_center: dict[str, float],
    long_axis: dict[str, float],
    staging: dict[str, float] | None = None,
    strategy: str = "table_pivot",
    tile_size: dict[str, float] | None = None,
    grasp_height: float = GRASP_HEIGHT,
    grip_offset: float = GRIP_OFFSET,
    lift_height: float = LIFT_HEIGHT,
    release_gap: float = RELEASE_GAP,
    grip_clearance: float = GRIP_CLEARANCE,
    retreat: float = RETREAT,
    min_tool_height: float = MIN_TOOL_HEIGHT,
    facing_step_deg: float = FACING_STEP_DEG,
    pitch_step_deg: float = PITCH_STEP_DEG,
    pitch_joint_jump: float = PITCH_JOINT_JUMP,
    air_pitch_deg: float = AIR_PITCH_DEG,
    pivot_step_deg: float = PIVOT_STEP_DEG,
    press: float = PRESS,
    probe_turn: float = PROBE_TURN,
    stage_step: float = STAGE_STEP,
    stage_reach: float = STAGE_REACH,
    stage_attempts: int = STAGE_ATTEMPTS,
    hinge_offset: float = HINGE_OFFSET,
    hinge_press: float = HINGE_PRESS,
    helper_arm_id: int = 1,
    handover_point: dict[str, float] | None = None,
    handover_height: float = HANDOVER_HEIGHT,
    helper_offset: float = HELPER_OFFSET,
    helper_bite: float = HELPER_BITE,
    top_bite: float = TOP_BITE,
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
        "strategy": str(strategy),
        "pitch_joint": -1,
        "picked": False,
        "released": False,
        "grasp_height": float(grasp_height),
        "grip_width": -1.0,
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

    if strategy not in STRATEGIES:
        return finish("unreachable", f"strategy {strategy!r} is not one of {', '.join(STRATEGIES)}")
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

    def ik(point: np.ndarray, rot: np.ndarray) -> dict[str, Any] | None:
        """The IK preview's reply, or None when it is refused or off the servo's tolerance."""
        if not check_reach:
            return {}
        try:
            reply = ctx.tool(
                "robot.solve_ik", pose={"position": _vec(point), "rotation": quat(rot)}, **on_arm
            )
        except Exception as exc:
            logger.info("standing-a-lying-domino: IK preview refused (%s)", exc)
            return None
        reply = reply if isinstance(reply, dict) else {}
        if reply.get("success") is False or reply.get("within_servo_tolerance") is False:
            return None
        return reply

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

    notes: list[str] = []
    roll_gain = [1.0]

    def turn_last(
        err_fn: Any, at: np.ndarray, key: str, period: float = 360.0, step_deg: float = 0.0
    ) -> np.ndarray:
        """Turn the last joint until ``err_fn(rotation)`` (deg about the approach) is within roll_tolerance_deg,
        re-centring over ``at`` when the turn moved the hand; a turn by ``period`` is the same, so the smallest
        one within the joint's limits is taken. Returns the hand's rotation."""
        r = rot_now()
        for _ in range(TURN_TRIES):
            err = float(err_fn(r))
            if abs(err) <= float(roll_tolerance_deg):
                break
            q = joints()
            if q.size == 0:
                notes.append("no joint reading, the wrist was not turned")
                break
            options = sorted((err + m * float(period) for m in (-2, -1, 0, 1, 2)), key=abs)
            gain = roll_gain[0]
            ok = [o for o in options if fits(q.size - 1, q[-1] + math.radians(o) / gain)]
            if not ok:
                notes.append(f"the last joint has no room to turn {err:+.0f} deg")
                break
            command = math.radians(ok[0]) / gain
            n_turn = (
                max(1, int(math.ceil(abs(math.degrees(command)) / step_deg)))
                if step_deg > 0.0
                else 1
            )
            for i in range(1, n_turn + 1):
                goal = q.copy()
                goal[-1] += command * i / n_turn
                why = to_joints(goal)
                if why:
                    break
            if why:
                notes.append(f"wrist turn: {why}")
                break
            before, r = r, rot_now()
            moved = turn_about(r @ local_approach, before @ local_close, r @ local_close)
            out[key] += moved  # type: ignore[literal-required]
            if abs(command) > math.radians(2.0) and 0.5 < abs(math.radians(moved) / command) < 1.5:
                roll_gain[0] = math.radians(moved) / command
            if float(np.linalg.norm(_xyz(here()) - at)) > CENTRE_TOLERANCE:
                why = go("robot.go_to_pose", at, quat(r))
                if why:
                    notes.append(f"re-centring after the wrist turn: {why}")
                    break
                r = rot_now()
        return r

    held: dict[str, Any] = {}

    def pick(
        grasp_point: np.ndarray, g_rot: np.ndarray, e0: np.ndarray, lift: bool = True
    ) -> Output | None:
        """Over the grasp point, square the jaws, straight down, close, lift. None when the tile is held; the
        measured grasp height, the hand's rotation at the close and the up end in the hand's frame go in ``held``."""
        why = set_width(grip_open)
        if why:
            return finish(
                "unreachable", f"the jaws did not open to {grip_open * 1e3:.0f} mm ({why})"
            )
        now = _xyz(here())
        if now[2] < lift_z - 1.0e-3:
            why = go(
                "robot.go_to_pose_cartesian",
                np.array([now[0], now[1], lift_z]),
                here()["rotation"],
            )
            if why:
                return finish("unreachable", f"rise before the pick: {why}")
        above = np.array([*grasp_point[:2], lift_z])
        why = go("robot.go_to_pose", above, quat(g_rot))
        if why:
            return finish("unreachable", f"over the tile: {why}")
        r_now = turn_last(lambda r: jaw_error(r, g_rot), above, "roll_turn_deg", period=180.0)
        out["roll_turn_deg"] = 0.0
        err = jaw_error(r_now, g_rot)
        if abs(err) > float(roll_tolerance_deg):
            return finish(
                "unreachable",
                f"the jaws stayed {err:+.0f} deg off the tile's width axis ({'; '.join(notes) or 'not turned'})",
            )
        why = move("robot.go_to_pose_cartesian", grasp_point, quat(r_now))
        reached = _xyz(here())
        high = float(reached[2] - grasp_point[2])
        if high > float(descent_tolerance) or float(
            np.linalg.norm(reached[:2] - grasp_point[:2])
        ) > float(reach_tolerance):
            go("robot.go_to_pose_cartesian", np.array([*reached[:2], lift_z]), quat(r_now))
            return finish(
                "ungraspable",
                f"the descent to the tile stopped {high * 1e3:.1f} mm high ({why or 'stalled'}): a fingertip on "
                "the tile, or it is not where it was seen; nothing closed",
            )
        out["grip_width"] = width(ctx.tool("robot.close_gripper", **on_arm))
        why = holding(out["grip_width"])
        if why:
            set_width(grip_open)
            go("robot.go_to_pose_cartesian", np.array([*reached[:2], lift_z]), quat(r_now))
            return finish("ungraspable", f"grasp: {why}; the hand is open over the tile")
        held["g"] = float(reached[2] - table_z)
        out["grasp_height"] = held["g"]
        out["picked"] = True
        r_grip = rot_now()
        held["r"] = r_grip
        held["e"] = r_grip.T @ e0  # the end that goes up, in the hand's frame
        held["at"] = reached
        if not lift:
            return None
        why = go("robot.go_to_pose_cartesian", np.array([*reached[:2], lift_z]), quat(r_grip))
        if why:
            return finish("unreachable", f"lift after the grasp: {why}; still holding")
        why = holding(width())
        if why:
            return finish("dropped", f"after the lift: {why}")
        return None

    def check(xy: np.ndarray, facing: float, hand_note: str) -> Output:
        """Out of the cameras' way, then the spot from depth."""
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
        v = judge_standing(cams, xy, facing, table_z, size)
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

    def pose_pitch_run() -> Output:
        """The quarter turn in pose steps about the level axis through the tool centre, over the staging spot;
        the facing is chosen by an IK preview of the grasp, every pitch step and the release."""
        # --- the plan, previewed before anything moves: a top-down grasp, then for each facing (the pitched
        # approach's heading) the pitched hand over the staging spot and at the release, the nearer ones first.
        cur = rot_now()
        grasp_rots.sort(key=lambda g: abs(jaw_error(cur, g)))
        step = max(1.0, float(facing_step_deg))
        facings = sorted(
            (wrap(theta + i * step, False) for i in range(int(round(360.0 / step)))),
            key=lambda f: (abs(wrap(f - theta, True)), abs(wrap(f - theta, False))),
        )
        n_pitch = max(1, int(math.ceil(90.0 / max(1.0, float(pitch_step_deg)))))

        def level_of(f: float) -> np.ndarray:
            """The level axis the quarter turn is about, for facing ``f``: the approach goes from down to ``f``."""
            return np.cross(np.array([0.0, 0.0, -1.0]), heading(f))

        def pitch_path(f: float, r_f: np.ndarray, high: np.ndarray) -> str:
            """What fails for facing ``f``, or an empty string when every pitch step over ``high`` solves (from the
            top-down hand to ``r_f``) and no joint but the last moves more than pitch_joint_jump between two steps."""
            prev = np.zeros(0)
            for i in range(n_pitch + 1):
                deg = 90.0 * i / n_pitch
                reply = ik(high, axis_angle(level_of(f), deg - 90.0) @ r_f)
                if reply is None:
                    return f"pitch at {deg:.0f} deg"
                q = (reply.get("joint_config") or {}).get("positions")
                q = np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)
                if q.size > 1 and prev.size == q.size:
                    jump = float(np.max(np.abs(q[:-1] - prev[:-1])))
                    if jump > float(pitch_joint_jump):
                        return f"pitch at {deg:.0f} deg: a joint jumps {jump:.2f} rad"
                prev = q
            return ""

        passing: list[float] = []
        tried: list[str] = []
        g_rot, forward = grasp_rots[0], True
        for g in grasp_rots:
            grasp_ok: dict[bool, bool] = {}
            for psi in facings:
                u = heading(psi)
                fwd = float(heading(theta) @ u) >= 0.0
                if passing and fwd != forward:
                    continue
                e0 = heading(theta) if fwd else heading(theta + 180.0)
                if fwd not in grasp_ok:
                    grasp_ok[fwd] = (
                        ik(np.array([*(center[:2] + s * e0[:2]), table_z + gh]), g) is not None
                    )
                    if not grasp_ok[fwd]:
                        tried.append(f"{psi:.0f}: grasp")
                if not grasp_ok[fwd]:
                    continue
                r_f = pitched_rotation(u, local_approach, local_close, g.T @ e0)
                release = plan_release(spot, psi, table_z, size, gh, s, float(release_gap))
                if ik(release, r_f) is None:
                    tried.append(f"{psi:.0f}: release")
                    continue
                why = pitch_path(psi, r_f, np.array([*release[:2], lift_z]))
                if why:
                    tried.append(f"{psi:.0f}: {why}")
                    continue
                if not passing:
                    g_rot, forward = g, fwd
                passing.append(psi)
            if passing:
                break
        if not passing:
            return finish(
                "unreachable",
                "no facing passes the IK preview for the grasp, the pitch over the staging spot and the "
                f"release ({', '.join(tried[:12])}{', ...' if len(tried) > 12 else ''})",
            )
        e0 = heading(theta) if forward else heading(theta + 180.0)
        grasp_point = np.array([*(center[:2] + s * e0[:2]), table_z + gh])
        psi = passing[0]
        out["facing_deg"] = float(psi)
        out["standing_tile"] = standing_pose(spot, psi, table_z, size)
        why = pick(grasp_point, g_rot, e0)
        if why is not None:
            return why
        g_meas, r_grip, e_local, reached = held["g"], held["r"], held["e"], held["at"]

        # --- the facing: the first that passed whose turn about the vertical fits the last joint as it is now.
        q = joints()

        def yaw_fits(f: float) -> bool:
            need = turn_about(r_grip @ local_approach, r_grip @ e_local, heading(f))
            return q.size == 0 or any(
                fits(q.size - 1, q[-1] + math.radians(need + m * 360.0) / roll_gain[0])
                for m in (-1, 0, 1)
            )

        roomy = [f for f in passing if yaw_fits(f)]
        if not roomy:
            return finish(
                "unreachable",
                f"the last joint has no room to turn the held tile to any reachable facing "
                f"({', '.join(f'{f:.0f}' for f in passing[:8])}); still holding",
            )
        psi = roomy[0]
        u = heading(psi)
        out["facing_deg"] = float(psi)
        out["standing_tile"] = standing_pose(spot, psi, table_z, size)

        # --- over the staging spot, the held tile turned about the vertical so the end that goes up points along the
        # facing, then the quarter turn about the level axis through the tool centre, in steps.
        release = plan_release(spot, psi, table_z, size, g_meas, s, float(release_gap))
        high_pt = np.array([*release[:2], lift_z])
        why = go("robot.go_to_pose", high_pt, quat(r_grip))
        if why:
            return finish("unreachable", f"to over the staging spot: {why}; still holding")
        r_now = turn_last(
            lambda r: turn_about(r @ local_approach, r @ e_local, u), high_pt, "yaw_turn_deg"
        )
        err = turn_about(r_now @ local_approach, r_now @ e_local, u)
        if abs(err) > float(roll_tolerance_deg):
            return finish(
                "unreachable",
                f"the held tile stayed {err:+.0f} deg off the facing ({'; '.join(notes) or 'not turned'}); "
                "still holding",
            )
        a_before = r_now @ local_approach
        level = level_of(psi)
        n = n_pitch
        for i in range(1, n + 1):
            target = (
                pitched_rotation(u, local_approach, local_close, e_local)
                if i == n
                else axis_angle(level, 90.0 * i / n) @ r_now
            )
            why = go("robot.go_to_pose", high_pt, quat(target))
            if why:
                return finish("unreachable", f"the pitch, step {i} of {n}: {why}; still holding")
            why = holding(width())
            if why:
                return finish("dropped", f"during the pitch (step {i} of {n}): {why}")
        r_now = rot_now()
        out["pitch_turn_deg"] = angle_between(a_before, r_now @ local_approach)
        a_now = r_now @ local_approach
        if angle_between(a_now, u) > 2.0 * float(roll_tolerance_deg):
            return finish(
                "unreachable",
                f"after the pitch the hand points {np.round(a_now, 2).tolist()}, not along the "
                f"{psi:.0f} deg facing; still holding",
            )

        # --- the roll read back (the tile's up end about the approach), finished on the last joint; the lean
        # checked before going down; straight down.
        r_now = turn_last(
            lambda r: turn_about(r @ local_approach, r @ e_local, z_up), high_pt, "roll_turn_deg"
        )
        tilt = angle_between(r_now @ e_local, z_up)
        out["tilt_at_release_deg"] = tilt
        if tilt > float(tilt_tolerance_deg):
            return finish(
                "unreachable",
                f"over the staging spot the tile's long axis leans {tilt:.1f} deg from vertical"
                + (f" ({'; '.join(notes)})" if notes else "")
                + "; still holding",
            )
        why = move("robot.go_to_pose_cartesian", release, quat(r_now))
        reached = _xyz(here())
        r_now = rot_now()
        high = float(reached[2] - release[2])
        tilt = angle_between(r_now @ e_local, z_up)
        out["tilt_at_release_deg"] = tilt
        if high > float(descent_tolerance) or float(
            np.linalg.norm(reached[:2] - release[:2])
        ) > float(reach_tolerance):
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
                    "unreachable",
                    f"the jaws did not open at the staging spot ({exc}); still holding",
                )
        out["released"] = True
        back = reached - float(retreat) * u
        why = go("robot.go_to_pose_cartesian", back, quat(r_now))
        if not why:
            why = go("robot.go_to_pose_cartesian", np.array([*back[:2], lift_z]), quat(r_now))
        if why:
            return finish("unreachable", f"released, but the retreat: {why}")

        hand_note = (
            f"set down {np.linalg.norm(landed - spot) * 1e3:.1f} mm from the spot, facing {psi:.0f} deg, "
            f"pitched {out['pitch_turn_deg']:.0f} deg, leaning {tilt:.1f} deg at the release"
        ) + (f"; {'; '.join(notes)}" if notes else "")
        if not verify:
            return finish("standing", hand_note + "; not checked from depth")

        return check(spot, psi, hand_note)

    def ik_q(point: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, str]:
        """The IK solution for a pose (the solver starts from where the arm is), and why it is refused: no
        solution, off the servo's tolerance, or a joint within JOINT_MARGIN of its limit."""
        try:
            reply = ctx.tool(
                "robot.solve_ik", pose={"position": _vec(point), "rotation": quat(rot)}, **on_arm
            )
        except Exception as exc:
            return np.zeros(0), f"IK refused ({exc})"
        reply = reply if isinstance(reply, dict) else {}
        if reply.get("success") is False or reply.get("within_servo_tolerance") is False:
            return np.zeros(0), "no IK solution"
        q = (reply.get("joint_config") or {}).get("positions")
        q = np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)
        if q.size == 0:
            return q, "the IK reply has no joints"
        bad = [j for j in range(q.size) if not fits(j, float(q[j]))]
        if bad:
            return q, f"joint {bad[0] + 1} would be at {q[bad[0]]:+.2f} rad, at its limit"
        return q, ""

    def step_to(point: np.ndarray, rot: np.ndarray) -> str:
        """One chained IK step, driven in joint space: refused before moving when a joint but the last would
        move more than pitch_joint_jump, judged by the joints read back."""
        q_now = joints()
        q, why = ik_q(point, rot)
        if why:
            return why
        if q_now.size == q.size and q.size > 1:
            jump = float(np.max(np.abs(q[:-1] - q_now[:-1])))
            if jump > float(pitch_joint_jump):
                return f"a joint would jump {jump:.2f} rad (limit {float(pitch_joint_jump):.2f})"
        why = to_joints(q)
        if why:
            return why
        q_after = joints()
        if q_after.size == q.size:
            off = float(np.max(np.abs(q_after - q)))
            if off > JOINT_SETTLE:
                return f"the joints stopped {off:.2f} rad short"
        return ""

    def line_to(target: np.ndarray, rot: np.ndarray) -> str:
        """A straight move cut into chained IK steps of at most IK_STEP."""
        here_now = _xyz(here())
        n = max(1, int(math.ceil(float(np.linalg.norm(target - here_now)) / IK_STEP)))
        for i in range(1, n + 1):
            why = step_to(here_now + (target - here_now) * i / n, rot)
            if why:
                return f"{why} (step {i} of {n})"
        return ""

    def table_pivot_run() -> Output:
        """Learn the wrist joint that pitches the hand about a level axis, pick, turn the held tile about the
        vertical into that joint's plane, pitch AIR_PITCH_DEG on that joint alone, set the tile's lower edge on
        the table at the spot, turn the rest about that edge by chained IK, release, back off."""
        out["strategy"] = "table_pivot"
        cur = rot_now()
        order = sorted(grasp_rots, key=lambda g: abs(jaw_error(cur, g)))
        hover = np.array([*center[:2], lift_z])
        g_rot = next((g for g in order if ik(hover, g) is not None), None)
        if g_rot is None:
            return finish("unreachable", "no wrist solution over the tile passes the IK preview")
        why = set_width(grip_open)
        if why:
            return finish(
                "unreachable", f"the jaws did not open to {grip_open * 1e3:.0f} mm ({why})"
            )
        now = _xyz(here())
        if now[2] < lift_z - 1.0e-3:
            why = go(
                "robot.go_to_pose_cartesian",
                np.array([now[0], now[1], lift_z]),
                here()["rotation"],
            )
            if why:
                return finish("unreachable", f"rise before the pick: {why}")
        why = go("robot.go_to_pose", hover, quat(g_rot))
        if why:
            return finish("unreachable", f"over the tile: {why}")

        # --- learn, with the hand empty over the tile: test-turn the joints from the hand back; the pitch joint is
        # the first whose axis is level and whose quarter turn fits its limits.
        q0 = joints()
        if q0.size < 3:
            return finish("unreachable", "no joint reading: the pitch joint cannot be found")
        r0 = rot_now()
        a0 = r0 @ local_approach
        axes: dict[int, np.ndarray] = {}
        pitch_j, sign, rate, best_rate, best_margin = -1, 0.0, 0.0, 0.0, -1.0
        for j in range(q0.size - 1, 0, -1):
            d = float(probe_turn) if fits(j, q0[j] + float(probe_turn)) else -float(probe_turn)
            test = q0.copy()
            test[j] += d
            why = to_joints(test)
            r1, q1 = rot_now(), joints()
            back = to_joints(q0)
            if why or back:
                return finish("unreachable", f"test turn of joint {j + 1}: {why or back}")
            dq = float(q1[j] - q0[j]) if q1.size == q0.size else d
            if abs(dq) < 0.3 * abs(d):
                continue
            axes[j] = rotvec(r1 @ r0.T) / dq
            if j == q0.size - 1:
                continue
            w = axes[j]
            rate = float(np.linalg.norm(w))
            if rate < 0.5 or abs(w[2]) / rate > LEVEL_TOLERANCE:
                continue
            quarter = (math.pi / 2.0) / rate
            for sg in (1.0, -1.0):
                end_q = q0[j] + sg * quarter
                margin = min(end_q - float(lower[j]), float(upper[j]) - end_q)
                if margin >= JOINT_MARGIN and margin > best_margin:
                    pitch_j, sign, best_rate, best_margin = j, sg, rate, margin
            if best_margin >= PITCH_ROOM:
                break
        rate = best_rate
        out["pitch_joint"] = pitch_j + 1
        if pitch_j < 0:
            return finish(
                "unreachable",
                "no joint turns the hand about a level axis with room for a quarter turn "
                f"(tested {', '.join(str(j + 1) for j in sorted(axes))})",
            )
        last = q0.size - 1
        if last in axes and abs(float(axes[last] @ a0)) > 0.5:
            roll_gain[0] = float(axes[last] @ a0)
        w_hat = sign * axes[pitch_j] / rate
        to_out = np.cross(w_hat, a0)
        to_out[2] = 0.0
        if float(np.linalg.norm(to_out)) < 0.5:
            return finish("unreachable", "the pitch joint would not bring the approach level")
        to_out /= float(np.linalg.norm(to_out))

        # The end that goes up: the one whose turn about the vertical into the pitch plane fits the last joint.
        def yaw_need(e: np.ndarray) -> float | None:
            need = turn_about(a0, e, to_out)
            room = [
                o
                for o in sorted((need + m * 360.0 for m in (-1, 0, 1)), key=abs)
                if fits(last, q0[last] + math.radians(o) / roll_gain[0])
            ]
            return room[0] if room else None

        ends = [(e, yaw_need(e)) for e in (heading(theta), heading(theta + 180.0))]
        ends = [(e, n) for e, n in ends if n is not None]
        if not ends:
            return finish(
                "unreachable",
                "the last joint has no room to turn the tile into the pitch joint's plane",
            )
        e0 = min(ends, key=lambda en: abs(en[1]))[0]

        # --- rehearse the upright hand with the hand empty: the pitch joint alone turns it a quarter, the solver
        # (starting from there) previews the upright hand at each spot, the turn is undone. Previewed from the
        # top-down arm, the upright hand failed to solve everywhere in a YAM station twin.
        n_q = max(1, int(math.ceil(90.0 / max(1.0, float(pitch_step_deg)))))
        for i in range(1, n_q + 1):
            goal = q0.copy()
            goal[pitch_j] += sign * math.radians(90.0 * i / n_q) / rate
            why = to_joints(goal)
            if why:
                to_joints(q0)
                return finish(
                    "unreachable", f"rehearsing the quarter turn, step {i} of {n_q}: {why}"
                )
        r_f = rot_now()
        a_f = r_f @ local_approach
        if staging:
            cands = [spot]
        else:
            n_out = (
                int(math.floor(float(stage_reach) / max(1.0e-3, float(stage_step)) + 1.0e-9)) + 1
            )
            cands = [center[:2] + k * float(stage_step) * to_out[:2] for k in range(n_out)]
        reach_ok: list[np.ndarray] = []
        missed: list[str] = []
        for xy in cands:
            p_f = np.array([*xy, table_z + tl / 2.0]) - (gh - tt / 2.0) * a_f + s * z_up
            _, why = ik_q(p_f, r_f)
            if why:
                missed.append(f"({xy[0]:.3f}, {xy[1]:.3f}): {why}")
            else:
                reach_ok.append(xy)
        for i in range(n_q - 1, -1, -1):
            goal = q0.copy()
            goal[pitch_j] += sign * math.radians(90.0 * i / n_q) / rate
            to_joints(goal)
        if not reach_ok:
            return finish(
                "unreachable",
                "the upright hand reaches no spot in the rehearsal "
                f"({'; '.join(missed[:4])}{', ...' if len(missed) > 4 else ''}); nothing picked",
            )
        grasp_point = np.array([*(center[:2] + s * e0[:2]), table_z + gh])
        why = pick(grasp_point, g_rot, e0)
        if why is not None:
            return why
        g_meas, e_local = held["g"], held["e"]

        # --- the held tile turned about the vertical so its up end points where the pitch takes the approach.
        p_lift = _xyz(here())
        r_now = turn_last(
            lambda r: turn_about(r @ local_approach, r @ e_local, to_out),
            p_lift,
            "yaw_turn_deg",
            step_deg=float(pitch_step_deg),
        )
        err = turn_about(r_now @ local_approach, r_now @ e_local, to_out)
        if abs(err) > float(roll_tolerance_deg):
            return finish(
                "unreachable",
                f"the held tile stayed {err:+.0f} deg off the pitch plane ({'; '.join(notes) or 'not turned'}); "
                "still holding",
            )

        # --- the pitch in the air, on the pitch joint alone, in steps.
        a_before = r_now @ local_approach
        q_start = joints()
        air = max(0.0, min(90.0, float(air_pitch_deg)))
        n_air = int(math.ceil(air / max(1.0, float(pitch_step_deg))))
        for i in range(1, n_air + 1):
            goal = q_start.copy()
            goal[pitch_j] += sign * math.radians(air * i / n_air) / rate
            why = to_joints(goal)
            if why:
                return finish(
                    "unreachable", f"the pitch, step {i} of {n_air}: {why}; still holding"
                )
            why = holding(width())
            if why:
                return finish("dropped", f"during the pitch (step {i} of {n_air}): {why}")

        # --- where the tile's lower edge goes: from the hand, the tile sits (grasp height - half its thickness)
        # ahead of the tool centre along the approach and grip_offset behind it along its up end.
        r = rot_now()
        p = _xyz(here())
        a, cl, e = r @ local_approach, r @ local_close, r @ e_local
        cl = cl / (np.linalg.norm(cl) or 1.0)
        c_t = p + (g_meas - tt / 2.0) * a - s * e
        b = c_t - (tl / 2.0) * e
        pivot = min((b + k * (tt / 2.0) * a for k in (1.0, -1.0)), key=lambda v: float(v[2]))
        phi = turn_about(cl, e, z_up)
        r_phi = axis_angle(cl, phi)
        final_c = pivot + r_phi @ (c_t - pivot)

        def plan_at(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            shift = np.array([*(xy - final_c[:2]), table_z + SETDOWN_GAP - pivot[2]])
            edge = pivot + shift - (SETDOWN_GAP + float(press)) * z_up
            return shift, p + shift, edge

        tried: list[str] = []
        n_piv = max(1, int(math.ceil(abs(phi) / max(1.0, float(pivot_step_deg)))))
        stood = None
        for xy in reach_ok[: int(stage_attempts)]:
            shift, down, edge = plan_at(xy)
            here_now = _xyz(here())
            why = line_to(np.array([down[0], down[1], max(here_now[2], down[2])]), r)
            if not why:
                why = line_to(down, r)
            if why:
                tried.append(f"({xy[0]:.3f}, {xy[1]:.3f}) set-down: {why}")
                continue
            path = [joints()]
            for i in range(1, n_piv + 1):
                turn = axis_angle(cl, phi * i / n_piv)
                why = step_to(edge + turn @ (down - pivot - shift), turn @ r)
                if why:
                    why = f"step {i} of {n_piv}: {why}"
                    break
                path.append(joints())
                logger.info(
                    "standing-a-lying-domino: upright step %d of %d at (%.3f, %.3f): hand %s, jaws %.1f mm",
                    i,
                    n_piv,
                    float(xy[0]),
                    float(xy[1]),
                    np.round(_xyz(here()), 4).tolist(),
                    width() * 1e3,
                )
            if not why:
                stood = xy
                break
            tried.append(f"({xy[0]:.3f}, {xy[1]:.3f}) upright {why}")
            # Back down flat the way it came, lift a little, and try the next spot.
            for q in reversed(path[:-1]):
                if q.size:
                    to_joints(q)
            line_to(down + np.array([0.0, 0.0, 0.01]), r)
        if stood is None:
            return finish(
                "unreachable",
                f"no spot let the hand stand the tile up ({'; '.join(tried[:6]) or 'none tried'}); still holding",
            )
        xy = stood
        out["standing_tile"] = standing_pose(
            xy, math.degrees(math.atan2(to_out[1], to_out[0])), table_z, size
        )
        if tried:
            notes.append(f"passed over {len(tried)} spot(s): {'; '.join(tried[:3])}")
        why = holding(width())
        if why:
            return finish("dropped", f"turning upright on the table: {why}")
        r_now = rot_now()
        out["pitch_turn_deg"] = angle_between(a_before, r_now @ local_approach)
        tilt = angle_between(r_now @ e_local, z_up)
        if tilt > float(tilt_tolerance_deg):
            p_now = _xyz(here())
            r_now = turn_last(
                lambda rr: turn_about(rr @ local_approach, rr @ e_local, z_up),
                p_now,
                "roll_turn_deg",
            )
            tilt = angle_between(r_now @ e_local, z_up)
        out["tilt_at_release_deg"] = tilt
        if tilt > float(tilt_tolerance_deg):
            return finish(
                "unreachable",
                f"standing on the table the tile's long axis leans {tilt:.1f} deg from vertical"
                + (f" ({'; '.join(notes)})" if notes else "")
                + "; still holding",
            )

        # --- release, back off along the approach, rise.
        p_now = _xyz(here())
        a_now = r_now @ local_approach
        e_now = r_now @ e_local
        landed = p_now + (g_meas - tt / 2.0) * a_now - s * e_now
        out["release_position"] = _vec(np.array([*landed[:2], table_z]))
        facing = math.degrees(math.atan2(a_now[1], a_now[0]))
        out["facing_deg"] = float(facing)
        out["standing_tile"] = standing_pose(landed[:2], facing, table_z, size)
        why = set_width(grip_open)
        if why:
            try:
                ctx.tool("robot.open_gripper", **on_arm)
                notes.append(f"opened fully ({why})")
            except Exception as exc:
                return finish(
                    "unreachable", f"the jaws did not open on the table ({exc}); still holding"
                )
        out["released"] = True
        level = np.array([*a_now[:2], 0.0])
        level = level / (np.linalg.norm(level) or 1.0)
        back = p_now - float(retreat) * level
        why = line_to(back, r_now)
        if not why:
            why = line_to(np.array([*back[:2], max(lift_z, back[2])]), r_now)
        if why:
            return finish("unreachable", f"released, but the retreat: {why}")
        hand_note = (
            f"pitch joint {pitch_j + 1}; set down {np.linalg.norm(landed[:2] - xy) * 1e3:.1f} mm from the spot, "
            f"facing {facing:.0f} deg, pitched {out['pitch_turn_deg']:.0f} deg, leaning {tilt:.1f} deg at the release"
        ) + (f"; {'; '.join(notes)}" if notes else "")
        if not verify:
            return finish("standing", hand_note + "; not checked from depth")
        return check(landed[:2], facing, hand_note)

    def hinge_tip_run() -> Output:
        """The pinch across the width near one end is a pin: the hand stays top-down and its tool centre moves on
        an arc about the far end's bottom edge, so the tile turns in the pads until it stands on that end."""
        out["strategy"] = "hinge_tip"
        cur = rot_now()
        order = sorted(grasp_rots, key=lambda g: abs(jaw_error(cur, g)))
        h = max(0.0, min(float(hinge_offset), tl / 2.0))
        tried: list[str] = []
        for e_up in (heading(theta), heading(theta + 180.0)):
            grasp_point = np.array([*(center[:2] + h * e_up[:2]), table_z + gh])
            for g in order:
                if (
                    ik(grasp_point, g) is None
                    or ik(np.array([*grasp_point[:2], lift_z]), g) is None
                ):
                    tried.append(f"end {math.degrees(math.atan2(e_up[1], e_up[0])):.0f}: grasp")
                    continue
                break
            else:
                continue
            break
        else:
            return finish(
                "unreachable",
                f"no grasp near either end passes the IK preview ({', '.join(tried)})",
            )
        why = pick(grasp_point, g, e_up, lift=False)
        if why is not None:
            return why
        g_meas, r_grip, at = held["g"], held["r"], held["at"]
        # The pin: the middle of the pads' bite on the tile's edge, under the tool centre.
        bite_low = max(0.0, g_meas - tip_reach)
        pin0 = np.array([*at[:2], table_z + (bite_low + tt) / 2.0])
        foot = np.array([*(center[:2] - (tl / 2.0) * e_up[:2]), table_z])
        k_axis = np.cross(e_up, z_up)
        k_axis = k_axis / (np.linalg.norm(k_axis) or 1.0)
        arm_vec = pin0 - foot
        radius = float(np.linalg.norm(arm_vec))
        arm_vec = arm_vec * max(0.0, radius - float(hinge_press)) / (radius or 1.0)
        hand_off = at - pin0
        n_arc = max(1, int(math.ceil(90.0 / max(1.0, float(pivot_step_deg)))))
        for i in range(1, n_arc + 1):
            turn = axis_angle(k_axis, 90.0 * i / n_arc)
            why = step_to(foot + turn @ arm_vec + hand_off, r_grip)
            logger.info(
                "standing-a-lying-domino: hinge arc step %d of %d: %s, jaws %.1f mm",
                i,
                n_arc,
                why or "ok",
                width() * 1e3,
            )
            if why:
                return finish("unreachable", f"the arc, step {i} of {n_arc}: {why}; still pinched")
        why = holding(width())
        if why:
            return finish("dropped", f"on the arc: {why}")
        spot_xy = foot[:2] - (tt / 2.0) * e_up[:2]
        facing = math.degrees(math.atan2(e_up[1], e_up[0]))
        out["facing_deg"] = float(facing)
        out["standing_tile"] = standing_pose(spot_xy, facing, table_z, size)
        out["release_position"] = _vec(np.array([*spot_xy, table_z]))
        out["pitch_turn_deg"] = 90.0
        why = set_width(grip_open)
        if why:
            try:
                ctx.tool("robot.open_gripper", **on_arm)
            except Exception as exc:
                return finish("unreachable", f"the jaws did not open ({exc}); still pinched")
        out["released"] = True
        here_now = _xyz(here())
        why = line_to(np.array([*here_now[:2], max(lift_z, here_now[2])]), r_grip)
        if why:
            return finish("unreachable", f"released, but the rise: {why}")
        hand_note = f"turned on the pin about the end at ({spot_xy[0]:.3f}, {spot_xy[1]:.3f}), facing {facing:.0f} deg"
        if not verify:
            return finish("standing", hand_note + "; not checked from depth")
        return check(spot_xy, facing, hand_note)

    def handover_run() -> Output:
        """Two arms, no pitch near the table: this arm carries the tile flat to a point in the air with its width
        along the helper's approach; the helper pinches its far end across the thickness with a level approach and
        rolls a quarter turn on its last joint, so the tile stands in its hand; this arm takes the top end top-down
        across the broad faces, the helper lets go, and this arm sets the tile down on its end."""
        out["strategy"] = "handover"
        b_id = int(helper_arm_id)
        if b_id == int(arm_id):
            return finish("unreachable", "the helper arm is the working arm")
        on_b: dict[str, Any] = {"arm_id": b_id} if b_id else {}
        try:
            b_arm = ctx.tool("robot.describe_arm", **on_b)
            jl = b_arm.get("joint_limits") if isinstance(b_arm, dict) else None
            b_limits = [jl["lower"], jl["upper"]] if jl else list(YAM_LIMITS)
        except Exception as exc:
            return finish("unreachable", f"no helper arm {b_id} ({exc})")
        hand_a = _Hand(ctx, int(arm_id), lower, upper, float(pitch_joint_jump))
        hand_b = _Hand(
            ctx, b_id, *(np.asarray(v, dtype=np.float64) for v in b_limits), float(pitch_joint_jump)
        )
        try:
            gf_b = ctx.tool("robot.grasp_frame", **on_b)
        except Exception as exc:
            return finish("unreachable", f"no grasp frame for the helper ({exc})")
        rb0 = matrix(gf_b["rotation"])
        b_approach = rb0.T @ _xyz(gf_b.get("approach") or {"x": 0.0, "y": 0.0, "z": -1.0})
        b_close = rb0.T @ _xyz(gf_b.get("close_axis") or {"x": 1.0, "y": 0.0, "z": 0.0})

        # --- where: the handover point (default over the middle of the two hands, handover_height over the table)
        # and the helper's approach, level and away from its base there (its base from a test turn of joint 1).
        p_a, _ = hand_a.pose()
        p_b, _ = hand_b.pose()
        if handover_point:
            hp = _xyz(handover_point)
        else:
            hp = np.array([*((p_a[:2] + p_b[:2]) / 2.0), table_z + float(handover_height)])
        base_b = hand_b.base()
        if base_b is None:
            return finish(
                "unreachable",
                "the helper's base could not be found from a test turn of its joint 1",
            )
        w_dir = np.array([*(hp[:2] - base_b[:2]), 0.0])
        if float(np.linalg.norm(w_dir)) < 0.05:
            return finish("unreachable", "the handover point is over the helper's base")
        w_dir /= float(np.linalg.norm(w_dir))
        out["facing_deg"] = 0.0

        # --- pick (this arm, top-down across the width near one end), lift.
        cur = rot_now()
        order = sorted(grasp_rots, key=lambda g: abs(jaw_error(cur, g)))
        e_n = heading(theta)
        if float(e_n @ np.array([*(hp[:2] - center[:2]), 0.0])) < 0.0:
            e_n = -e_n
        grasp_point = np.array([*(center[:2] + s * e_n[:2]), table_z + gh])
        g_rot = next(
            (g for g in order if ik(np.array([*grasp_point[:2], lift_z]), g) is not None), None
        )
        if g_rot is None:
            return finish("unreachable", "no wrist solution over the tile passes the IK preview")
        why = pick(grasp_point, g_rot, e_n)
        if why is not None:
            return why
        g_meas, e_local = held["g"], held["e"]
        r_now = turn_last(
            lambda r: wrap(turn_about(r @ local_approach, r @ local_close, w_dir), True),
            _xyz(here()),
            "yaw_turn_deg",
            period=180.0,
            step_deg=float(pitch_step_deg),
        )
        err = wrap(turn_about(r_now @ local_approach, r_now @ local_close, w_dir), True)
        if abs(err) > float(roll_tolerance_deg):
            return finish(
                "unreachable",
                f"the held tile's width stayed {err:+.0f} deg off the helper's approach; still holding",
            )

        # --- this arm to the handover point: the tile's centre at hp.
        def tile_from_a() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            p, r = hand_a.pose()
            a, e = r @ local_approach, r @ e_local
            return p + (g_meas - tt / 2.0) * a - s * e, e, r

        c_now, e_now, r_a = tile_from_a()
        why = hand_a.line(_xyz(here()) + (hp - c_now), r_a)
        if why:
            return finish("unreachable", f"to the handover point: {why}; still holding")
        why = holding(width())
        if why:
            return finish("dropped", f"at the handover point: {why}")

        # --- the helper: level approach along w_dir, jaws closing across the thickness (vertical), at the far end.
        c_now, e_now, r_a = tile_from_a()
        n_t = r_a @ local_approach  # the tile's thickness axis (down while it lies)
        n_t = n_t / (np.linalg.norm(n_t) or 1.0)
        rb = _frame(w_dir, n_t, b_approach, b_close)
        g_b = c_now - float(helper_offset) * e_now - (tw / 2.0 - float(helper_bite)) * w_dir
        pre_b = g_b - float(retreat) * 2.0 * w_dir
        b_start, _ = hand_b.pose()
        why = hand_b.grip(tt + 2.0 * float(grip_clearance))
        if not why:
            why = hand_b.line(pre_b, rb, turn=True)
        if not why:
            why = hand_b.line(g_b, rb)
        if why:
            return finish("unreachable", f"the helper to the tile: {why}; still holding (this arm)")
        w_b = hand_b.close()
        if w_b >= 0.0 and not (tt - NARROW_MARGIN <= w_b <= tt + FULL_MARGIN):
            hand_b.grip(tt + 2.0 * float(grip_clearance))
            hand_b.line(pre_b, rb)
            return finish(
                "unreachable",
                f"the helper closed to {w_b * 1e3:.1f} mm, not across the {tt * 1e3:.1f} mm tile; still holding",
            )
        # The tile in the helper's hand frame, from this arm's model of it.
        pb, rb_now = hand_b.pose()
        c_loc = rb_now.T @ (c_now - pb)
        e_loc = rb_now.T @ e_now
        n_loc = rb_now.T @ n_t
        why = set_width(grip_open)
        if why:
            return finish("unreachable", f"this arm did not open at the handover ({why})")
        p_up = _xyz(here()) + np.array([0.0, 0.0, 0.06])
        why = go("robot.go_to_pose_cartesian", p_up, quat(rot_now()))
        if why:
            return finish("unreachable", f"this arm's rise after the handover: {why}")
        w_b = hand_b.width()
        if 0.0 <= w_b < tt - NARROW_MARGIN:
            return finish(
                "dropped", f"the helper's jaws read {w_b * 1e3:.1f} mm after this arm let go"
            )

        # --- the helper rolls a quarter turn about its approach: the far end down, the tile upright.
        roll = turn_about(rb_now @ b_approach, rb_now @ e_loc, z_up)
        why = hand_b.roll(roll, float(pitch_step_deg), b_approach)
        if why:
            return finish("unreachable", f"the helper's roll: {why}; the helper holds the tile")
        pb, rb_now = hand_b.pose()
        c_up = pb + rb_now @ c_loc
        e_up = rb_now @ e_loc
        n_up = rb_now @ n_loc
        tilt = angle_between(e_up, z_up)
        out["pitch_turn_deg"] = abs(roll)
        out["tilt_at_release_deg"] = tilt
        if tilt > float(tilt_tolerance_deg):
            return finish(
                "unreachable",
                f"after the helper's roll the tile leans {tilt:.1f} deg; the helper holds the tile",
            )

        # --- this arm takes the top end top-down, jaws across the thickness.
        top = c_up + (tl / 2.0) * e_up
        try:
            gf_top = ctx.tool(
                "robot.grasp_frame",
                close_heading_deg=math.degrees(math.atan2(n_up[1], n_up[0])),
                **on_arm,
            )
        except Exception as exc:
            return finish(
                "unreachable", f"no grasp frame for the top ({exc}); the helper holds the tile"
            )
        top_rots = [matrix(gf_top[k]) for k in ("rotation", "rotation_flipped") if gf_top.get(k)]
        a_tcp = np.array([*top[:2], top[2] + tip_reach - float(top_bite)])
        cur = rot_now()
        top_rots.sort(
            key=lambda g: abs(
                wrap(turn_about(g @ local_approach, cur @ local_close, g @ local_close), True)
            )
        )
        r_top = next(
            (g for g in top_rots if ik(a_tcp + np.array([0.0, 0.0, 0.05]), g) is not None), None
        )
        if r_top is None:
            return finish(
                "unreachable",
                "this arm reaches no top-down grasp of the upright tile; the helper holds it",
            )
        why = set_width(tt + 2.0 * float(grip_clearance))
        if not why:
            why = go("robot.go_to_pose", a_tcp + np.array([0.0, 0.0, 0.05]), quat(r_top))
        if why:
            return finish(
                "unreachable", f"this arm over the upright tile: {why}; the helper holds it"
            )
        r_sq = turn_last(
            lambda r: jaw_error(r, r_top),
            a_tcp + np.array([0.0, 0.0, 0.05]),
            "roll_turn_deg",
            period=180.0,
        )
        why = move("robot.go_to_pose_cartesian", a_tcp, quat(r_sq))
        reached = _xyz(here())
        if float(np.linalg.norm(reached - a_tcp)) > float(descent_tolerance):
            go("robot.go_to_pose_cartesian", a_tcp + np.array([0.0, 0.0, 0.05]), quat(r_sq))
            return finish(
                "unreachable",
                f"this arm's descent onto the top stopped {np.linalg.norm(reached - a_tcp) * 1e3:.0f} mm off "
                f"({why or 'stalled'}); the helper holds the tile",
            )
        w_top = width(ctx.tool("robot.close_gripper", **on_arm))
        if w_top >= 0.0 and not (tt - NARROW_MARGIN <= w_top <= tt + FULL_MARGIN):
            return finish(
                "unreachable",
                f"this arm closed to {w_top * 1e3:.1f} mm on the top, not across the tile; the helper holds it",
            )
        top_off = c_up - reached  # the tile's centre from this arm's tool centre
        hand_b.grip(tt + 2.0 * float(grip_clearance) + 0.01)
        pb, rb_now = hand_b.pose()
        why = hand_b.line(pb - float(retreat) * 2.0 * (rb_now @ b_approach), rb_now)
        if why:
            return finish(
                "unreachable", f"the helper's retreat: {why}; this arm holds the tile upright"
            )
        hand_b.line(np.array([*b_start[:2], max(b_start[2], pb[2])]), rb_now, turn=False)
        why = holding(width())
        if why and w_top >= 0.0 and width() < tt - NARROW_MARGIN:
            return finish("dropped", f"after the helper let go: {why}")

        # --- this arm sets the tile down standing at the spot, opens, rises.
        spot_xy = spot
        down = np.array(
            [*(spot_xy - top_off[:2]), table_z + tl / 2.0 + float(release_gap) - top_off[2]]
        )
        r_hold = rot_now()
        here_now = _xyz(here())
        why = go(
            "robot.go_to_pose",
            np.array([*down[:2], max(here_now[2], down[2] + 0.05)]),
            quat(r_hold),
        )
        if not why:
            why = go("robot.go_to_pose_cartesian", down, quat(r_hold))
        if why:
            return finish("unreachable", f"to the staging spot: {why}; still holding (upright)")
        facing = math.degrees(math.atan2(n_up[1], n_up[0]))
        out["facing_deg"] = float(facing)
        out["standing_tile"] = standing_pose(spot_xy, facing, table_z, size)
        out["release_position"] = _vec(np.array([*spot_xy, table_z + float(release_gap)]))
        why = set_width(tt + 2.0 * float(grip_clearance))
        if why:
            try:
                ctx.tool("robot.open_gripper", **on_arm)
            except Exception as exc:
                return finish(
                    "unreachable", f"the jaws did not open at the spot ({exc}); still holding"
                )
        out["released"] = True
        why = go("robot.go_to_pose_cartesian", down + np.array([0.0, 0.0, 0.06]), quat(r_hold))
        if why:
            return finish("unreachable", f"released, but the rise: {why}")
        hand_note = f"handed over to arm {b_id} and back, rolled {roll:+.0f} deg, leaning {tilt:.1f} deg before the regrasp"
        if not verify:
            return finish("standing", hand_note + "; not checked from depth")
        return check(spot_xy, facing, hand_note)

    if strategy == "table_pivot":
        return table_pivot_run()
    if strategy == "hinge_tip":
        return hinge_tip_run()
    if strategy == "handover":
        return handover_run()
    return pose_pitch_run()
