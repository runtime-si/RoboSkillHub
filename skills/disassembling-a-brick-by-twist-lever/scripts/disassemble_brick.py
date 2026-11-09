# Adapted from Robotic_Lego_Manipulation
# (https://github.com/intelligent-control-lab/Robotic_Lego_Manipulation) at commit
# 9bc91654e708e4d8428ecfa76840103cacb2a19a, MIT License: the disassembly pick in
# src/ros_nodes/lego_task_planning_node.cpp (modes 1-5: the offset pre-grab pose,
# the descent into the grab pose, the twist of Twist_Deg = 14 deg about the tool's
# disassemble frame - robot_DH_tool_disassemble, 8 mm further down the tool axis
# than the tool tip - and the 15 mm lift with the twist kept), Lego.cpp
# calc_brick_grab_pose (the grab on one side of the brick, press_side 1-4) and
# config/user_config.json (Twist_Deg). The wrist-camera height reading and the
# last-joint turn are pressing-a-brick-onto-studs' (this hub, MIT).
# Changes: ICL's end-of-arm tool sits on one end of the brick and its twist pivots
# about a fixed tool frame on a compliant industrial arm, with the brick's pose
# known from its own build bookkeeping. Here a parallel gripper grasps the brick
# top-down across one pair of faces (the grasp from computing-brick-pose-from-
# stud-grid), the pivot is the brick's bottom edge under one finger, and the
# twist is commanded in small steps about that edge, each read back: the hand
# lagging its command (or a joint-effort residual where the observation carries
# efforts) stands in for a force sensor, so a brick that holds is "stuck" and
# a lag that rises and falls again marks the studs letting go. ICL's 2.8 mm
# descent into the grab pose becomes a 1 mm press after the close (bounded by
# the lag). A straight pull and a wiggle (alternate small levers about both
# edges) are added as strategies and as fallbacks, a slipped grip is regrasped,
# and the result is read from the wrist depth camera (the seat's height before
# and after, and the neighbouring cells), not from bookkeeping. Exits are routed.
"""Remove a seated LEGO or DUPLO brick by levering it about one bottom edge, with straight-pull and wiggle fallbacks."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PRESETS: dict[str, dict[str, float]] = {
    "lego": {
        "stud_pitch": 0.008,
        "stud_height": 0.0017,
        "stud_diameter": 0.0048,
        "brick_height": 0.0096,
    },
    "duplo": {
        "stud_pitch": 0.016,
        "stud_height": 0.0045,
        "stud_diameter": 0.0094,
        "brick_height": 0.0192,
    },
}
"""Brick systems [m]: pressing-a-brick-onto-studs' and computing-brick-pose-from-stud-grid's
presets (LEGO from ICL's Lego.hpp, DUPLO published dimensions)."""

FOOTPRINT_PLAY = 0.0001
"""How much shorter than n pitches a brick is on each side [m]: ICL's brick_len_offset_
(0.2 mm) split over both sides."""

GRASP_DEPTH = {"lego": 0.005, "duplo": 0.013}
ONE_WIDE_GRASP_DEPTH_EXTRA = 0.001
"""Fingertips under the brick's top [m]: computing-brick-pose-from-stud-grid's defaults
(LEGO BrickSim's 5 mm, 6 mm for 1-wide; DUPLO 13 mm, which keeps the tips 6.2 mm over
the plate, clear of the 4.5 mm studs beside the brick)."""

TIP_REACH = 0.0102
"""How far closed linear_4310 fingertips reach below the tool centre [m]."""

APPROACH_HEIGHT = 0.04
"""Hover over the grasp and clearance after the lift [m]: BrickSim's ASSEMBLY_HEIGHT_OFFSET,
as the press skill."""

TWIST_DEG = 14.0
"""The lever angle [deg]: ICL's Twist_Deg (config/user_config.json)."""

TWIST_STEP_DEG = 2.0
"""Angular step of the lever [deg]: small enough that the tool centre moves under a
millimetre per step on a DUPLO grasp (2 deg x ~23 mm), so a lag is read before the
grip has been loaded much."""

LIFT_HEIGHT = 0.015
"""Lift after the lever, and the straight pull's length [m]: ICL's mode 5 (+0.015 m with
the twist kept). Over three DUPLO stud heights. ICL lifts along world z; after a lever
this lifts along the brick's own tilted axis (lift_along='brick'), so the near wall slides
along the near studs instead of being driven into them: on a YAM station twin a vertical
lift of a DUPLO brick levered 14 deg stalled the hand within 3 mm (lag 4.3 mm)."""

SEAT_PRESS = 0.001
"""Press after the close [m]: ICL descends 2.8 mm into its grab pose (pick_offset z) with
its own tool; a parallel grip slides along the brick under a deeper press, so 1 mm."""

GRIP_SQUEEZE = 0.002
"""The jaws are closed to the brick's width less this [m] where the connector takes a
width (robot.set_grip): the press check's squeeze on a YAM station twin."""

FEEL_STEP_FRACTION = 0.25
MIN_FEEL_STEP = 0.0004
"""The pull's step: a quarter of the stud height (1.1 mm DUPLO, 0.43 mm LEGO), at least
0.4 mm (the press skill's descent step)."""

LAG_FRACTION = 0.5
"""Default lag threshold, as a fraction of the pull step (0.56 mm DUPLO): the press skill's,
tuned on a YAM station twin whose servo arrives within FINE_TOLERANCE. A real YAM's free
lag is millimetres (detecting-contact-from-joint-effort's LAG_THRESHOLD = 3 mm): pass
lag_threshold from a free-air run there."""

STALL_STEPS = 3
STALL_PROGRESS = 0.5
"""A stall is STALL_STEPS steps in a row in which the hand, loaded, advanced less than
half its commanded step: by then the command leads a stopped hand by about 3 mm on a
DUPLO pull (the press skill's PRESS_DEPTH bound on what a stiff servo is asked to push).
A load alone is not one: on a YAM station twin a 5 N pull (two DUPLO studs' clutch in the
check) stood the hand 0.75-0.95 mm behind its command while it kept advancing a full
step at a time."""

ROT_STALL_DEG = 4.0
"""The lever has stalled when the hand's approach axis stands this far short of the
commanded tilt [deg]: a servo calls a rotation arrived within 3 deg."""

EFFORT_THRESHOLD = 0.5
EFFORT_SIGMAS = 4.0
EFFORT_SAMPLES = 3
"""Joint-effort test where the observation carries efforts (detecting-contact-from-joint-
effort's, from i2rt's clog check): a residual over max(0.5 N.m, 4 sigma)."""

FINE_TOLERANCE = 0.0003
FINE_MAX_STEPS = 90
"""Arrival ball [m] and step budget asked of each fine move (passed only to connectors that
take them): the press skill's values."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from a free move's goal [m]: no tighter than a real YAM
station's own 12 mm."""

GRASP_STOP_TOLERANCE = 0.003
"""An open-jaw descent that stops this far over the grasp has met the brick's top or a
neighbour [m]."""

HOLD_MARGIN = 0.001
"""The jaws hold the brick while they read at least its width less max(this, 0.6 x the
squeeze) [m]: a twin's empty jaws close to the width less the squeeze."""

YAW_TOLERANCE_DEG = 2.0
"""Largest angle between the jaws and the brick's faces left before the descent [deg]."""

TURN_TRIES = 3
ROLL_LIMITS = (-2.0944, 2.0944)
LIMIT_MARGIN = 0.05
"""The last-joint turn (stacking-objects', via the press skill): tries, the i2rt YAM
joint 6 range [rad] when none is given, and the margin kept from it."""

WIGGLE_FRACTIONS = (0.5, 1.0)
"""Wiggle: alternate levers about both edges at these fractions of wiggle_deg, then a
straight pull."""

SETTLE_QUIET = 0.0002
SETTLE_PERIOD = 0.15
SETTLE_TIMEOUT = 2.0
"""The hand has settled when two reads SETTLE_PERIOD apart [s] differ by less than
SETTLE_QUIET [m], or after SETTLE_TIMEOUT [s] (the press skill's)."""

GRIP_SETTLE_S = 0.4
"""Wait after closing the jaws [s]."""

VIEW_RANGES = (0.20, 0.25)
MIN_VIEW_CLEARANCE = 0.05
IMAGE_MARGIN = 0.08
DEPTH_WINDOW = 1
MIN_DEPTH = 0.07
MAX_DEPTH = 0.5
MIN_SAMPLES = 3
SAMPLE_INSET = 0.0015
STUD_MARGIN = 0.0008
"""The wrist depth view and sampling: pressing-a-brick-onto-studs' (from refining-target-
from-wrist-closeup): D405 ranges, view clearance, 3 x 3 median windows, points inside the
brick's edge and clear of its studs, at least 3 readings."""

INNER_FRACTION = 0.25
"""The seat's height after the lift is the median of the points more than a quarter pitch
inside the brick's edge (where there are 3): an empty seat is a hole, and a sight line to a
point near its rim lands on the rim."""

GONE_FRACTION = 0.5
"""The seat reads empty when its top reads more than this fraction of a brick height under
the brick's top."""

BELOW_FRACTION = 1.5
"""...and the brick under it came off too when it reads more than this many brick heights
under it."""

RING_CHANGE_FRACTION = 0.5
MIN_RING_CHANGE = 0.002
"""A neighbouring cell has moved when its stud top reads more than max(half a stud height,
2 mm) off its height before the lift."""

OCCLUDER_MARGIN = 0.01
"""Depth readings more than this over the highest expected surface are the held brick or a
finger in the view, and are dropped [m]."""

REFUSALS = ("unreachable", "bad_speed_scale")
STRATEGIES = ("lever", "straight", "wiggle")
ROUTES = (
    "removed",
    "stuck",
    "slipped",
    "neighbour_lifted",
    "missed",
    "unverified",
    "unreachable",
)


class Output(TypedDict):
    route: str
    detail: str
    strategy: str
    attempts: int
    attempt_log: str
    pivot_side: str
    twist_deg: float
    peak_lag: float
    releases: str
    release_count: int
    seat_lag: float
    holding: bool
    top_height_change: float
    neighbour_change: float
    samples: int
    verified_by: str


# ------------------------------------------------------------------ geometry


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


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


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * k @ k


def _axis_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two rotations' approach (z) axes [deg]."""
    return math.degrees(math.acos(min(1.0, max(-1.0, float(a[:, 2] @ b[:, 2])))))


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


def _heading(v: np.ndarray) -> float:
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


def _wrap(deg: float, period: float) -> float:
    return (deg + period / 2.0) % period - period / 2.0


def _yaw_between(r_from: np.ndarray, r_to: np.ndarray) -> float:
    """How far [deg] the hand turned about the vertical from ``r_from`` to ``r_to``."""
    rel = r_to @ r_from.T
    return math.degrees(math.atan2(float(rel[1, 0]), float(rel[0, 0])))


def _matrix(pose: dict[str, Any]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rot(pose["rotation"])
    m[:3, 3] = _xyz(pose)
    return m


def _refusal(reply: Any) -> tuple[str, str]:
    """("ok" | "short" | "refused", why) from a move's reply (connectors that route instead
    of raising)."""
    if not isinstance(reply, dict):
        return "ok", ""
    outcome = reply.get("outcome")
    if outcome in REFUSALS:
        return "refused", f"{outcome}: {reply.get('reason', '')}".rstrip(": ")
    if outcome not in (None, "reached") or reply.get("reached") is False:
        return "short", str(outcome or reply.get("reason") or "not reached")
    if reply.get("success") is False:
        return "refused", str(reply.get("reason") or reply.get("message") or "failed")
    return "ok", ""


def _joint_efforts(obs: Any, arm_id: int) -> np.ndarray | None:
    if not isinstance(obs, dict):
        return None
    arms = obs.get("arms")
    js = None
    if isinstance(arms, list) and len(arms) > arm_id and isinstance(arms[arm_id], dict):
        js = arms[arm_id].get("joint_state")
    if not isinstance(js, dict):
        js = obs.get("joint_state")
    if not isinstance(js, dict):
        return None
    for name in ("efforts", "effort", "torques"):
        value = js.get(name)
        if value is not None:
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size:
                return arr
    return None


class _Arm:
    """The hand through gap's pose convention, with the fine-move knobs passed only to
    connectors that take them."""

    def __init__(self, ctx: NodeContext, arm_id: int) -> None:
        self.ctx = ctx
        self.arm_id = int(arm_id)
        self.on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
        self.knobs: dict[str, Any] = {"tolerance": FINE_TOLERANCE, "max_steps": FINE_MAX_STEPS}

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        pose = _pose(self.ctx.tool("robot.get_ee_pose", **self.on_arm))
        return _xyz(pose), _rot(pose["rotation"])

    def move(self, tool: str, p: np.ndarray, r: np.ndarray, fine: bool = False) -> tuple[str, str]:
        pose = {"position": _vec(p), "rotation": _quat(r)}
        try:
            if fine and self.knobs:
                try:
                    reply = self.ctx.tool(tool, pose=pose, **self.knobs, **self.on_arm)
                except TypeError as exc:
                    if "keyword" not in str(exc):
                        raise
                    logger.info(
                        "disassembling-a-brick: %s takes no fine-move knobs (%s)", tool, exc
                    )
                    self.knobs = {}
                    reply = self.ctx.tool(tool, pose=pose, **self.on_arm)
            else:
                reply = self.ctx.tool(tool, pose=pose, **self.on_arm)
        except Exception as exc:  # a station raises where a simulator routes
            why = str(exc)
            return ("short" if "did not reach" in why else "refused"), why
        return _refusal(reply)

    def still(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the hand stands once it stops moving (or after SETTLE_TIMEOUT)."""
        p, r = self.read()
        t0 = time.monotonic()
        while time.monotonic() - t0 < SETTLE_TIMEOUT:
            time.sleep(SETTLE_PERIOD)
            q, s = self.read()
            if float(np.linalg.norm(q - p)) < SETTLE_QUIET:
                return q, s
            p, r = q, s
        return p, r


def _ik_ok(ctx: NodeContext, p: np.ndarray, r: np.ndarray, on_arm: dict[str, Any]) -> bool:
    try:
        reply = ctx.tool(
            "robot.solve_ik", pose={"position": _vec(p), "rotation": _quat(r)}, **on_arm
        )
    except Exception as exc:
        if "unreachable" in str(exc).lower():
            return False
        logger.info("disassembling-a-brick: no IK preview (%s)", exc)
        return True
    return not (
        isinstance(reply, dict)
        and (reply.get("success") is False or reply.get("within_servo_tolerance") is False)
    )


def _top_down(close: np.ndarray) -> np.ndarray:
    """Tool +z down, the jaws (tool +y) along ``close``."""
    z = np.array([0.0, 0.0, -1.0])
    y = close / np.linalg.norm(close)
    return np.column_stack([np.cross(y, z), y, z])


def _wrist_camera(obs: dict[str, Any], tool: np.ndarray, name: str | None) -> dict[str, Any] | None:
    """The named camera, else the eye-in-hand / wrist camera riding nearest the tool."""
    cams = [c for c in obs.get("cameras") or [] if c.get("pose") is not None]
    if name:
        return next((c for c in cams if c.get("name") == name), None)
    wrist = [c for c in cams if any(k in str(c.get("name", "")) for k in ("eye_in_hand", "wrist"))]
    if not wrist:
        return None
    return min(wrist, key=lambda c: float(np.linalg.norm(_xyz(c["pose"]) - tool[:3, 3])))


def _views(tool: np.ndarray, tool_cam: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    """Hand poses putting ``target`` on the wrist camera's optical axis: looking straight
    down, then with the hand's rotation as it is."""
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


def _read_heights(cam: dict[str, Any], pts: list[np.ndarray], ceiling: float) -> list[float | None]:
    """World heights the wrist depth reads at the pixels the points project to (None where
    nothing usable reads, or the reading is over ``ceiling``: something in the view)."""
    depth = np.asarray(cam["depth"], dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    m = _matrix(cam["pose"])
    inv = np.linalg.inv(m)
    h, w = depth.shape[:2]
    out: list[float | None] = []
    for p in pts:
        pc = inv @ np.append(p, 1.0)
        if pc[2] <= 1.0e-6:
            out.append(None)
            continue
        u = k[0, 0] * pc[0] / pc[2] + k[0, 2]
        v = k[1, 1] * pc[1] / pc[2] + k[1, 2]
        if not (
            IMAGE_MARGIN * w <= u <= (1 - IMAGE_MARGIN) * w
            and IMAGE_MARGIN * h <= v <= (1 - IMAGE_MARGIN) * h
        ):
            out.append(None)
            continue
        col, row = int(round(u)), int(round(v))
        win = depth[
            max(0, row - DEPTH_WINDOW) : row + DEPTH_WINDOW + 1,
            max(0, col - DEPTH_WINDOW) : col + DEPTH_WINDOW + 1,
        ]
        good = win[np.isfinite(win) & (win >= MIN_DEPTH) & (win <= MAX_DEPTH)]
        if not good.size:
            out.append(None)
            continue
        z = float(np.median(good))
        q = m @ np.array([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z, 1.0])
        out.append(float(q[2]) if q[2] <= ceiling else None)
    return out


def _stud_centres(studs: list[int], pitch: float) -> list[tuple[float, float]]:
    return [
        ((i - (studs[0] - 1) / 2.0) * pitch, (j - (studs[1] - 1) / 2.0) * pitch)
        for i in range(studs[0])
        for j in range(studs[1])
    ]


def _top_samples(
    centre: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    lens: tuple[float, float],
    studs: list[int],
    pitch: float,
    stud_d: float,
    z: float,
) -> list[np.ndarray]:
    """Points on the brick's top, inside its edge and clear of its studs (the same xy read
    the surface under it once it is gone: the studs below stand under the brick's)."""
    hx, hy = lens[0] / 2.0 - SAMPLE_INSET, lens[1] / 2.0 - SAMPLE_INSET
    keep = stud_d / 2.0 + STUD_MARGIN
    centres = _stud_centres(studs, pitch)
    pts = []
    for u in np.linspace(-hx, hx, 2 * studs[0] + 1):
        for v in np.linspace(-hy, hy, 2 * studs[1] + 1):
            if all(math.hypot(u - a, v - b) > keep for a, b in centres):
                p = centre + bx * u + by * v
                pts.append(np.array([p[0], p[1], z]))
    return pts


def _ring_samples(
    centre: np.ndarray, bx: np.ndarray, by: np.ndarray, studs: list[int], pitch: float, z: float
) -> list[np.ndarray]:
    """The stud centres of the cells next to the brick along both its axes: a neighbour's
    top, or the bare studs of the plate or brick under them."""
    pts = []
    nx, ny = studs
    for j in range(ny):
        v = (j - (ny - 1) / 2.0) * pitch
        for side in (-1.0, 1.0):
            u = side * (nx + 1) / 2.0 * pitch
            pts.append(centre + bx * u + by * v)
    for i in range(nx):
        u = (i - (nx - 1) / 2.0) * pitch
        for side in (-1.0, 1.0):
            v = side * (ny + 1) / 2.0 * pitch
            pts.append(centre + bx * u + by * v)
    return [np.array([p[0], p[1], z]) for p in pts]


def _chain(strategy: str, fallbacks: Any) -> list[str]:
    if fallbacks is None:
        rest = ["straight"] if strategy != "straight" else []
    elif isinstance(fallbacks, str):
        rest = [s.strip() for s in fallbacks.split(",") if s.strip()]
    else:
        rest = [str(s) for s in fallbacks]
    chain = [strategy] + [s for s in rest if s != strategy]
    for s in chain:
        if s not in STRATEGIES:
            raise ValueError(f"strategies must be among {STRATEGIES}, got {s!r}")
    return list(dict.fromkeys(chain))


# ------------------------------------------------------------------ the skill


def run(
    ctx: NodeContext,
    brick_position: dict[str, float],
    brick_yaw_deg: float = 0.0,
    brick_studs: list[int] | None = None,
    system: str = "duplo",
    stud_pitch: float | None = None,
    stud_height: float | None = None,
    stud_diameter: float | None = None,
    brick_height: float | None = None,
    strategy: str = "lever",
    fallbacks: list[str] | str | None = None,
    pivot_side: str = "auto",
    close_axis: str = "short",
    twist_deg: float = TWIST_DEG,
    twist_step_deg: float = TWIST_STEP_DEG,
    wiggle_deg: float | None = None,
    lift_height: float = LIFT_HEIGHT,
    lift_along: str = "brick",
    approach_height: float = APPROACH_HEIGHT,
    grasp_depth: float | None = None,
    seat_press: float = SEAT_PRESS,
    grip_squeeze: float = GRIP_SQUEEZE,
    lag_threshold: float | None = None,
    regrasp_on_slip: bool = True,
    verify: bool = True,
    compare_before: bool = True,
    camera: str | None = None,
    check_reach: bool = True,
    reach_tolerance: float = REACH_TOLERANCE,
    yaw_tolerance_deg: float = YAW_TOLERANCE_DEG,
    roll_limits: list[float] | None = None,
    arm_id: int = 0,
) -> Output:
    if system not in PRESETS:
        raise ValueError(f"system must be one of {sorted(PRESETS)}, got {system!r}")
    if pivot_side not in ("auto", "+", "-"):
        raise ValueError(f"pivot_side must be 'auto', '+' or '-', got {pivot_side!r}")
    if lift_along not in ("brick", "vertical"):
        raise ValueError(f"lift_along must be 'brick' or 'vertical', got {lift_along!r}")
    if close_axis not in ("short", "x", "y"):
        raise ValueError(f"close_axis must be 'short', 'x' or 'y', got {close_axis!r}")
    chain = _chain(str(strategy), fallbacks)
    preset = PRESETS[system]
    pitch = float(stud_pitch if stud_pitch is not None else preset["stud_pitch"])
    stud_h = float(stud_height if stud_height is not None else preset["stud_height"])
    stud_d = float(stud_diameter if stud_diameter is not None else preset["stud_diameter"])
    body_h = float(brick_height if brick_height is not None else preset["brick_height"])
    studs = [int(v) for v in (brick_studs or [2, 2])]
    step = max(MIN_FEEL_STEP, FEEL_STEP_FRACTION * stud_h)
    lag_thr = float(lag_threshold) if lag_threshold is not None else LAG_FRACTION * step
    wiggle = float(wiggle_deg) if wiggle_deg is not None else float(twist_deg)
    up = np.array([0.0, 0.0, 1.0])

    arm = _Arm(ctx, arm_id)
    seat = _xyz(brick_position)
    yaw = math.radians(float(brick_yaw_deg))
    bx = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    by = np.array([-bx[1], bx[0], 0.0])
    lens = (studs[0] * pitch - 2 * FOOTPRINT_PLAY, studs[1] * pitch - 2 * FOOTPRINT_PLAY)
    along_x = close_axis == "x" or (close_axis == "short" and lens[0] <= lens[1])
    close_dir = bx if along_x else by
    jaw_width = lens[0] if along_x else lens[1]
    top_z = float(seat[2] + body_h)
    depth = (
        float(grasp_depth)
        if grasp_depth is not None
        else GRASP_DEPTH[system]
        + (ONE_WIDE_GRASP_DEPTH_EXTRA if system == "lego" and min(studs) == 1 else 0.0)
    )
    grasp_p = np.array([seat[0], seat[1], top_z - depth + TIP_REACH])
    hover_z = float(grasp_p[2] + approach_height)
    clear_z = float(top_z + TIP_REACH + approach_height)

    out: Output = {
        "route": "unreachable",
        "detail": "",
        "strategy": chain[0],
        "attempts": 0,
        "attempt_log": "",
        "pivot_side": "",
        "twist_deg": 0.0,
        "peak_lag": 0.0,
        "releases": "",
        "release_count": 0,
        "seat_lag": 0.0,
        "holding": False,
        "top_height_change": 0.0,
        "neighbour_change": 0.0,
        "samples": 0,
        "verified_by": "",
    }
    log: list[str] = []

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        out["attempt_log"] = "; ".join(log)
        return out

    def go(
        tool: str, p: np.ndarray, r: np.ndarray, fine: bool = False, check_rotation: bool = True
    ) -> str:
        kind, why = arm.move(tool, p, r, fine=fine)
        q, s = arm.read()
        off = float(np.linalg.norm(q - p))
        ang = _axis_deg(r, s)
        if (
            kind == "refused"
            or off > float(reach_tolerance)
            or (check_rotation and ang > ROT_STALL_DEG)
        ):
            return f"{why or 'ended off'}; the hand is {off * 1e3:.0f} mm / {ang:.1f} deg from it"
        return ""

    # -- the grasp rotation: jaws across close_dir, from robot.grasp_frame where served.
    candidates: list[np.ndarray] = []
    try:
        frame = ctx.tool("robot.grasp_frame", close_heading_deg=_heading(close_dir), **arm.on_arm)
        for key in ("rotation", "rotation_flipped"):
            if isinstance(frame, dict) and frame.get(key):
                candidates.append(_rot(frame[key]))
    except Exception as exc:
        logger.info("disassembling-a-brick: no grasp frame (%s)", exc)
    if not candidates:
        candidates = [_top_down(close_dir), _top_down(-close_dir)]
    grasp_r = None
    for r in candidates:
        if not check_reach or (
            _ik_ok(ctx, grasp_p, r, arm.on_arm)
            and _ik_ok(ctx, grasp_p + up * approach_height, r, arm.on_arm)
        ):
            grasp_r = r
            break
    if grasp_r is None:
        return finish("unreachable", "neither wrist solution of the grasp passes the IK preview")

    def _crosses_brick(eye: np.ndarray, q: np.ndarray) -> bool:
        """Whether the sight line from ``eye`` to ``q`` passes through the seated brick's
        box (slab test in the brick's frame)."""
        frame = np.column_stack([bx, by, up])
        a, b = frame.T @ (eye - seat), frame.T @ (q - seat)
        lo = np.array([-lens[0] / 2.0, -lens[1] / 2.0, 0.0])
        hi = np.array([lens[0] / 2.0, lens[1] / 2.0, body_h])
        t0, t1 = 0.0, 1.0
        for k in range(3):
            d = b[k] - a[k]
            if abs(d) < 1e-12:
                if not lo[k] <= a[k] <= hi[k]:
                    return False
                continue
            ta, tb = sorted(((lo[k] - a[k]) / d, (hi[k] - a[k]) / d))
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return False
        return True

    # -- the wrist camera, and the seat before anything moves.
    top_pts = _top_samples(seat, bx, by, lens, studs, pitch, stud_d, top_z)
    inner = [
        abs(float((q - seat) @ bx)) <= lens[0] / 2.0 - INNER_FRACTION * pitch
        and abs(float((q - seat) @ by)) <= lens[1] / 2.0 - INNER_FRACTION * pitch
        for q in top_pts
    ]
    ring_pts = _ring_samples(seat, bx, by, studs, pitch, seat[2] + stud_h)
    ceiling = top_z + stud_h + OCCLUDER_MARGIN

    def look(note: list[str]) -> tuple[list[float | None], list[float | None]] | None:
        try:
            obs = ctx.tool("robot.get_observation")
            tool_now = np.eye(4)
            tool_now[:3, :3] = _rot(obs["arms"][arm.arm_id]["ee_pose"]["rotation"])
            tool_now[:3, 3] = _xyz(obs["arms"][arm.arm_id]["ee_pose"])
            cam = _wrist_camera(obs, tool_now, camera)
        except Exception as exc:
            note.append(f"no observation ({exc})")
            return None
        if cam is None:
            note.append("no wrist camera with a pose")
            return None
        tool_cam = np.linalg.inv(tool_now) @ _matrix(cam["pose"])
        for view in _views(tool_now, tool_cam, np.array([seat[0], seat[1], top_z])):
            if view[2, 3] < top_z + MIN_VIEW_CLEARANCE:
                note.append("a view under the clearance floor")
                continue
            if check_reach and not _ik_ok(ctx, view[:3, 3], view[:3, :3], arm.on_arm):
                note.append("a view out of reach (IK preview)")
                continue
            why = go("robot.go_to_pose", view[:3, 3], view[:3, :3], check_rotation=False)
            if why:
                note.append(f"a view move: {why}")
                continue
            arm.still()
            cam = _wrist_camera(ctx.tool("robot.get_observation"), tool_now, camera)
            if cam is None:
                note.append("the wrist frame went missing")
                continue
            tops = _read_heights(cam, top_pts, ceiling)
            if sum(h is not None for h in tops) < MIN_SAMPLES:
                note.append(f"{sum(h is not None for h in tops)} depth readings on the seat")
                continue
            # A ring point seen past the brick's own body reads the brick: not compared.
            eye = _xyz(cam["pose"])
            ring = [
                None if _crosses_brick(eye, q) else h
                for q, h in zip(ring_pts, _read_heights(cam, ring_pts, ceiling), strict=True)
            ]
            return tops, ring
        return None

    before: tuple[list[float | None], list[float | None]] | None = None
    if verify and compare_before:
        note: list[str] = []
        before = look(note)
        if before is not None:
            dz0 = float(np.median([h for h in before[0] if h is not None])) - top_z
            if dz0 < -GONE_FRACTION * body_h:
                return finish(
                    "missed",
                    f"the wrist depth reads the seat {dz0 * 1e3:+.1f} mm from the brick's top: no "
                    "brick there; nothing moved",
                )
        else:
            log.append("no view before the lift (" + "; ".join(note) + ")")

    # -- motion helpers -------------------------------------------------------
    efforts_on = False
    try:
        efforts_on = _joint_efforts(ctx.tool("robot.get_observation"), arm.arm_id) is not None
    except Exception as exc:
        logger.info("disassembling-a-brick: no observation (%s)", exc)

    def effort_window() -> np.ndarray | None:
        kept = []
        for _ in range(EFFORT_SAMPLES):
            e = _joint_efforts(ctx.tool("robot.get_observation"), arm.arm_id)
            if e is not None:
                kept.append(e)
        return np.mean(kept, axis=0) if kept else None

    free_lags: list[float] = []
    carry = {"loaded": False}

    def follow(
        start: np.ndarray, poses: list[tuple[np.ndarray, np.ndarray, float]], base_r: np.ndarray
    ) -> dict[str, Any]:
        """Command fine poses in turn from the command ``start`` (each with the tilt it
        should have from ``base_r``) and read each back. A step is loaded when the hand
        stands lag_thr further from its command than in free motion (or a joint effort
        jumps); it is held when, loaded, the hand also advanced less than STALL_PROGRESS
        of the step (or the tilt stands ROT_STALL_DEG short). STALL_STEPS held steps in a
        row stop it. A load that clears again marks a release: the studs let go. The
        free-motion lags and whether the hand is loaded carry over between calls."""
        p0, _ = arm.still()
        base = free_lags
        if not base:
            base.append(min(float(np.linalg.norm(start - p0)), lag_thr))
        windows: list[np.ndarray] = []
        if efforts_on:
            w = effort_window()
            if w is not None:
                windows.append(w)
        res: dict[str, Any] = {
            "stalled": False,
            "peak": 0.0,
            "releases": [],
            "tilt": 0.0,
            "done": 0,
        }
        run_len = 0
        cmd_prev, q_prev = start, p0
        for i, (p, r, tilt) in enumerate(poses):
            kind, why = arm.move("robot.go_to_pose", p, r, fine=True)
            if kind == "refused":
                res.update(refused=why)
                return res
            q, s = arm.still()
            lag = float(np.linalg.norm(p - q))
            over = lag - float(np.median(base[-3:]))
            reached = _axis_deg(base_r, s)
            short = abs(tilt) - reached
            stride = p - cmd_prev
            length = float(np.linalg.norm(stride))
            progress = float(np.dot(q - q_prev, stride)) / length**2 if length > 1e-9 else 1.0
            loaded = over > lag_thr
            if efforts_on and windows:
                w = effort_window()
                if w is not None:
                    noise = np.std(windows[-5:], axis=0) if len(windows) >= 3 else np.zeros_like(w)
                    thr = np.maximum(EFFORT_THRESHOLD, EFFORT_SIGMAS * noise)
                    if float(np.max(np.abs(w - np.mean(windows[-3:], axis=0)) / thr)) > 1.0:
                        loaded = True
                    elif not loaded:
                        windows.append(w)
            held = (loaded and progress < STALL_PROGRESS) or short > ROT_STALL_DEG
            logger.debug(
                "step %d: tilt cmd %.1f read %.1f deg, lag %.2f mm (over %.2f), progress %.2f %s",
                i,
                tilt,
                reached,
                lag * 1e3,
                over * 1e3,
                progress,
                "HELD" if held else ("LOADED" if loaded else kind),
            )
            res["peak"] = max(res["peak"], over)
            res["tilt"], res["done"] = reached, i + 1
            cmd_prev, q_prev = p, q
            if held:
                run_len += 1
                if run_len >= STALL_STEPS:
                    res["stalled"] = True
                    res["why"] = (
                        f"the tilt stood {short:.1f} deg short"
                        if short > ROT_STALL_DEG
                        else f"the hand stood {over * 1e3:.1f} mm behind its command and stopped advancing"
                    )
                    return res
            else:
                run_len = 0
            if loaded:
                carry["loaded"] = True
            else:
                if carry["loaded"]:
                    res["releases"].append((reached, float(q[2] - start[2])))
                    carry["loaded"] = False
                base.append(lag)
        return res

    def lever_poses(pivot: np.ndarray, axis: np.ndarray, p_s: np.ndarray, r_s: np.ndarray, angles):
        poses = []
        for a in angles:
            rr = _axis_angle(axis, math.radians(a))
            poses.append((pivot + rr @ (p_s - pivot), rr @ r_s, a))
        return poses

    def steps_to(angle: float) -> list[float]:
        n = max(1, math.ceil(abs(angle) / max(0.1, float(twist_step_deg)) - 1e-9))
        return [angle * k / n for k in range(1, n + 1)]

    def lift_poses(
        p_s: np.ndarray, r_s: np.ndarray, tilt: float
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Up lift_height in fine steps: along the brick's own (tilted) axis after a lever
        when lift_along is 'brick', else straight up."""
        way = -r_s[:, 2] if lift_along == "brick" else up
        way = way / (np.linalg.norm(way) or 1.0)
        if way[2] <= 0.0:
            way = up
        n = max(1, math.ceil(float(lift_height) / step - 1e-9))
        return [(p_s + way * float(lift_height) * k / n, r_s, tilt) for k in range(1, n + 1)]

    def pivots(r_g: np.ndarray, only: bool = True) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """The bottom edges under the fingers, as (name in brick axes, pivot point, rotation
        axis lifting the far side): pivot_side's first (only it, with ``only``). '+' is the
        edge under the finger on the jaws' +y side."""
        c = r_g[:, 1].copy()
        c[2] = 0.0
        c /= np.linalg.norm(c) or 1.0
        name_axis = "x" if abs(float(c @ bx)) >= abs(float(c @ by)) else "y"
        found = []
        for sgn in (1.0, -1.0):
            d = sgn * c
            sign = "+" if float(d @ (bx if name_axis == "x" else by)) > 0 else "-"
            found.append((sign + name_axis, seat + d * (jaw_width / 2.0), np.cross(up, d)))
        if pivot_side == "-":
            found.reverse()
        return found[:1] if only and pivot_side != "auto" else found

    def pick_pivot(r_g: np.ndarray, angle: float) -> tuple[str, np.ndarray, np.ndarray] | None:
        for name, piv, axis in pivots(r_g):
            if not check_reach:
                return name, piv, axis
            rr = _axis_angle(axis, math.radians(angle))
            end = piv + rr @ (grasp_cmd - piv)
            if _ik_ok(ctx, end, rr @ r_g, arm.on_arm) and _ik_ok(
                ctx, end + up * float(lift_height), rr @ r_g, arm.on_arm
            ):
                return name, piv, axis
        return None

    # -- the grasp ------------------------------------------------------------
    grasp_cmd = grasp_p.copy()

    def jaws_hold() -> bool | None:
        try:
            fraction = float(ctx.tool("robot.get_gripper", **arm.on_arm)["position"])
            grip = ctx.tool("robot.describe_gripper", **arm.on_arm)
            fit = grip.get("width_fit") or {}
            at_open = float(fit.get("at_open_m") or grip.get("span_m") or 0.0)
            at_closed = float(fit.get("at_closed_m") or 0.0)
        except Exception as exc:
            logger.info("disassembling-a-brick: gripper not read (%s)", exc)
            return None
        if at_open <= at_closed:
            return None
        width = at_closed + (at_open - at_closed) * fraction
        return width >= jaw_width - max(HOLD_MARGIN, 0.6 * float(grip_squeeze))

    def grasp() -> tuple[str, str, np.ndarray]:
        """Open, hover, turn the jaws onto the faces, descend, close, press. Returns
        (route or "", why, the hand's rotation)."""
        nonlocal grasp_cmd
        ctx.tool("robot.open_gripper", **arm.on_arm)
        p, r = arm.read()
        if p[2] < hover_z - 1.0e-3:
            why = go(
                "robot.go_to_pose_cartesian",
                np.array([p[0], p[1], hover_z]),
                r,
                check_rotation=False,
            )
            if why:
                return "unreachable", f"rise: {why}", r
        why = go("robot.go_to_pose", np.array([grasp_p[0], grasp_p[1], hover_z]), grasp_r)
        if why:
            return "unreachable", f"hover: {why}", grasp_r
        _, r_now = arm.read()
        jaw = r_now[:, 1]
        off = _wrap(_heading(close_dir) - _heading(jaw), 180.0)
        if abs(off) > float(yaw_tolerance_deg):
            done, note = _turn_in_place(ctx, arm, off, float(yaw_tolerance_deg), roll_limits)
            if abs(off - done) > float(yaw_tolerance_deg):
                return (
                    "unreachable",
                    f"the jaws are {off - done:+.1f} deg off the faces ({note or 'short'})",
                    r_now,
                )
            _, r_now = arm.read()
        # The fold: re-command the grasp shifted by what the hand misses by in the plane.
        shift = np.zeros(3)
        for _ in range(3):
            kind, why = arm.move("robot.go_to_pose", grasp_p + shift, r_now, fine=True)
            if kind == "refused":
                return "unreachable", f"descent: {why}", r_now
            q, _ = arm.still()
            free_lags.append(float(np.linalg.norm(grasp_p + shift - q)))
            if float(q[2] - grasp_p[2]) > GRASP_STOP_TOLERANCE:
                return (
                    "missed",
                    f"the open jaws stopped {(q[2] - grasp_p[2]) * 1e3:.1f} mm over the grasp: on "
                    "the brick's top or a neighbour",
                    r_now,
                )
            err = grasp_p - q
            err[2] = 0.0
            if float(np.linalg.norm(err)) <= 2 * FINE_TOLERANCE:
                break
            shift = shift + err
        grasp_cmd = grasp_p + shift
        try:
            reply = ctx.tool(
                "robot.set_grip",
                object_width_m=float(jaw_width),
                squeeze_m=float(grip_squeeze),
                **arm.on_arm,
            )
            if _refusal(reply)[0] == "refused":
                raise RuntimeError(str(reply))
        except Exception as exc:
            logger.info("disassembling-a-brick: set_grip unavailable (%s); closing", exc)
            ctx.tool("robot.close_gripper", **arm.on_arm)
        time.sleep(GRIP_SETTLE_S)
        if jaws_hold() is False:
            return "missed", "the jaws closed past the brick's width: nothing between them", r_now
        if seat_press > 0:
            cmd = grasp_cmd - up * float(seat_press)
            arm.move("robot.go_to_pose", cmd, r_now, fine=True)
            q, _ = arm.still()
            out["seat_lag"] = float(q[2] - cmd[2])
            arm.move("robot.go_to_pose", grasp_cmd, r_now, fine=True)
            arm.still()
        return "", "", r_now

    def merge(first: dict[str, Any], then: dict[str, Any]) -> dict[str, Any]:
        """``then``'s outcome, with the larger peak lag and the releases of both."""
        merged = dict(then)
        merged["peak"] = max(first["peak"], then["peak"])
        merged["releases"] = list(first["releases"]) + list(then["releases"])
        return merged

    def release_and_rise() -> None:
        out["holding"] = False
        ctx.tool("robot.open_gripper", **arm.on_arm)
        p, r = arm.read()
        go(
            "robot.go_to_pose_cartesian",
            np.array([p[0], p[1], max(hover_z, p[2])]),
            r,
            check_rotation=False,
        )

    # -- the attempts ---------------------------------------------------------
    holding = False
    r_g = grasp_r
    last_fail = ("unreachable", "")
    ran_any = False
    for name in chain:
        if not holding:
            route, why, r_g = grasp()
            if route:
                if route == "missed":
                    release_and_rise()
                return finish(route, why + "; " + "; ".join(log) if log else why)
            holding = True
        out["strategy"] = name
        out["attempts"] += 1
        carry["loaded"] = False
        if name == "lever":
            pick = pick_pivot(r_g, float(twist_deg))
            if pick is None:
                log.append("lever: neither edge's lever passes the IK preview")
                out["attempts"] -= 1
                continue
            edge, piv, axis = pick
            out["pivot_side"] = edge
            poses = lever_poses(piv, axis, grasp_cmd, r_g, steps_to(float(twist_deg)))
            res = follow(grasp_cmd, poses, r_g)
            out["twist_deg"] = res["tilt"]
            if not res.get("refused") and not res["stalled"]:
                p_end, r_end, tilt_end = poses[-1]
                res = merge(res, follow(p_end, lift_poses(p_end, r_end, tilt_end), r_g))
        elif name == "wiggle":
            res = {"stalled": False, "peak": 0.0, "releases": [], "tilt": 0.0}
            for frac in WIGGLE_FRACTIONS:
                for _edge, piv, axis in pivots(r_g, only=False):
                    angles = steps_to(wiggle * frac)
                    there = lever_poses(piv, axis, grasp_cmd, r_g, angles)
                    back = lever_poses(piv, axis, grasp_cmd, r_g, angles[::-1][1:] + [0.0])
                    res = merge(res, follow(grasp_cmd, there + back, r_g))
                    if res.get("refused") or res["stalled"]:
                        break
                if res.get("refused") or res["stalled"]:
                    break
            out["pivot_side"] = "both"
            out["twist_deg"] = wiggle
            if not res.get("refused") and not res["stalled"]:
                res = merge(res, follow(grasp_cmd, lift_poses(grasp_cmd, r_g, 0.0), r_g))
        else:
            res = follow(grasp_cmd, lift_poses(grasp_cmd, r_g, 0.0), r_g)
        ran_any = True
        out["peak_lag"] = max(out["peak_lag"], float(res["peak"]))
        if res["releases"]:
            out["release_count"] += len(res["releases"])
            out["releases"] = "; ".join(
                filter(
                    None,
                    [out["releases"]]
                    + [f"{name} at {t:.1f} deg, {h * 1e3:.1f} mm up" for t, h in res["releases"]],
                )
            )
        if res.get("refused"):
            log.append(f"{name}: a step was refused ({res['refused']})")
            go("robot.go_to_pose", grasp_cmd, r_g, fine=True)
            last_fail = ("unreachable", f"{name}: a step was refused ({res['refused']})")
            continue
        if res["stalled"]:
            log.append(f"{name}: stuck ({res.get('why', 'lagging')})")
            go("robot.go_to_pose", grasp_cmd, r_g, fine=True, check_rotation=False)
            last_fail = ("stuck", f"{name}: {res.get('why', 'the arm lagged')}")
            continue
        # Free: up to the clearance, level, then the jaws and the depth.
        p, r = arm.read()
        go(
            "robot.go_to_pose_cartesian",
            np.array([p[0], p[1], max(clear_z, p[2])]),
            r,
            check_rotation=False,
        )
        p, _ = arm.read()
        go("robot.go_to_pose", p, r_g, check_rotation=False)
        hold = jaws_hold()
        out["holding"] = bool(hold) if hold is not None else False
        if hold is False:
            log.append(f"{name}: slipped (the jaws closed past the brick's width after the lift)")
            last_fail = ("slipped", f"{name}: the jaws are empty after the lift")
            holding = False
            if not regrasp_on_slip:
                break
            continue
        if not verify:
            log.append(f"{name}: lifted free")
            return finish(
                "unverified",
                f"{name}: lifted {lift_height * 1e3:.0f} mm without stalling"
                + (" and the jaws hold something" if hold else "")
                + "; not checked (verify=false)",
            )
        note = []
        after = look(note)
        if after is None:
            log.append(f"{name}: lifted free")
            return finish(
                "unverified",
                f"{name}: lifted free{' and the jaws hold something' if hold else ''}; "
                "the seat was not read: " + "; ".join(note),
            )
        tops = [h for h in after[0] if h is not None]
        centre = [h for h, c in zip(after[0], inner, strict=True) if h is not None and c]
        dz = float(np.median(centre if len(centre) >= MIN_SAMPLES else tops)) - top_z
        out["top_height_change"], out["samples"], out["verified_by"] = dz, len(tops), "wrist_depth"
        change = 0.0
        if before is not None:
            diffs = [
                abs(a - b)
                for a, b in zip(after[1], before[1], strict=True)
                if a is not None and b is not None
            ]
            change = max(diffs) if diffs else 0.0
        out["neighbour_change"] = change
        read = f"the seat reads {dz * 1e3:+.1f} mm from the brick's top over {len(tops)} points"
        if dz < -BELOW_FRACTION * body_h:
            log.append(f"{name}: the brick under it came too")
            return finish(
                "neighbour_lifted",
                f"{name}: {read}: the brick under it came off as well (still holding)",
            )
        if change > max(RING_CHANGE_FRACTION * stud_h, MIN_RING_CHANGE):
            log.append(f"{name}: a neighbour moved")
            return finish(
                "neighbour_lifted",
                f"{name}: {read}; a neighbouring cell moved {change * 1e3:.1f} mm (still holding)",
            )
        if dz < -GONE_FRACTION * body_h:
            log.append(f"{name}: removed")
            return finish(
                "removed",
                f"{name}: {read}: the seat is empty; the jaws hold the brick"
                + (f" (load released: {out['releases']})" if out["releases"] else ""),
            )
        log.append(f"{name}: slipped (the brick is still on the seat, {dz * 1e3:+.1f} mm)")
        last_fail = (
            "slipped",
            f"{name}: the hand lifted free but {read}: the brick stayed on the seat",
        )
        ctx.tool("robot.open_gripper", **arm.on_arm)
        holding = False
        if not regrasp_on_slip:
            break
    if not ran_any:
        if holding:
            release_and_rise()
        return finish(
            "unreachable", "no strategy's motion passes the IK preview: " + "; ".join(log)
        )
    if holding:
        release_and_rise()
    else:
        p, r = arm.read()
        go(
            "robot.go_to_pose_cartesian",
            np.array([p[0], p[1], max(hover_z, p[2])]),
            r,
            check_rotation=False,
        )
    route, why = last_fail
    return finish(route, why + "; the jaws are open over the seat")


def _turn_in_place(
    ctx: NodeContext, arm: _Arm, turn: float, tolerance: float, roll_limits: list[float] | None
) -> tuple[float, str]:
    """Turn the hand ``turn`` degrees about the vertical where it stands; read back, and
    finish on the last joint when the arm's solver left the wrist roll free (stacking-
    objects' turn, as in pressing-a-brick-onto-studs). Returns the turn made and a note."""
    p0, r0 = arm.read()
    kind, why = arm.move(
        "robot.go_to_pose", p0, _axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(turn)) @ r0
    )
    if kind == "refused":
        return 0.0, f"turn refused: {why}"
    gain = -1.0 if r0[2, 2] < 0 else 1.0
    lo, hi = (float(v) for v in (roll_limits or ROLL_LIMITS))
    note = ""
    for _ in range(TURN_TRIES):
        _, r = arm.read()
        residual = turn - _yaw_between(r0, r)
        if abs(residual) <= tolerance:
            break
        try:
            arms = ctx.tool("robot.get_observation").get("arms") or []
            js = arms[arm.arm_id].get("joint_state")
            q = np.asarray(
                js.get("positions") if isinstance(js, dict) else js, dtype=np.float64
            ).reshape(-1)
        except Exception as exc:
            note = f"no joint reading ({exc})"
            break
        command = math.radians(residual) / gain
        if not lo + LIMIT_MARGIN <= q[-1] + command <= hi - LIMIT_MARGIN:
            note = f"the last joint has no room to turn {residual:+.1f} deg"
            break
        target = q.copy()
        target[-1] += command
        try:
            reply = ctx.tool(
                "robot.move_to_joints", joint_config={"positions": target.tolist()}, **arm.on_arm
            )
            kind, why = _refusal(reply)
        except Exception as exc:
            kind, why = "refused", str(exc)
        if kind == "refused":
            note = f"wrist turn: {why}"
            break
        _, r2 = arm.read()
        moved = math.radians(_yaw_between(r, r2))
        if abs(command) > math.radians(0.5) and 0.5 < abs(moved / command) < 1.5:
            gain = moved / command
        arm.move("robot.go_to_pose", p0, r2, fine=True)
    _, r = arm.read()
    return _yaw_between(r0, r), note
