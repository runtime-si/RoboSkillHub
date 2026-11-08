# Adapted from BrickSim (https://github.com/intelligent-control-lab/BrickSim),
# python/bricksim/envs/assemble_brick/expert.py at commit
# cbd5de3238e6ac12f44ef699e1507a9f16bdafc3 (constants, _compute_assembly_state,
# _grasp_depth), MIT License, Copyright (c) 2026 Haowei Wen; and from
# Robotic_Lego_Manipulation (https://github.com/intelligent-control-lab/Robotic_Lego_Manipulation),
# src/ros_nodes/lego_task_planning_node.cpp (the place modes 7-10 and the
# attack-angle offsets) and config/user_config.json (Twist_Deg) at commit
# 9bc91654e708e4d8428ecfa76840103cacb2a19a, MIT License.
# Changes: BrickSim's memory-less expert reads privileged brick and target poses
# and a "connection created" flag every step; here the seat comes in as a world
# point (from perception), the brick's offset in the hand from its box measured
# after the grasp, and the steps run once, in order. BrickSim's descent gate
# (lateral error under 2 cm) and its fold of the measured lateral error into the
# press command are a re-commanded hover until the hand reads back within
# align_tolerance. Its press (a pose 3 mm under the seat, held until the
# simulator reports a connection) is a stepped descent judged by tracking lag
# (and joint efforts where the observation has them): it stops when the command
# leads the stopped hand by press_depth, which bounds the force a stiff position
# servo puts through a friction grip. Whether the brick seated is read after the
# release from the wrist depth camera (the height and flatness of the brick's
# top), not from a simulator flag. ICL's placement (offset hover, then over the
# seat, down, then a 14 deg twist about the tool's assemble frame that both
# presses and releases with their own end-of-arm tool) becomes the optional
# 'tilt' strategy for a parallel gripper: the brick is held tilted about its
# leading bottom edge, that edge is lowered among the studs, pulled back until
# its wall catches the stud row (which locates the brick along that direction),
# and the brick is rotated level about the edge before the straight press; the
# release twist is dropped. The yaw turn at the hover follows stacking-objects'
# (read back, finished on the last joint). Exits are routed instead of the
# policies' looping.
"""Press a held LEGO or DUPLO brick onto the studs under it, straight down or edge-first, and check its height."""

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
"""Brick systems [m]. LEGO: 8 mm pitch, 4.8 mm studs 1.7 mm high, 9.6 mm bricks
(BrickSim's BRICK_UNIT_LENGTH and 3 x PLATE_UNIT_HEIGHT). DUPLO: twice the
pitch, 9.4 mm studs 4.5 mm high, 19.2 mm bricks (published DUPLO dimensions)."""

FOOTPRINT_PLAY = 0.0001
"""How much shorter than n pitches a brick is on each side [m]: a 2x2 LEGO
brick is 15.8 mm, not 16."""

APPROACH_HEIGHT = 0.04
"""Hover over the seated pose [m]: BrickSim's ASSEMBLY_HEIGHT_OFFSET."""

PRESS_DEPTH = 0.003
"""How far the press command may lead the stopped hand [m]: BrickSim's
PRESS_DEPTH, the depth its press pose is under the seated pose. On a stiff
position servo the lead is the force: the press stops there."""

GRASP_DEPTH = 0.005
ONE_WIDE_GRASP_DEPTH_EXTRA = 0.001
"""How far under the brick's top the fingertips are when no box of the held
brick is given [m]: BrickSim's GRASP_DEPTH (6 mm for 1-wide bricks)."""

TIP_REACH = 0.0102
"""How far closed linear_4310 fingertips reach below the tool centre [m]
(the YAM station's gripper), for the hang when no held box is given."""

ALIGN_TOLERANCE = 0.0005
"""How close in the plane the hand must read back over its hover before it
descends [m]. BrickSim's descent gate is 2 cm (ASSEMBLY_XY_TOL) and it folds the
remaining error into the press pose; studs leave about a tenth of a millimetre,
so the fold is done before the descent, and to this bound."""

ALIGN_TRIES = 3
"""Re-commanded hovers (each shifted by the error read back) before the
hand's own offset is accepted as is."""

CONTACT_TOLERANCE = 0.006
"""How far over the stud tops the fast descent ends [m]: about one standard
deviation of a calibrated YAM station's camera error (RMS 5.4 mm), so an
estimated seat a few millimetres off is still met by the stepped part."""

FEEL_STEP_FRACTION = 0.25
MIN_FEEL_STEP = 0.0004
"""The stepped descent's step: a quarter of the stud height (1.1 mm DUPLO,
0.43 mm LEGO), at least 0.4 mm. Small enough to stop within half a stud of the
contact."""

LAG_FRACTION = 0.5
"""Contact is the hand standing more than this fraction of a step further
behind its command than in free motion (tuned on a YAM station twin, whose
servo arrives within FINE_TOLERANCE; a real YAM's lag noise is millimetres, see
detecting-contact-from-joint-effort's LAG_THRESHOLD = 3 mm)."""

EFFORT_THRESHOLD = 0.5
EFFORT_SIGMAS = 4.0
EFFORT_SAMPLES = 3
"""Joint-effort contact test where the observation carries efforts: a residual
over max(0.5 N.m, 4 sigma) of the free-motion windows, each the mean of 3 reads
(detecting-contact-from-joint-effort's test, from i2rt's clog check)."""

FINE_TOLERANCE = 0.0003
FINE_MAX_STEPS = 90
"""Arrival ball [m] and step budget (1.5 s of control steps) asked of each
fine move: a servo's default 4 mm ball would call a 1 mm step reached without
moving. Passed only to connectors that take them."""

MISALIGNED_FRACTION = 0.5
"""A brick whose bottom stops more than this fraction of the stud height over
the seat rests on the stud tops: its walls missed the gaps. Read from the hand
it is low by however far the brick slid up the jaws before the arm lagged
(1.5 mm on a YAM station twin with 4.5 mm DUPLO studs), hence half, not the
whole stud height."""

SEAT_FRACTION = 0.35
"""A brick top within this fraction of the stud height of the seated height
is seated (0.6 mm LEGO, 1.6 mm DUPLO): a brick resting on the studs reads a
whole stud height (1.7 / 4.5 mm) high."""

RESTING_FRACTION = 0.6
"""A released brick whose top reads this fraction of the stud height high or
more rests on the stud tops (the depth reading has no slide in it)."""

TILT_FRACTION = 0.5
"""A top whose readings spread more than this fraction of the stud height is
tilted (one side on the studs)."""

YAW_TOLERANCE_DEG = 1.0
"""Largest yaw between the held brick and the seat that is pressed [deg]: a
2x2 DUPLO's corner moves 0.4 mm per degree, past the studs' clearance."""

TURN_TRIES = 3
"""Last-joint turns tried after the hand's own turn, each read back (the
first learns which way the joint turns the hand): stacking-objects' value."""

ROLL_LIMITS = (-2.0944, 2.0944)
LIMIT_MARGIN = 0.05
"""The last joint's range when the caller states none [rad] (the i2rt YAM's
joint 6, +-120 deg) and the margin kept from it."""

TILT_DEG = 14.0
"""Tilt of the brick about its leading bottom edge [deg]: ICL's Twist_Deg.
Raised when the trailing edge would not clear the stud tops (TILT_CLEAR)."""

TILT_CLEAR = 1.25
"""The trailing edge is held at least this many stud heights up when the
leading edge is at the seat."""

TUBE_REACH_FRACTION = 0.58
TUBE_CLEAR = 1.1
"""How far in from the leading edge, in pitches, the first underside tube
starts (DUPLO 2x2: 9.4 mm of 16; LEGO 2x2: 4.65 mm of 8), and how many stud
heights over the plate the tilt keeps it while the edge hooks: a rigid tube
tilted lower lands on the leading row (measured on a YAM station twin with a
14 deg tilt: the edge stopped 1.8 mm over the plate, held up by the tube)."""

TILT_MAX_DEG = 27.0
"""Largest tilt [deg]: a YAM station twin's left arm reached 26 deg over the
seat but stopped 10 mm short of a 30 deg one."""

HOOK_OFFSET_FRACTION = 0.1
HOOK_REACH_FRACTION = 0.3
"""The tilted brick is lowered with its leading edge a tenth of a pitch past
the seat (outside the stud row: ICL lowers to an offset pose first), then pulled
back up to 0.4 pitch until its wall meets the row. Further than the gap to the
next row (3.4 mm DUPLO, 1.7 mm LEGO) would land the edge on that row."""

ENGAGE_FRACTION = 0.2
"""Height of the leading edge over what it met while it is pulled back, in
stud heights. The wall's inner bottom corner stands a wall thickness x sin(tilt)
higher (1.4 mm for DUPLO at 26 deg), and must stay under the stud tops to catch
the row: at 0.35 stud heights it rode over the row's chamfers on a YAM station
twin."""

FLOOR_SEARCH_FRACTION = 0.5
"""How far under the seat the tilted edge is sent looking for the plate, in
stud heights; it stops on contact."""

HOOK_BACKOFF = 0.0002
"""How far the edge is let off the stud row after the hook, so the rotation
does not drag along it [m]."""

ROTATE_STEPS = 4
"""Steps of the rotation from the tilt to level about the leading edge."""

ROTATION_TOLERANCE_DEG = 4.0
"""How far the hand's approach axis may read back from the commanded one
[deg]: a servo calls a rotation arrived within 3 deg (a real YAM station
within 0.04 rad), and some solvers leave the roll about that axis free."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from a free move's goal [m]: no tighter than a
real YAM station's own 12 mm per-move bound."""

HOLD_S = 0.5
"""Time the press is held before the release [s] (BrickSim holds its press
pose until the connection forms)."""

SETTLE_QUIET = 0.0002
SETTLE_PERIOD = 0.15
SETTLE_TIMEOUT = 2.0
"""The hand has settled when two reads SETTLE_PERIOD apart [s] differ by less
than SETTLE_QUIET [m], or after SETTLE_TIMEOUT [s]."""

VIEW_RANGES = (0.20, 0.25)
"""Wrist camera ranges tried for the height check [m]: a D405 reads nothing
under ~7 cm and is best inside 50 cm (refining-target-from-wrist-closeup)."""

MIN_VIEW_CLEARANCE = 0.05
"""Lowest tool centre over the brick's top for a view [m]."""

IMAGE_MARGIN = 0.08
"""Fraction of the image width / height a sample point must be inside."""

DEPTH_WINDOW = 1
MIN_DEPTH = 0.07
MAX_DEPTH = 0.5
MIN_SAMPLES = 3
"""Depth: median of the readings in 7-50 cm in a 3 x 3 window per sample point
on the brick's top (between the studs); at least 3 points must read."""

SAMPLE_INSET = 0.0015
STUD_MARGIN = 0.0008
"""Sample points stay this far inside the brick's edge and outside the studs
[m]."""

REFUSALS = ("unreachable", "bad_speed_scale")
"""Routed outcomes that mean a move was refused, not stopped."""

ROUTES = ("seated", "not_seated", "misaligned", "not_holding", "unverified", "unreachable")


class Output(TypedDict):
    route: str
    detail: str
    strategy: str
    released: bool
    seat_height_error: float
    seat_tilt: float
    contact_height: float
    lead_correction: float
    lateral_correction: float
    yaw_turn: float
    tilt_deg: float
    samples: int
    verified_by: str
    seated_position: dict[str, float]


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
    a = axis / np.linalg.norm(axis)
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


def _wrap(deg: float, period: float) -> float:
    return (deg + period / 2.0) % period - period / 2.0


def _refusal(reply: Any) -> tuple[str, str]:
    """("ok" | "short" | "refused", why) from a move's reply (connectors that
    route instead of raising)."""
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
    """The hand through gap's pose convention, with the fine-move knobs passed
    only to connectors that take them."""

    def __init__(self, ctx: NodeContext, arm_id: int) -> None:
        self.ctx = ctx
        self.on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
        self.knobs: dict[str, Any] = {
            "tolerance": FINE_TOLERANCE,
            "max_steps": FINE_MAX_STEPS,
        }

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
                    logger.info("pressing-a-brick: %s takes no fine-move knobs (%s)", tool, exc)
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


def _brick_axes(obb: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """The held box's horizontal axes (x, y, unit, in the plane), their full
    lengths, and its lowest corner's z."""
    rot = _rot(obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
    half = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    up = int(np.argmax(np.abs(rot[2, :])))
    horiz = [i for i in range(3) if i != up]
    axes = []
    for i in horiz:
        a = rot[:, i].copy()
        a[2] = 0.0
        axes.append(a / (np.linalg.norm(a) or 1.0))
    bottom = float(obb["center"]["z"]) - float(np.abs(rot[2, :]) @ half)
    return axes[0], axes[1], 2.0 * half[horiz[0]], 2.0 * half[horiz[1]], bottom


def _yaw_deg(axis: np.ndarray) -> float:
    return math.degrees(math.atan2(float(axis[1]), float(axis[0])))


def run(
    ctx: NodeContext,
    seat_position: dict[str, float],
    held_obb: dict[str, Any] | None = None,
    strategy: str = "straight",
    system: str = "duplo",
    brick_studs: list[int] | None = None,
    seat_yaw_deg: float | None = None,
    stud_pitch: float | None = None,
    stud_height: float | None = None,
    stud_diameter: float | None = None,
    brick_height: float | None = None,
    lead: str = "+x",
    tilt_deg: float = TILT_DEG,
    approach_height: float = APPROACH_HEIGHT,
    press_depth: float = PRESS_DEPTH,
    contact_tolerance: float = CONTACT_TOLERANCE,
    lag_threshold: float | None = None,
    align_tolerance: float = ALIGN_TOLERANCE,
    yaw_tolerance_deg: float = YAW_TOLERANCE_DEG,
    align_yaw: bool = True,
    roll_limits: list[float] | None = None,
    hold_s: float = HOLD_S,
    verify: bool = True,
    camera: str | None = None,
    check_holding: bool = True,
    check_reach: bool = True,
    reach_tolerance: float = REACH_TOLERANCE,
    arm_id: int = 0,
) -> Output:
    if strategy not in ("straight", "tilt"):
        raise ValueError(f"strategy must be 'straight' or 'tilt', got {strategy!r}")
    if system not in PRESETS:
        raise ValueError(f"system must be one of {sorted(PRESETS)}, got {system!r}")
    if lead not in ("+x", "-x", "+y", "-y"):
        raise ValueError(f"lead must be +x, -x, +y or -y, got {lead!r}")
    preset = PRESETS[system]
    pitch = float(stud_pitch if stud_pitch is not None else preset["stud_pitch"])
    stud_h = float(stud_height if stud_height is not None else preset["stud_height"])
    stud_d = float(stud_diameter if stud_diameter is not None else preset["stud_diameter"])
    body_h = float(brick_height if brick_height is not None else preset["brick_height"])
    step = max(MIN_FEEL_STEP, FEEL_STEP_FRACTION * stud_h)
    lag_thr = float(lag_threshold) if lag_threshold is not None else LAG_FRACTION * step

    arm = _Arm(ctx, arm_id)
    seat = _xyz(seat_position)
    tcp, rot0 = arm.read()

    out: Output = {
        "route": "unreachable",
        "detail": "",
        "strategy": strategy,
        "released": False,
        "seat_height_error": 0.0,
        "seat_tilt": 0.0,
        "contact_height": 0.0,
        "lead_correction": 0.0,
        "lateral_correction": 0.0,
        "yaw_turn": 0.0,
        "tilt_deg": 0.0,
        "samples": 0,
        "verified_by": "",
        "seated_position": _vec(seat),
    }

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        return out

    # The brick: its axes and footprint, and the hand's offset from its bottom
    # centre (measured now, carried unchanged: the hand keeps its rotation).
    if held_obb is not None:
        bx, by, len_x, len_y, bottom = _brick_axes(held_obb)
        centre = _xyz(held_obb["center"])
        offset = np.array([*(centre[:2] - tcp[:2]), bottom - tcp[2]])
    else:
        studs = list(brick_studs or [2, 2])
        len_x = studs[0] * pitch - 2 * FOOTPRINT_PLAY
        len_y = studs[1] * pitch - 2 * FOOTPRINT_PLAY
        yaw = math.radians(seat_yaw_deg or 0.0)
        bx = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        by = np.array([-bx[1], bx[0], 0.0])
        depth = GRASP_DEPTH + (ONE_WIDE_GRASP_DEPTH_EXTRA if min(studs) == 1 else 0.0)
        offset = np.array([0.0, 0.0, -(body_h + TIP_REACH - depth)])
    studs_xy = [
        int(brick_studs[0]) if brick_studs else max(1, round((len_x + 2 * FOOTPRINT_PLAY) / pitch)),
        int(brick_studs[1]) if brick_studs else max(1, round((len_y + 2 * FOOTPRINT_PLAY) / pitch)),
    ]

    # Yaw: the held brick's rows must run along the seat's (modulo its
    # symmetry). A larger error is turned out at the hover (align_yaw) or
    # refused before anything moves.
    yaw_err = 0.0
    if seat_yaw_deg is not None and held_obb is not None:
        square = abs(len_x - len_y) < 0.25 * pitch
        period = 90.0 if square else 180.0
        yaw_err = _wrap(float(seat_yaw_deg) - _yaw_deg(bx), period)
        if abs(yaw_err) > float(yaw_tolerance_deg) and not align_yaw:
            return finish(
                "misaligned",
                f"the held brick is {yaw_err:+.1f} deg off the seat's heading (tolerance "
                f"{yaw_tolerance_deg:.1f} deg) and align_yaw is false; nothing moved",
            )
        if abs(yaw_err) <= float(yaw_tolerance_deg):
            yaw_err = 0.0
        # The brick axis that will lie along the seat's x once turned.
        turn_rz = _axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(yaw_err))
        if square and abs(_wrap(float(seat_yaw_deg) - _yaw_deg(turn_rz @ by), 180.0)) < abs(
            _wrap(float(seat_yaw_deg) - _yaw_deg(turn_rz @ bx), 180.0)
        ):
            bx, by, len_x, len_y = by, -bx, len_y, len_x
            studs_xy = studs_xy[::-1]

    if check_holding:
        try:
            fraction = float(ctx.tool("robot.get_gripper", **arm.on_arm)["position"])
            grip = ctx.tool("robot.describe_gripper", **arm.on_arm)
            fit = grip.get("width_fit") or {}
            at_open = float(fit.get("at_open_m") or grip.get("span_m") or 0.0)
            at_closed = float(fit.get("at_closed_m") or 0.0)
            width = at_closed + (at_open - at_closed) * fraction
            if at_open > at_closed and width <= at_closed + 0.002:
                return finish(
                    "not_holding",
                    f"the jaws are at their stop ({width * 1e3:.1f} mm): no brick in the hand; nothing moved",
                )
        except Exception as exc:
            logger.info("pressing-a-brick: gripper not read (%s)", exc)

    hand_seat = seat - offset
    seat_tool_z = float(hand_seat[2])
    up = np.array([0.0, 0.0, 1.0])

    def lift_back(route: str, why: str) -> Output:
        """Not released: straight up to the hover height, still holding."""
        p, r = arm.read()
        kind, how = arm.move(
            "robot.go_to_pose_cartesian", np.array([p[0], p[1], seat_tool_z + approach_height]), r
        )
        tail = "; lifted back, still holding" if kind != "refused" else f"; lifting back: {how}"
        return finish(route, why + tail)

    def go(tool: str, p: np.ndarray, r: np.ndarray, check_rotation: bool = True) -> str:
        kind, why = arm.move(tool, p, r)
        q, s = arm.read()
        off = float(np.linalg.norm(q - p))
        ang = _axis_deg(r, s)
        turned = check_rotation and ang > ROTATION_TOLERANCE_DEG
        if kind == "refused" or off > float(reach_tolerance) or turned:
            return f"{why or 'ended off'}; the hand is {off * 1e3:.0f} mm / {ang:.1f} deg from it"
        return ""

    def aligned(goal: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, str]:
        """BrickSim's fold of the lateral error: re-command the pose shifted by
        what the hand missed by in the plane until it reads back within
        align_tolerance. Returns the command offset that got it there."""
        shift = np.zeros(3)
        miss = 0.0
        for _ in range(ALIGN_TRIES):
            kind, why = arm.move("robot.go_to_pose", goal + shift, r, fine=True)
            if kind == "refused":
                return shift, why
            p, _ = arm.still()
            err = goal - p
            err[2] = 0.0
            miss = float(np.linalg.norm(err))
            if miss <= float(align_tolerance):
                break
            shift = shift + err
        out["lateral_correction"] = float(np.linalg.norm(shift[:2]))
        if miss > float(align_tolerance):
            logger.info("pressing-a-brick: hover still %.2f mm off after the fold", miss * 1e3)
        return shift, ""

    efforts_on = False
    try:
        efforts_on = _joint_efforts(ctx.tool("robot.get_observation"), arm_id) is not None
    except Exception as exc:
        logger.info("pressing-a-brick: no observation (%s)", exc)

    def effort_window() -> np.ndarray | None:
        kept = []
        for _ in range(EFFORT_SAMPLES):
            e = _joint_efforts(ctx.tool("robot.get_observation"), arm_id)
            if e is not None:
                kept.append(e)
        return np.mean(kept, axis=0) if kept else None

    def feel(
        start: np.ndarray,
        direction: np.ndarray,
        distance: float,
        r: np.ndarray,
        press: bool = False,
    ) -> dict[str, Any]:
        """Fine steps from the command ``start`` along ``direction``. Contact is
        the hand lagging its command lag_thr more than in free motion (or a joint
        effort residual). Without ``press`` it stops there; with it, it goes on
        stepping until the command leads the hand by press_depth or the
        distance is done (the press, bounded by the lag a stiff servo needs to
        push that hard)."""
        p0, _ = arm.still()
        lags = [float(np.dot(start - p0, direction))]
        windows: list[np.ndarray] = []
        if efforts_on:
            w = effort_window()
            if w is not None:
                windows.append(w)
        n = max(1, math.ceil(distance / step - 1e-9))
        res: dict[str, Any] = {"contact": False, "at": p0, "cmd": start, "over": 0.0, "signal": ""}
        for i in range(1, n + 1):
            goal = start + direction * min(i * step, distance)
            kind, why = arm.move("robot.go_to_pose", goal, r, fine=True)
            if kind == "refused":
                res.update(refused=why)
                return res
            p, _ = arm.still()
            lag = float(np.dot(goal - p, direction))
            base = float(np.median(lags[-3:]))
            over = lag - base
            fired = over > lag_thr
            signal = "lag" if fired else ""
            if efforts_on and windows:
                w = effort_window()
                if w is not None:
                    noise = np.std(windows[-5:], axis=0) if len(windows) >= 3 else np.zeros_like(w)
                    thr = np.maximum(EFFORT_THRESHOLD, EFFORT_SIGMAS * noise)
                    if float(np.max(np.abs(w - np.mean(windows[-3:], axis=0)) / thr)) > 1.0:
                        fired, signal = True, "+".join(s for s in (signal, "effort") if s)
                    elif not fired:
                        windows.append(w)
            logger.debug(
                "feel step %d: goal %s hand %s lag %.2f mm over %.2f mm %s",
                i,
                np.round(goal * 1e3, 2).tolist(),
                np.round(p * 1e3, 2).tolist(),
                lag * 1e3,
                over * 1e3,
                signal or kind,
            )
            res.update(at=p, cmd=goal, over=over)
            if fired and not res["contact"]:
                res.update(contact=True, first=p.copy(), signal=signal)
                if not press:
                    return res
            if press and res["contact"] and over >= float(press_depth):
                return res
            if not fired:
                lags.append(lag)
        return res

    def press_and_release(start: np.ndarray, r: np.ndarray, corrected: np.ndarray) -> Output:
        """The straight press from the command ``start`` (a level hand over the
        seat), the hold, the release and the height check."""
        hand_bottom = corrected[2] - offset[2] - float(press_depth)
        res = feel(start, -up, max(step, float(start[2] - hand_bottom)), r, press=True)
        if "refused" in res:
            return lift_back("unreachable", f"a press step was refused: {res['refused']}")
        if res["contact"]:
            first = float(res["first"][2] + offset[2] - corrected[2])
            out["contact_height"] = first
            if first > MISALIGNED_FRACTION * stud_h:
                return lift_back(
                    "misaligned",
                    f"the brick stopped {first * 1e3:.1f} mm over the seat ({res['signal']}): "
                    f"on the stud tops (studs {stud_h * 1e3:.1f} mm), its walls missed the gaps",
                )
        else:
            out["contact_height"] = float(res["at"][2] + offset[2] - corrected[2])
        stop = float(res["at"][2] + offset[2] - corrected[2])
        how = (
            f"pressed to {stop * 1e3:+.1f} mm from the seat (first contact "
            f"{out['contact_height'] * 1e3:+.1f} mm, {res['signal']})"
            if res["contact"]
            else f"the press met no resistance down to {stop * 1e3:+.1f} mm from the seat"
            " (the brick slid in the jaws, or is not there)"
        )
        if hold_s > 0:
            time.sleep(float(hold_s))
        ctx.tool("robot.open_gripper", **arm.on_arm)
        out["released"] = True
        p, rr = arm.read()
        retreat = go(
            "robot.go_to_pose_cartesian",
            np.array([p[0], p[1], seat_tool_z + approach_height]),
            rr,
        )
        if retreat:
            how += f"; the retreat: {retreat}"
        if not verify:
            return finish("unverified", how + "; released; not checked (verify=false)")
        return check(corrected, how)

    def check(corrected: np.ndarray, how: str) -> Output:
        """The brick's top read with the wrist depth camera from a view over it."""
        top_z = float(corrected[2] + body_h)
        pts = _top_samples(corrected, bx, by, len_x, len_y, studs_xy, pitch, stud_d, top_z)
        target = np.array([corrected[0], corrected[1], top_z])
        try:
            obs = ctx.tool("robot.get_observation")
            tool_now = np.eye(4)
            tool_now[:3, :3] = _rot(obs["arms"][arm_id]["ee_pose"]["rotation"])
            tool_now[:3, 3] = _xyz(obs["arms"][arm_id]["ee_pose"])
            cam = _wrist_camera(obs, tool_now, camera)
        except Exception as exc:
            return finish(
                "unverified", how + f"; released; no observation for the height check ({exc})"
            )
        if cam is None:
            return finish(
                "unverified", how + "; released; no wrist camera with a pose to check the height"
            )
        tool_cam = np.linalg.inv(tool_now) @ _matrix(cam["pose"])
        tries: list[str] = []
        for view in _views(tool_now, tool_cam, target):
            if view[:3, 3][2] < top_z + MIN_VIEW_CLEARANCE:
                tries.append("a view under the clearance floor")
                continue
            if check_reach and not _ik_ok(ctx, view, arm.on_arm):
                tries.append("a view out of reach (IK preview)")
                continue
            # The view's roll is not checked: the frame is read with the pose
            # the camera reached, and the sample points must project into it.
            why = go("robot.go_to_pose", view[:3, 3], view[:3, :3], check_rotation=False)
            if why:
                tries.append(f"a view move: {why}")
                continue
            arm.still()
            obs = ctx.tool("robot.get_observation")
            cam = _wrist_camera(obs, tool_now, camera)
            if cam is None:
                tries.append("the wrist frame went missing")
                continue
            heights = _read_heights(cam, pts)
            if len(heights) < MIN_SAMPLES:
                tries.append(f"{len(heights)} depth readings on the brick's top")
                continue
            return judge(np.array(heights) - top_z, how)
        return finish("unverified", how + "; released; height not read: " + "; ".join(tries))

    def judge(dz: np.ndarray, how: str) -> Output:
        h = float(np.median(dz))
        spread = float(np.percentile(dz, 90) - np.percentile(dz, 10))
        out["seat_height_error"], out["seat_tilt"], out["samples"] = h, spread, int(dz.size)
        out["verified_by"] = "wrist_depth"
        read = (
            f"the top reads {h * 1e3:+.1f} mm from the seated height, spread {spread * 1e3:.1f} mm "
            f"over {dz.size} points"
        )
        if h < -0.5 * body_h:
            return finish("not_holding", f"{how}; released; {read}: no brick at the seat")
        if abs(h) <= SEAT_FRACTION * stud_h and spread <= TILT_FRACTION * stud_h:
            return finish("seated", f"{how}; released; {read}")
        if h >= RESTING_FRACTION * stud_h or spread > TILT_FRACTION * stud_h:
            return finish("misaligned", f"{how}; released; {read}: resting on the studs or tilted")
        return finish("not_seated", f"{how}; released; {read}: partly pressed")

    def turned(v: np.ndarray, deg: float) -> np.ndarray:
        """``v`` turned about the vertical by ``deg``."""
        return _axis_angle(up, math.radians(deg)) @ v

    # IK preview of the seated hand pose: a stall at the edge of reach reads
    # exactly like contact.
    if check_reach:
        m = np.eye(4)
        m[:3, :3] = _axis_angle(up, math.radians(yaw_err)) @ rot0
        m[:3, 3] = seat - np.array([*turned(offset, yaw_err)[:2], offset[2]])
        if not _ik_ok(ctx, m, arm.on_arm):
            return finish("unreachable", "the seated hand pose is out of reach (IK preview)")

    # Rise straight up first when under the hover (the hand holds a brick).
    hover_z = seat_tool_z + float(approach_height)
    if tcp[2] < hover_z - 1.0e-3:
        why = go("robot.go_to_pose_cartesian", np.array([tcp[0], tcp[1], hover_z]), rot0)
        if why:
            return finish("unreachable", f"rise: {why}")

    if yaw_err:
        done, why = _turn_in_place(ctx, arm, yaw_err, float(yaw_tolerance_deg), roll_limits)
        left = yaw_err - done
        out["yaw_turn"] = done
        if abs(left) > float(yaw_tolerance_deg):
            return finish(
                "misaligned",
                f"the held brick was {yaw_err:+.1f} deg off the seat's heading and the hand turned "
                f"{done:+.1f} deg ({why or 'short'}); still holding, at the hover",
            )
        _, rot0 = arm.read()
        offset = np.array([*turned(offset, done)[:2], offset[2]])
        bx, by = turned(bx, done), turned(by, done)
        hand_seat = seat - offset

    if strategy == "straight":
        hover = np.array([hand_seat[0], hand_seat[1], hover_z])
        why = go("robot.go_to_pose", hover, rot0)
        if why:
            return finish("unreachable", f"hover: {why}")
        shift, why = aligned(hover, rot0)
        if why:
            return finish("unreachable", f"hover fold: {why}")
        window = np.array([*(hand_seat[:2] + shift[:2]), seat_tool_z + stud_h + contact_tolerance])
        kind, why = arm.move("robot.move_cartesian_until_contact", window, rot0)
        p, _ = arm.still()
        if kind == "refused":
            return lift_back("unreachable", f"descent refused: {why}")
        early = float(p[2] - window[2])
        if kind == "short" and early > max(step, 0.002):
            out["contact_height"] = float(p[2] + offset[2] - seat[2])
            return lift_back(
                "misaligned",
                f"the descent stopped {out['contact_height'] * 1e3:.1f} mm over the seat, above the "
                "stud tops: the seat is higher than given or something is in the way",
            )
        start = np.array([window[0], window[1], p[2]])
        return press_and_release(start, rot0, seat)

    # Tilt: the leading bottom edge goes down first.
    sign = -1.0 if lead.startswith("-") else 1.0
    lvec = sign * (bx if lead.endswith("x") else by)
    length = len_x if lead.endswith("x") else len_y
    engage = ENGAGE_FRACTION * stud_h
    need = max(
        math.degrees(math.asin(min(1.0, TILT_CLEAR * stud_h / max(length, 1e-6)))),
        math.degrees(
            math.asin(min(1.0, (TUBE_CLEAR * stud_h - engage) / (TUBE_REACH_FRACTION * pitch)))
        ),
    )
    phi = math.radians(min(TILT_MAX_DEG, max(float(tilt_deg), need)))
    pivot_seat = seat + lvec * (length / 2.0)
    rel = hand_seat - pivot_seat
    axis = np.cross(up, lvec)

    def tilted(pivot: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
        rr = _axis_angle(axis, angle)
        return pivot + rr @ rel, rr @ rot0

    hook_offset = HOOK_OFFSET_FRACTION * pitch
    over_studs = float(seat[2] + stud_h + contact_tolerance)
    p_hover, r_tilt = tilted(pivot_seat + lvec * hook_offset + up * float(approach_height), phi)
    if check_reach:
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = r_tilt, tilted(pivot_seat, phi)[0]
        if not _ik_ok(ctx, m, arm.on_arm):
            return finish("unreachable", "the tilted pose at the seat is out of reach (IK preview)")
    # Level over the spot first, then tilted about the edge in small steps: a
    # brick held by fingertips slides in the jaws under one fast turning move
    # (3 mm on a YAM station twin turning 25 deg on the way to the hover).
    pivot_hover = pivot_seat + lvec * hook_offset + up * float(approach_height)
    why = go("robot.go_to_pose", tilted(pivot_hover, 0.0)[0], rot0)
    if why:
        return finish("unreachable", f"hover: {why}")
    for k in range(1, ROTATE_STEPS + 1):
        pk, rk = tilted(pivot_hover, phi * k / ROTATE_STEPS)
        kind, why = arm.move("robot.go_to_pose", pk, rk, fine=k == ROTATE_STEPS)
        if kind == "refused":
            return finish("unreachable", f"tilting step {k} refused: {why}")
    q, s = arm.still()
    off, ang = float(np.linalg.norm(q - p_hover)), _axis_deg(r_tilt, s)
    if off > float(reach_tolerance) or ang > ROTATION_TOLERANCE_DEG:
        return finish(
            "unreachable", f"tilted hover: the hand is {off * 1e3:.0f} mm / {ang:.1f} deg from it"
        )
    # From here the tilt is the one the hand reached (a servo calls a rotation
    # arrived within 3 deg): every hand pose keeps that rotation and is placed
    # for the edge, and the rotation back to level is measured from it.
    _, r_tilt = arm.read()
    t_axis, t_angle = _rotvec(r_tilt @ rot0.T)
    out["tilt_deg"] = math.degrees(t_angle)

    def at(pivot: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
        rr = _axis_angle(t_axis, t_angle * frac)
        return pivot + rr @ rel, rr @ rot0

    def edge(q: np.ndarray, r: np.ndarray) -> np.ndarray:
        """The leading bottom edge's midpoint under a measured hand pose."""
        return q - (r @ rot0.T) @ rel

    shift, why = aligned(
        at(pivot_seat + lvec * hook_offset + up * float(approach_height), 1.0)[0], r_tilt
    )
    if why:
        return finish("unreachable", f"tilted hover fold: {why}")
    above = at(pivot_seat + lvec * hook_offset + up * (over_studs - seat[2]), 1.0)[0] + shift
    kind, why = arm.move("robot.move_cartesian_until_contact", above, r_tilt)
    p, r = arm.still()
    if kind == "refused":
        return lift_back("unreachable", f"tilted descent refused: {why}")
    if kind == "short" and float(p[2] - above[2]) > max(step, 0.002):
        out["contact_height"] = float(edge(p, r)[2] - seat[2])
        return lift_back(
            "misaligned",
            f"the tilted descent stopped {out['contact_height'] * 1e3:.1f} mm over the seat, above "
            "the stud tops: the seat is higher than given or something is in the way",
        )
    # The edge goes down until it meets something (the plate between the stud
    # rows, or a stud top), so a brick turned a little in the jaws or a seat a
    # little off in height still puts it there; then it is raised to the hook
    # height over what it met.
    start = np.array([above[0], above[1], p[2]])
    down = float(edge(p, r)[2] - seat[2] + FLOOR_SEARCH_FRACTION * stud_h)
    res = feel(start, -up, max(step, down), r_tilt)
    if "refused" in res:
        return lift_back("unreachable", f"a tilted step was refused: {res['refused']}")
    if not res["contact"]:
        return lift_back(
            "misaligned",
            f"the leading edge went {FLOOR_SEARCH_FRACTION * stud_h * 1e3:.1f} mm under the seat "
            "without the arm lagging: the seat is lower than given, or the brick turned back "
            "toward level in the jaws instead of stalling the arm",
        )
    h_edge = float(edge(res["first"], r_tilt)[2] - seat[2])
    out["contact_height"] = h_edge
    if h_edge > MISALIGNED_FRACTION * stud_h:
        return lift_back(
            "misaligned",
            f"the leading edge stopped {h_edge * 1e3:.1f} mm over the seat: on a stud row, "
            "not in the gap beside it",
        )
    reach = hook_offset + HOOK_REACH_FRACTION * pitch
    hook_start = np.array([res["cmd"][0], res["cmd"][1], float(res["first"][2]) + engage])
    hook = feel(hook_start, -lvec, reach + step, r_tilt)
    if "refused" in hook:
        return lift_back("unreachable", f"a hook step was refused: {hook['refused']}")
    if not hook["contact"]:
        return lift_back(
            "misaligned",
            f"pulled the leading edge back {reach * 1e3:.1f} mm without meeting the stud row: "
            "the seat is off along the lead by more than the gap",
        )
    caught = edge(hook["first"], r_tilt)
    corr = float(np.dot(caught - pivot_seat, lvec))
    out["lead_correction"] = corr
    corrected = seat + lvec * corr
    out["seated_position"] = _vec(corrected)
    pivot = caught + lvec * HOOK_BACKOFF
    # The commands keep the hand's own offset in the plane found at the hover.
    for k in range(1, ROTATE_STEPS + 1):
        pk, rk = at(pivot, 1.0 - k / ROTATE_STEPS)
        kind, why = arm.move("robot.go_to_pose", pk + shift, rk, fine=True)
        if kind == "refused":
            return lift_back("unreachable", f"rotation step {k} refused: {why}")
    p, r_now = arm.still()
    left = _axis_deg(rot0, r_now)
    if left > ROTATION_TOLERANCE_DEG:
        return lift_back(
            "misaligned",
            f"the rotation to level stopped {left:.1f} deg short: the trailing side is held up",
        )
    level = at(pivot, 0.0)[0] + shift
    detail_hook = (
        f"tilted {out['tilt_deg']:.1f} deg; the edge caught the stud row {corr * 1e3:+.1f} mm "
        f"from the seat along the lead ({hook['signal']}); "
    )
    result = press_and_release(level, rot0, corrected)
    result["detail"] = detail_hook + result["detail"]
    return result


def _rotvec(r: np.ndarray) -> tuple[np.ndarray, float]:
    """Axis (unit) and angle [rad] of a rotation matrix."""
    angle = math.acos(min(1.0, max(-1.0, (float(np.trace(r)) - 1.0) / 2.0)))
    if angle < 1.0e-9:
        return np.array([1.0, 0.0, 0.0]), 0.0
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    n = float(np.linalg.norm(axis))
    if n < 1.0e-9:  # a half turn
        w, v = np.linalg.eigh((r + np.eye(3)) / 2.0)
        return v[:, int(np.argmax(w))], angle
    return axis / n, angle


def _yaw_between(r_from: np.ndarray, r_to: np.ndarray) -> float:
    """How far [deg] the hand turned about the vertical from ``r_from`` to ``r_to``."""
    rel = r_to @ r_from.T
    return math.degrees(math.atan2(float(rel[1, 0]), float(rel[0, 0])))


def _turn_in_place(
    ctx: NodeContext, arm: _Arm, turn: float, tolerance: float, roll_limits: list[float] | None
) -> tuple[float, str]:
    """Turn the hand ``turn`` degrees about the vertical where it stands; read
    back, and finish on the last joint when the arm's solver left the wrist roll
    free (stacking-objects' turn). Returns the turn made and a note."""
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
            js = arms[arm_id_of(arm)].get("joint_state")
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
        # The tool centre may be off the last joint's axis: back over the spot.
        arm.move("robot.go_to_pose", p0, r2, fine=True)
    _, r = arm.read()
    return _yaw_between(r0, r), note


def arm_id_of(arm: _Arm) -> int:
    return int(arm.on_arm.get("arm_id", 0))


def _matrix(pose: dict[str, Any]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rot(pose["rotation"])
    m[:3, 3] = _xyz(pose)
    return m


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
        logger.info("pressing-a-brick: no IK preview (%s)", exc)
        return True
    return not (
        isinstance(reply, dict)
        and (reply.get("success") is False or reply.get("within_servo_tolerance") is False)
    )


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
    """Hand poses putting ``target`` on the wrist camera's optical axis: looking
    straight down (the best depth), then with the hand's rotation as it is."""
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


def _top_samples(
    seat: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    len_x: float,
    len_y: float,
    studs: list[int],
    pitch: float,
    stud_d: float,
    top_z: float,
) -> list[np.ndarray]:
    """Points on the seated brick's top, inside its edge and clear of its studs."""
    centres = [
        ((i - (studs[0] - 1) / 2.0) * pitch, (j - (studs[1] - 1) / 2.0) * pitch)
        for i in range(studs[0])
        for j in range(studs[1])
    ]
    hx, hy = len_x / 2.0 - SAMPLE_INSET, len_y / 2.0 - SAMPLE_INSET
    keep = stud_d / 2.0 + STUD_MARGIN
    pts = []
    for u in np.linspace(-hx, hx, 2 * studs[0] + 1):
        for v in np.linspace(-hy, hy, 2 * studs[1] + 1):
            if all(math.hypot(u - a, v - b) > keep for a, b in centres):
                p = seat + bx * u + by * v
                pts.append(np.array([p[0], p[1], top_z]))
    return pts


def _read_heights(cam: dict[str, Any], pts: list[np.ndarray]) -> list[float]:
    """World heights the wrist depth reads at the pixels the sample points project to."""
    depth = np.asarray(cam["depth"], dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    m = _matrix(cam["pose"])
    inv = np.linalg.inv(m)
    h, w = depth.shape[:2]
    out = []
    for p in pts:
        pc = inv @ np.append(p, 1.0)
        if pc[2] <= 1.0e-6:
            continue
        u = k[0, 0] * pc[0] / pc[2] + k[0, 2]
        v = k[1, 1] * pc[1] / pc[2] + k[1, 2]
        if not (
            IMAGE_MARGIN * w <= u <= (1 - IMAGE_MARGIN) * w
            and IMAGE_MARGIN * h <= v <= (1 - IMAGE_MARGIN) * h
        ):
            continue
        col, row = int(round(u)), int(round(v))
        win = depth[
            max(0, row - DEPTH_WINDOW) : row + DEPTH_WINDOW + 1,
            max(0, col - DEPTH_WINDOW) : col + DEPTH_WINDOW + 1,
        ]
        good = win[np.isfinite(win) & (win >= MIN_DEPTH) & (win <= MAX_DEPTH)]
        if not good.size:
            continue
        z = float(np.median(good))
        q = m @ np.array([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z, 1.0])
        out.append(float(q[2]))
    return out
