# Reimplemented from the behaviour of Mark Rober's puzzle robot
# (https://github.com/markroberyoutube/puzzle_bot, src/application.py: the
# fully / nearly placed thresholds, the pick-up-and-wiggle grid and spiral and
# the scoot-and-wiggle, read at commit HEAD of 2024). That repository carries no
# licence, so no code was copied: the algorithm was re-expressed from its
# description. Changes: the robot's spring-mounted Z sled and linear encoder
# (passive compliance plus a height reading) become a stepped descent judged by
# tracking lag (and joint efforts where the observation carries them) that
# presses only until the command leads the stopped hand by press_depth; the
# encoder counts become millimetres of the piece's thickness; the grid / ring
# spiral becomes probes ordered by how likely each (x, y, yaw) is under the
# caller's error estimate; the suction cup's rotation stage becomes the arm's
# last joint; flushness is read from the wrist depth camera after the release.
# The pose, lag, depth and wrist-camera helpers follow pressing-a-brick-onto-studs.
# approach="tilt_first" re-expresses Watson, Miller and Correll's InsertTilt
# (arXiv:2002.02580, Algorithm 2; paper only); approach="offset_push" is the
# jigsaw design note's place-short-then-push, no source code.
"""Seat a held jigsaw piece in its slot among placed neighbours by probing small x / y / yaw offsets until it drops."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

THICKNESS = 0.002
"""Piece thickness [m]. Mass-market cardboard pieces are 1.8-2.2 mm; puzzle_bot
measured 1408-1459 encoder counts on its pieces and set its minimum at 1400."""

SEATED_FRACTION = 0.25
"""A piece whose bottom stops within this fraction of its thickness of the mat
(0.5 mm for 2 mm) has dropped into its slot. puzzle_bot calls a piece fully set
within 10% of the thickness on its encoder; an arm's stop height is coarser
than an encoder, so a quarter (still half of the 2 mm a piece on its
neighbours stands up)."""

NEARLY_FRACTION = 0.6
"""A piece that stops under this fraction of its thickness is partly in (one
side down, or a tab caught on a neighbour's blank): the search switches to
scooting at a light press there. puzzle_bot's first value for its nearly set
threshold (it later used 25% with its encoder)."""

FEEL_STEP = 0.0004
"""Stepped descent's step [m]: a fifth of the thickness, so a drop of one
thickness is at least four steps and a stop is located to under half a step."""

LAG_FRACTION = 0.5
"""Contact is the hand standing more than this fraction of a step further
behind its command than in free motion: pressing-a-brick-onto-studs' value,
tuned on a YAM station twin whose servo arrives within FINE_TOLERANCE. A real
YAM's lag noise is millimetres (detecting-contact-from-joint-effort's
LAG_THRESHOLD = 3 mm), more than a piece's thickness; there the arm's stop
cannot tell resting from seated and the wrist depth check decides."""

PRESS_DEPTH = 0.0
"""How far the command may lead the stopped hand past the first contact [m]; on
a stiff position servo the lead is the force. 0: stop at the contact. On the
YAM station twin a 0.6 mm lead (puzzle_bot's slight-press height) pushed the
piece 0.6-0.8 mm up in the jaws, which then read on every later stop as height."""

EFFORT_THRESHOLD = 0.5
EFFORT_SIGMAS = 4.0
EFFORT_SAMPLES = 3
"""Joint-effort contact test where the observation carries efforts: a residual
over max(0.5 N.m, 4 sigma) of the free-motion windows, each the mean of 3 reads
(detecting-contact-from-joint-effort's test, from i2rt's clog check)."""

APPROACH_HEIGHT = 0.015
"""Hover of the piece's bottom over the slot before the first descent [m]."""

CONTACT_WINDOW = 0.004
"""How far over the neighbours' tops the fast first descent ends [m]: covers a
slot height a few millimetres off; the stepped descent does the rest."""

LIFT_CLEARANCE = 0.0008
"""How far over the highest stop seen the piece is lifted between probes [m]:
clears the neighbours' tops without a long descent back."""

OVERTRAVEL_FRACTION = 1.0
"""How far under the mat, in thicknesses, a descent may go without contact
before the piece is called lost (out of the hand) [m / m]."""

SIGMA_XY = 0.001
SIGMA_YAW_DEG = 2.0
"""The slot pose's expected error (one standard deviation) along x / y [m] and
in yaw [deg]: a wrist close-up (refining-target-from-wrist-closeup) puts a
target within about a millimetre; localizing-a-piece-in-a-reference-image's
angle search runs in steps of a few degrees. Probes are taken in order of
(r / SIGMA_XY)^2 + (yaw / SIGMA_YAW_DEG)^2, most likely first."""

PITCH = 0.0004
"""Spacing of the x / y probe grid [m] (hexagonal). A slot with 0.2-0.4 mm of
clearance takes the piece within about the clearance, and a hexagonal grid
leaves every point within pitch / sqrt(3) = 0.23 mm of a probe. puzzle_bot's
grid stepped 0.25 mm (10 thou)."""

SEARCH_RADIUS = 0.0025
"""Largest x / y offset probed [m] (puzzle_bot's grid: +-2.5 mm)."""

YAW_STEP_DEG = 1.0
YAW_RANGE_DEG = 4.0
"""Yaw offsets probed [deg]: steps of 1 deg up to +-4. A 20 mm piece's tab tips
are about 14 mm from its centre, so 1 deg moves them 0.24 mm, about the slot's
clearance."""

MAX_PROBES = 40
"""Probes before the search gives up (each a lift, a move and a descent)."""

SCOOT = 0.0003
SCOOT_GROW = 0.0002
SCOOT_ROUNDS = 3
SCOOT_YAW_DEG = 0.75
"""Scoot-and-wiggle at a light press [m, deg]: from the best probe, out and
back along x, then y, then a yaw out and back, widening each round.
puzzle_bot's scoot: 50 counts widening by 25, up to 15 rounds of 6 moves."""

FINE_TOLERANCE = 0.0002
FINE_MAX_STEPS = 90
"""Arrival ball [m] and step budget asked of each fine move: a servo's default
4 mm ball would call a 0.4 mm step reached without moving. Passed only to
connectors that take them."""

ALIGN_TOLERANCE = 0.00015
ALIGN_TRIES = 2
"""A probe's hand must read back within this of its point in the plane [m];
the command is shifted by the miss and re-sent up to this many times."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from a free move's goal [m]: no tighter than a real
YAM station's own 12 mm per-move bound."""

TURN_TRIES = 3
YAW_TOLERANCE_DEG = 0.3
ROLL_LIMITS = (-2.0944, 2.0944)
LIMIT_MARGIN = 0.05
"""Yaw turns on the last joint: tries (the first learns which way the joint
turns the hand, as stacking-objects does), how close to the asked yaw [deg],
the joint's range when none is given (i2rt YAM joint 6, +-120 deg) and the
margin kept from it [rad]."""

HOLD_S = 0.3
"""Pause at the seat before the jaws open [s]."""

SETTLE_QUIET = 0.0001
SETTLE_PERIOD = 0.1
SETTLE_TIMEOUT = 2.0
"""The hand has settled when two reads SETTLE_PERIOD apart [s] differ by less
than SETTLE_QUIET [m], or after SETTLE_TIMEOUT [s]."""

TIP_REACH = 0.0102
"""How far closed linear_4310 fingertips reach below the tool centre [m]: the
default hang of the piece's bottom when no held_offset is given."""

FLUSH_FRACTION = 0.25
TILT_FRACTION = 0.25
"""After the release a top within this fraction of the thickness of the seated
height (0.5 mm), its readings spreading less than the same, is flush."""

TOP_RADIUS = 0.005
"""Radius around the seated piece's centre sampled on its top [m]: inside the
body of a 20 mm piece, clear of its knobs and blanks."""

VIEW_RANGES = (0.15, 0.20, 0.25)
"""Wrist camera ranges tried for the flush check [m]: a D405 reads nothing
under ~7 cm and is best inside 50 cm (refining-target-from-wrist-closeup)."""

MIN_VIEW_CLEARANCE = 0.05
IMAGE_MARGIN = 0.08
DEPTH_WINDOW = 1
MIN_DEPTH = 0.07
MAX_DEPTH = 0.5
MIN_SAMPLES = 5
"""Lowest tool centre over the piece for a view [m]; sample points must project
inside the image by this margin; depth is the median of the 3 x 3 window per
point within 7-50 cm; at least 5 points must read."""

TILT_DEG = 10.0
TILT_BACK = 0.003
PIECE_RADIUS = 0.010
"""approach="tilt_first": the piece tilted this much [deg] about the horizontal
axis across seam_direction, leading side low (the jigsaw design note's 8-15 deg;
Watson et al.'s InsertTilt), first lowered this far short of its slot [m] (a few
mm), its leading edge taken this far from its reference point [m] (half a 20 mm
piece) to know how much lower than the reference it hangs."""

PUSH_OFFSET = 0.004
PUSH_CLEAR = 0.003
PUSH_TIP_HEIGHT = 0.001
PUSH_OVER = 0.0005
PUSH_SLACK = 0.001
"""approach="offset_push": the piece is set down this far short of its slot [m]
(the design note's 3-6 mm), the closed fingertips start this far behind its
trailing edge [m], 1 mm over the mat (the design note), push through the slot by
PUSH_OVER more, and a push stopping more than PUSH_SLACK short of the slot is
jammed [m]."""

APPROACHES = ("wiggle", "tilt_first", "offset_push")

REFUSALS = ("unreachable", "bad_speed_scale")
"""Routed outcomes that mean a move was refused, not stopped."""

ROUTES = ("seated", "not_seated", "jammed", "piece_lost", "unverified", "unreachable")


class _Done(Exception):
    """Ends the skill early with its output."""

    def __init__(self, out: Any) -> None:
        super().__init__(out.get("route", ""))
        self.out = out


class Output(TypedDict):
    route: str
    detail: str
    released: bool
    probes: int
    moves: int
    elapsed_s: float
    stop_height: float
    best_height: float
    resting_height: float
    offset_found: dict[str, float]
    yaw_found_deg: float
    seated_position: dict[str, float]
    flush_error: float
    flush_spread: float
    samples: int
    verified_by: str


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


def _rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _about(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rotation by ``deg`` about the unit vector ``axis`` (Rodrigues)."""
    k = np.asarray(axis, float) / (float(np.linalg.norm(axis)) or 1.0)
    a = math.radians(deg)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * kx + (1.0 - math.cos(a)) * kx @ kx


def _yaw_between(r_from: np.ndarray, r_to: np.ndarray) -> float:
    """How far [deg] the hand turned about the vertical from ``r_from`` to ``r_to``."""
    rel = r_to @ r_from.T
    return math.degrees(math.atan2(float(rel[1, 0]), float(rel[0, 0])))


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


def _joint_state(obs: Any, arm_id: int) -> dict[str, Any] | None:
    if not isinstance(obs, dict):
        return None
    arms = obs.get("arms")
    if isinstance(arms, list) and len(arms) > arm_id and isinstance(arms[arm_id], dict):
        js = arms[arm_id].get("joint_state")
        if isinstance(js, dict):
            return js
    js = obs.get("joint_state")
    return js if isinstance(js, dict) else None


def _joint_efforts(obs: Any, arm_id: int) -> np.ndarray | None:
    js = _joint_state(obs, arm_id)
    if js is None:
        return None
    for name in ("efforts", "effort", "torques"):
        value = js.get(name)
        if value is not None:
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size:
                return arr
    return None


def search_order(
    radius: float,
    pitch: float,
    yaw_range: float,
    yaw_step: float,
    sigma_xy: float,
    sigma_yaw: float,
) -> list[tuple[float, float, float]]:
    """(dx, dy, dyaw_deg) offsets of a hexagonal x / y grid inside ``radius``
    times yaw steps inside ``yaw_range``, most likely first under independent
    Gaussian errors; ties go round a ring by angle and prefer +yaw."""
    xy = [(0.0, 0.0)]
    if pitch > 0.0 and radius > 0.0:
        n = int(math.ceil(radius / pitch)) + 1
        xy = []
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                x = pitch * (i + 0.5 * j)
                y = pitch * (math.sqrt(3.0) / 2.0) * j
                if math.hypot(x, y) <= radius + 1.0e-9:
                    xy.append((x, y))
    yaws = [0.0]
    if yaw_step > 0.0:
        k = 1
        while k * yaw_step <= yaw_range + 1.0e-9:
            yaws += [k * yaw_step, -k * yaw_step]
            k += 1
    sx, sy = max(sigma_xy, 1.0e-9), max(sigma_yaw, 1.0e-9)
    cands = [(x, y, a) for a in yaws for (x, y) in xy]
    return sorted(
        cands,
        key=lambda c: (
            round((math.hypot(c[0], c[1]) / sx) ** 2 + (c[2] / sy) ** 2, 6),
            round(math.hypot(c[0], c[1]), 9),
            -c[2],
            math.atan2(c[1], c[0]) % (2 * math.pi),
        ),
    )


class _Arm:
    """The hand through gap's pose convention, with the fine-move knobs passed
    only to connectors that take them, and a move counter."""

    def __init__(self, ctx: NodeContext, arm_id: int) -> None:
        self.ctx = ctx
        self.arm_id = int(arm_id)
        self.on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
        self.knobs: dict[str, Any] = {"tolerance": FINE_TOLERANCE, "max_steps": FINE_MAX_STEPS}
        self.moves = 0

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        pose = _pose(self.ctx.tool("robot.get_ee_pose", **self.on_arm))
        return _xyz(pose), _rot(pose["rotation"])

    def move(self, tool: str, p: np.ndarray, r: np.ndarray, fine: bool = False) -> tuple[str, str]:
        pose = {"position": _vec(p), "rotation": _quat(r)}
        self.moves += 1
        try:
            if fine and self.knobs:
                try:
                    reply = self.ctx.tool(tool, pose=pose, **self.knobs, **self.on_arm)
                except TypeError as exc:
                    if "keyword" not in str(exc):
                        raise
                    logger.info("seating-a-piece: %s takes no fine-move knobs (%s)", tool, exc)
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

    def joints(self) -> np.ndarray | None:
        try:
            js = _joint_state(self.ctx.tool("robot.get_observation"), self.arm_id)
        except Exception as exc:
            logger.info("seating-a-piece: no joint reading (%s)", exc)
            return None
        if js is None or js.get("positions") is None:
            return None
        return np.asarray(js["positions"], dtype=np.float64).reshape(-1)


def run(
    ctx: NodeContext,
    slot_position: dict[str, float],
    held_offset: dict[str, float] | None = None,
    yaw_error_deg: float = 0.0,
    thickness: float = THICKNESS,
    sigma_xy: float = SIGMA_XY,
    sigma_yaw_deg: float = SIGMA_YAW_DEG,
    search_radius: float = SEARCH_RADIUS,
    pitch: float = PITCH,
    yaw_range_deg: float = YAW_RANGE_DEG,
    yaw_step_deg: float = YAW_STEP_DEG,
    max_probes: int = MAX_PROBES,
    max_time_s: float = 0.0,
    scoot: bool = True,
    press_depth: float = PRESS_DEPTH,
    lag_threshold: float | None = None,
    approach_height: float = APPROACH_HEIGHT,
    contact_window: float = CONTACT_WINDOW,
    release_when_not_seated: bool = False,
    hold_s: float = HOLD_S,
    verify: bool = True,
    camera: str | None = None,
    top_radius: float = TOP_RADIUS,
    roll_limits: list[float] | None = None,
    check_reach: bool = True,
    reach_tolerance: float = REACH_TOLERANCE,
    approach: str = "wiggle",
    seam_direction: dict[str, float] | None = None,
    tilt_deg: float = TILT_DEG,
    tilt_back: float = TILT_BACK,
    piece_radius: float = PIECE_RADIUS,
    push_offset: float = PUSH_OFFSET,
    finger_reach: float = TIP_REACH,
    arm_id: int = 0,
) -> Output:
    t_start = time.monotonic()
    thick = float(thickness)
    if thick <= 0.0:
        raise ValueError(f"thickness must be positive, got {thickness!r}")
    if approach not in APPROACHES:
        raise ValueError(f"approach must be one of {APPROACHES}, got {approach!r}")
    s_dir = np.zeros(3)
    if approach != "wiggle":
        if seam_direction is None:
            raise ValueError(f"approach={approach!r} needs seam_direction (toward the neighbours)")
        s_dir = _xyz(seam_direction) * np.array([1.0, 1.0, 0.0])
        if float(np.linalg.norm(s_dir)) < 1.0e-9:
            raise ValueError("seam_direction must have a horizontal part")
        s_dir /= float(np.linalg.norm(s_dir))
    step = FEEL_STEP
    lag_thr = float(lag_threshold) if lag_threshold is not None else LAG_FRACTION * step
    seated_tol, nearly_tol = SEATED_FRACTION * thick, NEARLY_FRACTION * thick

    arm = _Arm(ctx, arm_id)
    slot = _xyz(slot_position)
    p_start, r0 = arm.read()
    offset = _xyz(held_offset) if held_offset is not None else np.array([0.0, 0.0, -TIP_REACH])
    up = np.array([0.0, 0.0, 1.0])

    out: Output = {
        "route": "unreachable",
        "detail": "",
        "released": False,
        "probes": 0,
        "moves": 0,
        "elapsed_s": 0.0,
        "stop_height": 0.0,
        "best_height": 0.0,
        "resting_height": 0.0,
        "offset_found": {"x": 0.0, "y": 0.0, "z": 0.0},
        "yaw_found_deg": 0.0,
        "seated_position": _vec(slot),
        "flush_error": 0.0,
        "flush_spread": 0.0,
        "samples": 0,
        "verified_by": "",
    }

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        out["moves"] = arm.moves
        out["elapsed_s"] = float(time.monotonic() - t_start)
        return out

    # The hand's yaw relative to where it started (r0), and the piece's offset
    # turned with it: the piece hangs at offset when the hand is at r0.
    state = {"yaw": 0.0, "gain": -1.0 if r0[2, 2] < 0 else 1.0}

    def hand_for(piece_xy: np.ndarray, yaw: float, bottom_z: float) -> np.ndarray:
        off = _rz(yaw) @ offset
        return np.array([piece_xy[0] - off[0], piece_xy[1] - off[1], bottom_z - offset[2]])

    def piece_at(p: np.ndarray, r: np.ndarray | None = None) -> np.ndarray:
        """Where the piece's reference point is under a measured hand position
        (and rotation, when the hand is not level)."""
        if r is None:
            return p + _rz(state["yaw"]) @ offset
        return p + r @ r0.T @ offset

    def hand_at(ref: np.ndarray, r: np.ndarray) -> np.ndarray:
        """The hand position putting the piece's reference at ``ref`` under rotation ``r``."""
        return ref - r @ r0.T @ offset

    def height(p: np.ndarray) -> float:
        """The piece's bottom over the mat under a measured hand position."""
        return float(p[2] + offset[2] - slot[2])

    def rot(yaw: float) -> np.ndarray:
        return _rz(yaw) @ r0

    def ik_ok(p: np.ndarray, r: np.ndarray) -> bool:
        try:
            reply = ctx.tool(
                "robot.solve_ik", pose={"position": _vec(p), "rotation": _quat(r)}, **arm.on_arm
            )
        except Exception as exc:
            if "unreachable" in str(exc).lower():
                return False
            logger.info("seating-a-piece: no IK preview (%s)", exc)
            return True
        return not (
            isinstance(reply, dict)
            and (reply.get("success") is False or reply.get("within_servo_tolerance") is False)
        )

    def go(tool: str, p: np.ndarray, r: np.ndarray) -> str:
        kind, why = arm.move(tool, p, r)
        q, _ = arm.read()
        off = float(np.linalg.norm(q - p))
        if kind == "refused" or off > float(reach_tolerance):
            return f"{why or 'ended off'}; the hand is {off * 1e3:.0f} mm from it"
        return ""

    def turn(target: float) -> str:
        """Turn the hand to ``target`` deg of yaw from r0 about the vertical where
        it stands: the last joint (read back, the gain learnt), else a rotation
        command. Returns "" or why it stopped short."""
        note = ""
        for _ in range(TURN_TRIES):
            p, r = arm.read()
            now = _yaw_between(r0, r)
            residual = target - now
            state["yaw"] = now
            if abs(residual) <= YAW_TOLERANCE_DEG:
                return ""
            q = arm.joints()
            if q is None or q.size == 0:
                kind, why = arm.move("robot.go_to_pose", p, rot(target), fine=True)
                _, r2 = arm.read()
                state["yaw"] = _yaw_between(r0, r2)
                return "" if abs(target - state["yaw"]) <= YAW_TOLERANCE_DEG else (why or "short")
            lo, hi = (float(v) for v in (roll_limits or ROLL_LIMITS))
            command = math.radians(residual) / state["gain"]
            if not lo + LIMIT_MARGIN <= q[-1] + command <= hi - LIMIT_MARGIN:
                return f"the last joint has no room to turn {residual:+.1f} deg"
            goal = q.copy()
            goal[-1] += command
            arm.moves += 1
            try:
                kind, why = _refusal(
                    ctx.tool(
                        "robot.move_to_joints",
                        joint_config={"positions": goal.tolist()},
                        **arm.on_arm,
                    )
                )
            except Exception as exc:
                kind, why = "refused", str(exc)
            if kind == "refused":
                return f"wrist turn: {why}"
            _, r2 = arm.read()
            moved = math.radians(_yaw_between(r, r2))
            if abs(command) > math.radians(0.3) and 0.5 < abs(moved / command) < 1.5:
                state["gain"] = moved / command
            state["yaw"] = _yaw_between(r0, r2)
            note = ""
        return note or ("" if abs(target - state["yaw"]) <= YAW_TOLERANCE_DEG else "short")

    def place(piece_xy: np.ndarray, bottom_z: float) -> tuple[np.ndarray, str]:
        """Fine move putting the piece's reference over ``piece_xy`` with its
        bottom at ``bottom_z`` (hand at the current yaw), the miss in the plane
        folded into the command. Returns the measured hand position."""
        _, r = arm.read()
        state["yaw"] = _yaw_between(r0, r)
        goal = hand_for(piece_xy, state["yaw"], bottom_z)
        shift = np.zeros(3)
        p = goal
        for k in range(ALIGN_TRIES + 1):
            kind, why = arm.move("robot.go_to_pose", goal + shift, r, fine=True)
            if kind == "refused":
                return p, why
            p, r_read = arm.still()
            # A wrist that drifted in yaw swings the piece round the tool centre: re-aim.
            state["yaw"] = _yaw_between(r0, r_read)
            goal = hand_for(piece_xy, state["yaw"], bottom_z)
            err = goal - p
            err[2] = 0.0
            if float(np.linalg.norm(err)) <= ALIGN_TOLERANCE or k == ALIGN_TRIES:
                break
            shift = shift + err
        return p, ""

    efforts_on = False
    try:
        efforts_on = _joint_efforts(ctx.tool("robot.get_observation"), arm.arm_id) is not None
    except Exception as exc:
        logger.info("seating-a-piece: no observation (%s)", exc)

    def effort_window() -> np.ndarray | None:
        kept = []
        for _ in range(EFFORT_SAMPLES):
            e = _joint_efforts(ctx.tool("robot.get_observation"), arm.arm_id)
            if e is not None:
                kept.append(e)
        return np.mean(kept, axis=0) if kept else None

    def feel(distance: float, press: bool = True) -> dict[str, Any]:
        """Fine steps straight down from where the hand stands. Contact is the
        hand lagging its command lag_thr more than in free motion (or a joint
        effort residual); with ``press`` it goes on until the command leads the
        hand by press_depth (a light press), else it stops there."""
        p0, r = arm.still()
        start = p0.copy()
        lags = [0.0]
        windows: list[np.ndarray] = []
        if efforts_on:
            w = effort_window()
            if w is not None:
                windows.append(w)
        n = max(1, math.ceil(distance / step - 1e-9))
        res: dict[str, Any] = {"contact": False, "at": p0, "cmd": start, "over": 0.0, "signal": ""}
        for i in range(1, n + 1):
            goal = start - up * min(i * step, distance)
            kind, why = arm.move("robot.go_to_pose", goal, r, fine=True)
            if kind == "refused":
                res.update(refused=why)
                return res
            p, _ = arm.still()
            lag = float(p[2] - goal[2])
            over = lag - float(np.median(lags[-3:]))
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
                "feel step %d: goal z %.2f mm hand z %.2f mm over %.2f mm %s",
                i,
                goal[2] * 1e3,
                p[2] * 1e3,
                over * 1e3,
                signal or kind,
            )
            res.update(at=p, cmd=goal, over=over)
            if fired and not res["contact"]:
                res.update(contact=True, first=p.copy(), signal=signal)
                if not press:
                    return res
            if res["contact"] and over >= float(press_depth):
                return res
            if not fired:
                lags.append(lag)
        return res

    def unpress() -> np.ndarray:
        """Stop pressing: command the hand where it stands."""
        p, r = arm.still()
        arm.move("robot.go_to_pose", p, r, fine=True)
        return arm.still()[0]

    def lift_back(route: str, why: str) -> Output:
        """Not released: straight up to the hover, still holding. A hand that
        cannot rise out of the slot is jammed."""
        p, r = arm.read()
        goal = np.array([p[0], p[1], slot[2] - offset[2] + float(approach_height)])
        kind, how = arm.move("robot.go_to_pose_cartesian", goal, r)
        q, _ = arm.read()
        if kind == "refused" or q[2] < goal[2] - float(reach_tolerance):
            return finish(
                "jammed",
                why + f"; lifting back stopped {max(0.0, goal[2] - q[2]) * 1e3:.1f} mm short"
                f" ({how or 'held'}): the piece is stuck, still held",
            )
        return finish(route, why + "; lifted back, still holding")

    def slide(travel: float, r: np.ndarray | None = None) -> dict[str, Any]:
        """Fine steps along seam_direction at the height the hand stands. A
        stop is the hand lagging its command along the slide lag_thr more than
        in free motion (the piece met the neighbours)."""
        p0, r_here = arm.still()
        r = r_here if r is None else r
        lags = [0.0]
        res: dict[str, Any] = {"met": False, "travel": 0.0, "at": p0}
        n = max(1, math.ceil(travel / step - 1e-9))
        for i in range(1, n + 1):
            goal = p0 + s_dir * min(i * step, travel)
            kind, why = arm.move("robot.go_to_pose", goal, r, fine=True)
            if kind == "refused":
                res.update(refused=why)
                return res
            p, _ = arm.still()
            lag = float(np.dot(goal - p, s_dir))
            res.update(at=p, travel=float(np.dot(p - p0, s_dir)))
            if lag - float(np.median(lags[-3:])) > lag_thr:
                res["met"] = True
                return res
            lags.append(lag)
        return res

    def push_in() -> Output:
        """approach="offset_push": set the piece down flat push_offset short of
        its slot, let go, close the jaws, push it along seam_direction into the
        slot with the fingertips 1 mm over the mat, press its centre, check."""
        back = slot[:2] - s_dir[:2] * float(push_offset)
        window_b = slot[2] + thick + float(contact_window)
        kind, why = arm.move(
            "robot.go_to_pose_cartesian", hand_for(back, state["yaw"], window_b), r_now
        )
        if kind == "refused":
            return lift_back("unreachable", f"the descent to the offset spot was refused: {why}")
        p_here, _ = place(back, window_b)
        res = feel(max(step, height(p_here) + OVERTRAVEL_FRACTION * thick))
        out["probes"] += 1
        if "refused" in res:
            return lift_back("unreachable", f"setting down was refused: {res['refused']}")
        h = height(res["at"])
        out["stop_height"] = out["best_height"] = h
        if not res["contact"]:
            return lift_back(
                "piece_lost",
                f"setting down went {-h * 1e3:.1f} mm under the mat: the piece is not in the hand",
            )
        if h > seated_tol:
            return lift_back(
                "not_seated",
                f"setting down {float(push_offset) * 1e3:.1f} mm short of the slot stopped {h * 1e3:.2f} mm "
                "over the mat: the spot is not clear",
            )
        p_set = unpress()
        piece = piece_at(p_set)
        if hold_s > 0:
            time.sleep(float(hold_s))
        ctx.tool("robot.open_gripper", **arm.on_arm)
        out["released"] = True
        hover_z = slot[2] + thick + float(approach_height) + float(finger_reach)
        _, r_here = arm.read()
        why = go("robot.go_to_pose_cartesian", np.array([p_set[0], p_set[1], hover_z]), r_here)
        if why:
            return finish("unreachable", f"set down and released; the retreat: {why}")
        ctx.tool("robot.close_gripper", **arm.on_arm)
        behind = piece[:2] - s_dir[:2] * (float(piece_radius) + PUSH_CLEAR)
        tip_tcp = slot[2] + PUSH_TIP_HEIGHT + float(finger_reach)
        for label, goal, tool in (
            (
                "over the piece's trailing side",
                np.array([behind[0], behind[1], hover_z]),
                "robot.go_to_pose",
            ),
            (
                "down behind the piece",
                np.array([behind[0], behind[1], tip_tcp]),
                "robot.go_to_pose_cartesian",
            ),
        ):
            why = go(tool, goal, r_here)
            if why:
                return finish("unreachable", f"set down and released; the move {label}: {why}")
        expected = PUSH_CLEAR + float(push_offset)
        res = slide(expected + PUSH_OVER, r_here)
        pushed = float(res["travel"])
        p_end = res["at"]
        _, r_here = arm.read()
        go("robot.go_to_pose_cartesian", np.array([p_end[0], p_end[1], hover_z]), r_here)
        how = (
            f"set down {float(push_offset) * 1e3:.1f} mm short, released, pushed "
            f"{max(0.0, pushed - PUSH_CLEAR) * 1e3:.1f} mm"
        )
        if "refused" in res:
            return finish("unreachable", how + f"; a push stroke was refused: {res['refused']}")
        if res["met"] and pushed < expected - PUSH_SLACK:
            return finish(
                "jammed",
                how
                + f": stopped {(expected - pushed) * 1e3:.1f} mm short of the slot, something is in the way",
            )
        # The press: closed fingertips down on the piece's centre (its top).
        why = go("robot.go_to_pose", np.array([slot[0], slot[1], hover_z]), r_here)
        if why:
            return finish("unreachable", how + f"; the move over the slot: {why}")
        top_window = slot[2] + thick + float(contact_window) + float(finger_reach)
        arm.move("robot.go_to_pose_cartesian", np.array([slot[0], slot[1], top_window]), r_here)
        p_now, _ = arm.still()
        res = feel(
            max(step, p_now[2] - float(finger_reach) - slot[2] + OVERTRAVEL_FRACTION * thick)
        )
        if "refused" in res:
            return finish("unreachable", how + f"; the press was refused: {res['refused']}")
        top = float(res["at"][2] - float(finger_reach) - slot[2])
        out["stop_height"] = top - thick
        p_up = unpress()
        go("robot.go_to_pose_cartesian", np.array([p_up[0], p_up[1], hover_z]), r_here)
        if not res["contact"]:
            return finish(
                "piece_lost",
                how + f"; the press went {-top * 1e3:.1f} mm under the mat: no piece at the slot",
            )
        how += f"; pressed its centre, its top stopped {top * 1e3:.2f} mm over the mat"
        out["seated_position"] = _vec(slot)
        if not verify:
            return finish("unverified", how + "; not checked (verify=false)")
        return _check(
            ctx, arm, out, finish, how, slot, slot, thick, top_radius, camera, check_reach
        )

    # -- the start: holding check, reach, hover, turn
    try:
        fraction = float(ctx.tool("robot.get_gripper", **arm.on_arm)["position"])
        grip = ctx.tool("robot.describe_gripper", **arm.on_arm)
        fit = grip.get("width_fit") or {}
        at_open = float(fit.get("at_open_m") or grip.get("span_m") or 0.0)
        at_closed = float(fit.get("at_closed_m") or 0.0)
        width = at_closed + (at_open - at_closed) * fraction
        if at_open > at_closed and width <= at_closed + 0.0005:
            return finish(
                "piece_lost",
                f"the jaws are at their stop ({width * 1e3:.1f} mm): no piece in the hand; nothing moved",
            )
    except Exception as exc:
        logger.info("seating-a-piece: gripper not read (%s)", exc)

    yaw0 = float(yaw_error_deg)
    if check_reach and not ik_ok(hand_for(slot, yaw0, slot[2]), rot(yaw0)):
        return finish("unreachable", "the seated hand pose is out of reach (IK preview)")
    hover_bottom = slot[2] + thick + float(approach_height)
    if height(p_start) < hover_bottom - slot[2] - 1.0e-3:
        why = go(
            "robot.go_to_pose_cartesian",
            np.array([p_start[0], p_start[1], hover_bottom - offset[2]]),
            r0,
        )
        if why:
            return finish("unreachable", f"rise: {why}")
    if abs(yaw0) > YAW_TOLERANCE_DEG:
        why = turn(yaw0)
        if why:
            return finish(
                "unreachable",
                f"could not turn the piece {yaw0:+.1f} deg to the slot ({why}); at the start",
            )
    _, r_now = arm.read()
    back = {"tilt_first": float(tilt_back), "offset_push": float(push_offset)}.get(approach, 0.0)
    why = go(
        "robot.go_to_pose", hand_for(slot[:2] - s_dir[:2] * back, state["yaw"], hover_bottom), r_now
    )
    if why:
        return finish("unreachable", f"hover: {why}")
    if approach == "offset_push":
        return push_in()

    def tilt_in() -> np.ndarray:
        """approach="tilt_first" (Watson et al.'s InsertTilt): tilt the piece
        leading side low, lower it tilt_back short of the slot until its edge
        touches, slide it along seam_direction until it meets the neighbours,
        level it there and feel down. Returns where the piece ended (the
        search's new centre)."""
        r_lvl = arm.read()[1]
        r_tilt = _about(np.cross(up, s_dir), float(tilt_deg)) @ r_lvl
        drop = float(piece_radius) * math.sin(math.radians(float(tilt_deg)))
        start = np.array([*(slot[:2] - s_dir[:2] * float(tilt_back)), hover_bottom])
        why = go("robot.go_to_pose", hand_at(start, r_tilt), r_tilt)
        if why:
            raise _Done(lift_back("unreachable", f"the tilt: {why}"))
        start[2] = slot[2] + drop + thick + float(contact_window)
        kind, why = arm.move("robot.go_to_pose_cartesian", hand_at(start, r_tilt), r_tilt)
        if kind == "refused":
            raise _Done(lift_back("unreachable", f"the tilted descent was refused: {why}"))
        p_here, r_here = arm.still()
        res = feel(
            max(step, piece_at(p_here, r_here)[2] - drop - slot[2] + OVERTRAVEL_FRACTION * thick),
            press=False,
        )
        if "refused" in res:
            raise _Done(lift_back("unreachable", f"a tilted step was refused: {res['refused']}"))
        edge = float(piece_at(res["at"], r_here)[2] - drop - slot[2])
        if not res["contact"]:
            raise _Done(
                lift_back(
                    "piece_lost",
                    f"the tilted descent went {-edge * 1e3:.1f} mm under the mat: the piece is not in the hand",
                )
            )
        met = slide(float(tilt_back) + float(search_radius), r_here)
        if "refused" in met:
            raise _Done(lift_back("unreachable", f"the tilted slide was refused: {met['refused']}"))
        p_here, r_here = arm.still()
        ref = piece_at(p_here, r_here)
        logger.info(
            "seating-a-piece: tilted edge touched %.2f mm over the mat, slid %.2f mm%s",
            edge * 1e3,
            met["travel"] * 1e3,
            " to the neighbours" if met["met"] else " without meeting them",
        )
        why = go("robot.go_to_pose", hand_at(ref, r_lvl), r_lvl)
        if why:
            raise _Done(lift_back("unreachable", f"levelling: {why}"))
        state["yaw"] = _yaw_between(r0, arm.read()[1])
        if abs(state["yaw"] - yaw0) > YAW_TOLERANCE_DEG:
            turn(yaw0)
        probe(ref[:2], "the levelled tilt")
        return ref[:2].copy()

    # -- the first descent: fast to the window over the neighbours, then felt
    if approach == "wiggle":
        window_bottom = slot[2] + thick + float(contact_window)
        goal = hand_for(slot, state["yaw"], window_bottom)
        kind, why = arm.move("robot.go_to_pose_cartesian", goal, r_now)
        p, _ = arm.still()
        if kind == "refused":
            return lift_back("unreachable", f"descent refused: {why}")
        if float(p[2] - goal[2]) > max(step, 0.002):
            out["stop_height"] = height(p)
            return lift_back(
                "not_seated",
                f"the descent stopped {height(p) * 1e3:.1f} mm over the mat, above the neighbours: the "
                "slot is higher than given or something is in the way",
            )
        place(slot, window_bottom)
        if abs(yaw0 - state["yaw"]) > 2.0 * YAW_TOLERANCE_DEG and not turn(yaw0):
            place(slot, window_bottom)
    overtravel = OVERTRAVEL_FRACTION * thick

    best: dict[str, Any] = {"h": math.inf}
    rest = {"h": -math.inf}

    def probe(target_xy: np.ndarray, label: str) -> float:
        """Feel down at the current x / y / yaw and record the stop: the piece's
        bottom over the mat. Ends the skill (_Done) on a refused step or a
        descent that meets nothing."""
        p_here, _ = arm.still()
        res = feel(max(step, height(p_here) + overtravel))
        out["probes"] += 1
        if "refused" in res:
            raise _Done(
                lift_back("unreachable", f"a step of {label} was refused: {res['refused']}")
            )
        at = res.get("first", res["at"])
        state["yaw"] = _yaw_between(r0, arm.read()[1])
        h = height(at)
        if not res["contact"]:
            raise _Done(
                lift_back(
                    "piece_lost",
                    f"{label} went {-h * 1e3:.1f} mm under the mat without the arm lagging: the "
                    "piece is not in the hand (or the slot is lower than given)",
                )
            )
        xy = piece_at(at)[:2] - slot[:2]
        logger.info(
            "seating-a-piece: %s at (%+.2f, %+.2f) mm %+.1f deg stopped %.2f mm over the mat (%s)",
            label,
            xy[0] * 1e3,
            xy[1] * 1e3,
            state["yaw"] - yaw0,
            h * 1e3,
            res["signal"],
        )
        if h < best["h"]:
            best.update(h=h, xy=xy, yaw=state["yaw"], target=np.asarray(target_xy, float)[:2])
        rest["h"] = max(rest["h"], h)
        out["stop_height"] = h
        return h

    def seat_found() -> bool:
        return best["h"] <= seated_tol

    def go_probe(target_xy: np.ndarray, yaw: float, label: str) -> float | None:
        """Lift clear of the neighbours, turn to ``yaw``, move over ``target_xy``
        and probe. None when the yaw could not be reached (skipped)."""
        lift_bottom = slot[2] + max(rest["h"], thick) + LIFT_CLEARANCE
        p_now, r_now = arm.read()
        kind, why = arm.move(
            "robot.go_to_pose",
            np.array([p_now[0], p_now[1], lift_bottom - offset[2]]),
            r_now,
            fine=True,
        )
        if kind == "refused":
            raise _Done(lift_back("unreachable", f"the lift before {label}: {why}"))
        if abs(yaw - state["yaw"]) > YAW_TOLERANCE_DEG:
            why = turn(yaw)
            if why:
                logger.info("seating-a-piece: yaw %+.1f skipped (%s)", yaw - yaw0, why)
                return None
        _, why = place(np.asarray(target_xy, float), lift_bottom)
        if not why and abs(yaw - state["yaw"]) > 2.0 * YAW_TOLERANCE_DEG and not turn(yaw):
            _, why = place(np.asarray(target_xy, float), lift_bottom)
        if why:
            raise _Done(lift_back("unreachable", f"the move to {label} was refused: {why}"))
        return probe(target_xy, label)

    def scoot_round(amp: float, yaw_amp: float, label: str) -> None:
        """puzzle_bot's scoot-and-wiggle around the best probe: out and back
        along x, then y, then a yaw out and back, each slid with the piece left
        resting where it stopped and felt again (two steps when it still rests)."""
        centre = np.asarray(best["target"], float)
        yaw_c = float(best["yaw"])
        pattern = [(-amp, 0.0, 0.0), (amp, 0.0, 0.0), (0.0, -amp, 0.0), (0.0, amp, 0.0)]
        if yaw_amp > 0.0:
            pattern += [(0.0, 0.0, yaw_amp), (0.0, 0.0, -yaw_amp)]
        for mx, my, myaw in pattern:
            if seat_found():
                return
            if myaw:
                why = turn(yaw_c + myaw)
                if why:
                    continue
            p_now, r_now = arm.read()
            goal = hand_for(centre + np.array([mx, my]), state["yaw"], 0.0)
            goal[2] = p_now[2]
            kind, why = arm.move("robot.go_to_pose", goal, r_now, fine=True)
            if kind == "refused":
                return
            probe(centre + np.array([mx, my]), label)
        if abs(state["yaw"] - yaw_c) > YAW_TOLERANCE_DEG and not seat_found():
            turn(yaw_c)

    stop_reason = "every candidate probed"
    scooted: list[np.ndarray] = []
    try:
        centre = slot[:2].copy()
        if approach == "tilt_first":
            centre = tilt_in()
        else:
            probe(slot[:2], "the first descent")
        # -- the probe search (puzzle_bot's pick-up-and-wiggle), most likely first
        order = search_order(
            float(search_radius),
            float(pitch),
            float(yaw_range_deg),
            float(yaw_step_deg),
            float(sigma_xy),
            float(sigma_yaw_deg),
        )[1:]
        k = 0

        def scoot_here() -> None:
            go_probe(best["target"], best["yaw"], "the scoot's start")
            scooted.append(np.asarray(best["target"], float).copy())
            for rnd in range(SCOOT_ROUNDS):
                if seat_found():
                    break
                scoot_round(
                    SCOOT + rnd * SCOOT_GROW,
                    SCOOT_YAW_DEG * (1.0 + 0.5 * rnd),
                    f"scoot round {rnd + 1}",
                )

        def fresh() -> bool:
            """The best probe has not been scooted around yet."""
            return not any(np.linalg.norm(best["target"] - t) < 1.0e-6 for t in scooted)

        while not seat_found():
            if scoot and best["h"] <= nearly_tol and fresh():
                stop_reason = "partly in"
                scoot_here()
                continue
            if k >= len(order):
                stop_reason = "every candidate probed"
                break
            if out["probes"] >= int(max_probes):
                stop_reason = f"max_probes ({int(max_probes)}) reached"
                break
            if max_time_s and time.monotonic() - t_start > float(max_time_s):
                stop_reason = f"max_time_s ({float(max_time_s):.0f} s) reached"
                break
            dx, dy, dyaw = order[k]
            k += 1
            go_probe(centre + np.array([dx, dy]), yaw0 + dyaw, f"probe {out['probes']}")

        # -- scoot and wiggle from the best probe (puzzle_bot's last phase)
        if not seat_found() and scoot and fresh():
            scoot_here()
        if not seat_found() and release_when_not_seated:
            go_probe(best["target"], best["yaw"], "the best probe")
    except _Done as done:
        return done.out

    if seat_found():
        stop_reason = "dropped" + (" in the scoot" if scooted else "")
    out["best_height"] = float(best["h"])
    out["resting_height"] = float(rest["h"])
    out["offset_found"] = _vec(np.array([*best["xy"], 0.0]))
    out["yaw_found_deg"] = float(best["yaw"] - yaw0)
    search = (
        f"{out['probes']} probes ({stop_reason}); the best stopped {best['h'] * 1e3:.2f} mm over "
        f"the mat at ({best['xy'][0] * 1e3:+.2f}, {best['xy'][1] * 1e3:+.2f}) mm, "
        f"{best['yaw'] - yaw0:+.1f} deg from the given slot; resting on the neighbours "
        f"read {rest['h'] * 1e3:.2f} mm"
    )

    if not seat_found():
        if not release_when_not_seated:
            route = "jammed" if best["h"] <= nearly_tol else "not_seated"
            note = (
                ": partly in, not finished by the scoot"
                if route == "jammed"
                else ": no drop within the search"
            )
            return lift_back(route, search + note)

    # -- release where it is and check the top
    p_seat = unpress()
    piece = piece_at(p_seat)
    out["seated_position"] = _vec(np.array([piece[0], piece[1], slot[2]]))
    out["stop_height"] = height(p_seat)
    if hold_s > 0:
        time.sleep(float(hold_s))
    ctx.tool("robot.open_gripper", **arm.on_arm)
    out["released"] = True
    _, r_now = arm.read()
    retreat = go(
        "robot.go_to_pose_cartesian",
        np.array([p_seat[0], p_seat[1], slot[2] - offset[2] + float(approach_height)]),
        r_now,
    )
    how = search + ("; dropped, released" if seat_found() else "; released at the best probe")
    if retreat:
        how += f"; the retreat: {retreat}"
    if not verify:
        return finish("unverified", how + "; not checked (verify=false)")
    return _check(ctx, arm, out, finish, how, piece, slot, thick, top_radius, camera, check_reach)


def _check(
    ctx: NodeContext,
    arm: _Arm,
    out: Output,
    finish: Any,
    how: str,
    piece: np.ndarray,
    slot: np.ndarray,
    thick: float,
    top_radius: float,
    camera: str | None,
    check_reach: bool,
) -> Output:
    """The piece's top read with the wrist depth camera from a view over it."""
    top_z = float(slot[2] + thick)
    pts = [
        np.array([piece[0] + dx, piece[1] + dy, top_z])
        for dx, dy, _ in search_order(float(top_radius), float(top_radius) / 2.0, 0, 0, 1, 1)
    ]
    target = np.array([piece[0], piece[1], top_z])
    try:
        obs = ctx.tool("robot.get_observation")
        tool_now = np.eye(4)
        ee = obs["arms"][arm.arm_id]["ee_pose"]
        tool_now[:3, :3] = _rot(ee["rotation"])
        tool_now[:3, 3] = _xyz(ee)
        cam = _wrist_camera(obs, tool_now, camera)
    except Exception as exc:
        return finish("unverified", how + f"; no observation for the flush check ({exc})")
    if cam is None:
        return finish("unverified", how + "; no wrist camera with a pose for the flush check")
    tool_cam = np.linalg.inv(tool_now) @ _matrix(cam["pose"])
    tries: list[str] = []
    for view in _views(tool_now, tool_cam, target):
        if view[2, 3] < top_z + MIN_VIEW_CLEARANCE:
            tries.append("a view under the clearance floor")
            continue
        if check_reach and not _ik_ok(ctx, view, arm.on_arm):
            tries.append("a view out of reach (IK preview)")
            continue
        kind, why = arm.move("robot.go_to_pose", view[:3, 3], view[:3, :3])
        q, _ = arm.read()
        if kind == "refused" or float(np.linalg.norm(q - view[:3, 3])) > REACH_TOLERANCE:
            tries.append(f"a view move: {why or 'ended off'}")
            continue
        arm.still()
        obs = ctx.tool("robot.get_observation")
        cam = _wrist_camera(obs, tool_now, camera)
        if cam is None:
            tries.append("the wrist frame went missing")
            continue
        heights = _read_heights(cam, pts)
        if len(heights) < MIN_SAMPLES:
            tries.append(f"{len(heights)} depth readings on the piece's top")
            continue
        dz = np.array(heights) - top_z
        h = float(np.median(dz))
        spread = float(np.percentile(dz, 90) - np.percentile(dz, 10))
        out["flush_error"], out["flush_spread"], out["samples"] = h, spread, int(dz.size)
        out["verified_by"] = "wrist_depth"
        read = (
            f"the top reads {h * 1e3:+.2f} mm from flush, spread {spread * 1e3:.2f} mm over "
            f"{dz.size} points"
        )
        if h < -0.5 * thick:
            return finish("piece_lost", f"{how}; {read}: no piece at the slot")
        if abs(h) <= FLUSH_FRACTION * thick and spread <= TILT_FRACTION * thick:
            return finish("seated", f"{how}; {read}")
        return finish("not_seated", f"{how}; {read}: resting on a neighbour or tilted")
    return finish("unverified", how + "; height not read: " + "; ".join(tries))


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
        logger.info("seating-a-piece: no IK preview (%s)", exc)
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
    """Hand poses putting ``target`` on the wrist camera's optical axis, looking
    straight down (the best depth), at each range in VIEW_RANGES."""
    cam_now = tool @ tool_cam
    down = _align(cam_now[:3, 2], np.array([0.0, 0.0, -1.0])) @ cam_now[:3, :3]
    views = []
    for rng in VIEW_RANGES:
        cam = np.eye(4)
        cam[:3, :3] = down
        cam[:3, 3] = target - down[:, 2] * rng
        views.append(cam @ np.linalg.inv(tool_cam))
    return views


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
