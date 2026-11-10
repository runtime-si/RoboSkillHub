# Written for RoboSkillHub (no upstream code). The geometry helpers (_frame, matrix, quat, axis_angle, turn_about,
# rotvec) and the chained-IK step with a joint-jump guard follow skills/standing-a-lying-domino-by-reorient-and-regrasp
# (Apache-2.0, this repository); the side approach follows skills/proposing-side-grasps (approach + close heading
# through robot.grasp_frame).
"""Pick a peg with the left arm across its shaft and hold it vertical, free end up, at a handover point."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PAD_WIDTH = 0.0155
"""Width of a linear_4310 fingertip across its closing axis [m] (the station's real tip, 15.5 x 3.9 mm). Held across
the shaft, the pad covers this much of the peg's length, centred on the tool centre."""

MIN_EXPOSED = 0.025
"""Least free length beyond the pads at the free end [m]: what the receiving top-down grasp needs (the strategy's
spec, candidates/PEG_SIDE_TOP_INSERT.md)."""

EXPOSED_MARGIN = 0.003
"""Added to MIN_EXPOSED when the grip point is chosen [m]: a servo's 4 mm arrival bound along the shaft, less the
1 mm the jaws centre the peg (a choice)."""

MIN_TOOL_HEIGHT = 0.014
"""Lowest tool-centre height over the table [m]: a real YAM station refuses lower by default."""

TIP_REACH = 0.0102
"""How far the closed fingertips reach past the tool centre along the approach [m] when robot.describe_gripper does
not say (linear_4310)."""

EQUATOR_BITE = 0.0005
"""A top-down grasp of a lying peg needs the fingertips at least this far under the shaft's equator [m]; above it
the closing jaws squeeze the peg out sideways (a choice)."""

TIP_TABLE_GAP = 0.002
"""Fingertips kept this far over the table on a top-down grasp [m]."""

STANDING_TOLERANCE_DEG = 20.0
"""A peg axis within this of vertical stands; within this of level it lies [deg]. Anything between is refused."""

LIFT_HEIGHT = 0.06
"""How far the hand rises straight up after the close [m] before it turns or travels."""

HOVER = 0.05
"""Height over the grasp of the top-down pre-grasp, and length of the side pre-grasp standoff [m]."""

HANDOVER_HEIGHT = 0.20
"""Default height of the handover point's tool centre over the table [m] (a choice: a 120 mm peg gripped at its middle
hangs 60 mm under it)."""

RIGHT_BITE = 0.012
"""How far under the free end the receiving arm's top-down tool centre is previewed [m]. Only a reach check: the
receiving skill picks its own depth within the exposed length."""

FACING_FAN_DEG = (
    0.0,
    20.0,
    -20.0,
    40.0,
    -40.0,
    60.0,
    -60.0,
    90.0,
    -90.0,
    120.0,
    -120.0,
    150.0,
    -150.0,
    180.0,
)
"""Offsets of the held approach's heading tried in the IK preview [deg], smallest turn first."""

IK_STEP = 0.02
"""Longest straight chunk of a chained IK move [m] (standing-a-lying-domino: 20 mm moved each joint 0.07-0.08 rad)."""

TURN_STEP_DEG = 10.0
"""Largest turn of the hand per chained IK step [deg]."""

JOINT_JUMP = 0.6
"""Largest change of any joint but the last between two chained steps [rad] (standing-a-lying-domino: larger jumps
flung the held tile)."""

JOINT_MARGIN = 0.05
"""Kept clear of each joint limit [rad]."""

JOINT_SETTLE = 0.075
"""A joint move ending farther than this from its target on any joint did not arrive [rad] (real YAM settle bound)."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from a move's target [m]: no tighter than a real YAM station's 12 mm."""

DESCENT_TOLERANCE = 0.006
"""A descent onto the grasp ending this far short stopped on something [m]: over a servo's 4 mm bound."""

ROLL_TOLERANCE_DEG = 4.0
"""Turn about the approach left uncorrected [deg]: a servo's 3 deg bound plus one."""

TILT_THRESHOLD_DEG = 8.0
"""Largest tilt of the held peg's axis from vertical for holding_vertical [deg]. A receiving top-down grasp centres
across the shaft; 8 deg on 25 mm of exposed length puts the end 3.5 mm off (a choice)."""

NARROW_MARGIN = 0.003
"""A jaw gap this far under the diameter holds nothing across it [m]."""

WIDE_MARGIN = 0.007
"""A jaw gap this far over the diameter closed on something else [m]; a YAM station twin reads a 25 mm part at
26.5-26.7 mm (tracking-gripper-opening-stall), so the band is lopsided."""

YAM_LIMITS = (
    (-2.61799, 0.0, 0.0, -1.5708, -1.5708, -2.0944),
    (3.05433, 3.65, 3.66519, 1.5708, 1.5708, 2.0944),
)
"""Joint limits [rad] when robot.describe_arm does not give them: the i2rt YAM's (joint 6 +/-2.09 rad)."""

Z = np.array([0.0, 0.0, 1.0])


class LeftGrip(TypedDict):
    T_base_tcp: dict[str, Any]
    T_TCP_object: dict[str, Any]
    exposed_len_m: float


class Output(TypedDict):
    route: str
    detail: str
    posture: str
    picked: bool
    left_grip: LeftGrip
    axis_tilt_deg: float
    tilt_source: str
    grip_from_free_end_m: float
    grip_width_m: float
    facing_deg: float
    free_end: dict[str, float]


# ----------------------------------------------------------------------------- geometry


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(p: Any) -> np.ndarray:
    p = p["position"] if isinstance(p, dict) and "position" in p else p
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
    k = np.asarray(axis, dtype=np.float64)
    k = k / (np.linalg.norm(k) or 1.0)
    a = math.radians(deg)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * kx + (1.0 - math.cos(a)) * (kx @ kx)


def rotvec(m: np.ndarray) -> np.ndarray:
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(m)) - 1.0) / 2.0)))
    if angle < 1.0e-9:
        return np.zeros(3)
    if angle > math.pi - 1.0e-6:
        w, v = np.linalg.eigh((m + np.eye(3)) / 2.0)
        return v[:, int(np.argmax(w))] * angle
    v = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    return v / (2.0 * math.sin(angle)) * angle


def turn_about(axis: np.ndarray, frm: np.ndarray, to: np.ndarray) -> float:
    """Signed angle [deg] turning ``frm`` onto ``to`` about ``axis`` (both projected on its normal plane)."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) or 1.0)
    f = frm - (frm @ a) * a
    t = to - (to @ a) * a
    return math.degrees(math.atan2(float(a @ np.cross(f, t)), float(f @ t)))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    c = float(a @ b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) or 1.0)


def se3(p: np.ndarray, r: np.ndarray) -> dict[str, Any]:
    return {"position": _vec(p), "rotation": quat(r)}


def choose_grip(length: float, min_exposed: float, pad: float) -> float:
    """Distance from the free end to the grip centre [m]: the middle (no gravity moment about the jaws) when that
    leaves ``min_exposed`` free, else as close to the middle as allows it; -1 when the pads would run off the far
    end."""
    s_min = min_exposed + EXPOSED_MARGIN + pad / 2.0
    s = max(length / 2.0, s_min)
    return s if s <= length - pad / 2.0 + 1.0e-9 else -1.0


def topdown_tool_height(
    axis_z: float, table_z: float, tip_reach: float, floor: float
) -> tuple[float, str]:
    """Tool-centre height for a top-down grasp across a lying shaft, or why there is none."""
    z = max(table_z + floor, table_z + tip_reach + TIP_TABLE_GAP)
    if z - tip_reach > axis_z - EQUATOR_BITE:
        return z, (
            f"at the {floor * 1e3:.0f} mm tool-centre floor the fingertips ({tip_reach * 1e3:.1f} mm under it) stand "
            f"{(z - tip_reach - table_z) * 1e3:.1f} mm over the table, over the shaft's equator at "
            f"{(axis_z - table_z) * 1e3:.1f} mm: the jaws would squeeze it out sideways"
        )
    return z, ""


def _refusal(reply: Any) -> str:
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


# ----------------------------------------------------------------------------- one arm


class _Hand:
    """One arm through robot.* tools: pose, joints, chained IK steps with the roll about the approach set from
    forward kinematics (a solver that leaves it free still lands on the commanded rotation), the jaws."""

    def __init__(self, ctx: Any, arm_id: int, limits: Any) -> None:
        self.ctx, self.id = ctx, int(arm_id)
        self.kw: dict[str, Any] = {"arm_id": self.id} if self.id else {}
        self.lower, self.upper = (np.asarray(v, dtype=np.float64) for v in limits)
        self.la = np.array([0.0, 0.0, 1.0])
        self.lc = np.array([1.0, 0.0, 0.0])
        self.at_open, self.at_closed = 0.0, 0.0

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        p = _pose(self.ctx.tool("robot.get_ee_pose", **self.kw))
        return _xyz(p), matrix(p["rotation"])

    def q(self) -> np.ndarray:
        try:
            arms = self.ctx.tool("robot.get_observation").get("arms") or []
        except Exception as exc:
            logger.info("picking-a-peg: no joint reading (%s)", exc)
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
            or self.lower[j] + JOINT_MARGIN <= value <= self.upper[j] - JOINT_MARGIN
        )

    def fk(self, q: np.ndarray) -> np.ndarray | None:
        try:
            reply = self.ctx.tool(
                "robot.forward_kinematics", joint_configs=[[float(v) for v in q]], **self.kw
            )
            return matrix(reply["poses"][0]["rotation"])
        except Exception as exc:
            logger.info("picking-a-peg: no forward kinematics (%s)", exc)
            return None

    def roll_error(self, r: np.ndarray, target: np.ndarray) -> float:
        """Turn about the approach [deg] that takes the jaws of ``r`` onto those of ``target``."""
        return turn_about(r @ self.la, r @ self.lc, target @ self.lc)

    def set_roll(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        """``q`` with its last joint turned so the hand's roll about its approach matches ``target``."""
        q = q.copy()
        for _ in range(3):
            r = self.fk(q)
            if r is None:
                return q
            err = self.roll_error(r, target)
            if abs(err) < 0.5:
                return q
            probe = q.copy()
            probe[-1] += 0.1
            r2 = self.fk(probe)
            rate = self.roll_error(r, r2) / math.degrees(0.1) if r2 is not None else 1.0
            rate = rate if abs(rate) > 0.3 else 1.0
            q[-1] += math.radians(err) / rate
        return q

    def ik(self, point: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, str]:
        try:
            reply = self.ctx.tool("robot.solve_ik", pose=se3(point, rot), **self.kw)
        except Exception as exc:
            return np.zeros(0), f"IK refused ({exc})"
        reply = reply if isinstance(reply, dict) else {}
        if reply.get("success") is False or reply.get("within_servo_tolerance") is False:
            return np.zeros(0), "no IK solution"
        q = (reply.get("joint_config") or {}).get("positions")
        q = np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)
        if q.size == 0:
            return q, "the IK reply has no joints"
        q = self.set_roll(q, rot)
        bad = [j for j in range(q.size) if not self.fits(j, float(q[j]))]
        if bad:
            return q, f"joint {bad[0] + 1} would be at {q[bad[0]]:+.2f} rad, at its limit"
        return q, ""

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

    def step(self, point: np.ndarray, rot: np.ndarray) -> str:
        q_now = self.q()
        q, why = self.ik(point, rot)
        if why:
            return why
        if q_now.size == q.size and q.size > 1:
            jump = float(np.max(np.abs(q[:-1] - q_now[:-1])))
            if jump > JOINT_JUMP:
                return f"a joint would jump {jump:.2f} rad (limit {JOINT_JUMP:.2f})"
        why = self.to_q(q)
        if why:
            return why
        q_after = self.q()
        if q_after.size == q.size and float(np.max(np.abs(q_after - q))) > JOINT_SETTLE:
            return f"the joints stopped {float(np.max(np.abs(q_after - q))):.2f} rad short"
        return ""

    def line(self, target: np.ndarray, rot: np.ndarray) -> str:
        """A straight move in chained IK steps, turning toward ``rot`` along the way."""
        p0, r0 = self.pose()
        rel = rotvec(np.asarray(rot) @ r0.T)
        turn = math.degrees(float(np.linalg.norm(rel)))
        n = max(
            1,
            math.ceil(float(np.linalg.norm(target - p0)) / IK_STEP),
            math.ceil(turn / TURN_STEP_DEG),
        )
        for i in range(1, n + 1):
            r_i = axis_angle(rel, turn * i / n) @ r0 if turn > 1.0e-6 else np.asarray(rot)
            why = self.step(p0 + (target - p0) * i / n, r_i)
            if why:
                return f"{why} (step {i} of {n})"
        p, _ = self.pose()
        if float(np.linalg.norm(p - target)) > REACH_TOLERANCE:
            return f"ended {float(np.linalg.norm(p - target)) * 1e3:.0f} mm from the target"
        return ""

    def go(self, point: np.ndarray, rot: np.ndarray) -> str:
        """A free move of the empty hand on the connector's own path, then the roll set on the last joint."""
        try:
            why = _refusal(self.ctx.tool("robot.go_to_pose", pose=se3(point, rot), **self.kw))
        except Exception as exc:
            why = str(exc) or type(exc).__name__
        p, _ = self.pose()
        if (why and not why.startswith("timeout")) or float(
            np.linalg.norm(p - point)
        ) > REACH_TOLERANCE:
            return f"{why or 'ended off'}; the hand is {float(np.linalg.norm(p - point)) * 1e3:.0f} mm from it"
        return self.line(point, rot)

    def width(self, reply: Any = None) -> float:
        try:
            if isinstance(reply, dict) and reply.get("width_m") is not None:
                return float(reply["width_m"])
            fraction = reply.get("position") if isinstance(reply, dict) else None
            if fraction is None:
                got = self.ctx.tool("robot.get_gripper", **self.kw)
                if isinstance(got, dict) and got.get("width_m") is not None:
                    return float(got["width_m"])
                fraction = got["position"]
            if self.at_open <= 0.0:
                return -1.0
            return self.at_closed + max(0.0, min(1.0, float(fraction))) * (
                self.at_open - self.at_closed
            )
        except Exception as exc:
            logger.info("picking-a-peg: could not read the gripper (%s)", exc)
            return -1.0


# ----------------------------------------------------------------------------- the skill


def run(
    ctx: NodeContext,
    peg: dict[str, Any],
    peg_pose: dict[str, Any],
    table_z: float,
    handover_point: dict[str, float] | None = None,
    arm_id: int = 0,
    other_arm_id: int = 1,
    object_name: str = "",
    min_exposed_m: float = MIN_EXPOSED,
    min_tool_height: float = MIN_TOOL_HEIGHT,
    tilt_threshold_deg: float = TILT_THRESHOLD_DEG,
    handover_height: float = HANDOVER_HEIGHT,
    lift_height: float = LIFT_HEIGHT,
    hover: float = HOVER,
    pad_width: float = PAD_WIDTH,
    joint_limits: list[list[float]] | None = None,
) -> Output:
    length, diameter = float(peg["length_m"]), float(peg["diameter_m"])
    table_z = float(table_z)
    r_obj = matrix(_pose(peg_pose).get("rotation"))
    c_obj = _xyz(_pose(peg_pose))
    a_local = unit(np.asarray(peg.get("axis_in_object") or [0.0, 0.0, 1.0], dtype=np.float64))
    tip_sign = -1.0 if str(peg.get("tip_end", "+z")).startswith("-") else 1.0
    f_local = (
        -tip_sign * a_local
    )  # the free end is the one opposite the tip: it goes up for the top-down regrasp
    f = r_obj @ f_local
    free_end = c_obj + f * length / 2.0
    kw: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}

    out: Output = {
        "route": "unreachable",
        "detail": "",
        "posture": "",
        "picked": False,
        "left_grip": {"T_base_tcp": {}, "T_TCP_object": {}, "exposed_len_m": -1.0},
        "axis_tilt_deg": -1.0,
        "tilt_source": "",
        "grip_from_free_end_m": -1.0,
        "grip_width_m": -1.0,
        "facing_deg": 0.0,
        "free_end": _vec(free_end),
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        logger.info("picking-a-peg: %s - %s", route, why)
        return out

    # --- posture
    up = angle_between(f, Z)
    if up <= STANDING_TOLERANCE_DEG:
        posture = "standing"
    elif up >= 180.0 - STANDING_TOLERANCE_DEG:
        out["posture"] = "standing_free_end_down"
        return finish(
            "upside_down",
            "the peg stands on its free end (tip up); one arm cannot turn it end over end for a top-down regrasp",
        )
    elif abs(up - 90.0) <= STANDING_TOLERANCE_DEG:
        posture = "lying"
    else:
        out["posture"] = "leaning"
        return finish(
            "ungraspable",
            f"the peg's axis is {up:.0f} deg from vertical: neither standing nor lying",
        )
    out["posture"] = posture

    # --- the hand and the arm
    limits = joint_limits
    if limits is None:
        try:
            jl = (ctx.tool("robot.describe_arm", **kw) or {}).get("joint_limits")
            limits = [jl["lower"], jl["upper"]] if jl else None
        except Exception as exc:
            logger.info("picking-a-peg: no arm description (%s)", exc)
    hand = _Hand(ctx, int(arm_id), limits or YAM_LIMITS)
    try:
        g = ctx.tool("robot.describe_gripper", **kw)
    except Exception as exc:
        logger.info("picking-a-peg: no gripper description (%s)", exc)
        g = {}
    g = g if isinstance(g, dict) else {}
    fit = g.get("width_fit") or {}
    hand.at_open = float(fit.get("at_open_m") or g.get("span_m") or 0.0)
    hand.at_closed = float(fit.get("at_closed_m") or 0.0)
    finger = g.get("finger") or {}
    tip_reach = (
        float(finger.get("reach_m") or 0.0) if finger.get("stated", True) else 0.0
    ) or TIP_REACH
    clearance = float(finger.get("clearance_m") or 0.004)
    if hand.at_open > 0.0 and diameter + 2.0 * clearance > hand.at_open:
        return finish(
            "ungraspable",
            f"a {diameter * 1e3:.1f} mm shaft does not fit the {hand.at_open * 1e3:.0f} mm open hand",
        )

    s = choose_grip(length, float(min_exposed_m), float(pad_width))
    if s < 0.0:
        return finish(
            "ungraspable",
            f"a {length * 1e3:.0f} mm peg cannot leave {float(min_exposed_m) * 1e3:.0f} mm free beyond "
            f"{float(pad_width) * 1e3:.1f} mm pads",
        )
    out["grip_from_free_end_m"] = float(s)
    grip_pt = free_end - f * s

    # --- the handover point and the receiver
    if handover_point:
        h = _xyz(handover_point)
    else:
        p0, _ = hand.pose()
        try:
            p1 = _xyz(_pose(ctx.tool("robot.get_ee_pose", arm_id=int(other_arm_id))))
        except Exception:
            p1 = p0
        h = np.array(
            [(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0, table_z + float(handover_height)]
        )
    other: _Hand | None = None
    if int(other_arm_id) >= 0 and int(other_arm_id) != int(arm_id):
        try:
            jl_o = (ctx.tool("robot.describe_arm", arm_id=int(other_arm_id)) or {}).get(
                "joint_limits"
            )
            other = _Hand(
                ctx, int(other_arm_id), [jl_o["lower"], jl_o["upper"]] if jl_o else YAM_LIMITS
            )
            p_self, _ = hand.pose()
            p_other, _ = other.pose()
            if float(np.linalg.norm(p_self - p_other)) < 1.0e-4:
                return finish(
                    "unreachable",
                    f"arm {other_arm_id} reads the same pose as arm {arm_id}: is it driven?",
                )
            gf_o = ctx.tool("robot.grasp_frame", arm_id=int(other_arm_id))
            r_o = matrix(gf_o["rotation"])
            other.la = r_o.T @ _xyz(gf_o.get("approach") or {"x": 0.0, "y": 0.0, "z": -1.0})
            other.lc = r_o.T @ _xyz(gf_o.get("close_axis") or {"x": 1.0, "y": 0.0, "z": 0.0})
        except Exception as exc:
            logger.info("picking-a-peg: receiver not previewed (%s)", exc)
            other = None

    def receiver_reaches(top: np.ndarray, u: np.ndarray) -> bool:
        if other is None:
            return True
        heading = math.degrees(math.atan2(u[1], u[0])) + 90.0
        try:
            gf_o = ctx.tool(
                "robot.grasp_frame", close_heading_deg=heading, arm_id=int(other_arm_id)
            )
        except Exception:
            return True
        for k in ("rotation", "rotation_flipped"):
            if gf_o.get(k):
                q, why = other.ik(top - Z * RIGHT_BITE, matrix(gf_o[k]))
                if not why:
                    return True
        return False

    # --- the grasp: top-down across a lying shaft, or level across a standing one
    def grasp_rotations(approach: np.ndarray | None, heading: float) -> list[np.ndarray]:
        args: dict[str, Any] = {"close_heading_deg": float(heading), **kw}
        if approach is not None:
            args["approach"] = _vec(approach)
        gf = ctx.tool("robot.grasp_frame", **args)
        rots = [
            matrix(gf[k])
            for k in ("rotation", "rotation_flipped")
            if isinstance(gf, dict) and gf.get(k)
        ]
        if rots:
            hand.la = rots[0].T @ _xyz(
                gf.get("approach") or _vec(approach if approach is not None else -Z)
            )
            hand.lc = rots[0].T @ _xyz(
                gf.get("close_axis")
                or {
                    "x": math.cos(math.radians(heading)),
                    "y": math.sin(math.radians(heading)),
                    "z": 0.0,
                }
            )
        return rots

    fh = unit(np.array([f[0], f[1], 0.0])) if posture == "lying" else np.zeros(3)
    try:
        p_start, _ = hand.pose()
    except Exception as exc:
        return finish("unreachable", f"the arm's pose could not be read ({exc})")

    plan: dict[str, Any] = {}
    if posture == "lying":
        tz, why = topdown_tool_height(float(grip_pt[2]), table_z, tip_reach, float(min_tool_height))
        if why:
            return finish("ungraspable", why)
        tcp_grasp = np.array([grip_pt[0], grip_pt[1], tz])
        try:
            rots = grasp_rotations(None, math.degrees(math.atan2(fh[1], fh[0])) + 90.0)
        except Exception as exc:
            return finish("unreachable", f"no grasp frame ({exc})")
        offset = tz - float(
            grip_pt[2]
        )  # tool centre over the shaft's axis; after the pitch, ahead of it
        for delta in FACING_FAN_DEG:
            u = axis_angle(Z, delta) @ fh
            k_axis = np.cross(-Z, u)
            top = h + u * offset + Z * s
            for r in rots:
                if hand.ik(tcp_grasp + Z * hover, r)[1] or hand.ik(tcp_grasp, r)[1]:
                    continue
                r_yawed = axis_angle(Z, delta) @ r
                r_pitched = axis_angle(k_axis, 90.0) @ r_yawed
                if hand.ik(h, r_yawed)[1] or hand.ik(h, r_pitched)[1]:
                    continue
                if not receiver_reaches(top, u):
                    continue
                plan = {
                    "u": u,
                    "delta": delta,
                    "r": r,
                    "r_yawed": r_yawed,
                    "r_held": r_pitched,
                    "tcp": tcp_grasp,
                }
                break
            if plan:
                break
    else:
        toward = grip_pt - p_start
        u0 = (
            unit(np.array([toward[0], toward[1], 0.0]))
            if float(np.hypot(toward[0], toward[1])) > 1e-6
            else np.array([1.0, 0.0, 0.0])
        )
        for delta in FACING_FAN_DEG:
            u = axis_angle(Z, delta) @ u0
            heading = math.degrees(math.atan2(u[1], u[0])) + 90.0
            try:
                rots = grasp_rotations(u, heading)
            except Exception as exc:
                return finish("unreachable", f"no grasp frame ({exc})")
            if float(grip_pt[2]) - table_z < float(min_tool_height):
                return finish("ungraspable", "the grip point is under the tool-centre floor")
            top = h + Z * s
            for r in rots:
                if hand.ik(grip_pt - u * hover, r)[1] or hand.ik(grip_pt, r)[1] or hand.ik(h, r)[1]:
                    continue
                if not receiver_reaches(top, u):
                    continue
                plan = {"u": u, "delta": delta, "r": r, "r_held": r, "tcp": grip_pt.copy()}
                break
            if plan:
                break
    if not plan:
        return finish(
            "unreachable",
            f"no {'top-down' if posture == 'lying' else 'level'} grasp and held pose at the handover point "
            f"({h[0]:.3f}, {h[1]:.3f}, {h[2]:.3f}) passed the IK preview for both arms; nothing moved",
        )
    u, r_grasp, tcp = plan["u"], plan["r"], plan["tcp"]
    out["facing_deg"] = float(math.degrees(math.atan2(u[1], u[0])))
    pre = tcp + Z * hover if posture == "lying" else tcp - u * hover

    # --- approach and close
    try:
        ctx.tool("robot.open_gripper", **kw)
    except Exception as exc:
        return finish("unreachable", f"the jaws did not open ({exc}); nothing moved")
    if posture == "standing":
        why = hand.go(pre + Z * hover, r_grasp) or hand.line(pre, r_grasp)
    else:
        why = hand.go(pre, r_grasp)
    if why:
        return finish("unreachable", f"the pre-grasp was refused: {why}; nothing picked")
    why = hand.line(tcp, r_grasp)
    p_at, r_at = hand.pose()
    short = float(np.linalg.norm(p_at - tcp))
    if short > DESCENT_TOLERANCE:
        hand.line(pre, r_grasp)
        return finish(
            "grasp_failed",
            f"the hand stopped {short * 1e3:.1f} mm short of the grasp ({why or 'stalled'}): a fingertip on the peg, "
            "or the peg is not where it was seen; backed out, nothing picked",
        )
    try:
        w = hand.width(ctx.tool("robot.close_gripper", **kw))
    except Exception as exc:
        return finish("grasp_failed", f"the jaws did not close ({exc})")
    out["grip_width_m"] = float(w)
    if w >= 0.0 and not (diameter - NARROW_MARGIN <= w <= diameter + WIDE_MARGIN):
        ctx.tool("robot.open_gripper", **kw)
        hand.line(pre, r_grasp)
        what = (
            "closed on nothing or not across the shaft"
            if w < diameter
            else "closed on something wider"
        )
        return finish(
            "grasp_failed",
            f"the jaws read {w * 1e3:.1f} mm on a {diameter * 1e3:.1f} mm shaft: {what}; opened and backed out",
        )

    # The peg in the hand: as it was seen, centred across the closing axis by the jaws.
    t_tcp_obj = r_at.T @ (c_obj - p_at)
    r_tcp_obj = r_at.T @ r_obj
    lc = unit(hand.lc)
    t_tcp_obj = t_tcp_obj - float(t_tcp_obj @ lc) * lc
    out["picked"] = True

    def held_ok(where: str) -> str:
        w_now = hand.width()
        out["grip_width_m"] = float(w_now)
        if 0.0 <= w_now < diameter - NARROW_MARGIN:
            return f"the jaws read {w_now * 1e3:.1f} mm {where}: the peg is gone"
        return ""

    # --- lift, turn, carry
    p_lift = p_at + Z * float(lift_height)
    why = hand.line(p_lift, r_at)
    if why:
        return finish(
            "unreachable",
            f"the lift was refused: {why}; the left hand still holds the peg at the grasp",
        )
    gone = held_ok("after the lift")
    if gone:
        return finish("dropped", gone)
    if posture == "lying":
        why = hand.line(p_lift, plan["r_yawed"]) if abs(plan["delta"]) > 1.0e-6 else ""
        if not why:
            why = hand.line(h, plan["r_yawed"])
        if not why:
            why = hand.line(h, plan["r_held"])
        if why:
            return finish(
                "unreachable",
                f"the turn upright was refused: {why}; the left hand still holds the peg",
            )
    else:
        why = hand.line(h, plan["r_held"])
        if why:
            return finish(
                "unreachable", f"the carry to the handover point was refused: {why}; still holding"
            )
    gone = held_ok("at the handover point")
    if gone:
        return finish("dropped", gone)

    # --- the axis read back and finished on the last joint (a solver that leaves the roll free)
    p_h, r_h = hand.pose()
    f_tcp = r_tcp_obj @ f_local
    lean = turn_about(r_h @ hand.la, r_h @ f_tcp, Z)
    if abs(lean) > ROLL_TOLERANCE_DEG:
        q = hand.q()
        if q.size:
            goal = hand.set_roll(q, axis_angle(r_h @ hand.la, lean) @ r_h)
            if not hand.fits(q.size - 1, float(goal[-1])):
                return finish(
                    "unreachable",
                    f"the last joint has no room to stand the peg up ({lean:+.0f} deg); still holding",
                )
            why = hand.to_q(goal)
            if why:
                return finish(
                    "unreachable", f"the last-joint correction was refused: {why}; still holding"
                )
            p_h, r_h = hand.pose()

    # --- the result: from the hand, or re-perceived when an object name is given
    source = "hand"
    if object_name:
        try:
            obb = ctx.tool("sim.get_object_obb", object_name=str(object_name))
            obb = obb.get("obb", obb)
            c_seen = _xyz(obb["center"])
            r_seen = matrix(obb["orientation"])
            t_tcp_obj = r_h.T @ (c_seen - p_h)
            r_tcp_obj = r_h.T @ r_seen
            source = "perceived"
        except Exception as exc:
            logger.info("picking-a-peg: the peg was not re-perceived (%s)", exc)
    f_world = r_h @ r_tcp_obj @ f_local
    c_world = p_h + r_h @ t_tcp_obj
    tilt = angle_between(f_world, Z)
    end_world = c_world + f_world * length / 2.0
    exposed = float((end_world - p_h) @ f_world) - float(pad_width) / 2.0
    out["axis_tilt_deg"], out["tilt_source"] = float(tilt), source
    out["free_end"] = _vec(end_world)
    out["left_grip"] = {
        "T_base_tcp": se3(p_h, r_h),
        "T_TCP_object": se3(t_tcp_obj, r_tcp_obj),
        "exposed_len_m": float(exposed),
    }
    gone = held_ok("at the end")
    if gone:
        return finish("dropped", gone)
    off_axis = float(np.linalg.norm(np.cross(p_h - c_world, f_world)))
    if source == "perceived" and off_axis > diameter / 2.0 + tip_reach + 0.01:
        return finish(
            "dropped",
            f"the peg's axis is {off_axis * 1e3:.0f} mm from the tool centre: not in the hand",
        )
    if tilt > float(tilt_threshold_deg):
        return finish(
            "slipped",
            f"the peg's axis is {tilt:.1f} deg from vertical ({source}), over {float(tilt_threshold_deg):.0f} deg: "
            "it turned in the jaws; still holding",
        )
    if exposed < float(min_exposed_m) - 0.004:
        return finish(
            "slipped",
            f"only {exposed * 1e3:.0f} mm of the free end is exposed ({source}); still holding",
        )
    return finish(
        "holding_vertical",
        f"{posture} peg held {tilt:.1f} deg from vertical ({source}), free end up with {exposed * 1e3:.0f} mm exposed, "
        f"jaws {out['grip_width_m'] * 1e3:.1f} mm",
    )
