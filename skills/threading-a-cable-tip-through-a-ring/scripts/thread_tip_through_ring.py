# Adapted from RoboTwin (https://github.com/RoboTwin-Platform/RoboTwin),
# envs/hanging_mug.py (play_once: place_actor(..., functional_point_id=0,
# constrain="align", pre_dis=0.05, dis=-0.05, pre_dis_axis="fp")) and
# envs/_base_task.py (get_place_pose / place_actor: the held object's
# functional point carried onto the target through the fixed hand-to-object
# transform, a pre-pose pre_dis back along the target's axis, then a second
# pose dis along it, negative = past the target), MIT, Copyright (c) 2025
# Tianxing Chen; the licence is LICENSE.robotwin beside this bundle.
# Changes: the held "object" is a cable's free end, its functional point the
# tip, given in the world (measured by a camera or registering-held-objects)
# or from how far behind the tip it was taken, so the transform is the tip's
# offset in the hand; the target pose is a ring's centre and axis (perceived)
# instead of the simulator's rack functional point; "align" turns the end
# about the vertical only (the hand stays top-down as the grasp left it), the
# turn read back and finished on the last joint where the arm's solver leaves
# the wrist roll free; the end's sag is handled by aiming the axis between the
# tip and where the cable leaves the fingers rather than at the tip; the
# overshoot is capped where the fingertips would reach the ring; the pass is
# judged by the hand's travel past the ring's plane (and optionally a camera
# point on the tip) with routed exits instead of a scripted success check.
# The insert-then-pull strategy for a deformable linear object through a ring
# follows DLOInsert / DLOPull (lmeee.github.io/DLOPull; no licence, not
# copied).
"""Thread a held cable's tip through a ring: align it on the ring's axis in front of the mouth, push through with an overshoot, check it passed."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PRE_DIS = 0.05
"""How far in front of the ring's plane the tip is put on its axis before the
push [m]: RoboTwin's pre_dis (0.05); a real YAM threading station put the tip
4.5 cm short of its ring (ring1_real.py i_plan)."""

OVERSHOOT = 0.05
"""How far past the ring's plane the push aims the tip [m]: RoboTwin's dis
(-0.05, 5 cm through). Capped by FINGER_ROOM where the fingertips would reach
the ring."""

MIN_PAST = 0.010
"""The tip at least this far past the ring's plane counts as through [m]: the
real threading station's confirm (s_tip > 0.01)."""

FINGER_HALF = 0.010
"""Half the fingertip pads' length along the held cable [m]: where the cable
leaves the fingers ahead of the tool centre (real station's FINGER_HALF)."""

PINCH_BELOW = 0.004
"""How far under the cable's centre the fingertips were when it was taken [m]
(grasping-a-cable-lying-flat): the held cable runs that far over the
fingertips."""

TIP_REACH = 0.0102
"""Closed fingertips past the tool centre along the approach [m] when
robot.describe_gripper states none: the YAM linear_4310's measured value."""

FINGER_CLEAR = 0.005
"""How far in front of the ring's wire the fingertips' front stops [m]. The real
threading station let them 5 mm into the plane (RING_CLEAR -0.005, the tapered
tips inside a 44 mm ring with the tip on the axis); in a YAM twin the hand
standing 19 mm over a 38 mm ring's centre to aim a sagging end was stopped by
the ring's top 4.6 mm in front of the plane (fingers and the gripper body sit
over the cable), so the push stops short of the wire. Negative lets them in."""

HOLE_MARGIN = 0.004
"""The held cable stays this far inside the ring's opening besides its own
radius [m] (real station's HOLE_MARGIN)."""

CABLE_RADIUS = 0.00175
"""The cable's radius when none is given [m]: a 3.5 mm charging cable."""

WIRE_RADIUS = 0.002
"""The ring wire's radius when none is given [m]: the threading station's
rings."""

CARRY_OVER = 0.06
"""A hand that starts past the pre-insert pose along the axis comes down onto it
from this far over it [m] (real station: over = mouth + 6 cm). One in front of
it goes level with the mouth and straight in, as the real station's way "in
front of the ring, at its mouth's height": in a YAM twin a 3.5 mm cable taken
4.5 cm behind its tip slid out of the pads when the hand rose 10 cm to come
over the mouth, with all 26 cm of it hanging."""

YAW_TOLERANCE_DEG = 5.0
"""How far the held end may point off the ring's axis in plan before the rest
of the turn is made on the last joint [deg]; ring1_real's TILT_TOL is 6 deg."""

TURN_TRIES = 3
"""Last-joint turns tried (the first one learns which way the joint turns the hand)."""

ROLL_LIMITS = (-2.0944, 2.0944)
"""Last joint's range [rad] when none is stated: the i2rt YAM's joint 6."""

LIMIT_MARGIN = 0.05

REACH_TOLERANCE = 0.015
"""How far the hand may end from a carrying move [m]: no tighter than a real
YAM station's own 12 mm per-move bound."""

PUSH_TOLERANCE = 0.006
"""A push that leaves the hand further than this short of its end along the
axis was held back [m]: past a servo's 4 mm arrival bound."""

ALIGN_LOOKS = 2
"""Camera looks at the tip in front of the mouth (each re-registers the tip
and re-aims the hand) before the push."""

ALIGN_OK = 0.008
"""The seen tip may stand this far from where it was aimed before another
look re-aims it [m] (real station's ALIGN_OK)."""

DEPTH_WINDOW = 6
"""Half-size of the pixel window round a pointed tip whose depth is read [px]."""

FOREGROUND = 0.02
"""Points nearer the camera than the window's median by more than this are the
cable, not what lies behind it [m] (real station's END_FG_GAP is 0.04 on a
view across the table; nearer here). Fewer than three such points: the tip is
not placed. On a real watching-wrist frame at a ring's mouth the pointed tip's
window held only the table behind it once in four, which read as the tip
13 cm short of the ring."""

EMPTY_MARGIN = 0.001
"""Jaws within this of their closed stop hold nothing [m] (as
grasping-a-cable-lying-flat): in a YAM twin a 3.5 mm cable that slid out of the
pads during a wrist turn or the push left the hand reporting its moves arrived."""

MAX_TILT_DEG = 35.0
"""The hand must be within this of straight down [deg]: the skill turns the end
about the vertical only (the real station leaned at most 35 deg)."""

REFUSALS = ("unreachable", "bad_speed_scale")
"""Routed outcomes that mean a move was refused, not stopped on the way."""


class Output(TypedDict):
    route: str
    detail: str
    tip_past: float
    hand_travel: float
    push_planned: float
    tip_offset: float
    exit_offset: float
    crossing_offset: float
    sag_deg: float
    turned_deg: float
    tip_seen: bool
    seen_tip_past: float
    tip_position: dict[str, float]
    cable_lost: bool


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _arr(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v.get("z", 0.0))])
    return np.asarray(v, dtype=float).reshape(-1)[:3]


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _matrix(q: dict[str, float]) -> np.ndarray:
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


def _quat(m: np.ndarray) -> dict[str, float]:
    """A rotation matrix as a unit quaternion (Shepperd's method)."""
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
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return {"w": w / n, "x": x / n, "y": y / n, "z": z / n}


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def _heading(v: np.ndarray) -> float:
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


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


def aim(
    tip: np.ndarray, exit_: np.ndarray, centre: np.ndarray, axis: np.ndarray, tip_s: float
) -> tuple[np.ndarray, float, float, float]:
    """Where to put the tip so the held stretch passes the ring's opening.

    The push is a translation along the axis, so the tip's and the exit's
    offsets from the axis stay as aimed. With the tip ``tip_s`` past the plane
    at the end of the push, the stretch from the exit to the tip crosses the
    plane a fraction f of the way from the tip; the point w = f / 2 along it is
    put on the axis, which leaves the tip and that crossing equally far off it.
    Returns the tip's aimed position on the plane through ``centre`` (its
    offset only; the caller sets how far along), and the tip's, the exit's and
    the crossing's offsets [m]."""
    d = exit_ - tip
    along = float(d @ axis)
    perp = d - along * axis
    exit_s = tip_s + along
    f = 1.0 if exit_s >= 0.0 else (tip_s / (tip_s - exit_s) if tip_s > 0.0 else 0.0)
    w = 0.5 * min(max(f, 0.0), 1.0)
    shift = -w * perp
    tip_off = float(np.linalg.norm(shift))
    exit_off = float(np.linalg.norm(shift + perp))
    cross_off = float(np.linalg.norm(shift + min(max(f, 0.0), 1.0) * perp))
    return centre + shift, tip_off, exit_off, cross_off


def run(
    ctx: NodeContext,
    ring_centre: dict[str, float],
    ring_axis: dict[str, float],
    ring_inner_radius: float,
    tip_position: dict[str, float] | None = None,
    exit_position: dict[str, float] | None = None,
    tip_in_tool: dict[str, float] | None = None,
    grasp_back: float | None = None,
    tip_direction: dict[str, float] | None = None,
    sag_deg: float = 0.0,
    ring_wire_radius: float = WIRE_RADIUS,
    cable_radius: float = CABLE_RADIUS,
    keep_axis_sign: bool = False,
    pre_dis: float = PRE_DIS,
    overshoot: float = OVERSHOOT,
    min_past: float = MIN_PAST,
    finger_clear: float = FINGER_CLEAR,
    camera: str = "",
    cable_description: str = "cable",
    release: bool = False,
    check_reach: bool = True,
    reach_tolerance: float = REACH_TOLERANCE,
    roll_limits: list[float] | None = None,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "tip_past": 0.0,
        "hand_travel": 0.0,
        "push_planned": 0.0,
        "tip_offset": -1.0,
        "exit_offset": -1.0,
        "crossing_offset": -1.0,
        "sag_deg": 0.0,
        "turned_deg": 0.0,
        "tip_seen": False,
        "seen_tip_past": 0.0,
        "tip_position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "cable_lost": False,
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        return out

    def move(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        try:
            reply = ctx.tool(tool, pose={"position": _vec(position), "rotation": rot}, **on_arm)
            return _refusal(reply)
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

    def reachable(position: np.ndarray, rot: dict[str, float]) -> str:
        if not check_reach:
            return ""
        try:
            reply = ctx.tool(
                "robot.solve_ik", pose={"position": _vec(position), "rotation": rot}, **on_arm
            )
        except Exception as exc:
            logger.info("threading-a-cable-tip-through-a-ring: no IK preview (%s)", exc)
            return ""
        if isinstance(reply, dict) and (
            reply.get("success") is False or reply.get("within_servo_tolerance") is False
        ):
            return str(reply.get("reason") or "IK preview failed")
        return ""

    # The ring.
    c = _arr(ring_centre)
    a = _arr(ring_axis)
    if float(np.linalg.norm(a)) < 1.0e-9:
        raise ValueError("ring_axis is zero")
    a = a / float(np.linalg.norm(a))
    start = here()
    p0, R0 = _xyz(start), _matrix(start["rotation"])
    if not keep_axis_sign and float((p0 - c) @ a) > 0.0:
        a = -a  # in from the hand's side: perception gives a plane normal without a sign
    radius = float(cable_radius) if cable_radius else CABLE_RADIUS
    hole = float(ring_inner_radius) - radius - HOLE_MARGIN
    if hole <= 0.0:
        return finish(
            "misaligned",
            f"the ring's opening ({float(ring_inner_radius) * 1e3:.1f} mm inner radius) leaves no room "
            f"for a {2 * radius * 1e3:.1f} mm cable with {HOLE_MARGIN * 1e3:.0f} mm to spare",
        )
    tilt = math.degrees(math.acos(float(np.clip(-R0[2, 2], -1.0, 1.0))))
    if min(tilt, 180.0 - tilt) > MAX_TILT_DEG:
        return finish(
            "unreachable",
            f"the hand stands {min(tilt, 180.0 - tilt):.0f} deg off straight down; this skill turns "
            f"the held end about the vertical only",
        )

    # The held end in the hand: the tip, and where the cable leaves the fingers.
    try:
        hand = ctx.tool("robot.describe_gripper", **on_arm)
    except Exception as exc:
        logger.info("threading-a-cable-tip-through-a-ring: no gripper description (%s)", exc)
        hand = {}
    reach = float((hand.get("finger") or {}).get("reach_m") or TIP_REACH)
    fit = hand.get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or hand.get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)
    below = reach - PINCH_BELOW  # the held cable's centre under the tool centre

    def empty() -> bool:
        """Whether the jaws read at their closed stop: nothing between them. An
        unreadable hand is not an empty one."""
        if at_open <= 0.0:
            return False
        try:
            fraction = float(ctx.tool("robot.get_gripper", **on_arm)["position"])
        except Exception as exc:
            logger.info(
                "threading-a-cable-tip-through-a-ring: could not read the gripper (%s)", exc
            )
            return False
        width = at_closed + max(0.0, min(1.0, fraction)) * (at_open - at_closed)
        return width <= at_closed + EMPTY_MARGIN

    if empty():
        return finish(
            "unreachable",
            "the jaws are at their closed stop: there is no cable in the hand to thread",
        )
    if tip_position is not None:
        tip_w = _arr(tip_position)
        how = "measured"
    elif tip_in_tool is not None:
        tip_w = p0 + R0 @ _arr(tip_in_tool)
        how = "given in the hand"
    elif grasp_back is not None and tip_direction is not None:
        u = _arr(tip_direction)
        u[2] = 0.0
        if float(np.linalg.norm(u)) < 1.0e-9:
            raise ValueError("tip_direction has no horizontal part")
        u = u / float(np.linalg.norm(u))
        s = math.radians(float(sag_deg))
        free = max(float(grasp_back) - FINGER_HALF, 0.0)
        root = p0 + FINGER_HALF * u - np.array([0.0, 0.0, below])
        tip_w = root + free * (math.cos(s) * u - math.sin(s) * np.array([0.0, 0.0, 1.0]))
        how = f"taken {float(grasp_back) * 1e3:.0f} mm behind it, sagging {float(sag_deg):.0f} deg"
    else:
        raise ValueError("give tip_position, tip_in_tool, or grasp_back with tip_direction")
    u_out = tip_w - p0
    u_out[2] = 0.0
    if float(np.linalg.norm(u_out)) < 1.0e-6:
        return finish(
            "misaligned",
            "the tip stands straight under the hand: no way out of the fingers to align",
        )
    u_out /= float(np.linalg.norm(u_out))
    if exit_position is not None:
        exit_w = _arr(exit_position)
    else:
        exit_w = p0 + FINGER_HALF * u_out - np.array([0.0, 0.0, below])
    tip_t, exit_t = R0.T @ (tip_w - p0), R0.T @ (exit_w - p0)
    chord = tip_w - exit_w
    out["sag_deg"] = float(math.degrees(math.atan2(-chord[2], float(np.linalg.norm(chord[:2])))))

    # Align: the end turned about the vertical to point along the axis in plan.
    a_plan = a.copy()
    a_plan[2] = 0.0
    if float(np.linalg.norm(a_plan)) < 1.0e-6:
        return finish(
            "unreachable", "the ring's axis is vertical: this skill threads a standing ring"
        )

    def pointing(R: np.ndarray) -> float:
        v = R @ (tip_t - exit_t)
        return _wrap(_heading(a_plan) - _heading(v))

    turn = pointing(R0)
    R_goal = _rot_z(math.radians(turn)) @ R0
    rot_goal = _quat(R_goal)

    def pre_pose(R: np.ndarray, tip_end_s: float) -> tuple[np.ndarray, float, float, float]:
        """The tool centre putting the tip pre_dis in front of the plane, aimed
        for a push that ends with it ``tip_end_s`` past."""
        tip_now, exit_now = R @ tip_t, R @ exit_t
        on_plane, t_off, e_off, x_off = aim(tip_now, exit_now, c, a, tip_end_s)
        return on_plane - float(pre_dis) * a - tip_now, t_off, e_off, x_off

    def push_room(tcp_pre: np.ndarray) -> float:
        """How far along the axis the hand may go from the pre-insert pose before
        its fingertips' front comes finger_clear in front of the ring's wire.
        Over the ring's top too: carried past the plane there, the cable behind
        the fingers would cross the plane outside the ring."""
        front = float((tcp_pre - c) @ a) + FINGER_HALF
        return -(float(ring_wire_radius) + float(finger_clear)) - front

    # The plan with the turned hand, before moving: does the end fit the ring?
    def geometry(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
        tip_end_s = float(overshoot)
        for _ in range(4):  # the overshoot capped by the fingers, the aim set for that depth
            tcp_pre, t_off, e_off, x_off = pre_pose(R, tip_end_s)
            capped = min(float(overshoot), -float(pre_dis) + push_room(tcp_pre))
            if abs(capped - tip_end_s) < 1.0e-4:
                break
            tip_end_s = capped
        push = tip_end_s + float(pre_dis)
        return tcp_pre, tcp_pre + push * a, push, t_off, e_off, x_off

    def fits(R: np.ndarray) -> str:
        tcp_pre, tcp_end, push, t_off, e_off, x_off = geometry(R)
        out["push_planned"] = float(push)
        out["tip_offset"], out["exit_offset"], out["crossing_offset"] = t_off, e_off, x_off
        tip_end_s = push - float(pre_dis)
        if tip_end_s < float(min_past):
            return (
                f"the tip can go only {tip_end_s * 1e3:.0f} mm past the ring's plane before the fingertips "
                f"reach the ring: take the cable further behind its tip"
            )
        if t_off > hole or x_off > hole:
            return (
                f"the held end ({how}) sags {out['sag_deg']:.0f} deg: aimed between its tip and the fingers, "
                f"the tip stands {t_off * 1e3:.0f} mm and the crossing {x_off * 1e3:.0f} mm off the axis, "
                f"past the {hole * 1e3:.0f} mm the opening leaves"
            )
        return ""

    why = fits(R_goal)
    if why:
        return finish("misaligned", why)
    tcp_pre, tcp_end, _, _, _, _ = geometry(R_goal)
    for label, pos in (("the pre-insert pose", tcp_pre), ("the end of the push", tcp_end)):
        why = reachable(pos, rot_goal)
        if why:
            return finish("unreachable", f"no wrist solution reaches {label} ({why})")

    # Turn where the hand stands, read back, the rest on the last joint.
    why = go("robot.go_to_pose", p0, rot_goal)
    if why and not why.startswith("ended off"):
        return finish("unreachable", f"turning the end toward the ring's axis: {why}")
    now = here()
    gain = -1.0 if _matrix(now["rotation"])[2, 2] < 0 else 1.0
    lo, hi = (float(v) for v in (roll_limits or ROLL_LIMITS))
    notes: list[str] = []
    for _ in range(TURN_TRIES):
        residual = pointing(_matrix(now["rotation"]))
        if abs(residual) <= YAW_TOLERANCE_DEG:
            break
        try:
            arms = ctx.tool("robot.get_observation").get("arms") or []
            q = arms[int(arm_id)].get("joint_state")
            q = np.asarray(q.get("positions") if isinstance(q, dict) else q, dtype=float).reshape(
                -1
            )
        except Exception as exc:
            notes.append(f"no joint reading to turn the wrist ({exc})")
            break
        room = (lo + LIMIT_MARGIN - q[-1], hi - LIMIT_MARGIN - q[-1])
        command = math.radians(residual) / gain
        if not room[0] <= command <= room[1]:
            notes.append(f"the last joint has no room to turn {residual:+.0f} deg")
            break
        target_q = q.copy()
        target_q[-1] += command
        try:
            why = _refusal(
                ctx.tool(
                    "robot.move_to_joints", joint_config={"positions": target_q.tolist()}, **on_arm
                )
            )
        except Exception as exc:
            why = str(exc) or type(exc).__name__
        if why:
            notes.append(f"wrist turn: {why}")
            break
        before = residual
        now = here()
        moved = math.radians(_wrap(before - pointing(_matrix(now["rotation"]))))
        if abs(command) > math.radians(2.0) and 0.5 < abs(moved / command) < 1.5:
            gain = moved / command
    R = _matrix(now["rotation"])
    rot = now["rotation"]
    left = pointing(R)
    out["turned_deg"] = float(_wrap(turn - left))
    if abs(left) > 2.0 * YAW_TOLERANCE_DEG:
        notes.append(f"the end points {left:+.0f} deg off the ring's axis in plan")

    # Re-planned with the rotation the hand has.
    why = fits(R)
    if why:
        return finish("misaligned", why + ("; " + "; ".join(notes) if notes else ""))
    tcp_pre, tcp_end, push, _, _, _ = geometry(R)

    # Carry: from in front of the ring, level with the mouth and straight to it;
    # from elsewhere, up, over the pre-insert pose and down onto it.
    p = _xyz(here())
    if float((p - c) @ a) <= float((tcp_pre - c) @ a) + 0.01:
        legs = (
            ("to the mouth's height", np.array([p[0], p[1], tcp_pre[2]])),
            ("to the ring's mouth, on its axis", tcp_pre),
        )
    else:
        over = tcp_pre + np.array([0.0, 0.0, CARRY_OVER])
        high = max(float(p[2]), float(over[2]))
        legs = (
            ("up", np.array([p[0], p[1], high])),
            ("over the ring's mouth", np.array([over[0], over[1], high])),
            ("down to the ring's mouth, on its axis", tcp_pre),
        )
    for label, pos in legs:
        why = go("robot.go_to_pose_cartesian", pos, rot)
        if why:
            return finish("unreachable", f"carrying the end {label}: {why}")

    # Optional camera looks: the tip re-registered and the hand re-aimed.
    def seen_tip() -> np.ndarray | None:
        if not camera:
            return None
        try:
            obs = ctx.tool("robot.get_observation")
        except Exception as exc:
            logger.info("threading-a-cable-tip-through-a-ring: no observation (%s)", exc)
            return None
        cams = obs.get("cameras") or []
        if isinstance(cams, dict):
            cams = list(cams.values())
        cams = [k for k in cams if k.get("depth") is not None and camera in str(k.get("name", ""))]
        for cam in cams:
            try:
                reply = ctx.tool(
                    "gemini-er.point",
                    image=cam["rgb"],
                    query=f"the free tip of the {cable_description} held in the gripper",
                    max_points=1,
                )
            except Exception as exc:
                logger.info(
                    "threading-a-cable-tip-through-a-ring: gemini-er.point failed (%s)", exc
                )
                continue
            if not reply.get("found"):
                continue
            point = _deproject(cam, float(reply["pixel_x"]), float(reply["pixel_y"]))
            if point is not None:
                return point
        return None

    for look in range(ALIGN_LOOKS):
        seen = seen_tip()
        if seen is None:
            if camera:
                notes.append("the camera did not find the tip in front of the mouth")
            break
        out["tip_seen"] = True
        hand_now = here()
        R_now, p_now = _matrix(hand_now["rotation"]), _xyz(hand_now)
        expected = p_now + R_now @ tip_t
        if float(np.linalg.norm(seen - expected)) <= ALIGN_OK:
            break
        tip_t = R_now.T @ (seen - p_now)  # re-registered: aim the tip the camera sees
        why = fits(R_now)
        if why:
            return finish("misaligned", f"seen in front of the mouth: {why}")
        tcp_pre, tcp_end, push, _, _, _ = geometry(R_now)
        why = go("robot.go_to_pose_cartesian", tcp_pre, rot)
        if why:
            return finish(
                "unreachable", f"re-aiming the tip the camera saw (look {look + 1}): {why}"
            )

    if empty():
        out["cable_lost"] = True
        return finish(
            "jammed",
            "the jaws closed to their stop before the push: the cable slid out of the hand while it "
            "turned or carried the end (held back by the cable trailing behind); nothing was pushed",
        )

    # Push through along the axis, judged by the hand's travel.
    start_p = _xyz(here())
    why = move("robot.go_to_pose_cartesian", tcp_end, rot)
    if why.startswith(REFUSALS):
        return finish("unreachable", f"the push was refused: {why}")
    end_p = _xyz(here())
    travel = float((end_p - start_p) @ a)
    tip_start_s = float((start_p + R @ tip_t - c) @ a)
    tip_past = tip_start_s + travel
    out["hand_travel"], out["tip_past"] = travel, tip_past
    out["tip_position"] = _vec(end_p + _matrix(here()["rotation"]) @ tip_t)
    short = float(push) - travel
    note = ("; " + "; ".join(notes)) if notes else ""
    if empty():
        out["cable_lost"] = True
        return finish(
            "jammed",
            f"the jaws closed to their stop during the push: the cable slid out of the hand (held back "
            f"by the ring or the cable trailing behind) while the hand went {travel * 1e3:.0f} of "
            f"{push * 1e3:.0f} mm{note}",
        )

    seen = seen_tip()
    if seen is not None:
        out["tip_seen"] = True
        s_seen = float((seen - c) @ a)
        out["seen_tip_past"] = s_seen
        exit_now = end_p + _matrix(here()["rotation"]) @ exit_t
        e_s = float((exit_now - c) @ a)
        if s_seen < float(min_past):
            go("robot.go_to_pose_cartesian", tcp_pre, rot)
            return finish(
                "jammed",
                f"after the push the camera sees the tip {s_seen * 1e3:+.0f} mm from the ring's plane "
                f"(held back or bent on the rim; the hand went {travel * 1e3:.0f} of {push * 1e3:.0f} mm); "
                f"backed off to the mouth{note}",
            )
        f = 1.0 if e_s >= 0.0 else s_seen / (s_seen - e_s)
        cross = exit_now + (1.0 - f) * (seen - exit_now) if e_s < 0.0 else exit_now
        off = cross - c
        r = float(np.linalg.norm(off - (off @ a) * a))
        if r > float(ring_inner_radius) - radius:
            return finish(
                "misaligned",
                f"the tip went {s_seen * 1e3:.0f} mm past the plane but the line from the fingers to it "
                f"crosses it {r * 1e3:.0f} mm off the axis: outside the ring{note}",
            )
        return finish(
            "threaded",
            f"the camera sees the tip {s_seen * 1e3:.0f} mm past the ring's plane, the cable crossing it "
            f"{r * 1e3:.0f} mm off the axis; the hand went {travel * 1e3:.0f} of {push * 1e3:.0f} mm{note}",
        )

    if tip_past >= float(min_past) and short <= PUSH_TOLERANCE:
        route, why = (
            "threaded",
            f"the hand went {travel * 1e3:.0f} mm along the axis, putting the tip (as held, {how}) "
            f"{tip_past * 1e3:.0f} mm past the ring's plane, aimed {out['tip_offset'] * 1e3:.0f} mm off "
            f"its axis; the tip itself was not seen{note}",
        )
    elif tip_past >= float(min_past):
        route, why = (
            "threaded",
            f"the push stopped {short * 1e3:.0f} mm short, the tip (as held) already {tip_past * 1e3:.0f} mm "
            f"past the ring's plane; the tip itself was not seen{note}",
        )
    else:
        go("robot.go_to_pose_cartesian", tcp_pre, rot)
        return finish(
            "jammed",
            f"the push stopped {short * 1e3:.0f} mm short ({why or 'held back'}), the tip (as held) "
            f"{tip_past * 1e3:+.0f} mm from the ring's plane; backed off to the mouth{note}",
        )
    if release:
        ctx.tool("robot.open_gripper", **on_arm)
        up = _xyz(here()) + np.array([0.0, 0.0, 0.04])
        move("robot.go_to_pose_cartesian", up, rot)
    return finish(route, why)


def _deproject(cam: dict[str, Any], u: float, v: float) -> np.ndarray | None:
    """A pointed pixel in the world: the near (cable) depths round it, through the
    camera's intrinsics and camera-to-world pose (OpenCV axes)."""
    depth = np.asarray(cam["depth"], dtype=float)
    depth = depth.reshape(depth.shape[0], depth.shape[1])
    h, w = depth.shape
    iu, iv = int(round(u)), int(round(v))
    win = depth[
        max(iv - DEPTH_WINDOW, 0) : iv + DEPTH_WINDOW + 1,
        max(iu - DEPTH_WINDOW, 0) : iu + DEPTH_WINDOW + 1,
    ]
    win = win[np.isfinite(win) & (win > 0.05) & (win < 5.0)]
    if win.size < 3:
        return None
    near = win[win < float(np.median(win)) - FOREGROUND]
    if near.size < 3:
        return (
            None  # the thin cable has no depth of its own here: the window reads what lies behind
        )
    z = float(np.median(near))
    k = np.asarray(cam["intrinsics"], dtype=float)
    x = np.array([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z, 1.0])
    pose = cam["pose"]
    if isinstance(pose, dict):
        m = np.eye(4)
        m[:3, :3] = _matrix(pose["rotation"])
        m[:3, 3] = _xyz(pose)
    else:
        m = np.asarray(pose, dtype=float).reshape(4, 4)
    if not (0 <= iu < w and 0 <= iv < h):
        return None
    return (m @ x)[:3]
