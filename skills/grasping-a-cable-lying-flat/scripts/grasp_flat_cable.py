# Adapted from Newton (https://github.com/newton-physics/newton),
# newton/examples/multiphysics/example_franka_cable_ik_pick_place.py
# (Example._build_keyframes: approach above the cable -> descend to the cable's
# centre height -> close to capture -> lift holding), Apache-2.0, Copyright (c)
# 2026 The Newton Developers.
# Changes: the cable's grasp point and tangent come from a perceived centreline
# (curve.fit_centerline / curve.endpoints) or are given, not from a simulated
# rod; the jaws close across the local tangent (robot.grasp_frame, the wrist
# roll finished on the last joint when the arm's solver leaves it free);
# Newton's descent to a fixed height (the tool centre at the cable's centre) is
# a robot.move_cartesian_until_contact descent that stops on the table, aimed
# so the tapered fingertips end under the cable's middle and floored at the
# station's lowest tool centre; two options from a real YAM threading station
# (pre-narrowing the fingers before the last descent, and holding the hand
# pressed on the table while they shut); Newton's separate capture and hold
# widths are one close (its own constants are now equal); and a lift-check
# (the jaw gap, then the cable beside the fingers seen to rise with the hand)
# with routed exits instead of a fixed keyframe script.
"""Grasp a thin cable lying flat on a table: top-down across its tangent, fingertips down to the table, close, lift-check."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

APPROACH_HEIGHT = 0.06
"""Height of the approach over the planned grasp height [m]. Newton's is 0.20 m
for a Franka; a real YAM threading station hovers 0.06 m over a cable before
going down to it."""

PINCH_BELOW = 0.004
"""How far under the cable's centre the fingertips are aimed [m] (the real
threading station's PINCH_BELOW). Newton puts the tool centre at the cable's
centre; a YAM's tool centre is 10 mm over its fingertips, so it is the
fingertips that are aimed."""

THIN_TOP = 0.0034
"""A cable thinner than this [m] is thin: its fingertips go THIN_DEEPER further
under it and as close as THIN_CLEARANCE to the table. On the real station the
thin cable slipped out of the hand most."""

THIN_DEEPER = 0.0015
"""Extra depth under a thin cable [m] (real station)."""

TABLE_CLEARANCE = 0.0008
"""Nearest the fingertips are aimed to the table [m]; THIN_CLEARANCE for a thin
cable (real station, 0.8 / 0.4 mm). The descent stops on the table anyway."""

THIN_CLEARANCE = 0.0004

TIP_REACH = 0.0102
"""Closed fingertips past the tool centre along the approach [m] when
robot.describe_gripper states none: the YAM linear_4310's measured value."""

MIN_TOOL_HEIGHT = 0.014
"""Lowest tool-centre height over the table [m]. A real YAM station refuses a
tool centre under 14 mm over its calibrated table, which leaves closed
fingertips about 3.8 mm over it: over a 3.5 mm cable lying flat. Lower it only
on a station whose floor is lower (the threading station's motion server was
restarted with a 5 mm floor for this)."""

OVERTRAVEL = 0.008
"""How far under the planned tool height the contact descent is aimed [m]: twice
a servo's 4 mm arrival bound, so a move that calls itself arrived has still put
the fingertips on the table (aimed 4 mm under, a YAM station twin's servo
called the move done 3.1 mm short). The hand is then commanded back to where it
stands, or CLOSE_PRESS under it, before the fingers shut."""

CONTACT_TOLERANCE = 0.006
"""A descent that stops more than this over the planned tool height [m] stood
on something other than the table near the planned depth: the cable under a
fingertip, or a table higher than stated. Recorded in the detail; the grasp
goes on, as the real station's did."""

CLOSE_PRESS = 0.003
"""How far under where the hand stands it is commanded while the fingers shut
[m] (0: off), so the fingertips stay on the table and a flat cable cannot slip
under them. On the real station's fingertips 3 mm is about 1 N (0.3-0.4 N/mm).
Never under the tool-centre floor."""

PRE_CLOSE_WIDTH = 0.010
"""Jaw gap the fingers are narrowed to before the last descent [m] (0: off, the
hand stays fully open). Shut from half open with the tips on the table, the
real station's tapered fingertips wedged a flat cable up and over them and
closed empty; narrowed first to about 10 mm, the close is short."""

PRE_CLOSE_OVER = 0.012
"""Height over the planned tool height at which the fingers are narrowed [m]:
clear of a cable lying on the table (real station)."""

LIFT_HEIGHT = 0.04
"""The test lift [m]. The real station looked at the cable from 4 cm up: lower,
the cable lay nearer the wrist camera than its depth reaches."""

MIN_RISE = 0.007
"""How far the cable beside the fingers must rise with the hand to count as
held [m] (real station's LIFT_TEST_RISE)."""

BESIDE = (0.008, 0.025)
"""Ring around the tool centre, in plan [m], where the cable's points count as
beside the fingers (real station)."""

MIN_POINTS = 15
"""Fewest cable points beside the fingers for the camera check to judge."""

EMPTY_MARGIN = 0.001
"""A closed jaw gap within this of the closed stop holds nothing [m]."""

FLAT_MAX = 0.012
"""A centreline more than this over the table plus its radius at the grasp
point does not lie flat there [m]: take it in the air (grasping-linear-feature)."""

TANGENT_SPAN = 0.01
"""Half the stretch of centreline the tangent is taken over [m] (real station:
the cable's way from 1 cm either side)."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from the approach or the lift [m]: no tighter than a
real YAM station's own 12 mm per-move bound."""

YAW_TOLERANCE_DEG = 5.0
"""How far the jaws' closing direction may stay from square to the cable [deg]
before the rest is turned on the last joint."""

TURN_TRIES = 3
"""Last-joint turns tried (the first one learns which way the joint turns the hand)."""

ROLL_LIMITS = (-2.0944, 2.0944)
"""Last joint's range [rad] when none is stated: the i2rt YAM's joint 6."""

LIMIT_MARGIN = 0.05

MIN_DESCENT = 0.002
"""A contact move that moved less than this did not move at all [m]."""

CONTACT_MOVES = 3
"""Contact moves sent while one times out still descending."""

REFUSALS = ("unreachable", "bad_speed_scale")
"""Routed outcomes that mean a move was refused, not stopped by contact."""


class Output(TypedDict):
    route: str
    detail: str
    grasp_point: dict[str, float]
    close_heading_deg: float
    planned_tool_z: float
    contact_z: float
    fingertips_over_table: float
    touched_table: bool
    pressed: bool
    pre_closed: bool
    jaw_width: float
    cable_rise: float
    lift_checked: bool


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


def _wrap180(deg: float) -> float:
    """An angle between two undirected lines, in [-90, 90)."""
    return (float(deg) + 90.0) % 180.0 - 90.0


def _along(points: np.ndarray, s: float) -> np.ndarray:
    """The point at arc length ``s`` [m] along a polyline (clamped to its ends)."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arcs = np.concatenate([[0.0], np.cumsum(seg)])
    s = float(np.clip(s, 0.0, arcs[-1]))
    return np.array([np.interp(s, arcs, points[:, i]) for i in range(3)])


def grasp_on_centerline(
    points: np.ndarray, grasp_s: float, end_node: int | None, back_from_end: float
) -> tuple[np.ndarray, np.ndarray, str]:
    """Where along the centreline the fingers take it and the cable's horizontal
    way there: ``back_from_end`` along it from the end node ``end_node`` (0 or
    the last), else at the fraction ``grasp_s`` of its length."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    length = float(seg.sum())
    if end_node is not None:
        s = float(back_from_end) if int(end_node) == 0 else length - float(back_from_end)
        how = f"{back_from_end * 1e3:.0f} mm from the end at node {int(end_node)}"
    else:
        s = float(np.clip(grasp_s, 0.0, 1.0)) * length
        how = f"at {float(grasp_s):.2f} of its {length * 1e3:.0f} mm"
    s = float(np.clip(s, 0.0, length))
    point = _along(points, s)
    way = _along(points, s + TANGENT_SPAN) - _along(points, s - TANGENT_SPAN)
    return point, way, how


def run(
    ctx: NodeContext,
    centerline: dict[str, Any] | None = None,
    grasp_s: float = 0.5,
    end: dict[str, Any] | None = None,
    back_from_end: float = 0.03,
    grasp_point: dict[str, float] | None = None,
    tangent: dict[str, float] | None = None,
    table_z: float | None = None,
    cable_radius: float | None = None,
    pinch_below: float = PINCH_BELOW,
    press: float = CLOSE_PRESS,
    pre_close_width: float = PRE_CLOSE_WIDTH,
    approach_height: float = APPROACH_HEIGHT,
    overtravel: float = OVERTRAVEL,
    min_tool_height: float = MIN_TOOL_HEIGHT,
    lift_height: float = LIFT_HEIGHT,
    cable_description: str = "",
    camera: str = "wrist",
    min_rise: float = MIN_RISE,
    reach_tolerance: float = REACH_TOLERANCE,
    check_reach: bool = True,
    roll_limits: list[float] | None = None,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "grasp_point": {"x": 0.0, "y": 0.0, "z": 0.0},
        "close_heading_deg": 0.0,
        "planned_tool_z": 0.0,
        "contact_z": 0.0,
        "fingertips_over_table": -1.0,
        "touched_table": False,
        "pressed": False,
        "pre_closed": False,
        "jaw_width": -1.0,
        "cable_rise": -1.0,
        "lift_checked": False,
    }

    def unreachable(why: str) -> Output:
        out["route"], out["detail"] = "unreachable", why
        return out

    # The grasp point and the cable's way there.
    radius = cable_radius
    if grasp_point is not None:
        if tangent is None:
            raise ValueError("grasp_point needs tangent (the cable's direction there)")
        point, way, how = _arr(grasp_point), _arr(tangent), "at the given point"
    elif centerline is not None:
        pts = centerline.get("points") if isinstance(centerline, dict) else centerline
        points = np.asarray(pts or [], dtype=float).reshape(-1, 3)
        if len(points) < 2:
            return unreachable("the centreline has fewer than two points")
        if radius is None and isinstance(centerline, dict) and centerline.get("radius_m"):
            radius = float(centerline["radius_m"])
        node = None
        if end is not None:
            node = 0 if int(end.get("node", 0)) == 0 else len(points) - 1
        point, way, how = grasp_on_centerline(points, grasp_s, node, back_from_end)
    else:
        raise ValueError("give a centerline, or grasp_point and tangent")
    radius = float(radius) if radius else 0.00175
    way[2] = 0.0
    if float(np.linalg.norm(way)) < 1.0e-9:
        return unreachable("the cable's direction at the grasp point is vertical or unknown")
    way = way / float(np.linalg.norm(way))

    if table_z is None:
        try:
            table_z = float(ctx.tool("robot.describe_workspace", **on_arm)["surface_z"])
        except Exception as exc:  # fall back to the cable's own height
            logger.info("grasping-a-cable-lying-flat: no workspace surface (%s)", exc)
            table_z = float(point[2]) - radius
    table_z = float(table_z)
    if float(point[2]) - (table_z + radius) > FLAT_MAX:
        return unreachable(
            f"the cable is {(point[2] - table_z) * 1e3:.0f} mm over the table at the grasp point, "
            "not lying flat there"
        )
    # Lying flat its centre is a radius over the table: the camera's height of
    # a 3.5 mm cable is noisier than that.
    centre_z = table_z + radius
    point[2] = centre_z
    out["grasp_point"] = _vec(point)

    # How deep: Newton puts the tool centre at the cable's centre; here the
    # fingertips go under it, never into the table.
    try:
        hand = ctx.tool("robot.describe_gripper", **on_arm)
    except Exception as exc:
        logger.info("grasping-a-cable-lying-flat: no gripper description (%s)", exc)
        hand = {}
    reach = float((hand.get("finger") or {}).get("reach_m") or TIP_REACH)
    fit = hand.get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or hand.get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)
    thin = 2.0 * radius < THIN_TOP
    tips_z = max(
        centre_z - float(pinch_below) - (THIN_DEEPER if thin else 0.0),
        table_z + (THIN_CLEARANCE if thin else TABLE_CLEARANCE),
    )
    tool_z = tips_z + reach
    floor = table_z + float(min_tool_height)
    out["planned_tool_z"] = float(tool_z)
    if tool_z < floor - 1.0e-3:
        return unreachable(
            f"pinching the {2 * radius * 1e3:.1f} mm cable needs the tool centre "
            f"{(tool_z - table_z) * 1e3:.1f} mm over the table (fingertips {(tips_z - table_z) * 1e3:.1f} mm), "
            f"under the {float(min_tool_height) * 1e3:.1f} mm floor: at the floor the fingertips stand "
            f"{(floor - reach - table_z) * 1e3:.1f} mm over the table and close over the cable"
        )
    tool_z = max(tool_z, floor)
    bottom_z = max(tool_z - float(overtravel), floor)
    approach_z = tool_z + float(approach_height)
    heading = math.degrees(math.atan2(way[1], way[0])) + 90.0
    out["close_heading_deg"] = float(_wrap180(heading))

    # Top-down rotations with the jaws closing across the cable, both wrist
    # solutions, previewed at the grasp pose.
    try:
        frame = ctx.tool("robot.grasp_frame", close_heading_deg=float(heading), **on_arm)
    except Exception as exc:
        return unreachable(f"no top-down grasp frame ({exc})")
    rotations = [frame[k] for k in ("rotation", "rotation_flipped") if frame.get(k)]
    if not rotations:
        return unreachable("robot.grasp_frame returned no rotation")
    across = np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading)), 0.0])
    grasp_pose = np.array([point[0], point[1], tool_z])
    chosen, refused = None, []
    for rot in rotations:
        if not check_reach:
            chosen = rot
            break
        try:
            reply = ctx.tool(
                "robot.solve_ik", pose={"position": _vec(grasp_pose), "rotation": rot}, **on_arm
            )
        except Exception as exc:
            logger.info("grasping-a-cable-lying-flat: no IK preview (%s)", exc)
            chosen = rot
            break
        # A stall at the edge of reach reads exactly like contact.
        if isinstance(reply, dict) and (
            reply.get("success") is False or reply.get("within_servo_tolerance") is False
        ):
            refused.append(str(reply.get("reason") or "IK preview failed"))
            continue
        chosen = rot
        break
    if chosen is None:
        return unreachable(f"no wrist solution reaches the grasp pose ({'; '.join(refused)})")
    close_local = _matrix(chosen).T @ across

    def move(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        try:
            reply = ctx.tool(tool, pose={"position": _vec(position), "rotation": rot}, **on_arm)
            return _refusal(reply)
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc) or type(exc).__name__

    def here() -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **on_arm))

    def where() -> np.ndarray:
        return _xyz(here())

    def go(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        """A free move judged by where the hand is: "" when it arrived within
        reach_tolerance, otherwise why not."""
        why = move(tool, position, rot)
        off = float(np.linalg.norm(where() - position))
        if (why and not why.startswith("timeout")) or off > float(reach_tolerance):
            return f"{why or 'ended off'}; the hand is {off * 1e3:.0f} mm from it"
        return ""

    def give_up(route: str, why: str, rot: dict[str, float]) -> Output:
        ctx.tool("robot.open_gripper", **on_arm)
        up = where()
        up[2] = max(up[2], approach_z)
        move("robot.go_to_pose_cartesian", up, rot)
        out["route"], out["detail"] = route, why
        return out

    start = here()
    tcp = _xyz(start)
    if tcp[2] < approach_z - 1.0e-3:
        why = go(
            "robot.go_to_pose_cartesian", np.array([tcp[0], tcp[1], approach_z]), start["rotation"]
        )
        if why:
            return unreachable(f"rise: {why}")
    ctx.tool("robot.open_gripper", **on_arm)
    above = np.array([point[0], point[1], approach_z])
    why = go("robot.go_to_pose", above, chosen)
    if why:
        return unreachable(f"approach: {why}")

    # The jaws square to the cable, read back: an arm whose solver keeps only
    # the approach axis arrives unturned, and the rest is turned on the last joint.
    def off_square(rot: dict[str, float]) -> float:
        c = _matrix(rot) @ close_local
        return _wrap180(math.degrees(math.atan2(c[1], c[0])) - heading)

    now = here()
    gain = -1.0 if _matrix(now["rotation"])[2, 2] < 0 else 1.0
    lo, hi = (float(v) for v in (roll_limits or ROLL_LIMITS))
    notes: list[str] = []
    for _ in range(TURN_TRIES):
        residual = -off_square(now["rotation"])
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
        options = [residual, residual - math.copysign(180.0, residual)]
        fits = [r for r in options if room[0] <= math.radians(r) / gain <= room[1]]
        if not fits:
            notes.append(f"the last joint has no room to turn {residual:+.0f} deg")
            break
        command = math.radians(fits[0]) / gain
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
        before = off_square(now["rotation"])
        now = here()
        moved = math.radians(_wrap180(off_square(now["rotation"]) - before))
        if abs(command) > math.radians(2.0) and 0.5 < abs(moved / command) < 1.5:
            gain = moved / command
        why = go("robot.go_to_pose", above, now["rotation"])
        if why:
            return give_up(
                "unreachable", f"re-centring after the wrist turn: {why}", now["rotation"]
            )
        now = here()
    rot = now["rotation"]
    square = off_square(rot)
    if abs(square) > 2.0 * YAW_TOLERANCE_DEG:
        notes.append(f"the jaws close {square:+.0f} deg off square to the cable")

    # Narrow the fingers over the cable before the last descent.
    if float(pre_close_width) > 0.0:
        why = go(
            "robot.go_to_pose_cartesian",
            np.array([point[0], point[1], tool_z + PRE_CLOSE_OVER]),
            rot,
        )
        if why:
            return give_up("unreachable", f"down over the cable: {why}", rot)
        try:
            ctx.tool("robot.set_grip", width_m=float(pre_close_width), **on_arm)
            out["pre_closed"] = True
        except Exception as exc:
            notes.append(f"the fingers were not narrowed ({exc})")

    # Down until the fingertips meet the table (the arm's stall detection).
    goal = np.array([point[0], point[1], bottom_z])
    stop, why = where(), ""
    for _ in range(CONTACT_MOVES):
        before = stop
        why = move("robot.move_cartesian_until_contact", goal, rot)
        stop = where()
        if not why.startswith("timeout") or before[2] - stop[2] < MIN_DESCENT:
            break
    if why.startswith(REFUSALS):
        return give_up("unreachable", f"descent: {why}", rot)
    out["contact_z"] = float(stop[2])
    out["fingertips_over_table"] = float(stop[2] - reach - table_z)
    stopped = float(stop[2] - goal[2]) > MIN_DESCENT
    # Only a descent aimed past where the fingertips meet the table can have
    # stopped on it; one aimed at the floor over it stops on nothing.
    reaches_table = goal[2] < table_z + reach - MIN_DESCENT
    out["touched_table"] = bool(stopped and reaches_table and stop[2] - tool_z <= CONTACT_TOLERANCE)
    if stopped and stop[2] - tool_z > CONTACT_TOLERANCE:
        notes.append(
            f"the descent stopped {(stop[2] - tool_z) * 1e3:.1f} mm over the planned depth "
            "(the cable under a fingertip, or the table higher than stated)"
        )
    elif stopped and not reaches_table:
        notes.append(
            f"the descent stopped {(stop[2] - goal[2]) * 1e3:.1f} mm over its aim, which is over the table"
        )
    elif not stopped:
        notes.append(
            "the descent met nothing down to "
            + ("the tool-centre floor" if bottom_z <= floor + 1e-6 else "its bottom")
        )

    # Close, held on the table (CLOSE_PRESS) or resting where the hand stands.
    if float(press) > 0.0:
        low = np.array([stop[0], stop[1], max(stop[2] - float(press), floor)])
        move("robot.go_to_pose_cartesian", low, rot)
        out["pressed"] = bool(low[2] < stop[2] - 1.0e-4)
    else:
        move("robot.go_to_pose_cartesian", stop, rot)

    def width_now(reply: Any) -> float:
        try:
            fraction = reply.get("position") if isinstance(reply, dict) else None
            if fraction is None:
                fraction = ctx.tool("robot.get_gripper", **on_arm)["position"]
            if at_open <= 0.0:
                return -1.0
            return at_closed + max(0.0, min(1.0, float(fraction))) * (at_open - at_closed)
        except Exception as exc:  # an unreadable hand is not an empty one
            logger.info("grasping-a-cable-lying-flat: could not read the gripper (%s)", exc)
            return -1.0

    width = width_now(ctx.tool("robot.close_gripper", **on_arm))
    if width > 4.0 * radius + 0.004:
        # Stopped part way (the real station's force cap on a long close): once more.
        notes.append(f"the first close stopped at {width * 1e3:.1f} mm")
        width = width_now(ctx.tool("robot.close_gripper", **on_arm))
    out["jaw_width"] = float(width)

    # The test lift, and the check.
    held_at = where()
    lifted = np.array([held_at[0], held_at[1], held_at[2] + float(lift_height)])
    lift_why = go("robot.go_to_pose_cartesian", lifted, rot)
    if lift_why:
        notes.append(f"lift: {lift_why}")
    note = ("; " + "; ".join(notes)) if notes else ""
    if 0.0 <= width <= at_closed + EMPTY_MARGIN:
        return give_up(
            "missed",
            f"the jaws closed to {width * 1e3:.1f} mm, at the closed stop: nothing in them{note}",
            rot,
        )
    seen = (
        _rise_beside(ctx, cable_description, camera, where(), centre_z)
        if cable_description
        else None
    )
    if seen is not None:
        out["lift_checked"] = True
        out["cable_rise"] = float(seen)
        if seen < float(min_rise):
            return give_up(
                "missed",
                f"the cable beside the fingers rose {seen * 1e3:.1f} mm with the hand "
                f"(under {float(min_rise) * 1e3:.0f} mm); jaws at {width * 1e3:.1f} mm{note}",
                rot,
            )
        out["route"] = "grasped"
        out["detail"] = (
            f"the cable beside the fingers rose {seen * 1e3:.1f} mm with the hand; "
            f"jaws at {width * 1e3:.1f} mm, taken {how}{note}"
        )
        return out
    out["route"] = "grasped"
    out["detail"] = (
        f"jaws stopped at {width * 1e3:.1f} mm, short of the closed stop; the cable was not seen "
        f"after the lift (judged by the jaw gap only), taken {how}{note}"
    )
    return out


def _rise_beside(
    ctx: NodeContext, description: str, camera: str, hand: np.ndarray, before_z: float
) -> float | None:
    """How far the cable's points beside the fingers stand over its height before
    the grasp [m], from a camera after the lift; None when it is not seen."""
    try:
        obs = ctx.tool("robot.get_observation")
    except Exception as exc:
        logger.info("grasping-a-cable-lying-flat: no observation (%s)", exc)
        return None
    cams = obs.get("cameras") or []
    if isinstance(cams, dict):
        cams = list(cams.values())
    cams = [c for c in cams if c.get("depth") is not None]
    named = [c for c in cams if camera and camera in str(c.get("name", ""))]
    for cam in named or cams[:1]:
        try:
            found = ctx.tool(
                "sam3.segment_text", image=cam["rgb"], query=description, max_results=1
            )
            if not found.get("masks"):
                continue
            cloud = ctx.tool(
                "geometry.mask_to_world_points",
                mask=np.asarray(found["masks"][0], dtype=np.uint8),
                depth=cam["depth"],
                intrinsics=cam["intrinsics"],
                camera_pose=cam["pose"],
            )["points"]
        except Exception as exc:
            logger.info("grasping-a-cable-lying-flat: camera check failed (%s)", exc)
            continue
        pts = np.asarray(cloud.get("points") if isinstance(cloud, dict) else cloud, dtype=float)
        pts = pts.reshape(-1, 3)
        d = np.linalg.norm(pts[:, :2] - hand[:2], axis=1)
        near = pts[(d > BESIDE[0]) & (d < BESIDE[1])]
        if len(near) >= MIN_POINTS:
            return float(np.median(near[:, 2]) - before_z)
    return None
