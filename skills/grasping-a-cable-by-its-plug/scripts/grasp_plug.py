# Adapted from yam-arcade (https://github.com/zzigak/yam-arcade),
# stations/cables/run.py (plan_pick_plugs, demo_plan_plug: a plug resting on the
# table, the pads closing across its horizontal width at the body centre, top-down,
# either finger direction, opened to the width plus PLUG_OPEN_MARGIN, the pad
# bottoms 1.5 mm over the plug's bottom and never over its mid-height, approach
# 6 cm over the table), MIT, Copyright (c) 2026 zzigak.
# Changes: the plug's pose comes from a perceived cable end (curve.endpoints, its
# outward direction is the plug's axis) and optionally a gemini-er.point on the
# plug with the depth under it, not from the simulator's plug body; the depth is
# the fingertips' (robot.describe_gripper's reach under the tool centre), floored
# at the station's lowest tool centre, from the plug's measured height; one
# robot.grasp_frame pair in place of the pitched-approach search (a YAM over a
# table reaches top-down) and an IK preview in place of the planner's error
# check; the wrist roll finished on the last joint when the arm's solver leaves
# it free; the force-capped close (15 N) is the connector's close_gripper; and
# the simulator's slip measurement is a test lift judged by the jaw gap against
# the plug's width (and, given a description, the plug seen to rise), with
# routed exits in place of the scripted carry into a tray.
"""Grasp a cable by its rigid plug: top-down across the plug's short axis at its body centre, lift-check."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PLUG_LENGTH = 0.020
"""Plug body length along the cable [m] when not given: a USB-C plug's moulded
body (yam-arcade plugs.py "usbc": 20 mm, then an 8.5 mm metal shell)."""

PLUG_WIDTH = 0.012
"""Plug body width across the cable, horizontal [m] (yam-arcade "usbc": 12 mm)."""

PLUG_HEIGHT = 0.008
"""Plug body height when neither given nor measured [m]. yam-arcade's USB-C body
is 6.5 mm; 8 mm is the thicker overmould of the charging leads the hub's cable
checks use. At a real YAM station's 14 mm floor the fingertips stand 3.8 mm over
the table, which is under 8 mm's mid-height and over 6.5 mm's."""

OPEN_MARGIN = 0.016
"""Jaw gap before closing = plug width + this [m] (yam-arcade PLUG_OPEN_MARGIN)."""

TIP_CLEARANCE = 0.0015
"""Fingertips over the plug's bottom at the grasp [m] (yam-arcade z_grasp: the pad
bottoms 1.5 mm over the plug's lowest point)."""

MAX_TIP_FRACTION = 0.5
"""Highest the fingertips may stand, as a fraction of the plug's height over its
bottom (yam-arcade rejects a grasp whose pads' lower corner is over half the
plug's height): higher and the pads squeeze the plug's upper edge and it pops out."""

RESTING_MAX = 0.003
"""A plug whose bottom stands more than this over the table is not resting on it
[m] (yam-arcade: hull bottom within 3 mm of the table)."""

APPROACH_HEIGHT = 0.06
"""Approach over the planned grasp height [m] (yam-arcade: 6 cm over the table)."""

TIP_REACH = 0.0102
"""Closed fingertips past the tool centre along the approach [m] when
robot.describe_gripper states none: the YAM linear_4310's measured value."""

MIN_TOOL_HEIGHT = 0.014
"""Lowest tool-centre height over the table [m]: a real YAM station refuses a
tool centre under 14 mm over its calibrated table."""

LIFT_HEIGHT = 0.04
"""The test lift [m]. yam-arcade lifts 10 cm while pulling 6 cm back on its way to
a tray; 4 cm is enough to see a plug come up or stay behind."""

WIDTH_SLACK = 0.004
"""How far the jaw gap may read under the plug's width and still hold it [m]: the
fingertips' compliance and a plug taken a little off square. Under it the jaws
took the jacket or the plug's edge; over the width by more than this the plug
is held askew (noted, still grasped)."""

EMPTY_MARGIN = 0.001
"""A closed jaw gap within this of the closed stop holds nothing [m]."""

MIN_RISE = 0.02
"""How far the plug seen by the camera after the lift must stand over where it
lay [m]: half the 4 cm lift."""

NEAR_HAND = 0.02
"""Plug points within this of the tool centre, in plan, count as in the hand [m]."""

MIN_POINTS = 15
"""Fewest plug points for the camera check to judge."""

MAX_POINT_OFFSET = 0.03
"""A pointed plug farther than this from the plug centre the cable end implies is
not this cable's plug [m]; the end is kept."""

HEIGHT_RANGE = (0.002, 0.030)
"""Plausible plug heights from the depth under the pointed pixel [m]."""

CONTACT_TOLERANCE = 0.006
"""A descent that stops more than this over the grasp height [m] met the plug or
the cable under a fingertip."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from a free move's target [m]: no tighter than a real
YAM station's own 12 mm per-move bound."""

YAW_TOLERANCE_DEG = 5.0
"""How far the jaws' closing direction may stay off the plug's short axis [deg]
before the rest is turned on the last joint."""

TURN_TRIES = 3
"""Last-joint turns tried (the first one learns which way the joint turns the hand)."""

ROLL_LIMITS = (-2.0944, 2.0944)
"""Last joint's range [rad] when none is stated: the i2rt YAM's joint 6."""

LIMIT_MARGIN = 0.05

TANGENT_M = 0.02
"""Length of centreline the plug's axis is taken over from its end [m]
(curve.endpoints' default tangent_m); about a plug's length."""

REFUSALS = ("unreachable", "bad_speed_scale")
"""Routed outcomes that mean a move was refused, not stopped by contact."""


class Output(TypedDict):
    route: str
    detail: str
    located_by: str
    grasp_point: dict[str, float]
    close_heading_deg: float
    plug_height: float
    plug_width: float
    planned_tool_z: float
    fingertips_over_table: float
    jaw_width: float
    jaw_width_lifted: float
    plug_rise: float
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


def _camera(obs: dict[str, Any], name: str) -> dict[str, Any] | None:
    cams = obs.get("cameras") or []
    if isinstance(cams, dict):
        cams = list(cams.values())
    cams = [c for c in cams if c.get("depth") is not None]
    named = [c for c in cams if name and name in str(c.get("name", ""))]
    return (named or cams[:1] or [None])[0]


def _point_at_plug(
    ctx: NodeContext, description: str, camera: str
) -> tuple[np.ndarray | None, str]:
    """The world point under gemini-er's point on the plug, or None and why."""
    try:
        cam = _camera(ctx.tool("robot.get_observation"), camera)
        if cam is None:
            return None, "no camera with depth"
        found = ctx.tool("gemini-er.point", image=cam["rgb"], query=description, max_points=1)
        if not found.get("found"):
            return None, f"gemini-er found no {description!r}"
        p = ctx.tool(
            "geometry.pixel_to_world_point",
            pixel_x=float(found["pixel_x"]),
            pixel_y=float(found["pixel_y"]),
            depth=cam["depth"],
            intrinsics=cam["intrinsics"],
            camera_pose=cam["pose"],
        )
        return _arr(p.get("point", p)), ""
    except Exception as exc:  # perception is optional: the cable end stands
        logger.info("grasping-a-cable-by-its-plug: pointing failed (%s)", exc)
        return None, f"pointing failed ({exc})"


def run(
    ctx: NodeContext,
    centerline: dict[str, Any] | None = None,
    end_node: int = 0,
    end: dict[str, Any] | None = None,
    tip: dict[str, float] | None = None,
    direction: dict[str, float] | None = None,
    table_z: float | None = None,
    plug_length: float = PLUG_LENGTH,
    plug_width: float = PLUG_WIDTH,
    plug_height: float | None = None,
    body_offset: float | None = None,
    plug_description: str = "",
    camera: str = "overhead",
    wrist_camera: str = "wrist",
    open_margin: float = OPEN_MARGIN,
    tip_clearance: float = TIP_CLEARANCE,
    max_tip_fraction: float = MAX_TIP_FRACTION,
    approach_height: float = APPROACH_HEIGHT,
    min_tool_height: float = MIN_TOOL_HEIGHT,
    lift_height: float = LIFT_HEIGHT,
    width_slack: float = WIDTH_SLACK,
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
        "located_by": "end",
        "grasp_point": {"x": 0.0, "y": 0.0, "z": 0.0},
        "close_heading_deg": 0.0,
        "plug_height": 0.0,
        "plug_width": float(plug_width),
        "planned_tool_z": 0.0,
        "fingertips_over_table": -1.0,
        "jaw_width": -1.0,
        "jaw_width_lifted": -1.0,
        "plug_rise": -1.0,
        "lift_checked": False,
    }

    def unreachable(why: str) -> Output:
        out["route"], out["detail"] = "unreachable", why
        return out

    # The plug's end and axis: the cable's tip and its outward direction.
    if end is not None:
        tip_p, axis = _arr(end["position"]), _arr(end["direction"])
    elif tip is not None and direction is not None:
        tip_p, axis = _arr(tip), _arr(direction)
    elif centerline is not None:
        pts = centerline.get("points") if isinstance(centerline, dict) else centerline
        points = np.asarray(pts or [], dtype=float).reshape(-1, 3)
        if len(points) < 2:
            return unreachable("the centreline has fewer than two points")
        if int(end_node) != 0:
            points = points[::-1]
        tip_p = points[0].copy()
        axis = tip_p - _along(points, TANGENT_M)
    else:
        raise ValueError(
            "give a centerline (and end_node), an end (curve.endpoints), or tip and direction"
        )
    axis[2] = 0.0
    if float(np.linalg.norm(axis)) < 1.0e-9:
        return unreachable("the cable's end points straight up or down: the plug is not lying flat")
    axis = axis / float(np.linalg.norm(axis))
    offset = float(plug_length) / 2.0 if body_offset is None else float(body_offset)
    centre = tip_p - axis * offset

    if table_z is None:
        try:
            table_z = float(ctx.tool("robot.describe_workspace", **on_arm)["surface_z"])
        except Exception as exc:
            return unreachable(f"no table height (give table_z): {exc}")
    table_z = float(table_z)
    notes: list[str] = []

    # Optionally, the plug pointed at, and its height from the depth under the point.
    measured_top = None
    if plug_description:
        p, why = _point_at_plug(ctx, plug_description, camera)
        if p is None:
            notes.append(f"plug not pointed ({why}): taken from the cable end")
        elif float(np.linalg.norm(p[:2] - centre[:2])) > MAX_POINT_OFFSET:
            notes.append(
                f"the pointed plug is {np.linalg.norm(p[:2] - centre[:2]) * 1e3:.0f} mm from the cable end's: "
                "taken from the cable end"
            )
        else:
            centre[:2] = p[:2]
            out["located_by"] = "point"
            if HEIGHT_RANGE[0] <= p[2] - table_z <= HEIGHT_RANGE[1]:
                measured_top = float(p[2])
    if plug_height is not None:
        height = float(plug_height)
    elif measured_top is not None:
        height = measured_top - table_z
    else:
        height = PLUG_HEIGHT
    out["plug_height"] = float(height)
    if measured_top is not None and measured_top - height > table_z + RESTING_MAX:
        return unreachable(
            f"the plug's top is {(measured_top - table_z) * 1e3:.1f} mm over the table, "
            f"{(measured_top - height - table_z) * 1e3:.1f} mm more than a {height * 1e3:.1f} mm plug "
            "resting on it: it lies on something"
        )
    if measured_top is None and float(tip_p[2]) - table_z > height + RESTING_MAX:
        return unreachable(
            f"the cable's end is {(tip_p[2] - table_z) * 1e3:.1f} mm over the table: the plug is not resting on it"
        )
    centre[2] = table_z + height / 2.0
    out["grasp_point"] = _vec(centre)

    # How deep: the fingertips just over the plug's bottom, never over its mid-height.
    try:
        hand = ctx.tool("robot.describe_gripper", **on_arm)
    except Exception as exc:
        logger.info("grasping-a-cable-by-its-plug: no gripper description (%s)", exc)
        hand = {}
    reach = float((hand.get("finger") or {}).get("reach_m") or TIP_REACH)
    fit = hand.get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or hand.get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)
    floor = table_z + float(min_tool_height)
    tool_z = max(table_z + float(tip_clearance) + reach, floor)
    tips_over = tool_z - reach - table_z
    out["planned_tool_z"] = float(tool_z)
    if tips_over > float(max_tip_fraction) * height + 1.0e-4:
        return unreachable(
            f"at the {float(min_tool_height) * 1e3:.1f} mm tool-centre floor the fingertips stand "
            f"{tips_over * 1e3:.1f} mm over the table, above {float(max_tip_fraction):.0%} of the "
            f"{height * 1e3:.1f} mm plug: they would squeeze its upper edge"
        )
    open_width = float(plug_width) + float(open_margin)
    if at_open > 0.0:
        if float(plug_width) + 0.002 > at_open:
            return unreachable(
                f"the {float(plug_width) * 1e3:.0f} mm plug is wider than the hand opens ({at_open * 1e3:.0f} mm)"
            )
        open_width = min(open_width, at_open)
    approach_z = tool_z + float(approach_height)
    heading = math.degrees(math.atan2(axis[1], axis[0])) + 90.0
    out["close_heading_deg"] = float(_wrap180(heading))

    # Top-down, the jaws closing across the plug: both wrist solutions (yam-arcade
    # tries either finger direction), previewed at the grasp and over it.
    try:
        frame = ctx.tool("robot.grasp_frame", close_heading_deg=float(heading), **on_arm)
    except Exception as exc:
        return unreachable(f"no top-down grasp frame ({exc})")
    rotations = [frame[k] for k in ("rotation", "rotation_flipped") if frame.get(k)]
    if not rotations:
        return unreachable("robot.grasp_frame returned no rotation")
    across = np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading)), 0.0])
    grasp_pose = np.array([centre[0], centre[1], tool_z])
    above = np.array([centre[0], centre[1], approach_z])
    chosen, refused = None, []
    for rot in rotations:
        if not check_reach:
            chosen = rot
            break
        why = ""
        for where_ in (grasp_pose, above):
            try:
                reply = ctx.tool(
                    "robot.solve_ik", pose={"position": _vec(where_), "rotation": rot}, **on_arm
                )
            except Exception as exc:
                logger.info("grasping-a-cable-by-its-plug: no IK preview (%s)", exc)
                break
            if isinstance(reply, dict) and (
                reply.get("success") is False or reply.get("within_servo_tolerance") is False
            ):
                why = str(reply.get("reason") or "IK preview failed")
                break
        if why:
            refused.append(why)
            continue
        chosen = rot
        break
    if chosen is None:
        return unreachable(f"no wrist solution reaches the plug ({'; '.join(refused)})")
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
        out["route"], out["detail"] = route, why + (("; " + "; ".join(notes)) if notes else "")
        return out

    def width_now(reply: Any = None) -> float:
        try:
            fraction = reply.get("position") if isinstance(reply, dict) else None
            if fraction is None:
                fraction = ctx.tool("robot.get_gripper", **on_arm)["position"]
            if at_open <= 0.0:
                return -1.0
            return at_closed + max(0.0, min(1.0, float(fraction))) * (at_open - at_closed)
        except Exception as exc:  # an unreadable hand is not an empty one
            logger.info("grasping-a-cable-by-its-plug: could not read the gripper (%s)", exc)
            return -1.0

    start = here()
    tcp = _xyz(start)
    if tcp[2] < approach_z - 1.0e-3:
        why = go(
            "robot.go_to_pose_cartesian", np.array([tcp[0], tcp[1], approach_z]), start["rotation"]
        )
        if why:
            return unreachable(f"rise: {why}")
    try:
        ctx.tool("robot.set_grip", width_m=float(open_width), **on_arm)
    except Exception as exc:
        logger.info("grasping-a-cable-by-its-plug: set_grip failed, opening fully (%s)", exc)
        ctx.tool("robot.open_gripper", **on_arm)
    why = go("robot.go_to_pose", above, chosen)
    if why:
        return unreachable(f"approach: {why}")

    # The jaws across the plug, read back: an arm whose solver keeps only the
    # approach axis arrives unturned, and the rest is turned on the last joint.
    def off_square(rot: dict[str, float]) -> float:
        c = _matrix(rot) @ close_local
        return _wrap180(math.degrees(math.atan2(c[1], c[0])) - heading)

    now = here()
    gain = -1.0 if _matrix(now["rotation"])[2, 2] < 0 else 1.0
    lo, hi = (float(v) for v in (roll_limits or ROLL_LIMITS))
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
        notes.append(f"the jaws close {square:+.0f} deg off the plug's short axis")

    # Straight down around the plug.
    why = move("robot.go_to_pose_cartesian", grasp_pose, rot)
    stop = where()
    out["fingertips_over_table"] = float(stop[2] - reach - table_z)
    if why.startswith(REFUSALS):
        return give_up("unreachable", f"descent: {why}", rot)
    if stop[2] - tool_z > CONTACT_TOLERANCE:
        return give_up(
            "missed",
            f"the descent stopped {(stop[2] - tool_z) * 1e3:.1f} mm over the grasp height "
            f"({why or 'stopped'}): a fingertip on the plug or the cable; re-perceive the plug",
            rot,
        )

    width = width_now(ctx.tool("robot.close_gripper", **on_arm))
    out["jaw_width"] = float(width)
    if 0.0 <= width <= at_closed + EMPTY_MARGIN:
        return give_up(
            "missed",
            f"the jaws closed to {width * 1e3:.1f} mm, at the closed stop: nothing in them",
            rot,
        )
    if 0.0 <= width < float(plug_width) - float(width_slack):
        return give_up(
            "missed",
            f"the jaws closed to {width * 1e3:.1f} mm, under the {float(plug_width) * 1e3:.0f} mm plug: "
            "they took the jacket or the plug's edge",
            rot,
        )

    # The test lift, and the check.
    held_at = where()
    lifted = np.array([held_at[0], held_at[1], held_at[2] + float(lift_height)])
    lift_why = go("robot.go_to_pose_cartesian", lifted, rot)
    if lift_why:
        notes.append(f"lift: {lift_why}")
    up_width = width_now()
    out["jaw_width_lifted"] = float(up_width)
    if 0.0 <= up_width < float(plug_width) - float(width_slack):
        return give_up(
            "missed",
            f"the jaws closed from {width * 1e3:.1f} to {up_width * 1e3:.1f} mm with the lift: "
            "the plug slipped out",
            rot,
        )
    judged = up_width if up_width >= 0.0 else width
    if judged > float(plug_width) + float(width_slack):
        notes.append(
            f"the jaws stand at {judged * 1e3:.1f} mm, wider than the {float(plug_width) * 1e3:.0f} mm plug: "
            "held askew or along its length"
        )
    note = ("; " + "; ".join(notes)) if notes else ""
    seen = (
        _rise_in_hand(ctx, plug_description, wrist_camera, where(), float(centre[2]))
        if plug_description
        else None
    )
    if seen is not None:
        out["lift_checked"] = True
        out["plug_rise"] = float(seen)
        if seen < float(min_rise):
            return give_up(
                "missed",
                f"the plug in the hand rose {seen * 1e3:.1f} mm with the {float(lift_height) * 1e3:.0f} mm lift "
                f"(under {float(min_rise) * 1e3:.0f} mm); jaws at {judged * 1e3:.1f} mm{note}",
                rot,
            )
    out["route"] = "grasped"
    out["detail"] = (
        f"jaws at {judged * 1e3:.1f} mm on the {float(plug_width) * 1e3:.0f} mm plug after the "
        f"{float(lift_height) * 1e3:.0f} mm lift"
        + (
            f"; the plug seen {seen * 1e3:.0f} mm up"
            if seen is not None
            else " (judged by the jaw gap only)"
        )
        + f"; located by the {out['located_by']}{note}"
    )
    return out


def _rise_in_hand(
    ctx: NodeContext, description: str, camera: str, hand: np.ndarray, before_z: float
) -> float | None:
    """How far the plug's points near the tool centre stand over its centre height
    before the grasp [m], from a camera after the lift; None when not seen."""
    try:
        cam = _camera(ctx.tool("robot.get_observation"), camera)
        if cam is None:
            return None
        found = ctx.tool("sam3.segment_text", image=cam["rgb"], query=description, max_results=1)
        if not found.get("masks"):
            return None
        cloud = ctx.tool(
            "geometry.mask_to_world_points",
            mask=np.asarray(found["masks"][0], dtype=np.uint8),
            depth=cam["depth"],
            intrinsics=cam["intrinsics"],
            camera_pose=cam["pose"],
        )["points"]
    except Exception as exc:
        logger.info("grasping-a-cable-by-its-plug: camera check failed (%s)", exc)
        return None
    pts = np.asarray(cloud.get("points") if isinstance(cloud, dict) else cloud, dtype=float)
    pts = pts.reshape(-1, 3)
    near = pts[np.linalg.norm(pts[:, :2] - hand[:2], axis=1) < NEAR_HAND]
    if len(near) < MIN_POINTS:
        return None
    return float(np.median(near[:, 2]) - before_z)
