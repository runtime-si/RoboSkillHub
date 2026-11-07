# Adapted from RoboRSI (https://github.com/nssmd/RoboRSI),
# roborsi/embodied/skills/base/grasp_rim/robotwin/policy.py and
# roborsi/embodied/skills/base/_lib/robotwin/_perception.py (container_opening), Apache-2.0.
# Changes: the container comes in as a world-frame oriented box (optionally refined
# by a rim band of its point cloud, RoboRSI's container_opening, taken along the
# box's own axes rather than world x / y); the rim points are the four wall
# midpoints in the box's frame, the nearest to the hand first (RoboRSI: the
# arm's side, then the opposite one), each previewed with robot.solve_ik; the
# jaws are turned across the wall with robot.grasp_frame, read back and finished
# on the last joint when the arm's solver leaves wrist roll free; the grip depth
# is the tool centre's, floored over the container's bottom; RoboRSI's
# is_holding + verify_holding_visual (simulator contact and a VLM) are replaced
# by the measured jaw gap against the wall thickness, after the close and after
# the lift; move refusals are routed rather than raised.
"""Grasp an open container by its rim: jaws across the wall, one inside the mouth, one outside."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

HOVER_HEIGHT = 0.06
"""Height of the hover over the grasp pose [m]. RoboRSI hovers 0.10 m; 0.06 keeps
a YAM's hover over a 40 mm box inside its reach and still clears the rim by
the fingertips' 10 mm with room to spare."""

GRIP_DEPTH = 0.012
"""How far below the rim top the tool centre goes [m] (RoboRSI's
``_BELOW_RIM_M``): the pads straddle the wall instead of skimming its edge."""

MIN_GRIP_DEPTH = 0.004
"""Shallowest grip worth closing [m]; a container whose floor leaves less is refused."""

LIFT_HEIGHT = 0.03
"""The lift after the close [m]. RoboRSI lifts 0.07 m to confirm the pinch; a
little is enough to see whether the wall stays between the pads."""

ASSUMED_WALL = 0.003
"""Wall thickness used to place the rim point when the caller states none [m].
It only shifts the tool centre by half of itself; the open jaws (94 mm on a YAM)
straddle any thin wall around it."""

EMPTY_MARGIN = 0.0008
"""A jaw gap within this of the closed stop is a close on nothing [m]. Under the
hub's usual 3 mm because a container wall is often thinner than that: in a YAM
station twin an empty close read 0.0 mm and a 2.3 mm wall 4.0-4.6 mm."""

FULL_MARGIN = 0.006
"""A jaw gap more than this over the wall thickness closed on something else [m]
(a corner, a handle, the whole wall's end), not across the wall. The twin's
2.3 mm wall read 1.7-2.3 mm over its thickness (the gap is the blade tips'
from the finger travel, not the pads' at the contact)."""

MAX_WALL = 0.012
"""Thickest wall accepted when the thickness is not given [m]."""

FLOOR_MARGIN = 0.003
"""Kept between the fingertips and the container's bottom [m]."""

MIN_TOOL_HEIGHT = 0.015
"""Lowest tool centre over the container's bottom [m]: a real YAM station refuses
a tool centre under 14 mm over its calibrated table."""

TIP_REACH = 0.0102
"""Fingertips past the tool centre [m] when robot.describe_gripper states none:
the YAM station's linear_4310, measured."""

MOUTH_MARGIN = 0.005
"""Kept between the open inner jaw and the opposite wall [m]."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from the hover or the lift [m]: no tighter than a real
YAM station's own 12 mm per-move bound."""

DESCENT_TOLERANCE = 0.008
"""How far above the grasp height the descent may stop [m]. A fingertip landing on
the rim top stops the tool centre grip_depth + its reach (22 mm) high."""

YAW_TOLERANCE_DEG = 5.0
"""How far the jaws may close off the wall's normal [deg] before the rest is
turned on the last joint; a servo's own bound is 3 deg, a real YAM station's
0.04 rad (2.3 deg)."""

TURN_TRIES = 3
"""Last-joint turns tried, each read back (the first learns which way the joint
turns the hand)."""

ROLL_LIMITS = (-2.0944, 2.0944)
"""Last joint's range [rad] when the caller states none: the i2rt YAM's joint 6."""

LIMIT_MARGIN = 0.05
"""Kept clear of each roll limit [rad]."""

RIM_BAND = 0.02
"""Depth of the rim band taken from a container cloud [m] (RoboRSI's ``rim_band_m``)."""

SIDES = ("+x", "-x", "+y", "-y")


class Output(TypedDict):
    route: str
    detail: str
    side: str
    grasp_position: dict[str, float]
    close_heading_deg: float
    jaw_error_deg: float
    wrist_turn_deg: float
    grip_width: float
    held_width: float
    wall_thickness: float
    lifted: float
    attempts: int


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


def yaw_between(now: dict[str, float], ref: dict[str, float]) -> float:
    """How far ``now`` is turned from ``ref`` about the vertical [deg]."""
    rel = _matrix(now) @ _matrix(ref).T
    return math.degrees(math.atan2(float(rel[1, 0]), float(rel[0, 0])))


def rim_frame(obb: dict[str, Any], cloud: Any = None, rim_band: float = RIM_BAND) -> dict[str, Any]:
    """The container seen from above: its centre, rim top, bottom, and its two
    horizontal axes with their half-extents (``{"+x": (unit, half), ...}``).

    With a cloud of the container (world frame, N x 3), the rim top, the
    opening's centre and its half-extents come from the cloud's top band
    (RoboRSI's ``container_opening``: rim at the 90th percentile of height,
    centre the band's mean, half-extent the 90th percentile of the band's
    distance from it), measured along the box's axes."""
    r = _matrix(obb.get("orientation"))
    half = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    centre = np.array([float(obb["center"][k]) for k in ("x", "y", "z")])
    reach_z = float(np.abs(r[2, :]) @ half)
    up = int(np.argmax(np.abs(r[2, :])))
    axes: dict[str, tuple[np.ndarray, float]] = {}
    for i, name in enumerate("xyz"):
        if i == up:
            continue
        flat = np.array([r[0, i], r[1, i]])
        flat = flat / (np.linalg.norm(flat) or 1.0)
        axes[f"+{name}"] = (flat, float(half[i]))
        axes[f"-{name}"] = (-flat, float(half[i]))
    frame = {
        "xy": centre[:2].copy(),
        "top": float(centre[2]) + reach_z,
        "bottom": float(centre[2]) - reach_z,
        "axes": axes,
        "source": "box",
    }
    points = None
    if cloud is not None:
        raw = cloud.get("points") if isinstance(cloud, dict) else cloud
        points = np.asarray(raw if raw is not None else [], dtype=np.float64).reshape(-1, 3)
    if points is not None and len(points) >= 12:
        rim_z = float(np.percentile(points[:, 2], 90))
        rim = points[points[:, 2] >= rim_z - float(rim_band)]
        if len(rim) >= 6:
            xy = rim[:, :2].mean(axis=0)
            frame["xy"], frame["top"], frame["source"] = xy, rim_z, "cloud"
            for name, (unit, _) in list(axes.items()):
                spread = float(np.percentile(np.abs((rim[:, :2] - xy) @ unit), 90))
                axes[name] = (unit, spread)
    return frame


def run(
    ctx: NodeContext,
    container_obb: dict[str, Any],
    container_cloud: Any = None,
    wall_thickness: float = -1.0,
    side: str = "auto",
    try_other_sides: bool = True,
    grip_depth: float = GRIP_DEPTH,
    hover_height: float = HOVER_HEIGHT,
    lift_height: float = LIFT_HEIGHT,
    empty_margin: float = EMPTY_MARGIN,
    full_margin: float = FULL_MARGIN,
    max_wall: float = MAX_WALL,
    yaw_tolerance_deg: float = YAW_TOLERANCE_DEG,
    roll_limits: list[float] | None = None,
    reach_tolerance: float = REACH_TOLERANCE,
    descent_tolerance: float = DESCENT_TOLERANCE,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    frame = rim_frame(container_obb, container_cloud)
    wall = float(wall_thickness) if wall_thickness and float(wall_thickness) > 0.0 else -1.0
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "side": "",
        "grasp_position": _vec(np.array([*frame["xy"], frame["top"]])),
        "close_heading_deg": 0.0,
        "jaw_error_deg": 0.0,
        "wrist_turn_deg": 0.0,
        "grip_width": -1.0,
        "held_width": -1.0,
        "wall_thickness": wall,
        "lifted": 0.0,
        "attempts": 0,
    }

    try:
        hand = ctx.tool("robot.describe_gripper", **on_arm)
    except Exception as exc:  # the width check and the reach fall back to defaults
        logger.info("grasping-container-rims: no gripper description (%s)", exc)
        hand = {}
    fit = hand.get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or hand.get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)
    finger = hand.get("finger") or {}
    tip_reach = float(finger.get("reach_m") or 0.0) if finger.get("stated", True) else 0.0
    tip_reach = tip_reach or TIP_REACH
    lo, hi = (float(v) for v in (roll_limits or ROLL_LIMITS))

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
            logger.info("grasping-container-rims: could not read the gripper (%s)", exc)
            return -1.0

    upper = (wall + float(full_margin)) if wall > 0.0 else float(max_wall)

    def verdict(w: float) -> str:
        """ "" when the gap reads a wall between the pads (or cannot be read), else why not."""
        if w < 0.0:
            return ""
        if w <= at_closed + float(empty_margin):
            return f"closed on nothing ({w * 1e3:.1f} mm)"
        if w > upper:
            return f"closed on something {w * 1e3:.1f} mm wide, over the {upper * 1e3:.1f} mm a wall may read"
        return ""

    def joints() -> np.ndarray:
        try:
            arms = ctx.tool("robot.get_observation").get("arms") or []
        except Exception as exc:
            logger.info("grasping-container-rims: no joint reading (%s)", exc)
            return np.zeros(0)
        if int(arm_id) >= len(arms):
            return np.zeros(0)
        q = arms[int(arm_id)].get("joint_state")
        if isinstance(q, dict):
            q = q.get("positions")
        return np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)

    def square_jaws(target: dict[str, float], at: np.ndarray, notes: list[str]) -> dict[str, Any]:
        """Read the jaws' heading back and, when the arm left the wrist roll free,
        finish the turn on the last joint and re-centre over ``at``. Returns the
        hand's pose."""
        now = here()
        gain = -1.0 if _matrix(target)[2, 2] < 0 else 1.0
        for _ in range(TURN_TRIES):
            err = _wrap_half(yaw_between(now["rotation"], target))
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
                if lo + LIMIT_MARGIN <= q[-1] + math.radians(r) / gain <= hi - LIMIT_MARGIN
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
            moved = math.radians(_wrap_half(yaw_between(now["rotation"], before)))
            out["wrist_turn_deg"] += math.degrees(moved)
            if abs(command) > math.radians(2.0) and 0.5 < abs(moved / command) < 1.5:
                gain = moved / command
            why = go("robot.go_to_pose", at, now["rotation"])
            if why:
                notes.append(f"re-centring after the wrist turn: {why}")
                break
            now = here()
        return now

    if side != "auto" and side not in frame["axes"]:
        raise ValueError(f"side must be 'auto' or one of {sorted(frame['axes'])}, got {side!r}")
    start = here()
    hand_xy = _xyz(start)[:2]

    def rim_point(name: str) -> np.ndarray:
        unit, half = frame["axes"][name]
        return frame["xy"] + unit * (half - (wall if wall > 0.0 else ASSUMED_WALL) / 2.0)

    names = sorted(frame["axes"], key=lambda n: float(np.linalg.norm(rim_point(n) - hand_xy)))
    if side != "auto":
        names = [side] + ([n for n in names if n != side] if try_other_sides else [])
    elif not try_other_sides:
        names = names[:1]

    floor = frame["bottom"] + max(tip_reach + FLOOR_MARGIN, MIN_TOOL_HEIGHT)
    grasp_z = max(frame["top"] - float(grip_depth), floor)
    if frame["top"] - grasp_z < MIN_GRIP_DEPTH:
        out["detail"] = (
            f"the container is too shallow: the tool centre may go only "
            f"{(frame['top'] - grasp_z) * 1e3:.1f} mm below its rim"
        )
        return out

    ctx.tool("robot.open_gripper", **on_arm)
    failures: list[str] = []
    descended = False
    for name in names:
        unit, half = frame["axes"][name]
        thick = wall if wall > 0.0 else ASSUMED_WALL
        if at_open > 0.0 and at_open / 2.0 > 2.0 * (half - thick) - MOUTH_MARGIN:
            failures.append(f"{name}: the open inner jaw would reach the opposite wall")
            continue
        point = np.array([*rim_point(name), grasp_z])
        hover = point + np.array([0.0, 0.0, float(hover_height)])
        heading = math.degrees(math.atan2(float(unit[1]), float(unit[0])))
        try:
            gf = ctx.tool("robot.grasp_frame", close_heading_deg=heading, **on_arm)
        except Exception as exc:
            failures.append(f"{name}: no grasp frame ({exc})")
            continue
        rotations = [
            gf[k] for k in ("rotation", "rotation_flipped") if isinstance(gf, dict) and gf.get(k)
        ]
        if check_reach:
            reachable = []
            for rot in rotations:
                try:
                    reply = ctx.tool(
                        "robot.solve_ik", pose={"position": _vec(point), "rotation": rot}, **on_arm
                    )
                except Exception as exc:
                    logger.info("grasping-container-rims: IK preview refused (%s)", exc)
                    continue
                # A stall at the edge of reach reads like a finger on the rim.
                if isinstance(reply, dict) and (
                    reply.get("success") is False or reply.get("within_servo_tolerance") is False
                ):
                    continue
                reachable.append(rot)
            rotations = reachable
        if not rotations:
            failures.append(f"{name}: the grasp pose is out of reach (IK preview)")
            continue
        out["attempts"] += 1
        out["side"] = name
        out["close_heading_deg"] = heading
        out["grasp_position"] = _vec(point)
        out["wrist_turn_deg"] = 0.0

        now = _xyz(here())
        if now[2] < hover[2] - 1.0e-3:
            why = go(
                "robot.go_to_pose_cartesian",
                np.array([now[0], now[1], hover[2]]),
                here()["rotation"],
            )
            if why:
                failures.append(f"{name}: rise: {why}")
                continue
        why, rot = "", rotations[0]
        for rot in rotations:
            why = go("robot.go_to_pose", hover, rot)
            if not why:
                break
        if why:
            failures.append(f"{name}: hover: {why}")
            continue
        notes: list[str] = []
        pose = square_jaws(rot, hover, notes)
        err = _wrap_half(yaw_between(pose["rotation"], rot))
        out["jaw_error_deg"] = err
        if abs(err) > float(yaw_tolerance_deg):
            failures.append(
                f"{name}: the jaws stayed {err:+.0f} deg off the wall's normal"
                + (f" ({'; '.join(notes)})" if notes else "")
            )
            continue

        descended = True
        why = move("robot.go_to_pose_cartesian", point, pose["rotation"])
        reached = _xyz(here())
        high = float(reached[2] - point[2])
        aside = float(np.linalg.norm(reached[:2] - point[:2]))
        if high > float(descent_tolerance) or aside > float(reach_tolerance):
            go("robot.go_to_pose_cartesian", np.array([*reached[:2], hover[2]]), pose["rotation"])
            failures.append(
                f"{name}: the descent stopped {high * 1e3:.0f} mm high and {aside * 1e3:.0f} mm aside"
                + (f" ({why})" if why else "")
                + "; a finger on the rim?"
            )
            continue

        out["grip_width"] = width(ctx.tool("robot.close_gripper", **on_arm))
        why = verdict(out["grip_width"])
        if why:
            ctx.tool("robot.open_gripper", **on_arm)
            go("robot.go_to_pose_cartesian", hover, pose["rotation"])
            failures.append(f"{name}: {why}")
            continue

        lift_note = ""
        if float(lift_height) > 0.0:
            lift_to = reached + np.array([0.0, 0.0, float(lift_height)])
            lift_why = move("robot.go_to_pose_cartesian", lift_to, pose["rotation"])
            out["lifted"] = float(_xyz(here())[2] - reached[2])
            if out["lifted"] < float(lift_height) - float(reach_tolerance):
                lift_note = (
                    f"; the lift rose {out['lifted'] * 1e3:.0f} of {float(lift_height) * 1e3:.0f} mm"
                    f" ({lift_why or 'stalled'}): the container is heavy, fixed or caught"
                )
        out["held_width"] = width()
        why = verdict(out["held_width"]) if float(lift_height) > 0.0 else ""
        if why:
            ctx.tool("robot.open_gripper", **on_arm)
            failures.append(f"{name}: the wall slipped out during the lift, {why}")
            continue
        unread = " (the jaw gap could not be read)" if out["grip_width"] < 0.0 else ""
        out["route"] = "grasped"
        out["detail"] = (
            f"rim {name}: closed at {out['grip_width'] * 1e3:.1f} mm"
            + (f" on a {wall * 1e3:.1f} mm wall" if wall > 0.0 else "")
            + (
                f", {out['held_width'] * 1e3:.1f} mm after the lift"
                if float(lift_height) > 0.0
                else ""
            )
            + unread
            + lift_note
        )
        return out

    out["route"] = "missed" if descended else "unreachable"
    out["detail"] = " | ".join(failures) or "no rim point to try"
    return out
