# Adapted from RoboRSI (https://github.com/nssmd/RoboRSI),
# roborsi/embodied/skills/base/grasp_flat/robotwin/policy.py, Apache-2.0.
# Changes: the object comes in as a world-frame oriented box (optionally with its
# point cloud, whose 5th height percentile is the support as in RoboRSI, or an
# explicit support height) instead of a Grounded-SAM mask read inside the
# simulator; the grasp height is the tool centre's, floored 15 mm over the support
# (a real YAM station refuses lower) with the fingertips' reach below it, and an
# axis whose fingertips would end above the slab's top is refused rather than
# closed on air; an optional tilt about the close plane's normal drops one
# fingertip toward the support (not in RoboRSI); each axis is previewed with
# robot.solve_ik and the jaws turned with robot.grasp_frame, read back and finished
# on the last joint when the arm's solver leaves wrist roll free; RoboRSI's
# is_holding + verify_holding_visual (simulator contact and a VLM) are replaced by
# the jaw gap measured against the slab's width, after the close and after the
# lift; move refusals are routed rather than raised.
"""Grasp a slab lying flat on a surface: fingertips just over the support, jaws across its narrow side."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

HOVER_HEIGHT = 0.06
"""Height of the hover over the grasp pose [m]. RoboRSI hovers 0.09 m over the
grasp height; 0.06 still clears a slab by the fingertips' 10 mm and keeps the
hover well inside a YAM's reach."""

GRASP_FRACTION = 0.45
"""Where the fingertips close, as a fraction of the slab's thickness over the
support (RoboRSI's ``_GRASP_FRAC``): the body near its base, not the top face."""

MIN_TIP_CLEARANCE = 0.004
"""Least height of the lowest fingertip over the support [m] (RoboRSI's
``_MIN_ABOVE_TABLE_M``)."""

MIN_TOOL_HEIGHT = 0.015
"""Lowest tool centre over the support [m]: a real YAM station refuses a tool
centre under 14 mm over its calibrated table. With the 10.2 mm fingertip reach
this puts a top-down pinch's fingertips 4.8 mm over the table."""

MIN_BITE = 0.0015
"""Least slab the fingertips must overlap below its top [m]; a slab whose top is
lower than the fingertips plus this cannot be pinched with this tilt."""

TIP_REACH = 0.0102
"""Fingertips past the tool centre [m] when robot.describe_gripper states none:
the YAM station's linear_4310, measured."""

OPEN_CLEARANCE = 0.012
"""Each open fingertip's distance outside the slab's edge [m] when the jaws are
opened only part way (a tilted grasp, so the lower fingertip stays near the support)."""

TILT_TIP_CLEARANCE = 0.001
"""Height of the lower fingertip over the support while a tilted grasp closes
[m]: it rides just over the surface into the slab's bottom edge (RoboRSI's 4 mm
floor is for a fingertip that does not track the surface)."""

FOLLOW_STEPS = 4
"""Steps in which a tilted grasp narrows the jaws to the slab's width, the hand
following each down so the lower fingertip keeps its height."""

LIFT_HEIGHT = 0.05
"""The lift after the close [m]. RoboRSI lifts 0.06 m to confirm the pinch."""

EMPTY_MARGIN = 0.003
"""A jaw gap within this of the closed stop is a close on nothing [m]: the hub's
usual margin (a real YAM station's empty close read 1.8 mm, a station twin's 0.0)."""

FULL_MARGIN = 0.008
"""A jaw gap more than this over the slab's width across the jaws closed on
something wider [m]. A YAM's gap reads the blade tips' separation, 2 mm over a
2.3 mm wall in a station twin (grasping-container-rims)."""

NARROW_MARGIN = 0.006
"""A jaw gap more than this under the slab's width across the jaws did not close
across it [m] (a fingertip went over its top or past a corner). Level pinches in
a station twin read 0.1 mm under to 1.3 mm over a 40 mm side; tilted closes that
pushed the slab aside read 5-26 mm."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from the hover or the lift [m]: no tighter than a real
YAM station's own 12 mm per-move bound."""

DESCENT_TOLERANCE = 0.006
"""How far above the grasp height the descent may stop [m]. Over a servo's own
4 mm arrival bound (a station twin's servo reported "reached" 2.5 mm high over
the 8 mm slab). A fingertip on a slab thinner than this plus the fingertips'
height is not told apart from arrival by the pose; the gap check after the close
catches it."""

YAW_TOLERANCE_DEG = 5.0
"""How far the jaws may close off the slab's axis [deg] before the rest is turned
on the last joint; a servo's own bound is 3 deg, a real YAM station's 0.04 rad."""

TURN_TRIES = 3
"""Last-joint turns tried, each read back (the first learns which way the joint
turns the hand)."""

ROLL_LIMITS = (-2.0944, 2.0944)
"""Last joint's range [rad] when the caller states none: the i2rt YAM's joint 6."""

LIMIT_MARGIN = 0.05
"""Kept clear of each roll limit [rad]."""

MAX_TILT_DEG = 35.0
"""Steepest tilt accepted [deg]; beyond it the hand's body nears the support."""


class Output(TypedDict):
    route: str
    detail: str
    axis: str
    close_heading_deg: float
    tilt_deg: float
    grasp_position: dict[str, float]
    tip_height: float
    object_width: float
    thickness: float
    jaw_error_deg: float
    wrist_turn_deg: float
    grip_width: float
    held_width: float
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


def jaw_error_deg(
    now: dict[str, float], target: dict[str, float], close: np.ndarray, approach: np.ndarray
) -> float:
    """How far the hand at ``now`` has its jaws turned about ``approach`` from where
    ``target`` puts them (``close`` is the target's close axis, world) [deg]."""
    turned = _matrix(now) @ _matrix(target).T @ close
    return math.degrees(
        math.atan2(float(np.cross(close, turned) @ approach), float(close @ turned))
    )


def slab_frame(
    obb: dict[str, Any], cloud: Any = None, support_z: float | None = None
) -> dict[str, Any]:
    """The slab seen from above: centre, top, support height, and its two
    horizontal axes with their half-extents, the shorter first.

    The support is ``support_z`` when given, else the 5th height percentile of
    ``cloud`` (RoboRSI: "the slab's floor; a low percentile sheds depth noise"),
    else the box's bottom."""
    r = _matrix(obb.get("orientation"))
    half = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    centre = np.array([float(obb["center"][k]) for k in ("x", "y", "z")])
    reach_z = float(np.abs(r[2, :]) @ half)
    up = int(np.argmax(np.abs(r[2, :])))
    axes = []
    for i in range(3):
        if i == up:
            continue
        flat = np.array([r[0, i], r[1, i]])
        flat = flat / (np.linalg.norm(flat) or 1.0)
        axes.append((flat, float(half[i])))
    axes.sort(key=lambda a: a[1])
    top = float(centre[2]) + reach_z
    bottom = float(centre[2]) - reach_z
    source = "box"
    if support_z is not None:
        bottom, source = float(support_z), "given"
    elif cloud is not None:
        raw = cloud.get("points") if isinstance(cloud, dict) else cloud
        points = np.asarray(raw if raw is not None else [], dtype=np.float64).reshape(-1, 3)
        if len(points) >= 12:
            bottom, source = float(np.percentile(points[:, 2], 5)), "cloud"
    return {
        "xy": centre[:2].copy(),
        "top": top,
        "support": bottom,
        "short": axes[0],
        "long": axes[1],
        "source": source,
    }


def diagonal_tilt_deg(thickness: float, half_width: float, tip_reach: float = TIP_REACH) -> float:
    """The tilt for ``auto_tilt`` [deg], capped at MAX_TILT_DEG: the close line
    runs from the slab's near bottom edge (the lower fingertip TILT_TIP_CLEARANCE
    over the support) to just under its far top edge (MIN_BITE below it), and is
    at least steep enough that the tool-centre floor still lets the lower
    fingertip come down to that clearance at the edge."""
    if half_width <= 0.0:
        return 0.0
    rise = float(thickness) - TILT_TIP_CLEARANCE - MIN_BITE
    diagonal = math.atan2(max(rise, 0.0), 2.0 * float(half_width))

    def short(t: float) -> float:  # > 0 while the floor keeps the fingertip high
        return (
            MIN_TOOL_HEIGHT
            - tip_reach * math.cos(t)
            - half_width * math.tan(t)
            - TILT_TIP_CLEARANCE
        )

    lo, hi = 0.0, math.radians(MAX_TILT_DEG)
    if short(hi) > 0.0:
        floor = hi
    elif short(lo) <= 0.0:
        floor = 0.0
    else:
        for _ in range(50):
            mid = (lo + hi) / 2.0
            lo, hi = (mid, hi) if short(mid) > 0.0 else (lo, mid)
        floor = hi
    return min(MAX_TILT_DEG, math.degrees(max(diagonal, floor)))


def pinch_geometry(
    support: float,
    top: float,
    half_width: float,
    open_width: float,
    tilt_rad: float,
    tip_reach: float,
    grasp_fraction: float = GRASP_FRACTION,
    min_tip_clearance: float = MIN_TIP_CLEARANCE,
) -> dict[str, float]:
    """Heights for a pinch across a slab of ``half_width`` [m], the jaws tilted
    ``tilt_rad`` about the normal of their close plane.

    Level: the fingertips at ``grasp_fraction`` of the thickness (at least
    ``min_tip_clearance``), the tool centre no lower than MIN_TOOL_HEIGHT.
    Tilted: the hand comes down with the jaws ``open_width`` apart and the lower
    open fingertip TILT_TIP_CLEARANCE over the support, then follows the closing
    jaws down so that fingertip stays there until it meets the slab's edge.

    Returns the fingertips' midpoint height at the descent (``start_mid_z``) and
    at the close (``mid_z``), the tool centre's at the close (``tool_z``), the
    lower and upper fingertips' heights at the slab's edges (``tip_z``,
    ``upper_z``) and the ``bite`` the lower one has below the slab's top."""
    s, c = math.sin(abs(tilt_rad)), math.cos(tilt_rad)
    floor_mid = support + MIN_TOOL_HEIGHT - tip_reach * c
    at_edge = half_width / c
    if s == 0.0:
        thickness = top - support
        mid_z = max(
            support + max(float(min_tip_clearance), float(grasp_fraction) * thickness), floor_mid
        )
        start = mid_z
    else:
        start = max(support + TILT_TIP_CLEARANCE + open_width / 2.0 * s, floor_mid)
        mid_z = max(support + TILT_TIP_CLEARANCE + at_edge * s, floor_mid)
    tip_z = mid_z - at_edge * s
    return {
        "start_mid_z": start,
        "mid_z": mid_z,
        "tool_z": mid_z + tip_reach * c,
        "tip_z": tip_z,
        "upper_z": mid_z + at_edge * s,
        "bite": top - tip_z,
    }


def run(
    ctx: NodeContext,
    object_obb: dict[str, Any],
    object_cloud: Any = None,
    support_z: float | None = None,
    tilt_deg: float = 0.0,
    auto_tilt: bool = False,
    try_long_axis: bool = True,
    grasp_fraction: float = GRASP_FRACTION,
    min_tip_clearance: float = MIN_TIP_CLEARANCE,
    hover_height: float = HOVER_HEIGHT,
    lift_height: float = LIFT_HEIGHT,
    empty_margin: float = EMPTY_MARGIN,
    full_margin: float = FULL_MARGIN,
    narrow_margin: float = NARROW_MARGIN,
    yaw_tolerance_deg: float = YAW_TOLERANCE_DEG,
    roll_limits: list[float] | None = None,
    reach_tolerance: float = REACH_TOLERANCE,
    descent_tolerance: float = DESCENT_TOLERANCE,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    frame = slab_frame(object_obb, object_cloud, support_z)
    thickness = frame["top"] - frame["support"]
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "axis": "",
        "close_heading_deg": 0.0,
        "tilt_deg": abs(float(tilt_deg)),
        "grasp_position": _vec(np.array([*frame["xy"], frame["top"]])),
        "tip_height": -1.0,
        "object_width": -1.0,
        "thickness": thickness,
        "jaw_error_deg": 0.0,
        "wrist_turn_deg": 0.0,
        "grip_width": -1.0,
        "held_width": -1.0,
        "lifted": 0.0,
        "attempts": 0,
    }
    if abs(float(tilt_deg)) > MAX_TILT_DEG:
        out["detail"] = (
            f"a {float(tilt_deg):.0f} deg tilt is steeper than the {MAX_TILT_DEG:.0f} deg allowed"
        )
        return out
    if thickness <= 0.0:
        out["detail"] = f"the slab's top is not above its support ({thickness * 1e3:.1f} mm)"
        return out

    try:
        hand = ctx.tool("robot.describe_gripper", **on_arm)
    except Exception as exc:  # the width check and the reach fall back to defaults
        logger.info("grasping-flat-objects: no gripper description (%s)", exc)
        hand = {}
    fit = hand.get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or hand.get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)
    finger = hand.get("finger") or {}
    tip_reach = float(finger.get("reach_m") or 0.0) if finger.get("stated", True) else 0.0
    tip_reach = tip_reach or TIP_REACH
    clearance = float(finger.get("clearance_m") or 0.0)
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
            logger.info("grasping-flat-objects: could not read the gripper (%s)", exc)
            return -1.0

    def verdict(w: float, across: float) -> str:
        """ "" when the gap reads the slab between the pads (or cannot be read), else why not."""
        if w < 0.0:
            return ""
        if w <= at_closed + float(empty_margin):
            return f"closed on nothing ({w * 1e3:.1f} mm)"
        if w > across + float(full_margin):
            return (
                f"closed on something {w * 1e3:.1f} mm wide, over the slab's {across * 1e3:.1f} mm"
            )
        if w < across - float(narrow_margin):
            return (
                f"closed to {w * 1e3:.1f} mm, under the slab's {across * 1e3:.1f} mm: not across it"
                " (over its top or past a corner)"
            )
        return ""

    def set_width(w: float) -> str:
        try:
            return _refusal(ctx.tool("robot.set_grip", width_m=float(w), **on_arm))
        except Exception as exc:
            return str(exc) or type(exc).__name__

    def joints() -> np.ndarray:
        try:
            arms = ctx.tool("robot.get_observation").get("arms") or []
        except Exception as exc:
            logger.info("grasping-flat-objects: no joint reading (%s)", exc)
            return np.zeros(0)
        if int(arm_id) >= len(arms):
            return np.zeros(0)
        q = arms[int(arm_id)].get("joint_state")
        if isinstance(q, dict):
            q = q.get("positions")
        return np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)

    def square_jaws(
        target: dict[str, float],
        close: np.ndarray,
        approach: np.ndarray,
        at: np.ndarray,
        notes: list[str],
    ) -> dict[str, Any]:
        """Read the jaws' heading back and, when the arm left the wrist roll free,
        finish the turn on the last joint and re-centre over ``at``. Returns the
        hand's pose."""
        now = here()
        gain = -1.0 if approach[2] < 0 else 1.0
        for _ in range(TURN_TRIES):
            err = _wrap_half(jaw_error_deg(now["rotation"], target, close, approach))
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
            moved = math.radians(
                _wrap_half(jaw_error_deg(now["rotation"], before, close, approach))
            )
            out["wrist_turn_deg"] += math.degrees(moved)
            if abs(command) > math.radians(2.0) and 0.5 < abs(moved / command) < 1.5:
                gain = moved / command
            why = go("robot.go_to_pose", at, now["rotation"])
            if why:
                notes.append(f"re-centring after the wrist turn: {why}")
                break
            now = here()
        return now

    plans = [("short", *frame["short"])]
    if try_long_axis:
        plans.append(("long", *frame["long"]))
    failures: list[str] = []
    descended = False
    for name, unit, half in plans:
        across = 2.0 * half
        tilt = diagonal_tilt_deg(thickness, half, tip_reach) if auto_tilt else abs(float(tilt_deg))
        t = math.radians(tilt)
        if at_open > 0.0 and across / math.cos(t) + 2.0 * clearance >= at_open:
            failures.append(
                f"{name} axis: {across * 1e3:.0f} mm does not fit the {at_open * 1e3:.0f} mm open hand"
            )
            continue
        # Level: open fully (RoboRSI). Tilted: part way, so the lower fingertip is
        # near the support at the descent and has little to travel to the edge.
        wide = (
            at_open
            if (t == 0.0 or at_open <= 0.0)
            else min(at_open, across / math.cos(t) + 2.0 * OPEN_CLEARANCE)
        )
        geo = pinch_geometry(
            frame["support"],
            frame["top"],
            half,
            wide,
            t,
            tip_reach,
            grasp_fraction,
            min_tip_clearance,
        )
        if geo["bite"] < MIN_BITE:
            failures.append(
                f"{name} axis: too thin - the lowest fingertip can come only to "
                f"{(geo['tip_z'] - frame['support']) * 1e3:.1f} mm over the support at "
                f"{tilt:.0f} deg tilt, the slab is {thickness * 1e3:.1f} mm "
                "(tilt the hand, or slide the slab to an edge)"
            )
            continue
        u0 = np.array([unit[0], unit[1], 0.0])
        approach = np.array([0.0, 0.0, -math.cos(t)]) + u0 * math.sin(t)
        close = u0 * math.cos(t) + np.array([0.0, 0.0, math.sin(t)])

        def point_at(mid_z: float, approach: np.ndarray = approach) -> np.ndarray:
            return np.array([*frame["xy"], mid_z]) - tip_reach * approach

        start = point_at(geo["start_mid_z"])
        point = point_at(geo["mid_z"])
        hover = start + np.array([0.0, 0.0, float(hover_height)])
        heading = math.degrees(math.atan2(float(unit[1]), float(unit[0])))
        frame_args: dict[str, Any] = {"close_heading_deg": heading}
        if t != 0.0:
            frame_args["approach"] = _vec(approach)
        try:
            gf = ctx.tool("robot.grasp_frame", **frame_args, **on_arm)
        except Exception as exc:
            failures.append(f"{name} axis: no grasp frame ({exc})")
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
                    logger.info("grasping-flat-objects: IK preview refused (%s)", exc)
                    continue
                # A stall at the edge of reach reads like a fingertip on the slab.
                if isinstance(reply, dict) and (
                    reply.get("success") is False or reply.get("within_servo_tolerance") is False
                ):
                    continue
                reachable.append(rot)
            rotations = reachable
        if not rotations:
            failures.append(f"{name} axis: the grasp pose is out of reach (IK preview)")
            continue
        out["attempts"] += 1
        out["axis"] = name
        out["close_heading_deg"] = heading
        out["tilt_deg"] = tilt
        out["grasp_position"] = _vec(point)
        out["tip_height"] = float(geo["tip_z"] - frame["support"])
        out["object_width"] = across
        out["wrist_turn_deg"] = 0.0

        if t == 0.0 or at_open <= 0.0:
            try:
                ctx.tool("robot.open_gripper", **on_arm)
                why = ""
            except Exception as exc:
                why = str(exc) or type(exc).__name__
        else:
            why = set_width(wide)
        if why:
            failures.append(f"{name} axis: the hand did not open ({why})")
            continue
        now = _xyz(here())
        if now[2] < hover[2] - 1.0e-3:
            why = go(
                "robot.go_to_pose_cartesian",
                np.array([now[0], now[1], hover[2]]),
                here()["rotation"],
            )
            if why:
                failures.append(f"{name} axis: rise: {why}")
                continue
        why, rot = "", rotations[0]
        for rot in rotations:
            why = go("robot.go_to_pose", hover, rot)
            if not why:
                break
        if why:
            failures.append(f"{name} axis: hover: {why}")
            continue
        notes: list[str] = []
        pose = square_jaws(rot, close, approach, hover, notes)
        err = _wrap_half(jaw_error_deg(pose["rotation"], rot, close, approach))
        out["jaw_error_deg"] = err
        if abs(err) > float(yaw_tolerance_deg):
            failures.append(
                f"{name} axis: the jaws stayed {err:+.0f} deg off the slab's axis"
                + (f" ({'; '.join(notes)})" if notes else "")
            )
            continue

        descended = True
        why = move("robot.go_to_pose_cartesian", start, pose["rotation"])
        reached = _xyz(here())
        high = float(reached[2] - start[2])
        aside = float(np.linalg.norm(reached[:2] - start[:2]))
        if high > float(descent_tolerance) or aside > float(reach_tolerance):
            go("robot.go_to_pose_cartesian", np.array([*reached[:2], hover[2]]), pose["rotation"])
            failures.append(
                f"{name} axis: the descent stopped {high * 1e3:.1f} mm high and {aside * 1e3:.0f} mm aside"
                + (f" ({why})" if why else "")
                + "; a fingertip on the slab?"
            )
            continue

        if t != 0.0:
            # Narrow the jaws to the slab's width, following them down so the lower
            # fingertip slides along the support into the slab's bottom edge.
            edge_width = across / math.cos(t)
            follow_why = ""
            for w in np.linspace(wide, edge_width, FOLLOW_STEPS + 1)[1:]:
                follow_why = set_width(w)
                if follow_why:
                    break
                mid_z = max(
                    frame["support"] + TILT_TIP_CLEARANCE + w / 2.0 * math.sin(t),
                    geo["mid_z"],
                )
                follow_why = move("robot.go_to_pose_cartesian", point_at(mid_z), pose["rotation"])
                if follow_why and not follow_why.startswith(("timeout", "contact")):
                    break
                follow_why = ""
            reached = _xyz(here())
            if follow_why:
                ctx.tool("robot.open_gripper", **on_arm)
                go(
                    "robot.go_to_pose_cartesian",
                    np.array([*reached[:2], hover[2]]),
                    pose["rotation"],
                )
                failures.append(f"{name} axis: following the closing jaws down: {follow_why}")
                continue

        out["grip_width"] = width(ctx.tool("robot.close_gripper", **on_arm))
        why = verdict(out["grip_width"], across)
        if why:
            ctx.tool("robot.open_gripper", **on_arm)
            go("robot.go_to_pose_cartesian", np.array([*reached[:2], hover[2]]), pose["rotation"])
            failures.append(f"{name} axis: {why}")
            continue

        lift_note = ""
        if float(lift_height) > 0.0:
            lift_to = reached + np.array([0.0, 0.0, float(lift_height)])
            lift_why = move("robot.go_to_pose_cartesian", lift_to, pose["rotation"])
            out["lifted"] = float(_xyz(here())[2] - reached[2])
            if out["lifted"] < float(lift_height) - float(reach_tolerance):
                lift_note = (
                    f"; the lift rose {out['lifted'] * 1e3:.0f} of {float(lift_height) * 1e3:.0f} mm"
                    f" ({lift_why or 'stalled'}): the slab is heavy, fixed or caught"
                )
        out["held_width"] = width()
        why = verdict(out["held_width"], across) if float(lift_height) > 0.0 else ""
        if why:
            ctx.tool("robot.open_gripper", **on_arm)
            failures.append(f"{name} axis: the slab slipped out during the lift, {why}")
            continue
        unread = " (the jaw gap could not be read)" if out["grip_width"] < 0.0 else ""
        out["route"] = "grasped"
        out["detail"] = (
            f"{name} axis: closed at {out['grip_width'] * 1e3:.1f} mm across a "
            f"{across * 1e3:.1f} mm side, the lower fingertip planned "
            f"{out['tip_height'] * 1e3:.1f} mm over the support"
            + (f", tilted {tilt:.1f} deg" if t else "")
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
    out["detail"] = " | ".join(failures) or "no axis to try"
    return out
