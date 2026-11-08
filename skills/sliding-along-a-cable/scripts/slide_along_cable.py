# Adapted from o2ac-ur (https://github.com/o2ac/o2ac-ur),
# catkin_ws/src/o2ac_routines/src/o2ac_routines/common.py,
# insert_motor_cables_without_tools_normal (cage the cable with
# gripper.send_command(0.005), pull the hand cable_straighten_distance along it
# with move_lin_rel at 0.03 m/s, then gripper.close()), MIT License,
# Copyright (c) 2020 Team O2AC.
# Changes: the path is the cable's perceived centreline toward a chosen end
# (curve.fit_centerline / curve.endpoints), not a fixed relative move along the
# tool's x; the slide is cut into short Cartesian steps that follow the
# centreline, with the jaws kept at their angle to the local tangent (the rest
# turned on the last joint when the arm's solver leaves the wrist roll free);
# the cage width is the jaws' own reading on the cable plus a margin (or the
# cable's diameter plus a clearance), not o2ac's absolute 5 mm; the slide stops
# at a planned distance before the end, or earlier when the hand stalls
# (tracking lag, no force sensor) or the jaws are pushed open, and every waypoint
# is previewed with robot.solve_ik first so a stall is not the edge of reach;
# the firm close is jaw-checked and the outcome routed.
"""Cage a cable between partly closed jaws, slide the hand along its centreline toward an end, stop there and close firmly."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

STOP_BEFORE_END = 0.04
"""Planned stop, along the centreline before the end's tip [m]. A charging
cable's plug is 33 mm of moulded boot and connector (measured on the real
station's cable, 2026-11-04): 4 cm from the tip is the cable just behind it.
o2ac's own stop is a planned distance too (cable_straighten_distance,
0.127 m, tuned 0.097-0.133 per cable)."""

CAGE_MARGIN = 0.0015
"""How much wider than the jaws' reading closed on the cable the cage is [m].
o2ac cages its motor cables at an absolute 5 mm; the jaws of a YAM station twin
read 5.4 mm closed on a 3.5 mm cable, so an absolute width would pinch there,
and a margin over the jaws' own reading is the same looseness on any
reading offset."""

CAGE_CLEARANCE = 0.0025
"""Cage over the cable's diameter when the hand arrives open [m]: 6 mm for a
3.5 mm cable, the candidate's 5-6 mm cage (o2ac's 5 mm on its motor cables)."""

HELD_WIDTH_SLACK = 0.004
"""A jaw reading under twice the diameter plus this is closed on the cable
[m] (the threading station's bound for a close that stopped on the cable)."""

EMPTY_MARGIN = 0.001
"""A jaw reading within this of the closed stop holds nothing [m]."""

STEP = 0.01
"""Length of one Cartesian step along the centreline [m]. o2ac makes one
straight move; short steps let the hand follow a curve, the jaws be read and a
stall be seen within a centimetre."""

SLIDE_LIFT = 0.001
"""How far the hand rises before sliding [m]: the fingertips stop scraping the
table and stay under a 3.5 mm cable's centre, so the table still closes the
cage's bottom (the candidate's note: tapered YAM fingertips let a cable out
under them)."""

MAX_OFFSET = 0.008
"""The hand more than this from the centreline in plan is not on the cable [m]
(a 6 mm cage holds the cable within 3 mm; the rest is the fit's error)."""

STALL_FRACTION = 0.5
"""A step that advanced less than this share of its length stalled: the hand
was stopped (the end's plug against the fingertips, a snag). A real YAM
station's arrival bound is 12 mm per move, so a stall is read from the progress
along the step, not from the distance to its target."""

WIDEN = 0.003
"""Jaws read this much wider than the cage [m]: something much thicker than
the cable has pushed between them. In a YAM station twin the jaws read up to
1.6 mm wider than a 7.0 mm cage while dragging a 3.5 mm cable round a 0.15 m
curve (1.5 mm stopped that slide 40 mm early), and a 6 mm plug in that cage
(about 5 mm between the fingertips, estimated from the 1.9 mm reading
offset) would open them only about 1 mm: this catches only a much thicker
end; the stall and the planned stop are the usual ones."""

END_WINDOW = 0.06
"""A stall this near the tip [m] is the end feature against the fingertips
(a 33 mm plug plus the margin of a centreline's length error); earlier it is a
snag."""

CLOSE_PRESS = 0.003
"""How far under the height the hand started at it is commanded while the jaws
shut firmly [m] (0: off): back down to the table and pressed on it, so the
tapered fingertips cannot ride up over the cable (the real threading
station's CLOSE_PRESS, about 1 N; in a YAM station twin a close 1 mm up lifted
the fingertips 1.5 mm and shut empty)."""

MIN_SLIDE = 0.002
"""A slide shorter than this is not made [m]."""

YAW_TOLERANCE_DEG = 4.0
"""How far the jaws may turn from their angle to the cable before the rest is
turned on the last joint [deg]. A 3.5 mm cable crossing a 5 mm gap at 8 deg,
across a fingertip about 10 mm wide along it (estimated), only just fits: in a
YAM station twin, with turns at 8 deg on a 0.15 m curve, the cable wedged the
jaws 1.6 mm open between turns."""

YAW_STOP_DEG = 10.0
"""Jaws further than this from their angle to the cable after the wrist turns
stop the slide [deg]: in a YAM station twin whose last joint ran out of range
on a curve the jaws drifted to 37 deg, wedged 2 mm open and dragged the cable
12 mm."""

TURN_TRIES = 2
"""Last-joint turns tried per step."""

ROLL_LIMITS = (-2.0944, 2.0944)
"""Last joint's range [rad] when none is stated: the i2rt YAM's joint 6."""

LIMIT_MARGIN = 0.05

RETREAT = 0.04
"""How far an empty hand rises after it opened [m], so a camera can see the
cable again (the grasp skill's lift)."""

REFUSALS = ("unreachable", "bad_speed_scale")
"""Routed outcomes that mean a move was refused, not stopped by contact."""


class Output(TypedDict):
    route: str
    detail: str
    stopped_by: str
    start_from_end: float
    end_from_end: float
    slid: float
    cage_width: float
    cage_reading: float
    jaw_width: float
    steps: int
    wrist_turns: int


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


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


def _yawed(q: dict[str, float], deg: float) -> dict[str, float]:
    """``q`` turned ``deg`` about the world's vertical."""
    h = math.radians(deg) / 2.0
    cw, sz = math.cos(h), math.sin(h)
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    return {
        "w": cw * w - sz * z,
        "x": cw * x - sz * y,
        "y": cw * y + sz * x,
        "z": cw * z + sz * w,
    }


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


def _heading(v: np.ndarray) -> float:
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


class Path:
    """A centreline ordered toward the end the hand slides to, by arc length."""

    def __init__(self, points: np.ndarray, extension: float) -> None:
        self.points = points
        self.arcs = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        )
        self.tip = float(self.arcs[-1]) + max(0.0, float(extension))

    def at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.arcs[-1]))
        return np.array([np.interp(s, self.arcs, self.points[:, i]) for i in range(3)])

    def tangent(self, s: float, span: float = 0.01) -> np.ndarray:
        """The horizontal unit way toward the end at ``s``, over ``span`` either side."""
        lo = max(0.0, s - span)
        hi = min(float(self.arcs[-1]), s + span)
        if hi - lo < 1.0e-6:
            lo, hi = max(0.0, hi - 2 * span), hi
        way = self.at(hi) - self.at(lo)
        way[2] = 0.0
        n = float(np.linalg.norm(way))
        return way / n if n > 1.0e-9 else np.array([1.0, 0.0, 0.0])

    def project(self, p: np.ndarray) -> tuple[float, float]:
        """Arc length of the point nearest ``p`` in plan, and the signed offset
        of ``p`` from it (positive to the left of the way toward the end)."""
        best = (float("inf"), 0.0, 0.0)
        for i in range(len(self.points) - 1):
            a, b = self.points[i, :2], self.points[i + 1, :2]
            d = b - a
            length2 = float(d @ d)
            t = 0.0 if length2 < 1.0e-12 else float(np.clip((p[:2] - a) @ d / length2, 0.0, 1.0))
            foot = a + t * d
            gap = float(np.linalg.norm(p[:2] - foot))
            if gap < best[0]:
                rel = p[:2] - foot
                side = float(d[0] * rel[1] - d[1] * rel[0])
                best = (gap, float(self.arcs[i] + t * math.sqrt(length2)), math.copysign(gap, side))
        return best[1], best[2]


def _ordered(centerline: Any, end: Any) -> tuple[np.ndarray, float, float | None]:
    """The centreline's points toward ``end``, how far the end's tip lies past
    its last node [m], and the centreline's radius if it states one."""
    pts = centerline.get("points") if isinstance(centerline, dict) else centerline
    points = np.asarray(pts if pts is not None else [], dtype=float).reshape(-1, 3)
    radius = None
    if isinstance(centerline, dict) and centerline.get("radius_m"):
        radius = float(centerline["radius_m"])
    if len(points) < 2:
        return points, 0.0, radius
    if end is None:
        raise ValueError("give end: the CurveEnd (curve.endpoints) the hand slides toward")
    if isinstance(end, dict) and "x" in end:
        end = [float(end["x"]), float(end["y"]), float(end.get("z", 0.0))]
    if isinstance(end, dict) and "node" in end:
        last = int(end["node"]) != 0
        tip = end.get("position")
    else:
        tip = end.get("position") if isinstance(end, dict) else end
        if tip is None:
            raise ValueError("end needs a node or a position")
        tip_xyz = np.asarray(tip, dtype=float).reshape(3)
        last = bool(np.linalg.norm(points[-1] - tip_xyz) <= np.linalg.norm(points[0] - tip_xyz))
    if not last:
        points = points[::-1].copy()
    extension = 0.0
    if tip is not None:
        tip_xyz = np.asarray(tip, dtype=float).reshape(3)
        extension = float(np.linalg.norm(tip_xyz[:2] - points[-1, :2]))
    return points, extension, radius


def run(
    ctx: NodeContext,
    centerline: dict[str, Any] | None = None,
    end: dict[str, Any] | None = None,
    stop_before_end: float = STOP_BEFORE_END,
    cable_radius: float | None = None,
    cage_width: float | None = None,
    cage_margin: float = CAGE_MARGIN,
    close_heading_deg: float | None = None,
    step: float = STEP,
    slide_lift: float = SLIDE_LIFT,
    press: float = CLOSE_PRESS,
    max_offset: float = MAX_OFFSET,
    end_window: float = END_WINDOW,
    check_reach: bool = True,
    roll_limits: list[float] | None = None,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "stopped_by": "",
        "start_from_end": -1.0,
        "end_from_end": -1.0,
        "slid": 0.0,
        "cage_width": -1.0,
        "cage_reading": -1.0,
        "jaw_width": -1.0,
        "steps": 0,
        "wrist_turns": 0,
    }
    if centerline is None:
        raise ValueError("give a centerline (curve.fit_centerline or curve.track_centerline)")
    points, extension, fitted_radius = _ordered(centerline, end)
    if len(points) < 2:
        out["detail"] = "the centreline has fewer than two points"
        return out
    path = Path(points, extension)
    radius = float(cable_radius or fitted_radius or 0.00175)

    try:
        hand = ctx.tool("robot.describe_gripper", **on_arm)
    except Exception as exc:
        logger.info("sliding-along-a-cable: no gripper description (%s)", exc)
        hand = {}
    fit = hand.get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or hand.get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)

    def width_now(reply: Any = None) -> float:
        try:
            fraction = reply.get("position") if isinstance(reply, dict) else None
            if fraction is None:
                fraction = ctx.tool("robot.get_gripper", **on_arm)["position"]
            if at_open <= 0.0:
                return -1.0
            return at_closed + max(0.0, min(1.0, float(fraction))) * (at_open - at_closed)
        except Exception as exc:  # an unreadable hand is not an empty one
            logger.info("sliding-along-a-cable: could not read the gripper (%s)", exc)
            return -1.0

    def move(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        try:
            return _refusal(
                ctx.tool(tool, pose={"position": _vec(position), "rotation": rot}, **on_arm)
            )
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc) or type(exc).__name__

    def here() -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **on_arm))

    start = here()
    tcp0 = _xyz(start)
    s0, offset = path.project(tcp0)
    out["start_from_end"] = float(path.tip - s0)
    if abs(offset) > float(max_offset):
        out["route"] = "lost_cable"
        out["detail"] = (
            f"the hand stands {abs(offset) * 1e3:.1f} mm from the cable's centreline in plan "
            f"(over {float(max_offset) * 1e3:.0f} mm): it is not on the cable; nothing moved"
        )
        return out

    # The jaws' closing direction, held at its angle to the local tangent.
    tangent0 = path.tangent(s0)
    close0 = (
        float(close_heading_deg) if close_heading_deg is not None else _heading(tangent0) + 90.0
    )
    across0 = np.array([math.cos(math.radians(close0)), math.sin(math.radians(close0)), 0.0])
    close_local = _matrix(start["rotation"]).T @ across0
    skew = _wrap180(close0 - _heading(tangent0))

    def jaw_heading(rot: dict[str, float]) -> float:
        c = _matrix(rot) @ close_local
        return _heading(c)

    def wanted_heading(s: float) -> float:
        return _heading(path.tangent(s)) + skew

    # Waypoints along the centreline, at the hand's own offset from it (the
    # hand holds the cable, so its offset is the fit's error there).
    s_stop = float(path.tip - float(stop_before_end))
    z = float(tcp0[2] + float(slide_lift))
    arcs: list[float] = []
    if s_stop > s0 + MIN_SLIDE:
        n = max(1, math.ceil((s_stop - s0) / float(step) - 1.0e-9))
        arcs = [s0 + (s_stop - s0) * (k + 1) / n for k in range(n)]

    def waypoint(s: float) -> np.ndarray:
        p = path.at(s)
        t = path.tangent(s)
        left = np.array([-t[1], t[0], 0.0])
        q = p + offset * left
        return np.array([q[0], q[1], z])

    def rotation_at(s: float, rot: dict[str, float]) -> dict[str, float]:
        return _yawed(rot, _wrap180(wanted_heading(s) - jaw_heading(rot)))

    # A stall at the edge of reach reads exactly like the end: preview it all.
    if check_reach:
        for s in arcs:
            try:
                reply = ctx.tool(
                    "robot.solve_ik",
                    pose={
                        "position": _vec(waypoint(s)),
                        "rotation": rotation_at(s, start["rotation"]),
                    },
                    **on_arm,
                )
            except Exception as exc:
                logger.info("sliding-along-a-cable: no IK preview (%s)", exc)
                break
            if isinstance(reply, dict) and (
                reply.get("success") is False or reply.get("within_servo_tolerance") is False
            ):
                out["detail"] = (
                    f"no arm solution {(path.tip - s) * 1e3:.0f} mm before the end "
                    f"({reply.get('reason') or 'IK preview failed'}); nothing moved"
                )
                return out

    lo, hi = (float(v) for v in (roll_limits or ROLL_LIMITS))
    gain = -1.0 if _matrix(start["rotation"])[2, 2] < 0 else 1.0

    def joints() -> np.ndarray | None:
        try:
            arms = ctx.tool("robot.get_observation").get("arms") or []
            q = arms[int(arm_id)].get("joint_state")
            return np.asarray(
                q.get("positions") if isinstance(q, dict) else q, dtype=float
            ).reshape(-1)
        except Exception as exc:
            logger.info("sliding-along-a-cable: no joint reading (%s)", exc)
            return None

    # The cable's turn along the path, against the last joint's room (a
    # top-down wrist's yaw is mostly that joint's, however the arm turns it).
    q0 = joints() if arcs else None
    if q0 is not None and len(q0):
        turned = [_wrap180(wanted_heading(s0) - jaw_heading(start["rotation"]))]
        before = wanted_heading(s0)
        for s in arcs:
            turned.append(turned[-1] + _wrap180(wanted_heading(s) - before))
            before = wanted_heading(s)
        slack = math.radians(YAW_TOLERANCE_DEG)
        ends = sorted(q0[-1] + math.radians(t) / gain for t in (min(turned), max(turned)))
        if ends[0] < lo + LIMIT_MARGIN - slack or ends[1] > hi - LIMIT_MARGIN + slack:
            out["detail"] = (
                f"the cable turns {min(turned):+.0f} to {max(turned):+.0f} deg along the path and the last "
                f"joint ({q0[-1]:+.2f} rad, range {lo:+.2f} to {hi:+.2f}) has no room to follow it: "
                "grasp it with the other wrist solution; nothing moved"
            )
            return out

    # The cage: just wider than the cable, from the jaws' own reading on it.
    reading0 = width_now()
    if cage_width is not None:
        cage = float(cage_width)
    elif 0.0 <= reading0 <= at_closed + EMPTY_MARGIN:
        out["route"] = "lost_cable"
        out["detail"] = (
            f"the jaws read {reading0 * 1e3:.1f} mm, at their closed stop: no cable between them; nothing moved"
        )
        return out
    elif 0.0 < reading0 <= 4.0 * radius + HELD_WIDTH_SLACK:
        cage = reading0 + float(cage_margin)
    else:
        cage = 2.0 * radius + CAGE_CLEARANCE
    out["cage_width"] = float(cage)
    notes: list[str] = []
    try:
        ctx.tool("robot.set_grip", width_m=float(cage), **on_arm)
    except Exception as exc:
        out["detail"] = (
            f"the jaws could not be set to a {cage * 1e3:.1f} mm cage ({exc}); nothing moved"
        )
        return out
    cage_reading = width_now()
    out["cage_reading"] = float(cage_reading)
    if cage_reading > cage + WIDEN:
        out["route"] = "lost_cable"
        out["detail"] = (
            f"the jaws stayed at {cage_reading * 1e3:.1f} mm, asked to cage at {cage * 1e3:.1f} mm: "
            "something holds them open (the cable under a fingertip); the hand did not move"
        )
        return out

    def turn(s: float) -> None:
        """Finish a yaw the arm's solver left undone on the last joint."""
        nonlocal gain
        for _ in range(TURN_TRIES):
            now = here()
            residual = _wrap180(wanted_heading(s) - jaw_heading(now["rotation"]))
            if abs(residual) <= YAW_TOLERANCE_DEG:
                return
            q = joints()
            if q is None or not len(q):
                notes.append("no joint reading to turn the wrist")
                return
            command = math.radians(residual) / gain
            if not lo + LIMIT_MARGIN <= q[-1] + command <= hi - LIMIT_MARGIN:
                notes.append(f"the last joint has no room to turn {residual:+.0f} deg")
                return
            target = q.copy()
            target[-1] += command
            try:
                why = _refusal(
                    ctx.tool(
                        "robot.move_to_joints",
                        joint_config={"positions": target.tolist()},
                        **on_arm,
                    )
                )
            except Exception as exc:
                why = str(exc) or type(exc).__name__
            if why:
                notes.append(f"wrist turn: {why}")
                return
            out["wrist_turns"] += 1
            moved = math.radians(
                _wrap180(jaw_heading(here()["rotation"]) - jaw_heading(now["rotation"]))
            )
            if abs(command) > math.radians(2.0) and 0.5 < abs(moved / command) < 1.5:
                gain = moved / command

    def finish(stopped_by: str, why: str) -> Output:
        """Close firmly where the hand stopped, and route."""
        out["stopped_by"] = stopped_by
        stop = here()
        s_end, _ = path.project(_xyz(stop))
        out["end_from_end"] = float(path.tip - s_end)
        out["slid"] = float(s_end - s0)
        if float(press) > 0.0:
            low = _xyz(stop)
            low[2] = float(tcp0[2]) - float(press)
            why_low = move("robot.go_to_pose_cartesian", low, stop["rotation"])
            if why_low and not why_low.startswith(("timeout", "contact")):
                notes.append(f"the press before the close was refused ({why_low})")
        width = width_now(ctx.tool("robot.close_gripper", **on_arm))
        if width > 4.0 * radius + HELD_WIDTH_SLACK:
            notes.append(f"the first close stopped at {width * 1e3:.1f} mm")
            width = width_now(ctx.tool("robot.close_gripper", **on_arm))
        out["jaw_width"] = float(width)
        note = ("; " + "; ".join(notes)) if notes else ""
        where_ = (
            f"the hand slid {out['slid'] * 1e3:.0f} mm along the perceived centreline, from "
            f"{out['start_from_end'] * 1e3:.0f} to {out['end_from_end'] * 1e3:.0f} mm before its end "
            f"(aim {float(stop_before_end) * 1e3:.0f} mm), caged at {cage * 1e3:.1f} mm"
        )
        if 0.0 <= width <= at_closed + EMPTY_MARGIN:
            ctx.tool("robot.open_gripper", **on_arm)
            up = _xyz(here())
            up[2] += RETREAT
            move("robot.go_to_pose_cartesian", up, stop["rotation"])
            out["route"] = "lost_cable"
            out["detail"] = (
                f"{why}; the firm close shut to {width * 1e3:.1f} mm, at the closed stop: the cable left "
                f"the cage; the hand opened and rose {RETREAT * 1e3:.0f} mm; {where_}{note}"
            )
            return out
        if stopped_by == "refused" or (
            stopped_by == "stall" and out["end_from_end"] > float(end_window)
        ):
            out["route"] = "unreachable"
        else:
            out["route"] = "reached_end"
        out["detail"] = (
            f"{why}; closed firmly at {width * 1e3:.1f} mm; {where_}. What the hand did: the cable "
            f"may have been dragged, re-perceive before relying on where it is{note}"
        )
        return out

    if not arcs:
        return finish("already_there", "the hand already stood at the planned stop")

    why = move("robot.go_to_pose_cartesian", np.array([tcp0[0], tcp0[1], z]), start["rotation"])
    if why.startswith(REFUSALS):
        return finish("refused", f"the {float(slide_lift) * 1e3:.0f} mm rise was refused ({why})")

    for s in arcs:
        now = here()
        prev = _xyz(now)
        target = waypoint(s)
        why = move("robot.go_to_pose_cartesian", target, rotation_at(s, now["rotation"]))
        out["steps"] += 1
        after = _xyz(here())
        if why and not why.startswith(("timeout", "contact")):
            return finish(
                "refused",
                f"the step to {(path.tip - s) * 1e3:.0f} mm before the end was refused ({why})",
            )
        way = target[:2] - prev[:2]
        length = float(np.linalg.norm(way))
        advance = float((after[:2] - prev[:2]) @ way / length) if length > 1.0e-9 else 0.0
        if why.startswith("contact") or (length > MIN_SLIDE and advance < STALL_FRACTION * length):
            gone = path.tip - path.project(after)[0]
            return finish(
                "stall",
                f"the hand stalled {gone * 1e3:.0f} mm before the end, advancing "
                f"{advance * 1e3:.1f} of {length * 1e3:.1f} mm"
                + (
                    " (read as the end feature against the fingertips)"
                    if gone <= float(end_window)
                    else " (a snag, or the arm held back, not the end)"
                ),
            )
        width = width_now()
        if cage_reading >= 0.0 and width > max(cage, cage_reading) + WIDEN:
            return finish(
                "jaws",
                f"the jaws were pushed open to {width * 1e3:.1f} mm (caged at {cage_reading * 1e3:.1f} mm): "
                "something thicker than the cable came between them",
            )
        turn(s)
        off = _wrap180(wanted_heading(s) - jaw_heading(here()["rotation"]))
        if abs(off) > YAW_STOP_DEG:
            return finish(
                "refused",
                f"the jaws stand {off:+.0f} deg off their angle to the cable "
                f"{(path.tip - s) * 1e3:.0f} mm before the end and the wrist could not turn them back",
            )
    return finish("planned_stop", "the hand reached the planned stop")
