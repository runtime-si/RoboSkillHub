# Adapted from Ravens (https://github.com/google-research/ravens),
# module ravens.tasks.primitives (PickPlace.__call__, the placing half), Apache-2.0,
# Copyright 2024 The Ravens Authors.
# Changes: Ravens' loop of 1 mm movep steps checked against the suction cup's
# contact sensor is two robot.move_cartesian_until_contact moves (the arm's own
# stall detection; the YAM has no contact or force sensor), one to the top of the
# contact window and one through it, judged by the hand's measured stop; the
# descent is bounded (Ravens' runs until contact) and floored over the support; the place
# point is the held object's, translated to the hand by the object's measured
# offset in the hand; contact far from the expected height is routed rather than
# released; the approach, retreat and the robot.solve_ik preview are routed
# instead of returning Ravens' timeout flag.
"""Place a held object by coming down above its place point until it meets the support, then release."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

APPROACH_HEIGHT = 0.08
"""Height of the approach over the expected contact [m]. Ravens' preplace is
0.32 m over the place pose for a UR5 with a suction cup; a YAM's reach is
shorter, and 0.08 m clears the 40 mm walls of a small box with a 30 mm part
hanging under the hand (pushing-objects' hover clearance)."""

OVERTRAVEL = 0.02
"""How far below the expected contact the descent is sent [m]; it stops earlier
on contact. Ravens' descent has no lower bound. Deeper than the object's slide up
the jaws before the arm stalls (up to 12 mm measured in a YAM station twin), so a
hand that reaches the bottom met nothing."""

CONTACT_TOLERANCE = 0.012
"""How far above the expected height a stop still counts as the support [m]:
about two standard deviations of a calibrated YAM station's camera error
(RMS 5.4 mm over the workspace). Below it there is no bound but the bottom: the
measured stop reads low by however far the object slid up in the jaws."""

MIN_TOOL_HEIGHT = 0.015
"""Lowest tool-centre height over the support [m]. A real YAM station refuses a
tool centre under 14 mm over its calibrated table."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from the approach or the retreat [m]: no tighter than
a real YAM station's own 12 mm per-move bound (a servo arrives within 4 mm)."""

MIN_DESCENT = 0.002
"""A refused contact move that descended less than this did not move at all [m]."""

REFUSALS = ("unreachable", "bad_speed_scale")
"""Routed outcomes that mean a move was refused, not stopped by contact."""

CONTACT_MOVES = 3
"""Contact moves sent while one times out still descending (a servo's move has a
fixed step budget)."""


class Output(TypedDict):
    route: str
    detail: str
    released: bool
    expected_contact_z: float
    contact_z: float
    contact_offset: float
    placed_position: dict[str, float]


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


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


def _bottom_z(obb: dict[str, Any]) -> float:
    """The lowest corner of an oriented box."""
    q = obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    row_z = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
    half = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    return float(obb["center"]["z"]) - float(np.abs(row_z) @ half)


def run(
    ctx: NodeContext,
    place_position: dict[str, float],
    held_obb: dict[str, Any] | None = None,
    support_z: float | None = None,
    table_z: float | None = None,
    approach_height: float = APPROACH_HEIGHT,
    overtravel: float = OVERTRAVEL,
    contact_tolerance: float = CONTACT_TOLERANCE,
    min_tool_height: float = MIN_TOOL_HEIGHT,
    retreat_height: float | None = None,
    reach_tolerance: float = REACH_TOLERANCE,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    place = np.array([float(place_position[k]) for k in ("x", "y", "z")])
    start = _pose(ctx.tool("robot.get_ee_pose", **on_arm))
    rotation = start["rotation"]
    tcp = _xyz(start)

    # Where the hand must be for the object to be where it is asked: the object's
    # offset in the hand, measured now, carried down unchanged (the hand keeps its
    # rotation). Without a held box the place point is the tool centre's.
    if held_obb is not None:
        centre = np.array([float(held_obb["center"][k]) for k in ("x", "y", "z")])
        offset = np.array([*(centre[:2] - tcp[:2]), _bottom_z(held_obb) - tcp[2]])
        if support_z is None:
            support_z = float(place[2])
    else:
        offset = np.zeros(3)
    target = place - offset
    expected = float(target[2])
    under = table_z if table_z is not None else support_z
    floor = None if under is None else float(under) + float(min_tool_height)
    bottom_z = expected - float(overtravel)
    if floor is not None:
        bottom_z = max(bottom_z, floor)
    approach = np.array([target[0], target[1], expected + float(approach_height)])
    bottom = np.array([target[0], target[1], bottom_z])
    lift = float(approach_height if retreat_height is None else retreat_height)

    out: Output = {
        "route": "blocked",
        "detail": "",
        "released": False,
        "expected_contact_z": expected,
        "contact_z": float(tcp[2]),
        "contact_offset": 0.0,
        "placed_position": _vec(place),
    }

    def move(tool: str, position: np.ndarray) -> str:
        try:
            reply = ctx.tool(
                tool, pose={"position": _vec(position), "rotation": rotation}, **on_arm
            )
            return _refusal(reply)
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc)

    def where() -> np.ndarray:
        return _xyz(_pose(ctx.tool("robot.get_ee_pose", **on_arm)))

    def go(tool: str, position: np.ndarray) -> str:
        """A free move judged by where the hand is: "" when it arrived within
        reach_tolerance (a servo's timeout that ended that close has arrived),
        otherwise why not."""
        why = move(tool, position)
        off = float(np.linalg.norm(where() - position))
        if (why and not why.startswith("timeout")) or off > float(reach_tolerance):
            return f"{why or 'ended off'}; the hand is {off * 1e3:.0f} mm from it"
        return ""

    def blocked(why: str) -> Output:
        out["route"], out["detail"] = "blocked", why
        return out

    if check_reach:
        try:
            reply = ctx.tool(
                "robot.solve_ik",
                pose={
                    "position": _vec(np.array([*target[:2], max(expected, bottom_z)])),
                    "rotation": rotation,
                },
                **on_arm,
            )
        except Exception as exc:
            logger.info("placing-with-contact-descent: no IK preview (%s)", exc)
            reply = None
        # A stall at the edge of reach reads exactly like contact.
        if isinstance(reply, dict) and (
            reply.get("success") is False or reply.get("within_servo_tolerance") is False
        ):
            return blocked("the hand cannot reach the expected contact pose (IK preview)")

    # Approach: straight up first if the hand is under the approach height (it
    # holds something), then over to the approach.
    if tcp[2] < approach[2] - 1.0e-3:
        why = go("robot.go_to_pose_cartesian", np.array([tcp[0], tcp[1], approach[2]]))
        if why:
            return blocked(f"rise: {why}")
    why = go("robot.go_to_pose", approach)
    if why:
        return blocked(f"approach: {why}")

    def descend(start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, str, bool]:
        """One contact move (repeated while it times out still descending): where
        the hand stopped, why, and whether it stopped short of the goal. A stop is
        the move ending early (a routed "contact", a station's "did not reach its
        target" error) with the hand still above the goal; a move that finished
        met nothing on the way."""
        stop, why = start, ""
        for _ in range(CONTACT_MOVES):
            before = stop
            why = move("robot.move_cartesian_until_contact", goal)
            stop = where()
            if not why.startswith("timeout") or before[2] - stop[2] < MIN_DESCENT:
                break
        return stop, why, bool(why) and float(stop[2] - goal[2]) > MIN_DESCENT

    tol = float(contact_tolerance)

    def record(stop: np.ndarray) -> float:
        d = float(stop[2] - expected)
        out["contact_z"], out["contact_offset"] = float(stop[2]), d
        out["placed_position"] = _vec(np.array([*(stop[:2] + offset[:2]), stop[2] + offset[2]]))
        return d

    def back_up(stop: np.ndarray, route: str, why: str) -> Output:
        """Not released: lift the object back to the approach height."""
        out["route"] = route
        up = go("robot.go_to_pose_cartesian", np.array([stop[0], stop[1], approach[2]]))
        out["detail"] = why + (f"; lifting back: {up}" if up else "; lifted back, still holding")
        return out

    # The descent until contact, in two moves. Ravens steps down 1 mm at a time
    # so its cup never presses far past the contact; a servo aimed far below the
    # contact presses until the object slides up in the jaws (measured in a YAM
    # station twin: 3-4 mm when aimed 8 mm below, 11-12 mm when aimed 16-28 mm
    # below). So the first move only goes to the top of the contact window, and
    # a stop there is an early contact read before the hand has pressed far.
    window_top = np.array([target[0], target[1], max(expected + tol, bottom_z)])
    stop, why, stopped = descend(approach, window_top)
    if why.startswith(REFUSALS) or (
        why and float(approach[2] - stop[2]) < MIN_DESCENT and not why.startswith("contact")
    ):
        return blocked(f"descent: {why}")
    if stopped:
        d = record(stop)
        return back_up(
            stop,
            "contact_too_early",
            f"stopped {d * 1e3:.0f} mm above the expected contact (tolerance {tol * 1e3:.0f} mm)",
        )
    stop, why, stopped = descend(stop, bottom)
    d = record(stop)
    if why.startswith(REFUSALS):
        return blocked(f"descent through the contact window: {why}; still holding")
    # A friction grip lets the object slide up the jaws until the servo's push
    # falls under the grip, about where a servo calls a move arrived, so near
    # the floor a pressed object and a free hand both just finish the move. But
    # a hand at the floor over the table whose object's bottom would be at or
    # under the table has the object resting on something.
    pressed = (
        not stopped
        and table_z is not None
        and held_obb is not None
        and float(stop[2] + offset[2]) <= float(table_z)
    )
    if not stopped and not pressed:
        return back_up(
            stop,
            "no_contact",
            f"reached the bottom of the descent, {-d * 1e3:.0f} mm below the expected contact, "
            f"without stopping ({why or 'the move finished'})",
        )

    ctx.tool("robot.open_gripper", **on_arm)
    out["released"] = True
    up = np.array([stop[0], stop[1], stop[2] + lift])
    why = go("robot.go_to_pose_cartesian", up)
    if why:
        return blocked(f"released, but the retreat: {why}")
    out["route"] = "placed"
    how = (
        "stopped on contact"
        if stopped
        else "ended at the tool-centre floor with the object's bottom at or under the table"
    )
    out["detail"] = f"{how}, {d * 1e3:+.1f} mm from the expected height; released"
    if d < -tol:
        out["detail"] += (
            " (low: the support is lower than expected, or the object slid in the jaws)"
        )
    return out
