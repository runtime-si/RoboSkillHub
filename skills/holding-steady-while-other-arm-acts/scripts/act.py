# Adapted from RoboTwin (https://github.com/robotwin-Platform/RoboTwin),
# envs/put_object_cabinet.py, MIT, Copyright (c) 2025 Tianxing Chen.
# Kept: while one arm holds, only the other arm is commanded (there it lifts the
# object and places it into the drawer the holder keeps open).
# Changed: the other arm's work is a plain waypoint move or a press (the last
# leg may end on contact) through robot.* tools with an explicit arm_id, in
# place of RoboTwin's place_actor; every arrival is read back, the press target
# is previewed with robot.solve_ik so a stall at the edge of reach is not read
# as contact, the hand backs out along its path, and the holder's pose is read
# (never commanded) before and after.
"""The other arm's step while one arm holds: a straight-line move, or a press that may end on contact."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

REACH_TOLERANCE = 0.02
"""How far a hand may end from where it was sent [m]: a servo's 4 mm ball plus
a real arm's sag, no tighter than the connectors' own bounds (PORTING lesson 3)."""

SAME_ARM = 0.001
"""Two arm_ids whose tool centres read this close are one arm [m]."""


class Output(TypedDict):
    route: str
    detail: str
    stopped_short: float
    contact: bool
    retreated: bool
    actor_end: dict[str, float]
    holder_shift: float


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _point(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v[k]) for k in ("x", "y", "z")])
    return np.asarray(v, dtype=float).reshape(3)


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


def run(
    ctx: NodeContext,
    hold: dict[str, Any],
    target: dict[str, float],
    start: dict[str, float] | None = None,
    via: list[Any] | None = None,
    mode: str = "move",
    hand: str = "as_is",
    rotation: dict[str, float] | None = None,
    close_heading_deg: float | None = None,
    retreat: bool = True,
    reach_tolerance: float = REACH_TOLERANCE,
) -> Output:
    if mode not in ("move", "press"):
        raise ValueError(f"mode must be 'move' or 'press', not {mode!r}")
    if hand not in ("as_is", "closed", "open"):
        raise ValueError(f"hand must be 'as_is', 'closed' or 'open', not {hand!r}")
    actor, holder = int(hold["other_arm"]), int(hold["arm"])
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "stopped_short": 0.0,
        "contact": False,
        "retreated": False,
        "actor_end": {},
        "holder_shift": 0.0,
    }

    def arm(a: int) -> dict[str, int]:
        # arm_id only when nonzero: a single-arm connector takes no such keyword.
        return {"arm_id": int(a)} if int(a) else {}

    on = arm(actor)

    def here(a: int = actor) -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **arm(a)))

    def move(tool: str, position: np.ndarray, rot: dict[str, float]) -> tuple[str, float]:
        """Send the actor; return why it did not arrive ("" if it did) and how far off it ended."""
        try:
            why = _refusal(ctx.tool(tool, pose={"position": _vec(position), "rotation": rot}, **on))
        except Exception as exc:  # a refused move is routed, not raised
            why = str(exc) or type(exc).__name__
        off = float(np.linalg.norm(_xyz(here()) - position))
        if off > float(reach_tolerance):
            return f"{why or 'ended off'}; stopped {off * 1e3:.0f} mm from its target", off
        return "", off

    notes: list[str] = []

    def finish(route: str, detail: str) -> Output:
        end = here()
        out["actor_end"] = dict(end["position"])
        out["holder_shift"] = float(np.linalg.norm(_xyz(here(holder)) - holder0))
        if out["holder_shift"] > float(reach_tolerance):
            notes.append(
                f"the holding hand reads {out['holder_shift'] * 1e3:.0f} mm from its hold pose"
            )
        out["route"], out["detail"] = route, "; ".join([d for d in (detail, *notes) if d])
        return out

    holder0 = _xyz(hold["pose"])
    if actor == holder:
        return finish("unreachable", f"the actor and the holder are both arm {actor}")
    mine = here()
    if float(np.linalg.norm(_xyz(mine) - _xyz(here(holder)))) < SAME_ARM:
        return finish("unreachable", f"the connector does not report a separate arm {actor}")
    # A free move to the start (where the hand is, when none is given), then
    # straight legs through the via points to the target.
    points = [
        _point(start) if start is not None else _xyz(mine),
        *(_point(p) for p in via or []),
        _point(target),
    ]

    if rotation is not None:
        rotations = [rotation]
    elif close_heading_deg is not None:
        try:
            frame = ctx.tool("robot.grasp_frame", close_heading_deg=float(close_heading_deg), **on)
        except Exception as exc:
            return finish("unreachable", f"no top-down rotation for the actor ({exc})")
        rotations = [
            frame[k]
            for k in ("rotation", "rotation_flipped")
            if isinstance(frame, dict) and frame.get(k)
        ]
    else:
        rotations = [mine["rotation"]]

    if hand == "closed":
        ctx.tool("robot.close_gripper", **on)
    elif hand == "open":
        ctx.tool("robot.open_gripper", **on)

    rot, why = None, "no rotation offered"
    for candidate in rotations:
        why, _ = move("robot.go_to_pose", points[0], candidate)
        if not why:
            rot = candidate
            break
    if rot is None:
        return finish("unreachable", f"actor to its first waypoint: {why}")

    reached = 0
    for i, point in enumerate(points[1:], start=1):
        last = i == len(points) - 1
        if last and mode == "press":
            # A stall at the edge of reach reads like contact (PORTING lesson 7).
            try:
                ik = ctx.tool(
                    "robot.solve_ik", pose={"position": _vec(point), "rotation": rot}, **on
                )
            except Exception as exc:
                logger.info("holding-steady: no IK preview (%s)", exc)
                ik = None
            if isinstance(ik, dict) and (
                ik.get("success") is False or ik.get("within_servo_tolerance") is False
            ):
                why = "the press target is out of reach (IK preview)"
                break
            why, off = move("robot.go_to_pose_cartesian", point, rot)
            out["stopped_short"] = off
            out["contact"] = bool(why)
            why = ""
        else:
            why, off = move("robot.go_to_pose_cartesian", point, rot)
            if last:
                out["stopped_short"] = off
        if why:
            why = f"waypoint {i}: {why}"
            break
        reached = i

    if retreat:
        # Back out the way it came: from where it stopped through the waypoints
        # behind it to the first, so the actor is clear before the holder lets go.
        behind = points[: reached + 1] if why else points[:reached]
        backed = ""
        for point in reversed(behind):
            backed, _ = move("robot.go_to_pose_cartesian", point, rot)
            if backed:
                notes.append(f"backing out: {backed}")
                break
        out["retreated"] = not backed
    if why:
        return finish("unreachable", why)
    if out["contact"]:
        notes.append(
            f"pressed: stopped {out['stopped_short'] * 1e3:.0f} mm short of the press target"
        )
    return finish("acted", "")
