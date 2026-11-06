# Adapted from RoboRSI (https://github.com/nssmd/RoboRSI),
# roborsi/embodied/skills/base/push_object/libero/policy.py, Apache-2.0.
# Changes: world-frame OBB and goal in place of image pixels; moves through the
# connector's robot.* tools; a vertical descent behind the object before the
# push; the contact height floored over the support; fingers across the push
# first (RoboRSI kept the hand's rotation first); routed move refusals.
"""Push a loose object across its support toward a goal, closed hand, never grasping it."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

MIN_PUSH = 0.03
"""Shortest push worth making [m]; a goal nearer the object's centre is refused."""

MAX_DISTANCE = 0.20
"""Longest push [m], RoboRSI's default bound."""

STANDOFF = 0.06
"""Free space behind the object's back face before the hand comes down [m]."""

MAX_HOVER_BACKOFF = 0.09
"""Cap on the hover's distance behind the object's centre, so a deep object does
not push the hover out of reach; the standoff shrinks to fit (never under
MIN_STANDOFF). RoboRSI's ``_MAX_SOURCE_TO_HOVER_BACKOFF``."""

MIN_STANDOFF = 0.02

HOVER_CLEARANCE = 0.08
"""Height of the hover over the contact point [m]."""

CONTACT_FRACTION = 0.5
"""Where up the object the tool centre pushes, as a fraction of its height."""

MIN_CONTACT_HEIGHT = 0.015
"""Lowest tool-centre height over the support [m]. A real YAM station refuses a
tool centre under 14 mm over its calibrated table."""

EDGE_PADDING = 0.003
"""Gap between the hand and the object's back face at contact [m] (RoboRSI)."""

REACH_TOLERANCE = 0.02
"""How far the hand may end from the hover or the lowered pose [m]."""

REQUIRED_PUSH = 0.08
"""A push counts once the hand has travelled min(this, 0.75 x the asked push)
past the contact point (RoboRSI's ``_MAX_REQUIRED_PUSH_DISTANCE`` rule)."""


class Output(TypedDict):
    route: str
    detail: str
    push_distance: float
    achieved: float
    expected_center: dict[str, float]


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


def _back_extent(obb: dict[str, Any], direction: np.ndarray) -> float:
    """How far the box reaches behind its centre along the push direction (xy)."""
    q = obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    half = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    projected = (corners @ rot.T)[:, :2] @ direction
    return max(0.0, -float(projected.min()))


def run(
    ctx: NodeContext,
    target_obb: dict[str, Any],
    goal_position: dict[str, float],
    max_distance: float = MAX_DISTANCE,
    standoff: float = STANDOFF,
    hover_clearance: float = HOVER_CLEARANCE,
    contact_fraction: float = CONTACT_FRACTION,
    min_contact_height: float = MIN_CONTACT_HEIGHT,
    reach_tolerance: float = REACH_TOLERANCE,
    keep_rotation: bool = False,
) -> Output:
    centre = np.array([float(target_obb["center"][k]) for k in ("x", "y", "z")])
    goal = np.array([float(goal_position[k]) for k in ("x", "y")])
    planar = goal - centre[:2]
    span = float(np.linalg.norm(planar))
    out: Output = {
        "route": "not_pushed",
        "detail": "",
        "push_distance": 0.0,
        "achieved": 0.0,
        "expected_center": _vec(centre),
    }
    if span < MIN_PUSH:
        out["detail"] = (
            f"the goal lies {span * 1e3:.0f} mm from the object's centre (under {MIN_PUSH * 1e3:.0f} mm)"
        )
        return out
    direction = planar / span
    push = min(span, float(max_distance))
    out["push_distance"] = push

    height = 2.0 * float(target_obb["extent"]["z"])
    bottom = centre[2] - height / 2.0
    backoff = _back_extent(target_obb, direction) + EDGE_PADDING
    contact = np.array(
        [
            *(centre[:2] - direction * backoff),
            bottom + max(contact_fraction * height, min_contact_height),
        ]
    )
    stand = min(float(standoff), max(MIN_STANDOFF, MAX_HOVER_BACKOFF - backoff))
    lowered = contact - np.array([*direction, 0.0]) * stand
    hover = lowered + np.array([0.0, 0.0, float(hover_clearance)])
    finish = contact + np.array([*direction, 0.0]) * push

    def unreachable(why: str) -> Output:
        out["route"] = "unreachable"
        out["detail"] = why
        return out

    def move(tool: str, position: np.ndarray, rotation: dict[str, float]) -> str:
        try:
            return _refusal(ctx.tool(tool, pose={"position": _vec(position), "rotation": rotation}))
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc)

    start = _pose(ctx.tool("robot.get_ee_pose"))
    ctx.tool("robot.close_gripper")

    # The hover rotations, in order: top-down with the closed fingers side by
    # side across the push (both wrist solutions), a broad face that keeps the
    # object from turning off the line; then the hand's own rotation. With
    # keep_rotation the hand's own comes first.
    across = []
    try:
        heading = math.degrees(math.atan2(direction[1], direction[0])) + 90.0
        frame = ctx.tool("robot.grasp_frame", close_heading_deg=heading)
        across = [
            frame[k]
            for k in ("rotation", "rotation_flipped")
            if isinstance(frame, dict) and frame.get(k)
        ]
    except Exception as exc:
        logger.info("pushing-objects: no top-down rotation (%s)", exc)
    own = [start["rotation"]]
    rotations = own + across if keep_rotation else across + own
    rotation, why = None, "no rotation reached the hover"
    for candidate in rotations:
        if float(start["position"]["z"]) < hover[2] - 1.0e-3:
            rise = _xyz(start)
            rise[2] = hover[2]
            why = move("robot.go_to_pose", rise, candidate)
        else:
            why = ""
        why = why or move("robot.go_to_pose", hover, candidate)
        off = float(np.linalg.norm(_xyz(_pose(ctx.tool("robot.get_ee_pose"))) - hover))
        if not why and off <= float(reach_tolerance):
            rotation = candidate
            break
        why = why or f"the hand ended {off * 1e3:.0f} mm from the hover"
    if rotation is None:
        return unreachable(f"hover: {why}")

    why = move("robot.go_to_pose_cartesian", lowered, rotation)
    reached = _xyz(_pose(ctx.tool("robot.get_ee_pose")))
    off = float(np.linalg.norm(reached - lowered))
    if off > float(reach_tolerance):
        return unreachable(
            f"descent: {why or 'ended off'}; the hand stopped {off * 1e3:.0f} mm from the lowered pose"
        )
    if why:
        # Stopped short but close, typically closed fingertips on the support:
        # push at the height the hand reached rather than into the table.
        logger.info("pushing-objects: descent stopped %.1f mm short (%s)", off * 1e3, why)
        contact[2] = finish[2] = reached[2]

    pushed_why = move("robot.go_to_pose_cartesian", finish, rotation)
    end = _xyz(_pose(ctx.tool("robot.get_ee_pose")))
    achieved = max(0.0, float(np.dot(end[:2] - contact[:2], direction)))
    out["achieved"] = achieved
    out["expected_center"] = _vec(centre + np.array([*direction, 0.0]) * achieved)

    retract_why = move(
        "robot.go_to_pose_cartesian", end + np.array([0.0, 0.0, float(hover_clearance)]), rotation
    )
    ctx.tool("robot.open_gripper")

    required = min(REQUIRED_PUSH, 0.75 * push)
    if achieved >= required and not retract_why:
        out["route"] = "pushed"
        return out
    parts = []
    if achieved < required:
        parts.append(
            f"the hand travelled {achieved * 1e3:.0f} mm past contact (needed {required * 1e3:.0f})"
        )
    if pushed_why:
        parts.append(f"push: {pushed_why}")
    if retract_why:
        parts.append(f"retract: {retract_why}")
    out["detail"] = "; ".join(parts)
    return out
