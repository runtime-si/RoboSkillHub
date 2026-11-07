# Adapted from MOKA (https://github.com/moka-manipulation/moka), MIT,
# Copyright (c) 2024 Kuan Fang, Fangchen Liu:
# moka/policies/franka_policy.py (transform_gripper_position, FrankaMarkPolicy
# phases: lift_open, reach_pre_grasp, reach_grasp, grip, lift,
# reach_pre_target, reach_target, reach_pre_motion, pre/post_contact, release).
# Changes: robot.* connector moves in place of Franka delta actions; the hand
# keeps its grasp rotation through the motion, so MOKA's grasp-to-current
# transform reduces to the measured hand-to-function offset (MOKA turned to
# yaw 0); a grasp read back from the jaw opening (MOKA timed the grip out);
# the closed gripper itself is the function point when nothing is grasped and
# runs the pre -> target -> post path (MOKA only came down on the target);
# tool-centre floor over the support; moves that refuse or end off are routed.
"""Execute a keypoint motion plan: grasp at the grasp point, carry the function point through its waypoints."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

REACH_TOLERANCE = 0.02
"""How far the hand may end from a commanded point [m]: a servo's 4 mm ball plus a
real arm's sag, no tighter than the connectors' own bounds (pushing-objects)."""

MIN_TOOL_HEIGHT = 0.015
"""Lowest tool-centre height over the support [m]. A real YAM station refuses a
tool centre under 14 mm over its calibrated table."""

GRASP_DEPTH = 0.01
"""The tool centre goes this far under the grasp keypoint's surface [m]; with the
fingertips ~10 mm below the tool centre they close 20 mm under the top."""

RELEASE_LIFT = 0.03
"""Rise before (empty hand) or after (holding) opening at the end [m] (MOKA's
release_z_offset)."""

EMPTY_FRACTION = 0.04
"""A jaw opening at or under this (robot.get_gripper, 1.0 = open) closed on
nothing: 3.8 mm on a 94 mm YAM span (handing-over-between-arms)."""


class Output(TypedDict):
    route: str
    detail: str
    phases: list[dict[str, Any]]
    hand_offset: dict[str, float]
    final_function_point: dict[str, float]
    held: bool


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _arr(v: Any) -> np.ndarray | None:
    return None if v is None else np.array([float(v["x"]), float(v["y"]), float(v["z"])])


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
    plan: dict[str, Any] | str,
    reach_tolerance: float = REACH_TOLERANCE,
    min_tool_height: float = MIN_TOOL_HEIGHT,
    grasp_depth: float = GRASP_DEPTH,
    release_lift: float = RELEASE_LIFT,
    empty_fraction: float = EMPTY_FRACTION,
    keep_rotation: bool = False,
) -> Output:
    if isinstance(plan, str):
        plan = json.loads(plan)
    mode = str(plan.get("mode", ""))
    grasp, function = _arr(plan.get("grasp")), _arr(plan.get("function"))
    target, pre, post = (
        _arr(plan.get("target")),
        _arr(plan.get("pre_contact")),
        _arr(plan.get("post_contact")),
    )
    support_z, safe_z = float(plan["support_z"]), float(plan["safe_z"])
    floor = support_z + float(min_tool_height)
    tol = float(reach_tolerance)
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "phases": [],
        "hand_offset": _vec(np.zeros(3)),
        "final_function_point": _vec(np.zeros(3)),
        "held": False,
    }

    def here() -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose"))

    def move(name: str, tool: str, position: np.ndarray, rotation: dict[str, float]) -> str:
        position = position.copy()
        position[2] = max(position[2], floor)
        try:
            why = _refusal(ctx.tool(tool, pose={"position": _vec(position), "rotation": rotation}))
        except Exception as exc:  # a refused or failed move is routed, not raised
            why = str(exc)
        off = float(np.linalg.norm(_xyz(here()) - position))
        reached = not why and off <= tol
        out["phases"].append(
            {
                "phase": name,
                "commanded": _vec(position),
                "reached": reached,
                "error_m": off,
                "why": why,
            }
        )
        if not reached and not why:
            why = f"the hand ended {off * 1e3:.0f} mm from it"
        return "" if reached else why

    start = here()
    own = start["rotation"]

    # The hand's rotation: top-down, closing across the object at the grasp or
    # across the push (both wrist solutions), then its own (pushing-objects).
    heading = plan.get("grasp_heading_deg") if grasp is not None else None
    if grasp is None and plan.get("push_heading_deg") is not None:
        heading = float(plan["push_heading_deg"]) + 90.0
    if grasp is not None and heading is None:
        heading = 0.0
    tops = []
    if heading is not None:
        try:
            frame = ctx.tool("robot.grasp_frame", close_heading_deg=float(heading))
            tops = [
                frame[k]
                for k in ("rotation", "rotation_flipped")
                if isinstance(frame, dict) and frame.get(k)
            ]
        except Exception as exc:
            logger.info("executing-keypoint-motion: no top-down rotation (%s)", exc)
    rotations = [own, *tops] if keep_rotation else [*tops, own]

    def first_reach(name: str, position: np.ndarray) -> tuple[dict[str, float] | None, str]:
        why = "no rotation to try"
        for rotation in rotations:
            why = move(name, "robot.go_to_pose", position, rotation)
            if not why:
                return rotation, ""
        return None, why

    def unreachable(why: str) -> Output:
        out["route"] = "unreachable"
        out["detail"] = why
        return out

    # MOKA lift_open: rise to the safe height where the hand stands.
    if _xyz(start)[2] < safe_z - 1.0e-3:
        rise = _xyz(start)
        rise[2] = safe_z
        why = move("lift_start", "robot.go_to_pose", rise, own)
        if why:
            return unreachable(f"lift_start: {why}")

    offset = np.zeros(3)
    rotation: dict[str, float] | None = None
    if grasp is not None:
        ctx.tool("robot.open_gripper")
        grasp_hand = np.array([grasp[0], grasp[1], grasp[2] - float(grasp_depth)])
        rotation, why = first_reach("reach_pre_grasp", np.array([grasp[0], grasp[1], safe_z]))
        if rotation is None:
            return unreachable(f"reach_pre_grasp: {why}")
        why = move("reach_grasp", "robot.go_to_pose_cartesian", grasp_hand, rotation)
        if why:
            move(
                "retreat",
                "robot.go_to_pose_cartesian",
                np.array([grasp[0], grasp[1], safe_z]),
                rotation,
            )
            return unreachable(f"reach_grasp: {why}")
        ctx.tool("robot.close_gripper")
        jaw = float(ctx.tool("robot.get_gripper").get("position", 1.0))
        hand_at_grasp = _xyz(here())
        if jaw <= float(empty_fraction):
            ctx.tool("robot.open_gripper")
            move(
                "retreat",
                "robot.go_to_pose_cartesian",
                np.array([grasp[0], grasp[1], safe_z]),
                rotation,
            )
            out["route"] = "grasp_failed"
            out["detail"] = (
                f"the jaws closed to {jaw:.3f} of open (at or under {empty_fraction}): nothing held"
            )
            return out
        out["held"] = True
        anchor = function if function is not None else grasp
        offset = hand_at_grasp - anchor
        why = move(
            "lift",
            "robot.go_to_pose_cartesian",
            np.array([*hand_at_grasp[:2], safe_z + offset[2]]),
            rotation,
        )
        if why:
            return unreachable(f"lift: {why}")
    else:
        ctx.tool("robot.close_gripper")
    out["hand_offset"] = _vec(offset)

    # The function point's path (MOKA's phase list).
    path: list[tuple[str, np.ndarray, bool]] = []
    if mode == "pick_and_place" and target is not None:
        path = [
            ("reach_pre_target", np.array([target[0], target[1], safe_z]), False),
            ("reach_target", target, True),
        ]
    else:
        lead = pre if pre is not None else target
        if lead is None:
            return unreachable("the plan has no waypoint or target to move to")
        path.append(("reach_pre_motion", np.array([lead[0], lead[1], safe_z]), False))
        if pre is not None:
            path.append(("reach_pre_contact", pre, True))
        if target is not None:
            path.append(("reach_target", target, True))
        if post is not None:
            path.append(("reach_post_contact", post, True))

    shortfall = ""
    for name, point, straight in path:
        hand = point + offset
        if rotation is None:
            rotation, why = first_reach(name, hand)
            if rotation is None:
                return unreachable(f"{name}: {why}")
            continue
        why = move(
            name, "robot.go_to_pose_cartesian" if straight else "robot.go_to_pose", hand, rotation
        )
        if why:
            if not straight:
                return unreachable(f"{name}: {why}")
            shortfall = f"{name}: {why}"
            break

    end = _xyz(here())
    out["final_function_point"] = _vec(end - offset)

    # MOKA release: let go and rise.
    lift = np.array([end[0], end[1], end[2] + float(release_lift)])
    if out["held"]:
        ctx.tool("robot.open_gripper")
        move("release_lift", "robot.go_to_pose_cartesian", lift, rotation or own)
    else:
        move("release_lift", "robot.go_to_pose_cartesian", lift, rotation or own)
        ctx.tool("robot.open_gripper")
    move(
        "retreat",
        "robot.go_to_pose",
        np.array([end[0], end[1], max(safe_z, lift[2])]),
        rotation or own,
    )

    if shortfall:
        return unreachable(shortfall + "; released and lifted")
    out["route"] = "done"
    travel = ""
    if target is not None and post is not None:
        d = (end - offset)[:2] - target[:2]
        travel = f", function point {math.hypot(*d) * 1e3:.0f} mm past the target"
    out["detail"] = f"{mode}: every phase reached{travel}; released and lifted"
    return out
