# Adapted from RoboTwin (https://github.com/robotwin-Platform/RoboTwin),
# envs/put_object_cabinet.py, MIT, Copyright (c) 2025 Tianxing Chen.
# Kept: one arm grasps the thing to be held (there the drawer bar, with a
# 0.05 m pre-grasp stand-off) and is then left alone while the other arm works;
# it is commanded again only after the other arm is done.
# Changed: the simulator actor pose is replaced by a world-frame OBB of the
# object; a top-down grasp (or a closed-hand press on the object's top) through
# robot.* tools with an explicit arm_id instead of RoboTwin's grasp_actor; the
# hold is judged from the jaw opening (grip) or from where the hand stopped
# (press), and the hand's read-back pose and jaw opening are recorded so
# release.py can check that nothing moved; refused moves are routed, not raised.
"""Hold an object still with one arm: a top-down grasp, or a closed hand pressed on its top."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PRE_GRASP = 0.05
"""Stand-off above the hold pose before the straight-down move [m]: RoboTwin
put_object_cabinet grasps the held fixture with pre_grasp_dis=0.05."""

GRASP_HEIGHT = 0.0
"""Tool centre above the object's centre when gripping [m]: the jaws close at
mid-height (RoboTwin grasp_dis 0)."""

FINGER_REACH = 0.010
"""How far the closed fingertips reach below the tool centre [m] (linear_4310:
10.2 mm; PORTING lesson 6)."""

FLOOR_CLEARANCE = 0.003
"""Fingertips kept this far above the object's bottom face when gripping [m]."""

PRESS_DEPTH = 0.010
"""How far below the point where the closed fingertips meet the object's top
the press is commanded [m]: a position-controlled arm pushes in proportion to
the shortfall. Not from the source (RoboTwin grasps); measured in the YAM twin."""

GRIP_SETTLE_STEPS = 300
"""Control steps close_gripper waits before the jaw opening is read [steps]
(gap's default is 60). tuned_on: YAM station twin, 30 mm cube: the jaws read
0.445 at the default 60, 0.420 at 150 and 0.336 (closed on the cube) at 270;
a hold recorded at 60 compared a still-closing hand with a settled one."""

REACH_TOLERANCE = 0.02
"""How far a hand may end from where it was sent [m]: a servo's 4 mm ball plus
a real arm's sag, no tighter than the connectors' own bounds (PORTING lesson 3)."""

EMPTY_FRACTION = 0.04
"""A jaw opening at or under this (robot.get_gripper, 1.0 = open) closed on
nothing: 3.8 mm on a 94 mm YAM span, the stop plus verifying-grasps' 3 mm margin."""

SAME_ARM = 0.001
"""Two arm_ids whose tool centres read this close are one arm [m]."""


class Output(TypedDict):
    route: str
    detail: str
    holder_grip: float
    hold: dict[str, Any]


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


def _rotation(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _box(obb: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, rotation (columns = box axes in world) and half-extents."""
    centre = np.array([float(obb["center"][k]) for k in ("x", "y", "z")])
    rot = _rotation(obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
    half = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    return centre, rot, half


def _narrow_heading(rot: np.ndarray, half: np.ndarray) -> float:
    """Heading [deg] of the box's narrower horizontal side: the jaws close across it."""
    tilt = np.abs(rot[2, :])
    horizontal = [int(i) for i in np.argsort(tilt)[:2]]
    axis = rot[:, min(horizontal, key=lambda i: half[i])]
    return math.degrees(math.atan2(axis[1], axis[0]))


def run(
    ctx: NodeContext,
    object_obb: dict[str, Any],
    holder_arm: int = 0,
    other_arm: int = 1,
    mode: str = "grip",
    close_heading_deg: float | None = None,
    grasp_height: float = GRASP_HEIGHT,
    press_depth: float = PRESS_DEPTH,
    pre_grasp: float = PRE_GRASP,
    reach_tolerance: float = REACH_TOLERANCE,
    empty_fraction: float = EMPTY_FRACTION,
    grip_settle_steps: int = GRIP_SETTLE_STEPS,
) -> Output:
    if mode not in ("grip", "press"):
        raise ValueError(f"mode must be 'grip' or 'press', not {mode!r}")
    out: Output = {"route": "unreachable", "detail": "", "holder_grip": -1.0, "hold": {}}

    def arm(a: int) -> dict[str, int]:
        # arm_id only when nonzero: a single-arm connector takes no such keyword.
        return {"arm_id": int(a)} if int(a) else {}

    on = arm(holder_arm)

    def here(a: int = holder_arm) -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **arm(a)))

    def move(tool: str, position: np.ndarray, rotation: dict[str, float]) -> tuple[str, float]:
        """Send the holder; return why it did not arrive ("" if it did) and how far off it ended."""
        try:
            why = _refusal(
                ctx.tool(tool, pose={"position": _vec(position), "rotation": rotation}, **on)
            )
        except Exception as exc:  # a refused move is routed, not raised
            why = str(exc) or type(exc).__name__
        off = float(np.linalg.norm(_xyz(here()) - position))
        if off > float(reach_tolerance):
            return f"{why or 'ended off'}; stopped {off * 1e3:.0f} mm from its target", off
        return "", off

    def grip() -> float:
        try:
            return float(ctx.tool("robot.get_gripper", **on)["position"])
        except Exception as exc:  # an unreadable hand is not an empty one
            logger.warning("holding-steady: arm %d gripper unreadable (%s)", holder_arm, exc)
            return -1.0

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        return out

    if int(holder_arm) == int(other_arm):
        return finish("unreachable", f"the holder and the other arm are both arm {int(holder_arm)}")
    mine, other = here(), here(other_arm)
    q = other.get("rotation") or {}
    if sum(float(q.get(k, 0.0)) ** 2 for k in "wxyz") < 0.25 or (
        float(np.linalg.norm(_xyz(mine) - _xyz(other))) < SAME_ARM
    ):
        # A connector that clamps or ignores arm_id reports the holder (or zeros) for
        # the other arm; the other arm's step would then move the holding hand.
        return finish(
            "unreachable", f"the connector does not report a separate arm {int(other_arm)}"
        )

    centre, rot, half = _box(object_obb)
    up = np.array([0.0, 0.0, 1.0])
    reach_up = float(np.abs(rot.T @ up) @ half)
    top, bottom = centre[2] + reach_up, centre[2] - reach_up
    heading = (
        float(close_heading_deg) if close_heading_deg is not None else _narrow_heading(rot, half)
    )
    if mode == "grip":
        z = max(centre[2] + float(grasp_height), bottom + FINGER_REACH + FLOOR_CLEARANCE)
    else:
        z = top + FINGER_REACH - float(press_depth)
    target = np.array([centre[0], centre[1], z])
    pre = target + up * float(pre_grasp)
    if mode == "press":
        pre[2] = max(pre[2], top + FINGER_REACH + float(pre_grasp))

    try:
        frame = ctx.tool("robot.grasp_frame", close_heading_deg=heading, **on)
        rotations = [
            frame[k]
            for k in ("rotation", "rotation_flipped")
            if isinstance(frame, dict) and frame.get(k)
        ]
    except Exception as exc:
        return finish("unreachable", f"no top-down rotation for heading {heading:.0f} deg ({exc})")

    if mode == "grip":
        ctx.tool("robot.open_gripper", **on)
    else:
        # A press is made with the closed hand: open jaws would straddle a small object.
        ctx.tool("robot.close_gripper", **on)
    rotation, why = None, "no rotation offered"
    for candidate in rotations:
        why, _ = move("robot.go_to_pose", pre, candidate)
        if not why:
            rotation = candidate
            break
    if rotation is None:
        return finish("unreachable", f"holder above the object: {why}")

    def back_off(route: str, detail: str) -> Output:
        if mode == "grip":
            ctx.tool("robot.open_gripper", **on)
        backed, _ = move("robot.go_to_pose_cartesian", pre, rotation)
        return finish(route, detail + (f"; backing off: {backed}" if backed else ""))

    if mode == "grip":
        why, _ = move("robot.go_to_pose_cartesian", target, rotation)
        if why:
            return back_off("unreachable", f"holder down onto the object: {why}")
        settle = {"settle_steps": int(grip_settle_steps)} if int(grip_settle_steps) > 0 else {}
        ctx.tool("robot.close_gripper", **settle, **on)
        out["holder_grip"] = g = grip()
        if 0.0 <= g <= float(empty_fraction):
            return back_off(
                "hold_lost", f"the holder's jaws closed to the stop ({g:.3f}): it missed"
            )
    else:
        # A stall at the edge of reach reads like a hand on the object (PORTING lesson 7).
        try:
            ik = ctx.tool(
                "robot.solve_ik", pose={"position": _vec(target), "rotation": rotation}, **on
            )
        except Exception as exc:
            logger.info("holding-steady: no IK preview (%s)", exc)
            ik = None
        if isinstance(ik, dict) and (
            ik.get("success") is False or ik.get("within_servo_tolerance") is False
        ):
            return back_off("unreachable", "the press pose is out of reach (IK preview)")
        move("robot.go_to_pose_cartesian", target, rotation)
        stop = _xyz(here())
        short = float(stop[2] - target[2])
        flat = float(np.linalg.norm(stop[:2] - target[:2]))
        if short < 0.5 * float(press_depth):
            return back_off(
                "hold_lost",
                f"the hand went down to {short * 1e3:.0f} mm above the press target: nothing under it",
            )
        if short > float(press_depth) + float(reach_tolerance) or flat > float(reach_tolerance):
            return back_off(
                "unreachable",
                f"the hand stopped {short * 1e3:.0f} mm above the press target and {flat * 1e3:.0f} mm"
                " off it: something else is in the way",
            )
        out["holder_grip"] = grip()

    now = here()
    out["hold"] = {
        "arm": int(holder_arm),
        "other_arm": int(other_arm),
        "mode": mode,
        "pose": {"position": now["position"], "rotation": now["rotation"]},
        "grip": out["holder_grip"],
        "retreat": _vec(pre),
        "object_center": _vec(centre),
        "close_heading_deg": heading,
    }
    return finish("holding", "")
