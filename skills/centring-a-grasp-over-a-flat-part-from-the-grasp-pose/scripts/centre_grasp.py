"""Bring a planned grasp onto the middle of a flat part, square to it, by looking again from the grasp's own pose."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrist_depth as wd  # noqa: E402

logger = logging.getLogger(__name__)

STANDOFF = 0.05
"""The hand looks from this far above the grasp pose, in the grasp's own orientation [m]. yam_dominos lifted the
leaning grasp 5 cm and the top-down one 7 cm; at both a 49 x 24.5 mm tile was whole in the D405's picture past the
open linear_4310 fingers (11 real picks, 2026-11-20: seen on every first look)."""

TOLERANCE = 0.0015
"""A look that has the part's middle within this of where the grasp is aimed ends the centring [m]. Half the 3 mm
pose tolerance of the YAM station's moves at the time; a 24.5 mm tile in jaws opened 16 mm wider has 8 mm a side."""

ACCEPT = 0.003
"""After the last correction, a look within this still counts as centred [m]. On the real left YAM the residual
after one correction was 0.7-2.9 mm and after two 0.4-1.8 mm (11 picks)."""

SQUARE_DEG = 2.5
"""A part turned less than this against the grasp is left as it is [deg]; depth outlines of a 49 mm tile gave its
turn to about a degree."""

MAX_TURN_DEG = 12.0
"""A turn larger than this is not made here [deg]: the grasp was planned for another way round, and a wrist near
its limits (the YAM's last joint) may not have the pose. The exit says so."""

MAX_LOOKS = 3
"""Looks at most: two corrections and a last look. The first look found the real hand 6-16 mm from the part's
middle (the look the grasp was planned from was taken at another pose of the arm, and the arm is not where its
joints say by the same amount at every pose); the second, 0.7-2.9 mm."""

NEAR = 0.03
"""A part further than this from where the grasp is aimed is another part [m]."""

SETTLE_READS = 5
SETTLE_DISTANCE = 0.0005
"""Two observations agreeing on the tool within this [m] before a frame is used."""


class Output(TypedDict):
    route: str
    grasp_pose_centred: dict[str, Any]
    pregrasp_pose: dict[str, Any]
    offset: dict[str, float]
    turn_deg: float
    residual_mm: float
    looks: int
    detail: str


def long_side_yaw(obb: dict[str, Any]) -> float:
    """The world yaw of the box's longest side [rad]."""
    r = wd.rot(obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
    axis = r[:, int(np.argmax([float(obb["extent"][k]) for k in ("x", "y", "z")]))]
    return math.atan2(axis[1], axis[0])


def half_turn(angle: float) -> float:
    """An angle between two undirected lines, in (-90, 90] deg as radians."""
    return (angle + math.pi / 2) % math.pi - math.pi / 2


def correction(
    part: dict[str, Any], aim_xy: np.ndarray, aim_yaw: float
) -> tuple[np.ndarray, float]:
    """How far the part's middle is from where the grasp is aimed [x, y, m] and how far it is turned against the
    way the grasp was planned for [rad, either way round]."""
    return np.asarray(part["centre"], float) - aim_xy, half_turn(float(part["yaw"]) - aim_yaw)


def moved(pose: np.ndarray, shift_xy: np.ndarray, turn: float, about_xy: np.ndarray) -> np.ndarray:
    """A tool pose shifted in the plane and turned about the vertical through ``about_xy`` (after the shift)."""
    c, s = math.cos(turn), math.sin(turn)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = pose.copy()
    out[:2, 3] += shift_xy
    centre = np.array([about_xy[0], about_xy[1], 0.0])
    out[:3, 3] = centre + rz @ (out[:3, 3] - centre)
    out[:3, :3] = rz @ out[:3, :3]
    return out


def run(
    ctx: NodeContext,
    target_obb: dict[str, Any],
    grasp_pose: dict[str, Any],
    standoff: float = STANDOFF,
    camera: str | None = None,
    arm_id: int = 0,
    support_z: float = 0.0,
    tolerance: float = TOLERANCE,
    accept: float = ACCEPT,
    max_turn_deg: float = MAX_TURN_DEG,
    max_looks: int = MAX_LOOKS,
    check_reach: bool = True,
) -> Output:
    """``target_obb``: the part as the plan has it (centre, extent, orientation), world frame. ``grasp_pose``: the
    tool pose the jaws are to close at, planned for that box. The hand is left at the corrected pregrasp pose."""
    arm = {"arm_id": int(arm_id)} if arm_id else {}
    size = sorted((2.0 * float(target_obb["extent"][k]) for k in ("x", "y", "z")), reverse=True)
    aim_xy = np.array([float(target_obb["center"]["x"]), float(target_obb["center"]["y"])])
    aim_yaw = long_side_yaw(target_obb)
    grasp = wd.matrix(wd.pose_of(grasp_pose))
    total, turned, residual, looks, notes = np.zeros(2), 0.0, float("nan"), 0, []

    def pre(g: np.ndarray) -> np.ndarray:
        p = g.copy()
        p[2, 3] += float(standoff)
        return p

    def out(route: str, detail: str) -> Output:
        return {
            "route": route,
            "grasp_pose_centred": wd.pose_dict(grasp),
            "pregrasp_pose": wd.pose_dict(pre(grasp)),
            "offset": wd.vec([total[0], total[1], 0.0]),
            "turn_deg": round(math.degrees(turned), 2),
            "residual_mm": round(1000.0 * residual, 2) if math.isfinite(residual) else -1.0,
            "looks": looks,
            "detail": "; ".join([detail, *notes]).strip("; "),
        }

    def go(pose: np.ndarray) -> str:
        goal = wd.pose_dict(pose)
        try:
            if check_reach:
                reply = ctx.tool("robot.solve_ik", pose=goal, **arm)
                if isinstance(reply, dict) and (
                    reply.get("success") is False or reply.get("within_servo_tolerance") is False
                ):
                    return "out of reach (IK preview)"
            return wd.refusal(ctx.tool("robot.go_to_pose", pose=goal, **arm))
        except Exception as exc:  # a connector that raises its refusals
            return str(exc) or type(exc).__name__

    why = go(pre(grasp))
    if why:
        return out("unreachable", f"the pregrasp pose: {why}")
    for look in range(1, int(max_looks) + 1):
        obs = ctx.tool("robot.get_observation")
        for _ in range(SETTLE_READS):
            nxt = ctx.tool("robot.get_observation")
            still = (
                np.linalg.norm(
                    wd.xyz(obs["arms"][arm_id]["ee_pose"]) - wd.xyz(nxt["arms"][arm_id]["ee_pose"])
                )
                <= SETTLE_DISTANCE
            )
            obs = nxt
            if still:
                break
        tool = wd.matrix(obs["arms"][arm_id]["ee_pose"])
        cam = wd.wrist_camera(obs, tool, camera)
        if cam is None:
            return out("not_seen", "no wrist camera with depth in the observation")
        looks = look
        try:
            found = wd.flat_parts(
                cam["depth"], cam["intrinsics"], wd.matrix(cam["pose"]), size, support_z
            )
        except ValueError as exc:
            return out("not_seen", str(exc))
        near = [
            p
            for p in found["parts"]
            if np.hypot(p["centre"][0] - aim_xy[0], p["centre"][1] - aim_xy[1]) < NEAR
        ]
        if not near:
            why_not = [r["why"] for r in found["rejected"]][:3]
            # a part never seen from here leaves the plan as it was; one lost after a correction keeps the correction
            return out(
                "not_seen",
                f"look {look}: the part not made out whole from the grasp's pose ({why_not})",
            )
        part = min(
            near, key=lambda p: np.hypot(p["centre"][0] - aim_xy[0], p["centre"][1] - aim_xy[1])
        )
        d_xy, d_yaw = correction(part, aim_xy, aim_yaw)
        residual = float(np.linalg.norm(d_xy))
        square = abs(d_yaw) < math.radians(SQUARE_DEG)
        if residual < float(tolerance) and square:
            return out(
                "centred", f"look {look}: {1000 * residual:.1f} mm, {math.degrees(d_yaw):+.1f} deg"
            )
        if look == int(max_looks):
            break
        turn = d_yaw if not square and abs(d_yaw) <= math.radians(float(max_turn_deg)) else 0.0
        if not square and turn == 0.0:
            notes.append(
                f"turned {math.degrees(d_yaw):+.1f} deg against the grasp: more than is squared here"
            )
        new_xy = aim_xy + d_xy
        new_grasp = moved(grasp, d_xy, turn, new_xy)
        why = go(pre(new_grasp))
        if why and turn:  # the turned pose is not to be had: the shift alone
            notes.append(f"not squared ({why})")
            turn, new_grasp = 0.0, moved(grasp, d_xy, 0.0, new_xy)
            why = go(pre(new_grasp))
        if why:
            return out("unreachable", f"the corrected pregrasp pose: {why}")
        grasp, aim_xy, aim_yaw = new_grasp, new_xy, aim_yaw + turn
        total, turned = total + d_xy, turned + turn
    if residual <= float(accept):
        return out("centred", f"after {looks} looks: {1000 * residual:.1f} mm")
    return out(
        "off_centre",
        f"after {looks} looks the part is still {1000 * residual:.1f} mm from the grasp",
    )
