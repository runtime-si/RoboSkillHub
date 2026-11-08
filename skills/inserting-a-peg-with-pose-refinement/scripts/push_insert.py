# Adapted from ManiSkill (https://github.com/haosulab/ManiSkill),
# mani_skill/examples/motionplanning/panda/solutions/peg_insertion_side.py, Apache-2.0.
# Changes: the straight push starts from the refined pre-insert pose (ManiSkill
# pushes to the unrefined insert pose plus 5 cm) and stops on contact
# (robot.move_cartesian_until_contact) in short steps; its depth is capped so the fingers stay
# out of the mouth; the result is judged from the hand read back times the
# registered tip-in-hand transform (ManiSkill reads the peg from the simulator);
# a push that stops short backs the hand off to the pre-insert pose.
"""Push a held, aligned peg straight along the hole's axis and judge how far its tip went in."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

INSERT_DEPTH = 0.05
"""How far past the mouth the tip is sent [m]: ManiSkill's final 5 cm push
(``insert_pose * Pose([0.05, 0, 0])``). It stops earlier on contact."""

MIN_DEPTH = 0.01
"""How far past the mouth the tip must get to count as inserted [m]: past a
rim's chamfer and the arm's contact-stop overshoot (a real YAM threading
station's 10 mm "through" test)."""

FINGER_CLEAR = 0.015
"""Least distance the tool centre keeps in front of the mouth [m]: closed
linear_4310 fingertips reach 10.2 mm beyond it, plus 5 mm."""

MOVED = 0.002
"""Least hand travel [m] for a refused push to count as having moved."""

PUSH_STEP = 0.005
"""Length of each stop-on-contact step of the push [m]. In a YAM station twin
one 60 mm push (25 control steps) wedged a 20 mm peg 8.5 mm into a 24 mm hole
(and once 21 mm into a 30 mm one), the peg sliding 10-18 mm up in the fingers;
in 5 mm steps it reached the 30 mm floor or within 1 mm of it, slipping
0.5-0.6 mm."""


class Output(TypedDict):
    route: str
    detail: str
    depth: float
    lateral_error: float
    tip_position: dict[str, float]


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _vec3(v: dict[str, Any]) -> np.ndarray:
    return np.array([float(v["x"]), float(v["y"]), float(v["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def matrix(pose: dict[str, Any]) -> np.ndarray:
    q = pose["rotation"]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(
        [float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])]
    ).as_matrix()
    out[:3, 3] = _vec3(pose["position"])
    return out


def pose_of(m: np.ndarray) -> dict[str, Any]:
    q = Rotation.from_matrix(m[:3, :3]).as_quat()
    return {
        "position": _vec(m[:3, 3]),
        "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
    }


def _refusal(reply: Any) -> str:
    """Why a push was refused, or "": arrival, contact, a stall and a timeout are
    all ways a push ends, read from where the hand is."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached", "contact", "stalled", "stall", "timeout"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    return ""


def run(
    ctx: NodeContext,
    hole_centre: dict[str, float],
    hole_axis: dict[str, float],
    peg_in_tcp: dict[str, Any],
    pre_insert_pose: dict[str, Any] | None = None,
    insert_depth: float = INSERT_DEPTH,
    min_depth: float = MIN_DEPTH,
    finger_clear: float = FINGER_CLEAR,
    push_step: float = PUSH_STEP,
    back_off: bool = True,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    centre = _vec3(hole_centre)
    axis = _vec3(hole_axis)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-9:
        raise ValueError("hole_axis must be a non-zero direction")
    axis = axis / norm
    peg = matrix(peg_in_tcp)
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "depth": 0.0,
        "lateral_error": 0.0,
        "tip_position": _vec(centre),
    }

    def hand() -> np.ndarray | None:
        try:
            return matrix(_pose(ctx.tool("robot.get_ee_pose", **on_arm)))
        except Exception as exc:
            logger.info("inserting-a-peg-with-pose-refinement: no pose reading (%s)", exc)
            return None

    def tip_depth(h: np.ndarray) -> tuple[float, float, np.ndarray]:
        tip = (h @ peg)[:3, 3]
        d = tip - centre
        along = float(d @ axis)
        return along, float(np.linalg.norm(d - along * axis)), tip

    start = hand()
    if start is None:
        out["detail"] = "the hand's pose could not be read"
        return out
    depth0, _, _ = tip_depth(start)
    # How far the tool centre sits behind the tip along the axis: the push may
    # take the tip no deeper than this less finger_clear.
    behind = float(peg[:3, 3] @ (start[:3, :3].T @ axis))
    reach = behind - float(finger_clear)
    depth = min(float(insert_depth), reach)
    note = ""
    if depth < float(insert_depth):
        note = (
            f"; the push was capped at {depth * 1e3:.0f} mm, where the fingers "
            f"would be {float(finger_clear) * 1e3:.0f} mm from the mouth"
        )
    if depth < float(min_depth):
        out["route"] = "stopped_short"
        out["detail"] = (
            f"the hand holds the peg {behind * 1e3:.0f} mm behind its tip: no room to push "
            f"{float(min_depth) * 1e3:.0f} mm in with the fingers {float(finger_clear) * 1e3:.0f} mm "
            "clear; hold it further back"
        )
        return out

    # From the refined command, not the read-back: the arm's steady landing
    # error that the refinement compensated is still there on the push.
    begin = matrix(pre_insert_pose) if pre_insert_pose else start.copy()
    travel = (depth - depth0) * axis
    finish = begin.copy()
    finish[:3, 3] += travel
    if check_reach:
        try:
            reply = ctx.tool("robot.solve_ik", pose=pose_of(finish), **on_arm)
        except Exception as exc:
            logger.info("inserting-a-peg-with-pose-refinement: no IK preview (%s)", exc)
            reply = None
        # A stall at the edge of reach looks exactly like a jam.
        if isinstance(reply, dict) and (
            reply.get("success") is False or reply.get("within_servo_tolerance") is False
        ):
            out["detail"] = "the end of the push is out of reach (IK preview)"
            return out

    # In steps, each stop-on-contact: one long push reaches a speed at which the
    # peg wedges on the hole's wall and slides up in the fingers. Each step's
    # target is the refined start plus its share of the travel, so a step that
    # lands short (inside a connector's arrival ball) does not shorten the next.
    n = max(1, math.ceil(float(np.linalg.norm(travel)) / max(float(push_step), 1.0e-4)))
    end = start
    for k in range(1, n + 1):
        command = begin.copy()
        command[:3, 3] += travel * k / n
        why, stopped = "", False
        try:
            reply = ctx.tool("robot.move_cartesian_until_contact", pose=pose_of(command), **on_arm)
            why = _refusal(reply)
            stopped = isinstance(reply, dict) and reply.get("outcome") in (
                "contact",
                "stall",
                "stalled",
            )
        except Exception as exc:  # a station's "did not reach its target" is a stop
            logger.info("inserting-a-peg-with-pose-refinement: push stopped (%s)", exc)
            stopped = True
        end = hand()
        if end is None:
            out["detail"] = "the hand's pose could not be read"
            return out
        if why and float(np.linalg.norm(end[:3, 3] - start[:3, 3])) < MOVED:
            out["detail"] = f"push: {why}"
            return out
        if why or stopped:
            break
    reached, lateral, tip = tip_depth(end)
    out.update(depth=reached, lateral_error=lateral, tip_position=_vec(tip))
    if reached >= float(min_depth):
        out["route"] = "inserted"
        out["detail"] = (
            f"the tip, as held, went {reached * 1e3:.1f} mm past the mouth "
            f"(sent to {depth * 1e3:.0f} mm), {lateral * 1e3:.1f} mm off the axis{note}"
        )
        return out
    out["route"] = "stopped_short"
    out["detail"] = (
        f"the push stopped with the tip, as held, {reached * 1e3:+.1f} mm past the mouth "
        f"(needed {float(min_depth) * 1e3:.0f} mm){note}"
    )
    if back_off:
        back = matrix(pre_insert_pose) if pre_insert_pose else start
        try:
            reason = _refusal(ctx.tool("robot.go_to_pose_cartesian", pose=pose_of(back), **on_arm))
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
        if reason:
            out["detail"] += f"; backing off failed ({reason})"
        else:
            out["detail"] += "; the hand backed off to the pre-insert pose"
    return out
