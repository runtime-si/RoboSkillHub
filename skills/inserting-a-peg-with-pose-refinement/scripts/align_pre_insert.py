# Adapted from ManiSkill (https://github.com/haosulab/ManiSkill),
# mani_skill/examples/motionplanning/panda/solutions/peg_insertion_side.py, Apache-2.0.
# Changes: the peg's pose in the hand and the hole's pose come from inputs
# (registering-held-objects, perceiving-functional-features) instead of the
# simulator's actors; the observed peg is the hand read back times the
# registered tip-in-hand transform; the pre-insert pose is set by the tip's
# standoff from the hole's mouth and only the peg's axis is aligned (its roll
# about the hole's axis is kept); the three refinements stop early inside a
# tolerance, the accumulated correction is capped, and the residual after the
# last one is judged (misaligned) instead of pushing regardless; IK preview and
# routed move refusals; moves through the connector's robot.* tools.
"""Carry a held peg to a pre-insert pose on the hole's axis and refine it from the observed residual."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

STANDOFF = 0.03
"""How far in front of the hole's mouth the peg's tip waits on the axis [m].
ManiSkill backs off 10 mm plus the peg's half-length (95-135 mm for its pegs)
from the pose with the tip at the hole; the peg's length is not an input here,
and a shorter straight push accumulates less line-tracking error."""

REFINEMENTS = 3
"""Most correction moves after the first arrival, ManiSkill's ``range(3)``."""

LATERAL_TOLERANCE = 0.002
"""Largest distance of the tip from the hole's axis that counts as aligned [m]:
the twin's fine-move ball (1.5 mm) plus margin. Set it at or under the peg's
radial clearance in the hole."""

AXIAL_TOLERANCE = 0.005
"""Largest error of the tip's standoff along the axis [m]; the push covers it."""

ANGLE_TOLERANCE_DEG = 3.0
"""Largest angle between the peg's axis and the hole's [deg]: the servo's own
rotation ball (3 deg) in a YAM station twin; no tighter is reachable there."""

FINE_TOLERANCE = 0.0015
"""Arrival ball asked of each refinement move [m] (``tolerance`` of
robot.go_to_pose). A connector's default ball (4 mm in the twin, 12 mm on a real
YAM station) swallows a correction smaller than itself: the move reports arrival
without moving. 1.5 mm is the twin servo's documented "fine leg"."""

MAX_CORRECTION = 0.02
"""Largest accumulated shift of the commanded pre-insert pose [m]. A residual
that keeps growing means the measurement, not the arm, is off."""

MAX_CORRECTION_DEG = 10.0
"""Largest accumulated turn of the commanded pre-insert pose [deg]."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from the commanded pre-insert pose [m] before the
carry counts as refused: a real YAM station's 12 mm per-move bound plus margin."""


class Output(TypedDict):
    route: str
    detail: str
    pre_insert_pose: dict[str, Any]
    tcp_at_mouth: dict[str, float]
    lateral_error: float
    axial_error: float
    angle_error_deg: float
    refinements: int
    correction: float


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
    """Why a move's reply says it did not arrive, or "" (connectors that route a
    refusal instead of raising)."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def aligned_target(tip: np.ndarray, centre: np.ndarray, axis: np.ndarray, standoff: float):
    """The tip frame wanted at the pre-insert pose: its z turned onto ``axis`` by
    the smallest rotation (the roll about the axis stays as held), its origin
    ``standoff`` in front of the mouth."""
    turn, _ = Rotation.align_vectors([axis], [tip[:3, 2]])
    target = np.eye(4)
    target[:3, :3] = turn.as_matrix() @ tip[:3, :3]
    target[:3, 3] = centre - float(standoff) * axis
    return target


def residual(tip: np.ndarray, target: np.ndarray, axis: np.ndarray) -> tuple[float, float, float]:
    """(distance of the tip from the hole's axis, its error along it, angle of the
    peg's axis from the hole's in degrees)."""
    d = tip[:3, 3] - target[:3, 3]
    along = float(d @ axis)
    lateral = float(np.linalg.norm(d - along * axis))
    cos = float(np.clip(tip[:3, 2] @ axis, -1.0, 1.0))
    return lateral, along, math.degrees(math.acos(cos))


def run(
    ctx: NodeContext,
    hole_centre: dict[str, float],
    hole_axis: dict[str, float],
    peg_in_tcp: dict[str, Any],
    standoff: float = STANDOFF,
    refinements: int = REFINEMENTS,
    lateral_tolerance: float = LATERAL_TOLERANCE,
    axial_tolerance: float = AXIAL_TOLERANCE,
    angle_tolerance_deg: float = ANGLE_TOLERANCE_DEG,
    fine_tolerance: float = FINE_TOLERANCE,
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
        "pre_insert_pose": {},
        "tcp_at_mouth": _vec(centre),
        "lateral_error": 0.0,
        "axial_error": 0.0,
        "angle_error_deg": 0.0,
        "refinements": 0,
        "correction": 0.0,
    }

    def hand() -> np.ndarray | None:
        try:
            return matrix(_pose(ctx.tool("robot.get_ee_pose", **on_arm)))
        except Exception as exc:
            logger.info("inserting-a-peg-with-pose-refinement: no pose reading (%s)", exc)
            return None

    def move(command: np.ndarray, fine: bool) -> str:
        knobs = dict(on_arm)
        if fine and fine_tolerance:
            knobs["tolerance"] = float(fine_tolerance)
        try:
            return _refusal(ctx.tool("robot.go_to_pose", pose=pose_of(command), **knobs))
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc) or type(exc).__name__

    def out_of_reach(command: np.ndarray) -> bool:
        try:
            reply = ctx.tool("robot.solve_ik", pose=pose_of(command), **on_arm)
        except Exception as exc:
            logger.info("inserting-a-peg-with-pose-refinement: no IK preview (%s)", exc)
            return False
        if not isinstance(reply, dict):
            return False
        return reply.get("success") is False or reply.get("within_servo_tolerance") is False

    start = hand()
    if start is None:
        out["detail"] = "the hand's pose could not be read"
        return out
    # ManiSkill: insert_pose = goal * peg_init^-1 * grasp; pre_insert = insert * offset.
    # Here the goal is the tip on the axis, the peg's pose the hand times peg_in_tcp.
    target = aligned_target(start @ peg, centre, axis, standoff)
    command = target @ np.linalg.inv(peg)
    initial = command.copy()
    out["tcp_at_mouth"] = _vec(command[:3, 3] + float(standoff) * axis)
    if check_reach and out_of_reach(command):
        out["detail"] = "the pre-insert pose is out of reach (IK preview)"
        return out

    why = move(command, fine=False)
    now = hand()
    if now is None:
        out["detail"] = "the hand's pose could not be read"
        return out
    off = float(np.linalg.norm(now[:3, 3] - command[:3, 3]))
    if why or off > REACH_TOLERANCE:
        out["detail"] = (
            f"carry to the pre-insert pose: {why or f'the hand ended {off * 1e3:.0f} mm away'}"
        )
        return out

    lateral, along, angle = residual(now @ peg, target, axis)
    for k in range(int(refinements)):
        if (
            lateral <= lateral_tolerance
            and abs(along) <= axial_tolerance
            and angle <= angle_tolerance_deg
        ):
            break
        # ManiSkill: delta = goal * offset * peg.pose^-1; pre_insert = delta * pre_insert.
        tip = now @ peg
        target = aligned_target(tip, centre, axis, standoff)
        command = target @ np.linalg.inv(tip) @ command
        shift = float(np.linalg.norm(command[:3, 3] - initial[:3, 3]))
        turn = math.degrees(Rotation.from_matrix(command[:3, :3] @ initial[:3, :3].T).magnitude())
        if shift > MAX_CORRECTION or turn > MAX_CORRECTION_DEG:
            out.update(route="misaligned", lateral_error=lateral, axial_error=along)
            out.update(angle_error_deg=angle, refinements=k, correction=shift)
            out["pre_insert_pose"] = pose_of(command)
            out["detail"] = (
                f"the correction grew to {shift * 1e3:.1f} mm / {turn:.1f} deg "
                f"(cap {MAX_CORRECTION * 1e3:.0f} mm / {MAX_CORRECTION_DEG:.0f} deg): "
                "re-register the peg or re-perceive the hole"
            )
            return out
        why = move(command, fine=True)
        out["refinements"] = k + 1
        if why:
            out["detail"] = f"refinement {k + 1}: {why}"
            out["pre_insert_pose"] = pose_of(command)
            return out
        now = hand()
        if now is None:
            out["detail"] = "the hand's pose could not be read"
            return out
        lateral, along, angle = residual(now @ peg, target, axis)
        logger.info(
            "inserting-a-peg-with-pose-refinement: refinement %d lateral %.1f mm axial %.1f mm angle %.1f deg",
            k + 1,
            lateral * 1e3,
            along * 1e3,
            angle,
        )

    out.update(lateral_error=lateral, axial_error=along, angle_error_deg=angle)
    out["correction"] = float(np.linalg.norm(command[:3, 3] - initial[:3, 3]))
    out["pre_insert_pose"] = pose_of(command)
    out["tcp_at_mouth"] = _vec(now[:3, 3] + (float(standoff) - along) * axis)
    within = (
        lateral <= lateral_tolerance
        and abs(along) <= axial_tolerance
        and angle <= angle_tolerance_deg
    )
    out["route"] = "aligned" if within else "misaligned"
    out["detail"] = (
        f"tip {lateral * 1e3:.1f} mm off the axis, {along * 1e3:+.1f} mm along it, "
        f"peg {angle:.1f} deg off it after {out['refinements']} refinement(s)"
        + ("" if within else " (outside tolerance)")
    )
    return out
