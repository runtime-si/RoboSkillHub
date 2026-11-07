# The drag is adapted from Ravens (https://github.com/google-research/ravens),
# module ravens.tasks.primitives (push), Apache-2.0, Copyright 2024 The Ravens
# Authors. Changes: a world-frame slab OBB and support edge in place of Ravens'
# two SE(3) push poses, the drag length being the slide that makes the slab
# overhang the edge by a chosen amount; robot.* connector moves, routed rather
# than returning Ravens' timeout flag; the 1 cm steps advance one step each
# (Ravens' loop sends every step to its end point, ``n_push * 0.01``) and stop
# when the hand stops advancing; the hand comes down behind the slab ("push",
# accepting a stop on the support) or onto its top until contact and then
# 1-2 mm deeper ("press"), in place of Ravens' fixed 5 mm height; the closed
# fingers stand across the drag, in place of Ravens' spatula yaw, turned on the
# last joint when the arm's solver leaves the wrist roll free. The
# overhang-then-pinch strategy is reimplemented from Eppner et al.,
# "Exploitation of environmental constraints in human and robotic grasping"
# (IJRR 2015) and Hang et al., "Pre-grasp sliding manipulation of thin objects
# using soft, compliant, or underactuated hands" (RA-L 2019); neither ships code.
"""Slide a flat object across its support until part of it overhangs the edge, closed hand, never grasping it."""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slide_edge_common as C  # noqa: E402

logger = logging.getLogger(__name__)

OVERHANG_FRACTION = 0.4
"""Overhang asked for, as a fraction of the slab's depth across the edge: its
centre stays a tenth of its depth on the support (the papers slide until "a
part" overhangs; 0.4 x 60 mm = 24 mm leaves 6 mm)."""

COM_INSET = 0.005
"""Least distance the slab's centre stays on the support side of the edge [m]."""

PAD_HALF_WIDTH = 0.010
"""Half a fingertip blade's width across the edge [m], the blades standing this
far ahead of the tool centre when they push and past it when they pinch. An
estimate for the linear_4310 blade, which robot.describe_gripper does not
state; without it a YAM station twin pushed a 60 mm slab off the edge (see
SKILL.md)."""

EDGE_CLEARANCE = 0.003
"""Gap between a blade and the support's edge during the pinch [m]."""

EDGE_PADDING = 0.003
"""Push mode: gap between the blades and the slab's back face at the contact
point [m] (pushing-objects, from RoboRSI)."""

MIN_PAD_ON_SLAB = 0.005
"""Least blade width that must lie on the overhang for a pinch to be tried [m]."""

STANDOFF = 0.02
"""Push mode: free run behind the slab before the hand meets it [m]. Ravens
starts its push 0.02 m before the start pose."""

STEP = 0.01
"""Length of one drag move [m]: Ravens' 0.01 m push step."""

MAX_SLIDE = 0.25
"""Longest slide [m]; a slab further from the edge is refused."""

HOVER_CLEARANCE = 0.08
"""Height of the hovers over the contact and the pinch [m] (pushing-objects)."""

CONTACT_FRACTION = 0.5
"""Push mode: where up the slab the tool centre pushes, as a fraction of its
height, floored by MIN_TOOL_HEIGHT (pushing-objects)."""

MIN_TOOL_HEIGHT = 0.015
"""Lowest tool-centre height over the support while the hand is over it [m]. A
real YAM station refuses a tool centre under 14 mm over its calibrated table."""

PRESS_INSET = 0.012
"""Press mode: how far inside the slab's back edge the fingertips press [m]."""

PRESS_DEPTH = 0.0015
"""Press mode: how far below the measured contact the drag is commanded [m]:
the candidate's 1-2 mm press, with no force sensor."""

PRESS_MIN_STOP = 0.0015
"""Press mode: a descent that ends this close to the floor met nothing [m]."""

SLIDE_TOLERANCE = 0.005
"""How far short of the planned slide the hand may stop and still count [m]."""

REACH_TOLERANCE = 0.02
"""How far the hand may end from a hover or the lowered pose [m]
(pushing-objects)."""

MODES = ("push", "press")


class Output(TypedDict):
    route: str
    detail: str
    overhang_before: float
    overhang_target: float
    slide_planned: float
    slide_achieved: float
    overhang_expected: float
    plan: dict[str, Any]


def slab_geometry(
    obb: dict[str, Any], edge_point: dict[str, float], edge_direction: dict[str, float] | None
) -> dict[str, Any]:
    """The slab seen against the edge: the outward normal n (horizontal, from the
    slab's side of the edge toward the drop), the edge direction d, the centre's
    distance inside the edge, the slab's reach ahead of and behind its centre
    along n, its half width along d, its height, top and bottom."""
    centre = np.array([float(obb["center"][k]) for k in ("x", "y", "z")])
    half = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    rot = C.matrix(obb.get("orientation"))
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = (signs * half) @ rot.T
    e = np.array([float(edge_point["x"]), float(edge_point["y"])])
    if edge_direction is not None:
        d = np.array([float(edge_direction["x"]), float(edge_direction["y"])])
        if float(np.linalg.norm(d)) < 1.0e-9:
            raise ValueError("edge_direction has no horizontal part")
        d = d / np.linalg.norm(d)
        n = np.array([d[1], -d[0]])
        if float(np.dot(e - centre[:2], n)) < 0.0:
            n = -n
    else:
        # A point only: the edge runs square to the line from the slab to it.
        n = e - centre[:2]
        if float(np.linalg.norm(n)) < 1.0e-9:
            raise ValueError("the edge point is the slab's centre; give edge_direction")
        n = n / np.linalg.norm(n)
        d = np.array([-n[1], n[0]])
    along_n = corners[:, :2] @ n
    along_d = corners[:, :2] @ d
    return {
        "centre": centre,
        "n": n,
        "d": d,
        "inside": float(np.dot(e - centre[:2], n)),
        "front": float(along_n.max()),
        "back": float(-along_n.min()),
        "half_width": float(max(along_d.max(), -along_d.min())),
        "height": float(corners[:, 2].max() - corners[:, 2].min()),
        "top": float(centre[2] + corners[:, 2].max()),
        "bottom": float(centre[2] + corners[:, 2].min()),
        "edge": e,
    }


def run(
    ctx: NodeContext,
    target_obb: dict[str, Any],
    edge_point: dict[str, float],
    edge_direction: dict[str, float] | None = None,
    support_z: float | None = None,
    mode: str = "push",
    overhang: float | None = None,
    overhang_fraction: float = OVERHANG_FRACTION,
    max_slide: float = MAX_SLIDE,
    standoff: float = STANDOFF,
    step: float = STEP,
    hover_clearance: float = HOVER_CLEARANCE,
    contact_fraction: float = CONTACT_FRACTION,
    min_tool_height: float = MIN_TOOL_HEIGHT,
    press_depth: float = PRESS_DEPTH,
    grasp_height: float | None = None,
    pad_half_width: float = PAD_HALF_WIDTH,
    reach_tolerance: float = REACH_TOLERANCE,
    speed_scale: float | None = None,
    roll_limits: list[float] | None = None,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    hand = C.Hand(ctx, arm_id, reach_tolerance)
    g = slab_geometry(target_obb, edge_point, edge_direction)
    n, d = g["n"], g["d"]
    n3 = np.array([*n, 0.0])
    centre = g["centre"]
    support = float(g["bottom"] if support_z is None else support_z)
    depth = g["front"] + g["back"]
    before = g["front"] - g["inside"]
    edge_n = float(np.dot(g["edge"], n))

    out: Output = {
        "route": "unreachable",
        "detail": "",
        "overhang_before": before,
        "overhang_target": 0.0,
        "slide_planned": 0.0,
        "slide_achieved": 0.0,
        "overhang_expected": before,
        "plan": {},
    }

    def unreachable(why: str) -> Output:
        out["route"], out["detail"] = "unreachable", "; ".join([why, *hand.notes])
        return out

    if g["inside"] <= 0.0:
        return unreachable("the slab's centre is not on the support side of the edge")
    # The overhang a pinch needs (a blade past the edge with clearance, and some
    # of the blade on the slab) and the most the slab takes with its centre on
    # the support.
    blade_from_edge = float(pad_half_width) + EDGE_CLEARANCE
    needed = blade_from_edge + MIN_PAD_ON_SLAB
    most = g["front"] - COM_INSET
    want = min(float(overhang) if overhang is not None else float(overhang_fraction) * depth, most)
    out["overhang_target"] = want
    if want < needed:
        return unreachable(
            f"the slab is {depth * 1e3:.0f} mm deep across the edge: it can overhang "
            f"{most * 1e3:.0f} mm with its centre on the support, under the {needed * 1e3:.0f} mm a pinch needs"
        )
    slide = max(0.0, want - before)
    out["slide_planned"] = slide
    if slide > float(max_slide):
        return unreachable(
            f"the slide would be {slide * 1e3:.0f} mm (over {float(max_slide) * 1e3:.0f})"
        )

    try:
        gripper = hand.tool("robot.describe_gripper")
    except Exception as exc:  # the width checks fall back to none
        logger.info("sliding-to-edge-then-picking: no gripper description (%s)", exc)
        gripper = {}
    fit = gripper.get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or gripper.get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)
    max_width = float(gripper.get("max_grasp_width_m") or at_open or 0.0)
    across = 2.0 * g["half_width"]
    if max_width and across > max_width:
        return unreachable(
            f"the slab is {across * 1e3:.0f} mm wide along the edge, over the hand's {max_width * 1e3:.0f} mm"
        )

    height = g["height"]
    if mode == "push":
        # The closed blades stand a half width ahead of the tool centre.
        behind = g["back"] + float(pad_half_width) + EDGE_PADDING
        contact = np.array(
            [
                *(centre[:2] - n * behind),
                support + max(float(contact_fraction) * height, float(min_tool_height)),
            ]
        )
        lowered = contact - n3 * float(standoff)
    else:
        inset = min(PRESS_INSET, depth / 4.0)
        contact = np.array([*(centre[:2] - n * (g["back"] - inset)), g["top"]])
        lowered = contact.copy()
    engage = EDGE_PADDING if mode == "push" else 0.0
    finish = contact + n3 * (slide + engage)
    finish[2] = max(finish[2], support + float(min_tool_height))
    hover = lowered + np.array([0.0, 0.0, float(hover_clearance)])
    # The pinch: on the slab's centreline, a blade's half width plus clearance
    # past the edge, the tool centre (between the pads) at the slab's mid-height.
    slab_after = centre + n3 * slide
    grasp_xy = float(np.dot(slab_after[:2], d)) * d + (edge_n + blade_from_edge) * n
    grasp = np.array(
        [*grasp_xy, support + (height / 2.0 if grasp_height is None else float(grasp_height))]
    )

    # Fingers across the drag, jaws closing along the edge: the same heading
    # pinches the overhang from its two sides.
    heading = math.degrees(math.atan2(d[1], d[0]))
    start = hand.here()
    rotations: list[dict[str, float]] = []
    try:
        frame = hand.tool("robot.grasp_frame", close_heading_deg=heading)
        rotations = [
            frame[k]
            for k in ("rotation", "rotation_flipped")
            if isinstance(frame, dict) and frame.get(k)
        ]
    except Exception as exc:
        logger.info("sliding-to-edge-then-picking: no top-down rotation (%s)", exc)
    rotations.append(start["rotation"])
    wanted = rotations[0]

    if check_reach:
        # A stall at the edge of reach reads exactly like contact, and a pinch
        # the arm cannot reach should not cost the slab its place.
        for label, pose in (("the end of the slide", finish), ("the pinch", grasp)):
            try:
                reply = hand.tool(
                    "robot.solve_ik", pose={"position": C.vec(pose), "rotation": wanted}
                )
            except Exception as exc:
                # Some connectors raise for an unreachable pose rather than
                # answering success=False; any other failure is no preview.
                if "unreachable" in str(exc).lower():
                    return unreachable(f"the hand cannot reach {label} (IK preview: {exc})")
                logger.info("sliding-to-edge-then-picking: no IK preview (%s)", exc)
                continue
            if isinstance(reply, dict) and (
                reply.get("success") is False or reply.get("within_servo_tolerance") is False
            ):
                return unreachable(f"the hand cannot reach {label} (IK preview)")

    out["plan"] = {
        "grasp_position": C.vec(grasp),
        "rotation": wanted,
        "edge_normal": C.vec(n3),
        "overhang_expected": before,
        "needed_overhang": needed,
        "hover_clearance": float(hover_clearance),
        "at_open": at_open,
        "at_closed": at_closed,
    }
    if slide <= 0.0:
        out["route"] = "slid"
        out["detail"] = f"the slab already overhangs {before * 1e3:.0f} mm; nothing to slide"
        return out

    hand.tool("robot.close_gripper")

    rotation, why = None, "no rotation reached the hover"
    for candidate in rotations:
        tcp = C.xyz(start)
        why = ""
        if tcp[2] < hover[2] - 1.0e-3:
            why = hand.move("robot.go_to_pose", np.array([tcp[0], tcp[1], hover[2]]), candidate)
        why = why or hand.go("robot.go_to_pose", hover, candidate)
        if not why:
            rotation = candidate
            break
        start = hand.here()
    if rotation is None:
        return unreachable(f"hover: {why}")
    rotation = hand.align(wanted, hover, roll_limits)

    if mode == "push":
        why = hand.move("robot.go_to_pose_cartesian", lowered, rotation)
        reached = hand.where()
        off = float(np.linalg.norm(reached - lowered))
        if off > float(reach_tolerance):
            return unreachable(
                f"descent: {why or 'ended off'}; the hand stopped {off * 1e3:.0f} mm from the lowered pose"
            )
        # Stopped short but close (closed fingertips on the support): push at
        # the height reached.
        drag_z = float(reached[2]) if why else float(contact[2])
    else:
        floor = support + float(min_tool_height)
        why = hand.move(
            "robot.move_cartesian_until_contact", np.array([*contact[:2], floor]), rotation
        )
        reached = hand.where()
        # Judged by height, not the move's word: a servo calls a stop within its
        # arrival ball "reached" (the fingertips met an 8 mm slab 3 mm above the
        # 15 mm floor in a YAM station twin).
        if float(reached[2]) <= floor + PRESS_MIN_STOP:
            return unreachable(
                f"press: the hand came down to {(reached[2] - support) * 1e3:.0f} mm over the support "
                f"without meeting the slab's top ({why or 'the move finished'})"
            )
        if float(np.linalg.norm(reached[:2] - contact[:2])) > float(reach_tolerance):
            return unreachable(f"press: the hand stopped off the press point ({why})")
        drag_z = float(reached[2]) - float(press_depth)
        hand.move("robot.go_to_pose_cartesian", np.array([*contact[:2], drag_z]), rotation)

    # Ravens' drag: one straight line in STEP-long moves, judged by the hand.
    origin = hand.where()
    begin = np.array([*origin[:2], drag_z])
    total = float(np.dot(finish[:2] - begin[:2], n))
    steps = max(1, int(math.ceil(total / float(step) - 1.0e-9)))
    extra = {"speed_scale": float(speed_scale)} if speed_scale is not None else {}
    drag_why, travelled = "", 0.0
    for i in range(1, steps + 1):
        target = begin + n3 * min(total, i * float(step))
        drag_why = hand.move("robot.go_to_pose_cartesian", target, rotation, **extra)
        progress = float(np.dot(hand.where()[:2] - begin[:2], n))
        if drag_why and not drag_why.startswith("timeout"):
            break
        if progress < travelled + 0.25 * min(float(step), total):
            drag_why = drag_why or "the hand stopped advancing"
            break
        travelled, drag_why = progress, ""
    end = hand.where()
    achieved = max(0.0, float(np.dot(end[:2] - contact[:2], n)) - engage)
    out["slide_achieved"] = achieved
    out["overhang_expected"] = before + achieved
    up = hand.go(
        "robot.go_to_pose_cartesian", end + np.array([0.0, 0.0, float(hover_clearance)]), rotation
    )

    out["plan"]["overhang_expected"] = before + achieved
    notes = hand.notes
    if achieved < slide - SLIDE_TOLERANCE:
        out["route"] = "slid_not_picked"
        out["detail"] = "; ".join(
            [
                f"the hand travelled {achieved * 1e3:.0f} mm of the {slide * 1e3:.0f} mm slide"
                + (f" ({drag_why})" if drag_why else ""),
                *notes,
            ]
        )
        return out
    if up:
        out["route"] = "slid_not_picked"
        out["detail"] = "; ".join([f"lifting off the slab: {up}", *notes])
        return out
    out["route"] = "slid"
    out["detail"] = "; ".join(
        [
            f"slid {achieved * 1e3:.0f} mm by the hand's travel; overhang about {(before + achieved) * 1e3:.0f} mm",
            *notes,
        ]
    )
    return out
