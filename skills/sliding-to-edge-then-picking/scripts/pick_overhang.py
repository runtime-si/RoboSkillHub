# Part of sliding-to-edge-then-picking; see slide_to_edge.py for the Ravens
# source, licence and changes. The pinch of the overhang, from beside and below
# the support's top, is reimplemented from Eppner et al. (IJRR 2015) and Hang
# et al. (RA-L 2019), which ship no code.
"""Pinch the part of a flat object that overhangs a support's edge, from its two sides with the fingertips below the support's top, and lift it."""

from __future__ import annotations

import os
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slide_edge_common as C  # noqa: E402

LIFT = 0.05
"""Lift after the pinch [m]."""

GRASP_TOLERANCE = 0.006
"""How far above the pinch height the open hand may stop and still close [m]:
a servo's 4 mm arrival ball plus 2 mm. A blade landing on the support stops
about 6 mm higher (the slab's half height plus the fingertips' reach under the
tool centre), on the slab about 14 mm."""

EMPTY_MARGIN = 0.004
"""A jaw gap under the closed gap plus this closed on nothing [m]."""

REACH_TOLERANCE = 0.02
"""How far the hand may end from the hover over the pinch [m] (pushing-objects)."""


class Output(TypedDict):
    route: str
    detail: str
    held_width: float
    grasp_position: dict[str, float]


def run(
    ctx: NodeContext,
    plan: dict[str, Any],
    slide_route: str = "slid",
    lift: float = LIFT,
    grasp_tolerance: float = GRASP_TOLERANCE,
    empty_margin: float = EMPTY_MARGIN,
    reach_tolerance: float = REACH_TOLERANCE,
    roll_limits: list[float] | None = None,
    arm_id: int = 0,
) -> Output:
    hand = C.Hand(ctx, arm_id, reach_tolerance)
    grasp = np.array([float(plan["grasp_position"][k]) for k in ("x", "y", "z")])
    wanted = plan["rotation"]
    at_open, at_closed = float(plan.get("at_open") or 0.0), float(plan.get("at_closed") or 0.0)
    out: Output = {
        "route": "slid_not_picked",
        "detail": "",
        "held_width": -1.0,
        "grasp_position": C.vec(grasp),
    }

    def not_picked(why: str) -> Output:
        out["route"], out["detail"] = "slid_not_picked", "; ".join([why, *hand.notes])
        return out

    if slide_route != "slid":
        return not_picked(f"the slide ended {slide_route!r}; no pinch tried")
    overhang, needed = (
        float(plan.get("overhang_expected", 0.0)),
        float(plan.get("needed_overhang", 0.0)),
    )
    if overhang < needed:
        return not_picked(
            f"the overhang is about {overhang * 1e3:.0f} mm, under the {needed * 1e3:.0f} mm a pinch needs"
        )

    def gap_width() -> float:
        """The jaw gap [m] (robot.get_gripper's 1.0 = open), or -1.0 when unreadable."""
        try:
            fraction = float(hand.tool("robot.get_gripper")["position"])
        except Exception:  # an unreadable hand is not an empty one
            return -1.0
        if at_open <= 0.0:
            return -1.0
        return at_closed + max(0.0, min(1.0, fraction)) * (at_open - at_closed)

    above = grasp + np.array([0.0, 0.0, float(plan.get("hover_clearance", 0.08))])
    rotation = hand.here()["rotation"]
    why = hand.go("robot.go_to_pose", above, wanted)
    if why:
        why = hand.go("robot.go_to_pose", above, rotation)
    if why:
        return not_picked(f"over the pinch: {why}")
    rotation = hand.align(wanted, above, roll_limits)
    hand.tool("robot.open_gripper")
    why = hand.move("robot.move_cartesian_until_contact", grasp, rotation)
    stop = hand.where()
    high = float(stop[2] - grasp[2])
    if high > float(grasp_tolerance):
        hand.go("robot.go_to_pose_cartesian", above, rotation)
        return not_picked(
            f"the open hand stopped {high * 1e3:.0f} mm above the pinch ({why or 'ended short'}): "
            "a blade landed on the slab or the support"
        )
    hand.tool("robot.close_gripper")
    closed = gap_width()
    lifted = hand.go(
        "robot.go_to_pose_cartesian", stop + np.array([0.0, 0.0, float(lift)]), rotation
    )
    held = gap_width()
    out["held_width"] = held
    if 0.0 <= held <= at_closed + float(empty_margin):
        hand.tool("robot.open_gripper")
        return not_picked(
            f"the jaws closed on nothing ({closed * 1e3:.1f} mm at the pinch, {held * 1e3:.1f} mm after the lift)"
        )
    if lifted:
        return not_picked(f"closed on {held * 1e3:.1f} mm, but the lift: {lifted}")
    out["route"] = "picked"
    holding = (
        f"holding {held * 1e3:.1f} mm between the jaws" if held >= 0.0 else "the jaw gap unreadable"
    )
    out["detail"] = "; ".join(
        [f"pinched the overhang and lifted {float(lift) * 1e3:.0f} mm, {holding}", *hand.notes]
    )
    return out
