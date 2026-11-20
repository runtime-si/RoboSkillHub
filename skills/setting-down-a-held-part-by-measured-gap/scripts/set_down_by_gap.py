"""Set a part held upright down on a support by measuring, in the wrist camera, how far its lower end still is from it."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrist_depth as wd  # noqa: E402

logger = logging.getLogger(__name__)

APPROACH = 0.09
"""The hand comes in this far above the place pose and is lowered from there [m] (yam_dominos' carry height: clear
of 49 mm tiles already standing 32 mm away)."""

RECHECK = 0.015
"""The first lowering stops with the part's lower end this far up, and it is measured again [m]: from 9 cm the
reading was good to a few millimetres, from 1.5 cm to about one."""

CREEP = 0.003
"""Below 8 mm the hand comes down this much a step [m]; above, by 0.6 of what is left."""

DROP = 0.001
"""The part is let go with its lower end this far up [m]. Let go 6 mm up, three tiles in a row fell over; from
-0.1 to 2.3 mm ten real set-downs were let go in the run that stood 11 tiles in a line (left YAM, 2026-11-20)."""

DROP_TOLERANCE = 0.0015
"""... give or take this [m]: a reading under DROP + this lets go."""

MAX_STEPS = 11
"""Creeping steps at most."""

CONTACT_HAND = 0.0015
CONTACT_SHARE = 0.34
"""The hand came down more than CONTACT_HAND in a step and the part's end came nearer the support by less than this
share of that: the part stands on the support and slides in the pads. (Where the part's upper end sat in the
camera's frame was tried as the signal first: it jumped 3-6 mm from look to look with the part a centimetre up.)"""

STALL = 0.0006
STALL_GAP = 0.004
"""Sent down, and neither the hand nor the gap changed by more than STALL with more than STALL_GAP to go, twice:
the hand goes no lower here (a workspace floor). The part is kept."""

EXPECT_WITHIN = 0.04
"""The part's upper end is looked for within this of the tool centre when no better place is given [m] ..."""

TRACK_WITHIN = 0.016
"""... and within this of where it was last seen [m]: a part standing 3.5 cm ahead in a line is as light and as
near, and was once read as the held one."""

MIN_SUPPORT_POINTS = 400
"""A gap is believed only with this many support points round the part's foot in the frame."""


class Output(TypedDict):
    route: str
    release_pose: dict[str, Any]
    gap: float
    lowered: float
    readings: int
    detail: str


def next_step(gap: float) -> float:
    """How far to lower for a part whose lower end is ``gap`` over the support [m]."""
    return float(min(CREEP if gap < 0.008 else 0.6 * gap, max(gap - DROP, 0.001)))


def on_the_support(gap_before: float, gap_after: float, hand_down: float) -> bool:
    return (
        hand_down > CONTACT_HAND
        and gap_before < 0.012
        and (gap_before - gap_after) < CONTACT_SHARE * hand_down
    )


def run(
    ctx: NodeContext,
    place_pose: dict[str, Any],
    part_tall: float,
    held_top_in_tool: dict[str, float] | None = None,
    light_part: bool = True,
    camera: str | None = None,
    arm_id: int = 0,
    support_z: float = 0.0,
    approach: float = APPROACH,
    drop: float = DROP,
    check_reach: bool = True,
) -> Output:
    """``place_pose``: the tool pose at which the model has the part's lower end on the support (its x, y and
    rotation are kept; its height is only where the lowering is expected to end). ``part_tall``: how far the part's
    lower end is under its upper end as it is held [m]. ``held_top_in_tool``: where the grasp's geometry has the
    part's upper end, tool frame (default: at the tool centre). ``support_z``: the world height of what it is set on.
    The hand is left where it let go, jaws open; a part that was not set down is still held."""
    arm = {"arm_id": int(arm_id)} if arm_id else {}
    place = wd.matrix(wd.pose_of(place_pose))
    top_tool = (
        np.zeros(3)
        if held_top_in_tool is None
        else np.array([float(held_top_in_tool[k]) for k in ("x", "y", "z")])
    )
    log: list[str] = []
    last_cam: np.ndarray | None = None
    readings = 0
    start_z = None

    def here() -> np.ndarray:
        return wd.matrix(wd.pose_of(ctx.tool("robot.get_ee_pose", **arm)))

    def out(route: str, gap: float, detail: str) -> Output:
        now = here()
        return {
            "route": route,
            "release_pose": wd.pose_dict(now),
            "gap": float(gap),
            "lowered": 0.0 if start_z is None else float(start_z - now[2, 3]),
            "readings": readings,
            "detail": "; ".join([detail, *log[-6:]]),
        }

    def measure() -> float | None:
        nonlocal last_cam, readings
        obs = ctx.tool("robot.get_observation")
        tool = wd.matrix(obs["arms"][arm_id]["ee_pose"])
        cam = wd.wrist_camera(obs, tool, camera)
        if cam is None:
            return None
        t_cam = wd.matrix(cam["pose"])
        expect = (
            last_cam
            if last_cam is not None
            else (np.linalg.inv(t_cam) @ tool @ np.append(top_tool, 1.0))[:3]
        )
        try:
            r = wd.held_gap(
                cam["depth"],
                cam["intrinsics"],
                t_cam,
                float(part_tall),
                expect,
                within=TRACK_WITHIN if last_cam is not None else EXPECT_WITHIN,
                rgb=cam.get("rgb") if light_part else None,
                support_z=float(support_z),
            )
        except ValueError:
            return None
        if (
            not r.get("seen")
            or not (-0.02 < r["gap"] < 0.12)
            or r["support_points"] < MIN_SUPPORT_POINTS
        ):
            return None
        last_cam = np.asarray(r["point_cam"], float)
        readings += 1
        return float(r["gap"])

    def lower(by: float) -> tuple[float, str]:
        """Lower the hand by ``by`` straight down; (how far it really came down, why not or "")."""
        before = here()
        goal = place.copy()
        goal[2, 3] = before[2, 3] - by
        try:
            why = wd.refusal(ctx.tool("robot.go_to_pose_cartesian", pose=wd.pose_dict(goal), **arm))
        except Exception as exc:  # a connector that raises its refusals
            why = str(exc) or type(exc).__name__
        return float(before[2, 3] - here()[2, 3]), why

    above = place.copy()
    above[2, 3] += float(approach)
    goal = wd.pose_dict(above)
    try:
        if check_reach:
            reply = ctx.tool("robot.solve_ik", pose=goal, **arm)
            if isinstance(reply, dict) and (
                reply.get("success") is False or reply.get("within_servo_tolerance") is False
            ):
                return out(
                    "unreachable",
                    float("nan"),
                    "the pose above the place is out of reach (IK preview)",
                )
        why = wd.refusal(ctx.tool("robot.go_to_pose", pose=goal, **arm))
    except Exception as exc:
        why = str(exc) or type(exc).__name__
    if why:
        return out("unreachable", float("nan"), f"the pose above the place: {why}")
    start_z = here()[2, 3]
    gap = measure()
    if gap is None:
        # never seen: NOT lowered by the model. Twice on the real arm the model's place for the part's end was believed
        # over a reading, and the hand drove the part 7 and 17 mm into the counter
        return out(
            "not_seen",
            float("nan"),
            "the held part was not made out over the support from above the place",
        )
    log.append(f"from above: {1000 * gap:.1f} mm")
    for _ in range(2):  # far: to RECHECK over the support, and measured again
        if gap <= RECHECK + 0.004:
            break
        went, why = lower(gap - RECHECK)
        nxt = measure()
        if nxt is None:
            return out(
                "not_seen",
                gap,
                f"the part is no longer seen after lowering {1000 * went:.1f} mm ({why})",
            )
        gap = nxt
        log.append(f"lowered {1000 * went:.1f} mm: {1000 * gap:.1f} mm")
    stalls = 0
    for _ in range(MAX_STEPS):
        if gap <= float(drop) + DROP_TOLERANCE:
            ctx.tool("robot.open_gripper", **arm)
            return out("set_down", gap, f"let go {1000 * gap:.1f} mm over the support")
        went, why = lower(next_step(gap))
        nxt = measure()
        if nxt is None:
            return out("not_seen", gap, f"the part is no longer seen ({why}); it is still held")
        log.append(f"hand {1000 * went:.1f} mm down, gap {1000 * gap:.1f} -> {1000 * nxt:.1f} mm")
        if on_the_support(gap, nxt, went):
            ctx.tool("robot.open_gripper", **arm)
            return out(
                "set_down",
                nxt,
                f"the hand came down {1000 * went:.1f} mm and the part's end "
                f"{1000 * (gap - nxt):.1f}: it stands on the support; let go",
            )
        stalls = stalls + 1 if (went < STALL and abs(gap - nxt) < STALL and nxt > STALL_GAP) else 0
        gap = nxt
        if stalls >= 2:
            return out(
                "cannot_lower",
                gap,
                f"the hand goes no lower with the part still {1000 * gap:.1f} mm up ({why}); it is still held",
            )
    return out(
        "cannot_lower",
        gap,
        f"{MAX_STEPS} steps and the part is still {1000 * gap:.1f} mm up; it is still held",
    )
