"""Fuse the hub's per-step checks into one verdict: done, failed (with the reason) or uncertain.

The checks are complementary (live comparison in SKILL.md): the gripper and the camera VLM judges catch grasp
failures but pass bricks resting a millimetre high; the depth seat and pull-up checks catch seating; a value
model (RynnValue) rarely passes a failure but rejects many successes. Each step type therefore trusts the
checks that see its failure modes, and uses the others only to veto or to break a tie.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from gap import NodeContext

logger = logging.getLogger(__name__)

SEAT_OK = {"seated"}
SEAT_FAIL = {"not_seated", "high_by_stud", "tilted", "shifted_off_grid", "no_brick"}
PULL_OK = {"connected"}
PULL_FAIL = {"not_connected", "not_holding"}
GRIP_OK = {"stalled_on_object", "held"}
RELEASE_FAIL = {"stalled_jammed", "stalled_on_object", "refused", "still_moving", "held"}
GRIP_FAIL = {
    "reached": "grasp_missed",
    "unexpected_width": "wrong_part_or_edge_grasp",
    "empty": "grasp_missed",
}
VLM_OK = {"succeeded", "complete", "changed_as_expected", "done"}
VLM_FAIL = {
    "not_yet",
    "failed",
    "in_progress",
    "changed_otherwise",
    "unchanged",
    "wrong_object",
    "wrong_destination",
}

MAX_HEIGHT_ERROR_M = 0.0003
"""Seat height error accepted as seated, chosen on the training split (SKILL.md); the depth skill's own
tolerance is min(1 mm, 0.35 stud heights)."""

MAX_PULL_RISE_M = 0.0006
"""Brick rise during the 3 mm pull-up accepted as connected, chosen on the training split (SKILL.md)."""

RYNN_VETO_S = 2.5
"""RynnValue remaining time above which a 'done' is held back as uncertain (#155: successful placements
ended at 0.9-3.0 s, failed ones at 1.9-5.1 s)."""


class Output(TypedDict):
    route: str
    reason: str
    step_type: str
    evidence: dict[str, Any]


def _norm(v: Any) -> str:
    return str(v or "").strip().lower()


def _vote(route: str, ok: set[str], fail: set[str] | dict[str, str]) -> bool | None:
    r = _norm(route)
    if not r:
        return None
    if r in ok:
        return True
    if r in fail:
        return False
    return None


def _out(route: str, reason: str, step_type: str, ev: dict[str, Any]) -> Output:
    logger.info("fuse_checks: %s %s (%s)", step_type, route, reason)
    return {"route": route, "reason": reason, "step_type": step_type, "evidence": ev}


def judge_pick(ev: dict[str, Any], vlm: bool | None) -> tuple[str, str]:
    grip = _norm(ev.get("grip_route"))
    if grip in GRIP_FAIL:
        return "failed", GRIP_FAIL[grip]
    held = ev.get("held")
    if held is False:
        return "failed", "grasp_missed: the hand reports nothing held"
    grip_ok = grip in GRIP_OK or held is True
    scene = _vote(ev.get("scene_diff_route", ""), VLM_OK, VLM_FAIL)
    if grip_ok and scene is not False and vlm is not False:
        return "done", "the jaws stopped on the part" + (
            "; the scene agrees" if scene or vlm else ""
        )
    if grip_ok:
        return (
            "uncertain",
            "the jaws hold something but the camera check disagrees (wrong part or object still at source)",
        )
    if scene is False or vlm is False:
        return "failed", "grasp_missed: the camera shows the part still at its source"
    return "uncertain", "no gripper reading"


def judge_place(
    ev: dict[str, Any], vlm: bool | None, max_h: float, max_rise: float
) -> tuple[str, str]:
    if _norm(ev.get("release_grip_route")) in RELEASE_FAIL:
        return (
            "failed",
            "the jaws did not open at the release: the part left with the hand or is dragged",
        )
    seat = _vote(ev.get("seat_route", ""), SEAT_OK, SEAT_FAIL)
    h = ev.get("height_error_m")
    if h is not None and seat is not None and h == h:
        seat = seat and abs(float(h)) <= max_h
    pull = _vote(ev.get("pull_route", ""), PULL_OK, PULL_FAIL)
    rise = ev.get("brick_rise_m")
    if (
        rise is not None
        and rise == rise
        and _norm(ev.get("pull_route")) not in ("not_holding", "blocked")
    ):
        # The brick's own rise is the direct observation; the pull-up's route reads the hand's tracking lag,
        # which a stiff (or simulated floating) hand never shows, so it said not_connected on every LEGO
        # station placement, seated or not (SKILL.md).
        pull = float(rise) <= max_rise
    if seat is False and pull is False:
        return (
            "failed",
            f"not seated ({_norm(ev.get('seat_route'))}) and not connected under the pull-up",
        )
    if seat is True and pull is True:
        return "done", "seated by depth and held under the pull-up"
    geo = [v for v in (seat, pull) if v is not None]
    if len(geo) == 1:
        one = "depth seat" if seat is not None else "pull-up"
        if geo[0] is False:
            return (
                "failed",
                f"{one} check failed ({_norm(ev.get('seat_route') if seat is not None else ev.get('pull_route'))})",
            )
        if vlm is False:
            return "uncertain", f"{one} passed but the camera judge says not done"
        return "uncertain", f"only the {one} check ran; seating needs both"
    if not geo:
        return "uncertain", "no geometric check ran; a camera judge alone passes unseated bricks"
    # The two geometric checks disagree: the camera judge breaks the tie only towards failure.
    if vlm is False:
        return "failed", "depth seat and pull-up disagree and the camera judge says not done"
    return "uncertain", (
        "seated by depth but not held under the pull-up"
        if seat
        else "held under the pull-up but the depth reads unseated"
    )


def run(
    ctx: NodeContext,
    step_type: str,
    grip_route: str = "",
    held: bool | None = None,
    scene_diff_route: str = "",
    release_grip_route: str = "",
    seat_route: str = "",
    height_error_m: float | None = None,
    pull_route: str = "",
    brick_rise_m: float | None = None,
    vlm_route: str = "",
    rynn_remaining_s: float | None = None,
    max_height_error_m: float = MAX_HEIGHT_ERROR_M,
    max_pull_rise_m: float = MAX_PULL_RISE_M,
    rynn_veto_s: float = RYNN_VETO_S,
) -> Output:
    """Verdict for one ``step_type`` (``pick`` or ``place``; ``press`` and ``seat`` mean ``place``) from the
    routes / numbers of the checks that ran. Empty strings and None mean "did not run"."""
    st = _norm(step_type)
    st = "place" if st in ("press", "seat", "place") else st
    if st not in ("pick", "place"):
        raise ValueError(f"step_type {step_type!r} is not pick or place")
    ev = {
        "grip_route": grip_route,
        "held": held,
        "scene_diff_route": scene_diff_route,
        "release_grip_route": release_grip_route,
        "seat_route": seat_route,
        "height_error_m": height_error_m,
        "pull_route": pull_route,
        "brick_rise_m": brick_rise_m,
        "vlm_route": vlm_route,
        "rynn_remaining_s": rynn_remaining_s,
    }
    vlm = _vote(vlm_route, VLM_OK, VLM_FAIL)
    if st == "pick":
        route, why = judge_pick(ev, vlm)
    else:
        route, why = judge_place(ev, vlm, float(max_height_error_m), float(max_pull_rise_m))
    if (
        route == "done"
        and rynn_remaining_s is not None
        and float(rynn_remaining_s) > float(rynn_veto_s)
    ):
        route, why = (
            "uncertain",
            f"{why}; but RynnValue still expects {float(rynn_remaining_s):.1f} s of work",
        )
    return _out(route, why, st, ev)
