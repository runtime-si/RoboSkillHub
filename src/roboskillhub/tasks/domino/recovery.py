"""Bounded recovery plans for a domino placement result. Plans only: nothing here moves.

Every plan starts from a fresh observation of the target and its neighbours (never
from the failed attempt's last state), names the capabilities each step needs (some
are Agent 01's, some not in the runtime yet) and is refused once the attempt budget is
spent. It never proposes re-running the whole layout.
"""

from __future__ import annotations

from typing import Any

OBSERVE = {"step": "observe_scene", "capability": "object_pose_feedback", "why": "fresh target + neighbour poses; recovery starts from these"}
REGRASP = {"step": "stage_and_regrasp", "capability": "agent01.stage_and_regrasp", "why": "hold the tile with a known T_tcp_object again"}
STAND = {"step": "stand_lying_tile", "capability": "agent01.stage_and_regrasp", "why": "a lying tile must be stood/regrasped before placing (Agent 01)"}
PLACE = {"step": "place_upright", "capability": "domino.place_upright", "why": "re-place at the same target with fresh evidence"}
RETREAT_HOLDING = {"step": "retreat_holding", "capability": "linear_cartesian_motion", "why": "back straight off the support, gripper unchanged"}
VERIFY = {"step": "verify_stable_placement", "capability": "domino.verify_stable_placement", "why": "judge again from new frames"}


def _plan(action: str, steps: list[dict], affected: list[str], note: str) -> dict[str, Any]:
    return {
        "action": action,
        "steps": steps,
        "affected": affected,
        "requires_capabilities": sorted({s["capability"] for s in steps}),
        "note": note,
    }


def plan_recovery(result: dict, *, attempts_used: int, max_attempts: int) -> dict[str, Any]:
    status = result.get("status")
    failure = result.get("failure") or {}
    code, phase = failure.get("code"), failure.get("phase")
    target = (result.get("evidence") or {}).get("plan", {}).get("target_id")
    base = {"bounded": True, "attempt": attempts_used + 1, "max_attempts": max_attempts, "executed": False}
    if status == "succeeded":
        return {**base, **_plan("none", [], [], "placed and verified")}
    if status == "rejected":
        if code == "observation_stale":
            p = _plan("reobserve", [OBSERVE], [], "nothing moved; get fresh evidence and resubmit")
        elif code in ("holding_not_confirmed",):
            p = _plan("regrasp", [OBSERVE, REGRASP], [], "nothing moved; the tile is not confirmed in the hand")
        elif code in ("neighbor_clearance", "retreat_blocked", "infeasible_geometry"):
            p = _plan(
                "replan",
                [{"step": "plan_straight_layout", "capability": "domino.plan_straight_layout", "why": "another order, pitch or target"}],
                [],
                "nothing moved; the target or its corridor is infeasible",
            )
        else:
            p = _plan("fix_request", [], [], f"nothing moved ({code}); not recoverable by motion")
        return {**base, **p}
    if attempts_used >= max_attempts:
        return {**base, **_plan("abort", [], [], f"recovery budget {max_attempts} spent; report and stop")}
    if code in ("timeout", "cancelled", "backend_error"):
        return {**base, **_plan("abort", [OBSERVE], [], f"{code}: observe and hand to an operator, no automatic motion")}
    if status == "unverified":
        return {**base, **_plan("reobserve", [OBSERVE, VERIFY], [], "the outcome is unknown; observe again (e.g. another viewpoint) before any motion")}
    if code == "neighbor_disturbed":
        ids = sorted({f.get("neighbor_id") for f in ((result.get("evidence") or {}).get("verification") or {}).get("faults", []) if f.get("neighbor_id")} | ({failure.get("neighbor_id")} - {None}))
        return {
            **base,
            **_plan(
                "restore_segment",
                [OBSERVE, {"step": "plan_straight_layout", "capability": "domino.plan_straight_layout", "why": "order the affected tiles (far-first) around those still standing"}, STAND, PLACE, VERIFY],
                ids,
                "restore the disturbed neighbours, then the target if it is down; each tile its own bounded attempt",
            ),
        }
    if code in ("descent_blocked", "support_mismatch") or (code == "placement_error" and phase in ("carry", "descent", "support_check")):
        return {**base, **_plan("retreat_and_retry", [RETREAT_HOLDING, OBSERVE, {"step": "check_support_calibration", "capability": "object_pose_feedback", "why": "measure the table under the target"}, PLACE], [], "still holding; nothing released")}
    if code == "object_dropped":
        return {**base, **_plan("regrasp", [OBSERVE, REGRASP, PLACE, VERIFY], [], "the tile left the hand in transit; locate and regrasp it")}
    if code == "release_not_confirmed":
        return {**base, **_plan("release_again", [OBSERVE, {"step": "release", "capability": "gripper_release", "why": "one more bounded opening"}, RETREAT_HOLDING, VERIFY], [], "the hand still supports or holds the tile")}
    if code == "unstable_after_release":
        return {**base, **_plan("recover_fallen_tile", [OBSERVE, STAND, PLACE, VERIFY], [target] if target else [], "the placed tile fell; stand it again (Agent 01), then re-place")}
    if code == "placement_error":
        return {**base, **_plan("regrasp_and_replace", [OBSERVE, REGRASP, PLACE, VERIFY], [target] if target else [], "the tile stands off target; pick it again and re-place")}
    if code == "retreat_blocked":
        return {**base, **_plan("abort", [OBSERVE], [], "released but the retreat did not finish; operator review")}
    return {**base, **_plan("abort", [OBSERVE], [], f"no recovery defined for {code}")}
