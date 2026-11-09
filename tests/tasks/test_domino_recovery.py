"""Recovery plans are bounded, start from a fresh observation and never claim execution."""

from __future__ import annotations

import pytest

from roboskillhub.tasks.domino import plan_recovery


def res(status, code=None, phase=None, **extra):
    return {
        "status": status,
        "failure": {"code": code, "phase": phase, **extra} if code else None,
        "evidence": {"plan": {"target_id": "d2"}},
    }


@pytest.mark.parametrize(
    "status, code, phase, action",
    [
        ("succeeded", None, None, "none"),
        ("rejected", "observation_stale", "validate", "reobserve"),
        ("rejected", "neighbor_clearance", "validate", "replan"),
        ("rejected", "capability_missing", "validate", "fix_request"),
        ("unverified", "orientation_missing", "settle", "reobserve"),
        ("failed", "descent_blocked", "descent", "retreat_and_retry"),
        ("failed", "placement_error", "settle", "regrasp_and_replace"),
        ("failed", "unstable_after_release", "release", "recover_fallen_tile"),
        ("failed", "neighbor_disturbed", "retreat", "restore_segment"),
        ("failed", "timeout", "descent", "abort"),
    ],
)
def test_plan_by_fault(status, code, phase, action):
    p = plan_recovery(res(status, code, phase, neighbor_id="d1"), attempts_used=0, max_attempts=2)
    assert p["action"] == action
    assert p["bounded"] and p["executed"] is False
    names = [s["step"] for s in p["steps"]]
    if "place_upright" in names:
        # Only backing off (gripper unchanged) may precede the fresh observation.
        assert names.index("observe_scene") < names.index("place_upright")
        assert set(names[: names.index("observe_scene")]) <= {"retreat_holding"}


def test_rejected_plans_never_move():
    for code in ("observation_stale", "holding_not_confirmed", "retreat_blocked"):
        p = plan_recovery(res("rejected", code, "validate"), attempts_used=0, max_attempts=2)
        assert "domino.place_upright" not in p["requires_capabilities"]


def test_budget_exhausted_aborts():
    p = plan_recovery(res("failed", "placement_error", "settle"), attempts_used=3, max_attempts=3)
    assert p["action"] == "abort" and p["steps"] == []
