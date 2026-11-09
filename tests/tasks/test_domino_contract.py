"""The domino module against the shared contract rsh-task/1 as the runtime enforces it:
SkillResult envelopes, failure_type vocabulary, recovery hints, schema references and
the TaskSkill declarations."""

from __future__ import annotations

import pytest
from _domino_world import FakeTaskContext, request

from roboskillhub.runtime.task import (
    BACKEND_CAPABILITIES,
    CONTRACT,
    READINESS_LEVELS,
    RECOVERY_HINTS,
    known_failure_type,
    offline_context,
    run_task_skill,
    validate,
)
from roboskillhub.tasks.domino import SKILLS, place_upright
from roboskillhub.tasks.domino.contract import FAILURE_TYPE, PHASE
from roboskillhub.tasks.domino.contract import RECOVERY_HINTS as MODULE_HINTS


def test_vocabularies_are_the_contracts():
    assert all(known_failure_type(v) for v in FAILURE_TYPE.values())
    assert all(set(h) <= RECOVERY_HINTS for h in MODULE_HINTS.values())
    assert set(PHASE.values()) <= set(CONTRACT["phases"])


def _refs(schema):
    if isinstance(schema, dict):
        if "$ref" in schema:
            yield schema["$ref"].rsplit("/", 1)[-1]
        for v in schema.values():
            yield from _refs(v)


def test_skill_declarations():
    ids = [s.skill_id for s in SKILLS]
    assert ids == [
        "domino.place_upright",
        "domino.verify_stable_placement",
        "domino.plan_straight_layout",
        "domino.plan_recovery",
    ]
    for s in SKILLS:
        assert s.kind in ("tool", "skill") and s.readiness in READINESS_LEVELS
        assert set(s.requires) <= BACKEND_CAPABILITIES
        assert set(_refs(s.input_schema)) <= set(CONTRACT["schemas"])
        assert s.input_schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "faults", [(), ("retreat_knocks",), ("stale_obs",), ("missed_grasp",), ("tcp_offset",)]
)
def test_place_results_are_valid_skill_results(faults):
    ctx = FakeTaskContext(faults=set(faults))
    out = place_upright(request(ctx), ctx)
    validate("SkillResult", out)
    for o in out["observations"]:
        validate("Observation", o)


def test_tools_run_offline_through_the_gate():
    from test_domino_layout import req as layout_req
    from test_domino_verify import req as verify_req
    from test_domino_verify import stream

    by_id = {s.skill_id: s for s in SKILLS}
    out = run_task_skill(by_id["domino.plan_straight_layout"], layout_req(), offline_context())
    assert out["status"] == "succeeded", out.get("error")
    validate("DominoLayoutTarget", out["evidence"]["layout_target"])
    out = run_task_skill(
        by_id["domino.verify_stable_placement"], verify_req(stream()), offline_context()
    )
    assert out["status"] == "succeeded", out.get("error")
    failed = place_upright(
        request(FakeTaskContext(faults={"fall_on_release"})),
        FakeTaskContext(faults={"fall_on_release"}),
    )
    out = run_task_skill(
        by_id["domino.plan_recovery"],
        {"result": failed, "attempts_used": 0, "max_attempts": 2},
        offline_context(),
    )
    assert (
        out["status"] == "succeeded"
        and out["outputs"]["recovery"]["action"] == "recover_fallen_tile"
    )


def test_place_without_a_station_is_rejected_not_success():
    by_id = {s.skill_id: s for s in SKILLS}
    ctx = offline_context()
    out = run_task_skill(by_id["domino.place_upright"], request(FakeTaskContext()), ctx)
    assert out["status"] == "rejected" and out["failure_type"] == "capability_missing"


def test_without_the_skill_scripts_entries_reject_clearly_and_the_observer_still_works(
    monkeypatch, tmp_path
):
    # As from a wheel: no skills/ anywhere the loader looks.
    from test_domino_layout import req as layout_req
    from test_domino_verify import req as verify_req
    from test_domino_verify import stream

    from roboskillhub.tasks.domino import upstream

    monkeypatch.setattr(upstream, "candidate_roots", lambda: [tmp_path / "skills"])
    upstream.load.cache_clear()
    try:
        by_id = {s.skill_id: s for s in SKILLS}
        out = run_task_skill(by_id["domino.plan_straight_layout"], layout_req(), offline_context())
        assert out["status"] == "rejected" and out["failure_type"] == "precondition_unmet"
        assert "place_domino.py is not installed" in out["error"]["message"]
        assert "ROBOSKILLHUB_ROOT" in out["error"]["message"]
        ctx = FakeTaskContext()
        out = place_upright(request(ctx), ctx)
        assert out["status"] == "rejected" and out["failure_type"] == "precondition_unmet"
        assert [c for c in ctx.calls if c != "observe_objects"] == []
        assert upstream.available()["layout"]
        out = run_task_skill(
            by_id["domino.verify_stable_placement"], verify_req(stream()), offline_context()
        )
        assert out["status"] == "succeeded"
    finally:
        upstream.load.cache_clear()
