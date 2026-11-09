"""Contract rsh-task/1 integration on scripted fakes (unit_or_dryrun, no physics):
the TaskSkills through the runtime's gate (run_task_skill) and the candidate graph through
the runtime's interpreter (load_graph / run_graph)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from lego_fixtures import DOWN, WORLD, held_object, mating_target

from roboskillhub.runtime.graph import load_graph, run_graph
from roboskillhub.runtime.task import SkillContext, run_task_skill, validate
from roboskillhub.tasks.lego import REGISTRY, SKILLS
from roboskillhub.tasks.lego import geometry as g
from roboskillhub.tasks.lego.fakes import FakeClock, FakeContact, FakeStation
from roboskillhub.tasks.lego.graph import CANDIDATE_GRAPH
from roboskillhub.tasks.lego.manifest import manifest

REPO = Path(__file__).resolve().parents[2]
TCP0 = {"frame_id": WORLD, "position_m": [0.25, 0.05, 0.15], "quaternion_xyzw": DOWN}
B1 = "b1"


def inputs(clock, **over):
    d = {
        "connection_id": "conn-1",
        "target": mating_target(B1, 4, 4),
        "held": held_object(clock()),
        "build": [],
        "align_limits": {
            "xy_tolerance_m": 0.0004,
            "yaw_tolerance_deg": 1.0,
            "workspace_min_m": [0, -0.5, 0],
            "workspace_max_m": [0.8, 0.5, 0.6],
            "limits_source": "tests/config (not a station calibration)",
            "max_obs_age_s": 1.0,
        },
        "press_limits": {
            "max_travel_m": 0.0105,
            "speed_m_s": 0.005,
            "timeout_s": 5.0,
            "force_limit_N": 15.0,
            "limits_source": "tests/config",
        },
        "release": {"open_width_m": 0.03, "retreat_height_m": 0.01},
        "verify_window": {"window_s": 0.5, "min_samples": 3, "max_obs_age_s": 2.0},
    }
    d.update(over)
    return d


def station(clock, **kw):
    contact = FakeContact().describe()
    return FakeStation(
        clock,
        tcp=TCP0,
        t_tcp_object=held_object(0)["T_TCP_object"],
        held_id=B1,
        contact_description=contact,
        **kw,
    )


def context_factory(st, clock):
    def make(node_id, request_id, timeout_s):
        ctx = SkillContext(
            task_id="t-1",
            request_id=request_id,
            capabilities=frozenset(st.capabilities),
            primitive=st.primitive,
            clock=clock,
        )
        ctx.contact_description = st.contact_description
        ctx.sleep = clock.advance
        return ctx

    return make


def episode(clock, st, **over):
    return run_graph(
        CANDIDATE_GRAPH, REGISTRY, inputs(clock, **over), context_factory(st, clock), clock=clock
    )


def nodes(ep):
    return [(s["node"], s["status"], s["failure_type"]) for s in ep["steps"]]


# ------------------------------------------------------------------ K and G load


def test_skills_satisfy_the_task_skill_protocol():
    for s in SKILLS:
        assert s.kind in ("tool", "skill") and s.readiness == "unit-tested"
        assert s.input_schema["additionalProperties"] is False
        assert set(s.requires) <= {
            "pose_feedback",
            "cartesian_motion",
            "stop",
            "gripper",
            "gripper_feedback",
            "object_feedback",
            "object_pose_feedback",
            "contact_force_feedback",
            "compliant_motion",
        }
    assert REGISTRY["lego.guarded_seat"].requires >= {"contact_force_feedback", "compliant_motion"}


def test_candidate_graph_loads_against_K():
    g_loaded = load_graph(CANDIDATE_GRAPH, REGISTRY)
    assert g_loaded["graph_version"] == "rsh-graph/1"


# ------------------------------------------------------------------ episodes


def test_happy_path_episode_on_fakes():
    clock = FakeClock()
    st = station(clock)
    ep = episode(clock, st)
    assert ep["status"] == "succeeded" and ep["terminal"] == "done", nodes(ep)
    assert [n for n, _, _ in nodes(ep)] == ["align", "seat", "release", "verify"]
    for step in ep["steps"]:
        validate("SkillResult", step["result"])
        assert step["result"]["evidence"]["postcondition"]["passed"] is True
        assert step["result"]["observations"]
    assert ep["privileged_inputs"] is False
    caps = [c for c, _ in st.calls]
    assert caps.index("guarded_press") < caps.index("set_gripper")


def test_backend_without_contact_capability_stops_before_pressing():
    clock = FakeClock()
    st = station(clock)
    st.capabilities -= {"contact_force_feedback", "compliant_motion"}
    ep = episode(clock, st)
    assert ep["terminal"] == "stop_replan"
    assert nodes(ep)[-1] == ("seat", "rejected", "capability_missing")
    assert "guarded_press" not in [c for c, _ in st.calls]
    assert "set_gripper" not in [c for c, _ in st.calls]


def test_contact_signal_undeclared_is_contact_control_unavailable():
    clock = FakeClock()
    st = station(clock)
    st.contact_description = None  # capabilities claimed, signal/controller not declared
    ep = episode(clock, st)
    assert nodes(ep)[-1] == ("seat", "rejected", "capability_missing")
    assert "guarded_press" not in [c for c, _ in st.calls]


def test_insufficient_support_rejected_before_any_motion():
    clock = FakeClock()
    st = station(clock)
    target = mating_target(B1, 4, 4, layer0=1)
    ep = episode(clock, st, target=target)
    assert ep["terminal"] == "stop_replan"
    assert nodes(ep)[0] == ("align", "rejected", "lego.insufficient_support")
    assert "move_pose" not in [c for c, _ in st.calls]


def test_damaging_the_build_rejected_before_any_motion():
    clock = FakeClock()
    nb = mating_target("n1", 6, 4)
    st = station(clock, placed={"n1": nb["mating_pose"]})
    ep = episode(clock, st, build=[nb])
    assert nodes(ep)[0] == ("align", "rejected", "lego.would_damage_build")
    assert "move_pose" not in [c for c, _ in st.calls]


def test_invalid_held_object_fails_the_pre_fact():
    clock = FakeClock()
    st = station(clock)
    ep = episode(
        clock, st, held=held_object(clock(), valid=False, invalidated_reason="slip_detected")
    )
    assert nodes(ep)[0] == ("align", "rejected", "precondition_unmet")
    assert st.calls == []


def test_tilted_grasp_routes_to_regrasp():
    clock = FakeClock()
    import math

    c, s = math.cos(math.radians(5)), math.sin(math.radians(5))
    tilt = [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]
    st = station(clock, scenario={"grasp_offset": tilt})
    ep = episode(clock, st)
    assert ep["terminal"] == "stop_regrasp"
    assert nodes(ep)[0] == ("align", "failed", "held_object_slipped")


@pytest.mark.parametrize(
    "scenario, terminal, last",
    [
        (
            {"press": {"final_shortfall_m": 0.0008}},
            "stop_human",
            ("seat", "failed", "lego.not_seated"),
        ),
        (
            {"press": {"final_shortfall_m": 0.0016}},
            "stop_human",
            ("seat", "failed", "lego.misaligned"),
        ),
        ({"stop_confirmed": False}, "stop_human", ("seat", "failed", "lego.stop_unconfirmed")),
        ({"holding_after_release": True}, "stop_human", ("release", "failed", "release_failed")),
        ({"lifted": True}, "connection_failed", ("verify", "failed", "lego.lifted_with_gripper")),
        (
            {"release_dz": 0.0009},
            "connection_failed",
            ("verify", "failed", "lego.lifted_with_gripper"),
        ),
        ({"slip_m": 0.0009}, "connection_failed", ("verify", "failed", "lego.not_seated")),
    ],
)
def test_failure_episodes_stop_with_their_reason(scenario, terminal, last):
    clock = FakeClock()
    st = station(clock, scenario=scenario)
    ep = episode(clock, st)
    assert ep["terminal"] == terminal, nodes(ep)
    assert nodes(ep)[-1] == last


def test_disturbed_neighbour_is_structure_damaged():
    clock = FakeClock()
    nb = mating_target("n1", 9, 4)
    st = station(
        clock, placed={"n1": nb["mating_pose"]}, scenario={"disturb": {"n1": (0.002, 0.0)}}
    )
    ep = episode(clock, st, build=[nb])
    assert ep["terminal"] == "connection_failed"
    assert nodes(ep)[-1] == ("verify", "failed", "lego.structure_damaged")


def test_coarse_observer_never_reaches_done():
    clock = FakeClock()
    st = station(clock, sigma_m=0.0054)  # a calibrated YAM top camera's RMS
    ep = episode(clock, st)
    assert ep["status"] != "succeeded"
    assert nodes(ep)[0][0] == "align" and nodes(ep)[0][1] == "unverified"
    assert "guarded_press" not in [c for c, _ in st.calls]


def test_mating_pose_convention_mismatch_rejected():
    clock = FakeClock()
    st = station(clock)
    t = mating_target(B1, 4, 4)
    # A body-centre mating pose (half a brick higher) is a convention error, not a target.
    t["mating_pose"]["position_m"][2] += 0.0048
    ep = episode(clock, st, target=t)
    assert nodes(ep)[0] == ("align", "rejected", "invalid_request")
    assert st.calls == []


# ------------------------------------------------------------------ gate and envelope


def test_gate_rejects_missing_capabilities_without_calling_run():
    clock = FakeClock()
    ctx = SkillContext(
        task_id="t",
        request_id="r",
        capabilities=frozenset({"pose_feedback"}),
        primitive=lambda *a: pytest.fail("primitive called"),
        clock=clock,
    )
    for s in SKILLS:
        out = run_task_skill(s, {}, ctx)
        assert out["status"] == "rejected" and out["failure_type"] == "capability_missing"


def test_seat_without_post_press_observation_is_downgraded_by_the_runtime():
    clock = FakeClock()
    st = station(clock)
    ep = episode(clock, st)
    al = ep["steps"][0]["result"]["outputs"]["alignment"]
    st.capabilities.discard("object_feedback")  # contact backend only, no observer
    ctx = context_factory(st, clock)("seat", "r-seat", 5.0)
    ctx.capabilities = frozenset(st.capabilities)
    out = run_task_skill(
        REGISTRY["lego.guarded_seat"],
        {"connection_id": "conn-1", "alignment": al, "press_limits": inputs(clock)["press_limits"]},
        ctx,
    )
    assert out["status"] == "unverified" and out["failure_type"] == "evidence_missing"


def test_alignment_evidence_is_not_reusable_for_another_connection():
    clock = FakeClock()
    st = station(clock)
    ep = episode(clock, st)
    al = copy.deepcopy(ep["steps"][0]["result"]["outputs"]["alignment"])
    ctx = context_factory(st, clock)("seat", "r-seat", 5.0)
    n = len(st.calls)
    out = run_task_skill(
        REGISTRY["lego.guarded_seat"],
        {"connection_id": "conn-2", "alignment": al, "press_limits": inputs(clock)["press_limits"]},
        ctx,
    )
    assert out["status"] == "rejected" and out["failure_type"] == "precondition_unmet"
    assert len(st.calls) == n


# ------------------------------------------------------------------ JSON artefacts


def test_written_json_matches_the_module(tmp_path):
    from roboskillhub.tasks.lego.manifest import write

    paths = {p.name: p for p in write(tmp_path)}
    m = json.loads(paths["skill-manifest.json"].read_text())
    assert [s["skill_id"] for s in m["skills"]] == [s.skill_id for s in SKILLS]
    assert m == json.loads(json.dumps(manifest()))
    seat = m["skills"][1]
    assert (
        seat["executable"] is False and "NOT compliant force control" in seat["control_statement"]
    )
    for name in ("candidate-graph.json", "skill-manifest.json", "source-provenance.json"):
        example = REPO / "examples/tasks/lego" / name
        assert json.loads(example.read_text()) == json.loads(paths[name].read_text()), name


def test_provenance_hashes_match_the_reused_files():
    import hashlib

    for src in manifest()["provenance"]:
        if src.get("sha256"):
            path = REPO / src["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == src["sha256"], src["path"]


def test_geometry_check_of_the_fixture_target():
    t = mating_target(B1, 4, 4)
    assert g.validate_pose(t["mating_pose"], WORLD)
