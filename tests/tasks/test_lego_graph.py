"""Candidate graph: guards, bounded retries, one deadline, failure reasons kept, and an
end-to-end dry run of align -> seat -> verify on fakes (unit_or_dryrun, no physics)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from lego_fixtures import align_request, obs, shifted

from roboskillhub.tasks.lego import align_connection, guarded_seat, verify_connection
from roboskillhub.tasks.lego import geometry as g
from roboskillhub.tasks.lego.fakes import (
    FakeClock,
    FakeContact,
    FakeMotion,
    FakeObserver,
    press_trace,
)
from roboskillhub.tasks.lego.graph import CANDIDATE_GRAPH, run_graph
from roboskillhub.tasks.lego.manifest import manifest

REPO = Path(__file__).resolve().parents[2]


def scripted(**per_node):
    """Nodes returning scripted statuses; ``per_node[name]`` is a list consumed in order."""
    calls = {k: 0 for k in CANDIDATE_GRAPH["nodes"]}

    def make(name):
        def node(ctx):
            seq = per_node.get(name, [("succeeded", "ok")])
            out = seq[min(calls[name], len(seq) - 1)]
            calls[name] += 1
            return {"status": out[0], "reason_code": out[1]}

        return node

    return {k: make(k) for k in CANDIDATE_GRAPH["nodes"]}, calls


def test_happy_path_reaches_done():
    nodes, calls = scripted()
    out = run_graph(nodes, clock=FakeClock())
    assert out["terminal"] == "done" and out["status"] == "succeeded"
    assert [h["node"] for h in out["history"]] == [
        "confirm_grasp",
        "align",
        "seat",
        "release_retreat",
        "verify",
    ]


def test_alignment_failure_blocks_the_press():
    nodes, calls = scripted(align=[("rejected", "insufficient_support")])
    out = run_graph(nodes, clock=FakeClock())
    assert out["terminal"] == "halt_replan" and calls["seat"] == 0
    assert out["first_failure"]["reason_code"] == "insufficient_support"


def test_align_retry_budget_is_bounded_and_keeps_the_first_reason():
    nodes, calls = scripted(align=[("failed", "alignment_not_converged")])
    out = run_graph(nodes, clock=FakeClock())
    assert calls["align"] == 2  # one try + retry_budget 1
    assert out["terminal"] == "halt_human" and calls["seat"] == 0
    assert out["first_failure"] == {
        "node": "align",
        "status": "failed",
        "reason_code": "alignment_not_converged",
    }
    assert out["history"][-1]["reason_code"] == "retry_budget_exhausted"


def test_unknown_verification_never_proceeds_to_next_brick():
    nodes, calls = scripted(verify=[("unverified", "resolution_insufficient")])
    out = run_graph(nodes, clock=FakeClock())
    assert out["terminal"] == "halt_human" and out["status"] == "failed"
    assert calls["verify"] == 2


def test_verified_failure_does_not_invent_a_repair():
    nodes, _ = scripted(verify=[("failed", "partial_engagement")])
    out = run_graph(nodes, clock=FakeClock())
    assert out["terminal"] == "halt_repair_unsupported"


def test_seat_rejection_has_no_position_only_fallback():
    nodes, calls = scripted(seat=[("rejected", "missing_capabilities")])
    out = run_graph(nodes, clock=FakeClock())
    assert out["terminal"] == "halt_replan" and calls["release_retreat"] == 0


def test_release_uncertain_never_releases_blind_into_verify():
    nodes, calls = scripted(release_retreat=[("unverified", "holding_state_unknown")])
    out = run_graph(nodes, clock=FakeClock())
    assert out["terminal"] == "halt_human" and calls["verify"] == 0


def test_retries_spend_the_single_deadline():
    clock = FakeClock()

    def slow_align(ctx):
        clock.advance(70.0)
        return {"status": "failed", "reason_code": "alignment_not_converged"}

    nodes, _ = scripted()
    nodes["align"] = slow_align
    out = run_graph(nodes, clock=clock)
    assert out["terminal"] == "halt_human"
    assert out["elapsed_s"] < 2 * CANDIDATE_GRAPH["deadline_s"]
    assert out["first_failure"]["reason_code"] == "alignment_not_converged"


def test_every_edge_targets_a_node_and_every_node_has_a_fallback():
    names = set(CANDIDATE_GRAPH["nodes"])
    for e in CANDIDATE_GRAPH["edges"]:
        assert e["from"] in names and e["to"] in names
    for n, spec in CANDIDATE_GRAPH["nodes"].items():
        if "terminal" not in spec:
            assert any(e["from"] == n and e["on"] == "*" for e in CANDIDATE_GRAPH["edges"]), n


def test_end_to_end_dry_run_on_fakes():
    """The three skills chained by the candidate graph on scripted ports. Proves wiring and
    evidence hand-off only: no physics, no simulator, no robot."""
    clock = FakeClock()
    req = align_request(clock)
    motion = FakeMotion(clock)
    observer = FakeObserver(clock, motion, req["held"]["T_TCP_object"])
    seated: dict = {}

    def confirm(ctx):
        return {"status": "succeeded", "reason_code": "held"}

    def align(ctx):
        r = copy.deepcopy(req)
        t = clock()
        r["held"]["holding_evidence"]["timestamp_s"] = t
        r["tcp_observation"]["timestamp_s"] = t
        return align_connection(r, motion, observer, clock=clock)

    def seat(ctx):
        trace = press_trace(0.01, g.SYSTEMS["lego"].stud_height_m)
        return guarded_seat(
            {
                "request_id": "req-s",
                "connection_id": "conn-1",
                "system": "lego",
                "alignment": ctx["results"]["align"],
                "axis": ctx["results"]["align"]["evidence"]["plan"]["engagement_axis"],
                "max_travel_m": 0.0105,
                "speed_m_s": 0.005,
                "timeout_s": 5,
                "force_limit_N": 15,
                "limits_source": "test",
            },
            FakeContact(samples=trace),
        )

    def release(ctx):
        seated["t"] = clock()
        clock.advance(0.1)
        return {"status": "succeeded", "reason_code": "released"}

    def verify(ctx):
        seat_pose = ctx["results"]["align"]["evidence"]["plan"]["seat_pose"]
        samples = []
        for i in range(3):
            clock.advance(0.3)
            samples.append(
                obs(shifted(seat_pose, dz=0.0001), clock(), source="overhead-depth", obs_id=f"v{i}")
            )
        return verify_connection(
            {
                "request_id": "req-v",
                "connection_id": "conn-1",
                "brick_id": "b1",
                "brick": req["brick"],
                "plate": req["plate"],
                "target_cell": req["target_cell"],
                "executor_id": "fake-contact",
                "released_at_s": seated["t"],
                "release": obs(
                    {"gripper_open": True, "retreated": True},
                    seated["t"] + 0.05,
                    source="gripper-encoder",
                    frame_id="tcp",
                ),
                "observations": samples,
            },
            clock=clock,
        )

    out = run_graph(
        {
            "confirm_grasp": confirm,
            "align": align,
            "seat": seat,
            "release_retreat": release,
            "verify": verify,
        },
        clock=clock,
    )
    assert out["terminal"] == "done", out
    assert [h["status"] for h in out["history"]] == ["succeeded"] * 5


def test_written_json_matches_the_module(tmp_path):
    from roboskillhub.tasks.lego.manifest import write

    paths = {p.name: p for p in write(tmp_path)}
    m = json.loads(paths["skill-manifest.json"].read_text())
    assert [s["skill_id"] for s in m["skills"]] == [
        "lego.align_connection",
        "lego.guarded_seat",
        "lego.verify_connection",
    ]
    assert m == json.loads(json.dumps(manifest()))
    for s in m["skills"]:
        assert s["evidence_level"] == "unit_or_dryrun"
    seat = m["skills"][1]
    assert (
        seat["executable"] is False and "NOT compliant force control" in seat["control_statement"]
    )
    example = REPO / "examples/tasks/lego/candidate-graph.json"
    assert json.loads(example.read_text()) == json.loads(json.dumps(CANDIDATE_GRAPH))


def test_provenance_hashes_match_the_reused_files():
    import hashlib

    for src in manifest()["provenance"]:
        if src.get("sha256"):
            path = REPO / src["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == src["sha256"], src["path"]
