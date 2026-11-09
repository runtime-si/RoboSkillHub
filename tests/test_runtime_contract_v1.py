"""rsh-task/1: new primitives, failure types, task lease, task-skill gate and graphs.

Uses a kinematic fake station: it proves contract logic, not physical behaviour.
"""

from __future__ import annotations

import math
import threading
import time

import pytest

from roboskillhub.runtime import api
from roboskillhub.runtime.contracts import (
    SCHEMAS,
    Limits,
    compose,
    invert,
    tcp_for_object_goal,
)
from roboskillhub.runtime.engine import Runtime
from roboskillhub.runtime.graph import GraphError, load_graph, run_graph
from roboskillhub.runtime.task import (
    CONTRACT,
    SkillContext,
    make_result,
    offline_context,
    primitive_skill_context,
    run_task_skill,
    validate,
)

BASE = ["pose_feedback", "cartesian_motion", "gripper", "stop", "object_feedback"]


def pose(x=0.3, y=0.0, z=0.2, frame="world", q=(0.0, 0.0, 0.0, 1.0)):
    return {"frame_id": frame, "position_m": [x, y, z], "quaternion_xyzw": list(q)}


class FakeStation:
    """Kinematic stand-in: arrives exactly, carries a held object rigidly."""

    def __init__(self, capabilities=BASE, *, holding_feedback=True, release_sticks=False):
        self.capabilities = list(capabilities)
        self.tcp = pose()
        self.width = 0.08
        self.held: str | None = None
        self.T_TCP_held = None
        self.objects = {"d0": pose(0.4, 0.0, 0.024)}
        self.holding_feedback = holding_feedback
        self.release_sticks = release_sticks
        self.slip = False
        self.calls: list[str] = []
        self.object_extra: dict = {}

    def describe(self):
        return {"frame_id": "world", "capabilities": self.capabilities, "backend": "fake"}

    def get_state(self, *, timeout_s):
        now = time.time() + 1e-4
        objects = {}
        for name, p in self.objects.items():
            if name == self.held and not self.slip:
                p = compose(self.tcp, self.T_TCP_held)
                self.objects[name] = p
            objects[name] = {
                "position_m": p["position_m"],
                "quaternion_xyzw": p["quaternion_xyzw"],
                "frame_id": "world",
                "timestamp_s": now,
                "source": "fake_tracker",
                **self.object_extra.get(name, {}),
            }
        gripper = {"width_m": self.width, "source": "fake_encoder", "timestamp_s": now}
        if self.holding_feedback:
            gripper["holding"] = self.held is not None
        return {"pose": self.tcp, "timestamp_s": now, "objects": objects, "gripper": gripper}

    def move_pose(self, target, *, speed_m_s, timeout_s, linear):
        self.calls.append("move")
        self.tcp = target
        return {"ok": True}

    def set_gripper(self, width_m, *, timeout_s):
        self.calls.append(f"gripper:{width_m}")
        self.width = width_m
        if width_m >= 0.05 and not self.release_sticks:
            self.held = None
        elif width_m < 0.05 and self.held is None:
            obj = self.objects["d0"]
            if math.dist(obj["position_m"], self.tcp["position_m"]) < 0.03:
                self.held = "d0"
                self.T_TCP_held = compose(invert(self.tcp, frame_id="tcp"), obj, frame_id="tcp")
                self.width = 0.024
        return {"ok": True}

    def stop(self):
        self.calls.append("stop")


@pytest.fixture
def limits():
    return Limits.from_dict(
        {
            "frame_id": "world",
            "workspace_min_m": [-1, -1, 0],
            "workspace_max_m": [1, 1, 1],
            "max_speed_m_s": 0.2,
            "max_contact_distance_m": 0.3,
            "gripper_min_width_m": 0.0,
            "gripper_max_width_m": 0.1,
        }
    )


def test_contract_json_is_the_published_v1():
    assert CONTRACT["contract_version"] == "rsh-task/1"
    assert {"set_gripper", "move_waypoints", "observe_objects"} <= set(SCHEMAS)
    assert set(CONTRACT["primitives"]) == set(SCHEMAS)


def test_transform_helpers_round_trip():
    T_W_tcp = pose(0.3, 0.1, 0.2, q=(0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)))
    T_TCP_obj = pose(0.0, 0.0, 0.05, frame="tcp")
    T_W_obj = compose(T_W_tcp, T_TCP_obj)
    back = tcp_for_object_goal(T_W_obj, T_TCP_obj)
    assert math.dist(back["position_m"], T_W_tcp["position_m"]) < 1e-9
    assert (
        abs(
            abs(
                sum(
                    a * b
                    for a, b in zip(
                        back["quaternion_xyzw"], T_W_tcp["quaternion_xyzw"], strict=True
                    )
                )
            )
            - 1
        )
        < 1e-9
    )


def test_legacy_results_gain_failure_type_without_losing_fields(limits):
    rt = Runtime(FakeStation(capabilities=["pose_feedback"]), limits)
    result = rt.invoke("move_pose", {"target_pose": pose()})
    assert result["status"] == "rejected"
    assert result["failure_type"] == "capability_missing"
    assert result["contract_version"] == "rsh-task/1"
    for key in (
        "schema_version",
        "invocation_id",
        "capability",
        "skill",
        "version",
        "evidence",
        "trace",
    ):
        assert key in result
    out = Runtime(FakeStation(), rt.limits).invoke("move_pose", {"target_pose": pose(x=5)})
    assert out["failure_type"] == "unreachable"


def test_release_requires_holding_feedback_false(limits):
    station = FakeStation()
    rt = Runtime(station, limits)
    station.tcp = pose(0.4, 0.0, 0.024)
    closed = rt.invoke("set_gripper", {"width_m": 0.01, "purpose": "close"})
    assert closed["status"] == "succeeded", closed
    released = rt.invoke("set_gripper", {"width_m": 0.08, "purpose": "release", "target_id": "d0"})
    assert released["status"] == "succeeded"
    assert released["evidence"]["object_after"]["position_m"]
    station.release_sticks = True
    rt.invoke("set_gripper", {"width_m": 0.01, "purpose": "close"})
    stuck = rt.invoke("set_gripper", {"width_m": 0.08, "purpose": "release"})
    assert (stuck["status"], stuck["failure_type"]) == ("failed", "release_failed")
    assert "stop" in station.calls


def test_release_without_holding_feedback_is_unverified(limits):
    rt = Runtime(FakeStation(holding_feedback=False), limits)
    result = rt.invoke("set_gripper", {"width_m": 0.08, "purpose": "release"})
    assert (result["status"], result["failure_type"]) == ("unverified", "evidence_missing")


def test_close_on_nothing_is_grasp_missed(limits):
    rt = Runtime(FakeStation(), limits)
    result = rt.invoke("set_gripper", {"width_m": 0.01, "purpose": "close"})
    assert (result["status"], result["failure_type"]) == ("failed", "grasp_missed")
    assert result["recovery_hints"] == ["reobserve", "regrasp"]


def test_linear_waypoints_need_a_declared_linear_backend(limits):
    station = FakeStation()
    rt = Runtime(station, limits)
    result = rt.invoke("move_waypoints", {"waypoints": [pose()], "linear": True})
    assert (result["status"], result["failure_type"]) == ("rejected", "capability_missing")
    assert station.calls == []


def test_waypoints_are_all_checked_before_any_motion(limits):
    station = FakeStation()
    rt = Runtime(station, limits)
    result = rt.invoke("move_waypoints", {"waypoints": [pose(), pose(x=3)], "linear": False})
    assert result["status"] == "rejected" and station.calls == []


def test_held_transport_verifies_object_follows_tcp(limits):
    station = FakeStation()
    rt = Runtime(station, limits)
    station.tcp = pose(0.4, 0.0, 0.024)
    rt.invoke("set_gripper", {"width_m": 0.01, "purpose": "close"})
    held = {"object_id": "d0", "T_TCP_object": pose(0, 0, 0, frame="tcp")}
    request = {"waypoints": [pose(0.4, 0, 0.1), pose(0.5, 0, 0.1)], "linear": False, "held": held}
    ok = rt.invoke("move_waypoints", request)
    assert ok["status"] == "succeeded", ok
    assert ok["evidence"]["held"]["error_m"] < 1e-6
    station.slip = True
    slipped = rt.invoke("move_waypoints", {**request, "waypoints": [pose(0.2, 0, 0.1)]})
    assert (slipped["status"], slipped["failure_type"]) == ("failed", "held_object_slipped")


def test_observe_objects_statuses_and_privilege(limits):
    station = FakeStation()
    rt = Runtime(station, limits)
    ok = rt.invoke("observe_objects", {"object_ids": ["d0"]})
    assert ok["status"] == "succeeded"
    obs = ok["evidence"]["observations"][0]
    validate("Observation", obs)
    assert obs["privileged"] is False
    missing = rt.invoke("observe_objects", {"object_ids": ["d0", "ghost"]})
    assert (missing["status"], missing["failure_type"]) == ("unverified", "object_not_observed")
    assert missing["evidence"]["observations"][1]["reason"]
    station.object_extra["d0"] = {"source": "simulation_truth"}
    truth = rt.invoke("observe_objects", {"object_ids": ["d0"]})
    assert truth["evidence"]["observations"][0]["privileged"] is True
    no_pose = rt.invoke("observe_objects", {"object_ids": ["d0"], "require_orientation": True})
    assert (no_pose["status"], no_pose["failure_type"]) == ("rejected", "capability_missing")
    station.object_extra["d0"] = {"timestamp_s": time.time() - 10}
    stale = rt.invoke("observe_objects", {"object_ids": ["d0"]})
    assert (stale["status"], stale["failure_type"]) == ("unverified", "stale_observation")


class LeasedStation(FakeStation):
    def __init__(self):
        super().__init__()
        self.leases = []

    def begin_invocation(self, *, timeout_s):
        self.leases.append("begin")
        return {"ok": True}

    def end_invocation(self):
        self.leases.append("end")
        return {"ok": True}


def test_task_lease_covers_many_primitives_and_rejects_nesting(limits):
    station = LeasedStation()
    rt = Runtime(station, limits)
    with rt.task_lease("ep-1", 30) as lease:
        a = rt.invoke("move_pose", {"target_pose": pose()}, lease=lease, request_id="ep-1/n0")
        b = rt.invoke("observe_objects", {"object_ids": ["d0"]}, lease=lease)
        nested = rt.invoke("move_pose", {"target_pose": pose()})
        assert (nested["status"], nested["failure_type"]) == ("rejected", "nested_lease")
        busy: dict = {}
        t = threading.Thread(
            target=lambda: busy.update(
                rt.invoke("move_pose", {"target_pose": pose(), "timeout_s": 0.2})
            )
        )
        t.start()
        t.join()
        assert (busy["status"], busy["failure_type"]) == ("rejected", "station_busy")
    assert a["status"] == b["status"] == "succeeded"
    assert a["task_id"] == "ep-1" and a["request_id"] == "ep-1/n0"
    assert station.leases == ["begin", "end"]
    late = rt.invoke("move_pose", {"target_pose": pose()}, lease=lease)
    assert late["failure_type"] == "lease_expired"


def test_cancelled_lease_stops_and_rejects_later_calls(limits):
    station = FakeStation()
    rt = Runtime(station, limits)
    with rt.task_lease("ep-2", 30) as lease:
        lease.cancel("operator abort")
        result = rt.invoke("move_pose", {"target_pose": pose()}, lease=lease)
        assert (result["status"], result["failure_type"]) == ("rejected", "cancelled")
    assert station.calls == []


def test_api_describe_lists_requirements_and_contract(monkeypatch):
    monkeypatch.delenv("ROBOSKILLHUB_STATION", raising=False)
    data = api.describe()
    assert data["contract_version"] == "rsh-task/1"
    rows = {r["capability"]: r for r in data["capabilities"]}
    assert rows["move_waypoints"]["requires_if"]["linear=true"] == ["linear_cartesian_motion"]
    assert rows["observe_objects"]["configured"] is False


# --- task-skill gate --------------------------------------------------------


class PlaceSkill:
    skill_id = "domino.place_upright"
    version = "0.1.0"
    kind = "skill"
    requires = frozenset(
        {"pose_feedback", "cartesian_motion", "gripper", "stop", "object_feedback"}
    )
    input_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["object_id"],
        "properties": {"object_id": {"type": "string"}},
    }
    readiness = "unit-tested"

    def __init__(self, claim_without_evidence=False):
        self.claim_without_evidence = claim_without_evidence

    def run(self, request, ctx):
        obs = ctx.observe([request["object_id"]])
        if self.claim_without_evidence:
            return ctx.result(self, "succeeded", phase="verify")
        observations = obs["evidence"]["observations"]
        return ctx.result(
            self,
            "succeeded",
            phase="verify",
            evidence={"postcondition": {"name": "observed", "passed": True, "criterion": "seen"}},
            observations=observations,
        )


def test_task_skill_gate(limits):
    station = FakeStation()
    rt = Runtime(station, limits)
    with rt.task_lease("ep-3", 30) as lease:
        ctx = primitive_skill_context(rt, lease, request_id="ep-3/place")
        good = run_task_skill(PlaceSkill(), {"object_id": "d0"}, ctx)
        assert good["status"] == "succeeded" and good["children"]
        validate("SkillResult", good)
        ctx = primitive_skill_context(rt, lease, request_id="ep-3/bluff")
        bluff = run_task_skill(PlaceSkill(claim_without_evidence=True), {"object_id": "d0"}, ctx)
        assert (bluff["status"], bluff["failure_type"]) == ("unverified", "evidence_missing")
        bad = run_task_skill(PlaceSkill(), {"object": "d0"}, ctx)
        assert (bad["status"], bad["failure_type"]) == ("rejected", "invalid_request")
    blind = run_task_skill(PlaceSkill(), {"object_id": "d0"}, offline_context())
    assert (blind["status"], blind["failure_type"]) == ("rejected", "capability_missing")


# --- graph -----------------------------------------------------------------


class Scripted:
    """A tool whose outcomes are scripted, for graph semantics only."""

    kind = "tool"
    requires = frozenset()
    input_schema = {"type": "object"}
    readiness = "unit-tested"

    def __init__(self, skill_id, outcomes, version="1.0.0"):
        self.skill_id = skill_id
        self.version = version
        self.outcomes = list(outcomes)
        self.calls = 0

    def run(self, request, ctx):
        self.calls += 1
        status, failure = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        return ctx.result(self, status, failure_type=failure, outputs={"n": self.calls, **request})


def ctx_factory(node, request_id, timeout_s):
    return SkillContext(task_id="g", request_id=request_id)


def graph(nodes, skills, start="a"):
    return {
        "graph_id": "t",
        "version": "1.0.0",
        "contract_version": "rsh-task/1",
        "skills": skills,
        "start": start,
        "nodes": nodes,
        "terminals": {"done": "succeeded", "abort": "failed", "unsure": "unverified"},
        "budgets": {"max_steps": 20, "max_total_attempts": 10, "timeout_s": 60},
    }


ALL = {"succeeded": "done", "failed": "abort", "unverified": "unsure", "rejected": "abort"}


def test_graph_load_rejects_unsafe_edges_and_unbounded_cycles():
    k = {"x": Scripted("x", [("succeeded", None)])}
    with pytest.raises(GraphError, match="must not end in a succeeded"):
        load_graph(
            graph({"a": {"skill": "x", "on": {**ALL, "unverified": "done"}}}, {"x": "1.0.0"}), k
        )
    with pytest.raises(GraphError, match="no edge for rejected"):
        on = {k2: v for k2, v in ALL.items() if k2 != "rejected"}
        load_graph(graph({"a": {"skill": "x", "on": on}}, {"x": "1.0.0"}), k)
    with pytest.raises(GraphError, match="no node with max_visits"):
        load_graph(graph({"a": {"skill": "x", "on": {**ALL, "failed": "a"}}}, {"x": "1.0.0"}), k)
    with pytest.raises(GraphError, match="graph binds 2.0.0"):
        load_graph(graph({"a": {"skill": "x", "on": ALL}}, {"x": "2.0.0"}), k)
    with pytest.raises(GraphError, match="unknown failure_type"):
        load_graph(
            graph({"a": {"skill": "x", "on": {**ALL, "failed:oops": "abort"}}}, {"x": "1.0.0"}), k
        )


def test_graph_retries_then_routes_by_failure_type_and_exhausts_visits():
    x = Scripted(
        "x", [("failed", "grasp_missed"), ("failed", "grasp_missed"), ("failed", "lego.not_seated")]
    )
    nodes = {
        "a": {
            "skill": "x",
            "attempts": 2,
            "retry_on": ["failed"],
            "max_visits": 2,
            "on": {
                **ALL,
                "failed:lego.not_seated": "abort",
                "failed": "a",
                "failed:retry_budget_exhausted": "abort",
            },
        }
    }
    episode = run_graph(graph(nodes, {"x": "1.0.0"}), {"x": x}, {}, ctx_factory)
    assert episode["status"] == "failed"
    assert [s["failure_type"] for s in episode["steps"]] == ["grasp_missed", "lego.not_seated"]
    # The second visit retries lego.not_seated too: retry_on is keyed by status.
    assert episode["counters"]["attempts"] == 4


def test_graph_budget_and_state_assignment():
    x = Scripted("x", [("succeeded", None)])
    nodes = {
        "a": {
            "skill": "x",
            "params": {"slot": "$input.slots.0"},
            "assign": {"placed": "$last.a.outputs.slot"},
            "max_visits": 100,
            "on": {**ALL, "succeeded": "a"},
        }
    }
    spec = graph(nodes, {"x": "1.0.0"})
    spec["budgets"]["max_total_attempts"] = 50
    episode = run_graph(spec, {"x": x}, {"slots": ["s0"]}, ctx_factory)
    assert (episode["status"], episode["failure_type"]) == ("failed", "step_budget_exhausted")
    assert episode["final_state"]["placed"] == "s0"


def test_graph_missing_input_is_a_rejected_step_not_a_crash():
    x = Scripted("x", [("succeeded", None)])
    nodes = {"a": {"skill": "x", "params": {"slot": "$input.nope"}, "on": ALL}}
    episode = run_graph(graph(nodes, {"x": "1.0.0"}), {"x": x}, {}, ctx_factory)
    assert episode["steps"][0]["failure_type"] == "precondition_unmet"
    assert x.calls == 0 and episode["status"] == "failed"


def test_make_result_envelope_is_valid():
    skill = Scripted("x", [])
    validate(
        "SkillResult", make_result(skill, "failed", failure_type="domino.toppled_after_release")
    )
    with pytest.raises(Exception, match="vocabulary"):
        validate("SkillResult", make_result(skill, "failed", failure_type="made_up"))


def test_graph_success_branch_can_finish_a_loop():
    x = Scripted("x", [("succeeded", None)])
    nodes = {
        "a": {
            "skill": "x",
            "params": {"left": "$input.left"},
            "branch": [{"path": "$last.a.outputs.n", "equals": 3, "next": "done"}],
            "max_visits": 5,
            "on": {**ALL, "succeeded": "a"},
        }
    }
    episode = run_graph(graph(nodes, {"x": "1.0.0"}), {"x": x}, {"left": 1}, ctx_factory)
    assert (episode["status"], episode["terminal"], x.calls) == ("succeeded", "done", 3)
