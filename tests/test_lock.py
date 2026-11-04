from __future__ import annotations

import json

from conftest import write

from robomotif.catalog import load_catalog
from robomotif.lock import build_lock, graph_references, graph_sha256


def workflow(*, extra_tool: str | None = None) -> dict:
    nodes = {"check": {"type": "tool", "tool": "judge.check"}}
    if extra_tool:
        nodes["peek"] = {"type": "tool", "tool": extra_tool}
    return {
        "version": 3,
        "nodes": {
            "grab": {"type": "subgraph"},
            "decide": {"type": "subgraph"},
            "abort": {"type": "end", "status": "failure", "recovery": [{"tool": "robot.open_gripper"}]},
        },
        "subgraphs": {
            "grab_sg": {"skill": "grabbing-things", "nodes": {}},
            "decide_sg": {"skill": "generic", "nodes": nodes},
        },
    }


def test_references_cover_subgraph_skills_tool_nodes_and_recoveries():
    skills, tools = graph_references(workflow())
    assert skills == {"grabbing-things", "generic"}
    assert tools == {"judge.check", "robot.open_gripper"}


def test_lock_pins_bundles_and_lists_what_the_robot_must_provide(hub, tmp_path):
    graph = tmp_path / "graph"
    write(graph / "workflow.json", json.dumps(workflow()))
    lock = build_lock(graph, load_catalog(hub))
    assert sorted(lock["bundles"]) == ["grabbing-things", "judge"]
    assert lock["bundles"]["grabbing-things"]["source"] == "upstream"
    assert lock["sources"] == {"upstream": {"repo": "https://example.invalid/up", "commit": "0" * 40}}
    assert lock["graph"]["generated_skills"] == ["generic"]
    assert lock["connector_tools"] == ["robot.close_gripper", "robot.open_gripper"]
    assert lock["deployable"] is True
    assert len(lock["bundles"]["judge"]["content_sha256"]) == 64


def test_a_ground_truth_tool_makes_the_graph_undeployable(hub, tmp_path):
    graph = tmp_path / "graph"
    write(graph / "workflow.json", json.dumps(workflow(extra_tool="sim.get_object_pose")))
    lock = build_lock(graph, load_catalog(hub))
    assert lock["privileged_tools"] == ["sim.get_object_pose"]
    assert lock["deployable"] is False


def test_graph_digest_follows_workflow_and_scripts_only(tmp_path):
    graph = tmp_path / "graph"
    write(graph / "workflow.json", "{}")
    write(graph / "scripts/a.py", "x = 1\n")
    before = graph_sha256(graph)
    write(graph / "skills.lock", "{}")
    write(graph / "trace/run.json", "{}")
    assert graph_sha256(graph) == before
    write(graph / "scripts/a.py", "x = 2\n")
    assert graph_sha256(graph) != before
