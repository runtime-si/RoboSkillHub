"""A small versioned policy-graph interpreter (``rsh-graph/1``).

A graph G binds task skills of a fixed library K by id and version. Nodes carry
parameters (``$input``/``$state``/``$last`` path expressions only, never code),
named pre/postcondition facts, a timeout, an attempt budget and edges keyed by
terminal status, optionally refined by failure type. Target layouts and mating
graphs are inputs to G, not part of it. Every run is bounded by the graph's
step, attempt and time budgets.
"""

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from .contracts import CONTRACT_VERSION, STATUSES, Rejected
from .task import (
    SkillContext,
    TaskSkill,
    known_failure_type,
    make_result,
    run_task_skill,
)

GRAPH_VERSION = "rsh-graph/1"
_NODE_KEYS = {
    "skill",
    "version",
    "params",
    "pre",
    "post",
    "timeout_s",
    "attempts",
    "retry_on",
    "max_visits",
    "assign",
    "on",
    "note",
}


class GraphError(Rejected):
    default_failure = "graph_error"


def _edge_status(key: str) -> str:
    return key.split(":", 1)[0]


def _cycles_without_visit_budget(nodes: Mapping[str, dict]) -> list[list[str]]:
    """Strongly connected components (with a cycle) none of whose nodes has max_visits."""
    graph = {n: {t for t in spec["on"].values() if t in nodes} for n, spec in nodes.items()}
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    found: list[list[str]] = []
    counter = [0]

    def visit(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in graph[v]:
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            cyclic = len(component) > 1 or v in graph[v]
            if cyclic and not any("max_visits" in nodes[n] for n in component):
                found.append(sorted(component))

    for v in graph:
        if v not in index:
            visit(v)
    return found


def load_graph(spec: Mapping, registry: Mapping[str, TaskSkill]) -> dict:
    """Validate a graph against K before anything runs; raise GraphError on any defect."""
    errors: list[str] = []
    g = copy.deepcopy(dict(spec))
    for key in ("graph_id", "version", "contract_version", "skills", "start", "nodes", "terminals"):
        if key not in g:
            errors.append(f"graph lacks {key}")
    if errors:
        raise GraphError("; ".join(errors))
    if g["contract_version"] != CONTRACT_VERSION:
        errors.append(f"contract_version must be {CONTRACT_VERSION}")
    if g.get("graph_version", GRAPH_VERSION) != GRAPH_VERSION:
        errors.append(f"graph_version must be {GRAPH_VERSION}")
    budgets = g.setdefault("budgets", {})
    budgets.setdefault("max_steps", 100)
    budgets.setdefault("max_total_attempts", 50)
    budgets.setdefault("timeout_s", 600.0)
    for key, value in budgets.items():
        if key not in ("max_steps", "max_total_attempts", "timeout_s"):
            errors.append(f"unknown budget {key}")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"budget {key} must be positive")
    terminals = g["terminals"]
    if not isinstance(terminals, dict) or not terminals:
        errors.append("terminals must map names to statuses")
        terminals = {}
    for name, status in terminals.items():
        if status not in STATUSES:
            errors.append(f"terminal {name} has unknown status {status!r}")
    nodes = g["nodes"]
    if not isinstance(nodes, dict) or not nodes:
        raise GraphError("nodes must be a nonempty object")
    if set(nodes) & set(terminals):
        errors.append("node and terminal names overlap")
    if g["start"] not in nodes:
        errors.append(f"start {g['start']!r} is not a node")
    for skill_id, version in g["skills"].items():
        bound = registry.get(skill_id)
        if bound is None:
            errors.append(f"skill {skill_id} is not in K")
        elif bound.version != version:
            errors.append(f"skill {skill_id} is {bound.version} in K, graph binds {version}")
    for name, node in nodes.items():
        unknown = set(node) - _NODE_KEYS
        if unknown:
            errors.append(f"node {name}: unknown keys {sorted(unknown)}")
        skill = node.get("skill")
        if skill not in g["skills"]:
            errors.append(f"node {name}: skill {skill!r} is not bound in graph.skills")
        elif node.get("version", g["skills"][skill]) != g["skills"][skill]:
            errors.append(f"node {name}: version differs from graph.skills")
        node.setdefault("params", {})
        node.setdefault("pre", [])
        node.setdefault("post", [])
        node.setdefault("attempts", 1)
        node.setdefault("retry_on", [])
        node.setdefault("assign", {})
        for key in ("attempts", "max_visits"):
            if key in node and (not isinstance(node[key], int) or node[key] < 1):
                errors.append(f"node {name}: {key} must be a positive integer")
        if "timeout_s" in node and (
            not isinstance(node["timeout_s"], (int, float)) or node["timeout_s"] <= 0
        ):
            errors.append(f"node {name}: timeout_s must be positive")
        if set(node["retry_on"]) - {"failed", "unverified"}:
            errors.append(f"node {name}: retry_on may only hold failed/unverified")
        edges = node.get("on")
        if not isinstance(edges, dict):
            errors.append(f"node {name}: on must be an object")
            node["on"] = {}
            continue
        for status in STATUSES:
            if status not in edges:
                errors.append(f"node {name}: no edge for {status}")
        for key, target in edges.items():
            status = _edge_status(key)
            if status not in STATUSES:
                errors.append(f"node {name}: edge {key!r} has unknown status")
            if ":" in key and not known_failure_type(key.split(":", 1)[1]):
                errors.append(f"node {name}: edge {key!r} names an unknown failure_type")
            if target not in nodes and target not in terminals:
                errors.append(f"node {name}: edge {key} -> unknown {target!r}")
            if status in ("unverified", "rejected") and terminals.get(target) == "succeeded":
                errors.append(f"node {name}: {key} must not end in a succeeded terminal")
    for component in _cycles_without_visit_budget(nodes):
        errors.append(f"cycle {component} has no node with max_visits")
    if errors:
        raise GraphError("; ".join(errors))
    g["graph_version"] = GRAPH_VERSION
    return g


_MISSING = object()


def _lookup(scope: Mapping, expression: str) -> Any:
    root, _, path = expression[1:].partition(".")
    if root not in ("input", "state", "last"):
        raise GraphError(f"expression {expression!r} must start with $input, $state or $last")
    value: Any = scope[root]
    for part in path.split(".") if path else []:
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return _MISSING
    return value


def resolve(value: Any, scope: Mapping) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        found = _lookup(scope, value)
        if found is _MISSING:
            raise GraphError(f"{value} is not defined", failure_type="precondition_unmet")
        return copy.deepcopy(found)
    if isinstance(value, dict):
        return {k: resolve(v, scope) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, scope) for v in value]
    return value


def _fact(fact: Mapping, scope: Mapping) -> bool:
    found = _lookup(scope, fact["path"])
    if "equals" in fact:
        return found is not _MISSING and found == fact["equals"]
    if "exists" in fact:
        return (found is not _MISSING and found is not None) == bool(fact["exists"])
    return found is not _MISSING and bool(found)


def _assign(state: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        state = state.setdefault(part, {})
    state[parts[-1]] = value


ContextFactory = Callable[[str, str, float], SkillContext]
"""``(node_id, request_id, timeout_s) -> SkillContext`` supplied by the caller."""


def run_graph(
    graph: Mapping,
    registry: Mapping[str, TaskSkill],
    inputs: Mapping,
    make_context: ContextFactory,
    *,
    task_id: str | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_step: Callable[[dict], None] | None = None,
) -> dict:
    """Execute a loaded graph and return the episode record."""
    g = graph if graph.get("graph_version") == GRAPH_VERSION else load_graph(graph, registry)
    task_id = task_id or f"task-{uuid.uuid4().hex[:8]}"
    budgets = g["budgets"]
    started = clock()
    scope: dict[str, Any] = {"input": copy.deepcopy(dict(inputs)), "state": {}, "last": {}}
    visits: dict[str, int] = {}
    steps: list[dict] = []
    attempts_used = 0
    primitive_calls = 0
    privileged = False
    node_id = g["start"]
    outcome: dict[str, Any] = {}

    def finish(status: str, failure_type: str | None, terminal: str | None, reason: str) -> dict:
        return {
            "status": status,
            "failure_type": None if status == "succeeded" else failure_type,
            "terminal": terminal,
            "reason": reason,
        }

    while True:
        if node_id in g["terminals"]:
            status = g["terminals"][node_id]
            last = steps[-1]["result"] if steps else {}
            outcome = finish(status, last.get("failure_type") or "graph_error", node_id, "terminal")
            break
        if len(steps) >= budgets["max_steps"]:
            outcome = finish("failed", "step_budget_exhausted", None, "max_steps reached")
            break
        if clock() - started >= budgets["timeout_s"]:
            outcome = finish("failed", "timeout", None, "graph timeout_s reached")
            break
        node = g["nodes"][node_id]
        visits[node_id] = visits.get(node_id, 0) + 1
        skill = registry[node["skill"]]
        step: dict[str, Any] = {"node": node_id, "visit": visits[node_id], "skill": skill.skill_id}
        result: dict
        if "max_visits" in node and visits[node_id] > node["max_visits"]:
            result = make_result(
                skill, "failed", failure_type="retry_budget_exhausted", task_id=task_id
            )
        else:
            try:
                params = resolve(node["params"], scope)
                unmet = [f for f in node["pre"] if not _fact(f, scope)]
                if unmet:
                    raise GraphError(
                        f"preconditions unmet: {unmet}", failure_type="precondition_unmet"
                    )
            except Rejected as exc:
                params = None
                result = make_result(
                    skill,
                    "rejected",
                    phase="validate",
                    failure_type=exc.failure_type,
                    error={"code": type(exc).__name__, "message": str(exc)},
                    task_id=task_id,
                )
            if params is not None:
                attempt = 0
                while True:
                    attempt += 1
                    if attempts_used >= budgets["max_total_attempts"]:
                        result = make_result(
                            skill, "failed", failure_type="retry_budget_exhausted", task_id=task_id
                        )
                        break
                    attempts_used += 1
                    request_id = f"{task_id}/{node_id}/v{visits[node_id]}a{attempt}"
                    remaining = budgets["timeout_s"] - (clock() - started)
                    timeout = min(node.get("timeout_s", remaining), remaining)
                    ctx = make_context(node_id, request_id, timeout)
                    t0 = clock()
                    result = run_task_skill(skill, params, ctx)
                    primitive_calls += len(ctx.children)
                    if clock() - t0 > timeout and result["status"] == "succeeded":
                        result["status"] = "failed"
                        result["failure_type"] = "timeout"
                    retry = result["status"] in node["retry_on"] and attempt < node["attempts"]
                    step.setdefault("attempts", []).append(
                        {"request_id": request_id, "status": result["status"]}
                    )
                    if not retry:
                        break
                scope["last"][node_id] = {
                    "status": result["status"],
                    "failure_type": result["failure_type"],
                    "outputs": result.get("outputs", {}),
                    "evidence": result.get("evidence", {}),
                }
                if result["status"] == "succeeded":
                    failed_post = [f for f in node["post"] if not _fact(f, scope)]
                    if failed_post:
                        result["status"] = "failed"
                        result["failure_type"] = "postcondition_failed"
                        result["graph_postcondition_failed"] = failed_post
                        scope["last"][node_id]["status"] = "failed"
                    else:
                        for key, expression in node["assign"].items():
                            _assign(scope["state"], key, resolve(expression, scope))
        privileged = privileged or bool(result.get("privileged_inputs"))
        status, failure = result["status"], result.get("failure_type")
        edges = node["on"]
        target = edges.get(f"{status}:{failure}") or edges[status]
        step.update(
            status=status,
            failure_type=failure,
            next=target,
            result=result,
            elapsed_s=clock() - started,
        )
        steps.append(step)
        if on_step is not None:
            on_step(step)
        node_id = target

    return {
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "graph_version": GRAPH_VERSION,
        "task_id": task_id,
        "graph_id": g["graph_id"],
        "graph": g["version"],
        "K": dict(g["skills"]),
        **outcome,
        "steps": steps,
        "counters": {
            "steps": len(steps),
            "attempts": attempts_used,
            "primitive_calls": primitive_calls,
        },
        "privileged_inputs": privileged,
        "elapsed_s": clock() - started,
        "final_state": scope["state"],
    }
