"""Candidate local graph G for one LEGO connection, and a small bounded executor for it.

confirm grasp -> align -> guarded seat -> release & retreat -> verify -> next brick

This is a candidate, not a validated G, and not a global graph-search engine. Nodes are
injected callables returning a result dict with ``status`` and ``reason_code``. Edges are
matched in order on ``status`` or ``status:reason_code`` (``*`` matches anything). A
self-loop is a retry and spends the node's ``retry_budget``; every retry also spends the
single graph deadline, which is never reset.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

CANDIDATE_GRAPH: dict[str, Any] = {
    "graph_id": "lego.single_connection",
    "graph_version": "0.1.0",
    "status": "candidate",
    "validated_under": None,
    "start": "confirm_grasp",
    "deadline_s": 120.0,
    "nodes": {
        "confirm_grasp": {
            "skill": "perception_regrasp.confirm_grasp",
            "owner": "agent-01",
            "retry_budget": 0,
            "note": "independent evidence the brick is held with a known T_TCP_object",
        },
        "align": {"skill": "lego.align_connection", "owner": "agent-03", "retry_budget": 1},
        "seat": {"skill": "lego.guarded_seat", "owner": "agent-03", "retry_budget": 0},
        "release_retreat": {
            "skill": "runtime.release_and_retreat",
            "owner": "agent-05",
            "retry_budget": 0,
            "note": "no public release capability yet (integration request)",
        },
        "verify": {"skill": "lego.verify_connection", "owner": "agent-03", "retry_budget": 1},
        "done": {"terminal": "succeeded"},
        "halt_replan": {"terminal": "rejected", "note": "upstream must replan; nothing moved"},
        "halt_human": {"terminal": "failed", "note": "stop; state handed to a human / upper G"},
        "halt_unknown": {"terminal": "unverified", "note": "state unknown; do not continue"},
        "halt_repair_unsupported": {
            "terminal": "failed",
            "note": "connection failed; automatic disassembly/repair is not implemented",
        },
    },
    "edges": [
        {
            "from": "confirm_grasp",
            "on": "succeeded",
            "to": "align",
            "guard": "held object == connection's brick; T_TCP_object known",
        },
        {"from": "confirm_grasp", "on": "*", "to": "halt_replan"},
        {
            "from": "align",
            "on": "succeeded",
            "to": "seat",
            "guard": "fresh independent observation within xy/yaw tolerance",
        },
        {
            "from": "align",
            "on": "rejected",
            "to": "halt_replan",
            "guard": "error detected before approach: nothing moved",
        },
        {
            "from": "align",
            "on": "failed:alignment_not_converged",
            "to": "align",
            "guard": "retry from a re-observed state",
        },
        {
            "from": "align",
            "on": "unverified",
            "to": "align",
            "guard": "re-observe once; unknown is not success",
        },
        {"from": "align", "on": "*", "to": "halt_human"},
        {
            "from": "seat",
            "on": "succeeded",
            "to": "release_retreat",
            "guard": "operation_complete; connection still unverified",
        },
        {"from": "seat", "on": "unverified:seat_depth_without_threshold", "to": "release_retreat"},
        {
            "from": "seat",
            "on": "rejected",
            "to": "halt_replan",
            "guard": "missing contact capability: no position-only fallback",
        },
        {
            "from": "seat",
            "on": "*",
            "to": "halt_human",
            "guard": "stop; bounded retreat only with fresh state (not implemented)",
        },
        {"from": "release_retreat", "on": "succeeded", "to": "verify"},
        {
            "from": "release_retreat",
            "on": "*",
            "to": "halt_human",
            "guard": "holding/contact state uncertain: no blind release or lift",
        },
        {"from": "verify", "on": "succeeded", "to": "done"},
        {"from": "verify", "on": "unverified", "to": "verify", "guard": "re-observe once"},
        {"from": "verify", "on": "failed", "to": "halt_repair_unsupported"},
        {"from": "verify", "on": "*", "to": "halt_unknown"},
    ],
    "on_budget_exhausted": "halt_human",
    "on_deadline": "halt_human",
}


def _matches(on: str, status: str, reason: str) -> bool:
    if on == "*":
        return True
    if ":" in on:
        s, r = on.split(":", 1)
        return s == status and r == reason
    return on == status


def run_graph(
    nodes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    graph: dict[str, Any] = CANDIDATE_GRAPH,
    *,
    clock: Callable[[], float] = time.time,
    max_steps: int = 50,
) -> dict[str, Any]:
    """Run ``graph`` with ``nodes[name](ctx)``. ``ctx`` carries ``history``,
    ``remaining_s`` and the previous results by node. Returns the terminal node, its
    status, the first non-success reason (kept through recovery) and the history."""
    t0 = clock()
    deadline = t0 + float(graph["deadline_s"])
    spent: dict[str, int] = {}
    history: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    node = graph["start"]
    first_failure: dict[str, Any] | None = None
    for _ in range(max_steps):
        spec = graph["nodes"][node]
        if "terminal" in spec:
            return {
                "terminal": node,
                "status": spec["terminal"],
                "first_failure": first_failure,
                "history": history,
                "elapsed_s": clock() - t0,
            }
        if clock() >= deadline:
            node = graph["on_deadline"]
            first_failure = first_failure or {"node": "graph", "reason_code": "deadline_exceeded"}
            continue
        ctx = {
            "history": list(history),
            "results": dict(results),
            "remaining_s": deadline - clock(),
            "attempt": spent.get(node, 0),
        }
        out = nodes[node](ctx)
        status, reason = out.get("status", ""), out.get("reason_code", "")
        results[node] = out
        history.append(
            {
                "node": node,
                "status": status,
                "reason_code": reason,
                "attempt": spent.get(node, 0),
                "t_s": clock() - t0,
            }
        )
        if status != "succeeded" and first_failure is None:
            first_failure = {"node": node, "status": status, "reason_code": reason}
        nxt = next(
            (
                e["to"]
                for e in graph["edges"]
                if e["from"] == node and _matches(e["on"], status, reason)
            ),
            None,
        )
        if nxt is None:
            nxt = graph["on_budget_exhausted"]
        if nxt == node:
            spent[node] = spent.get(node, 0) + 1
            if spent[node] > int(spec.get("retry_budget", 0)):
                history.append(
                    {
                        "node": node,
                        "status": "budget_exhausted",
                        "reason_code": "retry_budget_exhausted",
                        "t_s": clock() - t0,
                    }
                )
                nxt = graph["on_budget_exhausted"]
        node = nxt
    return {
        "terminal": None,
        "status": "failed",
        "first_failure": first_failure or {"reason_code": "max_steps"},
        "history": history,
        "elapsed_s": clock() - t0,
    }
