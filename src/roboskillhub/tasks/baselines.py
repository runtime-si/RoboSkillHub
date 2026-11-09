"""Three baselines over one K and one perception setting, on paired seeds.

1. ``fixed_script``: the graph with no recovery edges (any non-success aborts).
2. ``online_replan``: the graph that re-observes, re-grasps and retries a target
   with a fresh piece after observed failures, within visit budgets.
3. ``simulation_search``: candidate graphs (same K, varied structure/parameters)
   are each rolled out on training seeds by an evaluator; the best is then run on
   the held-out test seeds. Its rollouts are counted and reported as cost.

An evaluator is any ``(graph, seed) -> episode`` callable. ``StationEnv`` builds
one from a world factory; what that world is (Isaac, a fake) is stated in its
``describe()`` and copied into every report, so results are never relabelled.
"""

from __future__ import annotations

import collections
import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from roboskillhub.runtime.engine import Runtime
from roboskillhub.runtime.graph import load_graph, run_graph
from roboskillhub.runtime.task import TaskSkill, primitive_skill_context

from . import graphs

Evaluator = Callable[[dict, int], dict]


class StationEnv:
    """Runs a graph on a fresh station per seed, under one task lease.

    ``world(seed)`` returns ``{"backend", "limits", "inputs", "describe"}``; it may
    also return ``"interventions"`` and ``"task_success"`` (callables read after the
    episode) and ``"independent"``: a callable run once after the episode, before
    the other two, whose verdict (judged without the skill results) is kept as
    ``episode["independent"]``.

    ``independent(episode, verdict, built)``, if given, replaces that default: its
    return value is kept instead (``verdict`` is None when the world has no
    ``"independent"``), e.g. to pair each step's own result with the evaluator's.
    """

    def __init__(
        self,
        world: Callable[[int], dict],
        registry: Mapping[str, TaskSkill],
        *,
        independent: Callable[[dict, Any, dict], Any] | None = None,
    ):
        self.world = world
        self.registry = registry
        self.independent = independent
        self.episodes_run = 0

    def __call__(self, graph: dict, seed: int) -> dict:
        built = self.world(seed)
        runtime = Runtime(built["backend"], built["limits"])
        loaded = load_graph(graph, self.registry)
        task_id = f"{graph['graph_id']}/seed{seed}"
        started = time.monotonic()
        with runtime.task_lease(task_id, loaded["budgets"]["timeout_s"]) as lease:

            def make_context(node: str, request_id: str, timeout_s: float):
                return primitive_skill_context(
                    runtime,
                    lease,
                    request_id=request_id,
                    privileged_allowed=built.get("privileged_allowed", False),
                )

            episode = run_graph(
                loaded, self.registry, built["inputs"], make_context, task_id=task_id
            )
        self.episodes_run += 1
        episode["seed"] = seed
        episode["environment"] = built["describe"]
        episode["graph_params"] = graph.get("params", {})
        episode["wall_s"] = time.monotonic() - started
        verdict = built["independent"]() if callable(built.get("independent")) else None
        interventions = built.get("interventions")
        episode["interventions"] = interventions() if callable(interventions) else 0
        if "task_success" in built:
            episode["task_success"] = built["task_success"]()
        if self.independent is not None:
            episode["independent"] = self.independent(episode, verdict, built)
        elif verdict is not None:
            episode["independent"] = verdict
        return episode


def summarize(episodes: Iterable[dict]) -> dict:
    episodes = list(episodes)
    n = len(episodes)
    success = sum(e["status"] == "succeeded" for e in episodes)
    independent = [e["task_success"] for e in episodes if "task_success" in e]
    failures = collections.Counter(
        e.get("failure_type") or "none" for e in episodes if e["status"] != "succeeded"
    )
    recovery_steps = [sum(1 for s in e["steps"] if s["status"] != "succeeded") for e in episodes]
    return {
        "episodes": n,
        "graph_successes": success,
        "graph_success": f"{success}/{n}",
        "independent_task_success": (
            f"{sum(bool(x) for x in independent)}/{len(independent)}" if independent else None
        ),
        "failure_types": dict(failures),
        "mean_steps": statistics.fmean(e["counters"]["steps"] for e in episodes) if n else 0,
        "mean_primitive_calls": (
            statistics.fmean(e["counters"]["primitive_calls"] for e in episodes) if n else 0
        ),
        "mean_failed_steps": statistics.fmean(recovery_steps) if n else 0,
        "interventions": sum(e.get("interventions", 0) for e in episodes),
        "mean_wall_s": statistics.fmean(e.get("wall_s", e["elapsed_s"]) for e in episodes)
        if n
        else 0,
        "privileged_inputs": any(e.get("privileged_inputs") for e in episodes),
    }


def _score(summary: dict) -> tuple:
    return (
        summary["graph_successes"],
        -summary["interventions"],
        -summary["mean_failed_steps"],
        -summary["mean_primitive_calls"],
        -summary["mean_wall_s"],
    )


def simulation_search(
    candidates: list[dict],
    evaluate: Evaluator,
    train_seeds: Iterable[int],
) -> dict:
    """Roll out every candidate on the training seeds; pick the best by the stated score."""
    started = time.monotonic()
    train_seeds = list(train_seeds)
    table = []
    for graph in candidates:
        episodes = [evaluate(graph, seed) for seed in train_seeds]
        table.append(
            {"graph_id": graph["graph_id"], "params": graph["params"], **summarize(episodes)}
        )
    best_index = max(range(len(table)), key=lambda i: _score(table[i]))
    return {
        "selected": candidates[best_index],
        "selected_params": table[best_index]["params"],
        "train_table": table,
        "search_rollouts": len(candidates) * len(train_seeds),
        "planning_s": time.monotonic() - started,
        "score": "graph successes, then fewer interventions, failed steps, primitive calls, time",
    }


def compare(
    task: str,
    evaluate: Evaluator,
    *,
    train_seeds: Iterable[int],
    test_seeds: Iterable[int],
    environment: Any,
    slots: int = 5,
) -> dict:
    """The three baselines on the same held-out seeds."""
    size = {"domino": "slots", "lego": "bricks"}[task]
    base = {"domino": graphs.domino_line, "lego": graphs.lego_press}[task]

    def build(**kw):
        return base(**{size: slots}, **kw)

    candidates = graphs.domino_candidates(slots) if task == "domino" else graphs.lego_candidates()
    train_seeds, test_seeds = list(train_seeds), list(test_seeds)
    if set(train_seeds) & set(test_seeds):
        raise ValueError("training and test seeds must be disjoint")
    rows = {}
    for name, graph in (
        ("fixed_script", build(recovery=False, attempts=1)),
        ("online_replan", build(recovery=True)),
    ):
        episodes = [evaluate(graph, seed) for seed in test_seeds]
        rows[name] = {
            "params": graph["params"],
            **summarize(episodes),
            "planning_s": 0.0,
            "search_rollouts": 0,
            "episodes_detail": episodes,
        }
    search = simulation_search(candidates, evaluate, train_seeds)
    episodes = [evaluate(search["selected"], seed) for seed in test_seeds]
    rows["simulation_search"] = {
        "params": search["selected_params"],
        **summarize(episodes),
        "planning_s": search["planning_s"],
        "search_rollouts": search["search_rollouts"],
        "candidates": len(candidates),
        "train_table": search["train_table"],
        "episodes_detail": episodes,
    }
    return {
        "task": task,
        "environment": environment,
        "train_seeds": train_seeds,
        "test_seeds": test_seeds,
        "K": sorted(build()["skills"].items()),
        "baselines": rows,
    }
