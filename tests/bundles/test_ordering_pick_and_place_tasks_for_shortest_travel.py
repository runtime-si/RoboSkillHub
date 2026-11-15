"""ordering-pick-and-place-tasks-for-shortest-travel: Held-Karp against brute force, precedence, the heuristic,
#251 queues."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from gap.testing import FakeContext


@pytest.fixture(scope="module")
def ot(skills_registry):
    return (
        skills_registry.get("ordering-pick-and-place-tasks-for-shortest-travel")
        .canonical_scripts["order_tasks"]
        .module
    )


def _tasks(n, seed):
    rng = np.random.default_rng(seed)
    return [
        {
            "id": f"t{k}",
            "pick": list(rng.uniform(-0.3, 0.3, 3)),
            "place": list(rng.uniform(-0.3, 0.3, 3)),
        }
        for k in range(n)
    ]


def _brute(ot, tasks, start, preds=None):
    picks = {t["id"]: np.array(t["pick"]) for t in tasks}
    places = {t["id"]: np.array(t["place"]) for t in tasks}
    best = min(
        ot._cost(list(p), np.array(start), picks, places, ot.SPEED, ot.LIFT)
        for p in itertools.permutations([t["id"] for t in tasks])
        if preds is None or ot._feasible(list(p), preds)
    )
    return best


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_matches_brute_force(ot, seed):
    tasks = _tasks(7, seed)
    out = ot.run(FakeContext({}), tasks=tasks, start={"0": [0, 0, 0.2]})
    assert out["method"] == "held-karp"
    assert out["total_travel_s"] == pytest.approx(_brute(ot, tasks, [0, 0, 0.2]), abs=1e-3)
    assert out["total_travel_s"] <= out["total_given_s"] + 1e-9


def test_precedence_is_kept(ot):
    tasks = _tasks(6, 3)
    tasks[0]["after"] = ["t5"]
    tasks[2]["after"] = ["t4"]
    out = ot.run(FakeContext({}), tasks=tasks, start={"0": [0, 0, 0.2]})
    seq = out["order"]["0"]
    assert seq.index("t5") < seq.index("t0") and seq.index("t4") < seq.index("t2")
    preds = {"t0": {"t5"}, "t2": {"t4"}}
    assert out["total_travel_s"] == pytest.approx(_brute(ot, tasks, [0, 0, 0.2], preds), abs=1e-3)


def test_heuristic_for_large_sets_beats_the_given_order(ot):
    tasks = _tasks(25, 4)
    out = ot.run(FakeContext({}), tasks=tasks)
    assert out["method"] == "nearest + 2-opt/or-opt" and sorted(out["order"]["0"]) == sorted(
        t["id"] for t in tasks
    )
    assert out["total_travel_s"] < out["total_given_s"]


def test_reorders_each_arm_queue_from_the_scheduler(ot):
    tasks = _tasks(8, 5)
    by_arm = {"0": ["t0", "t1", "t2", "t3"], "1": ["t4", "t5", "t6", "t7"]}
    out = ot.run(FakeContext({}), tasks=tasks, by_arm=by_arm)
    assert {a: sorted(q) for a, q in out["order"].items()} == {
        a: sorted(q) for a, q in by_arm.items()
    }
    assert all(out["travel_s"][a] <= out["given_travel_s"][a] + 1e-9 for a in by_arm)


def test_cycles_and_bad_inputs(ot):
    tasks = _tasks(3, 6)
    tasks[0]["after"], tasks[1]["after"] = ["t1"], ["t0"]
    assert ot.run(FakeContext({}), tasks=tasks, method="heuristic")["route"] == "infeasible"
    with pytest.raises(ValueError, match="empty"):
        ot.run(FakeContext({}), tasks=[])
    with pytest.raises(ValueError, match="unique"):
        ot.run(FakeContext({}), tasks=_tasks(1, 0) * 2)
    with pytest.raises(ValueError, match="predecessor"):
        ot.run(
            FakeContext({}),
            tasks=[{"id": "a", "pick": [0, 0, 0], "place": [1, 0, 0], "after": ["z"]}],
        )
    with pytest.raises(ValueError, match="method"):
        ot.run(FakeContext({}), tasks=_tasks(2, 0), method="ga")
