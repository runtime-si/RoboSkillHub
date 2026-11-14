"""scheduling-tasks-across-two-arms-for-minimum-makespan on small task lists."""

from __future__ import annotations

import pytest
from gap.testing import FakeContext

NAME = "scheduling-tasks-across-two-arms-for-minimum-makespan"


@pytest.fixture(scope="module")
def st(skills_registry):
    return skills_registry.get(NAME).canonical_scripts["schedule_tasks"].module


def _t(i, d, region, **kw):
    return {"id": i, "duration": d, "region": {"points": [region]}, **kw}


def test_independent_far_tasks_run_in_parallel(st):
    tasks = [_t("a", 4, [0, 0.4, 0]), _t("b", 4, [0, -0.4, 0])]
    out = st.run(FakeContext({}), tasks=tasks)
    assert out["route"] == "scheduled" and out["makespan"] == pytest.approx(4.0)
    assert {r["arm"] for r in out["schedule"]} == {0, 1}
    assert out["concurrent_steps"][0]["arm_0"] and out["concurrent_steps"][0]["arm_1"]


def test_close_regions_never_overlap_in_time(st):
    tasks = [_t("a", 4, [0, 0.0, 0]), _t("b", 4, [0, 0.05, 0])]
    out = st.run(FakeContext({}), tasks=tasks)
    s = {r["id"]: r for r in out["schedule"]}
    assert s["a"]["end"] <= s["b"]["start"] + 1e-6 or s["b"]["end"] <= s["a"]["start"] + 1e-6
    assert out["makespan"] == pytest.approx(8.0)


def test_precedence_same_arm_and_reach(st):
    tasks = [
        _t("pick", {"0": 3, "1": 5}, [0, 0.4, 0]),
        _t("place", {"0": 3, "1": 5}, [0, -0.4, 0], after=["pick"], same_arm_as="pick"),
        _t("other", {"1": 2}, [0.3, -0.4, 0], arms=[1]),
    ]
    out = st.run(FakeContext({}), tasks=tasks)
    s = {r["id"]: r for r in out["schedule"]}
    assert s["pick"]["arm"] == s["place"]["arm"] and s["place"]["start"] >= s["pick"]["end"] - 1e-6
    assert s["other"]["arm"] == 1


def test_greedy_and_milp_agree_on_a_small_job(st):
    tasks = [_t(f"t{k}", 2 + k, [0, 0.4 * (-1) ** k, 0]) for k in range(5)]
    m = st.run(FakeContext({}), tasks=tasks, method="milp")
    g = st.run(FakeContext({}), tasks=tasks, method="greedy")
    assert m["makespan"] <= g["makespan"] + 1e-6 and m["makespan"] >= m["lower_bound"] - 1e-6


def test_infeasible_cases(st):
    out = st.run(FakeContext({}), tasks=[_t("a", {"0": 1}, [0, 0, 0], arms=[])])
    assert out["route"] == "infeasible" and "no arm" in out["detail"]
    out = st.run(
        FakeContext({}),
        tasks=[_t("a", 1, [0, 0, 0], after=["b"]), _t("b", 1, [0, 0, 0], after=["a"])],
    )
    assert out["route"] == "infeasible"


def test_bad_inputs(st):
    with pytest.raises(ValueError, match="empty"):
        st.run(FakeContext({}), tasks=[])
    with pytest.raises(ValueError, match="unique"):
        st.run(FakeContext({}), tasks=[_t("a", 1, [0, 0, 0]), _t("a", 1, [0, 0, 0])])
    with pytest.raises(ValueError, match="unknown predecessor"):
        st.run(FakeContext({}), tasks=[_t("a", 1, [0, 0, 0], after=["z"])])
    with pytest.raises(ValueError, match="method"):
        st.run(FakeContext({}), tasks=[_t("a", 1, [0, 0, 0])], method="cp")
