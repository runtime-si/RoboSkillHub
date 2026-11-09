"""assigning-bricks-to-two-arms: FakeContext tests of the APEX-MR assignment MILP."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

from ._apex_tasks import ARMS, load

BUNDLE = "assigning-bricks-to-two-arms"
P = 0.016  # DUPLO pitch
LEFT = {"arm_id": 0, "base": [0.0, 0.0, 0.0], "reach_m": 0.6}
RIGHT = {"arm_id": 1, "base": [0.8, 0.0, 0.0], "reach_m": 0.6}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["assign_arms"].module


def task(tid, x, y=0.0, layer=1, studs=(2, 2), **kw):
    return {"id": tid, "studs": list(studs), "position": [x, y, (layer - 1) * 0.0192], **kw}


def solve(script, tasks, arms=(LEFT, RIGHT), ctx=None, **kw):
    return script.run(ctx or FakeContext({}), arms=list(arms), tasks=tasks, **kw)


def assignment(out):
    return json.loads(out["assignment_json"])


def test_a_row_is_split_by_reach_and_balanced_in_windows(script):
    # Four bricks between two arms 0.8 m apart: every brick in reach of both.
    out = solve(script, [task(str(k), 0.34 + 0.04 * k) for k in range(4)])
    assert out["route"] == "assignment_ready", out["detail"]
    a = assignment(out)
    arms = [r["arm_id"] for r in a]
    # APEX-MR's windows of n = 2 consecutive tasks: the arms alternate.
    assert all(arms[k] != arms[k + 1] for k in range(3))
    assert out["imbalance"] == 0
    assert all(r["support_arm_id"] is None for r in a)
    by_arm = json.loads(out["by_arm_json"])
    assert sorted(len(v) for v in by_arm.values()) == [2, 2]


def test_reach_decides_who_can_place(script):
    out = solve(script, [task("near_left", 0.1), task("near_right", 0.7), task("mid", 0.4)])
    a = {r["task"]: r["arm_id"] for r in assignment(out)}
    assert a["near_left"] == 0 and a["near_right"] == 1


def test_a_task_out_of_every_arms_reach_is_infeasible(script):
    out = solve(script, [task("a", 0.4), task("far", 0.4, y=0.9)])
    assert out["route"] == "infeasible"
    assert "far" in out["detail"] and "reach" in out["detail"]


def test_a_supported_task_gets_the_other_arm_on_the_same_side(script):
    tasks = [task("base", 0.4), task("over", 0.42, layer=2, needs_support=True)]
    out = solve(script, tasks)
    assert out["route"] == "assignment_ready", out["detail"]
    a = {r["task"]: r for r in assignment(out)}
    assert a["over"]["support_arm_id"] is not None
    assert a["over"]["support_arm_id"] != a["over"]["arm_id"]
    assert out["n_support"] == 1


def test_a_support_only_the_placing_arm_reaches_is_needs_support_arm(script):
    # Only the left arm reaches the target and the support point: it cannot do both.
    left, right = {**LEFT, "reach_m": 0.6}, {**RIGHT, "reach_m": 0.45}
    tasks = [
        task("a", 0.4),
        task("b", 0.2, layer=2, needs_support=True, support_position=[0.05, 0.0, 0.0]),
    ]
    out = solve(script, tasks, arms=(left, right))
    assert out["route"] == "needs_support_arm", out["detail"]
    assert "b" in out["detail"]
    # Where the right arm reaches the target, it places and the left supports.
    tasks[1]["position"][0] = 0.45
    out = solve(script, tasks, arms=(left, right))
    assert out["route"] == "assignment_ready", out["detail"]
    b = {r["task"]: r for r in assignment(out)}["b"]
    assert (b["arm_id"], b["support_arm_id"]) == (1, 0)
    one_arm = solve(script, [task("a", 0.4, needs_support=True)], arms=(LEFT,))
    assert one_arm["route"] == "needs_support_arm"


def test_flush_neighbours_force_the_other_grasp_side(script):
    # A 2x4 (x: 2 studs, y: 4) placed after flush 2x2s on both of its x faces: the jaws
    # cannot close along x (fingers on the x faces), so they close along y.
    tasks = [
        task("w", 0.4 - 2 * P, studs=(2, 2)),
        task("e", 0.4 + 2 * P, studs=(2, 2)),
        task("mid", 0.4, studs=(2, 4)),
    ]
    out = solve(script, tasks)
    assert out["route"] == "assignment_ready", out["detail"]
    a = {r["task"]: r for r in assignment(out)}
    assert a["mid"]["close_along"] == "y"
    # Walled in on all four sides: no grasp at all.
    tasks += [task("n", 0.4, y=3 * P), task("s", 0.4, y=-3 * P)]
    tasks = tasks[:2] + tasks[3:] + [tasks[2]]
    out = solve(script, tasks)
    assert out["route"] == "infeasible" and "fingers" in out["detail"]


def test_jaws_too_narrow_for_the_long_side(script):
    # DUPLO 2x8: 128 mm along y does not fit 94 mm jaws; 32 mm along x does.
    out = solve(script, [task("long", 0.4, studs=(2, 8)), task("b", 0.3)])
    a = {r["task"]: r for r in assignment(out)}
    assert a["long"]["close_along"] == "x"


def test_offsets_prefer_the_centre(script):
    out = solve(script, [task("a", 0.4, studs=(2, 4)), task("b", 0.3)])
    a = {r["task"]: r for r in assignment(out)}
    assert a["a"]["grasp_offset_m"] == 0.0
    brick = script._Brick(task("a", 0.4, studs=(2, 4)), "a", "duplo")
    # Across the 2-stud x: offsets -16 to +16 mm in 8 mm steps along y; across the 4-stud y
    # (64 mm): 0 only (the 32 mm x side leaves 6 mm for a 20 mm finger).
    opts = script.grasp_options(brick, 7, 0.094, 0.004, 0.020)
    assert sorted(opts) == [(0, -0.016), (0, -0.008), (0, 0.0), (0, 0.008), (0, 0.016), (1, 0.0)]


def test_stock_is_used_once_unless_reusable_and_matched_by_size(script):
    stock = [
        {"id": "s1", "studs": [2, 2], "position": [0.2, -0.2, 0.0]},
        {"id": "s2", "studs": [2, 2], "position": [0.6, -0.2, 0.0]},
        {"id": "long", "studs": [4, 2], "position": [0.4, -0.3, 0.0]},
    ]
    tasks = [task("a", 0.35), task("b", 0.45), task("c", 0.4, studs=(2, 4))]
    out = solve(script, tasks, stock=stock)
    assert out["route"] == "assignment_ready", out["detail"]
    a = {r["task"]: r for r in assignment(out)}
    assert {a["a"]["stock"], a["b"]["stock"]} == {"s1", "s2"}
    assert a["c"]["stock"] == "long"  # a 4x2 brick fits a 2x4 task turned
    # A third 2x2 task with two 2x2 bricks in stock: infeasible; a reusable station fixes it.
    tasks.append(task("d", 0.5))
    assert solve(script, tasks, stock=stock)["route"] == "infeasible"
    stock[0]["reusable"] = True
    assert solve(script, tasks, stock=stock)["route"] == "assignment_ready"
    # No stock of a size at all.
    out = solve(script, [task("x", 0.4, studs=(1, 2)), task("y", 0.3)], stock=stock)
    assert out["route"] == "infeasible" and "no stock brick of size 1x2" in out["detail"]


def test_stock_precedence_picks_the_top_brick_first(script):
    # Two identical bricks stacked in stock: 'top' must be used no later than 'bottom'.
    stock = [
        {"id": "top", "studs": [2, 2], "position": [0.62, -0.2, 0.0192]},
        {"id": "bottom", "studs": [2, 2], "position": [0.62, -0.2, 0.0]},
    ]
    tasks = [task("a", 0.4), task("b", 0.42, y=0.1)]
    out = solve(script, tasks, stock=stock, stock_precedence=[["top", "bottom"]])
    a = [r["stock"] for r in assignment(out)]
    assert a == ["top", "bottom"]
    with pytest.raises(ValueError):
        solve(script, tasks, stock=stock, stock_precedence=[["top", "nope"]])


def test_ik_preview_rules_out_an_arm(script):
    calls = []

    def ik(pose, **kw):
        calls.append(kw.get("arm_id", 0))
        return {
            "success": kw.get("arm_id", 0) == 1,
            "within_servo_tolerance": kw.get("arm_id", 0) == 1,
        }

    ctx = FakeContext({"robot.solve_ik": ik})
    out = solve(script, [task("a", 0.4), task("b", 0.45)], ctx=ctx, check_ik=True)
    assert out["route"] == "assignment_ready"
    assert {r["arm_id"] for r in assignment(out)} == {1}
    assert 0 in calls and 1 in calls
    # arm_id 0 is not passed (gap connectors take it only when nonzero).
    ctx2 = FakeContext(
        {"robot.solve_ik": lambda pose, **kw: (calls.append(kw), {"success": True})[1]}
    )
    calls.clear()
    solve(script, [task("a", 0.4), task("b", 0.45)], ctx=ctx2, check_ik=True)
    assert {} in calls and {"arm_id": 1} in calls


def test_the_milp_matches_brute_force_on_small_instances(script):
    """Every arm / grasp choice enumerated for 4 tasks (no stock, no support): the MILP's
    objective equals the best enumerated one."""
    rng = np.random.default_rng(3)
    for _ in range(5):
        xs = 0.25 + 0.08 * np.arange(4) + rng.uniform(0.0, 0.02, 4)
        tasks = [task(str(k), float(x), y=float(rng.uniform(-0.1, 0.1))) for k, x in enumerate(xs)]
        out = solve(script, tasks)
        assert out["route"] == "assignment_ready"
        best = np.inf
        bases = [np.array(LEFT["base"]), np.array(RIGHT["base"])]
        for arms in itertools.product((0, 1), repeat=4):
            travel = sum(
                2 * float(np.linalg.norm(np.array(t["position"]) - bases[i]))
                for t, i in zip(tasks, arms, strict=True)
            )
            imb = sum(
                max(arms[w : w + 2].count(i) for i in (0, 1))
                - min(arms[w : w + 2].count(i) for i in (0, 1))
                for w in range(3)
            )
            best = min(best, travel + 1000.0 * imb)
        assert out["objective"] == pytest.approx(best, abs=1e-6)


def test_planning_sequence_on_a_plate(script):
    seq = [
        {
            "id": 1,
            "span_x": 2,
            "span_y": 4,
            "z_units": 0,
            "height_units": 3,
            "cell": {"x": 0, "y": 0},
        },
        {
            "id": 2,
            "span_x": 2,
            "span_y": 4,
            "z_units": 0,
            "height_units": 3,
            "cell": {"x": 4, "y": 0},
        },
        {
            "id": 3,
            "span_x": 4,
            "span_y": 2,
            "z_units": 3,
            "height_units": 3,
            "cell": {"x": 1, "y": 1},
        },
    ]
    out = script.run(
        FakeContext({}),
        arms=[LEFT, RIGHT],
        sequence_json=json.dumps(seq),
        plate_origin={"x": 0.36, "y": -0.04, "z": 0.0},
        system="lego",
    )
    assert out["route"] == "assignment_ready", out["detail"]
    assert out["n_tasks"] == 3
    t = script.tasks_from_sequence(seq, np.array([0.36, -0.04, 0.0]), 0.0, "lego")
    assert t[2]["position"] == pytest.approx([0.36 + 0.024, -0.04 + 0.016, 0.0096])


@pytest.mark.parametrize(
    ("name", "n", "n_support"),
    [
        ("cliff", 12, 3),
        ("stairs_rotated", 14, 4),
        ("S", 13, 6),
        ("faucet", 14, 3),
        ("guitar", 24, 0),
        ("R", 21, 0),
        ("bridge", 38, 18),
        ("rss", 47, 12),
    ],
)
def test_apex_mr_assembly_tasks(script, name, n, n_support):
    tasks, stock = load(name)
    out = script.run(FakeContext({}), arms=ARMS, tasks=tasks, stock=stock, system="lego")
    assert out["route"] == "assignment_ready", out["detail"]
    a = assignment(out)
    assert len(a) == n
    sup = [r for r in a if r["support_arm_id"] is not None]
    assert len(sup) == n_support
    assert all(r["support_arm_id"] != r["arm_id"] for r in sup)
    assert out["solve_s"] < 30.0


def test_apex_mr_fish_needs_a_narrower_tool_than_yam_fingers(script):
    # fish_high's tasks 12 and 17 are LEGO 2x4s with flush neighbours along a whole face:
    # a 20 mm YAM finger on the other faces overhangs the 16 mm brick into a neighbour.
    # APEX-MR's own end-of-arm tool needs no finger room; a 6 mm finger fits.
    tasks, stock = load("fish_high")
    out = script.run(FakeContext({}), arms=ARMS, tasks=tasks, stock=stock, system="lego")
    assert out["route"] == "infeasible"
    assert "task 12" in out["detail"] and "task 17" in out["detail"]
    out = script.run(
        FakeContext({}), arms=ARMS, tasks=tasks, stock=stock, system="lego", finger_width=0.006
    )
    assert out["route"] == "assignment_ready", out["detail"]
    a = assignment(out)
    assert sum(r["support_arm_id"] is not None for r in a) == 12


def test_bad_arguments(script):
    with pytest.raises(ValueError):
        script.run(FakeContext({}), arms=[], tasks=[task("a", 0.4)])
    with pytest.raises(ValueError):
        script.run(FakeContext({}), arms=[LEFT], tasks=None)
    with pytest.raises(ValueError):
        script.run(FakeContext({}), arms=[LEFT], tasks=[task("a", 0.4)], system="megablok")
    out = script.run(FakeContext({}), arms=[LEFT, RIGHT], tasks=[task("a", 0.4)])
    assert out["route"] == "infeasible" and "m >= n" in out["detail"]


def test_every_route_is_declared_in_the_example(script, skills_registry):
    bundle = skills_registry.get(BUNDLE)
    example = json.loads(
        (Path(bundle.bundle_dir) / "examples" / "canonical_subgraph.json").read_text()
    )
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert set(script.ROUTES) <= declared
    skill_md = (Path(bundle.bundle_dir) / "SKILL.md").read_text()
    for route in script.ROUTES:
        assert f"    {route}:" in skill_md
