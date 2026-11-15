"""profiling-a-skill-run-by-phase: phase rules, gap attribution, a fake child skill, and an offline recording."""

from __future__ import annotations

import json

import pytest

BUNDLE = "profiling-a-skill-run-by-phase"


@pytest.fixture(scope="module")
def mod(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["profile_skill"].module


@pytest.mark.parametrize(
    "name,phase",
    [
        ("robot.go_to_pose", "motion"),
        ("robot.go_to_pose_cartesian", "motion"),
        ("robot.execute_trajectory", "motion"),
        ("robot.move_to_joints", "motion"),
        ("robot.open_gripper", "gripper"),
        ("robot.close_gripper", "gripper"),
        ("robot.get_observation", "perception"),
        ("sam3.segment", "perception"),
        ("vlm.ask", "perception"),
        ("robot.solve_ik", "planning"),
        ("curobo.plan_motion", "planning"),
        ("robot.wait_steps", "settle"),
        ("robot.get_ee_pose", "other"),
        ("robot.get_gripper", "other"),
        ("move", "motion"),
        ("grip", "gripper"),
    ],
)
def test_classify(mod, name, phase):
    assert mod.classify(name) == phase


class Ctx:
    def tool(self, name, **kw):
        return {"outcome": "reached"}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_profiler_gaps_and_sim(mod):
    wall, sim = Clock(), Clock()
    durations = {
        "robot.solve_ik": (0.1, 0.0),
        "robot.go_to_pose": (1.0, 0.8),
        "robot.close_gripper": (0.5, 0.5),
    }

    class Ctx:
        def tool(self, name, **kw):
            dw, ds = durations[name]
            wall.t += dw
            sim.t += ds
            return {"outcome": "reached"}

    prof = mod.Profiler(Ctx(), sim_clock=sim, clock=wall)
    prof.mark("obj0")
    prof.tool("robot.solve_ik")
    wall.t += 0.2  # client code
    prof.tool("robot.go_to_pose", arm_id=0)
    wall.t += 0.3  # after arrival, before the next command
    sim.t += 0.1
    prof.tool("robot.close_gripper")
    tl = prof.timeline()
    assert [r["phase"] for r in tl] == ["planning", "other", "motion", "settle", "gripper"]
    t = mod.phase_table(tl)["all"]
    assert t["motion"]["wall_s"] == pytest.approx(1.0)
    assert t["settle"]["wall_s"] == pytest.approx(0.3)
    assert t["settle"]["sim_s"] == pytest.approx(0.1)
    assert t["other"]["wall_s"] == pytest.approx(0.2)
    assert t["motion"]["calls"] == 1 and t["settle"]["calls"] == 0
    assert sum(v["share"] for v in t.values()) == pytest.approx(1.0, abs=1e-3)
    assert set(mod.phase_table(tl, by="segment")) == {"obj0"}
    assert "motion" in mod.format_table(mod.phase_table(tl))


def test_arrival_splits_a_motion(mod):
    calls = [
        {
            "name": "robot.go_to_pose",
            "phase": "motion",
            "start_s": 0.0,
            "end_s": 2.0,
            "arrived_s": 1.5,
            "thread": "t",
        }
    ]
    tl = mod.build_timeline(calls)
    assert [(r["phase"], r["end_s"] - r["start_s"]) for r in tl] == [
        ("motion", 1.5),
        ("settle", 0.5),
    ]


def test_run_wraps_child(mod):
    def child(ctx, n=2):
        ctx.tool("robot.solve_ik", pose={})
        for _ in range(n):
            ctx.tool("robot.go_to_pose", pose={})
        ctx.tool("robot.open_gripper")
        return {"route": "placed"}

    mod.RUNNERS["fake-pick"] = child
    ctx = Ctx()
    out = mod.run(ctx, skill="fake-pick", inputs_json=json.dumps({"n": 3}))
    assert out["route"] == "profiled" and out["child_route"] == "placed"
    tl = json.loads(out["timeline_json"])
    assert [r["name"] for r in tl if r.get("kind") is None] == ["robot.solve_ik"] + [
        "robot.go_to_pose"
    ] * 3 + ["robot.open_gripper"]
    assert mod.run(ctx, skill="no-such-skill")["route"] == "refused"


def test_run_child_raises(mod):
    def child(ctx):
        ctx.tool("robot.get_observation")
        raise RuntimeError("boom")

    mod.RUNNERS["fake-raise"] = child
    out = mod.run(Ctx(), skill="fake-raise")
    assert out["route"] == "failed" and "boom" in out["child_outputs"]
    assert json.loads(out["timeline_json"])[0]["phase"] == "perception"


def test_profile_recording(mod):
    events = [
        {"op": "session", "t": 0.0},
        {"op": "move", "t": 100.0, "duration_s": 2.0, "label": "look pen_32_black_0: down"},
        {"op": "move", "t": 105.0, "duration_s": 1.0, "label": "over pen 32: across"},
        {"op": "grip", "t": 107.0, "duration_s": 0.5, "label": "close"},
        {"op": "move", "t": 108.0, "duration_s": 1.0, "label": "over pen 22: across"},
    ]
    moves = [
        {"t": 99.9, "label": "look pen_32_black_0: down", "settle_s": 1.0},
        {"t": 104.9, "label": "over pen 32", "settle_s": 0.4},
    ]
    tl = mod.profile_recording(events, moves)
    per = mod.phase_table(tl, by="segment")
    p32 = per["pen 32"]
    assert p32["motion"]["wall_s"] == pytest.approx(3.0)
    assert p32["settle"]["wall_s"] == pytest.approx(1.4)
    assert p32["perception"]["wall_s"] == pytest.approx(2.0)  # after the look move, past its settle
    assert p32["gripper"]["wall_s"] == pytest.approx(0.5)
    assert p32["other"]["wall_s"] == pytest.approx(0.6 + 0.5)
    assert per["pen 22"]["motion"]["wall_s"] == pytest.approx(1.0)
