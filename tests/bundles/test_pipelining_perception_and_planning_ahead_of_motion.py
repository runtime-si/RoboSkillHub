"""pipelining-perception-and-planning-ahead-of-motion: FakeContext tests with a slow fake arm, a slow
fake detector and a slow fake IK, so look-ahead shows up as idle time saved."""

from __future__ import annotations

import json
import threading
import time

import pytest
from gap.testing import FakeContext

BUNDLE = "pipelining-perception-and-planning-ahead-of-motion"
MOVE_S, PERCEIVE_S, IK_S = 0.3, 0.2, 0.05


@pytest.fixture(scope="module")
def script(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["pipeline_tasks"].module


@pytest.fixture(autouse=True)
def _clean(script):
    yield
    script._concurrent().RUNNERS.clear()


def child(ctx, object_obb, place):
    """One-arm pick-and-place: plan (grasp frame + 2 IK solves from the tool pose), then 3 moves back home."""
    home = ctx.tool("robot.get_ee_pose")["pose"]
    c = object_obb["center"]
    ctx.tool("robot.grasp_frame", close_heading_deg=0.0)
    for z in (0.1, 0.0):
        if not ctx.tool("robot.solve_ik", pose={"position": {**c, "z": c["z"] + z}})["success"]:
            return {"route": "unreachable"}
    ctx.tool("robot.go_to_pose", pose={"position": c})
    ctx.tool("robot.go_to_pose", pose={"position": place})
    ctx.tool("robot.go_to_pose", pose=home)
    return {"route": "placed" if c["x"] < 1.0 else "failed"}


class World:
    def __init__(self, n=4, arms=(0,)):
        self.objects = {
            f"o{k}": {"x": 0.3 + 0.01 * k, "y": 0.2 if k % 2 == 0 else -0.2, "z": 0.0}
            for k in range(n)
        }
        self.pose = {
            a: {
                "position": {"x": 0.0, "y": 0.0, "z": 0.3},
                "rotation": {"w": 1.0, "x": 0, "y": 0, "z": 0},
            }
            for a in arms
        }
        self.cameras: list[str] = []
        self.moves: list[tuple[int, float]] = []
        self.after_move = None
        self.lock = threading.Lock()

    def ctx(self):
        def perceive(object_name, camera=""):
            self.cameras.append(camera)
            time.sleep(PERCEIVE_S)
            return {"obb": {"center": dict(self.objects[object_name])}}

        def ik(pose, arm_id=0):
            time.sleep(IK_S)
            return {"success": True}

        def move(pose, arm_id=0):
            time.sleep(MOVE_S)
            with self.lock:
                self.pose[arm_id] = {
                    **self.pose[arm_id],
                    "position": dict(pose.get("position", pose)),
                }
                self.moves.append((arm_id, time.monotonic()))
            if self.after_move:
                self.after_move(self)
            return {"reached": True}

        return FakeContext(
            {
                "sim.get_object_obb": perceive,
                "robot.get_ee_pose": lambda arm_id=0: {
                    "pose": json.loads(json.dumps(self.pose[arm_id]))
                },
                "robot.grasp_frame": lambda close_heading_deg=0.0, arm_id=0: {"rotation": {}},
                "robot.solve_ik": ik,
                "robot.go_to_pose": move,
            }
        )


def run(script, world, pipelined=True, n=4, **kw):
    arm_of = kw.pop("arm_of", lambda k: 0)
    script._concurrent().RUNNERS["fake-pick-place"] = (child, {"placed"})
    tasks = [
        {
            "object": f"o{k}",
            "place": {"x": 0.4, "y": 0.0, "z": 0.0},
            "arm": arm_of(k),
        }
        for k in range(n)
    ]
    return script.run(
        world.ctx(),
        tasks=tasks,
        pick_place_skill="fake-pick-place",
        perceive_tool="sim.get_object_obb",
        pipelined=pipelined,
        **kw,
    )


def test_pipelining_cuts_the_idle_time_between_tasks(script):
    base = run(script, World(), pipelined=False)
    pipe = run(script, World(), pipelined=True)
    assert base["route"] == pipe["route"] == "all_done", (base["detail"], pipe["detail"])
    assert pipe["done"] == 4
    # without look-ahead every task waits for perception + 2 IK; with it only the first does
    assert base["idle_s"] > 4 * (PERCEIVE_S + 2 * IK_S) * 0.9
    assert pipe["idle_s"] < base["idle_s"] / 2
    assert pipe["makespan_s"] < base["makespan_s"] - 2 * PERCEIVE_S
    assert pipe["cache_hits"] >= 3 * 3  # grasp frame + 2 IK for each of tasks 2..4
    assert pipe["stale"] == 0


def test_the_lookahead_never_moves_the_arm_and_answers_from_the_expected_end(script):
    world = World()
    pipe = run(script, world, pipelined=True)
    assert len(world.moves) == 4 * 3  # only the real runs moved
    tasks = json.loads(pipe["tasks"])
    assert [t["route"] for t in tasks] == ["placed"] * 4


def test_an_object_moved_during_the_previous_task_is_replanned(script):
    world = World()
    moved = {"done": False}

    def nudge(w):
        if len(w.moves) == 1 and not moved["done"]:
            w.objects["o1"]["x"] += 0.05  # knocked while task 0 carries
            moved["done"] = True

    world.after_move = nudge
    out = run(script, world, pipelined=True)
    assert out["route"] == "all_done", out["detail"]
    assert out["stale"] == 1
    t1 = json.loads(out["tasks"])[1]
    assert "object moved 50 mm" in t1["stale"]
    assert t1["misses"] >= 3 and t1["hits"] == 0  # planned live from the new box
    picks = [p for p in world.moves]
    assert picks  # and it was picked where it is now
    assert world.pose[0]["position"] != {}


def test_an_arm_not_where_expected_makes_the_plan_stale(script):
    world = World()

    def drift(w):
        if len(w.moves) == 3:
            w.pose[0]["position"]["x"] += 0.03  # task 0 ended 30 mm off its start

    world.after_move = drift
    out = run(script, world, pipelined=True)
    assert out["stale"] >= 1
    assert "30 mm from the expected end" in json.loads(out["tasks"])[1]["stale"]


def test_a_failure_is_one_failed_and_the_rest_still_run(script):
    world = World()
    world.objects["o2"]["x"] = 5.0  # the child reports failed for it
    out = run(script, world, pipelined=True)
    assert out["route"] == "one_failed"
    assert out["done"] == 3 and out["failed"] == 1


def test_a_stale_plan_that_then_fails_is_stale_replan_failed(script):
    world = World()

    def knock(w):
        if len(w.moves) == 1:
            w.objects["o1"]["x"] = 5.0

    world.after_move = knock
    out = run(script, world, pipelined=True)
    assert out["route"] == "stale_replan_failed", out["detail"]


def test_the_moving_wrist_camera_is_refused(script):
    world = World()
    out = run(script, world, pipelined=True, camera="left_wrist")
    assert out["route"] != "all_done"
    assert "left_wrist" not in world.cameras


def test_the_idle_arm_wrist_camera_is_allowed_on_one_arm(script):
    world = World()
    out = run(script, world, pipelined=True, camera="right_wrist")
    assert out["route"] == "all_done", out["detail"]
    assert set(world.cameras) == {"right_wrist"}


def test_two_arms_each_pipeline_their_own_queue(script):
    world = World(n=4, arms=(0, 1))
    conc = script._concurrent()

    def ok(ctx, arm, plan, other, other_plan):
        return "ok", 1.0, "fake"

    orig = conc.Arbiter.__init__

    def init(self, *a, **kw):
        orig(self, *a, **kw)
        self.predictor = ok

    conc.Arbiter.__init__ = init
    try:
        out = run(script, world, pipelined=True, arm_of=lambda k: k % 2)
    finally:
        conc.Arbiter.__init__ = orig
    assert out["route"] == "all_done", out["detail"]
    tasks = json.loads(out["tasks"])
    assert [t["arm"] for t in tasks] == [0, 1, 0, 1]
    a0 = [m[1] for m in world.moves if m[0] == 0]
    a1 = [m[1] for m in world.moves if m[0] == 1]
    assert a0 and a1 and min(a1) < max(a0)  # the arms overlapped
