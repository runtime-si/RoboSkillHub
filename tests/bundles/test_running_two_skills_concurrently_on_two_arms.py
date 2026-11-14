"""running-two-skills-concurrently-on-two-arms: FakeContext tests on a fake two-arm YAM (bases 0.61 m apart)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "running-two-skills-concurrently-on-two-arms"
BASE_Y = {0: 0.305, 1: -0.305}
HOME = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
OUTWARD = {0: [0.5, 1.5, 1.0, 0.0, 0.0, 0.0], 1: [-0.5, 1.5, 1.0, 0.0, 0.0, 0.0]}
CROSS = {0: [-1.0, 1.5, 1.0, 0.0, 0.0, 0.0], 1: [1.0, 1.5, 1.0, 0.0, 0.0, 0.0]}
MOVE_S = 0.3


@pytest.fixture(scope="module")
def script(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["run_concurrent"].module


@pytest.fixture(scope="module")
def clearance(script):
    return script._clearance_module()


@pytest.fixture(autouse=True)
def _clean_runners(script):
    yield
    script.RUNNERS.clear()


class Robot:
    """Two YAMs: joint moves take MOVE_S and land; poses from the clearance skill's FK."""

    def __init__(self, model):
        self.m = model
        self.q = {0: np.array(HOME), 1: np.array(HOME)}
        self.tcp = model._frame(*model.DEFAULT_TCP)
        self.lock = threading.Lock()
        self.moves: list[tuple[int, float, float, list]] = []

    def _pose(self, arm, q):
        t = self.m.forward_kinematics(np.asarray(q, float))["gripper"] @ self.tcp
        t[1, 3] += BASE_Y[arm]
        return self.m._pose_dict(t)

    def ctx(self):
        def obs():
            with self.lock:
                return {
                    "arms": [
                        {
                            "joint_state": {"positions": self.q[a].tolist()},
                            "ee_pose": {"pose": self._pose(a, self.q[a])},
                            "gripper_fraction": 1.0,
                        }
                        for a in (0, 1)
                    ]
                }

        def fk(joint_configs, arm_id=0):
            return {"poses": [self._pose(arm_id, q) for q in joint_configs]}

        def move(joint_config, arm_id=0):
            t0 = time.monotonic()
            time.sleep(MOVE_S)
            with self.lock:
                self.q[arm_id] = np.array(joint_config["positions"], float)
            self.moves.append((arm_id, t0, time.monotonic(), list(joint_config["positions"])))
            return {"reached": True}

        def ik(pose, arm_id=0):
            return {"success": abs(pose["position"]["y"] - BASE_Y[arm_id]) < 0.4}

        return FakeContext(
            {
                "robot.get_observation": lambda: obs(),
                "robot.describe_arm": lambda arm_id=0: {},
                "robot.forward_kinematics": fk,
                "robot.move_to_joints": move,
                "robot.solve_ik": ik,
            }
        )


def joints_skill(*targets, fail_after=None):
    """A one-arm skill: move_to_joints through targets (each a dict arm -> joints)."""

    def run(ctx, tag=""):
        arm = ctx.arm
        for k, t in enumerate(targets):
            if fail_after is not None and k == fail_after:
                raise RuntimeError("grasp missed")
            ctx.tool("robot.move_to_joints", joint_config={"positions": t[arm]})
        return {"route": "done"}

    return run


def register(script, a, b):
    script.RUNNERS["skill-a"] = (a, {"done"})
    script.RUNNERS["skill-b"] = (b, {"done"})


def overlap(robot):
    m0 = [(s, e) for a, s, e, _ in robot.moves if a == 0]
    m1 = [(s, e) for a, s, e, _ in robot.moves if a == 1]
    return any(s0 < e1 and s1 < e0 for s0, e0 in m0 for s1, e1 in m1)


def test_the_poses_are_what_the_tests_assume(clearance):
    robot = Robot(clearance)
    check = clearance.run
    assert check(robot.ctx())["min_clearance_m"] > 0.2
    robot.q = {0: np.array(CROSS[0]), 1: np.array(HOME)}
    assert check(robot.ctx())["min_clearance_m"] > 0.03
    robot.q = {0: np.array(CROSS[0]), 1: np.array(CROSS[1])}
    assert check(robot.ctx())["min_clearance_m"] < 0.03


def test_non_conflicting_skills_run_at_the_same_time(script, clearance):
    robot = Robot(clearance)
    register(
        script, joints_skill(OUTWARD, {0: HOME, 1: HOME}), joints_skill(OUTWARD, {0: HOME, 1: HOME})
    )
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b")
    assert out["route"] == "both_done", out["detail"]
    assert (out["route_a"], out["route_b"]) == ("done", "done")
    assert out["waits"] == 0
    assert len(robot.moves) == 4
    assert overlap(robot)
    assert out["min_predicted_clearance_m"] >= 0.03


def test_arm_id_is_injected_over_the_skills_own(script, clearance):
    robot = Robot(clearance)
    seen = []

    def one_arm(ctx):
        ctx.tool("robot.move_to_joints", joint_config={"positions": OUTWARD[ctx.arm]}, arm_id=0)
        seen.append(ctx.arm)
        return {"route": "done"}

    register(script, one_arm, one_arm)
    ctx = robot.ctx()
    out = script.run(ctx, skill_a="skill-a", skill_b="skill-b", arm_a=1, arm_b=0)
    assert out["route"] == "both_done", out["detail"]
    arms = sorted(c.kwargs["arm_id"] for c in ctx.calls_to("robot.move_to_joints"))
    assert arms == [0, 1]
    np.testing.assert_allclose(robot.q[1], OUTWARD[1])


def test_a_crossing_conflict_serialises_the_arms(script, clearance):
    robot = Robot(clearance)
    there_and_back = joints_skill(CROSS, {0: HOME, 1: HOME})
    register(script, there_and_back, there_and_back)
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b")
    assert out["route"] == "both_done", out["detail"]
    assert out["waits"] >= 1
    # priority arm 0 crosses first; arm 1 crosses only after arm 0 came back.
    cross = {a: next(s for a2, s, _, q in robot.moves if a2 == a and q == CROSS[a]) for a in (0, 1)}
    back0 = next(e for a, _, e, q in robot.moves if a == 0 and q == HOME)
    assert cross[0] < cross[1]
    assert cross[1] >= back0
    assert out["min_predicted_clearance_m"] >= 0.03


def test_priority_arm_decides_who_crosses_first(script, clearance):
    robot = Robot(clearance)
    there_and_back = joints_skill(CROSS, {0: HOME, 1: HOME})
    register(script, there_and_back, there_and_back)
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b", priority_arm=1)
    assert out["route"] == "both_done", out["detail"]
    assert robot.moves[0][0] == 1


def test_one_failure_pauses_the_other_arm_at_its_start(script, clearance):
    robot = Robot(clearance)
    fails = joints_skill(OUTWARD, fail_after=0)  # raises before its first move

    def slow(ctx):
        time.sleep(0.1)
        for t in (OUTWARD, {0: HOME, 1: HOME}, OUTWARD):
            ctx.tool("robot.move_to_joints", joint_config={"positions": t[ctx.arm]})
        return {"route": "done"}

    register(script, fails, slow)
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b")
    assert out["route"] == "one_failed", out["detail"]
    assert out["route_a"] == "raised"
    assert out["route_b"] == "paused"
    assert out["paused_arms"] == "1"
    np.testing.assert_allclose(robot.q[1], HOME)
    assert not any(a == 0 for a, *_ in robot.moves)


def test_a_failed_exit_counts_as_a_failure(script, clearance):
    robot = Robot(clearance)
    register(script, lambda ctx: {"route": "grasp_missed"}, joints_skill(OUTWARD))
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b")
    assert out["route"] == "one_failed", out["detail"]
    assert out["route_a"] == "grasp_missed"


def test_an_arm_blocked_for_good_times_out(script, clearance):
    robot = Robot(clearance)
    register(script, joints_skill(CROSS), joints_skill(CROSS))  # arm 0 crosses and stays
    t0 = time.monotonic()
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b", wait_timeout_s=0.3)
    assert out["route"] == "deadlock_timeout", out["detail"]
    assert time.monotonic() - t0 < 5
    assert out["route_a"] == "done"
    assert out["route_b"] == "paused"
    assert "waited" in out["detail"]
    np.testing.assert_allclose(robot.q[1], HOME)
    np.testing.assert_allclose(robot.q[0], CROSS[0])


def test_arms_already_too_close_are_refused(script, clearance):
    robot = Robot(clearance)
    robot.q = {0: np.array(CROSS[0]), 1: np.array(CROSS[1])}
    register(script, joints_skill(OUTWARD), joints_skill(OUTWARD))
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b")
    assert out["route"] == "unsafe_refused"
    assert robot.moves == []


def test_a_move_the_model_cannot_follow_is_refused(script, clearance):
    robot = Robot(clearance)

    def homing(ctx):
        ctx.tool("robot.go_home")
        return {"route": "done"}

    register(script, homing, joints_skill(OUTWARD))
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b")
    assert out["route"] == "unsafe_refused", out["detail"]


def test_auto_assignment_by_reach(script, clearance):
    robot = Robot(clearance)
    register(script, joints_skill(OUTWARD), joints_skill(OUTWARD))
    out = script.run(
        robot.ctx(),
        skill_a="skill-a",
        skill_b="skill-b",
        arm_a=-1,
        arm_b=-1,
        target_a={"x": 0.4, "y": -0.3, "z": 0.1},
        target_b={"x": 0.4, "y": 0.3, "z": 0.1},
    )
    assert out["route"] == "both_done", out["detail"]
    assert (out["arm_a"], out["arm_b"]) == (1, 0)


def test_auto_assignment_refuses_when_nothing_reaches(script, clearance):
    robot = Robot(clearance)
    register(script, joints_skill(OUTWARD), joints_skill(OUTWARD))
    far = {"x": 0.4, "y": 2.0, "z": 0.1}
    out = script.run(
        robot.ctx(), skill_a="skill-a", skill_b="skill-b", arm_a=-1, target_a=far, target_b=far
    )
    assert out["route"] == "unsafe_refused"
    assert robot.moves == []


def test_loads_a_real_hub_skill(script):
    run, success = script.load_skill("checking-dual-arm-clearance")
    assert callable(run)
    assert success == {"clear", "too_close"}
