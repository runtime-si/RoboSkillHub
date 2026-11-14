"""running-two-skills-concurrently-on-two-arms: FakeContext tests on a fake two-arm YAM (bases 0.61 m apart).

Threaded mode first, then the interleaved and sequential modes for a server that moves one arm at a time.
"""

from __future__ import annotations

import math
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
MOVE_S = 0.6


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

    def __init__(self, model, describe=None, move_s=MOVE_S):
        self.move_s = move_s
        self.describe = describe if describe is not None else {"concurrent_arms": True}
        self.server = threading.Lock()  # held per move when the server serialises
        self.serialises = bool(self.describe.get("serializes_moves"))
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
            if self.serialises:
                with self.server:
                    return _move(joint_config, arm_id)
            return _move(joint_config, arm_id)

        def _move(joint_config, arm_id):
            t0 = time.monotonic()
            time.sleep(self.move_s)
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
                "robot.describe_robot": self.describe,
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
    assert out["route"] == "both_succeeded", out["detail"]
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
    assert out["route"] == "both_succeeded", out["detail"]
    arms = sorted(c.kwargs["arm_id"] for c in ctx.calls_to("robot.move_to_joints"))
    assert arms == [0, 1]
    np.testing.assert_allclose(robot.q[1], OUTWARD[1])


def test_a_crossing_conflict_serialises_the_arms(script, clearance):
    robot = Robot(clearance)
    there_and_back = joints_skill(CROSS, {0: HOME, 1: HOME})
    register(script, there_and_back, there_and_back)
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b")
    assert out["route"] == "both_succeeded", out["detail"]
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
    assert out["route"] == "both_succeeded", out["detail"]
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
    assert out["route"] == "refused"
    assert robot.moves == []


def test_a_move_the_model_cannot_follow_is_refused(script, clearance):
    robot = Robot(clearance)

    def homing(ctx):
        ctx.tool("robot.go_home")
        return {"route": "done"}

    register(script, homing, joints_skill(OUTWARD))
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b")
    assert out["route"] == "refused", out["detail"]


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
    assert out["route"] == "both_succeeded", out["detail"]
    assert (out["arm_a"], out["arm_b"]) == (1, 0)


def test_auto_assignment_refuses_when_nothing_reaches(script, clearance):
    robot = Robot(clearance)
    register(script, joints_skill(OUTWARD), joints_skill(OUTWARD))
    far = {"x": 0.4, "y": 2.0, "z": 0.1}
    out = script.run(
        robot.ctx(), skill_a="skill-a", skill_b="skill-b", arm_a=-1, target_a=far, target_b=far
    )
    assert out["route"] == "refused"
    assert robot.moves == []


def test_loads_a_real_hub_skill(script):
    run, success = script.load_skill("checking-dual-arm-clearance")
    assert callable(run)
    assert success == {"clear", "too_close"}


# ---------------------------------------------------------------- one-arm-at-a-time server
SERIAL = {"serializes_moves": True}


def serial_robot(clearance):
    return Robot(clearance, describe=SERIAL, move_s=0.01)


def test_mode_is_detected_from_describe_robot(script, clearance):
    for describe, mode in (
        (SERIAL, "interleaved"),
        ({"concurrent_arms": True}, "threaded"),
        ({"ops": ["move", "stream_joints"]}, "threaded"),
        ({}, "interleaved"),
    ):
        assert script.detect_mode(Robot(clearance, describe=describe).ctx())[0] == mode
    ctx = FakeContext({})
    assert script.detect_mode(ctx)[0] == "interleaved"


def test_interleaved_alternates_server_steps_one_thread(script, clearance):
    robot = serial_robot(clearance)
    out_and_back = joints_skill(OUTWARD, {0: HOME, 1: HOME})
    register(script, out_and_back, out_and_back)
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b")
    assert out["route"] == "both_succeeded", out["detail"]
    assert out["mode"] == "interleaved"
    arms = [a for a, *_ in robot.moves]
    # 0.5 rad each way in 0.30 rad server steps: 2 segments per move, 4 per arm.
    assert arms == [0, 1] * 4
    assert not overlap(robot)
    qs = {0: [np.array(HOME)], 1: [np.array(HOME)]}
    for a, _, _, q in robot.moves:
        assert np.max(np.abs(np.array(q) - qs[a][-1])) <= 0.30 + 1e-9
        qs[a].append(np.array(q))
    np.testing.assert_allclose(qs[0][2], OUTWARD[0])
    np.testing.assert_allclose(robot.q[0], HOME)


def test_interleaved_waits_for_the_tool_centre_margin(script, clearance):
    robot = serial_robot(clearance)
    register(script, joints_skill(CROSS), joints_skill(OUTWARD))
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b", min_tcp_separation_m=0.45)
    assert out["route"] == "both_succeeded", out["detail"]
    assert out["waits"] >= 1
    # arm 0 cannot finish crossing until arm 1 has gone outward.
    last1 = max(i for i, m in enumerate(robot.moves) if m[0] == 1)
    last0 = max(i for i, m in enumerate(robot.moves) if m[0] == 0)
    assert last1 < last0
    np.testing.assert_allclose(robot.q[0], CROSS[0])


def test_interleaved_crossing_conflict_serialises(script, clearance):
    robot = serial_robot(clearance)
    there_and_back = joints_skill(CROSS, {0: HOME, 1: HOME})
    register(script, there_and_back, there_and_back)
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b")
    assert out["route"] == "both_succeeded", out["detail"]
    assert out["waits"] >= 1
    first_cross1 = next(i for i, m in enumerate(robot.moves) if m[0] == 1 and m[3][0] > 0.7)
    back0 = max(i for i, m in enumerate(robot.moves) if m[0] == 0)
    assert first_cross1 > back0  # arm 1 enters only after arm 0 came all the way back


def test_interleaved_failure_pauses_the_other(script, clearance):
    robot = serial_robot(clearance)

    def fails_late(ctx):
        ctx.tool("robot.move_to_joints", joint_config={"positions": OUTWARD[ctx.arm]})
        raise RuntimeError("grasp missed")

    register(
        script, fails_late, joints_skill(OUTWARD, {0: HOME, 1: HOME}, OUTWARD, {0: HOME, 1: HOME})
    )
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b")
    assert out["route"] == "one_failed", out["detail"]
    assert out["route_a"] == "raised"
    assert out["route_b"] == "paused"
    np.testing.assert_allclose(robot.q[1], HOME)


def test_interleaved_deadlock_times_out(script, clearance):
    robot = serial_robot(clearance)
    register(script, joints_skill(CROSS), joints_skill(CROSS))
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b", wait_timeout_s=0.3)
    assert out["route"] == "deadlock_timeout", out["detail"]
    assert out["route_b"] == "paused"


def test_interleaved_segment_not_arriving_ends_that_move(script, clearance):
    robot = serial_robot(clearance)
    ctx = robot.ctx()
    ctx._responses["robot.move_to_joints"] = lambda joint_config, arm_id=0: {"reached": False}
    replies = []

    def one(ctx):
        replies.append(
            ctx.tool("robot.move_to_joints", joint_config={"positions": OUTWARD[ctx.arm]})
        )
        return {"route": "done"}

    register(script, one, one)
    out = script.run(ctx, skill_a="skill-a", skill_b="skill-b")
    assert out["route"] == "both_succeeded"
    assert replies == [{"reached": False}] * 2
    assert ctx.call_count("robot.move_to_joints") == 2  # one segment each, not the rest


def test_pose_moves_are_cut_into_server_steps(script):
    start = {
        "position": {"x": 0.3, "y": 0.2, "z": 0.2},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    ctx = FakeContext({"robot.get_ee_pose": {"pose": start}})
    target = {"position": {"x": 0.42, "y": 0.2, "z": 0.2}, "rotation": start["rotation"]}
    segs = script._segments(
        ctx, 1, "robot.go_to_pose_cartesian", {"pose": target, "arm_id": 1}, 0.05, 0.30
    )
    assert len(segs) == 3
    assert [round(s[1]["pose"]["position"]["x"], 3) for s in segs] == [0.34, 0.38, 0.42]
    assert all(s[0] == "robot.go_to_pose_cartesian" and s[1]["arm_id"] == 1 for s in segs)


def test_sequential_runs_one_skill_then_the_other(script, clearance):
    robot = serial_robot(clearance)
    out_and_back = joints_skill(OUTWARD, {0: HOME, 1: HOME})
    register(script, out_and_back, out_and_back)
    out = script.run(
        robot.ctx(), skill_a="skill-a", skill_b="skill-b", mode="sequential", priority_arm=1
    )
    assert out["route"] == "both_succeeded", out["detail"]
    assert [a for a, *_ in robot.moves] == [0, 0, 1, 1]


def test_sequential_failure_skips_the_second(script, clearance):
    robot = serial_robot(clearance)
    register(script, lambda ctx: {"route": "grasp_missed"}, joints_skill(OUTWARD))
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b", mode="sequential")
    assert out["route"] == "one_failed"
    assert out["route_b"] == "not_run"
    assert robot.moves == []


def test_both_failing_is_both_failed(script, clearance):
    robot = Robot(clearance)
    register(script, lambda ctx: {"route": "missed"}, lambda ctx: {"route": "missed"})
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b")
    assert out["route"] == "both_failed", out["detail"]


class SharedSetpointRobot(Robot):
    """The twin's failure (a_concurrent_s0_r2): one Cartesian-servo setpoint shared by both
    arms, re-seeded by every move, so a move started while the other arm's is in flight sends
    that arm toward somewhere it was never asked to go (here: across the middle). Moves step
    every 10 ms and honour robot.stop."""

    def __init__(self, model):
        super().__init__(model)
        self.moving: set[int] = set()
        self.hijack: dict[int, list] = {}
        self.stopped: set[int] = set()

    def ctx(self):
        ctx = super().ctx()

        def move(joint_config, arm_id=0):
            for other in self.moving:
                self.hijack[other] = CROSS[other]  # the other arm's setpoint is overwritten
            self.moving.add(arm_id)
            try:
                start = self.q[arm_id].copy()
                for k in range(1, 31):
                    if arm_id in self.stopped:
                        return {"reached": False, "outcome": "stopped"}
                    goal = np.array(self.hijack.get(arm_id, joint_config["positions"]), float)
                    with self.lock:
                        self.q[arm_id] = start + (goal - start) * k / 30
                    time.sleep(0.03)
                self.moves.append((arm_id, 0.0, 0.0, list(joint_config["positions"])))
                return {"reached": True}
            finally:
                self.moving.discard(arm_id)

        ctx._responses["robot.move_to_joints"] = move
        ctx._responses["robot.stop"] = lambda arm_id=0: self.stopped.add(arm_id)
        return ctx


def test_regression_a_connector_moving_a_different_path_is_stopped_by_the_guard(script, clearance):
    robot = SharedSetpointRobot(clearance)
    out_and_back = joints_skill(OUTWARD, {0: HOME, 1: HOME}, OUTWARD, {0: HOME, 1: HOME})
    register(script, out_and_back, out_and_back)
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b", mode="threaded")
    assert out["route"] == "refused", out["detail"]
    assert "guard: measured clearance" in out["detail"]
    assert robot.stopped == {0, 1}
    # predicted every grant clear; what stopped it was the measurement
    assert out["min_predicted_clearance_m"] >= 0.03
    gap = clearance.run(robot.ctx())["min_clearance_m"]
    assert gap > -0.05  # stopped on the way in, not driven through


def test_the_guard_stays_quiet_on_an_honest_connector(script, clearance):
    robot = Robot(clearance)
    there_and_back = joints_skill(CROSS, {0: HOME, 1: HOME})
    register(script, there_and_back, there_and_back)
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b")
    assert out["route"] == "both_succeeded", out["detail"]
    assert "guard" not in out["detail"]


# ---------------------------------------------------------------- deadlock -> sequential
#: Scene b of the twin in miniature: both arms lift toward the middle (still 55 mm clear of each
#: other), then each one's crossing move would end in the other's way (-2 mm): a true deadlock.
LIFT = {0: [-0.5, 1.5, 1.0, 0.0, 0.0, 0.0], 1: [0.5, 1.5, 1.0, 0.0, 0.0, 0.0]}


def test_the_lift_poses_deadlock_for_real(clearance):
    robot = Robot(clearance)
    robot.q = {0: np.array(LIFT[0]), 1: np.array(LIFT[1])}
    assert clearance.run(robot.ctx())["min_clearance_m"] > 0.03
    robot.q = {0: np.array(CROSS[0]), 1: np.array(LIFT[1])}
    assert clearance.run(robot.ctx())["min_clearance_m"] < 0.03


MID = {0: [-0.25, 1.25, 1.0, 0.0, 0.0, 0.0], 1: [0.25, 1.25, 1.0, 0.0, 0.0, 0.0]}


class CartesianRobot(Robot):
    """Adds the Cartesian side: robot.get_ee_pose, robot.solve_ik (numeric, seeded from the
    arm's joints) and robot.go_to_pose_cartesian (lands on that solution); grippers logged."""

    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.grip_calls: list = []
        self.cartesian: list[tuple[int, list]] = []

    def _ik(self, arm, pose):
        from scipy.optimize import least_squares

        goal = self.m._pose_matrix(pose)

        def err(q):
            t = self.m._pose_matrix(self._pose(arm, q))
            return np.concatenate(
                [t[:3, 3] - goal[:3, 3], 0.1 * (t[:3, :3] - goal[:3, :3]).ravel()]
            )

        sol = least_squares(err, self.q[arm], xtol=1e-10, ftol=1e-10)
        ok = float(np.linalg.norm(err(sol.x)[:3])) < 0.002
        return sol.x, ok

    def ctx(self):
        ctx = super().ctx()

        def ee(arm_id=0):
            with self.lock:
                return {"pose": self._pose(arm_id, self.q[arm_id])}

        def ik(pose, arm_id=0):
            q, ok = self._ik(arm_id, pose)
            return {
                "success": ok,
                "within_servo_tolerance": ok,
                "joint_config": {"positions": q.tolist()},
            }

        def cart(pose, arm_id=0):
            q, ok = self._ik(arm_id, pose)
            if not ok:
                return {"reached": False}
            time.sleep(self.move_s)
            with self.lock:
                self.q[arm_id] = q
            self.cartesian.append((arm_id, q.tolist()))
            self.moves.append((arm_id, 0.0, 0.0, q.tolist()))
            return {"reached": True}

        ctx._responses.update(
            {
                "robot.get_ee_pose": ee,
                "robot.solve_ik": ik,
                "robot.go_to_pose_cartesian": cart,
                "robot.open_gripper": lambda arm_id=0: self.grip_calls.append(("open", arm_id)),
                "robot.close_gripper": lambda arm_id=0: self.grip_calls.append(("close", arm_id)),
            }
        )
        return ctx


@pytest.mark.parametrize("serial", [False, True])
def test_a_deadlock_backs_the_yielding_arm_off_along_its_path(script, clearance, serial):
    describe = SERIAL if serial else {"concurrent_arms": True}
    robot = CartesianRobot(clearance, describe=describe, move_s=0.01 if serial else MOVE_S)
    job = joints_skill(MID, LIFT, CROSS, {0: HOME, 1: HOME})
    register(script, job, job)
    out = script.run(robot.ctx(), skill_a="skill-a", skill_b="skill-b", wait_timeout_s=0.3)
    assert out["route"] == "both_succeeded", out["detail"]
    assert out["sequential_fallback"] is True
    assert "arm 1 backed off along its path" in out["detail"]
    # arm 1 backed off one recorded waypoint (LIFT -> MID), Cartesian, orientation held, no
    # gripper call: just far enough for arm 0's crossing move to clear
    assert len(robot.cartesian) == 1 and robot.cartesian[0][0] == 1
    assert robot.grip_calls == []
    q_back = np.array(robot.cartesian[0][1])
    mid_tcp = robot._pose(1, MID[1])["position"]
    back_tcp = robot._pose(1, q_back)
    assert math.dist([mid_tcp[k] for k in "xyz"], [back_tcp["position"][k] for k in "xyz"]) < 0.003
    lift_rot = robot.m._pose_matrix(robot._pose(1, LIFT[1]))[:3, :3]
    assert np.allclose(robot.m._pose_matrix(back_tcp)[:3, :3], lift_rot, atol=1e-2)
    ends = [(a, q) for a, _, _, q in robot.moves]
    back1 = next(i for i, (a, q) in enumerate(ends) if a == 1 and np.allclose(q, q_back))
    cross0 = next(i for i, (a, q) in enumerate(ends) if a == 0 and np.allclose(q, CROSS[0]))
    done0 = max(i for i, (a, _) in enumerate(ends) if a == 0)
    cross1 = next(i for i, (a, q) in enumerate(ends) if a == 1 and np.allclose(q, CROSS[1]))
    assert back1 < cross0 < done0 < cross1
    assert not any(a == 1 and np.allclose(q, HOME) for a, q in ends[:done0])
    np.testing.assert_allclose(robot.q[0], HOME)
    np.testing.assert_allclose(robot.q[1], HOME)


def test_on_deadlock_pause_keeps_the_old_stop(script, clearance):
    robot = Robot(clearance)
    job = joints_skill(LIFT, CROSS, {0: HOME, 1: HOME})
    register(script, job, job)
    out = script.run(
        robot.ctx(), skill_a="skill-a", skill_b="skill-b", wait_timeout_s=0.3, on_deadlock="pause"
    )
    assert out["route"] == "deadlock_timeout", out["detail"]
    assert out["sequential_fallback"] is False


# ---------------------------------------------------------------- twin scene b, recorded
#: b_cint_s0_r3 (YAM station twin, 2026-11-13): both arms holding their cubes over their picks,
#: joints as robot.get_observation returned them (MuJoCo / model convention: the twin's grasp
#: sites at these joints are where its ee_pose said, (0.570, +-0.180)), and arm 0's place
#: pose solved by robot.solve_ik. MuJoCo mesh gap (yam_i2rt_bimanual.xml, mj_geomDistance over
#: link1-5 and fingertips): 245.0 mm now, 14.3 mm with arm 0 at its place and arm 1 standing.
TWIN_NOW = {
    0: [-0.386, 1.841, 1.498, -1.228, 0.0, 0.001],
    1: [0.365, 1.841, 1.498, -1.228, 0.0, -0.001],
}
TWIN_ARM0_PLACE = [-0.97, 2.229, 2.067, -1.409, 0.0, 0.001]
MESH_NOW, MESH_ARM0_PLACE = 0.2450, 0.0143


def test_twin_scene_b_prediction_is_the_place_pose_not_a_wrong_input(clearance):
    robot = Robot(clearance)
    robot.q = {a: np.array(q) for a, q in TWIN_NOW.items()}
    for arm, y in ((0, 0.180), (1, -0.180)):
        p = robot._pose(arm, robot.q[arm])["position"]
        assert abs(p["y"] - y) < 0.01  # same joints, same tool point as the twin reported
    now = clearance.run(robot.ctx())["min_clearance_m"]
    there = clearance.run(robot.ctx(), trajectory_a={"waypoints": [{"positions": TWIN_ARM0_PLACE}]})
    # capsules bound the meshes from below by tens of millimetres, never hundreds
    assert MESH_NOW - 0.06 < now <= MESH_NOW
    assert MESH_ARM0_PLACE - 0.06 < there["min_clearance_m"] <= MESH_ARM0_PLACE
    assert there["at_fraction"] == pytest.approx(1.0)


def test_a_hardware_joint1_offset_does_not_change_the_clearance(clearance):
    """#247: model joint1 = hardware joint1 - pi/2. A connector reporting hardware joints (with
    its own FK and tool poses consistent with them) gives the same answer: each base is located
    from the tool pose, and joint1 turns about the base's z through its origin, so the offset
    only turns the located base."""
    off = np.array([np.pi / 2, 0, 0, 0, 0, 0])
    robot = Robot(clearance)
    robot.q = {a: np.array(q) for a, q in TWIN_NOW.items()}
    model = clearance.run(robot.ctx(), trajectory_a={"waypoints": [{"positions": TWIN_ARM0_PLACE}]})
    ctx = robot.ctx()
    obs = ctx._responses["robot.get_observation"]

    def hw_obs():
        o = obs()
        for a, arm in enumerate(o["arms"]):
            arm["joint_state"]["positions"] = (robot.q[a] + off).tolist()
        return o

    ctx._responses["robot.get_observation"] = hw_obs
    ctx._responses["robot.forward_kinematics"] = lambda joint_configs, arm_id=0: {
        "poses": [robot._pose(arm_id, np.asarray(q) - off) for q in joint_configs]
    }
    hw = clearance.run(
        ctx, trajectory_a={"waypoints": [{"positions": list(np.array(TWIN_ARM0_PLACE) + off)}]}
    )
    assert hw["route"] == model["route"]
    assert hw["min_clearance_m"] == pytest.approx(model["min_clearance_m"], abs=1e-6)
