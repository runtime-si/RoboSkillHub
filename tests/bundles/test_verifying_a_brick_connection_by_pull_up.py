"""verifying-a-brick-connection-by-pull-up: FakeContext tests of the stepped lift and its verdicts."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "verifying-a-brick-connection-by-pull-up"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
START = (0.40, 0.10, 0.05)
FAST = {"settle_period": 0.0}
GRIPPER = {"span_m": 0.08, "width_fit": {"at_open_m": 0.08, "at_closed_m": 0.0}}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["pull_up_check"].module


class Arm:
    """A position-servoed hand gripping a brick. It lands ``sag`` under its command
    (gravity). A clutched brick holds it at ``hold_z`` + ``give`` x pull until the
    pull (command minus hold, mm) passes ``breaks_at``, then it follows freely.
    ``fraction`` is the gripper reading; ``drops_after`` steps of free lift it reads
    0 (the brick slipped out). ``efforts`` publishes ``stiffness`` x the pull."""

    def __init__(
        self,
        clutched=False,
        sag=0.0,
        breaks_at=None,
        fraction=0.3,
        drops_after=None,
        efforts=False,
        stiffness=200.0,
        refuse=None,
        lists=False,
    ):
        self.p = np.array(START, dtype=float)
        self.cmd = self.p.copy()
        self.clutched, self.sag = clutched, sag
        self.breaks_at = breaks_at
        self.fraction, self.drops_after = fraction, drops_after
        self.efforts, self.stiffness = efforts, stiffness
        self.refuse = refuse or (lambda target: None)
        self.lists = lists
        self.moves = []
        self.free_lifts = 0
        self.opened = 0

    def _land(self, target):
        z = target[2] - self.sag
        if self.clutched:
            pull = z - START[2]
            if self.breaks_at is not None and pull > self.breaks_at:
                self.clutched = False
            else:
                z = min(z, START[2])
        return np.array([target[0], target[1], z])

    def go(self, pose=None, position=None, quat_xyzw=None, label="", **kwargs):
        target = np.array(
            position if self.lists else [pose["position"][k] for k in "xyz"], dtype=float
        )
        self.moves.append((target.copy(), kwargs, label))
        why = self.refuse(target)
        if why:
            if self.lists:
                raise ToolError("robot.go_to_pose", why)
            return {"reached": False, "outcome": "unreachable", "reason": why}
        self.cmd = target
        self.p = self._land(target)
        if target[2] > START[2] + 1e-9 and not self.clutched:
            self.free_lifts += 1
        if target[2] - self.p[2] > self.sag + 0.004:
            return {"reached": False, "outcome": "timeout"}
        return {"reached": True, "outcome": "reached"}

    def pose(self, **kw):
        if self.lists:
            return {"position": self.p.tolist(), "quat_xyzw": [1.0, 0.0, 0.0, 0.0]}
        return {
            "pose": {
                "position": dict(zip("xyz", map(float, self.p), strict=True)),
                "rotation": DOWN,
            }
        }

    def gripper(self, **kw):
        if self.drops_after is not None and self.free_lifts >= self.drops_after:
            return {"position": 0.0}
        return {"position": self.fraction}

    def observation(self, **kw):
        js = {"positions": np.zeros(6)}
        if self.efforts:
            pull = max(0.0, self.cmd[2] - self.sag - self.p[2])
            js["efforts"] = np.array(
                [0.0, 2.0, -1.0, 0.3, 0.1, 0.0]
            ) + self.stiffness * pull * np.array([0, 1, 1, 0, 0, 0])
        return {"cameras": [], "arms": [{"joint_state": js}]}

    def open(self, **kw):
        self.opened += 1
        return {"position": 1.0}

    def ctx(self, ik=None, gripper=True):
        tools = {
            "robot.get_ee_pose": self.pose,
            "robot.go_to_pose": self.go,
            "robot.open_gripper": self.open,
        }
        if gripper:
            tools["robot.get_gripper"] = self.gripper
            tools["robot.describe_gripper"] = GRIPPER
        if not self.lists:
            tools["robot.solve_ik"] = (
                ik if ik is not None else {"success": True, "within_servo_tolerance": True}
            )
            tools["robot.get_observation"] = self.observation
        return FakeContext(tools)


def test_clutched_brick_is_connected_put_back_and_released(script):
    arm = Arm(clutched=True)
    out = script.run(arm.ctx(), hold_lag=0.0017, **FAST)
    assert out["route"] == "connected"
    assert out["connected"] and out["released"]
    assert out["signal"] == "lag"
    # 0.5 mm steps, held: fires on the fourth (2.0 mm over a 1.7 mm threshold).
    assert out["steps"] == 4
    assert out["rise"] == pytest.approx(0.0, abs=1e-9)
    # Put back to the start before opening, then the retreat.
    assert arm.moves[-2][0][2] == pytest.approx(START[2])
    assert arm.moves[-1][0][2] == pytest.approx(START[2] + 0.03)
    assert arm.opened == 1


def test_clutched_brick_kept_when_release_is_off(script):
    arm = Arm(clutched=True)
    out = script.run(arm.ctx(), release=False, **FAST)
    assert out["route"] == "connected"
    assert not out["released"]
    assert arm.opened == 0
    assert arm.moves[-1][0][2] == pytest.approx(START[2])


def test_loose_brick_comes_up_and_is_put_back(script):
    arm = Arm()
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "not_connected"
    assert not out["connected"]
    assert out["rise"] == pytest.approx(0.003, abs=1e-9)
    assert out["steps"] == 6
    assert arm.moves[-1][0][2] == pytest.approx(START[2])
    assert arm.opened == 0
    assert "put back" in out["detail"]


def test_loose_brick_left_lifted_without_put_back(script):
    arm = Arm()
    out = script.run(arm.ctx(), put_back=False, **FAST)
    assert out["route"] == "not_connected"
    assert arm.moves[-1][0][2] == pytest.approx(START[2] + 0.003)


def test_sagging_arm_is_not_read_as_a_hold(script):
    arm = Arm(sag=0.005)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "not_connected"
    assert abs(out["lag_over_baseline"]) < 1e-9


def test_weak_clutch_letting_go_early_reads_not_connected(script):
    arm = Arm(clutched=True, breaks_at=0.0002)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "not_connected"


def test_empty_jaws_before_the_lift(script):
    arm = Arm(fraction=0.0)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "not_holding"
    assert arm.moves == []
    assert out["jaw_width"] == pytest.approx(0.0)


def test_brick_slipping_out_during_the_lift(script):
    arm = Arm(drops_after=3)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "not_holding"
    assert "slipped" in out["detail"]


def test_no_gripper_tools_skips_the_holding_check(script):
    arm = Arm(fraction=0.0)
    out = script.run(arm.ctx(gripper=False), **FAST)
    assert out["route"] == "not_connected"
    assert out["jaw_width"] == -1.0


def test_unreachable_top_is_blocked_before_moving(script):
    arm = Arm()
    out = script.run(arm.ctx(ik={"success": False}), **FAST)
    assert out["route"] == "blocked"
    assert arm.moves == []


def test_refused_lift_step_is_blocked_and_put_back(script):
    arm = Arm(refuse=lambda t: "joint limit" if t[2] > START[2] + 0.0012 else None)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "blocked"
    assert "lift step 3 refused" in out["detail"]
    assert arm.moves[-1][0][2] == pytest.approx(START[2])


def test_effort_signal(script):
    arm = Arm(clutched=True, efforts=True, stiffness=600.0)
    out = script.run(arm.ctx(), signal="effort", **FAST)
    assert out["route"] == "connected"
    assert out["signal"] == "effort"
    # 0.5 N.m at 600 N.m/m is 0.83 mm of pull: fires on the second step.
    assert out["steps"] == 2


def test_effort_signal_without_efforts_is_blocked(script):
    arm = Arm()
    out = script.run(arm.ctx(), signal="effort", **FAST)
    assert out["route"] == "blocked"
    assert arm.moves == []


def test_station_connector_lists(script):
    arm = Arm(clutched=True, lists=True)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "connected"
    assert all(m[2] for m in arm.moves)


def test_knobs_passed_only_when_set(script):
    arm = Arm()
    script.run(arm.ctx(), arm_id=1, step_max_steps=40, **FAST)
    assert arm.moves[0][1] == {"tolerance": 0.0002, "max_steps": 40, "arm_id": 1}
    arm = Arm()
    script.run(arm.ctx(), step_tolerance=0.0, **FAST)
    assert arm.moves[0][1] == {}


def test_bad_arguments_raise(script):
    with pytest.raises(ValueError):
        script.run(Arm().ctx(), signal="force")
    with pytest.raises(ValueError):
        script.run(Arm().ctx(), lift_height=0.0)
