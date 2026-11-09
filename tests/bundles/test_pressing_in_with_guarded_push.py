"""pressing-in-with-guarded-push: FakeContext tests of the stepped push, its stops and the routing."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "pressing-in-with-guarded-push"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
START = (0.40, 0.10, 0.10)
FAST = {"settle_period": 0.0}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["guarded_push"].module


class Arm:
    """A position-servoed hand pushing down: it lands ``sag`` above its command (a
    real arm's free-motion lag) unless a seat holds it higher. The seat is at
    ``seat`` (z); when the command leads the held hand by more than ``give_at`` it
    gives and drops ``drop`` (a snap-fit), once. ``efforts`` publishes joint efforts
    of ``stiffness`` x how far the hand is held. ``refuse(target)`` refuses a move."""

    def __init__(
        self,
        start=START,
        seat=None,
        sag=0.0,
        give_at=None,
        drop=0.0,
        efforts=False,
        stiffness=200.0,
        refuse=None,
        raise_refusal=False,
        lists=False,
    ):
        self.p = np.array(start, dtype=float)
        self.cmd = self.p.copy()
        self.seat, self.sag = seat, sag
        self.give_at, self.drop = give_at, drop
        self.efforts, self.stiffness = efforts, stiffness
        self.refuse = refuse or (lambda target: None)
        self.raise_refusal, self.lists = raise_refusal, lists
        self.moves = []

    def _land(self, target):
        z = target[2] + self.sag
        if self.seat is not None:
            held = self.seat - z
            if self.give_at is not None and held > self.give_at:
                self.seat -= self.drop
                self.give_at = None
            z = max(z, self.seat)
        return np.array([target[0], target[1], z])

    def go(self, pose=None, position=None, quat_xyzw=None, label="", **kwargs):
        target = np.array(
            position if self.lists else [pose["position"][k] for k in "xyz"], dtype=float
        )
        self.moves.append((target.copy(), kwargs, label))
        why = self.refuse(target)
        if why:
            if self.raise_refusal or self.lists:
                raise ToolError("robot.go_to_pose", why)
            return {"reached": False, "outcome": "unreachable", "reason": why}
        self.cmd = target
        self.p = self._land(target)
        if self.p[2] - target[2] > self.sag + 0.004:
            if self.lists:
                raise RuntimeError("go_to_pose: the tool did not reach its target")
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

    def observation(self, **kw):
        js = {"positions": np.zeros(6)}
        if self.efforts:
            held = max(0.0, self.p[2] - self.cmd[2] - self.sag)
            js["efforts"] = np.array(
                [0.0, 2.0, -1.0, 0.3, 0.1, 0.0]
            ) + self.stiffness * held * np.array([0, 1, 1, 0, 0, 0])
        return {"cameras": [], "arms": [{"joint_state": js}]}

    def ctx(self, ik=None):
        tools = {"robot.get_ee_pose": self.pose, "robot.go_to_pose": self.go}
        if not self.lists:
            tools["robot.solve_ik"] = (
                ik if ik is not None else {"success": True, "within_servo_tolerance": True}
            )
            tools["robot.get_observation"] = self.observation
        return FakeContext(tools)

    def labels(self):
        return [m[2] for m in self.moves]


def test_free_air_travels_max_translation_and_is_seated(script):
    arm = Arm()
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "seated"
    assert out["travelled"] == pytest.approx(0.008, abs=1e-6)
    assert out["contact_travel"] == -1.0
    assert not out["relieved"]
    assert "no resistance" in out["detail"]
    assert out["remaining"] == pytest.approx(0.0, abs=1e-6)


def test_hard_seat_hits_the_force_limit_and_unloads(script):
    arm = Arm(seat=START[2] - 0.003)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "force_limit_no_travel"
    assert out["signal"] == "lag"
    assert out["travelled"] == pytest.approx(0.003, abs=1e-6)
    assert out["contact_travel"] == pytest.approx(0.003, abs=1e-6)
    assert 0.006 - 1e-9 < out["final_lag"] < 0.006 + 0.0011  # overshoots by at most a step
    assert out["relieved"]
    assert arm.moves[-1][0][2] == pytest.approx(START[2] - 0.003, abs=1e-6)
    assert out["remaining"] == pytest.approx(0.005, abs=1e-6)


def test_snap_fit_giving_way_is_detected(script):
    arm = Arm(seat=START[2] - 0.002, give_at=0.0035, drop=0.004)
    out = script.run(arm.ctx(), max_translation=0.010, **FAST)
    assert out["route"] == "snap_detected"
    assert out["peak_lag"] >= 0.002
    assert out["travelled"] == pytest.approx(0.006, abs=1e-6)
    assert out["contact_travel"] == pytest.approx(0.002, abs=1e-6)


def test_snap_without_stopping_continues_to_the_goal(script):
    arm = Arm(seat=START[2] - 0.002, give_at=0.003, drop=0.004)
    out = script.run(arm.ctx(), max_translation=0.005, stop_on_snap=False, **FAST)
    assert out["route"] == "seated"
    assert "snap felt" in out["detail"]


def test_sagging_arm_keeps_its_free_lag_out_of_the_push(script):
    arm = Arm(sag=0.005)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "seated"
    assert out["contact_travel"] == -1.0
    assert abs(out["final_lag"]) < 1e-6


def test_sagging_arm_still_feels_the_seat(script):
    arm = Arm(sag=0.005, seat=START[2] - 0.004)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "force_limit_no_travel"
    assert out["contact_travel"] == pytest.approx(0.004, abs=1e-6)


def test_target_reached_is_seated(script):
    arm = Arm()
    target = {"x": START[0], "y": START[1], "z": START[2] - 0.0035}
    out = script.run(arm.ctx(), target=target, **FAST)
    assert out["route"] == "seated"
    assert out["travelled"] == pytest.approx(0.004, abs=1e-6)
    assert "target" in out["detail"]


def test_lockup_near_the_target_counts_as_seated_only_within_tolerance(script):
    target = {"x": START[0], "y": START[1], "z": START[2] - 0.004}
    seat = START[2] - 0.003
    out = script.run(Arm(seat=seat).ctx(), target=target, goal_tolerance_if_lockup=0.0015, **FAST)
    assert out["route"] == "seated"
    assert "lockup" in out["detail"]
    out = script.run(Arm(seat=seat).ctx(), target=target, **FAST)
    assert out["route"] == "force_limit_no_travel"


def test_target_already_behind_the_hand_moves_nothing(script):
    arm = Arm()
    out = script.run(arm.ctx(), target={"x": 0.4, "y": 0.1, "z": START[2] + 0.002}, **FAST)
    assert out["route"] == "seated"
    assert arm.moves == []


def test_unreachable_end_is_blocked_before_moving(script):
    arm = Arm()
    out = script.run(arm.ctx(ik={"success": True, "within_servo_tolerance": False}), **FAST)
    assert out["route"] == "blocked"
    assert "out of reach" in out["detail"]
    assert arm.moves == []


def test_ik_preview_raising_unreachable_is_blocked(script):
    def ik(**kw):
        raise ToolError("robot.solve_ik", "pose unreachable")

    arm = Arm()
    out = script.run(arm.ctx(ik=ik), **FAST)
    assert out["route"] == "blocked"
    assert arm.moves == []


@pytest.mark.parametrize("raised", [False, True])
def test_refused_step_is_blocked(script, raised):
    arm = Arm(
        refuse=lambda t: "joint limit" if t[2] < START[2] - 0.0025 else None, raise_refusal=raised
    )
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "blocked"
    assert "step 3 refused" in out["detail"]


def test_refused_hold_is_blocked(script):
    arm = Arm(refuse=lambda t: "no")
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "blocked"
    assert "holding the start" in out["detail"]


def test_floor_ends_the_push(script):
    arm = Arm()
    out = script.run(arm.ctx(), floor_z=START[2] - 0.004, **FAST)
    assert out["route"] == "blocked"
    assert "floor" in out["detail"]
    assert min(m[0][2] for m in arm.moves) >= START[2] - 0.004 - 1e-9


def test_hand_on_the_floor_moves_nothing(script):
    arm = Arm()
    out = script.run(arm.ctx(), floor_z=START[2] - 0.0005, **FAST)
    assert out["route"] == "blocked"
    assert arm.moves == []


def test_effort_signal_stops_before_the_lag_limit(script):
    arm = Arm(seat=START[2] - 0.003, efforts=True, stiffness=200.0)
    out = script.run(arm.ctx(), signal="effort", **FAST)
    assert out["route"] == "force_limit_no_travel"
    assert out["signal"] == "effort"
    assert out["efforts_available"]
    # 0.5 N.m at 200 N.m/m is 2.5 mm held: stops on the third held step.
    assert out["steps"] == 6


def test_auto_reports_both_signals_when_both_fire(script):
    arm = Arm(seat=START[2] - 0.003, efforts=True, stiffness=90.0)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "force_limit_no_travel"
    assert out["signal"] == "lag+effort"


def test_effort_signal_without_efforts_is_blocked(script):
    arm = Arm()
    out = script.run(arm.ctx(), signal="effort", **FAST)
    assert out["route"] == "blocked"
    assert arm.moves == []


def test_station_connector_lists_and_raised_shorts(script):
    arm = Arm(seat=START[2] - 0.003, lists=True)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "force_limit_no_travel"
    assert all(m[2].startswith("push") for m in arm.moves)
    assert arm.labels()[-1] == "push: relieve"


def test_sideways_push_along_x(script):
    class Wall(Arm):
        def _land(self, target):
            return np.array([min(target[0], START[0] + 0.002), target[1], target[2]])

    arm = Wall()
    out = script.run(arm.ctx(), direction={"x": 1.0, "y": 0.0, "z": 0.0}, **FAST)
    assert out["route"] == "force_limit_no_travel"
    assert out["travelled"] == pytest.approx(0.002, abs=1e-6)
    assert all(abs(m[0][2] - START[2]) < 1e-9 for m in arm.moves)


def test_arm_id_and_knobs_are_passed_only_when_set(script):
    arm = Arm()
    script.run(arm.ctx(), arm_id=1, step_max_steps=50, **FAST)
    assert arm.moves[1][1] == {"tolerance": 0.001, "max_steps": 50, "arm_id": 1}
    arm = Arm()
    script.run(arm.ctx(), step_tolerance=0.0, **FAST)
    assert arm.moves[1][1] == {}


def test_bad_arguments_raise(script):
    arm = Arm()
    with pytest.raises(ValueError):
        script.run(arm.ctx(), signal="force")
    with pytest.raises(ValueError):
        script.run(arm.ctx(), contact_lag=0.007)
    with pytest.raises(ValueError):
        script.run(arm.ctx(), direction={"x": 0.0, "y": 0.0, "z": 0.0})
