"""detecting-contact-from-joint-effort: FakeContext tests of the stepping, both signals and the routing."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "detecting-contact-from-joint-effort"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
START = (0.40, 0.10, 0.10)
FAST = {"settle_period": 0.0}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["detect_contact"].module


class Arm:
    """A hand that lands where commanded, ``sag`` short of it along -z (a real arm's
    free-motion lag), and that cannot go under ``surface``. With ``efforts`` the
    observation carries joint efforts: ``gravity`` plus ``stiffness`` x how far the
    surface holds the hand above its command; ``noise`` adds a repeating jitter.
    ``refuse(target) -> reason`` refuses a move (routed, or raised with
    ``raise_refusal``); a blocked move raises a station's "did not reach its target"
    with ``raise_short`` or routes a servo's timeout otherwise."""

    def __init__(
        self,
        start=START,
        surface=None,
        sag=0.0,
        efforts=False,
        stiffness=200.0,
        noise=0.0,
        velocities=None,
        refuse=None,
        raise_refusal=False,
        raise_short=False,
        lists=False,
    ):
        self.p = np.array(start, dtype=float)
        self.cmd = self.p.copy()
        self.surface, self.sag = surface, sag
        self.efforts, self.stiffness, self.noise = efforts, stiffness, noise
        self.velocities = velocities
        self.refuse = refuse or (lambda target: None)
        self.raise_refusal, self.raise_short = raise_refusal, raise_short
        self.lists = lists
        self.moves = []
        self.reads = 0

    def _land(self, target):
        z = target[2] + self.sag
        if self.surface is not None:
            z = max(z, self.surface)
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
        blocked = self.p[2] - target[2] > self.sag + 0.012
        if blocked and (self.raise_short or self.lists):
            raise RuntimeError("go_to_pose: the tool did not reach its target (14 mm, 0.2 deg off)")
        if blocked:
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
            self.reads += 1
            held = max(0.0, self.p[2] - self.cmd[2] - self.sag)
            eff = np.array([0.0, 2.0, -1.0, 0.3, 0.1, 0.0]) + self.stiffness * held * np.array(
                [0, 1, 1, 0, 0, 0]
            )
            eff = eff + self.noise * ((-1) ** self.reads)
            js["efforts"] = eff
            if self.velocities is not None:
                js["velocities"] = np.full(6, self.velocities)
        return {"cameras": [], "arms": [{"joint_state": js, "gripper_fraction": 0.0}]}

    def ctx(self, ik=None, observation=True, gripper=True):
        tools = {
            "robot.get_ee_pose": self.pose,
            "robot.go_to_pose": self.go,
        }
        if not self.lists:
            tools["robot.solve_ik"] = (
                ik if ik is not None else {"success": True, "within_servo_tolerance": True}
            )
            if observation:
                tools["robot.get_observation"] = self.observation
            if gripper:
                tools["robot.close_gripper"] = {"position": 0.0}
        return FakeContext(tools)


def _z(v):
    return float(v["z"])


def test_lag_finds_a_surface_and_backs_off(script):
    arm = Arm(surface=0.05)
    ctx = arm.ctx()
    out = script.run(ctx, end={"x": 0.40, "y": 0.10, "z": 0.03}, **FAST)
    assert out["route"] == "contact", out["detail"]
    assert out["contact"] is True and out["signal"] == "lag"
    assert out["efforts_available"] is False
    assert _z(out["contact_position"]) == pytest.approx(0.05)
    assert out["lag_over_baseline"] > 0.003
    # The last move backs off 3 mm up from where the hand stood.
    np.testing.assert_allclose(arm.moves[-1][0], [0.40, 0.10, 0.053], atol=1e-9)
    # Steps of 2.5 mm, each with the 1 mm arrival ball and no max_steps.
    steps = [m[0][2] for m in arm.moves[:-1]]
    np.testing.assert_allclose(np.diff(steps), -0.0025, atol=1e-9)
    assert all(m[1] == {"tolerance": 0.001} for m in arm.moves)
    assert ctx.call_count("robot.close_gripper") == 1
    # No efforts in the first observation: it is not asked again.
    assert ctx.call_count("robot.get_observation") == 1


def test_free_line_ends_free_where_it_was_sent(script):
    arm = Arm(surface=0.0)
    out = script.run(arm.ctx(), end={"x": 0.40, "y": 0.10, "z": 0.06}, **FAST)
    assert out["route"] == "free", out["detail"]
    assert out["contact"] is False
    assert _z(out["contact_position"]) == pytest.approx(0.06)
    assert out["travelled"] == pytest.approx(0.04)
    assert out["steps"] == 16


def test_a_steady_sag_is_the_baseline_not_contact(script):
    # A real YAM stands ~5 mm short of its command in free air: no false alarm.
    arm = Arm(surface=0.0, sag=0.005)
    out = script.run(arm.ctx(), end={"x": 0.40, "y": 0.10, "z": 0.06}, **FAST)
    assert out["route"] == "free", out["detail"]
    assert out["baseline_lag"] == pytest.approx(0.005)


def test_contact_with_a_sag_reads_the_stop(script):
    arm = Arm(surface=0.07, sag=0.005)
    out = script.run(
        arm.ctx(),
        start={"x": 0.40, "y": 0.10, "z": 0.10},
        end={"x": 0.40, "y": 0.10, "z": 0.05},
        **FAST,
    )
    assert out["route"] == "contact", out["detail"]
    assert _z(out["contact_position"]) == pytest.approx(0.07)


def test_tip_reach_pushes_the_contact_point_along_the_line(script):
    arm = Arm(surface=0.05)
    out = script.run(arm.ctx(), end={"x": 0.40, "y": 0.10, "z": 0.03}, tip_reach=0.0102, **FAST)
    assert _z(out["contact_point"]) == pytest.approx(0.05 - 0.0102)


def test_start_is_visited_first_and_its_lag_is_the_first_sample(script):
    arm = Arm(start=(0.30, 0.0, 0.2), surface=0.05)
    out = script.run(
        arm.ctx(),
        start={"x": 0.40, "y": 0.10, "z": 0.08},
        end={"x": 0.40, "y": 0.10, "z": 0.03},
        **FAST,
    )
    assert out["route"] == "contact"
    np.testing.assert_allclose(arm.moves[0][0], [0.40, 0.10, 0.08], atol=1e-9)
    assert out["travelled"] == pytest.approx(0.03)


def test_effort_fires_before_lag_when_published(script):
    # A stiff contact: the hand is held 1 mm over its command, too little for lag.
    arm = Arm(surface=0.05, efforts=True, stiffness=2000.0)
    ctx = arm.ctx()
    out = script.run(ctx, end={"x": 0.40, "y": 0.10, "z": 0.03}, **FAST)
    assert out["route"] == "contact", out["detail"]
    assert out["efforts_available"] is True
    assert out["signal"] == "effort"
    assert out["effort_ratio"] > 1.0
    assert _z(out["contact_position"]) == pytest.approx(0.05)
    assert ctx.call_count("robot.get_observation") > 5


def test_effort_noise_raises_the_threshold(script):
    # 0.4 N.m of alternating jitter: sigma-scaled thresholds keep it free.
    arm = Arm(surface=0.0, efforts=True, noise=0.4)
    out = script.run(arm.ctx(), end={"x": 0.40, "y": 0.10, "z": 0.06}, signal="effort", **FAST)
    assert out["route"] == "free", out["detail"]
    assert out["effort_ratio"] < 1.0


def test_moving_samples_are_dropped_by_the_speed_gate(script):
    # Every sample moving: no effort window, so effort-only cannot decide.
    arm = Arm(surface=0.05, efforts=True, stiffness=2000.0, velocities=1.0)
    out = script.run(arm.ctx(), end={"x": 0.40, "y": 0.10, "z": 0.03}, signal="effort", **FAST)
    assert out["route"] == "no_signal", out["detail"]


def test_signal_effort_without_efforts_is_no_signal_and_does_not_move(script):
    arm = Arm(surface=0.05)
    out = script.run(arm.ctx(), end={"x": 0.40, "y": 0.10, "z": 0.03}, signal="effort", **FAST)
    assert out["route"] == "no_signal"
    assert arm.moves == []


def test_signal_lag_ignores_efforts(script):
    arm = Arm(surface=0.05, efforts=True, stiffness=2000.0)
    ctx = arm.ctx()
    out = script.run(ctx, end={"x": 0.40, "y": 0.10, "z": 0.03}, signal="lag", **FAST)
    assert out["route"] == "contact" and out["signal"] == "lag"
    assert ctx.call_count("robot.get_observation") == 0


def test_too_short_a_line_is_no_signal(script):
    arm = Arm()
    out = script.run(arm.ctx(), end={"x": 0.40, "y": 0.10, "z": 0.098}, **FAST)
    assert out["route"] == "no_signal", out["detail"]


def test_jumpy_tracking_is_no_signal(script):
    arm = Arm(surface=0.0)
    lags = iter([0.0, 0.0] + [0.004] * 20)  # the hand jumps 4 mm ahead of its command and stays
    land = arm._land
    arm._land = lambda target: land(target) - np.array([0, 0, next(lags)])
    out = script.run(arm.ctx(), end={"x": 0.40, "y": 0.10, "z": 0.08}, **FAST)
    assert out["route"] == "no_signal", out["detail"]
    assert "jumped" in out["detail"]


def test_a_raised_did_not_reach_is_judged_not_refused(script):
    arm = Arm(surface=0.05, raise_short=True)
    out = script.run(arm.ctx(), end={"x": 0.40, "y": 0.10, "z": 0.0}, **FAST)
    assert out["route"] == "contact", out["detail"]
    assert _z(out["contact_position"]) == pytest.approx(0.05)


@pytest.mark.parametrize("raised", [False, True])
def test_a_refused_step_is_blocked(script, raised):
    arm = Arm(
        refuse=lambda t: "outside the workspace" if t[2] < 0.09 else None, raise_refusal=raised
    )
    out = script.run(arm.ctx(), end={"x": 0.40, "y": 0.10, "z": 0.06}, **FAST)
    assert out["route"] == "blocked"
    assert "refused" in out["detail"]


def test_an_unreached_start_is_blocked(script):
    arm = Arm(refuse=lambda t: "unreachable: no IK" if t[0] > 0.45 else None)
    out = script.run(
        arm.ctx(), start={"x": 0.5, "y": 0.1, "z": 0.1}, end={"x": 0.5, "y": 0.1, "z": 0.05}, **FAST
    )
    assert out["route"] == "blocked"
    assert "start" in out["detail"]


@pytest.mark.parametrize(
    "ik",
    [
        {"success": True, "within_servo_tolerance": False},
        {"success": False},
    ],
)
def test_ik_preview_blocks_an_out_of_reach_end(script, ik):
    arm = Arm()
    out = script.run(arm.ctx(ik=ik), end={"x": 0.40, "y": 0.10, "z": 0.06}, **FAST)
    assert out["route"] == "blocked"
    assert arm.moves == []


def test_ik_preview_raising_unreachable_blocks(script):
    def ik(**kw):
        raise ToolError("robot.solve_ik", "Target pose is unreachable: no plan")

    arm = Arm()
    out = script.run(arm.ctx(ik=ik), end={"x": 0.40, "y": 0.10, "z": 0.06}, **FAST)
    assert out["route"] == "blocked"


def test_check_reach_off_skips_the_preview(script):
    arm = Arm(surface=0.0)
    ctx = arm.ctx(ik={"success": False})
    out = script.run(ctx, end={"x": 0.40, "y": 0.10, "z": 0.06}, check_reach=False, **FAST)
    assert out["route"] == "free"
    assert ctx.call_count("robot.solve_ik") == 0


def test_floor_clips_the_line(script):
    arm = Arm(surface=0.0)
    out = script.run(arm.ctx(), end={"x": 0.40, "y": 0.10, "z": 0.0}, floor_z=0.06, **FAST)
    assert out["route"] == "free", out["detail"]
    assert "floor" in out["detail"]
    assert min(m[0][2] for m in arm.moves) == pytest.approx(0.06)


def test_start_under_the_floor_is_blocked(script):
    out = script.run(Arm().ctx(), end={"x": 0.40, "y": 0.10, "z": 0.0}, floor_z=0.12, **FAST)
    assert out["route"] == "blocked"


@pytest.mark.parametrize("end_z", [0.10, -0.10])
def test_degenerate_or_long_lines_are_blocked(script, end_z):
    out = script.run(Arm().ctx(), end={"x": 0.40, "y": 0.10, "z": end_z}, **FAST)
    assert out["route"] == "blocked"


def test_knobs_are_passed_only_when_set(script):
    arm = Arm(surface=0.0, start=(0.4, 0.1, 0.1))
    ctx = arm.ctx()
    script.run(
        ctx,
        end={"x": 0.40, "y": 0.10, "z": 0.09},
        step_tolerance=0.0,
        step_max_steps=120,
        arm_id=1,
        **FAST,
    )
    kwargs = ctx.calls_to("robot.go_to_pose")[0].kwargs
    assert kwargs["max_steps"] == 120 and kwargs["arm_id"] == 1
    assert "tolerance" not in kwargs


def test_a_connector_without_the_knobs_is_retried_without_them(script):
    arm = Arm(surface=0.0)
    go = arm.go

    def strict(pose, arm_id=0):
        return go(pose=pose)

    ctx = arm.ctx()
    ctx._responses["robot.go_to_pose"] = strict
    out = script.run(ctx, end={"x": 0.40, "y": 0.10, "z": 0.08}, **FAST)
    assert out["route"] == "free", out["detail"]


def test_station_lists_convention(script):
    # The threading station's connector: position / quat_xyzw lists and a label, no
    # IK preview, no close_gripper, no observation; blocked steps raise.
    arm = Arm(surface=0.05, lists=True)
    ctx = arm.ctx()
    out = script.run(ctx, end={"x": 0.40, "y": 0.10, "z": 0.03}, **FAST)
    assert out["route"] == "contact", out["detail"]
    assert _z(out["contact_position"]) == pytest.approx(0.05)
    assert all(label.startswith("push") for _, _, label in arm.moves[:-1])
    assert "not closed" in out["detail"]
    call = ctx.calls_to("robot.go_to_pose")[0].kwargs
    assert call["quat_xyzw"] == [1.0, 0.0, 0.0, 0.0] and "pose" not in call


def test_bad_signal_name_raises(script):
    with pytest.raises(ValueError):
        script.run(Arm().ctx(), end={"x": 0.4, "y": 0.1, "z": 0.06}, signal="force", **FAST)


def test_settle_waits_until_the_hand_is_still(script, monkeypatch):
    arm = Arm(surface=0.0)
    monkeypatch.setattr(script.time, "sleep", lambda s: None)
    out = script.run(arm.ctx(), end={"x": 0.40, "y": 0.10, "z": 0.09}, settle_period=0.01)
    assert out["route"] == "free"


def test_every_route_is_declared_in_the_example(skills_registry):
    import json
    from pathlib import Path

    bundle = skills_registry.get(BUNDLE)
    root = Path(bundle.canonical_scripts["detect_contact"].module.__file__).parents[1]
    example = json.loads((root / "examples" / "canonical_subgraph.json").read_text())
    declared = {*example["exit"]["success_values"], example["on_error"]}
    assert declared == {"contact", "free", "no_signal", "blocked"}
