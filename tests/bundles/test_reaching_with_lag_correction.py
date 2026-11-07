"""reaching-with-lag-correction: FakeContext tests of the read-back, the aim-past move and the routing."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "reaching-with-lag-correction"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
TARGET = {"position": {"x": 0.45, "y": 0.05, "z": 0.06}, "rotation": DOWN}
FAST = {"settle_interval": 0.0, "settle_timeout": 0.0}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["reach_with_correction"].module


def _q(r):
    return np.array([r["w"], r["x"], r["y"], r["z"]], dtype=float)


def _about_z(q, angle):
    turn = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
    w1, x1, y1, z1 = turn
    w2, x2, y2, z2 = q
    out = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    return {"w": out[0], "x": out[1], "y": out[2], "z": out[3]}


class Arm:
    """Lands at the command plus ``bias(command)``; ``refuse`` (a predicate on the
    command) refuses before moving, ``raise_short`` raises after landing, as a real
    station does when the tool stops more than its bound short."""

    def __init__(self, bias=None, refuse=None, raise_short=False, routed=False, turn=0.0):
        self.p = np.array([0.30, 0.0, 0.25])
        self.rot = DOWN
        self.bias = bias or (lambda p: np.zeros(3))
        self.refuse = refuse or (lambda p: False)
        self.raise_short = raise_short
        self.routed = routed
        self.turn = turn
        self.commands = []

    def _go(self, pose, **kwargs):
        p = np.array([pose["position"][k] for k in "xyz"], dtype=float)
        self.commands.append((p.copy(), pose["rotation"], kwargs))
        if self.refuse(p):
            if self.routed:
                return {"reached": False, "outcome": "unreachable", "reason": "no IK"}
            raise ToolError("robot.go_to_pose", "no IK solution")
        self.p = p + self.bias(p)
        self.rot = _about_z(_q(pose["rotation"]), self.turn) if self.turn else pose["rotation"]
        off = float(np.linalg.norm(self.p - p))
        if self.raise_short and off > 0.012:
            raise ToolError(
                "robot.go_to_pose", f"the tool did not reach its target ({off * 1e3:.0f} mm)"
            )
        return {"reached": True, "outcome": "reached"}

    def ctx(self):
        return FakeContext(
            {
                "robot.get_ee_pose": lambda **kw: {
                    "pose": {
                        "position": dict(zip("xyz", map(float, self.p), strict=True)),
                        "rotation": self.rot,
                    }
                },
                "robot.go_to_pose": self._go,
                "robot.go_to_pose_cartesian": self._go,
            }
        )


def _sag(p):
    """The real left YAM's sag model (twin arm_sag): 3.3 mm, +3.1 mm per cm past 0.455 m."""
    r = float(np.hypot(p[0], p[1]))
    return np.array([0.0, 0.0, (3.3 + max(0.0, r - 0.455) * 100 * 3.1) / 1000])


def test_on_target_first_time_sends_one_move(script):
    arm = Arm()
    out = script.run(arm.ctx(), target_pose=TARGET, **FAST)
    assert out["route"] == "reached", out["detail"]
    assert not out["corrected"]
    assert len(arm.commands) == 1
    assert out["residual"] == pytest.approx(0.0, abs=1e-9)
    assert set(arm.commands[0][2]) == set()  # no connector knobs unless asked


def test_sag_is_aimed_past_and_removed(script):
    arm = Arm(bias=_sag)
    target = {"position": {"x": 0.50, "y": 0.0, "z": 0.06}, "rotation": DOWN}
    out = script.run(arm.ctx(), target_pose=target, **FAST)
    assert out["route"] == "reached", out["detail"]
    assert out["corrected"]
    assert out["first_residual"] == pytest.approx(0.01725)  # 3.3 + 4.5 cm x 3.1 mm
    assert out["residual"] == pytest.approx(0.0, abs=1e-9)
    second = arm.commands[1][0]
    np.testing.assert_allclose(second, [0.50, 0.0, 0.06 - out["first_residual"]], atol=1e-9)
    assert out["correction"]["z"] == pytest.approx(-out["first_residual"])


def test_small_sag_near_the_table_is_corrected_too(script):
    arm = Arm(bias=lambda p: np.array([0.0, 0.0, 0.0025]))
    out = script.run(arm.ctx(), target_pose=TARGET, **FAST)
    assert out["route"] == "reached"
    assert out["corrected"]
    assert out["first_residual"] == pytest.approx(0.0025)
    assert out["residual"] == pytest.approx(0.0, abs=1e-9)


def test_under_the_threshold_no_second_move(script):
    arm = Arm(bias=lambda p: np.array([0.0, 0.0, 0.0015]))
    out = script.run(arm.ctx(), target_pose=TARGET, **FAST)
    assert out["route"] == "reached"
    assert not out["corrected"]
    assert len(arm.commands) == 1


def test_a_station_that_raises_on_a_short_stop_is_read_back_and_corrected(script):
    arm = Arm(bias=lambda p: np.array([0.0, 0.0, 0.018]), raise_short=True)
    out = script.run(arm.ctx(), target_pose=TARGET, **FAST)
    assert out["route"] == "reached", out["detail"]
    assert out["corrected"]
    assert out["first_residual"] == pytest.approx(0.018)
    assert "did not reach" in out["detail"]


def test_a_shortfall_over_the_cap_is_retried_as_is_and_routed(script):
    arm = Arm(bias=lambda p: np.array([0.0, 0.0, 0.04]))
    out = script.run(arm.ctx(), target_pose=TARGET, **FAST)
    assert out["route"] == "not_reached"
    assert not out["corrected"]
    assert len(arm.commands) == 2
    np.testing.assert_allclose(arm.commands[1][0], arm.commands[0][0])
    assert out["residual"] == pytest.approx(0.04)
    assert "retried as is" in out["detail"]


def test_a_correction_that_does_not_land_is_not_reached(script):
    # The bias grows with every move (e.g. a hand pressed on an obstacle).
    pushes = iter([0.010, 0.020])
    arm = Arm(bias=lambda p: np.array([0.0, 0.0, next(pushes)]))
    out = script.run(arm.ctx(), target_pose=TARGET, **FAST)
    assert out["route"] == "not_reached"
    assert out["corrected"]
    assert out["residual"] == pytest.approx(0.010)
    assert "ended 10.0 mm" in out["detail"]


@pytest.mark.parametrize("routed", [False, True])
def test_a_refused_move_is_not_aimed_past(script, routed):
    near = {"position": {"x": 0.31, "y": 0.0, "z": 0.25}, "rotation": DOWN}
    arm = Arm(refuse=lambda p: True, routed=routed)
    out = script.run(arm.ctx(), target_pose=near, **FAST)
    assert out["route"] == "not_reached"
    assert len(arm.commands) == 1
    assert "refused" in out["detail"]
    assert out["residual"] == pytest.approx(0.01)


def test_a_refused_correction_keeps_the_first_landing(script):
    # The aim-past point lies under a floor the station guards: refused, the
    # hand stays 2.5 mm over the target, which the exit bound still accepts.
    arm = Arm(bias=lambda p: np.array([0.0, 0.0, 0.0025]), refuse=lambda p: p[2] < 0.0599)
    out = script.run(arm.ctx(), target_pose=TARGET, **FAST)
    assert out["route"] == "reached"
    assert out["residual"] == pytest.approx(0.0025)
    assert "second move" in out["detail"]


def test_a_wrist_lag_is_aimed_past(script):
    arm = Arm(turn=-0.04)
    out = script.run(arm.ctx(), target_pose=TARGET, **FAST)
    assert out["first_residual_rotation"] == pytest.approx(0.04, abs=1e-6)
    assert out["corrected"]
    # The second command turns 0.04 rad past the goal about z; the arm's lag brings it back.
    assert out["residual_rotation"] == pytest.approx(0.0, abs=1e-6)
    assert out["route"] == "reached"


def test_a_free_roll_is_ignored_without_check_roll(script):
    # An IK that leaves the roll about the approach axis free lands 1.5 rad off in roll.
    arm = Arm(turn=1.5)
    out = script.run(arm.ctx(), target_pose=TARGET, check_roll=False, **FAST)
    assert out["route"] == "reached", out["detail"]
    assert not out["corrected"]
    assert out["residual_rotation"] == pytest.approx(0.0, abs=1e-6)
    out = script.run(Arm(turn=1.5).ctx(), target_pose=TARGET, **FAST)
    assert out["route"] == "not_reached"
    assert out["residual_rotation"] == pytest.approx(1.5, abs=1e-6)


def test_a_tilt_is_aimed_past_without_check_roll(script):
    class Tilting(Arm):
        """Lands tilted 0.05 rad about world x, plus a free roll."""

        def _go(self, pose, **kwargs):
            reply = super()._go(pose, **kwargs)
            tilt = np.array([math.cos(0.025), math.sin(0.025), 0.0, 0.0])
            q = _q(self.rot)
            w1, x1, y1, z1 = tilt
            w2, x2, y2, z2 = q
            self.rot = {
                "w": w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                "x": w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                "y": w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                "z": w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            }
            return reply

    arm = Tilting()
    out = script.run(arm.ctx(), target_pose=TARGET, check_roll=False, **FAST)
    assert out["first_residual_rotation"] == pytest.approx(0.05, abs=1e-6)
    assert out["corrected"]
    assert out["residual_rotation"] == pytest.approx(0.0, abs=1e-6)
    assert out["route"] == "reached"


def test_tolerance_and_arm_id_and_cartesian_are_passed_only_when_set(script):
    arm = Arm()
    ctx = arm.ctx()
    script.run(ctx, target_pose=TARGET, servo_tolerance=0.001, arm_id=1, **FAST)
    assert arm.commands[0][2] == {"tolerance": 0.001, "arm_id": 1}
    assert ctx.calls_to("robot.get_ee_pose")[0].kwargs == {"arm_id": 1}
    arm = Arm()
    ctx = arm.ctx()
    script.run(ctx, target_pose=TARGET, cartesian=True, servo_tolerance=0.001, **FAST)
    assert [c.tool for c in ctx.calls if c.tool.startswith("robot.go")] == [
        "robot.go_to_pose_cartesian"
    ]
    assert arm.commands[0][2] == {}


def test_settle_waits_for_a_creeping_arm(script, monkeypatch):
    # Each reading creeps 1 mm closer, three times, then stops.
    arm = Arm(bias=lambda p: np.array([0.0, 0.0, 0.004]))
    creep = iter(range(3))

    def reading(**kw):
        if len(arm.commands) == 1 and next(creep, None) is not None:
            arm.p = arm.p - np.array([0.0, 0.0, 0.001])
        return {
            "pose": {
                "position": dict(zip("xyz", map(float, arm.p), strict=True)),
                "rotation": arm.rot,
            }
        }

    ctx = FakeContext({"robot.get_ee_pose": reading, "robot.go_to_pose": arm._go})
    monkeypatch.setattr(script.time, "sleep", lambda s: None)
    out = script.run(ctx, target_pose=TARGET, settle_interval=0.0, settle_timeout=5.0)
    # The creep closed 3 of the 4 mm before the read-back: no second move needed.
    assert out["first_residual"] == pytest.approx(0.001, abs=1e-9)
    assert not out["corrected"]
    assert out["route"] == "reached"


def test_example_declares_every_route(script):
    import json
    from pathlib import Path

    example = Path(script.__file__).resolve().parents[1] / "examples" / "canonical_subgraph.json"
    graph = json.loads(example.read_text())
    declared = set(graph["exit"]["success_values"]) | {graph["on_error"]}
    assert {"reached", "not_reached"} <= declared
