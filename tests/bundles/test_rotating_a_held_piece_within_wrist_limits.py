"""rotating-a-held-piece-within-wrist-limits: planning and execution on a fake arm whose hand yaw follows joint 6."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "rotating-a-held-piece-within-wrist-limits"
LIM = 2.0944


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["rotate_piece"].module


class Arm:
    """Hand yaw (deg) = yaw0 + gain * degrees(q6); joint 6 clamps at its limits; a piece follows the hand
    while the jaws are closed on it."""

    def __init__(
        self,
        q6_deg=0.0,
        gain=-1.0,
        yaw0=10.0,
        refuse=None,
        lose=False,
        lag=0.0,
        obstacles=(),
        table=0.0,
    ):
        self.obstacles, self.table = list(obstacles), table
        self.q = [0.0, 0.5, 0.5, 0.0, 0.3, math.radians(q6_deg)]
        self.gain, self.yaw0, self.refuse, self.lose, self.lag = (
            gain,
            yaw0,
            refuse or set(),
            lose,
            lag,
        )
        self.pos = {"x": 0.3, "y": 0.1, "z": 0.08}
        self.closed = True
        self.piece_yaw = 0.0
        self.calls = []
        self.max_abs_q6 = abs(self.q[5])

    def overhead(self):
        """A camera 0.6 m over (0.3, 0.1) looking straight down: depth of the table at ``table`` with
        ``obstacles`` ((x, y, half_size, height) boxes) and the held piece hanging under the hand."""
        f, n, cz = 600.0, 400, self.table + 0.6
        k = np.array([[f, 0, n / 2], [0, f, n / 2], [0, 0, 1.0]])
        vv, uu = np.mgrid[0:n, 0:n]
        x = 0.3 + (uu - n / 2) * 0.6 / f
        y = 0.1 - (vv - n / 2) * 0.6 / f
        h = np.zeros((n, n))
        for ox, oy, hs, oh in self.obstacles:
            h[(abs(x - ox) <= hs) & (abs(y - oy) <= hs)] = oh
        if self.closed:
            h[np.hypot(x - self.pos["x"], y - self.pos["y"]) < 0.015] = (
                self.pos["z"] - 0.013 - self.table
            )
        depth = (cz - (self.table + h)).astype(np.float32)
        # camera axes: x right = world +x, y down = world -y, z = world -z
        rot = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
        return {
            "name": "top",
            "depth": depth,
            "intrinsics": k,
            "pose": {"position": {"x": 0.3, "y": 0.1, "z": cz}, "rotation": rot},
        }

    def hand_yaw(self):
        return self.yaw0 + self.gain * math.degrees(self.q[5])

    def rotation(self):
        a = math.radians(self.hand_yaw()) / 2
        return {"w": math.cos(a), "x": 0.0, "y": 0.0, "z": math.sin(a)}

    def ctx(self):
        def move_to_joints(joint_config, **kw):
            self.calls.append("joints")
            if "joints" in self.refuse:
                return {"reached": False, "outcome": "unreachable"}
            q = list(joint_config["positions"])
            q6 = max(-LIM, min(LIM, q[5]))
            q6 = self.q[5] + (q6 - self.q[5]) * (1.0 - self.lag)
            before = self.hand_yaw()
            self.q = q[:5] + [q6]
            self.max_abs_q6 = max(self.max_abs_q6, abs(q6))
            if self.closed:
                self.piece_yaw += self.hand_yaw() - before
            return {"reached": True}

        def go(pose, **kw):
            self.calls.append(("pose", round(pose["position"]["z"], 4)))
            self.pos = dict(pose["position"])
            return {"reached": True}

        def open_gripper(**kw):
            self.calls.append("open")
            self.closed = False

        def close(**kw):
            self.calls.append("close")
            self.closed = True

        return FakeContext(
            {
                "robot.get_observation": lambda **kw: {
                    "arms": [{"joint_state": {"positions": list(self.q)}}],
                    "cameras": [self.overhead()],
                },
                "robot.get_ee_pose": lambda **kw: {
                    "position": dict(self.pos),
                    "rotation": self.rotation(),
                },
                "robot.move_to_joints": move_to_joints,
                "robot.go_to_pose": go,
                "robot.go_to_pose_cartesian": go,
                "robot.open_gripper": open_gripper,
                "robot.close_gripper": close,
                "robot.set_grip": lambda **kw: close(),
                "robot.get_gripper": lambda **kw: {"width_m": 0.0 if self.lose else 0.01},
                "robot.describe_arm": {"arm_id": 0},
                "robot.wait_steps": lambda **kw: None,
            }
        )


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def test_plan_takes_the_short_way_when_it_fits(script):
    p = script.plan_turn(0.0, 90.0, -117.0, 117.0)
    assert p == {"kind": "direct", "held": [90.0], "travel": 90.0}
    # 200 deg is -160 deg: from +50 the short way fits only going back.
    p = script.plan_turn(50.0, 200.0, -117.0, 117.0)
    assert p["kind"] == "direct" and p["held"] == [-160.0]
    # A negative gain turns the joint the other way for the same piece turn.
    assert script.plan_turn(0.0, 90.0, -117.0, 117.0, gain=-1.0)["held"] == [-90.0]


def test_plan_regrasps_with_a_half_turn_when_no_direct_turn_fits(script):
    p = script.plan_turn(0.0, 150.0, -117.0, 117.0)
    assert p["kind"] == "regrasp"
    dq1, dq2 = p["held"]
    assert abs(p["empty"]) == 180.0
    assert _wrap(dq1 + dq2 - 150.0) == pytest.approx(0.0, abs=1e-6)
    q = 0.0
    for step in (dq1, p["empty"], dq2):
        q += step
        assert -117.0 - 1e-6 <= q <= 117.0 + 1e-6


@pytest.mark.parametrize("q6", [-110.0, -60.0, 0.0, 45.0, 110.0])
@pytest.mark.parametrize("turn", [-179.0, -150.0, -90.0, 0.0, 30.0, 125.0, 180.0])
def test_every_turn_has_a_plan_from_anywhere_in_range(script, q6, turn):
    p = script.plan_turn(q6, turn, -117.0, 117.0)
    assert p is not None
    if p["kind"] == "direct":
        assert -117.0 <= q6 + p["held"][0] <= 117.0


def test_a_direct_turn_is_read_back_and_learns_the_joint_sign(script):
    arm = Arm(q6_deg=0.0, gain=-1.0)
    out = script.run(arm.ctx(), turn_deg=70.0)
    assert out["route"] == "turned", out["detail"]
    assert out["regrasps"] == 0
    assert _wrap(arm.piece_yaw - 70.0) == pytest.approx(0.0, abs=0.6)
    assert out["turned_deg"] == pytest.approx(arm.piece_yaw, abs=0.6)
    assert "open" not in arm.calls


def test_a_long_turn_regrasps_and_keeps_joint_6_in_range(script):
    # From 0 deg a 150 deg turn ends at +150 or -210: neither fits +-117.
    arm = Arm(q6_deg=0.0, gain=1.0)
    out = script.run(
        arm.ctx(), turn_deg=150.0, grasp_z=0.02, set_down_xy={"x": 0.33, "y": 0.1, "z": 0.0}
    )
    assert out["route"] == "turned_with_regrasp", out["detail"]
    assert out["regrasps"] == 1 and out["jaws_swapped"] is True
    assert _wrap(arm.piece_yaw - 150.0) == pytest.approx(0.0, abs=0.6)
    assert arm.max_abs_q6 <= LIM - 0.05 + 1e-9
    i_open, i_close = arm.calls.index("open"), arm.calls.index("close")
    assert i_open < i_close
    # Set down and taken again at the grasp height, carried at grasp height + lift between.
    assert ("pose", 0.02) in arm.calls[:i_open] and ("pose", 0.02) in arm.calls[i_open:i_close]


def test_a_known_gain_skips_the_probe(script):
    arm = Arm(q6_deg=0.0, gain=-1.0)
    out = script.run(arm.ctx(), turn_deg=40.0, gain=-1.0)
    assert out["route"] == "turned"
    assert _wrap(arm.piece_yaw - 40.0) == pytest.approx(0.0, abs=0.6)


def test_a_lagging_wrist_is_corrected(script):
    arm = Arm(q6_deg=0.0, gain=1.0, lag=0.3)
    out = script.run(arm.ctx(), turn_deg=-80.0)
    assert out["route"] == "turned"
    assert _wrap(arm.piece_yaw + 80.0) == pytest.approx(0.0, abs=0.6)


def test_plan_only_moves_nothing(script):
    arm = Arm(q6_deg=0.0)
    out = script.run(arm.ctx(), turn_deg=150.0, plan_only=True, gain=1.0)
    assert out["route"] == "planned"
    assert out["plan"]["kind"] == "regrasp" and out["jaws_swapped"] is True
    assert "joints" not in arm.calls
    out = script.run(arm.ctx(), turn_deg=150.0, plan_only=True, gain=1.0, joint6_deg=-60.0)
    assert out["plan"]["kind"] == "direct"


def test_a_regrasp_without_grasp_height_is_refused(script):
    arm = Arm(q6_deg=0.0, gain=1.0)
    out = script.run(arm.ctx(), turn_deg=150.0, gain=1.0)
    assert out["route"] == "unreachable" and "grasp_z" in out["detail"]
    assert "joints" not in arm.calls


def test_refused_moves_and_a_lost_piece_are_routed(script):
    arm = Arm(refuse={"joints"})
    assert script.run(arm.ctx(), turn_deg=30.0)["route"] == "unreachable"
    arm = Arm(q6_deg=0.0, gain=1.0, lose=True)
    out = script.run(
        arm.ctx(), turn_deg=150.0, gain=1.0, grasp_z=0.02, grip_width_m=0.01, set_down_xy=[0.3, 0.1]
    )
    assert out["route"] == "piece_lost"
    assert script.run(FakeContext(), turn_deg=30.0)["route"] == "unreachable"


def test_the_drop_spot_is_chosen_clear_of_what_stands_round_the_hand(script):
    # Placed pieces 2 mm proud right under the hand and to its west; the hand is at (0.3, 0.1).
    obstacles = [(0.3, 0.1, 0.04, 0.002), (0.24, 0.1, 0.02, 0.006)]
    arm = Arm(q6_deg=0.0, gain=1.0, obstacles=obstacles)
    out = script.run(arm.ctx(), turn_deg=150.0, grasp_z=0.02, gain=1.0)
    assert out["route"] == "turned_with_regrasp", out["detail"]
    spot = out["drop_spot"]
    radius = script.PIECE_REACH_M + script.DROP_MARGIN_M
    for ox, oy, hs, _ in obstacles:
        dx = max(abs(spot["x"] - ox) - hs, 0.0)
        dy = max(abs(spot["y"] - oy) - hs, 0.0)
        assert math.hypot(dx, dy) >= radius - 0.003
    # Nearest clear spot: not far beyond the board's edge.
    assert math.hypot(spot["x"] - 0.3, spot["y"] - 0.1) < 0.04 + radius + 0.01


def test_no_clear_spot_is_refused(script):
    # Pieces 2 mm proud every 3 cm (1 cm gaps): table everywhere between them, no clear 47 mm disc.
    field = [
        (0.3 + 0.03 * i, 0.1 + 0.03 * j, 0.01, 0.002) for i in range(-9, 10) for j in range(-9, 10)
    ]
    arm = Arm(q6_deg=0.0, gain=1.0, obstacles=field)
    out = script.run(arm.ctx(), turn_deg=150.0, grasp_z=0.02, gain=1.0)
    assert out["route"] == "unreachable" and "drop spot" in out["detail"]
    assert "open" not in arm.calls
