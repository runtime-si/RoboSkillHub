"""pouring-by-wrist-roll: FakeContext tests of the pour pose, the stepped roll, its checks and the routing."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "pouring-by-wrist-roll"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}  # tool z straight down: the roll spins
SIDE = {
    "w": math.sqrt(0.5),
    "x": 0.0,
    "y": math.sqrt(0.5),
    "z": 0.0,
}  # tool z along +x: the roll tips
# A 100 x 100 x 50 mm receiver whose top is at z = 0.06.
BOX = {
    "center": {"x": 0.30, "y": 0.05, "z": 0.035},
    "extent": {"x": 0.05, "y": 0.05, "z": 0.025},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}
POUR = np.array([0.30, 0.05, 0.06 + 0.10])
Q0 = np.array([1.5, 0.6, 0.4, -0.1, 0.0, -0.5])
N = math.ceil(math.radians(120.0) / (math.pi / 40))  # 27 steps each way


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["pour_wrist_roll"].module


@pytest.fixture(autouse=True)
def no_waiting(script, monkeypatch):
    sleeps = []
    monkeypatch.setattr(script.time, "sleep", sleeps.append)
    return sleeps


def _wxyz(q):
    return np.array([q[k] for k in ("w", "x", "y", "z")], dtype=float)


def _about_z(base, new):
    """The rotation about the tool's z taking ``base`` to ``new`` [rad]."""
    w, x, y, z = _wxyz(base)
    a = np.array([w, -x, -y, -z])
    b = _wxyz(new)
    rel = np.array(
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    )
    return 2.0 * math.atan2(rel[1], rel[0])


class Arm:
    """Moves land where commanded unless told otherwise.

    ``joint_cap``: the last joint stops there (an obstacle). ``drift``: the tool
    centre moves this far per radian of roll. ``joints_refused``: the call
    indices of robot.move_to_joints that raise (``"all"``: every one).
    ``pose_sign``: the last joint's turn per radian of tool roll about z.
    """

    def __init__(
        self,
        start=(0.30, 0.25, 0.12),
        q=Q0,
        rot=SIDE,
        joint_cap=None,
        drift=0.0,
        joints_refused=(),
        pose_sign=1.0,
        pose_moves_joint=True,
        sag=0.0,
        refuse_pose=None,
        land_off=0.0,
        grip=lambda q: 0.33,
    ):
        self.p = np.array(start, dtype=float)
        self.q = np.array(q, dtype=float)
        self.rot = rot
        self.joint_cap = joint_cap
        self.drift = drift
        self.joints_refused = joints_refused
        self.pose_sign = pose_sign
        self.pose_moves_joint = pose_moves_joint
        self.sag = sag
        self.refuse_pose = refuse_pose or (lambda tool, pose: None)
        self.land_off = land_off
        self.grip = grip
        self.moves = []
        self.joint_moves = []
        self.base = None

    def _hand(self):
        if self.base is None:
            return self.p
        return self.p + np.array([self.drift * abs(self.q[-1] - self.base[1][-1]), 0.0, 0.0])

    def _pose_move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy(), pose["rotation"]))
            why = self.refuse_pose(tool, pose)
            if why == "raise":
                raise ToolError(tool, "no IK solution")
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            if np.linalg.norm(target - self.p) > 1.0e-6 or self.base is None:
                self.p = target + np.array([self.land_off, 0.0, 0.0])
                self.rot = pose["rotation"]
                if np.linalg.norm(target - POUR) < 1.0e-9:
                    self.base = (pose["rotation"], self.q.copy())
            elif self.pose_moves_joint:
                self.q[-1] = self.base[1][-1] + self.pose_sign * _about_z(
                    self.base[0], pose["rotation"]
                )
                self.rot = pose["rotation"]
            return {"reached": True, "outcome": "reached"}

        return go

    def _joints(self, joint_config, **kwargs):
        index = len(self.joint_moves)
        self.joint_moves.append((np.array(joint_config["positions"], dtype=float), kwargs))
        if self.joints_refused == "all" or index in self.joints_refused:
            raise ToolError("robot.move_to_joints", "the station bridge takes poses only")
        target = np.array(joint_config["positions"], dtype=float)
        if self.joint_cap is not None:
            target[-1] = min(target[-1], self.joint_cap)
        self.q = target
        if self.sag and abs(self.q[-1] - Q0[-1]) > 1.0:
            self.sag_applied = True
        if getattr(self, "sag_applied", False):
            self.q[1] = target[1] + self.sag
        return None

    def ctx(self):
        return FakeContext(
            {
                "robot.get_ee_pose": lambda **kw: {
                    "pose": {
                        "position": dict(zip("xyz", map(float, self._hand()), strict=True)),
                        "rotation": self.rot,
                    }
                },
                "robot.get_observation": lambda **kw: {
                    "cameras": [],
                    "arms": [
                        {
                            "joint_state": {"positions": self.q.copy()},
                            "gripper_fraction": self.grip(self.q),
                        }
                    ],
                },
                "robot.go_to_pose": self._pose_move("robot.go_to_pose"),
                "robot.go_to_pose_cartesian": self._pose_move("robot.go_to_pose_cartesian"),
                "robot.move_to_joints": self._joints,
            }
        )


def test_stands_over_the_box_rolls_only_the_last_joint_and_rolls_back(script, no_waiting):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(ctx, target_obb=BOX)
    assert out["route"] == "poured", out["detail"]
    assert out["rolled_with"] == "joints"
    # Below the pour height: straight up first, then across, the hand's own rotation.
    assert [m[0] for m in arm.moves] == ["robot.go_to_pose_cartesian", "robot.go_to_pose"]
    np.testing.assert_allclose(arm.moves[0][1], [0.30, 0.25, POUR[2]], atol=1e-9)
    np.testing.assert_allclose(arm.moves[1][1], POUR, atol=1e-9)
    assert all(m[2] == SIDE for m in arm.moves)
    # 27 steps out, 27 back; every other joint held where it stood.
    targets = np.array([t for t, _ in arm.joint_moves])
    assert len(targets) == 2 * N
    np.testing.assert_allclose(targets[:, :5], np.tile(Q0[:5], (2 * N, 1)))
    rolls = np.concatenate([[Q0[-1]], targets[:, -1]])
    assert np.max(np.abs(np.diff(rolls))) <= math.pi / 40 + 1e-9
    # Joint 6 at -0.5 has more room toward +: it rolls up by 120 deg and back.
    assert targets[N - 1, -1] == pytest.approx(Q0[-1] + math.radians(120.0))
    assert targets[-1, -1] == pytest.approx(Q0[-1])
    assert out["roll_reached"] == pytest.approx(math.radians(120.0))
    assert out["roll_trajectory"][0] == pytest.approx(Q0[-1])
    assert out["roll_trajectory"][N] == pytest.approx(Q0[-1] + math.radians(120.0))
    assert len(out["roll_trajectory"]) == 2 * N + 2  # the start, out, the hold, back
    assert out["grip_trajectory"] == [0.33] * (2 * N + 2)
    assert out["hand_drift"] == pytest.approx(0.0)
    assert out["final_joint_error"] == pytest.approx(0.0)
    assert out["roll_axis_elevation_deg"] == pytest.approx(0.0, abs=1e-6)
    assert out["detail"] == ""
    # Connector knobs and arm 0 are not passed unless set.
    assert all(set(kw) == set() for _, kw in arm.joint_moves)


def test_each_step_takes_at_least_its_share_at_the_speed_bound(script, no_waiting):
    script.run(Arm().ctx(), target_obb=BOX, hold_s=0.5)
    period = math.radians(120.0) / N / 0.5
    steps = [s for s in no_waiting if s != 0.5]
    assert len(steps) == 2 * N
    # Instant fake moves: every step waits (almost) its whole period.
    assert all(s == pytest.approx(period, abs=0.02) for s in steps)
    assert 0.5 in no_waiting  # the hold


def test_a_given_rotation_offset_and_direction(script):
    arm = Arm(start=(0.30, 0.25, 0.30), rot=DOWN, q=[1.5, 0.6, 0.4, -0.1, 0.0, 0.5])
    out = script.run(
        arm.ctx(),
        target_obb=BOX,
        rotation=SIDE,
        offset={"x": -0.02, "y": 0.01},
        roll_direction=-1,
        pour_angle_deg=90.0,
        tolerance=0.02,
        max_steps=300,
    )
    assert out["route"] == "poured"
    # Above the pour height already: no rise, one move, the given rotation.
    assert [m[0] for m in arm.moves] == ["robot.go_to_pose"]
    np.testing.assert_allclose(arm.moves[0][1], [0.28, 0.06, POUR[2]], atol=1e-9)
    assert arm.moves[0][2] == SIDE
    assert arm.joint_moves[19][0][-1] == pytest.approx(0.5 - math.pi / 2)  # 20 steps of pi/40
    assert arm.joint_moves[0][1] == {"tolerance": 0.02, "max_steps": 300}


def test_a_top_down_hand_pours_but_says_the_roll_only_spun(script):
    out = script.run(Arm(rot=DOWN).ctx(), target_obb=BOX)
    assert out["route"] == "poured"
    assert out["roll_axis_elevation_deg"] == pytest.approx(90.0)
    assert "spun" in out["detail"]


def test_a_roll_that_does_not_fit_is_joint_limit(script):
    # A YAM joint 6 at 0 leaves exactly 120 deg each way, which the margin refuses.
    arm = Arm(q=[1.5, 0.6, 0.4, -0.1, 0.0, 0.0])
    out = script.run(arm.ctx(), target_obb=BOX)
    assert out["route"] == "joint_limit"
    assert "joint 6" in out["detail"]
    assert arm.joint_moves == []

    arm = Arm()
    out = script.run(arm.ctx(), target_obb=BOX, roll_direction=-1)
    assert out["route"] == "joint_limit"  # -0.5 leaves 1.54 rad toward -
    assert arm.joint_moves == []

    arm = Arm(q=[1.5, 0.6, 0.4, -0.1, 0.0, 0.0])
    out = script.run(arm.ctx(), target_obb=BOX, roll_limits=[-2.9, 2.9])
    assert out["route"] == "poured"  # a wider last joint (a Franka's) fits


def test_unreachable_pour_pose(script):
    arm = Arm(refuse_pose=lambda tool, pose: "raise" if tool == "robot.go_to_pose" else None)
    out = script.run(arm.ctx(), target_obb=BOX)
    assert out["route"] == "unreachable"
    assert "no IK solution" in out["detail"]
    assert arm.joint_moves == []

    arm = Arm(refuse_pose=lambda tool, pose: "joint limit" if tool == "robot.go_to_pose" else None)
    out = script.run(arm.ctx(), target_obb=BOX)
    assert out["route"] == "unreachable" and "unreachable: joint limit" in out["detail"]

    arm = Arm(land_off=0.03)  # arrived "reached", 30 mm off
    out = script.run(arm.ctx(), target_obb=BOX)
    assert out["route"] == "unreachable" and "30 mm" in out["detail"]


def test_a_joint_that_stops_following_is_refused_and_rolled_back(script):
    arm = Arm(joint_cap=Q0[-1] + 1.0)
    out = script.run(arm.ctx(), target_obb=BOX)
    assert out["route"] == "roll_refused"
    assert "off the step's target" in out["detail"]
    # Stopped at the first step past 1.0 + 0.075 rad and stepped back to the start.
    targets = np.array([t for t, _ in arm.joint_moves])
    assert targets[-1, -1] == pytest.approx(Q0[-1])
    assert out["final_joint_error"] == pytest.approx(0.0)
    assert out["roll_reached"] == pytest.approx(1.0)


def test_a_refused_step_is_routed_and_rolled_back(script):
    arm = Arm(joints_refused=(5,))
    out = script.run(arm.ctx(), target_obb=BOX)
    assert out["route"] == "roll_refused"
    assert out["detail"].startswith("step 6 of 27: move refused")
    assert out["rolled_with"] == "joints"
    assert arm.q[-1] == pytest.approx(Q0[-1])


def test_a_drifting_hand_is_refused(script):
    arm = Arm(drift=0.03)  # 30 mm per radian of roll
    out = script.run(arm.ctx(), target_obb=BOX)
    assert out["route"] == "roll_refused"
    assert "drifted" in out["detail"]
    assert out["hand_drift"] > 0.02


def test_a_roll_back_that_ends_off_is_refused(script):
    arm = Arm(sag=0.1)  # joint 2 sags 0.1 rad once the roll passes 1 rad, and stays
    out = script.run(arm.ctx(), target_obb=BOX)
    assert out["route"] == "roll_refused"
    assert "joint 2 ended 0.100 rad" in out["detail"]
    assert out["final_joint_error"] == pytest.approx(0.1)


@pytest.mark.parametrize("pose_sign", [1.0, -1.0])
def test_a_connector_without_joint_moves_rolls_by_pose(script, pose_sign):
    arm = Arm(joints_refused="all", pose_sign=pose_sign)
    out = script.run(arm.ctx(), target_obb=BOX)
    assert out["route"] == "poured", out["detail"]
    assert out["rolled_with"] == "pose"
    assert len(arm.joint_moves) == 1  # tried once, then by pose
    rolls = [m for m in arm.moves[2:] if m[0] == "robot.go_to_pose"]
    assert all(np.allclose(m[1], POUR) for m in rolls)
    # The sign is learned from the first step: the peak is +120 deg on the joint.
    assert max(out["roll_trajectory"]) == pytest.approx(Q0[-1] + math.radians(120.0))
    assert out["final_joint_error"] == pytest.approx(0.0, abs=1e-9)


def test_forced_joint_rolls_do_not_fall_back(script):
    arm = Arm(joints_refused="all")
    out = script.run(arm.ctx(), target_obb=BOX, roll_with="joints")
    assert out["route"] == "roll_refused"
    assert "takes poses only" in out["detail"]


def test_a_pose_roll_that_does_not_turn_the_joint_is_refused(script):
    arm = Arm(pose_moves_joint=False)
    out = script.run(arm.ctx(), target_obb=BOX, roll_with="pose")
    assert out["route"] == "roll_refused"
    assert "moved joint 6 only" in out["detail"]


def test_no_joint_reading_is_refused(script):
    arm = Arm()
    tools = dict(arm.ctx()._responses)
    tools["robot.get_observation"] = {"cameras": [], "arms": []}
    out = script.run(FakeContext(tools), target_obb=BOX)
    assert out["route"] == "roll_refused" and "no joint reading" in out["detail"]


def test_the_grip_reading_shows_a_held_solid_leaving_the_hand(script):
    # The fingers close on nothing once the roll passes 1 rad, as a cube slipping out does.
    arm = Arm(grip=lambda q: 0.33 if q[-1] - Q0[-1] < 1.0 else 0.0)
    out = script.run(arm.ctx(), target_obb=BOX)
    assert out["route"] == "poured"  # the hand's motion; the grip is reported, not judged
    rolls, grips = np.array(out["roll_trajectory"]), np.array(out["grip_trajectory"])
    assert len(rolls) == len(grips)
    assert np.all(grips[rolls - Q0[-1] < 1.0] == 0.33)
    assert np.all(grips[rolls - Q0[-1] >= 1.0] == 0.0)
