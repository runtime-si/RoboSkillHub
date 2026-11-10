"""searching-insertion-with-yaw-oscillation: FakeContext tests of the press-and-turn search."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "searching-insertion-with-yaw-oscillation"
START = (0.40, 0.10, 0.05)
FAST = {"settle_period": 0.0}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["yaw_search"].module


def _wrap(a, period):
    return (a + period / 2) % period - period / 2


class Arm:
    """A top-down hand holding a square part on its socket. The socket's surface is
    at ``surface``; when the hand's yaw is within ``fit`` deg of ``key`` (mod 90) the
    part drops ``depth`` into it. The hand lands ``sag`` above its command unless the
    surface holds it. The wrist's yaw is -``gain`` x the last joint (deg); a turn
    past ``jam_at`` deg is held there. ``joints=False`` publishes no joint reading."""

    def __init__(
        self,
        surface=START[2],
        key=20.0,
        fit=5.5,
        depth=0.010,
        sag=0.0,
        gain=1.0,
        jam_at=None,
        joints=True,
        refuse_joint=None,
        refuse_pose=None,
        start_z=None,
        walls=True,
        wall_gap=0.0015,
    ):
        z = START[2] if start_z is None else start_z
        self.p = np.array([START[0], START[1], z])
        self.cmd = self.p.copy()
        self.surface, self.key, self.fit, self.depth = surface, key, fit, depth
        self.sag, self.gain, self.jam_at = sag, gain, jam_at
        self.has_joints = joints
        self.q = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.3])
        self.refuse_joint = refuse_joint or (lambda q: None)
        self.refuse_pose = refuse_pose or (lambda p: None)
        self.poses, self.turns = [], []
        self.walls, self.wall_gap, self.in_x = walls, wall_gap, None
        self.rim = surface if surface is not None else 0.0

    @property
    def yaw(self):
        return -self.gain * math.degrees(self.q[-1] - 0.3)

    def _barrier(self):
        if abs(_wrap(self.yaw - self.key, 90.0)) <= self.fit:
            return self.surface - self.depth
        return self.surface

    def _land(self):
        z = self.cmd[2] + self.sag
        if self.surface is not None:
            z = max(z, self._barrier())
        x = self.cmd[0]
        # In the socket (under its rim) the walls allow wall_gap either way.
        if self.surface is not None and self.walls and z < self.rim - 0.001:
            if self.in_x is None:
                self.in_x = x
            x = min(max(x, self.in_x - self.wall_gap), self.in_x + self.wall_gap)
        self.p = np.array([x, self.cmd[1], z])

    def go(self, pose, **kwargs):
        target = np.array([pose["position"][k] for k in "xyz"], dtype=float)
        self.poses.append((target.copy(), kwargs))
        why = self.refuse_pose(target)
        if why:
            return {"reached": False, "outcome": "unreachable", "reason": why}
        self.cmd = target
        self._land()
        return {"reached": True, "outcome": "reached"}

    def joints_to(self, joint_config, **kwargs):
        goal = np.array(joint_config["positions"], dtype=float)
        self.turns.append(goal.copy())
        why = self.refuse_joint(goal)
        if why:
            raise ToolError("robot.move_to_joints", why)
        self.q = goal
        if self.jam_at is not None and abs(self.yaw) > abs(self.jam_at):
            self.q[-1] = 0.3 - math.radians(math.copysign(abs(self.jam_at), self.yaw)) / self.gain
        # Moving the joints to the measured ones takes the press off.
        self.cmd = self.p.copy() - np.array([0, 0, self.sag])
        self._land()
        return {"success": True}

    def pose(self, **kw):
        a = math.radians(self.yaw)
        rot = {"w": 0.0, "x": math.cos(a / 2), "y": math.sin(a / 2), "z": 0.0}
        return {
            "pose": {"position": dict(zip("xyz", map(float, self.p), strict=True)), "rotation": rot}
        }

    def observation(self, **kw):
        js = {"positions": self.q.tolist()} if self.has_joints else {}
        return {"cameras": [], "arms": [{"joint_state": js}]}

    def ctx(self, gripper=None):
        extra = {}
        if gripper is not None:
            extra = {
                "robot.get_gripper": gripper,
                "robot.describe_gripper": {
                    "span_m": 0.094,
                    "width_fit": {"at_open_m": 0.094, "at_closed_m": 0.0},
                },
            }
        return FakeContext(
            {
                **extra,
                "robot.get_ee_pose": self.pose,
                "robot.go_to_pose": self.go,
                "robot.move_to_joints": self.joints_to,
                "robot.get_observation": self.observation,
            }
        )


def test_yaw_order_alternates_nearest_first(script):
    assert script.yaw_order(12.0, 6.0) == [0.0, 6.0, -6.0, 12.0, -12.0]
    assert script.yaw_order(0.0, 6.0) == [0.0]


def test_drops_in_at_the_matching_yaw(script):
    arm = Arm(key=20.0)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "dropped_in"
    assert out["yaw_found_deg"] == pytest.approx(18.0, abs=0.3)
    assert out["tries"] == 6  # 0, +6, -6, +12, -12, +18
    assert out["sink"] >= 0.002
    # Unloaded where it sank.
    assert arm.poses[-1][0][2] == pytest.approx(arm.p[2])


def test_negative_key_is_found_on_the_other_side(script):
    arm = Arm(key=-13.0)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "dropped_in"
    assert out["yaw_found_deg"] == pytest.approx(-12.0, abs=0.3)


def test_already_aligned_slides_in_while_feeling_and_says_so(script):
    arm = Arm(key=2.0)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "blocked"
    assert "already in" in out["detail"]
    assert arm.turns == []


def test_aligned_after_the_first_press_drops_without_turning(script):
    arm = Arm(key=2.0, fit=5.5)
    seen = {"pressed": False}
    orig = arm._barrier

    def barrier():
        # The rim holds while feeling (the part is 1 deg out until pressed).
        return orig() if seen["pressed"] else arm.surface

    def go(pose, **kw):
        if pose["position"]["z"] < START[2] - 0.0022:
            seen["pressed"] = True
        return Arm.go(arm, pose, **kw)

    arm._barrier = barrier
    arm.go = go
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "dropped_in"
    assert out["tries"] == 1
    assert arm.turns == []


def test_four_fold_symmetry_wraps(script):
    arm = Arm(key=84.0)  # = -6 mod 90
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "dropped_in"
    assert out["yaw_found_deg"] == pytest.approx(-6.0, abs=0.3)


def test_no_drop_turns_back(script):
    arm = Arm(key=45.0, fit=1.0)
    out = script.run(arm.ctx(), yaw_range_deg=30.0, **FAST)
    assert out["route"] == "no_drop"
    assert out["tries"] == 11
    assert abs(arm.yaw) < 0.5
    assert out["sink"] == pytest.approx(0.0, abs=1e-9)


def test_no_drop_kept_turned_when_asked(script):
    arm = Arm(key=45.0, fit=1.0)
    out = script.run(arm.ctx(), yaw_range_deg=12.0, return_on_fail=False, **FAST)
    assert out["route"] == "no_drop"
    assert arm.yaw == pytest.approx(-12.0, abs=0.3)


def test_shallow_slack_is_not_a_drop(script):
    arm = Arm(key=20.0, depth=0.0015)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "no_drop"


def test_blocked_turn_is_jammed(script):
    arm = Arm(key=40.0, jam_at=8.0)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "jammed"
    assert "short" in out["detail"]
    assert out["tries"] == 4  # +12 is held at +8


def test_gain_is_learnt_for_a_wrist_that_turns_the_other_way(script):
    arm = Arm(key=20.0, gain=-1.0)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "dropped_in"
    assert out["yaw_found_deg"] == pytest.approx(18.0, abs=0.3)


def test_descent_finds_the_socket_first(script):
    arm = Arm(key=20.0, start_z=START[2] + 0.005)
    out = script.run(arm.ctx(), max_descent=0.01, **FAST)
    assert out["route"] == "dropped_in"
    assert out["rest_z"] == pytest.approx(START[2], abs=1e-9)
    # Felt at the command 1.5 mm under the surface, pressed 1 mm past it.
    assert out["sink"] == pytest.approx(0.0025, abs=1e-9)


def test_descent_without_a_socket_is_blocked(script):
    arm = Arm(surface=None, start_z=START[2] + 0.005)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "blocked"
    assert "no socket felt" in out["detail"]
    assert arm.turns == []


def test_sagging_arm_rests_and_still_finds_the_drop(script):
    arm = Arm(key=-6.0, sag=0.0054, start_z=START[2])
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "dropped_in"


def test_no_room_on_the_last_joint_is_blocked(script):
    arm = Arm(key=40.0, fit=1.0)
    out = script.run(arm.ctx(), roll_limits=[0.1, 0.55], **FAST)
    assert out["route"] == "blocked"
    assert "no room" in out["detail"]


def test_refused_wrist_turn_is_blocked(script):
    arm = Arm(key=40.0, refuse_joint=lambda q: "joint limit")
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "blocked"
    assert "wrist turn" in out["detail"]


def test_refused_press_is_blocked(script):
    arm = Arm(refuse_pose=lambda p: "no" if p[2] < START[2] - 0.0022 else None)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "blocked"
    assert "press" in out["detail"]


def test_connector_without_joints_turns_by_rotation(script):
    arm = Arm(key=20.0, joints=False)
    out = script.run(arm.ctx(), **FAST)
    # The fake ignores rotation commands (a free wrist roll): nothing turns, no drop.
    assert out["route"] == "jammed"
    assert arm.turns == []


def test_knobs_passed_only_when_set(script):
    arm = Arm(key=2.0)
    script.run(arm.ctx(), arm_id=1, step_max_steps=30, **FAST)
    assert arm.poses[0][1] == {"tolerance": 0.0002, "max_steps": 30, "arm_id": 1}
    arm = Arm(key=2.0)
    script.run(arm.ctx(), step_tolerance=0.0, **FAST)
    assert arm.poses[0][1] == {}


def test_bad_arguments_raise(script):
    with pytest.raises(ValueError):
        script.run(Arm().ctx(), press_depth=0.0)
    with pytest.raises(ValueError):
        script.run(Arm().ctx(), yaw_step_deg=0.0)


def test_a_part_creeping_through_the_jaws_is_not_a_drop(script):
    # The rim holds, but every press lets the part slide 0.6 mm up the jaws: the
    # pressed hand creeps down and follows its command, never suddenly.
    arm = Arm(key=45.0, fit=1.0)
    state = {"creep": 0.0}
    orig_land = arm._land

    def land():
        orig_land()
        if arm.cmd[2] < START[2] - 0.0022:
            state["creep"] += 0.0006
            arm.p[2] = max(arm.cmd[2], arm.p[2] - state["creep"])

    arm._land = land
    out = script.run(arm.ctx(), yaw_range_deg=24.0, **FAST)
    assert out["route"] == "no_drop", out["detail"]
    assert "sliding in the jaws" in out["detail"]


def test_a_part_lost_from_the_jaws_is_blocked_not_a_drop(script):
    arm = Arm(key=40.0, fit=1.0)
    presses = {"n": 0}

    def gripper(**kw):
        presses["n"] += 1
        return {"position": 0.32 if presses["n"] < 3 else 0.0}

    out = script.run(arm.ctx(gripper=gripper), **FAST)
    assert out["route"] == "blocked"
    assert "no longer in the hand" in out["detail"]
    assert out["tries"] == 3


def test_holding_part_reads_as_held(script):
    arm = Arm(key=20.0)
    out = script.run(arm.ctx(gripper=lambda **kw: {"position": 0.32}), **FAST)
    assert out["route"] == "dropped_in"


def test_a_turn_stopped_with_the_hand_sunk_is_a_drop(script):
    class SweepArm(Arm):
        """The part drops in as the wrist sweeps through +20 deg on the way to a
        wider yaw; the socket's walls then hold the turn at +20.5 deg."""

        def joints_to(self, joint_config, **kwargs):
            before = self.yaw
            reply = Arm.joints_to(self, joint_config, **kwargs)
            lo, hi = sorted((before, self.yaw))
            if lo <= self.key <= hi and self.yaw != before:
                self.surface -= 0.010
                self._land()
                self.p[2] = self.surface + 0.0005
                self.jam_at = self.key + 0.5
                if abs(self.yaw) > abs(self.jam_at):
                    self.q[-1] = 0.3 - math.radians(self.jam_at) / self.gain
            return reply

    arm = SweepArm(key=20.0, fit=0.5)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "dropped_in", out["detail"]
    assert "dropped in on the way" in out["detail"]


def test_a_drop_is_confirmed_by_a_wall_sideways(script):
    arm = Arm(key=20.0)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "dropped_in"
    assert out["confirmed"]
    assert out["confirm_travel"] == pytest.approx(0.0015, abs=0.0006)
    # The hand went back over the drop after the probe.
    assert arm.poses[-1][0][0] == pytest.approx(START[0], abs=1e-9)


def test_unloading_a_sagging_arm_keeps_the_part_down(script):
    arm = Arm(key=-6.0, sag=0.0054, start_z=START[2])
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "dropped_in", out["detail"]
    assert out["confirmed"]
    # Still under the rim after the unload and the probe.
    assert arm.p[2] < START[2] - 0.002


def test_a_drop_without_walls_is_unconfirmed(script):
    # The hand sank and followed its command, but nothing stops it sideways: the
    # part slid in the jaws on the rim (or walked off it).
    arm = Arm(key=20.0, walls=False)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "unconfirmed", out["detail"]
    assert not out["confirmed"]
    assert out["confirm_travel"] == -1.0
    assert "without meeting a wall" in out["detail"]


def test_confirmation_can_be_skipped(script):
    arm = Arm(key=20.0, walls=False)
    out = script.run(arm.ctx(), confirm=False, **FAST)
    assert out["route"] == "dropped_in"
    assert not out["confirmed"]


def test_a_part_caught_on_the_socket_edge_fails_the_seat_check(script):
    # The part goes 2.5 mm down (enough to read as a drop, and its edge meets a
    # wall sideways) but no further: caught by its lip, not in.
    arm = Arm(key=20.0, depth=0.0025)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "unconfirmed", out["detail"]
    assert "caught on the socket's edge" in out["detail"]
    assert out["seat_follow"] == pytest.approx(0.0, abs=1e-6)
    assert not out["confirmed"]


def test_a_part_in_its_socket_follows_the_seat_check(script):
    arm = Arm(key=20.0)
    out = script.run(arm.ctx(), **FAST)
    assert out["route"] == "dropped_in"
    assert out["seat_follow"] == pytest.approx(0.003, abs=1e-6)


def test_the_seat_check_can_be_skipped(script):
    arm = Arm(key=20.0, depth=0.0025)
    out = script.run(arm.ctx(), seat_check=0.0, **FAST)
    assert out["route"] == "dropped_in"
    assert out["seat_follow"] == -1.0
