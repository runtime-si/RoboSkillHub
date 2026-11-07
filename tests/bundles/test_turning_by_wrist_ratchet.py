"""turning-by-wrist-ratchet: FakeContext tests of the grip, the strokes, the stall / slip / lift checks and the routing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "turning-by-wrist-ratchet"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}  # tool z straight down
FLIPPED = {"w": 0.0, "x": 0.0, "y": 1.0, "z": 0.0}
# A 60 mm lid, 20 mm tall, its top at z = 0.10.
LID = {
    "center": {"x": 0.40, "y": 0.10, "z": 0.09},
    "extent": {"x": 0.03, "y": 0.03, "z": 0.01},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}
GRIP = np.array([0.40, 0.10, 0.09])  # 10 mm under the top
PRE = GRIP + np.array([0.0, 0.0, 0.08])
Q0 = np.array([0.2, 0.6, 0.4, -0.1, 0.0, 0.0])
STROKE = 3.8
N = math.ceil(STROKE / (math.pi / 40))  # 49 steps per stroke
N_START = math.ceil(1.9 / (math.pi / 40))  # 25 steps from 0 to a stroke end


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["turn_wrist_ratchet"].module


@pytest.fixture(autouse=True)
def no_waiting(script, monkeypatch):
    sleeps = []
    monkeypatch.setattr(script.time, "sleep", sleeps.append)
    return sleeps


def _rz(q, angle):
    w, x, y, z = (q[k] for k in "wxyz")
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return {"w": w * c - z * s, "x": x * c + y * s, "y": y * c - x * s, "z": z * c + w * s}


def _about_z(base, new):
    a = np.array([base["w"], -base["x"], -base["y"], -base["z"]])
    b = np.array([new[k] for k in "wxyz"])
    rw = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    rz = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return 2.0 * math.atan2(rz, rw)


class Arm:
    """A stateful fake YAM with a lid on a thread.

    Moves land where commanded unless told otherwise. The hand's rotation is
    DOWN turned by joint 6 (``pose_sign`` relates a tool roll about z to joint
    6). While the jaws are closed on the lid, joint 6 turns the lid: its
    tightening coordinate ``lid`` (+ with joint 6 rising) stops at ``seat``
    (tighten) and at ``stuck`` (loosen); below ``free_at`` the cap is off the
    thread and lifts with the hand. ``slip_at``: past this much turn in one
    grip, the jaws lose the lid.
    """

    def __init__(
        self,
        lid_present=True,
        seat=math.inf,
        stuck=-math.inf,
        free_at=-math.inf,
        slip_at=math.inf,
        joints_refused=(),
        pose_sign=1.0,
        pose_moves_joint=True,
        refuse_pose=None,
        land_off=0.0,
        drift=0.0,
        ik=True,
    ):
        self.p = np.array([0.30, 0.0, 0.25])
        self.q = Q0.copy()
        self.lid_present = lid_present
        self.seat, self.stuck, self.free_at, self.slip_at = seat, stuck, free_at, slip_at
        self.joints_refused = joints_refused
        self.pose_sign = pose_sign
        self.pose_moves_joint = pose_moves_joint
        self.refuse_pose = refuse_pose or (lambda tool, pose: None)
        self.land_off = land_off
        self.drift = drift
        self.ik = ik
        self.closed = False
        self.lost = False
        self.lid = 0.0
        self.grip_turn = 0.0
        self.calls = []
        self.moves = []
        self.joint_moves = []
        self.ik_calls = []

    def rot(self):
        return _rz(DOWN, self.pose_sign * (self.q[-1] - Q0[-1]))

    def holding(self):
        return self.closed and self.lid_present and not self.lost

    def _turn_to(self, q6):
        """Joint 6 toward q6, turning (and stopped by) the lid when held."""
        if self.holding():
            d = q6 - self.q[-1]
            new = min(self.seat, max(self.stuck, self.lid + d))
            d = new - self.lid
            self.lid = new
            self.grip_turn += abs(d)
            if self.grip_turn > self.slip_at:
                self.lost = True
            q6 = self.q[-1] + d
        self.q[-1] = q6

    def _pose_move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.calls.append(tool)
            self.moves.append((tool, target.copy(), pose["rotation"], kwargs))
            why = self.refuse_pose(tool, pose)
            if why == "raise":
                raise ToolError(tool, "no IK solution")
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            if (
                self.holding()
                and target[2] > self.p[2] + 1e-6
                and self.lid > self.free_at
                and tool == "robot.go_to_pose_cartesian"
            ):
                return {"reached": True, "outcome": "reached"}  # held down by the thread
            self.p = target + np.array([self.land_off, 0.0, 0.0])
            turn = _about_z(self.rot(), pose["rotation"])
            if abs(turn) > 1e-9 and self.pose_moves_joint:
                self._turn_to(self.q[-1] + self.pose_sign * turn)
            return {"reached": True, "outcome": "reached"}

        return go

    def _joints(self, joint_config, **kwargs):
        index = len(self.joint_moves)
        self.calls.append("robot.move_to_joints")
        target = np.array(joint_config["positions"], dtype=float)
        self.joint_moves.append((target, kwargs))
        if self.joints_refused == "all" or index in self.joints_refused:
            raise ToolError("robot.move_to_joints", "the station bridge takes poses only")
        self.p[2] += (target[0] - self.q[0]) / 10.0  # the IK below, inverted
        self.q[:-1] = target[:-1]
        self._turn_to(target[-1])
        return None

    def _gripper(self, close):
        def go(**kwargs):
            self.calls.append("robot.close_gripper" if close else "robot.open_gripper")
            self.closed = close
            self.lost = False
            self.grip_turn = 0.0
            return {"success": True}

        return go

    def fraction(self):
        if not self.closed:
            return 1.0
        return 0.33 if self.holding() else 0.05

    def hand(self):
        return self.p + np.array([self.drift * self.grip_turn * self.holding(), 0.0, 0.0])

    def _solve_ik(self, pose, **kwargs):
        self.ik_calls.append(pose)
        if not self.ik:
            raise ToolError("robot.solve_ik", "Target pose is unreachable")
        z = pose["position"]["z"]
        return {
            "joint_config": {"positions": [*(Q0[:5] + 10.0 * (z - GRIP[2])), 0.0]},
            "success": True,
        }

    def tools(self):
        return {
            "robot.get_ee_pose": lambda **kw: {
                "pose": {
                    "position": dict(zip("xyz", map(float, self.hand()), strict=True)),
                    "rotation": self.rot(),
                }
            },
            "robot.get_observation": lambda **kw: {
                "cameras": [],
                "arms": [
                    {
                        "joint_state": {"positions": self.q.copy()},
                        "gripper_fraction": self.fraction(),
                    }
                ],
            },
            "robot.grasp_frame": lambda **kw: {"rotation": DOWN, "rotation_flipped": FLIPPED},
            "robot.go_to_pose": self._pose_move("robot.go_to_pose"),
            "robot.go_to_pose_cartesian": self._pose_move("robot.go_to_pose_cartesian"),
            "robot.move_to_joints": self._joints,
            "robot.open_gripper": self._gripper(False),
            "robot.close_gripper": self._gripper(True),
            "robot.solve_ik": self._solve_ik,
        }

    def ctx(self):
        return FakeContext(self.tools())


def _phases(out):
    return [
        p
        for i, p in enumerate(out["trajectory_phase"])
        if i == 0 or out["trajectory_phase"][i - 1] != p
    ]


def test_tightening_grips_from_above_and_ratchets_the_upstream_stroke(script, no_waiting):
    arm = Arm()
    out = script.run(arm.ctx(), lid_obb=LID)
    assert out["route"] == "turned", out["detail"]
    assert out["strokes_done"] == 4  # yam-arcade's twist_on
    assert not out["seated"] and "no seat felt" in out["detail"]
    assert out["turn_per_stroke"] == pytest.approx([STROKE] * 4)
    assert out["total_turn"] == pytest.approx(4 * STROKE)
    assert out["rolled_with"] == "joints"
    assert arm.lid == pytest.approx(4 * STROKE)
    # The approach: open, over the lid top-down, joint 6 to -1.9, down, close.
    assert arm.calls[:3] == ["robot.open_gripper", "robot.go_to_pose", "robot.move_to_joints"]
    np.testing.assert_allclose(arm.moves[0][1], PRE, atol=1e-9)
    assert arm.moves[0][2] == DOWN
    assert arm.moves[1][0] == "robot.go_to_pose_cartesian"
    np.testing.assert_allclose(arm.moves[1][1], GRIP, atol=1e-9)
    # Each stroke -1.9 -> +1.9, open, roll back to -1.9 with the hand open, re-grip.
    assert _phases(out) == [
        "start", "to_start", "gripped", "close",
        "stroke1", "back1", "close", "stroke2", "back2", "close", "stroke3", "back3", "close", "stroke4",
    ]  # fmt: skip
    q6 = np.array(out["q6_trajectory"])
    ph = np.array(out["trajectory_phase"])
    assert q6[ph == "to_start"][-1] == pytest.approx(-1.9)
    for k in (1, 2, 3, 4):
        assert q6[ph == f"stroke{k}"][-1] == pytest.approx(1.9)
        assert len(q6[ph == f"stroke{k}"]) == N
    for k in (1, 2, 3):
        assert q6[ph == f"back{k}"][-1] == pytest.approx(-1.9)
        assert np.all(np.array(out["grip_trajectory"])[ph == f"back{k}"] == 1.0)  # open
    assert np.all(np.array(out["grip_trajectory"])[np.char.startswith(ph, "stroke")] == 0.33)
    assert out["grip_at_close"] == [0.33] * 4
    # Every step within pi/40, every other joint held.
    targets = np.array([t for t, _ in arm.joint_moves])
    assert np.max(np.abs(np.diff(np.concatenate([[0.0], targets[:, -1]])))) <= math.pi / 40 + 1e-9
    np.testing.assert_allclose(targets[:, :5], np.tile(Q0[:5], (len(targets), 1)))
    # Tightening lets go and rises 25 mm.
    assert arm.calls[-2:] == ["robot.open_gripper", "robot.go_to_pose_cartesian"]
    np.testing.assert_allclose(arm.p, GRIP + [0.0, 0.0, 0.025], atol=1e-9)
    assert not out["holding"]
    # No connector knobs unless set.
    assert all(kw == {} for _, kw in arm.joint_moves)
    assert all(kw == {} for *_, kw in arm.moves)


def test_steps_are_paced_to_half_a_radian_per_second(script, no_waiting):
    script.run(Arm().ctx(), lid_obb=LID, strokes=1)
    stroke_period = STROKE / N / 0.5
    start_period = 1.9 / N_START / 0.5
    assert len(no_waiting) == N_START + N
    assert all(s == pytest.approx(start_period, abs=0.02) for s in no_waiting[:N_START])
    assert all(s == pytest.approx(stroke_period, abs=0.02) for s in no_waiting[N_START:])
    assert max(math.pi / 40 / s for s in no_waiting) <= 0.5 * 1.1


def test_a_wrist_that_stalls_after_turning_has_seated_the_lid(script):
    arm = Arm(seat=STROKE + 1.0)  # seats 1 rad into the second stroke
    out = script.run(arm.ctx(), lid_obb=LID)
    assert out["route"] == "turned", out["detail"]
    assert out["seated"] and out["detail"].startswith("seated: stroke 2")
    assert out["strokes_done"] == 2
    assert out["turn_per_stroke"][0] == pytest.approx(STROKE)
    assert out["turn_per_stroke"][1] == pytest.approx(1.0)
    assert "stroke2_hold" in out["trajectory_phase"]
    # The lagging target is held, not advanced: never more than stall_lag and a step past the stop.
    targets = np.array([t for t, _ in arm.joint_moves])[:, -1]
    stroke2 = targets[np.flatnonzero(np.isclose(targets, -1.9))[-1] :]
    stop = -1.9 + 1.0
    assert np.max(stroke2) <= stop + 0.12 + math.pi / 40 + 1e-9
    # The relief: joint 6 commanded at its reading, then the hand lets go.
    assert targets[-1] == pytest.approx(stop)
    assert arm.calls[-2:] == ["robot.open_gripper", "robot.go_to_pose_cartesian"]


def test_a_tightening_stall_with_no_turn_is_stalled(script):
    arm = Arm(seat=0.05)  # already tight
    out = script.run(arm.ctx(), lid_obb=LID)
    assert out["route"] == "stalled"
    assert "the turn stopped" in out["detail"] and not out["seated"]
    assert out["total_turn"] == pytest.approx(0.05)
    assert arm.calls[-2:] == ["robot.open_gripper", "robot.go_to_pose_cartesian"]


def test_loosening_turns_the_other_way_and_lifts_the_cap(script):
    arm = Arm(free_at=-2 * STROKE)
    out = script.run(arm.ctx(), lid_obb=LID, direction="loosen")
    assert out["route"] == "turned", out["detail"]
    assert out["strokes_done"] == 5  # RoboWaiter's untwist
    assert out["turn_per_stroke"] == pytest.approx([STROKE] * 5)
    q6 = np.array(out["q6_trajectory"])
    ph = np.array(out["trajectory_phase"])
    assert q6[ph == "to_start"][-1] == pytest.approx(1.9)
    assert q6[ph == "stroke1"][-1] == pytest.approx(-1.9)
    assert arm.lid == pytest.approx(-5 * STROKE)
    # The lift check: the hand rose 25 mm still holding, and keeps the cap.
    assert out["lift_rise"] == pytest.approx(0.025)
    assert out["holding"] and arm.closed
    assert arm.calls[-1] == "robot.go_to_pose_cartesian"


def test_a_cap_still_on_its_thread_fails_the_lift_check(script):
    arm = Arm()  # never comes free
    out = script.run(arm.ctx(), lid_obb=LID, direction="loosen", strokes=2)
    assert out["route"] == "stalled"
    assert "still on its thread" in out["detail"]
    assert out["lift_rise"] == pytest.approx(0.0)
    assert not out["holding"]
    assert arm.calls[-2:] == ["robot.open_gripper", "robot.go_to_pose_cartesian"]


def test_a_cap_that_will_not_break_loose_is_stalled(script):
    arm = Arm(stuck=-0.5)
    out = script.run(arm.ctx(), lid_obb=LID, direction="loosen")
    assert out["route"] == "stalled"
    assert out["detail"].startswith("stroke 1: the turn stopped")
    assert out["turn_per_stroke"] == pytest.approx([0.5])


def test_loosening_without_the_lift_check_keeps_the_grip(script):
    arm = Arm()
    out = script.run(arm.ctx(), lid_obb=LID, direction="loosen", strokes=1, lift_check=False)
    assert out["route"] == "turned" and "not lift-checked" in out["detail"]
    assert out["holding"] and arm.closed
    assert arm.calls[-1] == "robot.move_to_joints"


def test_jaws_that_lose_the_lid_mid_stroke_route_slipped(script):
    arm = Arm(slip_at=2.0)
    out = script.run(arm.ctx(), lid_obb=LID)
    assert out["route"] == "slipped"
    assert out["detail"].startswith("stroke 1: the gripper reading fell to 0.05")
    assert not out["holding"]


def test_jaws_that_close_on_nothing_route_slipped(script):
    arm = Arm(lid_present=False)
    out = script.run(arm.ctx(), lid_obb=LID)
    assert out["route"] == "slipped"
    assert "closed on nothing" in out["detail"]
    assert out["strokes_done"] == 0


def test_an_unreachable_lid(script):
    arm = Arm(refuse_pose=lambda tool, pose: "raise" if tool == "robot.go_to_pose" else None)
    out = script.run(arm.ctx(), lid_obb=LID)
    assert out["route"] == "unreachable"
    assert out["detail"].count("no IK solution") == 2  # both grasp-frame rotations tried
    assert arm.joint_moves == []

    arm = Arm(refuse_pose=lambda tool, pose: "joint limit" if pose["rotation"] == DOWN else None)
    out = script.run(arm.ctx(), lid_obb=LID)
    assert out["route"] == "turned"  # the flipped rotation reached
    assert arm.moves[1][2] == FLIPPED

    arm = Arm(land_off=0.03)
    out = script.run(arm.ctx(), lid_obb=LID)
    assert out["route"] == "unreachable" and "30 mm off" in out["detail"]

    arm = Arm(
        refuse_pose=lambda tool, pose: "contact" if tool == "robot.go_to_pose_cartesian" else None
    )
    out = script.run(arm.ctx(), lid_obb=LID)
    assert out["detail"].startswith("descent to the grip point: unreachable: contact")
    assert out["route"] == "unreachable"


def test_a_stroke_that_does_not_fit_the_wrist(script):
    arm = Arm()
    out = script.run(arm.ctx(), lid_obb=LID, stroke_limits=[-2.1, 2.1])
    assert out["route"] == "unreachable" and "does not fit" in out["detail"]
    assert arm.joint_moves == []
    out = script.run(
        Arm().ctx(), lid_obb=LID, stroke_limits=[-2.1, 2.1], roll_limits=[-2.9, 2.9], strokes=1
    )
    assert out["route"] == "turned"


def test_a_refused_stroke_step_is_stalled_and_released(script):
    arm = Arm(joints_refused=(N_START + 10,))
    out = script.run(arm.ctx(), lid_obb=LID)
    assert out["route"] == "stalled"
    assert out["detail"].startswith("stroke 1: ") and "takes poses only" in out["detail"]
    assert not arm.closed


def test_a_drifting_hand_is_stalled(script):
    out = script.run(Arm(drift=0.03).ctx(), lid_obb=LID)
    assert out["route"] == "stalled" and "drifted" in out["detail"]
    assert out["hand_drift"] > 0.02


@pytest.mark.parametrize("pose_sign", [1.0, -1.0])
def test_a_connector_without_joint_moves_turns_by_pose(script, pose_sign):
    arm = Arm(joints_refused="all", pose_sign=pose_sign)
    out = script.run(arm.ctx(), lid_obb=LID, strokes=2)
    assert out["route"] == "turned", out["detail"]
    assert out["rolled_with"] == "pose"
    assert len(arm.joint_moves) == 1
    assert out["turn_per_stroke"] == pytest.approx([STROKE, STROKE], abs=1e-6)
    q6 = np.array(out["q6_trajectory"])
    ph = np.array(out["trajectory_phase"])
    assert q6[ph == "stroke1"][-1] == pytest.approx(1.9, abs=1e-6)
    # Stroke steps keep the tool centre over the grip point.
    rolls = [m for m in arm.moves if m[0] == "robot.go_to_pose"][2:]
    assert all(np.allclose(m[1][:2], GRIP[:2]) for m in rolls)


def test_a_pose_roll_that_does_not_turn_joint_6_is_unreachable(script):
    out = script.run(Arm(pose_moves_joint=False).ctx(), lid_obb=LID, roll_with="pose")
    assert out["route"] == "unreachable"
    assert "moved joint 6 only" in out["detail"]


def test_the_hand_follows_the_thread_pitch(script):
    arm = Arm()
    out = script.run(arm.ctx(), lid_obb=LID, strokes=2, pitch=0.004)
    assert out["route"] == "turned"
    drop = 0.004 * STROKE / (2 * math.pi)
    assert [p["position"]["z"] for p in arm.ik_calls] == pytest.approx(
        [GRIP[2] - drop, GRIP[2] - 2 * drop]
    )
    # Joints 1-5 walk to the IK solution over the stroke; joint 6 is still the stroke.
    stroke1 = [t for t, _ in arm.joint_moves][N_START : N_START + N]
    np.testing.assert_allclose(stroke1[-1][:5], Q0[:5] - 10.0 * drop, atol=1e-9)
    np.testing.assert_allclose(
        stroke1[N // 2][:5], Q0[:5] - 10.0 * drop * (N // 2 + 1) / N, atol=1e-9
    )
    assert stroke1[-1][-1] == pytest.approx(1.9)

    arm = Arm(ik=False)
    out = script.run(arm.ctx(), lid_obb=LID, strokes=1, pitch=0.004)
    assert out["route"] == "turned" and "held its height" in out["detail"]


def test_connector_knobs_are_passed_only_when_set(script):
    arm = Arm()
    script.run(arm.ctx(), lid_obb=LID, strokes=1, tolerance=0.01, max_steps=200, arm_id=0)
    assert all(kw == {"tolerance": 0.01, "max_steps": 200} for _, kw in arm.joint_moves)


def test_a_given_rotation_skips_the_grasp_frame(script):
    arm = Arm()
    tools = arm.tools()
    del tools["robot.grasp_frame"]
    out = script.run(FakeContext(tools), lid_obb=LID, rotation=DOWN, strokes=1)
    assert out["route"] == "turned"


def test_bad_inputs_raise(script):
    with pytest.raises(ValueError, match="direction"):
        script.run(Arm().ctx(), lid_obb=LID, direction="twist")
    with pytest.raises(ValueError, match="roll_with"):
        script.run(Arm().ctx(), lid_obb=LID, roll_with="wrist")


def test_no_joint_reading_is_unreachable(script):
    tools = Arm().tools()
    tools["robot.get_observation"] = {"cameras": [], "arms": []}
    out = script.run(FakeContext(tools), lid_obb=LID)
    assert out["route"] == "unreachable" and "no joint reading" in out["detail"]


def test_the_example_declares_every_route():
    path = (
        Path(__file__).resolve().parents[2]
        / "skills"
        / BUNDLE
        / "examples"
        / "canonical_subgraph.json"
    )
    example = json.loads(path.read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == {"turned", "stalled", "slipped", "unreachable"}
