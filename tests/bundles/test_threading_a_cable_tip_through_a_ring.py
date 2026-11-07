"""threading-a-cable-tip-through-a-ring: FakeContext tests of the aim, the turn, the push and the routing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "threading-a-cable-tip-through-a-ring"
REACH = 0.0102
INNER = 0.0191  # the threading station's ring: 21.1 mm radius, 2 mm wire
# A ring standing at z = 0.10 with its axis along -y; the hand holds a cable
# whose tip points -y, 45 mm behind the tip.
CENTRE = {"x": 0.40, "y": 0.10, "z": 0.10}
AXIS = {"x": 0.0, "y": -1.0, "z": 0.0}
START = (0.40, 0.25, 0.06)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["thread_tip_through_ring"].module


def down(a_deg: float) -> dict[str, float]:
    """Top-down: a half turn about the horizontal axis at a_deg (tool x to heading 2 * a_deg)."""
    a = math.radians(a_deg)
    return {"w": 0.0, "x": math.cos(a), "y": math.sin(a), "z": 0.0}


def matrix(a_deg: float) -> np.ndarray:
    a = math.radians(2.0 * a_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


class Arm:
    """A top-down hand holding a cable end fixed in it. Moves land where commanded,
    except that the hand cannot pass ``block_y`` going -y (a jam). ``free_roll``
    keeps the hand's own yaw on pose moves (the last joint turns it, gain -1)."""

    def __init__(
        self,
        tip_drop=0.0,
        free_roll=False,
        block_y=None,
        refuse=None,
        ik=True,
        raise_on=None,
        camera_error=None,
        camera_after=None,
        tilt=False,
        drop_on=None,
    ):
        self.p = np.array(START, dtype=float)
        self.a = 45.0  # tool x along +y: the jaws close across a cable lying along y
        self.q = np.zeros(6)
        exit_w = self.p + np.array([0.0, -0.010, -(REACH - 0.004)])
        tip_w = exit_w + np.array(
            [0.0, -0.035 * math.cos(math.asin(min(tip_drop / 0.035, 1.0))), -tip_drop]
        )
        self.tip_t = matrix(self.a).T @ (tip_w - self.p)
        self.exit_t = matrix(self.a).T @ (exit_w - self.p)
        self.free_roll = free_roll
        self.block_y = block_y
        self.refuse = refuse or (lambda tool, target: None)
        self.ik = ik
        self.raise_on = raise_on
        self.camera_error = camera_error  # added to the true tip in every look
        self.camera_after = camera_after  # what the look after the push sees, in place of the tip
        self.tilt = tilt
        self.drop_on = drop_on  # "start", "turn" or "push": when the cable slides out of the hand
        self.lost = drop_on == "start"
        self.pushed = False
        self.moves = []
        self.grips = []

    def tip(self) -> np.ndarray:
        return self.p + matrix(self.a) @ self.tip_t

    def exit(self) -> np.ndarray:
        return self.p + matrix(self.a) @ self.exit_t

    def _move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy()))
            if self.drop_on == "turn" and tool == "robot.go_to_pose":
                self.lost = True
            if self.raise_on and self.raise_on(tool, target):
                raise RuntimeError("did not reach its target (40 mm)")
            why = self.refuse(tool, target)
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            if self.block_y is not None and target[1] < self.block_y:
                self.p = np.array([target[0], self.block_y, target[2]])
                self.pushed = True
                return {"reached": False, "outcome": "timeout"}
            if target[1] < CENTRE["y"] + 0.05:  # past every pre-insert pose: the push
                self.pushed = True
                self.lost = self.lost or self.drop_on == "push"
            self.p = target
            if not self.free_roll:
                self.a = math.degrees(math.atan2(pose["rotation"]["y"], pose["rotation"]["x"]))
            return {"reached": True, "outcome": "reached"}

        return go

    def _joints(self, joint_config, **kwargs):
        q = np.asarray(joint_config["positions"], dtype=float)
        self.a -= math.degrees(q[-1] - self.q[-1]) / 2.0
        self.q = q
        return {"reached": True, "outcome": "reached"}

    @staticmethod
    def _depth():
        """The cable 0.3 m from the camera round the pointed pixel, the table 0.5 m behind."""
        d = np.full((100, 100), 0.5)
        d[48:53, 48:53] = 0.3
        return d

    def _observation(self, **kwargs):
        obs = {"arms": [{"joint_state": {"positions": self.q.tolist()}}], "cameras": []}
        if self.camera_error is not None or self.camera_after is not None:
            seen = self.tip() + (self.camera_error if self.camera_error is not None else 0.0)
            if self.pushed and self.camera_after is not None:
                seen = np.asarray(self.camera_after, dtype=float)
            pose = np.eye(4)
            pose[:3, 3] = seen - np.array([0.0, 0.0, 0.3])
            obs["cameras"] = [
                {
                    "name": "left_wrist",
                    "rgb": np.zeros((100, 100, 3), np.uint8),
                    "depth": self._depth(),
                    "intrinsics": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
                    "pose": pose,
                }
            ]
        return obs

    def ctx(self):
        def pose(**kw):
            rot = down(self.a)
            if self.tilt:
                rot = {
                    "w": math.cos(math.pi / 4),
                    "x": math.sin(math.pi / 4),
                    "y": 0.0,
                    "z": 0.0,
                }  # tool z level
            return {
                "pose": {
                    "position": dict(zip("xyz", map(float, self.p), strict=True)),
                    "rotation": rot,
                }
            }

        tools = {
            "robot.get_ee_pose": pose,
            "robot.go_to_pose": self._move("robot.go_to_pose"),
            "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
            "robot.move_to_joints": self._joints,
            "robot.solve_ik": lambda pose, **kw: {
                "success": self.ik,
                "within_servo_tolerance": self.ik,
            },
            "robot.describe_gripper": {
                "finger": {"reach_m": REACH},
                "width_fit": {"at_open_m": 0.094, "at_closed_m": 0.0},
            },
            "robot.get_gripper": lambda **kw: {"position": 0.0 if self.lost else 0.058},
            "robot.open_gripper": lambda **kw: self.grips.append("open") or {"position": 1.0},
            "robot.get_observation": self._observation,
            "gemini-er.point": lambda image, query, **kw: {
                "pixel_x": 50.0,
                "pixel_y": 50.0,
                "found": True,
            },
        }
        return FakeContext(tools)


def vec(v) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def run(script, arm, **kwargs):
    kwargs.setdefault("ring_centre", CENTRE)
    kwargs.setdefault("ring_axis", AXIS)
    kwargs.setdefault("ring_inner_radius", INNER)
    if "tip_position" not in kwargs and "grasp_back" not in kwargs and "tip_in_tool" not in kwargs:
        kwargs["tip_position"] = vec(arm.tip())
        kwargs.setdefault("exit_position", vec(arm.exit()))
    return script.run(arm.ctx(), **kwargs)


def axial(p) -> float:
    return float(
        (np.asarray(p) - np.array([CENTRE[k] for k in "xyz"])) @ np.array([0.0, -1.0, 0.0])
    )


def radial(p) -> float:
    d = np.asarray(p) - np.array([CENTRE[k] for k in "xyz"])
    return float(math.hypot(d[0], d[2]))


def test_happy_path_level_end(script):
    arm = Arm()
    out = run(script, arm)
    assert out["route"] == "threaded", out["detail"]
    tools = [m[0] for m in arm.moves]
    assert tools == ["robot.go_to_pose"] + ["robot.go_to_pose_cartesian"] * 3
    # The pre-insert pose: the tip 50 mm in front of the plane, on the axis.
    pre = arm.moves[2][1]
    tip_pre = pre + matrix(arm.a) @ arm.tip_t
    assert axial(tip_pre) == pytest.approx(-0.05, abs=1e-6)
    assert radial(tip_pre) < 0.003
    # The fingertips' front stops 5 mm in front of the ring's 2 mm wire.
    assert out["tip_past"] == pytest.approx(out["push_planned"] - 0.05, abs=1e-6)
    front = axial(arm.p) + 0.010
    assert front == pytest.approx(-0.007, abs=1e-6)
    assert axial(arm.tip()) == pytest.approx(out["tip_past"], abs=1e-6)
    assert out["tip_past"] > 0.01
    assert abs(out["sag_deg"]) < 1.0 and not out["tip_seen"]
    assert "not seen" in out["detail"]
    assert arm.grips == []


def test_carry_from_in_front_goes_level_with_the_mouth(script):
    arm = Arm()
    run(script, arm)
    level, mouth = (m[1] for m in arm.moves[1:3])
    assert level[:2] == pytest.approx(START[:2]) and level[2] == pytest.approx(mouth[2])


def test_carry_from_past_the_mouth_comes_down_from_above(script):
    arm = Arm()
    arm.p = np.array([0.40, 0.16, 0.06])  # the hand already past the pre-insert pose along the axis
    out = run(script, arm)
    assert out["route"] == "threaded", out["detail"]
    up, over, mouth = (m[1] for m in arm.moves[1:4])
    assert up[:2] == pytest.approx([0.40, 0.16]) and up[2] == pytest.approx(over[2])
    assert over[:2] == pytest.approx(mouth[:2]) and over[2] == pytest.approx(mouth[2] + 0.06)


def test_axis_sign_is_taken_from_the_hand_side(script):
    arm = Arm()
    out = run(script, arm, ring_axis={"x": 0.0, "y": 1.0, "z": 0.0})
    assert out["route"] == "threaded"
    assert axial(arm.tip()) > 0.01  # still pushed -y, away from the hand


def test_keep_axis_sign_pushes_the_given_way(script):
    arm = Arm()
    out = run(
        script,
        arm,
        ring_axis={"x": 0.0, "y": 1.0, "z": 0.0},
        keep_axis_sign=True,
        check_reach=False,
    )
    # Pushed +y as given: the end turned round, the hand (past the pre-insert
    # pose along that axis) carried over the ring and down on its far side.
    assert out["route"] == "threaded", out["detail"]
    assert abs(out["turned_deg"]) == pytest.approx(180.0, abs=0.5)
    pre, end = arm.moves[-2][1], arm.moves[-1][1]
    assert pre[1] < CENTRE["y"] and end[1] > pre[1]
    assert [m[0] for m in arm.moves].count("robot.go_to_pose_cartesian") == 4
    assert axial(arm.tip()) < -0.01  # past the plane the other way


def test_fingers_let_into_the_plane_pass_the_whole_stretch(script):
    arm = Arm(tip_drop=0.022)
    out = run(script, arm, finger_clear=-0.007)
    assert out["route"] == "threaded", out["detail"]
    assert out["sag_deg"] == pytest.approx(math.degrees(math.asin(0.022 / 0.035)), abs=0.5)
    # The fingers 5 mm into the plane: the whole stretch passes, its middle on the axis.
    assert axial(arm.p) + 0.010 == pytest.approx(0.005, abs=1e-6)
    assert out["tip_offset"] == pytest.approx(0.011, abs=1e-6)
    assert out["exit_offset"] == pytest.approx(0.011, abs=1e-6)


def test_sagging_end_is_aimed_between_tip_and_fingers(script):
    arm = Arm(tip_drop=0.030)
    out = run(script, arm)
    assert out["route"] == "threaded", out["detail"]
    # The cable at the fingers stands outside the opening, so they stop in front of the wire.
    assert out["exit_offset"] > INNER - 0.00175 - 0.004
    assert axial(arm.p) + 0.010 == pytest.approx(-0.007, abs=1e-6)
    # Tip and crossing equally far off the axis, both inside.
    assert out["tip_offset"] == pytest.approx(out["crossing_offset"], abs=1e-6)
    assert 0.0 < out["tip_offset"] < INNER - 0.00175 - 0.004
    tip_end = arm.tip()
    assert radial(tip_end) == pytest.approx(out["tip_offset"], abs=1e-6)
    assert tip_end[2] < CENTRE["z"]  # the tip under the axis, the fingers' end of the cable over it


def test_too_much_sag_is_misaligned_before_moving(script):
    arm = Arm(tip_drop=0.030)
    out = run(script, arm, ring_inner_radius=0.014)
    assert out["route"] == "misaligned"
    assert "sags" in out["detail"]
    assert arm.moves == []


def test_hold_too_short_is_misaligned(script):
    arm = Arm()
    out = run(script, arm, grasp_back=0.012, tip_direction={"x": 0.0, "y": -1.0, "z": 0.0})
    assert out["route"] == "misaligned"
    assert "further behind" in out["detail"]
    assert arm.moves == []


def test_ring_too_small_for_the_cable(script):
    arm = Arm()
    out = run(script, arm, ring_inner_radius=0.005)
    assert out["route"] == "misaligned" and "no room" in out["detail"]


def test_given_hold_with_sag(script):
    arm = Arm()
    out = run(
        script, arm, grasp_back=0.045, tip_direction={"x": 0.0, "y": -1.0, "z": 0.0}, sag_deg=30.0
    )
    assert out["route"] == "threaded", out["detail"]
    assert out["sag_deg"] == pytest.approx(30.0, abs=0.5)
    assert "sagging 30 deg" in out["detail"]


def test_tip_in_tool(script):
    arm = Arm()
    out = run(script, arm, tip_in_tool=vec(arm.tip_t))
    assert out["route"] == "threaded"
    assert out["tip_position"]["y"] == pytest.approx(arm.tip()[1])


def test_turn_finished_on_the_last_joint(script):
    arm = Arm(free_roll=True)
    heading = math.radians(-60.0)
    out = run(script, arm, ring_axis={"x": math.cos(heading), "y": math.sin(heading), "z": 0.0})
    assert out["route"] == "threaded", out["detail"]
    assert out["turned_deg"] == pytest.approx(30.0, abs=0.5)
    way = arm.tip() - arm.exit()
    assert math.degrees(math.atan2(way[1], way[0])) == pytest.approx(-60.0, abs=1.0)
    assert arm.q[-1] != 0.0


def test_turn_with_a_full_pose_arm(script):
    arm = Arm()
    heading = math.radians(-110.0)
    out = run(script, arm, ring_axis={"x": math.cos(heading), "y": math.sin(heading), "z": 0.0})
    assert out["route"] == "threaded", out["detail"]
    assert out["turned_deg"] == pytest.approx(-20.0, abs=0.5)
    assert arm.q[-1] == 0.0


def test_jam_backs_off_to_the_mouth(script):
    arm = Arm(block_y=CENTRE["y"] + 0.045)  # the hand held back: the tip never reaches the plane
    out = run(script, arm)
    assert out["route"] == "jammed", out["detail"]
    assert out["tip_past"] < 0.01
    assert "backed off" in out["detail"]
    assert arm.moves[-1][1] == pytest.approx(arm.moves[2][1])  # back at the pre-insert pose


def test_stopped_short_with_the_tip_past_is_threaded(script):
    arm = Arm(block_y=CENTRE["y"] + 0.030)
    out = run(script, arm)
    assert out["route"] == "threaded", out["detail"]
    assert "stopped" in out["detail"] and out["tip_past"] >= 0.01


def test_ik_preview_refusal_is_unreachable(script):
    arm = Arm(ik=False)
    out = run(script, arm)
    assert out["route"] == "unreachable" and "wrist solution" in out["detail"]
    assert arm.moves == []


def test_routed_refusal_while_carrying(script):
    arm = Arm(
        refuse=lambda tool, t: (
            "out of reach" if tool == "robot.go_to_pose_cartesian" and t[1] > 0.2 else None
        )
    )
    out = run(script, arm)
    assert out["route"] == "unreachable" and "carrying" in out["detail"]


def test_raised_push_failure_routed(script):
    arm = Arm(
        raise_on=lambda tool, t: tool == "robot.go_to_pose_cartesian" and t[1] < CENTRE["y"] + 0.03
    )
    out = run(script, arm)
    # The hand did not move on the push: the tip stays in front of the plane.
    assert out["route"] == "jammed"
    assert "did not reach" in out["detail"]


def test_refused_push_is_unreachable(script):
    arm = Arm(refuse=lambda tool, t: "no IK" if t[1] < CENTRE["y"] + 0.03 else None)
    out = run(script, arm)
    assert out["route"] == "unreachable" and "push was refused" in out["detail"]


def test_tilted_hand_is_unreachable(script):
    arm = Arm(tilt=True)
    out = run(script, arm)
    assert out["route"] == "unreachable" and "straight down" in out["detail"]
    assert arm.moves == []


def test_vertical_axis_is_unreachable(script):
    arm = Arm()
    out = run(script, arm, ring_axis={"x": 0.0, "y": 0.0, "z": -1.0})
    assert out["route"] == "unreachable" and "vertical" in out["detail"]


def test_camera_reaims_and_confirms(script):
    arm = Arm(camera_error=np.array([0.0, 0.0, -0.012]))  # the tip hangs 12 mm lower than given
    out = run(script, arm, camera="wrist")
    assert out["route"] == "threaded", out["detail"]
    assert out["tip_seen"] and "camera sees the tip" in out["detail"]
    # One re-aim: a fourth cartesian move before the push, higher by part of the
    # 12 mm the seen tip hangs under the given one (the aim is between it and the fingers).
    carts = [m for m in arm.moves if m[0] == "robot.go_to_pose_cartesian"]
    assert len(carts) == 4
    assert 0.004 < carts[2][1][2] - carts[1][1][2] < 0.012
    assert out["seen_tip_past"] > 0.01


def test_camera_sees_tip_short_after_push(script):
    short = np.array([CENTRE["x"], CENTRE["y"] + 0.02, CENTRE["z"] - 0.03])
    arm = Arm(camera_error=np.zeros(3), camera_after=short)
    out = run(script, arm, camera="wrist")
    assert out["route"] == "jammed" and "camera sees the tip" in out["detail"]
    assert out["seen_tip_past"] == pytest.approx(-0.02, abs=1e-3)


def test_camera_sees_tip_past_but_outside_the_ring(script):
    # Fingers stopped in front of the wire (a sagging end); the tip seen past the
    # plane but far to the side: the line to it crosses outside the ring.
    aside = np.array([CENTRE["x"] + 0.12, CENTRE["y"] - 0.02, CENTRE["z"]])
    arm = Arm(tip_drop=0.030, camera_error=np.zeros(3), camera_after=aside)
    out = run(script, arm, camera="wrist")
    assert out["route"] == "misaligned" and "outside the ring" in out["detail"]


def test_camera_not_finding_the_tip_falls_back_to_the_hand(script):
    arm = Arm()
    ctx = arm.ctx()
    ctx_tools = ctx  # the observation has no camera named wrist
    out = script.run(
        ctx_tools,
        ring_centre=CENTRE,
        ring_axis=AXIS,
        ring_inner_radius=INNER,
        tip_position=vec(arm.tip()),
        camera="wrist",
    )
    assert out["route"] == "threaded"
    assert not out["tip_seen"] and "did not find the tip" in out["detail"]


def test_release_opens_and_lifts(script):
    arm = Arm()
    out = run(script, arm, release=True)
    assert out["route"] == "threaded"
    assert arm.grips == ["open"]
    assert arm.moves[-1][1][2] == pytest.approx(arm.moves[-2][1][2] + 0.04)


def test_needs_a_hold(script):
    arm = Arm()
    with pytest.raises(ValueError):
        script.run(arm.ctx(), ring_centre=CENTRE, ring_axis=AXIS, ring_inner_radius=INNER)


def test_example_declares_every_route():
    root = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
    example = json.loads((root / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == {"threaded", "jammed", "misaligned", "unreachable"}


def test_empty_hand_is_refused_before_moving(script):
    arm = Arm(drop_on="start")
    out = run(script, arm)
    assert out["route"] == "unreachable" and "no cable in the hand" in out["detail"]
    assert arm.moves == []


def test_cable_lost_while_turning_is_jammed_without_a_push(script):
    arm = Arm(drop_on="turn")
    out = run(script, arm)
    assert out["route"] == "jammed" and out["cable_lost"]
    assert "before the push" in out["detail"]
    assert not arm.pushed


def test_cable_lost_during_the_push_is_jammed(script):
    arm = Arm(drop_on="push")
    out = run(script, arm)
    assert out["route"] == "jammed" and out["cable_lost"]
    assert "during the push" in out["detail"]


def test_held_cable_is_not_lost(script):
    arm = Arm()
    out = run(script, arm)
    assert out["route"] == "threaded" and not out["cable_lost"]


def test_tip_without_depth_of_its_own_is_not_placed(script):
    flat = {
        "rgb": None,
        "depth": np.full((100, 100), 0.5),
        "intrinsics": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
        "pose": np.eye(4),
    }
    assert script._deproject(flat, 50.0, 50.0) is None
    near = dict(flat, depth=Arm._depth())
    assert script._deproject(near, 50.0, 50.0) == pytest.approx([0.0, 0.0, 0.3])
