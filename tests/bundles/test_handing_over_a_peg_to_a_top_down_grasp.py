"""handing-over-a-peg-to-a-top-down-grasp: FakeContext tests of the vertical handover, its pre-checks and routing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "handing-over-a-peg-to-a-top-down-grasp"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
PEG = {"length_m": 0.10, "diameter_m": 0.010, "axis_in_object": [0, 0, 1], "tip_end": "-z"}
IDENTITY = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["hand_over_peg"].module


def _quat(r: np.ndarray) -> dict[str, float]:
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    if w > 1e-6:
        x, y, z = (
            (r[2, 1] - r[1, 2]) / (4 * w),
            (r[0, 2] - r[2, 0]) / (4 * w),
            (r[1, 0] - r[0, 1]) / (4 * w),
        )
    else:
        x = math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2.0
        y = (
            (r[0, 1] + r[1, 0]) / (4 * x)
            if x > 1e-6
            else math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2
        )
        z = (r[0, 2] + r[2, 0]) / (4 * x) if x > 1e-6 else (r[1, 2] + r[2, 1]) / (4 * y)
    return {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}


def _mat(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (q[k] for k in "wxyz")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def frame(approach, heading_deg) -> dict[str, float]:
    """A hand whose tool +z is its approach and +y its jaw axis."""
    a = np.asarray(approach, float) / np.linalg.norm(approach)
    h = math.radians(heading_deg)
    c = np.array([math.cos(h), math.sin(h), 0.0])
    c = c - np.dot(c, a) * a
    c /= np.linalg.norm(c)
    return _quat(np.column_stack([np.cross(c, a), c, a]))


LEFT_ROT = frame([1.0, 0.0, 0.0], 90.0)  # side grasp: approach +x, jaws along y


class Arms:
    """Arm 0 holds a 10 x 100 mm peg upright, its tool centre 15 mm below the peg's
    centre; arm 1 waits to the right. Moves land where commanded unless ``refuse``
    says why not."""

    def __init__(self, refuse=None, receiver_close=0.11, giver_grip_up=-0.015, roll_lag_deg=0.0):
        self.p = {0: np.array([0.45, 0.0, 0.20]), 1: np.array([0.45, -0.30, 0.35])}
        self.rot = {0: LEFT_ROT, 1: IDENTITY}
        self.grip = {0: 0.11, 1: 1.0}
        self.q = {0: [0.0] * 6, 1: [0.0] * 6}
        self.refuse = refuse or (lambda tool, a, pose: None)
        self.receiver_close = receiver_close
        self.roll_lag_deg = roll_lag_deg
        self.giver_opens = True
        self.on_giver_open = None
        self.on_giver_retreat = None
        self.log: list = []
        # The peg's centre in the giver's tool frame (tool +z = approach = world +x).
        r = _mat(LEFT_ROT)
        offset = r.T @ np.array([0.0, 0.0, -giver_grip_up])
        self.left_grip = {
            "T_base_tcp": {
                "position": dict(zip("xyz", self.p[0].tolist(), strict=False)),
                "rotation": LEFT_ROT,
            },
            "T_TCP_object": {
                "position": dict(zip("xyz", offset.tolist(), strict=False)),
                "rotation": _quat(r.T),
            },
            "exposed_len_m": 0.05 + giver_grip_up - 0.00775,
        }

    def _move(self, tool):
        def go(pose, arm_id=0, **kw):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.log.append((tool, arm_id, target.copy(), pose["rotation"]))
            why = self.refuse(tool, arm_id, pose)
            if why == "raise":
                raise ToolError(tool, "unreachable")
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            self.p[arm_id] = target
            rot = pose["rotation"]
            if arm_id == 1 and self.roll_lag_deg and tool == "robot.go_to_pose":
                h = math.radians(self.roll_lag_deg) / 2
                rot = _quat(
                    _mat(rot) @ _mat({"w": math.cos(h), "x": 0.0, "y": 0.0, "z": math.sin(h)})
                )
            self.rot[arm_id] = rot
            if arm_id == 0 and tool == "robot.go_to_pose_cartesian" and self.on_giver_retreat:
                self.on_giver_retreat(self)
            return {"reached": True, "outcome": "reached"}

        return go

    def ctx(self):
        def pose(arm_id=0):
            return {
                "pose": {
                    "position": dict(zip("xyz", map(float, self.p[arm_id]), strict=False)),
                    "rotation": self.rot[arm_id],
                }
            }

        def open_gripper(arm_id=0, **kw):
            self.log.append(("open", arm_id))
            if arm_id == 0 and not self.giver_opens:
                return {"position": self.grip[0]}
            self.grip[arm_id] = 1.0
            if arm_id == 0 and self.on_giver_open:
                self.on_giver_open(self)
            return {"position": 1.0}

        def close_gripper(arm_id=0, **kw):
            self.log.append(("close", arm_id))
            self.grip[arm_id] = self.receiver_close if arm_id == 1 else 0.11
            return {"position": self.grip[arm_id]}

        def grasp_frame(approach=None, close_heading_deg=0.0, arm_id=0):
            a = [approach[k] for k in "xyz"] if approach else [0.0, 0.0, -1.0]
            return {
                "rotation": frame(a, close_heading_deg),
                "rotation_flipped": frame(a, close_heading_deg + 180),
            }

        def move_to_joints(joint_config, arm_id=0, **kw):
            self.log.append(("joints", arm_id, list(joint_config["positions"])))
            turn = joint_config["positions"][-1] - self.q[arm_id][-1]
            self.q[arm_id] = list(joint_config["positions"])
            h = turn / 2
            self.rot[arm_id] = _quat(
                _mat(self.rot[arm_id])
                @ _mat({"w": math.cos(h), "x": 0.0, "y": 0.0, "z": math.sin(h)})
            )
            self.roll_lag_deg = 0.0  # the solver now holds the turned roll
            return None

        return FakeContext(
            {
                "robot.get_ee_pose": pose,
                "robot.get_gripper": lambda arm_id=0: {"position": self.grip[arm_id]},
                "robot.go_to_pose": self._move("robot.go_to_pose"),
                "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
                "robot.open_gripper": open_gripper,
                "robot.close_gripper": close_gripper,
                "robot.grasp_frame": grasp_frame,
                "robot.move_to_joints": move_to_joints,
                "robot.get_observation": lambda: {
                    "arms": [{"joint_state": {"positions": self.q[a]}} for a in (0, 1)]
                },
            }
        )

    def moves(self, arm_id=None):
        return [
            m for m in self.log if m[0].startswith("robot.") and (arm_id is None or m[1] == arm_id)
        ]


def run(script, arms, **kw):
    return script.run(arms.ctx(), peg=PEG, left_grip=arms.left_grip, **kw)


def test_happy_path_geometry_order_and_held(script):
    arms = Arms()
    out = run(script, arms)
    assert out["route"] == "handed_over", out["detail"]
    # Peg centre at z 0.215, top 0.265; pads' top 0.20775; exposed 57.25 mm -> bite 12 mm.
    assert out["peg_top"]["z"] == pytest.approx(0.265)
    assert out["bite_m"] == pytest.approx(0.012)
    recv = arms.moves(1)
    np.testing.assert_allclose(recv[0][2], [0.45, 0.0, 0.253 + 0.08], atol=1e-9)
    np.testing.assert_allclose(recv[1][2], [0.45, 0.0, 0.253], atol=1e-9)
    # Receiver's jaws along the giver's (world y): tool +y is +-y.
    assert abs(_mat(recv[0][3])[:, 1] @ [0, 1, 0]) == pytest.approx(1.0)
    # Giver backs out horizontally against its approach (+x).
    giver = arms.moves(0)
    np.testing.assert_allclose(giver[0][2], [0.37, 0.0, 0.20], atol=1e-9)
    # Receiver lifts 20 mm.
    np.testing.assert_allclose(recv[-1][2], [0.45, 0.0, 0.273], atol=1e-9)
    order = [m[:2] for m in arms.log if m[0] in ("open", "close")]
    assert order == [("open", 1), ("close", 1), ("open", 0)]
    held = out["held"]
    assert held["arm"] == "right" and held["arm_id"] == 1
    assert held["tip_offset_m"] == pytest.approx(0.253 - 0.165)
    assert held["axis_tilt_deg"] == pytest.approx(0.0, abs=1e-6)
    assert held["sigma_mm"] == pytest.approx(math.sqrt(4 + 2), rel=1e-6)
    p = held["T_TCP_object"]["position"]
    # Peg centre is 38 mm below the receiver's tool centre, along its approach (+z).
    assert [p["x"], p["y"], p["z"]] == pytest.approx([0.0, 0.0, 0.038], abs=1e-9)


def test_giver_axes_from_grasp_frame_not_from_tool_axes_order(script):
    # Same grasp, tool frame rolled half a turn about its approach: still jaws along y, back along -x.
    arms = Arms()
    arms.rot[0] = frame([1.0, 0.0, 0.0], -90.0)
    out = run(script, arms)
    assert out["route"] == "handed_over", out["detail"]
    np.testing.assert_allclose(arms.moves(0)[0][2], [0.37, 0.0, 0.20], atol=1e-9)


def test_short_exposed_end_is_a_collision_risk_before_anything_moves(script):
    arms = Arms(giver_grip_up=0.03)  # pads' top 12.25 mm under the top
    out = run(script, arms)
    assert out["route"] == "collision_risk"
    assert not arms.moves() and ("open", 1) not in arms.log


def test_exposed_len_from_left_grip_caps_the_bite(script):
    arms = Arms()
    arms.left_grip["exposed_len_m"] = 0.020
    out = run(script, arms)
    assert out["route"] == "handed_over"
    assert out["bite_m"] == pytest.approx(0.020 - 0.0102 - 0.005)


def test_tilted_peg_is_unreachable(script):
    arms = Arms()
    tilt = math.radians(25) / 2
    arms.rot[0] = _quat(
        _mat({"w": math.cos(tilt), "x": math.sin(tilt), "y": 0.0, "z": 0.0}) @ _mat(LEFT_ROT)
    )
    out = run(script, arms)
    assert out["route"] == "unreachable" and "leans" in out["detail"]
    assert not arms.moves()


def test_refuses_one_arm_twice_or_a_missing_second_arm(script):
    out = run(script, Arms(), receiver_arm=0)
    assert out["route"] == "unreachable" and "both arm 0" in out["detail"]
    arms = Arms()
    arms.p[1] = arms.p[0].copy()
    out = run(script, arms)
    assert out["route"] == "unreachable" and "separate arm" in out["detail"]
    assert not arms.moves()


def test_empty_giver_is_dropped(script):
    arms = Arms()
    arms.grip[0] = 0.01
    assert run(script, arms)["route"] == "dropped"
    assert not arms.moves()


@pytest.mark.parametrize("how", ["raise", "unreachable"])
def test_receiver_cannot_reach_over_the_peg(script, how):
    arms = Arms(refuse=lambda tool, a, pose: how if a == 1 else None)
    out = run(script, arms)
    assert out["route"] == "unreachable" and "receiver over the peg" in out["detail"]
    assert arms.grip[0] == 0.11  # giver still holds


def test_receiver_tries_the_flipped_wrist(script):
    first = {}

    def refuse(tool, a, pose):
        if a == 1 and tool == "robot.go_to_pose" and not first:
            first["rot"] = pose["rotation"]
            return "no IK"
        return None

    arms = Arms(refuse=refuse)
    out = run(script, arms)
    assert out["route"] == "handed_over"
    assert arms.moves(1)[1][3] != first["rot"]


def test_receiver_stopped_short_backs_off(script):
    arms = Arms(
        refuse=lambda tool, a, pose: (
            "contact"
            if a == 1 and tool.endswith("cartesian") and pose["position"]["z"] < 0.3
            else None
        )
    )
    out = run(script, arms)
    assert out["route"] == "receiver_missed"
    assert arms.grip[0] == 0.11 and arms.grip[1] == 1.0


def test_receiver_closed_on_nothing(script):
    arms = Arms(receiver_close=0.01)
    out = run(script, arms)
    assert out["route"] == "receiver_missed" and "stop" in out["detail"]
    assert ("open", 0) not in arms.log
    np.testing.assert_allclose(arms.p[1], [0.45, 0.0, 0.333], atol=1e-9)


def test_receiver_closed_wider_than_the_peg(script):
    arms = Arms(receiver_close=0.30)  # 28 mm on a 10 mm peg
    out = run(script, arms)
    assert out["route"] == "receiver_missed" and "not across the shaft" in out["detail"]
    assert ("open", 0) not in arms.log


def test_drop_when_the_giver_lets_go(script):
    arms = Arms()

    def lose(a):
        a.grip[1] = 0.0

    arms.on_giver_open = lose
    out = run(script, arms)
    assert out["route"] == "dropped" and "let go" in out["detail"]


def test_drop_during_the_back_out(script):
    arms = Arms()

    def lose(a):
        a.grip[1] = 0.0

    arms.on_giver_retreat = lose
    out = run(script, arms)
    assert out["route"] == "dropped" and "backed out" in out["detail"]


def test_giver_does_not_open(script):
    arms = Arms()
    arms.giver_opens = False
    out = run(script, arms)
    assert out["route"] == "unreachable" and "both hands" in out["detail"]


def test_giver_cannot_back_out(script):
    arms = Arms(refuse=lambda tool, a, pose: "raise" if a == 0 else None)
    out = run(script, arms)
    assert out["route"] == "unreachable" and "back out" in out["detail"]


def test_roll_lag_is_finished_on_the_last_joint(script):
    arms = Arms(roll_lag_deg=40.0)
    out = run(script, arms)
    assert out["route"] == "handed_over", out["detail"]
    assert any(m[0] == "joints" for m in arms.log)


def test_handover_position_moves_the_giver_first(script):
    arms = Arms()
    out = run(script, arms, handover_position={"x": 0.40, "y": 0.05, "z": 0.25})
    assert out["route"] == "handed_over"
    np.testing.assert_allclose(arms.moves(0)[0][2], [0.40, 0.05, 0.235], atol=1e-9)
    assert out["peg_top"]["z"] == pytest.approx(0.30)


def test_observed_tip_sets_the_tip_offset(script):
    arms = Arms()
    out = run(script, arms, observed_tip={"x": 0.45, "y": 0.0, "z": 0.180}, observed_sigma_mm=1.0)
    assert out["route"] == "handed_over"
    # After the 20 mm lift the receiver's tool centre is at 0.273.
    assert out["held"]["tip_offset_m"] == pytest.approx(0.273 - 0.180)
    assert out["held"]["sigma_mm"] == pytest.approx(1.0)


def test_off_centre_peg_is_centred_across_the_receivers_jaws(script):
    arms = Arms()
    # The peg sits 2 mm along y from where the giver thinks: the receiver's jaws centre it.
    t = arms.left_grip["T_TCP_object"]["position"]
    r = _mat(LEFT_ROT)
    shift = r.T @ np.array([0.0, 0.002, 0.0])
    arms.left_grip["T_TCP_object"]["position"] = {
        k: t[k] + float(s) for k, s in zip("xyz", shift, strict=False)
    }
    out = run(script, arms)
    p = out["held"]["T_TCP_object"]["position"]
    assert [p["x"], p["y"], p["z"]] == pytest.approx([0.0, 0.0, 0.038], abs=1e-9)


def test_example_declares_every_route(script):
    example = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    text = (SKILL_DIR / "scripts" / "hand_over_peg.py").read_text()
    import re

    used = set(re.findall(r'finish\(\s*"([a-z_]+)"', text)) | set(
        re.findall(r'back_off\(\s*"([a-z_]+)"', text)
    )
    used.add("handed_over")
    assert used <= declared, used - declared
