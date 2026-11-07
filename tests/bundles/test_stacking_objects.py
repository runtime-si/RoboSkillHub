"""stacking-objects: FakeContext tests of the goal geometry, the turn to the target's faces (held by the
pose or turned on the last joint), the contact verdicts and the routing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "stacking-objects"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
START = (0.40, 0.10, 0.10)
# A 30 mm cube held with its centre 2 mm along +x and 7 mm under the tool centre:
# its bottom hangs 22 mm under the hand.
HANG = 0.022
HELD_OFF = np.array([0.002, 0.0])
# The target: a 30 mm cube on the table at (0.50, 0.0), its top at 30 mm.
TOP = 0.03
TARGET_XY = np.array([0.50, 0.0])
EXPECTED = TOP + HANG
APPROACH_Z = EXPECTED + 0.08
WINDOW_TOP = EXPECTED + 0.012
BOTTOM = (
    TOP + 0.012
)  # 20 mm under the expected contact would be 32 mm; the clearance holds it at 42 mm


def _qz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def _mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _q(arr) -> dict[str, float]:
    return dict(zip("wxyz", map(float, arr), strict=True))


def _down(yaw_deg: float = 0.0) -> dict[str, float]:
    """A hand pointing straight down, its x axis at this heading."""
    return _q(_mul(_qz(yaw_deg), np.array([0.0, 1.0, 0.0, 0.0])))


def _yaw_of(rot: dict[str, float]) -> float:
    w, x, y, z = (rot[k] for k in "wxyz")
    return math.degrees(math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))


def _box(xy, z, yaw_deg=0.0, half=0.015, orientation=None):
    return {
        "center": {"x": float(xy[0]), "y": float(xy[1]), "z": float(z)},
        "extent": {"x": half, "y": half, "z": half},
        "orientation": orientation or _q(_qz(yaw_deg)),
    }


def _held(yaw_deg=0.0, at=START):
    return _box(np.array(at[:2]) + HELD_OFF, at[2] - 0.007, yaw_deg)


TARGET = _box(TARGET_XY, 0.015)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["stack_on_object"].module


class Arm:
    """Moves land where commanded. ``free_roll``: a pose move keeps the hand's
    heading (a solver that holds only the approach axis); the last joint turns
    the hand by ``roll_sign`` x its turn, and moves the tool centre by
    ``roll_shift`` [m per rad] when it does. A contact move stops where the held
    object's bottom meets ``surface`` (routed ``contact``, or raised with
    ``raise_contact``), shifted by ``push`` in xy. ``refuse`` (tool, kwargs) ->
    reason refuses a move, routed."""

    def __init__(
        self,
        start=START,
        surface=TOP,
        free_roll=False,
        roll_sign=-1.0,
        roll_shift=0.0,
        q_last=0.0,
        joints=True,
        raise_contact=False,
        push=(0.0, 0.0),
        refuse=None,
    ):
        self.p = np.array(start, dtype=float)
        self.yaw = 0.0
        self.surface = surface
        self.free_roll, self.roll_sign, self.roll_shift = free_roll, roll_sign, roll_shift
        self.q = np.array([0.0, 1.0, 1.0, 0.0, 0.0, q_last])
        self.joints = joints
        self.raise_contact = raise_contact
        self.push = np.array(push, dtype=float)
        self.refuse = refuse or (lambda tool, kw: None)
        self.moves: list[tuple[str, np.ndarray, float | None]] = []

    def _move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy(), _yaw_of(pose["rotation"])))
            why = self.refuse(tool, {"pose": pose, **kwargs})
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            if not self.free_roll:
                self.yaw = _yaw_of(pose["rotation"])
            if tool == "robot.move_cartesian_until_contact":
                contact = self.surface + HANG
                if target[2] < contact <= self.p[2]:
                    self.p = np.array([target[0], target[1], contact])
                    self.p[:2] += self.push
                    if self.raise_contact:
                        raise RuntimeError("the arm did not reach its target")
                    return {"reached": False, "outcome": "contact"}
            self.p = target
            return {"reached": True, "outcome": "reached"}

        return go

    def move_to_joints(self, joint_config, **kwargs):
        q = np.asarray(joint_config["positions"], dtype=float)
        self.moves.append(("robot.move_to_joints", q.copy(), None))
        why = self.refuse("robot.move_to_joints", {"joint_config": joint_config, **kwargs})
        if why:
            return {"reached": False, "outcome": "unreachable", "reason": why}
        turn = q[-1] - self.q[-1]
        self.q = q
        self.yaw += self.roll_sign * math.degrees(turn)
        self.p[0] += self.roll_shift * turn
        return {"reached": True}

    def ctx(self, ik=None):
        tools = {
            "robot.get_ee_pose": lambda **kw: {
                "pose": {
                    "position": dict(zip("xyz", map(float, self.p), strict=True)),
                    "rotation": _down(self.yaw),
                }
            },
            "robot.get_observation": lambda **kw: {
                "arms": [{"joint_state": {"positions": self.q.tolist()}}] if self.joints else []
            },
            "robot.go_to_pose": self._move("robot.go_to_pose"),
            "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
            "robot.move_cartesian_until_contact": self._move("robot.move_cartesian_until_contact"),
            "robot.move_to_joints": self.move_to_joints,
            "robot.solve_ik": ik
            if ik is not None
            else {"success": True, "within_servo_tolerance": True},
            "robot.open_gripper": {"position": 1.0},
        }
        return FakeContext(tools)

    def named(self, tool):
        return [m for m in self.moves if m[0] == tool]


def _rotated(v, deg):
    a = math.radians(deg)
    return np.array(
        [math.cos(a) * v[0] - math.sin(a) * v[1], math.sin(a) * v[0] + math.cos(a) * v[1]]
    )


def test_stacks_centred_turned_to_the_target_faces(script):
    arm = Arm()
    ctx = arm.ctx()
    target = _box(TARGET_XY, 0.015, yaw_deg=20.0)
    out = script.run(ctx, target_obb=target, held_obb=_held(), table_z=0.0)
    assert out["route"] == "stacked", out["detail"]
    assert out["released"] is True
    assert [m[0] for m in arm.moves] == [
        "robot.go_to_pose_cartesian",  # rise: the hand starts under the approach height
        "robot.go_to_pose",  # approach, turned
        "robot.move_cartesian_until_contact",  # to the top of the contact window
        "robot.move_cartesian_until_contact",  # through it
        "robot.go_to_pose_cartesian",  # retreat
    ]
    rise, approach, top, bottom, retreat = arm.moves
    np.testing.assert_allclose(rise[1], [START[0], START[1], APPROACH_Z], atol=1e-9)
    assert rise[2] == pytest.approx(0.0, abs=1e-9)  # the rise keeps the rotation
    hand_xy = TARGET_XY - _rotated(HELD_OFF, 20.0)
    np.testing.assert_allclose(approach[1], [*hand_xy, APPROACH_Z], atol=1e-9)
    assert approach[2] == pytest.approx(20.0)
    np.testing.assert_allclose(top[1], [*hand_xy, WINDOW_TOP], atol=1e-9)
    np.testing.assert_allclose(bottom[1], [*hand_xy, BOTTOM], atol=1e-9)
    np.testing.assert_allclose(retreat[1], [*hand_xy, EXPECTED + 0.08], atol=1e-9)
    assert out["yaw_turn_deg"] == pytest.approx(20.0)
    assert out["yaw_error_deg"] == pytest.approx(0.0, abs=1e-9)
    assert out["contact_offset"] == pytest.approx(0.0, abs=1e-9)
    placed = out["stacked_position"]
    assert [placed[k] for k in "xyz"] == pytest.approx([0.50, 0.0, TOP], abs=1e-9)
    assert ctx.call_count("robot.move_to_joints") == 0  # a full-pose arm arrives turned
    calls = [c.tool for c in ctx.calls]
    assert calls.index("robot.solve_ik") < calls.index("robot.go_to_pose")
    assert calls.index("robot.open_gripper") < len(calls) - 1


def test_the_turn_is_the_smallest_one_to_any_face(script):
    assert script.yaw_turns(0.0, 20.0, 90.0) == pytest.approx([20.0, -70.0, 110.0])
    assert script.yaw_turns(10.0, 100.0, 90.0)[0] == pytest.approx(0.0)
    assert script.yaw_turns(0.0, 80.0, 90.0)[0] == pytest.approx(-10.0)
    assert script.yaw_turns(0.0, 45.0, 90.0)[0] == pytest.approx(-45.0)
    assert script.yaw_turns(0.0, 100.0, 180.0)[0] == pytest.approx(-80.0)
    assert script.yaw_turns(0.0, 100.0, 360.0) == pytest.approx([100.0])


def test_a_real_station_raising_on_contact_is_a_stop(script):
    arm = Arm(raise_contact=True)
    out = script.run(arm.ctx(), target_obb=TARGET, held_obb=_held())
    assert out["route"] == "stacked", out["detail"]
    assert out["contact_z"] == pytest.approx(EXPECTED)


@pytest.mark.parametrize("roll_sign", [-1.0, 1.0])
def test_a_free_wrist_is_turned_on_its_last_joint(script, roll_sign):
    # The pose moves keep the hand's heading; the last joint turns it (either
    # way round: the wrong guess is learned from the first turn).
    arm = Arm(free_roll=True, roll_sign=roll_sign)
    ctx = arm.ctx()
    out = script.run(ctx, target_obb=_box(TARGET_XY, 0.015, yaw_deg=30.0), held_obb=_held())
    assert out["route"] == "stacked", out["detail"]
    assert arm.yaw == pytest.approx(30.0, abs=1e-6)
    assert out["yaw_turn_deg"] == pytest.approx(30.0, abs=1e-6)
    assert out["yaw_error_deg"] == pytest.approx(0.0, abs=1e-6)
    assert ctx.call_count("robot.move_to_joints") == (1 if roll_sign < 0 else 2)
    hand_xy = TARGET_XY - _rotated(HELD_OFF, 30.0)
    np.testing.assert_allclose(
        arm.named("robot.move_cartesian_until_contact")[0][1][:2], hand_xy, atol=1e-9
    )


def test_the_hand_is_re_centred_after_a_turn_that_moves_the_tool_centre(script):
    # A tool centre off the last joint's axis moves when it turns.
    arm = Arm(free_roll=True, roll_shift=0.04)
    out = script.run(arm.ctx(), target_obb=_box(TARGET_XY, 0.015, yaw_deg=30.0), held_obb=_held())
    assert out["route"] == "stacked", out["detail"]
    hand_xy = TARGET_XY - _rotated(HELD_OFF, 30.0)
    recentre = arm.named("robot.go_to_pose")[-1]
    np.testing.assert_allclose(recentre[1], [*hand_xy, APPROACH_Z], atol=1e-9)
    placed = out["stacked_position"]
    assert [placed[k] for k in "xy"] == pytest.approx(TARGET_XY.tolist(), abs=1e-9)


def test_a_joint_at_its_limit_turns_the_equivalent_way(script):
    # +30 deg needs the joint at -30 deg (sign -1), past -120 deg from -100 deg;
    # -60 deg (joint +60 deg) lines the faces up as well.
    arm = Arm(free_roll=True, q_last=math.radians(-100.0))
    out = script.run(arm.ctx(), target_obb=_box(TARGET_XY, 0.015, yaw_deg=30.0), held_obb=_held())
    assert out["route"] == "stacked", out["detail"]
    assert arm.yaw == pytest.approx(-60.0, abs=1e-6)
    assert out["yaw_error_deg"] == pytest.approx(0.0, abs=1e-6)


def test_no_joint_reading_still_stacks_and_says_the_faces_are_off(script):
    arm = Arm(free_roll=True, joints=False)
    out = script.run(arm.ctx(), target_obb=_box(TARGET_XY, 0.015, yaw_deg=30.0), held_obb=_held())
    assert out["route"] == "stacked", out["detail"]
    assert out["yaw_error_deg"] == pytest.approx(30.0)
    assert "no joint reading" in out["detail"]


def test_a_refused_wrist_turn_still_stacks_and_says_so(script):
    arm = Arm(
        free_roll=True, refuse=lambda tool, kw: "limit" if tool == "robot.move_to_joints" else None
    )
    out = script.run(arm.ctx(), target_obb=_box(TARGET_XY, 0.015, yaw_deg=30.0), held_obb=_held())
    assert out["route"] == "stacked", out["detail"]
    assert "wrist turn: unreachable: limit" in out["detail"]


def test_align_yaw_false_keeps_the_rotation(script):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(
        ctx, target_obb=_box(TARGET_XY, 0.015, yaw_deg=30.0), held_obb=_held(), align_yaw=False
    )
    assert out["route"] == "stacked", out["detail"]
    assert all(m[2] == pytest.approx(0.0, abs=1e-9) for m in arm.moves)
    assert out["yaw_turn_deg"] == pytest.approx(0.0, abs=1e-9)


def test_contact_above_the_window_is_misaligned_and_kept(script):
    # Something 30 mm taller than the target under the held object.
    arm = Arm(surface=TOP + 0.03)
    ctx = arm.ctx()
    out = script.run(ctx, target_obb=TARGET, held_obb=_held())
    assert out["route"] == "misaligned", out["detail"]
    assert out["released"] is False
    assert ctx.call_count("robot.open_gripper") == 0
    assert out["contact_offset"] == pytest.approx(0.03)
    tool, lift, _ = arm.moves[-1]
    assert tool == "robot.go_to_pose_cartesian"
    assert lift[2] == pytest.approx(APPROACH_Z)
    assert "still holding" in out["detail"]


def test_missing_the_target_top_is_misaligned_and_kept(script):
    # The held object passes beside the target: the next thing under it is the table.
    arm = Arm(surface=0.0)
    ctx = arm.ctx()
    out = script.run(ctx, target_obb=TARGET, held_obb=_held(), table_z=0.0)
    assert out["route"] == "misaligned", out["detail"]
    assert "without stopping" in out["detail"]
    assert ctx.call_count("robot.open_gripper") == 0
    assert out["contact_z"] == pytest.approx(BOTTOM)


def test_a_stop_pushed_sideways_is_misaligned(script):
    arm = Arm(push=(0.0, 0.012))
    ctx = arm.ctx()
    out = script.run(ctx, target_obb=TARGET, held_obb=_held())
    assert out["route"] == "misaligned", out["detail"]
    assert "12 mm sideways" in out["detail"]
    assert ctx.call_count("robot.open_gripper") == 0


def test_a_small_sideways_stop_is_stacked(script):
    arm = Arm(push=(0.0, 0.004))
    out = script.run(arm.ctx(), target_obb=TARGET, held_obb=_held())
    assert out["route"] == "stacked", out["detail"]
    assert out["stacked_position"]["y"] == pytest.approx(0.004)


def test_a_tilted_target_is_blocked_before_moving(script):
    a = math.radians(15.0)
    tilted = _box(
        TARGET_XY,
        0.015,
        orientation={"w": math.cos(a / 2), "x": math.sin(a / 2), "y": 0.0, "z": 0.0},
    )
    arm = Arm()
    out = script.run(arm.ctx(), target_obb=tilted, held_obb=_held())
    assert out["route"] == "blocked"
    assert "tilted 15 deg" in out["detail"]
    assert arm.moves == []


def test_the_ik_preview_picks_the_first_reachable_turn(script):
    previews = []

    def ik(pose, **kw):
        previews.append(_yaw_of(pose["rotation"]))
        return {"success": True, "within_servo_tolerance": len(previews) > 1}

    arm = Arm()
    out = script.run(
        arm.ctx(ik=ik), target_obb=_box(TARGET_XY, 0.015, yaw_deg=20.0), held_obb=_held()
    )
    assert out["route"] == "stacked", out["detail"]
    assert previews == pytest.approx([20.0, -70.0])
    assert out["yaw_turn_deg"] == pytest.approx(-70.0)
    assert out["yaw_error_deg"] == pytest.approx(0.0, abs=1e-9)


def test_no_reachable_turn_is_blocked(script):
    arm = Arm()
    out = script.run(
        arm.ctx(ik={"success": True, "within_servo_tolerance": False}),
        target_obb=_box(TARGET_XY, 0.015, yaw_deg=20.0),
        held_obb=_held(),
    )
    assert out["route"] == "blocked"
    assert "+20, -70, +110" in out["detail"]
    assert arm.moves == []


def test_no_ik_preview_gets_the_benefit_of_the_doubt(script):
    def missing(**kw):
        raise ToolError("robot.solve_ik", "not served")

    arm = Arm()
    out = script.run(arm.ctx(ik=missing), target_obb=TARGET, held_obb=_held())
    assert out["route"] == "stacked"


@pytest.mark.parametrize("raised", [False, True])
def test_a_refused_approach_is_blocked(script, raised):
    def refuse(tool, kw):
        if tool == "robot.go_to_pose":
            if raised:
                raise RuntimeError("IK failed")
            return "out of reach"
        return None

    arm = Arm(refuse=refuse)
    ctx = arm.ctx()
    out = script.run(ctx, target_obb=TARGET, held_obb=_held())
    assert out["route"] == "blocked"
    assert out["detail"].startswith("approach:")
    assert ctx.call_count("robot.move_cartesian_until_contact") == 0


def test_a_refused_descent_is_blocked_not_read_as_contact(script):
    def refuse(tool, kw):
        if tool == "robot.move_cartesian_until_contact":
            return "IK failed"
        return None

    arm = Arm(refuse=refuse)
    ctx = arm.ctx()
    out = script.run(ctx, target_obb=TARGET, held_obb=_held())
    assert out["route"] == "blocked", out["detail"]
    assert ctx.call_count("robot.open_gripper") == 0


def test_a_failed_retreat_after_release_is_blocked_but_released(script):
    retreat = {"n": 0}

    def refuse(tool, kw):
        if tool == "robot.go_to_pose_cartesian":
            retreat["n"] += 1
            return "joint limit" if retreat["n"] > 1 else None  # the rise passes
        return None

    arm = Arm(refuse=refuse)
    out = script.run(arm.ctx(), target_obb=TARGET, held_obb=_held())
    assert out["route"] == "blocked"
    assert out["released"] is True
    assert out["detail"].startswith("released, but the retreat")


def test_the_table_floor_holds_over_a_low_target(script):
    # A 5 mm tile with its top at 9 mm, on a table at 8 mm: 20 mm under the
    # expected contact is 11 mm, the clearance over the tile 21 mm, the table
    # floor 23 mm.
    tile = {**_box(TARGET_XY, 0.0065), "extent": {"x": 0.015, "y": 0.015, "z": 0.0025}}
    arm = Arm(surface=-0.01)  # nothing there after all
    out = script.run(arm.ctx(), target_obb=tile, held_obb=_held(), table_z=0.008)
    assert out["route"] == "misaligned"
    assert arm.named("robot.move_cartesian_until_contact")[-1][1][2] == pytest.approx(0.023)


def test_arm_id_is_passed_only_when_set(script):
    arm = Arm(free_roll=True)
    ctx = arm.ctx()
    script.run(ctx, target_obb=_box(TARGET_XY, 0.015, yaw_deg=30.0), held_obb=_held(), arm_id=1)
    robot_calls = [
        c for c in ctx.calls if c.tool.startswith("robot.") and c.tool != "robot.get_observation"
    ]
    assert all(c.kwargs.get("arm_id") == 1 for c in robot_calls)


def test_a_box_on_its_side_reads_its_top_and_faces(script):
    # Turned 90 deg about x: its y extent stands vertical; then 30 deg about z.
    q = _mul(_qz(30.0), np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]))
    box = {
        "center": {"x": 0.0, "y": 0.0, "z": 0.02},
        "extent": {"x": 0.03, "y": 0.02, "z": 0.01},
        "orientation": _q(q),
    }
    f = script.box_frame(box)
    assert f["top"] == pytest.approx(0.04)
    assert f["bottom"] == pytest.approx(0.0)
    assert f["tilt"] == pytest.approx(0.0, abs=1e-6)
    assert (f["yaw"] - 30.0 + 45.0) % 90.0 - 45.0 == pytest.approx(0.0, abs=1e-6)


def test_the_example_declares_every_route():
    example = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == {"stacked", "misaligned", "blocked"}
