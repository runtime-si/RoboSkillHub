"""inserting-mid-air-between-two-held-parts: FakeContext tests of the rendezvous, the alignment,
the stepped insertion with contact felt on both arms, and the routing."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "inserting-mid-air-between-two-held-parts"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
AXIS = np.array([0.0, 1.0, 0.0])
# Holder (arm 0) holds a socket whose mouth is 20 mm toward -y from its tool centre; the mover
# (arm 1) holds a peg whose tip is 45 mm toward +y from its tool centre. Both top-down.
HOLDER_AT = np.array([0.50, 0.05, 0.30])
MOVER_AT = np.array([0.48, -0.20, 0.32])
ENTRY = HOLDER_AT + np.array([0.0, -0.02, 0.0])
TIP = MOVER_AT + np.array([0.0, 0.045, 0.0])


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["insert_mid_air"].module


def _vec(v):
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _down_rolled(deg: float) -> dict[str, float]:
    """DOWN turned by ``deg`` about its own tool z."""
    h = math.radians(deg) / 2.0
    return {"w": 0.0, "x": math.cos(h), "y": -math.sin(h), "z": 0.0}


def _rot(q):
    w, x, y, z = (q[k] for k in "wxyz")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class World:
    """Two arms holding a peg and a socket. The mover's tip stops at ``block`` past the real
    entry (``bottom``, or ``jam_at``, or the rim at 0 when its real tip is off the entry's line by
    more than ``clear / 2``); of what a step asks past that, ``holder_yield`` pushes the holder."""

    def __init__(self, **kw):
        self.p = {0: HOLDER_AT.copy(), 1: MOVER_AT.copy()}
        self.rot = {0: DOWN, 1: DOWN}
        self.grip = {0: 0.33, 1: 0.13}
        self.q = {0: [0.0] * 6, 1: [0.0] * 6}
        self.tip_in = _rot(DOWN).T @ (TIP - MOVER_AT)
        self.entry_in = _rot(DOWN).T @ (ENTRY - HOLDER_AT)
        self.true_offset = np.asarray(kw.get("true_offset", (0.0, 0.0, 0.0)), float)
        self.clear = kw.get("clear", 0.001)
        self.bottom = kw.get("bottom", 0.025)
        self.jam_at = kw.get("jam_at")
        self.holder_yield = kw.get("holder_yield", 0.0)
        self.land_error = np.asarray(kw.get("land_error", (0.0, 0.0, 0.0)), float)
        self.fine_error = np.asarray(kw.get("fine_error", (0.0, 0.0, 0.0)), float)
        self.roll_deg = kw.get("roll_deg", 0.0)
        self.refuse = kw.get("refuse", lambda tool, a, target: None)
        self.efforts = kw.get("efforts", False)
        self.mover_opens = kw.get("mover_opens", True)
        self.on_open = kw.get("on_open")
        self.slide = kw.get("slide", False)
        self.pushing = False
        self.log = []

    def _inside(self):
        d = self.real_tip() - self.real_entry()
        depth = float(d @ AXIS)
        return depth > 1e-6 and float(np.linalg.norm(d - AXIS * depth)) <= self.clear / 2 + 1e-9

    def real_tip(self):
        return self.p[1] + _rot(self.rot[1]) @ self.tip_in + self.true_offset

    def real_entry(self):
        return self.p[0] + _rot(self.rot[0]) @ self.entry_in

    def _move(self, tool):
        def go(pose, arm_id=0, **kw):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.log.append((tool, arm_id, target.copy(), dict(kw)))
            why = self.refuse(tool, arm_id, target)
            if why == "raise":
                raise ToolError(tool, "unreachable")
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            fine = "tolerance" in kw
            target = target + (self.fine_error if fine else self.land_error) * (arm_id == 1)
            rot = pose["rotation"]
            if arm_id == 1 and self.roll_deg:
                rot = _down_rolled(self.roll_deg)  # the solver keeps the roll the wrist has
            self.rot[arm_id] = rot
            self.pushing = False
            if arm_id == 1:
                start = self.p[1].copy()
                inside = self._inside()
                self.p[1] = target
                d = self.real_tip() - self.real_entry()
                depth = float(d @ AXIS)
                across = d - AXIS * depth
                lateral = float(np.linalg.norm(across))
                if inside and lateral > self.clear / 2:
                    # The hole's walls stop the part across the axis.
                    self.p[1] = self.p[1] - across * (1.0 - self.clear / 2 / lateral)
                    self.pushing = True
                    return {"reached": False, "outcome": "timeout"}
                block = self.bottom if self.jam_at is None else self.jam_at
                if lateral > self.clear / 2:
                    block = 0.0
                if depth > block and float((target - start) @ AXIS) > 0:
                    over = depth - block
                    if self.slide:  # the part slides back in the jaws, the hand goes on
                        self.true_offset = self.true_offset - AXIS * over
                        return {"reached": True, "outcome": "reached"}
                    self.p[1] = target - AXIS * over * (1.0 - self.holder_yield)
                    self.p[0] = self.p[0] + AXIS * over * self.holder_yield
                    self.pushing = True
                    return {"reached": False, "outcome": "timeout"}
            else:
                self.p[0] = target
            return {"reached": True, "outcome": "reached"}

        return go

    def ctx(self):
        return FakeContext(self.tools())

    def tools(self):
        def pose(arm_id=0):
            return {"pose": {"position": _vec(self.p[arm_id]), "rotation": self.rot[arm_id]}}

        def open_gripper(arm_id=0, **kw):
            self.log.append(("open", arm_id))
            if not self.mover_opens:
                return {"position": self.grip[arm_id]}
            self.grip[arm_id] = 1.0
            if self.on_open:
                self.on_open(self)
            return {"position": 1.0}

        def move_to_joints(joint_config, arm_id=0, **kw):
            self.log.append(("joints", arm_id, list(joint_config["positions"])))
            self.q[arm_id] = list(joint_config["positions"])
            self.rot[arm_id] = DOWN
            self.roll_deg = 0.0
            return None

        def observation():
            arms = []
            for a in (0, 1):
                js = {"positions": self.q[a]}
                if self.efforts:
                    js["efforts"] = [0.1 + (2.0 if self.pushing else 0.0)] * 6
                arms.append({"joint_state": js})
            return {"arms": arms}

        return {
            "robot.get_ee_pose": pose,
            "robot.get_gripper": lambda arm_id=0: {"position": self.grip[arm_id]},
            "robot.go_to_pose": self._move("robot.go_to_pose"),
            "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
            "robot.open_gripper": open_gripper,
            "robot.move_to_joints": move_to_joints,
            "robot.get_observation": observation,
        }

    def moves(self, arm_id):
        return [m for m in self.log if m[0].startswith("robot.") and m[1] == arm_id]


def run(script, world, **kw):
    kw.setdefault("moving_tip", _vec(TIP))
    kw.setdefault("receiving_entry", _vec(ENTRY))
    kw.setdefault("settle_period", 0.0)
    kw.setdefault("insertion_axis", _vec(AXIS))
    return script.run(world.ctx(), **kw)


def _depth(world):
    return float((world.real_tip() - world.real_entry()) @ AXIS)


def test_happy_path_aligns_and_steps_in(script):
    w = World()
    ctx = w.ctx()
    out = script.run(
        ctx,
        moving_tip=_vec(TIP),
        receiving_entry=_vec(ENTRY),
        insertion_axis=_vec(AXIS),
        settle_period=0.0,
    )
    assert out["route"] == "inserted", out["detail"]
    assert out["depth"] == pytest.approx(0.02, abs=1e-6)
    assert _depth(w) == pytest.approx(0.02, abs=1e-6)
    assert out["contact_signal"] == ""
    assert out["lateral_error"] < 1e-6
    # The holder stays where it is (no meet_position); only the mover moves.
    assert w.moves(0) == []
    mover = w.moves(1)
    # Stand-off: tip 50 mm short of the entry, on its line.
    np.testing.assert_allclose(mover[0][2] + (TIP - MOVER_AT), ENTRY - AXIS * 0.05, atol=1e-9)
    # 5 mm short, then steps of 2.5 mm to 20 mm in: 10 steps.
    assert out["steps"] == 10
    steps = [m for m in mover if m[0] == "robot.go_to_pose" and "tolerance" in m[3]]
    assert steps[0][3]["tolerance"] == pytest.approx(0.0003)
    np.testing.assert_allclose(
        [m[2][1] for m in steps[:10]], -0.02 + np.arange(1, 11) * 0.0025, atol=1e-12
    )
    assert out["probes_resisted"] == 4  # then four probes across the axis, each undone
    for call in ctx.calls:
        if call.tool.startswith("robot.") and "arm_id" in call.kwargs:
            assert call.kwargs["arm_id"] == 1  # arm_id only for the nonzero arm
    assert out["entry_position"] == _vec(ENTRY)


def test_meet_position_moves_the_holder_first(script):
    w = World()
    meet = np.array([0.45, 0.0, 0.35])
    out = run(script, w, meet_position=_vec(meet))
    assert out["route"] == "inserted", out["detail"]
    np.testing.assert_allclose(w.moves(0)[0][2], meet - (ENTRY - HOLDER_AT), atol=1e-9)
    np.testing.assert_allclose(w.real_entry(), meet, atol=1e-9)
    assert w.log.index(w.moves(0)[0]) < w.log.index(w.moves(1)[0])


def test_both_move_closes_from_both_sides(script):
    w = World()
    meet = np.array([0.45, 0.0, 0.35])
    out = run(script, w, meet_position=_vec(meet), mode="both_move")
    assert out["route"] == "inserted", out["detail"]
    holder = w.moves(0)
    np.testing.assert_allclose(holder[0][2], meet + AXIS * 0.025 - (ENTRY - HOLDER_AT), atol=1e-9)
    np.testing.assert_allclose(holder[1][2], meet - (ENTRY - HOLDER_AT), atol=1e-9)
    assert holder[1][0] == "robot.go_to_pose_cartesian"
    np.testing.assert_allclose(w.moves(1)[0][2] + (TIP - MOVER_AT), meet - AXIS * 0.025, atol=1e-9)


def test_refuses_one_arm_twice_or_a_missing_second_arm(script):
    out = run(script, World(), holder_arm=1)
    assert out["route"] == "unreachable" and "both arm 1" in out["detail"]
    w = World()
    w.p[0] = w.p[1].copy()  # a connector that clamps arm_id answers one arm twice
    out = run(script, w)
    assert out["route"] == "unreachable" and "separate arm" in out["detail"]
    assert w.moves(1) == []


@pytest.mark.parametrize("empty", [0, 1])
def test_an_empty_hand_is_slipped_before_anything_moves(script, empty):
    w = World()
    w.grip[empty] = 0.01
    out = run(script, w)
    assert out["route"] == "slipped" and "nothing moved" in out["detail"]
    assert w.moves(0) == [] and w.moves(1) == []


def test_a_landing_error_is_corrected_on_the_holders_entry(script):
    w = World(land_error=(0.004, 0.0, -0.003))  # free moves land 5 mm off, fine moves exact
    out = run(script, w)
    assert out["route"] == "inserted", out["detail"]
    assert out["lateral_error"] < 1e-6
    assert _depth(w) == pytest.approx(0.02, abs=1e-6)


def test_alignment_that_never_converges_is_misaligned(script):
    w = World(
        land_error=(0.002, 0.0, 0.0), fine_error=(0.002, 0.0, 0.0)
    )  # every move lands 2 mm across
    out = run(script, w)
    assert out["route"] == "misaligned", out["detail"]
    assert "off the entry's line" in out["detail"]
    assert out["lateral_error"] == pytest.approx(0.002, abs=1e-6)
    assert _depth(w) < -0.04  # nothing touched, the mover waits back at the stand-off


def test_a_wrong_in_hand_offset_stops_on_the_rim(script):
    w = World(true_offset=(0.0015, 0.0, 0.0))  # the registered tip is 1.5 mm off the real one
    out = run(script, w)
    assert out["route"] == "misaligned", out["detail"]
    assert out["contact_signal"] == "lag" and "rim" in out["detail"]
    np.testing.assert_allclose(w.real_tip() - w.true_offset, ENTRY - AXIS * 0.05, atol=1e-9)


def test_a_part_that_slides_in_its_jaws_is_caught_by_the_probe(script):
    w = World(true_offset=(0.0015, 0.0, 0.0), slide=True)  # on the face, sliding back in the hand
    out = run(script, w)
    assert out["route"] == "slipped", out["detail"]
    assert out["probes_resisted"] == 0 and "slid in a hand" in out["detail"]
    assert out["depth"] == pytest.approx(0.02, abs=1e-6)  # what the hands alone say
    assert _depth(w) <= 1e-9
    out = run(script, World(true_offset=(0.0015, 0.0, 0.0), slide=True), verify=False)
    assert out["route"] == "inserted"  # without the probe the slide reads as inserted


def test_a_seated_part_resists_the_probes(script):
    w = World()
    out = run(script, w)
    assert out["route"] == "inserted", out["detail"]
    assert out["probes_resisted"] == 4
    assert _depth(w) == pytest.approx(0.02, abs=1e-6)
    out = run(script, World(clear=0.01), clearance=0.01)  # a loose fit, told: probes past it
    assert out["route"] == "inserted" and out["probes_resisted"] == 4


def test_a_jam_part_way_is_stalled_and_backs_off(script):
    w = World(jam_at=0.008)
    out = run(script, w)
    assert out["route"] == "stalled", out["detail"]
    assert out["contact_signal"] == "lag"
    assert 0.007 < out["depth"] < 0.009
    assert _depth(w) == pytest.approx(0.008 - 0.003, abs=1e-6)  # backed off 3 mm
    assert "8.0 mm in" in out["detail"]


def test_a_holder_that_gives_way_is_felt(script):
    w = World(
        jam_at=0.008, holder_yield=0.8
    )  # the push moves the holder more than it lags the mover
    out = run(script, w)
    assert out["route"] == "stalled", out["detail"]
    assert "holder" in out["contact_signal"]
    assert out["holder_shift"] > 0.002


def test_contact_at_the_bottom_is_inserted(script):
    w = World(bottom=0.019)  # the bottom is 1 mm short of the asked depth: within depth_tolerance
    out = run(script, w, insert_depth=0.02)
    assert out["route"] == "inserted", out["detail"]
    assert "at the bottom" in out["detail"] or out["contact_signal"] == ""


def test_efforts_fire_when_published(script):
    w = World(jam_at=0.008, efforts=True)
    out = run(script, w, lag_threshold=0.05)  # lag off: only the efforts can see the jam
    assert out["efforts_available"] is True
    assert out["route"] == "stalled", out["detail"]
    assert out["contact_signal"] == "effort"


def test_signal_effort_without_efforts_is_unreachable(script):
    out = run(script, World(), signal="effort")
    assert out["route"] == "unreachable" and "no joint efforts" in out["detail"]


@pytest.mark.parametrize("how", ["route", "raise"])
def test_unreachable_stand_off_or_meet_point(script, how):
    w = World(
        refuse=lambda tool, a, t: (how if how == "raise" else "out of reach") if a == 1 else None
    )
    out = run(script, w)
    assert out["route"] == "unreachable" and "stand-off" in out["detail"]
    w = World(refuse=lambda tool, a, t: "out of reach" if a == 0 else None)
    out = run(script, w, meet_position=_vec([0.45, 0.0, 0.35]))
    assert out["route"] == "unreachable" and "meet point" in out["detail"]


def test_a_refused_step_is_unreachable(script):
    def refuse(tool, a, t):
        return "joint limit" if a == 1 and tool == "robot.go_to_pose" and t[1] > -0.01 else None

    out = run(script, World(refuse=refuse))
    assert out["route"] == "unreachable" and "insertion step" in out["detail"]


def test_release_by_the_mover_backs_out_up(script):
    w = World()
    out = run(script, w, release="mover")
    assert out["route"] == "inserted", out["detail"]
    assert ("open", 1) in w.log
    last = w.moves(1)[-1]
    assert last[0] == "robot.go_to_pose_cartesian"
    assert last[2][2] == pytest.approx(ENTRY[2] + 0.05)  # DOWN hand: its approach reversed is up
    assert "holder holds the assembly" in out["detail"]


def test_release_with_jaws_that_do_not_open_is_stalled(script):
    out = run(script, World(mover_opens=False), release="mover")
    assert out["route"] == "stalled" and "after opening" in out["detail"]


def test_holder_dropping_after_the_release_is_slipped(script):
    def drop(world):
        world.grip[0] = 0.0

    out = run(script, World(on_open=drop), release="mover")
    assert out["route"] == "slipped" and "after the mover let go" in out["detail"]


def test_a_free_wrist_roll_is_turned_back_on_the_last_joint(script):
    w = World(roll_deg=6.0)
    out = run(script, w)
    assert out["route"] == "inserted", out["detail"]
    assert out["roll_turns"] >= 1
    assert any(m[0] == "joints" and m[1] == 1 for m in w.log)
    assert out["lateral_error"] < 1e-6


def test_a_tilted_hand_is_misaligned(script):
    w = World(roll_deg=6.0)
    out = run(script, w, fix_roll=False)
    assert out["route"] == "misaligned" and "axes are 6.0 deg apart" in out["detail"]


def test_offsets_in_the_tool_frame_match_world_points(script):
    w = World()
    out = run(
        script,
        w,
        moving_tip=None,
        receiving_entry=None,
        moving_tip_in_tcp={"position": _vec(w.tip_in), "rotation": DOWN},
        receiving_entry_in_tcp={"position": _vec(w.entry_in), "rotation": DOWN},
    )
    assert out["route"] == "inserted", out["detail"]
    assert _depth(w) == pytest.approx(0.02, abs=1e-6)


def test_the_axis_is_normalised_or_inferred_from_tip_to_entry(script):
    w = World()
    out = run(script, w, insertion_axis={"x": 0.0, "y": 2.0, "z": 0.0}, insert_depth=0.01)
    assert out["route"] == "inserted", out["detail"]
    assert _depth(w) == pytest.approx(0.01, abs=1e-6)
    w = World()
    w.p[1] = MOVER_AT + np.array([0.02, 0.0, -0.02])  # the tip already on the entry's line
    tip = w.real_tip()
    out = run(script, w, moving_tip=_vec(tip), insertion_axis=None)
    assert out["route"] == "inserted", out["detail"]
    assert _depth(w) == pytest.approx(0.02, abs=1e-6)
    out = run(script, World(), moving_tip=_vec(ENTRY), insertion_axis=None)
    assert out["route"] == "unreachable" and "no insertion_axis" in out["detail"]


def test_bad_modes_raise(script):
    with pytest.raises(ValueError):
        run(script, World(), mode="sideways")
    with pytest.raises(ValueError):
        run(script, World(), release="holder")
    with pytest.raises(ValueError):
        run(script, World(), moving_tip=None)


def test_example_declares_every_route():
    example = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    source = (SKILL_DIR / "scripts" / "insert_mid_air.py").read_text()
    routed = set(re.findall(r'finish\(\s*"([a-z_]+)"', source))
    assert routed == declared == {"inserted", "misaligned", "stalled", "slipped", "unreachable"}


def test_a_connector_without_servo_knobs_still_steps(script):
    w = World()
    go = w._move("robot.go_to_pose")

    def strict(pose, arm_id=0, **kw):
        if kw:
            raise TypeError(f"go_to_pose() got an unexpected keyword argument {next(iter(kw))!r}")
        return go(pose, arm_id=arm_id)

    ctx = FakeContext({**w.tools(), "robot.go_to_pose": strict})
    out = script.run(
        ctx,
        moving_tip=_vec(TIP),
        receiving_entry=_vec(ENTRY),
        insertion_axis=_vec(AXIS),
        settle_period=0.0,
    )
    assert out["route"] == "inserted", out["detail"]
    assert all("tolerance" not in m[3] for m in w.moves(1))
