"""holding-steady-while-other-arm-acts: FakeContext tests of hold, act and release, their checks and routing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "holding-steady-while-other-arm-acts"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
DOWN_FLIPPED = {"w": 0.0, "x": 0.0, "y": 1.0, "z": 0.0}
SIDE = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
# A 30 mm cube on a table at z = 0, centre 15 mm up.
CUBE = {
    "center": {"x": 0.50, "y": 0.0, "z": 0.015},
    "extent": {"x": 0.015, "y": 0.015, "z": 0.015},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}


@pytest.fixture(scope="module")
def scripts(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return {k: bundle.canonical_scripts[k].module for k in ("hold", "act", "release")}


class Arms:
    """Two arms. Moves land where commanded unless ``refuse(tool, arm, pose)`` says why not, or a
    ``floor[arm]`` (a surface under the hand) stops a downward move on it, or ``wall[arm]`` (a y
    the hand cannot pass) stops a +y move there."""

    def __init__(self, refuse=None, close=0.33, ik_ok=True):
        self.p = {0: np.array([0.40, 0.25, 0.30]), 1: np.array([0.40, -0.25, 0.30])}
        self.rot = {0: DOWN, 1: SIDE}
        self.grip = {0: 1.0, 1: 1.0}
        self.close = {0: close, 1: 0.0}
        self.refuse = refuse or (lambda tool, a, pose: None)
        self.floor: dict[int, float] = {}
        self.wall: dict[int, float] = {}
        self.ik_ok = ik_ok
        self.opens = True
        self.log: list = []

    def _move(self, tool):
        def go(pose, arm_id=0, **kw):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.log.append((tool, arm_id, target.copy(), pose["rotation"]))
            why = self.refuse(tool, arm_id, pose)
            if why == "raise":
                raise ToolError(tool, "unreachable")
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            land, stopped = target.copy(), False
            if arm_id in self.floor and land[2] < self.floor[arm_id]:
                land[2], stopped = self.floor[arm_id], True
            if arm_id in self.wall and land[1] > self.wall[arm_id] > self.p[arm_id][1] - 1e-9:
                land[1], stopped = self.wall[arm_id], True
            self.p[arm_id] = land
            self.rot[arm_id] = pose["rotation"]
            return {"reached": not stopped, "outcome": "contact" if stopped else "reached"}

        return go

    def ctx(self):
        def pose(arm_id=0):
            return {
                "pose": {
                    "position": dict(zip("xyz", map(float, self.p[arm_id]), strict=True)),
                    "rotation": self.rot[arm_id],
                }
            }

        def open_gripper(arm_id=0, **kw):
            self.log.append(("open", arm_id))
            if self.opens:
                self.grip[arm_id] = 1.0
            return {"position": self.grip[arm_id]}

        def close_gripper(arm_id=0, **kw):
            self.log.append(("close", arm_id, kw.get("settle_steps")))
            self.grip[arm_id] = self.close[arm_id]
            return {"position": self.grip[arm_id]}

        def solve_ik(pose, arm_id=0, **kw):
            self.log.append(("ik", arm_id))
            return {"success": True, "within_servo_tolerance": self.ik_ok}

        return FakeContext(
            {
                "robot.get_ee_pose": pose,
                "robot.get_gripper": lambda arm_id=0: {"position": self.grip[arm_id]},
                "robot.go_to_pose": self._move("robot.go_to_pose"),
                "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
                "robot.open_gripper": open_gripper,
                "robot.close_gripper": close_gripper,
                "robot.grasp_frame": {"rotation": DOWN, "rotation_flipped": DOWN_FLIPPED},
                "robot.solve_ik": solve_ik,
            }
        )

    def moves(self, arm_id=None):
        return [
            m for m in self.log if m[0].startswith("robot.") and (arm_id is None or m[1] == arm_id)
        ]


def _held(scripts, arms, **kw):
    out = scripts["hold"].run(arms.ctx(), object_obb=CUBE, **kw)
    assert out["route"] == "holding", out["detail"]
    return out["hold"]


# ---------------------------------------------------------------- hold


def test_grip_hold_geometry_and_record(scripts):
    arms = Arms()
    ctx = arms.ctx()
    out = scripts["hold"].run(ctx, object_obb=CUBE)
    assert out["route"] == "holding", out["detail"]
    m = arms.moves(0)
    np.testing.assert_allclose(m[0][2], [0.50, 0.0, 0.065], atol=1e-9)  # 0.05 above the hold pose
    np.testing.assert_allclose(m[1][2], [0.50, 0.0, 0.015], atol=1e-9)  # tool centre at mid-height
    assert [m[0] for m in m] == ["robot.go_to_pose", "robot.go_to_pose_cartesian"]
    assert ("open", 0) in arms.log and ("close", 0, 300) in arms.log  # jaws read once settled
    assert not arms.moves(1)
    hold = out["hold"]
    assert hold["arm"] == 0 and hold["other_arm"] == 1 and hold["mode"] == "grip"
    assert hold["grip"] == pytest.approx(0.33) and out["holder_grip"] == pytest.approx(0.33)
    assert hold["pose"]["position"]["z"] == pytest.approx(0.015)
    assert hold["retreat"]["z"] == pytest.approx(0.065)
    for call in ctx.calls:  # arm_id only for a nonzero arm
        if call.tool in ("robot.go_to_pose", "robot.close_gripper", "robot.grasp_frame"):
            assert "arm_id" not in call.kwargs


def test_jaws_close_across_the_narrow_side(scripts):
    bar = {**CUBE, "extent": {"x": 0.05, "y": 0.015, "z": 0.015}}
    arms = Arms()
    ctx = arms.ctx()
    scripts["hold"].run(ctx, object_obb=bar)
    assert ctx.calls_to("robot.grasp_frame")[0].kwargs["close_heading_deg"] == pytest.approx(90.0)
    ctx = Arms().ctx()
    scripts["hold"].run(ctx, object_obb=CUBE, close_heading_deg=30.0)
    assert ctx.calls_to("robot.grasp_frame")[0].kwargs["close_heading_deg"] == pytest.approx(30.0)


def test_a_flat_object_keeps_the_fingertips_off_its_bottom(scripts):
    flat = {
        **CUBE,
        "center": {"x": 0.5, "y": 0.0, "z": 0.005},
        "extent": {"x": 0.03, "y": 0.02, "z": 0.005},
    }
    arms = Arms()
    scripts["hold"].run(arms.ctx(), object_obb=flat)
    # bottom 0.0 + 10 mm fingertip reach + 3 mm clearance
    assert arms.moves(0)[1][2][2] == pytest.approx(0.013)


def test_jaws_at_the_stop_is_a_lost_hold(scripts):
    arms = Arms(close=0.01)
    out = scripts["hold"].run(arms.ctx(), object_obb=CUBE)
    assert out["route"] == "hold_lost" and "missed" in out["detail"]
    assert arms.log[-2] == ("open", 0) or ("open", 0) in arms.log[-3:]
    np.testing.assert_allclose(arms.p[0], [0.50, 0.0, 0.065], atol=1e-9)  # backed off
    assert out["hold"] == {}


def test_tries_the_flipped_wrist(scripts):
    arms = Arms(refuse=lambda tool, a, pose: "no IK" if pose["rotation"] == DOWN else None)
    out = scripts["hold"].run(arms.ctx(), object_obb=CUBE)
    assert out["route"] == "holding"
    assert out["hold"]["pose"]["rotation"] == DOWN_FLIPPED


@pytest.mark.parametrize("how", ["raise", "unreachable"])
def test_holder_cannot_reach_above_the_object(scripts, how):
    arms = Arms(refuse=lambda tool, a, pose: how)
    out = scripts["hold"].run(arms.ctx(), object_obb=CUBE)
    assert out["route"] == "unreachable" and "above the object" in out["detail"]


def test_holder_stopped_on_the_way_down(scripts):
    arms = Arms()
    arms.floor[0] = 0.045  # something 30 mm above the hold pose
    out = scripts["hold"].run(arms.ctx(), object_obb=CUBE)
    assert out["route"] == "unreachable" and "down onto the object" in out["detail"]
    assert ("close", 0, 300) not in arms.log


def test_refuses_one_arm_twice_or_a_clamped_second_arm(scripts):
    out = scripts["hold"].run(Arms().ctx(), object_obb=CUBE, other_arm=0)
    assert out["route"] == "unreachable" and "both arm 0" in out["detail"]
    arms = Arms()
    arms.p[1] = arms.p[0].copy()
    out = scripts["hold"].run(arms.ctx(), object_obb=CUBE)
    assert out["route"] == "unreachable" and "separate arm" in out["detail"]
    assert not arms.moves()
    arms = Arms()
    arms.rot[1] = {"w": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}
    assert scripts["hold"].run(arms.ctx(), object_obb=CUBE)["route"] == "unreachable"


def test_press_hold_stops_on_the_top(scripts):
    arms = Arms()
    arms.floor[0] = 0.040  # closed fingertips (10 mm below the tool centre) on the 30 mm top
    out = scripts["hold"].run(arms.ctx(), object_obb=CUBE, mode="press")
    assert out["route"] == "holding", out["detail"]
    assert ("close", 0, None) in arms.log and ("open", 0) not in arms.log
    m = arms.moves(0)
    np.testing.assert_allclose(m[1][2], [0.50, 0.0, 0.030], atol=1e-9)  # 10 mm into the top
    assert out["hold"]["mode"] == "press"
    assert out["hold"]["pose"]["position"]["z"] == pytest.approx(0.040)


def test_press_on_nothing_is_lost(scripts):
    arms = Arms()
    out = scripts["hold"].run(arms.ctx(), object_obb=CUBE, mode="press")
    assert out["route"] == "hold_lost" and "nothing under it" in out["detail"]


def test_press_stopped_high_or_out_of_reach_is_unreachable(scripts):
    arms = Arms()
    arms.floor[0] = 0.070
    out = scripts["hold"].run(arms.ctx(), object_obb=CUBE, mode="press")
    assert out["route"] == "unreachable" and "in the way" in out["detail"]

    arms = Arms(ik_ok=False)
    out = scripts["hold"].run(arms.ctx(), object_obb=CUBE, mode="press")
    assert out["route"] == "unreachable" and "IK preview" in out["detail"]
    assert all(m[2][2] > 0.04 for m in arms.moves(0))  # never sent into the object


def test_bad_mode_raises(scripts):
    with pytest.raises(ValueError):
        scripts["hold"].run(Arms().ctx(), object_obb=CUBE, mode="squeeze")


def test_holder_arm_one_names_its_arm(scripts):
    arms = Arms()
    arms.close[1] = 0.33
    ctx = arms.ctx()
    out = scripts["hold"].run(ctx, object_obb=CUBE, holder_arm=1, other_arm=0)
    assert out["route"] == "holding"
    assert arms.moves(1) and not arms.moves(0)
    assert ctx.calls_to("robot.close_gripper")[0].kwargs["arm_id"] == 1


# ---------------------------------------------------------------- act

START = {"x": 0.50, "y": -0.065, "z": 0.10}
VIA = [{"x": 0.50, "y": -0.065, "z": 0.02}]
TARGET = {"x": 0.50, "y": 0.005, "z": 0.02}


def test_move_step_goes_and_backs_out(scripts):
    arms = Arms()
    hold = _held(scripts, arms)
    before = len(arms.moves(0))
    out = scripts["act"].run(arms.ctx(), hold=hold, start=START, via=VIA, target=TARGET)
    assert out["route"] == "acted", out["detail"]
    m = arms.moves(1)
    assert [x[0] for x in m] == ["robot.go_to_pose"] + ["robot.go_to_pose_cartesian"] * 4
    np.testing.assert_allclose(m[2][2], [0.50, 0.005, 0.02], atol=1e-9)
    np.testing.assert_allclose(m[-1][2], [0.50, -0.065, 0.10], atol=1e-9)  # back at the start
    assert out["retreated"] and not out["contact"]
    assert len(arms.moves(0)) == before  # the holder is never commanded
    assert out["holder_shift"] == pytest.approx(0.0)
    assert m[0][3] == SIDE  # keeps its own rotation


def test_press_step_may_end_on_contact(scripts):
    arms = Arms()
    hold = _held(scripts, arms)
    arms.wall[1] = -0.025
    out = scripts["act"].run(
        arms.ctx(), hold=hold, start=START, via=VIA, target=TARGET, mode="press", hand="closed"
    )
    assert out["route"] == "acted", out["detail"]
    assert out["contact"] and out["stopped_short"] == pytest.approx(0.03)
    assert ("close", 1, None) in arms.log and ("ik", 1) in arms.log
    assert out["retreated"]


def test_move_step_stopped_short_is_unreachable_and_backs_out(scripts):
    arms = Arms()
    hold = _held(scripts, arms)
    arms.wall[1] = -0.025
    out = scripts["act"].run(arms.ctx(), hold=hold, start=START, via=VIA, target=TARGET)
    assert out["route"] == "unreachable" and "waypoint 2" in out["detail"]
    np.testing.assert_allclose(arms.p[1], [0.50, -0.065, 0.10], atol=1e-9)
    assert out["retreated"]


def test_press_target_out_of_reach_is_not_read_as_contact(scripts):
    arms = Arms()
    hold = _held(scripts, arms)
    arms.ik_ok = False
    out = scripts["act"].run(
        arms.ctx(), hold=hold, start=START, via=VIA, target=TARGET, mode="press"
    )
    assert out["route"] == "unreachable" and "IK preview" in out["detail"]
    assert not any(np.allclose(m[2], [0.50, 0.005, 0.02]) for m in arms.moves(1))


@pytest.mark.parametrize("how", ["raise", "unreachable"])
def test_actor_cannot_reach_its_start(scripts, how):
    arms = Arms()
    hold = _held(scripts, arms)
    arms.refuse = lambda tool, a, pose: how if a == 1 else None
    out = scripts["act"].run(arms.ctx(), hold=hold, start=START, target=TARGET)
    assert out["route"] == "unreachable" and "first waypoint" in out["detail"]


def test_actor_top_down_rotation_and_default_start(scripts):
    arms = Arms()
    hold = _held(scripts, arms)
    ctx = arms.ctx()
    out = scripts["act"].run(ctx, hold=hold, target=TARGET, close_heading_deg=0.0, hand="open")
    assert out["route"] == "acted"
    m = arms.moves(1)
    np.testing.assert_allclose(m[0][2], [0.40, -0.25, 0.30], atol=1e-9)  # starts where it is
    assert m[0][3] == DOWN and ctx.calls_to("robot.grasp_frame")[-1].kwargs["arm_id"] == 1
    assert ("open", 1) in arms.log


def test_act_reports_a_moved_holder_and_refuses_one_arm(scripts):
    arms = Arms()
    hold = _held(scripts, arms)
    arms.p[0] = arms.p[0] + np.array([0.0, 0.03, 0.0])
    out = scripts["act"].run(arms.ctx(), hold=hold, start=START, target=TARGET)
    assert out["holder_shift"] == pytest.approx(0.03) and "holding hand" in out["detail"]
    out = scripts["act"].run(arms.ctx(), hold={**hold, "other_arm": 0}, target=TARGET)
    assert out["route"] == "unreachable" and "both arm 0" in out["detail"]
    with pytest.raises(ValueError):
        scripts["act"].run(arms.ctx(), hold=hold, target=TARGET, mode="shove")


# ---------------------------------------------------------------- release


def test_intact_hold_is_released(scripts):
    arms = Arms()
    hold = _held(scripts, arms)
    out = scripts["release"].run(arms.ctx(), hold=hold, step_route="acted")
    assert out["route"] == "held_and_released", out["detail"]
    assert arms.log[-2] == ("open", 0) or ("open", 0) in arms.log
    np.testing.assert_allclose(arms.p[0], [0.50, 0.0, 0.065], atol=1e-9)
    assert out["holder_shift"] == pytest.approx(0.0) and out["grip_change"] == pytest.approx(0.0)
    assert out["object_shift"] == -1.0


def test_changed_jaw_opening_is_a_lost_hold(scripts):
    arms = Arms()
    hold = _held(scripts, arms)
    arms.grip[0] = 0.25
    out = scripts["release"].run(arms.ctx(), hold=hold)
    assert out["route"] == "hold_lost" and "moved in the hand" in out["detail"]
    assert ("open", 0) in arms.log[-3:]  # it still lets go
    arms = Arms()
    hold = _held(scripts, arms)
    arms.grip[0] = 0.02
    out = scripts["release"].run(arms.ctx(), hold=hold)
    assert out["route"] == "hold_lost" and "stop" in out["detail"]


def test_a_pushed_holder_is_a_lost_hold_and_backs_out_from_where_it_is(scripts):
    arms = Arms()
    hold = _held(scripts, arms)
    arms.p[0] = arms.p[0] + np.array([0.0, 0.008, 0.0])
    out = scripts["release"].run(arms.ctx(), hold=hold)
    assert out["route"] == "hold_lost" and "8.0 mm" in out["detail"]
    np.testing.assert_allclose(arms.p[0], [0.50, 0.008, 0.065], atol=1e-9)


def test_a_sunken_press_is_a_lost_hold(scripts):
    arms = Arms()
    arms.floor[0] = 0.040
    hold = _held(scripts, arms, mode="press")
    arms.p[0] = arms.p[0] - np.array([0.0, 0.0, 0.010])  # the object left from under the hand
    n_open = arms.log.count(("open", 0))
    out = scripts["release"].run(arms.ctx(), hold=hold)
    assert out["route"] == "hold_lost"
    assert arms.log.count(("open", 0)) == n_open  # a press has nothing to open


def test_a_reperceived_object_that_moved_is_a_lost_hold(scripts):
    arms = Arms()
    hold = _held(scripts, arms)
    moved = {**CUBE, "center": {"x": 0.50, "y": 0.016, "z": 0.015}}
    out = scripts["release"].run(arms.ctx(), hold=hold, object_obb=moved)
    assert out["route"] == "hold_lost" and "16.0 mm" in out["detail"]
    assert out["object_shift"] == pytest.approx(0.016)


def test_a_failed_step_or_stuck_holder_is_unreachable(scripts):
    arms = Arms()
    hold = _held(scripts, arms)
    out = scripts["release"].run(arms.ctx(), hold=hold, step_route="unreachable")
    assert out["route"] == "unreachable" and "'unreachable'" in out["detail"]
    assert ("open", 0) in arms.log[-3:]

    arms = Arms()
    hold = _held(scripts, arms)
    arms.refuse = lambda tool, a, pose: "raise"
    out = scripts["release"].run(arms.ctx(), hold=hold)
    assert out["route"] == "unreachable" and "backing out" in out["detail"]

    arms = Arms()
    hold = _held(scripts, arms)
    arms.opens = False
    out = scripts["release"].run(arms.ctx(), hold=hold)
    assert out["route"] == "unreachable" and "after opening" in out["detail"]


def test_a_custom_step_exit_counts_as_success(scripts):
    arms = Arms()
    hold = _held(scripts, arms)
    out = scripts["release"].run(
        arms.ctx(), hold=hold, step_route="pushed", step_success=["pushed"]
    )
    assert out["route"] == "held_and_released"


# ---------------------------------------------------------------- example


def test_example_declares_every_route():
    sg = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(sg["exit"]["success_values"]) | {sg["on_error"]}
    assert {"held_and_released", "hold_lost", "unreachable"} <= declared
    mapping = sg["conditional_edges"]["hold"]["mapping"]
    assert set(mapping) == {"holding", "hold_lost", "unreachable"}
    assert mapping["holding"] == "act" and ["act", "release"] in sg["edges"]
