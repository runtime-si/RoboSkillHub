"""inserting-a-peg-with-pose-refinement: align on the hole's axis, refine from the read-back, push in."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError
from scipy.spatial.transform import Rotation

BUNDLE = "inserting-a-peg-with-pose-refinement"
DOWN = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])  # tool z pointing at the table
MOUTH = np.array([0.40, 0.10, 0.032])
AXIS_DOWN = np.array([0.0, 0.0, -1.0])


@pytest.fixture(scope="module")
def bundle(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b


@pytest.fixture(scope="module")
def align(bundle):
    return bundle.canonical_scripts["align_pre_insert"].module


@pytest.fixture(scope="module")
def push(bundle):
    return bundle.canonical_scripts["push_insert"].module


def _v(a):
    return {"x": float(a[0]), "y": float(a[1]), "z": float(a[2])}


def _q(r: Rotation):
    x, y, z, w = r.as_quat()
    return {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}


def _m(pose):
    q = pose["rotation"]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    out[:3, 3] = [pose["position"][k] for k in "xyz"]
    return out


def peg_straight(behind=0.065):
    """A peg held straight along the tool's z, its tip ``behind`` past the tool centre."""
    return {"position": _v([0.0, 0.0, behind]), "rotation": _q(Rotation.identity())}


class Arm:
    """Moves land at the command plus ``bias`` (a tracking error the read-back shows),
    or nowhere at all when ``stuck``. A plate at the mouth with a hole of radius
    ``hole_r`` around ``hole_xy`` along ``axis``; the hole is ``hole_depth`` deep."""

    def __init__(
        self,
        peg,
        bias=(0.0, 0.0, 0.0),
        axis=AXIS_DOWN,
        mouth=MOUTH,
        hole_offset=(0.0, 0.0, 0.0),
        hole_r=0.0015,
        hole_depth=0.03,
        start_rot=DOWN,
    ):
        self.peg = _m(peg)
        self.bias = np.asarray(bias, dtype=float)
        self.axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        self.mouth = np.asarray(mouth, dtype=float)
        self.hole = self.mouth + np.asarray(hole_offset, dtype=float)
        self.hole_r = hole_r
        self.hole_depth = hole_depth
        self.H = np.eye(4)
        self.H[:3, :3] = start_rot.as_matrix()
        self.H[:3, 3] = self.mouth - 0.15 * self.axis + np.array([0.05, -0.03, 0.0])
        self.stuck = False
        self.calls: list[tuple[str, dict]] = []
        self.contact_raises = False
        self.ik_ok = True

    def ee(self, **kw):
        self.calls.append(("get_ee_pose", kw))
        q = Rotation.from_matrix(self.H[:3, :3])
        return {"pose": {"position": _v(self.H[:3, 3]), "rotation": _q(q)}}

    def _land(self, pose):
        if self.stuck:
            return
        target = _m(pose)
        target[:3, 3] += self.bias
        self.H = target

    def go(self, pose, **kw):
        self.calls.append(("go_to_pose", kw))
        self._land(pose)
        return {"reached": True, "outcome": "reached"}

    def cart(self, pose, **kw):
        self.calls.append(("go_to_pose_cartesian", kw))
        self._land(pose)
        return {"reached": True, "outcome": "reached"}

    def tip(self, H=None):
        return ((self.H if H is None else H) @ self.peg)[:3, 3]

    def probe(self, pose, **kw):
        self.calls.append(("move_cartesian_until_contact", kw))
        target = _m(pose)
        target[:3, 3] += self.bias
        tip_goal = self.tip(target)
        d = tip_goal - self.hole
        along = float(d @ self.axis)
        lateral = float(np.linalg.norm(d - along * self.axis))
        stop = min(along, self.hole_depth) if lateral <= self.hole_r else min(along, 0.0005)
        target[:3, 3] += (stop - along) * self.axis
        self.H = target
        if stop < along:
            if self.contact_raises:
                raise ToolError(
                    "robot.move_cartesian_until_contact", "the tool did not reach its target"
                )
            return {"reached": False, "outcome": "contact"}
        return {"reached": True, "outcome": "reached"}

    def ik(self, pose, **kw):
        self.calls.append(("solve_ik", kw))
        return {"success": True, "within_servo_tolerance": self.ik_ok}

    def ctx(self, **overrides):
        tools = {
            "robot.get_ee_pose": self.ee,
            "robot.go_to_pose": self.go,
            "robot.go_to_pose_cartesian": self.cart,
            "robot.move_cartesian_until_contact": self.probe,
            "robot.solve_ik": self.ik,
        }
        tools.update(overrides)
        return FakeContext(tools)

    def moves(self, name):
        return [kw for n, kw in self.calls if n == name]


def _hole(axis=AXIS_DOWN, mouth=MOUTH):
    return {"hole_centre": _v(mouth), "hole_axis": _v(axis)}


def _insert(align, push, arm, peg, **kw):
    a = align.run(arm.ctx(), peg_in_tcp=peg, **_hole(arm.axis, arm.mouth), **kw)
    if a["route"] != "aligned":
        return a, None
    p = push.run(
        arm.ctx(),
        peg_in_tcp=peg,
        pre_insert_pose=a["pre_insert_pose"],
        **_hole(arm.axis, arm.mouth),
    )
    return a, p


# ---------------------------------------------------------------- align


def test_a_tracking_bias_is_refined_out_then_the_peg_goes_in(align, push):
    peg = peg_straight()
    arm = Arm(peg, bias=(0.003, -0.002, 0.001))
    a, p = _insert(align, push, arm, peg)
    assert a["route"] == "aligned"
    assert a["refinements"] == 1
    assert a["lateral_error"] < 1e-9
    assert a["correction"] == pytest.approx(math.sqrt(0.003**2 + 0.002**2 + 0.001**2))
    # The carry keeps the connector's own ball; the refinement asks for a fine one.
    goes = arm.moves("go_to_pose")
    assert "tolerance" not in goes[0]
    assert goes[1]["tolerance"] == pytest.approx(align.FINE_TOLERANCE)
    assert p["route"] == "inserted"
    # 65 mm behind the tip less 15 mm finger clearance caps the 50 mm push at 50 mm
    # here, and the 30 mm deep hole stops it.
    assert p["depth"] == pytest.approx(0.03)
    assert "past the mouth" in p["detail"]


def test_tcp_at_mouth_is_where_the_tool_centre_is_with_the_tip_at_the_mouth(align):
    peg = peg_straight(0.065)
    arm = Arm(peg)
    a = align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    assert a["route"] == "aligned" and a["refinements"] == 0
    np.testing.assert_allclose(
        [a["tcp_at_mouth"][k] for k in "xyz"], MOUTH + np.array([0, 0, 0.065]), atol=1e-9
    )
    tip = arm.tip()
    np.testing.assert_allclose(tip, MOUTH - align.STANDOFF * AXIS_DOWN, atol=1e-9)


def test_a_side_insertion_turns_the_peg_onto_a_horizontal_axis(align, push):
    # The peg lies along the tool's x (a top-down grasp across a lying peg).
    peg = {
        "position": _v([0.05, 0.0, 0.0]),
        "rotation": _q(Rotation.from_euler("y", 90, degrees=True)),
    }
    axis = np.array([0.0, 1.0, 0.0])
    arm = Arm(peg, axis=axis, bias=(0.0, 0.0, -0.002), hole_depth=0.04)
    a, p = _insert(align, push, arm, peg)
    assert a["route"] == "aligned"
    tip_frame = arm.H @ arm.peg
    np.testing.assert_allclose(tip_frame[:3, 2], axis, atol=1e-6)
    assert a["angle_error_deg"] < 1e-4
    assert p["route"] == "inserted"
    assert p["depth"] == pytest.approx(0.035)  # capped: 50 mm behind less 15 mm


def test_a_peg_tilted_in_the_fingers_is_straightened_and_keeps_its_roll(align):
    tilt = Rotation.from_euler("x", 12, degrees=True) * Rotation.from_euler("z", 30, degrees=True)
    peg = {"position": _v(tilt.apply([0.0, 0.0, 0.065])), "rotation": _q(tilt)}
    arm = Arm(peg)
    a = align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    assert a["route"] == "aligned"
    tip_frame = arm.H @ arm.peg
    np.testing.assert_allclose(tip_frame[:3, 2], AXIS_DOWN, atol=1e-6)
    np.testing.assert_allclose(tip_frame[:3, 3], MOUTH - align.STANDOFF * AXIS_DOWN, atol=1e-9)


def test_without_refinements_a_biased_arm_is_misaligned(align):
    peg = peg_straight()
    arm = Arm(peg, bias=(0.004, 0.0, 0.0))
    a = align.run(arm.ctx(), peg_in_tcp=peg, refinements=0, **_hole())
    assert a["route"] == "misaligned"
    assert a["lateral_error"] == pytest.approx(0.004)
    assert "outside tolerance" in a["detail"]
    assert len(arm.moves("go_to_pose")) == 1


def test_a_hand_that_does_not_follow_is_misaligned_by_the_cap(align):
    peg = peg_straight()
    arm = Arm(peg)
    arm.H[:3, 3] = MOUTH - (align.STANDOFF + 0.065) * AXIS_DOWN + np.array([0.008, 0, 0])
    arm.stuck = True
    a = align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    assert a["route"] == "misaligned"
    assert "correction grew" in a["detail"]
    assert a["refinements"] == 2  # 8 and 16 mm moved; 24 mm would pass the 20 mm cap


def test_a_small_stuck_offset_runs_out_of_refinements(align):
    peg = peg_straight()
    arm = Arm(peg)
    arm.H[:3, 3] = MOUTH - (align.STANDOFF + 0.065) * AXIS_DOWN + np.array([0.003, 0, 0])
    arm.stuck = True
    a = align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    assert a["route"] == "misaligned" and a["refinements"] == 3
    assert a["lateral_error"] == pytest.approx(0.003)


def test_a_refused_carry_is_unreachable(align):
    peg = peg_straight()
    arm = Arm(peg)
    routed = align.run(
        arm.ctx(
            **{"robot.go_to_pose": lambda pose, **kw: {"reached": False, "outcome": "unreachable"}}
        ),
        peg_in_tcp=peg,
        **_hole(),
    )
    assert routed["route"] == "unreachable" and "unreachable" in routed["detail"]

    def boom(pose, **kw):
        raise ToolError("robot.go_to_pose", "IK failed")

    raised = align.run(arm.ctx(**{"robot.go_to_pose": boom}), peg_in_tcp=peg, **_hole())
    assert raised["route"] == "unreachable" and "IK failed" in raised["detail"]


def test_a_carry_that_ends_far_off_is_unreachable(align):
    peg = peg_straight()
    arm = Arm(peg, bias=(0.03, 0.0, 0.0))
    a = align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    assert a["route"] == "unreachable" and "30 mm away" in a["detail"]


def test_an_ik_preview_out_of_reach_moves_nothing(align):
    peg = peg_straight()
    arm = Arm(peg)
    arm.ik_ok = False
    a = align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    assert a["route"] == "unreachable" and "IK preview" in a["detail"]
    assert arm.moves("go_to_pose") == []


def test_a_refused_refinement_is_unreachable(align):
    peg = peg_straight()
    arm = Arm(peg, bias=(0.004, 0.0, 0.0))
    first = {"n": 0}

    def go(pose, **kw):
        first["n"] += 1
        if first["n"] > 1:
            return {"reached": False, "outcome": "timeout"}
        return arm.go(pose, **kw)

    a = align.run(arm.ctx(**{"robot.go_to_pose": go}), peg_in_tcp=peg, **_hole())
    assert a["route"] == "unreachable" and a["detail"].startswith("refinement 1")


def test_arm_id_is_passed_only_when_nonzero(align):
    peg = peg_straight()
    arm = Arm(peg, bias=(0.003, 0.0, 0.0))
    align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    assert all("arm_id" not in kw for _, kw in arm.calls)
    arm = Arm(peg, bias=(0.003, 0.0, 0.0))
    align.run(arm.ctx(), peg_in_tcp=peg, arm_id=1, **_hole())
    assert all(kw.get("arm_id") == 1 for _, kw in arm.calls)


def test_fine_tolerance_zero_leaves_the_connector_default(align):
    peg = peg_straight()
    arm = Arm(peg, bias=(0.003, 0.0, 0.0))
    align.run(arm.ctx(), peg_in_tcp=peg, fine_tolerance=0.0, **_hole())
    assert all("tolerance" not in kw for kw in arm.moves("go_to_pose"))


def test_a_zero_axis_is_an_error(align, push):
    peg = peg_straight()
    arm = Arm(peg)
    with pytest.raises(ValueError):
        align.run(arm.ctx(), peg_in_tcp=peg, hole_centre=_v(MOUTH), hole_axis=_v([0, 0, 0]))
    with pytest.raises(ValueError):
        push.run(arm.ctx(), peg_in_tcp=peg, hole_centre=_v(MOUTH), hole_axis=_v([0, 0, 0]))


def test_an_unreadable_hand_is_unreachable(align, push):
    peg = peg_straight()
    arm = Arm(peg)

    def nope(**kw):
        raise ToolError("robot.get_ee_pose", "no state")

    for script in (align, push):
        out = script.run(arm.ctx(**{"robot.get_ee_pose": nope}), peg_in_tcp=peg, **_hole())
        assert out["route"] == "unreachable" and "could not be read" in out["detail"]


# ---------------------------------------------------------------- push


def test_a_hole_off_its_estimate_stops_short_and_backs_off(align, push):
    peg = peg_straight()
    arm = Arm(peg, hole_offset=(0.004, 0.0, 0.0))
    a, p = _insert(align, push, arm, peg)
    assert a["route"] == "aligned"
    assert p["route"] == "stopped_short"
    assert p["depth"] < push.MIN_DEPTH
    assert "backed off" in p["detail"]
    np.testing.assert_allclose(
        arm.H[:3, 3], [a["pre_insert_pose"]["position"][k] for k in "xyz"], atol=1e-9
    )


def test_a_station_style_error_on_contact_is_read_as_a_stop(align, push):
    peg = peg_straight()
    arm = Arm(peg)
    arm.contact_raises = True
    a, p = _insert(align, push, arm, peg)
    assert p["route"] == "inserted" and p["depth"] == pytest.approx(0.03)


def test_the_push_is_capped_where_the_fingers_would_meet_the_rim(align, push):
    peg = peg_straight(0.04)
    arm = Arm(peg, hole_depth=0.08)
    a, p = _insert(align, push, arm, peg)
    assert p["route"] == "inserted"
    assert p["depth"] == pytest.approx(0.04 - push.FINGER_CLEAR)
    assert "capped at 25 mm" in p["detail"]


def test_a_hold_too_near_the_tip_leaves_no_room(align, push):
    peg = peg_straight(0.02)
    arm = Arm(peg)
    a, p = _insert(align, push, arm, peg)
    assert a["route"] == "aligned"
    assert p["route"] == "stopped_short" and "hold it further back" in p["detail"]
    assert arm.moves("move_cartesian_until_contact") == []


def test_a_push_out_of_reach_is_unreachable_before_moving(align, push):
    peg = peg_straight()
    arm = Arm(peg)
    a = align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    arm.ik_ok = False
    p = push.run(arm.ctx(), peg_in_tcp=peg, pre_insert_pose=a["pre_insert_pose"], **_hole())
    assert p["route"] == "unreachable" and "IK preview" in p["detail"]
    assert arm.moves("move_cartesian_until_contact") == []


def test_a_refused_push_that_did_not_move_is_unreachable(align, push):
    peg = peg_straight()
    arm = Arm(peg)
    align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    p = push.run(
        arm.ctx(
            **{
                "robot.move_cartesian_until_contact": lambda pose, **kw: {
                    "reached": False,
                    "outcome": "unreachable",
                }
            }
        ),
        peg_in_tcp=peg,
        **_hole(),
    )
    assert p["route"] == "unreachable" and p["detail"].startswith("push: unreachable")


def test_the_push_goes_in_short_steps_and_stops_at_the_floor(align, push):
    peg = peg_straight()
    arm = Arm(peg, hole_depth=0.03)
    targets = []

    def probe(pose, **kw):
        targets.append(np.array([pose["position"][k] for k in "xyz"]))
        return arm.probe(pose, **kw)

    a = align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    p = push.run(
        arm.ctx(**{"robot.move_cartesian_until_contact": probe}),
        peg_in_tcp=peg,
        pre_insert_pose=a["pre_insert_pose"],
        **_hole(),
    )
    assert p["route"] == "inserted" and p["depth"] == pytest.approx(0.03)
    steps = np.diff(np.array(targets), axis=0)
    assert np.all(np.linalg.norm(steps, axis=1) <= push.PUSH_STEP + 1e-9)
    # 30 mm standoff + 50 mm push = 80 mm in 16 steps; the floor (60 mm of travel)
    # stops the 13th.
    assert len(targets) == 13


def test_a_step_that_lands_short_does_not_shorten_the_next(align, push):
    peg = peg_straight()
    arm = Arm(peg, hole_depth=0.08)
    a = align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    start_z = arm.H[2, 3]

    def lagging(pose, **kw):
        # Lands 3 mm short of every step and calls it reached (an arrival ball).
        target = _m(pose)
        target[:3, 3] -= 0.003 * arm.axis
        arm.H = target
        return {"reached": True, "outcome": "reached"}

    p = push.run(
        arm.ctx(**{"robot.move_cartesian_until_contact": lagging}),
        peg_in_tcp=peg,
        pre_insert_pose=a["pre_insert_pose"],
        **_hole(),
    )
    travel = start_z - arm.H[2, 3]
    assert travel == pytest.approx(0.03 + 0.05 - 0.003)
    assert p["route"] == "inserted" and p["depth"] == pytest.approx(0.047)


def _slipping(arm, touch_at, bias=(0.0, 0.0, 0.0)):
    """A push that always reports "reached": the hand lands at the command plus ``bias``,
    shoved 1.5 mm across the axis once the commanded tip is ``touch_at`` past the mouth
    (the peg then slides in the fingers, which the read-back cannot see)."""
    targets = []

    def probe(pose, **kw):
        target = _m(pose)
        targets.append(target[:3, 3].copy())
        target[:3, 3] += np.asarray(bias, dtype=float)
        if float((arm.tip(_m(pose)) - arm.mouth) @ arm.axis) > touch_at:
            target[:3, 3] += [0.0015, 0.0, 0.0]
        arm.H = target
        return {"reached": True, "outcome": "reached"}

    return probe, targets


def test_a_touch_on_the_rim_that_only_shoves_the_hand_stops_short(align, push):
    peg = peg_straight()
    arm = Arm(peg)
    a = align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    probe, targets = _slipping(arm, touch_at=0.0)
    p = push.run(
        arm.ctx(**{"robot.move_cartesian_until_contact": probe}),
        peg_in_tcp=peg,
        pre_insert_pose=a["pre_insert_pose"],
        **_hole(),
    )
    assert p["route"] == "stopped_short"
    assert p["deflection"] == pytest.approx(0.0015)
    assert "pushed 1.5 mm off its line" in p["detail"] and "backed off" in p["detail"]
    # 30 mm standoff in 5 mm steps: the 7th step is the first past the mouth.
    assert len(targets) == 7


def test_a_touch_on_the_floor_ends_the_push_inserted(align, push):
    peg = peg_straight()
    arm = Arm(peg)
    a = align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    probe, targets = _slipping(arm, touch_at=0.03)
    p = push.run(
        arm.ctx(**{"robot.move_cartesian_until_contact": probe}),
        peg_in_tcp=peg,
        pre_insert_pose=a["pre_insert_pose"],
        **_hole(),
    )
    assert p["route"] == "inserted" and p["depth"] == pytest.approx(0.035)
    assert len(targets) == 13


def test_a_steady_landing_offset_is_not_a_touch(align, push):
    peg = peg_straight()
    arm = Arm(peg)
    a = align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    probe, targets = _slipping(arm, touch_at=1.0, bias=(0.003, -0.002, 0.0))
    p = push.run(
        arm.ctx(**{"robot.move_cartesian_until_contact": probe}),
        peg_in_tcp=peg,
        pre_insert_pose=a["pre_insert_pose"],
        **_hole(),
    )
    assert p["route"] == "inserted" and p["deflection"] < 1e-9
    assert len(targets) == 16


def test_deflection_stop_zero_pushes_through_a_touch(align, push):
    peg = peg_straight()
    arm = Arm(peg)
    a = align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    probe, targets = _slipping(arm, touch_at=0.0)
    p = push.run(
        arm.ctx(**{"robot.move_cartesian_until_contact": probe}),
        peg_in_tcp=peg,
        pre_insert_pose=a["pre_insert_pose"],
        deflection_stop=0.0,
        **_hole(),
    )
    assert len(targets) == 16
    assert p["route"] == "inserted" and p["deflection"] == pytest.approx(0.0015)


def test_the_push_moves_along_the_axis_only(align, push):
    peg = peg_straight()
    arm = Arm(peg)
    align.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    before = arm.H.copy()
    push.run(arm.ctx(), peg_in_tcp=peg, **_hole())
    d = arm.H[:3, 3] - before[:3, 3]
    np.testing.assert_allclose(d[:2], 0.0, atol=1e-12)
    np.testing.assert_allclose(arm.H[:3, :3], before[:3, :3], atol=1e-12)


# ---------------------------------------------------------------- the bundle


def test_every_route_is_declared_in_the_example(bundle):
    example = json.loads((bundle.bundle_dir / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    declared |= set(example["conditional_edges"]["align"]["mapping"])
    assert {"aligned", "misaligned", "unreachable", "inserted", "stopped_short"} <= declared
    for route, target in example["conditional_edges"]["align"]["mapping"].items():
        if target == "END":
            assert route in declared - {"aligned"}
