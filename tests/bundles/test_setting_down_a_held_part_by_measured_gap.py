"""setting-down-a-held-part-by-measured-gap: the gap reading on a ray-cast frame, the stepping rules, and the
skill's lowering and routing with FakeContext."""

from __future__ import annotations

import numpy as np
import pytest

from . import _wrist_rig as rig

BUNDLE = "setting-down-a-held-part-by-measured-gap"
TALL = 0.049
TOP = (0.0, 0.0, 0.010)  # the held part's upper end: 10 mm above the tool centre
HELD = (TOP, TALL, (0.0245, 0.0074))
TOP_IN_TOOL = {"x": 0.0, "y": 0.0, "z": -0.010}  # the same, in the tool's frame (its z points down)
PLACE = rig.T(
    rig.DOWN, [0.15, 0.0, TALL - TOP[2]]
)  # where the model has the part's lower end on the support


@pytest.fixture(scope="module")
def script(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b.canonical_scripts["set_down_by_gap"].module


def run(script, r, **kw):
    return script.run(
        r.ctx(), rig.pose(PLACE), TALL, held_top_in_tool=TOP_IN_TOOL, light_part=False, **kw
    )


def test_the_gap_is_read_to_half_a_millimetre(script):
    r = rig.Rig(rig.T(rig.DOWN, [0.15, 0.0, PLACE[2, 3] + 0.03]), held=HELD)
    cam = r.observation()["cameras"][0]
    t_cam = rig.matrix(cam["pose"])
    expect = (np.linalg.inv(t_cam) @ np.append(r.held_top(), 1.0))[:3]
    got = script.wd.held_gap(cam["depth"], cam["intrinsics"], t_cam, TALL, expect)
    assert got["seen"] and got["gap"] == pytest.approx(0.03, abs=0.0005)


def test_steps_shrink_and_contact_is_told_from_the_gap_not_closing(script):
    assert script.next_step(0.02) == pytest.approx(0.012)
    assert script.next_step(0.006) == pytest.approx(0.003)
    assert script.next_step(0.002) == pytest.approx(0.001)
    assert script.on_the_support(
        0.005, 0.0048, 0.003
    )  # the hand came down 3 mm, the part's end 0.2
    assert not script.on_the_support(0.005, 0.002, 0.003)  # both came down
    assert not script.on_the_support(
        0.005, 0.005, 0.0005
    )  # the hand did not move: not contact, a stall


def test_lowered_by_what_it_sees_and_let_go_a_millimetre_up(script):
    r = rig.Rig(rig.T(rig.DOWN, [0.0, 0.0, 0.2]), held=HELD)
    out = run(script, r)
    assert out["route"] == "set_down", out
    assert r.released
    low = r.tool[2, 3] + TOP[2] - TALL  # the part's lower end when it was let go
    assert 0.0 <= low <= 0.0025
    assert out["gap"] == pytest.approx(low, abs=0.0006)
    assert out["lowered"] == pytest.approx(0.09 - low, abs=0.001)


def test_a_part_that_sits_lower_in_the_hand_than_the_model_is_not_driven_into_the_support(script):
    # the upper end 8 mm lower than the grasp's geometry has it: lowered by the model it would be 8 mm into the support
    low_top = ((0.0, 0.0, 0.002), TALL, HELD[2])
    r = rig.Rig(rig.T(rig.DOWN, [0.0, 0.0, 0.2]), held=low_top)
    out = run(script, r)
    assert out["route"] == "set_down"
    assert r.tool[2, 3] + 0.002 - TALL >= 0.0  # its lower end never went under the support


def test_a_hand_that_cannot_go_lower_keeps_the_part(script):
    r = rig.Rig(rig.T(rig.DOWN, [0.0, 0.0, 0.2]), held=HELD, floor=PLACE[2, 3] + 0.0065)
    out = run(script, r)
    assert out["route"] == "cannot_lower", out
    assert not r.released and out["gap"] == pytest.approx(0.0065, abs=0.0008)


def test_a_part_that_is_not_seen_is_not_lowered_by_the_model(script):
    r = rig.Rig(rig.T(rig.DOWN, [0.0, 0.0, 0.2]), held=None)
    out = run(script, r)
    assert out["route"] == "not_seen" and not r.released
    assert r.tool[2, 3] == pytest.approx(PLACE[2, 3] + 0.09)  # still above the place


def test_a_place_out_of_reach_is_unreachable(script):
    r = rig.Rig(rig.T(rig.DOWN, [0.0, 0.0, 0.2]), held=HELD, reach=lambda m: False)
    assert run(script, r)["route"] == "unreachable" and r.moves == 0
