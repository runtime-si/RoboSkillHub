"""A brick placed flush beside another: the closing axis is chosen clear of it, and align hovers off the seat
away from it (seq3_stack brick 2)."""

from __future__ import annotations

import pytest
from lego_fixtures import align_request, placed

from roboskillhub.tasks.assembly_eval.lego_world import grasp_close_axes
from roboskillhub.tasks.lego.align import plan_alignment
from roboskillhub.tasks.lego.fakes import FakeClock


def test_end_grasp_is_clear_where_the_side_grasp_strikes_the_neighbour():
    clock = FakeClock()
    side = align_request(clock, build=[placed("n", 6, 4)])
    assert plan_alignment(side)["reason_code"] == "would_damage_build"
    ends = align_request(clock, build=[placed("n", 6, 4)])
    ends["held"]["close_axis"] = 1
    assert plan_alignment(ends)["ok"] is True


def test_hover_bias_points_away_from_the_flush_neighbour():
    clock = FakeClock()
    req = align_request(clock, build=[placed("n", 6, 4)])
    req["held"]["close_axis"] = 1
    req["limits"]["hover_bias_away_m"] = 0.0005
    plan = plan_alignment(req)
    assert plan["ok"] is True
    bx, by, bz = plan["hover_bias_m"]
    # The neighbour stands on +x (cells x 6-7 next to 4-5): the hover moves 0.5 mm along -x.
    assert bx == pytest.approx(-0.0005) and by == pytest.approx(0.0) and bz == pytest.approx(0.0)
    unbiased = dict(req, limits={**req["limits"], "hover_bias_away_m": 0.0})
    p0 = plan_alignment(unbiased)["hover_brick_pose"]["position_m"]
    p1 = plan["hover_brick_pose"]["position_m"]
    assert p1[0] - p0[0] == pytest.approx(-0.0005)
    # Nothing beside the target: no bias.
    alone = align_request(clock)
    alone["limits"]["hover_bias_away_m"] = 0.0005
    assert "hover_bias_m" not in plan_alignment(alone)
    # A brick on the layer above (or below) is not beside it.
    above = align_request(clock, build=[placed("u", 6, 4, layer=2)])
    above["limits"]["hover_bias_away_m"] = 0.0005
    assert "hover_bias_m" not in plan_alignment(above)


def test_hover_bias_is_bounded():
    clock = FakeClock()
    req = align_request(clock)
    req["limits"]["hover_bias_away_m"] = 0.01
    assert plan_alignment(req)["ok"] is False


def _target(tid, x, y, layer=0):
    """A lego_world-style target: mating pose of a 2x4 turned 90 deg (4 studs along x) on a 32 x 32 plate
    whose top centre is the origin."""
    p = 0.008
    return {
        "target_id": tid,
        "mating": {
            "mating_pose": {
                "frame_id": "world",
                "position_m": [(-16 + x + 2) * p, (-16 + y + 1) * p, layer * 0.0096],
                "quaternion_xyzw": [0.0, 0.0, 0.7071068, 0.7071068],
            }
        },
    }


def test_seq3_stack_grasps_the_second_brick_by_its_ends():
    plate = {"frame_id": "world", "position_m": [0.0, 0.0, 0.0], "quaternion_xyzw": [0, 0, 0, 1]}
    targets = [_target("b1", 14, 14), _target("b2", 14, 16), _target("b3", 14, 14, layer=1)]
    assert grasp_close_axes([], targets, plate) == [0, 1, 0]
    spaced = [_target("b1", 14, 14), _target("b2", 14, 17), _target("b3", 14, 14, layer=1)]
    assert grasp_close_axes([], spaced, plate) == [0, 0, 0]
