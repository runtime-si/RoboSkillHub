"""The restated BrickSim clutch trigger: which contacts may engage and which must not."""

from __future__ import annotations

import math

import pytest

from roboskillhub.tasks.assembly_eval.lego_clutch import (
    STUD_HEIGHT_M,
    STUD_PITCH_M,
    AssemblyThresholds,
    detect_assembly,
    footprint_overlap,
    gap_window_above_seat_m,
)
from roboskillhub.tasks.assembly_eval.lego_episode import load_config

BRICK_2X4 = (4, 2)
SEATED = (0.0, 0.0, 0.0)
PRESS = -2.0
WEIGHT_2X4 = -0.0023 * 9.81


def test_thresholds_match_the_versioned_config():
    th = AssemblyThresholds.from_config(load_config())
    assert th == AssemblyThresholds()


def test_seated_aligned_and_pressed_engages():
    r = detect_assembly(SEATED, 0.0, 0.0, PRESS, BRICK_2X4, BRICK_2X4)
    assert r["accepted"]
    assert r["offset"] == (0, 0) and r["yaw_q"] == 0
    assert r["overlap_studs"] == (4, 2)


def test_resting_under_own_weight_does_not_engage():
    r = detect_assembly(SEATED, 0.0, 0.0, WEIGHT_2X4, BRICK_2X4, BRICK_2X4)
    assert not r["accepted"]
    assert not r["checks"]["force"]
    assert all(v for k, v in r["checks"].items() if k != "force")


@pytest.mark.parametrize(
    "rel_p, tilt, yaw, failing",
    [
        ((0.003, 0.0, 0.0), 0.0, 0.0, "position"),
        ((0.0, 0.0, 0.0), 0.0, math.radians(9.0), "yaw"),
        ((0.0, 0.0, 0.0), math.radians(6.0), 0.0, "tilt"),
        ((0.0, 0.0, STUD_HEIGHT_M + 0.0015), 0.0, 0.0, "gap"),
    ],
)
def test_each_tolerance_blocks_engagement_even_when_pressed(rel_p, tilt, yaw, failing):
    r = detect_assembly(rel_p, tilt, yaw, PRESS, BRICK_2X4, BRICK_2X4)
    assert not r["accepted"]
    assert not r["checks"][failing]


def test_gap_window_reaches_above_the_seat():
    lo, hi = gap_window_above_seat_m()
    assert lo == pytest.approx(-0.0033)
    assert hi == pytest.approx(0.0027)
    # A brick still 2.5 mm above the seat (studs barely entered) engages if pressed: the weld then pulls it to
    # the seated pose, so the simulator cannot represent partial insertion depth.
    assert detect_assembly((0.0, 0.0, 0.0025), 0.0, 0.0, PRESS, BRICK_2X4, BRICK_2X4)["accepted"]


def test_one_stud_shift_engages_six_of_eight_studs():
    r = detect_assembly((0.0, STUD_PITCH_M, 0.0), 0.0, 0.0, PRESS, (2, 4), (2, 4))
    assert r["accepted"]
    assert r["offset"] == (0, 1)
    ox, oy = r["overlap_studs"]
    assert ox * oy == 6


def test_quarter_turn_and_off_footprint_overlap():
    assert footprint_overlap((4, 4), (4, 2), (4, 0), 1) == (2, 4)
    assert footprint_overlap((4, 2), (4, 2), (0, 0), 2) == (0, 0)
    assert footprint_overlap((4, 2), (4, 2), (4, 2), 2) == (4, 2)
    assert footprint_overlap((4, 2), (4, 2), (5, 0), 0) == (0, 2)
