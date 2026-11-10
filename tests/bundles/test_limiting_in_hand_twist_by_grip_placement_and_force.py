"""limiting-in-hand-twist-by-grip-placement-and-force: the limit model and the ranking."""

from __future__ import annotations

import json
import math

import pytest
from gap.testing import FakeContext

BUNDLE = "limiting-in-hand-twist-by-grip-placement-and-force"
COM = {"x": 0.40, "y": 0.10, "z": 0.02}
Y = {"x": 0.0, "y": 1.0, "z": 0.0}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["limit_twist"].module


def grip(dx, axis=Y):
    return {"position": {"x": COM["x"] + dx, "y": COM["y"], "z": COM["z"]}, "close_axis": axis}


def test_pure_force_and_pure_torque_limits(script):
    assert script.pad_force_needed(1.0, 0.0, 0.5, 0.005) == pytest.approx(2.0, rel=1e-3)
    assert script.pad_force_needed(0.0, 0.01, 0.5, 0.005) == pytest.approx(4.0, rel=1e-3)


def test_combined_load_needs_more_than_either_alone(script):
    both = script.pad_force_needed(1.0, 0.01, 0.5, 0.005)
    assert both > script.pad_force_needed(1.0, 0.0, 0.5, 0.005)
    assert both > script.pad_force_needed(0.0, 0.01, 0.5, 0.005)


def test_grip_through_the_centre_of_mass_is_chosen(script):
    out = script.run(
        FakeContext({}),
        center_of_mass=COM,
        candidates=[grip(0.04), grip(0.0), grip(-0.02)],
        mass=0.1,
    )
    assert out["route"] == "grip_selected"
    assert out["grip_index"] == 1
    assert out["torque"] == pytest.approx(0.0, abs=1e-9)
    assert [r["index"] for r in json.loads(out["ranked"])] == [1, 2, 0]


def test_torque_is_weight_times_lever(script):
    out = script.run(
        FakeContext({}), center_of_mass=COM, candidates=[grip(0.04)], mass=0.1, grip_force=100.0
    )
    assert out["torque"] == pytest.approx(0.1 * 9.81 * 0.04, rel=1e-6)
    assert out["lever"] == pytest.approx(0.04)


def test_an_off_centre_grip_that_cannot_hold_the_twist(script):
    # 100 g at 40 mm: 39 mN.m; plastic pads at 9.5 N cannot (limit 46 mN.m, margin 2).
    out = script.run(FakeContext({}), center_of_mass=COM, candidates=[grip(0.04)], mass=0.1)
    assert out["route"] == "no_grip_holds"
    assert out["required_force"] > 9.5
    assert "nearer the centre of mass" in out["detail"]


def test_a_part_too_heavy_for_the_squeeze_needs_suction(script):
    out = script.run(FakeContext({}), center_of_mass=COM, candidates=[grip(0.0)], mass=2.0)
    assert out["route"] == "needs_suction"


def test_closing_axis_along_the_lever_puts_no_twist(script):
    # The lever runs along the closing axis: the torque is about a horizontal axis across it, not the pads'.
    out = script.run(
        FakeContext({}),
        center_of_mass=COM,
        candidates=[grip(0.04, axis={"x": 1.0, "y": 0.0, "z": 0.0})],
        mass=0.1,
    )
    assert out["torque"] == pytest.approx(0.0, abs=1e-9)


def test_mass_from_area_thickness_density(script):
    out = script.run(
        FakeContext({}),
        center_of_mass=COM,
        candidates=[grip(0.02)],
        area=0.001,
        thickness=0.01,
        density=1000.0,
    )
    assert out["torque"] == pytest.approx(0.01 * 9.81 * 0.02, rel=1e-6)


def test_missing_mass_raises_and_no_candidates_routes(script):
    with pytest.raises(ValueError):
        script.run(FakeContext({}), center_of_mass=COM, candidates=[grip(0.0)])
    out = script.run(FakeContext({}), center_of_mass=COM, candidates=[], mass=0.1)
    assert out["route"] == "no_grip_holds"


def test_grasp_candidates_take_the_closing_axis_from_the_frame(script):
    # Top-down frame turned 90 deg: its y axis (closing) lies along world x, so a lever along x gives no twist.
    h = math.radians(90.0) / 2
    q = {"w": 0.0, "x": math.cos(h), "y": math.sin(h), "z": 0.0}
    cands = {"poses": [{"position": grip(0.04)["position"], "rotation": q}]}
    out = script.run(FakeContext({}), center_of_mass=COM, candidates=cands, mass=0.1)
    assert out["torque"] == pytest.approx(0.0, abs=1e-6)
    q0 = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
    out = script.run(
        FakeContext({}),
        center_of_mass=COM,
        candidates={"poses": [{"position": grip(0.04)["position"], "rotation": q0}]},
        mass=0.1,
        grip_force=100.0,
    )
    assert out["torque"] == pytest.approx(0.1 * 9.81 * 0.04, rel=1e-6)
