from __future__ import annotations

import importlib.util
import math

import pytest
from conftest import REPO

pytest.importorskip("gap_core")


@pytest.fixture(scope="module")
def p():
    spec = importlib.util.spec_from_file_location("predicates_tools", REPO / "tools/predicates/tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v(x, y, z):
    return {"x": x, "y": y, "z": z}


def yaw(deg):
    h = math.radians(deg) / 2
    return {"w": math.cos(h), "x": 0.0, "y": 0.0, "z": math.sin(h)}


IDENTITY = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}


def box(center, half, q=IDENTITY):
    return {"center": v(*center), "extent": v(*half), "orientation": q}


def test_point_in_obb_respects_rotation_and_margin(p):
    long_thin = box((0, 0, 0), (0.2, 0.02, 0.05), yaw(90))
    assert p.point_in_obb(v(0, 0.15, 0), long_thin)["inside"]
    out = p.point_in_obb(v(0.15, 0, 0), long_thin)
    assert out["verdict"] == "outside" and out["depth_m"] == pytest.approx(-0.13)
    assert p.point_in_obb(v(0.03, 0, 0), long_thin, margin_m=0.02)["inside"]
    assert p.point_in_obb(v(0, 0, 0), long_thin)["depth_m"] == pytest.approx(0.02)


def test_resting_on_reads_overlap_and_gap(p):
    table = box((0, 0, -0.01), (0.5, 0.5, 0.01))
    cup = box((0.1, 0, 0.05), (0.03, 0.03, 0.05))
    result = p.resting_on(cup, table)
    assert result["on"] and result["xy_overlap"] == pytest.approx(1.0) and result["gap_m"] == pytest.approx(0)
    hovering = p.resting_on(box((0.1, 0, 0.2), (0.03, 0.03, 0.05)), table)
    assert hovering["verdict"] == "off" and hovering["gap_m"] == pytest.approx(0.15)
    half_off = p.resting_on(box((0.5, 0, 0.05), (0.03, 0.03, 0.05)), table)
    assert half_off["xy_overlap"] == pytest.approx(0.5)


def test_within_distance_in_plane_and_space(p):
    a, b = v(0, 0, 0), v(0.03, 0.04, 0.5)
    assert p.within_distance(a, b, 0.06, axes="xy")["near"]
    far = p.within_distance(a, b, 0.06)
    assert far["verdict"] == "far" and far["distance_m"] == pytest.approx(math.sqrt(0.0025 + 0.25))
    with pytest.raises(ValueError):
        p.within_distance(a, b, 0.1, axes="xz")


def test_upright_measures_tilt_of_the_chosen_axis(p):
    assert p.upright(yaw(70))["tilt_deg"] == pytest.approx(0, abs=1e-6)
    h = math.radians(30) / 2
    tilted = {"w": math.cos(h), "x": math.sin(h), "y": 0.0, "z": 0.0}
    result = p.upright(tilted, max_tilt_deg=15)
    assert result["verdict"] == "tipped" and result["tilt_deg"] == pytest.approx(30)
    assert p.upright(tilted, max_tilt_deg=15, axis="x")["tilt_deg"] == pytest.approx(90)


RING = {
    "kind": "loop",
    "pose": {"position": v(0, 0, 0.1), "rotation": IDENTITY},
    "axis": v(1, 0, 0),
    "radius_inner": 0.02,
}


def rope(points, radius=0.0):
    return {"points": [list(q) for q in points], "arclength_m": 0.0, "radius_m": radius, "nodes": len(points)}


def line(x0, x1, y=0.0, z=0.1, n=41):
    return [(x0 + (x1 - x0) * i / (n - 1), y, z) for i in range(n)]


def test_a_rope_through_the_ring_is_threaded_either_way(p):
    forward = p.threaded_through(rope(line(-0.1, 0.1)), RING)
    assert forward["verdict"] == "threaded" and forward["crossings"] == 1
    assert forward["through_m"] == pytest.approx(0.1)
    assert forward["clearance_m"] == pytest.approx(0.02)
    backward = p.threaded_through(rope(line(0.1, -0.1)), RING)
    assert backward["threaded"] and backward["crossings"] == -1


def test_a_rope_beside_the_ring_misses_it_by_its_offset(p):
    beside = p.threaded_through(rope(line(-0.1, 0.1, y=0.05)), RING)
    assert beside["verdict"] == "not_threaded" and beside["passes"] == 0
    assert beside["clearance_m"] == pytest.approx(-0.03)
    short = p.threaded_through(rope(line(-0.2, -0.05)), RING)
    assert not short["threaded"] and short["clearance_m"] == pytest.approx(-0.05)


def test_through_and_back_nets_zero(p):
    there = line(-0.1, 0.05)
    back = line(0.05, -0.1, y=0.01)
    result = p.threaded_through(rope(there + back), RING)
    assert result["passes"] == 2 and result["crossings"] == 0
    assert result["verdict"] == "not_threaded" and result["through_m"] == 0.0


def test_a_tip_at_the_plane_is_not_through(p):
    tip = p.threaded_through(rope(line(-0.1, 0.005)), RING)
    assert tip["crossings"] == 1 and not tip["threaded"]
    assert tip["through_m"] == pytest.approx(0.005)
    assert p.threaded_through(rope(line(-0.1, 0.005)), RING, min_through_m=0.004)["threaded"]


def test_the_rope_radius_takes_from_the_clearance(p):
    thick = p.threaded_through(rope(line(-0.1, 0.1, y=0.01), radius=0.004), RING)
    assert thick["threaded"] and thick["clearance_m"] == pytest.approx(0.006)


def test_threading_needs_a_rope_and_a_loop_with_an_opening(p):
    with pytest.raises(ValueError):
        p.threaded_through(rope([(0, 0, 0.1)]), RING)
    with pytest.raises(ValueError):
        p.threaded_through(rope(line(-0.1, 0.1)), {**RING, "radius_inner": 0.0})
