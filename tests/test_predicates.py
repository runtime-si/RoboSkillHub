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
