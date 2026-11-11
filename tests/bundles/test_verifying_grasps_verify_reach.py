"""verifying-grasps/verify_reach: the old two routes unchanged, ``stopped_short`` only with ``approach_start``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "skills" / "verifying-grasps" / "scripts" / "verify_reach.py"


@pytest.fixture(scope="module")
def vr():
    spec = importlib.util.spec_from_file_location("verify_reach", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CAP = {"center": {"x": 0, "y": 0, "z": 0}, "extent": {"x": 0.00925, "y": 0.00925, "z": 0.015}}
GOAL = {"position": {"x": 0.30, "y": 0.30, "z": 0.08}}
START = {"x": 0.30, "y": 0.30, "z": 0.12}  # straight down onto the cap


def test_old_routes_without_approach(vr):
    ok = vr.run(None, commanded=GOAL, observed=[0.302, 0.30, 0.08], target_obb=CAP)
    assert ok["route"] == "reached" and ok["shortfall_m"] == -1.0
    off = vr.run(None, commanded=GOAL, observed=[0.30, 0.30, 0.0905], target_obb=CAP)
    assert off["route"] == "missed"  # 10.5 mm over a 9.25 mm half-extent: as before


def test_stopped_short_on_top(vr):
    out = vr.run(None, commanded=GOAL, observed=[0.301, 0.30, 0.0905], target_obb=CAP, approach_start=START)
    assert out["route"] == "stopped_short" and not out["reached"]
    assert out["shortfall_m"] == pytest.approx(0.0105, abs=1e-6)
    assert out["shortfall_vector"]["z"] == pytest.approx(0.0105, abs=1e-6)
    assert out["lateral_m"] == pytest.approx(0.001, abs=1e-6)


def test_beside_is_still_missed(vr):
    out = vr.run(None, commanded=GOAL, observed=[0.315, 0.30, 0.085], target_obb=CAP, approach_start=START)
    assert out["route"] == "missed" and out["lateral_m"] > 0.00925


def test_small_shortfall_is_reached(vr):
    out = vr.run(None, commanded=GOAL, observed=[0.30, 0.30, 0.082], target_obb=CAP, approach_start=START)
    assert out["route"] == "reached" and out["shortfall_m"] == pytest.approx(0.002, abs=1e-6)
