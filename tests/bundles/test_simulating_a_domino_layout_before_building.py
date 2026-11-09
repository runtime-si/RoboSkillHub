"""simulating-a-domino-layout-before-building: inputs, trigger geometry, physics verdicts, exits.

The physics tests need the mujoco package and skip without it; the rest run anywhere.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
from gap.testing import FakeContext

BUNDLE = "simulating-a-domino-layout-before-building"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["simulate_layout"].module


@pytest.fixture(scope="module")
def mujoco():
    return pytest.importorskip("mujoco")


def _line(n, pitch, y=0.3, z=None):
    return [[k * pitch, y, 0.0] if z is None else _pose(k * pitch, y, z, 0.0) for k in range(n)]


def _pose(x, y, z, yaw_deg):
    a = math.radians(yaw_deg) / 2
    return {
        "position": {"x": x, "y": y, "z": z},
        "rotation": {"w": math.cos(a), "x": 0.0, "y": 0.0, "z": math.sin(a)},
    }


def _run(script, slots, **kw):
    kw.setdefault("table_z", 0.0)
    kw.setdefault("trials", 0)
    return script.run(FakeContext(), slots=slots, **kw)


# --- geometry, no engine -----------------------------------------------------------------


def test_tip_speed_is_the_planners_lone_tile_speed(script):
    # planning-a-domino-layout: 2.69 rad/s just tips a 48 x 7.5 mm tile
    assert script.tip_speed(0.048, 0.0075) == pytest.approx(2.69, abs=0.01)


def test_tip_travel_matches_the_trigger_skill(script):
    # triggering-a-domino-chain-by-push: 5.3 mm at 0.7 h for 48 x 7.5 mm
    assert script.tip_travel(0.048, 0.0075, 0.7 * 0.048) == pytest.approx(0.0053, abs=1e-4)


def test_slot_forms(script):
    assert script._slot([0.1, 0.2, 90.0])[:3] == pytest.approx((0.1, 0.2, math.pi / 2))
    x, y, yaw, z = script._slot(_pose(0.1, 0.2, 0.01, 30.0))
    assert (x, y, z) == (0.1, 0.2, 0.01) and yaw == pytest.approx(math.radians(30))
    planner = {"index": 0, "position": {"x": 0.0, "y": 0.0, "z": 0.0}, "facing_deg": -45.0}
    assert script._slot(planner)[2] == pytest.approx(math.radians(-45))
    assert script._slot({"pose": _pose(0, 0, 0, 10)})[2] == pytest.approx(math.radians(10))
    with pytest.raises(ValueError):
        script._slot({"position": [0, 0, 0]})


def test_scene_reads_angles_in_radians(script):
    xml = script.build_xml([(0.0, 0.0, 1.0)], 0.0, dict(script.TILE_SIZE), 0.01, 0.5, 0.25, 0.0005)
    assert '<compiler angle="radian"/>' in xml and 'euler="0 0 1.000000"' in xml


@pytest.mark.parametrize(
    "kw",
    [
        {"slots": [[0, 0, 0]]},
        {"slots": [[0, 0, 0], [0.004, 0, 0]]},
        {"slots": [{"position": [0, 0, 0]}, [0.04, 0, 0]]},
        {"slots": _line(3, 0.04), "table_z": None},
        {"slots": _line(3, 0.04), "target_index": 0},
        {"slots": _line(3, 0.04), "target_index": 5},
        {"slots": _line(3, 0.04), "tile_size": {"width": 0.06}},
        {"slots": _line(3, 0.04), "trigger": "flick"},
        {"slots": _line(3, 0.04), "push_speed": 0.0},
        {"slots": _line(3, 0.04), "contact_fraction": 0.1},
    ],
)
def test_invalid_input(script, kw):
    slots = kw.pop("slots")
    kw.setdefault("table_z", 0.0)
    assert script.run(FakeContext(), slots=slots, trials=0, **kw)["route"] == "invalid_input"


def test_table_height_from_the_slots(script, monkeypatch):
    monkeypatch.setitem(sys.modules, "mujoco", None)
    res = script.run(FakeContext(), slots=_line(3, 0.04, z=0.02), trials=0)
    assert res["route"] == "sim_unavailable"  # parsed, then no engine


def test_push_too_tight_for_the_trigger_skill(script):
    res = _run(script, _line(6, 0.016))
    assert res["route"] == "push_failed" and "kick" in res["detail"]
    assert res["n_tiles"] == 6


def test_sim_unavailable_without_the_engine(script, monkeypatch):
    monkeypatch.setitem(sys.modules, "mujoco", None)
    res = _run(script, _line(5, 0.037))
    assert res["route"] == "sim_unavailable"


def test_jitter_never_overlaps_neighbours(script):
    import random

    rng = random.Random(0)
    slots = [(k * 0.012, 0.0, 0.0) for k in range(30)]
    for _ in range(20):
        jit = script._jitter(slots, rng, 0.004, math.radians(5), 0.0075, 0.024)
        feet = [script._footprint(*s, 0.0075, 0.024) for s in jit]
        assert not any(script._overlap(a, b) for a, b in zip(feet, feet[1:], strict=False))


def test_example_declares_every_route(script):
    ex = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert declared == {
        "cascades",
        "fragile",
        "stalls",
        "push_failed",
        "sim_unavailable",
        "invalid_input",
    }


# --- physics -----------------------------------------------------------------------------


def test_line_at_the_planners_pitch_cascades(script, mujoco):
    res = _run(script, _line(10, 0.037), trials=5, seed=1)
    assert res["route"] == "cascades" and res["reached_target"]
    assert res["n_fallen"] == 10 and res["stall_at"] == -1 and res["success_rate"] == 1.0
    times = res["topple_times"]
    assert all(b > a for a, b in zip(times, times[1:], strict=False))
    assert 0.5 < res["wave_speed"] < 1.0
    assert res["push_travel"] == pytest.approx(0.008 + 0.0053 + 0.008, abs=2e-4)


def test_line_past_the_critical_gap_stalls_at_slot_one(script, mujoco):
    res = _run(script, _line(6, 0.072))
    assert res["route"] == "stalls" and res["stall_at"] == 1
    assert res["fallen"][0] and not res["fallen"][1]


def test_kick_stalls_where_the_planner_says_a_slow_push_does(script, mujoco):
    assert _run(script, _line(8, 0.016), trigger="kick")["stall_at"] == 1
    assert _run(script, _line(8, 0.020), trigger="kick")["route"] == "cascades"


def test_target_index_judges_only_up_to_the_target(script, mujoco):
    slots = _line(4, 0.037) + [[3 * 0.037 + 0.08, 0.3, 0.0]]
    assert _run(script, slots)["route"] == "stalls"
    res = _run(script, slots, target_index=3)
    assert res["route"] == "cascades" and res["target_index"] == 3


def test_turned_slot_stands_at_its_facing(script, mujoco):
    size = dict(script.TILE_SIZE)
    m = mujoco.MjModel.from_xml_string(
        script.build_xml([(0.0, 0.0, math.radians(60))], 0.0, size, 0.01, 0.5, 0.25, 0.0005)
    )
    d = mujoco.MjData(m)
    for _ in range(200):
        mujoco.mj_step(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "d0")
    r = d.xmat[b].reshape(3, 3)
    assert math.degrees(math.atan2(r[1, 0], r[0, 0])) == pytest.approx(60.0, abs=0.5)
    assert r[2, 2] == pytest.approx(1.0, abs=1e-4)


def test_curve_from_turned_slots_cascades(script, mujoco):
    radius, pitch = 0.25, 0.0369
    slots = [
        _pose(
            radius * math.cos(-math.pi / 2 + k * pitch / radius),
            0.3 + radius * math.sin(-math.pi / 2 + k * pitch / radius),
            0.0,
            math.degrees(k * pitch / radius),
        )
        for k in range(10)
    ]
    assert _run(script, slots)["route"] == "cascades"


def test_wide_line_with_a_big_placing_error_is_fragile(script, mujoco):
    res = _run(script, _line(8, 0.052), trials=10, position_sigma=0.0054, yaw_sigma_deg=5, seed=2)
    assert res["route"] == "fragile"
    assert res["reached_target"] and res["success_rate"] < 0.9
    assert len(res["trial_stalls"]) == 10 and res["weakest_pair"] >= 0


def test_weak_kick_fails_to_trigger(script, mujoco):
    res = _run(script, _line(5, 0.037), trigger="kick", kick_factor=0.5)
    assert res["route"] == "push_failed" and not res["first_fell"]
