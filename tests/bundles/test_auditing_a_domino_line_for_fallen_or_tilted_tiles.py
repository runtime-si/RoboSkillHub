"""auditing-a-domino-line-for-fallen-or-tilted-tiles: per-tile states from one synthetic overhead depth frame."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from gap.testing import FakeContext

from . import _domino_depth_scene as S

BUNDLE = "auditing-a-domino-line-for-fallen-or-tilted-tiles"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["audit_line"].module


def _audit(script, slots, spec, noise=0.0, **kw):
    tiles = [
        S.tile_points(*slots[k], **(spec.get(k) or {}))
        for k in range(len(slots))
        if spec.get(k, {}) is not None
    ]
    ys = [s[1] for s in slots]
    frame = S.depth_frame(tiles, centre=(slots[0][0], (min(ys) + max(ys)) / 2), noise=noise)
    return script.run(FakeContext(), slots=slots, table_z=0.0, frame=frame, **kw)


def test_upright_line_is_ok(script):
    res = _audit(script, S.line_slots(6, 0.037), {})
    assert res["route"] == "line_ok" and res["states"] == ["upright"] * 6
    assert res["n_upright"] == 6 and res["first_fault"] == -1
    assert all(abs(a) < 2.0 for a in res["lean_deg"])


@pytest.mark.parametrize("noise", [0.0, 0.001])
def test_every_fault_state(script, noise):
    slots = S.line_slots(8, 0.037)
    c = S.contact_deg(0.037)
    spec = {
        1: {"shift": 0.006},
        2: {"lean_deg": c},
        4: {"lean_deg": 12.0},
        5: {"yaw_deg": 15.0},
        6: None,
        7: {"lean_deg": 90.0},
    }
    res = _audit(script, slots, spec, noise)
    assert res["states"] == [
        "upright",
        "displaced",
        "leaning",
        "upright",
        "tilted",
        "displaced",
        "missing",
        "fallen",
    ]
    assert res["route"] == "fallen" and res["first_fault"] == 1
    assert res["lean_deg"][2] == pytest.approx(c, abs=3.0)
    assert res["lean_deg"][4] == pytest.approx(12.0, abs=1.5)
    assert res["shift_mm"][1] == pytest.approx(6.0, abs=2.5)
    assert res["yaw_error_deg"][5] == pytest.approx(15.0, abs=2.0)
    assert res["fallen_runs"] == [[2, 2], [7, 7]]


def test_tile_behind_a_leaner_reads_upright(script):
    slots = S.line_slots(4, 0.037)
    res = _audit(script, slots, {1: {"lean_deg": S.contact_deg(0.037)}})
    assert res["states"][1] == "leaning" and res["states"][2] == "upright"


def test_backward_lean(script):
    res = _audit(script, S.line_slots(4, 0.037), {2: {"lean_deg": -12.0}})
    assert res["states"][2] == "tilted" and res["lean_deg"][2] == pytest.approx(-12.0, abs=1.5)
    assert res["faults"][0]["direction"] == "backward"


def test_lateral_offset_is_displaced(script):
    res = _audit(script, S.line_slots(4, 0.037), {2: {"lateral": 0.006}})
    assert res["states"][2] == "displaced" and res["route"] == "fix_needed"
    assert abs(res["lateral_mm"][2]) == pytest.approx(6.0, abs=1.5)


def test_missing_slot_under_the_tile_behind_lying_flat(script):
    # tile 2 lies flat across slot 3 (empty); slot 4 stands 30 mm on
    res = _audit(script, S.line_slots(6, 0.030), {2: {"lean_deg": 90.0}, 3: None})
    assert res["states"][:5] == ["upright", "upright", "fallen", "missing", "upright"]


def test_a_gap_the_wave_cannot_cross(script):
    res = _audit(script, S.line_slots(4, 0.060), {})
    assert res["route"] == "fix_needed" and res["states"] == ["upright"] * 4
    assert [g["pair"] for g in res["wide_gaps"]] == [[0, 1], [1, 2], [2, 3]]


def test_reads_the_observation_and_slot_forms(script):
    slots = S.line_slots(3, 0.037)
    frame = S.depth_frame([S.tile_points(*s) for s in slots], centre=(0.0, 0.237))
    ctx = FakeContext({"robot.get_observation": {"cameras": [frame]}})
    poses = [
        {
            "position": {"x": s[0], "y": s[1], "z": 0.0},
            "rotation": {"w": 0.7071068, "x": 0, "y": 0, "z": 0.7071068},
        }
        for s in slots
    ]
    assert script.run(ctx, slots=poses)["route"] == "line_ok"
    planner = [{"position": {"x": s[0], "y": s[1], "z": 0.0}, "facing_deg": 90.0} for s in slots]
    assert script.run(ctx, slots=planner, table_z=0.0)["route"] == "line_ok"


def test_not_seen(script):
    assert (
        script.run(
            FakeContext({"robot.get_observation": {"cameras": []}}),
            slots=S.line_slots(3, 0.037),
            table_z=0.0,
        )["route"]
        == "not_seen"
    )
    far = S.depth_frame([], centre=(1.0, 1.0))
    assert (
        script.run(FakeContext(), slots=S.line_slots(3, 0.037), table_z=0.0, frame=far)["route"]
        == "not_seen"
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"slots": []},
        {"slots": [[0, 0]]},
        {"slots": S.line_slots(2, 0.04), "tile_size": (0.02, 0.03, 0.01)},
    ],
)
def test_invalid_input(script, kw):
    assert script.run(FakeContext(), table_z=0.0, **kw)["route"] == "invalid_input"


def test_example_declares_every_route(script):
    ex = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert declared == {"line_ok", "fix_needed", "fallen", "not_seen", "invalid_input"}
