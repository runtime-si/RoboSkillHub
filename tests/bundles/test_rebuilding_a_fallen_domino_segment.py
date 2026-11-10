"""rebuilding-a-fallen-domino-segment: the next tile to rebuild from an audit."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from gap.testing import FakeContext

BUNDLE = "rebuilding-a-fallen-domino-segment"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
SLOTS = [[0.0, 0.2 + k * 0.03, 90.0] for k in range(8)]


@pytest.fixture(scope="module")
def script(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["plan_rebuild"].module


def _run(script, states, leans, **kw):
    return script.run(
        FakeContext(),
        slots=kw.pop("slots", SLOTS),
        states=states,
        lean_deg=leans,
        table_z=0.0,
        **kw,
    )


def test_downstream_end_of_a_forward_run(script):
    states = ["upright", "upright", "fallen", "fallen", "fallen", "missing", "upright", "upright"]
    leans = [0.0, 0.0, 76.0, 76.0, 90.0, None, 0.0, 0.0]
    res = _run(script, states, leans)
    assert res["route"] == "rebuild_next" and res["tile_index"] == 4 and res["segment"] == [2, 4]
    assert res["direction"] == "forward" and res["remaining"] == 3
    # lying flat forward of slot 4: centre h/2 + d/2 ahead of the slot centre, thickness/2 high
    assert res["tile_center"]["y"] == pytest.approx(0.2 + 4 * 0.03 + 0.0075 / 2 + 0.024)
    assert res["tile_center"]["z"] == pytest.approx(0.0075 / 2)
    assert res["long_axis"]["y"] == pytest.approx(1.0)
    assert res["slot"]["position"]["y"] == pytest.approx(0.32)


def test_backward_run_starts_at_its_low_end(script):
    states = ["upright", "fallen", "fallen", "upright"]
    res = _run(script, states, [0.0, -90.0, -78.0, 0.0], slots=SLOTS[:4])
    assert res["direction"] == "backward" and res["tile_index"] == 1
    assert res["long_axis"]["y"] == pytest.approx(-1.0)


def test_downstream_tile_still_leaning(script):
    res = _run(
        script, ["upright", "fallen", "leaning", "upright"], [0.0, 78.0, 38.0, 0.0], slots=SLOTS[:4]
    )
    assert res["route"] == "not_flat" and res["tile_index"] == 2


def test_nothing_fallen(script):
    res = _run(script, ["upright", "missing", "upright"], [0.0, None, 0.0], slots=SLOTS[:3])
    assert res["route"] == "nothing_fallen" and "missing slots [1]" in res["detail"]


def test_shift_moves_the_lying_centre(script):
    states = ["upright", "fallen", "upright"]
    a = _run(script, states, [0.0, 90.0, 0.0], slots=SLOTS[:3])
    b = _run(script, states, [0.0, 90.0, 0.0], slots=SLOTS[:3], shift_mm=[0, 4.0, 0])
    assert b["tile_center"]["y"] - a["tile_center"]["y"] == pytest.approx(0.004)
    assert math.isfinite(a["tile_center"]["x"])


@pytest.mark.parametrize("kw", [{"slots": []}, {"slots": SLOTS[:2]}])
def test_invalid_input(script, kw):
    assert _run(script, ["upright"] * 3, [0.0] * 3, **kw)["route"] == "invalid_input"


def test_example_declares_every_route(script):
    ex = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    assert set(ex["exit"]["success_values"]) | {ex["on_error"]} == {
        "rebuild_next",
        "nothing_fallen",
        "not_flat",
        "invalid_input",
    }


def test_flat_body_read_across_the_empty_next_slot(script):
    # the audit in the station twin: tile 4 read 67 deg, its body read flat at the empty slot 5
    states = ["upright", "upright", "fallen", "fallen", "fallen", "missing", "upright", "upright"]
    res = _run(script, states, [2.5, 2.5, 75.5, None, 67.0, 90.0, 0.5, 0.0])
    assert res["route"] == "rebuild_next" and res["tile_index"] == 4
