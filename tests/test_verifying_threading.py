from __future__ import annotations

import importlib.util
import json

import pytest
from conftest import REPO

pytest.importorskip("gap_core")
pytest.importorskip("gap")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def skill():
    return _load("verify_threading", REPO / "skills/verifying-threading/scripts/verify_threading.py")


@pytest.fixture(scope="module")
def ctx():
    predicates = _load("predicates_tools", REPO / "tools/predicates/tools.py")

    class Context:
        def tool(self, name, **inputs):
            assert name == "predicates.threaded_through"
            return predicates.threaded_through(**inputs)

    return Context()


LOOP = {
    "kind": "loop",
    "pose": {
        "position": {"x": 0.0, "y": 0.0, "z": 0.1},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    },
    "axis": {"x": 1.0, "y": 0.0, "z": 0.0},
    "radius_inner": 0.02,
}


def points(x0, x1, y=0.0, n=21):
    return [[x0 + (x1 - x0) * i / (n - 1), y, 0.1] for i in range(n)]


def test_routes_threaded_from_the_json_the_rope_perception_writes(skill, ctx):
    out = skill.run(ctx, rope=json.dumps(points(-0.1, 0.1)), loop=LOOP)
    assert out["exit"] == "threaded" and out["threaded"] and out["crossings"] == 1
    assert out["detail"].startswith("through: 100 mm past the ring")
    centerline = {"points": points(-0.1, 0.1), "radius_m": 0.005}
    as_centerline = skill.run(ctx, rope=centerline, loop=json.dumps(LOOP))
    assert as_centerline["exit"] == "threaded" and as_centerline["clearance_m"] == pytest.approx(0.015)


def test_says_how_it_missed(skill, ctx):
    beside = skill.run(ctx, rope=json.dumps(points(-0.1, 0.1, y=0.05)), loop=LOOP)
    assert beside["exit"] == "not_threaded" and beside["detail"] == "misses the opening by 30 mm"
    back = skill.run(ctx, rope=json.dumps(points(-0.1, 0.05) + points(0.05, -0.1, y=0.01)), loop=LOOP)
    assert back["exit"] == "not_threaded" and "came back" in back["detail"]
    tip = skill.run(ctx, rope=json.dumps(points(-0.1, 0.004)), loop=LOOP)
    assert tip["exit"] == "not_threaded" and tip["detail"] == "only 4 mm past the ring, under 10 mm"


def test_no_rope_or_no_opening_is_not_threaded_without_calling_the_tool(skill):
    class Silent:
        def tool(self, name, **inputs):
            raise AssertionError("the predicate should not be called")

    assert skill.run(Silent(), rope="", loop=LOOP)["detail"] == "no rope centreline to judge"
    no_radius = {k: v for k, v in LOOP.items() if k != "radius_inner"}
    assert skill.run(Silent(), rope=json.dumps(points(-0.1, 0.1)), loop=no_radius)["exit"] == "not_threaded"
