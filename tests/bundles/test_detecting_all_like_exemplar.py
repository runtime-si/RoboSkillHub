"""detecting-all-like-exemplar: routing over fake geco / owlv2 tools; the geco bundle's guards."""

import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

OBS = {"cameras": [{"name": "top", "rgb": np.zeros((48, 64, 3), np.uint8)}]}


@pytest.fixture(scope="module")
def skill(skills_registry):
    return (
        skills_registry.get("detecting-all-like-exemplar").canonical_scripts["detect_like"].module
    )


def test_geco_found_and_count(skill):
    calls = []

    def geco(**kw):
        calls.append(kw)
        return {"boxes": [[1, 1, 5, 5], [10, 10, 14, 14]], "scores": [0.9, 0.8], "count": 2}

    out = skill.run(
        FakeContext({"geco.detect_like": geco, "robot.get_observation": OBS}),
        exemplar_boxes=[[1, 1, 5, 5]] * 4,
        camera="top",
    )
    assert out["route"] == "found" and out["count"] == 2
    assert len(calls[0]["exemplar_boxes"]) == 3 and calls[0]["image"].shape == (48, 64, 3)


def test_owlv2_backend_reads_box_dicts(skill):
    owl = lambda **kw: {"detections": [{"box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}, "score": 0.9}]}  # noqa: E731
    out = skill.run(
        FakeContext({"owlv2.detect_like": owl}),
        exemplar_boxes=[[1, 2, 3, 4]],
        observation=OBS,
        backend="owlv2",
    )
    assert out["boxes"] == [[1, 2, 3, 4]] and out["route"] == "found"


def test_not_found_paths(skill):
    out = skill.run(
        FakeContext({"geco.detect_like": lambda **kw: {"boxes": [], "scores": [], "count": 0}}),
        exemplar_boxes=[[0, 0, 2, 2]],
        observation=OBS,
    )
    assert out["route"] == "not_found"
    assert (
        skill.run(FakeContext({}), exemplar_boxes=[[0, 0, 2, 2]], observation={"cameras": []})[
            "route"
        ]
        == "not_found"
    )

    def boom(**kw):
        raise RuntimeError("no weights")

    out = skill.run(
        FakeContext({"geco.detect_like": boom}), exemplar_boxes=[[0, 0, 2, 2]], observation=OBS
    )
    assert out["route"] == "not_found" and "no weights" in out["reason"]
    with pytest.raises(ValueError):
        skill.run(FakeContext({}), exemplar_boxes=[], observation=OBS)
    with pytest.raises(ValueError, match="backend"):
        skill.run(FakeContext({}), exemplar_boxes=[[0, 0, 1, 1]], observation=OBS, backend="yolo")


def test_geco_bundle_guards(skills_registry, monkeypatch, tmp_path):
    mod = skills_registry.get("geco").tools_module
    assert "torch" not in vars(mod)
    with pytest.raises(ValueError, match="1-3"):
        mod.detect_like(np.zeros((4, 4, 3), np.uint8), [])
    with pytest.raises(ValueError, match="uint8"):
        mod.detect_like(np.zeros((4, 4, 3), np.float32), [[0, 0, 1, 1]])


def test_example_routes(skills_registry):
    b = skills_registry.get("detecting-all-like-exemplar")
    ex = json.loads((Path(b.bundle_dir) / "examples" / "canonical_subgraph.json").read_text())
    assert set(ex["exit"]["success_values"]) | {ex["on_error"]} == {"found", "not_found"}
