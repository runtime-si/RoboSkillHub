"""mm-grounding-dino bundle: query building, the transformers-5 guard, thresholds, ordering and
the detection shape, with MM-Grounding-DINO faked."""

from __future__ import annotations

import numpy as np
import pytest
from gap_core.errors import PerceptionFailed, ToolError

H, W = 480, 640


@pytest.fixture(scope="module")
def mmg(skills_registry):
    return skills_registry.get("mm-grounding-dino").tools_module


class FakeRuntime:
    def __init__(self, boxes=None, scores=None, labels=None):
        self.boxes = np.array(
            boxes if boxes is not None else [[10, 20, 30, 40], [100, 100, 160, 150]], float
        )
        self.scores = np.array(scores if scores is not None else [0.35, 0.8])
        self.labels = labels if labels is not None else ["cube", "open box"]
        self.calls: list[tuple[str, float, float]] = []

    def __call__(self, image, text, box_thr, text_thr):
        self.calls.append((text, box_thr, text_thr))
        keep = self.scores > box_thr
        return {
            "boxes": self.boxes[keep],
            "scores": self.scores[keep],
            "labels": [lab for lab, k in zip(self.labels, keep, strict=True) if k],
        }


@pytest.fixture
def fake(mmg, monkeypatch):
    rt = FakeRuntime()
    monkeypatch.setattr(mmg, "_get_runtime", lambda: rt)
    return rt


def frame():
    return np.zeros((H, W, 3), dtype=np.uint8)


def test_tool_registered_with_grounding_dinos_inputs(tool_registry):
    t = tool_registry.get("mm-grounding-dino.detect")
    g = tool_registry.get("grounding-dino.detect")
    assert set(g.schema.inputs) <= set(t.schema.inputs)
    assert set(t.schema.inputs) == {
        "image",
        "query",
        "box_threshold",
        "text_threshold",
        "prompts",
        "max_detections",
    }
    assert t.schema.inputs["box_threshold"].default == pytest.approx(0.3)
    assert t.schema.inputs["text_threshold"].default == pytest.approx(0.2)
    assert set(t.schema.outputs) == {"detections"} == set(g.schema.outputs)
    assert t.tags == ("perception",)


def test_import_does_not_pull_torch_and_models_are_pinned(mmg):
    assert "torch" not in vars(mmg) and "transformers" not in vars(mmg)
    assert mmg.DEFAULT_MODEL in mmg.MODELS
    for rev in mmg.MODELS.values():
        assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def test_model_choice_is_checked(mmg, monkeypatch, tmp_path):
    monkeypatch.delenv("GAP_MMGDINO_MODEL", raising=False)
    assert mmg._model_source() == (mmg.DEFAULT_MODEL, mmg.MODELS[mmg.DEFAULT_MODEL])
    monkeypatch.setenv("GAP_MMGDINO_MODEL", "openmmlab-community/mm_grounding_dino_base_all")
    assert mmg._model_source()[1] == mmg.MODELS["openmmlab-community/mm_grounding_dino_base_all"]
    monkeypatch.setenv("GAP_MMGDINO_MODEL", str(tmp_path))
    assert mmg._model_source() == (str(tmp_path), None)
    assert mmg.weights_cached() is False
    monkeypatch.setenv("GAP_MMGDINO_MODEL", "IDEA-Research/grounding-dino-base")
    with pytest.raises(ValueError, match="must be one of"):
        mmg._model_source()


@pytest.mark.parametrize("version", ["5.0.0", "5.19.0", "6.1.0.dev0"])
def test_transformers_5_is_refused(mmg, version):
    with pytest.raises(PerceptionFailed, match="transformers < 5"):
        mmg._check_transformers(version)


@pytest.mark.parametrize("version", ["4.52.0", "4.57.1"])
def test_transformers_4_is_accepted(mmg, version):
    mmg._check_transformers(version)


@pytest.mark.parametrize(
    ("query", "prompts", "want"),
    [
        ("cube", None, "cube."),
        ("cube. open box.", None, "cube. open box."),
        (" cube. plug ", None, "cube. plug."),
        (None, ["cube", "open box."], "cube. open box."),
        (None, "cube. plug", "cube. plug."),
        ("ignored", ["plug"], "plug."),
    ],
)
def test_query_is_period_terminated_phrases(mmg, query, prompts, want):
    assert mmg._query(query, prompts) == want


def test_empty_queries_are_refused(mmg, fake):
    with pytest.raises(ToolError, match="give query"):
        mmg.detect(frame())
    with pytest.raises(ToolError, match="give query"):
        mmg.detect(frame(), " . ")
    with pytest.raises(ToolError, match="no phrase"):
        mmg.detect(frame(), prompts=[" ", "."])
    assert fake.calls == []


def test_detect_is_best_first_in_grounding_dinos_shape(mmg, fake):
    r = mmg.detect(frame(), "cube. open box.", box_threshold=0.2)
    assert [d["label"] for d in r["detections"]] == ["open box", "cube"]
    d = r["detections"][0]
    assert set(d) == {"box", "label", "score"}
    assert d["box"] == {"x1": 100.0, "y1": 100.0, "x2": 160.0, "y2": 150.0}
    assert d["score"] == pytest.approx(0.8)
    assert fake.calls[-1] == ("cube. open box.", 0.2, 0.2)


def test_thresholds_defaults_and_max_detections(mmg, fake):
    r = mmg.detect(frame(), "cube")
    assert [d["label"] for d in r["detections"]] == ["open box", "cube"]  # 0.35 > 0.3
    assert fake.calls[-1][1:] == (0.3, 0.2)
    mmg.detect(frame(), "cube", box_threshold=0, text_threshold=-1)
    assert fake.calls[-1][1:] == (0.3, 0.2)
    r = mmg.detect(frame(), "cube", box_threshold=0.5)
    assert len(r["detections"]) == 1
    r = mmg.detect(frame(), "cube", box_threshold=0.1, max_detections=1)
    assert [d["label"] for d in r["detections"]] == ["open box"]


def test_bad_images_and_model_errors(mmg, fake, monkeypatch):
    with pytest.raises(ToolError, match="RGB uint8"):
        mmg.detect(np.zeros((H, W), np.uint8), "cube")
    with pytest.raises(ToolError, match="RGB uint8"):
        mmg.detect(np.zeros((H, W, 3), np.float32), "cube")

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(mmg, "_get_runtime", lambda: boom)
    with pytest.raises(PerceptionFailed, match="out of memory"):
        mmg.detect(frame(), "cube")

    def refuse(*a, **k):
        raise PerceptionFailed("needs transformers < 5")

    monkeypatch.setattr(mmg, "_get_runtime", refuse)
    with pytest.raises(PerceptionFailed, match="transformers < 5"):
        mmg.detect(frame(), "cube")


class ToolCtx:
    def __init__(self, mmg):
        self.mmg, self.calls = mmg, []

    def tool(self, name, **kw):
        assert name == "mm-grounding-dino.detect"
        self.calls.append(kw)
        return self.mmg.detect(**kw)


def test_sahi_runs_it_through_its_generic_backend(mmg, fake):
    from gap.skills import load_skills

    from .conftest import SKILLS_ROOT

    sahi = load_skills(SKILLS_ROOT, only=["sahi"]).get("sahi").tools_module
    ctx = ToolCtx(mmg)
    r = sahi.detect_sliced(
        ctx, frame(), ["cube", "open box"], backend="mm-grounding-dino.detect", slice_px=0
    )
    assert ctx.calls[0]["query"] == "cube. open box"
    assert {d["label"] for d in r["detections"]} == {"cube", "open box"}
