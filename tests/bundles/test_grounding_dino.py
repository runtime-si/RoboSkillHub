"""grounding-dino bundle: signature/schema units + GPU smoke."""

from __future__ import annotations

import numpy as np
import pytest


def test_detect_registered(tool_registry):
    assert "grounding-dino.detect" in tool_registry
    desc = tool_registry.get("grounding-dino.detect")
    assert desc.tags == ("perception",)


def test_detect_schema(tool_registry):
    schema = tool_registry.get("grounding-dino.detect").schema
    assert set(schema.inputs) == {"image", "query", "box_threshold", "text_threshold"}
    assert schema.inputs["image"].required
    assert schema.inputs["query"].required
    assert schema.inputs["box_threshold"].default == pytest.approx(0.20)
    assert schema.inputs["text_threshold"].default == pytest.approx(0.20)
    assert set(schema.outputs) == {"detections"}


@pytest.mark.gpu
def test_detect_gpu_smoke(tool_registry):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    img = np.full((160, 160, 3), 255, dtype=np.uint8)
    img[40:120, 40:120] = (200, 30, 30)
    out = tool_registry.invoke("grounding-dino.detect", image=img, query="red square")
    assert set(out) == {"detections"}
    for det in out["detections"]:
        assert set(det) == {"box", "label", "score"}
        assert 0.0 <= det["score"] <= 1.0
        box = det["box"]
        assert box["x2"] >= box["x1"] and box["y2"] >= box["y1"]


# --- transformers 5.x box-head repair and the load guard -------------------------------


@pytest.fixture
def gd(skills_registry):
    return skills_registry.get("grounding-dino").tools_module


def test_default_model_is_pinned(gd):
    rev = gd._PINNED[gd._DEFAULT_MODEL_NAME]
    assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def test_head_tensors_group_both_namings(gd):
    ck = {
        "model.decoder.bbox_embed.0.layers.0.weight": 1,
        "model.decoder.bbox_embed.0.layers.2.bias": 2,
        "bbox_embed.3.layers.0.weight": 3,
        "model.encoder_output_bbox_embed.layers.0.weight": 4,
        "model.decoder.layers.0.self_attn.weight": 5,
    }
    assert gd._head_tensors(ck) == {
        0: {"layers.0.weight": 1, "layers.2.bias": 2},
        3: {"layers.0.weight": 3},
    }


def test_head_plan_shared_and_per_layer(gd):
    from gap_core.errors import PerceptionFailed

    assert gd._head_plan(True, 6, {0: {}}) == [0] * 6
    assert gd._head_plan(False, 3, {0: {}, 1: {}, 2: {}}) == [0, 1, 2]
    assert gd._head_plan(True, 6, {}) == []
    with pytest.raises(PerceptionFailed, match="head 0"):
        gd._head_plan(True, 6, {1: {}})
    with pytest.raises(PerceptionFailed, match=r"\[2\]"):
        gd._head_plan(False, 3, {0: {}, 1: {}})


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self


@pytest.fixture
def fake_load(gd, monkeypatch):
    """transformers faked; the checkpoint check and repair scripted per test."""
    import sys
    import types

    fake = types.ModuleType("transformers")
    calls = {}

    class _Auto:
        @classmethod
        def from_pretrained(cls, name, revision=None):
            calls.setdefault("revisions", []).append((name, revision))
            return _FakeModel()

    fake.AutoProcessor = _Auto
    fake.AutoModelForZeroShotObjectDetection = _Auto
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(gd, "_model", None)
    monkeypatch.setattr(gd, "_processor", None)
    monkeypatch.setattr(gd, "_checkpoint_tensors", lambda name, rev: {"k": 1})
    return calls


def test_a_clean_load_is_not_touched(gd, fake_load, monkeypatch):
    monkeypatch.setattr(gd, "_verify_against_checkpoint", lambda m, ck: [])
    monkeypatch.setattr(gd, "_repair_box_heads", lambda m, ck: pytest.fail("repaired a clean load"))
    gd._get_model()
    assert fake_load["revisions"][0] == (gd._MODEL_NAME, gd._REVISION)


def test_a_wrong_box_head_is_repaired_then_checked(gd, fake_load, monkeypatch):
    answers = [["model.decoder.bbox_embed.0.layers.0.weight"], []]
    monkeypatch.setattr(gd, "_verify_against_checkpoint", lambda m, ck: answers.pop(0))
    repaired = []
    monkeypatch.setattr(gd, "_repair_box_heads", lambda m, ck: repaired.append(1) or True)
    gd._get_model()
    assert repaired == [1] and answers == []


def test_a_load_still_wrong_after_repair_is_refused(gd, fake_load, monkeypatch):
    from gap_core.errors import PerceptionFailed

    monkeypatch.setattr(gd, "_verify_against_checkpoint", lambda m, ck: ["model.x.weight"])
    monkeypatch.setattr(gd, "_repair_box_heads", lambda m, ck: True)
    with pytest.raises(PerceptionFailed, match="differ from its checkpoint"):
        gd._get_model()
    assert gd._model is None


def test_an_unreadable_checkpoint_skips_the_check(gd, fake_load, monkeypatch):
    def boom(name, rev):
        raise FileNotFoundError("no model.safetensors")

    monkeypatch.setattr(gd, "_checkpoint_tensors", boom)
    monkeypatch.setattr(gd, "_verify_against_checkpoint", lambda m, ck: ["x"] if ck else [])
    model, _ = gd._get_model()
    assert isinstance(model, _FakeModel)
