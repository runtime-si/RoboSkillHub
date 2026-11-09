"""omdet-turbo bundle: prompt parsing, upstream's decode (top-k, floor, class-wise NMS, clip), the
detection shape and sahi as a caller, with OmDet-Turbo faked."""

from __future__ import annotations

import numpy as np
import pytest
from gap.skills import load_skills
from gap_core.errors import PerceptionFailed, ToolError

from .conftest import SKILLS_ROOT

H, W = 480, 640


@pytest.fixture(scope="module")
def om(skills_registry):
    return skills_registry.get("omdet-turbo").tools_module


@pytest.fixture(scope="module")
def sahi():
    return load_skills(SKILLS_ROOT, only=["sahi"]).get("sahi").tools_module


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


#: What the fake model sees, in frame pixels: label, box, probability for that label.
SCENE = [
    ("cube", (100, 300, 120, 320), 0.6),
    ("cube", (102, 301, 122, 321), 0.5),  # duplicate of the first: suppressed by NMS
    ("box", (200, 250, 260, 300), 0.45),
    ("plug", (400, 200, 416, 210), 0.15),
    ("cube", (630, 10, 660, 40), 0.4),  # hangs over the right edge: clipped
]


class FakeRuntime:
    """Queries = scene rows plus background rows; class logits -6 except a row's own label."""

    def __init__(self, scene=SCENE, n_background=3):
        self.scene, self.n_bg = list(scene), n_background
        self.calls: list[tuple[tuple[int, int], list[str], str]] = []

    def __call__(self, image, prompts, task):
        self.calls.append((image.shape[:2], list(prompts), task))
        h, w = image.shape[:2]
        rows = [*self.scene, *[("", (10, 10, 60, 40), 0.0)] * self.n_bg]
        logits = np.full((len(rows), len(prompts)), -6.0)
        boxes = []
        for i, (lab, (x1, y1, x2, y2), p) in enumerate(rows):
            if lab in prompts:
                logits[i, prompts.index(lab)] = logit(p)
            boxes.append([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h])
        return logits, np.array(boxes)


@pytest.fixture
def fake(om, monkeypatch):
    rt = FakeRuntime()
    monkeypatch.setattr(om, "_get_runtime", lambda: rt)
    return rt


def frame(h=H, w=W):
    return np.zeros((h, w, 3), dtype=np.uint8)


def xyxy(box):
    return [round(box[k], 3) for k in ("x1", "y1", "x2", "y2")]


# --- registration, pins, imports ----------------------------------------------------


def test_tool_registered_with_its_schema(tool_registry):
    t = tool_registry.get("omdet-turbo.detect")
    assert set(t.schema.inputs) == {
        "image",
        "prompts",
        "threshold",
        "query",
        "nms_iou",
        "task",
        "max_detections",
    }
    assert t.schema.inputs["image"].required and not t.schema.inputs["prompts"].required
    assert t.schema.inputs["threshold"].default == pytest.approx(0.2)
    assert t.schema.inputs["nms_iou"].default == pytest.approx(0.5)
    assert set(t.schema.outputs) == {"detections"}
    assert t.tags == ("perception",)


def test_import_does_not_pull_torch_and_the_model_is_pinned(om):
    assert "torch" not in vars(om) and "transformers" not in vars(om)
    assert om.DEFAULT_MODEL in om.MODELS
    for rev in om.MODELS.values():
        assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def test_model_choice_is_checked(om, monkeypatch, tmp_path):
    monkeypatch.delenv("GAP_OMDET_MODEL", raising=False)
    assert om._model_source() == (om.DEFAULT_MODEL, om.MODELS[om.DEFAULT_MODEL])
    monkeypatch.setenv("GAP_OMDET_MODEL", str(tmp_path))
    assert om._model_source() == (str(tmp_path), None)
    assert om.weights_cached() is False  # no config.json there
    (tmp_path / "config.json").write_text("{}")
    assert om.weights_cached() is True
    monkeypatch.setenv("GAP_OMDET_MODEL", "someone/omdet-finetune")
    with pytest.raises(ValueError, match="must be one of"):
        om._model_source()
    assert om.weights_cached() is False


# --- prompts --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompts", "query", "want"),
    [
        (["cube", "plug"], None, ["cube", "plug"]),
        ("cube. plug.", None, ["cube", "plug"]),
        ("cube, plug", None, ["cube", "plug"]),
        (None, "red brick. blue brick", ["red brick", "blue brick"]),
        ([" cube ", "cube", "plug."], None, ["cube", "plug"]),
    ],
)
def test_prompts_accept_lists_and_grounding_dino_strings(om, prompts, query, want):
    assert om._prompts(prompts, query) == want


def test_missing_or_empty_prompts_are_refused(om, fake):
    with pytest.raises(ToolError, match="give prompts"):
        om.detect(frame())
    with pytest.raises(ToolError, match="no phrase"):
        om.detect(frame(), " . ")
    assert fake.calls == []


# --- decode ---------------------------------------------------------------------------


def test_decode_matches_upstream_order_floor_nms_and_clip(om):
    rt = FakeRuntime()
    logits, boxes = rt(frame(), ["cube", "box", "plug"], "Detect cube, box, plug.")
    b, s, c = om._decode(logits, boxes, 3, W, H, threshold=0.2, nms_iou=0.5)
    assert s.tolist() == pytest.approx([0.6, 0.45, 0.4])
    assert c.tolist() == [0, 1, 0]
    assert b[0] == pytest.approx([100, 300, 120, 320])
    assert b[2] == pytest.approx([630, 10, 640, 40])  # clipped after NMS
    # NMS is per class: the duplicate cube goes, a same-place box of another class would stay
    b2, s2, _ = om._decode(logits, boxes, 3, W, H, threshold=0.2, nms_iou=1.0)
    assert len(b2) == 4 and s2[1] == pytest.approx(0.5)


def test_decode_ignores_padded_class_columns(om):
    logits = np.array([[logit(0.9), logit(0.95)], [logit(0.3), -6.0]])
    boxes = np.array([[0.5, 0.5, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1]])
    _, s, c = om._decode(logits, boxes, 1, W, H, threshold=0.1, nms_iou=0.5)
    assert c.tolist() == [0, 0] and s.tolist() == pytest.approx([0.9, 0.3])


def test_nms_suppresses_per_label_best_first(om):
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([0, 0, 1, 0])
    assert om._nms(boxes, scores, labels, 0.5).tolist() == [0, 2, 3]
    assert om._nms(boxes, scores, labels, 1.0).tolist() == [0, 1, 2, 3]
    assert om._nms(np.zeros((0, 4)), np.zeros(0), np.zeros(0), 0.5).tolist() == []


# --- the tool -------------------------------------------------------------------------


def test_detect_returns_grounding_dino_shaped_boxes_best_first(om, fake):
    r = om.detect(frame(), ["cube", "box", "plug"])
    assert [d["label"] for d in r["detections"]] == ["cube", "box", "cube"]
    assert xyxy(r["detections"][0]["box"]) == [100, 300, 120, 320]
    assert r["detections"][0]["score"] == pytest.approx(0.6)
    assert set(r["detections"][0]) == {"box", "label", "score"}
    assert fake.calls[-1][1:] == (["cube", "box", "plug"], "Detect cube, box, plug.")


def test_threshold_task_and_max_detections(om, fake):
    r = om.detect(frame(), ["cube", "box", "plug"], threshold=0.1)
    assert [d["label"] for d in r["detections"]][-1] == "plug"
    r = om.detect(frame(), ["cube", "box", "plug"], threshold=0.5)
    assert len(r["detections"]) == 1
    r = om.detect(frame(), ["cube", "box", "plug"], max_detections=2)
    assert len(r["detections"]) == 2
    om.detect(frame(), ["cube"], task="Detect the grey cube on the table.")
    assert fake.calls[-1][2] == "Detect the grey cube on the table."
    with pytest.raises(ToolError, match="threshold"):
        om.detect(frame(), ["cube"], threshold=1.0)


def test_nothing_above_the_floor_is_an_empty_list(om, monkeypatch):
    monkeypatch.setattr(om, "_get_runtime", lambda: FakeRuntime(scene=[]))
    assert om.detect(frame(), ["cube"]) == {"detections": []}


def test_bad_images_and_model_errors(om, fake, monkeypatch):
    with pytest.raises(ToolError, match="RGB uint8"):
        om.detect(np.zeros((H, W), np.uint8), ["cube"])
    with pytest.raises(ToolError, match="RGB uint8"):
        om.detect(np.zeros((H, W, 3), np.float32), ["cube"])

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(om, "_get_runtime", lambda: boom)
    with pytest.raises(PerceptionFailed, match="out of memory"):
        om.detect(frame(), ["cube"])


# --- sahi as a caller -------------------------------------------------------------------


class ToolCtx:
    def __init__(self, om):
        self.om, self.calls = om, []

    def tool(self, name, **kw):
        assert name == "omdet-turbo.detect"
        self.calls.append(kw)
        return self.om.detect(**kw)


def test_sahi_runs_it_per_slice_through_its_generic_backend(om, sahi, fake):
    ctx = ToolCtx(om)
    r = sahi.detect_sliced(
        ctx, frame(), ["cube", "box"], backend="omdet-turbo.detect", slice_px=320, min_score=0.2
    )
    assert r["n_backend_calls"] == len(ctx.calls) > 1
    assert ctx.calls[0]["query"] == "cube. box"
    assert {d["label"] for d in r["detections"]} <= {"cube", "box"}
    assert any(d["label"] == "cube" for d in r["detections"])
