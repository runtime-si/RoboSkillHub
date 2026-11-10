"""training-a-part-detector-from-auto-labels + tools/rfdetr: labelling, label checks, the COCO dataset,
the training call and the detector's decode, with the teacher and RF-DETR faked."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError
from PIL import Image

BUNDLE = "training-a-part-detector-from-auto-labels"


@pytest.fixture(scope="module")
def sk(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["train_from_auto_labels"].module


@pytest.fixture(scope="module")
def rf(skills_registry):
    return skills_registry.get("rfdetr").tools_module


def frames(tmp_path, n):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(n):
        Image.fromarray(np.full((48, 64, 3), i, np.uint8)).save(d / f"f{i:03d}.png")
    (d / "notes.txt").write_text("not a frame")
    return str(d)


def det(label, x, score=0.5, w=10):
    return {"box": {"x1": x, "y1": 5.0, "x2": x + w, "y2": 15.0}, "label": label, "score": score}


def teacher(per_frame):
    """sahi.detect_sliced answers, one list of detections per frame in order."""
    return [{"detections": d} for d in per_frame]


def trained(out):
    return lambda **kw: {
        "model_path": kw["out_dir"],
        "classes": ["brick", "domino"],
        "n_train": 1,
        "n_valid": 1,
        "epochs": kw["epochs"],
        "seconds": 1.0,
    }


ONT = "lego brick=brick; domino=domino"


def test_parse_ontology(sk):
    assert sk.parse_ontology(ONT) == {"lego brick": "brick", "domino": "domino"}
    assert sk.parse_ontology("cube") == {"cube": "cube"}
    assert sk.parse_ontology({"red brick": "brick", "blue brick": "brick"}) == {
        "red brick": "brick",
        "blue brick": "brick",
    }
    with pytest.raises(ValueError, match="names no prompt"):
        sk.parse_ontology(" ; ")


def test_labels_become_a_coco_dataset_and_the_student_is_trained(sk, tmp_path):
    fd = frames(tmp_path, 10)
    per = [[det("lego brick", 2), det("domino", 30), det("cube", 40)] for _ in range(10)]
    ctx = FakeContext({"sahi.detect_sliced": teacher(per), "rfdetr.train": trained(None)})
    out = str(tmp_path / "run")
    r = sk.run(ctx, fd, ONT, out, min_labels=5, epochs=3)
    assert r["route"] == "detector_trained"
    assert r["counts"] == {"brick": 10, "domino": 10} and r["n_images"] == 10 and r["n_empty"] == 0
    assert r["model_path"] == os.path.join(out, "model")
    calls = [c.kwargs for c in ctx.calls_to("sahi.detect_sliced")]
    assert len(calls) == 10 and calls[0]["prompts"] == ["lego brick", "domino"]
    assert calls[0]["backend"] == "mm-grounding-dino.detect" and calls[0]["slice_px"] == 320
    train = json.load(open(os.path.join(r["dataset_dir"], "train", "_annotations.coco.json")))
    valid = json.load(open(os.path.join(r["dataset_dir"], "valid", "_annotations.coco.json")))
    assert [c["name"] for c in train["categories"]] == ["brick", "domino"]
    assert len(train["images"]) + len(valid["images"]) == 10 and len(valid["images"]) == 2
    a = train["annotations"][0]
    assert a["bbox"] == [2.0, 5.0, 10.0, 10.0] and a["category_id"] in (0, 1)
    assert all(
        os.path.isfile(os.path.join(r["dataset_dir"], "train", im["file_name"]))
        for im in train["images"]
    )
    tc = ctx.calls_to("rfdetr.train")[0].kwargs
    assert tc["epochs"] == 3 and tc["size"] == "nano" and tc["resolution"] == 640


def test_too_few_labels_of_a_class(sk, tmp_path):
    fd = frames(tmp_path, 6)
    per = [[det("lego brick", 2)] + ([det("domino", 30)] if i == 0 else []) for i in range(6)]
    ctx = FakeContext({"sahi.detect_sliced": teacher(per), "rfdetr.train": trained(None)})
    r = sk.run(ctx, fd, ONT, str(tmp_path / "run"), min_labels=3)
    assert r["route"] == "too_few_labels" and "domino" in r["reason"]
    assert r["counts"] == {"brick": 6, "domino": 1} and ctx.call_count("rfdetr.train") == 0


def test_mostly_empty_frames_mean_low_label_quality(sk, tmp_path):
    fd = frames(tmp_path, 4)
    per = [[det("lego brick", 2)], [], [], []]
    ctx = FakeContext({"sahi.detect_sliced": teacher(per), "rfdetr.train": trained(None)})
    r = sk.run(ctx, fd, ONT, str(tmp_path / "run"), min_labels=1)
    assert (
        r["route"] == "label_quality_low"
        and r["n_empty"] == 3
        and ctx.call_count("rfdetr.train") == 0
    )


def test_tiny_boxes_and_unknown_labels_are_dropped(sk, tmp_path):
    fd = frames(tmp_path, 2)
    per = [[det("lego brick", 2, w=1), det("plug", 10), det("domino", 30)]] * 2
    ctx = FakeContext({"sahi.detect_sliced": teacher(per), "rfdetr.train": trained(None)})
    r = sk.run(ctx, fd, ONT, str(tmp_path / "run"), min_labels=0)
    assert r["counts"] == {"brick": 0, "domino": 2}
    assert r["median_score"] == pytest.approx(0.5)


def test_bad_inputs(sk, tmp_path):
    with pytest.raises(ValueError, match="not a folder"):
        sk.run(FakeContext({}), str(tmp_path / "nope"), ONT, str(tmp_path))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no frames"):
        sk.run(FakeContext({}), str(tmp_path / "empty"), ONT, str(tmp_path))


def test_example_declares_every_route(sk):
    root = os.path.dirname(os.path.dirname(os.path.abspath(sk.__file__)))
    ex = json.load(open(os.path.join(root, "examples", "canonical_subgraph.json")))
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert declared == {
        "detector_trained",
        "too_few_labels",
        "label_quality_low",
        "training_failed",
    }


# --- tools/rfdetr --------------------------------------------------------------------------


def test_rfdetr_tools_registered(tool_registry):
    t = tool_registry.get("rfdetr.train").schema
    assert {"dataset_dir", "out_dir", "size", "epochs", "resolution"} <= set(t.inputs)
    assert t.inputs["size"].default == "nano" and t.inputs["resolution"].default == 640
    d = tool_registry.get("rfdetr.detect").schema
    assert {"image", "model_path", "threshold", "query"} == set(d.inputs)
    assert set(d.outputs) == {"detections"}


def test_rfdetr_import_is_light_and_sizes_are_apache_only(rf):
    assert "torch" not in vars(rf) and "rfdetr" not in vars(rf)
    assert set(rf.SIZES) == {"nano", "small", "medium", "large"}
    with pytest.raises(ToolError, match="Apache"):
        rf._model_class("xlarge")


def coco(path, cats, anns=1):
    path.mkdir(parents=True)
    json.dump(
        {
            "images": [{"id": 0, "file_name": "a.png", "width": 4, "height": 4}],
            "annotations": [
                {"id": i, "image_id": 0, "category_id": 0, "bbox": [0, 0, 1, 1]}
                for i in range(anns)
            ],
            "categories": [{"id": i, "name": c} for i, c in enumerate(cats)],
        },
        open(path / "_annotations.coco.json", "w"),
    )


def test_rfdetr_train_validates_before_loading(rf, tmp_path):
    with pytest.raises(ToolError, match="multiple of 32"):
        rf.train(str(tmp_path), str(tmp_path / "m"), resolution=600)
    with pytest.raises(ToolError, match="no COCO annotations"):
        rf.train(str(tmp_path), str(tmp_path / "m"))
    coco(tmp_path / "train", ["brick", "domino"])
    coco(tmp_path / "valid", ["domino", "brick"])
    with pytest.raises(ToolError, match="different categories"):
        rf.train(str(tmp_path), str(tmp_path / "m"))
    ds = tmp_path / "ds"
    coco(ds / "train", ["brick"], anns=0)
    coco(ds / "valid", ["brick"])
    with pytest.raises(ToolError, match="no annotations"):
        rf.train(str(ds), str(tmp_path / "m"))


def test_rfdetr_decode_is_best_first_with_class_names(rf):
    out = rf._decode(
        [[0, 0, 2, 2], [1, 1, 3, 3], [5, 5, 6, 6]], [0.4, 0.9, 0.8], [0, 1, 7], ["brick", "domino"]
    )
    assert [d["label"] for d in out] == ["domino", "brick"]
    assert out[0]["box"] == {"x1": 1.0, "y1": 1.0, "x2": 3.0, "y2": 3.0} and out[0][
        "score"
    ] == pytest.approx(0.9)


def test_rfdetr_detect_query_filter_and_errors(rf, monkeypatch):
    class Det:
        xyxy = np.array([[0, 0, 2, 2], [1, 1, 3, 3]])
        confidence = np.array([0.6, 0.7])
        class_id = np.array([0, 1])

    class Model:
        def predict(self, image, threshold):
            return Det()

    monkeypatch.setattr(rf, "_load", lambda path: (Model(), ["brick", "domino"]))
    img = np.zeros((8, 8, 3), np.uint8)
    assert [d["label"] for d in rf.detect(img, "m")["detections"]] == ["domino", "brick"]
    assert [d["label"] for d in rf.detect(img, "m", query="brick.")["detections"]] == ["brick"]
    with pytest.raises(ToolError, match="RGB uint8"):
        rf.detect(np.zeros((8, 8), np.uint8), "m")
    monkeypatch.undo()
    with pytest.raises(ToolError, match="no trained model"):
        rf.detect(img, "/nonexistent/model")
