"""owlv2 bundle: box decoding, text and exemplar modes, the exemplar string and sahi as a caller,
with OWLv2 faked."""

from __future__ import annotations

import numpy as np
import pytest
from gap.skills import load_skills
from gap.testing import FakeContext
from gap_core.errors import PerceptionFailed, ToolError

from .conftest import SKILLS_ROOT

H, W = 480, 640
SIDE = 640.0
D = 8


@pytest.fixture(scope="module")
def owl(skills_registry):
    return skills_registry.get("owlv2").tools_module


@pytest.fixture(scope="module")
def sahi():
    return load_skills(SKILLS_ROOT, only=["sahi"]).get("sahi").tools_module


def unit(i: int) -> np.ndarray:
    v = np.zeros(D)
    v[i] = 1.0
    return v


#: What the fake model sees, in frame pixels of a W x H frame: label, box, text logit, embedding.
CUBE_A = ("cube", (100, 300, 120, 320), 1.0, unit(0) + 0.1 * unit(5))
CUBE_B = ("cube", (300, 350, 322, 372), 0.0, unit(0) + 0.2 * unit(6))
PLUG = ("plug", (400, 200, 416, 210), -0.5, unit(1))
OPEN_BOX = ("box", (200, 250, 260, 300), -1.0, unit(0) + 0.9 * unit(2))
SCENE = [CUBE_A, CUBE_B, PLUG, OPEN_BOX]


class FakeRuntime:
    """One box per scene object, a weaker duplicate of the first, a background box and one in the padding.

    Boxes are relative to the padded square (side = the longer image side) as OWLv2's are; ``text``
    gives a box its object's logit for prompts naming its label (a template around it is fine) and
    -6 otherwise.
    """

    def __init__(self, scene=SCENE, side=None):
        self.scene, self.side = list(scene), side
        self.texts: list[list[str]] = []
        self.embedded: list[tuple[int, int]] = []

    def _rows(self, image):
        side = self.side or float(max(image.shape[:2]))
        rows = [(lab, box, logit, emb, 0.0) for lab, box, logit, emb in self.scene]
        if self.scene:
            lab, (x1, y1, x2, y2), logit, emb = self.scene[0]
            rows.append(
                (lab, (x1 + 2, y1 + 1, x2 + 2, y2 + 1), logit - 1.0, emb * 0.99 + 0.01, -2.0)
            )
        rows.append(("", (10, 10, 60, 40), -6.0, unit(7), -4.0))
        rows.append(("cube", (300, side - 20, 330, side - 5), 3.0, unit(0), 0.0))  # in the padding
        boxes = []
        for _, (x1, y1, x2, y2), *_ in rows:
            boxes.append(
                [(x1 + x2) / 2 / side, (y1 + y2) / 2 / side, (x2 - x1) / side, (y2 - y1) / side]
            )
        return rows, np.array(boxes)

    def text(self, image, texts):
        self.texts.append(list(texts))
        rows, boxes = self._rows(image)
        logits = np.full((len(rows), len(texts)), -6.0)
        for i, (lab, _, logit, _, _) in enumerate(rows):
            for j, t in enumerate(texts):
                if lab and lab in t:
                    logits[i, j] = logit
        return logits, boxes

    def embed(self, image):
        self.embedded.append(image.shape[:2])
        rows, boxes = self._rows(image)
        embeds = np.array([r[3] for r in rows], dtype=np.float32)
        objectness = np.array([r[4] if r[0] else -4.0 for r in rows])
        return embeds, boxes, objectness


@pytest.fixture
def fake(owl, monkeypatch):
    rt = FakeRuntime()
    monkeypatch.setattr(owl, "_get_runtime", lambda: rt)
    return rt


def frame(h=H, w=W):
    return np.zeros((h, w, 3), dtype=np.uint8)


def xyxy(box):
    return [round(box[k], 3) for k in ("x1", "y1", "x2", "y2")]


# --- registration, pins, imports ----------------------------------------------------


def test_tools_registered_with_their_schemas(tool_registry):
    det = tool_registry.get("owlv2.detect").schema
    assert set(det.inputs) == {
        "image",
        "prompts",
        "threshold",
        "query",
        "template",
        "nms_iou",
        "max_detections",
    }
    assert det.inputs["image"].required and not det.inputs["prompts"].required
    assert det.inputs["threshold"].default == pytest.approx(0.2)
    assert det.inputs["template"].default == "a photo of a {}"
    assert set(det.outputs) == {"detections"}
    like = tool_registry.get("owlv2.detect_like").schema
    assert {"image", "query_image", "query_box", "threshold", "query", "exclude_query"} <= set(
        like.inputs
    )
    assert like.inputs["threshold"].default == pytest.approx(0.87)
    assert {"detections", "boxes", "scores", "found", "count", "exemplar"} == set(like.outputs)
    emb = tool_registry.get("owlv2.embed_exemplar").schema
    assert set(emb.inputs) == {"query_image", "query_box"}
    assert set(emb.outputs) == {"query", "exemplar", "dim"}
    for name in ("owlv2.detect", "owlv2.detect_like", "owlv2.embed_exemplar"):
        assert tool_registry.get(name).tags == ("perception",)


def test_import_does_not_pull_torch_and_models_are_pinned(owl):
    assert "torch" not in vars(owl) and "transformers" not in vars(owl)
    assert owl.DEFAULT_MODEL in owl.MODELS
    for rev in owl.MODELS.values():
        assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def test_model_choice_is_checked(owl, monkeypatch, tmp_path):
    monkeypatch.setenv("GAP_OWLV2_MODEL", "google/owlv2-large-patch14-ensemble")
    assert owl._model_source() == (
        "google/owlv2-large-patch14-ensemble",
        owl.MODELS["google/owlv2-large-patch14-ensemble"],
    )
    monkeypatch.setenv("GAP_OWLV2_MODEL", str(tmp_path))
    assert owl._model_source() == (str(tmp_path), None)
    assert owl.weights_cached() is False  # no config.json there
    monkeypatch.setenv("GAP_OWLV2_MODEL", "someone/owlv2-finetune")
    with pytest.raises(ValueError, match="must be one of"):
        owl._model_source()
    assert owl.weights_cached() is False


# --- geometry ---------------------------------------------------------------------


def test_boxes_scale_by_the_padded_square_and_padding_boxes_are_dropped(owl):
    # 640 x 480: the processor pads the bottom 160 rows, so y scales by 640 like x
    xyxy_, inside = owl._decode_boxes(np.array([[0.5, 0.5, 0.1, 0.05], [0.5, 0.9, 0.1, 0.1]]), W, H)
    assert xyxy_[0] == pytest.approx([288, 304, 352, 336])
    assert inside.tolist() == [True, False]
    # a box hanging over the frame edge is clipped
    xyxy_, inside = owl._decode_boxes(np.array([[0.01, 0.5, 0.1, 0.1]]), W, H)
    assert xyxy_[0][0] == 0.0 and inside[0]
    # a portrait frame pads on the right
    _, inside = owl._decode_boxes(np.array([[0.9, 0.5, 0.05, 0.05]]), 480, 640)
    assert not inside[0]


def test_nms_suppresses_per_label_and_both_paths_agree(owl):
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([0, 0, 1, 0])
    assert owl._nms(boxes, scores, labels, 0.3).tolist() == [0, 2, 3]
    assert owl._nms(boxes, scores, labels, 1.0).tolist() == [0, 1, 2, 3]
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 600, (2100, 2))
    many = np.hstack([xy, xy + rng.uniform(5, 40, (2100, 2))])
    s = rng.uniform(0, 1, 2100)
    lab = rng.integers(0, 2, 2100)

    def greedy(n):
        kept = []
        for i in np.argsort(-s[:n], kind="stable"):
            if all(lab[j] != lab[i] or owl._iou(many[j], many[i : i + 1])[0] <= 0.3 for j in kept):
                kept.append(i)
        return kept

    assert owl._nms(many[:1500], s[:1500], lab[:1500], 0.3).tolist() == greedy(
        1500
    )  # pairwise matrix
    assert owl._nms(many, s, lab, 0.3).tolist() == greedy(2100)  # over 2000 boxes: the loop


def test_pick_exemplar_takes_the_least_typical_box_among_the_best_overlaps(owl):
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 11], [0, 0, 30, 30], [100, 100, 110, 110]], float)
    embeds = np.array([unit(1), unit(0), unit(1), unit(1)])  # unit(0) is the atypical one
    i, iou = owl._pick_exemplar(boxes, embeds, np.array([0, 0, 10, 10.5]), np.ones(4, bool))
    assert i == 1 and iou > 0.9
    assert (
        owl._pick_exemplar(boxes, embeds, np.array([500, 500, 510, 510.0]), np.ones(4, bool))[0]
        == -1
    )


# --- text mode --------------------------------------------------------------------


def test_detect_labels_scores_and_pixels(owl, fake):
    out = owl.detect(frame(), ["cube", "plug", "box"], threshold=0.1)
    assert set(out) == {"detections"}
    dets = out["detections"]
    assert [d["label"] for d in dets] == ["cube", "cube", "plug", "box"]
    assert [round(d["score"], 3) for d in dets] == [0.731, 0.5, 0.378, 0.269]
    assert xyxy(dets[0]["box"]) == pytest.approx([100, 300, 120, 320])
    for d in dets:
        assert set(d) == {"box", "label", "score"}
    # the duplicate of cube A (IoU ~0.7) is suppressed, the padding box (logit 3) is dropped
    assert all(d["box"]["y2"] <= H for d in dets)
    assert fake.texts[-1] == ["a photo of a cube", "a photo of a plug", "a photo of a box"]


def test_detect_threshold_nms_and_max_detections(owl, fake):
    assert [
        d["label"] for d in owl.detect(frame(), ["cube", "plug"], threshold=0.45)["detections"]
    ] == [
        "cube",
        "cube",
    ]
    kept_all = owl.detect(frame(), ["cube"], threshold=0.1, nms_iou=1.0)["detections"]
    assert len(kept_all) == 3  # the duplicate is back
    assert len(owl.detect(frame(), ["cube"], threshold=0.1, max_detections=1)["detections"]) == 1
    assert owl.detect(frame(), ["screwdriver"], threshold=0.1) == {"detections": []}


def test_detect_takes_grounding_dino_and_sahi_strings(owl, fake):
    a = owl.detect(frame(), "cube. plug.", threshold=0.1)
    b = owl.detect(frame(), query="cube. plug", threshold=0.1)
    c = owl.detect(frame(), ["cube", "plug", "cube"], threshold=0.1)
    assert a == b == c
    assert fake.texts[-1] == ["a photo of a cube", "a photo of a plug"]
    owl.detect(frame(), ["cube"], template="{}")
    assert fake.texts[-1] == ["cube"]
    owl.detect(frame(), ["cube"], template="")
    assert fake.texts[-1] == ["cube"]


def test_detect_refuses_bad_input_and_wraps_model_errors(owl, fake, monkeypatch):
    with pytest.raises(ToolError, match="RGB uint8"):
        owl.detect(np.zeros((H, W), np.uint8), ["cube"])
    with pytest.raises(ToolError, match="RGB uint8"):
        owl.detect(np.zeros((H, W, 3), np.float32), ["cube"])
    with pytest.raises(ToolError, match="at least one object"):
        owl.detect(frame(), " . ")

    class Broken:
        def text(self, *a):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(owl, "_get_runtime", lambda: Broken())
    with pytest.raises(PerceptionFailed, match="out of memory"):
        owl.detect(frame(), ["cube"])


# --- exemplar mode ----------------------------------------------------------------


def test_detect_like_in_the_same_frame_finds_the_look_alikes(owl, fake):
    out = owl.detect_like(frame(), query_box=[99, 299, 121, 321], threshold=0.9)
    # cube A itself (cos 1) and cube B (cos ~0.98); the open box (cos ~0.74) and the plug stay out
    assert [xyxy(b) for b in out["boxes"]] == [[100, 300, 120, 320], [300, 350, 322, 372]]
    assert out["found"] and out["count"] == 2
    assert out["scores"][0] == pytest.approx(1.0) and 0.95 < out["scores"][1] < 1.0
    assert out["exemplar"]["iou"] > 0.8 and xyxy(out["exemplar"]["box"]) == [100, 300, 120, 320]
    assert all(d["label"] == "exemplar" for d in out["detections"])
    assert len(fake.embedded) == 1  # one embedding pass for query and search
    others = owl.detect_like(
        frame(), query_box=[99, 299, 121, 321], threshold=0.9, exclude_query=True
    )
    assert [xyxy(b) for b in others["boxes"]] == [[300, 350, 322, 372]]
    loose = owl.detect_like(frame(), query_box=[99, 299, 121, 321], threshold=0.7, label="brick")
    assert {d["label"] for d in loose["detections"]} == {"brick"} and loose["count"] == 3


def test_exemplar_string_carries_the_query_to_other_frames(owl, fake):
    e = owl.embed_exemplar(frame(), {"x1": 99, "y1": 299, "x2": 121, "y2": 321})
    assert e["query"].startswith("owlv2-exemplar:8:") and e["dim"] == D
    assert "." not in e["query"]  # sahi strips trailing periods and splits on none
    vec = owl._decode_query(e["query"], "t")
    assert vec == pytest.approx((unit(0) + 0.1 * unit(5)), abs=1e-3)
    by_string = owl.detect_like(frame(), query=e["query"], threshold=0.9)
    by_box = owl.detect_like(frame(), frame(), [99, 299, 121, 321], threshold=0.9)
    assert by_string["boxes"] == by_box["boxes"]
    assert by_string["exemplar"] is None and by_box["exemplar"] == e["exemplar"]
    assert by_string["scores"] == pytest.approx(by_box["scores"], abs=1e-3)


def test_a_crop_without_a_box_uses_the_most_object_like_box(owl, fake):
    out = owl.detect_like(frame(), query_image=frame(), threshold=0.9)
    # objectness ties at 0 between the scene objects: the first, cube A
    assert out["exemplar"]["iou"] == 1.0 and out["count"] == 2


def test_detect_like_refusals(owl, fake):
    with pytest.raises(ToolError, match="give the exemplar"):
        owl.detect_like(frame())
    with pytest.raises(ToolError, match="empty"):
        owl.detect_like(frame(), query_box=[10, 10, 5, 20])
    with pytest.raises(PerceptionFailed, match="no OWLv2 box overlaps"):
        owl.detect_like(frame(), query_box=[600, 5, 630, 8])
    with pytest.raises(ToolError, match="embed_exemplar"):
        owl.detect_like(frame(), query="cube")
    bad_dim = owl._encode_query(np.ones(4, np.float32))
    with pytest.raises(ToolError, match="4 dimensions"):
        owl.detect_like(frame(), query=bad_dim)
    with pytest.raises(ToolError, match="not a valid"):
        owl.detect_like(frame(), query="owlv2-exemplar:8:@@@")
    with pytest.raises(ToolError, match="says 9"):
        owl.detect_like(
            frame(), query=owl._encode_query(np.ones(8, np.float32)).replace(":8:", ":9:")
        )


def test_nothing_alike_is_an_empty_result(owl, monkeypatch):
    rt = FakeRuntime(scene=[PLUG])
    monkeypatch.setattr(owl, "_get_runtime", lambda: rt)
    e = owl._encode_query(unit(3))
    out = owl.detect_like(frame(), query=e)
    assert out == {
        "detections": [],
        "boxes": [],
        "scores": [],
        "found": False,
        "count": 0,
        "exemplar": None,
    }


# --- as a sahi backend --------------------------------------------------------------


def test_sahi_runs_text_and_exemplar_modes_through_the_generic_backend(owl, sahi, fake):
    ctx = FakeContext(
        {
            "owlv2.detect": lambda **kw: owl.detect(**kw, threshold=0.3),
            "owlv2.detect_like": lambda **kw: owl.detect_like(**kw, threshold=0.9),
        }
    )
    r = sahi.detect_sliced(ctx, frame(), ["cube", "plug"], backend="owlv2.detect", slice_px=0)
    assert r["labels"] == ["cube", "cube", "plug"]
    assert ctx.calls[-1].kwargs["query"] == "cube. plug"
    token = owl.embed_exemplar(frame(), [99, 299, 121, 321])["query"]
    r = sahi.detect_sliced(ctx, frame(), [token], backend="owlv2.detect_like", slice_px=0)
    assert [xyxy(b) for b in r["boxes"]] == [[100, 300, 120, 320], [300, 350, 322, 372]]
    assert ctx.calls[-1].kwargs["query"] == token
    assert set(r["labels"]) == {token}
