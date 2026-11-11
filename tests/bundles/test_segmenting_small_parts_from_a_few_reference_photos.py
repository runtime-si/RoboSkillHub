"""segmenting-small-parts-from-a-few-reference-photos and tools/dinov2: region cutting and pooling,
the tool with DINOv2 faked by mean patch colours, the classify rule, and the skill end to end on
synthetic frames of coloured parts with FakeContext."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

SKILL = "segmenting-small-parts-from-a-few-reference-photos"


@pytest.fixture(scope="module")
def dinov2(skills_registry):
    return skills_registry.get("dinov2").tools_module


@pytest.fixture(scope="module")
def label(skills_registry):
    bundle = skills_registry.get(SKILL)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["label"].module


class ColourRuntime:
    """DINOv2 faked: a patch's token is its mean RGB (plus a constant), the CLS the crop's mean."""

    patch = 14
    name = "fake-colour"

    def __init__(self):
        self.calls = []

    def encode(self, crops, size):
        import cv2

        self.calls.append((len(crops), size))
        n = size // 14
        cls, pat = [], []
        for c in crops:
            g = cv2.resize(c, (size, size)).astype(np.float32) / 255.0 - 0.45
            p = g.reshape(n, 14, n, 14, 3).mean(axis=(1, 3)).reshape(n * n, 3)
            pat.append(np.concatenate([p, np.full((n * n, 1), 0.05, np.float32)], axis=1))
            cls.append(np.concatenate([p.mean(axis=0), [0.05]]).astype(np.float32))
        return np.stack(cls), np.stack(pat)


@pytest.fixture
def runtime(dinov2, monkeypatch):
    rt = ColourRuntime()
    monkeypatch.setattr(dinov2, "_runtime", rt)
    return rt


COLOURS = {"red_brick": (200, 30, 30), "blue_brick": (30, 40, 210), "domino": (235, 230, 215)}


def _scene(parts, h=240, w=320, seed=0):
    """Grey table with rectangles of the given colours; returns the image and one mask per part."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 90, np.uint8) + rng.integers(0, 6, (h, w, 3), dtype=np.uint8)
    masks = []
    for name, (x, y, pw, ph) in parts:
        m = np.zeros((h, w), bool)
        m[y : y + ph, x : x + pw] = True
        img[m] = COLOURS[name]
        masks.append(m)
    return img, masks


def _layout(names, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i, n in enumerate(names):
        out.append((n, (20 + 70 * (i % 4) + int(rng.integers(0, 10)), 30 + 90 * (i // 4), 18, 12)))
    return out


# -- tools/dinov2 ---------------------------------------------------------------------------


def test_cut_regions_squares_and_clips(dinov2):
    img = np.zeros((100, 120, 3), np.uint8)
    m = np.zeros((100, 120), bool)
    m[0:10, 110:120] = True  # in the corner: the square crop is shifted inside the frame
    crops, cmasks, boxes = dinov2.cut_regions(img, [m], margin=0.15, use_mask=True)
    x1, y1, x2, y2 = boxes[0]
    assert x2 - x1 == y2 - y1 == dinov2.MIN_CROP and x2 == 120 and y1 == 0
    assert cmasks[0].sum() == 100 and crops[0].shape == (24, 24, 3)
    assert tuple(crops[0][~cmasks[0]][0]) == dinov2.FILL


def test_pool_uses_only_patches_inside_the_mask(dinov2):
    grid = 4
    pat = np.zeros((1, 16, 2), np.float32)
    pat[0, :, 0] = 1.0
    pat[0, 5, :] = [0.0, 1.0]  # the one patch inside the mask
    cm = np.zeros((56, 56), bool)
    cm[14:28, 14:28] = True
    emb, counts = dinov2.pool(np.array([[1.0, 0.0]], np.float32), pat, [cm], grid)
    assert counts == [1]
    assert np.allclose(emb[0], np.array([0, 1, 1, 0]) / np.sqrt(2), atol=1e-6)


def test_embed_regions_rejects_bad_input(dinov2, runtime):
    with pytest.raises(ToolError):
        dinov2.embed_regions(np.zeros((10, 10), np.uint8), boxes=[[0, 0, 5, 5]])
    with pytest.raises(ToolError):
        dinov2.embed_regions(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ToolError):
        dinov2.embed_regions(np.zeros((10, 10, 3), np.uint8), masks=[np.zeros((5, 5))])


def test_embed_regions_masks_and_boxes(dinov2, runtime):
    img, masks = _scene(_layout(["red_brick", "blue_brick", "red_brick"], 0))
    r = dinov2.embed_regions(img, masks=masks)
    e = r["embeddings"]
    assert e.shape == (3, 8) and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-5)
    assert e[0] @ e[2] > e[0] @ e[1]
    assert runtime.calls == [(3, 224)] and min(r["pooled_patches"]) >= 1
    rb = dinov2.embed_regions(img, boxes=[{"x1": 20, "y1": 30, "x2": 38, "y2": 42}])
    assert rb["embeddings"].shape == (1, 8) and rb["model"] == "fake-colour"


# -- the classify rule --------------------------------------------------------------------


def test_classify_top_k_and_gates(label):
    bank = np.eye(3, dtype=np.float32)[[0, 0, 1]]
    labs = ["a", "a", "b"]
    q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.7, 0.71, 0]], np.float32)
    out = label.classify(q, bank, labs, min_score=0.45, min_margin=0.0)
    assert [o["label"] for o in out] == ["a", "b", None, "b"]
    assert out[2]["best_class"] in ("a", "b") and out[2]["score"] == 0.0
    tight = label.classify(q, bank, labs, min_score=0.45, min_margin=0.05)
    assert tight[3]["label"] is None and abs(tight[3]["margin"] - 0.01) < 1e-3


# -- the skill ------------------------------------------------------------------------------


def _refs(seed=10):
    img, masks = _scene(_layout(["red_brick", "blue_brick", "domino"], seed), seed=seed)
    return [{"label": n, "mask": m, "image": img}
            for n, m in zip(["red_brick", "blue_brick", "domino"], masks, strict=True)]  # fmt: skip


def _ctx(dinov2, **tools):
    return FakeContext({"dinov2.embed_regions": lambda **kw: dinov2.embed_regions(**kw), **tools})


def test_skill_labels_the_frame(label, dinov2, runtime):
    names = ["blue_brick", "domino", "red_brick", "red_brick", "domino", "blue_brick"]
    img, masks = _scene(_layout(names, 3), seed=3)
    ctx = _ctx(dinov2)
    out = label.run(ctx, references=_refs(), masks=masks, frame={"rgb": img})
    assert out["route"] == "instances_found"
    assert [i["label"] for i in out["instances"]] == names
    assert out["counts"] == {"blue_brick": 2, "domino": 2, "red_brick": 2}
    assert out["bank"]["embeddings"].shape == (3, 8) and out["classes"] == sorted(COLOURS)
    assert out["instances"][0]["box"] == [int(masks[0].nonzero()[1].min()),
                                          int(masks[0].nonzero()[0].min()),
                                          int(masks[0].nonzero()[1].max()) + 1,
                                          int(masks[0].nonzero()[0].max()) + 1]  # fmt: skip
    # the bank is reusable without the references
    again = label.run(ctx, bank=out["bank"], masks=masks, frame={"rgb": img})
    assert [i["label"] for i in again["instances"]] == names


def test_skill_from_observation_and_sam3_prompt(label, dinov2, runtime):
    img, masks = _scene(_layout(["domino", "red_brick"], 4), seed=4)
    ctx = _ctx(
        dinov2,
        **{
            "robot.get_observation": lambda **kw: {"cameras": [{"name": "top", "rgb": img}]},
            "sam3.segment_text": lambda **kw: {"detections": [{"mask": m} for m in masks]},
        },
    )
    out = label.run(ctx, references=_refs(), prompt="small part", camera="top")
    assert out["route"] == "instances_found"
    assert [i["label"] for i in out["instances"]] == ["domino", "red_brick"]


def test_skill_exits(label, dinov2, runtime):
    img, masks = _scene(_layout(["red_brick"], 5), seed=5)
    ctx = _ctx(dinov2)
    assert label.run(ctx, references=[], masks=masks, frame={"rgb": img})["route"] == "bank_empty"
    assert label.run(ctx, references=[{"label": "x"}], masks=masks,
                     frame={"rgb": img})["route"] == "bank_empty"  # fmt: skip
    tiny = np.zeros(img.shape[:2], bool)
    tiny[0:3, 0:3] = True
    out = label.run(ctx, references=_refs(), masks=[tiny], frame={"rgb": img})
    assert out["route"] == "none_found" and out["n_proposals"] == 1
    out = label.run(ctx, references=_refs(), masks=masks, frame={"rgb": img}, min_score=1.01)
    assert out["route"] == "none_found" and out["instances"] == []
    out = label.run(ctx, references=_refs(), masks=masks, frame={"rgb": img}, min_score=1.01,
                    keep_unknown=True)  # fmt: skip
    assert out["instances"][0]["label"] is None and out["instances"][0]["best_class"] == "red_brick"
