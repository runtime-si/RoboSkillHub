"""flagging-odd-one-out-parts-by-mutual-scoring and tools/musc: the mutual-scoring maths on synthetic
tokens, the tool with DINOv2 faked by raw pixel patches, and the skill on rendered domino scenes
(tests/bundles/_odd_one_out_scenes.py) with musc.score faked or run on the fake runtime."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

from . import _odd_one_out_scenes as SC

SKILL = "flagging-odd-one-out-parts-by-mutual-scoring"


@pytest.fixture(scope="module")
def musc(skills_registry):
    return skills_registry.get("musc").tools_module


@pytest.fixture(scope="module")
def flag(skills_registry):
    bundle = skills_registry.get(SKILL)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["flag"].module


class PixelRuntime:
    """DINOv2 faked: a patch's token is its 14 x 14 RGB pixels."""

    patch = 14
    name = "fake-pixels"

    def __init__(self):
        self.calls = []

    def tokens(self, images, size, layers):
        import cv2

        self.calls.append((len(images), size, tuple(layers)))
        out = []
        for im in images:
            g = cv2.resize(im, (size, size)).astype(np.float32) / 255.0
            n = size // 14
            out.append(g.reshape(n, 14, n, 14, 3).transpose(0, 2, 1, 3, 4).reshape(n * n, -1))
        t = np.stack(out)
        return [t + 0.001 * k for k in range(len(layers))]


@pytest.fixture
def pixel_runtime(musc, monkeypatch):
    rt = PixelRuntime()
    monkeypatch.setattr(musc, "_runtime", rt)
    return rt


def _tokens(n=8, grid=6, c=16, odd=None, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(grid * grid, c))
    t = np.stack([base + 0.05 * rng.normal(size=base.shape) for _ in range(n)])
    if odd is not None:
        t[odd, 7] += 3.0 * rng.normal(size=c)  # one patch of one image is unlike any other
    return t.astype(np.float32)


# -- tools/musc -----------------------------------------------------------------------------


def test_mutual_scores_put_the_odd_patch_and_image_first(musc):
    t = _tokens(odd=3)
    s, rs, pm = musc.score_tokens([t, t], grid=6)
    assert int(np.argmax(s)) == 3 and pm.shape == (8, 6, 6)
    assert int(np.argmax(pm[3])) == 7
    assert s[3] > 2.0 * np.median(s)
    assert rs.shape == (8,) and 0.0 <= rs.min() and rs.max() <= 1.0


def test_identical_images_score_alike(musc):
    s, _, _ = musc.score_tokens([_tokens()], grid=6)
    assert s.max() < 1.5 * np.median(s)


def test_aggregation_keeps_shapes_and_is_identity_at_r1(musc):
    t = _tokens(n=3)
    assert musc._aggregate(t, 6, (1,)).shape == t.shape
    np.testing.assert_allclose(musc._aggregate(t, 6, (1,)), t)
    agg = musc._aggregate(t, 6, (1, 3, 5))
    assert agg.shape == t.shape and not np.allclose(agg, t)


def test_masked_patches_neither_score_nor_serve(musc):
    t = _tokens(odd=2)
    masks = [np.ones((6, 6)) for _ in range(8)]
    masks[2] = np.ones((6, 6))
    masks[2][1, 1] = 0  # patch 7 = row 1, col 1: the odd one, now outside the part
    s, _, pm = musc.score_tokens([t], grid=6, masks=masks)
    assert pm[2, 1, 1] == 0.0
    assert s[2] < 1.5 * np.median(s)
    with pytest.raises(ValueError, match="no patch"):
        musc.score_tokens([t], grid=6, masks=[np.zeros((6, 6))] * 8)


def test_three_images_use_one_reference(musc):
    s, _, _ = musc.score_tokens([_tokens(n=3, odd=0)], grid=6)
    assert int(np.argmax(s)) == 0


def test_rescore_averages_over_similar_images(musc):
    feats = np.eye(4)
    feats[1] = feats[0]
    r = musc._rescore(np.array([1.0, 0.0, 0.5, 0.5]), feats, (1, 7))
    assert r[0] < 1.0 and r[1] > 0.0


def test_the_tool_scores_crops_through_the_runtime(musc, pixel_runtime):
    rng = np.random.default_rng(0)
    base = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    imgs = [base.copy() for _ in range(6)]
    imgs[4][20:40, 20:40] = 255 - imgs[4][20:40, 20:40]
    out = musc.score(images=imgs, image_size=112)
    assert int(np.argmax(out["scores"])) == 4
    assert out["grid"] == 8 and len(out["patch_maps"]) == 6 and out["patch_maps"][0].shape == (8, 8)
    assert pixel_runtime.calls == [(6, 112, (4, 8, 12))] and out["model"] == "fake-pixels"


def test_the_tool_checks_its_arguments(musc, pixel_runtime):
    img = np.zeros((20, 20, 3), np.uint8)
    with pytest.raises(ToolError, match="at least 3"):
        musc.score(images=[img, img])
    with pytest.raises(ToolError, match="RGB uint8"):
        musc.score(images=[img, img, img.astype(np.float32)])
    with pytest.raises(ToolError, match="masks"):
        musc.score(images=[img] * 3, masks=[img])
    with pytest.raises(ToolError, match="odd"):
        musc.score(images=[img] * 3, r_list=[2])
    with pytest.raises(ToolError, match="topmin"):
        musc.score(images=[img] * 3, topmin=0.0)


def test_weights_probe_never_downloads(musc, monkeypatch):
    monkeypatch.setenv("GAP_MUSC_MODEL", "no/such-model")
    assert musc.weights_cached() is False


# -- the skill ------------------------------------------------------------------------------


def _scene(layout, odd, seed):
    frame, masks, odd_idx, meta = SC.scene(layout, odd, seed)
    return frame, masks, odd_idx


def _score_stub(odd_idx, n_calls):
    """musc.score faked: every crop scores 0.3 +- a little, the odd ones 0.8."""

    def score(images, masks=None, **_):
        n_calls.append(len(images))
        rng = np.random.default_rng(len(images))
        s = 0.3 + 0.005 * rng.standard_normal(len(images))
        for i in odd_idx:
            s[i] = 0.8
        return {"scores": s.tolist(), "rescored": s.tolist(), "patch_maps": [], "grid": 32}

    return score


def test_a_face_up_tile_among_face_down_ones_is_flagged_by_appearance(flag):
    frame, masks, odd = _scene("face_down_grid", "face_up", 7)
    calls = []
    out = flag.run(FakeContext({"musc.score": _score_stub(odd, calls)}), masks=masks, frame=frame)
    assert out["route"] == "outliers_found" and out["outliers"] == odd
    assert out["parts"][odd[0]]["why"] == ["appearance"]
    assert calls == [10] and out["appearance_scored"] and out["n_parts"] == 10
    assert len(out["outlier_positions"]) == 1 and len(out["outlier_positions"][0]) == 3


def test_identical_tiles_are_all_consistent(flag):
    frame, masks, _ = _scene("face_down_grid", None, 3)
    out = flag.run(FakeContext({"musc.score": _score_stub([], [])}), masks=masks, frame=frame)
    assert out["route"] == "all_consistent" and out["outliers"] == []
    assert not out["aligned_layout"] or all(p["turn_deg"] < 20 for p in out["parts"])


def test_a_standing_tile_among_lying_ones_is_flagged_by_height_and_footprint(flag):
    frame, masks, odd = _scene("face_down_grid", "standing", 5)
    out = flag.run(FakeContext({"musc.score": _score_stub([], [])}), masks=masks, frame=frame)
    assert out["route"] == "outliers_found" and out["outliers"] == odd
    why = out["parts"][odd[0]]["why"]
    assert "height" in why and "footprint" in why
    assert out["depth_scores"][odd[0]] > 0.02  # 48 mm against 8 mm tops


def test_a_turned_domino_in_a_line_is_flagged_by_orientation_without_appearance(flag):
    frame, masks, odd = _scene("upright_line", "turned", 2)
    calls = []
    out = flag.run(FakeContext({"musc.score": _score_stub([], calls)}), masks=masks, frame=frame)
    assert out["route"] == "outliers_found" and out["outliers"] == odd
    assert out["parts"][odd[0]]["why"] == ["orientation"]
    assert calls == [] and not out["appearance_scored"] and out["aligned_layout"]
    assert "appearance not scored" in out["detail"]


def test_a_fallen_domino_in_a_line_is_flagged(flag):
    frame, masks, odd = _scene("upright_line", "fallen", 1)
    out = flag.run(FakeContext({"musc.score": _score_stub([], [])}), masks=masks, frame=frame)
    assert odd[0] in out["outliers"]
    assert {"height", "footprint"} <= set(out["parts"][odd[0]]["why"])


def test_scattered_parts_skip_the_orientation_test(flag):
    frame, masks, _ = _scene("flat_same", None, 2)
    rng = np.random.default_rng(0)
    rotated = []
    for m in masks:
        import cv2

        ys, xs = np.nonzero(m)
        c = (float(xs.mean()), float(ys.mean()))
        r = cv2.getRotationMatrix2D(c, float(rng.uniform(0, 180)), 1.0)
        rotated.append(cv2.warpAffine(m.astype(np.uint8), r, m.shape[::-1]) > 0)
    out = flag.run(
        FakeContext({"musc.score": _score_stub([], [])}),
        masks=rotated,
        frame=frame,
        use_depth=False,
    )
    assert not out["aligned_layout"] and out["route"] == "all_consistent"
    out = flag.run(
        FakeContext({"musc.score": _score_stub([], [])}),
        masks=rotated,
        frame=frame,
        use_depth=False,
        check_orientation=True,
    )
    assert out["route"] == "outliers_found" and all(
        p["why"] == ["orientation"] for p in out["parts"] if p["odd"]
    )


def test_end_for_end_tiles_are_flipped_to_match(flag):
    frame, masks, _ = _scene("flat_same", None, 0)
    out = flag.run(FakeContext({"musc.score": _score_stub([], [])}), masks=masks, frame=frame)
    flips = [p["flipped"] for p in out["parts"]]
    assert any(flips) and not all(flips)


def test_the_whole_chain_with_the_pixel_runtime_finds_a_face_up_tile(flag, musc, pixel_runtime):
    frame, masks, odd = _scene("face_down_grid", "face_up", 9)
    out = flag.run(
        FakeContext({"musc.score": musc.score}), masks=masks, frame=frame, use_depth=False
    )
    assert int(np.argmax(out["scores"])) == odd[0]
    assert pixel_runtime.calls and pixel_runtime.calls[0][:2] == (10, 448)


def test_a_prompt_segments_with_sam3_and_boxes_are_accepted(flag):
    frame, masks, odd = _scene("face_down_grid", "face_up", 7)
    seen = {}

    def segment(image, query, max_results):
        seen.update(query=query, max_results=max_results, shape=image.shape)
        return {"masks": [m.astype(np.uint8) * 255 for m in masks], "scores": [], "boxes": []}

    ctx = FakeContext(
        {
            "robot.get_observation": {"cameras": [frame], "arms": []},
            "sam3.segment_text": segment,
            "musc.score": _score_stub(odd, []),
        }
    )
    out = flag.run(ctx, prompt="domino")
    assert seen == {"query": "domino", "max_results": 40, "shape": (480, 640, 3)}
    assert out["outliers"] == odd and out["camera_name"] == "left_wrist"
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    out = flag.run(FakeContext({"musc.score": _score_stub(odd, [])}), boxes=boxes, frame=frame)
    assert out["n_parts"] == 10 and odd[0] in out["outliers"]


def test_too_few_parts(flag):
    frame, masks, _ = _scene("face_down_grid", None, 1)
    out = flag.run(FakeContext({}), masks=masks[:3], frame=frame)
    assert out["route"] == "too_few_parts" and "3 part(s)" in out["detail"]
    out = flag.run(
        FakeContext({"sam3.segment_text": {"masks": [], "scores": [], "boxes": []}}),
        prompt="domino",
        frame=frame,
    )
    assert out["route"] == "too_few_parts"
    out = flag.run(
        FakeContext({"robot.get_observation": {"cameras": [], "arms": []}}), prompt="domino"
    )
    assert out["route"] == "too_few_parts" and "no camera" in out["detail"]


def test_few_parts_are_scored_with_a_warning(flag):
    frame, masks, odd = _scene("face_down_grid", "face_up", 7)
    keep = [i for i in range(10) if i != odd[0]][:4] + odd
    out = flag.run(
        FakeContext({"musc.score": _score_stub([4], [])}),
        masks=[masks[i] for i in keep],
        frame=frame,
    )
    assert out["outliers"] == [4] and "under 8" in out["detail"]


def test_bad_arguments_raise(flag):
    with pytest.raises(ValueError, match="masks, boxes, or a prompt"):
        flag.run(FakeContext({}))
    with pytest.raises(ValueError, match="min_parts"):
        flag.run(FakeContext({}), prompt="x", min_parts=2)
