"""Tests for the sahi tool bundle -- the detectors stubbed with FakeContext, no GPU.

The fake grounding-dino sees a scene of boxes in frame pixels and reports an
object only when it is at least ``SEEN_PX`` pixels on the model's input,
which it upsamples to an 800 px short side as the real processor does: a
12 px part is 20 px there on the 480 px frame (missed) and 30 px on a 320 px
slice (found). That is the effect slicing buys.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gap.skills import load_skills
from gap.testing import FakeContext
from gap_core.errors import ToolError

ROOT = Path(__file__).resolve().parents[2]
SEEN_PX = 30.0


@pytest.fixture(scope="module")
def sahi():
    reg = load_skills(ROOT, only=["sahi"])
    return reg.get("sahi").tools_module


@pytest.fixture()
def image() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _box(x1, y1, x2, y2):
    return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}


def _xyxy(b):
    return [b["x1"], b["y1"], b["x2"], b["y2"]]


def _scene_detector(scene, *, offset_of, whole_tile_score=0.0):
    """A grounding-dino.detect stand-in over ``scene`` = [(label, [x1, y1, x2, y2], score)].

    ``offset_of`` maps a tile's shape to where it sits in the frame (the
    tests know their slice windows); boxes come back clipped to the tile, in
    tile pixels, like a detector run on a crop.
    """

    def detect(image, query, **kw):
        h, w = image.shape[:2]
        ox, oy = offset_of(image)
        up = 800.0 / min(h, w)
        dets = []
        for label, (x1, y1, x2, y2), score in scene:
            cx1, cy1 = max(x1 - ox, 0), max(y1 - oy, 0)
            cx2, cy2 = min(x2 - ox, w), min(y2 - oy, h)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            if max(cx2 - cx1, cy2 - cy1) * up < SEEN_PX or label not in query:
                continue
            dets.append({"box": _box(cx1, cy1, cx2, cy2), "label": label, "score": score})
        if whole_tile_score:
            dets.append({"box": _box(0, 0, w, h), "label": "cube", "score": whole_tile_score})
        return {"detections": dets}

    return detect


def _tiles_by_content(frame):
    """Where a tile sits in ``frame``, read from its top-left pixel (see the ``frame`` fixture)."""

    def offset_of(tile):
        r, g, b = (int(v) for v in tile[0, 0])
        return r + 256 * (b % 4), g + 256 * (b // 4)

    return offset_of


@pytest.fixture()
def frame() -> np.ndarray:
    """A frame whose pixels encode their own position: x % 256, y % 256, then the high bits."""
    ys, xs = np.mgrid[0:480, 0:640]
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[..., 0] = xs % 256
    img[..., 1] = ys % 256
    img[..., 2] = xs // 256 + 4 * (ys // 256)
    return img


def _ctx(scene, frame, **kw):
    return FakeContext(
        {"grounding-dino.detect": _scene_detector(scene, offset_of=_tiles_by_content(frame), **kw)}
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_tools_registered_with_ctx_injection(tool_registry):
    for name in ("sahi.detect_sliced", "sahi.slice_windows", "sahi.merge"):
        assert name in tool_registry
    schema = tool_registry.get("sahi.detect_sliced").schema
    assert "ctx" not in schema.inputs
    assert schema.inputs["image"].required and schema.inputs["prompts"].required
    assert schema.inputs["backend"].default == "grounding-dino"
    assert tool_registry.get("sahi.detect_sliced").tags == ("perception",)


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------


def test_slice_windows_match_sahi_get_slice_bboxes(sahi):
    # sahi.slicing.get_slice_bboxes(480, 640, 320, 320, 0.2, 0.2) at 10061afd
    expected = [
        [0, 0, 320, 320],
        [256, 0, 576, 320],
        [320, 0, 640, 320],
        [0, 160, 320, 480],
        [256, 160, 576, 480],
        [320, 160, 640, 480],
    ]
    out = sahi.slice_windows(640, 480, 320, 0.2)
    assert [_xyxy(w) for w in out["windows"]] == expected and out["count"] == 6
    assert sahi._slice_bboxes(480, 640, 320, 320, 0.2, 0.2) == expected


def test_slice_windows_cover_the_image_and_shift_into_the_roi(sahi):
    for w, h, s, o in [(640, 480, 256, 0.25), (1280, 720, 320, 0.2), (500, 300, 200, 0.0)]:
        cover = np.zeros((h, w), dtype=bool)
        for b in sahi.slice_windows(w, h, s, o)["windows"]:
            assert b["x2"] - b["x1"] == min(s, w) and b["y2"] - b["y1"] == min(s, h)
            cover[int(b["y1"]) : int(b["y2"]), int(b["x1"]) : int(b["x2"])] = True
        assert cover.all()
    roi = sahi.slice_windows(640, 480, 200, 0.2, roi=_box(100, 250, 500, 470))["windows"]
    assert min(b["x1"] for b in roi) == 100 and max(b["x2"] for b in roi) == 500
    assert min(b["y1"] for b in roi) == 250 and max(b["y2"] for b in roi) == 470


def test_slice_px_zero_or_larger_than_the_image_is_one_window(sahi):
    assert sahi.slice_windows(640, 480, 0)["count"] == 1
    assert _xyxy(sahi.slice_windows(300, 200, 320)["windows"][0]) == [0, 0, 300, 200]


@pytest.mark.parametrize("kw", [{"overlap": 1.0}, {"overlap": -0.1}, {"slice_px": 16}])
def test_bad_slicing_arguments_raise(sahi, kw):
    with pytest.raises(ToolError):
        sahi.slice_windows(640, 480, **{"slice_px": 320, **kw})


# ---------------------------------------------------------------------------
# detect_sliced with a stubbed grounding-dino
# ---------------------------------------------------------------------------


SMALL = [("cube", [100, 100, 112, 112], 0.6), ("cube", [500, 400, 512, 411], 0.5)]


def test_small_parts_are_found_on_slices_and_missed_on_the_whole_frame(sahi, frame):
    whole = sahi.detect_sliced(_ctx(SMALL, frame), frame, ["cube"], slice_px=0)
    assert whole["count"] == 0 and whole["n_backend_calls"] == 1 and whole["n_slices"] == 0

    ctx = _ctx(SMALL, frame)
    out = sahi.detect_sliced(ctx, frame, ["cube"])
    assert out["n_slices"] == 6 and out["n_backend_calls"] == 7  # 6 slices + the whole frame
    assert len(ctx.calls) == 7
    assert sorted(_xyxy(b) for b in out["boxes"]) == [[100, 100, 112, 112], [500, 400, 512, 411]]
    assert out["labels"] == ["cube", "cube"] and out["scores"] == [0.6, 0.5]
    assert out["found"] and out["masks"] is None
    assert all(d["sources"] == ["slice"] for d in out["detections"])


def test_query_joins_prompts_with_periods_and_passes_thresholds_only_when_set(sahi, image):
    ctx = FakeContext({"grounding-dino.detect": {"detections": []}})
    sahi.detect_sliced(ctx, image, ["red brick", "domino."], slice_px=0)
    assert ctx.calls[0].kwargs["query"] == "red brick. domino."
    assert "box_threshold" not in ctx.calls[0].kwargs
    sahi.detect_sliced(ctx, image, "cube", slice_px=0, box_threshold=0.35, text_threshold=0.3)
    assert (
        ctx.calls[1].kwargs["box_threshold"] == 0.35
        and ctx.calls[1].kwargs["text_threshold"] == 0.3
    )


def test_a_part_cut_by_a_slice_edge_merges_into_one_box(sahi, frame):
    # x 250..270 straddles the first slice's right edge (320) only partly, and
    # the second slice (256..576) cuts it at 256: three slices see pieces.
    scene = [("cube", [250, 100, 270, 116], 0.7)]
    out = sahi.detect_sliced(_ctx(scene, frame), frame, ["cube"])
    assert out["count"] == 1
    assert _xyxy(out["boxes"][0]) == [250, 100, 270, 116]
    assert out["detections"][0]["n_merged"] >= 2 and out["raw_slice"] >= 2


def test_whole_slice_boxes_are_dropped_by_max_box_fraction(sahi, frame):
    out = sahi.detect_sliced(_ctx(SMALL, frame, whole_tile_score=0.9), frame, ["cube"])
    assert out["count"] == 2
    loose = sahi.detect_sliced(
        _ctx(SMALL, frame, whole_tile_score=0.9), frame, ["cube"], max_box_fraction=1.0
    )
    assert any(d["score"] == 0.9 for d in loose["detections"])


def test_parts_larger_than_a_slice_come_from_the_whole_frame_pass(sahi, frame):
    scene = [("box", [150, 60, 500, 300], 0.8)]
    out = sahi.detect_sliced(_ctx(scene, frame), frame, ["box"])
    big = [d for d in out["detections"] if d["sources"] == ["full"] or "full" in d["sources"]]
    assert big and _xyxy(big[0]["box"]) == [150, 60, 500, 300]
    no_full = sahi.detect_sliced(_ctx(scene, frame), frame, ["box"], full_frame=False)
    assert no_full["n_backend_calls"] == 6
    assert all("full" not in d["sources"] for d in no_full["detections"])


def test_score_and_size_filters(sahi, frame):
    scene = [*SMALL, ("cube", [300, 200, 340, 240], 0.25)]
    assert sahi.detect_sliced(_ctx(scene, frame), frame, ["cube"], min_score=0.55)["count"] == 1
    small_only = sahi.detect_sliced(_ctx(scene, frame), frame, ["cube"], max_box_px=20)
    assert small_only["count"] == 2 and all(
        d["box"]["x2"] - d["box"]["x1"] <= 20 for d in small_only["detections"]
    )
    assert sahi.detect_sliced(_ctx(scene, frame), frame, ["cube"], max_detections=1)["count"] == 1


def test_roi_restricts_the_search_and_reports_frame_pixels(sahi, frame):
    ctx = _ctx(SMALL, frame)
    out = sahi.detect_sliced(ctx, frame, ["cube"], roi=_box(400, 300, 640, 480), slice_px=160)
    assert [_xyxy(b) for b in out["boxes"]] == [[500, 400, 512, 411]]
    assert all(w["x1"] >= 400 and w["y1"] >= 300 for w in out["slice_windows"])
    assert all(c.kwargs["image"].shape[0] <= 180 for c in ctx.calls)
    with pytest.raises(ToolError):
        sahi.detect_sliced(ctx, frame, ["cube"], roi=_box(10, 10, 14, 300))


def test_labels_snap_to_prompts_and_merge_per_label(sahi, image):
    dets = {
        "detections": [
            {"box": _box(10, 10, 30, 30), "label": "lego brick", "score": 0.6},
            {"box": _box(12, 12, 30, 30), "label": "red lego brick", "score": 0.5},
            {"box": _box(11, 11, 30, 30), "label": "domino", "score": 0.4},
        ]
    }
    ctx = FakeContext({"grounding-dino.detect": dets})
    out = sahi.detect_sliced(ctx, image, ["lego brick", "domino"], slice_px=0)
    assert sorted(out["labels"]) == ["domino", "lego brick"]
    agn = sahi.detect_sliced(ctx, image, ["lego brick", "domino"], slice_px=0, class_agnostic=True)
    assert (
        agn["count"] == 1
        and agn["labels"] == ["lego brick"]
        and agn["detections"][0]["n_merged"] == 3
    )


# ---------------------------------------------------------------------------
# Other backends
# ---------------------------------------------------------------------------


def test_sam3_backend_runs_per_prompt_and_unions_masks_across_slices(sahi, frame):
    offset_of = _tiles_by_content(frame)
    obj = [250, 100, 270, 116]

    def segment_text(image, query, max_results):
        assert max_results == 0
        h, w = image.shape[:2]
        ox, oy = offset_of(image)
        x1, y1 = max(obj[0] - ox, 0), max(obj[1] - oy, 0)
        x2, y2 = min(obj[2] - ox, w), min(obj[3] - oy, h)
        if query != "cube" or x2 <= x1 or y2 <= y1:
            return {"masks": [], "scores": [], "boxes": []}
        m = np.zeros((h, w), dtype=np.uint8)
        m[y1:y2, x1:x2] = 255
        return {"masks": [m], "scores": [0.8], "boxes": [_box(x1, y1, x2, y2)]}

    ctx = FakeContext({"sam3.segment_text": segment_text})
    out = sahi.detect_sliced(ctx, frame, ["cube", "domino"], backend="sam3", return_masks=True)
    assert out["n_backend_calls"] == 14  # 7 passes x 2 prompts
    assert out["count"] == 1 and _xyxy(out["boxes"][0]) == obj
    mask = out["masks"][0]
    assert mask.shape == (480, 640) and mask.dtype == np.uint8
    ys, xs = np.nonzero(mask)
    assert [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] == obj


def test_sam3_low_scores_are_floored_before_the_merge(sahi, image):
    # SAM3 returns everything down to score 0: a faint box around the part
    # would swallow it in the IOS union merge.
    junk = {
        "masks": [],
        "scores": [0.8, 0.1],
        "boxes": [_box(100, 100, 115, 114), _box(60, 60, 200, 180)],
    }
    ctx = FakeContext({"sam3.segment_text": junk})
    out = sahi.detect_sliced(ctx, image, ["cube"], backend="sam3", slice_px=0)
    assert [_xyxy(b) for b in out["boxes"]] == [[100, 100, 115, 114]]
    swallowed = sahi.detect_sliced(ctx, image, ["cube"], backend="sam3", slice_px=0, min_score=0.0)
    assert [_xyxy(b) for b in swallowed["boxes"]] == [[60, 60, 200, 180]]


def test_locate_anything_and_named_tool_backends(sahi, image):
    hit = {"detections": [{"box": _box(5, 5, 15, 15), "label": "cube", "score": 1.0}]}
    ctx = FakeContext({"locate-anything.detect": hit, "gemini-er.detect": hit})
    out = sahi.detect_sliced(ctx, image, ["cube", "box"], backend="locate-anything", slice_px=0)
    assert ctx.calls[0].kwargs["categories"] == ["cube", "box"] and out["count"] == 1
    sahi.detect_sliced(ctx, image, ["cube", "box"], backend="gemini-er.detect", slice_px=0)
    assert ctx.calls[1].kwargs["query"] == "cube. box"
    assert out["masks"] is None


def test_input_errors(sahi, image):
    ctx = FakeContext({"grounding-dino.detect": {"detections": []}})
    with pytest.raises(ToolError):
        sahi.detect_sliced(ctx, image, [])
    with pytest.raises(ToolError):
        sahi.detect_sliced(ctx, image.astype(np.float32), ["cube"])
    with pytest.raises(ToolError):
        sahi.detect_sliced(ctx, image, ["cube"], backend="yolo")
    with pytest.raises(ToolError):
        sahi.detect_sliced(ctx, image, ["cube"], postprocess="WBF")
    with pytest.raises(ToolError):
        sahi.detect_sliced(ctx, image, ["cube"], match_metric="GIOU")
    assert ctx.calls == []


def test_backend_errors_propagate(sahi, image):
    def boom(**kw):
        raise RuntimeError("GPU out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        sahi.detect_sliced(FakeContext({"grounding-dino.detect": boom}), image, ["cube"])


# ---------------------------------------------------------------------------
# merge (SAHI postprocess semantics)
# ---------------------------------------------------------------------------


CHAIN = [
    {"box": _box(0, 0, 10, 10), "label": "a", "score": 0.9},
    {"box": _box(5, 0, 15, 10), "label": "a", "score": 0.8},
    {"box": _box(10, 0, 20, 10), "label": "a", "score": 0.7},
]


def test_merge_greedy_nmm_unions_direct_matches_only(sahi):
    out = sahi.merge(CHAIN, match_metric="IOU", match_threshold=0.3)
    assert out["count"] == 2
    assert _xyxy(out["detections"][0]["box"]) == [0, 0, 15, 10]
    assert out["detections"][0]["score"] == 0.9 and out["detections"][0]["n_merged"] == 2


def test_merge_nmm_is_transitive_and_nms_keeps_the_best(sahi):
    nmm = sahi.merge(CHAIN, postprocess="NMM", match_metric="IOU", match_threshold=0.3)
    # B joins A's group and claims C; as in SAHI's _apply_merge, C is then checked
    # against the grown A+B box (IOU 0.25 < 0.3), not merged, and dropped with the group.
    assert nmm["count"] == 1 and _xyxy(nmm["detections"][0]["box"]) == [0, 0, 15, 10]
    loose = sahi.merge(CHAIN, postprocess="NMM", match_metric="IOU", match_threshold=0.25)
    assert loose["count"] == 1 and _xyxy(loose["detections"][0]["box"]) == [0, 0, 20, 10]
    nms = sahi.merge(CHAIN, postprocess="NMS", match_metric="IOU", match_threshold=0.3)
    assert [_xyxy(d["box"]) for d in nms["detections"]] == [[0, 0, 10, 10], [10, 0, 20, 10]]


def test_merge_ios_joins_a_clipped_piece_to_its_whole(sahi):
    dets = [
        {"box": _box(100, 100, 120, 116), "label": "cube", "score": 0.5},
        {"box": _box(100, 100, 104, 116), "label": "cube", "score": 0.6},
    ]
    assert sahi.merge(dets, match_metric="IOU")["count"] == 2
    out = sahi.merge(dets)
    assert out["count"] == 1 and _xyxy(out["detections"][0]["box"]) == [100, 100, 120, 116]
    assert out["detections"][0]["score"] == 0.6


def test_merge_keeps_labels_apart_unless_class_agnostic(sahi):
    dets = [{**CHAIN[0], "label": "a"}, {**CHAIN[0], "label": "b", "score": 0.95}]
    assert sahi.merge(dets)["count"] == 2
    agn = sahi.merge(dets, class_agnostic=True)
    assert agn["count"] == 1 and agn["detections"][0]["label"] == "b"
    assert sahi.merge([])["count"] == 0
    with pytest.raises(ToolError):
        sahi.merge(dets, postprocess="soft-nms")
