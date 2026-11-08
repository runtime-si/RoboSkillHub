"""sampling-labeled-points-in-mask: FakeContext tests over a rendered depth scene.

A camera 1 m over a table at z = 0 looks straight down. On the table: a 40 mm
tall "mug" (a 60 mm disc) with a 20 mm tall "handle" bar on its +x side, and a
patch with no depth. sam3 and the VLM are scripted; the geometry tool is the
hub's own.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "sampling-labeled-points-in-mask"
H, W, F = 240, 320, 300.0
K = np.array([[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]])
# Camera-to-world: at (0, 0, 1), optical axis down; camera x = world x, camera y = -world y.
POSE = {
    "position": {"x": 0.0, "y": 0.0, "z": 1.0},
    "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
}
MUG = (0.0, 0.0, 0.03, 0.04)  # centre x, y, radius, height
HANDLE = (0.03, 0.06, -0.008, 0.008, 0.02)  # x0, x1, y0, y1, height
HOLE = (-0.30, -0.20, 0.10, 0.20)


def _height(x, y):
    cx, cy, r, h = MUG
    z = np.where((x - cx) ** 2 + (y - cy) ** 2 <= r * r, h, 0.0)
    x0, x1, y0, y1, hh = HANDLE
    return np.where((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z == 0), hh, z)


def _world_xy():
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    x, y = (u - W / 2) / F, -(v - H / 2) / F
    for _ in range(3):  # each ray meets the surface at depth 1 - z
        d = 1.0 - _height(x * 1.0, y * 1.0)
        x, y = (u - W / 2) / F * d, -(v - H / 2) / F * d
    return x, y, d


def _scene():
    x, y, d = _world_xy()
    x0, x1, y0, y1 = HOLE
    depth = np.where((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1), 0.0, d)
    return {
        "name": "overhead",
        "rgb": np.full((H, W, 3), 180, dtype=np.uint8),
        "depth": depth.astype(np.float32),
        "intrinsics": K,
        "pose": POSE,
    }


def _object_mask():
    x, y, d = _world_xy()
    return _height(x, y) > 0


def _hole_mask():
    x, y, _ = _world_xy()
    x0, x1, y0, y1 = HOLE
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["sample_labeled_points"].module


@pytest.fixture(scope="module")
def obs():
    return {"cameras": [{**_scene(), "name": "left_wrist"}, _scene()], "arms": []}


def _reply(*items, text=None):
    body = text if text is not None else json.dumps(list(items))
    return {"text": body}


@pytest.fixture
def ctx_for(tool_registry):
    def make(vlm=None, sam3=None):
        tools = {
            "geometry.mask_to_world_points": lambda **kw: tool_registry.invoke(
                "geometry.mask_to_world_points", **kw
            ),
        }
        if vlm is not None:
            tools["vlm.query"] = vlm
        if sam3 is not None:
            tools["sam3.segment_text"] = sam3
        return FakeContext(tools)

    return make


def _mask_u8(m):
    return m.astype(np.uint8) * 255


# --------------------------------------------------------------------------- grid


def test_best_grid_takes_the_largest_stride_in_range(script):
    mask = np.zeros((200, 200), bool)
    mask[20:180, 30:170] = True
    pts, stride = script.best_grid(mask, (35, 45))
    assert 35 <= len(pts) <= 45
    assert mask[pts[:, 0], pts[:, 1]].all()
    coarser = script._grid_inside(mask, stride + 1, (30, 169, 20, 179))
    assert len(coarser) < 35


def test_best_grid_falls_back_to_the_nearest_count(script):
    mask = np.zeros((60, 60), bool)
    mask[20:34, 20:34] = True  # 14 px: columns 25..28 are inside the 5 px inset
    pts, stride = script.best_grid(mask, (35, 45))
    assert len(pts) == 4  # 2 x 2 is the most any stride gives
    assert stride == 3  # the coarsest stride with that count


def test_boundary_samples_follow_the_largest_outline(script):
    mask = np.zeros((100, 100), bool)
    mask[10:20, 10:20] = True  # small blob
    mask[40:90, 30:80] = True  # large blob
    pts = script.boundary_samples(mask, 12)
    assert len(pts) == 12
    assert (pts[:, 0] >= 40).all() and (pts[:, 1] >= 30).all()
    assert mask[pts[:, 0], pts[:, 1]].all()


# --------------------------------------------------------------------------- selection


def test_a_chosen_marker_comes_back_as_its_world_point(script, obs, ctx_for):
    vlm = [_reply({"index": 30, "label": "on top"})]
    ctx = ctx_for(vlm=vlm)
    out = script.run(ctx, obs, "the top of the mug", mask=_mask_u8(_object_mask()))
    assert out["route"] == "selected", out["detail"]
    mask = _object_mask()
    pix = np.array(out["marker_pixels"], dtype=int)
    assert len(pix) == out["marker_count"] > 25
    assert mask[pix[:, 1], pix[:, 0]].all()
    assert out["pixel"] == out["marker_pixels"][29]
    u, v = out["pixel"]
    d = 1.0 - out["point"]["z"]
    assert out["point"]["x"] == pytest.approx((u - W / 2) / F * d, abs=0.004)
    assert out["point"]["y"] == pytest.approx(-(v - H / 2) / F * d, abs=0.004)
    assert out["point"]["z"] in (pytest.approx(0.04, abs=0.002), pytest.approx(0.02, abs=0.002))
    assert out["points"] == [out["point"]]
    assert out["selected"][0]["index"] == 30 and out["selected"][0]["label"] == "on top"
    call = ctx.calls_to("vlm.query")[0].kwargs
    assert call["image"] is out["marked_image"]
    assert '"index"' in call["prompt"] and "output []" in call["prompt"]
    assert "Instruction: the top of the mug" in call["prompt"]
    n = out["marker_count"]
    assert f"Points: {json.dumps(list(range(1, n + 1)))}" in call["prompt"]
    assert "model" not in call
    assert out["selection_image"].shape == (H, W, 3)
    assert "30 (on top)" in out["detail"]


def test_the_marked_crop_is_upscaled_around_the_mask(script, obs, ctx_for):
    out = script.run(ctx_for(vlm=[_reply()]), obs, "x", mask=_mask_u8(_object_mask()))
    ys, xs = np.nonzero(_object_mask())
    ch, cw = ys.max() - ys.min() + 21, xs.max() - xs.min() + 21  # the box + 10 px a side
    scale = min(H // ch, W // cw)
    assert scale >= 4
    assert out["marked_image"].shape == (ch * scale, cw * scale, 3)
    assert (out["marked_image"] == (255, 0, 0)).all(-1).any()  # red discs drawn


def test_markers_on_the_outline_come_first_and_read_the_object(script, obs, ctx_for):
    """A boundary marker sits on the mask's edge; its window's near side is the object."""
    bleed = _object_mask()
    grown = bleed.copy()
    grown[1:, :] |= bleed[:-1, :]
    grown[:-1, :] |= bleed[1:, :]
    grown[:, 1:] |= bleed[:, :-1]
    grown[:, :-1] |= bleed[:, 1:]  # the mask spills 1 px onto the table
    out = script.run(
        ctx_for(vlm=[_reply({"index": 1, "label": "edge"})]), obs, "edge", mask=_mask_u8(grown)
    )
    assert out["route"] == "selected"
    u, v = out["pixel"]
    assert not script.np.isnan(u)
    on_edge = grown[int(v), int(u)] and not all(
        grown[int(v) + dv, int(u) + du] for dv in (-2, 2) for du in (-2, 2)
    )
    assert on_edge
    assert out["point"]["z"] > 0.015  # the object (20 or 40 mm), not the table


def test_outline_markers_are_numbered_first_and_win_overlaps(script, obs, ctx_for):
    mask = _object_mask()
    out = script.run(ctx_for(vlm=[_reply()]), obs, "x", mask=_mask_u8(mask))
    pix = np.array(out["marker_pixels"], dtype=int)
    outline = {(int(c), int(r)) for r, c in script.boundary_samples(mask)}
    is_outline = np.array([(u, v) in outline for u, v in pix])
    n_outline = int(is_outline.sum())
    assert 10 <= n_outline <= len(outline)
    assert is_outline[:n_outline].all()
    gaps = np.hypot(*(pix[:, None, :] - pix[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(gaps, np.inf)
    scale = script._crop_box(mask, script.CROP_MARGIN)[4]
    assert gaps.min() >= 2 * script.MIN_MARKER_RADIUS / scale - 1e-9  # no disc covers another


def test_several_choices_keep_the_vlm_s_order_without_duplicates(script, obs, ctx_for):
    vlm = [_reply({"index": 5}, {"index": 2}, {"index": 5}, {"index": 9})]
    out = script.run(ctx_for(vlm=vlm), obs, "corners", mask=_mask_u8(_object_mask()))
    assert [s["index"] for s in out["selected"]] == [5, 2, 9]
    assert len(out["points"]) == 3 and out["point"] == out["points"][0]
    out = script.run(ctx_for(vlm=vlm), obs, "corners", mask=_mask_u8(_object_mask()), max_points=1)
    assert [s["index"] for s in out["selected"]] == [5]


def test_fences_comments_and_bare_numbers_are_read(script):
    text = 'Sure:\n```json\n[{"index": 3, "label": "a"}, // the handle\n 7]\n```'
    assert script.parse_selection(text, 10) == [(3, "a"), (7, "")]


# --------------------------------------------------------------------------- exits


def test_an_empty_list_is_none_suitable(script, obs, ctx_for):
    out = script.run(ctx_for(vlm=[_reply()]), obs, "a handle", mask=_mask_u8(_object_mask()))
    assert out["route"] == "none_suitable"
    assert "chose none" in out["detail"]
    assert out["points"] == [] and out["point"] == {}
    assert out["marker_count"] > 0 and out["marked_image"].size > 3


def test_a_bad_reply_is_asked_again(script, obs, ctx_for):
    vlm = [_reply(text="I think the handle."), _reply({"index": 4})]
    ctx = ctx_for(vlm=vlm)
    out = script.run(ctx, obs, "the handle", mask=_mask_u8(_object_mask()))
    assert out["route"] == "selected"
    assert len(ctx.calls_to("vlm.query")) == 2


def test_unoffered_numbers_twice_are_none_suitable(script, obs, ctx_for):
    vlm = [_reply({"index": 999}), _reply({"index": 0})]
    ctx = ctx_for(vlm=vlm)
    out = script.run(ctx, obs, "the handle", mask=_mask_u8(_object_mask()))
    assert out["route"] == "none_suitable"
    assert "was not offered" in out["detail"]
    assert len(ctx.calls_to("vlm.query")) == 2


def test_markers_without_depth_are_not_offered(script, obs, ctx_for):
    half = _hole_mask()
    x, y, _ = _world_xy()
    half |= (x >= -0.20) & (x <= -0.10) & (y >= 0.10) & (y <= 0.20)  # half in the hole
    full = script.best_grid(half, (35, 45))[0]
    out = script.run(ctx_for(vlm=[_reply({"index": 1})]), obs, "x", mask=_mask_u8(half))
    assert out["route"] == "selected"
    assert out["marker_count"] < len(full) + script.BOUNDARY_SAMPLES
    assert out["point"]["z"] == pytest.approx(0.0, abs=0.003)
    assert out["point"]["x"] > -0.205  # on the half with depth

    ctx = ctx_for(vlm=[_reply({"index": 1})])
    out = script.run(ctx, obs, "x", mask=_mask_u8(_hole_mask()))
    assert out["route"] == "none_suitable" and "has depth" in out["detail"]
    assert not ctx.calls_to("vlm.query")


def test_sam3_supplies_the_mask_from_the_object_name(script, obs, ctx_for):
    sam = {"masks": [_mask_u8(_object_mask())], "scores": [0.8], "boxes": [{}]}
    ctx = ctx_for(vlm=[_reply({"index": 2})], sam3=sam)
    out = script.run(ctx, obs, "the handle", object_name="mug")
    assert out["route"] == "selected"
    call = ctx.calls_to("sam3.segment_text")[0].kwargs
    assert call["query"] == "mug" and call["image"].shape == (H, W, 3)
    assert "sam3 0.80" in out["detail"]


def test_not_seen_when_sam3_finds_nothing_or_scores_low(script, obs, ctx_for):
    ctx = ctx_for(sam3={"masks": [], "scores": [], "boxes": []}, vlm=[])
    out = script.run(ctx, obs, "the handle", object_name="mug")
    assert out["route"] == "not_seen" and "found nothing" in out["detail"]
    low = {"masks": [_mask_u8(_object_mask())], "scores": [0.1], "boxes": [{}]}
    out = script.run(ctx_for(sam3=low, vlm=[]), obs, "the handle", object_name="mug")
    assert out["route"] == "not_seen" and "0.10" in out["detail"]


def test_not_seen_for_an_empty_or_tiny_mask(script, obs, ctx_for):
    out = script.run(ctx_for(vlm=[]), obs, "x", mask=np.zeros((H, W), np.uint8))
    assert out["route"] == "not_seen" and "empty" in out["detail"]
    tiny = np.zeros((H, W), np.uint8)
    tiny[100:103, 100:103] = 255
    out = script.run(ctx_for(vlm=[]), obs, "x", mask=tiny)
    assert out["route"] == "not_seen" and "9 pixels" in out["detail"]


def test_needs_a_mask_or_a_name(script, obs, ctx_for):
    with pytest.raises(ValueError, match="mask or an object_name"):
        script.run(ctx_for(), obs, "x")
    with pytest.raises(ValueError, match="differ in size"):
        script.run(ctx_for(), obs, "x", mask=np.ones((10, 10), np.uint8))


def test_model_and_camera_are_passed_through(script, obs, ctx_for):
    ctx = ctx_for(vlm=[_reply({"index": 1})])
    out = script.run(
        ctx,
        obs,
        "x",
        mask=_mask_u8(_object_mask()),
        camera_name="left_wrist",
        vlm_model="gemini-3-flash",
    )
    assert out["route"] == "selected"
    assert ctx.calls_to("vlm.query")[0].kwargs["model"] == "gemini-3-flash"
    with pytest.raises(ValueError, match="no camera named"):
        script.run(ctx_for(), obs, "x", mask=_mask_u8(_object_mask()), camera_name="nope")


def test_the_example_declares_every_route(script):
    example = Path(script.__file__).parents[1] / "examples" / "canonical_subgraph.json"
    graph = json.loads(example.read_text())
    declared = set(graph["exit"]["success_values"]) | {graph["on_error"]}
    assert declared == {"selected", "none_suitable", "not_seen"}
