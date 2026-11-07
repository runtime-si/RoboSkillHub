"""selecting-free-placement-spots: FakeContext tests over a rendered depth scene.

A camera 1 m over a table at z = 0 looks straight down. On the table: a 40 mm
box, a 16 mm board, an 11 deg ramp and a patch with no depth; past y = -0.25
the table ends and the depth lands on a floor a metre lower. The model tools
are scripted; the geometry tools are the hub's own.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "selecting-free-placement-spots"
H, W, F = 240, 320, 300.0
K = np.array([[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]])
# Camera-to-world: at (0, 0, 1), optical axis down; camera x = world x, camera y = -world y.
POSE = {
    "position": {"x": 0.0, "y": 0.0, "z": 1.0},
    "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
}
BOX = (0.15, 0.20, -0.05, 0.0, 0.04)
BOARD = (-0.25, -0.05, 0.05, 0.25, 0.016)
RAMP = (0.30, 0.45, -0.20, -0.05)
HOLE = (0.26, 0.50, 0.12, 0.30)
TABLE_END_Y = -0.25

FREE = (0.0, -0.12)
NEAR_BOX = (0.10, -0.02)
BESIDE_BOX = (0.13, -0.025)
ON_BOARD = (-0.15, 0.15)
BOARD_EDGE = (-0.05, 0.10)
TABLE_EDGE = (0.0, -0.23)
ON_RAMP = (0.375, -0.125)
IN_HOLE = (0.375, 0.15)


def _height(x, y):
    z = np.zeros_like(x)
    for x0, x1, y0, y1, h in (BOX, BOARD):
        z = np.where((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1), h, z)
    x0, x1, y0, y1 = RAMP
    return np.where((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1), 0.2 * (x - x0), z)


def _scene(seed=0):
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    x, y = (u - W / 2) / F, -(v - H / 2) / F
    for _ in range(3):  # each ray meets the surface at depth 1 - z
        d = 1.0 - _height(x * 1.0, y * 1.0)
        x, y = (u - W / 2) / F * d, -(v - H / 2) / F * d
    depth = d + np.random.default_rng(seed).normal(0.0, 0.0015, d.shape)
    depth = np.where(y < TABLE_END_Y, 2.0, depth)
    x0, x1, y0, y1 = HOLE
    depth = np.where((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1), 0.0, depth)
    rgb = np.full((H, W, 3), 180, dtype=np.uint8)
    return {
        "name": "overhead",
        "rgb": rgb,
        "depth": depth.astype(np.float32),
        "intrinsics": K,
        "pose": POSE,
    }


def _pixel(xy, z=0.0):
    d = 1.0 - z
    return {"pixel_x": W / 2 + F * xy[0] / d, "pixel_y": H / 2 - F * xy[1] / d}


def _points(*spots):
    pts = [{**_pixel(s, z), "label": "free"} for s, z in spots]
    if not pts:
        return {"pixel_x": 0.0, "pixel_y": 0.0, "found": False, "points": []}
    return {**{k: pts[0][k] for k in ("pixel_x", "pixel_y")}, "found": True, "points": pts}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["select_free_spots"].module


@pytest.fixture(scope="module")
def obs():
    return {"cameras": [_scene()], "arms": []}


@pytest.fixture
def ctx_for(tool_registry):
    def make(point=None, vlm=None):
        tools = {
            "geometry.pixel_to_world_point": lambda **kw: tool_registry.invoke(
                "geometry.pixel_to_world_point", **kw
            ),
            "geometry.mask_to_world_points": lambda **kw: tool_registry.invoke(
                "geometry.mask_to_world_points", **kw
            ),
        }
        if point is not None:
            tools["gemini-er.point"] = point
        if vlm is not None:
            tools["vlm.query"] = vlm
        return FakeContext(tools)

    return make


def test_a_free_spot_is_found_and_back_projected(script, obs, ctx_for):
    ctx = ctx_for(point=_points((FREE, 0.0)))
    out = script.run(ctx, obs, region="the table")
    assert out["route"] == "found", out["detail"]
    assert out["spot_count"] == 1
    p = out["place_position"]
    assert p["x"] == pytest.approx(FREE[0], abs=0.004)
    assert p["y"] == pytest.approx(FREE[1], abs=0.004)
    assert p["z"] == pytest.approx(0.0, abs=0.002)
    assert out["place_positions"] == [p]
    assert out["footprint_radius"] == pytest.approx(0.035)
    assert out["spots"][0]["free_radius"] == pytest.approx(0.07)
    query = ctx.calls_to("gemini-er.point")[0].kwargs
    assert "vacant area on the table" in query["query"] and "7 cm" in query["query"]
    assert query["max_points"] == 8


def test_each_failure_gets_its_reason(script, obs, ctx_for):
    spots = [(BESIDE_BOX, 0.0), (BOARD_EDGE, 0.0), (TABLE_EDGE, 0.0), (ON_RAMP, 0.0)]
    out = script.run(ctx_for(point=_points(*spots)), obs, region="the table")
    assert out["route"] == "none_free"
    reasons = [r["reason"] for r in out["rejected"]]
    assert reasons[0] == "occupied"
    assert reasons[1] in ("tilted", "edge")
    assert reasons[2] == "edge"
    assert reasons[3] == "tilted"
    assert "failed the depth check" in out["detail"]
    assert out["place_positions"] == []


def test_a_pixel_without_depth_is_rejected(script, obs, ctx_for):
    out = script.run(ctx_for(point=_points((IN_HOLE, 0.0))), obs, region="the table")
    assert out["route"] == "none_free"
    assert out["rejected"][0]["reason"] == "no_depth"


def test_spots_on_a_raised_board_sit_on_the_board(script, obs, ctx_for):
    out = script.run(ctx_for(point=_points((ON_BOARD, 0.016))), obs, region="the board")
    assert out["route"] == "found", out["rejected"]
    assert out["place_position"]["z"] == pytest.approx(0.016, abs=0.002)


def test_ranked_by_free_room_then_the_model_s_order(script, obs, ctx_for):
    out = script.run(ctx_for(point=_points((NEAR_BOX, 0.0), (FREE, 0.0))), obs)
    assert out["spot_count"] == 2
    first, second = out["spots"]
    assert first["source"] == "point 2" and second["source"] == "point 1"
    assert first["free_radius"] > second["free_radius"] >= 0.035
    assert second["free_radius"] < 0.05  # the box is 50 mm from NEAR_BOX


def test_near_duplicates_are_dropped(script, obs, ctx_for):
    near = (FREE[0] + 0.01, FREE[1])
    out = script.run(ctx_for(point=_points((FREE, 0.0), (near, 0.0))), obs)
    assert out["spot_count"] == 1
    assert [r["reason"] for r in out["rejected"]] == ["duplicate"]


def test_a_large_object_does_not_fit_beside_the_box(script, obs, ctx_for):
    out = script.run(ctx_for(point=_points((NEAR_BOX, 0.0))), obs)
    assert out["route"] == "found"
    obb = {
        "center": {"x": 0.0, "y": 0.0, "z": 0.03},
        "extent": {"x": 0.04, "y": 0.03, "z": 0.03},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    out = script.run(ctx_for(point=_points((NEAR_BOX, 0.0))), obs, object_obb=obb)
    assert out["footprint_radius"] == pytest.approx(0.05 + 0.01)
    assert out["route"] == "none_free"
    assert out["rejected"][0]["reason"] == "occupied"


def test_footprint_of_a_turned_box(script):
    q = {"w": math.cos(math.pi / 8), "x": 0.0, "y": 0.0, "z": math.sin(math.pi / 8)}
    obb = {
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "extent": {"x": 0.03, "y": 0.04, "z": 0.2},
        "orientation": q,
    }
    assert script.footprint_of(obb) == pytest.approx(0.05)
    tipped = {
        **obb,
        "orientation": {"w": math.cos(math.pi / 4), "x": math.sin(math.pi / 4), "y": 0.0, "z": 0.0},
    }
    assert script.footprint_of(tipped) == pytest.approx(math.hypot(0.03, 0.2))


def test_the_model_naming_nothing_is_none_free(script, obs, ctx_for):
    out = script.run(ctx_for(point=_points()), obs, region="the tray")
    assert out["route"] == "none_free"
    assert out["detail"] == "the model named no free spot on the tray"
    assert out["place_position"] == {"x": 0.0, "y": 0.0, "z": 0.0}


def test_grid_tiles_are_sampled_and_checked(script, obs, ctx_for):
    cols, rows = 8, 6
    free = _pixel(FREE)
    col = int(free["pixel_x"] // (W / cols))
    row_from_top = int(free["pixel_y"] // (H / rows))
    label = f"{'abcdefgh'[col]}{rows - row_from_top}"
    reply = f'```json\n{{"tiles": ["{label}", "g5", "z9"]}} // best first\n```'
    seen = {}

    def vlm(prompt, image):
        seen["prompt"], seen["image"] = prompt, image
        return {"text": reply}

    out = script.run(ctx_for(vlm=vlm), obs, region="the table", method="grid")
    assert out["route"] == "found", out["rejected"]
    spot = out["spots"][0]
    assert spot["source"] == f"tile {label}"
    assert col * W / cols <= spot["pixel_x"] <= (col + 1) * W / cols
    assert row_from_top * H / rows <= spot["pixel_y"] <= (row_from_top + 1) * H / rows
    assert out["rejected"][0]["source"] == "tile g5"  # inside the patch with no depth
    assert out["rejected"][0]["reason"] == "no_depth"
    assert out["rejected"][0]["tried"] == 9
    assert "the table" in seen["prompt"] and "bottom to top" in seen["prompt"]
    assert seen["image"].shape == (H, W, 3)
    assert not np.array_equal(seen["image"], obs["cameras"][0]["rgb"])  # the grid is drawn


def test_parse_tiles_follows_moka_s_convention(script):
    assert script.parse_tiles('{"tiles": ["a1", "h6", "c3"]}', 8, 6) == [(0, 5), (7, 0), (2, 3)]
    assert script.parse_tiles("I would pick b2 and then d4.", 8, 6) == [(1, 4), (3, 2)]
    assert script.parse_tiles('{"tiles": ["i1", "a7", "a1", "a1"]}', 8, 6) == [(0, 5)]
    assert script.parse_tiles("no", 8, 6) == []
    assert script.parse_json_string('```json\n{"a": 1} // note\n```') == {"a": 1}
    assert json.dumps(script.parse_json_string("nothing here")) == "null"


def test_tile_samples_stay_in_the_tile_centre_first(script):
    pts = script.tile_samples(320, 240, 8, 6, (2, 1))
    assert pts[0] == (100.0, 60.0)
    assert len(pts) == 9
    assert all(80 <= x <= 120 and 40 <= y <= 80 for x, y in pts)


def test_picks_the_overhead_camera_and_refuses_bad_arguments(script, obs, ctx_for):
    wrist = {**obs["cameras"][0], "name": "wrist_d405", "depth": np.zeros((H, W), np.float32)}
    two = {"cameras": [wrist, obs["cameras"][0]], "arms": []}
    out = script.run(ctx_for(point=_points((FREE, 0.0))), two)
    assert out["route"] == "found"
    with pytest.raises(ValueError, match="no camera named"):
        script.run(ctx_for(point=_points()), obs, camera_name="top")
    with pytest.raises(ValueError, match="method"):
        script.run(ctx_for(), obs, method="tiles")
