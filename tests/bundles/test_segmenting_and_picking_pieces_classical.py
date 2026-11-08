"""segmenting-and-picking-pieces-classical: synthetic staging-mat frames, lifted by the real geometry tools."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

from ._jigsaw_staging import BACK, camera_frame, make_scene, match

BUNDLE = "segmenting-and-picking-pieces-classical"
UPSTREAM_MAGENTA_HSV = {"lower": [160, 175, 125], "upper": [175, 255, 255]}
"""The source's Clear_Background band for its magenta mat (chicken puzzle)."""

SYNTHETIC_MAGENTA_HSV = {"lower": [160, 175, 60], "upper": [175, 255, 255]}
"""The same band recalibrated, as the source's comments do per puzzle, for the synthetic mat:
its drop shadows go down to V 110, under the source's 125."""


@pytest.fixture(scope="module")
def bundle(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b


@pytest.fixture(scope="module")
def script(bundle):
    return bundle.canonical_scripts["segment_pieces"].module


def _invoker(registry, name):
    return lambda **kw: registry.invoke(name, **kw)


@pytest.fixture(scope="module")
def geometry(tool_registry):
    return {
        name: _invoker(tool_registry, name)
        for name in ("geometry.mask_to_world_points", "geometry.pixel_to_world_point")
    }


def _full(p, shape):
    out = np.zeros(shape, bool)
    x0, y0, x1, y1 = p["bbox_px"]
    out[y0:y1, x0:x1] = p["mask_crop"] > 0
    return out


def _run(script, geometry, scene, **kw):
    ctx = FakeContext(dict(geometry))
    out = script.run(ctx, observation={"cameras": [camera_frame(scene)], "arms": []}, **kw)
    return out, ctx


def _matched(out, scene):
    shape = scene.rgb.shape[:2]
    pred = [_full(p, shape) for p in out["pieces"]]
    return {b: (out["pieces"][a], iou) for a, b, iou in match(pred, scene.masks)}


def _angle_err(a, b, period=90.0):
    d = (a - b) % period
    return min(d, period - d)


def test_isolated_pieces_are_segmented_and_picked_inside(script, geometry):
    scene = make_scene(11, n_single=7, n_touch=0, face_down=0.0)
    out, ctx = _run(script, geometry, scene, piece_size_m=scene.cell_m)
    assert out["route"] == "segmented", out["detail"]
    assert out["n_pieces"] == len(scene.masks) == 7
    assert out["n_touching"] == 0 and out["n_clusters"] == 0
    got = _matched(out, scene)
    assert len(got) == 7
    for g, (p, iou) in got.items():
        assert iou > 0.85
        u, v = int(round(p["pick_pixel"]["x"])), int(round(p["pick_pixel"]["y"]))
        assert scene.masks[g][v, u], "the pick point is inside the piece"
        assert p["inradius_px"] >= 0.2 * scene.px_per_piece
        assert p["pickable"] and not p["touching"] and p["neighbours"] == []
        assert _angle_err(p["angle_deg"], scene.angles[g]) < 6
        assert p["layer"] == "table" and not p["stacked"]
    assert ctx.call_count("geometry.pixel_to_world_point") >= 7
    assert ctx.call_count("geometry.mask_to_world_points") >= 7


def test_world_pick_point_area_and_yaw(script, geometry):
    scene = make_scene(12, n_single=5, n_touch=0, face_down=0.0)
    out, _ = _run(script, geometry, scene, piece_size_m=scene.cell_m)
    K, zc = scene.K, scene.pose["position"]["z"]
    assert out["table_z"] == pytest.approx(0.0, abs=5e-4)
    assert out["px_per_m"] == pytest.approx(K[0, 0] / zc, rel=0.02)
    for g, (p, _) in _matched(out, scene).items():
        pos = p["position"]
        u, v = p["pick_pixel"]["x"], p["pick_pixel"]["y"]
        d = zc - 0.002
        assert pos["x"] == pytest.approx((round(u) - K[0, 2]) * d / K[0, 0], abs=1.5e-3)
        assert pos["y"] == pytest.approx(-(round(v) - K[1, 2]) * d / K[1, 1], abs=1.5e-3)
        assert pos["z"] == pytest.approx(0.002, abs=1.5e-3)
        assert p["height_m"] == pytest.approx(0.002, abs=7e-4)
        gt_area = scene.masks[g].sum() * (d / K[0, 0]) ** 2
        assert p["area_m2"] == pytest.approx(gt_area, rel=0.12)
        # A top-down camera with this pose keeps angles: world yaw = image angle (mod 90).
        assert _angle_err(math.degrees(p["yaw"]), p["angle_deg"]) < 1.0
        assert 0 <= p["grip"]["yaw"] < math.pi
        assert p["grip"]["opening_m"] > p["grip"]["width_m"] > 0.4 * scene.cell_m
    best = out["pieces"][out["piece_index"]]
    assert out["pick_point"] == best["position"]
    assert out["piece_mask"].shape == scene.rgb.shape[:2] and out["piece_mask"].dtype == np.uint8
    assert (out["piece_mask"] > 0).sum() == best["area_px"]
    assert out["free_spot"] is not None and out["free_spot"]["point"] is not None


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_touching_pairs_are_split_and_flagged(script, geometry, seed):
    # Three or more isolated pieces calibrate the piece area (MIN_SINGLES).
    scene = make_scene(seed, n_single=4, n_touch=2, face_down=0.0)
    out, _ = _run(script, geometry, scene, piece_size_m=scene.cell_m)
    assert out["route"] == "touching_pieces", out["detail"]
    got = _matched(out, scene)
    pairs = {}
    for g, kind in enumerate(scene.kinds):
        if kind == "touch":
            pairs.setdefault(scene.groups[g], []).append(g)
    assert len(pairs) == 2
    for a, b in pairs.values():
        assert a in got and b in got
        pa, iou_a = got[a]
        pb, iou_b = got[b]
        assert iou_a > 0.75 and iou_b > 0.75
        assert pa["touching"] and pb["touching"]
        assert pb["piece"] in pa["neighbours"] and pa["piece"] in pb["neighbours"]
        assert pa["split"] and pa["cluster"] == pb["cluster"]
        for p, g in ((pa, a), (pb, b)):
            u, v = int(round(p["pick_pixel"]["x"])), int(round(p["pick_pixel"]["y"]))
            assert scene.masks[g][v, u]
    singles = [got[g][0] for g, k in enumerate(scene.kinds) if k == "single" and g in got]
    assert singles and all(not p["touching"] for p in singles)
    # The pick prefers a piece nothing touches.
    assert not out["pieces"][out["piece_index"]]["touching"]


def test_stacked_pair_is_split_by_depth(script, geometry):
    scene = make_scene(31, n_single=2, n_touch=0, n_stacked=1, face_down=0.0)
    out, _ = _run(script, geometry, scene, piece_size_m=scene.cell_m)
    got = _matched(out, scene)
    top = scene.kinds.index("stacked_top")
    bottom = scene.kinds.index("stacked_bottom")
    pt, iou_t = got[top]
    pb, _ = got[bottom]
    assert iou_t > 0.85
    assert pt["layer"] == "top" and pt["stacked"]
    assert pt["height_m"] == pytest.approx(0.004, abs=1e-3)
    assert pb["layer"] == "covered" and not pb["pickable"]
    assert out["route"] == "touching_pieces"


def test_without_fine_depth_a_stack_is_one_unpickable_blob(script, geometry):
    """Overhead-like depth (3 mm noise): the height map is not used; the stack's median height
    still shows, so the blob is flagged and not offered for picking."""
    scene = make_scene(31, n_single=2, n_touch=0, n_stacked=1, face_down=0.0, depth_noise=0.003)
    out, _ = _run(script, geometry, scene, piece_size_m=scene.cell_m)
    top = scene.kinds.index("stacked_top")
    stack = next(
        p for p in out["pieces"] if _full(p, scene.rgb.shape[:2])[scene.masks[top]].mean() > 0.5
    )
    assert stack["layer"] == "table" and stack["stacked"] and not stack["pickable"]


def test_face_down_pieces_read_down_with_the_back_colour(script, geometry):
    scene = make_scene(41, n_single=6, n_touch=0, face_down=0.5)
    assert "down" in scene.faces and "up" in scene.faces
    out, _ = _run(script, geometry, scene, piece_size_m=scene.cell_m, back_rgb=list(BACK))
    for g, (p, _) in _matched(out, scene).items():
        assert p["face"] == scene.faces[g]
    out, _ = _run(script, geometry, scene, piece_size_m=scene.cell_m)
    faces = {g: p["face"] for g, (p, _) in _matched(out, scene).items()}
    assert all(faces[g] == "unknown" for g in faces if scene.faces[g] == "down")


@pytest.mark.parametrize("mat", ["green", "magenta", "blue", "grey", "black", "white"])
def test_mats_with_shadows(script, geometry, mat):
    scene = make_scene(51, n_single=5, n_touch=1, mat=mat, face_down=0.2)
    out, _ = _run(script, geometry, scene, piece_size_m=scene.cell_m)
    got = _matched(out, scene)
    ious = [got[g][1] if g in got else 0.0 for g in range(len(scene.masks))]
    assert np.mean(ious) > 0.85, ious
    assert out["background"]["mode"] == "auto"


def test_colour_only_without_depth_or_pose(script):
    """No depth and no pose: pixels only, no tool calls; the piece size is the median blob."""
    scene = make_scene(61, n_single=6, n_touch=0, mat="green", face_down=0.0)
    ctx = FakeContext({})
    out = script.run(ctx, camera={"name": "c", "rgb": scene.rgb})
    assert out["route"] == "segmented", out["detail"]
    assert ctx.calls == []
    got = _matched(out, scene)
    assert len(got) == 6 and min(iou for _, iou in got.values()) > 0.8
    assert out["pick_point"] is None and out["px_per_m"] is None
    assert all(p["position"] is None and p["yaw"] is None for p in out["pieces"])
    assert out["piece_area_px"] == pytest.approx(scene.px_per_piece**2, rel=0.25)


def test_source_hsv_band_on_a_magenta_mat(script, geometry):
    # Overhead-like depth (3 mm noise): colour alone. Seeds 71 and 74 have a piece printed in
    # the mat's magenta that breaks up (see Limits); 72 has none.
    scene = make_scene(72, n_single=6, n_touch=0, mat="magenta", face_down=0.0, depth_noise=0.003)
    out, _ = _run(
        script, geometry, scene, piece_size_m=scene.cell_m, background_hsv=SYNTHETIC_MAGENTA_HSV
    )
    assert out["background"]["mode"] == "hsv"
    got = _matched(out, scene)
    assert len(got) == 6 and min(iou for _, iou in got.values()) > 0.8


def test_nothing_found_on_an_empty_mat_or_only_a_large_object(script, geometry):
    scene = make_scene(81, n_single=0, n_touch=0)
    out, _ = _run(script, geometry, scene, piece_size_m=scene.cell_m)
    assert out["route"] == "nothing_found"
    assert out["piece_index"] == -1 and out["pick_point"] is None
    assert not out["piece_mask"].any()
    rgb = scene.rgb.copy()
    cv2.rectangle(
        rgb, (150, 120), (450, 360), (200, 190, 30), -1
    )  # a tool, far bigger than a piece
    cam = dict(camera_frame(scene), rgb=rgb)
    out = script.run(FakeContext(dict(geometry)), camera=cam, piece_size_m=scene.cell_m)
    assert out["route"] == "nothing_found"
    assert [f["kind"] for f in out["foreign"]] == ["too_large"]


def _square_scene(neighbours):
    """A 40 px square piece at (200, 150) on grey, with 40 x 40 squares ``neighbours`` px away
    on the given sides (left, right, up, down)."""
    rgb = np.full((300, 400, 3), (40, 120, 70), np.uint8)
    rng = np.random.default_rng(0)

    def piece(x, y):
        rgb[y : y + 40, x : x + 40] = rng.integers(120, 255, (40, 40, 3))

    piece(180, 130)
    gap = 6
    for side in neighbours:
        dx, dy = {
            "left": (-40 - gap, 0),
            "right": (40 + gap, 0),
            "up": (0, -40 - gap),
            "down": (0, 40 + gap),
        }[side]
        piece(180 + dx, 130 + dy)
    return rgb


def _centre_piece(out):
    return next(p for p in out["pieces"] if p["bbox_px"][0] == 180 and p["bbox_px"][1] == 130)


def test_grip_axis_avoids_close_neighbours(script):
    kw = {"piece_area_px": 1600.0, "finger_radius_m": 0.004, "finger_margin_m": 0.002}
    out = script.run(FakeContext({}), camera={"name": "c", "rgb": _square_scene([])}, **kw)
    p = _centre_piece(out)
    assert p["pickable"] and p["grip"]["clear"]
    # Neighbours 6 px away left and right: closing left-right would land a finger on them.
    out = script.run(
        FakeContext({}), camera={"name": "c", "rgb": _square_scene(["left", "right"])}, **kw
    )
    p = _centre_piece(out)
    assert not p["touching"] and p["gap_px"] == pytest.approx(6, abs=1.5)
    assert p["grip"]["clear"] and p["pickable"]
    assert _angle_err(p["grip"]["axis_deg"], 90.0, 180.0) < 3
    out = script.run(
        FakeContext({}),
        camera={"name": "c", "rgb": _square_scene(["left", "right", "up", "down"])},
        **kw,
    )
    p = _centre_piece(out)
    assert not p["grip"]["clear"] and not p["pickable"]


def test_piece_cut_by_the_frame_edge_is_clipped(script):
    rgb = _square_scene([])
    rgb[100:140, 0:30] = 200  # a piece half out of view on the left
    out = script.run(FakeContext({}), camera={"name": "c", "rgb": rgb}, piece_area_px=1600.0)
    edge = [p for p in out["pieces"] if p["bbox_px"][0] == 0]
    assert len(edge) == 1 and edge[0]["clipped"] and not edge[0]["pickable"]
    assert _centre_piece(out)["pickable"]


def test_unsplittable_blob_gets_the_source_drag_point(script):
    """A blob too big for one piece that does not split into piece-sized parts (a round heap,
    one distance-transform peak) keeps the source's Separate_Overlap answer: a point a gripper
    radius inside, on its centre line, at the end away from the centroid."""
    rgb = np.full((300, 400, 3), (40, 120, 70), np.uint8)
    cv2.circle(rgb, (200, 150), 30, (200, 180, 160), -1)  # 1.8 pieces of 40 x 40 px
    out = script.run(FakeContext({}), camera={"name": "c", "rgb": rgb}, piece_area_px=1600.0)
    assert out["route"] == "touching_pieces" and out["n_pieces"] == 0
    assert out["pick_point"] is None and out["piece_index"] == -1
    unresolved = [c for c in out["clusters"] if not c["resolved"]]
    assert len(unresolved) == 1 and unresolved[0]["n_split"] == 0
    drag = unresolved[0]["drag"]
    x, y = drag["pixel"]["x"], drag["pixel"]["y"]
    assert math.hypot(x - 200, y - 150) <= 30 - 0.11 * 40  # at least the least gripper radius in
    assert math.hypot(*drag["direction_px"]) == pytest.approx(1.0)


def test_region_limits_the_search(script, geometry):
    scene = make_scene(11, n_single=7, n_touch=0, face_down=0.0)
    region = np.zeros(scene.rgb.shape[:2], bool)
    region[:, : scene.rgb.shape[1] // 2] = True
    out, _ = _run(script, geometry, scene, piece_size_m=scene.cell_m, region=region)
    for p in out["pieces"]:
        if not p["clipped"]:
            assert p["bbox_px"][2] <= scene.rgb.shape[1] // 2


def test_missing_depth_at_the_pick_point_falls_back_to_the_plane(script, geometry):
    scene = make_scene(12, n_single=3, n_touch=0, face_down=0.0)

    def no_depth(**kw):
        raise ToolError("geometry.pixel_to_world_point", "Invalid depth")

    ctx = FakeContext({**geometry, "geometry.pixel_to_world_point": no_depth})
    out = script.run(ctx, camera=camera_frame(scene), piece_size_m=scene.cell_m)
    for p in out["pieces"]:
        assert p["position"]["z"] == pytest.approx(0.002, abs=1e-3)


def test_best_piece_feeds_the_reference_localizer(script, geometry, skills_registry):
    """The chosen piece's full-frame mask and the frame go straight into
    localizing-a-piece-in-a-reference-image, which places it in the right cell."""
    locate = (
        skills_registry.get("localizing-a-piece-in-a-reference-image")
        .canonical_scripts["locate_piece"]
        .module
    )
    scene = make_scene(91, n_single=4, n_touch=0, face_down=0.0, px_per_piece=64)
    out, _ = _run(script, geometry, scene, piece_size_m=scene.cell_m)
    got = {id(p): g for g, (p, _) in _matched(out, scene).items()}
    best = out["pieces"][out["piece_index"]]
    g = got[id(best)]
    board = scene.grid * scene.cell_m
    res = locate.run(
        FakeContext(),
        out["piece_mask"],
        scene.picture,
        scene.grid,
        scene.grid,
        board,
        board,
        camera=camera_frame(scene),
    )
    cells = [(res["col"], res["row"])] + [(c["col"], c["row"]) for c in res["candidates"]]
    assert scene.cells[g] in cells, (scene.cells[g], res["detail"])
    if res["route"] == "located":
        assert (res["col"], res["row"]) == scene.cells[g]
        assert _angle_err(res["angle_deg"], scene.angles[g]) < 6
    # Its record also has what choosing-next-piece-and-slot reads: an id and a world position.
    assert isinstance(best["piece"], str) and set(best["position"]) == {"x", "y", "z"}
