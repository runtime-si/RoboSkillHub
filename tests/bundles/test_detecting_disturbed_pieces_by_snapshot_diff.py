"""detecting-disturbed-pieces-by-snapshot-diff: synthetic RGB-D boards before and after."""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest
from gap.testing import FakeContext

from . import _snapshot_scenes as S

BUNDLE = "detecting-disturbed-pieces-by-snapshot-diff"


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["diff_snapshots"].module


def _scene(kind, cam, seed=1):
    sc = S.build(kind, seed, cam)
    t, k = S.camera_pose(cam)
    ref, ids = S.render(sc, t, k, seed=seed, ss=2)
    items = [{"id": i, "mask": m} for i, m in S.item_masks(sc, ids).items() if m.sum() >= 4]
    return sc, t, k, ref, items


@pytest.fixture(scope="module")
def lego_wrist():
    return _scene("lego", "wrist")


@pytest.fixture(scope="module")
def jigsaw_overhead():
    return _scene("jigsaw", "overhead")


@pytest.fixture(scope="module")
def domino_wrist():
    return _scene("domino", "wrist")


def _render(sc, t, k, seed=7, **kw):
    return S.render(sc, t, k, seed=seed, ss=2, **kw)[0]


def _by_id(out):
    return {r["id"]: r for r in out["items"]}


def test_unchanged_board_with_new_sensor_noise(script, jigsaw_overhead):
    sc, t, k, ref, items = jigsaw_overhead
    out = script.run(FakeContext(), ref, items, current=_render(sc, t, k))
    assert out["route"] == "unchanged", out["detail"]
    assert all(r["status"] == "unchanged" for r in out["items"])
    assert out["registration"]["rho"] > 0.95
    assert out["n_disturbed"] == 0 and out["n_unexpected"] == 0


def test_lighting_drift_and_camera_bump_are_not_disturbances(script, jigsaw_overhead):
    sc, t, k, ref, items = jigsaw_overhead
    lg = S.Lighting(
        gain=0.85, tint=(1.04, 1.0, 0.96), gradient=(0.15, -0.1), shadow=(320, 240, 120, 0.15)
    )
    bumped = S.bump(t, np.random.default_rng(3), rot_deg=0.3, trans_m=0.002)
    cur = _render(sc, bumped, k, lighting=lg)
    out = script.run(FakeContext(), ref, items, current=cur)
    assert out["route"] == "unchanged", out["detail"]
    assert out["camera_shift_px"] > 1.0  # it did register a real camera move


def test_a_brick_pushed_4_mm_is_disturbed_with_its_shift(script, lego_wrist):
    sc, t, k, ref, items = lego_wrist
    cur = _render(S.shift(sc, "b4", 0.004 * math.cos(0.5), 0.004 * math.sin(0.5)), t, k)
    out = script.run(FakeContext(), ref, items, current=cur)
    assert out["route"] == "disturbed", out["detail"]
    assert out["disturbed"] == ["b4"]
    rec = _by_id(out)["b4"]
    assert rec["status"] == "moved"
    assert rec["shift_mm"] == pytest.approx(4.0, abs=0.5)
    w = rec["translation_world"]
    assert math.atan2(w["y"], w["x"]) == pytest.approx(0.5, abs=0.15)
    assert abs(w["z"]) < 0.001
    assert out["max_shift_mm"] == rec["shift_mm"]
    assert "b4 moved" in out["detail"]


def test_a_turned_piece_reports_its_turn_counter_clockwise_from_above(script):
    sc, t, k, ref, items = _scene("jigsaw", "wrist", seed=2)
    out = script.run(
        FakeContext(), ref, items, current=_render(S.shift(sc, "c1_1", 0, 0, 8.0), t, k)
    )
    rec = _by_id(out)["c1_1"]
    assert out["route"] == "disturbed" and rec["status"] == "moved"
    assert rec["rotation_deg"] == pytest.approx(8.0, abs=1.0)
    out = script.run(
        FakeContext(), ref, items, current=_render(S.shift(sc, "c1_1", 0, 0, -8.0), t, k)
    )
    assert _by_id(out)["c1_1"]["rotation_deg"] == pytest.approx(-8.0, abs=1.0)


def test_a_toppled_domino_is_lost_with_its_depth_change(script, domino_wrist):
    sc, t, k, ref, items = domino_wrist
    out = script.run(FakeContext(), ref, items, current=_render(S.topple(sc, "d2", 90.0), t, k))
    assert out["route"] == "disturbed", out["detail"]
    rec = _by_id(out)["d2"]
    assert rec["status"] in ("lost", "moved")
    assert rec["depth_change_mm"] > 10.0  # its top is now far lower: farther from the camera


def test_a_removed_item_is_lost(script, lego_wrist):
    sc, t, k, ref, items = lego_wrist
    out = script.run(FakeContext(), ref, items, current=_render(S.remove(sc, "b5"), t, k))
    assert out["route"] == "disturbed"
    assert _by_id(out)["b5"]["status"] == "lost"
    assert out["disturbed"] == ["b5"]


def test_an_object_set_down_is_unexpected_and_located(script, lego_wrist):
    sc, t, k, ref, items = lego_wrist
    region = {
        "center": {"x": 0, "y": 0, "z": 0.02},
        "extent": {"x": 0.11, "y": 0.11, "z": 0.03},
        "orientation": {"w": 1, "x": 0, "y": 0, "z": 0},
    }
    cur = _render(S.add_object(sc, (0.06, 0.07), size=(0.016, 0.016, 0.012)), t, k)
    out = script.run(FakeContext(), ref, items, current=cur, region=region)
    assert out["route"] == "unexpected_object", out["detail"]
    blob = max(out["unexpected"], key=lambda b: b["area_px"])
    assert blob["kind"] == "appeared"
    p = blob["position_world"]
    assert math.hypot(p["x"] - 0.06, p["y"] - 0.07) < 0.008
    assert blob["area_mm2"] > 150


def test_a_hand_over_the_board_is_an_occluder_not_a_disturbance(script, jigsaw_overhead):
    sc, t, k, ref, items = jigsaw_overhead
    target = sc.get("c1_1").center
    cur = _render(
        S.add_hand(sc, (float(target[0]), float(target[1])), height=0.12, direction_deg=-90), t, k
    )
    out = script.run(FakeContext(), ref, items, current=cur)
    assert out["route"] == "occluded", out["detail"]
    assert out["occluders"] and out["occluders"][0]["depth_change_mm"] < -50
    # Seen from the oblique camera a hand 12 cm up covers the pieces beyond the one it is over.
    assert out["occluded"] and all(
        _by_id(out)[i]["visible_fraction"] < 0.6 for i in out["occluded"]
    )
    assert out["n_disturbed"] == 0


def test_the_robot_arm_is_masked_from_its_pose(script, jigsaw_overhead):
    sc, t, k, ref, items = jigsaw_overhead
    target = sc.get("c2_1").center
    scene = S.add_arm(sc, (float(target[0]), float(target[1]), 0.09), (0.0, -0.45, 0.2))
    obs = S.observation(_render(scene, t, k), scene)
    out = script.run(FakeContext(), ref, items, current=obs, arm_base_positions=scene.arm_bases)
    assert out["route"] == "occluded", out["detail"]
    assert out["n_disturbed"] == 0 and out["n_unexpected"] == 0
    # Without the arm's pose the same frame is not clean.
    bare = script.run(FakeContext(), ref, items, current=obs, mask_arms=False)
    assert bare["route"] != "unchanged" or bare["occluders"]


def test_expected_items_and_ignore_masks_do_not_count(script, lego_wrist):
    sc, t, k, ref, items = lego_wrist
    cur = _render(S.shift(sc, "b2", 0.005, 0.0), t, k)
    out = script.run(FakeContext(), ref, items, current=cur, expected_ids=["b2"])
    assert out["route"] == "unchanged", out["detail"]
    rec = _by_id(out)["b2"]
    assert rec["expected"] and rec["status"] == "moved"  # still measured
    big = next(it["mask"] for it in items if it["id"] == "b2")
    ignore = cv2.dilate(big.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    out = script.run(
        FakeContext(), ref, items, current=cur, ignore_mask=ignore, reference_ignore_mask=ignore
    )
    assert out["n_disturbed"] == 0
    assert _by_id(out)["b2"]["status"] == "occluded"


def test_another_scene_is_unregistrable(script, lego_wrist):
    _, t, k, ref, items = lego_wrist
    other = S.build("jigsaw", 9, "wrist")
    cur = _render(other, S.look_at([0.3, 0.2, 0.25], [0.3, 0.2, 0.0]), k)
    out = script.run(FakeContext(), ref, items, current=cur)
    assert out["route"] == "unregistrable", out["detail"]
    assert out["items"] == []


def test_cells_of_a_world_grid_and_world_boxes(script):
    sc, t, k, ref, _ = _scene("jigsaw", "wrist", seed=3)
    g = sc.grid
    origin = {
        "position": {"x": g["x0"], "y": g["y0"], "z": 0.0},
        "rotation": {"w": 1, "x": 0, "y": 0, "z": 0},
    }
    grid = {
        "origin": origin,
        "cols": g["cols"],
        "rows": g["rows"],
        "cell_size": g["cell"],
        "item_height": g["thick"],
    }
    cells = [{"id": b.id, "cell": [int(b.id[1]), int(b.id[3])]} for b in sc.items()]
    cur = _render(S.shift(sc, "c2_0", 0.0, -0.003), t, k)
    out = script.run(FakeContext(), ref, cells, current=cur, grid=grid)
    assert out["disturbed"] == ["c2_0"], out["detail"]
    assert _by_id(out)["c2_0"]["translation_world"]["y"] == pytest.approx(-0.003, abs=0.0007)
    boxes = [{"id": b.id, "box": b.obb()} for b in sc.items()]
    out = script.run(FakeContext(), ref, boxes, current=cur)
    assert out["disturbed"] == ["c2_0"], out["detail"]


def test_cells_of_a_pixel_grid_map_blobs_to_cells(script):
    sc, t, k, ref, items = _scene("jigsaw", "wrist", seed=4)
    g = sc.grid
    corners_w = np.array(
        [
            [g["x0"], g["y0"]],
            [g["x0"] + g["cols"] * g["cell"], g["y0"]],
            [g["x0"] + g["cols"] * g["cell"], g["y0"] + g["rows"] * g["cell"]],
            [g["x0"], g["y0"] + g["rows"] * g["cell"]],
        ]
    )
    uv, _ = script._project(k, t, np.c_[corners_w, np.full(4, g["thick"])])
    grid = {"cols": g["cols"], "rows": g["rows"], "corners_px": uv.tolist()}
    # Pixel rows run against world y here, so cell (c, r) of this grid is world cell (c, rows-1-r).
    cells = [{"id": b.id, "cell": [int(b.id[1]), g["rows"] - 1 - int(b.id[3])]} for b in sc.items()]
    free = (5, 2)  # an empty cell of the partial assembly
    xy = (g["x0"] + (free[0] + 0.5) * g["cell"], g["y0"] + (free[1] + 0.5) * g["cell"])
    cur = _render(S.add_object(sc, xy, size=(0.012, 0.012, 0.004)), t, k)
    out = script.run(FakeContext(), ref, cells, current=cur, grid=grid)
    assert out["route"] == "unexpected_object", out["detail"]
    assert out["unexpected"][0]["cell"] == [free[0], g["rows"] - 1 - free[1]]


def test_rgb_only_uses_the_pose_and_plane_or_a_given_scale(script, lego_wrist):
    sc, t, k, ref, items = lego_wrist
    cur = _render(S.shift(sc, "b0", -0.003, 0.0), t, k)
    rgb_ref = {**ref, "depth": None}
    rgb_cur = {**cur, "depth": None}
    out = script.run(FakeContext(), rgb_ref, items, current=rgb_cur, plane_z=0.0128)
    assert _by_id(out)["b0"]["shift_mm"] == pytest.approx(3.0, abs=0.5)
    out = script.run(
        FakeContext(),
        rgb_ref["rgb"],
        items,
        current=rgb_cur["rgb"],
        mm_per_px=1000 * 0.2372 / k[0, 0],
    )
    assert out["disturbed"] == ["b0"]
    assert _by_id(out)["b0"]["shift_mm"] == pytest.approx(3.0, abs=0.5)
    out = script.run(FakeContext(), rgb_ref["rgb"], items, current=rgb_cur["rgb"])
    rec = _by_id(out)["b0"]
    assert rec["shift_mm"] is None and abs(rec["translation_px"][0]) > 4
    assert "px only" in out["detail"]


def test_current_defaults_to_a_fresh_observation(script, lego_wrist):
    sc, t, k, ref, items = lego_wrist
    obs = {
        "cameras": [{**_render(sc, t, k), "name": "wrist"}, {**ref, "name": "other"}],
        "arms": [],
    }
    ctx = FakeContext({"robot.get_observation": obs})
    out = script.run(
        ctx, {"cameras": [{**ref, "name": "wrist"}], "arms": []}, items, camera_name="wrist"
    )
    assert out["route"] == "unchanged", out["detail"]
    assert [c.tool for c in ctx.calls] == ["robot.get_observation"]


def test_large_frames_are_worked_on_shrunk_and_reported_in_input_pixels(script, lego_wrist):
    sc, t, k, ref, items = lego_wrist
    cur = _render(S.shift(sc, "b3", 0.004, 0.0), t, k)

    def up(f):
        g = dict(f)
        g["rgb"] = cv2.resize(f["rgb"], (1920, 1440), interpolation=cv2.INTER_CUBIC)
        g["depth"] = cv2.resize(f["depth"], (1920, 1440), interpolation=cv2.INTER_NEAREST)
        kk = f["intrinsics"].copy()
        kk[:2] *= 3
        g["intrinsics"] = kk
        return g

    big_items = [
        {
            "id": it["id"],
            "mask": cv2.resize(
                it["mask"].astype(np.uint8), (1920, 1440), interpolation=cv2.INTER_NEAREST
            ),
        }
        for it in items
    ]
    out = script.run(FakeContext(), up(ref), big_items, current=up(cur))
    rec = _by_id(out)["b3"]
    assert out["disturbed"] == ["b3"]
    assert rec["shift_mm"] == pytest.approx(4.0, abs=0.6)
    small = script.run(FakeContext(), ref, items, current=cur)
    assert rec["translation_px"][0] == pytest.approx(
        3 * _by_id(small)["b3"]["translation_px"][0], rel=0.15
    )


def test_ssim_map_matches_scikit_image(script):
    skm = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(0)
    a = cv2.GaussianBlur(rng.uniform(0, 255, (64, 80)).astype(np.float32), (0, 0), 1.0)
    b = np.clip(a * 0.9 + rng.normal(0, 6, a.shape), 0, 255).astype(np.float32)
    _, ref_map = skm.structural_similarity(
        a, b, gaussian_weights=True, full=True, data_range=255.0, sigma=1.5
    )
    ours = script._ssim_map(a, b)
    pad = 5
    assert np.abs(ours[pad:-pad, pad:-pad] - ref_map[pad:-pad, pad:-pad]).max() < 1e-3


def test_example_declares_every_route(script):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
    ex = json.loads((root / "examples" / "canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert declared == {"unchanged", "disturbed", "unexpected_object", "occluded", "unregistrable"}
    text = (root / "scripts" / "diff_snapshots.py").read_text()
    for route in declared:
        assert f'"{route}"' in text
