"""verifying-piece-flushness-from-wrist-depth: rendered pieces among neighbours under a tilted wrist camera."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext

from . import _flush_scene as S

BUNDLE = "verifying-piece-flushness-from-wrist-depth"
K = np.array([[394.4, 0.0, 321.9], [0.0, 394.4, 246.6], [0.0, 0.0, 1.0]])
SHAPE = (480, 640)
NOISE = 0.0004
"""Per-pixel depth noise std; real D405 frames at 8-16 cm read 0.39-0.44 mm."""


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["verify_flush"].module


def _frame(scene, seed=0, pose=None, noise=NOISE, **kw):
    pose = S.look_down() if pose is None else pose
    depth = S.render(scene, K, pose, SHAPE, **kw)
    depth = depth + np.random.default_rng(seed).normal(0.0, noise, SHAPE)
    return S.camera(depth, K, pose)


def _run(script, cam, scene, mode="neighbours", **kw):
    if mode == "neighbours":
        kw.setdefault("neighbour_outlines", scene.neighbours)
    elif mode == "table":
        kw.setdefault("table_z", 0.0)
    return script.run(FakeContext(), cam, piece_outline=scene.piece, **kw)


@pytest.mark.parametrize("mode", ["neighbours", "table", "ring"])
@pytest.mark.parametrize("neighbours", [("west", "south"), ("west", "south", "east", "north")])
def test_a_flush_piece_is_flush(script, mode, neighbours):
    scene = S.Scene(seed=1, neighbours=neighbours)
    out = _run(script, _frame(scene), scene, mode)
    assert out["route"] == "flush", out["detail"]
    assert abs(out["level_m"]) < 0.0002
    assert out["tilt_rise_m"] < 0.0004
    assert out["reference"] == "neighbours"


def test_a_lone_piece_flat_on_the_table_is_flush(script):
    scene = S.Scene(seed=2, neighbours=())
    for kw in ({}, {"table_z": 0.0}):
        out = script.run(FakeContext(), _frame(scene), piece_outline=scene.piece, **kw)
        assert out["route"] == "flush", out["detail"]
        assert out["reference"] == "table"


@pytest.mark.parametrize("mode", ["neighbours", "table", "ring"])
def test_a_raised_piece_is_raised(script, mode):
    scene = S.Scene(seed=3, neighbours=("west", "south", "east"))
    out = _run(script, _frame(scene, raise_m=0.001), scene, mode)
    assert out["route"] == "raised", out["detail"]
    assert abs(out["level_m"] - 0.001) < 0.0002


def test_a_piece_with_one_side_up_is_tilted_toward_that_side(script):
    scene = S.Scene(seed=4, neighbours=("west", "south"))
    out = _run(script, _frame(scene, tilt_m=0.002, tilt_dir=(0.0, 1.0)), scene)
    assert out["route"] == "tilted", out["detail"]
    assert out["tilt_rise_m"] > 0.0015
    assert out["high_side"][1] > 0.9


def test_a_piece_short_of_its_slot_leaves_an_open_seam(script):
    scene = S.Scene(seed=5, neighbours=("west", "south"))
    out = _run(script, _frame(scene, shift_m=(0.0012, 0.0)), scene)
    assert out["route"] == "gap", out["detail"]
    assert out["gap_kind"] == "seam"
    assert out["seam_gap_m"] > 0.008
    # The gap is on the west seam: its points lie west of the piece's centre.
    xs = [p[0] for p in out["seam_gap_points"]]
    assert np.median(xs) < -0.008
    # Without neighbour outlines there is no seam test.
    assert _run(script, _frame(scene, shift_m=(0.0012, 0.0)), scene, "table")["route"] == "flush"


def test_raised_neighbours_make_the_piece_sunk(script):
    scene = S.Scene(seed=6, neighbours=("west", "south", "east", "north"))
    out = _run(script, _frame(scene, neighbour_raise_m=0.0012), scene)
    assert out["route"] == "gap", out["detail"]
    assert out["gap_kind"] == "low"
    assert out["level_m"] < -0.001


def test_a_closed_seam_next_to_an_empty_diagonal_cell_is_not_a_gap(script):
    scene = S.Scene(seed=7, neighbours=("west", "south", "east", "north"))
    out = _run(script, _frame(scene, seed=3), scene)
    assert out["route"] == "flush", out["detail"]
    assert out["seam_gap_m"] < 0.003


def test_a_piece_mask_works_like_an_outline(script):
    scene = S.Scene(seed=8, neighbours=("west", "south"))
    pose = S.look_down()
    cam = _frame(scene, pose=pose, raise_m=0.001)
    out = _run(script, cam, scene)
    assert out["route"] == "raised"
    # The same piece as an image mask (projected at its top height).
    import cv2

    poly = np.column_stack(
        [scene.piece, np.full(len(scene.piece), S.THICK), np.ones(len(scene.piece))]
    )
    camp = (np.linalg.inv(pose) @ poly.T).T
    uv = np.column_stack(
        [K[0, 0] * camp[:, 0] / camp[:, 2] + K[0, 2], K[1, 1] * camp[:, 1] / camp[:, 2] + K[1, 2]]
    )
    mask = np.zeros(SHAPE, np.uint8)
    cv2.fillPoly(mask, [np.round(uv).astype(np.int32)], 1)
    out = script.run(
        FakeContext(), cam, piece_mask=mask.astype(bool), neighbour_outlines=scene.neighbours
    )
    assert out["route"] == "raised", out["detail"]


def test_a_matrix_pose_is_accepted(script):
    scene = S.Scene(seed=9, neighbours=("west",))
    pose = S.look_down(height=0.1, tilt_deg=12.0, yaw_deg=-40.0)
    cam = _frame(scene, pose=pose)
    cam["pose"] = pose
    assert _run(script, cam, scene)["route"] == "flush"


def test_missing_depth_is_unverifiable(script):
    scene = S.Scene(seed=10, neighbours=("west", "south"))
    cam = _frame(scene)
    holey = dict(cam)
    depth = cam["depth"].copy()
    depth[np.random.default_rng(0).random(SHAPE) < 0.7] = 0.0
    holey["depth"] = depth
    out = _run(script, holey, scene)
    assert out["route"] == "unverifiable"
    assert out["valid_fraction"] < 0.4
    assert (
        script.run(FakeContext(), {"rgb": cam["rgb"]}, piece_outline=scene.piece)["route"]
        == "unverifiable"
    )
    assert script.run(FakeContext(), cam)["route"] == "unverifiable"
    far = (np.asarray(scene.piece) + [1.0, 0.0]).tolist()
    assert script.run(FakeContext(), cam, piece_outline=far)["route"] == "unverifiable"
    wrong = np.zeros((10, 10), bool)
    assert script.run(FakeContext(), cam, piece_mask=wrong)["route"] == "unverifiable"


def test_segment_plane_ignores_steep_planes_and_finds_two_levels(script):
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.02, 0.02, (2000, 2))
    z = np.where(xy[:, 0] < 0, 0.0, 0.002) + rng.normal(0, 0.0002, 2000)
    pts = np.column_stack([xy, z])
    # One level: a plane sloping across the step can win, but never beyond the tilt limit.
    plane, _ = script.segment_plane(pts)
    assert abs(plane[2]) >= np.cos(np.radians(script.MAX_PLANE_TILT_DEG))
    # Two parallel levels a step apart: the fit is horizontal and lands on one level.
    plane2, inl = script.segment_plane(pts, offsets=(-0.002, 0.0, 0.002))
    assert abs(plane2[2]) > 0.9999
    assert 0.35 < inl.mean() < 0.65
