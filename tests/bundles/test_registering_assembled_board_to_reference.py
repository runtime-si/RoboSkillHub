"""registering-assembled-board-to-reference: a procedural picture laid out as a board, viewed in perspective."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from gap.testing import FakeContext

from ._jigsaw_pictures import make_reference

BUNDLE = "registering-assembled-board-to-reference"
COLS, ROWS = 8, 6
MAT = (70, 96, 84)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["register_board"].module


@pytest.fixture(scope="module")
def picture():
    # 48 px cells, a white band over the top row (low texture), no plaid.
    return make_reference(48 * COLS, 48 * ROWS, seed=5, band_rows=(0.0, 1 / ROWS), plaid=False)


def _cell(c, r, img):
    h, w = img.shape[:2]
    cw, ch = w / COLS, h / ROWS
    return slice(round(r * ch), round((r + 1) * ch)), slice(round(c * cw), round((c + 1) * cw))


def _view(
    board, seed=0, jitter=0.03, corner_error=0.0, shift_px=(0.0, 0.0), margin=0.2, light=True
):
    """The board on a mat under a perspective; returns the image, the true corners and the
    corners handed to the skill (off by up to ``corner_error`` cells, plus ``shift_px`` in
    board pixels: the content slid against the corners)."""
    rng = np.random.default_rng(seed)
    h, w = board.shape[:2]
    cw, ch = w / COLS, h / ROWS
    out_w, out_h = round(w * (1 + 2 * margin)), round(h * (1 + 2 * margin))
    canvas = np.empty((out_h, out_w, 3), np.uint8)
    canvas[:] = MAT
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + [margin * w, margin * h] + rng.uniform(-jitter, jitter, (4, 2)) * [w, h]).astype(
        np.float32
    )
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(board, m, (out_w, out_h))
    cover = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (out_w, out_h)) > 0
    canvas[cover] = warped[cover]
    if light:
        lit = canvas.astype(np.float32) * np.linspace(0.85, 1.1, out_w)[None, :, None]
        canvas = np.clip(
            cv2.GaussianBlur(lit, (0, 0), 0.8) + rng.normal(0, 2, lit.shape), 0, 255
        ).astype(np.uint8)
    # Corners handed over: where the board would be had the content not slid, plus error.
    expected = cv2.perspectiveTransform((src - np.float32(shift_px)).reshape(-1, 1, 2), m).reshape(
        -1, 2
    )
    given = expected + rng.uniform(-corner_error, corner_error, (4, 2)) * [cw, ch]
    return canvas, dst, given.tolist()


def _run(script, picture, image, **kw):
    return script.run(FakeContext(), picture, COLS, ROWS, image=image, **kw)


def test_a_right_board_matches_every_cell(script, picture):
    image, _, corners = _view(picture, corner_error=0.25)
    out = _run(script, picture, image, board_corners_px=corners)
    assert out["route"] == "registered", out["detail"]
    assert out["init"] == "corners"
    assert out["n_placed"] == COLS * ROWS
    assert out["n_match"] == COLS * ROWS
    assert out["mismatched_cells"] == out["rotated_cells"] == out["empty_cells"] == []
    assert out["ecc_correlation"] > 0.9
    flat = [e for e in out["cells"] if e["row"] == 0]
    assert flat and all(e["status"] == "low_texture_match" for e in flat)


def test_the_homography_maps_the_board_onto_the_picture(script, picture):
    image, true_corners, corners = _view(picture, seed=3, corner_error=0.3)
    out = _run(script, picture, image, board_corners_px=corners)
    hm = np.asarray(out["homography"])
    got = cv2.perspectiveTransform(true_corners.reshape(-1, 1, 2).astype(np.float64), hm).reshape(
        -1, 2
    )
    h, w = picture.shape[:2]
    want = np.float64([[0, 0], [w, 0], [w, h], [0, h]])
    assert np.max(np.linalg.norm(got - want, axis=1)) < 0.1 * w / COLS


def test_swapped_turned_and_missing_pieces_are_found(script, picture):
    board = picture.copy()
    a, b = (2, 2), (5, 4)
    ya, xa = _cell(*a, board)
    yb, xb = _cell(*b, board)
    board[ya, xa], board[yb, xb] = picture[yb, xb], picture[ya, xa]
    yr, xr = _cell(6, 1, board)
    board[yr, xr] = np.rot90(picture[yr, xr], 2)
    ye, xe = _cell(1, 3, board)
    board[ye, xe] = MAT
    image, _, corners = _view(board, seed=1, corner_error=0.2)
    out = _run(script, picture, image, board_corners_px=corners)
    assert out["route"] == "registered", out["detail"]
    assert sorted(out["mismatched_cells"]) == [[2, 2], [5, 4]]
    assert out["rotated_cells"] == [[6, 1]]
    assert out["empty_cells"] == [[1, 3]]
    cells = {(e["col"], e["row"]): e for e in out["cells"]}
    assert cells[(6, 1)]["turned_deg"] == 180
    assert cells[(2, 2)]["belongs_at"]["col"] == 5 and cells[(2, 2)]["belongs_at"]["row"] == 4
    assert cells[(5, 4)]["belongs_at"]["col"] == 2 and cells[(5, 4)]["belongs_at"]["row"] == 2
    assert out["n_match"] == COLS * ROWS - 4


def test_a_quarter_turn_is_found_in_square_cells(script, picture):
    board = picture.copy()
    y, x = _cell(3, 3, board)
    board[y, x] = np.rot90(picture[y, x], 1)  # content turned 90 degrees counter-clockwise
    image, _, corners = _view(board, seed=2)
    out = _run(script, picture, image, board_corners_px=corners)
    cells = {(e["col"], e["row"]): e for e in out["cells"]}
    assert cells[(3, 3)]["status"] == "rotated"
    assert cells[(3, 3)]["turned_deg"] == 90


def test_only_the_placed_cells_are_used_and_judged(script, picture):
    board = picture.copy()
    placed = [[c, r] for r in range(2, 4) for c in range(COLS)]
    for r in range(ROWS):
        for c in range(COLS):
            if [c, r] not in placed:
                y, x = _cell(c, r, board)
                board[y, x] = MAT
    image, _, corners = _view(board, seed=4, corner_error=0.2)
    out = _run(script, picture, image, board_corners_px=corners, placed_cells=placed)
    assert out["route"] == "registered", out["detail"]
    assert out["n_placed"] == len(placed) == len(out["cells"])
    assert out["n_match"] == len(placed)
    assert out["motion"] == "homography"
    few = placed[:6]
    out = _run(script, picture, image, board_corners_px=corners, placed_cells=few)
    assert out["route"] == "registered", out["detail"]
    assert out["motion"] == "affine"
    assert out["n_match"] == len(few)


def test_features_stand_in_for_missing_corners(script, picture):
    image, _, _ = _view(picture, seed=6)
    out = _run(script, picture, image)
    assert out["route"] == "registered", out["detail"]
    assert out["init"] == "features"
    assert out["n_match"] == COLS * ROWS


def test_far_off_corners_fall_back_to_features(script, picture):
    image, _, corners = _view(picture, seed=7)
    off = (np.asarray(corners) + [3 * 48, 0]).tolist()
    out = _run(script, picture, image, board_corners_px=off)
    assert out["route"] == "registered", out["detail"]
    assert out["init"] == "features"
    assert out["correction_cells"] > 1.0
    out = _run(script, picture, image, board_corners_px=off, use_features=False)
    assert out["route"] == "did_not_converge"


def test_a_slid_assembly_shows_as_a_content_shift(script, picture):
    image, _, corners = _view(picture, seed=8, jitter=0.0, shift_px=(12.0, -6.0), light=False)
    out = _run(script, picture, image, board_corners_px=corners, board_width=0.4, board_height=0.3)
    assert out["route"] == "registered", out["detail"]
    sx, sy = out["content_shift_px"]
    assert abs(sx - 12.0) < 2.0 and abs(sy + 6.0) < 2.0
    mx, my = out["content_shift_m"]
    assert abs(mx - 12.0 / picture.shape[1] * 0.4) < 0.003
    assert abs(my + 6.0 / picture.shape[0] * 0.3) < 0.003


def test_a_wrong_picture_does_not_register(script, picture):
    other = make_reference(48 * COLS, 48 * ROWS, seed=11, band_rows=(0.0, 0.0), plaid=False)
    image, _, corners = _view(other, seed=9)
    out = _run(script, picture, image, board_corners_px=corners)
    assert out["route"] == "did_not_converge"
    assert "cells" not in out


def test_too_little_placed(script, picture):
    image, _, corners = _view(picture)
    out = _run(
        script, picture, image, board_corners_px=corners, placed_cells=[[0, 2], [1, 2], [2, 2]]
    )
    assert out["route"] == "too_little_placed"
    # The white band is low-texture: four placed cells there give nothing to align on.
    band = [[c, 0] for c in range(4)]
    assert (
        _run(script, picture, image, board_corners_px=corners, placed_cells=band)["route"]
        == "too_little_placed"
    )


def test_cells_outside_the_view_are_unseen(script, picture):
    image, _, corners = _view(picture, seed=10, margin=0.2)
    h, w = image.shape[:2]
    cut = image[:, : int(0.7 * w)].copy()
    out = _run(script, picture, cut, board_corners_px=corners, use_features=False)
    assert out["route"] == "registered", out["detail"]
    unseen = [(e["col"], e["row"]) for e in out["cells"] if e["status"] == "unseen"]
    assert unseen and all(c >= COLS - 3 for c, _ in unseen)


def test_camera_frame_and_bad_inputs(script, picture):
    image, _, corners = _view(picture)
    camera = {"name": "top", "rgb": image}
    out = script.run(FakeContext(), picture, COLS, ROWS, camera=camera, board_corners_px=corners)
    assert out["route"] == "registered"
    assert script.run(FakeContext(), picture, COLS, ROWS)["route"] == "did_not_converge"
    assert (
        script.run(FakeContext(), "/nonexistent.png", COLS, ROWS, image=image)["route"]
        == "did_not_converge"
    )
    assert script.run(FakeContext(), picture, 0, ROWS, image=image)["route"] == "did_not_converge"
