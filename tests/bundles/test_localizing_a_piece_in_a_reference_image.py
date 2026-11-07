"""localizing-a-piece-in-a-reference-image: pieces cut from a procedural picture are put back."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from gap.testing import FakeContext

from ._jigsaw_pictures import cut_piece, make_reference

BUNDLE = "localizing-a-piece-in-a-reference-image"
COLS, ROWS = 8, 8
BOARD = (0.16, 0.16)
WRIST_PX = 96
CELL_M = BOARD[0] / COLS
PPM = WRIST_PX / CELL_M
FX = 400.0


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["locate_piece"].module


@pytest.fixture(scope="module")
def picture():
    # A flat white band over rows 0-1, no plaid.
    return make_reference(36 * COLS, 36 * ROWS, seed=3, band_rows=(0.0, 2 / ROWS), plaid=False)


@pytest.fixture(scope="module")
def textured():
    return make_reference(36 * COLS, 36 * ROWS, seed=4, band_rows=(0.0, 0.0), plaid=False)


def _camera(img, mask, depth_m=FX / PPM, size=320):
    """Put the crop in the middle of a wrist frame whose depth gives exactly the crop's scale."""
    frame = np.full((size, size, 3), 30, np.uint8)
    full = np.zeros((size, size), bool)
    y0, x0 = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    frame[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img
    full[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = mask
    camera = {
        "name": "wrist",
        "rgb": frame,
        "depth": np.full((size, size), depth_m, np.float32),
        "intrinsics": np.array([[FX, 0, size / 2], [0, FX, size / 2], [0, 0, 1.0]]),
        "pose": {
            "position": {"x": 0, "y": 0, "z": 0},
            "rotation": {"w": 1, "x": 0, "y": 0, "z": 0},
        },
    }
    return camera, full


def _locate(script, reference, mask, **kw):
    return script.run(FakeContext(), mask, reference, COLS, ROWS, BOARD[0], BOARD[1], **kw)


def _angle_err(a, b):
    return abs((a - b + 180) % 360 - 180)


@pytest.mark.parametrize(("col", "row", "seed"), [(3, 4, 1), (5, 6, 2), (2, 3, 3)])
def test_interior_piece_from_the_wrist_frame_is_located(script, picture, col, row, seed):
    img, mask, angle, _ = cut_piece(picture, COLS, ROWS, col, row, WRIST_PX, seed=seed)
    camera, full = _camera(img, mask)
    out = _locate(script, picture, full, camera=camera)
    assert out["route"] == "located", out["detail"]
    assert (out["col"], out["row"]) == (col, row)
    assert _angle_err(out["angle_deg"], angle) < 3
    assert "scale from depth" in out["detail"]
    centre = ((col + 0.5) * CELL_M, (row + 0.5) * CELL_M)
    assert out["cell_center_m"] == pytest.approx({"x": centre[0], "y": centre[1]})
    assert abs(out["position_m"]["x"] - centre[0]) < 0.3 * CELL_M
    assert abs(out["position_m"]["y"] - centre[1]) < 0.3 * CELL_M
    assert out["margin"] >= script.MIN_MARGIN and out["score"] >= script.MIN_SCORE
    assert out["candidates"][0]["col"] == col and out["candidates"][0]["row"] == row
    assert full[round(out["grip_pixel"]["y"]), round(out["grip_pixel"]["x"])]


@pytest.mark.parametrize(("col", "row"), [(0, 0), (7, 0), (0, 7), (7, 7), (4, 7), (7, 3)])
def test_border_and_corner_pieces_match_on_their_straight_sides(script, textured, col, row):
    img, mask, angle, _ = cut_piece(textured, COLS, ROWS, col, row, WRIST_PX, seed=col * 10 + row)
    out = _locate(script, textured, mask, piece_rgb=img, crop_px_per_m=PPM)
    assert out["route"] == "located", out["detail"]
    assert (out["col"], out["row"]) == (col, row)
    assert _angle_err(out["angle_deg"], angle) < 3


def test_quarter_turns_only(script, textured):
    img, mask, _, _ = cut_piece(textured, COLS, ROWS, 4, 2, WRIST_PX, angle=90.0, seed=5)
    out = _locate(
        script, textured, mask, piece_rgb=img, crop_px_per_m=PPM, allowed_angles=[0, 90, 180, 270]
    )
    assert out["route"] == "located"
    assert (out["col"], out["row"]) == (4, 2)
    assert _angle_err(out["angle_deg"], 90.0) < 3
    assert "4 coarse angles" in out["detail"]


def test_scale_guessed_from_the_mask_area(script, textured):
    img, mask, angle, _ = cut_piece(textured, COLS, ROWS, 3, 3, WRIST_PX, seed=8)
    out = _locate(script, textured, mask, piece_rgb=img)
    assert "scale from mask area" in out["detail"]
    assert (out["col"], out["row"]) == (3, 3)
    assert _angle_err(out["angle_deg"], angle) < 4


def test_a_flat_piece_is_ambiguous_with_the_flat_cells_of_its_colour(script, picture):
    img, mask, _, _ = cut_piece(picture, COLS, ROWS, 3, 1, WRIST_PX, seed=6)
    out = _locate(script, picture, mask, piece_rgb=img, crop_px_per_m=PPM)
    assert out["low_texture"]
    assert out["route"] == "ambiguous"
    cells = {(c["col"], c["row"]) for c in out["candidates"]}
    assert (3, 1) in cells
    assert all(row in (0, 1) for _, row in cells)
    assert out["candidates"][0]["angle_deg"] is None


def test_a_periodic_picture_is_ambiguous(script):
    tile = make_reference(36, 36, seed=9, band_rows=(0.0, 0.0), plaid=False)
    periodic = np.tile(tile, (ROWS, COLS, 1))
    img, mask, _, _ = cut_piece(periodic, COLS, ROWS, 3, 4, WRIST_PX, seed=1)
    out = _locate(script, periodic, mask, piece_rgb=img, crop_px_per_m=PPM)
    assert out["route"] == "ambiguous"
    assert out["margin"] < script.MIN_MARGIN
    assert len(out["candidates"]) == script.TOP_K


def test_a_piece_from_another_picture_is_not_located(script, picture, textured):
    img, mask, _, _ = cut_piece(textured, COLS, ROWS, 3, 4, WRIST_PX, seed=1)
    out = _locate(script, picture, mask, piece_rgb=img, crop_px_per_m=PPM, min_score=0.9)
    assert out["route"] == "not_found"


def test_search_cells_restrict_the_answer(script, textured):
    img, mask, _, _ = cut_piece(textured, COLS, ROWS, 3, 3, WRIST_PX, seed=2)
    out = _locate(
        script, textured, mask, piece_rgb=img, crop_px_per_m=PPM, search_cells=[[0, 0], [1, 0]]
    )
    assert (out["col"], out["row"]) in {(0, 0), (1, 0)}
    assert all((c["col"], c["row"]) in {(0, 0), (1, 0)} for c in out["candidates"])
    assert out["route"] != "located" or out["score"] < 0.8


def test_reference_from_a_file_and_pose_passthrough(script, textured, tmp_path):
    path = tmp_path / "box.png"
    cv2.imwrite(str(path), cv2.cvtColor(textured, cv2.COLOR_RGB2BGR))
    img, mask, _, _ = cut_piece(textured, COLS, ROWS, 6, 5, WRIST_PX, seed=4)
    pose = {
        "position": {"x": 0.3, "y": 0.1, "z": 0.0},
        "rotation": {"w": 1, "x": 0, "y": 0, "z": 0},
    }
    out = _locate(script, str(path), mask, piece_rgb=img, crop_px_per_m=PPM, piece_pose=pose)
    assert (out["col"], out["row"]) == (6, 5)
    assert out["piece_pose"] == pose
    # The native picture is 288 px wide for 0.16 m.
    assert out["position_px"]["x"] == pytest.approx(out["position_m"]["x"] * 288 / 0.16)


def test_bad_inputs_raise(script, textured):
    img, mask, _, _ = cut_piece(textured, COLS, ROWS, 3, 3, WRIST_PX, seed=2)
    with pytest.raises(ValueError, match="camera"):
        _locate(script, textured, mask)
    with pytest.raises(ValueError, match="does not match"):
        _locate(script, textured, mask[1:], piece_rgb=img)
    camera, full = _camera(img, mask)
    with pytest.raises(ValueError, match="empty"):
        _locate(script, textured, np.zeros_like(full), camera=camera)
    with pytest.raises(ValueError, match="positive"):
        script.run(FakeContext(), mask, textured, 0, ROWS, *BOARD, piece_rgb=img)
    with pytest.raises(ValueError, match="cannot read"):
        _locate(script, "/nonexistent/box.png", mask, piece_rgb=img)
    with pytest.raises(ValueError, match="search_cells"):
        _locate(script, textured, mask, piece_rgb=img, crop_px_per_m=PPM, search_cells=[[99, 99]])


def test_every_route_is_declared_in_the_example():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
    example = json.loads((root / "examples/canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == {"located", "ambiguous", "not_found"}
