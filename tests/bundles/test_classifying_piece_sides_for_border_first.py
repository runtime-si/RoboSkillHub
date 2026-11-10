"""classifying-piece-sides-for-border-first: masks of generated pieces, turned, with known side types."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from gap.testing import FakeContext

from .jigsaw_cut import make_puzzle

BUNDLE = "classifying-piece-sides-for-border-first"
PX = 64.0


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["classify_sides"].module


@pytest.fixture(scope="module")
def puzzle():
    return make_puzzle(6, 6, cell=(1.0, 1.0), seed=4)


def _mask(poly, angle=0.0, px=PX):
    c = poly.mean(axis=0)
    a = math.radians(angle)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    p = (poly - c) @ r.T * px
    size = int(2.2 * px)
    img = np.zeros((size, size), np.uint8)
    cv2.fillPoly(
        img, [np.round(np.column_stack([p[:, 0], -p[:, 1]]) + size / 2).astype(np.int32)], 255
    )
    return img


def _cyclic_equal(got, want):
    return any(got[k:] + got[:k] == want for k in range(4))


@pytest.mark.parametrize("angle", [0.0, 17.0, 45.0, 133.0, -70.0])
def test_every_piece_of_a_cut_is_typed(script, puzzle, angle):
    for i in range(puzzle.cols):
        for j in range(puzzle.rows):
            out = script.run(FakeContext(), piece_mask=_mask(puzzle.outline(i, j), angle))
            assert out["piece_class"] == puzzle.piece_class(i, j), (i, j, out["detail"])
            assert _cyclic_equal(out["side_types"], list(puzzle.side_types(i, j))), (
                i,
                j,
                out["detail"],
            )


@pytest.mark.parametrize("angle", [0.0, 30.0, -100.0])
def test_a_flat_side_points_where_the_board_edge_is(script, puzzle, angle):
    # Piece (0, 2) is on the west edge: its flat side faces west (180 deg) before turning.
    out = script.run(FakeContext(), piece_mask=_mask(puzzle.outline(0, 2), angle))
    assert out["piece_class"] == "edge"
    (normal,) = out["flat_normals_deg"]
    want = (180.0 + angle + 180.0) % 360.0 - 180.0
    assert abs((normal - want + 180.0) % 360.0 - 180.0) < 4.0
    # The corner (0, 0): flat sides west and south.
    out = script.run(FakeContext(), piece_mask=_mask(puzzle.outline(0, 0), angle))
    assert out["piece_class"] == "corner"
    got = sorted((n - angle) % 360.0 for n in out["flat_normals_deg"])
    assert abs(got[0] - 180.0) < 4.0 and abs(got[1] - 270.0) < 4.0


def test_a_list_of_pieces_is_sorted_for_a_border_first_build(script, puzzle):
    pieces = [
        {"id": f"{i},{j}", "mask": _mask(puzzle.outline(i, j), 11.0 * (i + j))}
        for i in range(puzzle.cols)
        for j in range(puzzle.rows)
    ]
    pieces.append({"id": "blank", "mask": np.zeros((50, 50), np.uint8)})
    out = script.run(FakeContext(), pieces=pieces)
    assert out["route"] == "classified"
    assert sorted(out["corners"]) == ["0,0", "0,5", "5,0", "5,5"]
    assert len(out["edges"]) == 16 and len(out["interiors"]) == 16
    assert out["uncertain"] == ["blank"]


def test_noise_on_the_outline_does_not_flip_a_side(script, puzzle):
    rng = np.random.default_rng(0)
    for i, j in [(0, 3), (2, 2), (5, 5)]:
        poly = puzzle.outline(i, j) + rng.uniform(-0.6 / PX, 0.6 / PX, puzzle.outline(i, j).shape)
        m = cv2.GaussianBlur(_mask(poly, 23.0), (5, 5), 1.0)
        out = script.run(FakeContext(), piece_mask=m)
        assert out["piece_class"] == puzzle.piece_class(i, j), out["detail"]


def test_a_side_near_the_threshold_is_uncertain(script, puzzle, monkeypatch):
    mask = _mask(puzzle.outline(2, 2), 0.0)
    devs = script.run(FakeContext(), piece_mask=mask)["side_deviations"]
    # Put the threshold right at one side's measured deviation: that side is now uncertain.
    monkeypatch.setattr(script, "KNOB_FRACTION", abs(devs[0]))
    out = script.run(FakeContext(), piece_mask=mask)
    assert out["route"] == "uncertain" and out["uncertain_sides"] >= 1
    assert script.run(FakeContext(), piece_mask=mask, max_uncertain=4)["route"] == "classified"


def test_a_square_is_not_a_piece_class(script):
    img = np.zeros((140, 140), np.uint8)
    img[30:94, 30:94] = 255
    out = script.run(FakeContext(), piece_mask=img)
    assert out["piece_class"] == "unknown" and out["route"] == "uncertain"


def test_unusable_masks(script):
    assert script.run(FakeContext())["route"] == "no_piece"
    assert script.run(FakeContext(), piece_mask=np.zeros((40, 40), np.uint8))["route"] == "no_piece"
    tiny = np.zeros((40, 40), np.uint8)
    tiny[10:20, 10:20] = 1
    assert script.run(FakeContext(), piece_mask=tiny)["route"] == "no_piece"
    long = np.zeros((100, 300), np.uint8)
    long[30:70, 20:280] = 1
    assert script.run(FakeContext(), piece_mask=long)["route"] == "no_piece"
    assert script.run(FakeContext(), pieces=[{"id": "x", "mask": tiny}])["route"] == "no_piece"


def test_orientation_for_a_slot_needs_all_four_sides(script, puzzle):
    want = list(puzzle.side_types(2, 3))  # south, east, north, west
    for angle in (0.0, 37.0, -120.0):
        out = script.run(
            FakeContext(), piece_mask=_mask(puzzle.outline(2, 3), angle), expected_sides=want
        )
        assert out["sides_agree"] == 4
        if out["orientations_tied"] == 1:
            assert out["route"] in ("classified", "uncertain")
            assert abs((out["yaw_deg"] - angle + 180.0) % 360.0 - 180.0) < 4.0
    # One side the piece does not have: no turn agrees on all four.
    wrong = want[:]
    wrong[0] = "flat" if wrong[0] != "flat" else "tab"
    out = script.run(
        FakeContext(), piece_mask=_mask(puzzle.outline(2, 3), 10.0), expected_sides=wrong
    )
    assert out["route"] == "mismatch" and out["sides_agree"] < 4
