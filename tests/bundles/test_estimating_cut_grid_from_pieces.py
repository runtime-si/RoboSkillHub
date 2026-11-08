"""estimating-cut-grid-from-pieces: the cut grid comes back from the count, the board and a few pieces."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext

from .jigsaw_cut import make_puzzle, render_piece

BUNDLE = "estimating-cut-grid-from-pieces"
BOARD = (0.4572, 0.6096)  # the 550-piece target, 18 x 24 in, cut 20 x 27


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["estimate_grid"].module


@pytest.fixture(scope="module")
def target():
    return make_puzzle(20, 27, cell=(BOARD[0] / 20, BOARD[1] / 27), seed=7, style="classic")


def _masks(puzzle, n, seed=0, noise=True):
    rng = np.random.default_rng(seed)
    out = []
    for k in rng.choice(puzzle.cols * puzzle.rows, n, replace=False):
        i, j = divmod(int(k), puzzle.rows)
        rd = render_piece(puzzle, i, j, rng, px_per_cell=64.0, noise=noise)
        out.append((rd, puzzle.piece_class(i, j)))
    return out


def _sides(cols, rows, board, n, seed=0, sigma=0.03, oriented=True):
    """Pre-measured pieces of a cols x rows cut: cell sides with relative noise."""
    rng = np.random.default_rng(seed)
    cw, ch = board[0] / cols, board[1] / rows
    out = []
    for k in range(n):
        s = [cw, ch, cw, ch] * np.exp(rng.normal(0, sigma, 4))
        swap = bool(rng.integers(2))
        if swap:
            s = np.roll(s, 1)
        rec = {"id": f"s{k}", "side_lengths": [float(x) for x in s]}
        if oriented:
            rec["width_pair"] = 1 if swap else 0
        out.append(rec)
    return out


def _run(script, *args, **kw):
    return script.run(FakeContext(), *args, **kw)


def test_target_grid_from_wrist_masks_with_known_turn(script, target):
    pieces = [
        {
            "id": str(k),
            "mask": rd.mask,
            "metres_per_pixel": rd.metres_per_pixel,
            "angle_deg": rd.angle_deg,
        }
        for k, (rd, _) in enumerate(_masks(target, 24, seed=1))
    ]
    out = _run(script, 550, *BOARD, pieces=pieces)
    assert out["route"] == "estimated", out["detail"]
    assert (out["grid_cols"], out["grid_rows"]) == (20, 27)
    assert out["grid_count"] == 540 and out["expected_edge_pieces"] == 90
    assert out["confidence"] >= script.MIN_CONFIDENCE
    assert (out["runner_up_cols"], out["runner_up_rows"]) != (20, 27)
    assert out["cell_width"] == pytest.approx(BOARD[0] / 20)
    assert out["cell_height"] == pytest.approx(BOARD[1] / 27)
    assert out["pieces_used"] >= 22
    assert 0.9 < out["scale_ratio"] < 1.05
    assert all("width_pair" in p for p in out["pieces"] if "side_lengths" in p)
    assert all("mask" not in p for p in out["pieces"])
    assert out["candidates"][0]["probability"] == out["confidence"]


def test_masks_measure_the_cells_corner_to_corner(script, target):
    for rd, cls in _masks(target, 6, seed=2, noise=False):
        m = script.measure_mask(rd.mask, rd.metres_per_pixel)
        s = np.array(m["side_lengths"])
        assert np.allclose(sorted(s), sorted([BOARD[0] / 20, BOARD[1] / 27] * 2), rtol=0.12)
        assert m["piece_class"] == cls


def test_width_pair_follows_the_turn(script):
    wide = make_puzzle(5, 5, cell=(0.026, 0.019), seed=3, style="classic")
    rng = np.random.default_rng(3)
    for angle in (0.0, 30.0, 90.0, 135.0, 200.0, 290.0):
        rd = render_piece(wide, 2, 2, rng, px_per_cell=80.0, angle_deg=angle, noise=False)
        m = script.measure_mask(rd.mask, rd.metres_per_pixel, angle_deg=angle)
        s = m["side_lengths"]
        pairs = (0.5 * (s[0] + s[2]), 0.5 * (s[1] + s[3]))
        assert pairs[m["width_pair"]] == pytest.approx(0.026, rel=0.08), angle
        assert pairs[1 - m["width_pair"]] == pytest.approx(0.019, rel=0.08), angle


@pytest.mark.parametrize(
    ("count", "cols", "rows", "board"),
    [
        (100, 10, 10, (0.25, 0.25)),
        (200, 12, 17, (0.30, 0.41)),
        (300, 15, 20, (0.33, 0.44)),
        (1000, 40, 25, (0.70, 0.50)),
    ],
)
def test_oriented_measured_sides_find_the_grid(script, count, cols, rows, board):
    out = _run(script, count, *board, pieces=_sides(cols, rows, board, 48, sigma=0.03))
    assert (out["grid_cols"], out["grid_rows"]) == (cols, rows), out["detail"]
    assert out["route"] == "estimated", out["detail"]


def test_a_large_near_square_cut_wins_but_stays_open(script):
    # 38 x 27 = 1026 against 37 x 26 = 962: the same cell shape 2.7-3.8 % larger, which the
    # shared scale absorbs, so only the count prior separates them.
    board = (0.70, 0.50)
    out = _run(script, 1000, *board, pieces=_sides(38, 27, board, 48, sigma=0.03))
    assert (out["grid_cols"], out["grid_rows"]) == (38, 27)
    assert out["route"] in ("estimated", "ambiguous")


def test_unoriented_pieces_cannot_tell_a_cell_from_its_turned_twin(script):
    # 10 x 20 on 30 x 41 cm: cells 30 x 20.5 mm; 15 x 14: 20 x 29.3 mm, the same cell turned.
    board = (0.30, 0.41)
    pieces = _sides(10, 20, board, 24, seed=4, sigma=0.03, oriented=False)
    out = _run(script, 200, *board, pieces=pieces)
    assert "width_pair" not in out["pieces"][0]
    assert (out["grid_cols"], out["grid_rows"]) == (10, 20), out["detail"]
    assert (out["runner_up_cols"], out["runner_up_rows"]) == (15, 14)
    assert out["route"] == "ambiguous"
    oriented = _run(script, 200, *board, pieces=_sides(10, 20, board, 24, seed=4, sigma=0.03))
    assert oriented["route"] == "estimated"
    assert (oriented["grid_cols"], oriented["grid_rows"]) == (10, 20)


def test_unoriented_near_square_neighbours_are_not_claimed(script):
    board = (0.30, 0.41)  # 12 x 17 (cells 25 x 24.1) against 12 x 16 (25 x 25.6)
    pieces = _sides(12, 17, board, 48, seed=4, sigma=0.03, oriented=False)
    out = _run(script, 200, *board, pieces=pieces)
    assert out["route"] == "ambiguous", out["detail"]
    assert (12, 17) in [(c["cols"], c["rows"]) for c in out["candidates"][:2]]
    oriented = _run(script, 200, *board, pieces=_sides(12, 17, board, 48, seed=4, sigma=0.03))
    assert (oriented["grid_cols"], oriented["grid_rows"]) == (12, 17)


def test_too_few_pieces_still_return_the_prior_guess(script):
    out = _run(script, 550, *BOARD, pieces=_sides(20, 27, BOARD, 3))
    assert out["route"] == "insufficient_pieces"
    assert out["pieces_used"] == 3
    assert "need 6" in out["detail"]
    assert out["grid_cols"] * out["grid_rows"] == out["grid_count"]
    assert abs(out["grid_count"] - 550) <= 55


def test_no_pieces_at_all(script):
    out = _run(script, 550, *BOARD)
    assert out["route"] == "insufficient_pieces"
    assert out["pieces_used"] == 0
    assert math.isnan(out["residual"])


def test_noisy_near_square_pieces_are_ambiguous(script):
    pieces = _sides(20, 27, BOARD, 8, seed=5, sigma=0.06, oriented=False)
    out = _run(script, 550, *BOARD, pieces=pieces)
    assert out["route"] == "ambiguous", out["detail"]
    assert out["confidence"] < script.MIN_CONFIDENCE
    assert "runner-up" in out["detail"]
    assert len(out["candidates"]) == 5
    probs = [c["probability"] for c in out["candidates"]]
    assert probs == sorted(probs, reverse=True)


def test_edge_count_agrees_or_disagrees(script):
    pieces = _sides(20, 27, BOARD, 24, seed=6)
    ok = _run(script, 550, *BOARD, pieces=pieces, edge_piece_count=89)
    assert ok["route"] == "estimated" and ok["edge_check"] == "agrees"
    bad = _run(script, 550, *BOARD, pieces=pieces, edge_piece_count=110)
    assert bad["edge_check"] == "disagrees"
    assert bad["route"] == "ambiguous"
    assert "edge count 110" in bad["detail"]
    assert _run(script, 550, *BOARD, pieces=pieces)["edge_check"] == "not_given"


def test_edge_count_alone_moves_the_prior(script):
    prior = _run(script, 550, *BOARD, top_k=20)
    other = next(
        c
        for c in prior["candidates"]
        if abs(c["expected_edge_pieces"] - prior["expected_edge_pieces"]) >= 4
    )
    out = _run(script, 550, *BOARD, edge_piece_count=other["expected_edge_pieces"])
    moved = abs(out["expected_edge_pieces"] - other["expected_edge_pieces"])
    assert moved < abs(prior["expected_edge_pieces"] - other["expected_edge_pieces"])
    assert out["edge_check"] == "agrees"


def test_aspect_only_from_ratio_or_reference(script):
    pieces = _sides(20, 27, BOARD, 24, seed=7)
    scaled = [dict(p, side_lengths=[s * 1.3 for s in p["side_lengths"]]) for p in pieces]
    out = _run(script, 550, board_aspect=BOARD[0] / BOARD[1], pieces=scaled)
    assert (out["grid_cols"], out["grid_rows"]) == (20, 27), out["detail"]
    assert out["cell_width"] == pytest.approx(1.3 * BOARD[0] / 20, rel=0.03)
    picture = np.zeros((1066, 800, 3), np.uint8)
    out = _run(script, 550, reference=picture, pieces=scaled)
    assert (out["grid_cols"], out["grid_rows"]) == (20, 27)


def test_landscape_board_swaps_cols_and_rows(script):
    board = (BOARD[1], BOARD[0])
    pieces = [dict(p, width_pair=1 - p["width_pair"]) for p in _sides(20, 27, BOARD, 24, seed=8)]
    out = _run(script, 550, *board, pieces=pieces)
    assert (out["grid_cols"], out["grid_rows"]) == (27, 20)


def test_reference_aspect_mismatch_is_noted(script):
    picture = np.zeros((800, 800, 3), np.uint8)
    out = _run(script, 550, *BOARD, reference=picture, pieces=_sides(20, 27, BOARD, 24))
    assert (out["grid_cols"], out["grid_rows"]) == (20, 27)
    assert "differs from the board" in out["detail"]


def test_no_board_shape_or_bad_count(script):
    out = _run(script, 550, pieces=_sides(20, 27, BOARD, 24))
    assert out["route"] == "insufficient_pieces"
    assert "board_width" in out["detail"]
    assert _run(script, 2, *BOARD)["route"] == "insufficient_pieces"


def test_bad_piece_entries_are_reported_not_used(script):
    pieces = _sides(20, 27, BOARD, 8) + [
        {"id": "short", "side_lengths": [0.02, 0.02]},
        {"id": "neg", "side_lengths": [0.02, -0.02, 0.02, 0.02]},
        {"id": "nothing"},
        {"id": "noscale", "mask": np.ones((40, 40), np.uint8)},
        {"id": "blob", "mask": np.zeros((40, 40), np.uint8), "metres_per_pixel": 0.0004},
    ]
    out = _run(script, 550, *BOARD, pieces=pieces)
    assert out["pieces_used"] == 8
    errors = {p["id"]: p["error"] for p in out["pieces"] if "error" in p}
    assert set(errors) == {"short", "neg", "nothing", "noscale", "blob"}
    assert "scale" in errors["noscale"]


def test_fed_back_pieces_take_a_turn_later(script, target):
    pieces = [
        {"id": str(k), "mask": rd.mask, "metres_per_pixel": rd.metres_per_pixel}
        for k, (rd, _) in enumerate(_masks(target, 12, seed=9))
    ]
    first = _run(script, 550, *BOARD, pieces=pieces)
    angles = {str(k): rd.angle_deg for k, (rd, _) in enumerate(_masks(target, 12, seed=9))}
    again = [dict(p, angle_deg=angles[p["id"]]) for p in first["pieces"] if "side_lengths" in p]
    second = _run(script, 550, *BOARD, pieces=again)
    assert all("width_pair" in p for p in second["pieces"])
    assert second["pieces_used"] == first["pieces_used"]


def _wrist_frame(target, n, depth_m=0.15, fx=430.0, depth_scale=1.0):
    """A 640 x 480 wrist frame with ``n`` pieces laid out in a row, at a depth whose scale
    matches the masks (metres per pixel = depth / fx), and one piece cut by the border."""
    rng = np.random.default_rng(11)
    mpp = depth_m / fx
    size = (480, 640)
    masks = []
    x = 10
    for k in range(n + 1):
        i, j = 3 + k, 4
        rd = render_piece(target, i, j, rng, px_per_cell=(BOARD[0] / 20) / mpp, noise=False)
        m = np.zeros(size, np.uint8)
        h, w = rd.mask.shape
        if k == n:  # clipped by the right border
            m[300 : 300 + h, 640 - (w - w // 2) :] = rd.mask[:, : w - w // 2]
        else:
            y = 40 if k % 2 == 0 else 200
            m[y : y + h, x : x + w] = rd.mask
            x += w // 2 + 40 if k % 2 == 0 else w + 10
        masks.append(m)
    cam = {
        "name": "left_wrist",
        "rgb": np.zeros((*size, 3), np.uint8),
        "depth": np.full(size, depth_m * depth_scale, np.float32),
        "intrinsics": np.array([[fx, 0, 320], [0, fx, 240], [0, 0, 1.0]]),
    }
    return cam, masks


def test_measure_view_segments_the_wrist_frame(script, target):
    cam, masks = _wrist_frame(target, 4)
    seg = {"masks": masks, "scores": [0.9] * len(masks)}
    ctx = FakeContext(
        {
            "robot.get_observation": {"cameras": [{"name": "top", "rgb": None}, cam]},
            "sam3.segment_text": seg,
        }
    )
    out = script.run(ctx, 550, *BOARD, measure_view=True)
    assert ctx.call_count("sam3.segment_text") == 1
    assert ctx.calls_to("sam3.segment_text")[0].kwargs["query"] == script.PIECE_QUERY
    assert out["pieces_used"] == 4
    clipped = [p for p in out["pieces"] if "error" in p]
    assert len(clipped) == 1 and "clipped" in clipped[0]["error"]
    for p in out["pieces"]:
        if "side_lengths" in p:
            assert p["metres_per_pixel"] == pytest.approx(0.15 / 430.0)
            assert np.median(p["side_lengths"]) == pytest.approx(0.0227, rel=0.08)
    assert out["route"] == "insufficient_pieces"


def test_camera_masks_with_depth_in_millimetres(script, target):
    cam, masks = _wrist_frame(target, 4, depth_scale=1000.0)
    out = _run(script, 550, *BOARD, camera=cam, piece_masks=masks[:4])
    assert out["pieces_used"] == 4
    assert out["pieces"][0]["metres_per_pixel"] == pytest.approx(0.15 / 430.0)


def test_measure_view_without_a_wrist_camera(script):
    ctx = FakeContext({"robot.get_observation": {"cameras": [{"name": "top", "rgb": None}]}})
    out = script.run(ctx, 550, *BOARD, measure_view=True)
    assert out["route"] == "insufficient_pieces"
    assert "no camera" in out["detail"]


def test_candidates_stay_near_the_count(script):
    grids = script.candidate_grids(550)
    assert (20, 27) in grids and (22, 25) in grids
    assert all(abs(c * r - 550) <= 55 for c, r in grids)
    out = _run(script, 550, *BOARD, pieces=_sides(20, 27, BOARD, 12))
    for c in out["candidates"]:
        ratio = (BOARD[0] / c["cols"]) / (BOARD[1] / c["rows"])
        assert 1 / script.MAX_CELL_ASPECT <= ratio <= script.MAX_CELL_ASPECT
