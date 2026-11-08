"""matching-a-piece-to-placed-neighbours: pieces of procedurally cut jigsaws are fitted back into their slots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

from . import jigsaw_cut as jc

BUNDLE = "matching-a-piece-to-placed-neighbours"
PX = 64


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["match_piece"].module


@pytest.fixture(scope="module")
def classic():
    return jc.make_puzzle(6, 6, seed=7)


@pytest.fixture(scope="module")
def wavy():
    return jc.make_puzzle(6, 6, seed=8, style="wavy")


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _corner_hits(script, puzzle, noise, seed):
    """Pieces whose 4 corners were found within 0.1 cell, whose side types and class were right."""
    rng = np.random.default_rng(seed)
    corners = types = classes = 0
    for i in range(puzzle.cols):
        for j in range(puzzle.rows):
            r = jc.render_piece(puzzle, i, j, rng, noise=noise)
            info = script.analyse_piece(r.mask, r.metres_per_pixel, cell_size=puzzle.cell)
            d = np.linalg.norm(np.array(info["corners_px"])[:, None] - r.corners_px[None], axis=2)
            ok = d.min(axis=0).max() < 0.1 * PX and len(set(d.argmin(axis=1))) == 4
            k0 = int(np.argmin(d[0]))
            got = [info["side_types"][(m - k0) % 4] for m in range(4)]
            corners += ok
            types += ok and got == puzzle.side_types(i, j)
            classes += info["piece_class"] == puzzle.piece_class(i, j)
    return corners, types, classes


def _slot(puzzle, slot, directions, rng=None):
    sides = {}
    for d in directions:
        q = puzzle.facing_side(*slot, d)
        if rng is not None:
            q = jc.jitter_polyline(q, puzzle.cell[0] / PX, rng)
        sides[d] = q.tolist()
    return sides


def _candidates(puzzle, keys, rng, noise=True):
    out, angles = [], {}
    for i, j in keys:
        r = jc.render_piece(puzzle, i, j, rng, noise=noise)
        out.append({"id": f"{i},{j}", "mask": r.mask, "metres_per_pixel": r.metres_per_pixel})
        angles[f"{i},{j}"] = r.angle_deg
    return out, angles


# --- the cut generator --------------------------------------------------------------


@pytest.mark.parametrize("style", ["classic", "wavy"])
def test_mating_sides_are_exact_complements(style):
    pz = jc.make_puzzle(4, 3, seed=1, style=style)
    for i in range(3):
        for j in range(3):
            a, b = pz.side(i, j, "east"), pz.side(i + 1, j, "west")
            assert np.allclose(a, b[::-1])
            ta, tb = pz.side_type(i, j, "east"), pz.side_type(i + 1, j, "west")
            assert {ta, tb} == {"tab", "blank"}
    assert pz.side_type(0, 0, "south") == pz.side_type(0, 0, "west") == "flat"
    assert (
        pz.piece_class(0, 0) == "corner"
        and pz.piece_class(1, 0) == "edge"
        and pz.piece_class(1, 1) == "interior"
    )


def test_wavy_sides_bend_and_no_piece_touches_itself():
    pz = jc.make_puzzle(6, 6, seed=2, style="wavy")
    assert all(pz.is_simple(i, j) for i in range(6) for j in range(6))
    # the shoulders of a wavy side leave the chord; a classic side's stay within 3 %
    devs = {}
    for style in ("classic", "wavy"):
        p = jc.make_puzzle(6, 6, seed=2, style=style)
        d = []
        for i in range(6):
            for j in range(1, 6):
                poly, _ = p.h_edges[i, j]
                a, b = p.vertices[i, j], p.vertices[i + 1, j]
                u = (poly - a) @ (b - a) / np.dot(b - a, b - a)
                e, rel = b - a, poly - a
                w = np.abs(e[0] * rel[:, 1] - e[1] * rel[:, 0]) / np.dot(e, e)
                d.append(w[(u > 0.05) & (u < 0.2)].max())
        devs[style] = np.median(d)
    assert devs["classic"] < 0.03 < devs["wavy"]


# --- piece analysis -----------------------------------------------------------------


@pytest.mark.parametrize("style", ["classic", "wavy"])
def test_clean_pieces_corners_sides_and_class(script, classic, wavy, style):
    puzzle = classic if style == "classic" else wavy
    corners, types, classes = _corner_hits(script, puzzle, noise=False, seed=0)
    assert corners >= 35 and types >= 35 and classes == 36


@pytest.mark.parametrize("style", ["classic", "wavy"])
def test_noisy_pieces_corners_sides_and_class(script, classic, wavy, style):
    puzzle = classic if style == "classic" else wavy
    corners, types, classes = _corner_hits(script, puzzle, noise=True, seed=1)
    assert corners >= 32 and types >= 32 and classes >= 34


def test_grip_pixel_is_inside_the_body(script, classic):
    r = jc.render_piece(classic, 2, 2, np.random.default_rng(3), noise=False, angle_deg=0.0)
    u, v = script.grip_pixel(r.mask)
    c = r.corners_px.mean(axis=0)
    assert r.mask[int(v), int(u)] > 0
    assert np.hypot(u - c[0], v - c[1]) < 0.2 * PX


def test_empty_mask_is_refused(script):
    with pytest.raises(ValueError):
        script.analyse_piece(np.zeros((40, 40), np.uint8))


# --- the slot match -----------------------------------------------------------------


@pytest.mark.parametrize(
    "style, directions", [("classic", ["south", "west", "north"]), ("wavy", ["south", "west"])]
)
def test_placed_neighbours_pick_the_right_piece_and_turn(script, classic, wavy, style, directions):
    puzzle = classic if style == "classic" else wavy
    rng = np.random.default_rng(11)
    slot = (2, 3)
    others = [p for p in puzzle.interior() if p != slot]
    keys = [slot] + [others[k] for k in rng.choice(len(others), 9, replace=False)]
    cands, angles = _candidates(puzzle, keys, rng)
    out = script.run(
        FakeContext(),
        cands,
        neighbour_sides=_slot(puzzle, slot, directions, rng),
        cell_size=list(puzzle.cell),
    )
    assert out["route"] == "fits", out["detail"]
    assert out["best_id"] == "2,3"
    assert abs(_wrap(out["best_rotation_deg"] + angles["2,3"])) < 10.0
    assert out["piece_class"] == "interior"
    assert (
        np.linalg.norm(np.array(out["grip_point_board"]) - puzzle.centre(*slot))
        < 0.25 * puzzle.cell[0]
    )
    assert out["margin"] >= script.MIN_MARGIN
    assert len(out["ranked"]) == 5 and len(out["pieces"]) == 10


def test_three_neighbours_without_cell_size(script, wavy):
    rng = np.random.default_rng(12)
    slot = (3, 2)
    others = [p for p in wavy.interior() if p != slot]
    keys = [others[k] for k in rng.choice(len(others), 7, replace=False)] + [slot]
    cands, angles = _candidates(wavy, keys, rng)
    out = script.run(
        FakeContext(), cands, neighbour_sides=_slot(wavy, slot, ["north", "east", "south"], rng)
    )
    assert out["route"] == "fits" and out["best_id"] == "3,2"
    assert abs(_wrap(out["best_rotation_deg"] + angles["3,2"])) < 10.0


def test_image_yaw_shifts_the_turn(script, classic):
    rng = np.random.default_rng(13)
    slot = (2, 2)
    cands, angles = _candidates(classic, [slot], rng, noise=False)
    nb = _slot(classic, slot, ["north", "east"])
    a = script.run(FakeContext(), cands, neighbour_sides=nb)
    cands[0]["image_yaw_deg"] = 30.0
    b = script.run(FakeContext(), cands, neighbour_sides=nb)
    assert abs(_wrap(b["best_rotation_deg"] - a["best_rotation_deg"] + 30.0)) < 1e-6
    assert abs(_wrap(a["best_rotation_deg"] + angles["2,2"])) < 5.0


def test_the_same_piece_twice_is_ambiguous(script, classic):
    rng = np.random.default_rng(14)
    slot = (3, 3)
    cands, _ = _candidates(classic, [slot], rng)
    twin = dict(cands[0], id="twin")
    out = script.run(
        FakeContext(), [cands[0], twin], neighbour_sides=_slot(classic, slot, ["south", "east"])
    )
    assert out["route"] == "ambiguous"
    assert {e["id"] for e in out["ranked"]} == {"3,3", "twin"}


def test_wrong_pieces_only_is_no_fit(script, classic):
    rng = np.random.default_rng(15)
    slot = (2, 2)
    want = [classic.side_type(*slot, d) for d in ("north", "east", "south")]
    # interior pieces whose three sides never line up with the slot's in any rotation
    wrong = [
        p
        for p in classic.interior()
        if p != slot
        and not any(
            [classic.side_types(*p)[(s + k) % 4] for k in (2, 1, 0)] == want  # north, east, south
            for s in range(4)
        )
    ]
    assert wrong
    cands, _ = _candidates(classic, wrong[:6], rng, noise=False)
    out = script.run(
        FakeContext(), cands, neighbour_sides=_slot(classic, slot, ["north", "east", "south"])
    )
    assert out["route"] == "no_fit"
    assert all(not e["compatible"] for e in out["ranked"])


def test_border_slot_without_neighbours_wants_a_corner_piece(script, classic):
    rng = np.random.default_rng(16)
    keys = [(0, 0), (2, 2), (3, 2), (2, 3)]
    cands, _ = _candidates(classic, keys, rng, noise=False)
    out = script.run(
        FakeContext(), cands, border_sides=["south", "west"], cell_size=list(classic.cell)
    )
    assert out["route"] == "fits", out["detail"]
    assert out["best_id"] == "0,0" and out["piece_class"] == "corner"
    assert "grip_point_board" not in out


def test_no_neighbours_and_several_compatible_is_ambiguous(script, classic):
    rng = np.random.default_rng(17)
    cands, _ = _candidates(classic, [(2, 2), (3, 3)], rng, noise=False)
    out = script.run(FakeContext(), cands)
    assert out["route"] == "ambiguous"


def test_unusable_inputs_route_no_fit(script, classic):
    out = script.run(
        FakeContext(), [{"id": "a", "mask": np.zeros((50, 50), np.uint8)}, {"id": "b"}]
    )
    assert (
        out["route"] == "no_fit"
        and len(out["pieces"]) == 2
        and all("error" in p for p in out["pieces"])
    )
    out = script.run(FakeContext(), [], neighbour_sides={"up": [[0, 0], [1, 0]]})
    assert out["route"] == "no_fit" and "unknown slot side" in out["detail"]
    out = script.run(FakeContext(), [], neighbour_sides={"north": [[0, 0]]})
    assert out["route"] == "no_fit"
    assert script.run(FakeContext(), [])["route"] == "no_fit"


def test_example_declares_every_route():
    path = Path(__file__).parents[2] / "skills" / BUNDLE / "examples" / "canonical_subgraph.json"
    example = json.loads(path.read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == {"fits", "ambiguous", "no_fit"}
