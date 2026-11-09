"""selecting-a-grip-clear-of-placed-neighbours: grips on generated pieces at a slot among placed neighbours."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from gap.testing import FakeContext

from .jigsaw_cut import make_puzzle

BUNDLE = "selecting-a-grip-clear-of-placed-neighbours"
CELL = 0.0225
NB = {"west": (0, 1), "east": (2, 1), "south": (1, 0), "north": (1, 2)}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["select_grip"].module


@pytest.fixture(scope="module")
def puzzle():
    return make_puzzle(3, 3, cell=(CELL, CELL), seed=1)


def _square(i, j, inset=0.0):
    x0, y0 = i * CELL + inset, j * CELL + inset
    x1, y1 = (i + 1) * CELL - inset, (j + 1) * CELL - inset
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def _footprint_clearance(out, neighbours, fw=0.008, ft=0.004, travel=0.001):
    """Smallest signed distance (m, positive outside) from the returned fingertips to any neighbour."""
    u = np.array(
        [
            math.cos(math.radians(out["closing_yaw_deg"])),
            math.sin(math.radians(out["closing_yaw_deg"])),
        ]
    )
    v = np.array([-u[1], u[0]])
    a, b = (np.asarray(c) for c in out["contacts"])
    best = math.inf
    for su in np.linspace(0.0, ft + travel, 11):
        for sv in np.linspace(-fw / 2, fw / 2, 17):
            for p in (a - su * u + sv * v, b + su * u + sv * v):
                for poly in neighbours:
                    d = -cv2.pointPolygonTest(
                        poly.astype(np.float32).reshape(-1, 1, 2), (float(p[0]), float(p[1])), True
                    )
                    best = min(best, d)
    return best


def test_a_lone_piece_is_gripped_through_its_centre(script, puzzle):
    out = script.run(FakeContext(), target_outline=puzzle.outline(1, 1))
    assert out["route"] == "grip_found", out["detail"]
    assert math.isinf(out["clearance_m"])
    centre = puzzle.centre(1, 1)
    assert np.linalg.norm(np.asarray(out["grip_point"]) - centre) < 0.25 * CELL
    assert 0.006 <= out["grip_width_m"] <= 0.094 - 0.006


@pytest.mark.parametrize("sides", [("west",), ("west", "south"), ("south", "east")])
def test_the_fingertips_stay_clear_of_one_or_two_neighbours(script, puzzle, sides):
    neighbours = [puzzle.outline(*NB[d]) for d in sides]
    out = script.run(
        FakeContext(), target_outline=puzzle.outline(1, 1), neighbour_outlines=neighbours
    )
    assert out["route"] == "grip_found", out["detail"]
    assert out["clearance_m"] >= 0.0005
    # Checked independently against the true outlines (1 mm release travel included).
    assert _footprint_clearance(out, neighbours) >= 0.0004
    assert out["n_clear"] < out["n_tried"]


def test_a_slot_closed_on_three_sides_has_no_clear_grip(script, puzzle):
    neighbours = [puzzle.outline(*NB[d]) for d in ("west", "south", "east")]
    out = script.run(
        FakeContext(), target_outline=puzzle.outline(1, 1), neighbour_outlines=neighbours
    )
    assert out["route"] == "no_clear_grip"
    assert out["n_clear"] == 0
    # The least-bad grip is still returned.
    assert "closing_yaw_deg" in out and out["clearance_m"] < 0.0005


def test_square_pieces_in_a_corner_are_set_down_offset(script):
    # Square pieces, west and south placed: closing along x or y puts a fingertip on a
    # neighbour, a 45 degree grip across a corner region touches inclined sides only (no
    # hold), and corner-to-corner grips put a fingertip's side over a neighbour.
    target = _square(1, 1)
    neighbours = [_square(0, 1), _square(1, 0)]
    out = script.run(FakeContext(), target_outline=target, neighbour_outlines=neighbours)
    assert out["route"] == "only_offset_place", out["detail"]
    off = np.asarray(out["offset_m"])
    assert off[0] >= -1e-9 and off[1] >= -1e-9 and np.linalg.norm(off) > 0.001
    assert np.linalg.norm(off) <= 0.6 * CELL * 1.25 + 1e-9
    # At the offset pose the grip clears the neighbours.
    assert _footprint_clearance(out, neighbours) >= 0.0004
    assert (
        script.run(
            FakeContext(), target_outline=target, neighbour_outlines=neighbours, allow_offset=False
        )["route"]
        == "no_clear_grip"
    )


def test_a_grip_on_inclined_sides_is_refused(script):
    # A lone square: a 45 degree closing direction only holds corner to corner.
    out = script.run(FakeContext(), target_outline=_square(0, 0), top_k=1000)
    diag = [c for c in out["candidates"] if c["closing_yaw_deg"] in (45.0, 135.0)]
    assert diag and all(c["grip_width_m"] > 0.9 * math.sqrt(2) * CELL for c in diag)


def test_occupied_cells_stand_in_for_outlines(script, puzzle):
    out = script.run(
        FakeContext(),
        target_outline=puzzle.outline(1, 1),
        occupied_cells=[[0, 1]],
        cell_size=[CELL, CELL],
        board_origin=[0.0, 0.0],
    )
    assert out["route"] == "grip_found", out["detail"]
    assert _footprint_clearance(out, [puzzle.outline(0, 1)]) >= 0.0005
    exact = script.run(
        FakeContext(),
        target_outline=puzzle.outline(1, 1),
        neighbour_outlines=[puzzle.outline(0, 1)],
    )
    assert out["n_clear"] <= exact["n_clear"]  # the cell box is conservative
    bad = script.run(FakeContext(), target_outline=puzzle.outline(1, 1), occupied_cells=[[0, 1]])
    assert bad["route"] == "no_clear_grip" and "cell_size" in bad["detail"]


def _mask_of(poly_m, mpp, angle_deg):
    """Top-down image mask of a board-metre polygon turned by ``angle_deg`` (as it lies)."""
    c = poly_m.mean(axis=0)
    a = math.radians(angle_deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    p = (poly_m - c) @ r.T
    px = np.column_stack([p[:, 0], -p[:, 1]]) / mpp + 80
    mask = np.zeros((160, 160), np.uint8)
    cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 1)
    return mask


def test_a_mask_turned_into_its_slot_gives_the_pick_grip(script, puzzle):
    mpp = 0.0003
    lying = 40.0  # the piece lies turned 40 degrees from how it fits
    mask = _mask_of(puzzle.outline(1, 1), mpp, lying)
    slot = puzzle.centre(1, 1)
    neighbours = [puzzle.outline(*NB[d]) for d in ("west", "south")]
    out = script.run(
        FakeContext(),
        piece_mask=mask,
        metres_per_pixel=mpp,
        rotation_deg=-lying,
        slot_center=slot.tolist(),
        neighbour_outlines=neighbours,
    )
    assert out["route"] == "grip_found", out["detail"]
    u, v = out["pick_grip_pixel"]
    assert mask[int(round(v)), int(round(u))] == 1
    assert out["pick_closing_yaw_deg"] == pytest.approx((out["closing_yaw_deg"] + lying) % 180.0)


def test_the_body_centre_ignores_the_knobs(script, puzzle):
    mpp = 0.0002
    poly = puzzle.outline(1, 1)
    mask = _mask_of(poly, mpp, 0.0)
    got = script.body_centre_px(mask)
    # The mask was drawn about the outline's mean; the body centre is the cell's.
    true = (puzzle.centre(1, 1) - poly.mean(axis=0)) * [1, -1] / mpp + 80
    assert np.linalg.norm(got - true) < 0.05 * CELL / mpp


def test_gripper_limits_and_bad_inputs(script, puzzle):
    narrow = script.run(FakeContext(), target_outline=puzzle.outline(1, 1), max_opening=0.01)
    assert narrow["route"] == "no_clear_grip"
    assert script.run(FakeContext())["route"] == "no_clear_grip"
    assert script.run(FakeContext(), target_outline=[[0, 0], [1, 1]])["route"] == "no_clear_grip"
    empty = np.zeros((20, 20), np.uint8)
    out = script.run(FakeContext(), piece_mask=empty, metres_per_pixel=0.001, slot_center=[0, 0])
    assert out["route"] == "no_clear_grip"
