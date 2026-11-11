"""reading-domino-tiles-in-clutter on rendered top-down tiles (tests/bundles/_domino_render.py, coloured pips)."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext

from . import _domino_render as D


@pytest.fixture(scope="module")
def rt(skills_registry):
    return (
        skills_registry.get("reading-domino-tiles-in-clutter")
        .canonical_scripts["read_tiles"]
        .module
    )


def _frame(tiles, **kw):
    frame, _, truth = D.render(tiles, style="colour", width_px=96.0, hw=(720, 960), **kw)
    return frame, truth


TILES = [
    {"value": (6, 3), "center_px": (250, 200), "angle_deg": 0.0},
    {"value": (5, 1), "center_px": (600, 220), "angle_deg": 35.0},
    {"value": (4, 4), "center_px": (300, 520), "angle_deg": 90.0},
    {"value": (2, 0), "center_px": (680, 520), "angle_deg": -20.0},
]


def test_reads_each_tile(rt):
    frame, truth = _frame(TILES)
    out = rt.run(FakeContext({}), frame=frame)
    assert out["n_tiles"] == 4
    got = sorted(tuple(t["value"]) for t in out["tiles"])
    assert got == sorted(tuple(sorted(v["value"])) for v in TILES)
    assert out["total_pips"] == 25


def test_touching_tiles_are_separated_by_their_bars(rt):
    tiles = [
        {"value": (3, 5), "center_px": (300, 300), "angle_deg": 0.0},
        {"value": (6, 2), "center_px": (300 + 192, 300), "angle_deg": 0.0},
        {"value": (1, 4), "center_px": (300, 300 + 96), "angle_deg": 0.0},
    ]
    frame, _ = _frame(tiles)
    out = rt.run(FakeContext({}), frame=frame)
    assert sorted(tuple(t["value"]) for t in out["tiles"]) == [(1, 4), (2, 6), (3, 5)]


def test_frame_from_the_observation_and_no_tiles(rt):
    frame, _ = _frame(TILES[:1])
    out = rt.run(FakeContext({"robot.get_observation": {"cameras": [frame]}}))
    assert (
        out["route"] == "tiles_read" and out["n_tiles"] == 1 and out["camera_name"] == "left_wrist"
    )
    blank = dict(frame, rgb=np.full_like(frame["rgb"], 90))
    assert rt.run(FakeContext({}), frame=blank)["route"] == "no_tiles"
    assert rt.run(FakeContext({"robot.get_observation": {"cameras": []}}))["route"] == "no_tiles"


def test_a_mask_restricts_the_read(rt):
    frame, _ = _frame(TILES)
    mask = np.zeros(frame["rgb"].shape[:2], np.uint8)
    mask[:360, :480] = 1
    out = rt.run(FakeContext({}), frame=frame, mask=mask)
    assert [tuple(t["value"]) for t in out["tiles"]] == [(3, 6)]
