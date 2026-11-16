"""finding-boxes-of-known-size-from-overhead-depth: the outline fit on synthetic rims and frames."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext

from ._bin_scenes import Scene, obb

BUNDLE = "finding-boxes-of-known-size-from-overhead-depth"
SIZE = (0.30, 0.20)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["find_known_box"].module


def _outline(cx, cy, size, yaw_deg, step=0.002, keep=lambda lx, ly: True):
    """Points along a rectangle's outline (in the world), optionally only some of them."""
    pts = []
    for lx in np.arange(-size[0] / 2, size[0] / 2 + 1e-9, step):
        pts += [(lx, -size[1] / 2), (lx, size[1] / 2)]
    for ly in np.arange(-size[1] / 2, size[1] / 2 + 1e-9, step):
        pts += [(-size[0] / 2, ly), (size[0] / 2, ly)]
    pts = np.array([p for p in pts if keep(*p)])
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.column_stack([cx + c * pts[:, 0] - s * pts[:, 1], cy + s * pts[:, 0] + c * pts[:, 1]])


def _yaw_off(a, b):
    return abs((a - b + 90.0) % 180.0 - 90.0)


def test_a_whole_outline_is_placed_within_a_cell_and_a_step(script):
    fit = script.fit_known_rectangle(_outline(0.12, -0.05, SIZE, 34.0), SIZE)
    assert np.linalg.norm(fit["center"] - [0.12, -0.05]) < 0.006
    assert _yaw_off(fit["yaw_deg"], 34.0) <= 2.0
    assert fit["coverage"] > 0.9


def test_a_rim_seen_in_part_is_placed_by_its_size_not_by_what_is_seen(script):
    # one long wall and half of each short wall missing: a minimum-area rectangle round the
    # rest would sit a quarter of the short side off
    seen = _outline(0.0, 0.0, SIZE, 20.0, keep=lambda lx, ly: ly > -0.01)
    fit = script.fit_known_rectangle(seen, SIZE)
    assert np.linalg.norm(fit["center"]) < 0.008
    assert _yaw_off(fit["yaw_deg"], 20.0) <= 2.0
    assert 0.35 < fit["coverage"] < 0.75


def test_too_few_points_or_too_little_outline_is_none(script):
    assert script.fit_known_rectangle(np.zeros((20, 2)), SIZE) is None
    one_wall = _outline(0.0, 0.0, SIZE, 0.0, keep=lambda lx, ly: ly > 0.09 and lx > 0)
    assert len(one_wall) >= 60
    dense = np.repeat(one_wall, 3, axis=0)
    assert script.fit_known_rectangle(dense, SIZE) is None


def test_a_restricted_turn_range_is_honoured(script):
    pts = _outline(0.0, 0.0, SIZE, 30.0)
    fit = script.fit_known_rectangle(pts, SIZE, yaws_deg=np.arange(20.0, 41.0, 2.0))
    assert _yaw_off(fit["yaw_deg"], 30.0) <= 2.0
    far = script.fit_known_rectangle(pts, SIZE, yaws_deg=[-60.0, -58.0], min_coverage=0.6)
    assert far is None


def test_found_from_an_overhead_frame_with_the_stated_size(script):
    frame = Scene().open_box(0.10, -0.06, 0.30, 0.20, 25.0, 0.15).frame()
    out = script.run(FakeContext({}), 0.30, 0.20, 0.15, camera_frame=frame)
    assert out["route"] == "found", out["detail"]
    box = out["container_obb"]
    assert abs(box["center"]["x"] - 0.10) < 0.008 and abs(box["center"]["y"] + 0.06) < 0.008
    assert _yaw_off(out["yaw_deg"], 25.0) <= 2.0
    assert box["extent"] == {"x": 0.15, "y": 0.10, "z": pytest.approx(0.075)}
    assert box["center"]["z"] == pytest.approx(0.075)
    assert not out["yaw_ambiguous"]


def test_the_camera_is_read_from_the_observation_and_a_wrist_camera_is_not_chosen(script):
    scene = Scene().open_box(0.0, 0.0, 0.30, 0.20, 0.0, 0.15)
    obs = {"cameras": [dict(scene.frame(), name="left_wrist", depth=None), scene.frame()]}
    obs["cameras"].insert(0, dict(Scene().frame(), name="right_wrist"))
    ctx = FakeContext({"robot.get_observation": obs})
    out = script.run(ctx, 0.30, 0.20, 0.15)
    assert out["route"] == "found" and out["camera"] == "top"


def test_two_touching_boxes_are_each_placed(script):
    # the tall box and a low one against its long wall
    scene = (
        Scene().open_box(0.0, 0.0, 0.30, 0.20, 0.0, 0.15).open_box(0.0, 0.19, 0.24, 0.18, 0.0, 0.07)
    )
    frame = scene.frame()
    tall = script.run(FakeContext({}), 0.30, 0.20, 0.15, camera_frame=frame)
    assert tall["route"] == "found"
    low = script.run(
        FakeContext({}),
        0.24,
        0.18,
        0.07,
        camera_frame=frame,
        exclude_obbs=[tall["container_obb"]],
        band_below=0.02,
        band_above=0.03,
    )
    assert low["route"] == "found", low["detail"]
    assert abs(low["center"]["x"]) < 0.008 and abs(low["center"]["y"] - 0.19) < 0.008


def test_a_bar_as_tall_as_the_rim_beside_the_box_does_not_move_it(script):
    # an arm lying along the box's long wall, at the rim's height
    scene = (
        Scene().open_box(0.0, 0.0, 0.30, 0.20, 0.0, 0.15).block(0.05, -0.16, 0.34, 0.06, 0.0, 0.15)
    )
    out = script.run(FakeContext({}), 0.30, 0.20, 0.15, camera_frame=scene.frame())
    assert out["route"] == "found"
    assert abs(out["center"]["x"]) < 0.01 and abs(out["center"]["y"]) < 0.01


def test_not_found_says_why(script):
    frame = Scene().open_box(0.0, 0.0, 0.30, 0.20, 0.0, 0.15).frame()
    none = script.run(FakeContext({}), 0.30, 0.20, 0.40, camera_frame=frame)
    assert none["route"] == "not_found" and "depth points" in none["detail"]
    assert none["container_obb"] is None
    outside = script.run(
        FakeContext({}),
        0.30,
        0.20,
        0.15,
        camera_frame=frame,
        roi={"x_min": 0.3, "x_max": 0.6, "y_min": -0.2, "y_max": 0.2},
    )
    assert outside["route"] == "not_found"
    wrong = script.run(FakeContext({}), 0.60, 0.45, 0.15, camera_frame=frame, min_coverage=0.6)
    assert wrong["route"] == "not_found" and "outline" in wrong["detail"]
    blind = script.run(FakeContext({"robot.get_observation": {"cameras": []}}), 0.3, 0.2, 0.15)
    assert blind["route"] == "not_found" and "camera" in blind["detail"]


def test_a_nearly_square_box_is_flagged(script):
    frame = Scene().open_box(0.0, 0.0, 0.24, 0.20, 10.0, 0.07).frame()
    out = script.run(FakeContext({}), 0.24, 0.20, 0.07, camera_frame=frame)
    assert out["route"] == "found" and out["yaw_ambiguous"]
    assert "quarter" in out["detail"]


def test_outside_and_the_returned_box(script):
    box = obb(0.1, 0.0, 0.30, 0.20, 90.0, 0.15)
    xy = np.array([[0.1, 0.0], [0.1, 0.16], [0.1, 0.19], [0.23, 0.0]])
    assert script.outside(xy, box, 0.02).tolist() == [False, False, True, True]
    made = script.upright_obb(np.array([0.1, 0.0]), (0.30, 0.20), 90.0, 0.15, 0.01)
    assert made["extent"]["z"] == pytest.approx(0.07) and made["center"]["z"] == pytest.approx(0.08)
    centre, half, yaw = script.footprint(made)
    assert yaw == pytest.approx(math.pi / 2) and half.tolist() == [0.15, 0.10]
