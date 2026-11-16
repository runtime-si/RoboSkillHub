"""anchoring-a-bin-to-its-walls-from-the-wrist-camera: the wall sheets, the median and the outlier rule."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext

from ._bin_scenes import Scene, obb

BUNDLE = "anchoring-a-bin-to-its-walls-from-the-wrist-camera"
BOX = (0.10, -0.05, 0.45, 0.26, 20.0, 0.154)  # where the bin really is


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["anchor_bin"].module


@pytest.fixture(scope="module")
def frame():
    scene = Scene().open_box(*BOX, wall=0.004)
    scene.block(0.10, -0.05, 0.20, 0.12, 20.0, 0.07, rgb=(128, 128, 128))  # the pile
    return scene.frame(cam_xyz=(0.10, -0.05, 0.50), focal=400.0, name="right_wrist")


def _given(dx=0.0, dy=0.0):
    return obb(BOX[0] + dx, BOX[1] + dy, *BOX[2:])


def test_the_walls_give_the_shift_from_where_the_bin_was_given(script, frame):
    out = script.run(FakeContext({}), _given(-0.030, 0.020), camera_frame=frame)
    assert out["route"] == "anchored", out["detail"]
    assert sorted(out["walls_read"]) == ["+x", "+y", "-x", "-y"]
    assert out["shift"]["x"] == pytest.approx(0.030, abs=0.007)
    assert out["shift"]["y"] == pytest.approx(-0.020, abs=0.007)
    assert out["anchored_obb"]["center"]["x"] == pytest.approx(BOX[0], abs=0.007)
    assert out["anchored_obb"]["center"]["y"] == pytest.approx(BOX[1], abs=0.007)
    assert out["anchored_obb"]["extent"] == _given()["extent"]
    assert out["looks"] == 1 and len(out["shifts"]) == 1


def test_the_camera_is_found_in_the_observation(script, frame):
    obs = {"cameras": [dict(frame, name="top"), frame]}
    obs["cameras"][0]["depth"] = np.zeros_like(frame["depth"])
    out = script.run(FakeContext({"robot.get_observation": obs}), _given())
    assert out["route"] == "anchored" and out["camera"] == "right_wrist"
    none = script.run(
        FakeContext({"robot.get_observation": {"cameras": [obs["cameras"][0]]}}), _given()
    )
    assert none["route"] == "walls_not_seen" and none["anchored_obb"] == _given()


def test_walls_of_another_colour_need_a_mask_or_any(script, frame):
    grey = dict(frame, rgb=np.full_like(frame["rgb"], 128))
    out = script.run(FakeContext({}), _given(), camera_frame=grey)
    assert out["route"] == "walls_not_seen" and "wall-coloured" in out["detail"]
    r, b = frame["rgb"][..., 0].astype(int), frame["rgb"][..., 2].astype(int)
    mask = ((r - b > 22) * 255).astype(np.uint8)
    masked = script.run(FakeContext({}), _given(0.02, 0.0), camera_frame=grey, wall_mask=mask)
    assert masked["route"] == "anchored"
    assert masked["shift"]["x"] == pytest.approx(-0.02, abs=0.007)
    with pytest.raises(ValueError):
        script.wall_points(frame, "blue")


def test_one_wall_in_view_corrects_one_direction(script):
    c, s = math.cos(math.radians(20.0)), math.sin(math.radians(20.0))
    # a sheet of points on the +y side wall only, 12 mm further out than given
    lx = np.random.default_rng(0).uniform(-0.15, 0.15, 4000)
    lz = np.random.default_rng(1).uniform(0.06, 0.15, 4000)
    ly = np.full(4000, 0.13 - 0.002 + 0.012)
    pts = np.column_stack([BOX[0] + c * lx - s * ly, BOX[1] + s * lx + c * ly, lz])
    shift, read = script.wall_shift(pts, _given())
    assert read == ["+y"]
    assert np.hypot(*shift) == pytest.approx(0.012, abs=0.004)
    assert shift @ np.array([c, s]) == pytest.approx(0.0, abs=1e-9)  # nothing along the long axis


def test_too_few_points_or_a_thin_sheet_reads_nothing(script):
    assert script.wall_shift(np.zeros((100, 3)), _given()) == (None, [])
    spread = np.random.default_rng(2).uniform([-0.3, -0.3, 0.06], [0.5, 0.3, 0.15], (900, 3))
    assert script.wall_shift(spread, _given())[0] is None


def test_a_reading_far_from_the_given_bin_or_from_the_median_is_not_counted(script):
    assert script.accept(np.array([0.05, 0.0]), [])
    assert not script.accept(np.array([0.11, 0.0]), [])
    one = [[0.02, 0.0]]
    assert script.accept(np.array([0.07, 0.0]), one)  # one look is no median yet
    two = [[0.02, 0.0], [0.03, 0.01]]
    assert not script.accept(np.array([0.07, 0.0]), two)
    assert script.accept(np.array([0.03, 0.02]), two)


def test_an_outlier_look_leaves_the_bin_on_the_earlier_median(script, frame):
    past = [[0.060, 0.0], [0.064, 0.002], [0.062, -0.002]]
    out = script.run(FakeContext({}), _given(), past_shifts=past, camera_frame=frame)
    assert out["route"] == "outlier", out["detail"]
    assert out["shifts"] == past and out["looks"] == 3
    assert out["shift"]["x"] == pytest.approx(0.0, abs=0.007)
    assert out["used_shift"] == {"x": pytest.approx(0.062), "y": pytest.approx(0.0), "z": 0.0}
    assert out["anchored_obb"]["center"]["x"] == pytest.approx(BOX[0] + 0.062)


def test_the_median_of_the_looks_moves_the_bin(script, frame):
    past = [[0.004, 0.0], [-0.020, 0.010]]
    out = script.run(FakeContext({}), _given(), past_shifts=past, camera_frame=frame)
    assert out["route"] == "anchored" and out["looks"] == 3
    used = np.median(np.array(out["shifts"]), axis=0)
    assert out["used_shift"]["x"] == pytest.approx(used[0])
    assert abs(out["used_shift"]["x"]) <= 0.007  # the middle one of the three
