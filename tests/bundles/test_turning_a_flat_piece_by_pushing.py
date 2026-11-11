"""turning-a-flat-piece-by-pushing: the yaw fit on a generated piece's footprint."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from gap.testing import FakeContext

from .jigsaw_cut import make_puzzle

BUNDLE = "turning-a-flat-piece-by-pushing"
MPP = 0.00025


@pytest.fixture(scope="module")
def script(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["push_turn"].module


@pytest.fixture(scope="module")
def mask():
    pz = make_puzzle(3, 3, cell=(0.0225, 0.0225), seed=3)
    poly = pz.outline(1, 1) - pz.centre(1, 1)
    m = np.zeros((200, 200), np.uint8)
    cv2.fillPoly(
        m, [np.round(np.column_stack([poly[:, 0], -poly[:, 1]]) / MPP + 100).astype(np.int32)], 1
    )
    return m


@pytest.mark.parametrize("yaw", [0.0, 23.0, -71.0, 140.0])
def test_the_yaw_of_a_footprint_is_recovered(script, mask, yaw):
    tmpl = script.template_points(mask, MPP)
    a = math.radians(yaw)
    pts = tmpl @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) + [0.6, 0.2]
    pts = pts + np.random.default_rng(0).normal(0, 0.0002, pts.shape)
    got, fit = script.fit_yaw(pts, tmpl, None)
    assert abs((got - yaw + 180) % 360 - 180) < 1.5
    assert fit > 0.6
    got2, _ = script.fit_yaw(pts, tmpl, yaw + 12.0)
    assert abs((got2 - yaw + 180) % 360 - 180) < 1.5


def test_no_camera_or_mask(script, mask):
    ctx = FakeContext(
        {
            "robot.get_ee_pose": {
                "position": {"x": 0, "y": 0, "z": 0.3},
                "rotation": {"w": 0, "x": 1, "y": 0, "z": 0},
            },
            "robot.get_observation": {"cameras": []},
        }
    )
    assert script.run(ctx, mask, MPP, {"x": 0.6, "y": 0.2}, 0.0)["route"] == "not_seen"
    assert (
        script.run(ctx, np.zeros((10, 10)), MPP, {"x": 0.6, "y": 0.2}, 0.0)["route"] == "not_seen"
    )
