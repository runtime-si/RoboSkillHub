"""placing-on-receptacle-top-surface: OK-Robot's drop point and surface height on synthetic point clouds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "placing-on-receptacle-top-surface"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE


@pytest.fixture(scope="module")
def script(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["place_point"].module


def _box_top(cx=0.1, cy=0.3, sx=0.08, sy=0.06, z=0.05, step=0.002):
    xs, ys = np.meshgrid(
        np.arange(cx - sx / 2, cx + sx / 2, step), np.arange(cy - sy / 2, cy + sy / 2, step)
    )
    return np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)])


def test_box_top(script):
    res = script.run(FakeContext(), points=_box_top().tolist())
    assert res["route"] == "place_point"
    assert res["surface_z"] == pytest.approx(0.05, abs=1e-6)
    assert res["place_pose"]["position"]["x"] == pytest.approx(0.1, abs=0.006)
    assert res["place_pose"]["position"]["z"] == pytest.approx(0.07, abs=1e-6)


def test_dense_side_does_not_pull_the_point(script):
    top = _box_top()
    side = _box_top(cx=0.13, sx=0.02, step=0.0005)  # the near quarter seen 16x denser
    res = script.run(FakeContext(), points=np.vstack([top, side]).tolist())
    assert res["place_pose"]["position"]["x"] == pytest.approx(0.1, abs=0.008)


def test_tray_rim_stays_out_of_the_surface(script):
    floor = _box_top(sx=0.12, sy=0.12, z=0.01)
    rim = np.vstack([_box_top(cx=0.1 + s * 0.058, sx=0.004, sy=0.12, z=0.03) for s in (-1, 1)])
    res = script.run(FakeContext(), points=np.vstack([floor, rim]).tolist())
    assert res["surface_z"] == pytest.approx(0.01, abs=1e-6)


def test_ring_has_no_surface(script):
    a = np.linspace(0, 2 * np.pi, 400)
    ring = np.column_stack([0.1 + 0.05 * np.cos(a), 0.3 + 0.05 * np.sin(a), np.full(a.size, 0.04)])
    assert script.run(FakeContext(), points=ring.tolist())["route"] == "receptacle_not_found"


def test_noisy_depth(script):
    rng = np.random.default_rng(0)
    pts = _box_top()
    pts[:, 2] += rng.normal(0, 0.002, len(pts))
    pts[0, 2] = 0.2  # one wild pixel
    res = script.run(FakeContext(), points=pts.tolist())
    assert res["surface_z"] == pytest.approx(0.05 + 1.645 * 0.002, abs=0.002)


def test_held_bottom_offset(script):
    res = script.run(FakeContext(), points=_box_top().tolist(), held_bottom_offset=0.015)
    assert res["place_pose"]["position"]["z"] == pytest.approx(0.085, abs=1e-6)


def test_camera_path(script):
    pts = _box_top()
    cam = {
        "name": "top",
        "rgb": np.zeros((4, 4, 3)),
        "depth": np.ones((4, 4)),
        "intrinsics": np.eye(3),
        "pose": np.eye(4),
    }
    ctx = FakeContext(
        {
            "robot.get_observation": {"cameras": [cam]},
            "sam3.segment_text": {"masks": [np.ones((4, 4))], "scores": [0.9]},
            "geometry.mask_to_world_points": {"points": pts.tolist()},
        }
    )
    res = script.run(ctx, query="blue box")
    assert res["route"] == "place_point" and res["n_points"] == len(pts)
    ctx = FakeContext(
        {"robot.get_observation": {"cameras": [cam]}, "sam3.segment_text": {"masks": []}}
    )
    assert script.run(ctx, query="blue box")["route"] == "receptacle_not_found"


def test_invalid_input(script):
    assert script.run(FakeContext())["route"] == "invalid_input"


def test_example_declares_every_route(script):
    ex = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    assert set(ex["exit"]["success_values"]) | {ex["on_error"]} == {
        "place_point",
        "receptacle_not_found",
        "invalid_input",
    }
