"""simulating-d405-stereo-depth + d405-stereo: SGBM depth on synthetic pairs and the observation swap."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

K = [[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]]


@pytest.fixture(scope="module")
def st(skills_registry):
    return skills_registry.get("d405-stereo").tools_module


@pytest.fixture(scope="module")
def skill(skills_registry):
    return (
        skills_registry.get("simulating-d405-stereo-depth").canonical_scripts["stereo_depth"].module
    )


def pair(z=0.15, textured=True, seed=0):
    """A fronto-parallel plane at depth z: right image = left shifted by the disparity f*b/z."""
    rng = np.random.default_rng(seed)
    from scipy import ndimage

    if textured:
        tex = ndimage.gaussian_filter(rng.uniform(0, 255, (480, 900)), 1.2)
        tex = ((tex - tex.min()) / np.ptp(tex) * 220 + 15).astype(np.uint8)
    else:
        tex = np.full((480, 900), 128, np.uint8)
    d = 400.0 * 0.018 / z
    left = tex[:, 100:740]
    shift = int(round(d))
    right = tex[:, 100 + shift : 740 + shift]
    return left, right, shift


def test_disparity_bounds_cover_the_range(st):
    dmin, n = st.disparity_bounds(400.0, 0.018, 0.07, 0.5)
    assert dmin <= 400 * 0.018 / 0.5 and dmin + n >= 400 * 0.018 / 0.07 and n % 16 == 0


def test_textured_plane_depth_is_right(st):
    left, right, shift = pair(0.15)
    t = time.perf_counter()
    d = st.stereo_depth(left, right, 400.0, ir_noise=False, quant_m=0)
    assert time.perf_counter() - t < 5.0
    z_true = 400.0 * 0.018 / shift
    ok = d[:, 200:600] > 0
    assert ok.mean() > 0.9
    assert np.median(d[:, 200:600][ok]) == pytest.approx(z_true, abs=0.0015)


def test_untextured_plane_is_a_hole(st):
    left, right, _ = pair(0.15, textured=False)
    d = st.stereo_depth(left, right, 400.0, ir_noise=False)
    assert (d[:, 200:600] > 0).mean() < 0.05


def test_quantisation_and_range(st):
    left, right, _ = pair(0.15)
    d = st.stereo_depth(left, right, 400.0)
    v = d[d > 0]
    assert np.allclose(v * 1000, np.round(v * 1000), atol=1e-6)
    assert v.min() >= 0.07 and v.max() <= 0.5


def test_tool_returns_holes_and_range(st):
    left, right, _ = pair(0.15)
    r = st.depth_from_pair(left, right, K)
    assert 0 < r["hole_fraction"] < 0.5 and r["disparity_range_px"][0] >= 0
    assert np.asarray(r["depth"]).shape == (480, 640)


def test_skill_swaps_depth_and_keeps_truth(skill, st):
    left, right, _ = pair(0.15)
    truth = np.full((480, 640), 0.15, np.float32)
    obs = {
        "cameras": [
            {"name": "wrist", "ir_left": left, "ir_right": right, "intrinsics": K, "depth": truth},
            {"name": "top", "rgb": np.zeros((4, 4, 3), np.uint8)},
        ]
    }
    ctx = FakeContext({"d405-stereo.depth_from_pair": st.depth_from_pair})
    out = skill.run(ctx, camera="wrist", observation=obs)
    assert out["route"] == "depth_rendered" and 0 < out["hole_fraction"] < 0.5
    cam = out["observation"]["cameras"][0]
    assert cam["depth_true"] is truth and cam["depth"].shape == (480, 640)
    assert out["observation"]["cameras"][1]["name"] == "top"


def test_skill_without_a_pair(skill):
    ctx = FakeContext(
        {
            "robot.get_observation": {
                "cameras": [
                    {"name": "wrist", "rgb": np.zeros((4, 4, 3), np.uint8), "intrinsics": K}
                ]
            }
        }
    )
    out = skill.run(ctx, camera="wrist")
    assert out["route"] == "no_pair"

    def boom(**kw):
        raise RuntimeError("opencv missing")

    left, right, _ = pair()
    obs = {"cameras": [{"name": "wrist", "rgb": left, "rgb_right": right, "intrinsics": K}]}
    out = skill.run(
        FakeContext({"d405-stereo.depth_from_pair": boom}), camera="wrist", observation=obs
    )
    assert out["route"] == "no_pair" and "opencv missing" in out["reason"]


def test_example_declares_every_route(skills_registry):
    b = skills_registry.get("simulating-d405-stereo-depth")
    ex = json.loads((Path(b.bundle_dir) / "examples" / "canonical_subgraph.json").read_text())
    assert set(ex["exit"]["success_values"]) | {ex["on_error"]} == {"depth_rendered", "no_pair"}
