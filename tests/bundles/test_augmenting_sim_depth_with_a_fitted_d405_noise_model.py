"""augmenting-sim-depth-with-a-fitted-d405-noise-model: the fitted noise's statistics on a flat plane."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext

from roboskillhub.tasks.lego_hub import depth as D

BUNDLE = "augmenting-sim-depth-with-a-fitted-d405-noise-model"


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["augment_depth"].module


def plane(z=0.15):
    return np.full((480, 640), z, np.float32)


def test_noise_grows_with_z_squared_and_is_quantised_to_1_mm():
    rng = np.random.default_rng(0)
    near = D.add_fitted_noise(plane(0.10), D.D405_K, rng)
    far = D.add_fitted_noise(plane(0.20), D.D405_K, rng)
    s_near = near[near > 0].std()
    s_far = far[far > 0].std()
    assert 3.0 < s_far / s_near < 5.0  # (0.20 / 0.10)^2 = 4
    q = np.unique(np.round(near[near > 0] * 1e4).astype(int) % 10)
    assert set(q.tolist()) <= {0, 9, 1}  # 1 mm steps (float32 rounding)


def test_flat_patch_noise_matches_the_fit_order_of_magnitude():
    rng = np.random.default_rng(1)
    d = D.add_fitted_noise(plane(0.15), D.D405_K, rng, params={"lowfreq_coeff_per_m": 0.0})
    s = d[100:160, 100:160].std()
    assert 0.0002 < s < 0.0012  # fitted fine noise 0.51 mm at 15 cm, quantised


def test_edges_drop_out_and_scale_zero_is_only_quantisation():
    rng = np.random.default_rng(2)
    step = plane(0.15)
    step[:, 320:] = 0.12
    d = D.add_fitted_noise(step, D.D405_K, rng)
    edge = d[:, 318:322]
    assert 0.2 < (edge == 0).mean() < 0.7
    clean = D.add_fitted_noise(plane(0.1504), D.D405_K, rng, scale=0.0)
    assert np.allclose(clean[clean > 0], 0.150, atol=1e-6)


def test_script_noises_every_depth_camera_and_keeps_rgb(script):
    obs = {
        "cameras": [
            {
                "name": "wrist",
                "depth": plane(),
                "intrinsics": D.D405_K,
                "rgb": np.zeros((480, 640, 3), np.uint8),
            },
            {"name": "top", "rgb": np.zeros((4, 4, 3), np.uint8)},
        ]
    }
    out = script.run(FakeContext({}), observation=obs, seed=3)
    assert out["route"] == "augmented" and out["cameras"] == ["wrist"]
    w = out["observation"]["cameras"][0]
    assert (
        w["depth"].std() > 0
        and "fitted D405 noise" in w["source"]
        and w["rgb"].shape == (480, 640, 3)
    )
    assert np.array_equal(obs["cameras"][0]["depth"], plane())  # the input is not modified


def test_script_without_depth_and_fresh_observation(script):
    ctx = FakeContext(
        {"robot.get_observation": {"cameras": [{"name": "top", "rgb": np.zeros((4, 4, 3))}]}}
    )
    assert script.run(ctx)["route"] == "no_depth"


def test_script_and_station_models_agree(script):
    a = script.add_fitted_noise(plane(), D.D405_K, np.random.default_rng(5))
    b = D.add_fitted_noise(plane(), D.D405_K, np.random.default_rng(5))
    assert np.array_equal(a, b) and script.D405_FIT == D.D405_FIT
