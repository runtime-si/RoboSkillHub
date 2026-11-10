"""D405 stereo depth option for the LEGO bridge and the YAM domino vision path."""

import numpy as np
import pytest

from roboskillhub.tasks.lego_hub import depth as D


def _pair(z, textured=True):
    rng = np.random.default_rng(0)
    from scipy import ndimage

    tex = (
        ndimage.gaussian_filter(rng.uniform(0, 255, (480, 900)), 1.2)
        if textured
        else np.full((480, 900), 128.0)
    )
    tex = ((tex - tex.min()) / (np.ptp(tex) + 1e-9) * 220 + 15).astype(np.uint8)
    shift = int(round(float(D.D405_K[0, 0]) * D.D405_BASELINE_M / z))
    return (
        np.dstack([tex[:, 100:740]] * 3),
        np.dstack([tex[:, 100 + shift : 740 + shift]] * 3),
        shift,
    )


def test_stereo_depth_on_a_textured_plane():
    left, right, shift = _pair(0.15)
    d = D.stereo_depth(left, right)
    ok = d[:, 200:600] > 0
    assert ok.mean() > 0.9
    assert np.median(d[:, 200:600][ok]) == pytest.approx(
        float(D.D405_K[0, 0]) * D.D405_BASELINE_M / shift, abs=0.0015
    )
    assert d.dtype == np.float32 and d.shape == (480, 640)


def test_stereo_depth_leaves_untextured_faces_empty():
    left, right, _ = _pair(0.15, textured=False)
    assert (D.stereo_depth(left, right)[:, 200:600] > 0).mean() < 0.05


def test_domino_vision_depth_option(monkeypatch):
    from roboskillhub.tasks.assembly_eval.yam_vision import YamVision

    assert YamVision().depth_mode == "rendered"
    assert YamVision(depth="stereo").depth_mode == "stereo"
    monkeypatch.setenv("DOMINO_VISION_DEPTH", "fitted")
    assert YamVision().depth_mode == "fitted"
    with pytest.raises(ValueError, match="rendered, fitted or stereo"):
        YamVision(depth="sgbm")
