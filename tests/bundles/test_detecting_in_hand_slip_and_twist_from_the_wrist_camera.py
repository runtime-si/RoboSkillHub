"""detecting-in-hand-slip-and-twist-from-the-wrist-camera: synthetic wrist frames for each exit."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "detecting-in-hand-slip-and-twist-from-the-wrist-camera"
H, W, F = 120, 160, 200.0
K = [[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]]
TOP_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["detect_slip"].module


def texture(seed=0):
    rng = np.random.default_rng(seed)
    import cv2

    t = rng.integers(0, 255, (H + 40, W + 40)).astype(np.uint8)
    return cv2.GaussianBlur(t, (5, 5), 1.5)


def frame(dx=0, tilt_deg=0.0, present=True, tex=None):
    """The part fills rows 30-90 x cols 40-120 at 0.1 m, the scene behind it at 0.5 m; ``dx`` shifts the part
    (texture and region) along the image x; ``tilt_deg`` tilts its plane about the camera x axis."""
    tex = texture() if tex is None else tex
    img = tex[20 : 20 + H, 20 : 20 + W].copy()
    depth = np.full((H, W), 0.5)
    if present:
        v, u = np.mgrid[0:H, 0:W]
        part = (v >= 30) & (v < 90) & (u >= 40 + dx) & (u < 120 + dx)
        shifted = tex[20 : 20 + H, 20 - dx : 20 - dx + W]
        img[part] = shifted[part]
        y_cam = (v - H / 2) * 0.1 / F
        depth[part] = (0.1 + math.tan(math.radians(tilt_deg)) * y_cam)[part]
    return {
        "name": "wrist_cam",
        "rgb": np.stack([img] * 3, axis=-1),
        "depth": depth,
        "intrinsics": K,
    }


def ctx(frames):
    it = iter(frames)
    last = {"f": None}

    def obs(**kw):
        try:
            last["f"] = next(it)
        except StopIteration:
            pass
        return {"cameras": [last["f"]]}

    return FakeContext(
        {
            "robot.get_observation": obs,
            "robot.get_ee_pose": {
                "pose": {"position": {"x": 0, "y": 0, "z": 0}, "rotation": TOP_DOWN}
            },
        }
    )


FAST = {"duration_s": 0.6, "period_s": 0.0}


def test_still_part_is_stable(script):
    out = script.run(ctx([frame()] * 5), **FAST)
    assert out["route"] == "stable", out["detail"]
    assert out["slip_mm"] < 0.3
    assert abs(out["twist_deg"]) < 0.5


def test_sliding_part_reads_slipping_in_mm(script):
    out = script.run(ctx([frame(), frame(dx=4), frame(dx=8), frame(dx=8)]), **FAST)
    assert out["route"] == "slipping", out["detail"]
    assert out["slip_mm"] == pytest.approx(4.0, abs=0.6)  # 8 px at 0.1 m with f = 200 px


def test_tilting_part_reads_twisted_about_the_closing_axis(script):
    out = script.run(ctx([frame(), frame(tilt_deg=6.0)] + [frame(tilt_deg=6.0)] * 2), **FAST)
    assert out["route"] == "twisted", out["detail"]
    assert abs(out["twist_deg"]) == pytest.approx(6.0, abs=0.5)


def test_part_leaving_the_hand_reads_dropped(script):
    out = script.run(ctx([frame(), frame(present=False)]), **FAST)
    assert out["route"] == "dropped"
    assert out["area_fraction"] < 0.4


def test_nothing_held_at_the_start_is_dropped(script):
    out = script.run(ctx([frame(present=False)]), **FAST)
    assert out["route"] == "dropped"


def test_no_wrist_camera(script):
    c = FakeContext({"robot.get_observation": {"cameras": [{"name": "overhead"}]}})
    out = script.run(c, **FAST)
    assert out["route"] == "no_camera"


def test_angle_about_and_plane_normal(script):
    a, b = np.array([0.0, 0.0, -1.0]), np.array([0.0, math.sin(0.1), -math.cos(0.1)])
    assert abs(script.angle_about(a, b, np.array([1.0, 0.0, 0.0]))) == pytest.approx(
        math.degrees(0.1), abs=1e-6
    )
    assert script.angle_about(a, b, np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0, abs=1e-6)
