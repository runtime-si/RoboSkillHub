"""calibrating-a-wrist-camera-on-one-fixed-part: the fit on exact synthetic looks, the tour of poses, and the whole
skill on a ray-cast rig handed a wrong calibration."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from scipy.spatial.transform import Rotation

from . import _wrist_rig as rig

BUNDLE = "calibrating-a-wrist-camera-on-one-fixed-part"
PART = (0.20, -0.03, math.radians(35.0))
START = rig.T(rig.DOWN, [0.20, -0.03, 0.06])
WRONG = rig.CAMERA_IN_TOOL @ rig.T(
    Rotation.from_euler("xyz", [2.0, -1.5, 1.0], degrees=True).as_matrix(), [0.008, -0.006, 0.012]
)


@pytest.fixture(scope="module")
def script(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b.canonical_scripts["calibrate_wrist_camera"].module


def exact_views(script, x_true):
    """What each look of the tour sees of a point on a level support, exactly."""
    point, views = np.array([PART[0], PART[1], rig.SIZE[2]]), []
    for _name, tool in script.view_poses(START, np.array(PART[:2])):
        cam = tool @ x_true
        n = cam[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        views.append(
            {
                "T_base_tool": tool,
                "point_cam": (np.linalg.inv(cam) @ np.append(point, 1.0))[:3],
                "plane_n": n,
                "plane_d": float(-cam[2, 3]),
                "camera_in_tool": WRONG,
            }
        )
    return views


def test_the_tour_turns_shifts_and_tilts_the_hand_about_the_part(script):
    poses = script.view_poses(START, np.array(PART[:2]))
    assert len(poses) == 20
    names = [n for n, _p in poses]
    assert (
        sum(n.startswith("tilt") for n in names) == 4
        and sum(n.startswith("turn+90") for n in names) == 4
    )
    tilted = dict(poses)["tilt+25"]
    assert math.degrees(math.acos(-tilted[2, 2])) == pytest.approx(
        25.0, abs=0.01
    )  # the hand 25 deg off straight down


def test_the_fit_finds_the_camera_from_exact_looks(script):
    out = script.run(
        FakeContext({}), rig.obb(PART[:2], PART[2]), views=exact_views(script, rig.CAMERA_IN_TOOL)
    )
    assert out["route"] == "calibrated", out
    mm, deg = script.differ(rig.CAMERA_IN_TOOL, np.array(out["T_tool_camera"]))
    assert mm < 0.05 and deg < 0.01
    assert out["point_spread_before_mm"] > 5.0 and out["point_spread_mm"] < 0.01
    assert out["moved_mm"] == pytest.approx(15.6, abs=0.5)


def test_too_few_looks_are_not_fitted(script):
    out = script.run(
        FakeContext({}),
        rig.obb(PART[:2], PART[2]),
        views=exact_views(script, rig.CAMERA_IN_TOOL)[:5],
    )
    assert out["route"] == "too_few_views" and out["views"] == 5


def test_looks_that_cannot_agree_are_inconsistent(script):
    views = exact_views(script, rig.CAMERA_IN_TOOL)
    for i, v in enumerate(views):  # an arm that is somewhere else at every pose
        v["T_base_tool"] = v["T_base_tool"].copy()
        v["T_base_tool"][:2, 3] += 0.008 * np.array([math.cos(2.4 * i), math.sin(2.4 * i)])
    out = script.run(FakeContext({}), rig.obb(PART[:2], PART[2]), views=views)
    assert out["route"] == "inconsistent" and out["point_spread_mm"] > 2.5


def test_the_whole_skill_on_a_rig_handed_a_wrong_calibration(script):
    r = rig.Rig(START, parts=[PART], believed=WRONG)
    out = script.run(r.ctx(), rig.obb(PART[:2], PART[2]))
    assert out["route"] == "calibrated", out
    assert out["views"] >= 12
    mm, deg = script.differ(rig.CAMERA_IN_TOOL, np.array(out["T_tool_camera"]))
    assert mm < 1.5 and deg < 0.3
    assert out["point_spread_before_mm"] > 3.0 and out["point_spread_mm"] < 1.0
    assert np.allclose(r.tool, START, atol=1e-9)  # back where it began
