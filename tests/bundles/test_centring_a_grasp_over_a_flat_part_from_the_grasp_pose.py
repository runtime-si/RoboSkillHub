"""centring-a-grasp-over-a-flat-part-from-the-grasp-pose: the depth finder on a ray-cast frame, and the skill's
corrections and routing with FakeContext."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import _wrist_rig as rig

BUNDLE = "centring-a-grasp-over-a-flat-part-from-the-grasp-pose"
PART = (0.21, -0.04, math.radians(20.0))


@pytest.fixture(scope="module")
def script(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b.canonical_scripts["centre_grasp"].module


def grasp_at(xy, yaw: float) -> np.ndarray:
    """A top-down grasp over ``xy``, the hand turned ``yaw`` about the vertical, the tool centre 12 mm up."""
    return rig.T(Rotation.from_euler("z", yaw).as_matrix() @ rig.DOWN, [xy[0], xy[1], 0.012])


def test_the_depth_finder_has_a_lying_part_to_half_a_millimetre(script):
    r = rig.Rig(rig.T(rig.DOWN, [0.20, -0.03, 0.07]), parts=[PART])
    obs = r.observation()
    cam = obs["cameras"][0]
    found = script.wd.flat_parts(cam["depth"], cam["intrinsics"], rig.matrix(cam["pose"]), rig.SIZE)
    assert len(found["parts"]) == 1
    p = found["parts"][0]
    assert np.hypot(p["centre"][0] - PART[0], p["centre"][1] - PART[1]) < 0.0005
    assert abs(script.half_turn(p["yaw"] - PART[2])) < math.radians(1.0)
    assert abs(found["support_off"]) < 0.0005


def test_a_grasp_planned_off_and_askew_is_brought_onto_the_part(script):
    planned_xy, planned_yaw = (PART[0] + 0.008, PART[1] - 0.005), PART[2] - math.radians(6.0)
    r = rig.Rig(rig.T(rig.DOWN, [0.1, -0.1, 0.2]), parts=[PART])
    out = script.run(
        r.ctx(), rig.obb(planned_xy, planned_yaw), rig.pose(grasp_at(planned_xy, planned_yaw))
    )
    assert out["route"] == "centred", out
    g = rig.matrix(out["grasp_pose_centred"])
    assert np.hypot(g[0, 3] - PART[0], g[1, 3] - PART[1]) < 0.001
    assert out["turn_deg"] == pytest.approx(6.0, abs=1.0)
    assert out["looks"] == 2 and out["residual_mm"] < 1.5
    assert np.allclose(
        r.tool, rig.matrix(out["pregrasp_pose"]), atol=1e-9
    )  # the hand is left over the corrected grasp
    assert g[2, 3] == pytest.approx(0.012)  # the grasp's height is the plan's


def test_a_grasp_already_on_the_part_is_left_alone(script):
    r = rig.Rig(rig.T(rig.DOWN, [0.1, -0.1, 0.2]), parts=[PART])
    out = script.run(r.ctx(), rig.obb(PART[:2], PART[2]), rig.pose(grasp_at(PART[:2], PART[2])))
    assert out["route"] == "centred" and out["looks"] == 1 and r.moves == 1
    assert abs(out["offset"]["x"]) < 1e-9 and out["turn_deg"] == 0.0


def test_no_part_under_the_hand_is_not_seen(script):
    r = rig.Rig(rig.T(rig.DOWN, [0.1, -0.1, 0.2]), parts=[])
    out = script.run(r.ctx(), rig.obb(PART[:2], PART[2]), rig.pose(grasp_at(PART[:2], PART[2])))
    assert out["route"] == "not_seen"
    assert np.allclose(
        rig.matrix(out["grasp_pose_centred"]), grasp_at(PART[:2], PART[2]), atol=1e-9
    )


def test_a_pregrasp_pose_out_of_reach_is_unreachable(script):
    r = rig.Rig(rig.T(rig.DOWN, [0.1, -0.1, 0.2]), parts=[PART], reach=lambda m: False)
    out = script.run(r.ctx(), rig.obb(PART[:2], PART[2]), rig.pose(grasp_at(PART[:2], PART[2])))
    assert out["route"] == "unreachable" and r.moves == 0
