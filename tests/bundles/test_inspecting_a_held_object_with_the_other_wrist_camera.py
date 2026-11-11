"""inspecting-a-held-object-with-the-other-wrist-camera: FakeContext tests of the moves, the framing test
and the routing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "inspecting-a-held-object-with-the-other-wrist-camera"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
# A 60 x 40 x 100 mm box held top-down by arm 0, its tool centre 30 mm above the box's centre.
BOX = {
    "center": {"x": 0.45, "y": 0.20, "z": 0.12},
    "extent": {"x": 0.03, "y": 0.02, "z": 0.05},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}
INSPECT = {"x": 0.45, "y": 0.0, "z": 0.30}
UNDER = {"x": 0.0, "y": 0.0, "z": -1.0}  # look at the underside
SIDE = {"x": 0.0, "y": -1.0, "z": 0.0}


@pytest.fixture(scope="module")
def script(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["inspect_held"].module


class Arms:
    def __init__(self, grip0=0.4, one_arm=False, refuse=(), sag=None):
        self.pos = {0: np.array([0.45, 0.20, 0.15]), 1: np.array([0.45, -0.25, 0.30])}
        self.rot = {0: DOWN, 1: DOWN}
        self.grip0, self.one_arm, self.refuse, self.sag = grip0, one_arm, set(refuse), sag
        self.moves: list[tuple[int, np.ndarray]] = []

    def ctx(self):
        def a(arm_id):
            return 0 if self.one_arm else arm_id

        def pose(arm_id=0):
            p = self.pos[a(arm_id)]
            return {"position": {"x": p[0], "y": p[1], "z": p[2]}, "rotation": self.rot[a(arm_id)]}

        def go(pose, arm_id=0):
            if arm_id in self.refuse:
                return {"outcome": "unreachable", "reason": "ik failed"}
            p = np.array([pose["position"][k] for k in "xyz"], float)
            if self.sag and arm_id == self.sag[0]:
                p = p + np.asarray(self.sag[1])
            self.pos[a(arm_id)], self.rot[a(arm_id)] = p, pose["rotation"]
            self.moves.append((arm_id, p))
            return {"outcome": "reached"}

        return FakeContext(
            {
                "robot.get_ee_pose": pose,
                "robot.get_gripper": lambda arm_id=0: {
                    "position": self.grip0 if arm_id == 0 else 1.0
                },
                "robot.go_to_pose": go,
            }
        )


def _run(script, arms, **kw):
    args = {"object_obb": BOX, "inspection_position": INSPECT, "view_axis": UNDER, **kw}
    return script.run(arms.ctx(), **args)


def test_underside_inspection_frames_the_object(script):
    arms = Arms()
    out = _run(script, arms)
    assert out["route"] == "inspected", out
    assert out["range_m"] == pytest.approx(0.05, abs=1e-6)
    assert out["lateral_m"] == pytest.approx(0.0, abs=1e-6)
    # The object's centre landed on the inspection point, the camera 5 cm under it looking up.
    assert [out["object_position"][k] for k in "xyz"] == pytest.approx([0.45, 0.0, 0.30])
    assert arms.pos[1] == pytest.approx([0.45, 0.0, 0.25])
    # Holding arm first, then the camera arm.
    assert [m[0] for m in arms.moves] == [0, 1]


def test_side_view_points_the_optical_axis_at_the_object(script):
    arms = Arms()
    out = _run(script, arms, view_axis=SIDE, standoff=0.06)
    assert out["route"] == "inspected", out
    r = script._rotation(out["camera_pose"]["rotation"])
    assert r[:, 2] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    assert r[2, 0] == pytest.approx(0.0, abs=1e-9)  # tool x kept level


def test_camera_offset_is_the_optical_centre(script):
    arms = Arms()
    out = _run(script, arms, camera_offset={"x": 0.0, "y": 0.03, "z": -0.02})
    assert out["route"] == "inspected", out
    assert [out["camera_pose"]["position"][k] for k in "xyz"] == pytest.approx([0.45, 0.0, 0.25])


def test_framing_band_from_scan_object(script):
    cam, axis = np.zeros(3), np.array([0.0, 0.0, 1.0])
    assert script.framing(cam, axis, np.array([0.0, 0.0, 0.05])) == pytest.approx((0.05, 0.0))
    d, lat = script.framing(cam, axis, np.array([0.03, 0.0, 0.05]))
    assert d == pytest.approx(0.05) and lat == pytest.approx(0.03)
    assert script.framing(cam, axis, np.array([0.0, 0.0, -0.02]))[0] < 0  # behind the camera


def test_too_far_is_not_in_view(script):
    out = _run(script, Arms(), standoff=0.09)
    assert out["route"] == "target_not_in_view"
    assert out["range_m"] == pytest.approx(0.09)


def test_camera_arm_sag_off_the_ray_is_not_in_view(script):
    out = _run(script, Arms(sag=(1, [0.015, 0.0, 0.0])), lateral_tolerance=0.01)
    assert out["route"] == "target_not_in_view"
    assert out["lateral_m"] == pytest.approx(0.015)


def test_arms_too_close_does_not_move_the_camera_arm(script):
    arms = Arms()
    # Viewing from above puts the camera right on the holding hand (30 mm over the object's centre).
    out = _run(script, arms, view_axis={"x": 0.0, "y": 0.0, "z": 1.0}, standoff=0.035)
    assert out["route"] == "arms_too_close"
    assert [m[0] for m in arms.moves] == [0]


def test_empty_holding_hand(script):
    out = _run(script, Arms(grip0=0.01))
    assert out["route"] == "target_not_in_view" and "holds nothing" in out["detail"]


def test_one_arm_connector_is_refused(script):
    out = _run(script, Arms(one_arm=True))
    assert out["route"] == "unreachable" and "no second arm" in out["detail"]
    assert _run(script, Arms(), camera_arm=0)["route"] == "unreachable"


@pytest.mark.parametrize("arm", [0, 1])
def test_refused_moves_route_unreachable(script, arm):
    out = _run(script, Arms(refuse=[arm]))
    assert out["route"] == "unreachable"


def test_example_declares_every_route():
    ex = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert declared == {"inspected", "target_not_in_view", "arms_too_close", "unreachable"}
