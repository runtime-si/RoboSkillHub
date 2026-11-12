"""inserting-a-held-tip-with-the-other-arm-assisting: FakeContext tests of the views, the correction, the
separation check and the routing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "inserting-a-held-tip-with-the-other-arm-assisting"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
HOLE = np.array([0.43, -0.10, 0.04])  # true mouth
TIP = 0.10  # tip under the holding hand's tool centre


@pytest.fixture(scope="module")
def script(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["assist_insert"].module


class Station:
    """Arm 1 holds a peg TIP under its tool centre; the face stops the tip unless within 1 mm of the axis."""

    def __init__(self, one_arm=False, visible=True, refuse_camera=False):
        self.pos = {0: np.array([0.30, 0.25, 0.30]), 1: np.array([0.43, -0.10, 0.30])}
        self.rot = {0: DOWN, 1: DOWN}
        self.one_arm, self.visible, self.refuse_camera = one_arm, visible, refuse_camera
        self.views = 0

    def tip(self):
        return self.pos[1] - np.array([0.0, 0.0, TIP])

    def ctx(self):
        def a(i):
            return 0 if self.one_arm else i

        def pose(arm_id=0):
            p = self.pos[a(arm_id)]
            return {"position": dict(zip("xyz", p, strict=True)), "rotation": self.rot[a(arm_id)]}

        def go(pose, arm_id=0):
            if arm_id == 0 and self.refuse_camera:
                return {"outcome": "unreachable", "reason": "ik"}
            p = np.array([pose["position"][k] for k in "xyz"], float)
            if a(arm_id) == 1:
                lateral = np.linalg.norm((p - np.array([0, 0, TIP]) - HOLE)[:2])
                floor = HOLE[2] + TIP if lateral > 0.001 else HOLE[2] + TIP - 0.03
                p[2] = max(p[2], floor)
            self.pos[a(arm_id)], self.rot[a(arm_id)] = p, pose["rotation"]
            return {"outcome": "reached"}

        def offset(arm_id, image_horizontal):
            self.views += 1
            h = np.array([image_horizontal[k] for k in "xyz"])
            return {"visible": self.visible, "lateral_m": float(h @ (self.tip() - HOLE))}

        return FakeContext(
            {
                "robot.get_ee_pose": pose,
                "robot.go_to_pose": go,
                "robot.go_to_pose_cartesian": go,
                "vision.tip_hole_offset": offset,
            }
        )


def _mouth(off):
    return dict(zip("xyz", HOLE + np.array([*off, 0.0]), strict=True))


@pytest.mark.parametrize("off", [(0.004, 0.0), (0.0, -0.007), (0.008, 0.008), (-0.011, 0.002)])
def test_assist_corrects_a_wrong_hole_estimate_and_inserts(script, off):
    st = Station()
    out = script.run(st.ctx(), hole_mouth=_mouth(off), tip_offset_m=TIP)
    assert out["route"] == "inserted", out
    assert out["correction_m"] == pytest.approx(float(np.hypot(*off)), abs=1e-9)
    assert out["min_tcp_separation_m"] >= 0.15
    assert st.views == 4  # two views, then two more that read under the tolerance
    assert st.pos[0] == pytest.approx([0.30, 0.25, 0.30])  # the camera arm went back


def test_without_assist_the_tip_lands_on_the_face(script):
    out = script.run(
        Station().ctx(), hole_mouth=_mouth((0.006, 0.0)), tip_offset_m=TIP, assist=False
    )
    assert out["route"] == "tip_on_face"
    assert out["shortfall_m"] > 0.004


def test_solve_offset_from_two_views(script):
    d = script.view_directions(np.zeros(3), np.array([0.0, 1.0, 0.0]))
    hs = [np.cross([0.0, 0.0, 1.0], -v) for v in d]
    e = np.array([0.003, -0.005, 0.0])
    assert script.solve_offset(hs, [float(h @ e) for h in hs]) == pytest.approx(e)


def test_views_too_close_are_refused(script):
    st = Station()
    out = script.run(st.ctx(), hole_mouth=_mouth((0.004, 0.0)), tip_offset_m=TIP, standoff=0.08)
    assert out["route"] == "arms_too_close" and st.views == 0


def test_unseen_tip_is_not_in_view(script):
    out = script.run(
        Station(visible=False).ctx(), hole_mouth=_mouth((0.004, 0.0)), tip_offset_m=TIP
    )
    assert out["route"] == "not_in_view"


def test_a_correction_past_the_cap_is_not_taken(script):
    out = script.run(Station().ctx(), hole_mouth=_mouth((0.02, 0.0)), tip_offset_m=TIP)
    assert out["route"] == "not_in_view" and out["correction_m"] == 0.0


def test_camera_unreachable_and_one_arm(script):
    assert (
        script.run(
            Station(refuse_camera=True).ctx(), hole_mouth=_mouth((0.004, 0)), tip_offset_m=TIP
        )["route"]
        == "unreachable"
    )
    out = script.run(Station(one_arm=True).ctx(), hole_mouth=_mouth((0.004, 0)), tip_offset_m=TIP)
    assert out["route"] == "unreachable"


def test_example_declares_every_route():
    ex = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    assert set(ex["exit"]["success_values"]) | {ex["on_error"]} == {
        "inserted",
        "tip_on_face",
        "not_in_view",
        "arms_too_close",
        "unreachable",
    }


def test_pointing_source_back_projects_tip_and_hole(script):
    """A pinhole camera at each view: gemini-er.point returns the true tip and mouth pixels."""
    st = Station()
    k = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]])
    base = st.ctx()
    tools = dict(base._responses)
    calls = []

    def observation():
        p = st.pos[0]
        r = script._rot(st.rot[0])
        return {
            "cameras": [
                {
                    "name": "left_wrist",
                    "rgb": np.zeros((480, 640, 3), np.uint8),
                    "intrinsics": k.tolist(),
                    "pose": {"position": dict(zip("xyz", p, strict=True)), "rotation": st.rot[0]},
                    "_r": r,
                }
            ]
        }

    def point(image, query, max_points=1):
        cam = observation()["cameras"][0]
        w = st.tip() if "tip" in query else HOLE
        pc = cam["_r"].T @ (w - st.pos[0])
        calls.append(query)
        return {
            "found": True,
            "pixel_x": k[0, 0] * pc[0] / pc[2] + k[0, 2],
            "pixel_y": k[1, 1] * pc[1] / pc[2] + k[1, 2],
        }

    ctx = FakeContext({**tools, "robot.get_observation": observation, "gemini-er.point": point})
    out = script.run(
        ctx, hole_mouth=_mouth((0.005, -0.004)), tip_offset_m=TIP, offset_source="point"
    )
    assert out["route"] == "inserted", out
    assert out["correction_m"] == pytest.approx(float(np.hypot(0.005, 0.004)), abs=2e-4)
    assert len(calls) >= 4
