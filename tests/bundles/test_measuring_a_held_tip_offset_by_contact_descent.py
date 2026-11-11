import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills/measuring-a-held-tip-offset-by-contact-descent/scripts/tip_offset.py"
)
spec = importlib.util.spec_from_file_location("tip_offset", SCRIPT)
tip_offset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tip_offset)

DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}  # half a turn about x: tool z points down


class FakeHand:
    """A hand whose tool centre cannot go below ``floor`` (the tip on the table)."""

    def __init__(self, z0: float, floor: float, give: float = 0.0):
        self.z, self.floor, self.give = z0, floor, give
        self.calls = []

    def tool(self, name, **kw):
        self.calls.append(name)
        if name == "robot.get_ee_pose":
            return {"pose": {"position": {"x": 0.3, "y": 0.0, "z": self.z}, "rotation": DOWN}}
        if name in ("robot.go_to_pose", "robot.go_to_pose_cartesian"):
            target = kw["pose"]["position"]["z"]
            self.z = max(target, self.floor - self.give) if target < self.floor else target
            return {"success": True}
        raise KeyError(name)


def test_touch_reads_the_offset():
    hand = FakeHand(z0=0.85, floor=0.80)
    out = tip_offset.run(hand, surface_z=0.70)
    assert out["route"] == "measured"
    assert abs(out["tip_offset_m"] - 0.10) < 0.0015
    assert hand.calls[-1] == "robot.go_to_pose_cartesian"


def test_no_contact_when_surface_is_lower():
    hand = FakeHand(z0=0.90, floor=0.0)
    out = tip_offset.run(hand, surface_z=0.70, max_descent_m=0.02)
    assert out["route"] == "no_contact"


def test_tilted_approach_scales_the_offset():
    a = np.array([0.0, np.sin(np.radians(30)), -np.cos(np.radians(30))])
    assert abs(tip_offset.tip_offset(0.80, 0.70, a) - 0.1 / np.cos(np.radians(30))) < 1e-9


def test_approach_not_down_is_refused():
    hand = FakeHand(z0=0.9, floor=0.8)
    hand.tool = lambda name, **kw: {
        "pose": {
            "position": {"x": 0, "y": 0, "z": 0.9},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        }
    }
    assert tip_offset.run(hand, surface_z=0.7)["route"] == "invalid_input"
