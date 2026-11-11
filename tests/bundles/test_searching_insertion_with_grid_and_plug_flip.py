import importlib.util
import math
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills/searching-insertion-with-grid-and-plug-flip/scripts/grid_search.py"
)
spec = importlib.util.spec_from_file_location("grid_search", SCRIPT)
gs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gs)

DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


class FakePort:
    """A base at z=0.80 (tool centre height at touch) with a hole at (hx, hy) that takes the plug only when
    ``keyed`` matches the hand's turn."""

    def __init__(self, hx, hy, needs_flip=False, z0=0.81):
        self.x, self.y, self.z = 0.0, 0.0, z0
        self.hx, self.hy, self.needs_flip = hx, hy, needs_flip
        self.rot = DOWN

    def floor(self):
        flipped = abs(self.rot["w"] - DOWN["w"]) > 1e-6 or abs(self.rot["x"] - DOWN["x"]) > 1e-6
        in_hole = (
            math.hypot(self.x - self.hx, self.y - self.hy) < 0.0008 and flipped == self.needs_flip
        )
        return 0.78 if in_hole else 0.80

    def tool(self, name, **kw):
        if name == "robot.get_ee_pose":
            return {
                "pose": {"position": {"x": self.x, "y": self.y, "z": self.z}, "rotation": self.rot}
            }
        if name == "robot.go_to_pose":
            p = kw["pose"]["position"]
            self.x, self.y, self.rot = p["x"], p["y"], kw["pose"]["rotation"]
            self.z = max(p["z"], self.floor())
            return {"success": True}
        raise KeyError(name)


def test_finds_offset_hole():
    out = gs.run(FakePort(0.0015, -0.003))
    assert out["route"] == "inserted"
    assert abs(out["offset_x"] - 0.0015) < 1e-9 and abs(out["offset_y"] + 0.003) < 1e-9
    assert abs(out["base_z"] - 0.80) < 0.0011


def test_flips_when_keyed_the_other_way():
    out = gs.run(FakePort(0.0, 0.0, needs_flip=True))
    assert out["route"] == "flipped_and_inserted" and out["flipped"]


def test_exhausted_without_hole():
    out = gs.run(FakePort(0.02, 0.02), allow_flip=False)
    assert out["route"] == "grid_exhausted" and out["probes"] == 49


def test_turn_about_tool_z_is_half_turn():
    q = gs.turned_about_tool_z(DOWN, 180.0)
    assert abs(abs(q["y"]) - 1.0) < 1e-9


def test_grid_is_centre_out():
    cells = gs.grid_cells(0.003, 0.0015)
    assert cells[0] == (0.0, 0.0) and len(cells) == 25
