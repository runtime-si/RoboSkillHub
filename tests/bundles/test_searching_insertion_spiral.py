"""searching-insertion-spiral: the probe grid, and the search against a fake surface with a hole in it."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "searching-insertion-spiral"
ROT = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
MOUTH = {"x": 0.40, "y": 0.10, "z": 0.05}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["spiral_search"].module


class Surface:
    """A plate at the mouth's height with a round hole: a probe over the hole sinks, elsewhere it stops."""

    def __init__(self, hole_xy, hole_r=0.0015, contact_raises=False, unreachable_xy=None):
        self.p = np.array([0.40, 0.10, 0.20])
        self.hole = np.asarray(hole_xy, dtype=float)
        self.hole_r = hole_r
        self.contact_raises = contact_raises
        self.unreachable_xy = unreachable_xy
        self.probed = []

    def ee(self, **kw):
        return {
            "pose": {"position": dict(zip("xyz", map(float, self.p), strict=True)), "rotation": ROT}
        }

    def cart(self, pose, **kw):
        target = np.array([pose["position"][k] for k in "xyz"])
        if (
            self.unreachable_xy is not None
            and np.linalg.norm(target[:2] - self.unreachable_xy) < 1e-9
        ):
            return {"reached": False, "outcome": "unreachable"}
        self.p = target
        return {"reached": True, "outcome": "reached"}

    def probe(self, pose, **kw):
        target = np.array([pose["position"][k] for k in "xyz"])
        self.probed.append(target[:2].copy())
        over_hole = np.linalg.norm(target[:2] - self.hole) <= self.hole_r
        stop = target[2] if over_hole else MOUTH["z"] + 0.0004
        self.p = np.array([target[0], target[1], stop])
        if not over_hole and self.contact_raises:
            raise ToolError(
                "robot.move_cartesian_until_contact", "the tool did not reach its target"
            )
        return {"reached": over_hole, "outcome": "reached" if over_hole else "contact"}

    def ik(self, pose, **kw):
        xy = np.array([pose["position"]["x"], pose["position"]["y"]])
        far = self.reach_limit_x is not None and xy[0] > self.reach_limit_x
        return {"success": True, "within_servo_tolerance": not far}

    reach_limit_x = None

    def ctx(self):
        return FakeContext(
            {
                "robot.get_ee_pose": self.ee,
                "robot.go_to_pose_cartesian": self.cart,
                "robot.move_cartesian_until_contact": self.probe,
                "robot.solve_ik": self.ik,
            }
        )


def test_the_grid_is_hexagonal_nearest_first_and_37_points_by_default(script):
    grid = script.hex_grid(script.SEARCH_RADIUS, script.PITCH)
    assert len(grid) == 37
    assert grid[0] == (0.0, 0.0)
    radii = [math.hypot(*p) for p in grid]
    assert radii == sorted(radii, key=lambda r: round(r, 9))
    assert sum(1 for r in radii if abs(r - script.PITCH) < 1e-9) == 6
    # Covering radius: no point of the disc is further than pitch / sqrt(3) from a probe.
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, (4000, 2)) * script.SEARCH_RADIUS * 0.8
    pts = pts[np.hypot(*pts.T) <= script.SEARCH_RADIUS * 0.8]
    g = np.asarray(grid)
    worst = np.max(np.min(np.linalg.norm(pts[:, None] - g[None], axis=2), axis=1))
    assert worst <= script.PITCH / math.sqrt(3) + 1e-9


def test_finds_a_hole_off_the_estimate_and_stays_in(script):
    hole = (MOUTH["x"] + 0.004, MOUTH["y"] - 0.0035)
    s = Surface(hole)
    out = script.run(s.ctx(), mouth_position=MOUTH)
    assert out["route"] == "inserted"
    assert out["depth"] == pytest.approx(script.MAX_DEPTH)
    assert (
        np.linalg.norm(np.array([out["entered_at"]["x"], out["entered_at"]["y"]]) - hole) <= 0.0015
    )
    assert out["probes"] == len(s.probed) > 1
    assert s.p[2] == pytest.approx(MOUTH["z"] - script.MAX_DEPTH)  # left in the hole


def test_a_station_style_error_on_contact_is_read_as_a_stop(script):
    s = Surface((MOUTH["x"] + 0.002, MOUTH["y"]), contact_raises=True)
    out = script.run(s.ctx(), mouth_position=MOUTH, stay_in=False)
    assert out["route"] == "inserted"
    assert s.p[2] == pytest.approx(MOUTH["z"] + script.HOVER)  # lifted back out


def test_no_hole_in_reach_is_exhausted(script):
    s = Surface((MOUTH["x"] + 0.03, MOUTH["y"]))
    out = script.run(s.ctx(), mouth_position=MOUTH)
    assert out["route"] == "search_exhausted"
    assert out["probes"] == 37
    assert "37 probes" in out["detail"] and "0.4 mm" in out["detail"]


def test_an_unreachable_point_is_skipped(script):
    centre = np.array([MOUTH["x"], MOUTH["y"]])
    s = Surface((MOUTH["x"] + 0.002, MOUTH["y"]), unreachable_xy=centre)
    out = script.run(s.ctx(), mouth_position=MOUTH)
    assert out["route"] == "inserted"
    assert not any(np.allclose(p, centre) for p in s.probed)


def test_nothing_reachable_is_unreachable(script):
    s = Surface((MOUTH["x"], MOUTH["y"]))
    s.cart = lambda pose, **kw: {"reached": False, "outcome": "unreachable"}
    out = script.run(s.ctx(), mouth_position=MOUTH)
    assert out["route"] == "unreachable" and out["probes"] == 0


def test_probes_out_of_reach_are_skipped_not_read_as_surface(script):
    # The hole lies past the arm's reach: without the preview the stall there
    # would read as contact; with it the probes are skipped and counted.
    s = Surface((MOUTH["x"] + 0.004, MOUTH["y"]))
    s.reach_limit_x = MOUTH["x"] + 0.001
    ctx = s.ctx()
    out = script.run(ctx, mouth_position=MOUTH)
    assert out["route"] == "search_exhausted"
    assert all(p[0] <= MOUTH["x"] + 0.001 + 1e-9 for p in s.probed)
    skipped = sum(
        1
        for c in ctx.calls_to("robot.solve_ik")
        if c.kwargs["pose"]["position"]["x"] > MOUTH["x"] + 0.001
    )
    assert f"({skipped} skipped)" in out["detail"] and out["probes"] == 37 - skipped


def test_no_ik_answer_does_not_skip(script):
    s = Surface((MOUTH["x"] + 0.002, MOUTH["y"]))
    tools = {
        "robot.get_ee_pose": s.ee,
        "robot.go_to_pose_cartesian": s.cart,
        "robot.move_cartesian_until_contact": s.probe,
    }
    out = script.run(FakeContext(tools), mouth_position=MOUTH)  # robot.solve_ik raises: not served
    assert out["route"] == "inserted"
