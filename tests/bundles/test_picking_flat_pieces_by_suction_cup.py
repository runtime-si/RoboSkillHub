"""picking-flat-pieces-by-suction-cup: the ARC affordance port and the pick routing with a fake station."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "picking-flat-pieces-by-suction-cup"
K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
H, W = 480, 640


@pytest.fixture(scope="module")
def bundle(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b


@pytest.fixture(scope="module")
def pick(bundle):
    return bundle.canonical_scripts["pick_suction"].module


@pytest.fixture(scope="module")
def sp(bundle):
    spec = importlib.util.spec_from_file_location(
        "sp_under_test", Path(bundle.bundle_dir) / "scripts" / "suction_points.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def scene(piece_top=0.298, table=0.300, half_px=40, bump=False):
    """A square piece 2 mm proud of the table under a camera looking down (depth in metres)."""
    d = np.full((H, W), table)
    mask = np.zeros((H, W), bool)
    mask[240 - half_px : 240 + half_px, 320 - half_px : 320 + half_px] = True
    d[mask] = piece_top
    if bump:  # a ridge across the left half of the piece
        v, u = np.mgrid[0:H, 0:W]
        ridge = mask & (u < 320) & (np.sin(u / 3.0) > 0)
        d[ridge] -= 0.002
    return d, mask


def test_flat_piece_scores_high_and_point_is_interior(sp):
    d, mask = scene()
    p = sp.best_point(d, K, mask, cup_radius_m=0.004)
    assert p is not None and p["affordance"] > 0.8
    u, v = p["pixel"]
    assert abs(u - 320) <= 3 and abs(v - 240) <= 3
    assert p["point_camera"][2] == pytest.approx(0.298)
    assert np.allclose(p["normal_camera"], [0, 0, -1], atol=1e-6)


def test_rough_half_is_avoided(sp):
    d, mask = scene(bump=True)
    aff, _ = sp.affordance(d, K, mask)
    assert aff[240, 340].mean() > aff[240, 300].mean()
    p = sp.best_point(d, K, mask, cup_radius_m=0.003)
    assert p["pixel"][0] > 320


def test_piece_too_small_for_the_cup(sp):
    d, mask = scene(half_px=4)
    assert sp.best_point(d, K, mask, cup_radius_m=0.004) is None
    assert sp.best_point(d, K, np.zeros_like(mask), cup_radius_m=0.004) is None


def test_rim_rise_skips_occluding_pixels(pick):
    before = np.full((10, 10), 0.300)
    after = before - 0.005
    after[:3] = 0.25  # the hand in front of part of the rim
    rim = np.ones((10, 10), bool)
    rise, seen = pick.rim_rise(before, after, rim, 0.005)
    assert rise == pytest.approx(0.005) and seen == pytest.approx(0.7)


class Station:
    """Camera looking down from 0.3 m over a table; the piece rises with the hand when sealed."""

    def __init__(self, seal=True, drop_in_transport=False, sensor=False, refuse=False):
        self.hand = np.array([0.0, 0.0, 0.1])
        self.seal, self.drop, self.sensor, self.refuse = seal, drop_in_transport, sensor, refuse
        self.vacuum = False
        self.attached = False
        self.piece_z = 0.002
        self.calls = []

    def depth(self):
        d, mask = scene()
        if self.attached:
            d[mask] = 0.300 - (self.piece_z)
        return d

    def obs(self):
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = [0, 0, 0.3]
        return {
            "cameras": [
                {"name": "top", "depth": self.depth(), "intrinsics": K.tolist(), "pose": T.tolist()}
            ]
        }

    def ee(self, **kw):
        return {
            "position": dict(zip("xyz", map(float, self.hand), strict=True)),
            "rotation": {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0},
        }

    def go(self, pose, **kw):
        self.calls.append(("go", pose["position"]["z"]))
        if self.refuse:
            return {"outcome": "unreachable", "reached": False}
        self.hand = np.array([pose["position"][k] for k in "xyz"])
        if self.attached:
            self.piece_z = self.hand[2]
        return {"outcome": "reached"}

    def contact(self, pose, **kw):
        self.calls.append(("contact", pose["position"]["z"]))
        self.hand = np.array([pose["position"]["x"], pose["position"]["y"], 0.002])
        return {"outcome": "contact", "reached": False}

    def suction(self, on, **kw):
        self.calls.append(("suction", on))
        if not on:
            self.vacuum = self.attached = False
            self.piece_z = 0.002
            return {"on": False}
        if self.vacuum and self.drop:
            self.attached = False
            return {"on": True, "sealed": False}
        self.vacuum = True
        self.attached = self.seal
        return {"on": True, **({"sealed": self.seal} if self.sensor else {})}

    def tools(self):
        return {
            "robot.get_observation": lambda **kw: self.obs(),
            "robot.get_ee_pose": self.ee,
            "robot.go_to_pose": self.go,
            "robot.go_to_pose_cartesian": self.go,
            "robot.move_cartesian_until_contact": self.contact,
            "robot.set_suction": self.suction,
        }


def test_picks_and_checks_the_seal_from_the_rim(pick):
    st = Station()
    _, mask = scene()
    out = pick.run(FakeContext(st.tools()), target_mask=mask)
    assert out["route"] == "picked" and out["sealed"]
    assert out["rim_rise_m"] == pytest.approx(0.005, abs=1e-4)
    assert out["cup_point"][2] == pytest.approx(0.002, abs=1e-6)
    assert ("suction", True) in st.calls and ("suction", False) not in st.calls
    assert st.hand[2] == pytest.approx(0.002 + 0.04)


def test_no_seal_releases_and_backs_off(pick):
    st = Station(seal=False)
    out = pick.run(FakeContext(st.tools()), target_mask=scene()[1])
    assert out["route"] == "no_seal" and not out["sealed"] and "rose 0.0 mm" in out["reason"]
    assert st.calls[-2] == ("suction", False)


def test_vacuum_sensor_reading_wins(pick):
    st = Station(seal=False, sensor=True)
    out = pick.run(FakeContext(st.tools()), target_mask=scene()[1])
    assert out["route"] == "no_seal"


def test_transport_and_place(pick):
    st = Station()
    out = pick.run(FakeContext(st.tools()), target_mask=scene()[1], place_point=[0.1, 0.05, 0.002])
    assert out["route"] == "picked" and "placed" in out["reason"]
    assert st.calls[-2] == ("suction", False)
    assert st.hand[:2] == pytest.approx([0.1, 0.05])


def test_dropped_in_transport(pick):
    st = Station(drop_in_transport=True)
    out = pick.run(FakeContext(st.tools()), target_mask=scene()[1], place_point=[0.1, 0.05, 0.002])
    assert out["route"] == "dropped_in_transport"


def test_no_valid_point_and_unreachable(pick):
    st = Station()
    out = pick.run(FakeContext(st.tools()), target_mask=scene(half_px=4)[1])
    assert out["route"] == "no_valid_cup_point" and not st.calls
    out = pick.run(FakeContext(st.tools()), target_mask=np.zeros((5, 5), bool))
    assert out["route"] == "no_valid_cup_point" and "does not match" in out["reason"]
    st = Station(refuse=True)
    out = pick.run(FakeContext(st.tools()), target_mask=scene()[1])
    assert out["route"] == "unreachable" and ("suction", True) not in st.calls


def test_cup_offset_moves_the_tool_not_the_cup(pick):
    st = Station()
    out = pick.run(FakeContext(st.tools()), target_mask=scene()[1], cup_offset_m=[0.02, 0.0, 0.0])
    # tool rotation 180 deg about x keeps tool +x = world +x: the tool centre sits 20 mm to -x of the cup
    assert st.hand[0] == pytest.approx(out["cup_point"][0] - 0.02, abs=1e-5)


def test_example_declares_every_route(bundle):
    ex = json.loads((Path(bundle.bundle_dir) / "examples" / "canonical_subgraph.json").read_text())
    assert set(ex["exit"]["success_values"]) | {ex["on_error"]} == {
        "picked",
        "no_seal",
        "dropped_in_transport",
        "no_valid_cup_point",
        "unreachable",
    }
