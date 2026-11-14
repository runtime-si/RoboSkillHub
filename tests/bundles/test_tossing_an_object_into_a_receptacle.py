"""tossing-an-object-into-a-receptacle: the ballistic planner and the routing against a fake station."""

from __future__ import annotations

import math

import pytest
from gap.testing import FakeContext

BUNDLE = "tossing-an-object-into-a-receptacle"
ROT = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


@pytest.fixture(scope="module")
def toss(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b.canonical_scripts["toss_object"].module


class Station:
    """IK is q = 3 rad/m x position inside a 0.5 m reach; the object lands where `land` says."""

    def __init__(self, held=True, land=(0.75, 0.0, 0.0), reach=0.5):
        self.hand = [0.3, 0.0, 0.25]
        self.held = held
        self.land = land
        self.reach = reach
        self.calls: list[str] = []

    def tools(self):
        def ik(pose, **_):
            p = pose["position"]
            if math.hypot(p["x"], p["y"]) > self.reach:
                return {"success": False}
            return {"joint_positions": [3 * p["x"], 3 * p["y"], 3 * p["z"]]}

        def go(pose, **_):
            self.hand = [pose["position"][k] for k in "xyz"]
            self.calls.append("go")

        def opened(**_):
            self.held = False
            self.calls.append("open")

        return {
            "robot.get_ee_pose": lambda **_: {
                "pose": {"position": dict(zip("xyz", self.hand, strict=True)), "rotation": ROT}
            },
            "robot.solve_ik": ik,
            "robot.go_to_pose_cartesian": go,
            "robot.move_to_joints": lambda **_: self.calls.append("joints"),
            "robot.execute_trajectory": lambda trajectory, **_: self.calls.append(
                f"traj{len(trajectory['waypoints'])}"
            ),
            "robot.open_gripper": opened,
            "robot.describe_gripper": lambda **_: {
                "width_fit": {"at_open_m": 0.08, "at_closed_m": 0.002}
            },
            "robot.get_gripper": lambda **_: {"position": 0.4 if self.held else 0.0},
            "sim.locate": lambda **_: {"obb": {"center": list(self.land)}},
        }


def test_release_speed_hits_the_target(toss):
    d, h, th = 0.35, 0.2, math.radians(40)
    v = toss.release_speed(d, h, th)
    t = toss.flight_time(d, v, th)
    assert h + v * math.sin(th) * t - 0.5 * toss.G * t * t == pytest.approx(0.0, abs=1e-9)


def test_reachable_opening_is_a_drop(toss):
    st = Station(land=(0.4, 0.0, 0.0))
    out = toss.run(
        FakeContext(st.tools()),
        opening_center=[0.4, 0.0, 0.05],
        opening_radius_m=0.1,
        locate_tool="sim.locate",
    )
    assert out["route"] == "landed" and out["mode"] == "drop" and st.calls == ["go", "open"]


def test_station_speed_refuses_and_says_what_it_needs(toss):
    st = Station()
    out = toss.run(FakeContext(st.tools()), opening_center=[0.75, 0.0, 0.05], opening_radius_m=0.1)
    assert out["route"] == "not_feasible" and out["required_joint_speed"] > 0.25
    assert "rad/s" in out["detail"] and "open" not in st.calls


def test_toss_within_limit_lands(toss):
    st = Station()
    out = toss.run(
        FakeContext(st.tools()),
        opening_center=[0.75, 0.0, 0.05],
        opening_radius_m=0.1,
        locate_tool="sim.locate",
        joint_speed_limit=10.0,
    )
    assert out["route"] == "landed" and out["mode"] == "toss"
    assert out["launch_deg"] > 0 and 0.5 < out["release_speed_mps"] < 3.0
    assert st.calls[0] == "joints" and st.calls[1].startswith("traj") and st.calls[2] == "open"


def test_toss_misses(toss):
    st = Station(land=(0.5, 0.2, 0.0))
    out = toss.run(
        FakeContext(st.tools()),
        opening_center=[0.75, 0.0, 0.05],
        opening_radius_m=0.1,
        locate_tool="sim.locate",
        joint_speed_limit=10.0,
    )
    assert out["route"] == "missed"


def test_empty_hand_is_dropped_early(toss):
    st = Station(held=False)
    out = toss.run(FakeContext(st.tools()), opening_center=[0.75, 0.0, 0.05], opening_radius_m=0.1)
    assert out["route"] == "dropped_early" and st.calls == []


def test_no_locate_tool_reports_released(toss):
    st = Station(land=(0.4, 0.0, 0.0))
    out = toss.run(FakeContext(st.tools()), opening_center=[0.4, 0.0, 0.05], opening_radius_m=0.1)
    assert out["route"] == "released"
