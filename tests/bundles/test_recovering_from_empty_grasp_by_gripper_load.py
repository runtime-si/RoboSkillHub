"""recovering-from-empty-grasp-by-gripper-load: the FAR escalation ladder against a fake station."""

from __future__ import annotations

import pytest
from gap.testing import FakeContext

BUNDLE = "recovering-from-empty-grasp-by-gripper-load"
ROT = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


@pytest.fixture(scope="module")
def rec(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b.canonical_scripts["recover_empty_grasp"].module


class Station:
    """A hand that holds the object when it closes within `reach` of it (open 0.08 m, closed 0.002 m)."""

    def __init__(self, obj=(0.3, 0.0, 0.02), reach=0.01, effort=None, refuse_after=None):
        self.obj = list(obj)
        self.reach = reach
        self.effort = effort
        self.refuse_after = refuse_after
        self.hand = [0.3, 0.0, 0.3]
        self.held = False
        self.calls: list[tuple] = []
        self.moves = 0

    def tools(self):
        def move(pose, **_):
            self.moves += 1
            if self.refuse_after is not None and self.moves > self.refuse_after:
                return {"outcome": "unreachable", "reason": "ik"}
            p = pose["position"]
            self.hand = [p["x"], p["y"], p["z"]]
            self.calls.append(("move", tuple(round(v, 3) for v in self.hand)))
            return {"outcome": "reached"}

        def close(**_):
            self.held = all(
                abs(h - o) <= self.reach for h, o in zip(self.hand, self.obj, strict=True)
            )
            self.calls.append(("close", self.held))

        def get_gripper(**_):
            r = {"position": (0.03 - 0.002) / 0.078 if self.held else 0.0}
            if self.effort is not None:
                r["effort"] = self.effort
            return r

        def home(**_):
            self.hand = [0.2, 0.0, 0.3]
            self.calls.append(("home",))

        return {
            "robot.describe_gripper": lambda **_: {
                "width_fit": {"at_open_m": 0.08, "at_closed_m": 0.002}
            },
            "robot.go_to_pose_cartesian": move,
            "robot.close_gripper": close,
            "robot.open_gripper": lambda **_: self.calls.append(("open",)),
            "robot.get_gripper": get_gripper,
            "robot.get_ee_pose": lambda **_: {
                "pose": {"position": dict(zip("xyz", self.hand, strict=True)), "rotation": ROT}
            },
            "robot.go_home": home,
            "sim.locate": lambda **_: {"obb": {"center": list(self.obj)}},
        }


def pose(x=0.3, y=0.0, z=0.02):
    return {"position": {"x": x, "y": y, "z": z}, "rotation": ROT}


def test_first_try_holds(rec):
    st = Station()
    out = rec.run(FakeContext(st.tools()), grasp_pose=pose())
    assert out["route"] == "held" and out["attempts"] == 1 and out["recovery"] == ""
    assert out["widths_m"][0] == pytest.approx(0.03, abs=1e-4)


def test_lift_retry_with_relocation_recovers(rec):
    st = Station()
    out = rec.run(FakeContext(st.tools()), grasp_pose=pose(z=0.06), locate_tool="sim.locate")
    assert out["route"] == "recovered" and out["attempts"] == 2 and out["recovery"] == "lift"
    assert out["grasp_position"] == pytest.approx([0.3, 0.0, 0.02])


def test_ladder_without_relocation_gives_up(rec):
    st = Station()
    out = rec.run(FakeContext(st.tools()), grasp_pose=pose(y=0.06))
    assert out["route"] == "gave_up" and out["attempts"] == 4 and out["recovery"] == "home"
    assert [c[0] for c in st.calls].count("close") == 4 and ("home",) in st.calls
    assert st.calls[-1] == ("open",)


def test_object_moves_until_home_level(rec):
    st = Station()
    ctx = FakeContext(st.tools())
    tools = st.tools()
    seen = {"n": 0}

    def locate(**_):
        seen["n"] += 1
        return {"position": [0.3, 0.0, 0.02] if seen["n"] >= 3 else [0.5, 0.2, 0.02]}

    ctx = FakeContext({**tools, "sim.locate": locate})
    out = rec.run(ctx, grasp_pose=pose(y=0.06), locate_tool="sim.locate")
    assert out["route"] == "recovered" and out["recovery"] == "home" and out["attempts"] == 4


def test_load_threshold_vetoes_a_gap(rec):
    st = Station(effort=0.1)
    out = rec.run(
        FakeContext(st.tools()), grasp_pose=pose(), load_grasp_threshold=0.5, max_total_retries=1
    )
    assert out["route"] == "gave_up" and out["attempts"] == 2


def test_refused_retry_move(rec):
    st = Station(refuse_after=2)
    out = rec.run(FakeContext(st.tools()), grasp_pose=pose(y=0.06))
    assert out["route"] == "move_failed" and out["detail"].startswith("lift retry")
