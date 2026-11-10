"""disassembling-a-stack-with-hold-and-twist: the support press, the hand-off to the twist-lever skill (stubbed),
the drift check, route mapping and refusals, with FakeContext."""

from __future__ import annotations

import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

SKILL = "disassembling-a-stack-with-hold-and-twist"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
SEAT = {"x": 0.57, "y": 0.0, "z": 0.756}
SUPPORT = {"x": 0.57, "y": -0.025, "z": 0.756}


@pytest.fixture(scope="module")
def ht(skills_registry):
    bundle = skills_registry.get(SKILL)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["hold_and_twist"].module


class Arms:
    """Arm 1 lands where commanded (moves recorded); `drift` is added to it once the twist has run."""

    def __init__(self, refuse=None, drift=(0.0, 0.0, 0.0)):
        self.p = {0: [0.5, 0.3, 0.99], 1: [0.5, -0.3, 0.95]}
        self.moves, self.refuse, self.drift, self.twisted = [], refuse, drift, False

    def ee(self, arm_id=0, **_):
        p = list(self.p[arm_id])
        if arm_id == 1 and self.twisted:
            p = [a + b for a, b in zip(p, self.drift, strict=True)]
        return {"pose": {"position": dict(zip("xyz", p, strict=True)), "rotation": DOWN}}

    def go(self, pose, arm_id=0, **_):
        self.moves.append((arm_id, dict(pose["position"])))
        why = self.refuse(pose) if self.refuse else None
        if why == "raise":
            raise ToolError("robot.go_to_pose_cartesian", "joint limit")
        if why:
            return {"reached": False, "outcome": "timeout"}
        self.p[arm_id] = [pose["position"][k] for k in "xyz"]
        return {"reached": True, "outcome": "reached"}


def _ctx(arms):
    return FakeContext(
        {
            "robot.get_ee_pose": arms.ee,
            "robot.go_to_pose_cartesian": arms.go,
            "robot.close_gripper": {},
            "robot.open_gripper": {},
            "robot.grasp_frame": {"rotation": DOWN},
        }
    )


@pytest.fixture
def twist(ht, monkeypatch):
    calls = []

    class Stub:
        route = "removed"
        arms = None

        @classmethod
        def run(cls, ctx, **kw):
            calls.append(kw)
            if cls.arms is not None:
                cls.arms.twisted = True
            return {"route": cls.route, "detail": "stub"}

    monkeypatch.setattr(ht, "_twist_module", lambda: Stub)
    Stub.calls = calls
    return Stub


def test_support_presses_then_twist_separates(ht, twist):
    arms = Arms()
    twist.arms = arms
    out = ht.run(
        _ctx(arms),
        seat_position=SEAT,
        support_point=SUPPORT,
        support_mode="press",
        twist_inputs={"system": "duplo"},
    )
    assert out["route"] == "separated" and out["support_used"] and out["support_drift_mm"] == 0.0
    sup = [m for a, m in arms.moves if a == 1]
    assert sup[0]["z"] == pytest.approx(SUPPORT["z"] + ht.TIP_BELOW_TCP + ht.APPROACH)
    assert sup[1]["z"] == pytest.approx(SUPPORT["z"] + ht.TIP_BELOW_TCP - ht.PRESS_DEPTH)
    assert sup[-1]["z"] > sup[1]["z"]  # risen after
    assert twist.calls[0]["arm_id"] == 0 and twist.calls[0]["system"] == "duplo"


def test_support_drift_is_base_moved(ht, twist):
    arms = Arms(drift=(0.0, 0.0, 0.006))
    twist.arms = arms
    out = ht.run(_ctx(arms), seat_position=SEAT, support_point=SUPPORT)
    assert out["route"] == "base_moved" and out["support_drift_mm"] == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("twist_route", "route"),
    [
        ("stuck", "clutch_too_strong"),
        ("neighbour_lifted", "base_moved"),
        ("missed", "missed"),
        ("unverified", "unverified"),
    ],
)
def test_twist_routes_map(ht, twist, twist_route, route):
    twist.route, twist.arms = twist_route, Arms()
    out = ht.run(_ctx(twist.arms), seat_position=SEAT, support_point=SUPPORT)
    assert out["route"] == route and out["twist_route"] == twist_route
    twist.route = "removed"


@pytest.mark.parametrize("refuse", ["timeout", "raise"])
def test_unreachable_support_stops_before_the_twist(ht, twist, refuse):
    arms = Arms(refuse=lambda pose: refuse)
    out = ht.run(_ctx(arms), seat_position=SEAT, support_point=SUPPORT)
    assert out["route"] == "support_unreachable" and not twist.calls


def test_without_support_only_twists(ht, twist):
    arms = Arms()
    out = ht.run(_ctx(arms), seat_position=SEAT, support=False)
    assert out["route"] == "separated" and not out["support_used"] and arms.moves == []


def test_tool_centre_floor(ht, twist):
    arms = Arms()
    low = {"x": 0.5, "y": 0.0, "z": 0.0}
    ht.run(_ctx(arms), seat_position=SEAT, support_point=low, support_mode="press")
    assert min(m["z"] for a, m in arms.moves if a == 1) >= ht.MIN_TCP_Z


def test_bad_arguments(ht):
    with pytest.raises(ValueError, match="support_point"):
        ht.run(FakeContext({}), seat_position=SEAT)
    with pytest.raises(ValueError, match="support_mode"):
        ht.run(FakeContext({}), seat_position=SEAT, support_point=SUPPORT, support_mode="hug")
    with pytest.raises(ValueError, match="differ"):
        ht.run(FakeContext({}), seat_position=SEAT, support_point=SUPPORT, support_arm=0)


def test_the_twist_skill_is_found_beside_this_bundle(ht):
    assert callable(ht._twist_module().run)


def test_a_press_leg_stalled_on_the_base_counts_as_support(ht, twist):
    arms = Arms()
    twist.arms = arms
    real_go = arms.go

    def go(pose, arm_id=0, **kw):
        r = real_go(pose, arm_id=arm_id, **kw)
        if arm_id == 1 and len([m for a, m in arms.moves if a == 1]) == 2:
            arms.p[1][2] += ht.PRESS_DEPTH  # stopped on the base, 4 mm above the target
            return {"reached": False, "outcome": "timeout"}
        return r

    arms.go = go
    out = ht.run(_ctx(arms), seat_position=SEAT, support_point=SUPPORT, support_mode="press")
    assert out["route"] == "separated" and out["support_used"]


def test_grip_mode_grips_the_lower_part_and_lets_go(ht, twist):
    arms = Arms()
    twist.arms = arms
    calls = []
    ctx = _ctx(arms)
    ctx_tools = {
        "robot.get_ee_pose": arms.ee,
        "robot.go_to_pose_cartesian": arms.go,
        "robot.close_gripper": lambda **kw: calls.append("close") or {},
        "robot.open_gripper": lambda **kw: calls.append("open") or {},
        "robot.grasp_frame": lambda **kw: (
            calls.append(("frame", kw.get("close_heading_deg"))) or {"rotation": DOWN}
        ),
    }
    ctx = FakeContext(ctx_tools)
    out = ht.run(ctx, seat_position=SEAT, support_point=SUPPORT, seat_yaw_deg=90.0)
    assert out["route"] == "separated" and out["support_used"]
    assert calls[0] == ("frame", 180.0) and calls[1:] == ["open", "close", "open"]
    sup = [m for a, m in arms.moves if a == 1]
    assert sup[1]["z"] == pytest.approx(SUPPORT["z"] + ht.TIP_BELOW_TCP / 2)
