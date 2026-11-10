"""supporting-an-overhang-with-the-second-arm with FakeContext and pressing-a-brick-onto-studs stubbed."""

from __future__ import annotations

import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

SKILL = "supporting-an-overhang-with-the-second-arm"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
SEAT = {"x": 0.5, "y": 0.0, "z": 0.80}
SUP = {"x": 0.5, "y": 0.05, "z": 0.78}


@pytest.fixture(scope="module")
def sp(skills_registry):
    bundle = skills_registry.get(SKILL)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["support_and_press"].module


class Arms:
    def __init__(self, refuse=None, drift=(0.0, 0.0, 0.0)):
        self.p = {0: [0.5, 0.3, 0.99], 1: [0.5, -0.3, 0.95]}
        self.moves, self.refuse, self.drift, self.pressed = [], refuse, drift, False

    def ee(self, arm_id=0, **_):
        p = list(self.p[arm_id])
        if arm_id == 1 and self.pressed:
            p = [a + b for a, b in zip(p, self.drift, strict=True)]
        return {"pose": {"position": dict(zip("xyz", p, strict=True)), "rotation": DOWN}}

    def go(self, pose, arm_id=0, **_):
        self.moves.append((arm_id, dict(pose["position"]), pose["rotation"]))
        why = self.refuse(pose) if self.refuse else None
        if why == "raise":
            raise ToolError("robot.go_to_pose_cartesian", "joint limit")
        if why:
            return {"reached": False, "outcome": "unreachable"}
        self.p[arm_id] = [pose["position"][k] for k in "xyz"]
        return {"reached": True, "outcome": "reached"}


def _ctx(arms):
    return FakeContext(
        {
            "robot.get_ee_pose": arms.ee,
            "robot.go_to_pose_cartesian": arms.go,
            "robot.close_gripper": {},
            "robot.grasp_frame": {"rotation": DOWN},
        }
    )


@pytest.fixture
def press(sp, monkeypatch):
    class Stub:
        route, arms, calls = "seated", None, []

        @classmethod
        def run(cls, ctx, **kw):
            cls.calls.append(kw)
            if cls.arms is not None:
                cls.arms.pressed = True
            return {"route": cls.route, "detail": "stub"}

    Stub.calls = []
    monkeypatch.setattr(sp, "_press_module", lambda: Stub)
    return Stub


def test_support_then_press_seats(sp, press):
    arms = Arms()
    press.arms = arms
    out = sp.run(
        _ctx(arms), seat_position=SEAT, support_point=SUP, press_inputs={"strategy": "straight"}
    )
    assert out["route"] == "seated" and out["support_used"]
    sup = [m for a, m, _ in arms.moves if a == 1]
    assert sup[0]["z"] == pytest.approx(SUP["z"] + sp.APPROACH)
    assert sup[1]["z"] == pytest.approx(SUP["z"] - sp.PRELOAD)
    assert sup[2]["z"] == pytest.approx(SUP["z"] - sp.PRELOAD + sp.APPROACH)
    assert press.calls[0] == {
        "seat_position": SEAT,
        "seat_yaw_deg": 0.0,
        "arm_id": 0,
        "system": "duplo",
        "strategy": "straight",
    }


def test_a_support_pose_from_the_side(sp, press):
    arms = Arms()
    side = {"w": 0.7071, "x": 0.0, "y": 0.7071, "z": 0.0}
    sp.run(
        _ctx(arms),
        seat_position=SEAT,
        support_pose={"position": SUP, "rotation": side},
        approach_offset=[0.0, 0.06, 0.0],
    )
    a1 = [(m, r) for a, m, r in arms.moves if a == 1]
    assert a1[0][0]["y"] == pytest.approx(SUP["y"] + 0.06) and a1[0][1] == side
    assert a1[1][0]["y"] == pytest.approx(SUP["y"] - sp.PRELOAD)


def test_apex_support_offset_in_studs(sp, press):
    p = sp.support_point_from_studs({"x": 0.5, "y": 0.0, "z": 0.8}, [2, -1, -1], "lego", 90.0)
    assert p == pytest.approx([0.5 + 0.008, 0.016, 0.8 - 0.0096])
    arms = Arms()
    sp.run(
        _ctx(arms), seat_position=SEAT, support_studs=[0, 0, -1], support_from=SEAT, system="lego"
    )
    assert [m for a, m, _ in arms.moves if a == 1][1]["z"] == pytest.approx(
        0.8 - 0.0096 - sp.PRELOAD
    )


def test_drift_is_structure_moved(sp, press):
    arms = Arms(drift=(0.004, 0.0, 0.0))
    press.arms = arms
    out = sp.run(_ctx(arms), seat_position=SEAT, support_point=SUP)
    assert out["route"] == "structure_moved" and out["support_drift_mm"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "route", ["not_seated", "misaligned", "not_holding", "unverified", "unreachable"]
)
def test_press_routes_pass_through(sp, press, route):
    press.route, press.arms = route, Arms()
    out = sp.run(_ctx(press.arms), seat_position=SEAT, support_point=SUP)
    assert out["route"] == route


@pytest.mark.parametrize("refuse", ["no", "raise"])
def test_unreachable_support_presses_nothing(sp, press, refuse):
    out = sp.run(_ctx(Arms(refuse=lambda pose: refuse)), seat_position=SEAT, support_point=SUP)
    assert out["route"] == "support_unreachable" and press.calls == []


def test_without_support(sp, press):
    arms = Arms()
    out = sp.run(_ctx(arms), seat_position=SEAT, support=False)
    assert out["route"] == "seated" and arms.moves == [] and not out["support_used"]


def test_bad_arguments(sp):
    with pytest.raises(ValueError, match="support_pose"):
        sp.run(FakeContext({}), seat_position=SEAT)
    with pytest.raises(ValueError, match="differ"):
        sp.run(FakeContext({}), seat_position=SEAT, support_point=SUP, support_arm=0)
    with pytest.raises(ValueError, match="system"):
        sp.run(FakeContext({}), seat_position=SEAT, system="mega")


def test_the_press_skill_is_found_beside_this_bundle(sp):
    assert callable(sp._press_module().run)
