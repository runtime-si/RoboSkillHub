"""carrying-over-an-obstacle-with-a-leaning-hand: the lean, the pose search and each way a carry ends."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

from ._bin_scenes import DOWN, obb

BUNDLE = "carrying-over-an-obstacle-with-a-leaning-hand"
SOURCE = obb(0.10, 0.00, 0.45, 0.26, 0.0, 0.154)  # the bin the part came from
TARGET = obb(0.50, 0.00, 0.24, 0.20, 0.0, 0.066)
CLEAR = 0.154 + 0.11
START = (0.10, 0.00, 0.21)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["carry_and_release"].module


class Robot:
    """A hand that goes where it is sent, unless ``arrive`` says where it ends instead."""

    def __init__(self, reach=lambda p, r: True, arrive=None, jaws=None):
        self.at = np.array(START)
        self.rot = DOWN
        self.reach, self.arrive = reach, arrive
        self.jaws = list(jaws or [])
        self.opened = False
        self.moves: list = []

    def ik(self, pose, **_):
        p = pose["position"]
        if not self.reach(np.array([p["x"], p["y"], p["z"]]), pose["rotation"]):
            raise ToolError("robot.solve_ik", "Target pose is unreachable")
        return {"joint_config": {"positions": [0.0] * 6}, "success": True}

    def go(self, pose, **_):
        p = pose["position"]
        want = np.array([p["x"], p["y"], p["z"]])
        self.moves.append(want)
        self.at = np.array(self.arrive(want, self.at)) if self.arrive else want
        self.rot = pose["rotation"]
        return {"success": True}

    def gripper(self, **_):
        return {"position": 0.0 if self.opened else (self.jaws.pop(0) if self.jaws else 0.3)}

    def open(self, **_):
        self.opened = True
        return {}

    def ctx(self, **extra):
        return FakeContext(
            {
                "robot.get_ee_pose": lambda **_: {
                    "pose": {
                        "position": dict(zip("xyz", map(float, self.at), strict=True)),
                        "rotation": self.rot,
                    }
                },
                "robot.get_gripper": self.gripper,
                "robot.grasp_frame": {"rotation": DOWN},
                "robot.solve_ik": self.ik,
                "robot.go_to_pose": self.go,
                "robot.go_to_pose_cartesian": self.go,
                "robot.open_gripper": self.open,
                **extra,
            }
        )


def _approach(script, quat):
    return script._rot(quat)[:, 2]  # the tool's z: straight down for DOWN


def test_the_lean_tilts_the_fingertips_left_and_forward_of_the_given_heading(script):
    down = script._rot(DOWN)
    assert np.allclose(down[:, 2], [0, 0, -1])
    s, c = math.sin(math.radians(55)), math.cos(math.radians(55))
    left = script.lean_rotation(down, 0.0, 0.0, 55.0, 90.0)  # forward is +y: left is -x
    assert np.allclose(left[:, 2], [-s, 0, -c])
    forward = script.lean_rotation(down, 0.0, 20.0, 0.0, 90.0)
    assert np.allclose(forward[:, 2], [0, math.sin(math.radians(20)), -math.cos(math.radians(20))])
    other = script.lean_rotation(down, 0.0, 0.0, 55.0, 0.0)  # forward is +x: left is +y
    assert np.allclose(other[:, 2], [0, s, -c])
    turned = script.lean_rotation(down, 90.0, 0.0, 55.0, 90.0)  # the jaws turn, the lean does not
    assert np.allclose(turned[:, 2], left[:, 2]) and not np.allclose(turned, left)
    assert np.allclose(script._rot(script._quat(left)), left)


def test_drop_points_start_in_the_middle_and_keep_off_the_walls(script):
    pts = script.drop_points(obb(0.5, 0.1, 0.24, 0.20, 30.0, 0.066))
    assert len(pts) == 16
    dist = [round(float(np.linalg.norm(p - [0.5, 0.1])), 9) for p in pts]
    assert dist == sorted(dist) and dist[0] < 0.03
    box = obb(0.5, 0.1, 0.24, 0.20, 30.0, 0.066)
    assert all(script.inside_by(p, box) >= 0.05 - 1e-9 for p in pts)
    assert script.inside_by([0.5, 0.1], box) == pytest.approx(0.10)
    assert script.inside_by([0.9, 0.1], box) < 0


def test_it_rises_leaning_crosses_at_height_lowers_lets_go_and_retreats(script):
    robot = Robot()
    ctx = robot.ctx()
    out = script.run(ctx, TARGET, over_obbs=[SOURCE])
    assert out["route"] == "placed", out["detail"]
    assert out["clear_z"] == pytest.approx(CLEAR) and out["carried_z"] == pytest.approx(
        CLEAR + 0.135
    )
    rise, across, low, back = robot.moves
    assert np.allclose(rise, [START[0], START[1], CLEAR + 0.135])  # up where it stands
    assert (
        across[2] == pytest.approx(CLEAR + 0.135) and np.linalg.norm(across[:2] - [0.5, 0.0]) < 0.03
    )
    assert low[2] == pytest.approx(0.066 + 0.12) and np.allclose(low[:2], across[:2])
    assert np.allclose(back, across)
    # the rise is made with the first lean, the fingertips 55 deg to the left of +y
    first = ctx.calls_to("robot.go_to_pose")[0].kwargs["pose"]["rotation"]
    assert _approach(script, first)[0] == pytest.approx(-math.sin(math.radians(55)), abs=0.05)
    assert robot.opened and out["inside_m"] >= 0.05
    assert out["release_position"]["z"] == pytest.approx(0.066 + 0.12)
    assert out["fell_short_at"] == [] and not out["clearance_checked"]


def test_the_highest_height_the_arm_reaches_is_used(script):
    robot = Robot(reach=lambda p, r: p[2] <= 0.35)
    out = script.run(robot.ctx(), TARGET, over_obbs=[SOURCE])
    assert out["route"] == "placed"
    assert out["carried_z"] == pytest.approx(CLEAR + 0.135 - 0.06)  # two 3 cm steps down
    assert out["carried_z"] <= 0.35


def test_another_hand_may_let_go_than_the_one_that_rose(script):
    # over the source only the first lean reaches up; over the target only the forward-only lean
    def reach(p, quat):
        lean_x = _approach(script, quat)[0]
        return lean_x < -0.5 if p[0] < 0.3 else abs(lean_x) < 0.1

    robot = Robot(reach=reach)
    ctx = robot.ctx()
    out = script.run(ctx, TARGET, over_obbs=[SOURCE], max_ik_queries=400)
    assert out["route"] == "placed", out["detail"]
    rise, across = (c.kwargs["pose"]["rotation"] for c in ctx.calls_to("robot.go_to_pose")[:2])
    assert _approach(script, rise)[0] < -0.5 and abs(_approach(script, across)[0]) < 0.1


def test_a_carry_that_ends_short_but_over_the_container_lets_go_there(script):
    def arrive(want, was):  # every move towards the target stops 45 mm short of it
        return want - [0.045, 0, 0] if want[0] > 0.3 and want[2] > 0.25 else want

    robot = Robot(arrive=arrive)
    out = script.run(robot.ctx(), TARGET, over_obbs=[SOURCE])
    assert out["route"] == "placed", out["detail"]
    assert len(out["fell_short_at"]) == 1 and "short, over the container" in out["detail"]
    assert out["release_position"]["x"] == pytest.approx(out["drop_point"]["x"] - 0.045, abs=1e-6)
    assert out["inside_m"] >= 0.03 and robot.opened


def test_short_and_not_over_the_container_it_goes_back_still_holding(script):
    def arrive(want, was):  # the arm cannot get past x = 0.33: 5 cm outside the near wall
        return [min(want[0], 0.33), want[1], want[2]]

    robot = Robot(arrive=arrive)
    out = script.run(robot.ctx(), TARGET, over_obbs=[SOURCE])
    assert out["route"] == "returned", out["detail"]
    assert not robot.opened and len(out["fell_short_at"]) == 3 == out["options"]
    assert np.allclose(robot.at, [START[0], START[1], CLEAR + 0.135])  # back where it rose
    assert out["release_position"] is None


def test_a_rise_that_ends_low_does_not_cross(script):
    robot = Robot(arrive=lambda want, was: [want[0], want[1], min(want[2], 0.22)])
    out = script.run(robot.ctx(), TARGET, over_obbs=[SOURCE])
    assert out["route"] == "rise_short" and "0.220 m" in out["detail"]
    assert len(robot.moves) == 1 and not robot.opened


def test_sinking_on_the_way_across_stops_it(script):
    def arrive(want, was):
        return [0.3, want[1], 0.20] if want[0] > 0.3 else want

    robot = Robot(arrive=arrive)
    out = script.run(robot.ctx(), TARGET, over_obbs=[SOURCE])
    assert out["route"] == "blocked" and not robot.opened and len(robot.moves) == 2


@pytest.mark.parametrize(
    ("jaws", "moves", "where"),
    [
        ([0.02], 0, "before anything moved"),
        ([0.3, 0.03], 1, "after the rise"),
        ([0.3, 0.3, 0.0], 2, "across"),
    ],
)
def test_an_empty_hand_is_lost_wherever_it_is_noticed(script, jaws, moves, where):
    robot = Robot(jaws=jaws)
    out = script.run(robot.ctx(), TARGET, over_obbs=[SOURCE])
    assert out["route"] == "lost" and where in out["detail"]
    assert len(robot.moves) == moves and not robot.opened


def test_no_pose_is_unreachable_within_the_ik_budget_and_nothing_moves(script):
    robot = Robot(reach=lambda p, r: False)
    ctx = robot.ctx()
    out = script.run(ctx, TARGET, over_obbs=[SOURCE], max_ik_queries=25)
    assert out["route"] == "unreachable" and "to rise with" in out["detail"]
    assert out["ik_queries"] == 25 == ctx.call_count("robot.solve_ik")
    assert "25 IK previews" in out["detail"] and robot.moves == [] and not robot.opened
    # the hand can rise but no drop point is reachable
    robot = Robot(reach=lambda p, r: p[0] < 0.3)
    out = script.run(robot.ctx(), TARGET, over_obbs=[SOURCE], max_ik_queries=2000)
    assert out["route"] == "unreachable" and "over the container" in out["detail"]
    assert robot.moves == []


def test_pulling_back_before_the_rise(script):
    # where it stands the arm cannot rise (something overhead); 12 cm back it can
    robot = Robot(reach=lambda p, r: not (abs(p[0] - 0.10) < 0.01 and p[1] > -0.05))
    out = script.run(
        robot.ctx(), TARGET, over_obbs=[SOURCE], pull_back_m=[0.06, 0.12], max_ik_queries=400
    )
    assert out["route"] == "placed", out["detail"]
    back, rise = robot.moves[:2]
    assert np.allclose(back, [0.10, -0.06, START[2]])  # straight back at the height it is at
    assert np.allclose(rise[:2], [0.10, -0.06]) and rise[2] >= CLEAR


def test_avoided_points_are_skipped_and_the_spread_index_rotates(script):
    first = script.run(Robot().ctx(), TARGET, over_obbs=[SOURCE])["drop_point"]
    second = script.run(Robot().ctx(), TARGET, over_obbs=[SOURCE], spread_index=1)["drop_point"]
    assert (first["x"], first["y"]) != (second["x"], second["y"])
    avoided = script.run(Robot().ctx(), TARGET, over_obbs=[SOURCE], avoid_points=[first])
    assert (avoided["drop_point"]["x"], avoided["drop_point"]["y"]) == (second["x"], second["y"])


def test_with_an_arm_model_the_search_is_local_and_the_connector_ik_is_never_asked(script):
    asked = []

    def local(model, pose, q_now, obstacles, margin_m, bounds):
        asked.append(pose["position"]["z"])
        ok = pose["position"]["z"] <= 0.37  # the cabinet over the counter
        return {"reachable": ok, "q": [0.1] * 6 if ok else [], "reason": "", "clearance_m": 0.05}

    robot = Robot()
    obs = {"arms": [{"joint_state": {"positions": [0.0] * 6}}], "cameras": []}
    ctx = robot.ctx(**{"arm-clearance.reachable": local, "robot.get_observation": obs})
    out = script.run(
        ctx,
        TARGET,
        over_obbs=[SOURCE],
        arm_model={"chain": []},
        obstacles=[{"name": "cabinet", "center": [0, 0, 0.6], "half": [1, 1, 0.1]}],
    )
    assert out["route"] == "placed", out["detail"]
    assert ctx.call_count("robot.solve_ik") == 0 and out["ik_queries"] == 0
    assert out["clearance_checked"] and out["carried_z"] <= 0.37
    assert len(asked) > 10  # the rise, the drops and every step of the carry
    blind = Robot().ctx(**{"arm-clearance.reachable": local, "robot.get_observation": {"arms": []}})
    assert script.run(blind, TARGET, arm_model={"chain": []})["route"] == "unreachable"


def test_the_real_arm_clearance_tool_takes_the_arguments_this_skill_sends(script, skills_registry):
    tool = skills_registry.get("arm-clearance").tools_module
    chain = [{"name": "base", "parent": None}] + [
        {
            "name": f"l{i}",
            "parent": "base" if i == 1 else f"l{i - 1}",
            "position": p,
            "joint": i - 1,
            "axis": a,
        }
        for i, (p, a) in enumerate(
            [
                ([0, 0, 0.3], [0, 0, 1]),
                ([0, 0, 0.05], [0, 1, 0]),
                ([0.3, 0, 0], [0, 1, 0]),
                ([0.25, 0, 0], [1, 0, 0]),
                ([0.05, 0, 0], [0, 1, 0]),
                ([0.05, 0, 0], [1, 0, 0]),
            ],
            start=1,
        )
    ]
    model = {
        "chain": chain,
        "tcp": {"body": "l6", "position": [0.08, 0, 0]},
        "limits": [[-3, 3]] * 6,
    }
    obs = {"arms": [{"joint_state": {"positions": [0.0, 0.3, 0.6, 0.0, 0.6, 0.0]}}]}
    robot = Robot()
    ctx = robot.ctx(**{"arm-clearance.reachable": tool.reachable, "robot.get_observation": obs})
    out = script.run(ctx, TARGET, over_obbs=[SOURCE], arm_model=model, obstacles=[])
    assert out["route"] in ("placed", "unreachable") and out["ik_queries"] == 0
    assert ctx.call_count("arm-clearance.reachable") > 0 and ctx.call_count("robot.solve_ik") == 0
