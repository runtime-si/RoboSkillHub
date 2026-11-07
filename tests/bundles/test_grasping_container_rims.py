"""grasping-container-rims: FakeContext tests of the rim geometry, the jaw turn, the gap check and the routing."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "grasping-container-rims"
SPAN = 0.094
# A 100 x 100 x 40 mm open box on a table at z = 0, 3 mm walls, centre (0.40, 0.10).
BOX = {
    "center": {"x": 0.40, "y": 0.10, "z": 0.02},
    "extent": {"x": 0.05, "y": 0.05, "z": 0.02},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}
WALL = 0.003


def _mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def top_down(heading_deg: float) -> dict[str, float]:
    """Tool +z down, turned ``heading_deg`` about the vertical."""
    h = math.radians(heading_deg) / 2.0
    q = _mul((math.cos(h), 0.0, 0.0, math.sin(h)), (0.0, 1.0, 0.0, 0.0))
    return dict(zip("wxyz", q, strict=True))


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["grasp_rim"].module


class Arm:
    """A top-down hand. Moves land where commanded unless ``refuse`` (a predicate
    on tool and target) says otherwise; ``free_roll`` keeps the hand's heading on
    pose moves (the last joint turns it, the hand's yaw = -q6); ``between``
    gives the jaw gap a close ends at, from the tool centre."""

    def __init__(
        self,
        start=(0.30, 0.10, 0.25),
        refuse=None,
        free_roll=False,
        between=None,
        stop_descent_at=None,
        stall_lift=False,
        slip=False,
        ik_ok=None,
    ):
        self.p = np.array(start, dtype=float)
        self.yaw = 0.0
        self.q = np.zeros(6)
        self.refuse = refuse or (lambda tool, target: None)
        self.free_roll = free_roll
        self.between = between or (lambda p: WALL - 0.0005)
        self.stop_descent_at = stop_descent_at
        self.stall_lift = stall_lift
        self.slip = slip
        self.ik_ok = ik_ok or (lambda p, rot: True)
        self.gap = SPAN
        self.moves = []
        self.closes = 0

    def rot(self):
        return top_down(self.yaw)

    def _move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy(), pose["rotation"]))
            why = self.refuse(tool, target)
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            if self.stop_descent_at is not None and target[2] < self.stop_descent_at < self.p[2]:
                target = target.copy()
                target[2] = self.stop_descent_at
                self.p = target
                return {"reached": False, "outcome": "contact"}
            if self.stall_lift and self.gap < SPAN and target[2] > self.p[2] + 1e-4:
                return {"reached": False, "outcome": "timeout", "reason": "stalled"}
            if self.slip and self.gap < SPAN and target[2] > self.p[2] + 1e-4:
                self.gap = 0.0
            self.p = target
            if not self.free_roll:
                r = pose["rotation"]
                m00 = 1 - 2 * (r["y"] ** 2 + r["z"] ** 2)
                m10 = 2 * (r["x"] * r["y"] + r["w"] * r["z"])
                self.yaw = math.degrees(math.atan2(m10, m00))
                self.q[-1] = -math.radians(self.yaw)
            return {"reached": True, "outcome": "reached"}

        return go

    def joints(self, joint_config, **kwargs):
        q = np.asarray(joint_config["positions"], dtype=float)
        self.moves.append(("robot.move_to_joints", q.copy(), None))
        self.yaw -= math.degrees(q[-1] - self.q[-1])
        self.q = q
        return {"reached": True, "outcome": "reached"}

    def close(self, **kwargs):
        self.closes += 1
        self.gap = max(0.0, float(self.between(self.p)))
        return {"position": self.gap / SPAN}

    def open(self, **kwargs):
        self.gap = SPAN
        return {"position": 1.0}

    def ik(self, pose, **kwargs):
        p = np.array([pose["position"][k] for k in "xyz"])
        ok = self.ik_ok(p, pose["rotation"])
        return {"success": ok, "within_servo_tolerance": ok}

    def ctx(self, **overrides):
        return FakeContext(
            {
                **{
                    "robot.get_ee_pose": lambda **kw: {
                        "pose": {
                            "position": dict(zip("xyz", map(float, self.p), strict=True)),
                            "rotation": self.rot(),
                        }
                    },
                    "robot.get_observation": lambda **kw: {
                        "arms": [{"joint_state": {"positions": self.q.tolist()}}]
                    },
                    "robot.get_gripper": lambda **kw: {"position": self.gap / SPAN},
                    "robot.describe_gripper": {
                        "span_m": SPAN,
                        "width_fit": {"at_open_m": SPAN, "at_closed_m": 0.0},
                        "finger": {"stated": True, "reach_m": 0.0102},
                    },
                    "robot.grasp_frame": lambda close_heading_deg=0.0, **kw: {
                        "rotation": top_down(close_heading_deg),
                        "rotation_flipped": top_down(close_heading_deg + 180.0),
                    },
                    "robot.solve_ik": self.ik,
                    "robot.go_to_pose": self._move("robot.go_to_pose"),
                    "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
                    "robot.move_to_joints": self.joints,
                    "robot.open_gripper": self.open,
                    "robot.close_gripper": self.close,
                },
                **{name.replace("__", "."): fn for name, fn in overrides.items()},
            }
        )


def test_grasps_the_nearest_wall(script):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(ctx, container_obb=BOX, wall_thickness=WALL)
    assert out["route"] == "grasped", out["detail"]
    assert out["side"] == "-x"  # the hand starts at x = 0.30, the box spans 0.35..0.45
    assert abs(out["close_heading_deg"]) == pytest.approx(180.0)
    point = [0.40 - 0.05 + WALL / 2, 0.10, 0.04 - 0.012]
    assert [out["grasp_position"][k] for k in "xyz"] == pytest.approx(point)
    names = [m[0] for m in arm.moves]
    assert names == [
        "robot.go_to_pose",  # hover (the hand starts above it)
        "robot.go_to_pose_cartesian",  # straight down
        "robot.go_to_pose_cartesian",  # lift
    ]
    hover, down, lift = (m[1] for m in arm.moves)
    np.testing.assert_allclose(hover, [point[0], point[1], point[2] + 0.06], atol=1e-9)
    np.testing.assert_allclose(down, point, atol=1e-9)
    np.testing.assert_allclose(lift, [point[0], point[1], point[2] + 0.03], atol=1e-9)
    assert out["grip_width"] == pytest.approx(0.0025)
    assert out["held_width"] == pytest.approx(0.0025)
    assert out["lifted"] == pytest.approx(0.03)
    assert out["attempts"] == 1
    calls = [c.tool for c in ctx.calls]
    assert calls.index("robot.open_gripper") < calls.index("robot.go_to_pose")
    assert calls.count("robot.close_gripper") == 1
    assert "robot.move_to_joints" not in calls
    assert out["wrist_turn_deg"] == 0.0


def test_free_wrist_roll_is_finished_on_the_last_joint(script):
    # The +y wall wants the jaws along y; the pose moves leave them along x.
    arm = Arm(free_roll=True)
    out = script.run(
        arm.ctx(), container_obb=BOX, wall_thickness=WALL, side="+y", try_other_sides=False
    )
    assert out["route"] == "grasped", out["detail"]
    turns = [m for m in arm.moves if m[0] == "robot.move_to_joints"]
    assert len(turns) == 1
    assert abs(out["jaw_error_deg"]) < 1e-6
    assert abs(out["wrist_turn_deg"]) == pytest.approx(90.0)
    assert abs(((arm.yaw - 90.0) + 90.0) % 180.0 - 90.0) < 1e-6
    names = [m[0] for m in arm.moves]
    assert names[:3] == [
        "robot.go_to_pose",
        "robot.move_to_joints",
        "robot.go_to_pose",
    ]  # hover, turn, re-centre


def test_free_roll_without_joint_moves_is_unreachable(script):
    arm = Arm(free_roll=True)

    def refuse(**kw):
        raise ToolError("robot.move_to_joints", "joint moves are refused on this station")

    out = script.run(
        arm.ctx(robot__move_to_joints=refuse),
        container_obb=BOX,
        wall_thickness=WALL,
        side="+y",
        try_other_sides=False,
    )
    assert out["route"] == "unreachable"
    assert "off the wall's normal" in out["detail"]
    assert arm.closes == 0


def test_an_empty_close_tries_the_next_wall(script):
    # Only the +y wall (y = 0.15) has anything between the pads.
    arm = Arm(between=lambda p: 0.0025 if p[1] > 0.14 else 0.0)
    ctx = arm.ctx()
    out = script.run(ctx, container_obb=BOX, wall_thickness=WALL)
    assert out["route"] == "grasped", out["detail"]
    assert out["side"] == "+y"
    assert out["attempts"] == 3  # -x, then the two equally near +-y walls, +y second
    assert arm.closes == 2 or arm.closes == 3
    calls = [c.tool for c in ctx.calls]
    assert calls.count("robot.open_gripper") == arm.closes  # the start, and after each miss


def test_every_close_empty_is_missed(script):
    arm = Arm(between=lambda p: 0.0)
    out = script.run(arm.ctx(), container_obb=BOX, wall_thickness=WALL)
    assert out["route"] == "missed"
    assert out["attempts"] == 4
    assert out["detail"].count("closed on nothing") == 4
    assert arm.gap == SPAN


def test_too_wide_a_close_is_missed(script):
    arm = Arm(between=lambda p: 0.02)
    out = script.run(
        arm.ctx(), container_obb=BOX, wall_thickness=WALL, side="-x", try_other_sides=False
    )
    assert out["route"] == "missed"
    assert "20.0 mm wide" in out["detail"]


def test_unknown_wall_uses_max_wall(script):
    arm = Arm(between=lambda p: 0.009)
    out = script.run(arm.ctx(), container_obb=BOX, try_other_sides=False)
    assert out["route"] == "grasped", out["detail"]
    assert out["wall_thickness"] == -1.0
    assert out["grasp_position"]["x"] == pytest.approx(0.35 + 0.0015)  # ASSUMED_WALL / 2 in


def test_ik_preview_refusing_every_wall_is_unreachable(script):
    arm = Arm(ik_ok=lambda p, rot: False)
    ctx = arm.ctx()
    out = script.run(ctx, container_obb=BOX, wall_thickness=WALL)
    assert out["route"] == "unreachable"
    assert out["detail"].count("out of reach") == 4
    assert arm.moves == []
    assert arm.closes == 0


def test_ik_preview_picks_the_reachable_wrist_solution(script):
    seen = []

    def only_the_second(p, rot):
        seen.append(rot)
        return len(seen) == 2

    arm = Arm(ik_ok=only_the_second)
    out = script.run(
        arm.ctx(), container_obb=BOX, wall_thickness=WALL, side="-x", try_other_sides=False
    )
    assert out["route"] == "grasped", out["detail"]
    assert seen[0] != seen[1]
    assert arm.moves[0][2] == seen[1]


def test_a_raised_hover_refusal_is_routed(script):
    def refuse(tool, target):
        if tool == "robot.go_to_pose":
            raise ToolError(tool, "Target pose is unreachable")
        return None

    arm = Arm(refuse=refuse)
    out = script.run(arm.ctx(), container_obb=BOX, wall_thickness=WALL)
    assert out["route"] == "unreachable"
    assert out["detail"].count("hover") == 4
    assert arm.closes == 0


def test_a_finger_on_the_rim_is_missed(script):
    # The descent stops 22 mm over the grasp height: a fingertip on the rim top.
    arm = Arm(stop_descent_at=0.05)
    out = script.run(
        arm.ctx(), container_obb=BOX, wall_thickness=WALL, side="+x", try_other_sides=False
    )
    assert out["route"] == "missed"
    assert "stopped 22 mm high" in out["detail"]
    assert arm.closes == 0
    assert arm.p[2] == pytest.approx(0.028 + 0.06)  # back up at the hover height


def test_a_wall_that_slips_out_during_the_lift_is_missed(script):
    arm = Arm(slip=True)
    out = script.run(
        arm.ctx(), container_obb=BOX, wall_thickness=WALL, side="-x", try_other_sides=False
    )
    assert out["route"] == "missed"
    assert "slipped out during the lift" in out["detail"]
    assert out["grip_width"] == pytest.approx(0.0025)
    assert out["held_width"] == 0.0


def test_a_stalled_lift_on_a_fixed_container_still_holds(script):
    arm = Arm(stall_lift=True)
    out = script.run(
        arm.ctx(), container_obb=BOX, wall_thickness=WALL, side="-x", try_other_sides=False
    )
    assert out["route"] == "grasped", out["detail"]
    assert out["lifted"] == pytest.approx(0.0)
    assert "heavy, fixed or caught" in out["detail"]
    assert out["held_width"] == pytest.approx(0.0025)


def test_no_lift(script):
    arm = Arm()
    out = script.run(arm.ctx(), container_obb=BOX, wall_thickness=WALL, lift_height=0.0)
    assert out["route"] == "grasped"
    assert [m[0] for m in arm.moves] == ["robot.go_to_pose", "robot.go_to_pose_cartesian"]


def test_rises_first_from_below_the_hover(script):
    arm = Arm(start=(0.30, 0.10, 0.05))
    out = script.run(
        arm.ctx(), container_obb=BOX, wall_thickness=WALL, side="-x", try_other_sides=False
    )
    assert out["route"] == "grasped"
    tool, target, _ = arm.moves[0]
    assert tool == "robot.go_to_pose_cartesian"
    np.testing.assert_allclose(target, [0.30, 0.10, 0.088], atol=1e-9)


def test_a_cup_too_narrow_for_the_open_hand_is_unreachable(script):
    cup = {**BOX, "extent": {"x": 0.02, "y": 0.02, "z": 0.02}}
    arm = Arm()
    out = script.run(arm.ctx(), container_obb=cup, wall_thickness=WALL)
    assert out["route"] == "unreachable"
    assert out["detail"].count("opposite wall") == 4
    assert arm.moves == []


def test_a_too_shallow_container_is_unreachable(script):
    tray = {
        **BOX,
        "center": {"x": 0.40, "y": 0.10, "z": 0.006},
        "extent": {"x": 0.05, "y": 0.05, "z": 0.006},
    }
    arm = Arm()
    out = script.run(arm.ctx(), container_obb=tray, wall_thickness=WALL)
    assert out["route"] == "unreachable"
    assert "too shallow" in out["detail"]
    assert arm.moves == []


def test_a_turned_box_turns_the_jaws(script):
    h = math.radians(30.0) / 2.0
    turned = {**BOX, "orientation": {"w": math.cos(h), "x": 0.0, "y": 0.0, "z": math.sin(h)}}
    arm = Arm()
    out = script.run(
        arm.ctx(), container_obb=turned, wall_thickness=WALL, side="+x", try_other_sides=False
    )
    assert out["route"] == "grasped"
    assert out["close_heading_deg"] == pytest.approx(30.0)
    d = 0.05 - WALL / 2
    assert out["grasp_position"]["x"] == pytest.approx(0.40 + d * math.cos(math.radians(30)))
    assert out["grasp_position"]["y"] == pytest.approx(0.10 + d * math.sin(math.radians(30)))


def test_an_unknown_side_raises(script):
    with pytest.raises(ValueError):
        script.run(Arm().ctx(), container_obb=BOX, side="north")


def test_rim_frame_from_a_cloud(script):
    # A square rim 2 mm thick at z 0.038..0.040 around (0.41, 0.09), half 0.045, plus a floor at z 0.002.
    rng = np.random.default_rng(0)
    pts = []
    for _ in range(2000):
        t = rng.uniform(-0.045, 0.045)
        r = rng.uniform(0.043, 0.045)
        side = rng.integers(4)
        x, y = [(r, t), (-r, t), (t, r), (t, -r)][side]
        pts.append([0.41 + x, 0.09 + y, rng.uniform(0.038, 0.040)])
    for _ in range(300):
        pts.append([0.41 + rng.uniform(-0.04, 0.04), 0.09 + rng.uniform(-0.04, 0.04), 0.002])
    frame = script.rim_frame(BOX, {"points": np.array(pts)})
    assert frame["source"] == "cloud"
    assert frame["top"] == pytest.approx(0.0398, abs=5e-4)
    np.testing.assert_allclose(frame["xy"], [0.41, 0.09], atol=2e-3)
    assert frame["axes"]["+x"][1] == pytest.approx(0.044, abs=1.5e-3)
    assert frame["bottom"] == pytest.approx(0.0)  # from the box


def test_rim_frame_ignores_a_tiny_cloud(script):
    frame = script.rim_frame(BOX, [[0.4, 0.1, 0.04]] * 5)
    assert frame["source"] == "box"
    assert frame["top"] == pytest.approx(0.04)


def test_yaw_between_folds_the_half_turn(script):
    assert script.yaw_between(top_down(40.0), top_down(10.0)) == pytest.approx(30.0)
    assert script._wrap_half(script.yaw_between(top_down(190.0), top_down(10.0))) == pytest.approx(
        0.0, abs=1e-9
    )
