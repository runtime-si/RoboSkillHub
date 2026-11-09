"""standing-a-lying-domino-by-reorient-and-regrasp: FakeContext tests of the facing search, the stepped pitch, the
roll read-back, the depth check and every exit."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "standing-a-lying-domino-by-reorient-and-regrasp"
ROOT = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
TABLE_Z = 0.75
TILE = {"x": 0.55, "y": 0.30, "z": TABLE_Z + 0.00375}
AT_OPEN = 0.094
# robot.get_gripper's fraction (1.0 = open) when the jaws hold a 24 mm tile.
HOLD = 0.024 / AT_OPEN


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["stand_domino"].module


def heading(deg):
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a), 0.0])


def top_down(close_heading_deg):
    """A top-down hand whose local z is the approach (world down) and local y the closing axis."""
    c = heading(close_heading_deg)
    z = np.array([0.0, 0.0, -1.0])
    return np.stack([np.cross(c, z), c, z], axis=1)


class Arm:
    """A stateful fake arm. Moves land where commanded unless ``refuse(tool, pose)`` says why not; joint moves
    to a solution robot.solve_ik returned land on its pose, rolled ``roll_error_deg`` about the approach (a solver
    that leaves the roll free); a move of the last joint alone turns the hand about the approach by
    ``roll_gain`` times the change."""

    def __init__(
        self,
        script,
        reachable=None,
        refuse=None,
        roll_error_deg=0.0,
        roll_gain=1.0,
        stop_descent_at=None,
        grip=None,
        standing=True,
        cameras=True,
        joints=None,
    ):
        self.s = script
        self.p = np.array([0.45, 0.30, TABLE_Z + 0.25])
        self.r = top_down(90.0)
        self.q = np.array([0.0, 1.6, 1.2, -1.1, 0.0, 0.0])
        self.reachable = reachable or (lambda point, rot: True)
        self.refuse = refuse or (lambda tool, pose: None)
        self.roll_error_deg = roll_error_deg
        self.roll_gain = roll_gain
        self.stop_descent_at = stop_descent_at
        self.grip = grip or (lambda phase: HOLD)
        self.closed = False
        self.phase = "start"
        self.solutions = {}
        self.moves = []
        self.rotations = []
        self.joint_moves = []
        self.standing = standing
        self.cameras = cameras
        self.joints = joints
        self.released_at = None

    # -- tools
    def pose(self, **kw):
        return {
            "pose": {
                "position": dict(zip("xyz", map(float, self.p), strict=True)),
                "rotation": self.s.quat(self.r),
            }
        }

    def move(self, tool):
        def go(pose, **kw):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy()))
            self.rotations.append((tool, self.s.matrix(pose["rotation"])))
            why = self.refuse(tool, pose)
            if why == "raise":
                raise ToolError(tool, "refused")
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            if (
                self.stop_descent_at is not None
                and tool.endswith("cartesian")
                and target[2] < self.p[2]
            ):
                target = target.copy()
                target[2] = max(target[2], self.stop_descent_at)
            self.p = target
            self.r = self._land(self.s.matrix(pose["rotation"]))
            return {"reached": True, "outcome": "reached"}

        return go

    def _land(self, rot):
        """With a free-roll solver (``roll_error_deg`` set) a pose move matches only the approach: the move that
        brings the approach level lands rolled by ``roll_error_deg``, the others keep the hand's present roll."""
        if not self.roll_error_deg:
            return rot
        z = np.array([0.0, 0.0, 1.0])
        a = rot @ z
        if abs(a[2]) < 0.2 <= abs((self.r @ z)[2]):
            return self.s.axis_angle(a, self.roll_error_deg) @ rot
        y = np.array([0.0, 1.0, 0.0])
        return self.s.axis_angle(a, self.s.turn_about(a, rot @ y, self.r @ y)) @ rot

    def solve_ik(self, pose, **kw):
        point = np.array([pose["position"][k] for k in "xyz"])
        rot = self.s.matrix(pose["rotation"])
        if not self.reachable(point, rot):
            return {"success": False, "within_servo_tolerance": False}
        q = tuple(np.round(np.array([0.3, 1.0, 0.9, 0.4, 0.2, 0.1]) + point[0] * 0.01, 6))
        if self.joints is not None:
            q = tuple(np.round(np.asarray(self.joints(point, rot), dtype=float), 6))
        self.solutions[q] = (point, rot)
        return {
            "success": True,
            "joint_config": {"positions": list(q)},
            "within_servo_tolerance": True,
        }

    def move_to_joints(self, joint_config, **kw):
        goal = np.asarray(joint_config["positions"], dtype=float)
        self.joint_moves.append(goal.copy())
        key = tuple(np.round(goal, 6))
        if key in self.solutions:
            point, rot = self.solutions[key]
            a = rot @ np.array([0.0, 0.0, 1.0])
            self.p, self.r = point.copy(), self.s.axis_angle(a, self.roll_error_deg) @ rot
        elif np.allclose(goal[:-1], self.q[:-1]):
            a = self.r @ np.array([0.0, 0.0, 1.0])
            self.r = (
                self.s.axis_angle(a, math.degrees(goal[-1] - self.q[-1]) * self.roll_gain) @ self.r
            )
        self.q = goal
        return None

    def grasp_frame(self, close_heading_deg, **kw):
        r = top_down(close_heading_deg)
        flipped = top_down(close_heading_deg + 180.0)
        return {
            "rotation": self.s.quat(r),
            "rotation_flipped": self.s.quat(flipped),
            "approach": {"x": 0.0, "y": 0.0, "z": -1.0},
            "close_axis": dict(zip("xyz", map(float, heading(close_heading_deg)), strict=True)),
        }

    def close(self, **kw):
        self.closed = True
        self.phase = "grasp"
        return {"position": self.grip("grasp")}

    def get_gripper(self, **kw):
        if not self.closed:
            return {"position": 1.0}
        a = self.r @ np.array([0.0, 0.0, 1.0])
        phase = "pitch" if abs(a[2]) < 0.5 else ("lift" if self.p[2] > TABLE_Z + 0.05 else "grasp")
        return {"position": self.grip(phase)}

    def set_grip(self, width_m, **kw):
        if self.closed and width_m > 0.024:
            self.closed = False
            self.released_at = self.p.copy()
        return {"requested_width_m": width_m}

    def observation(self, **kw):
        obs = {"arms": [{"joint_state": {"positions": self.q.tolist()}}]}
        if self.cameras:
            obs["cameras"] = [self.camera()]
        return obs

    def camera(self):
        """An overhead depth camera 0.9 m over the tile, looking down; the standing tile's top (or a lying tile)
        in its depth."""
        h, w, f = 480, 640, 460.0
        cam = np.array([TILE["x"], TILE["y"], TABLE_Z + 0.9])
        depth = np.full((h, w), 0.9, dtype=np.float32)
        top = 0.048 if self.standing else 0.0075
        vv, uu = np.mgrid[0:h, 0:w]
        x = cam[0] + (uu - 320.0) * (0.9 - top) / f
        y = cam[1] - (vv - 240.0) * (0.9 - top) / f
        along = heading(self.facing)[:2]
        sx, sy = self.stand_xy if self.stand_xy is not None else (TILE["x"], TILE["y"])
        rel = np.stack([x - sx, y - sy], axis=-1)
        a = rel @ along
        c = rel @ np.array([-along[1], along[0]])
        half_a, half_c = (0.00375, 0.012) if self.standing else (0.024, 0.012)
        depth[(np.abs(a) <= half_a) & (np.abs(c) <= half_c)] = 0.9 - top
        pose = np.eye(4)
        pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
        pose[:3, 3] = cam
        return {
            "name": "top",
            "depth": depth,
            "intrinsics": [[f, 0, 320.0], [0, f, 240.0], [0, 0, 1]],
            "pose": pose.tolist(),
        }

    facing = 0.0
    stand_xy = None

    def ctx(self, gripper=True):
        tools = {
            "robot.get_ee_pose": self.pose,
            "robot.go_to_pose": self.move("robot.go_to_pose"),
            "robot.go_to_pose_cartesian": self.move("robot.go_to_pose_cartesian"),
            "robot.solve_ik": self.solve_ik,
            "robot.move_to_joints": self.move_to_joints,
            "robot.grasp_frame": self.grasp_frame,
            "robot.close_gripper": self.close,
            "robot.open_gripper": lambda **kw: self.set_grip(0.09),
            "robot.get_gripper": self.get_gripper,
            "robot.set_grip": self.set_grip,
            "robot.get_observation": self.observation,
            "robot.describe_arm": {
                "joint_limits": {
                    "lower": [-2.6, 0.0, 0.0, -1.57, -1.57, -2.09],
                    "upper": [3.05, 3.65, 3.66, 1.57, 1.57, 2.09],
                }
            },
        }
        if gripper:
            tools["robot.describe_gripper"] = {
                "span_m": AT_OPEN,
                "width_fit": {"at_open_m": AT_OPEN, "at_closed_m": 0.0},
                "finger": {"reach_m": 0.0102, "stated": True},
            }
        return FakeContext(tools)


def stand(script, arm, **kw):
    args = {
        "table_z": TABLE_Z,
        "tile_center": TILE,
        "long_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
        "park_pose": {"position": {"x": 0.40, "y": 0.30, "z": TABLE_Z + 0.2}},
    }
    args.update(kw)
    args.setdefault("strategy", "pose_pitch")
    out = script.run(arm.ctx(), **args)
    arm.facing = out["facing_deg"]
    return out


def run_and_check(script, arm, **kw):
    """Run, and run the depth check against the facing the skill chose (the fake camera needs it)."""
    kw.setdefault("strategy", "pose_pitch")
    probe = Arm(script, reachable=arm.reachable)
    first = stand(script, probe, verify=False, **kw)
    arm.facing = first["facing_deg"]
    return script.run(
        arm.ctx(),
        table_z=TABLE_Z,
        tile_center=kw.pop("tile_center", TILE),
        long_axis=kw.pop("long_axis", {"x": 1.0, "y": 0.0, "z": 0.0}),
        park_pose={"position": {"x": 0.40, "y": 0.30, "z": TABLE_Z + 0.2}},
        **kw,
    )


# ----------------------------------------------------------------------------- geometry


def test_pitched_rotation_points_the_approach_along_the_facing_and_the_tile_end_up(script):
    g = top_down(90.0)
    a_l, c_l = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    for end in (0.0, 180.0, 4.0):  # 4 deg: the jaws closed a little off square
        e_l = g.T @ heading(end)
        for psi in (0.0, 37.0, 90.0, 200.0):
            u = heading(psi)
            r = script.pitched_rotation(u, a_l, c_l, e_l)
            assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
            assert np.linalg.det(r) == pytest.approx(1.0)
            assert np.allclose(r @ a_l, u, atol=1e-9)
            assert abs((r @ c_l)[2]) < 1e-9
            assert (r @ e_l)[2] > 0.99


def test_plan_release_hangs_the_tile_under_and_ahead_of_the_tool_centre(script):
    size = script.TILE_SIZE
    p = script.plan_release(np.array([0.5, 0.3]), 90.0, TABLE_Z, size, 0.0137, 0.0, 0.0015)
    assert p[2] == pytest.approx(TABLE_Z + 0.024 + 0.0015)
    assert p[0] == pytest.approx(0.5)
    assert p[1] == pytest.approx(0.3 - (0.0137 - 0.00375))
    q = script.plan_release(np.array([0.5, 0.3]), 0.0, TABLE_Z, size, 0.0137, 0.012, 0.0015)
    assert q[2] == pytest.approx(TABLE_Z + 0.024 + 0.012 + 0.0015)


def test_standing_pose_faces_the_broad_faces_normal(script):
    pose = script.standing_pose(np.array([0.5, 0.3]), 90.0, TABLE_Z, script.TILE_SIZE)
    assert pose["position"]["z"] == pytest.approx(TABLE_Z + 0.024)
    assert script.matrix(pose["rotation"]) @ np.array([1.0, 0, 0]) == pytest.approx(
        heading(90.0), abs=1e-9
    )


def test_wrap_and_turn_about(script):
    assert script.wrap(190.0, False) == pytest.approx(-170.0)
    assert script.wrap(100.0, True) == pytest.approx(-80.0)
    z = np.array([0.0, 0.0, 1.0])
    assert script.turn_about(z, heading(0.0), heading(30.0)) == pytest.approx(30.0)
    assert script.turn_about(-z, heading(0.0), heading(30.0)) == pytest.approx(-30.0)


# ----------------------------------------------------------------------------- the happy path


def test_stands_the_tile_where_it_lay_and_sees_it_from_depth(script):
    arm = Arm(script)
    out = run_and_check(script, arm)
    assert out["route"] == "standing", out["detail"]
    assert out["picked"] and out["released"] and out["verified"]
    assert out["facing_deg"] == pytest.approx(0.0)
    assert out["pitch_turn_deg"] == pytest.approx(90.0, abs=0.5)
    assert out["tilt_at_release_deg"] < 1.0
    assert out["tile_top"] == pytest.approx(0.048, abs=0.002)
    assert out["standing_offset"] < 0.003
    assert out["standing_tile"]["facing_deg"] == pytest.approx(0.0)
    assert out["standing_tile"]["position"]["z"] == pytest.approx(TABLE_Z + 0.024)
    rel = out["release_position"]
    assert math.hypot(rel["x"] - TILE["x"], rel["y"] - TILE["y"]) < 0.002
    assert rel["z"] == pytest.approx(TABLE_Z + 0.0015, abs=1e-6)
    # The hand let go where the tile's bottom is release_gap over the table.
    assert arm.released_at[2] == pytest.approx(TABLE_Z + 0.024 + 0.014 + 0.0015, abs=1e-6)


def test_the_pitch_is_stepped_about_the_level_axis(script):
    arm = Arm(script)
    out = stand(script, arm, verify=False, pitch_step_deg=10.0)
    assert out["route"] == "standing"
    z = np.array([0.0, 0.0, 1.0])
    approaches = [r @ z for tool, r in arm.rotations if tool == "robot.go_to_pose"]
    turns = [script.angle_between(a, b) for a, b in zip(approaches, approaches[1:], strict=False)]
    pitched = [t for t in turns if t > 0.5]
    assert len(pitched) == 9
    assert max(pitched) <= 10.0 + 1e-6
    # The tool centre stays over the release point through the turn.
    assert arm.joint_moves == []


def test_verify_false_reports_what_the_hand_did(script):
    out = stand(script, Arm(script), verify=False)
    assert out["route"] == "standing"
    assert "not checked from depth" in out["detail"]
    assert not out["verified"]


def test_staging_spot_elsewhere(script):
    arm = Arm(script)
    out = stand(script, arm, verify=False, staging={"x": 0.62, "y": 0.25, "z": TABLE_Z})
    assert out["route"] == "standing"
    rel = out["release_position"]
    assert math.hypot(rel["x"] - 0.62, rel["y"] - 0.25) < 0.002


def test_grip_offset_closes_toward_the_end_that_goes_up(script):
    arm = Arm(script)
    out = stand(script, arm, verify=False, grip_offset=0.012)
    assert out["route"] == "standing"
    descents = [p for tool, p in arm.moves if tool.endswith("cartesian") and p[2] < TABLE_Z + 0.02]
    assert descents[0][0] == pytest.approx(TILE["x"] + 0.012)
    assert out["release_position"]["z"] == pytest.approx(TABLE_Z + 0.0015, abs=1e-6)
    assert arm.released_at[2] == pytest.approx(TABLE_Z + 0.024 + 0.012 + 0.0015, abs=1e-6)


# ----------------------------------------------------------------------------- the facing search


def test_takes_the_first_facing_whose_pitched_release_solves(script):
    def reachable(point, rot):
        a = rot @ np.array([0.0, 0.0, 1.0])
        if a[2] < -0.9:
            return True
        return math.degrees(math.atan2(a[1], a[0])) == pytest.approx(120.0, abs=1.0)

    arm = Arm(script, reachable=reachable)
    out = stand(script, arm, verify=False)
    assert out["route"] == "standing", out["detail"]
    assert out["facing_deg"] == pytest.approx(120.0)
    assert out["standing_tile"]["facing_deg"] == pytest.approx(120.0)
    # The tile lies along x; facing 120 deg puts its -x end up, so the release sits ahead along 120 deg.
    assert out["tilt_at_release_deg"] < 1.0


def facing_of(rot):
    a = rot @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.atan2(a[1], a[0])), a


def test_a_facing_whose_pitch_path_does_not_solve_is_skipped(script):
    def reachable(point, rot):
        f, a = facing_of(rot)
        return not (-0.9 < a[2] < -0.1 and abs(f) < 1.0)

    out = stand(script, Arm(script, reachable=reachable), verify=False)
    assert out["route"] == "standing", out["detail"]
    # The other end up (facing 180 deg) needs no turn about the vertical either.
    assert abs(out["facing_deg"]) == pytest.approx(180.0)


def test_a_facing_whose_pitch_jumps_a_joint_is_skipped(script):
    def joints(point, rot):
        f, a = facing_of(rot)
        q = np.array([0.3, 1.0, 0.9, -1.0, 0.2, 0.1])
        # A quarter turn moves the wrist pitch by 1 rad, 0.03-0.26 rad a step.
        q[3] += 1.0 - abs(a[2])
        if abs(f) < 1.0 and abs(a[2]) < 0.1:
            q[3] = 1.5708  # the last step of facing 0 jumps onto the limit
        return q

    arm = Arm(script, joints=joints)
    out = stand(script, arm, verify=False)
    assert out["route"] == "standing", out["detail"]
    assert abs(out["facing_deg"]) == pytest.approx(180.0)


def test_no_facing_without_a_joint_jump_moves_nothing(script):
    def joints(point, rot):
        a = rot @ np.array([0.0, 0.0, 1.0])
        return [0.3, 1.0, 0.9, 1.5708 if abs(a[2]) < 0.1 else -1.0, 0.2, 0.1]

    arm = Arm(script, joints=joints)
    out = stand(script, arm, verify=False)
    assert out["route"] == "unreachable"
    assert "jumps" in out["detail"]
    assert not arm.moves and not out["picked"]


def test_a_facing_whose_turn_exceeds_the_last_joint_is_skipped(script):
    def reachable(point, rot):
        f, a = facing_of(rot)
        return a[2] < -0.9 or abs(abs(f) - 60.0) < 1.0

    arm = Arm(script, reachable=reachable)
    arm.q[-1] = -1.9
    out = stand(script, arm, verify=False)
    assert out["route"] == "standing", out["detail"]
    assert out["facing_deg"] == pytest.approx(-60.0)
    assert out["yaw_turn_deg"] == pytest.approx(60.0, abs=0.5)


def test_no_room_for_any_facing_is_routed_holding(script):
    def reachable(point, rot):
        f, a = facing_of(rot)
        return a[2] < -0.9 or abs(f - 60.0) < 1.0

    arm = Arm(script, reachable=reachable)
    arm.q[-1] = -1.9
    out = stand(script, arm, verify=False)
    assert out["route"] == "unreachable"
    assert "no room" in out["detail"] and "still holding" in out["detail"]
    assert out["picked"]


def test_no_reachable_facing_moves_nothing(script):
    def reachable(point, rot):
        return (rot @ np.array([0.0, 0.0, 1.0]))[2] < -0.9

    arm = Arm(script, reachable=reachable)
    out = stand(script, arm)
    assert out["route"] == "unreachable"
    assert "no facing passes the IK preview" in out["detail"]
    assert arm.moves == [] and not out["picked"]


def test_unreachable_grasp_tries_the_flipped_hand_then_gives_up(script):
    arm = Arm(script, reachable=lambda point, rot: point[2] > TABLE_Z + 0.02)
    out = stand(script, arm)
    assert out["route"] == "unreachable"
    assert "grasp" in out["detail"]
    assert arm.moves == []


def test_check_reach_false_skips_the_preview(script):
    arm = Arm(script, reachable=lambda point, rot: False)
    out = stand(script, arm, verify=False, check_reach=False)
    assert out["route"] == "standing"


# ----------------------------------------------------------------------------- the roll read-back


def test_a_free_roll_solver_is_finished_on_the_last_joint(script):
    arm = Arm(script, roll_error_deg=35.0)
    out = stand(script, arm, verify=False)
    assert out["route"] == "standing", out["detail"]
    assert out["roll_turn_deg"] == pytest.approx(-35.0, abs=1.0)
    assert out["tilt_at_release_deg"] < 4.0


def test_a_reversed_last_joint_is_learned(script):
    arm = Arm(script, roll_error_deg=30.0, roll_gain=-1.0)
    out = stand(script, arm, verify=False)
    assert out["route"] == "standing", out["detail"]
    assert out["tilt_at_release_deg"] < 4.0


def test_a_roll_past_the_last_joint_limit_is_routed(script):
    arm = Arm(script, roll_error_deg=170.0)
    arm.q[-1] = 1.9
    out = stand(
        script,
        arm,
        verify=False,
        joint_limits=[[-3, 0, 0, -1.6, -1.6, -0.2], [3, 3.7, 3.7, 1.6, 1.6, 0.2]],
    )
    assert out["route"] == "unreachable"
    assert "still holding" in out["detail"]
    assert out["picked"] and not out["released"]


def test_a_last_joint_that_does_not_turn_the_hand_leaves_a_lean(script):
    arm = Arm(script, roll_error_deg=20.0, roll_gain=0.0)
    out = stand(script, arm, verify=False)
    assert out["route"] == "unreachable"
    assert "leans" in out["detail"] and "still holding" in out["detail"]
    assert out["tilt_at_release_deg"] == pytest.approx(20.0, abs=0.5)


# ----------------------------------------------------------------------------- the pick and the drops


def test_jaws_closing_on_nothing_is_ungraspable(script):
    arm = Arm(script, grip=lambda phase: 0.0)
    out = stand(script, arm)
    assert out["route"] == "ungraspable"
    assert "closed on nothing" in out["detail"]
    assert not out["picked"] and not arm.closed


def test_a_descent_stopped_high_is_ungraspable(script):
    arm = Arm(script, stop_descent_at=TABLE_Z + 0.03)
    out = stand(script, arm)
    assert out["route"] == "ungraspable"
    assert "stopped" in out["detail"] and "high" in out["detail"]


def test_dropped_after_the_lift(script):
    arm = Arm(script, grip=lambda phase: HOLD if phase == "grasp" else 0.0)
    out = stand(script, arm)
    assert out["route"] == "dropped"
    assert "after the lift" in out["detail"]
    assert out["picked"]


def test_dropped_in_the_pitch(script):
    arm = Arm(script, grip=lambda phase: 0.0 if phase == "pitch" else HOLD)
    out = stand(script, arm)
    assert out["route"] == "dropped"
    assert "pitch" in out["detail"]


def test_tips_over_the_tile_are_ungraspable_before_moving(script):
    arm = Arm(script)
    out = stand(script, arm, grasp_height=0.02)
    assert out["route"] == "ungraspable"
    assert arm.moves == []


def test_grasp_under_the_floor_is_refused(script):
    out = stand(script, Arm(script), grasp_height=0.005)
    assert out["route"] == "unreachable"
    assert "floor" in out["detail"]


def test_a_tile_not_lying_flat(script):
    out = stand(script, Arm(script), long_axis={"x": 0.0, "y": 0.0, "z": 1.0})
    assert out["route"] == "unreachable"
    assert "does not lie flat" in out["detail"]


# ----------------------------------------------------------------------------- refused moves


@pytest.mark.parametrize("how", ["raise", "unreachable: servo"])
def test_a_refused_pitch_is_routed_still_holding(script, how):
    def refuse(tool, pose):
        r = script.matrix(pose["rotation"])
        return (
            how
            if tool == "robot.go_to_pose" and abs((r @ np.array([0, 0, 1.0]))[2]) < 0.5
            else None
        )

    out = stand(script, Arm(script, refuse=refuse))
    assert out["route"] == "unreachable"
    assert "still holding" in out["detail"]


@pytest.mark.parametrize("how", ["raise", "unreachable: servo"])
def test_a_refused_move_over_the_tile_is_routed(script, how):
    out = stand(
        script, Arm(script, refuse=lambda tool, pose: how if tool == "robot.go_to_pose" else None)
    )
    assert out["route"] == "unreachable"
    assert "over the tile" in out["detail"]


def test_a_refused_retreat_after_the_release(script):
    def refuse(tool, pose):
        p = pose["position"]
        return "blocked" if tool.endswith("cartesian") and p["x"] < TILE["x"] - 0.02 else None

    out = stand(script, Arm(script, refuse=refuse), verify=False)
    assert out["route"] == "unreachable"
    assert out["released"]
    assert "retreat" in out["detail"]


# ----------------------------------------------------------------------------- the depth check


def test_a_tile_lying_at_the_spot_after_the_release_fell(script):
    arm = Arm(script, standing=False)
    out = run_and_check(script, arm)
    assert out["route"] == "fell", out["detail"]
    assert out["verified"]
    assert out["tile_top"] == pytest.approx(0.0075, abs=0.002)


def test_no_camera_is_unverified(script):
    arm = Arm(script, cameras=False)
    out = run_and_check(script, arm)
    assert out["route"] == "unverified"
    assert out["released"] and not out["verified"]


def test_judge_standing_reads_a_leaning_tile_off_the_spot(script):
    arm = Arm(script)
    arm.facing = 0.0
    cam = arm.camera()
    ok = script.judge_standing(
        [cam], np.array([TILE["x"], TILE["y"]]), 0.0, TABLE_Z, script.TILE_SIZE
    )
    assert ok["standing"] and ok["seen"]
    off = script.judge_standing(
        [cam], np.array([TILE["x"] - 0.012, TILE["y"]]), 0.0, TABLE_Z, script.TILE_SIZE
    )
    assert not off["standing"]
    assert "off the spot" in off["why"]


def test_an_occluded_camera_does_not_judge(script):
    arm = Arm(script)
    cam = arm.camera()
    cam["depth"][:] = 0.5  # something 40 cm over the table fills the window
    v = script.judge_standing(
        [cam], np.array([TILE["x"], TILE["y"]]), 0.0, TABLE_Z, script.TILE_SIZE
    )
    assert not v["seen"] and v["why"] == "occluded"


# ----------------------------------------------------------------------------- the bundle


def test_every_route_is_declared_in_the_example():
    example = json.loads((ROOT / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    source = (ROOT / "scripts" / "stand_domino.py").read_text()
    routes = set(re.findall(r'finish\(\s*"([a-z_]+)"', source))
    assert routes, "no routes found"
    assert routes <= declared, routes - declared


def test_produced_outputs_are_in_the_output(script):
    import yaml

    text = (ROOT / "SKILL.md").read_text()
    front = yaml.safe_load(text.split("---")[1])
    produced = set(front["gap"]["produces_outputs"])
    assert produced <= set(script.Output.__annotations__)


# ----------------------------------------------------------------------------- table_pivot


PITCH_AXIS = np.array([0.0, -1.0, 0.0])
"""The fake arm's pitch joint (joint 4) turns the hand about this level axis."""
WRIST_LEVER = 0.2
"""How far behind the tool centre, along the approach, the fake pitch joint's axis passes [m]."""


class JointArm(Arm):
    """A fake arm whose joints move the hand: joint 4 pitches it about a level axis through a point WRIST_LEVER
    behind the tool centre, joint 5 turns it about a level axis with no room for a quarter turn, joint 6 turns it
    about the approach. robot.solve_ik answers a little off the present joints, and a move to that answer lands on
    the asked pose; ``jump(point, rot)`` adds a jump to an answer, ``ik_ok(point, rot)`` refuses one."""

    def __init__(self, script, jump=None, ik_ok=None, **kw):
        super().__init__(script, **kw)
        self.q = np.array([0.0, 1.6, 1.2, -1.1, 0.0, 0.0])
        self.jump = jump or (lambda point, rot: None)
        self.ik_ok = ik_ok or (lambda point, rot: True)
        self.asked = []

    def solve_ik(self, pose, **kw):
        point = np.array([pose["position"][k] for k in "xyz"])
        rot = self.s.matrix(pose["rotation"])
        self.asked.append((point, rot))
        if not self.ik_ok(point, rot):
            return {"success": False, "within_servo_tolerance": False}
        q = self.q + 0.001 * (len(self.solutions) % 7 + 1)
        extra = self.jump(point, rot)
        if extra is not None:
            q = q + np.asarray(extra, dtype=float)
        key = tuple(np.round(q, 6))
        self.solutions[key] = (point, rot)
        return {
            "success": True,
            "joint_config": {"positions": list(key)},
            "within_servo_tolerance": True,
        }

    def move_to_joints(self, joint_config, **kw):
        goal = np.asarray(joint_config["positions"], dtype=float)
        self.joint_moves.append(goal.copy())
        key = tuple(np.round(goal, 6))
        if key in self.solutions:
            self.p, self.r = (v.copy() for v in self.solutions[key])
        else:
            dq = goal - self.q
            z = np.array([0.0, 0.0, 1.0])
            if abs(dq[3]) > 1e-9:
                a = self.r @ z
                pivot = self.p - WRIST_LEVER * a
                turn = self.s.axis_angle(PITCH_AXIS, math.degrees(dq[3]))
                self.p = pivot + turn @ (self.p - pivot)
                self.r = turn @ self.r
            if abs(dq[4]) > 1e-9:
                self.r = self.s.axis_angle(np.array([1.0, 0.0, 0.0]), math.degrees(dq[4])) @ self.r
            if abs(dq[5]) > 1e-9:
                a = self.r @ z
                self.r = self.s.axis_angle(a, math.degrees(dq[5]) * self.roll_gain) @ self.r
        self.q = goal
        return None


def joint_stand(script, arm, **kw):
    args = {
        "table_z": TABLE_Z,
        "tile_center": TILE,
        "long_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
        "park_pose": {"position": {"x": 0.40, "y": 0.30, "z": TABLE_Z + 0.2}},
    }
    args.update(kw)
    return script.run(arm.ctx(), **args)


def test_table_pivot_stands_the_tile_and_sees_it_from_depth(script):
    first = joint_stand(script, JointArm(script), verify=False)
    assert first["route"] == "standing", first["detail"]
    arm = JointArm(script)
    arm.facing = first["facing_deg"]
    arm.stand_xy = (
        first["standing_tile"]["position"]["x"],
        first["standing_tile"]["position"]["y"],
    )
    out = joint_stand(script, arm)
    assert out["route"] == "standing", out["detail"]
    assert out["strategy"] == "table_pivot" and out["pitch_joint"] == 4
    assert out["picked"] and out["released"] and out["verified"]
    assert out["pitch_turn_deg"] == pytest.approx(90.0, abs=0.5)
    assert out["tilt_at_release_deg"] < 1.0
    # The approach ends level and along where the pitch joint takes it (+x here); the tile's broad faces face it.
    assert out["facing_deg"] == pytest.approx(0.0, abs=0.5)
    assert out["standing_tile"]["position"]["z"] == pytest.approx(TABLE_Z + 0.024)
    # Upright on the table: the hand let go with the tool centre half the tile's height plus the grip offset up.
    assert arm.released_at[2] == pytest.approx(TABLE_Z + 0.024 + 0.014, abs=1e-6)


def test_table_pivot_air_pitch_turns_only_the_pitch_joint(script):
    arm = JointArm(script)
    out = joint_stand(script, arm, verify=False, pitch_step_deg=15.0, air_pitch_deg=60.0)
    assert out["route"] == "standing", out["detail"]
    deltas = [b - a for a, b in zip(arm.joint_moves, arm.joint_moves[1:], strict=False)]
    alone = [d[3] for d in deltas if abs(d[3]) > 0.2 and np.allclose(np.delete(d, 3), 0.0)]
    assert all(abs(v) == pytest.approx(math.radians(15.0), abs=1e-6) for v in alone)
    # The empty rehearsal: six steps up and six back; then 60 deg in the air in four.
    assert sum(v > 0 for v in alone) == 6 + 4 and sum(v < 0 for v in alone) == 6


def test_table_pivot_refuses_a_step_that_jumps_a_joint(script):
    def jump(point, rot):
        a = rot @ np.array([0.0, 0.0, 1.0])
        if abs(a[2]) < 0.3:  # the last turns upright on the table
            return [0.0, 0.0, 0.0, 0.9, 0.0, 0.0]
        return None

    arm = JointArm(script, jump=jump)
    out = joint_stand(script, arm, verify=False)
    assert out["route"] == "unreachable"
    assert "would jump 0.90 rad" in out["detail"] and "still holding" in out["detail"]
    assert not out["released"]
    steps = [
        np.max(np.abs(b[:-1] - a[:-1]))
        for a, b in zip(arm.joint_moves, arm.joint_moves[1:], strict=False)
    ]
    assert max(steps) < 0.6


def test_table_pivot_routes_unreachable_when_no_joint_pitches_level_with_room(script):
    arm = JointArm(script)
    limits = [[-2.6, 0.0, 0.0, -1.2, -1.57, -2.09], [3.05, 3.65, 3.66, 0.2, 1.57, 2.09]]
    out = joint_stand(script, arm, verify=False, joint_limits=limits)
    assert out["route"] == "unreachable"
    assert "no joint turns the hand about a level axis" in out["detail"]
    assert not out["picked"]


def test_table_pivot_drop_in_the_air_is_dropped(script):
    arm = JointArm(script, grip=lambda phase: 0.0 if phase == "pitch" else HOLD)

    def gripper(**kw):
        if not arm.closed:
            return {"position": 1.0}
        a = arm.r @ np.array([0.0, 0.0, 1.0])
        return {"position": 0.0 if a[2] > -0.75 else HOLD}

    arm.get_gripper = gripper
    out = joint_stand(script, arm, verify=False, air_pitch_deg=60.0)
    assert out["route"] == "dropped"
    assert "during the pitch" in out["detail"]


def test_table_pivot_moves_the_spot_out_when_the_first_does_not_solve(script):
    first = joint_stand(script, JointArm(script), verify=False)
    x0 = first["standing_tile"]["position"]["x"]

    refused = []

    def ik_ok(point, rot):
        a = rot @ np.array([0.0, 0.0, 1.0])
        # The first set-down asked for (the hand pitched, near the table) has no solution.
        if not refused and point[2] < TABLE_Z + 0.06 and abs(a[2]) < 0.9:
            refused.append(point)
            return False
        return True

    arm = JointArm(script, ik_ok=ik_ok)
    out = joint_stand(script, arm, verify=False)
    assert out["route"] == "standing", out["detail"]
    assert out["standing_tile"]["position"]["x"] == pytest.approx(x0 + 0.03, abs=0.002)


def test_table_pivot_stands_it_at_a_given_staging_spot(script):
    spot = {"x": 0.70, "y": 0.26, "z": TABLE_Z}
    arm = JointArm(script)
    out = joint_stand(script, arm, verify=False, staging=spot)
    assert out["route"] == "standing", out["detail"]
    at = out["standing_tile"]["position"]
    assert math.hypot(at["x"] - 0.70, at["y"] - 0.26) < 0.002


def test_table_pivot_turns_a_tile_lying_across_the_pitch_plane(script):
    arm = JointArm(script)
    out = joint_stand(script, arm, verify=False, long_axis={"x": 0.0, "y": 1.0, "z": 0.0})
    assert out["route"] == "standing", out["detail"]
    assert abs(out["yaw_turn_deg"]) == pytest.approx(90.0, abs=1.0)
    assert out["facing_deg"] == pytest.approx(0.0, abs=0.5)


def test_unknown_strategy_is_refused(script):
    out = joint_stand(script, JointArm(script), strategy="tip_up")
    assert out["route"] == "unreachable" and "not one of" in out["detail"]
    assert not out["picked"]
