"""pressing-a-brick-onto-studs: FakeContext tests against a small rigid-brick world.

The fake arm holds a DUPLO 2x2 brick rigidly. A move goes where it is sent
unless the brick would pass through the studs or the plate; then the hand stops
at the last pose along the move that is clear (a servo's timeout / contact). The
studs are a rule, not a mesh: a level brick goes down to the plate only when its
centre is within CAPTURE of the true seat (otherwise it rests on the stud tops,
a stud height up); a tilted brick's leading edge goes down to the plate only in
the gap between the seat's edge and the next stud row, and stops against the
seat's leading row when pulled back. After the release a wrist camera renders
the brick's top as a plane of depth.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "pressing-a-brick-onto-studs"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
SEAT = np.array([0.50, 0.0, 0.05])
STUD_H, BODY_H, HALF = 0.0045, 0.0192, 0.0159
HANG = 0.016
CAPTURE = 0.0003
GAP = 0.0034
START = np.array([0.40, 0.10, 0.20])


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["press_brick"].module


@pytest.fixture(autouse=True)
def no_sleep(script, monkeypatch):
    monkeypatch.setattr(script.time, "sleep", lambda s: None)


def _rot(q):
    w, x, y, z = (float(q[k]) for k in "wxyz")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _axis_angle(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _rotvec(r):
    angle = math.acos(min(1.0, max(-1.0, (np.trace(r) - 1) / 2)))
    if angle < 1e-12:
        return np.array([1.0, 0, 0]), 0.0
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if np.linalg.norm(axis) < 1e-9:  # a half turn
        w, v = np.linalg.eigh((r + np.eye(3)) / 2)
        return v[:, int(np.argmax(w))], angle
    return axis / np.linalg.norm(axis), angle


def held_obb(yaw_deg=0.0, hand=START):
    """The held brick's box: its bottom centre HANG under the hand, its studs on top."""
    a = math.radians(yaw_deg) / 2
    return {
        "center": dict(
            zip("xyz", map(float, [hand[0], hand[1], hand[2] - HANG + 0.01185]), strict=True)
        ),
        "extent": {"x": HALF, "y": HALF, "z": 0.01185},
        "orientation": {"w": math.cos(a), "x": 0.0, "y": 0.0, "z": math.sin(a)},
    }


class World:
    """A YAM-like hand holding a brick over a baseplate (see the module docstring)."""

    def __init__(
        self,
        seat=SEAT,
        yaw_deg=0.0,
        roll_free=False,
        slide=False,
        absent=False,
        top_offset=None,
        tilt_top=0.0,
        camera=True,
        refuse=None,
        raise_blocked=False,
        gripper_fraction=0.35,
        ik_ok=True,
        studs=True,
    ):
        self.p = START.copy()
        self.r = _rot(DOWN)
        self.seat = np.asarray(seat, float)
        self.roll_free, self.slide, self.absent = roll_free, slide, absent
        self.top_offset, self.tilt_top, self.camera = top_offset, tilt_top, camera
        self.refuse = refuse or (lambda tool, p, r: None)
        self.raise_blocked = raise_blocked
        self.gripper_fraction, self.ik_ok, self.studs = gripper_fraction, ik_ok, studs
        a = math.radians(yaw_deg)
        bx = np.array([math.cos(a), math.sin(a), 0.0])
        # The brick in the hand's frame: its bottom centre and its x / y axes.
        self.v = self.r.T @ np.array([0.0, 0.0, -HANG])
        self.bx = self.r.T @ bx
        self.by = self.r.T @ np.array([-bx[1], bx[0], 0.0])
        self.held = True
        self.top_z = None
        self.q = np.zeros(6)
        self.moves = []
        self.lead = np.array([1.0, 0.0, 0.0])

    # -- the brick
    def bottom(self, p, r):
        return p + r @ self.v

    def corners(self, p, r):
        b = self.bottom(p, r)
        return [
            b + r @ (sx * HALF * self.bx + sy * HALF * self.by) for sx in (-1, 1) for sy in (-1, 1)
        ]

    def clear(self, p, r):
        if not self.held or self.slide:
            return True
        pts = self.corners(p, r)
        lowest = min(c[2] for c in pts)
        tilt = math.degrees(math.acos(min(1.0, abs(float((r @ np.cross(self.bx, self.by))[2])))))
        stud_top = self.seat[2] + STUD_H
        if tilt < 2.0:
            err = self.bottom(p, r)[:2] - self.seat[:2]
            floor = self.seat[2] if np.all(np.abs(err) <= CAPTURE) or not self.studs else stud_top
            return lowest >= floor - 1e-9
        b = self.bottom(p, r)
        edge = b + r @ (HALF * self.bx * float(np.sign(self.lead[0])))
        u = float(np.dot(edge - self.seat, self.lead))
        if edge[2] >= stud_top - 1e-9 or not self.studs:
            return edge[2] >= self.seat[2] - 1e-9
        if u < HALF - 1e-9:  # over the seat's leading row, or pulled into it
            return False
        if u > HALF + GAP:  # on the next row
            return False
        return edge[2] >= self.seat[2] - 1e-9

    # -- moves
    def _go(self, tool, p_goal, r_goal):
        p0, r0 = self.p.copy(), self.r.copy()
        if self.roll_free:
            # Keep the hand's yaw: only the approach axis follows (a twin's IK).
            axis, ang = _rotvec(r_goal @ r0.T)
            yaw = math.atan2((r_goal @ r0.T)[1, 0], (r_goal @ r0.T)[0, 0])
            if abs(axis[2]) > 0.99:
                r_goal = r0
            else:
                r_goal = _axis_angle([0, 0, 1], -yaw) @ r_goal
        axis, ang = _rotvec(r_goal @ r0.T)

        def pose(f):
            return p0 + f * (p_goal - p0), _axis_angle(axis, f * ang) @ r0 if ang else r0

        if self.clear(*pose(1.0)):
            self.p, self.r = pose(1.0)
            return {"outcome": "reached", "reached": True}
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if self.clear(*pose(mid)) else (lo, mid)
        self.p, self.r = pose(lo)
        if self.raise_blocked:
            raise RuntimeError("the arm did not reach its target")
        return {
            "outcome": "contact" if tool == "robot.move_cartesian_until_contact" else "timeout",
            "reached": False,
        }

    def _move(self, tool):
        def go(pose, **kw):
            p = np.array([pose["position"][k] for k in "xyz"], float)
            r = _rot(pose["rotation"])
            self.moves.append((tool, p.copy(), r.copy(), kw))
            why = self.refuse(tool, p, r)
            if why:
                if why == "raise":
                    raise RuntimeError("unreachable pose")
                return {"outcome": "unreachable", "reached": False, "reason": why}
            return self._go(tool, p, r)

        return go

    def _joints(self, joint_config, **kw):
        q = np.asarray(joint_config["positions"], float)
        turn = q[-1] - self.q[-1]
        self.q = q
        self.moves.append(("robot.move_to_joints", self.p.copy(), self.r.copy(), {}))
        # The last joint turns a top-down hand about the vertical, the other way.
        self.r = _axis_angle([0, 0, 1], -turn) @ self.r
        return {"outcome": "reached", "reached": True}

    def _open(self, **kw):
        if self.held:
            b = self.bottom(self.p, self.r)
            err = b[:2] - self.seat[:2]
            seated = np.all(np.abs(err) <= CAPTURE) and b[2] <= self.seat[2] + 1e-6
            self.top_z = self.seat[2] + BODY_H + (0.0 if seated else STUD_H)
            if self.top_offset is not None:
                self.top_z = self.seat[2] + BODY_H + self.top_offset
            if self.absent:
                self.top_z = self.seat[2]
        self.held = False
        return {"position": 1.0}

    # -- sensing
    def pose(self):
        return {
            "position": dict(zip("xyz", map(float, self.p), strict=True)),
            "rotation": _quat(self.r),
        }

    def camera_pose(self):
        tool = np.eye(4)
        tool[:3, :3], tool[:3, 3] = self.r, self.p
        tc = np.eye(4)
        tc[:3, :3] = _axis_angle([1, 0, 0], math.radians(-30))
        tc[:3, 3] = [0.0, -0.06, -0.07]
        return tool @ tc

    def depth(self, k, cam):
        h, w = 48, 64
        if self.top_z is None:
            return np.zeros((h, w), np.float32)
        u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
        d = np.stack([(u - k[0][2]) / k[0][0], (v - k[1][2]) / k[1][1], np.ones_like(u)], -1)
        world = d @ cam[:3, :3].T
        c = cam[:3, 3]
        # The top plane, tilted by tilt_top [rad] about y through the seat.
        s = math.tan(self.tilt_top)
        num = self.top_z + s * (c[0] - self.seat[0]) - c[2]
        den = world[..., 2] - s * world[..., 0]
        t = num / den
        return np.where(t > 0, t, 0.0).astype(np.float32)

    def observation(self, **kw):
        k = [[40.0, 0.0, 32.0], [0.0, 40.0, 24.0], [0.0, 0.0, 1.0]]
        cams = []
        if self.camera:
            cam = self.camera_pose()
            cams.append(
                {
                    "name": "left_eye_in_hand",
                    "pose": {
                        "position": dict(zip("xyz", map(float, cam[:3, 3]), strict=True)),
                        "rotation": _quat(cam[:3, :3]),
                    },
                    "intrinsics": k,
                    "depth": self.depth(k, cam),
                    "rgb": np.zeros((48, 64, 3), np.uint8),
                }
            )
        return {
            "arms": [{"ee_pose": self.pose(), "joint_state": {"positions": self.q.tolist()}}],
            "cameras": cams,
        }

    def ctx(self):
        return FakeContext(
            {
                "robot.get_ee_pose": lambda **kw: {"pose": self.pose()},
                "robot.get_observation": self.observation,
                "robot.get_gripper": lambda **kw: {"position": self.gripper_fraction},
                "robot.describe_gripper": {"width_fit": {"at_open_m": 0.094, "at_closed_m": 0.0}},
                "robot.go_to_pose": self._move("robot.go_to_pose"),
                "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
                "robot.move_cartesian_until_contact": self._move(
                    "robot.move_cartesian_until_contact"
                ),
                "robot.move_to_joints": self._joints,
                "robot.solve_ik": lambda **kw: {
                    "success": self.ik_ok,
                    "within_servo_tolerance": self.ik_ok,
                },
                "robot.open_gripper": self._open,
            }
        )


def _quat(r):
    t = float(np.trace(r))
    if t > 0:
        s = 2 * math.sqrt(t + 1)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2 * math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = 2 * math.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2])
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = 2 * math.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1])
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    return dict(zip("wxyz", map(float, q), strict=True))


def seat_at(dx=0.0, dy=0.0):
    return dict(zip("xyz", map(float, SEAT + [dx, dy, 0.0]), strict=True))


def tools(world):
    return [m[0] for m in world.moves]


# ---------------------------------------------------------------- straight


def test_straight_press_seats_and_checks_the_height(script):
    world = World()
    ctx = world.ctx()
    out = script.run(ctx, seat_position=seat_at(), held_obb=held_obb(), seat_yaw_deg=0.0)
    assert out["route"] == "seated", out["detail"]
    assert out["released"] is True
    assert out["verified_by"] == "wrist_depth"
    assert out["samples"] >= 3
    assert out["seat_height_error"] == pytest.approx(0.0, abs=1e-6)
    assert out["contact_height"] == pytest.approx(0.0, abs=0.0012)
    names = tools(world)
    # Hover, the fold, the fast descent to over the stud tops, then fine steps.
    assert names[0] == "robot.go_to_pose" and world.moves[0][3] == {}
    np.testing.assert_allclose(world.moves[0][1], [0.50, 0.0, SEAT[2] + HANG + 0.04], atol=1e-9)
    assert world.moves[1][3] == {"tolerance": 0.0003, "max_steps": 90}
    first_contact_move = names.index("robot.move_cartesian_until_contact")
    np.testing.assert_allclose(
        world.moves[first_contact_move][1], [0.50, 0.0, SEAT[2] + HANG + STUD_H + 0.006], atol=1e-9
    )
    calls = [c.tool for c in ctx.calls]
    # Released after the press, then the retreat and the view for the height.
    assert calls.index("robot.open_gripper") < len(calls) - 1
    after = names[names.index("robot.go_to_pose_cartesian") :]
    assert after[0] == "robot.go_to_pose_cartesian"
    # The press stopped with the command press_depth ahead of the stopped hand.
    assert out["seated_position"] == pytest.approx(seat_at())
    assert ctx.call_count("robot.solve_ik") >= 2  # the seat pose and the view


def test_straight_press_on_the_stud_tops_is_misaligned_and_kept(script):
    world = World()
    ctx = world.ctx()
    out = script.run(ctx, seat_position=seat_at(dx=0.002), held_obb=held_obb(), seat_yaw_deg=0.0)
    assert out["route"] == "misaligned", out["detail"]
    assert out["released"] is False
    assert ctx.call_count("robot.open_gripper") == 0
    assert out["contact_height"] == pytest.approx(STUD_H, abs=0.0012)
    assert "still holding" in out["detail"]
    tool, lift = world.moves[-1][:2]
    assert tool == "robot.go_to_pose_cartesian"
    assert lift[2] == pytest.approx(SEAT[2] + HANG + 0.04)


def test_a_brick_that_slid_in_the_jaws_is_read_resting_on_the_studs(script):
    # Nothing stops the hand (the brick slides up the jaws), so the press runs to
    # its bottom and releases; the depth then reads the brick a stud height up.
    world = World(seat=SEAT + [0.002, 0.0, 0.0], slide=True)
    out = script.run(world.ctx(), seat_position=seat_at(), held_obb=held_obb())
    assert out["route"] == "misaligned", out["detail"]
    assert out["released"] is True
    assert out["seat_height_error"] == pytest.approx(STUD_H, abs=1e-5)
    assert "met no resistance" in out["detail"]


def test_partly_pressed_is_not_seated(script):
    world = World(top_offset=0.002)
    out = script.run(world.ctx(), seat_position=seat_at(), held_obb=held_obb())
    assert out["route"] == "not_seated", out["detail"]
    assert out["seat_height_error"] == pytest.approx(0.002, abs=1e-5)


def test_a_tilted_top_is_misaligned(script):
    world = World(tilt_top=math.radians(12.0))
    out = script.run(world.ctx(), seat_position=seat_at(), held_obb=held_obb())
    assert out["route"] == "misaligned", out["detail"]
    assert out["seat_tilt"] > 0.5 * STUD_H


def test_jaws_at_their_stop_are_not_holding_and_nothing_moves(script):
    world = World(gripper_fraction=0.0)
    out = script.run(world.ctx(), seat_position=seat_at(), held_obb=held_obb())
    assert out["route"] == "not_holding", out["detail"]
    assert world.moves == []


def test_no_brick_top_at_the_seat_is_not_holding(script):
    world = World(absent=True)
    out = script.run(world.ctx(), seat_position=seat_at(), held_obb=held_obb())
    assert out["route"] == "not_holding", out["detail"]
    assert out["released"] is True


def test_no_wrist_camera_is_unverified(script):
    world = World(camera=False)
    out = script.run(world.ctx(), seat_position=seat_at(), held_obb=held_obb())
    assert out["route"] == "unverified", out["detail"]
    assert out["released"] is True
    assert "pressed to" in out["detail"]


def test_verify_false_is_unverified(script):
    world = World()
    out = script.run(world.ctx(), seat_position=seat_at(), held_obb=held_obb(), verify=False)
    assert out["route"] == "unverified"
    assert "verify=false" in out["detail"]


def test_ik_preview_refusal_is_unreachable_before_moving(script):
    world = World(ik_ok=False)
    out = script.run(world.ctx(), seat_position=seat_at(), held_obb=held_obb())
    assert out["route"] == "unreachable"
    assert world.moves == []


@pytest.mark.parametrize("how", ["raise", "routed"])
def test_a_refused_hover_is_unreachable(script, how):
    world = World(
        refuse=lambda tool, p, r: (
            (how if how == "raise" else "out of reach") if p[2] > 0.1 else None
        )
    )
    out = script.run(world.ctx(), seat_position=seat_at(), held_obb=held_obb())
    assert out["route"] == "unreachable", out["detail"]
    assert out["released"] is False


def test_a_station_raising_on_a_blocked_step_still_reads_contact(script):
    world = World(raise_blocked=True)
    out = script.run(world.ctx(), seat_position=seat_at(dx=0.002), held_obb=held_obb())
    assert out["route"] == "misaligned", out["detail"]
    assert out["contact_height"] == pytest.approx(STUD_H, abs=0.0012)


def test_without_a_held_box_the_grasp_depth_sets_the_hang(script):
    # BrickSim's 5 mm grasp depth and the YAM's 10.2 mm fingertip reach: the
    # brick's bottom hangs 19.2 + 10.2 - 5 = 24.4 mm under the tool centre.
    world = World()
    world.v = world.r.T @ np.array([0.0, 0.0, -0.0244])
    out = script.run(world.ctx(), seat_position=seat_at(), verify=False)
    assert out["route"] == "unverified", out["detail"]
    np.testing.assert_allclose(world.moves[0][1], [0.50, 0.0, SEAT[2] + 0.0244 + 0.04], atol=1e-9)


def test_arm_id_is_passed_only_when_set(script):
    world = World()
    ctx = world.ctx()
    script.run(ctx, seat_position=seat_at(), held_obb=held_obb(), verify=False)
    assert all("arm_id" not in c.kwargs for c in ctx.calls_to("robot.go_to_pose"))
    world = World()
    ctx = world.ctx()
    script.run(ctx, seat_position=seat_at(), held_obb=held_obb(), verify=False, arm_id=1)
    assert all(c.kwargs.get("arm_id") == 1 for c in ctx.calls_to("robot.go_to_pose"))


# ---------------------------------------------------------------- yaw


def test_a_yawed_brick_is_turned_at_the_hover(script):
    world = World(yaw_deg=-5.0)
    out = script.run(
        world.ctx(), seat_position=seat_at(), held_obb=held_obb(yaw_deg=-5.0), seat_yaw_deg=0.0
    )
    assert out["route"] == "seated", out["detail"]
    assert out["yaw_turn"] == pytest.approx(5.0, abs=0.01)
    assert "robot.move_to_joints" not in tools(world)


def test_a_free_wrist_roll_is_finished_on_the_last_joint(script):
    world = World(yaw_deg=-5.0, roll_free=True)
    out = script.run(
        world.ctx(), seat_position=seat_at(), held_obb=held_obb(yaw_deg=-5.0), seat_yaw_deg=0.0
    )
    assert "robot.move_to_joints" in tools(world)
    assert out["yaw_turn"] == pytest.approx(5.0, abs=1.0)
    assert out["route"] == "seated", out["detail"]


def test_a_yawed_brick_without_align_yaw_is_misaligned_before_moving(script):
    world = World(yaw_deg=-5.0)
    out = script.run(
        world.ctx(),
        seat_position=seat_at(),
        held_obb=held_obb(yaw_deg=-5.0),
        seat_yaw_deg=0.0,
        align_yaw=False,
    )
    assert out["route"] == "misaligned"
    assert world.moves == []


def test_a_square_brick_a_quarter_turn_off_is_not_turned(script):
    world = World(yaw_deg=90.0)
    out = script.run(
        world.ctx(), seat_position=seat_at(), held_obb=held_obb(yaw_deg=90.0), seat_yaw_deg=0.0
    )
    assert out["yaw_turn"] == 0.0
    assert out["route"] == "seated", out["detail"]


# ---------------------------------------------------------------- tilt


def test_tilt_hooks_the_row_and_corrects_the_seat_along_the_lead(script):
    # The seat handed in is 1 mm past the true one along the lead.
    world = World()
    out = script.run(
        world.ctx(), seat_position=seat_at(dx=0.001), held_obb=held_obb(), strategy="tilt"
    )
    assert out["route"] == "seated", out["detail"]
    assert out["lead_correction"] == pytest.approx(-0.001, abs=0.0003)
    assert out["seated_position"]["x"] == pytest.approx(SEAT[0], abs=0.0003)
    # The tilt is raised past ICL's 14 deg so the underside tube clears the studs.
    assert 20.0 < out["tilt_deg"] <= 27.0
    assert "caught the stud row" in out["detail"]


@pytest.mark.parametrize("dx", [-0.0015, 0.0, 0.0015])
def test_tilt_seats_within_the_hook_window(script, dx):
    world = World()
    out = script.run(
        world.ctx(), seat_position=seat_at(dx=dx), held_obb=held_obb(), strategy="tilt"
    )
    assert out["route"] == "seated", (dx, out["detail"])


def test_tilt_edge_on_the_next_row_is_misaligned_and_kept(script):
    world = World()
    ctx = world.ctx()
    out = script.run(ctx, seat_position=seat_at(dx=0.004), held_obb=held_obb(), strategy="tilt")
    assert out["route"] == "misaligned", out["detail"]
    assert out["released"] is False
    assert ctx.call_count("robot.open_gripper") == 0
    assert "on a stud row" in out["detail"]


def test_tilt_edge_short_of_the_seat_lands_on_the_leading_row(script):
    world = World()
    out = script.run(
        world.ctx(), seat_position=seat_at(dx=-0.003), held_obb=held_obb(), strategy="tilt"
    )
    assert out["route"] == "misaligned", out["detail"]
    assert out["released"] is False


def test_tilt_hook_that_meets_no_row_is_misaligned(script):
    world = World(studs=False)
    out = script.run(world.ctx(), seat_position=seat_at(), held_obb=held_obb(), strategy="tilt")
    assert out["route"] == "misaligned", out["detail"]
    assert "without meeting the stud row" in out["detail"]


def test_tilt_lead_minus_x_mirrors(script):
    world = World()
    world.lead = np.array([-1.0, 0.0, 0.0])
    out = script.run(
        world.ctx(),
        seat_position=seat_at(dx=-0.001),
        held_obb=held_obb(),
        strategy="tilt",
        lead="-x",
    )
    assert out["route"] == "seated", out["detail"]
    assert out["lead_correction"] == pytest.approx(-0.001, abs=0.0003)


# ---------------------------------------------------------------- systems and contract


def test_lego_preset_scales_the_steps_and_the_tilt(script):
    pitch, stud_h = 0.008, 0.0017
    step = max(0.0004, 0.25 * stud_h)
    assert step == pytest.approx(0.000425)
    need = math.degrees(math.asin((1.1 * stud_h - 0.2 * stud_h) / (0.58 * pitch)))
    assert 14.0 < need < 27.0
    assert script.PRESETS["lego"]["stud_pitch"] == pitch
    assert script.PRESETS["duplo"]["stud_pitch"] == 2 * pitch


def test_explicit_dimensions_override_the_preset(script):
    world = World()
    out = script.run(
        world.ctx(),
        seat_position=seat_at(),
        held_obb=held_obb(),
        system="lego",
        stud_pitch=0.016,
        stud_height=STUD_H,
        stud_diameter=0.0094,
        brick_height=BODY_H,
    )
    assert out["route"] == "seated", out["detail"]


@pytest.mark.parametrize(
    "kwargs",
    [{"strategy": "twist"}, {"system": "megablok"}, {"lead": "z"}],
)
def test_bad_arguments_raise(script, kwargs):
    with pytest.raises(ValueError):
        script.run(World().ctx(), seat_position=seat_at(), **kwargs)


def test_every_route_is_declared_in_the_example(script, skills_registry):
    bundle = skills_registry.get(BUNDLE)
    example = json.loads(
        (Path(bundle.bundle_dir) / "examples" / "canonical_subgraph.json").read_text()
    )
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert set(script.ROUTES) <= declared
