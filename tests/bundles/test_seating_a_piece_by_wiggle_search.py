"""seating-a-piece-by-wiggle-search: FakeContext tests against a small rigid-piece world.

The fake arm holds a 2 mm piece rigidly, offset from the tool centre. The slot is a
rule, not a mesh: the piece goes down to the mat only when its reference point is
within CLEAR of the true slot and its yaw within YAW_TOL; within PARTIAL of it the
piece stops half a thickness up (partly in); anywhere else it rests on the
neighbours, a thickness up. A move goes where it is sent, except that the hand is
held up so the piece's bottom stays over that floor (a servo's timeout). The last
joint turns the top-down hand about the vertical, the other way. After the release
a wrist camera looking along the tool axis sees the piece's top as a plane of depth.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "seating-a-piece-by-wiggle-search"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
SLOT = np.array([0.45, 0.05, 0.03])
THICK = 0.002
CLEAR = 0.0003
PARTIAL = 0.0006
YAW_TOL = 0.8
OFFSET = np.array([0.003, -0.002, -0.0095])
START = np.array([0.40, 0.10, 0.12])
PIECE_R = 0.010
TIP = 0.0102


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["seat_piece"].module


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


def _rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def _yaw(r):
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


class World:
    """A YAM-like hand holding a piece over its slot (see the module docstring)."""

    def __init__(
        self,
        true_dxy=(0.0, 0.0),
        true_dyaw=0.0,
        partial=True,
        absent=False,
        camera=True,
        joints=True,
        ik_ok=True,
        refuse=None,
        gripper_fraction=0.02,
        top_offset=None,
        stuck_lift=False,
        efforts=None,
        open_side=None,
        push_block=None,
        drift_deg=0.0,
    ):
        self.drift = float(drift_deg)
        self.p = START.copy()
        self.r = _rot(DOWN)
        self.r0 = self.r.copy()
        self.q = np.zeros(6)
        self.true = SLOT[:2] + np.asarray(true_dxy, float)
        self.true_yaw = float(true_dyaw)
        self.partial, self.absent, self.camera = partial, absent, camera
        self.has_joints, self.ik_ok = joints, ik_ok
        self.refuse = refuse or (lambda tool, p: None)
        self.gripper_fraction = gripper_fraction
        self.top_offset, self.stuck_lift, self.efforts = top_offset, stuck_lift, efforts
        self.v = self.r.T @ OFFSET  # the piece in the hand's frame
        self.held = not absent
        self.top_z = None
        self.moves = []
        self.down_moves = 0
        # The side the slot opens to (a unit x / y vector toward the neighbours): the
        # mat is clear behind the slot along it, and a piece sliding along it stops at
        # the slot. push_block: a free piece pushed along it stops this far short.
        self.s = None if open_side is None else np.array([*open_side, 0.0], float)
        self.push_block = push_block
        self.closed = False
        self.free = None  # a released piece's reference point
        self.pressed_at = []  # where closed fingertips came down on its top

    def tilt(self, r=None):
        r = self.r if r is None else r
        return math.degrees(math.acos(max(-1.0, min(1.0, float((r @ self.r0.T)[2, 2])))))

    def along(self, xy):
        return float(np.dot(np.asarray(xy)[:2] - self.true, self.s[:2]))

    # -- the piece
    def yaw(self):
        return _yaw(self.r @ self.r0.T)

    def piece(self, p=None, r=None):
        p = self.p if p is None else p
        r = self.r if r is None else r
        return p + r @ self.v

    def floor(self, xy, yaw):
        err = float(np.linalg.norm(np.asarray(xy) - self.true))
        dyaw = abs(yaw - self.true_yaw)
        if err <= CLEAR and dyaw <= YAW_TOL:
            return SLOT[2]
        if self.s is not None and self.along(xy) < 0.0:
            perp = np.asarray(xy) - self.true - self.along(xy) * self.s[:2]
            if float(np.linalg.norm(perp)) <= CLEAR:
                return SLOT[2]
        if self.partial and err <= PARTIAL and dyaw <= YAW_TOL:
            return SLOT[2] + 0.5 * THICK
        return SLOT[2] + THICK

    def settle(self):
        """Hold the hand up so a held piece stays over its floor; True if held up."""
        if not self.held:
            return False
        b = self.piece()
        tilt = self.tilt()
        if tilt > 1.0:
            # Tilted: the leading edge (PIECE_R ahead) touches the mat; sliding along
            # the open side stops at the slot.
            held_up = False
            edge = b[2] - PIECE_R * math.sin(math.radians(tilt))
            if edge < SLOT[2] - 1e-12:
                self.p[2] += SLOT[2] - edge
                held_up = True
            if self.s is not None and self.along(b) > 0.0:
                self.p -= self.s * self.along(b)
                held_up = True
            return held_up
        f = self.floor(b[:2], self.yaw())
        if b[2] < f - 1e-12:
            if self.stuck_lift:
                return True
            self.p[2] += f - b[2]
            return True
        return False

    # -- moves
    def _move(self, tool):
        def go(pose, **kw):
            p = np.array([pose["position"][k] for k in "xyz"], float)
            self.moves.append((tool, p.copy(), kw))
            why = self.refuse(tool, p)
            if why == "raise":
                raise RuntimeError("unreachable pose")
            if why:
                return {"outcome": "unreachable", "reached": False, "reason": why}
            if (
                self.stuck_lift
                and self.held
                and p[2] > self.p[2]
                and self.piece()[2] < SLOT[2] + THICK
            ):
                return {"outcome": "timeout", "reached": False}
            r = _rot(pose["rotation"])
            if not self.has_joints:
                self.r = r
            elif self.drift and "tolerance" not in kw:
                self.r = _rz(self.drift) @ self.r  # a long move with the roll left free creeps
            old = self.p.copy()
            self.p = p
            if self.closed and self.free is not None and self.fingertips(old, p):
                return {"outcome": "timeout", "reached": False}
            if self.settle():
                return {"outcome": "timeout", "reached": False}
            return {"outcome": "reached", "reached": True}

        return go

    def free_top(self):
        on = float(np.linalg.norm(self.free[:2] - self.true)) <= CLEAR
        return SLOT[2] + (THICK if on or self.s is None or self.along(self.free) < 0 else 2 * THICK)

    def fingertips(self, old, new):
        """Closed fingertips TIP below the hand against a released piece (a disc of
        PIECE_R): from above they stop on its top, low down they push it along the
        open side until it reaches the slot (or push_block short of it). True if
        the hand was held back."""
        tip, was = new - [0, 0, TIP], old - [0, 0, TIP]
        top = self.free_top()
        inside = float(np.linalg.norm(tip[:2] - self.free[:2])) < PIECE_R
        if inside and was[2] >= top - 1e-9 and tip[2] < top:
            self.p[2] += top - tip[2]
            self.pressed_at.append(self.p[:2].copy())
            return True
        if inside and tip[2] < top and self.s is not None:
            d = float(np.dot(tip[:2] - self.free[:2], self.s[:2]))
            self.free[:2] += self.s[:2] * (d + PIECE_R)
            stop = -(self.push_block or 0.0)
            over = self.along(self.free) - stop
            if over > 0.0:
                self.free[:2] -= self.s[:2] * over
                self.p -= self.s * over
                return True
        return False

    def _joints(self, joint_config, **kw):
        q = np.asarray(joint_config["positions"], float)
        turn = q[-1] - self.q[-1]
        self.q = q
        self.moves.append(("robot.move_to_joints", self.p.copy(), {}))
        self.r = _rz(-math.degrees(turn)) @ self.r
        self.settle()
        return {"outcome": "reached", "reached": True}

    def _open(self, **kw):
        if self.held:
            b = self.piece()
            self.top_z = (
                b[2] + THICK if self.top_offset is None else SLOT[2] + THICK + self.top_offset
            )
            self.free = b.copy()
        self.held = False
        self.closed = False
        return {"position": 1.0}

    def _close(self, **kw):
        self.closed = True
        return {"position": 0.0}

    # -- sensing
    def pose(self):
        return {
            "position": dict(zip("xyz", map(float, self.p), strict=True)),
            "rotation": _quat(self.r),
        }

    def depth(self, cam):
        h, w = 48, 64
        if self.top_z is None:
            return np.zeros((h, w), np.float32)
        # The camera looks along the tool axis (straight down): the top is a plane.
        return np.full((h, w), float(cam[2, 3] - self.top_z), np.float32)

    def observation(self, **kw):
        cams = []
        if self.camera:
            cams.append(
                {
                    "name": "left_eye_in_hand",
                    "pose": self.pose(),
                    "intrinsics": [[40.0, 0.0, 32.0], [0.0, 40.0, 24.0], [0.0, 0.0, 1.0]],
                    "depth": self.depth(np.block([[self.r, self.p[:, None]], [0, 0, 0, 1]])),
                }
            )
        js = {"positions": self.q.tolist()} if self.has_joints else {}
        if self.efforts is not None:
            js["efforts"] = self.efforts(self)
        return {"arms": [{"ee_pose": self.pose(), "joint_state": js}], "cameras": cams}

    def ctx(self):
        return FakeContext(
            {
                "robot.get_ee_pose": lambda **kw: {"pose": self.pose()},
                "robot.get_observation": self.observation,
                "robot.get_gripper": lambda **kw: {"position": self.gripper_fraction},
                "robot.describe_gripper": {"width_fit": {"at_open_m": 0.094, "at_closed_m": 0.0}},
                "robot.go_to_pose": self._move("robot.go_to_pose"),
                "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
                "robot.move_to_joints": self._joints,
                "robot.solve_ik": lambda **kw: {
                    "success": self.ik_ok,
                    "within_servo_tolerance": self.ik_ok,
                },
                "robot.open_gripper": self._open,
                "robot.close_gripper": self._close,
            }
        )


def slot(dx=0.0, dy=0.0):
    return dict(zip("xyz", map(float, SLOT + [dx, dy, 0.0]), strict=True))


def held():
    return dict(zip("xyz", map(float, OFFSET), strict=True))


def run(script, world, **kw):
    kw.setdefault("held_offset", held())
    return script.run(world.ctx(), slot_position=kw.pop("slot_position", slot()), **kw)


# ------------------------------------------------------------- the search order


def test_search_order_starts_at_the_estimate_and_goes_outwards(script):
    order = script.search_order(0.0025, 0.0004, 4.0, 1.0, 0.001, 2.0)
    assert order[0] == (0.0, 0.0, 0.0)
    cost = [round((math.hypot(x, y) / 0.001) ** 2 + (a / 2.0) ** 2, 6) for x, y, a in order]
    assert cost == sorted(cost)
    assert {a for _, _, a in order} == {0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0}
    assert max(math.hypot(x, y) for x, y, _ in order) <= 0.0025 + 1e-9
    # The six hexagonal neighbours (r = pitch, cost 0.16) come before +-1 deg (cost 0.25).
    assert all(math.isclose(math.hypot(x, y), 0.0004) and a == 0 for x, y, a in order[1:7])
    assert order[7][2] == 1.0 and order[8][2] == -1.0


def test_search_order_without_yaw_or_grid(script):
    assert script.search_order(0.0, 0.0004, 0.0, 0.0, 0.001, 2.0) == [(0.0, 0.0, 0.0)]
    only_yaw = script.search_order(0.0, 0.0, 2.0, 1.0, 0.001, 2.0)
    assert [a for _, _, a in only_yaw] == [0.0, 1.0, -1.0, 2.0, -2.0]


# ------------------------------------------------------------- the happy paths


def test_on_the_slot_it_drops_at_once_and_is_seated(script):
    world = World()
    out = run(script, world)
    assert out["route"] == "seated", out["detail"]
    assert out["released"] is True
    assert out["probes"] == 1
    assert out["best_height"] == pytest.approx(0.0, abs=1e-6)
    assert out["verified_by"] == "wrist_depth" and out["samples"] >= 5
    assert out["flush_error"] == pytest.approx(0.0, abs=1e-6)
    assert out["seated_position"]["x"] == pytest.approx(SLOT[0], abs=1e-6)
    assert out["seated_position"]["y"] == pytest.approx(SLOT[1], abs=1e-6)
    # The fine moves carry the servo knobs; the hover does not.
    fine = [m for m in world.moves if m[2]]
    assert fine and all(m[2] == {"tolerance": 0.0002, "max_steps": 90} for m in fine)
    assert out["moves"] > 0 and out["elapsed_s"] >= 0.0


def test_after_the_release_the_closed_fingertips_push_the_top_down(script):
    world = World()
    out = run(script, world, push_down_offset={"x": 0.0, "y": 0.004, "z": 0.0})
    assert out["route"] == "seated", out["detail"]
    assert world.closed
    assert "pushed down, its top met 2.00 mm" in out["detail"]
    assert world.pressed_at
    assert world.pressed_at[0] == pytest.approx(SLOT[:2] + [0.0, 0.004], abs=1e-6)
    skipped = World()
    out = run(script, skipped, push_down=False)
    assert out["route"] == "seated" and not skipped.closed
    assert "pushed down" not in out["detail"]


def test_a_lateral_error_is_found_by_the_probes(script):
    world = World(true_dxy=(0.0008, 0.0))
    out = run(script, world, scoot=False)
    assert out["route"] == "seated", out["detail"]
    assert out["probes"] > 1
    assert "dropped" in out["detail"]
    assert out["offset_found"]["x"] == pytest.approx(0.0008, abs=CLEAR)
    assert out["offset_found"]["y"] == pytest.approx(0.0, abs=CLEAR)
    assert out["seated_position"]["x"] == pytest.approx(SLOT[0] + 0.0008, abs=CLEAR)
    assert out["resting_height"] == pytest.approx(THICK, abs=0.0005)


def test_a_yaw_error_is_found_by_turning_the_last_joint(script):
    world = World(true_dyaw=2.0, partial=False)
    out = run(script, world, scoot=False, max_probes=80)
    assert out["route"] == "seated", out["detail"]
    assert out["yaw_found_deg"] == pytest.approx(2.0, abs=YAW_TOL)
    assert any(m[0] == "robot.move_to_joints" for m in world.moves)
    # The piece, not the hand, is kept over the slot while the hand turns.
    p = np.array([out["seated_position"][k] for k in "xy"])
    assert np.linalg.norm(p - SLOT[:2]) <= CLEAR


def test_the_given_yaw_error_is_turned_out_before_the_descent(script):
    world = World(true_dyaw=3.0)
    out = run(script, world, yaw_error_deg=3.0)
    assert out["route"] == "seated", out["detail"]
    assert out["probes"] == 1
    assert world.yaw() == pytest.approx(3.0, abs=0.3)


def test_without_joint_readings_the_yaw_is_a_rotation_command(script):
    world = World(true_dyaw=1.0, joints=False, partial=False)
    out = run(script, world, scoot=False)
    assert out["route"] == "seated", out["detail"]
    assert not any(m[0] == "robot.move_to_joints" for m in world.moves)


def test_partly_in_switches_to_the_scoot_which_finishes_it(script):
    # 0.5 mm off: the probe grid point at 0.4 mm leaves the piece 0.1 mm out... the
    # estimate itself stops partly in, and the scoot slides it home.
    world = World(true_dxy=(0.0, 0.0005))
    out = run(script, world)
    assert out["route"] == "seated", out["detail"]
    assert "dropped in the scoot" in out["detail"]
    assert out["probes"] > 2


def test_a_wrist_creeping_in_yaw_is_re_aimed_and_turned_back(script):
    # Every long move turns the hand 1.5 deg: the piece (3.6 mm off the tool centre)
    # swings and its yaw leaves the slot's; positions use the yaw read back, probes re-turn.
    world = World(drift_deg=1.5, partial=False)
    out = run(script, world)
    assert out["route"] == "seated", out["detail"]
    p = np.array([out["seated_position"][k] for k in "xy"])
    assert np.linalg.norm(p - SLOT[:2]) <= CLEAR
    assert abs(out["yaw_found_deg"]) <= YAW_TOL
    assert out["probes"] == 1


def test_efforts_fire_contact_where_published(script):
    def efforts(w):
        b = w.piece()
        pushing = w.held and b[2] <= w.floor(b[:2], w.yaw()) + 1e-9
        return [0.0, 3.0 if pushing else 0.0, 0.0, 0.0, 0.0, 0.0]

    world = World(efforts=efforts)
    out = run(script, world)
    assert out["route"] == "seated", out["detail"]


# ------------------------------------------------------------- the other exits


def test_nothing_within_the_search_is_not_seated_and_kept(script):
    world = World(true_dxy=(0.006, 0.0))
    out = run(script, world, max_probes=12)
    assert out["route"] == "not_seated", out["detail"]
    assert out["released"] is False
    assert "max_probes (12)" in out["detail"]
    assert "still holding" in out["detail"]
    assert out["best_height"] == pytest.approx(THICK, abs=0.0005)
    tool, lift, _ = world.moves[-1]
    assert tool == "robot.go_to_pose_cartesian"
    assert lift[2] == pytest.approx(SLOT[2] - OFFSET[2] + 0.015, abs=1e-9)


def test_released_anyway_when_asked_reads_high_and_is_not_seated(script):
    world = World(true_dxy=(0.006, 0.0))
    out = run(script, world, max_probes=6, scoot=False, release_when_not_seated=True)
    assert out["route"] == "not_seated", out["detail"]
    assert out["released"] is True
    assert out["flush_error"] == pytest.approx(THICK, abs=1e-6)


def test_partly_in_and_not_finished_is_jammed(script):
    world = World(true_dxy=(0.0, 0.0005))
    out = run(script, world, scoot=False, pitch=0.0, yaw_step_deg=0.0)
    assert out["route"] == "jammed", out["detail"]
    assert out["released"] is False
    assert out["best_height"] == pytest.approx(0.5 * THICK, abs=0.0005)


def test_a_piece_that_cannot_be_lifted_out_is_jammed(script):
    world = World(true_dxy=(0.006, 0.0), stuck_lift=True)
    out = run(script, world, max_probes=3)
    assert out["route"] == "jammed", out["detail"]
    assert "stuck" in out["detail"]


def test_jaws_at_their_stop_are_piece_lost_and_nothing_moves(script):
    world = World(gripper_fraction=0.0)
    out = run(script, world)
    assert out["route"] == "piece_lost", out["detail"]
    assert world.moves == []


def test_a_descent_meeting_nothing_is_piece_lost(script):
    world = World(absent=True)
    out = run(script, world)
    assert out["route"] == "piece_lost", out["detail"]
    assert out["released"] is False
    assert "not in the hand" in out["detail"]


def test_a_flush_check_finding_the_top_high_is_not_seated(script):
    world = World(top_offset=0.0015)
    out = run(script, world)
    assert out["route"] == "not_seated", out["detail"]
    assert out["released"] is True
    assert out["flush_error"] == pytest.approx(0.0015, abs=1e-6)


def test_no_top_at_the_slot_after_release_is_piece_lost(script):
    world = World(top_offset=-THICK)
    out = run(script, world)
    assert out["route"] == "piece_lost", out["detail"]


def test_no_wrist_camera_is_unverified(script):
    world = World(camera=False)
    out = run(script, world)
    assert out["route"] == "unverified", out["detail"]
    assert out["released"] is True


def test_verify_false_is_unverified(script):
    world = World()
    out = run(script, world, verify=False)
    assert out["route"] == "unverified"
    assert "verify=false" in out["detail"]


def test_out_of_reach_is_unreachable_before_moving(script):
    world = World(ik_ok=False)
    out = run(script, world)
    assert out["route"] == "unreachable", out["detail"]
    assert world.moves == []


def test_a_refused_hover_is_unreachable(script):
    world = World(refuse=lambda tool, p: "no" if tool == "robot.go_to_pose" else None)
    out = run(script, world)
    assert out["route"] == "unreachable", out["detail"]
    assert out["released"] is False


def test_a_raised_refusal_during_the_search_is_unreachable_and_lifted_back(script):
    calls = {"n": 0}

    def refuse(tool, p):
        if tool == "robot.go_to_pose" and p[2] < SLOT[2] - OFFSET[2] + THICK:
            calls["n"] += 1
            if calls["n"] > 12:
                return "raise"
        return None

    world = World(true_dxy=(0.006, 0.0), refuse=refuse)
    out = run(script, world)
    assert out["route"] == "unreachable", out["detail"]
    assert world.moves[-1][0] == "robot.go_to_pose_cartesian"


def test_a_descent_stopping_above_the_neighbours_is_not_seated(script):
    world = World()
    # The slot is given 6 mm too low: the fast descent stops on the neighbours.
    out = run(script, world, slot_position=slot() | {"z": float(SLOT[2] - 0.008)})
    assert out["route"] == "not_seated", out["detail"]
    assert "above the neighbours" in out["detail"]


# ------------------------------------------------------------- tilt_first


def test_tilt_first_slides_to_the_seam_and_drops_on_levelling(script):
    # The slot is 1.2 mm further along the seam than given: tilted, the piece slides there.
    world = World(true_dxy=(0.0012, 0.0), open_side=(1.0, 0.0), joints=False, partial=False)
    out = run(script, world, approach="tilt_first", seam_direction={"x": 1.0, "y": 0.0, "z": 0.0})
    assert out["route"] == "seated", out["detail"]
    assert out["probes"] == 1
    assert out["seated_position"]["x"] == pytest.approx(SLOT[0] + 0.0012, abs=CLEAR)
    tilted = [m for m in world.moves if m[0] == "robot.go_to_pose_cartesian"]
    assert tilted, "the tilted descent is a Cartesian move"
    assert world.tilt() < 0.5


def test_tilt_first_without_reaching_the_seam_falls_back_to_the_search(script):
    # Off across the seam: the slide meets the slot line but 0.8 mm aside; the probes find it.
    world = World(true_dxy=(0.0, 0.0008), open_side=(1.0, 0.0), joints=False, partial=False)
    out = run(
        script,
        world,
        approach="tilt_first",
        seam_direction={"x": 1.0, "y": 0.0, "z": 0.0},
        scoot=False,
    )
    assert out["route"] == "seated", out["detail"]
    assert out["probes"] > 1


def test_tilt_first_needs_a_seam_direction(script):
    with pytest.raises(ValueError, match="seam_direction"):
        script.run(World().ctx(), slot_position=slot(), approach="tilt_first")
    with pytest.raises(ValueError, match="approach"):
        script.run(World().ctx(), slot_position=slot(), approach="sideways")


# ------------------------------------------------------------- offset_push


def test_offset_push_sets_down_short_pushes_in_and_presses(script):
    world = World(open_side=(0.0, 1.0))
    out = run(script, world, approach="offset_push", seam_direction={"x": 0.0, "y": 1.0, "z": 0.0})
    assert out["route"] == "seated", out["detail"]
    assert out["released"] is True
    assert world.closed
    assert world.free[1] == pytest.approx(SLOT[1], abs=1e-6)
    assert out["stop_height"] == pytest.approx(0.0, abs=0.0005)
    assert "pushed 4.0 mm" in out["detail"]


def test_offset_push_stalling_short_is_jammed(script):
    world = World(open_side=(0.0, 1.0), push_block=0.002)
    out = run(script, world, approach="offset_push", seam_direction={"x": 0.0, "y": 1.0, "z": 0.0})
    assert out["route"] == "jammed", out["detail"]
    assert out["released"] is True
    assert "short of the slot" in out["detail"]


def test_offset_push_onto_a_blocked_spot_is_not_seated_and_kept(script):
    world = World()  # nothing clear behind the slot: the set-down lands on the neighbours
    out = run(script, world, approach="offset_push", seam_direction={"x": 0.0, "y": 1.0, "z": 0.0})
    assert out["route"] == "not_seated", out["detail"]
    assert out["released"] is False
    assert "not clear" in out["detail"]


def test_example_declares_every_route(script):
    import json
    from pathlib import Path

    example = Path(script.__file__).parents[1] / "examples" / "canonical_subgraph.json"
    graph = json.loads(example.read_text())
    declared = set(graph["exit"]["success_values"]) | {graph["on_error"]}
    assert set(script.ROUTES) == declared


def test_zero_thickness_is_refused(script):
    with pytest.raises(ValueError):
        script.run(World().ctx(), slot_position=slot(), thickness=0.0)
