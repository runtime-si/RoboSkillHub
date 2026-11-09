"""disassembling-a-brick-by-twist-lever: FakeContext tests against a small brick-on-studs world.

The fake hand grips a seated brick rigidly (robot.set_grip over it). The clutch is a
rule, not physics: each engaged stud resists in proportion to how far its socket has
risen over the seat (up to one stud height), and lets go for good past it. The hand
lands short of its command by LAG_PER_UNIT per unit of that resistance; a resistance
over the grip's limit slides the brick out of the jaws (it drops back onto its studs),
and a stuck world does not let a loaded brick move at all. A lever about a bottom edge
pops the far row before the near one, so its peak resistance (about 2.7 of 4 on a 2x2)
is under a straight pull's (4): with grip_limit 3 the lever lifts and the pull slips.
The wrist camera ray-casts the seat: the brick's top while it is on the seat, the plate
(or the brick under it, or a moved neighbour) once it is gone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "disassembling-a-brick-by-twist-lever"
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
SEAT = np.array([0.45, 0.10, 0.05])
START = np.array([0.40, 0.10, 0.20])
SYSTEMS = {"duplo": (0.016, 0.0045, 0.0192), "lego": (0.008, 0.0017, 0.0096)}
PLAY = 0.0001
LAG_PER_UNIT = 0.00035
SPAN = 0.094


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["disassemble_brick"].module


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


def _axis_angle(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _tf(p, r):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, p
    return m


class World:
    """A YAM-like hand over a brick seated on a baseplate (see the module docstring)."""

    def __init__(
        self,
        system="duplo",
        studs=(2, 2),
        yaw_deg=0.0,
        grip_limit=3.0,
        stuck=False,
        lag_per_unit=LAG_PER_UNIT,
        effort_gain=None,
        absent=False,
        camera=True,
        ik=None,
        refuse=None,
        set_grip=True,
        grip_fails=False,
        block_z=None,
        neighbour_moves=False,
        under_comes=False,
    ):
        self.pitch, self.stud_h, self.body_h = SYSTEMS[system]
        self.studs = studs
        a = math.radians(yaw_deg)
        self.bx = np.array([math.cos(a), math.sin(a), 0.0])
        self.by = np.array([-self.bx[1], self.bx[0], 0.0])
        self.lens = (studs[0] * self.pitch - 2 * PLAY, studs[1] * self.pitch - 2 * PLAY)
        self.seat_tf = _tf(SEAT, np.column_stack([self.bx, self.by, [0.0, 0.0, 1.0]]))
        self.sockets = [
            np.array(
                [(i - (studs[0] - 1) / 2) * self.pitch, (j - (studs[1] - 1) / 2) * self.pitch, 0, 1]
            )
            for i in range(studs[0])
            for j in range(studs[1])
        ]
        self.engaged = [True] * len(self.sockets)
        self.grip_limit, self.stuck, self.lag_per_unit = grip_limit, stuck, lag_per_unit
        self.effort_gain = effort_gain
        self.on_seat = not absent
        self.camera, self.ik = camera, ik or (lambda p, r: True)
        self.refuse = refuse or (lambda tool, p, r: None)
        self.serve_set_grip, self.grip_fails, self.block_z = set_grip, grip_fails, block_z
        self.neighbour_moves, self.under_comes = neighbour_moves, under_comes
        self.p, self.r = START.copy(), _rot(DOWN)
        self.held = False
        self.hand_brick = None
        self.width = SPAN
        self.load = 0.0
        self.peak_load = 0.0
        self.moves = []

    # -- the brick
    def brick_tf(self, p=None, r=None):
        if self.held:
            return _tf(self.p if p is None else p, self.r if r is None else r) @ self.hand_brick
        return self.seat_tf

    def rises(self, tf):
        return [float((tf @ s)[2] - SEAT[2]) for s in self.sockets]

    def resistance(self, rises):
        return sum(
            min(1.0, max(0.0, h / self.stud_h))
            for h, e in zip(rises, self.engaged, strict=True)
            if e
        )

    def drop_back(self):
        self.held, self.hand_brick = False, None
        self.width = self.grip_width - self.squeeze if self.grip_width else 0.0

    # -- moves
    def _go(self, tool, p_goal, r_goal):
        if self.block_z is not None and not self.held and p_goal[2] < self.block_z:
            self.p, self.r = np.array([p_goal[0], p_goal[1], self.block_z]), r_goal
            return {"outcome": "timeout", "reached": False}
        if not self.held or not any(self.engaged):
            self.p, self.r = p_goal.copy(), r_goal.copy()
            self.load = 0.0
            return {"outcome": "reached", "reached": True}
        rises = self.rises(self.brick_tf(p_goal, r_goal))
        if min(rises) < -1e-6:  # into the plate
            return {"outcome": "timeout", "reached": False}
        load = self.resistance(rises)
        if self.stuck and load > 0.05:
            self.load = max(self.load, load)
            return {"outcome": "timeout", "reached": False}
        self.peak_load = max(self.peak_load, load)
        if load > self.grip_limit:
            self.drop_back()
            self.p, self.r, self.load = p_goal.copy(), r_goal.copy(), 0.0
            return {"outcome": "reached", "reached": True}
        for i, h in enumerate(rises):
            if h > self.stud_h:
                self.engaged[i] = False
        if not any(self.engaged):
            self.on_seat = False
        way = p_goal - self.p
        way = way / (np.linalg.norm(way) or 1.0)
        lag = self.lag_per_unit * load
        self.p, self.r, self.load = p_goal - way * lag, r_goal.copy(), load
        return {"outcome": "timeout" if lag > 3e-4 else "reached", "reached": lag <= 3e-4}

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

    grip_width = 0.0
    squeeze = 0.0

    def _grip(self, width, squeeze):
        self.grip_width, self.squeeze = width, squeeze
        self.grip_axis = self.r[:, 1].copy()
        b = self.seat_tf[:3, 3]
        over = (
            self.on_seat
            and np.linalg.norm(self.p[:2] - b[:2]) < 0.003
            and self.p[2] < SEAT[2] + self.body_h + 0.0102
        )
        if over and not self.grip_fails:
            self.held = True
            self.hand_brick = np.linalg.inv(_tf(self.p, self.r)) @ self.seat_tf
            across = self.lens[0] if abs(float(self.r[:, 1] @ self.bx)) > 0.5 else self.lens[1]
            self.width = across - 0.0002
        else:
            self.width = max(0.0, width - squeeze)

    def _set_grip(self, object_width_m, squeeze_m, **kw):
        self._grip(object_width_m, squeeze_m)
        return {"outcome": "reached"}

    def _close(self, **kw):
        self._grip(0.0, 0.0)
        return {"position": self.width / SPAN}

    def _open(self, **kw):
        if self.held and not self.on_seat:
            self.on_seat = False  # dropped somewhere off the seat
        self.held, self.hand_brick = False, None
        self.width = SPAN
        return {"position": 1.0}

    # -- sensing
    def pose(self):
        return {
            "position": dict(zip("xyz", map(float, self.p), strict=True)),
            "rotation": _quat(self.r),
        }

    def camera_pose(self):
        tc = np.eye(4)
        tc[:3, :3] = _axis_angle([1, 0, 0], math.radians(-30))
        tc[:3, 3] = [0.0, -0.06, -0.07]
        return _tf(self.p, self.r) @ tc

    def in_box(self, hit, lo, hi):
        d = hit - SEAT
        u, v = d @ self.bx, d @ self.by
        return (u >= lo[0]) & (u <= hi[0]) & (v >= lo[1]) & (v <= hi[1])

    def depth(self, k, cam):
        h, w = 240, 320
        u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
        d = np.stack([(u - k[0][2]) / k[0][0], (v - k[1][2]) / k[1][1], np.ones_like(u)], -1)
        ray = d @ cam[:3, :3].T
        c = cam[:3, 3]

        def plane(z):
            t = (z - c[2]) / ray[..., 2]
            return t, c + ray * t[..., None]

        hx, hy = self.lens[0] / 2, self.lens[1] / 2
        out, _ = plane(SEAT[2])
        if not self.on_seat and self.under_comes:
            t, hit = plane(SEAT[2] - self.body_h)
            out = np.where(self.in_box(hit, (-hx, -hy), (hx, hy)), t, out)
        if not self.on_seat and self.neighbour_moves:
            t, hit = plane(SEAT[2] + 0.006)
            out = np.where(self.in_box(hit, (hx, -hy), (hx + 2 * self.pitch, hy)), t, out)
        if self.on_seat:
            t, hit = plane(SEAT[2] + self.body_h)
            out = np.where(self.in_box(hit, (-hx, -hy), (hx, hy)), t, out)
        return np.where(out > 0, out, 0.0).astype(np.float32)

    def observation(self, **kw):
        k = [[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]]
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
                }
            )
        js = {"positions": [0.0] * 6}
        if self.effort_gain is not None:
            js["efforts"] = [0.1, 0.2, self.effort_gain * self.load, 0.0, 0.0, 0.0]
        return {"arms": [{"ee_pose": self.pose(), "joint_state": js}], "cameras": cams}

    def ctx(self):
        tools = {
            "robot.get_ee_pose": lambda **kw: {"pose": self.pose()},
            "robot.get_observation": self.observation,
            "robot.get_gripper": lambda **kw: {"position": self.width / SPAN},
            "robot.describe_gripper": {"width_fit": {"at_open_m": SPAN, "at_closed_m": 0.0}},
            "robot.go_to_pose": self._move("robot.go_to_pose"),
            "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
            "robot.move_to_joints": lambda **kw: {"outcome": "reached"},
            "robot.solve_ik": lambda pose, **kw: {
                "success": self.ik(
                    np.array([pose["position"][k] for k in "xyz"]), _rot(pose["rotation"])
                ),
                "within_servo_tolerance": True,
            },
            "robot.open_gripper": self._open,
            "robot.close_gripper": self._close,
        }
        if self.serve_set_grip:
            tools["robot.set_grip"] = self._set_grip
        return FakeContext(tools)

    # -- for the asserts
    def tilts(self):
        """Signed tilt [deg] of every fine move's approach axis, about the brick's y axis."""
        return [
            math.degrees(math.atan2(-float(r[:, 2] @ self.bx), -float(r[:, 2][2])))
            for tool, _, r, kw in self.moves
            if tool == "robot.go_to_pose" and "tolerance" in kw
        ]


def seat(dx=0.0, dy=0.0):
    return dict(zip("xyz", map(float, SEAT + [dx, dy, 0.0]), strict=True))


def run(script, world, **kw):
    kw.setdefault("fallbacks", "")
    return script.run(world.ctx(), seat_position=seat(), **kw)


# ---------------------------------------------------------------- the lever


def test_lever_pops_the_far_row_first_lifts_and_reads_the_empty_seat(script):
    world = World()
    out = run(script, world)
    assert out["route"] == "removed", out["detail"]
    assert out["strategy"] == "lever" and out["attempts"] == 1
    assert out["pivot_side"] == "+x"
    assert out["twist_deg"] == pytest.approx(14.0, abs=0.5)
    assert out["holding"] is True and world.held and not world.on_seat
    assert out["verified_by"] == "wrist_depth" and out["samples"] >= 3
    assert out["top_height_change"] == pytest.approx(-0.0192, abs=0.0015)
    assert out["neighbour_change"] < 0.001
    assert out["release_count"] >= 1 and out["releases"].startswith("lever at")
    assert out["peak_lag"] > 0.0005
    assert world.peak_load < 3.0
    assert max(world.tilts()) == pytest.approx(14.0, abs=0.1)


def test_lever_steps_are_fine_moves_about_the_bottom_edge(script):
    world = World()
    run(script, world)
    fine = [m for m in world.moves if m[0] == "robot.go_to_pose" and "tolerance" in m[3]]
    assert fine and all(m[3]["tolerance"] == script.FINE_TOLERANCE for m in fine)
    tilts = world.tilts()
    lever = [m[1] for m, t in zip(fine, tilts, strict=True) if t > 0.1][:7]
    steps = np.diff([0.0, *[t for t in tilts if t > 0.1][:7]])
    assert len(lever) == 7 and np.all(steps <= script.TWIST_STEP_DEG + 1e-6)
    # The pivot (+x bottom edge) stays put: the commanded tool centre keeps its distance to it.
    pivot = SEAT + world.bx * world.lens[0] / 2
    grasp = [m[1] for m, t in zip(fine, tilts, strict=True) if abs(t) < 1e-6][-1]
    r0 = float(np.linalg.norm(grasp - pivot))
    assert all(abs(float(np.linalg.norm(p - pivot)) - r0) < 1e-6 for p in lever)


def test_lift_follows_the_brick_axis_or_the_vertical(script):
    for along, check in (("brick", lambda d: d[0] > 0.1), ("vertical", lambda d: abs(d[0]) < 1e-6)):
        world = World()
        out = run(script, world, lift_along=along)
        assert out["route"] == "removed", out["detail"]
        fine = [m for m in world.moves if m[0] == "robot.go_to_pose" and "tolerance" in m[3]]
        lift = [m[1] for m, t in zip(fine, world.tilts(), strict=True) if abs(t - 14.0) < 0.1]
        d = lift[-1] - lift[0]
        assert d[2] > 0.01 and check(d / np.linalg.norm(d)), (along, d)


# ---------------------------------------------------------------- pry vs pull


def test_straight_pull_slips_where_the_lever_lifts(script):
    world = World()
    out = run(script, world, strategy="straight")
    assert out["route"] == "slipped", out["detail"]
    assert "the grip let go" in out["detail"]
    assert out["release_count"] == 0 and out["releases"] == ""
    assert world.on_seat and all(world.engaged) and not world.held
    assert world.peak_load > 3.0


def test_a_slip_regrasps_and_the_lever_fallback_removes(script):
    world = World()
    out = run(script, world, strategy="straight", fallbacks="lever")
    assert out["route"] == "removed", out["detail"]
    assert out["strategy"] == "lever" and out["attempts"] == 2
    assert "straight: slipped" in out["attempt_log"]
    assert "straight" not in out["releases"] and out["release_count"] >= 1
    ctx_grips = [m for m in world.moves if m[0] == "robot.go_to_pose_cartesian"]
    assert ctx_grips, "rose back to the hover before the regrasp"


def test_no_regrasp_stops_at_the_first_slip(script):
    world = World()
    out = run(script, world, strategy="straight", fallbacks="lever", regrasp_on_slip=False)
    assert out["route"] == "slipped" and out["attempts"] == 1


def test_strong_clutch_slips_the_lever_too(script):
    world = World(grip_limit=2.0)
    out = run(script, world)
    assert out["route"] == "slipped", out["detail"]
    assert out["release_count"] == 0 and world.on_seat


def test_wiggle_levers_about_both_edges_then_lifts(script):
    world = World()
    out = run(script, world, strategy="wiggle")
    assert out["route"] == "removed", out["detail"]
    assert out["pivot_side"] == "both"
    tilts = world.tilts()
    assert max(tilts) == pytest.approx(14.0, abs=0.1) and min(tilts) == pytest.approx(
        -14.0, abs=0.1
    )
    assert any(abs(t - 7.0) < 0.1 for t in tilts) and any(abs(t + 7.0) < 0.1 for t in tilts)


def test_a_brick_that_holds_is_stuck_and_left_on_the_seat(script):
    world = World(stuck=True)
    out = script.run(world.ctx(), seat_position=seat())  # default fallback: straight
    assert out["route"] == "stuck", out["detail"]
    assert out["attempts"] == 2
    assert "lever: stuck" in out["attempt_log"] and "straight: stuck" in out["attempt_log"]
    assert world.on_seat and all(world.engaged)
    assert not world.held and world.width == SPAN
    assert world.p[2] > SEAT[2] + 0.04


def test_joint_effort_marks_the_load_where_the_lag_does_not(script):
    world = World(lag_per_unit=0.0, effort_gain=1.0)
    out = run(script, world)
    assert out["route"] == "removed", out["detail"]
    assert out["peak_lag"] < 1e-6
    assert out["release_count"] >= 1


# ---------------------------------------------------------------- before the grip


def test_an_empty_seat_is_missed_before_anything_is_gripped(script):
    world = World(absent=True)
    out = run(script, world)
    assert out["route"] == "missed", out["detail"]
    assert "no brick" in out["detail"]
    assert not any(m[3].get("tolerance") for m in world.moves)
    assert world.width == SPAN


def test_open_jaws_stopping_high_is_missed(script):
    world = World(block_z=SEAT[2] + 0.04)
    out = run(script, world)
    assert out["route"] == "missed", out["detail"]
    assert "stopped" in out["detail"] and world.width == SPAN


def test_jaws_closing_past_the_width_is_missed(script):
    world = World(grip_fails=True)
    out = run(script, world)
    assert out["route"] == "missed", out["detail"]
    assert "past the brick's width" in out["detail"]
    assert world.width == SPAN and world.on_seat


def test_close_gripper_stands_in_for_set_grip(script):
    world = World(set_grip=False)
    ctx = world.ctx()
    out = script.run(ctx, seat_position=seat(), fallbacks="")
    assert out["route"] == "removed", out["detail"]
    assert ctx.call_count("robot.close_gripper") == 1


# ---------------------------------------------------------------- after the lift


def test_a_neighbour_that_moved_is_neighbour_lifted(script):
    world = World(neighbour_moves=True)
    out = run(script, world)
    assert out["route"] == "neighbour_lifted", out["detail"]
    assert out["neighbour_change"] > 0.002 and out["holding"]


def test_the_brick_under_coming_along_is_neighbour_lifted(script):
    world = World(under_comes=True)
    out = run(script, world)
    assert out["route"] == "neighbour_lifted", out["detail"]
    assert "under it" in out["detail"]
    assert out["top_height_change"] < -1.5 * 0.0192


def test_without_a_wrist_camera_it_is_unverified(script):
    world = World(camera=False)
    out = run(script, world)
    assert out["route"] == "unverified", out["detail"]
    assert "jaws hold something" in out["detail"] and out["verified_by"] == ""
    assert out["release_count"] >= 1


def test_verify_false_is_unverified_without_views(script):
    world = World()
    out = run(script, world, verify=False)
    assert out["route"] == "unverified", out["detail"]
    assert "verify=false" in out["detail"]
    assert not any(
        m[0] == "robot.go_to_pose" and "tolerance" not in m[3] and m[1][2] > 0.2
        for m in world.moves
    )


# ---------------------------------------------------------------- reach and refusals


def test_a_grasp_out_of_reach_is_unreachable(script):
    world = World(ik=lambda p, r: False)
    out = run(script, world)
    assert out["route"] == "unreachable", out["detail"]
    assert world.moves == []


def test_a_lever_out_of_reach_falls_back_to_the_straight_pull(script):
    level = lambda p, r: float(-r[:, 2][2]) > math.cos(math.radians(3))  # noqa: E731
    world = World(ik=level, grip_limit=5.0)
    out = script.run(world.ctx(), seat_position=seat())
    assert out["route"] == "removed", out["detail"]
    assert out["strategy"] == "straight" and out["attempts"] == 1
    assert "neither edge" in out["attempt_log"]
    assert max(abs(t) for t in world.tilts()) < 0.1


@pytest.mark.parametrize("how", ["unreachable", "raise"])
def test_a_refused_lever_step_routes_unreachable(script, how):
    def refuse(tool, p, r):
        return (
            how
            if tool == "robot.go_to_pose" and float(-r[:, 2][2]) < math.cos(math.radians(5))
            else None
        )

    world = World(refuse=refuse)
    out = run(script, world)
    assert out["route"] == "unreachable", out["detail"]
    assert "refused" in out["detail"]
    assert world.on_seat and not world.held


# ---------------------------------------------------------------- geometry


def test_pivot_side_and_close_axis_pick_the_edge(script):
    world = World()
    out = run(script, world, close_axis="y", pivot_side="-")
    assert out["route"] == "removed", out["detail"]
    assert out["pivot_side"] == "-y"


def test_a_lego_2x4_turned_90_deg_levers_across_its_short_side(script):
    world = World(system="lego", studs=(2, 4), yaw_deg=90.0, grip_limit=6.0, lag_per_unit=0.0001)
    out = script.run(
        world.ctx(),
        seat_position=seat(),
        seat_yaw_deg=90.0,
        brick_studs=[2, 4],
        system="lego",
        fallbacks="",
    )
    assert out["route"] == "removed", out["detail"]
    assert out["pivot_side"] in ("+x", "-x")
    assert out["top_height_change"] == pytest.approx(-0.0096, abs=0.0015)
    # The jaws close across the 16 mm side, along the brick's x (world y).
    assert abs(float(world.grip_axis @ world.bx)) > 0.99


def test_long_close_axis_on_a_2x4(script):
    world = World(studs=(2, 4), grip_limit=20.0)
    out = script.run(
        world.ctx(), seat_position=seat(), brick_studs=[2, 4], close_axis="long", fallbacks=""
    )
    assert out["route"] == "removed", out["detail"]
    assert out["pivot_side"] in ("+y", "-y")


@pytest.mark.parametrize(
    "kw",
    [
        {"system": "megablok"},
        {"pivot_side": "left"},
        {"lift_along": "sideways"},
        {"close_axis": "diagonal"},
        {"strategy": "hammer"},
        {"fallbacks": "straight, hammer"},
    ],
)
def test_bad_inputs_raise(script, kw):
    with pytest.raises(ValueError):
        script.run(World().ctx(), seat_position=seat(), **kw)


def test_strategy_chain(script):
    assert script._chain("lever", None) == ["lever", "straight"]
    assert script._chain("straight", None) == ["straight"]
    assert script._chain("wiggle", "lever, straight") == ["wiggle", "lever", "straight"]
    assert script._chain("lever", ["lever", "wiggle"]) == ["lever", "wiggle"]
    assert script._chain("lever", "") == ["lever"]


def test_the_example_declares_every_route(script):
    root = Path(script.__file__).resolve().parents[1]
    example = json.loads((root / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert set(script.ROUTES) == declared
