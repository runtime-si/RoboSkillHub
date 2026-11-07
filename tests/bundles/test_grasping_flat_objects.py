"""grasping-flat-objects: FakeContext tests of the pinch heights, the tilt, the axis order, the gap check and the routing."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "grasping-flat-objects"
SPAN = 0.094
REACH = 0.0102
SUPPORT = 0.10
"""The table's height in these tests [m]."""


def slab(thick=0.008, long=0.060, short=0.040, yaw_deg=0.0, at=(0.40, 0.10)):
    """An oriented box (half-extents) for a slab lying flat on the table, its long side along +x turned by ``yaw_deg``."""
    h = math.radians(yaw_deg) / 2.0
    return {
        "center": {"x": at[0], "y": at[1], "z": SUPPORT + thick / 2.0},
        "extent": {"x": long / 2.0, "y": short / 2.0, "z": thick / 2.0},
        "orientation": {"w": math.cos(h), "x": 0.0, "y": 0.0, "z": math.sin(h)},
    }


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
    """Tool +z down, jaws (tool x) turned ``heading_deg`` about the vertical."""
    h = math.radians(heading_deg) / 2.0
    q = _mul((math.cos(h), 0.0, 0.0, math.sin(h)), (0.0, 1.0, 0.0, 0.0))
    return dict(zip("wxyz", q, strict=True))


def _quat(m: np.ndarray) -> dict[str, float]:
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        k = 2.0 * math.sqrt(1.0 + tr)
        q = (k / 4, (m[2, 1] - m[1, 2]) / k, (m[0, 2] - m[2, 0]) / k, (m[1, 0] - m[0, 1]) / k)
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        k = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = ((m[2, 1] - m[1, 2]) / k, k / 4, (m[0, 1] + m[1, 0]) / k, (m[0, 2] + m[2, 0]) / k)
    elif m[1, 1] >= m[2, 2]:
        k = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 2] - m[2, 0]) / k, (m[0, 1] + m[1, 0]) / k, k / 4, (m[1, 2] + m[2, 1]) / k)
    else:
        k = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[1, 0] - m[0, 1]) / k, (m[0, 2] + m[2, 0]) / k, (m[1, 2] + m[2, 1]) / k, k / 4)
    return dict(zip("wxyz", q, strict=True))


def frame(close_heading_deg=0.0, approach=None, **kw):
    """robot.grasp_frame of a hand whose tool z is the approach and tool x the close axis."""
    a = np.array([approach[k] for k in "xyz"]) if approach else np.array([0.0, 0.0, -1.0])
    a = a / np.linalg.norm(a)
    out = {}
    for key, sign in (("rotation", 1.0), ("rotation_flipped", -1.0)):
        h = math.radians(close_heading_deg)
        u = sign * np.array([math.cos(h), math.sin(h), 0.0])
        u = u - (u @ a) * a
        u = u / np.linalg.norm(u)
        out[key] = _quat(np.column_stack([u, np.cross(a, u), a]))
    return out


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["grasp_flat"].module


class Arm:
    """A hand whose moves land where commanded unless ``refuse`` (a predicate on
    tool and target) says otherwise; ``free_roll`` keeps the hand's heading on
    pose moves (the last joint turns it, the hand's yaw = -q6); ``between``
    gives the jaw gap a close ends at, from the arm."""

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
        self.r = top_down(0.0)
        self.q = np.zeros(6)
        self.refuse = refuse or (lambda tool, target: None)
        self.free_roll = free_roll
        self.between = between or (lambda arm: 0.041)
        self.stop_descent_at = stop_descent_at
        self.stall_lift = stall_lift
        self.slip = slip
        self.ik_ok = ik_ok or (lambda p, rot: True)
        self.gap = SPAN
        self.moves = []
        self.closes = 0
        self.grips = []

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
                self.r = pose["rotation"]
            return {"reached": True, "outcome": "reached"}

        return go

    def joints(self, joint_config, **kwargs):
        q = np.asarray(joint_config["positions"], dtype=float)
        self.moves.append(("robot.move_to_joints", q.copy(), None))
        turn = -(q[-1] - self.q[-1]) / 2.0
        self.r = dict(
            zip(
                "wxyz",
                _mul((math.cos(turn), 0.0, 0.0, math.sin(turn)), tuple(self.r[k] for k in "wxyz")),
                strict=True,
            )
        )
        self.q = q
        return {"reached": True, "outcome": "reached"}

    def close(self, **kwargs):
        self.closes += 1
        self.gap = max(0.0, float(self.between(self)))
        return {"position": self.gap / SPAN}

    def open(self, **kwargs):
        self.gap = SPAN
        return {"position": 1.0}

    def grip(self, width_m, **kwargs):
        self.grips.append(width_m)
        self.gap = width_m
        return {"position": width_m / SPAN}

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
                            "rotation": self.r,
                        }
                    },
                    "robot.get_observation": lambda **kw: {
                        "arms": [{"joint_state": {"positions": self.q.tolist()}}]
                    },
                    "robot.get_gripper": lambda **kw: {"position": self.gap / SPAN},
                    "robot.describe_gripper": {
                        "span_m": SPAN,
                        "width_fit": {"at_open_m": SPAN, "at_closed_m": 0.0},
                        "finger": {"stated": True, "reach_m": REACH, "clearance_m": 0.004},
                    },
                    "robot.grasp_frame": frame,
                    "robot.solve_ik": self.ik,
                    "robot.go_to_pose": self._move("robot.go_to_pose"),
                    "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
                    "robot.move_to_joints": self.joints,
                    "robot.open_gripper": self.open,
                    "robot.close_gripper": self.close,
                    "robot.set_grip": self.grip,
                },
                **{name.replace("__", "."): fn for name, fn in overrides.items()},
            }
        )


def test_level_pinch_across_the_short_side_of_an_8mm_slab(script):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(ctx, object_obb=slab())
    assert out["route"] == "grasped", out["detail"]
    assert out["axis"] == "short"
    assert abs(out["close_heading_deg"]) == pytest.approx(90.0)
    # 0.45 x 8 mm = 3.6 mm is under the tool-centre floor: 15 mm over the table, tips 4.8 mm.
    point = [0.40, 0.10, SUPPORT + 0.015]
    assert [out["grasp_position"][k] for k in "xyz"] == pytest.approx(point)
    assert out["tip_height"] == pytest.approx(0.015 - REACH)
    assert out["object_width"] == pytest.approx(0.040)
    assert out["thickness"] == pytest.approx(0.008)
    names = [m[0] for m in arm.moves]
    assert names == ["robot.go_to_pose", "robot.go_to_pose_cartesian", "robot.go_to_pose_cartesian"]
    hover, down, lift = (m[1] for m in arm.moves)
    np.testing.assert_allclose(hover, [point[0], point[1], point[2] + 0.06], atol=1e-9)
    np.testing.assert_allclose(down, point, atol=1e-9)
    np.testing.assert_allclose(lift, [point[0], point[1], point[2] + 0.05], atol=1e-9)
    assert out["grip_width"] == pytest.approx(0.041)
    assert out["held_width"] == pytest.approx(0.041)
    assert out["lifted"] == pytest.approx(0.05)
    assert out["attempts"] == 1
    calls = [c.tool for c in ctx.calls]
    assert calls.index("robot.open_gripper") < calls.index("robot.go_to_pose")
    assert "robot.set_grip" not in calls
    assert "robot.move_to_joints" not in calls
    frames = [c for c in ctx.calls if c.tool == "robot.grasp_frame"]
    assert "approach" not in frames[0].kwargs


def test_a_thicker_slab_is_pinched_at_its_grasp_fraction(script):
    arm = Arm()
    out = script.run(arm.ctx(), object_obb=slab(thick=0.015))
    assert out["route"] == "grasped", out["detail"]
    assert out["tip_height"] == pytest.approx(0.45 * 0.015)
    assert out["grasp_position"]["z"] == pytest.approx(SUPPORT + 0.45 * 0.015 + REACH)


def test_short_side_follows_the_slab_yaw(script):
    out = script.run(Arm().ctx(), object_obb=slab(yaw_deg=30.0))
    assert out["route"] == "grasped", out["detail"]
    assert _wrap(out["close_heading_deg"]) == pytest.approx(_wrap(120.0))


def _wrap(deg):
    return (deg + 90.0) % 180.0 - 90.0


def test_a_slab_thinner_than_the_level_reach_is_unreachable(script):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(ctx, object_obb=slab(thick=0.005))
    assert out["route"] == "unreachable"
    assert "too thin" in out["detail"]
    assert arm.moves == [] and arm.closes == 0
    assert out["attempts"] == 0


def test_a_tilt_slides_the_lower_fingertip_into_a_thin_slab(script):
    arm = Arm()
    ctx = arm.ctx()
    out = script.run(ctx, object_obb=slab(thick=0.005), tilt_deg=25.0)
    assert out["route"] == "grasped", out["detail"]
    t = math.radians(25.0)
    assert out["tilt_deg"] == pytest.approx(25.0)
    assert out["tip_height"] == pytest.approx(0.001)  # rides 1 mm over the support into the edge
    frames = [c for c in ctx.calls if c.tool == "robot.grasp_frame"]
    a = frames[0].kwargs["approach"]
    assert a["z"] == pytest.approx(-math.cos(t))
    assert math.hypot(a["x"], a["y"]) == pytest.approx(math.sin(t))
    av = np.array([a[k] for k in "xyz"])
    edge = 0.040 / math.cos(t)
    wide = edge + 0.024
    # Opened part way, then narrowed to the slab's edges in four steps.
    assert arm.grips == pytest.approx(list(np.linspace(wide, edge, 5)))
    names = [m[0] for m in arm.moves]
    assert names == ["robot.go_to_pose"] + ["robot.go_to_pose_cartesian"] * 6
    # The descent and each step keep the lower fingertip 1 mm over the table.
    for (_, target, _), w in zip(arm.moves[1:6], arm.grips, strict=True):
        mid = target + REACH * av
        np.testing.assert_allclose(mid[:2], [0.40, 0.10], atol=1e-9)
        assert mid[2] - w / 2.0 * math.sin(t) == pytest.approx(SUPPORT + 0.001)
    assert arm.closes == 1
    assert "robot.open_gripper" not in [c.tool for c in ctx.calls]


def test_auto_tilt_picks_the_diagonal_or_what_the_tool_floor_needs(script):
    # 15 mm slab, 40 mm across: the diagonal from the bottom edge to under the top edge.
    diag = script.diagonal_tilt_deg(0.015, 0.020, REACH)
    assert diag == pytest.approx(math.degrees(math.atan2(0.015 - 0.001 - 0.0015, 0.040)))
    # 5 mm slab: the floor (tool centre 15 mm up) needs a steeper tilt than the diagonal.
    thin = script.diagonal_tilt_deg(0.005, 0.020, REACH)
    geo = script.pinch_geometry(0.0, 0.005, 0.020, 0.07, math.radians(thin), REACH)
    assert geo["tip_z"] == pytest.approx(0.001, abs=1e-6)
    assert geo["tool_z"] == pytest.approx(0.015, abs=1e-6)
    out = script.run(Arm().ctx(), object_obb=slab(thick=0.005), auto_tilt=True)
    assert out["route"] == "grasped", out["detail"]
    assert out["tilt_deg"] == pytest.approx(thin)


def test_a_refused_follow_step_is_a_miss(script):
    arm = Arm()

    def grip(width_m, **kw):
        arm.grips.append(width_m)
        if len(arm.grips) > 2:
            raise ToolError("robot.set_grip", "jaws jammed")
        return {"position": width_m / SPAN}

    out = script.run(
        arm.ctx(robot__set_grip=grip),
        object_obb=slab(thick=0.005),
        tilt_deg=25.0,
        try_long_axis=False,
    )
    assert out["route"] == "missed"
    assert "jammed" in out["detail"]
    assert arm.closes == 0


def test_pinch_geometry_respects_the_tool_floor(script):
    level = script.pinch_geometry(0.0, 0.008, 0.02, SPAN, 0.0, REACH)
    assert level["tool_z"] == pytest.approx(0.015)
    assert level["start_mid_z"] == level["mid_z"]
    assert level["bite"] == pytest.approx(0.008 - (0.015 - REACH))
    t = math.radians(20.0)
    tilted = script.pinch_geometry(0.0, 0.008, 0.02, 0.064, t, REACH)
    assert tilted["tip_z"] < level["tip_z"]
    assert tilted["start_mid_z"] - 0.032 * math.sin(t) == pytest.approx(0.001)
    assert tilted["tool_z"] >= 0.015 - 1e-12
    assert tilted["upper_z"] > tilted["tip_z"]


def test_too_steep_a_tilt_is_refused(script):
    arm = Arm()
    out = script.run(arm.ctx(), object_obb=slab(), tilt_deg=50.0)
    assert out["route"] == "unreachable"
    assert arm.moves == []


def test_an_empty_close_across_the_short_side_retries_across_the_long_side(script):
    widths = iter([0.0, 0.061])
    arm = Arm(between=lambda arm: next(widths))
    out = script.run(arm.ctx(), object_obb=slab())
    assert out["route"] == "grasped", out["detail"]
    assert out["axis"] == "long"
    assert _wrap(out["close_heading_deg"]) == pytest.approx(0.0)
    assert out["attempts"] == 2
    assert arm.closes == 2


def test_both_axes_empty_is_missed(script):
    arm = Arm(between=lambda arm: 0.0)
    out = script.run(arm.ctx(), object_obb=slab())
    assert out["route"] == "missed"
    assert "closed on nothing" in out["detail"]
    assert arm.gap == SPAN  # left open


def test_no_long_axis_retry_when_asked(script):
    arm = Arm(between=lambda arm: 0.0)
    out = script.run(arm.ctx(), object_obb=slab(), try_long_axis=False)
    assert out["route"] == "missed"
    assert arm.closes == 1


def test_a_gap_wider_than_the_slab_is_a_miss(script):
    arm = Arm(between=lambda arm: 0.070)
    out = script.run(arm.ctx(), object_obb=slab(), try_long_axis=False)
    assert out["route"] == "missed"
    assert "wide" in out["detail"]


def test_a_gap_narrower_than_the_slab_is_a_miss(script):
    # A tilted close that pushed a 40 mm slab aside read 25.7 mm in a station twin.
    arm = Arm(between=lambda arm: 0.0257)
    out = script.run(
        arm.ctx(), object_obb=slab(thick=0.015), tilt_deg=17.4, try_long_axis=False, lift_height=0.0
    )
    assert out["route"] == "missed"
    assert "not across it" in out["detail"]


def test_a_slab_that_slips_out_on_the_lift_is_missed(script):
    arm = Arm(slip=True)
    out = script.run(arm.ctx(), object_obb=slab(), try_long_axis=False)
    assert out["route"] == "missed"
    assert "slipped" in out["detail"]
    assert out["held_width"] == pytest.approx(0.0)


def test_a_fingertip_landing_on_the_slab_is_a_miss(script):
    # The fingertips stop on the 15 mm top: the tool centre 8.25 mm over its target.
    arm = Arm(stop_descent_at=SUPPORT + 0.015 + REACH)
    out = script.run(arm.ctx(), object_obb=slab(thick=0.015), try_long_axis=False)
    assert out["route"] == "missed"
    assert "fingertip on the slab" in out["detail"]
    assert arm.closes == 0


def test_a_servo_stop_within_its_own_bound_is_arrival(script):
    # A servo that reports "reached" 2.5 mm high over an 8 mm slab (measured in a station twin).
    arm = Arm(stop_descent_at=SUPPORT + 0.015 + 0.0025)
    out = script.run(arm.ctx(), object_obb=slab(), try_long_axis=False)
    assert out["route"] == "grasped", out["detail"]


def test_a_lift_that_stalls_with_the_slab_held_is_still_grasped(script):
    arm = Arm(stall_lift=True)
    out = script.run(arm.ctx(), object_obb=slab())
    assert out["route"] == "grasped"
    assert "heavy, fixed or caught" in out["detail"]
    assert out["lifted"] == pytest.approx(0.0)


def test_out_of_reach_everywhere_is_unreachable(script):
    arm = Arm(ik_ok=lambda p, rot: False)
    out = script.run(arm.ctx(), object_obb=slab())
    assert out["route"] == "unreachable"
    assert "IK preview" in out["detail"]
    assert arm.moves == [] and arm.closes == 0


def test_a_refused_hover_is_routed(script):
    arm = Arm(refuse=lambda tool, target: "out of reach" if tool == "robot.go_to_pose" else None)
    out = script.run(arm.ctx(), object_obb=slab())
    assert out["route"] == "unreachable"
    assert "hover" in out["detail"]
    assert arm.closes == 0


def test_a_raised_move_is_routed(script):
    arm = Arm()

    def boom(**kw):
        raise ToolError("robot.go_to_pose", "station refused the pose")

    out = script.run(arm.ctx(robot__go_to_pose=boom), object_obb=slab())
    assert out["route"] == "unreachable"
    assert "station refused" in out["detail"]


def test_a_slab_wider_than_the_open_hand_is_unreachable(script):
    arm = Arm()
    out = script.run(arm.ctx(), object_obb=slab(long=0.12, short=0.10))
    assert out["route"] == "unreachable"
    assert "does not fit" in out["detail"]
    assert arm.moves == []


def test_free_wrist_roll_is_finished_on_the_last_joint(script):
    arm = Arm(free_roll=True)
    ctx = arm.ctx()
    out = script.run(ctx, object_obb=slab(), try_long_axis=False)
    assert out["route"] == "grasped", out["detail"]
    assert abs(out["jaw_error_deg"]) <= 5.0
    assert abs(_wrap(out["wrist_turn_deg"])) == pytest.approx(90.0, abs=1e-6)
    assert "robot.move_to_joints" in [c.tool for c in ctx.calls]


def test_the_support_comes_from_the_cloud_or_is_given(script):
    box = slab(thick=0.008)
    box["center"]["z"] += 0.003  # the box sits 3 mm high (depth noise)
    rng = np.random.default_rng(0)
    pts = np.column_stack(
        [
            rng.uniform(0.38, 0.42, 200),
            rng.uniform(0.08, 0.12, 200),
            rng.uniform(SUPPORT, SUPPORT + 0.008, 200),
        ]
    )
    frame_cloud = script.slab_frame(box, {"points": pts.tolist()})
    assert frame_cloud["source"] == "cloud"
    assert frame_cloud["support"] == pytest.approx(SUPPORT + 0.0004, abs=0.0003)
    frame_given = script.slab_frame(box, support_z=SUPPORT)
    assert frame_given["support"] == pytest.approx(SUPPORT)
    assert script.slab_frame(box)["support"] == pytest.approx(SUPPORT + 0.003)
    out = script.run(Arm().ctx(), object_obb=box, support_z=SUPPORT)
    assert out["thickness"] == pytest.approx(0.011)
    assert out["grasp_position"]["z"] == pytest.approx(SUPPORT + 0.45 * 0.011 + REACH)


def test_arm_id_is_passed_only_when_set(script):
    arm = Arm()
    ctx = arm.ctx()
    script.run(ctx, object_obb=slab())
    assert all("arm_id" not in c.kwargs for c in ctx.calls)
    ctx = Arm().ctx()
    script.run(ctx, object_obb=slab(), arm_id=1)
    assert all(c.kwargs.get("arm_id") == 1 for c in ctx.calls if c.tool != "robot.get_observation")
