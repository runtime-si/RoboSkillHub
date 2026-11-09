"""triggering-a-domino-chain-by-push: FakeContext tests with a small table world (standing tiles that topple when
the closed hand's blade passes their back face by the tipping travel, a stateful arm, and a straight-down depth
camera rendering the tiles' tops) - the stroke's geometry, the depth verdicts and every exit."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "triggering-a-domino-chain-by-push"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
H, W, T = 0.048, 0.024, 0.0075
TABLE = 0.0
START = (0.30, 0.15, 0.20)
REACH = 0.0102
FHD = 0.012
CAM_Z = 1.0
F = 600.0


def _q(arr) -> dict[str, float]:
    return dict(zip("wxyz", map(float, arr), strict=True))


def _down(yaw_deg: float) -> dict[str, float]:
    a = math.radians(yaw_deg) / 2
    # yaw about z, then pointing down (180 deg about x)
    return _q([0.0, math.cos(a), math.sin(a), 0.0])


def slot(x, y, facing=90.0):
    return {"position": {"x": x, "y": y, "z": TABLE}, "facing_deg": facing}


class Tile:
    def __init__(self, x, y, facing=90.0, glued=False, rocks_back=False):
        self.xy = np.array([x, y], dtype=float)
        self.facing = facing
        self.lying = False
        self.glued = glued
        self.rocks_back = rocks_back

    def footprint(self):
        """(half along the facing, half across, top height)."""
        return (H / 2, W / 2, T) if self.lying else (T / 2, W / 2, H)


class World:
    """A table with tiles and an arm whose moves land where commanded. A straight move whose blade front (FHD ahead
    of the tool centre along the move) passes a standing tile's back face by more than its tipping travel, with the
    fingertips under the tile's top, topples it forward (it lies half a height further on). A ``glued`` tile stops
    the hand at its back face instead; one that ``rocks_back`` lets the hand through and stands again.
    ``refuse(tool, pose)`` -> reason refuses a move (routed); ``raise_on`` raises.
    ``stop_high``: a descent to under this height stops there. ``short_by``: level moves end this much short and
    still report reached (a servo's arrival ball)."""

    def __init__(
        self,
        tiles=(),
        start=START,
        refuse=None,
        raise_on=None,
        cameras=True,
        stop_high=None,
        cover=False,
        short_by=0.0,
    ):
        self.short_by = short_by
        self.tiles = list(tiles)
        self.p = np.array(start, dtype=float)
        self.rot = _down(0.0)
        self.refuse = refuse or (lambda tool, pose: None)
        self.raise_on = raise_on or (lambda tool, pose: None)
        self.cameras = cameras
        self.stop_high = stop_high
        self.cover = cover
        self.closed = False
        self.moves: list[tuple[str, np.ndarray, dict]] = []

    def _sweep(self, a, b):
        d = b[:2] - a[:2]
        length = float(np.linalg.norm(d))
        if length < 1e-6 or abs(b[2] - a[2]) > 1e-6:
            return b
        u = d / length
        tip_z = a[2] - REACH
        for t in self.tiles:
            if t.lying:
                continue
            f = math.radians(t.facing)
            along = np.array([math.cos(f), math.sin(f)])
            if float(u @ along) < 0.9 or tip_z >= TABLE + H:
                continue
            back = t.xy - along * T / 2
            across = np.array([-along[1], along[0]])
            if abs(float((a[:2] - back) @ across)) > W / 2 + 0.03:
                continue
            front_start = float((a[:2] - back) @ u) + FHD
            front_end = front_start + length
            if front_start > 0 or front_end < 0:
                continue
            if t.glued:
                return np.array([*(a[:2] + u * (-front_start)), a[2]])
            theta = math.atan2(T, H)
            travel = T * (1 - math.cos(theta)) + (tip_z - TABLE) * math.sin(theta)
            if front_end >= travel and not t.rocks_back:
                t.lying = True
                t.xy = t.xy + along * H / 2
        return b

    def _move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"], dtype=float)
            self.moves.append((tool, target.copy(), dict(kwargs)))
            why = self.refuse(tool, pose)
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            boom = self.raise_on(tool, pose)
            if boom:
                raise RuntimeError(boom)
            if self.stop_high is not None and target[2] < self.stop_high <= self.p[2]:
                target[2] = self.stop_high
            self.rot = dict(pose["rotation"])
            run = target - self.p
            if self.short_by and abs(run[2]) < 1e-9 and np.linalg.norm(run) > self.short_by:
                target = target - run / np.linalg.norm(run) * self.short_by
            end = self._sweep(self.p, target)
            stopped = float(np.linalg.norm(end - target)) > 1e-6
            self.p = end
            if stopped:
                return {"reached": False, "outcome": "timeout", "steps": 250}
            return {"reached": True, "outcome": "reached", "steps": 120}

        return go

    def grasp_frame(self, close_heading_deg=0.0, **kwargs):
        return {
            "rotation": _down(close_heading_deg + 90.0),
            "rotation_flipped": _down(close_heading_deg - 90.0),
        }

    def render(self):
        cx, cy = 320.0, 240.0
        u, v = np.meshgrid(np.arange(640.0), np.arange(480.0))
        rx, ry = (u - cx) / F, (v - cy) / F
        cam = np.array([0.3, 0.3, CAM_Z])
        depth = np.full(u.shape, CAM_Z - TABLE)
        faces = []
        for t in self.tiles:
            ha, hw, top = t.footprint()
            faces.append((t.xy, t.facing, ha, hw, TABLE + top))
        faces.append((self.p[:2], 0.0, 0.03, 0.03, self.p[2] + 0.01))
        if self.cover:
            faces.append((np.array([0.3, 0.3]), 0.0, 0.2, 0.2, 0.5))
        for xy, facing, ha, hw, z in faces:
            d = CAM_Z - z
            if d <= 0.05:
                continue
            wx, wy = cam[0] + d * rx, cam[1] - d * ry
            a = math.radians(facing)
            da = (wx - xy[0]) * math.cos(a) + (wy - xy[1]) * math.sin(a)
            dw = -(wx - xy[0]) * math.sin(a) + (wy - xy[1]) * math.cos(a)
            hit = (np.abs(da) <= ha) & (np.abs(dw) <= hw) & (d < depth)
            depth[hit] = d
        return {
            "name": "top",
            "depth": depth.astype(np.float32),
            "intrinsics": [[F, 0.0, cx], [0.0, F, cy], [0.0, 0.0, 1.0]],
            "pose": {
                "position": {"x": 0.3, "y": 0.3, "z": CAM_Z},
                "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
            },
        }

    def observation(self, **kwargs):
        return {
            "arms": [{"joint_state": {"positions": [0.0] * 6}}],
            "cameras": [self.render()] if self.cameras else [],
        }

    def close(self, **kwargs):
        self.closed = True
        return {"position": 0.0}

    def ctx(self, ik=None, **overrides):
        return FakeContext(
            {
                "robot.get_ee_pose": lambda **kw: {
                    "pose": {
                        "position": dict(zip("xyz", map(float, self.p), strict=True)),
                        "rotation": self.rot,
                    }
                },
                "robot.get_observation": self.observation,
                "robot.describe_gripper": {"finger": {"stated": True, "reach_m": REACH}},
                "robot.grasp_frame": self.grasp_frame,
                "robot.solve_ik": ik
                if ik is not None
                else {"success": True, "within_servo_tolerance": True},
                "robot.go_to_pose": self._move("robot.go_to_pose"),
                "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
                "robot.close_gripper": self.close,
                **overrides,
            }
        )

    def named(self, tool):
        return [m for m in self.moves if m[0] == tool]


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["trigger_domino"].module


# The line runs along +y at x = 0.30 from y = 0.30, pitch 37 mm.
PITCH = 0.037
SLOTS = [slot(0.30, 0.30 + PITCH * k) for k in range(4)]


def _line(n=4):
    return [Tile(s["position"]["x"], s["position"]["y"]) for s in SLOTS[:n]]


def _push(script, world, **kw):
    args = {"table_z": TABLE, "slots": SLOTS, "slot_index": 0}
    args.update(kw)
    return script.run(world.ctx(), **args)


def test_tip_travel_is_the_back_face_motion_at_the_tipping_angle(script):
    theta = math.atan2(T, H)
    assert script.tip_travel(H, T, 0.0) == pytest.approx(T * (1 - math.cos(theta)))
    # 0.7 h up the tile: ~5.3 mm; a taller contact needs a longer travel.
    assert script.tip_travel(H, T, 0.7 * H) == pytest.approx(0.0053, abs=2e-4)
    assert script.tip_travel(H, T, 0.9 * H) > script.tip_travel(H, T, 0.5 * H)


def test_pushes_the_first_tile_over_and_checks_the_slot_from_depth(script):
    world = World(tiles=_line())
    out = _push(script, world)
    assert out["route"] == "triggered", out["detail"]
    assert out["pushed"] and out["verified"] and not out["first_standing"]
    assert out["next_standing"]  # the fake world has no cascade
    assert world.closed
    assert world.tiles[0].lying and not any(t.lying for t in world.tiles[1:])
    contact = 0.7 * H
    tool_z = TABLE + contact + REACH
    transit = TABLE + H + 0.02 + REACH
    assert out["contact_height"] == pytest.approx(contact)
    travel = script.tip_travel(H, T, contact)
    assert out["tip_travel"] == pytest.approx(travel)
    assert out["push_travel"] == pytest.approx(travel + 0.008)
    assert out["push_direction"] == pytest.approx({"x": 0.0, "y": 1.0, "z": 0.0}, abs=1e-12)
    assert out["stroke_steps"] == 120 and out["trigger_time"] > 0
    kinds = [m[0] for m in world.moves]
    assert kinds == [
        "robot.go_to_pose",  # over the stroke's start at transit height (the start is above it)
        "robot.go_to_pose_cartesian",  # straight down behind the tile
        "robot.go_to_pose_cartesian",  # the stroke
        "robot.go_to_pose_cartesian",  # back off along the stroke
        "robot.go_to_pose_cartesian",  # up
        "robot.go_to_pose",  # park where it started, out of the camera's view of the slot
    ]
    over, down, stroke, back, up, park = (m[1] for m in world.moves)
    back_face = 0.30 - T / 2
    start_y = back_face - 0.008 - FHD
    end_y = back_face + travel + 0.008 - FHD
    np.testing.assert_allclose(over, [0.30, start_y, transit], atol=1e-9)
    np.testing.assert_allclose(down, [0.30, start_y, tool_z], atol=1e-9)
    np.testing.assert_allclose(stroke, [0.30, end_y, tool_z], atol=1e-9)
    np.testing.assert_allclose(back, [0.30, end_y - 0.03, tool_z], atol=1e-9)
    np.testing.assert_allclose(up, [0.30, end_y - 0.03, transit], atol=1e-9)
    np.testing.assert_allclose(park, START, atol=1e-9)
    # The stroke goes slow (speed_scale), the free moves do not.
    assert world.moves[2][2] == {"speed_scale": 0.25}
    assert all("speed_scale" not in m[2] for i, m in enumerate(world.moves) if i != 2)
    assert out["stroke_length"] == pytest.approx(end_y - start_y)


def test_rises_first_when_the_hand_starts_low(script):
    world = World(tiles=_line(), start=(0.40, 0.20, 0.03))
    out = _push(script, world)
    assert out["route"] == "triggered", out["detail"]
    first = world.moves[0]
    assert first[0] == "robot.go_to_pose_cartesian"
    np.testing.assert_allclose(first[1], [0.40, 0.20, TABLE + H + 0.02 + REACH], atol=1e-9)


def test_parks_out_of_the_cameras_way_before_the_check(script):
    world = World(tiles=_line())
    park = {"position": {"x": 0.5, "y": 0.0, "z": 0.3}}
    out = _push(script, world, park_pose=park)
    assert out["route"] == "triggered", out["detail"]
    assert world.moves[-1][0] == "robot.go_to_pose"
    np.testing.assert_allclose(world.moves[-1][1], [0.5, 0.0, 0.3])


def test_the_facing_comes_from_the_slot_rotation(script):
    # A slot facing +x as an Se3Pose (x axis = facing): the stroke runs along +x.
    world = World(tiles=[Tile(0.30, 0.30, facing=0.0)])
    pose = {
        "position": {"x": 0.30, "y": 0.30, "z": TABLE},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    out = script.run(world.ctx(), table_z=TABLE, slot=pose)
    assert out["route"] == "triggered", out["detail"]
    stroke = world.named("robot.go_to_pose_cartesian")[1][1]
    assert stroke[1] == pytest.approx(0.30) and stroke[0] > world.moves[0][1][0]


def test_higher_contact_pushes_further_and_lower_less(script):
    lo = _push(script, World(tiles=_line()), contact_fraction=0.45)
    hi = _push(script, World(tiles=_line()), contact_fraction=0.85)
    assert lo["route"] == hi["route"] == "triggered"
    assert hi["contact_height"] > lo["contact_height"]
    assert hi["push_travel"] > lo["push_travel"]


def test_without_verify_reports_what_the_hand_did(script):
    world = World(tiles=_line(), cameras=False)
    out = _push(script, world, verify=False)
    assert out["route"] == "triggered" and not out["verified"]
    assert "not checked" in out["detail"]


def test_a_tile_that_does_not_give_is_blocked(script):
    tiles = _line()
    tiles[0].glued = True
    world = World(tiles=tiles)
    out = _push(script, world)
    assert out["route"] == "blocked", out["detail"]
    assert not out["pushed"] and out["push_travel"] == pytest.approx(0.0, abs=1e-9)
    # It still backs off and rises.
    assert [m[0] for m in world.moves][-2:] == ["robot.go_to_pose_cartesian"] * 2
    assert world.p[2] == pytest.approx(TABLE + H + 0.02 + REACH)


def test_a_stroke_inside_the_arrival_ball_but_short_of_tipping_is_not_blocked(script):
    # No overshoot and a servo that stops 2.5 mm short: the tile rocks back; depth sees it standing.
    world = World(tiles=_line(), short_by=0.0025)
    out = _push(script, world, overshoot=0.0)
    assert out["route"] == "no_contact", out["detail"]
    assert not out["pushed"] and out["verified"] and out["first_standing"]
    assert out["push_travel"] == pytest.approx(out["tip_travel"] - 0.0025)
    assert "inside the arrival tolerance" in out["detail"]
    world = World(tiles=_line(), short_by=0.0025)
    out = _push(script, world, overshoot=0.0, verify=False)
    assert out["route"] == "no_contact" and "tipping angle" in out["detail"]
    # The default overshoot covers the same servo.
    world = World(tiles=_line(), short_by=0.0025)
    out = _push(script, world)
    assert out["route"] == "triggered", out["detail"]


def test_a_descent_that_stops_high_is_blocked_and_lifts_back(script):
    world = World(tiles=_line(), stop_high=TABLE + 0.05)
    out = _push(script, world)
    assert out["route"] == "blocked", out["detail"]
    assert "descent" in out["detail"] and not out["pushed"]
    assert len(world.named("robot.go_to_pose_cartesian")) == 2  # the descent and the lift back
    assert not world.tiles[0].lying


def test_an_empty_slot_is_no_contact_before_anything_moves(script):
    world = World(tiles=_line()[1:])
    out = _push(script, world)
    assert out["route"] == "no_contact", out["detail"]
    assert world.moves == [] and not world.closed
    assert not out["first_standing"]


def test_the_tile_still_standing_after_the_stroke_is_no_contact(script):
    tiles = _line()
    tiles[0].rocks_back = True
    world = World(tiles=tiles)
    out = _push(script, world)
    assert out["route"] == "no_contact", out["detail"]
    assert out["pushed"] and out["verified"] and out["first_standing"]


def test_no_camera_after_the_push_is_unverified(script):
    world = World(tiles=_line(), cameras=False)
    out = _push(script, world)
    assert out["route"] == "unverified", out["detail"]
    assert out["pushed"] and world.tiles[0].lying


def test_an_occluded_view_is_unverified(script):
    world = World(tiles=_line(), cover=True)
    out = _push(script, world)
    assert out["route"] == "unverified", out["detail"]
    assert "occluded" in out["detail"]


def test_a_line_too_tight_for_a_lone_push_is_unreachable(script):
    tight = [slot(0.30, 0.30 + 0.018 * k) for k in range(3)]
    world = World(tiles=[Tile(s["position"]["x"], s["position"]["y"]) for s in tight])
    out = script.run(world.ctx(), table_z=TABLE, slots=tight)
    assert out["route"] == "unreachable" and "too tight" in out["detail"]
    assert world.moves == []


def test_a_tile_behind_where_the_hand_comes_down_is_unreachable(script):
    world = World(tiles=_line())
    out = _push(script, world, slot_index=1)
    assert out["route"] == "unreachable" and "behind" in out["detail"]
    assert world.moves == []


def test_a_contact_height_outside_the_band_is_unreachable(script):
    for frac in (0.2, 0.92):
        out = _push(script, World(tiles=_line()), contact_fraction=frac)
        assert out["route"] == "unreachable" and "contact_fraction" in out["detail"]


def test_a_short_tile_puts_the_tool_under_the_floor(script):
    out = _push(script, World(tiles=_line()), tile_size={"height": 0.010}, contact_fraction=0.36)
    assert out["route"] == "unreachable", out["detail"]


def test_no_slot_is_unreachable(script):
    out = script.run(World().ctx(), table_z=TABLE, slots=[], slot_index=0)
    assert out["route"] == "unreachable" and "no slot" in out["detail"]


def test_an_ik_preview_failure_on_both_wrists_is_unreachable(script):
    world = World(tiles=_line())
    out = script.run(
        world.ctx(ik={"success": True, "within_servo_tolerance": False}), table_z=TABLE, slots=SLOTS
    )
    assert out["route"] == "unreachable" and "IK preview" in out["detail"]
    assert world.moves == [] and not world.closed


def test_the_second_wrist_solution_is_used_when_the_first_is_out_of_reach(script):
    world = World(tiles=_line())
    # Facing 90: the jaws close along 180, grasp_frame gives the hand at 270 and, flipped, at 90.
    ok = _down(90.0)

    def ik(pose, **kw):
        return {"success": True, "within_servo_tolerance": pose["rotation"] == ok}

    out = script.run(world.ctx(ik=ik), table_z=TABLE, slots=SLOTS)
    assert out["route"] == "triggered", out["detail"]
    assert world.rot == ok


def test_a_refused_move_before_the_stroke_is_routed(script):
    world = World(
        tiles=_line(), refuse=lambda tool, pose: "no" if tool == "robot.go_to_pose" else None
    )
    out = _push(script, world)
    assert out["route"] == "unreachable" and "over the stroke's start" in out["detail"]


def test_a_raised_move_is_routed_not_raised(script):
    world = World(
        tiles=_line(),
        raise_on=lambda tool, pose: "station refused" if tool == "robot.go_to_pose" else None,
    )
    out = _push(script, world)
    assert out["route"] == "unreachable" and "station refused" in out["detail"]


def test_a_refused_descent_is_unreachable(script):
    def refuse(tool, pose):
        return "unreachable pose" if tool == "robot.go_to_pose_cartesian" else None

    world = World(tiles=_line(), refuse=refuse)
    out = _push(script, world)
    assert out["route"] == "unreachable" and "descent" in out["detail"]
    assert not out["pushed"] and not world.tiles[0].lying


def test_a_connector_without_speed_scale_still_pushes(script):
    world = World(tiles=_line())
    inner = world._move("robot.go_to_pose_cartesian")

    def strict(pose, arm_id=0):
        return inner(pose)

    out = script.run(
        world.ctx(**{"robot.go_to_pose_cartesian": strict}), table_z=TABLE, slots=SLOTS
    )
    assert out["route"] == "triggered", out["detail"]
    assert world.tiles[0].lying


def test_arm_id_is_passed_only_when_nonzero(script):
    world = World(tiles=_line())
    out = _push(script, world, arm_id=1)
    assert out["route"] == "triggered"
    assert all(m[2].get("arm_id") == 1 for m in world.moves)
    world = World(tiles=_line())
    _push(script, world)
    assert all("arm_id" not in m[2] for m in world.moves)


def test_the_example_declares_every_route(script):
    example = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == {"triggered", "no_contact", "blocked", "unverified", "unreachable"}
    text = (SKILL_DIR / "SKILL.md").read_text()
    for route in declared:
        assert f"    {route}:" in text
