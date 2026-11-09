"""picking-upright-dominoes-from-a-loading-rack: slot reading from depth, choice, the pick and every exit.

A fake world renders the rack (blocks between slots, standing tiles, a tile lying across a slot) as a heightfield
seen by an overhead pinhole depth camera, and a fake arm lands where it is sent unless told to stall.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "picking-upright-dominoes-from-a-loading-rack"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
H, W, T = 0.048, 0.024, 0.0075
TABLE_Z = 0.0
OPEN = 0.06


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["pick_from_rack"].module


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _topdown(heading_deg):
    a = math.radians(heading_deg) / 2
    q = _quat_mul((math.cos(a), 0.0, 0.0, math.sin(a)), (0.0, 1.0, 0.0, 0.0))
    return dict(zip("wxyz", q, strict=True))


class World:
    """A rack along +y at x = 0.1 (slots facing +y), an overhead depth camera and a fake arm."""

    def __init__(self, fill="1101J1", pitch=0.024, wall=0.020, opening=0.0095, x0=0.1, y0=0.3):
        self.fill, self.pitch, self.wall, self.opening = fill, pitch, wall, opening
        self.x0, self.y0 = x0, y0
        self.pos = np.array([0.1, 0.2, 0.25])
        self.rot = _topdown(0.0)
        self.gap = OPEN
        self.held = None
        self.calls: list[str] = []
        self.see = True
        self.stall_descent = 0.0
        self.stall_lift = 0.0
        self.close_empty = False
        self.slip_on_lift = False
        self.ik_ok = True

    # --- the scene
    def slot_xy(self, k):
        return np.array([self.x0, self.y0 + k * self.pitch])

    def height(self, x, y):
        """Surface height over (x, y) arrays."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        z = np.full(np.broadcast(x, y).shape, TABLE_Z)
        n = len(self.fill)
        block = self.pitch - self.opening
        for i in range(n + 1):
            cy = self.y0 + (i - 0.5) * self.pitch
            m = (np.abs(x - self.x0) <= 0.02) & (np.abs(y - cy) <= block / 2)
            z = np.where(m, np.maximum(z, TABLE_Z + self.wall), z)
        for k, c in enumerate(self.fill):
            sx, sy = self.slot_xy(k)
            if c == "1":
                m = (np.abs(x - sx) <= W / 2) & (np.abs(y - sy) <= T / 2)
                z = np.where(m, np.maximum(z, TABLE_Z + H), z)
            if c == "J":
                m = (np.abs(x - sx) <= H / 2) & (np.abs(y - sy) <= W / 2)
                z = np.where(m, np.maximum(z, TABLE_Z + self.wall + T), z)
        return z

    def camera(self):
        cam_z, f, w, h = 0.6, 600.0, 320, 400
        cx, cy = w / 2, h / 2
        centre = np.array([self.x0, self.y0 + 0.06])
        rot = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = [centre[0], centre[1], cam_z]
        uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        depth = np.full((h, w), cam_z - TABLE_Z)
        for _ in range(3):
            depth = cam_z - self.height(
                centre[0] + (uu - cx) / f * depth, centre[1] - (vv - cy) / f * depth
            )
        k = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
        return {"name": "top", "depth": depth, "intrinsics": k, "pose": pose}

    # --- tools
    def ctx(self, **extra):
        cam = self.camera() if self.see else None

        def log(name):
            def wrap(fn):
                def inner(**kw):
                    self.calls.append(name)
                    return fn(**kw)

                return inner

            return wrap

        def pose_of(kw):
            p = kw["pose"]["position"]
            return np.array([p["x"], p["y"], p["z"]]), kw["pose"]["rotation"]

        def go(**kw):
            self.pos, self.rot = pose_of(kw)
            return {"outcome": "reached"}

        def cart(**kw):
            target, rot = pose_of(kw)
            down = target[2] < self.pos[2] - 1e-6
            if down and self.stall_descent:
                target = target + np.array([0, 0, self.stall_descent])
            if not down and self.held is not None and self.stall_lift:
                target = np.array([*target[:2], min(target[2], self.pos[2] + 0.002)])
            if not down and self.held is not None and self.slip_on_lift:
                self.held, self.gap = None, 0.0
            self.pos, self.rot = target, rot
            return {"outcome": "reached"}

        def close(**kw):
            near = [
                k
                for k, c in enumerate(self.fill)
                if c == "1" and np.linalg.norm(self.slot_xy(k) - self.pos[:2]) < 0.004
            ]
            if near and not self.close_empty and self.pos[2] < TABLE_Z + H + 0.0102:
                self.held, self.gap = near[0], W + 0.0025
            else:
                self.gap = 0.0
            return {"position": self.gap / OPEN}

        def set_grip(**kw):
            self.gap = float(kw["width_m"])
            if self.gap > W + 0.003:
                self.held = None
            return {"outcome": "reached"}

        def solve_ik(**kw):
            return {"success": self.ik_ok, "within_servo_tolerance": self.ik_ok}

        tools = {
            "robot.describe_gripper": {
                "width_fit": {"at_open_m": OPEN, "at_closed_m": 0.0},
                "finger": {"reach_m": 0.0102},
            },
            "robot.get_observation": log("get_observation")(
                lambda **kw: {
                    "cameras": [cam] if cam is not None else [],
                    "arms": [{"joint_state": [0.0] * 6}],
                }
            ),
            "robot.get_ee_pose": lambda **kw: {
                "position": dict(zip("xyz", map(float, self.pos), strict=True)),
                "rotation": self.rot,
            },
            "robot.grasp_frame": lambda **kw: {
                "rotation": _topdown(kw["close_heading_deg"]),
                "rotation_flipped": _topdown(kw["close_heading_deg"] + 180.0),
                "close_axis": {
                    "x": math.cos(math.radians(kw["close_heading_deg"])),
                    "y": math.sin(math.radians(kw["close_heading_deg"])),
                    "z": 0.0,
                },
            },
            "robot.solve_ik": log("solve_ik")(solve_ik),
            "robot.set_grip": log("set_grip")(set_grip),
            "robot.go_to_pose": log("go_to_pose")(go),
            "robot.go_to_pose_cartesian": log("cartesian")(cart),
            "robot.move_to_joints": log("move_to_joints")(lambda **kw: {"outcome": "reached"}),
            "robot.close_gripper": log("close")(close),
            "robot.get_gripper": lambda **kw: {"position": self.gap / OPEN},
        }
        tools.update(extra)
        return FakeContext(tools)

    def rack(self):
        return {
            "origin": [self.x0, self.y0, TABLE_Z],
            "facing_deg": 90.0,
            "pitch": self.pitch,
            "n_slots": len(self.fill),
            "wall_height": self.wall,
        }


def _run(script, world, ctx=None, **kw):
    return script.run(ctx or world.ctx(), table_z=TABLE_Z, rack=kw.pop("rack", world.rack()), **kw)


# --- geometry -----------------------------------------------------------------------------


def test_grasp_band_over_the_walls(script):
    size = dict(script.TILE_SIZE)
    lo, hi = script.grasp_band(size, 0.020, 0.0102, 0.0)
    assert lo == pytest.approx(0.0322) and hi == pytest.approx(0.046)
    lo, hi = script.grasp_band(size, 0.025, 0.0102, 0.0)
    assert lo == pytest.approx(0.0372)
    lo, hi = script.grasp_band(size, 0.034, 0.0102, 0.0)
    assert lo > hi


def test_rack_slots_from_origin_and_from_poses(script):
    slots = script.rack_slots(
        {"origin": [0.1, 0.3, 0.01], "facing_deg": 90.0, "pitch": 0.024, "n_slots": 3}
    )
    assert np.allclose([s[0] for s in slots], [[0.1, 0.3], [0.1, 0.324], [0.1, 0.348]])
    assert all(s[1] == 90.0 and s[2] == 0.01 for s in slots)
    side = script.rack_slots(
        {
            "origin": {"x": 0, "y": 0, "z": 0},
            "facing_deg": 0.0,
            "pitch": 0.03,
            "n_slots": 2,
            "row_deg": 90.0,
        }
    )
    assert side[1][0].tolist() == pytest.approx([0.0, 0.03])
    posed = script.rack_slots(
        {
            "slots": [
                {
                    "position": {"x": 0.2, "y": 0.1, "z": 0.0},
                    "rotation": {"w": 1, "x": 0, "y": 0, "z": 0},
                }
            ]
        }
    )
    assert posed[0][1] == pytest.approx(0.0)


# --- reading and choosing -----------------------------------------------------------------


def test_reads_full_empty_and_jammed_and_picks_the_nearest(script):
    world = World()
    world.pos = np.array([0.1, 0.45, 0.25])  # nearest slot 5
    res = _run(script, world)
    assert res["slot_states"] == ["full", "full", "empty", "full", "jammed", "full"]
    assert res["route"] == "picked" and res["slot_index"] == 5 and res["picked"]
    assert res["full_count"] == 4 and res["remaining"] == 3
    assert res["grasp_height"] == pytest.approx(0.0322)
    assert res["tile"]["position"]["y"] == pytest.approx(0.3 + 5 * 0.024)
    assert world.held == 5


def test_lifts_straight_out_over_the_walls_before_moving_sideways(script):
    world = World()
    moves = []

    def record(**kw):
        p = kw["pose"]["position"]
        moves.append((p["x"], p["y"], p["z"]))
        world.pos = np.array([p["x"], p["y"], p["z"]])
        world.rot = kw["pose"]["rotation"]
        return {"outcome": "reached"}

    res = _run(script, world, ctx=world.ctx(**{"robot.go_to_pose_cartesian": record}))
    assert res["route"] == "picked"
    grasp_i = next(i for i, m in enumerate(moves) if m[2] == pytest.approx(0.0322))
    lift = moves[grasp_i + 1]
    assert lift[:2] == pytest.approx(moves[grasp_i][:2])
    assert lift[2] >= TABLE_Z + 0.020 + 0.010 + 0.0322 - 1e-9


@pytest.mark.parametrize(("order", "slot"), [("index", 0), ("reverse", 5)])
def test_orders(script, order, slot):
    assert _run(script, World(), order=order)["slot_index"] == slot


def test_requested_slot(script):
    assert _run(script, World(), slot_index=3)["slot_index"] == 3
    res = _run(script, World(), slot_index=2)
    assert res["route"] == "slot_empty" and res["slot_index"] == 2
    res = _run(script, World(), slot_index=4)
    assert res["route"] == "slot_empty" and "jammed" in res["detail"]


def test_rack_empty_lists_the_jams(script):
    world = World(fill="00J0")
    res = _run(script, world)
    assert res["route"] == "rack_empty" and "jammed: [2]" in res["detail"]
    assert "set_grip" not in world.calls and "go_to_pose" not in world.calls


def test_not_seen_and_the_bookkeeping_fallback(script):
    world = World()
    world.see = False
    assert _run(script, world)["route"] == "not_seen"
    world = World()
    world.see = False
    res = _run(script, world, occupied=[False, True, False, False, False, False])
    assert res["route"] == "picked" and res["slot_index"] == 1
    # the bookkeeping is trusted: a slot it calls full that is empty closes on nothing
    world = World()
    world.see = False
    res = _run(script, world, occupied=[False, False, True, False, False, False])
    assert res["route"] == "grasp_failed" and res["slot_index"] == 2


def test_look_pose_reads_again(script):
    world = World()
    cam = world.camera()
    seen = iter(
        [{"cameras": [], "arms": []}]
        + [{"cameras": [cam], "arms": [{"joint_state": [0.0] * 6}]}] * 10
    )
    look = {"position": {"x": 0.0, "y": 0.1, "z": 0.3}, "rotation": _topdown(0.0)}
    res = _run(
        script,
        world,
        ctx=world.ctx(**{"robot.get_observation": lambda **kw: next(seen)}),
        look_pose=look,
    )
    assert res["route"] == "picked"
    assert "go_to_pose" in world.calls


# --- failures -----------------------------------------------------------------------------


def test_hand_full_moves_nothing(script):
    world = World()
    world.gap = W + 0.002
    res = _run(script, world)
    assert res["route"] == "hand_full"
    assert not {"set_grip", "go_to_pose", "cartesian", "close"} & set(world.calls)


def test_grasp_closes_on_nothing(script):
    world = World()
    world.close_empty = True
    res = _run(script, world)
    assert res["route"] == "grasp_failed" and "nothing" in res["detail"]
    assert world.pos[2] > TABLE_Z + H


def test_descent_stopped_high(script):
    world = World()
    world.stall_descent = 0.012
    res = _run(script, world)
    assert res["route"] == "grasp_failed" and "high" in res["detail"]
    assert "close" not in world.calls


def test_lift_held_is_stuck(script):
    world = World()
    world.stall_lift = True
    res = _run(script, world)
    assert res["route"] == "stuck" and res["lift_shortfall"] > 0.006
    assert world.held is None  # the jaws reopened


def test_tile_slips_out_on_the_lift(script):
    world = World()
    world.slip_on_lift = True
    res = _run(script, world)
    assert res["route"] == "grasp_failed" and "slipped" in res["detail"]


def test_pitch_too_tight_for_the_open_fingers(script):
    world = World(fill="11111", pitch=0.018)
    res = _run(script, world)
    assert res["route"] == "unreachable" and "pitch" in res["detail"]
    assert "set_grip" not in world.calls


def test_walls_too_high_for_any_grasp(script):
    res = _run(script, World(wall=0.036, fill="1"))
    assert res["route"] == "unreachable" and "no grasp height" in res["detail"]


def test_ik_preview_fails(script):
    world = World()
    world.ik_ok = False
    res = _run(script, world)
    assert res["route"] == "unreachable" and "IK" in res["detail"]


def test_refused_move_is_routed(script):
    world = World()

    def refuse(**kw):
        raise RuntimeError("joint limit")

    res = _run(script, world, ctx=world.ctx(**{"robot.go_to_pose": refuse}))
    assert res["route"] == "unreachable" and "joint limit" in res["detail"]


@pytest.mark.parametrize(
    "kw",
    [
        {"rack": {}},
        {"rack": {"origin": [0, 0, 0], "facing_deg": 0, "pitch": 0.0, "n_slots": 3}},
        {"order": "random"},
        {"occupied": [True]},
        {"slot_index": 9},
        {"wall_height": 0.05},
        {"tile_size": {"width": 0.06}},
    ],
)
def test_invalid_input(script, kw):
    world = World()
    assert _run(script, world, **kw)["route"] == "invalid_input"


def test_example_declares_every_route(script):
    ex = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert declared == {
        "picked",
        "slot_empty",
        "rack_empty",
        "grasp_failed",
        "stuck",
        "hand_full",
        "not_seen",
        "unreachable",
        "invalid_input",
    }
