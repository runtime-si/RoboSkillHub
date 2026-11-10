"""refilling-a-domino-loading-rack: empty-slot choice, the pick, the descent into the slot, its retries, the check.

A fake world renders the rack and a staging tile as a heightfield seen by an overhead depth camera; a fake arm lands
where it is sent; a held tile stops on a wall's top when it comes down off the slot, unless told to slide up in the
pinch instead.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "refilling-a-domino-loading-rack"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
H, W, T = 0.048, 0.024, 0.0075
OPEN = 0.06
STAGE = (0.20, 0.20)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["refill_rack"].module


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
    def __init__(self, fill="100100", pitch=0.024, wall=0.020, opening=0.0095):
        self.fill = list(fill)
        self.pitch, self.wall, self.opening = pitch, wall, opening
        self.x0, self.y0 = 0.1, 0.3
        self.pos = np.array([0.15, 0.25, 0.25])
        self.rot = _topdown(0.0)
        self.gap = OPEN
        self.held = False
        self.staged = True
        self.hang = 0.0
        self.calls: list[str] = []
        self.jam_descents = 0
        self.slide = False
        self.off_along = 0.0  # the hand lands this far along +y from where it is sent

    def slot_xy(self, k):
        return np.array([self.x0, self.y0 + k * self.pitch])

    def height(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        z = np.zeros(np.broadcast(x, y).shape)
        block = self.pitch - self.opening
        for i in range(len(self.fill) + 1):
            cy = self.y0 + (i - 0.5) * self.pitch
            m = (np.abs(x - self.x0) <= 0.02) & (np.abs(y - cy) <= block / 2)
            z = np.where(m, np.maximum(z, self.wall), z)
        for k, c in enumerate(self.fill):
            sx, sy = self.slot_xy(k)
            if c == "1":
                z = np.where(
                    (np.abs(x - sx) <= W / 2) & (np.abs(y - sy) <= T / 2), np.maximum(z, H), z
                )
            if c == "J":
                z = np.where(
                    (np.abs(x - sx) <= H / 2) & (np.abs(y - sy) <= W / 2),
                    np.maximum(z, self.wall + T),
                    z,
                )
        if self.staged:
            z = np.where(
                (np.abs(x - STAGE[0]) <= T / 2) & (np.abs(y - STAGE[1]) <= W / 2),
                np.maximum(z, H),
                z,
            )
        return z

    def camera(self):
        cam_z, f, w, h = 3.0, 3000.0, 640, 480
        cx, cy = w / 2, h / 2
        centre = np.array([0.15, 0.33])
        pose = np.eye(4)
        pose[:3, :3] = [[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]
        pose[:3, 3] = [centre[0], centre[1], cam_z]
        uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        depth = np.full((h, w), cam_z)
        for _ in range(3):
            depth = cam_z - self.height(
                centre[0] + (uu - cx) / f * depth, centre[1] - (vv - cy) / f * depth
            )
        return {
            "name": "top",
            "depth": depth,
            "intrinsics": [[f, 0, cx], [0, f, cy], [0, 0, 1]],
            "pose": pose,
        }

    def slot_under(self):
        for k in range(len(self.fill)):
            d = self.pos[:2] - self.slot_xy(k)
            if abs(d[0]) < 0.01 and abs(d[1]) < self.pitch / 2:
                return k, float(d[1])
        return None, 0.0

    def ctx(self, **extra):
        def log(name, fn):
            def inner(**kw):
                self.calls.append(name)
                return fn(**kw)

            return inner

        def target(kw, off=0.0):
            p = kw["pose"]["position"]
            return np.array([p["x"], p["y"] + off, p["z"]]), kw["pose"]["rotation"]

        def go(**kw):
            # a free move lands off along the row once it holds a tile (the read-back corrects it)
            self.pos, self.rot = target(kw, self.off_along if self.held else 0.0)
            return {"outcome": "reached"}

        def cart(**kw):
            goal, rot = target(kw)
            k, along = self.slot_under()
            if self.held and k is not None and goal[2] < self.pos[2] - 1e-6:
                stop = self.wall + self.hang  # the tile's bottom on the walls' top
                off = abs(along) > (self.opening - T) / 2 + 1e-6
                if (off or self.jam_descents > 0) and goal[2] < stop:
                    self.jam_descents = max(0, self.jam_descents - 1)
                    if self.slide:
                        self.hang = (
                            goal[2] - self.wall
                        )  # the tile stays on the wall, sliding up between the pads
                    else:
                        goal = np.array([*goal[:2], stop])
            self.pos, self.rot = goal, rot
            return {"outcome": "reached"}

        def close(**kw):
            if (
                self.staged
                and np.linalg.norm(self.pos[:2] - np.array(STAGE)) < 0.004
                and self.pos[2] < H + 0.0102
            ):
                self.held, self.staged, self.hang, self.gap = (
                    True,
                    False,
                    float(self.pos[2]),
                    W + 0.0013,
                )
            else:
                self.gap = 0.0
            return {"position": self.gap / OPEN}

        def set_grip(**kw):
            self.gap = float(kw["width_m"])
            if self.held and self.gap > W + 0.003:
                self.held = False
                k, along = self.slot_under()
                bottom = self.pos[2] - self.hang
                if k is not None:
                    self.fill[k] = (
                        "1"
                        if bottom < self.wall and abs(along) <= (self.opening - T) / 2 + 1e-6
                        else "J"
                    )
            return {"outcome": "reached"}

        tools = {
            "robot.describe_gripper": {
                "width_fit": {"at_open_m": OPEN, "at_closed_m": 0.0},
                "finger": {"reach_m": 0.0102},
            },
            "robot.get_observation": lambda **kw: {
                "cameras": [self.camera()],
                "arms": [{"joint_state": [0.0] * 6}],
            },
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
            "robot.solve_ik": {"success": True, "within_servo_tolerance": True},
            "robot.set_grip": log("set_grip", set_grip),
            "robot.go_to_pose": log("go_to_pose", go),
            "robot.go_to_pose_cartesian": log("cartesian", cart),
            "robot.move_to_joints": {"outcome": "reached"},
            "robot.close_gripper": log("close", close),
            "robot.get_gripper": lambda **kw: {"position": self.gap / OPEN},
            "robot.open_gripper": {"outcome": "reached"},
        }
        tools.update(extra)
        return FakeContext(tools)

    def rack(self):
        return {
            "origin": [self.x0, self.y0, 0.0],
            "facing_deg": 90.0,
            "pitch": self.pitch,
            "n_slots": len(self.fill),
            "wall_height": self.wall,
        }


def _tile():
    return {
        "position": {"x": STAGE[0], "y": STAGE[1], "z": 0.0},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def _run(script, world, **kw):
    kw.setdefault("tile", _tile())
    return script.run(world.ctx(), table_z=0.0, rack=kw.pop("rack", world.rack()), **kw)


def test_refills_the_first_empty_slot(script):
    world = World()
    res = _run(script, world)
    assert res["route"] == "refilled" and res["slot_index"] == 1
    assert res["slot_states"] == ["full", "empty", "empty", "full", "empty", "empty"]
    assert world.fill[1] == "1" and res["picked"] and res["released"]
    assert res["attempts"] == 1 and res["slot_state_after"] == "full" and res["empty_count"] == 3
    # held over the walls: the fingertips clear them at the release
    assert res["grasp_height"] - 0.0102 >= world.wall + 0.002 - 1e-9


@pytest.mark.parametrize(("order", "slot"), [("reverse", 5), ("index", 1)])
def test_order(script, order, slot):
    assert _run(script, World(), order=order)["slot_index"] == slot


def test_aligns_the_tile_over_a_narrow_slot(script):
    world = World()
    world.off_along = 0.002  # the hand lands 2 mm off along the row: the read-back corrects it
    res = _run(script, world)
    assert res["route"] == "refilled" and abs(res["align_error_mm"]) <= 0.5


def test_offsets_retried_after_a_stop_on_the_wall(script):
    world = World()
    world.jam_descents = 1
    res = _run(script, world)
    assert (
        res["route"] == "refilled"
        and res["attempts"] == 2
        and res["insert_offset_mm"] == pytest.approx(1.0)
    )


def test_jammed_after_every_offset(script):
    world = World()
    world.jam_descents = 99
    res = _run(script, world)
    assert res["route"] == "jammed" and "still holding" in res["detail"] and world.held
    assert res["attempts"] == 5


def test_a_tile_that_slid_up_in_the_pinch_is_caught_after(script):
    world = World()
    world.jam_descents, world.slide = 1, True
    res = _run(script, world)
    assert res["route"] == "fell" and res["slot_state_after"] == "jammed"


def test_rack_full_lists_jams(script):
    res = _run(script, World(fill="11J1"))
    assert res["route"] == "rack_full" and "jammed: [2]" in res["detail"]


def test_places_a_held_tile(script):
    world = World()
    world.staged, world.held, world.hang, world.gap = False, True, 0.034, W + 0.0013
    world.pos = np.array([0.15, 0.25, 0.25])
    res = script.run(world.ctx(), table_z=0.0, rack=world.rack(), tile=None, grasp_height=0.034)
    assert res["route"] == "refilled" and not res["picked"] and world.fill[1] == "1"


def test_hand_full_and_not_holding(script):
    world = World()
    world.gap = W + 0.002
    assert _run(script, world)["route"] == "hand_full"
    world = World()
    res = script.run(world.ctx(), table_z=0.0, rack=world.rack(), tile=None, grasp_height=0.034)
    assert res["route"] == "not_holding"


def test_pick_closes_on_nothing(script):
    world = World()
    world.staged = False
    res = _run(script, world)
    assert res["route"] == "not_holding" and "nothing" in res["detail"]


def test_pitch_too_tight_for_the_open_fingers(script):
    res = _run(script, World(fill="1010", pitch=0.018))
    assert res["route"] == "unreachable" and "pitch" in res["detail"]


@pytest.mark.parametrize(
    "kw",
    [
        {"rack": {}},
        {"order": "random"},
        {"slot_index": 9},
        {"wall_height": 0.05},
        {"tile": None},
    ],
)
def test_invalid_input(script, kw):
    assert _run(script, World(), **kw)["route"] == "invalid_input"


def test_example_declares_every_route(script):
    ex = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert declared == {
        "refilled",
        "rack_full",
        "jammed",
        "fell",
        "unverified",
        "not_holding",
        "hand_full",
        "not_seen",
        "unreachable",
        "invalid_input",
    }
