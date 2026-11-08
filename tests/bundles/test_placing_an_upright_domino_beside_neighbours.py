"""placing-an-upright-domino-beside-neighbours: FakeContext tests with a small table world (standing and lying
tiles, a stateful arm and jaws, and a straight-down depth camera rendering the tiles' tops) - the pick, the
place beside neighbours, the depth verdicts and every exit."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "placing-an-upright-domino-beside-neighbours"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
H, W, T = 0.048, 0.024, 0.0075
TABLE = 0.0
START = (0.30, 0.25, 0.20)
SPAN = 0.094
REACH = 0.0102
CAM_Z = 1.0
F = 600.0


def _qz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def _mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _q(arr) -> dict[str, float]:
    return dict(zip("wxyz", map(float, arr), strict=True))


def _down(yaw_deg: float) -> dict[str, float]:
    """A hand pointing straight down, its x axis at this heading (jaws close along its y, yaw - 90)."""
    return _q(_mul(_qz(yaw_deg), np.array([0.0, 1.0, 0.0, 0.0])))


def _yaw_of(rot: dict[str, float]) -> float:
    w, x, y, z = (rot[k] for k in "wxyz")
    return math.degrees(math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))


def slot(x, y, facing=90.0):
    return {"position": {"x": x, "y": y, "z": TABLE}, "facing_deg": facing}


class Tile:
    def __init__(self, x, y, facing=90.0, lying=False):
        self.xy = np.array([x, y], dtype=float)
        self.facing = facing
        self.lying = lying

    def footprint(self):
        """(half along the facing, half across, top height)."""
        return (H / 2, W / 2, T) if self.lying else (T / 2, W / 2, H)


class World:
    """A table with tiles, an arm whose moves land where commanded and jaws that close on a tile under
    them. ``free_roll``: a pose move keeps the hand's heading and the last joint turns it (as the YAM twin).
    ``on_release``: called with (world, tile) when the jaws open on a held tile (faults). ``drop_in_transit``:
    the held tile falls out at the first move along the row. ``stop_high``: a descent to under this height
    stops there. ``refuse(tool, pose)`` -> reason refuses a move (routed); ``raise_on`` raises instead."""

    def __init__(
        self,
        tiles=(),
        start=START,
        free_roll=False,
        on_release=None,
        drop_in_transit=False,
        stop_high=None,
        refuse=None,
        raise_on=None,
        held=None,
        cameras=True,
        stall=None,
    ):
        self.tiles = list(tiles)
        self.p = np.array(start, dtype=float)
        self.yaw = 0.0
        self.q = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.free_roll = free_roll
        self.width = SPAN
        self.held = held
        self.on_release = on_release
        self.drop_in_transit = drop_in_transit
        self.stop_high = stop_high
        self.refuse = refuse or (lambda tool, pose: None)
        self.raise_on = raise_on or (lambda tool, pose: None)
        self.cameras = cameras
        self.stall = stall
        self.moves: list[tuple[str, np.ndarray, float]] = []
        self.grips: list[float] = []

    # --- arm
    def _move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"], dtype=float)
            self.moves.append((tool, target.copy(), _yaw_of(pose["rotation"])))
            why = self.refuse(tool, pose)
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            boom = self.raise_on(tool, pose)
            if boom:
                raise RuntimeError(boom)
            if not self.free_roll:
                self.yaw = _yaw_of(pose["rotation"])
            if self.stop_high is not None and target[2] < self.stop_high <= self.p[2]:
                target[2] = self.stop_high
            if (
                self.stall
                and len(self.named(tool)) == self.stall[0]
                and tool == "robot.go_to_pose_cartesian"
            ):
                target[2] += self.stall[1]
            if (
                self.drop_in_transit
                and self.held is not None
                and tool == "robot.go_to_pose_cartesian"
            ):
                if (
                    abs(target[2] - self.p[2]) < 1e-6
                    and np.linalg.norm(target[:2] - self.p[:2]) > 1e-3
                ):
                    self.held, self.width = None, 0.0
            self.p = target
            return {"reached": True, "outcome": "reached"}

        return go

    def move_to_joints(self, joint_config, **kwargs):
        q = np.asarray(joint_config["positions"], dtype=float)
        self.moves.append(("robot.move_to_joints", q.copy(), 0.0))
        self.yaw -= math.degrees(q[-1] - self.q[-1])
        self.q = q
        return {"reached": True}

    def grasp_frame(self, close_heading_deg=0.0, **kwargs):
        h = math.radians(close_heading_deg)
        return {
            "rotation": _down(close_heading_deg + 90.0),
            "rotation_flipped": _down(close_heading_deg - 90.0),
            "close_axis": {"x": math.cos(h), "y": math.sin(h), "z": 0.0},
        }

    # --- jaws
    def _fraction(self):
        return {"position": self.width / SPAN}

    def set_grip(self, width_m, **kwargs):
        self.grips.append(float(width_m))
        if self.held is not None and width_m > W + 1e-4:
            tile, self.held = self.held, None
            tile.xy = self.p[:2].copy()
            self.tiles.append(tile)
            if self.on_release:
                self.on_release(self, tile)
        self.width = float(width_m)
        return {"command_width_m": float(width_m)}

    def close_gripper(self, **kwargs):
        under = [
            t
            for t in self.tiles
            if not t.lying
            and np.linalg.norm(t.xy - self.p[:2]) < 0.004
            and TABLE + 0.012 <= self.p[2] <= TABLE + H + REACH - 0.003
        ]
        if under and self.width > W:
            self.held = under[0]
            self.tiles.remove(under[0])
            self.width = W
        else:
            self.width = 0.0
        return self._fraction()

    # --- camera: straight down from CAM_Z over (0.3, 0.3); renders tile tops and the hand (a 60 mm square
    # at the tool centre) as horizontal faces.
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
            "arms": [{"joint_state": {"positions": self.q.tolist()}}],
            "cameras": [self.render()] if self.cameras else [],
        }

    def ctx(self, ik=None):
        return FakeContext(
            {
                "robot.get_ee_pose": lambda **kw: {
                    "pose": {
                        "position": dict(zip("xyz", map(float, self.p), strict=True)),
                        "rotation": _down(self.yaw),
                    }
                },
                "robot.get_observation": self.observation,
                "robot.describe_gripper": {
                    "span_m": SPAN,
                    "width_fit": {"at_open_m": SPAN, "at_closed_m": 0.0},
                    "finger": {"stated": True, "reach_m": REACH, "clearance_m": 0.004},
                },
                "robot.get_gripper": lambda **kw: self._fraction(),
                "robot.grasp_frame": self.grasp_frame,
                "robot.solve_ik": ik
                if ik is not None
                else {"success": True, "within_servo_tolerance": True},
                "robot.go_to_pose": self._move("robot.go_to_pose"),
                "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
                "robot.move_to_joints": self.move_to_joints,
                "robot.set_grip": self.set_grip,
                "robot.open_gripper": lambda **kw: self.set_grip(SPAN),
                "robot.close_gripper": self.close_gripper,
            }
        )

    def named(self, tool):
        return [m for m in self.moves if m[0] == tool]


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["place_domino"].module


# The row runs along +y at x = 0.30; slots every 48 mm (one tile height) from y = 0.30.
SLOTS = [slot(0.30, 0.30 + 0.048 * k) for k in range(4)]
STAGED = (0.20, 0.40)


def _row(n):
    return [Tile(s["position"]["x"], s["position"]["y"]) for s in SLOTS[:n]]


def _place(script, world, k=2, **kw):
    args = {"table_z": TABLE, "slot": SLOTS[k], "tile": slot(*STAGED), "neighbours": SLOTS[:k]}
    args.update(kw)
    return script.run(world.ctx(), **args)


def test_picks_and_places_beside_neighbours_and_checks_from_depth(script):
    world = World(tiles=[*_row(2), Tile(*STAGED)])
    out = _place(script, world)
    assert out["route"] == "placed", out["detail"]
    assert out["released"] and out["picked"] and out["verified"]
    assert out["neighbours_standing"] == 2 and out["neighbours_knocked"] == []
    assert out["grasp_height"] == pytest.approx(0.030)
    assert out["grip_width"] == pytest.approx(W)
    assert out["tile_top"] == pytest.approx(H, abs=1e-3)
    assert 0.0 <= out["placement_offset"] < 0.002
    transit = TABLE + H + 0.02 + 0.030
    kinds = [m[0] for m in world.moves]
    assert kinds == [
        "robot.go_to_pose",  # over the tile at transit height
        "robot.go_to_pose_cartesian",  # straight down to the grasp
        "robot.go_to_pose_cartesian",  # lift
        "robot.go_to_pose",  # the lead-in, along the row from its free end
        "robot.go_to_pose_cartesian",  # along the row to over the slot
        "robot.go_to_pose_cartesian",  # straight down to the release
        "robot.go_to_pose_cartesian",  # straight up
        "robot.go_to_pose",  # park
    ]
    over_tile, grasp, lift, lead, over_slot, release, retreat, park = (m[1] for m in world.moves)
    np.testing.assert_allclose(over_tile, [*STAGED, transit], atol=1e-9)
    np.testing.assert_allclose(grasp, [*STAGED, TABLE + 0.030], atol=1e-9)
    np.testing.assert_allclose(lift, [*STAGED, transit], atol=1e-9)
    # The neighbours stand at y < slot: the lead-in comes from +y, the free end.
    np.testing.assert_allclose(lead, [0.30, 0.396 + 0.03, transit], atol=1e-9)
    np.testing.assert_allclose(over_slot, [0.30, 0.396, transit], atol=1e-9)
    np.testing.assert_allclose(release, [0.30, 0.396, TABLE + 0.030 + 0.0015], atol=1e-9)
    np.testing.assert_allclose(retreat, [0.30, 0.396, transit], atol=1e-9)
    np.testing.assert_allclose(park, [*START[:2], max(START[2], transit)], atol=1e-9)
    # Jaws open only the tile plus 4 mm a side, for the grasp and the release; never fully.
    assert world.grips == pytest.approx([W + 0.008, W + 0.008])
    # Jaws across the tile's width: a 90 deg facing closes along 180 deg (the hand's x at 270 / 90).
    assert all(abs(((m[2] - 90.0 + 90.0) % 180.0) - 90.0) < 1e-6 for m in world.moves)
    assert len(world.tiles) == 3 and not any(t.lying for t in world.tiles)


def test_the_transit_clears_every_standing_tile_with_the_held_one(script):
    world = World(tiles=[*_row(2), Tile(*STAGED)])
    out = _place(script, world, grasp_height=0.040)
    assert out["route"] == "placed", out["detail"]
    lowest_horizontal = min(
        m[1][2] for i, m in enumerate(world.moves) if i and m[0] == "robot.go_to_pose"
    )
    # The held tile's bottom hangs grasp_height under the tool centre and clears the tops by 20 mm.
    assert lowest_horizontal - 0.040 == pytest.approx(TABLE + H + 0.02)


def test_lead_in_from_the_hand_side_on_an_empty_row(script):
    world = World(tiles=[Tile(*STAGED)], start=(0.30, 0.10, 0.20))
    out = _place(script, world, k=0, neighbours=[])
    assert out["route"] == "placed", out["detail"]
    lead = world.named("robot.go_to_pose")[1][1]
    assert lead[1] == pytest.approx(0.30 - 0.03)  # the hand starts at y < slot


def test_free_wrist_roll_is_finished_on_the_last_joint(script):
    # The tile faces 90 deg, the slot 120 deg: the hand turns over the tile and again at the lead-in.
    world = World(tiles=[Tile(*STAGED)], free_roll=True)
    out = _place(script, world, k=0, slot=slot(0.30, 0.30, 120.0), neighbours=[])
    assert out["route"] == "placed", out["detail"]
    assert len(world.named("robot.move_to_joints")) == 2
    assert abs(out["jaw_error_deg"]) <= 4.0
    assert out["wrist_turn_deg"] != 0.0


def test_fell_when_the_released_tile_lies_down(script):
    def topple(world, tile):
        tile.lying = True
        tile.xy = tile.xy + np.array([0.0, 0.02])

    world = World(tiles=[*_row(2), Tile(*STAGED)], on_release=topple)
    out = _place(script, world)
    assert out["route"] == "fell", out["detail"]
    assert out["released"] and out["verified"]
    assert out["tile_top"] < 0.02
    assert "over the table" in out["detail"]


def test_fell_when_the_tile_stands_off_the_slot(script):
    def slide(world, tile):
        tile.xy = tile.xy + np.array([0.012, 0.0])

    world = World(tiles=[*_row(2), Tile(*STAGED)], on_release=slide)
    out = _place(script, world)
    assert out["route"] == "fell", out["detail"]
    # The search window ends 6 mm past the footprint, so the offset reads short of 12 mm, over 8.
    assert out["placement_offset"] > 0.008
    assert "off the slot" in out["detail"]


def test_knocked_neighbour_is_named(script):
    def knock(world, tile):
        world.tiles[1].lying = True

    world = World(tiles=[*_row(2), Tile(*STAGED)], on_release=knock)
    out = _place(script, world)
    assert out["route"] == "knocked_neighbour", out["detail"]
    assert out["neighbours_knocked"] == [1]
    assert out["neighbours_standing"] == 1
    assert "the tile stands" in out["detail"]


def test_a_neighbour_down_before_the_place_is_not_blamed(script):
    row = _row(2)
    row[0].lying = True
    world = World(tiles=[*row, Tile(*STAGED)])
    out = _place(script, world)
    assert out["route"] == "placed", out["detail"]
    assert out["neighbours_knocked"] == []
    assert out["neighbours_standing"] == 1


def test_not_holding_when_the_grasp_closes_on_nothing(script):
    world = World(tiles=[*_row(2)])  # the tile is not where it was seen
    out = _place(script, world)
    assert out["route"] == "not_holding", out["detail"]
    assert not out["released"] and not out["picked"]
    assert out["grip_width"] == 0.0
    assert world.width == pytest.approx(W + 0.008)  # reopened
    assert world.moves[-1][1][2] == pytest.approx(TABLE + H + 0.02 + 0.030)  # lifted clear


def test_not_holding_when_dropped_in_transit(script):
    world = World(tiles=[*_row(2), Tile(*STAGED)], drop_in_transit=True)
    out = _place(script, world)
    assert out["route"] == "not_holding", out["detail"]
    assert out["picked"] and not out["released"]
    assert "dropped in transit" in out["detail"]
    assert (
        world.named("robot.go_to_pose_cartesian")[-1][1][2] > TABLE + H
    )  # never descended at the slot


def test_unreachable_when_the_slot_crowds_a_neighbour(script):
    world = World(tiles=[Tile(*STAGED)])
    tight = [slot(0.30, 0.30 + 0.012)]
    out = script.run(
        world.ctx(), table_z=TABLE, slot=slot(0.30, 0.30), tile=slot(*STAGED), neighbours=tight
    )
    assert out["route"] == "unreachable"
    assert "open finger" in out["detail"]
    assert world.moves == []


def test_a_tile_in_a_parallel_row_beside_the_fingers_is_a_conflict(script):
    beside = [(np.array([0.30 + 0.030, 0.30]), 90.0)]
    ahead = [(np.array([0.30, 0.30 + 0.036]), 90.0)]
    size = {"height": H, "width": W, "thickness": T}
    assert script.finger_conflicts(np.array([0.30, 0.30]), 90.0, beside, size, W + 0.008) == [0]
    assert script.finger_conflicts(np.array([0.30, 0.30]), 90.0, ahead, size, W + 0.008) == []


def test_unreachable_when_the_ik_preview_refuses(script):
    world = World(tiles=[*_row(2), Tile(*STAGED)])
    ctx = world.ctx(ik={"success": True, "within_servo_tolerance": False})
    out = script.run(ctx, table_z=TABLE, slot=SLOTS[2], tile=slot(*STAGED), neighbours=SLOTS[:2])
    assert out["route"] == "unreachable"
    assert "IK preview" in out["detail"]
    assert world.moves == []


def test_a_refused_transit_is_routed_still_holding(script):
    def refuse(tool, pose):
        return (
            "joint limit" if tool == "robot.go_to_pose" and pose["position"]["x"] > 0.25 else None
        )

    world = World(tiles=[*_row(2), Tile(*STAGED)], refuse=refuse)
    out = _place(script, world)
    assert out["route"] == "unreachable"
    assert out["picked"] and not out["released"]
    assert "still holding" in out["detail"]


def test_a_raised_move_is_routed_not_raised(script):
    def boom(tool, pose):
        return (
            "servo fault" if tool == "robot.go_to_pose" and pose["position"]["x"] > 0.25 else None
        )

    world = World(tiles=[*_row(2), Tile(*STAGED)], raise_on=boom)
    out = _place(script, world)
    assert out["route"] == "unreachable"
    assert "servo fault" in out["detail"]


def test_a_descent_that_stops_high_lifts_back_without_releasing(script):
    # The fourth straight move is the descent to the slot; it stops 10 mm over its target.
    world = World(tiles=[*_row(2), Tile(*STAGED)], stall=(4, 0.010))
    out = _place(script, world)
    assert out["route"] == "unreachable", out["detail"]
    assert not out["released"]
    assert "stopped 10.0 mm high" in out["detail"]
    assert world.held is not None
    assert world.moves[-1][1][2] == pytest.approx(TABLE + H + 0.02 + 0.030)


@pytest.mark.parametrize("gh", [0.010, 0.050, 0.055])
def test_grasp_heights_the_fingertips_cannot_do_are_refused(script, gh):
    world = World(tiles=[*_row(2), Tile(*STAGED)])
    out = _place(script, world, grasp_height=gh)
    assert out["route"] == "unreachable"
    assert "grasp_height" in out["detail"]
    assert world.moves == []


def test_places_a_tile_already_in_the_hand(script):
    held = Tile(0.0, 0.0)
    world = World(tiles=_row(2), held=held, start=(0.30, 0.20, 0.10))
    world.width = W
    out = script.run(
        world.ctx(),
        table_z=TABLE,
        slot=SLOTS[2],
        tile=None,
        neighbours=SLOTS[:2],
        grasp_height=0.026,
    )
    assert out["route"] == "placed", out["detail"]
    assert not out["picked"]
    release = world.named("robot.go_to_pose_cartesian")[-2][1]
    assert release[2] == pytest.approx(TABLE + 0.026 + 0.0015)


def test_not_holding_when_the_hand_is_empty_and_no_tile_is_given(script):
    world = World(tiles=_row(2))
    world.width = 0.0
    out = script.run(world.ctx(), table_z=TABLE, slot=SLOTS[2], tile=None, neighbours=SLOTS[:2])
    assert out["route"] == "not_holding"
    assert world.moves == []


def test_slots_list_and_index_give_the_neighbours(script):
    world = World(tiles=[*_row(3), Tile(*STAGED)])
    out = script.run(world.ctx(), table_z=TABLE, slots=SLOTS, slot_index=3, tile=slot(*STAGED))
    assert out["route"] == "placed", out["detail"]
    assert out["neighbours_standing"] == 3
    assert out["release_position"]["y"] == pytest.approx(0.30 + 3 * 0.048)


def test_a_bad_slot_index_is_unreachable(script):
    out = script.run(World().ctx(), table_z=TABLE, slots=SLOTS, slot_index=7)
    assert out["route"] == "unreachable"


def test_se3_pose_slots_take_the_facing_from_the_x_axis(script):
    xy, facing = script.tile_pose(
        {"position": {"x": 0.1, "y": 0.2, "z": 0.0}, "rotation": _q(_qz(30.0))}
    )
    assert facing == pytest.approx(30.0)
    np.testing.assert_allclose(xy, [0.1, 0.2])


def test_unverified_when_the_hand_parks_over_the_slot(script):
    world = World(tiles=[*_row(2), Tile(*STAGED)])
    park = {"position": {"x": 0.30, "y": 0.396, "z": 0.15}}
    out = _place(script, world, park_pose=park)
    assert out["route"] == "unverified", out["detail"]
    assert "occluded" in out["detail"]
    assert out["released"] and not out["verified"]


def test_unverified_without_cameras_and_placed_without_the_check(script):
    world = World(tiles=[*_row(2), Tile(*STAGED)], cameras=False)
    assert _place(script, world)["route"] == "unverified"
    world = World(tiles=[*_row(2), Tile(*STAGED)], cameras=False)
    out = _place(script, world, verify=False)
    assert out["route"] == "placed"
    assert "not checked" in out["detail"]
    assert world.moves[-1][0] == "robot.go_to_pose_cartesian"  # no park without the check


def test_a_camera_name_that_is_not_there_leaves_it_unverified(script):
    world = World(tiles=[*_row(2), Tile(*STAGED)])
    out = _place(script, world, camera_name="wrist")
    assert out["route"] == "unverified"


def test_a_footprint_at_the_image_edge_is_not_judged(script):
    world = World(tiles=[Tile(0.3, 0.3)])
    cam = world.render()
    size = {"height": H, "width": W, "thickness": T}
    # 1.0 m away the image spans +-0.53 m in x; a tile at its very edge is cut off.
    v = script.judge_tile([cam], np.array([0.3 + 0.53, 0.3]), 90.0, TABLE, size, 0.0)
    assert not v["seen"]
    v = script.judge_tile([cam], np.array([0.3, 0.3]), 90.0, TABLE, size, 0.0)
    assert v["seen"] and v["standing"]


def test_every_route_is_declared_in_the_example():
    example = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    source = (SKILL_DIR / "scripts" / "place_domino.py").read_text()
    routes = set(re.findall(r'finish\(\s*"([a-z_]+)"', source)) | {"unreachable"}
    assert routes <= declared, routes - declared
    md = (SKILL_DIR / "SKILL.md").read_text()
    for r in routes:
        assert f"    {r}:" in md
