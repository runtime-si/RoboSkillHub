"""Stud-grid geometry: forward/inverse round trips, symmetric orientations, negatives, and a
cross-check against skills/computing-brick-pose-from-stud-grid's original chain."""

from __future__ import annotations

import importlib.util
import math
import random
from pathlib import Path

import pytest
from lego_fixtures import PLATE_POSE, T_TCP_OBJECT, WORLD

from roboskillhub.tasks.lego import geometry as g

REPO = Path(__file__).resolve().parents[2]


def _random_plate_pose(rng: random.Random) -> dict:
    yaw = rng.uniform(-180, 180)
    tilt = rng.uniform(-3, 3)
    c, s = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    rx = [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]
    m = g.mul(
        g.mul(g.tr(rng.uniform(0.2, 0.5), rng.uniform(-0.2, 0.2), rng.uniform(0, 0.1)), g.rz(yaw)),
        rx,
    )
    return g.to_pose(m, WORLD)


def test_known_seat_lego_2x4_rotated():
    plate = g.Plate(PLATE_POSE, (16, 16), g.SYSTEMS["lego"], "c")
    s = g.seat_pose(plate, g.brick_type("lego", "2x4"), g.Cell(2, 3, 1, 90))
    # 16 studs * 8 mm = 128 mm plate; 4x2 footprint at (2, 3) -> centre 32 mm from the corner.
    assert s["position_m"] == pytest.approx([0.30 - 0.032, -0.032, 0.02], abs=1e-12)
    assert abs(s["quaternion_xyzw"][2]) == pytest.approx(math.sqrt(0.5))


def test_duplo_is_twice_lego_laterally_and_heights_differ():
    lego = g.Plate(PLATE_POSE, (8, 8), g.SYSTEMS["lego"], "c")
    duplo = g.Plate(PLATE_POSE, (8, 8), g.SYSTEMS["duplo"], "c")
    cell = g.Cell(1, 2, 2, 0)
    sl = g.seat_pose(lego, g.brick_type("lego", "2x2"), cell)["position_m"]
    sd = g.seat_pose(duplo, g.brick_type("duplo", "2x2"), cell)["position_m"]
    for i in (0, 1):
        assert sd[i] - 0.30 * (i == 0) == pytest.approx(2 * (sl[i] - 0.30 * (i == 0)))
    assert sl[2] - 0.02 == pytest.approx(0.0096) and sd[2] - 0.02 == pytest.approx(0.0192)


def test_round_trips_under_random_plate_poses():
    rng = random.Random(0)
    for _ in range(300):
        system = rng.choice(["lego", "duplo"])
        studs = rng.choice(["1x2", "2x2", "2x4", "1x6", "2x3"])
        bt = g.brick_type(system, studs)
        plate = g.Plate(_random_plate_pose(rng), (16, 16), g.SYSTEMS[system], "c")
        cell = g.Cell(
            rng.randint(0, 8), rng.randint(0, 8), rng.randint(1, 4), rng.choice(g.ROTATIONS)
        )
        back, res = g.nearest_cell(plate, bt, g.seat_pose(plate, bt, cell))
        assert (back.x, back.y, back.layer) == (cell.x, cell.y, cell.layer)
        assert g.footprint(bt, back.rotation) == g.footprint(bt, cell.rotation)
        assert res["lateral_m"] < 1e-9 and abs(res["dz_m"]) < 1e-9 and res["tilt_deg"] < 1e-6


def test_engagement_axis_follows_a_rotated_plate():
    rng = random.Random(3)
    pose = _random_plate_pose(rng)
    plate = g.Plate(pose, (16, 16), g.SYSTEMS["lego"], "c")
    s = g.seat_pose(plate, g.brick_type("lego", "2x2"), g.Cell(0, 0, 1, 0))
    assert g.dot(g.column(g.matrix(s), 2), g.column(g.matrix(pose), 2)) == pytest.approx(1.0)


def test_one_stud_off_is_a_different_cell():
    plate = g.Plate(PLATE_POSE, (16, 16), g.SYSTEMS["lego"], "c")
    bt = g.brick_type("lego", "2x4")
    s = g.seat_pose(plate, bt, g.Cell(4, 4, 1, 0))
    moved = g.to_pose(g.mul(g.matrix(s), g.tr(0.008, 0, 0)), WORLD)
    cell, res = g.nearest_cell(plate, bt, moved)
    assert (cell.x, cell.y) == (5, 4) and res["lateral_m"] < 1e-9


@pytest.mark.parametrize(
    "pose",
    [
        {"frame_id": WORLD, "position_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 2]},
        {"frame_id": WORLD, "position_m": [300, 0, 20], "quaternion_xyzw": [0, 0, 0, 1]},
        {"frame_id": WORLD, "position_m": [0, 0], "quaternion_xyzw": [0, 0, 0, 1]},
        {"frame_id": "", "position_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]},
        {"frame_id": WORLD, "position_m": [0, 0, float("nan")], "quaternion_xyzw": [0, 0, 0, 1]},
        {"frame_id": WORLD, "position_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
    ],
)
def test_invalid_poses(pose):
    with pytest.raises(g.GeometryError):
        g.validate_pose(pose)


def test_frame_mismatch_needs_explicit_transform():
    with pytest.raises(g.GeometryError, match="explicit transform"):
        g.validate_pose({**PLATE_POSE, "frame_id": "camera"}, WORLD)


@pytest.mark.parametrize(
    "cell", [{"x": 0.008, "y": 0}, {"x": 1, "y": 1, "rotation": 45}, {"x": True, "y": 0}, {"y": 0}]
)
def test_invalid_cells(cell):
    with pytest.raises(g.GeometryError):
        g.Cell.from_dict(cell)


@pytest.mark.parametrize(
    "args", [("lego", "2x"), ("bionicle", "2x2"), ("lego", [0, 2]), ("lego", [2, 2, 2])]
)
def test_invalid_brick_types(args):
    with pytest.raises(g.GeometryError):
        g.brick_type(*args)


def _checks(plate_sys, brick_sys, cell, build, close_axis=0, frac=0.25):
    plate = g.Plate(PLATE_POSE, (16, 16), g.SYSTEMS[plate_sys], "c")
    return {
        c["name"]: c["ok"]
        for c in g.connection_checks(
            plate, g.brick_type(brick_sys, "2x4"), cell, build, close_axis, frac
        )
    }


def test_interface_mismatch_lego_on_duplo():
    assert _checks("duplo", "lego", g.Cell(0, 0, 1, 0), []) == {"interface_match": False}


def test_collision_support_and_finger_rows():
    lego = g.SYSTEMS["lego"]
    under = g.PlacedBrick("u", g.BrickType(lego, (2, 4)), g.Cell(4, 4, 1, 0))
    assert _checks("lego", "lego", g.Cell(4, 4, 1, 0), [under])["collision_free"] is False
    assert _checks("lego", "lego", g.Cell(10, 10, 2, 0), [under])["supported"] is False
    assert _checks("lego", "lego", g.Cell(4, 4, 2, 0), [under])["supported"] is True
    # A brick in the row beside the jaws (closing along brick x) is struck on approach.
    beside = g.PlacedBrick("n", g.BrickType(lego, (2, 4)), g.Cell(6, 4, 1, 0))
    assert _checks("lego", "lego", g.Cell(4, 4, 1, 0), [beside])["approach_clear_of_build"] is False
    assert (
        _checks("lego", "lego", g.Cell(4, 4, 1, 0), [beside], close_axis=1)[
            "approach_clear_of_build"
        ]
        is True
    )
    assert _checks("lego", "lego", g.Cell(15, 4, 1, 0), [])["on_plate"] is False


def test_tcp_target_inverts_the_grasp():
    seat = g.matrix(
        {
            "frame_id": WORLD,
            "position_m": [0.3, 0.1, 0.05],
            "quaternion_xyzw": [0, 0, 0.3826834, 0.9238795],
        }
    )
    t_tcp_obj = g.matrix(T_TCP_OBJECT)
    tcp = g.tcp_target(seat, t_tcp_obj)
    back = g.mul(tcp, t_tcp_obj)
    assert all(abs(back[i][j] - seat[i][j]) < 1e-9 for i in range(4) for j in range(4))
    # The hand is above the brick (tool z down, brick 2 cm along it).
    assert tcp[2][3] == pytest.approx(0.07)


def test_symmetric_yaw_picks_least_turn_and_refuses_ties():
    seat = g.matrix(g.to_pose(g.tr(0.3, 0, 0.02), WORLD))
    t_tcp_obj = g.matrix(T_TCP_OBJECT)
    hand = g.mul(g.tcp_target(seat, t_tcp_obj), g.rz(170))
    extra, _ = g.choose_yaw(g.brick_type("lego", "2x4"), g.Cell(0, 0, 1, 0), seat, hand, t_tcp_obj)
    assert extra == 180
    hand90 = g.mul(g.tcp_target(seat, t_tcp_obj), g.rz(90))
    with pytest.raises(g.GeometryError, match="silently"):
        g.choose_yaw(g.brick_type("lego", "2x4"), g.Cell(0, 0, 1, 0), seat, hand90, t_tcp_obj)
    extra, _ = g.choose_yaw(
        g.brick_type("lego", "2x2"), g.Cell(0, 0, 1, 0), seat, hand90, t_tcp_obj
    )
    assert extra in (90, 270)


def _load_brick_pose():
    pytest.importorskip("numpy")
    pytest.importorskip("gap")
    path = REPO / "skills/computing-brick-pose-from-stud-grid/scripts/brick_pose.py"
    spec = importlib.util.spec_from_file_location("_rsh_brick_pose", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_matches_computing_brick_pose_original_chain():
    import numpy as np

    bp = _load_brick_pose()
    rng = random.Random(7)
    for _ in range(100):
        system = rng.choice(["lego", "duplo"])
        size = rng.choice([[2, 2], [2, 4], [1, 6], [3, 2]])
        cell = g.Cell(
            rng.randint(0, 6), rng.randint(0, 6), rng.randint(1, 3), rng.choice(g.ROTATIONS)
        )
        bias = (rng.uniform(-3e-4, 3e-4), rng.uniform(-3e-4, 3e-4))
        s = g.SYSTEMS[system]
        ref = bp._brick_top_in_plate(
            [16, 16],
            cell.x,
            cell.y,
            cell.layer,
            cell.rotation,
            size,
            s.stud_pitch_m,
            s.brick_height_m,
            np.array(bias),
        )
        mine = g.top_in_plate((16, 16), g.BrickType(s, tuple(size)), cell, bias)
        assert np.allclose(np.array(mine), ref, atol=1e-9)
    assert bp.PRESETS["lego"]["stud_pitch"] == g.SYSTEMS["lego"].stud_pitch_m
    assert bp.PRESETS["duplo"]["stud_height"] == g.SYSTEMS["duplo"].stud_height_m
    assert bp.FOOTPRINT_PLAY == g.SYSTEMS["lego"].footprint_play_m
