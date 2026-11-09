"""verifying-brick-seated-by-height: FakeContext tests on rendered wrist depth.

A fake arm carries a wrist D405 (640 x 480, fx 394 px, tilted 30 deg on the tool) over a
ray-cast 12 x 12 baseplate with one brick posed seated, lifted, tilted or off its cell
(_seat_eval.World); every frame gets fresh D405-like noise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ._plate_render import Brick
from ._seat_eval import CELL, World, ring_neighbours

BUNDLE = "verifying-brick-seated-by-height"
DUPLO_STUD, LEGO_STUD = 0.0045, 0.0017


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["verify_seat"].module


def check(script, world, **kw):
    kw.setdefault("system", world.system)
    kw.setdefault("brick_studs", list(world.studs))
    seat = kw.pop("seat", None) or world.seat()
    return script.run(world.ctx(), seat_position=seat, **kw)


def test_seated_duplo_reads_seated_from_a_view_straight_down(script):
    w = World()
    out = check(script, w)
    assert out["route"] == "seated", out["detail"]
    assert abs(out["height_error"]) < 0.0005
    assert out["tilt_deg"] < 1.0
    assert out["coverage"] > 0.95
    assert out["reference"] == "local"
    assert out["samples"] > 1000
    # One view move, its camera looking straight down at the brick's top.
    assert len(w.moves) == 1 and len(w.ik_calls) == 1
    cam = w.camera_pose()
    assert cam[2, 2] < -0.999
    top = np.array([w.seat()[k] for k in "xyz"]) + [0, 0, 0.0192]
    assert np.linalg.norm(cam[:3, 3] - (top + [0, 0, 0.20])) < 1e-6


def test_partly_pressed_is_not_seated(script):
    out = check(script, World(dz=0.4 * DUPLO_STUD))
    assert out["route"] == "not_seated", out["detail"]
    assert out["height_error"] == pytest.approx(0.4 * DUPLO_STUD, abs=0.0004)


def test_resting_on_the_stud_tops_is_high_by_stud(script):
    out = check(script, World(dz=DUPLO_STUD))
    assert out["route"] == "high_by_stud", out["detail"]
    assert out["height_error"] == pytest.approx(DUPLO_STUD, abs=0.0004)


def test_one_end_on_the_studs_is_tilted(script):
    out = check(script, World(tilt_deg=8.0))
    assert out["route"] == "tilted", out["detail"]
    assert out["tilt_deg"] == pytest.approx(8.0, abs=0.5)
    assert out["edge_rise"] > DUPLO_STUD * 0.5


def test_one_cell_off_is_shifted_and_points_the_way(script):
    out = check(script, World(dx=0.016))
    assert out["route"] == "shifted_off_grid", out["detail"]
    assert 0.4 < out["coverage"] < 0.7
    shift = out["shift_estimate"]
    assert shift["x"] > 0.008 and abs(shift["y"]) < 0.002


def test_half_a_cell_off_on_the_studs_reports_resting_with_the_offset(script):
    out = check(script, World(dx=0.008, dz=DUPLO_STUD))
    assert out["route"] == "high_by_stud", out["detail"]
    assert "off the footprint" in out["detail"]


def test_no_brick_at_the_seat(script):
    out = check(script, World(brick=False))
    assert out["route"] == "no_brick", out["detail"]
    assert out["coverage"] < 0.05


def test_a_camera_height_error_cancels_on_the_local_reference(script):
    w = World(camera_error=(0.0, 0.0, 0.004))
    out = check(script, w)
    assert out["route"] == "seated", out["detail"]
    assert out["reference_offset"] == pytest.approx(0.004, abs=0.0005)
    # Forced onto the world reference, the same frame reads the brick high.
    out = check(script, World(camera_error=(0.0, 0.0, 0.004)), reference="world")
    assert out["route"] == "high_by_stud", out["detail"]
    assert out["reference"] == "world"


def test_flush_neighbours_do_not_disturb_the_reading(script):
    w = World(neighbours=ring_neighbours((2, 2)))
    out = check(script, w)
    assert out["route"] == "seated", out["detail"]
    assert out["reference"] == "local"


def test_a_small_plate_reads_its_own_surface_not_the_table_past_its_edge(script):
    # A 2x2 in the middle of a 4x4 plate (the station twin's check plate): the ring reaches
    # past the plate's edge onto the table 5 mm lower, which must not become the reference.
    w = World(plate=(4, 4), cell=(1, 1))
    out = check(script, w)
    assert out["route"] == "seated", out["detail"]
    assert out["reference"] == "local"
    assert abs(out["reference_offset"]) < 0.0005
    out = check(script, World(plate=(4, 4), cell=(1, 1), dx=0.016))
    assert out["route"] == "shifted_off_grid", out["detail"]
    out = check(script, World(plate=(4, 4), cell=(1, 1), dx=0.008, dz=DUPLO_STUD))
    assert out["route"] == "high_by_stud", out["detail"]


def test_the_grippers_own_fingers_over_the_brick_are_left_out(script):
    # A camera beside the fingers sees them over the footprint without them hiding it: a
    # finger-sized patch of points 12 cm over the seat, added to the rendered frame.
    w = World()
    check(script, w)
    cam = w.observation()["cameras"][0]
    pts = script.world_points(cam, np.asarray(cam["depth"], dtype=float))
    seat = np.array([w.seat()[k] for k in "xyz"])
    gx, gy = np.meshgrid(np.linspace(-0.01, 0.01, 40), np.linspace(-0.012, 0.012, 40))
    finger = np.stack(
        [seat[0] + gx.ravel(), seat[1] + gy.ravel(), np.full(gx.size, seat[2] + 0.12)], 1
    )
    args = (np.vstack([pts, finger]), seat, 0.0, 0.0318, 0.0318, 0.016, 0.0045, 0.0192)
    m = script.measure(*args)
    assert abs(m["height_error"]) < 0.0005 and m["tilt_deg"] < 1.0
    # Under the cap (3.5 cm over the seated top) the trimmed plane fit leaves them out too.
    low = np.vstack([pts, finger - [0.0, 0.0, 0.12 - 0.035]])
    m = script.measure(low, *args[1:])
    assert abs(m["height_error"]) < 0.0005 and m["tilt_deg"] < 1.0


def test_stacked_on_a_same_footprint_brick_falls_back_to_the_seat_height(script):
    # The brick under test on layer 2, over a 2x2 on layer 1 at the same cell: the plate
    # around reads a whole brick under the seat, so the seat's own z is the reference.
    below = [Brick(CELL[0], CELL[1], 2, 2, layer=1)]
    w = World(layer=2, neighbours=below)
    out = check(script, w)
    assert out["route"] == "seated", out["detail"]
    assert out["reference"] == "world"
    out = check(script, World(layer=2, neighbours=below), reference="local")
    assert out["route"] == "unverified"
    assert "no plate surface" in out["detail"]


def test_lego_2x4_turned_90_deg_reads_by_its_own_footprint(script):
    seated = World(system="lego", studs=(2, 4))
    assert check(script, seated)["route"] == "seated"
    high = check(script, World(system="lego", studs=(2, 4), dz=LEGO_STUD))
    assert high["route"] == "high_by_stud", high["detail"]
    tilt = math.degrees(math.atan(LEGO_STUD / 0.032))
    tilted = check(script, World(system="lego", studs=(2, 4), tilt_deg=tilt))
    assert tilted["route"] == "tilted", tilted["detail"]
    # Told the brick is 4x2 (turned 90 deg), the 2x4 covers only half that footprint.
    wrong = check(script, World(system="lego", studs=(2, 4)), brick_studs=[4, 2])
    assert wrong["route"] == "shifted_off_grid", wrong["detail"]


def test_lego_partly_pressed_is_not_seated(script):
    out = check(script, World(system="lego", studs=(2, 4), dz=0.45 * LEGO_STUD))
    assert out["route"] == "not_seated", out["detail"]


def test_current_view_reads_without_moving(script):
    w = World()
    # Put the camera straight over the brick first, then read in place.
    w.r = np.diag([1.0, -1.0, -1.0])
    tc = np.linalg.inv(w.tool()) @ w.camera_pose()
    seat = np.array([w.seat()[k] for k in "xyz"])
    cam = np.eye(4)
    cam[:3, :3] = w.camera_pose()[:3, :3]
    cam[:3, 3] = seat + [0, 0, 0.0192] - cam[:3, 2] * 0.22
    tool = cam @ np.linalg.inv(tc)
    w.p, w.r = tool[:3, 3], tool[:3, :3]
    out = check(script, w, view="current")
    assert out["route"] == "seated", out["detail"]
    assert w.moves == [] and w.ik_calls == []


def test_a_view_that_stops_short_is_still_read_from_where_the_camera_is(script):
    w = World(short_by=0.01)
    out = check(script, w)
    assert out["route"] == "seated", out["detail"]
    assert "left_eye_in_hand" in out["viewed_from"]


def test_no_camera_is_unverified_and_nothing_moves(script):
    w = World(camera=False)
    out = check(script, w)
    assert out["route"] == "unverified"
    assert w.moves == []


def test_views_out_of_reach_are_unreachable_without_moving(script):
    w = World(ik_ok=False)
    out = check(script, w)
    assert out["route"] == "unreachable", out["detail"]
    assert w.moves == [] and len(w.ik_calls) == 4


@pytest.mark.parametrize("how", ["raise", "route"])
def test_refused_view_moves_are_unreachable(script, how):
    w = World(refuse=how)
    out = check(script, w)
    assert out["route"] == "unreachable", out["detail"]
    assert len(w.moves) == 4


def test_a_wrong_seat_reads_off_grid_or_no_brick(script):
    # Told the seat is two cells over: nothing there.
    w = World()
    seat = w.seat(dx=0.032)
    assert check(script, w, seat=seat)["route"] == "no_brick"


def test_arm_id_is_passed_only_when_set(script):
    w = World()
    check(script, w)
    assert "arm_id" not in w.moves[0] and "arm_id" not in w.ik_calls[0]
    w = World()
    w_obs = w.observation

    def two_arms(**kw):
        obs = w_obs(**kw)
        obs["arms"] = [{"ee_pose": obs["arms"][0]["ee_pose"]}, obs["arms"][0]]
        return obs

    w.observation = two_arms
    out = check(script, w, arm_id=1)
    assert out["route"] == "seated", out["detail"]
    assert w.moves[0]["arm_id"] == 1 and w.ik_calls[0]["arm_id"] == 1


def test_explicit_tolerances_override_the_defaults(script):
    out = check(script, World(dz=0.4 * DUPLO_STUD), height_tolerance=0.002)
    assert out["route"] == "seated", out["detail"]
    out = check(script, World(tilt_deg=8.0), tilt_tolerance_deg=10.0)
    assert out["route"] == "high_by_stud" or out["route"] == "not_seated", out["detail"]


@pytest.mark.parametrize(
    "kwargs",
    [{"system": "megablok"}, {"view": "side"}, {"reference": "plate"}, {"brick_studs": [0, 2]}],
)
def test_bad_arguments_raise(script, kwargs):
    with pytest.raises(ValueError):
        script.run(World().ctx(), seat_position=World().seat(), **kwargs)


def test_every_route_is_declared_in_the_example(script, skills_registry):
    bundle = skills_registry.get(BUNDLE)
    example = json.loads(
        (Path(bundle.bundle_dir) / "examples" / "canonical_subgraph.json").read_text()
    )
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert set(script.ROUTES) <= declared
    skill_md = (Path(bundle.bundle_dir) / "SKILL.md").read_text()
    for route in script.ROUTES:
        assert f"    {route}:" in skill_md
