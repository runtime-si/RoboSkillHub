"""classifying-brick-resting-face-from-depth: FakeContext tests on rendered wrist depth.

A fake arm carries a wrist D405 (640 x 480, fx 394 px, tilted 30 deg on the tool) over hollow
ray-cast bricks lying on a table (_face_eval.World); every frame gets fresh D405-like noise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ._face_eval import World, true_up

BUNDLE = "classifying-brick-resting-face-from-depth"


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["classify_face"].module


def classify(script, world, dx=0.004, dy=-0.003, **kw):
    kw.setdefault("system", world.system)
    kw.setdefault("brick_studs", list(world.studs))
    return script.run(world.ctx(), brick_position=world.position(dx, dy), **kw)


def _up(out):
    return np.array([out["up_vector"][k] for k in "xyz"])


def _yaw_err(a, b, period):
    return abs((a - b + period / 2) % period - period / 2)


def test_studs_up_duplo_2x4_with_its_yaw_and_centre(script):
    w = World(studs=(2, 4), yaw_deg=33.0)
    out = classify(script, w)
    assert out["route"] == "studs_up", out["detail"]
    assert out["vertical_axis"] == "z"
    assert _up(out) == pytest.approx([0, 0, 1])
    # 2x4: the brick's x axis is the short side; yaw is that axis mod 180.
    assert _yaw_err(out["yaw_deg"], 33.0, 180.0) < 2.0
    pose = w.brick_pose_world()
    body = pose[:3, 3] + pose[:3, :3] @ np.array([0, 0, 0.0096])
    c = np.array([out["centre"][k] for k in "xyz"])
    assert np.linalg.norm(c - body) < 0.002
    assert out["table_height"] == pytest.approx(0.02, abs=0.0005)
    # One view, the camera looking straight down.
    assert len(w.moves) == 1 and w.camera_pose()[2, 2] < -0.999


def test_upside_down_shows_its_hollow_underside(script):
    out = classify(script, World(studs=(2, 4), face="upside_down", yaw_deg=62.0))
    assert out["route"] == "upside_down", out["detail"]
    assert _up(out) == pytest.approx([0, 0, -1])
    assert out["deep_share"] > 0.5


def test_on_a_side_names_the_vertical_axis_and_the_stud_direction(script):
    w = World(studs=(2, 4), face="side_x", yaw_deg=17.0)
    out = classify(script, w)
    assert out["route"] == "on_side", out["detail"]
    assert out["vertical_axis"] == "x"
    assert out["stud_direction_known"]
    assert float(_up(out) @ true_up(w)) > 0.95
    w = World(studs=(2, 4), face="side_y", yaw_deg=105.0)
    out = classify(script, w)
    assert out["route"] == "on_side", out["detail"]
    assert out["vertical_axis"] == "y"
    assert float(_up(out) @ true_up(w)) > 0.95


def test_square_brick_yaw_is_mod_90(script):
    out = classify(script, World(studs=(2, 2), yaw_deg=62.0))
    assert out["route"] == "studs_up", out["detail"]
    assert -45.0 <= out["yaw_deg"] < 45.0
    assert _yaw_err(out["yaw_deg"], 62.0, 90.0) < 2.0


@pytest.mark.parametrize("face", ["studs_up", "upside_down", "side_x", "side_y"])
def test_lego_2x4_every_face(script, face):
    out = classify(script, World(system="lego", studs=(2, 4), face=face, yaw_deg=17.0))
    want = {"side_x": "on_side", "side_y": "on_side"}.get(face, face)
    assert out["route"] == want, out["detail"]


@pytest.mark.parametrize("face", ["studs_up", "upside_down", "side_y"])
def test_lego_1x2_every_face(script, face):
    out = classify(script, World(system="lego", studs=(1, 2), face=face, yaw_deg=33.0))
    want = {"side_y": "on_side"}.get(face, face)
    assert out["route"] == want, out["detail"]


def test_a_plate_lying_flat_is_not_judged_up_or_down(script):
    w = World(system="lego", studs=(2, 4), brick_height=0.0032)
    out = classify(script, w, brick_height=0.0032)
    assert out["route"] == "unknown"
    assert "too shallow" in out["detail"]
    assert out["vertical_axis"] == "z"


def test_touching_bricks_fit_no_face(script):
    w = World(others=[("duplo", 2, 2, "studs_up", 0.032, 0.0, 0.0)])
    out = classify(script, w)
    assert out["route"] == "unknown"
    assert "no resting face" in out["detail"]


def test_a_neighbour_two_centimetres_off_is_left_out(script):
    w = World(others=[("duplo", 2, 4, "side_x", 0.06, 0.0, 30.0)])
    out = classify(script, w)
    assert out["route"] == "studs_up", out["detail"]


def test_a_wrong_size_fits_no_face(script):
    out = classify(script, World(studs=(2, 2)), brick_studs=[2, 4])
    assert out["route"] == "unknown"


def test_no_brick_near_the_position(script):
    out = classify(script, World(brick=False))
    assert out["route"] == "no_brick"


def test_a_given_table_height_is_used(script):
    out = classify(script, World(), table_height=0.02)
    assert out["route"] == "studs_up", out["detail"]
    assert out["table_height"] == 0.02


def test_current_view_reads_without_moving(script):
    w = World(studs=(2, 4), face="upside_down")
    tc = np.linalg.inv(w.tool()) @ w.camera_pose()
    cam = np.eye(4)
    cam[:3, :3] = np.diag([1.0, -1.0, -1.0])
    cam[:3, 3] = np.array([w.position()[k] for k in "xyz"]) + [0, 0, 0.23]
    tool = cam @ np.linalg.inv(tc)
    w.p, w.r = tool[:3, 3], tool[:3, :3]
    out = classify(script, w, view="current")
    assert out["route"] == "upside_down", out["detail"]
    assert w.moves == [] and w.ik_calls == []


def test_no_camera_is_unknown_and_nothing_moves(script):
    w = World(camera=False)
    out = classify(script, w)
    assert out["route"] == "unknown"
    assert w.moves == []


def test_views_out_of_reach_are_unreachable(script):
    w = World(ik_ok=False)
    out = classify(script, w)
    assert out["route"] == "unreachable"
    assert w.moves == [] and len(w.ik_calls) == 4


@pytest.mark.parametrize("how", ["raise", "route"])
def test_refused_view_moves_are_unreachable(script, how):
    w = World(refuse=how)
    out = classify(script, w)
    assert out["route"] == "unreachable", out["detail"]
    assert len(w.moves) == 4


def test_arm_id_is_passed_only_when_set(script):
    w = World()
    classify(script, w)
    assert "arm_id" not in w.moves[0] and "arm_id" not in w.ik_calls[0]
    w = World()
    base = w.observation

    def two_arms(**kw):
        obs = base(**kw)
        obs["arms"] = [{"ee_pose": obs["arms"][0]["ee_pose"]}, obs["arms"][0]]
        return obs

    w.observation = two_arms
    out = classify(script, w, arm_id=1)
    assert out["route"] == "studs_up", out["detail"]
    assert w.moves[0]["arm_id"] == 1 and w.ik_calls[0]["arm_id"] == 1


def test_faces_table_matches_the_brick_geometry(script):
    f = script.faces(2, 4, 0.008, 0.0017, 0.0096)
    assert f["z"]["height"] == pytest.approx(0.0113)
    assert f["x"]["height"] == pytest.approx(0.0158)
    assert f["y"]["height"] == pytest.approx(0.0318)
    assert f["y"]["sides"] == pytest.approx((0.0158, 0.0113))
    assert "y" not in script.faces(2, 2, 0.016, 0.0045, 0.0192)
    assert math.isclose(f["z"]["sides"][0], 0.0318)


@pytest.mark.parametrize(
    "kwargs", [{"system": "megablok"}, {"view": "side"}, {"brick_studs": [0, 2]}]
)
def test_bad_arguments_raise(script, kwargs):
    with pytest.raises(ValueError):
        script.run(World().ctx(), brick_position=World().position(), **kwargs)


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
