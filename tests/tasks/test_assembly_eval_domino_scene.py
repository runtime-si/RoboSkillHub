# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the domino evaluation scene spec, geometry and baseline plan."""

import math

import pytest

from roboskillhub.tasks.assembly_eval import domino_geometry as geo
from roboskillhub.tasks.assembly_eval.domino_baseline import (
    COMMAND_FAULTS,
    interpolate,
    plan_tile,
)
from roboskillhub.tasks.assembly_eval.domino_scene import (
    DominoSceneSpec,
    TargetLine,
    check_scene,
    scene_summary,
)


def test_default_scene_passes_cpu_checks():
    spec = DominoSceneSpec()
    assert check_scene(spec) == []
    s = scene_summary(spec)
    assert s["n_tiles"] == 4
    assert s["face_gap_m"] == pytest.approx(0.036 - 0.0075)
    bodies = {g["body"] for g in s["collision_geometry"]}
    assert {
        "table",
        "gripper_palm",
        "gripper_pad_l",
        "gripper_pad_r",
        "tile_0",
        "target_0",
    } <= bodies
    # identical tiles
    sizes = {tuple(g["size"]) for g in s["collision_geometry"] if g["body"].startswith("tile_")}
    assert sizes == {(0.0075, 0.024, 0.048)}


@pytest.mark.parametrize("n", [3, 5])
def test_three_to_five_tiles(n):
    assert check_scene(DominoSceneSpec(n_tiles=n)) == []


def test_tile_count_outside_first_scene_rejected():
    assert any(i["code"] == "tile_count" for i in check_scene(DominoSceneSpec(n_tiles=6)))


def test_slots_overlapping_rejected():
    spec = DominoSceneSpec(line=TargetLine(pitch=0.006))
    codes = {i["code"] for i in check_scene(spec)}
    assert "slot_overlap" in codes


def test_tight_pitch_along_facing_keeps_hand_clear():
    # 12 mm centre pitch (4.5 mm face gap): tiles do not overlap and the pads, which
    # stand beside the tile across its width, do not reach the neighbours.
    spec = DominoSceneSpec(line=TargetLine(pitch=0.012))
    codes = {i["code"] for i in check_scene(spec)}
    assert "slot_overlap" not in codes and "hand_near_neighbour" not in codes


def test_hand_near_neighbour_for_parallel_tiles():
    # Tiles facing +x but the row running along +y (side by side across their width):
    # build it by rotating the facing only via a custom spec subclass.
    class SideBySide(DominoSceneSpec):
        def slot_pose(self, k):
            p = super().slot_pose(k)
            p = dict(p)
            p["position"] = [
                self.line.start_xy[0],
                self.line.start_xy[1] + 0.030 * k,
                p["position"][2],
            ]
            p["yaw"] = 0.0
            p["quat_xyzw"] = list(geo.quat_from_yaw(0.0))
            return p

    issues = check_scene(SideBySide())
    assert any(i["code"] == "hand_near_neighbour" for i in issues)


def test_spec_roundtrip_and_hash_changes():
    spec = DominoSceneSpec()
    again = DominoSceneSpec.from_dict(spec.to_dict())
    assert again.config_hash() == spec.config_hash()
    other = DominoSceneSpec(line=TargetLine(pitch=0.040))
    assert other.config_hash() != spec.config_hash()
    with pytest.raises(ValueError):
        DominoSceneSpec.from_dict({**spec.to_dict(), "schema": "nope"})


def test_tilt_and_yaw_symmetry():
    up = geo.quat_from_yaw(0.3)
    assert geo.tilt_deg(up) == pytest.approx(0.0, abs=1e-6)
    lean = geo.quat_mul(
        geo.quat_from_yaw(0.3), geo.quat_from_axis_angle((0, 1, 0), math.radians(12))
    )
    assert geo.tilt_deg(lean) == pytest.approx(12.0, abs=1e-6)
    flipped = geo.quat_from_axis_angle((1, 0, 0), math.pi)  # upside down still stands
    assert geo.tilt_deg(flipped) == pytest.approx(0.0, abs=1e-6)
    lying = geo.quat_from_axis_angle((0, 1, 0), math.pi / 2)
    assert geo.tilt_deg(lying) == pytest.approx(90.0, abs=1e-6)
    assert geo.yaw_error_deg(geo.quat_from_yaw(math.pi + 0.01), 0.0) == pytest.approx(
        math.degrees(0.01)
    )
    with pytest.raises(ValueError):
        geo.quat_normalize((0, 0, 0, 0))


def test_polygon_gap():
    a = geo.rect_corners(0, 0, 0, 0.01, 0.01)
    b = geo.rect_corners(0.03, 0, 0, 0.01, 0.01)
    assert geo.polygon_gap(a, b) == pytest.approx(0.01)
    c = geo.rect_corners(0.015, 0, 0.3, 0.01, 0.01)
    assert geo.polygon_gap(a, c) < 0


def test_plan_shapes_and_faults():
    spec = DominoSceneSpec()
    tile = spec.feeder_pose(0)
    start = (0.0, 0.0, spec.transit_tcp_z(), 0.0)
    nominal = plan_tile(spec, 0, 0, tile, start, 0.04, 1.0)
    phases = [s.phase for s in nominal.segments]
    for ph in (
        "descend_pick",
        "close",
        "lift",
        "carry",
        "descend_place",
        "release",
        "retreat",
        "observe",
    ):
        assert ph in phases
    rel = next(s for s in nominal.segments if s.phase == "descend_place")
    assert rel.pose[2] == pytest.approx(
        spec.table.top_z + spec.place.release_gap + spec.place.grasp_height
    )
    hot = plan_tile(spec, 0, 0, tile, start, 0.04, 1.0, fault="table_height_error")
    hrel = next(s for s in hot.segments if s.phase == "descend_place")
    assert hrel.pose[2] == pytest.approx(rel.pose[2] + 0.06)
    nr = plan_tile(spec, 0, 0, tile, start, 0.04, 1.0, fault="no_release")
    assert next(s for s in nr.segments if s.phase == "release").opening < spec.tile.width
    for f in COMMAND_FAULTS:
        assert plan_tile(spec, 1, 1, tile, start, 0.04, 1.0, fault=f).segments
    with pytest.raises(ValueError):
        plan_tile(spec, 0, 0, tile, start, 0.04, 1.0, fault="teleport")


def test_interpolate_endpoints():
    spec = DominoSceneSpec()
    plan = plan_tile(spec, 0, 0, spec.feeder_pose(0), (0, 0, 0.2, 0), 0.04, 1.0)
    seg = plan.segments[1]
    p0, o0 = interpolate(seg, (0, 0, 0.2, 0), 0.04, 0.0)
    p1, o1 = interpolate(seg, (0, 0, 0.2, 0), 0.04, 1.0)
    assert p0 == pytest.approx((0, 0, 0.2, 0))
    assert p1 == pytest.approx(seg.pose)
    assert o1 == pytest.approx(seg.opening)
