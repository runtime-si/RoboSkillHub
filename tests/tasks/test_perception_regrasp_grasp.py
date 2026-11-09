"""Grasp candidates for the YAM linear_4310: jaw span, contact, table and
neighbour clearance, and the grasp request they feed."""

from __future__ import annotations

import math

import numpy as np
from _perception_regrasp_helpers import SPEC, resting

from roboskillhub.runtime.contracts import validate_request
from roboskillhub.tasks.perception_regrasp import (
    LINEAR_4310,
    GraspLimits,
    lego_brick,
    propose_grasps,
    to_grasp_request,
)
from roboskillhub.tasks.perception_regrasp.frames import make_T, pose_to_T, rot_z
from roboskillhub.tasks.perception_regrasp.grasp import hand_lowest, obb_overlap
from roboskillhub.tasks.perception_regrasp.objects import ObjectSpec, symmetry_rotations

UP = np.array([0.0, 0.0, 1.0])


def test_lying_tile_is_taken_from_above_as_low_as_the_station_allows():
    cands = propose_grasps("d1", resting("lying"), SPEC, "world", 0.0)
    ok = [c for c in cands if c.feasible]
    assert ok, "a lying tile must have a top grasp"
    for c in ok:
        assert c.approach_dir @ UP < -0.99  # straight down
        assert c.closing_axis in ("x", "y")  # never across the 7.5 mm thickness from the side
        assert c.T_frame_tcp[2, 3] >= GraspLimits().min_tcp_height_m - 1e-9
        assert c.checks["table_clearance"]["lowest_m"] >= GraspLimits().table_clearance_m - 1e-9
        assert 0.0 < c.insertion_m < SPEC.size_m[2]
    side = [c for c in cands if abs(c.approach_dir @ UP) < 0.1]
    assert side and not any(c.feasible for c in side)
    assert all(not c.checks["table_clearance"]["ok"] for c in side)


def test_widths_and_tcp_object_transform_are_consistent():
    T = resting("standing")
    for c in propose_grasps("d1", T, SPEC, "world", 0.0):
        assert np.allclose(T @ c.T_object_tcp, c.T_frame_tcp)
        assert np.allclose(c.T_frame_tcp @ c.T_tcp_object, T)
        assert c.close_width_m < c.grasp_width_m < c.open_width_m <= LINEAR_4310.max_opening_m
        y = c.T_frame_tcp[:3, 1]  # jaw axis
        size_along = sum(abs(y @ T[:3, k]) * SPEC.size_m[k] for k in range(3))
        assert math.isclose(size_along, c.grasp_width_m, rel_tol=1e-6)


def test_too_wide_parts_get_no_jaw_width():
    big = ObjectSpec("box", (0.12, 0.10, 0.09), "box")
    cands = propose_grasps("b", make_T(position=[0.3, 0, 0.045]), big, "world", 0.0)
    assert cands == []  # every closing axis exceeds the 86 mm usable span


def test_raised_table_and_station_tcp_floor_reject():
    T = resting("lying")
    low = GraspLimits(min_tcp_height_m=0.0, table_clearance_m=0.0)
    deep = [c for c in propose_grasps("d1", T, SPEC, "world", 0.0, limits=low) if c.feasible]
    assert min(c.T_frame_tcp[2, 3] for c in deep) < 0.014  # without the station floor it goes lower
    strict = GraspLimits(min_tcp_height_m=0.02)
    assert not any(c.feasible for c in propose_grasps("d1", T, SPEC, "world", 0.0, limits=strict))


def test_a_neighbour_beside_the_jaws_blocks_those_grasps():
    T = resting("standing", yaw=0.0, xy=(0.3, 0.0))
    free = {
        (c.closing_axis, c.approach_face)
        for c in propose_grasps("d1", T, SPEC, "world", 0.0)
        if c.feasible
    }
    # A second standing tile 12 mm beside the thickness faces (domino-row spacing).
    nb = resting("standing", yaw=0.0, xy=(0.3, 0.0))
    nb[:3, 3] += T[:3, 2] * (SPEC.size_m[2] + 0.012)
    cands = propose_grasps("d1", T, SPEC, "world", 0.0, neighbours=[("d2", nb, SPEC)])
    blocked = {(c.closing_axis, c.approach_face) for c in cands if c.feasible}
    assert ("z", "-x") in free and ("z", "-x") not in blocked  # jaws across the thickness hit d2
    hits = [
        c.checks["neighbours"]["hits"] for c in cands if c.closing_axis == "z" and not c.feasible
    ]
    assert any(any(":d2" in h for h in hs) for hs in hits)


def test_hand_boxes_follow_the_tcp_and_the_opening():
    T = make_T(np.diag([1.0, -1.0, -1.0]), [0.3, 0, 0.05])  # pointing down
    low_open = hand_lowest(T, LINEAR_4310, 0.09, UP)
    assert math.isclose(min(low_open.values()), 0.05 - LINEAR_4310.tip_reach_m, abs_tol=1e-9)
    a = make_T(position=[0, 0, 0])
    assert obb_overlap(
        a, np.array([0.01] * 3), make_T(position=[0.015, 0, 0]), np.array([0.01] * 3)
    )
    assert not obb_overlap(
        a, np.array([0.01] * 3), make_T(position=[0.025, 0, 0]), np.array([0.01] * 3)
    )
    rot = make_T(rot_z(math.pi / 4), [0.0245, 0, 0])  # corner-on: reaches 24.5 - 14.1 = 10.4 mm
    assert not obb_overlap(a, np.array([0.01] * 3), rot, np.array([0.01] * 3))
    assert obb_overlap(
        a, np.array([0.01] * 3), make_T(rot_z(math.pi / 4), [0.0235, 0, 0]), np.array([0.01] * 3)
    )


def test_grasp_request_is_a_valid_runtime_grasp():
    c = next(c for c in propose_grasps("d1", resting("lying"), SPEC, "world", 0.0) if c.feasible)
    req = to_grasp_request(c)
    validate_request("grasp", req)
    pre, g, lift = (pose_to_T(req[k]) for k in ("pregrasp_pose", "grasp_pose", "lift_pose"))
    assert np.allclose(g, c.T_frame_tcp)
    assert np.allclose(pre[:3, 3], g[:3, 3] - 0.04 * c.approach_dir)
    assert lift[2, 3] - g[2, 3] >= 0.02  # the grasp primitive needs >= 2 cm of lift


def test_symmetric_targets_give_the_same_part_placement():
    """180 deg twins of a target must not change where the part ends up."""
    T_goal = resting("standing", yaw=0.7)
    c = next(c for c in propose_grasps("d1", resting("lying"), SPEC, "world", 0.0) if c.feasible)
    corners = (
        np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        * np.asarray(SPEC.size_m)
        / 2
    )

    def occupied(T_obj):
        return np.sort(
            np.round((corners @ T_obj[:3, :3].T + T_obj[:3, 3]) * 1e6).astype(int), axis=0
        )

    ref = occupied(T_goal)
    for S in symmetry_rotations(SPEC):
        Tg = T_goal.copy()
        Tg[:3, :3] = T_goal[:3, :3] @ S
        T_tcp = Tg @ c.T_object_tcp  # where the hand goes for this twin
        assert np.array_equal(occupied(T_tcp @ c.T_tcp_object), ref)


def test_lego_brick_top_grasp_across_the_short_side():
    b = lego_brick((2, 4))
    T = make_T(position=[0.3, 0.0, b.size_m[2] / 2])
    best = next(c for c in propose_grasps("b1", T, b, "world", 0.0) if c.feasible)
    assert best.closing_axis == "x" and math.isclose(best.grasp_width_m, 0.0158, abs_tol=1e-6)
    assert best.approach_dir @ UP < -0.99
