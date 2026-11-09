"""Frames, units, quaternions (xyzw), the object origin / up-axis convention, TCP<->object
transforms and yaw symmetry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roboskillhub.tasks.domino import geometry as geo
from roboskillhub.tasks.domino.contract import (
    RequestRejected,
    check_geometry,
    check_pose,
    check_support,
)

F = "station"


def test_quaternion_matrix_round_trip_is_xyzw():
    q = geo.yaw_quat(math.radians(90))
    assert q == pytest.approx([0, 0, math.sqrt(0.5), math.sqrt(0.5)])
    r = geo.quat_matrix(q)
    assert r @ np.array([1, 0, 0]) == pytest.approx([0, 1, 0], abs=1e-12)
    for q in ([0.1, -0.2, 0.3, 0.9], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]):
        q = np.array(q) / np.linalg.norm(q)
        back = geo.matrix_quat(geo.quat_matrix(q))
        assert abs(float(np.dot(back, q))) == pytest.approx(1.0)


def test_tcp_object_transforms_are_inverse():
    goal = geo.pose(F, [0.4, 0.1, 0.0], geo.yaw_quat(0.7))
    t_tcp_obj = geo.pose("tcp", [0.003, 0.0, 0.03], [1, 0, 0, 0])
    tcp = geo.tcp_for_object(goal, t_tcp_obj)
    assert tcp["frame_id"] == F
    back = geo.object_from_tcp(tcp, t_tcp_obj)
    assert back["position_m"] == pytest.approx(goal["position_m"], abs=1e-12)
    assert abs(np.dot(back["quaternion_xyzw"], goal["quaternion_xyzw"])) == pytest.approx(1.0)
    # TCP is 30 mm over the tile's bottom (its origin), pointing down.
    assert tcp["position_m"][2] == pytest.approx(0.03)
    assert geo.quat_matrix(tcp["quaternion_xyzw"])[:, 2] == pytest.approx([0, 0, -1], abs=1e-12)


def test_tilt_uses_the_object_up_axis_against_the_support_normal():
    upright = geo.pose(F, [0, 0, 0], geo.yaw_quat(1.0))
    assert geo.tilt_rad(upright, [0, 0, 1]) == pytest.approx(0, abs=1e-9)
    upside_down = geo.pose(F, [0, 0, 0.048], [1, 0, 0, 0])
    assert geo.tilt_rad(upside_down, [0, 0, 1]) == pytest.approx(0, abs=1e-9)
    assert geo.tilt_rad(upside_down, [0, 0, 1], up_symmetric=False) == pytest.approx(math.pi)
    # A sloped support: an object upright in world z is tilted against it.
    n = [0, math.sin(0.2), math.cos(0.2)]
    assert geo.tilt_rad(upright, n) == pytest.approx(0.2)
    # Centre height alone would not see this: same z, lying.
    lying = geo.pose(
        F, [0, 0, 0], geo.matrix_quat(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float))
    )
    assert geo.tilt_rad(lying, [0, 0, 1]) == pytest.approx(math.pi / 2)


def test_yaw_symmetry_folds_only_when_declared():
    a = geo.pose(F, [0, 0, 0], geo.yaw_quat(0.0))
    b = geo.pose(F, [0, 0, 0], geo.yaw_quat(math.pi))
    assert geo.yaw_error_rad(a, b, [0, 0, 1], 180.0) == pytest.approx(0, abs=1e-9)
    assert geo.yaw_error_rad(a, b, [0, 0, 1], 360.0) == pytest.approx(math.pi)
    c = geo.pose(F, [0, 0, 0], geo.yaw_quat(math.radians(170)))
    assert math.degrees(geo.yaw_error_rad(a, c, [0, 0, 1], 180.0)) == pytest.approx(10)


def test_lowest_point_is_measured_from_the_bottom_face_origin():
    g = check_geometry({"height_m": 0.048, "width_m": 0.024, "thickness_m": 0.0075})
    p = geo.pose(F, [0, 0, 0.002], geo.yaw_quat(0.3))
    assert geo.lowest_point_height(p, g, [0, 0, 0], [0, 0, 1]) == pytest.approx(0.002)


@pytest.mark.parametrize(
    "bad",
    [
        {"frame_id": F, "position_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 2]},
        {"frame_id": "camera", "position_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]},
        {"frame_id": F, "position_m": [0, 0], "quaternion_xyzw": [0, 0, 0, 1]},
        {"frame_id": F, "position_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
        {"frame_id": F, "position_m": [0, 0, float("nan")], "quaternion_xyzw": [0, 0, 0, 1]},
    ],
)
def test_pose_validation_rejects(bad):
    with pytest.raises(RequestRejected):
        check_pose(bad, "p", F)


def test_geometry_and_support_contract():
    with pytest.raises(RequestRejected):
        check_geometry(
            {"height_m": 0.048, "width_m": 0.024, "thickness_m": 0.0075, "origin": "center"}
        )
    with pytest.raises(RequestRejected):  # millimetres passed as metres
        check_geometry({"height_m": 48, "width_m": 24, "thickness_m": 7.5})
    with pytest.raises(RequestRejected):
        check_support(
            {"frame_id": F, "point_m": [0, 0, 0], "normal": [0, 0, 1], "height_std_m": 0.001}, F
        )
    s = check_support(
        {
            "frame_id": F,
            "point_m": [0, 0, 0],
            "normal": [0, 0, 2],
            "height_std_m": 0.001,
            "calibration_version": "c1",
        },
        F,
    )
    assert s["normal"] == [0, 0, 1]
