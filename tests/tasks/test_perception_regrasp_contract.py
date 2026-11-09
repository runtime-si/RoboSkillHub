"""Frames, the observation contract and the v1 wire forms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roboskillhub.runtime.task import validate
from roboskillhub.tasks.perception_regrasp import (
    ObjectObservation,
    SimTruthProvider,
    Uncertainty,
    domino,
    prior_from_grasp,
    propose_grasps,
    result,
)
from roboskillhub.tasks.perception_regrasp.frames import (
    T_to_pose,
    interpolate_T,
    inv_T,
    make_T,
    matrix_to_quat_xyzw,
    pose_to_T,
    quat_xyzw_to_matrix,
    rot_x,
    rot_y,
    rot_z,
    rotation_angle,
)

SPEC = domino()


def _random_T(rng):
    q = rng.normal(size=4)
    return make_T(quat_xyzw_to_matrix(q / np.linalg.norm(q)), rng.uniform(-1, 1, 3))


def test_transforms_close_and_compose_in_the_T_A_B_direction():
    rng = np.random.default_rng(0)
    for _ in range(50):
        T_ab, T_bc = _random_T(rng), _random_T(rng)
        p_c = rng.uniform(-1, 1, 3)
        p_b = T_bc[:3, :3] @ p_c + T_bc[:3, 3]
        p_a = T_ab[:3, :3] @ p_b + T_ab[:3, 3]
        T_ac = T_ab @ T_bc
        assert np.allclose(T_ac[:3, :3] @ p_c + T_ac[:3, 3], p_a)
        assert np.allclose(inv_T(T_ab) @ T_ab, np.eye(4))
        back = pose_to_T(T_to_pose(T_ab, "world"), "world")
        assert np.allclose(back, T_ab, atol=1e-9)
    q = matrix_to_quat_xyzw(rot_z(math.pi / 2))
    assert np.allclose(q, [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)])


def test_pose_frame_is_checked():
    with pytest.raises(ValueError):
        pose_to_T(T_to_pose(np.eye(4), "camera"), "world")


def test_interpolation_follows_the_geodesic():
    Ta, Tb = make_T(np.eye(3), [0, 0, 0]), make_T(rot_x(math.radians(90)), [0.1, 0, 0])
    mid = interpolate_T(Ta, Tb, 0.5)
    assert math.isclose(rotation_angle(np.eye(3), mid[:3, :3]), math.radians(45), abs_tol=1e-9)
    assert np.allclose(mid[:3, 3], [0.05, 0, 0])


def _obs(**kw):
    base = dict(
        object_id="d1",
        object_class="domino",
        size_m=SPEC.size_m,
        frame_id="world",
        timestamp_s=100.0,
        provider="rgbd_pose6d",
        privileged=False,
        visibility="visible",
        T_frame_object=make_T(position=[0.3, 0, 0.004]),
        uncertainty=Uncertainty(0.001, 0.01, "fit_residual"),
    )
    base.update(kw)
    return ObjectObservation(**base)


def test_observation_invariants():
    with pytest.raises(ValueError, match="privileged"):
        _obs(provider="sim_truth", privileged=False)
    with pytest.raises(ValueError, match="no pose"):
        _obs(visibility="not_found")
    with pytest.raises(ValueError, match="needs a pose"):
        _obs(T_frame_object=None)
    _obs(visibility="occluded", T_frame_object=None, reason="hidden")


def test_usable_gives_contract_failure_types():
    assert _obs().check_usable(100.5, 1.0) == (True, "", None)
    assert _obs().check_usable(102.0, 1.0)[2] == "stale_observation"
    assert (
        _obs(visibility="not_found", T_frame_object=None).check_usable(100.0, 1.0)[2]
        == "object_not_observed"
    )
    priv = _obs(provider="sim_truth", privileged=True)
    assert priv.check_usable(100.0, 1.0)[2] == "perception.privileged_input"
    assert priv.check_usable(100.0, 1.0, allow_privileged=True)[0]
    amb = _obs(uncertainty=Uncertainty(0.001, 0.01, "fit_residual", ambiguous=True))
    assert amb.check_usable(100.0, 1.0)[2] == "perception.symmetry_ambiguous"


def test_observation_v1_round_trip_and_schema():
    T = make_T(rot_z(0.3) @ rot_y(math.pi / 2), [0.3, 0.1, 0.024])
    truth = {"timestamp_s": 5.0, "frame_id": "world", "objects": {"d1": {"T_frame_object": T}}}
    obs = SimTruthProvider(lambda: truth, {"d1": SPEC}, table_z=0.0).observe()[0]
    d = obs.to_v1()
    validate("Observation", d)
    assert d["privileged"] is True and d["source"] == "sim_truth"
    assert d["state_label"] == "standing"
    assert d["geometry"]["size_m"] == list(SPEC.size_m)
    back = ObjectObservation.from_v1(d)
    assert np.allclose(back.T_frame_object, T) and back.privileged
    unseen = _obs(visibility="not_found", T_frame_object=None, reason="no mask").to_v1()
    validate("Observation", unseen)
    assert unseen["position_m"] is None and unseen["quaternion_xyzw"] is None


def test_a_centroid_without_orientation_is_not_a_6d_pose():
    d = {
        "object_id": "d1",
        "frame_id": "world",
        "position_m": [0.3, 0, 0.01],
        "quaternion_xyzw": None,
        "timestamp_s": 1.0,
        "source": "rgbd_centroid",
        "privileged": False,
        "visibility": "visible",
    }
    o = ObjectObservation.from_v1(d, SPEC.size_m)
    assert not o.has_pose and o.visibility == "unknown"
    assert o.check_usable(1.0, 1.0)[2] == "object_not_observed"


def test_held_and_grasp_candidates_validate_as_v1():
    T = make_T(position=[0.3, 0, SPEC.size_m[2] / 2])
    c = propose_grasps("d1", T, SPEC, "world", 0.0)[0]
    validate("GraspCandidate", c.to_v1())
    h = prior_from_grasp(c, SPEC, "yam_left", 10.0).to_v1()
    validate("HeldObject", h)
    assert h["estimate"] == "grasp_geometry_prior" and h["T_TCP_object"]["frame_id"] == "tcp"


def test_contract_domino_frame_matches_the_layout_target_convention():
    """DominoLayoutTarget / Agent 2: a slot pose with identity rotation is a tile
    standing with its height along +z; module frame: long axis +x."""
    from roboskillhub.tasks.perception_regrasp import domino_state
    from roboskillhub.tasks.perception_regrasp.objects import from_contract_frame, to_contract_frame

    slot = make_T(position=[0.4, 0.0, 0.024])
    T = from_contract_frame(slot, "domino")
    assert domino_state(T, SPEC, table_z=0.0)["state"] == "standing"
    assert np.allclose(T[:3, 0], [0, 0, 1])  # long axis up
    assert np.allclose(to_contract_frame(T, "domino"), slot)
    truth = {"timestamp_s": 1.0, "frame_id": "world", "objects": {"d1": {"T_frame_object": T}}}
    v1 = SimTruthProvider(lambda: truth, {"d1": SPEC}).observe()[0].to_v1()
    assert np.allclose(v1["quaternion_xyzw"], [0, 0, 0, 1])  # identity on the wire


def test_result_semantics():
    with pytest.raises(ValueError):
        result("succeeded", "verify", failure_type="unreachable")
    with pytest.raises(ValueError):
        result("failed", "verify")
    with pytest.raises(ValueError):
        result("failed", "verify", failure_type="unreachable", recovery_hints=["pray"])
