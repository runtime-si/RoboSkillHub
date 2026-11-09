"""Held-object pose: prior vs estimate, and every invalidation path."""

from __future__ import annotations

import math

import numpy as np
from _perception_regrasp_helpers import SPEC, resting

from roboskillhub.runtime.task import offline_context, run_task_skill, validate
from roboskillhub.tasks.perception_regrasp import (
    SKILLS,
    ObjectObservation,
    Uncertainty,
    _hub,
    bind_domino_target,
    bind_lego_mating,
    check_held,
    estimate_from_observation,
    lego_brick,
    prior_from_grasp,
    propose_grasps,
    register_held_points,
)
from roboskillhub.tasks.perception_regrasp.frames import T_to_pose, make_T, rot_x, rot_z
from roboskillhub.tasks.perception_regrasp.objects import to_contract_frame

T0 = resting("lying", yaw=0.0)


def _held(t=10.0):
    c = next(c for c in propose_grasps("d1", T0, SPEC, "world", 0.0) if c.feasible)
    return c, prior_from_grasp(c, SPEC, "yam_left", t)


def _obs(T, t=10.0, oid="d1", cls="domino", vis="visible"):
    return ObjectObservation(oid, cls, SPEC.size_m, "world", t, "rgbd_pose6d", False, vis,
                             T if vis in ("visible", "partially_occluded") else None,
                             Uncertainty(0.0005, 0.005, "fit_residual"))  # fmt: skip


def test_prior_is_labelled_and_never_measured():
    c, h = _held()
    assert h.source == "geometric_prior" and h.valid
    v1 = h.to_v1()
    assert v1["estimate"] == "grasp_geometry_prior" and "not measured" in v1["quality"]["note"]
    assert np.allclose(c.T_frame_tcp @ h.T_tcp_object, T0)


def test_estimate_from_an_observation_while_holding():
    c, h = _held()
    T_tcp = c.T_frame_tcp.copy()
    T_tcp[2, 3] += 0.05  # lifted
    T_obj = T_tcp @ h.T_tcp_object @ make_T(position=[0.001, 0, 0])  # 1 mm off the prior
    new, ev = estimate_from_observation(h, _obs(T_obj, 11.0), T_tcp, 11.0, 11.2)
    assert new.valid and new.source == "estimated"
    assert math.isclose(ev["deviation_from_previous"]["translation_m"], 0.001, abs_tol=1e-6)
    validate("HeldObject", new.to_v1())
    assert new.to_v1()["estimate"] == "estimated_from_observation"


def test_symmetric_observation_does_not_read_as_a_slip():
    c, h = _held()
    T_obj = c.T_frame_tcp @ h.T_tcp_object @ make_T(rot_z(math.pi))  # 180 deg twin
    new, _ = estimate_from_observation(h, _obs(T_obj, 11.0), c.T_frame_tcp, 11.0, 11.0)
    assert new.valid and np.allclose(new.T_tcp_object, h.T_tcp_object, atol=1e-9)


def test_invalidations():
    c, h = _held()
    T = c.T_frame_tcp
    slipped = T @ h.T_tcp_object @ make_T(rot_x(math.radians(12)))
    new, _ = estimate_from_observation(h, _obs(slipped, 11.0), T, 11.0, 11.0)
    assert (
        not new.valid
        and new.invalid_kind == "slip"
        and new.to_v1()["invalidated_reason"] == "slip_detected"
    )
    new, _ = estimate_from_observation(h, _obs(None, 11.0, vis="occluded"), T, 11.0, 11.0)
    assert (
        not new.valid
        and new.invalid_kind == "occluded"
        and new.to_v1()["invalidated_reason"] == "occlusion"
    )
    new, _ = estimate_from_observation(h, _obs(T0, 11.0, oid="d7"), T, 11.0, 11.0)
    assert new.invalid_kind == "identity_changed"
    new, _ = estimate_from_observation(h, _obs(T0, 8.0), T, 8.0, 11.0)
    assert new.invalid_kind == "stale"
    new, _ = estimate_from_observation(h, _obs(T0, 11.0), T, 10.5, 11.0)
    assert new.invalid_kind == "unsynchronised"
    new, _ = check_held(h, 11.0, gripper_width_m=h.expected_width_m - 0.005)
    assert new.invalid_kind == "width_drop"
    new, _ = check_held(h, 11.0, gripper_width_m=h.expected_width_m + 0.01)
    assert new.invalid_kind == "width_open"
    ok, _ = check_held(h, 11.0, gripper_width_m=h.expected_width_m + 0.0005)
    assert ok.valid
    for bad in (new,):
        validate("HeldObject", bad.to_v1())
        assert "reobserve" in bad.quality["recovery"][0]


def test_register_held_points_with_the_prior_as_seed():
    """pose6d on the held part's (synthetic) depth points, seeded by the prior."""
    c, h = _held()
    syn = _hub.load("pose6d_synth")
    T_tcp = c.T_frame_tcp.copy()
    T_tcp[2, 3] += 0.08
    true_obj = T_tcp @ h.T_tcp_object @ make_T(rot_z(math.radians(3)), [0.002, 0, 0])
    Tc = syn.look_at(true_obj[:3, 3] + [0.0, -0.15, 0.1], true_obj[:3, 3])
    depth, mask = syn.render(
        {"box": list(SPEC.size_m)}, true_obj, Tc, K=syn.K_D405, noise=syn.D405, table_z=None
    )
    pts = syn.backproject(depth, mask, syn.K_D405) @ Tc[:3, :3].T + Tc[:3, 3]
    obs = register_held_points(h, pts, T_tcp, Tc[:3, 3], 12.0, "world")
    new, ev = estimate_from_observation(h, obs, T_tcp, 12.0, 12.0)
    assert new.valid  # 2 mm / 3 deg is under the slip tolerance
    assert math.isclose(ev["deviation_from_previous"]["translation_m"], 0.002, abs_tol=5e-4)


def test_bindings_for_domino_slot_and_lego_cell():
    c, h = _held()
    slot = resting("standing", yaw=0.3, xy=(0.4, 0.1))
    r = bind_domino_target(
        h, to_contract_frame(slot, "domino"), "world", T_frame_tcp_now=c.T_frame_tcp
    )
    assert r["status"] == "succeeded"
    targets = r["outputs"]["tcp_targets"]
    assert len(targets) == 4 and targets[0]["hand_turn_rad"] <= targets[-1]["hand_turn_rad"]
    from roboskillhub.tasks.perception_regrasp.frames import pose_to_T

    T_tcp = pose_to_T(targets[0]["tcp_pose"], "world")
    placed = T_tcp @ h.T_tcp_object
    assert np.allclose(placed[:3, 3], slot[:3, 3], atol=1e-9)
    b = lego_brick((2, 4))
    Tb = make_T(position=[0.3, 0, b.size_m[2] / 2])
    cb = next(x for x in propose_grasps("b1", Tb, b, "world", 0.0) if x.feasible)
    hb = prior_from_grasp(cb, b, "yam_left", 1.0)
    plate = make_T(rot_z(0.2), [0.35, -0.1, 0.0])
    rb = bind_lego_mating(
        hb, plate, (16, 16), {"x": 2, "y": 3, "layer": 1, "rotation_deg": 90}, "world"
    )
    assert rb["status"] == "succeeded" and len(rb["outputs"]["tcp_targets"]) == 2
    assert (
        bind_lego_mating(
            h, plate, (16, 16), {"x": 2, "y": 3, "layer": 1, "rotation_deg": 0}, "world"
        )["status"]
        == "rejected"
    )
    bad, _ = check_held(h, 1.0, gripper_width_m=0.0)
    r = bind_domino_target(bad, slot, "world")
    assert (
        r["status"] == "rejected" and r["failure_type"] == "held_object_slipped"
    )  # jaws closed past the part


def test_estimate_held_pose_tool_offline():
    tool = next(s for s in SKILLS if s.skill_id == "perception.estimate_held_pose")
    c, h = _held(t=1.0)
    T_obj = c.T_frame_tcp @ h.T_tcp_object @ make_T(position=[0, 0, -0.006])
    import time

    now = time.time()
    obs = _obs(T_obj, now).to_v1()
    req = {"held": {**h.to_v1(), "timestamp_s": now}, "observation": obs,
           "tcp_pose": T_to_pose(c.T_frame_tcp, "world"), "tcp_timestamp_s": now}  # fmt: skip
    r = run_task_skill(tool, req, offline_context())
    validate("SkillResult", r)
    assert r["status"] == "failed" and r["failure_type"] == "held_object_slipped"
    assert "regrasp" in r["recovery_hints"]
