"""Whole-joint-path feasibility, the reorient-or-regrasp planner and the skill.

The kinematic model is the i2rt YAM chain from checking-dual-arm-clearance with
an in-process DLS IK; results are model-only. The skill runs on a kinematic
``FakeKinematicStation`` through the real runtime primitives: contract logic, not
physics. Frames: the arm base raised 33 mm over the table, as on the station twin.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _perception_regrasp_helpers import SPEC, FakeKinematicStation, fake_limits

from roboskillhub.runtime.engine import Runtime
from roboskillhub.runtime.task import primitive_skill_context, run_task_skill, validate
from roboskillhub.tasks.perception_regrasp import (
    SKILLS,
    CallableIk,
    SimTruthProvider,
    Workcell,
    YamIk,
    evaluate_joint_path,
    plan_reorient_or_regrasp,
    plan_transfer,
    propose_grasps,
)
from roboskillhub.tasks.perception_regrasp import kinematics as K
from roboskillhub.tasks.perception_regrasp.frames import (
    T_to_pose,
    axis_angle,
    make_T,
    pose_to_T,
    rot_z,
    rotation_angle,
)
from roboskillhub.tasks.perception_regrasp.objects import (
    from_contract_frame,
    symmetry_rotations,
    to_contract_frame,
)
from roboskillhub.tasks.perception_regrasp.regrasp import _resting_pose

TABLE = -0.033  # table in the arm-base frame
DOWN = np.column_stack([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])  # TCP z down, jaws along base y


def lying(xy, yaw=0.0):
    return _resting_pose(SPEC, 2, 1.0, math.radians(yaw), xy, TABLE)


def standing(xy, yaw=0.0):
    return _resting_pose(SPEC, 0, 1.0, math.radians(yaw), xy, TABLE)


def observe(T, t=100.0, frame="base"):
    truth = {"timestamp_s": t, "frame_id": frame, "objects": {"d1": {"T_frame_object": T}}}
    return SimTruthProvider(lambda: truth, {"d1": SPEC}, table_z=TABLE).observe()[0]


def workcell(**kw):
    return Workcell("base", TABLE, np.eye(4), YamIk(), **kw)


def test_fk_ik_round_trip_and_tcp_convention():
    ik = YamIk()
    rng = np.random.default_rng(3)
    for _ in range(10):
        q = rng.uniform(K.JOINT_LIMITS[:, 0] + 0.2, K.JOINT_LIMITS[:, 1] - 0.2)
        T = K.fk(q)
        s = ik.solve(T, q + rng.normal(scale=0.05, size=6))
        assert s is not None
        Ts = K.fk(s)
        assert np.linalg.norm(Ts[:3, 3] - T[:3, 3]) < 1e-3 and rotation_angle(Ts, T) < math.radians(
            1
        )
    # At zero joints the hand points forward: TCP +z (approach) along base +x.
    assert np.allclose(K.fk(np.zeros(6))[:3, 2], [1, 0, 0], atol=1e-3)


def _pitch(x, psi_deg, z, a0, a1):
    ax = np.array([math.cos(math.radians(psi_deg)), math.sin(math.radians(psi_deg)), 0.0])
    base = rot_z(math.radians(psi_deg - 90)) @ DOWN
    return [make_T(axis_angle(ax, math.radians(a)) @ base, [x, 0.0, TABLE + z]) for a in (a0, a1)]


def test_pr106_reproduced_both_ends_reachable_but_the_pitch_jumps_a_wrist_joint():
    """PR #106: a 90 deg wrist pitch near the table jumped joint 4 by 1.27 rad in the
    twin. Same geometry here: each end has an IK solution, the path does not."""
    ik = YamIk()
    Ta, Tb = _pitch(0.38, 30, 0.06, 0, -90)
    assert any(ik.solve(Ta, s) is not None for s in K.default_seeds())
    assert any(ik.solve(Tb, s) is not None for s in K.default_seeds())
    rep = evaluate_joint_path([Ta, Tb], ik, labels=["top_down", "pitched_90"])
    assert not rep.feasible
    assert rep.reason.startswith("joint_jump") and rep.max_jump_joint == 4
    assert rep.max_jump_rad > 1.0


def test_a_split_pitch_is_continuous():
    Ta, Tb = _pitch(0.30, 0, 0.113, -45, 45)
    rep = evaluate_joint_path([Ta, Tb], YamIk())
    assert rep.feasible, rep.reason
    q = np.array(rep.q_path)
    assert np.max(np.abs(np.diff(q, axis=0))) <= K.JOINT_JUMP_RAD
    for qi, T in ((q[0], Ta), (q[-1], Tb)):
        assert np.linalg.norm(K.fk(qi)[:3, 3] - T[:3, 3]) < 1e-3


def test_a_station_ik_whose_solutions_miss_the_pose_is_not_trusted():
    """CallableIk checks a station solver's answers with the model's FK."""
    model = YamIk()
    Ta, Tb = _pitch(0.30, 0, 0.113, -30, 30)

    def off_by_a_wrist_turn(T, seed):
        q = model.solve(T, seed)
        if q is not None:
            q = q.copy()
            q[5] += 0.5
        return q

    ik = CallableIk(off_by_a_wrist_turn)
    rep = evaluate_joint_path([Ta, Tb], ik)
    assert not rep.feasible and rep.reason.startswith("unreachable") and ik.mismatches > 0


class ScriptedIk:
    """Returns joint vectors in order, ignoring the pose: tests the path logic alone."""

    def __init__(self, qs):
        self.qs = list(qs)

    def solve(self, T, seed):
        return np.asarray(self.qs.pop(0)) if self.qs else None


def test_jump_and_limit_margin_logic():
    T = [make_T(position=[0.3, 0, 0.1]), make_T(position=[0.3, 0, 0.1 + 0.01 * 3])]  # 4 samples
    q0 = np.array([0.0, 1.2, 1.0, -0.6, 0.0, 0.0])
    smooth = [q0 + [0, 0, 0, 0, 0.1 * k, 0] for k in range(4)]
    rep = evaluate_joint_path(T, ScriptedIk(smooth), seeds=[q0])
    assert rep.feasible and math.isclose(rep.max_jump_rad, 0.1, abs_tol=1e-9)
    jumpy = smooth[:2] + [smooth[2] + [0, 0, 0, 0, 0, 0.9], smooth[3]]
    rep = evaluate_joint_path(T, ScriptedIk(jumpy), seeds=[q0])
    assert not rep.feasible and rep.max_jump_joint == 6 and rep.failed_at == 2
    near = [
        q0 + [0, 0, 0, 2.1, 0, 0] for _ in range(4)
    ]  # joint 4 at -0.6 + 2.1 = 1.5, 0.07 from its limit
    rep = evaluate_joint_path(T, ScriptedIk(near), seeds=[q0], min_limit_margin_rad=0.1)
    assert (
        not rep.feasible
        and rep.reason.startswith("limit_margin")
        and rep.min_limit_margin_joint == 4
    )


def _independent_checks(transfer, wc):
    """Recompute from the returned joint path, not from the planner's verdict."""
    q = np.array(transfer["joint_path_q"])
    assert np.max(np.abs(np.diff(q, axis=0))) <= K.JOINT_JUMP_RAD + 1e-9
    assert np.all(q >= K.JOINT_LIMITS[:, 0] - 1e-9) and np.all(q <= K.JOINT_LIMITS[:, 1] + 1e-9)
    grasp = transfer["grasp"]
    T_grasp = pose_to_T(grasp["tcp_pose"], wc.frame_id)
    # q_path[0] is the pregrasp; its FK is the grasp pose backed off 40 mm along the approach.
    pre = K.fk(q[0])
    assert np.linalg.norm(pre[:3, 3] - (T_grasp[:3, 3] - 0.04 * T_grasp[:3, 2])) < 1e-3


@pytest.fixture(scope="module")
def direct_plan():
    obs = observe(lying((0.30, 0.0)))
    wc = workcell()
    goal = standing((0.30, 0.08))
    return (
        obs,
        wc,
        goal,
        plan_reorient_or_regrasp(obs, SPEC, goal, wc, now_s=100.1, allow_privileged=True),
    )


def test_direct_wrist_turn_lying_to_standing(direct_plan):
    obs, wc, goal, r = direct_plan
    assert r["status"] == "succeeded", r["reason"]
    out = r["outputs"]
    assert out["plan_type"] == "direct_wrist_turn" and r["verification"] == "kinematic_model"
    (tr,) = out["transfers"]
    assert tr["to_state"] == "standing"
    assert tr["pivot_arm_m"] <= wc.max_pivot_arm_m
    steps = out["steps"]
    assert [s["capability"] for s in steps] == [
        "grasp",
        "move_waypoints",
        "set_gripper",
        "move_pose",
        "observe_objects",
    ]
    _independent_checks({**tr, "joint_path_q": steps[0]["q_path"]}, wc)
    # The release puts the part on the goal (any symmetric twin), 1.5 mm over its resting pose.
    T_place_tcp = pose_to_T(steps[1]["request"]["waypoints"][-1], "base")
    T_tcp_obj = from_contract_frame(
        pose_to_T(steps[1]["request"]["held"]["T_TCP_object"], "tcp"), "domino"
    )
    placed = T_place_tcp @ T_tcp_obj
    assert np.linalg.norm(placed[:3, 3] - (goal[:3, 3] + [0, 0, 0.0015])) < 1e-6
    assert min(rotation_angle(placed, goal[:3, :3] @ S) for S in symmetry_rotations(SPEC)) < 1e-6
    for h in out["held_priors"]:
        validate("HeldObject", h)
        assert h["estimate"] == "privileged_sim" or h["estimate"] == "grasp_geometry_prior"


def test_direct_infeasible_but_staging_regrasp_feasible():
    """The worked example the task asks for (kinematic model only)."""
    obs = observe(lying((0.34, -0.08)))
    goal = standing((0.30, 0.10))
    wc = workcell()
    direct = plan_reorient_or_regrasp(
        obs, SPEC, goal, wc, now_s=100.1, allow_privileged=True, allow_staging=False
    )
    assert direct["status"] == "failed" and direct["failure_type"] == "unreachable"
    assert direct["evidence"]["rejections"]  # why each candidate failed
    r = plan_reorient_or_regrasp(
        obs, SPEC, goal, wc, now_s=100.1, allow_privileged=True, max_path_evals=150
    )
    assert r["status"] == "succeeded", r["reason"]
    out = r["outputs"]
    assert out["plan_type"] == "staging_regrasp" and len(out["transfers"]) == 2
    t1, t2 = out["transfers"]
    assert t1["to"] == t2["from"] and t2["to_state"] == "standing"
    assert out["direct_rejected"]["evals"] > 0
    caps = [s["capability"] for s in out["steps"]]
    assert caps.count("grasp") == 2 and caps.count("observe_objects") == 2
    for s in out["steps"]:
        if s["capability"] == "grasp":
            tr = (t1, t2)[s["transfer"]]
            _independent_checks({**tr, "joint_path_q": s["q_path"]}, wc)


def test_pivot_risk_rejects_an_off_centre_pinch():
    """PR #106 pinched 14 mm off centre and the tiles turned out of the hand."""
    T0 = lying((0.30, 0.0))
    off = [
        c
        for c in propose_grasps("d1", T0, SPEC, "base", TABLE, tilts_deg=(30.0,))
        if c.feasible and abs(c.offset_m) > 0.005
    ]
    assert off
    reasons = {
        plan_transfer(T0, standing((0.30, 0.08)), c, SPEC, workcell(), None)[1].split(":")[0]
        for c in off
    }
    assert "pivot_risk" in reasons


@pytest.mark.parametrize(
    "case,ftype",
    [
        ("stale", "stale_observation"),
        ("privileged", "perception.privileged_input"),
        ("leaning", "perception.uncertain_state"),
        ("frame", "invalid_request"),
    ],
)
def test_planner_rejects_before_any_search(case, ftype):
    T = lying((0.30, 0.0))
    allow = case != "privileged"
    now = 100.1
    if case == "stale":
        now = 105.0
    if case == "leaning":
        T = make_T(T[:3, :3] @ axis_angle([0, 1, 0], math.radians(25)), T[:3, 3] + [0, 0, 0.01])
    obs = observe(T, frame="world" if case == "frame" else "base")
    r = plan_reorient_or_regrasp(
        obs, SPEC, standing((0.3, 0.08)), workcell(), now_s=now, allow_privileged=allow
    )
    assert r["status"] == "rejected" and r["failure_type"] == ftype
    assert r["evidence"].get("path_evals") is None


# ----------------------------------------------------------------- the skill
SKILL = next(s for s in SKILLS if s.skill_id == "perception.reorient_or_regrasp")
T_W_BASE = make_T(position=[0.0, 0.0, -TABLE])  # world: table at z = 0


def _request(goal_world):
    return {
        "object_id": "d1",
        "class": "domino",
        "geometry": {"type": "box", "size_m": list(SPEC.size_m)},
        "goal_pose": T_to_pose(to_contract_frame(goal_world, "domino"), "world"),
        "table_z": 0.0,
        "T_W_base": T_to_pose(T_W_BASE, "world"),
        "allow_staging": False,
        "max_path_evals": 60,
    }


def _run_skill(station, goal_world):
    rt = Runtime(station, fake_limits())
    with rt.task_lease("ep-perception", 120) as lease:
        ctx = primitive_skill_context(rt, lease, request_id="ep-perception/n0")
        return run_task_skill(SKILL, _request(goal_world), ctx)


def test_skill_executes_through_primitives_and_verifies_by_observation():
    start = T_W_BASE @ lying((0.30, 0.0))
    goal = T_W_BASE @ standing((0.30, 0.08))
    station = FakeKinematicStation(start)
    r = _run_skill(station, goal)
    validate("SkillResult", r)
    assert r["status"] == "succeeded", (r.get("failure_type"), r["evidence"].get("reason"))
    post = r["evidence"]["postcondition"]
    assert post["passed"] and post["measured"]["state_label"] == "standing"
    assert r["observations"] and not r["privileged_inputs"]
    assert [s["capability"] for s in r["evidence"]["steps"]] == [
        "grasp",
        "move_waypoints",
        "set_gripper",
        "move_pose",
    ]
    assert len(r["children"]) >= 6  # observe, 4 motion/gripper primitives, observe
    final = r["outputs"]["final_observation"]
    assert final["source"] == "fake_tracker" and final["privileged"] is False
    assert np.linalg.norm(np.array(final["position_m"]) - (goal[:3, 3] + [0, 0, 0.0015])) < 1e-3


def test_skill_is_gated_on_orientation_feedback():
    start = T_W_BASE @ lying((0.30, 0.0))
    station = FakeKinematicStation(
        start,
        capabilities=["pose_feedback", "cartesian_motion", "gripper", "stop", "object_feedback"],
    )
    r = _run_skill(station, T_W_BASE @ standing((0.30, 0.08)))
    assert r["status"] == "rejected" and r["failure_type"] == "capability_missing"
    assert station.calls == []


def test_skill_reports_a_slip_from_the_held_check():
    start = T_W_BASE @ lying((0.30, 0.0))
    station = FakeKinematicStation(start, slip_after_grasp=True)
    r = _run_skill(station, T_W_BASE @ standing((0.30, 0.08)))
    assert r["status"] == "failed"
    assert r["failure_type"] in ("held_object_slipped", "postcondition_failed")
    assert r["phase"] in ("grasp", "transport")
