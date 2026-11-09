# SPDX-License-Identifier: Apache-2.0
"""The YAM arm station through the real workcell runtime primitives (MuJoCo, CPU): honest
description, measured pad geometry, IK, pick/hold/release, refusal of unreachable wrist
orientations and of the tool-centre floor before any motion, move_joints checks.
Skipped without mujoco or without the i2rt YAM assets."""

import math

import pytest

pytest.importorskip("mujoco")

from roboskillhub.tasks.assembly_eval.yam_model import default_asset_xml  # noqa: E402

if not default_asset_xml().is_file():
    pytest.skip("i2rt YAM MJCF not found (set ROBOSKILLHUB_YAM_MJCF)", allow_module_level=True)

import numpy as np  # noqa: E402

from roboskillhub.runtime.engine import Runtime  # noqa: E402
from roboskillhub.tasks.assembly_eval.domino_station import tcp_quaternion  # noqa: E402
from roboskillhub.tasks.assembly_eval.yam_model import JOINTS, WORK_POSE  # noqa: E402
from roboskillhub.tasks.assembly_eval.yam_station import _mat_xyzw  # noqa: E402
from roboskillhub.tasks.assembly_eval.yam_world import yam_domino_world  # noqa: E402

REACHABLE_YAW = math.pi / 2  # jaws across the rack tiles' thickness, wrist inside +-2.09 rad
UNREACHABLE_YAW = -math.pi / 2  # the same grasp turned 180 deg: joint 6 would need ~2.25 rad


def _pose(x, y, z, yaw=REACHABLE_YAW):
    return {"frame_id": "world", "position_m": [x, y, z], "quaternion_xyzw": tcp_quaternion(yaw)}


@pytest.fixture
def world():
    w = yam_domino_world(0, slots=3)
    return w, Runtime(w["backend"], w["limits"])


def test_describe_is_honest(world):
    w, _ = world
    d = w["backend"].describe()
    assert "YAM" in d["robot"] and d["embodiment"]["arms"] == 1
    assert d["embodiment"]["not_modelled"]
    assert d["evidence_level"] == "dynamic_sim_privileged"
    caps = set(d["capabilities"])
    assert {"joint_motion", "joint_position_feedback", "object_pose_feedback"} <= caps
    assert not caps & {"linear_cartesian_motion", "contact_force_feedback", "compliant_motion"}
    j = d["joints"]
    assert j["names"] == list(JOINTS)
    assert j["lower_rad"][5] == pytest.approx(-2.0944) and j["upper_rad"][5] == pytest.approx(
        2.0944
    )
    assert d["limits"]["tcp_floor_over_table_m"] == pytest.approx(0.014)
    assert w["limits"].workspace_min_m[2] == pytest.approx(0.014)
    assert d["physics_backend"]["assets"]["licence_file_present"] is True


def test_pads_match_the_jaw_gap(world):
    w, _ = world
    ph = w["station"].physics
    m, d = ph.m, ph.d
    import mujoco

    for cmd in (0.002, 0.00425, 0.02):
        d.qpos[ph.fadr] = cmd
        mujoco.mj_forward(m, d)
        p, R = ph.tcp()
        faces = []
        for name in ("gripper_pad_l", "gripper_pad_r"):
            g = m.geom(name)
            c = R.T @ (d.geom_xpos[g.id] - p)
            faces.append(abs(c[1]) - m.geom_size[g.id][1])
            assert -0.0137 < c[2] < 0.0  # pads end ~10.2 mm past the TCP along the approach
        assert sum(faces) == pytest.approx(ph.gap(), abs=5e-4)


def test_ik_roundtrip(world):
    w, _ = world
    k = w["station"].kin
    q = np.array([0.1, 1.6, 1.2, -1.0, 0.1, 0.5])
    p, R = k.fk(q)
    q2, pe, re_, ok = k.solve(p, R, np.array(WORK_POSE))
    assert ok and pe < 1e-3 and re_ < 0.02


def test_pick_hold_place_release_through_runtime(world):
    w, rt = world
    st = w["station"]
    tile = w["inputs"]["feeder"]["pieces"][0]
    obs = rt.invoke("observe_objects", {"object_ids": [tile], "require_orientation": True})
    assert obs["status"] == "succeeded" and obs["evidence"]["privileged"] is True
    x, y, z = obs["evidence"]["observations"][0]["position_m"]
    assert rt.invoke("set_gripper", {"width_m": 0.0195, "purpose": "open"})["status"] == "succeeded"
    for zz, v in ((z + 0.05, 0.05), (z + 0.004, 0.03)):
        r = rt.invoke("move_pose", {"target_pose": _pose(x, y, zz), "speed_m_s": v})
        assert r["status"] == "succeeded", r.get("error")
    close = rt.invoke("set_gripper", {"width_m": 0.0055, "purpose": "close", "target_id": tile})
    assert close["status"] == "succeeded"
    state = st.get_state(timeout_s=1.0)
    assert state["gripper"]["holding"] is True
    assert state["joints"]["source"] and len(state["joints"]["positions_rad"]) == 6
    slot = w["inputs"]["layout"]["slots"][0]["pose"]["position_m"]
    held = {
        "object_id": tile,
        "T_TCP_object": {
            "frame_id": "tcp",
            "position_m": [0, 0, 0.004],
            "quaternion_xyzw": [1, 0, 0, 0],
        },
    }
    carry = rt.invoke(
        "move_waypoints",
        {
            "waypoints": [_pose(x, y, z + 0.06), _pose(slot[0], slot[1], slot[2] + 0.06)],
            "linear": False,
            "speed_m_s": 0.05,
            "held": held,
        },
    )
    assert carry["status"] == "succeeded", carry.get("error")
    down = rt.invoke(
        "move_pose", {"target_pose": _pose(slot[0], slot[1], slot[2] + 0.0055), "speed_m_s": 0.03}
    )
    assert down["status"] == "succeeded"
    rel = rt.invoke("set_gripper", {"width_m": 0.0195, "purpose": "release", "target_id": tile})
    assert rel["status"] == "succeeded"
    assert (
        rt.invoke("move_pose", {"target_pose": _pose(slot[0], slot[1], slot[2] + 0.06)})["status"]
        == "succeeded"
    )
    ind = w["independent"]()
    assert [p["status"] for p in ind["placements"]] == ["succeeded"]


def test_unreachable_wrist_orientation_is_refused_before_motion(world):
    w, rt = world
    st = w["station"]
    q0 = st.physics.joints()
    tile = st.physics.truth()["objects"]["tile_0"]["position"]
    r = rt.invoke(
        "move_pose",
        {
            "target_pose": _pose(tile[0], tile[1], tile[2] + 0.044, UNREACHABLE_YAW),
            "speed_m_s": 0.05,
        },
    )
    assert r["status"] == "failed" and r["failure_type"] == "motion_not_converged"
    reply = next(t for t in r["trace"] if t["event"] == "move_pose")["reply"]
    assert reply["failure_type"] == "unreachable" and reply["refused_before_motion"] is True
    assert "left_joint6" in reply["error"]
    assert np.abs(st.physics.joints() - q0).max() < 0.01
    # the jaw-flipped twin of the same grasp is reachable
    ok = rt.invoke(
        "move_pose", {"target_pose": _pose(tile[0], tile[1], tile[2] + 0.044), "speed_m_s": 0.05}
    )
    assert ok["status"] == "succeeded"


def test_tool_centre_floor(world):
    w, rt = world
    st = w["station"]
    r = rt.invoke("move_pose", {"target_pose": _pose(0.3, 0.05, 0.010)})
    assert r["status"] == "rejected" and r["failure_type"] == "unreachable"
    reply = st.move_pose(_pose(0.3, 0.05, 0.010), speed_m_s=0.03, timeout_s=5.0, linear=False)
    assert reply["ok"] is False and "floor" in reply["error"]


def test_move_joints_checks(world):
    w, rt = world
    st = w["station"]
    q = st.physics.joints()
    near = (q + np.array([0.05, -0.05, 0.05, 0.0, 0.05, 0.1])).tolist()
    ok = rt.invoke("move_joints", {"q_path": [near], "speed_rad_s": 0.4})
    assert ok["status"] == "succeeded", ok.get("error")
    assert ok["evidence"]["max_joint_error_rad"] <= 0.02
    far = (np.array(near) + np.array([0, 0, 0, 0, 0, 0.5])).tolist()
    assert rt.invoke("move_joints", {"q_path": [far]})["failure_type"] == "precondition_unmet"
    out = list(near)
    out[5] = 2.2
    assert rt.invoke("move_joints", {"q_path": [out]})["failure_type"] == "unreachable"
    # a joint target whose tool centre is under the floor: inside the joint ranges, refused by the station
    k = st.kin
    qd, *_rest, conv = k.solve(
        [0.3, 0.0, 0.008], _mat_xyzw(tcp_quaternion(REACHABLE_YAW)), np.array(near)
    )
    assert conv
    reply = st.move_joints([qd.tolist()], speed_rad_s=0.3, timeout_s=10.0)
    assert reply["ok"] is False and "floor" in reply["error"]


def test_stop_holds_measured_joints(world):
    w, _ = world
    st = w["station"]
    st.stop()
    assert np.allclose(st.q_cmd, st.physics.joints())
    s = st.get_state(timeout_s=1.0)
    assert s["timestamp_s"] > 1.7e9 and s["sim_time_s"] >= 0
