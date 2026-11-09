# SPDX-License-Identifier: Apache-2.0
"""The domino physics station through the real workcell runtime primitives (MuJoCo, CPU):
top-down motion, contact-based holding, release, privileged observations with fresh
wall-clock stamps and sim time, refusal of non-top-down poses before motion (reported as
unreachable), injected faults. Skipped without mujoco."""

import math

import pytest

pytest.importorskip("mujoco")

from roboskillhub.runtime.engine import Runtime  # noqa: E402
from roboskillhub.tasks.assembly_eval.domino_geometry import quat_mul  # noqa: E402
from roboskillhub.tasks.assembly_eval.domino_skills import SKILLS  # noqa: E402
from roboskillhub.tasks.assembly_eval.domino_station import (  # noqa: E402
    tcp_quaternion,
    yaw_from_tcp,
)
from roboskillhub.tasks.assembly_eval.domino_world import physics_domino_world  # noqa: E402


def _pose(x, y, z, yaw=-math.pi / 2):
    return {"frame_id": "world", "position_m": [x, y, z], "quaternion_xyzw": tcp_quaternion(yaw)}


def _rt(fault=None):
    w = physics_domino_world(0, physics="mujoco", slots=3, fault=fault, fault_index=0)
    return w, Runtime(w["backend"], w["limits"])


def test_tcp_convention_roundtrip():
    for yaw in (-2.0, -math.pi / 2, 0.0, 1.0):
        y, off = yaw_from_tcp(tcp_quaternion(yaw))
        assert off == pytest.approx(0.0, abs=1e-9)
        assert math.cos(y - yaw) == pytest.approx(1.0)


def test_describe_is_honest():
    w, _ = _rt()
    d = w["backend"].describe()
    assert "NOT a YAM arm" in d["robot"]
    assert d["evidence_level"] == "dynamic_sim_privileged"
    assert "linear_cartesian_motion" not in d["capabilities"]
    assert "joint_motion" not in d["capabilities"]


def test_pick_hold_release_through_runtime():
    w, rt = _rt()
    tile = w["inputs"]["feeder"]["pieces"][0]
    obs = rt.invoke("observe_objects", {"object_ids": [tile, "table"], "require_orientation": True})
    assert obs["status"] == "succeeded"
    o = obs["evidence"]["observations"][0]
    assert o["privileged"] is True and o["quaternion_xyzw"] is not None
    x, y, z = o["position_m"]
    assert rt.invoke("set_gripper", {"width_m": 0.0195, "purpose": "open"})["status"] == "succeeded"
    assert (
        rt.invoke("move_pose", {"target_pose": _pose(x, y, z + 0.05), "speed_m_s": 0.05})["status"]
        == "succeeded"
    )
    assert (
        rt.invoke("move_pose", {"target_pose": _pose(x, y, z + 0.004), "speed_m_s": 0.03})["status"]
        == "succeeded"
    )
    close = rt.invoke("set_gripper", {"width_m": 0.0055, "purpose": "close", "target_id": tile})
    assert close["status"] == "succeeded", close.get("error")
    assert close["evidence"]["gripper_feedback"]["holding"] is True
    assert (
        rt.invoke("move_pose", {"target_pose": _pose(x, y, z + 0.06), "speed_m_s": 0.03})["status"]
        == "succeeded"
    )
    lifted = rt.invoke("observe_objects", {"object_ids": [tile]})["evidence"]["observations"][0]
    assert lifted["position_m"][2] > z + 0.04  # the tile came up with the hand: grasp by friction
    rel = rt.invoke("set_gripper", {"width_m": 0.032, "purpose": "release", "target_id": tile})
    assert rel["status"] == "succeeded"
    assert rel["evidence"]["gripper_feedback"]["holding"] is False


def test_close_on_air_is_grasp_missed():
    w, rt = _rt()
    assert (
        rt.invoke("move_pose", {"target_pose": _pose(0.5, 0.2, 0.05), "speed_m_s": 0.05})["status"]
        == "succeeded"
    )
    r = rt.invoke("set_gripper", {"width_m": 0.0055, "purpose": "close"})
    assert r["status"] == "failed" and r["failure_type"] == "grasp_missed"


def _tilted(x, y, z, deg):
    h = math.radians(deg) / 2
    q = quat_mul(tcp_quaternion(-math.pi / 2), (math.sin(h), 0.0, 0.0, math.cos(h)))
    return {"frame_id": "world", "position_m": [x, y, z], "quaternion_xyzw": list(q)}


def test_non_top_down_pose_is_refused_before_motion():
    w, _ = _rt()
    st = w["backend"]
    hand0 = list(st.hand_cmd)
    reply = st.move_pose(_tilted(0.4, 0.0, 0.1, 20), speed_m_s=0.05, timeout_s=5.0, linear=False)
    assert reply["ok"] is False and reply["refused_before_motion"] is True
    assert reply["failure_type"] == "unreachable"
    assert "top-down" in reply["error"] and "20.0 deg off vertical" in reply["error"]
    assert reply["sim_time_s"] == st.sim_t
    assert list(st.hand_cmd) == hand0 and not st.command_log
    assert [r["call"] for r in st.refusals] == ["move_pose"]
    assert st.refusals[0]["reason"] == reply["error"]
    frame = st.move_pose(
        {**_pose(0.4, 0.0, 0.1), "frame_id": "table"}, speed_m_s=0.05, timeout_s=5.0, linear=False
    )
    assert frame["refused_before_motion"] is True and frame["failure_type"] == "invalid_request"
    near = st.move_pose(_tilted(0.4, 0.0, 0.1, 2), speed_m_s=0.05, timeout_s=5.0, linear=False)
    assert near["ok"] is True and "refused_before_motion" not in near


def test_side_approach_is_unreachable_with_the_station_reason():
    w, rt = _rt()
    side = {"frame_id": "world", "position_m": [0.4, 0.0, 0.1], "quaternion_xyzw": [0, 0, 0, 1]}
    r = rt.invoke("move_pose", {"target_pose": side, "speed_m_s": 0.05})
    assert (r["status"], r["failure_type"]) == ("rejected", "unreachable")
    assert "top-down" in r["station_refusal"]["reason"]
    assert "station refused the target before moving" in r["error"]["message"]


def test_refusal_after_an_arrival_fails_unreachable():
    w, rt = _rt()
    waypoints = [_pose(0.4, 0.0, 0.12), _tilted(0.4, 0.0, 0.1, 30)]
    r = rt.invoke("move_waypoints", {"waypoints": waypoints, "linear": False, "speed_m_s": 0.05})
    assert (r["status"], r["failure_type"]) == ("failed", "unreachable")
    assert len(r["evidence"]["arrivals"]) == 1
    assert len(w["backend"].refusals) == 1


def test_replies_and_observations_carry_sim_time():
    w, rt = _rt()
    st = w["backend"]
    ack = st.move_pose(_pose(0.4, 0.0, 0.12), speed_m_s=0.05, timeout_s=5.0, linear=False)
    assert ack["ok"] is True and ack["sim_time_s"] == pytest.approx(st.sim_t) and st.sim_t > 0
    assert st.set_gripper(0.0195, timeout_s=5.0)["sim_time_s"] == pytest.approx(st.sim_t)
    tile = w["inputs"]["feeder"]["pieces"][0]
    o = rt.invoke("observe_objects", {"object_ids": [tile]})["evidence"]["observations"][0]
    assert o["sim_time_s"] >= ack["sim_time_s"] and o["timestamp_s"] > 1.7e9
    moved = rt.invoke("move_pose", {"target_pose": _pose(0.4, 0.0, 0.1), "speed_m_s": 0.05})
    assert moved["evidence"]["state_sim_time_s"] >= o["sim_time_s"]


def test_idle_time_advances_physics():
    w, rt = _rt()
    st = w["backend"]
    t0 = st.sim_t
    import time

    time.sleep(0.3)
    st.get_state(timeout_s=1.0)
    assert st.sim_t - t0 > 0.2


def test_stuck_jaws_release_failed():
    w, rt = _rt(fault="stuck_jaws")
    tile = w["inputs"]["feeder"]["pieces"][0]
    o = rt.invoke("observe_objects", {"object_ids": [tile]})["evidence"]["observations"][0]
    x, y, z = o["position_m"]
    rt.invoke("set_gripper", {"width_m": 0.0195, "purpose": "open"})
    rt.invoke("move_pose", {"target_pose": _pose(x, y, z + 0.05), "speed_m_s": 0.05})
    rt.invoke("move_pose", {"target_pose": _pose(x, y, z + 0.004), "speed_m_s": 0.03})
    rt.invoke("set_gripper", {"width_m": 0.0055, "purpose": "close"})
    r = rt.invoke("set_gripper", {"width_m": 0.032, "purpose": "release"})
    assert r["status"] == "failed" and r["failure_type"] == "release_failed"


def test_layout_check_tool_on_station_frames():
    w, rt = _rt()
    tool = SKILLS[0]
    frames = []
    import time

    for _ in range(8):
        time.sleep(0.15)
        st = w["backend"].get_state(timeout_s=1.0)
        frames.append({"timestamp_s": st["timestamp_s"], "objects": st["objects"]})
    slots = [s["pose"] for s in w["inputs"]["layout"]["slots"]]
    r = tool.run({"slots": slots, "frames": frames, "table_z": 0.0, "window_s": 0.9})
    assert (
        r["status"] == "failed" and r["failure_type"] == "eval.layout_not_standing"
    )  # nothing placed yet
    assert r["privileged_inputs"] is True
