"""planning-time-optimal-dual-arm-motions-with-curobo: CPU parts (robot build, timing, tracks, checks)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

P = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "planning-time-optimal-dual-arm-motions-with-curobo"
    / "scripts"
)
spec = importlib.util.spec_from_file_location("plan_dual_arm", P / "plan_dual_arm.py")
pda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pda)
M = pda.load_model()
WORK = [0.0, 0.6, 0.45, -0.12, 0.04, 0.0]


def test_dual_robot_checks_every_arm_arm_pair():
    k = pda.dual_robot_cfg(M, "x.urdf")["robot_cfg"]["kinematics"]
    assert len(k["cspace"]["joint_names"]) == 12 and k["tool_frames"] == ["left_tcp", "right_tcp"]
    for a, ign in k["self_collision_ignore"].items():
        assert all(i.split("_")[0] == a.split("_")[0] for i in ign if i != "right_base_link")
    assert "right_base_link" in pda.dual_urdf(M)


def test_retime_respects_velocity_and_acceleration():
    path = np.linspace(np.zeros(12), np.r_[[1.0] * 6, [-0.5] * 6], 20)
    t, q = pda.retime(path, vmax=0.25, amax=1.0, jmax=10.0)
    rows = pda.sample(t, q, 120)
    v = np.abs(np.diff(rows, axis=0)) * 120
    assert v.max() <= 0.25 * 1.05
    assert t[-1] >= 1.0 / 0.25  # the longest joint moves 1 rad at 0.25 rad/s


def test_tracks_are_stream_joints_chunks_in_hardware_angles():
    rows = np.tile(np.r_[WORK, WORK], (100, 1))
    ch = pda.to_tracks(rows)
    assert [len(c["left"]) for c in ch] == [64, 36]
    assert len(ch[0]["left"][0]) == 7 and math.isclose(ch[0]["right"][0][0], math.pi / 2)


def test_working_pose_free_and_home_flagged():
    assert pda.collisions(M, np.r_[WORK, WORK]) == []
    assert pda.inter_arm_distance(M, np.r_[WORK, WORK]) > 0
    assert any("link_2" in h and "link_5" in h for h in pda.collisions(M, np.zeros(12)))
