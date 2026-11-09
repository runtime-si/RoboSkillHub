# SPDX-License-Identifier: Apache-2.0
"""MuJoCo (CPU) physics runs of the domino scene: one nominal episode and three
physics negatives judged by the independent evaluator. Skipped without mujoco."""

import json

import pytest

mujoco = pytest.importorskip("mujoco")

from roboskillhub.tasks.assembly_eval.domino_mujoco import MujocoBackend  # noqa: E402
from roboskillhub.tasks.assembly_eval.domino_runner import run_episode  # noqa: E402
from roboskillhub.tasks.assembly_eval.domino_scene import DominoSceneSpec  # noqa: E402


@pytest.fixture(scope="module")
def spec():
    return DominoSceneSpec(n_tiles=3)


def test_nominal_episode_succeeds_and_logs(spec, tmp_path):
    rec = run_episode(MujocoBackend(), spec, 0, tmp_path)
    assert rec["status"] == "succeeded", rec["failure_types"]
    assert rec["placements_succeeded"] == 3
    assert rec["policy"]["privileged"] is True
    on_disk = json.loads((tmp_path / "record.json").read_text())
    for key in (
        "seed",
        "scene_config_sha256",
        "K_version",
        "G_version",
        "observation_mode",
        "goal",
        "trajectory_ref",
        "task_success",
        "failure_types",
        "recovery_count",
        "human_interventions",
        "sim_physics_steps",
        "planning_s",
        "wall_s",
    ):
        assert key in on_disk
    assert (tmp_path / "observations.jsonl").stat().st_size > 0


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("missed_grasp", "not_at_slot"),
        ("no_release", "release_not_confirmed"),
        ("retreat_sweeps_neighbour", "neighbor_disturbed"),
    ],
)
def test_physics_negatives_fail_although_executor_finished(spec, tmp_path, fault, expected):
    rec = run_episode(MujocoBackend(), spec, 1, tmp_path, fault=fault)
    assert rec["executor_claimed_all_finished"] is True
    assert rec["status"] == "failed"
    assert expected in rec["failure_types"]


def test_frozen_observer_is_unverified(spec, tmp_path):
    rec = run_episode(MujocoBackend(), spec, 2, tmp_path, fault="stale_observation")
    assert rec["status"] == "unverified"


def test_truth_adapter_frame_and_aliases(spec):
    import math

    from roboskillhub.tasks.assembly_eval.domino_geometry import quat_from_yaw, tilt_deg
    from roboskillhub.tasks.assembly_eval.domino_mujoco_truth import mujoco_truth
    from roboskillhub.tasks.assembly_eval.domino_runner import feeder_poses_for_seed

    b = MujocoBackend()
    b.reset(spec, feeder_poses_for_seed(spec, 0))
    for _ in range(20):
        b.step((0.0, 0.0, spec.transit_tcp_z(), 0.0), 0.04)
    tiles = {"tile_0": "tile_0"}
    world = mujoco_truth(b.m, b.d, tiles, geom_alias={"table": "bench"})
    assert world["contacts"]["tile_0"] == ["bench"]
    # table frame 1 m along +x and turned 90 deg in world
    shifted = mujoco_truth(
        b.m, b.d, tiles, T_world_table=((1.0, 0.0, 0.0), quat_from_yaw(math.pi / 2))
    )
    pw, pt = world["objects"]["tile_0"]["position"], shifted["objects"]["tile_0"]["position"]
    assert pt[0] == pytest.approx(pw[1], abs=1e-9)
    assert pt[1] == pytest.approx(-(pw[0] - 1.0), abs=1e-9)
    assert pt[2] == pytest.approx(pw[2], abs=1e-9)
    assert tilt_deg(shifted["objects"]["tile_0"]["quat_xyzw"]) == pytest.approx(0.0, abs=1e-3)
