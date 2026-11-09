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
