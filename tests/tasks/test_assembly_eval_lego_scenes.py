"""Scenes, faults and episode records of the LEGO evaluation (CPU checks; no simulator)."""

from __future__ import annotations

import json
import math

import pytest

from roboskillhub.tasks.assembly_eval import lego_faults
from roboskillhub.tasks.assembly_eval.lego_episode import (
    REQUIRED_KEYS,
    canonical_hash,
    load_config,
    missing_keys,
    summary_row,
)
from roboskillhub.tasks.assembly_eval.lego_scenes import (
    SCENES,
    scene_hash,
    user_config,
    write_task_dir,
)

CFG = load_config()
CONTRACT_FAILURES = {
    "lego.misaligned",
    "lego.not_seated",
    "lego.partially_seated",
    "lego.lifted_with_gripper",
    "lego.structure_damaged",
    "lego.insufficient_support",
}


@pytest.mark.parametrize("name", sorted(SCENES))
def test_scene_writes_a_roco_task_dir(tmp_path, name):
    d = write_task_dir(name, tmp_path / name)
    files = sorted(p.name for p in d.iterdir())
    if SCENES[name]["task_type"] == "1":
        assert files == ["structure_goal.json", "structure_start.json"]
    else:
        assert files == ["structure.json"]
    for f in d.iterdir():
        for brick in json.loads(f.read_text()).values():
            assert {"brick_id", "x", "y", "z", "ori"} <= brick.keys()


def test_user_config_uses_absolute_paths_and_no_cameras(tmp_path):
    ucfg = user_config(tmp_path / "task", tmp_path / "roco", "2")
    assert "Camera_Config" not in ucfg["Robot_Config"]
    assert ucfg["Robot_Config"]["Robot_URDF_Path"].startswith(str(tmp_path))
    assert len(ucfg["Robot_Config"]["Joint_Home_Position"]) == 23


def test_scene_hash_is_stable_and_distinct():
    hashes = {scene_hash(n) for n in SCENES}
    assert len(hashes) == len(SCENES)
    assert scene_hash("single_on_brick") == scene_hash("single_on_brick")


def test_every_fault_targets_a_scene_and_expects_contract_failures():
    for name, f in lego_faults.FAULTS.items():
        assert f["scenes"], name
        for scene, idx in f["scenes"].items():
            assert scene in SCENES
            n_to_place = len(SCENES[scene]["goal"]) - len(SCENES[scene].get("start", {}))
            assert 0 <= idx < n_to_place
        assert set(f["expect"]) <= CONTRACT_FAILURES, name


def test_fault_perturbations():
    p = lego_faults.perturbation
    assert p(None, CFG) == lego_faults.Perturbation()
    assert p("lateral_offset", CFG).place_offset_m == (0.003, 0.0, 0.0)
    assert p("angle_error", CFG).place_yaw_rad == pytest.approx(math.radians(9.0))
    assert p("partial_engagement", CFG).place_offset_m == (0.0, 0.008, 0.0)
    nf = p("not_fully_pressed", CFG)
    assert not nf.press and nf.hover_above_seat_m > 0
    assert p("insufficient_support", CFG, ("1", "2")).skip_parts == ("1", "2")
    assert not p("lifted_on_release", CFG).open_on_release
    d = p("damaging_build", CFG)
    assert (d.retreat_sweep_towards, d.retreat_sweep_distance_m) == ("1", 0.03)
    with pytest.raises(KeyError):
        p("teleport", CFG)


def test_faults_never_touch_the_clutch_model():
    for fault in lego_faults.FAULTS:
        assert (
            set(CFG["faults"][fault]) & {"required_force_n", "position_tolerance_m", "enabled"}
            == set()
        )


def test_config_hash_changes_with_any_threshold():
    h0 = canonical_hash(CFG)
    cfg = json.loads(json.dumps(CFG))
    cfg["postconditions"]["seated_xy_tol_m"] += 1e-4
    assert canonical_hash(cfg) != h0


def test_summary_row_and_required_keys():
    ep = {k: None for k in REQUIRED_KEYS}
    ep.update(
        episode_id="e",
        seed=3,
        scene="single_on_plate",
        fault="none",
        policy="goal_fsm",
        hashes={"eval_config": "a" * 64, "physics": "b" * 64},
        K={"lego.privileged_naive_fsm": "0.1.0"},
        G={"graph_id": "g", "version": "1"},
        outcome={
            "status": "failed",
            "failure_type": "lego.not_seated",
            "placements_succeeded": 0,
            "placements": 1,
        },
    )
    assert missing_keys(ep) == []
    row = summary_row(ep)
    assert row["placements"] == "0/1" and row["G"] == "g@1" and row["seed"] == 3
    del ep["seed"]
    assert missing_keys(ep) == ["seed"]
