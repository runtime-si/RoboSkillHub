from __future__ import annotations

import copy
import json

import pytest
from conftest import REPO

from roboskillhub.catalog import load_catalog
from roboskillhub.cli import main
from roboskillhub.vla_index import (
    CHECKPOINTS_PATH,
    DATASETS_PATH,
    PAGE_PATH,
    REJECTED_PATH,
    find_vla,
    load_vla_index,
    merge_entries,
    merge_files,
    our_servers,
    profile_gaps,
    render_page,
    validate_checkpoints,
    validate_datasets,
)


def checkpoint(ident="org/yam-model", **changes):
    row = {
        "id": ident,
        "revision": "a" * 40,
        "family": "pi05",
        "base_model": "lerobot/pi05_base",
        "format": "LeRobot PI05Policy",
        "embodiment": "yam_i2rt",
        "arms": 2,
        "action": {"space": "abs_joint", "dim": 14, "horizon": 30, "hz": 30},
        "state": {
            "dim": 14,
            "order": ["left.j1..j6", "left.gripper", "right.j1..j6", "right.gripper"],
        },
        "cameras": ["top", "left_wrist", "right_wrist"],
        "training": {"datasets": ["org/yam-data"], "episodes": 100, "tasks": ["pick block"]},
        "license": {"model": "Apache-2.0", "inherited": "Gemma", "verified": True},
        "gated": False,
        "size_gb": 9.3,
        "servable_by": ["policies/gr00t:pi05"],
        "yam_compatible": "as_is",
        "task_relevance": "bimanual pick and place",
        "notes": "",
        "source_urls": [f"https://huggingface.co/{ident}"],
        "checked_at": "2026-11-11",
    }
    row.update(changes)
    return row


def dataset(ident="org/yam-data", **changes):
    row = {
        "id": ident,
        "revision": "b" * 40,
        "embodiment": "yam_i2rt",
        "episodes": 100,
        "tasks": ["pick block"],
        "format": "LeRobot",
        "license": {"model": "Apache-2.0", "verified": True},
        "size_gb": None,
        "relevance": "training data",
        "source_urls": [f"https://huggingface.co/datasets/{ident}"],
    }
    row.update(changes)
    return row


def document(collection, entries):
    return {
        "schema_version": 1,
        "collection": collection,
        "scope": "test",
        "snapshot_date": "2026-11-11",
        "entries": entries,
    }


@pytest.fixture
def vla_root(tmp_path):
    aloha = checkpoint(
        "org/aloha-model",
        embodiment="aloha",
        yam_compatible="finetune_base",
        servable_by=["openpi"],
        task_relevance="bimanual towel folding",
    )
    adapter = checkpoint("org/yam-adapter", yam_compatible="adapter", servable_by=["openpi"])
    (tmp_path / "candidates").mkdir()
    (tmp_path / CHECKPOINTS_PATH).write_text(
        json.dumps(document("vla-checkpoints", [aloha, adapter, checkpoint()]))
    )
    (tmp_path / DATASETS_PATH).write_text(json.dumps(document("vla-datasets", [dataset()])))
    return tmp_path


def test_checked_in_catalog_is_valid_profiled_current_and_not_in_the_registry(capsys):
    checkpoints, datasets = load_vla_index(REPO)
    assert len(checkpoints["entries"]) >= 7 and datasets["entries"]
    assert main(["--root", str(REPO), "check-sources"]) == 0
    assert "vla-checkpoints" in capsys.readouterr().out
    assert profile_gaps(REPO, checkpoints["entries"]) == []
    assert (REPO / PAGE_PATH).read_text(encoding="utf-8") == render_page(checkpoints, datasets)
    ids = {row["id"] for row in checkpoints["entries"]}
    assert "allenai/MolmoAct2-BimanualYAM" in ids
    assert not ids & set(load_catalog(REPO).bundles)


def test_seeded_twin_evaluations_are_recorded():
    rows = {row["id"]: row for row in load_vla_index(REPO)[0]["entries"]}
    for ident in (
        "allenai/MolmoAct2-BimanualYAM",
        "helen9975/pi05-molmoact-yam",
        "robocurve/gr00t-n1.7-yam-molmoact2",
    ):
        evaluations = rows[ident]["evaluations"]
        assert evaluations and all(e["successes"] == 0 and e["episodes"] == 10 for e in evaluations)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"revision": "main"}, "revision must be a full 40-character commit"),
        ({"revision": "8dcbed66"}, "revision must be a full 40-character commit"),
        ({"license": {"model": "MIT"}}, "license.verified is required"),
        (
            {"license": {"model": "MIT", "verified": "yes"}},
            "license.verified must be true or false",
        ),
        ({"yam_compatible": "maybe"}, "yam_compatible must be one of"),
        ({"id": "not a repo"}, "id must be a Hugging Face repo id"),
        ({"source_urls": []}, "source_urls must be a nonempty list"),
        ({"gated": "auto"}, "gated must be true or false"),
        ({"action": {"dim": 14}}, "action.space must be a nonempty string"),
        (
            {
                "evaluations": [
                    {
                        "setting": "twin",
                        "task": "t",
                        "date": "d",
                        "evidence": "e",
                        "episodes": 10,
                        "successes": 11,
                    }
                ]
            },
            "successes must be an integer in 0..episodes",
        ),
    ],
)
def test_checkpoint_validator_rejects(change, message):
    errors = validate_checkpoints(document("vla-checkpoints", [checkpoint(**change)]))
    assert any(message in e for e in errors), errors


@pytest.mark.parametrize(
    "field", ["revision", "license", "yam_compatible", "servable_by", "checked_at"]
)
def test_every_schema_field_is_required(field):
    row = checkpoint()
    del row[field]
    errors = validate_checkpoints(document("vla-checkpoints", [row]))
    assert f"entries[0]: missing {field}" in errors


def test_duplicate_ids_and_dataset_rules():
    assert any(
        "duplicate id" in e
        for e in validate_checkpoints(
            document("vla-checkpoints", [checkpoint("Org/M"), checkpoint("org/m")])
        )
    )
    assert validate_datasets(document("vla-datasets", [dataset(license="Apache-2.0")])) == []
    assert any(
        "revision" in e
        for e in validate_datasets(document("vla-datasets", [dataset(revision=None)]))
    )
    assert validate_checkpoints([]) == ["vla-checkpoints must be an object"]


def test_ranking_prefers_matches_then_compatibility_then_our_servers(vla_root):
    rows = load_vla_index(vla_root)[0]["entries"]
    ranked = [row["id"] for row in find_vla(rows, "bimanual yam pick")]
    assert ranked == ["org/yam-model", "org/yam-adapter", "org/aloha-model"]
    assert [row["id"] for row in find_vla(rows, "towel")] == ["org/aloha-model"]
    assert find_vla(rows, "nothing-matches") == []
    yam = find_vla(rows, embodiment="yam_i2rt", compatible="as_is")
    assert [row["id"] for row in yam] == ["org/yam-model"]
    other = checkpoint("org/other", embodiment="i2rt_yam_bimanual")
    assert find_vla([other], embodiment="yam_i2rt") == [other]
    assert find_vla([checkpoint(embodiment="yamaha_arm")], embodiment="yam_i2rt") == []


def test_cli_sources_kind_vla_and_vla_search(vla_root, capsys):
    assert (
        main(["--root", str(vla_root), "sources", "bimanual yam pick", "--kind", "vla", "--json"])
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["collection"] == "vla-checkpoints" and out["invocable_in_hub"] is False
    assert out["entries"][0]["id"] == "org/yam-model"
    args = [
        "--root",
        str(vla_root),
        "vla",
        "search",
        "--embodiment",
        "yam_i2rt",
        "--compatible",
        "as_is",
    ]
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "org/yam-model" in text and "org/aloha-model" not in text
    assert "policies/gr00t:pi05" in text
    assert main(["--root", str(vla_root), "vla", "datasets", "pick", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["entries"][0]["id"] == "org/yam-data"
    assert main(["--root", str(vla_root), "vla", "search", "zzz"]) == 1


def test_page_write_and_check(vla_root, capsys):
    assert main(["--root", str(vla_root), "vla", "page", "--check"]) == 1
    assert main(["--root", str(vla_root), "vla", "page", "--write"]) == 0
    assert main(["--root", str(vla_root), "vla", "page", "--check"]) == 0
    page = (vla_root / PAGE_PATH).read_text()
    assert "as_is 1, adapter 1, finetune_base 1" in page
    assert "| org/yam-model | aaaaaaaa |" in page


def test_merge_keeps_the_most_complete_record_and_its_evaluations():
    evaluation = {
        "setting": "twin",
        "task": "t",
        "date": "2026-11-10",
        "evidence": "e",
        "episodes": 10,
        "successes": 0,
    }
    sparse = checkpoint(notes="", evaluations=[evaluation], source_urls=["https://a.example/x"])
    sparse["training"] = {}
    full = checkpoint("ORG/yam-model", notes="more detail")
    merged = merge_entries([sparse], [full, checkpoint("org/second")])
    assert [row["id"] for row in merged] == ["ORG/yam-model", "org/second"]
    assert merged[0]["notes"] == "more detail"
    assert merged[0]["evaluations"] == [evaluation]
    assert "https://a.example/x" in merged[0]["source_urls"]


def test_merge_files_validates_rejects_and_regenerates(vla_root):
    survey = vla_root / "yam.json"
    survey.write_text(
        json.dumps(
            [
                checkpoint("org/new-model", gated="manual", size_gb="3.5"),
                checkpoint("org/unpinned", revision="main"),
                copy.deepcopy(checkpoint()),
            ]
        )
    )
    datasets = vla_root / "datasets.json"
    datasets.write_text(json.dumps({"entries": [dataset("org/more-data")]}))
    report = merge_files(vla_root, [survey, datasets, vla_root / "absent.json"], write=True)
    assert any("absent.json: missing" in line for line in report)
    checkpoints, data = load_vla_index(vla_root)
    ids = [row["id"] for row in checkpoints["entries"]]
    assert ids == ["org/aloha-model", "org/yam-adapter", "org/yam-model", "org/new-model"]
    new = checkpoints["entries"][-1]
    assert new["gated"] is True and new["size_gb"] == 3.5
    assert [row["id"] for row in data["entries"]] == ["org/yam-data", "org/more-data"]
    rejected = json.loads((vla_root / REJECTED_PATH).read_text())
    assert [item["id"] for item in rejected] == ["org/unpinned"]
    assert "org/new-model" in (vla_root / PAGE_PATH).read_text()


def test_profile_gaps_name_the_policy_bundle(vla_root):
    rows = load_vla_index(vla_root)[0]["entries"]
    assert profile_gaps(vla_root, rows) == [("org/yam-model", ["policies/gr00t/profiles"])]
    profiles = vla_root / "policies/gr00t/profiles"
    profiles.mkdir(parents=True)
    (profiles / "p.yaml").write_text("checkpoint: {repo: org/yam-model}\n")
    assert profile_gaps(vla_root, rows) == []
    molmo = checkpoint("org/m2", servable_by=["policies/molmoact2-yam (official server)"])
    assert profile_gaps(vla_root, [molmo]) == [("org/m2", ["policies/molmoact2-yam/profiles"])]
    loose = checkpoint("org/p", servable_by=["policies/gr00t (pi0.5 backend)", "openpi"])
    assert our_servers(loose) == ["policies/gr00t:pi05"]
    assert our_servers(checkpoint(servable_by=["openpi", "lerobot"])) == []


def test_pick_place_sources_stay_the_default(capsys):
    assert main(["--root", str(REPO), "sources", "--limit", "1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["collection"] == "pick-place"
