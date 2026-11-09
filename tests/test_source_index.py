from __future__ import annotations

import copy
import json

import pytest
from conftest import REPO

from roboskillhub.catalog import load_catalog
from roboskillhub.cli import main
from roboskillhub.source_index import (
    INDEX_PATH,
    find_sources,
    load_source_index,
    validate_source_index,
)


@pytest.fixture
def index_data():
    row = {
        "id": "grasp-generator",
        "name": "Grasp Generator",
        "repository_url": "https://github.com/example/grasp",
        "source_revision": "a" * 40,
        "source_url": "https://github.com/example/grasp/tree/" + "a" * 40,
        "category": "grasp-generation",
        "record_type": "implementation",
        "capabilities": ["抓取候选", "diffusion grasp generation"],
        "summary": "Propose parallel-jaw grasps from a point cloud.",
        "license": {
            "spdx": "MIT",
            "status": "permissive",
            "url": "https://github.com/example/grasp/blob/" + "a" * 40 + "/LICENSE",
            "notes": "Code only; checkpoint terms not reviewed.",
        },
        "review_level": "metadata-indexed",
        "invocable_in_hub": False,
        "hub_destination": "tool",
        "entrypoints": [],
        "adaptation_required": ["Verify gripper frame and model weights."],
        "priority": "P1",
        "dedup_key": "example/grasp",
        "evidence": [{"url": "https://github.com/example/grasp", "kind": "readme", "sha256": None}],
    }
    research = copy.deepcopy(row)
    research.update(
        id="restricted-grasp",
        name="Research Grasp",
        dedup_key="example/research-grasp",
        review_level="implementation-inspected",
        entrypoints=["infer.py:predict"],
        priority="P3",
    )
    research["license"].update(status="research-only", spdx="CC-BY-NC-4.0")
    research["evidence"].append(
        {
            "url": "https://github.com/example/grasp/blob/main/infer.py",
            "kind": "source",
            "sha256": "b" * 64,
        }
    )
    unknown = copy.deepcopy(row)
    unknown.update(id="placement-generator", name="Placement Generator", dedup_key="example/place")
    unknown.update(
        category="placement", capabilities=["放置候选"], summary="Propose placement poses."
    )
    unknown["license"].update(status="unresolved", spdx=None, url=None)
    return {
        "schema_version": 1,
        "collection": "pick-place",
        "snapshot_date": "2026-11-08",
        "scope": "Upstream candidates, not runnable bundles.",
        "counting_policy": "One distinct implementation per deduplication key.",
        "entries": [row, research, unknown],
    }


def write_index(root, data):
    path = root / INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_separate_source_states_and_unresolved_license_are_preserved(tmp_path, index_data):
    write_index(tmp_path, index_data)
    loaded = load_source_index(tmp_path)
    assert loaded == index_data
    assert len(find_sources(loaded)) == 3  # Restricted and unresolved are visible by default.
    assert [r["id"] for r in find_sources(loaded, license_status="unresolved")] == [
        "placement-generator"
    ]
    assert [r["id"] for r in find_sources(loaded, review_level="implementation-inspected")] == [
        "restricted-grasp"
    ]


def test_queries_require_all_terms_and_filters_are_exact(index_data):
    assert len(find_sources(index_data, "DIFFUSION 抓取", category="grasp-generation")) == 2
    assert not find_sources(index_data, "grasp missing-term")
    assert not find_sources(index_data, "grasp", category="grasp")
    assert len(find_sources(index_data, "抓取", license_status="permissive")) == 1


@pytest.mark.parametrize("field", ["id", "dedup_key"])
def test_duplicate_identity_is_rejected(index_data, field):
    index_data["entries"][1][field] = index_data["entries"][0][field]
    assert any(f"duplicate {field}" in error for error in validate_source_index(index_data))


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", []),
        ("dedup_key", {}),
        ("name", ""),
        ("category", None),
        ("summary", 123),
        ("repository_url", "file:///tmp/repo"),
        ("source_url", "https://[broken"),
        ("source_revision", "main"),
        ("source_revision", None),
        ("record_type", []),
        ("review_level", "real-verified"),
        ("invocable_in_hub", True),
        ("invocable_in_hub", 0),
        ("hub_destination", {}),
        ("entrypoints", [""]),
        ("capabilities", []),
        ("adaptation_required", "none"),
        ("priority", ["P1"]),
        ("readiness", "ready"),
        ("maturity", "real-verified"),
        ("executable", True),
        ("license", None),
        ("evidence", []),
    ],
)
def test_malformed_or_executable_claims_return_errors(index_data, field, value):
    index_data["entries"][0][field] = value
    assert validate_source_index(index_data)


@pytest.mark.parametrize(
    "field,value",
    [("status", []), ("url", "https://"), ("spdx", []), ("notes", "  "), ("spdx", "NOASSERTION")],
)
def test_license_records_do_not_turn_unknown_into_permission(index_data, field, value):
    index_data["entries"][0]["license"][field] = value
    assert validate_source_index(index_data)


def test_metadata_readme_does_not_suffice_for_implementation_inspection(index_data):
    row = index_data["entries"][0]
    row["review_level"] = "implementation-inspected"
    assert any("source evidence" in error for error in validate_source_index(index_data))
    row["entrypoints"] = ["inference.py"]
    assert any("source evidence" in error for error in validate_source_index(index_data))


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "javascript:run()"),
        ("sha256", "xyz"),
        ("sha256", []),
        ("kind", {}),
        ("kind", "claimed"),
    ],
)
def test_malformed_evidence_is_rejected(index_data, field, value):
    index_data["entries"][0]["evidence"][0][field] = value
    assert validate_source_index(index_data)


@pytest.mark.parametrize("data", [None, [], {}, {"schema_version": 1, "entries": [None]}])
def test_malformed_index_returns_validation_errors_without_raising(data):
    assert validate_source_index(data)


@pytest.mark.parametrize("value", [True, "1", 2])
def test_schema_version_is_exact(index_data, value):
    index_data["schema_version"] = value
    assert validate_source_index(index_data)


def test_cli_json_filter_and_text_distinguish_restrictions(tmp_path, index_data, capsys):
    write_index(tmp_path, index_data)
    args = ["--root", str(tmp_path)]
    assert main(args + ["sources", "--license-status", "unresolved", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["matched_count"] == result["returned_count"] == 1
    assert result["entries"][0]["license"]["spdx"] is None
    assert result["invocable_in_hub"] is False
    assert main(args + ["sources"]) == 0
    text = capsys.readouterr().out
    assert "non-invocable" in text and "research-only" in text and "unresolved" in text
    assert main(args + ["check-sources"]) == 0
    assert "Metadata validation only" in capsys.readouterr().out


def test_cli_limits_text_but_json_can_return_complete_collection(tmp_path, index_data, capsys):
    row = index_data["entries"][0]
    index_data["entries"] = [dict(row, id=f"grasp-{i}", dedup_key=f"repo-{i}") for i in range(35)]
    write_index(tmp_path, index_data)
    args = ["--root", str(tmp_path), "sources"]
    assert main(args) == 0
    assert "35 matched; showing 30" in capsys.readouterr().out
    assert main(args + ["--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["entries"]) == 35
    assert main(args + ["--json", "--limit", "2"]) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["matched_count"] == 35 and response["returned_count"] == 2


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_invalid_cli_limit_is_not_an_unbounded_request(value):
    with pytest.raises(SystemExit) as error:
        main(["sources", "--limit", value])
    assert error.value.code == 2


def test_cli_rejects_corrupt_index_without_emitting_rows(tmp_path, index_data, capsys):
    index_data["entries"][0]["invocable_in_hub"] = True
    write_index(tmp_path, index_data)
    assert main(["--root", str(tmp_path), "sources", "--json"]) == 2
    output = capsys.readouterr()
    assert not output.out and "invocable_in_hub=false" in output.err


def test_source_index_never_changes_executable_registry(hub, index_data, capsys):
    before = set(load_catalog(hub).bundles)
    write_index(hub, index_data)
    assert main(["--root", str(hub), "sources", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["matched_count"] == 3
    assert set(load_catalog(hub).bundles) == before


def test_checked_in_index_is_valid_and_kept_out_of_the_catalog(capsys):
    data = load_source_index(REPO)
    assert data["collection"] == "pick-place"
    assert main(["--root", str(REPO), "check-sources"]) == 0
    assert "metadata valid" in capsys.readouterr().out
    catalog = load_catalog(REPO)
    indexed = {row["id"] for row in data["entries"]}
    assert not indexed & set(catalog.bundles)
    assert not any("candidates" in b.path.parts for b in catalog.bundles.values())


def test_check_sources_counts_license_and_review_level_separately(tmp_path, index_data, capsys):
    write_index(tmp_path, index_data)
    assert main(["--root", str(tmp_path), "check-sources"]) == 0
    out = capsys.readouterr().out
    assert "permissive: 1, copyleft: 0, research-only: 1, unresolved: 1" in out
    assert "metadata-indexed: 2, implementation-inspected: 1" in out
