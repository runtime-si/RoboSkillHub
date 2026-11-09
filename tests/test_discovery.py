from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from roboskillhub.catalog import load_catalog
from roboskillhub.cli import main
from roboskillhub.discovery import discover, load_candidates, validate_candidates

ROOT = Path(__file__).resolve().parents[1]


def test_reviewed_catalog_has_sources_contracts_and_no_executable_claims():
    data = load_candidates(ROOT)
    assert len(data["candidates"]) >= 20
    assert all(row["invocable_in_hub"] is False for row in data["candidates"])
    assert all(row["source_revision"] for row in data["candidates"])


@pytest.mark.parametrize("query", ["grasp", "抓取", "插入", "YAM"])
def test_discovery_supports_concrete_bilingual_capability_queries(query):
    assert discover(load_candidates(ROOT), query)


def test_unknown_capability_does_not_fill_arbitrary_stages():
    assert discover(load_candidates(ROOT), "unobtainium-zqxv") == []


def test_research_only_material_is_an_explicit_opt_in():
    data = load_candidates(ROOT)
    default = discover(data)
    all_rows = discover(data, include_research_only=True)
    assert len(all_rows) > len(default)
    assert all(row["intake_group"] == "open-source-code" for row in default)
    assert any(row["intake_group"] == "research-only" for row in all_rows)


def test_exact_filters_and_cli_json_preserve_evidence(capsys):
    assert (
        main(
            [
                "--root",
                str(ROOT),
                "discover",
                "",
                "--destination",
                "policy",
                "--priority",
                "P2",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["invocable_in_hub"] is False
    assert result["candidates"]
    for row in result["candidates"]:
        assert row["hub_destination"] == "policy"
        assert row["priority"] == "P2"
        assert row["evidence_urls"] and row["code_license"] and row["weights"]


@pytest.mark.parametrize("change", ["duplicate", "fake-ready", "moving-pin"])
def test_rejects_ambiguous_ids_readiness_and_unstable_revision(change):
    data = copy.deepcopy(load_candidates(ROOT))
    if change == "duplicate":
        data["candidates"].append(copy.deepcopy(data["candidates"][0]))
    elif change == "fake-ready":
        data["candidates"][0]["invocable_in_hub"] = True
    else:
        data["candidates"][0]["source_revision"] = "main"
    assert validate_candidates(data)


@pytest.mark.parametrize("group", [[], {}, None, 123, "unknown"])
def test_rejects_invalid_intake_groups_without_raising(group):
    data = copy.deepcopy(load_candidates(ROOT))
    data["candidates"][0]["intake_group"] = group
    assert any("intake_group" in error for error in validate_candidates(data))


@pytest.mark.parametrize(
    "record,field",
    [
        ("code_license", "id"),
        ("code_license", "notes"),
        ("weights", "availability"),
        ("weights", "license"),
        ("weights", "notes"),
    ],
)
@pytest.mark.parametrize("value", [None, [], 123, "", "   "])
def test_rejects_invalid_license_and_weight_text(record, field, value):
    data = copy.deepcopy(load_candidates(ROOT))
    data["candidates"][0][record][field] = value
    assert any(f"{record}.{field}" in error for error in validate_candidates(data))


@pytest.mark.parametrize("record", ["code_license", "weights"])
@pytest.mark.parametrize("value", [123, [], "", "not a url", "https://[broken", "https://"])
def test_rejects_invalid_license_and_weight_urls(record, value):
    data = copy.deepcopy(load_candidates(ROOT))
    data["candidates"][0][record]["url"] = value
    assert any(f"{record}.url" in error for error in validate_candidates(data))


@pytest.mark.parametrize(
    "value", [None, "gs://bucket/checkpoints/model", "https://example.com/model"]
)
def test_accepts_explicitly_absent_and_supported_weight_urls(value):
    data = copy.deepcopy(load_candidates(ROOT))
    data["candidates"][0]["weights"]["url"] = value
    assert validate_candidates(data) == []


@pytest.mark.parametrize("value", [None, "gs://bucket/LICENSE"])
def test_code_license_url_requires_http_evidence(value):
    data = copy.deepcopy(load_candidates(ROOT))
    data["candidates"][0]["code_license"]["url"] = value
    assert any("code_license.url" in error for error in validate_candidates(data))


@pytest.mark.parametrize("field", ["source_url", "evidence_urls"])
def test_malformed_bracket_urls_return_validation_errors(field):
    data = copy.deepcopy(load_candidates(ROOT))
    data["candidates"][0][field] = (
        ["https://[broken"] if field == "evidence_urls" else "https://[broken"
    )
    assert any(field in error for error in validate_candidates(data))


def test_source_revision_must_be_present_but_can_be_explicit_null():
    data = copy.deepcopy(load_candidates(ROOT))
    row = data["candidates"][0]
    row["source_revision"] = None
    assert validate_candidates(data) == []
    del row["source_revision"]
    assert any("source_revision is required" in error for error in validate_candidates(data))


def test_source_intake_never_adds_runtime_bundles(hub):
    before = set(load_catalog(hub).bundles)
    (hub / "candidates").mkdir()
    (hub / "candidates/catalog.json").write_text(
        (ROOT / "candidates/catalog.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert len(load_candidates(hub)["candidates"]) >= 20
    assert set(load_catalog(hub).bundles) == before
