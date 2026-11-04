from __future__ import annotations

import json

from conftest import write

from robomotif.catalog import load_catalog
from robomotif.checks import GapApi, check_all


def errors(root):
    return [str(i) for i in check_all(load_catalog(root), GapApi()) if i.level == "error"]


def test_the_fixture_hub_is_clean(hub):
    catalog = load_catalog(hub)
    assert sorted(catalog.bundles) == ["grabbing-things", "judge", "peeking-at-sim", "routing-cables"]
    assert catalog.bundles["judge"].native and catalog.bundles["grabbing-things"].source == "upstream"
    assert catalog.bundles["peeking-at-sim"].tier == "full"
    assert catalog.bundles["grabbing-things"].tier == "camera_only"
    assert errors(hub) == []


def test_native_bundle_shadows_upstream(hub):
    write(
        hub / ".motif/sources/upstream/tools/judge/SKILL.md",
        "---\nname: judge\ndescription: Upstream judge. Use when.\ngap: {requires: {}}\n---\n",
    )
    catalog = load_catalog(hub)
    assert catalog.bundles["judge"].native
    assert catalog.shadowed == [("judge", "hub", "upstream")]


def test_an_upstream_bundle_without_overlay_metadata_is_an_error(hub):
    write(
        hub / ".motif/sources/upstream/skills/new-skill/SKILL.md",
        "---\nname: new-skill\ndescription: New. Use when needed.\ngap:\n  exit_conditions: {ok: ok}\n---\n",
    )
    found = errors(hub)
    assert any("ships 'new-skill'; add its hub metadata" in e for e in found)
    assert any("needs a stage" in e for e in found)


def test_a_declared_tier_must_match_what_the_bundle_calls(hub):
    overlay = hub / "overlays/upstream.yaml"
    overlay.write_text(
        overlay.read_text().replace("peeking-at-sim: {", "peeking-at-sim: {tier: camera_only, ")
    )
    assert any("declares tier 'camera_only'" in e for e in errors(hub))


def test_native_bundles_need_use_when_license_and_version(hub):
    write(
        hub / "tools/judge/SKILL.md",
        """
        ---
        name: judge
        description: Decides things.
        metadata:
          hub: {stage: judge, embodiments: [any]}
        gap:
          requires: {}
        ---
        """,
    )
    found = errors(hub)
    assert any("Use when" in e for e in found)
    assert any("declares its license" in e for e in found)
    assert any("semver version" in e for e in found)


def test_heavy_imports_at_module_level_are_flagged(hub):
    (hub / "tools/judge/tools.py").write_text("import torch\n")
    assert any("load models on first call" in e for e in errors(hub))


def test_recipe_must_route_every_exit_and_reach_success(hub):
    path = hub / "recipes/peek-and-grab/RECIPE.md"
    path.write_text(
        path.read_text().replace("grab: {grasped: done, failed: abort}", "grab: {grasped: abort}")
    )
    found = errors(hub)
    assert any("exits ['failed'] of grabbing-things are not routed" in e for e in found)
    assert any("no success end is reachable" in e for e in found)
    assert any("'done' is unreachable" in e for e in found)


def test_recipe_tier_is_derived_from_its_skills(hub):
    path = hub / "recipes/peek-and-grab/RECIPE.md"
    path.write_text(path.read_text().replace("embodiments: [any],", "embodiments: [any], tier: camera_only,"))
    assert any("declares tier 'camera_only' but what it calls makes it 'full'" in e for e in errors(hub))


def test_paper_derived_entries_must_exist(hub):
    path = hub / "papers/grab-2026/PAPER.md"
    path.write_text(path.read_text().replace("skills/grabbing-things", "skills/no-such-skill"))
    assert any("'skills/no-such-skill'" in e for e in errors(hub))


def test_eval_records_are_validated_and_set_maturity(hub):
    record = {
        "bundle": "grabbing-things",
        "version": "0.1.0",
        "embodiment": "yam_i2rt",
        "setting": "sim",
        "tier": "camera_only",
        "suite": "rss/pick_place",
        "episodes": 20,
        "successes": 17,
        "passed": True,
        "date": "2026-11-04",
        "evidence": "https://example.invalid/run",
    }
    write(hub / "evals/grabbing-things/yam-sim.json", json.dumps(record))
    catalog = load_catalog(hub)
    assert catalog.bundles["grabbing-things"].maturity == "sim-verified"
    assert errors(hub) == []
    write(hub / "evals/grabbing-things/bad.json", json.dumps({**record, "successes": 30, "setting": "lab"}))
    found = errors(hub)
    assert any("between 0 and episodes" in e for e in found)
    assert any("setting is sim or real" in e for e in found)


def test_unsynced_source_is_an_error(hub, monkeypatch, tmp_path):
    monkeypatch.setenv("MOTIF_CACHE", str(tmp_path / "empty-cache"))
    assert any("source is not synced" in e for e in errors(hub))
