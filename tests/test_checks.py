from __future__ import annotations

import json

from conftest import write

from roboskillhub.catalog import load_catalog
from roboskillhub.checks import GapApi, check_all


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
        hub / ".roboskillhub/sources/upstream/tools/judge/SKILL.md",
        "---\nname: judge\ndescription: Upstream judge. Use when.\ngap: {requires: {}}\n---\n",
    )
    catalog = load_catalog(hub)
    assert catalog.bundles["judge"].native
    assert catalog.shadowed == [("judge", "hub", "upstream")]


def test_an_upstream_bundle_without_overlay_metadata_is_an_error(hub):
    write(
        hub / ".roboskillhub/sources/upstream/skills/new-skill/SKILL.md",
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


POLICY = """
    ---
    name: picking-cubes
    description: A checkpoint that picks a cube. Use when the gripper hovers over a cube.
    license: Apache-2.0
    metadata:
      hub:
        stage: execute
        version: 0.1.0
        embodiments: [any]
    gap:
      serving: {command: [python, server.py], protocol: websocket}
      exit_conditions: {completed: done, failed: errored}
    ---
    """
IO = """embodiments: [yam_i2rt]
        io:
          action: {space: joint_position, dim: 7, hz: 30}
          state: {dim: 7}
          cameras: [wrist, front]"""


def test_a_policy_names_its_robot_and_the_io_it_was_trained_with(hub):
    path = write(hub / "policies/picking-cubes/SKILL.md", POLICY)
    found = errors(hub)
    assert any("never [any]" in e for e in found)
    assert any("a policy declares io" in e for e in found)

    write(path, POLICY.replace("embodiments: [any]", IO))
    assert [e for e in errors(hub) if "picking-cubes" in e] == []

    write(path, POLICY.replace("embodiments: [any]", IO.replace("dim: 7, hz: 30", "dim: 0, hz: 30")))
    assert any("io.action needs space" in e for e in errors(hub))
    write(path, POLICY.replace("embodiments: [any]", IO.replace("[wrist, front]", "[]")))
    assert any("io.cameras must list" in e for e in errors(hub))


def test_only_a_policy_declares_io(hub):
    path = hub / "tools/judge/SKILL.md"
    io = "io: {action: {space: x, dim: 1, hz: 1}, cameras: [c]}"
    path.write_text(path.read_text().replace("embodiments: [any],", f"embodiments: [any], {io},"))
    assert any("only a policy declares it" in e for e in errors(hub))


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
    monkeypatch.setenv("ROBOSKILLHUB_CACHE", str(tmp_path / "empty-cache"))
    assert any("source is not synced" in e for e in errors(hub))
