from __future__ import annotations

import json

import pytest
from conftest import write

from roboskillhub.catalog import load_catalog
from roboskillhub.search import infer_families, select, tokens, wilson_lower
from roboskillhub.view import VIEW_FILE, make_view


def test_stems_agree_across_inflections():
    assert tokens("placing placed places place") == ["plac"] * 4
    assert tokens("grasping grasped grasp") == ["grasp"] * 3


@pytest.mark.parametrize(
    ("query", "family"),
    [
        ("put the red block into the bowl", "pick-place"),
        ("把红色方块放进碗里", "pick-place"),
        ("route the cable through both clips", "cable-routing"),
        ("hang the mug on the hook", "insertion"),
        ("flip the crate upright", "reorientation"),
        ("assemble the jigsaw puzzle", "jigsaw"),
    ],
)
def test_task_family_is_inferred_from_the_instruction(query, family):
    assert family in infer_families(query)


def test_camera_only_drops_ground_truth_skills_and_recipes_that_need_them(hub):
    selection = select(load_catalog(hub), "pick up the block", embodiment="yam_i2rt", arms=1)
    assert "peeking-at-sim" not in selection.names
    assert "simulator ground truth" in selection.excluded["peeking-at-sim"]
    assert selection.recipes == []
    assert "grabbing-things" in selection.names


def test_arms_and_embodiment_filter(hub):
    catalog = load_catalog(hub)
    one_arm = select(catalog, "route the cable", embodiment="yam_i2rt", arms=1)
    assert one_arm.excluded["routing-cables"] == "needs 2 arms"
    two_arms = select(catalog, "route the cable", embodiment="yam_i2rt", arms=2)
    assert "routing-cables" in two_arms.names
    panda = select(catalog, "route the cable", embodiment="panda", arms=2)
    assert "not panda" in panda.excluded["routing-cables"]


def test_full_tier_keeps_the_recipe_and_adds_its_skills_and_their_tools(hub):
    selection = select(load_catalog(hub), "pick up the block", tier="full")
    assert [r.name for r, _ in selection.recipes] == ["peek-and-grab"]
    assert {"peeking-at-sim", "grabbing-things", "judge"} <= set(selection.names)
    judge = next(p for p in selection.picks if p.bundle.name == "judge")
    assert judge.reason == "called by grabbing-things"


def test_a_named_family_outranks_one_rare_shared_word(hub):
    write(
        hub / "skills/plugging-ports/SKILL.md",
        """
        ---
        name: plugging-ports
        description: Pushes a held plug into a port; a blocked plug backs off. Use when a plug must seat.
        license: Apache-2.0
        metadata:
          hub: {stage: execute, embodiments: [any], version: 0.1.0, tasks: [insertion]}
        gap:
          allowed_tools: [robot.get_observation]
          exit_conditions: {seated: in, failed: out}
        ---
        """,
    )
    selection = select(load_catalog(hub), "put the block in the bin", embodiment="yam_i2rt")
    execute = [p.bundle.name for p in selection.picks if p.bundle.stage == "execute"]
    assert execute[0] == "grabbing-things"
    assert selection.names.index("grabbing-things") < selection.names.index("plugging-ports")


def _eval(hub, bundle, setting, successes, episodes, embodiment="yam_i2rt"):
    record = {
        "bundle": bundle,
        "version": "0.1.0",
        "embodiment": embodiment,
        "setting": setting,
        "tier": "camera_only",
        "suite": "test",
        "episodes": episodes,
        "successes": successes,
        "passed": successes * 2 > episodes,
        "date": "2026-11-06",
        "evidence": "test",
    }
    path = hub / "evals" / bundle / f"{embodiment}-{setting}-{successes}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def test_evidence_on_the_asked_robot_raises_a_bundle(hub):
    write(
        hub / "skills/lifting-things/SKILL.md",
        """
        ---
        name: lifting-things
        description: Lifts a thing off the table. Use when an object must be picked up.
        license: Apache-2.0
        metadata:
          hub: {stage: execute, embodiments: [any], version: 0.1.0, tasks: [pick-place]}
        gap:
          allowed_tools: [robot.close_gripper]
          exit_conditions: {lifted: up, failed: down}
        ---
        """,
    )
    query = "pick up the thing"
    before = select(load_catalog(hub), query, embodiment="yam_i2rt").names
    favoured = before[1] if before[0] == "grabbing-things" else "grabbing-things"
    _eval(hub, favoured, "real", 18, 20)
    _eval(hub, before[0] if favoured != before[0] else before[1], "sim", 40, 40, "panda")
    after = select(load_catalog(hub), query, embodiment="yam_i2rt")
    assert after.names[0] == favoured
    assert after.evidence[favoured] == "real 18/20"


def test_wilson_lower_bound():
    assert wilson_lower(0, 0) == 0.0
    assert 0.0 < wilson_lower(9, 10) < wilson_lower(90, 100) < 0.9


def test_failure_search_ranks_only_the_asked_stages_and_offers_no_recipe(hub):
    selection = select(
        load_catalog(hub),
        "pick up the block",
        tier="full",
        stages=("execute",),
        failure="the fingers closed over the thing",
    )
    assert selection.recipes == []
    ranked = [p for p in selection.picks if p.score > 0]
    assert ranked and all(p.bundle.stage == "execute" for p in ranked)
    assert "peeking-at-sim" not in selection.names


def test_view_links_bundles_and_writes_recipes(hub, tmp_path):
    selection = select(load_catalog(hub), "pick up the block", tier="full")
    out = make_view(
        selection.picks, tmp_path / "view", query="q", recipes=[r for r, _ in selection.recipes]
    )
    assert (out / "skills/grabbing-things/SKILL.md").is_file()
    assert (out / "tools/judge").is_symlink()
    record = json.loads((out / VIEW_FILE).read_text())
    assert record["recipes"] == ["peek-and-grab"]
    text = (out / "RECIPES.md").read_text()
    assert "- peek: peeking-at-sim [name=target]; found -> grab, not_found -> abort" in text
    make_view(selection.picks[:1], out)  # a view replaces a view
    assert len(json.loads((out / VIEW_FILE).read_text())["bundles"]) == 1


def test_view_refuses_to_replace_a_directory_it_did_not_write(hub, tmp_path):
    (tmp_path / "precious").mkdir()
    with pytest.raises(FileExistsError):
        make_view([], tmp_path / "precious")
