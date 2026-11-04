from __future__ import annotations

import json

import pytest

from robomotif.catalog import load_catalog
from robomotif.search import infer_families, select, tokens
from robomotif.view import VIEW_FILE, make_view


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


def test_view_links_bundles_and_writes_recipes(hub, tmp_path):
    selection = select(load_catalog(hub), "pick up the block", tier="full")
    out = make_view(selection.picks, tmp_path / "view", query="q", recipes=[r for r, _ in selection.recipes])
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
