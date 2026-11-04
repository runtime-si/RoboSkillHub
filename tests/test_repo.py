"""This repository itself: it checks clean, and its templates scaffold bundles that check clean."""

from __future__ import annotations

import shutil

import pytest
from conftest import REPO

from robomotif.catalog import load_catalog
from robomotif.checks import check_all, load_gap_api
from robomotif.scaffold import new
from robomotif.sources import cache_dir, source_dir


def synced() -> bool:
    return all(
        source_dir(cache_dir(REPO), name).is_dir() for name in ("open-robot-skills", "graph-as-policy")
    )


@pytest.mark.skipif(not synced(), reason="run `motif sync` first")
def test_the_hub_checks_clean():
    issues = check_all(load_catalog(REPO), load_gap_api())
    assert [str(i) for i in issues if i.level == "error"] == []


@pytest.mark.skipif(not synced(), reason="run `motif sync` first")
@pytest.mark.parametrize("kind", ["skill", "tool", "recipe", "paper"])
def test_templates_scaffold_bundles_that_check_clean(kind, tmp_path, monkeypatch):
    root = tmp_path / "hub"
    shutil.copytree(REPO, root, ignore=shutil.ignore_patterns(".venv", ".motif", ".git", "__pycache__"))
    monkeypatch.setenv("MOTIF_CACHE", str(cache_dir(REPO)))
    name = {"paper": "some-paper-2026"}.get(kind, f"my-{kind}")
    dest = new(root, kind, name)
    assert "{{name}}" not in "".join(p.read_text() for p in dest.rglob("*") if p.is_file())
    issues = check_all(load_catalog(root), load_gap_api())
    mine = [str(i) for i in issues if i.level == "error" and name in i.where]
    assert mine == []


def test_a_scaffolded_policy_is_rejected_until_its_weights_and_io_are_filled_in(tmp_path):
    root = tmp_path / "hub"
    shutil.copytree(REPO / "templates", root / "templates")
    dest = new(root, "policy", "my-policy")
    assert "hf://OWNER/REPO@REVISION" in (dest / "SKILL.md").read_text()
    from robomotif.checks import check_hub_metadata
    from robomotif.frontmatter import read

    hub = read(dest / "SKILL.md")[0]["metadata"]["hub"]
    found = [
        i.message for i in check_hub_metadata(hub, where="x", kind="policy", native=True, derived_tier=None)
    ]
    assert any("must pin a full 40-character commit" in m for m in found)
    assert any("sha256 of their file manifest" in m for m in found)
    assert any("io.action needs space" in m for m in found)
