from __future__ import annotations

import json
import re

from conftest import COMMIT

from roboskillhub.catalog import load_catalog
from roboskillhub.cli import main
from roboskillhub.site import markdown, render, write


def test_every_bundle_has_a_page_and_a_card(hub):
    pages = render(load_catalog(hub), "owner/hub")
    for name in ("judge", "grabbing-things", "peeking-at-sim", "routing-cables"):
        assert f"b/{name}.html" in pages
        assert f'"name":"{name}"' in pages["index.html"]
    for page in (
        "index.html",
        "start.html",
        "contribute.html",
        "recipes.html",
        "papers.html",
        "index.json",
        ".nojekyll",
    ):
        assert page in pages
    assert json.loads(pages["index.json"])["counts"]["skill"] == 3


def test_a_bundle_page_says_what_a_user_needs(hub):
    page = render(load_catalog(hub), "owner/hub")["b/grabbing-things.html"]
    assert "Grabs a thing and lifts it." in page
    assert (
        "<code>target_obb</code> OrientedBoundingBox" in page
        and "<code>grasp_pose</code> Se3Pose" in page
    )
    assert "<a href='judge.html'>judge</a>" in page
    assert "<code>robot.close_gripper</code>" in page
    assert "<code>grasped</code> held" in page and "<code>failed</code> not held" in page
    assert f"https://example.invalid/up/blob/{COMMIT}/skills/grabbing-things/SKILL.md" in page
    assert "roboskillhub view --bundles grabbing-things" in page
    judge = render(load_catalog(hub), "owner/hub")["b/judge.html"]
    assert "https://github.com/owner/hub/blob/main/tools/judge/SKILL.md" in judge
    assert "grabbing-things.html" in judge  # used by


def test_sim_only_and_two_arm_bundles_are_flagged(hub):
    pages = render(load_catalog(hub), "owner/hub")
    assert "simulator only" in pages["b/peeking-at-sim.html"]
    assert "2 arms" in pages["b/routing-cables.html"]
    cards = json.loads(re.search(r"const C=(\[.*?\]);", pages["index.html"]).group(1))
    by = {c["name"]: c for c in cards}
    assert by["peeking-at-sim"]["sim_only"] and not by["grabbing-things"]["sim_only"]


def test_the_site_is_deterministic_and_written_by_the_cli(hub, tmp_path):
    catalog = load_catalog(hub)
    assert render(catalog) == render(catalog)
    out = tmp_path / "site"
    assert main(["--root", str(hub), "site", "--out", str(out)]) == 0
    assert (out / "index.html").is_file() and (out / "b" / "judge.html").is_file()
    write(catalog, out)  # a second write empties the folder first
    assert sorted(p.name for p in (out / "b").iterdir()) == sorted(
        f"{n}.html" for n in catalog.bundles
    )


def test_markdown_covers_what_the_readme_uses():
    page = markdown(
        "# Title\n\nSome **bold** and `code` with [a link](CATALOG.md).\n\n- one\n- two\n\n"
        "```bash\nuv sync\n```\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n",
        "owner/hub",
    )
    assert '<h1 id="title">Title</h1>' in page
    assert "<strong>bold</strong>" in page and "<code>code</code>" in page
    assert 'href="https://github.com/owner/hub/blob/main/CATALOG.md"' in page
    assert "<ul><li>one</li><li>two</li></ul>" in page
    assert "<pre><code>uv sync</code></pre>" in page
    assert "<td>1</td><td>2</td>" in page
