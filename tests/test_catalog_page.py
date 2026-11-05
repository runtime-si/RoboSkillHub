from __future__ import annotations

from conftest import COMMIT

from roboskillhub.catalog import load_catalog
from roboskillhub.catalog_page import render, summary
from roboskillhub.cli import main


def section(page: str, name: str) -> str:
    return page.split(f"\n### {name}\n", 1)[1].split("\n### ", 1)[0].split("\n## ", 1)[0]


def test_every_bundle_has_an_overview_row_and_a_section(hub):
    page = render(load_catalog(hub))
    assert "1 tools, 3 skills, 0 policies, 1 recipe and 1 paper." in page
    for name in ("judge", "grabbing-things", "peeking-at-sim", "routing-cables"):
        assert f"| [`{name}`](#{name}) |" in page
        assert f"\n### {name}\n" in page
    assert page.index("## Tools") < page.index("## Skills") < page.index("## Tool details")


def test_a_section_says_what_a_reviewer_needs(hub):
    page = render(load_catalog(hub))
    grab = section(page, "grabbing-things")
    assert "execute · robots: any · tasks: pick-place" in grab
    assert "- **In**: `target_obb` OrientedBoundingBox" in grab
    assert "- **Out**: `grasp_pose` Se3Pose" in grab
    assert "- **Calls**: [`judge`](#judge); connector `robot.close_gripper`" in grab
    assert "  - `grasped`: held" in grab and "  - `failed`: not held" in grab
    assert f"https://example.invalid/up/blob/{COMMIT}/skills/grabbing-things/SKILL.md" in grab

    assert "[`SKILL.md`](tools/judge/SKILL.md)" in section(page, "judge")
    assert "  - `judge.check`: Check a thing." in section(page, "judge")


def test_robots_flag_sim_only_and_two_arm_bundles(hub):
    page = render(load_catalog(hub))
    assert "robots: any, **sim only**" in section(page, "peeking-at-sim")
    assert "robots: yam_i2rt, 2 arms" in section(page, "routing-cables")


def test_output_names_with_placeholders_survive_markdown(hub):
    page = render(load_catalog(hub))
    assert "- **Out**: `<name>_obb` OrientedBoundingBox" in section(page, "peeking-at-sim")


def test_summary_is_the_first_sentence_cut_when_long():
    assert summary("Grabs a thing and lifts it. Use when an object must be picked up.") == (
        "Grabs a thing and lifts it."
    )
    detail = "very " * 40 + "carefully. More."
    assert summary("Plans collision-free motion — " + detail) == "Plans collision-free motion."
    assert summary("Segment Anything 3 — " + detail).startswith("Segment Anything 3 — very")
    long = summary("Fits " + "a box, " * 40 + "then stops.")
    assert long.endswith("…") and len(long) <= 120


def test_the_command_writes_the_page_and_checks_it(hub, capsys):
    assert main(["--root", str(hub), "catalog", "--check"]) == 1
    assert main(["--root", str(hub), "catalog"]) == 0
    assert (hub / "CATALOG.md").read_text(encoding="utf-8") == render(load_catalog(hub))
    assert main(["--root", str(hub), "catalog", "--check"]) == 0
    assert "is current" in capsys.readouterr().out
