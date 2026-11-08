"""planning-a-brick-mosaic-from-an-image: pixelation, CIEDE2000 matching, dithering, stock,
merging, the placement convention, exports, every route, and the downstream brick skills."""

from __future__ import annotations

import itertools
import json
import re
import xml.etree.ElementTree as xmlET
from pathlib import Path

import cv2
import numpy as np
import pytest
from gap.testing import FakeContext
from scipy.optimize import linear_sum_assignment

BUNDLE = "planning-a-brick-mosaic-from-an-image"
ROOT = Path(__file__).resolve().parents[2]

RED = (180, 0, 0)  # LDraw red B40000
WHITE = (244, 244, 244)  # LDraw white F4F4F4
BLUE = (30, 90, 168)  # LDraw blue 1E5AA8
GREEN = (0, 133, 43)  # LDraw green 00852B
BLACK = (27, 42, 52)  # LDraw black 1B2A34


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["plan_mosaic"].module


@pytest.fixture(scope="module")
def order_script(skills_registry):
    return (
        skills_registry.get("planning-brick-assembly-order").canonical_scripts["plan_order"].module
    )


@pytest.fixture(scope="module")
def check_script(skills_registry):
    return (
        skills_registry.get("checking-design-connectivity-and-collisions")
        .canonical_scripts["check_design"]
        .module
    )


@pytest.fixture(scope="module")
def pose_script(skills_registry):
    return (
        skills_registry.get("computing-brick-pose-from-stud-grid")
        .canonical_scripts["brick_pose"]
        .module
    )


def solid(h, w, rgb):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = rgb
    return img


def plan(script, image, **kw):
    kw.setdefault("write_files", False)
    return script.run(FakeContext(), image=image, **kw)


def covered(out):
    """Stud cells -> placement, asserting no cell is covered twice."""
    seen = {}
    for p in out["placements"]:
        for c in p["cells"]:
            assert tuple(c) not in seen, f"cell {c} covered twice"
            seen[tuple(c)] = p
    return seen


# ---------------------------------------------------------------------------
# The basic pipeline
# ---------------------------------------------------------------------------


def test_exact_palette_blocks_plan_with_zero_error(script):
    img = np.zeros((40, 40, 3), np.uint8)
    img[:20, :20], img[:20, 20:], img[20:, :20], img[20:, 20:] = RED, WHITE, BLUE, GREEN
    out = plan(script, img, plate_studs=[8, 8])
    assert out["route"] == "planned"
    assert out["n_cells"] == out["n_placements"] == 64
    assert out["mean_delta_e"] == pytest.approx(0, abs=1e-6)
    assert out["ideal_mean_delta_e"] == pytest.approx(0, abs=1e-6)
    assert {b["colour"]: b["count"] for b in out["bill_of_materials"]} == {
        "red": 16,
        "white": 16,
        "blue": 16,
        "green": 16,
    }
    assert all(b["part"] == "3024" and b["size"] == "1x1" for b in out["bill_of_materials"])
    assert len(covered(out)) == 64
    assert all(p["layer"] == 1 and p["cell"][2] == 1 for p in out["placements"])
    assert out["n_colours"] == 4


def test_image_up_is_plate_plus_y_and_right_is_plus_x(script):
    """Image top-left red on white: the red part sits at x = origin_x, y = origin_y + h - 1."""
    img = solid(6, 8, WHITE)
    img[0, 0] = RED
    img[5, 7] = BLUE
    out = plan(script, img, plate_studs=[12, 10], mosaic_size=[8, 6], origin=[2, 3])
    by_colour = {p["colour"]: p for p in out["placements"] if p["colour"] != "white"}
    assert by_colour["red"]["cell"] == [2, 3 + 5, 1]
    assert by_colour["blue"]["cell"] == [2 + 7, 3, 1]
    assert out["colour_grid"][0][0] == "red" and out["colour_grid"][5][7] == "blue"
    xs = {c[0] for c in covered(out)}
    ys = {c[1] for c in covered(out)}
    assert xs == set(range(2, 10)) and ys == set(range(3, 9))


def test_cells_are_averaged_in_linear_light(script):
    """A fine black/white checker averages to linear 0.5 = sRGB ~188 (a mid grey); averaging sRGB
    values (brickmos' bilinear) would give 128-ish."""
    img = np.zeros((64, 64, 3), np.uint8)
    img[::2, ::2] = img[1::2, 1::2] = 255
    pal = [{"name": "grey188", "rgb": [188, 188, 188]}, {"name": "grey128", "rgb": [128] * 3}]
    assert plan(script, img, plate_studs=[4, 4], palette=pal)["colour_grid"][0][0] == "grey188"


def test_matching_is_ciede2000_not_rgb_distance(script):
    """A dark navy cell: nearer to black in sRGB distance, nearer to dark blue in CIEDE2000."""
    pal = [{"name": "black", "rgb": [0, 0, 0]}, {"name": "dark_blue", "rgb": [20, 30, 110]}]
    cell = (8, 12, 45)  # sRGB distance 47 vs 69; CIEDE2000 16.5 vs 12.6
    rgb_d = [np.linalg.norm(np.subtract(cell, p["rgb"])) for p in pal]
    assert rgb_d[0] < rgb_d[1]
    out = plan(script, solid(4, 4, cell), plate_studs=[2, 2], palette=pal)
    assert out["colour_grid"][0][0] == "dark_blue"


def test_vendored_ciede2000_matches_sharma_reference_pairs(script):
    """Sharma, Wu, Dalal (2005) table 1, pairs 1, 7 and 17."""
    from mosaic_ciede2000 import deltaE_ciede2000

    pairs = [
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
        ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ]
    for a, b, want in pairs:
        assert deltaE_ciede2000(np.array(a), np.array(b)) == pytest.approx(want, abs=1e-4)


@pytest.mark.parametrize(
    ("fit", "expected"),
    [
        ("crop", {"green"}),
        ("stretch", {"red", "green", "blue"}),
        ("pad", {"white", "red", "green", "blue"}),
    ],
)
def test_fit_modes(script, fit, expected):
    """A 3:1 strip red | green | blue on a square mosaic."""
    img = np.zeros((30, 90, 3), np.uint8)
    img[:, :30], img[:, 30:60], img[:, 60:] = RED, GREEN, BLUE
    out = plan(script, img, plate_studs=[9, 9], fit=fit, palette="lego_basic")
    assert {b["colour"] for b in out["bill_of_materials"]} == expected
    if fit == "pad":
        assert out["colour_grid"][0] == ["white"] * 9 and out["colour_grid"][-1] == ["white"] * 9


def test_rgba_is_composited_over_the_background_and_files_load(script, tmp_path):
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[..., :3] = RED
    rgba[:, :4, 3] = 255  # left half opaque red, right half transparent
    path = tmp_path / "logo.png"
    cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    out = plan(script, None, image_path=str(path), plate_studs=[4, 4], background_rgb=list(BLUE))
    assert out["route"] == "planned"
    assert out["colour_grid"][0] == ["red", "red", "blue", "blue"]


def test_greyscale_and_float_images(script):
    grey = np.full((10, 10), 244, np.uint8)
    assert plan(script, grey, plate_studs=[2, 2])["colour_grid"] == [["white"] * 2] * 2
    f = np.ones((10, 10, 3)) * np.array(BLUE) / 255.0
    assert plan(script, f, plate_studs=[2, 2])["colour_grid"] == [["blue"] * 2] * 2


def test_small_image_is_upsampled_cell_for_pixel(script):
    img = np.array([[RED, BLUE]], np.uint8)
    out = plan(script, img, plate_studs=[4, 2], fit="stretch")
    assert out["colour_grid"] == [["red", "red", "blue", "blue"]] * 2


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def test_uniform_region_merges_into_2x2(script):
    out = plan(script, solid(8, 8, RED), plate_studs=[4, 4], sizes=["1x1", "1x2", "2x2"])
    assert out["route"] == "planned"
    assert out["n_placements"] == 4
    assert {p["size"] for p in out["placements"]} == {"2x2"}
    assert {p["part"] for p in out["placements"]} == {"3022"}
    assert len(covered(out)) == 16


def test_odd_region_uses_2x2_1x2_and_1x1_with_rotation_convention(script):
    out = plan(script, solid(6, 6, RED), plate_studs=[3, 3], sizes=["1x1", "1x2", "2x2"])
    sizes = sorted(p["size"] for p in out["placements"])
    assert sizes == ["1x1", "1x2", "1x2", "2x2"]
    assert len(covered(out)) == 9
    for p in out["placements"]:
        a, b = (int(v) for v in p["size"].split("x"))
        assert (p["span_x"], p["span_y"]) == ((a, b) if p["rotation"] == 0 else (b, a))
        assert p["cell"] == [min(c[0] for c in p["cells"]), min(c[1] for c in p["cells"]), 1]


def test_merging_never_joins_two_colours(script):
    img = np.zeros((4, 4, 3), np.uint8)
    img[:] = WHITE
    img[:, 1] = RED  # a red column one wide
    out = plan(script, img, plate_studs=[4, 4], sizes=["1x1", "1x2", "2x2", "1x4"])
    for p in out["placements"]:
        assert len({out["colour_grid"][3 - (c[1])][c[0]] for c in p["cells"]}) == 1
    red = [p for p in out["placements"] if p["colour"] == "red"]
    assert len(red) == 1 and red[0]["size"] == "1x4" and red[0]["rotation"] == 0


def test_merged_part_stock_is_respected(script):
    out = plan(
        script,
        solid(8, 8, RED),
        plate_studs=[4, 4],
        sizes=["1x1", "2x2"],
        inventory={"red": {"2x2": 1, "1x1": 12}},
    )
    assert out["route"] == "planned"
    bom = {b["size"]: b["count"] for b in out["bill_of_materials"]}
    assert bom == {"2x2": 1, "1x1": 12}


def test_cells_no_part_fits_take_the_nearest_colour_with_1x1(script):
    """Red has only 2x2s; a 3x3 red area leaves 5 cells no 2x2 fits: they go to dark red."""
    out = plan(
        script,
        solid(6, 6, RED),
        plate_studs=[3, 3],
        sizes=["1x1", "2x2"],
        inventory={"red": {"2x2": 5}, "dark_red": 20, "white": 20},
    )
    assert out["route"] == "planned_with_substitutions"
    assert out["n_substituted"] == 5
    assert (
        out["substitutions"][0]["wanted"] == "red" and out["substitutions"][0]["used"] == "dark_red"
    )
    assert len(covered(out)) == 9


# ---------------------------------------------------------------------------
# Stock
# ---------------------------------------------------------------------------


def test_short_colour_is_substituted_by_the_nearest_in_stock(script):
    out = plan(
        script,
        solid(4, 4, RED),
        plate_studs=[4, 4],
        inventory={"red": 10, "dark_red": 100, "blue": 100},
    )
    assert out["route"] == "planned_with_substitutions"
    bom = {b["colour"]: b["count"] for b in out["bill_of_materials"]}
    assert bom == {"red": 10, "dark_red": 6}
    assert out["substitutions"] == [
        {
            "wanted": "red",
            "used": "dark_red",
            "cells": 6,
            "mean_extra_delta_e": pytest.approx(out["substitutions"][0]["mean_extra_delta_e"]),
        }
    ]
    assert out["substitutions"][0]["mean_extra_delta_e"] > 0
    assert all(b["in_stock"] is not None for b in out["bill_of_materials"])


def test_stock_assignment_is_optimal_against_brute_force(script):
    """Random cells and capacities: the program's total CIEDE2000 equals the Hungarian optimum on
    the expanded (one column per stud) cost matrix."""
    from mosaic_ciede2000 import deltaE_ciede2000

    rng = np.random.default_rng(3)
    for _ in range(5):
        cost = rng.uniform(0, 50, (12, 4))
        caps = rng.integers(1, 6, 4).astype(float)
        while caps.sum() < 12:
            caps[rng.integers(4)] += 1
        got = script._transport(cost, caps)
        assert (got >= 0).all()
        assert all((got == k).sum() <= caps[k] for k in range(4))
        cols = list(itertools.chain.from_iterable([k] * int(caps[k]) for k in range(4)))
        expanded = cost[:, cols]
        r, c = linear_sum_assignment(expanded)
        assert cost[np.arange(12), got].sum() == pytest.approx(expanded[r, c].sum(), abs=1e-6)
    assert deltaE_ciede2000 is not None


def test_stock_short_overall_leaves_cells_empty(script, tmp_path):
    out = plan(
        script,
        solid(4, 4, RED),
        plate_studs=[4, 4],
        inventory={"red": 5, "white": 3},
        write_files=True,
        out_dir=str(tmp_path),
    )
    assert out["route"] == "inventory_short"
    assert out["n_empty"] == 8 and len(out["empty_cells"]) == 8
    assert out["n_placements"] == 8
    assert {b["colour"]: b["count"] for b in out["bill_of_materials"]} == {"red": 5, "white": 3}
    assert not set(map(tuple, out["empty_cells"])) & set(covered(out))
    assert sum(row.count(None) for row in out["colour_grid"]) == 8
    assert Path(out["preview_path"]).is_file()


def test_no_stock_at_all_is_inventory_short_with_no_placements(script):
    out = plan(script, solid(4, 4, RED), plate_studs=[2, 2], inventory={"red": 0})
    assert out["route"] == "inventory_short" and out["n_placements"] == 0 and out["n_empty"] == 4


def test_unlimited_stock_when_no_inventory_and_names_are_forgiving(script):
    out = plan(
        script, solid(4, 4, (150, 150, 150)), plate_studs=[2, 2], inventory={"Light Bluish Gray": 4}
    )
    assert out["route"] == "planned"
    assert out["bill_of_materials"][0]["colour"] == "light_bluish_grey"


# ---------------------------------------------------------------------------
# Dithering
# ---------------------------------------------------------------------------


def test_dithering_a_mid_grey_with_black_and_white_mixes_them(script):
    pal = [{"name": "black", "rgb": [0, 0, 0]}, {"name": "white", "rgb": [255, 255, 255]}]
    grey = solid(32, 32, (119, 119, 119))  # CIELAB L* 50
    plain = plan(script, grey, plate_studs=[16, 16], palette=pal)
    dith = plan(script, grey, plate_studs=[16, 16], palette=pal, dither="floyd_steinberg")
    assert plain["n_colours"] == 1
    counts = {b["colour"]: b["count"] for b in dith["bill_of_materials"]}
    assert 0.3 < counts["white"] / 256 < 0.7
    assert dith["mean_delta_e_blurred"] < 0.5 * plain["mean_delta_e_blurred"]
    assert dith["mean_delta_e"] > plain["mean_delta_e"]
    assert dith["route"] == "planned"


def test_dithering_respects_stock(script):
    pal = [{"name": "black", "rgb": [0, 0, 0]}, {"name": "white", "rgb": [255, 255, 255]}]
    out = plan(
        script,
        solid(16, 16, (119, 119, 119)),
        plate_studs=[8, 8],
        palette=pal,
        dither="floyd_steinberg",
        inventory={"black": 10, "white": 60},
    )
    counts = {b["colour"]: b["count"] for b in out["bill_of_materials"]}
    assert counts["black"] <= 10 and counts["white"] <= 60
    assert sum(counts.values()) == 64
    assert out["route"] == "planned_with_substitutions"


# ---------------------------------------------------------------------------
# Order, palettes, exports
# ---------------------------------------------------------------------------


def test_orders(script):
    img = np.zeros((4, 4, 3), np.uint8)
    img[:] = WHITE
    img[:2, :2] = RED
    rows = plan(script, img, plate_studs=[4, 4])["placements"]
    assert [p["cell"][:2] for p in rows[:5]] == [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1]]
    assert [p["seq"] for p in rows] == list(range(16)) and [p["step"] for p in rows] == list(
        range(1, 17)
    )
    serp = plan(script, img, plate_studs=[4, 4], order="serpentine")["placements"]
    assert [p["cell"][:2] for p in serp[3:6]] == [[3, 0], [3, 1], [2, 1]]
    col = plan(script, img, plate_studs=[4, 4], order="colour")["placements"]
    assert [p["colour"] for p in col] == ["white"] * 12 + ["red"] * 4


def test_custom_and_brickmos_palettes(script):
    pal = [
        {"name": "mine", "hex": "#B40000", "bricklink_id": 5},
        {"name": "lab_white", "lab": [96, 0, 0]},
    ]
    out = plan(script, solid(4, 4, RED), plate_studs=[2, 2], palette=pal)
    assert (
        out["placements"][0]["colour"] == "mine"
        and out["placements"][0]["bricklink_colour_id"] == 5
    )
    bm = plan(script, solid(4, 4, (179, 0, 6)), plate_studs=[2, 2], palette="brickmos")
    assert (
        bm["placements"][0]["colour"] == "Red" and bm["placements"][0]["bricklink_colour_id"] == 5
    )
    assert bm["placements"][0]["ldraw_colour"] is None


def test_bricklink_xml_and_files(script, tmp_path):
    img = np.zeros((8, 8, 3), np.uint8)
    img[:, :4], img[:, 4:] = RED, BLUE
    out = plan(
        script,
        img,
        plate_studs=[8, 8],
        write_files=True,
        out_dir=str(tmp_path),
        spares=2,
        preview_px=10,
    )
    root = xmlET.fromstring(out["bricklink_xml"])
    items = {(i.findtext("ITEMID"), i.findtext("COLOR")): int(i.findtext("MINQTY")) for i in root}
    assert items == {("3024", "5"): 34, ("3024", "7"): 34}
    assert all(i.findtext("ITEMTYPE") == "P" and i.findtext("CONDITION") == "X" for i in root)
    prev = cv2.imread(out["preview_path"])
    assert prev.shape == (80, 80, 3)
    assert cv2.imread(out["pixelated_path"]).shape == (80, 80, 3)
    assert (tmp_path / "bricklink.xml").read_text() == out["bricklink_xml"]
    plan_json = json.loads((tmp_path / "plan.json").read_text())
    assert len(plan_json["placements"]) == 64
    # The preview's top-left stud shows red (BGR), the top-right blue.
    assert tuple(prev[2, 1]) == (0, 0, 180) and tuple(prev[2, 78]) == (168, 90, 30)


def test_brick_and_tile_kinds_get_their_part_numbers(script):
    for kind, part in (("brick", "3005"), ("tile", "3070b")):
        out = plan(script, solid(2, 2, RED), plate_studs=[1, 1], part_kind=kind)
        assert out["placements"][0]["kind"] == kind and out["placements"][0]["part"] == part
    duplo = plan(script, solid(2, 2, RED), plate_studs=[1, 1], part_kind="brick", system="duplo")
    assert duplo["placements"][0]["part"] is None and duplo["placements"][0]["system"] == "duplo"


# ---------------------------------------------------------------------------
# Invalid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image",
    [
        None,
        "/nonexistent/x.png",
        np.zeros((0, 4, 3)),
        np.zeros((4, 4, 2)),
        np.full((4, 4, 3), np.nan),
    ],
)
def test_invalid_images(script, image):
    out = plan(script, image, plate_studs=[2, 2])
    assert out["route"] == "invalid_image" and out["detail"]
    assert out["placements"] == []


def test_undecodable_file_is_invalid_image(script, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert plan(script, str(bad), plate_studs=[2, 2])["route"] == "invalid_image"


@pytest.mark.parametrize(
    "kw",
    [
        {"palette": "rainbow"},
        {"palette": []},
        {"palette": [{"name": "x"}]},
        {"palette": [{"name": "a", "rgb": [1, 2, 3]}, {"name": "A", "rgb": [3, 2, 1]}]},
        {"inventory": {"chartreuse": 4}},
        {"inventory": {"red": -1}},
        {"inventory": {"red": {"2x2": 3}}},
        {"sizes": ["1x2"]},
        {"sizes": ["1xQ"]},
        {"mosaic_size": [5, 2]},
        {"origin": [4, 0]},
        {"plate_studs": [0, 4]},
        {"plate_studs": [200, 200]},
        {"dither": "atkinson"},
        {"fit": "zoom"},
        {"resample": "cubic"},
        {"order": "random"},
        {"part_kind": "slope"},
        {"system": "megablocks"},
    ],
)
def test_invalid_inputs(script, kw):
    kw = {"plate_studs": [4, 4], **kw}
    out = plan(script, solid(4, 4, RED), **kw)
    assert out["route"] == "invalid_input", kw
    assert out["detail"]


def test_example_declares_every_route(script):
    example = json.loads(
        (ROOT / "skills" / BUNDLE / "examples/canonical_subgraph.json").read_text()
    )
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    source = (ROOT / "skills" / BUNDLE / "scripts/plan_mosaic.py").read_text()
    routes = set(re.findall(r'route(?:"\]|) = "([a-z_]+)"', source))
    routes |= set(re.findall(r'route="([a-z_]+)"', source))
    routes.add("invalid_input")
    assert routes == {
        "planned",
        "planned_with_substitutions",
        "inventory_short",
        "invalid_image",
        "invalid_input",
    }
    assert routes <= declared
    skill_md = (ROOT / "skills" / BUNDLE / "SKILL.md").read_text()
    for r in routes:
        assert f"    {r}:" in skill_md


# ---------------------------------------------------------------------------
# Downstream brick skills
# ---------------------------------------------------------------------------


def mosaic(script):
    img = np.zeros((48, 48, 3), np.uint8)
    img[:] = WHITE
    cv2.circle(img, (24, 24), 14, RED, -1)
    img[:8] = BLACK
    return plan(script, img, plate_studs=[12, 12], sizes=["1x1", "1x2", "2x2"])


def test_placements_pass_checking_design(script, check_script):
    out = mosaic(script)
    res = check_script.run(
        FakeContext(), bricks=out["placements_json"], plate_size="12x12", stability="off"
    )
    assert res["route"] == "valid", res["detail"]
    assert res["n_bricks"] == out["n_placements"]


def test_steps_json_feeds_assembly_order_and_keeps_the_order(script, order_script):
    out = mosaic(script)
    res = order_script.run(FakeContext(), steps_json=out["steps_json"], check_fingers=False)
    assert res["route"] == "sequence_ready"
    assert [e["id"] for e in res["sequence"]] == [p["id"] for p in out["placements"]]
    assert [e["cell"]["x"] for e in res["sequence"]] == [p["x"] for p in out["placements"]]
    # A dense floor leaves a parallel gripper's fingers no free pair of sides (SKILL.md limits).
    fingers = order_script.run(FakeContext(), steps_json=out["steps_json"], check_fingers=True)
    assert fingers["route"] == "finger_blocked"
    assert 0 < fingers["n_placed"] < out["n_placements"]


def test_pose_cell_and_cells_feed_the_pose_skill(script, pose_script):
    out = mosaic(script)
    done: list[dict] = []
    routes = []
    for p in out["placements"][:16]:
        res = pose_script.run(
            FakeContext(),
            plate_pose={"position": [0.4, 0.0, 0.0], "quaternion": [1.0, 0.0, 0.0, 0.0]},
            plate_studs=out["plate_studs"],
            brick_size=p["brick_size"],
            cell=p["pose_cell"],
            system="lego",
            brick_height=0.0032,
            grasp_depth=0.002,
            use_grasp_frame=False,
            check_gripper=False,
            occupied=done,
        )
        # Row by row, a part soon has a placed neighbour on one side along both axes: the pose
        # skill agrees with the assembly order's finger check.
        assert res["route"] in ("computed", "ungraspable"), res["detail"]
        routes.append(res["route"])
        assert sorted(map(list, res["cells"])) == sorted(p["cells"])
        done = [*done, p]
    assert routes[0] == "computed" and "ungraspable" in routes
    clash = pose_script.run(
        FakeContext(),
        plate_pose={"position": [0.4, 0.0, 0.0], "quaternion": [1.0, 0.0, 0.0, 0.0]},
        plate_studs=out["plate_studs"],
        brick_size="1x1",
        cell=out["placements"][0]["pose_cell"],
        system="lego",
        use_grasp_frame=False,
        check_gripper=False,
        occupied=out["placements"],
    )
    assert clash["route"] == "collides"
