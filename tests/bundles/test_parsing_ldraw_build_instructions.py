"""parsing-ldraw-build-instructions: LDraw files to stud-grid steps.

Test files in tests/bundles/data/ldraw: the hand-written ones are ours; pyramid.ldr and car.ldr
are James Jessiman's sample models from LDraw.org's complete parts library, and 1869-1.mpd
("1869 - South African Flag", Merlijn Wissink) is from LDraw.org's Official Model Repository,
both redistributable under CCAL 2.0 (CC BY 2.0) with attribution to LDraw.org and the authors.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from gap.testing import FakeContext

BUNDLE = "parsing-ldraw-build-instructions"
DATA = Path(__file__).parent / "data" / "ldraw"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["parse_ldraw"].module


def parse(script, name=None, **kw):
    if name is not None:
        kw.setdefault("ldraw_path", str(DATA / name))
    return script.run(FakeContext(), **kw)


def cells(out):
    return [
        (b["size"], b["rotation"], b["cell"]["x"], b["cell"]["y"], b["cell"]["layer"])
        for b in out["bricks"]
    ]


def test_pyramid_sample_model(script):
    out = parse(script, "pyramid.ldr")
    assert out["route"] == "steps_ready"
    assert (out["n_bricks"], out["n_steps"]) == (13, 4)
    assert [len(s["bricks"]) for s in out["steps"]] == [6, 4, 2, 1]
    assert [s["bricks"][0]["colour"] for s in out["steps"]] == ["blue", "red", "yellow", "black"]
    # An 8x8 ring of 2x4s, then smaller rings, a 2x2 on top in the middle.
    first = out["steps"][0]["bricks"]
    xs = [b["cell"]["x"] + b["span_x"] for b in first] + [b["cell"]["x"] for b in first]
    ys = [b["cell"]["y"] + b["span_y"] for b in first] + [b["cell"]["y"] for b in first]
    assert (min(xs), max(xs), min(ys), max(ys)) == (0, 8, 0, 8)
    assert {b["cell"]["layer"] for b in first} == {1}  # standing on the ground (ICL z = 1)
    assert cells(out)[0] == ("2x4", 90, 0, 6, 1)  # 3001 at x=-40 z=60, unturned: 4 studs along x
    assert cells(out)[2] == ("2x4", 0, 6, 2, 1)  # the turned one: 2 studs along x
    top = out["steps"][3]["bricks"][0]
    assert (top["size"], top["cell"], top["part"]) == (
        "2x2",
        {"x": 3, "y": 3, "layer": 4, "rotation": 0},
        "3003",
    )
    assert all(b["colour_in_palette"] for b in out["bricks"])


def test_car_sample_reports_unsupported_parts(script):
    out = parse(script, "car.ldr")
    assert out["route"] == "unsupported_part"
    assert (out["n_bricks"], out["n_unsupported"], out["n_off_grid"]) == (33, 28, 0)
    parts = {u["part"] for u in out["unsupported"]}
    assert {"3641", "4624", "3829c01", "3823"} <= parts  # tyres, wheels, steering wheel, windscreen
    assert all(u["reason"].startswith("not a basic brick") for u in out["unsupported"])
    kinds = {b["kind"] for b in out["bricks"]}
    assert kinds == {"brick", "plate", "tile"}
    tile = next(b for b in out["bricks"] if b["part"] == "3068b")
    assert tile["studs_on_top"] is False and tile["height_units"] == 1
    plate = next(b for b in out["bricks"] if b["part"] == "3031")
    assert plate["cell"]["layer"] == 1 and plate["size"] == "4x4"
    lights = [b for b in out["bricks"] if b["colour"] == "trans_yellow"]
    assert len(lights) == 4 and not any(b["colour_in_palette"] for b in lights)
    # A plate is a third of a brick: the 1x1 plates on the 4x4 plate are one plate up.
    first_plate_layer = out["steps"][1]["bricks"][0]
    assert first_plate_layer["z_units"] == 1 and first_plate_layer["cell"][
        "layer"
    ] == pytest.approx(4 / 3, abs=1e-6)


def test_steps_comments_and_empty_steps(script):
    out = parse(script, "two_steps.ldr")
    assert out["route"] == "steps_ready"
    assert out["n_steps"] == 2  # the trailing empty STEP is dropped
    assert cells(out) == [("2x4", 90, 0, 0, 1), ("1x2", 0, 4, 0, 1), ("2x2", 0, 3, 0, 2)]
    plate = out["bricks"][1]
    assert (plate["kind"], plate["z_units"], plate["height_units"]) == ("plate", 0, 1)
    assert out["bricks"][2]["part"] == "3003"  # upper-case 3003.DAT
    assert [b["step"] for b in out["bricks"]] == [1, 1, 2]


def test_text_input_gives_the_same_result(script):
    text = (DATA / "two_steps.ldr").read_text()
    assert cells(parse(script, ldraw_text=text)) == cells(parse(script, "two_steps.ldr"))


def test_mpd_submodels_inline_on_a_baseplate(script):
    out = parse(script, "house.mpd")
    assert out["route"] == "steps_ready"
    assert out["baseplate"]["part"] == "3811" and out["baseplate"]["bricks_outside"] == []
    assert out["n_steps"] == 5
    assert [s["model"] for s in out["steps"]] == ["house.ldr > wall.ldr"] * 4 + ["house.ldr"]
    assert [s["source_step"] for s in out["steps"]] == [1, 2, 1, 2, 3]
    assert cells(out) == [
        ("1x4", 90, 11, 15, 1),
        ("1x4", 90, 11, 15, 2),
        ("1x1", 0, 11, 15, 3),
        ("1x4", 0, 16, 16, 1),  # the second wall, turned 90 deg: 4 studs along y
        ("1x4", 0, 16, 16, 2),
        ("1x1", 0, 16, 19, 3),
        ("4x4", 0, 13, 14, 3),
    ]
    # Colour 16 takes the placing line's colour; an explicit colour stays.
    assert [b["colour"] for b in out["bricks"]] == [
        "red", "red", "black", "blue", "blue", "black", "light_bluish_grey",
    ]  # fmt: skip


def test_mpd_submodels_flatten_and_min_origin(script):
    flat = parse(script, "house.mpd", submodels="flatten")
    assert flat["n_steps"] == 3 and [len(s["bricks"]) for s in flat["steps"]] == [3, 3, 1]
    assert {s["model"] for s in flat["steps"]} == {"house.ldr"}
    assert cells(flat) == cells(parse(script, "house.mpd"))
    low = parse(script, "house.mpd", origin="min")
    assert low["baseplate"] == {}
    assert [(c[2], c[3]) for c in cells(low)][:4] == [(0, 1), (0, 1), (0, 1), (5, 2)]
    assert cells(low)[-1] == ("4x4", 0, 2, 0, 3)


def test_ldraw_origin_counts_from_the_ldraw_ground(script):
    out = parse(script, "pyramid.ldr", origin="ldraw")
    assert cells(out)[0] == ("2x4", 90, -4, 2, 1)  # corner x = -40/20 - 2, z = 60/20 - 1
    assert out["steps"][3]["bricks"][0]["cell"] == {"x": -1, "y": -1, "layer": 4, "rotation": 0}


def test_external_and_missing_sub_files(script):
    out = parse(script, "outer.ldr")
    assert out["route"] == "unsupported_part"
    assert cells(out) == [("2x2", 0, 0, 0, 1), ("2x2", 0, 0, 0, 2)]
    assert [b["colour"] for b in out["bricks"]] == ["red", "blue"]
    (miss,) = out["unsupported"]
    assert miss["part"] == "missing.ldr" and miss["reason"] == "submodel file not found"
    # The same sub-file passed as text instead of from disk.
    text = (DATA / "outer.ldr").read_text()
    inner = (DATA / "inner.ldr").read_text()
    by_text = parse(script, ldraw_text=text, files={"inner.ldr": inner})
    assert cells(by_text) == cells(out)


def test_self_including_submodel_is_invalid(script):
    out = parse(script, "loop.mpd")
    assert out["route"] == "invalid_file"
    assert "includes itself" in out["detail"]


def test_lpub_buffer_exchange_and_rotstep(script):
    out = parse(script, "lpub.ldr")
    assert out["route"] == "steps_ready"
    assert cells(out) == [("2x4", 90, 0, 0, 1), ("2x2", 0, 1, 0, 2)]  # the buffered 2x4 is gone
    assert out["n_steps"] == 2  # ROTSTEP ended the first step


def test_off_grid_parts_are_reported_with_reasons(script):
    out = parse(script, "off_grid.ldr")
    assert out["route"] == "off_grid"
    assert out["n_bricks"] == 1 and out["n_unsupported"] == 1  # the 3039 slope
    reasons = [o["reason"] for o in out["off_grid"]]
    assert reasons[0] == "turned 45.0 deg about the vertical, not a multiple of 90"
    assert reasons[1] == "upside down"
    assert reasons[2] == "tilted off the vertical"
    assert "off the grid" in reasons[3]  # a half-stud shift
    assert "between plate heights" in reasons[4]
    assert len(reasons) == 5


def test_mirrored_and_scaled_matrices(script):
    base = "1 4 0 -24 0 1 0 0 0 1 0 0 0 1 3001.dat\n"
    out = parse(script, ldraw_text=base + "1 4 0 -48 0 -1 0 0 0 1 0 0 0 1 3003.dat\n")
    assert out["off_grid"][0]["reason"] == "mirrored matrix"
    out = parse(script, ldraw_text=base + "1 4 0 -48 0 2 0 0 0 1 0 0 0 1 3003.dat\n")
    assert out["off_grid"][0]["reason"] == "scaled or sheared matrix"


def test_duplo_on_a_duplo_baseplate(script):
    out = parse(script, "duplo.ldr")
    assert out["system"] == "duplo"
    assert out["route"] == "unsupported_part"
    assert out["unsupported"][0]["reason"] == "lego part in a duplo build"
    assert out["baseplate"]["part"] == "5930" and out["baseplate"]["plate_studs"] == [16, 16]
    assert cells(out) == [
        ("2x4", 90, 0, 1, 1),
        ("2x2", 0, 0, 3, 1),
        ("2x4", 0, 0, 1, 2),  # across both
        ("2x4", 90, 2, 1, 2),  # the plate, on the first 2x4
    ]
    assert [b["z_units"] for b in out["bricks"]] == [0, 0, 2, 2]
    assert [b["height_units"] for b in out["bricks"]] == [2, 2, 2, 1]  # a plate is half a brick
    forced = parse(script, "duplo.ldr", system="lego")
    assert forced["n_bricks"] == 1 and forced["n_unsupported"] == 4


def test_printed_variant_uses_the_base_part(script):
    text = (
        "1 15 0 -24 0 1 0 0 0 1 0 0 0 1 3001p01.dat\n1 4 0 -24 100 1 0 0 0 1 0 0 0 1 3002peye.dat\n"
    )
    out = parse(script, ldraw_text=text)
    (b,) = out["bricks"]
    assert (b["part"], b["base_part"], b["size"]) == ("3001p01", "3001", "2x4")
    assert out["unsupported"][0]["part"] == "3002peye"


def test_direct_and_unknown_colours(script):
    text = (
        "1 0x2FF8800 0 -24 0 1 0 0 0 1 0 0 0 1 3003.dat\n"
        "1 999 40 -24 0 1 0 0 0 1 0 0 0 1 3003.dat\n"
        "1 16 80 -24 0 1 0 0 0 1 0 0 0 1 3003.dat\n"
    )
    out = parse(script, ldraw_text=text)
    assert [b["colour"] for b in out["bricks"]] == ["#FF8800", "ldraw_999", "main_colour"]


def test_half_stud_shifted_model_still_converts(script):
    # Every brick 10 LDU off LDraw's grid, one brick on a "jumper" 10 LDU further.
    text = (
        "1 4 10 -24 0 1 0 0 0 1 0 0 0 1 3003.dat\n"
        "1 4 50 -24 0 1 0 0 0 1 0 0 0 1 3003.dat\n"
        "1 4 90 -24 0 1 0 0 0 1 0 0 0 1 3003.dat\n"
        "1 4 40 -48 0 1 0 0 0 1 0 0 0 1 3003.dat\n"
    )
    out = parse(script, ldraw_text=text)
    assert cells(out) == [("2x2", 0, 0, 0, 1), ("2x2", 0, 2, 0, 1), ("2x2", 0, 4, 0, 1)]
    assert len(out["off_grid"]) == 1 and "off the grid" in out["off_grid"][0]["reason"]


def test_brickgpt_ldr_round_trips(script):
    """BrickGPT's Brick.to_ldr lines (its two matrices, colour 115, upper-case ids) read back to its
    (h, w, x, y, z): h studs along x, w along y, z its layer."""
    bricks = [(2, 4, 0, 0, 0), (4, 2, 2, 0, 0), (1, 2, 5, 3, 1), (2, 2, 1, 1, 1), (1, 1, 0, 6, 2)]
    part = {(1, 1): "3005", (1, 2): "3004", (2, 2): "3003", (2, 4): "3001"}
    lines = []
    for h, w, x, y, z in bricks:
        ori = 1 if h > w else 0
        matrix = "0 0 1 0 1 0 -1 0 0" if ori == 0 else "-1 0 0 0 1 0 0 0 -1"
        cx, cz, cy = (x + h * 0.5) * 20, (y + w * 0.5) * 20, z * -24
        lines.append(f"1 115 {cx} {cy} {cz} {matrix} {part[tuple(sorted((h, w)))]}.DAT\n0 STEP\n")
    out = parse(script, ldraw_text="".join(lines), origin="ldraw")
    assert out["n_steps"] == len(bricks)
    got = [(b["span_x"], b["span_y"], b["cell"]["x"], b["cell"]["y"]) for b in out["bricks"]]
    assert got == [(h, w, x, y) for h, w, x, y, _ in bricks]
    assert [b["ori"] for b in out["bricks"]] == [1 if h > w else 0 for h, w, *_ in bricks]
    # BrickGPT puts a brick's top at -24 z, so its z = 0 sits one under LDraw's ground, and
    # layers count from 1 on the ground: layer = z.
    assert [b["cell"]["layer"] for b in out["bricks"]] == [z for *_, z in bricks]


def test_flag_model_from_the_omr(script):
    out = parse(script, "1869-1.mpd")
    assert out["route"] == "steps_ready"
    assert out["n_bricks"] == 82
    assert {b["size"] for b in out["bricks"]} == {
        "1x1", "1x2", "1x4", "1x6", "2x2", "2x3", "2x4", "2x6",
    }  # fmt: skip
    occupied = set()
    for b in out["bricks"]:
        for i in range(b["span_x"]):
            for j in range(b["span_y"]):
                key = (b["cell"]["x"] + i, b["cell"]["y"] + j, b["z_units"])
                assert key not in occupied  # no two bricks overlap
                occupied.add(key)


def test_bad_arguments_and_missing_file(script):
    assert parse(script)["route"] == "invalid_file"
    assert parse(script, ldraw_path=str(DATA / "nope.ldr"))["route"] == "invalid_file"
    assert parse(script, "pyramid.ldr", submodels="deep")["route"] == "invalid_file"
    assert parse(script, "pyramid.ldr", origin="centre")["route"] == "invalid_file"
    assert parse(script, "pyramid.ldr", origin="baseplate")["route"] == "invalid_file"
    assert parse(script, ldraw_text="0 just a comment\n")["route"] == "invalid_file"
    bad = parse(script, ldraw_text="1 4 0 0 0 1 0 0\n")
    assert bad["route"] == "unsupported_part" and "unreadable" in bad["unsupported"][0]["reason"]


def test_steps_json_matches_steps(script):
    import json

    out = parse(script, "pyramid.ldr")
    assert json.loads(out["steps_json"]) == out["steps"]


def test_colour_names_match_the_colour_tool(script):
    spec = importlib.util.spec_from_file_location("_colour_impl", ROOT / "tools/colour/_impl.py")
    impl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(impl)
    for name, code, _hex in impl.LEGO:
        assert script.LDRAW_COLOURS[code] == name
    assert {n for n, _, _ in impl.LEGO} == set(script.PALETTE_NAMES)


def test_example_declares_every_route(script):
    import json

    example = json.loads(
        (ROOT / "skills" / BUNDLE / "examples/canonical_subgraph.json").read_text()
    )
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == {"steps_ready", "unsupported_part", "off_grid", "invalid_file"}
