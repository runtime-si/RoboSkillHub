"""colour bundle: colour maths, palette classification, white balance, dominant colours.

The bundle is out-of-process (``gap.serving``); the tests import ``_impl`` and
``tools`` through the synthetic ``gap_skills.tools.colour`` package the way
the bundle itself does. Scenes are synthetic: LDraw colours rendered in linear
light under an illuminant cast, an exposure, shading, shadow and specular
pixels and sensor noise.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

SKILLS_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = SKILLS_ROOT / "tools" / "colour"

WARM = np.array([1.25, 1.0, 0.70])  # ~3200 K tungsten against a D65-balanced camera
COOL = np.array([0.82, 1.0, 1.28])
HOT = np.array([1.4, 1.1, 0.55])  # ~2700 K incandescent


@pytest.fixture(scope="module")
def impl(skills_registry):
    assert skills_registry.get("colour").kind == "tool"
    return importlib.import_module("gap_skills.tools.colour._impl")


@pytest.fixture(scope="module")
def tools(impl):
    return importlib.import_module("gap_skills.tools.colour.tools")


def _rgb(impl, name, palette="lego"):
    return next(e["rgb"] for e in impl.PALETTES[palette] if e["name"] == name)


def _patch(impl, rgb, gains=(1.0, 1.0, 1.0), size=32, spec=0.0, shadow=0.0, seed=0):
    """A size x size uint8 patch of sRGB colour ``rgb`` lit by linear ``gains``."""
    rng = np.random.default_rng(seed)
    lin = impl.srgb_to_linear(np.asarray(rgb, dtype=float) / 255.0) * np.asarray(gains)
    p = np.broadcast_to(lin, (size, size, 3)).copy()
    p *= np.linspace(0.8, 1.0, size)[None, :, None]
    p[rng.random((size, size)) < shadow] *= 0.35
    p[rng.random((size, size)) < spec] += 0.6
    out = impl.linear_to_srgb(p) * 255.0 + rng.normal(0.0, 2.0, p.shape)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _scene(impl, colours, gains, ref_rgb=(244, 244, 244), **kw):
    """Patches side by side plus a reference patch last; returns image and masks."""
    size = 32
    tiles = [_patch(impl, c, gains, size, seed=i, **kw) for i, c in enumerate(colours)]
    tiles.append(_patch(impl, ref_rgb, gains, size, seed=99))
    img = np.concatenate(tiles, axis=1)
    masks = []
    for i in range(len(tiles)):
        m = np.zeros(img.shape[:2], np.uint8)
        m[:, i * size : (i + 1) * size] = 255
        masks.append(m)
    return img, masks[:-1], masks[-1]


# ---------------------------------------------------------------------------
# Bundle hygiene
# ---------------------------------------------------------------------------


def test_tools_module_does_not_import_the_implementation_at_load():
    tree = ast.parse((BUNDLE / "tools.py").read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module)
        elif isinstance(node, ast.Import):
            top.update(a.name for a in node.names)
    assert not any("colour" in name for name in top)


def test_tool_defaults_mirror_the_implementation(impl, tools):
    assert tools._AMBIGUOUS_MARGIN == impl.AMBIGUOUS_MARGIN
    assert tools._UNKNOWN_DELTA_E == impl.UNKNOWN_DELTA_E
    assert tools._ERODE_PX == impl.ERODE_PX


def test_all_four_tools_register(tool_registry):
    for name in ("classify", "measure", "white_balance", "dominant_colours"):
        assert tool_registry.get(f"colour.{name}") is not None


# ---------------------------------------------------------------------------
# Colour maths
# ---------------------------------------------------------------------------


def test_srgb_to_lab_matches_scikit_image(impl):
    color = pytest.importorskip("skimage.color")
    x = np.random.default_rng(0).random((500, 3))
    np.testing.assert_allclose(impl.srgb_to_lab(x), color.rgb2lab(x), atol=1e-3)
    np.testing.assert_allclose(impl.lab_to_srgb(impl.srgb_to_lab(x)) / 255.0, x, atol=1e-9)


@pytest.mark.parametrize(
    "lab1, lab2, expected",
    [  # Sharma, Wu, Dalal (2005), Table 1, pairs 1, 7, 17, 25
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
        ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ],
)
def test_ciede2000_reproduces_sharma_test_data(impl, lab1, lab2, expected):
    assert impl.deltaE_ciede2000(np.array(lab1), np.array(lab2)) == pytest.approx(
        expected, abs=1e-4
    )


def test_ciede2000_broadcasts_one_colour_against_a_palette(impl):
    _, labs = impl.resolve_palette("lego")
    d = impl.deltaE_ciede2000(labs[0][None], labs)
    assert d.shape == (len(labs),) and d[0] == 0.0 and (d[1:] > 5).all()


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------


def test_lego_palette_carries_ldraw_values(impl):
    by_name = {e["name"]: e for e in impl.PALETTES["lego"]}
    assert by_name["black"]["ldraw"] == 0 and by_name["black"]["rgb"] == [27.0, 42.0, 52.0]
    assert by_name["dark_bluish_grey"]["rgb"] == [100.0, 100.0, 100.0]
    assert by_name["tan"]["rgb"] == [215.0, 186.0, 140.0]
    assert by_name["yellow"]["rgb"] == [250.0, 200.0, 10.0]
    basic = {e["name"] for e in impl.PALETTES["lego_basic"]}
    assert basic == set(impl.LEGO_BASIC_NAMES) and basic <= set(by_name)


def test_custom_palettes_take_rgb_lab_hex_and_repeated_names(impl):
    names, labs = impl.resolve_palette(
        [
            {"name": "a", "rgb": [200, 30, 30]},
            {"name": "a", "hex": "#B40000"},
            {"name": "b", "lab": [50.0, 0.0, 0.0]},
        ]
    )
    assert names == ["a", "a", "b"] and labs.shape == (3, 3)
    with pytest.raises(ValueError):
        impl.resolve_palette("lego_technic")
    with pytest.raises(ValueError):
        impl.resolve_palette([{"name": "x"}])


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("palette", ["lego", "lego_basic", "domino"])
def test_every_palette_colour_under_neutral_light_classifies_as_itself(impl, palette):
    for entry in impl.PALETTES[palette]:
        img = _patch(impl, entry["rgb"])
        r = impl.classify(img, None, palette)
        assert r["name"] == entry["name"], (entry["name"], r["ranking"][:2])
        assert r["status"] == "classified" and r["margin_to_second"] >= impl.AMBIGUOUS_MARGIN


def test_specular_highlights_and_shadows_do_not_move_the_answer(impl):
    for name in ("tan", "dark_bluish_grey", "black", "red", "light_bluish_grey"):
        img = _patch(impl, _rgb(impl, name), spec=0.2, shadow=0.25)
        r = impl.classify(img, None, "lego")
        assert r["name"] == name and r["n_used"] < r["n_pixels"]


def test_underexposed_dark_grey_reads_black_until_white_balanced(impl):
    names = ["dark_bluish_grey", "black", "tan", "yellow"]
    img, masks, ref = _scene(impl, [_rgb(impl, n) for n in names], 0.35 * np.ones(3))
    raw = impl.classify(img, masks[0], "lego_basic")
    assert raw["name"] == "black"  # what an uncalibrated reading (or a VLM) says
    for name, m in zip(names, masks, strict=True):
        r = impl.classify(img, m, "lego_basic", white_ref=ref)
        assert r["name"] == name and r["status"] == "classified", (name, r["ranking"][:2])
        assert r["white_balanced"]


def test_tan_and_grey_under_incandescent_light_need_white_balance(impl):
    names = ["tan", "yellow", "light_bluish_grey"]
    img, masks, ref = _scene(impl, [_rgb(impl, n) for n in names], HOT)
    assert impl.classify(img, masks[0], "lego")["name"] != "tan"
    assert impl.classify(img, masks[2], "lego")["name"] in {"tan", "dark_tan"}  # grey reads tan
    for name, m in zip(names, masks, strict=True):
        r = impl.classify(img, m, "lego", white_ref=ref)
        assert r["name"] == name and r["status"] == "classified", (name, r["ranking"][:2])


def test_grey_reference_with_its_own_colour_corrects_a_cool_overexposed_scene(impl):
    grey = (150, 150, 150)
    names = ["white", "light_bluish_grey", "medium_azure", "tan"]
    img, masks, ref = _scene(impl, [_rgb(impl, n) for n in names], 1.4 * COOL, ref_rgb=grey)
    for name, m in zip(names, masks, strict=True):
        r = impl.classify(img, m, "lego", white_ref=ref, reference_rgb=list(grey))
        assert r["name"] == name, (name, r["ranking"][:2])


def test_a_colour_between_two_entries_is_ambiguous(impl):
    _, labs = impl.resolve_palette([{"name": "black", "rgb": _rgb(impl, "black")}])
    mid = 0.5 * (labs[0] + impl.srgb_to_lab(np.asarray(_rgb(impl, "dark_bluish_grey")) / 255.0))
    img = np.broadcast_to(impl.lab_to_srgb(mid), (20, 20, 3)).round().astype(np.uint8)
    r = impl.classify(img, None, "lego_basic")
    assert r["status"] == "ambiguous"
    assert {r["name"], r["second_name"]} == {"black", "dark_bluish_grey"}
    assert r["margin_to_second"] < impl.AMBIGUOUS_MARGIN


def test_a_colour_outside_the_palette_is_unknown(impl):
    img = np.full((20, 20, 3), (255, 0, 255), np.uint8)
    r = impl.classify(img, None, "lego_basic")
    assert r["status"] == "unknown_colour" and r["delta_e"] > impl.UNKNOWN_DELTA_E


def test_an_empty_mask_is_empty_and_a_wrong_shape_raises(impl):
    img = np.full((20, 20, 3), 128, np.uint8)
    r = impl.classify(img, np.zeros((20, 20), np.uint8))
    assert r["status"] == "empty" and r["name"] is None and r["n_pixels"] == 0
    with pytest.raises(ValueError):
        impl.classify(img, np.ones((10, 10), np.uint8))


def test_a_thin_mask_is_not_eroded_away(impl):
    img = np.full((60, 60, 3), (215, 186, 140), np.uint8)
    img[29:32, :] = (180, 0, 0)
    m = np.zeros((60, 60), np.uint8)
    m[29:32, 5:55] = 255
    r = impl.classify(img, m, "lego_basic")
    assert r["name"] == "red" and r["n_used"] >= impl.MIN_PIXELS


def test_white_ref_forms_agree(impl):
    img, masks, ref = _scene(impl, [_rgb(impl, "dark_bluish_grey")], 0.4 * WARM)
    wb = impl.white_balance(img, ref, "reference")
    by_mask = impl.classify(img, masks[0], white_ref=ref)
    by_dict = impl.classify(img, masks[0], white_ref=wb)
    by_gains = impl.classify(img, masks[0], white_ref=wb["gains"])
    assert by_mask["lab"] == pytest.approx(by_dict["lab"])
    assert by_dict["lab"] == pytest.approx(by_gains["lab"])
    assert impl.classify(img, masks[0], white_ref="grey_world")["white_balanced"]
    with pytest.raises(ValueError):
        impl.classify(img, masks[0], white_ref="max_rgb")


def test_a_learned_palette_beats_ldraw_values_for_a_camera_with_its_own_rendering(impl):
    # A camera that renders everything desaturated and dark: LDraw's values miss,
    # one labelled sample per colour measured with the same camera does not.
    def camera(rgb):
        lab = impl.srgb_to_lab(np.asarray(rgb, float) / 255.0)
        return impl.lab_to_srgb(lab * np.array([0.8, 0.35, 0.35]))

    names = ["tan", "yellow", "black", "dark_bluish_grey", "light_bluish_grey", "white"]
    learned = []
    for i, n in enumerate(names):
        img = _patch(impl, camera(_rgb(impl, n)), seed=10 + i)
        learned.append({"name": n, "lab": impl.measure(img)["lab"]})
    ldraw_right = learned_right = 0
    for i, n in enumerate(names):
        img = _patch(impl, camera(_rgb(impl, n)), seed=20 + i)
        ldraw_right += impl.classify(img, None, "lego")["name"] == n
        learned_right += impl.classify(img, None, learned)["name"] == n
    assert learned_right == len(names) and ldraw_right < len(names)


# ---------------------------------------------------------------------------
# white_balance
# ---------------------------------------------------------------------------


def test_grey_world_neutralises_a_cast_and_keeps_luminance(impl):
    img = _patch(impl, (128, 128, 128), WARM, size=40)
    wb = impl.white_balance(img, method="grey_world", apply=True)
    assert wb["gains"][2] > wb["gains"][1] > wb["gains"][0]
    bal = wb["image"].reshape(-1, 3).astype(float).mean(axis=0)
    assert np.ptp(bal) < 4.0
    assert wb["clipped_fraction"] == 0.0


def test_reference_fixes_exposure_unless_told_not_to(impl):
    img = _patch(impl, (244, 244, 244), 0.5 * np.ones(3))
    full = impl.white_balance(img, np.ones(img.shape[:2]), "reference")
    chroma = impl.white_balance(img, np.ones(img.shape[:2]), "reference", exposure=False)
    assert np.mean(full["gains"]) == pytest.approx(2.0 / 0.9, rel=0.1)  # 0.5 light, shading 0.8-1
    assert np.mean(chroma["gains"]) == pytest.approx(1.0, abs=0.05)
    with pytest.raises(ValueError):
        impl.white_balance(img, None, "reference")


def test_a_saturated_reference_reports_its_clipping(impl):
    img = _patch(impl, (244, 244, 244), 1.6 * np.ones(3))
    wb = impl.white_balance(img, np.ones(img.shape[:2]), "reference")
    assert wb["clipped_fraction"] > 0.5


# ---------------------------------------------------------------------------
# dominant_colours
# ---------------------------------------------------------------------------


def test_dominant_colours_splits_a_two_colour_part_and_merges_its_shading(impl):
    img = np.concatenate(
        [
            _patch(impl, _rgb(impl, "red"), size=40)[:, :28],
            _patch(impl, _rgb(impl, "white"), size=40)[:, :12],
        ],
        axis=1,
    )
    r = impl.dominant_colours(img, None, palette="lego_basic", return_labels=True)
    assert [c["name"] for c in r["colours"]] == ["red", "white"]
    assert r["colours"][0]["fraction"] == pytest.approx(0.7, abs=0.03)
    assert r["labels"].shape == img.shape[:2] and set(np.unique(r["labels"])) <= {-1, 0, 1}
    exact = impl.dominant_colours(img, None, k=2)
    assert len(exact["colours"]) == 2 and "name" not in exact["colours"][0]


def test_dominant_colours_of_an_empty_mask(impl):
    r = impl.dominant_colours(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10)))
    assert r["found"] is False and r["colours"] == []


# ---------------------------------------------------------------------------
# The @tool boundary
# ---------------------------------------------------------------------------


def test_tools_wrap_the_implementation(impl, tools):
    img, masks, ref = _scene(impl, [_rgb(impl, "dark_bluish_grey")], 0.4 * np.ones(3))
    wb = tools.white_balance(img, ref, method="reference")
    r = tools.classify(img, masks[0], palette="lego_basic", white_ref=wb)
    assert r["name"] == "dark_bluish_grey" and r["status"] == "classified"
    m = tools.measure(img, masks[0], white_ref=wb)
    assert m["found"] and m["lab"] == pytest.approx(r["lab"])
    d = tools.dominant_colours(img, masks[0], palette="lego_basic", white_ref=wb)
    assert d["colours"][0]["name"] == "dark_bluish_grey"
    with pytest.raises(ValueError):
        tools.classify(None)
