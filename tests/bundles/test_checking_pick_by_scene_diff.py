"""checking-pick-by-scene-diff: FakeContext tests of the panel, the prompt, the region numbers and the routing."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "checking-pick-by-scene-diff"
EXPECTED = "the red brick was picked up: it is no longer at its spot on the table"
H, W = 60, 80
BOX = [30, 20, 50, 40]  # x0, y0, x1, y1: the source region
TABLE = (180, 140, 100)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["check_scene_diff"].module


def _scene(brick=True, hand=False, other=False):
    rgb = np.empty((H, W, 3), dtype=np.uint8)
    rgb[:] = TABLE
    rgb[::4, :] = (170, 132, 95)  # wood grain, so the ring is not flat
    if brick:
        rgb[25:35, 35:45] = (200, 30, 30)
    if other:
        rgb[22:38, 32:48] = (30, 30, 200)
    if hand:
        rgb[10:50, 25:55] = (20, 20, 20)
        rgb[20:40, 35:45] = (230, 230, 230)
    return rgb


def _obs(rgb, name="top", extra=None):
    cams = [{"name": name, "rgb": rgb}]
    if extra:
        cams.append(extra)
    return {"cameras": cams}


class Vlm:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return {"text": self.reply}


def _run(script, reply, before, now, **kw):
    vlm = Vlm(reply)
    ctx = FakeContext({"vlm.query": vlm, "robot.get_observation": now})
    out = script.run(ctx, before=before, expected_change=EXPECTED, **kw)
    return out, vlm


YES_YES = "CHANGED: YES\nMATCH: YES\nThe brick is gone from its spot."
YES_NO = "CHANGED: YES\nMATCH: NO\nThe hand moved but the brick is still there."
NO_NO = "CHANGED: NO\nMATCH: NO\nNothing changed."
NO_YES = "CHANGED: NO\nMATCH: YES\nOdd."


def test_panel_is_before_after_heat_side_by_side_and_prompt_is_upstreams(script):
    before, now = _scene(), _scene(brick=False)
    out, vlm = _run(script, YES_YES, _obs(before), _obs(now))
    (call,) = vlm.calls
    (panel,) = call["images"]
    assert panel.shape == (H, 3 * W, 3) and panel.dtype == np.uint8
    assert np.array_equal(panel[30:, :W], before[30:])  # below the label
    assert np.array_equal(panel[30:, W : 2 * W], now[30:])
    heat = panel[30:, 2 * W :]
    assert heat[0:5, 36:44].max() > 200  # brick pixels changed -> hot
    assert heat[-5:, :10].max() == 0  # untouched table -> black
    assert "BEFORE | AFTER | DIFF heatmap" in call["prompt"]
    assert f"Expected: {EXPECTED!r}" in call["prompt"]
    assert "CHANGED: YES or NO" in call["prompt"] and "MATCH: YES or NO" in call["prompt"]
    assert out["route"] == "changed_as_expected"
    assert out["reason"] == "The brick is gone from its spot."
    assert out["camera"] == "top"


def test_pick_with_region_that_emptied_is_as_expected(script):
    out, _ = _run(script, YES_YES, _obs(_scene()), _obs(_scene(brick=False)), source_box_px=BOX)
    assert out["route"] == "changed_as_expected"
    assert out["source_changed_fraction"] == pytest.approx(0.25)
    assert out["source_contrast_ratio"] < 0.85


def test_vlm_yes_while_hand_covers_the_source_is_uncertain(script):
    out, _ = _run(script, YES_YES, _obs(_scene()), _obs(_scene(hand=True)), source_box_px=BOX)
    assert out["route"] == "uncertain"
    assert out["source_contrast_ratio"] > 0.85
    assert "did not empty" in out["reason"]


def test_vlm_yes_when_another_object_arrived_is_uncertain(script):
    out, _ = _run(
        script,
        YES_YES,
        _obs(_scene(brick=False)),
        _obs(_scene(brick=False, other=True)),
        source_box_px=BOX,
    )
    assert out["route"] == "uncertain"


def test_vlm_yes_but_region_unchanged_is_uncertain(script):
    out, _ = _run(script, YES_YES, _obs(_scene()), _obs(_scene()), source_box_px=BOX)
    assert out["route"] == "uncertain"
    assert out["source_changed_fraction"] == 0.0
    assert "only 0.0% of the region changed" in out["reason"]


def test_place_mode_needs_the_region_to_fill(script):
    empty, placed = _scene(brick=False), _scene()
    out, _ = _run(
        script, YES_YES, _obs(empty), _obs(placed), source_box_px=BOX, region_expect="filled"
    )
    assert out["route"] == "changed_as_expected" and out["source_contrast_ratio"] >= 1.5
    out, _ = _run(
        script, YES_YES, _obs(placed), _obs(empty), source_box_px=BOX, region_expect="filled"
    )
    assert out["route"] == "uncertain"
    out, _ = _run(
        script, YES_YES, _obs(placed), _obs(empty), source_box_px=BOX, region_expect="changed"
    )
    assert out["route"] == "changed_as_expected"


def test_source_mask_works_like_the_box(script):
    mask = np.zeros((H, W), dtype=bool)
    mask[20:40, 30:50] = True
    out, _ = _run(script, YES_YES, _obs(_scene()), _obs(_scene(brick=False)), source_mask_px=mask)
    assert out["route"] == "changed_as_expected"
    assert out["source_changed_fraction"] == pytest.approx(0.25)


def test_changed_but_not_matching_is_changed_otherwise(script):
    out, _ = _run(script, YES_NO, _obs(_scene()), _obs(_scene(hand=True)), source_box_px=BOX)
    assert out["route"] == "changed_otherwise"
    assert out["changed"] and not out["matches_expectation"]


def test_nothing_changed_is_unchanged(script):
    out, _ = _run(script, NO_NO, _obs(_scene()), _obs(_scene()), source_box_px=BOX)
    assert out["route"] == "unchanged"
    assert out["changed_pixels"] == 0


def test_vlm_says_unchanged_while_region_emptied_is_uncertain(script):
    out, _ = _run(script, NO_NO, _obs(_scene()), _obs(_scene(brick=False)), source_box_px=BOX)
    assert out["route"] == "uncertain"
    assert "VLM said nothing changed" in out["reason"]


def test_match_without_change_is_uncertain(script):
    out, _ = _run(script, NO_YES, _obs(_scene()), _obs(_scene(brick=False)))
    assert out["route"] == "uncertain"


def test_without_region_the_vlm_decides(script):
    out, _ = _run(script, YES_YES, _obs(_scene()), _obs(_scene(hand=True)))
    assert out["route"] == "changed_as_expected"
    assert out["source_changed_fraction"] == -1.0 and out["source_contrast_ratio"] == -1.0


@pytest.mark.parametrize(
    "reply",
    [
        "**CHANGED:** YES\n**MATCH:** YES\nreason",
        "changed: yes\nmatch: yes",
        "CHANGED : YES  MATCH : YES",
    ],
)
def test_parser_tolerates_markdown_and_case(script, reply):
    out, _ = _run(script, reply, _obs(_scene()), _obs(_scene(brick=False)))
    assert out["route"] == "changed_as_expected"


@pytest.mark.parametrize("reply", ["CHANGED: YES\nThe brick moved.", "I cannot tell.", ""])
def test_missing_answer_line_is_no_verdict(script, reply):
    out, _ = _run(script, reply, _obs(_scene()), _obs(_scene(brick=False)))
    assert out["route"] == "no_verdict"
    assert out["raw"] == reply.strip()


def test_vlm_failure_is_no_verdict_with_the_pixel_numbers(script):
    out, _ = _run(
        script, ToolError("no key"), _obs(_scene()), _obs(_scene(brick=False)), source_box_px=BOX
    )
    assert out["route"] == "no_verdict"
    assert "vlm.query failed" in out["reason"]
    assert out["source_changed_fraction"] == pytest.approx(0.25)


def test_no_shared_camera_is_no_verdict_without_asking(script):
    out, vlm = _run(script, YES_YES, _obs(_scene(), name="top"), _obs(_scene(), name="wrist"))
    assert out["route"] == "no_verdict" and not vlm.calls


def test_size_mismatch_is_no_verdict(script):
    small = np.zeros((10, 10, 3), dtype=np.uint8)
    out, vlm = _run(script, YES_YES, _obs(_scene()), _obs(small))
    assert out["route"] == "no_verdict" and "sizes differ" in out["reason"] and not vlm.calls


def test_external_camera_is_preferred_and_camera_can_be_named(script):
    wrist_b = {"name": "left_wrist", "rgb": np.zeros((H, W, 3), np.uint8)}
    wrist_n = {"name": "left_wrist", "rgb": np.full((H, W, 3), 255, np.uint8)}
    before = {"cameras": [wrist_b, {"name": "top", "rgb": _scene()}]}
    now = {"cameras": [wrist_n, {"name": "top", "rgb": _scene(brick=False)}]}
    out, _ = _run(script, YES_YES, before, now)
    assert out["camera"] == "top"
    out, _ = _run(script, YES_YES, before, now, camera="left_wrist")
    assert out["camera"] == "left_wrist" and out["changed_fraction"] == 1.0


def test_mapping_cameras_and_float_rgb_are_accepted(script):
    before = {"cameras": {"top": {"rgb": _scene().astype(np.float32) / 255.0}}}
    now = {"cameras": {"top": {"rgb": _scene(brick=False)}}}
    out, _ = _run(script, YES_YES, before, now, source_box_px=BOX)
    assert out["route"] == "changed_as_expected"


def test_now_defaults_to_a_fresh_observation_and_model_is_passed(script):
    vlm = Vlm(YES_YES)
    calls = []

    def observe(**kw):
        calls.append(kw)
        return _obs(_scene(brick=False))

    ctx = FakeContext({"vlm.query": vlm, "robot.get_observation": observe})
    out = script.run(ctx, before=_obs(_scene()), expected_change=EXPECTED, model="m1")
    assert out["route"] == "changed_as_expected" and len(calls) == 1
    assert vlm.calls[0]["model"] == "m1"


def test_box_is_clipped_and_bad_region_expect_raises(script):
    out, _ = _run(
        script,
        YES_YES,
        _obs(_scene()),
        _obs(_scene(brick=False)),
        source_box_px=[50, 40, -100, -100],
    )
    assert out["source_changed_fraction"] >= 0.0
    with pytest.raises(ValueError):
        _run(script, YES_YES, _obs(_scene()), _obs(_scene()), region_expect="moved")
