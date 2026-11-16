"""cross-checking-a-placed-count-from-overhead: the estimate's known errors and the half rule."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext

from ._bin_scenes import Scene, obb

BUNDLE = "cross-checking-a-placed-count-from-overhead"
BOX = (0.40, -0.07, 0.24, 0.20, 0.0, 0.066)
CARD, GREY = (170, 120, 60), (110, 110, 110)
PART = (0.06, 0.06)  # 36 cm2 from above
AREA = PART[0] * PART[1]


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["cross_check_count"].module


def _frame(parts=(), extra=None):
    scene = Scene().open_box(*BOX, wall=0.004, rgb=CARD)
    for x, y, top in parts:
        scene.block(BOX[0] + x, BOX[1] + y, *PART, 0.0, top, rgb=GREY)
    if extra:
        extra(scene)
    return scene.frame(cam_xyz=(BOX[0], BOX[1], 0.9), focal=600.0)


def _run(script, frame, claimed=-1, **kw):
    return script.run(FakeContext({}), obb(*BOX), AREA, claimed, camera_frame=frame, **kw)


APART = [(-0.07, -0.05, 0.03), (0.07, -0.05, 0.03), (-0.07, 0.05, 0.03), (0.07, 0.05, 0.03)]


def test_parts_lying_apart_are_counted_and_the_claim_supported(script):
    out = _run(script, _frame(APART), claimed=4)
    assert out["route"] == "supported", out["detail"]
    assert out["estimate"] == 4 and out["blobs"] == 4
    assert out["area_m2"] == pytest.approx(4 * AREA, rel=0.15)
    assert out["seen_fraction"] > 0.9


def test_stacked_parts_read_low_and_are_still_let_through(script):
    # four parts let go over one point: two stacks of two
    stacked = [(-0.05, 0.0, 0.03), (-0.05, 0.0, 0.05), (0.05, 0.0, 0.03), (0.05, 0.0, 0.05)]
    out = _run(script, _frame(stacked), claimed=4)
    assert out["estimate"] == 2  # wrong by two: the method cannot see a stack
    assert out["route"] == "supported"  # half of the claim is seen


def test_touching_parts_are_one_blob_and_the_area_decides(script):
    row = [(-0.06, 0.0, 0.03), (0.0, 0.0, 0.03), (0.06, 0.0, 0.03)]
    out = _run(script, _frame(row), claimed=3)
    assert out["blobs"] == 1 and out["by_area"] in (3, 4) and out["estimate"] == out["by_area"]


def test_an_empty_box_contradicts_a_claim_of_four(script):
    # every part went down beside the box
    def beside(scene):
        for k in range(4):
            scene.block(BOX[0] - 0.09 + 0.06 * k, BOX[1] - 0.15, *PART, 0.0, 0.03, rgb=GREY)

    out = _run(script, _frame(extra=beside), claimed=4)
    assert out["route"] == "contradicted", out["detail"]
    assert out["estimate"] == 0 and "against 4 claimed" in out["detail"]


def test_one_seen_of_four_claimed_is_contradicted_two_is_not(script):
    assert _run(script, _frame(APART[:1]), claimed=4)["route"] == "contradicted"
    assert _run(script, _frame(APART[:2]), claimed=4)["route"] == "supported"


def test_without_a_claim_nothing_is_decided(script):
    out = _run(script, _frame(APART))
    assert out["route"] == "uncertain" and out["estimate"] == 4
    assert "no independent count" in out["detail"]


def test_a_zero_claim_is_supported_only_by_an_empty_box(script):
    assert _run(script, _frame(), claimed=0)["route"] == "supported"
    assert _run(script, _frame(APART[:1]), claimed=0)["route"] == "contradicted"


def test_the_verdict_rule(script):
    v = script.verdict
    assert [v(0, 4), v(1, 4), v(2, 4), v(5, 4), v(1, 1), v(0, 1), v(1, 2), v(1, 3)] == [
        "contradicted",
        "contradicted",
        "supported",
        "supported",
        "supported",
        "contradicted",
        "supported",
        "contradicted",
    ]
    assert v(3, -1) == "uncertain" and v(0, 0) == "supported" and v(2, 0) == "contradicted"


def test_something_over_the_box_makes_it_uncertain(script):
    def arm(scene):  # an arm across the box, well over its rim
        scene.block(BOX[0], BOX[1], 0.30, 0.16, 0.0, 0.30, rgb=(60, 60, 60))

    out = _run(script, _frame(APART, extra=arm), claimed=4)
    assert out["route"] == "uncertain" and out["estimate"] == -1
    assert "in view" in out["detail"]


def test_flat_dark_parts_need_the_colour_rule_and_a_shadow_is_not_one(script):
    flat = [(-0.06, 0.0, 0.010), (0.06, 0.0, 0.010)]  # 6 mm over the 4 mm floor: under FLOOR_CLEAR
    frame = _frame(flat)
    frame["rgb"][frame["rgb"][..., 0] == GREY[0]] = (60, 60, 60)
    assert _run(script, frame, claimed=2)["estimate"] == 0
    assert _run(script, frame, claimed=2, dark_parts=True)["estimate"] == 2
    # a shadow on the cardboard: dark, still brown
    shadow = _frame()
    h, w = shadow["rgb"].shape[:2]
    patch = shadow["rgb"][h // 2 - 25 : h // 2 + 25, w // 2 - 25 : w // 2 + 25]
    patch[:] = (patch * 0.45).astype(np.uint8)
    assert _run(script, shadow, claimed=0, dark_parts=True)["estimate"] == 0


def test_the_overhead_camera_is_found_and_a_missing_one_is_uncertain(script):
    frame = _frame(APART)
    obs = {"cameras": [dict(frame, name="right_wrist", depth=np.zeros_like(frame["depth"])), frame]}
    out = script.run(FakeContext({"robot.get_observation": obs}), obb(*BOX), AREA, 4)
    assert out["route"] == "supported" and out["camera"] == "top"
    none = script.run(FakeContext({"robot.get_observation": {"cameras": []}}), obb(*BOX), AREA, 4)
    assert none["route"] == "uncertain" and "camera" in none["detail"]
    with pytest.raises(ValueError, match="part_area"):
        script.run(FakeContext({}), obb(*BOX), 0.0, 4, camera_frame=frame)
