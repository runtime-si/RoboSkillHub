"""proposing-open-hand-grasps-in-a-dense-pile: the open-hand fit on synthetic bins."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext

from ._bin_scenes import DOWN, Scene, obb

BUNDLE = "proposing-open-hand-grasps-in-a-dense-pile"
BIN = (0.0, 0.0, 0.40, 0.30, 0.0, 0.15)
PILE_Z = 0.05
GREY, BLACK = (128, 128, 128), (20, 20, 20)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["propose_pile_grasps"].module


def _scene(*blocks, pile=True):
    scene = Scene().open_box(*BIN, wall=0.004)
    if pile:  # a level bed of parts
        scene.block(0.0, 0.0, 0.39, 0.29, 0.0, PILE_Z, rgb=GREY)
    for b in blocks:
        scene.block(*b)
    return scene


def _frame(scene, cam=(0.0, 0.0, 0.50)):
    return scene.frame(cam_xyz=cam, focal=400.0, name="right_wrist")


def _run(script, scene, ctx=None, **kw):
    kw.setdefault("min_range", 0.0)
    kw.setdefault("frame", False)
    return script.run(ctx or FakeContext({}), obb(*BIN), camera_frame=_frame(scene), **kw)


BAR = (0.0, 0.0, 0.14, 0.03, 0.0, 0.09)  # long along x, 3 cm wide, 4 cm proud of the pile


def test_a_proud_bar_is_taken_across_its_narrow_side(script):
    out = _run(script, _scene(BAR))
    assert out["route"] == "proposed", out["detail"]
    assert abs(abs(out["close_heading_deg"]) - 90.0) <= 20.0  # the jaws close across the bar
    # the bar is between the open fingers (47 mm either side), not necessarily centred
    assert abs(out["position"]["y"]) + 0.015 < 0.047 and abs(out["position"]["x"]) < 0.08
    assert out["top_z"] == pytest.approx(0.09, abs=0.004)
    # the fingertips go 35 mm under the top (max_depth), which is still over the pile's level
    assert out["tip_z"] == pytest.approx(0.09 - 0.035, abs=0.004)
    assert out["grip_depth_m"] == pytest.approx(out["top_z"] - out["tip_z"])
    assert out["tool_z"] == pytest.approx(out["tip_z"] + 0.015)
    assert out["position"]["z"] == out["tool_z"] and out["jaw_width_m"] == pytest.approx(0.094)
    assert out["n_grasps"] == len(out["grasps"]) >= 1
    assert out["seen_fraction"] > 0.8


def test_along_the_bar_the_fingers_would_land_on_it(script):
    out = _run(script, _scene(BAR), headings_deg=[0.0])
    assert out["route"] == "nothing_fits", out["grasps"][:1]
    assert "94 mm" in out["detail"]


def test_proposals_keep_apart_and_every_depth_is_from_its_own_top(script):
    out = _run(script, _scene(BAR, (0.0, 0.09, 0.03, 0.03, 0.0, 0.08)))
    gs = out["grasps"]
    assert len(gs) >= 2
    for i, g in enumerate(gs):
        assert g["grip_depth_m"] >= 0.015 - 1e-6  # never negative, never under min_grip
        assert g["top_z"] - g["tip_z"] == pytest.approx(g["grip_depth_m"])
        for h in gs[:i]:
            assert (
                np.hypot(
                    g["position"]["x"] - h["position"]["x"], g["position"]["y"] - h["position"]["y"]
                )
                > 0.03
            )
    assert [g["score"] for g in gs] == sorted((g["score"] for g in gs), reverse=True)


def test_a_depth_spike_is_not_a_part(script):
    scene = _scene(BAR)
    frame = _frame(scene)
    frame["depth"][150:152, 200:202] = 0.25  # a reading 25 cm up, well clear of the bar
    spiked = script.run(FakeContext({}), obb(*BIN), camera_frame=frame, min_range=0.0, frame=False)
    assert spiked["route"] == "proposed"
    assert all(g["top_z"] < 0.10 for g in spiked["grasps"])
    # ... but the fingers still keep clear of it: no fingertip is planned under the spike's level
    pile = script.Pile(obb(*BIN))
    pile.add(frame)
    pile.close()
    assert np.nanmax(pile.z) > 0.2


def test_a_part_against_the_wall_has_no_room_for_the_outer_finger(script):
    by_wall = (0.0, 0.118, 0.14, 0.03, 0.0, 0.09)  # 17 mm from the +y wall's inner face
    out = _run(script, _scene(by_wall))
    assert out["route"] == "nothing_fits"


def test_a_level_pile_and_a_bare_floor_offer_nothing(script):
    assert _run(script, _scene())["route"] == "nothing_fits"
    # something low on the bare floor: under FLOOR_CLEAR it is the floor
    flat = (0.0, 0.0, 0.14, 0.03, 0.0, 0.010)
    assert _run(script, _scene(flat, pile=False), min_grip=0.004)["route"] == "nothing_fits"


def test_avoid_and_allow_masks_choose_among_parts(script):
    left, right = (-0.10, 0.0, 0.03, 0.12, 0.0, 0.09), (0.10, 0.0, 0.03, 0.12, 0.0, 0.09, BLACK)
    scene = _scene(left, right)
    frame = _frame(scene)
    both = _run(script, scene)
    assert {round(g["position"]["x"], 1) for g in both["grasps"]} == {-0.1, 0.1}
    dark = (frame["rgb"].mean(-1) < 60).astype(np.uint8) * 255
    # avoid: no hand centre on the dark part or within 8 mm of it (it may still be between the
    # fingers beside the centre: the mask is tested round the centre only) ...
    avoided = _run(script, scene, avoid_mask=dark)
    assert avoided["route"] == "proposed"
    for g in avoided["grasps"]:
        p = g["position"]
        assert abs(p["x"] - 0.10) > 0.015 + 0.008 or abs(p["y"]) > 0.06 + 0.008
    assert any(g["position"]["x"] < -0.05 for g in avoided["grasps"])
    # ... allow: every hand centre on it
    allowed = _run(script, scene, allow_mask=dark)
    assert allowed["route"] == "proposed"
    assert all(abs(g["position"]["x"] - 0.10) < 0.016 for g in allowed["grasps"])
    with pytest.raises(ValueError, match="mask"):
        _run(script, scene, allow_mask=np.ones((3, 3), bool))


def test_a_region_limits_where_grasps_may_be(script):
    left, right = (-0.10, 0.0, 0.03, 0.12, 0.0, 0.09), (0.10, 0.0, 0.03, 0.12, 0.0, 0.09)
    out = _run(script, _scene(left, right), roi={"x_max": 0.0})
    assert out["route"] == "proposed" and all(g["position"]["x"] <= 0.0 for g in out["grasps"])


def test_depth_rank_puts_the_deepest_grip_first(script):
    shallow, deep = (-0.10, 0.0, 0.03, 0.12, 0.0, 0.07), (0.10, 0.0, 0.03, 0.12, 0.0, 0.085)
    out = _run(script, _scene(shallow, deep), rank="depth", prefer_high=0.0)
    assert out["position"]["x"] > 0.05 and out["grip_depth_m"] > 0.03
    with pytest.raises(ValueError, match="rank"):
        _run(script, _scene(deep), rank="best")


def test_not_seen_without_depth_or_camera(script):
    frame = _frame(_scene(BAR))
    frame["depth"][:] = 0.0
    out = script.run(FakeContext({}), obb(*BIN), camera_frame=frame, frame=False)
    assert out["route"] == "not_seen" and "depth" in out["detail"]
    blind = script.run(FakeContext({"robot.get_observation": {"cameras": []}}), obb(*BIN))
    assert blind["route"] == "not_seen" and "camera" in blind["detail"]


def test_two_looks_are_merged_into_one_map(script):
    scene = _scene(BAR)
    a, b = _frame(scene), _frame(scene)
    a["depth"][:, 320:] = 0.0
    b["depth"][:, :320] = 0.0
    half = script.run(FakeContext({}), obb(*BIN), camera_frame=a, min_range=0.0, frame=False)
    both = script.run(FakeContext({}), obb(*BIN), camera_frames=[a, b], min_range=0.0, frame=False)
    assert half["seen_fraction"] < 0.6 < both["seen_fraction"]
    assert both["route"] == "proposed"


def test_the_hand_is_read_from_the_connector_and_the_frame_asked_for(script):
    scene = _scene(BAR)
    ctx = FakeContext(
        {
            "robot.get_observation": {"cameras": [dict(_frame(scene), name="top"), _frame(scene)]},
            "robot.describe_gripper": {
                "max_grasp_width_m": 0.07,
                "finger": {"reach_m": 0.0102, "stated": True},
            },
            "robot.grasp_frame": {"rotation": DOWN},
        }
    )
    out = script.run(ctx, obb(*BIN), min_range=0.0, arm_id=1)
    assert out["route"] == "proposed" and out["camera"] == "right_wrist"
    assert out["jaw_width_m"] == pytest.approx(0.07)
    assert out["tool_z"] == pytest.approx(out["tip_z"] + 0.0102)
    (call,) = ctx.calls_to("robot.grasp_frame")
    assert call.kwargs == {"close_heading_deg": out["close_heading_deg"], "arm_id": 1}
    assert out["grasp_pose"] == {"position": out["position"], "rotation": DOWN}
    assert ctx.calls_to("robot.describe_gripper")[0].kwargs == {"arm_id": 1}


def test_the_bin_may_be_turned_and_raised(script):
    turned = (0.05, -0.02, 0.40, 0.30, 35.0, 0.25)
    scene = Scene().open_box(*turned, wall=0.004, floor=0.104)
    scene.block(0.05, -0.02, 0.39, 0.29, 35.0, 0.15, rgb=GREY)
    scene.block(0.05, -0.02, 0.14, 0.03, 35.0, 0.19, rgb=GREY)
    out = script.run(
        FakeContext({}),
        obb(*turned, support_z=0.10),
        camera_frame=scene.frame(cam_xyz=(0.05, -0.02, 0.62), focal=400.0),
        min_range=0.0,
        frame=False,
    )
    assert out["route"] == "proposed", out["detail"]
    # any heading whose fingers clear the 3 cm bar ties; it is at least 30 deg off the bar's axis
    off_axis = abs((out["close_heading_deg"] - 35.0 + 90.0) % 180.0 - 90.0)
    assert off_axis >= 30.0
    assert out["top_z"] == pytest.approx(0.19, abs=0.004)
