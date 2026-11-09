"""proposing-parallel-grasps-in-tight-clusters: finger-fit grasps on packed flat parts from depth."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

from ._cluster_scenes import brick_grid, jigsaw_cluster, rect, render

BUNDLE = "proposing-parallel-grasps-in-tight-clusters"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
TOP_A = {"w": 0.0, "x": 0.7071, "y": 0.7071, "z": 0.0}
TOP_B = {"w": 0.0, "x": -0.7071, "y": 0.7071, "z": 0.0}
HAND = {"max_grasp_width_m": 0.086, "span_m": 0.094, "finger": {"stated": True, "reach_m": 0.0102}}
LEGO = (0.0158, 0.0318)
FRAMES = Path.home() / "yam_gap"


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["propose_cluster_grasps"].module


def _ctx(hand=HAND, frame=True, obs=None):
    tools = {"robot.describe_gripper": hand}
    if frame:
        tools["robot.grasp_frame"] = {"rotation": TOP_A, "rotation_flipped": TOP_B}
    if obs is not None:
        tools["robot.get_observation"] = obs
    return FakeContext(tools)


def _heading_diff(a, b):
    return abs((a - b + 90.0) % 180.0 - 90.0)


def _jigsaw(gap, seed=0, **kw):
    return render(jigsaw_cluster(4, 4, gap, seed=seed), 0.002, **kw)


# --- single parts -----------------------------------------------------------------------------


def test_lone_domino_is_gripped_across_its_short_side(script):
    sc = render([rect(0.0, 0.0, 0.024, 0.048, yaw_deg=30.0)], 0.0075)
    ctx = _ctx()
    out = script.run(ctx, camera_frame=sc.frame, mask=sc.mask(0))
    assert out["route"] == "proposed"
    g = out["grasps"][0]
    assert _heading_diff(g["close_heading_deg"], 30.0) < 3.0  # across the 24 mm side
    assert g["object_width_m"] == pytest.approx(0.024, abs=0.0015)
    assert g["jaw_width_m"] == pytest.approx(g["contact_width_m"] + 0.004, abs=0.0011)
    assert g["closing_is_short_axis"]
    assert math.hypot(g["position"]["x"], g["position"]["y"]) < 0.002
    assert g["part_height_m"] == pytest.approx(0.0075, abs=0.0005)
    assert g["position"]["z"] == pytest.approx(0.00375, abs=0.0005)
    assert g["tip_z"] < 0.0015  # down at the table beside it
    assert g["tool_z"] == pytest.approx(g["tip_z"] + 0.0102)
    assert g["clearance_m"] == pytest.approx(0.006)  # nothing within the search window
    assert out["fraction_graspable"] == 1.0


def test_grasp_pose_comes_from_grasp_frame_at_the_tool_centre(script):
    sc = render([rect(0.01, -0.02, 0.0158, 0.0318)], 0.0096)
    ctx = _ctx()
    out = script.run(ctx, camera_frame=sc.frame, mask=sc.mask(0))
    (call,) = ctx.calls_to("robot.grasp_frame")
    assert call.kwargs == {"close_heading_deg": out["close_heading_deg"]}
    assert out["grasp_pose"]["rotation"] == TOP_A
    assert out["rotation_flipped"] == TOP_B
    assert out["grasp_pose"]["position"]["z"] == pytest.approx(out["tool_z"])
    assert out["grasp_pose"]["position"]["x"] == pytest.approx(out["position"]["x"])
    assert _heading_diff(out["close_heading_deg"], 0.0) < 3.0


def test_arm_id_is_passed_only_when_set(script):
    sc = render([rect(0.0, 0.0, 0.0158, 0.0318)], 0.0096)
    ctx = _ctx()
    script.run(ctx, camera_frame=sc.frame, mask=sc.mask(0), arm_id=1)
    assert ctx.calls_to("robot.grasp_frame")[0].kwargs["arm_id"] == 1
    assert ctx.calls_to("robot.describe_gripper")[0].kwargs == {"arm_id": 1}
    ctx = _ctx()
    script.run(ctx, camera_frame=sc.frame, mask=sc.mask(0), frame=False)
    assert ctx.calls_to("robot.describe_gripper")[0].kwargs == {}
    assert not ctx.calls_to("robot.grasp_frame")


def test_without_describe_gripper_the_station_defaults_hold(script):
    sc = render([rect(0.0, 0.0, 0.0158, 0.0318)], 0.0096)

    def missing(**kw):
        raise ToolError("robot.describe_gripper", "not served")

    ctx = FakeContext({"robot.describe_gripper": missing, "robot.grasp_frame": {"rotation": TOP_A}})
    out = script.run(ctx, camera_frame=sc.frame, mask=sc.mask(0))
    assert out["route"] == "proposed"
    assert out["tool_z"] == pytest.approx(out["tip_z"] + script.TIP_REACH)


def test_wider_than_the_hand_is_nothing_graspable(script):
    sc = render([rect(0.0, 0.0, 0.10, 0.10)], 0.004)
    out = script.run(_ctx(), camera_frame=sc.frame, mask=sc.mask(0))
    assert out["route"] == "nothing_graspable"
    assert out["pieces"][0]["status"] == "too_wide"
    # a hand that opens wider takes it
    out = script.run(_ctx(), camera_frame=sc.frame, mask=sc.mask(0), max_width=0.12)
    assert out["route"] == "proposed"


def test_a_part_too_thin_to_stand_out_is_nothing_graspable(script):
    sc = render([rect(0.0, 0.0, 0.02, 0.03)], 0.0008)
    out = script.run(_ctx(), camera_frame=sc.frame, mask=sc.mask(0))
    assert out["route"] == "nothing_graspable"
    assert out["pieces"][0]["status"] == "too_flat"
    assert not out["grasps"] and out["grasp_pose"] is None


def test_an_empty_table_is_nothing_graspable(script):
    sc = render([], 0.0)
    out = script.run(_ctx(), camera_frame=sc.frame)
    assert out["route"] == "nothing_graspable"
    assert out["n_pieces"] == 0


def test_a_mask_on_no_depth_is_nothing_graspable(script):
    sc = render([rect(0.0, 0.0, 0.02, 0.03)], 0.004)
    frame = dict(sc.frame, depth=np.where(sc.labels > 0, 0.0, sc.frame["depth"]).astype(np.float32))
    out = script.run(_ctx(), camera_frame=frame, mask=sc.mask(0))
    assert out["route"] == "nothing_graspable"
    assert "no pixel with depth" in out["reason"]


# --- neighbours -------------------------------------------------------------------------------


def test_a_close_neighbour_turns_the_jaws_to_the_free_side(script):
    # A brick with a twin 4 mm away on +x (no finger fits there) and free space along y.
    parts = [rect(0.0, 0.0, 0.0318, 0.0158), rect(0.0318 + 0.004, 0.0, 0.0318, 0.0158)]
    sc = render(parts, 0.0096)
    out = script.run(_ctx(), camera_frame=sc.frame, mask=sc.mask(0), piece_masks=[sc.mask(1)])
    assert out["route"] == "proposed"
    for g in out["grasps"]:
        assert _heading_diff(g["close_heading_deg"], 90.0) < 25.0
    alone = render(parts[:1], 0.0096)
    out_alone = script.run(_ctx(), camera_frame=alone.frame, mask=alone.mask(0))
    assert _heading_diff(out_alone["close_heading_deg"], 90.0) < 3.0


def test_a_neighbour_between_the_jaws_is_refused(script):
    # Two 12 mm wide strips 3 mm apart: the only finger fit across x straddles both.
    parts = [rect(0.0, 0.0, 0.012, 0.08), rect(0.015, 0.0, 0.012, 0.08)]
    sc = render(parts, 0.004)
    out = script.run(
        _ctx(), camera_frame=sc.frame, mask=sc.mask(0), piece_masks=[sc.mask(1)], rotations=2
    )
    # along y the 80 mm strip does not fit the hand; across x the twin is between the jaws
    assert out["route"] == "too_crowded"
    assert out["pieces"][0]["fails"]["neighbour_between"] > 0
    # with the twin unmasked it is still a neighbour (it stands over the support)
    out = script.run(_ctx(), camera_frame=sc.frame, mask=sc.mask(0), rotations=2)
    assert out["route"] == "too_crowded"


def test_tall_neighbour_blocks_the_finger_body_but_a_low_one_does_not(script):
    """A 4 mm tip fits a 7 mm gap; the 10 mm body above it must also clear the neighbour."""
    piece = rect(0.0, 0.0, 0.02, 0.09)  # longer than the hand opens: only across x
    for height, expect in ((0.002, "proposed"), (0.04, "too_crowded")):
        walls = [rect(-0.023, 0.0, 0.012, 0.09), rect(0.023, 0.0, 0.012, 0.09)]  # 7 mm gaps
        sc = render([piece, *walls], [0.002, height, height])
        out = script.run(
            _ctx(),
            camera_frame=sc.frame,
            mask=sc.mask(0),
            piece_masks=[sc.mask(1), sc.mask(2)],
            tip_thickness=0.004,
            rotations=2,
        )
        assert out["route"] == expect, (height, out["reason"])


def test_more_clearance_ranks_first(script):
    # Free on -x, a neighbour 13 mm away on +x: both headings fit, the roomier one wins.
    parts = [rect(0.0, 0.0, 0.02, 0.02), rect(0.033, 0.0, 0.02, 0.06)]
    sc = render(parts, 0.003)
    out = script.run(_ctx(), camera_frame=sc.frame, mask=sc.mask(0), piece_masks=[sc.mask(1)])
    assert out["route"] == "proposed"
    best = out["grasps"][0]
    assert _heading_diff(best["close_heading_deg"], 90.0) < 3.0
    across = [g for g in out["grasps"] if _heading_diff(g["close_heading_deg"], 0.0) < 3.0]
    assert across and across[0]["clearance_m"] < best["clearance_m"]
    assert [g["score"] for g in out["grasps"]] == sorted(
        (g["score"] for g in out["grasps"]), reverse=True
    )


def test_a_corner_is_not_a_contact(script):
    """Jaws turned 45 degrees on a square meet two corners: refused, the square's axes win."""
    sc = render([rect(0.0, 0.0, 0.02, 0.02)], 0.003)
    out = script.run(_ctx(), camera_frame=sc.frame, mask=sc.mask(0), per_piece=10)
    headings = [g["close_heading_deg"] for g in out["grasps"]]
    assert headings
    assert all(min(_heading_diff(h, 0.0), _heading_diff(h, 90.0)) < 13.0 for h in headings)


# --- clusters ---------------------------------------------------------------------------------


def test_a_tight_jigsaw_cluster_is_too_crowded(script):
    sc = _jigsaw(0.0)
    out = script.run(_ctx(), camera_frame=sc.frame, piece_masks=sc.masks())
    assert out["route"] == "too_crowded"
    assert out["n_pieces"] == 16 and out["n_graspable"] == 0
    assert all(
        p["status"] in ("crowded", "neighbour_between", "poor_contact") for p in out["pieces"]
    )
    assert "singulate" in out["reason"]


def test_a_spread_jigsaw_cluster_is_all_graspable(script):
    sc = _jigsaw(0.016)
    out = script.run(_ctx(), camera_frame=sc.frame, piece_masks=sc.masks(), per_piece=1)
    assert out["route"] == "proposed"
    assert out["fraction_graspable"] == 1.0
    assert len(out["grasps"]) == 16
    # every grasp's fingertips come down at the mat, beside its own piece
    for g in out["grasps"]:
        assert g["tip_z"] < 0.0012
        c = np.array([g["position"]["x"], g["position"]["y"]])
        centroid = next(p["centroid"] for p in out["pieces"] if p["piece"] == g["piece"])
        assert np.hypot(*(c - [centroid["x"], centroid["y"]])) < 0.008


def test_graspable_fraction_grows_with_the_packing_gap(script):
    fractions = []
    for gap_mm in (0, 6, 12, 16):
        sc = _jigsaw(gap_mm / 1e3)
        out = script.run(
            _ctx(frame=False),
            camera_frame=sc.frame,
            piece_masks=sc.masks(),
            per_piece=1,
            frame=False,
        )
        fractions.append(out["fraction_graspable"])
    assert fractions == sorted(fractions)
    assert fractions[0] == 0.0 and fractions[1] == 0.0 and fractions[-1] == 1.0


def test_a_thinner_tip_opens_tighter_gaps(script):
    sc = render(brick_grid(3, 3, 0.008, size=LEGO), 0.0096)
    thick = script.run(
        _ctx(frame=False), camera_frame=sc.frame, piece_masks=sc.masks(), frame=False
    )
    thin = script.run(
        _ctx(frame=False),
        camera_frame=sc.frame,
        piece_masks=sc.masks(),
        tip_thickness=0.004,
        frame=False,
    )
    assert thick["n_graspable"] == 0
    assert thin["n_graspable"] == 9


def test_target_mask_limits_grasps_to_the_target(script):
    sc = _jigsaw(0.016)
    out = script.run(_ctx(), camera_frame=sc.frame, mask=sc.mask(5), piece_masks=sc.masks())
    assert out["route"] == "proposed"
    assert out["n_pieces"] == 1
    assert {g["piece"] for g in out["grasps"]} == {1}
    yx = np.argwhere(sc.labels == 6).mean(axis=0)
    u, v = out["grasps"][0]["center_px"]
    assert math.hypot(u - yx[1], v - yx[0]) < 20


def test_target_in_a_tight_cluster_is_too_crowded(script):
    sc = _jigsaw(0.0)
    out = script.run(_ctx(), camera_frame=sc.frame, mask=sc.mask(5), piece_masks=sc.masks())
    assert out["route"] == "too_crowded"
    assert out["n_pieces"] == 1


def test_without_masks_parts_are_found_by_height(script):
    sc = render(brick_grid(3, 2, 0.016, size=LEGO), 0.0096)
    out = script.run(_ctx(), camera_frame=sc.frame)
    assert out["route"] == "proposed"
    assert out["n_pieces"] == 6 and out["n_graspable"] == 6


def test_noise_and_holes_in_the_depth(script):
    sc = _jigsaw(0.016, noise=0.0003, holes=0.5, seed=3)
    out = script.run(_ctx(frame=False), camera_frame=sc.frame, piece_masks=sc.masks(), frame=False)
    assert out["route"] == "proposed"
    assert out["n_graspable"] >= 14
    assert 0.0002 < out["noise_m"] < 0.0005
    assert abs(out["support_z"]) < 0.0005


def test_unknown_cells_block_fingers_unless_told_otherwise(script):
    sc = render([rect(0.0, 0.0, 0.02, 0.03)], 0.004)
    depth = sc.frame["depth"].copy()
    near = (np.abs(np.arange(640)[None, :] - 320) < 100) & (sc.labels == 0)
    depth[near] = 0.0  # no depth within +-46 mm in x: wider than the hand opens
    frame = dict(sc.frame, depth=depth)
    blocked = script.run(_ctx(), camera_frame=frame, mask=sc.mask(0), rotations=2)
    assert blocked["route"] == "too_crowded"
    free = script.run(
        _ctx(), camera_frame=frame, mask=sc.mask(0), rotations=2, unknown_is_obstacle=False
    )
    assert free["route"] == "proposed"


def test_tilted_table_and_offset_camera(script):
    """A support plane tilted 3 degrees (a calibration tilt) still reads parts by their height."""
    sc = render(brick_grid(2, 2, 0.016, size=LEGO), 0.0096, cam_xy=(0.03, -0.02))
    a = math.radians(3.0)
    pose = {
        "position": sc.frame["pose"]["position"],
        "rotation": {"w": 0.0, "x": math.cos(a / 2), "y": math.sin(a / 2), "z": 0.0},
    }
    out = script.run(_ctx(), camera_frame=dict(sc.frame, pose=pose), piece_masks=sc.masks())
    assert out["route"] == "proposed"
    assert out["n_graspable"] == 4
    assert out["grasps"][0]["part_height_m"] == pytest.approx(0.0096, abs=0.0015)


def test_support_z_given_is_used(script):
    sc = render([rect(0.0, 0.0, 0.02, 0.03)], 0.004)
    out = script.run(_ctx(), camera_frame=sc.frame, mask=sc.mask(0), support_z=-0.001)
    assert out["grasps"][0]["support_z"] == pytest.approx(-0.001)
    assert out["grasps"][0]["part_height_m"] == pytest.approx(0.005, abs=0.0005)


def test_camera_is_read_from_the_observation_wrist_first(script):
    sc = render([rect(0.0, 0.0, 0.02, 0.03)], 0.004)
    top = dict(sc.frame, name="top", depth=np.full_like(sc.frame["depth"], 0.5))
    obs = {"cameras": [top, sc.frame]}
    ctx = _ctx(obs=obs)
    out = script.run(ctx, mask=sc.mask(0))
    assert out["route"] == "proposed"
    out_top = script.run(_ctx(obs=obs), camera="top")
    assert out_top["route"] == "nothing_graspable"
    with pytest.raises(ValueError):
        script.run(_ctx(obs={"cameras": [top]}), camera="side")


def test_ignore_mask_drops_pixels(script):
    sc = render([rect(0.0, 0.0, 0.02, 0.03)], 0.004)
    out = script.run(_ctx(), camera_frame=sc.frame, mask=sc.mask(0), ignore_mask=sc.mask(0))
    assert out["route"] == "nothing_graspable"


# --- bundle -----------------------------------------------------------------------------------


def test_example_declares_every_route():
    ex = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert declared == {"proposed", "too_crowded", "nothing_graspable"}
    assert ex["exit"]["router_field"] == "route"


def test_real_overhead_cube_is_proposed():
    f = FRAMES / "real/snapshots/20261031-003520/obs.npz"
    pose = FRAMES / "real/top_camera_pose.before-20261031-004738.json"
    if not (f.exists() and pose.exists()):
        pytest.skip("station frames not on this machine")
    from scipy import ndimage as ndi

    from ._cluster_grasp_eval import load

    mod = load()
    d = np.load(f)
    t = np.array(json.loads(pose.read_text())["T_base_from_camera"])
    dark = ndi.binary_opening(d["rgb"].astype(float).mean(-1) < 75, iterations=1)
    lab, _ = ndi.label(dark)
    cube = ndi.binary_fill_holes(lab == lab[256, 263])
    box = ndi.binary_fill_holes(lab == lab[270, 335])
    frame = {"depth": d["depth_m"], "intrinsics": d["K"], "pose": t}
    out = mod.run(_ctx(), camera_frame=frame, mask=cube, piece_masks=[box])
    assert out["route"] == "proposed"
    assert out["grasps"][0]["part_height_m"] == pytest.approx(0.03, abs=0.006)
    assert out["support_z"] == pytest.approx(-0.032, abs=0.003)  # the calibration's table
