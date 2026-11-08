"""reading-domino-pips-and-orientation: synthetic top-down domino frames, lifted by the real geometry tool."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

from ._domino_render import ALL_VALUES, TILE, render, world_to_pixel

BUNDLE = "reading-domino-pips-and-orientation"
HW = (180, 240)
CENTRE = (120, 90)


@pytest.fixture(scope="module")
def bundle(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b


@pytest.fixture(scope="module")
def script(bundle):
    return bundle.canonical_scripts["read_dominoes"].module


@pytest.fixture(scope="module")
def lift(tool_registry):
    return lambda **kw: tool_registry.invoke("geometry.mask_to_world_points", **kw)


def _ctx(lift, **extra):
    return FakeContext({"geometry.mask_to_world_points": lift, **extra})


def _u8(masks):
    return [m.astype(np.uint8) * 255 for m in masks]


def _read(script, lift, tiles, masks=None, **kw):
    render_kw = {
        k: kw.pop(k)
        for k in list(kw)
        if k
        in ("blur", "noise", "glare", "style", "spinner", "width_px", "seed", "grid", "divider")
    }
    frame, rmasks, truth = render(tiles, hw=kw.pop("hw", HW), **render_kw)
    ctx = _ctx(lift)
    out = script.run(ctx, camera=frame, masks=_u8(rmasks) if masks is None else masks(rmasks), **kw)
    return out, truth, frame, ctx


def _vec(d):
    return np.array([d["x"], d["y"], d["z"]])


def _expected_ab(out_tile, truth):
    """The truth's (a, b) in the order of the tile's reported ends."""
    ea = np.array(out_tile["ends_px"][0])
    flip = np.linalg.norm(ea - truth["end_a_px"]) > np.linalg.norm(ea - truth["end_b_px"])
    return list(truth["value_ab"][::-1] if flip else truth["value_ab"]), flip


@pytest.mark.parametrize("angle", [0.0, 67.0])
def test_all_28_double_six_values_are_read(script, lift, angle):
    for i, value in enumerate(ALL_VALUES):
        out, truth, _, _ = _read(
            script,
            lift,
            [{"value": value, "center_px": CENTRE, "angle_deg": angle + 0.4 * i}],
            seed=i,
        )
        assert out["route"] == "read", (value, out["detail"])
        assert out["value"] == sorted(value)
        want, _ = _expected_ab(out["tile"], truth[0])
        assert out["value_ab"] == want


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 135.0, 180.0, 250.0, 333.0])
def test_rotation_keeps_the_halves_with_their_ends(script, lift, angle):
    out, truth, _, _ = _read(
        script, lift, [{"value": (2, 5), "center_px": CENTRE, "angle_deg": angle}], seed=7
    )
    assert out["route"] == "read", out["detail"]
    want, flip = _expected_ab(out["tile"], truth[0])
    assert out["value_ab"] == want
    axis = _vec(out["axis_world"])
    t_axis = truth[0]["axis"] * (-1 if flip else 1)
    assert np.degrees(np.arccos(np.clip(axis @ t_axis, -1, 1))) < 3.0
    # The half with value a sits at end a.
    ca, cb = (_vec(c) for c in out["tile"]["half_centers_world"])
    ea, eb = (_vec(e) for e in out["ends_world"])
    assert np.linalg.norm(ca - ea) < np.linalg.norm(ca - eb)
    assert np.linalg.norm(cb - eb) < np.linalg.norm(cb - ea)


def test_world_geometry_of_a_flat_tile(script, lift):
    out, truth, _, _ = _read(
        script, lift, [{"value": (3, 6), "center_px": CENTRE, "angle_deg": 20.0}], seed=3
    )
    tile = out["tile"]
    assert out["state"] == "flat"
    assert tile["height"] == pytest.approx(TILE[2], abs=0.002)
    assert np.linalg.norm(_vec(out["center_world"])[:2] - truth[0]["center"][:2]) < 0.002
    ea, eb = (_vec(e) for e in out["ends_world"])
    assert np.linalg.norm(eb - ea) == pytest.approx(TILE[0], abs=0.004)
    assert tile["footprint"][0] == pytest.approx(TILE[0], abs=0.004)
    assert tile["footprint"][1] == pytest.approx(TILE[1], abs=0.004)
    assert len(tile["pip_scores"][0]) + len(tile["pip_scores"][1]) == 9
    assert min(tile["pip_scores"][0] + tile["pip_scores"][1]) > 0.5
    assert tile["divider_found"]


@pytest.mark.parametrize(
    "condition",
    [
        {"blur": 1.5},
        {"noise": 15.0},
        {"style": "black"},
        {"style": "ivory"},
        {"style": "colour"},
        {"spinner": True},
        {"grid": 0.2},
        {"grid": 0.3},
        {"width_px": 34.0},
    ],
    ids=[
        "blur",
        "noise",
        "black-tile",
        "ivory-tile",
        "coloured-pips",
        "spinner",
        "pips-spread",
        "pips-tight",
        "34px",
    ],
)
def test_degraded_frames_are_read(script, lift, condition):
    for i, value in enumerate([(0, 0), (1, 4), (2, 3), (5, 6), (6, 6), (3, 5)]):
        out, truth, _, _ = _read(
            script,
            lift,
            [{"value": value, "center_px": CENTRE, "angle_deg": 15.0 + 50 * i}],
            seed=50 + i,
            **condition,
        )
        assert out["route"] == "read", (condition, value, out["detail"])
        assert out["value"] == sorted(value)


def test_glare_never_reads_a_wrong_value(script, lift):
    read = 0
    for i, value in enumerate(ALL_VALUES[::2]):
        out, _, _, _ = _read(
            script,
            lift,
            [{"value": value, "center_px": CENTRE, "angle_deg": 33.0 * i}],
            seed=200 + i,
            glare=160.0,
        )
        if out["route"] == "read":
            read += 1
            assert out["value"] == sorted(value), out["detail"]
        else:
            assert out["route"] == "ambiguous"
            assert "glare" in out["detail"] or "margin" in out["detail"]
    assert read >= 7


def test_a_loose_mask_still_reads(script, lift):
    out, _, _, _ = _read(
        script,
        lift,
        [{"value": (4, 5), "center_px": CENTRE, "angle_deg": 40.0}],
        masks=lambda ms: [cv2.dilate(ms[0].astype(np.uint8), np.ones((5, 5), np.uint8)) * 255],
        seed=4,
    )
    assert out["route"] == "read", out["detail"]
    assert out["value"] == [4, 5]


def test_a_mask_joining_two_touching_tiles_is_not_read(script, lift):
    tiles = [
        {"value": (2, 2), "center_px": (80, 80), "angle_deg": 0.0},
        {"value": (4, 1), "center_px": (80 + 96, 104), "angle_deg": 0.0},
    ]
    out, _, _, _ = _read(
        script, lift, tiles, masks=lambda ms: [(ms[0] | ms[1]).astype(np.uint8) * 255]
    )
    assert out["route"] == "ambiguous"
    assert "not one flat tile" in out["tile"]["detail"]


def test_without_a_visible_divider_the_value_is_not_trusted(script, lift):
    out, _, _, _ = _read(
        script, lift, [{"value": (1, 5), "center_px": CENTRE, "angle_deg": 20.0}], divider=False
    )
    assert out["route"] == "ambiguous"
    assert "divider not seen" in out["detail"]
    assert out["value"] == [1, 5]


@pytest.mark.parametrize(("state", "height"), [("upright", TILE[0]), ("fallen", TILE[1])])
def test_standing_and_fallen_tiles_are_not_read(script, lift, state, height):
    out, truth, _, _ = _read(
        script, lift, [{"state": state, "center_px": CENTRE, "angle_deg": 25.0}]
    )
    tile = out["tile"]
    assert out["route"] == "ambiguous"
    assert tile["status"] == "not_flat"
    assert out["state"] == state
    assert tile["height"] == pytest.approx(height, abs=0.003)
    assert out["value"] == []
    axis = _vec(out["axis_world"])
    if state == "upright":
        assert axis @ np.array([0.0, 0.0, 1.0]) == pytest.approx(1.0)
        assert out["ends_world"][1]["z"] == pytest.approx(height, abs=0.003)
    else:
        assert abs(axis @ truth[0]["axis"]) > np.cos(np.radians(5))


def test_face_down_tile(script, lift):
    out, _, _, _ = _read(
        script, lift, [{"state": "face_down", "center_px": CENTRE, "angle_deg": 10.0}]
    )
    assert out["route"] == "ambiguous"
    assert out["tile"]["status"] == "face_down"
    assert out["state"] == "flat"
    assert out["value"] == []


def test_double_blank_with_its_divider_is_read(script, lift):
    out, _, _, _ = _read(script, lift, [{"value": (0, 0), "center_px": CENTRE, "angle_deg": 0.0}])
    assert out["route"] == "read", out["detail"]
    assert out["value"] == [0, 0]
    assert out["tile"]["divider_found"]


def test_overhead_scale_asks_for_a_closeup(script, lift):
    # ~11-14 px wide: what the station's overhead Orbbec sees at ~1 m.
    out, truth, _, _ = _read(
        script, lift, [{"value": (2, 4), "center_px": CENTRE, "angle_deg": 30.0}], width_px=13.0
    )
    assert out["route"] == "needs_closeup"
    assert out["tile"]["status"] == "too_small"
    assert out["state"] == "flat"
    assert len(out["closeup_targets"]) == 1
    assert np.linalg.norm(_vec(out["closeup_targets"][0])[:2] - truth[0]["center"][:2]) < 0.004


def test_segmenter_path_drops_weak_and_duplicate_masks(script, lift):
    frame, masks, _ = render([{"value": (1, 3), "center_px": CENTRE, "angle_deg": 12.0}], hw=HW)
    noise_mask = np.zeros(HW, np.uint8)
    noise_mask[5:30, 5:30] = 255
    reply = {"masks": [_u8(masks)[0], _u8(masks)[0], noise_mask], "scores": [0.9, 0.8, 0.1]}
    ctx = _ctx(lift, **{"sam3.segment_text": reply})
    out = script.run(ctx, camera=frame)
    assert out["route"] == "read", out["detail"]
    assert len(out["tiles"]) == 1
    call = ctx.calls_to("sam3.segment_text")[0]
    assert call.kwargs["query"] == "domino"


def test_not_seen_routes(script, lift):
    frame, _, _ = render([{"value": (1, 3), "center_px": CENTRE}], hw=HW)
    empty = _ctx(lift, **{"sam3.segment_text": {"masks": [], "scores": []}})
    assert script.run(empty, camera=frame)["route"] == "not_seen"

    def boom(**_):
        raise ToolError("sam3.segment_text", "no gpu")

    out = script.run(_ctx(lift, **{"sam3.segment_text": boom}), camera=frame)
    assert out["route"] == "not_seen" and "no gpu" in out["detail"]
    nocam = _ctx(lift, **{"robot.get_observation": {"cameras": [], "arms": []}})
    assert script.run(nocam)["route"] == "not_seen"


def test_reads_the_wrist_camera_from_the_observation(script, lift):
    frame, masks, _ = render([{"value": (5, 5), "center_px": CENTRE, "angle_deg": 70.0}], hw=HW)
    overhead = {**frame, "name": "overhead", "rgb": np.zeros_like(frame["rgb"])}
    obs = {"cameras": [overhead, frame], "arms": [{"ee_pose": frame["pose"]}]}
    ctx = _ctx(lift, **{"robot.get_observation": obs})
    out = script.run(ctx, masks=_u8(masks))
    assert out["camera_name"] == "left_wrist"
    assert out["route"] == "read" and out["value"] == [5, 5]


def test_several_tiles_and_a_target_point(script, lift):
    tiles = [
        {"value": (1, 2), "center_px": (70, 60), "angle_deg": 10.0},
        {"state": "upright", "center_px": (180, 130), "angle_deg": 0.0},
    ]
    out, truth, frame, _ = _read(script, lift, tiles, width_px=40.0)
    assert out["route"] == "ambiguous"
    assert [t["status"] for t in out["tiles"]] == ["read", "not_flat"] or [
        t["status"] for t in out["tiles"]
    ] == ["not_flat", "read"]
    target = {"x": float(truth[0]["center"][0]), "y": float(truth[0]["center"][1]), "z": 0.0}
    out2, _, _, _ = _read(script, lift, tiles, width_px=40.0, target_point=target)
    assert out2["route"] == "read"
    assert out2["value"] == [1, 2]
    assert world_to_pixel(
        frame["intrinsics"], frame["pose"], _vec(out2["center_world"])
    ) == pytest.approx(np.array([70, 60]), abs=2.0)


def test_given_table_height_is_used(script, lift):
    out, _, _, ctx = _read(
        script, lift, [{"value": (0, 3), "center_px": CENTRE, "angle_deg": 5.0}], table_z=-0.01
    )
    assert out["tile"]["height"] == pytest.approx(TILE[2] + 0.01, abs=0.002)
    # With the table given, no ring round the tiles is lifted: tile, half a, half b.
    assert ctx.call_count("geometry.mask_to_world_points") == 3


def test_without_depth_state_is_unknown_but_pips_are_read(script):
    frame, masks, _ = render([{"value": (2, 6), "center_px": CENTRE, "angle_deg": 45.0}], hw=HW)
    frame = {k: v for k, v in frame.items() if k != "depth"}
    out = script.run(FakeContext(), camera=frame, masks=_u8(masks))
    assert out["route"] == "read"
    assert out["state"] == "unknown"
    assert out["value"] == [2, 6]
    assert out["axis_world"] == {}


def test_example_declares_every_route(bundle):
    example = json.loads((bundle.bundle_dir / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared >= {"read", "ambiguous", "needs_closeup", "not_seen"}
    assert example["nodes"]["read"]["script"] == "scripts/read_dominoes.py"
