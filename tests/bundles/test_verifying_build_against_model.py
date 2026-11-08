"""verifying-build-against-model: LTRON's matching on stud-grid bricks (exact,
missing, extra, misplaced, wrong colour, unseated, ahead, offsets, edit
distance, input forms) and the depth path on ray-cast synthetic frames with the
station's depth noise (tests/bundles/_build_scene.py; the detection rates over
many trials are tests/bundles/_build_verify_eval.py's)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

from . import _build_scene as S
from . import _build_verify_eval as E

BUNDLE = "verifying-build-against-model"
BUNDLE_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
PLATE = {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["verify_build"].module


@pytest.fixture(scope="module")
def classify(skills_registry):
    assert skills_registry.get("colour").kind == "tool"
    return importlib.import_module("gap_skills.tools.colour._impl").classify


def b(x, y, layer, size, rotation=0, colour=None, **extra):
    out = {"cell": {"x": x, "y": y, "layer": layer, "rotation": rotation}, "size": size}
    if colour:
        out["colour"] = colour
    return {**out, **extra}


TARGET = [
    b(0, 0, 1, "2x4", 0, "red", id="a", step=1),
    b(4, 0, 1, "2x2", 0, "blue", id="b", step=1),
    b(0, 1, 2, "2x4", 90, "yellow", id="c", step=2),
    b(8, 8, 1, "1x2", 0, "white", id="d", step=3),
]


def verify(script, target=TARGET, observed=None, ctx=None, **kw):
    ctx = ctx or FakeContext()
    out = script.run(ctx, target=target, observed=observed, **kw)
    assert out["route"] in script.ROUTES
    json.loads(out["report_json"])
    return out


def ids(lst):
    return sorted(str(e["id"]) for e in lst)


# ------------------------------------------------------------------ brick lists


def test_identical_build_matches_with_f1_one_and_no_tool_calls(script):
    ctx = FakeContext()
    out = verify(script, observed=[dict(t) for t in TARGET], ctx=ctx)
    assert ctx.calls == []
    assert out["route"] == "matches"
    assert out["score"] == 1.0 and out["edit_distance"] == 0
    assert out["n_matched"] == 4 and out["repairs"] == []


def test_missing_brick_routes_missing_and_asks_to_place_it(script):
    out = verify(script, observed=TARGET[:3])
    assert out["route"] == "missing_bricks"
    assert ids(out["missing"]) == ["d"]
    assert out["repairs"] == [{"action": "place", "brick": out["missing"][0]}]
    # LTRON's f1: tp / (tp + (fp + fn) / 2) = 3 / 3.5
    assert out["score"] == pytest.approx(3 / 3.5, abs=1e-4)
    assert out["edit_distance"] == 1


def test_extra_brick_routes_extra_and_asks_to_remove_it_first(script):
    obs = TARGET[:3] + [b(10, 2, 1, "1x4", 0, "green", id="x")]
    out = verify(script, observed=obs)
    assert out["route"] == "extra_bricks"
    assert ids(out["extra"]) == ["x"] and ids(out["missing"]) == ["d"]
    assert [r["action"] for r in out["repairs"]] == ["remove", "place"]


def test_same_brick_elsewhere_is_misplaced_with_its_shift(script):
    obs = TARGET[:3] + [b(9, 6, 1, "1x2", 0, "white", id="moved")]
    out = verify(script, observed=obs)
    assert out["route"] == "misplaced"
    (m,) = out["misplaced"]
    assert m["id"] == "d" and m["observed"]["id"] == "moved"
    assert m["shift_studs"] == [1.0, -2.0] and m["shift_layers"] == 0.0
    assert out["missing"] == [] and out["extra"] == []
    assert out["repairs"][0]["action"] == "move"


def test_turned_brick_is_misplaced_with_its_yaw(script):
    # A 2x4 turned a quarter about the same corner: footprint 4x2 instead of 2x4.
    obs = [b(0, 0, 1, "2x4", 90, "red", id="t")] + TARGET[1:]
    out = verify(script, observed=obs)
    assert out["route"] == "misplaced"
    assert out["misplaced"][0]["yaw_off_deg"] in (90.0, -90.0)


def test_half_turn_and_size_order_are_the_same_brick(script):
    obs = [
        b(0, 0, 1, "4x2", 90, "red"),  # "4x2" turned 90 = "2x4" at 0
        b(4, 0, 1, "2x2", 270, "blue"),  # a square brick at any quarter turn
        b(0, 1, 2, "2x4", 270, "yellow"),  # a half turn of rotation 90
        TARGET[3],
    ]
    assert verify(script, observed=obs)["route"] == "matches"


def test_wrong_colour_in_place(script):
    obs = TARGET[:3] + [b(8, 8, 1, "1x2", 0, "black", id="w")]
    out = verify(script, observed=obs)
    assert out["route"] == "wrong_colour"
    assert out["wrong_colour"][0]["expected_colour"] == "white"
    assert out["wrong_colour"][0]["seen_colour"] == "black"
    assert out["repairs"][0]["action"] == "swap"


def test_unknown_colour_is_a_wildcard_and_check_colour_false_ignores_it(script):
    obs = [{k: v for k, v in t.items() if k != "colour"} for t in TARGET]
    assert verify(script, observed=obs)["route"] == "matches"
    wrong = TARGET[:3] + [b(8, 8, 1, "1x2", 0, "black")]
    assert verify(script, observed=wrong, check_colour=False)["route"] == "matches"


@pytest.mark.parametrize(
    "extra, route",
    [
        ({"height_residual_m": 0.0017}, "unseated"),
        ({"seated": False}, "unseated"),
        ({"residual": {"z": 0.0016}}, "unseated"),  # computing-brick-pose-from-stud-grid's
        ({"height_residual_m": 0.0005}, "matches"),
        ({"height_residual_m": 0.0030}, "unseated"),
    ],
)
def test_unseated_from_a_height_residual(script, extra, route):
    obs = TARGET[:3] + [{**TARGET[3], **extra}]
    out = verify(script, observed=obs)
    assert out["route"] == route
    if route == "unseated":
        assert ids(out["unseated"]) == ["d"] and out["repairs"][0]["action"] == "press"


def test_half_a_brick_too_high_is_not_matched(script):
    obs = TARGET[:3] + [{**TARGET[3], "height_residual_m": 0.0050}]
    out = verify(script, observed=obs)
    assert out["route"] == "misplaced"
    assert out["misplaced"][0]["shift_layers"] == pytest.approx(0.0050 / 0.0096, abs=0.01)


def test_duplo_unseated_height_still_matches(script):
    obs = TARGET[:3] + [{**TARGET[3], "height_residual_m": 0.0045}]
    out = verify(script, observed=obs, system="duplo")
    assert out["route"] == "unseated" and out["n_matched"] == 4


def test_two_observed_bricks_cannot_claim_one_target(script):
    # LTRON's validate_matches let both match; the assignment is one to one.
    out = verify(script, target=TARGET[3:], observed=[TARGET[3], dict(TARGET[3], id="dup")])
    assert out["n_matched"] == 1 and len(out["extra"]) == 1
    assert out["route"] == "extra_bricks"


def test_upto_step_counts_later_bricks_as_ahead(script):
    out = verify(script, observed=TARGET, upto_step=2)
    assert out["route"] == "matches"
    assert out["n_target"] == 3 and ids(out["ahead"]) == ["d"]
    assert out["score"] == 1.0 and out["edit_distance"] == 0
    out = verify(script, observed=TARGET[:2], upto_step=2)
    assert out["route"] == "missing_bricks" and ids(out["missing"]) == ["c"]


def test_upto_seq_on_a_planner_sequence(script):
    seq = [{**t, "seq": i + 1} for i, t in enumerate(TARGET)]
    out = verify(script, target={"sequence": seq}, observed=TARGET[:2], upto_seq=2)
    assert out["route"] == "matches"


def test_target_from_parse_ldraw_steps_json(script):
    steps = [
        {"step": 1, "bricks": [TARGET[0], TARGET[1]]},
        {"step": 2, "bricks": [TARGET[2]]},
        {"step": 3, "bricks": [TARGET[3]]},
    ]
    out = verify(script, target=json.dumps(steps), observed=json.dumps(TARGET[:3]), upto_step=2)
    assert out["route"] == "matches" and out["n_target"] == 3
    # The parser's plates: height_units 1 and z_units on a LEGO grid (3 per brick).
    plate = {
        "cell": {"x": 0, "y": 0, "layer": 1, "rotation": 0},
        "size": "2x2",
        "kind": "plate",
        "height_units": 1,
        "z_units": 0,
    }
    on_plate = {
        "cell": {"x": 0, "y": 0, "layer": 1.333333, "rotation": 0},
        "size": "2x2",
        "z_units": 1,
    }
    out = verify(script, target=[plate, on_plate], observed=[plate, on_plate])
    assert out["route"] == "matches"


def test_icl_ori_and_cell_list_forms(script):
    obs = [
        {"x": 0, "y": 0, "layer": 1, "ori": 0, "size": [2, 4], "colour": "red"},
        {"cell": [4, 0, 1], "brick_size": "2x2", "colour": "blue"},
        {"cell": [0, 1, 2], "size": "2x4", "rotation": 90, "colour": "Yellow"},
        TARGET[3],
    ]
    assert verify(script, observed=obs)["route"] == "matches"


def test_whole_build_shifted_on_the_plate(script):
    shifted = [
        {**t, "cell": {**t["cell"], "x": t["cell"]["x"] + 2, "y": t["cell"]["y"] + 1}}
        for t in TARGET
    ]
    out = verify(script, observed=shifted)
    assert out["route"] != "matches"
    assert "would match with the whole build moved by (-2.0, -1.0) studs" in out["detail"]
    free = verify(script, observed=shifted, align="plate")
    assert free["route"] == "matches"
    assert free["build_offset"]["dx"] == -2.0 and free["build_offset"]["dy"] == -1.0


def test_ltron_offset_search_finds_a_quarter_turned_build(script):
    turned = []
    for t in TARGET:
        x, y = t["cell"]["x"], t["cell"]["y"]
        a, bb = (int(v) for v in t["size"].split("x"))
        fx, fy = (bb, a) if t["cell"]["rotation"] % 180 else (a, bb)
        # Turn +90 deg about the origin: (x, y) -> (-y, x); the footprint's low corner moves.
        turned.append(
            {
                **t,
                "cell": {
                    "x": 20 - (y + fy),
                    "y": x,
                    "layer": t["cell"]["layer"],
                    "rotation": (t["cell"]["rotation"] + 90) % 360,
                },
            }
        )
    out = verify(script, observed=turned, align="plate")
    assert out["route"] == "matches"
    assert abs(out["build_offset"]["yaw_deg"]) == pytest.approx(90.0)


def test_edit_distance_counts_a_shifted_sub_assembly_once(script):
    # Bricks a and c (c stands on a) moved together by 5 studs: one rigid move.
    moved = [
        {**t, "cell": {**t["cell"], "x": t["cell"]["x"] + 5}} if t["id"] in ("a", "c") else t
        for t in TARGET
    ]
    out = verify(script, observed=moved)
    assert out["route"] == "misplaced" and out["n_misplaced"] == 2
    assert out["edit_distance"] == 1


def test_metric_pose_observation_reads_through_the_plate_pose(script):
    # A 2x2 LEGO body centre at cell (3, 4) of a 16x16 plate, 0.6 mm off and
    # resting on the studs (+1.7 mm): matched and unseated.
    plate = {"position": {"x": 0.1, "y": -0.2, "z": 0.05}, "rotation": PLATE["rotation"]}
    cx = 0.1 + (-8 + 3 + 1) * 0.008 + 0.0006
    cy = -0.2 + (-8 + 4 + 1) * 0.008
    pose = {
        "position": {"x": cx, "y": cy, "z": 0.05 + 0.0048 + 0.0017},
        "rotation": PLATE["rotation"],
    }
    out = verify(
        script,
        target=[b(3, 4, 1, "2x2", 0, "red")],
        observed=[{"brick_pose": pose, "size": "2x2", "colour": "red"}],
        plate_pose=plate,
        plate_studs=[16, 16],
    )
    assert out["route"] == "unseated"
    assert out["unseated"][0]["height_residual_mm"] == pytest.approx(1.7, abs=0.01)
    far = dict(pose, position=dict(pose["position"], x=cx + 0.004))
    out = verify(
        script,
        target=[b(3, 4, 1, "2x2", 0, "red")],
        observed=[{"brick_pose": far, "size": "2x2", "colour": "red"}],
        plate_pose=plate,
        plate_studs=[16, 16],
    )
    assert out["route"] == "misplaced"


def test_empty_observation_is_all_missing_and_empty_target_with_nothing_matches(script):
    out = verify(script, observed=[])
    assert out["route"] == "missing_bricks" and out["score"] == 0.0
    out = verify(script, target=[], observed=[])
    assert out["route"] == "matches" and out["score"] == 1.0


@pytest.mark.parametrize(
    "kw",
    [
        {"observed": [b(0, 0, 1, "2x4", 45)]},
        {"observed": [{"cell": [0, 0, 1]}]},
        {"observed": [b(0, 0, 1, "0x4")]},
        {"observed": None},  # the depth path without plate_pose
        {"observed": [], "system": "megablok"},
        {"observed": [], "align": "rigid"},
    ],
)
def test_malformed_input_raises(script, kw):
    with pytest.raises(ValueError):
        script.run(FakeContext(), target=TARGET, **kw)


def test_example_declares_every_route(script):
    ex = json.loads((BUNDLE_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert set(script.ROUTES) <= declared


def test_f1_and_edit_distance_follow_ltron(script):
    assert script._f1(3, 1, 1) == pytest.approx(3 / 4)
    assert script._f1(0, 0, 0) == 1.0


# ------------------------------------------------------------------ depth frames


@pytest.mark.parametrize("system", ["lego", "duplo"])
@pytest.mark.parametrize("error", ["clean", "missing", "extra", "misplaced", "unseated", "tilted"])
def test_wrist_depth_detects_each_error(script, system, error):
    hits = sum(E.trial(script, system, "wrist", error, 500 + s)[1] for s in range(3))
    assert hits == 3


def test_wrist_rgb_wrong_colour_through_the_colour_tool(script, classify):
    for s in range(2):
        route, ok, out = E.trial(script, "lego", "wrist", "wrong_colour", 600 + s, classify)
        assert ok, out["detail"]


@pytest.mark.parametrize("error", ["clean", "missing", "extra", "misplaced"])
def test_overhead_lego_sees_presence_but_not_seating(script, error):
    hits = sum(E.trial(script, "lego", "overhead", error, 700 + s)[1] for s in range(3))
    assert hits >= 2


def test_overhead_lego_seating_is_unresolved_and_can_be_required(script):
    rng = np.random.default_rng(3)
    scene = S.random_build(rng, "lego")
    target = [t.as_target(i) for i, t in enumerate(scene.bricks)]
    _, t_cam, k, noise = S.cameras(rng, "lego")[0]
    depth, _ = S.render(scene, np.eye(4), t_cam, k, noise, seed=3)
    cam = {"depth": depth, "intrinsics": k, "pose": t_cam}
    kw = {
        "target": target,
        "camera": cam,
        "plate_pose": PLATE,
        "plate_studs": [12, 12],
        "system": "lego",
    }
    out = script.run(FakeContext(), **kw)
    assert out["route"] == "matches" and out["seat_unresolved"]
    assert "within the depth noise" in out["detail"]
    assert 1.5 < out["depth_sigma_mm"] < 4.0
    out = script.run(FakeContext(), require_seating=True, **kw)
    assert out["route"] == "unverifiable"


def _wrist_frame(system="lego", seed=11, scene=None):
    rng = np.random.default_rng(seed)
    scene = scene or S.random_build(rng, system)
    _, t_cam, k, noise = S.cameras(rng, system)[1]
    depth, rgb = S.render(scene, np.eye(4), t_cam, k, noise, seed=seed)
    return scene, {"name": "left_wrist", "depth": depth, "intrinsics": k, "pose": t_cam, "rgb": rgb}


def test_camera_from_the_observation_by_name(script):
    scene, cam = _wrist_frame()
    target = [t.as_target(i) for i, t in enumerate(scene.bricks)]
    other = dict(cam, name="overhead", depth=np.zeros_like(cam["depth"]))
    ctx = FakeContext({"robot.get_observation": {"cameras": [other, cam]}})
    out = script.run(
        ctx,
        target=target,
        camera_name="left_wrist",
        plate_pose=PLATE,
        plate_studs=[12, 12],
        check_colour=False,
    )
    assert out["route"] == "matches" and out["mode"] == "depth"
    assert ctx.call_count("robot.get_observation") == 1


def test_no_camera_or_no_plate_in_view_is_unverifiable(script):
    target = TARGET
    ctx = FakeContext({"robot.get_observation": {"cameras": []}})
    out = script.run(ctx, target=target, plate_pose=PLATE, plate_studs=[12, 12])
    assert out["route"] == "unverifiable"

    def boom(**_):
        raise RuntimeError("no robot")

    out = script.run(
        FakeContext({"robot.get_observation": boom}),
        target=target,
        plate_pose=PLATE,
        plate_studs=[12, 12],
    )
    assert out["route"] == "unverifiable" and "no robot" in out["detail"]
    scene, cam = _wrist_frame()
    away = {**PLATE, "position": {"x": 1.0, "y": 0.0, "z": 0.0}}
    out = script.run(
        FakeContext(), target=target, camera=cam, plate_pose=away, plate_studs=[12, 12]
    )
    assert out["route"] == "unverifiable"


def test_plate_pose_a_millimetre_low_is_levelled(script):
    scene, cam = _wrist_frame()
    target = [t.as_target(i) for i, t in enumerate(scene.bricks)]
    low = {**PLATE, "position": {"x": 0.0, "y": 0.0, "z": -0.0015}}
    out = script.run(
        FakeContext(),
        target=target,
        camera=cam,
        plate_pose=low,
        plate_studs=[12, 12],
        check_colour=False,
    )
    assert out["route"] == "matches"
    assert out["level_correction_mm"][0] == pytest.approx(1.5, abs=0.2)
    out = script.run(
        FakeContext(),
        target=target,
        camera=cam,
        plate_pose=low,
        plate_studs=[12, 12],
        check_colour=False,
        level_plate=False,
    )
    assert out["route"] == "unseated"  # 1.5 mm high everywhere, uncorrected


def test_integer_depth_in_millimetres(script):
    scene, cam = _wrist_frame()
    target = [t.as_target(i) for i, t in enumerate(scene.bricks)]
    mm = dict(cam, depth=np.round(cam["depth"] * 1000).astype(np.uint16))
    out = script.run(
        FakeContext(),
        target=target,
        camera=mm,
        plate_pose=PLATE,
        plate_studs=[12, 12],
        check_colour=False,
    )
    assert out["route"] == "matches"


def test_colour_tool_failure_leaves_colours_unchecked(script):
    scene, cam = _wrist_frame()
    target = [t.as_target(i) for i, t in enumerate(scene.bricks)]
    target[0]["colour"] = "magenta"  # wrong on purpose: unchecked, so not reported

    def boom(**_):
        raise RuntimeError("colour bundle down")

    ctx = FakeContext({"colour.classify": boom})
    out = script.run(ctx, target=target, camera=cam, plate_pose=PLATE, plate_studs=[12, 12])
    assert out["route"] == "matches"
    assert "colours not checked" in out["detail"]
    assert ctx.call_count("colour.classify") == 1


def test_ahead_in_depth_mode(script):
    rng = np.random.default_rng(21)
    scene = S.random_build(rng, "duplo")
    target = [t.as_target(i) for i, t in enumerate(scene.bricks)]
    last = max(t["step"] for t in target)
    _, cam = _wrist_frame("duplo", 21, scene)
    out = script.run(
        FakeContext(),
        target=target,
        camera=cam,
        plate_pose=PLATE,
        plate_studs=[12, 12],
        system="duplo",
        upto_step=last - 1,
        check_colour=False,
    )
    assert out["route"] == "matches"
    assert [a["step"] for a in out["ahead"]] == [last]
