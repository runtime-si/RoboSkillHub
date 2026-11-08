"""planning-brick-assembly-order: support, cover, finger room, steps, arms, inventory.

tests/bundles/data/brick_plans holds parsing-ldraw-build-instructions' output for three LDraw
models: pyramid.ldr and car.ldr (James Jessiman, LDraw.org complete library) and 1869-1.mpd
("1869 - South African Flag", Merlijn Wissink, LDraw.org Official Model Repository), CC BY 2.0,
attribution LDraw.org and the authors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from gap.testing import FakeContext

BUNDLE = "planning-brick-assembly-order"
DATA = Path(__file__).parent / "data" / "brick_plans"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["plan_order"].module


def brick(x, y, layer=1, size="2x2", rotation=0, **kw):
    return {"cell": {"x": x, "y": y, "layer": layer}, "size": size, "rotation": rotation, **kw}


def plan(script, bricks=None, **kw):
    if bricks is not None:
        kw["bricks"] = [dict(b, id=i) if "id" not in b else b for i, b in enumerate(bricks)]
    return script.run(FakeContext(), **kw)


def order(out):
    return [e["id"] for e in out["sequence"]]


def test_stack_given_top_first_is_built_bottom_up(script):
    out = plan(script, [brick(0, 0, 3), brick(0, 0, 2), brick(0, 0, 1)])
    assert out["route"] == "sequence_ready"
    assert order(out) == [2, 1, 0]
    assert [e["supports"] for e in out["sequence"]] == [[], [2], [1]]
    first = out["sequence"][0]
    # The fields computing-brick-pose-from-stud-grid takes.
    assert first["cell"] == {"x": 0, "y": 0, "layer": 1, "rotation": 0}
    assert (first["brick_size"], first["rotation"], first["ori"]) == ("2x2", 0, 0)


def test_hanging_brick_and_brick_on_a_tile_are_not_buildable(script):
    bricks = [
        brick(0, 0, 1, "2x4", 90),
        brick(0, 0, 2, "2x4", 90),
        brick(4, 0, 1.666667, "1x2"),  # hangs beside the stack, nothing under it
        brick(10, 0, 1, "2x2", kind="tile", height_units=1),
        brick(10, 0, 1.333333, "2x2"),  # on the tile's smooth top
    ]
    out = plan(script, bricks, check_fingers=False)
    assert out["route"] == "not_buildable"
    assert order(out) == [3, 0, 1]
    reasons = {r["id"]: r["reason"] for r in out["not_placed"]}
    assert set(reasons) == {2, 4}
    assert reasons[2].startswith("nothing under it") and reasons[4].startswith("nothing under it")


def test_plates_stack_in_plate_heights(script):
    bricks = [brick(0, 0, 1.333333, "2x4"), brick(0, 0, 1, "2x4", kind="plate")]
    out = plan(script, bricks)
    assert out["route"] == "sequence_ready" and order(out) == [1, 0]
    assert [e["z_units"] for e in out["sequence"]] == [0, 1]
    assert [e["height_units"] for e in out["sequence"]] == [1, 3]


def test_support_from_one_stud_and_min_overlap(script):
    bricks = [brick(0, 0, 1, "2x2"), brick(1, 1, 2, "2x2")]  # one stud of overlap
    assert plan(script, bricks)["route"] == "sequence_ready"
    out = plan(script, bricks, min_overlap_studs=2)
    assert out["route"] == "not_buildable" and out["not_placed"][0]["id"] == 1


def test_brickgpt_layer_sweep_descending_x_then_y(script):
    row = [brick(x, y, 1, "1x1") for x, y in [(0, 0), (4, 0), (2, 3), (2, 0)]]
    out = plan(script, row, step_order="ignore")
    assert order(out) == [1, 2, 3, 0]  # x 4, (2, 3), (2, 0), 0
    out = plan(script, row, step_order="ignore", sweep="ascending")
    assert order(out) == [0, 3, 2, 1]


def test_step_order_prefer_strict_ignore(script):
    bricks = [
        brick(0, 0, 2, step=1),  # step 1 rests on a brick of step 2
        brick(0, 0, 1, step=2),
        brick(6, 0, 1, step=2),
        brick(6, 0, 2, step=3),
    ]
    prefer = plan(script, bricks)
    assert prefer["route"] == "sequence_ready"
    assert order(prefer) == [1, 0, 2, 3]
    assert prefer["step_deviations"] == [1]
    strict = plan(script, bricks, step_order="strict")
    assert strict["route"] == "not_buildable" and strict["n_placed"] == 0
    assert "strict step order" in strict["not_placed"][1]["reason"] or any(
        "strict" in r["reason"] for r in strict["not_placed"]
    )
    ignore = plan(script, bricks, step_order="ignore")
    assert order(ignore)[:2] == [2, 1]  # layer 1 first, larger x first
    assert ignore["step_deviations"] == []


def test_steps_input_and_json_give_the_same_order(script):
    steps = json.loads((DATA / "pyramid_steps.json").read_text())
    a = script.run(FakeContext(), steps=steps, check_fingers=False)
    b = script.run(FakeContext(), steps_json=json.dumps(steps), check_fingers=False)
    flat = [x for s in steps for x in s["bricks"]]
    c = script.run(FakeContext(), bricks=flat, check_fingers=False)
    assert order(a) == order(b) == order(c)


def test_pyramid_without_finger_check_follows_its_steps(script):
    steps = json.loads((DATA / "pyramid_steps.json").read_text())
    out = script.run(FakeContext(), steps=steps, check_fingers=False)
    assert out["route"] == "sequence_ready" and out["n_placed"] == 13
    assert [e["step"] for e in out["sequence"]] == [1] * 6 + [2] * 4 + [3] * 2 + [4]
    assert out["step_deviations"] == []
    assert json.loads(out["sequence_json"]) == out["sequence"]


def test_pyramid_pinwheel_leaves_no_room_for_the_last_brick(script):
    """The pyramid's second layer is four 2x4s in a pinwheel: whichever goes last touches one
    neighbour along each axis, so a parallel gripper has no free pair of opposite sides."""
    steps = json.loads((DATA / "pyramid_steps.json").read_text())
    out = script.run(FakeContext(), steps=steps)
    assert out["route"] == "finger_blocked"
    assert out["n_placed"] == 12
    (lost,) = out["not_placed"]
    assert lost["step"] == 2 and lost["reason"].startswith("no room for the fingers")


def test_l_shape_needs_the_corner_before_its_second_neighbour(script):
    """Three 2x2s in an L: the corner brick touches both others. BrickGPT's sweep alone would place
    it last and wall it in; the look-ahead places it before the second neighbour."""
    # DUPLO: a LEGO 2x2 (16 mm) is narrower than a fingertip and its margins (26 mm), so there the
    # fingers beside one arm of the L also touch the other arm's corner.
    duplo = {"system": "duplo"}
    bricks = [brick(0, 0, **duplo), brick(2, 0, **duplo), brick(0, 2, **duplo)]
    out = plan(script, bricks)
    assert out["route"] == "sequence_ready"
    seq = order(out)
    assert seq.index(0) < max(seq.index(1), seq.index(2))
    corner = out["sequence"][seq.index(0)]
    assert corner["close_along"] in ("x", "y")


def test_closed_square_of_four_is_finger_blocked(script):
    duplo = {"system": "duplo"}
    bricks = [
        brick(0, 0, **duplo),
        brick(2, 0, **duplo),
        brick(0, 2, **duplo),
        brick(2, 2, **duplo),
    ]
    out = plan(script, bricks)
    assert out["route"] == "finger_blocked" and out["n_placed"] == 3
    lego = plan(script, [brick(0, 0), brick(2, 0), brick(0, 2), brick(2, 2)])
    assert lego["n_placed"] == 2  # the fingertips also reach the diagonal 2x2
    assert plan(script, bricks, check_fingers=False)["route"] == "sequence_ready"


def test_wall_of_bricks_closes_across_the_wall(script):
    # A 1-wide wall along x, two courses, running bond: every brick is gripped across y.
    bricks = [brick(x, 0, 1, "1x4", 90) for x in (0, 4, 8)]
    bricks += [brick(x, 0, 2, "1x4", 90) for x in (2, 6)]
    out = plan(script, bricks)
    assert out["route"] == "sequence_ready"
    assert {e["close_along"] for e in out["sequence"]} == {"y"}
    assert {e["close_axis"] for e in out["sequence"]} == {"short"}


def test_lower_neighbour_does_not_block(script):
    # A plate beside a brick: the plate's top is 6.4 mm under the brick's, the fingertips 5 mm.
    bricks = [brick(2, 0, 1, "2x2", kind="plate"), brick(0, 0, 1, "2x2")]
    out = plan(script, bricks, close_axis="short")
    assert out["route"] == "sequence_ready"
    assert out["sequence"][1]["close_along"] in ("x", "y")


def test_long_side_fallback_and_forced_axes(script):
    # A 2x4 (2 along x) with flush neighbours on both long sides (at x - 1 and x + 2).
    bricks = [brick(0, 0, 1, "1x4"), brick(3, 0, 1, "1x4"), brick(1, 0, 1, "2x4", step=2)]
    out = plan(script, bricks)
    last = out["sequence"][-1]
    assert last["id"] == 2 and (last["close_along"], last["close_axis"]) == ("y", "long")
    short_only = plan(script, bricks, close_axis="short")
    assert short_only["route"] == "finger_blocked"


def test_duplo_jaw_opening(script):
    bricks = [brick(0, 0, 1, "8x8", system="duplo", kind="plate", height_units=1)]
    out = plan(script, bricks)
    assert out["route"] == "finger_blocked"
    assert out["not_placed"][0]["reason"].startswith("wider than the jaws")
    ok = plan(script, [brick(0, 0, 1, "2x10", system="duplo")])
    assert ok["route"] == "sequence_ready"
    assert ok["sequence"][0]["close_axis"] == "short"  # 160 mm the long way: only the short side


def test_two_arms_split_by_x(script):
    bricks = [brick(x, 0, 1, "2x2") for x in (0, 3, 6, 9)]
    out = plan(script, bricks, arms=["left", "right"])
    assert {e["id"]: e["arm"] for e in out["sequence"]} == {
        0: "left",
        1: "left",
        2: "right",
        3: "right",
    }
    assert out["by_arm"] == {"left": [3, 4], "right": [1, 2]}
    flipped = plan(script, bricks, arms=["left", "right"], left_side="+x", split_x=5)
    assert {e["id"]: e["arm"] for e in flipped["sequence"]} == {
        0: "right",
        1: "right",
        2: "left",
        3: "left",
    }


def test_reach_limits_and_unreachable(script):
    bricks = [brick(0, 0), brick(5, 0), brick(10, 0)]
    out = plan(script, bricks, arms=["left", "right"], reach={"left": [0, 7], "right": [9, 20]})
    assert out["route"] == "sequence_ready"
    assert [e["arm"] for e in sorted(out["sequence"], key=lambda e: e["id"])] == [
        "left",
        "left",
        "right",
    ]
    out = plan(script, bricks, arms=["left", "right"], reach={"left": [0, 4], "right": [9, 20]})
    assert out["route"] == "unreachable_step"
    assert out["not_placed"][0]["reason"] == "outside every arm's reach"


def test_inventory_by_part_kind_and_size(script):
    bricks = [brick(x, 0, 1, "2x4", 0, part="3001") for x in (0, 3, 6)]
    bricks.append(brick(0, 0, 2, "2x4", 0, part="3001"))
    out = plan(script, bricks, inventory={"3001": 3})
    assert out["route"] == "inventory_short"
    assert [r["reason"] for r in out["not_placed"]] == ["inventory: no 3001 left"]
    assert plan(script, bricks, inventory={"brick 2x4": 4})["route"] == "sequence_ready"
    assert plan(script, bricks, inventory={"2x4": 2})["n_placed"] == 2


def test_overlap_and_duplicate_are_left_out(script):
    bricks = [brick(0, 0), brick(0, 0), brick(1, 0), brick(5, 0)]
    out = plan(script, bricks)
    assert out["route"] == "not_buildable"
    reasons = {r["id"]: r["reason"] for r in out["not_placed"]}
    assert reasons == {1: "duplicate of brick 0", 2: "overlaps brick 0"}
    assert sorted(order(out)) == [0, 3]


def test_brick_above_covers_the_slot_under_it(script):
    # A bridge over a gap: the 1x2 that belongs in the gap must go before the bridge.
    bricks = [
        brick(0, 0, 1, "1x2", 90),
        brick(4, 0, 1, "1x2", 90),
        brick(2, 0, 1, "1x2", 90),  # the gap
        brick(0, 0, 2, "1x6", 90),  # the bridge
    ]
    out = plan(script, bricks, check_fingers=False)
    assert out["route"] == "sequence_ready"
    assert order(out).index(2) < order(out).index(3)


def test_flag_model_numbers(script):
    steps = json.loads((DATA / "flag_1869_steps.json").read_text())
    loose = script.run(FakeContext(), steps=steps, check_fingers=False)
    assert loose["route"] == "sequence_ready" and loose["n_placed"] == 82
    tight = script.run(FakeContext(), steps=steps)
    assert tight["route"] == "finger_blocked"
    # The flag is a wall one to two studs deep: a 1-wide brick beside 2-wide ones has the fingers'
    # width (26 mm with margins, 3.25 studs) spill onto them.
    assert tight["n_placed"] == 77
    reasons = [r["reason"] for r in tight["not_placed"]]
    assert sum(r.startswith("no room") for r in reasons) == 4
    assert sum(r == "its support was not placed" for r in reasons) == 1
    # Every placed brick rested on the ground or on a brick placed before it.
    placed = set()
    for e in tight["sequence"]:
        assert e["z_units"] == 0 or set(e["supports"]) & placed
        placed.add(e["id"])


def test_car_with_its_unsupported_parts_removed(script):
    steps = json.loads((DATA / "car_steps.json").read_text())
    out = script.run(FakeContext(), steps=steps, check_fingers=False)
    assert out["route"] == "not_buildable"
    assert out["n_placed"] == 11
    assert any(r["reason"].startswith("nothing under it") for r in out["not_placed"])


def test_bad_inputs(script):
    assert plan(script, [])["route"] == "invalid_input"
    assert plan(script, [{"cell": {"x": 0}}])["route"] == "invalid_input"
    assert plan(script, [brick(0, 0, rotation=45)])["route"] == "invalid_input"
    assert plan(script, [brick(0, 0)], step_order="later")["route"] == "invalid_input"
    assert plan(script, [brick(0, 0)], sweep="spiral")["route"] == "invalid_input"
    assert plan(script, [brick(0, 0)], close_axis="diagonal")["route"] == "invalid_input"
    assert plan(script, [brick(0, 0)], left_side="up")["route"] == "invalid_input"
    assert plan(script, [brick(0, 0)], arms=["a", "b", "c"])["route"] == "invalid_input"
    mixed = [brick(0, 0, system="lego"), brick(4, 0, system="duplo")]
    assert plan(script, mixed)["route"] == "invalid_input"
    assert script.run(FakeContext(), steps_json="{not json")["route"] == "invalid_input"


def test_example_declares_every_route(script):
    example = json.loads(
        (ROOT / "skills" / BUNDLE / "examples/canonical_subgraph.json").read_text()
    )
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == {
        "sequence_ready",
        "inventory_short",
        "unreachable_step",
        "not_buildable",
        "finger_blocked",
        "invalid_input",
    }
