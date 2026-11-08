"""checking-design-connectivity-and-collisions: hand-built brick structures, LDraw input,
a cross-check of the connectivity rule against BrickGPT's own networkx code, and timing
on 100-brick models. No robot tools are called; FakeContext has none scripted, so any
tool call would fail the test."""

from __future__ import annotations

import json
import random
import time
from itertools import combinations
from pathlib import Path

import pytest
from gap.testing import FakeContext

BUNDLE = "checking-design-connectivity-and-collisions"
BUNDLE_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["check_design"].module


def check(script, **kwargs):
    ctx = FakeContext()
    out = script.run(ctx, **kwargs)
    assert ctx.calls == []
    assert out["route"] in script.ROUTES
    return out


def b(x, y, layer, size, rotation=0, **extra):
    """layer counts from 0 here; the skill's cells count from 1 (a brick on the plate is layer 1)."""
    return {"cell": [x, y, layer + 1], "size": size, "rotation": rotation, **extra}


# ------------------------------------------------------------------ valid structures


def test_two_stacked_bricks_are_valid_and_need_no_clutch(script):
    out = check(script, bricks=[b(0, 0, 0, "2x4"), b(0, 2, 1, "2x4")])
    assert out["route"] == "valid"
    assert out["stability_max"] == pytest.approx(0.0, abs=1e-6)
    assert out["floating"] == [] and out["collisions"] == [] and out["weak_joints"] == []
    assert [br["connected"] for br in out["bricks"]] == [True, True]


def test_rotation_swaps_the_footprint(script):
    out = check(script, bricks=[b(0, 0, 0, "2x4"), b(5, 0, 0, "2x4", 90), b(10, 0, 0, "2x4", 270)])
    assert [br["footprint"] for br in out["bricks"]] == [[2, 4], [4, 2], [4, 2]]
    assert out["route"] == "valid"
    # 4 + 4 studs side by side at x=5..8 and x=10..13: not overlapping.
    assert out["collisions"] == []


def test_icl_task_form_with_first_layer_one_matches_the_hub_form(script):
    icl = {
        "1": {"brick_id": 2, "x": 3, "y": 4, "z": 1, "ori": 0, "brick_seq": 1},
        "2": {"brick_id": 2, "x": 3, "y": 5, "z": 2, "ori": 1, "brick_seq": 2},
    }
    hub = [b(3, 4, 0, "2x4"), b(3, 5, 1, "2x4", 90)]
    a = check(script, bricks=json.dumps(icl), first_layer=1)
    c = check(script, bricks=hub)
    assert a["route"] == c["route"] == "valid"
    assert [br["footprint"] for br in a["bricks"]] == [br["footprint"] for br in c["bricks"]]
    assert [br["cell"] for br in a["bricks"]] == [[3, 4, 1], [3, 5, 2]]


def test_a_brick_hanging_under_a_bridge_counts_as_connected(script):
    # BrickGPT's _connected links a brick to the one above it too.
    towers = [b(0, 0, 0, "2x2"), b(0, 0, 1, "2x2"), b(6, 0, 0, "2x2"), b(6, 0, 1, "2x2")]
    span = b(0, 0, 2, "8x2")
    hanging = b(3, 0, 1, "2x2")
    out = check(script, bricks=[*towers, hanging, span], stability="off")
    assert out["floating"] == []
    # Placed before the span it has nothing to hold it.
    assert out["route"] == "step_unsupported"
    assert out["step_unsupported"] == [4]
    out = check(script, bricks=[*towers, span, hanging], stability="off")
    assert out["floating"] == []
    assert out["route"] == "step_blocked"
    assert out["step_blocked"] == [[5, 4]]


# ------------------------------------------------------------------ BrickGPT exits


def test_out_of_plate(script):
    assert check(script, bricks=[b(-1, 0, 0, "2x2")])["route"] == "out_of_plate"
    out = check(script, bricks=[b(0, 0, 0, "2x2"), b(30, 0, 0, "2x4", 90)])
    assert out["route"] == "out_of_plate" and out["out_of_plate"] == [1]
    assert check(script, bricks=[b(30, 0, 0, "2x4", 90)], plate_size="48x48")["route"] == "valid"
    out = check(script, bricks=[b(0, 0, k, "2x2") for k in range(4)], max_layers=3)
    assert out["route"] == "out_of_plate" and out["out_of_plate"] == [3]


def test_collision_lists_the_overlapping_pairs(script):
    out = check(script, bricks=[b(0, 0, 0, "2x4"), b(1, 3, 0, "2x2"), b(5, 0, 0, "1x1")])
    assert out["route"] == "collision"
    assert out["collisions"] == [[0, 1]]
    assert out["stability_max"] != out["stability_max"]  # nan: not computed
    touching = check(script, bricks=[b(0, 0, 0, "2x4"), b(2, 0, 0, "2x4")])
    assert touching["route"] == "valid"


def test_the_same_brick_twice_collides(script):
    out = check(script, bricks=[b(0, 0, 0, "2x2"), b(0, 0, 0, "2x2")])
    assert out["route"] == "collision" and out["collisions"] == [[0, 1]]


def test_floating_brick_with_no_contact(script):
    out = check(script, bricks=[b(0, 0, 0, "2x4"), b(0, 0, 2, "2x4")])
    assert out["route"] == "floating"
    assert out["floating"] == [1] and out["isolated"] == [1]


def test_a_connected_pair_above_the_plate_still_floats(script):
    # BrickGPT's local brick_floats passes both (each touches the other); the
    # ground-connected component does not.
    out = check(script, bricks=[b(0, 0, 0, "2x2"), b(5, 5, 1, "2x4"), b(5, 5, 2, "2x4")])
    assert out["route"] == "floating"
    assert out["floating"] == [1, 2] and out["isolated"] == []


# ------------------------------------------------------------------ per-step checks


def test_step_unsupported_when_the_top_comes_first(script):
    out = check(script, bricks=[b(0, 0, 1, "2x4", step=0), b(0, 0, 0, "2x4", step=1)])
    assert out["floating"] == []
    assert out["route"] == "step_unsupported" and out["step_unsupported"] == [0]
    # The second brick goes under the first: also blocked.
    assert out["step_blocked"] == [[1, 0]]


def test_list_order_is_the_step_order_by_default(script):
    out = check(script, bricks=[b(0, 0, 1, "2x4"), b(0, 0, 0, "2x4")])
    assert out["route"] == "step_unsupported"
    out = check(script, bricks=[b(0, 0, 1, "2x4", step=5), b(0, 0, 0, "2x4", step=2)])
    assert out["route"] == "valid"


def test_step_blocked_by_a_higher_brick_not_directly_above(script):
    # Bricks 0-2 make a pier with a 2x6 cap at layer 2 overhanging x=2..5; brick 3
    # goes on the plate later under that overhang (layer 0, two layers down).
    bricks = [
        b(0, 0, 0, "2x2"),
        b(0, 0, 1, "2x2"),
        b(0, 0, 2, "6x2"),
        b(3, 0, 0, "2x2"),
    ]
    out = check(script, bricks=bricks, stability="off")
    assert out["route"] == "step_blocked"
    assert out["step_blocked"] == [[3, 2]]


# ------------------------------------------------------------------ weak joints


def test_a_long_brick_on_one_stud_is_weak(script):
    out = check(script, bricks=[b(0, 0, 0, "1x1"), b(0, 0, 1, "8x1")])
    assert out["route"] == "weakly_connected"
    assert [-1, 0] in out["weak_joints"] and [0, 1] in out["weak_joints"]
    assert out["weak_bricks"] == [0, 1]
    assert out["weak_at_step"] == [1]
    assert (
        check(script, bricks=[b(0, 0, 0, "1x1"), b(0, 0, 1, "8x1")], allow_weak=True)["route"]
        == "valid"
    )


def test_a_column_of_one_by_ones_is_not_an_overhang(script):
    out = check(script, bricks=[b(4, 4, k, "1x1") for k in range(6)])
    assert out["route"] == "valid" and out["weak_joints"] == []


def test_corner_stud_overlap_of_two_2x4s_is_weak_until_braced(script):
    corner = [b(0, 0, 0, "2x4"), b(1, 3, 1, "2x4")]
    out = check(script, bricks=corner)
    assert out["route"] == "weakly_connected" and out["weak_joints"] == [[0, 1]]
    # A second support under the far end of the top brick removes the single-stud cut.
    brace = b(1, 5, 0, "2x2")
    braced = check(script, bricks=[corner[0], brace, corner[1]])
    assert braced["route"] == "valid" and braced["weak_joints"] == []
    # Listed after the top brick, the brace would go under it.
    late = check(script, bricks=[*corner, brace])
    assert late["route"] == "step_blocked" and late["step_blocked"] == [[2, 1]]


# ------------------------------------------------------------------ stability


def test_cantilever_stack_needs_its_own_weight_in_clutch(script):
    # k 1x8 bricks stacked on a 1x2 base, overhanging 6 studs. Tipping about the
    # base's edge (x=2): weight W at x=4 is held by clutch t at x=0.5 and 1.5,
    # t*(1.5+0.5) = 2W, so t = W = k * 3.03 g * g.
    k = 10
    bricks = [b(0, 0, 0, "2x1")] + [b(0, 0, z, "8x1") for z in range(1, k + 1)]
    out = check(script, bricks=bricks)
    w = k * 3.03e-3 * script.GRAVITY
    assert out["route"] == "valid"
    assert out["stability_max"] == pytest.approx(w / script.CLUTCH_N, rel=1e-4)
    assert out["worst_brick"] in (0, 1)
    weak_clutch = check(script, bricks=bricks, clutch_n=0.25)
    assert weak_clutch["route"] == "unstable"
    assert weak_clutch["stability_max"] == pytest.approx(w / 0.25, rel=1e-4)


def test_a_balanced_build_can_be_unstable_half_way(script):
    # A 1x8 beam on a 1x2 base at x=5..6 overhangs left (centre x=4, tipping edge
    # x=5: t = W_beam / 2 = 0.0148 N). Two 1x2 counterweights at its right end
    # bring the combined centre to x=5.04, inside the support: no clutch needed.
    base = b(5, 0, 0, "2x1")
    beam = b(0, 0, 1, "8x1")
    weights = [b(6, 0, 2, "2x1"), b(6, 0, 3, "2x1")]
    bricks = [base, beam, *weights]
    final = check(script, bricks=bricks, clutch_n=0.01)
    assert final["route"] == "valid"
    assert final["stability_max"] == pytest.approx(0.0, abs=1e-4)
    steps = check(script, bricks=bricks, clutch_n=0.01, stability="steps")
    assert steps["route"] == "unstable" and steps["unstable_step"] == 1
    half = check(script, bricks=[base, beam], clutch_n=0.01)
    assert half["stability_max"] == pytest.approx(3.03e-3 * script.GRAVITY / 2 / 0.01, rel=1e-4)
    # With the real clutch the half-built state holds.
    assert check(script, bricks=bricks, stability="steps")["route"] == "valid"


def test_stability_off_skips_the_program(script):
    out = check(script, bricks=[b(0, 0, 0, "2x4")], stability="off")
    assert out["route"] == "valid"
    assert out["stability_per_brick"] == []


def test_masses_follow_brickgpt_and_fit_the_rest(script):
    assert script._mass_g(script.Brick(h=4, w=2, x=0, y=0, z=0)) == 2.16
    fit = script._mass_g(script.Brick(h=2, w=3, x=0, y=0, z=0))
    assert 1.15 < fit < 2.16


# ------------------------------------------------------------------ LDraw


def brickgpt_to_ldr(h, w, x, y, z, part):
    """BrickGPT's Brick.to_ldr (brick_structure.py L65-72), copied for a round trip."""
    lx = (x + h * 0.5) * 20
    lz = (y + w * 0.5) * 20
    ly = z * -24
    matrix = "0 0 1 0 1 0 -1 0 0" if h <= w else "-1 0 0 0 1 0 0 0 -1"
    return f"1 115 {lx} {ly} {lz} {matrix} {part}\n0 STEP\n"


def test_ldraw_round_trip_of_brickgpt_output(script):
    parts = {(2, 4): "3001.DAT", (1, 2): "3004.DAT", (2, 2): "3003.DAT", (1, 8): "3008.DAT"}
    bricks = [(2, 4, 0, 0, 0), (4, 2, 2, 2, 1), (1, 2, 6, 0, 0), (2, 1, 6, 1, 1), (8, 1, 0, 6, 0)]
    text = "".join(
        brickgpt_to_ldr(h, w, x, y, z, parts[(min(h, w), max(h, w))]) for h, w, x, y, z in bricks
    )
    out = check(script, ldraw=text)
    assert out["n_steps"] == len(bricks)
    got = [(br["footprint"][0], br["footprint"][1], *br["cell"]) for br in out["bricks"]]
    assert got == [(h, w, x, y, z + 1) for h, w, x, y, z in bricks]


def test_ldraw_several_parts_per_step_and_quarter_turns(script):
    text = "\n".join(
        [
            "0 a model",
            "1 4 20 0 40 0 0 1 0 1 0 -1 0 0 3001.dat",
            "1 4 80 0 40 0 0 -1 0 1 0 1 0 0 parts/3001.dat",
            "0 STEP",
            "1 4 40 -24 40 1 0 0 0 1 0 0 0 1 3001.dat",
            "2 24 0 0 0 1 1 1",
            "0 STEP",
        ]
    )
    out = check(script, ldraw=text)
    assert out["n_steps"] == 2
    assert [br["step"] for br in out["bricks"]] == [0, 0, 1]
    assert [br["footprint"] for br in out["bricks"]] == [[2, 4], [2, 4], [4, 2]]
    assert out["bricks"][2]["cell"] == [0, 1, 2]
    assert out["route"] == "valid"


@pytest.mark.parametrize(
    "line, why",
    [
        ("1 4 20 0 40 0 0 1 0 1 0 -1 0 0 3023.dat", "not a supported"),
        ("1 4 25 0 40 0 0 1 0 1 0 -1 0 0 3001.dat", "off the stud grid"),
        ("1 4 20 -8 40 0 0 1 0 1 0 -1 0 0 3001.dat", "off the stud grid"),
        ("1 4 20 0 40 0.707 0 0.707 0 1 0 -0.707 0 0.707 3001.dat", "quarter turn"),
        ("1 4 20 0 40 1 0 0 0 0 -1 0 1 0 3001.dat", "quarter turn"),
        ("1 4 20 0 40 3001.dat", "ill-formed"),
    ],
)
def test_ldraw_refuses_what_is_not_on_the_grid(script, line, why):
    out = check(script, ldraw=line)
    assert out["route"] == "invalid_input"
    assert why in out["detail"]


def test_ldraw_offset_fit_and_extra_parts(script):
    centred = (
        "1 4 0 0 0 0 0 1 0 1 0 -1 0 0 3001.dat\n0 STEP\n1 4 0 -24 0 0 0 1 0 1 0 -1 0 0 3001.dat\n"
    )
    assert check(script, ldraw=centred)["route"] == "out_of_plate"
    shifted = check(script, ldraw=centred, ldraw_offset=[10, 10, 0])
    assert shifted["route"] == "valid" and shifted["bricks"][0]["cell"] == [9, 8, 1]
    fitted = check(script, ldraw=centred, ldraw_fit=True)
    assert fitted["route"] == "valid" and fitted["bricks"][0]["cell"] == [0, 0, 1]
    custom = "1 4 30 0 30 1 0 0 0 1 0 0 0 1 99999.dat"
    assert check(script, ldraw=custom)["route"] == "invalid_input"
    out = check(script, ldraw=custom, extra_parts={"99999.dat": "3x3"})
    assert out["route"] == "valid" and out["bricks"][0]["cell"] == [0, 0, 1]


# ------------------------------------------------------------------ input errors


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"bricks": []},
        {"bricks": [b(0, 0, 0, "2by4")]},
        {"bricks": [b(0, 0, 0, "0x4")]},
        {"bricks": [b(0, 0, 0, "2x4", 45)]},
        {"bricks": [{"cell": [0, 0, 0]}]},
        {"bricks": [{"brick_id": 99, "x": 0, "y": 0, "z": 0, "ori": 0}]},
        {"bricks": [b(0, 0, 0, "2x4")], "stability": "sometimes"},
        {"bricks": [b(0, 0, 0, "2x4")], "plate_size": "big"},
        {"bricks": "not json"},
    ],
)
def test_invalid_input(script, kwargs):
    assert check(script, **kwargs)["route"] == "invalid_input"


# ------------------------------------------------------------------ cross-check with BrickGPT


def brickgpt_connectivity(bricks):
    """BrickGPT's connectivity_score (connectivity_analysis.py L6-55), with Brick objects
    swapped for (h, w, x, y, z) tuples; returns the indices not connected to the ground."""
    nx = pytest.importorskip("networkx")

    def connected(b1, b2):
        if b1[4] > b2[4]:
            b1, b2 = b2, b1
        if b1[4] != b2[4] - 1:
            return False
        return (
            b1[2] < b2[2] + b2[0]
            and b1[2] + b1[0] > b2[2]
            and b1[3] < b2[3] + b2[1]
            and b1[3] + b1[1] > b2[3]
        )

    graph = nx.Graph()
    graph.add_node("ground")
    graph.add_nodes_from(range(len(bricks)))
    for i, br in enumerate(bricks):
        if br[4] == 0:
            graph.add_edge("ground", i)
    for i, j in combinations(range(len(bricks)), 2):
        if connected(bricks[i], bricks[j]):
            graph.add_edge(i, j)
    ok = set(nx.node_connected_component(graph, "ground"))
    return [i for i in range(len(bricks)) if i not in ok]


def random_structure(rng, n, plate=16, layers=6):
    """n non-colliding bricks dropped anywhere on the grid (many float)."""
    sizes = [(1, 1), (1, 2), (2, 2), (1, 4), (2, 4), (2, 3), (1, 6)]
    occupied = set()
    out = []
    while len(out) < n:
        h, w = rng.choice(sizes)
        if rng.random() < 0.5:
            h, w = w, h
        x, y, z = (
            rng.randrange(0, plate - h + 1),
            rng.randrange(0, plate - w + 1),
            rng.randrange(0, layers),
        )
        cells = {(i, j, z) for i in range(x, x + h) for j in range(y, y + w)}
        if cells & occupied:
            continue
        occupied |= cells
        out.append((h, w, x, y, z))
    return out


def test_floating_matches_brickgpt_connectivity_on_random_structures(script):
    rng = random.Random(7)
    disagreements = 0
    for _ in range(200):
        tup = random_structure(rng, rng.randrange(3, 40))
        bricks = [{"cell": [x, y, z + 1], "size": f"{h}x{w}"} for h, w, x, y, z in tup]
        out = check(script, bricks=bricks, plate_size="16x16", max_layers=6, stability="off")
        disagreements += out["floating"] != brickgpt_connectivity(tup)
    assert disagreements == 0


# ------------------------------------------------------------------ timing on 100 bricks


def running_bond_walls():
    """Two 20-stud walls of 2x4 bricks, 10 layers, joints staggered by 2 studs: 100 bricks."""
    bricks = []
    for z in range(10):
        shift = 0 if z % 2 == 0 else 2
        for wall_y in (0, 6):
            for k in range(5):
                bricks.append(b(shift + 4 * k, wall_y, z, "4x2"))
    return bricks


def random_buildable(rng, n, plate=32, layers=20):
    """n bricks, each placed on top of what is built (on the plate or overlapping a brick
    below it by at least two studs), without collisions."""
    sizes = [(1, 2), (2, 2), (1, 4), (2, 4), (2, 3), (1, 6), (2, 6)]
    occ: dict[tuple[int, int, int], int] = {}
    out: list[dict] = []
    while len(out) < n:
        h, w = rng.choice(sizes)
        if rng.random() < 0.5:
            h, w = w, h
        x, y = rng.randrange(8, plate - 8 - h), rng.randrange(8, plate - 8 - w)
        cells = [(i, j) for i in range(x, x + h) for j in range(y, y + w)]
        z = max((k + 1 for (i, j, k) in occ if (i, j) in cells), default=0)
        if z >= layers:
            continue
        support = sum((i, j, z - 1) in occ for i, j in cells)
        if z > 0 and support < 2:
            continue
        for i, j in cells:
            occ[(i, j, z)] = len(out)
        out.append(b(x, y, z, f"{h}x{w}"))
    return out


@pytest.mark.parametrize("model", ["walls", "random"])
@pytest.mark.parametrize("stability", ["off", "final", "steps"])
def test_timing_on_a_100_brick_model(script, model, stability):
    bricks = running_bond_walls() if model == "walls" else random_buildable(random.Random(3), 100)
    assert len(bricks) == 100
    t0 = time.perf_counter()
    out = check(script, bricks=bricks, stability=stability)
    elapsed = time.perf_counter() - t0
    print(
        f"\nTIMING {model} stability={stability}: {elapsed * 1000:.1f} ms, route {out['route']}, "
        f"max utilisation {out['stability_max']:.3f}"
    )
    assert out["route"] == "valid", out["detail"]
    assert elapsed < {"off": 2.0, "final": 5.0, "steps": 60.0}[stability]


# ------------------------------------------------------------------ bundle contract


def test_example_declares_every_route(script):
    example = json.loads((BUNDLE_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == set(script.ROUTES)


def test_skill_md_names_every_route(script):
    text = (BUNDLE_DIR / "SKILL.md").read_text()
    for route in script.ROUTES:
        assert f"    {route}:" in text
