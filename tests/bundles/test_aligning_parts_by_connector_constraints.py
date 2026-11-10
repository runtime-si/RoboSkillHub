"""aligning-parts-by-connector-constraints: FakeContext tests (no robot tools)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from gap.testing import FakeContext

BUNDLE = "aligning-parts-by-connector-constraints"
PLANS = Path(__file__).parent / "data" / "brick_plans"


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["align_connectors"].module


def solve(script, statements, **kw):
    return script.run(FakeContext({}), statements=statements, **kw)


def parts(out):
    return {p["id"]: p for p in json.loads(out["parts_json"])}


def test_a_brick_on_the_plate_and_one_turned_on_top(script):
    out = solve(
        script,
        [
            {"part": "A", "size": "2x4", "on": "plate", "at": [3, 5]},
            {"part": "B", "size": "2x4", "on": "A", "at": [0, 2], "rotation": 90},
        ],
    )
    assert out["route"] == "poses_solved", out["detail"]
    p = parts(out)
    assert p["A"]["cell"] == {"x": 3, "y": 5, "layer": 1, "rotation": 0}
    assert (p["A"]["span_x"], p["A"]["span_y"]) == (2, 4)
    # B's stud (0, 0) over A's stud (0, 2) = plate (3, 7); turned 90: B runs along -x from there.
    assert p["B"]["cell"] == {"x": 0, "y": 7, "layer": 2, "rotation": 90}
    assert (p["B"]["span_x"], p["B"]["span_y"]) == (4, 2)
    assert p["B"]["z_units"] == 3
    assert json.loads(out["order_json"]) == ["A", "B"]


def test_a_part_stud_anchor_and_a_turned_base(script):
    # A turned 90 on the plate; C's stud (1, 3) over A's stud (1, 0), same heading as A.
    out = solve(
        script,
        [
            {"part": "A", "size": "2x4", "on": "plate", "at": [10, 10], "rotation": 90},
            {"part": "C", "size": "2x4", "on": "A", "at": [1, 0], "part_stud": [1, 3]},
        ],
    )
    p = parts(out)
    assert out["route"] == "poses_solved", out["detail"]
    # A's stud (1, 0) is plate (10, 11); C's stud (0, 0) is A-local (0, -3) -> plate (13, 10).
    assert p["C"]["rotation"] == 90
    assert p["C"]["cell"]["layer"] == 2
    cells_c = {
        (p["C"]["cell"]["x"] + i, p["C"]["cell"]["y"] + j) for i in range(4) for j in range(2)
    }
    assert (10, 11) in cells_c


def test_a_bridge_named_twice_must_agree(script):
    base = [
        {"part": "L", "size": "2x2", "on": "plate", "at": [0, 0]},
        {"part": "R", "size": "2x2", "on": "plate", "at": [4, 0]},
    ]
    # T spelled from L (its stud (0, 0) on L's) and from R (its stud (4, 0) on R's): the same cells.
    good = base + [
        {"part": "T", "size": "6x2", "on": "L", "at": [0, 0]},
        {"part": "T", "size": "6x2", "on": "R", "at": [0, 0], "part_stud": [4, 0]},
    ]
    out = solve(script, good)
    assert out["route"] == "poses_solved", out["detail"]
    assert parts(out)["T"]["cell"] == {"x": 0, "y": 0, "layer": 2, "rotation": 0}
    bad = base + [
        {"part": "T", "size": "6x2", "on": "L", "at": [0, 0]},
        {"part": "T", "size": "6x2", "on": "R", "at": [1, 0], "part_stud": [4, 0]},
    ]
    out = solve(script, bad)
    assert out["route"] == "constraints_inconsistent"
    assert json.loads(out["problems_json"])[0]["kind"] == "conflict"


@pytest.mark.parametrize(
    ("statements", "kind"),
    [
        ([{"part": "A", "size": "2x2", "on": "plate", "at": [0, 0]},
          {"part": "B", "size": "2x2", "on": "A", "at": [5, 5]}], "unsupported"),
        ([{"part": "A", "size": "2x2", "on": "plate", "at": [0, 0]},
          {"part": "B", "size": "2x2", "on": "plate", "at": [1, 1]}], "collision"),
        ([{"part": "A", "size": "2x2", "on": "B"}, {"part": "B", "size": "2x2", "on": "A"}], "cycle"),
        ([{"part": "A", "size": "2x2", "on": "ghost"}], "unknown_base"),
        ([{"part": "A", "size": "2x4", "on": "plate", "at": [31, 0]}], "off_plate"),
        ([{"part": "A", "size": "2x2", "on": "plate", "rotation": 45}], "bad_rotation"),
        ([{"part": "A", "on": "A"}], "self_reference"),
        ([{"size": "2x2", "on": "plate"}], "malformed"),
    ],
)  # fmt: skip
def test_inconsistent_statements_are_named(script, statements, kind):
    out = solve(script, statements)
    assert out["route"] == "constraints_inconsistent"
    assert json.loads(out["problems_json"])[0]["kind"] == kind


def test_one_stud_overhang_is_supported_and_min_support_raises_it(script):
    st = [
        {"part": "A", "size": "2x2", "on": "plate", "at": [0, 0]},
        {"part": "B", "size": "2x2", "on": "A", "at": [1, 1]},
    ]
    assert solve(script, st)["route"] == "poses_solved"
    assert solve(script, st, min_support=2)["route"] == "constraints_inconsistent"


def test_world_poses_from_the_plate_origin(script):
    out = solve(
        script,
        [{"part": "A", "size": "2x4", "on": "plate", "at": [0, 0]}],
        plate={"origin": {"x": 0.4, "y": -0.1, "z": 0.02}, "yaw_deg": 90.0},
        parts=[{"id": "A", "size": "2x4", "colour": "red"}],
    )
    a = parts(out)["A"]
    # Footprint centre (8, 16) mm in plate axes, turned 90: (-16, 8) mm in the world.
    assert a["position"]["x"] == pytest.approx(0.4 - 0.016)
    assert a["position"]["y"] == pytest.approx(-0.1 + 0.008)
    assert a["position"]["z"] == pytest.approx(0.02)
    assert a["yaw_deg"] == pytest.approx(90.0)
    assert a["colour"] == "red"


def test_plates_stack_by_their_own_height(script):
    out = solve(
        script,
        [
            {"part": "P", "size": "2x4", "on": "plate", "at": [0, 0], "height_units": 1},
            {"part": "B", "size": "2x4", "on": "P", "at": [0, 0]},
        ],
    )
    p = parts(out)
    assert p["P"]["kind"] == "plate" and p["B"]["z_units"] == 1


def _statements_from_plan(name):
    """Connector statements re-derived from an LDraw plan's cells: each brick on the plate, or on a
    brick under it that shares a stud, at the base-local stud of its own stud (0, 0)."""
    steps = json.loads((PLANS / name).read_text())
    bricks = [b for s in steps for b in s["bricks"]]
    rot = {
        0: lambda u, v: (u, v),
        90: lambda u, v: (-v, u),
        180: lambda u, v: (-u, -v),
        270: lambda u, v: (v, -u),
    }
    inv = {0: 0, 90: 270, 180: 180, 270: 90}

    def info(b):
        w, h = (int(x) for x in b["brick_size"].split("x"))
        r = int(b["cell"]["rotation"]) % 360
        offs = [rot[r](u, v) for u in range(w) for v in range(h)]
        mx, my = min(o[0] for o in offs), min(o[1] for o in offs)
        origin = (b["cell"]["x"] - mx, b["cell"]["y"] - my)
        cells = {(origin[0] + o[0], origin[1] + o[1]) for o in offs}
        return {
            "w": w,
            "h": h,
            "r": r,
            "origin": origin,
            "cells": cells,
            "z": b["z_units"],
            "hu": b["height_units"],
        }

    meta = {str(b["id"]): info(b) for b in bricks}
    sts = []
    for b in bricks:
        me = meta[str(b["id"])]
        base = "plate"
        for o in bricks:
            om = meta[str(o["id"])]
            if om["z"] + om["hu"] == me["z"] and om["cells"] & me["cells"]:
                base = str(o["id"])
                break
        if base == "plate":
            if me["z"] != 0:
                return None  # stands on something the plan does not list
            at, rr = me["origin"], me["r"]
        else:
            bm = meta[base]
            d = (me["origin"][0] - bm["origin"][0], me["origin"][1] - bm["origin"][1])
            at, rr = rot[inv[bm["r"]]](*d), (me["r"] - bm["r"]) % 360
        sts.append({"part": str(b["id"]), "size": [me["w"], me["h"]], "on": base, "at": list(at),
                    "rotation": rr, "height_units": me["hu"]})  # fmt: skip
    return sts, meta


@pytest.mark.parametrize("name", ["pyramid_steps.json", "flag_1869_steps.json"])
def test_ldraw_plans_round_trip_through_connector_statements(script, name, skills_registry):
    sts, meta = _statements_from_plan(name)
    out = solve(script, sts, plate={"studs": [48, 48]})
    assert out["route"] == "poses_solved", out["detail"]
    got = parts(out)
    assert len(got) == len(meta)
    for pid, m in meta.items():
        g = got[pid]
        cells = {
            (g["cell"]["x"] + i, g["cell"]["y"] + j)
            for i in range(g["span_x"])
            for j in range(g["span_y"])
        }
        assert cells == m["cells"] and g["z_units"] == m["z"], pid
    # The steps go straight into planning-brick-assembly-order.
    plan = (
        skills_registry.get("planning-brick-assembly-order").canonical_scripts["plan_order"].module
    )
    po = plan.run(FakeContext({}), steps_json=out["steps_json"], system="lego", check_fingers=False)
    assert po["n_placed"] == len(meta), po["detail"]


def test_every_route_is_declared_in_the_example(script, skills_registry):
    bundle = skills_registry.get(BUNDLE)
    example = json.loads(
        (Path(bundle.bundle_dir) / "examples" / "canonical_subgraph.json").read_text()
    )
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert set(script.ROUTES) <= declared
    skill_md = (Path(bundle.bundle_dir) / "SKILL.md").read_text()
    for route in script.ROUTES:
        assert f"    {route}:" in skill_md
