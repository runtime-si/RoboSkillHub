"""tracking-build-state: FakeContext tests of the brick world model (no robot tools)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from gap.testing import FakeContext

BUNDLE = "tracking-build-state"
PLANS = Path(__file__).parent / "data" / "brick_plans"
P, BH = 0.016, 0.0192  # DUPLO pitch and brick height


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["track_build"].module


def ctx():
    return FakeContext({})


def brick(bid, cx, cy, layer=1, studs=(2, 2), yaw=0.0, **kw):
    """A DUPLO brick whose footprint centre is (cx, cy) in pitches, on ``layer``."""
    return {
        "id": bid,
        "studs": list(studs),
        "position": {"x": cx * P, "y": cy * P, "z": (layer - 1) * BH},
        "yaw_deg": yaw,
        **kw,
    }


def build(script, *bricks, system="duplo"):
    state = ""
    out = None
    for b in bricks:
        out = script.run(
            ctx(), action="place", state_json=state, brick=b, system=system, ground_z=0.0
        )
        assert out["route"] == "state_updated", out["detail"]
        state = out["state_json"]
    return out


def ids(js):
    return json.loads(js)


def test_a_stack_connects_and_grounds(script):
    out = build(script, brick("a", 1, 1), brick("b", 1, 1, layer=2), brick("c", 1.5, 1, layer=3))
    assert out["n_bricks"] == 3 and out["n_connections"] == 2
    assert out["n_grounded"] == 3 and ids(out["floating_json"]) == []
    assert ids(out["removable_json"]) == ["c"]
    assert "on b" in out["detail"]


def test_flush_neighbours_do_not_connect(script):
    out = build(script, brick("a", 1, 1), brick("b", 3, 1))
    assert out["n_connections"] == 0 and out["n_grounded"] == 2
    assert ids(out["removable_json"]) == ["a", "b"]


def test_a_brick_bridging_two_connects_to_both(script):
    out = build(
        script, brick("a", 1, 1), brick("b", 3, 1), brick("top", 2, 1, layer=2, studs=(4, 2))
    )
    assert out["n_connections"] == 2
    state = json.loads(out["state_json"])
    assert {b["id"] for b in state["bricks"]} == {"a", "b", "top"}


def test_one_stud_overlap_connects_a_corner_touch_does_not(script):
    # Offset by one pitch in x and y: the 2x2s share one stud.
    out = build(script, brick("a", 1, 1), brick("b", 2, 2, layer=2))
    assert out["n_connections"] == 1
    # Offset by two pitches: corners touch, no stud shared.
    out = build(script, brick("a", 1, 1), brick("b", 3, 3, layer=2))
    assert out["n_connections"] == 0
    assert ids(out["floating_json"]) == ["b"]


def test_a_rotated_brick_overlaps_by_its_true_footprint(script):
    # A 2x4 turned 90 deg over a 2x2 at its end: overlaps; turned 0 deg it would not reach.
    out = build(script, brick("a", 1, 1), brick("b", 1, 3, layer=2, studs=(2, 4), yaw=90.0))
    assert out["n_connections"] == 0
    out = build(script, brick("a", 1, 1), brick("b", 1, 2, layer=2, studs=(2, 4), yaw=90.0))
    assert out["n_connections"] == 1
    out = build(script, brick("a", 1, 1), brick("b", 1, 1, layer=2, yaw=30.0))
    assert out["n_connections"] == 1


def test_a_brick_resting_on_the_studs_is_not_connected(script):
    high = brick("b", 1, 1, layer=2)
    high["position"]["z"] += 0.0045  # a stud height up: on the stud tops
    out = build(script, brick("a", 1, 1), high)
    assert out["n_connections"] == 0
    assert ids(out["floating_json"]) == ["b"]


def test_placing_into_a_brick_is_rejected_and_leaves_the_state(script):
    out = build(script, brick("a", 1, 1))
    state = out["state_json"]
    r = script.run(ctx(), action="place", state_json=state, brick=brick("b", 2, 1))
    assert r["route"] == "rejected" and "interpenetrate a" in r["detail"]
    assert r["n_bricks"] == 1
    r = script.run(ctx(), action="place", state_json=state, brick=brick("a", 5, 5))
    assert r["route"] == "rejected" and "already placed" in r["detail"]
    r = script.run(ctx(), action="place", state_json=state, brick={"id": "x"})
    assert r["route"] == "rejected"
    r = script.run(
        ctx(), action="place", state_json=state, brick={**brick("x", 5, 5), "studs": "0x2"}
    )
    assert r["route"] == "rejected" and "bad brick" in r["detail"]


def test_removing_a_brick_carries_what_is_on_it(script):
    out = build(
        script,
        brick("a", 1, 1),
        brick("b", 1, 1, layer=2),
        brick("c", 1, 1, layer=3),
        brick("d", 4, 1),
    )
    r = script.run(ctx(), action="remove", state_json=out["state_json"], brick_id="b")
    assert r["route"] == "state_updated"
    assert ids(r["changed_json"]) == ["b", "c"]
    assert r["n_bricks"] == 2 and ids(r["floating_json"]) == []


def test_removing_without_carry_leaves_the_upper_bricks_floating(script):
    out = build(script, brick("a", 1, 1), brick("b", 1, 1, layer=2), brick("c", 1, 1, layer=3))
    r = script.run(ctx(), action="remove", state_json=out["state_json"], brick_id="a", carry=False)
    assert ids(r["changed_json"]) == ["a"]
    assert ids(r["floating_json"]) == ["b", "c"]


def test_moving_a_brick_moves_everything_above_it(script):
    out = build(script, brick("a", 1, 1), brick("b", 1, 1, layer=2), brick("c", 1.5, 1, layer=3))
    r = script.run(
        ctx(),
        action="move",
        state_json=out["state_json"],
        brick_id="a",
        new_position={"x": 5 * P, "y": 1 * P, "z": 0.0},
        new_yaw_deg=90.0,
    )
    assert r["route"] == "state_updated", r["detail"]
    assert ids(r["changed_json"]) == ["a", "b", "c"]
    st = {b["id"]: b for b in json.loads(r["state_json"])["bricks"]}
    assert st["b"]["position"]["x"] == pytest.approx(5 * P)
    # c sat half a pitch off a's centre in x; turned 90 deg it is half a pitch off in y.
    assert st["c"]["position"]["x"] == pytest.approx(5 * P, abs=1e-9)
    assert st["c"]["position"]["y"] == pytest.approx(1.5 * P, abs=1e-9)
    assert st["c"]["yaw_deg"] == pytest.approx(90.0)
    assert r["n_connections"] == 2


def test_a_move_into_another_brick_is_rejected(script):
    out = build(script, brick("a", 1, 1), brick("b", 1, 1, layer=2), brick("w", 4, 1, layer=2))
    r = script.run(
        ctx(),
        action="move",
        state_json=out["state_json"],
        brick_id="a",
        new_position={"x": 4 * P, "y": 1 * P, "z": 0.0},
    )
    assert r["route"] == "rejected" and "into w" in r["detail"]
    r = script.run(ctx(), action="move", state_json=out["state_json"], brick_id="zz", new_yaw_deg=0)
    assert r["route"] == "rejected"
    r = script.run(ctx(), action="move", state_json=out["state_json"], brick_id="a")
    assert r["route"] == "rejected"


def _obs(b, dx=0.0, dy=0.0, dz=0.0, yaw=None, with_id=False):
    o = {
        "position": {
            "x": b["position"]["x"] + dx,
            "y": b["position"]["y"] + dy,
            "z": b["position"]["z"] + dz,
        },
        "studs": b["studs"],
        "yaw_deg": b["yaw_deg"] if yaw is None else yaw,
    }
    if with_id:
        o["id"] = b["id"]
    return o


def test_observation_that_agrees_keeps_the_state(script):
    a, b, c = brick("a", 1, 1), brick("b", 1, 1, layer=2), brick("c", 4, 1)
    out = build(script, a, b, c)
    # Only the bricks with nothing on top are expected in view (a is under b).
    r = script.run(
        ctx(),
        action="observe",
        state_json=out["state_json"],
        observations=[_obs(b, dx=0.002), _obs(c, yaw=180.0)],
    )
    assert r["route"] == "state_updated", r["detail"]
    assert ids(r["discrepancies_json"]) == []


def test_a_brick_that_left_with_the_gripper_is_missing(script):
    a, b = brick("a", 1, 1), brick("b", 4, 1)
    out = build(script, a, b)
    r = script.run(ctx(), action="observe", state_json=out["state_json"], observations=[_obs(a)])
    assert r["route"] == "inconsistent_with_observation"
    d = ids(r["discrepancies_json"])
    assert d == [{"id": "b", "kind": "missing"}]
    assert r["n_bricks"] == 2  # unchanged unless applied
    r = script.run(
        ctx(),
        action="observe",
        state_json=out["state_json"],
        observations=[_obs(a)],
        apply_observation=True,
    )
    assert r["n_bricks"] == 1 and ids(r["changed_json"]) == ["b"]


def test_displaced_high_and_extra_bricks_are_reported(script):
    a, b, c = brick("a", 1, 1), brick("b", 4, 1), brick("c", 7, 1)
    out = build(script, a, b, c)
    obs = [
        _obs(a, dx=0.006),  # 6 mm off: a quarter pitch is 4 mm
        _obs(b, dz=0.0045),  # resting on the studs
        _obs(c, yaw=30.0, with_id=True),
        {"position": {"x": 0.3, "y": 0.3, "z": 0.0}, "studs": [2, 4]},
    ]
    r = script.run(ctx(), action="observe", state_json=out["state_json"], observations=obs)
    assert r["route"] == "inconsistent_with_observation"
    d = ids(r["discrepancies_json"])
    kinds = {x.get("id", "extra"): x["kind"] for x in d}
    assert kinds == {"a": "displaced", "b": "displaced", "c": "displaced", "extra": "extra"}
    applied = script.run(
        ctx(),
        action="observe",
        state_json=out["state_json"],
        observations=obs,
        apply_observation=True,
    )
    st = {x["id"]: x for x in json.loads(applied["state_json"])["bricks"]}
    assert st["a"]["position"]["x"] == pytest.approx(P + 0.006)
    assert applied["n_bricks"] == 4


def test_expect_visible_all_counts_hidden_bricks(script):
    a, b = brick("a", 1, 1), brick("b", 1, 1, layer=2)
    out = build(script, a, b)
    r = script.run(
        ctx(),
        action="observe",
        state_json=out["state_json"],
        observations=[_obs(b)],
        expect_visible="all",
    )
    assert ids(r["discrepancies_json"]) == [{"id": "a", "kind": "missing"}]


def test_query_and_bad_requests(script):
    out = build(script, brick("a", 1, 1))
    q = script.run(ctx(), action="query", state_json=out["state_json"])
    assert q["route"] == "state_updated" and q["n_bricks"] == 1
    assert script.run(ctx(), action="observe", state_json=out["state_json"])["route"] == "rejected"
    with pytest.raises(ValueError):
        script.run(ctx(), action="paint")
    with pytest.raises(ValueError):
        script.run(ctx(), action="query", system="megablok")
    with pytest.raises(ValueError):
        script.run(ctx(), action="query", expect_visible="some")


def test_lego_plates_and_bricks_mix(script):
    # A LEGO 2x4 brick, a 2x4 plate on it (3.2 mm), a 2x2 brick on the plate.
    pl = 0.0032
    bricks = [
        {
            "id": "b",
            "system": "lego",
            "studs": [4, 2],
            "position": [0.016, 0.008, 0.0],
            "height": 0.0096,
        },
        {
            "id": "p",
            "system": "lego",
            "studs": [4, 2],
            "position": [0.016, 0.008, 0.0096],
            "height": pl,
        },
        {
            "id": "t",
            "system": "lego",
            "studs": [2, 2],
            "position": [0.012, 0.008, 0.0128],
            "height": 0.0096,
        },
    ]
    out = build(script, *bricks, system="lego")
    assert out["n_connections"] == 2 and out["n_grounded"] == 3


def _plan_bricks(name):
    steps = json.loads((PLANS / name).read_text())
    unit = 0.0032
    out = []
    for s in steps:
        for b in s["bricks"]:
            c = b["cell"]
            out.append(
                {
                    "id": str(b["id"]),
                    "system": b["system"],
                    "studs": [b["span_x"], b["span_y"]],
                    "position": [
                        (c["x"] + b["span_x"] / 2) * 0.008,
                        (c["y"] + b["span_y"] / 2) * 0.008,
                        b["z_units"] * unit,
                    ],
                    "height": b["height_units"] * unit,
                    "colour": b.get("colour"),
                }
            )
    return out


@pytest.mark.parametrize(
    ("name", "n", "floating"),
    [("pyramid_steps.json", 13, 0), ("flag_1869_steps.json", 82, 0), ("car_steps.json", 33, None)],
)
def test_ldraw_models_build_up_grounded(script, name, n, floating):
    bricks = _plan_bricks(name)
    assert len(bricks) == n
    t0 = time.perf_counter()
    out = build(script, *bricks, system="lego")
    took = time.perf_counter() - t0
    assert out["n_bricks"] == n
    if floating is not None:
        assert len(ids(out["floating_json"])) == floating, out["floating_json"]
    else:
        # The car stands partly on parts the LDraw parser could not convert.
        assert 0 < len(ids(out["floating_json"])) < n
    assert took < 30.0


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
