"""regrasping-a-brick-to-turn-studs-up: the tip motion against a fake top-down hand."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from gap.testing import FakeContext

BUNDLE = "regrasping-a-brick-to-turn-studs-up"


@pytest.fixture(scope="module")
def script(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["tip_brick"].module


def hand(refuse=False):
    moves = []

    def go(pose, **kw):
        moves.append(pose)
        if refuse:
            return {"outcome": "unreachable", "reached": False}
        return {"outcome": "reached", "reached": True}

    return moves, FakeContext({"robot.go_to_pose": go, "robot.close_gripper": {"success": True}})


def test_a_side_lying_brick_is_pushed_from_its_stud_face(script):
    moves, ctx = hand()
    out = script.run(
        ctx,
        brick_position={"x": 0.2, "y": -0.2, "z": 0.008},
        up_vector={"x": 1.0, "y": 0.0, "z": 0.0},
    )
    assert out["route"] == "tipped" and out["tips_left"] == 0
    assert out["push_direction"]["x"] == pytest.approx(-1.0)
    start, end = moves[1]["position"], moves[2]["position"]
    assert start["x"] > 0.2 and end["x"] < 0.2  # from the stud side across the brick
    assert start["z"] == end["z"] >= 0.014  # TCP floor
    # Pushes 70 % up the 16 mm the brick stands on its long side.
    assert out["push_height"] == pytest.approx(0.7 * 0.016)
    for m in moves:  # top-down only
        r = m["rotation"]
        assert r["w"] == 0.0 and r["z"] == 0.0


def test_an_upside_down_brick_is_tipped_across_its_short_side_first(script):
    moves, ctx = hand()
    out = script.run(
        ctx,
        brick_position={"x": 0.2, "y": -0.2, "z": 0.006},
        up_vector={"x": 0, "y": 0, "z": -1},
        yaw_deg=90.0,
    )
    assert out["route"] == "tipped" and out["tips_left"] == 1
    assert out["push_direction"]["y"] == pytest.approx(1.0, abs=1e-9)
    assert out["travel"] == pytest.approx(0.016 + 0.006 + 0.008)
    assert out["push_height"] >= 0.014 - 0.0102 - 1e-9


def test_studs_up_needs_nothing_and_refusals_block(script):
    moves, ctx = hand()
    assert (
        script.run(ctx, brick_position=[0.2, 0, 0], up_vector=[0, 0, 1])["route"]
        == "already_studs_up"
    )
    assert moves == []
    moves, ctx = hand(refuse=True)
    out = script.run(ctx, brick_position=[0.2, 0, 0], up_vector=[0, 1, 0])
    assert out["route"] == "blocked" and "refused" in out["detail"]
    with pytest.raises(ValueError):
        script.run(ctx, brick_position=[0, 0, 0], up_vector=[0, 1, 0], system="megablok")


def test_every_route_is_declared_in_the_example(script, skills_registry):
    bundle = skills_registry.get(BUNDLE)
    example = json.loads(
        (Path(bundle.bundle_dir) / "examples" / "canonical_subgraph.json").read_text()
    )
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert set(script.ROUTES) <= declared
    md = (Path(bundle.bundle_dir) / "SKILL.md").read_text()
    for route in script.ROUTES:
        assert f"    {route}:" in md
    assert math.isfinite(script.TCP_FLOOR)
