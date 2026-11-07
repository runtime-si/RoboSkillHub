"""checking-spatial-relations: FakeContext tests of REFLECT's relation tests and the routing.

Scenes are synthetic visible-surface clouds in a z-up world (metres); the
masks tests render a depth image from a camera 1 m over the table looking
straight down.
"""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "checking-spatial-relations"
RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["check_relation"].module


def _cube(center, half=0.015, n=3000):
    """Top and side faces of a cube resting at ``center`` (its bottom hidden)."""
    p = RNG.uniform(-half, half, (n, 3))
    face = RNG.integers(0, 3, n)
    p[np.arange(n), face] = np.sign(RNG.uniform(-1, 1, n)) * half
    return p[p[:, 2] > -half + 1e-6] + np.asarray(center)


def _open_box(center=(0.0, 0.0, 0.0), half=0.05, height=0.04, n=8000):
    floor = np.c_[RNG.uniform(-half, half, (n, 2)), np.full(n, 0.002)]
    walls = []
    for s in (-1, 1):
        w = np.c_[
            np.full(n // 4, s * half),
            RNG.uniform(-half, half, n // 4),
            RNG.uniform(0, height, n // 4),
        ]
        walls += [w, w[:, [1, 0, 2]]]
    return np.vstack([floor, *walls]) + np.asarray(center)


def _table(n=30000):
    return np.c_[RNG.uniform(-0.3, 0.3, (n, 2)), RNG.normal(0.0, 0.001, n)]


BOX = _open_box()
TABLE = _table()


def _ask(script, relation, sub, ref, ctx=None, **kw):
    return script.run(
        ctx or FakeContext(), relation, subject_points=sub, reference_points=ref, **kw
    )


@pytest.mark.parametrize(
    ("center", "relation", "route"),
    [
        ((0.0, 0.0, 0.0195), "inside", "holds"),
        ((0.0, 0.0, 0.0195), "near", "holds"),
        ((0.0, 0.0, 0.0195), "on_top_of", "does_not_hold"),
        ((0.0, 0.095, 0.015), "inside", "does_not_hold"),
        ((0.0, 0.095, 0.015), "near", "holds"),
        ((0.0, 0.25, 0.015), "near", "does_not_hold"),
        ((0.0, 0.25, 0.015), "inside", "does_not_hold"),
    ],
)
def test_cube_and_open_box(script, center, relation, route):
    out = _ask(script, relation, _cube(center), BOX)
    assert out["route"] == route, out["evidence"]
    assert out["holds"] is (route == "holds")


def test_cube_resting_on_a_table_is_on_top_of_it(script):
    out = _ask(script, "on", _cube((0.1, 0.1, 0.015)), TABLE)
    assert out["route"] == "holds"
    assert out["relation"] == "on_top_of"
    assert out["measures"]["over_fraction"] > 0.7 and out["measures"]["above_fraction"] > 0.7


def test_a_held_cube_is_not_on_the_table(script):
    out = _ask(script, "on_top_of", _cube((0.1, 0.1, 0.12)), TABLE)
    assert out["route"] == "does_not_hold"
    assert out["min_distance_m"] > 0.05


def test_cube_on_a_closed_block_is_on_top_of_it(script):
    block = _cube((0.0, 0.0, 0.02), half=0.02, n=6000)
    cube = _cube((0.0, 0.0, 0.0555), half=0.015)
    out = _ask(script, "on_top_of", cube, block)
    assert out["route"] == "holds", out["evidence"]
    # and the reverse edge reads "under"
    assert "under" in _ask(script, "on_top_of", block, cube)["relations"]


def test_flat_reference_turns_inside_into_on_top_of(script):
    # A board with 15 mm parts on it: a thin cable lying among them sits in its hull.
    board = np.vstack(
        [
            np.c_[RNG.uniform(-0.08, 0.08, (8000, 2)), np.full(8000, 0.010)],
            np.c_[RNG.uniform(-0.08, 0.08, (500, 2)), RNG.uniform(0.010, 0.016, 500)],
        ]
    )
    cable = np.c_[np.linspace(-0.06, 0.06, 300), np.zeros(300), np.full(300, 0.013)]
    auto = _ask(script, "on_top_of", cable, board)
    assert "inside" not in auto["relations"]
    assert auto["route"] == "holds", auto["evidence"]
    container = _ask(script, "inside", cable, board, reference_kind="container")
    assert container["route"] == "holds"


def test_too_few_points_is_not_visible(script):
    out = _ask(script, "inside", _cube((0, 0, 0.02))[:5], BOX)
    assert out["route"] == "not_visible"
    assert "subject has" in out["evidence"]


def test_detached_mask_bleed_is_dropped(script):
    # A cube cloud with a strip of edge pixels that took an occluder's depth 20 cm up.
    cube = _cube((0.0, 0.095, 0.015))
    bleed = np.c_[RNG.uniform(0.0, 0.01, (40, 2)), np.full(40, 0.22)]
    out = _ask(script, "near", np.vstack([cube, bleed]), BOX)
    assert out["route"] == "holds"
    assert out["measures"]["subject_points"] < len(cube)


def test_unknown_relation_raises(script):
    with pytest.raises(ValueError):
        _ask(script, "under_the_sea", _cube((0, 0, 0.02)), BOX)


# --- masks + depth from one top-down camera -----------------------------------------------------

H, W, F = 240, 320, 300.0
K = np.array([[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]])
# Camera at (0, 0, 1) looking down: camera x = world x, camera y = -world y.
POSE = {
    "position": {"x": 0.0, "y": 0.0, "z": 1.0},
    "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
}


def _scene(blocks):
    """Depth of a table at z = 0 with axis-aligned blocks (x0, x1, y0, y1, z_top); one mask per block."""
    v, u = np.indices((H, W))
    x = (u - K[0, 2]) / F
    y = -(v - K[1, 2]) / F
    depth = np.ones((H, W))
    masks = []
    for x0, x1, y0, y1, top in blocks:
        d = 1.0 - top
        hit = (x * d > x0) & (x * d < x1) & (y * d > y0) & (y * d < y1) & (d < depth)
        depth[hit] = d
        masks.append(hit)
    masks = [
        np.where(m & (depth == 1.0 - b[4]), 255, 0).astype(np.uint8)
        for m, b in zip(masks, blocks, strict=True)
    ]
    table = np.where(depth == 1.0, 255, 0).astype(np.uint8)
    rgb = np.zeros((H, W, 3), np.uint8)
    obs = {
        "cameras": [
            {
                "name": "overhead",
                "rgb": rgb,
                "depth": depth.astype(np.float32),
                "intrinsics": K,
                "pose": POSE,
            }
        ],
        "arms": [],
    }
    return obs, masks, table


def test_masks_cube_on_table(script):
    obs, (cube,), table = _scene([(0.05, 0.08, 0.05, 0.08, 0.03)])
    out = script.run(
        FakeContext(), "on_top_of", observation=obs, subject_mask=cube, reference_mask=table
    )
    assert out["route"] == "holds", out["evidence"]


def test_masks_bar_hovering_over_a_part_blocks_it(script):
    # An 80 mm part with a 50 mm bar 60 mm over it, straight down the camera's ray: the part
    # shows on both sides of the bar, whose image box covers two thirds of the part's.
    obs, (part, bar), _ = _scene(
        [(-0.04, 0.04, -0.04, 0.04, 0.02), (-0.025, 0.025, -0.05, 0.05, 0.08)]
    )
    out = script.run(
        FakeContext(), "blocking", observation=obs, subject_mask=bar, reference_mask=part
    )
    assert out["route"] == "holds", out["evidence"]
    assert out["measures"]["occlude_ratio"] > 0.5
    rev = script.run(
        FakeContext(), "blocking", observation=obs, subject_mask=part, reference_mask=bar
    )
    assert rev["route"] == "does_not_hold"
    far = script.run(
        FakeContext(),
        "blocking",
        observation=obs,
        subject_mask=bar,
        reference_mask=part,
        close_distance=0.05,
    )
    assert far["route"] == "does_not_hold"


def test_descriptions_go_through_sam3_and_fetch_the_observation(script):
    obs, (cube,), table = _scene([(0.05, 0.08, 0.05, 0.08, 0.03)])
    answers = {"cube": cube, "table": table}
    ctx = FakeContext(
        {
            "robot.get_observation": obs,
            "sam3.segment_text": lambda image, query, max_results=1: {
                "masks": [answers[query]],
                "scores": [0.9],
                "boxes": [None],
            },
        }
    )
    out = script.run(
        ctx, "on", subject_description="cube", reference_description="table", camera_name="top"
    )
    assert out["route"] == "holds"


def test_sam3_finding_nothing_is_not_visible(script):
    obs, (cube,), _ = _scene([(0.05, 0.08, 0.05, 0.08, 0.03)])
    ctx = FakeContext({"sam3.segment_text": {"masks": [], "scores": [], "boxes": []}})
    out = script.run(ctx, "inside", observation=obs, subject_mask=cube, reference_description="box")
    assert out["route"] == "not_visible"
    assert "sam3 found no 'box'" in out["evidence"]


def test_low_score_is_not_visible(script):
    obs, (cube,), table = _scene([(0.05, 0.08, 0.05, 0.08, 0.03)])
    ctx = FakeContext({"sam3.segment_text": {"masks": [table], "scores": [0.1], "boxes": [None]}})
    out = script.run(ctx, "on", observation=obs, subject_mask=cube, reference_description="table")
    assert out["route"] == "not_visible"
