"""judging-assembly-steps-by-fused-checks: routing per step type from the component checks' routes."""

from __future__ import annotations

import pytest
from gap.testing import FakeContext

BUNDLE = "judging-assembly-steps-by-fused-checks"


@pytest.fixture(scope="module")
def fuse(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["fuse_checks"].module


def place(fuse, **kw):
    return fuse.run(FakeContext({}), "place", **kw)


def test_place_done_needs_depth_and_pull_up(fuse):
    out = place(
        fuse,
        seat_route="seated",
        height_error_m=0.0001,
        pull_route="connected",
        brick_rise_m=0.0004,
    )
    assert out["route"] == "done" and out["step_type"] == "place"


def test_brick_rise_overrides_a_lag_based_not_connected(fuse):
    out = place(
        fuse,
        seat_route="seated",
        height_error_m=0.0001,
        pull_route="not_connected",
        brick_rise_m=0.0004,
    )
    assert out["route"] == "done"
    out = place(
        fuse,
        seat_route="seated",
        height_error_m=0.0001,
        pull_route="not_connected",
        brick_rise_m=0.0025,
    )
    assert out["route"] == "uncertain" and "not held under the pull-up" in out["reason"]


def test_both_geometric_checks_failing_is_failed_with_the_seat_reason(fuse):
    out = place(
        fuse,
        seat_route="high_by_stud",
        height_error_m=0.0017,
        brick_rise_m=0.0027,
        pull_route="connected",
    )
    assert out["route"] == "failed" and "high_by_stud" in out["reason"]


def test_height_error_over_the_threshold_is_not_seated(fuse):
    out = place(
        fuse,
        seat_route="seated",
        height_error_m=0.0009,
        brick_rise_m=0.0003,
        pull_route="connected",
    )
    assert out["route"] == "uncertain"
    out = place(
        fuse,
        seat_route="seated",
        height_error_m=0.0009,
        brick_rise_m=0.0003,
        pull_route="connected",
        max_height_error_m=0.001,
    )
    assert out["route"] == "done"


@pytest.mark.parametrize("vlm", ["succeeded", "complete", ""])
def test_a_camera_judge_never_breaks_a_tie_towards_done(fuse, vlm):
    out = place(
        fuse,
        seat_route="seated",
        height_error_m=0.0001,
        brick_rise_m=0.003,
        pull_route="connected",
        vlm_route=vlm,
    )
    assert out["route"] == "uncertain"


def test_a_camera_judge_breaks_a_tie_towards_failed(fuse):
    out = place(
        fuse,
        seat_route="not_seated",
        height_error_m=0.002,
        brick_rise_m=0.0003,
        pull_route="connected",
        vlm_route="not_yet",
    )
    assert out["route"] == "failed" and "camera judge" in out["reason"]


def test_one_geometric_check_alone_cannot_pass(fuse):
    assert place(fuse, brick_rise_m=0.0003, pull_route="connected")["route"] == "uncertain"
    assert place(fuse, seat_route="seated", height_error_m=0.0001)["route"] == "uncertain"
    assert place(fuse, seat_route="tilted")["route"] == "failed"


def test_no_geometric_check_is_uncertain_even_if_the_vlm_says_done(fuse):
    out = place(fuse, vlm_route="succeeded")
    assert out["route"] == "uncertain" and "camera judge alone" in out["reason"]


@pytest.mark.parametrize("rel", ["stalled_jammed", "stalled_on_object", "held"])
def test_jaws_that_did_not_open_fail_the_place(fuse, rel):
    out = place(
        fuse,
        release_grip_route=rel,
        seat_route="seated",
        height_error_m=0.0001,
        brick_rise_m=0.0003,
    )
    assert out["route"] == "failed" and "did not open" in out["reason"]


def test_rynnvalue_vetoes_a_done(fuse):
    kw = dict(
        seat_route="seated", height_error_m=0.0001, pull_route="connected", brick_rise_m=0.0003
    )
    assert place(fuse, rynn_remaining_s=2.0, **kw)["route"] == "done"
    out = place(fuse, rynn_remaining_s=4.0, **kw)
    assert out["route"] == "uncertain" and "RynnValue" in out["reason"]


@pytest.mark.parametrize(
    ("kw", "route"),
    [
        ({"grip_route": "stalled_on_object"}, "done"),
        ({"grip_route": "stalled_on_object", "scene_diff_route": "changed_as_expected"}, "done"),
        ({"grip_route": "reached"}, "failed"),
        ({"grip_route": "unexpected_width"}, "failed"),
        ({"held": False}, "failed"),
        ({"grip_route": "stalled_on_object", "scene_diff_route": "unchanged"}, "uncertain"),
        ({"grip_route": "stalled_on_object", "vlm_route": "not_yet"}, "uncertain"),
        ({"scene_diff_route": "unchanged"}, "failed"),
        ({}, "uncertain"),
    ],
)
def test_pick_routes(fuse, kw, route):
    out = fuse.run(FakeContext({}), "pick", **kw)
    assert out["route"] == route and out["step_type"] == "pick"


def test_step_type_aliases_and_errors(fuse):
    assert fuse.run(FakeContext({}), "press", seat_route="tilted")["step_type"] == "place"
    with pytest.raises(ValueError):
        fuse.run(FakeContext({}), "topple")
