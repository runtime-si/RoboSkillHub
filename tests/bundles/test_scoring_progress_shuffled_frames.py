"""scoring-progress-shuffled-frames: shuffling, per-label parsing, VOC and routing with a fake VLM."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext


@pytest.fixture(scope="module")
def gvl(skills_registry):
    return (
        skills_registry.get("scoring-progress-shuffled-frames")
        .canonical_scripts["score_shuffled"]
        .module
    )


def frames(n=5):
    return [np.full((8, 8, 3), 10 * i, np.uint8) for i in range(n)]


def reply_for(gvl, chrono, seed=0, bold=False):
    order = [int(i) for i in np.random.default_rng(seed).permutation(len(chrono))]
    lab = "**Frame {}**:" if bold else "Frame {}:"
    return "\n".join(
        f"{lab.format(k + 1)} Task Completion Percentages: {chrono[i]}%"
        for k, i in enumerate(order)
    )


@pytest.mark.parametrize(
    ("chrono", "route"),
    [
        ([0, 20, 40, 70, 95], "complete"),
        ([0, 20, 40, 60, 70], "progress_increasing"),
        ([30, 10, 40, 20, 30], "stalled"),
        ([0, 30, 80, 60, 20], "regressed"),
    ],
)
def test_routes_from_time_ordered_ratings(gvl, chrono, route):
    vlm = FakeContext({"vlm.query": {"text": reply_for(gvl, chrono)}})
    out = gvl.run(vlm, frames=frames(), instruction="stack the brick", seed=0)
    assert out["route"] == route and out["progress"] == [float(v) for v in chrono]


def test_images_are_start_then_shuffled_and_listed_in_the_prompt(gvl):
    calls = []

    def q(**kw):
        calls.append(kw)
        return {"text": reply_for(gvl, [0, 10, 20, 30, 40], bold=True)}

    out = gvl.run(FakeContext({"vlm.query": q}), frames=frames(), instruction="x", seed=0)
    imgs = calls[0]["images"]
    assert len(imgs) == 6 and imgs[0][0, 0, 0] == 0
    assert [int(a[0, 0, 0]) // 10 for a in imgs[1:]] == out["order"]
    assert (
        "Image 1: Initial robot scene" in calls[0]["prompt"]
        and "Image 6: Frame 5:" in calls[0]["prompt"]
    )
    assert out["voc"] == pytest.approx(1.0)


def test_missing_ratings_and_failures_are_no_verdict(gvl):
    assert (
        gvl.run(
            FakeContext({"vlm.query": {"text": "Frame 1: 50%"}}), frames=frames(), instruction="x"
        )["route"]
        == "no_verdict"
    )
    assert gvl.run(FakeContext({}), frames=frames(1), instruction="x")["route"] == "no_verdict"

    def boom(**kw):
        raise RuntimeError("no key")

    assert (
        "vlm.query failed"
        in gvl.run(FakeContext({"vlm.query": boom}), frames=frames(), instruction="x")["reason"]
    )


def test_voc_matches_spearman_with_ties(gvl):
    assert gvl.value_order_correlation([0, 10, 10, 30]) == pytest.approx(0.9486833, abs=1e-6)
    assert np.isnan(gvl.value_order_correlation([5, 5, 5]))
