"""estimating-relative-rotation-and-symmetry: synthetic parts turned by known angles."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "estimating-relative-rotation-and-symmetry"
H, W = 240, 320


@pytest.fixture(scope="module")
def rr(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["relative_rotation"].module


def part(angle_deg, size=(60, 30), mark=None, centre=(160, 120), colour=(200, 40, 40), lit=False):
    """A filled rectangle turned clockwise (image plane) by angle_deg, optionally with a dark mark
    (pips) at an offset along its long axis; ``lit`` adds a shading ramp across the frame."""
    img = np.full((H, W, 3), 140, np.uint8)
    rect = (
        (float(centre[0]), float(centre[1])),
        (float(size[0]), float(size[1])),
        float(angle_deg),
    )
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(img, [box], colour)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [box], 255)
    if mark is not None:
        a = math.radians(angle_deg)
        p = (int(centre[0] + mark * math.cos(a)), int(centre[1] + mark * math.sin(a)))
        cv2.circle(img, p, 5, (20, 20, 20), -1)
        q = (int(centre[0] + 0.5 * mark * math.cos(a)), int(centre[1] + 0.5 * mark * math.sin(a)))
        cv2.circle(img, q, 4, (20, 20, 20), -1)
    if lit:
        ramp = np.linspace(0.8, 1.15, W)[None, :, None]
        img = np.clip(img * ramp, 0, 255).astype(np.uint8)
    return img, mask


def err(a, b, period):
    d = (a - b) % period
    return min(d, period - d)


@pytest.mark.parametrize("turn", [-170.0, -95.0, -30.0, 0.0, 12.5, 60.0, 135.0])
def test_elongated_part_turn_modulo_180(rr, turn):
    ri, rm = part(20)
    ci, cm = part(20 + turn)
    r = rr.run(FakeContext({}), ri, rm, ci, cm)
    assert r["route"] == "rotation_found"
    assert r["outline_order"] == 2 and r["period_deg"] == 180.0 and r["symmetry_order"] == 2
    assert err(r["rotation_deg"], turn, 180.0) < 1.5
    assert -90.0 <= r["rotation_deg"] < 90.0


@pytest.mark.parametrize("turn", [-40.0, -10.0, 17.0, 44.0, 80.0])
def test_square_part_turn_modulo_90_from_the_rectangle(rr, turn):
    ri, rm = part(5, size=(40, 40))
    ci, cm = part(5 + turn, size=(40, 40))
    r = rr.run(FakeContext({}), ri, rm, ci, cm)
    assert r["outline_order"] == 4 and r["period_deg"] == 90.0
    assert err(r["rotation_deg"], turn, 90.0) < 2.0  # a 40 px square rasterised: +-1 px at a corner


@pytest.mark.parametrize("turn", [-150.0, -60.0, 30.0, 120.0, 175.0])
def test_marked_part_resolves_the_half_turn(rr, turn):
    ri, rm = part(10, size=(70, 34), mark=18)
    ci, cm = part(10 + turn, size=(70, 34), mark=18)
    r = rr.run(FakeContext({}), ri, rm, ci, cm, symmetry_order=1)
    assert r["route"] == "rotation_found"
    assert r["period_deg"] == 360.0 and r["symmetry_order"] == 1
    assert err(r["rotation_deg"], turn, 360.0) < 2.0
    assert len(r["scores"]) == 2 and r["scores"][0] - r["scores"][1] > rr.AMBIGUOUS_MARGIN


def test_plain_part_asked_to_resolve_is_ambiguous(rr):
    ri, rm = part(10, size=(70, 34))
    ci, cm = part(10 + 150, size=(70, 34))
    r = rr.run(FakeContext({}), ri, rm, ci, cm, symmetry_order=1)
    assert r["route"] == "symmetric_ambiguous"
    assert r["period_deg"] == 180.0 and r["symmetry_order"] == 2
    assert err(r["rotation_deg"], 150.0, 180.0) < 1.5


def test_shading_does_not_change_the_outline_answer(rr):
    ri, rm = part(30, size=(70, 34))
    ci, cm = part(30 + 100, size=(70, 34), lit=True, centre=(200, 100))
    r = rr.run(FakeContext({}), ri, rm, ci, cm)
    assert r["route"] == "rotation_found" and err(r["rotation_deg"], 100.0, 180.0) < 1.5


def test_symmetry_order_above_the_outline_falls_back_to_it(rr):
    ri, rm = part(0, size=(70, 34))
    ci, cm = part(40, size=(70, 34))
    r = rr.run(FakeContext({}), ri, rm, ci, cm, symmetry_order=4)
    assert r["symmetry_order"] == 2 and r["period_deg"] == 180.0


def test_outline_order_override(rr):
    ri, rm = part(0, size=(44, 40))  # elongation 1.1: square by the shape test
    ci, cm = part(70, size=(44, 40))
    assert rr.run(FakeContext({}), ri, rm, ci, cm)["outline_order"] == 4
    r = rr.run(FakeContext({}), ri, rm, ci, cm, outline_order=2)
    assert r["outline_order"] == 2 and err(r["rotation_deg"], 70.0, 180.0) < 2.0


def test_camera_frames_are_accepted(rr):
    ri, rm = part(0)
    ci, cm = part(45)
    r = rr.run(FakeContext({}), {"rgb": ri}, rm, {"rgb": ci}, cm)
    assert err(r["rotation_deg"], 45.0, 180.0) < 1.5


def test_small_masks_are_low_confidence(rr):
    ri, rm = part(0, size=(6, 4))
    ci, cm = part(30, size=(6, 4))
    r = rr.run(FakeContext({}), ri, rm, ci, cm)
    assert r["route"] == "low_confidence" and "under" in r["reason"]


def test_a_different_part_is_low_confidence_when_resolving(rr):
    ri, rm = part(0, size=(70, 34), mark=18)
    ci = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    _, cm = part(0, size=(70, 34))
    r = rr.run(FakeContext({}), ri, rm, ci, cm, symmetry_order=1)
    assert r["route"] == "low_confidence" and r["period_deg"] == 180.0


def test_bad_inputs_raise(rr):
    ri, rm = part(0)
    with pytest.raises(ValueError, match="differs"):
        rr.run(FakeContext({}), ri, rm[:-1], ri, rm)
    with pytest.raises(ValueError, match="RGB"):
        rr.run(FakeContext({}), ri[:, :, 0], rm, ri, rm)


def test_fold_and_axis_helpers(rr):
    assert rr._fold(100.0, 180.0) == pytest.approx(-80.0)
    assert rr._fold(-90.0, 180.0) == pytest.approx(-90.0)
    _, m = part(30, size=(80, 20))
    a, e, c = rr.axis(m > 0)
    assert err(a, 30.0, 180.0) < 1.0 and e > 3.0 and c == pytest.approx([160, 120], abs=1)


def test_example_declares_every_route(rr):
    import json
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(rr.__file__)))
    ex = json.load(open(os.path.join(root, "examples", "canonical_subgraph.json")))
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert declared == {"rotation_found", "symmetric_ambiguous", "low_confidence"}
