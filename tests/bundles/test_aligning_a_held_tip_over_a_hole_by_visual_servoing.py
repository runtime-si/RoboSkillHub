import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills/aligning-a-held-tip-over-a-hole-by-visual-servoing/scripts/align_tip.py"
)
spec = importlib.util.spec_from_file_location("align_tip", SCRIPT)
at = importlib.util.module_from_spec(spec)
spec.loader.exec_module(at)


def test_hole_centre_from_face_and_deep_points():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.04, 0.04, size=(40000, 2))
    true_c = np.array([0.003, -0.002])
    inside = np.linalg.norm(xy - true_c, axis=1) < 0.009
    off_block = np.linalg.norm(xy, axis=1) > 0.025  # the table past the holder's edge, deeper too
    z = np.where(inside, -0.03, np.where(off_block, -0.04, 0.0))
    keep = ~(inside & (xy[:, 0] > 0.006))  # part of the hole hidden
    pts = np.c_[xy, z][keep]
    h = at.hole_centre(pts, np.zeros(2), 0.009, 0.0)
    assert h["found"] and np.linalg.norm(h["centre_xy"] - true_c) < 0.0015


def test_foot_of_a_vertical_cylinder():
    t = np.eye(4)
    t[:3, :3] = np.diag([1.0, -1.0, -1.0])  # approach down
    t[:3, 3] = [0.0, 0.0, 0.12]
    s = np.linspace(0.01, 0.1, 60)
    ang = np.linspace(-1.2, 1.2, 30)
    r = 0.008
    off = np.array([0.004, 0.0])
    pts = np.array(
        [[off[0] + r * np.cos(a), off[1] + r * np.sin(a), 0.12 - si] for si in s for a in ang]
    )
    foot = at.held_foot(pts, t, r, 0.11, 0.0, np.array([0.3, 0.0, 0.3]))
    assert foot["found"] and np.linalg.norm(foot["foot_xy"] - off) < 0.0015


def test_surface_around_reads_table_when_off_the_holder():
    rng = np.random.default_rng(1)
    xy = rng.uniform(-0.04, 0.04, size=(20000, 2))
    on_holder = np.linalg.norm(xy - [0.03, 0.0], axis=1) < 0.02
    pts = np.c_[xy, np.where(on_holder, 0.0, -0.04)]
    assert abs(at.surface_around(pts, np.array([-0.02, 0.0]), 0.009) + 0.04) < 1e-6
    assert abs(at.surface_around(pts, np.array([0.03, 0.0]), 0.004)) < 1e-6


def test_foot_of_a_tilted_cylinder():
    t = np.eye(4)
    t[:3, :3] = np.diag([1.0, -1.0, -1.0])
    t[:3, 3] = [0.0, 0.0, 0.12]
    tilt = np.radians(10.0)
    axis = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    top = np.array([0.0, 0.0, 0.12])
    e1 = np.array([np.cos(tilt), 0.0, np.sin(tilt)])
    e2 = np.array([0.0, 1.0, 0.0])
    r = 0.008
    pts = np.array(
        [
            top + axis * s + r * (np.cos(a) * e1 + np.sin(a) * e2)
            for s in np.linspace(0.01, 0.11, 80)
            for a in np.linspace(-1.2, 1.2, 30)
        ]
    )
    foot_true = top + axis * (0.12 / np.cos(tilt))
    foot = at.held_foot(pts, t, r, 0.12, 0.0, np.array([0.4, 0.0, 0.3]))
    assert foot["found"] and np.linalg.norm(foot["foot_xy"] - foot_true[:2]) < 0.0015


def test_shadow_without_points_is_not_a_hole():
    rng = np.random.default_rng(2)
    xy = rng.uniform(-0.04, 0.04, size=(40000, 2))
    true_c = np.array([0.006, 0.0])
    inside = np.linalg.norm(xy - true_c, axis=1) < 0.009
    shadow = (
        np.linalg.norm(xy - [-0.012, 0.0], axis=1) < 0.007
    )  # the part hides the face here: no points
    z = np.where(inside, -0.03, 0.0)
    keep = ~shadow
    h = at.hole_centre(np.c_[xy, z][keep], np.array([-0.01, 0.0]), 0.009, 0.0)
    assert h["found"] and np.linalg.norm(h["centre_xy"] - true_c) < 0.0015
