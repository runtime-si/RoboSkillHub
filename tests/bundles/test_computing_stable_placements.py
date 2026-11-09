"""computing-stable-placements: rests of boxes and clouds against the analytic tipping angles."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "computing-stable-placements"
DOMINO = (0.048, 0.024, 0.0075)
BRICK = (0.0318, 0.0158, 0.0114)


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["compute_rests"].module


def _q(axis, deg):
    a = math.radians(deg) / 2
    v = np.asarray(axis, dtype=float) / np.linalg.norm(axis) * math.sin(a)
    return {"w": math.cos(a), "x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _obb(size, center=(0.3, 0.1, None), q=None):
    lx, ly, lz = size
    cz = lz / 2 if center[2] is None else center[2]
    return {
        "center": {"x": center[0], "y": center[1], "z": cz},
        "extent": {"x": lx / 2, "y": ly / 2, "z": lz / 2},
        "orientation": q or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def _analytic(size, label):
    h = dict(zip("xyz", size, strict=True))
    up = h[label[1]]
    others = [h[a] for a in "xyz" if a != label[1]]
    return math.degrees(math.atan2(min(others) / 2, up / 2)), up / 2


def _box_surface(size, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    half = np.asarray(size) / 2
    p = rng.uniform(-1, 1, (n, 3)) * half
    k = rng.integers(0, 3, n)
    p[np.arange(n), k] = np.sign(p[np.arange(n), k]) * half[k]
    return p


@pytest.mark.parametrize("size", [DOMINO, BRICK])
def test_box_rests_match_the_analytic_tipping_angle(script, size):
    out = script.run(FakeContext({}), object_obb=_obb(size))
    assert out["route"] == "placements" and out["n_rests"] == 6
    assert sorted(r["label"] for r in out["rests"]) == ["+x", "+y", "+z", "-x", "-y", "-z"]
    for r in out["rests"]:
        tip, com_h = _analytic(size, r["label"])
        assert r["tipping_deg"] == pytest.approx(tip, abs=0.01)
        assert r["com_height_m"] == pytest.approx(com_h, abs=1e-5)
        assert r["center_height_m"] == pytest.approx(com_h, abs=1e-5)
    tips = [r["tipping_deg"] for r in out["rests"]]
    assert tips == sorted(tips, reverse=True)
    assert out["best_tipping_deg"] == tips[0]


def test_domino_numbers(script):
    out = script.run(FakeContext({}), object_obb=_obb(DOMINO))
    by = {r["label"]: r for r in out["rests"]}
    assert by["-z"]["tipping_deg"] == pytest.approx(72.65, abs=0.01)
    assert by["+y"]["tipping_deg"] == pytest.approx(17.35, abs=0.01)
    assert by["+x"]["tipping_deg"] == pytest.approx(8.88, abs=0.01)
    assert out["current_index"] == 0 and out["current_stable"]
    assert out["rests"][out["current_index"]]["label"] == "-z"


def test_turns_and_rotation_put_the_face_down(script):
    out = script.run(FakeContext({}), object_obb=_obb(BRICK))
    for r in out["rests"]:
        n = np.asarray(r["face_normal_world"])
        R = np.asarray(r["rotation_to_rest"])
        assert np.allclose(R @ n, [0, 0, -1], atol=1e-6)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
        expected = math.degrees(math.acos(np.clip(-n[2], -1, 1)))
        assert r["turn_deg"] == pytest.approx(expected, abs=0.01)
    by = {r["label"]: r for r in out["rests"]}
    assert (
        by["-z"]["turn_deg"] == 0.0
        and by["+z"]["turn_deg"] == 180.0
        and by["+x"]["turn_deg"] == 90.0
    )


def test_rotated_obb_reports_world_normals_and_current_rest(script):
    q = _q([0, 1, 0], 90)  # local +x now points down
    out = script.run(FakeContext({}), object_obb=_obb(DOMINO, center=(0.3, 0.1, 0.024), q=q))
    cur = out["rests"][out["current_index"]]
    assert cur["label"] == "+x" and cur["turn_deg"] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(cur["face_normal_world"], [0, 0, -1], atol=1e-6)
    assert cur["tipping_deg"] == pytest.approx(8.88, abs=0.01)


def test_off_centre_com_changes_the_end_rests(script):
    out = script.run(FakeContext({}), object_obb=_obb(DOMINO), com=[0.31, 0.1, 0.00375])
    by = {r["label"]: r for r in out["rests"]}
    assert by["+x"]["tipping_deg"] == pytest.approx(
        math.degrees(math.atan2(0.00375, 0.014)), abs=0.01
    )
    assert by["-x"]["tipping_deg"] == pytest.approx(
        math.degrees(math.atan2(0.00375, 0.034)), abs=0.01
    )
    assert by["+x"]["center_height_m"] == pytest.approx(0.024, abs=1e-6)
    assert by["+x"]["com_height_m"] == pytest.approx(0.014, abs=1e-6)


def test_com_outside_a_face_removes_that_rest(script):
    # COM past the box along x: the flat and side faces lose it, the far end still holds it
    out = script.run(
        FakeContext({}), object_obb=_obb((0.04, 0.02, 0.01)), com=[0.3 + 0.03, 0.1, 0.005]
    )
    assert [r["label"] for r in out["rests"]] == ["-x"]
    # and also above it: no face holds it
    out = script.run(
        FakeContext({}), object_obb=_obb((0.04, 0.02, 0.01)), com=[0.3 + 0.03, 0.1, 0.017]
    )
    assert out["route"] == "no_stable_pose" and out["n_rests"] == 0
    assert out["best"] is None and out["best_pose"] is None


def test_min_margin_and_surface_filters(script):
    out = script.run(FakeContext({}), object_obb=_obb(DOMINO), min_margin_deg=10)
    assert {r["label"] for r in out["rests"]} == {"-z", "+z", "-y", "+y"}
    out = script.run(FakeContext({}), object_obb=_obb(DOMINO), min_margin_deg=80)
    assert out["route"] == "no_stable_pose" and "6 stable face(s); 6 under 80" in out["reason"]
    # flat: footprint = half-diagonal 26.8 mm; on its end: 12.4 mm
    out = script.run(FakeContext({}), object_obb=_obb(DOMINO), surface_half_extents=[0.02, 0.05])
    assert {r["label"] for r in out["rests"]} == {"+x", "-x"}
    assert all(r["footprint_radius_m"] <= 0.02 for r in out["rests"])


def test_goal_pose_rests_the_reference_point_at_its_height(script):
    goal = {"x": 0.4, "y": -0.1, "z": 0.02}
    out = script.run(FakeContext({}), object_obb=_obb(DOMINO), goal_position=goal)
    for r in out["rests"]:
        p = r["pose"]["position"]
        assert (p["x"], p["y"]) == (0.4, -0.1)
        assert p["z"] == pytest.approx(0.02 + r["center_height_m"], abs=1e-9)
        qn = np.array([r["pose"]["rotation"][k] for k in "wxyz"])
        assert np.linalg.norm(qn) == pytest.approx(1.0, abs=1e-6)
    assert out["best_pose"] == out["rests"][0]["pose"]


def test_rest_rotation_quaternion_turns_the_face_down(script):
    out = script.run(
        FakeContext({}), object_obb=_obb(BRICK, q=_q([1, 1, 0], 37)), goal_position=[0, 0, 0]
    )
    for r in out["rests"]:
        R = script.quat_to_matrix(r["pose"]["rotation"])
        R_obj = script.quat_to_matrix(_q([1, 1, 0], 37))
        n_local = R_obj.T @ np.asarray(r["face_normal_world"])
        assert np.allclose(R @ n_local, [0, 0, -1], atol=1e-5)


def test_cloud_of_a_box_surface_finds_its_six_faces(script):
    pts = _box_surface((0.04, 0.02, 0.01)) + [0.3, 0.0, 0.005]
    out = script.run(FakeContext({}), points={"points": pts.astype(np.float32)}, closing="none")
    assert out["route"] == "placements" and out["n_rests"] == 6
    tips = sorted(r["tipping_deg"] for r in out["rests"])
    for got, want in zip(tips, [14.04, 14.04, 26.57, 26.57, 63.43, 63.43], strict=True):
        assert got == pytest.approx(want, abs=1.5)
    assert out["com"]["z"] == pytest.approx(0.005, abs=5e-4)


def test_tilted_cloud_has_no_current_rest(script):
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    pts = _box_surface((0.04, 0.02, 0.01)) @ R.T + [0.3, 0.0, 0.05]
    out = script.run(FakeContext({}), points=pts)
    assert out["n_rests"] == 6 and out["current_index"] == -1 and not out["current_stable"]
    turns = sorted(r["turn_deg"] for r in out["rests"])
    assert turns == pytest.approx([30, 60, 90, 90, 120, 150], abs=1.5)


def _top_view(size, table_z=0.0, seed=2, flying=200):
    """Top face plus mixed pixels spread outside the footprint at low heights."""
    rng = np.random.default_rng(seed)
    lx, ly, lz = size
    top = np.c_[rng.uniform(-lx / 2, lx / 2, (1500, 2)) * [1, ly / lx], np.full(1500, table_z + lz)]
    ang = rng.uniform(0, 2 * np.pi, flying)
    edge = np.c_[
        (lx / 2 + 0.006) * np.cos(ang),
        (ly / 2 + 0.006) * np.sin(ang),
        rng.uniform(0.001, lz * 0.4, flying),
    ]
    return np.vstack([top, edge + [0, 0, table_z]])


def test_prism_closing_rebuilds_a_box_from_a_top_view(script):
    size = (0.076, 0.1, 0.0245)
    out = script.run(FakeContext({}), points=_top_view(size, table_z=-0.033), support_z=-0.033)
    flat = out["rests"][out["current_index"]]
    tip, com_h = _analytic(size, "-z")
    assert out["current_stable"] and flat["turn_deg"] == pytest.approx(0.0, abs=1e-6)
    assert flat["com_height_m"] == pytest.approx(com_h, abs=1e-3)
    assert flat["tipping_deg"] == pytest.approx(tip, abs=2.0)
    assert out["com"]["z"] == pytest.approx(-0.033 + com_h, abs=1e-3)
    assert out["n_rests"] == 6


def test_drop_closing_keeps_the_mixed_pixels(script):
    size = (0.076, 0.1, 0.0245)
    prism = script.run(FakeContext({}), points=_top_view(size), support_z=0.0, closing="prism")
    drop = script.run(FakeContext({}), points=_top_view(size), support_z=0.0, closing="drop")
    assert drop["route"] == "placements"
    # the mixed pixels widen the dropped hull's base, pulling its COM down
    assert drop["com"]["z"] < prism["com"]["z"]


def test_bad_inputs_raise(script):
    ctx = FakeContext({})
    with pytest.raises(ValueError):
        script.run(ctx)
    with pytest.raises(ValueError):
        script.run(ctx, object_obb=_obb(DOMINO), points=np.zeros((10, 3)))
    with pytest.raises(ValueError):
        script.run(ctx, object_obb=_obb((0.02, 0.0, 0.01)))
    with pytest.raises(ValueError):
        script.run(ctx, points=np.zeros((10, 3)), closing="none")
    with pytest.raises(ValueError):
        script.run(ctx, points=np.zeros((10, 2)))
    with pytest.raises(ValueError):
        script.run(ctx, points=_top_view(DOMINO), support_z=0.0, closing="melt")


def test_volume_centroid_ignores_point_density(script):
    pts = _box_surface((0.04, 0.02, 0.01))
    dense_end = pts[pts[:, 0] > 0.015]
    lopsided = np.vstack([pts] + [dense_end] * 5)
    assert np.allclose(script.hull_centroid(lopsided), 0.0, atol=5e-4)
    assert lopsided.mean(axis=0)[0] > 0.005
