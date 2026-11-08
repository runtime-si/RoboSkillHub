"""computing-brick-pose-from-stud-grid: the cell <-> pose geometry against a
literal transcription of ICL's calc_brick_loc, round trips over random plates,
cells, sizes and yaws, the exits (plate bounds, occupancy, support, finger room,
jaw opening, IK preview), the hand-off to pressing-a-brick-onto-studs, and a
pose6d chain (a rendered brick registered by pose6d, then mapped back to its
cell).
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "computing-brick-pose-from-stud-grid"
ROOT = Path(__file__).resolve().parents[2]
LEVEL = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
PLATE = {"position": {"x": 0.45, "y": -0.05, "z": 0.012}, "rotation": LEVEL}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["brick_pose"].module


def _rot(q):
    w, x, y, z = (float(q[k]) for k in "wxyz")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat(r):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(r).as_quat()
    return {"w": w, "x": x, "y": y, "z": z}


def _m(pose):
    m = np.eye(4)
    m[:3, :3] = _rot(pose["rotation"])
    m[:3, 3] = [pose["position"][k] for k in "xyz"]
    return m


def _pose(m):
    return {
        "position": dict(zip("xyz", map(float, m[:3, 3]), strict=True)),
        "rotation": _quat(m[:3, :3]),
    }


def _p(v):
    return np.array([v[k] for k in "xyz"]) if isinstance(v, dict) else np.asarray(v)


def _rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def _random_plate(rng, tilt_deg=3.0):
    from scipy.spatial.transform import Rotation

    tilt = Rotation.from_rotvec(
        np.r_[rng.normal(size=2), 0.0] / np.sqrt(2) * math.radians(tilt_deg)
    )
    m = np.eye(4)
    m[:3, :3] = _rz(rng.uniform(-180, 180)) @ tilt.as_matrix()
    m[:3, 3] = [rng.uniform(0.3, 0.6), rng.uniform(-0.2, 0.2), rng.uniform(0.0, 0.05)]
    return m


def _bare():
    """No robot tools served: the geometry alone."""
    return FakeContext({})


def _grasp_frame(close_heading_deg, **_):
    h = math.radians(close_heading_deg)
    yv = np.array([math.cos(h), math.sin(h), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    r = np.column_stack([np.cross(yv, z), yv, z])
    return {
        "rotation": _quat(r),
        "rotation_flipped": _quat(r @ np.diag([-1.0, -1.0, 1.0])),
        "close_axis": {"x": yv[0], "y": yv[1], "z": 0.0},
    }


def _icl_calc_brick_loc(plate, plate_w, plate_h, h, w, ori, x, y, z):
    """Lego::calc_brick_loc (src/Lego.cpp L551-584) transcribed line by line."""
    p_len, offset, brick_h = 0.008, 0.0002, 0.0096

    def t(a, b, c):
        m = np.eye(4)
        m[:3, 3] = (a, b, c)
        return m

    z_90 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    brick_offset = t(x * p_len - offset, y * p_len - offset, z * brick_h)
    centre = t((h * p_len - offset) / 2.0, (w * p_len - offset) / 2.0, 0)
    topleft = t(-(plate_w * p_len - offset) / 2.0, -(plate_h * p_len - offset) / 2.0, 0)
    out = plate @ topleft @ brick_offset @ centre
    if ori == 1:
        centre[1, 3] = -centre[1, 3]
        out = plate @ topleft @ brick_offset @ z_90 @ centre
    return out


# -- the upstream geometry ----------------------------------------------------


@pytest.mark.parametrize(
    ("size", "ori", "cell"),
    [
        ((2, 4), 0, (23, 15, 1)),
        ((2, 4), 1, (0, 30, 1)),
        ((1, 6), 1, (10, 3, 3)),
        ((2, 2), 0, (47 - 1, 0, 2)),
    ],
)
def test_matches_icl_calc_brick_loc_with_its_offset_as_grid_bias(script, size, ori, cell):
    plate = np.eye(4)
    plate[:3, :3] = _rz(math.degrees(0.01672))  # ICL's env_setup_stairs.json plate
    plate[:3, 3] = (0.4081, 0.04627, 0.188)
    want = _icl_calc_brick_loc(plate, 48, 48, size[0], size[1], ori, *cell)
    out = script.run(
        _bare(),
        _pose(plate),
        [48, 48],
        list(size),
        cell={"x": cell[0], "y": cell[1], "layer": cell[2], "rotation": 90 * ori},
        system="lego",
        grid_bias=[-0.0002, -0.0002],
    )
    got = _m(out["top_pose"])
    assert np.allclose(got[:3, 3], want[:3, 3], atol=1e-9)
    assert np.allclose(got[:3, :3], want[:3, :3], atol=1e-9)


def test_lego_2x4_at_the_corner_cell(script):
    out = script.run(
        _bare(), PLATE, [8, 8], "2x4", cell={"x": 0, "y": 0, "layer": 1}, system="lego"
    )
    assert out["route"] == "computed"
    # 8 x 8 plate is 64 mm: the corner cell's 2x4 centre is 32 - 8 / 32 - 16 mm in.
    assert np.allclose(
        _p(out["brick_pose"]["position"]), [0.45 - 0.024, -0.05 - 0.016, 0.012 + 0.0048]
    )
    assert np.allclose(_p(out["seat_position"]), [0.426, -0.066, 0.012])
    assert np.allclose(_p(out["top_pose"]["position"])[2], 0.012 + 0.0096)
    assert out["footprint"] == [2, 4] and len(out["cells"]) == 8
    assert out["pose6d_mesh"] == {"box": pytest.approx([0.0158, 0.0318, 0.0096])}
    assert out["support_plane"] == pytest.approx([0, 0, 1, -0.012])
    assert out["seat_yaw_deg"] == pytest.approx(0.0)


def test_rotation_90_swaps_the_footprint_and_keeps_the_anchor_corner(script):
    out = script.run(
        _bare(),
        PLATE,
        [8, 8],
        "2x4",
        cell={"x": 0, "y": 0, "layer": 1, "rotation": 90},
        system="lego",
    )
    assert out["footprint"] == [4, 2]
    assert sorted(out["cells"]) == sorted([[i, j, 1] for i in range(4) for j in range(2)])
    assert np.allclose(_p(out["seat_position"])[:2], [0.45 - 0.016, -0.05 - 0.024])
    assert out["seat_yaw_deg"] == pytest.approx(90.0)
    # Jaws across the short side: the brick's x, now along the plate's y.
    assert out["close_heading_deg"] == pytest.approx(90.0)
    assert out["jaw_width"] == pytest.approx(0.0158)


def test_rotation_180_and_270_occupy_the_same_cells_as_0_and_90(script):
    kw = {"system": "duplo"}
    for a, b in ((0, 180), (90, 270)):
        oa = script.run(
            _bare(), PLATE, [16, 16], [2, 3], cell={"x": 4, "y": 5, "layer": 2, "rotation": a}, **kw
        )
        ob = script.run(
            _bare(), PLATE, [16, 16], [2, 3], cell={"x": 4, "y": 5, "layer": 2, "rotation": b}, **kw
        )
        assert sorted(oa["cells"]) == sorted(ob["cells"])
        assert np.allclose(_p(oa["seat_position"]), _p(ob["seat_position"]))
        assert abs(((ob["seat_yaw_deg"] - oa["seat_yaw_deg"]) % 360) - 180) < 1e-9


def test_plate_thickness_lifts_from_the_plate_underside(script):
    a = script.run(_bare(), PLATE, [8, 8], "2x2", cell={"x": 1, "y": 1}, system="lego")
    b = script.run(
        _bare(), PLATE, [8, 8], "2x2", cell={"x": 1, "y": 1}, system="lego", plate_thickness=0.0032
    )
    assert _p(b["seat_position"])[2] - _p(a["seat_position"])[2] == pytest.approx(0.0032)


def test_custom_pitch_and_height_override_the_preset(script):
    out = script.run(
        _bare(),
        PLATE,
        [4, 4],
        "1x1",
        cell={"x": 0, "y": 0, "layer": 2},
        system="lego",
        stud_pitch=0.01,
        brick_height=0.012,
    )
    assert np.allclose(_p(out["seat_position"]), [0.45 - 0.015, -0.05 - 0.015, 0.012 + 0.012])
    assert out["stud_pitch"] == 0.01 and out["brick_height"] == 0.012


# -- round trips ----------------------------------------------------------------


@pytest.mark.parametrize("system", ["lego", "duplo"])
def test_round_trip_cell_to_pose_to_cell(script, system):
    rng = np.random.default_rng(7)
    sizes = [[1, 1], [1, 2], [2, 2], [2, 3], [2, 4], [1, 6], [4, 2]]
    for _ in range(300):
        plate = _random_plate(rng)
        n = [int(rng.integers(6, 33)), int(rng.integers(6, 33))]
        size = sizes[int(rng.integers(len(sizes)))]
        rot = int(rng.choice([0, 90, 180, 270]))
        fp = size[::-1] if rot % 180 else size
        cell = {
            "x": int(rng.integers(0, n[0] - fp[0] + 1)),
            "y": int(rng.integers(0, n[1] - fp[1] + 1)),
            "layer": int(rng.integers(1, 6)),
            "rotation": rot,
        }
        fwd = script.run(_bare(), _pose(plate), n, size, cell=cell, system=system)
        assert fwd["route"] == "computed"
        for ref, key in (("body_centre", "brick_pose"), ("top", "top_pose")):
            back = script.run(
                _bare(),
                _pose(plate),
                n,
                size,
                observed_pose=fwd[key],
                observed_reference=ref,
                system=system,
            )
            assert back["route"] == "computed", back["detail"]
            assert sorted(back["cells"]) == sorted(fwd["cells"])
            assert back["residual"]["lateral"] < 1e-9 and abs(back["residual"]["z"]) < 1e-9
            assert abs(back["residual"]["yaw_deg"]) < 1e-6
            # The inverse names one of the brick's symmetric twins: same seat.
            assert np.allclose(_p(back["seat_position"]), _p(fwd["seat_position"]), atol=1e-9)
            assert back["cell"]["rotation"] in ((0,) if size[0] == size[1] else (0, 90))


def test_round_trip_through_the_box_flipped_upside_down(script):
    fwd = script.run(
        _bare(), PLATE, [10, 10], "2x4", cell={"x": 3, "y": 2, "layer": 2, "rotation": 90}
    )
    m = _m(fwd["brick_pose"])
    m[:3, :3] = m[:3, :3] @ np.diag([1.0, -1.0, -1.0])  # pose6d may return this twin
    back = script.run(_bare(), PLATE, [10, 10], "2x4", observed_pose=_pose(m))
    assert back["route"] == "computed"
    assert back["cell"] == {"x": 3, "y": 2, "layer": 2, "rotation": 90}


def test_bottom_reference(script):
    fwd = script.run(_bare(), PLATE, [10, 10], "2x2", cell={"x": 3, "y": 2, "layer": 3})
    seat = {"position": fwd["seat_position"], "rotation": fwd["brick_pose"]["rotation"]}
    back = script.run(
        _bare(), PLATE, [10, 10], "2x2", observed_pose=seat, observed_reference="bottom"
    )
    assert back["cell"] == {"x": 3, "y": 2, "layer": 3, "rotation": 0}


def test_small_errors_snap_large_ones_are_off_grid(script):
    fwd = script.run(
        _bare(), PLATE, [10, 10], "2x4", cell={"x": 3, "y": 2, "layer": 1}, system="lego"
    )
    m = _m(fwd["brick_pose"])
    near = m.copy()
    near[:3, 3] += [0.0015, -0.0015, 0.001]
    near[:3, :3] = _rz(4.0) @ near[:3, :3]
    out = script.run(_bare(), PLATE, [10, 10], "2x4", observed_pose=_pose(near), system="lego")
    assert out["route"] == "computed"
    assert out["cell"] == {"x": 3, "y": 2, "layer": 1, "rotation": 0}
    assert out["residual"]["x"] == pytest.approx(0.0015) and out["residual"]["y"] == pytest.approx(
        -0.0015
    )
    assert out["residual"]["yaw_deg"] == pytest.approx(4.0)
    far = m.copy()
    far[:3, 3] += [0.0035, 0.0, 0.0]  # 0.44 pitch: between two cells
    out = script.run(_bare(), PLATE, [10, 10], "2x4", observed_pose=_pose(far), system="lego")
    assert out["route"] == "off_grid" and out["cell"]["x"] == 3
    yawed = m.copy()
    yawed[:3, :3] = _rz(30.0) @ yawed[:3, :3]
    out = script.run(_bare(), PLATE, [10, 10], "2x4", observed_pose=_pose(yawed), system="lego")
    assert out["route"] == "off_grid" and "yawed" in out["detail"]
    tipped = m.copy()
    tipped[:3, :3] = tipped[:3, :3] @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])  # on its side
    out = script.run(_bare(), PLATE, [10, 10], "2x4", observed_pose=_pose(tipped), system="lego")
    assert out["route"] == "off_grid" and "tilted" in out["detail"]


def test_a_2x4_yawed_near_90_is_the_rotated_cell(script):
    fwd = script.run(
        _bare(), PLATE, [12, 12], "2x4", cell={"x": 5, "y": 1, "layer": 1, "rotation": 270}
    )
    m = _m(fwd["brick_pose"])
    m[:3, :3] = _rz(-3.0) @ m[:3, :3]
    out = script.run(_bare(), PLATE, [12, 12], "2x4", observed_pose=_pose(m))
    assert out["route"] == "computed"
    assert out["cell"] == {"x": 5, "y": 1, "layer": 1, "rotation": 90}
    assert out["residual"]["yaw_deg"] == pytest.approx(-3.0)


# -- exits -----------------------------------------------------------------------


def test_out_of_plate(script):
    out = script.run(_bare(), PLATE, [8, 8], "2x4", cell={"x": 7, "y": 0, "layer": 1})
    assert out["route"] == "out_of_plate"
    out = script.run(_bare(), PLATE, [8, 8], "2x4", cell={"x": 0, "y": 5, "layer": 1})
    assert out["route"] == "out_of_plate"
    out = script.run(
        _bare(), PLATE, [8, 8], "2x4", cell={"x": 4, "y": 4, "layer": 1, "rotation": 90}
    )
    assert out["route"] == "computed"
    out = script.run(_bare(), PLATE, [8, 8], "2x4", cell={"x": 0, "y": 0, "layer": 0})
    assert out["route"] == "out_of_plate" and "layer 0" in out["detail"]
    m = _m(PLATE)
    m[:3, 3] += [0.2, 0.0, 0.0096 / 2]
    out = script.run(_bare(), PLATE, [8, 8], "2x2", observed_pose=_pose(m), system="lego")
    assert out["route"] == "out_of_plate"


def test_collides_with_cells_bricks_and_earlier_outputs(script):
    first = script.run(_bare(), PLATE, [16, 16], "2x4", cell={"x": 2, "y": 2, "layer": 1})
    out = script.run(
        _bare(), PLATE, [16, 16], "2x2", cell={"x": 3, "y": 5, "layer": 1}, occupied=[first]
    )
    assert out["route"] == "collides" and out["collisions"] == [[3, 5, 1]]
    out = script.run(
        _bare(), PLATE, [16, 16], "2x2", cell={"x": 3, "y": 6, "layer": 1}, occupied=[first]
    )
    assert out["route"] == "computed"
    out = script.run(
        _bare(),
        PLATE,
        [16, 16],
        "2x2",
        cell={"x": 0, "y": 0, "layer": 1},
        occupied=[[1, 1, 1], {"x": 9, "y": 9, "layer": 1, "brick_size": "2x4", "rotation": 90}],
    )
    assert out["route"] == "collides" and out["collisions"] == [[1, 1, 1]]


def test_unsupported_over_nothing(script):
    base = {"x": 2, "y": 2, "layer": 1, "brick_size": "2x2"}
    out = script.run(
        _bare(), PLATE, [16, 16], "2x2", cell={"x": 6, "y": 6, "layer": 2}, occupied=[base]
    )
    assert out["route"] == "unsupported"
    out = script.run(
        _bare(), PLATE, [16, 16], "2x4", cell={"x": 3, "y": 3, "layer": 2}, occupied=[base]
    )
    assert out["route"] == "computed"  # one stud of overlap holds it


def test_fingers_blocked_on_the_short_side_close_across_the_long_one(script):
    # A 2x4 at rotation 0 spans x 4-5, y 4-7; the short side is along x, so the
    # fingers go down into the columns x=3 and x=6.
    nb = [[3, 5, 1]]
    out = script.run(
        _bare(),
        PLATE,
        [16, 16],
        "2x4",
        cell={"x": 4, "y": 4, "layer": 1},
        occupied=nb,
        system="lego",
    )
    assert out["route"] == "computed" and out["close_axis"] == "long"
    assert out["jaw_width"] == pytest.approx(0.0318) and "other side" in out["detail"]
    assert out["close_heading_deg"] == pytest.approx(90.0)
    out = script.run(
        _bare(),
        PLATE,
        [16, 16],
        "2x4",
        cell={"x": 4, "y": 4, "layer": 1},
        occupied=nb,
        system="lego",
        close_axis="short",
    )
    assert out["route"] == "ungraspable"
    out = script.run(
        _bare(),
        PLATE,
        [16, 16],
        "2x4",
        cell={"x": 4, "y": 4, "layer": 1},
        occupied=[[3, 5, 1], [5, 8, 2]],
        system="lego",
    )
    assert out["route"] == "ungraspable" and "fingers" in out["detail"]
    # A neighbour one layer down does not touch the fingers.
    out = script.run(
        _bare(),
        PLATE,
        [16, 16],
        "2x4",
        cell={"x": 4, "y": 4, "layer": 2},
        occupied=[[3, 5, 1], [4, 4, 1]],
        system="lego",
    )
    assert out["route"] == "computed" and out["close_axis"] == "short"


def test_jaws_too_narrow_for_the_brick(script):
    ctx = FakeContext(
        {"robot.describe_gripper": {"span_m": 0.034, "width_fit": {"at_open_m": 0.034}}}
    )
    out = script.run(ctx, PLATE, [16, 16], "2x2", cell={"x": 0, "y": 0}, system="duplo")
    assert out["route"] == "ungraspable" and "34.0 mm" in out["detail"]
    ctx = FakeContext({"robot.describe_gripper": {"span_m": 0.094}})
    out = script.run(ctx, PLATE, [16, 16], "2x2", cell={"x": 0, "y": 0}, system="duplo")
    assert out["route"] == "computed"


def test_grasp_pose_from_grasp_frame_and_heights(script):
    ctx = FakeContext({"robot.grasp_frame": _grasp_frame})
    out = script.run(
        ctx, PLATE, [16, 16], "2x4", cell={"x": 0, "y": 0, "rotation": 90}, system="lego"
    )
    assert ctx.calls_to("robot.grasp_frame")[0].kwargs == {"close_heading_deg": pytest.approx(90.0)}
    g = _m(out["grasp_pose"])
    assert np.allclose(g[:3, 2], [0, 0, -1]) and np.allclose(g[:3, 1], [0, 1, 0], atol=1e-9)
    top = _p(out["top_pose"]["position"])
    assert g[2, 3] == pytest.approx(top[2] - 0.005 + 0.0102)
    assert np.allclose(g[:2, 3], top[:2])
    a = _m(out["approach_pose"])
    assert a[2, 3] - g[2, 3] == pytest.approx(0.04) and np.allclose(a[:3, :3], g[:3, :3])
    assert _m(out["press_pose"]) == pytest.approx(g)
    assert np.allclose(_m(out["grasp_pose_flipped"])[:3, 1], [0, -1, 0], atol=1e-9)
    assert "plate normal" not in out["detail"]
    one = script.run(ctx, PLATE, [16, 16], "1x4", cell={"x": 0, "y": 0}, system="lego")
    assert _m(one["grasp_pose"])[2, 3] == pytest.approx(
        _p(one["top_pose"]["position"])[2] - 0.006 + 0.0102
    )


def test_tilted_plate_grasps_along_its_normal(script):
    plate = np.eye(4)
    plate[:3, :3] = np.array(
        [[1, 0, 0], [0, math.cos(0.2), -math.sin(0.2)], [0, math.sin(0.2), math.cos(0.2)]]
    )
    plate[:3, 3] = [0.5, 0.0, 0.05]
    ctx = FakeContext({"robot.grasp_frame": _grasp_frame})
    out = script.run(ctx, _pose(plate), [8, 8], "2x2", cell={"x": 2, "y": 2}, system="lego")
    assert ctx.call_count("robot.grasp_frame") == 0
    g = _m(out["grasp_pose"])
    assert np.allclose(g[:3, 2], -plate[:3, 2])
    assert "plate normal" in out["detail"]


def test_unreachable_and_the_flipped_wrist(script):
    ctx = FakeContext({"robot.grasp_frame": _grasp_frame, "robot.solve_ik": {"success": False}})
    out = script.run(
        ctx, PLATE, [8, 8], "2x2", cell={"x": 0, "y": 0}, check_reach=True, system="lego"
    )
    assert out["route"] == "unreachable"

    def ik(pose, **_):
        return {"success": bool(_rot(pose["rotation"])[0, 1] < 0)}  # only the flipped jaws reach

    ctx = FakeContext({"robot.grasp_frame": _grasp_frame, "robot.solve_ik": ik})
    out = script.run(
        ctx, PLATE, [8, 8], "2x2", cell={"x": 0, "y": 0}, check_reach=True, system="lego"
    )
    assert out["route"] == "computed" and "flipped" in out["detail"]
    assert _m(out["grasp_pose"])[0, 1] < 0 and _m(out["grasp_pose_flipped"])[0, 1] > 0


def test_arm_id_only_when_nonzero(script):
    ctx = FakeContext(
        {"robot.grasp_frame": _grasp_frame, "robot.describe_gripper": {"span_m": 0.094}}
    )
    script.run(ctx, PLATE, [8, 8], "2x2", cell={"x": 0, "y": 0}, system="lego")
    assert all("arm_id" not in c.kwargs for c in ctx.calls)
    ctx = FakeContext(
        {"robot.grasp_frame": _grasp_frame, "robot.describe_gripper": {"span_m": 0.094}}
    )
    script.run(ctx, PLATE, [8, 8], "2x2", cell={"x": 0, "y": 0}, system="lego", arm_id=1)
    assert all(c.kwargs.get("arm_id") == 1 for c in ctx.calls)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cell": None},
        {"observed_pose": PLATE},  # both given
        {"system": "megablok"},
        {"close_axis": "diagonal"},
        {"cell": {"x": 0, "y": 0, "rotation": 45}},
        {"cell": {"y": 0}},
        {"brick_size": "2x"},
        {"grasp_depth": 0.03},
    ],
)
def test_bad_inputs_raise(script, kwargs):
    base = {"cell": {"x": 0, "y": 0}, "brick_size": "2x2"}
    base.update(kwargs)
    with pytest.raises(ValueError):
        script.run(_bare(), PLATE, [8, 8], **base)


def test_example_declares_every_route(script):
    ex = json.loads((ROOT / "skills" / BUNDLE / "examples" / "canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert set(script.ROUTES) <= declared


# -- the hand-offs ------------------------------------------------------------------


@pytest.fixture(scope="module")
def press(skills_registry):
    return (
        skills_registry.get("pressing-a-brick-onto-studs").canonical_scripts["press_brick"].module
    )


def test_outputs_plug_into_pressing_a_brick_onto_studs(script, press):
    for system in ("lego", "duplo"):
        for key in ("stud_pitch", "stud_height", "stud_diameter", "brick_height"):
            assert script.PRESETS[system][key] == press.PRESETS[system][key]
    assert script.FOOTPRINT_PLAY == press.FOOTPRINT_PLAY and script.TIP_REACH == press.TIP_REACH
    out = script.run(
        _bare(),
        PLATE,
        [16, 16],
        "2x4",
        cell={"x": 3, "y": 3, "layer": 2, "rotation": 90},
        system="lego",
    )
    # The press reads seat_position as the brick's bottom centre seated, z the
    # stud bases under it; without a held box its hand hangs body + tip reach -
    # GRASP_DEPTH over that: the grasp computed here.
    assert _p(out["seat_position"])[2] == pytest.approx(0.012 + 0.0096)
    hang = out["brick_height"] + press.TIP_REACH - press.GRASP_DEPTH
    assert _m(out["press_pose"])[:3, 3] == pytest.approx(_p(out["seat_position"]) + [0, 0, hang])
    kwargs = {
        k: out[k]
        for k in (
            "seat_position",
            "seat_yaw_deg",
            "brick_studs",
            "system",
            "stud_pitch",
            "stud_height",
            "brick_height",
        )
    }
    import inspect

    assert set(kwargs) <= set(inspect.signature(press.run).parameters)
    # The press's own brick footprint from these inputs is this brick's.
    yaw = math.radians(kwargs["seat_yaw_deg"])
    bx = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    corner = _p(out["seat_position"]) + bx * (out["brick_studs"][0] * out["stud_pitch"] / 2)
    assert corner == pytest.approx(_p(out["seat_position"]) + _m(out["brick_pose"])[:3, 0] * 0.008)


@pytest.fixture(scope="module")
def pose6d(skills_registry):
    pytest.importorskip("scipy")
    assert skills_registry.get("pose6d").kind == "tool"
    return (
        importlib.import_module("gap_skills.tools.pose6d._impl"),
        importlib.import_module("gap_skills.tools.pose6d._synth"),
    )


@pytest.mark.parametrize(("system", "camera"), [("lego", "wrist"), ("duplo", "overhead")])
def test_pose6d_registration_maps_back_to_the_cell(script, pose6d, system, camera):
    """A brick body rendered at a cell (station depth noise), registered by
    pose6d with this skill's mesh, support plane and pose as init, then mapped
    back: the same cell, residual under a quarter pitch."""
    from scipy.ndimage import binary_dilation

    impl, synth = pose6d
    rng = np.random.default_rng(3)
    plate = np.eye(4)
    plate[:3, :3] = _rz(17.0)
    plate[:3, 3] = [0.0, 0.0, 0.0]
    for _ in range(3):
        size = [[2, 4], [2, 2], [1, 4]][int(rng.integers(3))]
        cell = {
            "x": int(rng.integers(0, 6)),
            "y": int(rng.integers(0, 6)),
            "layer": 1,
            "rotation": int(rng.choice([0, 90])),
        }
        fwd = script.run(_bare(), _pose(plate), [12, 12], size, cell=cell, system=system)
        t_true = _m(fwd["brick_pose"])
        if camera == "wrist":
            eye = t_true[:3, 3] + 0.18 * np.array([0.35, -0.2, 0.92])
            cam, k, noise = synth.look_at(eye, t_true[:3, 3]), synth.K_D405, synth.D405
        else:
            cam, k, noise = (
                synth.look_at([0.0, -0.30, 0.85], [0, 0, 0]),
                synth.K_ORBBEC,
                synth.ORBBEC,
            )
        depth, mask = synth.render(
            fwd["pose6d_mesh"], t_true, cam, K=k, noise=noise, seed=int(rng.integers(1 << 30))
        )
        pts = synth.backproject(depth, binary_dilation(mask), k) @ cam[:3, :3].T + cam[:3, 3]
        init = t_true.copy()
        init[:3, 3] += rng.normal(scale=0.002, size=3) * [1, 1, 0]
        r = impl.register(
            pts,
            fwd["pose6d_mesh"],
            init=init,
            viewpoint=cam[:3, 3],
            up=[0, 0, 1],
            support_plane=fwd["support_plane"],
        )
        back = script.run(
            _bare(), _pose(plate), [12, 12], size, observed_pose=_pose(r["T"]), system=system
        )
        assert back["route"] == "computed", back["detail"]
        assert sorted(back["cells"]) == sorted(fwd["cells"])
