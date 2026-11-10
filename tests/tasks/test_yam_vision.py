"""yam_vision: the overhead camera placement, the symmetric-fit canonicalisation, the station flag,
and (with MuJoCo and an offscreen GL) the first image of a seeded YAM domino world against truth."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from roboskillhub.tasks.assembly_eval import yam_vision as yv


def _rot_z(a):
    return yv._rot_z(a)


def test_canonical_puts_height_up_and_keeps_the_facing_axis():
    R_true = _rot_z(0.3)
    flipped = R_true @ np.diag([1.0, -1.0, -1.0])  # the box fit's half-turn about x: +z down
    T = np.eye(4)
    T[:3, :3] = flipped
    out = yv._canonical(T, R_true[:, 0])
    assert out[:3, :3] == pytest.approx(R_true, abs=1e-9)
    T[:3, :3] = R_true @ np.diag([-1.0, -1.0, 1.0])  # half-turn about z: +x reversed
    assert yv._canonical(T, R_true[:, 0])[:3, :3] == pytest.approx(R_true, abs=1e-9)
    # no previous facing axis: any upright equivalent
    assert yv._canonical(T, None)[2, 2] == pytest.approx(1.0)


def test_quaternion_round_trip():
    R = _rot_z(1.1) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    assert yv._quat_to_R(yv._R_to_quat(R)) == pytest.approx(R, abs=1e-9)


def test_overhead_camera_maps_the_station_calibration_into_the_scene():
    physics = SimpleNamespace(
        cfg=SimpleNamespace(
            mount=SimpleNamespace(base_xy=(0.0, 0.0), base_over_table_m=0.021, yaw=0.0)
        ),
        spec=SimpleNamespace(table=SimpleNamespace(top_z=0.0)),
    )
    cam = yv.overhead_camera(physics)
    T = cam["T_world_camera"]
    # real base (x 0.290, y 0.037, z 0.909) -> model (y, -x, z + base height)
    assert T[:3, 3] == pytest.approx([0.036629, -0.29002, 0.908602 + 0.021], abs=1e-6)
    assert T[:3, :3] @ T[:3, :3].T == pytest.approx(np.eye(3), abs=1e-5)
    assert T[2, 2] < -0.8  # looking down
    assert cam["K"][1, 1] == pytest.approx(461.52) and cam["size"] == (640, 480)


def test_vision_rejects_bad_settings():
    with pytest.raises(ValueError, match="segmenter"):
        yv.YamVision(segmenter="sam")
    with pytest.raises(ValueError, match="detector"):
        yv.YamVision(segmenter="detector")
    with pytest.raises(ValueError, match="cameras"):
        yv.YamVision(cameras=("left_eye",))


def _world_or_skip(**kw):
    pytest.importorskip("mujoco")
    pytest.importorskip("cv2")
    from roboskillhub.tasks.assembly_eval.yam_world import yam_domino_world

    return yam_domino_world(0, **kw)


def test_station_keeps_truth_by_default_and_needs_a_provider_for_vision():
    w = _world_or_skip()
    st = w["station"].get_state(timeout_s=5)
    assert st["objects"]["tile_0"]["source"] == "simulation_truth"
    assert w["privileged_allowed"] is True
    with pytest.raises(ValueError, match="vision provider"):
        _world_or_skip(observations="vision")


def test_first_image_finds_every_rack_tile_within_3_mm():
    try:
        vision = yv.YamVision()
        w = _world_or_skip(observations="vision", vision=vision)
    except Exception as exc:  # no offscreen GL context on this machine
        if "GL" in str(exc) or "context" in str(exc).lower() or "display" in str(exc).lower():
            pytest.skip(f"no offscreen rendering: {exc}")
        raise
    assert w["privileged_allowed"] is False
    st = w["station"].get_state(timeout_s=5)
    truth = w["station"].physics.truth()["objects"]
    tiles = [k for k in st["objects"] if k.startswith("tile_")]
    assert sorted(tiles) == sorted(truth)
    for tid in tiles:
        o = st["objects"][tid]
        assert o["source"] == "rgbd_pose6d" and o["privileged"] is False
        assert np.linalg.norm(np.asarray(o["position_m"]) - truth[tid]["position"]) < 0.003
        R = yv._quat_to_R(o["quaternion_xyzw"])
        assert R[2, 2] > math.cos(math.radians(2))  # canonical: height axis up
    s = vision.summary()
    assert s["images"] == 1 and s["missing_rate"] == 0.0
