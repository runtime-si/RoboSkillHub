"""Domino state, the RGB-D provider on synthetic depth renders and LEGO lattice pose.

Errors are measured against the pose the renderer was given (independent of the
fit), never against the algorithm's own output. All data is synthetic (the hub's
pose6d ray-caster with the station's depth-noise model)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from _perception_regrasp_helpers import SPEC, render_frame, resting, rgbd_provider

from roboskillhub.tasks.perception_regrasp import (
    SimTruthProvider,
    brick_pose_in_plate,
    domino_state,
    lego_brick,
    pose_error,
    snap_to_lattice,
)
from roboskillhub.tasks.perception_regrasp.frames import make_T, rot_y, rot_z
from roboskillhub.tasks.perception_regrasp.objects import symmetry_rotations
from roboskillhub.tasks.perception_regrasp.providers import backend_objects


@pytest.mark.parametrize(
    "state,vertical,bottom", [("lying", "z", "-z"), ("standing", "x", "+x"), ("on_side", "y", "-y")]
)
def test_domino_state_axes_and_bottom_face(state, vertical, bottom):
    T = resting(state)  # rot_y(90): +x points down; rot_x(90): +y points up
    st = domino_state(T, SPEC, table_z=0.0)
    assert st["state"] == state
    assert st["vertical_axis"] == vertical
    assert st["bottom_face"] == bottom
    assert np.allclose(st["long_axis"], T[:3, 0]) and np.allclose(st["thickness_axis"], T[:3, 2])
    assert (
        st["height_check"]["reader_state"]
        == {"lying": "flat", "standing": "upright", "on_side": "fallen"}[state]
    )


def test_domino_state_is_the_same_for_every_symmetric_equivalent():
    for state in ("lying", "standing", "on_side"):
        T = resting(state)
        for S in symmetry_rotations(SPEC):
            T2 = T.copy()
            T2[:3, :3] = T[:3, :3] @ S
            assert domino_state(T2, SPEC, table_z=0.0)["state"] == state


def test_near_states_and_lifted_tiles_are_uncertain():
    lean = make_T(rot_z(0.4) @ rot_y(math.radians(25)), [0.3, 0, 0.012])
    st = domino_state(lean, SPEC, table_z=0.0)
    assert st["state"] == "uncertain" and "tilted" in st["reason"]
    near = make_T(rot_y(math.radians(4)), [0.3, 0, 0.0055])
    assert domino_state(near, SPEC, table_z=0.0)["state"] == "lying"
    lifted = resting("lying")
    lifted[2, 3] += 0.02
    st = domino_state(lifted, SPEC, table_z=0.0)
    assert st["state"] == "uncertain" and "not resting" in st["reason"]


@pytest.mark.parametrize("camera", ["wrist", "overhead"])
@pytest.mark.parametrize("state", ["lying", "standing", "on_side"])
def test_rgbd_provider_recovers_state_and_pose(camera, state):
    T = resting(state)
    frame, mask = render_frame(T, camera)
    obs = rgbd_provider(frame, mask).observe()[0]
    assert obs.provider == "rgbd_pose6d" and not obs.privileged
    assert obs.visibility in ("visible", "partially_occluded")
    assert obs.attributes["domino"]["state"] == state
    err = pose_error(obs.T_frame_object, T, SPEC)
    # pose6d SKILL.md: wrist < 0.5 deg / 0.3 mm p90; overhead with support 1-5 deg / ~1 mm.
    tol_deg, tol_m = (1.0, 0.001) if camera == "wrist" else (6.0, 0.003)
    assert err["rotation_deg"] < tol_deg and err["translation_m"] < tol_m
    assert 0.0 < obs.confidence <= 1.0
    assert obs.evidence["tool"].startswith("tools/pose6d")


def test_rgbd_leaning_tile_is_refit_without_the_support_plane_and_stays_uncertain():
    T = make_T(rot_z(0.4) @ rot_y(math.radians(25)), [0.30, 0.02, 0.012])
    frame, mask = render_frame(T, "wrist")
    obs = rgbd_provider(frame, mask).observe()[0]
    assert "support_refit" in obs.evidence and obs.evidence["support_plane"] is False
    assert obs.attributes["domino"]["state"] == "uncertain"
    assert pose_error(obs.T_frame_object, T, SPEC)["rotation_deg"] < 3.0


def test_rgbd_occluded_tile_has_no_pose():
    T = resting("lying")
    frame, mask = render_frame(T, "overhead")
    cols = np.nonzero(mask)[1]
    mask = mask.copy()
    mask[:, : int(np.percentile(cols, 85))] = False
    obs = rgbd_provider(frame, mask).observe()[0]
    assert obs.visibility == "occluded" and obs.T_frame_object is None and obs.reason
    assert obs.check_usable(frame.timestamp_s, 1.0)[2] == "object_not_observed"


def test_rgbd_missing_mask_and_stale_frame():
    T = resting("lying")
    frame, mask = render_frame(T, "wrist", t=50.0)
    p = rgbd_provider(frame, mask)
    p.specs["d2"] = SPEC
    o1, o2 = p.observe(["d1", "d2"])
    assert o2.visibility == "not_found" and o2.T_frame_object is None
    assert o1.check_usable(52.0, 1.0)[2] == "stale_observation"
    objs = backend_objects(p, ["d1", "d2"])
    assert set(objs) == {"d1", "_unseen"} and objs["d1"]["source"] == "rgbd_pose6d"
    assert objs["d1"]["quaternion_xyzw"] is not None and objs["d1"]["state_label"] == "lying"


def test_sim_truth_is_always_privileged_and_distinct_from_rgbd():
    T = resting("standing")
    truth = {"timestamp_s": 1.0, "frame_id": "world", "objects": {"d1": {"T_frame_object": T}}}
    o = SimTruthProvider(lambda: truth, {"d1": SPEC}, table_z=0.0).observe()[0]
    assert o.privileged and o.provider == "sim_truth" and o.uncertainty.kind == "simulator_truth"
    frame, mask = render_frame(T, "wrist")
    r = rgbd_provider(frame, mask).observe()[0]
    assert not r.privileged and r.uncertainty.kind == "fit_residual"


def _paint_face(frame, mask, T_obj, pips: bool):
    """A white tile with a divider and pips on its +z face (or blank: face down)."""
    rgb = np.full(mask.shape + (3,), 90, np.uint8)
    rgb[mask] = 235
    if pips:
        L, W, Th = SPEC.size_m
        Tc = np.linalg.inv(frame.T_frame_camera)

        def px(p_obj):
            p = Tc @ T_obj @ np.array([*p_obj, 1.0])
            uvw = frame.K @ p[:3]
            return int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))

        cv2.line(rgb, px((0, -0.4 * W, Th / 2)), px((0, 0.4 * W, Th / 2)), (20, 20, 20), 2)
        r = max(2, int(abs(px((0.002, 0, Th / 2))[0] - px((0, 0, Th / 2))[0])) + 1)
        for p in [(L / 4, 0), (-L / 4 - W / 4, -W / 4), (-L / 4 + W / 4, W / 4)]:
            cv2.circle(rgb, px((p[0], p[1], Th / 2)), r, (20, 20, 20), -1)
    return rgb


@pytest.mark.parametrize("face_up", [True, False])
def test_pip_face_resolves_the_lying_tile_up_down_symmetry(face_up):
    T = resting("lying", yaw=0.1)
    if not face_up:  # pips on the face that touches the table
        T = T @ make_T(np.diag([1.0, -1.0, -1.0]))
    frame, mask = render_frame(T, "wrist")
    frame.rgb = _paint_face(frame, mask, T, pips=face_up)
    obs = rgbd_provider(frame, mask).observe()[0]
    read = obs.evidence["pip_read"]
    assert read["source"].endswith("read_face")
    z_up = obs.T_frame_object[2, 2] > 0
    assert z_up == face_up
    assert obs.attributes["domino"]["bottom_face_resolved"] is True


def test_lego_lattice_round_trip_and_off_grid():
    b = lego_brick((2, 4))
    for cell in [(0, 0, 1, 0), (3, 5, 1, 90), (7, 2, 2, 0), (4, 4, 3, 90)]:
        T = brick_pose_in_plate(b, (16, 16), *cell)
        s = snap_to_lattice(T, b, (16, 16))
        assert s["route"] == "on_grid"
        assert (
            s["cell"]["x"],
            s["cell"]["y"],
            s["cell"]["layer"],
            s["cell"]["rotation_deg"],
        ) == cell
        T180 = T @ make_T(rot_z(math.pi))
        assert (
            snap_to_lattice(T180, b, (16, 16))["cell"] == s["cell"]
        )  # 180 deg twin: same placement
    T = brick_pose_in_plate(b, (16, 16), 3, 5, 1, 0)
    T[0, 3] += 0.004  # half a pitch
    assert snap_to_lattice(T, b, (16, 16))["route"] == "off_grid"
    T = brick_pose_in_plate(b, (16, 16), 15, 0, 1, 0)
    assert snap_to_lattice(T, b, (16, 16))["route"] == "out_of_plate"


def test_rgbd_lego_brick_gets_its_lattice_cell():
    b = lego_brick((2, 4))
    T_plate = make_T(rot_z(0.1), [0.25, -0.05, 0.0])
    T = T_plate @ brick_pose_in_plate(b, (16, 16), 3, 5, 1, 90)
    frame, mask = render_frame(T, "wrist", spec=b)
    obs = rgbd_provider(
        frame, mask, spec=b, oid="b1", lattice={"T_frame_plate": T_plate, "plate_studs": (16, 16)}
    ).observe()[0]
    lat = obs.attributes["lattice"]
    assert lat["route"] == "on_grid" and lat["cell"] == {
        "x": 3,
        "y": 5,
        "layer": 1,
        "rotation_deg": 90,
    }
    assert obs.to_v1()["state_label"] == "seated"
