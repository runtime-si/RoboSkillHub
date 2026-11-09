"""verify_stable_placement on synthetic observation streams (tier: synthetic_contract_test).

Each case builds what a camera would have reported; the observer gets no hint of the
intended outcome."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from _domino_world import FRAME, GEOMETRY, T0, lying, tile_pose

from roboskillhub.tasks.domino import verify_stable_placement
from roboskillhub.tasks.domino.contract import threshold_profile

TARGET = tile_pose(0.40, 0.08)
N0, N1 = tile_pose(0.40, 0.00), tile_pose(0.40, 0.04)
RELEASE, RETREAT = T0 + 1.0, T0 + 2.0


def sample(
    t,
    target=TARGET,
    n1=N1,
    *,
    sid=None,
    holding=False,
    kind="synthetic_contract_test",
    conf=0.95,
    tcp_z=0.15,
    **target_rec,
):
    objs = {
        "d2": {
            "pose": copy.deepcopy(target),
            "identity_confidence": conf,
            "visible": True,
            **target_rec,
        },
        "d0": {"pose": copy.deepcopy(N0), "identity_confidence": 0.95, "visible": True},
        "d1": {"pose": copy.deepcopy(n1), "identity_confidence": 0.95, "visible": True},
    }
    return {
        "sample_id": sid or f"f{t:.3f}",
        "timestamp_s": t,
        "frame_id": FRAME,
        "source": "synthetic-camera",
        "source_kind": kind,
        "objects": objs,
        "gripper": {"holding": holding, "source": "jaw-gap"},
        "tcp_pose": {
            "frame_id": FRAME,
            "position_m": [0.4, 0.08, tcp_z],
            "quaternion_xyzw": [1, 0, 0, 0],
        },
    }


def stream(**kw):
    return [sample(RELEASE + 0.5, **kw)] + [sample(RETREAT + 0.25 * k, **kw) for k in range(4)]


def req(observations, **over):
    r = {
        "request_id": "v1",
        "frame_id": FRAME,
        "target_id": "d2",
        "geometry": GEOMETRY,
        "target_object_pose": TARGET,
        "support": {
            "frame_id": FRAME,
            "point_m": [0, 0, 0],
            "normal": [0, 0, 1],
            "height_std_m": 0.001,
            "calibration_version": "c1",
        },
        "neighbors_baseline": {
            "d0": {"pose": N0, "timestamp_s": T0 - 1, "source": "synthetic-camera"},
            "d1": {"pose": N1, "timestamp_s": T0 - 1, "source": "synthetic-camera"},
        },
        "events": {"action_start_s": T0, "release_s": RELEASE, "retreat_done_s": RETREAT},
        "observations": observations,
        "thresholds": threshold_profile(),
        "now_s": RETREAT + 1.0,
        "gripper_tip_reach_m": 0.0102,
    }
    r.update(over)
    return r


def test_fresh_stable_stream_succeeds():
    out = verify_stable_placement(req(stream()))
    assert out["status"] == "succeeded", (out["failure"], out["unknowns"])
    assert out["evidence"]["evidence_tier"] == "synthetic_contract_test"
    assert out["evidence"]["window"]["samples"] == 4
    assert out["evidence"]["release"]["confirmed"]


def test_one_frame_repeated_does_not_fill_the_window():
    s = sample(RETREAT + 0.1, sid="same")
    obs = [sample(RELEASE + 0.5)] + [dict(s) for _ in range(4)]
    out = verify_stable_placement(req(obs))
    assert out["status"] == "unverified"
    assert {"observation_repeated", "insufficient_settle_window"} <= set(out["unverified"])


def test_stale_stream_is_not_success():
    out = verify_stable_placement(req(stream(), now_s=RETREAT + 30))
    assert out["status"] == "unverified" and "observation_stale" in out["unverified"]


def test_short_window_is_not_success():
    obs = [sample(RETREAT + 0.05 * k) for k in range(3)]
    out = verify_stable_placement(req(obs))
    assert out["status"] == "unverified" and "insufficient_settle_window" in out["unverified"]


def test_position_only_or_wrong_identity_is_not_success():
    out = verify_stable_placement(req(stream(target=None)))
    assert out["status"] == "unverified" and "orientation_missing" in out["unverified"]
    out = verify_stable_placement(req(stream(conf=0.1)))
    assert out["status"] == "unverified" and "identity_mismatch" in out["unverified"]


def test_unknown_release_is_not_success():
    obs = stream(holding=None, tcp_z=0.05)
    out = verify_stable_placement(req(obs))
    assert out["status"] == "unverified" and "release_not_confirmed" in out["unverified"]
    # The measured TCP well clear of the tile's top is release evidence on its own.
    out = verify_stable_placement(req(stream(holding=None, tcp_z=0.12)))
    assert out["status"] == "succeeded" and out["evidence"]["release"]["by"].startswith(
        "measured_tcp"
    )


def test_gripper_still_holding_fails():
    out = verify_stable_placement(req(stream(holding=True)))
    assert out["status"] == "failed" and out["failure"]["code"] == "release_not_confirmed"


def test_hanging_tile_is_not_resting_on_the_support():
    hung = copy.deepcopy(TARGET)
    hung["position_m"][2] = 0.02
    out = verify_stable_placement(req(stream(target=hung, holding=None)))
    assert out["status"] == "failed" and out["failure"]["code"] == "release_not_confirmed"


def test_right_xyz_but_lying_fails():
    flat = lying(TARGET)
    flat["position_m"] = list(TARGET["position_m"])
    out = verify_stable_placement(req(stream(target=flat)))
    assert out["status"] == "failed" and out["failure"]["code"] == "unstable_after_release"


def test_fell_between_release_and_retreat_is_located():
    obs = [sample(RELEASE + 0.5, target=lying(TARGET))] + [
        sample(RETREAT + 0.25 * k, target=lying(TARGET)) for k in range(4)
    ]
    out = verify_stable_placement(req(obs))
    assert (
        out["failure"]["code"] == "unstable_after_release" and out["failure"]["phase"] == "release"
    )


def test_drift_in_the_window_fails():
    obs = [sample(RELEASE + 0.5)]
    for k in range(4):
        p = copy.deepcopy(TARGET)
        p["position_m"][1] += 0.0015 * k
        obs.append(sample(RETREAT + 0.25 * k, target=p))
    out = verify_stable_placement(req(obs))
    assert out["status"] == "failed" and out["failure"]["code"] == "unstable_after_release"


def test_target_ok_but_neighbour_down_fails_and_locates_retreat():
    obs = [sample(RELEASE + 0.5)] + [sample(RETREAT + 0.25 * k, n1=lying(N1)) for k in range(4)]
    out = verify_stable_placement(req(obs))
    assert out["status"] == "failed"
    assert out["failure"]["code"] == "neighbor_disturbed" and out["failure"]["neighbor_id"] == "d1"
    assert out["failure"]["phase"] == "retreat"


def test_snapshot_diff_report_counts_as_neighbour_evidence():
    out = verify_stable_placement(
        req(stream(), snapshot_diff={"route": "disturbed", "disturbed": ["d0"]})
    )
    assert out["status"] == "failed" and out["failure"]["neighbor_id"] == "d0"


def test_offset_and_yaw_errors():
    off = copy.deepcopy(TARGET)
    off["position_m"][0] += 0.012
    out = verify_stable_placement(req(stream(target=off)))
    assert out["failure"]["code"] == "placement_error"
    flipped = tile_pose(0.40, 0.08, yaw_deg=270.0)
    assert verify_stable_placement(req(stream(target=flipped)))["status"] == "succeeded"
    g360 = {**GEOMETRY, "yaw_symmetry_deg": 360.0}
    assert (
        verify_stable_placement(req(stream(target=flipped), geometry=g360))["failure"]["code"]
        == "placement_error"
    )


def test_future_wrong_frame_and_unlabelled_samples_are_dropped():
    bad = stream()
    for s in bad:
        s["frame_id"] = "camera"
    out = verify_stable_placement(req(bad))
    assert out["status"] == "unverified"
    unlabelled = stream()
    for s in unlabelled:
        s["source_kind"] = None
    assert verify_stable_placement(req(unlabelled))["status"] == "unverified"


def test_simulation_truth_is_labelled_and_can_be_refused():
    obs = stream(kind="simulation_truth")
    out = verify_stable_placement(req(obs))
    assert out["status"] == "succeeded" and out["evidence"]["evidence_tier"] == "simulation_truth"
    assert verify_stable_placement(req(obs, require_sensor_source=True))["status"] == "unverified"


def test_baseline_taken_after_the_action_is_rejected():
    r = req(stream())
    r["neighbors_baseline"]["d0"]["timestamp_s"] = T0 + 5
    out = verify_stable_placement(r)
    assert out["status"] == "rejected"


def test_depth_check_seeing_no_standing_tile_fails_without_a_pose():
    obs = stream(
        target=None,
        depth_upright={"seen": True, "standing": False, "why": "its top is 8 mm over the table"},
    )
    out = verify_stable_placement(req(obs))
    assert out["status"] == "failed" and out["failure"]["code"] == "unstable_after_release"
    # ...but a depth "standing" alone is not an orientation: unverified.
    obs = stream(target=None, depth_upright={"seen": True, "standing": True})
    assert verify_stable_placement(req(obs))["status"] == "unverified"


def test_depth_record_reuses_the_placing_skill_judge():
    from roboskillhub.tasks.domino.depth import depth_upright_record

    f, cz = 600.0, 1.0
    h, w = 480, 640
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    depth = np.full((h, w), cz)
    # a standing tile's top at (0.40, 0.08): 24 mm along x (width, facing +y), 7.5 along y
    cx, cy = 0.40, 0.08
    top = cz - 0.048  # pinhole back-projection at the top's depth
    vv, uu = np.mgrid[0:h, 0:w]
    x, y = (uu - w / 2) * top / f, -(vv - h / 2) * top / f
    depth[(np.abs(x - cx) <= 0.012) & (np.abs(y - cy) <= 0.00375)] = top
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    cam_pose[2, 3] = cz
    obs = {"cameras": {"overhead": {"depth": depth, "intrinsics": k, "pose": cam_pose}}}
    rec = depth_upright_record(obs, TARGET, 0.0, GEOMETRY)
    assert rec["pose"] is None and rec["depth_upright"]["seen"]
    assert rec["depth_upright"]["standing"], rec
    assert rec["depth_upright"]["top"] == pytest.approx(0.048, abs=0.002)
    rec = depth_upright_record(obs, tile_pose(0.30, 0.0), 0.0, GEOMETRY)
    assert rec["depth_upright"]["seen"] and not rec["depth_upright"]["standing"]
    assert math.isfinite(rec["depth_upright"]["top"])
