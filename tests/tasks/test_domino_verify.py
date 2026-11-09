"""verify_stable_placement on synthetic rsh-task/1 Observation streams (tier:
synthetic_contract_test). Each case builds what a camera would have reported; the
observer gets no hint of the intended outcome."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from _domino_world import FRAME, GEOMETRY, SOURCE, T0, lying, observation, tile_pose

from roboskillhub.tasks.domino import SKILLS, verify_stable_placement

TARGET = tile_pose(0.40, 0.08)
N0, N1 = tile_pose(0.40, 0.00), tile_pose(0.40, 0.04)
RELEASE, RETREAT = T0 + 1.0, T0 + 2.0


def frame(t, target=TARGET, n1=N1, **target_kw):
    return [
        observation("d2", target, t, **target_kw),
        observation("d0", N0, t),
        observation("d1", n1, t),
    ]


def stream(**kw):
    return [
        r for t in [RELEASE + 0.5] + [RETREAT + 0.25 * k for k in range(4)] for r in frame(t, **kw)
    ]


def req(observations, *, holding=False, tcp_z=0.15, **over):
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
        "neighbors_baseline": [observation("d0", N0, T0 - 1), observation("d1", N1, T0 - 1)],
        "events": {"action_start_s": T0, "release_s": RELEASE, "retreat_done_s": RETREAT},
        "observations": observations,
        "gripper_states": []
        if holding is None
        else [{"holding": holding, "timestamp_s": RELEASE + 0.1, "source": "jaw-gap"}],
        "tcp_states": [
            {
                "pose": {
                    "frame_id": FRAME,
                    "position_m": [0.4, 0.08, tcp_z],
                    "quaternion_xyzw": [1, 0, 0, 0],
                },
                "timestamp_s": RETREAT,
                "source": "move",
            }
        ],
        "now_s": RETREAT + 1.0,
        "gripper_tip_reach_m": 0.0102,
    }
    r.update(over)
    return r


def test_fresh_stable_stream_succeeds():
    out = verify_stable_placement(req(stream()))
    assert out["status"] == "succeeded", (out["failure"], out["unknowns"])
    assert out["contract_version"] == "rsh-task/1" and out["kind"] == "tool"
    assert out["evidence"]["postcondition"]["passed"] and out["observations"]
    assert out["evidence"]["evidence_tier"] == "synthetic_contract_test"
    assert out["evidence"]["window"]["usable_samples"] == 4
    assert out["evidence"]["release"]["confirmed"]


def test_registered_tool_entry():
    entry = next(s for s in SKILLS if s.skill_id == "domino.verify_stable_placement")
    assert entry.kind == "tool" and entry.requires == frozenset()
    assert entry.run(req(stream()))["status"] == "succeeded"


def test_one_frame_repeated_does_not_fill_the_window():
    obs = frame(RELEASE + 0.5) + [r for _ in range(4) for r in frame(RETREAT + 0.1)]
    out = verify_stable_placement(req(obs))
    assert out["status"] == "unverified"
    assert {"observation_repeated", "insufficient_settle_window"} <= set(out["unverified"])


def test_stale_stream_is_not_success():
    out = verify_stable_placement(req(stream(), now_s=RETREAT + 30))
    assert out["status"] == "unverified" and out["failure_type"] == "stale_observation"


def test_short_window_is_not_success():
    obs = [r for k in range(3) for r in frame(RETREAT + 0.05 * k)]
    out = verify_stable_placement(req(obs))
    assert out["status"] == "unverified" and "insufficient_settle_window" in out["unverified"]


def test_position_only_or_wrong_identity_is_not_success():
    obs = stream()
    for r in obs:
        if r["object_id"] == "d2":
            r["quaternion_xyzw"], r["source"] = None, "rgbd_centroid"
    out = verify_stable_placement(req(obs))
    assert out["status"] == "unverified" and "orientation_missing" in out["unverified"]
    out = verify_stable_placement(req(stream(confidence=0.1)))
    assert out["status"] == "unverified" and out["failure_type"] == "perception.low_confidence"
    sensor = stream(confidence=None, source="rgbd_pose")
    assert verify_stable_placement(req(sensor))["failure"]["code"] == "identity_mismatch"


def test_occluded_target_is_not_success():
    out = verify_stable_placement(req(stream(visibility="occluded")))
    assert out["status"] == "unverified" and out["failure_type"] == "object_not_observed"


def test_unknown_release_is_not_success():
    out = verify_stable_placement(req(stream(), holding=None, tcp_z=0.05))
    assert out["status"] == "unverified" and "release_not_confirmed" in out["unverified"]
    # The measured TCP well clear of the tile's top is release evidence on its own.
    out = verify_stable_placement(req(stream(), holding=None, tcp_z=0.12))
    assert out["status"] == "succeeded" and out["evidence"]["release"]["by"].startswith(
        "measured_tcp"
    )


def test_gripper_still_holding_fails():
    out = verify_stable_placement(req(stream(), holding=True))
    assert out["status"] == "failed" and out["failure_type"] == "release_failed"


def test_hanging_tile_is_not_resting_on_the_support():
    hung = copy.deepcopy(TARGET)
    hung["position_m"][2] += 0.02
    out = verify_stable_placement(req(stream(target=hung), holding=None))
    assert out["status"] == "failed" and out["failure_type"] == "domino.not_resting_on_support"


def test_right_xyz_but_lying_fails():
    flat = lying(TARGET)
    flat["position_m"] = list(TARGET["position_m"])
    out = verify_stable_placement(req(stream(target=flat)))
    assert out["status"] == "failed" and out["failure_type"] == "domino.toppled_after_release"


def test_fell_between_release_and_retreat_is_located():
    out = verify_stable_placement(req(stream(target=lying(TARGET))))
    assert (
        out["failure"]["code"] == "unstable_after_release" and out["failure"]["phase"] == "release"
    )


def test_drift_in_the_window_fails():
    obs = frame(RELEASE + 0.5)
    for k in range(4):
        p = copy.deepcopy(TARGET)
        p["position_m"][1] += 0.0015 * k
        obs += frame(RETREAT + 0.25 * k, target=p)
    out = verify_stable_placement(req(obs))
    assert out["status"] == "failed" and out["failure"]["code"] == "unstable_after_release"


def test_target_ok_but_neighbour_down_fails_and_locates_retreat():
    obs = frame(RELEASE + 0.5) + [
        r for k in range(4) for r in frame(RETREAT + 0.25 * k, n1=lying(N1))
    ]
    out = verify_stable_placement(req(obs))
    assert out["status"] == "failed" and out["failure_type"] == "domino.neighbour_disturbed"
    assert out["failure"]["neighbor_id"] == "d1" and out["failure"]["phase"] == "retreat"


def test_snapshot_diff_report_counts_as_neighbour_evidence():
    out = verify_stable_placement(
        req(stream(), snapshot_diff={"route": "disturbed", "disturbed": ["d0"]})
    )
    assert out["status"] == "failed" and out["failure"]["neighbor_id"] == "d0"


def test_offset_and_yaw_errors():
    off = copy.deepcopy(TARGET)
    off["position_m"][0] += 0.012
    out = verify_stable_placement(req(stream(target=off)))
    assert out["failure_type"] == "domino.pose_out_of_tolerance"
    flipped = tile_pose(0.40, 0.08, yaw_deg=270.0)
    assert verify_stable_placement(req(stream(target=flipped)))["status"] == "succeeded"
    g360 = {**GEOMETRY, "yaw_symmetry_deg": 360.0}
    assert (
        verify_stable_placement(req(stream(target=flipped), geometry=g360))["failure"]["code"]
        == "placement_error"
    )


def test_wrong_frame_future_and_unlabelled_records_are_dropped():
    bad = stream()
    for r in bad:
        r["frame_id"] = "camera"
    assert verify_stable_placement(req(bad))["status"] == "unverified"
    unlabelled = stream()
    for r in unlabelled:
        del r["privileged"]
    assert verify_stable_placement(req(unlabelled))["status"] == "unverified"
    future = stream()
    for r in future:
        r["timestamp_s"] += 100
    assert verify_stable_placement(req(future))["status"] == "unverified"


def test_privileged_truth_is_labelled_and_can_be_refused():
    obs = stream(source="simulation_truth", privileged=True)
    out = verify_stable_placement(req(obs))
    assert out["status"] == "succeeded" and out["privileged_inputs"] is True
    assert "simulation_truth" in out["evidence"]["evidence_tiers"]
    assert verify_stable_placement(req(obs, require_unprivileged=True))["status"] == "unverified"


def test_baseline_taken_after_the_action_is_rejected():
    r = req(stream())
    r["neighbors_baseline"][0]["timestamp_s"] = T0 + 5
    assert verify_stable_placement(r)["status"] == "rejected"


def test_thresholds_only_tighten():
    assert (
        verify_stable_placement(req(stream(), thresholds_override={"tilt_tolerance_rad": 1.0}))[
            "status"
        ]
        == "rejected"
    )
    off = copy.deepcopy(TARGET)
    off["position_m"][0] += 0.005
    assert verify_stable_placement(req(stream(target=off)))["status"] == "succeeded"
    tight = verify_stable_placement(
        req(stream(target=off), thresholds_override={"position_tolerance_m": 0.003})
    )
    assert tight["failure"]["code"] == "placement_error"


def test_state_label_fall_without_a_pose():
    out = verify_stable_placement(
        req(
            stream(
                target=None, state_label="lying", source="depth_top_check:overhead", confidence=None
            )
        )
    )
    assert out["status"] == "failed" and out["failure"]["code"] == "unstable_after_release"
    # ...but "standing" without an orientation is not an upright proof.
    out = verify_stable_placement(req(stream(target=None, state_label="standing", confidence=None)))
    assert out["status"] == "unverified"


def test_depth_observation_reuses_the_placing_skill_judge():
    from roboskillhub.tasks.domino.depth import depth_upright_observation

    f, cz, h, w = 600.0, 1.0, 480, 640
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    depth = np.full((h, w), cz)
    top = cz - 0.048  # pinhole back-projection at the top's depth
    vv, uu = np.mgrid[0:h, 0:w]
    x, y = (uu - w / 2) * top / f, -(vv - h / 2) * top / f
    depth[(np.abs(x - 0.40) <= 0.012) & (np.abs(y - 0.08) <= 0.00375)] = top
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    cam_pose[2, 3] = cz
    obs = {"cameras": {"overhead": {"depth": depth, "intrinsics": k, "pose": cam_pose}}}
    g = {"height_m": 0.048, "width_m": 0.024, "thickness_m": 0.0075}
    rec = depth_upright_observation(obs, "d2", TARGET, 0.0, g, timestamp_s=T0)
    assert (
        rec["quaternion_xyzw"] is None
        and rec["confidence"] is None
        and rec["state_label"] == "standing"
    )
    assert rec["depth_check"]["top"] == pytest.approx(0.048, abs=0.002)
    rec = depth_upright_observation(obs, "dX", tile_pose(0.30, 0.0), 0.0, g, timestamp_s=T0)
    assert rec["state_label"] == "lying" and math.isfinite(rec["depth_check"]["top"])
    assert rec["source"].startswith("depth_top_check") and SOURCE not in rec["source"]


def test_assembly_eval_samples_adapter():
    from roboskillhub.tasks.domino.adapters import from_eval_samples

    def sample(t, contacts):
        return {
            "t": t,
            "stamp": t,
            "seq": int(t * 10),
            "source": "simulation_truth",
            "frame": FRAME,
            "objects": {
                k: {"position": p["position_m"], "quat_xyzw": p["quaternion_xyzw"]}
                for k, p in (("d2", TARGET), ("d0", N0), ("d1", N1))
            },
            "contacts": contacts,
        }

    samples = [sample(RELEASE + 0.5, {"d2": ["table"]})] + [
        sample(RETREAT + 0.25 * k, {"d2": ["table"]}) for k in range(4)
    ]
    records, grip = from_eval_samples(samples, target_id="d2")
    out = verify_stable_placement(req(records, holding=None, tcp_z=0.05, gripper_states=grip))
    assert out["status"] == "succeeded" and out["privileged_inputs"]
    assert out["evidence"]["release"]["by"] == "gripper:simulation_truth:contacts"
    held = [sample(RELEASE + 0.5, {"d2": ["table", "gripper_pad_l"]})] + samples[1:]
    records, grip = from_eval_samples(held, target_id="d2")
    out = verify_stable_placement(req(records, holding=None, gripper_states=grip))
    assert out["status"] == "failed" and out["failure_type"] == "release_failed"
