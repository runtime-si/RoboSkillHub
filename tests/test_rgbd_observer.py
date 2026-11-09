"""Metric/identity/deadline contracts; local synthetic frames, no robot actions."""

from __future__ import annotations

import copy
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import numpy as np
import pytest

from roboskillhub.runtime.contracts import EvidenceMissing, Rejected
from roboskillhub.runtime.rgbd_observer import RGBDPointObserver


def frame(*, stamp=None, depth=2.0, size=48, camera="test-rgbd"):
    rng = np.random.default_rng(73)
    image = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    return {
        "rgb": image,
        "depth_m": np.full((size, size), depth, dtype=np.float32),
        "K": np.array([[100.0, 0, 10], [0, 200.0, 10], [0, 0, 1]]),
        "timestamp_s": time.time() if stamp is None else stamp,
        "camera": camera,
        "source": "test_acquisition",
    }


class Camera:
    def __init__(self):
        self.frames = queue.Queue()
        self.read_started = threading.Event()
        self.timeouts = []
        self.closed = False

    def read_frame(self, *, timeout_s):
        self.read_started.set()
        self.timeouts.append(timeout_s)
        try:
            return self.frames.get(timeout=timeout_s)
        except queue.Empty as error:
            raise TimeoutError("test camera deadline") from error

    def close(self):
        self.closed = True


class SamePixels:
    def __init__(self, *, lost_first=False):
        self.calls = []
        self.lost_first = lost_first

    def track(self, frames, points):
        self.calls.append(points.copy())
        visible = np.ones(len(points), dtype=bool)
        if self.lost_first and len(self.calls) == 1:
            visible[0] = False
        return {
            "tracks": [points.tolist(), points.tolist()],
            "visible": [np.ones(len(points), dtype=bool).tolist(), visible.tolist()],
        }


def observer(*, seed=None, points=None, camera=None, **kwargs):
    return RGBDPointObserver(
        camera=camera or Camera(),
        seed_frame=seed or frame(),
        target_points=points or {"named_target": [[20, 20], [28, 28]]},
        frame_id="calibrated_left_base",
        T_base_from_camera=kwargs.pop("T_base_from_camera", np.eye(4)),
        **kwargs,
    )


def test_projection_uses_depth_intrinsics_rigid_transform_and_fixed_centroid():
    source = frame()
    transform = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float)
    obs = observer(
        seed=source,
        points={"explicit_cube_surface": [[12, 14], [14, 18]]},
        T_base_from_camera=transform,
    )
    result = obs._records["explicit_cube_surface"]
    assert result["position_m"] == pytest.approx([0.94, 2.06, 5.0])
    assert result["timestamp_s"] == source["timestamp_s"]
    assert result["frame_id"] == "calibrated_left_base"
    assert result["source"] == "calibrated_rgbd_lk_surface_points"
    assert result["seed_count"] == 2
    assert result["identity_source"] == "explicit_seed_frame_target_assignment"
    assert not {"holding", "grasped", "quaternion_xyzw"} & result.keys()
    source["depth_m"][:] = 9
    transform[:3, 3] = 99
    assert obs._records["explicit_cube_surface"]["position_m"] == pytest.approx([0.94, 2.06, 5.0])


@pytest.mark.parametrize(
    "transform", [np.eye(3), np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1]), np.full((4, 4), np.nan)]
)
def test_nonrigid_or_invalid_calibration_rejected(transform):
    with pytest.raises(Rejected, match="rigid"):
        observer(T_base_from_camera=transform)


def test_frame_must_be_named_and_calibrated_extrinsics_explicit():
    with pytest.raises(Rejected, match="frame_id"):
        RGBDPointObserver(
            camera=Camera(),
            seed_frame=frame(),
            target_points={"x": [[20, 20]]},
            frame_id="",
            T_base_from_camera=np.eye(4),
        )


@pytest.mark.parametrize("mutation", ["zero_center", "too_few", "edge_spread", "negative", "nan"])
def test_invalid_seed_depth_cannot_be_filled_from_neighbouring_background(mutation):
    source = frame()
    if mutation == "zero_center":
        source["depth_m"][20, 20] = 0
    elif mutation == "too_few":
        source["depth_m"][19:22, 19:22] = 0
        source["depth_m"][20, 20] = 2
    elif mutation == "edge_spread":
        source["depth_m"][19, 19] = 1
    elif mutation == "negative":
        source["depth_m"][20, 20] = -1
    else:
        source["depth_m"][20, 20] = np.nan
    with pytest.raises((Rejected, EvidenceMissing)):
        observer(seed=source, points={"surface": [[20, 20]]})


@pytest.mark.parametrize("mutation", ["singular_K", "stale", "future", "camera_missing"])
def test_bad_intrinsics_clock_or_camera_are_rejected(mutation):
    source = frame()
    if mutation == "singular_K":
        source["K"] = np.array([[1.0, 1, 0], [1, 1, 0], [0, 0, 1]])
    elif mutation == "stale":
        source["timestamp_s"] -= 2
    elif mutation == "future":
        source["timestamp_s"] += 1
    else:
        source.pop("camera")
    with pytest.raises(EvidenceMissing):
        observer(seed=source)


def test_lost_seed_removes_whole_target_without_redefining_or_reinitializing_it():
    seed = frame(stamp=time.time() - 0.04)
    obs = observer(
        seed=seed, points={"two_point_surface": [[20, 20], [28, 28]], "other_surface": [[32, 32]]}
    )
    tracker = SamePixels(lost_first=True)
    obs._tracker = tracker
    next_frame = frame(stamp=seed["timestamp_s"] + 0.02)
    obs._update(next_frame)
    assert "two_point_surface" not in obs._records
    assert set(obs._records) == {"other_surface"}
    obs._update(frame(stamp=seed["timestamp_s"] + 0.04))
    assert set(obs._records) == {"other_surface"}
    assert len(tracker.calls[0]) == 3 and len(tracker.calls[1]) == 2
    assert not np.array_equal(tracker.calls[1][0], [20, 20])
    assert obs._records["other_surface"]["seed_count"] == 1


def test_depth_loss_is_permanent_even_when_pixel_depth_recovers():
    seed = frame(stamp=time.time() - 0.04)
    obs = observer(seed=seed, points={"surface": [[20, 20]]})
    tracker = SamePixels()
    obs._tracker = tracker
    missing = frame(stamp=seed["timestamp_s"] + 0.02)
    missing["depth_m"][20, 20] = 0
    obs._update(missing)
    assert obs._records == {}
    obs._update(frame(stamp=seed["timestamp_s"] + 0.04))
    assert obs._records == {} and len(tracker.calls) == 1


def test_metric_jump_gate_rejects_false_track_depth_change():
    seed = frame(stamp=time.time() - 0.04)
    obs = observer(seed=seed)
    obs._tracker = SamePixels()
    obs._update(frame(stamp=seed["timestamp_s"] + 0.02, depth=2.2))
    assert obs._records == {}


@pytest.mark.parametrize("mutation", ["camera", "K", "shape", "backwards", "gap"])
def test_tracking_continuity_changes_fail_instead_of_reseeding(mutation):
    seed = frame(stamp=time.time() - 0.1)
    obs = observer(seed=seed, max_frame_gap_s=0.05)
    obs._tracker = SamePixels()
    changed = frame(stamp=seed["timestamp_s"] + 0.02)
    if mutation == "camera":
        changed["camera"] = "other-camera"
    elif mutation == "K":
        changed["K"][0, 0] += 1
    elif mutation == "shape":
        changed = frame(stamp=changed["timestamp_s"], size=64)
    elif mutation == "backwards":
        changed["timestamp_s"] = seed["timestamp_s"] - 0.01
    else:
        changed["timestamp_s"] = seed["timestamp_s"] + 0.06
    with pytest.raises(EvidenceMissing):
        obs._update(changed)


def test_actual_lk_translation_projects_to_metric_surface_motion():
    import cv2

    seed = frame(stamp=time.time() - 0.05, size=96)
    seed["rgb"] = cv2.GaussianBlur(seed["rgb"], (5, 5), 0.7)
    obs = observer(seed=seed, points={"surface": [[38, 38], [50, 50]]})
    later = copy.deepcopy(seed)
    later["timestamp_s"] += 0.05
    later["rgb"] = cv2.warpAffine(
        seed["rgb"],
        np.array([[1, 0, 1], [0, 1, 0]], dtype=float),
        (96, 96),
        borderMode=cv2.BORDER_REFLECT,
    )
    before = np.asarray(obs._records["surface"]["position_m"])
    obs._update(later)
    after = np.asarray(obs._records["surface"]["position_m"])
    assert after - before == pytest.approx([0.02, 0, 0], abs=0.001)
    assert obs._records["surface"]["timestamp_s"] == later["timestamp_s"]


def test_provider_requires_start_and_cannot_restart_after_close():
    obs = observer()
    with pytest.raises(EvidenceMissing, match="start"):
        obs(timeout_s=0.02)
    obs.close()
    with pytest.raises(Rejected, match="once"):
        obs.start()


def test_provider_waits_for_a_post_call_frame_and_preserves_acquisition_timestamp():
    cam = Camera()
    obs = observer(camera=cam, max_frame_gap_s=0.2)
    obs._tracker = SamePixels()
    with obs, ThreadPoolExecutor(max_workers=1) as pool:
        assert cam.read_started.wait(0.1)
        pending = pool.submit(obs, timeout_s=0.15)
        with pytest.raises(FutureTimeout):
            pending.result(timeout=0.02)
        source = frame()
        cam.frames.put(source)
        result = pending.result(timeout=0.1)
        assert result["named_target"]["timestamp_s"] == source["timestamp_s"]
        result["named_target"]["position_m"][0] = 999
        assert obs._records["named_target"]["position_m"][0] != 999
    assert not obs._worker.is_alive() and not cam.closed


def test_cached_future_timestamp_is_not_a_new_frame():
    cam = Camera()
    seed = frame(stamp=time.time() + 0.05)
    obs = observer(camera=cam, seed=seed, max_frame_gap_s=0.1)
    with obs:
        with pytest.raises(EvidenceMissing, match="deadline"):
            obs(timeout_s=0.02)


def test_duplicate_camera_timestamp_does_not_refresh_evidence():
    cam = Camera()
    seed = frame()
    obs = observer(camera=cam, seed=seed, max_frame_gap_s=0.1)
    obs._tracker = SamePixels()
    with obs:
        cam.frames.put(copy.deepcopy(seed))
        with pytest.raises(EvidenceMissing, match="deadline"):
            obs(timeout_s=0.02)
        assert obs._generation == 0


def test_callback_never_returns_a_frame_after_its_deadline():
    cam = Camera()
    obs = observer(camera=cam, max_frame_gap_s=0.2)
    obs._tracker = SamePixels()
    with obs, ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(obs, timeout_s=0.04)
        with pytest.raises(FutureTimeout):
            future.result(timeout=0.01)
        # Model scheduler/lock contention: new data is available when the waiter
        # resumes, but the caller's budget has already expired.
        with obs._condition:
            time.sleep(0.05)
            obs._update(frame())
        with pytest.raises(EvidenceMissing, match="deadline"):
            future.result(timeout=0.1)


def test_frame_and_records_publish_atomically_even_with_spurious_wakeup():
    cam = Camera()
    obs = observer(camera=cam, max_frame_gap_s=0.2)
    obs._tracker = SamePixels()
    building, release = threading.Event(), threading.Event()
    original = obs._records_for

    def slow_records(record):
        building.set()
        assert release.wait(1)
        return original(record)

    obs._records_for = slow_records
    with obs, ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(obs, timeout_s=0.15)
        with pytest.raises(FutureTimeout):
            pending.result(timeout=0.01)
        source = frame()
        cam.frames.put(source)
        try:
            assert building.wait(0.1)
            with obs._condition:
                obs._condition.notify_all()
            with pytest.raises(FutureTimeout):
                pending.result(timeout=0.02)
        finally:
            release.set()
        assert pending.result(timeout=0.1)["named_target"]["timestamp_s"] == source["timestamp_s"]


def test_worker_updates_while_robot_action_thread_is_blocked():
    cam = Camera()
    obs = observer(camera=cam, max_frame_gap_s=0.2)
    obs._tracker = SamePixels()
    blocked, release = threading.Event(), threading.Event()

    def robot_action():
        blocked.set()
        assert release.wait(1)

    with obs, ThreadPoolExecutor(max_workers=2) as pool:
        action = pool.submit(robot_action)
        assert blocked.wait(0.1)
        observation = pool.submit(obs, timeout_s=0.15)
        with pytest.raises(FutureTimeout):
            observation.result(timeout=0.01)
        cam.frames.put(frame())
        try:
            assert observation.result(timeout=0.1)["named_target"]
            assert not action.done()
        finally:
            release.set()
        action.result(timeout=0.1)


def test_worker_camera_failure_is_visible_to_consumers():
    obs = observer(camera=Camera(), max_frame_gap_s=0.02)
    with obs:
        with pytest.raises(EvidenceMissing, match="test camera deadline"):
            obs(timeout_s=0.15)


def test_close_wakes_waiters_without_closing_the_shared_camera():
    cam = Camera()
    obs = observer(camera=cam, max_frame_gap_s=0.03).start()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(obs, timeout_s=1)
        assert cam.read_started.wait(0.1)
        obs.close()
        with pytest.raises(EvidenceMissing):
            pending.result(timeout=0.1)
    assert not obs._worker.is_alive() and not cam.closed
    obs.close()


def test_noncooperative_camera_worker_is_reported_instead_of_silently_leaked():
    entered, release = threading.Event(), threading.Event()

    class BlockedCamera:
        def read_frame(self, *, timeout_s):
            entered.set()
            assert release.wait(1)
            return frame()

    obs = observer(camera=BlockedCamera(), max_frame_gap_s=0.02).start()
    try:
        assert entered.wait(0.1)
        with pytest.raises(EvidenceMissing, match="worker did not stop"):
            obs.close()
        assert obs._generation == 0
    finally:
        release.set()
        obs._worker.join(timeout=1)
    assert not obs._worker.is_alive() and obs._generation == 0
