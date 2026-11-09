# SPDX-License-Identifier: Apache-2.0
"""Trusted station factory joining the real YAM bridge to live RGB-D observations.

Target identity is deliberately supplied by a configured semantic selector, not
invented here. This factory creates no detector, grasps, motor session, or robot
coordinate transform beyond the explicitly calibrated camera-to-base matrix.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import time
from pathlib import Path

from .contracts import EvidenceMissing, Limits, Rejected, number
from .rgbd_observer import RGBDPointObserver
from .yam_tcp import CameraClient, YamTcpBackend


def _selector(reference: str):
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise Rejected("target_selector must be a trusted importable module:function")
    module_name, function_name = reference.split(":")
    if not module_name or not function_name.isidentifier():
        raise Rejected("target_selector must be a trusted importable module:function")
    try:
        function = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc:
        raise Rejected(f"cannot import configured target_selector {reference!r}") from exc
    if not callable(function):
        raise Rejected("configured target_selector must be callable")
    return function


def _calibration(
    path: str | Path, *, frame_id: str, declared_frame: str | None
) -> tuple[dict, str]:
    filename = Path(path).expanduser().resolve(strict=True)
    if not filename.is_file() or filename.stat().st_size > 1 << 20:
        raise Rejected("calibration_path must be a JSON file no larger than 1 MiB")
    raw = filename.read_bytes()
    try:
        calibration = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise Rejected("calibration_path must contain valid JSON") from exc
    if not isinstance(calibration, dict) or calibration.get("calibrated") is not True:
        raise Rejected("camera calibration must explicitly declare calibrated: true")
    if "T_base_from_camera" not in calibration:
        raise Rejected("camera calibration requires T_base_from_camera")
    file_frame = calibration.get("frame_id")
    if file_frame is not None and (not isinstance(file_frame, str) or not file_frame.strip()):
        raise Rejected("calibration frame_id must be a nonempty string when present")
    if declared_frame is not None and (
        not isinstance(declared_frame, str) or not declared_frame.strip()
    ):
        raise Rejected("calibration_frame_id must explicitly name the calibrated robot base")
    if file_frame is None and declared_frame is None:
        raise Rejected(
            "legacy calibration has no frame_id; explicitly set calibration_frame_id in the station config"
        )
    if file_frame is not None and declared_frame is not None and file_frame != declared_frame:
        raise Rejected("calibration file frame_id and station calibration_frame_id disagree")
    if (file_frame or declared_frame) != frame_id:
        raise Rejected("camera calibration frame must equal limits.frame_id; no frame is guessed")
    return calibration, hashlib.sha256(raw).hexdigest()


def _fresh_seed(timestamp, max_age_s: float) -> None:
    stamp = number(timestamp, "seed acquisition timestamp_s", 1e-6, 1e12)
    age = time.time() - stamp
    if age > max_age_s or age < -0.1:
        raise EvidenceMissing("selector seed frame is stale or its clock is not synchronized")


class _ObservedBackend:
    """Own only the observer lifecycle; explicit robot calls delegate unchanged."""

    def __init__(self, backend: YamTcpBackend, observer: RGBDPointObserver, metadata: dict):
        self._backend, self._observer, self._metadata = backend, observer, metadata

    def describe(self) -> dict:
        return {**self._backend.describe(), "object_observer": copy.deepcopy(self._metadata)}

    def get_state(self, *, timeout_s: float) -> dict:
        return self._backend.get_state(timeout_s=timeout_s)

    def move_pose(self, pose: dict, *, speed_m_s: float, timeout_s: float, linear: bool) -> dict:
        return self._backend.move_pose(
            pose, speed_m_s=speed_m_s, timeout_s=timeout_s, linear=linear
        )

    def set_gripper(self, width_m: float, *, timeout_s: float) -> dict:
        return self._backend.set_gripper(width_m, timeout_s=timeout_s)

    def stop(self) -> None:
        self._backend.stop()

    def begin_invocation(self, *, timeout_s: float) -> dict:
        return self._backend.begin_invocation(timeout_s=timeout_s)

    def end_invocation(self) -> dict:
        return self._backend.end_invocation()

    def close(self) -> None:
        # Observers have no authority over the motor/session lifecycle.
        self._observer.close()


def create_backend(
    *,
    limits: Limits,
    port: int,
    camera_port: int,
    calibration_path: str | Path,
    target_selector: str,
    host: str = "127.0.0.1",
    camera_host: str = "127.0.0.1",
    calibration_frame_id: str | None = None,
    selector_kwargs: dict | None = None,
    expected_kind: str = "real",
    seed_timeout_s: float = 5.0,
    connect_timeout_s: float = 3.0,
    stop_timeout_s: float = 3.0,
) -> _ObservedBackend:
    """YAML station factory for real YAM + calibrated live surface-point feedback.

    ``target_selector`` is a trusted station-level ``module:function``, never an
    agent request. It is called as ``selector(seed_frame, **selector_kwargs)`` and
    returns exactly ``{target_id: [[x,y], ...]}`` in that frame's RGB pixel grid.
    The selector must establish semantic identity (for example from an existing
    segmentation model). It receives a protected copy of the exact live frame;
    it must not fetch another image and return unrelated pixel coordinates.
    Selection that outlives the seed's freshness bound is rejected, not retimed.

    Calibration JSON must contain ``calibrated: true`` and the measured metre
    ``T_base_from_camera``. Its ``frame_id`` must match ``limits.frame_id``.
    Legacy calibration files such as YAM workcell's omit that key, so their frame
    must be declared explicitly via ``calibration_frame_id``. Free-text
    ``convention`` strings and matrix names are never used to infer the frame.
    Use the camera endpoint corresponding to this calibration; these legacy
    files contain no machine-verifiable camera serial binding.

    Construction only reads the camera and the bridge description. It starts
    continuous tracking, waits for a new complete target observation, and binds
    it to ``YamTcpBackend.object_provider``. On failure it closes its observer.
    ``close()`` closes only that owned observer, never the robot or camera server.
    Live-camera feedback cannot be combined with a dry-run YAM backend here.
    """
    if not isinstance(limits, Limits):
        raise Rejected("explicit station Limits are required")
    if expected_kind != "real":
        raise Rejected(
            "yam_rgbd requires expected_kind='real'; dry-run cannot use live object feedback"
        )
    if selector_kwargs is None:
        selector_kwargs = {}
    if not isinstance(selector_kwargs, dict) or "seed_frame" in selector_kwargs:
        raise Rejected("selector_kwargs must be an object and cannot override seed_frame")
    seed_timeout = number(seed_timeout_s, "seed_timeout_s", 0.01, 10)
    connect_timeout = number(connect_timeout_s, "connect_timeout_s", 0.01, 10)
    calibration, calibration_digest = _calibration(
        calibration_path, frame_id=limits.frame_id, declared_frame=calibration_frame_id
    )
    select = _selector(target_selector)
    camera = CameraClient(
        host=camera_host,
        port=camera_port,
        timeout_s=seed_timeout,
        max_age_s=limits.max_state_age_s,
    )
    seed = copy.deepcopy(camera.read_frame(timeout_s=seed_timeout))
    original_timestamp = seed.get("timestamp_s")
    _fresh_seed(original_timestamp, limits.max_state_age_s)
    selection_frame = copy.deepcopy(seed)
    for name in ("rgb", "depth_m", "K"):
        if hasattr(selection_frame.get(name), "setflags"):
            selection_frame[name].setflags(write=False)
    target_points = select(selection_frame, **copy.deepcopy(selector_kwargs))
    # The original acquisition timestamp is outside selector ownership.
    _fresh_seed(original_timestamp, limits.max_state_age_s)
    observer = RGBDPointObserver(
        camera=camera,
        seed_frame=seed,
        target_points=target_points,
        frame_id=limits.frame_id,
        T_base_from_camera=calibration["T_base_from_camera"],
        max_age_s=limits.max_state_age_s,
    )
    try:
        observer.start()
        backend = YamTcpBackend(
            limits=limits,
            host=host,
            port=port,
            expected_kind="real",
            connect_timeout_s=connect_timeout,
            stop_timeout_s=stop_timeout_s,
            object_provider=observer,
        )
        first = observer(timeout_s=min(connect_timeout, limits.max_timeout_s))
        if set(first) != set(target_points):
            raise EvidenceMissing(
                "one or more selected targets were lost before backend initialization"
            )
    except BaseException as error:
        try:
            observer.close()
        except Exception as cleanup_error:
            raise EvidenceMissing(
                f"backend initialization failed ({error}); observer cleanup also failed: {cleanup_error}"
            ) from error
        raise
    return _ObservedBackend(
        backend,
        observer,
        {
            "type": "calibrated_rgbd_lk_surface_points",
            "calibration_sha256": calibration_digest,
            "calibration_frame_id": limits.frame_id,
            "target_selector": target_selector,
            "target_ids": sorted(target_points),
            "seed_timestamp_s": original_timestamp,
            "camera": seed["camera"],
            "semantics": "fixed visible surface centroids; not object poses or holding measurements",
            "station_validated": False,
        },
    )
