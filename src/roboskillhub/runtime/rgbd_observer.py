"""Metric surface-point feedback from a calibrated live RGB-D stream.

Target identity is supplied by perception/the operator on the exact seed frame.
This observer is neither an object detector nor a grasp/force estimator. It never
uses TCP commands as observations. A lost seed stays lost; object centroids use a
fixed set of surface points, never a changing subset that could invent motion.
"""

from __future__ import annotations

import copy
import threading
import time

from .contracts import EvidenceMissing, Rejected, number
from .tracking import LKPointTracker


class RGBDPointObserver:
    """Continuously track explicit surface seeds; expose object_provider(timeout_s).

    Construct with a CameraClient frame and matching {target_id: [[x,y], ...]}.
    Start before motion, then pass the observer to YamTcpBackend(object_provider=).
    Calls wait for a frame acquired after the call began, not merely a cached
    recent frame. The caller binds the named frame to its measured extrinsics;
    this class does not estimate calibration or infer object identity. The camera
    must honor read_frame(timeout_s). close() never closes the shared camera and
    reports an error if a non-cooperative camera leaves its worker running.
    """

    def __init__(
        self,
        *,
        camera,
        seed_frame: dict,
        target_points: dict,
        frame_id: str,
        T_base_from_camera,
        max_age_s: float = 1.0,
        max_frame_gap_s: float = 0.25,
        max_point_speed_m_s: float = 0.5,
        depth_noise_m: float = 0.015,
        depth_spread_m: float = 0.03,
    ):
        import numpy as np

        if not isinstance(frame_id, str) or not frame_id.strip():
            raise Rejected("frame_id must name the calibrated robot frame")
        try:
            transform = np.asarray(T_base_from_camera, dtype=float)
        except (TypeError, ValueError) as exc:
            raise Rejected("T_base_from_camera must be a numeric rigid transform") from exc
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-4)
        ):
            raise Rejected("T_base_from_camera must be a finite rigid 4x4 transform in metres")
        if not isinstance(target_points, dict) or not target_points:
            raise Rejected("target_points must map known target IDs to seed pixels")
        if not callable(getattr(camera, "read_frame", None)):
            raise Rejected("camera must supply a deadline-aware read_frame method")
        self.camera, self.frame_id, self.transform = camera, frame_id, transform.copy()
        self.max_age_s = number(max_age_s, "max_age_s", 0.01, 5)
        self.max_gap_s = number(max_frame_gap_s, "max_frame_gap_s", 0.01, 1)
        self.max_speed = number(max_point_speed_m_s, "max_point_speed_m_s", 0.01, 2)
        self.depth_noise = number(depth_noise_m, "depth_noise_m", 0.001, 0.05)
        self.depth_spread = number(depth_spread_m, "depth_spread_m", 0.001, 0.1)
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._worker = None
        self._error = None
        self._generation = 0
        self._tracker = LKPointTracker()
        self._frame = self._validate_frame(seed_frame)
        h, w = self._frame["rgb"].shape[:2]
        groups, seeds = {}, []
        for target, points in target_points.items():
            if not isinstance(target, str) or not target.strip():
                raise Rejected("each target ID must be a nonempty string")
            try:
                values = np.asarray(points, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise Rejected("seed pixels must be a numeric [N,2] array") from exc
            if (
                values.ndim != 2
                or values.shape[1] != 2
                or not len(values)
                or not np.isfinite(values).all()
                or (values < 0).any()
                or (values[:, 0] >= w).any()
                or (values[:, 1] >= h).any()
            ):
                raise Rejected("seed pixels must be nonempty [N,2] points inside the seed image")
            groups[target] = list(range(len(seeds), len(seeds) + len(values)))
            seeds.extend(values.tolist())
        if len(seeds) > 256:
            raise Rejected("at most 256 surface seeds are supported")
        self._groups = groups
        self._pixels = np.asarray(seeds, dtype=np.float32)
        self._valid = np.ones(len(seeds), dtype=bool)
        self._xyz = self._project(self._frame, self._pixels)
        if not np.isfinite(self._xyz).all():
            raise Rejected("all seed points need valid locally consistent depth")
        self._records = self._records_for(self._frame)

    def _validate_frame(self, record):
        import numpy as np

        if not isinstance(record, dict):
            raise EvidenceMissing("RGB-D frame must be a dictionary")
        try:
            rgb, depth = (np.asarray(record[key]) for key in ("rgb", "depth_m"))
            k = np.asarray(record["K"], dtype=float)
            stamp = float(record["timestamp_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceMissing("RGB-D frame fields are missing or invalid") from exc
        if (
            rgb.ndim != 3
            or rgb.shape[-1] != 3
            or rgb.dtype != np.uint8
            or min(rgb.shape[:2]) < 3
            or depth.shape != rgb.shape[:2]
            or depth.dtype.kind != "f"
            or not np.isfinite(depth).all()
            or (depth < 0).any()
            or k.shape != (3, 3)
            or not np.isfinite(k).all()
            or k[0, 0] <= 0
            or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or not np.isclose(k[1, 0], 0)
            or not np.isfinite(stamp)
        ):
            raise EvidenceMissing("invalid aligned RGB-D image, intrinsics or timestamp")
        age = time.time() - stamp
        if age < -0.1 or age > self.max_age_s:
            raise EvidenceMissing("RGB-D source timestamp is stale or its clock is unsynchronized")
        if not isinstance(record.get("camera"), str) or not record["camera"]:
            raise EvidenceMissing("RGB-D frame must identify the camera stream")
        return {
            **record,
            "rgb": rgb.copy(),
            "depth_m": depth.copy(),
            "K": k.copy(),
            "timestamp_s": stamp,
        }

    def _project(self, record, pixels):
        import numpy as np

        depth, k = record["depth_m"], record["K"]
        h, w = depth.shape
        xyz = np.full((len(pixels), 3), np.nan)
        for index, (u, v) in enumerate(pixels):
            if not np.isfinite([u, v]).all() or not (0 <= u < w and 0 <= v < h):
                continue
            x, y = int(round(float(u))), int(round(float(v)))
            x, y = min(x, w - 1), min(y, h - 1)
            # A hole at the tracked pixel is not a measurement of that surface;
            # neighbouring background pixels must not silently fill it.
            if depth[y, x] <= 0:
                continue
            patch = depth[max(0, y - 1) : min(h, y + 2), max(0, x - 1) : min(w, x + 2)]
            valid = patch[patch > 0]
            if len(valid) < 3 or float(np.ptp(valid)) > self.depth_spread:
                continue
            z = float(np.median(valid))
            ray = np.linalg.solve(k, np.array([u, v, 1.0]))
            xyz[index] = self.transform[:3, :3] @ (ray * z) + self.transform[:3, 3]
        return xyz

    def _records_for(self, frame):
        import numpy as np

        records = {}
        for target, indices in self._groups.items():
            # Do not change the landmark definition when a point disappears.
            if not self._valid[indices].all():
                continue
            records[target] = {
                "frame_id": self.frame_id,
                "position_m": np.mean(self._xyz[indices], axis=0).tolist(),
                "timestamp_s": frame["timestamp_s"],
                "source": "calibrated_rgbd_lk_surface_points",
                "camera": frame["camera"],
                "landmark": "fixed_seed_surface_centroid",
                "seed_count": len(indices),
                "identity_source": "explicit_seed_frame_target_assignment",
                "observation_semantics": "visible surface-point centroid; not object center, pose or holding",
            }
        return records

    def _update(self, record):
        import numpy as np

        frame = self._validate_frame(record)
        previous = self._frame
        if (
            frame["camera"] != previous["camera"]
            or frame["rgb"].shape != previous["rgb"].shape
            or not np.allclose(frame["K"], previous["K"])
        ):
            raise EvidenceMissing("camera stream, image shape or intrinsics changed")
        dt = frame["timestamp_s"] - previous["timestamp_s"]
        if dt < 0:
            raise EvidenceMissing("camera timestamps moved backwards")
        if dt == 0:
            return
        if dt > self.max_gap_s:
            raise EvidenceMissing("frame gap exceeds the configured tracking continuity bound")
        active = np.flatnonzero(self._valid)
        if len(active):
            tracked = self._tracker.track(
                np.stack([previous["rgb"], frame["rgb"]]), self._pixels[active]
            )
            visible = np.asarray(tracked["visible"][-1], dtype=bool)
            next_pixels = np.asarray(tracked["tracks"][-1], dtype=float)
            if visible.shape != (len(active),) or next_pixels.shape != (len(active), 2):
                raise EvidenceMissing("tracker changed the fixed seed count or coordinate shape")
            xyz = self._project(frame, next_pixels)
            displacement = np.linalg.norm(xyz - self._xyz[active], axis=1)
            accepted = (
                visible
                & np.isfinite(xyz).all(axis=1)
                & (displacement <= self.depth_noise + self.max_speed * dt)
            )
            self._valid[active] = accepted
            self._pixels[active[accepted]] = next_pixels[accepted]
            self._xyz[active[accepted]] = xyz[accepted]
        records = self._records_for(frame)
        with self._condition:
            self._frame, self._records = frame, records
            self._generation += 1
            self._condition.notify_all()

    def start(self):
        with self._condition:
            if self._worker is not None or self._stop_event.is_set():
                raise Rejected("observer can be started exactly once")
            self._worker = threading.Thread(target=self._run, name="roboskillhub-rgbd", daemon=True)
            self._worker.start()
        return self

    def _run(self):
        try:
            while not self._stop_event.is_set():
                record = self.camera.read_frame(timeout_s=self.max_gap_s)
                if self._stop_event.is_set():
                    return
                self._update(record)
                if time.time() - self._frame["timestamp_s"] > self.max_gap_s:
                    raise EvidenceMissing("camera stopped producing distinct frames")
                self._stop_event.wait(0.005)
        except Exception as exc:
            with self._condition:
                self._error = f"{type(exc).__name__}: {exc}"
                self._condition.notify_all()

    def __call__(self, *, timeout_s: float):
        budget = number(timeout_s, "timeout_s", 0.001, 120)
        after, deadline = time.time(), time.monotonic() + budget
        with self._condition:
            if self._worker is None:
                raise EvidenceMissing("start the live object observer before invoking motion")
            generation = self._generation
            while True:
                if self._stop_event.is_set() or self._error:
                    raise EvidenceMissing(self._error or "RGB-D observer is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise EvidenceMissing(
                        "no fresh post-call RGB-D object observation before deadline"
                    )
                if self._generation > generation and self._frame["timestamp_s"] > after:
                    age = time.time() - self._frame["timestamp_s"]
                    if not -0.1 <= age <= self.max_age_s:
                        raise EvidenceMissing("object observation timestamp is stale")
                    return copy.deepcopy(self._records)
                self._condition.wait(remaining)

    def close(self):
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=self.max_gap_s + 0.1)
            if self._worker.is_alive():
                raise EvidenceMissing(
                    "RGB-D worker did not stop; camera read_frame must honor timeout_s"
                )

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
