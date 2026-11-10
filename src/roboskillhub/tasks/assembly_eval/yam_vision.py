# SPDX-License-Identifier: Apache-2.0
"""Vision in the loop for the YAM domino station: tile observations from a rendered RGB-D camera.

``YamDominoStation(observations="vision", vision=YamVision(...))`` replaces the simulator-truth
tile poses of ``get_state().objects`` with estimates from a camera image; ``observations="truth"``
(the default) keeps the privileged truth. The table object stays the calibrated support.

Pipeline, every time the objects are read and the simulation has advanced since the last image:

1. **Cameras.** The wrist D405 (``WRIST``: follows the reported TCP) and the real station's overhead Orbbec: its ChArUco calibration
   (``top_camera_pose_at_capture_20261103-1421``: left-arm base frame, OpenCV axes) mapped into the
   scene through the arm mount (the twin's left base frame is the model's turned -90 deg about z,
   ``yam_model.YamMount``), 640 x 480 with the real intrinsics' focal length (fy 461.5 px). MuJoCo
   renders RGB and depth (the twin's renderer; no sensor noise model).
2. **Segmentation.** ``segmenter="depth"``: pixels more than 3 mm over the calibrated table plane,
   minus the arm and the fixed fixtures (rack separators, table), split into connected blobs. The
   arm and fixtures are masked by rendering them at their known poses (robot FK, rack calibration)
   - the station knows where its own arm and rack are; no tile identity or pose enters.
   ``segmenter="detector"``: a 2D detector's boxes (e.g. grounding-dino, tools/README.md) intersected
   with that same foreground; blobs no box covers are dropped.
3. **Pose.** Each blob's depth points go to ``perception_regrasp.RgbdProvider``: tools/pose6d's box fit
   of the 48 x 24 x 7.5 mm tile with the table as support plane, refitted without it for a leaning
   tile, poor fits rejected.
4. **Identity.** Tiles start at their calibrated rack slots (``DominoSceneSpec.feeder_pose``, not the
   seeded truth); every frame the fitted poses are matched to the last estimates by nearest centre
   (greedy, 0.15 m gate). A tile no blob matches keeps no pose (``not_found``).
   A box fit returns any of the tile's four symmetric half-turns; the estimate is put in the
   contract's canonical one: height axis (+z) up, facing axis (+x) closest to the tile's previous
   estimate (first: the calibrated rack yaw). The truth's frame is a physical one, so without this a
   standing tile can come back "upside down" and a place computed from it is unreachable.
   A partly hidden fit (coverage < 0.6) within 1 cm of the tile's last full-coverage fit is
   replaced by that fit (``source: vision_memory``); memory keeps only full-coverage fits.
6. **Memory.** A tile last seen resting that no camera sees now (the arm is over it) keeps its last
   estimate (``source: vision_memory``, last-seen time in ``evidence``) as long as the reported TCP has
   not come within 8 cm of it since; a tile the hand went near is ``not_found`` until seen again. A
   tile knocked over while hidden is therefore only caught when it is seen again.
5. **Held tile.** From overhead, a tile in the closed jaws is hidden by the hand. At the first image
   after the jaw closes, the tracked tile nearest the reported TCP (within 3 cm) becomes the held
   tile, with its estimated pose relative to the TCP; while the jaw stays closed it is reported as
   the reported TCP pose times that offset (``source: grasp_kinematics``). The robot's own TCP and
   jaw state, no object truth - and no slip detection while held.

Measured 2026-11-09 (rshgpu5/6, MuJoCo EGL, depth segmentation, both cameras), Agent 5's graph and
baselines on test seeds 0-9: fixed_script and online_replan each 8/10 graph success vs truth's 10/10
(before the partial-fit rule below; with it, fixed_script 10/10 graph and evaluator, 2026-11-10).
The evaluator found all 10 layouts built in both, so the 2 failures are vision false alarms
(``place: domino.neighbour_disturbed``, seeds 4 and 5). Pose error against truth over 167 images
an episode: 1.7 mm median, 5.5 mm max position, 0.2-0.3 deg p90 tilt; held tiles 8 mm median
(20 mm max) from the grasp prediction while carried; 4 % of tile-reads unseen; 0.9 s an image.

``log`` keeps, per image, each estimate's error against the simulator truth (position, tilt of the
up axis, yaw modulo 180 deg) for reporting; the truth never feeds the estimates.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..perception_regrasp.objects import domino as domino_spec
from ..perception_regrasp.providers import MaskedObject, RgbdFrame, RgbdProvider

CALIBRATION = {
    "T_base_from_camera": [
        [0.998535, -0.046372, 0.027883, 0.29002],
        [-0.054109, -0.857869, 0.511012, 0.036629],
        [0.000223, -0.511772, -0.859121, 0.908602],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "fy_px": 461.52,
    "size_px": [640, 480],
    "source": "YAM station overhead Orbbec, ChArUco calibration 2026-11-03 14:21 (rms 5.4 mm), "
    "~/yam_gap/top_camera_pose_at_capture_20261103-1421.json; K from cable_frame_20261103-1421.npz",
}
#: The twin's left base frame -> the model's base frame: model (x, y) = real (y, -x).
R_MODEL_FROM_REAL_BASE = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
FOREGROUND_OVER_TABLE_M = 0.003
MIN_BLOB_PX = 12
MATCH_GATE_M = 0.15
#: Re-image at most this often in simulated time; reads in between reuse the last estimates.
MIN_IMAGE_INTERVAL_S = 0.05
#: A tracked tile within this of the TCP when the jaw closes is the held one.
HELD_GATE_M = 0.03
#: A partly hidden fit (coverage under pose6d's 0.6) within this of the tile's last full-coverage fit is
#: replaced by that fit: with the hand beside a placed tile its partial fits were 3-4 mm off (seed 4,
#: 2026-11-10) against the place skill's 3 mm neighbour tolerance. A real move over 1 cm passes.
PARTIAL_KEEP_M = 0.010
#: A remembered (unseen) tile is dropped once the reported TCP comes within this of it.
MEMORY_HAND_GATE_M = 0.08


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def overhead_camera(physics, calibration: dict = CALIBRATION) -> dict:
    """``T_world_camera`` (OpenCV axes) and K of the station's overhead camera in the scene."""
    mt = physics.cfg.mount
    T_bc = np.asarray(calibration["T_base_from_camera"], float)
    base_z = physics.spec.table.top_z + mt.base_over_table_m
    R_wb = _rot_z(mt.yaw) @ R_MODEL_FROM_REAL_BASE
    T = np.eye(4)
    T[:3, :3] = R_wb @ T_bc[:3, :3]
    T[:3, 3] = R_wb @ T_bc[:3, 3] + np.array([mt.base_xy[0], mt.base_xy[1], base_z])
    w, h = calibration["size_px"]
    f = calibration["fy_px"]
    K = np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])
    return {"T_world_camera": T, "K": K, "size": (w, h), "source": calibration["source"]}


#: The left wrist D405 (station snapshot 2026-11-03 threading-setup, wrist_left.npz K): fy 393.9 px at
#: 640 x 480. Mount (not calibrated for the left arm): 9 cm back along the approach axis and 3.5 cm
#: off the jaw plane, aimed 3 cm past the TCP, so it looks between the open fingers.
WRIST = {
    "fy_px": 393.89,
    "size_px": [640, 480],
    "offset_tcp_m": [0.035, 0.0, -0.09],
    "aim_tcp_m": [0.0, 0.0, 0.03],
}


class _Renderer:
    """MuJoCo RGB, depth and geom-id images from a free camera at a given position and forward axis.

    MuJoCo's free camera has no roll: the camera is rebuilt from its forward axis with world-up, and
    that rebuilt pose is the one used to back-project (the overhead calibration's roll is ~3 deg).
    """

    def __init__(self, physics, size):
        import mujoco

        self.mj = mujoco
        self.m, self.d = physics.m, physics.d
        w, h = size
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, w)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, h)
        self.size = (w, h)
        self.r = mujoco.Renderer(self.m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def view(self, pos, forward, fy: float, dist: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Point the camera; returns ``T_world_camera`` (OpenCV axes) and K."""
        w, h = self.size
        f = np.asarray(forward, float) / np.linalg.norm(forward)
        pos = np.asarray(pos, float)
        self.fovy = math.degrees(2 * math.atan(h / 2 / fy))
        self.cam.lookat[:] = pos + dist * f
        self.cam.distance = float(dist)
        self.cam.azimuth = math.degrees(math.atan2(f[1], f[0]))
        self.cam.elevation = math.degrees(math.asin(max(-1.0, min(1.0, f[2]))))
        right = np.cross(f, [0.0, 0.0, 1.0])
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        T = np.eye(4)
        T[:3, :3] = np.stack([right, -up, f], axis=1)
        T[:3, 3] = pos
        K = np.array([[fy, 0.0, w / 2], [0.0, fy, h / 2], [0.0, 0.0, 1.0]])
        return T, K

    def render(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self.r
        self.m.vis.global_.fovy = self.fovy
        r.update_scene(self.d, camera=self.cam)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.d, camera=self.cam)
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(self.d, camera=self.cam)
        seg = r.render()[:, :, 0].copy()
        r.disable_segmentation_rendering()
        return rgb, depth.astype(np.float32), seg


def _blobs(mask: np.ndarray, min_px: int) -> list[np.ndarray]:
    import cv2

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [lab == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]


class YamVision:
    """Tile observations for ``YamDominoStation`` from the overhead camera (see the module doc)."""

    def __init__(
        self,
        *,
        segmenter: str = "depth",
        detector: Callable[[np.ndarray], list[list[float]]] | None = None,
        calibration: dict = CALIBRATION,
        min_interval_s: float = MIN_IMAGE_INTERVAL_S,
        save_dir: str | None = None,
        cameras: tuple[str, ...] = ("overhead", "wrist"),
        depth: str | None = None,
    ):
        if segmenter not in ("depth", "detector"):
            raise ValueError(f"segmenter must be depth or detector, got {segmenter!r}")
        if segmenter == "detector" and detector is None:
            raise ValueError(
                "segmenter='detector' needs a detector(rgb) -> [[x1, y1, x2, y2], ...]"
            )
        if not cameras or set(cameras) - {"overhead", "wrist"}:
            raise ValueError(
                f"cameras must be a non-empty subset of (overhead, wrist), got {cameras!r}"
            )
        self.cameras = tuple(cameras)
        # depth the pipeline sees: the renderer's ("rendered"), with #185's fitted D405 noise
        # ("fitted"), or D405-like passive stereo from a second render 18 mm to the right ("stereo")
        self.depth_mode = depth or os.environ.get("DOMINO_VISION_DEPTH", "rendered")
        if self.depth_mode not in ("rendered", "fitted", "stereo"):
            raise ValueError(f"depth must be rendered, fitted or stereo, got {self.depth_mode!r}")
        self._rng = np.random.default_rng(0)
        self.segmenter, self.detector = segmenter, detector
        self.calibration = calibration
        self.min_interval_s = min_interval_s
        self.save_dir = save_dir
        self.log: list[dict] = []
        self._last_t: float | None = None
        self._est: dict[str, np.ndarray] = {}
        self._out: dict[str, dict] = {}
        self._T_est: dict[str, np.ndarray] = {}
        self._held: tuple[str, np.ndarray] | None = None
        self._x_prev: dict[str, np.ndarray] = {}
        self._memory: dict[str, dict] = {}
        self._closes_seen = 0

    # ---- set-up from the station ------------------------------------------------
    def attach(self, station) -> None:
        self.station = station
        ph = station.physics
        self.cam = overhead_camera(ph, self.calibration)
        self.render = _Renderer(ph, self.cam["size"])
        m = ph.m
        tile_bodies = {m.body(f"tile_{k}").id for k in range(ph.spec.n_tiles)}
        self.known_geom = np.array(
            [m.geom_bodyid[g] not in tile_bodies for g in range(m.ngeom)] + [True], bool
        )
        self.spec = domino_spec((ph.spec.tile.height, ph.spec.tile.width, ph.spec.tile.thickness))
        self.table_z = station.calibrated_table_z
        for k, tid in enumerate(station.piece_ids):
            fp = ph.spec.feeder_pose(k)
            self._est[tid] = np.asarray(fp["position"], float)
            self._x_prev[tid] = _quat_to_R(fp["quat_xyzw"])[:, 0]

    # ---- one image -----------------------------------------------------------------
    def _segment(self, rgb, depth, seg, T, K) -> list[np.ndarray]:
        h, w = depth.shape
        v, u = np.mgrid[0:h, 0:w]
        z = depth
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        wz = T[2, 0] * x + T[2, 1] * y + T[2, 2] * z + T[2, 3]
        known = self.known_geom[np.where(seg >= 0, seg, len(self.known_geom) - 1)]
        fg = (wz > self.table_z + FOREGROUND_OVER_TABLE_M) & ~known & (z > 0)
        if self.segmenter == "detector":
            keep = np.zeros_like(fg)
            for b in self.detector(rgb):
                x1, y1, x2, y2 = (int(round(c)) for c in b)
                keep[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = True
            fg &= keep
        return _blobs(fg, MIN_BLOB_PX)

    def _view(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Point the renderer's camera at view ``name``; its ``T_world_camera`` and K."""
        if name == "overhead":
            T = self.cam["T_world_camera"]
            return self.render.view(T[:3, 3], T[:3, 2], self.cam["K"][1, 1])
        Tt = self._tcp_T()
        pos = Tt[:3, :3] @ np.asarray(WRIST["offset_tcp_m"]) + Tt[:3, 3]
        aim = Tt[:3, :3] @ np.asarray(WRIST["aim_tcp_m"]) + Tt[:3, 3]
        return self.render.view(pos, aim - pos, WRIST["fy_px"], dist=0.1)

    def _sensor_depth(self, rgb, depth, T, K):
        if self.depth_mode == "rendered":
            return depth
        from ..lego_hub import depth as D

        if self.depth_mode == "fitted":
            return D.add_fitted_noise(depth, K, self._rng).astype(np.float32)
        self.render.view(T[:3, 3] + T[:3, 0] * D.D405_BASELINE_M, T[:3, 2], K[1, 1])
        right, _, _ = self.render.render()
        return D.stereo_depth(rgb, right, K, rng=self._rng)

    def _image(self, stamp: float, sim_t: float) -> None:
        t0 = time.time()
        fits, n_blobs, shots = [], 0, []
        for name in self.cameras:
            T, K = self._view(name)
            rgb, depth, seg = self.render.render()
            depth = self._sensor_depth(rgb, depth, T, K)
            blobs = self._segment(rgb, depth, seg, T, K)
            n_blobs += len(blobs)
            frame = RgbdFrame(depth, K, T, stamp, "world", name, rgb)
            temp = {
                f"{name}{i}": MaskedObject(f"{name}{i}", b, 1.0, self.segmenter)
                for i, b in enumerate(blobs)
            }
            prov = RgbdProvider(
                lambda frame=frame: frame,
                lambda f, temp=temp: list(temp.values()),
                {k: self.spec for k in temp},
                table_z=self.table_z,
                read_pips=False,
            )
            fits += [o for o in prov.observe(list(temp)) if o.has_pose]
            shots.append((name, rgb, blobs))
        blobs = [b for _, _, bs in shots for b in bs]
        # identity: greedy nearest match of fitted centres to the last estimates
        # nearest first; between two views' fits of one tile (within 5 mm) the more confident wins
        pairs = sorted(
            (
                round(float(np.linalg.norm(o.T_frame_object[:3, 3] - p)) / 0.005),
                -(o.confidence or 0.0),
                i,
                tid,
            )
            for i, o in enumerate(fits)
            for tid, p in self._est.items()
        )
        used_fit, used_id, out = set(), set(), {}
        for dist, _, i, tid in pairs:
            dist *= 0.005
            if dist > MATCH_GATE_M or i in used_fit or tid in used_id:
                continue
            used_fit.add(i)
            used_id.add(tid)
            o = fits[i]
            o.object_id = tid
            out[tid] = o
        prev = self._out
        self._out = {}
        for tid, o in out.items():
            d = o.to_backend_object()
            full = self._memory.get(tid)
            if (
                o.visibility == "partially_occluded"
                and full is not None
                and np.linalg.norm(o.T_frame_object[:3, 3] - np.asarray(full["position_m"]))
                < PARTIAL_KEEP_M
            ):
                # a partly hidden fit (the hand beside the tile) is less exact than the last full one
                d = {
                    **{k: v for k, v in full.items() if k != "_seen_s"},
                    "source": "vision_memory",
                    "evidence": {
                        "kept_full_fit_from_sim_s": full["_seen_s"],
                        "partial_fit_offset_m": float(
                            np.linalg.norm(o.T_frame_object[:3, 3] - np.asarray(full["position_m"]))
                        ),
                    },
                }
                self._out[tid] = d
                T = _contract_T(d)
                self._est[tid] = T[:3, 3].copy()
                self._T_est[tid] = T
                continue
            T = _canonical(_contract_T(d), self._x_prev.get(tid))
            d["quaternion_xyzw"] = _R_to_quat(T[:3, :3])
            self._out[tid] = d
            self._est[tid] = T[:3, 3].copy()
            self._T_est[tid] = T
            self._x_prev[tid] = T[:3, 0].copy()
        self._update_held(stamp)
        self._fitted = set(out)
        self._remember(prev, sim_t)
        for d in self._out.values():
            d["sim_time_s"] = sim_t
        self._record(stamp, sim_t, out, n_blobs, len(fits))
        self.log[-1]["processing_s"] = time.time() - t0
        self.log[-1]["views"] = {tid: o.evidence.get("camera") for tid, o in out.items()}
        if self.save_dir:
            for name, rgb, bl in shots:
                self._save(name, rgb, bl)

    def _remember(self, prev: dict, sim_t: float) -> None:
        """Unseen tiles keep their last resting estimate while the hand stays away from them."""
        held = self._held[0] if self._held else None
        tcp = self._tcp_T()[:3, 3]
        for tid, d in self._out.items():
            if d.get("source") == "rgbd_pose6d" and d.get("visibility") == "visible":
                self._memory[tid] = {**d, "_seen_s": sim_t}
        for tid in list(self._memory):
            if tid in self._out or tid == held:
                continue
            m = self._memory[tid]
            if prev.get(tid, {}).get("source") == "grasp_kinematics" or (
                np.linalg.norm(np.asarray(m["position_m"]) - tcp) < MEMORY_HAND_GATE_M
            ):
                del self._memory[tid]
                continue
            self._out[tid] = {
                **{k: v for k, v in m.items() if k != "_seen_s"},
                "source": "vision_memory",
                "evidence": {"last_seen_sim_s": m["_seen_s"], "hidden_for_s": sim_t - m["_seen_s"]},
            }

    def _tcp_T(self) -> np.ndarray:
        st = self.station
        p, R = st.physics.tcp()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p - st.offset  # the pose the robot reports (a tcp_offset fault stays unseen)
        return T

    def _update_held(self, stamp: float) -> None:
        st = self.station
        closed = st.closes > st.releases
        if not closed:
            self._held = None
            self._closes_seen = st.closes
            return
        if self._held is None and st.closes > self._closes_seen:
            self._closes_seen = st.closes
            T_tcp = self._tcp_T()
            best = min(
                ((float(np.linalg.norm(self._est[t] - T_tcp[:3, 3])), t) for t in self._T_est),
                default=(math.inf, None),
            )
            if best[1] is not None and best[0] <= HELD_GATE_M:
                self._held = (best[1], np.linalg.inv(T_tcp) @ self._T_est[best[1]])
        if self._held is not None:
            tid, T_tcp_obj = self._held
            T = self._tcp_T() @ T_tcp_obj
            base = dict(self._out.get(tid) or {})
            base.update(
                {
                    "position_m": [float(v) for v in T[:3, 3]],
                    "quaternion_xyzw": _R_to_quat(T[:3, :3]),
                    "frame_id": "world",
                    "timestamp_s": stamp,
                    "source": "grasp_kinematics",
                    "privileged": False,
                    "visibility": "visible",
                    "class": "domino",
                    "geometry": {"type": "box", "size_m": list(self.spec.size_m)},
                    "reason": "held: reported TCP pose x the tile's offset estimated when the jaw closed",
                }
            )
            self._out[tid] = base
            self._est[tid] = T[:3, 3].copy()
            self._T_est[tid] = T
            self._x_prev[tid] = T[:3, 0].copy()

    def _record(self, stamp, sim_t, out, n_blobs, n_fits) -> None:
        truth = self.station.physics.truth()["objects"]
        errs = {}
        for tid in self._out:
            t = truth.get(tid)
            if t is None:
                continue
            q = t["quat_xyzw"]
            R_true = _quat_to_R(q)
            d = self._out[tid]
            R_est = _quat_to_R(d["quaternion_xyzw"])
            up_t, up_e = R_true[:, 2], R_est[:, 2]
            tilt = math.degrees(math.acos(min(1.0, abs(float(up_t @ up_e)))))
            fx_t, fx_e = R_true[:2, 0], R_est[:2, 0]
            yaw = math.degrees(math.atan2(fx_e[1], fx_e[0]) - math.atan2(fx_t[1], fx_t[0])) % 180.0
            errs[tid] = {
                "position_m": float(
                    np.linalg.norm(np.asarray(d["position_m"]) - np.asarray(t["position"]))
                ),
                "xy_m": float(
                    np.linalg.norm(np.asarray(d["position_m"][:2]) - np.asarray(t["position"][:2]))
                ),
                "tilt_deg": tilt,
                "yaw_deg": min(yaw, 180.0 - yaw),
                "source": d.get("source"),
            }
        self.log.append(
            {
                "sim_time_s": sim_t,
                "blobs": n_blobs,
                "fits": n_fits,
                "observed": sorted(self._out),
                "held": self._held[0] if self._held else None,
                "missing": sorted(set(self.station.piece_ids) - set(self._out)),
                "errors": errs,
            }
        )

    def _save(self, name, rgb, blobs) -> None:
        from PIL import Image

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        k = len(self.log) - 1
        img = rgb.copy()
        for b in blobs:
            img[b] = (0.5 * img[b] + [0, 127, 0]).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(self.save_dir, f"frame_{k:04d}_{name}.png"))
        json.dump(self.log[-1], open(os.path.join(self.save_dir, f"frame_{k:04d}.json"), "w"))

    # ---- station hook --------------------------------------------------------------
    def objects(self, stamp: float, sim_t: float) -> dict:
        """Tile observations (backend-object dicts); re-images when the simulation moved on by
        ``min_interval_s`` since the last image. The simulation does not advance while an image is
        processed (it catches up to wall time at the next read), so observations are stamped when
        the processing ends; the processing time is in ``log``."""
        if self._last_t is None or sim_t - self._last_t >= self.min_interval_s:
            self._image(stamp, sim_t)
            self._last_t = sim_t
        else:
            self._update_held(stamp)
        now = time.time()
        return {tid: {**d, "timestamp_s": now} for tid, d in self._out.items()}

    def summary(self) -> dict:
        """Pose errors and misses over every image of the episode."""
        errs = [
            e
            for r in self.log
            for e in r["errors"].values()
            if e.get("source") not in ("grasp_kinematics", "vision_memory")
        ]
        mem = [
            e for r in self.log for e in r["errors"].values() if e.get("source") == "vision_memory"
        ]
        held = [
            e
            for r in self.log
            for e in r["errors"].values()
            if e.get("source") == "grasp_kinematics"
        ]
        n_tiles = len(self.station.piece_ids)

        def pct(key, q):
            v = [e[key] for e in errs]
            return float(np.percentile(v, q)) if v else None

        return {
            "images": len(self.log),
            "observations": len(errs),
            "missing_rate": 1.0
            - (len(errs) + len(held) + len(mem)) / max(1, n_tiles * len(self.log)),
            "held_observations": len(held),
            "memory_observations": len(mem),
            "memory_position_err_m": {
                "median": float(np.median([e["position_m"] for e in mem])) if mem else None,
                "max": float(np.max([e["position_m"] for e in mem])) if mem else None,
            },
            "held_position_err_m": {
                "median": float(np.median([e["position_m"] for e in held])) if held else None,
                "max": float(np.max([e["position_m"] for e in held])) if held else None,
            },
            "position_err_m": {
                "median": pct("position_m", 50),
                "p90": pct("position_m", 90),
                "max": pct("position_m", 100),
            },
            "tilt_err_deg": {"median": pct("tilt_deg", 50), "p90": pct("tilt_deg", 90)},
            "yaw_err_deg": {"median": pct("yaw_deg", 50), "p90": pct("yaw_deg", 90)},
            "processing_s": {
                "median": float(np.median([r["processing_s"] for r in self.log]))
                if self.log
                else None,
                "max": float(np.max([r["processing_s"] for r in self.log])) if self.log else None,
            },
            "segmenter": self.segmenter,
            "cameras": list(self.cameras),
            "overhead_calibration": self.cam["source"],
        }


#: The box's symmetric half-turns (object frame): identity and 180 deg about x, y, z.
_HALF_TURNS = [np.diag(v) for v in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]


def _canonical(T: np.ndarray, x_prev: np.ndarray | None) -> np.ndarray:
    """The symmetric equivalent of ``T`` with +z most upward, then +x closest to ``x_prev``."""
    best = None
    for H in _HALF_TURNS:
        R = T[:3, :3] @ H
        key = (round(float(R[2, 2]), 3), float(R[:, 0] @ x_prev) if x_prev is not None else 0.0)
        if best is None or key > best[0]:
            best = (key, R)
    out = T.copy()
    out[:3, :3] = best[1]
    return out


def _contract_T(d: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _quat_to_R(d["quaternion_xyzw"])
    T[:3, 3] = d["position_m"]
    return T


def _R_to_quat(R: np.ndarray) -> list[float]:
    from ..perception_regrasp.frames import matrix_to_quat_xyzw

    return [float(v) for v in matrix_to_quat_xyzw(R)]


def _quat_to_R(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
