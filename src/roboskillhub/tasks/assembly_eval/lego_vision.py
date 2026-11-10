# SPDX-License-Identifier: Apache-2.0
"""Vision in the loop for the Isaac LEGO station: brick observations from its rendered RGB-D cameras.

``LegoPhysicsStation(observations="vision", vision=LegoVision())`` replaces the simulator-truth brick
poses of ``get_state().objects`` with estimates from the RTX-rendered overhead and wrist cameras of #171
(``LegoIsaacPhysics.camera_frame``, ``LEGO_RENDER=1``); ``observations="truth"`` (default) keeps the truth.
The baseplate and the table stay calibrated fixtures (their configured poses).

Every read after the simulation moved on by ``min_interval_s``:

1. **Frames.** The overhead camera (0.45 m over the plate, straight down) and the wrist camera, posed
   ``WRIST_OVER_TCP_M`` over the reported TCP looking down along its yaw.
2. **Segmentation.** Back-projected points more than 3 mm over the support under them (the plate top
   inside its footprint, the table elsewhere; studs are 1.7 mm) and outside the hand's known volume
   (a ``HAND_RADIUS_M`` cylinder around the reported TCP, from ``HAND_BELOW_TCP_M`` under it up), split
   into connected image blobs.
3. **Pose.** ``perception_regrasp.RgbdProvider`` (tools/pose6d box fit of the 2 x 4 brick body), put in the
   canonical symmetric equivalent (studs up, x continuous) and moved from the body centre to the contract's
   bottom centre.
4. **Identity, held brick, memory** as ``yam_vision``: nearest-centre matching from the configured feeder
   poses; the brick nearest the TCP when the jaw closes is predicted from the TCP while held; a resting
   brick nobody sees keeps its last full fit until the hand comes near.

``log`` keeps each estimate's error against truth for reporting only.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from ..perception_regrasp.objects import lego_brick
from ..perception_regrasp.providers import MaskedObject, RgbdFrame, RgbdProvider
from .yam_vision import _blobs, _canonical, _quat_to_R, _R_to_quat

FOREGROUND_OVER_SUPPORT_M = 0.003
MIN_BLOB_PX = 30
MATCH_GATE_M = 0.025
"""A resting brick's fit within this of its last estimate keeps its id (bricks move only in the hand, whose
brick is predicted from the TCP; a 10 cm gate let a second camera's fit of the neighbouring spare take b1's
id, 2026-11-10)."""
DUPLICATE_M = 0.010
HELD_GATE_M = 0.02
HELD_GATE_Z_M = 0.08
#: The jaw counts as closed (on a brick) after a gripper command narrower than this: the 2 x 4's short
#: side is 16 mm; the graph opens wider before a pick.
JAW_CLOSED_BELOW_M = 0.020
#: Reported for a held brick (TCP x grasp offset): the brick can shift in the pads; the domino twin's held
#: predictions were 8 mm median in transport, a LEGO brick pinched at 2 N moved under 1 mm at carry speed.
HELD_UNCERTAINTY = {"position_m": 0.003, "rotation_rad": 0.03}
MEMORY_HAND_GATE_M = 0.06
PARTIAL_KEEP_M = 0.010
MIN_IMAGE_INTERVAL_S = 0.5
"""RTX frames cost 2-5 s of wall time each (two cameras): at most one image set per 0.5 s simulated."""
#: Wrist camera: this far over the reported TCP, looking straight down (the floating hand has no arm).
WRIST_OVER_TCP_M = 0.12
#: The hand's volume masked from the images: pads and palm within this radius of the TCP axis,
#: from this far under the TCP upward.
HAND_RADIUS_M = 0.035
HAND_BELOW_TCP_M = 0.004
PLATE_SIZE_M = 0.256


def _look_down(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = (x, y, z)
    return T


class LegoVision:
    def __init__(
        self, *, cameras=("overhead", "wrist"), min_interval_s: float = MIN_IMAGE_INTERVAL_S
    ):
        if not cameras or set(cameras) - {"overhead", "wrist"}:
            raise ValueError(
                f"cameras must be a non-empty subset of (overhead, wrist), got {cameras!r}"
            )
        self.cameras = tuple(cameras)
        self.min_interval_s = min_interval_s
        self.log: list[dict] = []
        self._last_t: float | None = None
        self._out: dict[str, dict] = {}
        self._est: dict[str, np.ndarray] = {}
        self._T_est: dict[str, np.ndarray] = {}
        self._x_prev: dict[str, np.ndarray] = {}
        self._memory: dict[str, dict] = {}
        self._held: tuple[str, np.ndarray] | None = None
        self._closes_seen = 0

    def attach(self, station) -> None:
        self.station = station
        self.spec = lego_brick((2, 4))
        self.half_h = self.spec.size_m[2] / 2
        sc = station.scene
        self.plate_xy = np.asarray(sc["plate_pose"][0][:2], float)
        self.plate_top = float(sc["plate_pose"][0][2]) + 0.0032
        for f in sc["feeder"]:
            p = np.asarray(f["pos"], float)
            self._est[f["id"]] = p + [0.0, 0.0, self.half_h]
            w, x, y, z = f["rot"]
            self._x_prev[f["id"]] = _quat_to_R([x, y, z, w])[:, 1]  # BrickSim x -> contract y
        for k, bid in enumerate(sc.get("prebuilt_bricks", [])):
            self._est[bid] = np.array([*self.plate_xy, self.plate_top + self.half_h + 0.0096 * k])

    # ---- one image set ---------------------------------------------------------
    def _hand(self):
        pose, opening = self.station.physics.hand()
        return pose, opening

    def _support_z(self, x, y):
        on_plate = (np.abs(x - self.plate_xy[0]) < PLATE_SIZE_M / 2) & (
            np.abs(y - self.plate_xy[1]) < PLATE_SIZE_M / 2
        )
        return np.where(on_plate, self.plate_top, 0.0)

    def _segment(self, frame, hand) -> list[np.ndarray]:
        d = frame["depth"]
        K, T = frame["intrinsics"], frame["pose"]
        h, w = d.shape
        v, u = np.mgrid[0:h, 0:w]
        x = (u - K[0, 2]) * d / K[0, 0]
        y = (v - K[1, 2]) * d / K[1, 1]
        P = np.stack([x, y, d], -1) @ T[:3, :3].T + T[:3, 3]
        fg = (d > 0) & (
            P[..., 2] > self._support_z(P[..., 0], P[..., 1]) + FOREGROUND_OVER_SUPPORT_M
        )
        hx, hy, hz = hand[0], hand[1], hand[2]
        in_hand = (np.hypot(P[..., 0] - hx, P[..., 1] - hy) < HAND_RADIUS_M) & (
            P[..., 2] > hz - HAND_BELOW_TCP_M
        )
        return _blobs(fg & ~in_hand, MIN_BLOB_PX)

    def _image(self, stamp: float, sim_t: float) -> None:
        t0 = time.time()
        ph = self.station.physics
        hand, _ = self._hand()
        fits, n_blobs = [], 0
        for name in self.cameras:
            pose = (
                None
                if name == "overhead"
                else _look_down(hand[0], hand[1], hand[2] + WRIST_OVER_TCP_M, hand[3])
            )
            fr = ph.camera_frame(name, pose_cv=pose)
            blobs = self._segment(fr, hand)
            n_blobs += len(blobs)
            frame = RgbdFrame(
                fr["depth"],
                np.asarray(fr["intrinsics"]),
                fr["pose"],
                stamp,
                "world",
                name,
                fr["rgb"],
            )
            temp = {
                f"{name}{i}": MaskedObject(f"{name}{i}", b, 1.0, "depth")
                for i, b in enumerate(blobs)
            }
            prov = RgbdProvider(
                lambda frame=frame: frame,
                lambda f, temp=temp: list(temp.values()),
                {k: self.spec for k in temp},
                read_pips=False,
            )
            fits += [o for o in prov.observe(list(temp)) if o.has_pose]
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
            if dist * 0.005 > MATCH_GATE_M or i in used_fit or tid in used_id:
                continue
            used_fit.add(i)
            used_id.add(tid)
            out[tid] = fits[i]
        prev = self._out
        self._out = {}
        for tid, o in out.items():
            full = self._memory.get(tid)
            T = _canonical(np.asarray(o.T_frame_object, float), self._x_prev.get(tid))
            if (
                o.visibility == "partially_occluded"
                and full is not None
                and np.linalg.norm(T[:3, 3] - full["_center"]) < PARTIAL_KEEP_M
            ):
                d = {k: v for k, v in full.items() if not k.startswith("_")}
                d["source"] = "vision_memory"
                self._out[tid] = d
                continue
            bottom = T[:3, 3] - T[:3, 2] * self.half_h
            d = {
                "position_m": [float(v) for v in bottom],
                "quaternion_xyzw": _R_to_quat(T[:3, :3]),
                "frame_id": "world",
                "class": "lego_brick",
                "geometry": {"type": "box", "size_m": list(self.spec.size_m)},
                "studs": [2, 4],
                "source": "rgbd_pose6d",
                "privileged": False,
                "visibility": o.visibility,
                "confidence": o.confidence,
                "uncertainty": o.uncertainty.to_v1(),
                "evidence": {
                    "camera": o.evidence.get("camera"),
                    "coverage": o.evidence.get("coverage"),
                },
            }
            self._out[tid] = d
            self._est[tid] = T[:3, 3].copy()
            self._T_est[tid] = T
            self._x_prev[tid] = T[:3, 0].copy()
            if o.visibility == "visible":
                self._memory[tid] = {**d, "_seen_s": sim_t, "_center": T[:3, 3].copy()}
        self._update_held(stamp)
        self._remember(prev, sim_t, hand)
        for d in self._out.values():
            d["sim_time_s"] = sim_t
        self._record(sim_t, n_blobs, len(fits), time.time() - t0)

    def _tcp_T(self) -> np.ndarray:
        hand, _ = self._hand()
        c, s = math.cos(hand[3]), math.sin(hand[3])
        T = np.eye(4)
        T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T[:3, 3] = hand[:3]
        return T

    def _jaw(self) -> tuple[bool, int]:
        """(jaw commanded closed, number of close commands) from the station's own command log."""
        grips = [c["grip"] for c in getattr(self.station, "command_log", []) if "grip" in c]
        closes = sum(1 for g in grips if g < JAW_CLOSED_BELOW_M)
        return bool(grips) and grips[-1] < JAW_CLOSED_BELOW_M, closes

    def _update_held(self, stamp: float) -> None:
        closed, closes = self._jaw()
        if not closed:
            self._held = None
            self._closes_seen = closes
            return
        if self._held is None and closes > self._closes_seen:
            self._closes_seen = closes
            Tt = self._tcp_T()
            if os.environ.get("LEGO_VISION_DEBUG"):
                tr = self.station._truth_objects()
                print(
                    "LVCLOSE tcp",
                    np.round(Tt[:3, 3], 4).tolist(),
                    "est",
                    {t: np.round(p, 4).tolist() for t, p in self._est.items()},
                    "T_est",
                    sorted(self._T_est),
                    "truth",
                    {
                        t: np.round(o["position_m"], 4).tolist()
                        for t, o in tr.items()
                        if t in self._est
                    },
                    flush=True,
                )
            # the hand's reported point is not at the pads' height: gate in xy, loosely in z
            best = min(
                (
                    (float(np.linalg.norm(self._est[t][:2] - Tt[:2, 3])), t)
                    for t in self._T_est
                    if abs(self._est[t][2] - Tt[2, 3]) < HELD_GATE_Z_M
                ),
                default=(math.inf, None),
            )
            if best[1] is not None and best[0] <= HELD_GATE_M:
                self._held = (best[1], np.linalg.inv(Tt) @ self._T_est[best[1]])
        if self._held is not None:
            tid, off = self._held
            T = self._tcp_T() @ off
            bottom = T[:3, 3] - T[:3, 2] * self.half_h
            self._out[tid] = {
                **(self._out.get(tid) or {}),
                "position_m": [float(v) for v in bottom],
                "quaternion_xyzw": _R_to_quat(T[:3, :3]),
                "frame_id": "world",
                "class": "lego_brick",
                "geometry": {"type": "box", "size_m": list(self.spec.size_m)},
                "studs": [2, 4],
                "source": "grasp_kinematics",
                "privileged": False,
                "visibility": "visible",
                "confidence": 0.8,
                "uncertainty": dict(HELD_UNCERTAINTY),
            }
            self._est[tid] = T[:3, 3].copy()
            self._T_est[tid] = T
            self._x_prev[tid] = T[:3, 0].copy()

    def _remember(self, prev: dict, sim_t: float, hand) -> None:
        held = self._held[0] if self._held else None
        for tid in list(self._memory):
            if tid in self._out or tid == held:
                continue
            m = self._memory[tid]
            if prev.get(tid, {}).get("source") == "grasp_kinematics" or (
                np.linalg.norm(m["_center"][:2] - np.asarray(hand[:2])) < MEMORY_HAND_GATE_M
                and m["_center"][2] < hand[2] + 0.02
            ):
                del self._memory[tid]
                continue
            self._out[tid] = {
                **{k: v for k, v in m.items() if not k.startswith("_")},
                "source": "vision_memory",
            }

    def _record(self, sim_t, n_blobs, n_fits, secs) -> None:
        truth = self.station._truth_objects() if hasattr(self.station, "_truth_objects") else {}
        errs = {}
        for tid, d in self._out.items():
            t = truth.get(tid)
            if t is None or d.get("position_m") is None:
                continue
            Rt, Re = _quat_to_R(t["quaternion_xyzw"]), _quat_to_R(d["quaternion_xyzw"])
            yaw = (
                math.degrees(math.atan2(Re[1, 0], Re[0, 0]) - math.atan2(Rt[1, 0], Rt[0, 0]))
                % 180.0
            )
            errs[tid] = {
                "position_m": float(
                    np.linalg.norm(np.asarray(d["position_m"]) - np.asarray(t["position_m"]))
                ),
                "tilt_deg": math.degrees(math.acos(min(1.0, abs(float(Rt[:, 2] @ Re[:, 2]))))),
                "yaw_deg": min(yaw, 180.0 - yaw),
                "source": d.get("source"),
            }
        ids = [i for i in self._est]
        self.log.append(
            {
                "sim_time_s": sim_t,
                "blobs": n_blobs,
                "fits": n_fits,
                "missing": sorted(set(ids) - set(self._out)),
                "held": self._held[0] if self._held else None,
                "errors": errs,
                "processing_s": secs,
            }
        )

    def objects(self, stamp: float, sim_t: float) -> dict:
        imaged = self._last_t is None or sim_t - self._last_t >= self.min_interval_s
        if imaged:
            self._image(stamp, sim_t)
            self._last_t = sim_t
        else:
            self._update_held(stamp)
        if os.environ.get("LEGO_VISION_DEBUG"):
            print(
                f"LVDBG sim {sim_t:.2f} imaged {imaged} held {self._held[0] if self._held else None} "
                f"jaw {self._jaw()} out {sorted(self._out)}",
                flush=True,
            )
        now = time.time()
        out = {tid: {**d, "timestamp_s": now} for tid, d in self._out.items()}
        if self._held is not None and self._held[0] in out:
            # recomputed from the TCP on this read, not from the last image
            out[self._held[0]]["sim_time_s"] = sim_t
        return out

    def summary(self) -> dict:
        def stats(src):
            v = [
                e["position_m"]
                for r in self.log
                for e in r["errors"].values()
                if e.get("source") in src
            ]
            return {
                "n": len(v),
                "median": float(np.median(v)) if v else None,
                "max": float(np.max(v)) if v else None,
            }

        n = len(self._est) * max(1, len(self.log))
        seen = sum(len(r["errors"]) for r in self.log)
        return {
            "images": len(self.log),
            "missing_rate": 1.0 - seen / n,
            "position_err_m": stats(("rgbd_pose6d",)),
            "held_position_err_m": stats(("grasp_kinematics",)),
            "memory_position_err_m": stats(("vision_memory",)),
            "tilt_err_deg_p90": float(
                np.percentile(
                    [e["tilt_deg"] for r in self.log for e in r["errors"].values()] or [0], 90
                )
            ),
            "yaw_err_deg_p90": float(
                np.percentile(
                    [e["yaw_deg"] for r in self.log for e in r["errors"].values()] or [0], 90
                )
            ),
            "processing_s_median": float(np.median([r["processing_s"] for r in self.log]))
            if self.log
            else None,
            "cameras": list(self.cameras),
        }
