# SPDX-License-Identifier: Apache-2.0
"""A workcell-runtime ``StationBackend`` driving one i2rt YAM arm in MuJoCo over the domino scene.

Embodiment: the left arm of the YAM station twin (``yam_model``), base 21 mm over the table,
linear_4310 gripper, joint position servos as shipped. Contract v1.2 surface:
``describe / get_state / move_pose / move_joints / set_gripper / stop``; the runtime builds
``move_waypoints``, ``set_gripper(open|close|release)`` and ``observe_objects`` on top.

- ``get_state``: TCP = FK of the measured joints at the MJCF ``grasp_site`` (TCP z = approach,
  TCP y = jaw closing axis, as the floating-gripper station); ``joints`` = measured arm joint
  angles (model convention; the real station's joint 1 reads +pi/2); ``gripper.width_m`` =
  blade-tip gap from the finger joints; ``gripper.holding`` = both pads touch the same tile and
  the measured gap is at most the tile width + 2 mm; objects = simulator truth,
  ``privileged: true``, plus ``table``. Stamped ``time.time()`` at the read, with ``sim_time_s``.
- ``move_pose``: refused before any motion (``ok: false``, ``failure_type: "unreachable"`` in
  the reply) when the TCP target is under the tool-centre floor or outside the station's tool
  box, has no IK solution inside the joint ranges (1 mm / 0.02 rad), or cannot be bridged
  from the current configuration along the straight TCP path in 1 cm / 0.05 rad steps of at
  most 0.75 rad joint travel each (the real motion server's bounded-IK rule). Executed as a
  joint path through those IK steps, timed by the TCP speed, ``angular_rate_rad_s`` and
  ``joint_rate_rad_s``; then up to 0.5 s settling. The ack is not arrival: the runtime checks
  the measured pose. Arbitrary orientations are served (not only top-down).
- ``move_joints``: every point inside the joint ranges and the interpolated path's TCP inside
  the tool box and over the floor, else refused before motion; executed at ``speed_rad_s``.
- ``set_gripper``: blade-tip width -> finger command ``(w + 1 mm) / 2``, ramped at 50 mm/s per
  finger; ``stop`` holds the measured joints.

The station faults are the floating-gripper station's (``domino_station.STATION_FAULTS``,
same trigger and size), applied to the arm: ``tcp_offset`` is a TCP calibration error (the
arm goes 12 mm along +x of where FK reports), ``release_jerk`` / ``retreat_deviation`` are arm
motions planned through the same IK.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np

from .domino_runner import Observer
from .domino_scene import DominoSceneSpec
from .domino_station import STATION_FAULTS
from .yam_model import (
    GAP_OFFSET,
    GAP_PER_CMD,
    GRIP_OPEN_CMD,
    J1_HW_OFFSET,
    JOINTS,
    WORK_POSE,
    rot_err,
)

GRIPPER = ("gripper_pad_l", "gripper_pad_r")
TCP_FLOOR_M = 0.014
"""Lowest tool centre over the calibrated table [m]: the YAM station's floor that the hub's
skills use (perception GraspLimits.min_tcp_height_m, grasp_flat_cable MIN_TOOL_HEIGHT). The
current ``motion_server.py`` defaults to 8.7 mm (``--floor-mm``)."""
STATION_BOX_BASE = ((0.10, -0.35), (0.55, 0.15), 0.35)
"""motion_server BOUNDS (left base frame x -0.15..0.35, y 0.10..0.55, z up to 0.35 m) in the
model's base frame (x = hw y, y = -hw x): ((xmin, ymin), (xmax, ymax), zmax)."""
STEP_M, STEP_RAD, STEP_TRAVEL_RAD = 0.01, 0.05, 0.75


def _rotvec_mat(w: np.ndarray) -> np.ndarray:
    a = float(np.linalg.norm(w))
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _quat_xyzw(R: np.ndarray) -> list[float]:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(9))
    w, x, y, z = q
    return [float(x), float(y), float(z), float(w)]


def _mat_xyzw(q) -> np.ndarray:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class Refused(Exception):
    """A motion request the station refuses before moving."""


class YamDominoStation:
    """``StationBackend`` over ``yam_model.YamDominoPhysics``."""

    def __init__(
        self,
        physics,
        spec: DominoSceneSpec,
        feeder_poses: list[dict],
        *,
        calibrated_table_z: float,
        piece_ids: list[str],
        fault: str | None = None,
        fault_index: int = 0,
        table_offset_m: float = 0.008,
        max_catch_up_s: float = 1.0,
        observe_hz: float = 30.0,
        tcp_floor_m: float = TCP_FLOOR_M,
        joint_rate_rad_s: float = 0.5,
        angular_rate_rad_s: float = 0.5,
        start_q=WORK_POSE,
    ):
        if fault is not None and fault not in STATION_FAULTS:
            raise ValueError(f"unknown station fault {fault!r}")
        self.physics = physics
        self.spec = spec
        self.calibrated_table_z = calibrated_table_z
        self.piece_ids = piece_ids
        self.fault, self.fault_index = fault, fault_index
        self.table_offset_m = table_offset_m
        self.max_catch_up_s = max_catch_up_s
        self.tcp_floor_m = tcp_floor_m
        self.joint_rate = joint_rate_rad_s
        self.angular_rate = angular_rate_rad_s
        physics.reset(spec, feeder_poses, q0=start_q, base_table_z=calibrated_table_z)
        self.kin = physics.kin
        bx, by = physics.cfg.mount.base_xy
        bz = calibrated_table_z + physics.cfg.mount.base_over_table_m
        (x0, y0), (x1, y1), z1 = STATION_BOX_BASE
        self.box_min = [bx + x0, by + y0, calibrated_table_z + tcp_floor_m]
        self.box_max = [bx + x1, by + y1, bz + z1]
        self.observer = Observer(observe_hz)
        self.q_cmd = np.array(start_q, float)
        self.grip_cmd = physics.finger_cmd_measured()
        self.offset = np.zeros(3)  # tcp_offset fault: actual = reported + offset (world)
        self._offset_target: np.ndarray | None = None
        self.closes = 0
        self.releases = 0
        self.events: list[dict] = []
        self.wall0 = time.time()
        self._last_wall = self.wall0
        self.sim_t = 0.0
        self._frozen: dict | None = None
        self._lost: str | None = None
        self._retreat_pending = False
        self.command_log: list[dict] = []
        self.refusals: list[dict] = []
        self.hand_track: list[tuple[float, float]] = []
        self.stopped = False

    # ---- runtime surface --------------------------------------------------
    def describe(self) -> dict:
        return {
            "frame_id": "world",
            "backend": "domino-yam-mujoco",
            "kind": "dynamic_sim_privileged",
            "evidence_level": "dynamic_sim_privileged",
            "physics": True,
            "robot": "i2rt YAM arm (one arm, the YAM station twin's left/work arm) with linear_4310 "
            "gripper, MuJoCo; joint position servos, gravity compensated",
            "embodiment": {
                "id": "yam_i2rt",
                "arms": 1,
                "source": "RoboSimStudio yam_i2rt_arms.xml (MIT, I2RT Robotics); twin "
                "yam-station-20261103-gsota differs only in base spacing",
                "modified": "fingertip and housing colliders are explicit boxes measured from the "
                "meshes (shipped tip hulls stand ~13 mm proud of the pads; the housing did not collide)",
                "not_modelled": [
                    "second arm, cameras, wrist camera cable",
                    "joint-3/4 gravity sag of the real arm (it needs 1.1-1.5x i2rt's gravity model)",
                    "station supervisor tracking/speed guards, 20 Hz command latency",
                    "gripper motor current limit (finger servo kp 600 / 80 N as shipped)",
                    "calibration error of base, table and TCP (only as injected faults)",
                ],
                "joint_convention": "model angles; real station joint 1 = model joint 1 + pi/2",
            },
            "tcp_convention": "MJCF grasp_site: TCP z = approach out of the fingers, TCP y = jaw "
            "closing axis; any orientation the IK reaches",
            "motion": f"straight TCP path in {STEP_M * 100:g} cm / {STEP_RAD} rad IK steps "
            f"(<= {STEP_TRAVEL_RAD} rad joint travel each) joined in joint space; joints <= "
            f"{self.joint_rate} rad/s, TCP rotation <= {self.angular_rate} rad/s; unreachable "
            "targets refused before motion",
            "limits": {
                "tcp_floor_over_table_m": self.tcp_floor_m,
                "tool_box_min_m": self.box_min,
                "tool_box_max_m": self.box_max,
                "ik_tolerance": {"position_m": self.kin.pos_tol, "rotation_rad": self.kin.rot_tol},
            },
            "observations": "simulation truth (tile poses, contact pairs), privileged",
            "clock": "simulation advanced to track wall time in get_state (catch-up capped at "
            f"{self.max_catch_up_s} s); timestamps are time.time() at the read",
            "physics_backend": self.physics.describe(),
            "fault": self.fault,
            "joints": {
                "names": list(JOINTS),
                "lower_rad": [float(v) for v in self.kin.lo],
                "upper_rad": [float(v) for v in self.kin.hi],
                "max_velocity_rad_s": [self.joint_rate] * 6,
            },
            "capabilities": [
                "pose_feedback",
                "cartesian_motion",
                "gripper",
                "gripper_feedback",
                "stop",
                "object_feedback",
                "object_pose_feedback",
                "joint_position_feedback",
                "joint_motion",
            ],
        }

    def get_state(self, *, timeout_s: float) -> dict:
        self._catch_up()
        return self._state()

    def move_pose(self, target: dict, *, speed_m_s: float, timeout_s: float, linear: bool) -> dict:
        self._catch_up()
        if target.get("frame_id") != "world":
            return {"ok": False, "error": "frame must be world"}
        goal_p = np.array(target["position_m"], float)
        goal_R = _mat_xyzw(target["quaternion_xyzw"])
        try:
            self._check_tcp(goal_p, "target")
        except Refused as exc:
            return self._refuse("move_pose", target, str(exc))
        if (
            self.fault == "tcp_offset"
            and self.closes - 1 == self.fault_index
            and not self.offset.any()
            and math.dist(goal_p[:2], self._cmd_tcp()[0][:2]) > 0.03
        ):
            self._offset_target = np.array([0.012, 0.0, 0.0])  # ramps in over this carry move
        if self._retreat_pending and goal_p[2] > self._cmd_tcp()[0][2] + 0.002:
            self._retreat_pending = False
            if self.fault == "retreat_deviation" and self.releases - 1 == self.fault_index:
                p, R = self._cmd_tcp()
                side = p + np.array([-0.025, 0.0, 0.0])
                try:
                    self._drive(side - self.offset, R, speed=0.03, timeout_s=timeout_s)
                except Refused as exc:
                    self.command_log.append({"wall_s": time.time(), "fault_motion_refused": str(exc)})
        off1 = self._offset_target if self._offset_target is not None else self.offset
        self._offset_target = None
        try:
            ok = self._drive(goal_p, goal_R, speed=max(1e-3, speed_m_s), timeout_s=timeout_s, offset_to=off1)
        except Refused as exc:
            return self._refuse("move_pose", target, str(exc))
        self.command_log.append(
            {"wall_s": time.time(), "sim_s": self.sim_t, "move": target, "ok": ok}
        )
        return {"ok": True} if ok else {"ok": False, "error": "timeout before arrival"}

    def move_joints(self, q_path, *, speed_rad_s: float, timeout_s: float) -> dict:
        self._catch_up()
        path = [np.asarray(q, float) for q in q_path]
        try:
            for i, q in enumerate(path):
                if np.any(q < self.kin.lo - 1e-9) or np.any(q > self.kin.hi + 1e-9):
                    raise Refused(f"q_path[{i}] outside the joint ranges")
            prev = self.q_cmd.copy()
            for i, q in enumerate(path):
                n = max(1, math.ceil(np.abs(q - prev).max() / 0.02))
                for k in range(1, n + 1):
                    p, _ = self.kin.fk(prev + (q - prev) * k / n)
                    self._check_tcp(p + self.offset, f"joint path to q_path[{i}]")
                prev = q
        except Refused as exc:
            return self._refuse("move_joints", {"points": len(path)}, str(exc))
        qs = [self.q_cmd.copy(), *path]
        durs = [max(np.abs(b - a).max() / max(speed_rad_s, 1e-4), self.physics.control_dt) for a, b in zip(qs, qs[1:])]
        ok = self._run(qs, durs, timeout_s=timeout_s)
        self.command_log.append({"wall_s": time.time(), "sim_s": self.sim_t, "move_joints": len(path), "ok": ok})
        return {"ok": True} if ok else {"ok": False, "error": "timeout before arrival"}

    def set_gripper(self, width_m: float, *, timeout_s: float) -> dict:
        self._catch_up()
        holding_tile = self._held_tile()
        target_cmd = self._cmd_for_width(width_m)
        widening = width_m > self.physics.gap() + 0.002
        start = self.physics.finger_cmd_measured()
        dur = max(0.2, abs(target_cmd - start) / 0.05) + 0.3
        if widening and holding_tile is not None:
            self._on_release(holding_tile)
            if self.fault == "stuck_jaws" and self.releases - 1 == self.fault_index:
                self.command_log.append(
                    {"wall_s": time.time(), "sim_s": self.sim_t, "grip": width_m, "ignored": "stuck_jaws"}
                )
                self._advance(0.5)
                return {"ok": True}
            if self.fault == "release_jerk" and self.releases - 1 == self.fault_index:
                p, R = self._cmd_tcp()
                try:
                    self._drive(
                        p + np.array([0.03, 0.0, 0.0]) - self.offset,
                        R,
                        speed=0.15,
                        timeout_s=timeout_s,
                        grip_to=target_cmd,
                        settle=False,
                    )
                except Refused as exc:
                    self.command_log.append({"wall_s": time.time(), "fault_motion_refused": str(exc)})
        elif width_m < self.spec.tile.thickness + 0.001:
            self._on_close()
        self._ramp_gripper(self.grip_cmd, target_cmd, dur)
        self.command_log.append({"wall_s": time.time(), "sim_s": self.sim_t, "grip": width_m})
        return {"ok": True}

    def stop(self) -> None:
        self.q_cmd = self.physics.joints()
        self.grip_cmd = self.physics.finger_cmd_measured()
        self.stopped = True

    # ---- independent-evaluation hooks ------------------------------------
    def settle(self, seconds: float) -> None:
        """Run physics with the hand parked 15 cm over the table (after the episode)."""
        p, R = self._cmd_tcp()
        park = p.copy()
        park[2] = max(park[2], self.spec.table.top_z + 0.15)
        try:
            self._drive(park - self.offset, R, speed=0.05, timeout_s=10.0)
        except Refused:
            pass
        self._advance(seconds)

    # ---- planning ---------------------------------------------------------
    def _check_tcp(self, p, what: str) -> None:
        floor = self.calibrated_table_z + self.tcp_floor_m
        if p[2] < floor - 1e-6:
            raise Refused(
                f"{what}: tool centre {p[2] - self.calibrated_table_z:.4f} m over the table is under "
                f"the station's {self.tcp_floor_m:.3f} m floor"
            )
        if any(v < lo - 1e-6 or v > hi + 1e-6 for v, lo, hi in zip(p, self.box_min, self.box_max)):
            raise Refused(f"{what}: tool centre {np.round(p, 4).tolist()} outside the station's tool box")

    def _cmd_tcp(self):
        """Commanded TCP as reported (FK of the commanded joints minus the calibration offset)."""
        p, R = self.kin.fk(self.q_cmd)
        return p - self.offset, R

    def plan(self, goal_p, goal_R, *, offset_to=None):
        """Joint waypoints from the commanded configuration to the (reported-frame) goal along
        the straight TCP path; raises ``Refused`` with the reason."""
        off0 = self.offset.copy()
        off1 = off0 if offset_to is None else np.asarray(offset_to, float)
        q0 = self.q_cmd.copy()
        p0, R0 = self.kin.fk(q0)
        p1 = np.asarray(goal_p, float) + off1
        # reachable at all?
        seeds = [q0, np.array(WORK_POSE)]
        rng = np.random.default_rng(0)
        seeds += [rng.uniform(self.kin.lo, self.kin.hi) for _ in range(6)]
        best = None
        for s in seeds:
            q, pe, re_, ok = self.kin.solve(p1, goal_R, s)
            if best is None or pe + 0.05 * re_ < best[1] + 0.05 * best[2]:
                best = (q, pe, re_)
            if ok:
                break
        else:
            q, pe, re_ = best
            raise Refused(
                f"unreachable: no IK solution within the joint ranges (best {pe * 1000:.1f} mm, "
                f"{re_:.3f} rad; at limits {self.kin.at_limits(q)})"
            )
        w = rot_err(goal_R, R0)
        dist = float(np.linalg.norm(p1 - p0))
        ang = float(np.linalg.norm(w))
        n = max(1, math.ceil(dist / STEP_M), math.ceil(ang / STEP_RAD))
        qs, segs = [q0], []
        q = q0
        for i in range(1, n + 1):
            u = i / n
            p_i = p0 + (p1 - p0) * u
            R_i = _rotvec_mat(w * u) @ R0
            q_i, pe, re_, ok = self.kin.solve(p_i, R_i, q, max_travel=STEP_TRAVEL_RAD)
            if not ok:
                raise Refused(
                    f"unreachable along the path from the current configuration: step {i}/{n} "
                    f"({pe * 1000:.1f} mm, {re_:.3f} rad off; at limits {self.kin.at_limits(q_i)})"
                )
            segs.append((dist / n, ang / n, float(np.abs(q_i - q).max())))
            qs.append(q_i)
            q = q_i
        return qs, segs, (off0, off1)

    def _drive(self, goal_p, goal_R, *, speed, timeout_s, offset_to=None, grip_to=None, settle=True) -> bool:
        qs, segs, (off0, off1) = self.plan(goal_p, goal_R, offset_to=offset_to)
        durs = [
            max(d / speed, a / self.angular_rate, j / self.joint_rate, self.physics.control_dt)
            for d, a, j in segs
        ]
        return self._run(qs, durs, timeout_s=timeout_s, offsets=(off0, off1), grip_to=grip_to, settle=settle)

    def _run(self, qs, durs, *, timeout_s, offsets=None, grip_to=None, settle=True) -> bool:
        dt = self.physics.control_dt
        total = sum(durs) if durs else 0.0
        n = max(1, round(total / dt))
        bounds = np.cumsum([0.0, *durs])
        wall_deadline = time.time() + max(timeout_s, 0.1)
        g0 = self.grip_cmd
        self.stopped = False
        for i in range(1, n + 1):
            t = min(total, i * dt)
            k = min(len(durs) - 1, int(np.searchsorted(bounds, t, side="right") - 1)) if durs else 0
            if durs:
                u = (t - bounds[k]) / max(durs[k], 1e-9)
                self.q_cmd = qs[k] + (qs[k + 1] - qs[k]) * min(1.0, u)
            if offsets is not None:
                self.offset = offsets[0] + (offsets[1] - offsets[0]) * (i / n)
            if grip_to is not None:
                self.grip_cmd = g0 + (grip_to - g0) * (i / n)
            self._step()
            if time.time() > wall_deadline:
                self._last_wall = time.time()
                return False
        if settle:
            p_t, R_t = self.kin.fk(self.q_cmd)
            for _ in range(round(0.5 / dt)):
                p, R = self.physics.tcp()
                if np.linalg.norm(p - p_t) < 0.001 and np.linalg.norm(rot_err(R_t, R)) < 0.01:
                    break
                self._step()
        self._last_wall = time.time()
        return True

    def _refuse(self, what: str, request, reason: str) -> dict:
        self.refusals.append({"wall_s": time.time(), "sim_s": self.sim_t, "call": what, "request": request, "reason": reason})
        return {"ok": False, "error": reason, "failure_type": "unreachable", "refused_before_motion": True}

    # ---- internals ---------------------------------------------------------
    @staticmethod
    def _cmd_for_width(width_m: float) -> float:
        return min(GRIP_OPEN_CMD, max(0.0, (width_m - GAP_OFFSET) / GAP_PER_CMD))

    def _step(self) -> None:
        self.sim_t = self.physics.step(self.q_cmd, self.grip_cmd)
        n = len(self.observer.samples)
        self.observer.maybe_sample(self.sim_t, self.physics)
        if len(self.observer.samples) > n:
            self.hand_track.append((self.sim_t, float(self.physics.tcp()[0][2])))

    def _advance(self, seconds: float) -> None:
        for _ in range(max(0, round(seconds / self.physics.control_dt))):
            self._step()
        self._last_wall = time.time()

    def _catch_up(self) -> None:
        now = time.time()
        self._advance(min(self.max_catch_up_s, max(0.0, now - self._last_wall)))

    def _ramp_gripper(self, start: float, target: float, dur: float) -> None:
        dt = self.physics.control_dt
        n = max(1, round(dur / dt))
        hold = round(0.3 / dt)
        for i in range(1, n + 1):
            self.grip_cmd = start + (target - start) * min(1.0, i / max(1, n - hold))
            self._step()
        self._last_wall = time.time()

    def _contacts(self) -> dict:
        return self.physics.truth().get("contacts") or {}

    def _held_tile(self) -> str | None:
        """Both pads touch the same tile and the measured gap is closed onto it (at most the
        tile's width + 2 mm), as the floating-gripper station decides."""
        if self.physics.gap() > self.spec.tile.width + 0.002:
            return None
        for tid, touching in self._contacts().items():
            if all(p in touching for p in GRIPPER):
                return tid
        return None

    def _on_close(self) -> None:
        self.closes += 1
        self.events.append({"event": "close", "sim_s": self.sim_t, "wall_s": time.time(), "index": self.closes - 1})

    def _on_release(self, tile: str) -> None:
        self.releases += 1
        p, _ = self.physics.tcp()
        self.events.append(
            {
                "event": "release",
                "sim_s": self.sim_t,
                "wall_s": time.time(),
                "tile": tile,
                "index": self.releases - 1,
                "hand_z": float(p[2]),
            }
        )
        self._retreat_pending = True
        if self.releases - 1 == self.fault_index:
            if self.fault == "frozen_objects":
                self._frozen = self._objects(time.time())
            if self.fault == "lost_target":
                self._lost = tile

    def _objects(self, stamp: float) -> dict:
        truth = self.physics.truth()
        objs = {}
        offset_active = self.fault == "perception_offset" and self.closes == self.fault_index
        rack_y = self.spec.feeder.origin_xy[1]
        t = self.spec.tile
        for tid, o in truth["objects"].items():
            pos = list(o["position"])
            if offset_active and abs(pos[1] - rack_y) < 0.01:
                pos[1] += 0.03  # only pieces still in the rack
            objs[tid] = {
                "position_m": pos,
                "quaternion_xyzw": list(o["quat_xyzw"]),
                "frame_id": "world",
                "timestamp_s": stamp,
                "sim_time_s": self.sim_t,
                "source": "simulation_truth",
                "privileged": True,
                "confidence": 1.0,
                "visibility": "visible",
                "class": "domino",
                "geometry": {"type": "box", "size_m": [t.height, t.width, t.thickness]},
            }
        tb = self.spec.table
        objs["table"] = {
            "position_m": [tb.center_xy[0], tb.center_xy[1], tb.top_z],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "frame_id": "world",
            "timestamp_s": stamp,
            "sim_time_s": self.sim_t,
            "source": "simulation_truth",
            "privileged": True,
            "confidence": 1.0,
            "visibility": "visible",
            "class": "support",
            "geometry": {"type": "box", "size_m": [tb.size_xy[0], tb.size_xy[1], tb.thickness]},
        }
        return objs

    def _state(self) -> dict:
        now = time.time()
        p, R = self.physics.tcp()
        reported = p - self.offset
        objects = self._objects(now)
        if self._frozen is not None:
            objects = copy.deepcopy(self._frozen)
        if self._lost is not None:
            objects[self._lost] = {
                "position_m": None,
                "quaternion_xyzw": None,
                "frame_id": "world",
                "timestamp_s": now,
                "source": "simulation_truth",
                "privileged": True,
                "visibility": "not_found",
                "reason": "injected: target dropped from the observations",
            }
        q = self.physics.joints()
        return {
            "pose": {
                "frame_id": "world",
                "position_m": [float(v) for v in reported],
                "quaternion_xyzw": _quat_xyzw(R),
            },
            "timestamp_s": now,
            "sim_time_s": self.sim_t,
            "objects": objects,
            "joints": {
                "names": list(JOINTS),
                "positions_rad": [float(v) for v in q],
                "hardware_positions_rad": [float(q[0] + J1_HW_OFFSET), *map(float, q[1:])],
                "timestamp_s": now,
                "source": "simulation_joint_encoders",
            },
            "gripper": {
                "holding": self._held_tile() is not None,
                "width_m": self.physics.gap(),
                "source": "simulation_contacts",
                "timestamp_s": now,
            },
        }
