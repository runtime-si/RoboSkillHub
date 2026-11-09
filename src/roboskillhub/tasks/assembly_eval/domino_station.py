# SPDX-License-Identifier: Apache-2.0
"""A workcell-runtime ``StationBackend`` over the domino physics scene (Isaac/PhysX or MuJoCo).

What it is: a **floating parallel-jaw gripper** (x/y/z prismatic + yaw revolute mount,
two force-limited sliding fingers) above a table with free rigid-body tiles. It is not a
YAM arm: no arm kinematics, no reach or wrist limits, no joint space. Only top-down TCP
poses (TCP z pointing down, TCP y = closing axis) can be served; any other orientation is
refused before motion (``ok: false, refused_before_motion: true, failure_type:
"unreachable"`` with the reason; contract 1.3), and kept in ``refusals``.

Contract v1.3 surface (``describe / get_state / move_pose / set_gripper / stop``; the
runtime builds ``move_waypoints``, ``set_gripper(open|close|release)`` and
``observe_objects`` on top of these). Motion and gripper replies carry ``sim_time_s``.

- objects: every tile with position, quaternion, ``source: "simulation_truth"``,
  ``privileged: true``; plus ``"table"`` (class ``support``, top-surface centre).
- ``gripper.holding`` comes from measurements, never from the command: both pads in contact
  with the same tile and the measured pad gap at most the tile width + 2 mm; ``width_m`` is
  the measured pad gap.
- Clock: physics only advances inside this object, so ``get_state`` first advances the
  simulation by the wall time elapsed since the last step (at most ``max_catch_up_s``) with
  the last command held; every state is stamped with ``time.time()`` at the read and carries
  ``sim_time_s``. A tile still falling is therefore seen falling across the runtime's
  sleeps, instead of being frozen between calls.

Fault injection (``fault``) changes the world or the hardware, never the skill or the
evaluator, and applies to placement number ``fault_index`` (0-based; default: the last
slot): ``perception_offset`` (the pieces' observed poses are 30 mm off across their width
until the next close), ``tcp_offset`` (after the close, the hand runs 12 mm along +x of
where it reports), ``release_jerk`` (the hand moves 30 mm along +x at 0.15 m/s while the
jaws open), ``retreat_deviation`` (after the release, the hand first slides 25 mm along -x at
release height), ``table_offset`` (the physical table is ``table_offset_m`` above its
calibration), ``stuck_jaws`` (release commands do not open the jaws), ``frozen_objects``
(object timestamps freeze at the release), ``lost_target`` (the released piece disappears
from the observations).
"""

from __future__ import annotations

import copy
import math
import time

from .domino_geometry import quat_mul, quat_rotate
from .domino_runner import Observer
from .domino_scene import DominoSceneSpec

GRIPPER = ("gripper_pad_l", "gripper_pad_r")
STATION_FAULTS = (
    "perception_offset",
    "tcp_offset",
    "release_jerk",
    "retreat_deviation",
    "table_offset",
    "stuck_jaws",
    "frozen_objects",
    "lost_target",
)
_DOWN = (1.0, 0.0, 0.0, 0.0)  # rotation pi about x: TCP z -> world -z, TCP y -> world -y


def tcp_quaternion(yaw: float) -> list[float]:
    """TCP orientation of the floating hand at mount yaw ``yaw``: Rz(yaw) * Rx(pi)."""
    qz = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    return list(quat_mul(qz, _DOWN))


def yaw_from_tcp(q) -> tuple[float, float]:
    """Mount yaw for a TCP orientation, and how far its z axis is from straight down (rad)."""
    z = quat_rotate(q, (0.0, 0.0, 1.0))
    x = quat_rotate(q, (1.0, 0.0, 0.0))
    off_down = math.acos(max(-1.0, min(1.0, -z[2])))
    return math.atan2(x[1], x[0]), off_down


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class DominoPhysicsStation:
    """``StationBackend`` over a ``DominoBackend`` (``MujocoBackend`` or ``IsaacBackend``)."""

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
        real_time_factor: float | None = None,
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
        self.declared_real_time_factor = real_time_factor
        self.measured_real_time_factor: float | None = None
        physics.reset(spec, feeder_poses)
        self.observer = Observer(observe_hz)
        self.hand_cmd, self.open_cmd = physics.hand()
        self.offset = [0.0, 0.0]  # tcp_offset fault: actual = reported + offset
        self._offset_target: list[float] | None = None
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
        self.hand_track: list[tuple[float, float]] = []  # (sim_s, measured hand z) per sample
        self.stopped = False

    # ---- runtime surface --------------------------------------------------
    def describe(self) -> dict:
        return {
            "frame_id": "world",
            "backend": f"domino-physics-{self.physics.name}",
            "kind": "dynamic_sim_privileged",
            "evidence_level": "dynamic_sim_privileged",
            "physics": True,
            "robot": "floating parallel-jaw gripper (x/y/z prismatic + yaw mount, two sliding "
            "fingers); NOT a YAM arm: no arm kinematics, reach, wrist or joint limits",
            "tcp_convention": "TCP z down, TCP y = closing axis; only top-down poses are served",
            "observations": "simulation truth (tile poses, contact pairs), privileged",
            "clock": "simulation advanced to track wall time in get_state (catch-up capped at "
            f"{self.max_catch_up_s} s); timestamps are time.time() at the read",
            "physics_backend": self.physics.describe(),
            "fault": self.fault,
            # Declared (or else measured over commanded motions) sim seconds per wall second;
            # the runtime stretches wall-clock deadlines when it is below 1.
            **self._real_time_factor(),
            "capabilities": [
                "pose_feedback",
                "cartesian_motion",
                "gripper",
                "gripper_feedback",
                "stop",
                "object_feedback",
                "object_pose_feedback",
            ],
        }

    def get_state(self, *, timeout_s: float) -> dict:
        self._catch_up()
        return self._state()

    def move_pose(self, target: dict, *, speed_m_s: float, timeout_s: float, linear: bool) -> dict:
        self._catch_up()
        if target.get("frame_id") != "world":
            return self._refuse("move_pose", target, "frame must be world", "invalid_request")
        yaw, off_down = yaw_from_tcp(target["quaternion_xyzw"])
        if off_down > math.radians(3):
            return self._refuse(
                "move_pose",
                target,
                f"floating gripper serves top-down TCP poses only "
                f"({math.degrees(off_down):.1f} deg off vertical)",
            )
        goal = [*target["position_m"], yaw]
        if (
            self.fault == "tcp_offset"
            and self.closes - 1 == self.fault_index
            and self.offset == [0.0, 0.0]
            and math.dist(goal[:2], self.hand_cmd[:2]) > 0.03
        ):
            self._offset_target = [0.012, 0.0]  # ramps in over this carry move
        if self._retreat_pending and goal[2] > self.hand_cmd[2] + 0.002:
            self._retreat_pending = False
            if self.fault == "retreat_deviation" and self.releases - 1 == self.fault_index:
                side = list(self.hand_cmd)
                side[0] -= 0.025
                self._drive(side, self.open_cmd, speed=0.03, timeout_s=timeout_s)
        ok = self._drive(goal, self.open_cmd, speed=max(1e-3, speed_m_s), timeout_s=timeout_s)
        self.command_log.append(
            {"wall_s": time.time(), "sim_s": self.sim_t, "move": goal, "ok": ok}
        )
        if ok:
            return {"ok": True, "sim_time_s": self.sim_t}
        return {"ok": False, "error": "timeout before arrival", "sim_time_s": self.sim_t}

    def set_gripper(self, width_m: float, *, timeout_s: float) -> dict:
        self._catch_up()
        holding_tile = self._held_tile()
        widening = width_m > self._measured_open() + 0.002
        if widening and holding_tile is not None:
            self._on_release(holding_tile)
            if self.fault == "stuck_jaws" and self.releases - 1 == self.fault_index:
                self.command_log.append(
                    {
                        "wall_s": time.time(),
                        "sim_s": self.sim_t,
                        "grip": width_m,
                        "ignored": "stuck_jaws",
                    }
                )
                self._advance(0.5)
                return {"ok": True, "sim_time_s": self.sim_t}
            if self.fault == "release_jerk" and self.releases - 1 == self.fault_index:
                goal = list(self.hand_cmd)
                goal[0] += 0.03
                self._drive(goal, width_m, speed=0.15, timeout_s=timeout_s, settle=False)
        elif width_m < self.spec.tile.thickness + 0.001:
            self._on_close()
        start = self._measured_open()
        dur = max(0.2, abs(width_m - start) / 2 / self.spec.gripper.finger_speed) + 0.3
        self._ramp_gripper(start, width_m, dur)
        self.command_log.append({"wall_s": time.time(), "sim_s": self.sim_t, "grip": width_m})
        return {"ok": True, "sim_time_s": self.sim_t}

    def stop(self) -> None:
        pose, opening = self.physics.hand()
        self.hand_cmd = list(pose)
        self.open_cmd = opening
        self.stopped = True

    # ---- independent-evaluation hooks ------------------------------------
    def settle(self, seconds: float) -> None:
        """Run physics with the hand parked (used after the episode for the final window)."""
        park = list(self.hand_cmd)
        park[2] = max(park[2], self.spec.table.top_z + 0.15)
        self._drive(park, self.open_cmd, speed=0.05, timeout_s=10.0)
        self._advance(seconds)

    # ---- internals ---------------------------------------------------------
    def _refuse(self, what: str, request, reason: str, failure_type: str = "unreachable") -> dict:
        """Decline a target before commanding anything (contract 1.3 station refusal)."""
        self.refusals.append(
            {
                "wall_s": time.time(),
                "sim_s": self.sim_t,
                "call": what,
                "request": request,
                "reason": reason,
            }
        )
        return {
            "ok": False,
            "error": reason,
            "failure_type": failure_type,
            "refused_before_motion": True,
            "sim_time_s": self.sim_t,
        }

    def _measured_open(self) -> float:
        return self.physics.hand()[1]

    def _actual(self, goal):
        return [goal[0] + self.offset[0], goal[1] + self.offset[1], goal[2], goal[3]]

    def _step(self) -> None:
        self.sim_t = self.physics.step(self._actual(self.hand_cmd), self.open_cmd)
        n = len(self.observer.samples)
        self.observer.maybe_sample(self.sim_t, self.physics)
        if len(self.observer.samples) > n:
            self.hand_track.append((self.sim_t, self.physics.hand()[0][2]))

    def _advance(self, seconds: float) -> None:
        n = max(0, round(seconds / self.physics.control_dt))
        for _ in range(n):
            self._step()
        self._last_wall = time.time()

    def _catch_up(self) -> None:
        now = time.time()
        self._advance(min(self.max_catch_up_s, max(0.0, now - self._last_wall)))

    def _drive(self, goal, opening, *, speed, timeout_s, settle=True) -> bool:
        g = self.spec.gripper
        start = list(self.hand_cmd)
        goal = list(goal)
        goal[3] = start[3] + _wrap(goal[3] - start[3])
        dist = math.dist(start[:3], goal[:3])
        dur = max(dist / speed, abs(goal[3] - start[3]) / g.yaw_speed, 0.05)
        n = max(1, round(dur / self.physics.control_dt))
        wall_start, sim_start = time.time(), self.sim_t
        wall_deadline = wall_start + max(timeout_s, 0.1)
        o0 = self.open_cmd
        off0, off1 = list(self.offset), self._offset_target
        self._offset_target = None
        for i in range(1, n + 1):
            u = i / n
            self.hand_cmd = [a + (b - a) * u for a, b in zip(start, goal, strict=True)]
            self.open_cmd = o0 + (opening - o0) * u
            if off1 is not None:
                self.offset = [a + (b - a) * u for a, b in zip(off0, off1, strict=True)]
            self._step()
            if time.time() > wall_deadline:
                self._last_wall = time.time()
                return False
        if settle:
            for _ in range(round(0.5 / self.physics.control_dt)):
                pose, _ = self.physics.hand()
                if (
                    math.dist(self._actual(goal)[:3], pose[:3]) < 0.001
                    and abs(_wrap(pose[3] - goal[3])) < 0.01
                ):
                    break
                self._step()
        self._last_wall = time.time()
        self._measure_rate(self.sim_t - sim_start, self._last_wall - wall_start)
        return True

    def _measure_rate(self, sim_s: float, wall_s: float) -> None:
        if wall_s < 0.05 or sim_s <= 0:
            return
        rate = sim_s / wall_s
        previous = self.measured_real_time_factor
        self.measured_real_time_factor = rate if previous is None else 0.7 * previous + 0.3 * rate

    def _real_time_factor(self) -> dict:
        if self.declared_real_time_factor is not None:
            return {
                "real_time_factor": self.declared_real_time_factor,
                "real_time_factor_source": "declared",
            }
        if self.measured_real_time_factor is not None:
            return {
                "real_time_factor": round(self.measured_real_time_factor, 4),
                "real_time_factor_source": "measured over commanded motions (EMA)",
            }
        return {}

    def _ramp_gripper(self, start: float, width: float, dur: float) -> None:
        n = max(1, round(dur / self.physics.control_dt))
        for i in range(1, n + 1):
            self.open_cmd = start + (width - start) * min(
                1.0, i / max(1, n - round(0.3 / self.physics.control_dt))
            )
            self._step()
        self._last_wall = time.time()

    def _contacts(self) -> dict:
        return self.physics.truth().get("contacts") or {}

    def _held_tile(self) -> str | None:
        """Both pads touch the same tile and the measured jaw gap is closed onto it (at most
        the tile's width + 2 mm): a fallen tile lying across open jaws is not held."""
        if self._measured_open() > self.spec.tile.width + 0.002:
            return None
        for tid, touching in self._contacts().items():
            if all(p in touching for p in GRIPPER):
                return tid
        return None

    def _on_close(self) -> None:
        self.closes += 1
        self.events.append(
            {"event": "close", "sim_s": self.sim_t, "wall_s": time.time(), "index": self.closes - 1}
        )

    def _on_release(self, tile: str) -> None:
        self.releases += 1
        pose, _ = self.physics.hand()
        self.events.append(
            {
                "event": "release",
                "sim_s": self.sim_t,
                "wall_s": time.time(),
                "tile": tile,
                "index": self.releases - 1,
                "hand_z": pose[2],
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
                "geometry": {
                    "type": "box",
                    "size_m": [
                        self.spec.tile.height,
                        self.spec.tile.width,
                        self.spec.tile.thickness,
                    ],
                },
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
        pose, opening = self.physics.hand()
        reported = [pose[0] - self.offset[0], pose[1] - self.offset[1], pose[2], pose[3]]
        objects = self._objects(now)
        if self._frozen is not None:
            objects = {k: v for k, v in copy.deepcopy(self._frozen).items()}
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
        holding = self._held_tile() is not None
        return {
            "pose": {
                "frame_id": "world",
                "position_m": reported[:3],
                "quaternion_xyzw": tcp_quaternion(reported[3]),
            },
            "timestamp_s": now,
            "sim_time_s": self.sim_t,
            "objects": objects,
            "gripper": {
                "holding": holding,
                "width_m": max(0.0, opening),
                "source": "simulation_contacts",
                "timestamp_s": now,
            },
        }
