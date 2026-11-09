# SPDX-License-Identifier: Apache-2.0
"""A workcell-runtime ``StationBackend`` over the LEGO physics scene (Isaac Sim 5.1 + BrickSim).

Same structure as 4b's ``domino_station.DominoPhysicsStation`` (and reusing its TCP helpers): a **floating
parallel-jaw hand** (x/y/z prismatic + yaw mount, two force-limited pads) above a table, serving only top-down
TCP poses. It is NOT a YAM arm and not RoCo's DexMate Vega U: no arm kinematics, reach or wrist limits.

Contract rsh-task/1.2 surface: ``describe / get_state / move_pose / set_gripper / stop`` (the runtime builds
``grasp``, ``move_waypoints``, ``set_gripper(open|close|release)`` and ``observe_objects`` on these) plus
``guarded_move`` for ``guarded_press``:

- objects: every brick (bottom centre, contract frame: +z up the studs, +x along studs[0] = the 2-stud side),
  the baseplate (top-surface centre) and ``table``; all ``source: simulation_truth``, ``privileged: true``.
- ``gripper.holding`` from pad contacts (both pads touching the same brick, measured jaw gap at most the
  brick's long side + 2 mm), never from the command; ``width_m`` is the measured jaw gap.
- clock: physics advances only inside this object; ``get_state`` first advances it by the wall time elapsed
  since the last step (capped), every state is stamped with ``time.time()`` and carries ``sim_time_s``.
- contact force (``contact_signal``): the z drive's force ``K (z - z_cmd) + D dz/dt`` from the measured joint
  position and velocity, tared at the start of the press, positive when the hand is pushed up; one sample per
  physics step (120 Hz), available the same step. It is a joint-force estimate inside a simulator, not an F/T
  sensor. The stop-on-force loop runs in this process, in the physics stepping loop (``guarded_move``).
- controller: ``impedance`` - the z axis is a spring-damper (K = 4000 N/m, D = 120 N s/m) whose set-point is
  ramped down; the press stops when the force estimate reaches ``force_limit_N`` (set-point frozen at the
  measured position) or the commanded travel reaches ``max_travel_m``.

For the independent evaluator the station records snapshots (brick poses, engine connections, contact
assembly events) at the physical events it sees: close, first move after a close, press start, release, first
move after a release; ``lego_world`` turns them into ``lego_postconditions`` records. It never reads a skill
result.
"""

from __future__ import annotations

import math
import time

from .domino_station import tcp_quaternion, yaw_from_tcp
from .lego_geometry import q_from_axis_angle, q_mul, q_rotate

Q_BRICKSIM_TO_CONTRACT = q_from_axis_angle([0.0, 0.0, 1.0], math.pi / 2)
"""Contract lego_brick frame = BrickSim part frame turned 90 deg about z (BrickSim's x runs along the
4-stud side of a 2x4; the contract's x runs along studs[0] = 2)."""

PLATE_HEIGHT_M = 0.0032
BRICK_SIZE_M = (0.016, 0.032, 0.0096)  # contract frame x, y, z of a 2x4


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class LegoPhysicsStation:
    """``StationBackend`` over a ``LegoIsaacPhysics`` (or a test double with the same methods)."""

    def __init__(
        self,
        physics,
        scene: dict,
        *,
        breakage: bool = False,
        max_catch_up_s: float = 1.0,
        force_limit_max_n: float = 10.0,
        max_press_travel_m: float = 0.02,
        max_press_speed_m_s: float = 0.02,
    ):
        self.physics = physics
        self.scene = scene
        self.paths = physics.reset(scene, breakage=breakage)
        self.brick_ids = [f["id"] for f in scene["feeder"]] + list(scene.get("prebuilt_bricks", []))
        self.max_catch_up_s = max_catch_up_s
        self.force_limit_max_n = force_limit_max_n
        self.max_press_travel_m = max_press_travel_m
        self.max_press_speed_m_s = max_press_speed_m_s
        self.hand_cmd, self.open_cmd = list(physics._cmd), physics._open
        self.sim_t = physics._t
        self._last_wall = time.time()
        self.stopped = False
        self.events: list[dict] = []
        self.engine_events: list[dict] = []
        self.snapshots: list[dict] = []
        self.command_log: list[dict] = []
        self._after_close = False
        self._after_release = False
        self._held_at_release: str | None = None

    # ---- runtime surface --------------------------------------------------
    def describe(self) -> dict:
        h = self.physics.hand_spec
        return {
            "frame_id": "world",
            "backend": f"lego-physics-{self.physics.name}",
            "kind": "dynamic_sim_privileged",
            "evidence_level": "dynamic_sim_privileged",
            "physics": True,
            "robot": "floating parallel-jaw hand (x/y/z prismatic + yaw mount, two force-limited pads); "
            "NOT a YAM arm and NOT RoCo's DexMate Vega U: no arm kinematics, reach, wrist or joint limits",
            "tcp_convention": "TCP z down (approach), TCP y = closing axis, pad tips 10.2 mm past the TCP "
            "(Agent 1's YAM linear_4310 convention); only top-down poses are served",
            "observations": "simulation truth (brick poses, BrickSim connections, pad contacts), privileged",
            "clock": f"simulation advanced to track wall time in get_state (catch-up capped at "
            f"{self.max_catch_up_s} s); timestamps are time.time() at the read",
            "physics_backend": self.physics.describe(),
            "capabilities": [
                "pose_feedback",
                "cartesian_motion",
                "gripper",
                "gripper_feedback",
                "stop",
                "object_feedback",
                "object_pose_feedback",
                "contact_force_feedback",
                "compliant_motion",
            ],
            "contact_signal": self.contact_signal(),
            "controller": {
                "kind": "impedance",
                "loop_hz": round(1.0 / self.physics.control_dt),
                "process": "Isaac Sim physics process: guarded_move steps PhysX and checks the force every step",
                "stiffness_N_m": h.xyz_stiffness,
                "damping_N_s_m": h.xyz_damping,
                "axes": "world -z only (top-down presses)",
            },
            "contact_limits": {
                "max_force_N": self.force_limit_max_n,
                "max_speed_m_s": self.max_press_speed_m_s,
                "max_travel_m": self.max_press_travel_m,
            },
            "grip_force_N": h.grip_force,
        }

    def contact_signal(self) -> dict:
        return {
            "source": "isaac_sim:hand z prismatic drive force K(z - z_cmd) + D dz/dt from measured joint "
            "state, tared at press start (simulated joint-force estimate, not an F/T sensor)",
            "quantity": "force opposing the press along the axis",
            "unit": "N",
            "frame_id": "world",
            "rate_hz": round(1.0 / self.physics.control_dt),
            "max_latency_s": 2.0 * self.physics.control_dt,
            "calibration_id": "sim:tare-at-press-start",
            # BrickSim moves the brick to its seat the step the clutch triggers (align policy
            # MoveHoleCC): no insertion travel, no press-fit force rise for this signal to show.
            "insertion_force_observable": False,
            "insertion_model": "bricksim_snap_to_seat",
        }

    def contact_description(self) -> dict:
        """What Agent 3's ContactBackend.describe() expects (``lego.ports``)."""
        d = self.describe()
        return {
            "backend_id": d["backend"],
            "evidence_level": d["evidence_level"],
            "capabilities": d["capabilities"],
            "contact_signal": d["contact_signal"],
            "controller": {k: d["controller"][k] for k in ("kind", "loop_hz", "process")},
            "limits": d["contact_limits"],
            "privileged": False,
        }

    def get_state(self, *, timeout_s: float) -> dict:
        self._catch_up()
        return self._state()

    def move_pose(self, target: dict, *, speed_m_s: float, timeout_s: float, linear: bool) -> dict:
        self._catch_up()
        if target.get("frame_id") != "world":
            return {"ok": False, "error": "frame must be world"}
        yaw, off_down = yaw_from_tcp(target["quaternion_xyzw"])
        if off_down > math.radians(3):
            return {
                "ok": False,
                "error": f"floating hand serves top-down TCP poses only "
                f"({math.degrees(off_down):.1f} deg off vertical)",
            }
        goal = [*target["position_m"], yaw]
        ok = self._drive(goal, self.open_cmd, speed=max(1e-3, speed_m_s), timeout_s=timeout_s)
        self.command_log.append(
            {"wall_s": time.time(), "sim_s": self.sim_t, "move": goal, "ok": ok}
        )
        if self._after_close:
            self._after_close = False
            self._snap("grasped")
        if self._after_release:
            self._after_release = False
            self._snap("retreated")
        return {"ok": ok} if ok else {"ok": False, "error": "timeout before arrival"}

    def set_gripper(self, width_m: float, *, timeout_s: float) -> dict:
        self._catch_up()
        held = self._held_brick()
        widening = width_m > self._measured_open() + 0.002
        if widening and held is not None:
            self._held_at_release = held
        elif not widening:
            self._snap("pick_start")
        start = self._measured_open()
        dur = max(0.2, abs(width_m - start) / 2 / self.physics.hand_spec.finger_speed) + 0.3
        self._ramp_gripper(start, width_m, dur)
        if widening and self._held_at_release is not None:
            self.events.append(
                {
                    "event": "release",
                    "brick": self._held_at_release,
                    "sim_s": self.sim_t,
                    "wall_s": time.time(),
                }
            )
            self._snap("released", brick=self._held_at_release)
            self._held_at_release = None
            self._after_release = True
        elif not widening:
            self.events.append(
                {
                    "event": "close",
                    "brick": self._held_brick(),
                    "sim_s": self.sim_t,
                    "wall_s": time.time(),
                }
            )
            self._after_close = True
        self.command_log.append({"wall_s": time.time(), "sim_s": self.sim_t, "grip": width_m})
        return {"ok": True}

    def stop(self) -> None:
        pose, opening = self.physics.hand()
        self.hand_cmd = list(pose)
        self.open_cmd = opening
        self.stopped = True

    def guarded_move(
        self, *, axis, max_travel_m, speed_m_s, force_limit_N, hold_s, timeout_s
    ) -> dict:
        """Bounded press along ``axis`` with the force check inside the physics loop."""
        self._catch_up()
        v = axis["vector"]
        n = math.sqrt(sum(c * c for c in v))
        if axis.get("frame_id") != "world" or n < 1e-9 or -v[2] / n < math.cos(math.radians(2.0)):
            return {
                "command_issued": False,
                "stop_reason": "fault",
                "stop_confirmed": True,
                "samples": [],
                "fault": "the floating hand presses only along world -z",
            }
        if force_limit_N > self.force_limit_max_n or max_travel_m > self.max_press_travel_m:
            return {
                "command_issued": False,
                "stop_reason": "fault",
                "stop_confirmed": True,
                "samples": [],
                "fault": "request beyond the station's contact limits",
            }
        dt = self.physics.control_dt
        speed = min(speed_m_s, self.max_press_speed_m_s)
        held = self._held_brick()
        self._snap("before_place", brick=held)
        bz0 = self.physics.truth()["objects"][held]["position"][2] if held else None

        def diag() -> dict:
            # privileged diagnostics for the evaluator report; not part of the contact signal
            if held is None:
                return {}
            return {
                "brick_dz_m": bz0 - self.physics.truth()["objects"][held]["position"][2],
                "interface_force_N": self.physics.interface_forces().get(held, 0.0),
            }

        pose, _ = self.physics.hand()
        z0 = pose[2]
        cmd = [self.hand_cmd[0], self.hand_cmd[1], z0, self.hand_cmd[3]]
        tare_n = 6
        tare = 0.0
        for _ in range(tare_n):
            self._step(cmd)
            tare += self._drive_force(cmd[2]) / tare_n
        t0 = self.sim_t
        deadline = time.time() + max(0.1, timeout_s)
        samples: list[dict] = []
        reason = None
        settle = 0
        while reason is None:
            if time.time() > deadline:
                reason = "timeout"
                break
            cmd[2] = max(z0 - max_travel_m, cmd[2] - speed * dt)
            self._step(cmd)
            f = self._drive_force(cmd[2]) - tare
            z = self.physics.hand()[0][2]
            samples.append({"t_s": self.sim_t, "travel_m": z0 - z, "force_N": f, **diag()})
            if f >= force_limit_N:
                reason = "force_threshold"
            elif cmd[2] <= z0 - max_travel_m + 1e-12:
                vz = abs(self.physics.hand_velocity()[2])
                settle = settle + 1 if vz < 0.002 else 0
                if settle >= 5:
                    reason = "travel_limit"
        cmd[2] = self.physics.hand()[0][2]  # freeze the set-point where the hand is
        self.hand_cmd = list(cmd)
        confirmed = False
        quiet = 0
        for k in range(max(1, round(max(hold_s, 0.5) / dt))):
            self._step(cmd)
            z = self.physics.hand()[0][2]
            samples.append(
                {
                    "t_s": self.sim_t,
                    "travel_m": z0 - z,
                    "force_N": self._drive_force(cmd[2]) - tare,
                    **diag(),
                }
            )
            quiet = quiet + 1 if abs(self.physics.hand_velocity()[2]) < 0.002 else 0
            if quiet >= 10 and k * dt >= hold_s:
                confirmed = True
                break
        self._last_wall = time.time()
        self.events.append(
            {
                "event": "press",
                "reason": reason,
                "sim_s": self.sim_t,
                "t0": t0,
                "peak_force_N": max((s["force_N"] for s in samples), default=None),
                "final_travel_m": samples[-1]["travel_m"] if samples else None,
                "force_limit_N": force_limit_N,
            }
        )
        return {
            "command_issued": True,
            "stop_reason": reason,
            "stop_confirmed": confirmed,
            "samples": samples,
            "fault": "",
        }

    # ---- independent-evaluation hooks ------------------------------------
    def settle(self, seconds: float) -> None:
        park = list(self.hand_cmd)
        park[2] = max(park[2], 0.12)
        self._drive(park, self.open_cmd, speed=0.05, timeout_s=10.0)
        self._advance(seconds)
        self._snap("final")

    def probe(self, brick: str, weight_ratio: float, steps: int, observe_steps: int) -> dict:
        m = self.physics.mass(brick)
        force = weight_ratio * m * 9.81
        z0 = self.physics.truth()["objects"][brick]["position"][2]
        rise = 0.0
        for k in range(steps + observe_steps):
            if k < steps:
                self.physics.apply_up_force(brick, force)
            self._step(self.hand_cmd)
            rise = max(rise, self.physics.truth()["objects"][brick]["position"][2] - z0)
        self._advance(0.25)
        return {"probe_max_lift_m": rise, "probe_force_n": force, "mass_kg": m}

    # ---- internals ---------------------------------------------------------
    def _drive_force(self, z_cmd: float) -> float:
        h = self.physics.hand_spec
        z = self.physics.hand()[0][2]
        vz = self.physics.hand_velocity()[2]
        return h.xyz_stiffness * (z - z_cmd) + h.xyz_damping * vz

    def _measured_open(self) -> float:
        return self.physics.hand()[1]

    def _step(self, cmd=None) -> None:
        if cmd is not None:
            self.hand_cmd = list(cmd)
        self.sim_t = self.physics.step(self.hand_cmd, self.open_cmd)
        for e in self.physics.drain_events():
            self.engine_events.append({**e, "sim_s": self.sim_t, "wall_s": time.time()})

    def _advance(self, seconds: float) -> None:
        for _ in range(max(0, round(seconds / self.physics.control_dt))):
            self._step()
        self._last_wall = time.time()

    def _catch_up(self) -> None:
        now = time.time()
        self._advance(min(self.max_catch_up_s, max(0.0, now - self._last_wall)))

    def _drive(self, goal, opening, *, speed, timeout_s) -> bool:
        start = list(self.hand_cmd)
        goal = list(goal)
        goal[3] = start[3] + _wrap(goal[3] - start[3])
        dist = math.dist(start[:3], goal[:3])
        dur = max(dist / speed, abs(goal[3] - start[3]) / math.radians(90), 0.05)
        n = max(1, round(dur / self.physics.control_dt))
        wall_deadline = time.time() + max(timeout_s, 0.1)
        o0 = self.open_cmd
        for i in range(1, n + 1):
            u = i / n
            self.hand_cmd = [a + (b - a) * u for a, b in zip(start, goal, strict=True)]
            self.open_cmd = o0 + (opening - o0) * u
            self._step()
            if time.time() > wall_deadline:
                self._last_wall = time.time()
                return False
        for _ in range(round(0.5 / self.physics.control_dt)):
            pose, _ = self.physics.hand()
            if math.dist(goal[:3], pose[:3]) < 0.0005 and abs(_wrap(pose[3] - goal[3])) < 0.01:
                break
            self._step()
        self._last_wall = time.time()
        return True

    def _ramp_gripper(self, start: float, width: float, dur: float) -> None:
        n = max(1, round(dur / self.physics.control_dt))
        tail = round(0.3 / self.physics.control_dt)
        for i in range(1, n + 1):
            self.open_cmd = start + (width - start) * min(1.0, i / max(1, n - tail))
            self._step()
        self._last_wall = time.time()

    def _held_brick(self) -> str | None:
        if self._measured_open() > BRICK_SIZE_M[1] + 0.002:
            return None
        c = self.physics.pad_contacts()
        common = c["gripper_pad_l"] & c["gripper_pad_r"]
        for b in self.brick_ids:
            if b in common:
                return b
        return None

    def _snap(self, label: str, brick: str | None = None) -> None:
        truth = self.physics.truth()
        self.snapshots.append(
            {
                "label": label,
                "brick": brick,
                "sim_s": self.sim_t,
                "wall_s": time.time(),
                "parts": {
                    k: {"p": o["position"], "q": o["quat_xyzw"]}
                    for k, o in truth["objects"].items()
                },
                "conns": self.physics.connections(),
                "n_engine_events": len(self.engine_events),
            }
        )

    def _objects(self, stamp: float) -> dict:
        truth = self.physics.truth()["objects"]
        out = {}
        for oid, o in truth.items():
            if oid == self.scene["plate_id"]:
                q = o["quat_xyzw"]
                top = q_rotate(q, [0.0, 0.0, PLATE_HEIGHT_M])
                out[oid] = {
                    "position_m": [a + b for a, b in zip(o["position"], top, strict=True)],
                    "quaternion_xyzw": list(q),
                    "class": "baseplate",
                    "geometry": {"type": "box", "size_m": [0.256, 0.256, PLATE_HEIGHT_M]},
                }
            else:
                out[oid] = {
                    "position_m": list(o["position"]),
                    "quaternion_xyzw": q_mul(o["quat_xyzw"], Q_BRICKSIM_TO_CONTRACT),
                    "class": "lego_brick",
                    "geometry": {"type": "box", "size_m": list(BRICK_SIZE_M)},
                    "studs": [2, 4],
                }
        out["table"] = {
            "position_m": [0.3, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "class": "support",
            "geometry": {"type": "box", "size_m": [0.8, 0.8, 0.02]},
        }
        for o in out.values():
            o.update(
                frame_id="world",
                timestamp_s=stamp,
                sim_time_s=self.sim_t,
                source="simulation_truth",
                privileged=True,
                confidence=1.0,
                visibility="visible",
                uncertainty={"position_m": 0.0001, "rotation_rad": 0.001},
            )
        return out

    def _state(self) -> dict:
        now = time.time()
        pose, opening = self.physics.hand()
        return {
            "pose": {
                "frame_id": "world",
                "position_m": list(pose[:3]),
                "quaternion_xyzw": tcp_quaternion(pose[3]),
            },
            "timestamp_s": now,
            "sim_time_s": self.sim_t,
            "objects": self._objects(now),
            "gripper": {
                "holding": self._held_brick() is not None,
                "width_m": max(0.0, opening),
                "source": "simulation_contacts",
                "timestamp_s": now,
            },
        }
