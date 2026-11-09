"""Legacy Isaac Core adapter for an already initialized, single articulation.

This module creates no stage, robot asset, or SimulationApp. Call it on the
standalone simulation thread, not inside a physics callback. The caller owns
World/reset, robot URDF/USD and Lula configuration, joint drive tuning, collision
checking, sensors, and gripper calibration. Joint targets are applied through
ArticulationAction; robot state is never teleported. IK is NOT a collision planner.

API reference (5.1 family; free-air YAM move exercised on 6.2.0-rc.14):
https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.robot_motion.motion_generation/docs/index.html
https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.api/docs/index.html
https://docs.isaacsim.omniverse.nvidia.com/5.1.0/manipulators/manipulators_lula_kinematics.html

ArticulationKinematicsSolver.compute_end_effector_pose returns stage-unit
position plus a 3x3 rotation matrix. IK consumes scalar-first wxyz quaternions.
The public Hub interface remains metres and xyzw in the configured USD world
frame. Explicit stage scale is required. Deadlines are checked between synchronous
Isaac/provider calls; a stuck native call requires an external process watchdog.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import EvidenceMissing, ExecutionFailed, Limits, Rejected, number, pose_error


def _matrix_to_xyzw(rotation: Any) -> list[float]:
    matrix = np.asarray(rotation, dtype=float)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5)
        or not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-5)
    ):
        raise EvidenceMissing("Isaac FK returned an invalid rotation matrix")
    # Symmetric eigenproblem avoids unstable divisions around 180-degree turns.
    r = matrix
    k = (
        np.array(
            [
                [
                    r[0, 0] - r[1, 1] - r[2, 2],
                    r[0, 1] + r[1, 0],
                    r[0, 2] + r[2, 0],
                    r[2, 1] - r[1, 2],
                ],
                [
                    r[0, 1] + r[1, 0],
                    r[1, 1] - r[0, 0] - r[2, 2],
                    r[1, 2] + r[2, 1],
                    r[0, 2] - r[2, 0],
                ],
                [
                    r[0, 2] + r[2, 0],
                    r[1, 2] + r[2, 1],
                    r[2, 2] - r[0, 0] - r[1, 1],
                    r[1, 0] - r[0, 1],
                ],
                [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
            ]
        )
        / 3.0
    )
    _, eigvectors = np.linalg.eigh(k)
    q = eigvectors[:, -1]
    if q[3] < 0:
        q = -q
    return q.tolist()


def _slerp(a: Sequence[float], b: Sequence[float], fraction: float) -> list[float]:
    q0, q1 = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + fraction * (q1 - q0)
    else:
        theta = math.acos(min(1.0, dot))
        q = (math.sin((1 - fraction) * theta) * q0 + math.sin(fraction * theta) * q1) / math.sin(
            theta
        )
    return (q / np.linalg.norm(q)).tolist()


class IsaacBackend:
    """Bridge initialized World + SingleArticulation + ArticulationKinematicsSolver.

    ``width_to_positions(width_m)`` must return targets for exactly the configured
    gripper DOF indices, in their USD joint units; it is never inferred from a
    robot name. Object/gripper providers are optional callables accepting
    ``timeout_s=`` and returning records with their ORIGINAL Unix timestamp and
    source. A gripper provider may establish holding; joint closure alone cannot.
    ``gripper_completion`` is optional station-specific completion evidence. It
    receives keyword arguments ``width_m``, ``current_positions``,
    ``target_positions`` and ``timeout_s``. Return False/None while incomplete,
    or a dict with complete=True, requested_width_m, timestamp_s, source and a
    reason such as contact_limited. The evidence must be collected after this
    command and identify its requested width; old close evidence cannot finish
    a new open command. Completion does NOT imply that an object is held.
    A stop preserves the last completed gripper target, so stopping an arm does
    not remove a completed contact close's drive preload. An unfinished gripper
    command is cancelled at measured positions instead. Recreate this backend
    after World.reset or externally replacing drive targets. Rewinding available
    World clocks invalidates saved targets, but a reset whose clocks return to
    the same or a later value cannot be reliably detected by this adapter.
    ``action_factory`` is normally omitted; injection supports CPU contract tests.
    """

    def __init__(
        self,
        *,
        world: Any,
        robot: Any,
        kinematics_solver: Any,
        limits: Limits,
        stage_units_in_meters: float,
        gripper_joint_indices: Sequence[int],
        width_to_positions: Callable[[float], Sequence[float]],
        object_provider: Callable[..., dict] | None = None,
        gripper_state_provider: Callable[..., dict] | None = None,
        gripper_completion: Callable[..., dict | bool | None] | None = None,
        action_factory: Callable[..., Any] | None = None,
        max_rotation_speed_rad_s: float = 0.5,
        gripper_joint_tolerance: float = 0.001,
        render: bool = False,
    ) -> None:
        self.world, self.robot, self.kinematics_solver = world, robot, kinematics_solver
        self.limits = limits
        self.scale = number(stage_units_in_meters, "stage_units_in_meters", 1e-6, 1e3)
        self.max_rotation_speed = number(
            max_rotation_speed_rad_s, "max_rotation_speed_rad_s", 1e-4, 2
        )
        self.gripper_tolerance = number(
            gripper_joint_tolerance, "gripper_joint_tolerance", 1e-6, 0.1
        )
        indices = list(gripper_joint_indices)
        if (
            not indices
            or any(type(i) is not int or i < 0 for i in indices)
            or len(set(indices)) != len(indices)
        ):
            raise Rejected("gripper_joint_indices must be distinct nonnegative integers")
        if not callable(width_to_positions):
            raise Rejected("a calibrated width_to_positions function is required")
        self.gripper_indices = np.asarray(indices, dtype=int)
        self.width_to_positions = width_to_positions
        self.object_provider, self.gripper_provider = object_provider, gripper_state_provider
        if gripper_completion is not None and not callable(gripper_completion):
            raise Rejected("gripper_completion must be callable when provided")
        self.gripper_completion = gripper_completion
        self.render = render
        if action_factory is None:
            from isaacsim.core.utils.types import ArticulationAction

            action_factory = ArticulationAction
        self.action_factory = action_factory
        self._stopped = False
        self._completed_gripper_targets: np.ndarray | None = None
        self._completed_gripper_clock: dict[str, float] | None = None

    def describe(self) -> dict:
        capabilities = ["pose_feedback", "cartesian_motion", "gripper", "stop"]
        if self.object_provider is not None:
            capabilities.append("object_feedback")
            # A provider states it reports quaternion_xyzw; simulator truth stays privileged.
            if getattr(self.object_provider, "provides_orientation", False) is True:
                capabilities.append("object_pose_feedback")
        if self.gripper_provider is not None:
            capabilities.append("gripper_feedback")
        return {
            "backend": "isaac",
            "kind": "sim",
            "embodiment": "yam_i2rt",
            "embodiment_binding": "caller-supplied articulation and kinematics; not station-validated",
            "api_family": "Isaac Sim legacy Core API",
            "frame_id": self.limits.frame_id,
            "frame_semantics": "USD stage world",
            "units": "m",
            "quaternion_order": "xyzw",
            "capabilities": capabilities,
            "object_evidence_available": self.object_provider is not None,
            "holding_evidence_available": self.gripper_provider is not None,
            "gripper_contact_completion_available": self.gripper_completion is not None,
            "collision_planning": False,
            "station_validated": False,
            "execution": "joint targets + physics stepping; no state teleportation",
            "timeout_semantics": "wall and physics-step budgets; native calls must return",
            "stop_semantics": "measured arm hold; preserve completed gripper target; cancel unfinished gripper movement",
            "reset_semantics": "recreate backend after World.reset; observed clock rewind invalidates saved gripper target",
        }

    def _world_clock(self) -> dict[str, float] | None:
        """Read optional World counters without advancing or resetting physics.

        An empty mapping means this World exposes no counters; None means an
        advertised counter is unavailable/invalid. Neither can prove an epoch.
        """
        result = {}
        for name in ("current_time_step_index", "current_time"):
            try:
                value = getattr(self.world, name)
            except AttributeError:
                continue
            except Exception:
                return None
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
                or value < 0
            ):
                return None
            result[name] = float(value)
        return result

    def _refresh_gripper_epoch(self) -> None:
        if self._completed_gripper_targets is None:
            return
        current, previous = self._world_clock(), self._completed_gripper_clock
        if (
            current is None
            or previous is None
            or not previous.keys() <= current.keys()
            or any(current[name] < value for name, value in previous.items())
        ):
            self._completed_gripper_targets = None
            self._completed_gripper_clock = None
            return
        # Remember the most recently observed counters, so a rewind after a
        # later arm movement is also detected even if it stays above close time.
        self._completed_gripper_clock = current

    def _deadline(self, timeout_s: float) -> float:
        timeout = number(timeout_s, "timeout_s", 1e-6, self.limits.max_timeout_s)
        return time.monotonic() + timeout

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ExecutionFailed("Isaac operation timed out")
        return remaining

    def _sync_base(self) -> None:
        position, orientation = self.robot.get_world_pose()
        self.kinematics_solver.get_kinematics_solver().set_robot_base_pose(position, orientation)

    def _pose(self) -> dict:
        self._sync_base()
        position, rotation = self.kinematics_solver.compute_end_effector_pose()
        position = np.asarray(position, dtype=float).reshape(-1)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise EvidenceMissing("Isaac FK returned an invalid position")
        return {
            "frame_id": self.limits.frame_id,
            "position_m": (position * self.scale).tolist(),
            "quaternion_xyzw": _matrix_to_xyzw(rotation),
        }

    def _gripper_state(self, deadline: float) -> dict:
        if self.gripper_provider is None:
            return {"holding": None}
        result = copy.deepcopy(self.gripper_provider(timeout_s=self._remaining(deadline)))
        self._remaining(deadline)
        if not isinstance(result, dict) or not (
            result.get("holding") is None or type(result.get("holding")) is bool
        ):
            raise EvidenceMissing("gripper provider must report holding as bool or null")
        self._evidence(result, "gripper")
        return result

    @staticmethod
    def _evidence(record: dict, name: str) -> None:
        if (
            not isinstance(record.get("source"), str)
            or not record["source"].strip()
            or isinstance(record.get("timestamp_s"), bool)
            or not isinstance(record.get("timestamp_s"), (int, float))
            or not math.isfinite(record["timestamp_s"])
        ):
            raise EvidenceMissing(f"{name} provider must preserve a source and Unix timestamp_s")

    def get_state(self, *, timeout_s: float) -> dict:
        deadline = self._deadline(timeout_s)
        self._refresh_gripper_epoch()
        pose = self._pose()
        timestamp = time.time()
        self._remaining(deadline)
        objects = {}
        if self.object_provider is not None:
            objects = copy.deepcopy(self.object_provider(timeout_s=self._remaining(deadline)))
            self._remaining(deadline)
            if not isinstance(objects, dict):
                raise EvidenceMissing("object provider must return an object-id mapping")
            for name, record in objects.items():
                if not isinstance(name, str) or not isinstance(record, dict):
                    raise EvidenceMissing("object provider returned an invalid record")
                self._evidence(record, name)
                p = np.asarray(record.get("position_m"), dtype=float)
                if p.shape != (3,) or not np.isfinite(p).all():
                    raise EvidenceMissing(f"{name} provider position must be 3 finite metres")
                if record.get("frame_id", self.limits.frame_id) != self.limits.frame_id:
                    raise EvidenceMissing(f"{name} provider frame differs from USD world frame")
        return {
            "pose": pose,
            "timestamp_s": timestamp,
            "objects": objects,
            "gripper": self._gripper_state(deadline),
            "source": "simulation_truth",
        }

    def _begin(self, timeout_s: float) -> tuple[float, float, int]:
        deadline = self._deadline(timeout_s)
        self._refresh_gripper_epoch()
        dt = number(float(self.world.get_physics_dt()), "physics_dt", 1e-6, 1)
        if not self.world.is_playing():
            raise Rejected("Isaac World must already be playing; adapter does not start/reset it")
        budget = math.floor(timeout_s / dt)
        if budget < 1:
            raise Rejected("timeout_s must cover at least one Isaac physics step")
        self._stopped = False
        return deadline, dt, budget

    def _step(self, deadline: float) -> None:
        self._remaining(deadline)
        if self._stopped or not self.world.is_playing():
            raise ExecutionFailed("Isaac motion stopped or timeline paused")
        self.world.step(render=self.render)
        self._remaining(deadline)
        self._refresh_gripper_epoch()
        if self._stopped or not self.world.is_playing():
            raise ExecutionFailed("Isaac motion stopped or timeline paused")

    def _abort(self, error: Exception) -> None:
        try:
            self.stop()
        except Exception as stop_error:
            raise ExecutionFailed(f"{error}; joint-hold stop also failed: {stop_error}") from error
        raise error

    def move_pose(self, pose: dict, *, speed_m_s: float, timeout_s: float, linear: bool) -> dict:
        target = self.limits.pose(pose)
        speed = number(speed_m_s, "speed_m_s", 1e-5, self.limits.max_speed_m_s)
        if type(linear) is not bool:
            raise Rejected("linear must be boolean")
        deadline, dt, budget = self._begin(timeout_s)
        start = self._pose()
        distance, angle = pose_error(start, target)
        segments = max(1, math.ceil(max(distance / speed, angle / self.max_rotation_speed) / dt))
        if segments > budget:
            raise Rejected("requested speed/path cannot fit the physics-time timeout budget")
        steps = 0
        try:
            for index in range(1, segments + 1):
                self._remaining(deadline)
                if self._stopped:
                    raise ExecutionFailed("Isaac motion stopped")
                fraction = index / segments
                position = np.asarray(start["position_m"]) + fraction * (
                    np.asarray(target["position_m"]) - start["position_m"]
                )
                q = _slerp(start["quaternion_xyzw"], target["quaternion_xyzw"], fraction)
                self._sync_base()
                action, success = self.kinematics_solver.compute_inverse_kinematics(
                    target_position=position / self.scale,
                    target_orientation=np.asarray([q[3], q[0], q[1], q[2]]),
                    position_tolerance=self.limits.position_tolerance_m / self.scale,
                    orientation_tolerance=self.limits.rotation_tolerance_rad,
                )
                self._remaining(deadline)
                if not success or action is None:
                    raise ExecutionFailed(
                        f"Isaac IK failed at Cartesian segment {index}/{segments}"
                    )
                self.robot.apply_action(action)
                self._step(deadline)
                steps += 1
            while True:
                actual = self._pose()
                self._remaining(deadline)
                error_m, error_rad = pose_error(actual, target)
                if (
                    error_m <= self.limits.position_tolerance_m
                    and error_rad <= self.limits.rotation_tolerance_rad
                ):
                    return {
                        "ok": True,
                        "pose": actual,
                        "position_error_m": error_m,
                        "rotation_error_rad": error_rad,
                        "physics_steps": steps,
                        "path": "interpolated Cartesian targets; no collision planning",
                    }
                if steps >= budget:
                    raise ExecutionFailed("Isaac pose did not converge within physics-time timeout")
                self._step(deadline)
                steps += 1
        except Exception as error:
            self._abort(error)

    def set_gripper(self, width_m: float, *, timeout_s: float) -> dict:
        width = number(
            width_m, "width_m", self.limits.gripper_min_width_m, self.limits.gripper_max_width_m
        )
        targets = np.asarray(self.width_to_positions(width), dtype=float)
        if targets.shape != self.gripper_indices.shape or not np.isfinite(targets).all():
            raise Rejected("width_to_positions must provide one finite target per gripper joint")
        deadline, _dt, budget = self._begin(timeout_s)
        command_started = time.time()
        # A new gripper action supersedes the old completed command. Cancelling
        # this action must never restore an earlier close or open target.
        self._completed_gripper_targets = None
        self._completed_gripper_clock = None
        try:
            self.robot.apply_action(
                self.action_factory(joint_positions=targets, joint_indices=self.gripper_indices)
            )
            for step in range(1, budget + 1):
                self._step(deadline)
                positions = np.asarray(
                    self.robot.get_joint_positions(joint_indices=self.gripper_indices), dtype=float
                )
                if positions.shape != targets.shape or not np.isfinite(positions).all():
                    raise EvidenceMissing("Isaac gripper joint feedback is unavailable")
                state = self._gripper_state(deadline)
                reached = bool(np.max(np.abs(positions - targets)) <= self.gripper_tolerance)
                completion = None
                if not reached and self.gripper_completion is not None:
                    completion = self._gripper_completion_evidence(
                        width, positions, targets, deadline, command_started
                    )
                if reached or completion is not None:
                    if self._stopped:
                        raise ExecutionFailed("Isaac gripper command was stopped before completion")
                    self._completed_gripper_targets = targets.copy()
                    self._completed_gripper_clock = self._world_clock()
                    return {
                        "ok": True,
                        "requested_width_m": width,
                        "joint_positions": positions.tolist(),
                        "physics_steps": step,
                        "gripper": state,
                        "completion": completion
                        or {
                            "complete": True,
                            "reason": "joint_target_reached",
                            "source": "simulation_truth",
                            "timestamp_s": time.time(),
                            "requested_width_m": width,
                        },
                    }
            raise ExecutionFailed("Isaac gripper did not converge within physics-time timeout")
        except Exception as error:
            self._abort(error)

    def _gripper_completion_evidence(
        self,
        width: float,
        current: np.ndarray,
        targets: np.ndarray,
        deadline: float,
        command_started: float,
    ) -> dict | None:
        evidence = copy.deepcopy(
            self.gripper_completion(
                width_m=width,
                current_positions=current.copy(),
                target_positions=targets.copy(),
                timeout_s=self._remaining(deadline),
            )
        )
        self._remaining(deadline)
        if evidence is None or evidence is False:
            return None
        if not isinstance(evidence, dict) or type(evidence.get("complete")) is not bool:
            raise EvidenceMissing(
                "gripper completion must return false or a documented evidence record"
            )
        if not evidence["complete"]:
            return None
        self._evidence(evidence, "gripper completion")
        try:
            requested = number(
                evidence.get("requested_width_m"), "completion requested_width_m", 0, 1
            )
        except Rejected as error:
            raise EvidenceMissing(str(error)) from error
        if not math.isclose(requested, width, rel_tol=0, abs_tol=1e-9):
            raise EvidenceMissing(
                "gripper completion evidence belongs to a different width command"
            )
        now = time.time()
        stamp = evidence["timestamp_s"]
        if (
            stamp < command_started
            or now - stamp > self.limits.max_state_age_s
            or stamp - now > 0.1
        ):
            raise EvidenceMissing("gripper completion evidence is stale or predates this command")
        if not isinstance(evidence.get("reason"), str) or not evidence["reason"].strip():
            raise EvidenceMissing("gripper completion evidence requires a reason")
        return evidence

    def stop(self) -> None:
        """Hold measured arms and preserve only a completed gripper target.

        Preserving the command retains contact preload; it is not holding
        evidence. An unfinished gripper command uses measured positions instead,
        including when it superseded a previous completed close. A detected
        World clock rewind discards completed targets from the previous epoch.
        This is a drive hold, not a state teleport or an emergency-stop certificate.
        The owner must continue stepping the world for the drive action to apply.
        """
        self._stopped = True
        self._refresh_gripper_epoch()
        joints = np.asarray(self.robot.get_joint_positions(), dtype=float)
        if joints.ndim != 1 or not len(joints) or not np.isfinite(joints).all():
            raise EvidenceMissing("cannot hold without measured Isaac joint positions")
        hold = joints.copy()
        if self._completed_gripper_targets is not None:
            if np.max(self.gripper_indices) >= len(hold):
                raise EvidenceMissing(
                    "completed gripper target no longer matches articulation DOFs"
                )
            hold[self.gripper_indices] = self._completed_gripper_targets
        self.robot.apply_action(
            self.action_factory(joint_positions=hold, joint_velocities=np.zeros_like(joints))
        )


def simulation_object_provider(
    objects: Mapping[str, Any],
    *,
    frame_id: str,
    stage_units_in_meters: float,
    include_orientation: bool = False,
) -> Callable[..., dict]:
    """Observe named Isaac prims and explicitly label their states as sim truth.

    Each value must implement get_world_pose(). The label denotes that prim's
    origin (e.g. a handle marker), not a guessed object centroid or task success.
    With ``include_orientation`` the prim's scalar-first world quaternion is also
    reported, as ``quaternion_xyzw``; every record is marked ``privileged``.
    """
    scale = number(stage_units_in_meters, "stage_units_in_meters", 1e-6, 1e3)
    if not frame_id:
        raise Rejected("simulation object provider requires a USD world frame_id")

    def observe(*, timeout_s: float) -> dict:
        deadline = time.monotonic() + number(timeout_s, "timeout_s", 1e-6, 120)
        records = {}
        for name, prim in objects.items():
            position, orientation = prim.get_world_pose()
            if time.monotonic() >= deadline:
                raise EvidenceMissing("simulation object observation timed out")
            records[name] = {
                "position_m": (np.asarray(position, dtype=float) * scale).tolist(),
                "timestamp_s": time.time(),
                "source": "simulation_truth",
                "privileged": True,
                "frame_id": frame_id,
            }
            if include_orientation:
                w, x, y, z = (float(v) for v in np.asarray(orientation, dtype=float))
                records[name]["quaternion_xyzw"] = [x, y, z, w]
        return records

    observe.provides_orientation = include_orientation
    return observe
