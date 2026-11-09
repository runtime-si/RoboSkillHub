"""Bounded feedback-based primitives, independent of robot and simulator APIs."""

from __future__ import annotations

import contextlib
import math
import threading
import time
import uuid

from .contracts import (
    CAPABILITY_BUNDLES,
    CONTRACT_VERSION,
    EvidenceMissing,
    ExecutionFailed,
    Limits,
    Rejected,
    StationBackend,
    compose,
    number,
    pose_error,
    translated,
    unit_vector,
    validate_pose,
    validate_request,
    vector,
)

REQUIRES = {
    "move_pose": {"pose_feedback", "cartesian_motion", "stop"},
    "grasp": {"pose_feedback", "cartesian_motion", "gripper", "object_feedback", "stop"},
    "push": {"pose_feedback", "cartesian_motion", "object_feedback", "stop"},
    "pull_drawer": {"pose_feedback", "cartesian_motion", "gripper", "object_feedback", "stop"},
    "set_gripper": {"pose_feedback", "gripper", "stop"},
    "move_waypoints": {"pose_feedback", "cartesian_motion", "stop"},
    "observe_objects": {"object_feedback"},
}

SIM_TRUTH_SOURCES = frozenset({"simulation_truth", "sim_truth"})

RECOVERY_HINTS = {
    "station_busy": ["retry_same"],
    "timeout": ["back_off", "reobserve"],
    "motion_not_converged": ["back_off", "replan"],
    "unreachable": ["replan"],
    "evidence_missing": ["reobserve"],
    "stale_observation": ["reobserve"],
    "object_not_observed": ["reobserve"],
    "grasp_missed": ["reobserve", "regrasp"],
    "grasp_lost": ["reobserve", "regrasp"],
    "held_object_slipped": ["reobserve", "regrasp"],
    "release_failed": ["retry_same", "human"],
    "postcondition_failed": ["reobserve", "replan"],
    "cancelled": ["abort"],
    "lease_expired": ["abort"],
    "backend_error": ["human"],
}


def required_capabilities(capability: str, request: dict) -> set[str]:
    """Backend capabilities a request needs, including those its optional fields add."""
    needed = set(REQUIRES[capability])
    if capability == "move_waypoints":
        if request.get("linear") is True:
            needed.add("linear_cartesian_motion")
        if "held" in request:
            needed.add("object_feedback")
    if capability == "observe_objects" and request.get("require_orientation") is True:
        needed.add("object_pose_feedback")
    return needed


class TaskLease:
    """Station ownership for a whole task graph; obtain it from ``Runtime.task_lease``.

    Primitives invoked with this lease do not reacquire the Runtime lock or the
    backend's invocation lease. There is no preemption: only the holder's
    ``cancel`` or the deadline ends it early.
    """

    def __init__(self, runtime: Runtime, task_id: str, deadline: float):
        self.runtime = runtime
        self.task_id = task_id
        self.deadline = deadline
        self.active = True
        self.cancel_reason: str | None = None
        self._serial = threading.Lock()

    def cancel(self, reason: str = "cancelled by lease holder") -> None:
        self.cancel_reason = reason

    def remaining(self) -> float:
        return self.deadline - time.monotonic()


class Runtime:
    """One instance serializes all movement for one configured station.

    Transport acknowledgements are never task success. Every move checks fresh
    measured TCP pose; manipulation checks independently observed object motion.
    The backend owns collision checking, control rate and cancellation.
    """

    def __init__(self, backend: StationBackend, limits: Limits):
        self.backend = backend
        self.limits = limits
        self._lock = threading.Lock()
        self._lease_owner: int | None = None
        self.phase = "validate"
        self.active_lease: TaskLease | None = None

    @contextlib.contextmanager
    def task_lease(self, task_id: str, timeout_s: float):
        """Hold the station for one task: Runtime lock plus backend invocation lease."""
        if not isinstance(task_id, str) or not task_id.strip():
            raise Rejected("task_id must be a nonempty string")
        budget = number(timeout_s, "timeout_s", 0.1, self.limits.max_task_s)
        if self._lease_owner == threading.get_ident():
            raise Rejected("this thread already holds a task lease", failure_type="nested_lease")
        start = time.monotonic()
        if not self._lock.acquire(timeout=budget):
            raise Rejected(
                "station is owned by another task until the lease deadline",
                failure_type="station_busy",
            )
        lease = TaskLease(self, task_id, start + budget)
        backend_lease = False
        try:
            begin = getattr(self.backend, "begin_invocation", None)
            end = getattr(self.backend, "end_invocation", None)
            if begin is not None or end is not None:
                if not callable(begin) or not callable(end):
                    raise Rejected("backend must provide both invocation ownership methods")
                begin(timeout_s=max(0.001, lease.remaining()))
                backend_lease = True
            self._lease_owner = threading.get_ident()
            yield lease
        finally:
            lease.active = False
            self._lease_owner = None
            try:
                if backend_lease:
                    self.backend.end_invocation()
            finally:
                self._lock.release()

    def invoke(
        self,
        capability: str,
        request: dict,
        *,
        lease: TaskLease | None = None,
        task_id: str | None = None,
        request_id: str | None = None,
    ) -> dict:
        start = time.monotonic()
        ids = {"task_id": lease.task_id if lease else task_id, "request_id": request_id}
        try:
            validate_request(capability, request)
            budget = number(
                request.get("timeout_s", min(30, self.limits.max_timeout_s)),
                "timeout_s",
                0.1,
                self.limits.max_timeout_s,
            )
            if lease is not None:
                if lease.runtime is not self:
                    raise Rejected("lease belongs to another station", failure_type="nested_lease")
                if lease.cancel_reason is not None:
                    raise Rejected(lease.cancel_reason, failure_type="cancelled")
                if not lease.active or lease.remaining() <= 0:
                    raise Rejected("task lease is no longer active", failure_type="lease_expired")
            elif self._lease_owner == threading.get_ident():
                raise Rejected(
                    "this thread holds a task lease; pass it instead of reacquiring the station",
                    failure_type="nested_lease",
                )
        except Rejected as exc:
            result = self._result(capability, **ids)
            result.update(
                error={"code": type(exc).__name__, "message": str(exc)},
                failure_type=exc.failure_type,
                recovery_hints=RECOVERY_HINTS.get(exc.failure_type, []),
                elapsed_s=time.monotonic() - start,
            )
            return result
        if lease is not None:
            budget = min(budget, lease.remaining())
            lock = lease._serial
        else:
            lock = self._lock
        remaining = start + budget - time.monotonic()
        if remaining <= 0 or not lock.acquire(timeout=max(0, remaining)):
            result = self._result(capability, **ids)
            result.update(
                error={
                    "code": "QueueTimeout",
                    "message": "skill deadline expired while waiting for station ownership; no action issued",
                },
                failure_type="station_busy",
                recovery_hints=RECOVERY_HINTS["station_busy"],
                elapsed_s=time.monotonic() - start,
            )
            return result
        try:
            result = self._invoke(capability, request, start=start, lease=lease)
            result.update({k: v for k, v in ids.items() if v is not None})
            return result
        finally:
            lock.release()

    @staticmethod
    def _result(capability, *, task_id=None, request_id=None):
        result = {
            "schema_version": 1,
            "contract_version": CONTRACT_VERSION,
            "invocation_id": str(uuid.uuid4()),
            "capability": capability,
            "skill": CAPABILITY_BUNDLES.get(capability) if isinstance(capability, str) else None,
            "version": "0.1.0",
            "kind": "primitive",
            "status": "rejected",
            "phase": "validate",
            "failure_type": None,
            "recovery_hints": [],
            "evidence": {},
            "trace": [],
        }
        if task_id is not None:
            result["task_id"] = task_id
        if request_id is not None:
            result["request_id"] = request_id
        return result

    def _invoke(
        self, capability: str, request: dict, *, start: float, lease: TaskLease | None = None
    ) -> dict:
        result = self._result(capability)
        self.trace = result["trace"]
        self.command_issued = False
        self.observation_attempted = False
        self.last_command_at = 0.0
        self.active_lease = lease
        self.phase = "validate"
        self.result = result
        lease_acquired = False
        try:
            validate_request(capability, request)
            if capability not in REQUIRES:
                raise Rejected("this runtime executes motion capabilities; use API for tracking")
            description = self.backend.describe()
            result["backend"] = description
            missing = required_capabilities(capability, request) - set(
                description.get("capabilities", [])
            )
            if missing:
                raise Rejected(
                    f"backend lacks capabilities: {sorted(missing)}",
                    failure_type="capability_missing",
                )
            if description.get("frame_id") != self.limits.frame_id:
                raise Rejected(
                    "backend frame does not match configured limits",
                    failure_type="precondition_unmet",
                )
            self.deadline = start + number(
                request.get("timeout_s", min(30, self.limits.max_timeout_s)),
                "timeout_s",
                0.1,
                self.limits.max_timeout_s,
            )
            if lease is not None:
                self.deadline = min(self.deadline, lease.deadline)
                self.trace.append({"event": "task_lease", "task_id": lease.task_id})
            self.speed = number(
                request.get("speed_m_s", min(0.03, self.limits.max_speed_m_s)),
                "speed_m_s",
                1e-5,
                self.limits.max_speed_m_s,
            )
            # Validate the complete geometric plan, before opening/moving anything.
            plan = self._prepare(capability, request)
            self._remaining()  # Includes all time spent waiting for this Runtime.
            begin = getattr(self.backend, "begin_invocation", None)
            end = getattr(self.backend, "end_invocation", None)
            if lease is None and (begin is not None or end is not None):
                if not callable(begin) or not callable(end):
                    raise Rejected("backend must provide both invocation ownership methods")
                begin(timeout_s=self._remaining())
                lease_acquired = True
                self.trace.append({"event": "invocation_lease", "acquired": True})
            if capability == "observe_objects":
                self.phase = "observe"
                self.observation_attempted = True
                result["evidence"] = self._observe(plan)
                result["status"] = "succeeded"
                return result
            self._state(after=0)
            initial = (
                self._object(request["target_id"], after=0)
                if capability in ("grasp", "push", "pull_drawer")
                else None
            )
            if capability == "set_gripper":
                result["evidence"] = self._set_gripper(plan)
            elif capability == "move_waypoints":
                self.phase = "transport"
                result["evidence"] = self._waypoints(plan)
            elif capability == "move_pose":
                self.phase = "transport"
                result["evidence"] = self._move(plan["target_pose"], linear=False)
            elif capability == "grasp":
                self.phase = "approach"
                self._gripper(plan["open_width_m"])
                self._move(plan["pregrasp_pose"], linear=False)
                self.phase = "grasp"
                self._move(plan["grasp_pose"])
                self._gripper(plan["close_width_m"])
                self._check_empty_gripper()
                self.phase = "lift"
                # Verify transport during the lift, not incidental movement
                # caused by approaching or closing the gripper.
                initial = self._object(request["target_id"], after=self.last_command_at)
                self._move(plan["lift_pose"])
                final = self._object(
                    request["target_id"], after=self.last_command_at, check_holding=True
                )
                expected = [
                    b - a
                    for a, b in zip(
                        plan["grasp_pose"]["position_m"],
                        plan["lift_pose"]["position_m"],
                        strict=True,
                    )
                ]
                result["evidence"] = self._verify_displacement(initial, final, expected, grasp=True)
            else:
                self.phase = "approach"
                if capability == "pull_drawer":
                    self._gripper(plan["open_width_m"])
                    self._move(plan["pregrasp_pose"], linear=False)
                    self._move(plan["grasp_pose"])
                    self._gripper(plan["close_width_m"])
                    self._check_empty_gripper()
                else:
                    self._move(plan["approach_pose"], linear=False)
                    self._move(plan["contact_pose"])
                initial = self._object(request["target_id"], after=self.last_command_at)
                self.phase = "contact"
                for waypoint in plan["waypoints"]:
                    self._move(waypoint)
                final = self._object(request["target_id"], after=self.last_command_at)
                result["evidence"] = self._verify_displacement(initial, final, plan["delta"])
                if capability == "pull_drawer" and request.get("release_on_success", False):
                    self._gripper(plan["open_width_m"])
            result["status"] = "succeeded"
        except (Rejected, EvidenceMissing, ExecutionFailed) as exc:
            attempted = self.command_issued or self.observation_attempted
            result["status"] = (
                "unverified"
                if isinstance(exc, EvidenceMissing) and attempted
                else "failed"
                if self.command_issued
                else "rejected"
            )
            result["error"] = {"code": type(exc).__name__, "message": str(exc)}
            result["failure_type"] = exc.failure_type
        except Exception as exc:
            # A broken driver must not be mistaken for successful execution.
            result["status"] = "failed" if self.command_issued else "rejected"
            result["error"] = {"code": "BackendError", "message": f"{type(exc).__name__}: {exc}"}
            result["failure_type"] = "backend_error"
        finally:
            result["phase"] = self.phase
            if result["status"] != "succeeded" and self.command_issued:
                try:
                    self.backend.stop()
                    self.trace.append({"event": "stop", "ok": True})
                except Exception as exc:
                    self.trace.append({"event": "stop", "ok": False, "error": str(exc)})
            if lease_acquired:
                try:
                    self.backend.end_invocation()
                    self.trace.append({"event": "invocation_release", "ok": True})
                except Exception as exc:
                    self.trace.append(
                        {"event": "invocation_release", "ok": False, "error": str(exc)}
                    )
                    if result["status"] == "succeeded":
                        result["status"] = "unverified"
                        result["failure_type"] = "evidence_missing"
                    result["ownership_error"] = str(exc)
            if result["status"] != "succeeded":
                result["recovery_hints"] = RECOVERY_HINTS.get(result["failure_type"], [])
            result["elapsed_s"] = time.monotonic() - start
        return result

    def _remaining(self) -> float:
        lease = self.active_lease
        if lease is not None and lease.cancel_reason is not None:
            raise ExecutionFailed(lease.cancel_reason, failure_type="cancelled")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ExecutionFailed("skill deadline exceeded", failure_type="timeout")
        return remaining

    def _prepare(self, capability: str, request: dict) -> dict:
        plan = dict(request)
        for key in (
            "target_pose",
            "pregrasp_pose",
            "grasp_pose",
            "lift_pose",
            "approach_pose",
            "contact_pose",
        ):
            if key in request:
                plan[key] = self.limits.pose(request[key])
        for key in ("open_width_m", "close_width_m"):
            if key in request:
                plan[key] = number(
                    request[key],
                    key,
                    self.limits.gripper_min_width_m,
                    self.limits.gripper_max_width_m,
                )
        if capability == "set_gripper":
            plan["width_m"] = number(
                request["width_m"],
                "width_m",
                self.limits.gripper_min_width_m,
                self.limits.gripper_max_width_m,
            )
        if capability == "move_waypoints":
            plan["waypoints"] = [self.limits.pose(w) for w in request["waypoints"]]
        if "open_width_m" in plan and plan["open_width_m"] <= plan["close_width_m"]:
            raise Rejected("open_width_m must exceed close_width_m")
        if capability == "grasp":
            lift = [
                b - a
                for a, b in zip(
                    plan["grasp_pose"]["position_m"], plan["lift_pose"]["position_m"], strict=True
                )
            ]
            if lift[2] < 0.02 or math.dist(lift, [0, 0, 0]) > self.limits.max_contact_distance_m:
                raise Rejected(
                    "grasp requires at least 2 cm upward lift, within contact distance limit"
                )
        if capability in ("push", "pull_drawer"):
            direction = unit_vector(request["direction_world"], "direction_world")
            if capability == "push" and abs(direction[2]) > 0.01:
                raise Rejected("push is planar in station XY; vertical pushing is unsupported")
            distance = number(
                request["distance_m"], "distance_m", 0.005, self.limits.max_contact_distance_m
            )
            step = number(request.get("step_m", 0.01), "step_m", 0.001, 0.02)
            count = math.ceil(distance / step)
            if count > 128:
                raise Rejected("contact path exceeds 128 bounded segments")
            start = plan["contact_pose" if capability == "push" else "grasp_pose"]
            plan["delta"] = [x * distance for x in direction]
            plan["waypoints"] = [
                self.limits.pose(translated(start, [x * (i + 1) / count for x in plan["delta"]]))
                for i in range(count)
            ]
        return plan

    def _timestamp_ready(self, timestamp: object, *, after: float) -> bool:
        try:
            ts = number(timestamp, "observation timestamp_s", 0, 1e12)
        except Rejected as exc:
            raise EvidenceMissing(str(exc)) from exc
        now = time.time()
        if now - ts > self.limits.max_state_age_s or ts - now > 0.1:
            raise EvidenceMissing(
                "observation exceeds the maximum age or its clock is not synchronized",
                failure_type="stale_observation",
            )
        return ts > after

    def _fresh(self, timestamp: object, *, after: float) -> None:
        if not self._timestamp_ready(timestamp, after=after):
            raise EvidenceMissing("observation was not acquired after this action completed")

    def _feedback_remaining(self, *, waiting: bool) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            if waiting:
                raise EvidenceMissing("deadline expired waiting for post-command observations")
            raise ExecutionFailed("skill deadline exceeded", failure_type="timeout")
        return remaining

    def _state(self, *, after: float, object_target: str | None = None) -> dict:
        # A 20/30 Hz state publisher can legitimately return a sample acquired
        # milliseconds before the motion ACK. Poll only such recent old samples;
        # malformed, over-age, missing, or wrong-frame evidence fails immediately.
        waiting = False
        while True:
            budget = self._feedback_remaining(waiting=waiting)
            try:
                state = self.backend.get_state(timeout_s=budget)
            except (TimeoutError, ExecutionFailed) as exc:
                if waiting and (isinstance(exc, TimeoutError) or time.monotonic() >= self.deadline):
                    raise EvidenceMissing(
                        "timed out waiting for post-command observations"
                    ) from exc
                raise
            self._feedback_remaining(waiting=waiting)
            if not isinstance(state, dict):
                raise EvidenceMissing("backend state is not an object")
            state = dict(state)
            try:
                state["pose"] = validate_pose(state.get("pose"), self.limits.frame_id)
            except Rejected as exc:
                raise EvidenceMissing(f"invalid measured TCP pose: {exc}") from exc
            ready = self._timestamp_ready(state.get("timestamp_s"), after=after)
            if object_target is not None:
                item = self._object_record(state, object_target)
                object_ready = self._timestamp_ready(item.get("timestamp_s"), after=after)
                ready = ready and object_ready
            if ready:
                return state
            if not waiting:
                self.trace.append(
                    {
                        "event": "feedback_wait",
                        "after_timestamp_s": after,
                        "state_timestamp_s": state.get("timestamp_s"),
                        "target_id": object_target,
                    }
                )
            waiting = True
            time.sleep(min(0.01, self._feedback_remaining(waiting=True)))

    def _object_record(self, state: dict, target: str) -> dict:
        objects = state.get("objects", {})
        item = objects.get(target) if isinstance(objects, dict) else None
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("source"), str)
            or not item["source"]
        ):
            raise EvidenceMissing(
                f"no independently observed object state for {target!r}",
                failure_type="object_not_observed",
            )
        try:
            position = vector(item.get("position_m"), "object position_m", 3)
        except Rejected as exc:
            raise EvidenceMissing(str(exc)) from exc
        if item.get("frame_id") != self.limits.frame_id:
            raise EvidenceMissing("object observation frame does not match station frame")
        return {"target_id": target, **item, "position_m": position}

    def _object(self, target: str, *, after: float, check_holding: bool = False) -> dict:
        state = self._state(after=after, object_target=target)
        observed = self._object_record(state, target)
        self.trace.append({"event": "object_observation", **observed})
        if check_holding:
            self._check_empty_gripper(state=state, after=after, failure_type="grasp_lost")
        return observed

    def _move(self, pose: dict, *, linear: bool = True) -> dict:
        self._remaining()
        self.command_issued = True
        self.last_command_at = time.time()
        ack = self.backend.move_pose(
            pose, speed_m_s=self.speed, timeout_s=self._remaining(), linear=linear
        )
        self._remaining()
        self.trace.append({"event": "move_pose", "target": pose, "linear": linear, "reply": ack})
        if not isinstance(ack, dict) or ack.get("ok") is not True:
            raise ExecutionFailed(
                "motion backend did not acknowledge completion",
                failure_type="motion_not_converged",
            )
        self.last_command_at = time.time()
        state = self._state(after=self.last_command_at)
        distance, angle = pose_error(state["pose"], pose)
        evidence = {
            "measured_pose": state["pose"],
            "state_timestamp_s": state["timestamp_s"],
            "position_error_m": distance,
            "rotation_error_rad": angle,
        }
        self.trace.append({"event": "pose_verification", **evidence})
        if (
            distance > self.limits.position_tolerance_m
            or angle > self.limits.rotation_tolerance_rad
        ):
            raise ExecutionFailed(
                f"TCP did not converge: {distance:.4f} m, {angle:.4f} rad",
                failure_type="motion_not_converged",
            )
        return evidence

    def _gripper(self, width: float) -> None:
        self._remaining()
        self.command_issued = True
        self.last_command_at = time.time()
        ack = self.backend.set_gripper(width, timeout_s=self._remaining())
        self._remaining()
        self.trace.append({"event": "set_gripper", "width_m": width, "reply": ack})
        if not isinstance(ack, dict) or ack.get("ok") is not True:
            raise ExecutionFailed("gripper command did not complete", failure_type="backend_error")
        self.last_command_at = time.time()

    def _check_empty_gripper(
        self,
        *,
        state: dict | None = None,
        after: float | None = None,
        failure_type: str = "grasp_missed",
    ) -> None:
        after = self.last_command_at if after is None else after
        if state is None:
            state = self._state(after=after)
        feedback = state.get("gripper", {})
        if not isinstance(feedback, dict):
            raise EvidenceMissing("gripper feedback must be an object")
        holding = feedback.get("holding")
        if holding is not None:
            self._fresh(feedback.get("timestamp_s"), after=after)
            if type(holding) is not bool or not feedback.get("source"):
                raise EvidenceMissing(
                    "gripper holding feedback requires a boolean and measurement source"
                )
        self.trace.append({"event": "gripper_verification", "feedback": feedback})
        if holding is False:
            raise ExecutionFailed(
                "gripper feedback reports no held object", failure_type=failure_type
            )

    def _verify_displacement(
        self, initial: dict, final: dict, expected: list[float], *, grasp: bool = False
    ) -> dict:
        actual = [b - a for a, b in zip(initial["position_m"], final["position_m"], strict=True)]
        distance = math.sqrt(sum(x * x for x in expected))
        unit = [x / distance for x in expected]
        progress = sum(x * y for x, y in zip(actual, unit, strict=True))
        lateral = math.sqrt(sum((x - progress * y) ** 2 for x, y in zip(actual, unit, strict=True)))
        error = math.dist(actual, expected)
        evidence = {
            "initial": initial,
            "final": final,
            "observed_displacement_m": actual,
            "expected_displacement_m": expected,
            "progress_m": progress,
            "lateral_error_m": lateral,
            "displacement_error_m": error,
            "criterion": "object-follows-lift" if grasp else "observed-translation",
        }
        self.trace.append({"event": "outcome_verification", **evidence})
        # Outcome thresholds are fixed in K, rather than freely enlarged by an agent.
        if (
            progress < 0.7 * distance
            or lateral > 0.02
            or (grasp and error > 0.02)
            or progress > 1.5 * distance + 0.01
        ):
            raise ExecutionFailed("observed object motion did not satisfy the skill postcondition")
        return evidence

    def _gripper_feedback(self, state: dict, *, after: float) -> dict:
        feedback = state.get("gripper", {})
        if not isinstance(feedback, dict):
            raise EvidenceMissing("gripper feedback must be an object")
        reported = {
            k: feedback.get(k) for k in ("holding", "width_m") if feedback.get(k) is not None
        }
        if reported:
            self._fresh(feedback.get("timestamp_s"), after=after)
            if not feedback.get("source"):
                raise EvidenceMissing("gripper feedback requires a measurement source")
            if "holding" in reported and type(reported["holding"]) is not bool:
                raise EvidenceMissing("gripper holding feedback must be boolean")
            if "width_m" in reported:
                reported["width_m"] = number(reported["width_m"], "gripper width_m", 0, 1)
        return {
            **reported,
            "source": feedback.get("source"),
            "timestamp_s": feedback.get("timestamp_s"),
        }

    def _set_gripper(self, plan: dict) -> dict:
        purpose = plan["purpose"]
        self.phase = {"release": "release", "close": "grasp", "open": "approach"}[purpose]
        before = (
            self._object(plan["target_id"], after=0) if plan.get("target_id") is not None else None
        )
        self._gripper(plan["width_m"])
        after = self.last_command_at
        state = self._state(after=after)
        feedback = self._gripper_feedback(state, after=after)
        evidence = {
            "purpose": purpose,
            "commanded_width_m": plan["width_m"],
            "gripper_feedback": feedback,
            "object_before": before,
        }
        if plan.get("target_id") is not None:
            evidence["object_after"] = self._object(plan["target_id"], after=after)
        self.trace.append({"event": "gripper_outcome", **evidence})
        holding = feedback.get("holding")
        width = feedback.get("width_m")
        at_width = width is not None and (
            abs(width - plan["width_m"]) <= self.limits.gripper_width_tolerance_m
        )
        if purpose == "release":
            if holding is True:
                raise ExecutionFailed(
                    "gripper still reports a held object after release",
                    failure_type="release_failed",
                )
            if holding is None:
                raise EvidenceMissing("no holding feedback to establish the release")
        elif purpose == "close":
            if holding is False:
                raise ExecutionFailed("gripper closed on nothing", failure_type="grasp_missed")
            if holding is None and not at_width:
                raise EvidenceMissing("no holding or width feedback to establish the closure")
        elif not at_width:
            if width is None:
                raise EvidenceMissing("no measured width to establish the opening")
            raise ExecutionFailed(
                f"gripper width {width:.4f} m differs from the commanded {plan['width_m']:.4f} m",
                failure_type="postcondition_failed",
            )
        evidence["criterion"] = {
            "release": "holding-false-after-command",
            "close": "holding-true-or-measured-width",
            "open": "measured-width-within-tolerance",
        }[purpose]
        return evidence

    def _waypoints(self, plan: dict) -> dict:
        linear = plan["linear"]
        arrivals = [self._move(waypoint, linear=linear) for waypoint in plan["waypoints"]]
        evidence = {"arrivals": arrivals, "linear": linear}
        held = plan.get("held")
        if held is not None:
            target = held["object_id"]
            state = self._state(after=self.last_command_at, object_target=target)
            observed = self._object_record(state, target)
            self.trace.append({"event": "object_observation", **observed})
            self._check_empty_gripper(
                state=state, after=self.last_command_at, failure_type="grasp_lost"
            )
            predicted = compose(state["pose"], held["T_TCP_object"])["position_m"]
            error = math.dist(predicted, observed["position_m"])
            evidence["held"] = {
                "object_id": target,
                "observed": observed,
                "predicted_position_m": predicted,
                "error_m": error,
                "tolerance_m": self.limits.held_object_tolerance_m,
                "criterion": "observed-object-follows-tcp",
            }
            self.trace.append({"event": "outcome_verification", **evidence["held"]})
            if error > self.limits.held_object_tolerance_m:
                raise ExecutionFailed(
                    f"held object is {error:.4f} m from where T_TCP_object puts it",
                    failure_type="held_object_slipped",
                )
        return evidence

    def _observe(self, plan: dict) -> dict:
        state = self.backend.get_state(timeout_s=self._remaining())
        if not isinstance(state, dict):
            raise EvidenceMissing("backend state is not an object")
        objects = state.get("objects", {})
        objects = objects if isinstance(objects, dict) else {}
        now = time.time()
        observations, problems = [], []
        for target in plan["object_ids"]:
            item = objects.get(target)
            record, reason = self._observation(target, item, now, plan)
            observations.append(record)
            if reason is not None:
                problems.append(reason)
        evidence = {
            "observations": observations,
            "privileged": any(o["privileged"] for o in observations),
            "max_state_age_s": self.limits.max_state_age_s,
        }
        self.trace.append({"event": "object_observations", "count": len(observations)})
        self.result["evidence"] = evidence
        if problems:
            failure, message = problems[0]
            raise EvidenceMissing(message, failure_type=failure)
        return evidence

    def _observation(self, target: str, item: object, now: float, plan: dict):
        record = {
            "object_id": target,
            "frame_id": self.limits.frame_id,
            "position_m": None,
            "quaternion_xyzw": None,
            "timestamp_s": None,
            "source": "",
            "privileged": False,
            "confidence": None,
            "uncertainty": None,
            "visibility": "not_found",
            "state_label": None,
            "reason": None,
        }
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("source"), str)
            or not item["source"]
        ):
            record["reason"] = "no independently observed state"
            return record, ("object_not_observed", f"{target}: no independently observed state")
        record["source"] = item["source"]
        record["privileged"] = bool(item.get("privileged")) or item["source"] in SIM_TRUTH_SOURCES
        for key in ("class", "confidence", "uncertainty", "state_label", "geometry"):
            if key in item:
                record[key] = item[key]
        record["visibility"] = item.get("visibility", "visible")
        if item.get("frame_id") != self.limits.frame_id:
            record["reason"] = "observation frame does not match station frame"
            return record, ("evidence_missing", f"{target}: {record['reason']}")
        try:
            record["position_m"] = vector(item.get("position_m"), "position_m", 3)
            record["timestamp_s"] = number(item.get("timestamp_s"), "timestamp_s", 0, 1e12)
            if item.get("quaternion_xyzw") is not None:
                q = vector(item["quaternion_xyzw"], "quaternion_xyzw", 4)
                n = math.sqrt(sum(x * x for x in q))
                if abs(n - 1) > 0.01:
                    raise Rejected("quaternion_xyzw must have unit norm")
                record["quaternion_xyzw"] = [x / n for x in q]
        except Rejected as exc:
            record["position_m"] = None
            record["reason"] = str(exc)
            return record, ("evidence_missing", f"{target}: {exc}")
        if record["visibility"] in ("occluded", "not_found"):
            record["reason"] = record["visibility"]
            return record, ("object_not_observed", f"{target}: {record['visibility']}")
        age = now - record["timestamp_s"]
        if age > self.limits.max_state_age_s or age < -0.1:
            record["reason"] = f"observation age {age:.3f} s exceeds the limit"
            return record, ("stale_observation", f"{target}: {record['reason']}")
        if plan.get("require_orientation") and record["quaternion_xyzw"] is None:
            record["reason"] = "orientation required but not observed"
            return record, ("evidence_missing", f"{target}: {record['reason']}")
        return record, None
