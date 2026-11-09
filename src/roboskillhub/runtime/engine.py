"""Bounded feedback-based primitives, independent of robot and simulator APIs."""

from __future__ import annotations

import math
import threading
import time
import uuid

from .contracts import (
    CAPABILITY_BUNDLES,
    EvidenceMissing,
    ExecutionFailed,
    Limits,
    Rejected,
    StationBackend,
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
}


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

    def invoke(self, capability: str, request: dict) -> dict:
        start = time.monotonic()
        try:
            validate_request(capability, request)
            budget = number(
                request.get("timeout_s", min(30, self.limits.max_timeout_s)),
                "timeout_s",
                0.1,
                self.limits.max_timeout_s,
            )
        except Rejected as exc:
            result = self._result(capability)
            result.update(
                error={"code": type(exc).__name__, "message": str(exc)},
                elapsed_s=time.monotonic() - start,
            )
            return result
        remaining = start + budget - time.monotonic()
        if remaining <= 0 or not self._lock.acquire(timeout=max(0, remaining)):
            result = self._result(capability)
            result.update(
                error={
                    "code": "QueueTimeout",
                    "message": "skill deadline expired while waiting for station ownership; no action issued",
                },
                elapsed_s=time.monotonic() - start,
            )
            return result
        try:
            return self._invoke(capability, request, start=start)
        finally:
            self._lock.release()

    @staticmethod
    def _result(capability):
        return {
            "schema_version": 1,
            "invocation_id": str(uuid.uuid4()),
            "capability": capability,
            "skill": CAPABILITY_BUNDLES.get(capability) if isinstance(capability, str) else None,
            "version": "0.1.0",
            "status": "rejected",
            "evidence": {},
            "trace": [],
        }

    def _invoke(self, capability: str, request: dict, *, start: float) -> dict:
        result = self._result(capability)
        self.trace = result["trace"]
        self.command_issued = False
        self.last_command_at = 0.0
        lease_acquired = False
        try:
            validate_request(capability, request)
            if capability not in REQUIRES:
                raise Rejected("this runtime executes motion capabilities; use API for tracking")
            description = self.backend.describe()
            result["backend"] = description
            missing = REQUIRES[capability] - set(description.get("capabilities", []))
            if missing:
                raise Rejected(f"backend lacks capabilities: {sorted(missing)}")
            if description.get("frame_id") != self.limits.frame_id:
                raise Rejected("backend frame does not match configured limits")
            self.deadline = start + number(
                request.get("timeout_s", min(30, self.limits.max_timeout_s)),
                "timeout_s",
                0.1,
                self.limits.max_timeout_s,
            )
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
            if begin is not None or end is not None:
                if not callable(begin) or not callable(end):
                    raise Rejected("backend must provide both invocation ownership methods")
                begin(timeout_s=self._remaining())
                lease_acquired = True
                self.trace.append({"event": "invocation_lease", "acquired": True})
            self._state(after=0)
            initial = (
                self._object(request["target_id"], after=0) if capability != "move_pose" else None
            )
            if capability == "move_pose":
                result["evidence"] = self._move(plan["target_pose"], linear=False)
            elif capability == "grasp":
                self._gripper(plan["open_width_m"])
                self._move(plan["pregrasp_pose"], linear=False)
                self._move(plan["grasp_pose"])
                self._gripper(plan["close_width_m"])
                self._check_empty_gripper()
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
                for waypoint in plan["waypoints"]:
                    self._move(waypoint)
                final = self._object(request["target_id"], after=self.last_command_at)
                result["evidence"] = self._verify_displacement(initial, final, plan["delta"])
                if capability == "pull_drawer" and request.get("release_on_success", False):
                    self._gripper(plan["open_width_m"])
            result["status"] = "succeeded"
        except (Rejected, EvidenceMissing, ExecutionFailed) as exc:
            result["status"] = (
                "unverified"
                if isinstance(exc, EvidenceMissing) and self.command_issued
                else "failed"
                if self.command_issued
                else "rejected"
            )
            result["error"] = {"code": type(exc).__name__, "message": str(exc)}
        except Exception as exc:
            # A broken driver must not be mistaken for successful execution.
            result["status"] = "failed" if self.command_issued else "rejected"
            result["error"] = {"code": "BackendError", "message": f"{type(exc).__name__}: {exc}"}
        finally:
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
                    result["ownership_error"] = str(exc)
            result["elapsed_s"] = time.monotonic() - start
        return result

    def _remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ExecutionFailed("skill deadline exceeded")
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
                "observation exceeds the maximum age or its clock is not synchronized"
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
            raise ExecutionFailed("skill deadline exceeded")
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
            raise EvidenceMissing(f"no independently observed object state for {target!r}")
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
            self._check_empty_gripper(state=state, after=after)
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
            raise ExecutionFailed("motion backend did not acknowledge completion")
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
            raise ExecutionFailed(f"TCP did not converge: {distance:.4f} m, {angle:.4f} rad")
        return evidence

    def _gripper(self, width: float) -> None:
        self._remaining()
        self.command_issued = True
        self.last_command_at = time.time()
        ack = self.backend.set_gripper(width, timeout_s=self._remaining())
        self._remaining()
        self.trace.append({"event": "set_gripper", "width_m": width, "reply": ack})
        if not isinstance(ack, dict) or ack.get("ok") is not True:
            raise ExecutionFailed("gripper command did not complete")
        self.last_command_at = time.time()

    def _check_empty_gripper(
        self, *, state: dict | None = None, after: float | None = None
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
            raise ExecutionFailed("gripper feedback reports no held object")

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
