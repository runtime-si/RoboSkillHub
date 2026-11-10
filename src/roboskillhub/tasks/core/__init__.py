"""Generic task skills composed from the rsh-task/1 primitives (Agent 5).

These are the shared steps both minimal graphs need: choose the next target,
observe, derive a grasp from an observed pose and a configured grasp offset,
pick, carry a held object, release and retreat. Task-specific postconditions
(a domino standing, a brick seated) belong to the domino and lego modules.
"""

from __future__ import annotations

import math
from typing import Any

from roboskillhub.runtime.contracts import translated, validate_pose
from roboskillhub.runtime.task import (
    SkillContext,
    compose,
    invert,
    tcp_for_object_goal,
    validate,
)

_POSE = {"$ref": "#/schemas/Pose"}
_NUM = {"type": "number"}


def _schema(required: list[str], **props: Any) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": props,
    }


def _observations(result: dict) -> list[dict]:
    return list(result.get("evidence", {}).get("observations", []))


def _object_observation(record: dict | None) -> list[dict]:
    """An Observation from a primitive's object evidence record (position only)."""
    if not isinstance(record, dict) or record.get("position_m") is None:
        return []
    source = record.get("source", "")
    return [
        {
            "object_id": record.get("target_id") or record.get("object_id"),
            "frame_id": record.get("frame_id"),
            "position_m": record["position_m"],
            "quaternion_xyzw": record.get("quaternion_xyzw"),
            "timestamp_s": record.get("timestamp_s"),
            "source": source,
            "privileged": bool(record.get("privileged"))
            or source in ("simulation_truth", "sim_truth"),
            "confidence": record.get("confidence"),
            "uncertainty": record.get("uncertainty"),
            "visibility": "visible",
        }
    ]


def _relay(skill, ctx: SkillContext, child: dict, phase: str, **extra) -> dict:
    """Report a primitive's non-success as this skill's result, keeping its failure type."""
    return ctx.result(
        skill,
        child["status"] if child["status"] != "succeeded" else "failed",
        phase=child.get("phase") or phase,
        failure_type=child.get("failure_type") or "backend_error",
        recovery_hints=child.get("recovery_hints", []),
        evidence={"child": child.get("capability"), "child_error": child.get("error"), **extra},
        observations=_observations(child),
    )


class SelectNextTarget:
    """Pick the first unfinished target whose dependencies are done, and a fresh piece."""

    skill_id = "core.select_next_target"
    version = "0.1.0"
    kind = "tool"
    requires = frozenset()
    readiness = "unit-tested"
    input_schema = _schema(
        ["targets", "done"],
        targets={"type": "array", "minItems": 1, "items": {"type": "object"}},
        done={"type": "array", "items": {"type": "string"}},
        pieces={"type": "array", "items": {"type": "string"}},
        used={"type": "array", "items": {"type": "string"}},
        placed={"type": "array", "items": {"type": "string"}},
        key={"type": "string"},
    )

    def run(self, request: dict, ctx: SkillContext) -> dict:
        key = request.get("key", "slot_id")
        done = list(request["done"])
        ids = [t.get(key) for t in request["targets"]]
        if None in ids or len(set(ids)) != len(ids):
            return ctx.result(self, "rejected", failure_type="invalid_request", phase="validate")
        remaining = [t for t in request["targets"] if t[key] not in done]
        if not remaining:
            used = list(request.get("used", []))
            return ctx.result(
                self,
                "succeeded",
                phase="validate",
                outputs={"complete": True, "done": done, "used_after": used},
            )
        ready = [t for t in remaining if set(t.get("depends_on", [])) <= set(done)]
        if not ready:
            return ctx.result(
                self,
                "failed",
                phase="validate",
                failure_type="precondition_unmet",
                evidence={"blocked": [t[key] for t in remaining]},
                recovery_hints=["replan"],
            )
        target = min(ready, key=lambda t: t.get("order", 0))
        outputs: dict[str, Any] = {
            "complete": False,
            "target": target,
            "target_id": target[key],
            "done_after": [*done, target[key]],
            "remaining": len(remaining),
        }
        if "pieces" in request:
            used = list(request.get("used", []))
            fresh = [p for p in request["pieces"] if p not in used]
            if not fresh:
                return ctx.result(
                    self,
                    "failed",
                    phase="validate",
                    failure_type="precondition_unmet",
                    evidence={"reason": "no unused piece left in the feeder"},
                    recovery_hints=["human"],
                )
            placed = list(request.get("placed", []))
            outputs.update(
                piece=fresh[0],
                used_after=[*used, fresh[0]],
                placed=placed,
                has_placed=bool(placed),
                placed_after=[*placed, fresh[0]],
            )
        return ctx.result(self, "succeeded", phase="validate", outputs=outputs)


class Observe:
    """Observe objects through ``observe_objects``; succeeded only if all are fresh and visible."""

    skill_id = "core.observe"
    version = "0.1.0"
    kind = "skill"
    requires = frozenset({"object_feedback"})
    readiness = "unit-tested"
    input_schema = _schema(
        ["object_ids"],
        object_ids={"type": "array", "minItems": 1, "maxItems": 32, "items": {"type": "string"}},
        require_orientation={"type": "boolean"},
    )

    def run(self, request: dict, ctx: SkillContext) -> dict:
        child = ctx.observe(
            request["object_ids"], require_orientation=request.get("require_orientation", False)
        )
        observations = _observations(child)
        poses = {o["object_id"]: o for o in observations}
        if child["status"] != "succeeded":
            return _relay(self, ctx, child, "observe", poses=poses)
        return ctx.result(
            self,
            "succeeded",
            phase="observe",
            evidence={
                "postcondition": {
                    "name": "objects_observed",
                    "passed": True,
                    "criterion": "every object fresh and visible",
                }
            },
            observations=observations,
            outputs={"poses": poses, "first": observations[0], "observations": observations},
        )


class TopdownGraspCandidate:
    """GraspCandidate from an observed object pose and a configured ``T_object_TCP``.

    The offset is calibration data for one piece type and gripper (supplied by the
    graph input), not inferred here. Requires an observed orientation.
    """

    skill_id = "core.grasp_candidate_from_pose"
    version = "0.1.0"
    kind = "tool"
    requires = frozenset()
    readiness = "unit-tested"
    input_schema = _schema(
        [
            "observation",
            "T_object_TCP",
            "pregrasp_height_m",
            "lift_height_m",
            "open_width_m",
            "close_width_m",
        ],
        observation={"type": "object"},
        T_object_TCP=_POSE,
        pregrasp_height_m=_NUM,
        lift_height_m=_NUM,
        open_width_m=_NUM,
        close_width_m=_NUM,
    )

    def run(self, request: dict, ctx: SkillContext) -> dict:
        obs = validate("Observation", request["observation"])
        if obs["position_m"] is None or obs["quaternion_xyzw"] is None:
            return ctx.result(
                self,
                "rejected",
                phase="validate",
                failure_type="evidence_missing",
                recovery_hints=["reobserve"],
                evidence={"reason": "a 6D observed pose is required"},
            )
        if (
            not 0.02 <= request["lift_height_m"] <= 0.2
            or not 0.0 < request["pregrasp_height_m"] <= 0.2
        ):
            return ctx.result(self, "rejected", phase="validate", failure_type="invalid_request")
        T_W_obj = {k: obs[k] for k in ("frame_id", "position_m", "quaternion_xyzw")}
        T_object_TCP = validate_pose(request["T_object_TCP"], "object")
        grasp = compose(T_W_obj, T_object_TCP)
        candidate = {
            "object_id": obs["object_id"],
            "pregrasp_pose": translated(grasp, [0, 0, request["pregrasp_height_m"]]),
            "grasp_pose": grasp,
            "lift_pose": translated(grasp, [0, 0, request["lift_height_m"]]),
            "open_width_m": request["open_width_m"],
            "close_width_m": request["close_width_m"],
            "T_TCP_object": invert(T_object_TCP, frame_id="tcp"),
            "score": 1.0,
            "checks": {"source": "configured T_object_TCP; no collision check"},
        }
        validate("GraspCandidate", candidate)
        return ctx.result(
            self,
            "succeeded",
            phase="validate",
            outputs={"candidate": candidate},
            observations=[obs],
        )


class Pick:
    """Execute a GraspCandidate with ``grasp``; output the HeldObject (geometry prior)."""

    skill_id = "core.pick"
    version = "0.1.0"
    kind = "skill"
    requires = frozenset(
        {"pose_feedback", "cartesian_motion", "gripper", "object_feedback", "stop"}
    )
    readiness = "unit-tested"
    input_schema = _schema(
        ["candidate", "gripper_id"],
        candidate={"type": "object"},
        gripper_id={"type": "string"},
        speed_m_s=_NUM,
        uncertainty={"$ref": "#/schemas/Uncertainty"},
        collision_geometry={"type": ["object", "null"]},
    )

    def run(self, request: dict, ctx: SkillContext) -> dict:
        c = validate("GraspCandidate", request["candidate"])
        primitive = {
            "pregrasp_pose": c["pregrasp_pose"],
            "grasp_pose": c["grasp_pose"],
            "lift_pose": c["lift_pose"],
            "target_id": c["object_id"],
            "open_width_m": c["open_width_m"],
            "close_width_m": c["close_width_m"],
        }
        if "speed_m_s" in request:
            primitive["speed_m_s"] = request["speed_m_s"]
        child = ctx.invoke("grasp", primitive)
        if child["status"] != "succeeded":
            return _relay(self, ctx, child, "grasp")
        held = {
            "object_id": c["object_id"],
            "gripper_id": request["gripper_id"],
            "T_TCP_object": c["T_TCP_object"],
            "estimate": "grasp_geometry_prior",
            "timestamp_s": ctx.now(),
            # A configured prior for this gripper and piece, not a measurement.
            "uncertainty": request.get("uncertainty"),
            "collision_geometry": request.get("collision_geometry"),
            "valid": True,
            "invalidated_reason": None,
        }
        validate("HeldObject", held)
        evidence = child["evidence"]
        return ctx.result(
            self,
            "succeeded",
            phase="lift",
            evidence={
                "postcondition": {
                    "name": "object_follows_lift",
                    "passed": True,
                    "criterion": evidence.get("criterion"),
                    "measured": {"displacement_error_m": evidence.get("displacement_error_m")},
                }
            },
            observations=_object_observation(evidence.get("final")),
            outputs={"held": held},
        )


class TransportHeld:
    """Carry a held object above (and optionally onto) an object goal pose."""

    skill_id = "core.transport_held"
    version = "0.1.0"
    kind = "skill"
    requires = frozenset({"pose_feedback", "cartesian_motion", "object_feedback", "stop"})
    readiness = "unit-tested"
    input_schema = _schema(
        ["held", "goal", "approach_height_m", "descend"],
        held={"type": "object"},
        goal=_POSE,
        approach_height_m=_NUM,
        descend={"type": "boolean"},
        speed_m_s=_NUM,
        clear_above_z_m=_NUM,
    )

    def run(self, request: dict, ctx: SkillContext) -> dict:
        held = validate("HeldObject", request["held"])
        if not held["valid"]:
            return ctx.result(
                self,
                "rejected",
                phase="validate",
                failure_type="precondition_unmet",
                recovery_hints=["reobserve", "regrasp"],
            )
        if not 0.005 <= request["approach_height_m"] <= 0.2:
            return ctx.result(self, "rejected", phase="validate", failure_type="invalid_request")
        tcp_goal = tcp_for_object_goal(request["goal"], held["T_TCP_object"])
        hover = translated(tcp_goal, [0, 0, request["approach_height_m"]])
        waypoints = []
        if "clear_above_z_m" in request:
            # Carry the object with its origin (a brick's bottom) above clear_above_z_m - over the build's
            # stud tops - and only then descend to the hover: a straight line from the lift down to the
            # hover sweeps the object through a flush neighbour's studs.
            lift_tcp = tcp_goal["position_m"][2] - request["goal"]["position_m"][2]
            travel_z = request["clear_above_z_m"] + lift_tcp
            if hover["position_m"][2] < travel_z:
                now = ctx.state()["tcp_pose"]
                if now["position_m"][2] < travel_z:
                    waypoints.append({**now, "position_m": [*now["position_m"][:2], travel_z]})
                waypoints.append({**hover, "position_m": [*hover["position_m"][:2], travel_z]})
        waypoints.append(hover)
        if request["descend"]:
            waypoints.append(tcp_goal)
        primitive = {
            "waypoints": waypoints,
            "linear": False,
            "held": {"object_id": held["object_id"], "T_TCP_object": held["T_TCP_object"]},
        }
        if "speed_m_s" in request:
            primitive["speed_m_s"] = request["speed_m_s"]
        child = ctx.invoke("move_waypoints", primitive)
        if child["status"] != "succeeded":
            out = _relay(self, ctx, child, "transport")
            if child.get("failure_type") in ("held_object_slipped", "grasp_lost"):
                out["outputs"] = {
                    "held": {**held, "valid": False, "invalidated_reason": "slip_detected"}
                }
            return out
        check = child["evidence"]["held"]
        return ctx.result(
            self,
            "succeeded",
            phase="transport",
            evidence={
                "postcondition": {
                    "name": "held_object_follows_tcp",
                    "passed": True,
                    "criterion": check["criterion"],
                    "measured": {"error_m": check["error_m"], "tolerance_m": check["tolerance_m"]},
                },
                "tcp_goal": tcp_goal,
            },
            observations=_object_observation(check["observed"]),
            outputs={"held": held, "tcp_goal": tcp_goal},
        )


class ConfirmHold:
    """Re-confirm a held object just before a step that needs fresh holding evidence.

    Reads the gripper's measured holding flag and observes the object; it must sit
    where ``TCP · T_TCP_object`` puts it. Re-emits the HeldObject stamped with the
    older of the two readings. No motion.
    """

    skill_id = "core.confirm_hold"
    version = "0.1.0"
    kind = "skill"
    requires = frozenset({"pose_feedback", "gripper_feedback", "object_feedback"})
    readiness = "unit-tested"
    input_schema = _schema(["held"], held={"type": "object"})

    def run(self, request: dict, ctx: SkillContext) -> dict:
        held = validate("HeldObject", request["held"])
        if not held["valid"]:
            return ctx.result(
                self,
                "rejected",
                phase="validate",
                failure_type="precondition_unmet",
                recovery_hints=["reobserve", "regrasp"],
            )
        try:
            state = ctx.state()
        except Exception as exc:  # refused or offline station
            return ctx.result(
                self,
                "unverified",
                phase="verify",
                failure_type="evidence_missing",
                evidence={"reason": f"state unavailable: {exc}"},
                recovery_hints=["reobserve"],
            )
        gripper = state.get("gripper") or {}
        if gripper.get("holding") is not True or not gripper.get("source"):
            dropped = {**held, "valid": False, "invalidated_reason": "grasp_lost"}
            return ctx.result(
                self,
                "failed" if gripper.get("holding") is False else "unverified",
                phase="verify",
                failure_type="grasp_lost"
                if gripper.get("holding") is False
                else "evidence_missing",
                evidence={"gripper": gripper},
                recovery_hints=["regrasp"],
                outputs={"held": dropped},
            )
        child = ctx.invoke(
            "observe_objects", {"object_ids": [held["object_id"]], "require_orientation": False}
        )
        if child["status"] != "succeeded":
            return _relay(self, ctx, child, "verify", gripper=gripper)
        seen = child["evidence"]["observations"][0]
        predicted = compose(state["tcp_pose"], held["T_TCP_object"])["position_m"]
        error = math.dist(predicted, seen["position_m"])
        tolerance = float(ctx.limits.get("held_object_tolerance_m", 0.01))
        passed = error <= tolerance
        stamp = min(float(gripper["timestamp_s"]), float(seen["timestamp_s"]))
        fresh = {**held, "timestamp_s": stamp}
        return ctx.result(
            self,
            "succeeded" if passed else "failed",
            phase="verify",
            failure_type=None if passed else "held_object_slipped",
            evidence={
                "postcondition": {
                    "name": "held_object_confirmed",
                    "passed": passed,
                    "criterion": "measured gripper holding and observed object within "
                    "held_object_tolerance_m of TCP · T_TCP_object",
                    "measured": {"error_m": error, "tolerance_m": tolerance, "gripper": gripper},
                },
            },
            observations=_observations(child),
            recovery_hints=[] if passed else ["regrasp"],
            outputs={"held": fresh if passed else {**held, "valid": False,
                                                     "invalidated_reason": "slip_detected"}},
        )  # fmt: skip


class BindTarget:
    """Bind a LEGO target to the piece actually carried, and list what is already built.

    ``mating.target_id`` becomes the piece's object id (the id observers report), its
    ``depends_on`` is mapped the same way, and ``build`` is the pre-built bricks plus the
    targets verified so far under the pieces that were placed there.
    """

    skill_id = "core.bind_target"
    version = "0.1.0"
    kind = "tool"
    requires = frozenset()
    readiness = "unit-tested"
    input_schema = _schema(
        ["target", "piece", "placed"],
        target={"type": "object"},
        piece={"type": "string"},
        placed={"type": "array", "items": {"type": "object"}},
        prebuilt={"type": "array", "items": {"type": "object"}},
    )

    def run(self, request: dict, ctx: SkillContext) -> dict:
        target, piece = request["target"], request["piece"]
        mating = target.get("mating")
        if not isinstance(mating, dict):
            return ctx.result(self, "rejected", phase="validate", failure_type="invalid_request")
        placed = list(request["placed"])
        by_target = {p["target_id"]: p["object_id"] for p in placed}
        bound = {
            **mating,
            "target_id": piece,
            "depends_on": [by_target.get(d, d) for d in mating.get("depends_on", [])],
        }
        build = [*request.get("prebuilt", []), *(p["mating"] for p in placed)]
        entry = {"target_id": target.get("target_id"), "object_id": piece, "mating": bound}
        return ctx.result(
            self,
            "succeeded",
            phase="validate",
            outputs={"mating": bound, "build": build, "placed_after": [*placed, entry]},
        )


class ReleaseAndRetreat:
    """Release a held object, retreat upward, and check the object was not carried away."""

    skill_id = "core.release_and_retreat"
    version = "0.1.0"
    kind = "skill"
    requires = frozenset(
        {"pose_feedback", "cartesian_motion", "gripper", "object_feedback", "stop"}
    )
    readiness = "unit-tested"
    input_schema = _schema(
        ["held", "tcp_pose", "release_width_m", "retreat_height_m", "max_drift_m"],
        held={"type": "object"},
        tcp_pose=_POSE,
        release_width_m=_NUM,
        retreat_height_m=_NUM,
        max_drift_m=_NUM,
    )

    def run(self, request: dict, ctx: SkillContext) -> dict:
        held = validate("HeldObject", request["held"])
        if (
            not 0.005 <= request["retreat_height_m"] <= 0.2
            or not 0 < request["max_drift_m"] <= 0.02
        ):
            return ctx.result(self, "rejected", phase="validate", failure_type="invalid_request")
        oid = held["object_id"]
        release = ctx.invoke(
            "set_gripper",
            {"width_m": request["release_width_m"], "purpose": "release", "target_id": oid},
        )
        if release["status"] != "succeeded":
            return _relay(self, ctx, release, "release")
        before = release["evidence"]["object_after"]
        retreat = ctx.invoke(
            "move_waypoints",
            {
                "waypoints": [translated(request["tcp_pose"], [0, 0, request["retreat_height_m"]])],
                "linear": False,
            },
        )
        if retreat["status"] != "succeeded":
            return _relay(self, ctx, retreat, "retreat")
        seen = ctx.observe([oid])
        if seen["status"] != "succeeded":
            return _relay(self, ctx, seen, "observe")
        after = _observations(seen)[0]
        drift = math.dist(before["position_m"], after["position_m"])
        passed = drift <= request["max_drift_m"]
        released = {**held, "valid": False, "invalidated_reason": "released"}
        return ctx.result(
            self,
            "succeeded" if passed else "failed",
            phase="retreat",
            failure_type=None if passed else "postcondition_failed",
            recovery_hints=[] if passed else ["reobserve", "reset_piece"],
            evidence={
                "postcondition": {
                    "name": "released_and_left_in_place",
                    "passed": passed,
                    "criterion": "holding false after release; object drift during retreat <= max_drift_m",
                    "measured": {"drift_m": drift, "max_drift_m": request["max_drift_m"]},
                }
            },
            observations=[after],
            outputs={"held": released, "placed": oid},
        )


class SelectCandidate:
    """Hand out the next feasible grasp candidate, at most ``max_tries`` per piece."""

    skill_id = "core.select_candidate"
    version = "0.1.0"
    kind = "tool"
    requires = frozenset()
    readiness = "unit-tested"
    input_schema = _schema(
        ["candidates", "tried"],
        candidates={"type": "array", "items": {"type": "object"}},
        tried={"type": "integer", "minimum": 0},
        max_tries={"type": "integer", "minimum": 1, "maximum": 20},
    )

    def run(self, request: dict, ctx: SkillContext) -> dict:
        usable = [c for c in request["candidates"] if c.get("checks", {}).get("feasible", True)]
        tried = request["tried"]
        limit = min(len(usable), request.get("max_tries", len(usable)))
        if tried >= limit:
            return ctx.result(
                self,
                "failed",
                phase="validate",
                failure_type="unreachable",
                evidence={"tried": tried, "usable": len(usable), "max_tries": limit},
                recovery_hints=["reobserve", "replan"],
            )
        return ctx.result(
            self,
            "succeeded",
            phase="validate",
            outputs={"candidate": usable[tried], "index": tried, "tried_after": tried + 1},
        )


SKILLS = [
    SelectNextTarget(),
    Observe(),
    TopdownGraspCandidate(),
    SelectCandidate(),
    ConfirmHold(),
    BindTarget(),
    Pick(),
    TransportHeld(),
    ReleaseAndRetreat(),
]
