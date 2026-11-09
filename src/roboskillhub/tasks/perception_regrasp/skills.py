"""Task skills (contract ``rsh-task/1``): four tools and one skill.

Tools compute only (``kind="tool"``, no requirements) and may run offline with
``ctx=None`` / ``offline_context()``. ``perception.reorient_or_regrasp`` is the one
skill: it observes through ``ctx.observe``, plans, and executes through
``ctx.invoke`` (``grasp``, ``move_waypoints``, ``set_gripper``, ``move_pose``,
``observe_objects``), and reports ``succeeded`` only when a fresh independent
observation puts the part at the goal.

Thresholds (observation age, verification tolerances, planning budget cap) are
this module's versioned ``CONFIG``; requests cannot enlarge them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _hub
from .contract import ObjectObservation, jsonable
from .domino import domino_state
from .frames import pose_to_T, rotation_angle
from .grasp import propose_grasps
from .held import HeldObject, check_held, estimate_from_observation
from .kinematics import YamIk
from .lego import snap_to_lattice
from .objects import ObjectSpec, domino, from_contract_frame, lego_brick, symmetry_rotations
from .regrasp import VERIFY_ROTATION_RAD, VERIFY_TRANSLATION_M, Workcell, plan_reorient_or_regrasp

VERSION = "0.1.0"
CONFIG = {
    "config_version": "perception_regrasp/1",
    "max_observation_age_s": 1.0,
    "max_path_evals_cap": 400,
    "verify_translation_m": VERIFY_TRANSLATION_M,
    "verify_rotation_rad": VERIFY_ROTATION_RAD,
}

_POSE = {"type": "object"}
_OBS = {"type": "object"}
_NUM = {"type": "number"}


def _schema(required: list[str], props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": props,
    }


def spec_for(cls: str | None, geometry: dict | None, studs: list[int] | None = None) -> ObjectSpec:
    """Object spec from an observation's class and box geometry."""
    size = None if not geometry else tuple(float(v) for v in geometry.get("size_m", ()))
    if cls == "domino":
        return domino(size) if size else domino()
    if cls in ("lego_brick", "lego_plate"):
        return lego_brick(tuple(studs or (2, 4)), plate=cls == "lego_plate")
    if size and len(size) == 3:
        return ObjectSpec(cls or "box", size, "box")
    raise ValueError(f"no geometry for class {cls!r}")


def _now(ctx) -> float:
    import time

    return ctx.now() if ctx is not None else time.time()


def _max_age(ctx) -> float:
    """The station's trusted ``max_state_age_s`` when the context carries limits
    (rsh-task/1.1), else this module's config."""
    limits = getattr(ctx, "limits", None) or {}
    return float(limits.get("max_state_age_s", CONFIG["max_observation_age_s"]))


def _privileged_ok(ctx) -> bool:
    return bool(ctx is not None and ctx.privileged_allowed)


def _result(skill, ctx, status: str, **kw) -> dict[str, Any]:
    from roboskillhub.runtime.task import make_result

    if ctx is not None:
        return ctx.result(skill, status, **kw)
    return make_result(skill, status, **kw)


def _from_pure(skill, ctx, r: dict[str, Any], observations=()) -> dict[str, Any]:
    """Wrap a pure function's result in the SkillResult envelope."""
    return _result(
        skill,
        ctx,
        r["status"],
        phase="validate" if r["status"] == "rejected" else "verify",
        failure_type=r["failure_type"],
        evidence={**r["evidence"], "reason": r["reason"], "verification": r["verification"]},
        recovery_hints=r["recovery_hints"],
        outputs=r["outputs"],
        observations=observations,
    )


def _observation(skill, ctx, request: dict, key: str = "observation"):
    d = request[key]
    geom = d.get("geometry") or request.get("geometry")
    spec = spec_for(d.get("class") or request.get("class"), geom, request.get("studs"))
    obs = ObjectObservation.from_v1(d, spec.size_m)
    ok, why, ftype = obs.check_usable(_now(ctx), _max_age(ctx), _privileged_ok(ctx))
    if not ok:
        return None, spec, _result(
            skill, ctx, "rejected", phase="validate", failure_type=ftype,
            evidence={"reason": why}, recovery_hints=["reobserve"], observations=[d],
        )  # fmt: skip
    return obs, spec, None


class ClassifyObjectState:
    skill_id = "perception.classify_object_state"
    version = VERSION
    kind = "tool"
    requires: frozenset[str] = frozenset()
    readiness = "unit-tested"
    input_schema = _schema(
        ["observation", "table_z"],
        {
            "observation": _OBS,
            "table_z": _NUM,
            "lattice": {"type": "object"},
            "studs": {"type": "array"},
        },
    )

    def run(self, request: dict, ctx=None) -> dict:
        obs, spec, rej = _observation(self, ctx, request)
        if rej:
            return rej
        assert obs.T_frame_object is not None
        out: dict[str, Any] = {"object_id": obs.object_id, "class": spec.object_class}
        if spec.object_class == "domino":
            st = domino_state(obs.T_frame_object, spec, table_z=float(request["table_z"]))
            out.update(state_label=st["state"], domino=st)
            status = "succeeded" if st["state"] != "uncertain" else "unverified"
        elif "lattice" in request:
            lat = request["lattice"]
            T_plate = pose_to_T(lat["T_W_baseplate"], obs.frame_id)
            snap = snap_to_lattice(
                np.linalg.inv(T_plate) @ obs.T_frame_object, spec, tuple(lat["plate_studs"])
            )
            snap.pop("T_plate_brick_snapped")
            out.update(
                state_label="seated" if snap["route"] == "on_grid" else "loose", lattice=snap
            )
            status = "succeeded" if snap["route"] == "on_grid" else "unverified"
        else:
            return _result(self, ctx, "rejected", phase="validate", failure_type="invalid_request",
                           evidence={"reason": "a LEGO part needs 'lattice'"})  # fmt: skip
        return _result(
            self, ctx, status, phase="verify",
            failure_type=None if status == "succeeded" else "perception.low_confidence",
            outputs=jsonable(out), observations=[request["observation"]],
            recovery_hints=[] if status == "succeeded" else ["reobserve"],
        )  # fmt: skip


class ProposeGrasps:
    skill_id = "perception.propose_grasps"
    version = VERSION
    kind = "tool"
    requires: frozenset[str] = frozenset()
    readiness = "unit-tested"
    input_schema = _schema(
        ["observation", "table_z"],
        {
            "observation": _OBS,
            "table_z": _NUM,
            "neighbours": {"type": "array", "items": _OBS},
            "tilts_deg": {"type": "array", "items": _NUM},
            "max_candidates": {"type": "integer", "minimum": 1, "maximum": 200},
            "studs": {"type": "array"},
        },
    )

    def run(self, request: dict, ctx=None) -> dict:
        obs, spec, rej = _observation(self, ctx, request)
        if rej:
            return rej
        nbrs = []
        for n in request.get("neighbours", []):
            o = ObjectObservation.from_v1(n)
            if o.has_pose:
                nbrs.append(
                    (o.object_id, o.T_frame_object, spec_for(n.get("class"), n.get("geometry")))
                )
        cands = propose_grasps(
            obs.object_id, obs.T_frame_object, spec, obs.frame_id, float(request["table_z"]),
            neighbours=nbrs, tilts_deg=tuple(request.get("tilts_deg", (0.0,))),
        )  # fmt: skip
        n_ok = sum(c.feasible for c in cands)
        keep = cands[: int(request.get("max_candidates", 20))]
        return _result(
            self, ctx, "succeeded" if n_ok else "failed", phase="verify",
            failure_type=None if n_ok else "perception.no_feasible_grasp",
            outputs={"candidates": [c.to_v1() for c in keep], "feasible_count": n_ok, "total": len(cands)},
            observations=[request["observation"]], recovery_hints=[] if n_ok else ["reobserve", "human"],
        )  # fmt: skip


class EstimateHeldPose:
    skill_id = "perception.estimate_held_pose"
    version = VERSION
    kind = "tool"
    requires: frozenset[str] = frozenset()
    readiness = "unit-tested"
    input_schema = _schema(
        ["held"],
        {
            "held": {"type": "object"},
            "observation": _OBS,
            "tcp_pose": _POSE,
            "tcp_timestamp_s": _NUM,
            "gripper_width_m": _NUM,
            "studs": {"type": "array"},
        },
    )

    def run(self, request: dict, ctx=None) -> dict:
        h = request["held"]
        spec = spec_for(h.get("class"), h.get("collision_geometry"), request.get("studs"))
        held = HeldObject.from_v1(h, spec)
        now = _now(ctx)
        held, ev = check_held(held, now, request.get("gripper_width_m"), _max_age(ctx))
        obs_list = []
        if held.valid and "observation" in request:
            if "tcp_pose" not in request or "tcp_timestamp_s" not in request:
                return _result(self, ctx, "rejected", phase="validate", failure_type="invalid_request",
                               evidence={"reason": "an observation needs tcp_pose and tcp_timestamp_s"})  # fmt: skip
            o = request["observation"]
            obs = ObjectObservation.from_v1(o, spec.size_m)
            T_tcp = pose_to_T(request["tcp_pose"], obs.frame_id)
            held, ev2 = estimate_from_observation(
                held, obs, T_tcp, float(request["tcp_timestamp_s"]), now,
                max_age_s=_max_age(ctx), allow_privileged=_privileged_ok(ctx),
            )  # fmt: skip
            ev.update(ev2)
            obs_list = [o]
        if held.valid:
            return _result(self, ctx, "succeeded", phase="verify", outputs={"held": held.to_v1()},
                           evidence=jsonable(ev), observations=obs_list)  # fmt: skip
        kind = held.invalid_kind
        ftype = {
            "slip": "held_object_slipped",
            "width_drop": "grasp_lost",
            "width_open": "grasp_lost",
            "stale": "stale_observation",
            "occluded": "object_not_observed",
            "identity_changed": "perception.identity_changed",
        }.get(kind or "", "perception.held_state_invalid")
        return _result(
            self, ctx, "failed", phase="verify", failure_type=ftype, outputs={"held": held.to_v1()},
            evidence=jsonable(ev), observations=obs_list,
            recovery_hints=["reobserve", "regrasp"] if kind in ("slip", "width_drop", "width_open") else ["reobserve"],
        )  # fmt: skip


def _workcell(request: dict, frame_id: str) -> Workcell:
    return Workcell(
        frame_id=frame_id,
        table_z=float(request["table_z"]),
        T_frame_base=pose_to_T(request["T_W_base"], frame_id),
        ik=YamIk(),
        staging_xy=[tuple(map(float, xy)) for xy in request.get("staging_xy", [])],
    )


_PLAN_PROPS = {
    "goal_pose": _POSE,
    "table_z": _NUM,
    "T_W_base": _POSE,
    "staging_xy": {"type": "array", "items": {"type": "array", "minItems": 2, "maxItems": 2}},
    "max_path_evals": {"type": "integer", "minimum": 1, "maximum": 400},
    "allow_staging": {"type": "boolean"},
    "studs": {"type": "array"},
}


class PlanReorientOrRegrasp:
    skill_id = "perception.plan_reorient_or_regrasp"
    version = VERSION
    kind = "tool"
    requires: frozenset[str] = frozenset()
    readiness = "unit-tested"
    input_schema = _schema(
        ["observation", "goal_pose", "table_z", "T_W_base"], {"observation": _OBS, **_PLAN_PROPS}
    )

    def run(self, request: dict, ctx=None) -> dict:
        obs, spec, rej = _observation(self, ctx, request)
        if rej:
            return rej
        if spec.object_class == "domino":
            obs.attributes["domino"] = domino_state(
                obs.T_frame_object, spec, table_z=float(request["table_z"])
            )
        r = plan_reorient_or_regrasp(
            obs, spec, from_contract_frame(pose_to_T(request["goal_pose"], obs.frame_id), spec.object_class), _workcell(request, obs.frame_id),
            now_s=_now(ctx), max_age_s=_max_age(ctx), allow_privileged=_privileged_ok(ctx),
            max_path_evals=min(int(request.get("max_path_evals", 120)), CONFIG["max_path_evals_cap"]),
            allow_staging=bool(request.get("allow_staging", True)),
        )  # fmt: skip
        return _from_pure(self, ctx, r, observations=[request["observation"]])


class ReorientOrRegrasp:
    """Observe, plan, execute, verify. ``succeeded`` needs a fresh independent 6D
    observation within tolerance of the goal (any symmetric equivalent)."""

    skill_id = "perception.reorient_or_regrasp"
    version = VERSION
    kind = "skill"
    requires = frozenset(
        {
            "pose_feedback",
            "cartesian_motion",
            "gripper",
            "stop",
            "object_feedback",
            "object_pose_feedback",
        }
    )
    readiness = "unit-tested"
    input_schema = _schema(
        ["object_id", "class", "geometry", "goal_pose", "table_z", "T_W_base"],
        {
            "object_id": {"type": "string"},
            "class": {"type": "string"},
            "geometry": {"type": "object"},
            **_PLAN_PROPS,
        },
    )

    def run(self, request: dict, ctx) -> dict:
        spec = spec_for(request["class"], request["geometry"], request.get("studs"))
        oid = request["object_id"]
        seen: list[dict] = []
        log: list[dict] = []

        def observe(phase: str):
            res = ctx.observe([oid], require_orientation=True)
            obs_d = (res.get("evidence") or {}).get("observations") or []
            seen.extend(obs_d)
            if res.get("status") != "succeeded" or not obs_d:
                return None, res
            o = ObjectObservation.from_v1({**obs_d[0], "class": spec.object_class}, spec.size_m)
            if spec.object_class == "domino" and o.has_pose:
                o.attributes["domino"] = domino_state(
                    o.T_frame_object, spec, table_z=float(request["table_z"])
                )
            return o, res

        def stop(status: str, phase: str, ftype: str, why: str, hints: list[str], **kw) -> dict:
            return ctx.result(
                self, status, phase=phase, failure_type=ftype,
                evidence={"reason": why, "steps": log, **kw.pop("evidence", {})},
                recovery_hints=hints, observations=seen, **kw,
            )  # fmt: skip

        obs, res = observe("observe")
        if obs is None:
            return stop("rejected", "observe", res.get("failure_type") or "object_not_observed",
                        "no usable 6D observation before planning", ["reobserve"])  # fmt: skip
        T_goal = from_contract_frame(
            pose_to_T(request["goal_pose"], obs.frame_id), spec.object_class
        )
        wc = _workcell(request, obs.frame_id)
        budget = min(int(request.get("max_path_evals", 120)), CONFIG["max_path_evals_cap"])
        plan = plan_reorient_or_regrasp(
            obs, spec, T_goal, wc, now_s=ctx.now(), max_age_s=_max_age(ctx),
            allow_privileged=ctx.privileged_allowed, max_path_evals=budget,
            allow_staging=bool(request.get("allow_staging", True)),
        )  # fmt: skip
        if plan["status"] != "succeeded":
            return stop("rejected", "validate", plan["failure_type"], plan["reason"], plan["recovery_hints"] or ["replan"],
                        evidence={"plan": plan["evidence"]})  # fmt: skip
        plan_type = plan["outputs"]["plan_type"]
        n_transfers = len(plan["outputs"]["transfers"])
        for t in range(n_transfers):
            steps = [s for s in plan["outputs"]["steps"] if s["transfer"] == t]
            if t > 0:
                # Replan the remaining transfer from what the camera saw after the put-down.
                plan = plan_reorient_or_regrasp(
                    obs, spec, T_goal, wc, now_s=ctx.now(), max_age_s=_max_age(ctx),
                    allow_privileged=ctx.privileged_allowed, max_path_evals=budget, allow_staging=False,
                )  # fmt: skip
                if plan["status"] != "succeeded":
                    return stop("failed", "validate", "perception.replan_failed", plan["reason"], ["replan", "human"],
                                evidence={"plan": plan["evidence"]})  # fmt: skip
                steps = [s for s in plan["outputs"]["steps"] if s["transfer"] == 0]
            for s in steps:
                if s["capability"] == "observe_objects":
                    continue
                r = ctx.invoke(s["capability"], s["request"])
                log.append({"transfer": t, "capability": s["capability"], "status": r.get("status"),
                            "failure_type": r.get("failure_type")})  # fmt: skip
                if r.get("status") != "succeeded":
                    status = "unverified" if r.get("status") == "unverified" else "failed"
                    return stop(status, s["phase"], r.get("failure_type") or "backend_error",
                                f"{s['capability']} returned {r.get('status')}", r.get("recovery_hints") or ["reobserve"])  # fmt: skip
            expect = steps[-1]["expect"]
            obs, res = observe("verify")
            if obs is None:
                return stop("unverified", "verify", res.get("failure_type") or "evidence_missing",
                            "no fresh 6D observation after the put-down", ["reobserve"])  # fmt: skip
            T_exp = from_contract_frame(pose_to_T(expect["pose"], obs.frame_id), spec.object_class)
            dt = float(np.linalg.norm(obs.T_frame_object[:3, 3] - T_exp[:3, 3]))
            dr = min(
                rotation_angle(obs.T_frame_object[:3, :3], T_exp[:3, :3] @ S)
                for S in symmetry_rotations(spec)
            )
            measured = {"translation_m": dt, "rotation_rad": dr, "state_label": obs.state_label(),
                        "expected_state_label": expect["state_label"], "transfer": t}  # fmt: skip
            passed = (
                dt <= CONFIG["verify_translation_m"]
                and dr <= CONFIG["verify_rotation_rad"]
                and (expect["state_label"] is None or obs.state_label() == expect["state_label"])
            )
            post = {
                "name": "object_at_goal_pose" if t == n_transfers - 1 else "object_at_staging_pose",
                "passed": passed,
                "criterion": "independent 6D observation within verify_translation_m / verify_rotation_rad "
                "of the target (any symmetric equivalent) and the expected state label",
                "measured": measured,
                "thresholds_ref": f"{CONFIG['config_version']} (skills.CONFIG)",
            }
            if not passed:
                return stop("failed", "verify", "perception.pose_out_of_tolerance",
                            "observed resting pose is off the planned target", ["reobserve", "regrasp"],
                            evidence={"postcondition": post})  # fmt: skip
        return ctx.result(
            self, "succeeded", phase="verify",
            evidence={"postcondition": post, "steps": log, "plan_type": plan_type,
                      "verification": "fresh observation via observe_objects"},
            outputs={"plan_type": plan_type, "final_observation": seen[-1]}, observations=seen,
        )  # fmt: skip


SKILLS = [
    ClassifyObjectState(),
    ProposeGrasps(),
    EstimateHeldPose(),
    PlanReorientOrRegrasp(),
    ReorientOrRegrasp(),
]


def reused_sources() -> dict[str, str]:
    """Hub files this module loads at run time (for provenance records)."""
    return dict(_hub.SOURCES)


__all__ = ["CONFIG", "SKILLS", "reused_sources", "spec_for"]
