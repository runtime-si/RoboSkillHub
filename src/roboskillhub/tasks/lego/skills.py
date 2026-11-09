"""Contract ``rsh-task/1`` TaskSkill adapters over the module's pure core.

Each skill reaches the station only through the ``SkillContext`` it is given
(``ctx.invoke`` / ``ctx.observe``) and returns the contract SkillResult envelope. The
runtime's ``run_task_skill`` applies the ``requires`` gate before ``run``.
"""

from __future__ import annotations

import time
from typing import Any

from . import contract_v1 as v1
from . import geometry as g
from .align import align_connection
from .contracts import Rejected, SkillResult
from .ports import CONTACT_CAPABILITIES
from .seat import guarded_seat
from .verify import verify_connection

VERSION = "0.1.0"
READINESS = "unit-tested"


def _ref(name: str) -> dict:
    return {"$ref": f"#/schemas/{name}"}


_OBJ = {"type": "object"}
_STR = {"type": "string"}
_NUM = {"type": "number"}


def _schema(required: list[str], props: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": props,
    }


def _level(ctx: Any) -> str:
    return getattr(ctx, "evidence_level", "unit_or_dryrun")


def _sleep(ctx: Any, s: float) -> None:
    (getattr(ctx, "sleep", None) or time.sleep)(s)


class _Observer:
    """BrickObserver over ``ctx.observe`` (observe_objects, orientation required)."""

    def __init__(self, ctx: Any, brick_height_m: float | None):
        self.ctx, self.h = ctx, brick_height_m
        self.used: list[dict] = []

    def describe(self) -> dict[str, Any]:
        return {
            "observer_id": "runtime:observe_objects",
            "evidence_level": _level(self.ctx),
            "uncertainty_m": None,
        }

    def observe_many(self, ids: list[str]) -> dict[str, dict]:
        r = self.ctx.observe(ids, require_orientation=True)
        got = {o.get("object_id"): o for o in (r.get("evidence") or {}).get("observations", [])}
        out = {}
        for i in ids:
            o = got.get(i)
            if o is None:
                out[i] = {
                    "obs_id": f"missing:{i}@{self.ctx.now()}",
                    "source": "observe_objects",
                    "kind": "measured",
                    "frame_id": "unknown",
                    "timestamp_s": self.ctx.now(),
                    "value": None,
                    "uncertainty_m": 0.0,
                    "unusable_reason": "brick_not_visible",
                }
                continue
            self.used.append(o)
            out[i] = v1.observation(o, brick_height_m=self.h)
        return out

    def observe(self, object_id: str, *, not_before_s: float, timeout_s: float) -> dict:
        return self.observe_many([object_id])[object_id]


class _Motion:
    """MotionPort over the runtime's ``move_pose`` primitive."""

    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.last: dict = {}

    def describe(self) -> dict[str, Any]:
        return {"backend_id": "runtime:move_pose", "evidence_level": _level(self.ctx)}

    def move_pose(self, pose: dict, *, speed_m_s: float, timeout_s: float) -> dict[str, Any]:
        self.last = self.ctx.invoke(
            "move_pose",
            {"target_pose": pose, "speed_m_s": speed_m_s, "timeout_s": max(0.1, timeout_s)},
        )
        ok = self.last.get("status") == "succeeded"
        return {
            "ok": ok,
            "error": self.last.get("failure_type"),
            "measured_tcp_pose": (self.last.get("evidence") or {}).get("measured_pose"),
        }

    def stop(self) -> dict[str, Any]:
        # The runtime stops the station itself when a primitive fails; no stop primitive.
        return {"confirmed": False, "note": "runtime-managed stop; not confirmable from a skill"}


class _Contact:
    """ContactBackend over a ``guarded_press`` primitive (requested from Agent 05; absent in
    runtime v1). ``describe`` comes from ``ctx.contact_description`` (also requested)."""

    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.result: dict = {}

    def describe(self) -> dict[str, Any]:
        d = getattr(self.ctx, "contact_description", None)
        if not isinstance(d, dict):
            return {
                "backend_id": "runtime",
                "capabilities": sorted(self.ctx.capabilities),
                "evidence_level": _level(self.ctx),
            }
        return {
            "evidence_level": _level(self.ctx),
            **d,
            "capabilities": sorted(set(d.get("capabilities", [])) & set(self.ctx.capabilities)),
        }

    def guarded_move(self, **kw: Any) -> dict[str, Any]:
        self.result = self.ctx.invoke("guarded_press", kw)
        ev = self.result.get("evidence") or {}
        return {
            "command_issued": self.result.get("status") != "rejected",
            "stop_reason": ev.get("stop_reason"),
            "stop_confirmed": ev.get("stop_confirmed"),
            "samples": ev.get("samples", []),
            "fault": ev.get("fault", ""),
        }

    def stop(self) -> dict[str, Any]:
        return {"confirmed": False}


def _type_parts(plan: dict) -> tuple[str, str]:
    """``system:NxM@version`` -> (system, "NxM")."""
    sysname, rest = str(plan["brick_type"]).split(":", 1)
    return sysname, rest.split("@", 1)[0]


def _rejected(skill: Any, ctx: Any, connection_id: str, exc: Rejected, phase="precheck") -> dict:
    res = SkillResult(skill.skill_id, skill.version, ctx.request_id, connection_id or "")
    res.phase = phase
    internal = res.finish("rejected", exc.reason_code, str(exc)).to_dict()
    return v1.envelope(
        skill,
        ctx,
        internal,
        observations=[],
        postcond=v1.postcondition(skill.postcondition_name, False, skill.criterion, {}, ""),
        outputs={},
    )


class AlignConnection:
    skill_id = "lego.align_connection"
    version = VERSION
    kind = "skill"
    readiness = READINESS
    requires = frozenset(
        {"pose_feedback", "cartesian_motion", "stop", "object_feedback", "object_pose_feedback"}
    )
    postcondition_name = "held_brick_aligned"
    criterion = (
        "fresh independent observation of the held brick, after the last move, "
        "within xy_tolerance_m and yaw_tolerance_deg of the hover pose"
    )
    input_schema = _schema(
        ["connection_id", "target", "held", "limits"],
        {
            "connection_id": _STR,
            "target": _ref("LegoMatingTarget"),
            "held": _ref("HeldObject"),
            "build": {"type": "array", "items": _ref("LegoMatingTarget")},
            "limits": _OBJ,
            "close_axis": {"enum": [0, 1]},
        },
    )

    def run(self, request: dict, ctx: Any) -> dict:
        cid = request["connection_id"]
        try:
            m = v1.mating_target(request["target"])
            build = [v1.placed(b) for b in request.get("build", [])]
            missing = set(m["depends_on"]) - {b["brick_id"] for b in build}
            if missing:
                raise Rejected("unknown_reference", f"depends_on {sorted(missing)} not placed")
            held = v1.held(request["held"], close_axis=request.get("close_axis", 0))
        except Rejected as exc:
            return _rejected(self, ctx, cid, exc)
        bt = g.brick_type(m["brick"]["system"], m["brick"]["studs"])
        observer = _Observer(ctx, bt.system.brick_height_m)
        first = observer.observe(held["object_id"], not_before_s=0, timeout_s=5)
        fid = m["plate"]["pose"]["frame_id"]
        if first.get("unusable_reason") or first["value"] is None or first["frame_id"] != fid:
            why = first.get("unusable_reason") or (
                "brick_not_visible" if first["value"] is None else "frame_mismatch"
            )
            return _rejected(
                self,
                ctx,
                cid,
                Rejected(why, "held brick not observable before approach; nothing moved"),
                "observe",
            )
        brick = first["value"] if "pose" not in first["value"] else first["value"]["pose"]
        tcp = g.to_pose(g.tcp_target(g.matrix(brick), g.matrix(held["T_TCP_object"])), fid)
        limits = dict(request["limits"])
        if m["requires_support"]:
            limits.setdefault("support_required_fraction", 0.25)
        internal_req = {
            "request_id": ctx.request_id,
            "connection_id": cid,
            "held_brick_id": held["object_id"],
            "reference_id": m["plate"]["plate_id"],
            "brick": m["brick"],
            "plate": m["plate"],
            "target_cell": m["target_cell"],
            "build": build,
            "held": held,
            "tcp_observation": {
                **first,
                "obs_id": first["obs_id"] + "+T_TCP_object",
                "source": first["source"] + "+T_TCP_object",
                "value": tcp,
            },
            "limits": limits,
        }
        out = align_connection(internal_req, _Motion(ctx), observer, clock=ctx.now)
        ev = out["evidence"]
        post = v1.postcondition(
            self.postcondition_name,
            out["status"] == "succeeded",
            self.criterion,
            ev.get("observed_error", {}),
            str(ev.get("limits_source", "")),
        )
        outputs = {}
        if out["status"] == "succeeded":
            outputs = {
                "alignment": {
                    "invocation_id": out["invocation_id"],
                    "skill_id": self.skill_id,
                    "status": "succeeded",
                    "connection_id": cid,
                    "held_object_id": held["object_id"],
                    "evidence": {
                        "plan": ev["plan"],
                        "commanded_tcp_pose": ev["commanded_tcp_pose"],
                    },
                }
            }
        return v1.envelope(
            self, ctx, out, observations=observer.used, postcond=post, outputs=outputs
        )


class GuardedSeat:
    skill_id = "lego.guarded_seat"
    version = VERSION
    kind = "skill"
    readiness = READINESS
    requires = CONTACT_CAPABILITIES
    postcondition_name = "bounded_press_complete"
    criterion = (
        "stop confirmed, peak <= 1.25 x force_limit_N, contact at the stud tops; the pressed brick "
        "observed after the press within seat_depth_tolerance_m (default 0.35 stud height) of its "
        "seat and within verify's lateral/yaw/tilt bands; plus a force_threshold stop when the "
        "station declares an insertion-observable force signal; without a usable observation a "
        "force-observable station falls back to TCP travel. Connection NOT verified here"
    )
    input_schema = _schema(
        ["connection_id", "alignment", "press_limits"],
        {
            "connection_id": _STR,
            "alignment": _OBJ,
            "press_limits": _OBJ,
            "axis": {"type": "object"},
        },
    )

    def run(self, request: dict, ctx: Any) -> dict:
        al = request["alignment"]
        plan = (al.get("evidence") or {}).get("plan") or {}
        lim = dict(request["press_limits"])
        internal_req = {
            "request_id": ctx.request_id,
            "connection_id": request["connection_id"],
            "system": str(plan.get("brick_type", "")).split(":")[0],
            "alignment": al,
            "axis": request.get("axis") or plan.get("engagement_axis"),
            **lim,
        }
        contact = _Contact(ctx)
        observer = None
        observe = None
        if "object_feedback" in ctx.capabilities and al.get("held_object_id"):
            try:
                height = g.brick_type(*_type_parts(plan)).system.brick_height_m
            except (g.GeometryError, IndexError, ValueError):
                height = None
            observer = _Observer(ctx, height)

            def observe() -> dict | None:
                # Independent post-press observation of the pressed brick (judged in guarded_seat).
                return observer.observe_many([al["held_object_id"]])[al["held_object_id"]]

        out = guarded_seat(internal_req, contact, clock=ctx.now, observe=observe)
        observations: list[dict] = observer.used if observer is not None else []
        ev = out["evidence"]
        press = {}
        if "final_travel_m" in ev and plan.get("engagement_axis"):
            axis = plan["engagement_axis"]["vector"]
            start = al["evidence"]["commanded_tcp_pose"]
            d = ev["final_travel_m"]
            press = {
                "operation_complete": ev["operation_complete"],
                "final_travel_m": d,
                "engagement_axis": plan["engagement_axis"],
                "press_end_tcp_pose": {
                    **start,
                    "position_m": [
                        p + a * d for p, a in zip(start["position_m"], axis, strict=True)
                    ],
                },
            }
        post = v1.postcondition(
            self.postcondition_name,
            out["status"] == "succeeded",
            self.criterion,
            {
                k: ev.get(k)
                for k in ("final_travel_m", "seat_travel_m", "peak_force_N", "shortfall_m")
            }
            | {"stop": out.get("stop"), "seat_check": ev.get("seat_check")},
            str(lim.get("limits_source", "")),
        )
        outputs = {
            "press": press,
            "executor_id": "runtime:guarded_press",
            "operation_complete": ev.get("operation_complete", False),
            "connection_verified": False,
        }
        return v1.envelope(
            self, ctx, out, observations=observations, postcond=post, outputs=outputs
        )


class ReleaseAndRetreat:
    skill_id = "lego.release_and_retreat"
    version = VERSION
    kind = "skill"
    readiness = READINESS
    requires = frozenset(
        {
            "gripper",
            "gripper_feedback",
            "pose_feedback",
            "cartesian_motion",
            "stop",
            "object_feedback",
            "object_pose_feedback",
        }
    )
    postcondition_name = "released_and_retreated"
    criterion = (
        "set_gripper(release) measured holding=false, then the measured TCP rose at "
        "least half of retreat_height_m along the engagement axis"
    )
    input_schema = _schema(
        ["connection_id", "object_id", "press", "open_width_m", "retreat_height_m"],
        {
            "connection_id": _STR,
            "object_id": _STR,
            "press": _OBJ,
            "open_width_m": _NUM,
            "retreat_height_m": _NUM,
            "speed_m_s": _NUM,
        },
    )

    def run(self, request: dict, ctx: Any) -> dict:
        cid = request["connection_id"]
        press = request["press"]
        h = request["retreat_height_m"]
        if not (0.002 <= h <= 0.05):
            return _rejected(
                self, ctx, cid, Rejected("invalid_input", "retreat_height_m outside [2, 50] mm")
            )
        if not press.get("press_end_tcp_pose") or not press.get("engagement_axis"):
            return _rejected(
                self,
                ctx,
                cid,
                Rejected(
                    "not_aligned",
                    "no measured press end: holding/contact state unknown, no blind release",
                ),
            )
        res = SkillResult(self.skill_id, self.version, ctx.request_id, cid)
        observer = _Observer(ctx, None)
        oid = request["object_id"]
        observer.observe_many([oid])
        pre_contract = observer.used[-1] if observer.used else None
        res.phase = "release"
        res.motion_issued = True
        r = ctx.invoke(
            "set_gripper",
            {"width_m": request["open_width_m"], "purpose": "release", "target_id": oid},
        )
        released_at = ctx.now()
        res.trace.append(
            {
                "event": "set_gripper",
                "status": r.get("status"),
                "failure_type": r.get("failure_type"),
            }
        )
        outputs: dict[str, Any] = {"pre_release_observation": pre_contract, "release": None}
        if r.get("status") != "succeeded":
            status = "failed" if r.get("status") in ("failed", "rejected") else "unverified"
            reason = (
                "release_failed"
                if r.get("failure_type") == "release_failed"
                else ("release_unconfirmed" if status == "unverified" else "motion_failed")
            )
            if r.get("status") == "rejected":
                res.motion_issued = False
                status = "rejected"
            internal = res.finish(
                status, reason, f"set_gripper: {r.get('failure_type')}", next_allowed=["human"]
            ).to_dict()
            return v1.envelope(
                self,
                ctx,
                internal,
                observations=observer.used,
                postcond=v1.postcondition(self.postcondition_name, False, self.criterion, {}, ""),
                outputs=outputs,
            )
        res.phase = "retreat"
        axis = press["engagement_axis"]["vector"]
        start = press["press_end_tcp_pose"]
        target = {
            **start,
            "position_m": [p - a * h for p, a in zip(start["position_m"], axis, strict=True)],
        }
        mv = ctx.invoke(
            "move_pose", {"target_pose": target, "speed_m_s": request.get("speed_m_s", 0.01)}
        )
        measured = (mv.get("evidence") or {}).get("measured_pose")
        rise = None
        if measured:
            rise = -sum(
                (m - s) * a
                for m, s, a in zip(measured["position_m"], start["position_m"], axis, strict=True)
            )
        retreated = mv.get("status") == "succeeded" and rise is not None and rise >= 0.5 * h
        res.evidence.update(rise_m=rise, retreat_target=target, released_at_s=released_at)
        observer.observe_many([oid])
        outputs["release"] = {
            "gripper_open": True,
            "retreated": retreated,
            "released_at_s": released_at,
            "source": "gripper_feedback",
            "timestamp_s": released_at,
            "set_gripper_invocation": r.get("invocation_id"),
        }
        if not retreated:
            internal = res.finish(
                "failed",
                "motion_failed",
                f"retreat: {mv.get('failure_type')}, rise {rise}",
                next_allowed=["human"],
            ).to_dict()
        else:
            internal = res.finish(
                "succeeded", "released", "", next_allowed=["lego.verify_connection"]
            ).to_dict()
        post = v1.postcondition(
            self.postcondition_name,
            retreated,
            self.criterion,
            {"rise_m": rise, "holding": False},
            "request.retreat_height_m",
        )
        return v1.envelope(
            self, ctx, internal, observations=observer.used, postcond=post, outputs=outputs
        )


class VerifyConnection:
    skill_id = "lego.verify_connection"
    version = VERSION
    kind = "skill"
    readiness = READINESS
    requires = frozenset({"object_feedback", "object_pose_feedback"})
    postcondition_name = "geometric_seating"
    criterion = (
        "after release and retreat, independent observations over window_s show the "
        "brick on its cell within lateral/yaw/tilt/height bands (stud geometry + 2 "
        "sigma), stable, and the listed neighbours undisturbed; no retention test"
    )
    input_schema = _schema(
        ["connection_id", "target", "executor_id", "release"],
        {
            "connection_id": _STR,
            "target": _ref("LegoMatingTarget"),
            "build": {"type": "array", "items": _ref("LegoMatingTarget")},
            "executor_id": _STR,
            "release": _OBJ,
            "pre_release_observation": {"type": ["object", "null"]},
            "must_observe_neighbours": {"type": "array", "items": _STR},
            "window": _OBJ,
            "observations": {"type": "array", "items": _ref("Observation")},
        },
    )

    def run(self, request: dict, ctx: Any) -> dict:
        cid = request["connection_id"]
        try:
            m = v1.mating_target(request["target"])
            build = [v1.placed(b) for b in request.get("build", [])]
        except Rejected as exc:
            return _rejected(self, ctx, cid, exc)
        bid = request["target"]["target_id"]
        bt = g.brick_type(m["brick"]["system"], m["brick"]["studs"])
        observer = _Observer(ctx, bt.system.brick_height_m)
        win = {
            "window_s": 0.5,
            "min_samples": 2,
            "max_obs_age_s": 2.0,
            **(request.get("window") or {}),
        }
        rel = request["release"] or {}
        ids = [bid] + [b["brick_id"] for b in build]
        samples: list[dict] = []
        neighbours: list[dict] = []
        if request.get("observations") is not None:
            for o in request["observations"]:
                observer.used.append(o)
                conv = v1.observation(o, brick_height_m=bt.system.brick_height_m)
                (samples if o.get("object_id") == bid else neighbours).append(
                    conv if o.get("object_id") == bid else _neighbour(o, conv)
                )
        else:
            n = max(2, int(win["min_samples"]))
            for i in range(n):
                got = observer.observe_many(ids)
                samples.append(got[bid])
                if i == n - 1:
                    neighbours = [
                        _neighbour({"object_id": k}, v) for k, v in got.items() if k != bid
                    ]
                else:
                    _sleep(ctx, float(win["window_s"]) / (n - 1))
        pre = request.get("pre_release_observation")
        internal_req = {
            "request_id": ctx.request_id,
            "connection_id": cid,
            "brick_id": bid,
            "brick": m["brick"],
            "plate": m["plate"],
            "target_cell": m["target_cell"],
            "build": build,
            "executor_id": request["executor_id"],
            "released_at_s": float(rel.get("released_at_s", 0.0)),
            "release": {
                "obs_id": f"release:{rel.get('set_gripper_invocation')}",
                "source": str(rel.get("source", "unknown")),
                "kind": "measured",
                "frame_id": "tcp",
                "timestamp_s": float(rel.get("timestamp_s", 0.0)),
                "value": {
                    "gripper_open": bool(rel.get("gripper_open")),
                    "retreated": bool(rel.get("retreated")),
                },
                "uncertainty_m": 0.0,
            },
            "observations": samples,
            "neighbours": neighbours,
            "must_observe_neighbours": request.get("must_observe_neighbours", []),
            "window": win,
            "observer_evidence_level": _level(ctx),
        }
        if pre:
            internal_req["pre_release_observation"] = v1.observation(
                pre, brick_height_m=bt.system.brick_height_m
            )
        out = verify_connection(internal_req, clock=ctx.now)
        ev = out["evidence"]
        last = (ev.get("deltas") or [{}])[-1]
        post = v1.postcondition(
            self.postcondition_name,
            out["status"] == "succeeded",
            self.criterion,
            {k: last.get(k) for k in ("dx_m", "dy_m", "dz_m", "yaw_deg", "tilt_deg")},
            str(ev.get("thresholds_source", "")),
        )
        return v1.envelope(
            self,
            ctx,
            out,
            observations=observer.used,
            postcond=post,
            outputs={"verdict": ev.get("verdict"), "claim": ev.get("claim"), "connection_id": cid},
        )


def _neighbour(contract_obs: dict, conv: dict) -> dict:
    """Internal neighbour observation: ``value = {brick_id, pose}``."""
    pose = (
        conv["value"]["pose"]
        if isinstance(conv["value"], dict) and "pose" in conv["value"]
        else conv["value"]
    )
    return {**conv, "value": {"brick_id": contract_obs.get("object_id"), "pose": pose}}


SKILLS = [AlignConnection(), GuardedSeat(), ReleaseAndRetreat(), VerifyConnection()]
REGISTRY = {s.skill_id: s for s in SKILLS}
