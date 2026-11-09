"""place_upright: carry a held tile to its target pose, set it down, let go, back off,
then hand the outcome to the independent observer (verify_stable_placement).

The motion is the hub's placing-an-upright-domino-beside-neighbours place half
(transit over the row, vertical descent to ``release_gap`` over the table, open only to
the tile's width plus ``release_clearance`` a side, retreat straight up), composed from
rsh-task/1 primitives through the task context (never a backend):

    validate + capability gate + fresh holding / scene evidence      no command
    plan preplace / release / retreat TCP poses and clearances      no command
    carry to preplace          move_waypoints(linear=false, held)
    bounded descent            move_waypoints(dense straight leg, held)    [linear=true if
                                                                             descent_path=linear]
    support assessment         observe_objects (target, optional support observation)
    controlled release         set_gripper(purpose=release, width)
    retreat                    move_waypoints(dense straight leg)
    settle-window observation  observe_objects x N
    verify_stable_placement    pure, on the collected observations

One deadline (``timeout_s``) covers every call and the settle window; ``ctx.invoke`` runs
inside the graph's task lease, which stops motion on failure. This module never opens the
gripper or presses again on its own after a failure.
"""

from __future__ import annotations

import math
import time
from typing import Any, Protocol

import numpy as np

from . import geometry as geo
from . import upstream
from .classify import classify
from .contract import (
    PHASE,
    RECOVERY_HINTS,
    RequestRejected,
    check_geometry,
    check_pose,
    check_support,
    envelope,
    finite,
    place_requires,
    set_failure,
    thresholds_for,
    vec,
)
from .recovery import plan_recovery
from .verify import VISIBLE, verify_stable_placement

SKILL_ID = "domino.place_upright"


class TaskContext(Protocol):
    """rsh-task/1 SkillContext (Agent 05). ``state()`` and ``sleep()`` are requested
    additions (integration-request.md); without them the holding pre-check relies on the
    held object's observation only and the settle loop uses time.sleep."""

    task_id: str
    request_id: str
    capabilities: frozenset[str]
    privileged_allowed: bool

    def invoke(self, capability: str, request: dict) -> dict: ...
    def observe(
        self, object_ids: list[str], *, require_orientation: bool = False
    ) -> list[dict]: ...
    def cancelled(self) -> bool: ...
    def now(self) -> float: ...


class PhaseFailed(RuntimeError):
    def __init__(
        self,
        code: str,
        phase: str,
        detail: str,
        status: str = "failed",
        failure_type: str | None = None,
    ):
        super().__init__(detail)
        self.code, self.phase, self.status, self.failure_type = code, phase, status, failure_type


# ----------------------------------------------------------------------------- planning (pure)


def plan_place(request: dict) -> dict:
    """Validate a request and compute every pose and clearance before anything moves.
    Raises RequestRejected with a module fault code."""
    pd, pl = upstream.place(), upstream.layout()
    frame = request.get("frame_id")
    if not isinstance(frame, str) or not frame:
        raise RequestRejected("invalid_request", "frame_id is required")
    target_id = request.get("target_id")
    if not isinstance(target_id, str) or not target_id:
        raise RequestRejected("invalid_request", "target_id is required")
    g = check_geometry(request.get("geometry"))
    size = {"height": g["height_m"], "width": g["width_m"], "thickness": g["thickness_m"]}
    goal = check_pose(request.get("target_object_pose"), "target_object_pose", frame)
    support = check_support(request.get("support"), frame)
    th, th_ref = thresholds_for(request)
    mode = request.get("mode", "geometric")
    descent_path = request.get("descent_path", "dense_waypoints")
    requires = place_requires(mode, descent_path)
    now = finite(request.get("now_s"), "now_s", 0, 1e12)
    normal = np.asarray(support["normal"])

    # Target: upright on the support, its bottom face on the plane, inside the region.
    tilt = geo.tilt_rad(goal, normal, up_symmetric=g["up_symmetric"])
    if tilt > th["tilt_tolerance_rad"]:
        raise RequestRejected(
            "infeasible_geometry",
            f"target pose is tilted {math.degrees(tilt):.1f} deg off the support normal",
        )
    low = geo.lowest_point_height(goal, g, support["point_m"], normal)
    if abs(low) > th["support_height_tolerance_m"]:
        raise RequestRejected(
            "infeasible_geometry",
            f"target pose puts the tile's bottom {low * 1e3:.1f} mm off the support plane (origin: {g['origin']})",
        )
    if "region_min_m" in support:
        xy = goal["position_m"][:2]
        if not all(
            lo <= v <= hi
            for lo, v, hi in zip(support["region_min_m"], xy, support["region_max_m"], strict=True)
        ):
            raise RequestRejected(
                "infeasible_geometry", "target is outside the support's usable region"
            )

    # Held object (rsh-task/1 HeldObject, Agent 01's postcondition).
    held = request.get("held")
    if not isinstance(held, dict):
        raise RequestRejected("holding_not_confirmed", "no held-object evidence")
    if held.get("object_id") != target_id:
        raise RequestRejected(
            "holding_not_confirmed", f"held object {held.get('object_id')!r} is not {target_id!r}"
        )
    if held.get("valid") is not True:
        raise RequestRejected(
            "holding_not_confirmed", f"held object invalid ({held.get('invalidated_reason')})"
        )
    if not held.get("gripper_id") or not held.get("estimate"):
        raise RequestRejected("holding_not_confirmed", "held object needs gripper_id and estimate")
    t_tcp_obj = check_pose(held.get("T_TCP_object"), "held.T_TCP_object", "tcp")
    age = now - finite(held.get("timestamp_s"), "held.timestamp_s", 0, 1e12)
    if age > th["max_observation_age_s"] or age < -0.1:
        raise RequestRejected("observation_stale", f"held-object evidence is {age:.2f} s old")
    std = (held.get("uncertainty") or {}).get("position_m")
    if std is None:
        raise RequestRejected("holding_not_confirmed", "held.uncertainty.position_m is unknown")
    if finite(std, "held.uncertainty.position_m", 0, 1) > th["max_holding_std_m"]:
        raise RequestRejected(
            "holding_not_confirmed",
            f"T_TCP_object uncertainty {std * 1e3:.1f} mm is over {th['max_holding_std_m'] * 1e3:.1f} mm",
        )

    # Neighbours: fresh rsh-task/1 Observations with orientation, no overlap, hand clear.
    neighbors = []
    for i, nb in enumerate(request.get("neighbors") or []):
        if (
            not isinstance(nb, dict)
            or not nb.get("object_id")
            or not nb.get("source")
            or nb.get("visibility") not in VISIBLE
        ):
            raise RequestRejected(
                "invalid_request", f"neighbors[{i}] must be a visible Observation"
            )
        if nb.get("quaternion_xyzw") is None:
            raise RequestRejected(
                "invalid_request",
                f"neighbour {nb['object_id']} has no orientation: its footprint is unknown",
            )
        p = check_pose(
            {
                "frame_id": nb.get("frame_id"),
                "position_m": nb.get("position_m"),
                "quaternion_xyzw": nb.get("quaternion_xyzw"),
            },
            f"neighbors[{i}]",
            frame,
        )
        a = now - finite(nb.get("timestamp_s"), f"neighbors[{i}].timestamp_s", 0, 1e12)
        if a > th["max_observation_age_s"] or a < -0.1:
            raise RequestRejected(
                "observation_stale", f"neighbour {nb['object_id']} was observed {a:.2f} s ago"
            )
        neighbors.append({"id": nb["object_id"], "pose": p, "observation": nb})
    goal_xy, goal_facing = geo.xy_facing(goal)
    foot = pl.footprint(goal_xy, goal_facing, size)
    for nb in neighbors:
        if (
            pl.poly_distance(foot, pl.footprint(*geo.xy_facing(nb["pose"]), size))
            < pl.OVERLAP_CLEARANCE
        ):
            raise RequestRejected("infeasible_geometry", f"target overlaps neighbour {nb['id']}")

    hand = request.get("hand") or {}
    tip_reach = finite(hand.get("tip_reach_m", pd.TIP_REACH), "hand.tip_reach_m", 0, 0.5)
    max_open = finite(hand.get("max_open_width_m", 1.0), "hand.max_open_width_m", 0, 1)
    rel = request.get("release") or {}
    width = finite(
        rel.get("width_m", g["width_m"] + 2 * th["release_clearance_m"]), "release.width_m", 0, 1
    )
    if width <= g["width_m"]:
        raise RequestRejected(
            "infeasible_geometry",
            f"release width {width * 1e3:.1f} mm does not clear the {g['width_m'] * 1e3:.1f} mm tile",
        )
    if width > max_open:
        raise RequestRejected(
            "infeasible_geometry",
            f"release width {width * 1e3:.1f} mm is over the hand's {max_open * 1e3:.1f} mm",
        )
    clash = pd.finger_conflicts(
        goal_xy, goal_facing, [geo.xy_facing(nb["pose"]) for nb in neighbors], size, width
    )
    if clash:
        raise RequestRejected(
            "neighbor_clearance",
            f"open fingers over the target would reach neighbour(s) {[neighbors[c]['id'] for c in clash]} "
            "(placing-an-upright-domino-beside-neighbours' finger check)",
        )

    # TCP poses: T_station_tcp = T_station_object * inv(T_TCP_object).
    gap = th["release_gap_m"]
    tcp_goal = geo.tcp_for_object(goal, t_tcp_obj)
    release_pose = geo.translated(tcp_goal, normal * gap)
    approach = request.get("approach") or {}
    a_dir = geo.unit(approach.get("direction", list(-normal)))
    a_dist = finite(approach.get("distance_m", 0.05), "approach.distance_m", 0.005, 0.5)
    retreat = request.get("retreat") or {}
    r_dir = geo.unit(retreat.get("direction", list(normal)))
    r_dist = finite(retreat.get("distance_m", 0.05), "retreat.distance_m", 0.005, 0.5)
    if a_dir @ normal > -0.5:
        raise RequestRejected(
            "invalid_request", "approach.direction must point down onto the support (within 60 deg)"
        )
    if r_dir @ normal < 0.5:
        raise RequestRejected(
            "invalid_request", "retreat.direction must point away from the support (within 60 deg)"
        )
    preplace = geo.translated(release_pose, -a_dir * a_dist)
    retreat_pose = geo.translated(release_pose, r_dir * r_dist)
    hang = geo.height_over_plane(tcp_goal["position_m"], support["point_m"], normal) - (
        geo.height_over_plane(goal["position_m"], support["point_m"], normal) + geo.bottom_z(g)
    )

    def sweep(start: dict, end: dict, carrying: bool) -> list[str]:
        """Neighbours the open hand's envelope (and the held tile) passes through below
        their tops plus clearance, along a straight leg."""
        hits = []
        for s in np.linspace(0.0, 1.0, 21):
            p = (1 - s) * np.asarray(start["position_m"]) + s * np.asarray(end["position_m"])
            tcp_h = geo.height_over_plane(p, support["point_m"], normal)
            lowest = tcp_h - (max(tip_reach, hang) if carrying else tip_reach)
            if lowest > g["height_m"] + pl.NEIGHBOUR_CLEARANCE:
                continue
            env = pl.hand_envelope(p[:2], goal_facing, size, width)
            for nb in neighbors:
                if (
                    pl.poly_distance(env, pl.footprint(*geo.xy_facing(nb["pose"]), size))
                    < pl.NEIGHBOUR_CLEARANCE
                ):
                    hits.append(nb["id"])
        return sorted(set(hits))

    if r_dir @ normal < 1 - 1e-6 and (hit := sweep(release_pose, retreat_pose, False)):
        raise RequestRejected(
            "retreat_blocked", f"the retreat sweeps the open hand through neighbour(s) {hit}"
        )
    if a_dir @ normal > -1 + 1e-6 and (hit := sweep(preplace, release_pose, True)):
        raise RequestRejected(
            "neighbor_clearance", f"the approach sweeps the hand through neighbour(s) {hit}"
        )

    ws = request.get("workspace")
    if not isinstance(ws, dict):
        raise RequestRejected("invalid_request", "workspace {min_m, max_m} is required")
    lo, hi = vec(ws.get("min_m"), "workspace.min_m", 3), vec(ws.get("max_m"), "workspace.max_m", 3)
    for name, p in (("preplace", preplace), ("release", release_pose), ("retreat", retreat_pose)):
        if not all(a <= v <= b for a, v, b in zip(lo, p["position_m"], hi, strict=True)):
            raise RequestRejected(
                "infeasible_geometry", f"{name} TCP pose is outside the workspace"
            )

    speed = finite(request.get("speed_m_s", 0.03), "speed_m_s", 1e-4, 0.5)
    d_speed = finite(
        request.get("descent_speed_m_s", min(speed, 0.01)), "descent_speed_m_s", 1e-4, speed
    )
    timeout = finite(request.get("timeout_s"), "timeout_s", 0.1, 600)
    need = a_dist / d_speed + r_dist / d_speed + th["settle_window_s"]
    if need > timeout:
        raise RequestRejected(
            "invalid_request",
            f"timeout_s {timeout:.1f} s is under the {need:.1f} s the descent, retreat and settle window need",
        )
    step = th["waypoint_step_m"]

    def leg(a: dict, b: dict) -> list[dict]:
        n = max(
            1,
            math.ceil(float(np.linalg.norm(np.subtract(b["position_m"], a["position_m"]))) / step),
        )
        if n > 32:
            raise RequestRejected(
                "invalid_request",
                "a straight leg needs more than 32 waypoints; raise waypoint_step_m in config",
            )
        pa, pb = np.asarray(a["position_m"]), np.asarray(b["position_m"])
        return [
            geo.pose(frame, pa + (pb - pa) * k / n, b["quaternion_xyzw"]) for k in range(1, n + 1)
        ]

    if mode == "contact_feedback":
        raise RequestRejected(
            "capability_missing",
            "contact_feedback mode needs a guarded-descent primitive that rsh-task/1 does not define yet",
        )
    return {
        "frame_id": frame,
        "target_id": target_id,
        "geometry": g,
        "goal": goal,
        "support": support,
        "thresholds": th,
        "thresholds_ref": th_ref,
        "mode": mode,
        "descent_path": descent_path,
        "requires": requires,
        "T_TCP_object": t_tcp_obj,
        "held": held,
        "neighbors": neighbors,
        "release_width_m": width,
        "tip_reach_m": tip_reach,
        "poses": {
            "preplace": preplace,
            "release": release_pose,
            "retreat": retreat_pose,
            "tcp_goal": tcp_goal,
        },
        "descent_waypoints": leg(preplace, release_pose),
        "retreat_waypoints": leg(release_pose, retreat_pose),
        "speed_m_s": speed,
        "descent_speed_m_s": d_speed,
        "timeout_s": timeout,
    }


# ----------------------------------------------------------------------------- execution


def place_upright(request: dict, ctx: TaskContext) -> dict:
    t0 = ctx.now()
    request = {**request, "now_s": request.get("now_s", t0)}
    result = envelope(SKILL_ID, request, kind="skill", ctx=ctx)
    result["scene_version"] = request.get("scene_version")
    trace = result["trace"]
    issued = [False]
    seen: list[dict] = []
    gripper_states: list[dict] = []
    tcp_states: list[dict] = []
    events: dict[str, float] = {"action_start_s": t0}
    sleep = getattr(ctx, "sleep", time.sleep)
    plan: dict | None = None

    def log(phase: str, **kw: Any) -> None:
        trace.append({"phase": phase, "t_s": ctx.now(), **kw})

    try:
        plan = plan_place(request)
        missing = sorted(plan["requires"] - set(ctx.capabilities))
        if missing:
            raise RequestRejected("capability_missing", f"backend lacks {missing}")
    except upstream.UpstreamUnavailable as exc:
        set_failure(result, "dependency_missing", "validate", str(exc), "rejected")
        return result
    except RequestRejected as exc:
        set_failure(result, exc.code, "validate", str(exc), "rejected")
        log("validate", ok=False, code=exc.code)
        result["recovery"] = plan_recovery(result, attempts_used=0, max_attempts=0)
        result["recovery_hints"] = RECOVERY_HINTS[result["recovery"]["action"]]
        return result
    log("validate", ok=True)
    result["requires"] = sorted(plan["requires"])
    result["evidence"]["plan"] = {
        k: plan[k]
        for k in ("target_id", "poses", "release_width_m", "mode", "descent_path", "thresholds_ref")
    }
    if plan["descent_path"] == "dense_waypoints":
        result["unverified"].append(
            "descent_and_retreat_are_waypoint_sequences_not_linear_interpolation"
        )
    th, frame, normal = plan["thresholds"], plan["frame_id"], np.asarray(plan["support"]["normal"])
    deadline = t0 + plan["timeout_s"]
    target = plan["target_id"]
    ids = [target] + [nb["id"] for nb in plan["neighbors"]]
    held_req = {"object_id": target, "T_TCP_object": plan["T_TCP_object"]}

    def remaining() -> float:
        if ctx.cancelled():
            raise PhaseFailed(
                "cancelled",
                trace[-1]["phase"] if trace else "validate",
                "cancel requested",
                "unverified",
            )
        r = deadline - ctx.now()
        if r <= 0:
            raise PhaseFailed(
                "timeout", trace[-1]["phase"] if trace else "validate", "skill deadline exceeded"
            )
        return r

    def invoke(phase: str, capability: str, req: dict) -> dict:
        req = {**req, "timeout_s": min(remaining(), float(req.get("timeout_s", 1e9)))}
        issued[0] = True
        r = ctx.invoke(capability, req)
        t_done = ctx.now()
        result["children"].append(r.get("invocation_id"))
        ev = r.get("evidence") or {}
        measured = ev.get("measured_pose") or (ev.get("final") or {}).get("measured_pose")
        ts = ev.get("state_timestamp_s") or (ev.get("final") or {}).get("state_timestamp_s")
        if isinstance(measured, dict) and isinstance(ts, (int, float)):
            tcp_states.append(
                {
                    "pose": measured,
                    "timestamp_s": float(ts),
                    "source": f"{capability}.measured_pose",
                }
            )
        g = ev.get("gripper")
        if isinstance(g, dict) and g.get("source"):
            gripper_states.append(g)
        log(
            phase,
            command=capability,
            status=r.get("status"),
            failure_type=r.get("failure_type"),
            invocation_id=r.get("invocation_id"),
        )
        st = r.get("status")
        if st == "succeeded":
            return {**r, "_measured": measured, "_t_done": t_done}
        ft = r.get("failure_type") or (r.get("error") or {}).get("code")
        detail = f"{capability} {st}: {ft} ({(r.get('error') or {}).get('message', '')})"
        if ft in ("grasp_lost", "held_object_slipped"):
            raise PhaseFailed("object_dropped", phase, detail, failure_type=ft)
        if ft in ("timeout", "cancelled"):
            raise PhaseFailed(ft, phase, detail, "failed" if ft == "timeout" else "unverified")
        if ft == "release_failed":
            raise PhaseFailed("release_failed", phase, detail)
        if st == "failed" and ft == "motion_not_converged":
            return {**r, "_measured": measured, "_t_done": t_done}  # judged from the measured pose
        if st == "unverified":
            raise PhaseFailed(
                "backend_error", phase, detail, "unverified", failure_type=ft or "evidence_missing"
            )
        raise PhaseFailed(
            "backend_error", phase, detail, "failed", failure_type=ft or "backend_error"
        )

    def observe(phase: str, object_ids: list[str], after: float) -> dict[str, dict]:
        """Fresh records (timestamp after ``after``, within max age) for every id; polled
        for at most twice the max observation age."""
        give_up = min(deadline, ctx.now() + 2.0 * th["max_observation_age_s"])
        got: dict[str, dict] = {}
        while True:
            remaining()
            for r in ctx.observe(object_ids, require_orientation=True):
                t = r.get("timestamp_s")
                if (
                    isinstance(t, (int, float))
                    and t > after
                    and ctx.now() - t <= th["max_observation_age_s"]
                    and r.get("object_id") in object_ids
                ):
                    seen.append(r)
                    got[r["object_id"]] = r
            if all(i in got for i in object_ids) or ctx.now() >= give_up:
                break
            sleep(min(0.05, max(0.0, give_up - ctx.now())))
        log(
            phase,
            observed=sorted(got),
            newest_s=max((r["timestamp_s"] for r in got.values()), default=None),
        )
        if not got:
            raise PhaseFailed(
                "observation_stale",
                phase,
                f"no observation newer than t={after:.3f}",
                "unverified" if issued[0] else "rejected",
            )
        return got

    def state() -> dict | None:
        fn = getattr(ctx, "state", None)
        if fn is None:
            return None
        s = fn()
        g = s.get("gripper") if isinstance(s, dict) else None
        if isinstance(g, dict) and g.get("source"):
            gripper_states.append(g)
        if isinstance(s, dict) and isinstance(s.get("tcp_pose"), dict):
            tcp_states.append(
                {"pose": s["tcp_pose"], "timestamp_s": s.get("timestamp_s"), "source": "state"}
            )
        return s

    def obs_pose(rec: dict | None) -> dict | None:
        if (
            not isinstance(rec, dict)
            or rec.get("visibility") not in VISIBLE
            or rec.get("quaternion_xyzw") is None
            or rec.get("position_m") is None
        ):
            return None
        try:
            return check_pose(
                {k: rec[k] for k in ("frame_id", "position_m", "quaternion_xyzw")},
                "observation",
                frame,
            )
        except RequestRejected:
            return None

    def held_offset(rec: dict | None, tcp: dict | None) -> float | None:
        p = obs_pose(rec)
        if p is None or tcp is None:
            return None
        return float(
            np.linalg.norm(
                np.subtract(
                    p["position_m"], geo.object_from_tcp(tcp, plan["T_TCP_object"])["position_m"]
                )
            )
        )

    def converged(phase: str, r: dict, goal: dict, code_lateral: str) -> dict:
        m = r.get("_measured")
        if not isinstance(m, dict):
            raise PhaseFailed(
                "backend_error",
                phase,
                f"{phase}: no measured TCP pose in the primitive's evidence",
                "unverified",
                failure_type="evidence_missing",
            )
        m = check_pose(m, "measured_pose", frame)
        d = np.subtract(m["position_m"], goal["position_m"])
        high, lateral = float(d @ normal), float(np.linalg.norm(d - (d @ normal) * normal))
        log(phase, tcp_error_along_normal_m=high, tcp_error_lateral_m=lateral)
        if phase == "descent" and high > th["tcp_tolerance_m"]:
            raise PhaseFailed(
                "descent_blocked",
                phase,
                f"descent stopped {high * 1e3:.1f} mm high; still holding, not released",
            )
        if (
            lateral > th["tcp_tolerance_m"]
            or abs(high) > th["tcp_tolerance_m"]
            or r.get("status") != "succeeded"
        ):
            raise PhaseFailed(
                code_lateral,
                phase,
                f"TCP {lateral * 1e3:.1f} mm lateral / {high * 1e3:.1f} mm vertical off its {phase} pose",
            )
        return m

    P = plan["poses"]
    try:
        # --- fresh pre-action evidence: the tile is in the hand; the neighbours' baseline
        s0 = state()
        got = observe("pre_observe", ids, after=t0 - th["max_observation_age_s"])
        holding = ((s0 or {}).get("gripper") or {}).get("holding")
        off = held_offset(got.get(target), (s0 or {}).get("tcp_pose"))
        if holding is False or (off is not None and off > th["position_tolerance_m"]):
            raise PhaseFailed(
                "holding_not_confirmed",
                "pre_observe",
                f"gripper holding={holding}; tile {'unseen' if off is None else f'{off * 1e3:.1f} mm'} from where T_TCP_object puts it",
                "rejected",
            )
        if off is None:
            result["unverified"].append("holding_precheck_without_tcp_state")
        baseline = []
        for nb in plan["neighbors"]:
            rec = got.get(nb["id"])
            baseline.append(rec if obs_pose(rec) is not None else nb["observation"])
        events["action_start_s"] = ctx.now()

        # --- carry to preplace (runtime checks the held tile follows), then our own check
        r = invoke(
            "carry",
            "move_waypoints",
            {
                "waypoints": [P["preplace"]],
                "linear": False,
                "held": held_req,
                "speed_m_s": plan["speed_m_s"],
            },
        )
        tcp = converged("carry", r, P["preplace"], "tcp_not_converged")
        rec = observe("carry_check", [target], after=r["_t_done"]).get(target)
        off = held_offset(rec, tcp)
        if off is not None and off > th["position_tolerance_m"]:
            raise PhaseFailed(
                "object_dropped",
                "carry",
                f"after the carry the tile is {off * 1e3:.1f} mm from where T_TCP_object puts it",
            )
        if off is None:
            raise PhaseFailed(
                "holding_not_confirmed",
                "carry",
                "the tile is not observed with a pose in the hand after the carry",
                "unverified",
            )

        # --- bounded descent: a straight leg as dense waypoints (or linear if declared)
        r = invoke(
            "descent",
            "move_waypoints",
            {
                "waypoints": plan["descent_waypoints"],
                "linear": plan["descent_path"] == "linear",
                "held": held_req,
                "speed_m_s": plan["descent_speed_m_s"],
            },
        )
        tcp = converged("descent", r, P["release"], "tcp_not_converged")

        # --- support assessment before letting go
        sup_id = plan["support"].get("observation_id")
        got = observe("support_check", [target] + ([sup_id] if sup_id else []), after=r["_t_done"])
        if sup_id:
            sp = got.get(sup_id)
            if sp is None or sp.get("position_m") is None:
                raise PhaseFailed(
                    "support_unconfirmed",
                    "support_check",
                    f"support observation {sup_id!r} missing; not released",
                    "unverified",
                )
            dz = geo.height_over_plane(sp["position_m"], plan["support"]["point_m"], normal)
            log("support_check", support_offset_m=dz, source=sp.get("source"))
            if abs(dz) > th["support_height_tolerance_m"]:
                raise PhaseFailed(
                    "support_mismatch",
                    "support_check",
                    f"observed support is {dz * 1e3:.1f} mm from the calibrated plane; not released",
                )
        else:
            result["unverified"].append("support_contact_not_sensed_before_release")
        held_pose = obs_pose(got.get(target))
        if held_pose is not None:
            tilt = geo.tilt_rad(held_pose, normal, up_symmetric=plan["geometry"]["up_symmetric"])
            low = geo.lowest_point_height(
                held_pose, plan["geometry"], plan["support"]["point_m"], normal
            )
            log(
                "support_check",
                held_tilt_rad=tilt,
                held_lowest_m=low,
                held_offset_m=held_offset(got.get(target), tcp),
            )
            if tilt > th["tilt_tolerance_rad"]:
                raise PhaseFailed(
                    "placement_error",
                    "support_check",
                    f"the held tile is tilted {math.degrees(tilt):.1f} deg before release; not released",
                )
            if low < -th["support_height_tolerance_m"]:
                raise PhaseFailed(
                    "support_mismatch",
                    "support_check",
                    f"the held tile reads {low * 1e3:.1f} mm into the support; not released",
                )

        # --- controlled release, then retreat
        invoke(
            "release",
            "set_gripper",
            {"width_m": plan["release_width_m"], "purpose": "release", "target_id": target},
        )
        events["release_s"] = ctx.now()
        observe("release_check", ids, after=events["release_s"])
        r = invoke(
            "retreat",
            "move_waypoints",
            {
                "waypoints": plan["retreat_waypoints"],
                "linear": plan["descent_path"] == "linear",
                "speed_m_s": plan["descent_speed_m_s"],
            },
        )
        events["retreat_done_s"] = ctx.now()
        try:
            converged("retreat", r, P["retreat"], "retreat_blocked")
        except PhaseFailed as exc:
            raise PhaseFailed("retreat_blocked", "retreat", f"{exc} (released)") from exc
        state()

        # --- the settle window: distinct new frames until it is long enough
        first, n = None, 0
        while True:
            got = observe(
                "settle",
                ids,
                after=max(events["retreat_done_s"], seen[-1]["timestamp_s"] if seen else 0.0),
            )
            t = got[target]["timestamp_s"] if target in got else None
            if t is not None:
                first = t if first is None else first
                n += 1
                if t - first >= th["settle_window_s"] and n >= th["min_window_samples"]:
                    break
            sleep(min(th["settle_window_s"] / max(1, th["min_window_samples"] - 1), remaining()))
    except PhaseFailed as exc:
        status = "failed" if exc.status == "rejected" and issued[0] else exc.status
        set_failure(result, exc.code, exc.phase, str(exc), status, failure_type=exc.failure_type)
    except RequestRejected as exc:
        set_failure(
            result,
            exc.code,
            trace[-1]["phase"] if trace else "validate",
            str(exc),
            "failed" if issued[0] else "rejected",
        )
    except Exception as exc:  # a broken context is never success
        set_failure(
            result,
            "backend_error",
            trace[-1]["phase"] if trace else "validate",
            f"{type(exc).__name__}: {exc}",
            "failed" if issued[0] else "rejected",
        )
    else:
        verdict = verify_stable_placement(
            {
                "request_id": request.get("request_id"),
                "frame_id": frame,
                "target_id": target,
                "geometry": request["geometry"],
                "target_object_pose": plan["goal"],
                "support": request["support"],
                "neighbors_baseline": baseline,
                "events": events,
                "observations": seen,
                "gripper_states": gripper_states,
                "tcp_states": tcp_states,
                "tolerances_ref": plan["thresholds_ref"],
                "thresholds_override": request.get("thresholds_override"),
                "now_s": ctx.now(),
                "gripper_tip_reach_m": plan["tip_reach_m"],
            }
        )
        result["evidence"]["verification"] = {
            k: verdict[k] for k in ("status", "failure", "faults", "unknowns", "invocation_id")
        }
        result["evidence"]["postcondition"] = verdict["evidence"]["postcondition"]
        for k in ("window", "target", "neighbors", "release", "evidence_tier", "evidence_tiers"):
            if k in verdict["evidence"]:
                result["evidence"][k] = verdict["evidence"][k]
        result["observations"] = verdict["observations"]
        result["privileged_inputs"] = verdict["privileged_inputs"]
        result["unverified"] += verdict["unverified"]
        result["status"] = verdict["status"]
        if verdict["failure"]:
            f = verdict["failure"]
            set_failure(
                result,
                f["code"],
                f["phase"],
                f["detail"],
                **{k: v for k, v in f.items() if k not in ("code", "phase", "detail")},
            )
            result["failure_type"] = verdict["failure_type"]
        result["phase"] = "verify"

    if any(r.get("privileged") for r in seen):
        result["privileged_inputs"] = True
    result["evidence"]["events"] = events
    result["evidence"]["observation_count"] = len(seen)
    result["elapsed_s"] = ctx.now() - t0
    result["cause_hypotheses"] = classify(result, seen, plan)
    result["recovery"] = plan_recovery(
        result,
        attempts_used=int(request.get("recovery_attempts_used", 0)),
        max_attempts=int(request.get("max_recovery_attempts", 1)),
    )
    result["recovery_hints"] = RECOVERY_HINTS[result["recovery"]["action"]]
    if result["status"] != "succeeded" and result["failure"]:
        result["phase"] = PHASE.get(result["failure"]["phase"], result["phase"])
    return result
