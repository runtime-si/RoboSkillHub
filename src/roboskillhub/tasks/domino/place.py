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
from collections.abc import Mapping
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


def plan_place(request: dict, limits: Mapping[str, Any] | None = None) -> dict:
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
    # Default: the held tile's bottom ends the carry TRANSIT_CLEARANCE over the standing
    # tops (the placing skill's transit height), straight over the release pose.
    transit = float(pd.TRANSIT_CLEARANCE)
    a_dist = finite(
        approach.get("distance_m", g["height_m"] + transit), "approach.distance_m", 0.005, 0.5
    )
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

    if neighbors:
        bottom = geo.height_over_plane(preplace["position_m"], support["point_m"], normal) - hang
        if bottom < g["height_m"] + transit - 1e-9:
            raise RequestRejected(
                "neighbor_clearance",
                f"the carry ends with the held tile's bottom {(bottom - g['height_m']) * 1e3:.1f} mm over the "
                f"standing tops; the placing skill keeps {transit * 1e3:.0f} mm (raise approach.distance_m)",
            )
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
    lim = dict(limits or {})
    # Per-primitive cap: the station's max_timeout_s (ctx.limits), else the runtime default.
    primitive_timeout = finite(
        request.get("primitive_timeout_s", lim.get("max_timeout_s", 30.0)),
        "primitive_timeout_s",
        0.1,
        float(lim.get("max_timeout_s", 120.0)),
    )

    # Contact mode (rsh-task/1.1 guarded_press): stop the held tile on the support by force.
    contact = None
    if mode == "contact_feedback":
        c = request.get("contact") or {}
        search = finite(
            c.get("search_m", th["support_height_tolerance_m"] + gap),
            "contact.search_m",
            0.001,
            0.05,
        )
        contact = {
            "force_limit_N": finite(c.get("force_limit_N"), "contact.force_limit_N", 0.01, 100.0),
            "search_m": search,
            "max_travel_m": finite(
                c.get("max_travel_m", 2.0 * search), "contact.max_travel_m", search, 0.1
            ),
            "speed_m_s": finite(
                c.get("speed_m_s", min(d_speed, 0.005)), "contact.speed_m_s", 1e-5, d_speed
            ),
            "precontact": geo.translated(tcp_goal, normal * search),
        }

    # The station's own limits, before anything moves (the runtime would refuse later,
    # possibly after earlier primitives already moved the arm).
    if lim:
        if lim.get("frame_id") not in (None, frame):
            raise RequestRejected(
                "invalid_request", f"station frame is {lim['frame_id']!r}, not {frame!r}"
            )
        vmax = lim.get("max_speed_m_s")
        if vmax is not None and max(speed, d_speed) > float(vmax):
            raise RequestRejected(
                "invalid_request",
                f"speed {max(speed, d_speed)} m/s is over the station's {vmax} m/s",
            )
        wlo, whi = lim.get("workspace_min_m"), lim.get("workspace_max_m")
        if wlo is not None and whi is not None:
            for name, p in (
                ("preplace", preplace),
                ("release", release_pose),
                ("retreat", retreat_pose),
            ):
                if not all(
                    float(a) <= v <= float(b)
                    for a, v, b in zip(wlo, p["position_m"], whi, strict=True)
                ):
                    raise RequestRejected(
                        "infeasible_geometry", f"{name} TCP pose is outside the station workspace"
                    )
        gmin, gmax = lim.get("gripper_min_width_m"), lim.get("gripper_max_width_m")
        if gmin is not None and gmax is not None and not float(gmin) <= width <= float(gmax):
            raise RequestRejected(
                "infeasible_geometry",
                f"release width {width} m is outside the station's gripper range",
            )
        if contact is not None:
            fmax = float(lim.get("max_contact_force_N", 0.0) or 0.0)
            if contact["force_limit_N"] > fmax:
                raise RequestRejected(
                    "capability_missing" if fmax <= 0 else "invalid_request",
                    f"contact.force_limit_N {contact['force_limit_N']} N is over the station's max_contact_force_N {fmax} N",
                )
            if contact["max_travel_m"] > float(lim.get("max_contact_distance_m", 1.0)):
                raise RequestRejected(
                    "invalid_request",
                    "contact.max_travel_m is over the station's max_contact_distance_m",
                )
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
        "descent_waypoints": leg(preplace, contact["precontact"] if contact else release_pose),
        "contact": contact,
        "retreat_waypoints": leg(release_pose, retreat_pose),
        "speed_m_s": speed,
        "descent_speed_m_s": d_speed,
        "timeout_s": timeout,
        "primitive_timeout_s": primitive_timeout,
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
        plan = plan_place(request, getattr(ctx, "limits", None))
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
    result["evidence"]["evidence_level"] = getattr(ctx, "evidence_level", None)
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

    def call(capability: str, req: dict) -> dict:
        """ctx.invoke, with a typed refusal (cancel, expired lease, no station) returned as
        a primitive result instead of escaping."""
        try:
            r = ctx.invoke(capability, req)
        except Exception as exc:
            ft = getattr(exc, "failure_type", None) or "backend_error"
            r = {
                "status": "rejected",
                "failure_type": ft,
                "error": {"code": type(exc).__name__, "message": str(exc)},
            }
        if r.get("invocation_id") and r["invocation_id"] not in result["children"]:
            result["children"].append(r["invocation_id"])
        return r

    def measured_of(r: dict) -> tuple[dict | None, float | None]:
        """The last measured TCP pose a motion primitive reports (evidence.arrivals, or its
        trace when it failed before writing evidence)."""
        ev = r.get("evidence") or {}
        arrivals = ev.get("arrivals") or ([ev] if "measured_pose" in ev else [])
        for a in reversed(arrivals):
            if isinstance(a, dict) and isinstance(a.get("measured_pose"), dict):
                return a["measured_pose"], a.get("state_timestamp_s")
        for e in reversed(r.get("trace") or []):
            if isinstance(e, dict) and isinstance(e.get("measured_pose"), dict):
                return e["measured_pose"], e.get("state_timestamp_s")
        return None, None

    def gripper_of(r: dict) -> dict | None:
        ev = r.get("evidence") or {}
        g = ev.get("gripper_feedback") or ev.get("gripper")
        if not isinstance(g, dict):
            g = next(
                (
                    e.get("gripper_feedback")
                    for e in reversed(r.get("trace") or [])
                    if isinstance(e, dict) and e.get("gripper_feedback")
                ),
                None,
            )
        return g if isinstance(g, dict) and g.get("source") else None

    def invoke(phase: str, capability: str, req: dict) -> dict:
        req = {**req, "timeout_s": max(0.1, min(remaining(), plan["primitive_timeout_s"]))}
        issued[0] = True
        r = call(capability, req)
        t_done = ctx.now()
        measured, ts = measured_of(r)
        if isinstance(measured, dict) and isinstance(ts, (int, float)):
            tcp_states.append(
                {
                    "pose": measured,
                    "timestamp_s": float(ts),
                    "source": f"{capability}.measured_pose",
                }
            )
        g = gripper_of(r)
        if g is not None:
            gripper_states.append(g)
        log(
            phase,
            command=capability,
            status=r.get("status"),
            failure_type=r.get("failure_type"),
            invocation_id=r.get("invocation_id"),
        )
        st = r.get("status")
        out = {**r, "_measured": measured, "_t_done": t_done}
        if st == "succeeded":
            return out
        ft = r.get("failure_type") or (r.get("error") or {}).get("code")
        detail = f"{capability} {st}: {ft} ({(r.get('error') or {}).get('message', '')})"
        if ft in ("grasp_lost", "held_object_slipped", "grasp_missed"):
            raise PhaseFailed("object_dropped", phase, detail, failure_type=ft)
        if ft in ("timeout", "cancelled", "lease_expired"):
            raise PhaseFailed(
                "timeout" if ft == "timeout" else "cancelled",
                phase,
                detail,
                "failed" if ft == "timeout" else "unverified",
                failure_type=ft,
            )
        if ft == "release_failed":
            raise PhaseFailed("release_failed", phase, detail)
        if st == "failed" and ft == "motion_not_converged":
            return out  # judged from the measured pose
        if capability == "set_gripper" and st == "unverified" and ft == "evidence_missing":
            result["unverified"].append("release_without_holding_feedback")
            return out  # the observer still needs the tile resting and the hand clear
        if st == "rejected" and ft == "capability_missing":
            raise PhaseFailed("capability_missing", phase, detail, "rejected")
        if st == "unverified":
            raise PhaseFailed(
                "backend_error", phase, detail, "unverified", failure_type=ft or "evidence_missing"
            )
        raise PhaseFailed(
            "backend_error", phase, detail, "failed", failure_type=ft or "backend_error"
        )

    def observe(phase: str, object_ids: list[str], after: float) -> dict[str, dict]:
        """Fresh records (timestamp after ``after``, within max age) for every id; polled
        for at most twice the max observation age. ``ctx.observe`` returns the
        observe_objects result (records in evidence.observations)."""
        give_up = min(deadline, ctx.now() + 2.0 * th["max_observation_age_s"])
        got: dict[str, dict] = {}
        while True:
            remaining()
            try:
                res = ctx.observe(object_ids, require_orientation=True)
            except Exception as exc:
                ft = getattr(exc, "failure_type", None) or "backend_error"
                raise PhaseFailed(
                    "cancelled" if ft == "cancelled" else "backend_error",
                    phase,
                    str(exc),
                    "unverified" if issued[0] else "rejected",
                    failure_type=ft,
                ) from exc
            if isinstance(res, dict):
                if res.get("invocation_id") and res["invocation_id"] not in result["children"]:
                    result["children"].append(res["invocation_id"])
                if res.get("status") == "rejected" and res.get("failure_type") in (
                    "capability_missing",
                    "cancelled",
                    "lease_expired",
                ):
                    raise PhaseFailed(
                        res["failure_type"]
                        if res["failure_type"] != "lease_expired"
                        else "cancelled",
                        phase,
                        str(res.get("error")),
                        "unverified" if issued[0] else "rejected",
                        failure_type=res["failure_type"],
                    )
                records = (res.get("evidence") or {}).get("observations") or []
            else:
                records = list(res or [])
            for r in records:
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
        try:
            s = fn()
        except Exception as exc:  # a context without station state (offline, refused)
            log("state", ok=False, error=f"{type(exc).__name__}: {exc}")
            return None
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
            pre_pose = obs_pose(got.get(target))
            if pre_pose is not None:
                low = geo.lowest_point_height(
                    pre_pose, plan["geometry"], plan["support"]["point_m"], normal
                )
                if low <= th["support_height_tolerance_m"]:
                    raise PhaseFailed(
                        "holding_not_confirmed",
                        "pre_observe",
                        f"the tile is observed resting on the support (lowest corner {low * 1e3:.1f} mm): "
                        "not in the hand",
                        "rejected",
                    )
        baseline = []
        for nb in plan["neighbors"]:
            rec = got.get(nb["id"])
            baseline.append(rec if obs_pose(rec) is not None else nb["observation"])
        # The action starts after the baseline was observed, on the observer's clock too:
        # a backend stamp a hair ahead of ctx.now() must not post-date the baseline.
        events["action_start_s"] = max(
            [ctx.now()]
            + [
                float(b["timestamp_s"])
                for b in baseline
                if isinstance(b.get("timestamp_s"), (int, float))
            ]
        )

        # --- carry to preplace (runtime checks the held tile follows), then our own check
        carry = [P["preplace"]]
        here = (s0 or {}).get("tcp_pose")
        if isinstance(here, dict) and here.get("frame_id") == frame:
            up = geo.height_over_plane(
                P["preplace"]["position_m"], plan["support"]["point_m"], normal
            ) - geo.height_over_plane(here["position_m"], plan["support"]["point_m"], normal)
            if up > 1e-3:  # rise first, as the placing skill does, then cross at transit height
                rise = np.asarray(here["position_m"]) + normal * up
                carry = [geo.pose(frame, rise, here["quaternion_xyzw"]), P["preplace"]]
        r = invoke(
            "carry",
            "move_waypoints",
            {
                "waypoints": carry,
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
        if plan["contact"] is None:
            tcp = converged("descent", r, P["release"], "tcp_not_converged")
        else:
            # rsh-task/1.1 guarded_press: from just above the expected contact, move the held
            # tile down until the controller stops on force (or runs out of travel).
            C = plan["contact"]
            converged("descent", r, C["precontact"], "tcp_not_converged")
            r = invoke(
                "contact",
                "guarded_press",
                {
                    "axis": {"frame_id": frame, "vector": [float(-x) for x in normal]},
                    "max_travel_m": C["max_travel_m"],
                    "speed_m_s": C["speed_m_s"],
                    "force_limit_N": C["force_limit_N"],
                },
            )
            ev = r.get("evidence") or {}
            stop = ev.get("stop_reason")
            m = r.get("_measured") or ev.get("measured_pose")
            if not isinstance(m, dict):
                raise PhaseFailed(
                    "support_unconfirmed",
                    "support_check",
                    "guarded_press gave no measured pose",
                    "unverified",
                    failure_type="evidence_missing",
                )
            tcp = check_pose(m, "measured_pose", frame)
            dz = float(np.subtract(tcp["position_m"], P["tcp_goal"]["position_m"]) @ normal)
            log(
                "support_check",
                contact_stop=stop,
                max_force_N=ev.get("max_force_N"),
                contact_offset_m=dz,
                support_offset_m=dz,
            )
            if stop == "travel_limit":
                raise PhaseFailed(
                    "support_mismatch",
                    "support_check",
                    f"no contact within {C['max_travel_m'] * 1e3:.1f} mm: the support is lower than calibrated (or absent); not released",
                )
            if stop != "force_threshold":
                raise PhaseFailed(
                    "support_unconfirmed",
                    "support_check",
                    f"guarded_press ended {stop}; not released",
                    "unverified",
                    failure_type="evidence_missing",
                )
            if abs(dz) > th["support_height_tolerance_m"]:
                raise PhaseFailed(
                    "support_mismatch",
                    "support_check",
                    f"contact {dz * 1e3:.1f} mm from where the calibrated support puts it; not released",
                )
            result["evidence"]["support_contact"] = {
                "stop_reason": stop,
                "offset_m": dz,
                "max_force_N": ev.get("max_force_N"),
            }

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
        elif plan["contact"] is None:
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

        # --- controlled release, then retreat (release_s: when the opening was commanded)
        events["release_s"] = ctx.now()
        r = invoke(
            "release",
            "set_gripper",
            {"width_m": plan["release_width_m"], "purpose": "release", "target_id": target},
        )
        observe("release_check", ids, after=r["_t_done"])
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
            k: verdict.get(k) for k in ("status", "failure", "faults", "unknowns", "invocation_id")
        }
        if verdict["status"] == "rejected":
            # The observer refused its inputs; motion has happened, so the placement is
            # not established either way.
            vf = verdict.get("failure") or {}
            result["observations"] = [
                r for r in seen if r.get("timestamp_s", 0) > events.get("release_s", 0)
            ]
            result["status"] = "unverified"
            set_failure(
                result,
                "verification_rejected",
                "verify",
                f"verify_stable_placement rejected its request ({vf.get('code')}): {vf.get('detail')}",
                verification_code=vf.get("code"),
            )
            result["unverified"].append("verification_rejected")
        else:
            ev = verdict.get("evidence") or {}
            result["evidence"]["postcondition"] = ev.get("postcondition")
            for k in (
                "window",
                "target",
                "neighbors",
                "release",
                "evidence_tier",
                "evidence_tiers",
            ):
                if k in ev:
                    result["evidence"][k] = ev[k]
            result["observations"] = verdict.get("observations", [])
            result["privileged_inputs"] = bool(verdict.get("privileged_inputs"))
            result["unverified"] += verdict.get("unverified", [])
            result["status"] = verdict["status"]
            if verdict["status"] == "succeeded":
                result["outputs"] = {
                    "placed_object_id": target,
                    "final_pose": ev["target"]["final_pose"],
                }
            if verdict.get("failure"):
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
