"""place_upright: carry a held tile to its target pose, set it down, let go, back off,
then hand the outcome to the independent observer (verify_stable_placement).

The motion is the hub's placing-an-upright-domino-beside-neighbours place half
(transit over the row, vertical descent to ``release_gap`` over the table, open only
to the tile's width plus ``release_clearance`` a side, retreat straight up), expressed
as runtime primitives with explicit capability requirements:

    validate + capabilities + fresh holding / scene evidence      (no command)
    -> plan preplace / release / retreat TCP poses + clearance     (no command)
    -> carry to preplace            move(linear=False)       cartesian_motion
    -> bounded descent              move(linear=True)        linear_cartesian_motion
                                    guarded_descent(...)     guarded_descent (contact mode)
    -> support assessment           observe()                object_pose_feedback
    -> controlled release           release(width)           gripper_release
    -> collision-checked retreat    move(linear=True)
    -> settle-window observation    observe() x N
    -> verify_stable_placement      (pure)

One deadline covers queueing, every call and the observation window; the whole
sequence runs inside one ``station_transaction`` lease. A failure or cancel after the
first command calls ``stop()`` and keeps the gripper as it is (no automatic opening,
no second press).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from . import geometry as geo
from . import upstream
from .classify import classify
from .contract import (
    RequestRejected,
    check_capabilities,
    check_geometry,
    check_pose,
    check_support,
    check_thresholds,
    finite,
    result_skeleton,
    vec,
)
from .recovery import plan_recovery
from .verify import verify_stable_placement


class PlacementBackend(Protocol):
    """The runtime primitives place_upright composes. Agent 05 provides them on the
    workcell runtime; tests use a synthetic double (never presented as YAM/Isaac)."""

    def describe(self) -> dict: ...
    def begin_transaction(self, *, timeout_s: float) -> None: ...
    def end_transaction(self) -> None: ...
    def move(self, pose: dict, *, linear: bool, speed_m_s: float, timeout_s: float) -> dict: ...
    def guarded_descent(
        self, pose: dict, *, max_travel_m: float, speed_m_s: float, timeout_s: float
    ) -> dict: ...
    def release(self, width_m: float, *, timeout_s: float) -> dict: ...
    def get_state(self, *, timeout_s: float) -> dict: ...
    def observe(self, *, timeout_s: float) -> dict: ...
    def stop(self) -> None: ...


class DeadlineExceeded(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


class PhaseFailed(RuntimeError):
    def __init__(self, code: str, phase: str, detail: str, status: str = "failed"):
        super().__init__(detail)
        self.code, self.phase, self.status = code, phase, status


class Clock:
    """Wall time for timestamps, monotonic for the deadline, an injectable sleep."""

    def __init__(
        self,
        wall: Callable[[], float] = time.time,
        mono: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.wall, self.mono, self.sleep = wall, mono, sleep


# ----------------------------------------------------------------------------- planning (pure)


def plan_place(request: dict) -> dict:
    """Validate a request and compute every pose and clearance before anything moves.
    Raises RequestRejected with a failure code."""
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
    th = check_thresholds(request.get("thresholds"))
    mode = request.get("mode", "geometric")
    now = finite(request.get("now_s"), "now_s", 0, 1e12)
    normal = np.asarray(support["normal"])

    # Target: upright on the support, on the plane, inside the region.
    tilt = geo.tilt_rad(goal, normal, up_symmetric=g["up_symmetric"])
    if tilt > th["tilt_tolerance_rad"]:
        raise RequestRejected(
            "infeasible_geometry",
            f"target pose is tilted {math.degrees(tilt):.1f} deg off the support normal",
        )
    h0 = geo.height_over_plane(goal["position_m"], support["point_m"], normal)
    if abs(h0) > th["support_height_tolerance_m"]:
        raise RequestRejected(
            "infeasible_geometry",
            f"target origin (bottom face centre) is {h0 * 1e3:.1f} mm off the support plane",
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

    # Holding evidence (Agent 01's postcondition): identity, T_tcp_object, freshness.
    hold = request.get("holding")
    if not isinstance(hold, dict):
        raise RequestRejected("holding_not_confirmed", "no holding evidence")
    if hold.get("object_id") != target_id:
        raise RequestRejected(
            "holding_not_confirmed", f"held object {hold.get('object_id')!r} is not {target_id!r}"
        )
    if not hold.get("source") or not hold.get("gripper_id"):
        raise RequestRejected(
            "holding_not_confirmed", "holding evidence needs source and gripper_id"
        )
    t_tcp_obj = check_pose(hold.get("T_tcp_object"), "holding.T_tcp_object", "tcp")
    age = now - finite(hold.get("timestamp_s"), "holding.timestamp_s", 0, 1e12)
    if age > th["max_observation_age_s"] or age < -0.1:
        raise RequestRejected("observation_stale", f"holding evidence is {age:.2f} s old")
    std = finite(hold.get("position_std_m"), "holding.position_std_m", 0, 1)
    if std > th["max_holding_std_m"]:
        raise RequestRejected(
            "holding_not_confirmed",
            f"T_tcp_object uncertainty {std * 1e3:.1f} mm is over {th['max_holding_std_m'] * 1e3:.1f} mm",
        )

    # Neighbours: fresh observations, no overlap, the open hand clear of them.
    neighbors = []
    for i, nb in enumerate(request.get("neighbors") or []):
        if not isinstance(nb, dict) or not nb.get("id") or not nb.get("source"):
            raise RequestRejected(
                "invalid_request", f"neighbors[{i}] needs id, pose, timestamp_s, source"
            )
        p = check_pose(nb.get("pose"), f"neighbors[{i}].pose", frame)
        a = now - finite(nb.get("timestamp_s"), f"neighbors[{i}].timestamp_s", 0, 1e12)
        if a > th["max_observation_age_s"] or a < -0.1:
            raise RequestRejected(
                "observation_stale", f"neighbour {nb['id']} was observed {a:.2f} s ago"
            )
        neighbors.append(
            {
                "id": str(nb["id"]),
                "pose": p,
                "timestamp_s": float(nb["timestamp_s"]),
                "source": nb["source"],
            }
        )
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

    # TCP poses: T_station_tcp = T_station_object * inv(T_tcp_object).
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

    # Hand clearance along the final approach (holding the tile) and the retreat (open).
    hang = geo.height_over_plane(tcp_goal["position_m"], support["point_m"], normal)

    def sweep(start: dict, end: dict, open_w: float, carrying: bool) -> list[str]:
        """Neighbours the hand's envelope (and the held tile) passes through below their
        tops plus clearance, along a straight leg."""
        hits = []
        for s in np.linspace(0.0, 1.0, 21):
            p = (1 - s) * np.asarray(start["position_m"]) + s * np.asarray(end["position_m"])
            tcp_h = geo.height_over_plane(p, support["point_m"], normal)
            lowest = tcp_h - (max(tip_reach, hang) if carrying else tip_reach)
            if lowest > g["height_m"] + pl.NEIGHBOUR_CLEARANCE:
                continue
            env = pl.hand_envelope(p[:2], goal_facing, size, open_w)
            for nb in neighbors:
                if (
                    pl.poly_distance(env, pl.footprint(*geo.xy_facing(nb["pose"]), size))
                    < pl.NEIGHBOUR_CLEARANCE
                ):
                    hits.append(nb["id"])
        return sorted(set(hits))

    if r_dir @ normal < 1 - 1e-6:
        hit = sweep(release_pose, retreat_pose, width, False)
        if hit:
            raise RequestRejected(
                "retreat_blocked", f"the retreat sweeps the open hand through neighbour(s) {hit}"
            )
    if a_dir @ normal > -1 + 1e-6:
        hit = sweep(preplace, release_pose, width, True)
        if hit:
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
    descent = request.get("descent") or {}
    d_speed = finite(descent.get("speed_m_s", min(speed, 0.01)), "descent.speed_m_s", 1e-4, speed)
    max_travel = finite(
        descent.get("max_travel_m", a_dist + th["tcp_tolerance_m"]),
        "descent.max_travel_m",
        0.001,
        0.5,
    )
    if max_travel < a_dist:
        raise RequestRejected(
            "invalid_request", "descent.max_travel_m is shorter than the approach distance"
        )
    timeout = finite(request.get("timeout_s"), "timeout_s", 0.1, 600)
    need = a_dist / d_speed + r_dist / d_speed + th["settle_window_s"]
    if need > timeout:
        raise RequestRejected(
            "invalid_request",
            f"timeout_s {timeout:.1f} s is under the {need:.1f} s the descent, retreat and settle window need",
        )
    return {
        "frame_id": frame,
        "target_id": target_id,
        "geometry": g,
        "goal": goal,
        "support": support,
        "thresholds": th,
        "mode": mode,
        "T_tcp_object": t_tcp_obj,
        "neighbors": neighbors,
        "release_width_m": width,
        "tip_reach_m": tip_reach,
        "poses": {
            "preplace": preplace,
            "release": release_pose,
            "retreat": retreat_pose,
            "tcp_goal": tcp_goal,
        },
        "speed_m_s": speed,
        "descent_speed_m_s": d_speed,
        "max_travel_m": max_travel,
        "timeout_s": timeout,
    }


# ----------------------------------------------------------------------------- execution


def place_upright(
    request: dict,
    backend: PlacementBackend,
    *,
    clock: Clock | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict:
    clock = clock or Clock()
    cancel = cancel or (lambda: False)
    t0_mono, t0_wall = clock.mono(), clock.wall()
    request = {**request, "now_s": request.get("now_s", t0_wall)}
    result = result_skeleton("place_upright", request.get("request_id"))
    result["scene_version"] = request.get("scene_version")
    trace = result["phase_trace"]
    state = {"issued": False, "lease": False}
    observations: list[dict] = []
    events: dict[str, float] = {"action_start_s": t0_wall}

    def log(phase: str, **kw: Any) -> None:
        trace.append({"phase": phase, "t_s": clock.wall(), **kw})

    try:
        plan = plan_place(request)
    except upstream.UpstreamUnavailable as exc:
        result["failure"] = {"code": "dependency_missing", "phase": "validate", "detail": str(exc)}
        return result
    except RequestRejected as exc:
        result["failure"] = {"code": exc.code, "phase": "validate", "detail": str(exc)}
        log("validate", ok=False, code=exc.code)
        return result
    log("validate", ok=True)
    result["evidence"]["plan"] = {
        k: plan[k] for k in ("target_id", "poses", "release_width_m", "mode")
    }
    th, frame, normal = plan["thresholds"], plan["frame_id"], np.asarray(plan["support"]["normal"])
    deadline = t0_mono + plan["timeout_s"]

    def remaining() -> float:
        if cancel():
            raise Cancelled("cancel requested")
        r = deadline - clock.mono()
        if r <= 0:
            raise DeadlineExceeded("deadline exceeded")
        return r

    try:
        description = backend.describe()
        result["backend"] = {k: description.get(k) for k in ("name", "mode", "frame_id")}
        missing = check_capabilities(description, plan["mode"])
        if missing:
            raise PhaseFailed(
                "capability_missing", "validate", f"backend lacks {missing}", "rejected"
            )
        if description.get("frame_id") != frame:
            raise PhaseFailed(
                "invalid_request",
                "validate",
                "backend frame differs from the request frame",
                "rejected",
            )

        backend.begin_transaction(timeout_s=remaining())
        state["lease"] = True
        log("transaction", acquired=True)

        def observe(phase: str, after: float) -> dict:
            """A fresh, new observation (timestamp after ``after``), polled until the deadline."""
            last_ts = None
            give_up = min(deadline, clock.mono() + 2.0 * th["max_observation_age_s"])
            while clock.mono() < give_up:
                obs = backend.observe(timeout_s=remaining())
                ts = obs.get("timestamp_s") if isinstance(obs, dict) else None
                if (
                    isinstance(ts, (int, float))
                    and ts > after
                    and clock.wall() - ts <= th["max_observation_age_s"]
                ):
                    observations.append(obs)
                    log(
                        phase,
                        observation=obs.get("sample_id"),
                        timestamp_s=ts,
                        source=obs.get("source"),
                    )
                    return obs
                last_ts = ts
                clock.sleep(min(0.02, max(0.0, deadline - clock.mono())))
            raise PhaseFailed(
                "observation_stale",
                phase,
                f"no observation newer than t={after:.3f} within {2.0 * th['max_observation_age_s']:.1f} s (last {last_ts})",
                "unverified" if state["issued"] else "rejected",
            )

        def measured_tcp() -> dict:
            s = backend.get_state(timeout_s=remaining())
            try:
                p = check_pose(s.get("pose"), "measured tcp pose", frame)
            except RequestRejected as exc:
                raise PhaseFailed("backend_error", "state", str(exc), "unverified") from exc
            return p

        def held_offset(obs: dict, tcp: dict) -> tuple[float | None, dict | None]:
            rec = (obs.get("objects") or {}).get(plan["target_id"])
            p = rec.get("pose") if isinstance(rec, dict) and rec.get("visible", True) else None
            if not isinstance(p, dict):
                return None, None
            pred = geo.object_from_tcp(tcp, plan["T_tcp_object"])
            return float(np.linalg.norm(np.subtract(p["position_m"], pred["position_m"]))), p

        def gripper_holding(obs: dict) -> bool | None:
            gr = obs.get("gripper")
            if isinstance(gr, dict) and gr.get("source") and isinstance(gr.get("holding"), bool):
                return gr["holding"]
            return None

        def move(phase: str, pose: dict, *, linear: bool, speed: float) -> dict:
            state["issued"] = True
            ack = backend.move(pose, linear=linear, speed_m_s=speed, timeout_s=remaining())
            tcp = measured_tcp()
            err = float(np.linalg.norm(np.subtract(tcp["position_m"], pose["position_m"])))
            log(
                phase,
                command="move",
                linear=linear,
                ack_ok=bool(isinstance(ack, dict) and ack.get("ok")),
                tcp_error_m=err,
                measured_tcp=tcp,
            )
            return {"ack": ack, "tcp": tcp, "error_m": err}

        # --- fresh pre-action evidence: the tile is in the hand, the neighbours' baseline
        pre = observe("pre_observe", after=t0_wall - th["max_observation_age_s"])
        tcp0 = measured_tcp()
        off, _ = held_offset(pre, tcp0)
        holding = gripper_holding(pre)
        if holding is False or (off is not None and off > th["position_tolerance_m"]):
            raise PhaseFailed(
                "holding_not_confirmed",
                "pre_observe",
                f"gripper holding={holding}, tile {off if off is None else round(off * 1e3, 1)} mm from where T_tcp_object puts it",
                "rejected",
            )
        if off is None and holding is None:
            raise PhaseFailed(
                "holding_not_confirmed",
                "pre_observe",
                "neither the tile nor the gripper state is observed",
                "rejected",
            )
        baseline = {}
        for nb in plan["neighbors"]:
            rec = (pre.get("objects") or {}).get(nb["id"])
            if (
                isinstance(rec, dict)
                and isinstance(rec.get("pose"), dict)
                and rec.get("visible", True)
            ):
                baseline[nb["id"]] = {
                    "pose": rec["pose"],
                    "timestamp_s": pre["timestamp_s"],
                    "source": pre.get("source"),
                }
            else:
                baseline[nb["id"]] = {k: nb[k] for k in ("pose", "timestamp_s", "source")}
        events["action_start_s"] = clock.wall()

        # --- carry to preplace, then confirm the tile came along
        P = plan["poses"]
        m = move("carry", P["preplace"], linear=False, speed=plan["speed_m_s"])
        if (
            not (isinstance(m["ack"], dict) and m["ack"].get("ok"))
            or m["error_m"] > th["tcp_tolerance_m"]
        ):
            raise PhaseFailed(
                "placement_error",
                "carry",
                f"TCP {m['error_m'] * 1e3:.1f} mm from preplace (ack {m['ack']})",
            )
        obs = observe("carry_check", after=trace[-1]["t_s"])
        off, _ = held_offset(obs, m["tcp"])
        holding = gripper_holding(obs)
        if holding is False or (off is not None and off > th["position_tolerance_m"]):
            raise PhaseFailed(
                "object_dropped",
                "carry",
                f"after the carry: gripper holding={holding}, tile offset {off}",
            )
        if off is None and holding is None:
            raise PhaseFailed(
                "holding_not_confirmed",
                "carry",
                "the tile is no longer observed in the hand",
                "unverified",
            )

        # --- bounded descent
        if plan["mode"] == "contact_feedback":
            state["issued"] = True
            ack = backend.guarded_descent(
                P["release"],
                max_travel_m=plan["max_travel_m"],
                speed_m_s=plan["descent_speed_m_s"],
                timeout_s=remaining(),
            )
            tcp = measured_tcp()
            err = float(np.linalg.norm(np.subtract(tcp["position_m"], P["release"]["position_m"])))
            log("descent", command="guarded_descent", ack=ack, tcp_error_m=err, measured_tcp=tcp)
            m = {"ack": ack, "tcp": tcp, "error_m": err}
        else:
            m = move("descent", P["release"], linear=True, speed=plan["descent_speed_m_s"])
        d = np.subtract(m["tcp"]["position_m"], P["release"]["position_m"])
        high, lateral = float(d @ normal), float(np.linalg.norm(d - (d @ normal) * normal))
        if high > th["tcp_tolerance_m"]:
            raise PhaseFailed(
                "descent_blocked",
                "descent",
                f"descent stopped {high * 1e3:.1f} mm high; still holding, not released",
            )
        if lateral > th["tcp_tolerance_m"] or not (
            isinstance(m["ack"], dict) and m["ack"].get("ok")
        ):
            raise PhaseFailed(
                "placement_error",
                "descent",
                f"TCP {lateral * 1e3:.1f} mm off the release pose (ack {m['ack']}); not released",
            )

        # --- support assessment before letting go
        obs = observe("support_check", after=trace[-1]["t_s"])
        sup = obs.get("support_plane")
        support_note = "not_sensed"
        if isinstance(sup, dict) and sup.get("source"):
            dz = geo.height_over_plane(sup["point_m"], plan["support"]["point_m"], normal)
            support_note = f"observed_offset_{dz * 1e3:.1f}mm"
            log("support_check", support_offset_m=dz, source=sup["source"])
            if abs(dz) > th["support_height_tolerance_m"]:
                raise PhaseFailed(
                    "support_mismatch",
                    "support_check",
                    f"observed support is {dz * 1e3:.1f} mm from the calibrated plane; not released",
                )
        _, held = held_offset(obs, m["tcp"])
        if held is not None:
            tilt = geo.tilt_rad(
                check_pose(held, "held pose", frame),
                normal,
                up_symmetric=plan["geometry"]["up_symmetric"],
            )
            low = geo.lowest_point_height(
                check_pose(held, "held pose", frame),
                plan["geometry"],
                plan["support"]["point_m"],
                normal,
            )
            log("support_check", held_tilt_rad=tilt, held_lowest_m=low)
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
        if plan["mode"] == "geometric" and support_note == "not_sensed":
            result["unverified"].append("support_contact_not_sensed_before_release")

        # --- controlled release
        state["issued"] = True
        ack = backend.release(plan["release_width_m"], timeout_s=remaining())
        events["release_s"] = clock.wall()
        log("release", command="release", width_m=plan["release_width_m"], ack=ack)
        if not (isinstance(ack, dict) and ack.get("ok")):
            raise PhaseFailed(
                "release_not_confirmed", "release", f"release command not acknowledged ({ack})"
            )
        observe("release_check", after=events["release_s"])

        # --- retreat
        m = move("retreat", P["retreat"], linear=True, speed=plan["descent_speed_m_s"])
        events["retreat_done_s"] = clock.wall()
        if (
            not (isinstance(m["ack"], dict) and m["ack"].get("ok"))
            or m["error_m"] > th["tcp_tolerance_m"]
        ):
            raise PhaseFailed(
                "retreat_blocked",
                "retreat",
                f"retreat ended {m['error_m'] * 1e3:.1f} mm from its pose (ack {m['ack']}); released",
            )

        # --- the settle window: distinct new frames until it is long enough
        first = None
        while True:
            obs = observe(
                "settle", after=max(observations[-1]["timestamp_s"], events["retreat_done_s"])
            )
            first = first if first is not None else obs["timestamp_s"]
            n = sum(1 for o in observations if o["timestamp_s"] >= events["retreat_done_s"])
            if (
                obs["timestamp_s"] - first >= th["settle_window_s"]
                and n >= th["min_window_samples"]
            ):
                break
            clock.sleep(
                min(th["settle_window_s"] / max(1, th["min_window_samples"] - 1), remaining())
            )
    except PhaseFailed as exc:
        result["status"] = (
            exc.status if not (exc.status == "rejected" and state["issued"]) else "failed"
        )
        result["failure"] = {"code": exc.code, "phase": exc.phase, "detail": str(exc)}
    except DeadlineExceeded as exc:
        result["status"] = "failed" if state["issued"] else "rejected"
        result["failure"] = {
            "code": "timeout",
            "phase": trace[-1]["phase"] if trace else "validate",
            "detail": str(exc),
        }
    except Cancelled as exc:
        result["status"] = "unverified" if state["issued"] else "rejected"
        result["failure"] = {
            "code": "cancelled",
            "phase": trace[-1]["phase"] if trace else "validate",
            "detail": str(exc),
        }
    except Exception as exc:  # a broken driver is never success
        result["status"] = "failed" if state["issued"] else "rejected"
        result["failure"] = {
            "code": "backend_error",
            "phase": trace[-1]["phase"] if trace else "validate",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    else:
        verdict = verify_stable_placement(
            {
                "request_id": request.get("request_id"),
                "frame_id": frame,
                "target_id": plan["target_id"],
                "geometry": request["geometry"],
                "target_object_pose": plan["goal"],
                "support": request["support"],
                "neighbors_baseline": baseline,
                "events": events,
                "observations": observations,
                "thresholds": th,
                "now_s": clock.wall(),
                "gripper_tip_reach_m": plan["tip_reach_m"],
            }
        )
        result["evidence"]["verification"] = verdict
        result["status"] = verdict["status"]
        result["failure"] = verdict["failure"]
        result["unverified"] += verdict["unverified"]
    finally:
        if result["status"] != "succeeded" and state["issued"]:
            try:
                backend.stop()
                log("stop", ok=True)
            except Exception as exc:
                log("stop", ok=False, error=str(exc))
        if state["lease"]:
            try:
                backend.end_transaction()
                log("transaction", released=True)
            except Exception as exc:
                log("transaction", released=False, error=str(exc))
                if result["status"] == "succeeded":
                    result["status"] = "unverified"
                    result["unverified"].append("transaction_release_failed")

    result["evidence"]["events"] = events
    result["evidence"]["observation_count"] = len(observations)
    result["evidence"]["observation_sources"] = sorted(
        {str(o.get("source_kind")) for o in observations}
    )
    result["elapsed_s"] = clock.mono() - t0_mono
    result["cause_hypotheses"] = classify(result, observations, plan)
    result["recovery"] = plan_recovery(
        result,
        attempts_used=int(request.get("recovery_attempts_used", 0)),
        max_attempts=int(request.get("max_recovery_attempts", 1)),
    )
    return result
