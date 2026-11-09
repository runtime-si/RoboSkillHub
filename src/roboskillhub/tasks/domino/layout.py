"""plan_straight_layout: target poses and a build order for a straight row. No motion.

Reuses the hub's planning-a-domino-layout (footprints, ``poly_distance``, the arm
corridor ``in_corridor``, the toppling physics ``critical_gap``) and the placing skill's
own finger-clearance rule (``finger_conflicts``), so every slot it accepts passes the
check place_upright makes. Placing is not toppling: cascade physics is reported, and
enforced only when ``cascade.require`` is set.

Spacing is given as ``center_pitch_m`` (centre to centre along the row) or
``face_gap_m`` (front face to the next tile's back face); for a straight row of
identical tiles facing along it, ``center_pitch = face_gap + thickness``.
"""

from __future__ import annotations

import math

import numpy as np

from . import geometry as geo
from . import upstream
from .contract import RequestRejected, check_geometry, finite, result_skeleton, vec

ORDERS = ("far_first", "forward", "reverse")
"""far_first is planning-a-domino-layout's rule (farthest from the arm first, from
DominoRobot's FieldPlanner); forward places from the trigger end, reverse toward it."""


def _upstream_size(g: dict) -> dict[str, float]:
    return {"height": g["height_m"], "width": g["width_m"], "thickness": g["thickness_m"]}


def plan_straight_layout(request: dict) -> dict:
    result = result_skeleton("plan_straight_layout", request.get("request_id"))
    try:
        pd, pl = upstream.place(), upstream.layout()
    except upstream.UpstreamUnavailable as exc:
        result["failure"] = {"code": "dependency_missing", "detail": str(exc)}
        return result
    try:
        frame = request.get("frame_id")
        if not isinstance(frame, str) or not frame:
            raise RequestRejected("invalid_request", "frame_id is required")
        g = check_geometry(request.get("geometry"))
        size = _upstream_size(g)
        start = np.array(vec(request.get("start_m"), "start_m", 2))
        heading = finite(request.get("heading_deg"), "heading_deg", -360, 360)
        n = int(finite(request.get("n_tiles"), "n_tiles", 1, 200))
        table_z = finite(request.get("table_z_m"), "table_z_m")
        spacing = request.get("spacing") or {}
        if ("center_pitch_m" in spacing) == ("face_gap_m" in spacing):
            raise RequestRejected(
                "invalid_request", "spacing needs exactly one of center_pitch_m, face_gap_m"
            )
        if "center_pitch_m" in spacing:
            pitch = finite(spacing["center_pitch_m"], "center_pitch_m", 0.0, 1.0)
        else:
            pitch = finite(spacing["face_gap_m"], "face_gap_m", 0.0, 1.0) + g["thickness_m"]
        base = request.get("arm_base_xy_m")
        base = np.array(vec(base, "arm_base_xy_m", 2)) if base is not None else None
        ws = request.get("workspace")
        if ws is not None:
            ws = (
                np.array(vec(ws.get("min_m"), "workspace.min_m", 2)),
                np.array(vec(ws.get("max_m"), "workspace.max_m", 2)),
            )
        open_w = finite(
            request.get("open_width_m", g["width_m"] + 2.0 * float(pd.GRIP_CLEARANCE)),
            "open_width_m",
            g["width_m"],
            1.0,
        )
        prefs = request.get("order_preference") or list(ORDERS)
        if any(o not in ORDERS for o in prefs):
            raise RequestRejected(
                "invalid_request", f"order_preference entries must be in {ORDERS}"
            )
        existing = []
        for e in request.get("existing") or []:
            p = e.get("pose") if isinstance(e, dict) else None
            if not isinstance(p, dict) or p.get("frame_id") != frame:
                raise RequestRejected(
                    "invalid_request", "existing tiles need an observed pose in frame_id"
                )
            existing.append((str(e.get("id")), geo.xy_facing(p)))
    except RequestRejected as exc:
        result["failure"] = {"code": exc.code, "detail": str(exc)}
        return result

    along = np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading))])
    xy = [start + k * pitch * along for k in range(n)]
    facing = [heading] * n
    slots = [
        {
            "index": k,
            "pose": geo.pose(frame, [*xy[k], table_z], geo.yaw_quat(math.radians(heading))),
        }
        for k in range(n)
    ]
    face_gap = pitch - g["thickness_m"]
    result["evidence"] = {
        "slots": slots,
        "center_pitch_m": pitch,
        "face_gap_m": face_gap,
        "facing_deg": heading,
        "open_width_m": open_w,
    }
    problems: list[str] = []

    # --- geometry: footprints apart from each other and from tiles already standing
    feet = [pl.footprint(xy[k], facing[k], size) for k in range(n)]
    ex_feet = [pl.footprint(xyf[0], xyf[1], size) for _, xyf in existing]
    for k in range(n - 1):
        if pl.poly_distance(feet[k], feet[k + 1]) < pl.OVERLAP_CLEARANCE:
            problems.append(f"slots {k} and {k + 1} overlap at {pitch * 1e3:.1f} mm centre pitch")
    for k in range(n):
        for j, ef in enumerate(ex_feet):
            if pl.poly_distance(feet[k], ef) < pl.OVERLAP_CLEARANCE:
                problems.append(f"slot {k} overlaps standing tile {existing[j][0]}")
        if ws is not None and not (np.all(feet[k] >= ws[0]) and np.all(feet[k] <= ws[1])):
            problems.append(f"slot {k} is outside the workspace")
    if problems:
        result["failure"] = {"code": "infeasible_geometry", "detail": "; ".join(problems)}
        return result

    # --- cascade physics, reported (enforced only on request)
    cascade = request.get("cascade") or {}
    if cascade.get("check") or cascade.get("require"):
        mu = float(cascade.get("friction", pl.DOMINO_FRICTION))
        crit = float(pl.critical_gap(size["height"], size["thickness"], round(pitch, 4), mu))
        margin = crit - face_gap
        result["evidence"]["cascade"] = {"critical_gap_m": crit, "margin_m": margin, "friction": mu}
        if cascade.get("require") and margin < float(cascade.get("margin_m", pl.TOPPLE_MARGIN_MIN)):
            result["failure"] = {
                "code": "infeasible_geometry",
                "detail": f"face gap {face_gap * 1e3:.1f} mm does not carry a cascade (critical {crit * 1e3:.1f} mm)",
            }
            return result

    # --- orders: each step's open hand clears the tiles standing then, and the arm does
    # not reach past a standing tile (other than the slot's own row neighbours)
    def sequence(name: str) -> list[int]:
        if name == "forward":
            return list(range(n))
        if name == "reverse":
            return list(range(n - 1, -1, -1))
        if base is None:
            return list(range(n - 1, -1, -1))
        return sorted(range(n), key=lambda k: (-float(np.linalg.norm(xy[k] - base)), -k))

    def check(order: list[int]) -> tuple[bool, str, list[dict]]:
        standing: list[int] = []
        steps = []
        for k in order:
            nbrs = [(xy[t], facing[t]) for t in standing] + [xyf for _, xyf in existing]
            clash = pd.finger_conflicts(xy[k], facing[k], nbrs, size, open_w)
            if clash:
                names = [
                    str(standing[c]) if c < len(standing) else existing[c - len(standing)][0]
                    for c in clash
                ]
                return False, f"slot {k}: open fingers reach standing tile(s) {names}", steps
            if base is not None:
                others = [t for t in standing if abs(t - k) > 1]
                hit = [t for t in others if pl.in_corridor(base, xy[k], xy[t])]
                hit += [eid for eid, xyf in existing if pl.in_corridor(base, xy[k], xyf[0])]
                if hit:
                    return (
                        False,
                        f"slot {k}: the arm would reach past standing tile(s) {hit}",
                        steps,
                    )
            if ws is not None and standing:
                near = min(standing, key=lambda t: float(np.linalg.norm(xy[t] - xy[k])))
                side = -1.0 if float((xy[near] - xy[k]) @ along) > 0 else 1.0
                lead = xy[k] + side * float(pd.LEAD_IN) * along
                if not (np.all(lead >= ws[0]) and np.all(lead <= ws[1])):
                    return (
                        False,
                        f"slot {k}: the lead-in from the free end leaves the workspace",
                        steps,
                    )
            steps.append({"slot_index": k, "neighbors": list(standing) + [e for e, _ in existing]})
            standing.append(k)
        return True, "", steps

    candidates = {}
    chosen = None
    for name in prefs:
        ok, why, steps = check(sequence(name))
        candidates[name] = {"feasible": ok, "reason": why, "order": sequence(name)}
        if ok and chosen is None:
            chosen = (name, steps)
    result["evidence"]["order_candidates"] = candidates
    if chosen is None:
        result["failure"] = {
            "code": "retreat_blocked",
            "detail": "no candidate order keeps the hand and arm clear: "
            + "; ".join(f"{k}: {v['reason']}" for k, v in candidates.items()),
        }
        return result
    result["evidence"]["order_name"] = chosen[0]
    result["evidence"]["build_order"] = [s["slot_index"] for s in chosen[1]]
    result["evidence"]["build_steps"] = chosen[1]
    result["status"] = "succeeded"
    result["unverified"] = ["plan only: no motion, no observation"]
    return result
