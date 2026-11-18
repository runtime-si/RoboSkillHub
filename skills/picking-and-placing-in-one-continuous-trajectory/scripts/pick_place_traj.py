# SPDX-License-Identifier: Apache-2.0
"""Pick and place as ONE timed joint trajectory: pre-grasp -> grasp -> lift -> carry -> pre-place -> place -> retreat -> home.

Straight (Cartesian) segments only near contact (the last ``hover`` above the grasp and the place), joint space
elsewhere, corners blended so the arm does not stop at via-points. The path is split only where physics needs
the hand still: closing around the object and releasing it. Each moving piece is timed for the shortest duration
under per-joint velocity / acceleration / jerk limits (forward/backward pass on the path parameter, jerk on the
30 Hz rows, from planning-time-optimal-dual-arm-motions-with-curobo), and the whole thing runs as one
``robot.execute_trajectory`` with ``gripper_close_s`` / ``gripper_release_s`` timed inside it.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

VMAX = [1.5, 1.5, 1.5, 4.5, 4.5, 4.5]
"""rad/s per joint (the servo's joint cap; 3x on the wrist)."""
AMAX = [15.0, 15.0, 15.0, 40.0, 40.0, 40.0]
"""rad/s^2 per joint."""
JMAX = [150.0, 150.0, 150.0, 400.0, 400.0, 400.0]
"""rad/s^3 per joint (10 x AMAX)."""
HZ = 30
"""Joint-stream row rate (stream_joints' highest); jerk is measured on rows at this rate."""
HOVER = 0.025
"""Pre-grasp / pre-place / lift / retreat height over the contact pose [m]."""
APEX = 0.06
"""Carry via-point height over the higher of grasp and place [m], so the joint-space carry clears the table."""
GRASP_DZ, PLACE_DZ = 0.004, 0.012
CLOSE_S, RELEASE_S = 0.35, 0.2
"""Stops for the jaws to close on / let go of the object [s]."""
LINE_PTS = 5
BLEND_RAD = 0.08
"""Corner blend radius in joint space [rad] (Gaussian smoothing of the free-space parts)."""


class Output(TypedDict):
    route: str
    detail: str
    arm_id: int
    duration_s: float
    max_tracking_error_rad: float | None
    tracks: list
    waypoints: list


def _v(p):
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}


def _retime(
    path: np.ndarray,
    vmax=VMAX,
    amax=AMAX,
    jmax=JMAX,
    ds: float | None = None,
    jerk_hz: float = HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Fastest timing of ``path`` (N, dof), at rest at both ends, under per-joint |v|, |a| (forward/backward
    pass on the path parameter) and |jerk| of the rows sampled at ``jerk_hz`` (the commands a
    joint stream sends; uniform slow-down). Returns times (M,) and positions (M, dof)."""
    path = np.asarray(path, float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s_knots = np.r_[0.0, np.cumsum(seg)]
    total = s_knots[-1]
    if total < 1e-9:
        return np.array([0.0]), path[:1].copy()
    n = max(50, int(total / (ds or 0.002)))
    s = np.linspace(0, total, n)
    q = np.stack([np.interp(s, s_knots, path[:, j]) for j in range(path.shape[1])], 1)
    h = s[1] - s[0]
    dq = np.gradient(q, h, axis=0)
    ddq = np.gradient(dq, h, axis=0)
    vmax_, amax_ = np.broadcast_to(vmax, dq.shape[1:]), np.broadcast_to(amax, dq.shape[1:])
    with np.errstate(divide="ignore"):
        sd_v = np.min(np.where(np.abs(dq) > 1e-9, vmax_ / np.abs(dq), np.inf), 1)
        sd_c = np.min(np.where(np.abs(ddq) > 1e-9, np.sqrt(amax_ / np.abs(ddq)), np.inf), 1)
    cap = np.minimum(sd_v, sd_c)

    nz = np.abs(dq) > 1e-9
    dq_safe = np.where(nz, dq, 1.0)

    def acc_bounds(i, sd):
        a1 = (amax_ - ddq[i] * sd * sd) / dq_safe[i]
        a2 = (-amax_ - ddq[i] * sd * sd) / dq_safe[i]
        m = nz[i]
        if not m.any():
            return -np.inf, np.inf
        return float(np.minimum(a1, a2)[m].max()), float(np.maximum(a1, a2)[m].min())

    sd = cap.copy()
    sd[0] = sd[-1] = 0.0
    for i in range(n - 1):
        _, hi = acc_bounds(i, sd[i])
        sd[i + 1] = min(sd[i + 1], math.sqrt(max(sd[i] ** 2 + 2 * max(hi, 0.0) * h, 0.0)))
    for i in range(n - 1, 0, -1):
        lo, _ = acc_bounds(i, sd[i])
        sd[i - 1] = min(sd[i - 1], math.sqrt(max(sd[i] ** 2 - 2 * min(lo, 0.0) * h, 0.0)))
    mid = 0.5 * (sd[1:] + sd[:-1])
    dt = np.where(mid > 1e-9, h / np.maximum(mid, 1e-9), 0.0)
    dt[mid <= 1e-9] = np.sqrt(2 * h / np.max(amax_)) * 2
    t = np.r_[0.0, np.cumsum(dt)]
    jmax_ = np.broadcast_to(jmax, dq.shape[1:])
    k = 1.0
    for _ in range(20):
        rows = sample(t * k, q, jerk_hz)
        if len(rows) < 4:
            break
        jerk = np.abs(np.diff(rows, 3, axis=0)).max(0) * jerk_hz**3
        r = float((jerk / jmax_).max())
        if r <= 1.0:
            break
        k *= max(1.01, r ** (1 / 3))
    return t * k, q


def retime(
    path: np.ndarray, vmax=VMAX, amax=AMAX, jmax=JMAX, jerk_hz: float = HZ
) -> tuple[np.ndarray, np.ndarray]:
    """Fastest timing of ``path`` (N, dof), at rest at both ends, within per-joint |v|, |a| and |jerk| limits.

    The jerk slow-down is uniform, so a sharper (faster) velocity profile can end slower; the timing is
    therefore also tried with the velocity limits scaled down, keeping the quickest."""
    best = None
    for f in (1.0, 0.75, 0.5, 0.35, 0.25):
        t, q = _retime(path, np.asarray(vmax, float) * f, amax, jmax, jerk_hz=jerk_hz)
        if best is None or t[-1] < best[0][-1]:
            best = (t, q)
    return best


def sample(t: np.ndarray, q: np.ndarray, hz: float = HZ) -> np.ndarray:
    """Rows of ``q`` at ``hz`` over ``t`` (last row = goal)."""
    tt = np.r_[np.arange(0.0, t[-1], 1.0 / hz), t[-1]]
    return np.stack([np.interp(tt, t, q[:, j]) for j in range(q.shape[1])], 1)


def build_path(
    pieces: list[tuple[np.ndarray, bool]], ds: float = 0.005, blend: float = BLEND_RAD
) -> np.ndarray:
    """One dense joint path through ``pieces`` [(q, straight)]: ``straight`` points belong to a Cartesian line
    near contact and are kept; free-space corners are blended by Gaussian smoothing (end points kept)."""
    qs = np.array([p[0] for p in pieces], float)
    keep = np.array([p[1] for p in pieces], float)
    dense, mask = [qs[:1]], [keep[:1]]
    for a in range(len(qs) - 1):
        n = max(1, int(np.ceil(np.linalg.norm(qs[a + 1] - qs[a]) / ds)))
        f = np.linspace(0, 1, n + 1)[1:, None]
        dense.append(qs[a] + f * (qs[a + 1] - qs[a]))
        mask.append(
            np.full(n, max(keep[a], keep[a + 1])) if keep[a] and keep[a + 1] else np.zeros(n)
        )
    P, K = np.vstack(dense), np.concatenate(mask)
    sig = max(1.0, blend / ds)
    w = np.exp(-0.5 * (np.arange(-int(3 * sig), int(3 * sig) + 1) / sig) ** 2)
    w /= w.sum()
    pad = len(w) // 2
    Pp = np.pad(P, ((pad, pad), (0, 0)), mode="edge")
    S = np.stack([np.convolve(Pp[:, j], w, mode="valid") for j in range(P.shape[1])], 1)
    Kp = np.pad(K, pad, mode="edge")
    m = 1.0 - np.clip(np.convolve(Kp, np.ones(len(w)) / len(w), mode="valid") * 3, 0, 1)
    m[K > 0] = 0.0
    out = m[:, None] * S + (1 - m[:, None]) * P
    out[0], out[-1] = P[0], P[-1]
    return out


def timeline(segments: list[np.ndarray], stops: list[float], vmax=VMAX, amax=AMAX, jmax=JMAX):
    """Time each segment (at rest at both ends), hold ``stops[i]`` after segment i. Returns (t, q, stop_starts)."""
    T, Q, starts, t0 = [], [], [], 0.0
    for k, seg in enumerate(segments):
        t, q = retime(seg, np.r_[vmax], np.r_[amax], np.r_[jmax])
        T.append(t + t0)
        Q.append(q)
        t0 += float(t[-1])
        if k < len(stops):
            starts.append(t0)
            t0 += stops[k]
            T.append(np.array([t0]))
            Q.append(q[-1:])
    return np.concatenate(T), np.vstack(Q), starts


def to_tracks(
    rows: np.ndarray, grip: np.ndarray, arm: str = "left", hz: int = HZ, chunk: int = 64
) -> dict[str, Any]:
    """The real server's ``stream_joints`` form: chunks of ``{arm: [[j1..j6 (hardware), grip_open], ...]}``."""
    hw = rows.copy()
    hw[:, 0] += math.pi / 2
    out = [
        {
            arm: [
                [*map(float, r), float(g)]
                for r, g in zip(hw[i : i + chunk], grip[i : i + chunk], strict=True)
            ]
        }
        for i in range(0, len(hw), chunk)
    ]
    jump = float(np.abs(np.diff(rows, axis=0)).max()) if len(rows) > 1 else 0.0
    return {
        "hz": hz,
        "mode": "abs",
        "chunks": out,
        "max_row_jump_rad": round(jump, 4),
        "row_jump_ok": jump <= 0.35,
    }


def run(
    ctx: NodeContext,
    object_obb: dict[str, Any],
    place: dict[str, float],
    arm_id: int = 0,
    vmax: list[float] | None = None,
    amax: list[float] | None = None,
    jmax: list[float] | None = None,
    hover: float = HOVER,
    waypoint_dt_s: float = 0.01,
    verify_grasp: bool = False,
    reach_tolerance_m: float = 0.01,
) -> Output:
    on = {"arm_id": int(arm_id)} if int(arm_id) else {}
    vmax, amax, jmax = vmax or VMAX, amax or AMAX, jmax or JMAX
    out = {
        "route": "planning_failed",
        "detail": "",
        "arm_id": int(arm_id),
        "duration_s": 0.0,
        "max_tracking_error_rad": None,
        "tracks": [],
        "waypoints": [],
    }
    c = np.array([object_obb["center"][k] for k in "xyz"])
    dst = np.array([place[k] for k in "xyz"])
    q = object_obb["orientation"]
    yaw = np.degrees(
        np.arctan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]), 1 - 2 * (q["y"] ** 2 + q["z"] ** 2))
    )
    heading = min((yaw + 90 * k for k in range(-4, 5)), key=abs)
    home = ctx.tool("robot.get_ee_pose", **on)
    home = home.get("pose", home)
    hp = np.array([home["position"][k] for k in "xyz"])
    frame = ctx.tool("robot.grasp_frame", close_heading_deg=heading, **on)

    def ik(p, rot):
        try:
            r = ctx.tool("robot.solve_ik", pose={"position": _v(p), "rotation": rot}, **on)
        except Exception:  # noqa: BLE001
            return None
        if (
            not isinstance(r, dict)
            or r.get("success") is False
            or r.get("within_servo_tolerance") is False
        ):
            return None
        if isinstance(r.get("position_error_m"), (int, float)) and r["position_error_m"] > 0.02:
            return None
        return np.array(r["joint_config"]["positions"][:6], float)

    g, pl = c + [0, 0, GRASP_DZ], dst + [0, 0, PLACE_DZ]
    apex_z = max(g[2], pl[2]) + APEX
    for rot in (frame["rotation"], frame["rotation_flipped"]):
        q_home = ik(hp, home["rotation"])
        down = [ik(g + [0, 0, hover * (1 - f)], rot) for f in np.linspace(0, 1, LINE_PTS)]
        up = [ik(g + [0, 0, hover * f], rot) for f in np.linspace(0, 1, LINE_PTS)]
        mid = (g + pl) / 2
        q_apex = ik([mid[0], mid[1], apex_z], rot)
        pdown = [ik(pl + [0, 0, hover * (1 - f)], rot) for f in np.linspace(0, 1, LINE_PTS)]
        pup = [ik(pl + [0, 0, hover * f], rot) for f in np.linspace(0, 1, LINE_PTS)]
        qs = [q_home, q_apex, *down, *up, *pdown, *pup]
        if all(x is not None for x in qs):
            steps = np.abs(np.diff(np.array(down + up), axis=0)).max()
            if steps < 0.5:
                break
    else:
        out["detail"] = (
            "no grasp rotation reaches the pick and place via-points without a joint flip"
        )
        return out
    S = lambda xs: [(x, True) for x in xs]  # noqa: E731
    seg1 = build_path([(q_home, False), *S(down)])
    seg2 = build_path([*S(up), (q_apex, False), *S(pdown)])
    seg3 = build_path([*S(pup), (q_home, False)])
    t, Q, (t_close, t_release) = timeline(
        [seg1, seg2, seg3], [CLOSE_S, RELEASE_S], vmax, amax, jmax
    )
    grip = np.where((t >= t_close) & (t < t_release), 0.0, 1.0)
    tt = np.r_[np.arange(0.0, t[-1], waypoint_dt_s), t[-1]]
    W = np.stack([np.interp(tt, t, Q[:, j]) for j in range(6)], 1)
    rows = np.stack(
        [np.interp(np.r_[np.arange(0.0, t[-1], 1 / HZ), t[-1]], t, Q[:, j]) for j in range(6)], 1
    )
    rgrip = np.where(
        (np.r_[np.arange(0.0, t[-1], 1 / HZ), t[-1]] >= t_close)
        & (np.r_[np.arange(0.0, t[-1], 1 / HZ), t[-1]] < t_release),
        0.0,
        1.0,
    )
    out["tracks"] = to_tracks(rows, rgrip, "left" if int(arm_id) == 0 else "right")
    idx = np.unique(np.linspace(0, len(t) - 1, 12).round().astype(int))
    out["waypoints"] = [
        {"t": float(t[i]), "positions": Q[i].tolist(), "gripper_open": float(grip[i])} for i in idx
    ]
    out["duration_s"] = float(t[-1])
    ctx.tool("robot.open_gripper", **on)
    wps = [{"positions": list(map(float, w))} for w in W]
    stats = {}
    if verify_grasp:
        k = int(np.searchsorted(tt, t_close + CLOSE_S))
        stats = (
            ctx.tool(
                "robot.execute_trajectory",
                trajectory={"waypoints": wps[: k + 1]},
                waypoint_dt_s=waypoint_dt_s,
                gripper_close_s=t_close,
                **on,
            )
            or {}
        )
        if float(ctx.tool("robot.get_gripper", **on)["position"]) < 0.04:
            ctx.tool("robot.open_gripper", **on)
            ctx.tool("robot.go_to_pose", pose=home, **on)
            out.update(route="grasp_missed", detail="the jaws closed on nothing")
            return out
        s2 = (
            ctx.tool(
                "robot.execute_trajectory",
                trajectory={"waypoints": wps[k:]},
                waypoint_dt_s=waypoint_dt_s,
                gripper_release_s=t_release - tt[k],
                **on,
            )
            or {}
        )
        stats["max_tracking_error_rad"] = max(
            stats.get("max_tracking_error_rad") or 0, s2.get("max_tracking_error_rad") or 0
        )
    else:
        stats = (
            ctx.tool(
                "robot.execute_trajectory",
                trajectory={"waypoints": wps},
                waypoint_dt_s=waypoint_dt_s,
                gripper_close_s=t_close,
                gripper_release_s=t_release,
                **on,
            )
            or {}
        )
    out["max_tracking_error_rad"] = stats.get("max_tracking_error_rad")
    end = ctx.tool("robot.get_ee_pose", **on)
    end = end.get("pose", end)
    gap = float(np.linalg.norm(np.array([end["position"][k] for k in "xyz"]) - hp))
    if gap > reach_tolerance_m:
        out.update(route="not_reached", detail=f"hand ended {1e3 * gap:.0f} mm from home")
        return out
    out["route"] = "placed"
    return out
