"""Delay one of two timed arm trajectories where their collision spheres would come too close; paths unchanged."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

MARGIN = 0.01
"""Clearance required between the two arms' sphere sets [m] (on top of the spheres' own 15 mm fit margin)."""

CHECK_DT = 0.05
"""Time step at which the arms are checked [s]."""

PAUSE_STEP = 0.25
"""A pause grows in steps of this [s]."""

MAX_PAUSE = 6.0
"""Longest total pause inserted before giving up [s]."""

ROUTES = ("no_conflict", "pauses_inserted", "unresolved")

_MODEL: dict[str, Any] = {}


def _q2r(q: list[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _model(path: str | None = None) -> dict[str, Any]:
    key = path or ""
    if key not in _MODEL:
        p = (
            Path(path)
            if path
            else Path(__file__).resolve().parents[1] / "assets" / "yam_spheres.json"
        )
        _MODEL[key] = json.loads(p.read_text())
    return _MODEL[key]


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def spheres(
    q: np.ndarray, base: np.ndarray, model: dict[str, Any], closed: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """World centres [N, 3] and radii [N] of one arm's spheres at joint angles q (6) with base pose (4x4)."""
    chain = model["chain"]
    frames = {"base": base}
    order = ["link1", "link2", "link3", "link4", "link5"]
    for i, name in enumerate(order):
        c = chain[name]
        t = np.eye(4)
        t[:3, :3] = _q2r(c["quat_wxyz"]) @ _rz(float(q[i]))
        t[:3, 3] = c["pos"]
        frames[name] = frames[c["parent"]] @ t
    # link6 is not in the chain: joint6 turns about link5's z at its origin (approximation).
    t6 = np.eye(4)
    t6[:3, :3] = _rz(float(q[5]))
    frames["link6"] = frames["link5"] @ t6
    cs, rs = [], []
    sets = [(n, model["spheres"][n]) for n in ("base", *order)]
    sets.append(
        ("link6", model["spheres_link6_fingers_closed" if closed else "spheres_link6_fingers_open"])
    )
    for name, ss in sets:
        f = frames[name]
        for s in ss:
            cs.append(f[:3, :3] @ np.asarray(s["center"]) + f[:3, 3])
            rs.append(float(s["radius"]))
    return np.asarray(cs), np.asarray(rs)


def clearance(qa, qb, base_a, base_b, model) -> float:
    ca, ra = spheres(np.asarray(qa, float), base_a, model)
    cb, rb = spheres(np.asarray(qb, float), base_b, model)
    d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2) - ra[:, None] - rb[None, :]
    return float(d.min())


def _base(model: dict[str, Any], arm_id: int, given: Any) -> np.ndarray:
    t = np.eye(4)
    if given is not None:
        g = given.get("pose", given) if isinstance(given, dict) else given
        if isinstance(g, dict):
            t[:3, :3] = _q2r([g["rotation"][k] for k in "wxyz"])
            t[:3, 3] = [g["position"][k] for k in "xyz"]
            return t
        return np.asarray(g, float).reshape(4, 4)
    b = model["arm_bases"]["left" if int(arm_id) == 0 else "right"]
    t[:3, :3] = _q2r(b["quat_wxyz"])
    t[:3, 3] = b["pos"]
    return t


def _at(times: np.ndarray, joints: np.ndarray, t: float) -> np.ndarray:
    if t <= times[0]:
        return joints[0]
    if t >= times[-1]:
        return joints[-1]
    i = int(np.searchsorted(times, t)) - 1
    a = (t - times[i]) / max(times[i + 1] - times[i], 1e-9)
    return joints[i] + a * (joints[i + 1] - joints[i])


def _first_conflict(ta, ja, tb, jb, base_a, base_b, model, margin, dt) -> tuple[float, float]:
    """(time of the first sample under margin or -1, the smallest clearance seen)."""
    end = max(ta[-1], tb[-1])
    worst = math.inf
    first = -1.0
    for t in np.arange(min(ta[0], tb[0]), end + 1e-9, dt):
        c = clearance(_at(ta, ja, t), _at(tb, jb, t), base_a, base_b, model)
        worst = min(worst, c)
        if c < margin and first < 0:
            first = float(t)
    return first, worst


def _parse(traj: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(traj.get("times"), dtype=float)
    joints = np.asarray(traj.get("joints") or traj.get("positions"), dtype=float)
    if joints.ndim != 2 or joints.shape[1] < 6 or len(times) != len(joints) or len(times) < 2:
        raise ValueError(f"{name}: needs 'times' (N) and 'joints' (N x 6), N >= 2")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{name}: times must increase")
    return times, joints[:, :6]


def run(
    ctx: NodeContext,
    trajectory_a: dict[str, Any],
    trajectory_b: dict[str, Any],
    arm_a: int = 0,
    arm_b: int = 1,
    base_a: Any = None,
    base_b: Any = None,
    margin: float = MARGIN,
    check_dt: float = CHECK_DT,
    max_pause: float = MAX_PAUSE,
    delay: str = "shorter",
    spheres_path: str | None = None,
) -> Output:
    if delay not in ("shorter", "a", "b"):
        raise ValueError(f"delay must be 'shorter', 'a' or 'b', got {delay!r}")
    model = _model(spheres_path)
    ta, ja = _parse(trajectory_a, "trajectory_a")
    tb, jb = _parse(trajectory_b, "trajectory_b")
    ba, bb = _base(model, arm_a, base_a), _base(model, arm_b, base_b)
    out: Output = {
        "route": "unresolved",
        "detail": "",
        "trajectory_a": {"times": ta.tolist(), "joints": ja.tolist()},
        "trajectory_b": {"times": tb.tolist(), "joints": jb.tolist()},
        "pauses": [],
        "delayed": "",
        "min_clearance_before": 0.0,
        "min_clearance_after": 0.0,
        "added_time": 0.0,
    }
    first, worst = _first_conflict(ta, ja, tb, jb, ba, bb, model, margin, check_dt)
    out["min_clearance_before"] = out["min_clearance_after"] = round(worst, 4)
    if first < 0:
        out["route"], out["detail"] = (
            "no_conflict",
            f"clearance never under {margin} m (min {worst:.3f} m)",
        )
        return out
    if delay == "shorter":
        first_pick = "a" if (ta[-1] - ta[0]) <= (tb[-1] - tb[0]) else "b"
        kw = dict(
            arm_a=arm_a,
            arm_b=arm_b,
            base_a=base_a,
            base_b=base_b,
            margin=margin,
            check_dt=check_dt,
            max_pause=max_pause,
            spheres_path=spheres_path,
        )
        res = run(ctx, trajectory_a, trajectory_b, delay=first_pick, **kw)
        if res["route"] == "unresolved":
            other = run(
                ctx, trajectory_a, trajectory_b, delay="b" if first_pick == "a" else "a", **kw
            )
            if other["route"] == "pauses_inserted":
                other["detail"] += " (the shorter arm could not wait clear; the other one waits)"
                return other
        return res
    late = delay
    out["delayed"] = late
    t_l, j_l = (ta.copy(), ja.copy()) if late == "a" else (tb.copy(), jb.copy())
    t_o, j_o = (tb, jb) if late == "a" else (ta, ja)
    total = 0.0
    for _ in range(int(max_pause / PAUSE_STEP) + 1):
        pair = (t_l, j_l, t_o, j_o) if late == "a" else (t_o, j_o, t_l, j_l)
        first, worst = _first_conflict(
            pair[0], pair[1], pair[2], pair[3], ba, bb, model, margin, check_dt
        )
        if first < 0:
            break
        if total + PAUSE_STEP > max_pause + 1e-9:
            out["route"] = "unresolved"
            out["detail"] = (
                f"still {worst:.3f} m apart at t={first:.2f} s after {total:.2f} s of pauses (max_pause)"
            )
            out["min_clearance_after"] = round(worst, 4)
            return out
        # Pause the delayed arm at the last waypoint before the conflict; if the conflict now falls while
        # it holds there, that waypoint itself clashes - hold one waypoint earlier.
        i = max(0, int(np.searchsorted(t_l, first, side="right")) - 1)
        while (
            i > 0
            and any(p["after_index"] == i for p in out["pauses"])
            and first <= t_l[i + 1] + 1e-9
        ):
            i -= 1
        t_l = np.r_[t_l[: i + 1], t_l[i] + PAUSE_STEP, t_l[i + 1 :] + PAUSE_STEP]
        j_l = np.vstack([j_l[: i + 1], j_l[i : i + 1], j_l[i + 1 :]])
        total += PAUSE_STEP
        if out["pauses"] and out["pauses"][-1]["after_index"] == i:
            out["pauses"][-1]["duration"] = round(out["pauses"][-1]["duration"] + PAUSE_STEP, 3)
        else:
            out["pauses"].append(
                {"after_index": i, "at_time": round(float(t_l[i]), 3), "duration": PAUSE_STEP}
            )
    pair = (t_l, j_l, t_o, j_o) if late == "a" else (t_o, j_o, t_l, j_l)
    first, worst = _first_conflict(
        pair[0], pair[1], pair[2], pair[3], ba, bb, model, margin, check_dt
    )
    out["min_clearance_after"] = round(worst, 4)
    key = "trajectory_a" if late == "a" else "trajectory_b"
    out[key] = {"times": t_l.tolist(), "joints": j_l.tolist()}
    out["added_time"] = round(total, 3)
    if first >= 0:
        out["route"], out["detail"] = "unresolved", f"still under margin at t={first:.2f} s"
        return out
    out["route"] = "pauses_inserted"
    out["detail"] = (
        f"arm {late} paused {total:.2f} s in {len(out['pauses'])} place(s); min clearance {worst:.3f} m"
    )
    return out


class Output(TypedDict):
    route: str
    detail: str
    trajectory_a: dict[str, Any]
    trajectory_b: dict[str, Any]
    pauses: list[dict[str, Any]]
    delayed: str
    min_clearance_before: float
    min_clearance_after: float
    added_time: float
