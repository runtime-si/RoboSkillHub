# Adapted from APEX-MR (https://github.com/intelligent-control-lab/APEX-MR), src/python/task_assignment.py
# TaskAssignmentSolver.solve_problem and parse_solution at commit 46a9448d9eaac5bd5973e0cd064623c5e7f5254e,
# MIT License, Copyright (c) 2025 Intelligent Control Lab.
# Changes: the MILP is APEX-MR's (each task gets one arm, one stock brick of its type and one grasp; a task that
# needs support gets a second arm, never the placing one, on the same grasp side; a non-reusable stock brick is
# used at most once; stock precedence by task index; the per-arm load in every window of n consecutive tasks is
# balanced with weight lambda = 1000), solved with SciPy's HiGHS (scipy.optimize.milp) instead of PuLP / CBC.
# APEX-MR reads its cost and support matrices from CSV files written by its ROS / MoveIt planner (planning time
# per arm, block and grasp, a large number when infeasible); here they are built in the skill from world poses:
# travel distance from the arm's base to the stock brick and on to the target, plus a cost per stud of grasp
# offset, with reach (a radius per arm, and optionally robot.solve_ik on the pick and place poses) and the
# fingers' room at the target (no brick placed earlier at that height in the jaws' way, the jaws wide enough)
# deciding what is feasible; infeasible combinations get no variable at all instead of a large cost. Grasp
# options are APEX-MR's press sides (jaws across the brick's x or y) times up to num_offsets offsets along the
# brick. Tasks come as world poses or as planning-brick-assembly-order's sequence on a plate. A support-only
# infeasibility is reported as its own exit.
"""Assign each brick of a build sequence to one of two (or more) arms - pick arm, stock brick, grasp and support arm - with APEX-MR's MILP."""

from __future__ import annotations

import json
import math
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

PRESETS: dict[str, dict[str, float]] = {
    "lego": {"stud_pitch": 0.008, "brick_height": 0.0096, "plate_unit": 0.0032},
    "duplo": {"stud_pitch": 0.016, "brick_height": 0.0192, "plate_unit": 0.0064},
}
"""Brick systems [m] (ICL / BrickSim LEGO units; DUPLO twice the pitch and height)."""

LAMBDA_BALANCE = 1000.0
"""APEX-MR's lambda_balance: per unit of load imbalance in a window, against costs in metres
of travel here (balance first, travel second, as in APEX-MR whose costs were seconds)."""

NUM_OFFSETS = 7
"""APEX-MR's num_offsets: grasp offsets per press side."""

OFFSET_COST = 0.01
"""Travel-equivalent cost of a grasp one pitch off the brick's centre [m]: centred grasps
first (APEX-MR's planner costs did the same through longer plans)."""

JAW_OPENING = 0.094
JAW_MARGIN = 0.004
FINGER_THICKNESS = 0.010
FINGER_WIDTH = 0.020
FINGER_MARGIN = 0.003
"""The YAM linear_4310 jaws and fingers: planning-brick-assembly-order's values [m]."""

SAME_LAYER_FRACTION = 0.5
"""A brick placed earlier blocks the fingers when its bottom is within this fraction of a
brick height of the target's (the fingers reach down the target's sides)."""

TIME_LIMIT_S = 30.0

ROUTES = ("assignment_ready", "infeasible", "needs_support_arm")


class Output(TypedDict):
    route: str
    detail: str
    assignment_json: str
    by_arm_json: str
    n_tasks: int
    n_support: int
    objective: float
    imbalance: float
    travel: float
    solve_s: float


def _f(v: Any, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    return float(v)


def _xyz(v: Any, name: str) -> np.ndarray:
    if isinstance(v, dict) and "position" in v:
        v = v["position"]
    if isinstance(v, dict):
        return np.array([_f(v[k], f"{name}.{k}") for k in ("x", "y", "z")])
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return np.array([_f(x, name) for x in v])
    raise ValueError(f"{name} must be a Vec3, got {v!r}")


def _studs(v: Any, name: str) -> tuple[int, int]:
    if isinstance(v, str):
        v = [int(s) for s in v.lower().split("x")]
    if (
        not isinstance(v, (list, tuple))
        or len(v) != 2
        or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in v)
    ):
        raise ValueError(f"{name}: studs must be two positive integers or 'AxB', got {v!r}")
    return int(v[0]), int(v[1])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def tasks_from_sequence(
    sequence: list[dict[str, Any]], origin: np.ndarray, yaw_deg: float, system: str
) -> list[dict[str, Any]]:
    """World tasks from planning-brick-assembly-order's sequence: cell (x, y) is the corner of
    its span on a plate whose cell (0, 0) corner is ``origin`` (top of the plate), turned by
    ``yaw_deg``; z from z_units plate units."""
    p, unit = PRESETS[system]["stud_pitch"], PRESETS[system]["plate_unit"]
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    out = []
    for b in sequence:
        cell = b.get("cell", b)
        sx, sy = int(b.get("span_x", 2)), int(b.get("span_y", 2))
        lx, ly = (cell["x"] + sx / 2) * p, (cell["y"] + sy / 2) * p
        z = int(b.get("z_units", 3 * (int(cell.get("layer", 1)) - 1))) * unit
        out.append(
            {
                "id": str(b.get("id", len(out))),
                "studs": [sx, sy],
                "position": [
                    origin[0] + c * lx - s * ly,
                    origin[1] + s * lx + c * ly,
                    origin[2] + z,
                ],
                "yaw_deg": yaw_deg,
                "height": int(b.get("height_units", 3)) * unit,
                "needs_support": bool(b.get("needs_support", False)),
            }
        )
    return out


class _Brick:
    def __init__(self, d: dict[str, Any], name: str, system: str) -> None:
        self.id = str(d.get("id", name))
        self.studs = _studs(d.get("studs", d.get("size", [2, 2])), name)
        self.p = _xyz(d["position"], f"{name}.position")
        self.yaw = _f(d.get("yaw_deg", 0.0), f"{name}.yaw_deg")
        self.h = _f(d.get("height", PRESETS[system]["brick_height"]), f"{name}.height")
        self.pitch = PRESETS[system]["stud_pitch"]
        a = math.radians(self.yaw)
        self.ax = (np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)]))
        self.half = (self.studs[0] * self.pitch / 2.0, self.studs[1] * self.pitch / 2.0)

    def size_key(self) -> tuple[int, int]:
        return tuple(sorted(self.studs))  # type: ignore[return-value]


def _rect_overlap(c1, ax1, h1, c2, ax2, h2) -> bool:
    """Separating-axis test of two rectangles (centre, unit axes, half sizes); touching
    edges do not overlap."""
    d = c2 - c1
    for axis in (*ax1, *ax2):
        r1 = sum(h1[i] * abs(float(axis @ ax1[i])) for i in range(2))
        r2 = sum(h2[i] * abs(float(axis @ ax2[i])) for i in range(2))
        if abs(float(d @ axis)) >= r1 + r2 - 1.0e-6:
            return False
    return True


def grasp_options(
    b: _Brick, num_offsets: int, jaw_opening: float, jaw_margin: float, finger_width: float
) -> list[tuple[int, float]]:
    """(side, offset along the other axis [m]): side 0 closes the jaws along the brick's x,
    side 1 along its y (APEX-MR's press sides); offsets in half-pitch steps from the centre
    (between and over the stud columns) while the finger stays on the brick, the
    num_offsets nearest the centre."""
    out = []
    for side in (0, 1):
        span = b.studs[side] * b.pitch
        if span + jaw_margin > jaw_opening:
            continue
        room = max(0.0, (b.studs[1 - side] * b.pitch - finger_width) / 2.0)
        step = b.pitch / 2.0
        k_max = int(math.floor(room / step + 1.0e-9))
        ks = sorted(range(-k_max, k_max + 1), key=abs)[: max(1, num_offsets)]
        out.extend((side, float(k) * step) for k in ks)
    return out


def fingers_clear(
    b: _Brick,
    side: int,
    offset: float,
    placed: list[_Brick],
    thickness: float,
    width: float,
    margin: float,
) -> bool:
    """No brick placed earlier at the target's height stands where a finger goes down
    beside it (planning-brick-assembly-order's finger boxes)."""
    close, along = b.ax[side], b.ax[1 - side]
    centre = b.p[:2] + along * offset
    for sign in (-1.0, 1.0):
        fc = centre + close * sign * (b.half[side] + margin + thickness / 2.0)
        fh = [0.0, 0.0]
        fh[side] = thickness / 2.0 + margin
        fh[1 - side] = width / 2.0
        for o in placed:
            if abs(o.p[2] - b.p[2]) > SAME_LAYER_FRACTION * min(o.h, b.h):
                continue
            if _rect_overlap(fc, b.ax, fh, o.p[:2], o.ax, o.half):
                return False
    return True


def _top_down(yaw_deg: float) -> dict[str, float]:
    """Tool z down, jaws closing along the world direction at ``yaw_deg``."""
    a = math.radians(yaw_deg) / 2.0
    return {"w": 0.0, "x": math.cos(a), "y": math.sin(a), "z": 0.0}


def run(
    ctx: NodeContext,
    arms: list[dict[str, Any]],
    tasks: list[dict[str, Any]] | None = None,
    sequence_json: str = "",
    plate_origin: dict[str, float] | None = None,
    plate_yaw_deg: float = 0.0,
    stock: list[dict[str, Any]] | None = None,
    stock_precedence: list[list[str]] | None = None,
    system: str = "duplo",
    num_offsets: int = NUM_OFFSETS,
    balance_weight: float = LAMBDA_BALANCE,
    offset_cost: float = OFFSET_COST,
    jaw_opening: float = JAW_OPENING,
    jaw_margin: float = JAW_MARGIN,
    finger_thickness: float = FINGER_THICKNESS,
    finger_width: float = FINGER_WIDTH,
    finger_margin: float = FINGER_MARGIN,
    check_ik: bool = False,
    hover: float = 0.05,
    time_limit_s: float = TIME_LIMIT_S,
) -> Output:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    if system not in PRESETS:
        raise ValueError(f"system must be one of {sorted(PRESETS)}, got {system!r}")
    if not arms:
        raise ValueError("arms must list at least one {arm_id, base, reach_m}")
    if tasks is None:
        if not sequence_json or plate_origin is None:
            raise ValueError("give tasks, or sequence_json with plate_origin")
        tasks = tasks_from_sequence(
            json.loads(sequence_json),
            _xyz(plate_origin, "plate_origin"),
            float(plate_yaw_deg),
            system,
        )
    out: Output = {
        "route": "infeasible",
        "detail": "",
        "assignment_json": "[]",
        "by_arm_json": "{}",
        "n_tasks": len(tasks),
        "n_support": 0,
        "objective": 0.0,
        "imbalance": 0.0,
        "travel": 0.0,
        "solve_s": 0.0,
    }

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        return out

    arm_ids = [int(a.get("arm_id", i)) for i, a in enumerate(arms)]
    bases = [_xyz(a["base"], f"arms[{i}].base") for i, a in enumerate(arms)]
    reach = [float(a.get("reach_m", math.inf)) for a in arms]
    T = [_Brick(t, f"tasks[{j}]", system) for j, t in enumerate(tasks)]
    needs = [bool(t.get("needs_support", False)) for t in tasks]
    sup_pos = [
        _xyz(t["support_position"], f"tasks[{j}].support_position")
        if t.get("support_position") is not None
        else T[j].p - np.array([0.0, 0.0, T[j].h])
        for j, t in enumerate(tasks)
    ]
    S = [_Brick(s, f"stock[{k}]", system) for k, s in enumerate(stock or [])]
    reusable = [bool(s.get("reusable", False)) for s in (stock or [])]
    n, m, p = len(arms), len(T), len(S)
    if m == 0:
        return finish("assignment_ready", "no tasks")
    if n > m:
        return finish("infeasible", f"{n} arms for {m} tasks: APEX-MR's balance needs m >= n")
    out["n_support"] = sum(needs)

    ik_cache: dict[tuple, bool] = {}

    def reachable(i: int, pt: np.ndarray, yaw: float) -> bool:
        if float(np.linalg.norm(pt[:2] - bases[i][:2])) > reach[i]:
            return False
        if not check_ik:
            return True
        key = (i, *np.round(pt, 4), round(yaw % 180.0, 1))
        if key not in ik_cache:
            on_arm = {"arm_id": arm_ids[i]} if arm_ids[i] else {}
            pose = {"position": _vec(pt + [0.0, 0.0, hover]), "rotation": _top_down(yaw)}
            try:
                r = ctx.tool("robot.solve_ik", pose=pose, **on_arm)
                ok = not (
                    isinstance(r, dict)
                    and (r.get("success") is False or r.get("within_servo_tolerance") is False)
                )
            except Exception as exc:
                ok = "unreachable" not in str(exc).lower()
            ik_cache[key] = ok
        return ik_cache[key]

    # Variables: x(i, j, t, g) for feasible combinations, y(i, j, side) for support.
    xs: list[tuple[int, int, int, int, float]] = []  # (i, j, t, side, offset) with cost below
    xcost: list[float] = []
    why: list[str] = []
    for j, b in enumerate(T):
        placed = T[:j]
        opts = [
            (side, off)
            for side, off in grasp_options(b, num_offsets, jaw_opening, jaw_margin, finger_width)
            if fingers_clear(b, side, off, placed, finger_thickness, finger_width, finger_margin)
        ]
        if not opts:
            why.append(
                f"task {b.id}: no grasp - the jaws do not open enough or placed neighbours block the fingers"
            )
            continue
        sources = [k for k in range(p) if S[k].size_key() == b.size_key()] if p else [-1]
        if not sources:
            why.append(f"task {b.id}: no stock brick of size {b.studs[0]}x{b.studs[1]}")
            continue
        found = False
        for i in range(n):
            for side, off in opts:
                yaw = b.yaw + (0.0 if side == 0 else 90.0)
                if not reachable(i, b.p, yaw):
                    continue
                for k in sources:
                    if k >= 0:
                        sk = S[k]
                        if not reachable(i, sk.p, sk.yaw):
                            continue
                        travel = float(np.linalg.norm(sk.p - bases[i]) + np.linalg.norm(b.p - sk.p))
                    else:
                        travel = 2.0 * float(np.linalg.norm(b.p - bases[i]))
                    xs.append((i, j, k, side, off))
                    xcost.append(travel + offset_cost * abs(off) / b.pitch)
                    found = True
        if not found:
            why.append(f"task {b.id}: out of every arm's reach (target or its stock)")
    if why:
        return finish("infeasible", "; ".join(why))

    def build_and_solve(with_support: bool):
        ys = []
        ycost = []
        if with_support:
            for j in range(m):
                if not needs[j]:
                    continue
                for i in range(n):
                    if reachable(i, sup_pos[j], T[j].yaw):
                        for side in (0, 1):
                            ys.append((i, j, side))
                            ycost.append(float(np.linalg.norm(sup_pos[j] - bases[i])))
        W = m - n + 1
        nx_, ny_ = len(xs), len(ys)
        iz = nx_ + ny_
        izmax = iz + n * W
        izmin = izmax + W
        nv = izmin + W
        c = np.zeros(nv)
        c[:nx_] = xcost
        c[nx_ : nx_ + ny_] = ycost
        c[izmax : izmax + W] = balance_weight
        c[izmin : izmin + W] = -balance_weight
        rows: list[tuple[dict[int, float], float, float]] = []
        by_j: dict[int, list[int]] = {}
        by_ij: dict[tuple[int, int], list[int]] = {}
        for v, (i, j, _k, _side, _o) in enumerate(xs):
            by_j.setdefault(j, []).append(v)
            by_ij.setdefault((i, j), []).append(v)
        y_by_j: dict[int, list[int]] = {}
        y_by_ij: dict[tuple[int, int], list[int]] = {}
        for v, (i, j, _side) in enumerate(ys):
            y_by_j.setdefault(j, []).append(nx_ + v)
            y_by_ij.setdefault((i, j), []).append(nx_ + v)
        # Each task: one arm, one brick, one grasp.
        for j in range(m):
            rows.append(({v: 1.0 for v in by_j.get(j, [])}, 1.0, 1.0))
        # An arm does not both place and support one task.
        for i in range(n):
            for j in range(m):
                coef = {v: 1.0 for v in by_ij.get((i, j), []) + y_by_ij.get((i, j), [])}
                if coef:
                    rows.append((coef, -np.inf, 1.0))
        if with_support:
            for j in range(m):
                need = 1.0 if needs[j] else 0.0
                rows.append(({v: 1.0 for v in y_by_j.get(j, [])}, need, need))
                if needs[j]:
                    # The support arm takes the placing grasp's side.
                    for side in (0, 1):
                        coef = {v: 1.0 for v in by_j[j] if xs[v][3] == side}
                        for v in y_by_j.get(j, []):
                            if ys[v - nx_][2] == side:
                                coef[v] = coef.get(v, 0.0) - 1.0
                        rows.append((coef, 0.0, 0.0))
        # A stock brick that is not a reusable station is used once.
        for k in range(p):
            if not reusable[k]:
                coef = {v: 1.0 for v, x in enumerate(xs) if x[2] == k}
                if coef:
                    rows.append((coef, -np.inf, 1.0))
        # Stock precedence: t1 is picked no later than t2 (APEX-MR's (m - j) weights).
        idx = {s.id: k for k, s in enumerate(S)}
        for a_id, b_id in stock_precedence or []:
            if a_id not in idx or b_id not in idx:
                raise ValueError(f"stock_precedence names unknown stock {a_id!r} / {b_id!r}")
            ka, kb = idx[a_id], idx[b_id]
            coef: dict[int, float] = {}
            for v, x in enumerate(xs):
                if x[2] == ka:
                    coef[v] = coef.get(v, 0.0) + (m - x[1])
                if x[2] == kb:
                    coef[v] = coef.get(v, 0.0) - (m - x[1])
            rows.append((coef, 0.0, np.inf))
        # Load balance in every window of n consecutive tasks.
        for i in range(n):
            for w in range(W):
                coef = {iz + i * W + w: -1.0}
                for j in range(w, w + n):
                    for v in by_ij.get((i, j), []) + y_by_ij.get((i, j), []):
                        coef[v] = 1.0
                rows.append((coef, 0.0, 0.0))
                rows.append(({izmin + w: 1.0, iz + i * W + w: -1.0}, -np.inf, 0.0))
                rows.append(({izmax + w: 1.0, iz + i * W + w: -1.0}, 0.0, np.inf))
        A = lil_matrix((len(rows), nv))
        lb = np.empty(len(rows))
        ub = np.empty(len(rows))
        for r, (coef, lo, hi) in enumerate(rows):
            for v, a in coef.items():
                A[r, v] = a
            lb[r], ub[r] = lo, hi
        integrality = np.ones(nv)
        lo_b = np.zeros(nv)
        hi_b = np.ones(nv)
        hi_b[iz:] = n  # window loads
        t0 = time.perf_counter()
        res = milp(
            c,
            constraints=LinearConstraint(A.tocsr(), lb, ub),
            integrality=integrality,
            bounds=Bounds(lo_b, hi_b),
            options={"time_limit": float(time_limit_s)},
        )
        dt = time.perf_counter() - t0
        return res, ys, dt, (iz, izmax, izmin, W)

    res, ys, dt, (iz, izmax, izmin, W) = build_and_solve(True)
    out["solve_s"] = round(dt, 3)
    if res.x is None:
        relaxed, _, dt2, _ = build_and_solve(False)
        out["solve_s"] = round(dt + dt2, 3)
        if relaxed.x is not None and any(needs):
            lacking = []
            for j in range(m):
                if needs[j]:
                    placers = {xs[v][0] for v in range(len(xs)) if xs[v][1] == j}
                    helpers = {i for i in range(n) if reachable(i, sup_pos[j], T[j].yaw)}
                    if not any(h != pl for h in helpers for pl in placers):
                        lacking.append(T[j].id)
            return finish(
                "needs_support_arm",
                "the bricks can be placed but not supported: "
                + (
                    f"no second arm reaches the support of {', '.join(lacking)}"
                    if lacking
                    else "the support duty cannot be fitted with the rest"
                ),
            )
        return finish("infeasible", f"no assignment satisfies the constraints ({res.message})")
    x = np.round(res.x).astype(int)
    nx_ = len(xs)
    assignment = []
    by_arm: dict[str, list[str]] = {str(a): [] for a in arm_ids}
    travel = 0.0
    for j in range(m):
        chosen = [v for v in range(nx_) if xs[v][1] == j and x[v] == 1]
        i, _, k, side, off = xs[chosen[0]]
        travel += xcost[chosen[0]]
        sup = [
            ys[v - nx_][0] for v in range(nx_, nx_ + len(ys)) if ys[v - nx_][1] == j and x[v] == 1
        ]
        b = T[j]
        rec = {
            "task": b.id,
            "arm_id": arm_ids[i],
            "stock": S[k].id if k >= 0 else None,
            "close_along": "x" if side == 0 else "y",
            "grasp_offset_m": round(off, 5),
            "grasp_yaw_deg": round((b.yaw + (0.0 if side == 0 else 90.0)) % 180.0, 2),
            "support_arm_id": arm_ids[sup[0]] if sup else None,
        }
        assignment.append(rec)
        by_arm[str(arm_ids[i])].append(b.id)
    imbalance = float(sum(res.x[izmax : izmax + W]) - sum(res.x[izmin : izmin + W]))
    out["assignment_json"] = json.dumps(assignment)
    out["by_arm_json"] = json.dumps(by_arm)
    out["objective"] = float(res.fun)
    out["imbalance"] = round(imbalance, 3)
    out["travel"] = round(travel, 4)
    counts = ", ".join(f"arm {a}: {len(v)}" for a, v in by_arm.items())
    n_sup = sum(1 for a in assignment if a["support_arm_id"] is not None)
    return finish(
        "assignment_ready",
        f"{m} tasks ({counts}; {n_sup} supported), window imbalance {imbalance:.0f}, "
        f"travel {travel:.2f} m, solved in {dt:.2f} s",
    )
