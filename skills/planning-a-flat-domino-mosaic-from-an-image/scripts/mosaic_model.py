# Adapted from DominoArtSolver (https://github.com/ArnoldF/DominoArtSolver) at commit
# 3f9e260ea97a6ad659784fc6931bbb64d9196eba: src/preprocessing.py (find_valid_row_col_combination,
# create_domino_tiles, generate_placements, compute_brightness_domino_diff,
# compute_diff_placement_domino), src/image_processing.py (ImageReader.get_field_brightness),
# src/domino_model.py (DominoModel) and the orientation rule of src/plotting.py
# (create_solution_image). MIT License, Copyright (c) 2025 Florian Arnold Thurm; the license is
# LICENSE.dominoartsolver beside SKILL.md.
# Changes: the pip range is a parameter (upstream hard-codes double-nine; double-six is the default
# here) and so are the fields per set (2 x the tiles in a set instead of 110); dark pips on light
# tiles invert the brightness target (upstream: light pips on dark tiles); a constant image is
# refused instead of dividing by a zero range; a given grid may hold fewer cells than the sets, then
# each tile is used at most `sets` times instead of exactly; the model is the same integer program
# but built from per-cell placement lists (upstream scans every placement for every cell); CBC is
# called through either PuLP API (2.x/3.x with its bundled CBC, or 4.x with cbcbox) with a time
# limit, and SciPy's HiGHS (scipy.optimize.milp) solves the same model when PuLP is not installed;
# the solution is checked (every cell once, usage per tile) before it is used; each tile's end
# order is chosen on the untruncated squared error, ties to the lower value first (upstream's
# plotting compared the truncated costs and sent ties to the reversed order).
"""DominoArtSolver's integer program: which domino covers which pair of grid cells, and how."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class SolverUnavailable(RuntimeError):
    """No MILP solver could be loaded (PuLP + CBC, or SciPy's HiGHS)."""


def domino_tiles(max_pips: int) -> list[tuple[int, int]]:
    """All tiles of one double-`max_pips` set, (low, high), in upstream's id order."""
    return [(i, j) for i in range(max_pips + 1) for j in range(i, max_pips + 1)]


def grid_for_sets(n_fields: int, aspect_ratio: float) -> tuple[int, int]:
    """Rows and columns with rows x cols = n_fields closest to the image's aspect (upstream)."""
    options = []
    for cols in range(1, n_fields + 1):
        rows = n_fields / cols
        if int(rows) == rows:
            options.append((int(rows), cols))
    if not options:
        raise ValueError("No combination of rows and cols could be found.")
    return sorted(options, key=lambda rc: abs((rc[1] / rc[0]) - aspect_ratio))[0]


def field_brightness(grey: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, int, int]:
    """Mean brightness of each grid field (upstream: floor-sized fields, the remainder unused)."""
    h, w = grey.shape
    fh, fw = h // rows, w // cols
    crop = grey[: fh * rows, : fw * cols].astype(np.float64)
    means = crop.reshape(rows, fh, cols, fw).mean(axis=(1, 3))
    return means, fh, fw


def target_values(means: np.ndarray, max_pips: int, dark_pips: bool) -> np.ndarray:
    """Field brightness -> the pip count a field wants, -0.5 .. max_pips + 0.5 (upstream's scale).

    Upstream (light pips on dark tiles): ``10 * (b - min) / ptp - 0.5`` for double-nine. Dark pips
    on light tiles want more pips where the image is darker: ``max_pips`` minus that.
    """
    span = float(np.ptp(means))
    light = (max_pips + 1) * (means - means.min()) / span - 0.5
    return max_pips - light if dark_pips else light


def placements(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Every horizontal and vertical pair of neighbouring cells, in upstream's order."""
    out = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                out.append(((r, c), (r, c + 1)))
            if r + 1 < rows:
                out.append(((r, c), (r + 1, c)))
    return out


def cost_matrix(
    target: np.ndarray,
    tiles: list[tuple[int, int]],
    places: list[tuple[tuple[int, int], tuple[int, int]]],
    truncate: bool,
) -> np.ndarray:
    """[tile, placement] squared error of the better end order; upstream truncates it to an int."""
    first = np.array([target[a] for a, _ in places])
    second = np.array([target[b] for _, b in places])
    lo = np.array([t[0] for t in tiles], dtype=np.float64)[:, None]
    hi = np.array([t[1] for t in tiles], dtype=np.float64)[:, None]
    normal = (first[None] - lo) ** 2 + (second[None] - hi) ** 2
    reverse = (first[None] - hi) ** 2 + (second[None] - lo) ** 2
    if truncate:
        normal, reverse = np.trunc(normal), np.trunc(reverse)
    return np.minimum(normal, reverse)


def lower_value_first(target: np.ndarray, tile: tuple[int, int], place) -> bool:
    """Whether the tile's low value goes on the placement's first cell (untruncated error)."""
    (a, b), (lo, hi) = place, tile
    normal = (target[a] - lo) ** 2 + (target[b] - hi) ** 2
    reverse = (target[a] - hi) ** 2 + (target[b] - lo) ** 2
    return bool(normal <= reverse)


@dataclass
class Solution:
    chosen: list[tuple[int, int]]  # (tile id, placement id)
    status: str  # optimal | feasible | no_solution | infeasible
    solver: str
    objective: float | None
    bound: float | None
    seconds: float
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class DominoModel:
    """Binary x[tile, placement]: every cell covered once, every tile used `sets` times
    (or at most `sets` times when the grid holds fewer cells than the sets), minimum total cost."""

    def __init__(
        self,
        cost: np.ndarray,
        places: list[tuple[tuple[int, int], tuple[int, int]]],
        rows: int,
        cols: int,
        sets: int,
        exact_usage: bool,
    ):
        self.cost = cost
        self.places = places
        self.rows, self.cols = rows, cols
        self.sets = sets
        self.exact_usage = exact_usage
        self.n_tiles, self.n_places = cost.shape
        self.cell_places: list[list[int]] = [[] for _ in range(rows * cols)]
        for p, ((r1, c1), (r2, c2)) in enumerate(places):
            self.cell_places[r1 * cols + c1].append(p)
            self.cell_places[r2 * cols + c2].append(p)

    def solve(self, solver: str, time_limit: float) -> Solution:
        if solver == "cbc":
            return self._solve_cbc(time_limit)
        if solver == "highs":
            return self._solve_highs(time_limit)
        try:
            return self._solve_cbc(time_limit)
        except SolverUnavailable as cbc_missing:
            try:
                return self._solve_highs(time_limit)
            except SolverUnavailable as highs_missing:
                raise SolverUnavailable(f"{cbc_missing}; {highs_missing}") from highs_missing

    def check(self, chosen: list[tuple[int, int]]) -> str:
        """'' when `chosen` covers every cell once and uses each tile within its count."""
        cover = np.zeros(self.rows * self.cols, dtype=int)
        use = np.zeros(self.n_tiles, dtype=int)
        for t, p in chosen:
            (r1, c1), (r2, c2) = self.places[p]
            cover[r1 * self.cols + c1] += 1
            cover[r2 * self.cols + c2] += 1
            use[t] += 1
        if (cover != 1).any():
            return f"{int((cover != 1).sum())} cells not covered exactly once"
        if self.exact_usage and (use != self.sets).any():
            return "a tile not used exactly `sets` times"
        if (use > self.sets).any():
            return "a tile used more than `sets` times"
        return ""

    # -- CBC through PuLP -------------------------------------------------------------------

    def _solve_cbc(self, time_limit: float) -> Solution:
        try:
            import pulp
        except ImportError as exc:
            raise SolverUnavailable(f"PuLP not installed ({exc})") from exc
        cmd_cls = getattr(pulp, "PULP_CBC_CMD", None) or getattr(pulp, "COIN_CMD", None)
        if cmd_cls is None:
            raise SolverUnavailable("this PuLP has neither PULP_CBC_CMD nor COIN_CMD")
        cmd = cmd_cls(msg=False, timeLimit=float(time_limit))
        if not cmd.available():
            raise SolverUnavailable(f"PuLP's {cmd_cls.__name__} finds no CBC binary")

        prob = pulp.LpProblem("DominoPlacement", pulp.LpMinimize)
        keys = [(t, p) for t in range(self.n_tiles) for p in range(self.n_places)]
        if hasattr(prob, "add_variable_dicts"):  # PuLP 4
            x = prob.add_variable_dicts("domino_at_place", keys, cat="Binary")
        else:
            x = pulp.LpVariable.dicts("domino_at_place", keys, cat="Binary")
        prob.setObjective(pulp.lpSum(float(self.cost[k]) * x[k] for k in keys))
        for cell, ps in enumerate(self.cell_places):
            prob.addConstraint(
                pulp.lpSum(x[t, p] for p in ps for t in range(self.n_tiles)) == 1,
                f"cover_field_once_{cell // self.cols}_{cell % self.cols}",
            )
        for t in range(self.n_tiles):
            usage = pulp.lpSum(x[t, p] for p in range(self.n_places))
            prob.addConstraint(
                (usage == self.sets) if self.exact_usage else (usage <= self.sets),
                f"use_domino_{t}",
            )

        t0 = time.perf_counter()
        result = prob.solve(cmd)
        seconds = time.perf_counter() - t0
        version = getattr(pulp, "__version__", "?")
        objective = bound = None
        if hasattr(result, "has_solution"):  # PuLP 4: LpSolveStats
            name = getattr(result.status, "name", str(result.status))
            objective, bound = result.objective, result.best_bound
            if not result.has_solution:
                status = "infeasible" if name == "Infeasible" else "no_solution"
            elif name == "Optimal" or (
                name == "GapLimit"
                and objective is not None
                and bound is not None
                and abs(objective - bound) <= 1e-4 * max(1.0, abs(objective))
            ):
                status = "optimal"
            else:
                status = "feasible"
            detail = f"CBC via PuLP {version} ({cmd_cls.__name__}): {name}"
        else:  # PuLP 2 / 3: int status, sol_status 1 optimal, 2 integer feasible
            sol_status = getattr(prob, "sol_status", None)
            if result == -1:
                status = "infeasible"
            elif result == 1 and sol_status in (1, None):
                status = "optimal"
            elif sol_status == 2 or result == 1:
                status = "feasible"
            else:
                status = "no_solution"
            objective = pulp.value(prob.objective)
            detail = (
                f"CBC via PuLP {version} ({cmd_cls.__name__}): "
                f"{pulp.LpStatus.get(result, result)}, sol_status {sol_status}"
            )
        chosen = []
        if status in ("optimal", "feasible"):
            for k in keys:
                v = pulp.value(x[k])
                if v is not None and v > 0.5:
                    chosen.append(k)
        return Solution(chosen, status, "cbc", objective, bound, seconds, detail)

    # -- HiGHS through SciPy ----------------------------------------------------------------

    def _solve_highs(self, time_limit: float) -> Solution:
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import coo_matrix
        except ImportError as exc:
            raise SolverUnavailable(f"scipy.optimize.milp not available ({exc})") from exc
        n_t, n_p = self.n_tiles, self.n_places
        n_cells = self.rows * self.cols
        rows, cols = [], []
        for cell, ps in enumerate(self.cell_places):
            for p in ps:
                for t in range(n_t):
                    rows.append(cell)
                    cols.append(t * n_p + p)
        for t in range(n_t):
            for p in range(n_p):
                rows.append(n_cells + t)
                cols.append(t * n_p + p)
        a = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_cells + n_t, n_t * n_p)).tocsr()
        lb = np.concatenate([np.ones(n_cells), np.full(n_t, self.sets if self.exact_usage else 0)])
        ub = np.concatenate([np.ones(n_cells), np.full(n_t, self.sets)])
        t0 = time.perf_counter()
        res = milp(
            c=self.cost.reshape(-1),
            constraints=LinearConstraint(a, lb, ub),
            integrality=np.ones(n_t * n_p),
            bounds=Bounds(0, 1),
            options={"time_limit": float(time_limit), "disp": False},
        )
        seconds = time.perf_counter() - t0
        if res.status == 0:
            status = "optimal"
        elif res.status == 2:
            status = "infeasible"
        elif res.x is not None:
            status = "feasible"
        else:
            status = "no_solution"
        chosen = []
        if res.x is not None and status in ("optimal", "feasible"):
            for k in np.flatnonzero(res.x > 0.5):
                chosen.append((int(k // n_p), int(k % n_p)))
        bound = getattr(res, "mip_dual_bound", None)
        return Solution(
            chosen,
            status,
            "highs",
            float(res.fun) if res.fun is not None else None,
            float(bound) if bound is not None else None,
            seconds,
            f"HiGHS via scipy.optimize.milp: {res.message}",
        )
