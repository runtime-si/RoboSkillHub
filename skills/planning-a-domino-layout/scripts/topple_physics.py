# Two 2D models of a toppling domino line, used by plan_layout.py.
#
# 1. GapChain is adapted from domino-fall-simulation
#    (https://github.com/maxencelaunayquerre/domino-fall-simulation, commit
#    f6147f635b2a4a3f3f7ca0589284f45dda6aa309, src/simulation.py Simulation /
#    SimulationConfig, src/analysis.py run_until_failure, scripts/spacing_threshold.py
#    create_gap / test_s_value / find_critical_s), MIT License, Copyright (c) 2026
#    Maxence Launay Querre (license: ../LICENSE.domino-fall-simulation).
#    Changes: SI units and g = 9.81 (upstream: h = 1, g = 10); a leading tile that
#    reaches the table without touching the next one, or a stall, is a failure
#    (upstream's run_until_failure only caught solver errors, so a gap the wave
#    could not reach read as crossed once the ground was hit); the critical-gap
#    scan is a bisection on the gap at a given pitch instead of upstream's fixed
#    linspace; the solver tolerances are explicit; no plotting.
#
# 2. StackedLine is adapted from domino-simulation
#    (https://github.com/eigenmatt/domino-simulation, commit
#    09169dbd605e78afcb71d122f0c113eba2896161, domino-simulation.py: calc_theta,
#    calc_omega, solve_alpha_newton, solve_collision_newton, step_simulation_rk4,
#    run_simulation), by Matthew Chapman, licensed CC BY 4.0
#    (https://creativecommons.org/licenses/by/4.0/); the article is
#    https://zmatt.net/domino-physics/.
#    Changes: module globals became parameters of a class; RK4 always; the loop
#    detects a stall (the leading tile turns back before reaching the next) and
#    returns the toppling times instead of plotting.
#
# Both assume the tiles do not slide on the table and that a falling tile stays
# leaning on the next one after it hits it.
"""Toppling physics of a straight domino line: does a wave cross a gap, and how fast it runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

G = 9.81
"""Gravity [m/s^2] (upstream GapChain used 10 in normalised units; StackedLine 9.8)."""


# ----------------------------------------------------------------------------- GapChain (MIT)


@dataclass
class GapChain:
    """A line of tiles with per-gap spacing (face-to-face gaps ``gaps``, n-1 of them).

    Angles are negative when falling forward (upstream's convention)."""

    heights: np.ndarray
    gaps: np.ndarray
    thickness: np.ndarray
    masses: np.ndarray
    mu: float = 0.0
    thetas: np.ndarray = field(init=False)
    dot_thetas: np.ndarray = field(init=False)
    last_active: int = field(init=False, default=-1)
    ground_hit: bool = field(init=False, default=False)
    time: float = field(init=False, default=0.0)
    hit_times: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.heights = np.asarray(self.heights, dtype=np.float64)
        self.gaps = np.asarray(self.gaps, dtype=np.float64)
        self.thickness = np.asarray(self.thickness, dtype=np.float64)
        self.masses = np.asarray(self.masses, dtype=np.float64)
        n = len(self.heights)
        self.thetas = np.zeros(n)
        self.dot_thetas = np.zeros(n)
        # Moment of inertia about the front bottom edge.
        self.inertia = self.masses * (self.heights**2 + self.thickness**2) / 3.0

    @classmethod
    def uniform(
        cls, n: int, height: float, gap: float, thickness: float, mu: float, mass: float = 0.01
    ) -> GapChain:
        return cls(
            heights=np.full(n, height),
            gaps=np.full(n - 1, gap),
            thickness=np.full(n, thickness),
            masses=np.full(n, mass),
            mu=mu,
        )

    @property
    def n(self) -> int:
        return len(self.heights)

    def start(self, omega0: float) -> None:
        self.thetas[0] = 0.0
        self.dot_thetas[0] = -abs(float(omega0))
        self.last_active = 0

    # Moment arms of the contact force between tile i and tile i+1.
    def _b(self, i: int, th: np.ndarray) -> float:
        return self.heights[i] * math.cos(th[i] - th[i + 1])

    def _a(self, i: int, th: np.ndarray) -> float:
        return self._b(i, th) + (self.gaps[i] + self.thickness[i + 1]) * math.sin(th[i + 1])

    def _beta(self, i: int, th: np.ndarray) -> float:
        return self._b(i, th) + self.mu * self.heights[i] * math.sin(th[i + 1] - th[i])

    def _alpha(self, i: int, th: np.ndarray) -> float:
        return self._a(i, th) - self.mu * self.thickness[i + 1]

    def accelerations(self, th: np.ndarray, om: np.ndarray) -> np.ndarray:
        n = len(th)
        gamma = (
            G
            * self.masses[:n]
            * (self.heights[:n] * np.sin(th) + self.thickness[:n] * np.cos(th))
            / 2.0
        )
        a = np.zeros((2 * n - 1, 2 * n - 1))
        b = np.zeros(2 * n - 1)
        for i in range(n):
            a[i, i] = self.inertia[i]
            b[i] = gamma[i]
            if i > 0:
                a[i, n + i - 1] = self._alpha(i - 1, th)
            if i < n - 1:
                a[i, n + i] = -self._beta(i, th)
        for i in range(n - 1):
            b[n + i] = -(self.gaps[i] + self.thickness[i + 1]) * om[i + 1] ** 2 * math.cos(
                th[i + 1]
            ) + self.heights[i] * (om[i + 1] - om[i]) ** 2 * math.sin(th[i + 1] - th[i])
            a[n + i, i] = -self._b(i, th)
            a[n + i, i + 1] = self._a(i, th)
        return np.linalg.solve(a, b)[:n]

    def _rhs(self, _t: float, y: np.ndarray) -> np.ndarray:
        n = len(y) // 2
        return np.concatenate([y[n:], self.accelerations(y[:n], y[n:])])

    def advance(self, horizon: float = 5.0) -> str:
        """Run to the next event: "hit" (the leading tile reached the next one),
        "ground" (it reached the table first), "stall" (it turned back), "end"."""
        k = self.last_active
        lead_h = self.heights[k]
        reach = self.gaps[k] / lead_h if k < self.n - 1 else 2.0

        def hit_next(_t: float, y: np.ndarray) -> float:
            return y[k] + math.asin(reach) + 1e-12 if reach < 1.0 else 1.0

        hit_next.terminal = True  # type: ignore[attr-defined]

        def hit_ground(_t: float, y: np.ndarray) -> float:
            return y[k] + math.pi / 2.0

        hit_ground.terminal = True  # type: ignore[attr-defined]

        def turned_back(_t: float, y: np.ndarray) -> float:
            return y[len(y) // 2 + k]  # the leading tile's angular speed crosses 0

        turned_back.terminal = True  # type: ignore[attr-defined]
        turned_back.direction = 1.0  # type: ignore[attr-defined]

        sol = solve_ivp(
            self._rhs,
            (0.0, horizon),
            [*self.thetas[: k + 1], *self.dot_thetas[: k + 1]],
            events=[hit_next, hit_ground, turned_back],
            rtol=1e-8,
            atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        y = sol.y[:, -1]
        self.time += float(sol.t[-1])
        self.thetas[: k + 1] = y[: k + 1]
        self.dot_thetas[: k + 1] = y[k + 1 :]
        if len(sol.t_events[0]):
            if k >= self.n - 1:
                return "end"
            self._collide()
            self.hit_times.append(self.time)
            return "hit"
        if len(sol.t_events[1]):
            self.ground_hit = True
            return "ground"
        return "stall"

    def _collide(self) -> None:
        k = self.last_active
        a = [self._a(i, self.thetas) for i in range(k + 1)]
        b = [self._b(i, self.thetas) for i in range(k + 1)]
        al = [self._alpha(i, self.thetas) for i in range(k + 1)]
        be = [self._beta(i, self.thetas) for i in range(k + 1)]
        r = [0.0] * (k + 2)
        rr = [0.0] * (k + 2)
        r[k + 1] = rr[k + 1] = 1.0
        for j in range(k, -1, -1):
            r[j] = r[j + 1] * a[j] / b[j]
            rr[j] = rr[j + 1] * al[j] / be[j]
        s = sum(self.inertia[i] * r[i] * rr[i] for i in range(k + 1))
        nxt = b[k] / a[k] * s / (self.inertia[k + 1] + s) * self.dot_thetas[k]
        for i in range(k + 1, -1, -1):
            self.dot_thetas[i] = r[i] * nxt
        self.last_active = k + 1


def run_until_failure(chain: GapChain, omega0: float) -> int | None:
    """The index of the last tile that fell before the wave stopped, or None when
    every tile fell (upstream's run_until_failure; a ground hit or stall fails)."""
    chain.start(omega0)
    for _ in range(chain.n - 1):
        try:
            event = chain.advance()
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return chain.last_active
        if event != "hit":
            return chain.last_active
    return None


def tipping_speed(height: float, thickness: float) -> float:
    """The least angular speed [rad/s] that carries a lone upright tile over its
    tipping point (its centre of mass over the front edge)."""
    rise = (math.hypot(height, thickness) - height) / 2.0
    radius_sq = (height**2 + thickness**2) / 3.0
    return math.sqrt(2.0 * G * rise / radius_sq)


def chain_crosses(
    height: float,
    thickness: float,
    pitch: float,
    mu: float,
    gap: float | None = None,
    run_up: int = 20,
    after: int = 4,
    omega0: float | None = None,
) -> bool:
    """Whether a wave that has run ``run_up`` tiles at ``pitch`` crosses one
    face-to-face ``gap`` (None: no gap, the uniform line) and topples ``after``
    more tiles (upstream create_gap: 20 tiles at the default spacing, then the gap)."""
    face = pitch - thickness
    if face <= 0.0:
        return False
    n = run_up + 1 + after
    chain = GapChain.uniform(n, height, face, thickness, mu)
    if gap is not None:
        chain.gaps[run_up] = gap
    w0 = omega0 if omega0 is not None else 1.2 * tipping_speed(height, thickness)
    return run_until_failure(chain, w0) is None


UPSTREAM_PUSH = 10.0 * math.sqrt(1.0 / 10.0)
"""Upstream's start, 10 rad/s on a tile of height 1 under g = 10, made dimensionless
(omega * sqrt(h / g)): a firm push that sets the wave going; after the run-up the wave
runs at its settled speed whatever the push was."""


def critical_gap(
    height: float,
    thickness: float,
    pitch: float,
    mu: float,
    run_up: int = 20,
    tol: float = 1e-4,
) -> float:
    """The widest face-to-face gap [m] a wave running at ``pitch`` still crosses
    (upstream find_critical_s, by bisection; upstream's push, scaled). 0.0 when the
    line itself stops."""
    w0 = UPSTREAM_PUSH * math.sqrt(G / height)
    if not chain_crosses(height, thickness, pitch, mu, None, run_up, omega0=w0):
        return 0.0
    lo, hi = 0.0, height
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if chain_crosses(height, thickness, pitch, mu, mid, run_up, omega0=w0):
            lo = mid
        else:
            hi = mid
    return lo


# ----------------------------------------------------------------------------- StackedLine (CC BY 4.0)


class StackedLine:
    """Matthew Chapman's no-slip stacked-wave model of a uniform line (pitch
    ``pitch`` = gap + thickness), with domino-domino friction ``mu``."""

    def __init__(
        self,
        height: float,
        width: float,
        thickness: float,
        pitch: float,
        mu: float,
        density: float = 1040.0,
        dt: float = 1e-4,
    ) -> None:
        self.L, self.w, self.d, self.lam, self.mu = height, width, thickness, pitch, mu
        self.dt = dt
        self.m = height * width * thickness * density
        self.W = self.m * G
        self.rw = math.hypot(height, thickness) / 2.0
        self.phi = math.atan(thickness / height)
        self.I = self.m * (height**2 + thickness**2) / 3.0
        reach = (pitch - thickness) / height
        self.collision_angle = math.asin(reach) if 0.0 < reach < 1.0 else math.nan

    def _theta(self, n: int, theta0: float) -> np.ndarray:
        th = np.empty(n)
        th[0] = theta0
        for i in range(1, n):
            th[i] = th[i - 1] + math.asin((self.lam * math.cos(th[i - 1]) - self.d) / self.L)
        return th

    def _omega(self, n: int, th: np.ndarray, omega0: float) -> np.ndarray:
        om = np.empty(n)
        om[0] = omega0
        for i in range(1, n):
            om[i] = om[i - 1] * (
                1 - (self.lam / self.L) * math.sin(th[i - 1]) / math.cos(th[i] - th[i - 1])
            )
        return om

    def _r1(self, th: np.ndarray, i: int) -> float:
        return (
            self.L * math.cos(th[i + 1]) / math.cos(th[i])
            - self.d * math.tan(th[i])
            - self.mu * self.d
        )

    def _r0(self, th: np.ndarray, i: int) -> float:
        return self.L * (math.cos(th[i] - th[i - 1]) + self.mu * math.sin(th[i] - th[i - 1]))

    def alpha(self, n: int, theta0: float, omega0: float) -> float:
        """The leading tile's angular acceleration (solve_alpha_newton)."""
        th = self._theta(n, theta0)
        om = self._omega(n, th, omega0)
        lam, L = self.lam, self.L
        n_coeff = n_const = 0.0
        a_coeff, a_const = 1.0, 0.0
        for i in range(n):
            if i > 0:
                sin0 = math.sin(th[i - 1])
                cos1 = math.cos(th[i])
                cos10 = math.cos(th[i] - th[i - 1])
                sin10 = math.sin(th[i] - th[i - 1])
                ratio = 1 - (lam / L) * sin0 / cos10
                a_ratio = (lam / L) * (cos1 + sin0 * sin10 * ratio) / (cos10**2)
                a_coeff *= ratio
                a_const *= ratio
                a_const -= a_ratio * om[i - 1] * om[i - 1]
            r0 = 0.0 if i == 0 else self._r0(th, i)
            r1 = L if i == n - 1 else self._r1(th, i)
            n_coeff = n_coeff * r0 / r1 + (self.I / r1) * a_coeff
            n_const = n_const * r0 / r1 + (self.I / r1) * a_const
            n_const -= (self.rw / r1) * self.W * math.sin(th[i] - self.phi)
        return -n_const / n_coeff

    def collide(self, n: int, theta0: float, omega0: float) -> float:
        """The new leading tile's angular speed after it is hit (solve_collision_newton)."""
        old_th = self._theta(n, theta0)
        old_om = self._omega(n, old_th, omega0)
        th = self._theta(n + 1, 0.0)
        rel = self._omega(n + 1, th, 1.0)
        i_coeff = i_const = 0.0
        for i in range(n + 1):
            d_coeff = rel[i]
            d_const = 0.0 if i == 0 else -old_om[i - 1]
            r0 = 0.0 if i == 0 else self._r0(th, i)
            r1 = self.L if i == n else self._r1(th, i)
            i_coeff = i_coeff * r0 / r1 + (self.I / r1) * d_coeff
            i_const = i_const * r0 / r1 + (self.I / r1) * d_const
        return -i_const / i_coeff

    def run(self, tiles: int, omega0: float, max_time: float = 5.0) -> list[float]:
        """Times [s] at which tiles 1.. were hit; shorter than tiles-1 when the
        wave stalled (or the tiles never touch)."""
        if not math.isfinite(self.collision_angle):
            return []
        n, theta, omega, t, dt = 1, 0.0, float(omega0), 0.0, self.dt
        hits: list[float] = []
        while t < max_time:
            k0 = omega
            a0 = self.alpha(n, theta, omega)
            k1 = omega + 0.5 * dt * a0
            a1 = self.alpha(n, theta + 0.5 * dt * k0, k1)
            k2 = omega + 0.5 * dt * a1
            a2 = self.alpha(n, theta + 0.5 * dt * k1, k2)
            k3 = omega + dt * a2
            a3 = self.alpha(n, theta + dt * k2, k3)
            theta += dt * (k0 + 2 * k1 + 2 * k2 + k3) / 6.0
            omega += dt * (a0 + 2 * a1 + 2 * a2 + a3) / 6.0
            t += dt
            if theta >= self.collision_angle:
                hits.append(t)
                if len(hits) >= tiles - 1:
                    break
                omega = self.collide(n, theta, omega)
                n += 1
                theta = 0.0
            elif omega <= 0.0:
                break  # turned back before reaching the next tile
        return hits

    def frequency(self, tiles: int = 30, omega0: float | None = None, skip: int = 10) -> float:
        """Tiles toppled per second once the wave has settled (the hits after
        ``skip``); 0.0 when it stalls."""
        w0 = omega0 if omega0 is not None else 5.0 * math.pi
        hits = self.run(tiles, w0)
        if len(hits) < tiles - 1 or len(hits) <= skip + 1:
            return 0.0
        return (len(hits) - 1 - skip) / (hits[-1] - hits[skip])
