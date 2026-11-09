"""Measures simulating-a-domino-layout-before-building against planning-a-domino-layout and
the twin results the hub's domino skills report. Not collected by pytest; needs the
physics engine importable:

    .venv/bin/python tests/bundles/_domino_layout_sim_eval.py [straight|arcs|jitter|friction|all]
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

from gap.testing import FakeContext

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str((ROOT / rel).parent))
    spec.loader.exec_module(mod)
    return mod


sim = _load(
    "simulate_layout",
    "skills/simulating-a-domino-layout-before-building/scripts/simulate_layout.py",
)
plan = _load("plan_layout", "skills/planning-a-domino-layout/scripts/plan_layout.py")

TABLE_Z = 0.0
WS = {"bounds": {"min": [-1.0, -1.0, -0.1], "max": [1.0, 1.0, 0.5]}}


def line(n: int, pitch: float) -> list[list[float]]:
    return [[k * pitch, 0.3, 0.0] for k in range(n)]


def straight() -> None:
    pitches = [
        0.014,
        0.015,
        0.016,
        0.017,
        0.018,
        0.020,
        0.030,
        0.037,
        0.048,
        0.055,
        0.056,
        0.057,
        0.058,
        0.060,
        0.072,
    ]
    for trigger in ("push", "kick"):
        for dt in (sim.TIMESTEP, sim.TIMESTEP / 2):
            row = []
            for p in pitches:
                r = sim.run(
                    FakeContext(),
                    slots=line(12, p),
                    table_z=TABLE_Z,
                    trials=0,
                    trigger=trigger,
                    timestep=dt,
                )
                tag = {"cascades": "R", "stalls": f"S{r['stall_at']}", "push_failed": "P"}.get(
                    r["route"], r["route"]
                )
                row.append(f"{p * 1e3:.0f}:{tag}")
            print(
                f"STRAIGHT trigger {trigger} dt {dt * 1e3:.2f} ms (12 tiles; R runs, Sk stops at slot k, P no trigger): "
                + " ".join(row)
            )


def arcs() -> None:
    """The planner's arcs at its auto pitch and at 1.0 h: its verdict against the rollout's."""
    for pitch in ("auto", 0.048):
        for radius in (0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.25):
            path = {
                "type": "arc",
                "center": [0.0, 0.3],
                "radius": radius,
                "start_deg": -90,
                "sweep_deg": 180,
            }
            pr = plan.run(
                FakeContext({"robot.describe_workspace": WS}),
                path=path,
                table_z=TABLE_Z,
                pitch=pitch,
            )
            slots = pr.get("slots") or []
            if len(slots) < 2:
                print(
                    f"ARC r {radius * 1e3:.0f} mm pitch {pitch}: planner {pr['route']} with no slots ({pr['detail'][:80]})"
                )
                continue
            t0 = time.perf_counter()
            r = sim.run(FakeContext(), slots=slots, table_z=TABLE_Z, trials=20, seed=1)
            print(
                f"ARC r {radius * 1e3:.0f} mm pitch {pr['pitch'] * 1e3:.1f} mm, {len(slots)} slots, max turn {pr['max_turn_deg']:.1f} deg: "
                f"planner {pr['route']} | sim {r['route']} stall {r['stall_at']} success {r['success_rate']:.2f} "
                f"({time.perf_counter() - t0:.1f} s)"
            )


def jitter() -> None:
    for sigma, ysig in ((0.002, 3.0), (0.0054, 5.0)):
        row = []
        for p in (0.020, 0.025, 0.030, 0.037, 0.042, 0.048, 0.052, 0.055):
            r = sim.run(
                FakeContext(),
                slots=line(12, p),
                table_z=TABLE_Z,
                trials=40,
                seed=2,
                position_sigma=sigma,
                yaw_sigma_deg=ysig,
            )
            row.append(f"{p * 1e3:.0f}:{r['success_rate']:.2f}({r['route'][0]})")
        print(
            f"JITTER sigma {sigma * 1e3:.1f} mm / {ysig:.0f} deg, 12 tiles, 40 rollouts, success rate by pitch: "
            + " ".join(row)
        )


def friction() -> None:
    for table_mu in (0.3, 0.5, 0.8):
        for tile_mu in (0.15, 0.25, 0.4):
            crit = None
            for p_mm in range(50, 64):
                r = sim.run(
                    FakeContext(),
                    slots=line(12, p_mm / 1000),
                    table_z=TABLE_Z,
                    trials=0,
                    table_friction=table_mu,
                    tile_friction=tile_mu,
                )
                if r["route"] != "cascades":
                    break
                crit = p_mm
            fast = sim.run(
                FakeContext(),
                slots=line(12, 0.037),
                table_z=TABLE_Z,
                trials=0,
                table_friction=table_mu,
                tile_friction=tile_mu,
            )
            print(
                f"FRICTION table {table_mu} tile {tile_mu}: widest running pitch {crit} mm; 37 mm pitch {fast['route']} {fast['wave_speed']:.2f} m/s"
            )


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (
        ("straight", straight),
        ("arcs", arcs),
        ("jitter", jitter),
        ("friction", friction),
    ):
        if what in (name, "all"):
            fn()
