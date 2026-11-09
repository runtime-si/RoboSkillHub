# SPDX-License-Identifier: Apache-2.0
"""One graph episode on the domino physics station, printed step by step (needs the
integration branch: ``roboskillhub.tasks.graphs`` and the registered K).

    PYTHONPATH=src python examples/tasks/assembly_eval/domino_graph_smoke.py --physics mujoco --seed 0
"""

from __future__ import annotations

import argparse
import time

from roboskillhub.tasks import graphs, load_registry
from roboskillhub.tasks.assembly_eval.domino_world import physics_domino_world
from roboskillhub.tasks.baselines import StationEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", default="mujoco")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--recovery", action="store_true")
    ap.add_argument("--fault")
    args = ap.parse_args()
    registry, _ = load_registry()
    holder = {}

    def world(seed):
        holder["w"] = physics_domino_world(
            seed, physics=args.physics, slots=args.slots, fault=args.fault
        )
        return holder["w"]

    graph = graphs.domino_line(
        recovery=args.recovery, attempts=2 if args.recovery else 1, slots=args.slots
    )
    t = time.time()
    ep = StationEnv(world, registry)(graph, args.seed)
    print(
        "episode",
        ep["status"],
        ep.get("failure_type"),
        ep.get("reason"),
        f"wall {time.time() - t:.1f}s",
    )
    for s in ep["steps"]:
        r = s.get("result", {})
        print(
            f"  {s['node']:20s} {s['status']:10s} {r.get('failure_type')}  "
            f"{(r.get('error') or {}).get('message', '')[:160]}"
        )
    ind = holder["w"]["independent"]()
    print("independent layout", ind["layout"]["status"])
    for k, v in ind["layout"].get("slots", {}).items():
        print(
            f"  {k} {v['status']} {v.get('piece')} err={v.get('position_error_m', 0) * 1e3:.2f}mm "
            f"tilt={v.get('max_tilt_deg', 0):.2f}"
        )
    for p in ind["placements"]:
        print(
            f"  release {p['target_id']} -> slot {p['slot_index']}: {p['status']} {p['failure_types']} "
            f"{p['unverified_reasons']}"
        )
    print(
        "sim",
        round(ind["sim_time_s"], 1),
        [(e["event"], round(e["sim_s"], 2)) for e in ind["events"]],
    )


if __name__ == "__main__":
    main()
