# SPDX-License-Identifier: Apache-2.0
"""Summarise graph episodes (``episode.json``) written by ``domino_graph_baselines.py``,
per run directory (graph + params, or fault), from the files alone (works on partial runs).

    python examples/tasks/assembly_eval/domino_graph_summarize.py RUNS_DIR [--seeds 0-9] [--out x.json]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def summarize_dir(d: Path, seeds: set[int] | None) -> dict | None:
    eps = []
    for p in sorted(d.glob("seed_*/episode.json")):
        e = json.loads(p.read_text())
        if seeds is None or e["seed"] in seeds:
            eps.append(e)
    if not eps:
        return None
    n = len(eps)
    place_fail, dis = collections.Counter(), []
    for e in eps:
        for p in e["independent"]["placements"]:
            for f in p.get("failure_types") or []:
                place_fail[f] += 1
            for u in p.get("unverified_reasons") or []:
                place_fail["unverified:" + u.split(":")[0]] += 1
        for q in e["skill_vs_evaluator"]:
            if not q["agree"]:
                dis.append(
                    {
                        "seed": e["seed"],
                        "skill": f"{q['skill_status']}/{q['skill_failure_type']}",
                        "evaluator": q["evaluator_status"],
                        "evaluator_failures": q["evaluator_failure_types"],
                    }
                )
    return {
        "dir": d.name,
        "episodes": n,
        "seeds": sorted(e["seed"] for e in eps),
        "graph_succeeded": f"{sum(e['status'] == 'succeeded' for e in eps)}/{n}",
        "graph_status": dict(collections.Counter(e["status"] for e in eps)),
        "graph_failure_types": dict(
            collections.Counter(
                e.get("failure_type") or "none" for e in eps if e["status"] != "succeeded"
            )
        ),
        "independent_task_success": f"{sum(bool(e['task_success']) for e in eps)}/{n}",
        "independent_placement_failure_types": dict(place_fail),
        "place_steps_compared": sum(len(e["skill_vs_evaluator"]) for e in eps),
        "disagreements": dis,
        "interventions": sum(e.get("interventions", 0) for e in eps),
        "mean_steps": sum(len(e["steps"]) for e in eps) / n,
        "mean_wall_s": sum(e["wall_s"] for e in eps) / n,
        "mean_sim_s": sum(e["independent"]["sim_time_s"] for e in eps) / n,
        "privileged_inputs": any(e.get("privileged_inputs") for e in eps),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--seeds")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    seeds = None
    if args.seeds:
        a, _, b = args.seeds.partition("-")
        seeds = set(range(int(a), int(b or a) + 1))
    rows = []
    for root in args.roots:
        for d in sorted({p.parent.parent for p in root.rglob("seed_*/episode.json")}):
            row = summarize_dir(d, seeds)
            if row:
                row["root"] = str(d.relative_to(root.parent))
                rows.append(row)
                print(
                    f"{row['root'][-70:]:70s} graph {row['graph_succeeded']:6s} {row['graph_failure_types']} "
                    f"indep {row['independent_task_success']} {row['independent_placement_failure_types']} "
                    f"disagree {len(row['disagreements'])}"
                )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
