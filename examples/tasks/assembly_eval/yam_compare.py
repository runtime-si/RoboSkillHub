# SPDX-License-Identifier: Apache-2.0
"""Side-by-side table of ``yam_graph_baselines.py`` runs (YAM vs floating gripper, G vs G').

python examples/tasks/assembly_eval/yam_compare.py RUN_DIR... --summary s.json > compare.md
(each RUN_DIR holds baselines.json / negatives.json; the label is the directory name)
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load(d: Path) -> dict:
    out = {}
    for name in ("baselines", "negatives"):
        p = d / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


def stop_counts(episodes: list[dict]) -> str:
    c = collections.Counter(
        f"{e['stopped_at']['node']}:{e['stopped_at']['failure_type']}"
        for e in episodes
        if e.get("stopped_at")
    )
    return ", ".join(f"{k} x{v}" for k, v in sorted(c.items())) or "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args()
    runs = {d.name: load(d) for d in args.runs}
    summary: dict = {"baselines": {}, "negatives": {}}
    print("## Baselines (3 slots; train 1000-1005, test 0-9)\n")
    print(
        "| run | baseline | graph | independent | disagreements | stopped at | params | mean wall s / sim s |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for label, r in runs.items():
        b = (r.get("baselines") or {}).get("baselines", {}).get("baselines", {})
        for name, row in b.items():
            ind = row["independent"]
            eps = row["episodes"]
            params = {
                k: v for k, v in row["params"].items() if k in ("recovery", "attempts", "speed_m_s")
            }
            print(
                f"| {label} | {name} | {row['graph_success']} | {ind['independent_task_success']} | "
                f"{len(ind['disagreements'])} / {ind['place_steps_compared']} | {stop_counts(eps)} | {params} | "
                f"{ind['mean_wall_s']:.1f} / {ind['mean_sim_s']:.1f} |"
            )
            summary["baselines"].setdefault(label, {})[name] = {
                "graph": row["graph_success"],
                "independent": ind["independent_task_success"],
                "disagreements": len(ind["disagreements"]),
                "place_steps": ind["place_steps_compared"],
                "stopped_at": stop_counts(eps),
                "params": row["params"],
                "train_table": row.get("train_table"),
                "independent_placement_failure_types": ind["independent_placement_failure_types"],
            }
    print("\n## Station negatives (fixed_script, fault on the last slot, seeds 0-2)\n")
    labels = list(runs)
    print("| fault | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    faults = []
    for r in runs.values():
        for f in (r.get("negatives") or {}).get("negatives", {}):
            if f not in faults:
                faults.append(f)
    for f in faults:
        cells = []
        for label in labels:
            n = (runs[label].get("negatives") or {}).get("negatives", {}).get(f)
            if not n:
                cells.append("-")
                continue
            ind = n["independent"]
            cell = (
                f"graph {n['graph_success']} ({stop_counts(n['episodes'])}); independent "
                f"{ind['independent_task_success']} {ind['independent_placement_failure_types'] or ''}; "
                f"disagree {len(ind['disagreements'])}"
            )
            cells.append(cell)
            summary["negatives"].setdefault(label, {})[f] = {
                "graph": n["graph_success"],
                "stopped_at": stop_counts(n["episodes"]),
                "independent": ind["independent_task_success"],
                "independent_placement_failure_types": ind["independent_placement_failure_types"],
                "disagreements": ind["disagreements"],
            }
        print(f"| {f} | " + " | ".join(cells) + " |")
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
