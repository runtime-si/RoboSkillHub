# SPDX-License-Identifier: Apache-2.0
"""Re-run the current domino evaluator on stored episodes (no simulator).

    PYTHONPATH=src python examples/tasks/assembly_eval/domino_reevaluate.py RUNS_DIR [...]

For every ``record.json`` it rebuilds the scene spec from the tile count and refuses when
its hash differs from the record's, reads ``observations.jsonl`` and each placement's
events, and recomputes every placement and the layout verdict. The neighbour baseline
is the record's ``neighbour_baseline`` when present, else the last fresh observation at
or before the placement's first phase (neighbours are at rest then). The previous
verdicts are kept under ``evaluation_history``; any status change is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboskillhub.tasks.assembly_eval.domino_postcondition import (
    EVAL_VERSION,
    Thresholds,
    evaluate_layout,
    evaluate_placement,
    fresh_samples,
)
from roboskillhub.tasks.assembly_eval.domino_scene import DominoSceneSpec

KEEP = ("events", "plan")


def reevaluate(rec_path: Path) -> list[str]:
    rec = json.loads(rec_path.read_text())
    if rec.get("status") == "rejected" or not rec.get("placements"):
        return []
    spec = DominoSceneSpec(n_tiles=len(rec["goal"]["slots"]))
    if spec.config_hash() != rec["scene_config_sha256"]:
        raise SystemExit(f"{rec_path}: scene hash differs from the record; not re-evaluating")
    th_fields = {k: v for k, v in (rec.get("thresholds") or {}).items() if k != "source"}
    th = Thresholds(**th_fields)
    samples = [json.loads(line) for line in (rec_path.parent / "observations.jsonl").open()]
    fresh, _ = fresh_samples(samples, th.max_obs_age_s)
    changes, new_placements = [], []
    for old in rec["placements"]:
        k, tid, ev = old["slot_index"], old["target_id"], old["events"]
        base = old.get("neighbour_baseline")
        if base is None:
            t0 = ev["phases"][0]["t0"]
            before = [s for s in fresh if s["stamp"] <= t0 + 1e-9]
            src = before[-1]["objects"] if before else {}
            base = {f"tile_{j}": src[f"tile_{j}"] for j in range(k) if f"tile_{j}" in src}
        new = evaluate_placement(
            spec,
            k,
            tid,
            base,
            samples,
            ev,
            th,
            executor_report={"status": old.get("executor_claim")},
            feeder_pose=rec["initial_feeder_poses"][k],
        )
        for key in KEEP:
            new[key] = old[key]
        new["neighbour_baseline"] = base
        if new["status"] != old["status"] or new["failure_types"] != old["failure_types"]:
            changes.append(
                f"{rec_path}: placement {k} {old['status']}{old['failure_types']} -> "
                f"{new['status']}{new['failure_types']}"
            )
        new_placements.append(new)
    last = new_placements[-1]["events"]
    layout = evaluate_layout(
        spec, samples, last["retreat_done_t"], last["eval_t"], new_placements, th
    )
    if layout["status"] != rec["status"]:
        changes.append(f"{rec_path}: episode {rec['status']} -> {layout['status']}")
    history = rec.get("evaluation_history", [])
    history.append(
        {
            "evaluator": rec["placements"][0].get("evaluator"),
            "status": rec["status"],
            "placements": [
                {
                    "status": p["status"],
                    "failure_types": p["failure_types"],
                    "cause_hypotheses": p.get("cause_hypotheses"),
                    "neighbour_first_moved": p.get("neighbour_first_moved"),
                }
                for p in rec["placements"]
            ],
            "replaced_by": EVAL_VERSION,
        }
    )
    failure_types = []
    for p in new_placements:
        for ft in p["failure_types"]:
            if ft not in failure_types:
                failure_types.append(ft)
    rec.update(
        placements=new_placements,
        layout=layout,
        status=layout["status"],
        task_success=layout["status"] == "succeeded",
        placements_succeeded=sum(1 for p in new_placements if p["status"] == "succeeded"),
        failure_types=failure_types,
        evaluation_history=history,
    )
    rec["K_version"]["evaluator"] = EVAL_VERSION
    rec_path.write_text(json.dumps(rec, indent=1))
    return changes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    args = ap.parse_args()
    n, changes = 0, []
    for root in args.roots:
        for p in sorted(root.rglob("record.json")):
            changes += reevaluate(p)
            n += 1
    print(f"re-evaluated {n} records with {EVAL_VERSION}; {len(changes)} verdict changes")
    for c in changes:
        print(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
