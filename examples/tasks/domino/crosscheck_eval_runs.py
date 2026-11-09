"""Judge Agent 04's assembly_eval domino episodes with verify_stable_placement and compare
with that runner's own independent post-condition. Reads files only; no simulator.

    PYTHONPATH=src python examples/tasks/domino/crosscheck_eval_runs.py \\
        ~/lego-domino-kit/output/lego-domino/agent-04/domino/runs out.json [--match-eval]

``--match-eval`` tightens this module's thresholds toward the evaluator's (only
tightening is allowed): 1.0 s / 5 frames window, 5 mm position, 5 deg tilt, 1 mm drift,
2 mm / 2 deg neighbours, 0.2 s observation age.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

from roboskillhub.tasks.domino import verify_stable_placement
from roboskillhub.tasks.domino.adapters import verify_request_from_eval_record

MATCH_EVAL = {
    "settle_window_s": 1.0,
    "min_window_samples": 5,
    "position_tolerance_m": 0.005,
    "tilt_tolerance_rad": math.radians(5.0),
    "drift_tolerance_m": 0.001,
    "drift_tilt_tolerance_rad": math.radians(1.0),
    "neighbor_displacement_tolerance_m": 0.002,
    "neighbor_tilt_tolerance_rad": math.radians(2.0),
    "max_observation_age_s": 0.2,
}


def main(root: str, out_path: str, match_eval: bool = False) -> dict:
    rows = []
    for rec_path in sorted(Path(root).expanduser().rglob("record.json")):
        record = json.loads(rec_path.read_text())
        obs_path = rec_path.with_name(record["trajectory_ref"]["observations"])
        samples = [json.loads(line) for line in obs_path.read_text().splitlines() if line.strip()]
        for i, p in enumerate(record["placements"]):
            req = verify_request_from_eval_record(
                record, samples, i, thresholds_override=MATCH_EVAL if match_eval else None
            )
            v = verify_stable_placement(req)
            rows.append(
                {
                    "run": str(rec_path.parent.relative_to(Path(root).expanduser())),
                    "fault": record["fault"]["name"],
                    "placement": i,
                    "target": p["target_id"],
                    "eval_status": p["status"],
                    "eval_failures": p["failure_types"],
                    "eval_unverified": p.get("unverified_reasons", []),
                    "ours_status": v["status"],
                    "ours_failure_type": v["failure_type"],
                    "ours_detail": (v.get("failure") or {}).get("detail"),
                    "agree": v["status"] == p["status"],
                }
            )
    summary = {
        "runs_root": root,
        "match_eval_thresholds": match_eval,
        "placements": len(rows),
        "agree": sum(r["agree"] for r in rows),
        "confusion": dict(Counter(f"{r['eval_status']}->{r['ours_status']}" for r in rows)),
        "by_fault": {
            f: dict(
                Counter(
                    f"{r['eval_status']}->{r['ours_status']}" for r in rows if str(r["fault"]) == f
                )
            )
            for f in sorted({str(r["fault"]) for r in rows}, key=str)
            for _ in [0]
        },
        "failure_pairs": dict(
            Counter(
                f"{'+'.join(r['eval_failures'] + r['eval_unverified']) or '-'} -> {r['ours_failure_type']}"
                for r in rows
                if r["eval_status"] != "succeeded" or r["ours_status"] != "succeeded"
            )
        ),
        "disagreements": [r for r in rows if not r["agree"]],
    }
    Path(out_path).write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    s = main(args[0], args[1], "--match-eval" in sys.argv)
    print(
        json.dumps(
            {k: s[k] for k in ("placements", "agree", "confusion", "failure_pairs")}, indent=1
        )
    )
