"""Run the three domino baselines on the seeded kinematic fake and write a JSON report.

    python examples/tasks/integration/run_baselines.py --out report.json

The fake world has no physics and its observations are privileged truth. The
report checks interpreter, recovery and search plumbing on paired seeds; it is
not a simulation or hardware result. With Agent 4's Isaac evaluator, pass that
evaluator to ``baselines.compare`` instead of ``StationEnv(domino_world, K)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roboskillhub.tasks import graphs, load_registry
from roboskillhub.tasks.baselines import StationEnv, compare
from roboskillhub.tasks.fake_world import DESCRIBE, domino_world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--graphs-dir", type=Path, help="also write the two minimal G as JSON")
    parser.add_argument("--train", type=int, default=6)
    parser.add_argument("--test", type=int, default=10)
    parser.add_argument("--slots", type=int, default=3, choices=(3, 4, 5))
    args = parser.parse_args()

    K, unavailable = load_registry()
    env = StationEnv(lambda seed: domino_world(seed, slots=args.slots), K)
    report = compare(
        "domino",
        env,
        train_seeds=range(1000, 1000 + args.train),
        test_seeds=range(args.test),
        environment={**DESCRIBE, "slots": args.slots},
        slots=args.slots,
    )
    report["unavailable_modules"] = unavailable
    for row in report["baselines"].values():
        row.pop("episodes_detail")
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.graphs_dir:
        args.graphs_dir.mkdir(parents=True, exist_ok=True)
        for g in (
            graphs.domino_line(recovery=False, attempts=1),
            graphs.domino_line(),
            graphs.lego_press(recovery=False, attempts=1),
            graphs.lego_press(),
        ):
            (args.graphs_dir / f"{g['graph_id']}.json").write_text(json.dumps(g, indent=2) + "\n")
    for name, row in report["baselines"].items():
        print(
            f"{name:18s} graph {row['graph_success']:>6s}  independent {row['independent_task_success']}"
            f"  steps {row['mean_steps']:.1f}  primitives {row['mean_primitive_calls']:.1f}"
            f"  search rollouts {row['search_rollouts']}  params {row['params']}"
        )


if __name__ == "__main__":
    main()
