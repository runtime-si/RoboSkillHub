# SPDX-License-Identifier: Apache-2.0
"""Run the domino evaluation scene with the scripted privileged baseline.

No robot, no station connection. MuJoCo runs on CPU; Isaac needs an Isaac Sim 5.1 Python
on a GPU host::

    PYTHONPATH=src python examples/tasks/assembly_eval/domino_run.py --backend mujoco \
        --out runs/domino_mj --seeds 0-9 --faults none,missed_grasp
    PYTHONPATH=src /path/to/isaac/venv/bin/python examples/tasks/assembly_eval/domino_run.py \
        --backend isaac --out runs/domino_isaac --seeds 0-4 --faults all

Each episode writes ``<out>/<backend>/<label>/seed_<n>/{record.json,observations.jsonl,
commands.jsonl}``; ``<out>/<backend>/summary.json`` lists numerators and denominators.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from roboskillhub.tasks.assembly_eval.domino_baseline import COMMAND_FAULTS, OBSERVER_FAULTS
from roboskillhub.tasks.assembly_eval.domino_runner import run_episode
from roboskillhub.tasks.assembly_eval.domino_scene import DominoSceneSpec


def seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", choices=("mujoco", "isaac"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument(
        "--faults", default="none", help="comma list (name or name:offset_m), 'all', or 'none'"
    )
    ap.add_argument("--n-tiles", type=int, default=4)
    ap.add_argument("--scene", type=Path, help="scene spec JSON (default: built-in v1)")
    ap.add_argument("--render-dir", type=Path, help="isaac only: save a few RGB frames per episode")
    args = ap.parse_args()
    spec = (
        DominoSceneSpec.from_dict(json.loads(args.scene.read_text()))
        if args.scene
        else DominoSceneSpec(n_tiles=args.n_tiles)
    )
    faults = (
        ["none", *COMMAND_FAULTS, *OBSERVER_FAULTS]
        if args.faults == "all"
        else args.faults.split(",")
    )
    if args.backend == "mujoco":
        from roboskillhub.tasks.assembly_eval.domino_mujoco import MujocoBackend

        backend = MujocoBackend()
    else:
        from roboskillhub.tasks.assembly_eval.domino_isaac import IsaacBackend

        backend = IsaacBackend(headless=True, render_dir=args.render_dir)
    root = args.out / backend.name
    rows = []
    try:
        for fault in faults:
            for seed in seeds(args.seeds):
                label = fault.replace(":", "@")
                d = root / label / f"seed_{seed}"
                if (d / "record.json").exists():
                    rec = json.loads((d / "record.json").read_text())
                else:
                    try:
                        name, _, off = fault.partition(":")
                        rec = run_episode(
                            backend,
                            spec,
                            seed,
                            d,
                            fault=None if name == "none" else name,
                            fault_offset=float(off) if off else None,
                            label=label,
                        )
                    except Exception:
                        d.mkdir(parents=True, exist_ok=True)
                        (d / "error.txt").write_text(traceback.format_exc())
                        rows.append({"label": label, "seed": seed, "status": "error"})
                        print(f"EPISODE {label} seed={seed} ERROR", flush=True)
                        continue
                pl = rec.get("placements", [])
                rows.append(
                    {
                        "label": label,
                        "seed": seed,
                        "status": rec["status"],
                        "task_success": rec.get("task_success"),
                        "placements": f"{rec.get('placements_succeeded', 0)}/{rec.get('placements_attempted', 0)}",
                        "failure_types": rec.get("failure_types"),
                        "last_placement_unverified": pl[-1].get("unverified_reasons")
                        if pl
                        else None,
                        "executor_claimed_all_finished": rec.get("executor_claimed_all_finished"),
                        "sim_time_s": rec.get("sim_time_s"),
                        "wall_s": rec.get("wall_s"),
                    }
                )
                print(
                    f"EPISODE {label} seed={seed} status={rec['status']} "
                    f"placements={rows[-1]['placements']} failures={rec.get('failure_types')}",
                    flush=True,
                )
    finally:
        if hasattr(backend, "close"):
            backend.close()
    summary = {
        "backend": backend.describe(),
        "scene_config_sha256": spec.config_hash(),
        "episodes": rows,
        "by_label": {},
    }
    for r in rows:
        b = summary["by_label"].setdefault(
            r["label"],
            {"n": 0, "succeeded": 0, "failed": 0, "unverified": 0, "rejected": 0, "error": 0},
        )
        b["n"] += 1
        b[r["status"]] = b.get(r["status"], 0) + 1
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["by_label"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
